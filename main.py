
import os
import threading
import time
from collections import defaultdict, deque
from typing import List, Literal, Optional
 
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
from youtube_chatbot import (
    start_load_job,
    needs_processing,
    extract_video_id,
    get_job_status,
    list_gemini_models,
    ask_question,
    generate_summary,
    GEMINI_MODEL,
    gemini_client,
    SUPADATA_API_KEY,
)
 
 
# =========================================
# 1. FASTAPI APPLICATION
# =========================================
 
app = FastAPI(
    title="YouTube AI Chatbot",
    description="Paste a YouTube link and chat with the video (RAG).",
    version="4.0.0",
)
 
 
# =========================================
# 2. CORS CONFIGURATION
# =========================================
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "https://myyoutube-chatbot.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# =========================================
# 2b. USAGE LIMITS (protect the free API quotas)
# =========================================
# Soft, in-memory limits per visitor (IP). They reset when the server restarts.
# Change them from Render > Environment without touching the code.
# A value of 0 turns that limit off.
 
LIMIT_VIDEOS_PER_DAY = int(os.getenv("LIMIT_VIDEOS_PER_DAY", "5"))
LIMIT_CHATS_PER_HOUR = int(os.getenv("LIMIT_CHATS_PER_HOUR", "30"))
LIMIT_SUMMARIES_PER_HOUR = int(os.getenv("LIMIT_SUMMARIES_PER_HOUR", "10"))
GLOBAL_VIDEOS_PER_DAY = int(os.getenv("GLOBAL_VIDEOS_PER_DAY", "30"))
 
_HITS = defaultdict(deque)
_HITS_LOCK = threading.Lock()
 
 
def _client_ip(http_request: Request) -> str:
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"
 
 
def _check_limit(key: str, limit: int, window_seconds: int, message: str):
    if limit <= 0:
        return
 
    now = time.time()
 
    with _HITS_LOCK:
        hits = _HITS[key]
 
        while hits and now - hits[0] > window_seconds:
            hits.popleft()
 
        if len(hits) >= limit:
            retry_after = int(window_seconds - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=message,
                headers={"Retry-After": str(retry_after)},
            )
 
        hits.append(now)
 
        # Keep memory small: forget visitors with no recent activity.
        if len(_HITS) > 5000:
            for stale in [k for k, v in _HITS.items() if not v]:
                del _HITS[stale]
 
 
def _enforce_video_limits(http_request: Request):
    _check_limit(
        "videos:global", GLOBAL_VIDEOS_PER_DAY, 86400,
        "Aaj ke liye is site ki free limit poori ho gayi hai. Kal dobara try karo.",
    )
    _check_limit(
        f"videos:{_client_ip(http_request)}", LIMIT_VIDEOS_PER_DAY, 86400,
        f"Ek din me {LIMIT_VIDEOS_PER_DAY} nayi videos hi load ho sakti hain. "
        "Kal dobara try karo (pehle load ki hui video dobara khol sakte ho).",
    )
 
 
# =========================================
# 3. REQUEST MODELS
# =========================================
 
Language = Literal["English", "Hindi", "Hinglish"]
 
 
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
 
 
class VideoRequest(BaseModel):
    video_url: str = Field(min_length=10)
    language: Language = "English"
 
 
class ChatRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)
    question: str = Field(min_length=1, max_length=2000)
    language: Language = "English"
    video_url: Optional[str] = None
    history: List[HistoryItem] = Field(default_factory=list, max_length=12)
 
 
class SummaryRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)
    language: Language = "English"
    video_url: Optional[str] = None
 
 
# =========================================
# 4. HOME + HEALTH
# =========================================
 
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "YouTube AI Chatbot RAG API is running!",
    }
 
 
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "YouTube AI Chatbot RAG",
        "gemini_configured": gemini_client is not None,
        "supadata_configured": bool(SUPADATA_API_KEY),
        "limits": {
            "videos_per_day": LIMIT_VIDEOS_PER_DAY,
            "chats_per_hour": LIMIT_CHATS_PER_HOUR,
            "summaries_per_hour": LIMIT_SUMMARIES_PER_HOUR,
            "global_videos_per_day": GLOBAL_VIDEOS_PER_DAY,
        },
        "gemini_model": GEMINI_MODEL,
    }
 
 
@app.get("/api/gemini-models")
def gemini_models():
    """Shows which Gemini model names work with your key (for GEMINI_MODEL)."""
    try:
        return {
            "configured_model": GEMINI_MODEL,
            "available_models": list_gemini_models(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not list models: {exc}")
 
 
# =========================================
# 5. LOAD YOUTUBE VIDEO (background job + status polling)
# =========================================
 
def _status_payload(video_id: str) -> dict:
    status = get_job_status(video_id)
 
    payload = {
        "status": "success",
        "video_id": video_id,
        "state": status["state"],
        "stage": status.get("stage"),
        "error": status.get("error"),
        "summary": None,
        "summary_error": None,
    }
 
    if status["state"] == "done":
        data = status["data"]
        payload.update({
            "video": data["video"],
            "source": data["source"],
            "chunks_indexed": len(data["chunks"]),
        })
 
    return payload
 
 
@app.post("/api/load-video")
def load_youtube_video(request: VideoRequest, http_request: Request):
    try:
        video_id = extract_video_id(request.video_url)  # ValueError -> 400
 
        # Only a really new video spends quota; cached ones are free.
        if needs_processing(video_id):
            _enforce_video_limits(http_request)
 
        job = start_load_job(request.video_url)
        return _status_payload(job["video_id"])
 
    except HTTPException:
        raise
 
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
 
    except Exception as exc:
        print("Video loading error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Video loading failed: {str(exc)}",
        )
 
 
@app.get("/api/load-status/{video_id}")
def load_status(video_id: str = Path(min_length=11, max_length=11)):
    return _status_payload(video_id)
 
 
# =========================================
# 7. SUMMARY
# =========================================
 
@app.post("/api/summary")
def video_summary(request: SummaryRequest, http_request: Request):
    _check_limit(
        f"summary:{_client_ip(http_request)}", LIMIT_SUMMARIES_PER_HOUR, 3600,
        f"Ek ghante me {LIMIT_SUMMARIES_PER_HOUR} summary hi ban sakti hain. Thodi der baad try karo.",
    )
 
    try:
        summary = generate_summary(
            request.video_id,
            request.language,
            video_url=request.video_url,
        )
 
        return {
            "status": "success",
            "summary": summary,
            "language": request.language,
        }
 
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
 
    except Exception as exc:
        print("Summary error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {str(exc)}",
        )
 
 
# =========================================
# 8. CHAT WITH VIDEO
# =========================================
 
@app.post("/api/chat")
def chat_with_video(request: ChatRequest, http_request: Request):
    _check_limit(
        f"chat:{_client_ip(http_request)}", LIMIT_CHATS_PER_HOUR, 3600,
        f"Ek ghante me {LIMIT_CHATS_PER_HOUR} sawal hi pooche ja sakte hain. Thodi der baad try karo.",
    )
 
    try:
        answer = ask_question(
            video_id=request.video_id,
            question=request.question,
            language=request.language,
            video_url=request.video_url,
            history=[
                {"role": item.role, "content": item.content}
                for item in request.history
            ],
        )
 
        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "language": request.language,
        }
 
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
 
    except Exception as exc:
        print("Chat error:", repr(exc))
 
        if "rate_limit" in str(exc).lower():
            raise HTTPException(
                status_code=429,
                detail="Groq rate limit reached. Please wait and retry.",
            )
 
        raise HTTPException(
            status_code=502,
            detail="AI response generation failed. Please try again.",
        )