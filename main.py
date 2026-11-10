from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from youtube_chatbot import (
    load_video,
    ask_question,
    generate_summary,
    GEMINI_MODEL,
    gemini_client,
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
        "gemini_model": GEMINI_MODEL,
    }


# =========================================
# 5. LOAD YOUTUBE VIDEO (link only)
# =========================================

@app.post("/api/load-video")
def load_youtube_video(request: VideoRequest):
    try:
        data = load_video(request.video_url)

        video = data["video"]

        return {
            "status": "success",
            "message": "Video loaded successfully",
            "video": video,
            "video_id": video["video_id"],
            "source": data["source"],
            "chunks_indexed": len(data["chunks"]),
            "summary": None,
            "summary_error": None,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        print("Video loading error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Video loading failed: {str(exc)}",
        )


# =========================================
# 7. SUMMARY
# =========================================

@app.post("/api/summary")
def video_summary(request: SummaryRequest):
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
def chat_with_video(request: ChatRequest):
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