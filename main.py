from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from youtube_chatbot import load_video, ask_question, generate_summary


app = FastAPI(
    title="YouTube AI Chatbot",
    description="Chat with YouTube videos using Hybrid RAG (Transcript + Local OCR)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Language = Literal["English", "Hindi", "Hinglish"]


class VideoRequest(BaseModel):
    video_url: str = Field(min_length=10, description="YouTube video URL")
    language: Language = "English"


class ChatRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)
    question: str = Field(min_length=1, max_length=2000)
    language: Language = "English"


class SummaryRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)
    language: Language = "English"


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "YouTube AI Chatbot Hybrid RAG API is running!",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "YouTube AI Chatbot Hybrid RAG",
    }


@app.post("/api/load-video")
def load_youtube_video(request: VideoRequest):
    try:
        # load_video extracts transcript and attempts local frame OCR.
        data = load_video(request.video_url)
        video_id = data["video"]["video_id"]

        summary = None
        summary_error = None

        try:
            summary = generate_summary(video_id, request.language)
        except Exception as exc:
            summary_error = str(exc)
            print("Summary generation failed:", summary_error)

        return {
            "status": "success",
            "message": "Video loaded successfully",
            "video": data["video"],
            "video_id": video_id,
            "transcript": data["segments"],
            "transcript_language": data["transcript_language"],
            "transcript_language_code": data["transcript_language_code"],
            "transcript_is_generated": data["transcript_is_generated"],
            "ocr_frames_indexed": len(data.get("ocr_segments", [])),
            "ocr_error": data.get("ocr_error"),
            "summary": summary,
            "summary_error": summary_error,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print("Video loading error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Video loading failed: {str(exc)}",
        )


@app.post("/api/summary")
def video_summary(request: SummaryRequest):
    try:
        summary = generate_summary(request.video_id, request.language)
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


@app.post("/api/chat")
def chat_with_video(request: ChatRequest):
    try:
        answer = ask_question(
            video_id=request.video_id,
            question=request.question,
            language=request.language,
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
        error_message = str(exc)
        print("Chat error:", repr(exc))

        if "rate_limit" in error_message.lower():
            raise HTTPException(
                status_code=429,
                detail="Groq rate limit reached. Please wait and retry.",
            )

        raise HTTPException(
            status_code=502,
            detail="AI response generation failed. Please try again.",
        )
