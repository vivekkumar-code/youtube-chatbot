# YouTube AI Chatbot (RAG)

Paste a YouTube link and chat with the video. The chatbot answers your doubts using the
video's own content and tells you **when** a topic was discussed. Click the timestamp
and the player jumps to that moment.

**Live demo:** https://myyoutube-chatbot.netlify.app
*(First load can take up to a minute - the backend runs on a free server that sleeps.)*

## Features

- **Link only** - no transcript copy-paste
- **Chat with the video** (RAG) with follow-up questions that remember the conversation
- **Clickable timestamps** `▶ 05:32` in answers and in the summary
- **AI summary** with main topics, key points and important terms
- **English, Hindi and Hinglish** answers
- **Works for long videos** - read in 20-minute parts to stay inside token limits
- **Background loading** with live status, so long videos never time out
- **Usage limits** per visitor to protect free API quotas

## How it works

**Getting the video text (automatic, in this order):**

1. YouTube captions (free)
2. Supadata API (works when YouTube blocks the server)
3. Gemini watches the video directly from its link

Then: text is split into chunks with timestamps, embedded with Gemini embeddings, and the
most relevant chunks are sent to a Groq LLM together with the recent chat history.

**Why a 3-step pipeline?** YouTube blocks most cloud servers, so captions often fail on
hosting platforms.

**Why API-based embeddings?** The first version used a local embedding model, FAISS, OpenCV
and OCR. They did not fit in a 512 MB free instance, so they were replaced with
lightweight API-based components.

## Tech stack

| Part | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, YouTube IFrame Player API |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| LLM | Groq (`gpt-oss-20b`) via LangChain |
| Video understanding + embeddings | Google Gemini API |
| Transcripts | `youtube-transcript-api`, Supadata API |
| Retrieval | NumPy cosine similarity (keyword search as backup) |
| Hosting | Netlify (frontend), Render (backend), GitHub auto-deploy |

## Project structure

```
.
├── main.py               # FastAPI app, routes, usage limits
├── youtube_chatbot.py    # Transcript pipeline, chunking, embeddings, chat, summary
├── requirements.txt
├── .env.example
└── frontend/
    ├── index.html
    ├── script.js
    └── style.css
```

## Run locally

```bash
git clone https://github.com/<vivekkumar-code>/youtube-chatbot.git
cd youtube-chatbot

python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your keys
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` with VS Code **Live Server** (`http://127.0.0.1:5500`).
`script.js` automatically uses `http://127.0.0.1:8000` when running locally.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Chat answers and summaries |
| `GEMINI_API_KEY` | Recommended | Video reading (fallback) and embeddings |
| `SUPADATA_API_KEY` | Optional | Captions when YouTube blocks the server |
| `YOUTUBE_API_KEY` | Optional | Title, channel and video length |
| `GEMINI_MODEL` | Optional | Gemini model name (default set in code) |
| `GEMINI_FALLBACK_MODEL` | Optional | Second Gemini model if the first keeps failing |
| `LIMIT_VIDEOS_PER_DAY` | Optional | New videos per visitor per day (default 5) |
| `LIMIT_CHATS_PER_HOUR` | Optional | Questions per visitor per hour (default 30) |
| `LIMIT_SUMMARIES_PER_HOUR` | Optional | Summaries per visitor per hour (default 10) |
| `GLOBAL_VIDEOS_PER_DAY` | Optional | New videos per day for the whole site (default 30) |

Set `0` on any limit to switch it off. **Never commit your `.env` file.**

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/load-video` | Start loading a video (returns immediately) |
| GET | `/api/load-status/{video_id}` | Check loading progress |
| POST | `/api/chat` | Ask a question about the video |
| POST | `/api/summary` | Generate the summary |
| GET | `/health` | Health check and configuration status |
| GET | `/api/gemini-models` | Models available for your Gemini key |

## Deployment

**Backend (Render):** Build command `pip install -r requirements.txt`, start command
`uvicorn main:app --host 0.0.0.0 --port $PORT`, then add the environment variables above.

**Frontend (Netlify):** Connect the repo and set the publish directory to `frontend`.
Update the Render URL in `frontend/script.js`, and add your Netlify URL to the CORS list in `main.py`.

## Limitations

- Runs on free tiers, so the first request after idle can be slow and quotas are limited.
- Usage limits are kept in memory and reset when the server restarts.
- Videos without captions need Gemini, and the video must be public.
- Text shown only on screen (slides, code) is available only when Gemini reads the video.
- Gemini timestamps can be a few seconds off; caption-based timestamps are exact.

## Future improvements

- Persistent storage for limits and saved videos
- Streaming answers
- Export notes as PDF
- User accounts and chat history