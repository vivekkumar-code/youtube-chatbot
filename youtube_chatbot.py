"""
YouTube RAG chatbot (no manual transcript needed).

How the video text is obtained (automatic, in this order):
  1. YouTube captions via youtube-transcript-api (fast, free).
  2. If captions are blocked/unavailable (common on Render), the YouTube link
     is sent straight to the Gemini API, which watches/listens to the video on
     Google's servers and returns detailed notes of everything said and shown.

RAG: the text is split into chunks, embedded with the Gemini embedding API
(no local model, so it fits in Render's 512 MB RAM), and the best chunks are
sent to Groq's LLM together with recent chat history for follow-up doubts.
"""

import os
import re
import json
import time
import gc
from collections import OrderedDict
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

import numpy as np
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from google import genai
from google.genai import types

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # optional (only for title/channel)

# Model names are configurable because Google renames/retires models.
# If you get a "model not found" error, set GEMINI_MODEL / EMBED_MODEL
# in Render's Environment variables to a current model name.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# Used automatically when the main model is overloaded (503) or unavailable.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in environment variables.")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Only YouTube captions / manual "
          "transcript will work (no Gemini video fallback, no embeddings).")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SUPPORTED_LANGUAGES = ["English", "Hindi", "Hinglish"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "video_data")
os.makedirs(DATA_DIR, exist_ok=True)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6

VIDEO_CACHE = OrderedDict()
MAX_CACHED_VIDEOS = 3

print("Initializing Groq LLM...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=3000,
    api_key=GROQ_API_KEY,
)

summary_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1000,
    api_key=GROQ_API_KEY,
)

print("Groq LLM initialized.")


def invoke_summary_with_retry(chain, payload, max_retries=2):
    """Retry summary calls after transient Groq rate-limit/network errors."""
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            message = str(exc).lower()
            is_rate_limit = (
                "429" in message
                or "rate_limit" in message
                or "rate limit" in message
            )
            is_connection_error = (
                "apiconnectionerror" in message
                or "connection error" in message
                or "connection reset" in message
            )

            if attempt >= max_retries or not (is_rate_limit or is_connection_error):
                raise

            wait_seconds = 65 if is_rate_limit else 5 * (attempt + 1)
            print(f"Groq call failed ({type(exc).__name__}). Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


# =========================================================
# HELPERS
# =========================================================

def extract_video_id(url: str) -> str:
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or "").lower()
    video_id = None

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]

        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            if len(parts) > 1:
                video_id = parts[1]

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube URL.")

    return video_id


def validate_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Choose English, Hindi or Hinglish.")
    return language


def format_timestamp(seconds) -> str:
    seconds = max(0, int(float(seconds or 0)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_iso_duration(value: str) -> str:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not match:
        return value or ""
    hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _oembed_info(video_id: str):
    """Title/channel without any API key (works from cloud servers too)."""
    try:
        url = (
            "https://www.youtube.com/oembed?url="
            f"https://www.youtube.com/watch?v={video_id}&format=json"
        )
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("title"), data.get("author_name")
    except Exception:
        return None, None


def get_video_info(video_id: str) -> dict:
    thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    info = {
        "video_id": video_id,
        "title": "YouTube Video",
        "channel": "Unknown",
        "description": "",
        "thumbnail": thumbnail,
        "duration": "",
        "default_audio_language": None,
    }

    if YOUTUBE_API_KEY:
        try:
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
            response = youtube.videos().list(
                part="snippet,contentDetails", id=video_id
            ).execute()

            items = response.get("items", [])
            if items:
                snippet = items[0]["snippet"]
                info.update({
                    "title": snippet["title"],
                    "channel": snippet["channelTitle"],
                    "description": snippet.get("description", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", thumbnail),
                    "duration": format_iso_duration(items[0]["contentDetails"]["duration"]),
                    "default_audio_language": (
                        snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
                    ),
                })
                return info
        except Exception as exc:
            print("YouTube Data API failed, using oEmbed:", repr(exc))

    title, channel = _oembed_info(video_id)
    if title:
        info["title"] = title
    if channel:
        info["channel"] = channel

    return info


# =========================================================
# STEP 1: YOUTUBE CAPTIONS (may be blocked on cloud servers)
# =========================================================

def fetch_captions(video_id: str, preferred_language=None) -> dict:
    api = YouTubeTranscriptApi()
    transcript_list = list(api.list(video_id))

    if not transcript_list:
        raise ValueError("No captions available for this video.")

    selected = None

    if preferred_language:
        base = preferred_language.lower().split("-")[0]
        matching = [
            item for item in transcript_list
            if item.language_code.lower().split("-")[0] == base
        ]
        if matching:
            manual = [item for item in matching if not item.is_generated]
            selected = manual[0] if manual else matching[0]

    if selected is None:
        manual = [item for item in transcript_list if not item.is_generated]
        selected = manual[0] if manual else transcript_list[0]

    fetched = selected.fetch()

    lines = []
    block_start = None
    block_text = []

    for item in fetched:
        piece = (item.text or "").strip().replace("\n", " ")
        if not piece:
            continue

        start = float(item.start)

        if block_start is None:
            block_start = start
        elif start - block_start >= 25:
            lines.append(f"[{format_timestamp(block_start)}] {' '.join(block_text)}")
            block_start = start
            block_text = []

        block_text.append(piece)

    if block_text:
        lines.append(f"[{format_timestamp(block_start)}] {' '.join(block_text)}")

    text = "\n".join(lines)

    if not text.strip():
        raise ValueError("Transcript is empty.")

    return {"text": text, "language": selected.language}


# =========================================================
# STEP 2: GEMINI WATCHES THE YOUTUBE LINK (no captions needed)
# =========================================================

VIDEO_NOTES_PROMPT = """
You are preparing complete study material from this YouTube video so that a
chatbot can later clear a student's doubts about it.

Watch and listen to the ENTIRE video, then write detailed, faithful notes.

RULES:
- Cover EVERY topic, definition, explanation, example, analogy, step, formula,
  code snippet, number, comparison and conclusion, in the order presented.
- Also include important text shown on screen or on slides.
- Write full explanatory sentences (not just keywords), so each part makes
  sense on its own.
- Keep the video's own language (Hindi in Devanagari, Hinglish in Roman
  script, or English). Keep technical terms as they are.
- Organize with '## [MM:SS] Topic name' headings.
- TIMESTAMPS ARE REQUIRED: start every heading and every paragraph with the
  real video playback time where that part is discussed, in the exact form
  [MM:SS] (use [H:MM:SS] after the first hour), for example [05:32]. Put each
  paragraph on its own line. Use only times that really occur in the video.
- Do not add outside information. Do not invent anything.
- Skip greetings, sponsor messages and filler.
"""


def gemini_video_to_text(video_id: str) -> str:
    if gemini_client is None:
        raise ValueError(
            "YouTube ne captions nahi diye aur GEMINI_API_KEY set nahi hai."
        )

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    config_kwargs = {"temperature": 0.2}

    # Lower frame resolution = far fewer tokens for long videos.
    media_resolution = getattr(types, "MediaResolution", None)
    if media_resolution is not None and hasattr(media_resolution, "MEDIA_RESOLUTION_LOW"):
        config_kwargs["media_resolution"] = media_resolution.MEDIA_RESOLUTION_LOW

    contents = types.Content(parts=[
        types.Part(file_data=types.FileData(file_uri=video_url)),
        types.Part(text=VIDEO_NOTES_PROMPT),
    ])

    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    response = None
    last_exc = None

    for model_name in models:
        # Up to 3 tries per model; wait a little when Google is overloaded.
        for attempt in range(3):
            try:
                print(f"Asking Gemini ({model_name}) to read the video "
                      f"(try {attempt + 1}/3)...")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                break
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                print("Gemini video error:", repr(exc))

                overloaded = (
                    "503" in message
                    or "unavailable" in message
                    or "overloaded" in message
                    or "high demand" in message
                    or "500" in message
                    or "internal" in message
                    or "deadline" in message
                )

                if overloaded and attempt < 2:
                    time.sleep(8 * (attempt + 1))
                    continue

                # Not retryable (or retries used up): try the next model.
                break

        if response is not None:
            break

    if response is None:
        message = str(last_exc).lower()

        if "getaddrinfo" in message or "11001" in message or "connecterror" in message \
                or "connection" in message or "timed out" in message:
            raise ValueError(
                "Internet/DNS problem: Gemini server tak request nahi pahunch paayi. "
                "Internet check karo (DNS 8.8.8.8, VPN band) ya Render par chalao."
            )
        if "503" in message or "unavailable" in message or "high demand" in message:
            raise ValueError(
                "Gemini par abhi bahut load hai (Google ki taraf se temporary). "
                "1-2 minute baad dobara Load Video dabao."
            )
        if "429" in message or "quota" in message or "resource_exhausted" in message:
            raise ValueError(
                "Gemini ki free limit abhi poori ho gayi hai. Thodi der baad try karo."
            )
        if "404" in message and "model" in message:
            raise ValueError(
                "Gemini model naam galat/purana hai. Render me GEMINI_MODEL "
                "environment variable me current model naam daalo."
            )
        if (
            "private" in message
            or "permission" in message
            or "not found" in message
            or "invalid_argument" in message
            or "400" in message
        ):
            raise ValueError(
                "Yeh video Gemini process nahi kar paya. Video public hona "
                "chahiye (private/unlisted/age-restricted nahi)."
            )
        raise ValueError(f"Gemini video processing failed: {last_exc}")

    text = (getattr(response, "text", None) or "").strip()

    if len(text) < 200:
        raise ValueError("Gemini ko is video se kaafi content nahi mila.")

    return text


# =========================================================
# CHUNKING + EMBEDDINGS + SEARCH (no local ML model)
# =========================================================

_MARKER = re.compile(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]")
_STARTS_WITH_MARKER = re.compile(r"\s*(?:#+\s*)?\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]")


def _last_marker(text: str):
    found = _MARKER.findall(text)
    return found[-1] if found else None


def split_into_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    """Split text into chunks. Every chunk begins with a [MM:SS] time marker
    (carried over from the previous chunk when needed) so answers can say
    when a topic was discussed."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)

    units = []
    for part in parts:
        part = part.strip()
        while len(part) > size:
            cut = part.rfind(" ", 0, size)
            if cut < size // 2:
                cut = size
            units.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            units.append(part)

    chunks = []
    current = ""
    carried = None

    for unit in units:
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            carried = _last_marker(current) or carried

            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]

            current = (tail + " " + unit).strip()

            if carried and not _STARTS_WITH_MARKER.match(current):
                current = f"{carried} {current}"
        else:
            current = (current + " " + unit).strip()

    if current:
        chunks.append(current)

    return chunks


def embed_texts(texts: list, task_type: str) -> np.ndarray:
    if gemini_client is None:
        raise RuntimeError("Gemini client not configured.")

    vectors = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        for attempt in range(3):
            try:
                response = gemini_client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend([item.values for item in response.embeddings])
                break
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "429" in message or "quota" in message or "resource_exhausted" in message
                if attempt == 2 or not is_rate_limit:
                    raise
                time.sleep(20 * (attempt + 1))

    array = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _tokenize(text: str) -> set:
    return {t for t in re.split(r"[\s,.;:!?()\[\]\"'।|/\\-]+", text.lower()) if len(t) > 1}


def _lexical_scores(query: str, chunks: list) -> np.ndarray:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return np.zeros(len(chunks), dtype="float32")

    return np.array(
        [len(query_tokens & _tokenize(chunk)) / len(query_tokens) for chunk in chunks],
        dtype="float32",
    )


def search_chunks(video_data: dict, query: str, k=TOP_K) -> list:
    chunks = video_data["chunks"]
    embeddings = video_data.get("embeddings")

    scores = None

    if embeddings is not None:
        try:
            query_vector = embed_texts([query], "RETRIEVAL_QUERY")[0]
            scores = embeddings @ query_vector
        except Exception as exc:
            print("Embedding search failed, using keyword search:", repr(exc))

    if scores is None:
        scores = _lexical_scores(query, chunks)

    top = np.argsort(-scores)[:k]

    # Keep the video's natural order so explanations read in sequence.
    return [chunks[i] for i in sorted(top.tolist())]


# =========================================================
# STORAGE (memory cache + disk)
# =========================================================

def _remember(video_id: str, data: dict):
    VIDEO_CACHE[video_id] = data
    VIDEO_CACHE.move_to_end(video_id)

    while len(VIDEO_CACHE) > MAX_CACHED_VIDEOS:
        VIDEO_CACHE.popitem(last=False)
        gc.collect()


def _paths(video_id: str):
    return (
        os.path.join(DATA_DIR, f"{video_id}.json"),
        os.path.join(DATA_DIR, f"{video_id}.npy"),
    )


def _save_to_disk(video_id: str, data: dict):
    json_path, npy_path = _paths(video_id)
    try:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump({
                "version": 2,
                "video": data["video"],
                "source": data["source"],
                "language": data["language"],
                "notes": data["notes"],
                "chunks": data["chunks"],
            }, file, ensure_ascii=False)

        if data.get("embeddings") is not None:
            np.save(npy_path, data["embeddings"])
        elif os.path.isfile(npy_path):
            os.remove(npy_path)
    except Exception as exc:
        print("Could not save video data to disk:", repr(exc))


def _load_from_disk(video_id: str):
    json_path, npy_path = _paths(video_id)

    if not os.path.isfile(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            saved = json.load(file)

        if saved.get("version") != 2:
            return None

        embeddings = np.load(npy_path) if os.path.isfile(npy_path) else None

        if embeddings is not None and len(embeddings) != len(saved["chunks"]):
            embeddings = None

        saved["embeddings"] = embeddings
        return saved
    except Exception as exc:
        print("Could not read saved video data:", repr(exc))
        return None


def build_index(video_id: str, info: dict, text: str, source: str, language: str) -> dict:
    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("Is video se koi text nahi nikla.")

    embeddings = None

    if gemini_client is not None:
        try:
            embeddings = embed_texts(chunks, "RETRIEVAL_DOCUMENT")
        except Exception as exc:
            print("Embedding failed, keyword search will be used:", repr(exc))

    data = {
        "video": info,
        "source": source,
        "language": language,
        "notes": text,
        "chunks": chunks,
        "embeddings": embeddings,
    }

    _save_to_disk(video_id, data)
    _remember(video_id, data)

    print(f"Indexed {len(chunks)} chunks (source: {source}, embeddings: {embeddings is not None}).")

    return data


# =========================================================
# LOAD VIDEO (only the link is needed)
# =========================================================

def load_video(video_url: str) -> dict:
    video_id = extract_video_id(video_url)

    if video_id in VIDEO_CACHE:
        VIDEO_CACHE.move_to_end(video_id)
        return VIDEO_CACHE[video_id]

    saved = _load_from_disk(video_id)
    if saved:
        print("Loaded video from disk.")
        _remember(video_id, saved)
        return saved

    info = get_video_info(video_id)

    text = None
    source = None
    language = "Unknown"

    # Try 1: YouTube captions.
    try:
        print("Trying YouTube captions...")
        captions = fetch_captions(video_id, info.get("default_audio_language"))
        if len(captions["text"]) >= 200:
            text = captions["text"]
            language = captions["language"]
            source = "captions"
    except Exception as exc:
        print(f"Captions unavailable: {exc}")

    # Try 2: Gemini reads the video directly from the YouTube link.
    if text is None:
        text = gemini_video_to_text(video_id)
        source = "gemini"

    return build_index(video_id, info, text, source, language)


def ensure_loaded(video_id: str, video_url=None) -> dict:
    """Return loaded video data; transparently reload after a server restart."""
    if video_id in VIDEO_CACHE:
        VIDEO_CACHE.move_to_end(video_id)
        return VIDEO_CACHE[video_id]

    saved = _load_from_disk(video_id)
    if saved:
        _remember(video_id, saved)
        return saved

    if video_url and extract_video_id(video_url) == video_id:
        return load_video(video_url)

    raise ValueError("Video is not loaded. Please load it again.")


# =========================================================
# CHAT WITH VIDEO (RAG)
# =========================================================

CHAT_PROMPT = ChatPromptTemplate.from_template("""
You are a patient, friendly tutor helping a student clear doubts about ONE
YouTube video titled "{title}".

You are given VIDEO CONTENT (the most relevant parts of the video) and the
recent CHAT HISTORY.

HOW TO ANSWER:
1. Answer the actual question directly first, in one or two lines.
2. Then explain clearly, step by step, like a good teacher: simple words,
   short paragraphs, bullets or numbered steps where they help. Reuse the
   video's own examples and analogies.
3. Base the answer on the VIDEO CONTENT. Never invent things the video did
   not say and never claim the video said something it did not.
4. If the video covers the doubt only partly, explain the covered part and
   then add a short, clearly labelled extra section (in the response
   language) saying it is extra information from outside the video. If the
   video does not cover the doubt at all, say so clearly and then give a
   short, accurate explanation under the same kind of label.
5. Use CHAT HISTORY to understand follow-up questions (what "it", "this",
   "uska", "isme" refer to).
6. Be precise with definitions, formulas and steps. For a long answer, end
   with a one-line recap.
7. Do not mention "context", "chunks" or "retrieval". Say "video me" /
   "in the video" instead.
8. TIMESTAMPS: the VIDEO CONTENT has time markers like [05:32] showing where
   each part is in the video. Tell the student WHEN the topic was discussed,
   e.g. "Yeh video me [05:32] par samjhaya gaya hai". Write every time
   exactly as [MM:SS] (or [H:MM:SS]) and use ONLY markers that appear in the
   VIDEO CONTENT; never guess or invent a time. Mention the 1-3 most
   relevant times and finish with a line like "Video me dekhein: [05:32],
   [07:10]" (translated to the response language). Extra information from
   outside the video gets no timestamp. If the content has no time markers,
   do not mention any time.

RESPONSE LANGUAGE:
English: natural English.
Hindi: natural Hindi in Devanagari.
Hinglish: Hindi-English mix in Roman script.
Keep technical terms in their original form where appropriate.

Selected response language: {language}

CHAT HISTORY:
{history}

VIDEO CONTENT:
{context}

STUDENT'S QUESTION:
{question}

Give a clear, helpful answer that actually clears the doubt.
""")


def _format_history(history) -> str:
    if not history:
        return "(no earlier messages)"

    lines = []
    for item in history[-6:]:
        role = "Student" if item.get("role") == "user" else "Tutor"
        content = str(item.get("content", "")).strip().replace("\n", " ")
        lines.append(f"{role}: {content[:600]}")

    return "\n".join(lines)


def ask_question(video_id: str, question: str, language="English",
                 video_url=None, history=None) -> str:
    language = validate_language(language)

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()
    data = ensure_loaded(video_id, video_url)

    # Short follow-ups ("explain more") retrieve better with the previous question.
    search_query = question
    if history and len(question) < 50:
        previous = [h for h in history if h.get("role") == "user"]
        if previous:
            search_query = f"{previous[-1].get('content', '')} {question}"

    relevant = search_chunks(data, search_query, k=TOP_K)

    if not relevant:
        raise ValueError("No relevant video content found.")

    context = "\n\n---\n\n".join(relevant)

    chain = CHAT_PROMPT | llm | StrOutputParser()

    return chain.invoke({
        "title": data["video"].get("title", "YouTube Video"),
        "language": language,
        "history": _format_history(history),
        "context": context,
        "question": question,
    })


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(video_id: str, language="English", video_url=None) -> str:
    language = validate_language(language)

    data = ensure_loaded(video_id, video_url)
    combined = (data.get("notes") or "").strip()

    if not combined:
        raise ValueError("No text is available to summarize.")

    language_instruction = """
    English: natural, simple English.
    Hindi: natural Hindi in Devanagari.
    Hinglish: simple Hindi-English mix in Roman script.
    Keep technical terms in their original form where appropriate.
    """

    chunk_size = 4000
    pieces = [combined[i:i + chunk_size] for i in range(0, len(combined), chunk_size)]

    if len(pieces) == 1:
        source_content = pieces[0]
    else:
        chunk_prompt = ChatPromptTemplate.from_template("""
You are extracting accurate study notes from one section of a YouTube video.

Write concise bullet-point notes in {language}.

- Keep important concepts, definitions, examples, steps and components.
- Do not invent information or add facts from outside knowledge.
- Keep the [MM:SS] time markers next to the points they belong to.
- Do not write an introduction or conclusion.
- Use headings and bullets, not tables.
- This is only one section; do not imply it represents the whole video.

SOURCE SECTION:
{content}
""")

        chunk_chain = chunk_prompt | summary_llm | StrOutputParser()

        notes_list = []
        for index, piece in enumerate(pieces, start=1):
            notes = invoke_summary_with_retry(chunk_chain, {
                "content": piece,
                "language": language,
            })
            notes_list.append(f"VIDEO SECTION {index} NOTES:\n{notes}")

        source_content = "\n\n".join(notes_list)

    final_prompt = ChatPromptTemplate.from_template("""
You are a professional YouTube video summarizer and study-notes creator.

Create a concise, well-organized, POINT-WISE summary in {language}.

STRICT FORMATTING RULES:
- Return Markdown headings, numbered lists and bullet points.
- Never use tables.
- Never write long paragraphs.
- Use short, clear, useful points.
- Do not output HTML tags such as <br>.
- Use **bold** only for important terms; do not escape Markdown characters.
- Where the source has [MM:SS] time markers, put the start time like
  [05:32] in each main topic heading. Never invent times.
- Do not invent names, definitions, examples, facts or components.
- Do not repeat the same information in multiple sections.
- Summarize only what the supplied source supports.

FOLLOW THIS OUTPUT STRUCTURE:

## 1. Video Overview
- Give 2-4 short bullet points.

## 2. Main Topics Covered
- List important topics in the order they appear.
- Use a short bold topic heading with 1-3 concise bullet points under it.

## 3. Key Learning Points
- Give the main concepts as numbered or bulleted points, explained briefly.

## 4. Important Terms / Components
- List important terms or names and explain them briefly.

## 5. Conclusion
- Give 2-3 short bullet points.

LANGUAGE:
{language_instruction}

Selected response language: {language}

VIDEO NOTES / SOURCE:
{content}
""")

    final_chain = final_prompt | summary_llm | StrOutputParser()

    summary = invoke_summary_with_retry(final_chain, {
        "content": source_content,
        "language": language,
        "language_instruction": language_instruction,
    })

    return summary.strip()