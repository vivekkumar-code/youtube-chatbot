
import os
import re
import cv2
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pytesseract
from dotenv import load_dotenv
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env file.")

SUPPORTED_LANGUAGES = ["English", "Hindi", "Hinglish"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

OCR_INTERVAL_SECONDS = int(os.getenv("OCR_INTERVAL_SECONDS", "30"))
OCR_MAX_WIDTH = 1600

VIDEO_CACHE = {}

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.name == "nt" and os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT

print("Loading multilingual embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding model loaded.")

print("Initializing Groq LLM...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=3000,
    api_key=GROQ_API_KEY,
)

print("Groq LLM initialized.")


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


def format_timestamp(seconds) -> str:

    seconds = max(0, int(float(seconds or 0)))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def get_video_info(video_id: str) -> dict:

    thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    if not YOUTUBE_API_KEY:
        return {
            "video_id": video_id,
            "title": "YouTube Video",
            "channel": "Unknown",
            "description": "",
            "thumbnail": thumbnail,
            "duration": "Unknown",
            "default_audio_language": None,
        }

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        cache_discovery=False
    )

    response = youtube.videos().list(
        part="snippet,contentDetails",
        id=video_id
    ).execute()

    items = response.get("items", [])

    if not items:
        raise ValueError("Video not found or unavailable.")

    video = items[0]
    snippet = video["snippet"]

    return {
        "video_id": video_id,
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "description": snippet.get("description", ""),
        "thumbnail": snippet.get("thumbnails", {}).get(
            "high", {}
        ).get("url", thumbnail),
        "duration": video["contentDetails"]["duration"],
        "default_audio_language": (
            snippet.get("defaultAudioLanguage")
            or snippet.get("defaultLanguage")
        ),
    }


# =========================================================
# TRANSCRIPT EXTRACTION
# =========================================================

def extract_transcript(video_id: str, preferred_language=None) -> dict:

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
    segments = []

    for item in fetched:

        text = item.text.strip()

        if text:
            segments.append({
                "timestamp": format_timestamp(item.start),
                "start": float(item.start),
                "text": text,
            })

    if not segments:
        raise ValueError("Transcript is empty.")

    return {
        "language": selected.language,
        "language_code": selected.language_code,
        "is_generated": selected.is_generated,
        "segments": segments,
    }


# =========================================================
# LOCAL VIDEO DOWNLOAD + OCR
# =========================================================

def download_video_for_ocr(video_url: str, output_dir: str) -> str:
    """
    Downloads one progressive video stream for local frame OCR.
    Prefers MP4 up to 360p to reduce download size.
    """

    options = {
        "format": (
            "best[height<=360][ext=mp4][vcodec!=none][acodec!=none]"
            "/best[height<=360][vcodec!=none][acodec!=none]"
            "/best[ext=mp4][vcodec!=none][acodec!=none]/best"
        ),
        "outtmpl": os.path.join(output_dir, "source_video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
    }

    with YoutubeDL(options) as ydl:

        info = ydl.extract_info(video_url, download=True)

        requested = info.get("requested_downloads") or []

        if requested and requested[0].get("filepath"):
            candidate = requested[0]["filepath"]

            if os.path.exists(candidate):
                return candidate

        candidate = ydl.prepare_filename(info)

        if os.path.exists(candidate):
            return candidate

    for path in Path(output_dir).iterdir():

        if path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return str(path)

    raise RuntimeError(
        "yt-dlp finished but the downloaded video file was not found."
    )


def _available_ocr_languages():

    try:

        langs = set(pytesseract.get_languages(config=""))

        if "eng" in langs and "hin" in langs:
            return "eng+hin"

        if "eng" in langs:
            return "eng"

        if "hin" in langs:
            return "hin"

        return "eng"

    except Exception:
        return "eng"


def preprocess_frame(frame):

    height, width = frame.shape[:2]

    if width > OCR_MAX_WIDTH:

        scale = OCR_MAX_WIDTH / width

        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold


def extract_slide_text(video_path: str, interval_seconds=None) -> list:
    """
    Samples a frame every N seconds and extracts visible slide text.
    """

    interval = interval_seconds or OCR_INTERVAL_SECONDS

    if interval < 5:
        interval = 5

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the downloaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    duration = frame_count / fps if fps > 0 else 0

    if duration <= 0:
        capture.release()
        raise RuntimeError("Could not determine video duration for OCR.")

    ocr_lang = _available_ocr_languages()

    records = []
    previous_normalized = ""

    print(
        f"Running local OCR every {interval}s "
        f"(duration: {format_timestamp(duration)}, lang: {ocr_lang})..."
    )

    second = 0.0

    while second < duration:

        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)

        ok, frame = capture.read()

        if ok and frame is not None:

            processed = preprocess_frame(frame)

            try:

                text = pytesseract.image_to_string(
                    processed,
                    lang=ocr_lang,
                    config="--oem 3 --psm 6"
                )

            except Exception as exc:

                capture.release()

                raise RuntimeError(
                    "Tesseract OCR failed. Check Tesseract installation "
                    f"and language data. Details: {exc}"
                )

            lines = [
                re.sub(r"\s+", " ", line).strip()
                for line in text.splitlines()
            ]

            cleaned = "\n".join(
                line for line in lines if len(line) >= 2
            )

            normalized = re.sub(r"\W+", "", cleaned.lower())

            if (
                cleaned
                and len(normalized) >= 8
                and normalized != previous_normalized
            ):

                records.append({
                    "timestamp": format_timestamp(second),
                    "start": float(second),
                    "text": cleaned,
                })

                previous_normalized = normalized

        second += interval

    capture.release()

    print(f"OCR extracted text from {len(records)} distinct frames.")

    return records


def extract_video_frames_ocr(video_url: str) -> list:
    """
    Uses a temporary directory. The downloaded video is deleted afterwards.
    OCR failure is non-fatal.
    """

    temp_dir = tempfile.mkdtemp(prefix="yt_chatbot_ocr_")

    try:

        video_path = download_video_for_ocr(video_url, temp_dir)

        return extract_slide_text(video_path)

    finally:

        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================
# HYBRID FAISS INDEX: TRANSCRIPT + SLIDE OCR
# =========================================================

def create_hybrid_vector_store(
    video_id: str,
    transcript_segments: list,
    ocr_segments: list,
    language_code: str,
):

    safe_language = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        language_code or "unknown"
    )

    index_folder = os.path.join(
        INDEX_DIR,
        f"{video_id}_{safe_language}_hybrid_ocr_v1"
    )

    index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if os.path.isfile(index_file) and os.path.isfile(metadata_file):

        print("Loading existing Hybrid RAG FAISS index...")

        return FAISS.load_local(
            index_folder,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents = []

    chunk = []
    chunk_start = None
    chunk_end = None
    chunk_chars = 0

    def save_transcript_chunk():

        nonlocal chunk, chunk_start, chunk_end, chunk_chars

        if not chunk:
            return

        start_label = format_timestamp(chunk_start)
        end_label = format_timestamp(chunk_end)

        body = "\n".join(
            f"[{item['timestamp']}] {item['text']}" for item in chunk
        )

        documents.append(Document(
            page_content=(
                f"SOURCE: VIDEO TRANSCRIPT\n"
                f"VIDEO TIME: {start_label} - {end_label}\n{body}"
            ),
            metadata={
                "video_id": video_id,
                "source": "transcript",
                "timestamp": start_label,
                "end_timestamp": end_label,
                "start": float(chunk_start),
                "language_code": language_code or "unknown",
            },
        ))

        chunk = []
        chunk_start = None
        chunk_end = None
        chunk_chars = 0

    for item in transcript_segments:

        text = str(item.get("text", "")).strip()

        if not text:
            continue

        current = float(item.get("start", 0))

        if chunk and (
            current - chunk_start >= 30
            or chunk_chars + len(text) > 1200
        ):

            save_transcript_chunk()

        if chunk_start is None:
            chunk_start = current

        chunk_end = current
        chunk.append(item)
        chunk_chars += len(text)

    save_transcript_chunk()

    for item in ocr_segments:

        text = str(item.get("text", "")).strip()

        if not text:
            continue

        timestamp = item.get("timestamp", "Unknown")
        start = float(item.get("start", 0))

        documents.append(Document(
            page_content=(
                "SOURCE: VIDEO SLIDE / ON-SCREEN TEXT (OCR)\n"
                f"VIDEO TIMESTAMP: {timestamp}\n"
                "TEXT READ FROM THE VIDEO FRAME:\n"
                f"{text}"
            ),
            metadata={
                "video_id": video_id,
                "source": "slide_ocr",
                "timestamp": timestamp,
                "end_timestamp": timestamp,
                "start": start,
                "language_code": language_code or "unknown",
            },
        ))

    if not documents:
        raise ValueError(
            "No searchable content was extracted from transcript or video frames."
        )

    print(
        f"Creating Hybrid FAISS index: "
        f"{len(transcript_segments)} transcript segments, "
        f"{len(ocr_segments)} OCR frames, {len(documents)} documents."
    )

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    os.makedirs(index_folder, exist_ok=True)

    vector_store.save_local(index_folder)

    print("Hybrid FAISS index saved successfully.")

    return vector_store


# =========================================================
# LOAD VIDEO
# =========================================================

def load_video(video_url: str) -> dict:

    video_id = extract_video_id(video_url)

    if video_id in VIDEO_CACHE:
        print("Video found in memory cache.")
        return VIDEO_CACHE[video_id]

    print("Fetching YouTube video information...")

    video_info = get_video_info(video_id)

    transcript_data = {
        "language": "Unavailable",
        "language_code": "unknown",
        "is_generated": False,
        "segments": [],
    }

    try:

        print("Extracting YouTube transcript...")

        transcript_data = extract_transcript(
            video_id,
            video_info.get("default_audio_language"),
        )

        print(
            f"Transcript extracted: {len(transcript_data['segments'])} "
            f"segments ({transcript_data['language']})."
        )

    except Exception as exc:

        print(f"Transcript extraction unavailable: {exc}")

    ocr_segments = []
    ocr_error = None

    try:

        print("Downloading video temporarily for local slide OCR...")

        ocr_segments = extract_video_frames_ocr(video_url)

    except Exception as exc:

        ocr_error = str(exc)

        print("Local OCR skipped/failed:", ocr_error)

    vector_store = create_hybrid_vector_store(
        video_id=video_id,
        transcript_segments=transcript_data["segments"],
        ocr_segments=ocr_segments,
        language_code=transcript_data["language_code"],
    )

    video_data = {
        "video": video_info,
        "segments": transcript_data["segments"],
        "transcript_language": transcript_data["language"],
        "transcript_language_code": transcript_data["language_code"],
        "transcript_is_generated": transcript_data["is_generated"],
        "ocr_segments": ocr_segments,
        "ocr_error": ocr_error,
        "vector_store": vector_store,
    }

    VIDEO_CACHE[video_id] = video_data

    print("Video loaded successfully with Hybrid RAG.")

    return video_data


# =========================================================
# LANGUAGE VALIDATION
# =========================================================

def validate_language(language: str) -> str:

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Choose English, Hindi or Hinglish.")

    return language


# =========================================================
# CHAT WITH VIDEO: HYBRID RETRIEVAL
# =========================================================

def ask_question(video_id: str, question: str, language="English") -> str:

    language = validate_language(language)

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if video_id not in VIDEO_CACHE:
        raise ValueError("Video is not loaded. Please load it again.")

    vector_store = VIDEO_CACHE[video_id]["vector_store"]

    relevant_docs = vector_store.similarity_search(
        question.strip(),
        k=10,
    )

    if not relevant_docs:
        raise ValueError("No relevant video content found.")

    context_parts = []

    for doc in relevant_docs:

        source = doc.metadata.get("source", "video")
        timestamp = doc.metadata.get("timestamp", "Unknown")
        end_timestamp = doc.metadata.get("end_timestamp", timestamp)

        source_label = (
            "Transcript" if source == "transcript"
            else "Slide / on-screen OCR"
        )

        context_parts.append(
            f"[Source: {source_label} | Video time: {timestamp} - {end_timestamp}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are an accurate YouTube video tutor using Hybrid RAG.

The retrieved context can contain:
1. Spoken video transcript.
2. Text read from presentation slides or other video frames using OCR.

RULES:
- Answer the user's actual question directly.
- Use only the supplied context as evidence about this video.
- Combine transcript and OCR evidence when both are relevant.
- OCR may contain recognition mistakes. If text is unclear, say so.
- Do not invent facts, definitions, examples, or timestamps.
- A topic heading alone is not enough to claim its full definition.
- If the context does not contain the requested information, clearly say
  that the available transcript/slide text does not provide it.
- Mention whether evidence came from spoken transcript or slide text
  when useful.
- Include timestamps only when present in the retrieved context.
- Use simple, clear language.

RESPONSE LANGUAGE:
English: natural English.
Hindi: natural Hindi in Devanagari.
Hinglish: Hindi-English mix in Roman script.
Keep technical terms in their original form where appropriate.

Selected response language: {language}

RETRIEVED VIDEO CONTEXT:
{context}

USER QUESTION:
{question}

Give a useful, accurate answer.
""")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question.strip(),
        "language": language,
    })


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(video_id: str, language="English") -> str:
    """
    Generate a point-wise summary from the complete available
    transcript and slide OCR content.

    Long videos are processed in chunks so that the beginning,
    middle and ending are all considered.
    """

    language = validate_language(language)

    if video_id not in VIDEO_CACHE:
        raise ValueError("Video is not loaded. Please load it first.")

    video_data = VIDEO_CACHE[video_id]

    transcript_segments = video_data.get("segments", [])
    ocr_segments = video_data.get("ocr_segments", [])

    # Combine transcript and OCR in chronological order.
    timed_pieces = []

    for item in transcript_segments:

        text = str(item.get("text", "")).strip()

        if text:
            timed_pieces.append((
                float(item.get("start", 0)),
                f"[{item.get('timestamp', 'Unknown')}] "
                f"[Transcript] {text}"
            ))

    for item in ocr_segments:

        text = str(item.get("text", "")).strip()

        if text:
            timed_pieces.append((
                float(item.get("start", 0)),
                f"[{item.get('timestamp', 'Unknown')}] "
                f"[Slide OCR] {text}"
            ))

    timed_pieces.sort(key=lambda item: item[0])

    if not timed_pieces:
        raise ValueError(
            "No transcript or slide text is available to summarize."
        )

    language_instruction = {
        "English": "Write in natural, simple English.",
        "Hindi": "Write in natural Hindi using Devanagari script.",
        "Hinglish": "Write in simple Hindi-English mix using Roman script."
    }

    # Split content without cutting individual transcript/OCR records.
    chunk_size = 9000
    chunks = []
    current_chunk = []
    current_length = 0

    for _, piece in timed_pieces:

        piece_length = len(piece)

        if current_chunk and current_length + piece_length > chunk_size:

            chunks.append("\n".join(current_chunk))

            current_chunk = []
            current_length = 0

        current_chunk.append(piece)
        current_length += piece_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Summarize each section when the video is long.
    if len(chunks) > 1:

        section_prompt = ChatPromptTemplate.from_template("""
You are creating study notes from one section of a YouTube video.

Extract only the important information in concise bullet points.

RULES:
- Use short bullet points, not paragraphs.
- Preserve definitions, explanations, examples, steps and components.
- Preserve timestamps exactly as provided.
- Do not invent facts or timestamps.
- Combine transcript and slide OCR only when they support the same point.
- OCR may contain errors; do not guess unclear text.
- Do not add an introduction, conclusion, or table.
- Keep the notes concise but retain the section's important information.
- Write in this language: {language_instruction}

VIDEO SECTION:
{content}

Return only the section notes.
""")

        section_chain = section_prompt | llm | StrOutputParser()

        section_notes = []

        for index, chunk in enumerate(chunks, start=1):

            print(
                f"Generating summary notes for section "
                f"{index}/{len(chunks)}..."
            )

            notes = section_chain.invoke({
                "content": chunk,
                "language_instruction": language_instruction[language]
            })

            section_notes.append(
                f"SECTION {index} NOTES:\n{notes.strip()}"
            )

        source_content = "\n\n".join(section_notes)

    else:
        source_content = chunks[0]

    # Final point-wise summary.
    final_prompt = ChatPromptTemplate.from_template("""
You are a professional YouTube video summarizer and study-notes creator.

Create an accurate, useful and POINT-WISE summary of the supplied video
content in the selected language.

STRICT FORMATTING RULES:
- Use Markdown headings, bullet points and numbered lists.
- Do not use Markdown tables.
- Do not write long paragraphs.
- Keep each bullet concise and informative.
- Use **bold** for important technical terms.
- Do not output HTML tags such as <br>.
- Do not escape Markdown characters with backslashes.
- Do not repeat the same information in multiple sections.
- Do not invent facts, names, definitions, examples or timestamps.
- Use only the supplied content as evidence.
- Preserve exact timestamps only when they appear in the supplied content.
- Treat OCR text cautiously; do not guess unclear words.
- If a detail is not available, leave it out rather than making it up.

FOLLOW THIS OUTPUT STRUCTURE:

## 1. Video Overview
- Give 2-4 short bullet points describing the video.

## 2. Main Topics Covered
- List the important topics in the order they appear.
- Give each topic its own ### heading.
- Add 1-3 short bullet points under each topic.
- Put an exact timestamp in the topic heading only when available.

## 3. Key Learning Points
1. List the most important learning point.
2. List the next important learning point.
3. Continue as needed, without repeating topics.

## 4. Important Terms / Components
- List only terms and components actually supported by the content.
- Explain each briefly when the source provides an explanation.

## 5. Conclusion
- Give 2-3 short bullet points with the main takeaways.

LANGUAGE:
{language_instruction}

VIDEO CONTENT:
{content}

Return only the final summary in the required structure.
""")

    final_chain = final_prompt | llm | StrOutputParser()

    summary = final_chain.invoke({
        "content": source_content,
        "language_instruction": language_instruction[language]
    })

    return summary.strip()
