"""
Hybrid Unsupervised Semantic Topic Segmentation
with LLaMA-assisted Human-like Chapter Generation

✔ Windows-safe
✔ Django-safe
✔ Filename-safe
✔ Production-ready
"""

# =========================================================
# IMPORTS
# =========================================================

from pathlib import Path
import json
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# =========================================================
# PATH CONFIGURATION (ABSOLUTE, NEVER BREAKS)
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRANSCRIPT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "Transcription"
SEGMENT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "T_segmentation"

SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# ENV & GROQ CLIENT
# =========================================================

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================================
# EMBEDDING MODEL (LAZY LOADING)
# =========================================================

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def sec_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def clean_text(text: str) -> str:
    """Minimal cleaning (best for embeddings)"""
    return re.sub(r"\s+", " ", text).strip()


def safe_stem(name: str, max_len: int = 60) -> str:
    """
    Fully Windows-safe filename:
    - removes unsafe characters
    - limits length
    """
    return "".join(c for c in name if c.isalnum() or c in "_-")[:max_len]


# =========================================================
# LLaMA TITLE GENERATION (GROQ)
# =========================================================

def generate_llama_title(text: str) -> str:
    prompt = f"""
Create a short podcast chapter title.

Rules:
- Max 6 words
- Natural language
- No punctuation

Text:
{text}

Title:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.3,
    )

    title = response.choices[0].message.content.strip()
    return title if title else "General Discussion"


# =========================================================
# CORE SEGMENTATION FUNCTION
# =========================================================

def segment_whisper_file(json_file: Path) -> Path:
    """
    Segments a Whisper JSON transcript.
    RETURNS the actual output file path (IMPORTANT for Django).
    """

    json_file = Path(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if len(segments) < 2:
        raise ValueError("Not enough segments for topic segmentation")

    texts = [clean_text(s["text"]) for s in segments]
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]

    embedder = get_embedder()
    embeddings = embedder.encode(texts)

    SIM_THRESHOLD = 0.55
    MIN_SEG_SIZE = 5
    MIN_GAP_SEC = 80

    blocks = []
    block_start = 0
    current_emb = embeddings[0]

    for i in range(1, len(embeddings)):
        sim = cosine_similarity(
            current_emb.reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if sim < SIM_THRESHOLD and i - block_start >= MIN_SEG_SIZE:
            blocks.append((block_start, i - 1))
            block_start = i
            current_emb = embeddings[i]
        else:
            current_emb = (current_emb + embeddings[i]) / 2

    blocks.append((block_start, len(texts) - 1))

    # =====================================================
    # SAFE OUTPUT FILE (NO GUESSING)
    # =====================================================

    safe_name = safe_stem(json_file.stem)
    out_file = SEGMENT_DIR / f"{safe_name}_topics.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("00:00 - Introduction\n")

        last_written_time = 0

        for s, _ in blocks:
            start_sec = starts[s]

            if start_sec < 10:
                continue

            if start_sec - last_written_time < MIN_GAP_SEC:
                continue

            context = " ".join(texts[s:s + 5])
            title = generate_llama_title(context)

            f.write(f"{sec_to_mmss(start_sec)} - {title}\n")
            last_written_time = start_sec

        f.write(f"{sec_to_mmss(ends[-1])} - Conclusion\n")

    return out_file  # 🔥 THIS IS THE KEY FIX


# =========================================================
# DJANGO PIPELINE ENTRY (USE THIS ONLY)
# =========================================================

def segment_from_json(json_path) -> Path:
    """
    Django-safe wrapper.
    Always returns the REAL output file path.
    """
    return segment_whisper_file(Path(json_path))


# =========================================================
# BATCH MODE (OPTIONAL)
# =========================================================

def segment_all():
    for json_file in TRANSCRIPT_DIR.glob("*.json"):
        segment_whisper_file(json_file)


if __name__ == "__main__":
    segment_all()
