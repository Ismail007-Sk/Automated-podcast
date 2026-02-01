from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse

import os
import uuid
import shutil
import time
import threading
from pathlib import Path
import json

from .Service.preprocessing import preprocess_audio
from .Service.transcription import transcribe_audio
from .Service.segmentation import segment_from_json
from .Service.question import answer_from_transcript


# ==================================================
# BASE PATHS (ABSOLUTE, SAFE)
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "appAudio" / "Service" / "Dataset" / "Raw"
PREPROCESS_DIR = BASE_DIR / "appAudio" / "Service" / "Dataset" / "Preprocessed"
TRANS_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "Transcription"
SEG_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "T_segmentation"

MEDIA_TRANS = Path(settings.MEDIA_ROOT) / "transcription"
MEDIA_SEG = Path(settings.MEDIA_ROOT) / "segmentation"

for d in [RAW_DIR, PREPROCESS_DIR, TRANS_DIR, SEG_DIR, MEDIA_TRANS, MEDIA_SEG]:
    d.mkdir(parents=True, exist_ok=True)


# ==================================================
# CLEANUP HELPERS
# ==================================================

def clean_old_files(folder: Path):
    if not folder.exists():
        return
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()


def delete_later(path: Path, delay=300):
    def _delete():
        time.sleep(delay)
        if path.exists():
            path.unlink()
    threading.Thread(target=_delete, daemon=True).start()


# ==================================================
# MAIN VIEW
# ==================================================

def upload_audio(request):
    context = {}

    if request.method == "POST":
        action = request.POST.get("action")

        # ---------------- CANCEL ----------------
        if action == "cancel":
            return render(request, "upload.html", {
                "status": "❌ Cancelled by user"
            })

        # ---------------- UPLOAD ----------------
        if action == "upload":
            audio_file = request.FILES.get("audio")
            if not audio_file:
                return render(request, "upload.html", {
                    "status": "No file selected"
                })

            # cleanup old files
            clean_old_files(RAW_DIR)
            clean_old_files(PREPROCESS_DIR)
            clean_old_files(TRANS_DIR)
            clean_old_files(SEG_DIR)

            filename = f"{uuid.uuid4()}_{audio_file.name}"
            file_path = RAW_DIR / filename

            with open(file_path, "wb+") as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            return render(request, "upload.html", {
                "status": "✅ Upload completed",
                "uploaded_file": filename
            })

        # ---------------- PROCESS ----------------
        if action == "process":
            start_time = time.time()

            uploaded_file = request.POST.get("uploaded_file")
            raw_audio_path = RAW_DIR / uploaded_file

            # ---- PIPELINE ----
            clean_audio_path = preprocess_audio(raw_audio_path)
            json_path = transcribe_audio(clean_audio_path)

            # 🔥 IMPORTANT: USE RETURNED PATH
            seg_path = segment_from_json(json_path)

            # transcription txt
            txt_path = TRANS_DIR / (Path(json_path).stem + ".txt")

            # ---- COPY TO MEDIA ----
            trans_dst = MEDIA_TRANS / txt_path.name
            seg_dst = MEDIA_SEG / seg_path.name

            shutil.copy(txt_path, trans_dst)
            shutil.copy(seg_path, seg_dst)

            # sizes
            txt_size = trans_dst.stat().st_size // 1024
            seg_size = seg_dst.stat().st_size // 1024

            # auto delete downloads
            delete_later(trans_dst, delay=1000)
            delete_later(seg_dst, delay=1000)

            # cleanup raw audio
            if raw_audio_path.exists():
                raw_audio_path.unlink()

            context.update({
                "status": "✅ Processing completed",
                "uploaded_file": uploaded_file,
                "transcription_file": f"{settings.MEDIA_URL}transcription/{trans_dst.name}",
                "segmentation_file": f"{settings.MEDIA_URL}segmentation/{seg_dst.name}",
                "txt_size": txt_size,
                "seg_size": seg_size,
                "time_taken": round(time.time() - start_time, 2),
                "done": True
            })

            return render(request, "upload.html", context)

    return render(request, "upload.html")


# ==================================================
# HOME
# ==================================================

def home(request):
    return render(request, "home.html")


# ==================================================
# CHAT / Q&A
# ==================================================

def chat_from_transcript(request):
    if request.method != "POST":
        return JsonResponse({"answer": "Invalid request"})

    data = json.loads(request.body)
    question = data.get("question", "").strip()

    answer = answer_from_transcript(question)
    return JsonResponse({"answer": answer})
