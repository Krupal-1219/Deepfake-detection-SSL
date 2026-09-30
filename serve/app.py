"""FastAPI server for the deepfake detection web demo.

Run from the repo root:
    uvicorn serve.app:app --port 7860
Uploads are processed in memory and never written to disk or logged.
"""
import asyncio
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .aigen import AIGenDetector
from .analyze import analyze_face, band
from .forensics import analyze_file
from .model import load_detector
from .whole import analyze_whole
from .preprocess import (MAX_FACES, MIN_FACE_PX, FaceFinder,
                         crop_face, load_image)

MAX_BYTES = 10 * 1024 * 1024
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", 12))
ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://localhost:4173,"
    "https://portfolio-gaurav-girish-rathod.vercel.app").split(",")
MODEL_CARD = {
    "name": "DINOv2 + SRM + FGW deepfake detector",
    "repo": "https://github.com/Krupal-1219/Deepfake-detection-SSL",
    "cross_domain_auc": 0.884,
    "trained_on": ["FaceForensics++", "WildDeepfake", "140k Real/Fake (StyleGAN)"],
}
state = {}


def _sniff(data: bytes):
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


@asynccontextmanager
async def lifespan(_app):
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    state["model"], state["info"] = await run_in_threadpool(load_detector)
    state["faces"] = FaceFinder()
    state["aigen"] = await run_in_threadpool(AIGenDetector)
    state["lock"] = asyncio.Lock()
    state["hits"] = defaultdict(deque)
    yield
    state.clear()


app = FastAPI(title="Deepfake forensics API", lifespan=lifespan)
# Any Vercel deployment of the Seam frontend (production and preview URLs) may call the API.
ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://[a-z0-9-]+\.vercel\.app")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_origin_regex=ORIGIN_REGEX,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _rate_limit(ip: str):
    now, q = time.monotonic(), state["hits"][ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many analyses from this address. Try again in a minute.")
    q.append(now)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


@app.get("/info")
def info():
    return {**MODEL_CARD, **state.get("info", {})}


def _run(data: bytes, filename: str):
    t0 = time.perf_counter()
    upright, original = load_image(data)
    faces = state["faces"].detect(upright)
    t_detect = time.perf_counter()

    warnings, results = [], []
    if len(faces) > MAX_FACES:
        warnings.append(f"Found {len(faces)} faces; analysed the {MAX_FACES} largest.")
    for idx, face in enumerate(faces[:MAX_FACES]):
        crop, crop_box, lm = crop_face(upright, face)
        r = analyze_face(state["model"], crop, lm)
        small = min(face["box"][2], face["box"][3]) < MIN_FACE_PX
        results.append({"id": idx, "box": [round(v, 1) for v in face["box"]],
                        "crop_box": [round(v, 1) for v in crop_box],
                        "detector_score": round(face["score"], 3),
                        "small": small, **r})
        if small:
            warnings.append(f"Face {idx + 1} is under {MIN_FACE_PX}px wide, so its result is less reliable.")
    if not faces:
        warnings.append("No face was detected, so the face close-up checks were skipped. "
                        "The whole-photo scan below still covers the entire image.")
    whole = analyze_whole(state["model"], upright)
    ai_p = state["aigen"].score(upright)
    t_model = time.perf_counter()

    file_report = analyze_file(data, filename, original, upright)
    t_end = time.perf_counter()

    name = filename.lower()
    if "screenshot" in name or "screen shot" in name or "whatsapp" in name:
        warnings.append("This looks like a screenshot or a forwarded copy. Re-saving wipes out the fine traces "
                        "detectors rely on, so upload the original image file if you can.")
    elif min(upright.size) < 512:
        warnings.append(f"This image is small ({upright.width} × {upright.height}). Results are more reliable "
                        "on the original, full-resolution file.")

    # Headline: the strongest of the independent signals.
    signals = [{"id": "ai_generated", "label": "AI-generated or AI-edited image", "probability": round(ai_p, 4)}]
    if results:
        top = max(results, key=lambda r: r["probability"])
        signals.append({"id": "face_swap", "label": f"Face swap or reenactment (face {top['id'] + 1})",
                        "probability": top["probability"]})
    signals.append({"id": "whole_frame", "label": "Whole-frame scan", "probability": whole["probability"]})
    if file_report["ai_markers"]:
        signals.append({"id": "metadata", "label": "Generator named in file metadata", "probability": 0.99})
    lead = max(signals, key=lambda s: s["probability"])
    summary = {"probability": lead["probability"], "verdict": band(lead["probability"]),
               "driver": lead["id"], "signals": signals}
    return {
        "id": uuid.uuid4().hex[:12],
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": {"width": upright.width, "height": upright.height},
        "face_detected": bool(faces),
        "summary": summary,
        "faces": results,
        "whole": whole,
        "file": file_report,
        "warnings": warnings,
        "model": {**MODEL_CARD, **state["info"]},
        "timing_ms": {"detect": round((t_detect - t0) * 1000),
                      "model": round((t_model - t_detect) * 1000),
                      "forensics": round((t_end - t_model) * 1000),
                      "total": round((t_end - t0) * 1000)},
    }


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    _rate_limit(request.client.host if request.client else "unknown")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    if not _sniff(data):
        raise HTTPException(415, "Only JPEG, PNG and WebP images are supported.")
    async with state["lock"]:
        try:
            return await run_in_threadpool(_run, data, file.filename or "upload")
        except (OSError, ValueError) as e:
            raise HTTPException(422, f"Could not read this image: {e}") from e
