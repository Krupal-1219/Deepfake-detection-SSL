"""Second detector: AI-generated / AI-edited images (diffusion, Midjourney, etc.).

Uses a ready-made classifier from Hugging Face; it complements the face-swap model,
which predates diffusion-based generators.

Model choice (2026-10-25): compared four pretrained detectors on 200 real + 200 AI
images, 150 real face crops and real phone photos. haywoodsloan/ai-image-detector-deploy
separated them best (AUC 0.984). Ateeqq/ai-vs-human-image-detector, used before,
flagged 31.5% of real images and real phone selfies as 100% AI.
"""
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = os.environ.get("AIGEN_MODEL", "haywoodsloan/ai-image-detector-deploy")

# The raw score is badly calibrated near the top: real photos often land at 0.8-0.95.
# Map it onto the report's verdict scale using thresholds measured on the test set:
#   raw 0.98 -> 0.70 (manipulated line): catches 94% of AI images, flags 4% of real ones
#   raw 0.90 -> 0.30 (inconclusive line): flags 11.5% of real images
CALIBRATION = ([0.0, 0.90, 0.98, 1.0], [0.0, 0.30, 0.70, 1.0])


class AIGenDetector:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ai_index = next(labels[n] for n in ("artificial", "ai", "fake") if n in labels)

    @torch.no_grad()
    def raw_score(self, img: Image.Image) -> float:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        return float(self.model(**inputs).logits.softmax(-1)[0, self.ai_index])

    def score(self, img: Image.Image) -> float:
        """Calibrated probability (0-1) that the image is AI-generated or AI-edited."""
        return float(np.interp(self.raw_score(img), *CALIBRATION))
