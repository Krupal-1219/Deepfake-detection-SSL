"""Second detector: AI-generated / AI-edited images (diffusion, Midjourney, etc.).

Uses a ready-made SigLIP classifier from Hugging Face; it complements the
face-swap model, which predates diffusion-based generators.
"""
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = os.environ.get("AIGEN_MODEL", "Ateeqq/ai-vs-human-image-detector")


class AIGenDetector:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = SiglipForImageClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ai_index = labels.get("ai", 0)

    @torch.no_grad()
    def score(self, img: Image.Image) -> float:
        """Probability (0-1) that the image is AI-generated or AI-edited."""
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        return float(self.model(**inputs).logits.softmax(-1)[0, self.ai_index])
