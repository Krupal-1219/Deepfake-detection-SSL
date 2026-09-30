"""Per-face analysis: verdict, test-time augmentation, heatmaps and regions."""
import base64
import io

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from .model import MEAN, STD

_PATCH_GRID = 16
_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# The five views from newphase2._TTA_TRANSFORMS. The brightness view uses a
# fixed shift instead of a random ColorJitter so results are repeatable.
TTA_VIEWS = [
    ("Original",     lambda im: TF.center_crop(TF.resize(im, 224), 224)),
    ("Mirrored",     lambda im: TF.hflip(TF.center_crop(TF.resize(im, 224), 224))),
    ("Zoomed in",    lambda im: TF.center_crop(TF.resize(im, 256), 224)),
    ("Zoomed out",   lambda im: TF.center_crop(TF.pad(TF.resize(im, 200), 12), 224)),
    ("Light shift",  lambda im: TF.adjust_contrast(TF.adjust_brightness(
        TF.center_crop(TF.resize(im, 224), 224), 1.1), 1.1)),
]


def to_data_url(arr_or_img, fmt="WEBP", quality=88):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


# Rendering helpers, same as newphase2._heatmap_to_color / _blend.
def _heatmap_to_color(scores_256, grid=_PATCH_GRID):
    patch_map = scores_256.reshape(grid, grid).astype(np.float32)
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
    patch_big = cv2.resize((patch_map * 255).astype(np.uint8), (224, 224),
                           interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(patch_big, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _blend(face_rgb, heatmap_rgb, alpha=0.45):
    return np.clip((1 - alpha) * face_rgb.astype(np.float32)
                   + alpha * heatmap_rgb.astype(np.float32), 0, 255).astype(np.uint8)


def _noise_map(model, x):
    """Magnitude of the SRM noise residuals the frequency branch reads."""
    res = model.freq_encoder.srm(x)[0].abs().mean(0).cpu().numpy()
    res = res / (np.percentile(res, 99.5) + 1e-8)
    res = (np.clip(res, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(res, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)


def _regions(scores_256, landmarks):
    """Average anomaly per facial region, using the detector's 5 landmarks."""
    if len(landmarks) != 5:
        return []
    lm = np.array(landmarks, dtype=np.float32)
    re, le, nose, mr, ml = lm
    d = max(float(np.linalg.norm(re - le)), 8.0)
    mouth = (mr + ml) / 2
    centre = (re + le + mouth * 2) / 4
    grid = scores_256.reshape(_PATCH_GRID, _PATCH_GRID)
    norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    buckets = {}
    for i in range(_PATCH_GRID):
        for j in range(_PATCH_GRID):
            p = np.array([7 + 14 * j, 7 + 14 * i], dtype=np.float32)
            e = ((p[0] - centre[0]) / (1.05 * d)) ** 2 + ((p[1] - centre[1]) / (1.4 * d)) ** 2
            if min(np.linalg.norm(p - re), np.linalg.norm(p - le)) < 0.38 * d:
                name = "Eyes"
            elif np.linalg.norm(p - mouth) < 0.5 * d:
                name = "Mouth"
            elif np.linalg.norm(p - nose) < 0.32 * d:
                name = "Nose"
            elif e <= 0.8:
                name = "Cheeks and forehead"
            elif e <= 1.5:
                name = "Face edge (jaw, hairline)"
            else:
                name = "Background"
            buckets.setdefault(name, []).append(float(norm[i, j]))
    hot = np.sort(norm.ravel())[-26]  # top 10% of patches
    out = []
    for name, vals in buckets.items():
        vals = np.array(vals)
        out.append({"name": name, "mean": round(float(vals.mean()), 4),
                    "hot_share": round(float((vals >= hot).mean()), 4),
                    "patches": int(len(vals))})
    return sorted(out, key=lambda r: r["mean"], reverse=True)


def band(p):
    if p < 0.3:
        return "authentic"
    if p < 0.7:
        return "inconclusive"
    return "manipulated"


@torch.no_grad()
def analyze_face(model, crop: Image.Image, landmarks):
    device = next(model.parameters()).device
    views = [fn(crop) for _, fn in TTA_VIEWS]
    batch = torch.stack([_normalize(v) for v in views]).to(device)
    logits, heat, fgw, *_ = model(batch)
    probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()

    scores = heat[0].cpu().numpy()
    face_rgb = np.asarray(views[0])
    heat_rgb = _heatmap_to_color(scores)
    p = float(probs.mean())
    std = float(probs.std())
    return {
        "probability": round(p, 4),
        "verdict": band(p),
        "tta": {
            "views": [{"name": n, "probability": round(float(v), 4)}
                      for (n, _), v in zip(TTA_VIEWS, probs)],
            "std": round(std, 4),
            "agreement": "high" if std < 0.08 else "medium" if std < 0.18 else "low",
        },
        "fgw_distance": round(float(fgw[0]), 4),
        "regions": _regions(scores, landmarks),
        "patch_scores": [round(float(s), 4) for s in scores],
        "images": {
            "crop": to_data_url(face_rgb),
            "overlay": to_data_url(_blend(face_rgb, heat_rgb)),
            "heatmap": to_data_url(heat_rgb),
            "noise": to_data_url(_noise_map(model, batch[:1])),
        },
    }
