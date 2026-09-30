"""Whole-photo analysis: a global score plus an overlapping tile sweep that is
stitched into one anomaly map covering the entire image."""
import os

import cv2
import numpy as np
import torch
from PIL import Image

from .analyze import _blend, band, to_data_url, _normalize

MAP_SIDE   = 768   # long side of the stitched maps sent to the browser
# Free CPU hosting is ~10x slower than a GPU, so the tile sweep is capped harder there.
MAX_TILES  = int(os.environ.get("MAX_TILES", 30 if torch.cuda.is_available() else 8))
BATCH      = 8


def _positions(length, side, stride):
    if length <= side:
        return [0]
    pos = list(range(0, length - side + 1, stride))
    if pos[-1] != length - side:
        pos.append(length - side)
    return pos


def _square(img: Image.Image):
    """Pad to a square on a neutral background so the whole frame fits in one input."""
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((224, 224), Image.BICUBIC)


@torch.no_grad()
def _run(model, crops):
    device = next(model.parameters()).device
    probs, heats = [], []
    for i in range(0, len(crops), BATCH):
        x = torch.stack([_normalize(c) for c in crops[i:i + BATCH]]).to(device)
        logits, heat, *_ = model(x)
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        heats.append(heat.cpu().numpy())
    return np.concatenate(probs), np.concatenate(heats)


@torch.no_grad()
def _noise(model, img: Image.Image):
    small = img.copy()
    small.thumbnail((MAP_SIDE, MAP_SIDE))
    x = _normalize(small).unsqueeze(0).to(next(model.parameters()).device)
    res = model.freq_encoder.srm(x)[0].abs().mean(0).cpu().numpy()
    res = res / (np.percentile(res, 99.5) + 1e-8)
    res = (np.clip(res, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(res, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)


def _hotspots(heat, k=3):
    """Top-k peaks of the stitched map, as fractions of width/height."""
    h, w = heat.shape
    work = cv2.GaussianBlur(heat, (0, 0), max(h, w) / 60)
    radius = max(h, w) // 8
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if work[y, x] <= 0:
            break
        out.append({"x": round(x / w, 4), "y": round(y / h, 4), "score": round(float(heat[y, x]), 4)})
        cv2.circle(work, (int(x), int(y)), radius, 0, -1)
    return out


def analyze_whole(model, img: Image.Image):
    W, H = img.size
    global_p, _ = _run(model, [_square(img)])

    # Tile sweep: windows half the short side (never below 224 px), 50% overlap.
    side = min(max(224, round(min(W, H) * 0.5)), min(W, H))
    stride = max(side // 2, 1)
    xs, ys = _positions(W, side, stride), _positions(H, side, stride)
    while len(xs) * len(ys) > MAX_TILES:
        stride = int(stride * 1.25) + 1
        xs, ys = _positions(W, side, stride), _positions(H, side, stride)
    boxes = [(x, y) for y in ys for x in xs]
    crops = [img.crop((x, y, x + side, y + side)).resize((224, 224), Image.BICUBIC) for x, y in boxes]
    probs, heats = _run(model, crops)

    # Stitch the 16x16 patch maps into one map at MAP_SIDE resolution.
    k = MAP_SIDE / max(W, H)
    mw, mh = max(1, round(W * k)), max(1, round(H * k))
    # Each tile is feathered with a raised-cosine window so overlaps blend without seams.
    acc = np.zeros((mh, mw), np.float32)
    cnt = np.zeros((mh, mw), np.float32)
    for (x, y), heat in zip(boxes, heats):
        x0, y0 = round(x * k), round(y * k)
        s = max(1, min(round(side * k), mw - x0, mh - y0))
        patch = cv2.resize(heat.reshape(16, 16).astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC)
        win = np.sin(np.linspace(0, np.pi, s, dtype=np.float32)) ** 2 + 0.02
        weight = np.outer(win, win)
        acc[y0:y0 + s, x0:x0 + s] += patch * weight
        cnt[y0:y0 + s, x0:x0 + s] += weight
    heat = cv2.GaussianBlur(acc / np.maximum(cnt, 1e-6), (0, 0), max(mw, mh) / 200)
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    base = np.asarray(img.resize((mw, mh), Image.BILINEAR))
    colored = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    p = float(global_p[0])
    return {
        "probability": round(p, 4),
        "verdict": band(p),
        "tiles": [{"box": [x, y, side, side], "probability": round(float(q), 4)} for (x, y), q in zip(boxes, probs)],
        "tile_stats": {"count": len(boxes), "side": side,
                       "max": round(float(probs.max()), 4), "mean": round(float(probs.mean()), 4),
                       "above_70": int((probs >= 0.7).sum())},
        "hotspots": _hotspots(norm),
        "images": {
            "overlay": to_data_url(_blend(base, colored, 0.5)),
            "heatmap": to_data_url(colored),
            "noise": to_data_url(_noise(model, img)),
        },
    }
