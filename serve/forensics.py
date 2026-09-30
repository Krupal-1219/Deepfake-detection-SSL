"""Model-free file forensics: metadata, AI-generator traces, JPEG history, ELA."""
import hashlib
import io
import re

import numpy as np
from PIL import ExifTags, Image, ImageChops

from .analyze import to_data_url

EXIF_FIELDS = ["Make", "Model", "LensModel", "Software", "DateTime", "DateTimeOriginal",
               "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength",
               "Artist", "Copyright", "ImageDescription", "HostComputer"]
AI_MARKERS = {
    "stable diffusion": "Stable Diffusion", "sd_xl": "Stable Diffusion XL",
    "sdxl": "Stable Diffusion XL", "midjourney": "Midjourney",
    "dall-e": "DALL-E", "dall·e": "DALL-E", "openai": "OpenAI", "firefly": "Adobe Firefly",
    "comfyui": "ComfyUI", "automatic1111": "AUTOMATIC1111", "novelai": "NovelAI",
    "invokeai": "InvokeAI", "imagen": "Google Imagen", "gemini": "Google Gemini",
    "leonardo.ai": "Leonardo.Ai", "flux": "FLUX", "ideogram": "Ideogram",
    "trainedalgorithmicmedia": "IPTC tag: made by a generative model",
    "compositewithtrainedalgorithmicmedia": "IPTC tag: composited with generative AI",
}
EDITORS = ["photoshop", "lightroom", "gimp", "snapseed", "facetune", "picsart", "canva",
           "affinity", "pixelmator", "faceapp", "remini", "photoroom", "meitu"]

# IJG standard luminance quantisation table (quality 50), in zigzag-free order.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def _jpeg_quality(img):
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    luma = np.array(tables[0], dtype=np.float64)
    if luma.size != 64:
        return None
    best, best_err = None, float("inf")
    for q in range(1, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        t = np.clip(np.floor((_STD_LUMA * s + 50) / 100), 1, 255)
        err = np.abs(np.sort(t) - np.sort(luma)).sum()
        if err < best_err:
            best, best_err = q, err
    return {"estimate": best, "exact_match": bool(best_err == 0)}


def _exif(original):
    out, gps = {}, False
    try:
        exif = original.getexif()
    except Exception:
        return out, gps
    if not exif:
        return out, gps
    tags = dict(exif.items())
    try:
        tags.update(exif.get_ifd(0x8769).items())  # Exif sub-IFD: capture settings
    except Exception:
        pass
    gps = bool(exif.get_ifd(0x8825)) if 0x8825 in exif else False
    for tag_id, value in tags.items():
        name = ExifTags.TAGS.get(tag_id)
        if name not in EXIF_FIELDS:
            continue
        if isinstance(value, bytes):
            value = value.decode(errors="ignore").strip("\x00 ")
        elif hasattr(value, "numerator"):
            value = float(value)
            value = round(value, 4) if value < 1 else round(value, 2)
        text = str(value).strip()
        if text:
            out[name] = text[:200]
    return out, gps


def _ela(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    diff = ImageChops.difference(img, Image.open(buf).convert("RGB"))
    arr = np.asarray(diff).astype(np.float32).max(axis=2)
    peak = float(np.percentile(arr, 99.7)) or 1.0
    arr = np.clip(arr * (255.0 / peak), 0, 255).astype(np.uint8)
    ela = Image.fromarray(arr).convert("RGB")
    ela.thumbnail((900, 900))
    return to_data_url(ela), round(float(np.asarray(diff).mean()), 3)


def analyze_file(data: bytes, filename: str, original: Image.Image, upright: Image.Image):
    head = data[:2_000_000].lower()
    exif, gps = _exif(original)

    text_chunks = {}
    for k, v in (original.info or {}).items():
        if isinstance(v, str) and k not in ("exif", "icc_profile", "xmp"):
            text_chunks[k] = v[:600]
    xmp = ""
    m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", data[:2_000_000], re.S)
    if m:
        xmp = m.group(0).decode(errors="ignore")
    creator_tool = re.search(r'CreatorTool[=>"\s]+([^"<]{2,120})', xmp)

    haystack = " ".join([exif.get("Software", ""), creator_tool.group(1) if creator_tool else "",
                         " ".join(f"{k} {v}" for k, v in text_chunks.items()), xmp]).lower()
    ai = sorted({label for key, label in AI_MARKERS.items() if key in haystack})
    if "parameters" in text_chunks and re.search(r"steps:\s*\d+", text_chunks["parameters"], re.I):
        ai.append("Diffusion generation parameters in PNG text")
    editors = sorted({e.title() for e in EDITORS if e in haystack})

    fmt = original.format or "unknown"
    jpeg = None
    if fmt == "JPEG":
        from PIL import JpegImagePlugin
        sampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(
            JpegImagePlugin.get_sampling(original), "unknown")
        jpeg = {"quality": _jpeg_quality(original), "subsampling": sampling,
                "progressive": bool(original.info.get("progressive"))}

    ela_img, ela_mean = _ela(upright)
    return {
        "name": filename,
        "format": fmt,
        "mime": Image.MIME.get(fmt, "application/octet-stream"),
        "bytes": len(data),
        "width": original.width,
        "height": original.height,
        "mode": original.mode,
        "sha256": hashlib.sha256(data).hexdigest(),
        "exif": exif,
        "gps_present": gps,
        "creator_tool": creator_tool.group(1).strip() if creator_tool else None,
        "text_chunks": text_chunks,
        "ai_markers": ai,
        "editing_software": editors,
        "c2pa": b"c2pa" in head,
        "icc_profile": bool(original.info.get("icc_profile")),
        "jpeg": jpeg,
        "ela": {"image": ela_img, "mean_error": ela_mean},
    }
