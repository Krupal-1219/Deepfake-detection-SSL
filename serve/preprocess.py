"""Image loading and face cropping for the web demo.

The detector was trained on tight 224x224 face crops, so every upload is run
through a face detector first and each face is cropped to a padded square.
"""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

YUNET_PATH  = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
MAX_SIDE    = 2048   # uploads are downscaled to this before anything else
DETECT_SIDE = 1024   # the detector runs on a smaller copy for speed
CROP_MARGIN = 1.3    # square crop side = longest box side * margin
MAX_FACES   = 5
MIN_FACE_PX = 48
INPUT_SIZE  = 224
LANDMARK_NAMES = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]


def load_image(data: bytes):
    """Decode bytes into (upright RGB image capped at MAX_SIDE, original PIL image)."""
    original = Image.open(io.BytesIO(data))
    original.load()
    img = ImageOps.exif_transpose(original).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img, original


class FaceFinder:
    def __init__(self, score_threshold=0.75):
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), score_threshold, 0.3, 50)

    def detect(self, img: Image.Image):
        """Return faces as dicts with box [x, y, w, h], score and 5 landmarks, largest first."""
        scale = min(1.0, DETECT_SIDE / max(img.size))
        small = img if scale == 1.0 else img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = self.detector.detect(bgr)
        faces = []
        for row in (found if found is not None else []):
            x, y, w, h = (row[:4] / scale).tolist()
            lm = (row[4:14].reshape(5, 2) / scale).tolist()
            faces.append({"box": [x, y, w, h], "score": float(row[14]), "landmarks": lm})
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        return faces


def crop_face(img: Image.Image, face, margin=CROP_MARGIN):
    """Square crop around a face, kept inside the image, resized to 224.

    Returns the crop, the crop box in image coordinates, and the landmarks
    mapped into the 224x224 crop.
    """
    x, y, w, h = face["box"]
    side = min(max(w, h) * margin, img.width, img.height)
    cx, cy = x + w / 2, y + h / 2
    left = min(max(cx - side / 2, 0), img.width - side)
    top  = min(max(cy - side / 2, 0), img.height - side)
    box = (left, top, left + side, top + side)
    crop = img.crop(tuple(round(v) for v in box)).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    k = INPUT_SIZE / side
    landmarks = [[(px - left) * k, (py - top) * k] for px, py in face.get("landmarks", [])]
    return crop, [left, top, side, side], landmarks


def center_crop(img: Image.Image):
    """Fallback when no face is found: the central square of the whole image."""
    side = min(img.size)
    left, top = (img.width - side) / 2, (img.height - side) / 2
    crop = img.crop((round(left), round(top), round(left + side), round(top + side)))
    return crop.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC), [left, top, side, side]
