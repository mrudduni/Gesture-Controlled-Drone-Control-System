"""Shared settings and hand-cropping helpers.

Data collection, dataset preparation, training and the drone simulation all
import from here, so a hand is cropped exactly the same way everywhere.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# Your own webcam captures: data/<class>/*.jpg
OWN_DATA_DIR = ROOT / "data"
# Hand crops cut out of the HaGRID sample by prepare_hagrid.py
HAGRID_CROPS_DIR = ROOT / "datasets" / "hagrid_crops"
MODEL_DIR = ROOT / "models"
# Two interchangeable architectures (train_cnn.py --arch ...):
#   cnn          small CNN trained from scratch
#   mobilenetv2  MobileNetV2 pretrained on ImageNet, fine-tuned
ARCHES = {
    "cnn": "gesture_cnn_rgb",
    "mobilenetv2": "gesture_mobilenetv2",
}
DEFAULT_ARCH = "mobilenetv2"  # most accurate overall (see compare_models.py)


def model_paths(arch=DEFAULT_ARCH):
    """(model .keras path, metadata .json path, training-curves .png path) for an architecture."""
    stem = ARCHES[arch]
    return MODEL_DIR / f"{stem}.keras", MODEL_DIR / f"{stem}.json", MODEL_DIR / f"{stem}_curves.png"

# Alphabetical, which is also the order Keras used for the original model.
CLASS_NAMES = ["down", "hover", "land", "left", "right", "up"]

# Model input size (square, RGB).
IMG_SIZE = 128
# Size crops are stored at on disk; a little larger than IMG_SIZE so training
# can zoom/shift without upsampling.
CROP_STORE_SIZE = 160
# Padding added around the tight hand box, as a fraction of its longer side.
PAD_RATIO = 0.25


def landmarks_to_box(hand_landmarks, frame_w, frame_h):
    """Tight pixel box (x0, y0, x1, y1) around MediaPipe hand landmarks."""
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
    return min(xs), min(ys), max(xs), max(ys)


def square_box(x0, y0, x1, y1, pad_ratio=PAD_RATIO):
    """Square box centred on the given box, padded by pad_ratio on each side.

    The result may extend past the image; crop_square pads with black there,
    so the hand is never squashed or shifted off-centre.
    """
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad_ratio)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2
    return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))


def crop_square(image, box):
    """Crop box from image, padding with black wherever the box leaves the image."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    crop = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return None
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    return crop


def hand_crop_from_landmarks(frame_bgr, hand_landmarks):
    """Square hand crop (BGR) plus the box used, or (None, None)."""
    h, w = frame_bgr.shape[:2]
    box = square_box(*landmarks_to_box(hand_landmarks, w, h))
    return crop_square(frame_bgr, box), box


def to_model_input(crop_bgr, img_size=IMG_SIZE):
    """BGR crop -> float32 RGB batch of shape (1, img_size, img_size, 3), values 0-255.

    The model rescales to 0-1 itself, so no normalisation happens here.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32)[None, ...]


def load_model_meta(meta_path):
    """Class names and input size saved next to the trained model."""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta["class_names"], meta["img_size"]
    return CLASS_NAMES, IMG_SIZE
