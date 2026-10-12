"""Preprocessing, tiling and ONNX inference for the vessel segmentation models.

Mirrors the training pipeline: a 1024x1024 patch is resized to 256x256 with
area interpolation, scaled to [0, 1] and normalised with the ImageNet
mean/std. The class map is the argmax over the 5 output channels.
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(APP_DIR, "..", "models"))

PATCH_SIZE = 256          # network input size
NATIVE_TILE = 1024        # source pixels covered by one training patch
SINGLE_TILE_LIMIT = 1536  # "auto" treats anything up to this as one patch
MAX_SIDE = 8192
MAX_TILES = 64
BATCH_SIZE = 8
PREVIEW_SIDE = 1400

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with open(os.path.join(APP_DIR, "models.json"), encoding="utf-8") as f:
    REGISTRY = json.load(f)
CLASSES = REGISTRY["classes"]
MODELS = {m["id"]: m for m in REGISTRY["models"]}


class InferenceError(ValueError):
    """Raised for problems the user can fix (bad file, image too large...)."""


_sessions = {}
_sessions_lock = threading.Lock()


def model_available(model_id):
    return os.path.exists(os.path.join(MODELS_DIR, MODELS[model_id]["file"]))


def get_session(model_id):
    with _sessions_lock:
        if model_id not in _sessions:
            path = os.path.join(MODELS_DIR, MODELS[model_id]["file"])
            if not os.path.exists(path):
                raise InferenceError("Ce modèle n'est pas installé sur le serveur.")
            _sessions[model_id] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return _sessions[model_id]


def decode_image(data):
    """Bytes -> RGB uint8 array. Handles TIFF/PNG/JPEG/WebP/BMP, 16-bit and alpha."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise InferenceError("Fichier illisible. Formats acceptés : TIFF, PNG, JPEG, WebP, BMP.")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        raise InferenceError(f"Image trop grande ({w}×{h} px). Maximum : {MAX_SIDE} px de côté.")
    return img


def _preview_size(h, w):
    scale = min(1.0, PREVIEW_SIDE / max(h, w))
    return max(1, round(h * scale)), max(1, round(w * scale))


def encode_preview(rgb):
    """Downscaled JPEG of the input, so the browser can show TIFFs too."""
    h, w = rgb.shape[:2]
    ph, pw = _preview_size(h, w)
    small = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA) if (ph, pw) != (h, w) else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise InferenceError("Impossible de générer l'aperçu.")
    return buf.tobytes()


def _tile_size(h, w, mode):
    if mode == "whole" or (mode == "auto" and max(h, w) <= SINGLE_TILE_LIMIT):
        return max(h, w)
    return NATIVE_TILE


def segment(rgb, model_id, mode="auto"):
    """Returns (class map at preview size, info dict)."""
    if model_id not in MODELS:
        raise InferenceError("Modèle inconnu.")
    if mode not in ("auto", "whole", "tiles"):
        raise InferenceError("Mode inconnu.")

    h, w = rgb.shape[:2]
    tile = _tile_size(h, w, mode)
    rows, cols = -(-h // tile), -(-w // tile)
    if rows * cols > MAX_TILES:
        raise InferenceError(f"Image trop grande pour le découpage en tuiles ({rows * cols} tuiles, maximum {MAX_TILES}).")

    # Pad by reflection up to a whole number of square tiles, so every tile keeps
    # the aspect ratio the networks were trained on.
    padded = cv2.copyMakeBorder(rgb, 0, rows * tile - h, 0, cols * tile - w, cv2.BORDER_REFLECT)

    batch = np.empty((rows * cols, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = padded[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
            patch = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            batch[r * cols + c] = patch.transpose(2, 0, 1)

    session = get_session(model_id)
    start = time.perf_counter()
    preds = []
    for i in range(0, len(batch), BATCH_SIZE):
        logits = session.run(None, {"image": batch[i:i + BATCH_SIZE]})[0]
        preds.append(logits.argmax(axis=1).astype(np.uint8))
    elapsed_ms = (time.perf_counter() - start) * 1000
    preds = np.concatenate(preds)

    full = preds.reshape(rows, cols, PATCH_SIZE, PATCH_SIZE).transpose(0, 2, 1, 3)
    full = full.reshape(rows * PATCH_SIZE, cols * PATCH_SIZE)
    # Drop the padded margin, then bring the map to the preview size.
    valid_h = max(1, round(h * PATCH_SIZE / tile))
    valid_w = max(1, round(w * PATCH_SIZE / tile))
    full = full[:valid_h, :valid_w]

    counts = np.bincount(full.ravel(), minlength=len(CLASSES))
    total = int(counts.sum())
    ph, pw = _preview_size(h, w)
    class_map = cv2.resize(full, (pw, ph), interpolation=cv2.INTER_NEAREST)

    info = {
        "model": model_id,
        "width": w,
        "height": h,
        "tiles": rows * cols,
        "tile_size": tile,
        "elapsed_ms": round(elapsed_ms),
        "classes": [
            {"id": c["id"], "pixels": int(counts[c["id"]]), "percent": round(100 * counts[c["id"]] / total, 2)}
            for c in CLASSES
        ],
    }
    return class_map, info


def encode_class_map(class_map):
    """Lossless grayscale PNG; class ids are spread out (x50) so they survive any decoder."""
    ok, buf = cv2.imencode(".png", class_map * 50)
    if not ok:
        raise InferenceError("Impossible d'encoder le masque.")
    return buf.tobytes()
