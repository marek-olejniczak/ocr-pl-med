"""Preprocessing adapters for the pipeline, both PIL -> PIL.

Two stages with different scopes:

- geometric (deskew): GLOBAL. Its output is the base image for everything
  downstream - detection, OCR crops, display. The benchmark skips this stage
  because ground-truth boxes must stay valid; the app has no such constraint
  and real uploads are skewed.
- photometric (CLAHE, grid removal, ink separation): detector input ONLY.
  The detector's training data is dominated by black ink, so e.g. blue
  handwriting lands far from the training distribution without it. OCR crops
  are cut from the base image; the coordinates transfer because this stage is
  geometry-preserving (same contract as the benchmark's prep variant,
  enforced by a size check).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from preprocessing import DocumentPreprocessor

# keep in sync with line_benchmark/data_prep/preprocess_dataset.py
PHOTOMETRIC_CONFIG = {"deskew": False, "border_px": 0, "alpha_crop": False}


def _to_bgr(image):
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(bgr):
    if bgr.ndim == 3:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(bgr).convert("RGB")


def geometric(image):
    """Deskew the page. PIL.Image -> PIL.Image, same canvas size."""
    return _to_pil(DocumentPreprocessor().geometric(_to_bgr(image)))


def photometric(image):
    """PIL.Image -> PIL.Image of identical size, pixel-value ops only."""
    out = DocumentPreprocessor(PHOTOMETRIC_CONFIG).photometric(_to_bgr(image))
    result = _to_pil(out)
    if result.size != image.size:
        raise RuntimeError("photometric stage changed geometry: "
                           f"{image.size} -> {result.size}")
    return result
