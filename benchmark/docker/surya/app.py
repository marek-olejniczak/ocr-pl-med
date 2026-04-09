from __future__ import annotations

import importlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import PredictRequest, decode_base64_image_to_temp_file

app = FastAPI(title="surya-service")
_STATE: dict | None = None


def _build_surya_state(options: dict) -> dict:
    # Surya reads settings from env at import/init time.
    os.environ.setdefault("MODEL_CACHE_DIR", options.get("cache_dir", "modele/cache/surya"))
    os.environ.setdefault("TORCH_DEVICE", options.get("device", "cpu"))
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(int(options.get("batch_size", 32))))

    try:
        foundation_module = importlib.import_module("surya.foundation")
        recognition_module = importlib.import_module("surya.recognition")
    except Exception as exc:
        raise RuntimeError(
            "Brakuje zaleznosci dla Surya OCR. Zainstaluj: surya-ocr, torch, torchvision "
            "oraz runtime libs (libgl1, libglib2.0-0, libgomp1). "
            f"Szczegoly: {exc}"
        ) from exc

    FoundationPredictor = getattr(foundation_module, "FoundationPredictor")
    RecognitionPredictor = getattr(recognition_module, "RecognitionPredictor")

    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    return {
        "recognition_predictor": recognition_predictor,
        "task_name": str(options.get("task_name", "ocr_without_boxes")),
        "disable_math": bool(options.get("disable_math", True)),
        "batch_size": int(options.get("batch_size", 32)),
    }


def get_state(options: dict) -> dict:
    global _STATE
    if _STATE is None:
        _STATE = _build_surya_state(options)
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "surya"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    temp_path = None
    try:
        state = get_state(req.options)
        temp_path = decode_base64_image_to_temp_file(req.image_base64)

        with Image.open(temp_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            # Input is an already-cropped text line, so we provide one bbox spanning the full image.
            bboxes = [[[0, 0, width, height]]]

            recognition_predictor = state["recognition_predictor"]
            results = recognition_predictor(
                [image],
                task_names=[state["task_name"]],
                bboxes=bboxes,
                math_mode=not state["disable_math"],
                recognition_batch_size=state["batch_size"],
                sort_lines=False,
                return_words=False,
            )

        if not results:
            return {"text": ""}

        text_lines = getattr(results[0], "text_lines", []) or []
        parts = [str(getattr(line, "text", "")).strip() for line in text_lines]
        text = " ".join(part for part in parts if part)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
