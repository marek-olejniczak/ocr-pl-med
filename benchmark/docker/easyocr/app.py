from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.easyocr_wrapper import EasyOCRWrapper

app = FastAPI(title="easyocr-service")
_MODEL: EasyOCRWrapper | None = None


def get_model(options: dict) -> EasyOCRWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = EasyOCRWrapper(
            langs=options.get("langs") or ["pl", "en"],
            device=options.get("device", "auto"),
            batch_size=int(options.get("batch_size", 8)),
            model_storage_dir=options.get("model_storage_dir", "modele/cache/easyocr"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "easyocr"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    temp_path = None
    try:
        model = get_model(req.options)
        temp_path = decode_base64_image_to_temp_file(req.image_base64)
        text = model.predict(temp_path)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
