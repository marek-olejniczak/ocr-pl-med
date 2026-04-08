from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.paddleocr_wrapper import PaddleOCRWrapper

app = FastAPI(title="paddleocr-service")
_MODEL: PaddleOCRWrapper | None = None


def get_model(options: dict) -> PaddleOCRWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = PaddleOCRWrapper(
            rec_model_name=options.get("rec_model_name", "PP-OCRv4_mobile_rec"),
            lang=options.get("lang", "pl"),
            device=options.get("device", "auto"),
            use_angle_cls=bool(options.get("use_angle_cls", False)),
            rec_batch_size=int(options.get("rec_batch_size", 8)),
            cache_dir=options.get("cache_dir", "modele/cache/paddlex"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "paddleocr"}


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
