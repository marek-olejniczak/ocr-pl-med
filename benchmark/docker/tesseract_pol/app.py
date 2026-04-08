from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.tesseract_pol_wrapper import TesseractPolWrapper

app = FastAPI(title="tesseract-pol-service")
_MODEL: TesseractPolWrapper | None = None


def get_model(options: dict) -> TesseractPolWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = TesseractPolWrapper(
            language=options.get("language", "pol"),
            psm=int(options.get("psm", 7)),
            oem=int(options.get("oem", 1)),
            tesseract_cmd=options.get("tesseract_cmd"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "tesseract_pol"}


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
