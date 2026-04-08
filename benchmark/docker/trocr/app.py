from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.trocr_wrapper import TrOCRWrapper

app = FastAPI(title="trocr-service")
_MODEL: TrOCRWrapper | None = None


def get_model(options: dict) -> TrOCRWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = TrOCRWrapper(
            model_id=options.get("model_id", "microsoft/trocr-small-handwritten"),
            max_new_tokens=int(options.get("max_new_tokens", 128)),
            device=options.get("device"),
            local_files_only=bool(options.get("local_files_only", False)),
            batch_size=int(options.get("batch_size", 4)),
            use_amp=bool(options.get("use_amp", False)),
            cache_dir=options.get("cache_dir", "modele/cache/trocr"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "trocr"}


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
