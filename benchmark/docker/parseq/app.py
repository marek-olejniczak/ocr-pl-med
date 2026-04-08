from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.parseq_wrapper import PARSeqWrapper

app = FastAPI(title="parseq-service")
_MODEL: PARSeqWrapper | None = None


def get_model(options: dict) -> PARSeqWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = PARSeqWrapper(
            device=options.get("device", "auto"),
            batch_size=int(options.get("batch_size", 8)),
            cache_dir=options.get("cache_dir", "modele/cache/parseq"),
            input_size=options.get("input_size", "32x128"),
            use_amp=bool(options.get("use_amp", False)),
            language=options.get("language", "pl"),
            model_id=options.get("model_id"),
            local_files_only=bool(options.get("local_files_only", False)),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "parseq"}


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
