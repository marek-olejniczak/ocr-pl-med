from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.calamari_wrapper import CalamariWrapper

app = FastAPI(title="calamari-service")
_MODEL: CalamariWrapper | None = None


def get_model(options: dict) -> CalamariWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = CalamariWrapper(
            model=options.get("model", "idiotikon"),
            batch_size=int(options.get("batch_size", 8)),
            cache_dir=options.get("cache_dir", "modele/cache/calamari"),
            local_files_only=bool(options.get("local_files_only", False)),
            checkpoints=options.get("checkpoints"),
            device=options.get("device", "auto"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "calamari"}


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
