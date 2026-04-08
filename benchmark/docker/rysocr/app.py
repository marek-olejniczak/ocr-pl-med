from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from modele.rysocr_wrapper import RysOCRWrapper

app = FastAPI(title="rysocr-service")
_MODEL: RysOCRWrapper | None = None


def get_model(options: dict) -> RysOCRWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = RysOCRWrapper(
            adapter_model_id=options.get("adapter_model_id", "kacperwikiel/RysOCR"),
            base_model_id=options.get("base_model_id", "PaddlePaddle/PaddleOCR-VL"),
            prompt=options.get("prompt", "Transcribe the text exactly."),
            max_new_tokens=int(options.get("max_new_tokens", 256)),
            device=options.get("device"),
            local_files_only=bool(options.get("local_files_only", False)),
            batch_size=int(options.get("batch_size", 2)),
            use_amp=bool(options.get("use_amp", False)),
            cache_dir=options.get("cache_dir", "modele/cache/rysocr"),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rysocr"}


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
