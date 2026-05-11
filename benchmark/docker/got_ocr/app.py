from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import (
    LoadRequest,
    PredictRequest,
    PredictBatchRequest,
    decode_base64_image_to_temp_file,
    detect_cache_presence,
)
from docker.logging_utils import elapsed_ms, emit_event, get_service_logger, new_request_id, timer_start

app = FastAPI(title="got-ocr-service")
_STATE: dict | None = None
SERVICE_NAME = "got_ocr"
logger = get_service_logger("got-ocr-service")


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


def _resolve_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.lower().strip()
    if normalized == "float16":
        return torch_module.float16
    if normalized == "bfloat16":
        return torch_module.bfloat16
    if normalized == "float32":
        return torch_module.float32

    if torch_module.cuda.is_available():
        return torch_module.bfloat16
    return torch_module.float32


def _build_state(options: dict) -> dict:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Brakuje zaleznosci dla GOT-OCR 2.0. Zainstaluj: torch, torchvision, transformers oraz runtime libs "
            "(libgomp1, libgl1, libglib2.0-0). "
            f"Szczegoly: {exc}"
        ) from exc

    device_option = str(options.get("device", "auto")).lower().strip()
    if device_option == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_option

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano device=cuda, ale CUDA nie jest dostepna w kontenerze.")

    model_id = str(options.get("model_id", "ucaslcl/got-ocr-2.0"))
    cache_dir = str(options.get("cache_dir", "modele/cache/got_ocr"))
    dtype = _resolve_dtype(torch, str(options.get("dtype", "auto")))
    trust_remote_code = bool(options.get("trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        device_map=device,
        trust_remote_code=trust_remote_code,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    model = model.eval()

    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }


def get_state(options: dict) -> dict:
    global _STATE
    if _STATE is None:
        init_started_at = timer_start()
        emit_event(
            logger,
            logging.INFO,
            "state_init_started",
            service=SERVICE_NAME,
            options_keys=sorted(options.keys()),
        )
        try:
            _STATE = _build_state(options)
        except Exception as exc:
            emit_event(
                logger,
                logging.ERROR,
                "state_init_failed",
                service=SERVICE_NAME,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            raise

        emit_event(
            logger,
            logging.INFO,
            "state_init_succeeded",
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(init_started_at),
            resolved_device=_STATE.get("device"),
        )
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "got-ocr"}


@app.post("/load")
def load(req: LoadRequest) -> dict:
    request_id = new_request_id()
    started_at = timer_start()
    options = req.options or {}
    cache_dir = options.get("cache_dir", "modele/cache/got_ocr")
    cache_present = detect_cache_presence(cache_dir)

    emit_event(
        logger,
        logging.INFO,
        "load_request_started",
        request_id=request_id,
        service=SERVICE_NAME,
        options_keys=sorted(options.keys()),
        cache_dir=cache_dir,
        cache_present=cache_present,
    )

    if _STATE is not None:
        emit_event(
            logger,
            logging.INFO,
            "load_skipped",
            request_id=request_id,
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(started_at),
        )
        return {
            "status": "already_loaded",
            "service": SERVICE_NAME,
            "cache_present": cache_present,
        }

    try:
        get_state(options)
    except Exception as exc:
        emit_event(
            logger,
            logging.ERROR,
            "load_request_failed",
            request_id=request_id,
            service=SERVICE_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = elapsed_ms(started_at)
    emit_event(
        logger,
        logging.INFO,
        "load_request_succeeded",
        request_id=request_id,
        service=SERVICE_NAME,
        duration_ms=duration_ms,
    )
    return {
        "status": "loaded",
        "service": SERVICE_NAME,
        "duration_ms": duration_ms,
        "cache_present": cache_present,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    request_id = new_request_id()
    request_started_at = timer_start()
    temp_path = None
    success = False

    emit_event(
        logger,
        logging.INFO,
        "predict_request_received",
        request_id=request_id,
        service=SERVICE_NAME,
        image_base64_len=len(req.image_base64),
        options_keys=sorted(req.options.keys()),
    )

    try:
        state = get_state(req.options)
        temp_path = decode_base64_image_to_temp_file(
            req.image_base64,
            logger=logger,
            request_id=request_id,
        )

        inference_started_at = timer_start()
        emit_event(
            logger,
            logging.INFO,
            "predict_inference_started",
            request_id=request_id,
            service=SERVICE_NAME,
        )

        with Image.open(temp_path) as image:
            image = image.convert("RGB")
            tokenizer = state["tokenizer"]
            model = state["model"]
            torch = state["torch"]

            with torch.inference_mode():
                text = model.chat(
                    tokenizer,
                    temp_path,
                    ocr_type="ocr",
                )

        success = True
        emit_event(
            logger,
            logging.INFO,
            "predict_inference_succeeded",
            request_id=request_id,
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(inference_started_at),
            text_len=len(text),
        )
        return {"text": text}
    except Exception as exc:
        emit_event(
            logger,
            logging.ERROR,
            "predict_request_failed",
            request_id=request_id,
            service=SERVICE_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
                emit_event(
                    logger,
                    logging.INFO,
                    "temp_file_cleanup_succeeded",
                    request_id=request_id,
                    service=SERVICE_NAME,
                )
            except Exception as cleanup_exc:
                emit_event(
                    logger,
                    logging.WARNING,
                    "temp_file_cleanup_failed",
                    request_id=request_id,
                    service=SERVICE_NAME,
                    error_type=type(cleanup_exc).__name__,
                    error=str(cleanup_exc),
                    exc_info=True,
                )

        emit_event(
            logger,
            logging.INFO,
            "predict_request_completed",
            request_id=request_id,
            service=SERVICE_NAME,
            success=success,
            duration_ms=elapsed_ms(request_started_at),
        )


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest) -> dict:
    request_id = new_request_id()
    request_started_at = timer_start()
    temp_paths = []
    success = False

    emit_event(
        logger,
        logging.INFO,
        "predict_batch_request_received",
        request_id=request_id,
        service=SERVICE_NAME,
        num_images=len(req.images_base64),
        options_keys=sorted(req.options.keys()),
    )

    if not req.images_base64:
        return {"texts": []}

    try:
        state = get_state(req.options)

        for idx, img_b64 in enumerate(req.images_base64):
            path = decode_base64_image_to_temp_file(
                img_b64,
                logger=logger,
                request_id=f"{request_id}_{idx}",
            )
            temp_paths.append(path)

        inference_started_at = timer_start()
        emit_event(
            logger,
            logging.INFO,
            "predict_batch_inference_started",
            request_id=request_id,
            service=SERVICE_NAME,
            batch_size=len(temp_paths),
        )

        tokenizer = state["tokenizer"]
        model = state["model"]
        torch = state["torch"]

        texts = []
        with torch.inference_mode():
            for path in temp_paths:
                with Image.open(path) as image:
                    image.convert("RGB")
                    text = model.chat(
                        tokenizer,
                        path,
                        ocr_type="ocr",
                    )
                texts.append(str(text).strip())

        success = True
        emit_event(
            logger,
            logging.INFO,
            "predict_batch_inference_succeeded",
            request_id=request_id,
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(inference_started_at),
            num_results=len(texts),
        )
        return {"texts": texts}
    except Exception as exc:
        emit_event(
            logger,
            logging.ERROR,
            "predict_batch_request_failed",
            request_id=request_id,
            service=SERVICE_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for path in temp_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as cleanup_exc:
                emit_event(
                    logger,
                    logging.WARNING,
                    "temp_file_cleanup_failed",
                    request_id=request_id,
                    service=SERVICE_NAME,
                    error_type=type(cleanup_exc).__name__,
                    error=str(cleanup_exc),
                    file_path=path,
                )

        emit_event(
            logger,
            logging.INFO,
            "predict_batch_request_completed",
            request_id=request_id,
            service=SERVICE_NAME,
            success=success,
            duration_ms=elapsed_ms(request_started_at),
        )
