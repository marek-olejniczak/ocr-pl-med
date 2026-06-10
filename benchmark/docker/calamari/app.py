from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from docker.common import (
    LoadRequest,
    PredictRequest,
    PredictBatchRequest,
    decode_base64_image_to_temp_file,
    detect_cache_presence,
)
from docker.logging_utils import elapsed_ms, emit_event, get_service_logger, new_request_id, timer_start
from modele.calamari_wrapper import CalamariWrapper

app = FastAPI(title="calamari-service")
_MODEL: CalamariWrapper | None = None
SERVICE_NAME = "calamari"
logger = get_service_logger("calamari-service")


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


def get_model(options: dict) -> CalamariWrapper:
    global _MODEL
    if _MODEL is None:
        init_started_at = timer_start()
        emit_event(
            logger,
            logging.INFO,
            "model_init_started",
            service=SERVICE_NAME,
            options_keys=sorted(options.keys()),
        )
        try:
            _MODEL = CalamariWrapper(
                model=options.get("model", "idiotikon"),
                batch_size=int(options.get("batch_size", 8)),
                cache_dir=options.get("cache_dir", "modele/cache/calamari"),
                local_files_only=bool(options.get("local_files_only", False)),
                checkpoints=options.get("checkpoints"),
                device=options.get("device", "auto"),
            )
        except Exception as exc:
            emit_event(
                logger,
                logging.ERROR,
                "model_init_failed",
                service=SERVICE_NAME,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            raise

        emit_event(
            logger,
            logging.INFO,
            "model_init_succeeded",
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(init_started_at),
        )
    return _MODEL


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "calamari"}


@app.post("/load")
def load(req: LoadRequest) -> dict:
    request_id = new_request_id()
    started_at = timer_start()
    options = req.options or {}
    cache_dir = options.get("cache_dir", "modele/cache/calamari")
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

    if _MODEL is not None:
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
        get_model(options)
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
        model = get_model(req.options)
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
        text = model.predict(temp_path)
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
        model = get_model(req.options)

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

        if hasattr(model, "predict_batch"):
            batch_texts = model.predict_batch(temp_paths)
            texts = [str(text).strip() for text in batch_texts]
        else:
            texts = [str(model.predict(path)).strip() for path in temp_paths]

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
