from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import PredictRequest, decode_base64_image_to_temp_file
from docker.logging_utils import elapsed_ms, emit_event, get_service_logger, new_request_id, timer_start

app = FastAPI(title="surya-service")
_STATE: dict | None = None
SERVICE_NAME = "surya"
logger = get_service_logger("surya-service")


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


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
        init_started_at = timer_start()
        emit_event(
            logger,
            logging.INFO,
            "state_init_started",
            service=SERVICE_NAME,
            options_keys=sorted(options.keys()),
        )
        try:
            _STATE = _build_surya_state(options)
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
            task_name=_STATE.get("task_name"),
            batch_size=_STATE.get("batch_size"),
            disable_math=_STATE.get("disable_math"),
            resolved_device=os.environ.get("TORCH_DEVICE"),
        )
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "surya"}


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
            success = True
            emit_event(
                logger,
                logging.INFO,
                "predict_inference_succeeded",
                request_id=request_id,
                service=SERVICE_NAME,
                duration_ms=elapsed_ms(inference_started_at),
                text_len=0,
            )
            return {"text": ""}

        text_lines = getattr(results[0], "text_lines", []) or []
        parts = [str(getattr(line, "text", "")).strip() for line in text_lines]
        text = " ".join(part for part in parts if part)
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
