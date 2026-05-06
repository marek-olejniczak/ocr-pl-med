from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import LoadRequest, PredictRequest, decode_base64_image_to_temp_file, detect_cache_presence
from docker.logging_utils import elapsed_ms, emit_event, get_service_logger, new_request_id, timer_start

app = FastAPI(title="kraken-service")
_STATE: dict | None = None
SERVICE_NAME = "kraken"
logger = get_service_logger("kraken-service")

_PRECISIONS = {
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
}


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


def _resolve_device(device_name: str) -> tuple[str, str | list[int]]:
    normalized = device_name.lower().strip()

    if normalized == "auto":
        return "auto", "auto"

    devices = [entry.strip() for entry in normalized.split(",") if entry.strip()]
    if not devices:
        raise RuntimeError("Nieprawidlowy parametr device: pusty ciag znakow.")

    if devices[0] in {"cpu", "mps"}:
        return devices[0], "auto"

    if any(devices[0].startswith(prefix) for prefix in ("tpu", "cuda", "hpu", "ipu")):
        parsed = []
        for entry in devices:
            parts = entry.split(":", 1)
            if len(parts) != 2:
                raise RuntimeError(
                    "Nieprawidlowy format device. Uzyj np. cpu, auto, cuda:0 lub cuda:0,cuda:1."
                )
            parsed.append((parts[0].strip(), parts[1].strip()))

        if len({kind for kind, _ in parsed}) > 1:
            raise RuntimeError("Mozna uzyc tylko jednego typu urzadzen jednoczesnie.")

        accelerator, _ = parsed[0]
        if accelerator == "cuda":
            accelerator = "gpu"

        try:
            indices = [int(index) for _, index in parsed]
        except ValueError as exc:
            raise RuntimeError(f"Nieprawidlowy indeks urzadzenia w device={device_name}") from exc

        return accelerator, indices

    raise RuntimeError(f"Nieobslugiwane urzadzenie dla Kraken: {device_name}")


def _build_state(raw_options: dict) -> dict:
    try:
        import torch
        from kraken.configs import RecognitionInferenceConfig
        from kraken.containers import BBoxLine, Segmentation
        from kraken.tasks import RecognitionTaskModel
    except Exception as exc:
        raise RuntimeError(
            "Brakuje zaleznosci dla Kraken. Zainstaluj pakiet `kraken` i jego zaleznosci runtime."
        ) from exc

    model_path_raw = str(raw_options.get("model_path", "")).strip()
    if not model_path_raw:
        raise RuntimeError(
            "Brak opcji `model_path`. Podaj sciezke do modelu Kraken przez request options lub --kraken-model-path."
        )

    model_path = Path(model_path_raw).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise RuntimeError(f"Model Kraken nie istnieje lub nie jest plikiem: {model_path}")

    precision = str(raw_options.get("precision", "32-true")).strip()
    if precision not in _PRECISIONS:
        allowed = ", ".join(sorted(_PRECISIONS))
        raise RuntimeError(f"Nieobslugiwane precision={precision}. Dozwolone: {allowed}")

    batch_size = int(raw_options.get("batch_size", 8))
    if batch_size < 1:
        raise RuntimeError("batch_size musi byc >= 1")

    num_line_workers = int(raw_options.get("num_line_workers", 2))
    if num_line_workers < 0:
        raise RuntimeError("num_line_workers musi byc >= 0")

    text_direction = str(raw_options.get("text_direction", "horizontal-tb")).strip()
    if text_direction not in {"horizontal-tb", "vertical-lr", "vertical-rl"}:
        raise RuntimeError(
            "Nieobslugiwany text_direction. Dozwolone: horizontal-tb, vertical-lr, vertical-rl"
        )

    raw_device = str(raw_options.get("device", "cuda:0")).strip()
    accelerator, device = _resolve_device(raw_device)
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano device=cuda, ale CUDA nie jest dostepna w kontenerze.")

    resolved_device = raw_device
    if raw_device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    no_legacy_polygons = bool(raw_options.get("no_legacy_polygons", False))

    model = RecognitionTaskModel.load_model(str(model_path))
    config = RecognitionInferenceConfig(
        accelerator=accelerator,
        device=device,
        precision=precision,
        batch_size=batch_size,
        num_line_workers=num_line_workers,
        text_direction=text_direction,
        no_legacy_polygons=no_legacy_polygons,
    )

    return {
        "model": model,
        "config": config,
        "model_path": str(model_path),
        "raw_device": raw_device,
        "resolved_device": resolved_device,
        "accelerator": accelerator,
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "num_line_workers": num_line_workers,
        "text_direction": text_direction,
        "no_legacy_polygons": no_legacy_polygons,
        "BBoxLine": BBoxLine,
        "Segmentation": Segmentation,
    }


def _state_signature(raw_options: dict) -> tuple:
    return (
        str(raw_options.get("model_path", "")).strip(),
        str(raw_options.get("device", "cuda:0")).strip(),
        str(raw_options.get("precision", "32-true")).strip(),
        int(raw_options.get("batch_size", 8)),
        int(raw_options.get("num_line_workers", 2)),
        str(raw_options.get("text_direction", "horizontal-tb")).strip(),
        bool(raw_options.get("no_legacy_polygons", False)),
    )


def get_state(options: dict) -> dict:
    global _STATE
    current_signature = _state_signature(options)

    if _STATE is not None and _STATE.get("signature") == current_signature:
        return _STATE

    event_name = "state_init_started" if _STATE is None else "state_reinit_started"
    init_started_at = timer_start()
    emit_event(
        logger,
        logging.INFO,
        event_name,
        service=SERVICE_NAME,
        options_keys=sorted(options.keys()),
    )

    try:
        state = _build_state(options)
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

    state["signature"] = current_signature
    _STATE = state

    emit_event(
        logger,
        logging.INFO,
        "state_init_succeeded",
        service=SERVICE_NAME,
        duration_ms=elapsed_ms(init_started_at),
        model_path=state["model_path"],
        raw_device=state["raw_device"],
        resolved_device=state["resolved_device"],
        accelerator=state["accelerator"],
        fabric_device=state["device"],
        precision=state["precision"],
        batch_size=state["batch_size"],
        num_line_workers=state["num_line_workers"],
    )
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "kraken"}


@app.post("/load")
def load(req: LoadRequest) -> dict:
    request_id = new_request_id()
    started_at = timer_start()
    options = req.options or {}
    model_path = str(options.get("model_path", "")).strip() or None
    cache_present = detect_cache_presence(model_path)
    signature = _state_signature(options)

    emit_event(
        logger,
        logging.INFO,
        "load_request_started",
        request_id=request_id,
        service=SERVICE_NAME,
        options_keys=sorted(options.keys()),
        model_path=model_path,
        cache_present=cache_present,
    )

    if _STATE is not None and _STATE.get("signature") == signature:
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
            width, height = image.size

            bbox_line = state["BBoxLine"](id="line_0", bbox=(0, 0, width, height), text_direction="horizontal-lr")
            segmentation = state["Segmentation"](
                type="bbox",
                imagename=temp_path,
                text_direction="horizontal-lr",
                script_detection=False,
                lines=[bbox_line],
            )

            records = list(state["model"].predict(im=image, segmentation=segmentation, config=state["config"]))
            text = "\n".join(record.prediction for record in records).strip()

        success = True
        emit_event(
            logger,
            logging.INFO,
            "predict_inference_succeeded",
            request_id=request_id,
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(inference_started_at),
            text_len=len(text),
            lines_count=len(records),
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
