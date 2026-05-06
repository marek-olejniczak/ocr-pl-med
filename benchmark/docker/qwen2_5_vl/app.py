from __future__ import annotations

import logging
import importlib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import LoadRequest, PredictRequest, decode_base64_image_to_temp_file, detect_cache_presence
from docker.logging_utils import elapsed_ms, emit_event, get_service_logger, new_request_id, timer_start

app = FastAPI(title="qwen2_5_vl-service")
_STATE: dict | None = None
SERVICE_NAME = "qwen2_5_vl"
logger = get_service_logger("qwen2_5_vl-service")

_DEFAULT_PROMPT = (
    "Odczytaj dokladnie caly tekst z obrazu w jezyku polskim. "
    "Zachowaj oryginalne polskie znaki diakrytyczne oraz interpunkcje. "
    "Zwroc tylko sam tekst bez komentarzy."
)


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


def _resolve_device(torch_module, device_name: str) -> str:
    normalized = device_name.lower().strip()

    if normalized == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "Qwen2.5-VL jest skonfigurowany jako GPU-first i wymaga CUDA. "
            "Brak dostepnej karty GPU w kontenerze. "
            "Aby wymusic wolniejszy fallback, ustaw opcje requestu: device=cpu."
        )

    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Wybrano device=cuda, ale CUDA nie jest dostepna w kontenerze.")
        return "cuda"

    if normalized == "cpu":
        return "cpu"

    raise RuntimeError(f"Nieobslugiwane urzadzenie dla Qwen2.5-VL: {device_name}")


def _resolve_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.lower().strip()

    if normalized == "float16":
        return torch_module.float16
    if normalized == "bfloat16":
        return torch_module.bfloat16
    if normalized == "float32":
        return torch_module.float32

    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16

    return torch_module.float32


def _build_state(options: dict) -> dict:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Brakuje zaleznosci dla Qwen2.5-VL. Zainstaluj: torch, torchvision, transformers>=4.57.0, "
            "accelerate oraz qwen-vl-utils."
        ) from exc

    process_vision_info = getattr(importlib.import_module("qwen_vl_utils"), "process_vision_info")

    model_id = str(options.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"))
    cache_dir = str(options.get("cache_dir", "modele/cache/qwen2_5_vl"))
    max_new_tokens = int(options.get("max_new_tokens", 256))
    prompt = str(options.get("prompt", _DEFAULT_PROMPT))

    device = _resolve_device(torch, str(options.get("device", "auto")))
    dtype = _resolve_dtype(torch, str(options.get("dtype", "auto")))

    if device == "cpu" and dtype != torch.float32:
        emit_event(
            logger,
            logging.WARNING,
            "dtype_adjusted_for_cpu",
            service=SERVICE_NAME,
            requested_dtype=str(options.get("dtype", "auto")),
            resolved_dtype="float32",
        )
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    load_started_at = timer_start()
    if device == "cuda":
        # Na kartach z mala iloscia VRAM pozwalamy Accelerate rozdzielic warstwy (GPU/CPU),
        # aby ograniczyc ryzyko OOM przy starcie modelu.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = model.eval()
        model_device = str(getattr(model, "device", "cuda"))
        emit_event(
            logger,
            logging.INFO,
            "model_loaded_with_device_map_auto",
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(load_started_at),
            model_device=model_device,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device).eval()
        model_device = device
        emit_event(
            logger,
            logging.INFO,
            "model_loaded_on_cpu",
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(load_started_at),
        )

    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "process_vision_info": process_vision_info,
        "device": model_device,
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "model_id": model_id,
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
            model_id=_STATE.get("model_id"),
            resolved_device=_STATE.get("device"),
            max_new_tokens=_STATE.get("max_new_tokens"),
        )
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "qwen2_5_vl"}


@app.post("/load")
def load(req: LoadRequest) -> dict:
    request_id = new_request_id()
    started_at = timer_start()
    options = req.options or {}
    cache_dir = options.get("cache_dir", "modele/cache/qwen2_5_vl")
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

        prompt = str(req.options.get("prompt", state["prompt"]))
        image_uri = f"file://{temp_path}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        processor = state["processor"]
        model = state["model"]
        torch = state["torch"]
        process_vision_info = state["process_vision_info"]

        with Image.open(temp_path) as image:
            image = image.convert("RGB")

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(state["device"]) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=state["max_new_tokens"],
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        result_text = str(outputs[0]).strip() if outputs else ""
        success = True

        emit_event(
            logger,
            logging.INFO,
            "predict_inference_succeeded",
            request_id=request_id,
            service=SERVICE_NAME,
            duration_ms=elapsed_ms(inference_started_at),
            text_len=len(result_text),
        )
        return {"text": result_text}
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
