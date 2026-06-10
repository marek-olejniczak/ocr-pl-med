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

app = FastAPI(title="glm-4v-service")
_STATE: dict | None = None
SERVICE_NAME = "glm_4v"
logger = get_service_logger("glm-4v-service")

_DEFAULT_PROMPT = (
    "Odczytaj dokladnie caly tekst z obrazu w jezyku polskim. "
    "Zachowaj oryginalne polskie znaki diakrytyczne oraz interpunkcje. "
    "Zwroc tylko sam tekst bez komentarzy."
)


@app.on_event("startup")
def on_startup() -> None:
    emit_event(logger, logging.INFO, "service_startup", service=SERVICE_NAME, title=app.title)


def _resolve_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _resolve_device(torch_module, device_name: str) -> str:
    normalized = device_name.lower().strip()

    if normalized == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "GLM-4V jest skonfigurowany jako GPU-first i wymaga CUDA. "
            "Brak dostepnej karty GPU w kontenerze. "
            "Aby wymusic wolniejszy fallback, ustaw opcje requestu: device=cpu."
        )

    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Wybrano device=cuda, ale CUDA nie jest dostepna w kontenerze.")
        return "cuda"

    if normalized == "cpu":
        return "cpu"

    raise RuntimeError(f"Nieobslugiwane urzadzenie dla GLM-4V: {device_name}")


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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    except Exception as exc:
        raise RuntimeError(
            "Brakuje zaleznosci dla GLM-4V. Zainstaluj: torch>=2.6.0, torchvision>=0.21.0, "
            "transformers>=4.51.3, accelerate, sentencepiece, timm, tiktoken, einops."
        ) from exc

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def _all_tied_weights_keys(self):
            return getattr(self, "_tied_weights_keys", [])

        PreTrainedModel.all_tied_weights_keys = property(_all_tied_weights_keys)

    device = _resolve_device(torch, str(options.get("device", "auto")))
    model_id = str(options.get("model_id", "zai-org/glm-4v-9b"))
    cache_dir = str(options.get("cache_dir", "modele/cache/glm_4v"))
    max_new_tokens = int(options.get("max_new_tokens", 512))
    prompt = str(options.get("prompt", _DEFAULT_PROMPT))
    local_files_only = _resolve_bool(options.get("local_files_only", False))
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if not hasattr(config, "max_length"):
        inferred_max_length = getattr(config, "seq_length", None) or getattr(
            config, "max_position_embeddings", None
        )
        if inferred_max_length is not None:
            setattr(config, "max_length", inferred_max_length)
            emit_event(
                logger,
                logging.WARNING,
                "config_max_length_inferred",
                service=SERVICE_NAME,
                max_length=inferred_max_length,
            )

    if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
        setattr(config, "num_hidden_layers", getattr(config, "num_layers"))
        emit_event(
            logger,
            logging.WARNING,
            "config_num_hidden_layers_inferred",
            service=SERVICE_NAME,
            num_hidden_layers=getattr(config, "num_hidden_layers"),
        )

    if not hasattr(config, "num_key_value_heads") and hasattr(config, "multi_query_group_num"):
        setattr(config, "num_key_value_heads", getattr(config, "multi_query_group_num"))
        emit_event(
            logger,
            logging.WARNING,
            "config_num_key_value_heads_inferred",
            service=SERVICE_NAME,
            num_key_value_heads=getattr(config, "num_key_value_heads"),
        )

    load_started_at = timer_start()
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            local_files_only=local_files_only,
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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
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
        "tokenizer": tokenizer,
        "model": model,
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
    return {"status": "ok", "service": "glm_4v"}


@app.post("/load")
def load(req: LoadRequest) -> dict:
    request_id = new_request_id()
    started_at = timer_start()
    options = req.options or {}
    cache_dir = options.get("cache_dir", "modele/cache/glm_4v")
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

        with Image.open(temp_path) as image:
            image = image.convert("RGB")
            messages = [{"role": "user", "image": image, "content": prompt}]
            tokenizer = state["tokenizer"]
            model = state["model"]
            torch = state["torch"]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = inputs.to(state["device"])

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=state["max_new_tokens"],
                    do_sample=False,
                    use_cache=False,
                )

            input_token_len = inputs["input_ids"].shape[1]
            output_ids = generated[:, input_token_len:]
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

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

        prompt = str(req.options.get("prompt", state["prompt"]))
        tokenizer = state["tokenizer"]
        model = state["model"]
        torch = state["torch"]
        device = state["device"]

        texts = []
        for path in temp_paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                messages = [{"role": "user", "image": image, "content": prompt}]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = inputs.to(device)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=state["max_new_tokens"],
                        do_sample=False,
                        use_cache=False,
                    )

                input_token_len = inputs["input_ids"].shape[1]
                output_ids = generated[:, input_token_len:]
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                texts.append(text)

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
