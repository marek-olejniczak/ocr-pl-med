from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image

from docker.common import PredictRequest, decode_base64_image_to_temp_file

app = FastAPI(title="got-ocr-service")
_STATE: dict | None = None


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
        from transformers import AutoModelForImageTextToText, AutoProcessor
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

    model_id = str(options.get("model_id", "stepfun-ai/GOT-OCR-2.0-hf"))
    cache_dir = str(options.get("cache_dir", "modele/cache/got_ocr"))
    max_new_tokens = int(options.get("max_new_tokens", 512))
    dtype = _resolve_dtype(torch, str(options.get("dtype", "auto")))

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        use_safetensors=True,
    )
    model = model.to(device).eval()

    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
        "max_new_tokens": max_new_tokens,
    }


def get_state(options: dict) -> dict:
    global _STATE
    if _STATE is None:
        _STATE = _build_state(options)
    return _STATE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "got-ocr"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    temp_path = None
    try:
        state = get_state(req.options)
        temp_path = decode_base64_image_to_temp_file(req.image_base64)

        with Image.open(temp_path) as image:
            image = image.convert("RGB")
            processor = state["processor"]
            model = state["model"]
            torch = state["torch"]

            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(state["device"]) for key, value in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=state["max_new_tokens"],
                )

            input_token_len = inputs["input_ids"].shape[1]
            output_ids = generated[0, input_token_len:]
            text = processor.decode(output_ids, skip_special_tokens=True).strip()

        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
