from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path
import re
from typing import Iterable

from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class RysOCRWrapper(HTRModelWrapper):
    """Wrapper dla modelu RysOCR (LoRA na PaddleOCR-VL).

    Deprecated — preferuj tryb HTTP przez Docker (autorunner.py).
    """

    def __init__(
        self,
        adapter_model_id: str = "kacperwikiel/RysOCR",
        base_model_id: str = "PaddlePaddle/PaddleOCR-VL",
        prompt: str = "Transcribe the text exactly.",
        max_new_tokens: int = 256,
        device: str | None = None,
        local_files_only: bool = False,
        batch_size: int = 2,
        use_amp: bool = False,
        cache_dir: str = "modele/cache/rysocr",
    ) -> None:
        warnings.warn(
            "RysOCRWrapper (local inference) is deprecated. "
            "Use Docker HTTP mode instead (autorunner.py + experiments.yaml).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(model_name="RysOCR")
        self.adapter_model_id = adapter_model_id
        self.base_model_id = base_model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.cache_dir = str(Path(cache_dir))
        # batch_size steruje inferencja grupowa wewnatrz predict_batch.
        self.batch_size = max(1, int(batch_size))
        # use_amp wlacza mixed precision podczas generate na CUDA.
        self.use_amp = bool(use_amp)

        try:
            import torch
            import torchvision
            import sentencepiece
            import google.protobuf
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoProcessor
            import transformers.utils.generic as transformers_generic
            import transformers.modeling_rope_utils as rope_utils
            import transformers.masking_utils as masking_utils
        except Exception as exc:
            raise RuntimeError(
                "Brakuje zaleznosci dla RysOCR. Zainstaluj: torch, torchvision, transformers, "
                "peft, accelerate, sentencepiece, protobuf"
            ) from exc

        # PaddleOCR-VL remote code importuje `check_model_inputs` z transformers.utils.generic,
        # ale w czesci wersji (np. 5.x) ta funkcja zostala usunieta.
        if not hasattr(transformers_generic, "check_model_inputs"):
            def _check_model_inputs(func=None, **_kwargs):
                if func is None:
                    def _decorator(inner_func):
                        return inner_func
                    return _decorator
                return func

            transformers_generic.check_model_inputs = _check_model_inputs

        # PaddleOCR-VL moze przekazywac rope_type='default'. W transformers 5.x
        # ten klucz bywa usuniety z ROPE_INIT_FUNCTIONS, co konczy sie KeyError.
        # Definiujemy bazowa wersje RoPE (bez dodatkowego skalowania), zeby
        # nie wymagac pola `factor` jak w trybie `linear`.
        if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
                if config is None:
                    raise ValueError("config is required for default RoPE parameters")

                config.standardize_rope_params()
                rope_parameters_dict = (
                    config.rope_parameters[layer_type]
                    if layer_type is not None
                    else config.rope_parameters
                )

                base = rope_parameters_dict.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)

                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                        / dim
                    )
                )
                attention_factor = 1.0
                return inv_freq, attention_factor

            rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

        # PaddleOCR-VL remote code wywoluje create_causal_mask z argumentem
        # `inputs_embeds`, a niektore wersje transformers oczekuja `input_embeds`.
        # Podmieniamy funkcje tak, by akceptowala obie nazwy.
        original_create_causal_mask = masking_utils.create_causal_mask
        if not getattr(original_create_causal_mask, "_rysocr_compat", False):
            def _create_causal_mask_compat(*args, **kwargs):
                if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                return original_create_causal_mask(*args, **kwargs)

            _create_causal_mask_compat._rysocr_compat = True
            masking_utils.create_causal_mask = _create_causal_mask_compat

        self._torch = torch
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        if self.use_amp and self.device != "cuda":
            print("[RysOCR] --rysocr-use-amp zignorowane: AMP dziala tylko na CUDA.")

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir,
        }

        print(
            "[RysOCR] Inicjalizacja modelu. "
            f"base={self.base_model_id}, adapter={self.adapter_model_id}, device={self.device}"
        )
        if self.local_files_only:
            print("[RysOCR] Tryb offline: local_files_only=True (bez pobierania z Hugging Face).")
        else:
            print(
                "[RysOCR] Pierwsze uruchomienie moze pobrac duze wagi (ok. 2GB+) z Hugging Face."
            )

        try:
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=torch_dtype,
                    **load_kwargs,
                )
            except TypeError:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch_dtype,
                    **load_kwargs,
                )

            model = PeftModel.from_pretrained(
                base_model,
                self.adapter_model_id,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            if self.local_files_only:
                raise RuntimeError(
                    "Nie udalo sie zaladowac RysOCR w trybie offline. "
                    "Uruchom raz bez --rysocr-local-files-only, aby pobrac model do cache."
                ) from exc
            raise

        if self.device != "cuda":
            model = model.to(self.device)

        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.adapter_model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir,
        )
        self.image_token = getattr(self.processor, "image_token", "<|IMAGE_PLACEHOLDER|>")
        print("[RysOCR] Model i processor gotowe do inferencji.")

    def _decode_generated(self, generated_tokens) -> str:
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

        # Dodatkowe czyszczenie, gdy model mimo wszystko echo-uje prompt.
        if self.image_token in decoded:
            decoded = decoded.replace(self.image_token, "").strip()

        prompt_prefix = self.prompt.strip()
        if prompt_prefix:
            decoded = re.sub(rf"^{re.escape(prompt_prefix)}\s*", "", decoded).strip()

        # Czasem model zwraca fragment promptu zakonczony "Assistant:".
        # Jesli prefix przypomina prompt, obcinamy go i zostawiamy tresc odpowiedzi.
        if "Assistant:" in decoded:
            prefix, suffix = decoded.split("Assistant:", 1)
            prefix_norm = " ".join(prefix.lower().split())
            prompt_norm = " ".join(prompt_prefix.lower().split()) if prompt_prefix else ""
            if prefix_norm and prompt_norm and (prefix_norm in prompt_norm or prompt_norm in prefix_norm):
                decoded = suffix.strip()

        decoded = re.sub(r"^Assistant:\s*", "", decoded).strip()

        return decoded

    def _extract_generated_tokens(self, outputs_row, prompt_token_count: int):
        generated_tokens = outputs_row[prompt_token_count:]
        if generated_tokens.numel() == 0:
            generated_tokens = outputs_row
        return generated_tokens

    def _build_prompt(self) -> str:
        user_text = self.prompt.strip() or "Read text."

        # PaddleOCR-VL tokenizer posiada chat_template. Uzycie go daje
        # poprawny format wejscia (rola User/Assistant + tokeny obrazu).
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback, gdy chat_template nie jest dostepny.
        if self.image_token in user_text:
            return user_text
        return f"{self.image_token}\n{user_text}"

    def predict(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        with Image.open(path) as image:
            image = image.convert("RGB")
            prompt = self._build_prompt()
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        amp_enabled = self.use_amp and model_device.type == "cuda"
        autocast_ctx = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16, enabled=amp_enabled)
            if model_device.type == "cuda"
            else nullcontext()
        )

        with self._torch.no_grad():
            with autocast_ctx:
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_token_count = 0
        if "input_ids" in inputs:
            input_token_count = int(inputs["input_ids"].shape[-1])

        generated_tokens = self._extract_generated_tokens(outputs[0], input_token_count)
        return self._decode_generated(generated_tokens)

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        image_paths_list = list(image_paths)
        if not image_paths_list:
            return []

        if self.batch_size <= 1:
            return [self.predict(path) for path in image_paths_list]

        prompt = self._build_prompt()
        predictions: list[str] = []
        model_device = next(self.model.parameters()).device
        amp_enabled = self.use_amp and model_device.type == "cuda"

        for start in range(0, len(image_paths_list), self.batch_size):
            batch_paths = image_paths_list[start:start + self.batch_size]
            images = []
            for image_path in batch_paths:
                path = Path(image_path)
                if not path.exists():
                    raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

            prompts = [prompt] * len(images)
            inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            if "attention_mask" in inputs:
                input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            elif "input_ids" in inputs:
                input_token_counts = [inputs["input_ids"].shape[-1]] * len(images)
            else:
                input_token_counts = [0] * len(images)

            autocast_ctx = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16, enabled=amp_enabled)
                if model_device.type == "cuda"
                else nullcontext()
            )

            with self._torch.no_grad():
                with autocast_ctx:
                    outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            for idx, output_row in enumerate(outputs):
                generated_tokens = self._extract_generated_tokens(output_row, int(input_token_counts[idx]))
                predictions.append(self._decode_generated(generated_tokens))

        return predictions
