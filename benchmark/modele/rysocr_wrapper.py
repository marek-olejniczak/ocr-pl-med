from __future__ import annotations

from pathlib import Path
import re

from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class RysOCRWrapper(HTRModelWrapper):
    """Wrapper dla modelu RysOCR (LoRA na PaddleOCR-VL)."""

    def __init__(
        self,
        adapter_model_id: str = "kacperwikiel/RysOCR",
        base_model_id: str = "PaddlePaddle/PaddleOCR-VL",
        prompt: str = "Transcribe the text exactly.",
        max_new_tokens: int = 256,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(model_name="RysOCR")
        self.adapter_model_id = adapter_model_id
        self.base_model_id = base_model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only

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

        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "local_files_only": self.local_files_only,
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
        )
        self.image_token = getattr(self.processor, "image_token", "<|IMAGE_PLACEHOLDER|>")
        print("[RysOCR] Model i processor gotowe do inferencji.")

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

        with self._torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_token_count = 0
        if "input_ids" in inputs:
            input_token_count = int(inputs["input_ids"].shape[-1])

        generated_tokens = outputs[0][input_token_count:]
        if generated_tokens.numel() == 0:
            generated_tokens = outputs[0]

        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

        # Dodatkowe czyszczenie, gdy model mimo wszystko echo-uje prompt.
        if self.image_token in decoded:
            decoded = decoded.replace(self.image_token, "").strip()

        prompt_prefix = self.prompt.strip()
        if prompt_prefix:
            decoded = re.sub(rf"^{re.escape(prompt_prefix)}\s*", "", decoded).strip()
        decoded = re.sub(r"^Assistant:\s*", "", decoded).strip()

        return decoded
