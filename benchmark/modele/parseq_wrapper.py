from __future__ import annotations

import importlib
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class PARSeqWrapper(HTRModelWrapper):
    """Wrapper dla PARSeq (docTR) w trybie recognition-only."""

    SUPPORTED_INPUT_SIZES = {
        "32x128": (32, 128),
        "128x128": (128, 128),
    }

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str = "modele/cache/parseq",
        input_size: str = "32x128",
        use_amp: bool = False,
        language: str = "pl",
        model_id: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        model_slug = (model_id or "parseq").split("/")[-1].replace("-", "_")
        super().__init__(model_name=f"PARSeq_{model_slug}")

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = str(Path(cache_dir))
        self.input_size = input_size if input_size in self.SUPPORTED_INPUT_SIZES else "32x128"
        self.use_amp = bool(use_amp)
        self.language = language
        self.model_id = model_id
        self.local_files_only = bool(local_files_only)

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._configure_cache_environment()

        try:
            torch = importlib.import_module("torch")
            doctr_models = importlib.import_module("doctr.models")
            doctr_transforms = importlib.import_module("doctr.transforms")
        except Exception as exc:
            raise RuntimeError(
                "Brakuje zaleznosci dla PARSeq/docTR. Zainstaluj: python-doctr, torch, torchvision"
            ) from exc

        self._torch = torch
        self._recognition_predictor = doctr_models.recognition_predictor
        self._from_hub = getattr(doctr_models, "from_hub", None)
        self._resize_cls = doctr_transforms.Resize

        self.resolved_device = self._resolve_device(device, torch.cuda.is_available())
        if self.use_amp and self.resolved_device != "cuda":
            print("[PARSeq] --parseq-use-amp zignorowane: AMP dziala tylko na CUDA.")

        if self.language.lower() == "pl":
            print(
                "[PARSeq] PARSeq pretrained nie jest dedykowany stricte PL; "
                "jakosc polskich diakrytykow zalezy od charsetu checkpointu."
            )

        self.predictor = self._build_predictor()
        self.effective_input_size = self._resolve_effective_input_size()
        self._apply_resize_preset(self.effective_input_size)
        self._move_predictor_to_device(self.resolved_device)

        print(
            f"[PARSeq] Inicjalizacja gotowa. device={self.resolved_device}, "
            f"batch_size={self.batch_size}, input_size={self.effective_input_size}, cache_dir={self.cache_dir}"
        )

    def _configure_cache_environment(self) -> None:
        base = Path(self.cache_dir)
        hf_home = base / "hf"
        torch_home = base / "torch"
        hf_home.mkdir(parents=True, exist_ok=True)
        torch_home.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("DOCTR_CACHE_DIR", str(base))
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TORCH_HOME", str(torch_home))

        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    @staticmethod
    def _resolve_device(device: str, cuda_available: bool) -> str:
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "cuda" if cuda_available else "cpu"
        return "cuda" if cuda_available else "cpu"

    def _build_predictor(self):
        if self.model_id and self._from_hub is not None:
            try:
                model_from_hub = self._from_hub(self.model_id, cache_dir=self.cache_dir)
                return self._recognition_predictor(
                    arch=model_from_hub,
                    pretrained=False,
                    batch_size=self.batch_size,
                    symmetric_pad=False,
                )
            except Exception as exc:
                if self.local_files_only:
                    raise RuntimeError(
                        "Nie udalo sie zaladowac PARSeq z lokalnego cache w trybie offline. "
                        "Uruchom raz bez --parseq-local-files-only, aby pobrac model."
                    ) from exc
                print("[PARSeq] Ostrzezenie: fallback do domyslnego PARSeq pretrained.")

        return self._recognition_predictor(
            arch="parseq",
            pretrained=True,
            batch_size=self.batch_size,
            symmetric_pad=False,
        )

    def _resolve_effective_input_size(self) -> str:
        requested_h, requested_w = self.SUPPORTED_INPUT_SIZES[self.input_size]

        model = getattr(self.predictor, "model", None)
        cfg = getattr(model, "cfg", None)
        if isinstance(cfg, dict):
            cfg_shape = cfg.get("input_shape")
            if isinstance(cfg_shape, (tuple, list)) and len(cfg_shape) >= 3:
                model_h = int(cfg_shape[-2])
                model_w = int(cfg_shape[-1])
                if (requested_h, requested_w) != (model_h, model_w):
                    print(
                        "[PARSeq] Ostrzezenie: wybrany checkpoint wspiera input "
                        f"{model_h}x{model_w}; fallback z {self.input_size} do {model_h}x{model_w}."
                    )
                    resolved = f"{model_h}x{model_w}"
                    if resolved in self.SUPPORTED_INPUT_SIZES:
                        return resolved
                    return "32x128"

        return self.input_size

    def _apply_resize_preset(self, input_size: str) -> None:
        if input_size not in self.SUPPORTED_INPUT_SIZES:
            input_size = "32x128"

        input_h, input_w = self.SUPPORTED_INPUT_SIZES[input_size]
        pre_processor = getattr(self.predictor, "pre_processor", None)
        if pre_processor is None:
            return

        pre_processor.resize = self._resize_cls(
            (input_h, input_w),
            preserve_aspect_ratio=True,
            symmetric_pad=False,
        )

    def _move_predictor_to_device(self, device: str) -> None:
        candidate_attrs = ["model", "reco_model", "predictor"]
        for attr in candidate_attrs:
            candidate = getattr(self.predictor, attr, None)
            if candidate is None:
                continue
            try:
                candidate.to(device)
                return
            except Exception:
                continue

        model = getattr(self.predictor, "model", None)
        if model is not None:
            try:
                self.predictor.model = model.to(device)
            except Exception:
                pass

    @staticmethod
    def _normalize_prediction_item(item) -> str:
        if isinstance(item, str):
            return item.strip()

        if isinstance(item, (list, tuple)):
            if item:
                first = item[0]
                if isinstance(first, str):
                    return first.strip()
            joined = " ".join(str(part) for part in item if part is not None).strip()
            return joined

        if isinstance(item, dict):
            for key in ("value", "text", "label", "word"):
                value = item.get(key)
                if isinstance(value, str):
                    return value.strip()
            return str(item).strip()

        return str(item).strip()

    @staticmethod
    def _load_rgb_array(image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    def predict(self, image_path: str) -> str:
        predictions = self.predict_batch([image_path])
        return predictions[0] if predictions else ""

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        image_paths_list = list(image_paths)
        if not image_paths_list:
            return []

        predictions: list[str] = []
        amp_enabled = self.use_amp and self.resolved_device == "cuda"

        for start in range(0, len(image_paths_list), self.batch_size):
            batch_paths = image_paths_list[start:start + self.batch_size]
            batch_images = [self._load_rgb_array(path) for path in batch_paths]

            autocast_ctx = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16, enabled=amp_enabled)
                if self.resolved_device == "cuda"
                else nullcontext()
            )

            with self._torch.inference_mode():
                with autocast_ctx:
                    batch_output = self.predictor(batch_images)

            if not isinstance(batch_output, list):
                raise RuntimeError("PARSeq/docTR zwrocil nieoczekiwany format wyniku.")

            predictions.extend(self._normalize_prediction_item(item) for item in batch_output)

        return predictions
