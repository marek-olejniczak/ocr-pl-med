"""
Deprecated — local wrapper for EasyOCR.

The easyocr package is not installed by default in the host environment.
Use the Docker HTTP service instead (autorunner.py).
"""

from __future__ import annotations

import warnings
import importlib
from pathlib import Path
from typing import Iterable

from modele.base_wrapper import HTRModelWrapper


class EasyOCRWrapper(HTRModelWrapper):
    """Wrapper dla EasyOCR z obsluga batchingu i lokalnego cache wag.

    Deprecated — preferuj tryb HTTP przez Docker (autorunner.py).
    """

    def __init__(
        self,
        langs: list[str] | None = None,
        device: str = "auto",
        batch_size: int = 8,
        model_storage_dir: str = "modele/cache/easyocr",
    ) -> None:
        warnings.warn(
            "EasyOCRWrapper (local inference) is deprecated. "
            "Use the Docker HTTP service instead (autorunner.py + experiments.yaml).",
            DeprecationWarning,
            stacklevel=2,
        )
        resolved_langs = langs or ["pl", "en"]
        model_slug = "_".join(resolved_langs)
        super().__init__(model_name=f"EasyOCR_{model_slug}")

        self.langs = resolved_langs
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.model_storage_dir = str(Path(model_storage_dir))

        Path(self.model_storage_dir).mkdir(parents=True, exist_ok=True)

        try:
            easyocr = importlib.import_module("easyocr")
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise RuntimeError(
                "Brakuje zaleznosci dla EasyOCR. Zainstaluj: easyocr, torch, torchvision"
            ) from exc

        use_gpu = self._resolve_gpu_flag(device, torch.cuda.is_available())
        if device == "cuda" and not use_gpu:
            print("[EasyOCR] Zadano CUDA, ale GPU niedostepne. Fallback do CPU.")

        self.reader = easyocr.Reader(
            self.langs,
            gpu=use_gpu,
            model_storage_directory=self.model_storage_dir,
        )

        print(
            f"[EasyOCR] Inicjalizacja gotowa. langs={self.langs}, "
            f"device={'cuda' if use_gpu else 'cpu'}, model_storage_dir={self.model_storage_dir}"
        )

    @staticmethod
    def _resolve_gpu_flag(device: str, cuda_available: bool) -> bool:
        if device == "cpu":
            return False
        if device == "cuda":
            return bool(cuda_available)
        return bool(cuda_available)

    @staticmethod
    def _normalize_readtext_output(result) -> str:
        if not result:
            return ""

        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                    # Fallback dla detail=1.
                    text = item[1].strip()
                    if text:
                        parts.append(text)
            return " ".join(parts)

        if isinstance(result, str):
            return result.strip()

        return str(result).strip()

    def predict(self, image_path: str) -> str:
        predictions = self.predict_batch([image_path])
        return predictions[0] if predictions else ""

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        image_paths_list = list(image_paths)
        if not image_paths_list:
            return []

        valid_paths: list[str] = []
        for image_path in image_paths_list:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
            valid_paths.append(str(path))

        predictions: list[str] = []
        if self.batch_size <= 1:
            for image_path in valid_paths:
                result = self.reader.readtext(
                    image_path,
                    detail=0,
                    paragraph=False,
                    batch_size=self.batch_size,
                )
                predictions.append(self._normalize_readtext_output(result))
            return predictions

        for start in range(0, len(valid_paths), self.batch_size):
            batch_paths = valid_paths[start:start + self.batch_size]

            # Zakladamy heterogeniczne rozmiary obrazow wejsciowych.
            # readtext_batched bywa niestabilny dla takiego wejscia, wiec
            # przetwarzamy obrazy pojedynczo, ale w chunkach logicznych.
            batch_results = [
                self.reader.readtext(
                    path,
                    detail=0,
                    paragraph=False,
                    batch_size=self.batch_size,
                )
                for path in batch_paths
            ]

            predictions.extend(self._normalize_readtext_output(result) for result in batch_results)

        return predictions
