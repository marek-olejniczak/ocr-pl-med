"""
Deprecated — local wrapper for Calamari OCR.

Calamari 2.x requires TensorFlow on Python <= 3.11. This environment
(Python 3.12+) does not support it. Use the Docker HTTP service instead.
"""

from __future__ import annotations

import warnings
import importlib
import importlib.metadata
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class CalamariWrapper(HTRModelWrapper):
    """Wrapper dla Calamari OCR (line recognition).

    Deprecated — preferuj tryb HTTP przez Docker (autorunner.py).
    """

    MODEL_RELEASE_BASE_URL = "https://github.com/Calamari-OCR/calamari_models/releases/download/2.1"

    def __init__(
        self,
        model: str = "idiotikon",
        batch_size: int = 8,
        cache_dir: str = "modele/cache/calamari",
        local_files_only: bool = False,
        checkpoints: str | None = None,
        device: str = "auto",
    ) -> None:
        warnings.warn(
            "CalamariWrapper (local inference) is deprecated. "
            "It requires TensorFlow on Python <= 3.11, which is unsupported "
            "in this environment. Use the Docker HTTP service instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        model_slug = model.replace("-", "_")
        super().__init__(model_name=f"Calamari_{model_slug}")

        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = Path(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.checkpoints_raw = checkpoints
        self.device = device

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._configure_runtime_device()

        self.calamari_version = self._detect_calamari_version()
        self._ensure_supported_calamari_version()
        self._predictor_cls, self._multi_predictor_cls, self._predictor_params_cls = self._load_calamari_api()
        self.checkpoint_paths = self._resolve_checkpoint_paths()
        self.predictor = self._build_predictor(self.checkpoint_paths)
        runtime_backend, gpu_count = self._detect_runtime_backend()

        if self.model == "idiotikon":
            print(
                "[Calamari] Uzywany model idiotikon (wielojezyczny, bogaty zestaw diakrytykow). "
                "To najlepszy publiczny kandydat pod polskie znaki w oficjalnych modelach Calamari."
            )

        print(
            f"[Calamari] Inicjalizacja gotowa. model={self.model}, checkpoints={len(self.checkpoint_paths)}, "
            f"batch_size={self.batch_size}, cache_dir={self.cache_dir}, "
            f"device={self.device}, backend={runtime_backend}, gpu_count={gpu_count}"
        )

    def _configure_runtime_device(self) -> None:
        # Calamari korzysta z backendu TensorFlow; CPU mozna wymusic przez CUDA_VISIBLE_DEVICES.
        if self.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    @staticmethod
    def _detect_runtime_backend() -> tuple[str, int]:
        try:
            tensorflow = importlib.import_module("tensorflow")
            devices = tensorflow.config.list_physical_devices("GPU")
            return "tensorflow", len(devices)
        except Exception:
            return "unknown", 0

    @staticmethod
    def _detect_calamari_version() -> str:
        try:
            return importlib.metadata.version("calamari-ocr")
        except Exception:
            return "unknown"

    def _ensure_supported_calamari_version(self) -> None:
        version = self.calamari_version
        if version == "unknown":
            return

        try:
            major = int(version.split(".", 1)[0])
        except Exception:
            return

        if major < 2:
            raise RuntimeError(
                "Wykryto niekompatybilna wersje Calamari "
                f"{version}. Modele release 2.1/2.2 (np. idiotikon) wymagaja Calamari 2.x. "
                "Na Python 3.12 czesto brakuje kompatybilnego stosu TensorFlow dla 2.x; "
                "zalecane osobne srodowisko z Python 3.11."
            )

    @staticmethod
    def _load_calamari_api():
        """Loads Calamari API with compatibility across predictor module layouts."""
        module = None
        import_error = None
        for module_name in (
            "calamari_ocr.ocr.predict.predictor",
            "calamari_ocr.ocr.predictor",
        ):
            try:
                module = importlib.import_module(module_name)
                break
            except Exception as exc:  # pragma: no cover - depends on runtime env
                import_error = exc

        if module is None:
            raise RuntimeError(
                "Brakuje zaleznosci dla Calamari. Zainstaluj kompatybilne `calamari-ocr` (zalecane 2.x)."
            ) from import_error

        predictor_cls = getattr(module, "Predictor", None)
        multi_predictor_cls = getattr(module, "MultiPredictor", None)
        predictor_params_cls = getattr(module, "PredictorParams", None)

        if predictor_cls is None:
            raise RuntimeError("Nie znaleziono klasy Predictor w zainstalowanym Calamari.")

        return predictor_cls, multi_predictor_cls, predictor_params_cls

    def _resolve_checkpoint_paths(self) -> list[str]:
        if self.checkpoints_raw:
            paths = [Path(p.strip()) for p in self.checkpoints_raw.split(",") if p.strip()]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "Nie znaleziono podanych checkpointow Calamari: " + ", ".join(missing)
                )
            return [str(p) for p in paths]

        model_root = self.cache_dir / "models" / self.model
        if not self._has_checkpoint_dirs(model_root):
            if self.local_files_only:
                raise RuntimeError(
                    "Calamari lokalnie nie ma checkpointow dla modelu "
                    f"{self.model}. Uruchom raz bez --calamari-local-files-only, aby je pobrac."
                )
            self._download_and_extract_model(self.model)

        checkpoint_dirs = sorted(model_root.glob("*.ckpt"))
        if not checkpoint_dirs:
            raise RuntimeError(
                f"Nie znaleziono checkpointow .ckpt w {model_root}."
            )

        return [str(path) for path in checkpoint_dirs]

    @staticmethod
    def _has_checkpoint_dirs(model_root: Path) -> bool:
        return model_root.exists() and any(model_root.glob("*.ckpt"))

    def _download_and_extract_model(self, model: str) -> None:
        downloads_dir = self.cache_dir / "downloads"
        models_dir = self.cache_dir / "models"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        archive_path = downloads_dir / f"{model}.tar.gz"
        model_url = f"{self.MODEL_RELEASE_BASE_URL}/{model}.tar.gz"

        print(f"[Calamari] Pobieranie modelu: {model_url}")
        urllib.request.urlretrieve(model_url, archive_path)

        print(f"[Calamari] Rozpakowywanie modelu do: {models_dir}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(models_dir)

    def _build_predictor(self, checkpoint_paths: list[str]):
        params = self._predictor_params_cls() if self._predictor_params_cls is not None else None

        # Newer Calamari API (2.x docs): classmethods from_checkpoint/from_paths
        if hasattr(self._predictor_cls, "from_checkpoint"):
            if len(checkpoint_paths) == 1:
                return self._predictor_cls.from_checkpoint(
                    params=params,
                    checkpoint=checkpoint_paths[0],
                )

            if self._multi_predictor_cls is None:
                raise RuntimeError("Brak MultiPredictor w zainstalowanym Calamari dla ensemble.")

            kwargs = {
                "checkpoints": checkpoint_paths,
            }
            if params is not None:
                kwargs["predictor_params"] = params
            return self._multi_predictor_cls.from_paths(**kwargs)

        # Legacy API fallback.
        if len(checkpoint_paths) == 1:
            return self._predictor_cls(
                checkpoint=checkpoint_paths[0],
                batch_size=self.batch_size,
            )

        if self._multi_predictor_cls is None:
            raise RuntimeError("Brak MultiPredictor w zainstalowanym Calamari dla ensemble.")

        return self._multi_predictor_cls(
            checkpoints=checkpoint_paths,
            batch_size=self.batch_size,
        )

    @staticmethod
    def _load_grayscale_array(image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        with Image.open(path) as image:
            gray = image.convert("L")
            return np.asarray(gray, dtype=np.uint8)

    @staticmethod
    def _extract_sentence(sample) -> str:
        def _extract_from_candidate(candidate) -> str:
            if candidate is None:
                return ""

            if isinstance(candidate, str):
                return candidate.strip()

            if isinstance(candidate, dict):
                for key in ("sentence", "text", "label", "value"):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                for value in candidate.values():
                    text = _extract_from_candidate(value)
                    if text:
                        return text
                return ""

            if isinstance(candidate, tuple):
                # MultiPredictor czesto zwraca tuple: (predictions, voted_prediction).
                if len(candidate) >= 2:
                    voted_text = _extract_from_candidate(candidate[1])
                    if voted_text:
                        return voted_text
                for item in candidate:
                    text = _extract_from_candidate(item)
                    if text:
                        return text
                return ""

            if isinstance(candidate, list):
                for item in candidate:
                    text = _extract_from_candidate(item)
                    if text:
                        return text
                return ""

            for attr in ("sentence", "text", "label", "value"):
                value = getattr(candidate, attr, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            for attr in (
                "outputs",
                "prediction",
                "predictions",
                "voted",
                "voted_prediction",
                "best_prediction",
                "sample",
            ):
                nested = getattr(candidate, attr, None)
                text = _extract_from_candidate(nested)
                if text:
                    return text

            return ""

        return _extract_from_candidate(sample)

    def predict(self, image_path: str) -> str:
        predictions = self.predict_batch([image_path])
        return predictions[0] if predictions else ""

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        image_paths_list = list(image_paths)
        if not image_paths_list:
            return []

        images = [self._load_grayscale_array(path) for path in image_paths_list]
        predictions: list[str] = []

        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            batch_results = list(self.predictor.predict_raw(batch))
            predictions.extend(self._extract_sentence(sample) for sample in batch_results)

        return predictions
