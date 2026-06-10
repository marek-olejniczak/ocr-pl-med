"""
Deprecated — local wrapper for PaddleOCR.

Most dependencies (paddlepaddle, paddleocr, paddlex) are not installed by
default. Use the Docker HTTP service instead (autorunner.py).
"""

from __future__ import annotations

import warnings
import importlib
import os
from pathlib import Path
from typing import Iterable

from modele.base_wrapper import HTRModelWrapper


class PaddleOCRWrapper(HTRModelWrapper):
    """Wrapper dla PaddleOCR (tryb recognition-only, bez detekcji dokumentu).

    Deprecated — preferuj tryb HTTP przez Docker (autorunner.py).
    """

    def __init__(
        self,
        rec_model_name: str = "PP-OCRv4_mobile_rec",
        lang: str = "latin",
        device: str = "auto",
        use_angle_cls: bool = False,
        rec_batch_size: int = 8,
        cache_dir: str = "modele/cache/paddlex",
    ) -> None:
        warnings.warn(
            "PaddleOCRWrapper (local inference) is deprecated and its "
            "dependencies (paddlepaddle, paddleocr, paddlex) are not installed "
            "by default. Use the Docker HTTP service instead (autorunner.py).",
            DeprecationWarning,
            stacklevel=2,
        )
        model_slug = rec_model_name.replace("-", "_")
        super().__init__(model_name=f"PaddleOCR_{model_slug}")
        self.rec_model_name = rec_model_name
        self.lang = lang
        self.device = device
        self.use_angle_cls = bool(use_angle_cls)
        self.rec_batch_size = max(1, int(rec_batch_size))
        self.cache_dir = str(Path(cache_dir))
        self._backend = ""

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", self.cache_dir)
        # Wylacza dodatkowy check hostow modeli (przyspiesza start i unika timeoutow).
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self._check_paddle_stack_compatibility()
        self._patch_paddle_analysis_config_compat()

        resolved_device = self._resolve_device(self.device)

        # Preferowana sciezka dla PaddleOCR >= 3.x: czyste text recognition bez detekcji.
        if self._try_init_text_recognition_backend(resolved_device):
            return

        # Fallback awaryjny dla starszych/niestandardowych instalacji PaddleOCR.
        self._init_legacy_ocr_backend(resolved_device)
        print(
            f"[PaddleOCR] Inicjalizacja gotowa (legacy OCR fallback). rec_model={self.rec_model_name}, "
            f"lang={self.lang}, device={resolved_device}."
        )

    @staticmethod
    def _parse_major(version: str) -> int:
        try:
            return int(str(version).split(".")[0])
        except Exception:
            return 0

    def _check_paddle_stack_compatibility(self) -> None:
        """Chroni przed znanym segfaultem przy niekompatybilnych wersjach Paddle stack."""
        if os.environ.get("OCR_BENCH_ALLOW_UNSUPPORTED_PADDLE_STACK") == "1":
            return

        try:
            paddle = importlib.import_module("paddle")
            paddleocr = importlib.import_module("paddleocr")
            paddlex = importlib.import_module("paddlex")
        except Exception:
            # Braki zaleznosci obsluzy dalsza inicjalizacja.
            return

        paddle_v = getattr(paddle, "__version__", "0")
        paddleocr_v = getattr(paddleocr, "__version__", "0")
        paddlex_v = getattr(paddlex, "__version__", "0")

        if self._parse_major(paddleocr_v) >= 3 and self._parse_major(paddle_v) < 3:
            raise RuntimeError(
                "Niekompatybilny stack Paddle wykryty (moze powodowac SIGSEGV): "
                f"paddle={paddle_v}, paddleocr={paddleocr_v}, paddlex={paddlex_v}. "
                "Dla paddleocr/paddlex 3.x uzyj paddle >= 3.0 albo przejdz na paddleocr<3 i paddlex<3. "
                "Awaryjnie mozna pominac ten check: OCR_BENCH_ALLOW_UNSUPPORTED_PADDLE_STACK=1"
            )

    @staticmethod
    def _patch_paddle_analysis_config_compat() -> None:
        """Compat patch: PaddleX oczekuje set_optimization_level, ktorego moze brakowac."""
        try:
            paddle = importlib.import_module("paddle")
            analysis_config = paddle.base.core.AnalysisConfig
            if hasattr(analysis_config, "set_optimization_level"):
                return
            if not hasattr(analysis_config, "set_tensorrt_optimization_level"):
                return

            def _set_optimization_level(self, level: int):
                return self.set_tensorrt_optimization_level(level)

            analysis_config.set_optimization_level = _set_optimization_level
        except Exception:
            # Brak paddle albo inna implementacja - nic nie robimy.
            return

    def _try_init_text_recognition_backend(self, resolved_device: str) -> bool:
        try:
            TextRecognition = importlib.import_module("paddleocr").TextRecognition
        except Exception:
            return False

        if self.use_angle_cls:
            print("[PaddleOCR] --paddleocr-use-angle-cls zignorowane dla backendu TextRecognition.")

        try:
            self.recognizer = TextRecognition(
                model_name=self.rec_model_name,
                device=resolved_device,
            )
        except Exception as exc:
            print(f"[PaddleOCR] TextRecognition backend niedostepny, fallback do PaddleOCR: {exc}")
            return False

        self._backend = "text_recognition"
        print(
            f"[PaddleOCR] Inicjalizacja gotowa (TextRecognition). rec_model={self.rec_model_name}, "
            f"device={resolved_device}, det=wylaczone"
        )
        return True

    def _init_legacy_ocr_backend(self, resolved_device: str) -> None:
        try:
            PaddleOCR = importlib.import_module("paddleocr").PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "Brakuje zaleznosci dla PaddleOCR lub brakuje bibliotek systemowych. "
                "Zainstaluj: paddleocr, paddlepaddle/paddlepaddle-gpu oraz runtime libs "
                "(np. libgomp1, libgl1, libglib2.0-0). Szczegoly: "
                f"{exc}"
            ) from exc

        try:
            # Sciezka dla nowszego API PaddleOCR.
            self.ocr = PaddleOCR(
                text_recognition_model_name=self.rec_model_name,
                text_recognition_batch_size=self.rec_batch_size,
                use_textline_orientation=self.use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=self.lang,
                ocr_version="PP-OCRv4",
                device=resolved_device,
                show_log=False,
            )
        except TypeError:
            # Starsze wersje PaddleOCR moga nie wspierac wszystkich parametrow.
            use_gpu = resolved_device.startswith("gpu")
            fallback_kwargs = {
                "use_gpu": use_gpu,
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
                "rec_batch_num": self.rec_batch_size,
                "show_log": False,
            }
            self.ocr = PaddleOCR(**fallback_kwargs)
        self._backend = "legacy_ocr"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"

        try:
            paddle = importlib.import_module("paddle")
            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count():
                return "gpu:0"
        except Exception:
            pass

        if device == "gpu":
            print("[PaddleOCR] Zadano GPU, ale PaddlePaddle GPU jest niedostepne. Fallback do CPU.")
        return "cpu"

    @staticmethod
    def _extract_text(result) -> str:
        if not result:
            return ""

        # Najczestszy format dla det=False: [[("tekst", score)], ...]
        first = result[0] if isinstance(result, list) else result
        if isinstance(first, list):
            parts: list[str] = []
            for item in first:
                if isinstance(item, tuple) and item:
                    parts.append(str(item[0]))
                elif isinstance(item, list) and len(item) >= 2 and isinstance(item[1], tuple):
                    # Fallback dla formatu z bbox (gdyby det zostalo wlaczone poza wrapperem).
                    parts.append(str(item[1][0]))
            return " ".join(part.strip() for part in parts if str(part).strip())

        if isinstance(first, tuple) and first:
            return str(first[0]).strip()

        if isinstance(first, str):
            return first.strip()

        return str(first).strip()

    @staticmethod
    def _extract_text_from_recognition_result(result) -> str:
        if result is None:
            return ""

        # Najczestszy format TextRecognition.predict: lista slownikow, np.
        # {'rec_text': '...', 'rec_score': ...}
        if isinstance(result, dict):
            for key in ("rec_text", "text", "label"):
                if key in result and result[key] is not None:
                    return str(result[key]).strip()
            return str(result).strip()

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

        if self._backend == "text_recognition":
            predictions: list[str] = []
            for start in range(0, len(valid_paths), self.rec_batch_size):
                batch_paths = valid_paths[start:start + self.rec_batch_size]
                batch_results = self.recognizer.predict(batch_paths)
                predictions.extend(self._extract_text_from_recognition_result(item) for item in batch_results)
            return predictions

        predictions: list[str] = []
        for image_path in valid_paths:
            try:
                result = self.ocr.ocr(
                    image_path,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self.use_angle_cls,
                )
            except TypeError:
                try:
                    # Starsze API 2.x.
                    result = self.ocr.ocr(image_path, det=False, rec=True, cls=self.use_angle_cls)
                except TypeError:
                    result = self.ocr.ocr(image_path)

            predictions.append(self._extract_text(result))

        return predictions
