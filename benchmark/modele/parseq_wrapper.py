"""
PARSeq/docTR wrapper.

Dwie sciezki:
- **lokalny katalog treningowy** (`model_id` = katalog: `model.pt` + `vocab.json` +
  `meta.json`) — nasz fine-tune; architektura budowana pod rozmiar wejscia i vocab
  Z TRENINGU, wejscie letterboxowane bialym/centrowanym paddingiem jak w treningu.
- **HF hub** (`model_id` = id z huba, lub brak) — surowy docTR, preprocessing domyslny
  docTR (padding czarny, prawy/dolny brzeg).

Uruchamiane wewnatrz kontenera HTTP (`docker/parseq/app.py`, autorunner.py).
Lokalnie na hoscie brakuje zaleznosci docTR — wtedy uzywaj trybu HTTP.
"""

from __future__ import annotations

import warnings
import importlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from modele.base_wrapper import HTRModelWrapper


def letterbox_as_in_training(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox 1:1 z `training/ocr/parseq/cli.py::_letterbox`: aspect-preserve,
    BILINEAR, bialy canvas (255), centrowanie. Wejscie/wyjscie RGB uint8 (H, W, 3).

    Kolejnosc i zaokraglenia musza byc identyczne jak w treningu — inaczej model
    dostaje na inferencji obraz inny niz ten, na ktorym sie uczyl.
    """
    pil = Image.fromarray(img)
    iw, ih = pil.size
    scale = min(width / iw, height / ih)
    nw = max(int(round(iw * scale)), 1)
    nh = max(int(round(ih * scale)), 1)
    pil = pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(pil, ((width - nw) // 2, (height - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


class PARSeqWrapper(HTRModelWrapper):
    """Wrapper dla PARSeq (docTR) w trybie recognition-only."""

    # Presety rozmiaru wejscia dla modeli z HF hub (pochodza z docTR). Dla lokalnego
    # checkpointu rozmiar bierze sie z `meta.json` — moze byc dowolny (np. 32x512).
    SUPPORTED_INPUT_SIZES = {
        "32x128": (32, 128),
        "128x128": (128, 128),
    }

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str = "modele/cache/parseq",
        input_size: str | None = None,
        use_amp: bool = False,
        language: str = "pl",
        model_id: str | None = None,
        local_files_only: bool = False,
        preprocess: str | None = None,
    ) -> None:
        model_slug = (model_id or "parseq").split("/")[-1].replace("-", "_")
        super().__init__(model_name=f"PARSeq_{model_slug}")

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = str(Path(cache_dir))
        # None = nie podano; wtedy rozmiar bierze sie z architektury modelu
        self.input_size = input_size if (input_size or "") in self.SUPPORTED_INPUT_SIZES else None
        self.use_amp = bool(use_amp)
        self.language = language
        self.model_id = model_id
        self.local_files_only = bool(local_files_only)
        self.preprocess = (preprocess or "auto").lower()
        # meta.json lokalnego checkpointu (None dla modelu z huba)
        self.checkpoint_meta: dict | None = None
        # czy `model_id` wskazuje nasz katalog treningowy (model.pt+vocab.json+meta.json)
        self.local_checkpoint_dir = bool(model_id) and Path(str(model_id)).is_dir()
        # Musi byc znane PRZED budowa predictora — decyduje o split_wide_crops.
        self.train_letterbox = self._use_train_letterbox()

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._configure_cache_environment()

        try:
            torch = importlib.import_module("torch")
            doctr_models = importlib.import_module("doctr.models")
            doctr_transforms = importlib.import_module("doctr.transforms")
        except Exception as exc:
            raise RuntimeError(
                "Brakuje zaleznosci dla PARSeq/docTR lub bibliotek systemowych. "
                "Zainstaluj: python-doctr[torch], torch, torchvision oraz runtime libs "
                "(np. libgl1, libglib2.0-0, libgomp1). Szczegoly: "
                f"{exc}"
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
        # docTR domyslnie dzieli wycinki o aspekcie > 8 na kawalki o aspekcie ~6
        # (`RecognitionPredictor.split_crops`: critical_ar=8, target_ar=6). Ten wrapper
        # ZAWSZE podaje predictorowi plotno dokladnie w rozmiarze wejscia modelu
        # (letterbox w _prepare_array albo docTR T.Resize), wiec split nie ma czego
        # poprawic, a potrafi zepsuc: przy 32x512 (aspekt 16) docTR pocialby plotno na
        # kawalki 32x192, model dostalby 192 patchy zamiast 513 i forward wywalilby sie
        # na ksztalcie `positions` (RuntimeError 193 vs 513). Przy 32x128 (aspekt 4)
        # split nigdy nie zachodzil — dlatego wyszlo to dopiero przy szerszym wejsciu.
        # Nie da sie tego przekazac przez `recognition_predictor`: jego **kwargs ida do
        # PreProcessor/Resize, nie do RecognitionPredictor.
        self.predictor.split_wide_crops = False
        self.effective_input_size = self._resolve_effective_input_size()
        self._apply_resize_preset(self.effective_input_size)
        self._move_predictor_to_device(self.resolved_device)

        source = (
            f"lokalny checkpoint {self.model_id}" if self.checkpoint_meta is not None
            else f"HF hub {self.model_id or 'parseq (domyslny)'}"
        )
        print(
            f"[PARSeq] Inicjalizacja gotowa. device={self.resolved_device}, "
            f"batch_size={self.batch_size}, input_size={self.effective_input_size}, "
            f"preprocessing={'letterbox jak w treningu (bialy, centrowany)' if self.train_letterbox else 'docTR (czarny, prawy/dolny)'}, "
            f"zrodlo={source}, cache_dir={self.cache_dir}"
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
        if self.model_id and Path(str(self.model_id)).is_dir():
            return self._build_predictor_from_checkpoint()

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

    def _build_predictor_from_checkpoint(self):
        """Buduje model pod ROZMIAR i VOCAB z treningu i laduje `model.pt`.

        Nie uzywamy `from_pretrained`: docTR przy niepustym `ignore_keys` robi
        `load_state_dict(strict=False)` i wyrzuca z checkpointu 4 klucze
        `decoder.*_norm.*` (hack na kompatybilnosc ze starymi wagami) — czyli
        gubilby wytrenowane wartosci tych warstw. Nasz `model.pt` to czysty
        state_dict tej samej architektury, wiec ladujemy go wprost.
        """
        ckpt_dir = Path(str(self.model_id))
        vocab_path = ckpt_dir / "vocab.json"
        meta_path = ckpt_dir / "meta.json"
        model_path = ckpt_dir / "model.pt"
        for path in (model_path, vocab_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"Brak {path} w katalogu checkpointu {ckpt_dir}")

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        if isinstance(vocab, list):
            vocab = "".join(vocab)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.checkpoint_meta = meta

        input_shape = tuple(int(v) for v in meta.get("input_shape", (3, 32, 128)))
        if len(input_shape) != 3:
            raise ValueError(
                f"meta.json w {ckpt_dir} ma input_shape={input_shape}, oczekuje (C, H, W)"
            )
        max_label_length = int(meta.get("max_label_length", 32))

        parseq_fn = importlib.import_module("doctr.models").parseq
        model = parseq_fn(
            vocab=vocab,
            pretrained=False,
            input_shape=input_shape,
            max_length=max_label_length,
        )

        state_dict = self._torch.load(model_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint {model_path} nie pasuje do architektury z meta.json "
                f"(vocab_len={len(vocab)}, input_shape={input_shape}, "
                f"max_label_length={max_label_length}).\n"
                f"Brakujace klucze: {missing_keys}\nNieoczekiwane klucze: {unexpected_keys}"
            )

        print(
            f"[PARSeq] Lokalny checkpoint {ckpt_dir}: vocab_len={len(vocab)}, "
            f"input_shape={input_shape}, max_label_length={max_label_length}"
        )

        kwargs = {"batch_size": self.batch_size, "symmetric_pad": False}
        # mean/std z meta (u treningu brane z cfg modelu bazowego) — gdyby kiedys
        # sie rozjechaly z domyslnymi cfg, inferencja ma uzyc tych z treningu.
        for key in ("mean", "std"):
            if isinstance(meta.get(key), (list, tuple)) and len(meta[key]) == 3:
                kwargs[key] = tuple(float(v) for v in meta[key])

        return self._recognition_predictor(arch=model, pretrained=False, **kwargs)

    def _resolve_effective_input_size(self) -> str:
        """Rozmiar wejscia = rozmiar z architektury modelu (dla lokalnego checkpointu:
        ten z treningu). `input_size` z opcji jest tylko zyczeniem — przy rozbieznosci
        wygrywa model, bo inny rozmiar wywali forward (ksztalt `positions` ViT).

        Predictor docTR bierze output_size z `model.cfg["input_shape"]`, wiec
        zbudowanie modelu pod rozmiar z meta.json zalatwia tez resize w preprocessingu.
        """
        model = getattr(self.predictor, "model", None)
        cfg = getattr(model, "cfg", None)
        cfg_shape = cfg.get("input_shape") if isinstance(cfg, dict) else None
        if not (isinstance(cfg_shape, (tuple, list)) and len(cfg_shape) >= 3):
            return self.input_size or "32x128"

        model_h, model_w = int(cfg_shape[-2]), int(cfg_shape[-1])
        requested = self.SUPPORTED_INPUT_SIZES.get(self.input_size)
        if requested is not None and requested != (model_h, model_w):
            print(
                f"[PARSeq] Ostrzezenie: model wymaga wejscia {model_h}x{model_w}, "
                f"a podano input_size={self.input_size} — uzywam {model_h}x{model_w}."
            )
        return f"{model_h}x{model_w}"

    def _use_train_letterbox(self) -> bool:
        if self.preprocess == "train":
            return True
        if self.preprocess == "doctr":
            return False
        if self.preprocess != "auto":
            print(f"[PARSeq] Ostrzezenie: nieznane preprocess={self.preprocess!r} — uzywam auto.")
        # auto: nasz checkpoint trenowal na bialym, centrowanym letterboxie
        return self.local_checkpoint_dir

    def _make_passthrough_resize(self, height: int, width: int):
        """Zamiennik docTR `T.Resize`, ktory NIE zmienia obrazu.

        Obrazy sa juz letterboxowane 1:1 z treningiem w `_prepare_array`, a domyslny
        `T.Resize` docTR skalowalby/padowal je ponownie: `torch.nn.functional.pad`
        z domyslna wartoscia 0 (czarne tlo) i — przy `symmetric_pad=False` — do
        prawego/dolnego brzegu, podczas gdy trening paddowal bialo i centrowal.
        Przy 32x128 padding to 22 z 32 wierszy, wiec roznica jest istotna.
        """
        torch = self._torch
        size = (int(height), int(width))

        class _TrainLetterboxPassthrough(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.size = size
                self.return_padding_mask = False

            def forward(self, sample):
                return sample

            def extra_repr(self) -> str:
                return f"{size[0]}x{size[1]}, letterbox jak w treningu"

        return _TrainLetterboxPassthrough()

    def _apply_resize_preset(self, input_size: str) -> None:
        pre_processor = getattr(self.predictor, "pre_processor", None)
        if pre_processor is None:
            return

        input_h, input_w = (int(part) for part in input_size.lower().split("x"))
        if self.train_letterbox:
            pre_processor.resize = self._make_passthrough_resize(input_h, input_w)
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

    def _prepare_array(self, image_path: str) -> np.ndarray:
        """RGB uint8 (H, W, 3); przy wlaczonym letterboxie od razu pod rozmiar modelu."""
        image = self._load_rgb_array(image_path)
        if self.train_letterbox:
            height, width = (int(part) for part in self.effective_input_size.lower().split("x"))
            image = letterbox_as_in_training(image, height, width)
        return image

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        image_paths_list = list(image_paths)
        if not image_paths_list:
            return []

        predictions: list[str] = []
        amp_enabled = self.use_amp and self.resolved_device == "cuda"

        for start in range(0, len(image_paths_list), self.batch_size):
            batch_paths = image_paths_list[start:start + self.batch_size]
            batch_images = [self._prepare_array(path) for path in batch_paths]

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
