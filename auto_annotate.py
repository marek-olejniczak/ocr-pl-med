"""
auto_annotate.py — automatyczna pre-anotacja stron: detekcja linii + rozpoznanie tekstu (HTR).

Pipeline:
    obraz strony -> detect_lines() -> boxy linii -> Recognizer.recognize() -> tekst

Detekcja linii jest heurystyczna (OpenCV, morfologia) — działa też dla notatek
rozproszonych wokół rysunków, gdzie zwykła projekcja pozioma zawodzi.

Rozpoznawanie jest wymienne (pluggable) — patrz get_recognizer(). Domyślnie
lokalny model wizyjny przez Ollama.

UWAGA METODOLOGICZNA (istotne dla pracy):
    Jeśli zbiór anotowany ręcznie służy jako TEST SET, do pre-anotacji należy
    używać modelu NIEZALEŻNEGO od modelu ocenianego — inaczej pojawia się
    verification bias (anotator akceptuje podpowiedzi, wyniki są zawyżone).
"""

from __future__ import annotations

import atexit
import base64
import io
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# ── Konfiguracja (nadpisywalna zmiennymi srodowiskowymi) ───────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
HTR_MODEL = os.environ.get("HTR_MODEL", "qwen3-vl:8b")

# Silnik rozpoznawania tekstu:
#   surya_lora — Surya + adapter LoRA zespołu (osobny proces, .venv-surya17)
#   trocr  — własny model dostrojony do pisma z notatek (domyślny, ~0.1 s/linię)
#   ollama — ogólny model wizyjno-językowy (wolny; qwen3-vl marnuje budżet
#            tokenów na rozumowanie i zwraca pustą odpowiedź)
#   none   — sama detekcja linii, bez odczytu tekstu
HTR_BACKEND = os.environ.get("HTR_BACKEND", "trocr")  # trocr | ollama | none

# Model dostrojony na własnych notatkach. Wariant `_pl` startował z modelu
# pretrenowanego na polskim piśmie — na zbiorze walidacyjnym osiąga CER 0.047
# wobec 0.112 dla wariantu z bazy angielskiej.
HTR_TROCR_DIR = os.environ.get("HTR_TROCR_DIR", "trocr_finetuned_pl")
# Detektor linii. Zmierzono na własnym zbiorze (bench_detectors.py, Noga, coverage):
#   craft F1 0.712 (recall 0.83) | surya F1 0.681 (recall 0.83, 5x szybsza) | morph 0.03
# CRAFT jest domyślny: najlepsze F1 i działa w tym samym venv co narzędzie
# (surya wymaga Pythona >=3.10 -> osobny .venv-surya).
HTR_DETECTOR = os.environ.get("HTR_DETECTOR", "yolo")  # yolo | craft | surya | morph

# YOLO — detektor linii wytrenowany przez zespół (gałąź `server`, best_iou_median.pt).
HTR_YOLO_WEIGHTS = os.environ.get("HTR_YOLO_WEIGHTS", "models/yolo_lines_best_iou_median.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))  # punkt pracy z benchmarku
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))  # próg NMS jak w inference.ipynb
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "512"))  # rozmiar z treningu (1024 daje ~6x więcej FP)
# Etap fotometryczny (CLAHE, usunięcie kratki, separacja atramentu) — tylko wejście detektora.
YOLO_PHOTOMETRIC = os.environ.get("YOLO_PHOTOMETRIC", "1") == "1"

# Progi detektora CRAFT. Wartości domyślne biblioteki (0.6/0.35/0.4) są dobrane
# pod tekst drukowany. Poniższe wynikają z pomiaru na własnym zbiorze notatek
# odręcznych (bench_detectors.py): F1 0.36 -> 0.48 przy IoU 0.3.
CRAFT_TEXT_THRESHOLD = float(os.environ.get("CRAFT_TEXT_THRESHOLD", "0.8"))
CRAFT_LOW_TEXT = float(os.environ.get("CRAFT_LOW_TEXT", "0.5"))
CRAFT_LINK_THRESHOLD = float(os.environ.get("CRAFT_LINK_THRESHOLD", "0.5"))

# Margines (w pikselach) dodawany do wycinka linii przed rozpoznaniem —
# odrobina kontekstu poprawia jakosc odczytu.
CROP_MARGIN = 6


@dataclass
class LineBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float | None = None  # pewność detektora, jeśli ją zwraca

    def as_dict(self) -> dict:
        data = {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }
        if self.confidence is not None:
            data["confidence"] = round(float(self.confidence), 3)
        return data


# ══════════════════════════════════════════════════════════════════════
# 1. DETEKCJA LINII
# ══════════════════════════════════════════════════════════════════════

def detect_lines(
    pil_img: Image.Image, *, max_lines: int = 300, detector: str | None = None
) -> list[LineBox]:
    """Znajduje prostokąty otaczające linie tekstu.

    Dostępne metody (kolejność = malejąca jakość, z automatycznym fallbackiem):
      surya — model segmentacyjny do dokumentów; zwraca też pewność detekcji.
      craft — detektor tekstu z EasyOCR.
      morph — heurystyka morfologiczna, bez zależności od torcha.

    Podanie `detector` wymusza konkretną metodę (bez fallbacku) — przydatne
    przy porównywaniu metod w benchmarku.
    """
    if detector:
        name = detector.lower()
        if name == "yolo":
            return _detect_lines_yolo(pil_img)[:max_lines]
        if name == "surya":
            return _detect_lines_surya(pil_img)[:max_lines]
        if name == "craft":
            return _detect_lines_craft(pil_img)[:max_lines]
        return _detect_lines_morph(pil_img)[:max_lines]

    chain = ["yolo", "surya", "craft", "morph"]
    start = chain.index(HTR_DETECTOR) if HTR_DETECTOR in chain else 0
    for name in chain[start:]:
        try:
            fn = {
                "yolo": _detect_lines_yolo,
                "surya": _detect_lines_surya,
                "craft": _detect_lines_craft,
                "morph": _detect_lines_morph,
            }[name]
            boxes = fn(pil_img)
            if boxes:
                return boxes[:max_lines]
        except Exception:  # noqa: BLE001 — brak biblioteki/wag: próbuj dalej
            continue
    return []


_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        weights = Path(HTR_YOLO_WEIGHTS)
        if not weights.is_absolute():
            weights = Path(__file__).parent.resolve() / weights
        _yolo_model = YOLO(str(weights))
    return _yolo_model


def _detect_lines_yolo(pil_img: Image.Image) -> list[LineBox]:
    model = _get_yolo_model()
    det_input = pil_img.convert("RGB")
    if YOLO_PHOTOMETRIC:
        import cv2
        from preprocessing import DocumentPreprocessor

        prep = DocumentPreprocessor({"deskew": False, "border_px": 0, "alpha_crop": False})
        out = prep.photometric(cv2.cvtColor(np.array(det_input), cv2.COLOR_RGB2BGR))
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        det_input = Image.fromarray(out).convert("RGB")
    result = model.predict(
        det_input, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False
    )[0]

    w, h = pil_img.size
    boxes: list[LineBox] = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        box = _clamp_box(x1, y1, x2, y2, w, h)
        box.confidence = float(b.conf[0])
        boxes.append(box)

    boxes = [b for b in boxes if b.x_max > b.x_min and b.y_max > b.y_min]
    heights = [b.y_max - b.y_min for b in boxes] or [20]
    return _sort_reading_order(boxes, line_tol=max(float(np.median(heights)) * 0.6, 10))


_surya_predictor = None


def _get_surya_predictor():
    """Leniwie tworzy detektor Surya (sam model detekcji działa na torchu,
    bez uruchamiania ciężkiego backendu VLM)."""
    global _surya_predictor
    if _surya_predictor is None:
        from surya.detection import DetectionPredictor

        _surya_predictor = DetectionPredictor()
    return _surya_predictor


def _detect_lines_surya(pil_img: Image.Image) -> list[LineBox]:
    predictor = _get_surya_predictor()
    image = pil_img.convert("RGB")
    result = predictor([image])[0]

    w, h = image.size
    boxes: list[LineBox] = []
    for line in result.bboxes:
        x_min, y_min, x_max, y_max = line.bbox
        box = _clamp_box(x_min, y_min, x_max, y_max, w, h)
        box.confidence = getattr(line, "confidence", None)
        boxes.append(box)

    boxes = [b for b in boxes if b.x_max > b.x_min and b.y_max > b.y_min]
    boxes = _merge_overlapping(boxes)
    boxes = _group_into_lines(boxes)
    heights = [b.y_max - b.y_min for b in boxes] or [20]
    return _sort_reading_order(boxes, line_tol=max(float(np.median(heights)) * 0.6, 10))


_craft_reader = None


def _get_craft_reader():
    """Leniwie tworzy czytnik EasyOCR (ładuje wagi przy pierwszym użyciu).

    `recognizer=False` — potrzebujemy wyłącznie detektora CRAFT, więc model
    rozpoznawania nie jest pobierany ani ładowany do pamięci.
    """
    global _craft_reader
    if _craft_reader is None:
        import easyocr

        _craft_reader = easyocr.Reader(
            ["pl"], gpu=False, verbose=False, recognizer=False
        )
    return _craft_reader


def _detect_lines_craft(pil_img: Image.Image) -> list[LineBox]:
    """Detekcja linii siecią CRAFT; EasyOCR sam grupuje słowa w linie."""
    reader = _get_craft_reader()
    arr = np.array(pil_img.convert("RGB"))

    horizontal, free = reader.detect(
        arr,
        text_threshold=CRAFT_TEXT_THRESHOLD,
        low_text=CRAFT_LOW_TEXT,
        link_threshold=CRAFT_LINK_THRESHOLD,
        # progi grupowania słów w linie — pismo odręczne ma nierówne odstępy
        ycenter_ths=0.6,
        height_ths=0.7,
        width_ths=0.9,
        slope_ths=0.3,
        add_margin=0.1,
    )

    w, h = pil_img.size
    boxes: list[LineBox] = []

    for group in horizontal or []:
        for b in group:
            x_min, x_max, y_min, y_max = b[0], b[1], b[2], b[3]
            boxes.append(_clamp_box(x_min, y_min, x_max, y_max, w, h))

    # Obiekty pochyłe/obrócone wracają jako czworokąty — bierzemy ich obrys.
    for group in free or []:
        for poly in group:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            boxes.append(_clamp_box(min(xs), min(ys), max(xs), max(ys), w, h))

    boxes = [b for b in boxes if b.x_max > b.x_min and b.y_max > b.y_min]
    boxes = _merge_overlapping(boxes)
    boxes = _group_into_lines(boxes)
    heights = [b.y_max - b.y_min for b in boxes] or [20]
    return _sort_reading_order(boxes, line_tol=max(float(np.median(heights)) * 0.6, 10))


def _group_into_lines(
    boxes: list[LineBox], *, v_overlap: float = 0.35, gap_factor: float = 1.2
) -> list[LineBox]:
    """Scala boxy słów w boxy linii.

    Detektory tekstu (CRAFT) zwracają zwykle pojedyncze słowa. Anotacja
    działa na liniach, więc sąsiadujące słowa trzeba połączyć.

    Dwa boxy należą do tej samej linii, gdy:
      * zachodzą na siebie pionowo w co najmniej `v_overlap` niższego z nich,
      * odstęp poziomy jest mniejszy niż `gap_factor` × wysokość linii.
    Łączenie powtarza się, dopóki coś się zmienia (przechodniość).
    """
    if not boxes:
        return []

    current = list(boxes)
    changed = True
    while changed:
        changed = False
        current.sort(key=lambda b: (b.y_min, b.x_min))
        merged: list[LineBox] = []
        consumed = [False] * len(current)

        for i, box in enumerate(current):
            if consumed[i]:
                continue
            acc = box
            for j in range(i + 1, len(current)):
                if consumed[j]:
                    continue
                other = current[j]
                if _same_line(acc, other, v_overlap, gap_factor):
                    acc = LineBox(
                        min(acc.x_min, other.x_min),
                        min(acc.y_min, other.y_min),
                        max(acc.x_max, other.x_max),
                        max(acc.y_max, other.y_max),
                        confidence=_min_conf(acc.confidence, other.confidence),
                    )
                    consumed[j] = True
                    changed = True
            merged.append(acc)
        current = merged

    return current


def _same_line(a: LineBox, b: LineBox, v_overlap: float, gap_factor: float) -> bool:
    overlap = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if overlap <= 0:
        return False
    min_height = min(a.y_max - a.y_min, b.y_max - b.y_min)
    if min_height <= 0 or overlap / min_height < v_overlap:
        return False

    gap = max(a.x_min, b.x_min) - min(a.x_max, b.x_max)
    max_height = max(a.y_max - a.y_min, b.y_max - b.y_min)
    return gap <= gap_factor * max_height


def _min_conf(a: float | None, b: float | None) -> float | None:
    values = [c for c in (a, b) if c is not None]
    return min(values) if values else None


def _clamp_box(x_min, y_min, x_max, y_max, w: int, h: int) -> LineBox:
    """Przycina prostokąt do granic obrazu."""
    return LineBox(
        max(0, int(x_min)),
        max(0, int(y_min)),
        min(w, int(x_max)),
        min(h, int(y_max)),
    )


def _detect_lines_morph(pil_img: Image.Image) -> list[LineBox]:
    """Heurystyka: binaryzacja -> usunięcie kratki -> dylatacja pozioma ->
    kontury -> filtracja -> scalanie -> kolejność czytania."""
    import cv2  # import lokalny — biblioteka jest opcjonalna

    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    # Binaryzacja: tekst = biały (255) na czarnym tle.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    # Notatki bywają pisane w kratce/linie — linie papieru trzeba usunąć,
    # inaczej dylatacja zszywa całą stronę w jeden blob.
    binary = _remove_ruling_lines(binary, cv2)

    # Usunięcie drobnego szumu (pojedyncze piksele).
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    char_h = _estimate_char_height(binary, cv2)

    # Scalanie znaków w linie: jądro szerokie poziomo, wąskie pionowo.
    # Szerokość ~1.2x wysokości znaku zszywa litery i spacje w słowach,
    # nie łącząc przy tym osobnych akapitów/kolumn.
    kx = max(9, int(char_h * 1.2))
    ky = max(3, int(char_h * 0.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    merged = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h = max(6, int(char_h * 0.5))
    max_h = int(h * 0.25)
    min_w = max(10, int(char_h * 0.8))
    page_area = float(w * h)

    boxes: list[LineBox] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < min_h or bh > max_h or bw < min_w:
            continue
        # Odrzuć obiekty zajmujące znaczną część strony (ramki, tabele, tło).
        if (bw * bh) / page_area > 0.5:
            continue
        pad = 2
        boxes.append(
            LineBox(
                max(0, x - pad),
                max(0, y - pad),
                min(w, x + bw + pad),
                min(h, y + bh + pad),
            )
        )

    boxes = _merge_overlapping(boxes)
    boxes = _sort_reading_order(boxes, line_tol=max(char_h, 10))
    return boxes


def _remove_ruling_lines(binary: np.ndarray, cv2) -> np.ndarray:
    """Usuwa linie papieru (kratka, linatura, marginesy).

    Wykrywa długie, proste ciągi poziome i pionowe. Próg długości jest
    ułamkiem wymiaru strony — znacznie więcej niż pojedyncze pociągnięcie
    piórem, więc pismo odręczne pozostaje nienaruszone.
    """
    h, w = binary.shape
    len_h = max(40, int(w * 0.15))
    len_v = max(40, int(h * 0.15))

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (len_h, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, len_v))
    )
    ruling = cv2.bitwise_or(horizontal, vertical)
    if not ruling.any():
        return binary

    # Pogrubienie, by usunąć całą grubość linii wraz z antyaliasingiem.
    ruling = cv2.dilate(
        ruling, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    cleaned = cv2.subtract(binary, ruling)

    # Sklejenie liter przerwanych w miejscu usuniętej linii.
    return cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )


def _estimate_char_height(binary: np.ndarray, cv2) -> float:
    """Mediana wysokości składowych spójnych ≈ wysokość znaku."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= 6
    ]
    if not heights:
        return 20.0
    med = float(np.median(heights))
    return min(max(med, 8.0), 120.0)


def _merge_overlapping(boxes: list[LineBox], iou_thresh: float = 0.15) -> list[LineBox]:
    """Scala prostokąty, które istotnie na siebie nachodzą."""
    result: list[LineBox] = []
    for box in sorted(boxes, key=lambda b: (b.y_min, b.x_min)):
        for i, kept in enumerate(result):
            if _overlap_ratio(box, kept) > iou_thresh:
                result[i] = LineBox(
                    min(kept.x_min, box.x_min),
                    min(kept.y_min, box.y_min),
                    max(kept.x_max, box.x_max),
                    max(kept.y_max, box.y_max),
                )
                break
        else:
            result.append(box)
    return result


def _overlap_ratio(a: LineBox, b: LineBox) -> float:
    ix = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / min(area_a, area_b)


def _sort_reading_order(boxes: list[LineBox], line_tol: float) -> list[LineBox]:
    """Sortuje w kolejności czytania: wiersz po wierszu, w wierszu od lewej."""
    rows: list[list[LineBox]] = []
    for box in sorted(boxes, key=lambda b: b.y_min):
        center = (box.y_min + box.y_max) / 2
        for row in rows:
            row_center = sum((b.y_min + b.y_max) / 2 for b in row) / len(row)
            if abs(center - row_center) <= line_tol:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: list[LineBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.x_min))
    return ordered


# ══════════════════════════════════════════════════════════════════════
# 2. ROZPOZNAWANIE TEKSTU (pluggable)
# ══════════════════════════════════════════════════════════════════════

PROMPT_PL = (
    "Przepisz DOKŁADNIE tekst widoczny na obrazku. To fragment odręcznych "
    "notatek po polsku (tematyka medyczna/biologiczna).\n"
    "Zasady:\n"
    "- Zwróć WYŁĄCZNIE przepisany tekst, bez komentarzy i cudzysłowów.\n"
    "- Zachowaj polskie znaki diakrytyczne (ą, ć, ę, ł, ń, ó, ś, ź, ż).\n"
    "- Zachowaj oryginalną pisownię, skróty i numerację.\n"
    "- Strzałki zapisz jako ->\n"
    "- Jeśli nie ma czytelnego tekstu, zwróć pusty ciąg."
)


class Recognizer:
    """Interfejs rozpoznawania tekstu z wycinka linii."""

    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def recognize(self, crop: Image.Image) -> str:
        raise NotImplementedError


class NullRecognizer(Recognizer):
    """Zwraca pusty tekst — służy do samej detekcji boxów."""

    name = "none"

    def available(self) -> tuple[bool, str]:
        return True, "tylko detekcja linii (bez rozpoznawania tekstu)"

    def recognize(self, crop: Image.Image) -> str:
        return ""


class TrOCRRecognizer(Recognizer):
    """Własny model TrOCR dostrojony do pisma z anotowanych notatek.

    Model ładowany jest leniwie i pozostaje w pamięci między wywołaniami.
    Na Apple Silicon korzysta z akceleracji MPS.
    """

    name = "trocr"

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or HTR_TROCR_DIR)
        if not self.model_dir.is_absolute():
            self.model_dir = Path(__file__).parent.resolve() / self.model_dir
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._device = None
        self._start_token_id = None

    def available(self) -> tuple[bool, str]:
        if not (self.model_dir / "config.json").exists():
            return False, (
                f"Brak modelu w {self.model_dir}. "
                "Uruchom: python train_trocr.py"
            )
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            return False, f"Brak biblioteki: {e}"
        return True, f"TrOCR z {self.model_dir.name}"

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

        self._device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        source = str(self.model_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(source).to(self._device).eval()

        # Checkpoint bywa zapisany nowszą wersją `transformers`, która używa
        # innej nazwy klasy tokenizera. Wtedy wczytujemy plik tokenizera
        # bezpośrednio, a w ostateczności bierzemy tokenizer RoBERTy
        # (TrOCR korzysta z identycznego słownika).
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(source)
        except (ValueError, OSError, KeyError):
            from transformers import PreTrainedTokenizerFast

            tokenizer_file = self.model_dir / "tokenizer.json"
            if tokenizer_file.exists():
                self._tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_file),
                    bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                    sep_token="</s>", cls_token="<s>", pad_token="<pad>",
                    mask_token="<mask>",
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        try:
            self._image_processor = AutoImageProcessor.from_pretrained(source)
        except Exception:  # noqa: BLE001 — checkpoint bez konfiguracji obrazu
            self._image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )

        # Konfiguracja generowania bywa zapisana z innym tokenem startowym niż
        # ten użyty podczas treningu (wartość z podkonfiguracji dekodera).
        # Niezgodność powoduje, że model natychmiast kończy generowanie i zwraca
        # pusty lub jednoznakowy wynik.
        start_id = self._model.config.decoder_start_token_id
        if start_id is not None:
            self._model.generation_config.decoder_start_token_id = start_id
        self._start_token_id = start_id

    def recognize(self, crop: Image.Image) -> str:
        import torch

        self._load()
        pixel_values = self._image_processor(
            crop.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        kwargs = {"max_new_tokens": 96}
        if self._start_token_id is not None:
            kwargs["decoder_start_token_id"] = self._start_token_id

        with torch.no_grad():
            generated = self._model.generate(pixel_values, **kwargs)
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return _clean_output(text)


class SuryaLoraRecognizer(Recognizer):
    """Surya 0.17.1 + adapter LoRA zespołu (gałąź `server`).

    Surya wymaga Pythona >=3.10 i dokładnie tych wersji bibliotek co trening,
    więc model działa w osobnym procesie (surya_worker.py, .venv-surya17).
    Worker jest startowany automatycznie przy pierwszym użyciu.
    """

    name = "surya_lora"

    def __init__(self, url: str | None = None):
        port = os.environ.get("SURYA_WORKER_PORT", "5055")
        self.url = (url or os.environ.get("SURYA_WORKER_URL", f"http://127.0.0.1:{port}")).rstrip("/")
        self._root = Path(__file__).parent.resolve()
        self._python = self._root / ".venv-surya17" / "bin" / "python"
        self._adapter = self._root / "models" / "surya_lora_adapter"
        self._proc = None

    def _healthy(self) -> bool:
        import requests

        try:
            return requests.get(f"{self.url}/health", timeout=2).ok
        except Exception:  # noqa: BLE001
            return False

    def available(self) -> tuple[bool, str]:
        if self._healthy():
            return True, "Surya + adapter LoRA gotowy"
        if not self._python.exists():
            return False, (
                "Brak .venv-surya17. Utwórz: python3.13 -m venv .venv-surya17 && "
                ".venv-surya17/bin/pip install surya-ocr==0.17.1 peft transformers==4.56.1 flask"
            )
        if not (self._adapter / "adapter_model.safetensors").exists():
            return False, f"Brak adaptera w {self._adapter}"
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [str(self._python), str(self._root / "surya_worker.py")],
                cwd=self._root,
                stdout=open(self._root / "surya_worker.log", "ab"),
                stderr=subprocess.STDOUT,
            )
            atexit.register(self._proc.terminate)
        return True, "Surya + adapter LoRA startuje (ładowanie modelu)"

    def _wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._healthy():
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError("surya_worker zakończył się błędem — zob. surya_worker.log")
            time.sleep(1)
        raise RuntimeError("surya_worker nie wystartował w limicie czasu")

    def recognize_many(self, crops: list[Image.Image]) -> list[str]:
        import requests

        self._wait_ready()
        images = []
        for crop in crops:
            buf = io.BytesIO()
            crop.convert("RGB").save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        resp = requests.post(
            f"{self.url}/recognize_batch", json={"images_base64": images}, timeout=300
        )
        resp.raise_for_status()
        return [_clean_output(t) for t in resp.json()["texts"]]

    def recognize(self, crop: Image.Image) -> str:
        return self.recognize_many([crop])[0]


class OllamaVLMRecognizer(Recognizer):

    name = "ollama"

    def __init__(self, model: str = HTR_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def available(self) -> tuple[bool, str]:
        import requests

        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            return False, f"Ollama nie odpowiada ({self.url}): {e}"

        tags = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(t == self.model or t.startswith(f"{self.model}:") for t in tags):
            return False, f"Brak modelu '{self.model}'. Pobierz: ollama pull {self.model}"
        return True, f"{self.model} gotowy"

    def recognize(self, crop: Image.Image) -> str:
        import requests

        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        resp = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": PROMPT_PL,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return _clean_output(resp.json().get("response", ""))


def _clean_output(text: str) -> str:
    """Usuwa typowe ozdobniki modelu (cudzysłowy, prefiksy, bloki kodu)."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    text = re.sub(
        r"^(tekst|transkrypcja|odczytany tekst|text)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if len(text) >= 2 and text[0] in "\"'„»" and text[-1] in "\"'”«":
        text = text[1:-1]
    # Model bywa gadatliwy przy nieczytelnym wejściu — traktuj jako brak tekstu.
    if re.search(r"(nie (ma|widać|jest)|brak) .*(czyteln|tekst)", text, re.IGNORECASE):
        return ""
    return " ".join(text.split())


_recognizer_cache: dict[str, Recognizer] = {}


def get_recognizer(backend: str | None = None) -> Recognizer:
    """Zwraca rozpoznawacz danego typu.

    Instancje są współdzielone — model waży ponad gigabajt, więc tworzenie go
    przy każdym żądaniu oznaczałoby wielosekundowe przeładowania.
    """
    name = (backend or HTR_BACKEND).lower()
    if name not in _recognizer_cache:
        if name == "trocr":
            _recognizer_cache[name] = TrOCRRecognizer()
        elif name == "surya_lora":
            _recognizer_cache[name] = SuryaLoraRecognizer()
        elif name == "ollama":
            _recognizer_cache[name] = OllamaVLMRecognizer()
        else:
            _recognizer_cache[name] = NullRecognizer()
    return _recognizer_cache[name]


# ══════════════════════════════════════════════════════════════════════
# 3. ORKIESTRACJA
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AutoAnnotateProgress:
    total: int = 0
    done: int = 0
    stage: str = "queued"  # queued | detecting | recognizing | finished | error
    error: str = ""
    annotations: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "done": self.done,
            "stage": self.stage,
            "error": self.error,
            "annotations": self.annotations,
        }


def auto_annotate(
    pil_img: Image.Image,
    recognizer: Recognizer,
    progress: AutoAnnotateProgress | None = None,
    should_cancel=None,
) -> list[dict]:
    """Wykrywa linie i rozpoznaje tekst. Zwraca listę anotacji-propozycji."""
    progress = progress or AutoAnnotateProgress()

    progress.stage = "detecting"
    boxes = detect_lines(pil_img)
    progress.total = len(boxes)
    progress.stage = "recognizing"

    results: list[dict] = []
    batch = 8 if hasattr(recognizer, "recognize_many") else 1
    for start in range(0, len(boxes), batch):
        if should_cancel and should_cancel():
            break

        chunk = boxes[start:start + batch]
        crops = [
            pil_img.crop(
                (
                    max(0, box.x_min - CROP_MARGIN),
                    max(0, box.y_min - CROP_MARGIN),
                    min(pil_img.width, box.x_max + CROP_MARGIN),
                    min(pil_img.height, box.y_max + CROP_MARGIN),
                )
            )
            for box in chunk
        ]
        try:
            texts = (
                recognizer.recognize_many(crops)
                if batch > 1
                else [recognizer.recognize(crops[0])]
            )
        except Exception as e:  # noqa: BLE001
            texts = [""] * len(chunk)
            progress.error = str(e)

        for box, text in zip(chunk, texts):
            ann = box.as_dict()
            ann["label"] = text
            ann["source"] = "auto"
            results.append(ann)

        progress.done += len(chunk)
        progress.annotations = results

    progress.stage = "finished"
    return results
