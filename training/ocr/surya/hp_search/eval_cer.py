#!/usr/bin/env python
"""Ewaluacja adaptera LoRA Surya na zbiorze handlabeled — CER / WER / EMA.

Po co osobny skrypt, skoro jest benchmark? Bo kolejka hiperparametrów
(`run_hp_queue.sh`) musi ocenić ~10 adapterów *przed* tym, jak ktokolwiek
doda je do `experiments.yaml`, a benchmark jest do tego za ciężki: startuje
serwis HTTP, chodzi przez kontener `benchmark` i wymaga wpisu w YAML na każdy
wariant. Tutaj liczymy to samo bezpośrednio w kontenerze treningowym.

KLUCZOWE: ten skrypt liczy DOKŁADNIE to, co benchmark — inaczej wyniki z
kolejki nie byłyby porównywalne z baseline'em (v2_200k -> CER 0.1272).
Odwzorowane 1:1:
  - budowa modelu     : benchmark/docker/surya/app.py::_build_surya_state
  - wywołanie predykcji: app.py::predict_batch (batch po `options.batch_size`,
                        task_name "ocr_without_boxes", math_mode=True,
                        sort_lines=False, return_words=False)
  - czyszczenie tekstu : app.py::_clean_surya_text + _lines_to_text
  - metryki            : benchmark/src/metrics.py::HTRMetricsEvaluator
                         (poziom "line" — tak benchmark liczy handlabeled,
                         bo w experiments.yaml ma `single_words: false`)
  - normalizacja       : " ".join(str(text).strip().split())

Uruchomienie (w kontenerze surya-training, WORKDIR=/app):
    python training/ocr/surya/hp_search/eval_cer.py \
        --adapter training/results/ocr/surya/hp/hp_lr1e-4_r64/adapter \
        --labels-csv benchmark/dane/handlabeled/labels.csv \
        --images-dir benchmark/dane/handlabeled/images \
        --output-json training/results/ocr/surya/hp/results/hp_lr1e-4_r64.json

Do smoke-testu: `--max-samples 50`.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Metryki — 1:1 z benchmark/src/metrics.py (HTRMetricsEvaluator)
# --------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _levenshtein_distance(a: str, b: str) -> int:
    """Odległość edycyjna na znakach (dwie linie pamięci, jak w benchmarku)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _token_levenshtein_distance(ref_tokens: list[str], pred_tokens: list[str]) -> int:
    """Odległość edycyjna na tokenach (słowach)."""
    if ref_tokens == pred_tokens:
        return 0
    if not ref_tokens:
        return len(pred_tokens)
    if not pred_tokens:
        return len(ref_tokens)

    prev = list(range(len(pred_tokens) + 1))
    for i, r_tok in enumerate(ref_tokens, start=1):
        curr = [i]
        for j, p_tok in enumerate(pred_tokens, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if r_tok == p_tok else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def character_error_rate(labels: list[str], predictions: list[str]) -> float:
    """Suma edycji / suma znaków w etykietach (mikro, nie średnia z CER-ów)."""
    total_chars = 0
    total_edits = 0
    for gt, pred in zip(labels, predictions):
        gt_n = _normalize_text(gt)
        pred_n = _normalize_text(pred)
        total_chars += len(gt_n)
        total_edits += _levenshtein_distance(gt_n, pred_n)

    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def word_error_rate(labels: list[str], predictions: list[str]) -> float:
    """Suma edycji na tokenach / suma słów w etykietach."""
    total_words = 0
    total_edits = 0
    for gt, pred in zip(labels, predictions):
        gt_tokens = _normalize_text(gt).split()
        pred_tokens = _normalize_text(pred).split()
        total_words += len(gt_tokens)
        total_edits += _token_levenshtein_distance(gt_tokens, pred_tokens)

    if total_words == 0:
        return 0.0
    return total_edits / total_words


def exact_match_accuracy(labels: list[str], predictions: list[str]) -> float:
    """Udział par, w których znormalizowana predykcja == znormalizowana etykieta."""
    if not labels:
        return 0.0
    matches = sum(
        1
        for gt, pred in zip(labels, predictions)
        if _normalize_text(gt) == _normalize_text(pred)
    )
    return matches / len(labels)


def near_perfect_match_accuracy(
    labels: list[str], predictions: list[str], tolerance: int = 1
) -> float:
    """Udział par z odległością Levenshteina <= tolerance."""
    if not labels:
        return 0.0
    matches = sum(
        1
        for gt, pred in zip(labels, predictions)
        if _levenshtein_distance(_normalize_text(gt), _normalize_text(pred)) <= tolerance
    )
    return matches / len(labels)


def mean_levenshtein_distance(labels: list[str], predictions: list[str]) -> float:
    if not labels:
        return 0.0
    distances = [
        _levenshtein_distance(_normalize_text(gt), _normalize_text(pred))
        for gt, pred in zip(labels, predictions)
    ]
    return float(sum(distances) / len(distances))


# --------------------------------------------------------------------------
# Czyszczenie tekstu Suryi — 1:1 z benchmark/docker/surya/app.py
# --------------------------------------------------------------------------

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")


def _clean_surya_text(raw: str) -> str:
    """Zamienia HTML-ową wydmuszkę Suryi na czysty tekst.

    Wejściem są pojedyncze linie, więc wszystko po `<br>` to halucynacja
    drugiej linii i jest ucinane (tak samo robi benchmark).
    """
    text = _BR_RE.split(raw, maxsplit=1)[0]
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def _lines_to_text(result) -> str:
    text_lines = getattr(result, "text_lines", []) or []
    parts = [_clean_surya_text(str(getattr(line, "text", ""))) for line in text_lines]
    return " ".join(part for part in parts if part)


# --------------------------------------------------------------------------
# Dane wejściowe — 1:1 z benchmark/src/data_generator.py
# --------------------------------------------------------------------------


def _extract_ids(file_name: str) -> tuple[str, str]:
    """`data11_line1.jpg` -> ("data11_line1", "0").

    Ta sama reguła co `data_generator._extract_ids`: dopasowuje tylko końcówkę
    `_<cyfry>`, więc `data11_line1` (końcówka nie jest liczbą) wpada w gałąź
    zapasową. Znaczenie: dla handlabeled każda próbka jest własną grupą na
    poziomie "line", czyli agregacja jest identycznością.
    """
    stem = Path(file_name).stem
    match = re.match(r"^(?P<doc>.+)_(?P<line>\d+)$", stem)
    if match:
        return match.group("doc"), match.group("line")
    return stem, "0"


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    """Indeks nazwa pliku -> ścieżka, z wariantami NFC/NFD (polskie znaki)."""
    index: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for path in images_dir.glob(f"**/{pattern}"):
            name = path.name
            for key in (name, unicodedata.normalize("NFC", name), unicodedata.normalize("NFD", name)):
                index.setdefault(key, path)
    return index


def load_samples(
    labels_csv: Path, images_dir: Path, max_samples: int | None
) -> list[tuple[str, Path, str]]:
    """Zwraca [(file_name, image_path, ground_truth)] — pomija wiersze bez obrazu."""
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Nie ma {labels_csv}. Na serwerze to katalog z benchmarku "
            "(benchmark/dane/ jest w .gitignore, więc nie ma go w repo)."
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"Nie ma katalogu obrazów {images_dir}")

    index = _build_image_index(images_dir)

    samples: list[tuple[str, Path, str]] = []
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file_name" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV musi zawierać kolumny: file_name, label")
        for row in reader:
            file_name = str(row["file_name"])
            image_path = (
                index.get(file_name)
                or index.get(unicodedata.normalize("NFC", file_name))
                or index.get(unicodedata.normalize("NFD", file_name))
            )
            if image_path is None or not image_path.exists():
                continue
            samples.append((file_name, image_path, str(row["label"])))

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    return samples


def _aggregate_line_level(
    rows: list[tuple[str, str, str]]
) -> tuple[list[str], list[str]]:
    """Odtwarza `HTRMetricsEvaluator._aggregate_for_level(level="line")`.

    Benchmark grupuje po (document_id, line_id) i skleja teksty w grupie
    spacją. Dla handlabeled klucze są unikalne (2160/2160), więc wynik jest
    identyczny z liczeniem parami — ale odtwarzamy grupowanie, żeby liczba
    nie rozjechała się cicho na innym zbiorze z kolizjami kluczy.
    """
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for file_name, ground_truth, prediction in rows:
        grouped.setdefault(_extract_ids(file_name), []).append((ground_truth, prediction))

    labels: list[str] = []
    predictions: list[str] = []
    for pairs in grouped.values():
        labels.append(" ".join(gt for gt, _ in pairs))
        predictions.append(" ".join(pred for _, pred in pairs))
    return labels, predictions


# --------------------------------------------------------------------------
# Model — 1:1 z benchmark/docker/surya/app.py::_build_surya_state
# --------------------------------------------------------------------------


def _resolve_adapter_path(raw: str) -> Path:
    """Przyjmuje katalog adaptera ALBO katalog runu (wtedy wchodzi w `adapter/`)."""
    path = Path(raw)
    if (path / "adapter_config.json").is_file():
        return path
    nested = path / "adapter"
    if (nested / "adapter_config.json").is_file():
        return nested
    raise FileNotFoundError(
        f"{path} nie wygląda na adapter LoRA (brak adapter_config.json "
        f"w {path} ani w {nested})"
    )


TRAIN_CONFIG_KEYS = (
    "run_name", "learning_rate", "lora_rank", "lora_alpha", "lora_dropout",
    "lora_target_modules", "max_steps", "batch_size", "weight_decay",
    "warmup_ratio", "lr_scheduler_type", "seed",
)


def _read_train_config(adapter_path: Path) -> dict:
    """Konfiguracja treningu z `adapter/meta.json` (zapisuje ją cli.py).

    Trafia do wyniku tylko po to, żeby tabela i archiwum wyników niosły
    parametry runu (nazwa runu nie zawsze je koduje). Metryki jej nie używają,
    więc brak pliku (albo brak klucza w starszym adapterze, np. `seed`) jest
    niegroźny — po prostu pomijamy.
    """
    meta_path = adapter_path / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    config = meta.get("config") or {}
    return {key: config[key] for key in TRAIN_CONFIG_KEYS if key in config}


def _build_state(adapter_path: Path, device: str, batch_size: int, cache_dir: str):
    """Buduje RecognitionPredictor z adapterem LoRA.

    Env ustawiamy PRZED importem surya — biblioteka czyta konfigurację
    w momencie importu/inicjalizacji (tak samo robi app.py w benchmarku).
    """
    os.environ.setdefault("MODEL_CACHE_DIR", cache_dir)
    os.environ.setdefault("TORCH_DEVICE", device)
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(batch_size))

    try:
        foundation_module = importlib.import_module("surya.foundation")
        recognition_module = importlib.import_module("surya.recognition")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Brakuje zależności Surya OCR. Ten skrypt ma chodzić w kontenerze "
            f"surya-training. Szczegóły: {exc}"
        ) from exc

    foundation_predictor = foundation_module.FoundationPredictor()

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PEFT nie jest zainstalowany (pip install peft)") from exc

    foundation_predictor.model = PeftModel.from_pretrained(
        foundation_predictor.model, str(adapter_path)
    )
    recognition_predictor = recognition_module.RecognitionPredictor(foundation_predictor)
    return recognition_predictor


# --------------------------------------------------------------------------
# Główna pętla
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CER/WER/EMA adaptera LoRA Surya na handlabeled (parity z benchmarkiem)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--adapter", required=True,
                    help="Katalog adaptera LoRA (albo katalog runu z adapter/)")
    ap.add_argument("--labels-csv", default="benchmark/dane/handlabeled/labels.csv")
    ap.add_argument("--images-dir", default="benchmark/dane/handlabeled/images")
    ap.add_argument("--output-json", default=None,
                    help="Gdzie zapisać metryki (domyślnie tylko stdout)")
    ap.add_argument("--dump-predictions", default=None,
                    help="Opcjonalny plik JSONL z predykcjami per linia (do diagnozy)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Ogranicz liczbę próbek (smoke test)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Tyle samo, ile `batch_size` wpisu surya w experiments.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=os.environ.get("MODEL_CACHE_DIR", "modele/cache/surya"))
    ap.add_argument("--task-name", default="ocr_without_boxes")
    ap.add_argument("--disable-math", action="store_true", default=False,
                    help="W experiments.yaml surya ma disable_math: false")
    ap.add_argument("--progress-every", type=int, default=200)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    adapter_path = _resolve_adapter_path(args.adapter)
    samples = load_samples(Path(args.labels_csv), Path(args.images_dir), args.max_samples)
    if not samples:
        print("BŁĄD: zero próbek do oceny", file=sys.stderr)
        return 1

    print(f"[eval_cer] adapter     : {adapter_path}")
    print(f"[eval_cer] próbek      : {len(samples)}")
    print(f"[eval_cer] batch/device: {args.batch_size} / {args.device}")
    print(f"[eval_cer] math_mode   : {not args.disable_math}")

    from PIL import Image  # noqa: PLC0415 — dopiero po ustawieniu env

    started_at = time.perf_counter()
    recognition_predictor = _build_state(
        adapter_path, args.device, args.batch_size, args.cache_dir
    )
    print(f"[eval_cer] model gotowy po {time.perf_counter() - started_at:.1f}s")

    rows: list[tuple[str, str, str]] = []  # (file_name, ground_truth, prediction)
    inference_started_at = time.perf_counter()

    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        images = []
        bboxes = []
        for _file_name, image_path, _gt in chunk:
            with Image.open(image_path) as img:
                rgb_image = img.convert("RGB")
                images.append(rgb_image)
                width, height = rgb_image.size
                # Wejściem są już wycinki linii, więc bbox obejmuje cały obraz.
                bboxes.append([[0, 0, width, height]])

        results = recognition_predictor(
            images,
            task_names=[args.task_name] * len(chunk),
            bboxes=bboxes,
            math_mode=not args.disable_math,
            recognition_batch_size=args.batch_size,
            sort_lines=False,
            return_words=False,
        )

        for index, (_file_name, _image_path, ground_truth) in enumerate(chunk):
            if results and index < len(results) and results[index]:
                prediction = _lines_to_text(results[index])
            else:
                prediction = ""
            rows.append((chunk[index][0], ground_truth, prediction))

        done = min(start + args.batch_size, len(samples))
        if args.progress_every and (done % args.progress_every == 0 or done == len(samples)):
            elapsed = time.perf_counter() - inference_started_at
            so_far = character_error_rate(
                [_normalize_text(r[1]) for r in rows],
                [_normalize_text(r[2]) for r in rows],
            )
            eta = elapsed / done * (len(samples) - done) if done else 0.0
            print(
                f"[eval_cer] {done}/{len(samples)}  CER(cząstk.)={so_far:.4f}  "
                f"upłynęło {elapsed:.0f}s  ETA {eta:.0f}s",
                flush=True,
            )

    inference_seconds = time.perf_counter() - inference_started_at

    labels, predictions = _aggregate_line_level(rows)
    metrics = {
        "cer": character_error_rate(labels, predictions),
        "wer": word_error_rate(labels, predictions),
        "ema": exact_match_accuracy(labels, predictions),
        "near_perfect_match": near_perfect_match_accuracy(labels, predictions, tolerance=1),
        "levenshtein_distance": mean_levenshtein_distance(labels, predictions),
    }

    payload = {
        "adapter": str(adapter_path),
        "train_config": _read_train_config(adapter_path),
        "labels_csv": args.labels_csv,
        "images_dir": args.images_dir,
        "task_name": args.task_name,
        "math_mode": not args.disable_math,
        "batch_size": args.batch_size,
        "count": len(labels),
        "samples_loaded": len(samples),
        "metrics_level": "line",  # tak benchmark liczy handlabeled (single_words: false)
        "metrics": metrics,
        "elapsed_seconds": round(time.perf_counter() - started_at, 1),
        "inference_seconds": round(inference_seconds, 1),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    print("\n=== WYNIK ===")
    print(f"  CER  = {metrics['cer']:.6f}")
    print(f"  WER  = {metrics['wer']:.6f}")
    print(f"  EMA  = {metrics['ema']:.6f}")
    print(f"  liczba próbek = {payload['count']}  (czas {payload['elapsed_seconds']}s)")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  zapisano: {out_path}")

    if args.dump_predictions:
        dump_path = Path(args.dump_predictions)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as handle:
            for file_name, ground_truth, prediction in rows:
                handle.write(json.dumps(
                    {"file_name": file_name, "ground_truth": ground_truth, "prediction": prediction},
                    ensure_ascii=False,
                ) + "\n")
        print(f"  predykcje: {dump_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
