"""CER / WER mikro, zgodne 1:1 z `training/ocr/surya/hp_search/eval_cer.py`.

eval_cer.py (i benchmark) normalizuje tylko białe znaki: `" ".join(t.strip().split())`.
To jest metryka GŁÓWNA (`raw`). NFC na naszych danych nic nie zmienia (sprawdza to test),
więc nie dokładamy go do metryki głównej, żeby liczby były porównywalne z benchmarkiem.

Metryka POMOCNICZA (`aux`): casefold + usunięcie interpunkcji, tylko do raportu
(ticket 01); nigdy nie stosujemy jej do samego tekstu wyjściowego korektora.

Levenshtein z rapidfuzz daje te same odległości co pętla z eval_cer.py (jednostkowe
koszty), tylko szybciej; test porównuje oba na całym baseline.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from rapidfuzz.distance import Levenshtein

from ocr_postprocess.resources import cache_dir


def normalize_ws(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_aux(text: str) -> str:
    """Casefold + bez interpunkcji/symboli (kategorie Unicode P*, S*), potem białe znaki."""
    t = unicodedata.normalize("NFC", str(text)).casefold()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return normalize_ws(t)


@dataclass(frozen=True)
class Scores:
    cer: float
    wer: float
    char_edits: int
    chars: int
    word_edits: int
    words: int
    n: int


def line_char_errors(gt: str, pred: str, norm: Callable[[str], str] = normalize_ws) -> tuple[int, int]:
    g, p = norm(gt), norm(pred)
    return Levenshtein.distance(g, p), len(g)


def line_word_errors(gt: str, pred: str, norm: Callable[[str], str] = normalize_ws) -> tuple[int, int]:
    g, p = norm(gt).split(), norm(pred).split()
    return Levenshtein.distance(g, p), len(g)


def score(labels: Iterable[str], preds: Iterable[str], norm: Callable[[str], str] = normalize_ws) -> Scores:
    ce = cn = we = wn = n = 0
    for gt, pred in zip(labels, preds, strict=True):
        e, c = line_char_errors(gt, pred, norm)
        ce += e
        cn += c
        e, w = line_word_errors(gt, pred, norm)
        we += e
        wn += w
        n += 1
    return Scores(
        cer=ce / cn if cn else 0.0,
        wer=we / wn if wn else 0.0,
        char_edits=ce, chars=cn, word_edits=we, words=wn, n=n,
    )


BASELINE_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset-v1"
# Predykcje dataset-v1 zawierają tekst ręcznych notatek, więc leżą poza repo.
BASELINE_PREDICTIONS = cache_dir() / "dataset-v1" / "predictions.jsonl"


# Linie bez tekstu (np. 68 w miesnie, test/real/v1.1 i v1.2) zostają w zbiorze dla testów detekcji;
# w ocenie OCR i korektora je pomijamy.
BLANK = "[BLANK]"


def load_predictions(path: Path | None = None) -> list[dict]:
    with open(path or BASELINE_PREDICTIONS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if normalize_ws(r["ground_truth"]) != BLANK]


if __name__ == "__main__":
    rows = load_predictions()
    gts = [r["ground_truth"] for r in rows]
    prs = [r["prediction"] for r in rows]
    for name, fn in (("raw", normalize_ws), ("aux", normalize_aux)):
        s = score(gts, prs, fn)
        print(f"{name}: n={s.n} CER={s.cer:.5f} ({s.char_edits}/{s.chars}) "
              f"WER={s.wer:.5f} ({s.word_edits}/{s.words})")
