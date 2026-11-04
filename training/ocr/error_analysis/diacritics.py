#!/usr/bin/env python
"""Diagnostyka pomyłek diakrytyk -> baza (ś→s, ż→z, ą→a, ó→o, ę→e).

Dwa tryby:

  model-input   Co widzi Surya? Dla linii z pomyłką odtwarza przetwarzanie wejścia
                (scale_to_fit z img_size zadania ocr_without_boxes + zaokrąglenie do
                wielokrotności patch·merge = 28, INTER_CUBIC) i rysuje wejście modelu
                z siatką tokenów 28 px (4 najniższe, 4 środkowe, 4 najwyższe linie).
                Wypisuje też odsetek gubienia diakrytyków vs wysokość oryginału.
                -> out_ngram/diacritics_model_input.png

  sample        Losowa próbka 16 oryginalnych linii z daną pomyłką, do ręcznej
                oceny (czy autor w ogóle postawił diakrytyk).
                -> out_ngram/diacritics_sample_<znak>.png  (np. _ś, _ż)

Logika skalowania skopiowana z surya 0.17.1 (common/surya/processor:
scale_to_fit, _process_and_tile) — kontener surya-training nie ma matplotlib.

Uruchomienie (z katalogu narzędzia):
    python diacritics.py model-input
    python diacritics.py sample --char ś --seed 1
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import analyze as A

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]

# surya 0.17.1: FoundationPredictor.tasks["ocr_without_boxes"]["img_size"]
IMG_SIZE = (1024, 512)
FACTOR = 28  # patch_size (14) * merge_size (2)
PAIRS = {"ś": "s", "ż": "z", "ą": "a", "ó": "o", "ę": "e"}
BASE_OF = {"ą": "a", "ć": "c", "ę": "e", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
           "Ą": "A", "Ć": "C", "Ę": "E", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z"}
HEIGHT_BINS = [(0, 30, "<30"), (30, 40, "30-39"), (40, 55, "40-54"), (55, 75, "55-74"),
               (75, 10 ** 6, "≥75")]


def scale_to_fit(img, max_size, min_size=(168, 168)):
    """Kopia SuryaOCRProcessor.scale_to_fit (surya 0.17.1)."""
    height, width = img.shape[:2]
    px = width * height
    if px > max_size[0] * max_size[1]:
        f = (max_size[0] * max_size[1] / px) ** 0.5
        size = (math.floor(width * f), math.floor(height * f))
    elif px < min_size[0] * min_size[1]:
        f = (min_size[0] * min_size[1] / px) ** 0.5
        size = (math.ceil(width * f), math.ceil(height * f))
    else:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def model_input(img: np.ndarray) -> tuple[np.ndarray, float]:
    """scale_to_fit + zaokrąglenie do wielokrotności 28 jak w _process_and_tile."""
    h0 = img.shape[0]
    img = scale_to_fit(img, IMG_SIZE)
    h, w = img.shape[:2]
    hb, wb = math.ceil(h / FACTOR) * FACTOR, math.ceil(w / FACTOR) * FACTOR
    if (hb, wb) != (h, w):
        img = cv2.resize(img, (wb, hb), interpolation=cv2.INTER_CUBIC)
    return img, img.shape[0] / h0


def find_hits(rows, images_dir: Path, wanted: dict[str, str]):
    """Linie z pierwszą pomyłką diakrytyk -> baza: (h, file, ref, pred, idx, znak)."""
    hits = []
    for r in rows:
        ref, pred = A.normalize(r["ground_truth"]), A.normalize(r["prediction"])
        for kind, ch, i, j in A.align(ref, pred):
            if kind == A.SUB and wanted.get(ch) == pred[j]:
                with Image.open(images_dir / r["file_name"]) as im:
                    hits.append((im.size[1], r["file_name"], ref, pred, i, ch))
                break
    return hits


def height_table(rows, images_dir: Path) -> None:
    """Odsetek gubienia diakrytyków vs wysokość oryginału (kontrola: litery bez diakrytyków)."""
    st = defaultdict(lambda: [0, 0, 0, 0])  # n diakr., zgubione, n liter, błędy liter
    for r in rows:
        with Image.open(images_dir / r["file_name"]) as im:
            h = im.size[1]
        label = next(lab for lo, hi, lab in HEIGHT_BINS if lo <= h < hi)
        ref, pred = A.normalize(r["ground_truth"]), A.normalize(r["prediction"])
        for kind, ch, i, j in A.align(ref, pred):
            if kind == A.INS:
                continue
            if ch in BASE_OF:
                st[label][0] += 1
                st[label][1] += kind == A.SUB and pred[j] == BASE_OF[ch]
            elif ch.isalpha():
                st[label][2] += 1
                st[label][3] += kind != A.MATCH
    print("wysokość oryg. | n diakr. | diakrytyk→baza | błąd liter bez diakr.")
    for _, _, lab in HEIGHT_BINS:
        n, d, nl, el = st[lab]
        if n and nl:
            print(f"{lab:>14} | {n:8} | {d / n:14.1%} | {el / nl:6.1%}")


def cmd_model_input(args, rows) -> None:
    images_dir = Path(args.images_dir)
    hits = sorted(find_hits(rows, images_dir, PAIRS))
    n = len(hits)
    chosen = hits[:4] + hits[n // 2 - 2:n // 2 + 2] + hits[-4:]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(15, 1.9 * len(chosen)),
                             gridspec_kw={"width_ratios": [4, 1]})
    for (h0, fn, ref, pred, i, ch), (ax_full, ax_zoom) in zip(chosen, axes):
        img = np.asarray(Image.open(images_dir / fn).convert("RGB"))
        inp, scale = model_input(img)
        ax_full.imshow(inp)
        for x in range(0, inp.shape[1], FACTOR):
            ax_full.axvline(x - 0.5, color="#1f77b4", lw=0.3, alpha=0.5)
        for y in range(0, inp.shape[0], FACTOR):
            ax_full.axhline(y - 0.5, color="#1f77b4", lw=0.3, alpha=0.5)
        ax_full.set_title(f"{fn}  orig h={h0}px → wejście {inp.shape[1]}×{inp.shape[0]} (×{scale:.2f})\n"
                          f"GT: {A._window(ref, i)}   OCR: {A._window(pred, i)}",
                          fontsize=7, loc="left")
        ax_full.axis("off")
        # powiększenie: przybliżona pozycja znaku proporcjonalnie do długości tekstu
        cx = int((i + 0.5) / max(len(ref), 1) * inp.shape[1])
        half = max(inp.shape[0], 40)
        x0, x1 = max(0, cx - half), min(inp.shape[1], cx + half)
        ax_zoom.imshow(inp[:, x0:x1], interpolation="nearest")
        ax_zoom.set_title(f"„{ch}” (okolica, ~)", fontsize=7)
        ax_zoom.axis("off")
    fig.suptitle("Wejście modelu Surya przy zgubionym diakrytyku (siatka = tokeny 28 px)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out_dir) / "diacritics_model_input.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"zapisano {out} ({n} linii z pomyłką diakrytyk→baza)")
    height_table(rows, images_dir)


def cmd_sample(args, rows) -> None:
    images_dir = Path(args.images_dir)
    base = BASE_OF[args.char]
    hits = find_hits(rows, images_dir, {args.char: base})
    random.seed(args.seed)
    sample = random.sample(hits, min(16, len(hits)))
    fig, axes = plt.subplots(8, 2, figsize=(16, 16))
    axes = axes.ravel()
    for k, (ax, (_, fn, ref, _, i, _)) in enumerate(zip(axes, sample), start=1):
        lo = ref.rfind(" ", 0, i) + 1
        hi = ref.find(" ", i)
        word = ref[lo:hi if hi >= 0 else len(ref)]
        ax.imshow(Image.open(images_dir / fn).convert("RGB"))
        ax.set_title(f"#{k} słowo: {word}   ({fn[:30]})", fontsize=9, loc="left")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = Path(args.out_dir) / f"diacritics_sample_{args.char}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"zapisano {out} ({len(hits)} linii z „{args.char}”→„{base}”, próbka {len(sample)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(BASE / "data" / "predictions.jsonl"))
    ap.add_argument("--images-dir",
                    default=str(REPO / "benchmark" / "dane" / "handlabeled" / "images"))
    ap.add_argument("--out-dir", default=str(BASE / "out_ngram"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model-input", help="wejście modelu + tabela wg wysokości")
    sp = sub.add_parser("sample", help="losowa próbka linii do ręcznej oceny")
    sp.add_argument("--char", default="ś", choices=sorted(BASE_OF))
    sp.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = A.load_predictions(Path(args.predictions))
    {"model-input": cmd_model_input, "sample": cmd_sample}[args.cmd](args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
