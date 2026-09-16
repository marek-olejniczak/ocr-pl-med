#!/usr/bin/env python
"""Tabela wyników kolejki HP z plików `results/<name>.json`.

Czyta wszystkie JSON-y zapisane przez `eval_cer.py` i wypisuje je posortowane
po CER (rosnąco = od najlepszego). Baseline (v2_200k-lora-r64, lr 2e-5/r64)
nie jest częścią sweepu, więc dopisujemy go jako wiersz odniesienia, żeby
od razu było widać, czy którykolwiek wariant go bije.

Użycie (w kontenerze surya-training, WORKDIR=/app):
    python training/ocr/surya/hp_search/summarize.py training/results/ocr/surya/hp/results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Baseline zmierzony wcześniej na tym samym zbiorze (benchmark `ocr-ablacja-v2`).
BASELINE = {
    "name": "BASELINE v2_200k-lora-r64",
    "lr": "2e-5",
    "rank": "64",
    "cer": 0.127242,
    "wer": None,
    "ema": None,
}


def _lr_rank_from_name(name: str) -> tuple[str, str]:
    """`hp_lr1e-4_r64` -> ("1e-4", "64"); inaczej "-"."""
    if not name.startswith("hp_lr") or "_r" not in name:
        return "-", "-"
    lr_part, rank_part = name[len("hp_lr"):].split("_r", 1)
    return lr_part, rank_part


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    results_dir = Path(argv[1])
    if not results_dir.is_dir():
        print(f"Brak katalogu {results_dir}")
        return 1

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[summarize] pomijam {path.name}: {exc}")
            continue
        metrics = payload.get("metrics", {})
        if "cer" not in metrics:
            print(f"[summarize] pomijam {path.name}: brak metryki cer")
            continue
        name = path.stem
        lr, rank = _lr_rank_from_name(name)
        rows.append({
            "name": name,
            "lr": lr,
            "rank": rank,
            "cer": metrics["cer"],
            "wer": metrics.get("wer"),
            "ema": metrics.get("ema"),
            "count": payload.get("count"),
            "seconds": payload.get("elapsed_seconds"),
        })

    if not rows:
        print(f"Brak wyników w {results_dir} — kolejka jeszcze nic nie policzyła?")
        return 1

    rows.sort(key=lambda row: row["cer"])

    header = f"{'run':22s} {'lr':>6s} {'rank':>4s} {'CER':>9s} {'WER':>9s} {'EMA':>8s} {'n':>5s} {'czas':>7s}"
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for row in rows:
        wer = f"{row['wer']:.6f}" if isinstance(row["wer"], float) else "-"
        ema = f"{row['ema']:.6f}" if isinstance(row["ema"], float) else "-"
        seconds = f"{row['seconds']:.0f}s" if isinstance(row["seconds"], float) else "-"
        print(f"{row['name']:22s} {row['lr']:>6s} {row['rank']:>4s} "
              f"{row['cer']:9.6f} {wer:>9s} {ema:>8s} {str(row['count'] or '-'):>5s} {seconds:>7s}")
    print(line)

    best = rows[0]
    # diff < 0 znaczy, że najlepszy wariant ma NIŻSZY CER, czyli jest lepszy.
    diff = (best["cer"] - BASELINE["cer"]) * 100
    sign = "+" if diff >= 0 else "-"
    print(f"Baseline {BASELINE['name']}: CER {BASELINE['cer']:.6f}")
    print(f"Najlepszy {best['name']}: CER {best['cer']:.6f} "
          f"({sign}{abs(diff):.2f} p.p. vs baseline)")

    if len(rows) >= 2:
        # Rozrzut w obrębie siatki: jeśli jest mniejszy niż różnica do baseline,
        # to sweep nic nie wniósł i nie ma sensu zawężać rundy 1b.
        spread = (rows[-1]["cer"] - rows[0]["cer"]) * 100
        print(f"Rozrzut w siatce: {spread:.2f} p.p. (od {rows[0]['cer']:.4f} do {rows[-1]['cer']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
