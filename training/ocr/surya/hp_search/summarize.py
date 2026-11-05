#!/usr/bin/env python
"""Tabela wyników kolejki HP z plików `results/<name>.json`.

Czyta wszystkie JSON-y zapisane przez `eval_cer.py` i wypisuje je posortowane
po CER (rosnąco = od najlepszego). Baseline (v2_200k-lora-r64, lr 2e-5/r64)
nie jest częścią sweepu, więc dopisujemy go jako wiersz odniesienia, żeby
od razu było widać, czy którykolwiek wariant go bije.

Kolumna `config` bierze się z `adapter/meta.json` (zapisuje ją `cli.py`) —
dzięki temu wiersz niesie prawdziwe lr/rank/alpha/dropout/seed, a nie to, co
uda się wywnioskować z nazwy runu. Gdy mety brak (albo wynik jest starszy niż
to pole), wchodzi odczyt mety z dysku po ścieżce `adapter` z wyniku, a gdy i to
się nie uda — "-" (metryki zostają poprawne, ginie tylko opis).

Gdy w wynikach jest replikat z innym seedem (REPLICATE_NAME), skrypt dopisuje
zmierzony rozrzut między runami i mówi wprost, czy zwycięzca jest poza nim.

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

# Run o konfiguracji baseline'u, wytrenowany z innym seedem (runda 1b).
# Różnica CER względem baseline'u to zmierzony szum między runami.
REPLICATE_NAME = "hp1b_seed1234"

try:  # ta sama logika, która zapisuje `train_config` do wyniku
    from eval_cer import _read_train_config
except ImportError:  # uruchomione z innego katalogu — opis konfiguracji padnie
    def _read_train_config(_adapter_path: Path) -> dict:
        return {}


def _config_label(name: str, config: dict) -> str:
    """Zwięzły opis konfiguracji: `lr 2e-5 r64 a64 d0.2 seed 1234 20000 krokow`."""
    if not config:
        return "-"
    bits = []
    if "learning_rate" in config:
        bits.append(f"lr {config['learning_rate']}")
    if "lora_rank" in config:
        bits.append(f"r{config['lora_rank']}")
    if "lora_alpha" in config:
        bits.append(f"a{config['lora_alpha']}")
    if "lora_dropout" in config:
        bits.append(f"d{config['lora_dropout']}")
    # seed 42 i 10k kroków to wartości domyślne — pokazujemy tylko odstępstwa
    if config.get("seed") not in (None, "", "42"):
        bits.append(f"seed {config['seed']}")
    if config.get("max_steps") not in (None, "", "10000", "-1"):
        bits.append(f"{config['max_steps']} krokow")
    return " ".join(bits) if bits else name


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
        config = payload.get("train_config")
        if not config and payload.get("adapter"):
            config = _read_train_config(Path(payload["adapter"]))
        rows.append({
            "name": name,
            "config": _config_label(name, config or {}),
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

    header = (f"{'run':18s} {'config':30s} {'CER':>9s} {'WER':>9s} "
              f"{'EMA':>8s} {'n':>5s} {'czas':>7s}")
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for row in rows:
        wer = f"{row['wer']:.6f}" if isinstance(row["wer"], float) else "-"
        ema = f"{row['ema']:.6f}" if isinstance(row["ema"], float) else "-"
        seconds = f"{row['seconds']:.0f}s" if isinstance(row["seconds"], float) else "-"
        print(f"{row['name']:18s} {row['config']:30s} "
              f"{row['cer']:9.6f} {wer:>9s} {ema:>8s} {str(row['count'] or '-'):>5s} {seconds:>7s}")
    print(line)

    # Szum między runami: replikat baseline'u z innym seedem, jeśli jest policzony.
    replicate = next((row for row in rows if row["name"] == REPLICATE_NAME), None)
    noise_pp = None
    if replicate is not None:
        noise_pp = abs(replicate["cer"] - BASELINE["cer"]) * 100
        print(f"Replikat {REPLICATE_NAME} (baseline + inny seed): CER {replicate['cer']:.6f} "
              f"-> rozrzut między runami {noise_pp:.2f} p.p.")

    best = rows[0]
    # diff < 0 znaczy, że najlepszy wariant ma NIŻSZY CER, czyli jest lepszy.
    diff = (best["cer"] - BASELINE["cer"]) * 100
    sign = "+" if diff >= 0 else "-"
    verdict = ""
    if noise_pp is not None and best["name"] != REPLICATE_NAME:
        inside = "w granicach szumu" if abs(diff) <= noise_pp else "POZA szumem"
        verdict = f"  <- {inside} ({noise_pp:.2f} p.p.)"
    print(f"Baseline {BASELINE['name']}: CER {BASELINE['cer']:.6f}")
    print(f"Najlepszy {best['name']}: CER {best['cer']:.6f} "
          f"({sign}{abs(diff):.2f} p.p. vs baseline){verdict}")

    if len(rows) >= 2:
        # Rozrzut w obrębie siatki: jeśli jest mniejszy niż różnica do baseline,
        # to sweep nic nie wniósł i nie ma sensu zawężać rundy 1b.
        spread = (rows[-1]["cer"] - rows[0]["cer"]) * 100
        print(f"Rozrzut w tabeli: {spread:.2f} p.p. (od {rows[0]['cer']:.4f} do {rows[-1]['cer']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
