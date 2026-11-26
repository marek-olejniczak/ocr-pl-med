"""CER/WER surowego OCR (bez korekty) dla wszystkich modeli z katalogu predykcji: całość, dev, test i każda grupa.

    python -m evaluation.raw_ocr --preds KATALOG --groups data/NAZWA/groups.csv --out ../results/NAZWA/raw_ocr.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.folds import line_groups, read_group_map, read_split_map
from evaluation.metrics import load_predictions, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gm, sm = read_group_map(a.groups), read_split_map(a.groups)
    out = []
    for f in sorted(a.preds.glob("*.jsonl")):
        rows = load_predictions(f)
        groups = line_groups(rows, "fine", gm)
        subsets = {"all": range(len(rows))}
        if sm:
            for s in ("dev", "test"):
                subsets[s] = [i for i, r in enumerate(rows) if sm[r["file_name"]] == s]
        for g in sorted(set(groups)):
            subsets[g] = [i for i, x in enumerate(groups) if x == g]
        for name, idx in subsets.items():
            s = score([rows[i]["ground_truth"] for i in idx], [rows[i]["prediction"] for i in idx])
            out.append({"model": f.stem, "subset": name, "n": s.n, "cer": f"{s.cer:.6f}", "wer": f"{s.wer:.6f}",
                        "char_edits": s.char_edits, "chars": s.chars})
    if not out:
        raise SystemExit(f"brak *.jsonl w {a.preds}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r["subset"] in ("all", "dev", "test"):
            print(f"{r['model']:34s} {r['subset']:5s} n={r['n']:5d} CER {r['cer']} WER {r['wer']}")


if __name__ == "__main__":
    main()
