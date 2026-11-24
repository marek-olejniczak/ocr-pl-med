"""Predykcje z przebiegu benchmarku (`benchmark/wyniki/<run>/<model>/<dataset>/raw_predictions.csv`) → jsonl
w formacie data/baseline/predictions.jsonl, plus mapa linia → grupa dla test/real/v1.2.

    PYTHONPATH=. python -m evaluation.import_bench --run ~/.cache/.../ocr-v4_20261123_140539 \\
        --dataset-dir ~/.cache/ocr-pl-med/dataset/test/real/v1.2 --out ~/.cache/ocr-pl-med/postprocessing/preds-v4

Etykiety z benchmarku muszą się zgadzać z crops.csv zbioru (inaczej przerywa). Linie [BLANK] wypadają.
jsonl zawiera tekst linii, więc `--out` ma leżeć poza repo; do repo trafia tylko mapa grup (`--groups-out`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.folds import assign_group, source_group
from evaluation.metrics import BLANK

V12_GROUPS = Path(__file__).resolve().parent.parent / "data" / "v12" / "groups.csv"


def line_index(dataset_dir: Path) -> dict[str, dict]:
    """file_name -> {label, group, split} z crops.csv i manifest.csv."""
    pages = {r["page_id"]: r for r in csv.DictReader(open(dataset_dir / "manifest.csv", encoding="utf-8"))}
    out = {}
    for r in csv.DictReader(open(dataset_dir / "crops.csv", encoding="utf-8")):
        fn = Path(r["file_name"]).name
        p = pages[r["line_id"].rsplit("_l", 1)[0]]
        g = source_group(p["original_name"]) if p["document"] == "new" else assign_group(fn)
        out[fn] = {"label": r["label"], "group": g, "split": p["split"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="katalog przebiegu benchmarku")
    ap.add_argument("--dataset-dir", type=Path, required=True, help="np. dataset/test/real/v1.2 (crops.csv, manifest.csv)")
    ap.add_argument("--dataset-id", default="real_v12", help="podkatalog zbioru w przebiegu")
    ap.add_argument("--out", type=Path, required=True, help="katalog na <model>.jsonl (poza repo)")
    ap.add_argument("--suffix", default=".v12")
    ap.add_argument("--groups-out", type=Path, default=V12_GROUPS)
    a = ap.parse_args()
    idx = line_index(a.dataset_dir)
    kept = {fn: v for fn, v in idx.items() if v["label"] != BLANK}
    a.groups_out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.groups_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "group", "split"])
        for fn in sorted(kept):
            w.writerow([fn, kept[fn]["group"], kept[fn]["split"]])
    print(f"grupy: {len(kept)} linii (bez {len(idx) - len(kept)} {BLANK}) -> {a.groups_out}")
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.run.glob(f"*/{a.dataset_id}/raw_predictions.csv")):
        model = f.parent.parent.name
        raw = list(csv.DictReader(open(f, encoding="utf-8")))
        if len(raw) != len(idx):
            raise SystemExit(f"{model}: {len(raw)} linii, zbiór ma {len(idx)}")
        bad = [r["file_name"] for r in raw if idx[r["file_name"]]["label"] != r["ground_truth"]]
        if bad:
            raise SystemExit(f"{model}: etykiety różne od crops.csv, np. {bad[:3]}")
        with open(a.out / f"{model}{a.suffix}.jsonl", "w", encoding="utf-8") as fo:
            for r in raw:
                if r["file_name"] in kept:
                    fo.write(json.dumps({k: r[k] for k in ("file_name", "ground_truth", "prediction")},
                                        ensure_ascii=False) + "\n")
        print(f"{model}: {len(kept)} linii -> {a.out / (model + a.suffix + '.jsonl')}")


if __name__ == "__main__":
    main()
