"""Line metrics over a grid of confidence thresholds, one predictions file.

Usage (from line_benchmark/):
    python evaluation/sweep_thresholds.py --gt dataset/annotations/instances_val.json \
        --pred results_val/predictions/<exp_id>/predictions.json \
        --out results_val/thresholds/<exp_id>.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate import _aggregate, _per_source  # noqa: E402

DEFAULT_GRID = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]


def sweep(gt_json_path, predictions, thresholds):
    gt = json.loads(Path(gt_json_path).read_text())
    ids = [i["id"] for i in gt["images"]]
    rows = []
    for t in thresholds:
        op = [p for p in predictions if p["score"] >= t]
        row = {"conf_thresh": t,
               **_aggregate(gt, op, predictions, gt_json_path, ids,
                            with_map=False)}
        for src, st in _per_source(gt, op, ids).items():
            row[f"recall_{src}"] = st["recall"]
        rows.append(row)
    return rows


def best_row(rows, key="f1"):
    return max(rows, key=lambda r: r[key])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds",
                    help="comma-separated; default 0.05..0.95 step 0.05")
    args = ap.parse_args(argv)

    grid = ([float(t) for t in args.thresholds.split(",")]
            if args.thresholds else DEFAULT_GRID)
    rows = sweep(args.gt, json.loads(Path(args.pred).read_text()), grid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    b = best_row(rows)
    print(f"{out.stem}: best F1 {b['f1']:.3f} at {b['conf_thresh']} "
          f"(P={b['precision']:.3f} R={b['recall']:.3f} "
          f"split={b['split_rate']:.3f})")


if __name__ == "__main__":
    main()
