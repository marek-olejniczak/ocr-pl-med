"""Evaluate predictions (COCO results format) against COCO GT.

Usage (from benchmark/):
    python evaluation/evaluate.py --gt instances_test.json \
        --pred results/predictions/<exp_id>/predictions.json \
        --exp-id <exp_id> --out-dir results/
"""

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import coco_map, detection_metrics, ece, line_metrics

SUMMARY_COLUMNS = [
    "exp_id",
    "conf_thresh",
    "ap",
    "ap50",
    "ap75",
    "ar100",
    "precision",
    "recall",
    "f1",
    "missed_rate",
    "split_rate",
    "merge_rate",
    "iou_mean",
    "iou_median",
    "ece",
    "n_images",
    "n_gt",
    "n_pred",
]


def _group_by_image(gt, predictions):
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    return gt_by_img, pred_by_img


def _aggregate(gt, op_preds, all_preds, gt_json_path, image_ids):
    """Micro-averaged metrics over the given image subset.

    Matching happens per image - boxes from different pages can never pair up.
    Line/detection/calibration metrics use op_preds (operating point, already
    filtered by confidence); COCO mAP uses all_preds - the PR curve needs the
    low-confidence detections too.
    """
    gt_by_img, pred_by_img = _group_by_image(gt, op_preds)
    totals = defaultdict(float)
    all_ious, all_confs, all_correct = [], [], []
    n_missed = n_split = n_merged = 0

    for img_id in image_ids:
        anns = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        boxes_gt = [a["bbox"] for a in anns]
        boxes_pr = [p["bbox"] for p in preds]
        scores = [p["score"] for p in preds]
        m = line_metrics(boxes_gt, boxes_pr, scores)
        totals["n_gt"] += m["n_gt"]
        totals["n_pred"] += m["n_pred"]
        totals["n_matched"] += m["n_matched"]
        n_missed += m["n_missed"]
        n_split += m["n_split"]
        n_merged += m["n_merged"]
        all_ious.extend(m["matched_ious"])
        # ECE: a pred is "correct" iff it got a one-to-one match (TP)
        matched = set(m["matched_pred_ids"])
        all_confs.extend(scores)
        all_correct.extend(1.0 if i in matched else 0.0 for i in range(len(preds)))

    n_gt, n_pred = int(totals["n_gt"]), int(totals["n_pred"])
    out = detection_metrics(int(totals["n_matched"]), n_gt, n_pred)
    out.update(
        {
            "missed_rate": n_missed / n_gt if n_gt else 0.0,
            "split_rate": n_split / n_gt if n_gt else 0.0,
            "merge_rate": n_merged / n_gt if n_gt else 0.0,
            "iou_mean": float(np.mean(all_ious)) if all_ious else 0.0,
            "iou_median": float(np.median(all_ious)) if all_ious else 0.0,
            "ece": ece(np.array(all_confs), np.array(all_correct)),
            "n_images": len(image_ids),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_matched": int(totals["n_matched"]),
        }
    )
    ids = set(image_ids)
    out.update(coco_map(gt_json_path, [p for p in all_preds if p["image_id"] in ids]))
    return out


def _image_sources(gt):
    """Map image_id -> source (from its annotations; '?' if absent)."""
    src = {}
    for a in gt["annotations"]:
        src.setdefault(a["image_id"], a.get("source", "?"))
    return src


def _subset_gt_file(gt, image_ids, tmp_dir):
    ids = set(image_ids)
    sub = {
        **gt,
        "images": [i for i in gt["images"] if i["id"] in ids],
        "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
    }
    path = Path(tmp_dir) / "gt_subset.json"
    path.write_text(json.dumps(sub))
    return path


def evaluate_run(gt_json_path, predictions, conf_thresh=0.25):
    gt = json.loads(Path(gt_json_path).read_text())
    op_preds = [p for p in predictions if p["score"] >= conf_thresh]
    all_ids = [i["id"] for i in gt["images"]]
    result = {
        "conf_thresh": conf_thresh,
        "overall": _aggregate(gt, op_preds, predictions, gt_json_path, all_ids),
    }

    sources = _image_sources(gt)
    by_source = defaultdict(list)
    for img_id, s in sources.items():
        by_source[s].append(img_id)
    result["per_source"] = {}
    if len(by_source) > 1:
        with tempfile.TemporaryDirectory() as td:
            for s, ids in by_source.items():
                sub_path = _subset_gt_file(gt, ids, td)
                result["per_source"][s] = _aggregate(
                    gt, op_preds, predictions, sub_path, ids)
    return result


def _append_summary(out_dir, exp_id, result):
    path = Path(out_dir) / "summary.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"exp_id": exp_id, "conf_thresh": result["conf_thresh"],
                    **result["overall"]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--conf-thresh", type=float, default=0.25,
                    help="operating point for line/detection metrics; "
                         "mAP always uses all predictions")
    args = ap.parse_args(argv)

    predictions = json.loads(Path(args.pred).read_text())
    result = evaluate_run(args.gt, predictions, conf_thresh=args.conf_thresh)

    metrics_dir = Path(args.out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.exp_id}.json").write_text(json.dumps(result, indent=2))
    _append_summary(args.out_dir, args.exp_id, result)
    print(
        f"{args.exp_id}: AP50={result['overall']['ap50']:.3f} "
        f"F1={result['overall']['f1']:.3f} "
        f"missed={result['overall']['missed_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
