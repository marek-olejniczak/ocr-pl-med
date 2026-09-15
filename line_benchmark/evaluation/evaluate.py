"""Evaluate predictions (COCO results format) against COCO GT.

Usage (from line_benchmark/):
    python evaluation/evaluate.py --gt instances_test.json \
        --pred results/predictions/<exp_id>/predictions.json \
        --exp-id <exp_id> --out-dir results/
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import (coco_map, detection_metrics, ece,
                               line_metrics, match_greedy)

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
    "ms_per_image_mean",
    "ms_per_image_median",
    "peak_gpu_mem_mb",
    "peak_ram_mb",
]

# numeric fields copied from the predict step's meta.json into the result/summary
META_FIELDS = ("ms_per_image_mean", "ms_per_image_median",
               "peak_gpu_mem_mb", "peak_ram_mb")


def _group_by_image(gt, predictions):
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    return gt_by_img, pred_by_img


def _aggregate(gt, op_preds, all_preds, gt_json_path, image_ids, with_map=True):
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
    if with_map:
        # the expensive part, and it does not depend on the operating point -
        # skip it when sweeping confidence thresholds
        ids = set(image_ids)
        out.update(coco_map(gt_json_path,
                            [p for p in all_preds if p["image_id"] in ids]))
    return out


def _per_source(gt, op_preds, image_ids):
    """Recall per GT source (printed / handwritten / stamp / ...).

    Attribution is per annotation, not per page: one form mixes printed,
    typed and handwritten lines, so labelling a whole page by its first
    annotation says nothing. Only recall-side numbers make sense here - a
    prediction carries no source, so precision has no counterpart.
    """
    gt_by_img, pred_by_img = _group_by_image(gt, op_preds)
    stats = defaultdict(lambda: {"n_gt": 0, "n_matched": 0, "ious": []})
    for img_id in image_ids:
        anns = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        matches = match_greedy([a["bbox"] for a in anns],
                               [p["bbox"] for p in preds],
                               [p["score"] for p in preds]) if preds else []
        matched = {g: iou for g, _, iou in matches}
        for i, a in enumerate(anns):
            st = stats[a.get("source", "?")]
            st["n_gt"] += 1
            if i in matched:
                st["n_matched"] += 1
                st["ious"].append(matched[i])

    out = {}
    for src, st in stats.items():
        n_gt, n_hit = st["n_gt"], st["n_matched"]
        out[src] = {
            "n_gt": n_gt,
            "n_matched": n_hit,
            "recall": n_hit / n_gt if n_gt else 0.0,
            "missed_rate": 1 - n_hit / n_gt if n_gt else 0.0,
            "iou_mean": float(np.mean(st["ious"])) if st["ious"] else 0.0,
            "iou_median": float(np.median(st["ious"])) if st["ious"] else 0.0,
        }
    return out


def evaluate_run(gt_json_path, predictions, conf_thresh=0.25, with_map=True):
    gt = json.loads(Path(gt_json_path).read_text())
    op_preds = [p for p in predictions if p["score"] >= conf_thresh]
    all_ids = [i["id"] for i in gt["images"]]
    result = {
        "conf_thresh": conf_thresh,
        "overall": _aggregate(gt, op_preds, predictions, gt_json_path, all_ids,
                              with_map),
    }

    result["per_source"] = _per_source(gt, op_preds, all_ids)
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
    ap.add_argument("--no-map", action="store_true",
                    help="skip COCO mAP - it ignores --conf-thresh anyway, so "
                         "recomputing it per threshold is wasted work")
    args = ap.parse_args(argv)

    predictions = json.loads(Path(args.pred).read_text())
    result = evaluate_run(args.gt, predictions, conf_thresh=args.conf_thresh,
                          with_map=not args.no_map)

    # inference speed comes from the predict step's meta.json (sibling of the
    # predictions file); fold it into the result + summary so cost sits next to
    # quality - kraken is slow, yolo/surya are cheap, and that matters
    meta_path = Path(args.pred).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k in META_FIELDS:
            if k in meta:
                result["overall"][k] = meta[k]

    metrics_dir = Path(args.out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.exp_id}.json").write_text(json.dumps(result, indent=2))
    _append_summary(args.out_dir, args.exp_id, result)
    o = result["overall"]
    ap50 = f"{o['ap50']:.3f}" if "ap50" in o else "skipped"
    print(f"{args.exp_id}: AP50={ap50} P={o['precision']:.3f} "
          f"R={o['recall']:.3f} F1={o['f1']:.3f} "
          f"missed={o['missed_rate']:.3f}")


if __name__ == "__main__":
    main()
