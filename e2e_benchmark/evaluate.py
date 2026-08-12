"""Score pipeline output against page-level ground truth."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "benchmark" / "src"))
sys.path.insert(0, str(_REPO))

from metrics import HTRMetricsEvaluator  # noqa: E402
from line_benchmark.evaluation.metrics import (  # noqa: E402
    detection_metrics, match_greedy)

import gt as gt_mod  # noqa: E402


def page_eval(gt_lines, results, iou_thresh=0.5):
    gt_boxes = [l.bbox for l in gt_lines]
    pred_boxes = [r.bbox for r in results]
    scores = [r.score for r in results]
    matches = (match_greedy(gt_boxes, pred_boxes, scores, iou_thresh)
               if pred_boxes else [])
    pairs = [(gt_lines[g].text, results[p].text) for g, p, _ in matches]
    return {"n_gt": len(gt_lines),
            "n_pred": len(results),
            "n_matched": len(matches),
            "matches": matches,
            "pairs": pairs,
            "gt_texts": [l.text for l in gt_lines],
            "pred_texts": [r.text for r in results],
            "gt_text": gt_mod.page_text(gt_lines),
            "pred_text": "\n".join(r.text for r in results)}


def summarize(page_evals, seconds_per_page=None):
    ev = HTRMetricsEvaluator()

    doc = ev.evaluate([p["gt_text"] for p in page_evals],
                      [p["pred_text"] for p in page_evals],
                      level="document")

    pairs = [pair for p in page_evals for pair in p["pairs"]]
    lines = ev.evaluate([g for g, _ in pairs], [pr for _, pr in pairs],
                        level="line")

    n_matched = sum(p["n_matched"] for p in page_evals)
    n_gt = sum(p["n_gt"] for p in page_evals)
    n_pred = sum(p["n_pred"] for p in page_evals)
    det = detection_metrics(n_matched, n_gt, n_pred)
    det.update({"n_gt": n_gt, "n_pred": n_pred, "n_matched": n_matched,
                "count": len(page_evals), "level": "detection"})

    summary = {"metrics": {"document": doc,
                           "matched_lines": lines,
                           "detection": det}}
    if seconds_per_page is not None:
        summary["timing"] = {"prediction_seconds_per_page":
                             float(seconds_per_page)}
    return summary
