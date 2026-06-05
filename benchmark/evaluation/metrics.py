"""Model-agnostic detection metrics for line extraction.

All boxes use COCO xywh format: [x_top_left, y_top_left, width, height].
"""
import numpy as np


def _to_xyxy(boxes):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = a.copy()
    out[:, 2] = a[:, 0] + a[:, 2]
    out[:, 3] = a[:, 1] + a[:, 3]
    return out


def intersection_matrix(boxes_a, boxes_b):
    """Pairwise intersection areas, shape (len_a, len_b)."""
    a, b = _to_xyxy(boxes_a), _to_xyxy(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, 0:1], b[None, :, 0])
    y1 = np.maximum(a[:, 1:2], b[None, :, 1])
    x2 = np.minimum(a[:, 2:3], b[None, :, 2])
    y2 = np.minimum(a[:, 3:4], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU, shape (len_a, len_b)."""
    inter = intersection_matrix(boxes_a, boxes_b)
    if inter.size == 0:
        return inter
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_greedy(gt_boxes, pred_boxes, scores, iou_thresh=0.5):
    """One-to-one matching. Preds in descending-score order claim the best
    still-unmatched GT with IoU >= iou_thresh.

    Returns list of (gt_idx, pred_idx, iou).
    """
    ious = iou_matrix(gt_boxes, pred_boxes)
    if ious.size == 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=float))
    matched_gt, matches = set(), []
    for p in order:
        col = ious[:, p].copy()
        col[list(matched_gt)] = -1.0
        g = int(np.argmax(col))
        if col[g] >= iou_thresh:
            matched_gt.add(g)
            matches.append((g, int(p), float(col[g])))
    return matches


def detection_metrics(n_matched, n_gt, n_pred):
    """Precision/recall/F1 from one-to-one match counts."""
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def line_metrics(gt_boxes, pred_boxes, scores, iou_thresh=0.5,
                 part_thresh=0.5, coverage_thresh=0.25):
    """Line-level error analysis for a single image.

    split: GT containing >=2 preds mostly inside it (inter/area_pred >= part_thresh)
    merge: pred containing >=2 GTs mostly inside it (inter/area_gt >= part_thresh);
           merge_rate counts absorbed GTs, so split_rate and merge_rate share a
           denominator (n_gt)
    missed: GT unmatched 1:1, not split, with total coverage < coverage_thresh.
            Coverage sums per-pred intersections so overlapping preds overcount;
            upper bound is fine here - it only makes "missed" stricter.
    """
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    n_gt, n_pred = gt.shape[0], pred.shape[0]

    matches = match_greedy(gt, pred, scores, iou_thresh) if n_pred else []
    matched_gts = {g for g, _, _ in matches}
    matched_ious = [iou for _, _, iou in matches]

    split_ids, absorbed_gts = set(), set()
    coverage = np.zeros(n_gt)
    if n_gt and n_pred:
        inter = intersection_matrix(gt, pred)
        area_gt = gt[:, 2] * gt[:, 3]
        area_pred = pred[:, 2] * pred[:, 3]
        frac_of_pred = inter / np.maximum(area_pred[None, :], 1e-9)
        frac_of_gt = inter / np.maximum(area_gt[:, None], 1e-9)
        coverage = np.minimum(1.0, frac_of_gt.sum(axis=1))
        for g in range(n_gt):
            if (frac_of_pred[g] >= part_thresh).sum() >= 2:
                split_ids.add(g)
        for p in range(n_pred):
            inside = np.where(frac_of_gt[:, p] >= part_thresh)[0]
            if len(inside) >= 2:
                absorbed_gts.update(inside.tolist())

    missed = [g for g in range(n_gt)
              if g not in matched_gts and g not in split_ids
              and coverage[g] < coverage_thresh]

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": len(matches),
        "n_missed": len(missed),
        "n_split": len(split_ids),
        "n_merged": len(absorbed_gts),
        "missed_rate": len(missed) / n_gt if n_gt else 0.0,
        "split_rate": len(split_ids) / n_gt if n_gt else 0.0,
        "merge_rate": len(absorbed_gts) / n_gt if n_gt else 0.0,
        "iou_mean": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "iou_median": float(np.median(matched_ious)) if matched_ious else 0.0,
        "matched_ious": matched_ious,
        "matched_pred_ids": sorted(p for _, p, _ in matches),
    }


def coco_map(gt_json_path, predictions):
    """COCO mAP via pycocotools (reference implementation).

    predictions: list of dicts in COCO results format
    Returns {"ap": AP@[.5:.95], "ap50": ..., "ap75": ..., "ar100": ...}.
    """
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json_path))
        if not predictions:
            return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0, "ar100": 0.0}
        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    # stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    return {"ap": float(s[0]), "ap50": float(s[1]),
            "ap75": float(s[2]), "ar100": float(s[8])}


def ece(confs, corrects, n_bins=10):
    """Expected Calibration Error over equal-width confidence bins."""
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if confs.size == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi) if hi < 1 else (confs >= lo) & (confs <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(corrects[mask].mean() - confs[mask].mean())
    return float(total)
