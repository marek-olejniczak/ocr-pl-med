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
