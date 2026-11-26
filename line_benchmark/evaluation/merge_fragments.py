"""Joins predicted fragments of one text line into a single box.

Two boxes on a page belong to the same line when they overlap vertically by
at least min_v_overlap of the shorter box's height and the horizontal gap
between them is at most max_gap times that height (negative gap = overlap).
Merging is transitive; a merged box is the union, scored with the best part.
"""

from collections import defaultdict

import numpy as np


def _components(adj):
    parent = list(range(len(adj)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(adj, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = defaultdict(list)
    for i in range(len(adj)):
        groups[find(i)].append(i)
    return list(groups.values())


def merge_fragments(preds, min_v_overlap, max_gap):
    """preds: COCO results of one page (bbox xywh, score)."""
    if not 0 <= min_v_overlap <= 1:
        raise ValueError("min_v_overlap is a fraction of the box height")
    if len(preds) < 2:
        return [dict(p) for p in preds]
    b = np.array([p["bbox"] for p in preds], float)
    x0, y0, w, h = b.T
    x1, y1 = x0 + w, y0 + h
    hmin = np.minimum(h[:, None], h[None])
    v_ov = np.clip(np.minimum(y1[:, None], y1[None])
                   - np.maximum(y0[:, None], y0[None]), 0, None) / hmin
    gap = np.maximum(x0[None] - x1[:, None], x0[:, None] - x1[None]) / hmin
    adj = (v_ov >= min_v_overlap) & (gap <= max_gap)

    out = []
    for idx in _components(adj):
        part = b[idx]
        bx0, by0 = part[:, 0].min(), part[:, 1].min()
        bx1 = (part[:, 0] + part[:, 2]).max()
        by1 = (part[:, 1] + part[:, 3]).max()
        best = max(idx, key=lambda i: preds[i]["score"])
        out.append({**preds[best],
                    "bbox": [float(bx0), float(by0),
                             float(bx1 - bx0), float(by1 - by0)]})
    return out


def merge_predictions(preds, min_v_overlap, max_gap):
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return [m for page in by_img.values()
            for m in merge_fragments(page, min_v_overlap, max_gap)]
