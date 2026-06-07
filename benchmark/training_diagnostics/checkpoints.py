"""Multi-metric best-checkpoint tracking + per-epoch val line metrics.

Ultralytics keeps best.pt by its detection fitness (0.9*mAP50-95 + 0.1*mAP50),
which barely penalises a fully missed line - the worst failure for OCR
downstream. ValLineMetrics re-scores the val subset with the SAME line
metrics the benchmark evaluator uses and keeps best_<metric>.pt per metric.
"""

import shutil
from pathlib import Path

import numpy as np

CHECKPOINT_METRICS = {"missed_rate": "min", "iou_median": "max", "f1": "max"}


class BestTracker:
    def __init__(self, spec):
        self.spec, self.best = dict(spec), {}

    def update(self, metrics):
        improved = []
        for key, mode in self.spec.items():
            if key not in metrics:
                continue
            v, cur = metrics[key], self.best.get(key)
            better = cur is None or (v < cur if mode == "min" else v > cur)
            if better:
                self.best[key] = v
                improved.append(key)
        return improved


def read_yolo_labels(txt_path, img_w, img_h):
    """Inverse of coco_to_yolo_lines: YOLO txt -> COCO xywh boxes."""
    boxes = []
    for line in Path(txt_path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                      w * img_w, h * img_h])
    return boxes


class ValLineMetrics:
    """on_fit_epoch_end callback.

    Loads the freshly written last.pt, predicts on a capped val subset,
    computes the same line metrics as the benchmark evaluator, tracks
    per-metric bests and copies best_<metric>.pt on improvement.
    sink(dict) and artifact_logger(path, name) are injected - wandb stays
    out of unit tests.
    """

    def __init__(self, val_images_dir, val_labels_dir, conf=0.25,
                 max_images=100, imgsz=640, sink=None, artifact_logger=None):
        self.val_images = sorted(p for p in Path(val_images_dir).iterdir()
                                 if p.suffix.lower() in
                                 {".jpg", ".jpeg", ".png"})[:max_images]
        self.val_labels = Path(val_labels_dir)
        self.conf, self.imgsz = conf, imgsz
        self.sink, self.artifact_logger = sink, artifact_logger
        self.tracker = BestTracker(CHECKPOINT_METRICS)

    def __call__(self, trainer):
        import cv2
        from ultralytics import YOLO

        from evaluation.metrics import detection_metrics, line_metrics

        last = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last.exists():
            return
        model = YOLO(str(last))
        totals = {"n_gt": 0, "n_pred": 0, "n_matched": 0, "n_missed": 0}
        ious = []
        for img_path in self.val_images:
            lbl = self.val_labels / f"{img_path.stem}.txt"
            if not lbl.exists():
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            gt = read_yolo_labels(lbl, w, h)
            r = model.predict(str(img_path), conf=self.conf, imgsz=self.imgsz,
                              verbose=False)[0]
            pred = [[float(a), float(b), float(c - a), float(d - b)]
                    for a, b, c, d in r.boxes.xyxy.tolist()]
            m = line_metrics(gt, pred, r.boxes.conf.tolist())
            for k in totals:
                totals[k] += m[k]
            ious.extend(m["matched_ious"])

        agg = detection_metrics(totals["n_matched"], totals["n_gt"],
                                totals["n_pred"])
        agg["missed_rate"] = (totals["n_missed"] / totals["n_gt"]
                              if totals["n_gt"] else 0.0)
        agg["iou_median"] = float(np.median(ious)) if ious else 0.0

        for key in self.tracker.update(agg):
            dst = last.parent / f"best_{key}.pt"
            shutil.copy2(last, dst)
            if self.artifact_logger:
                self.artifact_logger(dst, f"best_{key}")
        if self.sink:
            self.sink({"epoch": int(trainer.epoch),
                       **{f"val/{k}": v for k, v in agg.items()}})
