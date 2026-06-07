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


def select_viz_images(paths, per_source):
    """Pick up to per_source images per source for visualisation.

    Source = filename prefix before the first underscore (iam_/auto_/seg_ in
    the dev dataset). Deterministic: keeps input order, first N per group.
    """
    groups = {}
    for p in paths:
        key = Path(p).stem.split("_")[0]
        groups.setdefault(key, [])
        if len(groups[key]) < per_source:
            groups[key].append(Path(p))
    return [(p, key) for key, ps in groups.items() for p in ps]


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
                 max_images=100, imgsz=640, sink=None, artifact_logger=None,
                 image_logger=None, viz_per_source=2, viz_every=10,
                 viz_max_dim=1024):
        all_val = sorted(p for p in Path(val_images_dir).iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        # strided sample, not the first N: the list is sorted so the first 100
        # would be one source (auto_*) and bias the metrics. Stride spreads the
        # sample across all sources.
        step = max(1, len(all_val) // max_images) if max_images else 1
        self.val_images = all_val[::step][:max_images] if max_images else all_val
        self.viz_images = select_viz_images(all_val, viz_per_source)
        self.val_labels = Path(val_labels_dir)
        self.conf, self.imgsz = conf, imgsz
        self.sink, self.artifact_logger = sink, artifact_logger
        self.image_logger = image_logger
        self.viz_every, self.viz_max_dim = viz_every, viz_max_dim
        self.tracker = BestTracker(CHECKPOINT_METRICS)

    def _draw(self, im, gt_xywh, pred_xyxy):
        """GT green, predictions red; resize keeping aspect ratio -> RGB."""
        import cv2

        vis = im.copy()
        t = max(2, im.shape[1] // 500)
        for x, y, w, h in gt_xywh:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)),
                          (0, 255, 0), t)
        for x1, y1, x2, y2 in pred_xyxy:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), t)
        m = max(vis.shape[:2])
        if m > self.viz_max_dim:
            s = self.viz_max_dim / m
            vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def __call__(self, trainer):
        import cv2
        from ultralytics import YOLO

        from evaluation.metrics import detection_metrics, ece, line_metrics

        last = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last.exists():
            return
        model = YOLO(str(last))
        totals = {"n_gt": 0, "n_pred": 0, "n_matched": 0,
                  "n_missed": 0, "n_split": 0, "n_merged": 0}
        ious, confs, correct = [], [], []
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
            scores = r.boxes.conf.tolist()
            pred = [[float(a), float(b), float(c - a), float(d - b)]
                    for a, b, c, d in r.boxes.xyxy.tolist()]
            m = line_metrics(gt, pred, scores)
            for k in totals:
                totals[k] += m[k]
            ious.extend(m["matched_ious"])
            matched = set(m["matched_pred_ids"])
            confs.extend(scores)
            correct.extend(1.0 if i in matched else 0.0
                           for i in range(len(scores)))

        n_gt = totals["n_gt"]
        agg = detection_metrics(totals["n_matched"], n_gt, totals["n_pred"])
        agg["missed_rate"] = totals["n_missed"] / n_gt if n_gt else 0.0
        agg["split_rate"] = totals["n_split"] / n_gt if n_gt else 0.0
        agg["merge_rate"] = totals["n_merged"] / n_gt if n_gt else 0.0
        agg["iou_mean"] = float(np.mean(ious)) if ious else 0.0
        agg["iou_median"] = float(np.median(ious)) if ious else 0.0
        agg["ece"] = ece(np.array(confs), np.array(correct))

        for key in self.tracker.update(agg):
            dst = last.parent / f"best_{key}.pt"
            shutil.copy2(last, dst)
            if self.artifact_logger:
                self.artifact_logger(dst, f"best_{key}")
        if self.sink:
            self.sink({"epoch": int(trainer.epoch),
                       **{f"val/{k}": v for k, v in agg.items()}})

        self._log_predictions(trainer, model, cv2)

    def _log_predictions(self, trainer, model, cv2):
        """GT-vs-prediction overlays, a few per source, to wandb. Throttled by
        viz_every (these are media, not free) and always on the last epoch."""
        epoch = int(trainer.epoch)
        last_epoch = epoch + 1 == getattr(trainer, "epochs", epoch + 1)
        if self.image_logger is None or not self.viz_images:
            return
        if epoch % self.viz_every != 0 and not last_epoch:
            return
        import wandb

        by_source = {}
        for path, source in self.viz_images:
            lbl = self.val_labels / f"{path.stem}.txt"
            if not lbl.exists():
                continue
            im = cv2.imread(str(path))
            if im is None:
                continue
            h, w = im.shape[:2]
            gt = read_yolo_labels(lbl, w, h)
            r = model.predict(str(path), conf=self.conf, imgsz=self.imgsz,
                              verbose=False)[0]
            pred = r.boxes.xyxy.tolist()
            vis = self._draw(im, gt, pred)
            by_source.setdefault(f"viz/{source}", []).append(
                wandb.Image(vis, caption=f"{path.stem} | green=GT red=pred"))
        if by_source:
            self.image_logger(by_source)
