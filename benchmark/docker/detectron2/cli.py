"""Train / predict CLI for Detectron2 (Faster R-CNN R50-FPN).

COCO-native: trains directly on the COCO json + images, no converter needed.
Honors the benchmark contract - predict writes COCO results + meta.json, so the
shared evaluator treats it like any other model. Single class (line); we use
detection (Faster R-CNN), not Mask R-CNN, because the GT has no masks.

Usage:
    python cli.py train --weights <model_zoo.yaml> --train-coco ... --val-coco ... \
        --images-root <root> --out results/checkpoints/<exp_id>
    python cli.py predict --weights <model_final.pth> --coco instances_test.json \
        --images-root <root> --out results/predictions/<exp_id>
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

LINE_CATEGORY_ID = 1
# Faster R-CNN R50-FPN 3x; detection baseline, COCO-pretrained
DEFAULT_ZOO = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def xyxy_to_coco(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def instances_to_coco(boxes_xyxy, scores, image_id):
    """Detectron2 instances (xyxy boxes + scores) -> COCO results records."""
    return [{"image_id": int(image_id), "category_id": LINE_CATEGORY_ID,
             "bbox": xyxy_to_coco(b), "score": float(s)}
            for b, s in zip(boxes_xyxy, scores)]


def speed_stats(speeds_ms):
    if not speeds_ms:
        return {"ms_per_image_mean": 0.0, "ms_per_image_median": 0.0}
    return {"ms_per_image_mean": float(statistics.fmean(speeds_ms)),
            "ms_per_image_median": float(statistics.median(speeds_ms))}


def _zoo_id(weights):
    """Accept a full model_zoo path or a short alias for the default."""
    return DEFAULT_ZOO if weights in ("frcnn_r50.yaml", "frcnn", "") else weights


def _base_cfg(args, zoo_config, num_classes=1):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_config))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.MIN_SIZE_TRAIN = (args.imgsz,)
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.MODEL.DEVICE = args.device or "cuda"
    return cfg


def cmd_train(args):
    from detectron2 import model_zoo
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, HookBase
    from detectron2.utils.events import get_event_storage

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project,
                               name=Path(args.out).name, config=vars(args))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, coco in [("bench_train", args.train_coco),
                       ("bench_val", args.val_coco)]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        register_coco_instances(name, {}, coco, args.images_root)

    n_train = len(json.loads(Path(args.train_coco).read_text())["images"])
    iters_per_epoch = max(1, n_train // args.batch)

    zoo = _zoo_id(args.weights)            # for train, --weights is the zoo cfg id
    cfg = _base_cfg(args, zoo)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo)  # COCO-pretrained
    cfg.DATASETS.TRAIN = ("bench_train",)
    cfg.DATASETS.TEST = ("bench_val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr0
    cfg.SOLVER.MAX_ITER = iters_per_epoch * args.epochs
    cfg.SOLVER.STEPS = []          # no LR step decay; keep it simple/comparable
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.OUTPUT_DIR = str(out)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if wandb_run:
        # push detectron2's EventStorage scalars (losses, lr, and bbox/AP from
        # the periodic COCO eval) to wandb. Our 9 line metrics are ultralytics-
        # only during training; detectron2 gets them at the end via predict->eval.
        class _WandbHook(HookBase):
            def __init__(self, period):
                self._period = period

            def after_step(self):
                it = self.trainer.iter
                if (it + 1) % self._period:
                    return
                latest = get_event_storage().latest()
                wandb.log({k: v for k, (v, _) in latest.items()}, step=it)

        trainer.register_hooks([_WandbHook(args.wandb_period)])

    trainer.train()              # writes OUTPUT_DIR/model_final.pth

    ckpt = out / "model_final.pth"
    if wandb_run:
        if ckpt.exists():
            art = wandb.Artifact(f"{wandb_run.name}-model_final", type="model")
            art.add_file(str(ckpt))
            wandb.log_artifact(art)
        wandb_run.finish()
    print(f"best checkpoint: {ckpt}")


def cmd_predict(args):
    from detectron2.engine import DefaultPredictor

    # architecture is fixed (frcnn R50-FPN); --weights is the trained .pth, not
    # a config, so the config comes from DEFAULT_ZOO, weights are set separately
    cfg = _base_cfg(args, DEFAULT_ZOO)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf
    predictor = DefaultPredictor(cfg)

    import cv2

    coco = json.loads(Path(args.coco).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions, speeds = [], []
    for img in coco["images"]:
        im = cv2.imread(str(Path(args.images_root) / img["file_name"]))
        if im is None:
            continue
        t0 = time.perf_counter()
        inst = predictor(im)["instances"].to("cpu")
        speeds.append((time.perf_counter() - t0) * 1000.0)
        boxes = inst.pred_boxes.tensor.tolist()    # xyxy
        scores = inst.scores.tolist()
        predictions.extend(instances_to_coco(boxes, scores, img["id"]))

    (out / "predictions.json").write_text(json.dumps(predictions))

    import detectron2
    import torch
    meta = {"weights": str(args.weights), "device": str(args.device),
            "conf": args.conf, "imgsz": args.imgsz,
            "n_images": len(coco["images"]),
            "n_predictions": len(predictions),
            **speed_stats(speeds),
            "versions": {"detectron2": detectron2.__version__,
                         "torch": torch.__version__,
                         "python": platform.python_version()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(predictions)} predictions for {len(coco['images'])} images")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--weights", required=True)        # model_zoo yaml or alias
    t.add_argument("--train-coco", required=True)
    t.add_argument("--val-coco", required=True)
    t.add_argument("--images-root", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr0", type=float, default=0.0005)
    t.add_argument("--device", default=None)
    t.add_argument("--wandb", action="store_true")
    t.add_argument("--wandb-project", default="line-benchmark")
    t.add_argument("--wandb-period", type=int, default=20,
                   help="log detectron2 metrics to wandb every N iterations")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--coco", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
