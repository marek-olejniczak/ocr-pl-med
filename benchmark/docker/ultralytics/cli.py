"""Train / predict CLI for ultralytics models (YOLOv8, YOLO11, RT-DETR).

Runs identically inside the benchmark Docker image and on a dev machine.
Training pins every relevant hyperparameter explicitly - notably the
optimizer; 'auto' silently overrides lr0 (AdamW lr=0.002 picked for the
nano models after the notebook lr/warmup debugging).

Usage:
    python cli.py train --weights yolov8n.pt --data .../data.yaml \
        --out results/checkpoints/yolov8_ft-raw [--epochs 100 ...]
    python cli.py predict --weights .../best.pt --coco instances_test.json \
        --images-root <dataset root> --out results/predictions/<exp_id>
"""

import argparse
import json
import platform
import statistics
from pathlib import Path


def is_rtdetr(weights):
    return "rtdetr" in Path(weights).name.lower()


def get_model(weights):
    """RT-DETR has its own ultralytics class; YOLO covers v8/v11."""
    from ultralytics import RTDETR, YOLO
    return RTDETR(weights) if is_rtdetr(weights) else YOLO(weights)


def xyxy_to_coco(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def speed_stats(speeds_ms):
    if not speeds_ms:
        return {"ms_per_image_mean": 0.0, "ms_per_image_median": 0.0}
    return {"ms_per_image_mean": float(statistics.fmean(speeds_ms)),
            "ms_per_image_median": float(statistics.median(speeds_ms))}


def cmd_train(args):
    trainer = None
    if args.diagnostics:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # benchmark/
        from training_diagnostics.core import JsonlSink
        from training_diagnostics.ultralytics_hooks import pick_trainer
        cls = pick_trainer(args.weights)
        cls.sink = JsonlSink(Path(args.out) / "train" / "diagnostics.jsonl")
        cls.probe_every = args.probe_every
        trainer = cls

    model = get_model(args.weights)
    model.train(
        trainer=trainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=str(args.out),
        name="train",
        exist_ok=True,
    )
    print(f"best checkpoint: {Path(args.out) / 'train' / 'weights' / 'best.pt'}")


def cmd_predict(args):
    model = get_model(args.weights)
    coco = json.loads(Path(args.coco).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    predictions, speeds = [], []
    for img in coco["images"]:
        path = Path(args.images_root) / img["file_name"]
        result = model.predict(str(path), conf=args.conf, imgsz=args.imgsz,
                               max_det=args.max_det, device=args.device,
                               verbose=False)[0]
        speeds.append(sum(result.speed.values()))      # pre+infer+post [ms]
        for box, score in zip(result.boxes.xyxy.tolist(),
                              result.boxes.conf.tolist()):
            predictions.append({"image_id": img["id"], "category_id": 1,
                                "bbox": xyxy_to_coco(box),
                                "score": float(score)})

    (out / "predictions.json").write_text(json.dumps(predictions))

    import torch
    import ultralytics
    meta = {"weights": str(args.weights), "device": str(args.device),
            "conf": args.conf, "imgsz": args.imgsz,
            "n_images": len(coco["images"]),
            "n_predictions": len(predictions),
            **speed_stats(speeds),
            "versions": {"ultralytics": ultralytics.__version__,
                         "torch": torch.__version__,
                         "python": platform.python_version()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(predictions)} predictions for {len(coco['images'])} images")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--weights", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr0", type=float, default=0.002)
    t.add_argument("--lrf", type=float, default=0.01)
    t.add_argument("--warmup-epochs", type=float, default=3.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--diagnostics", action="store_true",
                   help="log per-step optimisation diagnostics to "
                        "train/diagnostics.jsonl")
    t.add_argument("--probe-every", type=int, default=100,
                   help="probe-batch diagnostics interval (0 disables)")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--coco", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
