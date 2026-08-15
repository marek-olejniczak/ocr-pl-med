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
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path


def dataset_fingerprint(data_yaml):
    """Which images each split holds, as counts + a hash of their names.

    Every dataset is materialized into the same yolo layout, so the path in
    the config cannot tell two of them apart; this can."""
    import yaml
    root = Path(data_yaml).resolve().parent
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    out = {}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        d = root / rel if rel else None
        if d is None or not d.is_dir():
            continue
        names = sorted(p.name for p in d.iterdir())
        out[f"data_{split}_images"] = len(names)
        out[f"data_{split}_md5"] = hashlib.md5(
            "\n".join(names).encode()).hexdigest()[:12]
    return out


def git_commit():
    """Set GIT_COMMIT when running in the container - .git is not mounted."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def ensure_tmpdir():
    """tempfile needs TMPDIR to exist; compose points it into the bind mount."""
    tmp = os.environ.get("TMPDIR")
    if tmp:
        Path(tmp).mkdir(parents=True, exist_ok=True)


def guard_out_dir(out, overwrite=False):
    ckpt = Path(out) / "train" / "weights" / "best.pt"
    if ckpt.exists() and not overwrite:
        raise SystemExit(
            f"{ckpt} already exists - a previous run would be overwritten. "
            "Use a different --out or pass --overwrite.")


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


def _wandb_artifact_logger(path, name):
    import wandb
    if wandb.run is None:
        return
    art = wandb.Artifact(f"{wandb.run.name}-{name}", type="model")
    art.add_file(str(path))
    wandb.log_artifact(art)


def cmd_train(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # line_benchmark/

    ensure_tmpdir()
    guard_out_dir(args.out, args.overwrite)

    # single source of truth for the training hyperparameters: the same dict
    # feeds model.train() and the wandb config, so the config can never drift
    # from what actually ran (e.g. optimizer, cos_lr were hardcoded before and
    # showed up as null in wandb)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        # explicit because rtdetr needs it off: its pipeline runs
        # v8_transforms(stretch=True), which never crops the 2*imgsz mosaic
        # canvas back, so training silently happens at double the resolution
        mosaic=args.mosaic,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        device=args.device,
    )

    provenance = {"git_commit": git_commit(),
                  **dataset_fingerprint(args.data)}
    # same record on disk, so a run stays documented without wandb
    meta = {**vars(args), **train_kwargs, **provenance}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=Path(args.out).name,
            group=args.wandb_group,
            tags=[t for t in (args.wandb_tags or "").split(",") if t],
            config={**vars(args), **train_kwargs, **provenance})

    trainer = None
    if args.diagnostics:
        from training_diagnostics.core import Fanout, JsonlSink
        from training_diagnostics.ultralytics_hooks import pick_trainer
        cls = pick_trainer(args.weights)
        cls.sink = Fanout(
            JsonlSink(Path(args.out) / "train" / "diagnostics.jsonl"),
            wandb_run.log if wandb_run else None)
        cls.probe_every = args.probe_every
        trainer = cls

    model = get_model(args.weights)

    if args.line_val:
        import yaml
        from training_diagnostics.checkpoints import ValLineMetrics
        from training_diagnostics.core import Fanout, JsonlSink
        data = yaml.safe_load(Path(args.data).read_text())
        root = Path(data.get("path", str(Path(args.data).parent)))
        val_rel = str(data.get("val", "images/val"))
        cb = ValLineMetrics(
            val_images_dir=root / val_rel,
            val_labels_dir=root / val_rel.replace("images", "labels"),
            imgsz=args.imgsz,
            max_images=args.line_val_max_images,
            sink=Fanout(
                JsonlSink(Path(args.out) / "train" / "val_metrics.jsonl"),
                wandb_run.log if wandb_run else None),
            image_logger=wandb_run.log if wandb_run else None,
            viz_per_source=args.viz_per_source,
            viz_every=args.viz_every,
            viz_max=args.viz_max)
        model.add_callback("on_fit_epoch_end", cb)

    model.train(
        trainer=trainer,
        # absolute, else ultralytics buries a relative project under
        # runs/detect/<project> and our paths (+ wandb artifacts) miss it
        project=str(Path(args.out).resolve()),
        name="train",
        exist_ok=True,
        **train_kwargs,
    )

    weights_dir = Path(args.out).resolve() / "train" / "weights"
    if wandb_run:
        # upload every checkpoint once at the end. best_<metric>.pt are
        # overwritten in place during training, so these are the final best
        # of each - one wandb artifact version per file instead of a new
        # version on every val improvement (which fills storage).
        for ckpt in sorted(weights_dir.glob("*.pt")):
            _wandb_artifact_logger(ckpt, ckpt.stem)
        wandb_run.finish()
    print(f"best checkpoint: {weights_dir / 'best.pt'}")


def cmd_predict(args):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # line_benchmark/
    from common.resources import reset_gpu_peak, resource_meta

    ensure_tmpdir()

    reset_gpu_peak()
    model = get_model(args.weights)
    coco = json.loads(Path(args.coco).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    predictions, speeds = [], []
    for img in tqdm(coco["images"], desc="predict"):
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
            **resource_meta(),
            "versions": {"ultralytics": ultralytics.__version__,
                         "torch": torch.__version__,
                         "python": platform.python_version()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(predictions)} predictions for {len(coco['images'])} images")


def cmd_lr_find(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # line_benchmark/
    from training_diagnostics.lr_finder import pick_lr_finder

    cls = pick_lr_finder(args.weights)
    cls.lr_min = args.lr_min
    cls.lr_max = args.lr_max
    cls.n_steps = args.steps
    cls.out_path = Path(args.out) / "lr_find.json"

    model = get_model(args.weights)
    try:
        model.train(
            trainer=cls,
            data=args.data,
            epochs=50,             # self.stop ends the sweep much earlier
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="AdamW",
            lr0=args.lr_min,
            lrf=1.0,               # flat scheduler - the sweep owns the LR
            warmup_epochs=0,       # warmup would overwrite per-step LR!
            nbs=args.batch,        # no grad accumulation - step every batch
            cos_lr=False,
            seed=0,
            device=args.device,
            project=str(args.out),
            name="lr_find_run",
            exist_ok=True,
            val=False,
            plots=False,
            save=False,
        )
    except FileNotFoundError:
        # expected: save=False means no checkpoint, which ultralytics'
        # post-train bookkeeping treats as an error; lr_find.json is
        # already written by the sweep's _finish()
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--weights", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr0", type=float, default=0.002)
    t.add_argument("--lrf", type=float, default=0.01)
    t.add_argument("--warmup-epochs", type=float, default=3.0)
    t.add_argument("--mosaic", type=float, default=1.0,
                   help="ultralytics default is 1.0; set 0 for rtdetr")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--diagnostics", action="store_true",
                   help="log per-step optimisation diagnostics to "
                        "train/diagnostics.jsonl")
    t.add_argument("--probe-every", type=int, default=100,
                   help="probe-batch diagnostics interval (0 disables)")
    t.add_argument("--patience", type=int, default=30,
                   help="early-stopping patience (ultralytics fitness)")
    t.add_argument("--line-val", action="store_true",
                   help="per-epoch line metrics on val + best_<metric>.pt")
    t.add_argument("--line-val-max-images", type=int, default=100)
    t.add_argument("--viz-per-source", type=int, default=2,
                   help="GT-vs-pred overlays logged per source (needs --wandb)")
    t.add_argument("--viz-max", type=int, default=10,
                   help="cap on overlays per viz round (media add up)")
    t.add_argument("--viz-every", type=int, default=10,
                   help="log prediction overlays every N epochs")
    t.add_argument("--wandb", action="store_true")
    t.add_argument("--wandb-project", default="line-benchmark")
    t.add_argument("--wandb-group",
                   help="groups runs in the UI, e.g. the dataset name")
    t.add_argument("--wandb-tags", help="comma-separated")
    t.add_argument("--overwrite", action="store_true",
                   help="replace an --out that already holds a checkpoint")
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

    f = sub.add_parser("lr-find")
    f.add_argument("--weights", required=True)
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=200)
    f.add_argument("--lr-min", type=float, default=1e-6)
    f.add_argument("--lr-max", type=float, default=1e-1)
    f.add_argument("--imgsz", type=int, default=640)
    f.add_argument("--batch", type=int, default=64)
    f.add_argument("--device", default=None)
    f.set_defaults(fn=cmd_lr_find)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
