"""Run the benchmark experiment matrix declared in experiments.yaml.

The matrix is generated programmatically from models x data variants:
    train:   {model}_ft-{variant}
    predict: {model}_ft-{variant}_eval-{variant} / {model}_zeroshot_eval-{variant}
    eval:    same ids as predict
Each stage is idempotent - a job whose artifact already exists is skipped
(--force re-runs). Failures are logged to results/run_log.csv and the runner
moves on to the next job.

Usage (from line_benchmark/):
    python orchestrator/run_experiments.py --dry-run
    python orchestrator/run_experiments.py --only yolov8 --stage train
    python orchestrator/run_experiments.py --local        # no docker
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from common.batching import describe, effective_batch  # noqa: E402

# canonical checkpoint to evaluate, per framework (best_<metric>.pt are analysis extras)
CKPT_REL = {
    "ultralytics": "train/weights/best.pt",  # ultralytics fitness
    "detectron2": "model_final.pth",
    "kraken": "model_best.mlmodel",
}
DEFAULT_CKPT = "train/weights/best.pt"

# CLIs that understand the wandb grouping flags
WANDB_SERVICES = {"ultralytics", "detectron2"}


def _ckpt_rel(service):
    return CKPT_REL.get(service, DEFAULT_CKPT)


def git_commit():
    """Containers only get line_benchmark/ bind-mounted, so .git is not visible
    inside them; the commit has to come in as an environment variable."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=BENCH_ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def job_env(base=None):
    env = dict(os.environ if base is None else base)
    sha = git_commit()
    if sha:
        env["GIT_COMMIT"] = sha
    return env


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def build_matrix(cfg, results_dir="results"):
    """models x data variants -> ordered job list (train -> predict -> eval)."""
    jobs = []
    # the dataset belongs in the id: the same model and variant get retrained
    # on other datasets, and every layout is materialized to the same paths
    prefix = f"{cfg['dataset']}_" if cfg.get("dataset") else ""
    # run 1 answered the preprocessing question (prep never won), so training
    # can sit on one variant while evaluation still covers both
    train_variants = cfg.get("train_variants") or list(cfg["data"])
    for model, mc in cfg["models"].items():
        ckpts = {}
        if mc.get("finetune"):
            for variant in train_variants:
                dc = cfg["data"][variant]
                train_id = f"{prefix}{model}_ft-{variant}"
                ckpts[train_id] = (f"{results_dir}/checkpoints/{train_id}/"
                                   f"{_ckpt_rel(mc['service'])}")
                jobs.append({"kind": "train", "exp_id": train_id,
                             "model": model, "service": mc["service"],
                             "variant": variant,
                             "weights": mc["weights"], "data_yaml": dc.get("yolo"),
                             "images_root": dc["images_root"],
                             "pagexml": dc.get("pagexml")})
                for ev, edc in cfg["data"].items():
                    eid = f"{train_id}_eval-{ev}"
                    jobs.append({"kind": "predict", "exp_id": eid,
                                 "model": model, "service": mc["service"],
                                 "weights": ckpts[train_id],
                                 "images_root": edc["images_root"]})
                    jobs.append({"kind": "eval", "exp_id": eid, "model": model})
        if mc.get("zeroshot"):
            for ev, edc in cfg["data"].items():
                eid = f"{prefix}{model}_zeroshot_eval-{ev}"
                jobs.append({"kind": "predict", "exp_id": eid,
                             "model": model, "service": mc["service"],
                             "weights": mc["weights"],
                             "images_root": edc["images_root"]})
                jobs.append({"kind": "eval", "exp_id": eid, "model": model})
    return jobs


def wandb_flags(cfg, job, imgsz, batch, nbs):
    """Group by campaign, not by dataset: re-running the matrix on the same
    data produced runs with identical names last time, separable only by
    wandb's own id."""
    dataset = cfg.get("dataset", "")
    run = str(cfg.get("run", "")) or None
    group = "/".join(x for x in (dataset, run) if x)
    tags = [dataset, run, job["model"], job["service"], "ft", job["variant"],
            f"imgsz{imgsz}"]
    notes = [cfg.get("notes", "").strip()]
    if nbs:
        tags.append(f"eb{effective_batch(nbs, batch)}")
        # the run's own batching, spelled out where the UI shows it without
        # opening the config
        notes.append(describe(nbs, batch))
    out = []
    if group:
        out += ["--wandb-group", group]
    note = " | ".join(n for n in notes if n)
    if note:
        out += ["--wandb-notes", note]
    return [*out, "--wandb-tags", ",".join(t for t in tags if t)]


def job_command(job, cfg, local, results_dir="results"):
    d = cfg["defaults"]
    if job["kind"] == "train":
        mc = cfg["models"][job["model"]]
        # per-model lr0 (from lr-find) overrides defaults.lr0; ultralytics 'auto'
        # default (~0.002) sits on the unstable side for these models
        lr0 = mc.get("lr0", d.get("lr0"))
        # heavier backbones need their own batch/imgsz: one 24 GB card, and a
        # detector that fits at nano scale does not fit at rtdetr-l scale
        batch = mc.get("batch", d["batch"])
        # what fits on the card varies per model; the batch the optimizer sees
        # must not, or the models are not comparable
        nbs = mc.get("nbs", d.get("nbs"))
        imgsz = mc.get("imgsz", d["imgsz"])
        mosaic = mc.get("mosaic", d.get("mosaic"))
        # data_format seam: YOLO models take a data.yaml; COCO-native frameworks
        # (detectron2) take the shared train/val COCO + the variant's images_root
        fmt = mc.get("data_format", "yolo")
        if fmt == "coco":
            data_args = ["--train-coco", cfg["train_coco"],
                         "--val-coco", cfg["val_coco"],
                         "--images-root", job["images_root"]]
        elif fmt == "pagexml":
            # kraken trains on a dir of PAGE XML (pre-generated by to_pagexml)
            data_args = ["--data", job["pagexml"]]
        else:
            data_args = ["--data", job["data_yaml"]]
        # train flags are framework-specific (--line-val/--diagnostics are
        # ultralytics-only); per-model override falls back to defaults
        flags = mc.get("train_flags", d.get("train_flags", []))
        if "--wandb" in flags and job["service"] in WANDB_SERVICES:
            flags = [*flags, *wandb_flags(cfg, job, imgsz, batch, nbs)]
        argv = ["python", f"docker/{job['service']}/cli.py", "train",
                "--weights", job["weights"],
                *data_args,
                "--out", f"{results_dir}/checkpoints/{job['exp_id']}",
                "--epochs", str(mc.get("epochs", d["epochs"])),
                "--imgsz", str(imgsz),
                "--batch", str(batch),
                *(["--nbs", str(nbs)] if nbs is not None else []),
                *(["--lr0", str(lr0)] if lr0 is not None else []),
                *(["--mosaic", str(mosaic)] if mosaic is not None else []),
                *flags]
    elif job["kind"] == "predict":
        argv = ["python", f"docker/{job['service']}/cli.py", "predict",
                "--weights", job["weights"],
                "--coco", cfg["test_coco"],
                "--images-root", job["images_root"],
                "--out", f"{results_dir}/predictions/{job['exp_id']}",
                "--imgsz", str(d["imgsz"]),
                *d.get("predict_flags", [])]
    else:  # eval
        argv = ["python", "evaluation/evaluate.py",
                "--gt", cfg["test_coco"],
                "--pred", f"{results_dir}/predictions/{job['exp_id']}/predictions.json",
                "--exp-id", job["exp_id"],
                "--out-dir", str(results_dir),
                "--conf-thresh", str(d["conf_thresh"])]
    # eval is model-agnostic host-side python; only train/predict need the
    # model's container (pycocotools lives in the host env, not in images)
    if not local and job["kind"] != "eval":
        argv = ["docker", "compose", "run", "--rm",
                job.get("service", "ultralytics")] + argv
    return argv


def is_done(job, results_dir):
    results_dir = Path(results_dir)
    if job["kind"] == "train":
        ckpt = _ckpt_rel(job.get("service", "ultralytics"))
        return (results_dir / "checkpoints" / job["exp_id"] / ckpt).exists()
    if job["kind"] == "predict":
        return (results_dir / "predictions" / job["exp_id"]
                / "predictions.json").exists()
    return (results_dir / "metrics" / f"{job['exp_id']}.json").exists()


def _log_run(results_dir, job, status):
    # Auxiliary audit log - must never crash the matrix. results/ can be
    # root-owned (containers write as root) while the orchestrator runs as the
    # host user, so writing the log may fail; degrade gracefully.
    path = Path(results_dir) / "run_log.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "kind", "exp_id", "status"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        job["kind"], job["exp_id"], status])
    except OSError as e:
        print(f"  (run_log not written: {e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="orchestrator/experiments.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--only", help="substring filter on exp_id")
    ap.add_argument("--stage", choices=["train", "predict", "eval"])
    ap.add_argument("--local", action="store_true",
                    help="run cli.py directly instead of docker compose")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    jobs = build_matrix(cfg, args.results_dir)
    if args.only:
        jobs = [j for j in jobs if args.only in j["exp_id"]]
    if args.stage:
        jobs = [j for j in jobs if j["kind"] == args.stage]

    env = job_env()
    ran = skipped = failed = 0
    for job in jobs:
        cmd = job_command(job, cfg, args.local, args.results_dir)
        if cmd[0] == "python":
            cmd = [sys.executable] + cmd[1:]   # the env's python, not PATH's
        if not args.force and is_done(job, args.results_dir):
            print(f"skip  {job['kind']:8s} {job['exp_id']}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"would {job['kind']:8s} {job['exp_id']}: {' '.join(cmd)}")
            continue
        print(f"run   {job['kind']:8s} {job['exp_id']}")
        # cwd pinned to line_benchmark/ so relative script paths resolve no matter
        # where the runner itself was invoked from
        result = subprocess.run(cmd, cwd=BENCH_ROOT, env=env)
        if result.returncode == 0:
            _log_run(args.results_dir, job, "ok")
            ran += 1
        else:
            _log_run(args.results_dir, job, "failed")
            print(f"FAILED ({result.returncode}): {job['exp_id']} - moving on")
            failed += 1
    print(f"done: {ran} ran, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
