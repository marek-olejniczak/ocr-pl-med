"""Run the benchmark experiment matrix declared in experiments.yaml.

The matrix is generated programmatically from models x data variants:
    train:   {model}_ft-{variant}
    predict: {model}_ft-{variant}_eval-{variant} / {model}_zeroshot_eval-{variant}
    eval:    same ids as predict
Each stage is idempotent - a job whose artifact already exists is skipped
(--force re-runs). Failures are logged to results/run_log.csv and the runner
moves on to the next job.

Usage (from benchmark/):
    python orchestrator/run_experiments.py --dry-run
    python orchestrator/run_experiments.py --only yolov8 --stage train
    python orchestrator/run_experiments.py --local        # no docker
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

CANONICAL_CKPT = "train/weights/best.pt"   # ultralytics fitness; best_<metric>.pt = analysis


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def build_matrix(cfg, results_dir="results"):
    """models x data variants -> ordered job list (train -> predict -> eval)."""
    jobs = []
    for model, mc in cfg["models"].items():
        ckpts = {}
        if mc.get("finetune"):
            for variant, dc in cfg["data"].items():
                train_id = f"{model}_ft-{variant}"
                ckpts[train_id] = f"{results_dir}/checkpoints/{train_id}/{CANONICAL_CKPT}"
                jobs.append({"kind": "train", "exp_id": train_id,
                             "model": model, "service": mc["service"],
                             "weights": mc["weights"], "data_yaml": dc["yolo"]})
                for ev, edc in cfg["data"].items():
                    eid = f"{train_id}_eval-{ev}"
                    jobs.append({"kind": "predict", "exp_id": eid,
                                 "model": model, "service": mc["service"],
                                 "weights": ckpts[train_id],
                                 "images_root": edc["images_root"]})
                    jobs.append({"kind": "eval", "exp_id": eid, "model": model})
        if mc.get("zeroshot"):
            for ev, edc in cfg["data"].items():
                eid = f"{model}_zeroshot_eval-{ev}"
                jobs.append({"kind": "predict", "exp_id": eid,
                             "model": model, "service": mc["service"],
                             "weights": mc["weights"],
                             "images_root": edc["images_root"]})
                jobs.append({"kind": "eval", "exp_id": eid, "model": model})
    return jobs


def job_command(job, cfg, local, results_dir="results"):
    d = cfg["defaults"]
    if job["kind"] == "train":
        argv = ["python", f"docker/{job['service']}/cli.py", "train",
                "--weights", job["weights"],
                "--data", job["data_yaml"],
                "--out", f"{results_dir}/checkpoints/{job['exp_id']}",
                "--epochs", str(d["epochs"]),
                "--imgsz", str(d["imgsz"]),
                "--batch", str(d["batch"]),
                *d.get("train_flags", [])]
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
    if not local:
        argv = ["docker", "compose", "run", "--rm",
                job.get("service", "ultralytics")] + argv
    return argv


def is_done(job, results_dir):
    results_dir = Path(results_dir)
    if job["kind"] == "train":
        return (results_dir / "checkpoints" / job["exp_id"]
                / "train" / "weights" / "best.pt").exists()
    if job["kind"] == "predict":
        return (results_dir / "predictions" / job["exp_id"]
                / "predictions.json").exists()
    return (results_dir / "metrics" / f"{job['exp_id']}.json").exists()


def _log_run(results_dir, job, status):
    path = Path(results_dir) / "run_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "kind", "exp_id", "status"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    job["kind"], job["exp_id"], status])


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

    ran = skipped = failed = 0
    for job in jobs:
        cmd = job_command(job, cfg, args.local, args.results_dir)
        if args.local:
            cmd = [sys.executable] + cmd[1:]   # the env's python, not PATH's
        if not args.force and is_done(job, args.results_dir):
            print(f"skip  {job['kind']:8s} {job['exp_id']}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"would {job['kind']:8s} {job['exp_id']}: {' '.join(cmd)}")
            continue
        print(f"run   {job['kind']:8s} {job['exp_id']}")
        # cwd pinned to benchmark/ so relative script paths resolve no matter
        # where the runner itself was invoked from
        result = subprocess.run(cmd, cwd=BENCH_ROOT)
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
