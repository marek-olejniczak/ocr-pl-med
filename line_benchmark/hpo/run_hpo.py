"""Optuna search over one model's hyperparameters on the hpo/prepare.py
subsamples, maximising val mAP50-95. A trial is the matrix's own job_command
with the trial's values in the model config; pruning reads the per-epoch files
training writes anyway.

Rerunning the same command continues the SQLite study until --trials trials
are complete or pruned. Ctrl-C requeues the running trial's parameters; a
trial lost with the driver comes back through the heartbeat.

Usage (from line_benchmark/):
    python hpo/run_hpo.py --model yolov8 --trials 25
    python hpo/run_hpo.py --model frcnn --dry-run
"""

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from orchestrator.run_experiments import (job_command, job_env,  # noqa: E402
                                          load_config)

FRCNN_RATIOS = ["0.5,1.0,2.0",       # zoo
                "0.1,0.2,0.5",
                "0.07,0.13,0.25",    # p25 / median / p75 of the train boxes
                "0.04,0.1,0.25"]
FRCNN_SIZES = ["32,64,128,256,512",  # zoo
               "16,32,64,128,256"]


def suggest(model, trial):
    """Trial values as a model-config override: lr0/mosaic where job_command
    already reads them, everything else as train_args."""
    args = {"warmup-epochs": trial.suggest_float("warmup_epochs", 0.5, 3.0)}
    if model == "frcnn":
        lr0 = trial.suggest_float("lr0", 2e-3, 4e-2, log=True)
        # 1.0 is a flat schedule after warmup: in r2 the flat 0.016 beat the
        # cosine to 1% by 3.6 mAP50 points, so the shape is left to the search
        args["lrf"] = trial.suggest_float("lrf", 0.01, 1.0, log=True)
        args["weight-decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3,
                                                   log=True)
        args["anchor-ratios"] = trial.suggest_categorical("anchor_ratios",
                                                          FRCNN_RATIOS)
        args["anchor-sizes"] = trial.suggest_categorical("anchor_sizes",
                                                         FRCNN_SIZES)
        return {"lr0": lr0, "train_args": args}
    lo, hi = (5e-5, 1e-3) if model == "rtdetr" else (2e-4, 5e-3)
    out = {"lr0": trial.suggest_float("lr0", lo, hi, log=True)}
    args["weight-decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2,
                                               log=True)
    args["scale"] = trial.suggest_float("scale", 0.1, 0.7)
    args["fliplr"] = trial.suggest_categorical("fliplr", [0.0, 0.5])
    if model != "rtdetr":    # rtdetr keeps mosaic off, see experiments.yaml
        out["mosaic"] = trial.suggest_categorical("mosaic", [0.0, 0.5, 1.0])
    out["train_args"] = args
    return out


def baseline(cfg, model):
    """The r2 settings as trial params, so the study has the untuned run on
    the same subsample to compare against."""
    lr0 = cfg["models"][model].get("lr0", cfg["defaults"]["lr0"])
    params = {"warmup_epochs": 3.0, "lr0": lr0}
    if model == "frcnn":
        return {**params, "lrf": 0.01, "weight_decay": 1e-4,
                "anchor_ratios": FRCNN_RATIOS[0],
                "anchor_sizes": FRCNN_SIZES[0]}
    params.update(weight_decay=0.0005, scale=0.5, fliplr=0.5)
    if model != "rtdetr":
        params["mosaic"] = 1.0
    return params


def trial_setup(cfg, model, number, override, data, epochs, tag=""):
    """(cfg, job) for one trial: the matrix config narrowed to this model and
    the subsamples, with the trial's values on top."""
    cfg = copy.deepcopy(cfg)
    mc = {**cfg["models"][model], **override, "epochs": epochs,
          "train_flags": ["--wandb"]}
    if mc["service"] == "ultralytics":
        # ultralytics closes mosaic for the last 10 epochs, which in a
        # 10-epoch trial is all of them; the full runs close it halfway
        mc["train_args"] = {**mc.get("train_args", {}),
                            "close-mosaic": epochs // 2}
    label = f"hpo-{tag}" if tag else "hpo"
    cfg["models"] = {model: mc}
    cfg["run"] = f"{label}-{model}"
    cfg["notes"] = (f"HPO trial {number}: "
                    + json.dumps(override, sort_keys=True, default=str))
    if mc.get("data_format") == "coco":
        cfg["train_coco"], cfg["val_coco"] = data["train_coco"], data["val_coco"]
    variant = (cfg.get("train_variants") or list(cfg["data"]))[0]
    dc = cfg["data"][variant]
    exp_id = f"{cfg.get('dataset', 'data')}_{model}_{label}-t{number:03d}"
    job = {"kind": "train", "exp_id": exp_id, "model": model,
           "service": mc["service"], "variant": variant,
           "weights": mc["weights"], "data_yaml": data["data_yaml"],
           "images_root": dc["images_root"], "pagexml": dc.get("pagexml")}
    return cfg, job


def container_name(exp_id):
    return exp_id.replace("_", "-")


def with_container_name(argv, name):
    """Named, so pruning can stop it - stopping the compose client alone can
    leave the container running on the GPU."""
    if argv[:4] == ["docker", "compose", "run", "--rm"]:
        return argv[:4] + ["--name", name] + argv[4:]
    return argv


def read_curve(service, out_dir):
    """[(epoch, val mAP50-95)] so far. Tolerates a file mid-write."""
    out_dir = Path(out_dir)
    curve = []
    if service == "ultralytics":
        path = out_dir / "train" / "results.csv"
        if not path.exists():
            return curve
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items() if k}
                try:
                    curve.append((int(float(row["epoch"])),
                                  float(row["metrics/mAP50-95(B)"])))
                except (KeyError, TypeError, ValueError):
                    break
        return curve
    path, meta = out_dir / "metrics.json", out_dir / "run_meta.json"
    if not path.exists() or not meta.exists():
        return curve
    per_epoch = json.loads(meta.read_text())["iters_per_epoch"]
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            break
        if "bbox/AP" in rec:                   # percent in detectron2
            curve.append(((rec["iteration"] + 1) // per_epoch,
                          rec["bbox/AP"] / 100.0))
    return curve


def final_value(service, curve):
    """The value of the checkpoint the benchmark would evaluate: best.pt is the
    best epoch for ultralytics, model_final.pth the last one for detectron2."""
    if not curve:
        return None
    if service == "ultralytics":
        return max(v for _, v in curve)
    return curve[-1][1]


def drop_weights(out_dir):
    for pattern in ("train/weights/*.pt", "*.pth"):
        for p in Path(out_dir).glob(pattern):
            p.unlink()


def stop(proc, name, local):
    if local:
        proc.terminate()
    else:
        subprocess.run(["docker", "stop", "-t", "20", name],
                       capture_output=True, check=False)
    proc.wait()


def stop_orphans(prefix):
    """Containers of a driver that died - one driver per model, so any
    container with this prefix is ours."""
    out = subprocess.run(["docker", "ps", "-q", "--filter", f"name={prefix}"],
                         capture_output=True, text=True, check=False).stdout
    for cid in out.split():
        subprocess.run(["docker", "stop", "-t", "20", cid],
                       capture_output=True, check=False)


def make_objective(cfg, model, data, args):
    import optuna

    def objective(trial):
        override = suggest(model, trial)
        tcfg, job = trial_setup(cfg, model, trial.number, override, data,
                                args.epochs, args.tag)
        out_dir = BENCH_ROOT / args.results_dir / "checkpoints" / job["exp_id"]
        if out_dir.exists():
            shutil.rmtree(out_dir)       # a retried trial starts clean
        name = container_name(job["exp_id"])
        cmd = with_container_name(
            job_command(job, tcfg, args.local, args.results_dir), name)
        if cmd[0] == "python":
            cmd = [sys.executable] + cmd[1:]
        trial.set_user_attr("exp_id", job["exp_id"])
        print(f"\n=== trial {trial.number}: {job['exp_id']} "
              f"{json.dumps(override, sort_keys=True, default=str)}", flush=True)

        service = job["service"]
        proc = subprocess.Popen(cmd, cwd=BENCH_ROOT, env=job_env())
        reported = 0
        try:
            while proc.poll() is None:
                time.sleep(args.poll)
                curve = read_curve(service, out_dir)
                for epoch, value in curve[reported:]:
                    trial.report(value, epoch)
                    reported += 1
                    # the reference point for the whole study, so it runs out
                    if (trial.should_prune()
                            and not trial.user_attrs.get("baseline")):
                        print(f"--- trial {trial.number} pruned at epoch "
                              f"{epoch} (mAP50-95 {value:.4f})", flush=True)
                        stop(proc, name, args.local)
                        raise optuna.TrialPruned()
        except KeyboardInterrupt:
            stop(proc, name, args.local)
            trial.study.enqueue_trial(trial.params, user_attrs={
                k: v for k, v in trial.user_attrs.items() if k == "baseline"})
            raise
        finally:
            if not args.keep_weights:
                drop_weights(out_dir)

        curve = read_curve(service, out_dir)
        for epoch, value in curve[reported:]:
            trial.report(value, epoch)
        if proc.returncode != 0:
            raise RuntimeError(f"training exited {proc.returncode}")
        value = final_value(service, curve)
        if value is None:
            raise RuntimeError(f"no val mAP50-95 in {out_dir}")
        trial.set_user_attr("epochs_done", len(curve))
        return value

    return objective


def write_trials(study, path):
    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": t.state.name,
                     "value": t.value, "epochs_done":
                         len(t.intermediate_values),
                     **{f"p_{k}": v for k, v in t.params.items()},
                     "exp_id": t.user_attrs.get("exp_id")})
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (not k == "number", k))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["yolov8", "rtdetr", "frcnn"])
    ap.add_argument("--trials", type=int, default=25,
                    help="total completed + pruned trials the study runs to")
    ap.add_argument("--epochs", type=int, default=10, help="per trial")
    ap.add_argument("--config", default="orchestrator/experiments.yaml")
    ap.add_argument("--results-dir", default="results_hpo")
    ap.add_argument("--train-per-group", type=int, default=100)
    ap.add_argument("--val-per-group", type=int, default=67)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll", type=float, default=30.0, help="seconds")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--tag", default="",
                    help="a separate study, e.g. v2 after a fix: its own "
                         "SQLite study, exp_ids and wandb group")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first trial's command and stop")
    args = ap.parse_args(argv)

    cfg = load_config(BENCH_ROOT / args.config)
    tag_t, tag_v = f"g{args.train_per_group}", f"g{args.val_per_group}"
    data = {"train_coco": f"dataset/hpo/instances_train_{tag_t}.json",
            "val_coco": f"dataset/hpo/instances_val_{tag_v}.json",
            "data_yaml": f"dataset/yolo_raw/data_hpo_{tag_t}_{tag_v}.yaml"}
    missing = [p for p in data.values() if not (BENCH_ROOT / p).exists()]
    if missing and not args.dry_run:
        raise SystemExit(f"missing {missing[0]} - run hpo/prepare.py first")

    import optuna
    from optuna.storages import RDBStorage, RetryFailedTrialCallback
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState

    if args.dry_run:
        t = optuna.create_study(sampler=optuna.samplers.RandomSampler(
            seed=args.seed)).ask()
        override = suggest(args.model, t)
        tcfg, job = trial_setup(cfg, args.model, 0, override, data,
                                args.epochs, args.tag)
        cmd = with_container_name(
            job_command(job, tcfg, args.local, args.results_dir),
            container_name(job["exp_id"]))
        print(" ".join(cmd))
        return

    name = f"{args.model}-{args.tag}" if args.tag else args.model
    results = BENCH_ROOT / args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    storage = RDBStorage(
        f"sqlite:///{results / 'hpo.db'}", heartbeat_interval=60,
        grace_period=180,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1))
    study = optuna.create_study(
        study_name=f"{cfg.get('dataset', 'data')}-{name}",
        storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=3))
    if not any(t.user_attrs.get("baseline") for t in study.trials):
        study.enqueue_trial(baseline(cfg, args.model),
                            user_attrs={"baseline": True})
    if not args.local:
        label = f"hpo-{args.tag}" if args.tag else "hpo"
        stop_orphans(container_name(f"{cfg.get('dataset', 'data')}_"
                                    f"{args.model}_{label}-t"))
    study.optimize(
        make_objective(cfg, args.model, data, args),
        callbacks=[MaxTrialsCallback(
            args.trials, states=(TrialState.COMPLETE, TrialState.PRUNED))],
        catch=(RuntimeError,))

    write_trials(study, results / f"{name}_trials.csv")
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\n{len(done)} complete, "
          f"{sum(t.state == TrialState.PRUNED for t in study.trials)} pruned")
    for t in sorted(done, key=lambda t: -t.value)[:5]:
        print(f"  t{t.number:03d}  mAP50-95={t.value:.4f}  {t.params}")


if __name__ == "__main__":
    main()
