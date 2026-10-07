"""Evaluation sets for measuring how much each augmentation hurts reading.

Two sets, both in the benchmark's format (labels.csv with file_name,label plus
an images/ folder), each with a labels.jsonl that says what was done to every
image:

robustness
    A paired design in the style of ImageNet-C (Hendrycks & Dietterich, 2019).
    Each base line - one text, one hand, one ink, one framing - is rendered
    under every condition below. Inside one base line the paper grain, the
    effect's own randomness and the capture noise are seeded identically, so
    two renders of a line differ only by the condition. The difference in
    CER against the "clean" render is therefore the cost of that condition
    alone, not of a harder word that happened to land in one bucket.

    Severities 1-3 are the gentle end, the middle and the harsh end of the
    range the training generator draws from, so the set measures exactly the
    conditions the models were trained on. The phone profile is also split
    into its components, each alone at its harshest value, to show which
    part of a phone photo does the damage.

    Rows are ordered base line by base line, so the benchmark's `limit`
    option takes a smaller set that is still complete: limit = N x conditions
    gives the first N base lines under every condition.

val
    A random subset of a training dataset's held-out validation lines,
    copied with their metadata. The models never trained on these lines;
    reading them gives CER per font, per content kind and per augmentation
    as they occur in the training distribution.

Usage:
    python src/build_eval_sets.py robustness --output-dir output/robustness --base-lines 1500
    python src/build_eval_sets.py val --source output/v2_800k --output-dir output/v2_val10k --count 10000
    python src/build_eval_sets.py pack --dvc-repo "C:/Users/tomek/Desktop/dane/inzynierka" robustness v2_val10k
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import fill_form
import generate_ocr_lines as gol
import line_effects as fx
from dataset_card import build_card, write_card
from ocr_content import LinePools
from renderer import find_fonts
from vocabulary import Vocabulary

REPO = Path(__file__).resolve().parent.parent

# Phone components alone: everything neutral except one knob at its harshest.
_PHONE_NEUTRAL = {"shadow": 0.0, "cast": "neutral", "exposure": 1.0, "noise": 0.0,
                  "blur": 0.0, "motion_blur": None, "downscale": None, "jpeg": None}


def _phone_only(**kw) -> dict:
    return {**_PHONE_NEUTRAL, **kw}


# name, family, severity, stage, params
#   stage "pre"     drawn on the paper before the ink (grid, neighbours)
#   stage "post"    applied to the inked line (morphology, elastic)
#   stage "phone"   phone capture with these settings
#   stage "scan"    scanner capture forced to this profile
CONDITIONS = [
    ("clean", "none", 0, None, None),

    ("neighbours_s1", "neighbour_glyphs", 1, "pre", {"sides": ["top", "bottom"], "depth_frac": 0.12}),
    ("neighbours_s2", "neighbour_glyphs", 2, "pre", {"sides": ["top", "bottom"], "depth_frac": 0.28}),
    ("neighbours_s3", "neighbour_glyphs", 3, "pre", {"sides": ["top", "bottom"], "depth_frac": 0.45}),

    ("grid_s1", "grid_paper", 1, "pre", {"kind": "grid", "alpha": 0.45, "pitch_frac": 0.80}),
    ("grid_s2", "grid_paper", 2, "pre", {"kind": "grid", "alpha": 0.72, "pitch_frac": 0.80}),
    ("grid_s3", "grid_paper", 3, "pre", {"kind": "grid", "alpha": 1.00, "pitch_frac": 0.80}),
    ("ruled_s3", "grid_paper", 3, "pre", {"kind": "ruled", "alpha": 1.00, "pitch_frac": 1.35}),

    ("erosion_s1", "morphology", 1, "post", {"op": "erosion", "mix": 0.50}),
    ("erosion_s2", "morphology", 2, "post", {"op": "erosion", "mix": 0.75}),
    ("erosion_s3", "morphology", 3, "post", {"op": "erosion", "mix": 1.00}),
    ("dilation_s1", "morphology", 1, "post", {"op": "dilation", "size": 1}),
    ("dilation_s3", "morphology", 3, "post", {"op": "dilation", "size": 2}),

    ("elastic_s1", "elastic", 1, "post", {"kind": "elastic", "alpha_frac": 0.6, "sigma_frac": 0.24}),
    ("elastic_s2", "elastic", 2, "post", {"kind": "elastic", "alpha_frac": 1.1, "sigma_frac": 0.24}),
    ("elastic_s3", "elastic", 3, "post", {"kind": "elastic", "alpha_frac": 1.6, "sigma_frac": 0.24}),
    ("griddist_s1", "elastic", 1, "post", {"kind": "grid_distortion", "num_steps": 4, "distort_limit": 0.08}),
    ("griddist_s2", "elastic", 2, "post", {"kind": "grid_distortion", "num_steps": 4, "distort_limit": 0.15}),
    ("griddist_s3", "elastic", 3, "post", {"kind": "grid_distortion", "num_steps": 4, "distort_limit": 0.22}),

    ("phone_s1", "phone_photo", 1, "phone", fx.PHONE_PRESETS[1]),
    ("phone_s2", "phone_photo", 2, "phone", fx.PHONE_PRESETS[2]),
    ("phone_s3", "phone_photo", 3, "phone", fx.PHONE_PRESETS[3]),
    ("phone_shadow", "phone_photo", 3, "phone", _phone_only(shadow=0.40)),
    ("phone_noise", "phone_photo", 3, "phone", _phone_only(noise=7.0)),
    ("phone_blur", "phone_photo", 3, "phone", _phone_only(blur=1.3)),
    ("phone_motion", "phone_photo", 3, "phone", _phone_only(motion_blur=5)),
    ("phone_downscale", "phone_photo", 3, "phone", _phone_only(downscale=0.45)),
    ("phone_jpeg", "phone_photo", 3, "phone", _phone_only(jpeg=55)),

    ("scan_clean_color", "scanner", 1, "scan", "clean_color"),
    ("scan_grayscale", "scanner", 2, "scan", "grayscale"),
    ("scan_photocopy", "scanner", 3, "scan", "photocopy"),
]

# pandas turns these into NaN even in a text column; the benchmark would
# then score the line against the string "nan".
_PANDAS_NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
              "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
              "nan", "null"}

JPEG_QUALITY = 93  # what generate_ocr_lines saves with


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s % (2 ** 32))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def render_condition(base: dict, cond: tuple, s: int, pools, config):
    """One base line under one condition. Returns (image, record)."""
    name, family, severity, stage, params = cond
    record: dict = {}

    _seed(s + 1)
    canvas = gol.paper_canvas(base["width"], base["height"])

    _seed(s + 2)
    if stage == "pre" and family == "grid_paper":
        rec = {}
        fx.draw_grid_paper(canvas, base["text_img"].height, base["baseline"], params=params, record=rec)
        record["grid_params"] = rec
    elif stage == "pre" and family == "neighbour_glyphs":
        rec: list = []
        fx.intrude_neighbour_glyphs(canvas, base["ink"], base["font_path"], base["font_size"],
                                    config, base["style"], pools.word, params=params, record=rec)
        record["neighbour_params"] = rec

    gol.paste_ink(canvas, base)

    _seed(s + 3)
    if stage == "post" and family == "morphology":
        rec = {}
        canvas, _ = fx.apply_morphology(canvas, params=params, record=rec)
        record["morphology_params"] = rec
    elif stage == "post" and family == "elastic":
        rec = {}
        canvas, _ = fx.apply_elastic(canvas, params=params, record=rec)
        record["elastic_params"] = rec

    _seed(s + 4)
    if stage == "phone":
        canvas, meta = fx.apply_phone_photo(canvas, params=params)
        record["scan"] = meta
    elif stage == "scan":
        original = fill_form.pick_scan_profile
        fill_form.pick_scan_profile = lambda: params
        try:
            canvas, meta = fill_form.apply_scan_augmentation(canvas)
        finally:
            fill_form.pick_scan_profile = original
        record["scan"] = meta
    return canvas, record


def cmd_robustness(args) -> None:
    out = Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(args.resource_dir)
    pools = LinePools(args.resource_dir)
    fonts = [f for f in find_fonts(args.font_dir) if Path(f).name not in fill_form.EXCLUDED_FONTS]
    config = gol.build_line_config()
    enabled = set(gol.ALL_FAMILIES)

    n_cond = len(CONDITIONS)
    print(f"{args.base_lines} linii bazowych x {n_cond} warunkow = {args.base_lines * n_cond} obrazow")

    rows = 0
    base_id = 0
    attempt = 0
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as fcsv, \
         open(out / "labels.jsonl", "w", encoding="utf-8") as fjs:
        w = csv.writer(fcsv)
        w.writerow(["file_name", "label"])
        while base_id < args.base_lines:
            s = args.seed * 1_000_003 + attempt * 10
            attempt += 1
            _seed(s)
            base = gol.prepare_line(vocab, fonts, config, pools, enabled)
            if base is None or base["text"].strip() in _PANDAS_NA:
                continue
            common = gol.line_meta(base)
            for cond in CONDITIONS:
                img, rec = render_condition(base, cond, s, pools, config)
                fname = f"r{base_id:05d}_{cond[0]}.jpg"
                img.save(out / "images" / fname, quality=JPEG_QUALITY)
                w.writerow([fname, base["text"]])
                fjs.write(json.dumps(_jsonable({
                    "file_name": fname, "text": base["text"], "base_id": base_id,
                    "condition": cond[0], "family": cond[1], "severity": cond[2],
                    "forced": cond[4], **common, **rec,
                    "size": list(img.size),
                }), ensure_ascii=False) + "\n")
                rows += 1
            base_id += 1
            if base_id % 100 == 0:
                print(f"  {base_id}/{args.base_lines} linii bazowych ({rows} obrazow)", flush=True)

    card = build_card(
        name=out.name,
        command="python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
        seed=args.seed, repo_root=REPO,
        counts={"lines": rows, "base_lines": base_id, "conditions": n_cond},
        sources={"synthetic": rows}, fonts=sorted({Path(f).name for f in fonts}),
        observed={"render_attempts": attempt},
        note="zbior kontrolny odpornosci: kazda linia bazowa w kazdym warunku",
    )
    card["augmentations"] = {"conditions": [
        {"name": c[0], "family": c[1], "severity": c[2], "stage": c[3], "params": c[4]}
        for c in CONDITIONS]}
    write_card(out, _jsonable(card))
    print(f"Gotowe: {rows} obrazow w {out}")


def cmd_val(args) -> None:
    src, out = Path(args.source), Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    with open(src / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "val" and r["text"].strip() not in _PANDAS_NA:
                rows.append(r)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.count]
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as fcsv, \
         open(out / "labels.jsonl", "w", encoding="utf-8") as fjs:
        w = csv.writer(fcsv)
        w.writerow(["file_name", "label"])
        for r in rows:
            name = Path(r["file_name"]).name
            shutil.copy2(src / r["file_name"], out / "images" / name)
            w.writerow([name, r["text"]])
            fjs.write(json.dumps({**r, "file_name": name}, ensure_ascii=False) + "\n")
    card_src = src / "dataset_card.json"
    card = json.loads(card_src.read_text(encoding="utf-8")) if card_src.exists() else {}
    card = {"name": out.name, "derived_from": src.name, "seed": args.seed,
            "counts": {"lines": len(rows)}, "note": "losowy podzbior linii walidacyjnych",
            "source_card": card}
    (out / "dataset_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gotowe: {len(rows)} linii walidacyjnych z {src.name} w {out}")


def cmd_pack(args) -> None:
    """One tar of images plus the label files, under dataset/<name>/."""
    for name in args.names:
        src = Path(args.output_root) / name
        dst = Path(args.dvc_repo) / "dataset" / name
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "src/shard_for_dvc.py", str(src / "images"),
                        "--output-dir", str(dst / "images_shards"), "--prefix", "images",
                        "--shard-gb", "0.9"], check=True, cwd=str(REPO))
        for fname in ("labels.csv", "labels.jsonl", "dataset_card.json"):
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
        print(f"[{name}] -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("robustness")
    r.add_argument("--output-dir", default="output/robustness")
    r.add_argument("--base-lines", type=int, default=1500)
    r.add_argument("--seed", type=int, default=4040)
    r.add_argument("--font-dir", default="resources/fonts")
    r.add_argument("--resource-dir", default="resources")
    v = sub.add_parser("val")
    v.add_argument("--source", default="output/v2_800k")
    v.add_argument("--output-dir", default="output/v2_val10k")
    v.add_argument("--count", type=int, default=10000)
    v.add_argument("--seed", type=int, default=4041)
    p = sub.add_parser("pack")
    p.add_argument("--dvc-repo", required=True)
    p.add_argument("--output-root", default="output")
    p.add_argument("names", nargs="+")
    args = ap.parse_args()
    {"robustness": cmd_robustness, "val": cmd_val, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    main()
