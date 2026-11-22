"""Handwritten-lines COCO set (IAM + Bengali pages) as an ultralytics YOLO dataset.

Split by source document, so variants of one IAM form (a01_000, a01_000u,
a01_000x) and pages of one Bengali document never land in both train and val.
Train pages are linked as they are and additionally written once as an
augmented copy (*_aug.jpg). Only pixel-level augmentations happen here, boxes
stay where they are; rotation/perspective are left to ultralytics.

--extra-train/--extra-val add pages from image lists of another YOLO layout
(hpo/prepare.py lists of docs_20k), linked image and label next to ours.

Creates:
    out/images/{train,val}/   symlinks to source images (+ *_aug.jpg in train)
    out/labels/{train,val}/
    out/data.yaml

Usage (from line_benchmark/):
    python data_prep/converters/hw_lines_to_yolo.py \
        --coco dataset/handwritten_lines/lines_coco.json \
        --images-root dataset/handwritten_lines --out-dir dataset/yolo_hw_mix \
        --extra-train dataset/yolo_raw/hpo_train_g100.txt \
        --extra-val dataset/yolo_raw/hpo_val_g67.txt
"""

import argparse
import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_prep.converters.to_yolo import (coco_to_yolo_lines,  # noqa: E402
                                          write_data_yaml)

SEED = 0
GROUP_RES = (re.compile(r"^(iam_[a-z]\d+_\d+)[a-z]?$"),
             re.compile(r"^(seg_\d+_\d+)_\d+$"),
             re.compile(r"^(auto_\d+)_\d+$"))

P_BARS, P_BLUE, P_SCAN, P_RULED = 0.4, 0.3, 0.6, 0.15


def group_key(file_name):
    stem = Path(file_name).stem
    for rx in GROUP_RES:
        m = rx.match(stem)
        if m:
            return m[1]
    raise ValueError(f"unknown file name pattern: {file_name}")


def split_groups(images, val_frac, seed=SEED):
    groups = defaultdict(list)
    for img in images:
        groups[group_key(img["file_name"])].append(img)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    n_val = round(len(names) * val_frac)
    val = set(names[:n_val])
    return ([i for n in names if n not in val for i in groups[n]],
            [i for n in names if n in val for i in groups[n]])


def _overlaps(box, boxes):
    x0, y0, x1, y1 = box
    return any(x0 < bx + bw and bx < x1 and y0 < by + bh and by < y1
               for bx, by, bw, bh in boxes)


def redaction_bars(img, boxes, rng, max_bars=3):
    """Black bars right next to a line end, like the anonymised patient cards.
    The line box itself stays on the text."""
    d = ImageDraw.Draw(img)
    for x, y, w, h in rng.sample(boxes, min(max_bars, len(boxes))):
        bw = rng.uniform(2, 6) * h
        gap = rng.uniform(0.05, 0.3) * h
        bar = ((x + w + gap, y, x + w + gap + bw, y + h) if rng.random() < 0.7
               else (x - gap - bw, y, x - gap, y + h))
        inside = bar[0] >= 0 and bar[2] <= img.width
        if inside and not _overlaps(bar, boxes):
            d.rectangle(bar, fill=(0, 0, 0))


def blue_ink(img, rng):
    a = np.asarray(img, dtype=np.float32)
    ink = 1 - a.mean(axis=2, keepdims=True) / 255
    pen = np.array([rng.uniform(10, 60), rng.uniform(40, 90), rng.uniform(150, 220)])
    return Image.fromarray((255 - ink * (255 - pen)).clip(0, 255).astype(np.uint8))


def ruled_lines(img, boxes, rng):
    d = ImageDraw.Draw(img)
    step = max(20, int(np.median([b[3] for b in boxes]) * rng.uniform(0.8, 1.2)))
    color = (rng.randint(150, 200), rng.randint(170, 210), rng.randint(200, 235))
    for y in range(rng.randint(0, step), img.height, step):
        d.line([(0, y), (img.width, y)], fill=color, width=2)
    if rng.random() < 0.5:
        for x in range(rng.randint(0, step), img.width, step):
            d.line([(x, 0), (x, img.height)], fill=color, width=2)


def scan_profile(img, rng):
    kind = rng.choice(["grayscale", "photocopy", "blur"])
    if kind == "grayscale":
        img = img.convert("L").convert("RGB")
    elif kind == "photocopy":
        a = np.asarray(img.convert("L"), dtype=np.float32)
        a = (a - 128) * rng.uniform(1.2, 1.8) + 128 + rng.uniform(10, 30)
        a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 12, a.shape)
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGB")
    else:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(40, 85))
    return Image.open(buf).convert("RGB")


def augment(img, boxes, rng):
    img = img.convert("RGB")
    if rng.random() < P_RULED:
        ruled_lines(img, boxes, rng)
    if rng.random() < P_BLUE:
        img = blue_ink(img, rng)
    if rng.random() < P_BARS:
        redaction_bars(img, boxes, rng)
    if rng.random() < P_SCAN:
        img = scan_profile(img, rng)
    return img


def write_split(images, anns, images_root, out_dir, split, aug_rng=None):
    img_out = Path(out_dir) / "images" / split
    lbl_out = Path(out_dir) / "labels" / split
    for d in (img_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)
        stale = list(d.iterdir())
        if stale:
            raise SystemExit(f"{d} is not empty, refusing to mix datasets")
    for img in images:
        src = (Path(images_root) / img["file_name"]).resolve()
        stem = Path(img["file_name"]).stem
        (img_out / src.name).symlink_to(os.path.relpath(src, img_out.resolve()))
        lines = "\n".join(coco_to_yolo_lines(anns[img["id"]], img["width"],
                                              img["height"])) + "\n"
        (lbl_out / f"{stem}.txt").write_text(lines)
        if aug_rng is not None:
            boxes = [a["bbox"] for a in anns[img["id"]]]
            augment(Image.open(src), boxes, aug_rng).save(
                img_out / f"{stem}_aug.jpg", quality=90)
            (lbl_out / f"{stem}_aug.txt").write_text(lines)


def list_pairs(list_txt):
    """./images/<split>/x.jpg lines, resolved against the list's directory."""
    if not list_txt:
        return []
    base = Path(list_txt).resolve().parent
    pairs = []
    for ln in filter(None, Path(list_txt).read_text().splitlines()):
        img = base / ln
        lbl = (base / ln.replace("images/", "labels/", 1)).with_suffix(".txt")
        for f in (img, lbl):
            if not f.exists():
                raise SystemExit(f"missing {f}")
        pairs.append((img, lbl))
    return pairs


def link_pairs(pairs, out_dir, split):
    for img, lbl in pairs:
        for src, dst in ((img, Path(out_dir) / "images" / split / img.name),
                         (lbl, Path(out_dir) / "labels" / split / lbl.name)):
            dst.symlink_to(os.path.relpath(src, dst.parent.resolve()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, help="first N images only, for a dry look")
    ap.add_argument("--extra-train")
    ap.add_argument("--extra-val")
    args = ap.parse_args(argv)

    coco = json.loads(Path(args.coco).read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    images = coco["images"][:args.limit] if args.limit else coco["images"]

    extra_train, extra_val = list_pairs(args.extra_train), list_pairs(args.extra_val)
    train, val = split_groups(images, args.val_frac)
    assert not {group_key(i["file_name"]) for i in train} & {group_key(i["file_name"]) for i in val}
    write_split(train, anns, args.images_root, args.out_dir, "train", random.Random(SEED))
    write_split(val, anns, args.images_root, args.out_dir, "val")
    link_pairs(extra_train, args.out_dir, "train")
    link_pairs(extra_val, args.out_dir, "val")
    write_data_yaml(args.out_dir)
    print(f"train {len(train)} (+{len(train)} aug, +{len(extra_train)} extra), "
          f"val {len(val)} (+{len(extra_val)} extra)")


if __name__ == "__main__":
    main()
