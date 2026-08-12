"""Convert the labeling-tool CSV export into a COCO file for line detection.

One row per hand-labeled line:
    filename,label,x_min,y_min,x_max,y_max,crop

`label` holds the transcription, except for rows that were boxed but not
transcribed, where it is the field type (printed/text/number). Those rows
still carry a valid box, so they count for detection while staying out of
the OCR ground truth: the transcription is written to the annotation's
`text` field and left null for them (e2e_benchmark/gt.py reads that field).

Usage (from line_benchmark/):
    python data_prep/converters/manual_to_coco.py --csv ../dataset/annotations.csv \\
        --dataset-root ../dataset --out dataset/annotations/instances_manual.json \\
        --images-out dataset/real_scans
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

NON_TEXT = ("printed", "text", "number")
CATEGORY = {"id": 1, "name": "text_line", "supercategory": "text"}


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_page(dataset_root, file_name):
    """Pages sit either flat or in a directory named after them."""
    root = Path(dataset_root)
    for cand in (root / Path(file_name).stem / file_name, root / file_name):
        if cand.exists():
            return cand
    return None


def build(rows, dataset_root, non_text=NON_TEXT):
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["filename"]].append(r)

    images, annotations, missing, crops = [], [], [], []
    for file_name in sorted(by_page):
        src = find_page(dataset_root, file_name)
        if src is None:
            missing.append(file_name)
            continue
        image_id = len(images) + 1
        with Image.open(src) as im:
            width, height = im.size
        images.append({"id": image_id, "file_name": file_name,
                       "width": width, "height": height})
        for r in by_page[file_name]:
            x1, y1 = int(r["x_min"]), int(r["y_min"])
            x2, y2 = int(r["x_max"]), int(r["y_max"])
            w, h = x2 - x1, y2 - y1
            label = (r.get("label") or "").strip()
            text = None if label in non_text or not label else label
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CATEGORY["id"],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "text": text,
            })
            if text and r.get("crop"):
                stem = Path(file_name).stem
                crops.append({
                    # every page directory has its own line1.jpg, so the
                    # flattened name needs the page in it
                    "file_name": f"{stem}__{r['crop']}",
                    "src": Path(dataset_root) / stem / "lines" / r["crop"],
                    "text": text,
                })

    coco = {"info": {"description": "hand-labeled real scans"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [CATEGORY]}
    return coco, crops, missing


def collect_images(coco, dataset_root, images_out, copy=True):
    """Flatten the page images into one directory next to the COCO file."""
    out = Path(images_out)
    out.mkdir(parents=True, exist_ok=True)
    for img in coco["images"]:
        src = find_page(dataset_root, img["file_name"])
        dst = out / img["file_name"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
    return len(coco["images"])


def write_crops(crops, out_dir):
    """metadata.jsonl + images/, the layout the OCR training side expects."""
    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    written, missing = 0, []
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for c in crops:
            if not Path(c["src"]).exists():
                missing.append(c["file_name"])
                continue
            shutil.copy2(c["src"], images / c["file_name"])
            f.write(json.dumps({"file_name": c["file_name"],
                                "text": c["text"]}, ensure_ascii=False) + "\n")
            written += 1
    return written, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset-root", required=True,
                    help="directory holding the page images")
    ap.add_argument("--out", required=True, help="COCO output path")
    ap.add_argument("--images-out",
                    help="flatten page images here (needed for to_yolo)")
    ap.add_argument("--crops-dir",
                    help="write metadata.jsonl + images/ with the transcribed "
                         "line crops (real-scan test set for the OCR side)")
    ap.add_argument("--non-text", default=",".join(NON_TEXT),
                    help="label values that mean 'boxed but not transcribed'")
    args = ap.parse_args(argv)

    rows = read_rows(args.csv)
    coco, crops, missing = build(rows, args.dataset_root,
                                 tuple(args.non_text.split(",")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, ensure_ascii=False))

    n_text = sum(1 for a in coco["annotations"] if a["text"])
    print(f"{out}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} lines, {n_text} with a transcription")
    if missing:
        print(f"  page image not found ({len(missing)}): "
              + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""))

    if args.images_out:
        n = collect_images(coco, args.dataset_root, args.images_out)
        print(f"{args.images_out}: {n} page images")
    if args.crops_dir:
        written, gone = write_crops(crops, args.crops_dir)
        print(f"{args.crops_dir}: {written} line crops with text")
        if gone:
            print(f"  crop file not found ({len(gone)}): "
                  + ", ".join(gone[:5]) + ("..." if len(gone) > 5 else ""))


if __name__ == "__main__":
    main()
