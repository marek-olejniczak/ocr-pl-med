"""Build the unified templates/ directory for the dataset generator.

Merges two sources into one COCO annotations.json + per-page folders:

1. The new labeled dataset (dataset_30zPierwszegoPDF): COCO annotations with
   p/n/t/f/mix categories and per-page folders containing _blank/_partial
   PNGs. Only _blank/_partial images are copied (originals and lines/ stay
   behind — they contain unmasked or redundant data).

2. Old templates (forms/segmentation/images) labeled via the flat CSV from
   labeling_tool.py. Labels are mapped printed->p, text->t, number->n (plus
   legacy field-specific labels). The template PNG is already an unfilled
   form, so it becomes its own _blank.

Usage:
    python src/build_templates.py \
        --new-dataset dataset_30zPierwszegoPDF \
        --old-images forms/segmentation/images \
        --old-csv "C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv" \
        --output templates
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "p"},
    {"id": 2, "name": "n"},
    {"id": 3, "name": "t"},
    {"id": 4, "name": "f"},
    {"id": 5, "name": "mix"},
]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}

# Old generic labels
_OLD_GENERIC = {"printed": "p", "text": "t", "number": "n"}
# Legacy field-specific labels from early labeling sessions
_OLD_NUMBERISH = {
    "pesel", "pesel_grid", "date", "full_date", "date_of_birth", "phone",
    "phone_num", "telefon", "icd_10", "icd10", "icd", "icd_code", "age",
    "lat", "year", "rok", "day_and_month", "last_2_digits_year",
}
_OLD_TEXTISH = {
    "city", "miasto", "name", "name_and_surname", "patient_name", "address",
    "adres", "diagnosis", "rozpoznanie", "approval", "hospital",
    "hospital_name", "szpital", "full_signature", "doctor", "doctor_name",
    "lekarz", "department", "oddzial",
}


def map_old_label(label: str) -> Optional[str]:
    """Map an old CSV label to the new p/t/n scheme; None for junk labels."""
    l = label.strip().lower()
    if l in _OLD_GENERIC:
        return _OLD_GENERIC[l]
    if l in _OLD_NUMBERISH:
        return "n"
    if l in _OLD_TEXTISH:
        return "t"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified templates/ directory.")
    parser.add_argument("--new-dataset", type=str, default="dataset_30zPierwszegoPDF")
    parser.add_argument("--old-images", type=str, default="forms/segmentation/images")
    parser.add_argument("--old-csv", type=str,
                        default="C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv")
    parser.add_argument("--output", type=str, default="templates")
    return parser.parse_args()


def build(new_dataset: Path, old_images: Path, old_csv: Path, output: Path) -> None:
    """Build the unified templates/ directory from the two source layouts.

    See module docstring for the merge rules. Writes PNGs and
    ``annotations.json`` under ``output``.
    """
    out_dir = output
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_out: dict = {"images": [], "annotations": [], "categories": CATEGORIES}
    next_image_id = 1
    next_ann_id = 1

    # --- 1) New dataset: copy _blank/_partial + re-id annotations ---
    new_root = new_dataset
    with open(new_root / "annotations.json", "r", encoding="utf-8") as f:
        src = json.load(f)
    src_cats = {c["id"]: c["name"] for c in src["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for a in src["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    n_new_pages = 0
    for im in src["images"]:
        stem = Path(im["file_name"]).stem
        src_dir = new_root / stem
        blank_src = src_dir / f"{stem}_blank.png"
        if not blank_src.exists():
            print(f"  SKIP {stem}: no _blank.png in source", file=sys.stderr)
            continue
        dst_dir = out_dir / stem
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(blank_src, dst_dir / blank_src.name)
        partial_src = src_dir / f"{stem}_partial.png"
        if partial_src.exists():
            shutil.copy2(partial_src, dst_dir / partial_src.name)

        image_id = next_image_id
        next_image_id += 1
        coco_out["images"].append({
            "id": image_id,
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        })
        for a in anns_by_image.get(im["id"], []):
            name = src_cats.get(a["category_id"])
            if name not in CAT_ID:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            coco_out["annotations"].append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": CAT_ID[name],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            next_ann_id += 1
        n_new_pages += 1

    # --- 2) Old templates: CSV labels -> p/t/n, template PNG becomes _blank ---
    rows_by_file: dict[str, list[dict]] = {}
    with open(old_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_file.setdefault(row["filename"].strip(), []).append(row)

    n_old_pages = 0
    n_skipped_labels = 0
    for filename, rows in sorted(rows_by_file.items()):
        img_src = old_images / filename
        if not img_src.exists():
            print(f"  SKIP old '{filename}': not in {old_images}", file=sys.stderr)
            continue
        stem = Path(filename).stem
        dst_dir = out_dir / stem
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(img_src, dst_dir / f"{stem}_blank.png")
        with Image.open(img_src) as img:
            width, height = img.size

        image_id = next_image_id
        next_image_id += 1
        coco_out["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        for row in rows:
            label = map_old_label(row["label"])
            if label is None:
                n_skipped_labels += 1
                continue
            x1, y1 = int(row["x_min"]), int(row["y_min"])
            x2, y2 = int(row["x_max"]), int(row["y_max"])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            coco_out["annotations"].append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": CAT_ID[label],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            next_ann_id += 1
        n_old_pages += 1

    with open(out_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False)

    print(f"Done. {n_new_pages} new + {n_old_pages} migrated pages -> {out_dir}/")
    print(f"  {len(coco_out['annotations'])} annotations total")
    if n_skipped_labels:
        print(f"  Skipped {n_skipped_labels} rows with unknown labels")


def main() -> None:
    args = parse_args()
    build(
        new_dataset=Path(args.new_dataset),
        old_images=Path(args.old_images),
        old_csv=Path(args.old_csv),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
