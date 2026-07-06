"""Generate a COCO line-detection training dataset from labeled form templates.

Reads `dataset/annotations.csv` (produced by labeling_tool.py) which marks
WHERE on each blank form template text can be written, generates many random
variants per template, and writes a COCO-format dataset:

    output-dir/
    ├── images/
    │   ├── {template_stem}_0001.jpg
    │   └── ...
    ├── annotations.json               (COCO: images / annotations / categories)
    ├── ground_truth.csv               (same bboxes, flat CSV for easy metrics)
    └── metadata/                      (optional: --no-metadata)
        └── {template_stem}_0001.json

COCO annotation format: bbox = [x_min, y_min, width, height] in absolute
pixels, one category: "text_line" (id=1).

The ground-truth bboxes are computed automatically from the actually-inked
pixels (not the original field bbox), so they tightly match the rendered
text — independent of how long the random content turned out to be.

Usage:
    python generate_yolo_dataset.py \\
        --forms-dir forms/segmentation/images \\
        --annotations dataset/annotations.csv \\
        --output-dir coco_dataset \\
        --variants-per-form 111
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

# Reuse helpers from fill_form.py and friends
from vocabulary import Vocabulary
from renderer import find_fonts
from fill_form import (
    EMPTY_FIELD_PROB,
    EXCLUDED_FONTS,
    fill_single_form,
    load_annotations,
)
from transforms import AugmentConfig, TransformPipeline


CLASS_NAME = "text_line"
COCO_CATEGORY_ID = 1  # COCO category ids start at 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO line-detection dataset from labeled forms."
    )
    parser.add_argument(
        "--forms-dir",
        type=str,
        required=True,
        help="Directory containing form template images (PNG/JPG).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to dataset/annotations.csv (from labeling_tool).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write the YOLO dataset (will be created).",
    )
    parser.add_argument(
        "--variants-per-form",
        type=int,
        default=111,
        help="Number of filled variants to generate per template (default: 111 ~ 1000 total over 9 templates).",
    )
    parser.add_argument(
        "--font-dir",
        type=str,
        default="resources/fonts",
        help="Font directory (default: resources/fonts).",
    )
    parser.add_argument(
        "--resource-dir",
        type=str,
        default="resources",
        help="Resource directory (default: resources).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing per-image JSON metadata (image+labels only).",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable scan/photo simulation (clean output, perfect bboxes).",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Re-enable page rotation (OFF by default — axis-aligned bboxes "
             "get looser on rotated text, hurting IoU).",
    )
    return parser.parse_args()


def list_csv_filenames(csv_path: Path) -> set[str]:
    """Return the unique `filename` values present in the annotations CSV."""
    out: set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if fn:
                out.add(fn)
    return out


def build_form_filling_config() -> AugmentConfig:
    """Build the same toned-down AugmentConfig that fill_form.main() uses.

    Keeps form filling behavior consistent between the CLI and the YOLO
    batch generator.
    """
    config = AugmentConfig()
    config.char.rotation_max_deg = 2.5
    config.char.scale_min = 0.95
    config.char.scale_max = 1.05
    config.line.baseline_wander_amplitude = 1.5
    config.line.spacing_jitter_px = 0.8
    config.line.slant_max_deg = 4.0
    config.line.baseline_drift_max_px = 4.0  # line gradually climbs/falls
    config.paper.enabled = False
    config.scan.enabled = False
    return config


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> Optional[tuple[int, int, int, int]]:
    """Clamp (x_min, y_min, x_max, y_max) to image bounds.

    Returns the clamped (x_min, y_min, x_max, y_max), or None if the bbox
    has degenerate area (< 1 px) after clamping.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_w, int(x1)))
    y1 = max(0, min(image_h, int(y1)))
    x2 = max(0, min(image_w, int(x2)))
    y2 = max(0, min(image_h, int(y2)))
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return (x1, y1, x2, y2)


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    forms_dir = Path(args.forms_dir)
    if not forms_dir.is_dir():
        print(f"ERROR: forms-dir not found: {forms_dir}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.annotations)
    if not csv_path.exists():
        print(f"ERROR: annotations CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Set up output structure
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_metadata:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    # COCO skeleton — filled incrementally, written once at the end
    coco: dict = {
        "info": {
            "description": "Synthetic Polish medical forms — text line detection",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": COCO_CATEGORY_ID, "name": CLASS_NAME, "supercategory": "text"}
        ],
    }
    next_image_id = 1
    next_ann_id = 1

    # Open ground_truth.csv for cumulative writing (one row per bbox).
    # Same shape as dataset/annotations.csv so existing tooling works.
    csv_out_path = output_dir / "ground_truth.csv"
    csv_out = open(csv_out_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["filename", "label", "x_min", "y_min", "x_max", "y_max", "source"])

    # Discover labeled templates that physically exist in forms-dir
    csv_filenames = list_csv_filenames(csv_path)
    templates: list[Path] = []
    for fn in sorted(csv_filenames):
        p = forms_dir / fn
        if p.exists():
            templates.append(p)
        else:
            print(f"  SKIP: '{fn}' in CSV but not found in {forms_dir}", file=sys.stderr)

    if not templates:
        print("ERROR: no labeled templates exist in forms-dir", file=sys.stderr)
        sys.exit(1)

    print(f"Templates to fill: {len(templates)}")
    for t in templates:
        n = len(load_annotations(csv_path, t.name))
        print(f"  {t.name} ({n} annotations)")

    # Load vocab + fonts
    print("Loading vocabulary...")
    vocab = Vocabulary(args.resource_dir)
    all_fonts = find_fonts(args.font_dir)
    fonts = [f for f in all_fonts if Path(f).name not in EXCLUDED_FONTS]
    if not fonts:
        print(f"ERROR: no usable fonts in {args.font_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(fonts)} fonts available")

    config = build_form_filling_config()
    pipeline = TransformPipeline(config)
    apply_scan = not args.no_scan

    n_variants = max(1, args.variants_per_form)
    print(f"Generating {n_variants} variants per template "
          f"({n_variants * len(templates)} total)")
    print(f"Scan augmentation: {'ON' if apply_scan else 'OFF'}")

    total_count = 0
    skipped_blank = 0

    for template_path in templates:
        annotations = load_annotations(csv_path, template_path.name)
        if not annotations:
            print(f"  WARN: no annotations for {template_path.name}, skipping")
            continue

        for v in range(1, n_variants + 1):
            font_path = random.choice(fonts)
            stem = f"{template_path.stem}_{v:04d}"

            result = fill_single_form(
                form_path=template_path,
                annotations=annotations,
                vocab=vocab,
                font_path=font_path,
                config=config,
                pipeline=pipeline,
                apply_scan=apply_scan,
                filler_mode=True,            # content irrelevant for line detection
                empty_field_prob=EMPTY_FIELD_PROB,
            )

            img: Image.Image = result["image"]
            iw, ih = img.size
            img_path = images_dir / f"{stem}.jpg"

            # Ground truth = `printed_bboxes` (tight around static printed text)
            # PLUS `tight_bboxes` (tight around actually-rendered fill-in text).
            # The user labels these as SEPARATE regions in labeling_tool — the
            # printed label and its fill-in space sit side by side, not nested —
            # so they produce non-overlapping annotations with accurate IoU.
            ground_truth_records = (
                [("printed", r) for r in result["printed_bboxes"]]
                + [("rendered", r) for r in result["tight_bboxes"]]
            )

            image_id = next_image_id
            next_image_id += 1
            coco["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": iw,
                "height": ih,
            })

            n_boxes = 0
            for source, rec in ground_truth_records:
                clamped = clamp_bbox(tuple(rec["bbox"]), iw, ih)
                if clamped is None:
                    skipped_blank += 1
                    continue
                x1, y1, x2, y2 = clamped
                w, h = x2 - x1, y2 - y1
                coco["annotations"].append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY_ID,
                    "bbox": [x1, y1, w, h],   # COCO: [x, y, width, height]
                    "area": w * h,
                    "iscrowd": 0,
                })
                next_ann_id += 1
                n_boxes += 1
                csv_writer.writerow([img_path.name, CLASS_NAME, x1, y1, x2, y2, source])

            # Save image
            img.save(img_path, quality=92)

            # Optional metadata
            if not args.no_metadata:
                metadata = {
                    "form_image": template_path.name,
                    "output_image": img_path.name,
                    "image_size": [iw, ih],
                    "font": result["font"],
                    "ink_color": result["ink_color"],
                    "fields": result["tight_bboxes"],
                    "printed_lines": result["printed_bboxes"],
                    "num_annotations": n_boxes,
                }
                if result["text_style"] is not None:
                    metadata["text_style"] = result["text_style"]
                if result["scan_augmentation"] is not None:
                    metadata["scan_augmentation"] = result["scan_augmentation"]
                meta_path = metadata_dir / f"{stem}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            total_count += 1
            if total_count % 50 == 0 or total_count == 1:
                print(f"  [{total_count}] {stem}.jpg  ({n_boxes} bboxes)")

    csv_out.close()

    # Write the COCO annotation file
    coco_path = output_dir / "annotations.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\nDone. {total_count} images generated in {output_dir}/")
    print(f"  images/           --> {total_count} JPGs")
    print(f"  annotations.json  --> COCO ({len(coco['annotations'])} annotations, "
          f"1 category: '{CLASS_NAME}')")
    print(f"  ground_truth.csv  --> all bboxes (filename, label, x_min, y_min, x_max, y_max, source)")
    if skipped_blank:
        print(f"  Skipped {skipped_blank} blank/degenerate bbox(es) across all images")


if __name__ == "__main__":
    main()
