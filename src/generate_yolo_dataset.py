"""Generate a COCO line-detection + OCR training dataset from form templates.

Input:  a templates/ directory built by build_templates.py (COCO
        annotations.json with p/n/t/f/mix labels + per-page _blank/_partial
        base images).
Output: output-dir/
        ├── images/{page}_{v:04d}.jpg
        ├── annotations.json    COCO; each annotation carries "text" (what was
        │                       written; None for printed/handwritten) and
        │                       "source" (printed|synthetic|handwritten)
        ├── ground_truth.csv    filename,label,x_min,y_min,x_max,y_max,source,text
        └── metadata/*.json     per-image generation parameters

Usage:
    python src/generate_yolo_dataset.py --templates-dir templates \
        --output-dir output/run1 --variants-per-form 25 --seed 42
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from vocabulary import Vocabulary
from renderer import find_fonts
from fill_form import EXCLUDED_FONTS, fill_single_form
from template_loader import load_templates
from transforms import AugmentConfig, TransformPipeline

CLASS_NAME = "text_line"
COCO_CATEGORY_ID = 1

# When a page has a _partial base, half the variants use it (real handwriting
# in f-fields), half use _blank (f-fields filled synthetically)
PARTIAL_BASE_PROB = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO line-detection dataset from form templates."
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="Directory containing templates (annotations.json + per-page images).",
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
    return parser.parse_args()


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

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        print(f"ERROR: templates-dir not found: {templates_dir}", file=sys.stderr)
        sys.exit(1)

    pages = load_templates(templates_dir)
    if not pages:
        print("ERROR: no usable template pages", file=sys.stderr)
        sys.exit(1)
    print(f"Templates to fill: {len(pages)}")
    for p in pages:
        partial = " (+partial)" if p.partial_path else ""
        print(f"  {p.name} ({len(p.fields)} fields){partial}")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_metadata:
        metadata_dir.mkdir(parents=True, exist_ok=True)

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

    coco: dict = {
        "info": {
            "description": "Synthetic Polish medical forms — text lines with transcriptions",
            "version": "2.0",
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

    csv_out = open(output_dir / "ground_truth.csv", "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(
        ["filename", "label", "x_min", "y_min", "x_max", "y_max", "source", "text"])

    n_variants = max(1, args.variants_per_form)
    print(f"Generating {n_variants} variants per template "
          f"({n_variants * len(pages)} total)")
    print(f"Scan augmentation: {'ON' if apply_scan else 'OFF'}")

    total_count = 0
    skipped_blank = 0

    for page in pages:
        for v in range(1, n_variants + 1):
            font_path = random.choice(fonts)
            stem = f"{page.name}_{v:04d}"

            use_partial = (
                page.partial_path is not None and random.random() < PARTIAL_BASE_PROB
            )
            base_path = page.partial_path if use_partial else page.blank_path

            result = fill_single_form(
                form_path=base_path,
                fields=page.fields,
                vocab=vocab,
                font_path=font_path,
                config=config,
                pipeline=pipeline,
                apply_scan=apply_scan,
                skip_f_fields=use_partial,
            )

            img: Image.Image = result["image"]
            iw, ih = img.size
            img_path = images_dir / f"{stem}.jpg"

            image_id = next_image_id
            next_image_id += 1
            coco["images"].append({
                "id": image_id, "file_name": img_path.name,
                "width": iw, "height": ih,
            })

            n_boxes = 0
            for rec in result["records"]:
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
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "source": rec["source"],
                    "text": rec["text"],
                })
                next_ann_id += 1
                n_boxes += 1
                csv_writer.writerow(
                    [img_path.name, CLASS_NAME, x1, y1, x2, y2,
                     rec["source"], rec["text"] or ""])

            img.save(img_path, quality=92)

            if not args.no_metadata:
                metadata = {
                    "template": page.name,
                    "base": "partial" if use_partial else "blank",
                    "output_image": img_path.name,
                    "image_size": [iw, ih],
                    "font": result["font"],
                    "ink_color": result["ink_color"],
                    "empty_field_prob": result["empty_field_prob"],
                    "multiline_fields": result["multiline_fields"],
                    "fields": result["records"],
                    "num_annotations": n_boxes,
                }
                if result["text_style"] is not None:
                    metadata["text_style"] = result["text_style"]
                if result["scan_augmentation"] is not None:
                    metadata["scan_augmentation"] = result["scan_augmentation"]
                with open(metadata_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            total_count += 1
            if total_count % 50 == 0 or total_count == 1:
                print(f"  [{total_count}] {stem}.jpg  ({n_boxes} bboxes)")

    csv_out.close()
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\nDone. {total_count} images generated in {output_dir}/")
    print(f"  annotations.json  --> COCO ({len(coco['annotations'])} annotations "
          f"with text+source)")
    if skipped_blank:
        print(f"  Skipped {skipped_blank} blank/degenerate bbox(es)")


if __name__ == "__main__":
    main()
