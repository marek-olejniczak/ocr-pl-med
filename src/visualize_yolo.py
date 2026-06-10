"""Visualize COCO annotations by overlaying bboxes on a dataset image.

Usage:
    python visualize_yolo.py coco_dataset/images/foo_0001.jpg
    # → reads ../annotations.json relative to the image, writes foo_0001_viz.jpg

Used to spot-check that ground-truth bboxes hug the rendered text.
For source-colored boxes (printed vs rendered) use visualize_bboxes.py,
which reads ground_truth.csv instead.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay COCO bboxes on an image for visual check.")
    parser.add_argument("image", type=str, help="Path to a dataset image (.jpg/.png)")
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to COCO annotations.json (default: ../annotations.json relative to image)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <image_stem>_viz.jpg next to source)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: {img_path} not found", file=sys.stderr)
        sys.exit(1)

    ann_path = (
        Path(args.annotations) if args.annotations
        else img_path.parent.parent / "annotations.json"
    )
    if not ann_path.exists():
        print(f"ERROR: annotations file not found: {ann_path}", file=sys.stderr)
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_entry = next(
        (im for im in coco["images"] if im["file_name"] == img_path.name), None
    )
    if image_entry is None:
        print(f"ERROR: {img_path.name} not present in {ann_path}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    n = 0
    for ann in coco["annotations"]:
        if ann["image_id"] != image_entry["id"]:
            continue
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
        n += 1

    out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_viz.jpg")
    img.save(out_path, quality=90)
    print(f"Drew {n} bbox(es) on {out_path}")


if __name__ == "__main__":
    main()
