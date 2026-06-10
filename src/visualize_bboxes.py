"""Overlay all ground-truth bboxes on a generated form image.

Reads ground_truth.csv from the dataset and draws bboxes with different
colors per source:
    - red   = `printed` (static printed text on the form template)
    - green = `rendered` (tight bbox of actually-rendered fill-in text)

Usage:
    python visualize_bboxes.py yolo_dataset/images/88_0001.jpg
    # → writes 88_0001_viz.jpg next to the source image
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw


COLOR_BY_SOURCE = {
    "printed": (255, 50, 50),    # red
    "rendered": (50, 200, 80),   # green
}
DEFAULT_COLOR = (50, 120, 255)   # blue, for any unexpected source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw ground-truth bboxes on an image.")
    parser.add_argument("image", type=str, help="Path to a generated dataset image")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to ground_truth.csv (default: ../ground_truth.csv relative to image)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <image_stem>_viz.jpg next to source)",
    )
    parser.add_argument("--width", type=int, default=2, help="Line width in px (default: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    # ground_truth.csv lives at the dataset root, one level above images/
    csv_path = Path(args.csv) if args.csv else img_path.parent.parent / "ground_truth.csv"
    if not csv_path.exists():
        print(f"ERROR: ground_truth.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    counts = {"printed": 0, "rendered": 0, "other": 0}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["filename"] != img_path.name:
                continue
            x1 = int(row["x_min"])
            y1 = int(row["y_min"])
            x2 = int(row["x_max"])
            y2 = int(row["y_max"])
            source = row.get("source", "")
            color = COLOR_BY_SOURCE.get(source, DEFAULT_COLOR)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=args.width)
            counts[source if source in counts else "other"] += 1

    out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_viz.jpg")
    img.save(out_path, quality=90)

    total = sum(counts.values())
    print(f"Drew {total} bbox(es) on {out_path}")
    print(f"  red   (printed):  {counts['printed']}")
    print(f"  green (rendered): {counts['rendered']}")
    if counts["other"]:
        print(f"  blue  (other):    {counts['other']}")


if __name__ == "__main__":
    main()
