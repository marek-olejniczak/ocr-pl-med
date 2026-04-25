"""Main generation script for synthetic OCR training data.

Generates word/phrase-level images with ground truth labels
for training handwritten Polish medical document OCR models.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from vocabulary import Vocabulary
from renderer import render_text, find_fonts
from char_renderer import render_text_per_char
from transforms import AugmentConfig, TransformPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic OCR training data for Polish medical documents."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--font-dir",
        type=str,
        default="resources/fonts",
        help="Path to fonts directory (default: resources/fonts/)",
    )
    parser.add_argument(
        "--resource-dir",
        type=str,
        default="resources",
        help="Path to resources directory (default: resources/)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable all augmentations (produce clean baseline images)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate synthetic training samples."""
    args = parse_args()

    # Load vocabulary
    print(f"Loading vocabulary from {args.resource_dir}...")
    vocab = Vocabulary(args.resource_dir)
    print(
        f"  Loaded: {len(vocab.drugs)} drugs, {len(vocab.icd_entries)} ICD-10 entries, "
        f"{len(vocab.abbreviations)} abbreviations, {len(vocab.dosages)} dosages, "
        f"{len(vocab.diagnoses)} diagnoses, "
        f"{len(vocab.first_names_male) + len(vocab.first_names_female)} first names, "
        f"{len(vocab.last_names)} last names"
    )

    # Find fonts
    fonts = find_fonts(args.font_dir)
    if not fonts:
        print(f"ERROR: No .ttf fonts found in {args.font_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(fonts)} fonts")

    # Set up augmentation
    use_augment = not args.no_augment
    if use_augment:
        config = AugmentConfig()
        pipeline = TransformPipeline(config)
        print("  Augmentations: ENABLED")
    else:
        config = None
        pipeline = None
        print("  Augmentations: DISABLED")

    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []

    print(f"Generating {args.count} samples...")
    total_start = time.perf_counter()
    for i in range(1, args.count + 1):
        sample_start = time.perf_counter()

        # Decide: 50% single category text, 50% phrase
        if random.random() < 0.5:
            text, category = vocab.get_random_text()
        else:
            text, category = vocab.get_random_phrase()

        # Pick random font and size
        font_path = random.choice(fonts)
        font_name = os.path.basename(font_path)
        font_size = random.randint(32, 72)

        # Render
        render_start = time.perf_counter()
        if use_augment:
            # Per-character rendering with char-level transforms
            img, bbox = render_text_per_char(text, font_path, font_size, config=config)
            # Apply post-render transforms (slant, paper, scan)
            img = pipeline.apply(img)
        else:
            # Clean rendering
            img, bbox = render_text(text, font_path, font_size)
        render_ms = (time.perf_counter() - render_start) * 1000

        # Save image
        sample_id = f"{i:06d}"
        img_filename = f"{sample_id}.png"
        img_relpath = f"images/{img_filename}"
        save_start = time.perf_counter()
        img.save(images_dir / img_filename)
        save_ms = (time.perf_counter() - save_start) * 1000

        total_ms = (time.perf_counter() - sample_start) * 1000

        # Record metadata with timing
        record = {
            "id": sample_id,
            "image": img_relpath,
            "text": text,
            "category": category,
            "font": font_name,
            "font_size": font_size,
            "bbox": bbox,
            "augmented": use_augment,
            "timing_ms": {
                "render": round(render_ms, 2),
                "save": round(save_ms, 2),
                "total": round(total_ms, 2),
            },
        }
        labels.append(record)

        if i % 100 == 0 or i == args.count:
            print(f"  {i}/{args.count} done (last sample: {total_ms:.1f}ms)")

    total_elapsed = time.perf_counter() - total_start

    # Compute aggregate timing stats
    render_times = [r["timing_ms"]["render"] for r in labels]
    save_times = [r["timing_ms"]["save"] for r in labels]
    total_times = [r["timing_ms"]["total"] for r in labels]

    timing_summary = {
        "total_elapsed_seconds": round(total_elapsed, 2),
        "samples_per_second": round(args.count / total_elapsed, 2) if total_elapsed > 0 else 0,
        "per_sample_ms": {
            "render_avg": round(sum(render_times) / len(render_times), 2),
            "render_min": round(min(render_times), 2),
            "render_max": round(max(render_times), 2),
            "save_avg": round(sum(save_times) / len(save_times), 2),
            "total_avg": round(sum(total_times) / len(total_times), 2),
            "total_min": round(min(total_times), 2),
            "total_max": round(max(total_times), 2),
        },
    }

    # Save labels
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # Save timing summary
    timing_path = output_dir / "timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"Done! {args.count} samples saved to {output_dir}/ in {total_elapsed:.2f}s")
    print(f"  Speed: {timing_summary['samples_per_second']} samples/sec")
    print(f"  Avg per sample: {timing_summary['per_sample_ms']['total_avg']}ms "
          f"(render {timing_summary['per_sample_ms']['render_avg']}ms, "
          f"save {timing_summary['per_sample_ms']['save_avg']}ms)")
    print(f"  Images: {images_dir}/")
    print(f"  Labels: {labels_path}")
    print(f"  Timing: {timing_path}")


if __name__ == "__main__":
    main()
