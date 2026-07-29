"""Generate standalone handwritten line images for OCR training.

The line-detection model hands the OCR model cropped text lines, so this
script produces exactly that: one short image per line plus the transcription
that was written into it.

Lines are rendered from scratch rather than cropped out of generated forms,
which gives unlimited volume and full control over content — at the cost of
having to recreate the conditions a real crop arrives in. Three things do
that work here:

    paper background — the tone and grain of a scanned sheet, sometimes with
        the dotted or ruled fill-in line the writing sat on
    imperfect framing — a detector's box is never pixel-tight, so margins are
        random and occasionally clip a few pixels of ink
    scan profiles — the same clean colour / grayscale / photocopy simulation
        the full-page generator applies

Content, fonts, handwriting style and sanitization are shared with the form
filler, so the OCR model sees the same medical vocabulary and the same hands.

Usage:
    python src/generate_ocr_lines.py --output-dir output/ocr_lines --count 50000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_card import build_card, write_card, update_registry
from field_content import generate_field_content
from fill_form import (
    EXCLUDED_FONTS,
    FORM_FONT_SIZE_RANGE,
    INK_BLACK,
    INK_BLUE,
    PEN_FADE_PROB,
    _make_ink_mask,
    apply_pen_fade,
    apply_scan_augmentation,
)
from renderer import find_fonts
from transforms import AugmentConfig, WordStyle
from vocabulary import Vocabulary

# Width the line is asked to fill, in pixels. Spans a short date to a full
# diagnosis running the width of a form field.
TARGET_WIDTH_RANGE = (120, 1400)
# Content kinds and how often each is drawn: letters dominate real fill-ins.
CONTENT_KINDS = [("t", 0.70), ("n", 0.22), ("mix", 0.08)]
# Paper tone the line was written on
PAPER_BASE = (252, 250, 247)
# Margin around the ink, as a fraction of text height. The detector's box is
# never tight, so this is deliberately wide and occasionally negative.
MARGIN_FRAC_RANGE = (-0.08, 0.55)
# Probability the crop shows the fill-in rule the text was written on
RULE_PROB = 0.35
# Probability a neighbouring line bleeds into the top or bottom edge
NEIGHBOUR_BLEED_PROB = 0.18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render standalone handwritten line images for OCR training."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--count", type=int, default=50000,
                        help="Number of line images to render (default: 50000).")
    parser.add_argument("--font-dir", type=str, default="resources/fonts")
    parser.add_argument("--resource-dir", type=str, default="resources")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction held out for validation (default: 0.05).")
    parser.add_argument("--no-scan", action="store_true",
                        help="Skip scan simulation (clean renders).")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Registry file to update (default: docs/datasets.md in the repo). "
             "Pass 'none' to skip the registry — use this for throwaway runs, "
             "which should not append rows to the repo's registry.",
    )
    return parser.parse_args()


def build_line_config() -> AugmentConfig:
    """Same handwriting jitter the form filler uses, so the hands match."""
    config = AugmentConfig()
    config.char.rotation_max_deg = 2.5
    config.char.scale_min = 0.95
    config.char.scale_max = 1.05
    config.line.baseline_wander_amplitude = 1.5
    config.line.spacing_jitter_px = 0.8
    config.line.baseline_drift_max_px = 4.0
    config.paper.enabled = False
    config.scan.enabled = False
    return config


def pick_content_kind() -> str:
    kinds, weights = zip(*CONTENT_KINDS)
    return random.choices(kinds, weights=weights, k=1)[0]


def make_measure(font_path: str, font_size: int, style: WordStyle):
    """Width estimator matching what the renderer will actually draw."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    tracking = style.tracking_ratio * font_size
    return lambda t: font.getlength(t) * style.x_stretch + tracking * len(t)


def paper_canvas(width: int, height: int) -> Image.Image:
    """A slice of scanned paper: warm-white with a little grain."""
    tone = tuple(
        max(0, min(255, c + random.randint(-4, 3))) for c in PAPER_BASE
    )
    canvas = Image.new("RGB", (width, height), tone)
    grain = np.random.normal(0, 1.6, (height, width, 1)).astype(np.float32)
    arr = np.clip(np.array(canvas, dtype=np.float32) + grain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def draw_rule(canvas: Image.Image, baseline_y: int, ink: tuple[int, int, int]) -> str:
    """Draw the fill-in line the text was written on. Returns its kind."""
    draw = ImageDraw.Draw(canvas)
    faded = tuple(min(255, c + random.randint(60, 110)) for c in ink)
    y = min(canvas.height - 1, baseline_y)
    if random.random() < 0.6:
        step = random.randint(5, 9)
        for x in range(random.randint(0, step), canvas.width, step):
            draw.point((x, y), fill=faded)
        return "dotted"
    draw.line([(0, y), (canvas.width, y)], fill=faded, width=1)
    return "solid"


def bleed_neighbour(canvas: Image.Image, ink: tuple[int, int, int]) -> None:
    """Clip a sliver of the line above or below into the frame.

    A tight crop around one line often catches the descenders of the line
    above or the ascenders of the one below; the OCR model has to learn to
    ignore ink that is cut off by the frame edge.
    """
    draw = ImageDraw.Draw(canvas)
    from_top = random.random() < 0.5
    faded = tuple(min(255, c + random.randint(20, 60)) for c in ink)
    for _ in range(random.randint(2, 6)):
        x = random.uniform(0, canvas.width)
        length = random.uniform(4, 18)
        if from_top:
            y0, y1 = -random.uniform(2, 6), random.uniform(1, 4)
        else:
            y0 = canvas.height - random.uniform(1, 4)
            y1 = canvas.height + random.uniform(2, 6)
        draw.line([(x, y0), (x + random.uniform(-3, 3), y1)],
                  fill=faded, width=random.randint(1, 2))


def render_line(
    vocab: Vocabulary, fonts: list[str], config: AugmentConfig, apply_scan: bool
) -> tuple[Image.Image, str, dict]:
    """Render one handwritten line on paper. Returns (image, text, metadata)."""
    font_path = random.choice(fonts)
    font_size = random.randint(*FORM_FONT_SIZE_RANGE)
    style = WordStyle.random(config.char)
    style.do_thicken = False

    base_ink = random.choice([INK_BLACK, INK_BLUE])
    ink = tuple(max(0, min(255, c + random.randint(-8, 8))) for c in base_ink)

    kind = pick_content_kind()
    target_width = random.randint(*TARGET_WIDTH_RANGE)
    measure = make_measure(font_path, font_size, style)
    text = generate_field_content(
        kind, vocab, measure, target_width, font_path=font_path
    )
    if not text:
        return None, "", {}

    from char_renderer import render_text_per_char

    text_img, _ = render_text_per_char(
        text, font_path, font_size, padding=2, config=config, word_style=style
    )
    if random.random() < PEN_FADE_PROB:
        text_img = apply_pen_fade(text_img)

    mask = _make_ink_mask(text_img)
    ink_box = mask.getbbox()
    if ink_box is None:
        return None, "", {}
    text_img = text_img.crop(ink_box)
    mask = mask.crop(ink_box)

    # Framing: a detector's box is never tight, and sometimes it clips.
    margins = [
        int(text_img.height * random.uniform(*MARGIN_FRAC_RANGE)) for _ in range(4)
    ]
    left, top, right, bottom = margins
    width = max(8, text_img.width + left + right)
    height = max(8, text_img.height + top + bottom)

    canvas = paper_canvas(width, height)
    baseline = top + text_img.height + max(1, int(text_img.height * 0.10))
    rule = None
    if random.random() < RULE_PROB and 0 < baseline < height:
        rule = draw_rule(canvas, baseline, ink)
    if random.random() < NEIGHBOUR_BLEED_PROB:
        bleed_neighbour(canvas, ink)

    ink_layer = Image.new("RGB", text_img.size, ink)
    canvas.paste(ink_layer, (left, top), mask)

    scan_meta = None
    if apply_scan:
        canvas, scan_meta = apply_scan_augmentation(canvas)

    meta = {
        "font": Path(font_path).name,
        "font_size": font_size,
        "kind": kind,
        "ink": list(ink),
        "rule": rule,
        "size": list(canvas.size),
        "scan_profile": scan_meta["profile"] if scan_meta else None,
    }
    return canvas, text, meta


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary...")
    vocab = Vocabulary(args.resource_dir)
    fonts = [f for f in find_fonts(args.font_dir) if Path(f).name not in EXCLUDED_FONTS]
    if not fonts:
        print(f"ERROR: no usable fonts in {args.font_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(fonts)} fonts available")

    config = build_line_config()
    apply_scan = not args.no_scan
    print(f"Rendering {args.count} lines (scan simulation: "
          f"{'ON' if apply_scan else 'OFF'})")

    fonts_used: set[str] = set()
    profiles: dict[str, int] = {}
    kinds: dict[str, int] = {}
    skipped = 0
    index = 0
    n_val = 0

    # Labels are written as each line is rendered, not collected in memory and
    # dumped at the end: a run of this length WILL sometimes be interrupted,
    # and images without transcriptions are worthless for OCR training. The
    # label files are the source of truth — a killed run leaves a consistent
    # dataset covering however many lines it managed to finish.
    handles = {
        "all": open(output_dir / "labels.txt", "w", encoding="utf-8"),
        "train": open(output_dir / "labels_train.txt", "w", encoding="utf-8"),
        "val": open(output_dir / "labels_val.txt", "w", encoding="utf-8"),
        "jsonl": open(output_dir / "labels.jsonl", "w", encoding="utf-8"),
    }
    try:
        while index < args.count:
            image, text, meta = render_line(vocab, fonts, config, apply_scan)
            if image is None:
                skipped += 1
                if skipped > args.count:
                    print("ERROR: content generation kept coming back empty",
                          file=sys.stderr)
                    sys.exit(1)
                continue

            name = f"{index:07d}.jpg"
            image.save(images_dir / name, quality=93)
            split = "val" if random.random() < args.val_frac else "train"
            path = f"images/{name}"

            handles["all"].write(f"{path}\t{text}\n")
            handles[split].write(f"{path}\t{text}\n")
            handles["jsonl"].write(json.dumps(
                {"file_name": path, "text": text, "split": split, **meta},
                ensure_ascii=False) + "\n")

            fonts_used.add(meta["font"])
            kinds[meta["kind"]] = kinds.get(meta["kind"], 0) + 1
            if meta["scan_profile"]:
                profiles[meta["scan_profile"]] = (
                    profiles.get(meta["scan_profile"], 0) + 1
                )
            n_val += int(split == "val")

            index += 1
            # Flush often so a killed run's four label files stay in step with
            # each other; flush() only reaches the OS buffer, so it is cheap.
            if index % 200 == 0:
                for handle in handles.values():
                    handle.flush()
            if index % 1000 == 0 or index == 1:
                print(f"  [{index}] {name}  {text[:48]!r}")
    finally:
        for handle in handles.values():
            handle.close()

    n_lines = index
    repo_root = Path(__file__).resolve().parent.parent
    card = build_card(
        name=args.dataset_name or output_dir.name,
        command="python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
        seed=args.seed,
        repo_root=repo_root,
        counts={"lines": n_lines, "train": n_lines - n_val, "val": n_val},
        sources={"synthetic": n_lines},
        fonts=sorted(fonts_used),
        observed={"scan_profiles": profiles, "content_kinds": kinds,
                  "empty_content_retries": skipped},
        note=args.note or "standalone OCR line renderer",
    )
    # Page-level artifacts do not apply to single-line renders.
    card["augmentations"] = {
        k: v for k, v in card["augmentations"].items()
        if not k.startswith(("stamp_", "stroke_", "highlight_", "partial_",
                             "multiline_", "line_pitch"))
    }
    card["augmentations"]["target_width_range"] = list(TARGET_WIDTH_RANGE)
    card["augmentations"]["margin_frac_range"] = list(MARGIN_FRAC_RANGE)
    card["augmentations"]["rule_prob"] = RULE_PROB
    card["augmentations"]["neighbour_bleed_prob"] = NEIGHBOUR_BLEED_PROB
    card["augmentations"]["content_kinds"] = [list(k) for k in CONTENT_KINDS]
    card_path = write_card(output_dir, card)
    if args.registry != "none":
        registry_path = (
            Path(args.registry) if args.registry
            else repo_root / "docs" / "datasets.md"
        )
        update_registry(registry_path, card)

    print(f"\nDone. {n_lines} lines in {output_dir}/")
    print(f"  images/           --> {n_lines} JPGs")
    print(f"  labels.txt        --> path<TAB>text (all)")
    print(f"  labels_train.txt  --> {n_lines - n_val} lines")
    print(f"  labels_val.txt    --> {n_val} lines")
    print(f"  labels.jsonl      --> per-line metadata (HuggingFace style)")
    print(f"  dataset_card.json --> {card_path}")


if __name__ == "__main__":
    main()
