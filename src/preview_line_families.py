"""Contact sheets that show each OCR-line augmentation family on its own.

For every family the sheet forces that family ON at probability 1.0 and turns
every other phase-1 family OFF, so what you see is the effect of that family
alone against the old renderer. Two more sheets bracket the set: the old
renderer with everything off, and the new renderer with all defaults. A sheet
of real test crops sits alongside for comparison, and a font sheet shows the
letterforms of "k" and "a" in every hand.

Usage:
    python src/preview_line_families.py --output-dir output/phase1_samples \\
        --real-dataset "C:/Users/tomek/Desktop/ocr-main/dataset" --seed 7
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_ocr_lines as gol
import line_effects
import ocr_content
from fill_form import EXCLUDED_FONTS
from renderer import find_fonts
from vocabulary import Vocabulary

SHEET_WIDTH = 1100
CAPTION_FONT = "resources/fonts_print"
PER_SHEET = 12

# family -> (probability constants to force, LINE_KINDS override)
ISOLATION = {
    "short_words": ({}, {"word": 1.0}),
    "caps": ({}, {"heading": 0.6, "phrase": 0.4}),
    "arrows_bullets": ({}, {"bullet": 0.5, "arrow": 0.5}),
    "anatomy_vocab": ({}, {"phrase": 0.7, "word": 0.3}),
    "neighbour_glyphs": ({"NEIGHBOUR_GLYPH_PROB": 1.0}, None),
    "grid_paper": ({"GRID_PAPER_PROB": 1.0}, None),
    "morphology": ({"MORPHOLOGY_PROB": 1.0}, None),
    "elastic": ({"ELASTIC_PROB": 1.0}, None),
    "phone_photo": ({"PHONE_PHOTO_PROB": 1.0}, None),
}
CONTENT_ONLY = set(ocr_content.CONTENT_FAMILIES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", default="output/phase1_samples")
    p.add_argument("--font-dir", default="resources/fonts")
    p.add_argument("--resource-dir", default="resources")
    p.add_argument("--real-dataset", default=None,
                   help="Folder with annotations.csv and <page>/lines/*.jpg (real crops).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per-sheet", type=int, default=PER_SHEET)
    return p.parse_args()


def caption_font(size: int = 15) -> ImageFont.FreeTypeFont:
    for path in find_fonts(CAPTION_FONT):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(items: list[tuple[Image.Image, str]], title: str) -> Image.Image:
    """Stack (image, caption) pairs vertically at native size."""
    font = caption_font()
    title_font = caption_font(20)
    pad = 10
    rows = []
    for img, cap in items:
        if img.width > SHEET_WIDTH - 2 * pad:
            scale = (SHEET_WIDTH - 2 * pad) / img.width
            img = img.resize((int(img.width * scale), max(1, int(img.height * scale))), Image.LANCZOS)
            cap += f"   [scaled x{scale:.2f}]"
        rows.append((img, cap))
    height = 44 + sum(r[0].height + 24 + pad for r in rows)
    sheet = Image.new("RGB", (SHEET_WIDTH, height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 10), title, fill=(0, 0, 0), font=title_font)
    y = 44
    for img, cap in rows:
        sheet.paste(img, (pad, y))
        draw.rectangle([pad - 1, y - 1, pad + img.width, y + img.height], outline=(120, 120, 120))
        draw.text((pad, y + img.height + 4), cap, fill=(40, 40, 40), font=font)
        y += img.height + 24 + pad
    return sheet


def force(constants: dict[str, float]) -> dict[str, float]:
    """Set probability constants in both modules; return the originals."""
    saved = {}
    for name, value in constants.items():
        saved[name] = getattr(gol, name)
        setattr(gol, name, value)
        if hasattr(line_effects, name):
            setattr(line_effects, name, value)
    return saved


def render_family(family: str | None, n: int, vocab, pools, fonts, config, mode: str) -> list:
    """Render n lines. mode: 'isolate' one family, 'baseline', or 'all'."""
    if mode == "baseline":
        enabled = set()
    elif mode == "all":
        enabled = set(gol.ALL_FAMILIES)
    else:
        enabled = {family}
        if family in CONTENT_ONLY:
            enabled.add("anatomy_vocab")   # word/heading kinds live in the pools

    consts, kinds = ISOLATION.get(family, ({}, None)) if mode == "isolate" else ({}, None)
    saved = force(consts)
    saved_kinds = ocr_content.LINE_KINDS
    saved_cap = ocr_content.CAPITALISE_PROB
    if kinds:
        ocr_content.LINE_KINDS = kinds
    if family == "caps" and mode == "isolate":
        ocr_content.CAPITALISE_PROB = 1.0
    try:
        out = []
        tries = 0
        while len(out) < n and tries < n * 10:
            tries += 1
            img, text, meta = gol.render_line(
                vocab, fonts, config, apply_scan=True,
                pools=pools if "anatomy_vocab" in enabled else None,
                enabled=enabled)
            if img is None:
                continue
            detail = meta.get("kind", "")
            for key in ("neighbours", "rule", "morphology", "elastic", "scan_profile"):
                if meta.get(key):
                    detail += f"  {key}={meta[key]}"
            out.append((img, f"{text}    |  {meta['font']} {meta['font_size']}px  {detail}"))
        return out
    finally:
        force(saved)
        ocr_content.LINE_KINDS = saved_kinds
        ocr_content.CAPITALISE_PROB = saved_cap


def font_sheet(fonts: list[str]) -> Image.Image:
    """Every hand writing the letters the model confuses most."""
    sample = "kakao hobby kłykieć łokieć aorta żyła"
    items = []
    for path in sorted(fonts, key=lambda p: Path(p).name):
        try:
            f = ImageFont.truetype(path, 40)
        except OSError:
            continue
        w = int(f.getlength(sample)) + 20
        img = Image.new("RGB", (max(50, w), 64), (252, 250, 247))
        ImageDraw.Draw(img).text((10, 6), sample, fill=(20, 20, 28), font=f)
        items.append((img, Path(path).name))
    return contact_sheet(items, "Fonty: jak każda ręka pisze k / a / ł / ę  (\"kakao hobby kłykieć łokieć aorta żyła\")")


def real_sheet(dataset_dir: Path, n: int) -> Image.Image | None:
    ann = dataset_dir / "annotations.csv"
    if not ann.exists():
        return None
    rows = list(csv.DictReader(open(ann, encoding="utf-8")))
    random.shuffle(rows)
    items = []
    for r in rows:
        page = Path(r["filename"]).stem
        path = dataset_dir / page / "lines" / r["crop"]
        if not path.exists():
            continue
        items.append((Image.open(path).convert("RGB"), f"{r['label']}    |  {page}/{r['crop']}"))
        if len(items) >= n:
            break
    return contact_sheet(items, "PRAWDZIWE wycinki z testsetu (dla porównania)") if items else None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(args.resource_dir)
    pools = ocr_content.LinePools(args.resource_dir)
    fonts = [f for f in find_fonts(args.font_dir) if Path(f).name not in EXCLUDED_FONTS]
    config = gol.build_line_config()
    n = args.per_sheet

    sheets = [("00_real", None)]
    if args.real_dataset:
        sheet = real_sheet(Path(args.real_dataset), n)
        if sheet:
            sheet.save(out / "00_prawdziwe_wycinki.png")
    contact_sheet(render_family(None, n, vocab, pools, fonts, config, "baseline"),
                  "BASELINE: stary renderer, wszystkie rodziny wyłączone").save(out / "01_baseline.png")
    for i, family in enumerate(gol.ALL_FAMILIES, start=2):
        items = render_family(family, n, vocab, pools, fonts, config, "isolate")
        contact_sheet(items, f"RODZINA: {family}  (wymuszona 100%, reszta wyłączona)").save(
            out / f"{i:02d}_{family}.png")
        print(f"  {family}: {len(items)} lines")
    contact_sheet(render_family(None, n, vocab, pools, fonts, config, "all"),
                  "WSZYSTKO: nowy renderer z domyślnymi prawdopodobieństwami").save(out / "20_wszystko.png")
    font_sheet(fonts).save(out / "21_fonty_k_a.png")
    print(f"Sheets in {out}/")


if __name__ == "__main__":
    main()
