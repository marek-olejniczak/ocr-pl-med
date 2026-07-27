"""Physical artifacts that end up on a filled paper form.

Applied after the text is written and before the scan simulation, because a
scanner captures them like anything else lying on the page:

    stray pen strokes — the pen accidentally touching the paper
    stamps            — clinic/doctor stamps, usually slapped on at an angle
    highlighter       — someone marking a line they care about

None of them change the image geometry, so ground-truth boxes produced by the
filling step stay valid to the pixel. Only rectangular stamps pressed straight
enough (|angle| <= STAMP_GT_MAX_ANGLE) contribute new boxes; everything else
here is deliberately unlabelled — the detector should learn that not every
dark mark on paper is a line of text.
"""

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from text_sanitize import sanitize
from transforms import rotate_bbox
from vocabulary import Vocabulary

# --- stray pen strokes ---
STROKE_PROB = 0.20
STROKE_COUNT_RANGE = (1, 3)
_STROKE_KINDS = ("sweep", "tick", "scribble")


def _stroke_points(
    x: float, y: float, length: float, angle_deg: float, curvature: float
) -> list[tuple[float, float]]:
    """Walk a pen tip across the page, drifting slightly each step."""
    points: list[tuple[float, float]] = []
    angle = math.radians(angle_deg)
    step = 4.0
    for _ in range(max(2, int(length / step))):
        points.append((x, y))
        angle += curvature + random.uniform(-0.03, 0.03)
        x += step * math.cos(angle)
        y += step * math.sin(angle)
    return points


def _scribble_points(x: float, y: float, width_px: float) -> list[tuple[float, float]]:
    """A tight back-and-forth zigzag, like someone testing whether a pen works."""
    n = random.randint(4, 9)
    dx = width_px / n
    amplitude = random.uniform(8.0, 26.0)
    return [
        (x + i * dx, y + (amplitude if i % 2 else -amplitude) + random.uniform(-3, 3))
        for i in range(n + 1)
    ]


def draw_stray_strokes(form: Image.Image, ink_color: tuple[int, int, int]) -> dict:
    """Draw 1-3 accidental pen marks on the form, in place.

    Uses the same pen as the handwriting — it is the same person's hand
    brushing the paper. No ground truth is produced: these are not text, and
    the detector benefits from seeing dark marks it must NOT report.

    Args:
        form: Filled form image (RGB), modified in place.
        ink_color: The pen colour used for this form's handwriting.

    Returns:
        Metadata dict with the stroke count and the kind of each stroke.
    """
    draw = ImageDraw.Draw(form)
    count = random.randint(*STROKE_COUNT_RANGE)
    kinds: list[str] = []

    for _ in range(count):
        kind = random.choice(_STROKE_KINDS)
        kinds.append(kind)
        x = random.uniform(0.03, 0.95) * form.width
        y = random.uniform(0.05, 0.95) * form.height

        if kind == "sweep":
            points = _stroke_points(
                x, y, random.uniform(150, 500),
                random.uniform(-180, 180), random.uniform(-0.05, 0.05),
            )
        elif kind == "tick":
            points = _stroke_points(
                x, y, random.uniform(30, 120),
                random.uniform(-180, 180), random.uniform(-0.12, 0.12),
            )
        else:
            points = _scribble_points(x, y, random.uniform(40, 150))

        draw.line(points, fill=ink_color, width=random.randint(1, 3), joint="curve")

    return {"count": count, "kinds": kinds}


# --- highlighter ---
HIGHLIGHT_PROB = 0.12
HIGHLIGHT_COUNT_RANGE = (1, 3)
# Multiplying a colour into white paper yields the colour itself, so the
# marker's own luminance decides how dark the band gets. Below this floor the
# photocopy profile (which crushes everything under ~80-115 to black) would
# swallow the writing underneath and leave a labelled but unreadable line.
HIGHLIGHT_MIN_LUMA = 150.0
HIGHLIGHT_COLORS: list[tuple[int, int, int]] = [
    (255, 242, 60),    # yellow
    (170, 255, 90),    # green
    (255, 140, 200),   # pink
    (255, 190, 70),    # orange
]
_HIGHLIGHT_SOURCES = ("synthetic", "printed")


def _luminance(color: tuple[int, int, int]) -> float:
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def _ensure_min_luma(
    color: tuple[int, int, int], min_luma: float = HIGHLIGHT_MIN_LUMA
) -> tuple[int, int, int]:
    """Lighten a colour toward white until it clears the readability floor."""
    min_luma = min(min_luma, 255.0)  # white is the ceiling; higher would never terminate
    r, g, b = color
    while _luminance((r, g, b)) < min_luma:
        r = min(255, int(r + (255 - r) * 0.15) + 1)
        g = min(255, int(g + (255 - g) * 0.15) + 1)
        b = min(255, int(b + (255 - b) * 0.15) + 1)
    return (r, g, b)


def _band_mask(size: tuple[int, int], bbox: list[int], tilt_deg: float) -> Image.Image:
    """Build the coverage mask of one marker stroke over a text bbox.

    Real highlighting overshoots the word, runs a little crooked and has a
    wobbly edge where the felt tip wandered.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = bbox

    pad_top = random.randint(2, 6)
    pad_bottom = random.randint(2, 6)
    x_start = x1 - random.randint(0, 8)
    x_end = x2 + random.randint(0, 12)
    phase = random.uniform(0.0, 2 * math.pi)
    wobble = random.uniform(1.0, 3.0)
    wavelength = random.uniform(30.0, 90.0)
    slope = math.tan(math.radians(tilt_deg))

    for x in range(max(0, x_start), min(size[0], x_end)):
        drift = (x - x_start) * slope
        top = y1 - pad_top + drift + wobble * math.sin(x / wavelength + phase)
        bottom = y2 + pad_bottom + drift + wobble * math.sin(x / wavelength + phase + 1.7)
        draw.line([(x, top), (x, bottom)], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(1.0))


def draw_highlights(form: Image.Image, records: list[dict]) -> dict:
    """Run a highlighter over 1-3 existing text lines, in place.

    Multiply blending keeps dark ink dark while the paper takes the colour,
    which is exactly how a translucent marker behaves. Ground truth is
    untouched — the text is still there, just on a coloured background.

    Args:
        form: Filled form image (RGB), modified in place.
        records: Ground-truth records; only "synthetic" and "printed" lines
            are eligible targets.

    Returns:
        Metadata with the count, the colours used and the indices (into
        `records`) of the highlighted lines.
    """
    eligible = [
        i for i, r in enumerate(records) if r["source"] in _HIGHLIGHT_SOURCES
    ]
    if not eligible:
        return {"count": 0, "colors": [], "targets": []}

    count = min(random.randint(*HIGHLIGHT_COUNT_RANGE), len(eligible))
    targets = random.sample(eligible, count)

    arr = np.array(form, dtype=np.float32)
    colors: list[list[int]] = []

    for index in targets:
        color = _ensure_min_luma(random.choice(HIGHLIGHT_COLORS))
        colors.append(list(color))
        mask = _band_mask(form.size, records[index]["bbox"], random.uniform(-2.0, 2.0))
        coverage = (np.array(mask, dtype=np.float32) / 255.0)[:, :, None]
        tinted = arr * (np.array(color, dtype=np.float32) / 255.0)
        arr = arr * (1.0 - coverage) + tinted * coverage

    form.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    return {"count": count, "colors": colors, "targets": targets}


# --- stamps ---
STAMP_PROB = 0.30
STAMP_ROUND_PROB = 0.30
STAMP_ANGLE_MAX = 20.0
# Above this angle an axis-aligned box around slanted text gets too loose to
# be useful ground truth, so those stamps stay unlabelled noise.
STAMP_GT_MAX_ANGLE = 5.0
STAMP_PLACEMENT_TRIES = 8
STAMP_MAX_COVER_FRAC = 0.15
PRINT_FONT_DIR = "resources/fonts_print"
STAMP_COLORS: list[tuple[int, int, int]] = [
    (30, 60, 140),   # blue pad
    (35, 35, 40),    # black pad
    (150, 40, 45),   # red pad
]


@lru_cache(maxsize=16)
def _print_font(font_dir: str, size: int, bold: bool) -> ImageFont.FreeTypeFont:
    """Load the printed font stamps are set in (cached: same file every call)."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(Path(font_dir) / name), size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _stamp_lines(vocab: Vocabulary, font_dir: str) -> list[str]:
    """Compose the 2-4 lines a clinic stamp carries."""
    department, _ = vocab.get_random_text("department")
    doctor, _ = vocab.get_random_text("doctor_name")
    lines = [department, doctor]
    if random.random() < 0.7:
        lines.append(f"nr prawa wyk. zawodu {random.randint(1000000, 9999999)}")
    if random.random() < 0.3:
        lines.append(
            f"NIP {random.randint(100, 999)}-{random.randint(100, 999)}"
            f"-{random.randint(10, 99)}-{random.randint(10, 99)}"
        )
    font_path = str(Path(font_dir) / "DejaVuSans.ttf")
    return [sanitize(line, font_path) for line in lines[:4]]


def _render_rect_stamp(
    lines: list[str], font: ImageFont.FreeTypeFont, color: tuple[int, int, int]
) -> tuple[Image.Image, list[list[int]]]:
    """Draw a bordered rectangular stamp; also return each line's ink bbox."""
    pad_x, pad_y = 14, 10
    line_h = int(font.size * 1.5)
    width = int(max(font.getlength(t) for t in lines) + 2 * pad_x)
    height = int(len(lines) * line_h + 2 * pad_y)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    border = random.choice((1, 2))
    draw.rectangle([0, 0, width - 1, height - 1], outline=color + (255,), width=border)
    if random.random() < 0.35:
        gap = border + 3
        draw.rectangle(
            [gap, gap, width - 1 - gap, height - 1 - gap],
            outline=color + (255,), width=1,
        )

    boxes: list[list[int]] = []
    for i, text in enumerate(lines):
        y = pad_y + i * line_h
        draw.text((pad_x, y), text, font=font, fill=color + (255,))
        left, top, right, bottom = draw.textbbox((pad_x, y), text, font=font)
        boxes.append([int(left), int(top), int(right), int(bottom)])
    return layer, boxes


def _render_round_stamp(
    lines: list[str], font: ImageFont.FreeTypeFont, color: tuple[int, int, int]
) -> Image.Image:
    """Draw a circular stamp: arc text on top, straight text in the middle."""
    diameter = random.randint(180, 300)
    layer = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        [2, 2, diameter - 3, diameter - 3],
        outline=color + (255,), width=random.choice((2, 3)),
    )
    inset = random.randint(10, 18)
    draw.ellipse(
        [inset, inset, diameter - 1 - inset, diameter - 1 - inset],
        outline=color + (255,), width=1,
    )

    arc_text = lines[0][:32]
    centre = diameter / 2
    radius = centre - inset - font.size * 0.7
    span = math.radians(min(200.0, 12.0 * len(arc_text)))
    start = -math.pi / 2 - span / 2
    for i, char in enumerate(arc_text):
        angle = start + span * i / max(1, len(arc_text) - 1)
        glyph = Image.new("RGBA", (font.size * 2, font.size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text(
            (font.size // 2, font.size // 2), char, font=font, fill=color + (255,)
        )
        glyph = glyph.rotate(-math.degrees(angle) - 90, resample=Image.BICUBIC)
        layer.alpha_composite(
            glyph,
            (int(centre + radius * math.cos(angle) - glyph.width / 2),
             int(centre + radius * math.sin(angle) - glyph.height / 2)),
        )

    if len(lines) > 1:
        text = lines[1][:22]
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (centre - (right - left) / 2, centre - (bottom - top) / 2),
            text, font=font, fill=color + (255,),
        )
    return layer


def _apply_ink_unevenness(layer: Image.Image) -> Image.Image:
    """Fade parts of the stamp: a rubber pad never transfers ink evenly."""
    width, height = layer.size
    small = np.random.random((max(2, height // 12), max(2, width // 12)))
    noise = np.array(
        Image.fromarray((small * 255).astype(np.uint8)).resize(
            (width, height), Image.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    factor = np.clip(0.45 + noise * 0.85, 0.0, 1.0)
    alpha = np.array(layer.getchannel("A"), dtype=np.float32) * factor
    layer.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    return layer


def _overlaps_text(
    rect: tuple[int, int, int, int],
    records: list[dict],
    max_cover: float = STAMP_MAX_COVER_FRAC,
) -> bool:
    """True if the rect would bury more than `max_cover` of any text line."""
    rx1, ry1, rx2, ry2 = rect
    for record in records:
        x1, y1, x2, y2 = record["bbox"]
        area = max(1, (x2 - x1) * (y2 - y1))
        overlap_w = max(0, min(rx2, x2) - max(rx1, x1))
        overlap_h = max(0, min(ry2, y2) - max(ry1, y1))
        if overlap_w * overlap_h / area > max_cover:
            return True
    return False


def _find_spot(
    form_size: tuple[int, int], stamp_size: tuple[int, int], records: list[dict]
) -> Optional[tuple[int, int]]:
    """Look for a free patch in the bottom third, where stamps really land."""
    form_w, form_h = form_size
    stamp_w, stamp_h = stamp_size
    if stamp_w >= form_w or stamp_h >= form_h:
        return None

    y_low = int(form_h * 0.60)
    y_high = max(y_low, form_h - stamp_h - 10)
    x_high = max(10, form_w - stamp_w - 10)
    for _ in range(STAMP_PLACEMENT_TRIES):
        y = random.randint(y_low, y_high)
        x = random.randint(10, x_high)
        if not _overlaps_text((x, y, x + stamp_w, y + stamp_h), records):
            return x, y
    return None


def draw_stamp(
    form: Image.Image,
    vocab: Vocabulary,
    records: list[dict],
    print_font_dir: str = PRINT_FONT_DIR,
) -> tuple[Optional[dict], list[dict]]:
    """Press one clinic stamp onto the form, in place.

    Rectangular stamps pressed straight (|angle| <= STAMP_GT_MAX_ANGLE) hand
    back one ground-truth record per text line, because an axis-aligned box
    still hugs the text at that angle. Round stamps and heavily tilted ones
    are unlabelled noise.

    Args:
        form: Filled form image (RGB), modified in place.
        vocab: Vocabulary supplying department and doctor names.
        records: Existing ground-truth records, used to avoid burying text.
        print_font_dir: Directory holding the printed stamp font.

    Returns:
        Tuple of (metadata, new records). Metadata is None when no free spot
        was found, in which case nothing was drawn and no records are returned.
    """
    lines = _stamp_lines(vocab, print_font_dir)
    color = random.choice(STAMP_COLORS)
    is_round = random.random() < STAMP_ROUND_PROB
    font = _print_font(print_font_dir, random.randint(14, 22), random.random() < 0.4)

    if is_round:
        layer = _render_round_stamp(lines, font, color)
        line_boxes: list[list[int]] = []
    else:
        layer, line_boxes = _render_rect_stamp(lines, font, color)

    layer = _apply_ink_unevenness(layer)
    if random.random() < 0.6:
        layer = layer.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.6)))

    angle = random.uniform(-STAMP_ANGLE_MAX, STAMP_ANGLE_MAX)
    src_size = layer.size
    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=True)

    spot = _find_spot(form.size, rotated.size, records)
    if spot is None:
        return None, []
    x, y = spot
    form.paste(rotated, (x, y), rotated)

    in_gt = (not is_round) and abs(angle) <= STAMP_GT_MAX_ANGLE
    new_records: list[dict] = []
    if in_gt:
        for text, box in zip(lines, line_boxes):
            bx1, by1, bx2, by2 = rotate_bbox(
                tuple(box), angle, src_size, rotated.size
            )
            new_records.append({
                "label": "stamp",
                "source": "stamp",
                "text": text,
                "bbox": [bx1 + x, by1 + y, bx2 + x, by2 + y],
            })

    meta = {
        "shape": "round" if is_round else "rect",
        "angle_deg": round(angle, 2),
        "color": list(color),
        "in_gt": in_gt,
        "lines": len(lines),
    }
    return meta, new_records
