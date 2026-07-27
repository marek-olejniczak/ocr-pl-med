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

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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
