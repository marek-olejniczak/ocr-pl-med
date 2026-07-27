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

from PIL import Image, ImageDraw

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
