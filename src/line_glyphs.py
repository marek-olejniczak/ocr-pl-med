"""Glyph-level families for OCR line renders (generator v3).

The error analysis of the best v2 model (branch `error-analysis`,
notes/vault/Generator_v3_wskazowki.md) found the remaining mistakes are about
letter SHAPES the v2 renderer never draws:

    cursive_joins        letters touch and overlap. v2 always pushes letters
                         apart (positive tracking, widened glyphs), so fused
                         pairs like "ci", "cz", "rz", "ls" (read as u, w, n,
                         b) never appear in training.
    glyph_variants       one letter of the line drawn in another font, picked
                         from the fonts where that letter looks most like the
                         letter it is mistaken for (k like h, a like o, t like
                         l-stroke ...). Fonts alone did not move k->h because
                         the confusing shape was too rare, not absent.
    diacritic_placement  accents, dots and tails that are not where the font
                         puts them: shifted over the next letter, rotated,
                         detached, or left out. The label keeps the correct
                         spelling (the test set is labelled that way too).

Each family draws from its own generator, keyed by the line's text, font and
size, never from the global `random`. The global stream therefore does not
notice which of these families is on: with all three off a seed reproduces
the v2 dataset line for line, and the leave-one-out variants share their
text, fonts and every other effect with the full v3 set.
"""

import math
import random
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_FAMILIES = ("cursive_joins", "glyph_variants", "diacritic_placement")

# --- cursive_joins ---------------------------------------------------------
JOIN_PROB = 0.40
# Tracking and glyph width of a joined hand (v2: 0.06..0.20 and 1.05..1.25).
JOIN_TRACKING_RANGE = (-0.06, 0.03)
JOIN_STRETCH_RANGE = (0.95, 1.10)
# Pairs the model fuses (multi-character events from analyze_ngram.py), pulled
# further together so their strokes overlap.
JOIN_PAIRS = frozenset({
    "ci", "cz", "rz", "oc", "ls", "rn", "co", "sz", "ks", "ko",
    "as", "ru", "rw", "tr", "ki", "śc",
})
JOIN_PAIR_PROB = 0.70
JOIN_PAIR_PULL_FRAC = (0.02, 0.09)  # of the font size

# --- glyph_variants --------------------------------------------------------
VARIANT_PROB = 0.50
# letter -> the letter the model reads it as (top 1:1 confusions of v2).
VARIANT_PAIRS = {"k": "h", "a": "o", "t": "ł", "ł": "t", "g": "o", "r": "n", "o": "a"}
VARIANT_LETTER_PROB = 0.40
# Donor fonts per letter: the ones where it looks most like its partner.
VARIANT_TOP_FONTS = 8

# --- diacritic_placement ---------------------------------------------------
DIACRITIC_PROB = 0.50
DIACRITIC_CHAR_PROB = 0.60
# letter -> the same letter without its mark
DIACRITIC_BASE = {
    "ą": "a", "ć": "c", "ę": "e", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
}
TAILS = frozenset("ąęĄĘ")
DOTTED = frozenset("ij")  # the dot is the topmost separate stroke
DIACRITIC_DROP_PROB = 0.12
DIACRITIC_DX_FRAC = (-0.15, 0.60)  # of the letter width, mostly to the right
DIACRITIC_DY_FRAC = (-0.35, 0.05)  # of the letter height, mostly up
DIACRITIC_ROT_DEG = 30.0
DIACRITIC_SCALE_RANGE = (0.70, 1.40)
TAIL_DX_FRAC = (0.0, 0.30)
TAIL_DY_FRAC = (0.0, 0.25)


def family_rng(family: str, key: str) -> random.Random:
    """The generator one family uses for one line."""
    return random.Random(zlib.crc32(f"{family}|{key}".encode("utf-8")))


# ---------------------------------------------------------------------------
# Glyph rasters
# ---------------------------------------------------------------------------

def _raster(char: str, font: ImageFont.FreeTypeFont, size: int) -> np.ndarray:
    """Alpha of one character on a fixed canvas, baseline at a fixed point.

    Two characters drawn this way line up pixel for pixel, so a letter minus
    its base letter leaves the mark.
    """
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, int(size * 2)), char, font=font, fill=255, anchor="ls")
    return np.asarray(img)


def _ink_box(alpha: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_rgba(alpha: np.ndarray, keep_box: Optional[tuple] = None,
             margin_frac: float = 0.2) -> Optional[Image.Image]:
    """Crop an alpha raster to its ink plus a margin, as black RGBA.

    The renderer centres every glyph image on its pen position. With
    `keep_box` (the untouched letter's ink box) the crop stays centred on
    that box, so a moved or dropped mark leaves the letter where it was.
    """
    box = _ink_box(alpha)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if keep_box is not None:
        cx, cy = (keep_box[0] + keep_box[2]) / 2, (keep_box[1] + keep_box[3]) / 2
        hx = max(cx - x0, x1 - cx, (keep_box[2] - keep_box[0]) / 2)
        hy = max(cy - y0, y1 - cy, (keep_box[3] - keep_box[1]) / 2)
        x0, x1 = int(math.floor(cx - hx)), int(math.ceil(cx + hx))
        y0, y1 = int(math.floor(cy - hy)), int(math.ceil(cy + hy))
    m = max(4, int(max(x1 - x0, y1 - y0) * margin_frac))
    padded = np.pad(alpha, m)
    crop = padded[y0:y1 + 2 * m, x0:x1 + 2 * m]
    out = np.zeros(crop.shape + (4,), np.uint8)
    out[..., 3] = crop
    return Image.fromarray(out)


@lru_cache(maxsize=None)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=None)
def has_glyph(path: str, char: str) -> bool:
    """False when the font would draw its missing-glyph box instead."""
    font = _font(path, 64)
    a = _raster(char, font, 64)
    if not a.any():
        return False
    return not np.array_equal(a, _raster("￿", font, 64))


@lru_cache(maxsize=None)
def x_height(path: str) -> float:
    a = _raster("x", _font(path, 100), 100)
    ys = np.nonzero(a.any(axis=1))[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else 50.0


@lru_cache(maxsize=None)
def letter_height(path: str, char: str) -> float:
    a = _raster(char, _font(path, 100), 100)
    ys = np.nonzero(a.any(axis=1))[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else 50.0


def donor_size(char: str, base: str, donor: str, font_size: int) -> int:
    """Donor font size that makes its letter fit the base hand.

    Matching x-height alone blows up letters with ascenders in fonts with
    tall ones (Playwrite k); the geometric mean with the letter's own height
    keeps both the body and the stem close to the base font.
    """
    by_x = x_height(base) / x_height(donor)
    by_letter = letter_height(base, char) / letter_height(donor, char)
    return max(8, int(round(font_size * math.sqrt(by_x * by_letter))))


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton (Lantuejoul): the pen path without its width."""
    img = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(img)
    while img.any():
        eroded = cv2.erode(img, kernel)
        skel |= img & ~cv2.dilate(eroded, kernel)
        img = eroded
    return skel > 0


@lru_cache(maxsize=None)
def _shape(char: str, path: str) -> np.ndarray:
    """The letter's pen path in a 32x32 box, thickened to a fixed width.

    Comparing raw masks favours bold fonts (thick strokes overlap whatever
    the shape); the skeleton compares where the pen went.
    """
    a = _raster(char, _font(path, 64), 64) > 96
    ys, xs = np.nonzero(a)
    crop = _skeleton(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).astype(np.uint8) * 255
    small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA) > 0
    return cv2.dilate(small.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def glyph_similarity(path: str, a: str, b: str) -> float:
    """IoU of two letters' pen paths in one font, each scaled to the same box."""
    sa, sb = _shape(a, path), _shape(b, path)
    return float((sa & sb).sum() / max(1, (sa | sb).sum()))


@lru_cache(maxsize=None)
def stroke_width(path: str) -> float:
    """Pen width in pixels at x-height 100: ink area over pen path length."""
    a = _raster("o", _font(path, 100), 100) > 96
    return float(a.sum()) / max(1, _skeleton(a).sum()) * 100.0 / x_height(path)


# A donor letter must be written with a similar pen: width ratio within this.
VARIANT_MAX_WIDTH_RATIO = 1.3


@lru_cache(maxsize=None)
def variant_donors(fonts: tuple[str, ...]) -> dict[str, list[tuple[str, float]]]:
    """letter -> [(font, similarity)], fonts whose letter looks most like its partner."""
    donors = {}
    for letter, partner in VARIANT_PAIRS.items():
        scored = [(glyph_similarity(f, letter, partner), f) for f in fonts
                  if has_glyph(f, letter) and has_glyph(f, partner)]
        scored.sort(reverse=True)
        donors[letter] = [(f, s) for s, f in scored[:VARIANT_TOP_FONTS]]
    return donors


def compatible_pen(a: str, b: str) -> bool:
    ra, rb = stroke_width(a), stroke_width(b)
    return max(ra, rb) / max(1e-6, min(ra, rb)) <= VARIANT_MAX_WIDTH_RATIO


# ---------------------------------------------------------------------------
# Per-line plans
# ---------------------------------------------------------------------------

def plan_joins(text: str, font_size: int, rng: random.Random,
               record: Optional[dict] = None) -> Optional[tuple[float, float, Callable]]:
    """Tracking, stretch and a kerning function for one joined line, or None."""
    if rng.random() >= JOIN_PROB:
        return None
    tracking = rng.uniform(*JOIN_TRACKING_RANGE)
    stretch = rng.uniform(*JOIN_STRETCH_RANGE)
    pulls = {}
    for i in range(1, len(text)):
        if text[i - 1:i + 1].lower() in JOIN_PAIRS and rng.random() < JOIN_PAIR_PROB:
            pulls[i] = -rng.uniform(*JOIN_PAIR_PULL_FRAC) * font_size
    if record is not None:
        record.update({"tracking": round(tracking, 4), "x_stretch": round(stretch, 3),
                       "pulled_pairs": [text[i - 1:i + 1] for i in sorted(pulls)]})
    return tracking, stretch, lambda i: pulls.get(i, 0.0)


def plan_variants(text: str, font_path: str, fonts: list[str], rng: random.Random,
                  record: Optional[dict] = None) -> dict[str, str]:
    """letter -> donor font for this line (the same hand writes k the same way)."""
    if rng.random() >= VARIANT_PROB:
        return {}
    donors = variant_donors(tuple(sorted(fonts)))
    chosen = {}
    for letter in VARIANT_PAIRS:
        pool = [f for f, _ in donors.get(letter, [])
                if f != font_path and compatible_pen(f, font_path)]
        if letter in text and pool and rng.random() < VARIANT_LETTER_PROB:
            chosen[letter] = rng.choice(pool)
    if record is not None:
        record.update({k: Path(v).name for k, v in chosen.items()})
    return chosen


def plan_diacritics(text: str, rng: random.Random,
                    record: Optional[list] = None) -> dict[int, dict]:
    """Character index -> how its mark is moved or dropped."""
    if rng.random() >= DIACRITIC_PROB:
        return {}
    ops = {}
    for i, ch in enumerate(text):
        if ch not in DIACRITIC_BASE and ch not in DOTTED:
            continue
        if rng.random() >= DIACRITIC_CHAR_PROB:
            continue
        if rng.random() < DIACRITIC_DROP_PROB:
            op = {"op": "drop"}
        elif ch in TAILS:
            op = {"op": "detach", "dx": rng.uniform(*TAIL_DX_FRAC),
                  "dy": rng.uniform(*TAIL_DY_FRAC)}
        else:
            op = {"op": "shift", "dx": rng.uniform(*DIACRITIC_DX_FRAC),
                  "dy": rng.uniform(*DIACRITIC_DY_FRAC),
                  "rot": rng.uniform(-DIACRITIC_ROT_DEG, DIACRITIC_ROT_DEG),
                  "scale": rng.uniform(*DIACRITIC_SCALE_RANGE)}
        ops[i] = op
        if record is not None:
            record.append({"char": ch, "index": i,
                           **{k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in op.items()}})
    return ops


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _split_mark(char: str, path: str, size: int
                ) -> Optional[tuple[np.ndarray, np.ndarray, tuple]]:
    """(letter without its mark, the mark, the whole letter's ink box), or None."""
    font = _font(path, size)
    full = _raster(char, font, size).astype(np.int16)
    if char in DOTTED:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((full > 96).astype(np.uint8))
        if n < 3:  # background + body + dot
            return None
        dot = 1 + int(np.argmin(stats[1:, cv2.CC_STAT_TOP]))
        mark_px = cv2.dilate((labels == dot).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    else:
        base = _raster(DIACRITIC_BASE[char], font, size)
        grown = cv2.dilate((base > 32).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mark_px = (full > 32) & ~grown
        if mark_px.sum() < 4:
            return None
    mark = np.where(mark_px, full, 0)
    body = np.where(mark_px, 0, full)
    return body.astype(np.uint8), mark.astype(np.uint8), _ink_box(full)


@lru_cache(maxsize=4096)
def _plain_glyph(char: str, path: str, size: int) -> Optional[Image.Image]:
    return _to_rgba(_raster(char, _font(path, size), size))


def _move_mark(body: np.ndarray, mark: np.ndarray, op: dict) -> np.ndarray:
    if op["op"] == "drop":
        return body
    ys, xs = np.nonzero(body > 32)
    w = (xs.max() - xs.min() + 1) if len(xs) else 10
    h = (ys.max() - ys.min() + 1) if len(ys) else 10
    my, mx = np.nonzero(mark > 32)
    cx, cy = float(mx.mean()), float(my.mean())
    m = cv2.getRotationMatrix2D((cx, cy), op.get("rot", 0.0), op.get("scale", 1.0))
    m[0, 2] += op["dx"] * w
    m[1, 2] += op["dy"] * h
    moved = cv2.warpAffine(mark, m, (mark.shape[1], mark.shape[0]), flags=cv2.INTER_LINEAR)
    return np.maximum(body, moved)


def make_glyph_hook(font_path: str, font_size: int, variants: dict[str, str],
                    diacritics: dict[int, dict]) -> Optional[Callable]:
    """A render_text_per_char hook drawing the planned glyphs, or None."""
    if not variants and not diacritics:
        return None

    def hook(index: int, char: str, font: ImageFont.FreeTypeFont):
        op = diacritics.get(index)
        donor = variants.get(char)
        if donor is None and op is None:
            return None
        path, size = font_path, font_size
        if donor is not None:
            path, size = donor, donor_size(char, font_path, donor, font_size)
        advance = int(_font(path, size).getlength(char))
        if op is not None:
            split = _split_mark(char, path, size)
            if split is not None:
                body, mark, keep = split
                img = _to_rgba(_move_mark(body, mark, op), keep_box=keep)
                if img is not None:
                    return img, advance
        if donor is None:
            return None
        img = _plain_glyph(char, path, size)
        return (img, advance) if img is not None else None

    return hook
