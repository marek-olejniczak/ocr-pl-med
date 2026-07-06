"""Fill a form image with synthetic handwritten text.

Reads a CSV with bounding-box annotations (from labeling_tool.py),
generates appropriate text for each labeled region using vocabulary.py,
renders it as handwritten text with augmentations, and pastes it into
the form at the specified position.

Special labels:
    pesel_grid — one digit per cell, splits the bbox into 11 equal cells

Usage:
    python fill_form.py --form forms/skierowanie.png \\
                        --annotations dataset/annotations.csv \\
                        --output filled_form.jpg
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFont

from vocabulary import Vocabulary
from renderer import find_fonts
from char_renderer import render_text_per_char
from transforms import (
    AugmentConfig,
    TransformPipeline,
    WordStyle,
    gaussian_noise,
    gaussian_blur,
    uneven_brightness,
    jpeg_artifacts,
    to_grayscale,
    photocopy_contrast,
    salt_pepper_noise,
    toner_streak,
)


# Maps labels in the CSV to vocabulary categories.
#
# New generic workflow uses just two labels — `text` (any letters) and `number`
# (any digits) — because the YOLO line-detection model only learns WHERE text
# is, not which specific field it represents. The old field-specific labels
# (`patient_name`, `pesel`, ...) are kept for backward compatibility with
# previously labeled forms.
LABEL_TO_CATEGORY = {
    # Generic (new workflow — preferred)
    "printed": "printed",  # special: don't render anything, just keep the bbox as ground truth
    "text": "text",
    "number": "number",
    # Backward-compat: specific field types from older labeling sessions
    "patient_name": "patient_name",
    "name": "patient_name",
    "name_and_surname": "patient_name",
    "full_signature": "patient_name",
    "pesel": "pesel",
    "pesel_grid": "pesel",
    "date": "date",
    "full_date": "date",
    "date_of_birth": "date",
    "day_and_month": "day_and_month",
    "last_2_digits_year": "last_2_digits_year",
    "year": "year",
    "rok": "year",
    "icd_code": "icd_code",
    "icd": "icd_code",
    "icd_10": "icd_code",
    "icd10": "icd_code",
    "diagnosis": "icd_description",
    "rozpoznanie": "icd_description",
    "address": "address",
    "adres": "address",
    "city": "city",
    "miasto": "city",
    "phone": "phone",
    "phone_num": "phone",
    "telefon": "phone",
    "doctor_name": "doctor_name",
    "doctor": "doctor_name",
    "lekarz": "doctor_name",
    "hospital_name": "hospital_name",
    "hospital": "hospital_name",
    "szpital": "hospital_name",
    "department": "department",
    "oddzial": "department",
    "age": "age",
    "lat": "age",
    "approval": "approval",
}


# Aspect ratio threshold for auto-detecting PESEL grid (very wide bbox)
PESEL_GRID_ASPECT_RATIO = 6.0

# Fonts excluded from form filling (user-curated list, 2026-05):
#   - bold/medium weights: too thick, look like a marker
#   - overly calligraphic/decorative fonts: unreadable as form handwriting
#   - caps-only fonts: real handwriting is mixed-case (Annie is the exception,
#     re-approved by the user)
EXCLUDED_FONTS = {
    "AmaticSC-Regular.ttf",
    "AmaticSC-Bold.ttf",
    "ShadowsIntoLight-Regular.ttf",
    "Caveat-Bold.ttf",
    "Caveat-Medium.ttf",
    "Caveat-SemiBold.ttf",
    "Caveat-VariableFont_wght.ttf",  # variable font defaults vary by platform
    "TheGirlNextDoor-Regular.ttf",
    "Qwigley-Regular.ttf",
    "OoohBaby-Regular.ttf",
    "MySoul-Regular.ttf",
    "MsMadi-Regular.ttf",
    "Montez-Regular.ttf",
    "Inspiration-Regular.ttf",
}

# Stroke thickening is disabled entirely (user decision 2026-05): MaxFilter
# dilation made thin fonts unreadable at small sizes and marker-like at large
# ones. Set stays defined (empty) so the lookup below keeps working.
FONTS_NEEDING_THICKENING: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill a form image with synthetic text.")
    parser.add_argument("--form", type=str, required=True, help="Path to form image (PNG/JPG)")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--output", type=str, default="filled_form.jpg", help="Output image path")
    parser.add_argument("--font-dir", type=str, default="resources/fonts", help="Font directory")
    parser.add_argument("--resource-dir", type=str, default="resources", help="Resource directory")
    parser.add_argument("--no-augment", action="store_true", help="Disable text-level augmentations")
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable scan/photo simulation on the final filled form (clean output).",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=1,
        help="Generate N variants per form (different text, font, scan conditions). "
             "Use this to build OCR training datasets (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--all-fonts",
        action="store_true",
        help="Cycle through every available font (skips uppercase-only and bold). "
             "Combined with --num-variants gives N variants per font.",
    )
    return parser.parse_args()


def load_annotations(csv_path: Path, form_filename: str) -> list[dict]:
    """Load annotations for a specific form image from the CSV.

    Args:
        csv_path: Path to the annotations CSV.
        form_filename: Original filename of the form image (e.g. "skierowanie.png").

    Returns:
        List of annotation dicts with keys: label, x_min, y_min, x_max, y_max.
    """
    annotations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["filename"] == form_filename:
                annotations.append({
                    "label": row["label"].strip().lower(),
                    "x_min": int(row["x_min"]),
                    "y_min": int(row["y_min"]),
                    "x_max": int(row["x_max"]),
                    "y_max": int(row["y_max"]),
                })
    return annotations


# Minimum readable font size (px). Below this handwriting fonts become an
# unreadable smudge — and a real hand wouldn't squeeze 11 digits into a
# 2-character-wide box anyway.
MIN_FONT_SIZE = 14
# Maximum handwriting size (px). At 200 DPI this is ~5 mm letters — nobody
# writes bigger just because the box is tall (e.g. multi-line address boxes).
MAX_FONT_SIZE = 40
# Each "person" (form variant) writes at a consistent size within this range;
# field height can shrink it but never enlarge it.
FORM_FONT_SIZE_RANGE = (26, 40)
# Number of times to resample random content trying to find one that fits
MAX_CONTENT_TRIES = 10

# --- Dataset-realism knobs (used by fill_single_form) ---
# Probability that a fill-in field is left empty (real forms are never 100% filled;
# the model must learn that an empty dotted line is NOT a text line)
EMPTY_FIELD_PROB = 0.15
# Probability that a field's text fades out at the end (pen running dry)
PEN_FADE_PROB = 0.15
# Vertical overflow: max fraction of bbox height the text may shift up/down,
# so ascenders/descenders realistically cross the dotted line
V_OVERFLOW_FRAC = 0.18
# Ink colors: one pen per form — either black or blue (BIC-style)
INK_BLACK = (20, 20, 28)
INK_BLUE = (28, 42, 120)

# Letter pools for pseudo-Polish filler words (frequency-weighted-ish)
_FILLER_CONSONANTS = "bcdghjklmnprstwzzkmnrsw"
_FILLER_CONS_RARE = "łżźćśńf"
_FILLER_VOWELS = "aaeeiouy"
_FILLER_VOWELS_RARE = "ąęó"


def _random_filler_word() -> str:
    """Generate one pseudo-Polish word (letter shapes matter, meaning doesn't)."""
    n_syllables = random.randint(1, 4)
    word = ""
    for _ in range(n_syllables):
        c = random.choice(_FILLER_CONS_RARE) if random.random() < 0.12 else random.choice(_FILLER_CONSONANTS)
        v = random.choice(_FILLER_VOWELS_RARE) if random.random() < 0.15 else random.choice(_FILLER_VOWELS)
        word += c + v
        if random.random() < 0.25:
            word += random.choice(_FILLER_CONSONANTS)
    if random.random() < 0.35:
        word = word.capitalize()
    return word


def _random_filler_number_group() -> str:
    """Generate one digit group like '472', '08', '1024'."""
    return "".join(random.choice("0123456789") for _ in range(random.randint(1, 4)))


def generate_filler_text(
    font_path: str,
    font_size: int,
    bbox_w: int,
    kind: str,
    x_stretch: float = 1.0,
    tracking_px: float = 0.0,
) -> str:
    """Build random filler content that fills 30-100% of the bbox width.

    For YOLO line detection the content is irrelevant — what matters is that
    the written line spans most of the field (like real handwriting does),
    so the model sees realistically long text lines instead of short words
    floating in mostly-empty fields.

    Args:
        font_path: Font used for rendering (needed to measure text width).
        font_size: Font size the field will be rendered at.
        bbox_w: Field width in pixels.
        kind: "number" for digit groups, anything else for pseudo-words.
        x_stretch: Glyph widening factor the renderer will apply — the
            measured width is scaled by it so the final render still fits.
        tracking_px: Extra per-letter spacing the renderer will add.

    Returns:
        Filler string whose rendered width is <= bbox_w and >= ~30% of it
        (single unit may be shorter if the box fits only one short unit).
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    def rendered_width(t: str) -> float:
        """Estimate final on-form width incl. glyph stretch and tracking."""
        return font.getlength(t) * x_stretch + tracking_px * len(t)

    target = bbox_w * random.uniform(0.30, 1.00)
    # Hard cap so augmentation jitter doesn't push us past the field edge
    hard_cap = bbox_w * 0.97

    make_unit = _random_filler_number_group if kind == "number" else _random_filler_word
    separators = [" ", ".", "-", "/", " "] if kind == "number" else [" "]

    text = make_unit()
    # Even the first unit may overflow a tiny box — trim it down
    while text and rendered_width(text) > hard_cap:
        text = text[:-1]
    if not text:
        return ""

    while True:
        sep = random.choice(separators)
        candidate = text + sep + make_unit()
        w = rendered_width(candidate)
        if w > hard_cap:
            break
        text = candidate
        if w >= target:
            break
    return text


def apply_pen_fade(text_img: Image.Image) -> Image.Image:
    """Simulate a pen running dry toward the end of the written text.

    The last ~25-45% of the image fades progressively toward white, with
    random speckle dropout so strokes visibly break up instead of just
    getting uniformly lighter.

    Args:
        text_img: Rendered text image (RGB, dark ink on white).

    Returns:
        Image with the right-hand portion faded.
    """
    arr = np.array(text_img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    fade_start = random.uniform(0.55, 0.75)
    x0 = int(w * fade_start)
    if x0 >= w - 2:
        return text_img

    ramp = np.linspace(0.0, 1.0, w - x0, dtype=np.float32)
    strength = random.uniform(0.45, 0.8)
    factor = (ramp * strength)[None, :, None]
    region = arr[:, x0:, :]
    arr[:, x0:, :] = region + (255.0 - region) * factor

    # Speckle dropout — random pixels in the faded zone go fully white,
    # denser toward the very end of the stroke
    drop = np.random.random((h, w - x0)) < (ramp * 0.3)
    arr[:, x0:, :][drop] = 255.0

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

# Typical maximum character length per subcategory.  Used to pre-filter which
# subcategories the meta-categories `text` / `number` can draw from when the
# annotated bbox is too narrow to fit longer content at MIN_FONT_SIZE.
_NUMBER_SUBCAT_MAX_LEN = {
    "age": 2,
    "year": 4,
    "last_2_digits_year": 2,
    "day_and_month": 5,
    "icd_code": 6,
    "date": 10,
    "phone": 14,
    "pesel": 11,
}

_TEXT_SUBCAT_MAX_LEN = {
    "city": 25,
    "patient_name": 35,
    "diagnosis": 70,
    "icd_description": 90,
    "address": 70,
    "hospital_name": 70,
    "doctor_name": 35,
    "department": 50,
    "approval": 60,
    "drug": 30,
}


def estimate_char_capacity(font_path: str, bbox_w: int) -> int:
    """Approximate how many average-width characters fit in `bbox_w` at MIN_FONT_SIZE.

    Uses a representative mix of letters to estimate average glyph width;
    keeps a safety margin so per-char rotation, scale jitter and inter-character
    spacing don't push the rendered text past the bbox edge.
    """
    try:
        font = ImageFont.truetype(font_path, MIN_FONT_SIZE)
    except (OSError, IOError):
        return max(1, bbox_w // 8)
    sample = "aeionrtMS0"  # mix of narrow/wide glyphs typical for our vocab
    avg_w = font.getlength(sample) / len(sample)
    return max(1, int((bbox_w * 0.85) / max(1.0, avg_w)))


def fit_font_size(
    text: str,
    font_path: str,
    bbox_w: int,
    bbox_h: int,
    target_height_ratio: float = 0.7,
    max_size: Optional[int] = None,
) -> tuple[int, bool]:
    """Find a font size where the text fits within the given bounding box.

    Starts at target_height_ratio * bbox_h (capped at MAX_FONT_SIZE — a tall
    box doesn't make a human write taller letters), then shrinks if too wide,
    but never below MIN_FONT_SIZE — text at that size is no longer readable
    and doesn't reflect how a human would fill the form (they'd shorten
    the content, not microscope it).

    Args:
        text: The text to fit.
        font_path: Path to the font file.
        bbox_w: Bounding box width.
        bbox_h: Bounding box height.
        target_height_ratio: Fraction of bbox height the text should occupy.
        max_size: Optional extra cap (e.g. the per-form handwriting size).

    Returns:
        Tuple of (font_size, fits) — `fits=False` means the text does NOT
        fit at MIN_FONT_SIZE; the caller should pick shorter content or
        truncate, not blindly render.
    """
    cap = min(MAX_FONT_SIZE, max_size) if max_size else MAX_FONT_SIZE
    target_size = max(MIN_FONT_SIZE, min(int(bbox_h * target_height_ratio), cap))

    font_size = target_size
    while font_size >= MIN_FONT_SIZE:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            return font_size, True
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        if text_w <= bbox_w * 0.95:  # 5% safety margin
            return font_size, True
        font_size -= 2

    return MIN_FONT_SIZE, False


def _eligible_subcategories(meta_category: str, capacity: int) -> list[str]:
    """Return subcategories whose typical max length fits within `capacity` chars.

    For the meta categories `text` and `number`, this lets a narrow box draw
    from short types (year, age, day_and_month) instead of always rolling
    11-digit PESELs or 60-char diagnoses and then skipping when they don't fit.
    """
    if meta_category == "number":
        table = _NUMBER_SUBCAT_MAX_LEN
    elif meta_category == "text":
        table = _TEXT_SUBCAT_MAX_LEN
    else:
        return [meta_category]
    eligible = [k for k, max_len in table.items() if max_len <= capacity]
    if not eligible:
        # Box is tinier than the shortest subcategory — fall back to all so
        # we at least try (and truncate later if truly impossible)
        eligible = sorted(table.keys(), key=lambda k: table[k])[:2]
    return eligible


def pick_fitting_content(
    vocab: Vocabulary,
    category: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str,
    max_size: Optional[int] = None,
) -> str:
    """Sample text from `category` that fits in the bbox at a readable size.

    For meta categories (`text`, `number`), narrows down to subcategories whose
    typical content length fits the available width — a narrow box for `number`
    samples from year/age/day_month, a wide box samples from PESEL/phone/date.
    Falls back to truncation if nothing fits naturally.
    """
    candidates: list[str] = []

    if category in ("text", "number"):
        capacity = estimate_char_capacity(font_path, bbox_w)
        subcats = _eligible_subcategories(category, capacity)
        for _ in range(MAX_CONTENT_TRIES):
            sub = random.choice(subcats)
            text, _ = vocab.get_random_text(sub)
            _, fits = fit_font_size(text, font_path, bbox_w, bbox_h, max_size=max_size)
            if fits:
                return text
            candidates.append(text)
    else:
        for _ in range(MAX_CONTENT_TRIES):
            text, _ = vocab.get_random_text(category)
            _, fits = fit_font_size(text, font_path, bbox_w, bbox_h, max_size=max_size)
            if fits:
                return text
            candidates.append(text)

    # Nothing fit on its own — pick the shortest candidate and truncate
    if not candidates:
        return ""
    text = min(candidates, key=len)
    while text:
        _, fits = fit_font_size(text, font_path, bbox_w, bbox_h, max_size=max_size)
        if fits:
            return text
        text = text[:-1]
    return ""


def render_field_to_bbox(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    word_style: Optional[WordStyle] = None,
    font_size: Optional[int] = None,
) -> Image.Image:
    """Render text sized to fit inside (bbox_w, bbox_h).

    Args:
        text: Text to render.
        bbox_w: Target width in pixels.
        bbox_h: Target height in pixels.
        font_path: Font path.
        config: Augmentation config (None to skip char-level augmentation).
        pipeline: Post-render pipeline (None to skip).
        word_style: Shared word style for consistent look across the form.
        font_size: Explicit size to render at (e.g. when content was already
            generated/measured at that size). None = auto-fit to the bbox.

    Returns:
        RGBA image of rendered text, scaled to fit in bbox.
    """
    if font_size is None:
        font_size, _ = fit_font_size(text, font_path, bbox_w, bbox_h)

    if config is not None:
        img, _ = render_text_per_char(
            text, font_path, font_size, padding=2, config=config, word_style=word_style
        )
    else:
        # Clean fallback rendering
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        img = Image.new("RGB", (text_w + 4, text_h + 4), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((2 - bbox[0], 2 - bbox[1]), text, fill="black", font=font)

    # Final size check — if still too big, downscale
    if img.width > bbox_w or img.height > bbox_h:
        scale = min(bbox_w / img.width, bbox_h / img.height)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


# Scan profiles: how a filled paper form typically enters the system.
# Weights chosen with the user (2026-07): no phone-photo profile, no rotation.
SCAN_PROFILES: list[tuple[str, float]] = [
    ("clean_color", 0.45),
    ("grayscale", 0.35),
    ("photocopy", 0.20),
]


def pick_scan_profile() -> str:
    """Pick a scan profile name according to SCAN_PROFILES weights."""
    names, weights = zip(*SCAN_PROFILES)
    return random.choices(names, weights=weights, k=1)[0]


def apply_scan_augmentation(form: Image.Image) -> tuple[Image.Image, dict]:
    """Apply scan simulation to a fully-filled form using a random profile.

    Profiles model the three ways documents reach the system:
        clean_color — office scanner, mild noise/blur, color kept
        grayscale   — same scanner in mono mode (the common default)
        photocopy   — repeatedly-copied document: crushed contrast,
                      salt-pepper dropout, occasional toner streak

    All effects are photometric only — geometry (and thus every ground-truth
    bbox) is untouched. Page rotation was removed for good: axis-aligned
    boxes degrade on rotated text and the detector copes with skew anyway.

    Returns:
        Tuple of (degraded RGB image, metadata dict incl. "profile").
    """
    profile = pick_scan_profile()
    meta: dict = {"profile": profile}

    if profile in ("clean_color", "grayscale"):
        brightness_var = random.uniform(0.05, 0.15)
        noise_sigma = random.uniform(2.0, 6.0)
        blur_radius = random.uniform(0.3, 0.8)
        jpeg_q = (80, 92)
        form = uneven_brightness(form, brightness_var)
        form = gaussian_noise(form, noise_sigma)
        form = gaussian_blur(form, blur_radius)
        if profile == "grayscale":
            form = to_grayscale(form)
        meta.update({
            "noise_sigma": round(noise_sigma, 2),
            "blur_radius": round(blur_radius, 2),
            "brightness_variation": round(brightness_var, 3),
        })
    else:  # photocopy
        low = random.randint(80, 115)
        high = random.randint(175, 205)
        sp_amount = random.uniform(0.002, 0.008)
        blur_radius = random.uniform(0.2, 0.5)
        jpeg_q = (75, 90)
        form = to_grayscale(form)
        form = photocopy_contrast(form, low=low, high=high)
        form = salt_pepper_noise(form, amount=sp_amount)
        has_streak = random.random() < 0.35
        if has_streak:
            form = toner_streak(form)
        form = gaussian_blur(form, blur_radius)
        meta.update({
            "contrast_low": low,
            "contrast_high": high,
            "salt_pepper_amount": round(sp_amount, 4),
            "blur_radius": round(blur_radius, 2),
            "toner_streak": has_streak,
        })

    form = jpeg_artifacts(form, *jpeg_q)
    meta["jpeg_quality_range"] = list(jpeg_q)
    return form, meta


def _make_ink_mask(text_img: Image.Image, white_threshold: int = 245) -> Image.Image:
    """Convert a black-on-white text image to an alpha mask for compositing.

    Pixels brighter than `white_threshold` become fully transparent.
    Darker pixels get a boosted alpha so anti-aliased edges from rendering
    and downscaling stay visibly opaque (otherwise text looks faded).

    Args:
        text_img: Rendered text image (RGB or L, black ink on white background).
        white_threshold: Grayscale value above which pixels are treated as paper.

    Returns:
        L-mode (grayscale) PIL image to use as the alpha mask in `Image.paste`.
    """
    gray = np.array(text_img.convert("L"), dtype=np.int16)
    # Smooth alpha ramp: pixels darker than ~150 are fully opaque (the actual
    # stroke), 150-240 form an anti-alias gradient, brighter pixels become paper.
    # This avoids artificially thickening strokes by forcing edge greys to solid.
    alpha = np.clip((240 - gray) * (255 / 90), 0, 255).astype(np.uint8)
    alpha[gray >= white_threshold] = 0
    return Image.fromarray(alpha, mode="L")


def paste_text_on_form(
    form: Image.Image,
    text_img: Image.Image,
    bbox_w: int,
    bbox_h: int,
    x_min: int,
    y_min: int,
    ink_color: Optional[tuple[int, int, int]] = None,
    v_jitter_px: int = 0,
) -> Optional[tuple[int, int, int, int]]:
    """Paste rendered text onto the form, centered vertically in the bbox.

    Treats white pixels as transparent so only the ink shows on the form.

    Args:
        form: Form image (modified in place).
        text_img: Rendered text image (RGB, white background).
        bbox_w: Bounding box width.
        bbox_h: Bounding box height.
        x_min: Top-left X of the bbox in the form.
        y_min: Top-left Y of the bbox in the form.
        ink_color: Solid ink color (RGB). None keeps the original (black) pixels.
        v_jitter_px: Max random vertical shift — lets ascenders/descenders
            realistically cross the field's dotted line instead of always
            being perfectly centered.

    Returns:
        Tight bbox of the actually-inked pixels in form coordinates
        (x_min, y_min, x_max, y_max), or None if no ink was drawn.
    """
    mask = _make_ink_mask(text_img)
    paste_x = x_min
    if bbox_h > text_img.height * 2.5:
        # Tall (multi-line-style) box: people start writing near the top,
        # they don't vertically center a single line in a big rectangle
        paste_y = y_min + int(bbox_h * 0.12)
    else:
        paste_y = y_min + (bbox_h - text_img.height) // 2
    if v_jitter_px > 0:
        paste_y += random.randint(-v_jitter_px, v_jitter_px)
    # Keep the paste inside the form canvas
    paste_y = max(0, min(paste_y, form.height - text_img.height))

    if ink_color is not None:
        ink_layer = Image.new("RGB", text_img.size, ink_color)
        form.paste(ink_layer, (paste_x, paste_y), mask)
    else:
        form.paste(text_img.convert("RGB"), (paste_x, paste_y), mask)

    # Tight bbox of actually-inked pixels, offset to form coords
    ink_bbox = mask.getbbox()
    if ink_bbox is None:
        return None
    ix1, iy1, ix2, iy2 = ink_bbox
    return (paste_x + ix1, paste_y + iy1, paste_x + ix2, paste_y + iy2)


def detect_grid_cells(
    form: Image.Image,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    min_cell_w: int = 8,
) -> list[tuple[int, int]]:
    """Detect printed digit-grid cells (kratki) inside a bbox on the form.

    Forms draw digit grids as boxes separated by vertical lines (PESEL has 11
    cells, IBAN has 26, etc.). Instead of assuming a cell count, this scans
    the column profile of dark pixels in the region: columns where ink spans
    most of the height are separator lines, and the gaps between them are
    cells.

    Args:
        form: The form image (region must still be unfilled).
        x_min, y_min, x_max, y_max: Field bbox in form coordinates.
        min_cell_w: Minimum width for a gap to count as a cell.

    Returns:
        List of (cell_x_start, cell_x_end) in form coordinates, ordered
        left to right. Empty list if the region doesn't look like a grid
        (e.g. it's a dotted fill-in line instead).
    """
    crop = form.crop((x_min, y_min, x_max, y_max)).convert("L")
    arr = np.array(crop)
    h, w = arr.shape
    if h < 6 or w < 20:
        return []

    dark = arr < 128
    col_frac = dark.mean(axis=0)

    # Vertical separator lines span most of the grid height. The user's bbox
    # may include some white margin, so the threshold adapts to the tallest
    # line found in the region.
    peak = float(col_frac.max())
    if peak < 0.4:
        return []  # no vertical lines anywhere — not a grid
    threshold = max(0.35, peak * 0.6)

    # Collapse runs of consecutive separator columns into single positions
    separators: list[float] = []
    i = 0
    while i < w:
        if col_frac[i] > threshold:
            j = i
            while j < w and col_frac[j] > threshold:
                j += 1
            separators.append((i + j - 1) / 2.0)
            i = j
        else:
            i += 1

    if len(separators) < 3:
        return []  # need at least 3 lines to form 2 cells

    cells: list[tuple[int, int]] = []
    for a, b in zip(separators, separators[1:]):
        if (b - a) >= min_cell_w:
            cells.append((int(x_min + a + 1), int(x_min + b)))

    return cells if len(cells) >= 2 else []


def fill_digit_cells(
    form: Image.Image,
    digits: str,
    cells: list[tuple[int, int]],
    y_min: int,
    bbox_h: int,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    word_style: Optional[WordStyle] = None,
    ink_color: Optional[tuple[int, int, int]] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Write one digit into each given grid cell.

    Args:
        form: Form image (modified in place).
        digits: Digit string; digits[i] goes into cells[i].
        cells: List of (cell_x_start, cell_x_end) in form coordinates.
        y_min: Top of the field bbox.
        bbox_h: Height of the field bbox.
        font_path: Font path.
        config: Augmentation config.
        pipeline: Post-render pipeline.
        word_style: Shared word style for consistent look.
        ink_color: Solid ink color (RGB). None keeps the original (black) pixels.

    Returns:
        Union bbox covering all rendered digits (x_min, y_min, x_max, y_max)
        in form coordinates, or None if no digit produced any ink.
    """
    union: Optional[list[int]] = None  # [x1, y1, x2, y2]

    for digit, (cell_x, cell_x_end) in zip(digits, cells):
        actual_cell_w = cell_x_end - cell_x
        if actual_cell_w < 2:
            continue

        digit_img = render_field_to_bbox(
            digit, actual_cell_w, bbox_h, font_path, config, pipeline, word_style
        )
        # Center horizontally inside the cell
        paste_x = cell_x + (actual_cell_w - digit_img.width) // 2
        paste_y = y_min + (bbox_h - digit_img.height) // 2

        mask = _make_ink_mask(digit_img)
        if ink_color is not None:
            ink_layer = Image.new("RGB", digit_img.size, ink_color)
            form.paste(ink_layer, (paste_x, paste_y), mask)
        else:
            form.paste(digit_img.convert("RGB"), (paste_x, paste_y), mask)

        # Track tight bbox of this digit's ink, accumulate as union
        ink_bbox = mask.getbbox()
        if ink_bbox is None:
            continue
        ix1, iy1, ix2, iy2 = ink_bbox
        gx1, gy1 = paste_x + ix1, paste_y + iy1
        gx2, gy2 = paste_x + ix2, paste_y + iy2
        if union is None:
            union = [gx1, gy1, gx2, gy2]
        else:
            union[0] = min(union[0], gx1)
            union[1] = min(union[1], gy1)
            union[2] = max(union[2], gx2)
            union[3] = max(union[3], gy2)

    if union is None:
        return None
    return (union[0], union[1], union[2], union[3])


def fill_pesel_grid(
    form: Image.Image,
    pesel: str,
    x_min: int,
    y_min: int,
    bbox_w: int,
    bbox_h: int,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    word_style: Optional[WordStyle] = None,
    ink_color: Optional[tuple[int, int, int]] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Fill a digit grid assuming 11 equal cells (legacy PESEL fallback).

    Used only when a field is explicitly labeled `pesel_grid` but no printed
    cell separators could be detected in the region. Prefer detect_grid_cells
    + fill_digit_cells, which adapt to the actual number of cells.
    """
    cell_w = bbox_w / 11.0
    cells = [
        (int(x_min + i * cell_w), int(x_min + (i + 1) * cell_w))
        for i in range(11)
    ]
    return fill_digit_cells(
        form, pesel, cells, y_min, bbox_h,
        font_path, config, pipeline, word_style, ink_color,
    )


# Subcategories whose content is digit-based — used to decide filler kind
# for legacy field-specific labels when running in filler mode
_NUMBERISH_CATEGORIES = {
    "number", "pesel", "date", "phone", "icd_code",
    "year", "age", "day_and_month", "last_2_digits_year",
}


def fill_single_form(
    form_path: Path,
    annotations: list[dict],
    vocab: Vocabulary,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    apply_scan: bool,
    filler_mode: bool = False,
    empty_field_prob: float = 0.0,
) -> dict:
    """Generate one filled-form variant and return image + ground-truth bboxes.

    This is the core pipeline shared between the per-form CLI in this script
    and the bulk YOLO-dataset generator. It fills every annotation, records
    the tight bbox of the actual ink for each field, then optionally applies
    scan/photo simulation. Scan simulation is photometric only, so bboxes
    stay valid in the final image without remapping.

    Args:
        form_path: Path to the source form image (PNG/JPG).
        annotations: List of annotation dicts (from labeling_tool CSV).
        vocab: Loaded Vocabulary.
        font_path: Font to use for this variant.
        config: Augmentation config for text rendering (None = no augment).
        pipeline: TransformPipeline (currently unused inside renders, kept
            for API stability; scan augmentation is applied below).
        apply_scan: Whether to apply scan/photo simulation at the end.
        filler_mode: If True, fields are filled with random pseudo-words /
            digit groups sized to span 30-100% of the field width (content
            is irrelevant for line detection; line LENGTH realism matters).
            If False, content comes from the medical vocabulary (for OCR
            datasets where the text itself is the label).
        empty_field_prob: Probability of leaving a fill-in field empty.
            Real forms are never 100% filled; the detector must learn that
            an empty dotted line is not a text line.

    Returns:
        Dict with keys:
            image (PIL.Image) — final RGB image
            tight_bboxes (list[dict]) — per-field {label, text, bbox} where
                bbox is in the final image's coordinate space
            font (str) — basename of the font used
            text_style (dict|None) — WordStyle parameters used
            ink_color (list[int]) — RGB pen color used for this form
            scan_augmentation (dict|None) — scan params used
    """
    font_name = Path(font_path).name

    # Build a fresh word style per variant for natural variation
    if config is not None and config.char.enabled:
        form_style = WordStyle.random(config.char)
        form_style.do_thicken = font_name in FONTS_NEEDING_THICKENING
        form_style.thicken_kernel = 3
    else:
        form_style = None

    # One pen per form: black or blue, with slight per-form shade variation
    base_ink = random.choice([INK_BLACK, INK_BLUE])
    ink_color = tuple(
        max(0, min(255, c + random.randint(-8, 8))) for c in base_ink
    )

    # One handwriting size per form: a person writes letters of consistent
    # height regardless of how big the printed box is. Small fields can
    # shrink it; tall fields must NOT enlarge it.
    form_font_px = random.randint(*FORM_FONT_SIZE_RANGE)

    # Open a fresh copy of the form
    form = Image.open(form_path).convert("RGB")

    # Two record streams:
    #   - tight_records: tight bboxes of actually-rendered text (for text/number).
    #     Useful as debug/inspection info but not the YOLO ground truth.
    #   - printed_records: bboxes of printed/static text lines (for `printed` label).
    #     These are the YOLO ground truth — they cover entire visible lines on
    #     the form (printed labels plus any fill-in spots inside them), and stay
    #     constant across variants of the same template.
    tight_records: list[dict] = []
    printed_records: list[dict] = []

    for ann in annotations:
        label = ann["label"]
        x_min, y_min = ann["x_min"], ann["y_min"]
        x_max, y_max = ann["x_max"], ann["y_max"]
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        category = LABEL_TO_CATEGORY.get(label)
        if category is None:
            continue

        # `printed` is a pass-through: don't render anything, just keep the
        # bbox so it ends up in the YOLO ground truth.
        if category == "printed":
            printed_records.append({
                "label": label,
                "text": None,
                "bbox": [x_min, y_min, x_max, y_max],
            })
            continue

        # Real forms are never fully filled — randomly leave some fields empty
        # so the detector learns that a blank dotted line is not a text line
        if empty_field_prob > 0 and random.random() < empty_field_prob:
            continue

        # Digit grids (kratki): when a number-ish field is much wider than
        # tall, check whether the form actually prints cell separators there.
        # The cell count is DETECTED from the image (PESEL=11, IBAN=26, date
        # boxes=8, ...) instead of assumed — one digit goes into each real cell.
        grid_cells: list[tuple[int, int]] = []
        if (
            label in ("pesel", "pesel_grid", "number")
            and bbox_h > 0
            and (bbox_w / bbox_h) >= PESEL_GRID_ASPECT_RATIO
        ):
            grid_cells = detect_grid_cells(form, x_min, y_min, x_max, y_max)
            if not grid_cells and label == "pesel_grid":
                # Explicit grid label but no detectable separators —
                # legacy fallback: assume 11 equal PESEL cells
                cell_w = bbox_w / 11.0
                grid_cells = [
                    (int(x_min + i * cell_w), int(x_min + (i + 1) * cell_w))
                    for i in range(11)
                ]

        if grid_cells:
            n_cells = len(grid_cells)
            if not filler_mode and n_cells == 11:
                # 11 cells = PESEL — use a checksum-valid one
                text, _ = vocab.get_random_text("pesel")
            else:
                text = "".join(random.choice("0123456789") for _ in range(n_cells))
            tight_bbox = fill_digit_cells(
                form, text, grid_cells, y_min, bbox_h,
                font_path, config, pipeline, form_style, ink_color,
            )
        else:
            # Field font size: the person's handwriting size, shrunk only
            # if the field is too short to fit it
            field_font_size = max(
                MIN_FONT_SIZE, min(form_font_px, int(bbox_h * 0.7))
            )

            if filler_mode:
                # Random pseudo-words / digit groups spanning 30-100% of the
                # field width — realistic line length, irrelevant content
                kind = "number" if category in _NUMBERISH_CATEGORIES else "text"
                stretch = form_style.x_stretch if form_style is not None else 1.0
                tracking = (
                    form_style.tracking_ratio * field_font_size
                    if form_style is not None else 0.0
                )
                text = generate_filler_text(
                    font_path, field_font_size, bbox_w, kind,
                    x_stretch=stretch, tracking_px=tracking,
                )
            else:
                # Pick vocabulary content sized to fit at a readable font
                text = pick_fitting_content(
                    vocab, category, bbox_w, bbox_h, font_path, max_size=form_font_px
                )
            if not text:
                # Box is too small even for a single character — skip
                continue
            if not filler_mode:
                # Vocab content may still need width-shrinking below the form size
                field_font_size, _ = fit_font_size(
                    text, font_path, bbox_w, bbox_h, max_size=form_font_px
                )
            text_img = render_field_to_bbox(
                text, bbox_w, bbox_h, font_path, config, pipeline, form_style,
                font_size=field_font_size,
            )
            # Pen running dry toward the end of the line
            if random.random() < PEN_FADE_PROB:
                text_img = apply_pen_fade(text_img)
            v_jitter = int(bbox_h * V_OVERFLOW_FRAC)
            tight_bbox = paste_text_on_form(
                form, text_img, bbox_w, bbox_h, x_min, y_min,
                ink_color=ink_color, v_jitter_px=v_jitter,
            )

        if tight_bbox is None:
            continue

        tight_records.append({
            "label": label,
            "text": text,
            "bbox": list(tight_bbox),  # tight ink bbox in current form coords
            "field_bbox": [x_min, y_min, x_max, y_max],  # original field bbox for reference
        })

    # Scan simulation is photometric only — bboxes stay valid as-is.
    scan_meta = None
    if apply_scan:
        form, scan_meta = apply_scan_augmentation(form)

    text_style_meta = None
    if form_style is not None:
        text_style_meta = {
            "base_rotation_deg": round(form_style.base_rotation, 2),
            "base_scale": round(form_style.base_scale, 3),
            "handwriting_size_px": form_font_px,
            "x_stretch": round(form_style.x_stretch, 3),
            "tracking_ratio": round(form_style.tracking_ratio, 3),
        }

    return {
        "image": form,
        "tight_bboxes": tight_records,
        "printed_bboxes": printed_records,
        "font": font_name,
        "text_style": text_style_meta,
        "ink_color": list(ink_color),
        "scan_augmentation": scan_meta,
    }


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    form_path = Path(args.form)
    if not form_path.exists():
        print(f"ERROR: Form image not found: {form_path}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.annotations)
    if not csv_path.exists():
        print(f"ERROR: Annotations CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load resources
    print("Loading vocabulary...")
    vocab = Vocabulary(args.resource_dir)
    all_fonts = find_fonts(args.font_dir)
    # Exclude uppercase-only fonts and bold weights
    fonts = [f for f in all_fonts if Path(f).name not in EXCLUDED_FONTS]
    if not fonts:
        print(f"ERROR: No usable fonts found in {args.font_dir}", file=sys.stderr)
        sys.exit(1)
    skipped = len(all_fonts) - len(fonts)
    print(f"  Using {len(fonts)} fonts ({skipped} excluded: caps-only and bold weights)")

    # Set up augmentation — toned down for form filling, where text
    # should look like neat real handwriting, not exaggerated stylization.
    if not args.no_augment:
        config = AugmentConfig()

        # Subtler character variation
        config.char.rotation_max_deg = 2.5      # was 5.0
        config.char.scale_min = 0.95            # was 0.92
        config.char.scale_max = 1.05            # was 1.08

        # Subtler line-level effects
        config.line.baseline_wander_amplitude = 1.5  # was 3.0
        config.line.spacing_jitter_px = 0.8          # was 2.0
        config.line.slant_max_deg = 4.0              # was 12.0
        config.line.baseline_drift_max_px = 4.0      # line gradually climbs/falls

        # We paste onto an existing form — paper texture and scan effects
        # would double up with the original form's appearance
        config.paper.enabled = False
        config.scan.enabled = False
        pipeline = TransformPipeline(config)
    else:
        config = None
        pipeline = None

    # Load annotations
    annotations = load_annotations(csv_path, form_path.name)
    if not annotations:
        print(f"ERROR: No annotations found for '{form_path.name}' in {csv_path}", file=sys.stderr)
        print("  (annotations are matched by filename)", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(annotations)} annotations")

    # Decide which fonts to render with
    if args.all_fonts:
        font_loop = fonts
        print(f"  --all-fonts: cycling through {len(font_loop)} fonts")
    else:
        font_loop = [None]  # None means random font per variant

    n_variants = max(1, args.num_variants)
    apply_scan = not args.no_scan and not args.no_augment
    print(f"  Variants per font: {n_variants}")
    print(f"  Scan augmentation: {'ON' if apply_scan else 'OFF'}")

    output_arg = Path(args.output)
    output_arg.parent.mkdir(parents=True, exist_ok=True)
    total_count = 0

    for font_choice in font_loop:
        for v in range(1, n_variants + 1):
            # Resolve font for this variant
            font_path = font_choice if font_choice is not None else random.choice(fonts)
            font_name = Path(font_path).name
            font_stem = Path(font_path).stem

            # Build output path encoding variant number and/or font
            stem = output_arg.stem
            suffix = output_arg.suffix
            parts = [stem]
            if n_variants > 1:
                parts.append(f"{v:03d}")
            if args.all_fonts:
                parts.append(font_stem)
            out_image = output_arg.with_name("_".join(parts) + suffix)

            result = fill_single_form(
                form_path=form_path,
                annotations=annotations,
                vocab=vocab,
                font_path=font_path,
                config=config,
                pipeline=pipeline,
                apply_scan=apply_scan,
            )

            # Save filled form as JPG
            result["image"].save(out_image, quality=92)

            # Save metadata JSON next to the image
            meta_path = out_image.with_suffix(".json")
            metadata = {
                "form_image": form_path.name,
                "output_image": out_image.name,
                "image_size": [result["image"].width, result["image"].height],
                "font": result["font"],
                "fields": result["tight_bboxes"],
                "printed_lines": result["printed_bboxes"],
            }
            if result["text_style"] is not None:
                metadata["text_style"] = result["text_style"]
            if result["scan_augmentation"] is not None:
                metadata["scan_augmentation"] = result["scan_augmentation"]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            total_count += 1
            print(f"  [{total_count}] {out_image.name}  font={font_name}")

    print(f"\nDone. {total_count} form variant(s) generated in {output_arg.parent}/")


if __name__ == "__main__":
    main()
