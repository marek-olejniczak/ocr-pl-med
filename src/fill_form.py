"""Form-filling core: renders synthetic handwriting into labeled template
fields. Used by generate_yolo_dataset.py; see
docs/superpowers/specs/2026-07-06-medical-dataset-v2-design.md."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFont

from field_content import generate_field_content
from vocabulary import Vocabulary
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
# Probability that a field's text fades out at the end (pen running dry)
PEN_FADE_PROB = 0.15
# Vertical overflow: max fraction of bbox height the text may shift up/down,
# so ascenders/descenders realistically cross the dotted line
V_OVERFLOW_FRAC = 0.18
# Ink colors: one pen per form — either black or blue (BIC-style)
INK_BLACK = (20, 20, 28)
INK_BLUE = (28, 42, 120)

# Digit grids: number field much wider than tall -> check for printed cells
GRID_ASPECT_RATIO = 6.0

# Multi-line filling of tall description fields
MULTILINE_MAX_LINES = 3
LINE_PITCH_RANGE = (1.2, 1.6)  # line spacing as multiple of handwriting size


def plan_line_slots(bbox_h: int, font_px: int) -> list[int]:
    """Plan y-offsets (from the field top) for writing 1-3 lines in a tall box.

    A field qualifies for multi-line filling when at least two line pitches
    fit into its height. Returns [] for non-qualifying (single-line) fields —
    the caller then uses the regular single-line placement.
    """
    pitch = int(font_px * random.uniform(*LINE_PITCH_RANGE))
    if pitch <= 0:
        return []
    usable = bbox_h - int(font_px * 0.4)  # bottom margin for descenders
    max_lines = usable // pitch
    if max_lines < 2:
        return []
    n = random.randint(1, min(MULTILINE_MAX_LINES, max_lines))
    top = int(bbox_h * 0.06)  # people start writing near the top
    return [top + i * pitch for i in range(n)]


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
            generated/measured at that size). None = auto-fit to the bbox
            height (callers that need width-aware sizing, like
            fill_single_form, always pass an explicit font_size instead).

    Returns:
        RGBA image of rendered text, scaled to fit in bbox.
    """
    if font_size is None:
        # Simple height-based fallback (used by fill_digit_cells, where a
        # single character never needs width-fitting logic) — the final
        # size check below still downscales if this overshoots bbox_w.
        font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, int(bbox_h * 0.7)))

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
    y_offset: Optional[int] = None,
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
        y_offset: Explicit paste offset from the field top (used for
            multi-line slots); overrides the centering/top-anchoring logic.

    Returns:
        Tight bbox of the actually-inked pixels in form coordinates
        (x_min, y_min, x_max, y_max), or None if no ink was drawn.
    """
    mask = _make_ink_mask(text_img)
    paste_x = x_min
    if y_offset is not None:
        paste_y = y_min + y_offset
    elif bbox_h > text_img.height * 2.5:
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


def fill_single_form(
    form_path: Path,
    fields: list[dict],
    vocab: Vocabulary,
    font_path: str,
    config: Optional[AugmentConfig],
    pipeline: Optional[TransformPipeline],
    apply_scan: bool,
    skip_f_fields: bool = False,
    empty_field_range: tuple[float, float] = (0.0, 0.40),
) -> dict:
    """Generate one filled-form variant and return image + ground-truth records.

    Args:
        form_path: Path to the base image (the _blank or _partial variant).
        fields: Field dicts from TemplatePage.fields
            ({"label": "p|t|n|f|mix", "x_min", "y_min", "x_max", "y_max"}).
        vocab: Loaded Vocabulary (source of all written content).
        font_path: Font for this variant (one handwriting per form).
        config: Augmentation config for text rendering (None = no augment).
        pipeline: Kept for API stability (unused inside renders).
        apply_scan: Whether to apply scan-profile simulation at the end.
        skip_f_fields: True when form_path is the _partial base — f fields
            already contain real handwriting; they are not filled, but their
            labeled bbox is recorded as a "handwritten" text line.
        empty_field_range: Per-FORM diligence: one empty-probability is drawn
            from this range per variant and applied to every fill-in field
            (real forms are correlated — one is fully filled, another half-empty).

    Returns:
        Dict with keys:
            image (PIL.Image) — final RGB image
            records (list[dict]) — {"label", "source", "text", "bbox"} where
                source is printed|synthetic|handwritten; text is None unless
                synthetic; bbox is [x_min, y_min, x_max, y_max]
            font (str), text_style (dict|None), ink_color (list[int]),
            scan_augmentation (dict|None), empty_field_prob (float),
            multiline_fields (int) — how many fields got >= 2 lines
    """
    font_name = Path(font_path).name

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

    # One handwriting size per form; small fields shrink it, tall never enlarge
    form_font_px = random.randint(*FORM_FONT_SIZE_RANGE)

    # Per-form diligence: how likely each fill-in field stays empty
    form_empty_prob = random.uniform(*empty_field_range)

    form = Image.open(form_path).convert("RGB")

    records: list[dict] = []
    multiline_fields = 0

    def _measure_fn(font_size: int):
        """Width estimator matching what the renderer will actually draw."""
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        stretch = form_style.x_stretch if form_style is not None else 1.0
        tracking = (
            form_style.tracking_ratio * font_size if form_style is not None else 0.0
        )
        return lambda t: font.getlength(t) * stretch + tracking * len(t)

    def _render_and_paste(text: str, bbox_w: int, bbox_h: int, x_min: int,
                          y_min: int, font_size: int,
                          y_offset: Optional[int] = None):
        """Render one line, optionally pen-fade it, paste, return tight bbox."""
        text_img = render_field_to_bbox(
            text, bbox_w, bbox_h, font_path, config, pipeline, form_style,
            font_size=font_size,
        )
        if random.random() < PEN_FADE_PROB:
            text_img = apply_pen_fade(text_img)
        v_jitter = int(bbox_h * V_OVERFLOW_FRAC) if y_offset is None else 3
        return paste_text_on_form(
            form, text_img, bbox_w, bbox_h, x_min, y_min,
            ink_color=ink_color, v_jitter_px=v_jitter, y_offset=y_offset,
        )

    for field in fields:
        label = field["label"]
        x_min, y_min = field["x_min"], field["y_min"]
        x_max, y_max = field["x_max"], field["y_max"]
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w <= 0 or bbox_h <= 0:
            continue

        # Printed text: pass the labeled bbox straight through to GT
        if label == "p":
            records.append({
                "label": label, "source": "printed",
                "text": None, "bbox": [x_min, y_min, x_max, y_max],
            })
            continue

        # Real handwriting already on the _partial base: record, don't fill
        if label == "f" and skip_f_fields:
            records.append({
                "label": label, "source": "handwritten",
                "text": None, "bbox": [x_min, y_min, x_max, y_max],
            })
            continue

        # Per-form diligence: some fields stay empty
        if random.random() < form_empty_prob:
            continue

        field_font_size = max(MIN_FONT_SIZE, min(form_font_px, int(bbox_h * 0.7)))
        measure = _measure_fn(field_font_size)
        content_kind = "t" if label == "f" else label  # f on blank behaves like t

        # Digit grids (kratki): detected from the printed separators
        if label == "n" and (bbox_w / bbox_h) >= GRID_ASPECT_RATIO:
            grid_cells = detect_grid_cells(form, x_min, y_min, x_max, y_max)
            if grid_cells:
                n_cells = len(grid_cells)
                if n_cells == 11:
                    text, _ = vocab.get_random_text("pesel")
                else:
                    text = "".join(
                        random.choice("0123456789") for _ in range(n_cells)
                    )
                tight = fill_digit_cells(
                    form, text, grid_cells, y_min, bbox_h,
                    font_path, config, pipeline, form_style, ink_color,
                )
                if tight is not None:
                    records.append({
                        "label": label, "source": "synthetic",
                        "text": text, "bbox": list(tight),
                    })
                continue

        # Multi-line filling for tall text-ish boxes
        slots: list[int] = []
        if content_kind in ("t", "mix"):
            slots = plan_line_slots(bbox_h, field_font_size)

        if len(slots) >= 2:
            multiline_fields += 1
            line_h = int(field_font_size * 1.4)
            for slot in slots:
                line_text = generate_field_content(
                    content_kind, vocab, measure, bbox_w
                )
                if not line_text:
                    continue
                tight = _render_and_paste(
                    line_text, bbox_w, line_h, x_min, y_min,
                    field_font_size, y_offset=slot,
                )
                if tight is not None:
                    records.append({
                        "label": label, "source": "synthetic",
                        "text": line_text, "bbox": list(tight),
                    })
            continue

        # Single-line fill
        text = generate_field_content(content_kind, vocab, measure, bbox_w)
        if not text:
            continue
        tight = _render_and_paste(
            text, bbox_w, bbox_h, x_min, y_min, field_font_size
        )
        if tight is not None:
            records.append({
                "label": label, "source": "synthetic",
                "text": text, "bbox": list(tight),
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
        "records": records,
        "font": font_name,
        "text_style": text_style_meta,
        "ink_color": list(ink_color),
        "scan_augmentation": scan_meta,
        "empty_field_prob": round(form_empty_prob, 3),
        "multiline_fields": multiline_fields,
    }
