"""Image-side effects for standalone OCR line renders.

Everything here recreates a condition of the real test crops (phone photos of
student notebooks, boxes cut by a line detector) that the paper-and-scanner
model of the form generator does not produce:

    neighbour_glyphs  real letters of the line above / below poking into the
                      frame: descender tails from above, ascenders and
                      diacritics from below
    grid_paper        squared or ruled notebook paper instead of plain white
    morphology        ink erosion / dilation (thin pens lose diacritics, wet
                      pens blob them together)
    elastic           ElasticTransform / GridDistortion — the local warp of a
                      curved page and a shaky hand
    phone_photo       a phone camera instead of a scanner: shadow gradient,
                      colour cast, soft focus, downscale-upscale, heavy JPEG

Each family has its own probability constant and can be switched off by name
so the ablation can measure it in isolation. Geometry-changing effects are
applied to the whole crop, so the transcription stays valid.
"""

import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import albumentations as A

from fill_form import _make_ink_mask, apply_scan_augmentation
from transforms import jpeg_artifacts

IMAGE_FAMILIES = ("neighbour_glyphs", "grid_paper", "morphology", "elastic", "phone_photo")

NEIGHBOUR_GLYPH_PROB = 0.40
# How deep the neighbour's ink reaches into the frame, as a fraction of the
# font size (a 32 px hand's descender is ~8 px long).
NEIGHBOUR_DEPTH_FRAC = (0.12, 0.45)
# Chance both neighbours show at once, given the effect fires.
NEIGHBOUR_BOTH_PROB = 0.25

GRID_PAPER_PROB = 0.45
# Grid pitch relative to the text height: notebook squares are 5 mm and a
# student's line of writing spans roughly one to two squares.
GRID_PITCH_FRAC = (0.55, 1.10)
RULED_PITCH_FRAC = (1.10, 1.60)
GRID_LINE_COLOURS = [
    (168, 190, 222), (150, 175, 215), (190, 200, 215),   # blue-ish
    (200, 200, 200), (185, 185, 190),                    # grey
    (225, 190, 190),                                     # faint red margin
]

MORPHOLOGY_PROB = 0.30
ELASTIC_PROB = 0.30
PHONE_PHOTO_PROB = 0.40

# Words whose lowest ink is a tail (above the frame) or whose highest ink is
# an ascender / diacritic (below the frame). Drawn from anatomy notes.
DESCENDER_WORDS = [
    "jajnik", "gałęzie", "przyczepy", "zginacz", "języka", "płytka", "jądra",
    "ujścia", "gruczoły", "pęczek", "przyśrodkowy", "biegnący", "jelito",
    "przepona", "gałąź", "powięź", "żyły", "guzowatość", "gąbczasta", "łączy",
    "przegroda", "gęsty", "jajowód", "ząb", "ręka", "pęcherzyk", "grzebień",
]
ASCENDER_WORDS = [
    "kłykieć", "łopatka", "tętnica", "odbytnica", "kość", "błona", "tchawica",
    "kolanowy", "łódkowata", "dół", "bloczek", "biodrowy", "tylna", "łuk",
    "kostka", "trzustka", "półkoliste", "śledziona", "głębokie", "śródstopia",
    "boczny", "dolny", "wątroba", "kątowy", "obłej", "łokieć", "tłoczy",
]


# --------------------------------------------------------------------------
# neighbour glyphs
# --------------------------------------------------------------------------

def _neighbour_text(from_top: bool, pools_word) -> str:
    """Two to four words, biased towards tails (top) or tall strokes (bottom)."""
    biased = DESCENDER_WORDS if from_top else ASCENDER_WORDS
    words = []
    for _ in range(random.randint(2, 4)):
        words.append(random.choice(biased) if random.random() < 0.7 else pools_word())
    return " ".join(words)


def intrude_neighbour_glyphs(
    canvas: Image.Image,
    ink: tuple[int, int, int],
    font_path: str,
    font_size: int,
    config,
    style,
    pools_word,
) -> list[str]:
    """Paste the clipped edge of a neighbouring line into the frame.

    The neighbour is rendered with the same font, size and word style as the
    main line (same hand), then positioned so only its bottom (descenders,
    from the top edge) or its top (ascenders + diacritics, from the bottom
    edge) falls inside the canvas. The frame does the clipping.

    Returns the sides that were drawn, for metadata.
    """
    from char_renderer import render_text_per_char

    sides = ["top", "bottom"]
    if random.random() >= NEIGHBOUR_BOTH_PROB:
        sides = [random.choice(sides)]

    drawn: list[str] = []
    for side in sides:
        from_top = side == "top"
        text = _neighbour_text(from_top, pools_word)
        try:
            text_img, _ = render_text_per_char(
                text, font_path, font_size, padding=2, config=config, word_style=style
            )
        except Exception:
            continue
        mask = _make_ink_mask(text_img)
        box = mask.getbbox()
        if box is None:
            continue
        mask = mask.crop(box)

        depth = max(1, int(font_size * random.uniform(*NEIGHBOUR_DEPTH_FRAC)))
        depth = min(depth, canvas.height - 1)
        x = random.randint(-mask.width // 3, max(0, canvas.width - 2 * mask.width // 3))
        if from_top:
            y = depth - mask.height          # bottom `depth` px are visible
        else:
            y = canvas.height - depth        # top `depth` px are visible

        # Neighbour ink is usually a touch lighter: different pressure, and
        # phone focus falls off away from the line the detector chose.
        faded = tuple(min(255, c + random.randint(0, 25)) for c in ink)
        layer = Image.new("RGB", mask.size, faded)
        canvas.paste(layer, (x, y), mask)
        drawn.append(side)
    return drawn


# --------------------------------------------------------------------------
# paper
# --------------------------------------------------------------------------

def draw_grid_paper(canvas: Image.Image, text_height: int, baseline_y: int) -> str:
    """Draw squared or ruled notebook lines onto the paper. Returns the kind."""
    draw = ImageDraw.Draw(canvas)
    colour = random.choice(GRID_LINE_COLOURS)
    # Photographed grid lines are soft; vary the strength per crop.
    alpha = random.uniform(0.45, 1.0)
    bg = canvas.getpixel((0, 0))
    colour = tuple(int(bg[i] * (1 - alpha) + colour[i] * alpha) for i in range(3))

    ruled_only = random.random() < 0.30
    pitch_frac = random.uniform(*(RULED_PITCH_FRAC if ruled_only else GRID_PITCH_FRAC))
    pitch = max(6, int(text_height * pitch_frac))
    # A student writes on the line, so one horizontal rule sits near the
    # baseline; the rest follow the pitch.
    y0 = baseline_y + random.randint(-2, 2)
    y = y0 % pitch
    while y < canvas.height:
        draw.line([(0, y), (canvas.width, y)], fill=colour, width=1)
        y += pitch
    if not ruled_only:
        x = random.randint(0, pitch - 1)
        while x < canvas.width:
            draw.line([(x, 0), (x, canvas.height)], fill=colour, width=1)
            x += pitch
    return "ruled" if ruled_only else "grid"


# --------------------------------------------------------------------------
# albumentations: ink morphology and elastic geometry
# --------------------------------------------------------------------------

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def apply_morphology(img: Image.Image) -> tuple[Image.Image, str]:
    """Thin or fatten the ink by a pixel or two.

    Erosion is applied on the inverted image so that it eats *ink*, not
    paper: a thin ballpoint loses the dot of an "i" and the tail of an "ę",
    a wet gel pen fills the eye of an "e". Both shapes appear in the test
    material, and the model must keep the diacritic either way.

    Erosion is kept to a 2x2 kernel and blended with the original: a full
    3x3 erosion wipes out one-pixel strokes and leaves nothing to read.
    """
    arr = _pil_to_np(img)
    op = random.choice(["erosion", "dilation"])
    if op == "erosion":
        kernel = np.ones((2, 2), np.uint8)
        eroded = 255 - cv2.erode(255 - arr, kernel, iterations=1)
        mix = random.uniform(0.5, 1.0)
        arr = np.clip(arr * (1 - mix) + eroded * mix, 0, 255).astype(np.uint8)
        return _np_to_pil(arr), f"erosion{mix:.2f}"
    size = random.choice([1, 1, 2])
    kernel = np.ones((size + 1, size + 1), np.uint8)
    arr = 255 - cv2.dilate(255 - arr, kernel, iterations=1)
    return _np_to_pil(arr), f"dilation{size}"


def apply_elastic(img: Image.Image) -> tuple[Image.Image, str]:
    """Local warp of the whole crop: curved page, uneven hand, lens."""
    arr = _pil_to_np(img)
    h, w = arr.shape[:2]
    if random.random() < 0.6:
        alpha = random.uniform(0.6, 1.6) * h
        sigma = random.uniform(0.18, 0.30) * h
        tf = A.ElasticTransform(alpha=alpha, sigma=sigma, border_mode=cv2.BORDER_REPLICATE, p=1.0)
        name = "elastic"
    else:
        tf = A.GridDistortion(num_steps=random.randint(3, 6),
                              distort_limit=random.uniform(0.08, 0.22),
                              border_mode=cv2.BORDER_REPLICATE, p=1.0)
        name = "grid_distortion"
    out = tf(image=arr)["image"]
    return _np_to_pil(out), name


# --------------------------------------------------------------------------
# phone photo
# --------------------------------------------------------------------------

def apply_phone_photo(img: Image.Image) -> tuple[Image.Image, dict]:
    """Simulate a phone snapshot of the page instead of a flatbed scan.

    Shadow gradient across the crop, a warm or cool white balance, soft
    focus from a hand-held camera, resolution loss from the detector's crop
    being upscaled, and the aggressive JPEG a messaging app applies.
    """
    arr = _pil_to_np(img).astype(np.float32)
    h, w = arr.shape[:2]
    meta: dict = {}

    # Brightness gradient: a shadow from one side or one corner.
    strength = random.uniform(0.10, 0.40)
    angle = random.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(angle) * xs / max(1, w) + np.sin(angle) * ys / max(1, h))
    ramp = (ramp - ramp.min()) / max(1e-6, ramp.max() - ramp.min())
    shade = 1.0 - strength * ramp
    arr *= shade[..., None]
    meta["shadow"] = round(strength, 2)

    # White balance: indoor lamps go yellow, daylight shade goes blue.
    cast = random.choice(["warm", "cool", "neutral"])
    if cast == "warm":
        arr *= np.array([1.0, 0.97, random.uniform(0.82, 0.92)], np.float32)
    elif cast == "cool":
        arr *= np.array([random.uniform(0.86, 0.94), 0.97, 1.0], np.float32)
    meta["cast"] = cast

    # Overall exposure wobble.
    arr *= random.uniform(0.85, 1.08)
    arr = np.clip(arr, 0, 255)

    # Sensor noise, stronger in the shadows.
    noise = np.random.normal(0, random.uniform(2.0, 7.0), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = _np_to_pil(arr)

    # Soft focus and the odd motion blur.
    blur = random.uniform(0.4, 1.3)
    out = out.filter(ImageFilter.GaussianBlur(blur))
    meta["blur"] = round(blur, 2)
    if random.random() < 0.25:
        k = random.choice([3, 5])
        out = _np_to_pil(A.MotionBlur(blur_limit=(k, k), p=1.0)(image=_pil_to_np(out))["image"])
        meta["motion_blur"] = k

    # Downscale-upscale: the crop was small on the sensor.
    if random.random() < 0.55:
        scale = random.uniform(0.45, 0.85)
        small = out.resize((max(4, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR)
        out = small.resize((w, h), random.choice([Image.BILINEAR, Image.BICUBIC]))
        meta["downscale"] = round(scale, 2)

    out = jpeg_artifacts(out, 55, 85)
    meta["profile"] = "phone_photo"
    return out, meta


def finish_capture(
    img: Image.Image, enabled: set[str], apply_scan: bool
) -> tuple[Image.Image, Optional[dict]]:
    """Pick the capture device for this crop and apply it.

    Returns (image, metadata of the profile used); metadata is None when
    scan simulation is off.
    """
    if not apply_scan:
        return img, None
    if "phone_photo" in enabled and random.random() < PHONE_PHOTO_PROB:
        return apply_phone_photo(img)
    return apply_scan_augmentation(img)
