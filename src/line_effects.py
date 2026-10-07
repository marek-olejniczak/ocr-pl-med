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

Every effect takes two optional arguments:

    params   force the effect's parameters instead of drawing them. Used by
             the robustness set, where one base line is rendered at fixed
             severities.
    record   a dict (or list, for neighbours) the effect fills with every
             parameter it used, so a dataset can be analysed per setting.

With both left as None an effect draws from `random` in exactly the order it
always has, so a given --seed reproduces earlier datasets line for line.
Albumentations transforms draw from their own generator, which --seed never
reached; they are now seeded from the image they transform, which keeps them
deterministic without taking a single number from the global stream.
"""

import io
import random
import zlib
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import albumentations as A

from fill_form import _make_ink_mask, apply_scan_augmentation

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
GRID_ALPHA = (0.45, 1.0)
GRID_LINE_COLOURS = [
    (168, 190, 222), (150, 175, 215), (190, 200, 215),   # blue-ish
    (200, 200, 200), (185, 185, 190),                    # grey
    (225, 190, 190),                                     # faint red margin
]

MORPHOLOGY_PROB = 0.30
EROSION_MIX = (0.5, 1.0)
ELASTIC_PROB = 0.30
ELASTIC_ALPHA_FRAC = (0.6, 1.6)
ELASTIC_SIGMA_FRAC = (0.18, 0.30)
GRID_DISTORT_STEPS = (3, 6)
GRID_DISTORT_LIMIT = (0.08, 0.22)
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


def _derived_seed(arr: np.ndarray) -> int:
    """A seed that is a function of the image alone.

    Seeding albumentations from `random` or `numpy` would take numbers out of
    the global streams and shift every later draw, so all earlier datasets
    would stop reproducing. The pixels are already fully determined by the
    run's seed, so hashing them gives the same determinism for free.
    """
    return zlib.crc32(np.ascontiguousarray(arr).tobytes()) & 0x7FFFFFFF


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode at a fixed quality (same encoder path as transforms.jpeg_artifacts)."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


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
    params: Optional[dict] = None,
    record: Optional[list] = None,
) -> list[str]:
    """Paste the clipped edge of a neighbouring line into the frame.

    The neighbour is rendered with the same font, size and word style as the
    main line (same hand), then positioned so only its bottom (descenders,
    from the top edge) or its top (ascenders + diacritics, from the bottom
    edge) falls inside the canvas. The frame does the clipping.

    params: {"sides": ["top", "bottom"], "depth_frac": 0.3} forces which
    neighbours appear and how far they reach in.

    Returns the sides that were drawn, for metadata.
    """
    from char_renderer import render_text_per_char

    if params is not None:
        sides = list(params["sides"])
    else:
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

        depth_frac = params["depth_frac"] if params is not None else random.uniform(*NEIGHBOUR_DEPTH_FRAC)
        depth = max(1, int(font_size * depth_frac))
        depth = min(depth, canvas.height - 1)
        x = random.randint(-mask.width // 3, max(0, canvas.width - 2 * mask.width // 3))
        if from_top:
            y = depth - mask.height          # bottom `depth` px are visible
        else:
            y = canvas.height - depth        # top `depth` px are visible

        # Neighbour ink is usually a touch lighter: different pressure, and
        # phone focus falls off away from the line the detector chose.
        fade = [random.randint(0, 25) for _ in ink]
        faded = tuple(min(255, c + f) for c, f in zip(ink, fade))
        layer = Image.new("RGB", mask.size, faded)
        canvas.paste(layer, (x, y), mask)
        drawn.append(side)
        if record is not None:
            record.append({
                "side": side, "depth_frac": round(depth_frac, 3), "depth_px": depth,
                "x": x, "fade": fade, "text": text,
            })
    return drawn


# --------------------------------------------------------------------------
# paper
# --------------------------------------------------------------------------

def draw_grid_paper(
    canvas: Image.Image,
    text_height: int,
    baseline_y: int,
    params: Optional[dict] = None,
    record: Optional[dict] = None,
) -> str:
    """Draw squared or ruled notebook lines onto the paper. Returns the kind.

    params: {"kind": "grid" | "ruled", "alpha": 0.45-1.0, "pitch_frac": ...}
    forces the line style and strength; the colour is still drawn.
    """
    draw = ImageDraw.Draw(canvas)
    base_colour = random.choice(GRID_LINE_COLOURS)
    # Photographed grid lines are soft; vary the strength per crop.
    alpha = params["alpha"] if params is not None else random.uniform(*GRID_ALPHA)
    bg = canvas.getpixel((0, 0))
    colour = tuple(int(bg[i] * (1 - alpha) + base_colour[i] * alpha) for i in range(3))

    if params is not None:
        ruled_only = params["kind"] == "ruled"
        pitch_frac = params["pitch_frac"]
    else:
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
    kind = "ruled" if ruled_only else "grid"
    if record is not None:
        record.update({
            "kind": kind, "alpha": round(alpha, 3), "pitch_frac": round(pitch_frac, 3),
            "pitch_px": pitch, "colour": list(base_colour),
        })
    return kind


# --------------------------------------------------------------------------
# albumentations: ink morphology and elastic geometry
# --------------------------------------------------------------------------

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def apply_morphology(
    img: Image.Image, params: Optional[dict] = None, record: Optional[dict] = None
) -> tuple[Image.Image, str]:
    """Thin or fatten the ink by a pixel or two.

    Erosion is applied on the inverted image so that it eats *ink*, not
    paper: a thin ballpoint loses the dot of an "i" and the tail of an "ę",
    a wet gel pen fills the eye of an "e". Both shapes appear in the test
    material, and the model must keep the diacritic either way.

    Erosion is kept to a 2x2 kernel and blended with the original: a full
    3x3 erosion wipes out one-pixel strokes and leaves nothing to read.

    params: {"op": "erosion", "mix": 0.5-1.0} or {"op": "dilation", "size": 1|2}.
    """
    arr = _pil_to_np(img)
    op = params["op"] if params is not None else random.choice(["erosion", "dilation"])
    if op == "erosion":
        kernel = np.ones((2, 2), np.uint8)
        eroded = 255 - cv2.erode(255 - arr, kernel, iterations=1)
        mix = params["mix"] if params is not None else random.uniform(*EROSION_MIX)
        arr = np.clip(arr * (1 - mix) + eroded * mix, 0, 255).astype(np.uint8)
        if record is not None:
            record.update({"op": "erosion", "mix": round(mix, 3)})
        return _np_to_pil(arr), f"erosion{mix:.2f}"
    size = params["size"] if params is not None else random.choice([1, 1, 2])
    kernel = np.ones((size + 1, size + 1), np.uint8)
    arr = 255 - cv2.dilate(255 - arr, kernel, iterations=1)
    if record is not None:
        record.update({"op": "dilation", "size": size})
    return _np_to_pil(arr), f"dilation{size}"


def apply_elastic(
    img: Image.Image, params: Optional[dict] = None, record: Optional[dict] = None
) -> tuple[Image.Image, str]:
    """Local warp of the whole crop: curved page, uneven hand, lens.

    params: {"kind": "elastic", "alpha_frac": .., "sigma_frac": ..} or
            {"kind": "grid_distortion", "num_steps": .., "distort_limit": ..}.
    alpha and sigma are fractions of the crop height, so the warp scales with
    the text instead of tearing small crops apart.
    """
    arr = _pil_to_np(img)
    h, w = arr.shape[:2]
    if params is not None:
        is_elastic = params["kind"] == "elastic"
    else:
        is_elastic = random.random() < 0.6
    if is_elastic:
        if params is not None:
            alpha_frac, sigma_frac = params["alpha_frac"], params["sigma_frac"]
        else:
            alpha_frac = random.uniform(*ELASTIC_ALPHA_FRAC)
            sigma_frac = random.uniform(*ELASTIC_SIGMA_FRAC)
        tf = A.ElasticTransform(alpha=alpha_frac * h, sigma=sigma_frac * h,
                                border_mode=cv2.BORDER_REPLICATE, p=1.0)
        name = "elastic"
        rec = {"kind": name, "alpha_frac": round(alpha_frac, 3), "sigma_frac": round(sigma_frac, 3)}
    else:
        if params is not None:
            steps, limit = params["num_steps"], params["distort_limit"]
        else:
            steps = random.randint(*GRID_DISTORT_STEPS)
            limit = random.uniform(*GRID_DISTORT_LIMIT)
        tf = A.GridDistortion(num_steps=steps, distort_limit=limit,
                              border_mode=cv2.BORDER_REPLICATE, p=1.0)
        name = "grid_distortion"
        rec = {"kind": name, "num_steps": steps, "distort_limit": round(limit, 3)}
    seed = _derived_seed(arr)
    tf.set_random_seed(seed)
    out = tf(image=arr)["image"]
    if record is not None:
        record.update(rec, seed=seed)
    return _np_to_pil(out), name


# --------------------------------------------------------------------------
# phone photo
# --------------------------------------------------------------------------

# Severity presets for the robustness set: the generator's ranges taken at
# their gentle end, middle and harsh end, every component moving together.
PHONE_PRESETS = {
    1: {"shadow": 0.10, "cast": "neutral", "exposure": 1.00, "noise": 2.0,
        "blur": 0.40, "motion_blur": None, "downscale": None, "jpeg": 85},
    2: {"shadow": 0.25, "cast": "warm", "cast_gain": 0.87, "exposure": 0.95, "noise": 4.5,
        "blur": 0.85, "motion_blur": None, "downscale": 0.65, "jpeg": 70},
    3: {"shadow": 0.40, "cast": "warm", "cast_gain": 0.82, "exposure": 0.85, "noise": 7.0,
        "blur": 1.30, "motion_blur": 5, "downscale": 0.45, "jpeg": 55},
}


def apply_phone_photo(img: Image.Image, params: Optional[dict] = None) -> tuple[Image.Image, dict]:
    """Simulate a phone snapshot of the page instead of a flatbed scan.

    Shadow gradient across the crop, a warm or cool white balance, soft
    focus from a hand-held camera, resolution loss from the detector's crop
    being upscaled, and the aggressive JPEG a messaging app applies.

    params forces every component (see PHONE_PRESETS for the keys; a missing
    "angle" is still drawn). Components can be switched off individually:
    shadow 0, exposure 1, noise 0, blur 0, motion_blur None, downscale None,
    jpeg None. The returned metadata lists everything that was applied.
    """
    P = params
    arr = _pil_to_np(img).astype(np.float32)
    h, w = arr.shape[:2]
    meta: dict = {}

    # Brightness gradient: a shadow from one side or one corner.
    strength = P["shadow"] if P is not None else random.uniform(0.10, 0.40)
    angle = P["angle"] if P is not None and "angle" in P else random.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(angle) * xs / max(1, w) + np.sin(angle) * ys / max(1, h))
    ramp = (ramp - ramp.min()) / max(1e-6, ramp.max() - ramp.min())
    shade = 1.0 - strength * ramp
    arr *= shade[..., None]
    meta["shadow"] = round(strength, 3)
    meta["shadow_angle"] = round(float(angle), 3)

    # White balance: indoor lamps go yellow, daylight shade goes blue.
    cast = P["cast"] if P is not None else random.choice(["warm", "cool", "neutral"])
    gain = None
    if cast == "warm":
        gain = P["cast_gain"] if P is not None else random.uniform(0.82, 0.92)
        arr *= np.array([1.0, 0.97, gain], np.float32)
    elif cast == "cool":
        gain = P["cast_gain"] if P is not None else random.uniform(0.86, 0.94)
        arr *= np.array([gain, 0.97, 1.0], np.float32)
    meta["cast"] = cast
    meta["cast_gain"] = None if gain is None else round(gain, 3)

    # Overall exposure wobble.
    exposure = P["exposure"] if P is not None else random.uniform(0.85, 1.08)
    arr *= exposure
    arr = np.clip(arr, 0, 255)
    meta["exposure"] = round(exposure, 3)

    # Sensor noise, stronger in the shadows.
    noise_sigma = P["noise"] if P is not None else random.uniform(2.0, 7.0)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
        arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = _np_to_pil(arr)
    meta["noise"] = round(noise_sigma, 3)

    # Soft focus and the odd motion blur.
    blur = P["blur"] if P is not None else random.uniform(0.4, 1.3)
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    meta["blur"] = round(blur, 3)
    if P is not None:
        k = P["motion_blur"]
    else:
        k = random.choice([3, 5]) if random.random() < 0.25 else None
    if k:
        mb_in = _pil_to_np(out)
        tf = A.MotionBlur(blur_limit=(k, k), p=1.0)
        tf.set_random_seed(_derived_seed(mb_in))
        out = _np_to_pil(tf(image=mb_in)["image"])
    meta["motion_blur"] = k

    # Downscale-upscale: the crop was small on the sensor.
    if P is not None:
        scale = P["downscale"]
        resample = Image.BILINEAR
    elif random.random() < 0.55:
        scale = random.uniform(0.45, 0.85)
        resample = random.choice([Image.BILINEAR, Image.BICUBIC])
    else:
        scale = None
    if scale:
        small = out.resize((max(4, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR)
        out = small.resize((w, h), resample)
        meta["downscale_resample"] = "bicubic" if resample == Image.BICUBIC else "bilinear"
    meta["downscale"] = None if not scale else round(scale, 3)

    quality = P["jpeg"] if P is not None else random.randint(55, 85)
    if quality:
        out = _jpeg(out, quality)
    meta["jpeg"] = quality
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
