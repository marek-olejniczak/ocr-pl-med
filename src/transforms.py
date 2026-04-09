"""Composable image transforms for simulating handwritten document appearance.

Each transform is a callable that takes a PIL Image and returns a transformed PIL Image.
Transforms are grouped into stages: character-level, line-level, paper, and scan simulation.
"""

import io
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CharTransformConfig:
    """Configuration for per-character transforms."""
    enabled: bool = True
    rotation_max_deg: float = 5.0
    scale_min: float = 0.92
    scale_max: float = 1.08
    stroke_variation: bool = True


@dataclass
class LineTransformConfig:
    """Configuration for word/line-level transforms."""
    enabled: bool = True
    baseline_wander_amplitude: float = 3.0
    spacing_jitter_px: float = 2.0
    slant_max_deg: float = 12.0


@dataclass
class PaperTransformConfig:
    """Configuration for paper/background transforms."""
    enabled: bool = True
    texture: bool = True
    yellowing_intensity: float = 0.1
    coffee_stain_prob: float = 0.05
    fold_mark_prob: float = 0.1


@dataclass
class ScanTransformConfig:
    """Configuration for scan/photo simulation transforms."""
    enabled: bool = True
    noise_sigma: float = 5.0
    blur_radius: float = 0.5
    brightness_variation: float = 0.1
    rotation_max_deg: float = 1.5
    jpeg_quality_min: int = 60
    jpeg_quality_max: int = 95


@dataclass
class AugmentConfig:
    """Master configuration for the augmentation pipeline."""
    char: CharTransformConfig = field(default_factory=CharTransformConfig)
    line: LineTransformConfig = field(default_factory=LineTransformConfig)
    paper: PaperTransformConfig = field(default_factory=PaperTransformConfig)
    scan: ScanTransformConfig = field(default_factory=ScanTransformConfig)


# ---------------------------------------------------------------------------
# Word style — consistent augmentation across all characters in a word
# ---------------------------------------------------------------------------

class WordStyle:
    """Defines a consistent visual style for an entire word/phrase.

    Instead of each character getting fully random transforms, a WordStyle
    picks base values (rotation, scale) that apply to every character,
    with small per-character jitter on top. This mimics real handwriting
    where a person's style is consistent within a word.
    """

    def __init__(
        self,
        base_rotation: float,
        base_scale: float,
        rotation_jitter: float,
        scale_jitter: float,
        do_thicken: bool,
    ) -> None:
        self.base_rotation = base_rotation
        self.base_scale = base_scale
        self.rotation_jitter = rotation_jitter
        self.scale_jitter = scale_jitter
        self.do_thicken = do_thicken

    @classmethod
    def random(cls, cfg: CharTransformConfig) -> "WordStyle":
        """Create a random word style from the config ranges.

        Args:
            cfg: Character transform configuration.

        Returns:
            A WordStyle with randomly chosen base values.
        """
        base_rotation = random.uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg)
        base_scale = random.uniform(cfg.scale_min, cfg.scale_max)
        # Per-character jitter is ~30% of the full range
        rotation_jitter = cfg.rotation_max_deg * 0.3
        scale_jitter = (cfg.scale_max - cfg.scale_min) * 0.15
        do_thicken = cfg.stroke_variation and random.random() < 0.3
        return cls(base_rotation, base_scale, rotation_jitter, scale_jitter, do_thicken)

    def char_rotation(self) -> float:
        """Get a rotation angle for one character (base + small jitter)."""
        return self.base_rotation + random.uniform(-self.rotation_jitter, self.rotation_jitter)

    def char_scale(self) -> float:
        """Get a scale factor for one character (base + small jitter)."""
        return self.base_scale + random.uniform(-self.scale_jitter, self.scale_jitter)


# ---------------------------------------------------------------------------
# Character-level transforms (operate on individual RGBA character images)
# ---------------------------------------------------------------------------

def char_rotate(img: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate a character image by the given angle.

    Args:
        img: RGBA character image.
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated RGBA image.
    """
    return img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)


def char_scale(img: Image.Image, factor: float) -> Image.Image:
    """Scale a character image by the given factor.

    Args:
        img: RGBA character image.
        factor: Scale factor (e.g., 1.05 = 5% larger).

    Returns:
        Scaled RGBA image.
    """
    new_w = max(1, int(img.width * factor))
    new_h = max(1, int(img.height * factor))
    return img.resize((new_w, new_h), Image.BICUBIC)


def char_stroke_variation(img: Image.Image) -> Image.Image:
    """Thicken character strokes using a morphological MaxFilter.

    Applied uniformly to the whole word (the decision to thicken
    is made once per word in WordStyle, not per character).

    Args:
        img: RGBA character image.

    Returns:
        Image with thickened strokes.
    """
    return img.filter(ImageFilter.MaxFilter(3))


# ---------------------------------------------------------------------------
# Line/word-level transforms (operate on composited full image)
# ---------------------------------------------------------------------------

def global_slant(img: Image.Image, max_deg: float) -> Image.Image:
    """Apply a global slant (horizontal shear) to the image.

    Args:
        img: RGB image.
        max_deg: Maximum slant angle in degrees.

    Returns:
        Sheared image on white background.
    """
    angle = random.uniform(-max_deg, max_deg)
    shear = math.tan(math.radians(angle))

    # Affine transform coefficients for horizontal shear
    # (a, b, c, d, e, f) where x' = ax + by + c, y' = dx + ey + f
    w, h = img.size
    shift = abs(shear) * h
    new_w = int(w + shift)

    transform_matrix = (1, shear, -min(0, shear * h), 0, 1, 0)
    result = img.transform(
        (new_w, h),
        Image.AFFINE,
        transform_matrix,
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    return result


# ---------------------------------------------------------------------------
# Paper/background transforms
# ---------------------------------------------------------------------------

def paper_texture(img: Image.Image) -> Image.Image:
    """Add subtle paper texture noise to the background.

    Args:
        img: RGB image.

    Returns:
        Image with paper texture applied.
    """
    arr = np.array(img, dtype=np.float32)
    # Generate smooth noise for paper texture
    noise = np.random.normal(0, 4, arr.shape).astype(np.float32)
    # Smooth it to make it look like paper grain, not pixel noise
    from PIL import ImageFilter
    noise_img = Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=2))
    noise_arr = np.array(noise_img, dtype=np.float32) - 128

    result = np.clip(arr + noise_arr * 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def yellowing(img: Image.Image, intensity: float) -> Image.Image:
    """Apply slight yellowing/aging to the image background.

    Args:
        img: RGB image.
        intensity: Yellowing strength (0.0 to 1.0).

    Returns:
        Yellowed image.
    """
    arr = np.array(img, dtype=np.float32)
    # Sepia-like tint: reduce blue, slightly reduce green
    tint = np.array([1.0, 1.0 - intensity * 0.1, 1.0 - intensity * 0.3], dtype=np.float32)
    arr = arr * tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def coffee_stain(img: Image.Image) -> Image.Image:
    """Add a random semi-transparent brownish elliptical stain.

    Args:
        img: RGB image.

    Returns:
        Image with coffee stain overlay.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Random ellipse position and size
    cx = random.randint(0, img.width)
    cy = random.randint(0, img.height)
    rx = random.randint(20, max(21, img.width // 3))
    ry = random.randint(15, max(16, img.height // 3))

    # Brown-ish color with low alpha
    color = (
        random.randint(100, 140),
        random.randint(60, 90),
        random.randint(20, 50),
        random.randint(15, 40),
    )
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    # Blur the stain for soft edges
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rx // 3))

    img_rgba = img.convert("RGBA")
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert("RGB")


def fold_mark(img: Image.Image) -> Image.Image:
    """Add a faint fold mark (diagonal or vertical line).

    Args:
        img: RGB image.

    Returns:
        Image with fold mark.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Random line orientation
    if random.random() < 0.5:
        # Vertical-ish fold
        x = random.randint(w // 4, 3 * w // 4)
        x_jitter = random.randint(-10, 10)
        draw.line([(x, 0), (x + x_jitter, h)], fill=(200, 195, 190), width=random.randint(1, 2))
    else:
        # Horizontal-ish fold
        y = random.randint(h // 4, 3 * h // 4)
        y_jitter = random.randint(-10, 10)
        draw.line([(0, y), (w, y + y_jitter)], fill=(200, 195, 190), width=random.randint(1, 2))

    return img


# ---------------------------------------------------------------------------
# Scan/photo simulation transforms
# ---------------------------------------------------------------------------

def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise to the image.

    Args:
        img: RGB image.
        sigma: Standard deviation of the noise.

    Returns:
        Noisy image.
    """
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur.

    Args:
        img: RGB image.
        radius: Blur radius.

    Returns:
        Blurred image.
    """
    if radius > 0:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def uneven_brightness(img: Image.Image, variation: float) -> Image.Image:
    """Apply uneven brightness using a smooth radial gradient.

    Args:
        img: RGB image.
        variation: Maximum brightness variation (0.0 to 1.0).

    Returns:
        Image with uneven brightness.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random center for the brightness gradient
    cx = random.uniform(0.2, 0.8)
    cy = random.uniform(0.2, 0.8)

    y_coords = np.linspace(0, 1, h)
    x_coords = np.linspace(0, 1, w)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()

    # Brightness multiplier: brighter at center, darker at edges
    multiplier = 1.0 - variation * dist
    multiplier = multiplier[:, :, np.newaxis]

    result = np.clip(arr * multiplier, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def page_rotation(img: Image.Image, max_deg: float) -> Image.Image:
    """Rotate the entire image by a small random angle.

    Args:
        img: RGB image.
        max_deg: Maximum rotation in degrees.

    Returns:
        Rotated image with white fill.
    """
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def jpeg_artifacts(img: Image.Image, quality_min: int, quality_max: int) -> Image.Image:
    """Simulate JPEG compression artifacts.

    Args:
        img: RGB image.
        quality_min: Minimum JPEG quality.
        quality_max: Maximum JPEG quality.

    Returns:
        Re-compressed image with artifacts.
    """
    quality = random.randint(quality_min, quality_max)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class TransformPipeline:
    """Applies a sequence of post-render transforms based on configuration."""

    def __init__(self, config: AugmentConfig) -> None:
        """Initialize the pipeline with the given config.

        Args:
            config: Augmentation configuration.
        """
        self.config = config

    def apply(self, img: Image.Image) -> Image.Image:
        """Apply all enabled transforms in the correct order.

        Order: slant → paper texture → yellowing → coffee stain → fold mark
        → uneven brightness → noise → blur → page rotation → JPEG artifacts

        Args:
            img: RGB image from the renderer.

        Returns:
            Augmented image.
        """
        cfg = self.config

        # Line-level transforms
        if cfg.line.enabled:
            img = global_slant(img, cfg.line.slant_max_deg)

        # Paper transforms
        if cfg.paper.enabled:
            if cfg.paper.texture:
                img = paper_texture(img)
            if cfg.paper.yellowing_intensity > 0:
                img = yellowing(img, cfg.paper.yellowing_intensity)
            if random.random() < cfg.paper.coffee_stain_prob:
                img = coffee_stain(img)
            if random.random() < cfg.paper.fold_mark_prob:
                img = fold_mark(img)

        # Scan simulation transforms
        if cfg.scan.enabled:
            if cfg.scan.brightness_variation > 0:
                img = uneven_brightness(img, cfg.scan.brightness_variation)
            if cfg.scan.noise_sigma > 0:
                img = gaussian_noise(img, cfg.scan.noise_sigma)
            if cfg.scan.blur_radius > 0:
                img = gaussian_blur(img, cfg.scan.blur_radius)
            if cfg.scan.rotation_max_deg > 0:
                img = page_rotation(img, cfg.scan.rotation_max_deg)
            img = jpeg_artifacts(img, cfg.scan.jpeg_quality_min, cfg.scan.jpeg_quality_max)

        return img
