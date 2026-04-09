"""Per-character text renderer with character-level transforms.

Renders each character as an individual image, applies per-character
distortions (rotation, scaling, stroke variation), then composites
them onto a single canvas with baseline wander and spacing jitter.
"""

import math
import random
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from transforms import (
    AugmentConfig,
    char_rotate,
    char_scale,
    char_stroke_variation,
    WordStyle,
)


def _render_single_char(
    char: str, font: ImageFont.FreeTypeFont
) -> tuple[Image.Image, int]:
    """Render a single character on a transparent RGBA canvas.

    Args:
        char: Single character string.
        font: Loaded PIL font.

    Returns:
        Tuple of (RGBA image with transform margin, font advance width).
        The advance width is used for spacing instead of the image width,
        so characters are placed at natural font-defined distances.
    """
    # Measure character size
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), char, font=font)

    # Font advance width — the distance the cursor should move
    advance_w = int(font.getlength(char))

    # bbox = (left, top, right, bottom) — can have negative offsets
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    # Add margin for transforms (rotation/scaling may expand the character)
    margin = max(4, int(max(char_w, char_h) * 0.2))
    canvas_w = char_w + 2 * margin
    canvas_h = char_h + 2 * margin

    if canvas_w < 1 or canvas_h < 1:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), advance_w

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Position the character so its content starts at (margin, margin)
    x_offset = margin - bbox[0]
    y_offset = margin - bbox[1]
    draw.text((x_offset, y_offset), char, fill=(0, 0, 0, 255), font=font)

    return img, advance_w


def _baseline_offset(index: int, total: int, amplitude: float) -> int:
    """Calculate vertical baseline offset for a character position.

    Uses a combination of a slow sine wave and small random jitter
    to simulate natural handwriting baseline wander.

    Args:
        index: Character index in the string.
        total: Total number of characters.
        amplitude: Maximum vertical offset in pixels.

    Returns:
        Vertical offset in pixels.
    """
    if total <= 1 or amplitude == 0:
        return 0

    # Low-frequency wave
    phase = random.uniform(0, 2 * math.pi)  # phase is per-string, seeded outside ideally
    wave = math.sin(phase + (index / max(total - 1, 1)) * math.pi * 2) * amplitude

    # Small random jitter
    jitter = random.uniform(-amplitude * 0.3, amplitude * 0.3)

    return int(wave + jitter)


def render_text_per_char(
    text: str,
    font_path: str,
    font_size: Optional[int] = None,
    padding: int = 15,
    config: Optional[AugmentConfig] = None,
) -> tuple[Image.Image, list[int]]:
    """Render text character-by-character with per-character transforms.

    Each character is rendered individually, optionally distorted,
    then composited with baseline wander and spacing jitter.

    Args:
        text: Text string to render.
        font_path: Path to a .ttf font file.
        font_size: Font size in pixels. Random 32-72 if None.
        padding: Padding around the text in pixels.
        config: Augmentation configuration. Uses defaults if None.

    Returns:
        Tuple of (image, bbox) where bbox is [x_min, y_min, x_max, y_max].
    """
    if font_size is None:
        font_size = random.randint(32, 72)

    if config is None:
        config = AugmentConfig()

    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    char_cfg = config.char
    line_cfg = config.line

    # Pick a consistent word style — all characters share the same base
    # with small per-character jitter on top
    word_style = WordStyle.random(char_cfg) if char_cfg.enabled else None

    # Render each character individually
    char_entries: list[tuple[Image.Image, int]] = []  # (image, advance_width)
    for char in text:
        if char == " ":
            # For spaces, use font's space advance width
            space_w = max(1, int(font.getlength(" ")))
            space_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            char_entries.append((space_img, space_w))
            continue

        char_img, advance_w = _render_single_char(char, font)

        # Apply character-level transforms with word-consistent style
        if char_cfg.enabled and word_style is not None:
            char_img = char_rotate(char_img, word_style.char_rotation())
            char_img = char_scale(char_img, word_style.char_scale())
            if word_style.do_thicken:
                char_img = char_stroke_variation(char_img)

        char_entries.append((char_img, advance_w))

    if not char_entries:
        img = Image.new("RGB", (2 * padding, 2 * padding), "white")
        return img, [padding, padding, padding, padding]

    # Calculate total canvas size using font advance widths (not image widths)
    total_advance = sum(adv for _, adv in char_entries)
    max_height = max(ci.height for ci, _ in char_entries)

    # Add extra space for spacing jitter and baseline wander
    jitter_budget = int(abs(line_cfg.spacing_jitter_px) * len(char_entries)) if line_cfg.enabled else 0
    wander_budget = int(line_cfg.baseline_wander_amplitude * 3) if line_cfg.enabled else 0

    canvas_w = total_advance + jitter_budget + 2 * padding + max_height  # extra for overhang
    canvas_h = max_height + wander_budget + 2 * padding

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Composite characters using font advance widths for cursor movement
    x_cursor = padding
    text_min_x = canvas_w
    text_min_y = canvas_h
    text_max_x = 0
    text_max_y = 0

    for i, (char_img, advance_w) in enumerate(char_entries):
        # Baseline wander
        if line_cfg.enabled and line_cfg.baseline_wander_amplitude > 0:
            y_offset = _baseline_offset(i, len(char_entries), line_cfg.baseline_wander_amplitude)
        else:
            y_offset = 0

        # Vertical centering + wander
        y_pos = padding + (max_height - char_img.height) // 2 + wander_budget // 2 + y_offset

        # Center the character image on the advance position
        x_pos = x_cursor - (char_img.width - advance_w) // 2

        # Ensure within bounds
        y_pos = max(0, min(y_pos, canvas_h - char_img.height))
        x_pos = max(0, min(x_pos, canvas_w - char_img.width))

        # Paste with alpha compositing
        canvas.paste(char_img, (x_pos, y_pos), char_img)

        # Track bounding box
        content_bbox = char_img.getbbox()
        if content_bbox is not None:
            text_min_x = min(text_min_x, x_pos)
            text_min_y = min(text_min_y, y_pos)
            text_max_x = max(text_max_x, x_pos + char_img.width)
            text_max_y = max(text_max_y, y_pos + char_img.height)

        # Advance cursor by font advance width + optional jitter
        x_cursor += advance_w
        if line_cfg.enabled and line_cfg.spacing_jitter_px > 0:
            jitter = random.uniform(-line_cfg.spacing_jitter_px, line_cfg.spacing_jitter_px)
            x_cursor += int(jitter)

    # Crop canvas to content + padding
    content_right = min(text_max_x + padding, canvas_w)
    content_bottom = min(text_max_y + padding, canvas_h)
    content_left = max(text_min_x - padding, 0)
    content_top = max(text_min_y - padding, 0)

    canvas = canvas.crop((content_left, content_top, content_right, content_bottom))

    # Update bbox relative to cropped image
    final_bbox = [
        text_min_x - content_left,
        text_min_y - content_top,
        text_max_x - content_left,
        text_max_y - content_top,
    ]

    # Convert to RGB
    result = Image.new("RGB", canvas.size, (255, 255, 255))
    result.paste(canvas, (0, 0), canvas)

    return result, final_bbox
