"""Text renderer module for generating word/phrase images.

Renders text strings onto white background images using Pillow,
with configurable fonts and sizes.
"""

import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


def render_text(
    text: str,
    font_path: str,
    font_size: Optional[int] = None,
    padding: int = 15,
) -> tuple[Image.Image, list[int]]:
    """Render text to a PIL Image with tight cropping.

    Args:
        text: The text string to render.
        font_path: Path to a .ttf font file.
        font_size: Font size in pixels. Random 32-72 if None.
        padding: Padding around the text in pixels.

    Returns:
        Tuple of (image, bbox) where bbox is [x_min, y_min, x_max, y_max]
        of the text within the image.
    """
    if font_size is None:
        font_size = random.randint(32, 72)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text bounding box
    # Use a temporary image to get accurate measurements
    tmp_img = Image.new("RGB", (1, 1), "white")
    tmp_draw = ImageDraw.Draw(tmp_img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    # bbox = (left, top, right, bottom) relative to (0,0) anchor

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create image with padding
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Ensure minimum dimensions
    img_width = max(img_width, 1)
    img_height = max(img_height, 1)

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw text offset so the text content starts at (padding, padding)
    x_offset = padding - bbox[0]
    y_offset = padding - bbox[1]
    draw.text((x_offset, y_offset), text, fill="black", font=font)

    # The text bounding box within the image
    text_bbox = [padding, padding, padding + text_width, padding + text_height]

    return img, text_bbox


def find_fonts(font_dir: str) -> list[str]:
    """Recursively find all .ttf font files in the given directory.

    Args:
        font_dir: Path to the fonts directory.

    Returns:
        List of absolute paths to .ttf files.
    """
    font_path = Path(font_dir)
    if not font_path.exists():
        return []
    return [str(p) for p in font_path.rglob("*.ttf")]
