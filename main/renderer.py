"""
Text replacement renderer.

Handles:
1. Inpainting (cleaning) original text using the ML-detected text mask
2. Rendering translated English text that fits within the region
3. Compositing the result onto the original image
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Try to find a suitable font
_FONT_SEARCH_PATHS = [
    # Common Linux font paths (bold preferred for readability)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]


def _find_font() -> str | None:
    """Find an available system font."""
    for path in _FONT_SEARCH_PATHS:
        if Path(path).exists():
            return path
    return None


_DEFAULT_FONT_PATH = _find_font()


def inpaint_text_region(
    image: NDArray,
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None = None,
) -> NDArray:
    """
    Remove original text from a region using inpainting.

    If a text mask is provided (from the ML detector), uses it directly
    for precise inpainting. Otherwise falls back to threshold-based detection.

    Args:
        image: Full page image (BGR).
        x, y, w, h: Bounding box of the text region.
        region_mask: Per-pixel text mask for this region (255=text, 0=background).

    Returns:
        Modified image with the text region cleaned.
    """
    result = image.copy()
    region = result[y : y + h, x : x + w]

    if region_mask is not None and region_mask.shape[:2] == (h, w):
        # Use the ML-provided mask directly — much more accurate
        mask = region_mask.copy()
        # Ensure binary
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = (mask > 0).astype(np.uint8) * 255
    else:
        # Fallback: create mask from threshold
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 127:
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    # Dilate mask to cover text edges and anti-aliasing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Inpaint using Telea algorithm (better for text removal)
    inpainted = cv2.inpaint(region, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    result[y : y + h, x : x + w] = inpainted

    return result


def render_text_on_image(
    image: NDArray,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    font_path: str | None = None,
    padding: int = 6,
) -> NDArray:
    """
    Render translated text within a bounding box on the image.

    Auto-sizes the font to fit, with word wrapping and centering.

    Args:
        image: Full page image (BGR).
        text: Translated text to render.
        x, y, w, h: Bounding box of the target region.
        font_path: Path to a TTF font file.
        padding: Internal padding.

    Returns:
        Modified image with text rendered.
    """
    if not text.strip():
        return image

    result = image.copy()
    font_file = font_path or _DEFAULT_FONT_PATH

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    # Available space
    avail_w = w - 2 * padding
    avail_h = h - 2 * padding
    if avail_w <= 10 or avail_h <= 10:
        return image

    # Determine text color based on background brightness of the cleaned region
    region = result[y : y + h, x : x + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness > 140:
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
    else:
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

    # Auto-size font to fit
    font, lines = _fit_text_to_box(draw, text, avail_w, avail_h, font_file)
    if not lines:
        return image

    # Calculate total text height
    line_metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_metrics.append((lw, lh))

    line_spacing = 3
    total_text_height = sum(lh for _, lh in line_metrics) + (len(lines) - 1) * line_spacing

    # Center vertically
    start_y = y + padding + max(0, (avail_h - total_text_height) // 2)

    # Draw each line centered horizontally
    current_y = start_y
    for i, line in enumerate(lines):
        lw, lh = line_metrics[i]
        line_x = x + padding + max(0, (avail_w - lw) // 2)

        # Draw outline for readability (stroke)
        draw.text(
            (line_x, current_y),
            line,
            font=font,
            fill=text_color,
            stroke_width=2,
            stroke_fill=outline_color,
        )

        current_y += lh + line_spacing

    # Convert back to BGR
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def _fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    font_path: str | None,
    min_size: int = 8,
    max_size: int = 36,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str]]:
    """Find the largest font size where the wrapped text fits in the box."""

    for size in range(max_size, min_size - 1, -2):
        try:
            if font_path:
                font = ImageFont.truetype(font_path, size)
            else:
                font = ImageFont.load_default(size=size)
        except (OSError, TypeError):
            font = ImageFont.load_default()

        lines = _wrap_text(draw, text, font, max_w)
        if not lines:
            continue

        # Calculate total height
        total_h = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            total_h += bbox[3] - bbox[1] + 3  # +3 for line spacing

        if total_h <= max_h:
            return font, lines

    # Fallback to minimum size
    try:
        if font_path:
            font = ImageFont.truetype(font_path, min_size)
        else:
            font = ImageFont.load_default(size=min_size)
    except (OSError, TypeError):
        font = ImageFont.load_default()

    lines = _wrap_text(draw, text, font, max_w)
    return font, lines


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    # If a single word is wider than max_width, split by character
    result: list[str] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            result.append(line)
        else:
            # Character-level wrapping
            current = ""
            for c in line:
                test = current + c
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current = test
                else:
                    if current:
                        result.append(current)
                    current = c
            if current:
                result.append(current)

    return result
