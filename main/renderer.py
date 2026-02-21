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

# Try to find suitable dialogue and SFX fonts.
_DIALOGUE_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]

_SFX_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
    "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-BoldItalic.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

_NARROW_DIALOGUE_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
]

_CJK_DIALOGUE_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def _find_font(search_paths: list[str]) -> str | None:
    """Find an available font from an ordered path list."""
    for path in search_paths:
        if Path(path).exists():
            return path
    return None


_DIALOGUE_FONT_PATH = _find_font(_DIALOGUE_FONT_SEARCH_PATHS)
_SFX_FONT_PATH = _find_font(_SFX_FONT_SEARCH_PATHS) or _DIALOGUE_FONT_PATH
_NARROW_DIALOGUE_FONT_PATH = (
    _find_font(_NARROW_DIALOGUE_FONT_SEARCH_PATHS) or _DIALOGUE_FONT_PATH
)
_CJK_DIALOGUE_FONT_PATH = (
    _find_font(_CJK_DIALOGUE_FONT_SEARCH_PATHS) or _DIALOGUE_FONT_PATH
)
_DEFAULT_FONT_PATH = _DIALOGUE_FONT_PATH or _SFX_FONT_PATH


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
    region_mask: NDArray | None = None,
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
    text_style = _infer_text_style(text, w, h)

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    region_slice = result[y : y + h, x : x + w]

    # Place text near the original glyph cluster inside the region when possible.
    # For longer dialogue, optionally expand toward the enclosing bubble shape.
    box_x, box_y, box_w, box_h, bubble_used = _resolve_text_box(
        x=x,
        y=y,
        w=w,
        h=h,
        region_mask=region_mask,
        region_image=region_slice,
        translated_text=text,
        text_style=text_style,
    )
    non_bubble_dialogue = text_style == "dialogue" and not bubble_used
    if non_bubble_dialogue:
        # Non-bubble regions are often SFX/noise; keep wording compact.
        max_words = 5 if (box_w * box_h) < 12000 else 7
        text = _compress_dialogue_for_tiny_box(text, max_words=max_words)
    if _is_punctuation_only(text):
        return image

    is_tall_dialogue = text_style == "dialogue" and box_h > box_w * 1.35
    if font_path:
        font_file = font_path
    elif _contains_cjk(text):
        font_file = _CJK_DIALOGUE_FONT_PATH
    elif text_style == "sfx":
        font_file = _SFX_FONT_PATH
    elif is_tall_dialogue:
        font_file = _NARROW_DIALOGUE_FONT_PATH
    else:
        font_file = _DIALOGUE_FONT_PATH
    if font_file is None:
        font_file = _DEFAULT_FONT_PATH

    text_padding = max(2, padding - 2) if text_style == "dialogue" else padding
    avail_w = box_w - 2 * text_padding
    avail_h = box_h - 2 * text_padding
    if avail_w <= 10 or avail_h <= 10:
        return image

    # Determine text color based on background brightness of the cleaned region
    region = result[box_y : box_y + box_h, box_x : box_x + box_w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness > 140:
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
    else:
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

    # Auto-size font to fit
    font_cap = 34 if non_bubble_dialogue else None
    font, lines, line_spacing = _fit_text_to_box(
        draw=draw,
        text=text,
        max_w=avail_w,
        max_h=avail_h,
        font_path=font_file,
        style=text_style,
        max_size_cap=font_cap,
    )

    # If dialogue still overflows into too many tiny lines, aggressively shorten.
    rendered_area = box_w * box_h
    if (
        text_style == "dialogue"
        and rendered_area < 9000
        and lines
        and (len(lines) >= 5 or getattr(font, "size", 12) <= 11)
    ):
        short_text = _compress_dialogue_for_tiny_box(
            text,
            max_words=4 if rendered_area < 6000 else 6,
        )
        if short_text != text:
            font2, lines2, line_spacing2 = _fit_text_to_box(
                draw=draw,
                text=short_text,
                max_w=avail_w,
                max_h=avail_h,
                font_path=font_file,
                style=text_style,
                max_size_cap=font_cap,
            )
            if lines2:
                font, lines, line_spacing = font2, lines2, line_spacing2

    if not lines:
        return image
    if _should_skip_render_text(
        text=text,
        box_w=box_w,
        box_h=box_h,
        text_style=text_style,
        bubble_used=bubble_used,
    ):
        return image

    # Calculate total text height
    line_metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_metrics.append((lw, lh))

    total_text_height = sum(lh for _, lh in line_metrics) + (len(lines) - 1) * line_spacing

    # Center vertically
    start_y = box_y + text_padding + max(0, (avail_h - total_text_height) // 2)
    x_anchor = box_x + text_padding + max(0, (box_w - 2 * text_padding - avail_w) // 2)
    stroke_width = 1 if getattr(font, "size", 12) < 18 else 2

    # Draw each line centered horizontally
    current_y = start_y
    for i, line in enumerate(lines):
        lw, lh = line_metrics[i]
        line_x = x_anchor + max(0, (avail_w - lw) // 2)

        # Draw outline for readability (stroke)
        draw.text(
            (line_x, current_y),
            line,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

        current_y += lh + line_spacing

    # Convert back to BGR
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def _infer_text_style(text: str, w: int, h: int) -> str:
    """Classify translated text into a coarse style bucket."""
    clean = text.strip()
    words = clean.split()
    alpha_len = sum(1 for c in clean if c.isalpha())
    has_sentence_punct = any(c in clean for c in ".?!")

    if (len(words) <= 2 and alpha_len <= 12 and not has_sentence_punct) or (
        h < int(w * 0.55) and len(words) <= 3
    ):
        return "sfx"
    return "dialogue"


def _resolve_text_box(
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None,
    region_image: NDArray,
    translated_text: str,
    text_style: str,
) -> tuple[int, int, int, int, bool]:
    """
    Resolve a final text placement box.

    Dialogue uses bubble-aware bounds, while SFX stays anchored to source glyph
    clusters for closer placement.
    """
    if text_style == "dialogue":
        return _resolve_dialogue_box(
            x=x,
            y=y,
            w=w,
            h=h,
            region_mask=region_mask,
            region_image=region_image,
            translated_text=translated_text,
        )
    sx, sy, sw, sh = _resolve_sfx_box(x=x, y=y, w=w, h=h, region_mask=region_mask)
    return (sx, sy, sw, sh, False)


def _resolve_dialogue_box(
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None,
    region_image: NDArray,
    translated_text: str,
) -> tuple[int, int, int, int, bool]:
    """
    Resolve a box for dialogue text.

    This follows the same idea as comic-translate's best-render-area flow:
    prioritize bubble interior (when detectable), otherwise keep the full region.
    """
    inset = max(2, int(min(w, h) * 0.03))
    default_box = (
        x + inset,
        y + inset,
        max(1, w - 2 * inset),
        max(1, h - 2 * inset),
    )
    fallback_box = _dialogue_fallback_box(
        x=x,
        y=y,
        w=w,
        h=h,
        default_box=default_box,
        region_mask=region_mask,
        translated_text=translated_text,
    )

    anchor = _mask_centroid(region_mask, w, h)
    if anchor is None:
        anchor = (w // 2, h // 2)
    bubble = _estimate_bubble_box(region_image=region_image, anchor=anchor)
    if bubble is None:
        return (*fallback_box, False)

    bx, by, bw, bh = bubble
    bubble_area = bw * bh
    region_area = max(1, w * h)
    if bubble_area < int(region_area * 0.16):
        return (*fallback_box, False)
    if bubble_area > int(region_area * 0.97):
        return (*fallback_box, False)

    # Shrink inside the bubble so text stays off the border.
    shrink_x = max(4, int(bw * 0.10))
    shrink_y = max(4, int(bh * 0.12))
    x1 = min(w - 1, max(0, bx + shrink_x))
    y1 = min(h - 1, max(0, by + shrink_y))
    x2 = min(w, max(x1 + 1, bx + bw - shrink_x))
    y2 = min(h, max(y1 + 1, by + bh - shrink_y))

    fit_w = x2 - x1
    fit_h = y2 - y1
    min_w = max(28, int(w * 0.30))
    min_h = max(24, int(h * 0.30))
    if fit_w < min_w or fit_h < min_h:
        return (*fallback_box, False)

    # For longer dialogue, avoid overly tight boxes.
    if len(translated_text.strip()) >= 18:
        min_long_w = max(min_w, int(w * 0.45))
        min_long_h = max(min_h, int(h * 0.42))
        if fit_w < min_long_w or fit_h < min_long_h:
            return (*fallback_box, False)

    return (x + x1, y + y1, fit_w, fit_h, True)


def _dialogue_fallback_box(
    x: int,
    y: int,
    w: int,
    h: int,
    default_box: tuple[int, int, int, int],
    region_mask: NDArray | None,
    translated_text: str,
) -> tuple[int, int, int, int]:
    """
    Fallback dialogue box when bubble detection is unreliable.

    Uses a tighter mask-anchored region only if it is meaningfully smaller than
    the merged region; otherwise keep the default region box.
    """
    tight_box = _resolve_sfx_box(x=x, y=y, w=w, h=h, region_mask=region_mask)
    default_area = max(1, default_box[2] * default_box[3])
    tight_area = max(1, tight_box[2] * tight_box[3])
    area_ratio = tight_area / float(default_area)

    if tight_box[2] < 22 or tight_box[3] < 20:
        return default_box
    if area_ratio < 0.15:
        return default_box
    if area_ratio > 0.88:
        return default_box
    # Avoid forcing long dialogue into very narrow fallback boxes.
    long_dialogue = len(translated_text.strip()) >= 18
    if long_dialogue and tight_box[2] < int(default_box[2] * 0.50):
        return default_box
    if long_dialogue and (tight_box[2] * 1.0 / max(1, tight_box[3])) < 0.34:
        return default_box
    return tight_box


def _resolve_sfx_box(
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None,
) -> tuple[int, int, int, int]:
    """Resolve a tighter box for short SFX/interjection text."""
    if region_mask is None or region_mask.shape[:2] != (h, w):
        return (x, y, w, h)

    if region_mask.max() > 1:
        _, binary = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (region_mask > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary) < 20:
        return (x, y, w, h)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        components.append((cx, cy, cx + cw, cy + ch, area))

    if not components:
        return (x, y, w, h)

    components.sort(key=lambda item: item[4], reverse=True)
    px1, py1, px2, py2, _ = components[0]
    pcx = (px1 + px2) // 2
    pcy = (py1 + py2) // 2
    neighbor_dist = max(24, int(min(w, h) * 0.28))

    selected: list[tuple[int, int, int, int]] = [(px1, py1, px2, py2)]
    for cx1, cy1, cx2, cy2, _ in components[1:]:
        ccx = (cx1 + cx2) // 2
        ccy = (cy1 + cy2) // 2
        if abs(ccx - pcx) <= neighbor_dist and abs(ccy - pcy) <= neighbor_dist:
            selected.append((cx1, cy1, cx2, cy2))

    tx1 = min(item[0] for item in selected)
    ty1 = min(item[1] for item in selected)
    tx2 = max(item[2] for item in selected)
    ty2 = max(item[3] for item in selected)

    text_w = tx2 - tx1
    text_h = ty2 - ty1
    expand_x = max(5, int(text_w * 0.45))
    expand_y = max(5, int(text_h * 0.60))

    bx1 = max(0, tx1 - expand_x)
    by1 = max(0, ty1 - expand_y)
    bx2 = min(w, tx2 + expand_x)
    by2 = min(h, ty2 + expand_y)

    min_w = max(28, int(w * 0.28))
    min_h = max(22, int(h * 0.24))
    if (bx2 - bx1) < min_w:
        cx = (bx1 + bx2) // 2
        half = min_w // 2
        bx1 = max(0, cx - half)
        bx2 = min(w, cx + half)
    if (by2 - by1) < min_h:
        cy = (by1 + by2) // 2
        half = min_h // 2
        by1 = max(0, cy - half)
        by2 = min(h, cy + half)

    return (x + bx1, y + by1, max(1, bx2 - bx1), max(1, by2 - by1))


def _mask_centroid(
    region_mask: NDArray | None, w: int, h: int
) -> tuple[int, int] | None:
    """Return centroid of non-zero mask pixels in local coordinates."""
    if region_mask is None or region_mask.shape[:2] != (h, w):
        return None
    if region_mask.max() > 1:
        _, binary = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (region_mask > 0).astype(np.uint8) * 255

    ys, xs = np.where(binary > 0)
    if len(xs) < 20:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


def _estimate_bubble_box(
    region_image: NDArray,
    anchor: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Estimate speech-bubble bounds from bright connected components."""
    if region_image.size == 0:
        return None

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    region_h, region_w = gray.shape[:2]
    ax = int(np.clip(anchor[0], 0, max(0, region_w - 1)))
    ay = int(np.clip(anchor[1], 0, max(0, region_h - 1)))

    _, bright = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    # Fast path: if anchor lies inside a plausible bright component, use it.
    if bright[ay, ax] > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        label = int(labels[ay, ax])
        if label > 0 and label < num_labels:
            x, y, w, h, area = stats[label]
            if (
                area >= 600
                and w >= 28
                and h >= 24
                and (w * h) < int(region_w * region_h * 0.92)
            ):
                return (int(x), int(y), int(w), int(h))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_box: tuple[int, int, int, int] | None = None
    best_score = -1e9

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 26 or h < 22:
            continue

        # Ignore near-full panel regions.
        if (w * h) > int(region_w * region_h * 0.9):
            continue

        inside = cv2.pointPolygonTest(contour, (float(ax), float(ay)), False) >= 0
        center_dist = abs((x + w // 2) - ax) + abs((y + h // 2) - ay)
        score = area - center_dist * 1.6
        if inside:
            score += 4000

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box


def _fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    font_path: str | None,
    style: str = "dialogue",
    min_size: int = 8,
    max_size: int = 48,
    max_size_cap: int | None = None,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
    """Find the largest font size where the wrapped text fits in the box."""

    if style == "dialogue":
        max_size = min(max_size, 42)
    else:
        max_size = min(max_size, 54)
    if max_size_cap is not None:
        max_size = min(max_size, max_size_cap)

    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
            return ImageFont.load_default(size=size)
        except (OSError, TypeError):
            return ImageFont.load_default()

    def _layout_at_size(
        size: int,
    ) -> tuple[
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
        list[str],
        int,
        int,
        int,
    ]:
        font = _load_font(size)
        lines = _wrap_text(draw, text, font, max_w)
        spacing = _line_spacing_for_size(size, style)
        if not lines:
            return font, [], spacing, 0, 0

        widths: list[int] = []
        heights: list[int] = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(max(0, bbox[2] - bbox[0]))
            heights.append(max(0, bbox[3] - bbox[1]))

        max_line_w = max(widths) if widths else 0
        total_h = sum(heights) + (len(lines) - 1) * spacing
        return font, lines, spacing, max_line_w, total_h

    lo, hi = min_size, max_size
    best: tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font, lines, spacing, used_w, used_h = _layout_at_size(mid)
        if lines and used_w <= max_w and used_h <= max_h:
            best = (font, lines, spacing)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    # Fallback to minimum size.
    font, lines, spacing, _, _ = _layout_at_size(min_size)
    return font, lines, spacing


def _line_spacing_for_size(size: int, style: str) -> int:
    """Compute line spacing as a function of font size and text style."""
    if style == "sfx":
        return max(1, int(size * 0.10))
    return max(2, int(size * 0.14))


def _compress_dialogue_for_tiny_box(text: str, max_words: int) -> str:
    """Shorten long dialogue for very small render boxes."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean

    words = clean.split(" ")
    if len(words) <= max_words:
        return clean

    trimmed = " ".join(words[:max_words]).rstrip(".,;:!?")
    return f"{trimmed}..."


def _is_punctuation_only(text: str) -> bool:
    """Return True when text has no letters/digits and is only punctuation/symbols."""
    cleaned = text.strip()
    if not cleaned:
        return True
    return not any(ch.isalnum() for ch in cleaned)


def _should_skip_render_text(
    text: str,
    box_w: int,
    box_h: int,
    text_style: str,
    bubble_used: bool,
) -> bool:
    """Suppress tiny noisy fragments that produce unreadable clutter."""
    clean = " ".join(text.split())
    if not clean:
        return True
    if _is_punctuation_only(clean):
        return True

    area = max(1, box_w * box_h)
    alpha_num = sum(1 for c in clean if c.isalnum())
    words = clean.split(" ")

    if area < 900 and alpha_num <= 2:
        return True
    if text_style == "sfx" and area < 1400 and len(words) <= 1 and alpha_num <= 3:
        return True
    if (not bubble_used) and area < 1700 and len(words) <= 2 and alpha_num <= 4:
        return True
    if (not bubble_used) and area < 2400 and len(words) <= 1 and alpha_num <= 5:
        return True
    if (not bubble_used) and area < 3000 and len(words) <= 2 and alpha_num <= 3:
        return True
    return False


def _contains_cjk(text: str) -> bool:
    """Return True if the string contains CJK characters."""
    for c in text:
        code = ord(c)
        if (
            0x3040 <= code <= 0x30FF  # Hiragana / Katakana
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        ):
            return True
    return False


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    paragraphs = [p.strip() for p in text.replace("\r", "\n").split("\n") if p.strip()]
    if not paragraphs:
        return []

    wrapped: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

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

        # If a word/line exceeds width, fall back to character-level wrapping.
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                wrapped.append(line)
                continue

            current = ""
            for c in line:
                test = current + c
                cbbox = draw.textbbox((0, 0), test, font=font)
                if cbbox[2] - cbbox[0] <= max_width:
                    current = test
                else:
                    if current:
                        wrapped.append(current)
                    current = c
            if current:
                wrapped.append(current)

    return wrapped
