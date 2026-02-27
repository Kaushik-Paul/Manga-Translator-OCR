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
import re

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_FONTS_DIR = _MODULE_DIR / "fonts"


def _font_paths(*font_files: str) -> list[str]:
    """Build candidate font paths from bundled assets."""
    return [str(_FONTS_DIR / file_name) for file_name in font_files]


# Use bundled fonts so deployments (e.g. Spaces) do not depend on host system fonts.
_DIALOGUE_FONT_SEARCH_PATHS = _font_paths(
    "NimbusSans-Bold.otf",
    "LiberationSansNarrow-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "FreeSansBold.ttf",
)

_SFX_FONT_SEARCH_PATHS = _font_paths(
    "NimbusSans-BoldItalic.otf",
    "FreeSansBoldOblique.ttf",
    "LiberationSansNarrow-BoldItalic.ttf",
    "LiberationSans-BoldItalic.ttf",
    "FreeSansOblique.ttf",
)

_NARROW_DIALOGUE_FONT_SEARCH_PATHS = _font_paths(
    "LiberationSansNarrow-Bold.ttf",
    "NimbusSansNarrow-Bold.otf",
    "UbuntuSans[wdth,wght].ttf",
)

_CJK_DIALOGUE_FONT_SEARCH_PATHS = _font_paths(
    "NotoSansCJK-Bold.ttc",
    "wqy-zenhei.ttc",
    "DroidSansFallbackFull.ttf",
)


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
    style_hint: str | None = None,
    padding: int = 4,
) -> NDArray:
    """
    Render translated text within a bounding box on the image.

    Auto-sizes the font to fit, with word wrapping and centering.

    Args:
        image: Full page image (BGR).
        text: Translated text to render.
        x, y, w, h: Bounding box of the target region.
        font_path: Path to a TTF font file.
        style_hint: Optional style override ("dialogue" or "sfx").
        padding: Internal padding.

    Returns:
        Modified image with text rendered.
    """
    if not text.strip():
        return image

    result = image.copy()
    pre_render = result.copy()
    if style_hint in {"dialogue", "sfx"}:
        text_style = style_hint
    else:
        text_style = _infer_text_style(text, w, h)

    # Heuristic override: the detector/OCR may label a region as SFX based on
    # short source text, but the translation can clearly be dialogue.
    if text_style == "sfx":
        maybe_dialogue = _infer_text_style(text, w, h) == "dialogue"
        word_count = len(text.strip().split())
        has_sentence_punct = any(c in text for c in ".?!")
        dialogue_markers = {
            "i",
            "i'm",
            "im",
            "you",
            "you're",
            "youre",
            "we",
            "he",
            "she",
            "they",
            "it",
            "this",
            "that",
            "that's",
            "thats",
            "what",
            "why",
            "how",
            "when",
            "where",
        }
        alpha_words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
        marker_like = word_count >= 3 and any(w in dialogue_markers for w in alpha_words)

        if (
            maybe_dialogue
            or word_count >= 4
            or (word_count >= 2 and has_sentence_punct)
            or marker_like
        ):
            text_style = "dialogue"

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    region_slice = result[y : y + h, x : x + w]

    # Place text near the original glyph cluster inside the region when possible.
    # For longer dialogue, optionally expand toward the enclosing bubble shape.
    box_x, box_y, box_w, box_h, bubble_used, bubble_clip_mask_local = _resolve_text_box(
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
        # Non-bubble regions are often noisy merged text; force compact phrasing.
        text = _tighten_non_bubble_dialogue(text=text, box_w=box_w, box_h=box_h)
        if _is_low_signal_dialogue_fragment(text):
            return image
        if _prefer_sfx_for_free_text(text=text, box_w=box_w, box_h=box_h):
            text_style = "sfx"
            non_bubble_dialogue = False
    if _is_punctuation_only(text):
        return image

    is_tall_dialogue = text_style == "dialogue" and box_h > box_w * 1.20
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

    text_padding = max(2, padding - 2) if text_style == "dialogue" else max(1, padding - 3)
    avail_w = box_w - 2 * text_padding
    avail_h = box_h - 2 * text_padding
    box_clip_mask = _crop_clip_mask_to_box(
        bubble_clip_mask_local=bubble_clip_mask_local,
        region_x=x,
        region_y=y,
        region_w=w,
        region_h=h,
        box_x=box_x,
        box_y=box_y,
        box_w=box_w,
        box_h=box_h,
    )
    if box_clip_mask is not None:
        effective_w = _effective_mask_text_width(box_clip_mask)
        if effective_w > 14:
            # Only narrow if the mask is meaningfully narrower than the box
            mask_avail = max(10, effective_w - 2 * text_padding)
            if mask_avail < avail_w * 0.85:
                avail_w = int(max(avail_w * 0.85, mask_avail))
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
    font, lines, line_spacing = _fit_text_to_box(
        draw=draw,
        text=text,
        max_w=avail_w,
        max_h=avail_h,
        font_path=font_file,
        style=text_style,
    )
    stroke_width = _stroke_width_for_font(font)
    if text_style == "sfx" and lines:
        widest = _max_line_width(draw, lines, font, stroke_width=stroke_width)
        if widest > int(avail_w * 1.08):
            compact_text = _compact_sfx_text(text)
            if compact_text and compact_text != text:
                font2, lines2, line_spacing2 = _fit_text_to_box(
                    draw=draw,
                    text=compact_text,
                    max_w=avail_w,
                    max_h=avail_h,
                    font_path=font_file,
                    style=text_style,
                )
                if lines2:
                    font, lines, line_spacing = font2, lines2, line_spacing2
                    text = compact_text
                    stroke_width = _stroke_width_for_font(font)
                    widest = _max_line_width(draw, lines, font, stroke_width=stroke_width)
        if (not bubble_used) and widest > int(avail_w * 1.12):
            return image

    # If dialogue still overflows into too many tiny lines, aggressively shorten.
    rendered_area = box_w * box_h
    if (
        text_style == "dialogue"
        and rendered_area < 4000
        and lines
        and (len(lines) >= 6 or getattr(font, "size", 12) <= 9)
    ):
        short_text = _compress_dialogue_for_tiny_box(
            text,
            max_words=3 if rendered_area < 2500 else 5,
        )
        if short_text != text:
            font2, lines2, line_spacing2 = _fit_text_to_box(
                draw=draw,
                text=short_text,
                max_w=avail_w,
                max_h=avail_h,
                font_path=font_file,
                style=text_style,
            )
            if lines2:
                font, lines, line_spacing = font2, lines2, line_spacing2
                text = short_text
                stroke_width = _stroke_width_for_font(font)

    if not lines:
        return image
    if _is_overbroken_sfx(text=text, text_style=text_style, lines=lines, font=font):
        return image
    if _should_skip_render_text(
        text=text,
        box_w=box_w,
        box_h=box_h,
        text_style=text_style,
        bubble_used=bubble_used,
    ):
        return image

    # Calculate total text height (anchor-aware to avoid clipping)
    line_metrics: list[tuple[int, int, int, int]] = []
    for line in lines:
        left, top, right, bottom = draw.textbbox(
            (0, 0),
            line,
            font=font,
            anchor="lt",
            stroke_width=stroke_width,
        )
        lw = max(0, int(right - left))
        lh = max(0, int(bottom - top))
        line_metrics.append((lw, lh, int(left), int(top)))

    total_text_height = (
        sum(lh for _, lh, _, _ in line_metrics) + (len(lines) - 1) * line_spacing
    )

    # Center vertically
    start_y = box_y + text_padding + max(0, (avail_h - total_text_height) // 2)
    x_anchor = box_x + text_padding

    # Bubble masks can be narrower near the top/bottom. If a centered layout would
    # spill outside the mask, try shifting vertically within the slack, then
    # retry with a slightly narrower wrap width to keep text inside the bubble.
    if box_clip_mask is not None and text_style == "dialogue":
        base_local_start = int(start_y - box_y)
        if not _layout_fits_bubble_mask(
            mask=box_clip_mask,
            start_y=base_local_start,
            line_metrics=line_metrics,
            line_spacing=line_spacing,
            margin=2,
        ):
            slack = max(0, int(avail_h - total_text_height))
            local_min = int(text_padding)
            local_max = int(text_padding + slack)
            max_shift = min(12, slack)
            shift_candidates = [min(6, max_shift), -min(6, max_shift), max_shift, -max_shift]
            for shift in shift_candidates:
                cand = int(np.clip(base_local_start + shift, local_min, local_max))
                if _layout_fits_bubble_mask(
                    mask=box_clip_mask,
                    start_y=cand,
                    line_metrics=line_metrics,
                    line_spacing=line_spacing,
                    margin=2,
                ):
                    start_y = box_y + cand
                    break
            else:
                narrowed_w = int(avail_w)
                for _ in range(2):
                    narrowed_w = max(10, int(narrowed_w * 0.90))
                    font2, lines2, line_spacing2 = _fit_text_to_box(
                        draw=draw,
                        text=text,
                        max_w=narrowed_w,
                        max_h=avail_h,
                        font_path=font_file,
                        style=text_style,
                    )
                    if not lines2:
                        break
                    font, lines, line_spacing = font2, lines2, line_spacing2
                    stroke_width = _stroke_width_for_font(font)

                    line_metrics = []
                    for line in lines:
                        left, top, right, bottom = draw.textbbox(
                            (0, 0),
                            line,
                            font=font,
                            anchor="lt",
                            stroke_width=stroke_width,
                        )
                        lw = max(0, int(right - left))
                        lh = max(0, int(bottom - top))
                        line_metrics.append((lw, lh, int(left), int(top)))

                    total_text_height = (
                        sum(lh for _, lh, _, _ in line_metrics)
                        + (len(lines) - 1) * line_spacing
                    )
                    start_y = box_y + text_padding + max(
                        0, (avail_h - total_text_height) // 2
                    )
                    base_local_start = int(start_y - box_y)
                    if _layout_fits_bubble_mask(
                        mask=box_clip_mask,
                        start_y=base_local_start,
                        line_metrics=line_metrics,
                        line_spacing=line_spacing,
                        margin=2,
                    ):
                        break

    # Draw each line centered horizontally
    current_y = start_y
    for i, line in enumerate(lines):
        lw, lh, left, top = line_metrics[i]
        line_x = x_anchor + max(0, (avail_w - lw) // 2)
        if box_clip_mask is not None:
            local_y = current_y - box_y
            target_rel_x = line_x - box_x
            masked_rel_x = _fit_line_x_to_mask(
                mask=box_clip_mask,
                y=local_y,
                line_h=lh,
                line_w=lw,
                default_x=target_rel_x,
            )
            if masked_rel_x is not None:
                line_x = box_x + masked_rel_x

        # Draw outline for readability (stroke)
        draw.text(
            (int(line_x - left), int(current_y - top)),
            line,
            font=font,
            anchor="lt",
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

        current_y += lh + line_spacing

    # Convert back to BGR
    rendered = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return rendered


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
) -> tuple[int, int, int, int, bool, NDArray | None]:
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
    return (sx, sy, sw, sh, False, None)


def _resolve_dialogue_box(
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None,
    region_image: NDArray,
    translated_text: str,
) -> tuple[int, int, int, int, bool, NDArray | None]:
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

    center_anchor = (w // 2, h // 2)
    mask_anchor = _mask_centroid(region_mask, w, h)
    anchors: list[tuple[int, int]] = [center_anchor]
    if mask_anchor is not None and mask_anchor != center_anchor:
        anchors.append(mask_anchor)

    bubble_mask = None
    for anchor in anchors:
        bubble_mask = _extract_bubble_mask(
            region_image=region_image,
            anchor=anchor,
            region_mask=region_mask,
        )
        if bubble_mask is not None:
            break
    if bubble_mask is None:
        for anchor in anchors:
            bubble_box = _estimate_bubble_box(region_image=region_image, anchor=anchor)
            if bubble_box is None:
                continue
            bx, by, bw, bh = bubble_box
            bubble_mask = np.zeros((h, w), dtype=np.uint8)
            bubble_mask[by : by + bh, bx : bx + bw] = 255
            break
        if bubble_mask is None:
            return (*fallback_box, False, None)

    bubble_bbox = _mask_bbox(bubble_mask)
    if bubble_bbox is None:
        return (*fallback_box, False, None)
    bx1, by1, bx2, by2 = bubble_bbox
    bx, by, bw, bh = bx1, by1, (bx2 - bx1), (by2 - by1)
    bubble_area = bw * bh
    region_area = max(1, w * h)
    if bubble_area < int(region_area * 0.16):
        return (*fallback_box, False, None)
    if bubble_area > int(region_area * 0.97):
        return (*fallback_box, False, None)

    bubble_render_mask = _prepare_bubble_render_mask(bubble_mask)
    if bubble_render_mask is None:
        return (*fallback_box, False, None)

    fit_bbox = _mask_bbox(bubble_render_mask)
    if fit_bbox is None:
        return (*fallback_box, False, None)
    x1, y1, x2, y2 = fit_bbox
    shrink_ratio = 0.08 if (y2 - y1) > (x2 - x1) * 1.20 else 0.06
    x1, y1, x2, y2 = _shrink_centered_box(x1, y1, x2, y2, shrink_ratio)
    fit_w = x2 - x1
    fit_h = y2 - y1
    min_w = max(28, int(w * 0.30))
    min_h = max(24, int(h * 0.30))
    if fit_w < min_w or fit_h < min_h:
        return (*fallback_box, False, None)

    # For longer dialogue, avoid overly tight boxes.
    if len(translated_text.strip()) >= 18:
        min_long_w = max(min_w, int(w * 0.45))
        min_long_h = max(min_h, int(h * 0.42))
        if fit_w < min_long_w or fit_h < min_long_h:
            return (*fallback_box, False, None)

    return (x + x1, y + y1, fit_w, fit_h, True, bubble_render_mask)


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


def _extract_bubble_mask(
    region_image: NDArray,
    anchor: tuple[int, int],
    region_mask: NDArray | None = None,
) -> NDArray | None:
    """
    Extract an explicit bubble mask using flood-fill, similar to MIT's approach.

    The seed is chosen near OCR-mask centroid and constrained by Canny edges so
    fill stays inside enclosed speech bubble boundaries.
    """
    if region_image.size == 0:
        return None

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None

    ax = int(np.clip(anchor[0], 0, max(0, w - 1)))
    ay = int(np.clip(anchor[1], 0, max(0, h - 1)))

    edges = cv2.Canny(blurred, 70, 140, L2gradient=True)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1 : h + 1, 1 : w + 1][edges > 0] = 1

    blocked_seed_mask: NDArray | None = None
    if region_mask is not None and region_mask.shape[:2] == (h, w):
        if region_mask.max() > 1:
            _, seed_block = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            seed_block = (region_mask > 0).astype(np.uint8) * 255
        blocked_seed_mask = cv2.dilate(
            seed_block.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    seed = _find_fill_seed(
        blurred,
        flood_mask,
        (ax, ay),
        blocked_mask=blocked_seed_mask,
    )
    if seed is None:
        return None
    sx, sy = seed

    fill_src = blurred.copy()
    diff = int(np.clip(np.std(blurred) * 0.30, 8, 24))
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    area, _, _, _ = cv2.floodFill(
        fill_src,
        flood_mask,
        (sx, sy),
        255,
        loDiff=diff,
        upDiff=diff,
        flags=flags,
    )
    min_area = max(220, int(h * w * 0.02))
    if area < min_area:
        return None

    filled = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    if filled[sy, sx] == 0:
        return None

    # Keep only the connected component containing the seed.
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    label = int(labels[sy, sx])
    if label <= 0 or label >= num_labels:
        return None

    component = (labels == label).astype(np.uint8) * 255
    comp_area = cv2.countNonZero(component)
    if comp_area < min_area:
        return None
    if comp_area > int(h * w * 0.94):
        return None

    # Very dark, huge fills are often panel/background leakage.
    mean_val = float(np.mean(gray[component > 0])) if comp_area > 0 else 0.0
    if mean_val < 95 and comp_area > int(h * w * 0.55):
        return None
    # Reject colorful large fills (skin/clothes/panel background) that are not
    # speech bubbles. Bubbles are typically low-saturation interiors.
    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
    sat_vals = hsv[:, :, 1][component > 0]
    if sat_vals.size > 0:
        mean_sat = float(np.mean(sat_vals))
        bright_ratio = float(np.mean(gray[component > 0] > 185))
        if mean_sat > 52 and comp_area > int(h * w * 0.12):
            return None
        if mean_sat > 42 and bright_ratio < 0.35 and comp_area > int(h * w * 0.18):
            return None

    k = _odd(max(3, int(round(np.sqrt(comp_area) / 30))))
    component = cv2.morphologyEx(
        component,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    component = cv2.morphologyEx(
        component,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return component


def _find_fill_seed(
    gray: NDArray,
    flood_mask: NDArray,
    seed: tuple[int, int],
    blocked_mask: NDArray | None = None,
) -> tuple[int, int] | None:
    """Find a nearby unblocked flood-fill seed, preferring brighter pixels."""
    h, w = gray.shape[:2]
    sx = int(np.clip(seed[0], 0, max(0, w - 1)))
    sy = int(np.clip(seed[1], 0, max(0, h - 1)))

    def is_open(x: int, y: int) -> bool:
        if not (0 <= x < w and 0 <= y < h):
            return False
        if flood_mask[y + 1, x + 1] != 0:
            return False
        if blocked_mask is not None and blocked_mask.shape[:2] == (h, w):
            return blocked_mask[y, x] == 0
        return True

    if is_open(sx, sy):
        return (sx, sy)

    max_radius = max(6, int(min(h, w) * 0.35))
    best: tuple[int, int] | None = None
    best_score = -1e9
    for r in range(1, max_radius + 1):
        x0 = max(0, sx - r)
        x1 = min(w - 1, sx + r)
        y0 = max(0, sy - r)
        y1 = min(h - 1, sy + r)

        for x in range(x0, x1 + 1):
            for y in (y0, y1):
                if not is_open(x, y):
                    continue
                score = float(gray[y, x]) - (abs(x - sx) + abs(y - sy)) * 1.2
                if score > best_score:
                    best_score = score
                    best = (x, y)

        for y in range(y0 + 1, y1):
            for x in (x0, x1):
                if not is_open(x, y):
                    continue
                score = float(gray[y, x]) - (abs(x - sx) + abs(y - sy)) * 1.2
                if score > best_score:
                    best_score = score
                    best = (x, y)

        if best is not None and r >= 4:
            break

    return best


def _prepare_bubble_render_mask(mask: NDArray) -> NDArray | None:
    """Slightly shrink the bubble mask to keep text away from the outline."""
    if mask.size == 0:
        return None
    comp_area = cv2.countNonZero(mask.astype(np.uint8))
    if comp_area < 180:
        return None

    k = _odd(max(3, int(round(np.sqrt(comp_area) / 36))))
    eroded = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    if cv2.countNonZero(eroded) >= max(120, int(comp_area * 0.35)):
        return eroded
    return mask.astype(np.uint8)


def _mask_bbox(mask: NDArray) -> tuple[int, int, int, int] | None:
    """Return x1,y1,x2,y2 bounds for non-zero mask pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _shrink_centered_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    shrink_percent: float,
) -> tuple[int, int, int, int]:
    """Shrink a box uniformly from the center (comic-translate style)."""
    if shrink_percent <= 0:
        return (x1, y1, x2, y2)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    scale = max(0.2, 1.0 - shrink_percent)
    nw = w * scale
    nh = h * scale
    sx1 = int(round(cx - nw * 0.5))
    sy1 = int(round(cy - nh * 0.5))
    sx2 = int(round(cx + nw * 0.5))
    sy2 = int(round(cy + nh * 0.5))
    if sx2 <= sx1 or sy2 <= sy1:
        return (x1, y1, x2, y2)
    return (sx1, sy1, sx2, sy2)


def _odd(value: int) -> int:
    """Round up to odd integer."""
    return value if value % 2 == 1 else value + 1


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
    max_size: int = 120,
    max_size_cap: int | None = None,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
    """Find the largest font size where the wrapped text fits in the box."""

    # Dynamically compute max font size from box dimensions so text fills
    # the available space, similar to comic-translate's approach.
    word_count = len(text.split())
    if style == "dialogue":
        # For dialogue: scale based on box height but also consider width.
        # More words need smaller starting sizes to leave room for wrapping.
        if word_count <= 3:
            dynamic_max = min(120, max(max_h, max_w))
        else:
            dynamic_max = min(120, max(24, int(max_h * 0.7)))
        max_size = min(max_size, dynamic_max)
    else:
        # SFX: can go large, scale with box dimensions
        dynamic_max = min(120, max(max_h, max_w))
        max_size = min(max_size, dynamic_max)
        min_size = min(min_size, 6)
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
        allow_hyphenation: bool = False,
    ) -> tuple[
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
        list[str],
        int,
        int,
        int,
    ]:
        stroke_width = max(1, size // 12)
        font = _load_font(size)
        lines = _wrap_text(
            draw,
            text,
            font,
            max_w,
            allow_char_wrap=False,
            allow_hyphenation=allow_hyphenation,
            stroke_width=stroke_width,
        )
        spacing = _line_spacing_for_size(size, style)
        if not lines:
            return font, [], spacing, 0, 0

        widths: list[int] = []
        heights: list[int] = []
        for line in lines:
            left, top, right, bottom = draw.textbbox(
                (0, 0),
                line,
                font=font,
                anchor="lt",
                stroke_width=stroke_width,
            )
            widths.append(max(0, int(right - left)))
            heights.append(max(0, int(bottom - top)))

        max_line_w = max(widths) if widths else 0
        total_h = sum(heights) + (len(lines) - 1) * spacing
        return font, lines, spacing, max_line_w, total_h

    # Binary search WITHOUT hyphenation — prefer smaller font over word breaks.
    lo, hi = min_size, max_size
    best: tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font, lines, spacing, used_w, used_h = _layout_at_size(mid, allow_hyphenation=False)
        if lines and used_w <= max_w and used_h <= max_h:
            best = (font, lines, spacing)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    # Binary search WITH hyphenation if the first pass failed.
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font, lines, spacing, used_w, used_h = _layout_at_size(mid, allow_hyphenation=True)
        if lines and used_w <= max_w and used_h <= max_h:
            best = (font, lines, spacing)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    # Fallback to minimum size WITH hyphenation as last resort.
    font, lines, spacing, _, _ = _layout_at_size(min_size, allow_hyphenation=True)
    return font, lines, spacing


def _stroke_width_for_font(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """Match the outline thickness used at render time."""
    font_size = getattr(font, "size", 12)
    return max(1, font_size // 12)


def _line_spacing_for_size(size: int, style: str) -> int:
    """Compute line spacing as a function of font size and text style."""
    if style == "sfx":
        return max(1, int(size * 0.08))
    return max(2, int(size * 0.12))


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


def _tighten_non_bubble_dialogue(text: str, box_w: int, box_h: int) -> str:
    """Apply light budgets for dialogue rendered outside detected bubbles."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean
    # Just return the cleaned text — let the renderer handle fitting.
    # Only do a light trim for extremely long text in tiny boxes.
    area = max(1, box_w * box_h)
    words = clean.split()
    if area < 3000 and len(words) > 6:
        return _compress_dialogue_for_tiny_box(clean, max_words=4)
    return clean


def _is_punctuation_only(text: str) -> bool:
    """Return True when text has no letters/digits and is only punctuation/symbols."""
    cleaned = text.strip()
    if not cleaned:
        return True
    # Keep ellipsis-only bubbles instead of blanking them out.
    if _is_ellipsis_like(cleaned):
        return False
    return not any(ch.isalnum() for ch in cleaned)


def _should_skip_render_text(
    text: str,
    box_w: int,
    box_h: int,
    text_style: str,
    bubble_used: bool,
) -> bool:
    """Suppress truly tiny noise that would be unreadable."""
    clean = " ".join(text.split())
    if not clean:
        return True
    if _is_punctuation_only(clean):
        return True
    if _is_ellipsis_like(clean):
        return False

    area = max(1, box_w * box_h)
    # Only skip truly tiny regions where no text could be legible
    if area < 400:
        return True
    return False


def _prefer_sfx_for_free_text(text: str, box_w: int, box_h: int) -> bool:
    """Use SFX style for very short non-bubble snippets."""
    clean = " ".join(text.split())
    words = clean.split(" ") if clean else []
    return len(words) <= 2


def _is_low_signal_dialogue_fragment(text: str) -> bool:
    """Detect truly empty/meaningless fragments from noisy OCR/translation."""
    tokens = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if not tokens:
        return True
    return False


def _compact_sfx_text(text: str) -> str:
    """Compress noisy SFX phrases to improve fit in narrow boxes."""
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return text.strip()

    compacted: list[str] = []
    for token in tokens[:3]:
        t = token.lower()
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        compacted.append(t)

    clean = " ".join(compacted)
    if len(compacted) >= 2:
        clean = " ".join(compacted[:2])
    if len(clean) > 14:
        clean = clean[:14].rstrip()
    return clean.capitalize()


def _max_line_width(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    stroke_width: int = 0,
) -> int:
    """Return the widest line in pixels."""
    if not lines:
        return 0
    widths: list[int] = []
    for line in lines:
        left, _, right, _ = draw.textbbox(
            (0, 0),
            line,
            font=font,
            anchor="lt",
            stroke_width=stroke_width,
        )
        widths.append(max(0, int(right - left)))
    return max(widths) if widths else 0


def _is_overbroken_sfx(
    text: str,
    text_style: str,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> bool:
    """
    Suppress tiny SFX renders that break one short word into many fragments.
    """
    if text_style != "sfx":
        return False
    words = text.split()
    if len(lines) <= 1:
        return False

    font_size = getattr(font, "size", 12)
    if len(words) == 1:
        token = words[0].strip(".,;:!?-_'\"")
        if len(token) <= 7 and len(lines) >= 2 and font_size <= 10:
            return True
        if len(token) <= 10 and len(lines) >= 3:
            return True

    if len(words) <= 2 and len(lines) >= 4 and font_size <= 11:
        return True

    return font_size <= 8


def _crop_clip_mask_to_box(
    bubble_clip_mask_local: NDArray | None,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
) -> NDArray | None:
    """Crop region-local bubble mask into the chosen render box."""
    if (
        bubble_clip_mask_local is None
        or bubble_clip_mask_local.shape[:2] != (region_h, region_w)
    ):
        return None

    lx1 = box_x - region_x
    ly1 = box_y - region_y
    lx2 = lx1 + box_w
    ly2 = ly1 + box_h
    if lx1 < 0 or ly1 < 0 or lx2 > region_w or ly2 > region_h:
        return None

    crop = bubble_clip_mask_local[ly1:ly2, lx1:lx2].astype(np.uint8)
    if crop.shape[:2] != (box_h, box_w):
        return None
    if cv2.countNonZero(crop) < 20:
        return None
    return crop


def _effective_mask_text_width(mask: NDArray) -> int:
    """Estimate an interior usable line width from a bubble mask."""
    rows = mask > 0
    widths = rows.sum(axis=1)
    widths = widths[widths > 0]
    if widths.size == 0:
        return 0
    return int(np.percentile(widths, 45))


def _mask_band_span(mask: NDArray, y: int, line_h: int, line_w: int) -> tuple[int, int, int] | None:
    """Return (left, right, avail) span for a horizontal band inside the mask."""
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return None
    y0 = max(0, int(y))
    y1 = min(h, int(y + max(1, line_h)))
    if y1 <= y0:
        return None

    band = mask[y0:y1, :] > 0
    row_left: list[int] = []
    row_right: list[int] = []
    for row in band:
        xs = np.where(row)[0]
        if xs.size == 0:
            continue
        row_left.append(int(xs.min()))
        row_right.append(int(xs.max()))

    if not row_left:
        return None

    left = int(np.percentile(np.array(row_left), 25))
    right = int(np.percentile(np.array(row_right), 75))
    avail = right - left + 1
    if avail < max(8, int(line_w * 0.55)):
        # Fall back to widest visible span in this band.
        best_idx = int(np.argmax(np.array(row_right) - np.array(row_left)))
        left = row_left[best_idx]
        right = row_right[best_idx]
        avail = right - left + 1

    if avail <= 6:
        return None
    return left, right, avail


def _fit_line_x_to_mask(
    mask: NDArray,
    y: int,
    line_h: int,
    line_w: int,
    default_x: int,
) -> int | None:
    """Center a line inside the mask span for the corresponding scan rows."""
    span = _mask_band_span(mask=mask, y=y, line_h=line_h, line_w=line_w)
    if span is None:
        return None
    left, right, avail = span
    if avail < line_w:
        return None

    h, w = mask.shape[:2]
    x = left + max(0, (avail - line_w) // 2)
    max_x = max(0, w - line_w)
    if max_x <= 0:
        return 0
    return int(np.clip(x, 0, max_x))


def _layout_fits_bubble_mask(
    mask: NDArray,
    start_y: int,
    line_metrics: list[tuple[int, int, int, int]],
    line_spacing: int,
    margin: int = 2,
) -> bool:
    """Return True if every line's width fits within the bubble mask band."""
    if mask.size == 0:
        return True

    current_y = int(start_y)
    for line_w, line_h, _, _ in line_metrics:
        span = _mask_band_span(mask=mask, y=current_y, line_h=line_h, line_w=line_w)
        if span is None:
            return False
        _, _, avail = span
        if avail < (line_w + margin):
            return False
        current_y += int(line_h + line_spacing)

    return True


def _is_ellipsis_like(text: str) -> bool:
    """Return True when text is only ellipsis-like punctuation."""
    compact = text.replace(" ", "").strip()
    if not compact:
        return False
    return all(ch in ".…·･・" for ch in compact)


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


def _split_long_token_to_fit(
    *,
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    stroke_width: int = 0,
) -> list[str]:
    """
    Split a single token into multiple pieces so each piece fits max_width.

    This is a last-resort safeguard for extremely long words or hyphenation
    parts that still don't fit, to avoid rendering outside the bubble.
    """
    if not token:
        return [token]

    left, _, right, _ = draw.textbbox(
        (0, 0),
        token,
        font=font,
        anchor="lt",
        stroke_width=stroke_width,
    )
    if (right - left) <= max_width:
        return [token]

    # If the token already ends with a hyphen (e.g. from hyphenation),
    # drop it here — this fallback prefers breaking without inserting
    # additional hyphens.
    core = token[:-1] if token.endswith("-") and len(token) > 1 else token

    # Keep trailing punctuation on the last piece when possible.
    suffix = ""
    while core and core[-1] in ".,;:!?)]}\"":
        suffix = core[-1] + suffix
        core = core[:-1]

    pieces: list[str] = []
    current = ""
    for idx, ch in enumerate(core):
        candidate = current + ch
        t_left, _, t_right, _ = draw.textbbox(
            (0, 0),
            candidate,
            font=font,
            anchor="lt",
            stroke_width=stroke_width,
        )
        if (t_right - t_left) <= max_width or not current:
            current = candidate
            continue

        pieces.append(current)
        current = ch

    if current:
        pieces.append(current + suffix)
    elif suffix:
        if pieces:
            pieces[-1] = pieces[-1].rstrip("-") + suffix
        else:
            pieces.append(suffix)

    return pieces if pieces else [token]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    allow_char_wrap: bool = True,
    allow_hyphenation: bool = True,
    stroke_width: int = 0,
) -> list[str]:
    """Wrap text to fit within max_width pixels.

    Uses word-boundary wrapping only. Words are never split unless
    allow_hyphenation is True (syllable-aware splits) or allow_char_wrap
    is True (last-resort character splits).

    When both are False and a word exceeds max_width, returns [] to signal
    the caller that this font size is too large.
    """
    paragraphs = [p.strip() for p in text.replace("\r", "\n").split("\n") if p.strip()]
    if not paragraphs:
        return []

    def _measure(txt: str) -> int:
        bbox = draw.textbbox((0, 0), txt, font=font, anchor="lt", stroke_width=stroke_width)
        return max(0, int(bbox[2] - bbox[0]))

    wrapped: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        # Check if any single word exceeds max_width
        if not allow_char_wrap and not allow_hyphenation:
            for word in words:
                if _measure(word) > max_width:
                    return []  # Signal: font too large, shrink it

        # Greedy word-boundary wrapping
        lines: list[str] = []
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            if _measure(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        # Post-process lines that still exceed max_width
        for line in lines:
            if _measure(line) <= max_width:
                wrapped.append(line)
                continue

            if not allow_char_wrap and not allow_hyphenation:
                # Should not happen (checked above), but safety net
                return []

            # Line exceeds width — split individual words if needed
            line_words = line.split()
            sub_lines: list[str] = []
            for lw in line_words:
                lw_width = _measure(lw)
                if lw_width <= max_width:
                    if sub_lines:
                        test = sub_lines[-1] + " " + lw
                        if _measure(test) <= max_width:
                            sub_lines[-1] = test
                        else:
                            sub_lines.append(lw)
                    else:
                        sub_lines.append(lw)
                else:
                    # Word too wide — try hyphenation first
                    tokens: list[str]
                    if allow_hyphenation:
                        parts = _hyphenate_word(lw)
                        if len(parts) > 1:
                            tokens = [
                                part + "-" if pi < len(parts) - 1 else part
                                for pi, part in enumerate(parts)
                            ]
                        else:
                            tokens = [lw]
                    else:
                        tokens = [lw]

                    # Hard char splits as absolute last resort
                    safe_tokens: list[str] = []
                    for token in tokens:
                        if _measure(token) <= max_width:
                            safe_tokens.append(token)
                        elif allow_char_wrap:
                            safe_tokens.extend(
                                _split_long_token_to_fit(
                                    draw=draw,
                                    token=token,
                                    font=font,
                                    max_width=max_width,
                                    stroke_width=stroke_width,
                                )
                            )
                        else:
                            return []  # Can't fit without char splitting

                    for ti, token in enumerate(safe_tokens):
                        if not sub_lines:
                            sub_lines.append(token)
                            continue
                        sep = " " if ti == 0 else ""
                        test = sub_lines[-1] + sep + token
                        if _measure(test) <= max_width:
                            sub_lines[-1] = test
                        else:
                            sub_lines.append(token)
            wrapped.extend(sub_lines if sub_lines else [line])

    return wrapped


def _hyphenate_word(word: str) -> list[str]:
    """Split a word at natural English syllable boundaries.

    Returns a list of parts. If no good split is found, returns [word].
    Uses common English suffix patterns for break points.
    """
    clean = word.rstrip(".,;:!?-_'\"")
    suffix = word[len(clean):]
    if len(clean) < 4:
        return [word]

    # Common English break patterns (ordered by preference — try longest
    # suffix first so we get the most balanced split).
    patterns = [
        "tion", "sion", "ment", "ness", "able", "ible",
        "ful", "less", "ous", "ive", "ing", "ting",
        "ally", "ely", "ily", "ly",
        "er", "ed", "en", "est", "ize", "ise",
        "ated", "ious", "eous", "tial", "cial",
    ]

    low = clean.lower()
    best_split = -1
    for pat in patterns:
        idx = low.rfind(pat)
        if idx > 1 and idx + len(pat) >= len(clean) - 1:
            if idx > best_split:
                best_split = idx

    if best_split > 1 and best_split < len(clean) - 1:
        return [clean[:best_split], clean[best_split:] + suffix]

    # Fallback: split at roughly the middle between consonant/vowel boundary
    vowels = set("aeiouAEIOU")
    mid = len(clean) // 2
    # Search outward from the middle for a consonant→vowel transition
    for offset in range(0, len(clean) // 2):
        for pos in (mid + offset, mid - offset):
            if 2 <= pos < len(clean) - 1:
                if clean[pos - 1] not in vowels and clean[pos] in vowels:
                    return [clean[:pos], clean[pos:] + suffix]

    return [word]
