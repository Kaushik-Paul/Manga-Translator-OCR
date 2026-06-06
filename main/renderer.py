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


# Use bundled comic-lettering fonts for natural manga dialogue/SFX rendering.
# animeace2.otf has full printable ASCII (0x20-0x7E) and is safer as primary.
# CCWildWordsRoman.ttf has a limited charset (missing ^, _, {, }, |, ~) and
# renders unsupported chars as tofu boxes.
_DIALOGUE_FONT_SEARCH_PATHS = _font_paths(
    "animeace2.otf",
    "CCWildWordsRoman.ttf",
)

_SFX_FONT_SEARCH_PATHS = _font_paths(
    "animeace2.otf",
    "CCWildWordsRoman.ttf",
)

_NARROW_DIALOGUE_FONT_SEARCH_PATHS = _font_paths(
    "CCWildWordsRoman.ttf",
    "animeace2.otf",
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

# ---------------------------------------------------------------------------
# Font charset cache — prevents tofu □ boxes by checking glyph coverage
# ---------------------------------------------------------------------------
_FONT_CHARSET_CACHE: dict[str, set[int]] = {}


def _get_font_charset(font_path: str) -> set[int]:
    """Parse supported codepoints from a font file (cached)."""
    if font_path in _FONT_CHARSET_CACHE:
        return _FONT_CHARSET_CACHE[font_path]

    import subprocess

    supported: set[int] = set()
    try:
        result = subprocess.run(
            ["fc-query", "--format=%{charset}\n", font_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for part in result.stdout.strip().split():
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    for cp in range(int(start_s, 16), int(end_s, 16) + 1):
                        supported.add(cp)
                else:
                    supported.add(int(part, 16))
    except Exception as e:
        logger.debug("fc-query failed for %s: %s — using permissive fallback", font_path, e)
        # Fallback: assume printable ASCII + common Latin-1 are safe
        for cp in range(0x20, 0x7F):
            supported.add(cp)

    _FONT_CHARSET_CACHE[font_path] = supported
    return supported


def _sanitize_for_font(text: str, font_path: str | None) -> str:
    """Strip or replace characters that the given font cannot render.

    Characters outside the font's charset would render as □ (tofu) boxes.
    Common replacements (smart quotes → ASCII quotes, em-dash → hyphen, etc.)
    are applied first, then any remaining unsupported chars are removed.
    """
    if not font_path or not text:
        return text

    charset = _get_font_charset(font_path)
    if not charset:
        return text  # Can't determine charset, pass through

    # Quick check: if all characters are supported, return as-is
    if all(ord(c) in charset for c in text):
        return text

    # Apply common replacements for unsupported chars
    _REPLACEMENTS = {
        "\u2018": "'", "\u2019": "'", "\u201A": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u2013": "-", "\u2014": "-", "\u2012": "-",
        "\u2026": "...",
        "\u2605": "*", "\u2606": "*",
        "\u2022": "-",
        "\u00B7": ".",
        "\u2192": "->", "\u2190": "<-",
        "\u266A": "~", "\u266B": "~",
        "\u2665": "<3", "\u2764": "<3",
        "\u301C": "~", "\uFF5E": "~",
    }

    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in charset:
            out.append(ch)
        elif ch in _REPLACEMENTS:
            # Replace with ASCII equivalent, but only keep chars the font supports
            replacement = _REPLACEMENTS[ch]
            out.append("".join(c for c in replacement if ord(c) in charset) or "")
        elif 0x20 <= cp <= 0x7E:
            # Basic ASCII that the font doesn't support — unlikely but skip
            out.append("")
        else:
            # Unknown unsupported char — silently drop
            pass

    result = "".join(out)
    # Clean up doubled spaces from removed chars
    result = " ".join(result.split())
    return result.strip() if result.strip() else text


def _snap_extreme_neutrals(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Snap achromatic colours to pure black or white.

    Comic text is almost never intentionally grey.  If the detected
    colour is achromatic (low chroma) it is meant to be either black
    or white, so snap to whichever is closer.  Coloured text (high
    chroma) is returned unchanged.  For achromatic (low chroma)
    colours, returns ``None`` — the caller should fall back to its
    own brightness-based logic, which is more reliable for B&W manga.
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    chroma = max(r, g, b) - min(r, g, b)

    # Achromatic → return None so the brightness fallback handles it.
    # Most manga text is pure black or white; the brightness-based
    # fallback in render_text_on_image is more reliable for these.
    if chroma < 60:
        return None

    return (r, g, b)


def detect_text_color(
    region_image: NDArray,
    region_mask: NDArray | None = None,
) -> tuple[int, int, int] | None:
    """Detect the foreground (text) colour from an original manga region.

    Uses spatial analysis: border pixels define the background colour,
    then Otsu thresholding on the colour-distance-from-background map
    cleanly separates text from background.  The median colour of
    the text pixels is returned.

    Only returns a colour for clearly *chromatic* text (e.g. coloured
    dialogue in colour manga).  For black, white, or grey text the
    function returns ``None`` so that the caller's brightness-based
    fallback handles it — that logic is more robust for B&W pages.

    Args:
        region_image: Cropped region from the *original* image (BGR).
        region_mask: Per-pixel text mask for this region (255=text).

    Returns:
        RGB tuple of the detected text colour, or None if detection
        fails or the colour is achromatic (black/white/grey).
    """
    if region_image is None or region_image.size == 0:
        return None

    h, w = region_image.shape[:2]
    if h < 6 or w < 6:
        return None

    if len(region_image.shape) != 3 or region_image.shape[2] < 3:
        return None

    # Work in RGB for the colour result.
    img_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)

    # ── Strategy A: use the ML text mask directly if available ──
    if region_mask is not None and region_mask.shape[:2] == (h, w):
        if region_mask.max() > 1:
            _, binary_mask = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            binary_mask = (region_mask > 0).astype(np.uint8) * 255

        text_pixels = img_rgb.reshape(-1, 3)[binary_mask.ravel() > 0]
        if len(text_pixels) >= 5:
            median_color = np.median(text_pixels.astype(np.float64), axis=0)
            rgb = tuple(int(c) for c in median_color.round())
            return _snap_extreme_neutrals(rgb)

    # ── Strategy B: border-sampling + Otsu (comic-translate approach) ──
    # 1. Border sampling — thin ring of pixels around the edge = background
    bw = max(2, min(h, w) // 8)
    top = img_rgb[:bw, :]
    bottom = img_rgb[-bw:, :]
    left = img_rgb[bw:-bw, :bw]
    right = img_rgb[bw:-bw, -bw:]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float64)

    bg = np.median(border_pixels, axis=0)

    # 2. Per-pixel Euclidean distance from the background colour.
    flat = img_rgb.reshape(-1, 3).astype(np.float64)
    dist = np.sqrt(np.sum((flat - bg) ** 2, axis=1))

    # 3. Otsu threshold on the distance map.
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    _, otsu_mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(cv2.threshold(dist_u8, 0, 255, cv2.THRESH_OTSU)[0]), 25.0)

    # 4. Extract text pixels and compute their median colour.
    text_mask_flat = dist > threshold
    n_text = int(np.sum(text_mask_flat))
    if n_text < 5:
        return None

    fg = np.median(flat[text_mask_flat], axis=0).round().astype(int)
    rgb = (int(fg[0]), int(fg[1]), int(fg[2]))
    return _snap_extreme_neutrals(rgb)


def _outline_color_for_text(
    text_rgb: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Choose an outline/stroke colour that contrasts with the text colour."""
    r, g, b = text_rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luma < 128 else (0, 0, 0)


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

    # Dilate mask to cover text edges, anti-aliasing, and surrounding halo
    # Japanese text often has ink bleed that needs a larger dilation radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Inpaint using Telea algorithm (better for text removal)
    inpainted = cv2.inpaint(region, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
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
    text_color: tuple[int, int, int] | None = None,
    allow_bubble_expansion: bool = True,
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
            or (word_count >= 1 and has_sentence_punct and h > w * 1.2)
            or marker_like
        ):
            text_style = "dialogue"

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    region_slice = result[y : y + h, x : x + w]

    # For very narrow tall regions (typical of vertical Japanese text in bubbles),
    # expand the search area horizontally so bubble detection has a chance to find
    # the actual bubble outline, which is usually wider than the text mask.
    # Keep the expansion conservative so nearby regions don't collide into the
    # same bubble and produce overlapping translations.
    search_x, search_y, search_w, search_h = x, y, w, h
    search_region_image = region_slice
    search_region_mask = region_mask
    if allow_bubble_expansion and w < 80 and h > w * 1.5 and text_style == "dialogue":
        pad_x = min(max(int(w * 0.55), 20), 40)
        sx1 = max(0, x - pad_x)
        sx2 = min(result.shape[1], x + w + pad_x)
        search_x, search_y = sx1, y
        search_w, search_h = sx2 - sx1, h
        search_region_image = result[search_y : search_y + search_h, search_x : search_x + search_w]
        if region_mask is not None and region_mask.shape[:2] == (h, w):
            padded_mask = np.zeros((search_h, search_w), dtype=np.uint8)
            offset_x = x - search_x
            padded_mask[:, offset_x : offset_x + w] = (
                region_mask.astype(np.uint8) if region_mask.max() <= 1 else (region_mask > 0).astype(np.uint8) * 255
            )
            search_region_mask = padded_mask
        else:
            search_region_mask = None

    # Place text near the original glyph cluster inside the region when possible.
    # For longer dialogue, optionally expand toward the enclosing bubble shape.
    box_x, box_y, box_w, box_h, bubble_used, bubble_clip_mask_local = _resolve_text_box(
        x=search_x,
        y=search_y,
        w=search_w,
        h=search_h,
        region_mask=search_region_mask,
        region_image=search_region_image,
        translated_text=text,
        text_style=text_style,
    )

    def _skip_render(reason: str) -> NDArray:
        logger.debug(
            "Skipping render (%s): style=%s bubble=%s box=(%d,%d,%d,%d) text='%s'",
            reason,
            text_style,
            bubble_used,
            box_x,
            box_y,
            box_w,
            box_h,
            " ".join(text.split())[:80],
        )
        return image

    non_bubble_dialogue = text_style == "dialogue" and not bubble_used
    forced_box_clip_mask: NDArray | None = None
    if non_bubble_dialogue:
        # Non-bubble regions are often noisy merged text; force compact phrasing.
        text = _tighten_non_bubble_dialogue(text=text, box_w=box_w, box_h=box_h)
        if _is_low_signal_dialogue_fragment(text):
            return _skip_render("low-signal non-bubble dialogue")
        if _prefer_sfx_for_free_text(text=text, box_w=box_w, box_h=box_h):
            text_style = "sfx"
            non_bubble_dialogue = False
        else:
            surface = _non_bubble_dialogue_surface(
                region_image=result[box_y : box_y + box_h, box_x : box_x + box_w],
                text=text,
                box_w=box_w,
                box_h=box_h,
            )
            if surface is None:
                return _skip_render("no safe non-bubble surface")
            local_x, local_y, local_w, local_h, forced_box_clip_mask = surface
            box_x += local_x
            box_y += local_y
            box_w = local_w
            box_h = local_h
    if _is_punctuation_only(text):
        return _skip_render("punctuation-only")

    if (
        forced_box_clip_mask is not None
        and forced_box_clip_mask.shape[:2] != (box_h, box_w)
    ):
        forced_box_clip_mask = None

    if (
        non_bubble_dialogue
        and forced_box_clip_mask is None
        and not _non_bubble_dialogue_has_safe_surface(
            region_image=result[box_y : box_y + box_h, box_x : box_x + box_w],
            text=text,
            box_w=box_w,
            box_h=box_h,
        )
    ):
        return _skip_render("unsafe non-bubble surface")

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

    # Strip characters the selected font can't render (prevents □ tofu boxes).
    text = _sanitize_for_font(text, font_file)

    text_padding = max(1, padding - 4) if text_style == "dialogue" else max(1, padding - 4)
    avail_w = box_w - 2 * text_padding
    avail_h = box_h - 2 * text_padding
    # If the clip mask is already box-sized (from expanded bubble search),
    # use it directly; otherwise crop from the region-local mask.
    if (
        forced_box_clip_mask is not None
        and forced_box_clip_mask.shape[:2] == (box_h, box_w)
    ):
        box_clip_mask = forced_box_clip_mask
    elif (
        bubble_clip_mask_local is not None
        and bubble_clip_mask_local.shape[:2] == (box_h, box_w)
    ):
        box_clip_mask = bubble_clip_mask_local
    else:
        box_clip_mask = _crop_clip_mask_to_box(
            bubble_clip_mask_local=bubble_clip_mask_local,
            region_x=search_x,
            region_y=search_y,
            region_w=search_w,
            region_h=search_h,
            box_x=box_x,
            box_y=box_y,
            box_w=box_w,
            box_h=box_h,
        )
    if box_clip_mask is not None:
        effective_w = _effective_mask_text_width(box_clip_mask)
        if effective_w > 14:
            # Only narrow if the mask is meaningfully narrower than the box.
            # Use a relaxed threshold (0.80) so oval bubbles keep more width.
            mask_avail = max(10, effective_w - 2 * text_padding)
            if mask_avail < avail_w * 0.80:
                avail_w = int(max(avail_w * 0.80, mask_avail))
    if avail_w <= 10 or avail_h <= 10:
        return _skip_render("no available text area")

    # Determine outline color from background brightness (original logic).
    region = result[box_y : box_y + box_h, box_x : box_x + box_w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness > 140:
        outline_color = (255, 255, 255)
        brightness_color = (0, 0, 0)
    else:
        outline_color = (0, 0, 0)
        brightness_color = (255, 255, 255)

    # Use detected text colour for fill if provided, otherwise brightness fallback.
    # Guard: ensure detected colour has sufficient contrast against the background.
    render_color = brightness_color
    if text_color is not None:
        tr, tg, tb = text_color
        text_luma = 0.299 * tr + 0.587 * tg + 0.114 * tb
        text_chroma = max(tr, tg, tb) - min(tr, tg, tb)
        contrast_ratio = abs(text_luma - mean_brightness)
        if contrast_ratio >= 40:  # Minimum perceptual contrast
            if not (text_style == "dialogue" and text_chroma > 45 and text_luma > 55):
                render_color = text_color

    # Auto-size font to fit
    max_size_cap = None
    if non_bubble_dialogue:
        max_size_cap = max(14, min(24, int(min(avail_w, avail_h) * 0.42)))
    font, lines, line_spacing = _fit_text_to_box(
        draw=draw,
        text=text,
        max_w=avail_w,
        max_h=avail_h,
        font_path=font_file,
        style=text_style,
        min_size=10 if text_style == "sfx" else 12,
        max_size_cap=max_size_cap,
    )
    stroke_width = _stroke_width_for_font(font)
    if text_style == "dialogue" and not lines:
        rescue = _fit_narrow_dialogue_rescue(
            draw=draw,
            text=text,
            max_w=avail_w,
            max_h=avail_h,
            font_path=font_file,
            max_size_cap=max_size_cap,
        )
        if rescue is not None:
            text, font, lines, line_spacing = rescue
            stroke_width = _stroke_width_for_font(font)
            logger.debug(
                "Dialogue constraints: using narrow rescue text='%s' size=%d lines=%d",
                text,
                getattr(font, "size", 0),
                len(lines),
            )
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
            return _skip_render("non-bubble sfx too wide")

    # Enforce minimum readable font size (12px) AND a dynamic max line count for dialogue.
    # Tall narrow bubbles need more lines; cap based on available height.
    _DIALOGUE_MIN_SIZE = 12
    if box_h > box_w * 3:
        _DIALOGUE_MAX_LINES = 10
    elif box_h > box_w * 2:
        _DIALOGUE_MAX_LINES = 8
    else:
        _DIALOGUE_MAX_LINES = 6
    if text_style == "dialogue" and lines:
        current_size = getattr(font, "size", 99)
        current_lines = len(lines)
        needs_fix = current_size < _DIALOGUE_MIN_SIZE or current_lines > _DIALOGUE_MAX_LINES
        if needs_fix:
            # First: try fitting the full text at exactly min_size — no truncation yet.
            font_full, lines_full, spacing_full = _fit_text_to_box(
                draw=draw,
                text=text,
                max_w=avail_w,
                max_h=avail_h,
                font_path=font_file,
                style=text_style,
                min_size=_DIALOGUE_MIN_SIZE,
                max_size=_DIALOGUE_MIN_SIZE,
            )
            if lines_full and getattr(font_full, "size", 0) >= _DIALOGUE_MIN_SIZE:
                # Full text fits at min readable size — use it, skip truncation entirely.
                font, lines, line_spacing = font_full, lines_full, spacing_full
                stroke_width = _stroke_width_for_font(font)
                logger.debug(
                    "Dialogue constraints: full text fits at min size=%d, lines=%d (no truncation)",
                    _DIALOGUE_MIN_SIZE, len(lines_full),
                )
                needs_fix = False
            if needs_fix:
                # Full text doesn't fit at min size — fall back to word truncation.
                words = text.split()
                fixed = False
                seen_candidates: set[str] = set()
                best_candidate: str | None = None
                best_font2 = None
                best_lines2: list[str] | None = None
                best_spacing2 = None
                for max_w in range(len(words) - 1, 0, -1):
                    candidate = _compress_dialogue_for_tiny_box(text, max_words=max_w)
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    font2, lines2, spacing2 = _fit_text_to_box(
                        draw=draw,
                        text=candidate,
                        max_w=avail_w,
                        max_h=avail_h,
                        font_path=font_file,
                        style=text_style,
                        min_size=_DIALOGUE_MIN_SIZE,
                    )
                    if not lines2:
                        continue
                    size2 = getattr(font2, "size", 0)
                    # Track the best candidate seen so far (largest font that fits line count)
                    if best_lines2 is None or (
                        len(lines2) <= _DIALOGUE_MAX_LINES
                        and (best_lines2 is None or size2 > getattr(best_font2, "size", 0))
                    ):
                        best_candidate = candidate
                        best_font2 = font2
                        best_lines2 = lines2
                        best_spacing2 = spacing2
                    if size2 >= _DIALOGUE_MIN_SIZE and len(lines2) <= _DIALOGUE_MAX_LINES:
                        font, lines, line_spacing, text = font2, lines2, spacing2, candidate
                        stroke_width = _stroke_width_for_font(font)
                        logger.debug(
                            "Dialogue constraints: truncated to %d words, size=%d lines=%d",
                            max_w, font2.size, len(lines2),
                        )
                        fixed = True
                        break
                if not fixed:
                    # Use the best candidate found even if it doesn't perfectly satisfy both
                    # constraints — better to render something than leave the bubble empty.
                    if best_candidate and best_lines2 and best_font2:
                        font, lines, line_spacing, text = best_font2, best_lines2, best_spacing2, best_candidate
                        stroke_width = _stroke_width_for_font(font)
                        logger.debug(
                            "Dialogue constraints: using best-effort candidate, size=%d lines=%d",
                            getattr(font, "size", 0), len(lines),
                        )

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
        return _skip_render("no fitted lines")
    if _is_overbroken_sfx(text=text, text_style=text_style, lines=lines, font=font):
        return _skip_render("overbroken sfx")
    if _should_skip_render_text(
        text=text,
        box_w=box_w,
        box_h=box_h,
        text_style=text_style,
        bubble_used=bubble_used,
    ):
        return _skip_render("tiny/noise text")

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

    if (
        text_style == "dialogue"
        and box_clip_mask is None
        and line_metrics
        and max(lw for lw, _, _, _ in line_metrics) > avail_w + 2
    ):
        return _skip_render("dialogue line wider than unmasked box")

    total_text_height = (
        sum(lh for _, lh, _, _ in line_metrics) + (len(lines) - 1) * line_spacing
    )

    # Center vertically
    start_y = box_y + text_padding + max(0, (avail_h - total_text_height) // 2)
    x_anchor = box_x + text_padding

    # Bubble masks can be narrower near the top/bottom. Prefer a layout that is
    # proven to fit every line into the usable mask band instead of only fitting
    # the rectangular bounding box.
    if box_clip_mask is not None and text_style == "dialogue":
        mask_safe_layout = _fit_dialogue_to_bubble_mask(
            draw=draw,
            text=text,
            font_path=font_file,
            avail_w=avail_w,
            avail_h=avail_h,
            mask=box_clip_mask,
            text_padding=text_padding,
            min_size=_DIALOGUE_MIN_SIZE,
            max_size=getattr(font, "size", _DIALOGUE_MIN_SIZE),
            max_lines=_DIALOGUE_MAX_LINES,
        )
        if mask_safe_layout is not None:
            font, lines, line_spacing, line_metrics, local_start_y = mask_safe_layout
            stroke_width = _stroke_width_for_font(font)
            total_text_height = (
                sum(lh for _, lh, _, _ in line_metrics)
                + (len(lines) - 1) * line_spacing
            )
            start_y = box_y + local_start_y

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
                        min_size=_DIALOGUE_MIN_SIZE,
                    )
                    if not lines2:
                        break
                    if len(lines2) > _DIALOGUE_MAX_LINES or getattr(font2, "size", 0) < _DIALOGUE_MIN_SIZE:
                        break
                    font, lines, line_spacing = font2, lines2, line_spacing2
                    stroke_width = _stroke_width_for_font(font)

                    line_metrics.clear()
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

        if not _layout_fits_bubble_mask(
            mask=box_clip_mask,
            start_y=int(start_y - box_y),
            line_metrics=line_metrics,
            line_spacing=line_spacing,
            margin=3,
        ):
            shortened_layout = _fit_shortened_dialogue_to_bubble_mask(
                draw=draw,
                text=text,
                font_path=font_file,
                avail_w=avail_w,
                avail_h=avail_h,
                mask=box_clip_mask,
                text_padding=text_padding,
                min_size=_DIALOGUE_MIN_SIZE,
                max_size=getattr(font, "size", _DIALOGUE_MIN_SIZE),
                max_lines=_DIALOGUE_MAX_LINES,
            )
            if shortened_layout is None:
                relaxed_layout = _fit_relaxed_spill_dialogue_to_bubble_mask(
                    draw=draw,
                    text=text,
                    font_path=font_file,
                    avail_w=avail_w,
                    avail_h=avail_h,
                    mask=box_clip_mask,
                    text_padding=text_padding,
                    min_size=_DIALOGUE_MIN_SIZE,
                    max_size=getattr(font, "size", _DIALOGUE_MIN_SIZE),
                    max_lines=_DIALOGUE_MAX_LINES,
                )
                if relaxed_layout is None:
                    return _skip_render("dialogue does not fit bubble mask")
                (
                    font,
                    lines,
                    line_spacing,
                    line_metrics,
                    local_start_y,
                    text,
                    box_clip_mask,
                ) = relaxed_layout
            else:
                font, lines, line_spacing, line_metrics, local_start_y, text = shortened_layout
            stroke_width = _stroke_width_for_font(font)
            start_y = box_y + local_start_y

    # Draw each line centered horizontally
    clip_dialogue_to_mask = box_clip_mask is not None and text_style == "dialogue"
    if clip_dialogue_to_mask:
        text_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        target_draw = ImageDraw.Draw(text_layer)
        fill_color = (*render_color, 255)
        stroke_color = (*outline_color, 255)
    else:
        text_layer = None
        target_draw = draw
        fill_color = render_color
        stroke_color = outline_color

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
        target_draw.text(
            (int(line_x - left), int(current_y - top)),
            line,
            font=font,
            anchor="lt",
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )

        current_y += lh + line_spacing

    if clip_dialogue_to_mask and text_layer is not None:
        alpha = np.array(text_layer.getchannel("A"), dtype=np.uint8)
        clip = np.zeros_like(alpha)
        clip[box_y : box_y + box_h, box_x : box_x + box_w] = (
            (box_clip_mask > 0).astype(np.uint8) * 255
        )
        alpha = np.minimum(alpha, clip)
        text_layer.putalpha(Image.fromarray(alpha))
        pil_image = Image.alpha_composite(pil_image.convert("RGBA"), text_layer).convert("RGB")

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
    inset = max(2, int(min(w, h) * 0.05))
    default_box = (
        x + inset,
        y + inset,
        max(1, w - 2 * inset),
        max(1, h - 2 * inset),
    )
    fallback_box = _dialogue_fallback_box_from_mask(
        x=x,
        y=y,
        w=w,
        h=h,
        region_mask=region_mask,
        default_box=default_box,
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
            # Reject boxes that are suspiciously narrow ONLY when the region is
            # wide enough that a narrow result likely means Otsu cut off the bubble.
            # For genuinely narrow bubbles in narrow regions, keep the estimate.
            if bw < int(w * 0.40) and bh >= int(h * 0.70) and w >= 120:
                logger.debug(
                    "Rejecting narrow estimated bubble box %dx%d for region %dx%d",
                    bw, bh, w, h,
                )
                continue
            bubble_mask = _surface_mask_from_estimated_bubble_box(
                region_image=region_image,
                bubble_box=(bx, by, bw, bh),
                translated_text=translated_text,
            )
            if bubble_mask is None:
                continue
            break
        if bubble_mask is None:
            return (*fallback_box, False, None)

    bubble_mask = _isolate_bubble_lobe_near_text(
        bubble_mask=bubble_mask,
        region_mask=region_mask,
    )
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
    if bubble_area > int(region_area * 0.99):
        return (*fallback_box, False, None)
    if not _bubble_mask_is_plausible_for_text(
        bubble_mask=bubble_mask,
        region_mask=region_mask,
    ):
        return (*fallback_box, False, None)
    if not _bubble_mask_has_paper_surface(
        region_image=region_image,
        bubble_mask=bubble_mask,
    ):
        return (*fallback_box, False, None)

    bubble_render_mask = _prepare_bubble_render_mask(bubble_mask)
    if bubble_render_mask is None:
        return (*fallback_box, False, None)

    fit_bbox = _mask_bbox(bubble_render_mask)
    if fit_bbox is None:
        return (*fallback_box, False, None)
    x1, y1, x2, y2 = fit_bbox
    # The mask is already eroded; we don't need additional aggressive shrinkage.
    shrink_ratio = 0.00
    x1, y1, x2, y2 = _shrink_centered_box(x1, y1, x2, y2, shrink_ratio)
    fit_w = x2 - x1
    fit_h = y2 - y1
    min_w = max(20, int(w * 0.20))
    min_h = max(18, int(h * 0.20))
    if fit_w < min_w or fit_h < min_h:
        return (*fallback_box, False, None)

    # For longer dialogue, avoid overly tight boxes.
    if len(translated_text.strip()) >= 18:
        min_long_w_ratio = 0.70 if w < 120 else 0.50
        min_long_w = max(min_w, int(w * min_long_w_ratio))
        min_long_h = max(min_h, int(h * 0.38))
        if fit_w < min_long_w or fit_h < min_long_h:
            return (*fallback_box, False, None)

    return (x + x1, y + y1, fit_w, fit_h, True, bubble_render_mask)


def _dialogue_fallback_box_from_mask(
    x: int,
    y: int,
    w: int,
    h: int,
    region_mask: NDArray | None,
    default_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Use a local glyph cluster fallback instead of a whole panel-sized box."""
    if region_mask is None or region_mask.shape[:2] != (h, w):
        return default_box

    if region_mask.max() > 1:
        _, binary = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (region_mask > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary) < 20:
        return default_box

    bbox = _mask_bbox(binary)
    if bbox is None:
        return default_box

    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1
    region_area = max(1, w * h)
    bbox_area = max(1, bw * bh)

    # If the glyph mask already occupies nearly the whole context, keep the
    # context. Otherwise stay close to the original glyph cluster; this prevents
    # missed bubble extraction from using an over-large lettering box.
    if bbox_area > int(region_area * 0.75):
        return default_box

    # Keep missed-bubble fallback close to the original glyph cluster.  The
    # previous expansion could turn a narrow vertical Japanese column into a
    # panel-wide English text lane when balloon extraction failed.
    expand_x = max(8, int(bw * 0.38), int(min(w, h) * 0.03))
    expand_y = max(8, int(bh * 0.36), int(min(w, h) * 0.03))
    if bh > bw * 1.35:
        expand_x = max(expand_x, min(58, int(bh * 0.14)))
    if bw > bh * 1.8:
        expand_y = max(expand_y, min(52, int(bw * 0.12)))

    lx1 = max(0, bx1 - expand_x)
    ly1 = max(0, by1 - expand_y)
    lx2 = min(w, bx2 + expand_x)
    ly2 = min(h, by2 + expand_y)

    max_fallback_w = max(48, int(bw * (1.42 if bh > bw * 1.20 else 1.70)))
    max_fallback_h = max(42, int(bh * (1.38 if bh > bw * 1.20 else 1.62)))
    lx1, lx2 = _cap_span_around_center(lx1, lx2, max_fallback_w, 0, w)
    ly1, ly2 = _cap_span_around_center(ly1, ly2, max_fallback_h, 0, h)

    fw = lx2 - lx1
    fh = ly2 - ly1
    if fw < 18 or fh < 18:
        return default_box

    return (x + lx1, y + ly1, fw, fh)


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
    # Reject near-full fills, but be less aggressive for narrow regions that
    # are genuinely mostly bubble interior.
    area_ratio = comp_area / float(max(1, h * w))
    if area_ratio > 0.98:
        return None
    if area_ratio > 0.94 and min(h, w) < 80:
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

    k = _odd(max(3, int(round(np.sqrt(comp_area) / 70))))
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


def _surface_mask_from_estimated_bubble_box(
    region_image: NDArray,
    bubble_box: tuple[int, int, int, int],
    translated_text: str,
) -> NDArray | None:
    """Create a real white-surface mask from an estimated bubble rectangle."""
    if region_image.size == 0:
        return None

    region_h, region_w = region_image.shape[:2]
    bx, by, bw, bh = bubble_box
    bx = int(np.clip(bx, 0, max(0, region_w - 1)))
    by = int(np.clip(by, 0, max(0, region_h - 1)))
    bw = int(max(1, min(bw, region_w - bx)))
    bh = int(max(1, min(bh, region_h - by)))
    crop = region_image[by : by + bh, bx : bx + bw]
    if crop.size == 0:
        return None

    surface = _non_bubble_dialogue_surface(
        region_image=crop,
        text=translated_text,
        box_w=bw,
        box_h=bh,
    )
    if surface is None:
        return None

    sx, sy, sw, sh, local_mask = surface
    mask = np.zeros((region_h, region_w), dtype=np.uint8)
    mask[by + sy : by + sy + sh, bx + sx : bx + sx + sw] = local_mask
    if cv2.countNonZero(mask) < 120:
        return None
    return mask


def _isolate_bubble_lobe_near_text(
    bubble_mask: NDArray,
    region_mask: NDArray | None,
) -> NDArray | None:
    """Keep the speech-bubble lobe nearest the OCR text cluster.

    Adjacent balloons can be connected by a tiny white bridge or by a broad
    detector context. A light erosion often separates those lobes; choosing the
    one anchored to the OCR mask prevents rendering one translation across two
    neighbouring bubbles.
    """
    if bubble_mask.size == 0:
        return None

    bubble = (bubble_mask > 0).astype(np.uint8) * 255
    bubble_bbox = _mask_bbox(bubble)
    if bubble_bbox is None:
        return None

    text_info = _text_mask_geometry(region_mask, bubble.shape[1], bubble.shape[0])
    if text_info is None:
        return bubble
    text_bbox, text_centroid = text_info

    comp_area = cv2.countNonZero(bubble)
    if comp_area < 300:
        return bubble

    # Stronger than the final render-mask erosion: this is only used to reveal
    # whether the accepted white area is really multiple adjacent lobes.
    k = _odd(max(5, int(round(np.sqrt(comp_area) / 34))))
    eroded = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    if cv2.countNonZero(eroded) < max(140, int(comp_area * 0.18)):
        return bubble

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        eroded,
        connectivity=8,
    )
    components: list[tuple[int, float]] = []
    tcx, tcy = text_centroid
    tx1, ty1, tx2, ty2 = text_bbox
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(80, int(comp_area * 0.035)):
            continue

        cx = x + w * 0.5
        cy = y + h * 0.5
        overlap_w = max(0, min(tx2, x + w) - max(tx1, x))
        overlap_h = max(0, min(ty2, y + h) - max(ty1, y))
        overlap = overlap_w * overlap_h
        inside = x <= tcx <= x + w and y <= tcy <= y + h
        dist = abs(cx - tcx) + abs(cy - tcy)
        score = overlap * 8.0 + area * 0.05 - dist * 2.0
        if inside:
            score += 2000.0
        components.append((label, score))

    if len(components) < 2:
        return bubble

    best_label = max(components, key=lambda item: item[1])[0]
    selected = (labels == best_label).astype(np.uint8) * 255
    selected = cv2.dilate(
        selected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    selected = cv2.bitwise_and(selected, bubble)
    if cv2.countNonZero(selected) < max(160, int(comp_area * 0.12)):
        return bubble

    return selected


def _bubble_mask_is_plausible_for_text(
    bubble_mask: NDArray,
    region_mask: NDArray | None,
) -> bool:
    """Check that a bubble candidate is anchored to the OCR mask."""
    h, w = bubble_mask.shape[:2]
    text_info = _text_mask_geometry(region_mask, w, h)
    if text_info is None:
        return True

    text_bbox, text_centroid = text_info
    bubble_bbox = _mask_bbox(bubble_mask)
    if bubble_bbox is None:
        return False

    tx1, ty1, tx2, ty2 = text_bbox
    bx1, by1, bx2, by2 = bubble_bbox
    tcx, tcy = text_centroid
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    tw = max(1, tx2 - tx1)
    th = max(1, ty2 - ty1)

    if not (bx1 - 6 <= tcx <= bx2 + 6 and by1 - 6 <= tcy <= by2 + 6):
        return False

    overlap_w = max(0, min(tx2, bx2) - max(tx1, bx1))
    overlap_h = max(0, min(ty2, by2) - max(ty1, by1))
    if (overlap_w * overlap_h) < max(24, int(tw * th * 0.18)):
        return False

    # If the text cluster is pushed far toward one edge of a huge accepted
    # bubble, it is usually because two adjacent balloons were merged. Keep this
    # relaxed for tall/narrow vertical dialogue where the source text can sit a
    # little high or low.
    rel_x = (tcx - bx1) / float(bw)
    rel_y = (tcy - by1) / float(bh)
    if bw >= max(150, tw * 4) and (rel_x < 0.18 or rel_x > 0.82):
        return False
    if bh >= max(150, th * 4) and (rel_y < 0.12 or rel_y > 0.88):
        return False

    return True


def _bubble_mask_has_paper_surface(
    region_image: NDArray,
    bubble_mask: NDArray,
) -> bool:
    """Return True when the accepted bubble mask is mostly paper-like."""
    if region_image.size == 0 or bubble_mask.size == 0:
        return False
    if bubble_mask.shape[:2] != region_image.shape[:2]:
        return False

    mask = bubble_mask > 0
    if int(np.sum(mask)) < 120:
        return False

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
    gray_vals = gray[mask]
    sat_vals = hsv[:, :, 1][mask]
    if gray_vals.size == 0:
        return False

    paper = (gray_vals > 190) & (sat_vals < 65)
    paper_ratio = float(np.mean(paper))
    mean_sat = float(np.mean(sat_vals))
    mean_gray = float(np.mean(gray_vals))

    bbox = _mask_bbox(bubble_mask)
    if bbox is None:
        return False
    bx1, by1, bx2, by2 = bbox
    bh, bw = bubble_mask.shape[:2]
    bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
    edge_touches = int(bx1 <= 1) + int(by1 <= 1)
    edge_touches += int(bx2 >= bw - 1) + int(by2 >= bh - 1)
    if (
        edge_touches >= 2
        and bbox_area > int(bw * bh * 0.45)
        and mean_gray < 232
    ):
        return False

    if paper_ratio >= 0.38:
        return True
    if paper_ratio >= 0.26 and mean_gray >= 185 and mean_sat < 52:
        return True

    return False


def _text_mask_geometry(
    region_mask: NDArray | None,
    w: int,
    h: int,
) -> tuple[tuple[int, int, int, int], tuple[float, float]] | None:
    """Return OCR mask bbox and centroid in local region coordinates."""
    if region_mask is None or region_mask.shape[:2] != (h, w):
        return None

    if region_mask.max() > 1:
        _, binary = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (region_mask > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary) < 20:
        return None

    bbox = _mask_bbox(binary)
    if bbox is None:
        return None

    ys, xs = np.where(binary > 0)
    if xs.size < 20:
        return None

    return bbox, (float(np.mean(xs)), float(np.mean(ys)))


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
    """Estimate speech-bubble bounds from bright connected components.

    Uses a fixed low threshold (120) instead of Otsu to avoid the common failure
    where Otsu picks a threshold above the bubble interior brightness (~170),
    splitting a valid bubble into a narrow partial component.
    """
    if region_image.size == 0:
        return None

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    region_h, region_w = gray.shape[:2]
    ax = int(np.clip(anchor[0], 0, max(0, region_w - 1)))
    ay = int(np.clip(anchor[1], 0, max(0, region_h - 1)))

    # Use a fixed low threshold so typical bubble interiors (brightness ~150-220)
    # are captured as bright. Otsu often picks ~180 which cuts off valid bubbles.
    # Try multiple thresholds and pick the one whose component best covers the anchor.
    best_box: tuple[int, int, int, int] | None = None
    best_score = -1e9

    for thresh in (100, 130, 160, 190, 220):
        _, bright = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

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
                    and (w * h) < int(region_w * region_h * 0.98)
                ):
                    # Prefer boxes that cover more of the region width/height
                    coverage = (w / region_w) + (h / region_h)
                    if coverage > best_score:
                        best_score = coverage
                        best_box = (int(x), int(y), int(w), int(h))
                    break  # anchor is inside — no need to try other thresholds

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 700:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 26 or h < 22:
                continue

            # Ignore near-full panel regions.
            if (w * h) > int(region_w * region_h * 0.98):
                continue

            inside = cv2.pointPolygonTest(contour, (float(ax), float(ay)), False) >= 0
            center_dist = abs((x + w // 2) - ax) + abs((y + h // 2) - ay)
            coverage = (w / region_w) + (h / region_h)
            score = coverage * 1000 - center_dist * 0.5
            if inside:
                score += 5000

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
    min_size: int = 10,
    max_size: int = 200,
    max_size_cap: int | None = None,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
    """Find the largest font size where the wrapped text fits in the box."""

    # Dynamically compute max font size from box dimensions so text fills
    # the available space.  Let the binary search find the largest size that
    # fits — only cap to prevent absurdly large single-word renders.
    word_count = len(text.split())
    if style == "dialogue":
        if word_count <= 2:
            dynamic_max = min(200, max(max_h, int(max_w * 1.5)))
        elif word_count <= 5:
            dynamic_max = min(200, max(max_h, int(max_w * 1.4)))
        else:
            dynamic_max = min(200, max(24, int(min(max_h, max_w) * 1.5)))
        if word_count >= 9:
            dynamic_max = min(dynamic_max, max(20, min(28, int(max_w * 0.20))))
        elif word_count >= 6:
            dynamic_max = min(dynamic_max, max(20, min(30, int(max_w * 0.22))))
        elif word_count >= 4:
            dynamic_max = min(dynamic_max, max(20, min(34, int(max_w * 0.28))))
        max_size = min(max_size, dynamic_max)
    else:
        # SFX: can go large, scale with box dimensions
        dynamic_max = min(200, max(max_h, max_w))
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
        allow_char_wrap: bool = False,
        width_slack: int = 0,
    ) -> tuple[
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
        list[str],
        int,
        int,
        int,
    ]:
        stroke_width = max(1, size // 12)
        font = _load_font(size)
        wrap_w = max_w + max(0, int(width_slack))
        lines = _wrap_text(
            draw,
            text,
            font,
            wrap_w,
            allow_char_wrap=allow_char_wrap,
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

    # Binary search WITHOUT hyphenation. Dialogue gets a tiny slack allowance
    # so one slightly wide word does not become an ugly hyphenated split.
    no_hyphen_width_slack = 1 if style == "dialogue" else 0
    lo, hi = min_size, max_size
    best_no_hyphen: tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font, lines, spacing, used_w, used_h = _layout_at_size(
            mid,
            allow_hyphenation=False,
            width_slack=no_hyphen_width_slack,
        )
        if lines and used_w <= max_w + no_hyphen_width_slack and used_h <= max_h:
            best_no_hyphen = (font, lines, spacing)
            lo = mid + 1
        else:
            hi = mid - 1

    best_hyphen: tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int] | None = None
    if style != "dialogue":
        # Binary search WITH hyphenation. Dialogue deliberately avoids inserted
        # hyphen line breaks; those are visually noisy in manga bubbles.
        lo, hi = min_size, max_size
        while lo <= hi:
            mid = (lo + hi) // 2
            font, lines, spacing, used_w, used_h = _layout_at_size(
                mid,
                allow_hyphenation=True,
            )
            if lines and used_w <= max_w and used_h <= max_h:
                best_hyphen = (font, lines, spacing)
                lo = mid + 1
            else:
                hi = mid - 1

    # Prefer natural word-boundary wrapping. Hyphenation is useful as a rescue
    # path for genuinely tight boxes, but choosing it for every small font-size
    # gain makes dialogue choppy and hard to read.
    best = best_no_hyphen
    if best_hyphen is not None:
        if best is None:
            best = best_hyphen
        else:
            size_no_hyphen = getattr(best_no_hyphen[0], "size", 0)
            size_hyphen = getattr(best_hyphen[0], "size", 0)
            if _should_prefer_hyphenated_layout(
                no_hyphen_size=size_no_hyphen,
                hyphen_size=size_hyphen,
                hyphen_lines=best_hyphen[1],
            ):
                best = best_hyphen

    if best is not None:
        return best

    # Fallback to minimum size with character wrapping for non-dialogue only.
    # Dialogue should either fit at word boundaries or let the caller shrink,
    # truncate, or skip it; splitting English words is too distracting in bubbles.
    if style != "dialogue":
        font, lines, spacing, _, _ = _layout_at_size(
            min_size,
            allow_hyphenation=False,
            allow_char_wrap=True,
            width_slack=no_hyphen_width_slack,
        )
        if lines:
            return font, lines, spacing

    if style == "dialogue":
        font, lines, spacing, _, _ = _layout_at_size(
            min_size,
            allow_hyphenation=False,
            allow_char_wrap=False,
            width_slack=no_hyphen_width_slack,
        )
        return font, lines, spacing

    # Absolute last resort: allow both hyphenation and character wrapping for
    # non-dialogue only.
    font, lines, spacing, _, _ = _layout_at_size(
        min_size,
        allow_hyphenation=True,
        allow_char_wrap=True,
    )
    return font, lines, spacing


def _fit_narrow_dialogue_rescue(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    font_path: str | None,
    max_size_cap: int | None,
) -> tuple[
    str,
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    list[str],
    int,
] | None:
    """Fit otherwise-blank dialogue using shorter word-boundary candidates."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return None

    if max_h > max_w * 3:
        max_lines = 10
    elif max_h > max_w * 2:
        max_lines = 8
    else:
        max_lines = 6

    best: tuple[
        float,
        str,
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
        list[str],
        int,
    ] | None = None
    for order, candidate in enumerate(_narrow_dialogue_rescue_candidates(clean)):
        if _is_punctuation_only(candidate):
            continue
        font, lines, spacing = _fit_text_to_box(
            draw=draw,
            text=candidate,
            max_w=max_w,
            max_h=max_h,
            font_path=font_path,
            style="dialogue",
            min_size=12,
            max_size_cap=max_size_cap,
        )
        if not lines or len(lines) > max_lines:
            continue
        stroke_width = _stroke_width_for_font(font)
        if _max_line_width(draw, lines, font, stroke_width=stroke_width) > max_w + 1:
            continue
        total_h = _text_layout_height(
            draw=draw,
            lines=lines,
            font=font,
            line_spacing=spacing,
            stroke_width=stroke_width,
        )
        if total_h > max_h:
            continue

        word_score = len(re.findall(r"[A-Za-z0-9']+", candidate))
        size_score = getattr(font, "size", 0)
        score = word_score * 100.0 + size_score * 2.0 - len(lines) * 3.0 - order * 0.1
        if best is None or score > best[0]:
            best = (score, candidate, font, lines, spacing)

    if best is None:
        return None

    _, candidate, font, lines, spacing = best
    return candidate, font, lines, spacing


def _narrow_dialogue_rescue_candidates(text: str) -> list[str]:
    """Generate short dialogue variants without breaking words or adding hyphens."""
    normalized = re.sub(r"\b([A-Z])-([A-Z][A-Z]+)", r"\2", text)
    normalized = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", normalized)
    normalized = " ".join(normalized.split())
    words = normalized.split()
    if not words:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split()).strip()
        candidate = candidate.strip(" ,;:")
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    add(normalized)

    max_words = min(8, len(words))
    for count in range(max_words, 0, -1):
        add(_truncate_dialogue_at_word_boundary(normalized, max_words=count))
        head = " ".join(words[:count]).rstrip(".,;:!?")
        if count < len(words):
            head = f"{head}..."
        add(head)
        add(" ".join(words[-count:]))

    # Prefer meaningful short words over rendering nothing when the first word
    # is the one that makes the layout impossible, e.g. "ESPECIALLY".
    shortish_words = [
        word
        for word in words
        if len(word.strip(".,;:!?\"'")) <= 8
        and any(ch.isalnum() for ch in word)
    ]
    for count in range(min(6, len(shortish_words)), 0, -1):
        add(" ".join(shortish_words[-count:]))

    return candidates


def _should_prefer_hyphenated_layout(
    *,
    no_hyphen_size: int,
    hyphen_size: int,
    hyphen_lines: list[str],
) -> bool:
    """Return True when hyphenation buys enough readability to be worth it."""
    if hyphen_size <= no_hyphen_size:
        return False

    inserted_breaks = _hyphenated_line_break_count(hyphen_lines)
    if inserted_breaks == 0:
        return True

    return inserted_breaks <= 1 and hyphen_size >= no_hyphen_size + 4


def _hyphenated_line_break_count(lines: list[str]) -> int:
    """Count likely renderer-inserted word breaks across adjacent lines."""
    count = 0
    for current, following in zip(lines, lines[1:]):
        left = current.rstrip()
        right = following.lstrip()
        if not left.endswith("-") or not right:
            continue
        if re.search(r"[A-Za-z]-$", left) and right[0].isalpha():
            count += 1
    return count


def _stroke_width_for_font(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """Match the outline thickness used at render time."""
    font_size = getattr(font, "size", 12)
    return max(1, font_size // 12)


def _line_spacing_for_size(size: int, style: str) -> int:
    """Compute line spacing as a function of font size and text style."""
    if style == "sfx":
        return max(2, int(size * 0.12))
    # Comic fonts have generous ascenders/descenders and strokes can bleed
    # 1-2px beyond textbbox. Use spacing that guarantees no overlap.
    return max(4, int(size * 0.25))


def _compress_dialogue_for_tiny_box(text: str, max_words: int) -> str:
    """Shorten long dialogue for very small render boxes.

    Extends past max_words to find the next natural sentence boundary
    (., !, ?, ...) so the result is always a complete sentence/phrase.
    Falls back to word-boundary truncation with ellipsis only when no
    sentence end exists in the remaining words.
    """
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean

    words = clean.split(" ")
    if len(words) <= max_words:
        return clean

    _SENTENCE_END = re.compile(r"[.!?]$|^\.\.\.$")

    # Start from max_words and scan forward for the next sentence boundary.
    for i in range(max_words - 1, len(words)):
        if _SENTENCE_END.search(words[i].rstrip()):
            return " ".join(words[: i + 1])

    # No sentence boundary found — cut at max_words with ellipsis.
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
    """Use SFX style for very short non-bubble snippets.

    Does NOT force SFX when the text has sentence punctuation (it's dialogue)
    or when the box is tall relative to its width (vertical dialogue column).
    """
    clean = " ".join(text.split())
    words = clean.split(" ") if clean else []
    if len(words) > 2:
        return False
    # Keep as dialogue if it has sentence-ending punctuation
    if any(c in clean for c in ".?!"):
        return False
    # Keep as dialogue if the box is a tall column (vertical speech bubble)
    if box_h > box_w * 1.5:
        return False
    return True


def _non_bubble_dialogue_has_safe_surface(
    region_image: NDArray,
    text: str,
    box_w: int,
    box_h: int,
) -> bool:
    """Allow no-bubble dialogue only on a likely white lettering surface."""
    clean = " ".join(text.split())
    words = re.findall(r"[A-Za-z']+", clean)
    if len(words) <= 2 and len(clean) <= 18:
        return True
    if region_image is None or region_image.size == 0:
        return False
    if box_w < 18 or box_h < 18:
        return False

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
    bright = gray > 200
    low_sat = hsv[:, :, 1] < 45
    paper = bright & low_sat
    paper_ratio = float(np.mean(paper)) if paper.size else 0.0
    mean_brightness = float(np.mean(gray)) if gray.size else 0.0

    # White rectangular narration boxes and missed speech balloons both have a
    # substantial low-saturation bright surface. Art/background crops do not.
    if paper_ratio >= 0.36:
        return True
    if paper_ratio >= 0.24 and mean_brightness >= 190 and len(words) <= 5:
        return True

    return False


def _non_bubble_dialogue_surface(
    region_image: NDArray,
    text: str,
    box_w: int,
    box_h: int,
) -> tuple[int, int, int, int, NDArray] | None:
    """Find a bright local paper surface for fallback dialogue rendering."""
    if region_image is None or region_image.size == 0:
        return None

    h, w = region_image.shape[:2]
    if h < 18 or w < 18:
        return None

    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
    clean = " ".join(text.split())
    word_count = len(re.findall(r"[A-Za-z']+", clean))
    crop_area = max(1, w * h)

    high_confidence_surface = _bright_rect_dialogue_surface(
        gray=gray,
        hsv=hsv,
        crop_area=crop_area,
        word_count=word_count,
    )
    if high_confidence_surface is not None:
        return high_confidence_surface

    paper = ((gray > 200) & (hsv[:, :, 1] < 48)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        paper,
        connectivity=8,
    )
    if num_labels <= 1:
        return None

    cx = w * 0.5
    cy = h * 0.5
    best_label = -1
    best_score = -1e9
    best_edge_large = False
    best_mean_gray = 0.0
    best_area = 0
    best_bbox_area = 0
    best_fill_ratio = 0.0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_area = max(1, cw * ch)
        fill_ratio = area / float(bbox_area)
        if area < max(180, int(crop_area * 0.055)):
            continue
        if cw < 18 or ch < 18:
            continue
        if fill_ratio < 0.34:
            continue
        edge_touches = int(x <= 1) + int(y <= 1)
        edge_touches += int((x + cw) >= w - 1)
        edge_touches += int((y + ch) >= h - 1)
        edge_large = edge_touches >= 2 and bbox_area > int(crop_area * 0.45)
        comp_mask = labels == label
        comp_mean_gray = float(np.mean(gray[comp_mask])) if np.any(comp_mask) else 0.0
        comp_cx = x + cw * 0.5
        comp_cy = y + ch * 0.5
        center_dist = abs(comp_cx - cx) + abs(comp_cy - cy)
        score = area * 1.4 + bbox_area * 0.15 - center_dist * 3.0
        if edge_large:
            # Pale skin/walls often pass the low-saturation paper test. True
            # narration boxes are usually near-white, so only heavily penalize
            # large edge-touching surfaces that are not very bright.
            if comp_mean_gray < 232:
                score -= area * 4.0
            else:
                score -= area * 0.35
        if score > best_score:
            best_score = score
            best_label = label
            best_edge_large = edge_large
            best_mean_gray = comp_mean_gray
            best_area = area
            best_bbox_area = bbox_area
            best_fill_ratio = fill_ratio

    if best_label < 0:
        return None
    if best_edge_large and best_mean_gray < 232:
        return None
    if word_count > 2:
        area_ratio = best_area / float(crop_area)
        bbox_ratio = best_bbox_area / float(crop_area)
        high_confidence_jagged_paper = (
            best_mean_gray >= 245
            and best_fill_ratio >= 0.48
            and (area_ratio >= 0.075 or bbox_ratio >= 0.14)
        )
        if (
            area_ratio < 0.11
            and bbox_ratio < 0.18
            and not high_confidence_jagged_paper
        ):
            return None

    component = (labels == best_label).astype(np.uint8) * 255
    bbox = _mask_bbox(component)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 18 or bh < 18:
        return None

    local_mask = component[y1:y2, x1:x2].copy()
    if cv2.countNonZero(local_mask) < max(120, int(bw * bh * 0.25)):
        return None

    return x1, y1, bw, bh, local_mask


def _bright_rect_dialogue_surface(
    *,
    gray: NDArray,
    hsv: NDArray,
    crop_area: int,
    word_count: int,
) -> tuple[int, int, int, int, NDArray] | None:
    """Find a clean white narration/bubble surface before broader paper fill."""
    h, w = gray.shape[:2]
    bright = ((gray > 225) & (hsv[:, :, 1] < 45)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )
    if num_labels <= 1:
        return None

    cx = w * 0.5
    cy = h * 0.5
    best_label = -1
    best_score = -1e9
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_area = max(1, cw * ch)
        if area < max(240, int(crop_area * 0.035)):
            continue
        if cw < 28 or ch < 28:
            continue
        fill_ratio = area / float(bbox_area)
        if fill_ratio < 0.78:
            continue
        comp_mask = labels == label
        comp_mean_gray = float(np.mean(gray[comp_mask])) if np.any(comp_mask) else 0.0
        if comp_mean_gray < 238:
            continue
        if word_count > 2:
            area_ratio = area / float(crop_area)
            bbox_ratio = bbox_area / float(crop_area)
            if area_ratio < 0.08 and bbox_ratio < 0.10:
                continue
        edge_touches = int(x <= 1) + int(y <= 1)
        edge_touches += int((x + cw) >= w - 1)
        edge_touches += int((y + ch) >= h - 1)
        if edge_touches >= 2 and bbox_area > int(crop_area * 0.62):
            continue
        center_dist = abs((x + cw * 0.5) - cx) + abs((y + ch * 0.5) - cy)
        score = area * 1.4 + fill_ratio * 2000.0 - center_dist * 5.0
        if score > best_score:
            best_score = score
            best_label = label

    if best_label < 0:
        return None

    component = (labels == best_label).astype(np.uint8) * 255
    bbox = _mask_bbox(component)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 18 or bh < 18:
        return None
    return x1, y1, bw, bh, component[y1:y2, x1:x2].copy()


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


def _text_layout_height(
    *,
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    line_spacing: int,
    stroke_width: int = 0,
) -> int:
    """Return the rendered text block height in pixels."""
    if not lines:
        return 0
    heights: list[int] = []
    for line in lines:
        _, top, _, bottom = draw.textbbox(
            (0, 0),
            line,
            font=font,
            anchor="lt",
            stroke_width=stroke_width,
        )
        heights.append(max(0, int(bottom - top)))
    return sum(heights) + max(0, len(lines) - 1) * line_spacing


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
    # Use 75th percentile instead of 45th to preserve more usable width,
    # since text is centered and most lines sit in the wider middle of a bubble.
    return int(np.percentile(widths, 75))


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

    left_values = np.array(row_left)
    right_values = np.array(row_right)

    # Use an inner span instead of the widest visible row.  Manga balloons often
    # curve sharply near the top/bottom; the previous 25/75 percentile span let
    # a line pass if only a few rows were wide enough, which caused edge spills.
    left = int(np.percentile(left_values, 55))
    right = int(np.percentile(right_values, 45))
    avail = right - left + 1
    if avail < max(8, int(line_w * 0.60)):
        left = int(np.percentile(left_values, 40))
        right = int(np.percentile(right_values, 60))
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


def _fit_dialogue_to_bubble_mask(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    avail_w: int,
    avail_h: int,
    mask: NDArray,
    text_padding: int,
    min_size: int,
    max_size: int,
    max_lines: int,
) -> tuple[
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    list[str],
    int,
    list[tuple[int, int, int, int]],
    int,
] | None:
    """Find the largest dialogue layout that fits the speech-bubble mask."""
    if mask.size == 0 or avail_w <= 0 or avail_h <= 0:
        return None

    hi = max(min_size, int(max_size))
    for size in range(hi, min_size - 1, -1):
        for width_ratio in (1.0, 0.94, 0.88, 0.82, 0.76, 0.70):
            wrap_w = max(10, int(avail_w * width_ratio))
            font, lines, line_spacing = _fit_text_to_box(
                draw=draw,
                text=text,
                max_w=wrap_w,
                max_h=avail_h,
                font_path=font_path,
                style="dialogue",
                min_size=size,
                max_size=size,
            )
            if not lines or len(lines) > max_lines:
                continue
            if getattr(font, "size", size) < size:
                continue

            stroke_width = _stroke_width_for_font(font)
            metrics: list[tuple[int, int, int, int]] = []
            for line in lines:
                left, top, right, bottom = draw.textbbox(
                    (0, 0),
                    line,
                    font=font,
                    anchor="lt",
                    stroke_width=stroke_width,
                )
                metrics.append(
                    (
                        max(0, int(right - left)),
                        max(0, int(bottom - top)),
                        int(left),
                        int(top),
                    )
                )

            total_h = sum(lh for _, lh, _, _ in metrics) + (
                len(metrics) - 1
            ) * line_spacing
            if total_h > avail_h:
                continue

            slack = max(0, int(avail_h - total_h))
            centered = int(text_padding + slack // 2)
            candidates = _candidate_mask_start_ys(centered, text_padding, slack)
            for local_start_y in candidates:
                if _layout_fits_bubble_mask(
                    mask=mask,
                    start_y=local_start_y,
                    line_metrics=metrics,
                    line_spacing=line_spacing,
                    margin=3,
                ):
                    return font, lines, line_spacing, metrics, local_start_y

    return None


def _fit_shortened_dialogue_to_bubble_mask(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    avail_w: int,
    avail_h: int,
    mask: NDArray,
    text_padding: int,
    min_size: int,
    max_size: int,
    max_lines: int,
) -> tuple[
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    list[str],
    int,
    list[tuple[int, int, int, int]],
    int,
    str,
] | None:
    """Find a mask-safe dialogue layout, shortening only at word boundaries."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return None

    words = clean.split()
    candidates: list[str] = [clean]
    seen = {clean}
    for max_words in range(min(len(words) - 1, 12), 1, -1):
        candidate = _compress_dialogue_for_tiny_box(clean, max_words=max_words)
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    for max_words in range(min(len(words) - 1, 10), 1, -1):
        candidate = _truncate_dialogue_at_word_boundary(clean, max_words=max_words)
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    for candidate in _narrow_dialogue_rescue_candidates(clean):
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for candidate in candidates:
        layout = _fit_dialogue_to_bubble_mask(
            draw=draw,
            text=candidate,
            font_path=font_path,
            avail_w=avail_w,
            avail_h=avail_h,
            mask=mask,
            text_padding=text_padding,
            min_size=min_size,
            max_size=max_size,
            max_lines=max_lines,
        )
        if layout is None:
            continue
        font, lines, line_spacing, metrics, local_start_y = layout
        return font, lines, line_spacing, metrics, local_start_y, candidate

    return None


def _fit_relaxed_spill_dialogue_to_bubble_mask(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    avail_w: int,
    avail_h: int,
    mask: NDArray,
    text_padding: int,
    min_size: int,
    max_size: int,
    max_lines: int,
) -> tuple[
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    list[str],
    int,
    list[tuple[int, int, int, int]],
    int,
    str,
    NDArray,
] | None:
    """Last-resort readable dialogue fit with a tiny controlled mask overflow."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean or mask.size == 0:
        return None

    # Keep this as an emergency path for awkward speech balloons. Normal
    # medium/large rectangular bubbles should keep the strict no-spill behavior.
    min_dim = min(avail_w, avail_h)
    max_dim = max(avail_w, avail_h)
    aspect = max_dim / float(max(1, min_dim))
    area = avail_w * avail_h
    if min_dim > 120 and aspect < 1.45 and area > 22000:
        return None

    spill_px = int(np.clip(round(min_dim * 0.10), 3, 10))
    if aspect >= 2.0:
        spill_px = min(12, spill_px + 2)
    kernel_size = spill_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    relaxed_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    words = clean.split()
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str) -> None:
        candidate = " ".join(candidate.split()).strip()
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    add_candidate(clean)
    for max_words in range(min(len(words) - 1, 12), 0, -1):
        add_candidate(_compress_dialogue_for_tiny_box(clean, max_words=max_words))
    for max_words in range(min(len(words) - 1, 10), 0, -1):
        add_candidate(_truncate_dialogue_at_word_boundary(clean, max_words=max_words))
    for candidate in _narrow_dialogue_rescue_candidates(clean):
        add_candidate(candidate)

    relaxed_max_lines = max_lines + (1 if avail_h >= 80 else 0)
    for candidate in candidates:
        layout = _fit_dialogue_to_bubble_mask(
            draw=draw,
            text=candidate,
            font_path=font_path,
            avail_w=avail_w,
            avail_h=avail_h,
            mask=relaxed_mask,
            text_padding=text_padding,
            min_size=min_size,
            max_size=max(min_size, max_size),
            max_lines=relaxed_max_lines,
        )
        if layout is None:
            continue
        font, lines, line_spacing, metrics, local_start_y = layout
        logger.debug(
            "Dialogue constraints: using controlled spill fallback, spill=%dpx, lines=%d",
            spill_px,
            len(lines),
        )
        return (
            font,
            lines,
            line_spacing,
            metrics,
            local_start_y,
            candidate,
            relaxed_mask,
        )

    return None


def _truncate_dialogue_at_word_boundary(text: str, max_words: int) -> str:
    """Shorten dialogue for tiny masks without breaking or hyphenating words."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean
    words = clean.split()
    if len(words) <= max_words:
        return clean
    trimmed = " ".join(words[:max_words]).rstrip(".,;:!?")
    return f"{trimmed}..."


def _candidate_mask_start_ys(centered: int, padding: int, slack: int) -> list[int]:
    """Return centered-first vertical starts within the available text slack."""
    low = int(padding)
    high = int(padding + max(0, slack))
    centered = int(np.clip(centered, low, high))
    values = [centered]
    for delta in (4, -4, 8, -8, 12, -12, 18, -18, 24, -24):
        if abs(delta) > slack + 1:
            continue
        cand = int(np.clip(centered + delta, low, high))
        if cand not in values:
            values.append(cand)
    for cand in (low, high):
        if cand not in values:
            values.append(cand)
    return values


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


def _cap_span_around_center(
    start: int,
    end: int,
    max_len: int,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    """Cap a 1D span around its center while staying inside bounds."""
    span = int(end - start)
    if span <= max_len:
        return start, end
    center = (start + end) * 0.5
    half = max_len * 0.5
    new_start = int(round(center - half))
    new_end = int(round(center + half))
    if new_start < lower:
        new_end += lower - new_start
        new_start = lower
    if new_end > upper:
        new_start -= new_end - upper
        new_end = upper
    new_start = max(lower, new_start)
    new_end = min(upper, new_end)
    if new_end <= new_start:
        return start, end
    return new_start, new_end


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
    if len(clean) < 8:
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
