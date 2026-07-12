from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from main.detector import TextRegion
from main.pipeline import (
    RenderTextUnit,
    _cluster_same_surface_row_regions,
    _is_renderable_unit,
    _restore_unit_source_text,
)
from main.renderer import (
    _is_mask_anchored_art_dialogue_box,
    build_inpaint_mask,
    _crop_source_mask_to_box,
    _fit_text_to_box,
    _layout_fits_bubble_mask,
    _resolve_dialogue_box,
)


def _region(x: int, y: int, w: int, h: int) -> TextRegion:
    return TextRegion(
        x=x,
        y=y,
        w=w,
        h=h,
        cropped=np.full((h, w, 3), 255, dtype=np.uint8),
        mask=np.full((h, w), 255, dtype=np.uint8),
    )


def test_distant_vertical_columns_are_not_merged() -> None:
    regions = [
        _region(100, 80, 95, 230),
        _region(405, 90, 110, 240),
        _region(750, 85, 100, 225),
        _region(1090, 85, 100, 230),
    ]

    clusters = _cluster_same_surface_row_regions(regions, list(range(len(regions))))

    assert clusters == [[0], [1], [2], [3]]


def test_close_vertical_columns_can_still_share_a_bubble() -> None:
    regions = [
        _region(100, 80, 45, 210),
        _region(158, 86, 45, 205),
        _region(216, 82, 45, 215),
    ]

    clusters = _cluster_same_surface_row_regions(regions, list(range(len(regions))))

    assert clusters == [[0, 1, 2]]


def test_dialogue_wrap_never_inserts_hyphen_breaks() -> None:
    draw = ImageDraw.Draw(Image.new("RGB", (300, 300), "white"))
    _, lines, _ = _fit_text_to_box(
        draw=draw,
        text="EXTRAORDINARILY COMPLICATED DIALOGUE",
        max_w=90,
        max_h=200,
        font_path=None,
        style="dialogue",
        min_size=10,
        max_size=28,
    )

    assert lines
    assert all(not line.endswith("-") for line in lines)


def test_bubble_fit_rejects_a_line_wider_than_oval_edge() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    ImageDraw.Draw(Image.fromarray(mask)).ellipse((5, 5, 114, 94), fill=255)
    metrics = [(100, 18, 0, 0), (60, 18, 0, 0)]

    assert not _layout_fits_bubble_mask(
        mask=mask,
        start_y=7,
        line_metrics=metrics,
        line_spacing=4,
        margin=3,
    )


def test_maskless_expansion_is_not_claimed_as_a_bubble() -> None:
    image = np.full((220, 160, 3), 235, dtype=np.uint8)
    text_mask = np.zeros((220, 160), dtype=np.uint8)
    text_mask[40:190, 68:92] = 255

    *_, bubble_used, clip_mask = _resolve_dialogue_box(
        x=0,
        y=0,
        w=160,
        h=220,
        region_mask=text_mask,
        region_image=image,
        translated_text="THIS TRANSLATION FITS INSIDE",
    )

    assert not bubble_used
    assert clip_mask is None


def test_source_mask_is_projected_into_fallback_box() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[30:70, 45:60] = 255

    cropped = _crop_source_mask_to_box(
        region_mask=mask,
        region_x=100,
        region_y=200,
        box_x=130,
        box_y=220,
        box_w=50,
        box_h=60,
    )

    assert cropped is not None
    assert cropped.shape == (60, 50)
    assert np.count_nonzero(cropped) > 0


def test_failed_render_restores_the_complete_inpaint_footprint() -> None:
    original = np.full((80, 100, 3), 245, dtype=np.uint8)
    original[28:52, 43:57] = 20
    mask = np.zeros((60, 70), dtype=np.uint8)
    mask[18:42, 28:42] = 255
    region = TextRegion(
        x=15,
        y=10,
        w=70,
        h=60,
        cropped=original[10:70, 15:85].copy(),
        mask=mask,
    )
    unit = RenderTextUnit(region, region, "日本語", "dialogue", 0)
    erased = original.copy()
    footprint = build_inpaint_mask(region.cropped, mask) > 0
    erased_crop = erased[10:70, 15:85]
    erased_crop[footprint] = 255

    restored = _restore_unit_source_text(erased, original, unit)

    assert np.array_equal(restored, original)


def test_failed_restore_does_not_cover_an_existing_translation() -> None:
    original = np.full((70, 90, 3), 240, dtype=np.uint8)
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[15:35, 20:40] = 255
    region = TextRegion(
        x=15,
        y=10,
        w=60,
        h=50,
        cropped=original[10:60, 15:75].copy(),
        mask=mask,
    )
    unit = RenderTextUnit(region, region, "日本語", "dialogue", 0)
    result = original.copy()
    result[35, 45] = (0, 0, 0)
    protected = np.zeros(result.shape[:2], dtype=np.uint8)
    protected[35, 45] = 255

    restored = _restore_unit_source_text(result, original, unit, protected)

    assert np.array_equal(restored[35, 45], np.array([0, 0, 0], dtype=np.uint8))


def test_art_fallback_rejects_a_white_balloon_surface() -> None:
    image = np.full((100, 80, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:80, 34:46] = 255

    assert not _is_mask_anchored_art_dialogue_box(
        region_image=image,
        region_mask=mask,
        search_x=0,
        search_y=0,
        box_x=0,
        box_y=0,
        box_w=80,
        box_h=100,
        text="THIS IS DIALOGUE",
    )


def test_art_fallback_accepts_source_anchored_text_on_colored_art() -> None:
    image = np.full((100, 80, 3), (120, 165, 220), dtype=np.uint8)
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:80, 30:50] = 255

    assert _is_mask_anchored_art_dialogue_box(
        region_image=image,
        region_mask=mask,
        search_x=0,
        search_y=0,
        box_x=0,
        box_y=0,
        box_w=80,
        box_h=100,
        text="NARRATION HERE",
    )


def test_art_fallback_accepts_narration_on_pale_beige_art() -> None:
    image = np.full((400, 170, 3), (210, 225, 235), dtype=np.uint8)
    mask = np.zeros((400, 170), dtype=np.uint8)
    mask[20:380, 60:110] = 255

    assert _is_mask_anchored_art_dialogue_box(
        region_image=image,
        region_mask=mask,
        search_x=0,
        search_y=0,
        box_x=0,
        box_y=0,
        box_w=170,
        box_h=400,
        text="NARRATION ON A PALE WALL",
    )


def test_valid_multiword_sfx_translation_is_renderable() -> None:
    region = _region(0, 0, 90, 100)
    unit = RenderTextUnit(region, region, "ドキドキッ", "sfx", 0)

    assert _is_renderable_unit(unit, "HEART POUNDING LOUD")


def test_kana_narration_on_art_is_not_discarded_as_noise() -> None:
    crop = np.full((150, 70, 3), (110, 155, 215), dtype=np.uint8)
    mask = np.zeros((150, 70), dtype=np.uint8)
    mask[10:140, 25:45] = 255
    region = TextRegion(0, 0, 70, 150, crop, mask)
    unit = RenderTextUnit(region, region, "それからうちにかえると", "dialogue", 0)

    assert _is_renderable_unit(unit, "WHEN I WENT HOME AFTERWARD")


def test_single_character_detected_sfx_is_not_silently_filtered() -> None:
    region = _region(0, 0, 34, 42)
    unit = RenderTextUnit(region, region, "ド", "sfx", 0)

    assert _is_renderable_unit(unit, "THUD")


def test_non_japanese_ocr_fragment_is_not_replaced() -> None:
    region = _region(0, 0, 34, 42)
    unit = RenderTextUnit(region, region, "K,", "sfx", 0)

    assert not _is_renderable_unit(unit, "WHAT?")


def test_long_outside_translation_is_not_silently_filtered() -> None:
    crop = np.full((300, 220, 3), (120, 160, 200), dtype=np.uint8)
    mask = np.zeros((300, 220), dtype=np.uint8)
    mask[20:280, 80:140] = 255
    region = TextRegion(0, 0, 220, 300, crop, mask)
    unit = RenderTextUnit(region, region, "これは絵の上にある説明文です", "dialogue", 0)

    assert _is_renderable_unit(
        unit,
        "THIS EXPLANATORY TEXT IS PRINTED DIRECTLY OVER THE ARTWORK",
    )
