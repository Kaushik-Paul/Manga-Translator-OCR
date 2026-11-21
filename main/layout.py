"""
Bubble-aware grouping of detected text into translation units.

The detector gives us a pixel text mask and (optionally) YOLO text blocks.
Neither is a reliable unit on its own: morphology on the mask both merges
neighbouring balloons and splits one balloon's columns apart, while the block
head occasionally boxes two touching balloons together or misses a balloon.

This module resolves units with the strongest signal available on the page:
the balloon itself.  For every candidate block we flood-fill the paper surface
around its glyphs, bounded by image edges (balloon outlines).  Text that shares
one plausible balloon becomes one unit; a block whose glyphs sit in two
balloons is split.  Text that does not sit in a balloon keeps its block
grouping (free text / narration on art), and glyphs that belong to neither
are returned as a residual mask for the legacy SFX/free-text path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector import DetectedBlock, extract_regions_from_mask

logger = logging.getLogger(__name__)


@dataclass
class BubbleMask:
    """Interior of a speech balloon in page coordinates (uint8 0/255 local mask)."""

    x: int
    y: int
    mask: NDArray

    @property
    def w(self) -> int:
        return int(self.mask.shape[1])

    @property
    def h(self) -> int:
        return int(self.mask.shape[0])

    @property
    def area(self) -> int:
        return int(cv2.countNonZero(self.mask))


@dataclass
class TextGroup:
    """One translation unit: all glyphs of one balloon (or one free-text block)."""

    x1: int
    y1: int
    x2: int
    y2: int
    text_mask: NDArray  # uint8 0/255, local to (x1, y1, x2, y2)
    bubble: BubbleMask | None
    component_ids: list[int] = field(default_factory=list)

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass
class _Fill:
    x: int
    y: int
    mask: NDArray  # bool, local
    valid: bool

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _fill_holes(mask: NDArray) -> NDArray:
    """Fill interior holes of a binary mask (uint8 0/255)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _bbox_of(stats: NDArray, ids: list[int]) -> tuple[int, int, int, int]:
    xs1 = stats[ids, cv2.CC_STAT_LEFT]
    ys1 = stats[ids, cv2.CC_STAT_TOP]
    xs2 = xs1 + stats[ids, cv2.CC_STAT_WIDTH]
    ys2 = ys1 + stats[ids, cv2.CC_STAT_HEIGHT]
    return int(xs1.min()), int(ys1.min()), int(xs2.max()), int(ys2.max())


def _flood_fills_for_components(
    image: NDArray,
    gray_edges_cache: dict,
    labels: NDArray,
    stats: NDArray,
    text_binary: NDArray,
    comp_ids: list[int],
) -> tuple[list[_Fill], dict[int, int]]:
    """Flood-fill the paper surface around a set of glyph components.

    Returns the fills and a mapping component id -> fill index.  Components
    whose fill is not a plausible balloon are still mapped (to an invalid
    fill) so callers can tell "not in a balloon" apart from "not processed".
    """
    page_h, page_w = labels.shape
    bx1, by1, bx2, by2 = _bbox_of(stats, comp_ids)
    bw, bh = bx2 - bx1, by2 - by1
    pad = int(max(bw, bh) * 1.25) + 16
    rx1, ry1 = max(0, bx1 - pad), max(0, by1 - pad)
    rx2, ry2 = min(page_w, bx2 + pad), min(page_h, by2 + pad)

    roi = image[ry1:ry2, rx1:rx2]
    text_roi = text_binary[ry1:ry2, rx1:rx2]
    labels_roi = labels[ry1:ry2, rx1:rx2]

    # Glyph pixels (all text, not just this set) are treated as paper so the
    # fill can pass through every column of the balloon.
    text_d = cv2.dilate(text_roi, np.ones((5, 5), np.uint8))
    own = np.isin(labels_roi, comp_ids)
    own_d = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = cv2.dilate(own.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    ring &= text_d == 0
    if int(np.count_nonzero(ring)) < 12:
        return [], {}
    paper = np.median(roi[ring].reshape(-1, 3), axis=0).astype(np.uint8)

    fill_img = roi.copy()
    fill_img[text_d > 0] = paper
    fill_img = cv2.medianBlur(fill_img, 3)

    cache_key = (rx1, ry1, rx2, ry2)
    edges = gray_edges_cache.get(cache_key)
    if edges is None:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
        gray_edges_cache[cache_key] = edges
    barrier = edges.copy()
    barrier[text_d > 0] = 0
    barrier = cv2.dilate(barrier, np.ones((2, 2), np.uint8))

    roi_h, roi_w = roi.shape[:2]
    # Sides of the ROI that are page borders may legitimately be touched by a
    # balloon cut off at the page edge; any other side means the fill leaked.
    open_sides = (rx1 > 0, ry1 > 0, rx2 < page_w, ry2 < page_h)

    base_mask = np.zeros((roi_h + 2, roi_w + 2), np.uint8)
    base_mask[1:-1, 1:-1] = (barrier > 0).astype(np.uint8)

    fills: list[_Fill] = []
    fill_label = np.full((roi_h, roi_w), -1, dtype=np.int32)
    comp_to_fill: dict[int, int] = {}
    tol = (20, 20, 20)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)

    order = sorted(comp_ids, key=lambda c: -int(stats[c, cv2.CC_STAT_AREA]))
    for cid in order:
        ys, xs = np.nonzero(labels_roi == cid)
        if ys.size == 0:
            continue
        existing = fill_label[ys, xs]
        hit = existing[existing >= 0]
        if hit.size:
            comp_to_fill[cid] = int(np.bincount(hit).argmax())
            continue
        k = ys.size // 2
        sy, sx = int(ys[k]), int(xs[k])
        if base_mask[sy + 1, sx + 1]:
            continue
        flood_mask = base_mask.copy()
        cv2.floodFill(fill_img, flood_mask, (sx, sy), 0, tol, tol, flags)
        region = flood_mask[1:-1, 1:-1] == 255
        if not region.any():
            continue

        touches = (
            (open_sides[0] and region[:, 0].any())
            or (open_sides[1] and region[0, :].any())
            or (open_sides[2] and region[:, -1].any())
            or (open_sides[3] and region[-1, :].any())
        )
        filled = _fill_holes(region.astype(np.uint8) * 255) > 0
        idx = len(fills)
        fill_label[filled & (fill_label < 0)] = idx
        comp_to_fill[cid] = idx
        fills.append(_Fill(x=rx1, y=ry1, mask=filled, valid=not touches))

    # Plausibility: a balloon must surround its glyphs with clearly more paper
    # than the glyph footprint itself (otherwise the "fill" is just the
    # paper-coloured glyph area on top of artwork).
    for idx, fill in enumerate(fills):
        if not fill.valid:
            continue
        members = [c for c, f in comp_to_fill.items() if f == idx]
        if not members:
            fill.valid = False
            continue
        mx1, my1, mx2, my2 = _bbox_of(stats, members)
        lx1, ly1, lx2, ly2 = mx1 - rx1, my1 - ry1, mx2 - rx1, my2 - ry1
        text_box_area = max(1, (lx2 - lx1) * (ly2 - ly1))
        covered = float(np.count_nonzero(fill.mask[ly1:ly2, lx1:lx2])) / text_box_area
        glyph_footprint = int(np.count_nonzero(own_d & fill.mask))
        fill_area = fill.area
        if covered < 0.85 or fill_area < max(text_box_area * 1.08, glyph_footprint * 1.6):
            fill.valid = False

    return fills, comp_to_fill


def _fills_overlap(a: _Fill, b: _Fill) -> bool:
    ax2, ay2 = a.x + a.mask.shape[1], a.y + a.mask.shape[0]
    bx2, by2 = b.x + b.mask.shape[1], b.y + b.mask.shape[0]
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    sub_a = a.mask[iy1 - a.y : iy2 - a.y, ix1 - a.x : ix2 - a.x]
    sub_b = b.mask[iy1 - b.y : iy2 - b.y, ix1 - b.x : ix2 - b.x]
    inter = int(np.count_nonzero(sub_a & sub_b))
    smaller = max(1, min(a.area, b.area))
    return inter / smaller >= 0.6


def _union_fills(fills: list[_Fill]) -> BubbleMask:
    x1 = min(f.x for f in fills)
    y1 = min(f.y for f in fills)
    x2 = max(f.x + f.mask.shape[1] for f in fills)
    y2 = max(f.y + f.mask.shape[0] for f in fills)
    canvas = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for f in fills:
        sub = canvas[f.y - y1 : f.y - y1 + f.mask.shape[0], f.x - x1 : f.x - x1 + f.mask.shape[1]]
        sub[f.mask] = 255
    # Remove thin leaks through outline gaps and smooth the contour.
    k = _odd(max(3, int(round(min(canvas.shape) * 0.03))))
    opened = cv2.morphologyEx(
        canvas, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    if cv2.countNonZero(opened) > cv2.countNonZero(canvas) * 0.6:
        canvas = opened
    # Keep the component(s) that actually hold paper, crop to their bbox.
    canvas = _fill_holes(canvas)
    ys, xs = np.nonzero(canvas)
    if ys.size == 0:
        return BubbleMask(x=x1, y=y1, mask=canvas)
    cx1, cy1, cx2, cy2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return BubbleMask(x=x1 + cx1, y=y1 + cy1, mask=canvas[cy1:cy2, cx1:cx2].copy())


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_text_by_bubbles(
    image: NDArray,
    text_mask: NDArray,
    blocks: list[DetectedBlock],
) -> tuple[list[TextGroup], NDArray]:
    """Group text-mask glyphs into balloon/block units.

    Returns:
        (groups, residual_mask): groups in manga reading order, and a uint8
        mask of glyph pixels not claimed by any group (left for the legacy
        free-text/SFX path).
    """
    page_h, page_w = text_mask.shape[:2]
    binary = (text_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return [], np.zeros_like(binary)

    min_cc = max(6, int(min(page_h, page_w) ** 2 * 4e-6))
    valid_comp = stats[:, cv2.CC_STAT_AREA] >= min_cc
    valid_comp[0] = False

    cx1 = stats[:, cv2.CC_STAT_LEFT]
    cy1 = stats[:, cv2.CC_STAT_TOP]
    cx2 = cx1 + stats[:, cv2.CC_STAT_WIDTH]
    cy2 = cy1 + stats[:, cv2.CC_STAT_HEIGHT]
    comp_area = np.maximum(1, stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT])

    # --- 1. candidate blocks: YOLO blocks, then mask regions for the rest ---
    comp_block = np.full(n, -1, dtype=np.int64)
    candidate_boxes: list[tuple[int, int, int, int]] = []
    candidate_is_model: list[bool] = []
    for b in sorted(blocks, key=lambda blk: blk.w * blk.h):
        m = max(2, int(min(b.w, b.h) * 0.08))
        bx1, by1, bx2, by2 = b.x1 - m, b.y1 - m, b.x2 + m, b.y2 + m
        ix = np.clip(np.minimum(cx2, bx2) - np.maximum(cx1, bx1), 0, None)
        iy = np.clip(np.minimum(cy2, by2) - np.maximum(cy1, by1), 0, None)
        inside = (ix * iy) / comp_area >= 0.5
        take = inside & valid_comp & (comp_block < 0)
        if not np.any(take):
            continue
        comp_block[take] = len(candidate_boxes)
        candidate_boxes.append((b.x1, b.y1, b.x2, b.y2))
        candidate_is_model.append(True)

    unassigned = valid_comp & (comp_block < 0)
    if np.any(unassigned):
        rest = np.isin(labels, np.nonzero(unassigned)[0]).astype(np.uint8) * 255
        for region in extract_regions_from_mask(image, rest):
            rx1, ry1, rx2, ry2 = region.x, region.y, region.x + region.w, region.y + region.h
            ix = np.clip(np.minimum(cx2, rx2) - np.maximum(cx1, rx1), 0, None)
            iy = np.clip(np.minimum(cy2, ry2) - np.maximum(cy1, ry1), 0, None)
            take = ((ix * iy) / comp_area >= 0.5) & unassigned & (comp_block < 0)
            if not np.any(take):
                continue
            comp_block[take] = len(candidate_boxes)
            candidate_boxes.append((rx1, ry1, rx2, ry2))
            candidate_is_model.append(False)

    # --- 2. balloon fills per candidate block ---
    edges_cache: dict = {}
    all_fills: list[_Fill] = []
    comp_fill = np.full(n, -1, dtype=np.int64)
    for bi in range(len(candidate_boxes)):
        ids = [int(c) for c in np.nonzero(comp_block == bi)[0]]
        if not ids:
            continue
        fills, mapping = _flood_fills_for_components(
            image, edges_cache, labels, stats, binary, ids
        )
        offset = len(all_fills)
        all_fills.extend(fills)
        for cid, fidx in mapping.items():
            comp_fill[cid] = offset + fidx

    # --- 3. build units: split blocks across balloons, merge blocks sharing one ---
    uf = _UnionFind(len(all_fills))
    for i in range(len(all_fills)):
        if not all_fills[i].valid:
            continue
        for j in range(i + 1, len(all_fills)):
            if all_fills[j].valid and _fills_overlap(all_fills[i], all_fills[j]):
                uf.union(i, j)

    # unit key: ("bubble", root fill) or ("block", block index)
    unit_of_comp: dict[int, tuple[str, int]] = {}
    for bi in range(len(candidate_boxes)):
        ids = [int(c) for c in np.nonzero(comp_block == bi)[0]]
        if not ids:
            continue
        in_bubble: dict[int, list[int]] = {}
        loose: list[int] = []
        for cid in ids:
            f = int(comp_fill[cid])
            if f >= 0 and all_fills[f].valid:
                in_bubble.setdefault(uf.find(f), []).append(cid)
            else:
                loose.append(cid)

        block_pixels = float(sum(int(stats[c, cv2.CC_STAT_AREA]) for c in ids))
        bubble_pixels = sum(
            int(stats[c, cv2.CC_STAT_AREA]) for cs in in_bubble.values() for c in cs
        )
        if in_bubble and bubble_pixels >= 0.5 * block_pixels:
            # Balloon text. Loose glyphs (punctuation touching the outline,
            # small marks) join the nearest balloon of this block.
            roots = list(in_bubble)
            for cid in ids:
                f = int(comp_fill[cid])
                if f >= 0 and all_fills[f].valid:
                    unit_of_comp[cid] = ("bubble", uf.find(f))
            for cid in loose:
                ccx = (cx1[cid] + cx2[cid]) / 2.0
                ccy = (cy1[cid] + cy2[cid]) / 2.0
                best, best_d = roots[0], float("inf")
                for root in roots:
                    mx1, my1, mx2, my2 = _bbox_of(stats, in_bubble[root])
                    dx = max(mx1 - ccx, 0, ccx - mx2)
                    dy = max(my1 - ccy, 0, ccy - my2)
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best, best_d = root, d
                unit_of_comp[cid] = ("bubble", best)
        elif candidate_is_model[bi]:
            # Free text the model boxed as one block (narration on art,
            # captions, titles): keep it as a single unit.
            for cid in ids:
                unit_of_comp[cid] = ("block", bi)
        # Mask-only candidates outside balloons stay residual (legacy path).

    units: dict[tuple[str, int], list[int]] = {}
    for cid, key in unit_of_comp.items():
        units.setdefault(key, []).append(cid)

    # Guard against fills that wrapped a whole plain panel: only keep a merged
    # balloon unit when its glyphs are a sensible share of the "balloon".
    groups: list[TextGroup] = []
    claimed = np.zeros(n, dtype=bool)
    for key, ids in units.items():
        gx1, gy1, gx2, gy2 = _bbox_of(stats, ids)
        local = labels[gy1:gy2, gx1:gx2]
        gmask = np.isin(local, ids).astype(np.uint8) * 255
        bubble: BubbleMask | None = None
        if key[0] == "bubble":
            root = key[1]
            members = [f for f in range(len(all_fills)) if all_fills[f].valid and uf.find(f) == root]
            bubble = _union_fills([all_fills[f] for f in members])
            if bubble.area == 0:
                bubble = None
        groups.append(
            TextGroup(
                x1=gx1, y1=gy1, x2=gx2, y2=gy2,
                text_mask=gmask,
                bubble=bubble,
                component_ids=sorted(ids),
            )
        )
        claimed[ids] = True

    residual_ids = np.nonzero(valid_comp & ~claimed)[0]
    residual = (
        np.isin(labels, residual_ids).astype(np.uint8) * 255
        if residual_ids.size
        else np.zeros_like(binary)
    )

    groups = _sort_groups_manga_order(groups, page_w)
    logger.debug(
        "Layout: %d model blocks -> %d units (%d in balloons), residual glyph comps=%d.",
        len(blocks),
        len(groups),
        sum(1 for g in groups if g.bubble is not None),
        int(residual_ids.size),
    )
    return groups, residual


def _sort_groups_manga_order(groups: list[TextGroup], page_w: int) -> list[TextGroup]:
    """Right-to-left columns, top-to-bottom within a column band."""
    col_w = page_w / 3.0

    def key(g: TextGroup) -> tuple[int, int]:
        cx = (g.x1 + g.x2) / 2.0
        col = 2 - min(2, int(cx / col_w))
        return (col, (g.y1 + g.y2) // 2)

    return sorted(groups, key=key)
