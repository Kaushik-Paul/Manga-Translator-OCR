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


def _regularize_balloon_shape(filled: NDArray) -> NDArray:
    """Recover a balloon interior from a ragged fill (returns bool mask).

    Translucent balloons let the artwork show through, so the flood fill stops
    at background texture and leaves a ragged, partial interior.  Balloons are
    (nearly) convex, so when the fill is clearly ragged its convex hull is a
    much better estimate of the interior.  Solid fills keep their exact shape
    so spiky/irregular balloons are not over-estimated.
    """
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return filled > 0
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return filled > 0
    solidity = float(cv2.countNonZero(filled)) / hull_area
    if solidity >= 0.85 or solidity < 0.45:
        return filled > 0
    hull_mask = np.zeros_like(filled)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return hull_mask > 0


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
    require_paper: bool = False,
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
    ring = cv2.dilate(own.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    ring &= text_d == 0
    if int(np.count_nonzero(ring)) < 12:
        return [], {}
    paper = _dominant_color(roi[ring].reshape(-1, 3))
    # Balloons are white (or, rarely, black) and unsaturated.  Lettering on
    # skin, clothes or coloured backgrounds is SFX/free text even when the
    # surrounding surface happens to be flat and outlined.  Mask-only
    # candidates (mostly SFX) must additionally sit on a bright surface.
    paper_px = paper.reshape(1, 1, 3)
    paper_gray = int(cv2.cvtColor(paper_px, cv2.COLOR_BGR2GRAY)[0, 0])
    paper_sat = int(cv2.cvtColor(paper_px, cv2.COLOR_BGR2HSV)[0, 0, 1])
    is_light = paper_gray >= (200 if require_paper else 190)
    is_dark = paper_gray <= 60 and not require_paper
    if paper_sat > 50 or not (is_light or is_dark):
        return [], {}

    fill_img = roi.copy()
    fill_img[text_d > 0] = paper
    fill_img = cv2.medianBlur(fill_img, 3)

    cache_key = (rx1, ry1, rx2, ry2)
    edges = gray_edges_cache.get(cache_key)
    if edges is None:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
        gray_edges_cache[cache_key] = edges
    barrier = _strip_glyph_edges(edges, text_roi)
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
        flood_mask = base_mask.copy()
        k = ys.size // 2
        sy, sx = int(ys[k]), int(xs[k])
        if not base_mask[sy + 1, sx + 1]:
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
        filled_u8 = _fill_holes(region.astype(np.uint8) * 255)
        filled = filled_u8 > 0
        # A balloon's only holes are its glyphs.  A flat art surface (wall,
        # skin, pillow) usually encloses other objects - even other balloons.
        holes = filled & ~region & (text_d == 0)
        if int(np.count_nonzero(holes)) > 0.15 * max(1, int(np.count_nonzero(filled))):
            touches = True
        if not touches:
            filled = _regularize_balloon_shape(filled_u8)
        idx = len(fills)
        fill_label[filled & (fill_label < 0)] = idx
        comp_to_fill[cid] = idx
        fills.append(_Fill(x=rx1, y=ry1, mask=filled, valid=not touches))

    # Plausibility: a balloon must surround its glyphs with clearly more paper
    # than the glyph footprint itself (otherwise the "fill" is just the
    # paper-coloured glyph area on top of artwork).
    roi_sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    for idx, fill in enumerate(fills):
        if not fill.valid:
            continue
        # Balloon paper is unsaturated across its whole interior; skin,
        # clothes and coloured backgrounds are not (measured on the surface
        # itself, so one lucky sample cannot pass a whole region).
        surface = fill.mask & ~own_d
        if np.count_nonzero(surface) >= 16:
            sat = roi_sat[surface]
            if float(np.median(sat)) > 30 or float(np.percentile(sat, 90)) > 40:
                fill.valid = False
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
        if covered < 0.7 or fill_area < max(text_box_area * 1.08, glyph_footprint * 1.25):
            fill.valid = False
        elif fill_area > text_box_area * (10 if require_paper else 14) or (
            fill_area > page_h * page_w * 0.08
        ):
            # Lettering on a large plain surface (wall, sky, bed sheet, title
            # page background) is free text/SFX, not balloon dialogue.
            fill.valid = False

    return fills, comp_to_fill


def _dominant_color(pixels: NDArray) -> NDArray:
    """Most common colour (coarse 16-level bins) of a pixel set, as BGR uint8.

    Unlike the median this stays on the balloon paper even when the sample
    ring around tight text also catches the outline and the art outside it.
    """
    q = (pixels.astype(np.int32) // 16)
    keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    top = np.bincount(keys).argmax()
    return np.median(pixels[keys == top], axis=0).astype(np.uint8)


def _strip_glyph_edges(edges: NDArray, text: NDArray) -> NDArray:
    """Remove glyph contours from an edge map, keep balloon outlines.

    Canny traces every glyph stroke; those closed contours would box in a
    fill seeded on the glyph.  Edge chains lying entirely next to text are
    glyph contours; a balloon outline continues well away from the text and
    is kept, even where a column touches it (only its 1px glyph halo goes).
    """
    out = edges.copy()
    out[cv2.dilate(text, np.ones((3, 3), np.uint8)) > 0] = 0
    near = cv2.dilate(text, np.ones((9, 9), np.uint8)) > 0
    n, lab = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    total = np.bincount(lab.ravel(), minlength=n)
    inside = np.bincount(lab[near].ravel(), minlength=n)
    glyph_like = inside >= total * 0.95
    glyph_like[0] = False
    out[glyph_like[lab]] = 0
    return out


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
    union = max(1, a.area + b.area - inter)
    return inter / union >= 0.5


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


def _include_text_footprint(
    bubble: BubbleMask, text_mask: NDArray, tx: int, ty: int
) -> BubbleMask:
    """Add the (column-closed) source glyph footprint to the balloon interior.

    The source text was printed there, so it is always safe render area, even
    where the flood fill was blocked (panel borders or tones crossing a
    translucent balloon).
    """
    ys, xs = np.nonzero(text_mask)
    if ys.size == 0:
        return bubble
    span = max(4, int(round(np.sqrt(ys.size) * 0.25)))
    k = _odd(min(41, span))
    footprint = cv2.morphologyEx(
        text_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )
    x1 = min(bubble.x, tx)
    y1 = min(bubble.y, ty)
    x2 = max(bubble.x + bubble.w, tx + text_mask.shape[1])
    y2 = max(bubble.y + bubble.h, ty + text_mask.shape[0])
    canvas = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    canvas[bubble.y - y1 : bubble.y - y1 + bubble.h, bubble.x - x1 : bubble.x - x1 + bubble.w] = bubble.mask
    sub = canvas[ty - y1 : ty - y1 + text_mask.shape[0], tx - x1 : tx - x1 + text_mask.shape[1]]
    np.maximum(sub, footprint, out=sub)
    return BubbleMask(x=x1, y=y1, mask=_fill_holes(canvas))


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
            image, edges_cache, labels, stats, binary, ids,
            require_paper=not candidate_is_model[bi],
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
            else:
                bubble = _include_text_footprint(bubble, gmask, gx1, gy1)
        groups.append(
            TextGroup(
                x1=gx1, y1=gy1, x2=gx2, y2=gy2,
                text_mask=gmask,
                bubble=bubble,
                component_ids=sorted(ids),
            )
        )
        claimed[ids] = True

    _demote_surfaces_containing_balloons(groups)
    groups = _absorb_text_inside_balloons(groups)
    groups = _merge_split_columns(groups)

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


def _mask_overlap(a: BubbleMask, b: BubbleMask) -> int:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    sub_a = a.mask[iy1 - a.y : iy2 - a.y, ix1 - a.x : ix2 - a.x] > 0
    sub_b = b.mask[iy1 - b.y : iy2 - b.y, ix1 - b.x : ix2 - b.x] > 0
    return int(np.count_nonzero(sub_a & sub_b))


def _demote_surfaces_containing_balloons(groups: list[TextGroup]) -> None:
    """A balloon never contains another balloon.

    When one "balloon" swallows most of another one, it is really a plain
    surface (wall, sheet, sky) that happens to hold lettering - keep that unit
    as free text so it cannot claim render space from the real balloons.
    """
    bubbles = [g for g in groups if g.bubble is not None]
    areas = {id(g): g.bubble.area for g in bubbles}
    demote: set[int] = set()
    for outer in bubbles:
        for inner in bubbles:
            if inner is outer or areas[id(outer)] <= 1.5 * areas[id(inner)]:
                continue
            if _mask_overlap(outer.bubble, inner.bubble) >= 0.5 * areas[id(inner)]:
                demote.add(id(outer))
                break
    for g in groups:
        if id(g) in demote:
            g.bubble = None


def _absorb_text_inside_balloons(groups: list[TextGroup]) -> list[TextGroup]:
    """Merge balloon-less units whose glyphs lie inside another unit's balloon.

    Happens when the detector boxes part of a balloon's text separately and
    only one part's fill was conclusive: it is all the same balloon.
    """
    bubbles = [g for g in groups if g.bubble is not None]
    if not bubbles:
        return groups
    absorbed: set[int] = set()
    for g in groups:
        if g.bubble is not None:
            continue
        ys, xs = np.nonzero(g.text_mask)
        if ys.size == 0:
            continue
        px, py = xs + g.x1, ys + g.y1
        for host in bubbles:
            b = host.bubble
            inb = (px >= b.x) & (px < b.x + b.w) & (py >= b.y) & (py < b.y + b.h)
            if not np.any(inb):
                continue
            hits = int(np.count_nonzero(b.mask[py[inb] - b.y, px[inb] - b.x] > 0))
            if hits >= 0.5 * ys.size:
                _merge_group_into(host, g)
                absorbed.add(id(g))
                break
    return [g for g in groups if id(g) not in absorbed]


def _merge_split_columns(groups: list[TextGroup]) -> list[TextGroup]:
    """Re-join lettering that the detector cut in two.

    Two units whose text boxes overlap, or that continue the same vertical
    column (or horizontal line) with almost no gap, are one sentence.
    Distinct balloons are separated by their outlines and some air, so the
    gap test keeps them apart; column continuations whose balloons are
    disjoint are never joined (overlapping text boxes are, e.g. when a panel
    border cuts through the balloon).
    """
    merged = True
    while merged:
        merged = False
        for a in groups:
            for b in groups:
                if a is b:
                    continue
                relation = _continues_line(a, b)
                if relation is None:
                    continue
                if relation == "line" and a.bubble is not None and b.bubble is not None:
                    small = min(a.bubble.area, b.bubble.area)
                    if _mask_overlap(a.bubble, b.bubble) < 0.15 * small:
                        continue
                bubble_b = b.bubble
                _merge_group_into(a, b)
                if bubble_b is not None:
                    a.bubble = (
                        bubble_b
                        if a.bubble is None
                        else _union_bubble_masks(a.bubble, bubble_b)
                    )
                groups = [g for g in groups if g is not b]
                merged = True
                break
            if merged:
                break
    return groups


def _continues_line(a: TextGroup, b: TextGroup) -> str | None:
    """True when ``b`` directly continues ``a``'s lettering.

    Either the two text boxes overlap substantially (one block cut across,
    e.g. at a panel border running through the balloon), or ``b`` continues
    ``a``'s single column / line after a tiny gap.
    """
    ix = min(a.x2, b.x2) - max(a.x1, b.x1)
    iy = min(a.y2, b.y2) - max(a.y1, b.y1)
    if ix > 0 and iy > 0 and ix * iy >= 0.25 * min(a.w * a.h, b.w * b.h):
        return "overlap"
    for vertical in (True, False):
        if vertical:
            a_lo, a_hi, b_lo, b_hi = a.x1, a.x2, b.x1, b.x2
            a_len, b_len = a.h, b.h
            gap = b.y1 - a.y2
        else:
            a_lo, a_hi, b_lo, b_hi = a.y1, a.y2, b.y1, b.y2
            a_len, b_len = a.w, b.w
            gap = b.x1 - a.x2
        thick = min(a_hi - a_lo, b_hi - b_lo)
        if thick <= 0:
            continue
        # Both must look like one line: long along the axis, one glyph thick.
        if max(a_hi - a_lo, b_hi - b_lo) > 1.6 * thick:
            continue
        if a_len < 1.5 * (a_hi - a_lo):
            continue
        overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
        if overlap < 0.7 * thick:
            continue
        if -2 <= gap <= 0.5 * thick and b_len <= 2.5 * a_len:
            return "line"
    return None


def _union_bubble_masks(a: BubbleMask, b: BubbleMask) -> BubbleMask:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2, y2 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for m in (a, b):
        sub = mask[m.y - y1 : m.y - y1 + m.h, m.x - x1 : m.x - x1 + m.w]
        np.maximum(sub, m.mask, out=sub)
    return BubbleMask(x=x1, y=y1, mask=_fill_holes(mask))


def _merge_group_into(host: TextGroup, other: TextGroup) -> None:
    x1, y1 = min(host.x1, other.x1), min(host.y1, other.y1)
    x2, y2 = max(host.x2, other.x2), max(host.y2, other.y2)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for g in (host, other):
        sub = mask[g.y1 - y1 : g.y2 - y1, g.x1 - x1 : g.x2 - x1]
        np.maximum(sub, g.text_mask, out=sub)
    host.x1, host.y1, host.x2, host.y2 = x1, y1, x2, y2
    host.text_mask = mask
    host.component_ids = sorted(set(host.component_ids) | set(other.component_ids))
    if host.bubble is not None:
        host.bubble = _include_text_footprint(host.bubble, other.text_mask, other.x1, other.y1)


def _sort_groups_manga_order(groups: list[TextGroup], page_w: int) -> list[TextGroup]:
    """Right-to-left columns, top-to-bottom within a column band."""
    col_w = page_w / 3.0

    def key(g: TextGroup) -> tuple[int, int]:
        cx = (g.x1 + g.x2) / 2.0
        col = 2 - min(2, int(cx / col_w))
        return (col, (g.y1 + g.y2) // 2)

    return sorted(groups, key=key)
