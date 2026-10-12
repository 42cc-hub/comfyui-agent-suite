"""Automatic comic-page layout: tile N images into one page with straight or slanted cuts.

Model
-----
The page is subdivided by recursively cutting ONE convex cell with ONE straight line.
That keeps every cell convex, makes the union exactly the page (no gaps, no overlap),
and supports arbitrary angles -- so slanted dividers need no special case.

Reading order is guaranteed by construction: a cut hands cell [lo..k] to one side and
[k+1..hi] to the other, so the index ranges stay contiguous and the tree is traversed
in order. Nothing is sorted afterwards.

Placement inside a cell
-----------------------
`cover` scale (max of the two axis ratios) makes the image fill the cell completely, so
there is never a white margin; whatever overflows the polygon is simply clipped away.
The crop window is then slid so the key region sits inside it, centred as far as the
image edges allow.

Key region
----------
Per image, the region that must stay visible. Derived here from tile variance (the
subject is the high-detail area against smooth path/sky), with the page-border tiles
discounted; a sidecar `<stem>.keymask.png` overrides it when present. A debug overlay is
written for every image so the derivation can be eyeballed instead of trusted.

Objective (minimised)
---------------------
  fit failure     key region cannot fit the visible window at cover scale  (heaviest)
  crop waste      fraction of the image hidden
  zoom excess     cover scale over fit scale
  thin cell       extreme aspect ratio
  key-edge band   key region too close to the cell edge
Search: best-first beam over (cell, split point, axis, ratio, angle).
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- key region

GRID = 20
KEEP_PCT = 72          # tiles at or above this variance percentile are "key"
KEY_PAD = 0.05         # pad the derived bbox by this fraction of the short side


def _tile_var(gray: np.ndarray, grid: int) -> np.ndarray:
    h, w = gray.shape
    th, tw = h // grid, w // grid
    v = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            blk = gray[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            v[i, j] = blk.std()
    return v


def _components(mask: np.ndarray):
    """8-connected components of a boolean tile grid."""
    gy, gx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(gy):
        for j in range(gx):
            if not mask[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gy and 0 <= nx < gx and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(comp)
    return comps


def _key_tiles(v: np.ndarray) -> np.ndarray:
    """High-detail tiles joined into one cluster around the largest component.

    A plain percentile threshold fails on a park photo: trees, grass and path texture
    are all high-variance, so their combined bounding box spans nearly the whole frame
    and the "keep the key region visible" constraint becomes vacuous. Taking the largest
    connected cluster -- then absorbing neighbours that touch its box -- locks onto the
    subject instead.
    """
    edge = np.ones_like(v, dtype=bool)
    edge[1:-1, 1:-1] = False
    vv = v.copy()
    vv[edge] *= 0.35
    hot = vv >= np.percentile(vv, KEEP_PCT)
    comps = _components(hot)
    if not comps:
        return hot
    comps.sort(key=len, reverse=True)
    main = comps[0]
    ys = [p[0] for p in main]
    xs = [p[1] for p in main]
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    keep = np.zeros_like(hot)
    for y, x in main:
        keep[y, x] = True
    for comp in comps[1:]:
        cy = sum(p[0] for p in comp) / len(comp)
        cx = sum(p[1] for p in comp) / len(comp)
        if y0 - 2 <= cy <= y1 + 2 and x0 - 2 <= cx <= x1 + 2:
            for y, x in comp:
                keep[y, x] = True
    return keep


def derive_key_bbox(path: str) -> tuple[int, int, int, int]:
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    small = im.resize((320, max(1, int(320 * ih / iw))), Image.LANCZOS).convert("L")
    a = np.asarray(small, dtype=np.float32)
    a = np.asarray(Image.fromarray(a.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    v = _tile_var(a, GRID)
    keep = _key_tiles(v)
    ys, xs = np.where(keep)
    if len(xs) == 0:
        return (int(iw * 0.15), int(ih * 0.15), int(iw * 0.85), int(ih * 0.85))
    th, tw = a.shape[0] // GRID, a.shape[1] // GRID
    x0, x1 = xs.min() * tw, min(a.shape[1], (xs.max() + 1) * tw)
    y0, y1 = ys.min() * th, min(a.shape[0], (ys.max() + 1) * th)
    sx, sy = iw / a.shape[1], ih / a.shape[0]
    pad = KEY_PAD * min(iw, ih)
    return (max(0, int(x0 * sx - pad)), max(0, int(y0 * sy - pad)),
            min(iw, int(x1 * sx + pad)), min(ih, int(y1 * sy + pad)))


def load_key_bbox(path: str) -> tuple[int, int, int, int]:
    side = os.path.splitext(path)[0] + ".keymask.png"
    if os.path.isfile(side):
        m = np.asarray(Image.open(side).convert("L"))
        ys, xs = np.where(m > 127)
        if len(xs):
            return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return derive_key_bbox(path)


# ---------------------------------------------------------------- geometry


def bbox_of(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _clip_half(poly, p, d, keep_positive):
    """Sutherland-Hodgman clip of a convex polygon against one half-plane."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]

        def side(q):
            s = d[0] * (q[1] - p[1]) - d[1] * (q[0] - p[0])
            return s if keep_positive else -s

        sa, sb = side(a), side(b)
        if sa >= 0:
            out.append(a)
        if (sa >= 0) != (sb >= 0):
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def split_convex(poly, axis, frac, angle_deg):
    """Cut one convex cell with a straight line; return the two convex halves.

    frac is the position along the cell's bbox on the chosen axis; angle_deg tilts the
    cut away from axis-aligned. Both halves keep the bbox's extent on the cut axis, so
    the pair still tiles the cell exactly.
    """
    x0, y0, x1, y1 = bbox_of(poly)
    w, h = x1 - x0, y1 - y0
    th = math.radians(angle_deg)
    if axis == "v":
        p = (x0 + frac * w, y0 + h * 0.5)
        d = (-math.sin(th), math.cos(th))
    else:
        p = (x0 + w * 0.5, y0 + frac * h)
        # tilt the horizontal cut; the sign makes +angle lean the same way for both axes
        d = (math.cos(th), math.sin(th))
    a = _clip_half(poly, p, d, True)
    b = _clip_half(poly, p, d, False)
    return a, b


def poly_area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


# ---------------------------------------------------------------- objective

W_FIT = 8.0        # key region does not fit the visible window
W_WASTE = 1.5      # fraction of the image cropped away
W_ZOOM = 0.7       # cover scale above fit scale
W_THIN = 2.2       # sliver cells
W_EDGES = 1.6      # a cell bounded by 3 or 5 edges is allowed, but not free
W_EDGE = 3.0       # key region hugging the cell border
W_CENTER = 3.2     # key region off the cell centre
CENTER_TOL = 0.10  # allowed offset from centre, as a fraction of the cell, before penalty
W_AREA = 2.0       # SUM of relative deviations from each panel's target share
W_UNSPLIT = 2.5    # an index range still holding several images must be paid for
W_BLANK_UNDER = 7.0   # want some reserved space, or there is nowhere for dialogue
W_BLANK_OVER = 25.0   # going over is far worse: a hole in the page reads as a misprint
W_BLANK_QUAD = 120.0  # and the slope steepens, so "slightly over" is not a cheap buy
BLANK_TARGET = 0.08  # target fraction of the page reserved for text
MAX_BLANK = 2      # at most this many reserved regions, or the search hides panels
MIN_FILL = 0.62    # a cell should use at least this much of the image's short side


def placement(poly, img_size, key):
    """Cover-scale crop window that keeps the key region inside, centred if possible."""
    bw, bh = bbox_of(poly)[2] - bbox_of(poly)[0], bbox_of(poly)[3] - bbox_of(poly)[1]
    iw, ih = img_size
    kx0, ky0, kx1, ky1 = key
    kw, kh = max(1, kx1 - kx0), max(1, ky1 - ky0)
    s = max(bw / iw, bh / ih)
    w_vis, h_vis = bw / s, bh / s
    cx = (kx0 + kx1) * 0.5 - w_vis * 0.5
    cy = (ky0 + ky1) * 0.5 - h_vis * 0.5
    cx = min(max(0.0, cx), max(0.0, iw - w_vis))
    cy = min(max(0.0, cy), max(0.0, ih - h_vis))
    return s, w_vis, h_vis, cx, cy


def cell_energy(poly, idx, sizes, keys):
    x0, y0, x1, y1 = bbox_of(poly)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    iw, ih = sizes[idx]
    kx0, ky0, kx1, ky1 = keys[idx]
    kw, kh = max(1, kx1 - kx0), max(1, ky1 - ky0)
    s, w_vis, h_vis, cx, cy = placement(poly, (iw, ih), keys[idx])

    over_w = max(0.0, kw - w_vis) / kw
    over_h = max(0.0, kh - h_vis) / kh
    e = W_FIT * (over_w + over_h)

    if over_w == 0 and over_h == 0:
        # margin between the key box and the window edge, on the tight axis
        m = min((kx0 - cx) / w_vis, (cx + w_vis - kx1) / w_vis,
                (ky0 - cy) / h_vis, (cy + h_vis - ky1) / h_vis)
        e += W_EDGE * max(0.0, 0.06 - m) / 0.06

        # Where the key region lands inside the cell. placement() can only centre it as
        # far as the image edges allow, so a key region near an image border comes out
        # visibly off-centre and the cell reads wrong. Penalising the offset pushes the
        # search toward cells whose shape lets the key region actually sit in the middle.
        u = ((kx0 + kx1) * 0.5 - cx) / w_vis
        v = ((ky0 + ky1) * 0.5 - cy) / h_vis
        off = max(0.0, abs(u - 0.5) - CENTER_TOL) + max(0.0, abs(v - 0.5) - CENTER_TOL)
        e += W_CENTER * off / 0.8

    e += W_WASTE * (1.0 - (w_vis * h_vis) / (iw * ih))
    s_fit = min(bw / iw, bh / ih)
    e += W_ZOOM * max(0.0, s / s_fit - 1.0)

    thin = min(bw, bh) / max(bw, bh)
    e += W_THIN * max(0.0, 0.34 - thin) / 0.34
    e += W_EDGES * abs(len(poly) - 4)

    # a cell whose short side leaves the image mostly unused reads as a mistake
    fill = min(w_vis / iw, h_vis / ih)
    e += 4.0 * max(0.0, MIN_FILL - fill) / MIN_FILL
    return e


def partial_energy(poly, lo, hi, sizes, keys):
    """Cheap proxy for an unsplit cell: how well do its images' aspects fit this shape?

    The flat term matters as much as the shape term: without it, leaving a whole index
    range unsplit and carving blank area instead can look cheaper than actually placing
    the panels, and the search settles on a page that renders only a few of them.
    """
    x0, y0, x1, y1 = bbox_of(poly)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    n = hi - lo + 1
    # target: the box should look like a stack of its images' aspects
    ar_img = np.mean([sizes[i][0] / sizes[i][1] for i in range(lo, hi + 1)])
    ar_cell = bw / bh
    aspect_pen = abs(math.log(ar_cell / ar_img)) * 0.5
    thin = min(bw, bh) / max(bw, bh)
    thin_pen = W_THIN * max(0.0, 0.34 - thin) / 0.34
    return aspect_pen * n + thin_pen + W_UNSPLIT * (hi - lo)


def area_penalty(cells, weights, page_area):
    """Each panel has a target share of the page, set by its own weight.

    A median-based balance term implicitly treats every panel as equally important, which
    is wrong for this page: the two shoe close-ups are inserts and read fine at a fraction
    of the walking shot's size. Penalising the RELATIVE deviation from each panel's own
    target says that directly, instead of fighting it.
    """
    tot = sum(weights)
    budget = page_area * (1.0 - BLANK_TARGET)
    e = 0.0
    n = 0
    for c in cells:
        if c.get("blank"):
            continue
        tw = (weights[c["lo"]] if c["hi"] == c["lo"]
              else sum(weights[c["lo"]:c["hi"] + 1]))
        target = budget * tw / tot
        e += abs(poly_area(c["poly"]) - target) / max(1.0, target)
        n += 1
    return W_AREA * e


def blank_penalty(cells, page_area):
    """Reserved empty area is ALLOWED but must not be free.

    This is not the same as a letterbox inside a cell: every image cell still has to be
    completely filled by its image. Blank cells hold no image at all -- they exist so the
    page can carry dialogue. Without a cost the search would leave large gaps because
    that trivially satisfies every per-cell constraint; with a one-sided cost it would go
    to zero. A target band gives both: room for text, no laziness.
    """
    b = sum(poly_area(c["poly"]) for c in cells if c.get("blank"))
    frac = b / page_area
    over = max(0.0, frac - BLANK_TARGET)
    under = max(0.0, BLANK_TARGET - frac)
    return W_BLANK_UNDER * under + W_BLANK_OVER * over + W_BLANK_QUAD * over * over


def total_energy(cells, sizes, keys, page_area, weights):
    e = blank_penalty(cells, page_area) + area_penalty(cells, weights, page_area)
    for c in cells:
        if c.get("blank"):
            continue
        if c["hi"] == c["lo"]:
            e += cell_energy(c["poly"], c["lo"], sizes, keys)
        else:
            e += partial_energy(c["poly"], c["lo"], c["hi"], sizes, keys)
    return e


# ---------------------------------------------------------------- search

RATIOS = (0.34, 0.42, 0.50, 0.58, 0.66)
ANGLES = (0.0, 14.0, -14.0)


def _centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def order_pair(a, b, axis):
    """Put the half that reads FIRST first: left for a vertical cut, top for a horizontal.

    Without this the index ranges stay contiguous but land in the wrong half, so the
    page tiles correctly and still tells the story out of order.
    """
    ca, cb = _centroid(a), _centroid(b)
    delta = (ca[0] - cb[0]) if axis == "v" else (ca[1] - cb[1])
    return (a, b) if delta < 0 else (b, a)


ROW_TOL = 0.15     # rows are bands of top edges, within this fraction of page height


def reading_order_ok(cells, page_h):
    """Do the cells read in index order, band by band?

    Bands are clustered by TOP EDGE (y0), not by vertical overlap. Overlap-based
    grouping passes layouts that read wrong: cell 4 at y 1982-2800 overlaps the tall
    cell 5 (1125-2800) by more than half the smaller height, so it gets merged into the
    middle band and the sequence looks like 3,4,5,6 -- while a reader, going by where
    each cell STARTS, sees 3,5,6 and then 4. Top-edge banding catches that.
    """
    items = []
    for c in cells:
        if c.get("blank"):
            continue
        x0, y0, x1, y1 = bbox_of(c["poly"])
        items.append((c["lo"], c["hi"], y0, x0))
    items.sort(key=lambda t: t[2])
    bands = []
    for it in items:
        if bands and it[2] - bands[-1][0][2] <= ROW_TOL * page_h:
            bands[-1].append(it)
        else:
            bands.append([it])
    seq = []
    for band in bands:
        for it in sorted(band, key=lambda t: t[3]):
            seq.append((it[0], it[1]))
    return all(seq[i][1] < seq[i + 1][0] for i in range(len(seq) - 1))


def touches_border(poly, page_w, page_h, tol=2.0):
    """Does this cell run to the page edge?

    Reserved space is constrained to the page margin on purpose. A blank cell in the
    middle of the page reads as a misprint (a hole); the same area along an edge reads as
    a text margin, which is what comic pages actually do with narration and dialogue.
    """
    x0, y0, x1, y1 = bbox_of(poly)
    return (x0 <= tol or y0 <= tol or x1 >= page_w - tol or y1 >= page_h - tol)


def _done(cells):
    return all(c.get("blank") or c["hi"] == c["lo"] for c in cells)


def search(sizes, keys, page_w, page_h, beam=8, max_cells_per_step=4, weights=None):
    page = [(0.0, 0.0), (page_w, 0.0), (page_w, page_h), (0.0, page_h)]
    n = len(sizes)
    weights = weights or [1.0] * n
    page_area = page_w * page_h
    frontier = [{"cells": [{"poly": page, "lo": 0, "hi": n - 1}]}]

    # upper bound: n-1 index splits plus MAX_BLANK carves, with slack
    for _ in range(2 * (n - 1) + MAX_BLANK):
        if all(_done(st["cells"]) for st in frontier):
            break
        cands = []
        for st in frontier:
            splittable = [i for i, c in enumerate(st["cells"])
                          if not c.get("blank") and c["hi"] > c["lo"]]
            splittable.sort(key=lambda i: -poly_area(st["cells"][i]["poly"]))
            for ci in splittable[:max_cells_per_step]:
                c = st["cells"][ci]
                rest = [x for i, x in enumerate(st["cells"]) if i != ci]
                for axis in ("v", "h"):
                    for frac in RATIOS:
                        for ang in ANGLES:
                            a, b = split_convex(c["poly"], axis, frac, ang)
                            if len(a) < 3 or len(b) < 3:
                                continue
                            if poly_area(a) < 2000 or poly_area(b) < 2000:
                                continue
                            # (1) carve a blank region and keep the whole range on the other side
                            if sum(1 for x in st["cells"] if x.get("blank")) < MAX_BLANK:
                                small, big = (a, b) if poly_area(a) < poly_area(b) else (b, a)
                                if touches_border(small, page_w, page_h):
                                    nc = rest + [{"poly": big, "lo": c["lo"], "hi": c["hi"]},
                                                 {"poly": small, "blank": True}]
                                    if reading_order_ok(nc, page_h):
                                        cands.append({"cells": nc})
                            # (2) ordinary split of the index range
                            for k in range(c["lo"], c["hi"]):
                                first, second = order_pair(a, b, axis)
                                nc = rest + [{"poly": first, "lo": c["lo"], "hi": k},
                                             {"poly": second, "lo": k + 1, "hi": c["hi"]}]
                                if reading_order_ok(nc, page_h):
                                    cands.append({"cells": nc})
        if not cands:
            break
        for st in cands:
            st["E"] = total_energy(st["cells"], sizes, keys, page_area, weights)
        cands.sort(key=lambda s: s["E"])
        frontier = cands[:beam]

    done = [st for st in frontier if _done(st["cells"])]
    if not done:
        print("WARNING: no complete layout in the beam; increase --beam")
        done = frontier
    best = min(done, key=lambda s: s["E"])
    return best


# ---------------------------------------------------------------- render


def render(cells, images, sizes, keys, page, out_path, seam=5, seam_color=(12, 12, 14)):
    page_img = Image.new("RGB", page, seam_color)
    for c in cells:
        if c.get("blank") or c["hi"] != c["lo"]:
            continue
        idx = c["lo"]
        x0, y0, x1, y1 = bbox_of(c["poly"])
        bw, bh = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
        if bw <= 0 or bh <= 0:
            continue
        s, w_vis, h_vis, cx, cy = placement(c["poly"], sizes[idx], keys[idx])

        crop_w = min(sizes[idx][0], max(1, int(round(w_vis))))
        crop_h = min(sizes[idx][1], max(1, int(round(h_vis))))
        left = int(round(min(max(0, cx), sizes[idx][0] - crop_w)))
        top = int(round(min(max(0, cy), sizes[idx][1] - crop_h)))

        src = images[idx].crop((left, top, left + crop_w, top + crop_h))
        if crop_w > crop_h * 1.2 and w_vis > h_vis:
            src = src.resize((bw, bh), Image.LANCZOS)
        elif h_vis > w_vis:
            src = src.resize((bw, bh), Image.LANCZOS)
        else:
            src = src.resize((bw, bh), Image.LANCZOS)

        # clip to the polygon so slanted cuts land where the maths says they do
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).polygon([(p[0] - x0, p[1] - y0) for p in c["poly"]], fill=255)
        page_img.paste(src, (int(round(x0)), int(round(y0))), mask)

    # draw the actual cut lines as seams, so neighbours are visually separated
    d = ImageDraw.Draw(page_img)
    for c in cells:
        pts = [(p[0], p[1]) for p in c["poly"]]
        d.line(pts + [pts[0]], fill=seam_color, width=seam)
    page_img.save(out_path)
    return page_img


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--page", default="2000x2800")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--debug-masks", default="")
    ap.add_argument("--weights", default="",
                    help="comma-separated per-image area weights, e.g. 1,0.6,1,0.6,1,1,1")
    a = ap.parse_args()

    pw, ph = (int(v) for v in a.page.lower().split("x"))
    wts = [float(v) for v in a.weights.split(",")] if a.weights else []
    images, sizes, keys = [], [], []
    for p in a.images:
        im = Image.open(p).convert("RGB")
        images.append(im)
        sizes.append(im.size)
        keys.append(load_key_bbox(p))
        print(f"  {os.path.basename(p):22} {im.size[0]:>5}x{im.size[1]:<5} "
              f"key={keys[-1]}")

    if a.debug_masks:
        os.makedirs(a.debug_masks, exist_ok=True)
        for p, k in zip(a.images, keys):
            im = Image.open(p).convert("RGB").copy()
            ImageDraw.Draw(im).rectangle(k, outline=(255, 0, 0), width=max(3, im.width // 160))
            im.thumbnail((760, 760))
            im.save(os.path.join(a.debug_masks, "key_" + os.path.basename(p)))

    best = search(sizes, keys, pw, ph, beam=a.beam, weights=wts)
    print(f"\nenergy={best['E']:.3f}")
    blanks = [c for c in best["cells"] if c.get("blank")]
    imgs = sorted([c for c in best["cells"] if not c.get("blank")], key=lambda c: c["lo"])
    if len(imgs) != len(sizes):
        print(f"WARNING: {len(imgs)} image cells for {len(sizes)} images -- "
              f"some panels were left unplaced")
    for c in imgs:
        x0, y0, x1, y1 = bbox_of(c["poly"])
        print(f"  #{c['lo'] + 1}  verts={len(c['poly'])}  bbox={int(x1 - x0)}x{int(y1 - y0)}")
    for b in blanks:
        x0, y0, x1, y1 = bbox_of(b["poly"])
        print(f"  [blank]  bbox={int(x1 - x0)}x{int(y1 - y0)}")

    render(best["cells"], images, sizes, keys, (pw, ph), a.out)
    print(f"\nwrote {a.out} {pw}x{ph} ({os.path.getsize(a.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
