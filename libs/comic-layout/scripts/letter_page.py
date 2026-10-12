"""Letter the page, treating each text box as a decision variable.

Three things beyond the first lettering pass:

1. A cell may be bounded by 3, 4 or 5 edges. The recursive convex splitting already
   produces those (a slanted cut can clip a corner), so the only change needed is an
   explicit penalty for |edges - 4|, which keeps 3/5 available but not free.

2. The text box is SEARCHED, not guessed. For a given cell the box's width, height and
   position are chosen to (a) keep at least HALF the box inside its own cell, (b) avoid
   the key region, and (c) waste as little box area as possible. Narrow-and-tall boxes
   are in the candidate set precisely because a slanted cell often has no wide free
   corner.

3. Captions no longer pick "the corner opposite the key region" per panel -- that made
   panel 4's caption sit above panel 2's once the two cells overlapped vertically. The
   search now carries an ordering term so caption order follows panel order.
"""

import json
import math
import os
import sys

sys.path.insert(0, ".")
import comic_layout as cl  # noqa: E402
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont  # noqa: E402

NAMES = ["01_walk.png", "02_lace_loose.png", "03_tying.png", "04_lace_tied.png",
         "05_teleport.png", "06_hop.png", "07_hiding.png"]
WEIGHTS = [1, 0.6, 1, 0.6, 1, 1, 1]
PAGE = (1400, 3200)
OUT = "candidates/_PAGE_LETTERED2.png"

FONT_PATH = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_SIZE = 28
PAPER = (246, 243, 237)
INK = (30, 28, 26)
SEAM = (18, 18, 20)

ANCHORS = json.load(open("candidates/anchors/anchors.json", encoding="utf-8"))

NARRATION = "在哥伦比娅的软磨硬泡下，桑多涅终于同意一起去逛公园了，不过是以这种方式"

# one entry per panel: LINES[0] belongs to panel 01, LINES[6] to panel 07.
# NARRATION is entry 0 and goes in the reserved cell, NOT here -- a leading None here
# shifted every caption by one panel.
LINES = [
    ["桑多涅：我只是做实验做累了想出来散散心，才不是为了陪你玩呢",
     "哥伦比娅：好吧，不过。。。。"],
    ["哥伦比娅：桑多涅的鞋带好像松了呢",
     "桑多涅：啊？那我。。。。（努力扭动身体试图弯下腰）"],
    ["哥伦比娅：桑多涅既然动不了就不要乱动哦，我来帮你系鞋带吧"],
    ["旁白：于是两边的鞋带就被这样绑在了一起"],
    ["哥伦比娅：我家里的月灵好像忘喂了，就先走啦~~",
     "桑多涅：哥！伦！比！娅！！你要干嘛！！！"],
    ["桑多涅：居然真的抛下我走了，要是别人看见我这样就糟了，快跑"],
    ["桑多涅：哥伦比娅什么时候回来把我放开啊~~~"],
]

# speaker -> where the tail points. Default is the key-region centre (i.e. the character);
# in panel 05 Columbina has already teleported away, so her line points at the particles.
SPEAKER_TOKEN = {"桑多涅": "sandrone", "哥伦比娅": "columbina", "旁白": "narr"}
ANCHOR_OVERRIDE = {(4, "columbina"): "particles"}


MIN_INSIDE = 0.50   # a box must keep at least this share of its area in its own cell
W_OUTSIDE = 14.0    # cost of box area falling outside the cell
W_KEYHIT = 0.6      # partial overlap of the key region is fine (a balloon over a figure
                    # is normal comic practice); "the key region must not be CROPPED" and
                    # "nothing may cover it" are different requirements
W_WASTE = 0.6       # cost of box area the text does not use
W_HIGH = 2.6
W_OVERLAP = 26.0   # balloons must not crowd each other
W_SLOT = 7.0       # same-panel balloons share the cell height instead of clustering
W_ANCHOR = 26.0    # a balloon belongs next to its speaker; this is the primary rule now        # prefer the top of the cell: a caption parked at the bottom of a tall
                    # cell overtakes the captions of the panels below it

# Box style per entry. Three registers, three shapes -- the shape tells the reader what
# kind of voice it is before they read a word.
STYLE_NARRATION = "narration"   # rectangle
STYLE_SPEECH = "speech"         # oval with a tail pointing at the speaker
STYLE_THOUGHT = "thought"       # scalloped cloud

# Box style per panel, aligned with LINES (panel 01 .. panel 07). Panel 04 is the
# narration "于是两边的鞋带就被这样绑在了一起", and 06/07 are Sandrone's unspoken
# thoughts while she is alone -- so those get the cloud.
STYLES = [STYLE_SPEECH, STYLE_SPEECH, STYLE_SPEECH, STYLE_NARRATION,
          STYLE_SPEECH, STYLE_THOUGHT, STYLE_THOUGHT]
NARRATION_STYLE = STYLE_NARRATION


def to_ccw(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return poly if a > 0 else list(reversed(poly))


def clip_convex(subject, clip):
    """Sutherland-Hodgman clip of `subject` by convex CCW `clip`."""
    out = list(subject)
    for i in range(len(clip)):
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        ex, ey = bx - ax, by - ay

        def side(p):
            return ex * (p[1] - ay) - ey * (p[0] - ax)

        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                new.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                new.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        out = new
        if not out:
            break
    return out


def wrap(text, font, max_w):
    out = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if font.getlength(cur + ch) > max_w and cur:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    return out


def box_metrics(box, cell_ccw, key_rect):
    bp = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    ba = (box[2] - box[0]) * (box[3] - box[1])
    inter = cl.poly_area(clip_convex(bp, cell_ccw))
    inside = inter / ba if ba > 0 else 0.0
    hit = 0.0
    if key_rect:
        kx0, ky0, kx1, ky1 = key_rect
        ox = max(0.0, min(box[2], kx1) - max(box[0], kx0))
        oy = max(0.0, min(box[3], ky1) - max(box[1], ky0))
        hit = (ox * oy) / ba if ba > 0 else 0.0
    return inside, hit


def place_text(cell_poly, size, key, text, font, pad=14, y_floor=0.0, placed=None,
               slot=None, anchor=None, diag=1.0, above=True):
    """Search width x position for the text box inside this cell.

    `y_floor` is the ordering constraint: a caption may not start above the previous
    panel's caption. Panel order is already fixed by the layout, so pinning caption order
    the same way is what stops a lower panel's line from being read first.
    """
    x0, y0, x1, y1 = cl.bbox_of(cell_poly)
    bw, bh = x1 - x0, y1 - y0
    # usable region for the box, inset so a balloon never runs off the cell or the page
    M = 26
    px0, py0, px1, py1 = x0 + M, y0 + M, x1 - M, y1 - M
    iw, ih = size
    kx0, ky0, kx1, ky1 = key
    s = max(bw / iw, bh / ih)
    w_vis, h_vis = bw / s, bh / s
    cx = min(max(0.0, (kx0 + kx1) / 2 - w_vis / 2), max(0.0, iw - w_vis))
    cy = min(max(0.0, (ky0 + ky1) / 2 - h_vis / 2), max(0.0, ih - h_vis))
    # key rect in page coords
    kxp0 = x0 + (kx0 - cx) / w_vis * bw
    kyp0 = y0 + (ky0 - cy) / h_vis * bh
    key_rect = (kxp0, kyp0,
                x0 + (kx1 - cx) / w_vis * bw, y0 + (ky1 - cy) / h_vis * bh)
    cell_ccw = to_ccw(cell_poly)

    best = None
    for frac in (0.30, 0.38, 0.46, 0.56, 0.68, 0.80, 0.92):
        maxw = max(120, int(bw * frac))
        lines = wrap(text, font, maxw - 2 * pad)
        lh = int(font.size * 1.42)
        tw = max(int(font.getlength(t)) for t in lines)
        w = min(maxw, tw + 2 * pad)
        h = lh * len(lines) + 2 * pad
        if w > bw or h > bh:
            continue
        # slide over a coarse grid; a narrow tall box often only fits along one edge
        for gx in (0.0, 0.12, 0.25, 0.5, 0.75, 0.88, 1.0):
            for gy in (0.0, 0.12, 0.25, 0.5, 0.75, 0.88, 1.0):
                bx = px0 + gx * max(0.0, (px1 - px0) - w)
                by = py0 + gy * max(0.0, (py1 - py0) - h)
                if by < y_floor:
                    continue
                box = (bx, by, bx + w, by + h)
                inside, hit = box_metrics(box, cell_ccw, key_rect)
                if inside < MIN_INSIDE:
                    continue
                used = (tw * lh * len(lines)) / max(1.0, w * h)
                # already-placed balloons compete for the same free space; without this
                # the hard ordering floor is the only thing keeping them apart, and it
                # packs them shoulder to shoulder instead of using the room below
                ov = 0.0
                for pb in (placed or []):
                    ox = max(0.0, min(box[2], pb[2]) - max(box[0], pb[0]))
                    oy = max(0.0, min(box[3], pb[3]) - max(box[1], pb[1]))
                    if ox > 0 and oy > 0:
                        ov = max(ov, (ox * oy) / max(1.0, w * h))
                e = (W_OUTSIDE * (1.0 - inside) + W_KEYHIT * hit
                     + W_WASTE * (1.0 - used)
                     + W_OVERLAP * ov)
                if anchor:
                    # ideal: the balloon sits just above its speaker, the way a real
                    # balloon's tail points down at whoever is talking
                    ix, iy = box[0] + w / 2.0, box[1] + h / 2.0
                    ax, ay = anchor
                    ideal_y = ay - (h / 2.0 + 16.0 if above else 0.0)
                    e += W_ANCHOR * math.hypot(ix - ax, iy - ideal_y) / max(1.0, diag)
                elif slot:
                    i, k = slot
                    want = py0 + ((i + 0.5) / k) * (py1 - py0) - h / 2.0
                    e += W_SLOT * abs(by - want) / max(1.0, py1 - py0)
                else:
                    e += W_HIGH * ((by - y0) / max(1.0, bh))
                if best is None or e < best[0]:
                    best = (e, box, lines, lh, pad)
    return best



def key_tip(cell, size, key):
    """Page-space centre of this cell's key region, used as the speech-tail target."""
    x0, y0, x1, y1 = cl.bbox_of(cell["poly"])
    bw, bh = x1 - x0, y1 - y0
    iw, ih = size
    kx0, ky0, kx1, ky1 = key
    s = max(bw / iw, bh / ih)
    w_vis, h_vis = bw / s, bh / s
    cx = min(max(0.0, (kx0 + kx1) / 2 - w_vis / 2), max(0.0, iw - w_vis))
    cy = min(max(0.0, (ky0 + ky1) / 2 - h_vis / 2), max(0.0, ih - h_vis))
    return (x0 + ((kx0 + kx1) / 2 - cx) / w_vis * bw,
            y0 + ((ky0 + ky1) / 2 - cy) / h_vis * bh)


def particle_anchor(cell, size, key):
    """Centre of the brightest blob in this cell -- the teleport glow."""
    import numpy as np
    x0, y0, x1, y1 = cl.bbox_of(cell["poly"])
    im = Image.open("candidates/final/" + NAMES[cell["lo"]]).convert("L")
    bw, bh = x1 - x0, y1 - y0
    iw, ih = size
    s = max(bw / iw, bh / ih)
    w_vis, h_vis = bw / s, bh / s
    cx = min(max(0.0, (key[0] + key[2]) / 2 - w_vis / 2), max(0.0, iw - w_vis))
    cy = min(max(0.0, (key[1] + key[3]) / 2 - h_vis / 2), max(0.0, ih - h_vis))
    crop = im.crop((int(cx), int(cy), int(cx + w_vis), int(cy + h_vis))).resize((64, 64))
    a = np.asarray(crop, dtype=np.float32)
    yy, xx = np.unravel_index(np.argmax(a), a.shape)
    return (x0 + xx / 63.0 * bw, y0 + yy / 63.0 * bh)


def union_shape(page_size, box, style, tip, tail_len):
    """Filled mask of the whole balloon: ellipse/rect plus a tail that starts INSIDE it."""
    m = Image.new("L", page_size, 0)
    d = ImageDraw.Draw(m)

    if style == STYLE_NARRATION:
        d.rounded_rectangle(list(box), radius=6, fill=255)
        return m

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2

    if style == STYLE_THOUGHT:
        lobes, depth, power = 10, 0.26, 0.85
        n = 320
        pts = []
        for i in range(n):
            th = 2 * math.pi * i / n
            # cycloid radius: continuous bulges, real cusps between them; the small
            # harmonics keep it from reading as a machined gear
            # |cos|, not |sin|: the cusp must sit at the SMALLEST radius, so the
            # round puffs are the outer edge and the notches cut inward
            r = 1.0 - depth * (1.0 - abs(math.cos(lobes * th / 2.0)) ** power)
            r *= 1.0 + 0.030 * math.sin(3 * th + 1.1) + 0.018 * math.cos(5 * th - 0.4)
            # anisotropic stretch: the same curve becomes a wide or tall cloud as needed
            pts.append((cx + math.cos(th) * rx * r, cy + math.sin(th) * ry * r))
        d.polygon(pts, fill=255)
    else:
        d.ellipse([x0, y0, x1, y1], fill=255)

    if tip:
        vx, vy = tip[0] - cx, tip[1] - cy
        dist = math.hypot(vx, vy)
        if dist > 1.0:
            ux, uy = vx / dist, vy / dist
            # base sits INSIDE the ellipse, so the union is one connected region
            bx = cx + ux * rx * 0.80
            by = cy + uy * ry * 0.80
            px, py = -uy, ux
            half = max(10.0, min(rx, ry) * 0.22)
            # short tail: stop well before a distant speaker rather than stretching to it
            reach = min(max(20.0, dist - max(rx, ry)), max(24.0, tail_len))
            txp, typ = bx + ux * reach, by + uy * reach
            d.polygon([(bx + px * half, by + py * half),
                       (bx - px * half, by - py * half), (txp, typ)], fill=255)
    return m


def draw_balloon(page, box, lines, font, style, pad, tip=None, tail_len=46):
    """Semi-transparent balloon whose border is traced from a single union mask."""
    if style in (STYLE_SPEECH, STYLE_THOUGHT):
        # an ellipse only contains a rectangle about 0.71 of its extent, so drawing it at
        # the text box's own footprint always spills. Inflate it about the centre instead.
        gx, gy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        # sqrt(2) is the exact bound for an ellipse to contain its text box; a cloud
        # dips to (1-depth) of the ellipse, so it needs the bound divided by that
        infl = 1.86 if style == STYLE_THOUGHT else 1.45
        hx, hy = (box[2] - box[0]) / 2 * infl, (box[3] - box[1]) / 2 * infl
        ebox = (gx - hx, gy - hy, gx + hx, gy + hy)
        ebox = (max(2.0, ebox[0]), max(2.0, ebox[1]),
                min(page.size[0] - 2.0, ebox[2]), min(page.size[1] - 2.0, ebox[3]))
    else:
        ebox = box
    # geometry is built at 2x and reduced, so the outline carries a real alpha edge
    # instead of the staircase a binary threshold produces
    S = 2
    big = (page.size[0] * S, page.size[1] * S)
    m = union_shape(big, tuple(v * S for v in ebox), style,
                    (tip[0] * S, tip[1] * S) if tip else None, tail_len * S)
    mask = m.filter(ImageFilter.GaussianBlur(1.2 * S)).point(
        lambda v: 255 if v > 127 else 0)
    border = mask.filter(ImageFilter.MaxFilter(5))
    inner = mask.filter(ImageFilter.MinFilter(5))
    edge = Image.eval(ImageChops.subtract(border, inner), lambda v: 255 if v > 40 else 0)
    mask = mask.resize(page.size, Image.LANCZOS)
    edge = edge.resize(page.size, Image.LANCZOS)

    lay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    lay.paste((252, 251, 249, 214), (0, 0, page.size[0], page.size[1]), mask)
    lay.paste((38, 36, 40, 255), (0, 0, page.size[0], page.size[1]), edge)
    d = ImageDraw.Draw(lay)
    lh = int(font.size * 1.42)
    total = lh * len(lines)
    ty = (ebox[1] + ebox[3]) / 2 - total / 2 + (2 if style != STYLE_NARRATION else 0)
    for i, s in enumerate(lines):
        tw = d.textlength(s, font=font)
        d.text(((ebox[0] + ebox[2]) / 2 - tw / 2, ty + i * lh), s, font=font, fill=INK)
    page.paste(Image.alpha_composite(page.convert("RGBA"), lay).convert("RGB"), (0, 0))


def main():
    cl.W_BLANK_OVER = 5000.0
    cl.W_BLANK_QUAD = 5000.0
    cl.W_BLANK_UNDER = 0.0
    cl.W_EDGES = 1.6          # penalty per edge away from four

    paths = ["candidates/final/" + n for n in NAMES]
    images = [Image.open(p).convert("RGB") for p in paths]
    sizes = [im.size for im in images]
    keys = [cl.load_key_bbox(p) for p in paths]

    best = cl.search(sizes, keys, PAGE[0], PAGE[1], beam=12, weights=WEIGHTS)
    print(f"E={best['E']:.2f}  order_ok={cl.reading_order_ok(best['cells'], PAGE[1])}")

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    cl.render(best["cells"], images, sizes, keys, PAGE, OUT)
    page = Image.open(OUT).convert("RGB")
    d = ImageDraw.Draw(page)
    for c in best["cells"]:
        if c.get("blank"):
            d.polygon([(p[0], p[1]) for p in c["poly"]], fill=PAPER)

    placements = []
    placed_boxes = []
    # narration first: it lives in the reserved cell, and the captions that follow may not
    # start above it
    y_floor = 0.0
    for c in [x for x in best["cells"] if x.get("blank")]:
        res = place_text(c["poly"], PAGE, (0, 0, PAGE[0], PAGE[1]), NARRATION, font,
                         y_floor=0.0, anchor=ANCHORS.get("n0"),
                         diag=math.hypot(PAGE[0], PAGE[1]), above=False)
        if not res:
            print("  narration: no box fitted")
            continue
        _, box, lines, lh, pad = res
        draw_balloon(page, box, lines, font, NARRATION_STYLE, pad)
        placed_boxes.append(box)
        y_floor = box[1]
        print(f"  narration box {tuple(int(v) for v in box)} [{NARRATION_STYLE}]")

    for c in sorted([x for x in best["cells"] if not x.get("blank")], key=lambda x: x["lo"]):
        idx = c["lo"]
        parts = LINES[idx]
        if not parts:
            continue
        style = STYLES[idx]
        cy0 = cl.bbox_of(c["poly"])[1]
        for si, part in enumerate(parts):
            who = part.split("：", 1)[0]
            token = SPEAKER_TOKEN.get(who, "narr")
            override = ANCHOR_OVERRIDE.get((idx, token))
            if override == "particles":
                tip = particle_anchor(c, sizes[idx], keys[idx])
            else:
                tip = key_tip(c, sizes[idx], keys[idx])
            # the anchor already says who is talking, so the name is not repeated in the
            # balloon -- it would only make every balloon wider
            body = part.split("：", 1)[1] if "：" in part else part
            res = place_text(c["poly"], sizes[idx], keys[idx], body, font,
                             y_floor=0.0, placed=placed_boxes, slot=None,
                             anchor=ANCHORS.get(f"n{idx + 1}_{si}"),
                             diag=math.hypot(PAGE[0], PAGE[1]))
            if not res:
                print(f"  #{idx + 1} [{token}]: no box fitted within the ordering floor")
                continue
            _, box, lines, lh, pad = res
            draw_balloon(page, box, lines, font, style, pad, tip=tip)
            placed_boxes.append(box)
            placements.append((idx, box[1]))
            # no ordering floor any more -- the anchor decides where the balloon goes,
            # and the anchors are already in reading order
            y_floor = 0.0
            print(f"  #{idx + 1} {who} -> box {tuple(int(v) for v in box)} [{style}]")

    seq = [i for i, _ in sorted(placements, key=lambda t: t[1])]
    print(f"  台词框自上而下的次序: {[i + 1 for i in seq]}  "
          f"正确: {seq == sorted(seq)}")
    page.save(OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
