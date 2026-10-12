"""Matte a busy illustration (poster, official art) using a rough human scribble + SAM.

Why a human scribble is worth asking for: on a poster the figure is not the only object.
The human's painted silhouette already answers the semantic question "which islands are the
character", and it is the input the human can produce in seconds. Measured on a real poster
against a hand-cut ground truth: the scribble's island decisions agreed with the ground truth
on **every** island (7/7), while three hand-written heuristics tried in the same session --
"keep only the largest island", "islands right of x=500 are flames", "flames are the pale
islands" -- each deleted real body parts (the long hair sweep, the lower leg, the red wing
ornament, 0.55-1.9% of the frame each). Read the reference for the full post-mortem.

Inputs: the image, plus two PNG masks of the same size (any colour, alpha > 127 = painted):
  --fg  scribble over the subject
  --bg  scribble over the background (optional; improves the SAM box)
If a mask is unavailable, --box/--pos let you prompt SAM by hand instead.

Decision rule (deliberately the human's, not a new heuristic):
  keep = islands whose SAM mask is >= --painted-covered of the fg scribble
       (+ interior holes of the largest such island)

usage:
  python matte_from_scribbles.py poster.png fg.png bg.png --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam_object_mask import DEFAULT_CKPT, load_predictor, parse_points, segment  # noqa: E402


def as_bool_png(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))[..., 3] > 127


def interior_points(mask: np.ndarray, n: int = 6) -> list[tuple[int, int]]:
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    pts, work = [], d.copy()
    for _ in range(n):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] < 4:
            break
        pts.append((int(x), int(y)))
        cv2.circle(work, (int(x), int(y)), int(work[y, x] * 1.6), 0, -1)
    return pts


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return mask | ((ff[1:-1, 1:-1] == 0) & (~mask))


def soft_alpha(mask: np.ndarray) -> np.ndarray:
    d_in = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(0.5 + (d_in - d_out) * 0.5, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--fg", required=True, help="scribble PNG over the subject")
    ap.add_argument("--bg", default="", help="optional scribble PNG over the background")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--pick", type=int, default=None,
                    help="force a SAM proposal index instead of the automatic choice")
    ap.add_argument("--painted-covered", type=float, default=0.5,
                    help="island kept if this share of it lies inside the fg scribble")
    ap.add_argument("--subtract", default="", help="npy masks to subtract (e.g. a sword)")
    ap.add_argument("--ckpt", default=os.environ.get("SAM_CKPT", DEFAULT_CKPT))
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    fg_scribble = as_bool_png(args.fg)
    bg_scribble = as_bool_png(args.bg) if args.bg else None

    ys, xs = np.where(fg_scribble)
    box = [int(xs.min()) - 6, int(ys.min()) - 6, int(xs.max()) + 6, int(ys.max()) + 6]
    predictor = load_predictor(args.ckpt)
    predictor.set_image(rgb)
    pos = interior_points(fg_scribble)
    mask, score, masks, scores = segment(predictor, rgb, box, pos, [], iou_target=fg_scribble)
    # Every proposal is printed: the automatic choice can be too large when the scribble
    # covers props along with the figure (the scribble is a rough upper bound, so a proposal
    # far bigger than it is suspicious). Re-run with --pick to override.
    scribble_area = float(fg_scribble.mean())
    print(f"scribble covers {100.0*scribble_area:.2f}% of the frame; proposals:")
    for i, (m, s) in enumerate(zip(masks, scores)):
        inter = float((m & fg_scribble).sum())
        union = float((m | fg_scribble).sum())
        tag = "   <- auto" if np.array_equal(m, mask) else ""
        print(f"  {i}: score={s:.4f} area={100.0*m.mean():6.2f}% "
              f"({m.mean()/max(scribble_area,1e-6):4.2f}x scribble) "
              f"IoU(scribble)={inter/max(union,1):.3f}{tag}")
    if args.pick is not None:
        mask, score = masks[args.pick], float(scores[args.pick])
        print(f"overridden with proposal {args.pick}")
    print(f"using: score={score:.4f} area={100.0*mask.mean():.2f}%")

    for extra in filter(None, args.subtract.split(",")):
        sub = np.load(extra).astype(bool)
        mask = mask & ~sub
        print(f"subtracted {os.path.basename(extra)} (100.0*sub.mean()={100.0*sub.mean():.2f}%)")

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    total = mask.size
    keep = np.zeros_like(mask)
    kept = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < total * 0.0002:
            continue
        share = float(fg_scribble[lab == i].mean())
        flag = "keep" if share >= args.painted_covered else "drop"
        kept.append((i, area, share, flag))
        if flag == "keep":
            keep |= lab == i
    print(f"{'id':>4} {'area%':>6} {'in-scribble':>11}  verdict")
    for i, area, share, flag in sorted(kept, key=lambda r: -r[1]):
        print(f"{i:>4} {100.0*area/total:6.2f} {share:11.2f}  {flag}")
    if kept:
        biggest = max((r for r in kept if r[3] == "keep"), key=lambda r: r[1], default=None)
        if biggest:
            keep |= fill_holes(lab == biggest[0]) & ~keep

    os.makedirs(args.out, exist_ok=True)
    a3 = soft_alpha(keep)[..., None]
    stem = os.path.splitext(os.path.basename(args.image))[0]
    Image.fromarray(np.concatenate([rgb, (a3 * 255).astype(np.uint8)], axis=2)).save(
        os.path.join(args.out, f"{stem}_cutout.png"))
    mag = np.zeros_like(rgb, dtype=np.float32)
    mag[..., 0], mag[..., 2] = 255, 200
    Image.fromarray((rgb * a3 + mag * (1 - a3)).astype(np.uint8)).save(
        os.path.join(args.out, f"{stem}_on_magenta.png"))
    np.save(os.path.join(args.out, f"{stem}_mask.npy"), keep)

    # Colour-coded island map: makes a wrong keep/drop decision visible instead of numerical.
    pal = [(230, 60, 60), (60, 200, 90), (70, 130, 240), (240, 200, 60), (200, 80, 220),
           (60, 220, 220), (250, 140, 40), (150, 150, 150)]
    vis = np.zeros((*mask.shape, 3), np.uint8)
    for rank, (i, area, share, flag) in enumerate(sorted(kept, key=lambda r: -r[1])[:len(pal)]):
        vis[lab == i] = pal[rank] if flag == "keep" else (70, 70, 70)
    Image.fromarray(vis).save(os.path.join(args.out, f"{stem}_islands.png"))
    print(f"final {100.0*keep.mean():.2f}% of frame; wrote {args.out}")


if __name__ == "__main__":
    main()


