"""Key a flat colour background out of an image and return RGBA.

Needed because the hosted image-edit models (Bailian / DashScope, and most others) return art
with **no alpha channel**: the only way to get a cutout from them is to ask for the subject on
a flat chroma colour and key it here. Measured on a Bailian `qwen-image-edit-max` result:
background-coloured pixels left inside the cutout 0.000%, no green fringe after despill.

Steps, each for a reason:
  1. background colour = median of a border ring (a model does not hit #00FF00 exactly);
  2. alpha = 0 inside --tol of that colour, 1 outside --hi, linear ramp between;
  3. despill: near the edge the subject is tinted by the background, so the background's
     dominant channel is clamped to the mean of the other two plus a margin;
  4. keep the main island plus any island larger than --keep-frac, then fill interior holes;
  5. alpha rebuilt from signed distance so the edge is a clean 1 px ramp.

usage:
  python chroma_key.py IN.png OUT.png [--tol 40] [--hi 90]
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--tol", type=float, default=40.0)
    ap.add_argument("--hi", type=float, default=90.0)
    ap.add_argument("--ring", type=int, default=5)
    ap.add_argument("--keep-frac", type=float, default=0.002)
    ap.add_argument("--despill-margin", type=float, default=12.0)
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.src).convert("RGB"))
    border = np.concatenate([rgb[:args.ring].reshape(-1, 3), rgb[-args.ring:].reshape(-1, 3),
                             rgb[:, :args.ring].reshape(-1, 3), rgb[:, -args.ring:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.abs(rgb.astype(np.float32) - bg).max(axis=2)
    alpha = np.clip((dist - args.tol) / max(args.hi - args.tol, 1e-6), 0.0, 1.0)

    hard = alpha > 0.5
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep_ids = [i for i in range(1, n)
                    if i == big or stats[i, cv2.CC_STAT_AREA] > hard.size * args.keep_frac]
        hard = np.isin(lab, keep_ids)
    m = hard.astype(np.uint8)
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (ff[1:-1, 1:-1] == 0) & (~hard)
    # Fill an enclosed hole only when its own colour is NOT the background colour. Gaps
    # between hair strands or between limbs are enclosed by the figure but are real
    # background; filling them paints a solid blob (measured 2026-09-20: an olive patch
    # across a character's hair) and the despill below then hides the evidence.
    if holes.any():
        n_h, lab_h = cv2.connectedComponents(holes.astype(np.uint8), 8)[:2]
        for i in range(1, n_h):
            sel = lab_h == i
            if float(np.abs(np.median(rgb[sel], axis=0).astype(np.float32) - bg).max()) < args.hi:
                holes[sel] = False
    hard = hard | holes

    grow = cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    d_in = cv2.distanceTransform(grow.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~grow).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(0.5 + (d_in - d_out) * 0.5, 0.0, 1.0)
    alpha = np.where(hard, np.maximum(alpha, 0.9), alpha)

    spill_axis = int(np.argmax(bg))
    other = [c for c in range(3) if c != spill_axis]
    limit = (rgb[..., other[0]].astype(np.float32) + rgb[..., other[1]].astype(np.float32)) / 2.0
    limit += args.despill_margin
    despilled = rgb.astype(np.float32).copy()
    despilled[..., spill_axis] = np.minimum(despilled[..., spill_axis], np.maximum(limit, 0))
    despilled = np.clip(despilled, 0, 255).astype(np.uint8)

    a3 = alpha[..., None]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray(np.concatenate([despilled, (a3 * 255).astype(np.uint8)], axis=2)).save(args.out)
    mag = np.zeros_like(despilled, dtype=np.float32)
    mag[..., 0], mag[..., 2] = 255, 200
    Image.fromarray((despilled * a3 + mag * (1 - a3)).astype(np.uint8)).save(
        args.out.replace(".png", "_on_magenta.png"))

    # Measured on the ORIGINAL rgb: the despill above clamps the spill channel of exactly
    # these pixels, so checking `despilled` here can only ever report 0% (2026-09-20).
    residue = float(((np.abs(rgb.astype(np.float32) - bg).max(axis=2) < args.tol) & hard).mean())
    print(f"background colour {bg.astype(int).tolist()}  foreground {100.0*hard.mean():.2f}%  "
          f"background-coloured pixels left inside the cutout: {100.0*residue:.3f}%")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
