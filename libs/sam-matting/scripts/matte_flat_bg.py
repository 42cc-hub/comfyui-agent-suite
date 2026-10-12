"""Cut the background out of rendered images that were generated on a flat background.

This is the cheap case and it should be preferred whenever the image is ours to control:
if the prompt asked for "plain white background", the render is a single object on a flat
field, and SAM segments it in one shot (measured on this machine: mean proposal score 0.961
over 19 Anima renders, 1-3 islands per image after cleanup, no manual decisions).

Pipeline per image:
  1. background colour = median of a border ring (a render is rarely pure #FFFFFF);
  2. rough foreground = pixels further than --tol from it, largest blob only;
  3. SAM prompt: box = rough bbox, positive points = largest inscribed discs of the rough
     mask (guaranteed-inside prompts, so no clicking);
  4. island cleanup -- keep the main island, plus islands within a few px of it or big
     enough to be a body part, minus islands whose own colour IS the background colour
     (that rule removes white sketch patches SAM picks up inside its box);
  5. fill interior holes, then alpha from signed distance so the edge is a 1 px ramp on the
     lineart rather than a blur.

usage:
  python matte_flat_bg.py "out/**/*.png" --out DIR [--tol 10] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam_object_mask import DEFAULT_CKPT, load_predictor, segment  # noqa: E402


def background_colour(rgb: np.ndarray, ring: int = 6) -> np.ndarray:
    border = np.concatenate([rgb[:ring].reshape(-1, 3), rgb[-ring:].reshape(-1, 3),
                             rgb[:, :ring].reshape(-1, 3), rgb[:, -ring:].reshape(-1, 3)])
    return np.median(border, axis=0)


def rough_foreground(rgb: np.ndarray, bg: np.ndarray, tol: float) -> np.ndarray:
    dist = np.abs(rgb.astype(np.float32) - bg).max(axis=2)
    rough = dist > tol
    rough = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rough.astype(np.uint8), 8)
    if n > 1:
        rough = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return rough


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


def fill_holes(mask: np.ndarray) -> tuple[np.ndarray, float]:
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (ff[1:-1, 1:-1] == 0) & (~mask)
    return mask | holes, float(holes.mean())


def clean_islands(mask: np.ndarray, rgb: np.ndarray, bg: np.ndarray, tol: float,
                  near_px: int = 3, big_frac: float = 0.004, min_frac: float = 0.0002) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = lab == main
    for _ in range(2):
        near = cv2.dilate(keep.astype(np.uint8),
                          np.ones((2 * near_px + 1,) * 2, np.uint8)).astype(bool)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if i == main or area < mask.size * min_frac:
                continue
            sel = lab == i
            if float(np.abs(np.median(rgb[sel], axis=0).astype(np.float32) - bg).max()) < tol:
                continue  # the island is background-coloured: SAM picked up background
            if area >= mask.size * big_frac or sel[near].any():
                keep |= sel
    return keep


def soft_alpha(mask: np.ndarray) -> np.ndarray:
    d_in = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(0.5 + (d_in - d_out) * 0.5, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default=os.environ.get("SAM_CKPT", DEFAULT_CKPT))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(args.glob, recursive=True))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no images matched {args.glob}")

    predictor = load_predictor(args.ckpt)
    rows = []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        bg = background_colour(rgb)
        rough = rough_foreground(rgb, bg, args.tol)
        if rough.sum() < 100:
            print(f"SKIP {os.path.basename(p)}: no foreground found")
            continue
        ys, xs = np.where(rough)
        box = [int(xs.min()) - 4, int(ys.min()) - 4, int(xs.max()) + 4, int(ys.max()) + 4]
        predictor.set_image(rgb)
        mask, score, _, _ = segment(predictor, rgb, box, interior_points(rough), [], iou_target=rough)

        keep = clean_islands(mask, rgb, bg, args.tol)
        keep, hole_frac = fill_holes(keep)
        islands = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)[0] - 1

        a3 = soft_alpha(keep)[..., None]
        name = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(np.concatenate([rgb, (a3 * 255).astype(np.uint8)], axis=2)).save(
            os.path.join(args.out, f"{name}.png"))
        mag = np.zeros_like(rgb, dtype=np.float32)
        mag[..., 0], mag[..., 2] = 255, 200
        Image.fromarray((rgb * a3 + mag * (1 - a3)).astype(np.uint8)).save(
            os.path.join(args.out, f"{name}_on_magenta.png"))

        rows.append({"file": p, "sam_score": score, "fg_pct": 100.0 * float(keep.mean()),
                     "holes_pct": 100.0 * hole_frac, "islands": int(islands)})
        print(f"{name:34s} score={score:.3f} fg={rows[-1]['fg_pct']:5.2f}% "
              f"holes={rows[-1]['holes_pct']:.2f}% islands={islands}")

    json.dump(rows, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    if rows:
        print(f"\n{len(rows)} images; mean score {np.mean([r['sam_score'] for r in rows]):.3f}, "
              f"mean islands {np.mean([r['islands'] for r in rows]):.1f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

