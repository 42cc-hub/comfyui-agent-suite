"""Draw a matte against a ground truth (or another matte) so the disagreement is visible.

Numbers hide the interesting part: whether the missing 6% is one limb or scattered pixels one
pixel wide. This renders:
  green outline = ground truth boundary
  magenta line  = matte boundary
  red fill      = matte missed this (ground truth foreground, matte background)
  blue fill     = matte added this (matte foreground, ground truth background)

usage:
  python compare_matte.py OUT.png BASE.png GT.png MATTE
BASE is the image the outlines are drawn on (usually the original artwork).
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def as_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    return mask > (1 if mask.max() <= 1 else 127)


def load_any(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return as_bool(np.load(path))
    im = Image.open(path)
    if im.mode == "RGBA":
        return as_bool(np.asarray(im)[..., 3])
    return as_bool(np.asarray(im.convert("L")))


def outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("base", help="image the outlines are drawn on (usually the artwork)")
    ap.add_argument("gt")
    ap.add_argument("matte")
    a = ap.parse_args()

    base = np.asarray(Image.open(a.base).convert("RGB")).astype(np.float32)
    gt, mt = load_any(a.gt), load_any(a.matte)
    if gt.shape != mt.shape:
        raise SystemExit(f"size mismatch: GT {gt.shape} vs matte {mt.shape}")

    missed, added = gt & ~mt, mt & ~gt
    vis = base * 0.62
    vis[missed] = vis[missed] * 0.25 + np.array([255, 30, 30]) * 0.75
    vis[added] = vis[added] * 0.25 + np.array([40, 110, 255]) * 0.75
    vis[outline(gt)] = np.array([0, 255, 90])
    vis[outline(mt)] = np.array([255, 0, 230])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(a.out)
    print(f"wrote {a.out}  missed {100.0*missed.mean():.2f}% (red)  "
          f"added {100.0*added.mean():.2f}% (blue)")


if __name__ == "__main__":
    main()
