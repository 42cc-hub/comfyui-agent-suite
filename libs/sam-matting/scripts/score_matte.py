"""Score a matte against a hand-cut ground truth, and localise every error.

Use this whenever a matte is claimed to be good. A single number cannot tell "wrong by a
pixel at the edge" from "a whole limb missing", so:

  IoU             intersection over union of the two foreground sets
  FP              matte keeps what the ground truth calls background (over-inclusion)
  FN              matte drops what the ground truth calls foreground (over-removal)
  edge            mean distance from the matte's outline to the ground truth's outline
  clusters        every error blob >= --min-blob px, with area, bbox and centroid, so the
                  failure can be named ("the hair sweep") instead of counted

Masks may be boolean/0-1 .npy or 0-255 alpha images; the ground truth must have alpha.

usage:
  python score_matte.py GT.png matte1.npy matte2.png ...
"""
from __future__ import annotations

import os
import sys

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


def describe_place(bbox, shape) -> str:
    x, y, w, h = bbox
    sx, sy = (x + w / 2) / shape[1], (y + h / 2) / shape[0]
    horiz = "left" if sx < 0.4 else ("right" if sx > 0.6 else "centre")
    vert = "top" if sy < 0.35 else ("bottom" if sy > 0.65 else "middle")
    return f"{vert}-{horiz}"


def report(tag: str, gt: np.ndarray, matte: np.ndarray, min_blob: int = 200) -> dict:
    inter, union = (gt & matte).sum(), (gt | matte).sum()
    fp, fn = matte & ~gt, gt & ~matte
    print(f"\n=== {tag} ===")
    print(f"  GT {100.0*gt.mean():.2f}%  matte {100.0*matte.mean():.2f}%  IoU {inter/max(union,1):.4f}")
    print(f"  over-inclusion (FP) {100.0*fp.mean():.3f}% of frame "
          f"= {100.0*fp.sum()/max(matte.sum(),1):.1f}% of the matte")
    print(f"  over-removal   (FN) {100.0*fn.mean():.3f}% of frame "
          f"= {100.0*fn.sum()/max(gt.sum(),1):.1f}% of the ground truth")
    for label, mask in (("over-inclusion", fp), ("over-removal", fn)):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        blobs = [i for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
                 if stats[i, cv2.CC_STAT_AREA] >= min_blob]
        if not blobs:
            continue
        print(f"  {label} clusters:")
        for i in blobs[:8]:
            bbox = [int(v) for v in stats[i, :4]]
            print(f"    {100.0*stats[i, cv2.CC_STAT_AREA]/mask.size:6.3f}%  bbox={bbox}  "
                  f"({describe_place(bbox, mask.shape)})  centroid=[{cent[i][0]:.0f},{cent[i][1]:.0f}]")
    g_edge = gt & ~cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    m_edge = matte & ~cv2.erode(matte.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dist = cv2.distanceTransform((~g_edge).astype(np.uint8), cv2.DIST_L2, 3)[m_edge] if m_edge.any() else np.array([0.0])
    print(f"  matte outline -> GT outline: mean {dist.mean():.2f}px  p95 {np.percentile(dist,95):.2f}px")
    return {"iou": float(inter / max(union, 1)), "fp_pct": float(100.0 * fp.mean()),
            "fn_pct": float(100.0 * fn.mean()), "edge_mean_px": float(dist.mean())}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    gt_path, mattes = sys.argv[1], sys.argv[2:]
    gt = load_any(gt_path)
    print(f"ground truth {os.path.basename(gt_path)}: {100.0*gt.mean():.2f}% foreground")
    for p in mattes:
        m = load_any(p)
        if m.shape != gt.shape:
            print(f"\n=== {os.path.basename(p)} ===\n  size mismatch {m.shape} vs GT {gt.shape}: "
                  f"resize both to a common frame first")
            continue
        report(os.path.basename(p), gt, m)


if __name__ == "__main__":
    main()
