"""Extract the 18 COCO keypoints from a reference OpenPose skeleton image.

The reference draws limbs at 60% palette brightness and joint dots at full
brightness, so the full-brightness pixels of each palette colour isolate one dot.

usage: python extract_kps.py <image> <out.json> [--overlay <png>]
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PALETTE = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]
ORDER = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
         "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle",
         "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]

img = Image.open(sys.argv[1]).convert("RGB")
rgb = np.asarray(img)

results = {}
report = []
for i, c in enumerate(PALETTE):
    mask = (rgb == np.array(c, dtype=np.uint8)).all(axis=-1)
    n = int(mask.sum())
    if n == 0:
        report.append(f"  {ORDER[i]:15s} {c} MISSING")
        continue
    lab, k = ndimage.label(mask)
    if k > 1:
        sizes = ndimage.sum(mask, lab, range(1, k + 1))
        keep = int(np.argmax(sizes)) + 1
    else:
        keep = 1
    ys, xs = np.where(lab == keep)
    cx, cy = float(xs.mean()), float(ys.mean())
    results[ORDER[i]] = [round(cx, 1), round(cy, 1)]
    report.append(f"  {ORDER[i]:15s} {c} px={n:4d} comps={k} bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")

print("extracted joints:")
print("\n".join(report))

with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump({"resolution": list(img.size), "keypoints": results}, fh, indent=1)
print("wrote", sys.argv[2])

if "--overlay" in sys.argv:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, name in enumerate(ORDER):
        if name not in results:
            continue
        x, y = results[name]
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255), width=3)
        d.text((x + 8, y - 20), name[:6], fill=(255, 255, 255))
    out.save(sys.argv[sys.argv.index("--overlay") + 1])
    print("wrote overlay")
