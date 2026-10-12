"""Zoom into the hand regions of the pose map + clay render to verify fingers.

Hands are ~20 px wide in a full-body 832x1216 frame, so the only way to actually
check the finger skeleton is to crop and upscale around the wrist keypoints.

usage: python blender_crop_hands.py <scene> [out_dir=./out]
"""
import json
import os
import sys

from PIL import Image

OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "out")
scene = sys.argv[1]
BOX = 130  # half-size of the crop window in px

with open(os.path.join(OUT, f"{scene}_kps.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)

tiles = []
for key, label in (("hand_left", "L"), ("hand_right", "R")):
    pts = data["keypoints"][key]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    box = (int(cx - BOX), int(cy - BOX), int(cx + BOX), int(cy + BOX))
    for src, tag in ((f"{scene}_pose.png", "pose"), (f"{scene}_clay.png", "clay")):
        img = Image.open(os.path.join(OUT, src)).convert("RGB")
        crop = img.crop(box).resize((BOX * 4, BOX * 4), Image.LANCZOS)
        tiles.append((crop, f"{label}/{tag}"))

W = sum(t[0].width for t in tiles)
H = max(t[0].height for t in tiles)
canvas = Image.new("RGB", (W, H), (20, 20, 24))
x = 0
for img, _ in tiles:
    canvas.paste(img, (x, 0))
    x += img.width
canvas.save(os.path.join(OUT, f"{scene}_hands_zoom.png"))
print(f"wrote {scene}_hands_zoom.png {canvas.size}")
