"""A/B the hand region: control map -> before (no hand keypoints) -> after.

Crops around the projected hand keypoints so the fingers are actually visible at
832x1216 full-body scale. Same seeds in both runs, so any difference comes from
the control map.

usage: python blender_compare_hands.py <scene> <before.png> <after.png> <out.png> [out_dir=./out]
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.abspath("out")
BOX = 150
ZOOM = 3
LABEL_H = 24


def font(size=16):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(img, cx, cy, label):
    box = (int(cx - BOX), int(cy - BOX), int(cx + BOX), int(cy + BOX))
    crop = img.crop(box).resize((BOX * 2 * ZOOM, BOX * 2 * ZOOM), Image.LANCZOS)
    t = Image.new("RGB", (crop.width, crop.height + LABEL_H), (20, 20, 24))
    t.paste(crop, (0, LABEL_H))
    ImageDraw.Draw(t).text((6, 4), label, fill=(235, 235, 240), font=font())
    return t


scene, before_path, after_path, out_path = sys.argv[1:5]
if len(sys.argv) > 5:
    OUT = os.path.abspath(sys.argv[5])
with open(os.path.join(OUT, f"{scene}_kps.json"), "r", encoding="utf-8") as fh:
    data = json.load(fh)

pose = Image.open(os.path.join(OUT, f"{scene}_pose.png")).convert("RGB")
before = Image.open(before_path).convert("RGB")
after = Image.open(after_path).convert("RGB")

rows = []
for key, side in (("hand_left", "L"), ("hand_right", "R")):
    pts = data["keypoints"][key]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    rows.append([
        tile(pose, cx, cy, f"{side} control map"),
        tile(before, cx, cy, f"{side} BEFORE (body-only)"),
        tile(after, cx, cy, f"{side} AFTER (+hand kps)"),
    ])

W = sum(t.width for t in rows[0])
H = sum(row[0].height for row in rows)
canvas = Image.new("RGB", (W, H), (20, 20, 24))
y = 0
for row in rows:
    x = 0
    for t in row:
        canvas.paste(t, (x, y))
        x += t.width
    y += row[0].height
canvas.save(out_path)
print(f"wrote {out_path} {canvas.size}")
