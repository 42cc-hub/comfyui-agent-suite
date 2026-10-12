"""Draw a keypoint set on top of an image, to judge a detector's output in place.

usage: python draw_kps_on_image.py <image> <kps.json> <out.png> [scale]
"""
import json
import sys

from PIL import Image, ImageDraw

ORDER = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
         "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle",
         "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
          (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

img = Image.open(sys.argv[1]).convert("RGB")
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
if scale != 1.0:
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
with open(sys.argv[2], "r", encoding="utf-8-sig") as fh:
    d = json.load(fh)
kps = [d["keypoints"][n] for n in ORDER]

draw = ImageDraw.Draw(img)
w = max(2, int(min(img.size) / 200))
for i, (a, b) in enumerate(LIMBS):
    draw.line([tuple(kps[a]), tuple(kps[b])], fill=COLORS[i], width=w)
for i, p in enumerate(kps):
    r = w
    draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=COLORS[i % len(COLORS)])
img.save(sys.argv[3])
print("wrote", sys.argv[3], img.size)
