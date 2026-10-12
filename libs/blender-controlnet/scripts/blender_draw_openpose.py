"""Draw a standard COCO-18 OpenPose skeleton from Blender-projected keypoints.

Keypoints come from build_scene.py's <scene>_kps.json (image-space x, y in pixels,
already projected through the render camera), so the skeleton is pixel-aligned with
<scene>_depth.png by construction.

Two keypoint shapes exist and both are accepted, because stage_multi.py emits one
skeleton per character while build_scene.py emits a single flat one. Every skeleton
found is drawn into the same map.
  flat  (build_scene):  {"resolution", "keypoints": {name: [x,y,z], "hand_left": ...}}
  multi (stage_multi):  {"resolution", "characters": [{"name", "keypoints": {...}}]}

usage: python draw_openpose.py <scene>_kps.json <out.png>
"""
from __future__ import annotations

import json
import sys

from PIL import Image, ImageDraw

ORDER = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
]

LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# OpenPose hand skeleton: 21 points = wrist + 4 joints x 5 fingers
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def draw_hand(draw, pts, width, joint_r):
    for idx, (a, b) in enumerate(HAND_EDGES):
        draw.line([tuple(pts[a]), tuple(pts[b])], fill=COLORS[idx % len(COLORS)], width=width)
    for i, p in enumerate(pts):
        draw.ellipse([p[0] - joint_r, p[1] - joint_r, p[0] + joint_r, p[1] + joint_r],
                     fill=COLORS[i % len(COLORS)])


def skeletons(data):
    if "characters" in data:
        return [c["keypoints"] for c in data["characters"]]
    return [data["keypoints"]]


def draw_skeleton(draw, by_name, stick, joint_r, hand_stick, hand_joint_r):
    kps = [by_name[name] for name in ORDER]
    for idx, (a, b) in enumerate(LIMBS):
        draw.line([(kps[a][0], kps[a][1]), (kps[b][0], kps[b][1])],
                  fill=COLORS[idx], width=stick)
    for i, (x, y) in enumerate([(k[0], k[1]) for k in kps]):
        color = COLORS[i] if i < len(COLORS) else (255, 255, 255)
        draw.ellipse([x - joint_r, y - joint_r, x + joint_r, y + joint_r], fill=color)
    for key in ("hand_left", "hand_right"):
        if key in by_name:
            draw_hand(draw, by_name[key], hand_stick, hand_joint_r)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    w, h = data["resolution"]

    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    stick = max(4, int(min(w, h) / 144))
    joint_r = max(3, stick // 2 + 1)
    hand_stick = max(2, stick // 2)
    hand_joint_r = max(2, joint_r - 2)

    groups = skeletons(data)
    with_hands = 0
    for by_name in groups:
        draw_skeleton(draw, by_name, stick, joint_r, hand_stick, hand_joint_r)
        if "hand_left" in by_name:
            with_hands += 1

    img.save(dst)
    print(f"wrote {dst} {w}x{h} stick={stick} skeletons={len(groups)} hands={with_hands}")


if __name__ == "__main__":
    main()
