"""Convert a comfyui_controlnet_aux DWPose/OpenPose JSON into our 18-point format.

The preprocessor's `pose_keypoints_2d` is already COCO-18 in the same order the
Blender pipeline uses, so this is a straight repack -- no colour extraction needed
when the pose comes from ComfyUI itself. Confidence is carried through: detectors
routinely fail on one limb (a raised leg hidden behind a tutu, an occluded hand)
and emit a garbage coordinate with score 0, which would otherwise dominate any
downstream fit.

usage: python dwpose_to_kps.py <dsh_pose_photo_00003.json> <out.json>
"""
import json
import sys

ORDER = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
         "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle",
         "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]

with open(sys.argv[1], "r", encoding="utf-8-sig") as fh:
    data = json.load(fh)
rec = data[0]
person = rec["people"][0]
flat = person["pose_keypoints_2d"]

kps, conf = {}, {}
for i, name in enumerate(ORDER):
    x, y, c = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
    kps[name] = [round(float(x), 1), round(float(y), 1)]
    conf[name] = round(float(c), 3)

low = [f"{n}={conf[n]:.2f}" for n in ORDER if conf[n] < 0.5]
print(f"canvas {rec['canvas_width']}x{rec['canvas_height']}  people={len(rec['people'])}")
print("low-confidence joints:", ", ".join(low) if low else "none")
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump({"resolution": [rec["canvas_width"], rec["canvas_height"]],
               "keypoints": kps, "confidence": conf}, fh, indent=1)
print("wrote", sys.argv[2])
