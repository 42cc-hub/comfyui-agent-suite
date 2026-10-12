"""Overlay the projected OpenPose skeleton on the clay render.

The two images come from the same camera, so any offset here is a projection bug,
not a pose difference. This is the verification step for the whole Blender->ControlNet
chain: if the skeleton does not sit on the mannequin, the depth map and the pose map
would steer the generation in different directions.

usage: python overlay_check.py <clay.png> <pose.png> <out.png>
"""
import sys

from PIL import Image

clay = Image.open(sys.argv[1]).convert("RGB")
pose = Image.open(sys.argv[2]).convert("RGB")
if clay.size != pose.size:
    pose = pose.resize(clay.size)
Image.blend(clay, pose, 0.55).save(sys.argv[3])
print("wrote " + sys.argv[3])
