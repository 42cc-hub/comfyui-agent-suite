"""从 .blend 读回人调过的相机/角色/骨骼/道具位置，写出一份可直接重渲染的 spec。

人在 Blender GUI 里拖动 → Ctrl+S → agent 跑这个脚本 → 得到新 spec → 重出控制图。
骨骼方向按 armature 空间读（与 apply_pose 的约定一致），因此整体拖动/旋转角色也能正确回读。

usage:
  blender.exe --background --python extract_blend.py -- \
      --blend battle2.blend --spec scenes_battle2.json --out scenes_from_blend.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
from mathutils import Vector

FINGER_PREFIX = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")


def is_finger(name: str) -> bool:
    return name.split(".")[0] in FINGER_PREFIX


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    blend = os.path.abspath(argv[argv.index("--blend") + 1])
    spec_path = os.path.abspath(argv[argv.index("--spec") + 1])
    out_path = os.path.abspath(argv[argv.index("--out") + 1])

    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = json.load(fh)

    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise SystemExit("no camera in the .blend")

    loc = cam.matrix_world.translation
    fwd = (cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    target = loc + fwd * 5.0

    sc = spec["scenes"][0]
    sc["camera"] = {"location": [round(v, 4) for v in loc], "target": [round(v, 4) for v in target]}
    sc["lens"] = round(float(cam.data.lens), 3)
    sc["resolution"] = [int(scene.render.resolution_x), int(scene.render.resolution_y)]
    print(f"[cam] loc={tuple(round(v,3) for v in loc)} lens={sc['lens']} res={sc['resolution']}")

    for c in spec["characters"]:
        ob = bpy.data.objects.get("rig_" + c["name"])
        if ob is None:
            print(f"[warn] rig_{c['name']} not found")
            continue
        c["root"] = [round(v, 4) for v in ob.location]
        c["root_rot"] = [round(math.degrees(a), 3) for a in ob.rotation_euler]
        bones = {}
        for pb in ob.pose.bones:
            if is_finger(pb.name):
                continue                      # 手指由 hands.curl 驱动，不回写
            d = pb.tail - pb.head
            if d.length < 1e-6:
                continue
            bones[pb.name] = [round(v, 5) for v in d.normalized()]
        c["bones"] = bones
        print(f"[rig] {c['name']} root={c['root']} rot={c['root_rot']} bones={len(bones)}")

    boxes = []
    for ob in sorted(scene.objects, key=lambda o: o.name):
        if ob.name.startswith("env_box_"):
            boxes.append({"loc": [round(v, 4) for v in ob.location],
                          "scale": [round(v, 4) for v in ob.scale]})
    if boxes:
        spec["env"]["boxes"] = boxes
        print(f"[env] {len(boxes)} boxes read back")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=1)
    print("[done] wrote", out_path)


if __name__ == "__main__":
    main()
