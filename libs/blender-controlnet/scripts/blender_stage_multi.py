"""Multi-character staging on top of build_scene.py.

build_scene.py stages ONE mannequin. A two-character tableau needs three things it
does not have:

  - N armatures + N mannequins, each with its own root offset / pose / facing
  - props rigidly attached to a bone (rifle in hand, pistol on belt), so the depth
    map actually shows the long objects the prompt describes
  - per-character keypoints, so draw_openpose can emit a multi-person skeleton

The rig, the FK (aim_bone), the projection and the depth material are all reused
from build_scene, so there is exactly one implementation of the geometry.

Spec shape:
  {
    "characters": [{"name","root","bones","hands","facing"}],
    "props":      [{"owner","bone","offset","rot","size","name"}],
    "scenes":     [{"name","resolution","lens","camera","env","characters?","props?"}]
  }
A scene may override `characters` / `props` for its own frame.

Run:
  blender.exe --background --factory-startup --python stage_multi.py -- \
      --spec scenes_battle.json --out out
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # blender_build_scene.py 与本脚本同目录
import blender_build_scene as bs  # noqa: E402


def add_bone_prop(spec, arm):
    """A box rigidly attached to a pose bone (local axes: Y along the bone)."""
    pb = arm.pose.bones[spec["bone"]]
    base = arm.matrix_world @ pb.matrix
    off = Matrix.Translation(Vector(spec.get("offset", (0.0, 0.0, 0.0))))
    rot = Euler([math.radians(a) for a in spec.get("rot", (0.0, 0.0, 0.0))], "XYZ").to_matrix().to_4x4()
    scl = Matrix.Diagonal(Vector(tuple(spec.get("size", (0.1, 0.1, 0.1))) + (1.0,)))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    ob = bpy.context.active_object
    ob.name = spec.get("name", "prop")
    ob.matrix_world = base @ off @ rot @ scl
    return ob


def make_emission(color):
    """纯色自发光材质：用来渲染剪影 mask（白=目标，黑=其余）。"""
    key = "M_emis_%s" % ("w" if color > 0.5 else "b")
    mat = bpy.data.materials.get(key)
    if mat:
        return mat
    mat = bpy.data.materials.new(key)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (color, color, color, 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def clear_objs(objs):
    for ob in objs:
        mesh = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    spec_path = os.path.abspath(argv[argv.index("--spec") + 1] if "--spec" in argv else "scenes.json")
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else "out"
    save_blend = argv[argv.index("--save-blend") + 1] if "--save-blend" in argv else None
    render_masks = "--masks" in argv
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(spec_path), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = json.load(fh)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    try:
        scene.eevee.taa_render_samples = 16
    except Exception:
        pass
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    # ---- 角色（只建一次）-----------------------------------------------------
    arms, bodies, body_of = {}, [], {}
    for c in spec["characters"]:
        arm = bs.build_armature()
        arm.name = "rig_" + c["name"]
        body = bs.build_mannequin(arm)
        body.name = "body_" + c["name"]
        arms[c["name"]] = arm
        bodies.append(body)
        body_of[c["name"]] = body
    bs.setup_lights()

    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    scene.world = world

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    clay_mat = bs.make_clay_material()
    summary = []

    for sc in spec["scenes"]:
        env_objs = bs.build_env(sc.get("env", spec.get("env", {"boxes": []})))
        chars = sc.get("characters", spec["characters"])
        props_spec = sc.get("props", spec.get("props", []))

        for c in chars:
            arm = arms[c["name"]]
            bs.reset_pose(arm)
            arm.location = Vector(c.get("root", (0.0, 0.0, 0.0)))
            rot = c.get("root_rot")
            arm.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ") if rot else Euler((0.0, 0.0, 0.0))
            bs.apply_pose(arm, c.get("bones", {}))
            bs.apply_hands(arm, c.get("hands", {}))
        bpy.context.view_layer.update()

        props = [add_bone_prop(p, arms[p["owner"]]) for p in props_spec]
        # 道具跟着角色走：父级设为骨架，人在 Blender 里整体拖动角色时道具不脱节
        for p, ob in zip(props_spec, props):
            ob.parent = arms[p["owner"]]
            ob.matrix_parent_inverse = arms[p["owner"]].matrix_world.inverted()
        all_objs = bodies + env_objs + props

        res = tuple(sc.get("resolution", [1216, 832]))
        scene.render.resolution_x, scene.render.resolution_y = res
        scene.render.resolution_percentage = 100
        cam_data.lens = sc.get("lens", 35.0)
        cam_data.sensor_width = 36.0
        cam.location = Vector(sc["camera"]["location"])
        bs.look_at(cam, sc["camera"]["target"])
        bpy.context.view_layer.update()

        # 深度范围锚定在「角色+道具」上（地板不参与，否则人物被压成中灰）
        near, far = bs.world_bounds_z(cam, bodies + props)
        near, far = max(0.05, near - 0.5), far + 1.5
        cam_data.clip_start, cam_data.clip_end = max(0.01, near), far + 1.0

        bs.assign_material(all_objs, bs.make_depth_material(near, far))
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_depth.png")
        bpy.ops.render.render(write_still=True)

        bs.assign_material(all_objs, clay_mat)
        scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_clay.png")
        bpy.ops.render.render(write_still=True)

        # 逐角色剪影 mask：白=该角色（含其道具），黑=其余。局部重绘要用它当先验。
        if render_masks:
            white, black = make_emission(1.0), make_emission(0.0)
            for c in chars:
                mine = [body_of[c["name"]]] + [ob for p, ob in zip(props_spec, props) if p["owner"] == c["name"]]
                bs.assign_material([ob for ob in all_objs if ob not in mine], black)
                bs.assign_material(mine, white)
                scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_mask_{c['name']}.png")
                bpy.ops.render.render(write_still=True)
            # 环境物（沙袋墙等）单独一张：场景元素也要能局部重绘。地板不算——
            # 地板投影覆盖整个下半幅，混进来会把地面也一起重绘。
            boxes_only = [ob for ob in env_objs if ob.name.startswith("env_box_")]
            bs.assign_material(all_objs, black)
            bs.assign_material(boxes_only, white)
            scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_mask_env.png")
            bpy.ops.render.render(write_still=True)
            bs.assign_material(all_objs, clay_mat)

        kps = {"resolution": list(res), "characters": []}
        for c in chars:
            kps["characters"].append({
                "name": c["name"],
                "keypoints": bs.collect_keypoints(arms[c["name"]], cam, res,
                                                  facing=tuple(c.get("facing", (0.0, -1.0, 0.0)))),
            })
        with open(os.path.join(out_dir, f"{sc['name']}_kps.json"), "w", encoding="utf-8") as fh:
            json.dump(kps, fh, indent=1)

        summary.append({"name": sc["name"], "resolution": list(res),
                        "characters": [c["name"] for c in chars], "props": len(props)})
        print(f"[stage] {sc['name']}  {res[0]}x{res[1]}  lens={cam_data.lens}  "
              f"chars={[c['name'] for c in chars]}  props={len(props)}  z=[{near:.2f},{far:.2f}]")

        # 最后一帧若要把 .blend 交给人在 GUI 里拖，就不要清掉对象
        if not (save_blend and sc is spec["scenes"][-1]):
            clear_objs(props)
            bs.clear_env(env_objs)

    if save_blend:
        scene.render.filepath = os.path.join(out_dir, f"{spec['scenes'][-1]['name']}_clay.png")
        note = bpy.data.texts.new("README_给agent")
        note.write(
            "人在 Blender 里怎么调：\n"
            "1. 选 cam（相机）按 G 移动 / 按 R 旋转 / 滚轮缩放视图；按小键盘 0 切相机视角\n"
            "2. 选 rig_keqing / rig_barbara 整体拖动角色（道具会跟着走）\n"
            "3. 选 env_box_* 移动沙袋墙\n"
            "4. 想改姿势就进 Pose Mode 旋转骨骼（可选）\n"
            "5. 调完 Ctrl+S 保存这个 .blend，然后回对话告诉我\n"
            "agent 会从 .blend 里读回相机/角色/骨骼，重新出控制图与生图。\n"
        )
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(save_blend))
        print(f"[blend] saved {os.path.abspath(save_blend)}")

    print("[done] " + json.dumps(summary))


if __name__ == "__main__":
    main()
