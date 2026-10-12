"""Blender scene builder for ControlNet-guided generation.

Builds a rigid-segment mannequin skinned to a hand-authored armature, applies a
pose from a JSON spec, then renders per scene:

  <name>_depth.png   camera-space depth, near = white (emission material driven by
                     ShaderNodeCameraData "View Z Depth" -> MapRange; no compositor,
                     no Z-pass plumbing, so it survives compositor API churn)
  <name>_clay.png    flat shaded preview, for human verification of the pose
  <name>_kps.json    COCO-18 keypoints projected to image space

Invariant: the depth render and the keypoint projection share one camera and one
resolution, so the depth map and the OpenPose skeleton can never disagree.

Run:
  blender.exe --background --factory-startup --python build_scene.py -- \
      --spec scenes.json --out <dir>
"""
from __future__ import annotations

import json
import math
import os
import sys

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

# ---------------------------------------------------------------- rig definition
# Character faces -Y, so "person's left" is +X (up x forward = Z x -Y = +X).
# That keeps .L bones on the OpenPose left_* keypoints without mirroring later.
_BONES = [
    # name, head, tail, parent, connect
    ("hips", (0.0, 0.0, 0.95), (0.0, 0.0, 1.05), None, False),
    ("spine", (0.0, 0.0, 1.05), (0.0, 0.0, 1.22), "hips", True),
    ("chest", (0.0, 0.0, 1.22), (0.0, 0.0, 1.42), "spine", True),
    ("neck", (0.0, 0.0, 1.42), (0.0, 0.0, 1.53), "chest", True),
    ("head", (0.0, 0.0, 1.53), (0.0, 0.0, 1.74), "neck", True),
    ("shoulder.L", (0.02, 0.0, 1.40), (0.17, 0.0, 1.40), "chest", False),
    ("upper_arm.L", (0.17, 0.0, 1.40), (0.17, 0.0, 1.14), "shoulder.L", True),
    ("forearm.L", (0.17, 0.0, 1.14), (0.17, 0.0, 0.90), "upper_arm.L", True),
    ("hand.L", (0.17, 0.0, 0.90), (0.17, 0.0, 0.80), "forearm.L", True),
    ("thigh.L", (0.09, 0.0, 0.93), (0.09, 0.0, 0.52), "hips", False),
    ("shin.L", (0.09, 0.0, 0.52), (0.09, 0.0, 0.10), "thigh.L", True),
    ("foot.L", (0.09, 0.0, 0.10), (0.09, -0.13, 0.03), "shin.L", True),
]
# mirror to the right side
for _n, _h, _t, _p, _c in list(_BONES):
    if _n.endswith(".L"):
        _BONES.append((
            _n[:-2] + ".R",
            (-_h[0], _h[1], _h[2]),
            (-_t[0], _t[1], _t[2]),
            (_p[:-2] + ".R") if _p and _p.endswith(".L") else _p,
            _c,
        ))

# segment radius per bone -> capsule body. Rigid segments read as a wooden
# mannequin, which is exactly what a depth ControlNet wants: clean volumes.
_RADII = {
    "hips": 0.135, "spine": 0.120, "chest": 0.140, "neck": 0.045, "head": 0.100,
    "shoulder": 0.055, "upper_arm": 0.048, "forearm": 0.040, "hand": 0.030,
    "thigh": 0.075, "shin": 0.058, "foot": 0.045,
    "thumb": 0.011, "f_index": 0.010, "f_middle": 0.010, "f_ring": 0.0095, "f_pinky": 0.0085,
}

# Fingers get REAL bones rather than being derived analytically: the mesh, the
# depth render and the OpenPose hand skeleton then share one source of truth, so a
# curled hand shows up identically in all three. Rest pose = straight, fingers down.
_FINGERS = {
    # name: (across-palm offset in m, + = thumb side, phalanx lengths)
    "thumb": (0.038, (0.032, 0.026, 0.020)),
    "f_index": (0.022, (0.033, 0.021, 0.017)),
    "f_middle": (0.007, (0.036, 0.024, 0.018)),
    "f_ring": (-0.008, (0.032, 0.023, 0.017)),
    "f_pinky": (-0.022, (0.026, 0.017, 0.014)),
}

# flexion per phalanx in degrees at curl=1.0 (a full fist)
_HAND_CURL = {
    "thumb": (28.0, 24.0, 20.0),
    "f_index": (85.0, 100.0, 70.0),
    "f_middle": (85.0, 105.0, 75.0),
    "f_ring": (85.0, 100.0, 70.0),
    "f_pinky": (85.0, 95.0, 65.0),
}


def _is_finger(name: str) -> bool:
    return name.split(".")[0] in _FINGERS


def _finger_bone_defs():
    """Finger chains for both hands, in armature rest space.

    Both thumbs point toward the character's front (-Y) and the palm normal is
    -X*s (palm faces the body), which is anatomically right for arms at rest.
    """
    defs = []
    for side in ("L", "R"):
        s = 1.0 if side == "L" else -1.0
        wrist = Vector((0.17 * s, 0.0, 0.90))
        fwd = Vector((0.0, 0.0, -1.0))
        across = Vector((0.0, -1.0, 0.0))
        for fname, (off, lengths) in _FINGERS.items():
            if fname == "thumb":
                head = wrist + fwd * 0.030 + across * off
                d = (across * 0.85 + fwd * 0.45).normalized()
            else:
                head = wrist + fwd * 0.085 + across * off
                d = fwd
            parent = f"hand.{side}"
            for i, length in enumerate(lengths, start=1):
                tail = head + d * length
                defs.append((f"{fname}.0{i}.{side}", tuple(head), tuple(tail), parent, i > 1))
                head, parent = tail, f"{fname}.0{i}.{side}"
    return defs


_BONES.extend(_finger_bone_defs())


def _radius_for(name: str) -> float:
    return _RADII.get(name.split(".")[0], 0.06)


def _create_cone(bm, r1, r2, depth, matrix, segments=10):
    try:
        return bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                                     radius1=r1, radius2=r2, depth=depth, matrix=matrix)
    except TypeError:  # pre-3.0 naming
        return bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                                     diameter1=r1 * 2, diameter2=r2 * 2, depth=depth, matrix=matrix)


def _create_sphere(bm, radius, matrix, u=12, v=8):
    try:
        return bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius, matrix=matrix)
    except TypeError:
        return bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, diameter=radius * 2, matrix=matrix)


def _segment_matrix(p0, p1):
    """Local +Z maps onto p0->p1; translation at the midpoint. Returns (matrix, length)."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    length = d.length
    z = d.normalized()
    up = Vector((0.0, 0.0, 1.0))
    if abs(z.dot(up)) > 0.99:
        up = Vector((0.0, 1.0, 0.0))
    x = up.cross(z).normalized()
    y = z.cross(x).normalized()
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = (p0 + p1) * 0.5
    return m, length


def build_armature():
    arm_data = bpy.data.armatures.new("rig")
    arm_obj = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for name, head, tail, parent, connect in _BONES:
        eb = arm_data.edit_bones.new(name)
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        created[name] = eb
    for name, head, tail, parent, connect in _BONES:
        if parent:
            created[name].parent = created[parent]
            created[name].use_connect = connect
    # Roll finger bones so local +X is the flexion axis (local Z ~= palm normal);
    # apply_hands then only has to set rotation_euler.x.
    for name, eb in created.items():
        if _is_finger(name):
            s = 1.0 if name.endswith(".L") else -1.0
            eb.align_roll(Vector((-s, 0.0, 0.0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def build_mannequin(arm_obj):
    bm = bmesh.new()
    groups: dict[str, list] = {}

    def tag(verts, group):
        groups.setdefault(group, []).extend(verts)

    for name, head, tail, parent, connect in _BONES:
        r = _radius_for(name)
        if name.split(".")[0] == "head":
            # a single ellipsoid reads far better than a capsule + skull ball
            c = (Vector(head) + Vector(tail)) * 0.5
            m = Matrix.Translation(c) @ Matrix.Diagonal(Vector((0.085, 0.098, 0.118, 1.0)))
            tag(_create_sphere(bm, 1.0, m, u=16, v=12)["verts"], name)
            continue
        m, length = _segment_matrix(head, tail)
        # taper limbs slightly toward the far end for a less blocky silhouette
        r2 = r * (0.85 if name.split(".")[0] in ("upper_arm", "forearm", "thigh", "shin") else 1.0)
        res = _create_cone(bm, r, r2, max(length, 0.02), m)
        tag(res["verts"], name)
        # joint ball on the head end keeps the silhouette continuous when posed
        ball = _create_sphere(bm, r * 1.02, Matrix.Translation(Vector(head)))
        tag(ball["verts"], name)

    bm.verts.index_update()
    idx_by_group = {g: [v.index for v in verts] for g, verts in groups.items()}

    me = bpy.data.meshes.new("mannequin")
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new("mannequin", me)
    bpy.context.scene.collection.objects.link(obj)
    for group, idx in idx_by_group.items():
        vg = obj.vertex_groups.new(name=group)
        vg.add(idx, 1.0, "REPLACE")

    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    obj.parent = arm_obj
    return obj


def build_env(env):
    """Floor + boxes. Plain boxes are enough: ControlNet only needs the depth
    structure, and a minimal set overconstrains the generation less."""
    objs = []
    if env.get("floor", True):
        bpy.ops.mesh.primitive_plane_add(size=env.get("floor_size", 24.0), location=(0, 0, 0))
        objs.append(bpy.context.active_object)
    for i, box in enumerate(env.get("boxes", [])):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=box["loc"])
        ob = bpy.context.active_object
        ob.scale = box.get("scale", [1, 1, 1])
        ob.name = f"env_box_{i}"
        objs.append(ob)
    for ob in objs:
        ob.name = ob.name.replace(".001", "")
    return objs


def clear_env(env_objs):
    for ob in env_objs:
        mesh = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_depth_material(near, far):
    mat = bpy.data.materials.get("M_depth") or bpy.data.materials.new("M_depth")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    cam = nt.nodes.new("ShaderNodeCameraData")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = near
    mr.inputs["From Max"].default_value = far
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    mr.clamp = True
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(cam.outputs["View Z Depth"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_clay_material():
    mat = bpy.data.materials.get("M_clay") or bpy.data.materials.new("M_clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.58, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.75
    return mat


def assign_material(objs, mat):
    for ob in objs:
        ob.data.materials.clear()
        ob.data.materials.append(mat)


def setup_lights():
    for kind, loc, energy, rot in (
        ("SUN", (3, -4, 6), 4.0, (math.radians(50), 0, math.radians(35))),
        ("AREA", (-3, -3, 3), 300.0, (math.radians(65), 0, math.radians(-50))),
    ):
        light_data = bpy.data.lights.new(f"L_{kind}", type=kind)
        light_data.energy = energy
        if kind == "AREA":
            light_data.size = 4.0
        ob = bpy.data.objects.new(f"L_{kind}", light_data)
        ob.location = loc
        ob.rotation_euler = rot
        bpy.context.scene.collection.objects.link(ob)


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def aim_bone(arm_obj, name, target_dir, roll_hint=(0.0, -1.0, 0.0)):
    """Point a bone along a world-space direction, preserving its head position.

    Authoring poses as world directions (not local Eulers) keeps the spec readable
    and independent of bone roll; Blender solves the local basis from pose_bone.matrix.
    """
    pb = arm_obj.pose.bones[name]
    bpy.context.view_layer.update()
    head = pb.head.copy()
    y = Vector(target_dir).normalized()
    z = Vector(roll_hint)
    z = z - y * z.dot(y)
    if z.length < 1e-4:
        z = Vector((0.0, 0.0, 1.0)) if abs(y.z) < 0.9 else Vector((0.0, 1.0, 0.0))
        z = z - y * z.dot(y)
    z.normalize()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = head
    pb.matrix = m
    bpy.context.view_layer.update()


def apply_pose(arm_obj, bones):
    """Apply world-direction targets parent-first so children see posed parents."""
    order = [b[0] for b in _BONES if b[0] in bones]
    depth = {}
    for name, *_ in _BONES:
        d, p = 0, dict((b[0], b[3]) for b in _BONES)[name]
        while p:
            d += 1
            p = dict((b[0], b[3]) for b in _BONES)[p]
        depth[name] = d
    for name in sorted(order, key=lambda n: depth[n]):
        spec = bones[name]
        if isinstance(spec, dict):
            aim_bone(arm_obj, name, spec["dir"], spec.get("roll", (0.0, -1.0, 0.0)))
        else:
            aim_bone(arm_obj, name, spec)


def apply_hands(arm_obj, hands):
    """Curl/spread fingers. Bones were rolled with align_roll(palm normal), so
    local +X is the flexion axis and +rotation bends toward the palm."""
    for side, spec in hands.items():
        curl = float(spec.get("curl", 0.0))
        spread = float(spec.get("spread", 0.0))
        for fname, angles in _HAND_CURL.items():
            for i, angle in enumerate(angles, start=1):
                pb = arm_obj.pose.bones[f"{fname}.0{i}.{side}"]
                pb.rotation_mode = "XYZ"
                # spread only at the knuckle, or every phalanx would twist
                pb.rotation_euler = (math.radians(angle * curl), 0.0,
                                     math.radians(spread * 14.0) if i == 1 else 0.0)
    bpy.context.view_layer.update()


def world_bounds_z(cam_obj, objs):
    """Camera-space view-Z range over the evaluated (posed) geometry."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    zs = []
    inv = cam_obj.matrix_world.inverted()
    for ob in objs:
        ob_eval = ob.evaluated_get(depsgraph)
        try:
            me = ob_eval.to_mesh()
        except Exception:
            continue
        mw = ob_eval.matrix_world
        for v in me.vertices:
            zs.append(-(inv @ (mw @ v.co)).z)
        ob_eval.to_mesh_clear()
    return (min(zs), max(zs)) if zs else (1.0, 10.0)


def collect_keypoints(arm_obj, cam_obj, res, facing=(0.0, -1.0, 0.0)):
    """COCO-18 keypoints (OpenPose order) from posed bone head/tail positions.

    pose_bone.head/tail are ARMATURE-space; the scene root offset lives on the
    object transform, so every point must go through matrix_world or the skeleton
    drifts whenever a scene sets "root".

    Only the synthesized face points (nose/eyes/ears) use `facing`, which is the
    character's world-space forward direction. The body points come from bones and
    need no facing. Default (0,-1,0) reproduces the original fixed-forward result
    exactly, so single-character callers are unaffected; stage_multi passes the
    per-character facing so a rotated or profile character does not get its face
    keypoints on the back of its head.
    """
    mw = arm_obj.matrix_world

    def head(n):
        return mw @ arm_obj.pose.bones[n].head

    def tail(n):
        return mw @ arm_obj.pose.bones[n].tail

    hc = (head("head") + tail("head")) * 0.5
    fwd = Vector(facing).normalized()
    up = Vector((0.0, 0.0, 1.0))
    left = up.cross(fwd).normalized()
    kps = {
        "nose": hc + fwd * 0.105 + up * -0.005,
        "neck": head("neck"),
        "right_shoulder": tail("shoulder.R"), "right_elbow": head("forearm.R"), "right_wrist": head("hand.R"),
        "left_shoulder": tail("shoulder.L"), "left_elbow": head("forearm.L"), "left_wrist": head("hand.L"),
        "right_hip": head("thigh.R"), "right_knee": head("shin.R"), "right_ankle": head("foot.R"),
        "left_hip": head("thigh.L"), "left_knee": head("shin.L"), "left_ankle": head("foot.L"),
        "right_eye": hc + fwd * 0.085 + left * -0.035 + up * 0.030,
        "left_eye": hc + fwd * 0.085 + left * 0.035 + up * 0.030,
        "right_ear": hc + fwd * 0.010 + left * -0.095 + up * 0.015,
        "left_ear": hc + fwd * 0.010 + left * 0.095 + up * 0.015,
    }
    w, h = res
    out = {}
    for name, p in kps.items():
        co = world_to_camera_view(bpy.context.scene, cam_obj, p)
        out[name] = [round(co.x * w, 2), round((1.0 - co.y) * h, 2), round(co.z, 4)]

    # OpenPose hand format: 21 points = wrist + (base, joint1, joint2, tip) x 5.
    # With 3 phalanges per finger the joint list falls out of the bone chain.
    for side in ("L", "R"):
        pts = [head(f"hand.{side}")]
        for fname in _FINGERS:
            pts += [head(f"{fname}.01.{side}"), tail(f"{fname}.01.{side}"),
                    tail(f"{fname}.02.{side}"), tail(f"{fname}.03.{side}")]
        key = "hand_left" if side == "L" else "hand_right"
        out[key] = [[round(world_to_camera_view(bpy.context.scene, cam_obj, p).x * w, 2),
                     round((1.0 - world_to_camera_view(bpy.context.scene, cam_obj, p).y) * h, 2)]
                    for p in pts]
    return out


def reset_pose(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    spec_path = argv[argv.index("--spec") + 1] if "--spec" in argv else "scenes.json"
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else "out"
    # Blender resolves relative render paths against C:\ when no .blend is open,
    # so anchor everything to the spec file's directory.
    spec_path = os.path.abspath(spec_path)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(spec_path), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # utf-8-sig: PowerShell's Set-Content -Encoding UTF8 prepends a BOM
    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = json.load(fh)

    # clean slate
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "BLENDER_WORKBENCH"
    try:
        scene.eevee.taa_render_samples = 16
    except Exception:
        pass
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    arm_obj = build_armature()
    mannequin = build_mannequin(arm_obj)
    setup_lights()

    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    scene.world = world

    cam_data = bpy.data.cameras.new("cam")
    cam_obj = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    depth_mat = make_depth_material(1.0, 10.0)
    clay_mat = make_clay_material()
    env_objs = []
    summary = []

    for sc in spec["scenes"]:
        # env is rebuilt per scene: a stool belongs under a seated pose, not under a
        # standing one, and stale props would leak into the depth map.
        clear_env(env_objs)
        env_objs = build_env(sc.get("env", spec.get("env", {"boxes": []})))
        all_objs = [mannequin] + env_objs

        reset_pose(arm_obj)
        # root offset moves the whole character (sitting on a prop, crouching, ...)
        arm_obj.location = Vector(sc.get("root", [0.0, 0.0, 0.0]))
        apply_pose(arm_obj, sc.get("bones", {}))
        apply_hands(arm_obj, sc.get("hands", {}))

        res = tuple(sc.get("resolution", [832, 1216]))
        scene.render.resolution_x, scene.render.resolution_y = res
        scene.render.resolution_percentage = 100
        cam_data.lens = sc.get("lens", 50.0)
        cam_data.sensor_width = 36.0
        cam_obj.location = Vector(sc["camera"]["location"])
        look_at(cam_obj, sc["camera"]["target"])
        bpy.context.view_layer.update()

        # Range is anchored to the SUBJECT, not the whole scene: a 40 m floor would
        # otherwise stretch the range and flatten the character into mid-gray.
        near, far = world_bounds_z(cam_obj, [mannequin])
        near = max(0.05, near - 0.4)
        far = far + 1.2
        cam_data.clip_start, cam_data.clip_end = max(0.01, near), far + 1.0
        mat = make_depth_material(near, far)
        assign_material(all_objs, mat)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_depth.png")
        bpy.ops.render.render(write_still=True)

        assign_material(all_objs, clay_mat)
        scene.render.filepath = os.path.join(out_dir, f"{sc['name']}_clay.png")
        bpy.ops.render.render(write_still=True)

        kps = collect_keypoints(arm_obj, cam_obj, res)
        with open(os.path.join(out_dir, f"{sc['name']}_kps.json"), "w", encoding="utf-8") as fh:
            json.dump({"resolution": list(res), "keypoints": kps}, fh, indent=1)

        summary.append({"name": sc["name"], "resolution": list(res), "near": round(near, 3), "far": round(far, 3)})
        print(f"[scene] {sc['name']}  {res[0]}x{res[1]}  lens={cam_data.lens}  z=[{near:.2f},{far:.2f}]")

    print("[done] " + json.dumps(summary))


if __name__ == "__main__":
    main()
