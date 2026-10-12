"""Fit a 3D Blender pose to a target 2D OpenPose skeleton.

Blender itself is the forward model: the same armature, the same aim_bone FK and
the same world_to_camera_view projection that produce the control maps. Nothing is
reimplemented, so a fitted pose is guaranteed to render exactly as scored.

Objective: the 18 COCO keypoints are aligned to the target with a similarity
transform (Umeyama) and scored by RMS residual / target radius, i.e. framing is
free and only pose shape is penalised.

Search: greedy coordinate descent over per-bone (azimuth, elevation) plus the
camera distance, with a shrinking step. Depth is ambiguous from a single 2D view,
so the hand-authored init acts as the prior -- the fit refines a plausible
arabesque rather than flipping it inside out.

usage:
  blender --background --factory-startup --python fit_pose.py -- \
      --target ref/target1_kps.json --spec scenes_dance.json --scene dance_front \
      --out scenes_dance_fitted.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "skill_scripts"))
import blender_build_scene as bs  # noqa: E402

ORDER = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
         "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle",
         "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]


def dir_to_angles(v):
    v = Vector(v).normalized()
    el = math.asin(max(-1.0, min(1.0, v.z)))
    az = math.atan2(v.y, v.x)
    return az, el


def angles_to_dir(az, el):
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = (S * np.array([1.0, d])).sum() / var_s if var_s > 1e-12 else 1.0
    return scale, R, mu_d - scale * (R @ mu_s)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    arg = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    target_path = os.path.join(HERE, arg("--target"))
    spec_path = os.path.join(HERE, arg("--spec"))
    scene_name = arg("--scene")
    out_path = os.path.join(HERE, arg("--out"))

    with open(target_path, "r", encoding="utf-8-sig") as fh:
        target_data = json.load(fh)
    T_all = np.array([target_data["keypoints"][n][:2] for n in ORDER], dtype=float)
    # Detectors emit garbage coordinates with score ~0 for occluded limbs (a raised
    # leg behind a tutu). Those must not steer the fit, so drop them up front.
    conf = target_data.get("confidence")
    if conf is None:
        keep = np.ones(len(ORDER), dtype=bool)
    else:
        keep = np.array([float(conf.get(n, 1.0)) >= 0.3 for n in ORDER])
    keep &= np.isfinite(T_all).all(axis=1)
    T = T_all[keep]
    print(f"[fit] target joints used {keep.sum()}/18"
          + (f"  excluded: {', '.join(ORDER[i] for i in range(len(ORDER)) if not keep[i])}" if not keep.all() else ""))
    radius = float(np.sqrt(((T - T.mean(0)) ** 2).sum(1).mean()))

    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = json.load(fh)
    scene_spec = next(s for s in spec["scenes"] if s["name"] == scene_name)
    bones_init = scene_spec["bones"]
    res = tuple(scene_spec.get("resolution", [1024, 1024]))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    arm = bs.build_armature()

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.lens = scene_spec.get("lens", 50.0)
    cam_data.sensor_width = 36.0
    cam_dir = Vector(scene_spec["camera"]["location"]) - Vector(scene_spec["camera"]["target"])
    cam_dist0 = cam_dir.length
    cam_target = Vector(scene_spec["camera"]["target"])

    scene.render.resolution_x, scene.render.resolution_y = res

    def set_camera(dist):
        loc = cam_target + cam_dir.normalized() * dist
        cam.location = loc
        bs.look_at(cam, cam_target)
        bpy.context.view_layer.update()

    names = list(bones_init.keys())
    params = []
    for n in names:
        az, el = dir_to_angles(bones_init[n])
        params += [az, el]
    params.append(cam_dist0)
    n_pose = len(names) * 2

    def apply(params):
        set_camera(params[-1])
        bones = {n: angles_to_dir(params[2 * i], params[2 * i + 1]) for i, n in enumerate(names)}
        bs.apply_pose(arm, bones)

    def evaluate(params):
        apply(params)
        kps = bs.collect_keypoints(arm, cam, res)
        P = np.array([kps[n][:2] for n in ORDER], dtype=float)[keep]
        scale, R, t = umeyama(P, T)
        aligned = (scale * (R @ P.T)).T + t
        rms = float(np.sqrt(((aligned - T) ** 2).sum(1).mean()))
        return rms / radius, aligned

    t0 = time.time()
    best, _ = evaluate(params)
    print(f"[fit] init error {100 * best:.2f}%  ({time.time() - t0:.1f}s/eval warmup)")

    t0 = time.time()
    step = 0.30
    for it in range(14):
        improved = 0
        for i in range(len(params)):
            cur = params[i]
            for delta in (step, -step):
                trial = params.copy()
                trial[i] = cur + delta
                if i < n_pose and i % 2 == 1:            # elevation stays in range
                    trial[i] = max(-1.55, min(1.55, trial[i]))
                if i == len(params) - 1:                  # camera distance stays positive
                    trial[i] = max(1.5, min(12.0, trial[i]))
                if i < n_pose and i % 2 == 0:             # wrap azimuth
                    trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                e, _ = evaluate(trial)
                if e < best - 1e-9:
                    best, params = e, trial
                    improved += 1
        print(f"[fit] pass {it:2d}  step={step:.3f}  error={100 * best:.2f}%  improvements={improved}  {time.time() - t0:.0f}s")
        if improved == 0 and step < 0.02:
            break
        step *= 0.72

    err, aligned = evaluate(params)
    print(f"[fit] final error {100 * err:.2f}% of target radius  ({time.time() - t0:.0f}s)")

    fitted = {n: list(angles_to_dir(params[2 * i], params[2 * i + 1])) for i, n in enumerate(names)}
    with open(os.path.join(HERE, "fitted_pose.json"), "w", encoding="utf-8") as fh:
        json.dump({"scene": scene_name, "error_pct": 100 * err, "cam_distance": params[-1], "bones": fitted}, fh, indent=1)
    with open(os.path.join(HERE, "fitted_kps.json"), "w", encoding="utf-8") as fh:
        json.dump({"resolution": list(res),
                   "keypoints": {n: [round(float(aligned[i][0]), 1), round(float(aligned[i][1]), 1)]
                                 for i, n in enumerate([n for n, k in zip(ORDER, keep) if k])}}, fh, indent=1)

    out = json.loads(json.dumps(spec))
    for s in out["scenes"]:
        s["bones"] = dict(fitted)
        if s["name"] == scene_name:
            s["camera"]["location"] = [round(v, 3) for v in (cam_target + cam_dir.normalized() * params[-1])]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("[fit] wrote", out_path)


if __name__ == "__main__":
    main()
