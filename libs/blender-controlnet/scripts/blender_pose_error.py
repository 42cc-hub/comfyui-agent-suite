"""Compare two 18-point COCO keypoint sets after a similarity alignment.

Framing (scale, rotation, translation) is absorbed by an Umeyama fit, so the
residual measures POSE shape agreement only -- which is the part we are trying to
reproduce. Error is reported as a percentage of the target's RMS radius, so it is
resolution-independent.

Joints below `--min-conf` (or missing on either side) are excluded from both the
alignment and the score: detectors routinely fail on one occluded limb and emit a
garbage coordinate, which would otherwise swamp the metric.

usage: python pose_error.py <target.json> <candidate.json> [--min-conf 0.3] [--overlay out.png]
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

ORDER = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
         "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle",
         "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


def load(path, min_conf):
    with open(path, "r", encoding="utf-8-sig") as fh:
        d = json.load(fh)
    kp = d["keypoints"]
    pts = np.array([kp[n][:2] if n in kp else (np.nan, np.nan) for n in ORDER], dtype=float)
    conf = d.get("confidence")
    valid = np.array([n in kp for n in ORDER])
    if conf is not None:
        valid &= np.array([float(conf.get(n, 1.0)) >= min_conf for n in ORDER])
    valid &= np.isfinite(pts).all(axis=1)
    return pts, valid


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
    min_conf = 0.3
    if "--min-conf" in sys.argv:
        min_conf = float(sys.argv[sys.argv.index("--min-conf") + 1])
    target_all, tv = load(sys.argv[1], min_conf)
    cand_all, cv = load(sys.argv[2], min_conf)
    keep = tv & cv
    dropped = [ORDER[i] for i in range(len(ORDER)) if not keep[i]]

    target, cand = target_all[keep], cand_all[keep]
    scale, R, t = umeyama(cand, target)
    aligned = (scale * (R @ cand.T)).T + t
    err = np.linalg.norm(aligned - target, axis=1)
    radius = float(np.sqrt(((target - target.mean(0)) ** 2).sum(1).mean()))
    rms = float(np.sqrt((err ** 2).mean()))

    print(f"joints used       : {keep.sum()}/18" + (f"   excluded: {', '.join(dropped)}" if dropped else ""))
    print(f"aligned RMS error : {rms:.1f} px  ({100 * rms / radius:.1f}% of target radius)")
    print(f"max joint error   : {err.max():.1f} px  ({ORDER[int(np.where(keep)[0][err.argmax()])]})")
    print(f"similarity        : scale={scale:.3f}")
    print("per-joint (px):")
    for idx in np.argsort(-err):
        print(f"   {ORDER[int(np.where(keep)[0][idx])]:15s} {err[idx]:6.1f}")

    if "--overlay" in sys.argv:
        out = sys.argv[sys.argv.index("--overlay") + 1]
        # draw in FULL 18-point space: LIMBS indexes the unfiltered table
        aligned_full = (scale * (R @ cand_all.T)).T + t
        W = int(max(target_all[:, 0].max(), aligned_full[:, 0].max()) + 80)
        H = int(max(target_all[:, 1].max(), aligned_full[:, 1].max()) + 80)
        img = Image.new("RGB", (W, H), (12, 12, 16))
        d = ImageDraw.Draw(img)
        for a, b in LIMBS:
            if not (keep[a] and keep[b]):
                continue
            d.line([tuple(target_all[a]), tuple(target_all[b])], fill=(70, 70, 80), width=5)
            d.line([tuple(aligned_full[a]), tuple(aligned_full[b])], fill=(255, 90, 90), width=3)
        for i, p in enumerate(target_all):
            if keep[i]:
                d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(230, 230, 235))
        img.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
