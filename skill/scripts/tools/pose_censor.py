#!/usr/bin/env python
"""pose_censor.py —— 无人可提供掩版时的**自动打码**通道（只应在全自动模式下用）。

用骨骼关键点（COCO-17）自动框出两个区域并打码：
  · 胸部：肩（5,6）→ 髋（11,12）之间的躯干带
  · 胯部：髋线向下约 1/3 躯干高的骨盆带

硬约束（每一条都为了不产出"看起来打了码其实没盖住"的图）：
  1. 拿不到关键点（没人 / 关节不可见）⇒ **拒绝**，退出码 3；
  2. 覆盖面积异常（< --min-cover% 或 > --max-cover%）⇒ **拒绝**，退出码 4；
  3. 只写副本（`--out` 不能等于 `--src`），掩版外像素必须逐位不变；
  4. 打印下一步：**必须**对副本重跑 `libs/wd14-gate/scripts/wd14_gate.py`，判定改善才允许看。

姿态 JSON 三种形态都认（前两种是通用格式，第三种是检测套件 `locate.py --out DIR` 的原生契约）：
  {"people": [{"keypoints": [[x, y, v], ... 17 ...]}]}
  {"keypoints": [[x, y, v], ... 17 ...]}
  {"points": [{"name": "body.left_shoulder", "x": .., "y": .., "vis": 2, "chain": "M1#0"}, ...]}

例：
  python pose_censor.py --src a.png --pose pose.json --out a_mos.png --cell 32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows 控制台常是 GBK：⇒ / ✅ 之类字符会让 print 抛 UnicodeEncodeError，
# 而崩溃发生在产物写完之后 —— 看起来像"脚本失败"，实际是打印失败。stdout 与 stderr 一起转 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosaic_censor import (apply_censor, mask_from_polygons, mask_from_boxes,  # noqa: E402
                           parse_fill, trapezoid_polygon, inverted_triangle_polygon)

# COCO-17
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
VISIBLE = 1  # v >= 1 视为可用（0 = 未标注）
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def from_points(record):
    """检测套件契约：扁平点列表，按 chain（M1#0…）分人、按 body.<joint> 名字对位。"""
    people = {}
    for point in record.get("points", []) or []:
        name = str(point.get("name", ""))
        if not name.startswith("body."):
            continue
        joint = name.split(".", 1)[1]
        if joint not in COCO17:
            continue
        vis = point.get("vis")
        if vis is None:
            prob = float(point.get("prob", 0) or 0)
            vis = 2 if prob >= 0.5 else 1 if prob >= 0.3 else 0
        chain = str(point.get("chain") or "M1#0")
        kps = people.setdefault(chain, [[0.0, 0.0, 0.0] for _ in range(17)])
        kps[COCO17.index(joint)] = [float(point.get("x", 0) or 0), float(point.get("y", 0) or 0), float(vis)]
    return [kps for kps in people.values()]


def load_people(pose_path):
    data = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "people" in data:
        people = data["people"]
    elif isinstance(data, dict) and "keypoints" in data:
        people = [data]
    elif isinstance(data, dict) and "points" in data:
        return from_points(data)
    elif isinstance(data, list):
        people = data
    else:
        sys.exit("REFUSED: pose JSON has no people/keypoints/points")
    out = []
    for person in people:
        kps = person.get("keypoints")
        if isinstance(kps, dict):
            kps = [kps.get(str(i)) or kps.get(i) for i in range(17)]
        if not isinstance(kps, list) or len(kps) < 17:
            continue
        out.append([[float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else VISIBLE)] for p in kps])
    return out


def garment_polygons(kps, kinds, margin):
    """按骨架画"一套内衣"：胸罩 = 上宽下窄的梯形，内裤 = 上宽下尖的倒三角。

    位置取自解剖比例（躯干高 T = 髋 y − 肩 y）：
      胸罩带 y = 肩 + 0.35T … 肩 + 0.62T，宽度取肩宽；下边内收 22%
      内裤带 y = 髋 − 0.05T … 髋 + 0.22T，宽度取髋宽
    """
    def pt(i):
        return kps[i][0], kps[i][1], kps[i][2]

    lx, ly, lv = pt(L_SHOULDER)
    rx, ry, rv = pt(R_SHOULDER)
    hx1, hy1, hv1 = pt(L_HIP)
    hx2, hy2, hv2 = pt(R_HIP)
    if min(lv, rv, hv1, hv2) < VISIBLE:
        return None
    shoulder_y = min(ly, ry)
    hip_y = max(hy1, hy2)
    torso = max(8.0, hip_y - shoulder_y)
    polys = []
    if "bra" in kinds:
        mid = (lx + rx) / 2.0
        half_top = abs(lx - rx) / 2.0 * (1 + margin)
        top = shoulder_y + 0.35 * torso
        bottom = shoulder_y + 0.62 * torso
        polys.append(trapezoid_polygon(mid - half_top, top, mid + half_top, bottom, 0.22))
    if "panty" in kinds:
        mid = (hx1 + hx2) / 2.0
        half = abs(hx1 - hx2) / 2.0 * (1 + margin)
        top = hip_y - 0.05 * torso
        apex = hip_y + 0.22 * torso
        polys.append(inverted_triangle_polygon(mid - half, top, mid + half, apex))
    return polys


def torso_boxes(kps, margin, chest, crotch):
    def pt(i):
        x, y, v = kps[i]
        return x, y, v

    lx, ly, lv = pt(L_SHOULDER)
    rx, ry, rv = pt(R_SHOULDER)
    hx1, hy1, hv1 = pt(L_HIP)
    hx2, hy2, hv2 = pt(R_HIP)
    if min(lv, rv, hv1, hv2) < VISIBLE:
        return None
    shoulder_y = min(ly, ry)
    hip_y = max(hy1, hy2)
    torso_h = max(8.0, hip_y - shoulder_y)
    left = min(lx, rx, hx1, hx2)
    right = max(lx, rx, hx1, hx2)
    boxes = []
    if chest:
        boxes.append((left, shoulder_y, right, hip_y))
    if crotch:
        boxes.append((min(hx1, hx2), hip_y, max(hx1, hx2), hip_y + torso_h * 0.35))
    grown = []
    for x0, y0, x1, y1 in boxes:
        mx = (x1 - x0) * margin
        my = (y1 - y0) * margin
        grown.append((x0 - mx, y0 - my, x1 + mx, y1 + my))
    return grown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--fill", type=parse_fill, default=None,
                    help="平涂代替马赛克：black / white / grey / R,G,B")
    ap.add_argument("--margin", type=float, default=0.15, help="每个框按自身尺寸外扩的比例")
    ap.add_argument("--no-chest", action="store_true")
    ap.add_argument("--no-crotch", action="store_true")
    ap.add_argument("--garment", choices=["bra", "panty", "both"], default=None,
                    help="画「内衣形状」（胸罩梯形 + 内裤倒三角），比矩形小、会留缝，**不改变闸门判定**"
                         "（裸底图上各填色仍是 BLOCK）—— 只作示意/排版，别当骗过闸门的手段")
    ap.add_argument("--min-cover", type=float, default=0.5, help="覆盖低于该百分比即拒绝")
    ap.add_argument("--max-cover", type=float, default=60.0, help="覆盖高于该百分比即拒绝")
    ap.add_argument("--report")
    args = ap.parse_args()

    from PIL import Image
    with Image.open(args.src) as im:
        shape = (im.height, im.width)

    people = load_people(args.pose)
    if not people:
        sys.exit("REFUSED: no usable person/keypoints in the pose JSON — nothing to censor, do NOT look at the original")
    boxes = []
    polygons = []
    kinds = [] if args.garment is None else (["bra", "panty"] if args.garment == "both" else [args.garment])
    for kps in people:
        if kinds:
            found_polys = garment_polygons(kps, kinds, args.margin)
            if found_polys:
                polygons.extend(found_polys)
        else:
            found = torso_boxes(kps, args.margin, not args.no_chest, not args.no_crotch)
            if found:
                boxes.extend(found)
    if not boxes and not polygons:
        sys.exit("REFUSED: shoulders/hips not visible in the pose JSON — nothing to censor")

    mask = mask_from_boxes(shape, [(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
                                   for x0, y0, x1, y1 in boxes])
    if polygons:
        mask = mask | mask_from_polygons(shape, polygons)
    cover = mask.sum() / mask.size * 100
    if cover < args.min_cover:
        sys.exit(f"REFUSED: computed cover is only {cover:.2f}% (< {args.min_cover}%) — keypoints look wrong")
    if cover > args.max_cover:
        sys.exit(f"REFUSED: computed cover is {cover:.2f}% (> {args.max_cover}%) — keypoints look wrong")

    route = "pose-garment-" + (args.garment or "none") if polygons else "pose-chest-crotch"
    report = apply_censor(args.src, args.out, mask, args.cell, args.report,
                          extra={"boxes": [[round(v, 1) for v in b] for b in boxes],
                                 "polygons": [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in polygons],
                                 "people": len(people), "route": route,
                                 "margin": args.margin},
                          fill=args.fill)
    print(json.dumps(report, ensure_ascii=False))
    print("\n这是**粗框**自动打码（比人工掩版粗）。下一步（必做）：对副本重跑闸门，"
          "只有判定改善（REVIEW/OK）才允许 read_image；仍 BLOCK ⇒ 不看，只报路径。\n"
          "  python libs/wd14-gate/scripts/wd14_gate.py " + args.out)


if __name__ == "__main__":
    main()
