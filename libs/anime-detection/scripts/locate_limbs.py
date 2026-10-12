#!/usr/bin/env python3
r"""按关键点坐标**自动裁出手/脚放大图**（人工核对用），不再手工猜坐标。

    python locate_limbs.py <图片目录> --json-dir <你的 ComfyUI 安装目录>\output\dsh_probe [--prefix op_]
                           [--out crops] [--zoom 3]

工作方式：读 `det_probe.py`（或 `OpenposePreprocessor`）产出的 JSON —— 注意
**坐标是归一化 0~1**（乘原图宽高才是像素）。用身体点的腕(4/7)/踝(10/13)坐标做中心裁窗。
手部关键点（`hand_*_keypoints_2d`）在本机是**占位假数据**（坐标全 0、分数全 1.0），所以只用来判断
"是否非零"，不用来定位。
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

BODY = {4: 'Rwrist', 7: 'Lwrist', 10: 'Rankle', 13: 'Lankle'}
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


def trip(flat):
    return [(float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
            for i in range(0, len(flat) - 2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir")
    ap.add_argument("--json-dir", required=True, help="OpenPose JSON 所在目录（如 output\\dsh_probe）")
    ap.add_argument("--prefix", default="op_")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--zoom", type=float, default=3.0)
    ap.add_argument("--hand-half", type=int, default=130, help="以腕为中心的半窗（原图像素）")
    ap.add_argument("--overlay", action="store_true", help="同时把骨架画回原图（核对检测准不准）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    n_crop = 0
    for f in sorted(glob.glob(os.path.join(a.img_dir, "*.png"))):
        tag = os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
        js = sorted(glob.glob(os.path.join(a.json_dir, a.prefix + tag + "*.json")))
        if not js:
            print('%-28s 无 JSON，跳过' % tag); continue
        d = json.load(open(js[-1], encoding='utf-8'))
        im = Image.open(f).convert('RGB')
        W, H = im.size
        people = (d[0] or {}).get('people') or []
        if not people:
            print('%-28s people=0（检测器没找到人）' % tag); continue
        if a.overlay:
            ov = im.copy(); dr = ImageDraw.Draw(ov)
            for p in people:
                b = trip(p.get('pose_keypoints_2d') or [])
                if not b:
                    continue
                pts = [(x * W, y * H) for x, y, _ in b]
                for i, j in LIMBS:
                    if i < len(pts) and j < len(pts):
                        dr.line([pts[i], pts[j]], fill=(255, 40, 40), width=max(3, W // 300))
                for k, (x, y, _) in enumerate(b):
                    col = (255, 235, 0) if k in BODY else (0, 255, 255)
                    r = max(4, W // 240)
                    dr.ellipse([x * W - r, y * H - r, x * W + r, y * H + r], fill=col)
            ov.save(os.path.join(a.out, 'skeleton_%s.png' % tag))
        for pi, p in enumerate(people):
            b = trip(p.get('pose_keypoints_2d') or [])
            for idx, name in BODY.items():
                if idx >= len(b):
                    continue
                x, y, s = b[idx]
                if s < 0.2 or (abs(x) < 1e-6 and abs(y) < 1e-6):
                    continue
                cx, cy = x * W, y * H
                half = a.hand_half
                box = (int(max(0, cx - half)), int(max(0, cy - half)),
                       int(min(W, cx + half)), int(min(H, cy + half)))
                c = im.crop(box)
                if c.width < 24 or c.height < 24:
                    continue
                c = c.resize((int(c.width * a.zoom), int(c.height * a.zoom)), Image.LANCZOS)
                dst = os.path.join(a.out, '%s_p%d_%s.png' % (tag, pi + 1, name))
                c.save(dst); n_crop += 1
                print('  → %s  (中心 %.0f,%.0f 置信 %.2f)' % (os.path.basename(dst), cx, cy, s))
    print('\n共裁出 %d 张 → %s' % (n_crop, a.out))


if __name__ == '__main__':
    main()
