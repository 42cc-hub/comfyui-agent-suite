"""split_pair_masks.py —— 两人贴在一起时，把画面切成**互不重叠**的两块掩版（逐人重绘的前提）。

为什么要"切一刀"而不是直接用检测框：两人靠肩时 person 框会重叠（实测重叠 ~116 px），
两次重绘若各自覆盖重叠区，后一次会把前一次改好的人又改一遍。**按重叠中点切一刀**比任何提示词都可靠。

掩版约定（ComfyUI `LoadImage`）：**MASK = 1 − alpha ⇒ 要重绘的区域 alpha = 0**，
所以这里输出的两张图是 RGBA，重绘区 alpha=0、其余 alpha=255。

输入：`check_parts.py --json` 或 `locate.py` 产出的 JSON（含 M2a 的 person 框）
用法：
    python split_pair_masks.py --base pair_base.png --contract parts_pair/pair_base.json \
        --out-dir out/ --grow 14 [--order left,right] [--preview]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def load_person_boxes(contract):
    d = json.load(open(contract, encoding='utf-8'))
    boxes = []
    for b in d.get('boxes', d.get('boxes_all', [])):
        if b.get('cls') == 'person':
            boxes.append([float(v) for v in b['box']])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='底图（两人同框那张）')
    ap.add_argument('--contract', required=True, help='check_parts.py/locate.py 的 JSON（含 person 框）')
    ap.add_argument('--out-dir', default='out')
    ap.add_argument('--grow', type=int, default=14, help='上下各外扩多少像素（头发/肩会超出检测框）')
    ap.add_argument('--left-name', default='pair_mask_left.png')
    ap.add_argument('--right-name', default='pair_mask_right.png')
    ap.add_argument('--preview', action='store_true', help='再出一张两块掩版叠加在底图上的预览')
    a = ap.parse_args()

    img = Image.open(a.base).convert('RGB')
    W, H = img.size
    boxes = sorted(load_person_boxes(a.contract), key=lambda b: b[0])
    if len(boxes) < 2:
        raise SystemExit('只检出 %d 个人物框（需要 2 个）：%s' % (len(boxes), boxes))
    left, right = boxes[0], boxes[1]
    overlap = left[2] - right[0]
    split = int((right[0] + left[2]) / 2.0)
    print('左框 x=[%d,%d] 右框 x=[%d,%d] ⇒ 重叠 %d px，切线 x=%d'
          % (left[0], left[2], right[0], right[2], overlap, split))
    if overlap <= 0:
        print('（两人没重叠，切线只是形式上的分界 —— 也可以直接用各自框）')

    os.makedirs(a.out_dir, exist_ok=True)
    sides = {
        a.left_name: (0, left[1] - a.grow, split, left[3] + a.grow),
        a.right_name: (split, right[1] - a.grow, W, right[3] + a.grow),
    }
    masks = {}
    rgb = np.asarray(img)
    for name, (x0, y0, x1, y1) in sides.items():
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        region = np.zeros((H, W), bool)
        region[y0:y1, x0:x1] = True
        alpha = np.where(region, 0, 255).astype(np.uint8)      # 重绘区 alpha=0
        out = os.path.join(a.out_dir, name)
        Image.merge('RGBA', (*[Image.fromarray(rgb[..., c]) for c in range(3)], Image.fromarray(alpha))).save(out)
        masks[name] = (region, out)
        print('%s → %s（覆盖 %.2f%%）' % (name, out, 100.0 * region.mean()))

    if a.preview:
        prev = rgb.copy()
        for i, (region, _p) in enumerate(masks.values()):
            col = np.array([255, 60, 60]) if i == 0 else np.array([60, 160, 255])
            prev[region] = (0.55 * prev[region] + 0.45 * col).astype('uint8')
        pv = os.path.join(a.out_dir, 'pair_masks_preview.png')
        Image.fromarray(prev).save(pv)
        print('预览（红=左 / 蓝=右）→ %s' % pv)

    print('\n下一步：把底图与两张掩版上传到 ComfyUI，再跑 pair_repaint_left / pair_repaint_right')
    print('  python scripts/tools/comfy_upload.py %s %s %s --sub dsh_ctrl'
          % (a.base, os.path.join(a.out_dir, a.left_name), os.path.join(a.out_dir, a.right_name)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
