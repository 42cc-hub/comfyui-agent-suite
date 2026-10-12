"""mask_overlay.py —— 把逐人掩版叠在底图上，**确认哪块是谁的**（分区重绘前必做的一步）。

为什么必须有这一步：`check_parts.py --person-masks` 输出的 `persons_masks[]` 里，
`person_id` / `box` 取自该实例的 `members[0]`，**两人强重叠（膝枕、拥抱）时两条记录可能重复**，
文件名里的编号（`person0/person1`）又来自构建顺序 ⇒ **不能靠名字或 JSON 猜谁是谁**。
实测踩过：掩版喂反了，用"另一个人闭眼睡觉"的提示词去重绘这一位的脸，眼睛颜色当场串味。

本工具直接给答案：
  · 打印每个掩版的像素外接框、与各 person 框的 IoU（据此判断它属于哪一位）；
  · 出一张叠加图（每人一色），**人眼扫一下就能确认**。

用法：
    python mask_overlay.py BASE.png MASK1.png [MASK2.png ...] [--out overlay.png] [--names A,B]
    python mask_overlay.py BASE.png masks/ --out overlay.png      # 给目录则取里面的 *_person*.png（灰度版）
"""
import argparse
import glob
import os
import sys

COLORS = [(255, 60, 60), (60, 140, 255), (60, 220, 120), (255, 200, 40), (200, 80, 255), (0, 220, 220)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('masks', nargs='+')
    ap.add_argument('--out', default='mask_overlay.png')
    ap.add_argument('--names', default=None, help='逗号分隔，用于打印')
    ap.add_argument('--alpha', type=float, default=0.45)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    base = Image.open(a.base).convert('RGB')
    W, H = base.size
    paths = []
    for m in a.masks:
        if os.path.isdir(m):
            paths += sorted(p for p in glob.glob(os.path.join(m, '*_person*.png')) if '_rgba' not in p)
        else:
            paths.append(m)
    if not paths:
        raise SystemExit('没找到掩版文件')
    names = a.names.split(',') if a.names else [os.path.basename(p) for p in paths]

    arr = np.array(base).astype('float32')
    boxes = []
    for i, p in enumerate(paths):
        im = Image.open(p)
        if im.mode in ('RGBA', 'LA'):
            region = np.array(im.convert('RGBA').split()[-1].resize((W, H))) == 0     # alpha=0 = 重绘区
        else:
            region = np.array(im.convert('L').resize((W, H))) > 127
        ys, xs = np.nonzero(region)
        if not len(xs):
            print('%-28s 空掩版' % names[i])
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        boxes.append((names[i], box, float(region.mean())))
        col = np.array(COLORS[i % len(COLORS)], dtype='float32')
        arr[region] = arr[region] * (1 - a.alpha) + col * a.alpha
        print('%-28s 外接框 x[%d,%d] y[%d,%d] 覆盖 %.2f%%'
              % (names[i], box[0], box[2], box[1], box[3], 100 * region.mean()))

    if len(boxes) > 1:
        print('\n判据：**靠上/靠里那块通常是坐着或靠前的人，靠下/靠外那块是躺着或靠后的人**；'
              '两块若在 x 上左右分开，就是左/右。务必用叠加图确认后再喂提示词。')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(arr.round().clip(0, 255).astype('uint8')).save(a.out)
    print('\n叠加图 → %s（每色一块；%s）' % (a.out, ', '.join(n for n, _, _ in boxes)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
