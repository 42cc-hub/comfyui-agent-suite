"""canvas_mask.py —— 把人在画布上涂的两层合成为 mask（L1 要打码/要改 − L2 要保留）。

用途：`human_canvas` 让人涂"要处理的区域"（L1）和"要保护的区域"（L2），本脚本算出最终 mask：

    python canvas_mask.py --paint L1.png [--keep L2.png] --out mask.png \
        [--src IMG.png --overlay preview.png] [--dilate 4] [--feather 3] [--grow-box x0,y0,x1,y1]

约定（与 ComfyUI 对齐）：
  · **白色 = 要处理**（重绘区 / 打码区），黑色 = 不动；
  · `--keep` 里的白色区域会从 mask 里扣掉（保护层，例如"这缕头发别动"）；
  · `--dilate` 在合成**之后**做（像素），`--feather` 做高斯羽化 —— 贴回时边缘才不会出硬缝；
  · 输出 mask 是灰度 PNG；给了 `--src` 就再出一张**叠加预览**（红=处理区）方便确认。

绝不覆盖输入文件。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paint', '--rope', '--L1', dest='paint', required=True,
                    help='要处理的区域（白=处理）；别名 --rope/--L1')
    ap.add_argument('--keep', '--L2', dest='keep', default=None, help='要保护的区域（白=保留）')
    ap.add_argument('--out', required=True, help='输出 mask（灰度 PNG）')
    ap.add_argument('--src', default=None, help='原图；给了就出叠加预览')
    ap.add_argument('--overlay', default=None, help='叠加预览输出路径（默认 <out>_overlay.png）')
    ap.add_argument('--dilate', type=int, default=0, help='膨胀像素（合成后）')
    ap.add_argument('--feather', type=float, default=0, help='羽化半径（像素）')
    ap.add_argument('--invert-keep', action='store_true', help='把 --keep 当"要处理"而不是"要保护"')
    ap.add_argument('--grow-box', default=None, help='x0,y0,x1,y1：额外并入一个矩形区域')
    a = ap.parse_args()

    import numpy as np
    from PIL import Image, ImageFilter

    for p in (a.paint, a.keep, a.src):
        if p and not os.path.exists(p):
            raise SystemExit('找不到文件：%s' % p)

    mask = np.array(Image.open(a.paint).convert('L'))
    if a.keep:
        keep = np.array(Image.open(a.keep).convert('L').resize((mask.shape[1], mask.shape[0])))
        thr = 127
        if a.invert_keep:
            mask = np.maximum(mask, keep)
        else:
            mask = np.where(keep > thr, 0, mask)          # 保护层里的白 ⇒ 从 mask 扣掉
    if a.grow_box:
        x0, y0, x1, y1 = [int(float(v)) for v in a.grow_box.split(',')]
        mask[max(0, y0):y1, max(0, x0):x1] = 255
    out = Image.fromarray(mask.astype('uint8'), 'L')
    if a.dilate > 0:
        out = out.filter(ImageFilter.MaxFilter(2 * a.dilate + 1))
    if a.feather > 0:
        out = out.filter(ImageFilter.GaussianBlur(a.feather))

    if os.path.abspath(a.out) in {os.path.abspath(p) for p in (a.paint, a.keep, a.src) if p}:
        raise SystemExit('--out 不能覆盖输入文件')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out)
    cover = float((np.array(out) > 127).mean())
    print('mask → %s（覆盖 %.2f%%，dilate=%d feather=%.1f）' % (a.out, cover * 100, a.dilate, a.feather))

    if a.src:
        base = Image.open(a.src).convert('RGB')
        m = np.array(out.resize(base.size))
        arr = np.array(base).copy()
        sel = m > 127
        arr[sel] = (0.45 * arr[sel] + 0.55 * np.array([255, 40, 40])).astype('uint8')   # 红=处理区
        ov = a.overlay or os.path.splitext(a.out)[0] + '_overlay.png'
        Image.fromarray(arr).save(ov)
        print('叠加预览 → %s' % ov)
    return 0


if __name__ == '__main__':
    sys.exit(main())
