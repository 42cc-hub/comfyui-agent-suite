"""red_mask.py —— 颜色 → 掩版（经典 CV，够快够稳；"红绳/红布条"这类高饱和目标很好用）。

    python red_mask.py IMG.png OUT.png [--color 255,0,0] [--tol 70] [--dilate 2] [--feather 1] \
        [--min-area 40] [--preview preview.png]

说明：颜色阈值在**HSV**里判（比 RGB 更抗光照），并把 `--min-area` 以下的小块当噪声去掉。
适合"先把候选找出来给人确认"，不适合替代人工标注 —— 它是**快而粗**的工具。
绝不覆盖输入。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('img')
    ap.add_argument('out')
    ap.add_argument('--color', default='255,0,0', help='目标颜色 R,G,B')
    ap.add_argument('--tol', type=float, default=70, help='色相容差（0-180，越大越宽松）')
    ap.add_argument('--sat-min', type=int, default=60, help='最低饱和度（0-255）')
    ap.add_argument('--val-min', type=int, default=40, help='最低亮度（0-255）')
    ap.add_argument('--dilate', type=int, default=0)
    ap.add_argument('--feather', type=float, default=0)
    ap.add_argument('--min-area', type=int, default=0, help='小于该面积的连通块丢弃')
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--preview', default=None)
    a = ap.parse_args()

    import cv2
    import numpy as np
    from PIL import Image, ImageFilter
    if os.path.abspath(a.out) == os.path.abspath(a.img):
        raise SystemExit('--out 不能覆盖输入')
    bgr = cv2.imread(a.img)
    if bgr is None:
        raise SystemExit('读不到图片：%s' % a.img)
    r, g, b = [int(v) for v in a.color.split(',')]
    target = np.uint8([[[b, g, r]]])
    hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0][0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype('int16'), hsv[..., 1], hsv[..., 2]
    dh = np.minimum(np.abs(h - int(hsv_t[0])), 180 - np.abs(h - int(hsv_t[0])))
    mask = ((dh <= a.tol) & (s >= a.sat_min) & (v >= a.val_min)).astype('uint8') * 255
    if a.invert:
        mask = 255 - mask
    if a.min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'), 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= a.min_area:
                keep[lab == i] = 255
        mask = keep
    out = Image.fromarray(mask, 'L')
    if a.dilate > 0:
        out = out.filter(ImageFilter.MaxFilter(2 * a.dilate + 1))
    if a.feather > 0:
        out = out.filter(ImageFilter.GaussianBlur(a.feather))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out)
    cov = float((np.array(out) > 127).mean())
    print('%s → %s（颜色 %s，覆盖 %.2f%%）' % (a.img, a.out, a.color, cov * 100))
    if a.preview:
        base = np.array(Image.open(a.img).convert('RGB'))
        m = np.array(out) > 127
        base[m] = (0.4 * base[m] + 0.6 * np.array([0, 255, 0])).astype('uint8')
        Image.fromarray(base).save(a.preview)
        print('预览 → %s（绿=命中）' % a.preview)
    return 0


if __name__ == '__main__':
    sys.exit(main())
