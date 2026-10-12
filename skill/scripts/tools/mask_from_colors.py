"""mask_from_colors.py —— 把"分色块图"转成掩版（配合视觉模型做语义分割那条路）。

用途：让视觉模型按"把 A 涂红、B 涂蓝、背景涂黑"输出一张**纯色块图**（实测 DashScope `wan2.7-image`
能稳定做到），再用本脚本按颜色阈值转成掩版。也适用于任何"用颜色标记区域"的产物（人画的、程序画的）。

    python mask_from_colors.py COLORBLOCK.png --out-dir out/masks --size 1024x1536 \
        [--colors "red=#FF0000,blue=#0000FF"] [--tol 90] [--grow 0] [--rgba-base BASE.png] [--preview]

输出（每个颜色一份）：
    <name>.png        灰度掩版（**白 = 该颜色区域**，可直接当 mask 用）
    <name>_rgba.png   给了 `--rgba-base` 才有：RGBA，**该区域 alpha=0**（ComfyUI LoadImage 直接用）

要点：
  · 分色图的尺寸常与原图不同（模型会放大）⇒ 一定给 `--size` 或 `--rgba-base`，并用**最近邻**缩放；
  · **选的颜色不要和图里已有颜色撞**（比如画面本来有红绳，就别用红当角色色，改用品红）；
  · 边缘会有 1~2 px 抗锯齿（实测 1.9% 像素不属于任何纯色）⇒ `--grow 1~2` 可以补上，但别指望它逐像素精确。
"""
import argparse
import os
import re
import sys


def parse_color(s):
    if s.startswith('#'):
        h = s[1:]
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = [int(v) for v in s.split(',')]
    return (r, g, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('colorblock')
    ap.add_argument('--out-dir', default='masks')
    ap.add_argument('--colors', default='red=#FF0000,blue=#0000FF,yellow=#FFFF00',
                    help='name=#RRGGBB 或 name=R,G,B，逗号分隔')
    ap.add_argument('--size', default=None, help='输出尺寸 WxH（应等于原图尺寸）')
    ap.add_argument('--rgba-base', default=None, help='原图；给了就额外输出 <name>_rgba.png')
    ap.add_argument('--tol', type=int, default=90, help='颜色距离阈值（L1，0-765）')
    ap.add_argument('--grow', type=int, default=0, help='掩版膨胀像素（补抗锯齿边缘）')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--check-boxes', default=None,
                    help='check_parts.py 的 JSON：核对每种颜色落在哪个 person 框里 + 数碎片（生成式掩版的质检）')
    a = ap.parse_args()

    import numpy as np
    from PIL import Image, ImageFilter
    if a.rgba_base:
        base = Image.open(a.rgba_base).convert('RGB')
        size = base.size
    elif a.size:
        w, h = [int(v) for v in a.size.lower().split('x')]
        size = (w, h); base = None
    else:
        base = None
        size = Image.open(a.colorblock).size
    cb = Image.open(a.colorblock).convert('RGB').resize(size, Image.NEAREST)
    arr = np.array(cb).astype('int16')
    os.makedirs(a.out_dir, exist_ok=True)
    masks, total_other = {}, None
    for item in a.colors.split(','):
        name, val = item.split('=')
        col = np.array(parse_color(val.strip()), dtype='int16')
        m = np.abs(arr - col).sum(axis=2) < a.tol
        masks[name.strip()] = m
        img = Image.fromarray((m * 255).astype('uint8'), 'L')
        if a.grow > 0:
            img = img.filter(ImageFilter.MaxFilter(2 * a.grow + 1))
            m = np.array(img) > 127
        img.save(os.path.join(a.out_dir, '%s.png' % name.strip()))
        line = '%-10s %7.2f%%' % (name.strip(), 100 * m.mean())
        if base is not None:
            rgba = np.dstack([np.array(base), np.where(m, 0, 255).astype('uint8')])
            Image.fromarray(rgba, 'RGBA').save(os.path.join(a.out_dir, '%s_rgba.png' % name.strip()))
            line += '  (+ _rgba.png)'
        print(line)
        total_other = m.copy() if total_other is None else (total_other | m)
    if len(masks) > 1:
        ks = list(masks)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                inter = int((masks[ks[i]] & masks[ks[j]]).sum())
                if inter:
                    print('⚠ %s ∩ %s = %d px（分色图不该重叠；阈值放太宽了？）' % (ks[i], ks[j], inter))
        print('各色并集覆盖 %.2f%%' % (100 * total_other.mean()))
    if a.preview and base is not None:
        pal = [(255, 40, 40), (40, 80, 255), (255, 220, 40), (80, 255, 120), (210, 90, 255)]
        arr2 = np.array(base).astype('float32')
        for i, (k, m) in enumerate(masks.items()):
            c = np.array(pal[i % len(pal)], dtype='float32')
            arr2[m] = arr2[m] * 0.45 + c * 0.55
        pv = os.path.join(a.out_dir, 'colors_overlay.png')
        Image.fromarray(arr2.round().clip(0, 255).astype('uint8')).save(pv)
        print('预览 → %s' % pv)

    if a.check_boxes:
        # 生成式掩版的质检：① 碎片多不多 ② 每种颜色主要落在哪个 person 框里（判断有没有"画错人"）
        import json
        import cv2
        d = json.load(open(a.check_boxes, encoding='utf-8'))
        d = (d if isinstance(d, list) else [d])[0]
        pboxes = {b.get('person_id'): [int(v) for v in b['box']] for b in (d.get('boxes_all') or [])
                  if b.get('cls') == 'person'}
        print('\n-- 质检（%s 里的 person 框）--' % os.path.basename(a.check_boxes))
        for k, m in masks.items():
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(m.astype('uint8'), 8)
            areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True) if n > 1 else []
            big = [x for x in areas if x > m.sum() * 0.002]
            frac = []
            for pid, bx in pboxes.items():
                inner = np.zeros_like(m)
                inner[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]] = True
                frac.append((pid, (m & inner).sum() / max(1, m.sum())))
            frac.sort(key=lambda t: -t[1])
            print('   %-8s 连通块 %2d（>0.2%% 的 %d）；最匹配 person pid=%s（框内占比 %.2f）'
                  % (k, max(0, n - 1), len(big), frac[0][0] if frac else '-', frac[0][1] if frac else 0))
            if len(big) > 3:
                print('            ⚠ 碎片偏多（>3 块）—— 提示词可能没被完全遵守，建议重跑或换颜色')
            if len(frac) > 1 and frac[0][1] - frac[1][1] < 0.15:
                print('            ⚠ 与第二名的框内占比只差 %.2f ⇒ 这块到底是谁不够明确，务必叠加核对'
                      % (frac[0][1] - frac[1][1]))
    print('\n灰度掩版：白=该色区域；RGBA 掩版：该色区域 alpha=0（ComfyUI `LoadImage` 直接用）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
