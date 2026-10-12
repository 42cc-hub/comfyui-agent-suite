"""crop_view.py —— 裁一块区域放大看，并可叠加坐标网格（沟通"哪里"的最省事工具）。

    python crop_view.py IMG.png OUT.png x0,y0,x1,y1 [--scale 1.5] [--grid 100] [--label]

为什么需要它：和人或模型讨论"画面左上那块"这种话没有精度；裁出来 + 网格 + 坐标，
"把 (740,380)-(1000,560) 这块重画"才是一句可执行的话。坐标一律用**原图像素**。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('img')
    ap.add_argument('out')
    ap.add_argument('box', help='x0,y0,x1,y1（原图像素）')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--grid', type=int, default=0, help='网格间距（原图像素；0=不画）')
    ap.add_argument('--label', action='store_true', help='在图上标出原图坐标')
    ap.add_argument('--pad', type=int, default=0, help='向外扩多少像素')
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    im = Image.open(a.img).convert('RGB')
    x0, y0, x1, y1 = [int(float(v)) for v in a.box.split(',')]
    x0, y0 = max(0, x0 - a.pad), max(0, y0 - a.pad)
    x1, y1 = min(im.width, x1 + a.pad), min(im.height, y1 + a.pad)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit('坐标不合法：%s' % a.box)
    crop = im.crop((x0, y0, x1, y1))
    if a.scale != 1.0:
        crop = crop.resize((max(1, int(crop.width * a.scale)), max(1, int(crop.height * a.scale))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    if a.grid:
        for gx in range(0, crop.width, max(1, int(a.grid * a.scale))):
            d.line([(gx, 0), (gx, crop.height)], fill=(255, 0, 0), width=1)
        for gy in range(0, crop.height, max(1, int(a.grid * a.scale))):
            d.line([(0, gy), (crop.width, gy)], fill=(255, 0, 0), width=1)
    if a.label:
        d.text((4, 4), '%s  (%d,%d)-(%d,%d)  x%.2f' % (os.path.basename(a.img), x0, y0, x1, y1, a.scale),
               fill=(255, 255, 0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    crop.save(a.out)
    print('%s → %s  %dx%d  （原图坐标 (%d,%d)-(%d,%d)）' % (a.img, a.out, crop.width, crop.height, x0, y0, x1, y1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
