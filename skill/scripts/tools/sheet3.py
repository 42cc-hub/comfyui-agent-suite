"""sheet3.py —— 把 N 张图拼成一张带标签的联系表（多方案/多稿对比用）。

    python sheet3.py OUT.png A.png B.png C.png [--labels 方案A,方案B,方案C] [--cell 300x420] [--cols 3]

为什么要它：给人看"三个方案"时，一张并排的图比三张分开的图好判断得多（人对并排差异敏感）。
每格保持同一尺寸与同一起点，标签写在格子左上角。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--labels', default=None, help='逗号分隔；缺省用文件名')
    ap.add_argument('--cell', default='300x420', help='每格像素，如 300x420')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--bg', default='24,24,28')
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    cw, ch = [int(v) for v in a.cell.lower().split('x')]
    cols = max(1, min(a.cols, len(a.images)))
    rows = (len(a.images) + cols - 1) // cols
    lab = 24 if cols > 1 or len(a.images) > 1 else 0
    bg = tuple(int(v) for v in a.bg.split(','))
    sheet = Image.new('RGB', (cols * cw, rows * (ch + lab)), bg)
    d = ImageDraw.Draw(sheet)
    labels = a.labels.split(',') if a.labels else None
    for i, p in enumerate(a.images):
        if not os.path.exists(p):
            raise SystemExit('找不到：%s' % p)
        im = Image.open(p).convert('RGB')
        im.thumbnail((cw, ch), Image.LANCZOS)
        r, c = divmod(i, cols)
        x, y = c * cw, r * (ch + lab)
        sheet.paste(im, (x + (cw - im.width) // 2, y + lab + (ch - im.height) // 2))
        name = (labels[i] if labels and i < len(labels) else os.path.basename(p))[:44]
        d.text((x + 6, y + 6), name, fill=(235, 235, 240))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print('%d 张 → %s（%dx%d，%d 列）' % (len(a.images), a.out, sheet.width, sheet.height, cols))
    return 0


if __name__ == '__main__':
    sys.exit(main())
