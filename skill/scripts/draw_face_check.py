#!/usr/bin/env python
"""draw_face_check.py —— 把"面部检测到底准不准"画出来给人看（口/鼻/眼三件事并排对照）。

用途（2026-10-13 用户要求）：判断"鼻点落在嘴上 / 嘴位落在下巴"这类错误是**普遍**还是**偶然**。
把同一批图上各家给出的位置叠在一起，人一眼就能判：
  · 黄点 = **M1 pose** 的鼻/眼/耳（COCO-17，**里面根本没有嘴点**）
  · 绿框 = **M2b face** 的 face 框 / 橙框 = eye 框
  · 青框 = **由脸框推出来的嘴**（脸框下 42%）—— 这是工具现在给的那个"嘴"
  · 品红短线 = 旧写法"鼻点往下 0.22×脸高"推出来的嘴（报告里判定为**错**的那一版）
每格下面写：文件名 / 鼻点置信度 / 检测到几只眼 / 脸框高度，便于对照"错得离不离谱"。

用法：
  python draw_face_check.py --out <check_parts 的输出目录> --sheet <输出表.png> [--per-row 2] [--cell 620]
需要同目录下 check_parts.py 生成的 <name>.json（`--src` 给同样的图）。
"""
import argparse
import glob
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--out', required=True, help='check_parts 的输出目录（里面有 <name>.json）')
    ap.add_argument('--src', nargs='+', required=True, help='同一批原图（按文件名匹配 JSON）')
    ap.add_argument('--sheet', required=True, help='输出联系表 PNG')
    ap.add_argument('--per-row', type=int, default=2)
    ap.add_argument('--cell', type=int, default=620, help='每格边长（px，越大越好看细节）')
    ap.add_argument('--pad', type=float, default=0.28, help='head 框外扩比例（把嘴/下巴也框进来）')
    ap.add_argument('--region', choices=['head', 'mouth'], default='head',
                    help='head=整头（默认）；mouth=只放大"脸框下半"那一块，专门看嘴')
    a = ap.parse_args()

    src_map = {}
    for s in a.src:
        if os.path.isdir(s):
            for e in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
                src_map.update({os.path.basename(p): p for p in glob.glob(os.path.join(s, e))})
        else:
            src_map[os.path.basename(s)] = s

    cells = []
    for jp in sorted(glob.glob(os.path.join(a.out, '*.json'))):
        if os.path.basename(jp) == 'summary.json':
            continue
        rec = json.load(io.open(jp, encoding='utf-8'))
        img_name = rec.get('image')
        sp = src_map.get(os.path.basename(img_name or ''))
        if not sp or not os.path.exists(sp):
            print('跳过（找不到原图）%s' % img_name)
            continue
        heads = [b for b in (rec.get('boxes_all') or rec.get('boxes', [])) if b.get('cls') == 'head']
        if not heads:
            print('跳过（没有 head 框）%s' % img_name)
            continue
        faces_all = [b for b in (rec.get('boxes_all') or []) if b.get('cls') == 'face']
        if a.region == 'mouth':
            if not faces_all:
                print('跳过（没有 face 框，无法定位嘴）%s' % img_name)
                continue
            fb = max(faces_all, key=lambda b: (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]))['box']
            fw, fh = fb[2] - fb[0], fb[3] - fb[1]
            hb = [fb[0] - 0.18 * fw, fb[1] + 0.34 * fh, fb[2] + 0.18 * fw, fb[3] + 0.18 * fh]
        else:
            hb = max(heads, key=lambda b: (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]))['box']
        W, H = hb[2] - hb[0], hb[3] - hb[1]
        pad = a.pad * max(W, H)
        x0, y0 = max(0, int(hb[0] - pad)), max(0, int(hb[1] - pad))
        x1, y1 = min(Image.open(sp).width, int(hb[2] + pad)), min(Image.open(sp).height, int(hb[3] + pad))
        crop = Image.open(sp).convert('RGB').crop((x0, y0, x1, y1))
        scale = a.cell / float(max(crop.size))
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        T = lambda bx, by: ((bx - x0) * scale, (by - y0) * scale)   # noqa: E731

        def rect(box, color, w=2):
            (px0, py0), (px1, py1) = T(box[0], box[1]), T(box[2], box[3])
            d.rectangle([px0, py0, px1, py1], outline=color, width=w)

        faces = [b for b in (rec.get('boxes_all') or []) if b.get('cls') == 'face']
        for b in faces:
            rect(b['box'], (0, 255, 0), 2)
        for b in (rec.get('boxes_all') or []):
            if b.get('cls') == 'eye':
                rect(b['box'], (255, 160, 0), 2)
        # 工具实际给出的嘴（`--feature-masks` 的 mouth*）优先；没有才退回几何估计
        mouths = [f for f in rec.get('feature_masks', []) if f.get('cls') == 'mouth']
        if mouths:
            for f in mouths:
                rect(f['box'], (0, 230, 230), 2)
        else:
            for b in faces:
                bx0, by0, bx1, by1 = [float(v) for v in b['box']]
                fw, fh = bx1 - bx0, by1 - by0
                rect([bx0 + 0.12 * fw, by0 + 0.58 * fh, bx1 - 0.12 * fw, by1], (0, 230, 230), 2)
        # M1 点
        pts = [p for p in rec.get('points', []) if p.get('cls') == 'body']
        nose = next((p for p in pts if p.get('name', '').endswith('.nose')), None)
        for p in pts:
            nm = p.get('name', '').split('.')[-1]
            if nm in ('left_eye', 'right_eye', 'left_ear', 'right_ear'):
                col = (255, 255, 0) if 'eye' in nm else (255, 120, 255)
                cx, cy = T(p['x'], p['y'])
                r = 5 if 'eye' in nm else 4
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
        cap = '%s  nose=%.2f  eyes=%d  faces=%d' % (
            os.path.basename(img_name), (nose or {}).get('prob', 0),
            sum(1 for p in pts if p.get('name', '').endswith(('left_eye', 'right_eye'))),
            len(faces))
        if nose and faces:
            b = max(faces, key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]))
            fh = b['box'][3] - b['box'][1]
            old_y = nose['y'] + 0.22 * fh
            cx, cy = T(nose['x'], old_y)
            d.line([cx - 26, cy, cx + 26, cy], fill=(255, 0, 255), width=4)
        if nose:
            cx, cy = T(nose['x'], nose['y'])
            d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 255, 0), width=3)
            d.text((cx + 9, cy - 16), 'nose', fill=(255, 255, 0))
        cells.append((crop, cap))

    if not cells:
        raise SystemExit('没有可比对的图')
    per = a.per_row
    rows = (len(cells) + per - 1) // per
    cw = a.cell
    ch = max(c[0].height for c in cells) + 26
    sheet = Image.new('RGB', (per * cw, rows * ch), (12, 12, 12))
    ds = ImageDraw.Draw(sheet)
    for i, (im, cap) in enumerate(cells):
        cx, cy = (i % per) * cw, (i // per) * ch
        sheet.paste(im, (cx, cy))
        ds.text((cx + 6, cy + im.height + 6), cap[:110], fill=(240, 240, 240))
    sheet.save(a.sheet)
    print('%d 格 → %s' % (len(cells), a.sheet))


if __name__ == '__main__':
    main()
