#!/usr/bin/env python
"""人标的 mask → 可训练标签（多边形 / YOLO-seg），保证这条链可重跑。

  python scripts/rope_mask_to_label.py --src data/masks/rope/raw_ps --out data/masks/rope/seg_labels

输入：<src>/<name>_mask.png（二值 0/255，绳子=255；就是人标的真值）
      原图按 --img-dir 顺序查找，用来做边界裁剪与尺寸校验（找不到也照样出标签）。
输出（单类，class 0 = rope）：
  <out>/<name>.txt        YOLO-seg 格式：`0 x1 y1 x2 y2 ...`（归一化 0~1，多边形按外轮廓）
  <out>/<name>.json       {name,size,polygons:[[[x,y],...]],areas:[...]}
  <out>/dataset.jsonl     每行一条 {image,label,size}，供 build_dataset 之类消费

约定：孔洞用 RETR_CCOMP 处理 —— 外轮廓出多边形，洞不单独成实例（绳子 mask 不做 alpha，
      也不分上下压）；小于 --min-area 的块丢弃（与推理端 --min-area 口径一致）。
"""
import argparse
import glob
import io
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_image(name, dirs):
    for d in dirs:
        for ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
        g = glob.glob(os.path.join(d, name + '.*'))
        if g:
            return g[0]
    return None


def main():
    ap = argparse.ArgumentParser(description='绳子 mask → YOLO-seg / 多边形标签')
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'seg_labels'))
    ap.add_argument('--img-dir', default=os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work'))
    ap.add_argument('--min-area', type=int, default=32)
    ap.add_argument('--eps', type=float, default=1.5, help='多边形简化容差（px，越大点越少）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = [a.img_dir, ROOT]
    lines = []
    for mp in sorted(glob.glob(os.path.join(a.src, '*_mask.png'))):
        name = os.path.basename(mp)[:-len('_mask.png')]
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print('读取失败:', mp)
            continue
        h, w = m.shape
        ip = find_image(name, dirs)
        if ip:
            im = cv2.imread(ip, cv2.IMREAD_COLOR)
            if im is not None and im.shape[:2] != (h, w):
                m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
                h, w = m.shape
        binm = (m > 127).astype(np.uint8)
        cnts, hier = cv2.findContours(binm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys, areas, yolo = [], [], []
        for i, c in enumerate(cnts):
            if hier is not None and hier[0][i][3] != -1:
                continue                      # 洞：跳过（单类分割不需要）
            area = cv2.contourArea(c)
            if area < a.min_area:
                continue
            ap_ = cv2.approxPolyDP(c, a.eps, True).reshape(-1, 2)
            if len(ap_) < 3:
                continue
            polys.append([[int(x), int(y)] for x, y in ap_])
            areas.append(float(area))
            yolo.append([v for x, y in ap_ for v in (round(float(x) / w, 6), round(float(y) / h, 6))])
        with io.open(os.path.join(a.out, name + '.txt'), 'w', encoding='utf-8') as fh:
            for row in yolo:
                fh.write('0 ' + ' '.join('%.6f' % v for v in row) + '\n')
        with io.open(os.path.join(a.out, name + '.json'), 'w', encoding='utf-8') as fh:
            json.dump({'name': name, 'image': os.path.relpath(ip, ROOT) if ip else None,
                       'size': [w, h], 'polygons': polys, 'areas': areas},
                      fh, ensure_ascii=False, indent=1)
        lines.append({'image': os.path.relpath(ip, ROOT) if ip else None,
                      'label': os.path.relpath(os.path.join(a.out, name + '.txt'), ROOT),
                      'size': [w, h], 'n_poly': len(polys),
                      'rope_px': int(binm.sum())})
        print('[%s] 多边形 %d 个  绳子 %d px  → %s.txt/.json' % (name, len(polys), binm.sum(), name))
    with io.open(os.path.join(a.out, 'dataset.jsonl'), 'w', encoding='utf-8') as fh:
        for r in lines:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('共 %d 张 → %s' % (len(lines), a.out))


if __name__ == '__main__':
    main()
