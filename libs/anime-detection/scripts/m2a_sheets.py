# -*- coding: utf-8 -*-
"""生成两组未参与训练的图的联络表（用于人工分类 横躺/直立/斜靠）。

A 组：data/hand_gt_multi/holdout15.txt（15 张，原图路径在 path 列）
B 组：data/hand_gt/index.csv（40 张产线图，images/<crop>）
每张缩到长边 520px，2×3 一页，左上角标 stem。
"""
import csv
import io
import os

import cv2
import numpy as np

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'real_lying')
os.makedirs(OUT, exist_ok=True)

items = []
lst = os.path.join(SUITE, 'data', 'hand_gt_multi', 'holdout15.txt')
for line in io.open(lst, encoding='utf-8'):
    c = line.rstrip('\n').split('\t')
    if not c or not c[0] or c[0] == 'stem':
        continue
    items.append(('A', c[0], c[1]))
seen = set()
for r in csv.DictReader(io.open(os.path.join(SUITE, 'data', 'hand_gt', 'index.csv'), encoding='utf-8-sig')):
    stem = os.path.splitext(r['crop'])[0]
    if stem in seen:
        continue
    seen.add(stem)
    items.append(('B', stem, os.path.join(SUITE, 'data', 'hand_gt', 'images', r['crop'])))

print('items', len(items))
tiles = []
for tag, stem, path in items:
    im = cv2.imread(path)
    if im is None:
        print('MISSING', path)
        continue
    h, w = im.shape[:2]
    sc = 520.0 / max(h, w)
    im = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))))
    canvas = np.full((im.shape[0] + 34, im.shape[1], 3), 18, np.uint8)
    canvas[34:, :] = im
    cv2.putText(canvas, '%s %s' % (tag, stem[:34]), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    tiles.append(canvas)

per = 6
for pi in range(0, len(tiles), per):
    chunk = tiles[pi:pi + per]
    H = max(t.shape[0] for t in chunk)
    W = max(t.shape[1] for t in chunk)
    chunk = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, W - t.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))
             for t in chunk]
    while len(chunk) % 3:
        chunk.append(np.zeros((H, W, 3), np.uint8))
    rows = [np.hstack(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]
    sheet = np.vstack(rows)
    p = os.path.join(OUT, 'sheet_%02d.jpg' % (pi // per + 1))
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p, sheet.shape)
