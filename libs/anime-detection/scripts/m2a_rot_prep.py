# -*- coding: utf-8 -*-
"""task-10 数据准备：holdout 切分 + 旋转增强数据集（图与框同步旋转）。

产出：
  data/hand_gt_multi/holdout15.txt / train45.txt   —— 冻结划分（stem 一行一个，带路径）
  data/ds_det_rot/images/*.jpg + labels/*.txt      —— train45 的 0/30/60/90 度四份（4 类：person/hand/foot，无 head）
  data/ds_det_rot/holdout/images|labels            —— holdout15 每张 0/30/45/60/90/135 度（评测用，45/135 归"其它"桶）
  data/ds_det_rot/build_stats.csv                  —— 每张图每角度的丢弃/裁剪统计
规则：
  * 旋转用 cv2.getRotationMatrix2D(center, angle, 1.0) + warpAffine，同一矩阵变换框角点，保证图框一致；
  * expand=False（同尺寸画布、黑色填充）—— 不缩小目标，避免额外改变目标尺度；
  * 框裁到画布内后，若保留面积 < 原面积 60% 或任一边 < 4px ⇒ 丢弃该框；每图至少剩 1 框才保留该角度。
"""
import argparse
import csv
import io
import json
import os
import sys

import cv2
import numpy as np

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(SUITE, 'data', 'ds_det_rot')
MULTI = os.path.join(SUITE, 'data', 'hand_gt_multi')
GT_CSV = os.path.join(MULTI, 'index.csv')
IMG_LIST = os.path.join(MULTI, 'image_list.txt')
PREDS = os.path.join(MULTI, 'preds', 'queue_preds.json')
CLS = {'person': 0, 'hand': 1, 'foot': 2, 'head': 3}
TRAIN_ANGLES = (0, 30, 60, 90)
HOLDOUT_ANGLES = (0, 30, 45, 60, 90, 135)
KEEP_RATIO = 0.60
MIN_SIDE = 4.0


def load_gt():
    per = {}
    for r in csv.DictReader(io.open(GT_CSV, encoding='utf-8-sig')):
        stem = r['source']
        d = per.setdefault(stem, {'crop': r['crop'], 'targets': {}})
        if r['verdict'] == 'draw' and r['boxes'].strip():
            boxes = [[float(v) for v in b.split(',')] for b in r['boxes'].split(';') if b.strip()]
            d['targets'].setdefault(r['target'], []).extend(boxes)
    return per


def load_paths():
    m = {}
    with io.open(IMG_LIST, encoding='utf-8') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) >= 2:
                m[c[0]] = c[1]
    return m


def pick_holdout(gt, preds):
    """15 张：3 张手小（image_list hand_small=1）+ 5 张双人框重叠最高 + 其余按 stem 排序补齐。"""
    small = []
    with io.open(IMG_LIST, encoding='utf-8') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) >= 6 and c[5] == '1':
                small.append(c[0])

    def overlap(stem):
        p = preds.get(stem, {}).get('det', {}).get('person', [])
        best = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                a, b = p[i]['box'], p[j]['box']
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                best = max(best, inter / ua if ua > 0 else 0.0)
        return best

    all_stems = sorted(gt)
    hold = []
    for s in sorted(small, key=lambda z: (-overlap(z), z))[:3]:
        hold.append(s)
    cand = sorted((s for s in all_stems if s not in hold and overlap(s) > 0.0),
                  key=lambda s: (-overlap(s), s))
    picked = 0
    for s in cand:
        if picked >= 5:
            break
        hold.append(s)
        picked += 1
    for s in all_stems:
        if len(hold) >= 15:
            break
        if s not in hold:
            hold.append(s)
    return hold[:15]


def rotate_image(img, boxes, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    kept, drop = [], 0
    for (x0, y0, x1, y1) in boxes:
        corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]]).T
        p = M @ corners
        xs, ys = p[0], p[1]
        cx0, cx1 = max(0.0, float(xs.min())), min(float(w), float(xs.max()))
        cy0, cy1 = max(0.0, float(ys.min())), min(float(h), float(ys.max()))
        bw, bh = cx1 - cx0, cy1 - cy0
        if bw < MIN_SIDE or bh < MIN_SIDE:
            drop += 1
            continue
        if bw * bh < KEEP_RATIO * (x1 - x0) * (y1 - y0):
            drop += 1
            continue
        kept.append([cx0, cy0, cx1, cy1])
    return out, kept, drop


def write_yolo(label_path, boxes_by_cls, w, h):
    lines = []
    for cls, boxes in boxes_by_cls.items():
        for (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append('%d %.6f %.6f %.6f %.6f' % (CLS[cls], cx, cy, bw, bh))
    io.open(label_path, 'w', encoding='utf-8').write('\n'.join(lines) + ('\n' if lines else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-holdout', action='store_true')
    a = ap.parse_args()
    gt = load_gt()
    paths = load_paths()
    preds = json.load(io.open(PREDS, encoding='utf-8'))
    hold = pick_holdout(gt, preds)
    assert len(hold) == 15, len(hold)
    train = [s for s in sorted(gt) if s not in hold]
    missing = [s for s in gt if not os.path.exists(paths.get(s, ''))]
    if missing:
        print('!! 源图缺失', missing[:5])
    for name, stems in (('holdout15.txt', hold), ('train45.txt', train)):
        with io.open(os.path.join(MULTI, name), 'w', encoding='utf-8') as f:
            for s in stems:
                f.write('%s\t%s\n' % (s, paths[s]))
    print('holdout15:', len(hold), 'train45:', len(train))
    print('holdout 手小:', sum(1 for s in hold if s in open_img_small()), '（image_list hand_small=1）')

    if a.only_holdout:
        return
    tr_img = os.path.join(OUT, 'images')
    tr_lab = os.path.join(OUT, 'labels')
    ho_img = os.path.join(OUT, 'holdout', 'images')
    ho_lab = os.path.join(OUT, 'holdout', 'labels')
    for d in (tr_img, tr_lab, ho_img, ho_lab):
        os.makedirs(d, exist_ok=True)
    stats = []
    for split, stems, angles, idir, ldir in (('train', train, TRAIN_ANGLES, tr_img, tr_lab),
                                            ('holdout', hold, HOLDOUT_ANGLES, ho_img, ho_lab)):
        for stem in stems:
            img = cv2.imread(paths[stem], cv2.IMREAD_COLOR)
            if img is None:
                print('!! imread fail', stem)
                continue
            h, w = img.shape[:2]
            by_cls = {k: v for k, v in gt[stem]['targets'].items()}
            for angle in angles:
                rot, boxes, drops = rotate_image(img, [b for v in by_cls.values() for b in v], angle)
                kept_by = {k: [] for k in by_cls}
                for cls, v in by_cls.items():
                    for b in v:
                        r, k, d = rotate_image(img, [b], angle)
                        if k:
                            kept_by[cls].append(k[0])
                if not any(kept_by.values()):
                    stats.append([split, stem, angle, 'drop-image', 0, drops])
                    continue
                name = '%s__a%d' % (stem, angle)
                cv2.imwrite(os.path.join(idir, name + '.jpg'), rot, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_yolo(os.path.join(ldir, name + '.txt'), kept_by, w, h)
                stats.append([split, stem, angle, 'ok',
                              sum(len(v) for v in kept_by.values()), drops])
    with io.open(os.path.join(OUT, 'build_stats.csv'), 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'stem', 'angle', 'status', 'kept_boxes', 'dropped_boxes'])
        wr.writerows(stats)
    ok = [s for s in stats if s[3] == 'ok']
    print('写出 %d 个样本（train %d / holdout %d），丢弃框 %d'
          % (len(ok), sum(1 for s in stats if s[0] == 'train' and s[3] == 'ok'),
             sum(1 for s in stats if s[0] == 'holdout' and s[3] == 'ok'),
             sum(s[5] for s in stats)))


def open_img_small():
    small = set()
    with io.open(IMG_LIST, encoding='utf-8') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) >= 6 and c[5] == '1':
                small.add(c[0])
    return small


if __name__ == '__main__':
    main()
