# -*- coding: utf-8 -*-
"""阶段 B 数据准备：batch-1 + batch-2 person 真值 + 30/60/90 旋转增强，按项目切 holdout。

产物 data/ds_person/：
  images/train|val, labels/train|val     （rot 变体命名 <stem>__a<ang>）
  person.yaml                            （4 类名保留，便于 warm start；标签只有 person=0）
  build_stats.csv / split.json
口径与 task-10 的旋转一致：cv2.getRotationMatrix2D + warpAffine(expand=False, 黑边)；
框四角变换后裁回画布，面积保留 <0.60 或边 <4px 的框丢弃；一角度全丢则丢该图。
"""
import csv
import io
import json
import os
import shutil

import cv2
import numpy as np

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(SUITE, 'data')
OUT = os.path.join(DATA, 'ds_person')
IMG = os.path.join(DATA, 'hand_gt_multi', 'images')
KEEP_RATIO, MIN_SIDE = 0.60, 4.0
ANGLES = (0, 30, 60, 90)
CLASSES = ['person', 'hand', 'foot', 'head']


def boxes_of(s):
    return [tuple(float(v) for v in seg.split(',')) for seg in (s or '').split(';') if seg.strip()]


def collect():
    """-> {crop: [person boxes]}。

    注意：batch-2（`index_person_topup.csv`）是**同一批 60 张图的 50 行校正版**（同 crop、同框数，
    实测 64 对里 51 对完全一致、7 对 IoU<0.95），所以必须**覆盖**而不是累加 batch-1，否则同一图会
    被重复计数（曾得到 60 图/161 框的错误结果）。"""
    gt = {}
    for path, target in ((os.path.join(DATA, 'hand_gt_multi', 'index.csv'), None),
                         (os.path.join(DATA, 'hand_gt_multi', 'index_person_topup.csv'), 'person')):
        for r in csv.DictReader(io.open(path, encoding='utf-8-sig')):
            if r['verdict'].strip() == 'neg':
                continue
            if r['target'] != 'person':
                continue
            b = boxes_of(r['boxes'])
            if b:
                gt[r['crop']] = b          # 后读的 batch-2 覆盖 batch-1
    return gt


def projects():
    p = os.path.join(DATA, 'hand_gt_multi', 'tags_delivered.csv')
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            out[r['stem'] + '.png'] = r.get('project', '')
    return out


def rot_boxes(boxes, M, w, h):
    out = []
    for b in boxes:
        pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]], dtype=np.float64)
        q = pts @ M.T
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
            continue
        orig = (b[2] - b[0]) * (b[3] - b[1])
        if orig > 0 and (x1 - x0) * (y1 - y0) / orig < KEEP_RATIO:
            continue
        out.append((x0, y0, x1, y1))
    return out


def write_split(root, items, stats, tag):
    for crop, boxes in items:
        img = cv2.imread(os.path.join(IMG, crop))
        if img is None:
            stats.append((tag, crop, 'MISSING', 0, 0))
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(crop)[0]
        for ang in ANGLES:
            if ang == 0:
                im, bb = img, boxes
            else:
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
                im = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                bb = rot_boxes(boxes, M, w, h)
            if not bb:
                stats.append((tag, crop, 'drop_all_a%d' % ang, 0, 0))
                continue
            name = '%s__a%d' % (stem, ang)
            cv2.imwrite(os.path.join(root, 'images', tag, name + '.jpg'), im,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with io.open(os.path.join(root, 'labels', tag, name + '.txt'), 'w', encoding='utf-8') as f:
                for b in bb:
                    f.write('0 %.6f %.6f %.6f %.6f\n' % ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h,
                                                         (b[2] - b[0]) / w, (b[3] - b[1]) / h))
            stats.append((tag, crop, 'a%d' % ang, len(bb), ang))


def main():
    gt = collect()
    proj = projects()
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    # 按项目切：holdout 取项目内张数最少但覆盖 ≥18% 图的整项目组合
    by_proj = {}
    for crop in gt:
        by_proj.setdefault(proj.get(crop, 'unknown:' + crop), []).append(crop)
    order = sorted(by_proj, key=lambda p: (len(by_proj[p]), p))
    need = max(1, int(round(0.2 * len(gt))))
    val_proj, n = [], 0
    for p in order:
        if n >= need and len(val_proj) >= 2:
            break
        # 单个项目太大（>40% 图）不整体留出，避免训练集被抽空
        if len(by_proj[p]) > 0.4 * len(gt):
            continue
        val_proj.append(p)
        n += len(by_proj[p])
    val_set = {c for p in val_proj for c in by_proj[p]}
    train = [(c, gt[c]) for c in sorted(gt) if c not in val_set]
    val = [(c, gt[c]) for c in sorted(gt) if c in val_set]
    stats = []
    shutil.rmtree(os.path.join(OUT, 'images'), ignore_errors=True)
    shutil.rmtree(os.path.join(OUT, 'labels'), ignore_errors=True)
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    write_split(OUT, train, stats, 'train')
    write_split(OUT, val, stats, 'val')
    with io.open(os.path.join(OUT, 'person.yaml'), 'w', encoding='utf-8') as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n' % OUT)
        for i, c in enumerate(CLASSES):
            f.write('  %d: %s\n' % (i, c))
    # 同项目重叠统计
    tr_proj = {proj.get(c, '?') for c, _ in train}
    overlap = [c for c in val_set if proj.get(c, '?') in tr_proj]
    with io.open(os.path.join(OUT, 'split.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(images=len(gt), boxes=sum(len(v) for v in gt.values()),
                       train_imgs=len(train), val_imgs=len(val), val_projects=val_proj,
                       val_gt_boxes=sum(len(v) for _, v in val),
                       val_same_project_as_train=overlap,
                       n_projects=len(by_proj)), f, ensure_ascii=False, indent=1)
    with io.open(os.path.join(OUT, 'build_stats.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'crop', 'status', 'n_boxes', 'angle'])
        w.writerows(stats)
    n_tr = sum(1 for s in stats if s[0] == 'train' and str(s[2]).startswith('a'))
    n_va = sum(1 for s in stats if s[0] == 'val' and str(s[2]).startswith('a'))
    print('images %d boxes %d | train %d imgs × %d rot = %d files | val %d imgs → %d files'
          % (len(gt), sum(len(v) for v in gt.values()), len(train), len(ANGLES), n_tr,
             len(val), n_va))
    print('val projects', val_proj, '| val same-project-as-train imgs', len(overlap))
    print('yaml', os.path.join(OUT, 'person.yaml'))


if __name__ == '__main__':
    main()
