# -*- coding: utf-8 -*-
"""task-10 评测：holdout15 在 0/30/45/60/90/135 度上的 per-class R/P（IoU 0.5 贪心一对一，conf 0.30）。

角度桶：0/30/60/90/其它（45、135 归"其它"）。手再按 GT 长边分桶：<72 / 72-110 / >=110 px（在 0 度原图上量）。
产出 team/reports/m2a-verifier/rot_semi/{eval.txt,eval.json}
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
ROT = os.path.join(SUITE, 'data', 'ds_det_rot', 'holdout')
MULTI = os.path.join(SUITE, 'data', 'hand_gt_multi')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'rot_semi')
CLS = {'person': 0, 'hand': 1, 'foot': 2, 'head': 3}
MODELS = {
    'old': os.path.join(SUITE, 'runs', 'm2a_det_996img_960', 'weights', 'best.pt'),
    'v2': os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'),
    'rot': os.path.join(SUITE, 'runs', 'm2a_det_rot_only', 'weights', 'best.pt'),
    'rot_semi': os.path.join(SUITE, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt'),
    'semi': os.path.join(SUITE, 'runs', 'm2a_det_semi_only', 'weights', 'best.pt'),
    # 追补：0° 加权版（45 张 0° 原图 ×3），warm start 自 rot_only
    'bal': os.path.join(SUITE, 'runs', 'm2a_det_rot_bal', 'weights', 'best.pt'),
}
ANGLE_BUCKET = {0: '0', 30: '30', 60: '60', 90: '90', 45: 'other', 135: 'other'}


def load_gt():
    gt = {}
    for line in io.open(os.path.join(MULTI, 'holdout15.txt'), encoding='utf-8'):
        stem = line.split('\t')[0].strip()
        if stem:
            gt[stem] = {}
    for r in csv.DictReader(io.open(os.path.join(MULTI, 'index.csv'), encoding='utf-8-sig')):
        if r['source'] in gt and r['verdict'] == 'draw' and r['boxes'].strip():
            gt[r['source']].setdefault(r['target'], []).extend(
                [[float(v) for v in b.split(',')] for b in r['boxes'].split(';') if b.strip()])
    return gt


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(gts, preds, thr=0.5):
    used = set()
    tp = 0
    for g in gts:
        best, bi = thr, -1
        for j, p in enumerate(preds):
            if j in used:
                continue
            v = iou(g, p)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            tp += 1
            used.add(bi)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--models', default='old,v2,rot,rot_semi,semi')
    ap.add_argument('--out-name', default='eval', help='输出文件名前缀（默认 eval，不覆盖原表时就换名）')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gt = load_gt()
    names = [n for n in a.models.split(',') if n in MODELS and os.path.exists(MODELS[n])]
    from ultralytics import YOLO
    models = {n: YOLO(MODELS[n]) for n in names}
    hand_size = {}
    for f in os.listdir(os.path.join(ROT, 'images')):
        if not f.endswith('__a0.jpg'):
            continue
        stem = f[:-len('__a0.jpg')]
        lab = os.path.join(ROT, 'labels', f[:-4] + '.txt')
        img = cv2.imread(os.path.join(ROT, 'images', f))
        ih, iw = img.shape[:2]
        sizes = []
        for line in io.open(lab, encoding='utf-8'):
            c = line.split()
            if len(c) >= 5 and int(c[0]) == CLS['hand']:
                sizes.append(max(float(c[3]) * iw, float(c[4]) * ih))
        if sizes:
            m = float(np.median(sizes))
            hand_size[stem] = '<72' if m < 72 else ('72-110' if m < 110 else '>=110')

    bucket = {'0': [], '30': [], '60': [], '90': [], 'other': []}
    for f in sorted(os.listdir(os.path.join(ROT, 'images'))):
        if not f.endswith('.jpg'):
            continue
        stem, ang = f[:-4].rsplit('__a', 1)
        ang = int(ang)
        ip = os.path.join(ROT, 'images', f)
        img = cv2.imread(ip)
        ih, iw = img.shape[:2]
        lab = os.path.join(ROT, 'labels', f[:-4] + '.txt')
        gb = {'person': [], 'hand': [], 'foot': [], 'head': []}
        for line in io.open(lab, encoding='utf-8'):
            c = line.split()
            if len(c) < 5:
                continue
            cid = int(c[0])
            name_cls = {v: k for k, v in CLS.items()}[cid]
            cx, cy, bw, bh = [float(v) for v in c[1:5]]
            gb[name_cls].append([(cx - bw / 2) * iw, (cy - bh / 2) * ih,
                                 (cx + bw / 2) * iw, (cy + bh / 2) * ih])
        gb.pop('head')
        for name, model in models.items():
            res = model.predict(ip, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            pr = {}
            for b in (res.boxes or []):
                pr.setdefault(res.names[int(b.cls[0])], []).append([float(v) for v in b.xyxy[0].tolist()])
            for cls in ('person', 'hand', 'foot'):
                g = gb.get(cls, [])
                p = pr.get(cls, [])
                if not g and not p:
                    continue
                tp, fp, fn = score(g, p)
                bucket[ANGLE_BUCKET[ang]].append(
                    {'stem': stem, 'angle': ang, 'model': name, 'cls': cls,
                     'tp': tp, 'fp': fp, 'fn': fn, 'gt': len(g), 'pred': len(p),
                     'hand_size': hand_size.get(stem, '-')})
    lines = []

    def say(s=''):
        print(s)
        lines.append(s)
    say('# task-10 holdout15（15 张，永不入训练）R/P  IoU 0.5 / conf %.2f / imgsz %d' % (a.conf, a.imgsz))
    say('')
    for cls in ('person', 'hand', 'foot'):
        say('## %s' % cls)
        say('| 角度桶 | 模型 | GT | TP | FP | R | P |')
        say('|---|---|---|---|---|---|---|')
        for bk in ('0', '30', '60', '90', 'other'):
            rows = [r for r in bucket[bk] if r['cls'] == cls]
            if not rows:
                continue
            for name in names:
                rr = [r for r in rows if r['model'] == name]
                if not rr:
                    continue
                tp = sum(r['tp'] for r in rr)
                fp = sum(r['fp'] for r in rr)
                fn = sum(r['fn'] for r in rr)
                g = sum(r['gt'] for r in rr)
                say('| %s | %s | %d | %d | %d | %.1f%% | %.1f%% |'
                    % (bk, name, g, tp, fp, 100.0 * tp / max(1, tp + fn), 100.0 * tp / max(1, tp + fp)))
        say('')
    say('## 手按尺寸桶（各角度合计）')
    say('| 尺寸桶 | 模型 | GT | TP | FP | R | P |')
    say('|---|---|---|---|---|---|---|')
    for hs in ('<72', '72-110', '>=110'):
        for name in names:
            rr = [r for r in bucket_all(bucket) if r['cls'] == 'hand' and r['model'] == name and r['hand_size'] == hs]
            if not rr:
                continue
            tp = sum(r['tp'] for r in rr)
            fp = sum(r['fp'] for r in rr)
            fn = sum(r['fn'] for r in rr)
            say('| %s | %s | %d | %d | %d | %.1f%% | %.1f%% |'
                % (hs, name, sum(r['gt'] for r in rr), tp, fp,
                   100.0 * tp / max(1, tp + fn), 100.0 * tp / max(1, tp + fp)))
    say('')
    say('## 全角度合计')
    say('| 模型 | person R/P | hand R/P | foot R/P |')
    say('|---|---|---|---|')
    for name in names:
        cells = []
        for cls in ('person', 'hand', 'foot'):
            rr = [r for r in bucket_all(bucket) if r['cls'] == cls and r['model'] == name]
            tp = sum(r['tp'] for r in rr)
            fp = sum(r['fp'] for r in rr)
            fn = sum(r['fn'] for r in rr)
            cells.append('%.1f%% / %.1f%%' % (100.0 * tp / max(1, tp + fn), 100.0 * tp / max(1, tp + fp)))
        say('| %s | %s |' % (name, ' | '.join(cells)))
    with io.open(os.path.join(OUT, a.out_name + '.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    json.dump(bucket, io.open(os.path.join(OUT, a.out_name + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('wrote', os.path.join(OUT, a.out_name + '.txt'))


def bucket_all(bucket):
    out = []
    for v in bucket.values():
        out += v
    return out


if __name__ == '__main__':
    main()
