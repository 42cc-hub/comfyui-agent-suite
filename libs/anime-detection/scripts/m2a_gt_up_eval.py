# -*- coding: utf-8 -*-
"""task-10 补充：upright（未旋转）独立泛化检查。

用 `data/hand_gt`（40 张产线交付图，用户手/脚真值，**未参与本批任何训练**）在原始朝向上评
old / v2 / rot / rot_semi / semi 的 hand & foot R/P（IoU 0.5、conf 0.30、imgsz 960）。
产出 team/reports/m2a-verifier/rot_semi/upright_eval.txt
"""
import csv
import io
import os

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
HG = os.path.join(SUITE, 'data', 'hand_gt')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'rot_semi')
MODELS = {
    'old': os.path.join(SUITE, 'runs', 'm2a_det_996img_960', 'weights', 'best.pt'),
    'v2': os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'),
    'rot': os.path.join(SUITE, 'runs', 'm2a_det_rot_only', 'weights', 'best.pt'),
    'rot_semi': os.path.join(SUITE, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt'),
    'semi': os.path.join(SUITE, 'runs', 'm2a_det_semi_only', 'weights', 'best.pt'),
    # 追补：0° 加权版
    'bal': os.path.join(SUITE, 'runs', 'm2a_det_rot_bal', 'weights', 'best.pt'),
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(gts, preds, thr=0.5):
    used, tp = set(), 0
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
    return tp, len(preds) - tp, len(gts) - tp


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = {}
    for r in csv.DictReader(io.open(os.path.join(HG, 'index.csv'), encoding='utf-8-sig')):
        if r['verdict'] == 'draw' and r['boxes'].strip():
            gt.setdefault(os.path.splitext(r['crop'])[0], {}).setdefault(r['target'], []).extend(
                [[float(v) for v in b.split(',')] for b in r['boxes'].split(';') if b.strip()])
    imgs = sorted(os.listdir(os.path.join(HG, 'images')))
    from ultralytics import YOLO
    models = {k: YOLO(v) for k, v in MODELS.items() if os.path.exists(v)}
    res = {}
    for tag, m in models.items():
        agg = {}
        for f in imgs:
            stem = os.path.splitext(f)[0]
            if stem not in gt:
                continue
            p = os.path.join(HG, 'images', f)
            r0 = m.predict(p, conf=0.30, imgsz=960, verbose=False)[0]
            pr = {}
            for b in (r0.boxes or []):
                pr.setdefault(r0.names[int(b.cls[0])], []).append([float(v) for v in b.xyxy[0].tolist()])
            for cls in ('hand', 'foot'):
                g = gt[stem].get(cls, [])
                pv = pr.get(cls, [])
                tp, fp, fn = score(g, pv)
                a = agg.setdefault(cls, [0, 0, 0, 0])
                a[0] += len(g)
                a[1] += tp
                a[2] += fp
                a[3] += fn
        res[tag] = agg
    lines = ['# upright 独立集（data/hand_gt 40 张，未参与本批训练）IoU .5 / conf .30 / imgsz 960', '',
             '| 模型 | hand GT/TP/FP | hand R | hand P | foot GT/TP/FP | foot R | foot P |',
             '|---|---|---|---|---|---|---|']
    for tag, agg in res.items():
        h = agg.get('hand', [0, 0, 0, 0])
        ft = agg.get('foot', [0, 0, 0, 0])
        lines.append('| %s | %d/%d/%d | %.1f%% | %.1f%% | %d/%d/%d | %.1f%% | %.1f%% |'
                     % (tag, h[0], h[1], h[2], 100.0 * h[1] / max(1, h[1] + h[3]),
                        100.0 * h[1] / max(1, h[1] + h[2]),
                        ft[0], ft[1], ft[2], 100.0 * ft[1] / max(1, ft[1] + ft[3]),
                        100.0 * ft[1] / max(1, ft[1] + ft[2])))
    print('\n'.join(lines))
    io.open(os.path.join(OUT, 'upright_eval.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
