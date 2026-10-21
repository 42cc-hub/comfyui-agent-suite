# -*- coding: utf-8 -*-
"""手第二遍（`--hand-imgsz`）在两份 GT 上的 R/P：1280-only / 1536-only / 1280∪1536。

口径与 `scripts/locate.py` 的新语义一致：
  * 模型：`runs/m2a_det_fix_v2/weights/best.pt`（DEF_DET），4 类，只取 `hand` 类；
  * conf 0.30（产线默认），IoU 0.5 贪心一对一匹配算 R/P（与 `agent_work/team/eval_hand_gt.py` 同口径）；
  * `union` = 1280 那遍的 hand 框 + 1536 那遍的 hand 框，按 IoU 0.5 去重（保留置信高者）。
两份 GT：
  * `data/hand_gt`：40 张单人交付图，79 个手框；
  * `data/hand_gt_multi`：60 张多人图，手框取自 `index.csv` 的 hand 行。
"""
import argparse
import csv
import io
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
DEF_DET = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'hand2pass')


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_boxes(s):
    out = []
    for seg in (s or '').split(';'):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(x) for x in seg.split(',')]
        if len(v) == 4:
            out.append(v)
    return out


def load_gt(gt_dir):
    """→ {crop: [boxes]}，只取 hand 行；verdict=neg 视为 0 框（但图仍参与评测，用于算误报）。"""
    rows = list(csv.DictReader(io.open(os.path.join(gt_dir, 'index.csv'), encoding='utf-8-sig')))
    imgs, boxed = {}, 0
    for r in rows:
        if r['target'] != 'hand':
            continue
        bx = [] if r['verdict'] == 'neg' else parse_boxes(r['boxes'])
        imgs[r['crop']] = bx
        boxed += len(bx)
    return imgs, boxed


def match(gts, preds, thr=0.5):
    pairs = []
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            v = iou(g, p['box'])
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    ug, up, ious = set(), set(), []
    for v, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        ious.append(v)
    tp = len(ious)
    return tp, len(gts) - tp, len(preds) - tp, ious


def score(gt_map, pred_map, thr=0.5):
    tp = fn = fp = 0
    ious = []
    full = 0
    for crop, gts in gt_map.items():
        preds = pred_map.get(crop, [])
        a, b, c, io_ = match(gts, preds, thr)
        tp += a
        fn += b
        fp += c
        ious += io_
        if gts and b == 0 and c == 0:
            full += 1
    r = tp / (tp + fn) if tp + fn else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    return {'gt': tp + fn, 'pred': tp + fp, 'tp': tp, 'fn': fn, 'fp': fp,
            'recall50': round(100 * r, 1), 'precision50': round(100 * p, 1),
            'median_iou': round(statistics.median(ious), 3) if ious else 0.0,
            'img_full': full, 'n_img': len(gt_map)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=DEF_DET)
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(a.det)
    if getattr(model, 'task', None) == 'det':
        model.task = 'detect'

    sets = [('hand_gt', os.path.join(SUITE, 'data', 'hand_gt')),
            ('hand_gt_multi', os.path.join(SUITE, 'data', 'hand_gt_multi'))]
    report, lines = {}, []
    for tag, gdir in sets:
        gt_map, n_gt = load_gt(gdir)
        img_dir = os.path.join(gdir, 'images')
        preds = {k: {'1280': [], '1536': [], 'union': []} for k in gt_map}
        miss = 0
        for crop in sorted(gt_map):
            path = os.path.join(img_dir, crop)
            if not os.path.exists(path):
                miss += 1
                continue
            for size in (1280, 1536):
                r = model.predict(path, conf=a.conf, imgsz=size, verbose=False)[0]
                pb = [{'box': [round(v, 1) for v in b.xyxy[0].tolist()],
                       'prob': round(float(b.conf[0]), 3)} for b in (r.boxes or [])
                      if r.names[int(b.cls[0])] == 'hand']
                preds[crop][str(size)] = pb
                preds[crop]['union'].append(pb)
            u = list(preds[crop]['1280'])
            for b in preds[crop]['1536']:
                if any(iou(b['box'], x['box']) > 0.5 for x in u):
                    continue
                u.append(b)
            preds[crop]['union'] = u
        report[tag] = {'gt_boxes': n_gt, 'n_img': len(gt_map), 'crop_missing': miss,
                       'gt_total_from_rows': sum(len(v) for v in gt_map.values())}
        for key, label in (('1280', '1280-only'), ('1536', '1536-only'), ('union', '1280∪1536')):
            pm = {c: preds[c][key if key == 'union' else str(key)] for c in gt_map}
            s = score(gt_map, pm)
            report[tag][key] = s
            lines.append('%-14s %-10s R %5.1f  P %5.1f  TP %3d/%3d  FP %3d  medIoU %.3f  全中图 %d/%d' % (
                tag, label, s['recall50'], s['precision50'], s['tp'], s['gt'], s['fp'],
                s['median_iou'], s['img_full'], s['n_img']))
            print(lines[-1], flush=True)

    with io.open(os.path.join(a.out, 'hand2pass.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    with io.open(os.path.join(a.out, 'hand2pass.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print('wrote', os.path.join(a.out, 'hand2pass.json'))


if __name__ == '__main__':
    main()
