# -*- coding: utf-8 -*-
"""阶段 B 候选权重 vs fix_v2：产线几何（conf .30 / imgsz 1280）配对对比。

用 locate.py 单模型路径（不带专用脚/第二遍手，避免合并逻辑干扰）在多人真值集上跑两遍，
再按 person/hand/foot 三类分别算 R/P（IoU .5 贪心一对一）。

用法：
  python -X utf8 scripts/m2a_person_cand_eval.py --run        # 跑两套权重 + 打分
  python -X utf8 scripts/m2a_person_cand_eval.py --score      # 只重算分
"""
import csv
import io
import json
import os
import subprocess
import sys

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REP = os.path.abspath(os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier'))
OUT = os.path.join(REP, 'person_gt2')
PY = sys.executable
W_FIX = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
W_CAND = os.path.join(SUITE, 'runs', 'm2a_det_person_v1', 'weights', 'best.pt')
W_CAND_LAST = os.path.join(SUITE, 'runs', 'm2a_det_person_v1', 'weights', 'last.pt')
GT = os.path.join(SUITE, 'data', 'hand_gt_multi', 'index.csv')
CONF, IMGSZ = 0.30, 1280
CLS = ('person', 'hand', 'foot')


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load_gt():
    gt = {}
    for r in csv.DictReader(io.open(GT, encoding='utf-8-sig')):
        t = r['target']
        if t not in CLS:
            continue
        boxes = []
        if r['boxes'].strip() and r['verdict'] != 'neg':
            boxes = [tuple(float(v) for v in s.split(',')) for s in r['boxes'].split(';') if s.strip()]
        gt.setdefault(os.path.splitext(r['crop'])[0], {})[t] = boxes
    return gt


def run(weights, tag):
    src = os.path.join(SUITE, 'data', 'hand_gt_multi', 'images')
    dst = os.path.join(OUT, 'preds_prod_%s' % tag)
    os.makedirs(dst, exist_ok=True)
    cmd = [PY, '-X', 'utf8', os.path.join(SUITE, 'scripts', 'locate.py'), '--src', src,
           '--out', dst, '--det-weights', weights, '--conf', str(CONF), '--imgsz', str(IMGSZ), '--no-vitpose']
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print('run %s rc=%d' % (tag, p.returncode))
    if p.returncode != 0:
        print((p.stderr or '')[-2000:])


def load_pred(tag):
    d = os.path.join(OUT, 'preds_prod_%s' % tag)
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        boxes = rec.get('boxes_all', rec.get('boxes', []))
        per = {}
        for cls in CLS:
            per[cls] = [b['box'] for b in boxes if b.get('cls') == cls]
        out[os.path.splitext(f)[0]] = per
    return out


def score(gt, pred):
    res = {}
    for cls in CLS:
        tp = fp = ngt = 0
        ious = []
        for stem, per in gt.items():
            g = per.get(cls, [])
            p = pred.get(stem, {}).get(cls, [])
            ngt += len(g)
            cand = sorted(((iou(gb, pb), gi, pi) for gi, gb in enumerate(g) for pi, pb in enumerate(p)), reverse=True)
            ug, up = set(), set()
            for v, gi, pi in cand:
                if v < 0.5 or gi in ug or pi in up:
                    continue
                ug.add(gi)
                up.add(pi)
                ious.append(v)
            tp += len(ug)
            fp += len(p) - len(up)
        ious.sort()
        res[cls] = dict(gt=ngt, tp=tp, fp=fp, fn=ngt - tp,
                        recall=round(100.0 * tp / ngt, 1) if ngt else None,
                        precision=round(100.0 * tp / (tp + fp), 1) if tp + fp else None,
                        median_iou=round(ious[len(ious) // 2], 3) if ious else None)
    return res


def main():
    gt = load_gt()
    tags = [('fix_v2', W_FIX)]
    if os.path.exists(W_CAND):
        tags.append(('cand_best', W_CAND))
    if os.path.exists(W_CAND_LAST):
        tags.append(('cand_last', W_CAND_LAST))
    if '--run' in sys.argv:
        for tag, w in tags:
            run(w, tag)
    lines = ['# 产线几何（locate.py 单模型 conf .30 / imgsz 1280）多人真值集配对对比',
             'GT：%d 张图 / person %d / hand %d / foot %d' % (
                 len(gt), sum(len(v.get('person', [])) for v in gt.values()),
                 sum(len(v.get('hand', [])) for v in gt.values()),
                 sum(len(v.get('foot', [])) for v in gt.values())), '']
    table = {}
    for tag, _ in tags:
        d = os.path.join(OUT, 'preds_prod_%s' % tag)
        if not os.path.isdir(d):
            continue
        s = score(gt, load_pred(tag))
        table[tag] = s
        for cls in CLS:
            v = s[cls]
            lines.append('%-11s %-6s GT %4d TP %4d FP %4d FP/图 %.2f R %s P %s medIoU %s' % (
                tag, cls, v['gt'], v['tp'], v['fp'], v['fp'] / max(1, len(gt)),
                v['recall'], v['precision'], v['median_iou']))
        lines.append('')
    io.open(os.path.join(OUT, 'cand_prod_cmp.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    json.dump(table, io.open(os.path.join(OUT, 'cand_prod_cmp.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
