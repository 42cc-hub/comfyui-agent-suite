r"""M2a holdout 复核：在 `data/boxfix_holdout.csv` 上量 **per-class recall/precision** + 误报。

与 `eval_boxfix.py` 的关系：取队列、裁图根、conf/imgsz 的口径与它完全一致（同一套 holdout、
`conf=0.25`、`imgsz=960`、同一批裁图），所以它的 median IoU/命中率应当能在这里复现；
本脚本额外给出它没有的 **precision**（预测里有多少落在人工框上）和按类拆开的分子/分母。

口径说明（读数字前必须知道）：
  * recall@0.5 = 人工框里有多少个能被同类预测以 IoU>=0.5 命中（= eval_boxfix 的 `IoU>=0.5` 命中率）；
  * precision@0.5 = 预测框里有多少个与人工框 IoU>=0.5。**分母会包含裁图里人没标的同类目标**
    （裁图按原框外扩了 75~125%，相邻目标可能在图里），所以它是 precision 的**下界**；
  * 误报两个口径：strict = 落在"被人工否掉的那个预填框"（IoU>=0.3）上的预测数；
    loose = 负样本裁图里该类的**全部**预测数（会把"裁图里真有别的目标"也算进来，是上界）。
  * 单类手/脚模型只在**本类**行上评（拿手模型去评脚行永远是 0，没有信息量）；4 类模型两类都评。

用法：
  python scripts/m2a_eval_holdout.py --out-dir <目录> [--models name=path ...] [--conf 0.25] [--imgsz 960]
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')

DEFAULT_MODELS = [
    ('old_det', 'runs/m2a_det_996img_960/weights/best.pt'),
    ('v1_det', 'runs/m2a_det_fix_v1/weights/best.pt'),
    ('v2_det', 'runs/m2a_det_fix_v2/weights/best.pt'),
    ('old_hand', 'runs/m2a_hand_v2/weights/best.pt'),
    ('v1_hand', 'runs/m2a_hand_v3/weights/best.pt'),
    ('v2_hand', 'runs/m2a_hand_v4/weights/best.pt'),
]
DET_CLS = {'hand': 1, 'foot': 2}          # 4 类 det；单类模型用 0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_items(queue, img_roots):
    items = []
    for r in csv.DictReader(io.open(queue, encoding='utf-8-sig')):
        v = (r.get('verdict') or '').strip()
        if not v:
            continue
        ip = next((os.path.join(d, r['crop']) for d in img_roots
                   if os.path.exists(os.path.join(d, r['crop']))), None)
        if not ip:
            continue
        boxes = [[float(x) for x in b.split(',')] for b in (r.get('boxes') or '').split(';') if b.strip()]
        prop = [float(x) for x in (r.get('prop') or '').split(',')] if r.get('prop') else []
        items.append({'crop': r['crop'], 'img': ip, 'kind': r['target'],
                      'verdict': v, 'boxes': boxes, 'prop': prop})
    return items


def model_cls_map(model):
    """返回 (kind→类别号) 映射；单类模型只映射自己那一类。"""
    names = [str(v) for v in (model.names or {}).values()] \
        if isinstance(model.names, dict) else [str(v) for v in model.names]
    if len(names) >= 4:
        return dict(DET_CLS), 'det4'
    low = [n.lower() for n in names]
    if any('foot' in n for n in low):
        return {'foot': 0}, 'single_foot'
    return {'hand': 0}, 'single_hand'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(DATA, 'boxfix_holdout.csv'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--models', nargs='+', default=None, help='name=path 列表')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    img_roots = [os.path.join(DATA, 'boxfix', 'ds', 'images'),
                 os.path.join(DATA, 'boxfix2', 'ds', 'images')]
    items = load_items(a.queue, img_roots)
    pos = [it for it in items if it['verdict'] == 'draw']
    neg = [it for it in items if it['verdict'] == 'neg']
    kinds = {}
    for it in pos:
        kinds[it['kind']] = kinds.get(it['kind'], 0) + len(it['boxes'])
    negk = {}
    for it in neg:
        negk[it['kind']] = negk.get(it['kind'], 0) + 1

    from ultralytics import YOLO
    MODELS = [tuple(m.split('=', 1)) for m in a.models] if a.models else DEFAULT_MODELS

    out_txt = io.open(os.path.join(a.out_dir, 'm2a_eval_holdout.txt'), 'w', encoding='utf-8', newline='\n')
    rows_csv = []
    head = ('队列 %s｜正样本 %d 行 / 人工框 %s｜负样本 %d 行 %s｜conf=%.2f imgsz=%d'
            % (os.path.relpath(a.queue, PROJ), len(pos), kinds, len(neg), negk, a.conf, a.imgsz))
    print(head)
    out_txt.write(head + '\n')

    for tag, w in MODELS:
        if not os.path.exists(w):
            print('  跳过 %s（缺权重 %s）' % (tag, w))
            out_txt.write('跳过 %s（缺权重 %s）\n' % (tag, w))
            continue
        model = YOLO(w)
        cmap, kind_model = model_cls_map(model)
        per = {}          # kind -> dict
        for it in pos:
            if it['kind'] not in cmap:
                continue                      # 单类模型不评不适用类别
            cls = cmap[it['kind']]
            r = model.predict(it['img'], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            preds = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])
                     if int(b.cls[0]) == cls]
            d = per.setdefault(it['kind'], {'gt': 0, 'hit': 0, 'miss': 0, 'ious': [],
                                            'pred': 0, 'pred_hit': 0})
            for gt in it['boxes']:
                best = max([iou(gt, p) for p in preds], default=0.0)
                d['ious'].append(best)
                d['gt'] += 1
                d['hit'] += best >= 0.5
                d['miss'] += best < 0.1
            for p in preds:
                d['pred'] += 1
                d['pred_hit'] += max([iou(gt, p) for gt in it['boxes']], default=0.0) >= 0.5
        # 负样本：strict 只算落在被否掉的预填框上；loose 算该类全部预测
        fps = {}
        for it in neg:
            if it['kind'] not in cmap:
                continue
            cls = cmap[it['kind']]
            r = model.predict(it['img'], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            preds = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])
                     if int(b.cls[0]) == cls]
            d = fps.setdefault(it['kind'], {'neg': 0, 'strict': 0, 'loose': 0})
            d['neg'] += 1
            d['loose'] += len(preds)
            if it['prop']:
                d['strict'] += sum(1 for p in preds if iou(it['prop'], p) >= 0.3)

        g_gt = sum(v['gt'] for v in per.values())
        g_hit = sum(v['hit'] for v in per.values())
        g_miss = sum(v['miss'] for v in per.values())
        g_ious = [x for v in per.values() for x in v['ious']]
        g_pred = sum(v['pred'] for v in per.values())
        g_predhit = sum(v['pred_hit'] for v in per.values())
        neg_all = sum(v['neg'] for v in fps.values())
        fps_strict = sum(v['strict'] for v in fps.values())
        fps_loose = sum(v['loose'] for v in fps.values())

        lines = ['%-9s [%s] 整体: GT %d 框｜recall@0.5 %.1f%%｜median IoU %.3f｜几乎没框住(<0.1) %.1f%%'
                 '｜precision@0.5 %.1f%%(%d/%d 预测)｜误报 strict %d/%d loose %d'
                 % (tag, kind_model, g_gt, 100.0 * g_hit / max(1, g_gt), float(np.median(g_ious)) if g_ious else 0.0,
                    100.0 * g_miss / max(1, g_gt), 100.0 * g_predhit / max(1, g_pred), g_predhit, g_pred,
                    fps_strict, neg_all, fps_loose)]
        for k in sorted(per):
            v = per[k]
            f = fps.get(k, {'neg': 0, 'strict': 0, 'loose': 0})
            lines.append('    %-4s GT %2d 框｜recall@0.5 %.1f%%｜median IoU %.3f｜miss %.1f%%'
                         '｜precision@0.5 %.1f%%(%d/%d)｜误报 strict %d/%d loose %d'
                         % (k, v['gt'], 100.0 * v['hit'] / max(1, v['gt']),
                            float(np.median(v['ious'])), 100.0 * v['miss'] / max(1, v['gt']),
                            100.0 * v['pred_hit'] / max(1, v['pred']), v['pred_hit'], v['pred'],
                            f['strict'], f['neg'], f['loose']))
            rows_csv.append({'model': tag, 'kind_model': kind_model, 'cls': k,
                             'n_gt': v['gt'], 'recall50': round(100.0 * v['hit'] / max(1, v['gt']), 1),
                             'median_iou': round(float(np.median(v['ious'])), 3),
                             'miss_lt0.1': round(100.0 * v['miss'] / max(1, v['gt']), 1),
                             'n_pred': v['pred'],
                             'precision50': round(100.0 * v['pred_hit'] / max(1, v['pred']), 1),
                             'neg_n': f['neg'], 'fp_strict': f['strict'], 'fp_loose': f['loose']})
        for ln in lines:
            print(ln)
            out_txt.write(ln + '\n')
        out_txt.flush()

    fields = ['model', 'kind_model', 'cls', 'n_gt', 'recall50', 'median_iou', 'miss_lt0.1',
              'n_pred', 'precision50', 'neg_n', 'fp_strict', 'fp_loose']
    with io.open(os.path.join(a.out_dir, 'm2a_eval_holdout.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)
    out_txt.close()
    print('\n输出：%s' % os.path.join(a.out_dir, 'm2a_eval_holdout.txt'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
