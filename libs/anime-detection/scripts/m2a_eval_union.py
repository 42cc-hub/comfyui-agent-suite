r"""M2a 接线前的**并集实测**：复刻 `locate.py` 的合并规则（4 类 det + 专用单类取并集、IoU>0.5 去重），
在 `data/boxfix_holdout.csv` 上比较"现默认 / 换成 fix_v2 / fix_v2 + hand_v3"三套配置。

为什么必须量并集：`locate.py` 的产出不是单模型结果。现状是 `DEF_DET=old` + `DEF_FOOT=m2a_foot_v1`
（脚取并集、手不取）；提议是 `DEF_DET=v2` + `DEF_FOOT` 不变 + `--hand-weights` 默认给 `m2a_hand_v3`。
单模型分数相加不能预测并集分数（去重、重复框、专用模型的高召回低精度都会改变结果）。

口径：
  * 两个协议 —— `--conf 0.25 / imgsz 960`（= eval_boxfix 的 holdout 口径，与已有数字可比）
    与 `--conf 0.3 / imgsz 1280`（= locate.py / check_parts.py 的产线默认）；
  * 专用单类模型按 `locate.py` 的做法用 `conf = max(0.15, conf-0.15)`、`imgsz` 同一个；
  * recall@0.5 = 人工框被同类并集预测以 IoU>=0.5 命中；precision@0.5 = 预测里 IoU>=0.5 的占比
    （裁图外扩 75~125%，裁图里没标到的同类目标会被记成 FP ⇒ precision 是下界）；
  * 误报 strict = 落在被人工否掉的预填框（IoU>=0.3）上的预测数（分母 9 条负样本）。

用法：python scripts/m2a_eval_union.py --out-dir <目录>
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


CONFIGS = [
    ('A_now', dict(det='runs/m2a_det_996img_960/weights/best.pt',
                   foot='runs/m2a_foot_v1/weights/best.pt', hand=None)),
    ('B_newdet', dict(det='runs/m2a_det_fix_v2/weights/best.pt',
                      foot='runs/m2a_foot_v1/weights/best.pt', hand=None)),
    ('C_newdet_hand', dict(det='runs/m2a_det_fix_v2/weights/best.pt',
                           foot='runs/m2a_foot_v1/weights/best.pt',
                           hand='runs/m2a_hand_v3/weights/best.pt')),
    ('D_olddet_hand', dict(det='runs/m2a_det_996img_960/weights/best.pt',
                           foot='runs/m2a_foot_v1/weights/best.pt',
                           hand='runs/m2a_hand_v3/weights/best.pt')),
]
PROTOCOLS = [(0.25, 960, 'holdout 口径 conf0.25/imgsz960'),
             (0.30, 1280, '产线口径 conf0.3/imgsz1280')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(DATA, 'boxfix_holdout.csv'))
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    img_roots = [os.path.join(DATA, 'boxfix', 'ds', 'images'),
                 os.path.join(DATA, 'boxfix2', 'ds', 'images')]
    items = load_items(a.queue, img_roots)
    pos = [it for it in items if it['verdict'] == 'draw']
    neg = [it for it in items if it['verdict'] == 'neg']

    from ultralytics import YOLO
    cache = {}

    def get(w):
        if not w:
            return None
        if w not in cache:
            cache[w] = YOLO(os.path.join(PROJ, w))
        return cache[w]

    out = io.open(os.path.join(a.out_dir, 'm2a_eval_union.txt'), 'w', encoding='utf-8', newline='\n')

    def say(s):
        print(s)
        out.write(s + '\n')

    say('队列 %s｜正 %d 行/48 框｜负 %d 行（strict 误报分母）' % (os.path.relpath(a.queue, PROJ), len(pos), len(neg)))
    for conf, imgsz, plabel in PROTOCOLS:
        say('\n=== 协议 %s' % plabel)
        for name, cfg in CONFIGS:
            det, foot, hand = get(cfg['det']), get(cfg['foot']), get(cfg['hand'])
            per = {'hand': {'gt': 0, 'hit': 0, 'miss': 0, 'ious': [], 'pred': 0, 'predhit': 0},
                   'foot': {'gt': 0, 'hit': 0, 'miss': 0, 'ious': [], 'pred': 0, 'predhit': 0}}
            fps = {'hand': 0, 'foot': 0}
            for it in pos:
                kind = it['kind']
                cls = {'hand': 1, 'foot': 2}[kind]
                r = det.predict(it['img'], conf=conf, imgsz=imgsz, verbose=False)[0]
                preds = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])
                         if int(b.cls[0]) == cls]
                extra = foot if kind == 'foot' else hand
                if extra is not None:
                    rc = max(0.15, conf - 0.15)
                    re_ = extra.predict(it['img'], conf=rc, imgsz=imgsz, verbose=False)[0]
                    for b in (re_.boxes or []):
                        nb = [float(v) for v in b.xyxy[0].tolist()]
                        if int(b.cls[0]) != 0:
                            continue
                        if any(iou(nb, p) > 0.5 for p in preds):
                            continue
                        preds.append(nb)
                d = per[kind]
                for gt in it['boxes']:
                    best = max([iou(gt, p) for p in preds], default=0.0)
                    d['ious'].append(best)
                    d['gt'] += 1
                    d['hit'] += best >= 0.5
                    d['miss'] += best < 0.1
                for p in preds:
                    d['pred'] += 1
                    d['predhit'] += max([iou(gt, p) for gt in it['boxes']], default=0.0) >= 0.5
            for it in neg:
                kind = it['kind']
                if not it['prop']:
                    continue
                cls = {'hand': 1, 'foot': 2}[kind]
                r = det.predict(it['img'], conf=conf, imgsz=imgsz, verbose=False)[0]
                preds = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])
                         if int(b.cls[0]) == cls]
                extra = foot if kind == 'foot' else hand
                if extra is not None:
                    re_ = extra.predict(it['img'], conf=max(0.15, conf - 0.15), imgsz=imgsz, verbose=False)[0]
                    for b in (re_.boxes or []):
                        if int(b.cls[0]) == 0:
                            preds.append([float(v) for v in b.xyxy[0].tolist()])
                fps[kind] += sum(1 for p in preds if iou(it['prop'], p) >= 0.3)
            ggt = sum(v['gt'] for v in per.values())
            ghit = sum(v['hit'] for v in per.values())
            gpred = sum(v['pred'] for v in per.values())
            gpredhit = sum(v['predhit'] for v in per.values())
            s = ('%-14s GT %d｜recall@0.5 %.1f%%｜precision@0.5 %.1f%%(%d/%d 预测)｜误报 strict %d/9'
                 % (name, ggt, 100.0 * ghit / max(1, ggt), 100.0 * gpredhit / max(1, gpred),
                    gpredhit, gpred, fps['hand'] + fps['foot']))
            say(s)
            for k in ('hand', 'foot'):
                v = per[k]
                say('    %-4s GT %2d｜recall@0.5 %5.1f%%｜median IoU %.3f｜precision@0.5 %5.1f%%(%d/%d)'
                    '｜误报 strict %d'
                    % (k, v['gt'], 100.0 * v['hit'] / max(1, v['gt']),
                       float(np.median(v['ious'])) if v['ious'] else 0.0,
                       100.0 * v['predhit'] / max(1, v['pred']), v['predhit'], v['pred'], fps[k]))
    out.close()
    print('\n输出：%s' % os.path.join(a.out_dir, 'm2a_eval_union.txt'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
