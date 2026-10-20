r"""M2a 手/脚框修复的验收：**不用 val mAP**，直接对着人工框量。

为什么不用 val mAP（关键）：`ds_det` 的 val 标签来自预标注（也就是"我们要修的那套框"），
一个学会人工口径的模型在这个 val 上**可能反而更低分** —— 拿它当判据会把修复判成退步。

三条判据（都在**人工标过的裁图**上量，属于"学会了没有"的检查；这批图在新模型的训练集里，
所以**不是泛化结论**，泛化只能靠交付图肉眼看）：
  1. 248 个人工正框：新旧模型各自预测，取同类里 IoU 最高的那个 → 报 median IoU 与 IoU≥0.5 命中率；
  2. 37 个"没有这个目标"的纯负样本：报 conf≥0.25 的误报数（误报率）；
  3. 交付稿（未标注）上：报新旧模型的框数与框尺寸中位（只作"行为有没有变"的参照），
     并把叠图存下来给人看。

用法：python scripts/eval_boxfix.py [--n-final 6]
"""
import argparse
import csv
import glob
import io
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PAIRS = [('m2a_det_996img_960', 'runs/m2a_det_996img_960/weights/best.pt', 'm2a_det_fix_v1', 'runs/m2a_det_fix_v1/weights/best.pt'),
         ('m2a_foot_v1', 'runs/m2a_foot_v1/weights/best.pt', 'm2a_foot_v2', 'runs/m2a_foot_v2/weights/best.pt'),
         ('m2a_hand_v2', 'runs/m2a_hand_v2/weights/best.pt', 'm2a_hand_v3', 'runs/m2a_hand_v3/weights/best.pt')]
DET_CLS = {'hand': 1, 'foot': 2}          # 4 类；单类模型用 0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-final', type=int, default=6)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--out-dir', default=os.path.join(DATA, 'boxfix_eval'))
    ap.add_argument('--queues', nargs='+', default=None,
                    help='在哪些队列上量（默认两个训练队列；量泛化时指 data/boxfix_holdout.csv）')
    ap.add_argument('--models', nargs='+', default=None,
                    help='name=path 列表（默认旧的三个 + 本轮新的）。4 类/单类按 model.names 自动判定')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from ultralytics import YOLO
    if a.models:
        MODELS = [tuple(m.split('=', 1)) for m in a.models]
    else:
        MODELS = [('old_det', 'runs/m2a_det_996img_960/weights/best.pt'),
                  ('v1_det', 'runs/m2a_det_fix_v1/weights/best.pt'),
                  ('v2_det', 'runs/m2a_det_fix_v2/weights/best.pt'),
                  ('old_foot', 'runs/m2a_foot_v1/weights/best.pt'),
                  ('v1_foot', 'runs/m2a_foot_v2/weights/best.pt'),
                  ('v2_foot', 'runs/m2a_foot_v3/weights/best.pt'),
                  ('old_hand', 'runs/m2a_hand_v2/weights/best.pt'),
                  ('v1_hand', 'runs/m2a_hand_v3/weights/best.pt'),
                  ('v2_hand', 'runs/m2a_hand_v4/weights/best.pt')]
    queues = a.queues or [os.path.join(DATA, 'boxfix', 'index.csv'), os.path.join(DATA, 'boxfix2', 'index.csv')]

    # ---- 人工框（裁图坐标）来自队列；裁图在 data/boxfix{,_2}/ds/images（逐条探测，队列文件可来自任何地方）----
    items = []
    img_roots = [os.path.join(DATA, 'boxfix', 'ds', 'images'),
                 os.path.join(DATA, 'boxfix2', 'ds', 'images')]
    for q in queues:
        for r in csv.DictReader(io.open(q, encoding='utf-8-sig')):
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
    pos = [it for it in items if it['verdict'] == 'draw']
    neg = [it for it in items if it['verdict'] == 'neg']
    print('判据来源 %s' % ', '.join(os.path.basename(q) for q in queues))
    print('人工数据：正 %d 条（%d 框） / 负 %d 条' %
          (len(pos), sum(len(it['boxes']) for it in pos), len(neg)))

    report = io.open(os.path.join(a.out_dir, 'report.txt'), 'w', encoding='utf-8', newline='\n')
    report.write('queues: %s\n' % ', '.join(queues))
    for tag, w in MODELS:
        if not os.path.exists(w):
            print('  跳过 %s（缺权重 %s）' % (tag, w))
            continue
        model = YOLO(w)
        is_det = len(model.names) >= 4          # 4 类=det（hand=1/foot=2）；单类（hand/foot）=0
        ious, hit, miss, per = [], 0, 0, {}
        for it in pos:
            cls = DET_CLS[it['kind']] if is_det else 0
            r = model.predict(it['img'], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            preds = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])
                     if int(b.cls[0]) == cls]
            for gt in it['boxes']:
                best = max([iou(gt, p) for p in preds], default=0.0)
                ious.append(best)
                per.setdefault(it['kind'], []).append(best)
                hit += best >= 0.5
                miss += best < 0.1
        if not ious:
            continue
        # 严格误报：只算"落在被人工否掉的那个旧框上"的预测（IoU>=0.3）。
        # 不能用"裁图里检测总数"当误报率：裁图外扩了 75~125%，里面可能本来就有别的同类目标。
        fps = 0
        for it in neg:
            cls = DET_CLS[it['kind']] if is_det else 0
            if not it['prop']:
                continue
            r = model.predict(it['img'], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            fps += sum(1 for b in (r.boxes or []) if int(b.cls[0]) == cls
                       and iou(it['prop'], [float(v) for v in b.xyxy[0].tolist()]) >= 0.3)
        line = ('%-10s median IoU %.3f｜IoU>=0.5 %.1f%%｜几乎没框住(<0.1) %.1f%%｜误报 %d/%d  %s'
                % (tag, float(np.median(ious)), 100.0 * hit / len(ious),
                   100.0 * miss / len(ious), fps, len(neg),
                   ' '.join('%s median %.3f(n=%d)' % (k, float(np.median(v)), len(v))
                            for k, v in sorted(per.items()))))
        print('  ' + line)
        report.write(line + '\n')

    # ---- 交付稿上的行为（未标注，只看有没有变 + 存叠图）----
    finals = sorted(glob.glob(r'<你的 ComfyUI 项目>\task\*\upload\*_2000.png'))[:a.n_final]
    det_models = [(t, w) for t, w in MODELS if os.path.exists(w) and 'det' in t]
    if finals and len(det_models) >= 2:
        loaded = [(t, YOLO(w)) for t, w in det_models]
        print('\n交付稿（%d 张，未标注）：手/脚框数与尺寸中位' % len(finals))
        for p in finals:
            line = '  %-40s' % os.path.basename(p)[:38]
            for tag, m in loaded:
                r = m.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                hs = [float(b.xyxy[0][2] - b.xyxy[0][0]) for b in (r.boxes or []) if int(b.cls[0]) == 1]
                fs = [float(v) for b in (r.boxes or []) if int(b.cls[0]) == 2
                      for v in (b.xyxy[0][2] - b.xyxy[0][0], b.xyxy[0][3] - b.xyxy[0][1])]
                line += '  %s: 手%d 脚%d 手中位宽%.0f 脚中位max%.0f' % (
                    tag, len(hs), len(fs), np.median(hs) if hs else 0, np.median(fs) if fs else 0)
            print(line)
            report.write(line + '\n')
    report.close()
    print('\n报告：%s' % os.path.join(a.out_dir, 'report.txt'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
