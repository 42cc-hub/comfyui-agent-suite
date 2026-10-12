"""给队列的每一行补一个 `p_bad`（当前二元模型的预测），供标注器显示为**提示**。

只对队列里已有的行做预测（不重跑整池），并**保留已有人工标签** —— 覆盖标签是不可接受的。
用法：python scripts/add_pred_to_queue.py [--queue data/cls_queue_mined] [--weights ...]
"""
import argparse
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
CROPS = os.path.join(DATA, 'crops_todo')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(DATA, 'cls_queue_mined'))
    ap.add_argument('--weights', default=os.path.join(PROJ, 'runs', 'm3_cls_bin_v1', 'weights', 'best.pt'))
    a = ap.parse_args()
    p = os.path.join(a.queue, 'index.csv')
    rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
    fields = list(rows[0].keys())
    if 'p_bad' not in fields:
        fields.append('p_bad')
    from ultralytics import YOLO
    m = YOLO(a.weights)
    bad_i = [i for i, n in enumerate(m.names.values()) if n == 'bad']
    n = 0
    for r in rows:
        cp = os.path.join(CROPS, r['crop'])
        if not os.path.exists(cp):
            continue
        try:
            res = m.predict(cp, verbose=False)[0]
            probs = res.probs.data.tolist()
            r['p_bad'] = '%.3f' % max(probs[j] for j in bad_i)
            n += 1
        except Exception:
            r['p_bad'] = ''
    with io.open(p, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print('已为 %d/%d 行补上 p_bad（人工 label 原样保留）→ %s' % (n, len(rows), p))


if __name__ == '__main__':
    main()
