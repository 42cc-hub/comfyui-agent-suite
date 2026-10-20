"""按**产线口径**重切已标注的手/脚裁图（R39 的消融准备）。

背景（R39/R40）：M3 的训练裁图是 `prelabel_batch` 用"姿态脚点包围盒 + 0.35 外扩"（脚）、
"imgutils 手框 + 0.10 外扩"（手）做的；而产线 `locate.py` 用"检测器框 + 0.10 外扩"。
实测同一张图同一只脚，训练口径 p(bad)=1.00、产线口径 0.00 ⇒ 训练/推断口径不一致。

本脚本不猜几何：对每张有标注的裁图，**用它自己的源图跑一遍产线检测器**，取与该裁图框 IoU 最大的
产线框，按产线外扩（0.10）重切，文件名保持不变 ⇒ `index.csv` 里的人工标签直接沿用。
IoU 太低（< --min-iou）说明产线在那张图上根本没检出这个部位 ⇒ 该裁图**排除**并记进报告
（这类样本本身就是"M2a 漏检"，不该混进 M3 的对比）。

用法：
  python scripts/recut_crops_prod.py --out data/crops_prod [--min-iou 0.3] [--limit 0]
报告：<out>/recut_report.csv（crop/kind/label/iou/kept）
"""
import argparse
import collections
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
CROPS = os.path.join(DATA, 'crops_todo')
PRE = os.path.join(DATA, 'prelabel')


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, ua)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA, 'crops_prod'))
    ap.add_argument('--min-iou', type=float, default=0.3)
    ap.add_argument('--det', default=os.path.join(PROJ, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'))
    ap.add_argument('--foot', default=os.path.join(PROJ, 'runs', 'm2a_foot_v1', 'weights', 'best.pt'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--imgsz', type=int, default=1280)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    from ultralytics import YOLO

    # 有标注的裁图（队列里 label 非空）
    rows = []
    seen = {}
    for q in ('cls_queue_all2', 'cls_queue', 'cls_queue_mined', 'cls_queue_next'):
        p = os.path.join(DATA, q, 'index.csv')
        if not os.path.exists(p):
            continue
        with io.open(p, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if (r.get('label') or '').strip() and r['crop'] not in seen:
                    seen[r['crop']] = r
    man = {}
    with io.open(os.path.join(CROPS, 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            man[r['crop']] = r
    rows = [r for c, r in seen.items() if c in man]
    if a.limit:
        rows = rows[:a.limit]
    print('待重切 %d 张（源图 %d 张）' % (len(rows), len({man[r['crop']]['source'] for r in rows})), flush=True)

    det = YOLO(a.det)
    foot = YOLO(a.foot) if os.path.exists(a.foot) else None
    srcs = {}
    rep = []
    cache = {}
    for i, r in enumerate(rows, 1):
        m = man[r['crop']]
        src = m['source']
        if src not in cache:
            # 原图路径在预标注 JSON 的 `src` 字段里（prelabel/ 下只有裁图与 JSON，没有原图）
            p = None
            jp = os.path.join(PRE, src + '.json')
            if os.path.exists(jp):
                try:
                    import json
                    cand = json.load(io.open(jp, encoding='utf-8')).get('src')
                    if cand and os.path.exists(cand):
                        p = cand
                except Exception:
                    p = None
            if p is None:
                cache[src] = None
            else:
                img = Image.open(p).convert('RGB')
                boxes = []
                try:
                    for b in det.predict(p, conf=0.25, imgsz=a.imgsz, verbose=False)[0].boxes:
                        cls = det.names[int(b.cls[0])]
                        if cls in ('hand', 'foot'):
                            boxes.append([float(v) for v in b.xyxy[0]])
                except Exception as e:
                    print('  检测失败 %s: %s' % (src, e))
                if foot is not None:
                    try:
                        for b in foot.predict(p, conf=0.2, imgsz=a.imgsz, verbose=False)[0].boxes:
                            boxes.append([float(v) for v in b.xyxy[0]])
                    except Exception:
                        pass
                cache[src] = (img, boxes)
        got = cache[src]
        if got is None:
            rep.append(dict(r, iou='', kept=0, why='源图缺失'))
            continue
        img, boxes = got
        want = [int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])]
        best, best_iou = None, 0.0
        for b in boxes:
            v = iou(want, b)
            if v > best_iou:
                best, best_iou = b, v
        if best is None or best_iou < a.min_iou:
            rep.append(dict(r, iou=round(best_iou, 3), kept=0, why='产线框 IoU<%.2f' % a.min_iou))
            continue
        x0, y0, x1, y1 = best
        pad = int(0.10 * max(x1 - x0, y1 - y0))
        c = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
             min(img.width, int(x1 + pad)), min(img.height, int(y1 + pad)))
        if min(c[2] - c[0], c[3] - c[1]) < 32:
            rep.append(dict(r, iou=round(best_iou, 3), kept=0, why='重切后过小'))
            continue
        img.crop(c).save(os.path.join(a.out, r['crop']))
        rep.append(dict(r, iou=round(best_iou, 3), kept=1, why=''))
        if i % 50 == 0:
            print('  ...%d/%d' % (i, len(rows)), flush=True)

    with io.open(os.path.join(a.out, 'recut_report.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop', 'kind', 'label', 'iou', 'kept', 'why'],
                           extrasaction='ignore')
        w.writeheader()
        for r in rep:
            w.writerow(r)
    kept = collections.Counter(r['label'] for r in rep if r['kept'] == 1)
    drop = collections.Counter(r['label'] for r in rep if r['kept'] == 0)
    print('\n重切保留 %d 张 / 排除 %d 张 → %s' % (sum(kept.values()), sum(drop.values()), a.out))
    print('  保留分布：%s' % dict(kept))
    print('  排除分布：%s' % dict(drop))


if __name__ == '__main__':
    main()
