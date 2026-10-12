"""把多组四类权重的 **val 指标 + 交付图验收 + 倒置头探针** 并成一张表（一次命令看全）。

为什么要它：`eval_face4.py` 一次只跑一个权重，val 指标又得另翻 `results.csv`，
选型时要在"val（会骗人）"和"交付图（用户的眼睛）"之间来回切 —— 手工拼过几轮，容易抄错。
这里把三件事一次算完并打印 markdown 表：

  val        = `results.csv` 里 **mAP50-95 最高那个 epoch**（best，不是最后一轮）
  交付图      = 8 张交付图（真值：02~07 有 gag、全无眼罩）→ eye 总数 / gag 检出 / blindfold 误报
  倒置头探针  = `.runtime/invert10.txt` 那 10 张**脸上下颠倒**的 head（旋转增强出不了 180°，专门盯这一格）

用法：
  python scripts/ab_face4.py                                  # 所有 m2b_face4_* 运行
  python scripts/ab_face4.py --runs m2b_face4_v1 m2b_face4_v2_rot30_s20261013
  python scripts/ab_face4.py --newest 4 --skip-delivery        # 只看 val（快）
"""
import argparse
import csv
import glob
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_face4 as E                                     # noqa: E402  （同一份交付图实现）

INVERT = os.path.join(E.PROJ, '.runtime', 'invert10.txt')
CLS = ['eye', 'mouth', 'gag', 'blindfold']


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def val_pr(weights, conf=0.4):
    """val（与 train 无来源重叠）上**按类**算 recall / precision @conf、IoU 0.5。

    为什么要它：交付图只有 6 张有口塞 ⇒ "gag 6/6 vs 4/6" 的差在噪声内，选不了型。
    val 有 ~96 个 gag 实例，统计力高一个量级，且同样无来源重叠。返回 {类: (R, P, 人标数, 预测数)}。
    """
    import collections
    import glob
    from ultralytics import YOLO
    V = os.path.join(E.PROJ, 'data', 'ds_face4')
    m = YOLO(weights)
    tp, ng, npd = collections.Counter(), collections.Counter(), collections.Counter()
    dup = collections.Counter()          # 一张 head 上同类预测 ≥2 个框的次数（重复框）
    for f in sorted(glob.glob(os.path.join(V, 'labels', 'val', '*.txt'))):
        img = os.path.join(V, 'images', 'val', os.path.basename(f).replace('.txt', '.png'))
        if not os.path.exists(img):
            continue
        gt = collections.defaultdict(list)
        for ln in io.open(f, encoding='utf-8'):
            q = ln.split()
            if len(q) != 5:
                continue
            c = int(q[0])
            cx, cy, w, h = [float(v) for v in q[1:]]
            gt[CLS[c]].append([(cx - w / 2) * 256, (cy - h / 2) * 256,
                               (cx + w / 2) * 256, (cy + h / 2) * 256])
            ng[CLS[c]] += 1
        r = m.predict(img, conf=conf, verbose=False)[0]
        pr = collections.defaultdict(list)
        for b in (r.boxes or []):
            pr[m.names[int(b.cls[0])]].append([float(v) for v in b.xyxy[0].tolist()])
        for c in CLS:
            used = [False] * len(pr[c])
            npd[c] += len(pr[c])
            if len(pr[c]) >= 2:
                dup[c] += 1
            for g in gt[c]:
                for i, x in enumerate(pr[c]):
                    if not used[i] and _iou(g, x) >= 0.5:
                        used[i] = True
                        tp[c] += 1
                        break
    n_img = len(glob.glob(os.path.join(V, 'labels', 'val', '*.txt')))
    return {c: (tp[c] / max(1, ng[c]), tp[c] / max(1, npd[c]), ng[c], npd[c],
                dup[c] / max(1, n_img)) for c in CLS}


def val_metrics(run):
    p = os.path.join(E.PROJ, 'runs', run, 'results.csv')
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(io.open(p)))
    if not rows:
        return None
    b = max(rows, key=lambda x: float(x['metrics/mAP50-95(B)']))
    g = lambda n: float(b['metrics/%s(B)' % n])            # noqa: E731
    return b['epoch'], g('precision'), g('recall'), g('mAP50'), g('mAP50-95')


def inverted_probe(weights, conf=0.25):
    """10 张倒置脸：返回 (两只眼都出的张数, 出嘴的张数)。

    口径按**张**不按框数：低 conf 下偶发多出的一两个框是**不同位置的低分误报**（实测 conf 0.27~0.28、
    两两 IoU=0），不是同一个眼的重复框 ⇒ 数框会把这种噪声读成"更好"。
    """
    from ultralytics import YOLO
    if not os.path.exists(INVERT):
        return None
    m = YOLO(weights)
    n_two, n_mouth = 0, 0
    for ln in io.open(INVERT, encoding='utf-8'):
        f = ln.strip()
        if not f:
            continue
        img = os.path.join(E.PROJ, 'data', 'prelabel', os.path.basename(f).replace('.txt', '.png'))
        if not os.path.exists(img):
            continue
        r = m.predict(img, conf=conf, verbose=False)[0]
        got = {}
        for b in (r.boxes or []):
            got[m.names[int(b.cls[0])]] = got.get(m.names[int(b.cls[0])], 0) + 1
        n_two += 1 if got.get('eye', 0) >= 2 else 0
        n_mouth += 1 if got.get('mouth') else 0
    return n_two, n_mouth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=None)
    ap.add_argument('--newest', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--skip-delivery', action='store_true')
    a = ap.parse_args()
    runs = a.runs
    if not runs:
        runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(E.PROJ, 'runs', 'm2b_face4_*'))
                      if os.path.isdir(d))
    if a.newest:
        runs = sorted(runs, key=lambda r: os.path.getmtime(os.path.join(E.PROJ, 'runs', r)))[-a.newest:]

    hdr = ['run', 'best ep', 'P', 'R', 'mAP50', 'mAP50-95', 'val R eye/mouth/gag/bf',
           'val P eye/mouth/gag/bf', 'val 重复框率 gag']
    if not a.skip_delivery:
        hdr += ['交付 eye/16', 'gag/6', 'bf 误报', '出嘴图', '倒置 双眼张/10', '倒置 出嘴张/10']
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    for r in runs:
        w = os.path.join(E.PROJ, 'runs', r, 'weights', 'best.pt')
        m = val_metrics(r)
        if m is None or not os.path.exists(w):
            print('| %s | （缺 results.csv 或 best.pt） |' % r + ' |' * (len(hdr) - 2))
            continue
        cells = [r, str(m[0]), '%.3f' % m[1], '%.3f' % m[2], '%.3f' % m[3], '**%.3f**' % m[4]]
        vp = val_pr(w, a.conf)
        cells.append('/'.join('%.2f' % vp[c][0] for c in CLS))
        cells.append('/'.join('%.2f' % vp[c][1] for c in CLS))
        cells.append('%.2f' % vp['gag'][4])
        if not a.skip_delivery:
            d = E.run_eval(w, conf=a.conf, quiet=True)
            inv = inverted_probe(w)
            cells += ['**%d**' % d['eye'], '%d/6' % len(d['gag_hit']), str(len(d['bf_fp'])),
                      str(d['mouth_imgs']),
                      ('%d' % inv[0]) if inv else '—', ('%d' % inv[1]) if inv else '—']
        print('| ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
