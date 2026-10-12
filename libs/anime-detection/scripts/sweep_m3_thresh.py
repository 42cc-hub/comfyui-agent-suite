"""M3 工作点（阈值）扫描：一次前向，离线扫阈值，同时看 val 与池外滥报。

产线现在的判定是"**任一**权重判为非 ok 即可疑"（`m3_weights` 并集），等价于每个模型各自 argmax。
但交付图上"已知缺陷 4/4"是靠收紧旗标凑出来的，而池外报出率在不同权重间差 3 倍（0.38 vs 0.12）
⇒ 工作点其实没被选过。本脚本把每个模型的 p(bad) 一次性算出来，再离线扫阈值 t：

  · val（按图切分、公共 val 集）：bad 召回 / ok 误报率
  · 池外（没标注过、过 size/edge 门限的裁图）：报出率
  · 并集（任一模型 ≥ t 即 bad）：同上两条 —— 这才是产线口径

用法：
  python scripts/sweep_m3_thresh.py                       # 用 locate.py 的生产并集权重
  python scripts/sweep_m3_thresh.py --weights runs/ab_res288_s20261010/weights/best.pt
  python scripts/sweep_m3_thresh.py --grid 0.2,0.3,0.4,0.5,0.6,0.7
输出：屏幕表格 + data/sweep_m3_thresh.csv
"""
import argparse
import csv
import glob
import io
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
CROPS = os.path.join(DATA, 'crops_todo')
DEF_M3 = [os.path.join(PROJ, 'runs', 'm3_cls_aug_v1', 'weights', 'best.pt'),
          os.path.join(PROJ, 'runs', 'm3_review_v1', 'weights', 'best.pt')]


def val_files(val_dir):
    out = []
    for lab in sorted(os.listdir(val_dir)):
        d = os.path.join(val_dir, lab)
        if os.path.isdir(d):
            out += [(p, lab == 'bad') for p in sorted(glob.glob(os.path.join(d, '*.png')))]
    return out


def pool_files(pool_n, min_side, min_edge):
    man = {}
    with io.open(os.path.join(CROPS, 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            man[r['crop']] = r
    done = set()
    for q in ('cls_queue', 'cls_queue_mined', 'cls_queue_next', 'cls_queue_all2'):
        p = os.path.join(DATA, q, 'index.csv')
        if os.path.exists(p):
            with io.open(p, encoding='utf-8-sig') as fh:
                for r in csv.DictReader(fh):
                    if (r.get('label') or '').strip():
                        done.add(r['crop'])
    out = []
    for c in sorted(man):
        if c in done or len(out) >= pool_n:
            continue
        p = os.path.join(CROPS, c)
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        if min(im.size) < min_side:
            continue
        g = np.asarray(im.convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32)
        gy, gx = np.gradient(g)
        if float(np.mean(np.hypot(gx, gy))) < min_edge:
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', default=DEF_M3)
    ap.add_argument('--val-dir', default=os.path.join(DATA, 'cls_val_common'))
    ap.add_argument('--pool-n', type=int, default=200)
    ap.add_argument('--min-side', type=int, default=64)
    ap.add_argument('--min-edge', type=float, default=4.0)
    ap.add_argument('--grid', default='0.2,0.3,0.4,0.5,0.6,0.7')
    ap.add_argument('--vote', action='store_true',
                    help='额外扫"k 票制"（n 个模型里至少 k 个 p(bad)≥t）—— 并集是 k=1，更严的 k 可能提精度')
    ap.add_argument('--out', default=os.path.join(DATA, 'sweep_m3_thresh.csv'))
    a = ap.parse_args()
    from ultralytics import YOLO

    grid = [float(x) for x in a.grid.split(',')]
    vs = val_files(a.val_dir)
    ps = pool_files(a.pool_n, a.min_side, a.min_edge)
    print('val %d（bad %d） / 池外 %d' % (len(vs), sum(1 for _, b in vs if b), len(ps)), flush=True)

    # 每个模型一次前向：val 与池外的 p(bad)
    pv, pp = [], []          # pv[model] = [(p_bad, is_bad_true)], pp[model] = [p_bad]
    for w in a.weights:
        m = YOLO(w)
        bi = [i for i, n in m.names.items() if n == 'bad']

        def pbad(path):
            return max(m.predict(path, verbose=False)[0].probs.data.tolist()[j] for j in bi)

        tag = os.path.basename(os.path.dirname(os.path.dirname(w)))
        pv.append([(pbad(p), b) for p, b in vs])
        pp.append([pbad(p) for p in ps])
        print('  %s 完成' % tag, flush=True)

    def stats(sel_v, sel_p, t):
        tp = sum(1 for v in sel_v if v[1] and v[0] >= t)
        nb = sum(1 for v in sel_v if v[1])
        fp = sum(1 for v in sel_v if (not v[1]) and v[0] >= t)
        nok = len(sel_v) - nb
        fire = sum(1 for x in sel_p if x >= t)
        return (round(tp / max(1, nb), 4), round(fp / max(1, nok), 4),
                round(fire / max(1, len(sel_p)), 4))

    un_v = [(max(pv[i][k][0] for i in range(len(a.weights))), pv[0][k][1]) for k in range(len(vs))]
    un_p = [max(pp[i][k] for i in range(len(a.weights))) for k in range(len(ps))]

    def vote_stats(sel_v, sel_p, t, k):
        """k 票制：数出 p(bad)≥t 的模型数，≥k 才判 bad（k=1 就是产线的并集口径）。
        `sel_v` 元素 = (各模型 p(bad) 列表, 真值 bad?)；`sel_p` 元素 = 各模型 p(bad) 列表。"""
        tp = sum(1 for v in sel_v if v[1] and count(v[0], t) >= k)
        nb = sum(1 for v in sel_v if v[1])
        fp = sum(1 for v in sel_v if (not v[1]) and count(v[0], t) >= k)
        nok = len(sel_v) - nb
        fire = sum(1 for x in sel_p if count(x, t) >= k)
        return (round(tp / max(1, nb), 4), round(fp / max(1, nok), 4), round(fire / max(1, len(sel_p)), 4))

    def count(pr, t):
        return sum(1 for x in pr if x >= t)

    # 每个样本的"各模型 p(bad) 列表"，供 k 票制使用
    vote_v = [([pv[i][k][0] for i in range(len(a.weights))], pv[0][k][1]) for k in range(len(vs))]
    vote_p = [[pp[i][k] for i in range(len(a.weights))] for k in range(len(ps))]

    rows = []
    print('\n%-7s %-26s %-26s %s' % ('t', '单模型' + ' / '.join('#' + str(i + 1) for i in range(len(a.weights))),
                                     '', '并集（产线口径）'))
    print('%-7s %-26s %-26s %s' % ('', 'recall  fp    池外', '', 'recall  fp    池外'))
    for t in grid:
        cells = []
        for i in range(len(a.weights)):
            r, f, pr = stats(pv[i], pp[i], t)
            cells.append('%5.3f  %5.3f  %5.3f' % (r, f, pr))
            rows.append(dict(mode='m%d' % (i + 1), weights=os.path.basename(os.path.dirname(os.path.dirname(a.weights[i]))),
                             thresh=t, bad_recall=r, ok_fp=f, pool_rate=pr))
        r, f, pr = stats(un_v, un_p, t)
        rows.append(dict(mode='union', weights='+'.join(os.path.basename(os.path.dirname(os.path.dirname(w))) for w in a.weights),
                         thresh=t, bad_recall=r, ok_fp=f, pool_rate=pr))
        print('%-7.2f %-26s %-26s %5.3f  %5.3f  %5.3f' % (t, cells[0],
                                                          ' '.join(cells[1:]), r, f, pr))
    if a.vote and len(a.weights) >= 2:
        print('\nk 票制（n=%d 个模型里至少 k 个 p(bad)≥t）' % len(a.weights))
        print('%-7s %-5s %-9s %-9s %s' % ('t', 'k', 'recall', 'fp', '池外'))
        for t in grid:
            for k in range(1, len(a.weights) + 1):
                r, f, pr = vote_stats(vote_v, vote_p, t, k)
                rows.append(dict(mode='vote_k%d' % k, weights='+'.join(
                    os.path.basename(os.path.dirname(os.path.dirname(w))) for w in a.weights),
                    thresh=t, bad_recall=r, ok_fp=f, pool_rate=pr))
                if k > 1 or t in grid[:1]:
                    print('%-7.2f %-5d %-9s %-9s %s' % (t, k, r, f, pr))
    with io.open(a.out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['mode', 'weights', 'thresh', 'bad_recall', 'ok_fp', 'pool_rate'])
        w.writeheader()
        w.writerows(rows)
    print('→ %s' % a.out)


if __name__ == '__main__':
    main()
