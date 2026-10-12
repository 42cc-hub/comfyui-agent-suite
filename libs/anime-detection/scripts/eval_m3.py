"""给一个 M3 权重算**同一套指标**（消融 harness 与人工报告都用它，避免内联代码两套口径）。

输出 JSON：
  val_top1 / bad_recall / n_val / n_bad   —— 在**按图切分**的 val 上（无泄漏）
  pool_n / pool_fire / pool_rate          —— 池外（没标过、且过 conf/size/edge 门限的裁图）里判 bad 的比例
用法：
  python scripts/eval_m3.py --weights runs/<name>/weights/best.pt [--val-dir data/cls_bin_v8/val] [--pool-n 200]
"""
import argparse
import collections
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val-dir', default=os.path.join(DATA, 'cls_bin_v8', 'val'))
    ap.add_argument('--pool-n', type=int, default=200)
    ap.add_argument('--min-side', type=int, default=64)
    ap.add_argument('--min-edge', type=float, default=4.0)
    ap.add_argument('--tta', choices=['none', 'hflip'], default='none',
                    help='测试时增强：hflip = 原图与水平翻转各判一次取概率均值（推理 2×，不改权重）')
    ap.add_argument('--thresh', type=float, default=None,
                    help='按 p(bad) ≥ 该阈值判 bad（默认 None = argmax，等价于 0.5 两分类）')
    a = ap.parse_args()
    from ultralytics import YOLO

    m = YOLO(a.weights)
    # 二元模型用类名 `bad`；三类/多类模型（类别口径消融）没有 `bad` ⇒ 把**除 ok 以外**的类都算 bad
    bad_idx = [i for i, n in m.names.items() if n == 'bad']
    if not bad_idx:
        bad_idx = [i for i, n in m.names.items() if n != 'ok']
        if bad_idx:
            print('  （多类模型：把 %s 合并视为 bad）' % [m.names[i] for i in bad_idx])
    out = {'names': list(m.names.values()), 'tta': a.tta, 'thresh': a.thresh}

    def probs_of(path_or_img):
        """返回类别概率列表；`--tta hflip` 时对水平翻转取均值（同一张图两次前向）。"""
        arr = np.asarray(Image.open(path_or_img).convert('RGB')) if isinstance(path_or_img, str) else path_or_img
        p = m.predict(np.ascontiguousarray(arr[:, :, ::-1]), verbose=False)[0].probs.data.tolist()
        if a.tta == 'hflip':
            pf = m.predict(np.ascontiguousarray(arr[:, ::-1, :][:, :, ::-1]), verbose=False)[0].probs.data.tolist()
            p = [(x + y) / 2.0 for x, y in zip(p, pf)]
        return p

    def is_bad_of(pr):
        """判定 bad：默认 argmax；--thresh 给定时按 p(bad) 阈值（用于扫工作点）。"""
        if a.thresh is None:
            return max(range(len(pr)), key=lambda i: pr[i]) in bad_idx
        return max(pr[j] for j in bad_idx) >= a.thresh

    # ---- val（按图切分）----
    tot = ok = nb = nbh = 0
    for lab in sorted(os.listdir(a.val_dir)):
        d = os.path.join(a.val_dir, lab)
        if not os.path.isdir(d):
            continue
        want_bad = (lab == 'bad')
        for f in glob.glob(os.path.join(d, '*.png')):
            pr = probs_of(f)
            is_bad = is_bad_of(pr)
            tot += 1
            ok += 1 if is_bad == want_bad else 0
            if want_bad:
                nb += 1
                nbh += 1 if is_bad else 0
    out.update({'n_val': tot, 'val_top1': round(ok / max(1, tot), 4),
                'n_bad': nb, 'bad_recall': round(nbh / max(1, nb), 4)})

    # ---- 池外（没标注过、过门限的裁图）----
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
    n = fire = 0
    kinds = collections.Counter()
    for c, r in sorted(man.items()):
        if c in done or n >= a.pool_n:
            continue
        p = os.path.join(CROPS, c)
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        if min(im.size) < a.min_side:
            continue
        g = np.asarray(im.convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32)
        gy, gx = np.gradient(g)
        if float(np.mean(np.hypot(gx, gy))) < a.min_edge:
            continue
        n += 1
        pr = probs_of(p)
        if is_bad_of(pr):
            fire += 1
            kinds[r['kind']] += 1
    out.update({'pool_n': n, 'pool_fire': fire, 'pool_rate': round(fire / max(1, n), 4),
                'pool_kinds': dict(kinds)})
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
