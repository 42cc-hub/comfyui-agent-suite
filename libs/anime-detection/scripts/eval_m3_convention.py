"""M3 裁图口径的**交叉评估**：同一批裁图（同一 basename、同一人工标签）的两种口径，喂给同一个权重。

为什么需要它：R39 发现"同一只脚，训练口径裁图 p(bad)=1.00、产线口径 0.00"，但只在一张图上验过。
`ablate.py` 里 `crop_prod`（按产线口径重切后重训）却是在**旧口径**的 `cls_val_common` 上评的
—— 那正是 R39 说的口径错配，等于让产线口径的模型考旧口径的卷子。本脚本把两种口径都考一遍：

  旧口径 = `crops_todo`/`prelabel_batch`：脚 = 姿态 FOOT3 框 + 0.35 外扩，手 = imgutils 手框 + 0.10
          （= `data/cls_val_common`，也是 `cls_bin_v8` 的 val）
  产线口径 = `locate.py` 默认：检测器框 + 0.10（= `data/cls_bin_prod/val`，由 recut_crops_prod.py 重切）

对每个权重输出：两种口径下的 top1 / bad 召回，以及 bad 与 ok 两类裁图的 p(bad) **均值漂移**
（漂移就是"口径错配"的量化：同一个模型换口径后概率往哪边跑）。

用法：
  python scripts/eval_m3_convention.py --weights runs/m3_review_v1/weights/best.pt runs/ab_crop_prod_s20261010/weights/best.pt
  python scripts/eval_m3_convention.py --all-ab        # 自动收集 runs/ab_*/weights/best.pt
输出：屏幕表格 + data/m3_convention_eval.csv
"""
import argparse
import csv
import glob
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
OLD = os.path.join(DATA, 'cls_val_common')            # 旧口径（prelabel）
PROD = os.path.join(DATA, 'cls_bin_prod', 'val')      # 产线口径（检测器框）


def pairs():
    """返回 [(name, label, old_path, prod_path)]：同一 basename 的两种口径。"""
    out, miss = [], []
    for lab in ('ok', 'bad'):
        for p in sorted(glob.glob(os.path.join(OLD, lab, '*.png'))):
            n = os.path.basename(p)
            q = os.path.join(PROD, lab, n)
            if os.path.exists(q):
                out.append((n, lab == 'bad', p, q))
            else:
                miss.append(n)
    return out, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='*', default=[])
    ap.add_argument('--all-ab', action='store_true', help='收集 runs/ab_*/weights/best.pt')
    ap.add_argument('--out', default=os.path.join(DATA, 'm3_convention_eval.csv'))
    a = ap.parse_args()
    ws = list(a.weights)
    if a.all_ab:
        ws += sorted(glob.glob(os.path.join(PROJ, 'runs', 'ab_*', 'weights', 'best.pt')))
    if not ws:
        ap.error('需要 --weights 或 --all-ab')
    from ultralytics import YOLO

    pr, miss = pairs()
    print('配对 %d 张（ok %d / bad %d），旧口径有而产线口径缺 %d 张'
          % (len(pr), sum(1 for x in pr if not x[1]), sum(1 for x in pr if x[1]), len(miss)), flush=True)

    rows = []
    hdr = '%-26s %-9s %-9s | %-9s %-9s | %s' % ('weights', 'top1旧', 'top1产线', '召回旧', '召回产线', 'p(bad) 均值 旧→产线')
    print('\n' + hdr)
    print('-' * len(hdr))
    for w in ws:
        tag = os.path.basename(os.path.dirname(os.path.dirname(w)))
        m = YOLO(w)
        bi = [i for i, n in m.names.items() if n == 'bad']

        def pbad(p):
            return max(m.predict(p, verbose=False)[0].probs.data.tolist()[j] for j in bi)

        st = {'old': {'ok': 0, 'n': 0, 'bad': 0, 'nb': 0, 's_ok': 0.0, 's_bad': 0.0},
              'prod': {'ok': 0, 'n': 0, 'bad': 0, 'nb': 0, 's_ok': 0.0, 's_bad': 0.0}}
        for n, is_bad, po, pp in pr:
            for key, path in (('old', po), ('prod', pp)):
                v = pbad(path)
                hit = (v >= 0.5)
                st[key]['n'] += 1
                st[key]['ok'] += 1 if hit == is_bad else 0
                if is_bad:
                    st[key]['nb'] += 1
                    st[key]['bad'] += 1 if hit else 0
                    st[key]['s_bad'] += v
                else:
                    st[key]['s_ok'] += v
        o, d = st['old'], st['prod']
        row = dict(weights=tag,
                   top1_old=round(o['ok'] / max(1, o['n']), 4), top1_prod=round(d['ok'] / max(1, d['n']), 4),
                   rec_old=round(o['bad'] / max(1, o['nb']), 4), rec_prod=round(d['bad'] / max(1, d['nb']), 4),
                   pbad_ok_old=round(o['s_ok'] / max(1, o['n'] - o['nb']), 4),
                   pbad_ok_prod=round(d['s_ok'] / max(1, d['n'] - d['nb']), 4),
                   pbad_bad_old=round(o['s_bad'] / max(1, o['nb']), 4),
                   pbad_bad_prod=round(d['s_bad'] / max(1, d['nb']), 4))
        rows.append(row)
        print('%-26s %-9s %-9s | %-9s %-9s | ok %.3f→%.3f  bad %.3f→%.3f' % (
            tag, row['top1_old'], row['top1_prod'], row['rec_old'], row['rec_prod'],
            row['pbad_ok_old'], row['pbad_ok_prod'], row['pbad_bad_old'], row['pbad_bad_prod']))
    with io.open(a.out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('→ %s' % a.out)


if __name__ == '__main__':
    main()
