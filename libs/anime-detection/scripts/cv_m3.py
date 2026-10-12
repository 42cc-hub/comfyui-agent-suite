"""M3 的**按图分组 5 折交叉验证（OOF）**：用全部 382 张已标注裁图给出诚实指标。

为什么需要它：`cls_bin_v8` 的 val 只有 57 张（bad 16 张），一次 ±1 张就是 ±6%；
"工作点该取多少阈值"这种问题在 16 个正例上根本不可判（本窗口的教训：线上权重还带 42%/58% 的泄漏）。
5 折 OOF 让**每一张**已标注裁图都得到一个"模型没见过的"预测 ⇒ 召回/误报/阈值曲线建立在 ~167 张 bad 上。

纪律：折按**源图**分组（同一张图的全部裁图只在一折），与 `export_labels.py` 的按图划分一致 —— 否则
又回到 R39 之前那种"val 里有 37%~81% 的裁图在 train 里"的自欺。

用法：
  python scripts/cv_m3.py                      # 5 折，yolo11n-cls @224
  python scripts/cv_m3.py --folds 5 --imgsz 224 --epochs 80 --model yolo11n-cls.pt --seeds 1
输出：data/cv_m3/fold_k/（数据与 runs）、runs/cv_m3_fold_k/、data/cv_m3_oof.csv（每张裁图的 OOF p(bad)）+ 屏幕汇总
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
SRC = os.path.join(DATA, 'cls_bin_v8_noaug')      # 已标注裁图（按图切分的旧口径数据）
OUT = os.path.join(DATA, 'cv_m3')
OOF = os.path.join(DATA, 'cv_m3_oof.csv')
PY_M = r'python'


def rows(dataset=None):
    out = []
    src = dataset or SRC
    for sub in ('train', 'val'):
        for lab in ('ok', 'bad'):
            d = os.path.join(src, sub, lab)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith('.png'):
                    out.append((os.path.join(d, f), f, lab))
    return out


def group_of(name, man):
    r = man.get(name)
    if r:
        return r.get('source') or name
    # 兜底：去掉 _hand/_foot/_left_foot/_right_foot 后缀
    import re
    m = re.match(r'^(.*?)_(hand|left_foot|right_foot|foot)\d*', name)
    return m.group(1) if m else name


ap = None


def main():
    global SRC, OOF, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC, help='已标注裁图数据集（含 train/val 的 ok/bad 目录）')
    ap.add_argument('--out', default=OOF, help='折外预测 CSV 的输出路径')
    ap.add_argument('--work', default=OUT, help='折数据集与折间产物的目录')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--imgsz', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--model', default='yolo11n-cls.pt')
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--seeds', type=int, default=1,
                    help='每折训几个种子（>1 时可以额外给出"多种子并集"的 OOF —— 产线就是并集口径）')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--only-eval', action='store_true', help='只对已训好的折做 OOF 预测')
    a = ap.parse_args()
    SRC, OOF, OUT = a.src, a.out, a.work
    from ultralytics import YOLO

    man = {}
    with io.open(os.path.join(DATA, 'crops_todo', 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            man[r['crop']] = r
    rs = rows()
    # 按源图分桶；把桶按"含 bad 的多少"贪心分给当前 bad 最少的折（尽量让每折正例数接近）
    groups = {}
    for p, n, lab in rs:
        groups.setdefault(group_of(n, man), []).append((p, n, lab))
    order = sorted(groups.items(), key=lambda kv: -sum(1 for x in kv[1] if x[2] == 'bad'))
    fold_of, cnt = {}, [{'bad': 0, 'n': 0} for _ in range(a.folds)]
    for g, items in order:
        k = min(range(a.folds), key=lambda i: (cnt[i]['bad'], cnt[i]['n']))
        fold_of[g] = k
        cnt[k]['bad'] += sum(1 for x in items if x[2] == 'bad')
        cnt[k]['n'] += len(items)
    print('总裁图 %d（bad %d），源图 %d 张 → %d 折；每折 bad/总数：%s' % (
        len(rs), sum(1 for x in rs if x[2] == 'bad'), len(groups), a.folds,
        [(c['bad'], c['n']) for c in cnt]), flush=True)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    oof = []
    for si in range(a.seeds):
        sd = a.seed + si * 1000
        for k in range(a.folds):
            d = os.path.join(OUT, 'fold_%d' % k)
            for sub in ('train', 'val'):
                for lab in ('ok', 'bad'):
                    os.makedirs(os.path.join(d, sub, lab), exist_ok=True)
            for p, n, lab in rs:
                sub = 'val' if fold_of[group_of(n, man)] == k else 'train'
                dst = os.path.join(d, sub, lab, n)
                if not os.path.exists(dst):
                    shutil.copy(p, dst)
            name = 'cv_m3_f%d_s%d_%s_%d_e%d' % (k, sd, os.path.basename(a.model).replace('.pt', ''),
                                                a.imgsz, a.epochs)
            w = os.path.join(PROJ, 'runs', name, 'weights', 'best.pt')
            if not a.only_eval:
                logp = os.path.join(PROJ, 'runs', '_logs', '%s.log' % name)
                os.makedirs(os.path.dirname(logp), exist_ok=True)
                cmd = [PY_M, '-X', 'utf8', os.path.join(HERE, 'train.py'), '--task', 'cls', '--data', d,
                       '--model', a.model, '--epochs', str(a.epochs), '--imgsz', str(a.imgsz),
                       '--name', name, '--seed', str(sd), '--batch', str(a.batch)]
                t0 = time.time()
                with io.open(logp, 'w', encoding='utf-8') as fh:
                    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
                print('  fold %d seed %d 训练完成 %.0fs（rc=%s）' % (k, sd, time.time() - t0, r.returncode),
                      flush=True)
            if not os.path.exists(w):
                print('  !! fold %d seed %d 没有权重，跳过' % (k, sd), flush=True)
                continue
            m = YOLO(w)
            bi = [i for i, n in enumerate(m.names.values()) if n == 'bad']
            for lab in ('ok', 'bad'):
                for f in sorted(os.listdir(os.path.join(d, 'val', lab))):
                    p = os.path.join(d, 'val', lab, f)
                    pr = m.predict(p, verbose=False)[0].probs.data.tolist()
                    oof.append(dict(crop=f, fold=k, seed=sd, label=lab,
                                    p_bad=round(max(pr[j] for j in bi), 4)))

    with io.open(OOF, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop', 'fold', 'seed', 'label', 'p_bad'])
        w.writeheader()
        w.writerows(oof)
    print('\nOOF 共 %d 行 → %s' % (len(oof), OOF))
    nb = sum(1 for r in oof if r['label'] == 'bad' and r['seed'] == a.seed)
    nok = sum(1 for r in oof if r['label'] == 'ok' and r['seed'] == a.seed)

    def curve(rows, label):
        print('%-10s %-8s %-9s %-9s %s' % (label, '阈值', '召回', 'ok 误报', 'top1'))
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            tp = sum(1 for r in rows if r['label'] == 'bad' and r['p_bad'] >= t)
            fp = sum(1 for r in rows if r['label'] == 'ok' and r['p_bad'] >= t)
            n_b = sum(1 for r in rows if r['label'] == 'bad')
            n_o = sum(1 for r in rows if r['label'] == 'ok')
            print('%-10s %-8.2f %-8.3f %-9.3f %-9.3f' % ('', t, tp / max(1, n_b), fp / max(1, n_o),
                                                         (tp + (n_o - fp)) / max(1, len(rows))))

    for sd in sorted({r['seed'] for r in oof}):
        curve([r for r in oof if r['seed'] == sd], '单模型 s%d' % sd)
    if a.seeds > 1:
        # 并集口径（产线就是"任一模型 bad 即报"）：同一张裁图取各种子的最大 p(bad)
        by = {}
        for r in oof:
            e = by.setdefault(r['crop'], {'label': r['label'], 'p': []})
            e['p'].append(r['p_bad'])
        un = [dict(crop=c, label=v['label'], p_bad=max(v['p'])) for c, v in by.items()]
        curve(un, '并集(%d 种子)' % a.seeds)
    per = []
    for k in range(a.folds):
        rs_k = [r for r in oof if r['fold'] == k and r['seed'] == a.seed]
        nbk = sum(1 for r in rs_k if r['label'] == 'bad')
        tpk = sum(1 for r in rs_k if r['label'] == 'bad' and r['p_bad'] >= 0.5)
        per.append(tpk / max(1, nbk))
    if per:
        import statistics
        print('每折召回（阈值 0.5，种子 %d）：%s → 均值 %.3f ± %.3f' % (
            a.seed, ['%.2f' % x for x in per], statistics.mean(per),
            statistics.pstdev(per) if len(per) > 1 else 0.0))


if __name__ == '__main__':
    main()
