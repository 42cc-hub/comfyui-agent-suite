"""构造 `data/cls_bin_all`：把**全部**已标注裁图放进 train，用来产出正式权重（无 val 评估用途）。

为什么可以"拿全部数据训"：`scripts/cv_m3.py` 的按图分组 5 折 OOF 已经给出这套配置的诚实指标
（382 张 / 167 bad / 并集召回 0.79、ok 误报 0.186 —— 见 docs/runs.md R46）。有了折外估计，
正式权重就不必再留 val（留 val 只会少用 1/5 的数据）。
`val/` 里放 12 张裁图仅为满足 ultralytics 的目录约定，**不要**引用它的指标（它同时也在 train 里）。

用法：python scripts/build_cls_all.py [--val-n 12]
输出：data/cls_bin_all/{train,val}/{ok,bad}
"""
import argparse
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
SRC = os.path.join(DATA, 'cls_bin_v8_noaug')
OUT = os.path.join(DATA, 'cls_bin_all')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-n', type=int, default=12)
    a = ap.parse_args()
    files = []
    for sub in ('train', 'val'):
        for lab in ('ok', 'bad'):
            d = os.path.join(SRC, sub, lab)
            if os.path.isdir(d):
                files += [(os.path.join(d, f), lab, f) for f in sorted(os.listdir(d)) if f.endswith('.png')]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    rng = random.Random(20261010)
    rng.shuffle(files)
    val = files[:a.val_n]
    cnt = {}
    for p, lab, f in files:                       # 全部进 train
        d = os.path.join(OUT, 'train', lab)
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(d, f))
        cnt[('train', lab)] = cnt.get(('train', lab), 0) + 1
    for p, lab, f in val:                         # val 只是满足目录约定（**与 train 重复**）
        d = os.path.join(OUT, 'val', lab)
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(d, f))
        cnt[('val', lab)] = cnt.get(('val', lab), 0) + 1
    for k in sorted(cnt):
        print('  %-6s %-4s %d' % (k[0], k[1], cnt[k]))
    print('%s：train %d（全部已标注裁图）/ val %d（与 train 重复，仅满足目录约定，指标不可引用）' % (
        OUT, sum(v for k, v in cnt.items() if k[0] == 'train'), sum(v for k, v in cnt.items() if k[0] == 'val')))


if __name__ == '__main__':
    main()
