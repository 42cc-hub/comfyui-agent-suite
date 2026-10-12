"""M3 **类别口径**消融的数据构建：同一批人工标签，三种"什么算 bad"的定义。

为什么做这条（2026-10-12 窗口的目标之一）：产线要拦的是"**指/趾数不对/畸形**"，但当前二元任务把所有
`not_hand`/`not_foot`（框根本不是该部位）也并进 `bad`（`export_labels.py --collapse-bad`）。
而产线上这些裁图**本来就该被 WD14 的 state/部位错配挡掉**（`locate.py` 的 SKIP_STATES）⇒ 它们进训练集
只会教模型"不像手 = bad"，而在线第 1 页实测的正是这种样本（花纹 p=1.00、分歧 0.99）。
本脚本用**同一套按源图切分**产出三个数据集，交给 `ablate.py` 用同一套指标比：

  `cls_tax_binary`  二元：ok vs {bad_count, bad_shape, not_hand, not_foot}   ← 现产线口径（对照）
  `cls_tax_parts`   二元：ok vs {bad_count, bad_shape}（**丢掉 not_\***）      ← 想验证的主假设
  `cls_tax_3cls`    三类：ok / bad_count / bad_shape（丢掉 not_*）

用法：
  python scripts/build_cls_taxonomy.py            # 三个都建
  python scripts/build_cls_taxonomy.py --only parts
输出：data/cls_tax_{binary,parts,3cls}/{train,val}/<label>/ + 屏幕计数与"源图泄漏=0"自检
"""
import argparse
import collections
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(PROJ, 'data')
CROPS = os.path.join(DATA, 'crops_todo')
from retrain_m3 import collect, src_of        # noqa: E402  复用"汇总所有队列标签 + 源图"两个函数

NOTPART = ('not_hand', 'not_foot')
REAL = ('bad_count', 'bad_shape')


def raw_labels():
    """{crop: 原始五类标签}（不折叠），只保留非 skip。"""
    import csv
    import glob
    import io
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, 'cls_queue*', 'index.csv'))):
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            v = (r.get('label') or '').strip()
            if v and v != 'skip':
                out[r['crop']] = v
    return out


def build(name, mapping, val_frac=0.15):
    raw = raw_labels()
    items = [(c, mapping[v]) for c, v in raw.items() if mapping.get(v)]
    out = os.path.join(DATA, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    by_img = collections.defaultdict(list)
    for c, y in items:
        by_img[src_of(c)].append((c, y))
    imgs = sorted(by_img)
    random.Random(20261010).shuffle(imgs)
    target = int(round(len(items) * val_frac))
    val_imgs, n = set(), 0
    for im in imgs:
        if n >= target:
            break
        val_imgs.add(im)
        n += len(by_img[im])
    cnt = collections.Counter()
    for c, y in items:
        src = os.path.join(CROPS, c)
        if not os.path.exists(src):
            continue
        sub = 'val' if src_of(c) in val_imgs else 'train'
        d = os.path.join(out, sub, y)
        os.makedirs(d, exist_ok=True)
        shutil.copy(src, os.path.join(d, c))
        cnt[(sub, y)] += 1

    def srcs(sub):
        s = set()
        for y in os.listdir(os.path.join(out, sub)):
            d = os.path.join(out, sub, y)
            if os.path.isdir(d):
                s |= {src_of(f) for f in os.listdir(d)}
        return s

    leak = srcs('train') & srcs('val') if os.path.isdir(os.path.join(out, 'val')) else set()
    print('%-16s train %s | val %s | 源图泄漏 %d' % (
        name, {y: v for (s, y), v in sorted(cnt.items()) if s == 'train'},
        {y: v for (s, y), v in sorted(cnt.items()) if s == 'val'}, len(leak)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, choices=['binary', 'parts', '3cls'])
    a = ap.parse_args()
    maps = {
        'binary': {**{k: 'ok' for k in ('ok',)}, **{k: 'bad' for k in REAL + NOTPART}},
        'parts': {**{k: 'ok' for k in ('ok',)}, **{k: 'bad' for k in REAL}},
        '3cls': {**{k: 'ok' for k in ('ok',)}, **{k: k for k in REAL}},
    }
    for name, m in maps.items():
        if a.only and a.only != name:
            continue
        build('cls_tax_%s' % name, m)


if __name__ == '__main__':
    main()
