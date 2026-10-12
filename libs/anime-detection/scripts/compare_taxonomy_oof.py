"""比较两种**类别口径**训练出来的 M3（折外预测），重点看"真部位"那一批裁图。

背景：现产线二元定义把 `not_hand`/`not_foot` 也算 bad，占了 bad 的 126/174 = 72%；
真正"数错/畸形"只有 48 张。所以必须把"模型在真部位上判不判得准"单独拎出来看：
同一个权重在混合 val 上召回高，可能只是"很会认这不是手"，而不是"会数指头"。

用法：
  python scripts/compare_taxonomy_oof.py --a data/cv_m3_oof.csv --b data/cv_m3_parts_oof.csv
输入：两份折外 CSV（`cv_m3.py` 产出，列 crop/fold/seed/label/p_bad；label 是各自数据集里的 ok/bad）
      + 队列里的**原始五类标签**（从 data/cls_queue*/index.csv 读，做"真部位/非部位"的划分）
输出：屏幕三张表（全体 / 真部位 / 非部位）
"""
import argparse
import collections
import csv
import glob
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
REAL = ('bad_count', 'bad_shape')
NOTPART = ('not_hand', 'not_foot')


def raw_labels():
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, 'cls_queue*', 'index.csv'))):
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            v = (r.get('label') or '').strip()
            if v and v != 'skip':
                out[r['crop']] = v
    return out


def load(path):
    """→ {crop: [p_bad, ...]}（各种子/各折的折外预测，一张裁图一条）"""
    d = collections.defaultdict(list)
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        d[r['crop']].append(float(r['p_bad']))
    return d


def curve(name, preds, raw, sel):
    """sel(crop) → True/False；并集口径（多种子取最大 p）"""
    pos = [c for c in preds if sel(c) and raw.get(c) in REAL]
    neg = [c for c in preds if sel(c) and raw.get(c) == 'ok']
    if not pos and not neg:
        print('  %-22s （无样本）' % name)
        return
    print('  %-22s 正例 %2d / 反例 %3d' % (name, len(pos), len(neg)))
    print('     %-6s %-8s %-8s' % ('阈值', '召回', '误报'))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        tp = sum(1 for c in pos if max(preds[c]) >= t)
        fp = sum(1 for c in neg if max(preds[c]) >= t)
        print('     %-6.2f %-8.3f %-8.3f' % (t, tp / max(1, len(pos)), fp / max(1, len(neg))))
    import statistics
    pm = [max(preds[c]) for c in pos]
    nm = [max(preds[c]) for c in neg]
    print('     平均 p(bad)：正例 %.3f  反例 %.3f' % (statistics.mean(pm) if pm else 0,
                                                    statistics.mean(nm) if nm else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=os.path.join(DATA, 'cv_m3_oof.csv'))
    ap.add_argument('--b', default=os.path.join(DATA, 'cv_m3_parts_oof.csv'))
    ap.add_argument('--label-a', default='binary(现口径)')
    ap.add_argument('--label-b', default='parts(只留数错/畸形)')
    a = ap.parse_args()
    raw = raw_labels()
    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    print('折外预测：A %d 张 / B %d 张 / 交集 %d 张' % (len(A), len(B), len(common)))
    rl = collections.Counter(raw.get(c, '?') for c in common)
    print('交集的原始标签分布：%s\n' % dict(rl))

    print('【表 1】全体裁图（各自口径的 universe，含 not_*）')
    curve(a.label_a, A, raw, lambda c: True)
    curve(a.label_b, B, raw, lambda c: True)

    print('\n【表 2】只统计**真部位**裁图（正例=bad_count/bad_shape，反例=ok）—— 这才是"会不会数指头"')
    curve(a.label_a, A, raw, lambda c: raw.get(c) in REAL or raw.get(c) == 'ok')
    curve(a.label_b, B, raw, lambda c: raw.get(c) in REAL or raw.get(c) == 'ok')

    print('\n【表 3】只统计**非部位**裁图（not_hand/not_foot）—— 这类样本在训练里对 B 是不存在的')
    nA = [c for c in A if raw.get(c) in NOTPART]
    nB = [c for c in B if raw.get(c) in NOTPART]
    for lab, preds, sel in ((a.label_a, A, nA), (a.label_b, B, nB)):
        if not sel:
            print('  %-22s （无样本）' % lab)
            continue
        fire = sum(1 for c in sel if max(preds[c]) >= 0.5)
        print('  %-22s 非部位 %3d 张，判 bad(=不认这是手/脚) %3d 张（%.2f）' % (
            lab, len(sel), fire, fire / len(sel)))


if __name__ == '__main__':
    main()
