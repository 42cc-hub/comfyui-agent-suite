# -*- coding: utf-8 -*-
"""task-10 补充④：按类路由的混合模型指标（纯算术合并，不重新推理）。

输入（已有测评产物）：
  rot_semi/eval.json          —— 5 个角度桶 × 模型 × 类 的 tp/fp/fn/gt/pred
  rot_semi/upright_eval.txt   —— upright 独立集 hand/foot（无 person 真值）
  real_lying/real_lying.txt   —— 真实横躺 14 张的 person/hand/foot
合并方式：每个类从不同模型取 tp/fp（hybrid_A: person←old, hand/foot←rot_semi；
hybrid_B: person←v2, hand/foot←rot_semi）。**这是算术合并，不是真跑一个混合模型。**
产物：rot_semi/hybrid_merge.txt（同时用于报告 §11）
"""
import io
import json
import os
import re

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REP = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'rot_semi')
RL = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'real_lying')
MODELS = ['old', 'v2', 'rot', 'rot_semi']


def agg_eval():
    """{bucket: {model: {cls: [tp, fp, gt, pred]}}}"""
    d = json.load(io.open(os.path.join(REP, 'eval.json'), encoding='utf-8'))
    out = {}
    for bucket, recs in d.items():
        for r in recs:
            a = out.setdefault(bucket, {}).setdefault(r['model'], {}).setdefault(r['cls'], [0, 0, 0, 0])
            a[0] += r['tp']
            a[1] += r['fp']
            a[2] += r['gt']
            a[3] += r['pred']
    return out


def hybrid(base, person_from):
    """返回一个新模型视图：person 取 person_from，hand/foot 取 base（去掉 base 的 person）。"""
    h = {}
    for cls in ('person', 'hand', 'foot'):
        src = person_from if cls == 'person' else base
        if cls in src:
            h[cls] = src[cls]
    return h


def fmt(cls_stats, cls):
    s = cls_stats.get(cls)
    if not s or s[2] == 0:
        return '—'
    tp, fp, gt, pred = s
    return '%.1f / %.1f' % (100.0 * tp / gt, 100.0 * tp / max(1, tp + fp))


def overall(stats):
    tp = sum(v[0] for v in stats.values())
    fp = sum(v[1] for v in stats.values())
    gt = sum(v[2] for v in stats.values())
    return '%.1f / %.1f' % (100.0 * tp / max(1, gt), 100.0 * tp / max(1, tp + fp)), gt


def parse_like_table(path, want_header):
    """从报告 txt 的 markdown 表里抽 model -> {cls: [tp,fp,gt,pred]}。

    这些表的单元格形如 `GT/TP/FP`（如 `79/31/31`），百分比单元格忽略。
    """
    txt = io.open(path, encoding='utf-8').read()
    i = txt.index(want_header)
    seg = txt[i:]
    # 只取该标题下的第一张表：遇到下一个 "## " 标题就截断
    j = seg.find('\n## ', 1)
    if j > 0:
        seg = seg[:j]
    rows = {}
    for line in seg.split('\n'):
        m = re.match(r'\|\s*(old|v2|rot|rot_semi|semi)\s*\|(.*)\|', line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        cells = [c.strip() for c in rest.split('|')]
        triples = []
        for c in cells:
            mm = re.match(r'^(\d+)/(\d+)/(\d+)$', c)
            if mm:
                gt, tp, fp = [int(x) for x in mm.groups()]
                triples.append([tp, fp, gt, tp + fp])
            if c == '—':
                triples.append(None)
        names = ('person', 'hand', 'foot') if len(triples) == 3 else ('hand', 'foot')
        if len(triples) in (2, 3):
            rows[tag] = {n: t for n, t in zip(names, triples) if t}
    return rows


def main():
    A = agg_eval()
    L = []

    def say(s=''):
        print(s)
        L.append(s)

    say('# 按类路由的混合指标（纯算术合并：每个类取不同模型的 tp/fp，**不是真跑一个混合模型**）')
    say('')
    say('hybrid_A = person←old，hand/foot←rot_semi；hybrid_B = person←v2，hand/foot←rot_semi。')
    say('R / P 为该类别的 micro 值；"整体" = 三类 tp 之和 / 三类 gt 之和（召回）与 tp/(tp+fp)（精度）。')
    say('')
    for bucket in ('0', '30', '60', '90', 'other'):
        st = A[bucket]
        views = {'old': st['old'], 'v2': st['v2'], 'rot': st['rot'], 'rot_semi': st['rot_semi'],
                 'hybrid_A (person←old)': hybrid(st['rot_semi'], st['old']),
                 'hybrid_B (person←v2)': hybrid(st['rot_semi'], st['v2'])}
        say('## 合成角度桶 %s°（holdout15）' % bucket)
        gt = {c: st['old'].get(c, [0, 0, 0, 0])[2] for c in ('person', 'hand', 'foot')}
        say('GT 框数：person %d / hand %d / foot %d' % (gt['person'], gt['hand'], gt['foot']))
        say('')
        say('| 模型 | person R/P | hand R/P | foot R/P | 整体 R/P |')
        say('|---|---|---|---|---|')
        for name, v in views.items():
            o, _ = overall(v)
            say('| %s | %s | %s | %s | %s |' % (name, fmt(v, 'person'), fmt(v, 'hand'), fmt(v, 'foot'), o))
        say('')

    # 全角度合计
    tot = {}
    for bucket, st in A.items():
        for model in ('old', 'v2', 'rot', 'rot_semi'):
            for cls, vals in st[model].items():
                a = tot.setdefault(model, {}).setdefault(cls, [0, 0, 0, 0])
                for i in range(4):
                    a[i] += vals[i]
    views = {'old': tot['old'], 'v2': tot['v2'], 'rot': tot['rot'], 'rot_semi': tot['rot_semi'],
             'hybrid_A (person←old)': hybrid(tot['rot_semi'], tot['old']),
             'hybrid_B (person←v2)': hybrid(tot['rot_semi'], tot['v2'])}
    say('## 合成角度桶 · 全角度合计（0/30/60/90/other）')
    say('GT 框数：person %d / hand %d / foot %d'
        % (tot['old']['person'][2], tot['old']['hand'][2], tot['old']['foot'][2]))
    say('')
    say('| 模型 | person R/P | hand R/P | foot R/P | 整体 R/P |')
    say('|---|---|---|---|---|')
    for name, v in views.items():
        o, _ = overall(v)
        say('| %s | %s | %s | %s | %s |' % (name, fmt(v, 'person'), fmt(v, 'hand'), fmt(v, 'foot'), o))
    say('')

    # upright：无 person 真值，hybrid 等同于 rot_semi（hand/foot 全取 rot_semi）
    up = parse_like_table(os.path.join(REP, 'upright_eval.txt'), '| 模型 | hand GT/TP/FP |')
    say('## upright 独立集（data/hand_gt，GT hand %d / foot %d；该集无 person 真值）'
        % (up['old']['hand'][2], up['old']['foot'][2]))
    say('')
    say('注意：hybrid 只改 person，本集没有 person 真值 ⇒ **hybrid_A/B 与 rot_semi 逐项相同**。')
    say('')
    say('| 模型 | hand R/P | foot R/P |')
    say('|---|---|---|')
    for name in ('old', 'v2', 'rot', 'rot_semi'):
        v = up[name]
        say('| %s | %s | %s |' % (name, fmt(v, 'hand'), fmt(v, 'foot')))
    say('| hybrid_A / hybrid_B（= rot_semi） | %s | %s |' % (fmt(up['rot_semi'], 'hand'), fmt(up['rot_semi'], 'foot')))
    say('')

    # 真实横躺
    rl = parse_like_table(os.path.join(RL, 'real_lying.txt'), '## 真实横躺（A+B 全部）')
    say('## 真实横躺子集（14 张：A 组 holdout15 里 5 张 + B 组 hand_gt 里 9 张）')
    say('GT 框数：person %d / hand %d / foot %d' % (rl['old']['person'][2], rl['old']['hand'][2], rl['old']['foot'][2]))
    say('')
    say('| 模型 | person R/P | hand R/P | foot R/P | 整体 R/P |')
    say('|---|---|---|---|---|')
    views = {'old': rl['old'], 'v2': rl['v2'], 'rot': rl['rot'], 'rot_semi': rl['rot_semi'],
             'hybrid_A (person←old)': hybrid(rl['rot_semi'], rl['old']),
             'hybrid_B (person←v2)': hybrid(rl['rot_semi'], rl['v2'])}
    for name, v in views.items():
        o, _ = overall(v)
        say('| %s | %s | %s | %s | %s |' % (name, fmt(v, 'person'), fmt(v, 'hand'), fmt(v, 'foot'), o))
    say('')
    io.open(os.path.join(REP, 'hybrid_merge.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(REP, 'hybrid_merge.txt'))


if __name__ == '__main__':
    main()
