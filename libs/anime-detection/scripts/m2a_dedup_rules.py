# -*- coding: utf-8 -*-
"""person 去重规则对比（纯 CPU，用预填期缓存预测）。

评测集：data/hand_gt_multi/index.csv 的 60 图 / 97 个 GT person 框；
预测：data/hand_gt_multi/preds/queue_preds.json（det_v2 person，conf>=0.30）。
规则候选 R0..R4 只作用于 person 框（与 m2a_multi_queue.py 的队列构建口径一致）。

指标：
  - 丢掉的真人数：GT 在规则后没有任何 pred IoU>=0.5（并给相对 R4 off 的增量）
  - |n_pred-n_gt| 均值/中位
  - 残留近重复对数：pred 两两 IoU>=0.9 的数目
  - 单列"横躺/贴身"子集
"""
import csv
import io
import json
import os
import statistics

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REP = os.path.abspath(os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier'))
GT = os.path.join(SUITE, 'data', 'hand_gt_multi', 'index.csv')
PRED = os.path.join(SUITE, 'data', 'hand_gt_multi', 'preds', 'queue_preds.json')
CONF = 0.30


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contain(small, big):
    """小框被大框包含的比例"""
    iw = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    ih = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = (small[2] - small[0]) * (small[3] - small[1])
    return (iw * ih) / a if a > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def load():
    gt = {}
    for r in csv.DictReader(io.open(GT, encoding='utf-8-sig')):
        if r['target'] != 'person' or not r['boxes'].strip() or r['verdict'] == 'neg':
            continue
        gt[r['crop']] = [tuple(float(v) for v in s.split(',')) for s in r['boxes'].split(';') if s.strip()]
    raw = json.load(io.open(PRED, encoding='utf-8'))
    pr = {}
    for stem, v in raw.items():
        ps = [(float(p['conf']), p['box']) for p in v.get('det', {}).get('person', []) if float(p.get('conf', 1)) >= CONF]
        pr[stem + '.png'] = dict(w=v['w'], h=v['h'], boxes=ps)
    return gt, pr


def apply_rule(rule, items, W, H):
    """items: [(conf, box)] -> 去重后的 [(conf, box)]（按 conf 降序处理）"""
    items = sorted(items, key=lambda t: -t[0])
    if rule == 'R4':                       # off
        return list(items)
    keep = []
    for conf, box in items:
        drop = False
        for kc, kb in keep:
            v = iou(box, kb)
            if rule == 'R0':
                drop = v >= 0.5 and area(box) / (W * H) > 0.5 and area(kb) / (W * H) > 0.5
            elif rule == 'R1':
                r = area(box) / area(kb) if area(kb) > 0 else 9
                drop = v >= 0.5 and 0.9 <= r <= 1.1 and area(box) / (W * H) > 0.5 and area(kb) / (W * H) > 0.5
            elif rule == 'R2':
                drop = v >= 0.75
            elif rule == 'R3':
                small, big = (box, kb) if area(box) <= area(kb) else (kb, box)
                drop = v >= 0.5 and contain(small, big) >= 0.95
            if drop:
                break
        if not drop:
            keep.append((conf, box))
    return keep


def metrics(gt, pr, rule, subset=None):
    lost = 0
    n_gt = n_pred = 0
    diffs = []
    dup_pairs = 0
    for crop, g in gt.items():
        if subset is not None and crop not in subset:
            continue
        rec = pr.get(crop)
        if rec is None:
            continue
        kept = apply_rule(rule, rec['boxes'], rec['w'], rec['h'])
        n_gt += len(g)
        n_pred += len(kept)
        diffs.append(abs(len(kept) - len(g)))
        for gb in g:
            if not any(iou(gb, kb) >= 0.5 for _, kb in kept):
                lost += 1
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if iou(kept[i][1], kept[j][1]) >= 0.9:
                    dup_pairs += 1
    return dict(rule=rule, gt=n_gt, pred=n_pred, lost_gt=lost,
                absdiff_mean=round(statistics.mean(diffs), 2) if diffs else 0,
                absdiff_median=statistics.median(diffs) if diffs else 0,
                dup_pairs=dup_pairs, imgs=len(diffs))


def main():
    gt, pr = load()
    # 横躺/贴身子集：GT>=2 且两个 GT 人框互相 IoU>=0.25（贴在一起）——另加已知横躺图
    lying = set()
    for crop, g in gt.items():
        if len(g) >= 2:
            best = max(iou(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g)))
            if best >= 0.25:
                lying.add(crop)
    for extra in ('02_duo_wariza_lying_2000.png', '03_duo_wariza_lying_b_2000.png', '04_duo_69_2000.png',
                  '01_bed_tailwrap_2000.png', 'g4_01_lumine_sickbed_fixed_2000.png',
                  'arrest3_01_jail_2000x1334.png', '01_vivian_trio_2000.png', '01_wet_sheet_2000x2624.png'):
        if extra in gt:
            lying.add(extra)
    rules = ['R4', 'R0', 'R1', 'R2', 'R3']
    all_m = [metrics(gt, pr, r) for r in rules]
    ly_m = [metrics(gt, pr, r, lying) for r in rules]
    base = all_m[0]
    lines = ['person 去重规则对比（60 图 / %d GT 人框；预测 = det_v2 person @conf %.2f 缓存）' % (base['gt'], CONF),
             '=' * 96,
             '%-4s %6s %6s %8s %10s %12s %10s %10s' % ('规则', '图', 'GT', 'rule后pred', '丢掉真人', '相对R4多丢', '|Δ|均值', '残留重对'),
             ]
    desc = {'R0': 'area>0.5双方 & IoU>=0.5（现规则）', 'R1': 'IoU>=0.5 & 面积比[0.9,1.1]（+area>0.5）',
            'R2': 'IoU>=0.75', 'R3': 'IoU>=0.5 & 小框被包含>=0.95', 'R4': '不删（基线）'}
    for m in all_m:
        lines.append('%-4s %6d %6d %8d %10d %12s %12.2f %10d   %s' % (
            m['rule'], m['imgs'], m['gt'], m['pred'], m['lost_gt'],
            '%+d' % (m['lost_gt'] - base['lost_gt']), m['absdiff_mean'], m['dup_pairs'], desc[m['rule']]))
    lines += ['', '横躺/贴身子集（%d 张：GT>=2 且两人框 IoU>=0.25，另加已知横躺图）' % len(lying),
              '%-4s %6s %6s %8s %10s %12s %10s %10s' % ('规则', '图', 'GT', 'rule后pred', '丢掉真人', '相对R4多丢', '|Δ|均值', '残留重对')]
    ly_base = ly_m[0]
    for m in ly_m:
        lines.append('%-4s %6d %6d %8d %10d %12s %12.2f %10d   %s' % (
            m['rule'], m['imgs'], m['gt'], m['pred'], m['lost_gt'],
            '%+d' % (m['lost_gt'] - ly_base['lost_gt']), m['absdiff_mean'], m['dup_pairs'], ''))
    txt = '\n'.join(lines)
    os.makedirs(os.path.join(REP, 'dedup_rules'), exist_ok=True)
    io.open(os.path.join(REP, 'dedup_rules', 'rules.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)
    json.dump(dict(all=all_m, lying=ly_m, lying_imgs=sorted(lying)),
              io.open(os.path.join(REP, 'dedup_rules', 'rules.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
