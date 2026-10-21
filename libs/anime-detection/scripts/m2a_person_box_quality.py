# -*- coding: utf-8 -*-
"""person 框质量 + 「巨型重复框」缺陷量化（纯 CPU，不跑模型）。

数据：`data/hand_gt_multi/index.csv` 的 target=person 行（用户真值 97 框）
      `data/hand_gt_multi/preds/queue_preds.json`（预填期 det 结果，含各阈值 conf）

四组数：①框质量（最佳匹配 IoU / 面积比 / 中心偏移）②巨型框率 ③重复框（IoU≥0.8 或
containment≥0.7，分开报被 --drop-nested 0.90 拦掉的与漏掉的）④去重规则排序。
排序的两个数用「误删代价」而不是"该删比例"：删掉一个重复框但真值仍有别的框覆盖 = 无损；
真值因此失去唯一的 ≥0.5 覆盖 = 有害。
"""
import argparse
import csv
import io
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
DATA = os.path.join(SUITE, 'data')
BASE = os.path.join(SUITE, '..', '..', 'team')

CONF_MAIN = 0.30
CONF_LO = 0.10
GIANT = 0.6
NEST_CUR = 0.90


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def containment(a, b):
    """小框被大框包含的比例。"""
    m = min(area(a), area(b))
    return inter(a, b) / m if m > 0 else 0.0


def parse_boxes(s):
    out = []
    for seg in (s or '').split(';'):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(x) for x in seg.split(',')]
        if len(v) == 4:
            out.append(v)
    return out


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_gt():
    rows = list(csv.DictReader(io.open(os.path.join(DATA, 'hand_gt_multi', 'index.csv'), encoding='utf-8-sig')))
    gt = {}
    for r in rows:
        if r['target'] != 'person':
            continue
        bx = parse_boxes(r['boxes']) if r['verdict'] in ('draw', 'pass') else []
        if bx:
            gt[r['crop'].replace('.png', '')] = bx
    return gt


def load_preds():
    j = json.load(io.open(os.path.join(DATA, 'hand_gt_multi', 'preds', 'queue_preds.json'), encoding='utf-8'))
    out = {}
    for stem, v in j.items():
        out[stem] = {'size': (float(v.get('w') or 0), float(v.get('h') or 0)),
                     'person': [{'box': [float(x) for x in b['box']], 'conf': float(b['conf'])}
                                for b in (v.get('det', {}).get('person') or [])]}
    return out


def best_iou_boxes(box, boxes):
    return max([iou(box, g) for g in boxes], default=0.0)


def collect(gt_map, pred_map, conf):
    rows = []
    for stem, gts in gt_map.items():
        pm = pred_map.get(stem)
        if not pm:
            continue
        preds = [p for p in pm['person'] if p['conf'] >= conf]
        W, H = pm['size']
        rows.append({'stem': stem, 'gt': gts, 'pred': preds, 'W': W, 'H': H,
                     'diag': (W * W + H * H) ** 0.5, 'img_area': W * H})
    return rows


def quality(rows):
    ious, ratios, offs = [], [], []
    for r in rows:
        for g in r['gt']:
            best, bj = 0.0, -1
            for j, p in enumerate(r['pred']):
                v = iou(g, p['box'])
                if v > best:
                    best, bj = v, j
            ious.append(best)
            if bj >= 0 and best > 0:
                p = r['pred'][bj]['box']
                ratios.append(area(p) / area(g))
                dx = ((p[0] + p[2]) - (g[0] + g[2])) / 2.0
                dy = ((p[1] + p[3]) - (g[1] + g[3])) / 2.0
                offs.append(((dx * dx + dy * dy) ** 0.5) / r['diag'])
    n = len(ious)
    return {'n_gt': n,
            'iou_median': round(statistics.median(ious), 3) if ious else 0.0,
            'iou_lt50': round(100.0 * sum(1 for v in ious if v < 0.5) / max(1, n), 1),
            'iou_lt70': round(100.0 * sum(1 for v in ious if v < 0.7) / max(1, n), 1),
            'ratio_median': round(statistics.median(ratios), 3) if ratios else 0.0,
            'ratio_p10': round(pct(ratios, 0.10), 3) if ratios else 0.0,
            'ratio_p90': round(pct(ratios, 0.90), 3) if ratios else 0.0,
            'off_median_pct_diag': round(100.0 * statistics.median(offs), 2) if offs else 0.0}


def giant(rows):
    tot = n_giant = imgs = 0
    ious, confs = [], []
    for r in rows:
        hit = False
        for p in r['pred']:
            tot += 1
            if r['img_area'] > 0 and area(p['box']) / r['img_area'] > GIANT:
                n_giant += 1
                hit = True
                confs.append(p['conf'])
                ious.append(best_iou_boxes(p['box'], r['gt']))
        imgs += 1 if hit else 0
    return {'n_pred': tot, 'n_giant': n_giant,
            'share_pred': round(100.0 * n_giant / max(1, tot), 1),
            'imgs': imgs, 'n_imgs': len(rows), 'share_imgs': round(100.0 * imgs / max(1, len(rows)), 1),
            'iou_median': round(statistics.median(ious), 3) if ious else 0.0,
            'iou_lt50': round(100.0 * sum(1 for v in ious if v < 0.5) / max(1, len(ious)), 1),
            'conf_median': round(statistics.median(confs), 3) if confs else 0.0}


def dup_pairs(rows):
    tot = caught = 0
    missed = []
    for r in rows:
        P = r['pred']
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                a, b = P[i]['box'], P[j]['box']
                if iou(a, b) >= 0.8 or containment(a, b) >= 0.7:
                    tot += 1
                    small = i if area(a) <= area(b) else j
                    big = j if small == i else i
                    if containment(P[small]['box'], P[big]['box']) >= NEST_CUR:
                        caught += 1
                    else:
                        missed.append({'stem': r['stem'], 'pair': (i, j), 'iou': round(iou(a, b), 3),
                                       'cont': round(containment(a, b), 3),
                                       'area_pct': (round(100.0 * area(a) / max(1.0, r['img_area']), 1),
                                                    round(100.0 * area(b) / max(1.0, r['img_area']), 1)),
                                       'conf': (round(P[i]['conf'], 3), round(P[j]['conf'], 3))})
    return tot, caught, missed


def kept_by_rule(r, mode, thr, prio, cond=None):
    """贪心：按 prio 排序依次保留；与已保留框冲突（mode 指标 ≥ thr）则丢。
    cond(r, k, q) 为额外的"才算冲突"条件（用于只处理巨型对）。"""
    P = r['pred']
    key = (lambda k: (-area(P[k]['box']), -P[k]['conf'])) if prio == 'area' \
        else (lambda k: (-P[k]['conf'], -area(P[k]['box'])))
    kept = []
    for k in sorted(range(len(P)), key=key):
        drop = False
        for q in kept:
            a, b = P[k]['box'], P[q]['box']
            v = iou(a, b) if mode == 'iou' else containment(a, b)
            if v >= thr and (cond is None or cond(r, k, q)):
                drop = True
                break
        if not drop:
            kept.append(k)
    return set(kept)


def covered(r, kept):
    n = 0
    for g in r['gt']:
        if any(iou(g, r['pred'][k]['box']) >= 0.5 for k in kept):
            n += 1
    return n


def eval_rules(rows):
    base_cov = {id(r): covered(r, range(len(r['pred']))) for r in rows}
    tot_gt = sum(len(r['gt']) for r in rows)
    base_total = sum(base_cov.values())

    def giant_pair(r, k, q):
        a, b = r['pred'][k]['box'], r['pred'][q]['box']
        return (area(a) / max(1.0, r['img_area']) > 0.5) and (area(b) / max(1.0, r['img_area']) > 0.5)

    def small_inside(r, k, q):
        a = r['pred'][k]['box']
        return area(a) / max(1.0, r['img_area']) < 0.20

    cands = [
        ('containment>=0.90 keep-area', 'cont', 0.90, 'area', None),
        ('containment>=0.90 keep-conf', 'cont', 0.90, 'conf', None),
        ('containment>=0.80 keep-area', 'cont', 0.80, 'area', None),
        ('containment>=0.80 keep-conf', 'cont', 0.80, 'conf', None),
        ('containment>=0.70 keep-area', 'cont', 0.70, 'area', None),
        ('containment>=0.70 keep-conf', 'cont', 0.70, 'conf', None),
        ('IoU>=0.90 keep-area', 'iou', 0.90, 'area', None),
        ('IoU>=0.80 keep-area', 'iou', 0.80, 'area', None),
        ('IoU>=0.80 keep-conf', 'iou', 0.80, 'conf', None),
        ('containment>=0.70 仅删小框(<20%图) keep-area', 'cont', 0.70, 'area', small_inside),
        ('巨型对专用 IoU>=0.50 keep-conf', 'iou', 0.50, 'conf', giant_pair),
        ('巨型对专用 containment>=0.60 keep-conf', 'cont', 0.60, 'conf', giant_pair),
    ]
    out = []
    for name, mode, thr, prio, cond in cands:
        deleted = spurious = lost = 0
        cov_after = 0
        for r in rows:
            P = r['pred']
            kept = kept_by_rule(r, mode, thr, prio, cond)
            for k in range(len(P)):
                if k in kept:
                    continue
                deleted += 1
                if best_iou_boxes(P[k]['box'], r['gt']) < 0.3:
                    spurious += 1
            c = covered(r, kept)
            cov_after += c
            lost += base_cov[id(r)] - c
        out.append({'rule': name, 'deleted': deleted, 'spurious_deleted': spurious,
                    'harmful_gt_loss': lost, 'cover_before': base_total, 'cover_after': cov_after,
                    'cover_pct': round(100.0 * cov_after / max(1, tot_gt), 1)})
    out.sort(key=lambda d: (d['harmful_gt_loss'], -d['spurious_deleted'], -d['cover_after']))
    return out, base_total, tot_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(BASE, 'reports', 'm2a-verifier', 'person_box_quality.txt'))
    a = ap.parse_args()
    gt_map, pred_map = load_gt(), load_preds()
    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    say('# person 框质量 + 巨型重复框（纯 CPU）')
    say('')
    for conf in (CONF_MAIN, CONF_LO):
        rows = collect(gt_map, pred_map, conf)
        q = quality(rows)
        g = giant(rows)
        tot, caught, missed = dup_pairs(rows)
        say('## conf ≥ %.2f（%d 张图有 person 真值，GT %d 框，pred %d 框）'
            % (conf, len(rows), q['n_gt'], sum(len(r['pred']) for r in rows)))
        say('1) 框质量：最佳匹配 IoU 中位 **%.3f**；IoU<0.5 **%.1f%%**；IoU<0.7 **%.1f%%**；'
            '面积比 pred/GT 中位 **%.2f**（P10 %.2f / P90 %.2f）；中心偏移中位 **%.2f%%** 对角线'
            % (q['iou_median'], q['iou_lt50'], q['iou_lt70'], q['ratio_median'],
               q['ratio_p10'], q['ratio_p90'], q['off_median_pct_diag']))
        say('2) 巨型框（面积/图面积>%.1f）：**%d/%d 个 pred（%.1f%%）**，出现在 **%d/%d 张图（%.1f%%）**；'
            '这些框与真值最佳 IoU 中位 **%.3f**（<0.5 占 %.1f%%），conf 中位 %.3f'
            % (GIANT, g['n_giant'], g['n_pred'], g['share_pred'], g['imgs'], g['n_imgs'],
               g['share_imgs'], g['iou_median'], g['iou_lt50'], g['conf_median']))
        say('3) 重复框（IoU≥0.8 或 containment≥0.7）：**%d 对**；现役 `--drop-nested %.2f` 拦掉 **%d 对**，'
            '**漏掉 %d 对（%.1f%%）**' % (tot, NEST_CUR, caught, len(missed),
                                        100.0 * len(missed) / max(1, tot)))
        for d in missed[:6]:
            say('   - `%s` 框%s IoU %.2f / containment %.2f，面积占图 %s%%，conf %s'
                % (d['stem'][:32], d['pair'], d['iou'], d['cont'], d['area_pct'], d['conf']))
        rk, base_total, tot_gt = eval_rules(rows)
        say('4) 无去重基线：覆盖 %d/%d 真值（%.1f%%）'
            % (base_total, tot_gt, 100.0 * base_total / max(1, tot_gt)))
        say('')
        say('   | 规则 | 删框 | 其中垃圾(与真值IoU<0.3) | 真值被误删(失去唯一覆盖) | 留下覆盖 |')
        say('   |---|---|---|---|---|')
        for d in rk:
            say('   | %s | %d | %d | **%d** | %d/%d（%.1f%%） |'
                % (d['rule'], d['deleted'], d['spurious_deleted'], d['harmful_gt_loss'],
                   d['cover_after'], tot_gt, d['cover_pct']))
        best = rk[0]
        say('')
        say('→ 推荐 **%s**：误删真值 %d 个、删掉垃圾 %d 个、覆盖 %.1f%%（基线 %.1f%%）'
            % (best['rule'], best['harmful_gt_loss'], best['spurious_deleted'], best['cover_pct'],
               100.0 * base_total / max(1, tot_gt)))
        say('')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with io.open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
