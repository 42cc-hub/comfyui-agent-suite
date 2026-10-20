# -*- coding: utf-8 -*-
"""task-7：多人交互标注队列（M1 骨骼队列共用同一份图清单）。

四段（按顺序跑；`--stage all` 一次跑完）：
  rank    用 det_v2（conf 0.05 / imgsz 960）粗扫 376 张可用图，算"手框尺寸 + 交互代理"，缓存 preds/rank_preds.json
  select  取 22 张三人（按代理强度）+ 44 张双人（36 张按代理 + 8 张"手小"名额），写
          `image_list.txt`（**冻结清单**，列 stem/path/total/proxy_score/proxy_reason/hand_small）+ selection.csv
  dump    对选中的 66 张按**产线口径**跑 det_v2 ∪ hand_v3（conf 0.01 / imgsz 1280），缓存 preds/queue_preds.json
  build   写 `index.csv`（hand/foot/person 各一行，预填 conf≥0.10）、`dropped_nested.csv`、`images/` 硬链接

口径与理由：
  * 人数 = WD14 人数代理（每个性别族取概率最高的计数标签；**不是交互度真值**），默认不做阈值（`--count-th 0` = 与 lead 的 311/59 对齐）
  * 交互代理 proxy_score = 0.5×两人手框中心距<8%对角线的对数 + 0.3×手框跨两人 person 框 + 0.2×两 person 框重叠
    —— 代的是"两人挨得近、手互相碰"，**不是**真实交互标注；进 selection.csv 供复核
  * hand_small = det_v2 手框长边中位数 < 72px（**按 rank 阶段 960 口径判**，只用于名额分配）
  * person 预填做包含度去重：贪心保留面积大/置信高的框，删掉被 ≥0.90 包含的框，删除逐条记 `dropped_nested.csv`
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
DATA = os.path.join(SUITE, 'data')
OUT = os.path.join(DATA, 'hand_gt_multi')
HAND_GT = os.path.join(DATA, 'hand_gt', 'index.csv')
W_DET_V2 = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
W_HAND_V3 = os.path.join(SUITE, 'runs', 'm2a_hand_v3', 'weights', 'best.pt')
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
          'wd14', 'verdict', 'boxes', 'edited', 'note']
DROP_FIELDS = ['image', 'stem', 'target', 'keep_idx', 'drop_idx', 'containment', 'keep_area', 'drop_area']
PREF_CONF = 0.10
NEST_THR = 0.90
SMALL_PX = 72.0


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(inner, outer):
    a = area(inner)
    if a <= 0:
        return 0.0
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    return (max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)) / a


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def long_edge(b):
    return max(b[2] - b[0], b[3] - b[1])


def dedupe_conf(preds, thr=0.5):
    keep = []
    for p in sorted(preds, key=lambda z: -z['conf']):
        if any(iou(p['box'], k['box']) >= thr for k in keep):
            continue
        keep.append(p)
    return keep


def drop_nested(boxes, thr=NEST_THR):
    order = sorted(range(len(boxes)), key=lambda i: -area(boxes[i]['box']))
    kept, drops = [], []
    for i in order:
        b = boxes[i]['box']
        host = next((k for k in kept if containment(b, k['box']) >= thr), None)
        if host is None:
            kept.append(boxes[i])
        else:
            drops.append({'keep': host, 'drop': boxes[i], 'containment': round(containment(b, host['box']), 4)})
    kept.sort(key=lambda z: -z['conf'])
    return kept, drops


def load_pool(count_th, include_candidates=False):
    labeled = set()
    if os.path.exists(HAND_GT):
        with io.open(HAND_GT, encoding='utf-8-sig') as fh:
            labeled = {r['crop'] for r in csv.DictReader(fh)}
    pool = []
    for name in (('tags_delivered.csv',) if not include_candidates else ('tags_delivered.csv', 'tags_candidates.csv')):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['labeled_in_hand_gt'] == '1' or r['no_humans']:
                    continue
                g = int(r['girls']) if float(r['girls_p']) >= count_th else 0
                b = int(r['boys']) if float(r['boys_p']) >= count_th else 0
                tot = g + b
                if tot not in (2, 3):
                    continue
                r['n_person'] = tot
                r['hand_gt_labeled'] = int(os.path.basename(r['path']) in labeled)
                if r['hand_gt_labeled']:
                    continue
                pool.append(r)
    return pool


def features(preds, files, near_frac):
    out = []
    for r in files:
        rec = preds.get(r['sha256'])
        if not rec:
            continue
        W, H = float(rec['w']), float(rec['h'])
        diag = (W * W + H * H) ** 0.5
        persons = [p for p in rec['det'].get('person', []) if p['conf'] >= PREF_CONF]
        hands = [p for p in rec['det'].get('hand', []) if p['conf'] >= PREF_CONF]
        hands += [p for p in rec.get('hand', {}).get('hand', []) if p['conf'] >= PREF_CONF]
        feet = [p for p in rec['det'].get('foot', []) if p['conf'] >= PREF_CONF]
        hands, feet = dedupe_conf(hands), dedupe_conf(feet)
        kept_p, drops_p = drop_nested(persons)
        near = sum(1 for i in range(len(hands)) for j in range(i + 1, len(hands))
                   if ((center(hands[i]['box'])[0] - center(hands[j]['box'])[0]) ** 2 +
                       (center(hands[i]['box'])[1] - center(hands[j]['box'])[1]) ** 2) ** 0.5 < near_frac * diag)
        cross = sum(1 for h in hands if sum(1 for k in kept_p if containment(h['box'], k['box']) >= 0.5) >= 2)
        pov = max([iou(kept_p[i]['box'], kept_p[j]['box'])
                   for i in range(len(kept_p)) for j in range(i + 1, len(kept_p))] or [0.0])
        hand_med = float(np.median([long_edge(h['box']) for h in hands])) if hands else None
        out.append({'r': r, 'persons': kept_p, 'drops': drops_p, 'hands': hands, 'feet': feet,
                    'near': near, 'cross': cross, 'pov': round(pov, 4), 'hand_med': hand_med,
                    'n_person_pred': len(kept_p), 'nested': int(bool(drops_p)),
                    'mismatch': int(len(kept_p) != r['n_person']),
                    'proxy': round(0.5 * min(1.0, near / 2.0) + 0.3 * min(1.0, cross / 2.0)
                                   + 0.2 * min(1.0, pov / 0.2), 4)})
    return out


def reason(f):
    return 'near=%d;cross=%d;pov=%.2f;hand_med=%s;pred_person=%d/%d' % (
        f['near'], f['cross'], f['pov'],
        '-' if f['hand_med'] is None else int(round(f['hand_med'])), f['n_person_pred'], f['r']['n_person'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['rank', 'select', 'dump', 'build', 'all'], default='all')
    ap.add_argument('--count-th', type=float, default=0.0)
    ap.add_argument('--n-two', type=int, default=44)
    ap.add_argument('--n-three', type=int, default=22)
    ap.add_argument('--n-small', type=int, default=8)
    ap.add_argument('--small-px', type=float, default=SMALL_PX)
    ap.add_argument('--near-frac', type=float, default=0.08)
    ap.add_argument('--rank-imgsz', type=int, default=960)
    ap.add_argument('--rank-conf', type=float, default=0.05)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf-dump', type=float, default=0.01)
    ap.add_argument('--include-candidates', action='store_true', help='把 candidates 原始渲染池也算进候选（默认只用品线交付图）')
    a = ap.parse_args()

    for d in (OUT, os.path.join(OUT, 'preds'), os.path.join(OUT, 'images')):
        os.makedirs(d, exist_ok=True)
    pool = load_pool(a.count_th, a.include_candidates)
    print('可用图 %d（2 人 %d / 3 人 %d）' % (
        len(pool), sum(1 for r in pool if r['n_person'] == 2), sum(1 for r in pool if r['n_person'] == 3)))

    # ---------- rank ----------
    rank_path = os.path.join(OUT, 'preds', 'rank_preds.json')
    if a.stage in ('rank', 'all'):
        from ultralytics import YOLO
        m = YOLO(W_DET_V2)
        cache = {}
        todo = sorted(pool, key=lambda r: (r['project'], r['stem']))
        for i, r in enumerate(todo, 1):
            if r['sha256'] in cache:
                continue
            res = m.predict(r['path'], conf=a.rank_conf, imgsz=a.rank_imgsz, verbose=False, device=0)[0]
            names = {int(k): v for k, v in res.names.items()}
            rec = {'w': r['w'], 'h': r['h'], 'det': {}}
            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                rec['det'].setdefault(names[int(c)], []).append(
                    {'box': [round(x, 1) for x in b], 'conf': round(float(s), 4)})
            cache[r['sha256']] = rec
            if i % 25 == 0:
                print('  rank %d/%d' % (i, len(todo)), flush=True)
        with io.open(rank_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
        print('wrote', rank_path, len(cache))
    if a.stage == 'rank':
        return

    # ---------- select ----------
    list_path = os.path.join(OUT, 'image_list.txt')
    if a.stage in ('select', 'all'):
        cache = json.load(io.open(rank_path, encoding='utf-8'))
        feats = features(cache, pool, a.near_frac)
        print('rank 特征 %d 张' % len(feats))
        threes = sorted([f for f in feats if f['r']['n_person'] == 3], key=lambda f: -f['proxy'])
        twos = sorted([f for f in feats if f['r']['n_person'] == 2], key=lambda f: -f['proxy'])
        pick3 = threes[:a.n_three]
        # 双人：先保 8 个"手小"名额（按手框长边升序），再按代理强度补满
        small = sorted([f for f in twos if f['hand_med'] is not None and f['hand_med'] < a.small_px],
                       key=lambda f: f['hand_med'])[:a.n_small]
        small_sha = {f['r']['sha256'] for f in small}
        rest = [f for f in twos if f['r']['sha256'] not in small_sha][:max(0, a.n_two - len(small))]
        pick2 = small + rest
        picked = sorted(pick3 + pick2, key=lambda f: (f['r']['project'], f['r']['stem']))
        with io.open(list_path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('stem\tpath\ttotal\tproxy_score\tproxy_reason\thand_small\n')
            for f in picked:
                fh.write('%s\t%s\t%d\t%.4f\t%s\t%d\n' % (
                    f['r']['stem'], f['r']['path'], f['r']['n_person'], f['proxy'], reason(f),
                    int(f['hand_med'] is not None and f['hand_med'] < a.small_px)))
        with io.open(os.path.join(OUT, 'selection.csv'), 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['stem', 'project', 'path', 'total', 'n_person_pred', 'mismatch', 'nested',
                        'n_hand', 'n_foot', 'hand_med', 'near', 'cross', 'pov', 'proxy_score', 'hand_small'])
            for f in picked:
                w.writerow([f['r']['stem'], f['r']['project'], f['r']['path'], f['r']['n_person'],
                            f['n_person_pred'], f['mismatch'], f['nested'], len(f['hands']), len(f['feet']),
                            '' if f['hand_med'] is None else round(f['hand_med'], 1),
                            f['near'], f['cross'], f['pov'], f['proxy'],
                            int(f['hand_med'] is not None and f['hand_med'] < a.small_px)])
        print('=== 选图 %d：三人 %d / 双人 %d（目标 %d/%d）；手小 %d 张' % (
            len(picked), len(pick3), len(pick2), a.n_three, a.n_two,
            sum(1 for f in picked if f['hand_med'] is not None and f['hand_med'] < a.small_px)))
        print('清单（冻结）', list_path)
    if a.stage == 'select':
        return

    # ---------- dump（产线口径） ----------
    dump_path = os.path.join(OUT, 'preds', 'queue_preds.json')
    picked_rows = []
    with io.open(list_path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                picked_rows.append({'stem': parts[0], 'path': parts[1], 'n_person': int(parts[2])})
    if a.stage in ('dump', 'all'):
        from ultralytics import YOLO
        m_det, m_hand = YOLO(W_DET_V2), YOLO(W_HAND_V3)
        cache = {}
        for i, row in enumerate(picked_rows, 1):
            rec = {'w': None, 'h': None, 'det': {}, 'hand': {}}
            for tag, model in (('det', m_det), ('hand', m_hand)):
                res = model.predict(row['path'], conf=a.conf_dump, imgsz=a.imgsz, verbose=False, device=0)[0]
                names = {int(k): v for k, v in res.names.items()}
                if tag == 'det':
                    rec['h'], rec['w'] = res.orig_shape
                for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    rec[tag].setdefault(names[int(c)], []).append(
                        {'box': [round(x, 1) for x in b], 'conf': round(float(s), 4)})
            cache[row['stem']] = rec
            if i % 10 == 0:
                print('  dump %d/%d' % (i, len(picked_rows)), flush=True)
        with io.open(dump_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
        print('wrote', dump_path, len(cache))
    if a.stage == 'dump':
        return

    # ---------- build ----------
    cache = json.load(io.open(dump_path, encoding='utf-8'))
    used, rows, drops, stats = set(), [], [], []
    img_dir = os.path.join(OUT, 'images')
    for row in picked_rows:
        rec = cache[row['stem']]
        persons = [p for p in rec['det'].get('person', []) if p['conf'] >= PREF_CONF]
        hands = [p for p in rec['det'].get('hand', []) if p['conf'] >= PREF_CONF]
        hands += [p for p in rec['hand'].get('hand', []) if p['conf'] >= PREF_CONF]
        feet = [p for p in rec['det'].get('foot', []) if p['conf'] >= PREF_CONF]
        hands, feet = dedupe_conf(hands), dedupe_conf(feet)
        kept_p, drops_p = drop_nested(persons)

        crop = os.path.basename(row['path'])
        base, ext = os.path.splitext(crop)
        if base in used:
            import hashlib
            crop = '%s__%s%s' % (base, hashlib.sha256(row['path'].encode('utf-8')).hexdigest()[:8], ext)
        used.add(os.path.splitext(crop)[0])
        dst = os.path.join(img_dir, crop)
        if not os.path.exists(dst):
            try:
                os.link(row['path'], dst)
            except OSError:
                import shutil
                shutil.copy2(row['path'], dst)

        note = 'multi%d;pred_person=%d;nest=%d;hand=%d;foot=%d' % (
            row['n_person'], len(kept_p), len(drops_p), len(hands), len(feet))
        stats.append({'stem': row['stem'], 'total': row['n_person'], 'pred_person': len(kept_p),
                      'nested': len(drops_p), 'mismatch': int(len(kept_p) != row['n_person'])})

        def mk(boxes, src):
            if not boxes:
                return {'prop': '', 'prop_src': 'none', 'prop_conf': ''}
            return {'prop': ';'.join('%d,%d,%d,%d' % tuple(int(round(x)) for x in b['box']) for b in boxes),
                    'prop_src': src, 'prop_conf': '%.2f' % max(b['conf'] for b in boxes)}

        for target, pack in (('hand', mk(hands, 'model_union')), ('foot', mk(feet, 'det_v2')),
                             ('person', mk(kept_p, 'det_v2'))):
            rows.append({'idx': len(rows) + 1, 'crop': crop, 'source': row['stem'], 'head': '0', 'target': target,
                         'prop': pack['prop'], 'prop_src': pack['prop_src'], 'prop_conf': pack['prop_conf'],
                         'wd14': '', 'verdict': '', 'boxes': '', 'edited': '', 'note': note})
        for d in drops_p:
            drops.append({'image': crop, 'stem': row['stem'], 'target': 'person',
                          'keep_idx': ','.join('%.0f' % v for v in d['keep']['box']),
                          'drop_idx': ','.join('%.0f' % v for v in d['drop']['box']),
                          'containment': d['containment'],
                          'keep_area': int(area(d['keep']['box'])), 'drop_area': int(area(d['drop']['box']))})

    with io.open(os.path.join(OUT, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with io.open(os.path.join(OUT, 'dropped_nested.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=DROP_FIELDS)
        w.writeheader()
        for d in drops:
            w.writerow(d)
    print('图 %d 张 → 队列行 %d ／嵌套重复人框：图 %d 张、删除 %d 条／person 框数≠人数：%d 张' % (
        len(picked_rows), len(rows), sum(1 for s in stats if s['nested']), len(drops),
        sum(1 for s in stats if s['mismatch'])))
    print('按人数：2 人 %d（数对不上 %d）/ 3 人 %d（数对不上 %d）' % (
        sum(1 for s in stats if s['total'] == 2), sum(1 for s in stats if s['total'] == 2 and s['mismatch']),
        sum(1 for s in stats if s['total'] == 3), sum(1 for s in stats if s['total'] == 3 and s['mismatch'])))
    print('队列', os.path.join(OUT, 'index.csv'))
    print('删除记录', os.path.join(OUT, 'dropped_nested.csv'))
    print('--pre 目录', img_dir)


if __name__ == '__main__':
    main()
