# -*- coding: utf-8 -*-
"""person-only 标注批 2：50 行补标（topup，队首）+ 120 张新图 = 170 行。

- 池：`data/hand_gt_multi/tags_delivered.csv` 里 `labeled_in_hand_gt=0`、WD14 人数 = 2 或 3、且不在已标 60 张里的图
- 选图：三人全取（供给只有 30 张）、双人按项目轮转取满 120，尽量跨项目分散
- 预填：现役 det_v2 person 框（conf ≥0.30、imgsz 960 粗扫缓存）+ 新去重规则（双方 area/图>0.5 且 IoU≥0.5 删低 conf）
- 产物：`data/hand_gt_person/{index.csv, images/}`，schema 与 `data/hand_gt_multi/index.csv` 完全一致
"""
import csv
import io
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(SUITE, 'scripts'))
from m2a_multi_queue import FIELDS, dedupe_person  # noqa: E402

DATA = os.path.join(SUITE, 'data')
SRC = os.path.join(DATA, 'hand_gt_multi')
OUT = os.path.join(DATA, 'hand_gt_person')
CONF = 0.30
N_TOTAL = 120
CAP_PER_PROJECT = 12
POOL_RANK = ['deliver', 'uploads_img', 'upload_img', 'trash', 'other']


def pool_rank(r):
    p = (r.get('pool') or '') + ' ' + (r.get('path') or '')
    for i, k in enumerate(POOL_RANK):
        if k in p:
            return i
    return len(POOL_RANK)


def main():
    rows = list(csv.DictReader(io.open(os.path.join(SRC, 'tags_delivered.csv'), encoding='utf-8-sig')))
    multi = set(os.path.splitext(r['crop'])[0]
                for r in csv.DictReader(io.open(os.path.join(SRC, 'index.csv'), encoding='utf-8-sig')))
    preds = json.load(io.open(os.path.join(SRC, 'preds', 'rank_preds.json'), encoding='utf-8'))

    # 一图一行（同 stem 多副本按 pool 优先级取一条）
    by_stem = {}
    for r in rows:
        s = r['stem']
        if s not in by_stem or pool_rank(r) < pool_rank(by_stem[s]):
            by_stem[s] = r

    three, two = [], []
    detail = re.compile(r'(__|_(face|head|feet|hand|foot)\d*$)')
    for s, r in by_stem.items():
        if r['labeled_in_hand_gt'] != '0' or s in multi:
            continue
        if detail.search(s):
            continue  # 局部特写渲染（__foot#4 / _face / _head 等）：画面里没有完整人形，标不了 person 整身框
        if r['total'] == '3':
            three.append(r)
        elif r['total'] == '2':
            two.append(r)
    three.sort(key=lambda r: (r['project'], r['sha256']))
    two.sort(key=lambda r: (r['project'], r['sha256']))
    print('候选：三人 %d 张 / 双人 %d 张（已标 60 张与单人、≥4 人已剔除）' % (len(three), len(two)))

    picks = list(three)
    # 双人：按项目轮转 + 每项目上限，跨项目分散
    buckets = defaultdict(list)
    for r in two:
        buckets[r['project']].append(r)
    order = sorted(buckets)
    used = Counter()
    while len(picks) < N_TOTAL and any(buckets[p] for p in order):
        for p in order:
            if len(picks) >= N_TOTAL:
                break
            if not buckets[p]:
                continue
            if used[p] >= CAP_PER_PROJECT:
                buckets[p].clear()
                continue
            picks.append(buckets[p].pop(0))
            used[p] += 1
    picks = picks[:N_TOTAL]
    print('选取：%d 张（三人 %d / 双人 %d），项目数 %d'
          % (len(picks), sum(1 for r in picks if r['total'] == '3'),
             sum(1 for r in picks if r['total'] == '2'), len(set(r['project'] for r in picks))))

    os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)
    for n in os.listdir(os.path.join(OUT, 'images')):
        os.remove(os.path.join(OUT, 'images', n))

    def link(src_path, crop):
        dst = os.path.join(OUT, 'images', crop)
        if os.path.exists(dst):
            return
        try:
            os.link(src_path, dst)
        except OSError:
            shutil.copyfile(src_path, dst)

    out_rows, stats = [], []
    # ① 50 行补标（预填 = 当前 GT 人框），队首
    topup = list(csv.DictReader(io.open(os.path.join(SRC, 'index_person_topup.csv'), encoding='utf-8-sig')))
    for i, r in enumerate(topup, 1):
        stem = os.path.splitext(r['crop'])[0]
        tag = by_stem.get(stem, {})
        link(os.path.join(SRC, 'images', r['crop']), r['crop'])
        note = (r['note'] or '') + ';orig_idx=%s;batch2=topup' % r['idx']
        out_rows.append({'idx': str(i), 'crop': r['crop'], 'source': r['source'], 'head': '0',
                         'target': 'person', 'prop': r['prop'], 'prop_src': 'gt_current',
                         'prop_conf': '', 'wd14': tag.get('total', ''), 'verdict': '',
                         'boxes': '', 'edited': '', 'note': note})
    # ② 120 张新图（预填 = det_v2 person @0.30 + 新去重）
    base = len(out_rows)
    for j, r in enumerate(picks, 1):
        ext = os.path.splitext(r['path'])[1] or '.png'
        crop = r['stem'] + ext
        link(r['path'], crop)
        e = preds.get(r['sha256'])
        img_area = float(r['w']) * float(r['h'])
        got = [{'box': [float(v) for v in b['box']], 'conf': float(b['conf'])}
               for b in (e['det'].get('person', []) if e else []) if float(b['conf']) >= CONF]
        kept, drops = dedupe_person(got, img_area)
        prop = ';'.join(','.join(str(int(round(v))) for v in b['box']) for b in kept)
        pconf = '%.2f' % max([b['conf'] for b in kept]) if kept else ''
        note = ('person_only;batch2;total=%s;pred_person=%d;nest=%d;wd14=%s;project=%s'
                % (r['total'], len(kept), len(drops), r['total'], r['project']))
        out_rows.append({'idx': str(base + j), 'crop': crop, 'source': r['stem'], 'head': '0',
                         'target': 'person', 'prop': prop,
                         'prop_src': 'det_v2' if kept else 'none',
                         'prop_conf': pconf, 'wd14': r['total'], 'verdict': '',
                         'boxes': '', 'edited': '', 'note': note})
        stats.append((r['project'], r['total'], len(kept), len(drops), r['stem']))

    with io.open(os.path.join(OUT, 'index.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print('\n写盘 %s（%d 行）' % (os.path.join(OUT, 'index.csv'), len(out_rows)))
    print('项目分布（120 张新图）：')
    for p, n in Counter(s[0] for s in stats).most_common():
        print('  %-40s %d' % (p, n))
    print('预填框数分布：', dict(sorted(Counter(s[2] for s in stats).items())))
    print('预填 0 框：%d 张 → %s'
          % (sum(1 for s in stats if s[2] == 0), [s[4] for s in stats if s[2] == 0]))
    print('去重删框合计 %d 个（nest 中位 %.0f）'
          % (sum(s[3] for s in stats),
             sorted(s[3] for s in stats)[len(stats) // 2] if stats else 0))
    print('topup 50 行预填框：', dict(sorted(Counter(
        len([b for b in r['prop'].split(';') if b.strip()]) for r in topup).items())))


if __name__ == '__main__':
    main()
