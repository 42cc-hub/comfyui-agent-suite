# -*- coding: utf-8 -*-
"""人框补标队列：只放 person 行，`prop` 填用户当前已标的人框（只补缺、不重画）。

入选判据（lead 2026-10-21 定）：当前 GT 人框数 < max(pred_person 去嵌套后, WD14 人数代理)。
两个数都取自建队列时写进 `note` 的 `multi<N>;pred_person=<P>;nest=<K>`（N=WD14 代理人数，K=被 ≥0.90
包含度删掉的嵌套重复框数 ⇒ 去嵌套后 = P − K）。

不覆盖原 `index.csv`；写前把当轮 `index.csv` 快照到 `versions/`（lead 指定的时间戳文件名），
队列另存 `index_person_topup.csv`，补完由 lead 合并。
"""
import argparse
import csv
import io
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
GT = os.path.join(SUITE, 'data', 'hand_gt_multi')
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf', 'wd14',
          'verdict', 'boxes', 'edited', 'note']


def n_boxes(s):
    return len([b for b in (s or '').split(';') if b.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=GT)
    ap.add_argument('--snapshot', default='index_20261021_0522.csv')
    ap.add_argument('--out', default='index_person_topup.csv')
    a = ap.parse_args()

    src = os.path.join(a.gt, 'index.csv')
    rows = list(csv.DictReader(io.open(src, encoding='utf-8-sig')))
    persons = [r for r in rows if r['target'] == 'person']

    picked = []
    for r in persons:
        m = re.search(r'multi(\d+);pred_person=(\d+);nest=(\d+)', r['note'])
        if not m:
            raise SystemExit('note 解析失败: %s' % r['source'])
        wd, pred, nest = int(m.group(1)), int(m.group(2)), int(m.group(3))
        kept = pred - nest
        gt_n = 0 if r['verdict'] == 'neg' else n_boxes(r['boxes'])
        need = max(kept, wd)
        if gt_n < need:
            picked.append({'row': r, 'gt_n': gt_n, 'need': need, 'pred': pred, 'nest': nest,
                           'kept': kept, 'wd': wd, 'gap': need - gt_n})

    picked.sort(key=lambda x: (-x['gap'], -x['wd'], x['row']['source']))
    idx0 = max(int(r['idx']) for r in rows) + 1
    out_rows = []
    for i, p in enumerate(picked):
        r = p['row']
        keep = ';'.join(b for b in (r['boxes'] or '').split(';') if b.strip())
        out_rows.append({
            'idx': idx0 + i,
            'crop': r['crop'],
            'source': r['source'],
            'head': r['head'] or '0',
            'target': 'person',
            'prop': keep,
            'prop_src': 'gt_current',
            'prop_conf': '',
            'wd14': '',
            'verdict': '',
            'boxes': '',
            'edited': '',
            'note': 'topup;gt_n=%d;need=%d;pred_kept=%d;pred_n=%d;nest=%d;wd14_n=%d' % (
                p['gt_n'], p['need'], p['kept'], p['pred'], p['nest'], p['wd']),
        })

    vdir = os.path.join(a.gt, 'versions')
    os.makedirs(vdir, exist_ok=True)
    snap = os.path.join(vdir, a.snapshot)
    if not os.path.exists(snap):
        shutil.copyfile(src, snap)
    dst = os.path.join(a.gt, a.out)
    with io.open(dst, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print('队列 %s：%d 张（GT 人框合计 %d，目标 %d，缺 %d）' % (
        dst, len(out_rows), sum(p['gt_n'] for p in picked), sum(p['need'] for p in picked),
        sum(p['gap'] for p in picked)))
    print('快照 %s' % (snap if os.path.exists(snap) else '(已存在，跳过)'))
    from collections import Counter
    print('gap 分布', dict(sorted(Counter((p['gt_n'], p['need']) for p in picked).items())))
    print('按图看：gt==wd 但 pred_kept 更大而入选', sum(1 for p in picked if p['kept'] > p['wd']))
    miss = [r['crop'] for r in out_rows if not os.path.exists(os.path.join(a.gt, 'images', r['crop']))]
    print('crop 缺失', len(miss), miss[:3])


if __name__ == '__main__':
    main()
