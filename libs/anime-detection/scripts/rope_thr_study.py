#!/usr/bin/env python
"""工作点（阈值）选择法研究——纯离线，只用已保存的置信度图，不需要 GPU。

回答两个问题：
  1) 主 KPI 对阈值有多敏感？（画出 cost curve）
  2) "在训练图的验证块上选阈值"这套做法，和固定阈值 / 事后知道答案（oracle）差多少？
     以及它在不同 seed 下的波动有多大（阈值方差是不是实验结论的主要噪声源）？

    python scripts\\rope_thr_study.py --tags "AB_G0_*,PE7_P0_rgb"
"""
import argparse
import fnmatch
import glob
import io
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, '.runtime', 'rope_cv')
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
HOLD = ['s01', 's02', 's03', 's04', 's05']
THS = [round(x, 2) for x in np.arange(0.30, 0.985, 0.02)]
MIN_AREA = 64
FIXED = [0.5, 0.7, 0.8, 0.9]


def n_blocks(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_AREA))


def measure(gt, conf, thr):
    tp = fp = fn = blk = miss = 0
    for name in HOLD:
        rope, neg = gt[name]
        m = conf[name] >= thr
        tp += int((m & rope).sum())
        fp += int((m & neg).sum())
        fn += int((~m & rope).sum())
        blk += n_blocks(m & neg)
        miss += n_blocks(rope & ~m)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return {'P': P, 'R': R, 'F1': 2 * P * R / max(P + R, 1e-9),
            'blk': blk, 'miss': miss, 'strokes': blk + miss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', required=True, help='逗号分隔的 glob，如 "AB_G0_*"')
    ap.add_argument('--json', default=os.path.join(W, 'thr_study.json'))
    ap.add_argument('--md', default=os.path.join(W, 'thr_study.md'))
    a = ap.parse_args()

    gt = {}
    for n in HOLD:
        lab = cv2.imread(os.path.join(RAW, n + '_labels.png'), 0)
        gt[n] = (lab == 1, lab == 2)

    pats = [p.strip() for p in a.tags.split(',') if p.strip()]
    dirs = []
    for p in pats:
        dirs += sorted(d for d in glob.glob(os.path.join(W, p)) if os.path.isdir(d))
    out = {}
    lines = ['# 阈值（工作点）选择法研究', '',
             '留出 s01~s05；`val_tile` = 训练图随机验证块上选出的阈值（现方案，读自 unet_loo.json）。', '']
    for d in dirs:
        tag = os.path.basename(d)
        conf = {}
        ok = True
        for n in HOLD:
            p = os.path.join(d, n + '_conf.png')
            if not os.path.exists(p):
                ok = False
                break
            conf[n] = cv2.imread(p, 0).astype(np.float32) / 255.0
        if not ok:
            continue
        jp = os.path.join(d, 'unet_loo.json')
        vt = None
        if os.path.exists(jp):
            recs = json.load(io.open(jp, encoding='utf-8'))
            vt = [r.get('thr') for r in recs if r.get('thr') is not None]
        rows = [dict(thr=t, **measure(gt, conf, t)) for t in THS]
        best = max(rows, key=lambda r: r['F1'])
        cur = measure(gt, conf, float(np.mean(vt))) if vt else None
        row = {'tag': tag, 'val_tile_thr': vt, 'val_tile': cur, 'oracle': best,
               'fixed': {str(f): measure(gt, conf, f) for f in FIXED},
               'curve': rows}
        out[tag] = row
        lines += ['## %s' % tag, '',
                  '- 现方案阈值（每张图）: %s' % vt,
                  '- 现方案: F1 %.4f, 召回 %.3f, 笔数 %d' % (cur['F1'], cur['R'], cur['strokes']) if cur else '- 现方案: n/a',
                  '- oracle: thr %.2f, F1 %.4f, 笔数 %d（F1 上限 +%.4f，笔数下限 %+d）'
                  % (best['thr'], best['F1'], best['strokes'], best['F1'] - (cur['F1'] if cur else 0),
                     best['strokes'] - (cur['strokes'] if cur else 0)),
                  '- 固定阈值: ' + ', '.join('%.2f→F1 %.4f/笔数 %d' % (f, row['fixed'][str(f)]['F1'],
                                                                        row['fixed'][str(f)]['strokes'])
                                             for f in FIXED), '']
    # 汇总：同一实验的不同 seed
    lines += ['## 汇总（同一配置不同 seed 的阈值波动）', '',
              '| 组 | val_tile 阈值 | 阈值极差 | F1 极差 | 笔数极差 | oracle F1 上限 |', '|---|---|---|---|---|---|']
    groups = {}
    for tag, r in out.items():
        g = tag.rsplit('_s', 1)[0]
        groups.setdefault(g, []).append(r)
    for g, rs in sorted(groups.items()):
        ths = [t for r in rs for t in (r['val_tile_thr'] or [])]
        f1 = [r['val_tile']['F1'] for r in rs if r['val_tile']]
        st = [r['val_tile']['strokes'] for r in rs if r['val_tile']]
        orc = [r['oracle']['F1'] for r in rs]
        lines.append('| %s | %s | %.2f | %.4f | %d | %.4f |'
                     % (g, '/'.join('%.2f' % t for t in ths),
                        (max(ths) - min(ths)) if ths else 0,
                        (max(f1) - min(f1)) if f1 else 0,
                        (max(st) - min(st)) if st else 0, max(orc) if orc else 0))
    with io.open(a.json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    with io.open(a.md, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
