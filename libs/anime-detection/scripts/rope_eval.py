#!/usr/bin/env python
"""绳子 mask 评测：预测 vs 人标真值，给出人工成本 KPI。

  python scripts/rope_eval.py --pred <mask 目录或图> --gt data/masks/rope/raw_ps

口径（对外接口要求）：
  * 只在人标过的像素上算（未标注=忽略）；正样本=raw_ps/<name>_mask.png，负样本=<name>_labels.png 里 ==2 的部分。
  * P/R/F1 + **误检连通块数**（落在非绳区、面积≥--min-block 的块）+ **漏检段数**（人标绳区里没被覆盖、
    面积≥--min-block 的块）；两者相加＝**预计需要几笔**（主 KPI）。
  * 难负类单列：人标非绳里"基线红规则也会命中"的像素（潮红/勒痕/红衣物/皮肤/金属反光的集中地）。
"""
import argparse
import glob
import io
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def blocks(mask, min_area):
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def main():
    ap = argparse.ArgumentParser(description='绳子 mask 评测（P/R/F1 + 人工成本 KPI）')
    ap.add_argument('--pred', required=True, help='预测 mask 目录（找 <name>_mask.png）或单张图')
    ap.add_argument('--gt', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps'))
    ap.add_argument('--img-dir', default=os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work', '_orig'))
    ap.add_argument('--min-block', type=int, default=64)
    ap.add_argument('--json', default='')
    a = ap.parse_args()

    rows = []
    for lp in sorted(glob.glob(os.path.join(a.gt, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(a.gt, name + '_mask.png'), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        pp = a.pred
        if os.path.isdir(pp):
            cand = glob.glob(os.path.join(pp, name + '*_mask.png')) or glob.glob(os.path.join(pp, name + '*.png'))
            if not cand:
                print('缺预测:', name)
                continue
            pp = cand[0]
        pred = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print('预测读不到:', pp)
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred = pred > 127
        rope, other = gt > 127, lab == 2
        tp = int((pred & rope).sum())
        fp = int((pred & other).sum())
        fn = int((~pred & rope).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        fp_blocks = blocks(pred & other, a.min_block)
        miss_blocks = blocks(rope & ~pred, a.min_block)
        # 难负类：基线红规则命中的非绳像素
        im = None
        for e in ('.png', '.jpg', '.jpeg'):
            q = os.path.join(a.img_dir, name + e)
            if os.path.exists(q):
                im = cv2.imread(q)
                break
        hard = {'n': 0, 'fp': 0}
        if im is not None:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            H = hsv[..., 0].astype(np.int32)
            hh = np.minimum(np.abs(H), 180 - np.abs(H))
            base = (hh <= 14) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 60) & other
            hard = {'n': int(base.sum()), 'fp': int((base & pred).sum())}
        if rope.sum() == 0:
            # 全负样本图（用户确认这张没有绳子）：F1 无定义 → 只报误检量，且不进 F1 汇总
            rows.append({'image': name, 'pred': os.path.relpath(pp, ROOT), 'neg_only': True,
                         'P': None, 'R': None, 'F1': None, 'TP': 0, 'FP': fp, 'FN': 0,
                         'FP_blocks': len(fp_blocks), 'FP_block_px': int(sum(fp_blocks)),
                         'miss_segments': 0, 'miss_px': 0, 'est_strokes': len(fp_blocks),
                         'hard_neg_px': 0, 'hard_neg_FP': 0, 'hard_neg_FP_rate': 0.0})
            print('[%s] 全负样本图：误检 %d px / %d 块 —— 不计入 F1 汇总' % (name, fp, len(fp_blocks)))
            continue
        rows.append({'image': name, 'pred': os.path.relpath(pp, ROOT), 'P': round(p, 4), 'R': round(r, 4),
                     'F1': round(2 * p * r / max(p + r, 1e-9), 4), 'TP': tp, 'FP': fp, 'FN': fn,
                     'FP_blocks': len(fp_blocks), 'FP_block_px': int(sum(fp_blocks)),
                     'miss_segments': len(miss_blocks), 'miss_px': int(sum(miss_blocks)),
                     'est_strokes': len(fp_blocks) + len(miss_blocks),
                     'hard_neg_px': hard['n'], 'hard_neg_FP': hard['fp'],
                     'hard_neg_FP_rate': round(hard['fp'] / max(1, hard['n']), 4)})
        x = rows[-1]
        print('[%s] P %.3f R %.3f F1 %.3f | 误检块 %3d(%6dpx) 漏检段 %3d(%6dpx) 预计笔数 %3d | 难负类 %6dpx 误检率 %.3f'
              % (name, x['P'], x['R'], x['F1'], x['FP_blocks'], x['FP_block_px'], x['miss_segments'],
                 x['miss_px'], x['est_strokes'], x['hard_neg_px'], x['hard_neg_FP_rate']))
    if rows:
        pos_rows = [x for x in rows if not x.get('neg_only')]
        tp = sum(x['TP'] for x in pos_rows); fp = sum(x['FP'] for x in pos_rows); fn = sum(x['FN'] for x in pos_rows)
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print('汇总: P %.4f R %.4f F1 %.4f | 预计笔数 %d（误检块 %d + 漏检段 %d）'
              % (P, R, 2 * P * R / max(P + R, 1e-9), sum(x['est_strokes'] for x in rows),
                 sum(x['FP_blocks'] for x in rows), sum(x['miss_segments'] for x in rows)))
    if a.json:
        with io.open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
