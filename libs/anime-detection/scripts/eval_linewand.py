r""""超级魔棒"验收：把（学出来的）线稿当屏障做 **flooding**，看能不能选出人工标注的那一块。

做法（正是用户描述的那条路）：线稿 → 阈值 → 可选膨胀成"隔墙" → 在**原图**上从块内一点
`cv2.floodFill`（固定色差容差 + 4 邻域，mask=墙）⇒ 得到一块。和人工块比 IoU。

同时回答"**线稿闭合性没法保证**"这件事——用数据回答，而不是靠感觉：
  * 把墙**膨胀 K px**（K=0/1/2/3）扫一遍，看漏检（leak）怎么随 K 下降；
  * 与"没有线的纯魔棒"（K 无意义、walls 为空）对比，看线稿到底贡献了多少。
这两条一起给出可用的工作点：**不闭合也能用，缺的那几段由"膨胀 + 色差容差"兜住**。

用法：python scripts/eval_linewand.py [--name lw_v1] [--thr 0.5] [--min-block 400]
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
DS = os.path.join(DATA, 'colorblock_ds')
HOLDOUT = ('ink20260928', 'odette20260928')


def grow(img, walls, seed, tol):
    """在 img 上从 seed 做固定色差容差的 flooding，walls 非零处不可穿越。"""
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    mask[1:-1, 1:-1] = (walls > 0).astype(np.uint8)
    im = img.copy()
    cv2.floodFill(im, mask, seed, (0, 0, 0), (tol,) * 3, (tol,) * 3,
                  cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 4)
    return mask[1:-1, 1:-1] > 0


def interior_seed(block):
    d = cv2.distanceTransform(block.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='lw_v1')
    ap.add_argument('--dir', default=DS)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--min-block', type=int, default=400)
    ap.add_argument('--tols', nargs='+', type=int, default=[12, 24, 40])
    ap.add_argument('--dilates', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--holdout', nargs='*', default=list(HOLDOUT))
    a = ap.parse_args()
    pred_dir = os.path.join(DATA, 'colorblock_linewand', a.name, 'pred')
    rows = []
    for stem in sorted(set(os.path.splitext(os.path.basename(p))[0].replace('_prob', '')
                           for p in glob.glob(os.path.join(pred_dir, '*_prob.png')))):
        if not any(stem.startswith(h) for h in a.holdout):
            continue
        op = os.path.join(a.dir, 'originals', stem + '.png')
        bp = os.path.join(a.dir, 'targets', stem + '_blocks.png')
        pp = os.path.join(pred_dir, stem + '_prob.png')
        if not (os.path.exists(op) and os.path.exists(bp) and os.path.exists(pp)):
            continue
        img = cv2.imread(op, cv2.IMREAD_COLOR)
        blocks = cv2.imread(bp, cv2.IMREAD_UNCHANGED)
        prob = cv2.imread(pp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        line = (prob > a.thr).astype(np.uint8)
        n_blk = 0
        for bid in np.unique(blocks):
            if bid == 0:
                continue
            blk = blocks == bid
            if blk.sum() < a.min_block:
                continue
            n_blk += 1
            seed = interior_seed(blk)
            for tol in a.tols:
                for k in a.dilates:
                    if k > 0:
                        w = cv2.dilate(line, np.ones((2 * k + 1,) * 2, np.uint8))
                    else:
                        w = line
                    g = grow(img, w, seed, tol)
                    inter = float((g & blk).sum())
                    uni = float((g | blk).sum())
                    iou = inter / uni if uni else 0.0
                    other = float((g & ~blk).sum()) / max(1.0, g.sum())     # 漏到别的块的比例
                    rows.append((stem, int(bid), int(blk.sum()), tol, k, iou, other))
        # 纯魔棒基线（没有线稿）：tol 扫描
        for tol in a.tols:
            for bid in np.unique(blocks):
                if bid == 0:
                    continue
                blk = blocks == bid
                if blk.sum() < a.min_block:
                    continue
                seed = interior_seed(blk)
                g = grow(img, np.zeros_like(line), seed, tol)
                iou = float((g & blk).sum()) / max(1.0, float((g | blk).sum()))
                other = float((g & ~blk).sum()) / max(1.0, g.sum())
                rows.append((stem, int(bid), int(blk.sum()), tol, -1, iou, other))
    if not rows:
        raise SystemExit('没有可评的切片（确认 pred/ 里有留出集的 *_prob.png）')
    print('评估块数（含多种配置）：%d' % len(rows))
    # ---- 0) 假线诊断：预测的线有多少落在**人工块内部**（那正是把 flooding 切碎的东西）----
    for stem in sorted(set(r[0] for r in rows)):
        op = os.path.join(a.dir, 'originals', stem + '.png')
        bp = os.path.join(a.dir, 'targets', stem + '_blocks.png')
        pp = os.path.join(pred_dir, stem + '_prob.png')
        if not (os.path.exists(op) and os.path.exists(bp) and os.path.exists(pp)):
            continue
        blocks = cv2.imread(bp, cv2.IMREAD_UNCHANGED)
        prob = cv2.imread(pp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        line = (prob > a.thr).astype(np.uint8)
        inner = blocks > 0
        inner_er = cv2.erode(inner.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        frac = float((line & inner_er).sum()) / max(1, int(line.sum()))
        print('  假线诊断 %-42s 预测线里 %.0f%% 落在块内部（越少越好）' % (stem[:40], 100 * frac))
    print('\n%-4s %-4s %8s %10s %10s' % ('tol', 'K', '通过@0.9', '中位 IoU', '漏出比例中位'))
    for tol in a.tols:
        for k in list(a.dilates) + [None]:
            sel = [r for r in rows if r[3] == tol and ((r[4] == k) if k is not None else r[4] == -1)]
            if not sel:
                continue
            ious = [r[5] for r in sel]
            print('%-4d %-4s %7.0f%% %10.3f %10.3f   (n=%d)'
                  % (tol, '纯魔棒' if k is None else k, 100.0 * np.mean([i >= 0.9 for i in ious]),
                     np.median(ious), np.median([r[6] for r in sel]), len(sel)))
    # ---- 不耍赖的口径：**每张切片只用同一个容差**（"每块取最佳容差"是作弊）----
    print('\n单容差口径（一张切片内所有块共用同一 tol；报切片间平均通过率）')
    print('%-4s %-6s %12s %12s' % ('tol', '配置', '切片平均通过@0.9', '切片平均中位IoU'))
    stems = sorted(set(r[0] for r in rows))
    for tol in a.tols:
        for k in list(a.dilates) + [None]:
            per, per_iou = [], []
            for st in stems:
                sel = [r for r in rows if r[0] == st and r[3] == tol
                       and ((r[4] == k) if k is not None else r[4] == -1)]
                if not sel:
                    continue
                per.append(np.mean([r[5] >= 0.9 for r in sel]))
                per_iou.append(np.median([r[5] for r in sel]))
            if per:
                print('%-4d %-6s %11.0f%% %12.3f'
                      % (tol, '纯魔棒' if k is None else k, 100.0 * np.mean(per), np.mean(per_iou)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
