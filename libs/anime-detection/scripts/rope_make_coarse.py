#!/usr/bin/env python
"""为"扩散式 mask 精修器"（SegRefiner 路线 A）造训练对：真值 → 粗 mask。

**纯 CPU/磁盘，不占 GPU。**

做法（每张图生成 N 个粗 mask）：
  1) 边界扰动：随机的膨胀/腐蚀（核 3~9），模拟边界外扩/内缩；
  2) 断裂：在绳上随机抹掉几段细长条，模拟"漏检段"；
  3) 误检块：在绳附近随机加几个块（椭圆/矩形），模拟"误检连通块"；
  4) 盐噪：少量绳像素随机掉点。
每次扰动的强度随机，最终 IoU 会被限制在 [--iou-min, --iou-max] 区间内（默认 0.45~0.92），
太干净学不到东西、太烂学不到正确解。

切分（**按源图切，绝不同图跨界**）：
  * 精修器的 train / val 来自 79 张训练图（全局留出 s01~s05 与探针 x05 不参与，永远当最终评测）；
  * val 取排序后每隔 8 张挑一张（10 张），train = 其余 69 张；
  * split.json 里写明名单，任何人可复核。

输出：
  data/masks/rope/coarse/train/<name>_c0.png ... （uint8 0/255）
  data/masks/rope/coarse/val/...
  data/masks/rope/coarse/split.json、manifest.json、stats.md
"""
import argparse
import glob
import io
import json
import os
import re

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
ORIG = os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work', '_orig')
OUT = os.path.join(ROOT, 'data', 'masks', 'rope', 'coarse')
FINAL_EVAL = ['s01', 's02', 's03', 's04', 's05', 'x05']


def find_image(name):
    for e in ('.png', '.jpg', '.jpeg', '.webp'):
        p = os.path.join(ORIG, name + e)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(ORIG, name + '_*')))
    return g[0] if g else None


def perturb(rope, rng, iou_min, iou_max, tries=8):
    """随机形态扰动 + 断裂 + 误检块，返回粗 mask（bool）。"""
    h, w = rope.shape
    best = None
    for _ in range(tries):
        c = rope.copy()
        # 1) 边界：膨胀或腐蚀
        k = rng.choice([3, 5, 7, 9])
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if rng.rand() < 0.5:
            c = cv2.dilate(c.astype(np.uint8), ker).astype(bool)
        else:
            c = cv2.erode(c.astype(np.uint8), ker).astype(bool)
        # 2) 断裂：抹掉几段细长条
        for _ in range(rng.randint(1, 4)):
            if c.sum() < 500:
                break
            ys, xs = np.where(c)
            i = rng.randint(len(ys))
            y0, x0 = int(ys[i]), int(xs[i])
            L = rng.randint(20, 90)
            th = rng.randint(2, 7)
            ang = rng.uniform(0, np.pi)
            dy, dx = np.sin(ang) * L, np.cos(ang) * L
            p1 = (int(x0 - dx / 2), int(y0 - dy / 2))
            p2 = (int(x0 + dx / 2), int(y0 + dy / 2))
            cv2.line(c.view(np.uint8).reshape(h, w), p1, p2, 0, th)
        # 3) 误检块：绳附近的块
        band = cv2.dilate(rope.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        bys, bxs = np.where(band & ~rope)
        for _ in range(rng.randint(1, 4)):
            if len(bys) == 0:
                break
            i = rng.randint(len(bys))
            y0, x0 = int(bys[i]), int(bxs[i])
            ax, ay = rng.randint(8, 60), rng.randint(8, 60)
            if rng.rand() < 0.5:
                cv2.ellipse(c.view(np.uint8).reshape(h, w), (x0, y0), (ax, ay), 0, 0, 360, 1, -1)
            else:
                cv2.rectangle(c.view(np.uint8).reshape(h, w),
                              (max(0, x0 - ax), max(0, y0 - ay)),
                              (min(w - 1, x0 + ax), min(h - 1, y0 + ay)), 1, -1)
        # 4) 盐噪
        drop = rng.rand(h, w) < rng.uniform(0.005, 0.03)
        c = c & ~(drop & rope)
        inter = float((c & rope).sum())
        union = float((c | rope).sum())
        iou = inter / max(union, 1.0)
        if iou_min <= iou <= iou_max:
            return c, iou
        if best is None or abs(iou - (iou_min + iou_max) / 2) < abs(best[1] - (iou_min + iou_max) / 2):
            best = (c, iou)
    return best


def n_blocks(m, min_area=64):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-image', type=int, default=3)
    ap.add_argument('--iou-min', type=float, default=0.45)
    ap.add_argument('--iou-max', type=float, default=0.92)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-every', type=int, default=8, help='排序后每隔多少张进 val')
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    names = sorted(os.path.basename(p)[:-len('_labels.png')]
                   for p in glob.glob(os.path.join(RAW, '*_labels.png')))
    pool = [n for n in names if n not in FINAL_EVAL]
    val = [pool[i] for i in range(0, len(pool), a.val_every)]
    train = [n for n in pool if n not in val]
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    io.open(os.path.join(OUT, 'split.json'), 'w', encoding='utf-8').write(json.dumps(
        {'rule': '按源图切分：先剔除最终评测图(%s)，其余排序后每隔 %d 张进 val' % (FINAL_EVAL, a.val_every),
         'train': train, 'val': val, 'final_eval': FINAL_EVAL,
         'n_train': len(train), 'n_val': len(val)}, ensure_ascii=False, indent=1))

    man = []
    for split, group in (('train', train), ('val', val)):
        for name in group:
            lp = os.path.join(RAW, name + '_labels.png')
            lab = cv2.imread(lp, 0)
            if lab is None:
                continue
            rope = (lab == 1)
            if rope.sum() < 500:
                continue
            for i in range(a.n_per_image):
                c, iou = perturb(rope, rng, a.iou_min, a.iou_max)
                p = os.path.join(OUT, split, '%s_c%d.png' % (name, i))
                cv2.imwrite(p, (c * 255).astype(np.uint8))
                man.append({'split': split, 'name': name, 'pair': i, 'iou': round(iou, 4),
                            'coarse_px': int(c.sum()), 'gt_px': int(rope.sum()),
                            'coarse_blocks': n_blocks(c), 'gt_blocks': n_blocks(rope),
                            'missing_blocks': n_blocks(rope & ~c), 'extra_blocks': n_blocks(c & ~rope),
                            'path': os.path.relpath(p, ROOT)})
    io.open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8').write(
        json.dumps(man, ensure_ascii=False, indent=1))

    ious = np.array([m['iou'] for m in man])
    q = np.percentile(ious, [0, 10, 25, 50, 75, 90, 100])
    mb = np.array([m['missing_blocks'] for m in man])
    eb = np.array([m['extra_blocks'] for m in man])
    lines = ['# 粗 mask 训练对（路线 A / SegRefiner 式精修器）', '',
             '生成：`.runtime` 之外的数据目录 `data/masks/rope/coarse/`；脚本 `scripts/rope_make_coarse.py`。',
             '扰动=随机膨胀/腐蚀(核3~9) + 1~3 段细长"漏检"抹除 + 1~3 个绳附近"误检块" + 少量盐噪；',
             '每次扰动重复采样直到 IoU 落在 [%.2f, %.2f]（默认）。' % (a.iou_min, a.iou_max), '',
             '| 项 | 值 |', '|---|---|',
             '| 训练对 | %d（train %d 张图 × %d 个扰动） |' % (len([m for m in man if m['split'] == 'train']), len(train), a.n_per_image),
             '| 验证对 | %d（val %d 张图 × %d 个扰动） |' % (len([m for m in man if m['split'] == 'val']), len(val), a.n_per_image),
             '| 切分方式 | **按源图**：先剔除最终评测图 %s，其余排序后每隔 %d 张进 val |' % (FINAL_EVAL, a.val_every),
             '| val 图名单 | %s |' % ', '.join(val),
             '| 粗/真值 IoU | 均值 %.3f，min %.3f / p10 %.3f / p25 %.3f / 中位 %.3f / p75 %.3f / p90 %.3f / max %.3f |'
             % (ious.mean(), q[0], q[1], q[2], q[3], q[4], q[5], q[6]),
             '| 每对"漏检段"数 | 均值 %.1f（p50 %.0f, p90 %.0f） |' % (mb.mean(), np.percentile(mb, 50), np.percentile(mb, 90)),
             '| 每对"误检块"数 | 均值 %.1f（p50 %.0f, p90 %.0f） |' % (eb.mean(), np.percentile(eb, 50), np.percentile(eb, 90)),
             '', '最终评测图（%s）**不在** train/val 里，保持干净。' % ', '.join(FINAL_EVAL)]
    io.open(os.path.join(OUT, 'stats.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
