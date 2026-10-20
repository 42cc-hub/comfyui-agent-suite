r"""把"色块分区"的人工标注变成**1 通道线稿目标**（给"升级版魔棒"用）。

标注语义（用户 2026-10-14 明确）：**每个颜色都只是"一块"，颜色本身没有任何含义**；
黑色也是涂过的颜色之一。因此标注**覆盖整幅图**（每个"能按色彩融并的区域"都要涂）——
这就是工作量大的原因。⇒ 本文档里：
  * **块 = 同色连通域**（含黑色；只丢 <min-block 的碎点）；
  * **监督区 = 全部像素**（不存在"未涂"）；
  * 块内部的纹理/渐变被人为溶掉了 ⇒ 目标线稿在块内为 0，**这正是要模型学的"该溶就溶"**。

出两种线稿目标（"闭合性"这件事的核心分叉），训练时各训一个比下游效果：
  * `line_edge`：**块边界带 ∩ 原图强边缘**（细、准；但边界处没有图像边缘时会丢线 → 可能漏缝）；
  * `line_band`：**块边界本身**（人工认定必分的地方一定有线；略粗 ~2px，但不漏）。

产物（每张）：`targets/<stem>_{blocks,line_edge,line_band,valid}.png` + `targets/index.csv`。

用法：python scripts/colorblock_targets.py [--min-block 24] [--band 3] [--edge-pct 88]
"""
import argparse
import csv
import glob
import io
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')


def frontier(blocks):
    """块与块的边界（1px）。"""
    e = np.zeros_like(blocks, np.uint8)
    d = (blocks[:, :-1] != blocks[:, 1:])
    e[:, :-1] |= d
    d = (blocks[:-1, :] != blocks[1:, :])
    e[:-1, :] |= d
    return e


def despeckle(m, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(DATA, 'colorblock_ds'))
    ap.add_argument('--min-block', type=int, default=24, help='小于这个面积的同色连通域当噪声丢掉')
    ap.add_argument('--band', type=int, default=3, help='块边界带宽度（px）')
    ap.add_argument('--edge-pct', type=float, default=88.0, help='带内原图边缘保留的百分位')
    ap.add_argument('--min-line', type=int, default=6, help='线连通域小于这个面积当噪点丢掉')
    a = ap.parse_args()
    o_dir, c_dir = os.path.join(a.dir, 'originals'), os.path.join(a.dir, 'copies')
    t_dir = os.path.join(a.dir, 'targets')
    os.makedirs(t_dir, exist_ok=True)
    rows = []
    for op in sorted(glob.glob(os.path.join(o_dir, '*.png'))):
        stem = os.path.splitext(os.path.basename(op))[0]
        cp = os.path.join(c_dir, stem + '.png')
        if not os.path.exists(cp) or open(op, 'rb').read() == open(cp, 'rb').read():
            continue                                       # 还没标
        bgr = cv2.imread(op, cv2.IMREAD_COLOR)
        painted = cv2.imread(cp, cv2.IMREAD_COLOR)
        if bgr is None or painted is None:
            continue
        if painted.shape != bgr.shape:
            painted = cv2.resize(painted, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        # ---- 1) 块 = 同色连通域（**含黑色**；只丢小碎点）----
        flat = painted.reshape(-1, 3).astype(np.uint32)
        key = ((flat[:, 2] << 16) | (flat[:, 1] << 8) | flat[:, 0]).reshape(painted.shape[:2])
        blocks = np.zeros(painted.shape[:2], np.int32)
        n_ext = 0
        for color in np.unique(key):
            m = (key == color).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < a.min_block:
                    continue
                n_ext += 1
                blocks[lab == i] = n_ext
        if n_ext == 0:
            continue
        fr = frontier(blocks)
        band = cv2.dilate(fr, np.ones((2 * a.band + 1,) * 2, np.uint8))
        # ---- 2a) 带内取原图强边缘（细而准，可能漏"没有图像边缘的边界"）----
        lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        g = np.zeros(bgr.shape[:2], np.float32)
        for c in range(3):
            gx = cv2.Sobel(lab_img[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab_img[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
            g = np.maximum(g, cv2.magnitude(gx, gy))
        inside = g[band > 0]
        thr = float(np.percentile(inside, a.edge_pct)) if inside.size else 0.0
        line_edge = despeckle(((g >= thr) & (band > 0)).astype(np.uint8), a.min_line)
        # ---- 2b) 直接用边界本身（人工认定必分处一定有线，略粗不漏）----
        line_band = despeckle(cv2.dilate(fr, np.ones((3, 3), np.uint8)), a.min_line)
        valid = np.full(blocks.shape, 255, np.uint8)        # 监督区 = 全图
        cv2.imwrite(os.path.join(t_dir, stem + '_blocks.png'), blocks.astype(np.uint16))
        cv2.imwrite(os.path.join(t_dir, stem + '_line_edge.png'), line_edge * 255)
        cv2.imwrite(os.path.join(t_dir, stem + '_line_band.png'), line_band * 255)
        cv2.imwrite(os.path.join(t_dir, stem + '_valid.png'), valid)
        rows.append({'item': stem, 'blocks': int(n_ext),
                     'line_edge_pct': round(100.0 * line_edge.mean(), 3),
                     'line_band_pct': round(100.0 * line_band.mean(), 3),
                     'edge_kept_pct': round(100.0 * line_edge.sum() / max(1, line_band.sum()), 1)})
    with io.open(os.path.join(t_dir, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['item', 'blocks', 'line_edge_pct', 'line_band_pct',
                                           'edge_kept_pct'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if rows:
        print('生成 %d 张目标 → %s' % (len(rows), t_dir))
        print('  每张块数：中位 %.0f（%d~%d）｜line_edge 中位 %.2f%%｜line_band 中位 %.2f%%'
              '｜band→edge 保留率 中位 %.1f%%'
              % (np.median([r['blocks'] for r in rows]), min(r['blocks'] for r in rows),
                 max(r['blocks'] for r in rows), np.median([r['line_edge_pct'] for r in rows]),
                 np.median([r['line_band_pct'] for r in rows]),
                 np.median([r['edge_kept_pct'] for r in rows])))
    else:
        print('没有已标注的切片')
    return 0


if __name__ == '__main__':
    sys.exit(main())
