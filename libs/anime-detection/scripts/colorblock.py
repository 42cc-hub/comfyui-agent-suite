"""色块对象识别：给定 1 个种子点，把"视觉上同一块"的对象（绳子/头发/衣服…）整块选出来。

用户 2026-10-13 的真实痛点：PS 魔棒选一个内部颜色统一的色块要点很多下，还漏 ——
原因是魔棒①沿渐变扩散不出去（对象内部明暗渐变被当成不同色）②对很淡或断掉的描边没适应性（漏到外面）。

本工具的对策（无需训练、本地 CPU 秒级）：
  1. **超像素**（SLIC）先把图切成贴边的小块 —— 边界代价只在块之间算，比逐像素稳，也快；
  2. 相邻块之间算**边界带颜色差**（两侧各取几像素的带，取均值）＋**边缘屏障**
     （Lab 梯度幅值，弱的描边也被放大成屏障），合成"通过代价"；
  3. **两级区域生长**（关键）：每一步只跟**当前前沿**比颜色（所以能顺着渐变一路走），
     同时对"离种子色的总偏差"设一个宽松上限（所以不会一路漂到别的对象）；
  4. 可选 `--refine rw`：把生长结果当种子，在像素图上跑**随机游走**，让边界贴合真实分界线。

用法：
  python scripts/colorblock.py IMG --seed 320,480 --out mask.png [--overlay ov.png]
                                  [--exclude x,y ...] [--tau-w 0.35] [--tau-local 20] [--tau-global 60]
                                  [--fill-holes] [--refine rw]
"""
import argparse
import io
import os
import sys

import numpy as np

import cv2


def load_lab(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit('读不到图：%s' % path)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def barrier_map(lab, gap_close=1):
    """Lab 梯度幅值 → [0,1] 屏障；gap_close 用膨胀把断掉的淡描边接上一点。"""
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        g = np.maximum(g, cv2.magnitude(gx, gy))
    g = g / (np.percentile(g, 99.0) + 1e-6)
    g = np.clip(g, 0, 1)
    if gap_close:
        g = cv2.dilate(g, np.ones((3, 3), np.uint8), iterations=gap_close)
    return g


def fz_components(bgr, scale=180.0, sigma=1.0, min_size=60):
    """Felzenszwalb–Huttenlocher 分割：边界差异小于两侧内部变化才合并。"""
    from skimage.segmentation import felzenszwalb
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    seg = felzenszwalb(lab, scale=scale, sigma=sigma, min_size=min_size, channel_axis=2)
    return seg.astype(np.int32)


def superpixels(bgr, n=1400, compactness=12.0):
    from skimage.segmentation import slic
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    seg = slic(lab_s, n_segments=max(50, n // 4), compactness=compactness,
               start_label=0, channel_axis=2, sigma=1.0)
    seg = cv2.resize(seg.astype(np.int32), (bgr.shape[1], bgr.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    return seg


def region_stats(lab, seg, k):
    m = (seg == k)
    px = lab[m]
    return px.mean(axis=0), m


def graph(lab, seg, bar):
    """相邻超像素 → 边界带颜色差 + 屏障均值（向量化，交互要快）。"""
    ks = np.unique(seg)
    kmap = np.zeros(int(ks.max()) + 1, np.int32)
    kmap[ks] = np.arange(len(ks))
    lum = lab[:, :, 0]
    edges = {}
    for dx, dy in ((1, 0), (0, 1)):
        a = seg[:seg.shape[0] - dy, :seg.shape[1] - dx]
        b = seg[dy:, dx:]
        ia = kmap[a][a != b]
        ib = kmap[b][a != b]
        if ia.size == 0:
            continue
        lo = np.minimum(ia, ib)
        hi = np.maximum(ia, ib)
        key = lo.astype(np.int64) * (len(ks) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        # 边界带颜色：用同一条边界两侧的像素均值近似（取边界像素本身与其邻居）
        bb = bar[dy:, dx:][a != b]
        sums = np.bincount(inv, weights=bb, minlength=uk.size)
        cnts = np.bincount(inv, minlength=uk.size)
        klo = (uk // (len(ks) + 1)).astype(np.int32)
        khi = (uk % (len(ks) + 1)).astype(np.int32)
        for n in range(uk.size):
            k = (int(klo[n]), int(khi[n]))
            edges.setdefault(k, [0.0, 0.0, 0])
            edges[k][1] += float(sums[n])
            edges[k][2] += int(cnts[n])
    out = {}
    colk = np.zeros((len(ks), 3), np.float32)
    for i, k in enumerate(ks):
        colk[i] = lab[seg == k].mean(axis=0)
    # 边界带颜色差：用两侧块的均值之差（块内均值对渐变是稳的），屏障用边界像素均值
    for (a, b), (_d0, s, c) in edges.items():
        d = float(np.linalg.norm(colk[a] - colk[b]))
        bv = s / max(1, c)
        out[(a, b)] = (d, bv)
    return out, len(ks), kmap, colk


def grow(seg, edges, seed_k, tau_w, tau_local, tau_global, col, bar_pct=0.995):
    """颜色只跟"当前边界层"比（顺渐变），屏障只在**极强边**时否决（淡描边不误挡）。

    实现要点（2026-10-13 修）：邻接优先队列必须**在每轮 relax 时把新邻居入队**，
    否则长完第一圈就空队列（实测 tau_local 10→60 面积不变 = 前沿没扩展的指纹）。
    """
    import heapq
    bvals = np.array([bv for (_d, bv) in edges.values()], np.float32)
    bar_cut = float(np.percentile(bvals, 100 * bar_pct)) if bvals.size else 1.0
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    region = {seed_k}
    seed_col = col[seed_k].copy()
    border = col[seed_k].copy()
    border_n = 1
    dist = {}
    heap = []

    def relax(k):
        for nb in adj.get(k, ()):
            if nb in region:
                continue
            dcol = float(np.linalg.norm(col[nb] - border))
            if nb not in dist or dcol < dist[nb][0] - 1e-6:
                dist[nb] = (dcol, k)
                heapq.heappush(heap, (dcol, nb))

    relax(seed_k)
    while heap:
        dcol, nb = heapq.heappop(heap)
        if nb in region:
            continue
        if dcol > dist.get(nb, (1e9,))[0] + 1e-6:
            continue                      # 已被更近的路径取代（懒删除）
        if dcol > tau_local:
            break                         # 队列里最近的都超过阈值 ⇒ 贴边即停
        src = dist[nb][1]
        _d, bv = edges[(min(src, nb), max(src, nb))]
        if bv > bar_cut:
            continue                      # 极强边才否决
        if float(np.linalg.norm(col[nb] - seed_col)) > tau_global:
            continue
        region.add(nb)
        border = (border * border_n + col[nb]) / (border_n + 1)
        border_n += 1
        relax(nb)
    return np.isin(seg, list(region))


def refine_rw(bgr, mask):
    """随机游走让边界贴合真实分界线（把当前 mask 与它外面一圈当种子）。"""
    from skimage.segmentation import random_walker
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    small = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    k = np.ones((3, 3), np.uint8)
    outer = (cv2.dilate(m.astype(np.uint8), k, iterations=6) > 0) & (~m.astype(bool))
    markers = np.zeros(small.shape[:2], np.int8)
    markers[m] = 1
    markers[outer] = 2
    if markers.max() == 0 or (markers == 1).sum() == 0 or (markers == 2).sum() == 0:
        return mask
    g = np.zeros(small.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(small[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        g = np.maximum(g, cv2.magnitude(gx, gy))
    beta = 90.0
    res = random_walker(small, markers, beta=beta, mode='bf')
    out = (res == 1)
    out = cv2.resize(out.astype(np.uint8), (mask.shape[1], mask.shape[0]),
                     interpolation=cv2.INTER_NEAREST) > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--seed', nargs='+', required=True, help='x,y（可多个；取所有种子的块一起长）')
    ap.add_argument('--exclude', nargs='*', default=None, help='x,y 明确不要的点（会把所在块拉黑）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlay', default=None)
    ap.add_argument('--engine', choices=['slic', 'fz'], default='fz',
                    help='节点分割器：fz=Felzenszwalb（默认，贴合真边界）、slic=超像素')
    ap.add_argument('--fz-scale', type=float, default=180.0)
    ap.add_argument('--fz-sigma', type=float, default=1.0)
    ap.add_argument('--fz-min-size', type=int, default=60)
    ap.add_argument('--n-seg', type=int, default=1400)
    ap.add_argument('--tau-w', type=float, default=0.25)
    ap.add_argument('--bar-pct', type=float, default=0.995,
                    help='屏障分位阈值：只有比本图这么多比例的边都强的边才否决')
    ap.add_argument('--tau-local', type=float, default=26.0)
    ap.add_argument('--tau-global', type=float, default=100.0)
    ap.add_argument('--gap-close', type=int, default=1)
    ap.add_argument('--fill-holes', action='store_true')
    ap.add_argument('--refine', choices=['none', 'rw'], default='none')
    a = ap.parse_args()
    bgr, lab = load_lab(a.image)
    h, w = bgr.shape[:2]
    bar = barrier_map(lab, a.gap_close)
    seg = (fz_components(bgr, a.fz_scale, a.fz_sigma, a.fz_min_size)
           if a.engine == 'fz' else superpixels(bgr, a.n_seg))
    edges, nk, kmap, col = graph(lab, seg, bar)

    def k_of(xy):
        x, y = [int(v) for v in xy.split(',')]
        x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        return int(seg[y, x])

    seeds = [k_of(s) for s in a.seed]
    banned = {k_of(s) for s in (a.exclude or [])}
    mask = np.zeros((h, w), bool)
    for sk in seeds:
        m = grow(seg, edges, sk, a.tau_w, a.tau_local, a.tau_global, col, a.bar_pct)
        mask |= m
    for bk in banned:
        mask &= (seg != bk)
    if a.fill_holes:
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
    if a.refine == 'rw':
        mask = refine_rw(bgr, mask)
    out = (mask.astype(np.uint8) * 255)
    cv2.imwrite(a.out, out)
    if a.overlay:
        ov = bgr.copy()
        ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(a.overlay, ov)
    print('  面积 %d px（占画面 %.2f%%）→ %s' % (int(mask.sum()), 100.0 * mask.mean(), a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
