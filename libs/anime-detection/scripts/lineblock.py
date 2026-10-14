"""S0：描边线 → 面块 → 层次融并，做到"可选颗粒度"的色块对象识别。

思路（对应 2026-10-13 的构想与两篇参考）：
  1. **线图**：形态学 blackhat（薄暗线）+ skimage 的 ridge 滤波（black_ridges）取并；
  2. **密封断线**：把线图膨胀 `--seal` 像素 —— 这直接对应 Allen 2024 "leak-resistant"要解决的问题：
     描边有断点/太淡时，先把它接上，再让种子困在里面（而不是靠调魔棒容差）；
  3. **面块**：取线图屏障的连通域 ⇒ 每个封闭小面就是一个"面块"（你说的细颗粒度）；
  4. **层次融并**：面块之间按"边界线越弱、两侧色差越小 ⇒ 越该合并"排序，用并查集逐条合并，
     得到一个**合并树**；阈值 τ 切树 = 一个颗粒度（小 τ：单个发片级；大 τ：整片头发级）。
  5. `--taus` 可一次输出多级 mask（多级描边线的"级别"就是这里的 τ）。

用法：
  python scripts/lineblock.py IMG --seed 320,180 --out mask.png --tau 0.35 [--overlay ov.png]
  python scripts/lineblock.py IMG --seed 320,180 --levels 3 --out-prefix out/lv
"""
import argparse
import io
import os
import sys

import numpy as np
import cv2


def line_map(bgr, k=7, ridge=True, ridge_sigmas=(1, 2), blur=3, ref_side=512):
    """薄暗线响应；**核与 σ 按图像尺寸缩放**（ref_side 是调参时的参考边长）。

    一个固定 k=7 的 blackhat 在 2364px 的图上什么都检不到（线宽本身就有 8-15px），
    在 512px 图上却是合适甚至偏大 —— 所以必须随尺寸缩放。
    """
    h, w = bgr.shape[:2]
    s = max(1.0, max(h, w) / float(ref_side))
    k_s = int(round(k * s)) | 1
    blur_s = int(round(blur * s)) | 1
    sig = tuple(max(1.0, x * s) for x in ridge_sigmas)
    g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g0, (blur_s, blur_s), 0) if blur_s > 1 else g0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_s, k_s))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker)
    bh = bh / (np.percentile(bh, 99.5) + 1e-6)
    if ridge:
        from skimage.filters import meijering
        r = meijering(255.0 - g, sigmas=list(sig), black_ridges=True)
        r = r / (np.percentile(r, 99.5) + 1e-6)
        return np.clip(np.maximum(bh, r), 0, 1)
    return np.clip(bh, 0, 1)


def _hysteresis(lm, thr_hi, thr_lo, min_comp=8):
    """Canny 式滞后：只保留与强线相连的弱线；再扔掉太小的连通域。"""
    strong = (lm > thr_hi).astype(np.uint8)
    weak = (lm > thr_lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return strong.astype(bool)
    keep = np.unique(lab[strong > 0])
    keep = keep[keep > 0]
    line = np.isin(lab, keep)
    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    small = {i for i in range(1, n2) if stats[i, cv2.CC_STAT_AREA] < min_comp}
    if small:
        line &= ~np.isin(lab2, list(small))
    return line


def build_faces(lm, thr=0.25, seal=1, thr_hi=None, min_comp=8):
    """线图 → 屏障 → 面块连通域。

    thr   : 弱阈值（滞后阈值的下限）
    thr_hi: 强阈值（默认 2×thr）；只有与强线相连的弱线才保留
    seal  : 屏障膨胀像素（容许多大的断口；线图干净后 1 就够）
    """
    line = _hysteresis(lm, thr_hi if thr_hi else min(0.95, thr * 2.0), thr, min_comp)
    bar = line.astype(np.uint8)
    if seal > 0:
        bar = cv2.dilate(bar, np.ones((3, 3), np.uint8), iterations=seal)
    n, lab = cv2.connectedComponents((1 - bar).astype(np.uint8), connectivity=4)
    return lab.astype(np.int32), n, bar


def face_colors(lab, n, lab_img):
    """每个面块的平均 Lab 与面积。"""
    flat = lab.ravel()
    col = np.zeros((n, 3), np.float32)
    area = np.bincount(flat, minlength=n).astype(np.float32)
    for c in range(3):
        s = np.bincount(flat, weights=lab_img[:, :, c].ravel(), minlength=n)
        col[:, c] = s / np.maximum(area, 1)
    return col, area


def rag(lab, n, lm):
    """相邻面块：边界上的平均线强度 + 接触长度。"""
    pairs = {}
    for dx, dy in ((1, 0), (0, 1)):
        a = lab[:lab.shape[0] - dy, :lab.shape[1] - dx].ravel()
        b = lab[dy:, dx:].ravel()
        l = lm[dy:, dx:].ravel() if dy or dx else lm.ravel()
        m = a != b
        if not m.any():
            continue
        for ka, kb, vl in zip(a[m], b[m], l[m]):
            if ka == 0 and kb == 0:
                continue
            k = (min(ka, kb), max(ka, kb))
            e = pairs.get(k)
            if e is None:
                pairs[k] = [float(vl), 1]
            else:
                e[0] += float(vl)
                e[1] += 1
    return {k: (v[0] / max(1, v[1]), v[1]) for k, v in pairs.items()}


def merge_tree(lab, n, lm, col, area, w_col=1.0, w_line=0.15, min_area=0, dc0=10.0):
    """按 (边界线强度, 色差) 排序合并 → 返回合并顺序（供阈值切树）。"""
    E = rag(lab, n, lm)
    items = []
    for (a, b), (bline, cnt) in E.items():
        if min_area and (area[a] < min_area or area[b] < min_area):
            pass
        dc = float(np.linalg.norm(col[a] - col[b]))
        # 代价以**颜色**为主：同色跨内部线先合（=溶解内部线条），色差大后合（=对象边界）；
        # 线条强度只做轻微阻尼（w_line 默认 0.15）。
        cost = w_col * min(1.0, (dc / dc0) ** 2) + w_line * bline
        items.append((cost, a, b, bline, dc, cnt))
    items.sort(key=lambda t: t[0])
    return items


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def region_at(lab, n, items, tau, seed_label):
    uf = UF(n)
    for cost, a, b, _bl, _dc, _c in items:
        if cost > tau:
            break
        uf.union(a, b)
    root = uf.find(seed_label)
    keep = {i for i in range(n) if uf.find(i) == root}
    return np.isin(lab, list(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--seed', required=True, help='x,y')
    ap.add_argument('--out', default=None)
    ap.add_argument('--overlay', default=None)
    ap.add_argument('--out-prefix', default=None, help='给了就按 --levels 输出多级 mask')
    ap.add_argument('--tau', type=float, default=0.35)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--line-thr', type=float, default=0.25)
    ap.add_argument('--seal', type=int, default=1, help='线图膨胀像素：容许多大的断口')
    ap.add_argument('--min-comp', type=int, default=8, help='线连通域小于这个面积就丢掉（去噪）')
    ap.add_argument('--line-k', type=int, default=7)
    ap.add_argument('--ref-side', type=int, default=512, help='线检测窗口的参考边长（按图缩放）')
    ap.add_argument('--no-ridge', action='store_true')
    ap.add_argument('--line-source', choices=['lineart', 'heuristic'], default='lineart',
                    help='lineart=用训练好的 AnimeLineart(netG.pth)；heuristic=手搓形态学检测')
    ap.add_argument('--add', nargs='*', default=None, help='x,y 面块级加入（可多个）')
    ap.add_argument('--drop', nargs='*', default=None, help='x,y 面块级扣除（可多个）')
    ap.add_argument('--w-line', type=float, default=1.0)
    ap.add_argument('--w-col', type=float, default=1.0)
    ap.add_argument('--dc0', type=float, default=10.0, help='颜色项尺度（Lab 单位）')
    a = ap.parse_args()
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit('读不到图：%s' % a.image)
    lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if a.line_source == 'lineart':
        from lineart_anime_local import load_netG, line_response
        net, dev = load_netG()
        lm, _info = line_response(bgr, net, dev)
    else:
        lm = line_map(bgr, a.line_k, not a.no_ridge, ref_side=a.ref_side)
    lab, n, bar = build_faces(lm, a.line_thr, a.seal, min_comp=a.min_comp)
    col, area = face_colors(lab, n, lab_img)
    x, y = [int(v) for v in a.seed.split(',')]
    sk = int(lab[y, x])
    if sk == 0:
        # 种子落在线/屏障上：取最近的可用面块
        d, lbl = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        sk = int(lab[(y, x)][0]) if False else 0
        ys, xs = np.where(lab > 0)
        j = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        sk = int(lab[ys[j], xs[j]])
        print('  种子落在线上，改用最近的面块 %d' % sk)
    items = merge_tree(lab, n, lm, col, area, a.w_line, a.w_col, dc0=a.dc0)

    def region_of(xy, tau):
        xx, yy = [int(v) for v in xy.split(',')]
        k = int(lab[yy, xx])
        if k == 0:                       # 点在线上：取最近的可用面块
            ys, xs = np.where(lab > 0)
            j = int(np.argmin((xs - xx) ** 2 + (ys - yy) ** 2))
            k = int(lab[ys[j], xs[j]])
        return region_at(lab, n, items, tau, k)
    print('  面块 %d 个；相邻边界 %d 条；种子面块 %d（%d px）' % (n, len(items), sk, int(area[sk])))
    if a.out_prefix:
        for i in range(a.levels):
            tau = a.tau * (i + 1)                      # 级 1..N：阈值递增 = 颗粒度变粗
            m = region_at(lab, n, items, tau, sk)
            fp = '%s%d.png' % (a.out_prefix, i + 1)
            cv2.imwrite(fp, (m.astype(np.uint8) * 255))
            print('   级别%d τ=%.3f ⇒ %d px（%.1f%% 画面）→ %s'
                  % (i + 1, tau, int(m.sum()), 100 * m.mean(), fp))
    else:
        m = region_at(lab, n, items, a.tau, sk)
        for xy in (a.add or []):
            m = m | region_of(xy, a.tau)
        for xy in (a.drop or []):
            m = m & ~region_of(xy, a.tau)
        cv2.imwrite(a.out, (m.astype(np.uint8) * 255))
        print('  τ=%.3f ⇒ %d px（%.1f%% 画面）→ %s' % (a.tau, int(m.sum()), 100 * m.mean(), a.out))
        if a.overlay:
            ov = bgr.copy()
            ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(a.overlay, ov)
    return 0


if __name__ == '__main__':
    sys.exit(main())
