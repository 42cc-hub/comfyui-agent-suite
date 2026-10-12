#!/usr/bin/env python
"""绳子 mask 的传统 CV 预标注 v3 —— 面块层(B)/描边层(L) 分层 + ROI 限制 + 正负颜色原型。

用户 2026-10-10 的架构意见（本版的核心）：
  「面块和描边应该单独处理，而不是任凭 flooding 直接沿着描边线蔓延。」
落到实现上是四条硬约束：
  1) L 层（描边）＝局部暗脊 ∩ 细（2×内切直径 ≤ line_width），只当**边界**并被单独保存；
  2) B 层（面块）＝绳色像素的连通域，**整块判定**（细度用块级最大内切直径，不看单像素），
     块间生长只走邻接图(RAG)，一次只跨一条描边，远处块之间没有边——这是"不许蔓延"；
  3) 描边块**单独判定**（两侧面块是否都是绳子 / 自身颜色是否绳色 / 父线稿是否整条绳色）；
  4) 所有扩张（RAG 生长、像素补全、桥接）**只在 ROI 内**，ROI 由高置信种子膨胀得到，
     "沿背景缝隙泄漏到全画幅"于是结构上不可能发生。

为什么必须"整块判定"（复现过的上版问题，也是上一版的死路）：
  上版逐像素判绳色，再用"细长像素一律算绳子"补漏，有两个系统性后果：
  (a) 极暗的沙发绗缝阴影楔被当成高饱和红：那些像素近黑偏棕，H≈2 / S≈224 / V≈49
      （OpenCV 量程），HSV 里就是"高饱和红"——本版加了 V 下限，暗楔进不来；
  (b) 大团块（大腿潮红、整片皮肤）的**边界层**总能满足"局部很细"，团块外圈因此被收进来。
      细度改成块级量后团块整块否掉，边界层属于团块、不单独存活。

种子为什么要"相对周边的颜色跳变"（--contrast-*）：
  绳子是**外来物体**，紧邻邻域不是绳色；绗缝阴影/皮肤皱褶/潮红是**同一材质的加深**，
  紧邻邻域也是同色高饱和，比值≈1。实测 test(1) 绗缝块 S_block/S_邻域≈1.0~1.2、|ΔHue|≈0~4，
  真绳块是 S 比 1.5~16 或 |ΔHue| 48~69。这条只用来定**种子**：绳子的高光/暗部子块跳变不明显，
  靠 RAG 从种子上长回来（治 test(6) 的断续）。

正/负两类证据（--fg-mask/--bg-mask 与负原型）：
  人做 mask 的常态是"先多画一块、再把外面点掉"，所以"确定不是绳子"和"确定是绳子"一样有用。
  - 自动模式：除正原型外还估**负原型**（同色大块、灰细线、远离带状结构的饱和像素），存在
    rope_color_prior.json 的 pos/neg 两栏里，可跨图复用（这批图是同一种红麻绳，跨图先验比单图自适应稳）。
  - 交互模式：给了 --fg-mask/--bg-mask 就在 ROI 内用 grabCut(GC_INIT_WITH_MASK) **重新解算**，
    而不是把用户笔画并/差到旧 mask 上（并差法边界脏，解算才会把边界拉到描边线上）。

已知失败模式与死路见 docs/rope_prelabel_cv.md。本脚本只做**预标注**：
允许不完美（人在 canvas 上修），但必须贴着绳子，不能把皮肤/背景/整张图吃进来。

用法：
  python scripts/rope_prelabel_cv.py --src <图或目录> --out <目录>
  python scripts/rope_prelabel_cv.py --src test.png --out out --fg-mask fg.png --bg-mask bg.png
"""
import argparse
import glob
import io
import json
import os

import cv2
import numpy as np
from scipy import ndimage

HUE_PERIOD = 180                 # OpenCV 的 H 通道是 0~179 的环，红色横跨两端
RED_HUE = 0
E8 = np.ones((3, 3), int)
ELL = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1,) * 2)


# --------------------------------------------------------------------------- 颜色工具

def ring_hue_dist(h, hue):
    """环形色相距离：不把 0/179 接起来的话，红色系会在 0 附近断开。"""
    d = np.abs(np.asarray(h, np.float64) - hue)
    return np.minimum(d, HUE_PERIOD - d)


def median_per_label(labels, n, ch):
    """每个连通域的中位通道值（0~255 量程）。

    bincount 直方图一次过，比逐块 np.median 快一个数量级；中位数是必需的鲁棒量
    （一块里混着绳子高光和暗部时，均值会被单侧拉走）。
    """
    idx = labels.ravel().astype(np.int64) * 256 + ch.ravel().astype(np.int64)
    hist = np.bincount(idx, minlength=(n + 1) * 256).reshape(n + 1, 256)
    cdf = np.cumsum(hist, axis=1)
    return (cdf * 2 >= cdf[:, -1:]).argmax(1).astype(np.float64)


def block_contrast(hsv, labels, i, sl, in_r, out_r):
    """块相对紧邻邻域的颜色跳变：饱和度比 / 色相差 / 明度比。

    只看"紧邻"（默认 3~9px）：绗缝阴影的紧邻邻域还在同一条渐变里，比值≈1；
    绳子紧邻就是皮肤/背景，跳变明显。邻域取整幅图（不限于绳色像素），否则又变成自证。
    """
    y0, y1 = max(0, sl[0].start - out_r - 2), min(hsv.shape[0], sl[0].stop + out_r + 2)
    x0, x1 = max(0, sl[1].start - out_r - 2), min(hsv.shape[1], sl[1].stop + out_r + 2)
    sub = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
    if sub.sum() == 0:
        return None
    ann = (cv2.dilate(sub, ELL(out_r)) > 0) & (cv2.dilate(sub, ELL(in_r)) == 0)
    if ann.sum() < 20:
        return None
    hs = hsv[y0:y1, x0:x1]
    blk = sub > 0
    return {'s_ratio': float(np.median(hs[..., 1][blk]) / max(1.0, np.median(hs[..., 1][ann]))),
            'hue_jump': float(ring_hue_dist(np.median(hs[..., 0][blk]),
                                            np.median(hs[..., 0][ann]))),
            'v_ratio': float(np.median(hs[..., 2][blk]) / max(1.0, np.median(hs[..., 2][ann])))}


def jump_ok(c, s_ratio, hue_jump, v_ratio):
    return bool(c and (c['s_ratio'] >= s_ratio or c['hue_jump'] >= hue_jump
                       or c['v_ratio'] <= v_ratio))


# --------------------------------------------------------------------------- L 层：描边

def stroke_layer(img, dark_delta, bg_kernel, line_width, min_area):
    """描边层＝"比局部背景暗 dark_delta" ∩ "本身是细的"。

    细度约束不能省：绳子本体（暗红）也比背景暗，整条会落进暗脊；不筛细度的话绳子被 L 层
    整条吃掉，面块层里就没有绳子了。2×内切直径 ≤ line_width 正好把"线"和"块"分开。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ELL(bg_kernel // 2))
    dark = (bg.astype(np.int16) - gray.astype(np.int16)) >= dark_delta
    dt = ndimage.distance_transform_edt(dark)
    L = dark & (2.0 * dt <= line_width)
    if min_area > 1:
        lab, n = ndimage.label(L, structure=E8)
        if n:
            sizes = np.bincount(lab.ravel())
            # 必须带 lab>0：背景的 sizes[0] 是"非描边像素总数"，不排掉会把整幅图变成描边
            L = (lab > 0) & (sizes[lab] >= min_area)
        else:
            L = np.zeros_like(L)
    return L


def label_extent(labels, n, radius):
    """把每个块的 label 向外扩张 radius 像素（取邻域最大 label）。

    灰度膨胀实现：C 层一次调用就够，比逐块做距离变换快得多。
    只能用 CV_32F（cv2.dilate 不支持 int32 标签图）；背景先置 -1 再还原。
    """
    m = labels.astype(np.float32)
    m[labels == 0] = -1.0
    d = cv2.dilate(m, ELL(radius))
    d[labels == 0] = -1.0
    return d


# --------------------------------------------------------------------------- B 层：面块

def block_stats(labels, n, hsv, thin_width):
    """块级量：面积 / 最大内切直径 wmax / 骨架长 / 中位色。

    wmax 是"细带还是团块"的唯一可靠判据：面积/骨架长那种平均宽度对大片背景会失效
    （大片背景骨架稀疏，算出来反而"细"），而最大内切直径对大团块一定很大。
    """
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1).astype(np.float64)
    dt = ndimage.distance_transform_edt(labels > 0)
    wmax = np.zeros(n + 1, np.float64)
    np.maximum.at(wmax, flat, dt.ravel())
    wmax *= 2.0
    skel = cv2.ximgproc.thinning(((labels > 0) * 255).astype(np.uint8)) > 0
    skel_len = np.bincount(labels[skel].ravel(), minlength=n + 1).astype(np.float64)
    med = {c: median_per_label(labels, n, hsv[..., c]) for c in (0, 1, 2)}
    return {'area': area, 'wmax': wmax, 'skel_len': skel_len, 'med': med,
            'thin_ok': wmax <= thin_width}


def build_rag(labels, n, radius):
    """块邻接图：只有"隔着一条描边直接相邻"的块之间才有边，边权＝接触长度。

    只在 shell 偏移上取对（半径 radius-1~radius 那一圈）：隔着一条描边（宽 3~7px）的两块
    最近距离就落在这个壳里，扫全盘只是白算。远处块之间不建边——这是"不许蔓延"的保证。
    """
    h, w = labels.shape
    offs = [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if (radius - 1) ** 2 < dy * dy + dx * dx <= radius * radius]
    keys, n1 = [], n + 1
    for dy, dx in offs:
        a = labels[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)].ravel()
        b = labels[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)].ravel()
        sel = (a > 0) & (b > 0) & (a != b)
        if sel.any():
            lo = np.minimum(a[sel], b[sel]).astype(np.int64)
            hi = np.maximum(a[sel], b[sel]).astype(np.int64)
            keys.append(lo * n1 + hi)
    if not keys:
        return {}
    uniq, cnt = np.unique(np.concatenate(keys), return_counts=True)
    adj = {}
    for x, y, c in zip((uniq // n1).astype(np.int32), (uniq % n1).astype(np.int32), cnt):
        adj.setdefault(int(x), []).append((int(y), int(c)))
        adj.setdefault(int(y), []).append((int(x), int(c)))
    for k in adj:                        # 生长时先用接触面最宽的邻居
        adj[k].sort(key=lambda t: -t[1])
    return adj


def block_median_1d(values_img, labels, n):
    """按块取一维图（如颜色分）的中位数。"""
    flat = labels.ravel()
    order = np.argsort(flat, kind='stable')
    fs, vs = flat[order], values_img.ravel()[order]
    bounds = np.searchsorted(fs, np.arange(1, n + 2))
    out = np.full(n + 1, 1e9)
    for i in range(1, n + 1):
        if bounds[i] > bounds[i - 1]:
            out[i] = np.median(vs[bounds[i - 1]:bounds[i]])
    return out


# --------------------------------------------------------------------------- 正/负颜色原型

def fit_prototypes(lab, samples, k=3, min_px=60, tag='pos'):
    """在 Lab 上对样本做 k-means，得到几个颜色原型（受光/主体/阴影 等）。

    单一绳色不够用：同一根绳子高光和暗部在 Lab 上差得比"绳子 vs 潮红"还远。
    正负两栏都存 json，可跨图复用。
    """
    if samples is None or samples.sum() < min_px:
        return None
    samples = samples.astype(bool)          # 必须是 bool，否则会被当成整型花式索引
    px = np.ascontiguousarray(lab[samples].reshape(-1, 3), dtype=np.float32)
    if len(px) < max(20, min_px // 2):
        return None
    if len(px) > 60000:
        px = px[np.random.RandomState(0).choice(len(px), 60000, replace=False)]
    k = int(min(k, max(1, len(px) // 200)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    try:
        _, lbl, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return None
    scale = []
    for i in range(k):
        sel = px[lbl.ravel() == i]
        scale.append((np.maximum(np.median(np.abs(sel - centers[i]), axis=0) * 1.4826,
                                 np.array([10.0, 7.0, 7.0])).tolist()
                      if len(sel) > 20 else [10.0, 7.0, 7.0]))
    return {'tag': tag, 'k': int(k), 'centers': centers.tolist(), 'scale': scale}


def proto_dist_map(lab, proto):
    """像素到最近原型的归一化距离（Lab，按每簇 MAD 归一）。"""
    best = np.full(lab.shape[:2], 1e9, np.float32)
    for c, s in zip(proto['centers'], proto['scale']):
        d = np.sqrt((((lab.astype(np.float32) - np.array(c, np.float32))
                      / np.array(s, np.float32)) ** 2).sum(-1))
        np.minimum(best, d, out=best)
    return best


# --------------------------------------------------------------------------- 种子/ROI（自动与 QA 共用）

def build_seeds(img, kw):
    """算 L 层 / 面块 / 高置信种子 / ROI / 正负原型。返回 dict。

    抽成函数是为了让"人工成本 QA"能对**任意** mask（包括上一版输出）用同一套种子口径，
    否则"误检块数"这种指标在不同版本之间没法比。
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    thin_width = kw['thin_width'] or float(np.clip(0.035 * min(h, w), 16.0, 90.0))
    line_width = kw['line_width'] or float(np.clip(0.012 * min(h, w), 5.0, 14.0))
    s_min, v_min, hue, half = kw['s_min'], kw['v_min'], kw['hue'], kw['half']
    stroke_s_min = kw['stroke_s_min'] if kw['stroke_s_min'] is not None else 0.85 * s_min
    stroke_v_min = kw['stroke_v_min'] if kw['stroke_v_min'] is not None else v_min
    L = stroke_layer(img, kw['dark_delta'], kw['bg_kernel'], line_width, 4)
    barrier = cv2.dilate(L.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=kw['dilate']).astype(bool)
    # 绳色像素证据：V 下限不是可选项——近黑像素的 H/S 是退化的，暗楔就靠它挡在外面
    M = ((ring_hue_dist(hsv[..., 0], hue) <= half) & (hsv[..., 1] >= s_min)
         & (hsv[..., 2] >= v_min))
    strict = ((ring_hue_dist(hsv[..., 0], hue) <= half) & (hsv[..., 1] >= stroke_s_min)
              & (hsv[..., 2] >= stroke_v_min))
    labels, n = ndimage.label(M & ~barrier, structure=E8)
    out = {'hsv': hsv, 'lab': lab, 'L': L, 'barrier': barrier, 'M': M, 'strict': strict,
           'thin_width': thin_width, 'line_width': line_width, 's_min': s_min, 'v_min': v_min,
           'stroke_s_min': stroke_s_min, 'stroke_v_min': stroke_v_min,
           'labels': labels, 'n': n, 'r1': np.zeros(n + 1, bool), 'core': np.zeros((h, w), bool),
           'roi': np.zeros((h, w), bool), 'score': None, 'proto': None}
    if n == 0:
        return out
    st = block_stats(labels, n, hsv, thin_width)
    sls = ndimage.find_objects(labels)
    idx = np.arange(n + 1)
    shape_ok = ((idx > 0) & (st['area'] >= kw['min_block_area']) & st['thin_ok']
                & (st['skel_len'] / np.maximum(st['wmax'], 1.0) >= kw['min_ratio']))
    r1 = np.zeros(n + 1, bool)
    for i in np.where(shape_ok)[0]:
        if jump_ok(block_contrast(hsv, labels, i, sls[i - 1], kw['ann_in'], kw['ann_out']),
                   kw['contrast_s'], kw['contrast_hue'], kw['contrast_v']):
            r1[i] = True
    core = r1[labels] > 0
    roi_r = int(np.clip(kw['roi_factor'] * min(h, w), kw['roi_min'], kw['roi_max']))
    roi = cv2.dilate(core.astype(np.uint8), ELL(roi_r)) > 0
    roi |= core
    # 正原型：种子块 + 贴着种子块的绳色描边像素（暗红描边是独立的一族）
    seed_px = core
    ext = label_extent(labels, n, kw['rag_radius'] + 1)
    near_seed = (np.isin(ext, np.where(r1)[0].astype(np.float32)) if r1.any()
                 else np.zeros((h, w), bool))
    proto, neg = None, None
    pf = kw.get('proto_file')
    if kw.get('proto_reuse') and pf and os.path.exists(pf):
        try:
            j = json.load(io.open(pf, encoding='utf-8'))
            proto, neg = j.get('pos'), j.get('neg')
        except Exception:
            proto, neg = None, None
    if proto is None:
        proto = fit_prototypes(lab, seed_px | (strict & near_seed), kw['proto_k'], 60, 'pos')
    if proto is None:
        proto = fit_prototypes(lab, M, kw['proto_k'], 60, 'pos')   # 种子太少时退回整幅绳色像素
    # 负原型：像绳子但不是绳子的三类像素——同色大块 / 灰暗细线 / 远离带状结构的饱和像素
    dtM = ndimage.distance_transform_edt(M)
    neg_fat = M & (2.0 * dtM > thin_width)
    neg_gray = L & ~strict
    neg_far = M & ~(cv2.dilate(core.astype(np.uint8), ELL(roi_r)) > 0)
    if neg is None:
        neg = fit_prototypes(lab, neg_fat | neg_gray | neg_far, 3, 400, 'neg')
    if pf and not (kw.get('proto_reuse') and os.path.exists(pf)):
        try:                                     # 存正负两栏，给下一张图复用
            with io.open(pf, 'w', encoding='utf-8') as fh:
                json.dump({'pos': proto, 'neg': neg}, fh, ensure_ascii=False, indent=1)
        except Exception:
            pass
    out.update(r1=r1, st=st, adj=build_rag(labels, n, kw['rag_radius']), core=core, roi=roi,
               proto=proto, neg=neg, neg_px=int((neg_fat | neg_gray | neg_far).sum()),
               score=(proto_dist_map(lab, proto) if proto else None))
    if out['score'] is not None:
        out['bscore'] = block_median_1d(out['score'], labels, n)
    if neg is not None:
        out['negscore'] = proto_dist_map(lab, neg)
    return out


def body_surface_model(hsv, barrier, kw):
    """皮肤模型 → 体表区域（皮肤 ∪ 潮红 ∪ 皮肤上的阴影）：自动的"确定非绳子"种子。

    为什么这条能治纯颜色判据治不了的"大腿潮红"：潮红是画在皮肤上的红，和皮肤之间**没有描边**，
    所以它和皮肤落在同一个"被描边围住的面块"里；从皮肤种子在面块**内部**做连通生长，
    潮红就被一起吞进体表 ⇒ 自动成了负种子。绳子不会被吞，因为绳身有自己的描边/强梯度边界，
    面块在绳子的轮廓处断开（沿面块内部扩张＝可以；靠描边连通性跳出去＝不可以，两件事不矛盾）。
    """
    hue_r = ring_hue_dist(hsv[..., 0], kw['hue'])
    skin_win = ((hsv[..., 0] >= kw['skin_hue_min']) & (hsv[..., 0] <= kw['skin_hue_max'])
                & (hsv[..., 1] >= kw['skin_s_min']) & (hsv[..., 1] <= kw['skin_s_max'])
                & (hsv[..., 2] >= kw['skin_v_min']))
    red_win = hue_r <= kw['skin_red_half']
    perm = ((skin_win | red_win) & (hsv[..., 1] >= kw['skin_perm_s'])
            & (hsv[..., 2] >= kw['skin_perm_v'])) & ~barrier
    lb, nb = ndimage.label(perm, structure=E8)
    body = np.zeros(hsv.shape[:2], bool)
    if nb == 0:
        return skin_win, body
    sizes = np.bincount(lb.ravel(), minlength=nb + 1)
    share = np.bincount(lb[skin_win].ravel(), minlength=nb + 1) / np.maximum(sizes, 1)
    body = np.isin(lb, np.where((np.arange(nb + 1) > 0) & (share >= kw['skin_share']))[0])
    # 再跨一条弱边界长一步：潮红块/紧贴皮肤的阴影块一块也不能漏
    if kw['skin_grow'] and body.any():
        adj = build_rag(lb, nb, kw['rag_radius'])
        med_h = median_per_label(lb, nb, hsv[..., 0])
        med_s = median_per_label(lb, nb, hsv[..., 1])
        med_v = median_per_label(lb, nb, hsv[..., 2])
        add = set()
        for c in np.unique(lb[body]):
            if c <= 0:
                continue
            for y, _ in adj.get(int(c), []):
                ok_skin = (kw['skin_hue_min'] - 4 <= med_h[y] <= kw['skin_hue_max'] + 6
                           and med_s[y] >= kw['skin_perm_s'] and med_v[y] >= kw['skin_perm_v'])
                ok_red = (ring_hue_dist(med_h[y], kw['hue']) <= kw['skin_red_half']
                          and med_s[y] >= kw['skin_perm_s'])
                if ok_skin or ok_red:
                    add.add(int(y))
        if add:
            body = body | np.isin(lb, list(add))
    return skin_win, body


# --------------------------------------------------------------------------- 主判定

def rope_mask(img, **kw):
    """返回 (mask_u8, conf_u8, L_u8, 描边判定可视化, info)。"""
    h, w = img.shape[:2]
    s = build_seeds(img, kw)
    hsv, lab, L, labels, n, r1 = s['hsv'], s['lab'], s['L'], s['labels'], s['n'], s['r1']
    info = {'s_min': kw['s_min'], 'v_min': kw['v_min'], 'hue': kw['hue'], 'half': kw['half'],
            'thin_width': round(s['thin_width'], 1), 'line_width': round(s['line_width'], 1),
            'coverage': 0.0, 'n_comp': 0, 'n_comp_weak': 0, 'skel_len_max': 0, 'mean_width': 0.0,
            'n_region': int(n), 'n_region_rope_r1': int(r1.sum()), 'n_region_rope': 0,
            'rope_px': 0, 'stroke_px': 0, 'roi_ratio': 0.0, 'core_px': int(s['core'].sum()),
            'core_seg_missed': 0, 'kpi_est_strokes': 0, 'region_growth': [], 'proto': None,
            'neg_px': s.get('neg_px', 0), 'body_ratio': 0.0, 'body_color_fp_px': 0,
            'body_removed_px': 0}
    if n == 0:
        return (np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8),
                (L * 255).astype(np.uint8), img.copy(), info)
    st, adj, core, roi = s['st'], s['adj'], s['core'], s['roi']
    score, bscore, proto, neg, negscore = (s['score'], s.get('bscore'), s['proto'],
                                           s['neg'], s.get('negscore'))
    thin_width = s['thin_width']
    info['roi_ratio'] = round(float(roi.mean()), 4)
    info['proto'] = ({'k': proto['k'],
                      'centers': [[round(v, 1) for v in c] for c in proto['centers']],
                      'neg_k': (neg['k'] if neg else 0)}
                     if proto else None)
    layer = r1.astype(np.uint8)

    # RAG 生长：治 test(6) 高光/暗部断续。形状 + 颜色分 + ROI 三个门槛一起用，否则
    # "夹在两条绳带之间的大团块"（大腿潮红）会因为邻居够多而被吃进来。
    growth = []
    for rnd in range(max(0, kw['rag_rounds'])):
        added = []
        for c in np.where(layer == 0)[0]:
            if c == 0 or st['area'][c] < kw['min_block_area'] or st['wmax'][c] > thin_width * kw['rag_wmax']:
                continue
            if st['skel_len'][c] / max(st['wmax'][c], 1.0) < kw['min_ratio'] * 0.6:
                continue
            if score is not None and bscore[c] > kw['rag_dist']:
                continue
            if not roi[labels == c].any():
                continue
            if (st['med'][1][c] < kw['rag_sat_frac'] * kw['s_min']
                    or ring_hue_dist(st['med'][0][c], kw['hue']) > kw['half'] * 1.6):
                continue
            k_nb = len([y for y, _ in adj.get(int(c), []) if layer[y] == 1])
            if k_nb < kw['rag_neighbors']:
                continue
            added.append(int(c))
        for c in added:
            layer[c] = 1
        growth.append({'round': rnd + 1, 'added': len(added), 'regions': added[:20]})
        if not added:
            break
    info['region_growth'] = growth
    info['n_region_rope'] = int((layer == 1).sum())
    blocks_px = (layer[labels] > 0) & roi

    st_in = classify_strokes(s, layer, kw, roi, score)
    info.update(st_in['stats'])
    mask = blocks_px | st_in['in']
    info['rope_px'] = int(blocks_px.sum())
    info['stroke_px'] = int(st_in['in'].sum())

    # 像素级补全（多颜色原型 + ROI + 必须连通到已有 mask + 仍然细）：
    # 块级判定只给出"证据在哪"，重绘蒙版要把绳身一起盖上。只在 ROI 内、只补细结构，
    # 所以同色的团块（腿潮红）不会整片进来；残留靠 ROI 半径封顶。
    weak = np.zeros_like(mask)
    if kw['grow_pixels'] and proto is not None and score is not None:
        dtM = ndimage.distance_transform_edt(s['M'])
        cand = (score <= kw['proto_dist'] * kw['grow_strict']) & roi \
            & (2.0 * dtM <= thin_width * kw['grow_thin'])
        lb, nb2 = ndimage.label(cand, structure=E8)
        if nb2:
            touch = np.unique(lb[mask])
            touch = touch[touch > 0]
            if len(touch):
                grow = np.isin(lb, touch)
                weak |= (grow & ~mask)
                mask = mask | grow

    # 体表（皮肤 ∪ 潮红 ∪ 皮肤阴影）＝自动负种子：把"弱证据"里落在体表上的像素剔掉。
    # 强证据（种子块/RAG 块、两侧都是绳子的描边、整条线稿都是绳色的描边）不动——绳子压在
    # 潮红上时，绳身本身就在这些强证据里，不会被体表规则误删。
    body = np.zeros_like(mask)
    if kw['skin']:
        skin_win, body = body_surface_model(hsv, s['barrier'], kw)
        strong = blocks_px | st_in['in_strong']
        wrong = body & (weak | st_in['in_weak']) & ~strong
        info['body_ratio'] = round(float(body.mean()), 4)
        info['body_color_fp_px'] = int((body & s['M']).sum())
        info['body_removed_px'] = int(wrong.sum())
        mask = mask & ~wrong

    # 交互再解算：给了前景/背景笔画就 **重新解算**（不是并差），这样边界会被拉到描边线上
    if kw['grabcut'] or kw['fg_mask'] is not None or kw['bg_mask'] is not None:
        mask = grabcut_resolve(img, mask, core, roi, kw, body)
    if kw['bridge']:
        mask = bridge_gaps(mask, L, hsv, kw['hue'], kw['half'], kw['bridge'],
                           kw['s_min'], kw['v_min'], roi)
    if kw['fill']:
        mask = fill_small_holes(mask, kw['max_hole_frac'] * h * w)
    mask = drop_small(mask, max(1.0, kw['min_comp_frac'] * h * w))
    # 不做无条件闭运算：只靠定向桥接，避免把两根相邻绳粘成一根、抹掉窄缝

    info.update(measure(mask, core, roi, info))
    info['body_vis'] = body
    conf = confidence_map(score, mask, roi, proto, kw['proto_dist'], kw['conf_blur'])
    return ((mask.astype(np.uint8) * 255), conf, (L.astype(np.uint8) * 255), st_in['vis'], info)


def measure(mask, core, roi, info):
    """人手成本相关的指标：连通块数 / 无核心支撑的块数 / 最长骨架 / 核心漏检段数。"""
    h, w = mask.shape
    lab2, n2 = ndimage.label(mask, structure=E8)
    sizes = np.bincount(lab2.ravel(), minlength=n2 + 1) if n2 else np.zeros(1)
    skel = cv2.ximgproc.thinning((mask * 255).astype(np.uint8)) > 0
    slen = np.bincount(lab2[skel].ravel(), minlength=n2 + 1) if n2 else np.zeros(1)
    core_share = np.bincount(lab2[core].ravel(), minlength=n2 + 1) / np.maximum(sizes, 1)
    weak = int(sum(1 for i in range(1, n2 + 1) if sizes[i] and core_share[i] < 0.2)) if n2 else 0
    # 漏检段数：高置信种子的连通块里，被 mask 覆盖不足一半的段数（≈人还要补几笔）
    cl, nc = ndimage.label(core, structure=E8)
    missed = 0
    if nc:
        cs = np.bincount(cl.ravel(), minlength=nc + 1)
        cov = np.bincount(cl[mask].ravel(), minlength=nc + 1) / np.maximum(cs, 1)
        missed = int(sum(1 for i in range(1, nc + 1) if cs[i] > 0 and cov[i] < 0.5))
    return {'coverage': float(mask.mean()), 'n_comp': int(n2), 'n_comp_weak': weak,
            'skel_len_max': int(slen[1:].max()) if n2 and len(slen) > 1 else 0,
            'mean_width': float(mask.sum() / max(1, skel.sum())),
            'core_covered': round(float(mask[core].mean()) if core.any() else 0.0, 4),
            'core_seg_missed': missed, 'kpi_est_strokes': weak + missed}


def confidence_map(score, mask, roi, proto, proto_dist, blur):
    """置信度图：像素级颜色分（到最近正原型的距离）× ROI × mask 邻域软衰减，再羽化。

    二值 mask 是"判定结果"，conf 保留亚阈值证据与软边，二者故意分开存：
    重绘时直接用 conf 当软蒙版（SetLatentNoiseMask），不用再自己做羽化。
    """
    h, w = mask.shape
    if proto is not None and score is not None:
        c = np.clip(1.0 - score / max(1e-6, proto_dist * 1.5), 0.0, 1.0).astype(np.float32)
    else:
        c = np.zeros((h, w), np.float32)
    c = c * roi
    d_out = ndimage.distance_transform_edt(~mask)
    soft = np.clip(1.0 - d_out / 3.0, 0.0, 1.0).astype(np.float32)
    conf = np.maximum(c * 0.85, soft)
    if blur:
        conf = cv2.GaussianBlur(conf, (0, 0), blur)
    conf[mask] = 1.0
    return (np.clip(conf, 0, 1) * 255).astype(np.uint8)


def load_mask(path, shape):
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def grabcut_resolve(img, mask_auto, core, roi, kw, body=None):
    """ROI 内用 grabCut 重新解算分割（GC_INIT_WITH_MASK）。

    前景＝用户 --fg-mask（确定是绳子）+ 自动高置信核心；背景＝用户 --bg-mask（确定不是）；
    自动候选＝GC_PR_FGD；ROI 内其余＝GC_PR_BGD。**输出是重新解算的结果**而不是把笔画
    并/差到旧 mask 上：并差法边界脏，解算才会把边界拉到真正的描边/色阶处。
    注意 color-only 的 GMM 分不开同色的腿潮红，所以结果仍要 ROI 卡住、仍要形状先验兜着。
    """
    fg = load_mask(kw.get('fg_mask'), img.shape[:2])
    bg = load_mask(kw.get('bg_mask'), img.shape[:2])
    gc = np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
    gc[roi] = cv2.GC_PR_BGD
    gc[core] = cv2.GC_FGD
    auto = mask_auto | core
    gc[auto] = np.where(gc[auto] == cv2.GC_FGD, cv2.GC_FGD, cv2.GC_PR_FGD)
    if fg is not None:
        gc[fg] = cv2.GC_FGD
        gc[cv2.dilate(fg.astype(np.uint8), ELL(3)) > 0] = np.where(
            gc[cv2.dilate(fg.astype(np.uint8), ELL(3)) > 0] == cv2.GC_BGD, cv2.GC_PR_FGD,
            gc[cv2.dilate(fg.astype(np.uint8), ELL(3)) > 0])
    if bg is not None:
        gc[bg] = cv2.GC_BGD
    if body is not None and kw.get('skin'):
        gc[body & (gc != cv2.GC_FGD)] = cv2.GC_BGD     # 体表＝自动确定背景（潮红在里头）
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), kw['grabcut_iters'], cv2.GC_INIT_WITH_MASK)
    except Exception:
        return mask_auto
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & (roi | (fg if fg is not None else False))
    if bg is not None:
        out &= ~bg
    if fg is not None:
        out |= fg                       # 用户明说是绳子的地方，解算不许丢
    return out


def classify_strokes(s, layer, kw, roi, score):
    """逐个描边块判定：两侧都是绳子 ⇒ 纳入；否则看它自己的颜色是不是绳色。

    这段是"描边绝不当通道"的落点：判据只看这块描边**自己**的颜色、以及它**紧邻**的块，
    没有任何一步从描边走向描边（不做描边连通域传播），所以一条灰缝线不可能把 mask 引到别处。

    为什么先按"绳色像素"切片再判，而不是拿整个 L 连通块判：线稿在这些图上是**一个大连通块**
    （人物轮廓+沙发绗缝+阴影线全连着，实测 74k~77k px），中位色恰好压在绳色边缘
    （H9~13 / S102~125 / V122~210），整块判一次会吃进 7~10% 画面。这些大连通块的
    "绳色像素占比"只有 0.2~0.5，而真绳描边接近 1.0，所以先切片再逐片找证据。
    """
    img, hsv, L, labels, n = s.get('img', None), s['hsv'], s['L'], s['labels'], s['n']
    sl, ns = ndimage.label(L, structure=E8)
    vis = (img.copy() if img is not None else np.zeros(L.shape + (3,), np.uint8))
    out = np.zeros(L.shape, bool)
    out_strong = np.zeros(L.shape, bool)
    out_weak = np.zeros(L.shape, bool)
    stats = {'stroke_total': 0, 'stroke_in': 0, 'stroke_in_rope_color': 0,
             'stroke_in_twosided': 0, 'stroke_out': 0, 'stroke_in_nonrope_share': 0.0,
             'stroke_frag_in': 0, 'stroke_in_huejump': 0, 'stroke_in_local': 0,
             'stroke_in_free': 0}
    if ns == 0:
        return {'in': out, 'stats': stats, 'vis': vis}
    sizes = np.bincount(sl.ravel())
    comps = [c for c in np.where(sizes >= 4)[0] if c > 0]
    stats['stroke_total'] = len(comps)
    Lr = L & s['strict']
    parent_frac = np.bincount(sl[Lr].ravel(), minlength=ns + 1) / np.maximum(sizes, 1)
    fr, nf = ndimage.label(Lr, structure=E8)
    if nf == 0:
        vis[L] = (0.35 * vis[L] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        return {'in': out, 'stats': stats, 'vis': vis}
    ext = label_extent(labels, n, kw['rag_radius'] + 1)
    objs = ndimage.find_objects(fr)
    fsz = np.bincount(fr.ravel())
    rope_flag = layer > 0
    nz = fr.ravel() > 0
    fl, pl = fr.ravel()[nz], sl.ravel()[nz]
    o = np.argsort(fl, kind='stable')
    fl, pl = fl[o], pl[o]
    bounds = np.searchsorted(fl, np.arange(1, nf + 2))
    mu = np.zeros(nf + 1)
    for i in range(1, nf + 1):
        if bounds[i] > bounds[i - 1]:
            vals, cnts = np.unique(pl[bounds[i - 1]:bounds[i]], return_counts=True)
            mu[i] = vals[cnts.argmax()]
    for i in range(1, nf + 1):
        if fsz[i] < kw['min_stroke_area']:
            continue
        sm = fr == i
        nb = np.unique(ext[sm])
        nb = nb[nb > 0].astype(np.int32)
        k_rope = int(rope_flag[nb].sum()) if len(nb) else 0
        if k_rope >= 2:                       # 两侧都是绳子：绳子自己的轮廓/内部绞线
            out |= sm
            out_strong |= sm
            stats['stroke_frag_in'] += 1
            continue
        pf = parent_frac[int(mu[i])] if mu[i] > 0 else 0.0
        jump = jump_ok(block_contrast(hsv, fr, i, objs[i - 1], kw['ann_in'], kw['ann_out']),
                       kw['contrast_s'], kw['contrast_hue'], kw['contrast_v'])
        in_roi = bool(roi[sm].any())
        col_ok = True if score is None else float(np.median(score[sm])) <= kw['proto_dist']
        if pf >= kw['stroke_frac']:           # 整条线稿几乎全是绳色 ⇒ 它本来就是绳子
            out |= sm
            out_strong |= sm
            stats['stroke_frag_in'] += 1
            stats['stroke_in_local'] += 1
        elif jump and k_rope >= 1 and in_roi:
            out |= sm
            out_weak |= sm
            stats['stroke_frag_in'] += 1
            stats['stroke_in_huejump'] += 1
        elif jump and kw['stroke_free'] and in_roi and col_ok:
            out |= sm
            out_weak |= sm
            stats['stroke_frag_in'] += 1
            stats['stroke_in_free'] += 1
    eaten = np.bincount(sl[out].ravel(), minlength=ns + 1) / np.maximum(sizes, 1)
    stats['stroke_in'] = int(sum(1 for c in comps if eaten[c] >= 0.5))
    stats['stroke_out'] = stats['stroke_total'] - stats['stroke_in']
    col_comp = np.zeros(ns + 1, bool)
    col_comp[1:] = parent_frac[1:] >= 0.5
    stats['stroke_in_rope_color'] = int(sum(1 for c in comps if eaten[c] >= 0.5 and col_comp[c]))
    stats['stroke_in_twosided'] = stats['stroke_in'] - stats['stroke_in_rope_color']
    stats['stroke_in_nonrope_share'] = round(
        stats['stroke_in_twosided'] / max(1, stats['stroke_in']), 3)
    vis[out] = (0.35 * vis[out] + 0.65 * np.array([80, 255, 80])).astype(np.uint8)
    other = L & ~out
    vis[other] = (0.35 * vis[other] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    return {'in': out, 'in_strong': out_strong, 'in_weak': out_weak,
            'stats': stats, 'vis': vis}


def bridge_gaps(mask, L, hsv, hue, half, gap, s_min, v_min, roi=None):
    """沿四个方向的定向闭合补断点，只允许在"绳色宽松带 或 描边"（且 ROI 内）上补像素。

    治的是绳子高光/暗部造成的断点：断点像素颜色已经不像绳子（第一轮没收），
    但断口两侧都是绳子时补上是对的。限制 allow 掩膜是为了不让两条平行绳带被焊死。
    """
    loose = ((ring_hue_dist(hsv[..., 0], hue) <= half * 1.2)
             & (hsv[..., 1] >= 0.75 * s_min) & (hsv[..., 2] >= 0.75 * v_min))
    allow = loose | L
    if roi is not None:
        allow = allow & cv2.dilate(roi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    k = max(3, int(gap))
    out = mask.copy()
    for ang in (0, 45, 90, 135):
        kernel = np.zeros((k, k), np.uint8)
        c = k // 2
        if ang == 0:
            kernel[c, :] = 1
        elif ang == 90:
            kernel[:, c] = 1
        elif ang == 45:
            np.fill_diagonal(kernel, 1)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1)
        cl = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        out |= (cl & allow)
    return out


def fill_small_holes(mask, max_hole):
    """只填小孔：绳子绕一圈形成的闭环里是腿/背景，填大孔会把它们整块吃掉。"""
    inv = ~mask
    lab, n = ndimage.label(inv, structure=E8)
    if n == 0:
        return mask
    border = set(np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]).tolist())
    sizes = np.bincount(lab.ravel())
    fill = [k for k in range(1, n + 1) if k not in border and sizes[k] <= max_hole]
    return mask | np.isin(lab, fill) if fill else mask


def drop_small(mask, min_px):
    """只去极小的碎块：不做"只留最大连通域"——交叉/遮挡本来就会断成多块。"""
    lab, n = ndimage.label(mask, structure=E8)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    drop = np.where(sizes < min_px)[0]
    drop = drop[drop > 0]
    return mask & ~np.isin(lab, drop) if len(drop) else mask


# --------------------------------------------------------------------------- QA / 可视化

def analyze_mask(img, mask, **kw):
    """对**任意** mask 用同一套种子口径算 QA（含误检块数/漏检段数），给版本对比用。"""
    s = build_seeds(img, kw)
    info = {'core_px': int(s['core'].sum()),
            'roi_ratio': round(float(s['roi'].mean()), 4)}
    info.update(measure(mask > 0, s['core'], s['roi'], info))
    return info


def overlay(img_bgr, mask, body=None):
    """原图 + 半透明红＝mask + 青＝mask 边界；给了体表就再用蓝＝自动负种子（含潮红）。"""
    vis = img_bgr.copy()
    if body is not None and body.any():
        vis[body] = (0.55 * vis[body] + 0.45 * np.array([255, 120, 0])).astype(np.uint8)
    m = mask > 0
    vis[m] = (0.45 * vis[m] + 0.55 * np.array([60, 60, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def stroke_qa(img, hsv, mask, hue, half, stroke_s_min, stroke_v_min,
              dark_delta, bg_kernel, line_width, min_px=8):
    """独立可复算的"描边误检"指标：被 mask 吃掉的描边块里，有多少不是绳色。

    单独成函数是为了能对**任意** mask（包括上一版输出）算同一个数，前后比较才成立。
    """
    L = stroke_layer(img, dark_delta, bg_kernel, line_width, 4)
    sl, ns = ndimage.label(L, structure=E8)
    if ns == 0:
        return {'stroke_total': 0, 'stroke_in': 0, 'stroke_in_rope_color': 0,
                'stroke_in_nonrope': 0, 'stroke_in_nonrope_share': 0.0}
    sizes = np.bincount(sl.ravel())
    tot = inn = col = 0
    for c in np.where(sizes >= min_px)[0]:
        if c == 0:
            continue
        sm = sl == c
        tot += 1
        if mask[sm].mean() >= 0.5:
            inn += 1
            if (ring_hue_dist(np.median(hsv[..., 0][sm]), hue) <= half
                    and np.median(hsv[..., 1][sm]) >= stroke_s_min
                    and np.median(hsv[..., 2][sm]) >= stroke_v_min):
                col += 1
    return {'stroke_total': tot, 'stroke_in': inn, 'stroke_in_rope_color': col,
            'stroke_in_nonrope': inn - col,
            'stroke_in_nonrope_share': round((inn - col) / max(1, inn), 3)}


def hue_peaks(hsv, topk=4, s_pct=80, smooth=9, min_dist=10):
    """饱和度加权色相直方图峰值，仅作 --hue auto 的候选来源。"""
    h = hsv[..., 0].astype(np.int32)
    strong = (hsv[..., 1] >= np.percentile(hsv[..., 1], s_pct)) & (hsv[..., 2] > 40)
    hist = np.bincount(h[strong].ravel(), minlength=HUE_PERIOD).astype(np.float64)
    hist = np.r_[hist, hist]
    k = np.ones(smooth) / smooth
    pad = smooth // 2
    hist = np.convolve(np.r_[hist[-pad:], hist, hist[:pad]], k, 'same')[pad:-pad]
    hist = hist[:HUE_PERIOD] + hist[HUE_PERIOD:]
    peaks = [i for i in range(HUE_PERIOD)
             if hist[i] == hist[max(0, i - min_dist):i + min_dist + 1].max() and hist[i] > 0]
    peaks.sort(key=lambda i: -hist[i])
    return peaks[:topk]


def rope_likeness(mask_u8, hue, cov_min, cov_max, cdist_red=25):
    """像不像绳子：覆盖率合理 + 细长 + 连通块不多 + 偏红加成（--hue auto 用）。"""
    m = mask_u8 > 0
    cov = float(m.mean())
    if not (cov_min <= cov <= cov_max):
        return -1.0, {'coverage': round(cov, 4), 'reason': 'coverage out of window'}
    skel = max(int((cv2.ximgproc.thinning((m * 255).astype(np.uint8)) > 0).sum()), 1)
    width = int(m.sum()) / skel
    ncomp = int(ndimage.label(m, structure=E8)[1])
    thin = float(np.clip(1.0 - (width - 6.0) / 30.0, 0.0, 1.0))
    few = float(np.clip(1.0 - (ncomp - 1) / 12.0, 0.0, 1.0))
    red = 0.25 if ring_hue_dist(np.array([hue]), RED_HUE)[0] <= cdist_red else 0.0
    return thin * few + red, {'coverage': round(cov, 4), 'width_px': round(width, 1),
                              'n_comp': ncomp, 'red_bonus': red}


# --------------------------------------------------------------------------- CLI

def build_parser():
    ap = argparse.ArgumentParser(description='绳子 mask 的传统 CV 预标注（面块/描边分层 + ROI）')
    ap.add_argument('--src', required=True, help='图片路径 / 目录 / glob')
    ap.add_argument('--out', required=True)
    ap.add_argument('--hue', default='0', help='色相 0~179，或 auto')
    ap.add_argument('--half', type=int, default=14, help='色相窗口半宽（14 实测：18 会吃进暗金头发）')
    ap.add_argument('--s-min', default=120, help='绳色像素饱和度下限（块里的像素级门槛）')
    ap.add_argument('--v-min', default=70,
                    help='明度下限：极暗像素的 H/S 是退化的（近黑棕在 HSV 里读成高饱和红）')
    ap.add_argument('--thin-width', default='auto', help='绳子最大内切直径上限，auto＝0.035×短边')
    ap.add_argument('--line-width', default='auto', help='描边细度上限，auto＝0.012×短边')
    ap.add_argument('--dark-delta', type=int, default=26, help='比局部背景暗多少算暗脊')
    ap.add_argument('--bg-kernel', type=int, default=15, help='估局部背景亮度的闭运算核')
    ap.add_argument('--dilate', type=int, default=1, help='描边层膨胀次数＝边界厚度')
    ap.add_argument('--min-block-area', type=int, default=40)
    ap.add_argument('--min-ratio', type=float, default=3.0, help='块骨架长/最大内切直径下限')
    ap.add_argument('--contrast-s', type=float, default=1.4, help='种子：块/紧邻邻域饱和度比下限')
    ap.add_argument('--contrast-hue', type=float, default=10.0, help='种子：块/紧邻邻域色相差下限')
    ap.add_argument('--contrast-v', type=float, default=0.8, help='种子：块/紧邻邻域明度比上限')
    ap.add_argument('--ann-in', type=int, default=3, help='邻域内半径')
    ap.add_argument('--ann-out', type=int, default=9, help='邻域外半径')
    ap.add_argument('--rag-rounds', type=int, default=2)
    ap.add_argument('--rag-neighbors', type=int, default=2, help='纳入所需"已是绳子"的邻居数')
    ap.add_argument('--rag-radius', type=int, default=5, help='跨描边建边的最大距离')
    ap.add_argument('--rag-dist', type=float, default=3.0, help='到绳色原型的距离上限（块级）')
    ap.add_argument('--rag-wmax', type=float, default=1.5, help='生长允许的粗细上限倍数')
    ap.add_argument('--proto-k', type=int, default=3, help='正原型簇数（受光/主体/阴影）')
    ap.add_argument('--proto-dist', type=float, default=3.0, help='像素到最近原型的距离上限')
    ap.add_argument('--proto-json', default='', help='正负原型的 json（跨图复用载体）')
    ap.add_argument('--proto-reuse', type=int, default=0, help='1=读已有原型 json，不重估')
    ap.add_argument('--stroke-s-min', type=float, default=None, help='默认 0.85×--s-min')
    ap.add_argument('--stroke-v-min', type=float, default=None, help='默认跟随 --v-min')
    ap.add_argument('--stroke-frac', type=float, default=0.75,
                    help='父描边连通块里绳色像素占比达到它就认定"整条线就是绳子"')
    ap.add_argument('--min-stroke-area', type=int, default=6)
    ap.add_argument('--stroke-free', type=int, default=1,
                    help='1=允许 ROI 内"不挨着绳块的绳色描边片"（保阴影段，稍脏）')
    ap.add_argument('--roi-factor', type=float, default=0.05, help='ROI 膨胀半径＝factor×短边')
    ap.add_argument('--roi-min', type=int, default=12)
    ap.add_argument('--roi-max', type=int, default=120)
    ap.add_argument('--grow-pixels', type=int, default=1,
                    help='1=用多颜色原型在 ROI 内把绳身补全（只保留与已有 mask 连通的成分）')
    ap.add_argument('--grow-strict', type=float, default=0.75, help='像素补全的原型距离缩放')
    ap.add_argument('--grow-thin', type=float, default=1.2, help='像素补全的粗细上限倍数')
    ap.add_argument('--fg-mask', default='', help='人确定"是绳子"的图（R≠0 即前景），触发重新解算')
    ap.add_argument('--bg-mask', default='', help='人确定"不是绳子"的图，触发重新解算')
    ap.add_argument('--grabcut', type=int, default=0, help='1=ROI 内无条件跑一次 grabCut 修边界')
    ap.add_argument('--grabcut-iters', type=int, default=3)
    ap.add_argument('--skin', type=int, default=1,
                    help='1=估皮肤模型并生成"体表"自动负种子（含潮红），剔除落在体表上的弱证据')
    ap.add_argument('--skin-hue-min', type=int, default=5)
    ap.add_argument('--skin-hue-max', type=int, default=30)
    ap.add_argument('--skin-s-min', type=int, default=45, help='皮肤色板饱和度下限')
    ap.add_argument('--skin-s-max', type=int, default=165, help='皮肤色板饱和度上限')
    ap.add_argument('--skin-v-min', type=int, default=140, help='皮肤色板亮度下限（皮肤是亮的）')
    ap.add_argument('--skin-red-half', type=int, default=28, help='体表连通用：绳色环的放宽半宽')
    ap.add_argument('--skin-perm-s', type=int, default=60,
                    help='体表连通用：饱和度下限（压低可防奶油色沙发被并进体表）')
    ap.add_argument('--skin-perm-v', type=int, default=60, help='体表连通用：明度下限')
    ap.add_argument('--skin-share', type=float, default=0.15,
                    help='面块里皮肤像素占比达到它才算体表（潮红块与皮肤同块，会一起被吞）')
    ap.add_argument('--skin-grow', type=int, default=1, help='1=体表再跨一条弱边界长一步')
    ap.add_argument('--bridge', type=int, default=7, help='沿四方向补断点的最大长度；0=关')
    ap.add_argument('--min-comp-frac', type=float, default=2e-4)
    ap.add_argument('--max-hole-frac', type=float, default=2e-3)
    ap.add_argument('--no-fill', action='store_true')
    ap.add_argument('--conf-blur', type=float, default=2.0, help='置信度图羽化 sigma')
    ap.add_argument('--cov-min', type=float, default=0.003, help='--hue auto 的覆盖率接受窗')
    ap.add_argument('--cov-max', type=float, default=0.15)
    ap.add_argument('--flag-cov-max', type=float, default=0.20, help='超过就建议人工从头标')
    ap.add_argument('--flag-cov-min', type=float, default=0.002)
    ap.add_argument('--flag-comp-max', type=int, default=200)
    ap.add_argument('--prefix', default='v3_')
    ap.add_argument('--save-layers', action='store_true', help='额外存描边判定图/负证据图')
    return ap


def kw_from_args(a, tw, lw, pf):
    return dict(hue=int(a.hue) if a.hue != 'auto' else RED_HUE, half=a.half,
                s_min=float(a.s_min), v_min=a.v_min, thin_width=tw, line_width=lw,
                dark_delta=a.dark_delta, bg_kernel=a.bg_kernel, dilate=a.dilate,
                min_block_area=a.min_block_area, min_ratio=a.min_ratio,
                contrast_s=a.contrast_s, contrast_hue=a.contrast_hue, contrast_v=a.contrast_v,
                ann_in=a.ann_in, ann_out=a.ann_out, rag_rounds=a.rag_rounds,
                rag_neighbors=a.rag_neighbors, rag_radius=a.rag_radius, rag_dist=a.rag_dist,
                rag_wmax=a.rag_wmax, rag_sat_frac=0.55, proto_k=a.proto_k,
                proto_dist=a.proto_dist,
                proto_file=pf, proto_reuse=a.proto_reuse, stroke_s_min=a.stroke_s_min,
                stroke_v_min=a.stroke_v_min, stroke_frac=a.stroke_frac,
                min_stroke_area=a.min_stroke_area, stroke_free=a.stroke_free,
                roi_factor=a.roi_factor, roi_min=a.roi_min, roi_max=a.roi_max,
                grow_pixels=a.grow_pixels, grow_strict=a.grow_strict, grow_thin=a.grow_thin,
                fg_mask=a.fg_mask or None, bg_mask=a.bg_mask or None, grabcut=a.grabcut,
                grabcut_iters=a.grabcut_iters, min_comp_frac=a.min_comp_frac,
                max_hole_frac=a.max_hole_frac, bridge=a.bridge, fill=not a.no_fill,
                conf_blur=a.conf_blur, skin=a.skin, skin_hue_min=a.skin_hue_min,
                skin_hue_max=a.skin_hue_max, skin_s_min=a.skin_s_min,
                skin_s_max=a.skin_s_max, skin_v_min=a.skin_v_min,
                skin_red_half=a.skin_red_half, skin_perm_s=a.skin_perm_s,
                skin_perm_v=a.skin_perm_v, skin_share=a.skin_share, skin_grow=a.skin_grow)


def resolve_files(src):
    if any(c in src for c in '*?['):
        return sorted(glob.glob(src))
    if os.path.isdir(src):
        return sorted(glob.glob(os.path.join(src, '*.png')) + glob.glob(os.path.join(src, '*.jpg')))
    return [src]


def main():
    a = build_parser().parse_args()
    os.makedirs(a.out, exist_ok=True)
    tw = None if a.thin_width == 'auto' else float(a.thin_width)
    lw = None if a.line_width == 'auto' else float(a.line_width)
    pf = a.proto_json or os.path.join(a.out, a.prefix + 'rope_color_prior.json')
    summary = []
    for fp in resolve_files(a.src):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print('读取失败: %s' % fp, flush=True)
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')
        rec = {'image': os.path.basename(fp), 'size': '%dx%d' % (w, h), 'tried': []}
        if a.hue == 'auto':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cand = hue_peaks(hsv)
            for f in (RED_HUE, (RED_HUE - 5) % HUE_PERIOD, (RED_HUE + 5) % HUE_PERIOD):
                if f not in cand:
                    cand.append(f)
            rec['hue_candidates'] = cand
            best = None
            for hue in cand:
                kw = kw_from_args(a, tw, lw, pf)
                kw['hue'] = hue
                m, cf, l, svv, info = rope_mask(img, **kw)
                sc, det = rope_likeness(m, hue, a.cov_min, a.cov_max)
                rec['tried'].append(dict(hue=hue, score=round(sc, 3), **det))
                if sc > 0 and (best is None or sc > best[0]):
                    best = (sc, m, cf, l, svv, info)
            if best is None:                       # 全被否掉 → 回落到红色系
                kw = kw_from_args(a, tw, lw, pf)
                kw['hue'] = int(RED_HUE)
                m, cf, l, svv, info = rope_mask(img, **kw)
                rec['auto_fallback'] = 'no candidate passed; fell back to red hue 0'
                best = (0.0, m, cf, l, svv, info)
        else:
            kw = kw_from_args(a, tw, lw, pf)
            m, cf, l, svv, info = rope_mask(img, **kw)
            sc, det = rope_likeness(m, int(a.hue), a.cov_min, a.cov_max)
            rec['tried'].append(dict(hue=int(a.hue), score=round(sc, 3), **det))
            best = (sc, m, cf, l, svv, info)

        score, mask, conf, L, svis, info = best
        body = info.pop('body_vis', None)
        cv2.imwrite(os.path.join(a.out, a.prefix + base + '_mask.png'), mask)
        cv2.imwrite(os.path.join(a.out, a.prefix + base + '_conf.png'), conf)
        cv2.imwrite(os.path.join(a.out, a.prefix + base + '_overlay.png'), overlay(img, mask))
        cv2.imwrite(os.path.join(a.out, a.prefix + base + '_body.png'), overlay(img, mask, body))
        cv2.imwrite(os.path.join(a.out, a.prefix + base + '_L.png'), L)
        if a.save_layers:
            cv2.imwrite(os.path.join(a.out, a.prefix + base + '_strokes.png'), svis)
        qa = stroke_qa(img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV), mask > 0, info['hue'],
                       info['half'], info['s_min'] * 0.85, info['v_min'], a.dark_delta,
                       a.bg_kernel, info['line_width'])
        info['stroke_qa'] = qa
        rec.update(picked_hue=info['hue'], score=round(score, 3), **info)
        cov = info['coverage']
        flags = []
        if cov > a.flag_cov_max:
            flags.append('覆盖率>%.0f%% 建议人工从头标' % (100 * a.flag_cov_max))
        if cov < a.flag_cov_min:
            flags.append('覆盖率<%.1f%% 可能整条漏检' % (100 * a.flag_cov_min))
        if info['n_comp'] > a.flag_comp_max:
            flags.append('区域数>%d 过于碎片化' % a.flag_comp_max)
        rec['flags'] = flags
        summary.append(rec)
        print('[%s] 覆盖率=%.3f%% ROI=%.1f%% 块=%d(弱%d) 漏段=%d 预计笔数≈%d | 描边 入%d/%d 非绳色%d(%.2f) %s'
              % (rec['image'], 100 * cov, 100 * info['roi_ratio'], info['n_comp'],
                 info['n_comp_weak'], info['core_seg_missed'], info['kpi_est_strokes'],
                 qa['stroke_in'], qa['stroke_total'], qa['stroke_in_nonrope'],
                 qa['stroke_in_nonrope_share'],
                 ('  <-- ' + '; '.join(flags)) if flags else ''), flush=True)

    with io.open(os.path.join(a.out, a.prefix + 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump({'params': vars(a), 'images': summary}, fh, ensure_ascii=False, indent=1)
    print('完成 %d 张 → %s' % (len(summary), a.out))


if __name__ == '__main__':
    main()
