"""合理化嵌入（harmonization）：把"补上去的一块"揉进底图，让它不像贴的。

输入：BASE（底图）、PATCH（要嵌进去的层，如程序画的绳/UI/Blender 件）、MASK（**BASE 上的区域**，白=要放 patch 的地方）
输出：OUT（合成图）＋ 一行可测报告（接缝阶梯、均值/标准差、噪声、锐度）

做法（全部是 PS 里那几招的可编程等价物，顺序也照 PS）：
  ① **收缩＋羽化** mask（`--shrink` 防边缘带出原图的杂色＝去 halo；`--feather` 防硬边）
  ② **局部配色**：统计 BASE 上 mask **外圈环带**的 LAB 均值/标准差，把 PATCH 映射过去
     （`--match lab` 三通道；`--match luma` 只配亮度、保留补块自身色彩；`--match hist` 直方图匹配）
  ③ **锐度匹配**：比较环带与补块的高频能量（Laplacian 方差），用 unsharp/高斯把补块调到同一水平
  ④ **颗粒匹配**：比较环带与补块的高频残差 σ，补上差值的高斯噪声（AI 图/JPEG 都有底噪）
  ⑤ **接触阴影/AO**：沿 mask 边缘向外一圈按 `--ao` 压暗（可选 `--ao-dir` 偏向光来的反方向）
  ⑥ **合成**：OUT = BASE*(1-w) + PATCH'*w

用法例：
  python harmonize.py base.png patch.png --mask m.png --out o.png --match lab --report
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


# ----------------------------------------------------------------- 工具
def _lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def hf_energy(gray):
    """高频能量：Laplacian 方差（锐度）＋高频残差 σ（噪声）。"""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    resid = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    return float(lap.var()), float(resid.std())


def noise_sigma(gray, sel=None, sigma_blur=1.2):
    """稳健的底噪估计：只在**平坦**像素上算高频残差的 σ（用梯度最小的 25% 像素）。

    直接对整块区域取 resid.std() 会把"纹理/边缘"当成噪声、把要补的颗粒量估大很多
    （实测：环带 resid σ=10.96 里大部分是桌腿边缘与明暗交界，不是噪声）。
    """
    g = gray.astype(np.float32)
    resid = g - cv2.GaussianBlur(g, (0, 0), sigma_blur)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    if sel is not None:
        resid, grad = resid[sel], grad[sel]
    if resid.size < 64:
        return float(resid.std()) if resid.size else 0.0
    flat = grad <= np.percentile(grad, 25)
    return float(resid[flat].std()) if flat.any() else 0.0


def ring_mask(mask, grow=12, band=14):
    """mask 外的一圈环带（用来统计"底图在附近长什么样"）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    inner = cv2.dilate(mask, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (grow + band) + 1,) * 2)
    outer = cv2.dilate(mask, k2)
    return cv2.subtract(outer, inner)


def seam_step(base_lab, out_lab, mask, band=3):
    """接缝阶梯：mask 边界内外一圈的 |ΔL| 平均值（越小越"不像贴的"）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    edge = cv2.subtract(cv2.dilate(mask, k), cv2.erode(mask, k))
    d = np.abs(out_lab[..., 0] - base_lab[..., 0])
    return float(d[edge > 0].mean()) if (edge > 0).any() else 0.0


def match_stats(patch_lab, ref_lab, ref_sel, mode='lab'):
    """把 patch 的统计量映射到 ref（ref_sel 是参考像素的 bool 蒙版）。

    mode='lab'    : L/a/b 三通道都配
    mode='chroma' : **只配 a/b（色度）**，L 原样保留 —— 白纸、金属、发丝这类"本来就该比周围亮/暗"的东西
                    只能用这个（把亮度一起配过去＝把白纸染成旧纸，实测接缝反而更差）
    mode='luma'   : 只配 L
    """
    out = patch_lab.copy()
    chans = {'lab': [0, 1, 2], 'chroma': [1, 2], 'luma': [0]}[mode]
    for c in chans:
        p = patch_lab[..., c]
        r = ref_lab[..., c][ref_sel]
        if r.size < 32:
            continue
        ps, pm = p.std() + 1e-6, p.mean()
        rs, rm = r.std() + 1e-6, r.mean()
        scale = float(np.clip(rs / ps, 0.4, 2.5))
        out[..., c] = (p - pm) * scale + rm
    return out


def cast_shadow(base_bgr, mask, amount=0.35, length=12, blur=6.0, direction='down-right'):
    """投影：把 mask 沿一个方向平移出"实心影子"，模糊后**正片叠底**到底图上。

    与 AO 的区别：AO 只压 mask 外沿一圈（无方向），投影是**沿光反向偏移的一整块**形状
    —— 挂墙的画/牌、站在地上的人，缺的都是这个。
    """
    if amount <= 0 or length <= 0:
        return base_bgr
    dx = {'down-right': (int(length * 0.7), int(length)),
          'right': (int(length), 0), 'down': (0, int(length)),
          'down-left': (-int(length * 0.7), int(length))}.get(direction, (int(length * 0.7), int(length)))
    M = np.float32([[1, 0, dx[0]], [0, 1, dx[1]]])
    sh = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    sh = cv2.subtract(sh, mask)                       # 别压到物件自己身上
    if blur > 0:
        sh = cv2.GaussianBlur(sh.astype(np.float32) / 255.0, (0, 0), blur)
    else:
        sh = sh.astype(np.float32) / 255.0
    factor = 1.0 - amount * np.clip(sh, 0, 1)
    out = base_bgr.astype(np.float32) * factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def match_hist(patch_lab, ref_lab, ref_sel):
    """逐通道直方图匹配（平涂/渐变图往往比 mean-std 更贴）。"""
    out = patch_lab.copy()
    for c in range(3):
        src = patch_lab[..., c]
        ref = ref_lab[..., c][ref_sel]
        s_hist, _ = np.histogram(src, bins=256, range=(0, 255))
        r_hist, _ = np.histogram(ref, bins=256, range=(0, 255))
        s_cdf = np.cumsum(s_hist).astype(np.float64)
        r_cdf = np.cumsum(r_hist).astype(np.float64)
        s_cdf /= s_cdf[-1] + 1e-9
        r_cdf /= r_cdf[-1] + 1e-9
        lut = np.interp(s_cdf, r_cdf, np.arange(256))
        out[..., c] = cv2.LUT(np.clip(src, 0, 255).astype(np.uint8), lut.astype(np.uint8)).astype(np.float32)
    return out


def match_sharpness(patch_bgr, patch_alpha, target_e):
    """把补块的锐度（Laplacian 方差）调到 target_e：不够就 unsharp，过了就轻微高斯。"""
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e, _ = hf_energy(g)
    if e < 1e-6:
        return patch_bgr, 1.0
    ratio = target_e / e
    if 0.85 <= ratio <= 1.2:
        return patch_bgr, 1.0
    if ratio < 0.85:                       # 太锐 ⇒ 轻微模糊
        sigma = float(np.clip(0.35 * (1.0 / max(ratio, 0.2) - 1.0), 0.2, 1.6))
        out = cv2.GaussianBlur(patch_bgr, (0, 0), sigma)
        return out, sigma
    amt = float(np.clip(0.6 * (ratio - 1.0), 0.05, 1.4))   # 太软 ⇒ unsharp
    blur = cv2.GaussianBlur(patch_bgr, (0, 0), 1.0)
    return cv2.addWeighted(patch_bgr, 1 + amt, blur, -amt, 0), -amt


def add_grain(img, sigma):
    if sigma <= 0.6:
        return img, 0.0
    noise = np.random.default_rng(7).normal(0.0, sigma, img.shape[:2]).astype(np.float32)
    out = img.astype(np.float32).copy()
    for c in range(3):
        out[..., c] += noise
    return np.clip(out, 0, 255).astype(np.uint8), sigma


def apply_ao(base_bgr, mask, amount=0.25, width=14, direction=None):
    """沿 mask 边缘向外一圈压暗（接触阴影）。direction: 'left|right|up|down' ⇒ 只在某侧加强。"""
    if amount <= 0:
        return base_bgr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
    outer = cv2.dilate(mask, k).astype(np.float32)
    dist = cv2.distanceTransform((outer > 0).astype(np.uint8), cv2.DIST_L2, 3)
    band = np.clip(1.0 - (dist - 0.0) / float(width), 0.0, 1.0)     # 边缘处 1，外沿 0
    band[outer == 0] = 0.0
    band[mask > 0] = 0.0
    if direction in ('left', 'right', 'up', 'down'):
        h, w = band.shape
        ramp = {'left': np.linspace(1, 0, w)[None, :], 'right': np.linspace(0, 1, w)[None, :],
                'up': np.linspace(1, 0, h)[:, None], 'down': np.linspace(0, 1, h)[:, None]}[direction]
        band = band * ramp.astype(np.float32)
    factor = 1.0 - amount * band
    out = base_bgr.astype(np.float32) * factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- 主流程
def apply_lowfreq(base_bgr, patch_bgr, mask, sigma=60.0, clip=0.15):
    """低频光照匹配：让补块跟着**底图的大尺度明暗场**走（灯在一侧、墙面有渐变时最重要）。

    做法：底图灰度做高斯模糊（sigma 决定"多低频"）得到光照场 → 以 mask 外圈环带的均值为基准，
    把补块的亮度按 (场/基准) 缩放（限幅 clip），于是"靠灯的一侧更亮、离灯的一侧更暗"自动成立。
    """
    if sigma <= 0:
        return patch_bgr
    g = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    field = cv2.GaussianBlur(g, (0, 0), sigma)
    ring = ring_mask(mask) > 0
    if ring.sum() < 64:
        return patch_bgr
    ref = float(field[ring].mean()) + 1e-6
    ratio = np.clip(field / ref, 1.0 - clip, 1.0 + clip)
    out = patch_bgr.astype(np.float32) * ratio[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonize(base_path, patch_path, mask_path, out_path, *, match='lab', feather=6, shrink=1,
              ao=0.0, ao_dir=None, ao_width=14, cast=0.0, cast_len=12, cast_blur=6.0,
              cast_dir='down-right', lowfreq=0.0, lowfreq_clip=0.15,
              grain='auto', grain_max=5.0, sharp=True, report=False):
    base = cv2.imread(base_path, cv2.IMREAD_COLOR)
    patch = cv2.imread(patch_path, cv2.IMREAD_COLOR)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if base is None or patch is None or m is None:
        raise SystemExit('读不到 base/patch/mask')
    h, w = base.shape[:2]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    if patch.shape[:2] != (h, w):
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LANCZOS4)

    mask = (m > 127).astype(np.uint8) * 255
    if shrink > 0:                                     # 去 halo：往里收
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1,) * 2)
        inner = cv2.erode(mask, k)
        if inner.sum() > 0:
            mask = inner
    w_soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(feather / 2.0, 0.1))
    w_soft = np.clip(w_soft, 0, 1)[..., None]

    base_lab, patch_lab = _lab(base), _lab(patch)
    ref_sel = ring_mask(mask) > 0
    pre = seam_step(base_lab, _lab(base * (1 - w_soft) + patch * w_soft), mask)

    # ② 配色
    if match == 'hist':
        patch_lab2 = match_hist(patch_lab, base_lab, ref_sel)
    elif match == 'luma':
        patch_lab2 = match_stats(patch_lab, base_lab, ref_sel, mode='luma')
    elif match == 'chroma':
        patch_lab2 = match_stats(patch_lab, base_lab, ref_sel, mode='chroma')
    else:
        patch_lab2 = match_stats(patch_lab, base_lab, ref_sel, mode='lab')
    patch2 = _from_lab(patch_lab2)

    # ②b 低频光照（底图的大尺度明暗场）——在配色之前贴，之后配色再收色偏
    if lowfreq > 0:
        patch = apply_lowfreq(base, patch, mask, sigma=lowfreq, clip=lowfreq_clip)
        patch_lab = _lab(patch)

    # ③ 锐度匹配
    ref_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring_pix = ring_mask(mask) > 0
    target_e = float(cv2.Laplacian(ref_gray, cv2.CV_32F, ksize=3).var())
    sharp_note = 1.0
    if sharp:
        patch2, sharp_note = match_sharpness(patch2, mask, target_e)

    # ④ 颗粒匹配（用"平坦区稳健估计"，并给补颗粒设上限，别把画面噪化）
    gain_note = 0.0
    if grain != 'none':
        g2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sig_p = noise_sigma(g2, mask > 0)
        sig_b = noise_sigma(ref_gray, ring_pix)
        patch2, gain_note = add_grain(patch2, min(max(sig_b - sig_p, 0.0), grain_max))

    # ⑤ AO / 接触阴影 ＋ 投影
    base_ao = apply_ao(base, mask, amount=ao, width=ao_width, direction=ao_dir)
    base_ao = cast_shadow(base_ao, mask, amount=cast, length=cast_len, blur=cast_blur,
                          direction=cast_dir)

    # ⑥ 合成
    out = (base_ao.astype(np.float32) * (1 - w_soft) + patch2.astype(np.float32) * w_soft)
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)

    out_lab = _lab(out)
    post = seam_step(base_lab, out_lab, mask)
    info = {
        'seam_before': round(pre, 3), 'seam_after': round(post, 3),
        'match': match, 'sharp_factor': round(float(sharp_note), 3), 'grain_sigma': round(float(gain_note), 3),
        'ao': ao, 'feather': feather, 'shrink': shrink, 'cast': cast, 'lowfreq': lowfreq,
    }
    if report:
        print(' '.join('%s=%s' % (k, v) for k, v in info.items()), flush=True)
        print('OUT %s %s' % (out_path, os.path.getsize(out_path)), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('patch')
    ap.add_argument('--mask', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--match', default='lab', choices=['lab', 'chroma', 'luma', 'hist'])
    ap.add_argument('--feather', type=int, default=6)
    ap.add_argument('--shrink', type=int, default=1)
    ap.add_argument('--ao', type=float, default=0.0)
    ap.add_argument('--ao-dir', default=None, choices=[None, 'left', 'right', 'up', 'down'])
    ap.add_argument('--ao-width', type=int, default=14)
    ap.add_argument('--cast', type=float, default=0.0)
    ap.add_argument('--cast-len', type=int, default=12)
    ap.add_argument('--cast-blur', type=float, default=6.0)
    ap.add_argument('--cast-dir', default='down-right',
                    choices=['down-right', 'down', 'right', 'down-left'])
    ap.add_argument('--lowfreq', type=float, default=0.0)
    ap.add_argument('--lowfreq-clip', type=float, default=0.15)
    ap.add_argument('--grain', default='auto')
    ap.add_argument('--grain-max', type=float, default=5.0)
    ap.add_argument('--no-sharp', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    harmonize(a.base, a.patch, a.mask, a.out, match=a.match, feather=a.feather, shrink=a.shrink,
              ao=a.ao, ao_dir=a.ao_dir, ao_width=a.ao_width, cast=a.cast, cast_len=a.cast_len,
              cast_blur=a.cast_blur, cast_dir=a.cast_dir, lowfreq=a.lowfreq,
              lowfreq_clip=a.lowfreq_clip, grain=a.grain, grain_max=a.grain_max,
              sharp=not a.no_sharp, report=a.report)


if __name__ == '__main__':
    main()
