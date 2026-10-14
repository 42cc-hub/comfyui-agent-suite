#!/usr/bin/env python
"""头发 mask 可行性探针 v2：把 SAM 的提议单独铺开看，判断"提议里到底有没有头发"。

v1 的问题：全景图把所有提议叠在一起 + alpha 太淡，肉眼分不清哪个是头发；
粗糙启发式挑出来的候选基本都落在背景/腿/墙面上，说明"选"这一步不可靠，
但并不能回答"SAM 本身能不能把头发切干净"。

v2 做法：
  1) `SamAutomaticMaskGenerator` 出提议，按面积从大到小排；
  2) 先按"贴边率/面积"把明显的背景大块压下去（背景通常大面积贴图像四条边）；
  3) 取剩下的前 K 个，每个单独渲染成小图（原图 + 该提议单独染红），拼成一张 contact sheet，
     每格标注编号/面积占比/predicted_iou；机器先猜一个"最像头发"的格子用绿框标出；
  4) 另存 union 图（前 K 个中"顶部 45%、面积 1%~25%"的并集），看拼起来像不像头发。
产物：每张图 `<name>_sheet.png`、`<name>_union.png`、`<name>_props2.json` + `summary2.json`。

    python scripts\\hair_sam_probe2.py --n 8 --points 16 --maxpix 1024
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_CKPT = r'<你的 ComfyUI 安装目录>\models\sams\sam_vit_b_01ec64.pth'
POOL = [os.path.join(r'<你的图片目录>', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')]
EXTS = ('.png', '.jpg', '.jpeg', '.webp')
# 想覆盖的风格：白发/黑发/长直发/双马尾/大场景/近景
WANT = ['aquarium', 'miyabi_shumping', 'kzsd', 'ydhl', 'vva_punish', 'cp5',
        'shirogane', 'yukikaze']


def pick_files(n):
    cand = []
    for d in POOL:
        for e in EXTS:
            cand += glob.glob(os.path.join(d, '*' + e))
    out = []
    for w in WANT:
        for p in sorted(cand):
            if w.lower() in os.path.basename(p).lower() and p not in out:
                out.append(p)
                break
    for p in sorted(cand):
        if len(out) >= n:
            break
        if p not in out:
            out.append(p)
    return out[:n]


def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def border_frac(mask, band=2):
    """提议像素落在图像四条边带里的比例——背景/地面/水面大块的典型特征。"""
    m = mask.astype(bool)
    tot = m.sum()
    if tot == 0:
        return 1.0
    b = np.zeros_like(m)
    b[:band, :] = True
    b[-band:, :] = True
    b[:, :band] = True
    b[:, -band:] = True
    return float((m & b).sum()) / tot


def hair_prior(mask, img):
    """机器猜"这是不是头发"：靠画面顶部 + 面积适中 + 不太贴边 + 比背景暗。

    这不是分类器，只是一个排序用的粗糙先验，用来在 sheet 上标出机器认为最像的那格。
    """
    h, w = mask.shape
    m = mask.astype(bool)
    area = m.sum()
    if area == 0:
        return -9.9
    ys, xs = np.nonzero(m)
    cy = ys.mean() / h
    top = ys.min() / h
    ar = area / float(h * w)
    bf = border_frac(m)
    val = float(img[m].mean()) / 255.0  # 平均亮度（BGR 均值）
    sc = 0.0
    sc += 0.6 * (1.0 - min(1.0, top / 0.5))          # 越靠顶部越好
    sc += 0.4 * (1.0 - min(1.0, abs(cy - 0.25) / 0.5))
    sc += 0.5 * (1.0 if 0.01 <= ar <= 0.25 else -0.3)
    sc -= 1.2 * min(1.0, bf / 0.15)
    sc += 0.4 * (1.0 - val)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(ROOT, '.runtime', 'hair_cv', 'sam_probe2'))
    ap.add_argument('--points', type=int, default=16)
    ap.add_argument('--maxpix', type=int, default=1024)
    ap.add_argument('--topk', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    t0 = time.time()
    sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to('cuda')
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.92, min_mask_region_area=300)
    print('SAM 载入 %.1fs；points_per_side=%d' % (time.time() - t0, a.points), flush=True)

    files = pick_files(a.n)
    summary = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        sc = min(1.0, a.maxpix / max(h0, w0))
        if sc < 1.0:
            img = cv2.resize(img, (int(w0 * sc), int(h0 * sc)))
        h, w = img.shape[:2]
        t1 = time.time()
        with torch.no_grad():
            ms = gen.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dt = time.time() - t1
        ms = sorted(ms, key=lambda m: -m['area'])

        recs = []
        for i, m in enumerate(ms):
            seg = m['segmentation']
            recs.append({'i': i, 'area': int(m['area']), 'area_pct': round(100 * m['area'] / (h * w), 2),
                         'bbox': [int(v) for v in m['bbox']], 'iou': round(float(m['predicted_iou']), 3),
                         'stab': round(float(m['stability_score']), 3),
                         'border_frac': round(border_frac(seg), 3),
                         'prior': round(hair_prior(seg, img), 3)})

        # 背景压制：贴边太多或面积过大的先扔掉
        keep = [r for r in recs if r['border_frac'] <= 0.12 and r['area_pct'] <= 35.0]
        if len(keep) < 4:
            keep = [r for r in recs if r['area_pct'] <= 35.0]
        show = keep[:a.topk]
        if show:
            best = max(show, key=lambda r: r['prior'])
        else:
            best = {'i': -1}

        # contact sheet：每格 = 原图 + 该提议单独染红
        tile_w = 420
        tsc = tile_w / float(w)
        tile_h = int(h * tsc)
        cols = 4
        rows = max(1, (len(show) + cols - 1) // cols)
        sheet = np.full((rows * (tile_h + 26), cols * tile_w, 3), 24, np.uint8)
        for k, r in enumerate(show):
            vis = tint(img, ms[r['i']]['segmentation'], (0, 0, 255), 0.5)
            vis = cv2.resize(vis, (tile_w, tile_h))
            ry, rx = divmod(k, cols)
            y0 = ry * (tile_h + 26)
            sheet[y0:y0 + tile_h, rx * tile_w:(rx + 1) * tile_w] = vis
            label = '#%d  %.1f%%  iou%.2f  bf%.2f' % (r['i'], r['area_pct'], r['iou'], r['border_frac'])
            cv2.putText(sheet, label, (rx * tile_w + 4, y0 + tile_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            col = (0, 255, 0) if r['i'] == best['i'] else (70, 70, 70)
            cv2.rectangle(sheet, (rx * tile_w, y0), (rx * tile_w + tile_w - 1, y0 + tile_h - 1), col, 3)
        ssc = min(1.0, 1500 / max(sheet.shape[:2]))
        cv2.imwrite(os.path.join(a.out, name + '_sheet.png'),
                    cv2.resize(sheet, (int(sheet.shape[1] * ssc), int(sheet.shape[0] * ssc))))

        # union：顶部 45%、面积 1%~25%、不太贴边的提议并起来，看"拼出来像不像头发"
        uni = np.zeros((h, w), bool)
        parts = []
        for r in recs:
            if not (1.0 <= r['area_pct'] <= 25.0) or r['border_frac'] > 0.12:
                continue
            if r['bbox'][1] > 0.45 * h:
                continue
            uni |= ms[r['i']]['segmentation'].astype(bool)
            parts.append(r['i'])
        uvis = tint(img, uni, (0, 0, 255), 0.55) if uni.any() else img.copy()
        uvis = np.concatenate([img, uvis], 1)
        usc = min(1.0, 1400 / max(uvis.shape[:2]))
        cv2.imwrite(os.path.join(a.out, name + '_union.png'),
                    cv2.resize(uvis, (int(uvis.shape[1] * usc), int(uvis.shape[0] * usc))))

        recs_sorted = sorted(recs, key=lambda r: -r['prior'])
        json.dump({'name': name, 'size': [w, h], 'n_props': len(ms), 'sec': round(dt, 1),
                   'shown': [r['i'] for r in show], 'best_prior': best['i'],
                   'union_parts': parts, 'union_pct': round(100 * uni.mean(), 2),
                   'props': recs},
                  open(os.path.join(a.out, name + '_props2.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        summary.append({'name': name, 'size': [w, h], 'n_props': len(ms), 'sec': round(dt, 1),
                        'n_kept': len(keep), 'best_prior': best['i'],
                        'best_prior_area_pct': best.get('area_pct'), 'union_pct': round(100 * uni.mean(), 2),
                        'top_prior': [(r['i'], r['area_pct'], r['prior']) for r in recs_sorted[:5]]})
        print('%-26s %dx%d 提议%3d 保留%2d 机器猜#%s(%.1f%%) 并集%.1f%%  %.1fs'
              % (name, w, h, len(ms), len(keep), best['i'], best.get('area_pct') or 0,
                 uni.mean() * 100, dt), flush=True)

    json.dump(summary, open(os.path.join(a.out, 'summary2.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n产物目录：%s' % a.out)


main()
