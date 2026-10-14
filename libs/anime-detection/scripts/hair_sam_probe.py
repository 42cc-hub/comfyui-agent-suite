#!/usr/bin/env python
"""头发 mask 可行性探针：用 SAM（自动分割）看能不能直接拿到头发 mask。

做法（纯 SAM，不训练）：
  1) `SamAutomaticMaskGenerator` 出一堆类别无关的分割提议；
  2) 存"提议全景图"（每个提议一种颜色 + 编号）+ 提议统计（面积/bbox/IoU/稳定性）；
  3) 再按一个粗糙启发式挑一个"最像头发"的提议（面积 2%~30%、bbox 顶部落在画面上 45% 以内、
     优先面积大、IoU 高）作为候选人，单独存叠加图。
产物：每张图 `<name>_all.png`（提议全景）、`<name>_cand.png`（候选叠加）、`<name>_props.json`。

    python scripts\\hair_sam_probe.py --n 6 --out .runtime\\hair_cv\\sam_probe
"""
import argparse
import glob
import io
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
# 想覆盖的风格：白发/黑发/双马尾/长发/大场景
WANT = ['aquarium', 'miyabi_shumping', 'kzsd', 'ydhl', 'vva_punish', 'cp5']


def pick_files(n):
    cand = []
    for d in POOL:
        for e in EXTS:
            cand += glob.glob(os.path.join(d, '*' + e))
    out = []
    for w in WANT:
        for p in cand:
            if w in os.path.basename(p) and p not in out:
                out.append(p)
                break
    for p in sorted(cand):
        if len(out) >= n:
            break
        if p not in out:
            out.append(p)
    return out[:n]


def tint(img, mask, color, alpha=0.55):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default=os.path.join(ROOT, '.runtime', 'hair_cv', 'sam_probe'))
    ap.add_argument('--points', type=int, default=32)
    ap.add_argument('--maxpix', type=int, default=1600, help='长边超过则等比缩小（省显存/时间）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    t0 = time.time()
    sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to('cuda')
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.92, min_mask_region_area=300)
    print('SAM 载入 %.1fs；自动分割器 points_per_side=%d' % (time.time() - t0, a.points), flush=True)

    files = pick_files(a.n)
    print('测试图 %d 张' % len(files), flush=True)
    summary = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = min(1.0, a.maxpix / max(h, w))
        if sc < 1.0:
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
        h, w = img.shape[:2]
        t1 = time.time()
        with torch.no_grad():
            ms = gen.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dt = time.time() - t1
        ms = sorted(ms, key=lambda m: -m['area'])
        # 提议全景（每个提议一种颜色，按面积从大到小叠；编号写在 bbox 左上）
        canvas = img.copy()
        rng = np.random.RandomState(0)
        for i, m in enumerate(ms):
            col = tuple(int(c) for c in rng.randint(60, 255, 3))
            canvas = tint(canvas, m['segmentation'], col, alpha=0.35)
        for i, m in enumerate(ms[:40]):
            x, y, bw, bh = [int(v) for v in m['bbox']]
            cv2.putText(canvas, str(i), (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
        sc2 = min(1.0, 1100 / max(canvas.shape[:2]))
        cv2.imwrite(os.path.join(a.out, name + '_all.png'),
                    cv2.resize(canvas, (int(canvas.shape[1] * sc2), int(canvas.shape[0] * sc2))))
        # 粗糙的"头发候选"：面积 2%~30%、bbox 顶部在上 45%
        best, best_key = None, -1
        for i, m in enumerate(ms):
            ar = m['area'] / float(h * w)
            if not (0.02 <= ar <= 0.30):
                continue
            y = m['bbox'][1]
            if y > 0.45 * h:
                continue
            key = ar + 0.3 * m['predicted_iou']
            if key > best_key:
                best, best_key = (i, m), key
        cand_path = ''
        if best:
            i, m = best
            vis = tint(img, m['segmentation'], (0, 0, 255), 0.45)
            vis = np.concatenate([img, vis], 1)
            sc3 = min(1.0, 1200 / max(vis.shape[:2]))
            cv2.imwrite(os.path.join(a.out, name + '_cand.png'),
                        cv2.resize(vis, (int(vis.shape[1] * sc3), int(vis.shape[0] * sc3))))
            cand_path = name + '_cand.png'
        props = [{'i': i, 'area': int(m['area']), 'area_pct': round(100 * m['area'] / (h * w), 2),
                  'bbox': [int(v) for v in m['bbox']], 'iou': round(float(m['predicted_iou']), 3),
                  'stab': round(float(m['stability_score']), 3)} for i, m in enumerate(ms)]
        io.open(os.path.join(a.out, name + '_props.json'), 'w', encoding='utf-8').write(
            json.dumps(props, ensure_ascii=False, indent=1))
        summary.append({'name': name, 'size': [w, h], 'n_props': len(ms), 'sec': round(dt, 1),
                        'top_area_pct': [p['area_pct'] for p in props[:8]],
                        'cand': cand_path, 'cand_i': best[0] if best else -1,
                        'cand_area_pct': round(100 * best[1]['area'] / (h * w), 2) if best else 0})
        print('%-28s %dx%d 提议 %3d 个  %.1fs  候选#%s(%.1f%%)'
              % (name, w, h, len(ms), dt, best[0] if best else '-',
                 best[1]['area'] * 100.0 / (h * w) if best else 0), flush=True)
    io.open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8').write(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print('\n产物目录：%s' % a.out)


main()
