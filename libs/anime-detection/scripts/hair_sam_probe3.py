#!/usr/bin/env python
"""头发 mask 探针 v3：用 SAM 的 point prompt（模拟"人点一下"）看能不能一次拿到整头头发。

v2 结论（自动提议 + 启发式挑选）：
  - 头发是大块、高对比、边界清楚时，SAM 的自动提议里**确实存在**一条完整的头发 mask
    （ydhl 的 #1、miyabi 的 #4 都是整头头发）；
  - 但"哪条提议是头发"纯靠面积/位置先验猜不出来（v2 里机器猜的格子基本都错）；
  - 黑发压深色衣服（cp5）、白发压白云（kzsd）时，自动提议里干脆没有干净的头发。

v3 要回答的是**标注加速**这条路能不能走：在头发上点 1 个点（或多点），SAM 能不能给出整头头发？
点坐标按比例写死在下面（人工看图给的），不依赖自动挑选。

    python scripts\\hair_sam_probe3.py --maxpix 1024
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_CKPT = r'<你的 ComfyUI 安装目录>\models\sams\sam_vit_b_01ec64.pth'
POOL = [os.path.join(r'<你的图片目录>', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')]

# (图名子串, [单点(比例)], [多点(比例)])
CASES = [
    ('ydhl', [(0.20, 0.12)], [(0.13, 0.10), (0.22, 0.16), (0.32, 0.14)]),
    ('miyabi_shumping', [(0.30, 0.10)], [(0.26, 0.10), (0.33, 0.14), (0.24, 0.26)]),
    ('cp5', [(0.32, 0.10)], [(0.28, 0.08), (0.40, 0.16), (0.24, 0.20)]),
    ('kzsd', [(0.06, 0.15)], [(0.04, 0.10), (0.09, 0.20), (0.05, 0.30)]),
    ('vva_punish', [(0.10, 0.12)], [(0.06, 0.10), (0.14, 0.16), (0.17, 0.30)]),
    ('aquarium', [(0.22, 0.10)], [(0.18, 0.09), (0.27, 0.11), (0.22, 0.20)]),
]


def find_img(key):
    for d in POOL:
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            import glob
            for p in sorted(glob.glob(os.path.join(d, '*' + ext))):
                if key in os.path.basename(p).lower():
                    return p
    return None


def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def draw_points(img, pts):
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), 2)
        cv2.putText(out, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, '.runtime', 'hair_cv', 'sam_probe3'))
    ap.add_argument('--maxpix', type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from segment_anything import sam_model_registry, SamPredictor
    t0 = time.time()
    sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to('cuda')
    pred = SamPredictor(sam)
    print('SAM(predictor) 载入 %.1fs' % (time.time() - t0), flush=True)

    rows = []
    for key, one, multi in CASES:
        fp = find_img(key)
        if fp is None:
            print('找不到 %s' % key, flush=True)
            continue
        name = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')[:40]
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        h0, w0 = img.shape[:2]
        sc = min(1.0, a.maxpix / max(h0, w0))
        if sc < 1.0:
            img = cv2.resize(img, (int(w0 * sc), int(h0 * sc)))
        h, w = img.shape[:2]
        t1 = time.time()
        with torch.no_grad():
            pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        panels = [draw_points(img, [(p[0] * w, p[1] * h) for p in one])]
        info = {'name': name, 'src': fp, 'size': [w, h], 'sets': []}
        for tag, pts in (('1pt', one), ('3pt', multi)):
            xy = np.array([[p[0] * w, p[1] * h] for p in pts], np.float32)
            lb = np.ones(len(xy), np.int64)
            with torch.no_grad():
                masks, scores, _ = pred.predict(point_coords=xy, point_labels=lb,
                                                multimask_output=True)
            ar = [float(m.mean()) for m in masks]
            ok = [i for i in range(len(masks)) if ar[i] <= 0.35]
            pick = max(ok, key=lambda i: scores[i]) if ok else int(np.argmax(scores))
            panels.append(tint(img, masks[pick], (0, 0, 255), 0.5))
            cv2.putText(panels[-1], '%s: mask#%d %.1f%% score%.3f' % (tag, pick, ar[pick] * 100, scores[pick]),
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            info['sets'].append({'tag': tag, 'pts': pts, 'pick': pick,
                                 'area_pct': round(ar[pick] * 100, 2),
                                 'score': round(float(scores[pick]), 3),
                                 'all_area_pct': [round(x * 100, 2) for x in ar],
                                 'all_score': [round(float(s), 3) for s in scores]})
            print('%-22s %s 挑中 mask#%d 面积%.1f%% score%.3f   (3个候选面积 %s)'
                  % (name[:22], tag, pick, ar[pick] * 100, scores[pick],
                     ' '.join('%.1f%%' % (x * 100) for x in ar)), flush=True)
        row = np.concatenate(panels, 1)
        rsc = min(1.0, 1500 / row.shape[1])
        cv2.imwrite(os.path.join(a.out, name + '_prompt.png'),
                    cv2.resize(row, (int(row.shape[1] * rsc), int(row.shape[0] * rsc))))
        info['sec'] = round(time.time() - t1, 2)
        rows.append(info)

    json.dump(rows, open(os.path.join(a.out, 'prompt_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n产物目录：%s' % a.out)


main()
