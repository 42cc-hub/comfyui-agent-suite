#!/usr/bin/env python
"""头发 mask 探针 v4：点位网格扫描。

v3 的教训：我按比例估的点落在了头发外面（打在背景/墙上），SAM 就老老实实给出背景 mask，
所以"点错"和"SAM 不会切头发"看起来一样，分不清。
v4 改成在图面上部打 5x4 网格点，每个点单独 prompt，把 20 张结果拼成一张 contact sheet，
格子里标出点坐标和 mask 面积占比——这样"头发在哪一格能被切出来"一眼就能看出来。

    python scripts\\hair_sam_probe4.py --maxpix 1024
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
KEYS = ['ydhl', 'miyabi_shumping', 'cp5', 'kzsd', 'vva_punish', 'aquarium']
XS = [0.10, 0.30, 0.50, 0.70, 0.90]
YS = [0.06, 0.18, 0.30, 0.42]


def find_img(key):
    for d in POOL:
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            for p in sorted(glob.glob(os.path.join(d, '*' + ext))):
                if key in os.path.basename(p).lower():
                    return p
    return None


def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, '.runtime', 'hair_cv', 'sam_probe4'))
    ap.add_argument('--maxpix', type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from segment_anything import sam_model_registry, SamPredictor
    t0 = time.time()
    sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT).to('cuda')
    pred = SamPredictor(sam)
    print('SAM(predictor) 载入 %.1fs' % (time.time() - t0), flush=True)

    summary = []
    for key in KEYS:
        fp = find_img(key)
        if fp is None:
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
        tile_w = 360
        tsc = tile_w / float(w)
        tile_h = int(h * tsc)
        cols, rows = len(XS), len(YS)
        sheet = np.full((rows * (tile_h + 24), cols * tile_w, 3), 24, np.uint8)
        rec = []
        for ri, yy in enumerate(YS):
            for ci, xx in enumerate(XS):
                px, py = xx * w, yy * h
                with torch.no_grad():
                    masks, scores, _ = pred.predict(point_coords=np.array([[px, py]], np.float32),
                                                    point_labels=np.array([1], np.int64),
                                                    multimask_output=True)
                ar = [float(m.mean()) for m in masks]
                ok = [i for i in range(len(masks)) if ar[i] <= 0.35]
                pick = max(ok, key=lambda i: scores[i]) if ok else int(np.argmax(scores))
                vis = tint(img, masks[pick], (0, 0, 255), 0.5)
                cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 255), -1)
                tile = cv2.resize(vis, (tile_w, tile_h))
                y0, x0 = ri * (tile_h + 24), ci * tile_w
                sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile
                cv2.putText(sheet, '(%.2f,%.2f) %.1f%%' % (xx, yy, ar[pick] * 100),
                            (x0 + 4, y0 + tile_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                rec.append({'x': xx, 'y': yy, 'pick': pick, 'area_pct': round(ar[pick] * 100, 2),
                            'score': round(float(scores[pick]), 3),
                            'all_area_pct': [round(v * 100, 2) for v in ar]})
        ssc = min(1.0, 1500 / sheet.shape[1])
        cv2.imwrite(os.path.join(a.out, name + '_grid.png'),
                    cv2.resize(sheet, (int(sheet.shape[1] * ssc), int(sheet.shape[0] * ssc))))
        summary.append({'name': name, 'src': fp, 'size': [w, h], 'sec': round(time.time() - t1, 1),
                        'grid': rec})
        print('%-26s %dx%d 网格 %d 点  %.1fs' % (name, w, h, len(rec), time.time() - t1), flush=True)

    json.dump(summary, open(os.path.join(a.out, 'grid_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n产物目录：%s' % a.out)


main()
