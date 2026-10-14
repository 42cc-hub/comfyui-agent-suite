#!/usr/bin/env python
"""两阶段头发 mask 探针：动漫人脸检测（deepghs/anime_face_detection）→ 头部裁切 → DP 解析模型 → 贴回原图。

DP 的解析模型是 512x512 人脸裁切域（训练/测试都是这种裁切），整幅插画直接送进去会碎（见 docs/hair_dp_check.md）。
所以这里按"检测 → 裁切 → 解析 → 贴回"走一遍，看不动任何训练能不能拿到能用的头发 mask。

    python scripts\\hair_dp_twostage.py --n 6 --expand 1.8
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = os.path.join(ROOT, '.runtime', 'hair_cv', 'faceparsing')
DP = os.path.join(ROOT, '.runtime', 'hair_cv', 'dp')
DET = os.path.join(ROOT, '.runtime', 'hair_cv', 'facedet', 'face_detect_v1.4_s', 'model.pt')
sys.path.insert(0, FP)

HAIR = 17
RNG = np.random.RandomState(1)
PAL = np.concatenate([np.array([[30, 30, 30]], np.uint8), RNG.randint(60, 255, (18, 3)).astype(np.uint8)])
POOL = [os.path.join(r'<你的图片目录>', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')]
WANT = ['aquarium', 'miyabi_shumping', 'cp5', 'kzsd', 'vva_punish', 'ydhl']
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def parse_model(ckpt, dev):
    from model import BiSeNet
    m = BiSeNet(n_classes=19)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd, strict=False)
    return m.eval().to(dev)


def hair_prob(m, crop_bgr, dev, size=512):
    im = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = ((im[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    with torch.no_grad():
        out, _, _ = m(torch.from_numpy(x)[None].to(dev))
    return F.softmax(out[0], 0)[HAIR].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DP, 'twostage'))
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--expand', type=float, default=1.8, help='人脸框向外扩张倍数（把头发包进来）')
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    from ultralytics import YOLO
    det = YOLO(DET)
    pars = parse_model(os.path.join(DP, 'pretrained_anime_parsing.pth'), dev)

    files = []
    for w in WANT[:a.n]:
        for d in POOL:
            hit = None
            for ext in ('.png', '.jpg', '.jpeg', '.webp'):
                hs = [p for p in sorted(glob.glob(os.path.join(d, '*' + ext))) if w in os.path.basename(p).lower()]
                if hs:
                    hit = hs[0]
                    break
            if hit:
                files.append(hit)
                break

    rows, rep = [], []
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')[:30]
        H, W = img.shape[:2]
        t0 = time.time()
        r = det.predict(img, conf=a.conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        prob = np.zeros((H, W), np.float32)
        crops = []
        for (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = (x1 - x0), (y1 - y0)
            ew, eh = w * a.expand, h * a.expand
            gx0, gy0 = int(max(0, cx - ew / 2)), int(max(0, cy - eh / 2))
            gx1, gy1 = int(min(W, cx + ew / 2)), int(min(H, cy + eh / 2))
            if gx1 - gx0 < 16 or gy1 - gy0 < 16:
                continue
            crop = img[gy0:gy1, gx0:gx1]
            p = hair_prob(pars, crop, dev)
            p = cv2.resize(p, (gx1 - gx0, gy1 - gy0), interpolation=cv2.INTER_LINEAR)
            prob[gy0:gy1, gx0:gx1] = np.maximum(prob[gy0:gy1, gx0:gx1], p)
            crops.append((gx0, gy0, gx1, gy1))
        dt = time.time() - t0
        mask = (prob >= 0.5).astype(np.uint8)
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        big = [int(s[4]) for s in stats[1:] if s[4] >= 64]
        rep.append({'name': name, 'faces': int(len(boxes)), 'used_crops': len(crops), 'sec': round(dt, 2),
                    'hair_pct': round(100 * float(mask.mean()), 2), 'hair_blocks_ge64': len(big),
                    'hair_max_block': max(big) if big else 0,
                    'boxes': [[round(float(v)) for v in b] for b in boxes[:8]]})
        # 可视化
        vis_box = img.copy()
        for b in boxes:
            cv2.rectangle(vis_box, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 3)
        over = (0.45 * img + 0.55 * cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)).astype(np.uint8)
        mvis = img.copy()
        mvis[mask.astype(bool)] = (0.35 * mvis[mask.astype(bool)] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        tiles = [vis_box, over, mvis]
        if crops:
            gx0, gy0, gx1, gy1 = max(crops, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
            tiles.insert(1, cv2.resize(img[gy0:gy1, gx0:gx1], (img.shape[1], img.shape[0])))
        row = np.concatenate([cv2.resize(t, (img.shape[1], img.shape[0])) for t in tiles], 1)
        sc = 1100 / row.shape[1]
        rows.append(cv2.resize(row, (1100, int(row.shape[0] * sc))))
    if rows:
        cv2.imwrite(os.path.join(a.out, 'twostage.png'), np.concatenate(rows, 0))
    json.dump(rep, open(os.path.join(a.out, 'twostage.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print('\n产物：%s' % a.out)


main()
