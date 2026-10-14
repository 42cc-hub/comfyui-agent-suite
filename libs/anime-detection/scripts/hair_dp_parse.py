#!/usr/bin/env python
"""用 Danbooru-Parsing（DP）官方放的解析模型做两件事：
  1) 域内校验：在 DP 自带的 test_68 上算 per-class IoU / mIoU / 头发 IoU，确认权重是可用的；
  2) 跨域检查：直接把模型跑在我们的整幅插画上（resize 512x512），看头发 mask 能不能用。
第二条是决定"能不能拿它当预标注器"的关键：DP 是 512x512 人脸裁切域，我们的图是全身大场景。

    python scripts\\hair_dp_parse.py --n 6
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
sys.path.insert(0, FP)

NAMES = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear',
         'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
HAIR, SKIN, CLOTH, BG = 17, 1, 16, 0
RNG = np.random.RandomState(1)
PAL = np.concatenate([np.array([[30, 30, 30]], np.uint8), RNG.randint(60, 255, (18, 3)).astype(np.uint8)])
POOL = [os.path.join(r'<你的图片目录>', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')]
WANT = ['aquarium', 'miyabi_shumping', 'cp5', 'kzsd', 'vva_punish', 'ydhl']
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build(n_classes=19):
    from model import BiSeNet
    m = BiSeNet(n_classes=n_classes)
    return m


def load(ckpt, dev):
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m = build()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print('载入权重：缺失 %d 键 / 多余 %d 键' % (len(missing), len(unexpected)))
    if missing[:5]:
        print('  缺失示例:', missing[:5])
    if unexpected[:5]:
        print('  多余示例:', unexpected[:5])
    m.eval().to(dev)
    return m


def prep(bgr, size=512, keep_ratio=False):
    """按官方 test.py 的做法：resize 到 size×size + ImageNet 归一化。"""
    im = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = im[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def logits_of(model, bgr, dev, size=512):
    with torch.no_grad():
        out, _, _ = model(prep(bgr, size).to(dev))
    return out[0]  # (19,H,W) logits


def iou_stats(pred, gt, n=19):
    res = {}
    for c in range(n):
        g = gt == c
        if g.sum() == 0:
            continue
        p = pred == c
        inter = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()
        res[NAMES[c]] = round(float(inter) / max(1, union), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(DP, 'pretrained_anime_parsing.pth'))
    ap.add_argument('--out', default=os.path.join(DP, 'eval'))
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    print('device=%s' % dev)
    model = load(a.ckpt, dev)

    # ---------- 1) 域内校验 ----------
    tests = sorted(glob.glob(os.path.join(DP, 'extract', 'Danbooru-Parsing', 'anime_face_test_68', '*.jpg')))
    print('\n[1] 域内校验：DP test_68，共 %d 张' % len(tests))
    ious, hair_ious, secs = [], [], []
    sheet_rows = []
    for i, ip in enumerate(tests):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(DP, 'extract', 'Danbooru-Parsing', 'anime_seg_test_68', stem + '.png')
        if not os.path.exists(mp):
            continue
        img = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        t0 = time.time()
        lg = logits_of(model, img, dev)
        secs.append(time.time() - t0)
        pred = lg.argmax(0).cpu().numpy().astype(np.uint8)
        st = iou_stats(pred, gt)
        ious.append(float(np.mean(list(st.values()))))
        hair_ious.append(st.get('hair', 0.0))
        if i < 4:
            vis = np.concatenate([img, PAL[np.clip(gt, 0, 18)], PAL[np.clip(pred, 0, 18)]], 1)
            sc = 320 / vis.shape[0]
            sheet_rows.append(cv2.resize(vis, (int(vis.shape[1] * sc), 320)))
    if sheet_rows:
        cv2.imwrite(os.path.join(a.out, 'dp_indomain.png'), np.concatenate(sheet_rows, 0))
    ind = {'n': len(ious), 'mIoU': round(float(np.mean(ious)), 4) if ious else None,
           'hair_IoU_mean': round(float(np.mean(hair_ious)), 4) if hair_ious else None,
           'sec_per_img': round(float(np.mean(secs)), 3) if secs else None}
    print('   mIoU=%(mIoU)s  头发IoU=%(hair_IoU_mean)s  %(sec_per_img)s s/张' % ind)

    # ---------- 2) 跨域：直接跑我们的整幅插画 ----------
    print('\n[2] 跨域：整幅插画 resize 512 直接送进模型')
    files = []
    for w in WANT[:a.n]:
        for d in POOL:
            for ext in ('.png', '.jpg', '.jpeg', '.webp'):
                hits = [p for p in sorted(glob.glob(os.path.join(d, '*' + ext))) if w in os.path.basename(p).lower()]
                if hits:
                    files.append(hits[0])
                    break
            else:
                continue
            break
    cross = []
    rows = []
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')[:30]
        t0 = time.time()
        lg = logits_of(model, img, dev)
        prob = F.softmax(lg, 0).cpu().numpy()
        pred = prob.argmax(0).astype(np.uint8)
        dt = time.time() - t0
        hair = (pred == HAIR).astype(np.uint8)
        # 统计
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(hair, 8)
        big = [int(s[4]) for s in stats[1:] if s[4] >= 64]
        cross.append({'name': name, 'src': fp, 'sec': round(dt, 2),
                      'hair_pct': round(100 * float(hair.mean()), 2),
                      'skin_pct': round(100 * float((pred == SKIN).mean()), 2),
                      'cloth_pct': round(100 * float((pred == CLOTH).mean()), 2),
                      'bg_pct': round(100 * float((pred == BG).mean()), 2),
                      'hair_blocks_ge64': len(big),
                      'hair_max_block': max(big) if big else 0})
        # 可视化：原图 | 头发概率热力 | 全类着色
        h, w = img.shape[:2]
        ph = cv2.resize(prob[HAIR], (w, h), interpolation=cv2.INTER_LINEAR)
        over = (0.45 * img + 0.55 * cv2.applyColorMap((ph * 255).astype(np.uint8), cv2.COLORMAP_JET)).astype(np.uint8)
        lab = PAL[np.clip(cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST), 0, 18)]
        row = np.concatenate([img, over, lab], 1)
        sc = 900 / row.shape[1]
        rows.append(cv2.resize(row, (900, int(row.shape[0] * sc))))
    if rows:
        cv2.imwrite(os.path.join(a.out, 'dp_crossdomain.png'), np.concatenate(rows, 0))
    rep = {'indomain': ind, 'crossdomain': cross,
           'label_names': NAMES,
           'note': 'crossdomain 的 pred 是把整幅插画 resize 到 512x512 后的结果，DP 是 512x512 人脸裁切域'}
    json.dump(rep, open(os.path.join(a.out, 'dp_eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep['crossdomain'], ensure_ascii=False, indent=1))
    print('\n产物：%s' % a.out)


main()
