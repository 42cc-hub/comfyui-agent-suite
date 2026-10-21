# -*- coding: utf-8 -*-
"""task-10 补充②：独立旋转来源对照（纯 CPU，不占 GPU）。

目的：训练增广用 cv2.getRotationMatrix2D + warpAffine（同尺寸、黑边），测试也走同一实现，
旋转桶增益可能部分是"同源实现"带来的。这里换一条**完全不同的实现**做 90° 一档：
  * PIL `Image.rotate(90, expand=True)`（真转置、无黑边、画布 H×W 互换）；
  * 框用自己推的旋转矩阵重算（并用合成图自检：白块位置必须对回来）。
只测 `data/hand_gt` 40 张 upright 独立集（未参与任何训练）的手 R/P（IoU 0.5 / conf 0.30 / imgsz 960），
模型 old / v2 / rot_semi。产物 team/reports/m2a-verifier/rot_semi/indep_rot90.txt
"""
import csv
import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
HG = os.path.join(SUITE, 'data', 'hand_gt')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'rot_semi')
MODELS = {
    'old': os.path.join(SUITE, 'runs', 'm2a_det_996img_960', 'weights', 'best.pt'),
    'v2': os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'),
    'rot_semi': os.path.join(SUITE, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt'),
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(gts, preds, thr=0.5):
    used, tp = set(), 0
    for g in gts:
        best, bi = thr, -1
        for j, p in enumerate(preds):
            if j in used:
                continue
            v = iou(g, p)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            tp += 1
            used.add(bi)
    return tp, len(preds) - tp, len(gts) - tp


def rot90_expand(img_rgb, boxes):
    """PIL rotate 90°, expand=True；返回 (out_rgb, new_boxes, W', H')。

    90° 逆时针 + expand 是纯转置：原 (x,y) → (y, H-1-y?) 用矩阵统一处理：
    内容逆时针转 90° 时，原坐标 p 在输出里满足  q = (y, W - x)（对 W×H 图，输出尺寸 H×W）。
    下面同时用合成图自检，若符号不对会断言失败。
    """
    h, w = img_rgb.shape[:2]
    im = Image.fromarray(img_rgb)
    rot = im.rotate(90, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    ow, oh = rot.size
    out = np.asarray(rot)
    new = []
    for (x0, y0, x1, y1) in boxes:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        # 候选映射：q = (y, w - x)（内容逆时针 90°、输出宽=h、高=w）
        q = np.stack([pts[:, 1], w - pts[:, 0]], axis=1)
        nx0, nx1 = float(q[:, 0].min()), float(q[:, 0].max())
        ny0, ny1 = float(q[:, 1].min()), float(q[:, 1].max())
        nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
        nx1, ny1 = min(float(ow), nx1), min(float(oh), ny1)
        if nx1 - nx0 >= 4 and ny1 - ny0 >= 4:
            new.append([nx0, ny0, nx1, ny1])
    return out, new, ow, oh


def self_test():
    w, h = 200, 100
    img = np.zeros((h, w, 3), np.uint8)
    x0, y0, x1, y1 = 30, 20, 50, 30
    img[y0:y1, x0:x1] = 255
    out, new, ow, oh = rot90_expand(img, [[x0, y0, x1, y1]])
    mask = (out[:, :, 0] > 200).astype(np.uint8)
    ys, xs = np.where(mask)
    real = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert len(new) == 1, new
    err = max(abs(real[i] - new[0][i]) for i in range(4))
    print('自检 90°：白块实测 bbox=%s 变换后 bbox=%s 最大误差=%.1fpx 输出尺寸=%dx%d'
          % (real, [round(v, 1) for v in new[0]], err, ow, oh))
    assert err <= 2.0, '框变换与 PIL 旋转不一致'
    return err


def main():
    os.makedirs(OUT, exist_ok=True)
    err = self_test()
    gt = {}
    for r in csv.DictReader(io.open(os.path.join(HG, 'index.csv'), encoding='utf-8-sig')):
        if r['verdict'] == 'draw' and r['boxes'].strip():
            gt.setdefault(os.path.splitext(r['crop'])[0], {}).setdefault(r['target'], []).extend(
                [[float(v) for v in b.split(',')] for b in r['boxes'].split(';') if b.strip()])
    imgs = sorted(os.listdir(os.path.join(HG, 'images')))
    from ultralytics import YOLO
    models = {k: YOLO(v) for k, v in MODELS.items()}
    tmp = tempfile.mkdtemp(prefix='m2a_rot90_')
    tasks = []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        if stem not in gt:
            continue
        bgr = cv2.imread(os.path.join(HG, 'images', f))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = [b for v in gt[stem].values() for b in v]
        out, new, ow, oh = rot90_expand(rgb, boxes)
        # 逐框按类别重算（保持类别对应，避免上面的扁平顺序假设）
        by_cls = {}
        for cls, v in gt[stem].items():
            for b in v:
                o2, nb, _, _ = rot90_expand(rgb, [b])
                if nb:
                    by_cls.setdefault(cls, []).append(nb[0])
        p = os.path.join(tmp, stem + '.jpg')
        cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        tasks.append((stem, p, by_cls))
    print('独立旋转（PIL rotate 90 expand）样本 %d 张，临时目录 %s' % (len(tasks), tmp))
    res = {}
    for tag, m in models.items():
        agg = {}
        for stem, p, by_cls in tasks:
            r0 = m.predict(p, conf=0.30, imgsz=960, device='cpu', verbose=False)[0]
            pr = {}
            for b in (r0.boxes or []):
                pr.setdefault(r0.names[int(b.cls[0])], []).append([float(v) for v in b.xyxy[0].tolist()])
            for cls in ('hand', 'foot'):
                g = by_cls.get(cls, [])
                tp, fp, fn = score(g, pr.get(cls, []))
                a = agg.setdefault(cls, [0, 0, 0, 0])
                a[0] += len(g)
                a[1] += tp
                a[2] += fp
                a[3] += fn
        res[tag] = agg
    L = ['# 独立旋转来源对照（PIL rotate 90 expand=True；data/hand_gt 40 张，未参与训练）',
         '框变换自检：白块最大误差 **%.1f px**（阈值 2px）' % err, '',
         '| 模型 | hand GT/TP/FP | hand R | hand P | foot GT/TP/FP | foot R | foot P |',
         '|---|---|---|---|---|---|---|']
    for tag in ('old', 'v2', 'rot_semi'):
        h = res[tag].get('hand', [0, 0, 0, 0])
        ft = res[tag].get('foot', [0, 0, 0, 0])
        L.append('| %s | %d/%d/%d | %.1f%% | %.1f%% | %d/%d/%d | %.1f%% | %.1f%% |'
                 % (tag, h[0], h[1], h[2], 100.0 * h[1] / max(1, h[1] + h[3]), 100.0 * h[1] / max(1, h[1] + h[2]),
                    ft[0], ft[1], ft[2], 100.0 * ft[1] / max(1, ft[1] + ft[3]),
                    100.0 * ft[1] / max(1, ft[1] + ft[2])))
    print('\n'.join(L))
    io.open(os.path.join(OUT, 'indep_rot90.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
