"""Blender 剪影 mask → 生成图上的局部重绘 mask。

为什么不能直接用 Blender 的 mask：生成的图不会和 3D 渲染像素级对齐（姿态/轮廓都会走形），
直接拿去重绘会把周围一圈也涂掉或漏掉。

做法：
  1. Blender 剪影 mask 做腐蚀/膨胀，得到 trimap（腐蚀后=确定前景，外环=可能前景，更外=可能背景）
  2. 在【生成图】上跑 GrabCut：边界会吸附到生成图里真实的轮廓边缘
  3. 结果与"膨胀后的先验 mask"求交 —— 防止 GrabCut 顺着相似颜色泄漏到背景
  4. 最大连通域 + 填洞 + 羽化

usage: python make_inpaint_mask.py <base.png> <blender_mask.png> <out_mask.png> [--grow 14] [--feather 6]
"""
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

argv = sys.argv[1:]
base_path, prior_path, out_path = argv[0], argv[1], argv[2]
grow = int(argv[argv.index('--grow') + 1]) if '--grow' in argv else 14
feather = int(argv[argv.index('--feather') + 1]) if '--feather' in argv else 6

img = cv2.imread(base_path, cv2.IMREAD_COLOR)
prior = np.asarray(Image.open(prior_path).convert('L'))
prior = (prior > 96).astype(np.uint8)
if prior.shape[:2] != img.shape[:2]:
    prior = cv2.resize(prior, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
print(f'prior coverage: {100*prior.mean():.1f}%')

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
outer = cv2.dilate(prior, k, iterations=1)          # 允许边界外扩的范围（硬约束）
inner = cv2.erode(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

gc = np.full(prior.shape, cv2.GC_PR_BGD, np.uint8)
gc[outer > 0] = cv2.GC_PR_FGD
gc[inner > 0] = cv2.GC_FGD
gc[outer == 0] = cv2.GC_BGD

bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(img, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

fg = fg & outer                                       # 不允许跑到先验的膨胀范围之外
n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
if n > 1:
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (lab == keep).astype(np.uint8)
fg = ndimage.binary_fill_holes(fg).astype(np.uint8)
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

# 重绘区通常要比人物本身再大一圈，给模型留出重建边界的余地
fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 6 + 1, 2 * 6 + 1)), iterations=1)
soft = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), feather)
mask8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
Image.fromarray(mask8, 'L').save(out_path)

# 预览：mask 覆盖在生成图上
vis = img.copy()
vis[mask8 > 128] = (0.45 * vis[mask8 > 128] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite(out_path.replace('.png', '_preview.png'), vis)
print(f'mask coverage: {100*(mask8>128).mean():.1f}%  -> {out_path}  (preview: *_preview.png)')
