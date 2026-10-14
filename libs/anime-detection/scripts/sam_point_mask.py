#!/usr/bin/env python
"""SAM 点提示一键选区：给种子点，整块取出一个"色块对象"（绳子/头发这类）。

设计目标（用户痛点）：在 PS 里用魔棒点很多下还会漏；这里**点一下出整块**，
并保留"颗粒度"开关：同一次点击 SAM 给 3 条提议（multimask），
  整片级 = 挑面积大的 / 把多条提议并起来；单片级 = 挑面积小的那条。

    python scripts/sam_point_mask.py --image x.png --point 400,300 --out mask.png
    python scripts/sam_point_mask.py --image x.png --point 400,300 --point 520,360 \
        --neg 180,120 --pick largest --merge 0.6 --overlay prev.png
    python scripts/sam_point_mask.py --image x.png --box 120,60,520,400 --point 300,200 --out mask.png

参数
  --point X,Y    正点（要选进来的东西上），可重复；第一个是种子点
  --neg   X,Y    负点（不该选进来的东西上），可重复
  --box x0,y0,x1,y1  粗框提示（一次拖拽）；可单独用，也可配 --point。实测：DP 头发上单独拖框
                 1 次操作中位 IoU 0.646（10% 达标），框+点 0.763；绳子上没用（绳子铺满全图）
  --mask-input PNG   低分辨率 mask 提示：把魔棒/colorblock 点出来的块喂给 SAM，SAM 修边界。
                 --mask-scale 是幅值（默认 10），校准方法见 docs/colorblock_sam_or_wan.md 第 8 节
  --pick         提议选择：largest(默认，整片级) | score(SAM 自评分，UI 原生) | smallest | #0/#1/#2 | all
  --merge T      把与当前选中块 IoU≥T 的其它提议并进来（修"双马尾/长发被切成多条"）
  --auto-merge T 追加：跑一次自动提议，把与当前块 IoU≥T 的自动提议并进来（更慢）
  --maxpix N     推理时长边上限（默认 1280；SAM 内部永远是 1024）
  --out          输出 mask PNG（255=选中，与原图同尺寸）
  --overlay      可选，输出叠色预览
  --cpu          强制 CPU（显存紧张时用）
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch

DEFAULT_CKPT = r'<你的用户目录>\.dsh\skills\comfyui-image-generation\models\sams\sam_vit_b_01ec64.pth'


def parse_pts(vals):
    out = []
    for v in vals or []:
        x, y = v.split(',')
        out.append((float(x), float(y)))
    return out


def mask_to_input(mask, scale=10.0, size=256):
    """二值 mask → SAM 的 mask_input（约定：256×256 float32 低分辨率 logits）。

    常见做法是把二值 mask 缩到 256×256 后映射成 ±scale。幅值越大 = 越"确信"，
    实测要在校准集上选（幅值太大反而压过点提示）。
    """
    m = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    b = (m > 0.5).astype(np.float32) * 2.0 - 1.0
    return (b * float(scale))[None].astype(np.float32)


class SamEngine:
    """持有一个 SAM vit_b；同一进程内复用（编码每张图一次）。"""

    _sam = None

    def __init__(self, ckpt=DEFAULT_CKPT, dev=None, maxpix=1280):
        self.maxpix = int(maxpix)
        if dev is None:
            dev = 'cpu' if os.environ.get('SAM_DEVICE') == 'cpu' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dev = dev
        if SamEngine._sam is None:
            from segment_anything import sam_model_registry
            if not os.path.exists(ckpt):
                raise SystemExit('找不到 SAM 权重：%s' % ckpt)
            SamEngine._sam = sam_model_registry['vit_b'](checkpoint=ckpt)
        self.sam = SamEngine._sam.to(dev)
        from segment_anything import SamPredictor
        self.pred = SamPredictor(self.sam)
        self.hw = None
        self.scale = 1.0
        self.orig_hw = None

    def set_image(self, bgr):
        h0, w0 = bgr.shape[:2]
        self.orig_hw = (h0, w0)
        sc = min(1.0, self.maxpix / float(max(h0, w0)))
        self.scale = sc
        img = bgr if sc >= 1.0 else cv2.resize(bgr, (int(round(w0 * sc)), int(round(h0 * sc))),
                                              interpolation=cv2.INTER_AREA)
        self.hw = img.shape[:2]
        with torch.no_grad():
            self.pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def _to_model(self, pts):
        return np.array([[x * self.scale, y * self.scale] for x, y in pts], np.float32)

    def proposals(self, pos, neg, box=None, mask_input=None, multimask=True):
        """返回 (masks[N,h,w] bool, scores[N])，坐标是原图坐标。

        box        = (x0,y0,x1,y1) 原图坐标，粗框提示（可单独用，也可配点）；
        mask_input = (1,256,256) float32 低分辨率 logits 提示（见 mask_to_input()）。
        """
        pts = list(pos) + list(neg)
        lab = [1] * len(pos) + [0] * len(neg)
        kw = {}
        if box is not None:
            kw['box'] = np.array([box[0] * self.scale, box[1] * self.scale,
                                  box[2] * self.scale, box[3] * self.scale], np.float32)
        if mask_input is not None:
            kw['mask_input'] = mask_input
        pc = self._to_model(pts) if pts else None
        pl = np.array(lab, np.int64) if lab else None
        with torch.no_grad():
            masks, scores, _ = self.pred.predict(point_coords=pc, point_labels=pl,
                                                 multimask_output=multimask and len(pts) <= 1, **kw)
        if masks.ndim == 2:
            masks, scores = masks[None], np.array([float(scores)])
        return masks, np.asarray(scores, np.float64)

    def mask_at_original(self, m):
        h0, w0 = self.orig_hw
        if m.shape[:2] != (h0, w0):
            m = cv2.resize(m.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
        return m

    def auto_proposals(self, img, points_per_side=16):
        from segment_anything import SamAutomaticMaskGenerator
        g = SamAutomaticMaskGenerator(self.sam, points_per_side=points_per_side,
                                      pred_iou_thresh=0.86, stability_score_thresh=0.92,
                                      min_mask_region_area=300)
        with torch.no_grad():
            res = g.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return [r['segmentation'] for r in res], [float(r['predicted_iou']) for r in res]


def pick_proposal(masks, scores, pick):
    """颗粒度选择。返回选中下标列表（'all' 返回全部）。"""
    n = len(masks)
    if pick in (None, '', 'score'):
        return [int(np.argmax(scores))]
    if pick == 'all':
        return list(range(n))
    if pick == 'largest':
        return [int(np.argmax([m.sum() for m in masks]))]
    if pick == 'smallest':
        return [int(np.argmin([m.sum() for m in masks]))]
    if pick.startswith('#'):
        return [int(pick[1:]) % n]
    raise SystemExit('未知 --pick：%s' % pick)


def merge_similar(mask, masks, scores, thr, skip):
    """把与当前块 IoU≥thr 的提议并进来（同一对象的多个碎片）。"""
    for i, m in enumerate(masks):
        if i in skip:
            continue
        inter = float((m & mask).sum())
        union = float((m | mask).sum())
        if union > 0 and inter / union >= thr:
            mask = mask | m
            skip = skip | {i}
    return mask, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--point', action='append', default=[])
    ap.add_argument('--neg', action='append', default=[])
    ap.add_argument('--box', default=None, help='粗框提示 x0,y0,x1,y1（原图坐标），可单独用，也可配 --point')
    ap.add_argument('--mask-input', default=None,
                    help='低分辨率 mask 提示：给一个 PNG（白=要选，黑=不要）。典型用法是把魔棒/colorblock '
                         '点出来的块当提示，SAM 负责把边界修干净（配 --point 一起用最稳）')
    ap.add_argument('--mask-scale', type=float, default=10.0,
                    help='mask 提示的幅值（默认 10；SAM 约定 256×256 logits，二值 mask 映射成 ±幅值）')
    ap.add_argument('--pick', default='largest',
                    help='score=SAM 自评分最高(UI 原生，常年给的是小碎片) | largest(默认，整片级，'
                         '在 48 例 DP 上半校准/下半报数优于 score：0.548 vs 0.482) | smallest | #0/#1/#2 | all')
    ap.add_argument('--merge', type=float, default=None)
    ap.add_argument('--auto-merge', type=float, default=None)
    ap.add_argument('--maxpix', type=int, default=1280)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlay', default=None)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()

    pos, neg = parse_pts(a.point), parse_pts(a.neg)
    box = None
    if a.box:
        box = [float(v) for v in a.box.split(',')]
        if len(box) != 4:
            raise SystemExit('--box 需要 x0,y0,x1,y1')
    if not pos and box is None and a.mask_input is None:
        raise SystemExit('至少要一个 --point X,Y、一个 --box 或一个 --mask-input')

    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit('读不到图片：%s' % a.image)

    eng = SamEngine(a.ckpt, dev='cpu' if a.cpu else None, maxpix=a.maxpix)
    t0 = time.time()
    img = eng.set_image(bgr)
    mi = None
    if a.mask_input:
        mm = cv2.imread(a.mask_input, cv2.IMREAD_GRAYSCALE)
        if mm is None:
            raise SystemExit('读不到 mask：%s' % a.mask_input)
        mi = mask_to_input(mm > 127, a.mask_scale)
    masks, scores = eng.proposals(pos, neg, box=box, mask_input=mi)
    keep = pick_proposal(masks, scores, a.pick)
    mask = np.any(masks[keep], axis=0)
    t_pick = time.time() - t0
    if a.merge is not None:
        mask, _ = merge_similar(mask, masks, scores, a.merge, set(keep))
    if a.auto_merge is not None:
        am, asc = eng.auto_proposals(img)
        mask, _ = merge_similar(mask, am, asc, a.auto_merge, set())

    mask_full = eng.mask_at_original(mask)
    ok = os.path.dirname(os.path.abspath(a.out))
    if ok:
        os.makedirs(ok, exist_ok=True)
    cv2.imwrite(a.out, (mask_full.astype(np.uint8) * 255))
    if a.overlay:
        vis = bgr.copy()
        vis[mask_full] = (0.45 * vis[mask_full] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        for x, y in pos:
            cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 255), -1)
        for x, y in neg:
            cv2.circle(vis, (int(x), int(y)), 6, (255, 0, 0), -1)
        if box:
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
        cv2.imwrite(a.overlay, vis)
    print('选中 %d 像素（占全图 %.2f%%）｜ 提议面积%% %s ｜ 分数 %s ｜ 选了 %s ｜ %.2fs ｜ %s'
          % (int(mask_full.sum()), 100 * mask_full.mean(),
             [round(100 * float(m.mean()), 2) for m in masks],
             [round(float(s), 3) for s in scores], keep, t_pick, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
