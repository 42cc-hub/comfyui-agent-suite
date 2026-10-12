"""三级数据流水线的两个工具：预标注导出（给 PS 修）与审核后升格。

三级（用户 2026-10-12 定的方案）：
  A 高质量  data/masks/rope/raw_ps/   人工标注（reviewed=true）→ 唯一训练真值
  B 中等质量 data/masks/rope/pre/     模型预标注(+SAM 对象级剔除) → 人只需"改几笔"再存回
  C 无标注  H:\FILE\temple\_PIC*      半监督池，只出不进
B 经人检查修正 → 升格进 A（`promote`），于是"标一张图"从 5~6 分钟降到 1~2 分钟。

用法：
  # 1) 预标注：对一批图跑模型，导出成 PS 能直接改的"红=绳 / 蓝=非绳"图
  python scripts/rope_prelabel_ps.py --src <图或目录> --out data/masks/rope/pre [--sam]
  # 2) 人在 PS 里改完存回 pre/<name>_ps.png 后，升格为高质量
  python scripts/rope_prelabel_ps.py promote --src data/masks/rope/pre --name <name>
"""
import argparse
import glob
import io
import json
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
PRE = os.path.join(ROOT, 'data', 'masks', 'rope', 'pre')
OL = os.path.join(ROOT, '.runtime', 'rope_cv', 'overlay')


def parse_ps_mask(path):
    """PS 存回的"红=绳 蓝=非绳"图 → (labels 0/1/2, 原样红蓝计数)。"""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    B, G, R = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    rope = (R > 140) & (R - G > 70) & (R - B > 70)
    other = (B > 140) & (B - R > 70) & (B - G > 70)
    lab = np.zeros(im.shape[:2], np.uint8)
    lab[other] = 2
    lab[rope] = 1
    return lab, int(rope.sum()), int(other.sum()), im.shape[:2]


def sam_refine(mask, img, ckpt, model_type='vit_b'):
    """用 SAM 把预测的连通块吸附到对象边界，并丢掉"不像绳"的大团块。

    做法：把每个连通块的重心当 SAM 的 point prompt，取回对象级 mask；
    若对象 mask 面积远大于连通块（说明模型把一整块物体当成绳了，例如红格子裙），
    或对象形状不细长，就丢掉该块。这就是"对象级先验"能一刀切掉的那类误检。
    """
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as e:
        print('  (未装 segment_anything，跳过 SAM 剔除: %s)' % e)
        return mask, 0
    if not os.path.exists(ckpt):
        print('  (缺 SAM 权重 %s，跳过)' % ckpt)
        return mask, 0
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(dev)
    pred = SamPredictor(sam)
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    n, cc, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    dropped = 0
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < 64:
            continue
        cx, cy = cen[i]
        m, sc, _ = pred.predict(point_coords=np.array([[cx, cy]]),
                                point_labels=np.array([1]), multimask_output=True)
        best = m[int(np.argmax(sc))]
        comp = cc == i
        # 对象 mask 面积不能远大于连通块（否则就是"整块物体"被误当绳）
        if best.sum() > 4.0 * area:
            dropped += 1
            continue
        out |= (best & comp) | (comp & cv2.dilate(best.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    return out, dropped


def prelabel(a):
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rope_predict as RP
    os.makedirs(a.out, exist_ok=True)
    m = RP.load_model(a.weights)
    files = RP.resolve(a.src)
    print('权重 %s  thr %.3f  待预标注 %d 张' % (os.path.basename(a.weights), m['thr'], len(files)))
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        base = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')
        if os.path.exists(os.path.join(RAW, base + '_labels.png')):
            print('[%s] 已有高质量标注，跳过' % base)
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        prob = RP.dense_prob(hsv, m, backend=a.backend)
        mask = prob >= a.conf if a.conf else prob >= m['thr']
        dropped = 0
        if a.sam:
            mask, dropped = sam_refine(mask, img, a.sam_ckpt)
        # 导出"红=绳 蓝=其余"的 PS 可改图（纯色，便于我按颜色解析回来）
        ps = np.zeros_like(img)
        ps[...] = (255, 0, 0)                      # BGR: 蓝
        ps[mask] = (0, 0, 255)                     # BGR: 红
        cv2.imwrite(os.path.join(a.out, base + '_ps.png'), ps)
        cv2.imwrite(os.path.join(a.out, base + '_orig_copy.png'), img)
        lab = np.zeros(img.shape[:2], np.uint8)
        lab[~mask] = 2
        lab[mask] = 1
        cv2.imwrite(os.path.join(a.out, base + '_labels.png'), lab)
        cv2.imwrite(os.path.join(a.out, base + '_mask.png'), (mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OL, base + '_prelabel_overlay.png'), RP.overlay(img, mask))
        with io.open(os.path.join(a.out, base + '_meta.json'), 'w', encoding='utf-8') as fh:
            json.dump({'image': os.path.basename(fp), 'size': [img.shape[1], img.shape[0]],
                       'tier': 'B_pre', 'reviewed': False, 'sam_dropped_blocks': dropped,
                       'rope_px': int(mask.sum()), 'weights': os.path.basename(a.weights),
                       'thr': float(a.conf or m['thr']),
                       'howto': 'PS 里改 %s_ps.png（红=绳 蓝=非绳），存回同路径后跑 promote' % base},
                      fh, ensure_ascii=False, indent=1)
        print('[%s] 预标注完成：绳 %d px(%.2f%%) SAM 剔除 %d 块 → %s_ps.png'
              % (base, mask.sum(), 100 * mask.mean(), dropped, base))


def promote(a):
    src = os.path.join(a.src, a.name + '_ps.png')
    r = parse_ps_mask(src) if os.path.exists(src) else None
    if r is None:
        print('找不到 %s，人在 PS 里改完请存回这个路径' % src)
        return
    lab, nrope, nother, shape = r
    if nrope < 100 and nother < 100:
        print('解析不出红/蓝（红 %d 蓝 %d），确认用的是纯 #FF0000 / #0000FF 且没混进原图' % (nrope, nother))
        return
    os.makedirs(RAW, exist_ok=True)
    cv2.imwrite(os.path.join(RAW, a.name + '_labels.png'), lab)
    cv2.imwrite(os.path.join(RAW, a.name + '_mask.png'), ((lab == 1) * 255).astype(np.uint8))
    meta_p = os.path.join(a.src, a.name + '_meta.json')
    meta = json.load(io.open(meta_p, encoding='utf-8')) if os.path.exists(meta_p) else {}
    meta.update({'tier': 'A_human', 'reviewed': True, 'rope_px': nrope, 'nonrope_px': nother,
                 'promoted_from': 'pre/' + a.name, 'source': 'pre-annotation + human review'})
    with io.open(os.path.join(RAW, a.name + '_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print('[%s] 已升格为高质量：绳 %d px / 非绳 %d px → %s' % (a.name, nrope, nother, RAW))
    print('   记得确认原图在 .runtime/rope_cv/ps_work/_orig/ 或 pre/ 里以便训练时配对')


def main():
    ap = argparse.ArgumentParser(description='绳子三级数据：预标注导出 + 审核升格')
    ap.add_argument('mode', nargs='?', default='prelabel', choices=['prelabel', 'promote'])
    ap.add_argument('--src', default='')
    ap.add_argument('--out', default=PRE)
    ap.add_argument('--name', default='', help='promote 用：图名（不含 _ps.png）')
    ap.add_argument('--weights', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'weights',
                                                      'rope_mlp_k9_8img.npz'))
    ap.add_argument('--conf', type=float, default=0.0)
    ap.add_argument('--backend', default='auto', choices=['auto', 'torch', 'numpy'])
    ap.add_argument('--sam', action='store_true', help='用 SAM 做对象级剔除')
    ap.add_argument('--sam-ckpt', default=os.path.join(
        os.path.dirname(ROOT), 'models', 'sams', 'sam_vit_b_01ec64.pth'),
        help='本机已有 SAM 权重（ComfyUI 的 models/sams/），无需另外下载')
    a = ap.parse_args()
    (prelabel if a.mode == 'prelabel' else promote)(a)


if __name__ == '__main__':
    main()
