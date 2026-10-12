#!/usr/bin/env python
"""mask_compose.py —— **掩版组合器**：把"检测结果 / 几何图元 / 已有掩版"拼成你要的重绘区。

为什么单独有它（2026-10-13 用户提的 agent-in-the-loop）：不必指望 `check_parts.py` 一次就给出
"恰好对"的那张掩版。检测已经把**可靠的东西**（person/手/脸/头框、逐人 silhouette 及其零件）落盘了，
剩下的"这一块要改哪里"由 agent 按需要自己组合 —— 本脚本就是那个组合器，**不需要 GPU、秒级**。

图元与来源（全部取并集，再减掉 `--sub`）：
  --box x0,y0,x1,y1      矩形（可给多个，逗号分隔 x0,y0,x1,y1,x0,y0,x1,y1…）
  --circle x,y,r         圆（指尖那种"小圆"最适合）
  --poly "x,y;x,y;..."   多边形（自己描一圈）
  --union FILE.png       已有的掩版文件（例如 check_parts 出的 face0 / hand1 / person0 / parts/skel_*）
  --sub FILE.png         从结果里**减掉**（例如减掉手里那根杆的掩版）
修饰：--shrink N（图元统一内缩）、--grow N（最后外扩）、--feather N（边缘羽化，给潜空间 8× 下采样留余量）
输出：灰度 PNG（**255 = 要重绘的区域**）+ 可选 ComfyUI 语义 RGBA（重绘区 alpha=0）

用法示例：
  # 指尖小圆 + 脸框下半（来自 check_parts 的 feature 掩版），羽化 8px
  python mask_compose.py --like base.png --circle 430,360,74 --union out/mask/feature/base_mouth0.png \
      --grow 6 --feather 8 --out mask_hand.png --rgba base.png
  # 取某个人并去掉另一人：person0 减 person1
  python mask_compose.py --like base.png --union out/mask/base_person0.png --sub out/mask/base_person1.png \
      --out only_p0.png
"""
import argparse
import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    import cv2
    import numpy as np
    from PIL import Image
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--like', default=None, help='按这张图的尺寸建掩版（与 --size 二选一）')
    ap.add_argument('--size', default=None, help='WxH')
    ap.add_argument('--box', action='append', default=[], help='x0,y0,x1,y1[,x0,y0,x1,y1...]（可多次）')
    ap.add_argument('--circle', action='append', default=[], help='x,y,r（可多次）')
    ap.add_argument('--poly', default=None, help='"x,y;x,y;..."')
    ap.add_argument('--union', nargs='+', action='extend', default=[], help='并进来的掩版文件（可多次/可多个）')
    ap.add_argument('--sub', nargs='+', action='extend', default=[], help='减掉的掩版文件（可多次/可多个）')
    ap.add_argument('--shrink', type=int, default=0, help='图元统一内缩 N px（0=不动）')
    ap.add_argument('--grow', type=int, default=0, help='最后统一外扩 N px')
    ap.add_argument('--feather', type=int, default=0, help='边缘羽化 N px（建议 8~16，避免硬边接缝）')
    ap.add_argument('--out', required=True, help='输出灰度掩版（255 = 重绘区）')
    ap.add_argument('--rgba', default=None, help='额外输出 ComfyUI 语义 RGBA（需要给一张原图当底）')
    a = ap.parse_args()

    # 防呆：**绝不允许覆盖输入图**（`--rgba` 指到 `--like` 那张图上会把原图改写成带透明区的 RGBA，
    # 2026-10-13 我自己踩过一次 —— 虽然 RGB 通道没变、可还原，但下游会把"半透明底图"当成新素材）
    def same(p, q):
        return p and q and os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(q))

    for flag, val in (('--rgba', a.rgba), ('--out', a.out)):
        for src_flag, src in (('--like', a.like), ('--rgba', a.rgba)):
            if val and same(val, src) and (flag, val) != (src_flag, src):
                ap.error('%s 不能和 %s 指向同一个文件（%s）—— 会覆盖输入' % (flag, src_flag, val))

    if a.like:
        W, H = Image.open(a.like).size
    elif a.size:
        W, H = [int(v) for v in a.size.lower().split('x')]
    else:
        ap.error('需要 --like 或 --size')
    m = np.zeros((H, W), np.uint8)

    def load(p):
        im = Image.open(p)
        if im.size != (W, H):
            im = im.resize((W, H), Image.NEAREST)
        arr = np.asarray(im.convert('L'))
        return (arr > 127).astype(np.uint8)

    for spec in a.box:
        vals = [int(v) for v in spec.replace(' ', '').split(',')]
        for i in range(0, len(vals) - 3, 4):
            x0, y0, x1, y1 = vals[i:i + 4]
            if a.shrink:
                x0, y0, x1, y1 = x0 + a.shrink, y0 + a.shrink, x1 - a.shrink, y1 - a.shrink
            m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1
    for spec in a.circle:
        x, y, r = [int(v) for v in spec.replace(' ', '').split(',')]
        r = max(1, r - a.shrink)
        cv2.circle(m, (x, y), r, 1, -1)
    if a.poly:
        pts = np.array([[int(v) for v in p.split(',')] for p in a.poly.split(';') if p.strip()], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
        else:
            print('mask_compose: --poly 至少 3 个点', file=sys.stderr)
    for p in a.union:
        if not os.path.exists(p):
            print('mask_compose: 找不到 --union %s' % p, file=sys.stderr)
            continue
        m |= load(p)
    for p in a.sub:
        if not os.path.exists(p):
            print('mask_compose: 找不到 --sub %s' % p, file=sys.stderr)
            continue
        m &= (1 - load(p))
    if a.grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
        m = cv2.dilate(m, k, iterations=1)
    gray = (m * 255).astype(np.uint8)
    if a.feather > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), a.feather / 2.0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(gray, 'L').save(a.out)
    print('掩版 %s：面积 %.2f%%（%d px，%dx%d）%s' % (
        a.out, 100.0 * float((m > 0).mean()), int((m > 0).sum()), W, H,
        '，羽化 %d px' % a.feather if a.feather else ''))
    if a.rgba:
        base = a.like or a.rgba
        img = np.asarray(Image.open(base).convert('RGB').resize((W, H)))
        # ComfyUI 的 LoadImage MASK = 1 − alpha ⇒ 重绘区 alpha=0
        alpha = 255 - (gray if a.feather else (m * 255).astype(np.uint8))
        Image.fromarray(np.dstack([img, alpha]), 'RGBA').save(a.rgba)
        print('RGBA %s（重绘区 alpha=0；羽化时 alpha 是软边）' % a.rgba)


if __name__ == '__main__':
    main()
