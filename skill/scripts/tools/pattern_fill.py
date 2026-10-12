"""pattern_fill.py —— 在指定区域内**填充打底**（underpaint）：纯色 / 条纹 / 格子 / 圆点。

打底是"加一个原本不存在的物件"的前置步骤：局部重绘是基于**已有像素**去改的，那块什么都没有时
结果不可控（报告实测：直出两次都失败，程序粗画的红球 → 一次重绘就成了真苹果）。
本脚本负责"画出那个形"；形的来源可以是：

  ① agent 自己按几何粗画（本脚本的 `--pattern`，或 `mask_compose.py` 的图元）
  ② **复杂图案建议人手绘**（PS 里画，或画布上涂）—— 有机/不规则/细致的图案模型与程序都难画准
  ③ **人用画布给出区域 → 本脚本填充**（`human_canvas` 涂一层，`--mask` 传那层）

用法：
    python pattern_fill.py --base BASE.png --mask MASK.png --out UNDER.png \
        [--color '#C8321E'] [--pattern flat|stripes|checker|dots] [--period 26] [--thickness 12] \
        [--angle 35] [--grow 2] [--feather 1] [--outline '#000000@3'] [--mask-out USED.png]

  · `--mask` 收灰度（白=区域）或 RGBA（alpha=0 = 区域，与 ComfyUI `LoadImage` 一致）；
  · `--pattern stripes --angle 35` 适合"带子/绳子/绑带"这类有走向的物件；
  · `--outline` 给区域描一圈边（`颜色@像素`）—— 结构件常靠一条边就能被模型认清；
  · `--mask-out` 写出实际用到的掩版（含 grow），**直接拿去重绘**（外扩一点，别贴死物件边缘）。
绝不覆盖输入。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--mask', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--color', default='#C8321E')
    ap.add_argument('--pattern', choices=['flat', 'stripes', 'checker', 'dots'], default='flat')
    ap.add_argument('--period', type=int, default=24, help='条纹/格子/点阵的周期（像素）')
    ap.add_argument('--thickness', type=int, default=None, help='条纹宽度（默认 period/2）')
    ap.add_argument('--angle', type=float, default=0.0, help='条纹角度（度）')
    ap.add_argument('--grow', type=int, default=0, help='区域外扩（打底通常比物件大一点更好）')
    ap.add_argument('--feather', type=float, default=0.0)
    ap.add_argument('--outline', default=None, help='给区域描边，格式 #RRGGBB@像素')
    ap.add_argument('--mask-out', default=None, help='写出实际使用的掩版（灰度 + RGBA 同名 .rgba）')
    a = ap.parse_args()

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    base = Image.open(a.base).convert('RGB')
    W, H = base.size
    m = Image.open(a.mask)
    if m.mode in ('RGBA', 'LA'):
        region = np.array(m.convert('RGBA').split()[-1].resize((W, H))) == 0
    else:
        region = np.array(m.convert('L').resize((W, H))) > 127
    if a.grow > 0:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
        region = cv2.dilate(region.astype('uint8'), k, iterations=1).astype(bool)
    if not region.any():
        raise SystemExit('掩版是空的——检查 --mask 的方向（灰度白 / RGBA alpha=0）')

    def hex2rgb(s):
        s = s.lstrip('#')
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype='float32')

    col = hex2rgb(a.color)
    yy, xx = np.mgrid[0:H, 0:W].astype('float32')
    th = np.radians(a.angle)
    u = xx * np.cos(th) + yy * np.sin(th)
    per = max(2, a.period)
    thick = a.thickness if a.thickness is not None else max(1, per // 2)
    if a.pattern == 'flat':
        pat = np.ones((H, W), bool)
    elif a.pattern == 'stripes':
        pat = (np.mod(u, per) < thick)
    elif a.pattern == 'checker':
        pat = ((np.mod(u, per) < per / 2) ^ (np.mod(xx * np.sin(th) + yy * np.cos(th), per) < per / 2))
    else:                                        # dots
        d = np.hypot(np.mod(xx, per) - per / 2, np.mod(yy, per) - per / 2)
        pat = d < max(1, thick / 2)

    arr = np.array(base).astype('float32')
    if a.pattern == 'flat':
        arr[region] = col
    else:                                        # 条纹/格子/点：底色（区域色）× 图案色
        base_col = col * 0.45 + 255 * 0.05
        arr[region] = base_col
        sel = region & pat
        arr[sel] = col

    out = Image.fromarray(arr.round().clip(0, 255).astype('uint8'))
    if a.outline:
        spec = a.outline.split('@')
        oc = tuple(int(v) for v in hex2rgb(spec[0]))
        ow = int(float(spec[1])) if len(spec) > 1 else 2
        import cv2
        edge = cv2.morphologyEx(region.astype('uint8'), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)) > 0
        edge = cv2.dilate(edge.astype('uint8'), np.ones((2 * ow + 1, 2 * ow + 1), np.uint8)) > 0
        ea = np.array(out).copy()
        ea[edge] = oc
        out = Image.fromarray(ea)
    if a.feather > 0:
        soft = Image.fromarray((region * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(a.feather))
        w = (np.array(soft).astype('float32') / 255.0)[..., None]
        out = Image.fromarray((np.array(base).astype('float32') * (1 - w)
                               + np.array(out).astype('float32') * w).round().clip(0, 255).astype('uint8'))
    if os.path.abspath(a.out) in {os.path.abspath(x) for x in (a.base, a.mask)}:
        raise SystemExit('--out 不能覆盖输入')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out)
    print('打底 → %s（%s，区域 %.2f%%）' % (a.out, a.pattern, 100 * region.mean()))
    if a.mask_out:
        Image.fromarray((region * 255).astype('uint8'), 'L').save(a.mask_out)
        Image.fromarray(np.dstack([np.array(out), np.where(region, 0, 255).astype('uint8')]), 'RGBA').save(
            os.path.splitext(a.mask_out)[0] + '.rgba')
        print('掩版 → %s（灰度 + 同名 .rgba，直接拿去重绘）' % a.mask_out)
    print('下一步：把打底图与掩版上传 ComfyUI，掩版重绘 denoise 0.65~0.75')
    return 0


if __name__ == '__main__':
    sys.exit(main())
