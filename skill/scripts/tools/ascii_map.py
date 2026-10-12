"""ascii_map.py —— 把一块区域压成 ASCII 图，让 agent 用很少的 token“看清”版式。

    python ascii_map.py IMG.png x0,y0,x1,y1 [--step 4] [--width 96] [--mode lum|edge]

为什么需要它：让模型直接看大图既贵又容易漏；先用 ASCII 摸清"人、道具、空处各在哪"，
再决定裁哪块、让人涂哪块，比盲猜坐标稳得多。`--mode edge` 更适合找轮廓（如绳、边框）。
坐标一律用**原图像素**；输出可直接贴进对话。
"""
import argparse
import sys

RAMP_LUM = ' .:-=+*#%@'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('img')
    ap.add_argument('box', help='x0,y0,x1,y1（原图像素）')
    ap.add_argument('--step', type=int, default=4, help='采样步长（原图像素/字符）')
    ap.add_argument('--width', type=int, default=96, help='最多输出多少列')
    ap.add_argument('--mode', choices=['lum', 'edge'], default='lum')
    ap.add_argument('--invert', action='store_true')
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    im = Image.open(a.img).convert('L')
    x0, y0, x1, y1 = [int(float(v)) for v in a.box.split(',')]
    crop = np.asarray(im.crop((x0, y0, x1, y1)), dtype='float32')
    step = max(1, a.step)
    h, w = crop.shape
    rows, cols = h // step, w // step
    if cols > a.width:
        shrink = cols / float(a.width)
        step = int(step * shrink) or 1
        rows, cols = h // step, w // step
    if rows == 0 or cols == 0:
        raise SystemExit('区域太小或 step 太大')
    small = crop[:rows * step, :cols * step].reshape(rows, step, cols, step).mean(axis=(1, 3))
    if a.mode == 'edge':
        gy, gx = np.gradient(small)
        v = np.hypot(gx, gy)
    else:
        v = small
    v = (v - v.min()) / max(1e-6, (v.max() - v.min()))
    if a.invert:
        v = 1.0 - v
    print('# %s (%d,%d)-(%d,%d)  step=%d  %dx%d  mode=%s'
          % (a.img.split('/')[-1].split('\\')[-1], x0, y0, x1, y1, step, cols, rows, a.mode))
    for r in range(rows):
        print(''.join(RAMP_LUM[min(len(RAMP_LUM) - 1, int(c * len(RAMP_LUM)))] for c in v[r]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
