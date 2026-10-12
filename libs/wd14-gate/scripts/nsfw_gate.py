"""nsfw_gate.py —— 读图前的自检闸门。

为什么需要它：agent 用视觉工具看图 = **把这张图上传到模型服务商**。一旦画面露骨，
账号有被封的风险（用户 2026-09-24 明确提出）。所以在 `read_image` 之前先跑这个脚本：
用客观指标估一遍风险，超过阈值就**不要看**，改为放进 `quarantine/` 交给人过目
（`dsh_show_media` 是本地渲染给人看的通道，不走模型 ✓）。

指标（都不依赖外部模型，纯像素统计）：
  - 肤色像素占比：YCbCr 经典规则（R>95, G>40, B>20, max-min>15, |R-G|>15, R>G, R>B,
    且 80<=Cb<=120, 133<=Cr<=173）
  - 最大连通肤色块的占比（零散小面积肤色多数是手/脸，大面积整块才可疑）

判定：
  skin < 0.12 且最大块 < 0.05   -> SAFE        可以看
  skin < 0.22 或 最大块 < 0.10   -> WARN        谨慎，最好让人过目
  否则                           -> QUARANTINE  不要看，交给人

用法：
    python nsfw_gate.py a.png [b.png ...]
    python nsfw_gate.py --move-quarantine out.png a.png b.png ...   # 把危险图移进 quarantine/
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage


def skin_stats(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    rule = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) &
            (np.abs(r - g) > 15) & (r > g) & (r > b) &
            (cb >= 80) & (cb <= 120) & (cr >= 133) & (cr <= 173))
    frac = float(rule.mean())
    lab, n = ndimage.label(rule)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        big = float(sizes.max() / rule.size)
    else:
        big = 0.0
    if frac < 0.07 and big < 0.03:
        verdict = "SAFE"
    elif frac < 0.12 and big < 0.06:
        verdict = "WARN"
    else:
        verdict = "QUARANTINE"
    return dict(path=path, size=im.size, skin=frac, biggest_blob=big, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--move-quarantine", default=None,
                    help="给一个目录：QUARANTINE 的图会被移动过去（默认只报告，不动文件）")
    a = ap.parse_args()
    worst = "SAFE"
    for p in a.images:
        if not os.path.exists(p):
            print("MISSING %s" % p)
            continue
        st = skin_stats(p)
        print("%-10s skin=%5.1f%%  biggest=%5.1f%%  %s" %
              (st["verdict"], 100 * st["skin"], 100 * st["biggest_blob"], os.path.basename(p)))
        if st["verdict"] == "QUARANTINE" and a.move_quarantine:
            os.makedirs(a.move_quarantine, exist_ok=True)
            shutil.move(p, os.path.join(a.move_quarantine, os.path.basename(p)))
            print("           -> 已移入 %s（不要用视觉工具看它 ✓）" % a.move_quarantine)
        if st["verdict"] != "SAFE":
            worst = st["verdict"]
    print("OVERALL=%s" % worst)


if __name__ == "__main__":
    main()
