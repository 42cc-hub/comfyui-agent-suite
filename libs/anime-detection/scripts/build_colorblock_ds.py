r"""给"色块分区"标注准备数据：**一张图切 9 块（3×3、1/3 边长、带随机抖动），每块一对 原图/副本**。

为什么要切片而不是整图（用户 2026-10-14 的口径）：整图里块太多、颜色列表不够用；而实际用的时候
要选的就是**一小块**，切片后每张结构简单、标注负担低。

**尺度归一化**（用户特别提的那点）：交付稿里有些是过了放大程序的（`_2000`/`_2x`/`_4x` 命名，
同目录 `pre_upscale/` 里留着原生版），直接按像素切会让"切片里的内容尺度"差好几倍。做法：
  1. 有 `pre_upscale/` 同名件 → 以它的尺寸为**原生尺度**，放大倍数 = 交付尺寸 / 原生尺寸（精确）；
  2. 没有就按文件名/尺寸推断倍数（`_2000` ⇒ 交付尺度，原生长边通常 ~1216）；
  3. **只降不升**：长边 > `--long`(默认 2000) 的缩到 2000；原生就小于 `--min-long`(默认 1600) 的**不要**
     （那种是草稿，放大它会造假细节）。

产物：
  <out>/originals/<item>.png   切片原图
  <out>/copies/<item>.png      **同名副本**（人在 PS 里改这个并覆盖）
  <out>/index.csv              清单（item / 源图 / 切片框 / 尺寸 / 放大倍数）
  <out>/README.md              怎么标 + 这次做了什么

用法：python scripts/build_colorblock_ds.py --out data/colorblock_ds --images 20 --seed 20261014
"""
import argparse
import csv
import glob
import io
import os
import random
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
POOL = r'<你的 ComfyUI 项目>\task'
EXTS = ('.png', '.jpg', '.jpeg', '.webp')
UP_MARK = ('_2000', '_2x', '_3x', '_4x', 'upscale', '_1800')


def pool_files():
    """交付稿 + 它们的 pre_upscale 原生版；按 (task, stem) 去重，优先原生。"""
    up, native = {}, {}
    for p in glob.glob(os.path.join(POOL, '*', 'upload', '*')):
        if os.path.splitext(p)[1].lower() not in EXTS or not os.path.isfile(p):
            continue
        task = os.path.basename(os.path.dirname(os.path.dirname(p)))
        stem = os.path.splitext(os.path.basename(p))[0]
        up[(task, stem)] = p
    for p in glob.glob(os.path.join(POOL, '*', 'upload', 'pre_upscale', '*')):
        if os.path.splitext(p)[1].lower() not in EXTS or not os.path.isfile(p):
            continue
        task = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        stem = os.path.splitext(os.path.basename(p))[0]
        native[(task, stem)] = p
    return up, native


def upscale_info(path, native_path, w, h):
    """返回 (是否放大过, 原生长边估计, 倍数说明)。"""
    long_ = max(w, h)
    if native_path:
        with Image.open(native_path) as im:
            nl = max(im.size)
        return True, nl, '%.2fx (pre_upscale 实测 %d→%d)' % (long_ / float(nl), nl, long_)
    name = os.path.basename(path).lower()
    if any(m in name for m in UP_MARK) and long_ > 1600:
        nl = 1216 if long_ / 1216 > 1.15 else long_
        return True, nl, '%.2fx (文件名标记推断)' % (long_ / float(nl))
    return False, long_, '1.00x (原生)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'data', 'colorblock_ds'))
    ap.add_argument('--images', type=int, default=20, help='选多少张源图')
    ap.add_argument('--tiles', type=int, default=9, help='每图几块（默认 9 = 3x3）')
    ap.add_argument('--tile-frac', type=float, default=1 / 3.0)
    ap.add_argument('--jitter', type=float, default=0.12, help='每块的随机偏移（占块边长比例）')
    ap.add_argument('--long', type=int, default=2000, help='归一化目标长边（只降不升）')
    ap.add_argument('--min-long', type=int, default=1600, help='原生长边下限，低于此值不用（草稿）')
    ap.add_argument('--seed', type=int, default=20261014)
    ap.add_argument('--upscaled-frac', type=float, default=0.5,
                    help='这批里"来自放大过的源图"占多少（池子里 171 张有 59 张带标记，'
                         '随机抽容易一张都抽不到 —— 2026-10-14 第一版就抽到 0 张）。'
                         '归一化会同时作用于两类，混着给才能看出差别')
    ap.add_argument('--clear', action='store_true', help='先清空输出目录（重跑时用）')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    up, native = pool_files()

    def big_enough(k):
        try:
            with Image.open(up[k]) as im:      # 只读文件头，很快
                return max(im.size) >= a.min_long
        except Exception:
            return False

    marked = [k for k in up if (any(m in k[1].lower() for m in UP_MARK) or native.get(k)) and big_enough(k)]
    plain = [k for k in up if k not in set(marked) and big_enough(k)]
    rng.shuffle(marked)
    rng.shuffle(plain)
    n_up_want = int(round(a.images * a.upscaled_frac))
    keys = marked[:n_up_want] + plain[:max(0, a.images - n_up_want)]
    rng.shuffle(keys)
    print('候选：放大过 %d 张（要 %d）/ 原生 %d 张（要 %d）'
          % (len(marked), n_up_want, len(plain), max(0, a.images - n_up_want)))
    o_dir = os.path.join(a.out, 'originals')
    c_dir = os.path.join(a.out, 'copies')
    if a.clear and os.path.isdir(a.out):
        for sub in ('originals', 'copies'):
            d = os.path.join(a.out, sub)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    os.remove(os.path.join(d, fn))
    os.makedirs(o_dir, exist_ok=True)
    os.makedirs(c_dir, exist_ok=True)
    rows, picked = [], 0
    for key in keys:
        if picked >= a.images:
            break
        p = up[key]
        np_ = native.get(key)
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        w, h = im.size
        is_up, nl, note = upscale_info(p, np_, w, h)
        # 尺寸决策只看**文件自身尺寸**（只降不升）：放大与否只作为记录。
        # 早先拿"推断的原生长边"当依据 ⇒ 放大稿（推断原生 1216）会被 min-long 当草稿整批拒掉（实测只出 3 张）。
        long_ = max(w, h)
        if long_ < a.min_long:                   # 真正的草稿尺度（本身就没有 1600px），放大它只会造假细节
            continue
        scale = (a.long / float(long_)) if long_ > a.long else 1.0
        if scale < 1.0:
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
        W, H = im.size
        tile = int(round(max(W, H) * a.tile_frac))
        tile = max(tile, 320)
        cols = rows_per = int(round(a.tiles ** 0.5))
        xs = [0, (W - tile) // 2, W - tile]
        ys = [0, (H - tile) // 2, H - tile]
        n_tile = 0
        for r_i, y0 in enumerate(ys):
            for c_i, x0 in enumerate(xs):
                jx = int(rng.uniform(-a.jitter, a.jitter) * tile)
                jy = int(rng.uniform(-a.jitter, a.jitter) * tile)
                x0 = max(0, min(W - tile, x0 + jx))
                y0 = max(0, min(H - tile, y0 + jy))
                box = (x0, y0, x0 + tile, y0 + tile)
                item = '%s__%s__t%d%d.png' % (key[0], key[1][:40], r_i + 1, c_i + 1)
                crop = im.crop(box)
                crop.save(os.path.join(o_dir, item))
                crop.save(os.path.join(c_dir, item))          # 副本 = 同字节，人改它并覆盖
                rows.append({'item': item, 'source': p, 'native': np_ or '', 'native_long': nl,
                             'long_after': max(W, H), 'scale_applied': round(scale, 3),
                             'upscaled': int(is_up), 'upscale_note': note,
                             'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3],
                             'tile': tile, 'w': crop.width, 'h': crop.height})
                n_tile += 1
        picked += 1
        print('  [%2d/%d] %-46s %dx%d  %s  切 %d 块' % (picked, a.images, key[1][:44], w, h, note, n_tile))
    with io.open(os.path.join(a.out, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    n_up = sum(1 for r in rows if r['upscaled'])
    readme = os.path.join(a.out, 'README.md')
    with io.open(readme, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# 色块分区标注数据（%d 张源图 → %d 块）\n\n' % (picked, len(rows)))
        fh.write('## 怎么标\n\n')
        fh.write('1. 在 **`copies/`** 里打开切片（`originals/` 是原样备份，别动）；\n')
        fh.write('2. 把图里**每一块**（同色/同对象的连通区域）用不同**纯色**涂出来；\n')
        fh.write('3. 存回**同一个文件名**（`Ctrl+S` / 导出覆盖），格式 PNG、尺寸别改；\n')
        fh.write('4. 判不了的块涂**纯黑**当"忽略"（跨块混在一起、边界是判断而非测量时用）。\n\n')
        fh.write('配色规则（**这样颜色永远不会不够用**）：\n\n')
        fh.write('* 只需要**同一张图内相邻的两块不同色**；不相邻的块、以及不同图之间，**可以复用同一批颜色**；\n')
        fh.write('* 建议固定一个小调色板（例如 8 色：红/绿/蓝/黄/青/品红/橙/紫）循环用，人眼也好看；\n')
        fh.write('* 同一个颜色出现两次且**互不相邻**时，工具按连通域拆开，不会被当成一块。\n\n')
        fh.write('## 这次做了什么\n\n')
        fh.write('* 每张源图 **3×3 = 9 块**，块边长 = 长边的 1/3，位置上带 ±12%% 随机抖动；\n')
        fh.write('* 块大小随图自适应（本批 %d–%d px），因此切片里的"块"尺度是可比的；\n'
                 % (min(r['tile'] for r in rows), max(r['tile'] for r in rows)))
        fh.write('* **尺度归一化**：长边 > %d 的按比例缩到 %d（只降不升，避免造假细节）；'
                 '本批来自"放大过"的源图 **%d/%d 张**（`index.csv` 的 `upscaled`/`upscale_note` 列有依据）；\n'
                 % (a.long, a.long, n_up, picked))
        fh.write('* 原生就小于 %d 的（草稿尺度）直接不要。\n\n' % a.min_long)
        fh.write('## 清单\n\n`index.csv`：item / 源图路径 / 原生长边 / 归一化后长边 / scale / 是否放大过 / 切片框。\n')
    print('\n源图 %d 张 → %d 块（其中来自"放大过"的 %d 块）' % (picked, len(rows), n_up))
    print('切片原图: %s' % o_dir)
    print('副本目录: %s   ← 人在 PS 里编辑并覆盖这里' % c_dir)
    print('清单: %s' % os.path.join(a.out, 'index.csv'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
