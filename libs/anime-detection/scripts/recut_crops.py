"""按新的外扩比例**重裁** M3 的手/脚小图（裁图名与原来一致 ⇒ `index.csv` 里已标好的 label 直接沿用）。

为什么：现在裁图只有手/脚本体（外扩 10%），模型看不到肢体连接处的上下文；
而"指/趾数量对不对"恰恰要连着手腕/脚踝看。这个脚本用来做 margin 的 A/B。

用法：python scripts/recut_crops.py --margin 0.30 --out data/crops_m30
"""
import argparse
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PRE = os.path.join(PROJ, 'data', 'prelabel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin', type=float, default=0.30)
    ap.add_argument('--min-side', type=int, default=48)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    n_h = n_f = 0
    for fn in sorted(os.listdir(PRE)):
        if not fn.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(PRE, fn), encoding='utf-8'))
        src = rec['src']
        if not os.path.exists(src):
            continue
        img = Image.open(src).convert('RGB')
        name = rec['image']
        # 手：与 prelabel_batch 同一套命名（handNN），保证标签能对上
        hands = [b for b in rec['boxes'] if b['cls'] == 'hand']
        for j, b in enumerate(hands):
            x0, y0, x1, y1 = b['box']
            w, h = x1 - x0, y1 - y0
            pad = int(a.margin * max(w, h))
            box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
            if min(box[2] - box[0], box[3] - box[1]) < a.min_side:
                continue
            img.crop(box).save(os.path.join(a.out, '%s_hand%02d.png' % (name, j + 1)))
            n_h += 1
        # 脚：同一套命名（<side>_pP）
        by = {}
        for pt in rec['points']:
            if pt['cls'] in ('left_foot', 'right_foot') and pt['vis'] >= 1:
                by.setdefault((pt['chain'], pt['cls']), []).append(pt)
        for (chain, side), pts in by.items():
            if len(pts) < 2:
                continue
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            pad = int(a.margin * max(x1 - x0, y1 - y0))
            box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
            if min(box[2] - box[0], box[3] - box[1]) < a.min_side:
                continue
            pi = chain.split('#')[-1] if '#' in chain else '0'
            img.crop(box).save(os.path.join(a.out, '%s_%s_p%s.png' % (name, side, pi)))
            n_f += 1
    print('margin=%.2f → %s：手 %d 张 / 脚 %d 张（共 %d）' % (a.margin, a.out, n_h, n_f, n_h + n_f))


if __name__ == '__main__':
    main()
