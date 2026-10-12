"""从**预标注里的 FOOT3 脚点**导出"远端脚区"检测数据集，并训练一个专用小检测器。

为什么需要它（R39）：M3 的训练裁图 = `prelabel_batch` 用 dwpose 的 FOOT3（趾/跟三点）包围盒 + 0.35 外扩；
而产线 `locate.py` 用的是**检测器脚框 + 0.10**，两者不是同一个区域（实测同一只脚 p(bad) 1.00 → 0.00）。
踝点+膝方向的几何规则**标定不出来**（在 1006 只脚上最高 IoU 只有 0.29），所以改成**学**：
FOOT3 的框本来就在我们的预标注里（1006 个，零人工成本）⇒ 训一个单类检测器输出这种框，
产线用它切 M3 输入，训练/推断口径就一致了。

用法：
  python scripts/export_foot_distal.py                      # 出数据集
  python scripts/train.py --task det --data <abs>/data/ds_foot_distal/foot_distal.yaml \
      --model yolo11n.pt --epochs 80 --imgsz 768 --name m2a_foot_distal_v1
"""
import argparse
import io
import json
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
OUT = os.path.join(DATA, 'ds_foot_distal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--min-side', type=int, default=24)
    a = ap.parse_args()
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(OUT, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(OUT, 'labels', sub), exist_ok=True)

    recs = []
    for fn in sorted(os.listdir(PRE)):
        if fn.endswith('.json'):
            recs.append(json.load(io.open(os.path.join(PRE, fn), encoding='utf-8')))
    import sys
    sys.path.insert(0, HERE)
    from build_dataset import split_names                     # 复用同一套按图划分（含逃出固定 val）
    val = split_names(recs, a.val_frac, random.Random(20261010))
    escape = {r['image'] for r in recs if r.get('origin') == 'escape'}

    n = {'train': 0, 'val': 0, 'boxes': 0, 'skip': 0}
    for r in recs:
        if not os.path.exists(r['src']):
            continue
        img = None
        iw, ih = [int(v) for v in r['size'].split('x')]
        lines = []
        for side in ('left_foot', 'right_foot'):
            pts = [p for p in r.get('points', []) if p.get('cls') == side and p.get('vis', 0) >= 1]
            if len(pts) < 2:
                continue
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if x1 - x0 < a.min_side or y1 - y0 < a.min_side:
                n['skip'] += 1
                continue
            cx, cy = ((x0 + x1) / 2.0) / iw, ((y0 + y1) / 2.0) / ih
            bw, bh = (x1 - x0) / float(iw), (y1 - y0) / float(ih)
            lines.append('0 %.5f %.5f %.5f %.5f' % (cx, cy, min(bw, 1.0), min(bh, 1.0)))
            n['boxes'] += 1
        if not lines:
            continue
        sub = 'val' if r['image'] in val else 'train'
        ext = os.path.splitext(r['src'])[1] or '.png'
        dst_img = r['image'] + ext
        base = r['image']
        try:
            if img is None and not os.path.exists(os.path.join(OUT, 'images', sub, dst_img)):
                shutil.copy(r['src'], os.path.join(OUT, 'images', sub, dst_img))
        except Exception as e:
            print('  拷贝失败 %s: %s' % (r['image'], e))
            continue
        with io.open(os.path.join(OUT, 'labels', sub, base + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        n[sub] += 1

    with io.open(os.path.join(OUT, 'foot_distal.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: foot_distal\n'
                 % OUT.replace('\\', '/'))
    print('远端脚区数据集：%s（train %d / val %d，框 %d，跳过 %d —— 逃出那 8 张里的 %d 张仍在 val）'
          % (OUT, n['train'], n['val'], n['boxes'], n['skip'], len(val & escape)))
    print('训练：python scripts/train.py --task det --data %s --model yolo11n.pt --epochs 80 '
          '--imgsz 768 --name m2a_foot_distal_v1' % os.path.join(OUT, 'foot_distal.yaml'))


if __name__ == '__main__':
    main()
