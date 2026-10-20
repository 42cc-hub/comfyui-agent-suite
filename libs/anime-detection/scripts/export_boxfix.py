"""把 **M2a 手/脚框修复** 的标注队列导成两种可直接用的产物。

队列里的框是**裁图坐标**，`note` 里记着 `src=`（源图）与 `crop_org=`（裁图原点），
所以回源图坐标只是一步加法；本脚本同时给出"源图口径"和"裁图口径"两套产物：

  1. `data/boxfix/gt_boxes.csv`  —— **真值清单**（源图坐标）：
     `src,kind,x0,y0,x1,y1,idx,crop,edge`
     正样本 = 画了框；负样本（verdict=neg）也写进来（`neg` 列标 1，无框），
     并给一条"该区域里没有这个目标"的记录，供后面做 hard negative。
  2. `data/boxfix/ds/`          —— **YOLO 直接可训**（裁图口径，0=hand / 1=foot）：
     `images/*.png`（复制裁图）+ `labels/*.txt`（归一化框；负样本给空 label）+ `data.yaml`。
     裁图口径的好处：目标占画面比例稳定、样本小；缺点：尺度与产线不一致
     ⇒ 建议**混进**既有 M2a 数据里微调，而不是单独训（见 README）。

用法：python scripts/export_boxfix.py [--queue data/boxfix/index.csv]
"""
import argparse
import csv
import io
import os
import shutil
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CLASSES = {'hand': 0, 'foot': 1}


def parse_note(note):
    d = {}
    for part in (note or '').split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(PROJ, 'data', 'boxfix', 'index.csv'))
    ap.add_argument('--out', default=os.path.join(PROJ, 'data', 'boxfix'))
    ap.add_argument('--edge-tol', type=int, default=2, help='离裁图边界多少 px 内算"贴边"（说明外扩不够）')
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(a.queue))
    crops = os.path.join(root, 'crops')
    rows = list(csv.DictReader(io.open(a.queue, encoding='utf-8-sig')))
    ds_img = os.path.join(a.out, 'ds', 'images')
    ds_lab = os.path.join(a.out, 'ds', 'labels')
    os.makedirs(ds_img, exist_ok=True)
    os.makedirs(ds_lab, exist_ok=True)

    out, n_pos, n_neg, n_edge, n_skip = [], 0, 0, 0, 0
    for r in rows:
        v = (r.get('verdict') or '').strip()
        if not v:
            n_skip += 1
            continue
        note = parse_note(r.get('note'))
        src = note.get('src')
        org = [int(x) for x in (note.get('crop_org') or '0,0').split(',')]
        kind = r['target']
        cp = os.path.join(crops, r['crop'])
        if not src or not os.path.exists(cp) or not os.path.exists(src):
            print('  ⚠ 跳过 %s（源图或裁图不在）' % r['crop'])
            n_skip += 1
            continue
        W, H = Image.open(src).size
        cw, ch = Image.open(cp).size
        boxes = [b for b in (r.get('boxes') or '').split(';') if b.strip()]
        edge = ''
        if v == 'draw':
            n_pos += 1
            for b in boxes:
                x0, y0, x1, y1 = [float(t) for t in b.split(',')]
                if x0 <= a.edge_tol or y0 <= a.edge_tol or x1 >= cw - a.edge_tol or y1 >= ch - a.edge_tol:
                    edge = 'edge'
                sx0, sy0 = max(0.0, x0 + org[0]), max(0.0, y0 + org[1])
                sx1, sy1 = min(float(W), x1 + org[0]), min(float(H), y1 + org[1])
                out.append({'src': src, 'kind': kind, 'x0': int(round(sx0)), 'y0': int(round(sy0)),
                            'x1': int(round(sx1)), 'y1': int(round(sy1)), 'neg': 0,
                            'idx': r['idx'], 'crop': r['crop'], 'edge': edge})
            # 裁图口径的 YOLO 标签
            lines = []
            for b in boxes:
                x0, y0, x1, y1 = [float(t) for t in b.split(',')]
                cx, cy = (x0 + x1) / 2.0 / cw, (y0 + y1) / 2.0 / ch
                bw, bh = abs(x1 - x0) / cw, abs(y1 - y0) / ch
                lines.append('%d %.6f %.6f %.6f %.6f' % (CLASSES.get(kind, 0), cx, cy, bw, bh))
            shutil.copy2(cp, os.path.join(ds_img, r['crop']))
            io.open(os.path.join(ds_lab, os.path.splitext(r['crop'])[0] + '.txt'), 'w',
                    encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
        else:                                    # neg：这一块里没有这个目标 ⇒ hard negative
            n_neg += 1
            out.append({'src': src, 'kind': kind, 'x0': org[0], 'y0': org[1],
                        'x1': org[0] + cw, 'y1': org[1] + ch, 'neg': 1,
                        'idx': r['idx'], 'crop': r['crop'], 'edge': ''})
            shutil.copy2(cp, os.path.join(ds_img, r['crop']))
            io.open(os.path.join(ds_lab, os.path.splitext(r['crop'])[0] + '.txt'), 'w',
                    encoding='utf-8', newline='\n').write('')   # 空标签 = 纯负样本
        if edge:
            n_edge += 1

    fp = os.path.join(a.out, 'gt_boxes.csv')
    with io.open(fp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['src', 'kind', 'x0', 'y0', 'x1', 'y1', 'neg', 'idx',
                                           'crop', 'edge'])
        w.writeheader()
        for r in out:
            w.writerow(r)
    with io.open(os.path.join(a.out, 'ds', 'data.yaml'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('path: %s\n' % os.path.join(a.out, 'ds').replace('\\', '/'))
        fh.write('train: images\nval: images\nnames:\n  0: hand\n  1: foot\n')
    print('真值清单 %s（%d 条：正 %d / 负 %d）' % (fp, len(out),
                                            sum(1 for r in out if not r['neg']),
                                            sum(1 for r in out if r['neg'])))
    print('YOLO 数据集 %s（%d 图 / %d 标签；0=hand 1=foot）'
          % (os.path.join(a.out, 'ds'), len(os.listdir(ds_img)), len(os.listdir(ds_lab))))
    print('画框的 %d 条里 %d 条贴裁图边界（外扩不够，建议加大 --pad 重裁后复核）' % (n_pos, n_edge))
    print('跳过 %d 条（未判/源图缺失）' % n_skip)
    return 0


if __name__ == '__main__':
    sys.exit(main())
