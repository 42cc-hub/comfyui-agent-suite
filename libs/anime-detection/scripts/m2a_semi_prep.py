# -*- coding: utf-8 -*-
"""task-10 半监督素材：从未标产线池抽 N 张，用现役 det_v2 预标注写 YOLO 标签。

抽样规则（写进报告）：
  * 池 = `data/hand_gt_multi/pool_inventory.csv`（432 张唯一产线交付图）中 `labeled_in_hand_gt=0`；
  * 排除已交用户标注的 60 张（`data/hand_gt_multi/index.csv` 的 source）与 15 张 holdout —— holdout 是这 60 张的子集；
  * 按 sha256 排序后按"多人优先"分层抽：先取 total>=2 的前一半配额，其余从 total==1 补，均按 sha 定序（可复现）。

权重 0.5 的等效实现：人工标注的旋转样本各复制一份（`__w2` 后缀硬链接），伪标签样本只出现一次
⇒ 伪标签相对人工样本权重 = 1/2（报告里写明这是"重复人工样本"等效，不是真加权 loss）。
"""
import argparse
import csv
import io
import json
import os

import cv2

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MULTI = os.path.join(SUITE, 'data', 'hand_gt_multi')
OUT = os.path.join(SUITE, 'data', 'ds_det_semi')
ROT = os.path.join(SUITE, 'data', 'ds_det_rot')
CLS = {'person': 0, 'hand': 1, 'foot': 2, 'head': 3}
DET_V2 = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')


def pick(n, seed=20261021):
    inv = list(csv.DictReader(io.open(os.path.join(MULTI, 'pool_inventory.csv'), encoding='utf-8-sig')))
    queue = set(r['source'] for r in csv.DictReader(io.open(os.path.join(MULTI, 'index.csv'), encoding='utf-8-sig')))
    pool = [r for r in inv if r['labeled_in_hand_gt'] == '0' and r['stem'] not in queue]
    seen, uniq = set(), []
    for r in sorted(pool, key=lambda r: (0 if 'deliver' in r['path'] else 1, r['sha256'])):
        if r['stem'] in seen:
            continue
        seen.add(r['stem'])
        uniq.append(r)
    pool = uniq
    multi = sorted([r for r in pool if int(r['total'] or 0) >= 2], key=lambda r: r['sha256'])
    single = sorted([r for r in pool if int(r['total'] or 0) < 2], key=lambda r: r['sha256'])
    half = min(len(multi), n // 2)
    chosen = multi[:half] + single[:n - half]
    if len(chosen) < n:
        chosen += multi[half:][:n - len(chosen)]
    assert len({r['stem'] for r in chosen}) == len(chosen)
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=180)
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--imgsz', type=int, default=960)
    a = ap.parse_args()
    rows = pick(a.n)
    img_dir = os.path.join(OUT, 'images')
    lab_dir = os.path.join(OUT, 'labels')
    for d in (img_dir, lab_dir):
        os.makedirs(d, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(DET_V2)
    sel_path = os.path.join(OUT, 'selection.csv')
    with io.open(sel_path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['stem', 'path', 'sha256', 'total', 'person', 'hand', 'foot', 'head'])
        stat = {'imgs': 0, 'person': 0, 'hand': 0, 'foot': 0, 'head': 0, 'empty': 0}
        for i, r in enumerate(rows, 1):
            p = r['path']
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print('!! imread fail', p)
                continue
            h, w = img.shape[:2]
            res = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            lines = []
            cnt = {'person': 0, 'hand': 0, 'foot': 0, 'head': 0}
            for b in (res.boxes or []):
                name = res.names[int(b.cls[0])]
                if name not in CLS:
                    continue
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
                lines.append('%d %.6f %.6f %.6f %.6f' % (CLS[name], (x0 + x1) / 2 / w,
                                                        (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
                cnt[name] += 1
            if not lines:
                stat['empty'] += 1
                continue
            name = r['stem'] + '.jpg'
            cv2.imwrite(os.path.join(img_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            io.open(os.path.join(lab_dir, r['stem'] + '.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
            wr.writerow([r['stem'], p, r['sha256'], r['total'], cnt['person'], cnt['hand'], cnt['foot'], cnt['head']])
            stat['imgs'] += 1
            for k in cnt:
                stat[k] += cnt[k]
            if i % 40 == 0:
                print('...', i, stat, flush=True)
    print('semi 伪标签样本', stat)
    # 人工样本复制一份（权重 0.5 等效）
    w2_img = os.path.join(ROT, 'images_w2')
    w2_lab = os.path.join(ROT, 'labels_w2')
    for d in (w2_img, w2_lab):
        os.makedirs(d, exist_ok=True)
    n = 0
    for f in os.listdir(os.path.join(ROT, 'images')):
        stem = os.path.splitext(f)[0]
        src = os.path.join(ROT, 'images', f)
        dst = os.path.join(w2_img, stem + '__w2.jpg')
        if not os.path.exists(dst):
            os.link(src, dst)
        io.open(os.path.join(w2_lab, stem + '__w2.txt'), 'w', encoding='utf-8').write(
            io.open(os.path.join(ROT, 'labels', stem + '.txt'), encoding='utf-8').read())
        n += 1
    print('人工重复样本', n)


if __name__ == '__main__':
    main()
