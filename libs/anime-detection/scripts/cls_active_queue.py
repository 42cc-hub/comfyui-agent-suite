"""M3 主动学习队列：用**无泄漏**的模型给未标注裁图排序，挑最该人工判的那一批。

背景（2026-10-11 夜窗口）：R43 查出线上两个 M3 权重的 val 被污染（42% / 58% 的评测裁图就在它们 train 里），
换成无泄漏权重重测后，**bad 召回只有 0.50~0.69** ⇒ M3 的瓶颈是 bad 类样本量，不是网络容量。
800 个未标注裁图里挑哪 200 个给人判，是这一步唯一能提升召回的动作。

挑选规则（三类，都留可追溯的分数）：
  · `near`    近边界：0.35 ≤ p_max ≤ 0.85 —— 模型自己犹豫的，信息量最大
  · `disagree`多模型分歧 ≥ 0.4 —— 不同权重的错误分布不同，分歧处通常是难例
  · `likely`  p_max ≥ 0.9 —— 模型很有把握的 bad，人工确认即成正样本（便宜的收益）
配额：`--quota-likely`（默认 0.35）给 likely，其余按 near/disagree 轮转；同一张源图最多 `--per-source` 张。

用法：
  python scripts/cls_active_queue.py --weights runs/ab_x_s1/weights/best.pt runs/ab_x_s2/weights/best.pt --n 200
  python scripts/cls_active_queue.py --weights runs/m3_cls_aug_v1/weights/best.pt   # 也可以拿旧权重排（仅作对比）
输出：data/cls_queue_active/index.csv（labeler 可直接 --queue 读）+ page_XX.png 联系表
"""
import argparse
import csv
import glob
import io
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CROPS = os.path.join(PROJ, 'data', 'crops_todo')
OUT = os.path.join(PROJ, 'data', 'cls_queue_active')
QUEUES = ('cls_queue', 'cls_queue_all', 'cls_queue_all2', 'cls_queue_mined', 'cls_queue_next',
          'cls_queue_mined_mismatch', 'cls_queue_next_mismatch', 'cls_queue_next_gatedout',
          'cls_queue_active')
TILE = 480
FIELDS = ['idx', 'crop', 'kind', 'source', 'origin', 'w', 'h', 'conf', 'state', 'mismatch',
          'bad_flags', 'label', 'auto_tags']


def labeled_crops():
    done = set()
    for q in QUEUES:
        p = os.path.join(PROJ, 'data', q, 'index.csv')
        if os.path.exists(p):
            with io.open(p, encoding='utf-8-sig') as fh:
                for r in csv.DictReader(fh):
                    if (r.get('label') or '').strip():
                        done.add(r['crop'])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--per-source', type=int, default=2)
    ap.add_argument('--quota-likely', type=float, default=0.35)
    ap.add_argument('--min-side', type=int, default=64)
    ap.add_argument('--min-edge', type=float, default=4.0)
    ap.add_argument('--per-page', type=int, default=6)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--no-sheet', action='store_true')
    a = ap.parse_args()
    from ultralytics import YOLO

    done = labeled_crops()
    cand = []
    with io.open(os.path.join(CROPS, 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['crop'] in done:
                continue
            p = os.path.join(CROPS, r['crop'])
            if not os.path.exists(p):
                continue
            im = Image.open(p)
            if min(im.size) < a.min_side:
                continue
            g = np.asarray(im.convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32)
            gy, gx = np.gradient(g)
            if float(np.mean(np.hypot(gx, gy))) < a.min_edge:
                continue
            cand.append(dict(r, path=p, w=im.width, h=im.height))
    print('候选（未标注 + 过门限）%d 张' % len(cand), flush=True)
    if not cand:
        return

    probs = []
    for w in a.weights:
        m = YOLO(w)
        bi = [i for i, n in m.names.items() if n == 'bad']
        tag = os.path.basename(os.path.dirname(os.path.dirname(w)))
        probs.append([max(m.predict(c['path'], verbose=False)[0].probs.data.tolist()[j] for j in bi)
                      for c in cand])
        print('  打分 %s 完成' % tag, flush=True)
    for i, c in enumerate(cand):
        ps = [pr[i] for pr in probs]
        c['p_max'] = max(ps)
        c['p_min'] = min(ps)
        c['dis'] = max(ps) - min(ps)
        c['bucket'] = ('likely' if c['p_max'] >= 0.9 else
                       'near' if 0.35 <= c['p_max'] <= 0.85 else
                       'disagree' if c['dis'] >= 0.4 else 'other')

    # 排序：先分数（越接近边界/分歧越大越靠前），再按 kind 与长边保证多样性
    def score(c):
        near = 1.0 - min(1.0, abs(c['p_max'] - 0.5) / 0.5)
        return max(near, c['dis'])

    cand.sort(key=lambda c: -score(c))
    quota_likely = int(a.n * a.quota_likely)
    picked, per_src, n_likely = [], {}, 0
    for c in cand:
        if len(picked) >= a.n:
            break
        if per_src.get(c['source'], 0) >= a.per_source:
            continue
        if c['bucket'] == 'likely':
            if n_likely >= quota_likely:
                continue
            n_likely += 1
        picked.append(c)
        per_src[c['source']] = per_src.get(c['source'], 0) + 1
    # likely 配额没用完就用剩下的补
    if len(picked) < a.n:
        for c in cand:
            if len(picked) >= a.n:
                break
            if c in picked or per_src.get(c['source'], 0) >= a.per_source + 1:
                continue
            picked.append(c)
            per_src[c['source']] = per_src.get(c['source'], 0) + 1

    picked.sort(key=lambda c: (0 if c['kind'] == 'hand' else 1, -max(c['w'], c['h'])))
    os.makedirs(OUT, exist_ok=True)
    idx_path = os.path.join(OUT, 'index.csv')
    old = {}
    if os.path.exists(idx_path):
        with io.open(idx_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                old[r['crop']] = r
    for i, c in enumerate(picked, 1):
        c['idx'] = i
        c['conf'] = round(c['p_max'], 3)
        c['state'] = ''
        for k in ('label', 'mismatch', 'bad_flags'):
            c[k] = old.get(c['crop'], {}).get(k, '')
        c['auto_tags'] = '%s|p=%.2f|dis=%.2f|n=%d' % (c['bucket'], c['p_max'], c['dis'], len(a.weights))
    with io.open(idx_path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for c in picked:
            w.writerow({k: c.get(k, '') for k in FIELDS})

    buckets = {}
    for c in picked:
        buckets[c['bucket']] = buckets.get(c['bucket'], 0) + 1
    print('队列 %d 张（手 %d / 脚 %d），桶：%s' % (
        len(picked), sum(1 for c in picked if c['kind'] == 'hand'),
        sum(1 for c in picked if c['kind'] != 'hand'), buckets))
    print('索引表 %s' % idx_path)

    if a.no_sheet:
        return
    pages = [picked[i:i + a.per_page] for i in range(0, len(picked), a.per_page)]
    for pi, page in enumerate(pages, 1):
        cols = a.cols
        rws = (len(page) + cols - 1) // cols
        bar = 26
        sheet = Image.new('RGB', (cols * TILE, rws * (TILE + bar)), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(page):
            im = Image.open(r['path']).convert('RGB')
            s = TILE / float(max(im.size))
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            cx, cy = (k % cols) * TILE, (k // cols) * (TILE + bar) + bar
            cell = Image.new('RGB', (TILE, TILE), (16, 16, 16))
            cell.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
            sheet.paste(cell, (cx, cy))
            d.text((cx + 6, cy - 18), '#%d %s %s p=%.2f dis=%.2f %dx%d' % (
                r['idx'], r['kind'], r['bucket'], r['p_max'], r['dis'], r['w'], r['h']),
                fill=(255, 220, 80))
        sheet.save(os.path.join(OUT, 'page_%02d.png' % pi))
    print('联系表 %d 页 → %s/page_*.png' % (len(pages), OUT))


if __name__ == '__main__':
    main()
