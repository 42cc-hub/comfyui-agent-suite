"""建**四类面部元素**数据集（eye / mouth / gag / blindfold）—— 用户 2026-10-13 的合并方案。

背景与理由：
  · 用户决定把"眼/嘴"并进 gag/blindfold 那个检测器（同一张 256 head 画布，共享面部结构知识），
    输出只要 **眼 + 嘴**（二次元的鼻子不明显，可由眼/嘴推断 ⇒ 不标鼻子，`face` 也不再单列：
    head 框由 M2a 给，face 区域以后用 eye∪mouth 推）；
  · 数据**复用已人工判过 gag/blindfold 的 head**（`data/m2b_queue4/index.csv` 里补标了眼/嘴），
    省掉新开一批图的成本；
  · `gag` 与 `mouth` **不互斥**：口塞没完全挡住嘴时两者都要标（实测这样的 head 有 76 张）。

规矩（与老管线一致的几条，别改）：
  · 只有**四个目标都判过**的 head 才收（"没标的目标 = 背景" ⇒ 否则是假负样本）；
  · 切分**按源图分组**（`<src>_head<k>_256` 之前那段），并且**排除 origin=escape 的那几张验收图**；
  · 标签坐标按实际 crop 尺寸归一化。

用法：python scripts/build_face4.py [--val-frac 0.2]
输出：data/labels/face4/*.txt（4 类标签）+ data/ds_face4/{images,labels}/{train,val} + face4.yaml
"""
import argparse
import collections
import csv
import glob
import io
import json
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
LBL = os.path.join(DATA, 'labels', 'face4')
DS = os.path.join(DATA, 'ds_face4')
CLASSES = ['eye', 'mouth', 'gag', 'blindfold']
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
QUEUES = ['m2b_queue', 'm2b_queue2', 'm2b_queue3', 'm2b_queue4', 'm2b_queue5']


def parse_boxes(s):
    out = []
    for part in str(s or '').split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            v = [float(x) for x in part.split(',')]
        except ValueError:
            continue
        if len(v) == 4 and v[2] > v[0] and v[3] > v[1]:
            out.append(v)
    return out


def drop_contained(boxes, thr=0.8):
    """丢掉"被同 head 里更大的口塞框包住"的小框 —— 这是早期标注口径混用留下的（球 vs 球+系带）。

    用户 2026-10-12：口径该以**球+系带**为准，而这种混用正是"有些图有两个重叠口塞框"的来源；
    既然小框总是嵌在大框里，直接**取大的那个**即可，不必再让人重标。
    实测 22 张双框 head：16 对小框≥82% 落在大框内（13 对完全包含），6 对完全不相交 ⇒ 只丢包含关系。
    返回 (保留的框, 丢掉的数量)。
    """
    keep, n_drop = [], 0
    for i, b in enumerate(boxes):
        area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        if area <= 0:
            continue
        dropped = False
        for j, o in enumerate(boxes):
            if i == j:
                continue
            oa = max(0.0, (o[2] - o[0]) * (o[3] - o[1]))
            if oa <= area:                       # 只跟"更大的框"比；等大不动，避免互相删除
                continue
            ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0]))
            iy = max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
            if ix * iy >= thr * area:
                dropped = True
                break
        if dropped:
            n_drop += 1
        else:
            keep.append(b)
    return keep, n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=20261013)
    a = ap.parse_args()
    from PIL import Image

    rows = []
    for q in QUEUES:
        p = os.path.join(DATA, q, 'index.csv')
        if os.path.exists(p):
            rows += list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
    last = {}
    for r in rows:                       # 同一个 (crop,target) 只认最后一条（队列可能重复/重跑）
        if (r.get('verdict') or '').strip():
            last[(r['crop'], r['target'])] = r
    done = collections.defaultdict(set)
    for (crop, tgt) in last:
        done[crop].add(tgt)
    complete = sorted(c for c, s in done.items() if set(CLASSES) <= s)
    print('已判轮次 %d；四目标全覆盖 head %d 张' % (len(last), len(complete)))

    # escape（验收基准）的 head 一律不训练
    escape = set()
    for fn in os.listdir(PRE):
        if fn.endswith('.json'):
            try:
                d = json.load(io.open(os.path.join(PRE, fn), encoding='utf-8'))
            except Exception:
                continue
            if d.get('origin') == 'escape':
                escape.add(d['image'])

    if os.path.isdir(LBL):
        shutil.rmtree(LBL)
    os.makedirs(LBL, exist_ok=True)
    per_crop, n_skip = {}, collections.Counter()
    for crop in complete:
        src = os.path.join(PRE, crop)
        if not os.path.exists(src):
            n_skip['缺裁图'] += 1
            continue
        img = crop.split('_head')[0]
        if img in escape:
            n_skip['escape 基准'] += 1
            continue
        w, h = Image.open(src).size
        lines = []
        for tgt in CLASSES:
            r = last.get((crop, tgt))
            if not r:
                continue
            bxs = parse_boxes(r.get('boxes'))
            if tgt == 'gag':
                bxs, n_drop = drop_contained(bxs)
                n_skip['口塞去包含'] += n_drop
            for (x0, y0, x1, y1) in bxs:
                cx, cy = ((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h
                bw, bh = (x1 - x0) / float(w), (y1 - y0) / float(h)
                lines.append('%d %.5f %.5f %.5f %.5f' % (
                    CLS_IDX[tgt], min(max(cx, 0), 1), min(max(cy, 0), 1),
                    min(max(bw, 0), 1), min(max(bh, 0), 1)))
                n_skip['框:' + tgt] += 1
        per_crop[crop] = lines
        with io.open(os.path.join(LBL, os.path.splitext(crop)[0] + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
    print('标签 → %s；跳过 %s；框数 %s' % (
        LBL, dict(n_skip) if n_skip else '无',
        {k.split(':')[1]: v for k, v in n_skip.items() if k.startswith('框:')}))

    # ---- 按源图分组切分（同源图的 head 只落一边；老 ds_face 就是这里漏了才 77% 泄漏）----
    by_img = collections.defaultdict(list)
    for crop in per_crop:
        by_img[crop.split('_head')[0]].append(crop)
    imgs = sorted(by_img)
    random.Random(a.seed).shuffle(imgs)
    target = int(round(len(per_crop) * a.val_frac))
    val_imgs, n = set(), 0
    for im in imgs:
        if n >= target:
            break
        val_imgs.add(im)
        n += len(by_img[im])
    for sub in ('train', 'val'):
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(DS, d, sub), exist_ok=True)
            for f in glob.glob(os.path.join(DS, d, sub, '*')):
                os.remove(f)
    cnt = collections.Counter()
    for crop in per_crop:
        sub = 'val' if crop.split('_head')[0] in val_imgs else 'train'
        shutil.copy(os.path.join(PRE, crop), os.path.join(DS, 'images', sub, crop))
        shutil.copy(os.path.join(LBL, os.path.splitext(crop)[0] + '.txt'),
                    os.path.join(DS, 'labels', sub, os.path.splitext(crop)[0] + '.txt'))
        cnt[sub] += 1

    def srcs(sub):
        return {f.split('_head')[0] for f in os.listdir(os.path.join(DS, 'images', sub))}

    leak = srcs('train') & srcs('val')
    n_img = {s: len(os.listdir(os.path.join(DS, 'images', s))) for s in ('train', 'val')}
    n_lbl = {s: len(os.listdir(os.path.join(DS, 'labels', s))) for s in ('train', 'val')}
    with io.open(os.path.join(DS, 'face4.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n' % DS.replace('\\', '/'))
        fh.write(''.join('  %d: %s\n' % (i, c) for i, c in enumerate(CLASSES)))
    print('数据集 %s：train %d / val %d 张；**train/val 源图重叠 %d（必须为 0）**；源图 %d 张' % (
        DS, n_img['train'], n_img['val'], len(leak), len(imgs)))
    # 旧脚本这里漏了 images 的清理（只清了 labels），第二轮起会把老图留在两边 ⇒ 直接判数据集不可用
    if leak or n_img != n_lbl or n_img['train'] != cnt['train'] or n_img['val'] != cnt['val']:
        print('*** 数据集不可用：图/标签数 %s vs %s，本次写入 %s；重跑前先删掉 %s ***' % (
            n_img, n_lbl, dict(cnt), DS))
        return 1
    per_cls = collections.Counter()
    for f in glob.glob(os.path.join(LBL, '*.txt')):
        for ln in io.open(f, encoding='utf-8'):
            if ln.strip():
                per_cls[CLASSES[int(ln.split()[0])]] += 1
    print('标签总数（含 val）：%s' % dict(per_cls))


if __name__ == '__main__':
    raise SystemExit(main())
