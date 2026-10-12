"""把预标注结果整理成 **ultralytics 能直接吃的数据集**（图像 + 标签同目录，train/val 划分 + yaml）。

为什么需要它：预标注把图像留在**来源目录**（逃出那 8 张在 pixiv 任务下、temple 的在 <你的图片目录>），
而 YOLO 训练要求 `images/<split>` 与 `labels/<split>` 成对出现 —— 这里负责**只拷贝**过去并写 yaml。

划分纪律：
  - **按图划分**，绝不按框/按像素随机切（同一张图的相邻裁剪会泄漏）；
  - 逃出收容所那 8 张是"已知缺陷基准"，**固定 5 张进 train、3 张进 val**（seed 固定），保证每轮都能看它们的指标；
  - 其余按来源分层随机。

用法：python scripts/build_dataset.py [--val-frac 0.2] [--copy]
"""
import argparse
import csv
import io
import json
import os
import random
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
LBL = os.path.join(DATA, 'labels')
DS_DET = os.path.join(DATA, 'ds_det')
DS_POSE = os.path.join(DATA, 'ds_pose')
DS_POSE17 = os.path.join(DATA, 'ds_pose17')
ESCAPE_BENCH = 3                      # 逃出那 8 张里固定留 3 张在 val


def split_names(recs, val_frac, rng):
    esc = [r['image'] for r in recs if r['origin'] == 'escape']
    rest = [r['image'] for r in recs if r['origin'] != 'escape']
    rng.shuffle(esc)
    rng.shuffle(rest)
    val = set(esc[:ESCAPE_BENCH])
    need = max(1, int(round(len(recs) * val_frac))) - len(val)
    val |= set(rest[:max(0, need)])
    return val


def stage(recs, val, root, label_sub, yaml_path, extra_yaml=''):
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(root, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels', sub), exist_ok=True)
    n = {'train': 0, 'val': 0}
    for r in recs:
        sub = 'val' if r['image'] in val else 'train'
        src_img = r['src']
        if not os.path.exists(src_img):
            continue
        stem = os.path.splitext(r['image'])[0]
        shutil.copy(src_img, os.path.join(root, 'images', sub, stem + os.path.splitext(src_img)[1]))
        for f in (stem + '.txt',):
            lp = os.path.join(LBL, label_sub, f)
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(root, 'labels', sub, f))
        n[sub] += 1
    with io.open(yaml_path, 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\n' % root.replace('\\', '/'))
        fh.write(extra_yaml)
    return n


FACE_CLASSES = ['face', 'eye']          # M2b 本体：只有脸/眼（在全部 1462 张 head 上都有自动标注）
FACE_EXTRA = os.path.join(LBL, 'face_extra')   # 人复核出的 gag/blindfold（256 归一化，类别序里 gag=2/blind=3）
DS_GAG = os.path.join(DATA, 'ds_face_gag')
GAG_CLASSES = ['gag', 'blindfold']      # 口塞/眼罩**专用**检测器（单任务小模型，见 stage_gag 的说明）


def head_crop_box(head_box, img_size, pad_frac=0.15):
    """复现 prelabel_batch 里 head 裁剪的框（外扩 15% 后夹到图内）—— 不依赖旧 JSON 里是否存了裁剪框。"""
    iw, ih = img_size
    x0, y0, x1, y1 = [int(v) for v in head_box]
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    return (max(0, x0 - pad), max(0, y0 - pad), min(iw, x1 + pad), min(ih, y1 + pad))


def stage_face(recs, val):
    """M2b 面部组：**输入是 head 裁剪后的 256×256**（尺度固定的"标准画布"，用户 2026-10-10 定），
    所以数据集里的"图像"就是这些 256 小图、标签也按 256 归一化 ⇒ 训练时尺度天然统一。

    坐标来源：M2b 的框是**原图坐标**，这里按对应的 head 裁剪框换算回 256 空间。
    （标注规范里那张"head 框 → 256×256 → 再检脸/眼"的流程图，落地就是这一步。）
    """
    root = os.path.join(DATA, 'ds_face')
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(root, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels', sub), exist_ok=True)
    n = {'train': 0, 'val': 0, 'boxes': 0}
    for r in recs:
        sub = 'val' if r['image'] in val else 'train'
        iw, ih = [int(v) for v in r['size'].split('x')]
        heads = [b for b in r.get('boxes', []) if b.get('layer') == 'M2a' and b.get('cls') == 'head']
        by_head = {}
        for b in r.get('boxes', []):
            if b.get('layer') == 'M2b':
                by_head.setdefault(b.get('from', ''), []).append(b)
        for head, boxes in by_head.items():
            try:
                hi = int(re.findall(r'head(\d+)', head)[0])
            except Exception:
                continue
            if hi >= len(heads):
                continue
            hx0, hy0, hx1, hy1 = head_crop_box(heads[hi]['box'], (iw, ih))
            sw, sh = float(hx1 - hx0), float(hy1 - hy0)
            if sw <= 1 or sh <= 1:
                continue
            src = os.path.join(PRE, '%s_%s.png' % (r['image'], head))
            if not os.path.exists(src):
                continue
            dst_name = '%s_%s.png' % (os.path.splitext(r['image'])[0], head)
            shutil.copy(src, os.path.join(root, 'images', sub, dst_name))
            lines = []
            for b in boxes:
                if b['cls'] not in FACE_CLASSES:
                    continue
                x0, y0, x1, y1 = b['box']
                cx, cy = ((x0 + x1) / 2.0 - hx0) / sw, ((y0 + y1) / 2.0 - hy0) / sh
                bw, bh = (x1 - x0) / sw, (y1 - y0) / sh
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
                if bw <= 0.002 or bh <= 0.002:
                    continue
                lines.append('%d %.5f %.5f %.5f %.5f' % (FACE_CLASSES.index(b['cls']), cx, cy, bw, bh))
                n['boxes'] += 1
            lp = os.path.join(root, 'labels', sub, os.path.splitext(dst_name)[0] + '.txt')
            with io.open(lp, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(lines) + ('\n' if lines else ''))
            n[sub] += 1
    with io.open(os.path.join(root, 'face.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n' % root.replace('\\', '/'))
        fh.write(''.join('  %d: %s\n' % (i, c) for i, c in enumerate(FACE_CLASSES)))
    return n


def stage_gag(val):
    """gag / blindfold **专用**小检测器（256 的 head 裁图上跑二类）。

    为什么单独建数据集、而不是把它们并进 `ds_face`：检测训练里**没标的目标就是背景**。
    本机实测（`data/m2b_queue_tags.csv`）：全池 1462 张 head 里有 **605 张** WD14 说"有口塞/眼罩"，
    而人工只覆盖了其中 **163 张**（27%）。若把它们并进 `ds_face`，剩下 442 张**很可能真有目标**的图
    会变成"这里没有"的监督 ⇒ 等于系统地教模型压制这两个类。
    所以这里只收**目标覆盖完整**的 head（gag/blindfold 都问过、都判过，来自
    `export_face_extra.py` 写的 `_coverage.csv`），其余一律不进——宁可数据少，不要假负样本。
    """
    cov = os.path.join(FACE_EXTRA, '_coverage.csv')
    if not os.path.exists(cov):
        print('gag 数据集：跳过（还没有 %s；先跑 scripts/export_face_extra.py）' % cov)
        return {'train': 0, 'val': 0, 'boxes': 0, 'skip': 0}
    root = DS_GAG
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(root, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels', sub), exist_ok=True)
    # 逃出那 8 张是**验收基准**：它们的 head 一律不进训练（实测有 1 张 test_2 的 head 混进过 train，
    # 那会让"交付图上检出"这句话不可信——第一次训练后发现的，已排除）
    escape = set()
    for fn in os.listdir(PRE):
        if fn.endswith('.json'):
            try:
                d = json.load(io.open(os.path.join(PRE, fn), encoding='utf-8'))
            except Exception:
                continue
            if d.get('origin') == 'escape':
                escape.add(d['image'])
    n = {'train': 0, 'val': 0, 'boxes': 0, 'skip': 0}
    per = {'gag': 0, 'blindfold': 0}
    with io.open(cov, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r.get('complete') != '1':
                n['skip'] += 1
                continue
            crop = r['crop']
            src = os.path.join(PRE, crop)
            if not os.path.exists(src):
                n['skip'] += 1
                continue
            img = crop.split('_head')[0]                 # 还原到"原图"，用于按图划分 train/val
            if img in escape:
                n['skip'] += 1
                continue
            sub = 'val' if img in val else 'train'
            shutil.copy(src, os.path.join(root, 'images', sub, crop))
            lines = []
            ex = os.path.join(FACE_EXTRA, os.path.splitext(crop)[0] + '.txt')
            if os.path.exists(ex):
                for ln in io.open(ex, encoding='utf-8').read().strip().splitlines():
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    ci = int(p[0])
                    if ci not in (2, 3):                 # 只搬 gag/blindfold（face_extra 里 2=gag 3=blindfold）
                        continue
                    lines.append('%d %s' % (ci - 2, ' '.join(p[1:])))   # 新类别序：0=gag 1=blindfold
                    per[GAG_CLASSES[ci - 2]] += 1
                    n['boxes'] += 1
            with io.open(os.path.join(root, 'labels', sub, os.path.splitext(crop)[0] + '.txt'),
                         'w', encoding='utf-8') as fh:
                fh.write('\n'.join(lines) + ('\n' if lines else ''))
            n[sub] += 1
    with io.open(os.path.join(root, 'gag.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n' % root.replace('\\', '/'))
        fh.write(''.join('  %d: %s\n' % (i, c) for i, c in enumerate(GAG_CLASSES)))
    n['per'] = per
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--copy', action='store_true', help='保留开关（默认就是拷贝）')
    a = ap.parse_args()
    rng = random.Random(20261010)

    recs = []
    for fn in sorted(os.listdir(PRE)):
        if fn.endswith('.json'):
            recs.append(json.load(io.open(os.path.join(PRE, fn), encoding='utf-8')))
    val = split_names(recs, a.val_frac, rng)
    print('按图划分：train %d / val %d，其中 val 固定含逃出 %d 张' %
          (len(recs) - len(val), len(val), sum(1 for r in recs if r['image'] in val and r['origin'] == 'escape')))

    classes = open(os.path.join(LBL, 'det', 'body', 'classes.txt'), encoding='utf-8').read().split()
    n = stage(recs, val, DS_DET, os.path.join('det', 'body'), os.path.join(DS_DET, 'det_body.yaml'),
              'names:\n' + ''.join('  %d: %s\n' % (i, c) for i, c in enumerate(classes)))
    print('det 数据集：%s（train %d / val %d，类别 %s）' % (DS_DET, n['train'], n['val'], classes))

    kpt = len(open(os.path.join(LBL, 'pose', 'body', 'pose_config.txt'), encoding='utf-8')
              .read().split('kpt_names:')[1].strip().strip('[]').split(','))
    n = stage(recs, val, DS_POSE, os.path.join('pose', 'body'), os.path.join(DS_POSE, 'pose_body.yaml'),
              'names:\n  0: person\nkpt_shape: [%d, 3]\nflip_idx: %s\n' %
              (kpt, list(range(kpt))))
    print('pose 数据集：%s（train %d / val %d，kpt_shape=[%d,3]）' % (DS_POSE, n['train'], n['val'], kpt))
    # COCO-17 版（复用 yolo11n-pose 的预训练头；见 export_labels.export_pose_coco17 的说明）
    if os.path.isdir(os.path.join(LBL, 'pose', 'body17')):
        from export_labels import COCO17_FLIP
        n17 = stage(recs, val, DS_POSE17, os.path.join('pose', 'body17'),
                    os.path.join(DS_POSE17, 'pose_body17.yaml'),
                    'names:\n  0: person\nkpt_shape: [17, 3]\nflip_idx: %s\n' % COCO17_FLIP)
        print('pose17(COCO-17) 数据集：%s（train %d / val %d，kpt_shape=[17,3]）'
              % (DS_POSE17, n17['train'], n17['val']))
    nf = stage_face(recs, val)
    print('face 数据集：%s（256 裁剪图 train %d / val %d，框 %d，类别 %s）'
          % (os.path.join(DATA, 'ds_face'), nf['train'], nf['val'], nf['boxes'], FACE_CLASSES))
    ng = stage_gag(val)
    if ng['train'] or ng['val']:
        print('gag 数据集（专用小检测器）：%s（**只收覆盖完整的 head** train %d / val %d，框 %d：%s；'
              '因覆盖不完整排除 %d 张）' % (DS_GAG, ng['train'], ng['val'], ng['boxes'], ng.get('per'),
                                       ng['skip']))
    print('cls 数据集在 %s（由 scripts/export_labels.py 从 index.csv 切分）' % os.path.join(DATA, 'cls'))


if __name__ == '__main__':
    main()
