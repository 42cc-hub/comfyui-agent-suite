"""把预标注结果导成**训练用的统一格式**（YOLO 家族），并做 cls 数据集切分。

三条线：
  det  ← data/prelabel/*.json 的 boxes（person/hand/head + 由 M1 脚点推出的 foot）
  pose ← data/prelabel/*.json 的 points（body 18 点；手/脚的点留给各自的模型，需先做"框↔实例"关联）
  cls  ← data/cls_queue/index.csv（人工填好 label 的裁图）→ data/cls/{train,val}/<label>/

用法：python scripts/export_labels.py [--min-box 24] [--min-prob 0.25] [--val-frac 0.15]
"""
import argparse
import csv
import io
import json
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.environ.get('ADS_PRE') or os.path.join(DATA, 'prelabel')   # 换预标注源（如 prelabel_face = 面点用 detection 修正过）
GT = os.path.join(DATA, 'pose_gt')      # 人工校准过的骨架真值（scripts/label_pose.py 写出），有则优先
CROPS = os.environ.get('ADS_CROPS') or os.path.join(DATA, 'crops_todo')   # 换裁图目录做 A/B（如 crops_m30）
CLS = os.environ.get('ADS_CLS') or os.path.join(DATA, 'cls')
QUEUE = os.environ.get('ADS_QUEUE') or os.path.join(DATA, 'cls_queue')   # 换队列（如主动学习挑出的 mined 队列）
LBL = os.environ.get('ADS_LBL') or os.path.join(DATA, 'labels')          # 换标签输出目录（半监督那套写 labels_semi/）
META = os.path.join(DATA, 'meta')

DET_CLASSES = ['person', 'hand', 'foot', 'head']       # M2a 本体（chest/hip 暂无来源，见 README）
BODY18 = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'r_ear', 'l_eye', 'l_ear']


def norm_box(x0, y0, x1, y1, w, h):
    cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
    return cx, cy, (x1 - x0) / float(w), (y1 - y0) / float(h)


def export_det(min_box, min_prob):
    out = os.path.join(LBL, 'det', 'body')
    os.makedirs(out, exist_ok=True)
    with io.open(os.path.join(out, 'classes.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(DET_CLASSES) + '\n')
    n_img = n_box = 0
    raw = os.path.join(LBL, 'det', 'raw')
    os.makedirs(raw, exist_ok=True)
    for fn in sorted(os.listdir(PRE)):
        if not fn.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(PRE, fn), encoding='utf-8'))
        iw, ih = [int(v) for v in rec['size'].split('x')]
        lines, raw_lines = [], []
        for b in rec['boxes']:
            cls, (x0, y0, x1, y1), p = b['cls'], b['box'], float(b['prob'])
            if p < min_prob or min(x1 - x0, y1 - y0) < min_box:
                continue
            raw_lines.append('%-10s %.3f %.4f %.4f %.4f %.4f' %
                             (cls, p, *norm_box(x0, y0, x1, y1, iw, ih)))
            if cls in DET_CLASSES:
                lines.append('%d %.4f %.4f %.4f %.4f' %
                             (DET_CLASSES.index(cls), *norm_box(x0, y0, x1, y1, iw, ih)))
        # 脚：M1 的脚点包围盒（至少 2 个点可信）——detect 没有脚这一类，只能这么来
        per_foot = {}
        for pt in rec['points']:
            if pt['cls'] in ('left_foot', 'right_foot') and pt['vis'] >= 1:
                per_foot.setdefault(pt['chain'] + '|' + pt['cls'], []).append(pt)
        for _k, pts in per_foot.items():
            if len(pts) < 2:
                continue
            xs, ys = [p['x'] for p in pts], [p['y'] for p in pts]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            pad = int(0.35 * max(x1 - x0, y1 - y0))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(iw, x1 + pad), min(ih, y1 + pad)
            if min(x1 - x0, y1 - y0) < min_box:
                continue
            lines.append('%d %.4f %.4f %.4f %.4f' %
                         (DET_CLASSES.index('foot'), *norm_box(x0, y0, x1, y1, iw, ih)))
            raw_lines.append('%-10s %.3f %.4f %.4f %.4f %.4f' %
                             ('foot(pose)', float(sum(p['prob'] for p in pts) / len(pts)),
                              *norm_box(x0, y0, x1, y1, iw, ih)))
        stem = os.path.splitext(fn)[0]
        with io.open(os.path.join(out, stem + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        with io.open(os.path.join(raw, stem + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(raw_lines) + ('\n' if raw_lines else ''))
        n_img += 1
        n_box += len(lines)
    print('det/body: %d 张 / %d 框（另有 det/raw 保留原始类别名）' % (n_img, n_box))


# COCO-17 顺序（ultralytics yolo11n-pose 预训练头的顺序）→ 对应我们 BODY18 的下标。
# 为什么要它：用 OpenPose18（含 neck）时预训练头等于被重新初始化，实测 60 epoch 关键点 mAP50 只有 0.02；
# 换成 COCO-17 可以**复用预训练头**，这是"从零学 vs 微调"的差别。
COCO17_FROM_BODY18 = [0, 16, 14, 17, 15, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
COCO17_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
COCO17_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def export_pose_coco17(min_vis_pts=7):
    """再导一份 **COCO-17** 的骨架标签（给 yolo11n-pose 的预训练头用）。

    `min_vis_pts` 从 10 降到 7：这个门槛算的是"可信点个数"，而面点自 2026-10-13 起只保留 M2b 真检出的眼
    （其余置 vis=0，见 .runtime/prelabel_face_synth.py）。可信点分母从 18 掉到 13，10/18=0.56 对应的
    等价门槛是 0.56×13≈7。实测：面点不计入后仍用 10 会丢 476/1416 个实例，用 7 丢 263 个，
    而被丢的实例平均 prob 和保留的差不多（0.54 vs 0.55）—— 说明它们多是"人只露一半"而不是 dwpose 垃圾，
    真正的垃圾（28 个 0 点 chain）在任何门槛下都被丢掉。
    """
    out = os.path.join(LBL, 'pose', 'body17')
    os.makedirs(out, exist_ok=True)
    n_img = n_inst = n_drop = n_gt = 0
    for fn in sorted(os.listdir(PRE)):
        if not fn.endswith('.json'):
            continue
        src = os.path.join(GT, fn)                          # 人校准过的真值优先，其余用 dwpose 预标注
        if os.path.exists(src):
            n_gt += 1
        else:
            src = os.path.join(PRE, fn)
        rec = json.load(io.open(src, encoding='utf-8'))
        iw, ih = [int(v) for v in rec['size'].split('x')]
        by_chain = {}
        for pt in rec['points']:
            if pt['cls'] == 'body':
                by_chain.setdefault(pt['chain'], {})[pt['name']] = pt
        lines = []
        for chain, pts in by_chain.items():
            vis_pts = [p for p in pts.values() if p['vis'] >= 1]
            if len(vis_pts) < min_vis_pts:
                n_drop += 1
                continue
            xs = [p['x'] for p in vis_pts]
            ys = [p['y'] for p in vis_pts]
            pad = 0.10 * max(max(xs) - min(xs), max(ys) - min(ys))
            x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x1, y1 = min(iw, max(xs) + pad), min(ih, max(ys) + pad)
            row = ['0'] + ['%.4f' % v for v in norm_box(x0, y0, x1, y1, iw, ih)]
            for bi in COCO17_FROM_BODY18:
                p = pts.get('body.%s' % BODY18[bi])
                row += ['0.0000', '0.0000', '0'] if p is None else \
                       ['%.4f' % (p['x'] / iw), '%.4f' % (p['y'] / ih), str(p['vis'])]
            lines.append(' '.join(row))
            n_inst += 1
        with io.open(os.path.join(out, os.path.splitext(fn)[0] + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        n_img += 1
    with io.open(os.path.join(out, 'pose_config.txt'), 'w', encoding='utf-8') as fh:
        fh.write('kpt_shape: [17, 3]\nflip_idx: %s\nkpt_names: %s\n' % (COCO17_FLIP, COCO17_NAMES))
    print('pose/body17(COCO-17): %d 张 / %d 实例（丢弃 %d 个低置信实例；其中 %d 张用人工校准真值）'
          % (n_img, n_inst, n_drop, n_gt))


def export_pose(min_vis_pts=12, min_conf=0.5):
    """导出 M1 骨架标签。

    `min_vis_pts`：**只要"高置信点够多"的实例**。先前是 6，实测 60 epoch 后关键点 mAP50 只有 0.01
    ⇒ 怀疑是自动标注（dwpose）噪声太大、监督自相矛盾；这里把门槛提到 12 个点 vis≥1 做对照实验。
    注意：面点自 2026-10-13 起不再计入可信点（见 export_pose_coco17），这条 18 点流程若要重跑，
    等价门槛是 9 而不是 12 —— 这里没动，因为它不参与当前的姿态实验。
    """
    out = os.path.join(LBL, 'pose', 'body')
    os.makedirs(out, exist_ok=True)
    cfg = os.path.join(out, 'pose_config.txt')
    n_img = n_inst = n_drop = n_gt = 0
    for fn in sorted(os.listdir(PRE)):
        if not fn.endswith('.json'):
            continue
        src = os.path.join(GT, fn)                          # 同上：人工真值优先
        if os.path.exists(src):
            n_gt += 1
        else:
            src = os.path.join(PRE, fn)
        rec = json.load(io.open(src, encoding='utf-8'))
        iw, ih = [int(v) for v in rec['size'].split('x')]
        by_chain = {}
        for pt in rec['points']:
            if pt['cls'] == 'body':
                by_chain.setdefault(pt['chain'], {})[pt['name']] = pt
        lines = []
        for chain, pts in by_chain.items():
            vis_pts = [p for p in pts.values() if p['vis'] >= 1]
            if len(vis_pts) < min_vis_pts:
                n_drop += 1
                continue
            xs = [p['x'] for p in vis_pts]
            ys = [p['y'] for p in vis_pts]
            bw, bh = (max(xs) - min(xs)), (max(ys) - min(ys))
            pad = 0.10 * max(bw, bh)
            x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x1, y1 = min(iw, max(xs) + pad), min(ih, max(ys) + pad)
            row = ['0'] + ['%.4f' % v for v in norm_box(x0, y0, x1, y1, iw, ih)]
            for nm in BODY18:
                p = pts.get('body.%s' % nm)
                if p is None:
                    row += ['0.0000', '0.0000', '0']
                else:
                    row += ['%.4f' % (p['x'] / iw), '%.4f' % (p['y'] / ih), str(p['vis'])]
            lines.append(' '.join(row))
            n_inst += 1
        with io.open(os.path.join(out, os.path.splitext(fn)[0] + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        n_img += 1
    with io.open(cfg, 'w', encoding='utf-8') as fh:
        fh.write('kpt_shape: [%d, 3]\nflip_idx: %s\nkpt_names: %s\n' %
                 (len(BODY18), list(range(len(BODY18))), BODY18))
    print('pose/body: %d 张 / %d 个实例（丢弃 %d 个低置信实例；kpt_shape=[%d,3]；其中 %d 张用人工校准真值）'
          % (n_img, n_inst, n_drop, len(BODY18), n_gt))


def _augment(im, k):
    """按序号 k 造一个增强变体（翻转/旋转/缩放/亮度对比度/HSV 抖动/高斯模糊）。

    为什么不用"复制文件"过采样：复制等于喂重复样本，模型只会记样本不会学不变性。
    用户 2026-10-11 指出的正是这个 —— 收藏的图里几乎没有手脚出错，`bad` 类天然稀缺，
    必须靠增强把少数类的**有效样本数**做上去（判"手/脚合不合理"这件事对翻转/缩放/亮度是不变的）。
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as _np
    ops = [('hflip', k % 2 == 1),
           ('rot', (k // 2) % 3),
           ('scale', (k // 6) % 3),
           ('bc', (k // 18) % 3),
           ('hsv', (k // 54) % 3),
           ('blur', (k // 162) % 3)]
    for name, v in ops:
        if v == 0:
            continue
        if name == 'hflip':
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif name == 'rot':
            im = im.rotate([-12, 12][v - 1], resample=Image.BILINEAR, expand=False)
        elif name == 'scale':
            f = [0.8, 1.25][v - 1]
            w, h = int(im.width * f), int(im.height * f)
            im = im.resize((w, h), Image.LANCZOS)
            if f > 1:                       # 放大后中心裁回原尺寸，保证输入尺度一致
                l, t = (w - int(im.width / f)) // 2, (h - int(im.height / f)) // 2
                im = im.crop((l, t, l + int(w / f), t + int(h / f)))
        elif name == 'bc':
            im = ImageEnhance.Brightness(im).enhance([0.88, 1.12][v - 1])
            im = ImageEnhance.Contrast(im).enhance([0.9, 1.1][v - 1])
        elif name == 'hsv':
            a = _np.asarray(im.convert('RGB')).astype(_np.float32)
            a[..., 0] = _np.clip(a[..., 0] * [0.95, 1.05][v - 1], 0, 255)        # R 通道近似色相抖动
            a[..., 1] = _np.clip(a[..., 1] * [0.9, 1.1][v - 1], 0, 255)          # G 通道近似饱和度
            im = Image.fromarray(a.astype(_np.uint8))
        elif name == 'blur':
            im = im.filter(ImageFilter.GaussianBlur([0.6, 1.2][v - 1]))
    return im


def expand_minority(root, split, target, max_mult=8):
    """把 train 里的少数类用**增强变体**补到 target 张（上限 max_mult 倍于原始张数）。"""
    from PIL import Image          # 这里也要导入：上一版漏了，异常被 except 吞掉 ⇒ 静默返回 0（踩过）
    n_aug = 0
    for lab in sorted(os.listdir(os.path.join(root, split))):
        d = os.path.join(root, split, lab)
        if not os.path.isdir(d):
            continue
        orig = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
                and not any(t in f for t in ('_dup', '_aug'))]
        if not orig or len(orig) >= target:
            continue
        need = min(target - len(orig), (max_mult - 1) * len(orig))
        k = 0
        while n_aug < need and k < max_mult * len(orig):
            src = orig[k % len(orig)]
            try:
                im = Image.open(os.path.join(d, src)).convert('RGB')
                out = _augment(im, k // len(orig) + 1)
                stem, ext = os.path.splitext(src)
                out.save(os.path.join(d, '%s_aug%02d%s' % (stem, k + 1, ext)))
                n_aug += 1
            except Exception as e:
                if k == 0:
                    print('  增强失败（首个样本）：%s: %s' % (type(e).__name__, e))
            k += 1
    return n_aug


def export_cls(val_frac, balance=True, max_dup=4, collapse_bad=False, keep_notpart=False,
               pseudo_weights=None, pseudo_cap=150, aug_minority=False):
    """切 cls 数据集。

    小数据集上两个坑都踩过了，这里一并处理：
      1. **按类别分层**切分 —— 否则 val 会缺类，ultralytics 直接报 "requires N classes"；
      2. **过采样少数类**（复制文件名 `_dupN`）—— 否则分类器退化成多数类（实测 v1 把 bad_count/not_* 全判成 ok）。
    """
    idx = os.path.join(QUEUE, 'index.csv')
    if not os.path.exists(idx):
        print('cls: 还没有 data/cls_queue/index.csv（先跑 scripts/cls_queue_sheet.py）')
        return
    rows = [r for r in csv.DictReader(io.open(idx, encoding='utf-8-sig')) if r.get('label', '').strip()]
    if not rows:
        print('cls: index.csv 里还没有人工标签（label 列全空）——这一栏要等修正阶段填完')
        return
    # `skip` 按规范**不进训练集**（太小/糊/遮/穿鞋袜，判不了），单独挪到 data/cls_skip/ 留档
    skipped = [r for r in rows if r['label'] == 'skip']
    rows = [r for r in rows if r['label'] != 'skip']
    if collapse_bad:
        # 任务重构：产线真正要的是"这张能不能用" ⇒ 把 bad_count/bad_shape/not_* 合并成 bad。
        # 5 类时模型退化成多数类（实测 bad_count 0/3），二元+平衡后召回通常能起来。
        # `--keep-notpart`：把"部位认错"单独留一类（不是数量/形状问题，而是检测器错）——
        #   这类样本修的是 M2a，不该和 M3 的 bad 混在一起。
        for r in rows:
            lab = r['label']
            if lab == 'ok':
                continue
            if keep_notpart and lab in ('not_hand', 'not_foot'):
                r['label'] = 'not_part'
            else:
                r['label'] = 'bad'
    if skipped:
        d = os.path.join(DATA, 'cls_skip')
        os.makedirs(d, exist_ok=True)
        for r in skipped:
            src = os.path.join(CROPS, r['crop'])
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, r['crop']))
    for sub in ('train', 'val'):
        for lab in sorted({r['label'] for r in rows}):
            os.makedirs(os.path.join(CLS, sub, lab), exist_ok=True)
    # **按类别分层**切分：小数据集上随机切会让 val 缺类（ultralytics 会直接报 "requires N classes"），
    # 所以每个类别各自切一份 val；只有 1 张的类别留在 train（并在报告里说明它没被评估）。
    rng = random.Random(20261010)
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r['label'], []).append(r)
    # 坑（2026-10-11 自查，同类第 4 次）：原来是"按类别 shuffle 裁图"切分 ⇒ **同一张源图的裁图会同时进 train 与 val**。
    # 实测旧分法 `cls_bin_v6` 的 val 里有 **37%** 的裁图其源图也在 train（更早的 `cls` 是 81%）⇒ val 数字系统性偏乐观。
    # 现在按**源图**分组切分：一张图的全部裁图只能落在一边（与 build_dataset 的"按图划分"纪律一致）。
    by_img = {}
    for r in rows:
        by_img.setdefault(r.get('source') or r['crop'], []).append(r)
    imgs = sorted(by_img)
    rng.shuffle(imgs)
    target = max(1, int(round(len(rows) * val_frac)))
    val_imgs, n_pick = set(), 0
    for im in imgs:
        if n_pick >= target:
            break
        val_imgs.add(im)
        n_pick += len(by_img[im])
    train_rows = [r for r in rows if (r.get('source') or r['crop']) not in val_imgs]
    val_rows = [r for r in rows if (r.get('source') or r['crop']) in val_imgs]
    for lab, rs in sorted(by_cls.items()):
        nv = sum(1 for r in rs if (r.get('source') or r['crop']) in val_imgs)
        if nv == 0 and len(rs) >= 2:
            print('  ⚠ %s 这一类在 val 里是 0 张（按图分组后可能发生）⇒ 它的指标不可读' % lab)
    cnt = {}
    for sub, rs in (('train', train_rows), ('val', val_rows)):
        for r in rs:
            src = os.path.join(CROPS, r['crop'])
            if not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(CLS, sub, r['label'], r['crop']))
            cnt[(sub, r['label'])] = cnt.get((sub, r['label']), 0) + 1
    for k in sorted(cnt):
        print('  %-6s %-12s %d' % (k[0], k[1], cnt[k]))
    miss = [lab for lab, rs in by_cls.items() if len(rs) < 2]
    # 过采样：把 train 里样本少的类别按轮转复制，直到接近最大类（上限 max_dup 倍）
    # `--aug-minority` 时改用**增强变体**（翻转/旋转/缩放/亮度对比度/HSV/模糊）而不是复制文件
    if balance and train_rows and not aug_minority:
        cnt_tr = {}
        for r in train_rows:
            cnt_tr[r['label']] = cnt_tr.get(r['label'], 0) + 1
        target = max(cnt_tr.values())
        dup = 0
        for lab, n in sorted(cnt_tr.items()):
            k = min(max_dup, int(target / max(1, n)) - 1)
            pool = [r for r in train_rows if r['label'] == lab]
            for i in range(k):
                for j, r in enumerate(pool):
                    src = os.path.join(CROPS, r['crop'])
                    if not os.path.exists(src):
                        continue
                    stem, ext = os.path.splitext(r['crop'])
                    shutil.copy(src, os.path.join(CLS, 'train', lab, '%s_dup%d_%d%s' % (stem, i + 1, j, ext)))
                    dup += 1
        print('cls 过采样：为少数类额外生成 %d 张（目标每类 ~%d 张，上限 %d 倍）' % (dup, target, max_dup))
    if aug_minority:
        cnt_tr = {}
        for r in train_rows:
            cnt_tr[r['label']] = cnt_tr.get(r['label'], 0) + 1
        tgt = max(cnt_tr.values()) if cnt_tr else 0
        na = expand_minority(CLS, 'train', tgt, max_mult=8)
        print('cls 增强扩充少数类：%d 张（目标每类 ~%d 张；翻转/旋转/缩放/亮度对比度/HSV/模糊）' % (na, tgt))
    # ---- 自训练：用当前模型给"池内未人工标注"的裁图打**高置信伪标签**，只进 train（val 永远是纯人工）----
    if pseudo_weights:
        if not collapse_bad:
            print('伪标签需要 --collapse-bad（二元）才能定类别名，已跳过')
        else:
            from ultralytics import YOLO
            model = YOLO(pseudo_weights)
            names_ = model.names
            bad_i = [i for i, n in enumerate(names_) if n == 'bad']
            known = {r['crop'] for r in rows} | {r['crop'] for r in skipped}
            pool = []
            man = os.path.join(CROPS, 'manifest.csv')
            if os.path.exists(man):
                with io.open(man, encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        if r['crop'] not in known and os.path.exists(os.path.join(CROPS, r['crop'])):
                            pool.append(r['crop'])
            got = {'ok': 0, 'bad': 0}
            for crop in pool:
                if got['ok'] >= pseudo_cap and got['bad'] >= pseudo_cap:
                    break
                try:
                    res = model.predict(os.path.join(CROPS, crop), verbose=False)[0]
                    p_bad = max(res.probs.data.tolist()[j] for j in bad_i)
                except Exception:
                    continue
                lab = 'bad' if p_bad > 0.90 else ('ok' if p_bad < 0.10 else None)
                if lab is None or got[lab] >= pseudo_cap:
                    continue
                shutil.copy(os.path.join(CROPS, crop), os.path.join(CLS, 'train', lab, crop))
                got[lab] += 1
            print('自训练伪标签：%s（阈值 >0.90 / <0.10，每类上限 %d，池内候选 %d 张）'
                  % (got, pseudo_cap, len(pool)))
    print('cls: 共 %d 张（train %d / val %d，val_frac=%.2f）%s'
          % (len(rows), len(train_rows), len(val_rows), val_frac,
             '；以下类别只有 1 张、无法评估：' + ','.join(miss) if miss else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-box', type=int, default=24)
    ap.add_argument('--min-prob', type=float, default=0.25)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--collapse-bad', action='store_true',
                    help='把 bad_count/bad_shape/not_* 合并成 bad（二元任务，做 A/B 用）')
    ap.add_argument('--keep-notpart', action='store_true',
                    help='合并时把 not_hand/not_foot 单独留成 not_part 一类（三元 ok/bad/not_part）')
    ap.add_argument('--pseudo-weights', default=None,
                    help='自训练：用该权重的二元模型给池内未标注裁图打高置信伪标签（需配合 --collapse-bad）')
    ap.add_argument('--pseudo-cap', type=int, default=150)
    ap.add_argument('--aug-minority', action='store_true',
                    help='用**增强变体**（翻转/旋转/缩放/亮度对比度/HSV/模糊）扩充少数类，而不是复制文件')
    a = ap.parse_args()
    for d in (LBL, CLS, META):
        os.makedirs(d, exist_ok=True)
    export_det(a.min_box, a.min_prob)
    export_pose()
    export_pose_coco17()
    export_cls(a.val_frac, collapse_bad=a.collapse_bad, keep_notpart=a.keep_notpart,
               pseudo_weights=a.pseudo_weights, pseudo_cap=a.pseudo_cap,
               aug_minority=a.aug_minority)


if __name__ == '__main__':
    main()
