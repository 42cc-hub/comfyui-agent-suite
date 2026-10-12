"""把 label_face.py 的复核结果落成 **M2b 的额外类别标签**（gag / blindfold）。

为什么单独一个目录：M2b 原本的 face/eye 框是预标注自动给的（`data/prelabel/*.json` 里层 M2b），
而 gag/blindfold 是人复核出来的 —— 两者来源、可信度都不同，分开存，`build_dataset.stage_face`
再把它们并成一份标签，这样重跑自动标注不会抹掉人的劳动。

标签坐标：head 裁剪图是 256×256 的标准画布（用户 2026-10-10 定的"尺度固定"），
所以归一化就是除以 256；这里按**实测尺寸**除，避免以后改了画布大小还不知道。

类别序（与 build_dataset.FACE_CLASSES 一致）：0 face, 1 eye, 2 gag, 3 blindfold

用法：python scripts/export_face_extra.py [--queue data/m2b_queue/index.csv] [--val-frac 0.2]
"""
import argparse
import collections
import csv
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
OUT = os.path.join(DATA, 'labels', 'face_extra')
FACE_CLASSES = ['face', 'eye', 'gag', 'blindfold']
CLS_IDX = {'gag': 2, 'blindfold': 3}
TARGETS = tuple(CLS_IDX)            # 覆盖完整性检查要用的目标全集


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', nargs='+', default=[os.path.join(DATA, 'm2b_queue', 'index.csv')])
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--clean', action='store_true', help='先清空输出目录（重跑时用）')
    ap.add_argument('--allow-partial', action='store_true',
                    help='默认**拒绝**导出"目标覆盖不完整"的 head（见 _coverage.csv）；这个开关只用于自查')
    a = ap.parse_args()
    if a.clean and os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for q in a.queue:
        if not os.path.exists(q):
            print('跳过（不存在）：%s' % q)
            continue
        with io.open(q, encoding='utf-8-sig') as fh:
            rows += list(csv.DictReader(fh))
    judged = [r for r in rows if (r.get('verdict') or '').strip()]
    print('队列 %d 轮，已判 %d 轮' % (len(rows), len(judged)))

    # ---- 目标覆盖检查（用户 2026-10-11 发现的问题）----
    # 检测训练里"没标的目标 = 背景"：一张 head 只问了 gag、没问 blindfold，
    # 那它上面真有眼罩就等于在教模型"这里没有眼罩"。所以只有**每个目标都问过、都判过**的 head 才收。
    asked = collections.defaultdict(set)
    done = collections.defaultdict(set)
    for r in rows:
        asked[r['crop']].add(r['target'])
        if (r.get('verdict') or '').strip():
            done[r['crop']].add(r['target'])
    complete = {c for c in asked if set(TARGETS) <= done[c]}
    partial = sorted(set(asked) - complete)
    print('head %d 张：**目标覆盖完整** %d 张 / 不完整 %d 张'
          % (len(asked), len(complete), len(partial)))
    if partial and not a.allow_partial:
        print('  ⚠ 不完整的 head 一律**不导出**（它们会让模型学到"没有口塞/眼罩"这种假负样本）。')
        print('  补覆盖：python scripts/m2b_complete_coverage.py --queue <队列>')
    cov_path = os.path.join(a.out, '_coverage.csv')
    with io.open(cov_path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'asked', 'judged', 'complete', 'boxes'])
        for c in sorted(asked):
            bx = ';'.join(r.get('boxes', '') for r in rows
                          if r['crop'] == c and (r.get('boxes') or '').strip())
            w.writerow([c, '|'.join(sorted(asked[c])), '|'.join(sorted(done[c])),
                        '1' if c in complete else '0', bx])
    print('  覆盖清单 → %s' % cov_path)

    per_crop = collections.defaultdict(list)
    stat = collections.Counter()
    seen = {}
    for r in judged:
        seen[(r['crop'], r['target'])] = r          # 同一个 (图, 目标) 只认最后一条，防队列重复
    for r in seen.values():
        v = r['verdict'].strip()
        if r['crop'] not in complete and not a.allow_partial:
            stat['覆盖不完整，跳过'] += 1
            continue
        boxes = parse_boxes(r.get('boxes'))
        stat[v] += 1
        if v in ('draw', 'pass', 'fail') and boxes:      # draw = 新口径（人直接画的框）；pass/fail = 老口径
            per_crop[r['crop']] += [(r['target'], b) for b in boxes]
            stat['有框'] += 1
        else:
            stat['判为无目标'] += 1

    from PIL import Image
    n_line = collections.Counter()
    written = set()
    for crop, items in per_crop.items():
        p = os.path.join(PRE, crop)
        if not os.path.exists(p):
            print('  找不到裁图，跳过：%s' % crop)
            continue
        w, h = Image.open(p).size
        lines = []
        for target, (x0, y0, x1, y1) in items:
            if target not in CLS_IDX:
                continue
            cx, cy = ((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h
            bw, bh = (x1 - x0) / float(w), (y1 - y0) / float(h)
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
            lines.append('%d %.5f %.5f %.5f %.5f' % (CLS_IDX[target], cx, cy, bw, bh))
            n_line[target] += 1
        if not lines:
            continue
        # 文件名要跟 build_dataset.stage_face 里 dst_name 的 stem 完全一致（`<image>_head0_256`）
        stem = os.path.splitext(crop)[0]
        with io.open(os.path.join(a.out, stem + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        written.add(stem + '.txt')

    # 清掉过期标签：上一轮导出过、这一轮因为"覆盖不完整"被排除的 crop，不能留旧文件误导人
    stale = [f for f in os.listdir(a.out) if f.endswith('.txt') and f not in written]
    for f in stale:
        os.remove(os.path.join(a.out, f))
    if stale:
        print('  清掉 %d 个过期标签（对应的 head 现在覆盖不完整/没目标）' % len(stale))

    print('\n判定分布：%s' % {k: v for k, v in stat.items()})
    print('写出 %d 张 head 的额外标签 → %s' % (len(per_crop), a.out))
    print('  框数：%s（类别序 %s）' % (dict(n_line), FACE_CLASSES))
    print('  用法：python scripts/build_dataset.py   # stage_face 会把这份并进 ds_face')


if __name__ == '__main__':
    main()
