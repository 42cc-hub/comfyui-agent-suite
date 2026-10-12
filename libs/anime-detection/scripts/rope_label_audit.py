#!/usr/bin/env python
"""标签颜色审计：现有 30 张标注里，"正样本（绳）"到底是什么颜色。

为什么现在做：用户说"标注里已经有别的颜色的绳了，目标是绳子不是红绳"。
而文档里写的是"只标红绳，白绳/棕绳算负样本"。两者必须对齐——
如果不一致，就是**监督信号自相矛盾**（同类外观一半标正一半标负），
这比换输入色彩空间重要得多，也直接决定"该不该继续用 HSV"这个问题的答案。

分类（对 lab==1 的像素，用 OpenCV uint8 HSV）：
  red     H≤12 或 H≥168，S≥90
  brown   12<H≤28，S≥70，V≥40
  white   S<45            （白/灰绳、高光）
  dark    V<45            （暗部/黑绳/阴影）
  other   其余（绿/蓝/紫/黄/粉等真正"别的颜色"）
  自检：shadow 与 dark 会覆盖一部分红绳暗部，所以 red/brown 是**下界**，other 才是关键证据。
"""
import glob
import io
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
ORIG = os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work', '_orig')
W = os.path.join(ROOT, '.runtime', 'rope_cv')
BUCKETS = ['red', 'brown', 'white', 'dark', 'other']


def find_image(name):
    for e in ('.png', '.jpg', '.jpeg', '.webp'):
        p = os.path.join(ORIG, name + e)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(ORIG, name + '_*')))
    return g[0] if g else None


def classify(h, s, v):
    red = ((h <= 12) | (h >= 168)) & (s >= 90)
    brown = (~red) & (h > 12) & (h <= 28) & (s >= 70) & (v >= 40)
    white = (~red) & (~brown) & (s < 45)
    dark = (~red) & (~brown) & (~white) & (v < 45)
    other = ~(red | brown | white | dark)
    return {'red': red, 'brown': brown, 'white': white, 'dark': dark, 'other': other}


def main():
    rows = []
    tot = {k: 0 for k in BUCKETS}
    tot_pos = 0
    for lp in sorted(glob.glob(os.path.join(RAW, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        ip = find_image(name)
        if ip is None:
            print('!! 找不到原图:', name)
            continue
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cls = classify(h, s, v)
        pos = lab == 1
        neg = lab == 2
        npos = int(pos.sum())
        rec = {'name': name, 'pos': npos, 'neg': int(neg.sum())}
        if npos:
            for k in BUCKETS:
                c = int((pos & cls[k]).sum())
                rec[k] = c
                rec[k + '_pct'] = round(100.0 * c / npos, 2)
                tot[k] += c
            tot_pos += npos
            # "other" 里最多的色相区间（每 12 度一档）
            oh = h[pos & cls['other']]
            rec['other_top_h'] = (int(np.bincount(oh // 12, minlength=15).argmax() * 12) if oh.size else -1)
            hist = np.bincount((h[pos] // 12).astype(np.int32), minlength=15)
            rec['hue_hist'] = [int(x) for x in hist]
            # 真正"不是红/粉"的正样本（H 30~150 是绿青蓝紫）；粉/品红单独一档（150~168）
            rec['far_red_pct'] = round(100.0 * float(((h[pos] > 30) & (h[pos] < 150)).sum()) / npos, 2)
            rec['pink_pct'] = round(100.0 * float(((h[pos] >= 150) & (h[pos] < 168) & (s[pos] >= 120)).sum()) / npos, 2)
            rec['S_mean'] = round(float(s[pos].mean()), 1)
            rec['V_mean'] = round(float(v[pos].mean()), 1)
        else:
            for k in BUCKETS:
                rec[k] = 0
                rec[k + '_pct'] = 0.0
        # 负样本里的"红"占比：这是我们故意留的陷阱（红衣物），太低说明标注太干净
        rec['neg_red_pct'] = round(100.0 * float((neg & cls['red']).sum()) / max(int(neg.sum()), 1), 2)
        rows.append(rec)

    lines = ['# 标签颜色审计：正样本（绳）的色相构成', '',
             '分类：red H≤12|H≥168&S≥90；brown 12<H≤28&S≥70&V≥40；white S<45；dark V<45；other 其余。',
             '注意：红绳的暗部/高光会被算进 dark/white，所以 red 是**下界**；关键看 **other**（真正别的颜色）。', '',
             '| 图 | 绳px | red% | brown% | white% | dark% | **other%** | **确非红%** | 粉/品红% | other主色相 | S均 | V均 | 负样本里red% |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda x: -x['other_pct']):
        lines.append('| %s | %d | %.1f | %.1f | %.1f | %.1f | **%.1f** | %.1f | %.1f | %s | %.0f | %.0f | %.1f |'
                     % (r['name'], r['pos'], r['red_pct'], r['brown_pct'], r['white_pct'],
                        r['dark_pct'], r['other_pct'], r.get('far_red_pct', 0), r.get('pink_pct', 0),
                        ('H=%d' % r['other_top_h']) if r.get('other_top_h', -1) >= 0 else '-',
                        r.get('S_mean', 0), r.get('V_mean', 0), r['neg_red_pct']))
    lines += ['', '## 全体合计（%d 张有绳像素的图，共 %d 绳px）' % (len(rows), tot_pos), '',
              '| 类别 | px | 占比 |', '|---|---|---|']
    for k in BUCKETS:
        lines.append('| %s | %d | %.2f%% |' % (k, tot[k], 100.0 * tot[k] / max(tot_pos, 1)))
    lines += ['', '## 文件', '- 原图目录：`%s`' % ORIG,
              '- 标签目录：`%s`' % RAW,
              '- 本审计：`.runtime/rope_cv/label_color_audit.md`']
    with io.open(os.path.join(W, 'label_color_audit.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines[:8]))
    print('...')
    print('\n'.join(lines[-len(BUCKETS) - 5:]))


def labels_sheet(out_path, cell=300, cols=5):
    """把每张已标图的"原图 + 标签叠加"排成一张接触表：用来人工找
    "被标成负类的非红绳"（换了类别定义后必须复盘的东西）。"""
    items = []
    for lp in sorted(glob.glob(os.path.join(RAW, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        ip = find_image(name)
        if ip is None:
            continue
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if int((lab == 1).sum()) < 2000:      # 纯负样本图不用看叠加
            continue
        ov = img.copy()
        ov[lab == 1] = (0.35 * ov[lab == 1] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        ov[lab == 2] = (0.65 * ov[lab == 2] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        sc = min(cell / img.shape[1], cell / img.shape[0], 1.0)
        items.append((name, cv2.resize(ov, (max(1, int(img.shape[1] * sc)),
                                           max(1, int(img.shape[0] * sc))))))
    ch, cw = cell + 26, cell
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * ch, cols * cw, 3), 245, np.uint8)
    for i, (name, th) in enumerate(items):
        r, c = divmod(i, cols)
        canvas[r * ch + 26:r * ch + 26 + th.shape[0], c * cw:c * cw + th.shape[1]] = th
        cv2.putText(canvas, name, (c * cw + 4, r * ch + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print('标签复盘表（红=正类/绳，蓝=负类）→ %s（%d 张）' % (out_path, len(items)))


if __name__ == '__main__':
    import sys
    if '--labels-sheet' in sys.argv:
        labels_sheet(os.path.join(W, 'label_review_sheet.png'))
    else:
        main()
