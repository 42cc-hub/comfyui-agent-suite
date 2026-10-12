"""口塞标注的"只框了球"(ball-only) 与"球+系带"两种口径的**量化 + 定位**。

背景（用户 2026-10-12 提出）：早期标口塞时口径模糊，数据里混了两种框法 ——
  ① **只框球**：框紧贴球、接近正方形，球几乎填满框；
  ② **球+系带/扣具**（用户认为更准）：把系带一并包进去，通常明显更宽或更大。
用户给的判据："前一种的框是方形的框且有一个内切圆在里面" ⇒ 用传统 CV 抓：
  几何先验（宽高比≈1、面积小）+ 圆的证据（框内最大轮廓/霍夫圆**圆度**高、**占框面积比**高）。

输出：
  data/gag_fix/features.csv   每个口塞框的宽高比/面积/圆度/占框比/分数 + 判定（可回溯）
  data/gag_fix/sheet_top.png  判定分最高 24 个（肉眼看是不是"只有球"）
  data/gag_fix/sheet_mid.png  分数中层 24 个（看阈值切得干不干净）
  data/gag_fix/sheet_low.png  分数最低 24 个（对照组，应为"球+带"）

用法：python scripts/find_gag_ballonly.py [--thr 0.55]
"""
import argparse
import csv
import glob
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LBL = os.path.join(PROJ, 'data', 'labels', 'face4')
PRE = os.path.join(PROJ, 'data', 'prelabel')
OUT = os.path.join(PROJ, 'data', 'gag_fix')
GAG_IDX = 2                      # CLASSES = ['eye','mouth','gag','blindfold']
W = H = 256


def box_features(img, box):
    """返回 (宽高比, 面积, 圆度, 占框面积比, 圆分数)。img 是 RGB ndarray，box 是像素 x0,y0,x1,y1。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    roi = img[y0:y1, x0:x1]
    bwd, bht = x1 - x0, y1 - y0
    area = bwd * bht
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = fill = 0.0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
        ca = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if ca < 0.05 * area or p <= 0:
            continue
        c_circ = 4.0 * np.pi * ca / (p * p)
        (_, _), r = cv2.minEnclosingCircle(c)          # 圆要"贴满框"：外接圆与外接矩形都接近整框
        bx, by, bw, bh = cv2.boundingRect(c)
        c_fill = min(1.0, (np.pi * r * r) / area) * min(1.0, (bw * bh) / area)
        if c_circ * c_fill > circ * fill:
            circ, fill = float(c_circ), float(c_fill)
    hough = 0.0                                        # 霍夫圆兜底（球通常是干净的圆）
    try:
        rounds = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=100, param2=25,
                                  minRadius=int(0.18 * min(bwd, bht)),
                                  maxRadius=max(2, int(0.75 * min(bwd, bht))))
        if rounds is not None:
            for (hx, hy, hr) in rounds[0]:
                hough = max(hough, float(np.pi * hr * hr) / area)
    except Exception:
        pass
    return (bwd / float(bht), area, circ, fill, max(circ * fill, hough))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=float, default=0.55, help='圆分数阈值（≥ 判 ball-only）')
    ap.add_argument('--ar-max', type=float, default=1.45, help='宽高比上限（超过就不像"只有球"）')
    ap.add_argument('--area-max', type=float, default=2000.0, help='面积上限 px²')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows, per_head, n_missing = [], {}, 0
    for f in sorted(glob.glob(os.path.join(LBL, '*.txt'))):
        crop = os.path.basename(f)[:-4] + '.png'
        p = os.path.join(PRE, crop)
        if not os.path.exists(p):
            n_missing += 1
            continue
        img = np.array(Image.open(p).convert('RGB'))    # cv2.imread 在这些裁剪名上会返回 None
        for ln in io.open(f, encoding='utf-8'):
            q = ln.split()
            if len(q) != 5 or int(q[0]) != GAG_IDX:
                continue
            cx, cy, w, h = [float(v) for v in q[1:]]
            bx = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            ft = box_features(img, bx)
            if not ft:
                continue
            ar, area, circ, fill, score = ft
            ball = (ar <= a.ar_max) and (area <= a.area_max) and (score >= a.thr)
            rows.append(dict(crop=crop, box='%.0f,%.0f,%.0f,%.0f' % tuple(bx), ar=round(ar, 2),
                             area=int(area), circ=round(circ, 3), fill=round(fill, 3),
                             score=round(score, 3), ball_only=int(ball)))
            per_head.setdefault(crop, []).append(rows[-1])
    if not rows:
        raise SystemExit('没读到任何口塞框')
    with io.open(os.path.join(OUT, 'features.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_ball = sum(r['ball_only'] for r in rows)
    print('口塞框 %d 个（缺裁图 %d 张）｜ 判 ball-only %d 个（%.0f%%）'
          '｜ 条件：圆分数≥%.2f 且 宽高比≤%.2f 且 面积≤%.0f'
          % (len(rows), n_missing, n_ball, 100.0 * n_ball / max(1, len(rows)), a.thr, a.ar_max, a.area_max))

    multi = {c: v for c, v in per_head.items() if len(v) > 1}
    print('一张 head 有多个口塞框的：%d 张' % len(multi))
    for c, v in sorted(multi.items())[:8]:
        v = sorted(v, key=lambda r: -r['area'])
        print('   %-46s %s' % (c[:46], ' | '.join('a%d ar%.2f s%.2f%s' % (
            r['area'], r['ar'], r['score'], ' ball' if r['ball_only'] else '') for r in v)))

    srt = sorted(rows, key=lambda r: -r['score'])
    for tag, sel in (('top', srt[:24]), ('mid', srt[len(srt) // 2:len(srt) // 2 + 24]), ('low', srt[-24:])):
        cols = 6
        sheet = Image.new('RGB', (cols * 256, ((len(sel) + cols - 1) // cols) * 256), (12, 12, 12))
        for k, r in enumerate(sel):
            im = Image.open(os.path.join(PRE, r['crop'])).convert('RGB')
            d = ImageDraw.Draw(im)
            d.rectangle([float(v) for v in r['box'].split(',')],
                        outline=(0, 255, 0) if r['ball_only'] else (255, 60, 60), width=2)
            d.text((4, 4), '%s ar%.2f a%d s%.2f' % (r['crop'].split('_')[-2][:10], r['ar'], r['area'], r['score']),
                   fill=(255, 255, 0))
            sheet.paste(im, ((k % cols) * 256, (k // cols) * 256))
        sheet.save(os.path.join(OUT, 'sheet_%s.png' % tag))
    print('对照表 → %s/sheet_{top,mid,low}.png（绿框=判 ball-only、红框=没判）' % OUT)


if __name__ == '__main__':
    main()
