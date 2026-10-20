"""SAM 提议 + 种子归属 + **负样本做减法** —— 做"同一件东西散成几块、又和邻居同色"的物件掩版。

为什么需要它（2026-10-14 实测的失败）：色块法（colorblock）原理上分不开
**数值几乎同色的布料与皮肤**（本图 (249,233,226) vs (250,229,223)），它只会顺着颜色长过去；
而同一件衣服的**两只袖子是分开的连通块**，单类生长跨不过肩部的皮肤/发丝间隙。这两件事都不是"调参"能修的。

机制（用户 2026-10-14 的口径："SAM 的 mask 也可以参与加减法组合"）：

  1. **正种子 → 提议**：每个正种子跑一次 SAM（multimask 给 3 条），**按 SAM 自评分选**，
     只用一个面积上限挡掉"把半张图都圈进来"的提议。**不做颜色投票** —— 实测负样本（锁骨皮肤）
     与目标（白布）本来就近同色，"离谁更近"会把正确的提议也否决掉（dpos 5 / dneg 2 ⇒ 误杀）；
  2. **负种子 → 提议 → 减掉**：每个负点也跑一次 SAM，取**含该点且面积最小**的那条承载块，
     从结果里减掉 ⇒ "布料 vs 皮肤"这种同色问题由**描边与连通性**解决，而不是由颜色阈值解决；
  3. **邻接回收**（可选）：跑一次自动提议网格，把"与本类已选区域相接 ≤ join px、且不含任何负点"的
     提议并进来（两轮，允许"衣身→袖子→袖口"这种链式），这样漏给种子的碎片也能被接上；
  4. 输出逐类掩版 + 彩色叠图 + **逐块来源/分数/面积**的日志（一眼看出是哪一步出的错）。

    python scripts/sam_multi.py --image IMG --spec spec.json --out-dir out
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sam_point_mask import SamEngine                                        # noqa: E402


def med_colour(bgr, mask):
    px = bgr[mask][:, ::-1].astype(np.float32)
    return np.median(px, axis=0) if len(px) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-score', type=float, default=0.55)
    ap.add_argument('--grid', type=int, default=24, help='自动提议网格密度（0=不做邻接回收）')
    ap.add_argument('--join', type=int, default=6, help='回收时允许的邻接距离（px）')
    a = ap.parse_args()
    spec = json.load(io.open(a.spec, encoding='utf-8'))
    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit('读不到图：%s' % a.image)
    h, w = bgr.shape[:2]
    os.makedirs(a.out_dir, exist_ok=True)
    eng = SamEngine()
    eng.set_image(bgr)

    log = []
    masks = {}
    taken = np.zeros((h, w), bool)
    for c in spec['classes']:
        name = c['name']
        pos = [tuple(int(v) for v in p) for p in c['seeds']]
        neg = [tuple(int(v) for v in p) for p in c.get('negs') or []]
        max_frac = float(c.get('max_frac', 0.28))
        union = np.zeros((h, w), bool)
        for i, p in enumerate(pos):
            ms, sc = eng.proposals([p], [], multimask=True)
            cand = []
            for k, m in enumerate(ms):
                mo = eng.mask_at_original(m)
                if sc[k] < a.min_score or mo.mean() > max_frac:
                    log.append('  %-9s 正点%d(%d,%d) 提议#%d 丢弃：score %.2f 面积 %.1f%%'
                               % (name, i, p[0], p[1], k, sc[k], 100.0 * mo.mean()))
                    continue
                cand.append((float(sc[k]), int(mo.sum()), k, mo))
            if not cand:
                log.append('  %-9s 正点%d(%d,%d) **没有可用提议**' % (name, i, p[0], p[1]))
                continue
            cand.sort(key=lambda t: -t[0])
            s0, a0, k0, mo = cand[0]
            union |= mo
            log.append('  %-9s 正点%d(%d,%d) → 提议#%d score %.3f，%d px' % (name, i, p[0], p[1], k0, s0, a0))
        negm = np.zeros((h, w), bool)
        for j, p in enumerate(neg):
            ms, sc = eng.proposals([p], [], multimask=True)
            cand = []
            for k, m in enumerate(ms):
                mo = eng.mask_at_original(m)
                if mo[p[1], p[0]]:
                    cand.append((int(mo.sum()), float(sc[k]), k, mo))
            if not cand:
                log.append('  %-9s 负点%d(%d,%d) **没有含它的提议**' % (name, j, p[0], p[1]))
                continue
            cand.sort()
            a0, s0, k0, mo = cand[0]
            negm |= mo
            log.append('  %-9s 负点%d(%d,%d) → 承载块#%d score %.3f，%d px（将减掉）'
                       % (name, j, p[0], p[1], k0, s0, a0))
        before = int(union.sum())
        union = union & ~negm
        cut = before - int(union.sum())
        if before and cut > 0.35 * before:
            log.append('  ⚠ %-9s 负样本减掉了目标的 %.0f%%（%d/%d px）—— 负点可能落在目标里，先核对'
                       % (name, 100.0 * cut / before, cut, before))
        else:
            log.append('  %-9s 负样本减掉 %d px（目标 %d → %d）' % (name, cut, before, int(union.sum())))

        if a.grid and c.get('recover', True):
            props, _ps = eng.auto_proposals(eng.set_image(bgr), points_per_side=a.grid)
            added = 0
            for _ in range(2):                                    # 两轮：允许链式相接
                joined = union.copy()
                grow = cv2.dilate(union.astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.join + 1,) * 2))
                for m in props:
                    mo = eng.mask_at_original(m)
                    if (mo & union).any() or (mo & negm).any():
                        continue
                    if any(mo[y, x] for x, y in neg):
                        continue
                    if not (grow & mo.astype(np.uint8)).any():
                        continue
                    joined |= mo
                    added += 1
                if not (joined & ~union).any():
                    break
                union = joined
            log.append('  %-9s 邻接回收并入 %d 块自动提议' % (name, added))

        if c.get('grow'):
            union = cv2.dilate(union.astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (2 * int(c['grow']) + 1,) * 2)) > 0
        union &= ~taken
        taken |= union
        masks[name] = union
        cv2.imwrite(os.path.join(a.out_dir, 'mask_%s.png' % name), (union * 255).astype(np.uint8))
        mc = med_colour(bgr, union)
        log.append('  %-9s 合计 %d px（%.2f%%）中位色 %s'
                   % (name, int(union.sum()), 100.0 * union.mean(),
                      tuple(int(v) for v in mc) if mc is not None else '-'))

    ov = bgr.copy()
    for c in spec['classes']:
        m = masks[c['name']]
        col = np.array(c.get('color', [255, 0, 255]), np.uint8)
        ov[m] = (0.45 * ov[m] + 0.55 * col[::-1]).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out_dir, 'annot_overlay.png'), ov)
    txt = '\n'.join(log)
    io.open(os.path.join(a.out_dir, 'report.txt'), 'w', encoding='utf-8', newline='\n').write(txt)
    print(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
