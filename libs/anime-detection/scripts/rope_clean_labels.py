"""标注卫生：清掉 PS 魔棒残留的 1~9px 零星错标（用户 2026-10-13 反馈）。

判据不是"小就删"——真实的绳头/结也只有几十像素。只删**孤立**的：
  取某个 <=max_px 的连通块，看它周围 ring 宽度内的**其它已标注像素**属于哪一类；
  若周围多数是相反类 → 它是残留（魔棒漏掉的几个像素），改判成周围那一类；
  若周围多数同类 → 它本来就是绳子的一小块，保留。
改前把原文件备份到 .runtime/rope_cv/labels_backup/，随时可回滚；并给出每张图/每个斑点的坐标，
方便抽查。视觉上也把改动处放大出来看（--viz）。

用法：
  python scripts/rope_clean_labels.py --dry-run     # 只看会改什么
  python scripts/rope_clean_labels.py               # 真改（自动备份）
"""
import argparse
import glob
import io
import json
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
BAK = os.path.join(ROOT, '.runtime', 'rope_cv', 'labels_backup')
VIZ = os.path.join(ROOT, '.runtime', 'rope_cv', 'label_cleanup')


def clean_one(lab, max_px=9, ring=3):
    """返回 (新标签, 改动列表)。改动＝(类别, 面积, 质心 y, x, 改成哪一类)。"""
    out = lab.copy()
    changes = []
    k = np.ones((2 * ring + 1,) * 2, np.uint8)
    for cls in (1, 2):
        m = (lab == cls).astype(np.uint8)
        n, cc, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > max_px:
                continue
            comp = cc == i
            nb = cv2.dilate(comp.astype(np.uint8), k).astype(bool) & ~comp
            vals = lab[nb & (lab > 0)]
            if len(vals) == 0:
                continue                     # 周围没有任何标注 → 不动（看图的人自己决定）
            other = 3 - cls
            frac_other = float((vals == other).mean())
            if frac_other > 0.6:             # 周围压倒性是相反类 → 判为残留
                out[comp] = other
                changes.append((cls, int(st[i, cv2.CC_STAT_AREA]), int(cen[i][1]), int(cen[i][0]),
                                other))
    return out, changes


def main():
    ap = argparse.ArgumentParser(description='清理人标 mask 里的零星残留（魔棒漏点）')
    ap.add_argument('--src', default=RAW)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-px', type=int, default=9)
    ap.add_argument('--ring', type=int, default=3)
    ap.add_argument('--viz', action='store_true', help='把改动处放大拼图存出来')
    a = ap.parse_args()
    os.makedirs(BAK, exist_ok=True)
    os.makedirs(VIZ, exist_ok=True)
    report = []
    for lp in sorted(glob.glob(os.path.join(a.src, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        new, ch = clean_one(lab, a.max_px, a.ring)
        rec = {'image': name, 'changed_spots': len(ch),
               'changed_px': int((new != lab).sum()),
               'detail': [{'was': c, 'area': ar, 'y': y, 'x': x, 'to': t} for c, ar, y, x, t in ch]}
        report.append(rec)
        if ch:
            print('[%s] 待改 %d 处 / %d px  %s' % (name, len(ch), rec['changed_px'],
                  '; '.join('y%d,x%d %dpx %d→%d' % (d['y'], d['x'], d['area'], d['was'], d['to'])
                            for d in rec['detail'][:4])))
            if not a.dry_run:
                shutil.copy2(lp, os.path.join(BAK, name + '_labels.png'))
                cv2.imwrite(lp, new)
                mp = os.path.join(a.src, name + '_mask.png')
                if os.path.exists(mp):
                    shutil.copy2(mp, os.path.join(BAK, name + '_mask.png'))
                    cv2.imwrite(mp, ((new == 1) * 255).astype(np.uint8))
            if a.viz:
                h, w = lab.shape
                for j, d in enumerate(rec['detail'][:6]):
                    y, x = d['y'], d['x']
                    y0, y1 = max(0, y - 16), min(h, y + 16)
                    x0, x1 = max(0, x - 16), min(w, x + 16)
                    def tile(l):
                        t = np.zeros((32, 32, 3), np.uint8)
                        sub = l[y0:y1, x0:x1]
                        t[:sub.shape[0], :sub.shape[1]] = np.stack(
                            [np.where(sub == 1, 0, np.where(sub == 2, 255, 0)).astype(np.uint8),
                             np.where(sub == 1, 0, np.where(sub == 2, 255, 0)).astype(np.uint8),
                             np.where(sub == 1, 255, 0).astype(np.uint8)], -1)
                        return cv2.resize(t, (160, 160), interpolation=cv2.INTER_NEAREST)
                    pair = np.hstack([tile(lab), np.full((160, 6, 3), 255, np.uint8), tile(new)])
                    cv2.imwrite(os.path.join(VIZ, '%s_spot%02d_before_after.png' % (name, j)), pair)
    with io.open(os.path.join(VIZ, 'cleanup_report.json' if not a.dry_run else 'cleanup_dryrun.json'),
                 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    tot = sum(r['changed_px'] for r in report)
    print('%s：%d 张图共 %d 处 / %d px（备份在 %s）'
          % ('DRY-RUN' if a.dry_run else '已清理', len([r for r in report if r['changed_spots']]),
             sum(r['changed_spots'] for r in report), tot, BAK))


if __name__ == '__main__':
    main()
