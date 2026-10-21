# -*- coding: utf-8 -*-
"""batch-2 人框真值评测（阶段 A），并与 batch-1 用同一套口径重算做对比。

用法（先跑推理再打分，或只打分）：
  python -X utf8 scripts/m2a_person_gt2_eval.py --run      # CPU 推理（不占 GPU 票）+ 打分
  python -X utf8 scripts/m2a_person_gt2_eval.py            # 只打分（用已有 preds）

口径：conf 0.30 / imgsz 960 / 一对一贪心 IoU 0.5；person 框只来自 fix_v2（现役双头混合的 person 分支）。
"""
import csv
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REP = os.path.abspath(os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier'))
OUT = os.path.join(REP, 'person_gt2')
PRED_DIR = os.path.join(OUT, 'preds')
IMG2 = os.path.join(SUITE, 'data', 'hand_gt_multi', 'images')
GT2 = os.path.join(SUITE, 'data', 'hand_gt_multi', 'index_person_topup.csv')
GT1 = os.path.join(SUITE, 'data', 'hand_gt_multi', 'index.csv')
QPRED = os.path.join(SUITE, 'data', 'hand_gt_multi', 'preds', 'queue_preds.json')
CONF = 0.30


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def boxes_of(s):
    return [tuple(float(v) for v in seg.split(',')) for seg in (s or '').split(';') if seg.strip()]


def sha(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest().upper()


def load_gt2():
    gt = {}
    for r in csv.DictReader(io.open(GT2, encoding='utf-8-sig')):
        if r['verdict'].strip() == 'neg':
            continue
        b = boxes_of(r['boxes'])
        if b:
            gt.setdefault(r['crop'], []).extend(b)
    return gt


def load_gt1():
    gt = {}
    for r in csv.DictReader(io.open(GT1, encoding='utf-8-sig')):
        if r['target'] != 'person' or not r['boxes'].strip() or r['verdict'] == 'neg':
            continue
        gt.setdefault(r['crop'], []).extend(boxes_of(r['boxes']))
    return gt


def preds_queue():
    """batch-1 的缓存预测（det person @conf>=0.30）"""
    raw = json.load(io.open(QPRED, encoding='utf-8'))
    out = {}
    for stem, v in raw.items():
        ps = [p['box'] for p in v.get('det', {}).get('person', []) if float(p.get('conf', 1)) >= CONF]
        out[stem + '.png'] = ps
    return out


def preds_dir(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        out[f[:-5] + '.png'] = [b['box'] for b in rec.get('boxes', []) if b['cls'] == 'person']
    return out


def evaluate(gt, pr, tag, sizes):
    tp = fp = fn = 0
    best_ious, ratios, offs, offs_img = [], [], [], []
    gt_area_frac, miss_by_bucket = [], {}
    for crop, g in gt.items():
        p = pr.get(crop, [])
        used = set()
        for gb in g:
            # 最佳 IoU（不设阈值，用于分布）
            bi, bo = -1, 0.0
            for i, pb in enumerate(p):
                v = iou(gb, pb)
                if v > bo:
                    bo, bi = v, i
            best_ious.append(bo)
            W, H = sizes.get(crop, (2000, 2000))
            fa = (gb[2] - gb[0]) * (gb[3] - gb[1]) / float(W * H)
            gt_area_frac.append(fa)
            bk = ('<2%' if fa < 0.02 else '2-5%' if fa < 0.05 else '5-10%' if fa < 0.10
                  else '10-20%' if fa < 0.20 else '>=20%')
            if bo >= 0.5:
                # 一对一：占掉这个 pred
                taken = None
                for i in sorted(range(len(p)), key=lambda i: -iou(gb, p[i])):
                    if i in used:
                        continue
                    if iou(gb, p[i]) >= 0.5:
                        taken = i
                        break
                if taken is not None:
                    used.add(taken)
                    tp += 1
                    pb = p[taken]
                    ratios.append(((pb[2] - pb[0]) * (pb[3] - pb[1])) /
                                  max(1e-6, (gb[2] - gb[0]) * (gb[3] - gb[1])))
                    gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                    pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    offs.append(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 /
                                max(1e-6, ((gb[2] - gb[0]) ** 2 + (gb[3] - gb[1]) ** 2) ** 0.5))
                    offs_img.append(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 /
                                    ((W ** 2 + H ** 2) ** 0.5))
                else:
                    fn += 1
                    miss_by_bucket[bk] = miss_by_bucket.get(bk, 0) + 1
            else:
                fn += 1
                miss_by_bucket[bk] = miss_by_bucket.get(bk, 0) + 1
        fp += len(p) - len(used)
    n = len(best_ious)
    out = {
        'tag': tag, 'images_with_gt': len(gt), 'gt_boxes': n, 'pred_boxes': sum(len(v) for v in pr.values()),
        'tp': tp, 'fp': fp, 'fn': fn,
        'recall50': round(100.0 * tp / n, 1) if n else 0.0,
        'precision50': round(100.0 * tp / (tp + fp), 1) if tp + fp else 0.0,
        'iou_median': round(statistics.median(best_ious), 3) if best_ious else 0.0,
        'iou_lt50_pct': round(100.0 * sum(1 for v in best_ious if v < 0.5) / n, 1) if n else 0.0,
        'iou_lt70_pct': round(100.0 * sum(1 for v in best_ious if v < 0.7) / n, 1) if n else 0.0,
        'area_ratio_median': round(statistics.median(ratios), 3) if ratios else 0.0,
        'area_ratio_p10': round(sorted(ratios)[max(0, int(0.10 * len(ratios)) - 0)], 3) if ratios else 0.0,
        'area_ratio_p90': round(sorted(ratios)[min(len(ratios) - 1, int(0.90 * len(ratios)))], 3) if ratios else 0.0,
        'center_off_gt_diag_median_pct': round(100 * statistics.median(offs), 2) if offs else 0.0,
        'center_off_img_diag_median_pct': round(100 * statistics.median(offs_img), 2) if offs_img else 0.0,
        'miss_by_gt_area_bucket': miss_by_bucket,
        'gt_by_area_bucket': {k: sum(1 for f in gt_area_frac if (
            '<2%' if f < 0.02 else '2-5%' if f < 0.05 else '5-10%' if f < 0.10
            else '10-20%' if f < 0.20 else '>=20%') == k) for k in ('<2%', '2-5%', '5-10%', '10-20%', '>=20%')},
    }
    return out


def run_inference(gt, weights=None, pred_dir=None):
    from PIL import Image
    import shutil
    PRED_DIR_ = pred_dir or PRED_DIR
    shutil.rmtree(PRED_DIR_, ignore_errors=True)
    os.makedirs(PRED_DIR_, exist_ok=True)
    imgs = [os.path.join(IMG2, c) for c in sorted(gt)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1')
    cmd = [sys.executable, '-X', 'utf8', os.path.join(SUITE, 'scripts', 'locate.py'),
           '--src'] + imgs + ['--out', PRED_DIR_, '--no-vitpose', '--conf', str(CONF),
                              '--imgsz', '960', '--hand-imgsz', '960', '--foot-weights', '',
                              '--face-weights', '', '--gag-weights', '', '--m3-weights', '']
    if weights:
        cmd += ['--det-weights', weights]
    t0 = __import__('time').time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    print('推理 rc=%d %.1fs' % (p.returncode, __import__('time').time() - t0))
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit(1)


def img_sizes(gt):
    from PIL import Image
    out = {}
    for c in gt:
        p = os.path.join(IMG2, c)
        if os.path.exists(p):
            with Image.open(p) as im:
                out[c] = im.size
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    gt2, gt1 = load_gt2(), load_gt1()
    weights, pred_dir, tag = None, None, 'batch2_hybrid_conf0.30_960'
    argv = sys.argv[1:]
    if '--weights' in argv:
        weights = argv[argv.index('--weights') + 1]
        tag = 'batch2_%s_conf0.30_960' % os.path.basename(os.path.dirname(os.path.dirname(weights)))
    if '--pred-dir' in argv:
        pred_dir = argv[argv.index('--pred-dir') + 1]
    if '--run' in argv:
        run_inference(gt2, weights, pred_dir)
    sizes = img_sizes(gt2)
    sizes.update(img_sizes(gt1))
    p2 = preds_dir(pred_dir or PRED_DIR) if os.path.isdir(pred_dir or PRED_DIR) else {}
    p1 = preds_queue()
    r2 = evaluate(gt2, p2, tag, sizes)
    r1 = evaluate(gt1, p1, 'batch1_baseline_conf0.30_960', sizes)
    rep = dict(gt2=dict(path=GT2, sha256=sha(GT2), rows=sum(1 for _ in csv.DictReader(io.open(GT2, encoding='utf-8-sig'))),
                       boxes=sum(len(v) for v in gt2.values()), images=len(gt2)),
               gt1=dict(path=GT1, sha256=sha(GT1), boxes=sum(len(v) for v in gt1.values()), images=len(gt1)),
               batch2=r2, batch1=r1)
    lines = ['batch-2 人框评测（阶段 A）', '=' * 60,
             'GT2 %s sha=%s rows=%d boxes=%d images=%d' % (GT2, rep['gt2']['sha256'], rep['gt2']['rows'],
                                                           rep['gt2']['boxes'], rep['gt2']['images']),
             'GT1 %s sha=%s boxes=%d images=%d' % (GT1, rep['gt1']['sha256'], rep['gt1']['boxes'], rep['gt1']['images']),
             '', '%-34s %10s %10s' % ('指标', 'batch-2', 'batch-1'),
             '%-34s %10s %10s' % ('GT 框数', r2['gt_boxes'], r1['gt_boxes']),
             '%-34s %10s %10s' % ('预测框数', r2['pred_boxes'], r1['pred_boxes']),
             '%-34s %10s %10s' % ('R@IoU.5 (%)', r2['recall50'], r1['recall50']),
             '%-34s %10s %10s' % ('P@IoU.5 (%)', r2['precision50'], r1['precision50']),
             '%-34s %10s %10s' % ('IoU 中位', r2['iou_median'], r1['iou_median']),
             '%-34s %10s %10s' % ('IoU<0.5 (%)', r2['iou_lt50_pct'], r1['iou_lt50_pct']),
             '%-34s %10s %10s' % ('IoU<0.7 (%)', r2['iou_lt70_pct'], r1['iou_lt70_pct']),
             '%-34s %10s %10s' % ('面积比 中位', r2['area_ratio_median'], r1['area_ratio_median']),
             '%-34s %10s %10s' % ('面积比 P10', r2['area_ratio_p10'], r1['area_ratio_p10']),
             '%-34s %10s %10s' % ('面积比 P90', r2['area_ratio_p90'], r1['area_ratio_p90']),
             '%-34s %10s %10s' % ('中心偏移 %GT对角 中位', r2['center_off_gt_diag_median_pct'], r1['center_off_gt_diag_median_pct']),
             '%-34s %10s %10s' % ('中心偏移 %图对角 中位', r2['center_off_img_diag_median_pct'], r1['center_off_img_diag_median_pct']),
             '', 'GT 面积分桶（桶内 GT 数 / 漏检数）',
             '%-10s %12s %12s' % ('桶', 'batch-2', 'batch-1')]
    for k in ('<2%', '2-5%', '5-10%', '10-20%', '>=20%'):
        lines.append('%-10s %5s / %-6s %5s / %-6s' % (k, r2['gt_by_area_bucket'].get(k, 0),
                                                      r2['miss_by_gt_area_bucket'].get(k, 0),
                                                      r1['gt_by_area_bucket'].get(k, 0),
                                                      r1['miss_by_gt_area_bucket'].get(k, 0)))
    txt = '\n'.join(lines)
    suffix = '' if weights is None else '_' + tag.split('_')[1]
    io.open(os.path.join(OUT, 'eval%s.txt' % suffix), 'w', encoding='utf-8').write(txt)
    io.open(os.path.join(OUT, 'eval%s.json' % suffix), 'w', encoding='utf-8').write(json.dumps(rep, ensure_ascii=False, indent=1))
    print(txt)


if __name__ == '__main__':
    main()
