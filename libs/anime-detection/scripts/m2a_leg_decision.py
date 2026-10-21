"""人腿开关（--det-person-weights）决策打分：A=现状、B=带人腿，两套 GT。

A 直接复用同配置（conf .30 / imgsz 1280 / --no-vitpose）的既有产物
（`variants/preds_multi60_v1_now`、`variants/preds_upright40_v1_now`，已核对与改动前逐字段一致）；
B 现跑。指标：person R/P/medIoU/面积比中位、过分割率（多 1 框 / 多 >=2 框）、hand/foot 是否变化、耗时。
"""
import csv, io, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2a_variant_eval as V

SUITE = V.SUITE
D = V.D
REP = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier')
OUT = os.path.join(REP, 'person_leg')
PERSON_W = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid', 'weights', 'best.pt')
SETS = {
    'multi60': {'dir': os.path.join(D, 'hand_gt_multi', 'images'),
                'gt': os.path.join(D, 'hand_gt_multi', 'index.csv'),
                'A': os.path.join(REP, 'variants', 'preds_multi60_v1_now')},
    'upright40': {'dir': os.path.join(D, 'hand_gt', 'images'),
                  'gt': os.path.join(D, 'hand_gt', 'index.csv'),
                  'A': os.path.join(REP, 'variants', 'preds_upright40_v1_now')},
}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_recs(d):
    recs = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        crop = os.path.basename(rec.get('image') or f[:-5])
        recs[crop] = rec
    return recs


def boxes_of(rec, cls):
    return [[float(v) for v in b['box']] for b in (rec.get('boxes_all') or []) if b.get('cls') == cls]


def area_ratio(recs, gts):
    rs = []
    for crop, g in gts.items():
        if not g.get('person') or crop not in recs:
            continue
        preds = boxes_of(recs[crop], 'person')
        for gt in g['person']:
            ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
            if ga <= 0:
                continue
            best, bi = 0.0, None
            for b in preds:
                v = iou(gt, b)
                if v > best:
                    best, bi = v, b
            if bi is not None and best >= 0.5:
                pa = (bi[2] - bi[0]) * (bi[3] - bi[1])
                rs.append(pa / ga)
    return rs


def overseg(recs, gts):
    tot = multi1 = multi2 = 0
    rows = []
    for crop, g in gts.items():
        n_gt = len(g.get('person') or [])
        if n_gt == 0 or crop not in recs:
            continue
        n_pr = len(boxes_of(recs[crop], 'person'))
        tot += 1
        d = n_pr - n_gt
        if d == 1:
            multi1 += 1
        elif d >= 2:
            multi2 += 1
        rows.append((crop, n_gt, n_pr, d))
    return tot, multi1, multi2, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    say = []
    for setname, info in SETS.items():
        gts = V.load_gt(info['gt'])
        gtp = sum(len(g.get('person') or []) for g in gts.values())
        gth = sum(len(g.get('hand') or []) for g in gts.values())
        gtf = sum(len(g.get('foot') or []) for g in gts.values())
        t0 = time.time()
        bdir, dt = V.run_variant('B_' + setname, ['--det-person-weights', PERSON_W], info['dir'],
                                 os.path.join(OUT, 'preds_%s_B' % setname))
        say.append('== %s images=%d GT person/hand/foot=%d/%d/%d  B 用时 %.1fs (%.2f s/img)' % (
            setname, len(gts), gtp, gth, gtf, dt, dt / max(1, len(gts))))
        if bdir is None:
            say.append('   B FAILED')
            continue
        recA, recB = read_recs(info['A']), read_recs(bdir)
        scA, scB = V.score(info['A'], gts), V.score(bdir, gts)
        for cls in ('person', 'hand', 'foot'):
            a, b = scA[cls], scB[cls]
            say.append('   %-6s A R %.3f P %.3f medIoU %.3f | B R %.3f P %.3f medIoU %.3f | predA %d predB %d' % (
                cls, a.get('R', 0), a.get('P', 0), a.get('medIoU', 0),
                b.get('R', 0), b.get('P', 0), b.get('medIoU', 0),
                scA['_pred_boxes'][cls], scB['_pred_boxes'][cls]))
        ra, rb = area_ratio(recA, gts), area_ratio(recB, gts)
        say.append('   person 面积比中位 A %.3f (n=%d) | B %.3f (n=%d)' % (
            statistics.median(ra) if ra else 0, len(ra), statistics.median(rb) if rb else 0, len(rb)))
        if gtp:
            tot, m1, m2, rows = overseg(recA, gts)
            tB, m1B, m2B, _ = overseg(recB, gts)
            say.append('   过分割 A: %d/%d 图 (%.1f%%) 多1框 %d 多>=2框 %d | B: %d/%d (%.1f%%) 多1框 %d 多>=2框 %d' % (
                m1 + m2, tot, 100.0 * (m1 + m2) / tot, m1, m2,
                m1B + m2B, tB, 100.0 * (m1B + m2B) / tB, m1B, m2B))
            bad = [r for r in rows if r[3] != 0][:8]
            say.append('   A 框数不符示例: ' + '; '.join('%s gt%d pred%d' % (c[:28], g, p) for c, g, p, d in bad))
        diffs = []
        for crop in gts:
            if crop not in recA or crop not in recB:
                continue
            for cls in ('hand', 'foot'):
                if boxes_of(recA[crop], cls) != boxes_of(recB[crop], cls):
                    diffs.append((crop, cls))
        say.append('   hand/foot 变化的 (图,类) 对数: %d %s' % (len(diffs), diffs[:5]))
    txt = '\n'.join(say)
    io.open(os.path.join(OUT, 'leg_decision.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
