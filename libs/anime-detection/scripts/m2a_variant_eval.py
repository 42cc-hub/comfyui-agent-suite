"""A 段：locate.py 三变体端到端评测（60 图多人集 + 40 图 upright 集）。

变体1 = 现状（fix_v2 + limb rot_semi + hand-imgsz 1536）
变体2 = hybrid2 单模型（显式 --det-weights ⇒ limb 自动关闭）+ hand-imgsz 1536
变体3 = hybrid2 单模型 + hand-imgsz 1280
产物 team/reports/m2a-verifier/variants/{ratio.txt,ratio.json,preds_*}
"""
import io, json, os, subprocess, sys, time

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
REP = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'variants')
LOC = os.path.join(SUITE, 'scripts', 'locate.py')
PY = sys.executable
W = {
    'fix_v2': os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'),
    'hybrid2': os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid2', 'weights', 'best.pt'),
}
SETS = {
    'multi60': {'dir': os.path.join(D, 'hand_gt_multi', 'images'), 'gt': os.path.join(D, 'hand_gt_multi', 'index.csv')},
    'upright40': {'dir': os.path.join(D, 'hand_gt', 'images'), 'gt': os.path.join(D, 'hand_gt', 'index.csv')},
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def metric(gt, pred):
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            v = iou(g, p)
            if v >= 0.5:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    ug, up = set(), set()
    ious = []
    for v, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        ious.append(v)
    tp = len(ious)
    return {'gt': len(gt), 'pred': len(pred), 'tp': tp, 'fp': len(pred) - tp, 'fn': len(gt) - tp,
            'R': tp / len(gt) if gt else 0.0, 'P': tp / (tp + len(pred) - tp) if pred else 0.0,
            'medIoU': sorted(ious)[len(ious) // 2] if ious else 0.0}


def load_gt(path):
    import csv
    out = {}
    for r in csv.DictReader(io.open(path, encoding='utf-8-sig')):
        if (r.get('verdict') or '') not in ('draw', 'pass'):
            continue
        boxes = []
        for seg in (r.get('boxes') or '').split(';'):
            seg = seg.strip()
            if not seg:
                continue
            v = [float(x) for x in seg.split(',')]
            if len(v) == 4:
                boxes.append(tuple(v))
        if not boxes:
            continue
        crop = r['crop']
        out.setdefault(crop, {'person': [], 'hand': [], 'foot': []})
        out[crop][r['target']] = boxes
    return out


def run_variant(tag, extra, src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, LOC, '--src', src_dir, '--out', out_dir, '--conf', '0.30', '--imgsz', '1280',
           '--no-vitpose'] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    dt = time.time() - t0
    if r.returncode != 0:
        print('[%s] FAILED rc=%s %s' % (tag, r.returncode, (r.stderr or '')[-400:]))
        return None, dt
    return out_dir, dt


def score(out_dir, gts):
    got = {'person': [], 'hand': [], 'foot': []}
    n = 0
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith('.json'):
            continue
        rec = json.load(io.open(os.path.join(out_dir, f), encoding='utf-8'))
        crop = rec.get('image') or f[:-5]
        crop = os.path.basename(crop)
        n += 1
        g = gts.get(crop)
        boxes = rec.get('boxes') or []
        boxes_all = rec.get('boxes_all') or []
        for b in boxes_all:
            cls = b.get('cls')
            if cls in got:
                got[cls].append((crop, [float(v) for v in b.get('box')]))
    res = {}
    for cls in ('person', 'hand', 'foot'):
        gt_boxes, pred_boxes = [], []
        for crop, g in gts.items():
            gt_boxes.extend(g.get(cls, []))
        for crop, b in got[cls]:
            pred_boxes.append(b)
        res[cls] = metric(gt_boxes, pred_boxes)
        res[cls]['img_with_gt'] = sum(1 for g in gts.values() if g.get(cls))
    res['_images'] = n
    res['_pred_boxes'] = {c: len(got[c]) for c in got}
    return res


def main():
    os.makedirs(REP, exist_ok=True)
    variants = [
        ('v1_now', []),
        ('v2_hybrid2_1536', ['--det-weights', W['hybrid2'], '--hand-imgsz', '1536']),
        ('v3_hybrid2_1280', ['--det-weights', W['hybrid2'], '--hand-imgsz', '1280']),
    ]
    res = {}
    txt = []

    def say(s=''):
        txt.append(s)
        print(s, flush=True)

    for setname, info in SETS.items():
        gts = load_gt(info['gt'])
        say('== %s  images=%d  GT(person/hand/foot) = %d/%d/%d' % (
            setname, len(gts),
            sum(len(g.get('person', [])) for g in gts.values()),
            sum(len(g.get('hand', [])) for g in gts.values()),
            sum(len(g.get('foot', [])) for g in gts.values())))
        for tag, extra in variants:
            out_dir = os.path.join(REP, 'preds_%s_%s' % (setname, tag))
            _, dt = run_variant(setname + '/' + tag, extra, info['dir'], out_dir)
            sc = score(out_dir, gts)
            n = max(1, sc['_images'])
            sc['sec_per_img'] = dt / n
            sc['wall_s'] = dt
            res['%s/%s' % (setname, tag)] = sc
            say('  %-18s %5.1fs  %.3f s/img  框(person/hand/foot)=%d/%d/%d' % (
                tag, dt, dt / n, sc['_pred_boxes']['person'], sc['_pred_boxes']['hand'], sc['_pred_boxes']['foot']))
            for cls in ('person', 'hand', 'foot'):
                m = sc[cls]
                say('     %-6s R %.3f P %.3f TP %d/%d medIoU %.3f' % (cls, m['R'], m['P'], m['tp'], m['gt'], m['medIoU']))
    io.open(os.path.join(REP, 'ratio.txt'), 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
    json.dump(res, io.open(os.path.join(REP, 'ratio.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
