"""person hybrid 验收三条：① batch-2 person ② 产线几何配对（person/hand/foot） ③ holdout15 hand/foot。

用法: python m2a_person_hybrid_eval.py --weights <best.pt> [--device cuda] [--out-name hybrid]
产物: team/reports/m2a-verifier/person_hybrid/eval_<name>.txt|.json
"""
import argparse, csv, io, json, os, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
REP = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'person_hybrid')
W_FIX = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
GT2 = os.path.join(D, 'hand_gt_multi', 'index_person_topup.csv')
GT1 = os.path.join(D, 'hand_gt_multi', 'index.csv')
IMG1 = os.path.join(D, 'hand_gt_multi', 'images')
HOLD_IMG = os.path.join(D, 'ds_det_rot', 'holdout', 'images')
HOLD_LBL = os.path.join(D, 'ds_det_rot', 'holdout', 'labels')
CLS = {'person': 0, 'hand': 1, 'foot': 2, 'head': 3}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy(gt, pred, thr=0.5):
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            v = iou(g, p)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    ug, up, m = set(), set(), []
    for v, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        m.append((v, i, j))
    return m


def parse(s):
    out = []
    for seg in (s or '').split(';'):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(x) for x in seg.split(',')]
        if len(v) == 4:
            out.append(tuple(v))
    return out


def load_gt(path, target, img_dir):
    rows = []
    for r in csv.DictReader(io.open(path, encoding='utf-8-sig')):
        # draw 与 pass 都带框（pass = 整张采纳预填），neg 才是无目标
        if (r.get('verdict') or '') not in ('draw', 'pass'):
            continue
        if (r.get('target') or '') != target:
            continue
        boxes = parse(r.get('boxes'))
        if not boxes:
            continue
        p = os.path.join(img_dir, r['crop'])
        if not os.path.exists(p):
            continue
        rows.append({'img': p, 'crop': r['crop'], 'boxes': boxes})
    return rows


def metric(gt, pred):
    m = greedy(gt, pred)
    tp = len(m)
    ious = [v for v, _, _ in m]
    fp = len(pred) - tp
    fn = len(gt) - tp
    rec = tp / len(gt) if gt else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    ratios = []
    for v, i, j in m:
        ga = (gt[i][2] - gt[i][0]) * (gt[i][3] - gt[i][1])
        pa = (pred[j][2] - pred[j][0]) * (pred[j][3] - pred[j][1])
        if ga > 0:
            ratios.append(pa / ga)
    ratios.sort()
    med_r = ratios[len(ratios) // 2] if ratios else 0.0
    ious_sorted = sorted(ious)
    med_iou = ious_sorted[len(ious_sorted) // 2] if ious_sorted else 0.0
    return {'gt': len(gt), 'pred': len(pred), 'tp': tp, 'fp': fp, 'fn': fn,
            'R': rec, 'P': pre, 'medIoU': med_iou, 'area_ratio_med': med_r,
            'iou_lt05': sum(1 for v in ious if v < 0.5) / len(ious) if ious else 0.0}


def predict(model, img, imgsz, conf, cls_filter=None):
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=None)[0]
    names = res.names
    out = []
    for b in res.boxes:
        cname = names[int(b.cls)]
        if cls_filter and cname not in cls_filter:
            continue
        xy = [float(v) for v in b.xyxy[0]]
        out.append({'cls': cname, 'box': xy, 'conf': float(b.conf)})
    return out


def run_set(model, rows, imgsz, conf, target, img_dir=None):
    gts, preds = [], []
    per_img = []
    for r in rows:
        img = r['img']
        pr = [p['box'] for p in predict(model, img, imgsz, conf, {target})]
        gts.extend(r['boxes'])
        preds.extend(pr)
        mi = greedy(r['boxes'], pr)
        per_img.append({'crop': r['crop'], 'gt': len(r['boxes']), 'pred': len(pr), 'tp': len(mi)})
    return metric(gts, preds), per_img


def holdout_rows():
    rows = []
    for f in sorted(os.listdir(HOLD_IMG)):
        if not f.endswith('__a0.jpg'):
            continue
        stem = f[:-len('__a0.jpg')]
        lbl = os.path.join(HOLD_LBL, stem + '__a0.txt')
        if not os.path.exists(lbl):
            continue
        import cv2
        im = cv2.imread(os.path.join(HOLD_IMG, f))
        h, w = im.shape[:2]
        by = {'hand': [], 'foot': []}
        for line in io.open(lbl, encoding='utf-8'):
            v = line.split()
            if len(v) < 5:
                continue
            c = int(float(v[0]))
            cx, cy, bw, bh = [float(x) for x in v[1:5]]
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            if c == 1:
                by['hand'].append(box)
            elif c == 2:
                by['foot'].append(box)
        rows.append({'img': os.path.join(HOLD_IMG, f), 'crop': f, 'hand': by['hand'], 'foot': by['foot']})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--name', default='hybrid')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--prod-imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.30)
    a = ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(REP, exist_ok=True)
    out = {'weights': a.weights, 'conf': a.conf}

    m_new = YOLO(a.weights)
    m_fix = YOLO(W_FIX)

    # ① batch-2 person
    rows2 = load_gt(GT2, 'person', IMG1)
    m2, _ = run_set(m_new, rows2, a.imgsz, a.conf, 'person')
    m2b, _ = run_set(m_fix, rows2, a.imgsz, a.conf, 'person')
    out['batch2_person'] = {'new': m2, 'fix_v2': m2b, 'images': len(rows2)}

    # ② 产线几何配对：person/hand/foot on GT1
    prod = {}
    for cls in ('person', 'hand', 'foot'):
        rows1 = load_gt(GT1, cls, IMG1)
        n, per_n = run_set(m_new, rows1, a.prod_imgsz, a.conf, cls)
        b, per_b = run_set(m_fix, rows1, a.prod_imgsz, a.conf, cls)
        prod[cls] = {'new': n, 'fix_v2': b, 'images': len(rows1),
                     'per_image': {x['crop']: [x['tp'], x['pred'], x['gt']] for x in per_n},
                     'per_image_fix': {x['crop']: [x['tp'], x['pred'], x['gt']] for x in per_b}}
    out['prod'] = prod

    # ③ holdout15 hand/foot
    hrows = holdout_rows()
    hold = {}
    for cls in ('hand', 'foot'):
        gts, pn, pb = [], [], []
        for r in hrows:
            gts.extend(r[cls])
            pn.extend([p['box'] for p in predict(m_new, r['img'], a.imgsz, a.conf, {cls})])
            pb.extend([p['box'] for p in predict(m_fix, r['img'], a.imgsz, a.conf, {cls})])
        hold[cls] = {'new': metric(gts, pn), 'fix_v2': metric(gts, pb)}
    out['holdout15'] = hold

    txt = []
    def line(s=''):
        txt.append(s)
        print(s)
    line('person hybrid 验收  weights=%s conf=%.2f' % (a.weights, a.conf))
    line('=' * 78)
    line('① batch-2 GT（%d 图）person：' % len(rows2))
    for k, v in (('new', m2), ('fix_v2', m2b)):
        line('   %-7s GT %d pred %d TP %d  R %.3f P %.3f medIoU %.3f 面积比中位 %.3f IoU<.5 %.1f%%'
             % (k, v['gt'], v['pred'], v['tp'], v['R'], v['P'], v['medIoU'], v['area_ratio_med'], v['iou_lt05'] * 100))
    line('② 产线几何（imgsz %d）' % a.prod_imgsz)
    for cls, v in prod.items():
        line('   %-6s new: R %.3f P %.3f TP %d/%d medIoU %.3f | fix_v2: R %.3f P %.3f TP %d/%d medIoU %.3f'
             % (cls, v['new']['R'], v['new']['P'], v['new']['tp'], v['new']['gt'], v['new']['medIoU'],
                v['fix_v2']['R'], v['fix_v2']['P'], v['fix_v2']['tp'], v['fix_v2']['gt'], v['fix_v2']['medIoU']))
    line('③ holdout15（imgsz %d）' % a.imgsz)
    for cls, v in hold.items():
        line('   %-4s new: R %.3f P %.3f (%d/%d) | fix_v2: R %.3f P %.3f (%d/%d)'
             % (cls, v['new']['R'], v['new']['P'], v['new']['tp'], v['new']['gt'],
                v['fix_v2']['R'], v['fix_v2']['P'], v['fix_v2']['tp'], v['fix_v2']['gt']))
    # ③ 逐图配对差值（三类）
    line('② 逐图配对差值（TP 变化 >0 的图数 / TP 变化总和 / FP 变化总和）')
    for cls, v in prod.items():
        dn = dt = df = 0
        for crop, (tp_n, pred_n, gt) in v['per_image'].items():
            tp_b, pred_b, _ = v['per_image_fix'][crop]
            dt += tp_n - tp_b
            dn += 1 if tp_n != tp_b else 0
            df += pred_n - pred_b
        line('   %-6s TP 变动的图 %d 张，TP 合计 %+d，预测框合计 %+d' % (cls, dn, dt, df))

    io.open(os.path.join(REP, 'eval_%s.txt' % a.name), 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
    json.dump(out, io.open(os.path.join(REP, 'eval_%s.json' % a.name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
