"""标注批 3 队列：60 张（双人 40 / 三人 20）、交互优先、手小 >=12、R5 去重。

预填用 m2a_det_person_hybrid2（4 类，conf 0.10 / imgsz 1280）。
产物：data/hand_gt_multi/index_batch3.csv、image_list_batch3.txt、images/ 硬链接、dedup_meta_batch3.json
"""
import io, json, os, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
OUT = os.path.join(D, 'hand_gt_multi')
W = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid2', 'weights', 'best.pt')
CONF = 0.10
IMGSZ = 1280
N2, N3 = 40, 20
SMALL_MIN = 12
GIANT, DUP_IOU, HI, LO = 0.50, 0.50, 0.85, 0.85


def load_csv(p):
    import csv
    return list(csv.DictReader(io.open(p, encoding='utf-8-sig')))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def dedupe_person(boxes, img_area):
    """R5：双方 area/图 > 0.50 且 IoU>=0.50 且面积比 < 0.85 ⇒ 删低 conf。"""
    order = sorted(range(len(boxes)), key=lambda i: (-boxes[i]['conf'], -area(boxes[i]['box'])))
    kept, drops = [], []
    for i in order:
        b = boxes[i]['box']
        bad = None
        for k in kept:
            kb = boxes[k]['box']
            if area(b) / img_area <= GIANT or area(kb) / img_area <= GIANT:
                continue
            if iou(b, kb) < DUP_IOU:
                continue
            r = min(area(b), area(kb)) / max(area(b), area(kb))
            if r < LO:
                bad = k
                break
        if bad is None:
            kept.append(i)
        else:
            drops.append({'keep_idx': bad, 'drop_idx': i, 'iou': round(iou(b, boxes[bad]['box']), 4),
                          'containment': round(area(b) / area(boxes[bad]['box']), 4) if area(b) <= area(boxes[bad]['box']) else round(area(boxes[bad]['box']) / area(b), 4),
                          'keep_area': round(area(boxes[bad]['box']), 1), 'drop_area': round(area(b), 1), 'rule': 'r5'})
    return [boxes[i] for i in sorted(kept)], drops


def main():
    inv = load_csv(os.path.join(OUT, 'pool_inventory.csv'))
    labels = set()
    for f in ('hand_gt/index.csv', 'hand_gt_multi/index.csv', 'pose_multi/index.csv'):
        for r in load_csv(os.path.join(D, f)):
            labels.add(os.path.splitext(r.get('crop') or r.get('source') or '')[0])
    hold = set()
    hp = os.path.join(OUT, 'holdout15.txt')
    if os.path.exists(hp):
        for line in io.open(hp, encoding='utf-8'):
            if line.strip():
                hold.add(os.path.splitext(line.split('\t')[0].strip())[0])
    preds = json.load(io.open(os.path.join(OUT, 'preds', 'rank_preds.json'), encoding='utf-8'))

    cand = []
    for r in inv:
        stem = r['stem']
        if r.get('labeled_in_hand_gt') == '1' or stem in labels or stem in hold:
            continue
        try:
            n = int(float(r.get('total') or 0))
        except Exception:
            continue
        if n not in (2, 3):
            continue
        pr = preds.get(r['sha256'])
        if not pr:
            continue
        w, h = float(pr['w']), float(pr['h'])
        diag = (w * w + h * h) ** 0.5
        persons = [x['box'] for x in pr['det'].get('person', []) if x['conf'] >= 0.30]
        hands = [x['box'] for x in pr['det'].get('hand', [])]
        if not persons:
            continue
        # 交互代理
        hand_cross = 0.0
        if hands and len(persons) >= 2:
            for hb in hands:
                cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                inside = [p for p in persons if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]]
                if len(persons) >= 2 and inside:
                    # 手框中心在某人框内、但框跨到另一人框范围
                    others = [p for p in persons if p is not inside[0] and iou(hb, p) > 0]
                    if others:
                        hand_cross = 1.0
                        break
        pair_iou = 0.0
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pair_iou = max(pair_iou, iou(persons[i], persons[j]))
        close = 0.0
        if len(hands) >= 2:
            hc = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in hands]
            for i in range(len(hc)):
                for j in range(i + 1, len(hc)):
                    d = ((hc[i][0] - hc[j][0]) ** 2 + (hc[i][1] - hc[j][1]) ** 2) ** 0.5
                    if d < 0.08 * diag:
                        close = 1.0
        proxy = 0.5 * close + 0.3 * hand_cross + 0.2 * pair_iou
        hand_long = min([max(b[2] - b[0], b[3] - b[1]) for b in hands], default=10**9)
        cand.append({'stem': stem, 'path': r['path'], 'sha': r['sha256'], 'total': n,
                     'proxy': round(proxy, 4), 'hand_long': round(hand_long, 1),
                     'small': hand_long < 110, 'tiny': hand_long < 72,
                     'reason': 'close%.0f cross%.0f ovl%.2f' % (close, hand_cross, pair_iou)})

    three = sorted([c for c in cand if c['total'] == 3], key=lambda x: -x['proxy'])
    two = sorted([c for c in cand if c['total'] == 2], key=lambda x: -x['proxy'])
    sel = three[:N3]
    used = {c['stem'] for c in sel}
    small_pool = [c for c in two if c['small'] and c['stem'] not in used]
    picks = sorted(small_pool, key=lambda x: (-x['tiny'], -x['proxy']))[:SMALL_MIN]
    used |= {c['stem'] for c in picks}
    rest = [c for c in two if c['stem'] not in used]
    picks += rest[:N2 - len(picks)]
    sel += picks
    print('candidates 2p=%d 3p=%d; selected %d (2p=%d 3p=%d) small=%d tiny=%d'
          % (len(two), len(three), len(sel), len(picks), len(three), sum(1 for c in sel if c['small']), sum(1 for c in sel if c['tiny'])))

    from ultralytics import YOLO
    m = YOLO(W)
    files = [c['path'] for c in sel]
    res = m.predict(files, imgsz=IMGSZ, conf=CONF, verbose=False, stream=True, batch=4, device=None)
    rows, drops_all, nsmall = [], [], 0
    for c, r in zip(sel, res):
        stem, ext = os.path.splitext(os.path.basename(c['path']))
        crop = stem + ext
        dst = os.path.join(OUT, 'images', crop)
        if not os.path.exists(dst):
            os.link(c['path'], dst)
        w, h = float(r.orig_shape[1]), float(r.orig_shape[0])
        bycls = {'person': [], 'hand': [], 'foot': []}
        for b in r.boxes:
            nm = r.names[int(b.cls)]
            if nm in bycls:
                bycls[nm].append({'box': [float(v) for v in b.xyxy[0]], 'conf': float(b.conf)})
        kept_p, dr = dedupe_person(bycls['person'], w * h)
        drops_all.append({'image': crop, 'stem': stem, 'target': 'person', 'drops': dr})
        nsmall += 1 if c['small'] else 0
        note = 'batch3;total=%s;proxy=%s;reason=%s;hand_long=%s;pred:person=%d,hand=%d,foot=%d' % (
            c['total'], c['proxy'], c['reason'], c['hand_long'], len(bycls['person']), len(bycls['hand']), len(bycls['foot']))
        for target, boxes in (('hand', bycls['hand']), ('foot', bycls['foot']), ('person', kept_p)):
            if not boxes:
                continue
            prop = ';'.join('%.1f,%.1f,%.1f,%.1f' % tuple(b['box']) for b in boxes)
            confs = ';'.join('%.3f' % b['conf'] for b in boxes)
            rows.append({'idx': str(len(rows) + 1), 'crop': crop, 'source': stem, 'head': '0',
                         'target': target, 'prop': prop, 'prop_src': 'hybrid2', 'prop_conf': confs,
                         'wd14': str(c['total']), 'verdict': '', 'boxes': '', 'edited': '', 'note': note})
    for i, r in enumerate(rows, 1):
        r['idx'] = str(i)
    FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf', 'wd14', 'verdict', 'boxes', 'edited', 'note']
    import csv
    with io.open(os.path.join(OUT, 'index_batch3.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with io.open(os.path.join(OUT, 'image_list_batch3.txt'), 'w', encoding='utf-8') as fh:
        fh.write('stem\tpath\ttotal\tproxy\thand_long\tsmall\n')
        for c in sel:
            fh.write('%s\t%s\t%d\t%s\t%s\t%d\n' % (c['stem'], c['path'], c['total'], c['proxy'], c['hand_long'], int(c['small'])))
    nd = sum(len(d['drops']) for d in drops_all)
    json.dump({'dedup_rule': 'r5', 'dropped_boxes': nd, 'images_with_drops': sum(1 for d in drops_all if d['drops']),
               'images': len(sel), 'rows': len(rows), 'small': nsmall, 'weights': W, 'conf': CONF, 'imgsz': IMGSZ},
              io.open(os.path.join(OUT, 'dedup_meta_batch3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('rows=%d images=%d small=%d dedup=r5 删 %d 框（尺度错位对）；尺寸相近的重复框请人工删'
          % (len(rows), len(sel), nsmall, nd))
    return 0


if __name__ == '__main__':
    sys.exit(main())
