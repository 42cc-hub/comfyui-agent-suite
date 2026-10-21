"""批 3 补货：从 candidates 池补到 60 张（保留已选的 38 张，只给新增图做预填）。

产物：重写 data/hand_gt_multi/index_batch3.csv、image_list_batch3.txt、dedup_meta_batch3.json
"""
import csv, io, json, os, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
OUT = os.path.join(D, 'hand_gt_multi')
W = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid2', 'weights', 'best.pt')
TARGET = 60
CONF, IMGSZ = 0.10, 1280
RANK_CONF, RANK_IMGSZ = 0.05, 960
GIANT, DUP_IOU, LO = 0.50, 0.50, 0.85


def load_csv(p):
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
            if min(area(b), area(kb)) / max(area(b), area(kb)) < LO:
                bad = k
                break
        if bad is None:
            kept.append(i)
        else:
            drops.append({'keep_idx': bad, 'drop_idx': i, 'iou': round(iou(b, boxes[bad]['box']), 4),
                          'keep_area': round(area(boxes[bad]['box']), 1), 'drop_area': round(area(b), 1), 'rule': 'r5'})
    return [boxes[i] for i in sorted(kept)], drops


def main():
    import csv as _csv
    labeled = set()
    for f in ('hand_gt/index.csv', 'hand_gt_multi/index.csv', 'pose_multi/index.csv'):
        for r in load_csv(os.path.join(D, f)):
            labeled.add(os.path.splitext(r.get('crop') or r.get('source') or '')[0])
    hold = set()
    hp = os.path.join(OUT, 'holdout15.txt')
    if os.path.exists(hp):
        for line in io.open(hp, encoding='utf-8'):
            if line.strip():
                hold.add(os.path.splitext(line.split('\t')[0].strip())[0])

    rows = load_csv(os.path.join(OUT, 'index_batch3.csv'))
    have = []
    seen = set()
    for r in rows:
        if r['source'] not in seen:
            seen.add(r['source'])
            have.append(r['source'])
    need = TARGET - len(have)
    print('existing images', len(have), 'need', need)

    tags = load_csv(os.path.join(OUT, 'tags_candidates.csv'))
    cand = []
    for r in tags:
        stem = os.path.splitext(os.path.basename(r['path']))[0]
        if stem in labeled or stem in hold or stem in seen:
            continue
        try:
            total = int(float(r.get('total') or 0))
        except Exception:
            continue
        if total not in (2, 3) or not os.path.exists(r['path']):
            continue
        cand.append({'stem': stem, 'path': r['path'], 'total': total})
    print('candidates after filter', len(cand))

    from ultralytics import YOLO
    m = YOLO(W)
    if cand:
        res = m.predict([c['path'] for c in cand], imgsz=RANK_IMGSZ, conf=RANK_CONF, verbose=False, stream=True, batch=8, device='cpu')
        ranked = []
        for c, r in zip(cand, res):
            persons, hands = [], []
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                if nm == 'person' and float(b.conf) >= 0.30:
                    persons.append([float(v) for v in b.xyxy[0]])
                elif nm == 'hand':
                    hands.append([float(v) for v in b.xyxy[0]])
            if not persons:
                continue
            w, h = float(r.orig_shape[1]), float(r.orig_shape[0])
            diag = (w * w + h * h) ** 0.5
            pair = max([iou(persons[i], persons[j]) for i in range(len(persons)) for j in range(i + 1, len(persons))] or [0.0])
            close = 0.0
            if len(hands) >= 2:
                hc = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in hands]
                for i in range(len(hc)):
                    for j in range(i + 1, len(hc)):
                        d = ((hc[i][0] - hc[j][0]) ** 2 + (hc[i][1] - hc[j][1]) ** 2) ** 0.5
                        if d < 0.08 * diag:
                            close = 1.0
            hand_long = min([max(b[2] - b[0], b[3] - b[1]) for b in hands], default=10**9)
            c.update({'proxy': round(0.5 * close + 0.2 * pair, 4), 'hand_long': round(hand_long, 1),
                      'small': hand_long < 110, 'tiny': hand_long < 72})
            ranked.append(c)
        three = sorted([c for c in ranked if c['total'] == 3], key=lambda x: -x['proxy'])
        small = sorted([c for c in ranked if c['small']], key=lambda x: (-x['tiny'], -x['proxy']))
        rest = sorted(ranked, key=lambda x: -x['proxy'])
        picks, used = [], set()

        def take(lst, n):
            out = []
            for c in lst:
                if len(out) >= n:
                    break
                if c['stem'] in used:
                    continue
                used.add(c['stem'])
                out.append(c)
            return out
        picks += take(three, max(0, need - 8))
        picks += take(small, need - len(picks))
        picks += take(rest, need - len(picks))
        print('candidates 2p/3p', sum(1 for c in ranked if c['total'] == 2), sum(1 for c in ranked if c['total'] == 3),
              'picked', len(picks), '3p', sum(1 for c in picks if c['total'] == 3),
              'small', sum(1 for c in picks if c['small']))
    else:
        picks = []

    files = [c['path'] for c in picks]
    new_rows, drops_all = [], []
    if files:
        res = m.predict(files, imgsz=IMGSZ, conf=CONF, verbose=False, stream=True, batch=8, device='cpu')
        for c, r in zip(picks, res):
            stem, ext = os.path.splitext(os.path.basename(c['path']))
            crop = stem + ext
            dst = os.path.join(OUT, 'images', crop)
            if not os.path.exists(dst):
                os.link(c['path'], dst)
            w, h = float(r.orig_shape[1]), float(r.orig_shape[0])
            by = {'person': [], 'hand': [], 'foot': []}
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                if nm in by:
                    by[nm].append({'box': [float(v) for v in b.xyxy[0]], 'conf': float(b.conf)})
            kept_p, dr = dedupe_person(by['person'], w * h)
            drops_all.append({'image': crop, 'drops': dr})
            note = 'batch3;cand;total=%s;proxy=%s;hand_long=%s;pred:person=%d,hand=%d,foot=%d' % (
                c['total'], c['proxy'], c['hand_long'], len(by['person']), len(by['hand']), len(by['foot']))
            for target, boxes in (('hand', by['hand']), ('foot', by['foot']), ('person', kept_p)):
                if not boxes:
                    continue
                new_rows.append({'crop': crop, 'source': stem, 'head': '0', 'target': target,
                                 'prop': ';'.join('%.1f,%.1f,%.1f,%.1f' % tuple(b['box']) for b in boxes),
                                 'prop_src': 'hybrid2', 'prop_conf': ';'.join('%.3f' % b['conf'] for b in boxes),
                                 'wd14': str(c['total']), 'verdict': '', 'boxes': '', 'edited': '', 'note': note})

    all_rows = []
    for r in rows:
        all_rows.append({k: r.get(k, '') for k in r})
    for r in new_rows:
        all_rows.append(r)
    for i, r in enumerate(all_rows, 1):
        r['idx'] = str(i)
    FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf', 'wd14', 'verdict', 'boxes', 'edited', 'note']
    with io.open(os.path.join(OUT, 'index_batch3.csv'), 'w', encoding='utf-8', newline='') as fh:
        wr = _csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k, '') for k in FIELDS})

    meta = json.load(io.open(os.path.join(OUT, 'dedup_meta_batch3.json'), encoding='utf-8'))
    old_drops = meta.get('dropped_boxes', 0)
    meta.update({'dedup_rule': 'r5', 'dropped_boxes': old_drops + sum(len(d['drops']) for d in drops_all),
                 'images': len(have) + len(picks), 'rows': len(all_rows),
                 'small': None, 'conf': CONF, 'imgsz': IMGSZ, 'weights': W})
    json.dump(meta, io.open(os.path.join(OUT, 'dedup_meta_batch3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # image_list：重写（旧 38 + 新补）
    stem_info = {}
    for r in rows:
        stem_info[r['source']] = {'total': r.get('wd14', ''), 'proxy': '', 'hand_long': '', 'small': ''}
    for c in picks:
        stem_info[c['stem']] = {'total': c['total'], 'proxy': c['proxy'], 'hand_long': c['hand_long'], 'small': int(c['small'])}
    with io.open(os.path.join(OUT, 'image_list_batch3.txt'), 'w', encoding='utf-8') as fh:
        fh.write('stem\tpath\ttotal\tproxy\thand_long\tsmall\n')
        for stem, info in stem_info.items():
            p = ''
            for r in rows:
                if r['source'] == stem:
                    p = r['crop']
                    break
            if not p:
                for c in picks:
                    if c['stem'] == stem:
                        p = c['path']
            fh.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (stem, p, info['total'], info['proxy'], info['hand_long'], info['small']))

    three_n = sum(1 for s, i in stem_info.items() if str(i['total']) == '3')
    small_n = sum(1 for s, i in stem_info.items() if i['small'] == 1)
    print('TOTAL images %d | three %d | small %d | rows %d | dedup dropped %d' % (
        len(stem_info), three_n, small_n, len(all_rows), meta['dropped_boxes']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
