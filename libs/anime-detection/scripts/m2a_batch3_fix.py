"""补齐批 3 队列中「入选但没有任何预填行」的图（重建后的一致性修复）。"""
import csv, io, json, os, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
OUT = os.path.join(D, 'hand_gt_multi')
W = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid2', 'weights', 'best.pt')
CONF, IMGSZ = 0.10, 1280
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf', 'wd14', 'verdict', 'boxes', 'edited', 'note']


def main():
    rows = list(csv.DictReader(io.open(os.path.join(OUT, 'index_batch3.csv'), encoding='utf-8-sig')))
    covered = {r['source'] for r in rows}
    lines = [l.rstrip('\n') for l in io.open(os.path.join(OUT, 'image_list_batch3.txt'), encoding='utf-8') if l.strip()][1:]
    entries = []
    for l in lines:
        p = l.split('\t')
        stem, path = p[0], p[1]
        if not os.path.isabs(path):
            path = os.path.join(OUT, 'images', path)
        entries.append({'stem': stem, 'path': path, 'total': p[2] if len(p) > 2 else '', 'proxy': p[3] if len(p) > 3 else '',
                        'hand_long': p[4] if len(p) > 4 else '', 'small': p[5] if len(p) > 5 else ''})
    missing = [e for e in entries if e['stem'] not in covered]
    print('images', len(entries), 'with rows', len(covered), 'missing', len(missing))
    if missing:
        from ultralytics import YOLO
        m = YOLO(W)
        res = m.predict([e['path'] for e in missing], imgsz=IMGSZ, conf=CONF, verbose=False, stream=True, batch=8, device='cpu')
        for e, r in zip(missing, res):
            crop = os.path.basename(e['path'])
            w, h = float(r.orig_shape[1]), float(r.orig_shape[0])
            by = {'person': [], 'hand': [], 'foot': [], 'head': []}
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                if nm in by:
                    by[nm].append({'box': [float(v) for v in b.xyxy[0]], 'conf': float(b.conf)})
            note = 'batch3;filled;total=%s;pred:person=%d,hand=%d,foot=%d' % (
                e['total'], len(by['person']), len(by['hand']), len(by['foot']))
            for target in ('hand', 'foot', 'person'):
                boxes = by[target]
                if not boxes:
                    continue
                rows.append({'idx': '', 'crop': crop, 'source': e['stem'], 'head': '0', 'target': target,
                             'prop': ';'.join('%.1f,%.1f,%.1f,%.1f' % tuple(b['box']) for b in boxes),
                             'prop_src': 'hybrid2', 'prop_conf': ';'.join('%.3f' % b['conf'] for b in boxes),
                             'wd14': str(e['total']), 'verdict': '', 'boxes': '', 'edited': '', 'note': note})
    for i, r in enumerate(rows, 1):
        r['idx'] = str(i)
    with io.open(os.path.join(OUT, 'index_batch3.csv'), 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in FIELDS})
    with io.open(os.path.join(OUT, 'image_list_batch3.txt'), 'w', encoding='utf-8') as fh:
        fh.write('stem\tpath\ttotal\tproxy\thand_long\tsmall\n')
        for e in entries:
            fh.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (e['stem'], e['path'], e['total'], e['proxy'], e['hand_long'], e['small']))
    images = len({r['source'] for r in rows})
    three = sum(1 for e in entries if str(e['total']) == '3')
    small = sum(1 for e in entries if str(e['small']) == '1')
    print('FINAL images %d (rows image %d) | three %d | small %d | rows %d' % (images, len(entries), three, small, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
