"""person-only 鍥捐ˉ hand/foot/head 浼爣绛撅紙rot_semi, conf>=0.50, imgsz 960锛夈€?
- person 鐢ㄥ凡鏈夌湡鍊兼锛堟爣绛鹃噷鐨?class 0锛夛紝涓嶉噰鐢ㄦā鍨?person 妗?- 鍙彇 hand(1)/foot(2)/head(3) 浼爣锛涗竴寮犲浘 hand/foot 涓€涓兘娌℃湁 => 鏁村紶涓㈠純
浜х墿锛歞ata/ds_person_pl/{images,labels}/train + pl_stats.json + pl_sample12.json
"""
import io, json, os, shutil, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
SRC_IMG = os.path.join(D, 'ds_person', 'images', 'train')
SRC_LBL = os.path.join(D, 'ds_person', 'labels', 'train')
OUT = os.path.join(D, 'ds_person_pl')
W = os.path.join(SUITE, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt')
CONF = 0.50
IMGSZ = 960


def main():
    from ultralytics import YOLO
    import random
    m = YOLO(W)
    files = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith(('.jpg', '.png')))
    paths = [os.path.join(SRC_IMG, f) for f in files]
    print('person-only files', len(paths))
    kept, dropped = [], []
    stats = {'files': len(files), 'kept': 0, 'dropped': 0, 'pseudo': {'hand': 0, 'foot': 0, 'head': 0}}
    os.makedirs(os.path.join(OUT, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'labels', 'train'), exist_ok=True)
    res = m.predict(paths, imgsz=IMGSZ, conf=CONF, verbose=False, stream=True, device='cpu', batch=8)
    for f, r in zip(files, res):
        stem = os.path.splitext(f)[0]
        person_lines, extra, cnt = [], [], {'hand': 0, 'foot': 0, 'head': 0}
        lp = os.path.join(SRC_LBL, stem + '.txt')
        if os.path.exists(lp):
            for line in io.open(lp, encoding='utf-8'):
                v = line.split()
                if v and int(float(v[0])) == 0:
                    person_lines.append('0 ' + ' '.join(v[1:5]))
        h, w = r.orig_shape[:2]
        for b in r.boxes:
            c = int(b.cls)
            if c not in (1, 2, 3):
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            extra.append('%d %.6f %.6f %.6f %.6f' % (c, cx, cy, bw, bh))
            cnt['hand' if c == 1 else ('foot' if c == 2 else 'head')] += 1
        if cnt['hand'] + cnt['foot'] == 0:
            dropped.append(f)
            continue
        kept.append(f)
        stats['pseudo']['hand'] += cnt['hand']
        stats['pseudo']['foot'] += cnt['foot']
        stats['pseudo']['head'] += cnt['head']
        dst_img = os.path.join(OUT, 'images', 'train', f)
        if not os.path.exists(dst_img):
            os.link(os.path.join(SRC_IMG, f), dst_img)
        io.open(os.path.join(OUT, 'labels', 'train', stem + '.txt'), 'w', encoding='utf-8').write(
            '\n'.join(person_lines + extra) + '\n')
    stats['kept'] = len(kept)
    stats['dropped'] = len(dropped)
    stats['dropped_files'] = dropped
    random.seed(7)
    stats['sample12'] = sorted(random.sample(kept, min(12, len(kept))))
    json.dump(stats, io.open(os.path.join(OUT, 'pl_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in ('files', 'kept', 'dropped', 'pseudo')}, ensure_ascii=False))
    print('sample12', stats['sample12'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
