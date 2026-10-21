"""伪标质量抽查图：12 张 person-only 图的伪标框（hand 黄 / foot 蓝 / head 紫 / person 真值 绿）。"""
import io, json, os, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
PL = os.path.join(D, 'ds_person_pl')
REP = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'person_hybrid2')
COLORS = {0: (0, 220, 0), 1: (0, 220, 255), 2: (255, 160, 0), 3: (180, 120, 255)}
NAMES = {0: 'person', 1: 'hand', 2: 'foot', 3: 'head'}
TILE_H = 780
PER_ROW = 2


def main():
    import cv2
    import numpy as np
    stats = json.load(io.open(os.path.join(PL, 'pl_stats.json'), encoding='utf-8'))
    files = stats['sample12']
    os.makedirs(REP, exist_ok=True)
    tiles = []
    for f in files:
        p = os.path.join(PL, 'images', 'train', f)
        im = cv2.imread(p)
        h, w = im.shape[:2]
        stem = os.path.splitext(f)[0]
        lab = os.path.join(PL, 'labels', 'train', stem + '.txt')
        cnt = {0: 0, 1: 0, 2: 0, 3: 0}
        if os.path.exists(lab):
            for line in io.open(lab, encoding='utf-8'):
                v = line.split()
                if len(v) < 5:
                    continue
                c = int(float(v[0]))
                cx, cy, bw, bh = [float(x) for x in v[1:5]]
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(im, (x1, y1), (x2, y2), COLORS.get(c, (255, 255, 255)), 4)
                cv2.putText(im, NAMES.get(c, str(c)), (x1 + 2, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(c, (255, 255, 255)), 2)
                cnt[c] += 1
        im = cv2.resize(im, (int(w * TILE_H / h), TILE_H))
        cv2.putText(im, '%s  p%d h%d f%d hd%d' % (f, cnt[0], cnt[1], cnt[2], cnt[3]), (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), PER_ROW):
        row = tiles[i:i + PER_ROW]
        W = sum(t.shape[1] for t in row)
        canvas = np.zeros((TILE_H, W, 3), np.uint8)
        x = 0
        for t in row:
            canvas[:, x:x + t.shape[1]] = t
            x += t.shape[1]
        rows.append(canvas)
    maxw = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], maxw - r.shape[1], 3), np.uint8)]) for r in rows]
    out = np.vstack(rows)
    dst = os.path.join(REP, 'pl_sample12.jpg')
    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote', dst, out.shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
