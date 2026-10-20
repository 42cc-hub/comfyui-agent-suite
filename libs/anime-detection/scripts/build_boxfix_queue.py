"""给 **M2a 手/脚框修复** 建一条标注队列（复用 `label_face.py` 那套画框交互）。

为什么要单独建队列：`data/crops_todo/` 里那 1972 张裁图是**按原框 1:1 裁的（无外扩）** ——
框贴腕/踝、手指脚趾本来就落在裁图外，拿它修框根本看不见要补的部分。
所以这里**从源图重新裁**：把原框外扩 `pad`，原框作为预填（`prop`，crop 坐标）画在裁图上，
人拖角改正后，`note` 里记着 crop 原点，**回标到源图坐标**是一步加法。

产物：
  data/boxfix/crops/<name>.png   外扩后的裁图（给标注器看）
  data/boxfix/index.csv          队列（列名与 label_face.py 的 FIELDS 一致）

用法：
  python scripts/build_boxfix_queue.py [--limit 120] [--pad 0.75] [--per-source 1]
"""
import argparse
import csv
import io
import os
import re
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MANIFEST = os.path.join(PROJ, 'data', 'crops_todo', 'manifest.csv')
OUT = os.path.join(PROJ, 'data', 'boxfix')
SRC_ROOTS = [r'<你的图片目录>']
EXTS = ('.png', '.jpg', '.jpeg', '.webp')
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
          'wd14', 'verdict', 'boxes', 'edited', 'note']


def resolve(src):
    """`temple__PIC1_glby (21)` → `<你的图片目录>\\PIC1\\glby (21).png`（99.4% 可解析）。"""
    m = re.match(r'^temple__(PIC\d)_(.*)$', src or '')
    if not m:
        return None
    pic, rest = m.group(1), m.group(2)
    for base in SRC_ROOTS:
        for sub in (pic, '_' + pic):
            for e in EXTS:
                p = os.path.join(base, sub, rest + e)
                if os.path.exists(p):
                    return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=120, help='只取前 N 行（小批试手感；0=全要）')
    ap.add_argument('--pad', type=float, default=0.75,
                    help='外扩比例（相对框的长边）；最小 70px。第一批实测 17 个框里 5 个贴裁图边界 ⇒ '
                         '下一批建议 1.25 / 最小 140')
    ap.add_argument('--pad-min', type=int, default=70, help='外扩的绝对下限 px')
    ap.add_argument('--per-source', type=int, default=1, help='每张源图每类最多取几条')
    ap.add_argument('--out', default=OUT, help='输出目录（换批次时换一个，别覆盖已判的）')
    ap.add_argument('--exclude', default=None,
                    help='已有的 gt_boxes.csv / index.csv：里面出现过的 (src,kind) 不再出题')
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(os.path.join(out_dir, 'crops'), exist_ok=True)
    done = set()
    if a.exclude and os.path.exists(a.exclude):
        import re as _re
        for r in csv.DictReader(io.open(a.exclude, encoding='utf-8-sig')):
            s = r.get('src') or ''
            if not s:                       # index.csv 的 note 里才带 src
                s = (dict(p.split('=', 1) for p in (r.get('note') or '').split(';') if '=' in p)
                     ).get('src', '')
            if s:
                done.add((s, r.get('kind') or r.get('target')))
        print('  排除已判过的 %d 个 (源图,类) 组合' % len(done))
    rows = list(csv.DictReader(io.open(MANIFEST, encoding='utf-8-sig')))
    cand, seen, n_skip = [], {}, 0
    for r in rows:
        sp = resolve(r['source'])
        if not sp:
            n_skip += 1
            continue
        kind = 'foot' if 'foot' in r['kind'] else 'hand'
        if (sp, kind) in done:
            n_skip += 1
            continue
        key = (r['source'], kind)
        if seen.get(key, 0) >= a.per_source:
            continue
        try:
            x0, y0, x1, y1 = [int(float(r[k])) for k in ('x0', 'y0', 'x1', 'y1')]
            w, h = Image.open(sp).size
        except Exception:
            n_skip += 1
            continue
        seen[key] = seen.get(key, 0) + 1
        # 排序键：框相对画面越小 ⇒ 越可能是"只框到腕/踝"的那种（先看这些）
        cand.append((max(x1 - x0, y1 - y0) / float(max(w, h)), sp, kind, (x0, y0, x1, y1), r))
    cand.sort(key=lambda t: t[0])
    if a.limit:
        cand = cand[:a.limit]

    out_rows, n_pad = [], 0
    for i, (_frac, sp, kind, (x0, y0, x1, y1), r) in enumerate(cand, 1):
        im = Image.open(sp).convert('RGB')
        W, H = im.size
        pad = max(a.pad_min, int(a.pad * max(x1 - x0, y1 - y0)))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
        crop = im.crop((cx0, cy0, cx1, cy1))
        name = '%s_%s_%02d.png' % (r['source'], kind, i)
        crop.save(os.path.join(out_dir, 'crops', name))
        n_pad += (cx1 - cx0) * (cy1 - cy0) - (x1 - x0) * (y1 - y0)
        out_rows.append({
            'idx': i, 'crop': name, 'source': r['source'], 'head': '',
            'target': kind,
            'prop': '%d,%d,%d,%d' % (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0),   # crop 坐标的预填
            'prop_src': 'm2a_det(旧框)', 'prop_conf': r.get('conf', ''),
            'wd14': '', 'verdict': '', 'boxes': '', 'edited': '',
            'note': 'src=%s;crop_org=%d,%d;box_src=%d,%d,%d,%d'
                    % (sp, cx0, cy0, x0, y0, x1, y1),
        })
    with io.open(os.path.join(out_dir, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    kinds = {}
    for r in out_rows:
        kinds[r['target']] = kinds.get(r['target'], 0) + 1
    print('写出 %d 行 → %s' % (len(out_rows), os.path.join(out_dir, 'index.csv')))
    print('  分类：%s｜跳过的行（源图找不到）%d｜平均外扩 %.0f%%'
          % (kinds, n_skip, 100.0 * a.pad))
    print('  例：%s' % out_rows[0]['note'] if out_rows else '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
