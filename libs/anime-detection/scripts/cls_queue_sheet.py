"""把 M3（手/脚合理性分类）的待判裁图做成**编号队列**：索引表 + 分页联系表。

为什么分页：整图预览会被 harness 压到 0.64 MP，59 张挤一页就数不清指/趾；
每页 6 张、每张长边 480px ⇒ 预览后仍能看清，用户按编号回 "3 ok / 4 bad_count" 即可。

用法：python scripts/cls_queue_sheet.py [--min-side 48] [--per-page 6]
"""
import argparse
import csv
import io
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CROPS = os.path.join(PROJ, 'data', 'crops_todo')
OUT = os.path.join(PROJ, 'data', 'cls_queue')
TILE = 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-side', type=int, default=48)
    ap.add_argument('--per-page', type=int, default=6)
    ap.add_argument('--cols', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    with io.open(os.path.join(CROPS, 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            p = os.path.join(CROPS, r['crop'])
            if not os.path.exists(p):
                continue
            im = Image.open(p)
            side = max(im.size)
            if side < a.min_side:
                continue
            rows.append(dict(r, path=p, w=im.width, h=im.height, side=side))

    # 手在前、脚在后；同类按长边降序（大的先判，信息多）
    rows.sort(key=lambda r: (0 if r['kind'] == 'hand' else 1, -r['side']))
    # 保留已有人工/自动列（label、state …）——重建队列**不能**把标好的标签冲掉
    old = {}
    idx_path = os.path.join(OUT, 'index.csv')
    if os.path.exists(idx_path):
        with io.open(idx_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                old[r['crop']] = r
    for i, r in enumerate(rows, 1):
        r['idx'] = i
        for k in ('label', 'state', 'bad_flags', 'mismatch', 'auto_tags'):
            r[k] = old.get(r['crop'], {}).get(k, '')

    with io.open(idx_path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['idx', 'crop', 'kind', 'source', 'origin',
                                           'w', 'h', 'conf', 'state', 'mismatch', 'bad_flags',
                                           'label', 'auto_tags'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})

    pages = [rows[i:i + a.per_page] for i in range(0, len(rows), a.per_page)]
    for pi, page in enumerate(pages, 1):
        cols = a.cols
        rws = (len(page) + cols - 1) // cols
        bar = 26
        sheet = Image.new('RGB', (cols * TILE, rws * (TILE + bar)), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(page):
            im = Image.open(r['path']).convert('RGB')
            s = TILE / float(max(im.size))
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            cx, cy = (k % cols) * TILE, (k // cols) * (TILE + bar) + bar
            cell = Image.new('RGB', (TILE, TILE), (16, 16, 16))
            cell.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
            sheet.paste(cell, (cx, cy))
            d.text((cx + 6, cy - 18), '#%d  %s  %s%s  %dx%d  conf=%.2f  %s' %
                   (r['idx'], r['kind'], r.get('state') or '?',
                    ('/' + r['mismatch']) if r.get('mismatch') else '',
                    r['w'], r['h'], float(r['conf'] or 0), r.get('label') or ''),
                   fill=(255, 220, 80))
        p = os.path.join(OUT, 'page_%02d.png' % pi)
        sheet.save(p)
        print('page %02d: #%d~#%d  -> %s' % (pi, page[0]['idx'], page[-1]['idx'], p))

    n_hand = sum(1 for r in rows if r['kind'] == 'hand')
    print('\n队列 %d 张（手 %d / 脚 %d），最小长边 %dpx' % (len(rows), n_hand, len(rows) - n_hand, a.min_side))
    print('索引表 %s' % os.path.join(OUT, 'index.csv'))


if __name__ == '__main__':
    main()
