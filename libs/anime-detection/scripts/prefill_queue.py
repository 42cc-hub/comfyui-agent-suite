"""给待判队列**预填"规则上可以确定"的标签**，减少人工量。

只填两类（都是有规范依据的规则，不是猜测）：
  1. WD14 状态 = `covered`（裤袜/袜子）或 `shod`（鞋/靴） ⇒ **`skip`**（穿鞋穿袜根本数不了趾，标注规范 §2.1 已定）；
  2. 该裁图同时在"部位认错"队列里（`data/cls_queue_mined_mismatch/index.csv`）⇒ 不填（留给人工按 `5 not_part`，
     因为姿态判定只有 22/423 的召回，宁可让人判）。

其余一律**留空**给人判。默认 `--dry-run`，加 `--apply` 才写回。

用法：python scripts/prefill_queue.py [--queue data/cls_queue_mined] [--apply]
"""
import argparse
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
FIELDS = ['idx', 'crop', 'kind', 'source', 'origin', 'w', 'h', 'conf',
          'state', 'mismatch', 'bad_flags', 'label', 'auto_tags']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(DATA, 'cls_queue_mined'))
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    p = os.path.join(a.queue, 'index.csv')
    rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
    n_skip = n_keep = 0
    for r in rows:
        if (r.get('label') or '').strip():
            continue
        st = (r.get('state') or '').strip()
        if st in ('covered', 'shod'):
            r['label'] = 'skip'
            n_skip += 1
        else:
            n_keep += 1
    print('队列 %d 张：预填 skip（穿鞋/穿袜）%d 张；留给人工 %d 张' % (len(rows), n_skip, n_keep))
    if not a.apply:
        print('（dry-run；加 --apply 才写回）')
        return
    with io.open(p, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    print('已写回 %s' % p)


if __name__ == '__main__':
    main()
