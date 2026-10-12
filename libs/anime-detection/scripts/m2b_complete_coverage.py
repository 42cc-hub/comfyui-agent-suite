"""补全 M2b 队列的**目标覆盖**（用户 2026-10-11 发现的标注漏洞）。

问题：轮次是按"WD14 命中的目标"生成的，所以一张 head 往往**只问了 gag 或只问了 blindfold**。
而用户的原话点中要害：「有的图中 gag 和 blindfold 都有，但它只问了我其中一个，所以另一个就没有被标注，
甚至有时候是有口塞却在问眼罩，那么我自然就标注了没有。这些数据拿去训练 detection 可能会有问题。」
在检测训练里，**没标的目标就是背景** ⇒ 等于拿"这里没有口塞"当监督教模型（系统性假负样本）。

本脚本：对**每一个已经出现在队列里的 head**，把它缺的目标轮次补成一整轮（仍然一次一个目标），
这样"完整覆盖"才成立 —— 只有完整覆盖的 head 才能进 gag/blindfold 的训练集
（`export_face_extra.py --require-complete` 是那道闸）。

这一轮不再生成预标注（预标注已被实测否决，通过率 0/41）：`prop` 留空，
但把 WD14 的命中情况写进 `note`，方便判的时候心里有数。

用法：python scripts/m2b_complete_coverage.py --queue data/m2b_queue/index.csv [--dry-run]
"""
import argparse
import collections
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
sys.path.insert(0, HERE)
from m2b_propose import TARGETS, FIELDS                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(DATA, 'm2b_queue', 'index.csv'))
    ap.add_argument('--tags', default=os.path.join(DATA, 'm2b_queue_tags.csv'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    with io.open(a.queue, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    cache = {}
    if os.path.exists(a.tags):
        with io.open(a.tags, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                cache[r['crop']] = r.get('tags', '')

    by_crop = collections.OrderedDict()
    for r in rows:
        by_crop.setdefault(r['crop'], []).append(r)

    add = []
    n_incomplete = 0
    for crop, rs in by_crop.items():
        asked = {r['target'] for r in rs}
        missing = [t for t in TARGETS if t not in asked]
        if missing:
            n_incomplete += 1
        tags = set((cache.get(crop) or '').split('|'))
        for t in missing:
            hit = sorted(TARGETS[t] & tags)
            add.append({'crop': crop, 'source': rs[0].get('source', ''), 'head': rs[0].get('head', ''),
                        'target': t, 'prop': '', 'prop_src': 'none（不再生成预标注）', 'prop_conf': '',
                        'wd14': '|'.join(hit), 'verdict': '', 'boxes': '', 'edited': '',
                        'note': '补全覆盖（原队列只问了 %s）· WD14 %s'
                                % ('/'.join(sorted(asked)), ('命中 ' + '|'.join(hit)) if hit else '未命中')})

    print('队列 %d 轮 / %d 张 head；只问了一个目标的 head %d 张 ⇒ 需要补 %d 轮'
          % (len(rows), len(by_crop), n_incomplete, len(add)))
    c = collections.Counter(r['target'] for r in add)
    print('  补的轮次：%s' % dict(c))
    hitn = sum(1 for r in add if r['wd14'])
    print('  其中 WD14 有命中的（大概率真有该目标）%d 轮，未命中的 %d 轮' % (hitn, len(add) - hitn))
    if a.dry_run:
        return
    if not add:
        print('已经全覆盖，无需补。')
        return

    base = 0
    for r in rows:
        try:
            base = max(base, int(r['idx']))
        except ValueError:
            pass
    for i, r in enumerate(add, 1):
        r['idx'] = base + i
        for k in FIELDS:
            r.setdefault(k, '')
    rows += add
    with io.open(a.queue, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('已补 %d 轮 → %s（总 %d 轮）' % (len(add), a.queue, len(rows)))
    print('  用法：python scripts/label_face.py --queue %s' % a.queue)


if __name__ == '__main__':
    main()
