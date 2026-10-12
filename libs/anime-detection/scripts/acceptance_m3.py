"""M3 验收脚本：把"4 处已知缺陷是否都报出来"变成一条命令。

依据用户 2026-10-10 的人工判断（这是本项目唯一的 ground truth）：
    test1 脚趾少 / 手指多　test4 脚趾多 / 手指畸形　test6 脚趾畸形　test7 脚趾多
对应的裁图标签在 `data/cls_queue/index.csv` 里（`origin=escape` 的那些行）。

判定口径（二元）：真实标签是 `ok` 就是 ok；是 `bad_count`/`bad_shape`/`not_*` 都算 `bad`；
`skip`（太小/糊/穿鞋袜）**不计入**（生产里会被 WD14 的 state 提前拦掉）。

用法：python scripts/acceptance_m3.py [--weights runs/m3_cls_bin_v1/weights/best.pt] [--crops data/crops_todo]
"""
import argparse
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(PROJ, 'runs', 'm3_cls_bin_v1', 'weights', 'best.pt'))
    ap.add_argument('--crops', default=os.path.join(DATA, 'crops_todo'))
    ap.add_argument('--index', default=os.path.join(DATA, 'cls_queue', 'index.csv'))
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    names = m.names
    rows = [r for r in csv.DictReader(io.open(a.index, encoding='utf-8-sig'))
            if r.get('origin') == 'escape' and (r.get('label') or '').strip()]
    hit = miss = skipped = 0
    print('%-34s %-11s %-9s %s' % ('裁图', '人工标签', '判定', '对否'))
    for r in rows:
        p = os.path.join(a.crops, r['crop'])
        if not os.path.exists(p):
            print('%-34s %-11s %-9s 缺图' % (r['crop'], r['label'], '-'))
            continue
        if r['label'] == 'skip':
            skipped += 1
            print('%-34s %-11s %-9s (不计入)' % (r['crop'], r['label'], '-'))
            continue
        true = 'ok' if r['label'] == 'ok' else 'bad'
        res = m.predict(p, verbose=False)[0]
        pred = names[int(res.probs.top1)]
        pbad = 0.0
        try:
            pbad = float(res.probs.data.tolist()[list(names.values()).index('bad')])
        except Exception:
            pass
        ok = (pred == true)
        hit += ok
        miss += (not ok)
        print('%-34s %-11s %-9s %s  p(bad)=%.2f'
              % (r['crop'], r['label'], pred, '✓' if ok else '✗', pbad))
    tot = hit + miss
    print('\n验收：%d/%d 正确（不计入 skip %d 张）%s'
          % (hit, tot, skipped, '　**通过**' if miss == 0 else '　未通过：还有 %d 张漏/误' % miss))
    print('⚠️ 泄漏提醒：这些裁图**多数在训练集里**（逃出那 8 张同时用于训练与验收）⇒ 这里只当"冒烟"；'
          '诚实指标看 val 划分（`runs/*/results.csv`）与后续的人工队列。')


if __name__ == '__main__':
    main()
