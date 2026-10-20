"""把 **M2a 手/脚框修复**的人工框混进既有数据集，生成可以直接开训的 yaml。

为什么是"另开一个训练源"而不是"改已有标签"：我标的那 200 张源图**不在 996 预标注池里**
（池里是 `temple__PIC0_020` 这类短名，我标的是 `temple__PIC0_131000608_p0_master1200` 等另一批），
所以修不了现有标签，只能作为**新增图**混进去；而"新增图"最怕别的目标没标（会被当成背景），
所以只在**裁图**上加（裁图里除目标外基本没别的物体），并且**绝不放 val**（val 保持原样，指标才无偏）。

产物（图像用硬链接，不占额外磁盘）：
  data/ds_det_fix/{images,labels}   + det_body_fix.yaml   hand→1 foot→2（对齐 person/hand/foot/head）
  data/ds_hand_fix/{images,labels}  + hand_fix.yaml       本类→0；脚裁图给**空标签**当负样本
  data/ds_foot_fix/{images,labels}  + foot_fix.yaml       本类→0；手裁图给空标签当负样本

用法：python scripts/mix_boxfix.py
"""
import argparse
import csv
import glob
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
SRC_DIRS = [os.path.join(DATA, 'boxfix', 'ds'), os.path.join(DATA, 'boxfix2', 'ds')]
SRC_CLS = {'hand': 0, 'foot': 1}          # boxfix 自己的类别号
TARGETS = {
    'ds_det_fix':  {'names': ['person', 'hand', 'foot', 'head'], 'map': {0: 1, 1: 2}, 'yaml': 'det_body_fix.yaml'},
    'ds_hand_fix': {'names': ['hand'], 'map': {0: 0}, 'yaml': 'hand_fix.yaml'},
    'ds_foot_fix': {'names': ['foot'], 'map': {1: 0}, 'yaml': 'foot_fix.yaml'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clear', action='store_true', help='先清空输出（重跑时用）')
    ap.add_argument('--holdout-frac', type=float, default=0.0,
                    help='按**源图**留出这个比例的修正数据不参与训练（0=全进训练）。'
                         '留出的写进 --holdout-out，用来量"泛化"而不是"背下来"')
    ap.add_argument('--holdout-out', default=os.path.join(DATA, 'boxfix_holdout.csv'))
    ap.add_argument('--repeat', type=int, default=1,
                    help='修正裁图重复几份（>1 = 同一张图在同个 epoch 里出现多次）。'
                         '小修正集 vs 大底集时用 3~5 可以让新口径不被底集冲淡')
    ap.add_argument('--seed', type=int, default=20261014)
    a = ap.parse_args()

    import random
    rng = random.Random(a.seed)
    # 先收集所有修正条目（两个批次的队列），按源图分组
    entries = []
    for q, root in ((os.path.join(DATA, 'boxfix', 'index.csv'), os.path.join(DATA, 'boxfix', 'ds')),
                    (os.path.join(DATA, 'boxfix2', 'index.csv'), os.path.join(DATA, 'boxfix2', 'ds'))):
        if not os.path.exists(q):
            continue
        for r in csv.DictReader(io.open(q, encoding='utf-8-sig')):
            v = (r.get('verdict') or '').strip()
            if not v:
                continue
            note = dict(p.split('=', 1) for p in (r.get('note') or '').split(';') if '=' in p)
            entries.append({'queue': q, 'root': root, 'row': r, 'verdict': v,
                            'kind': 'foot' if 'foot' in r['target'] else 'hand',
                            'src': note.get('src', '')})
    srcs = sorted(set(e['src'] for e in entries))
    rng.shuffle(srcs)
    n_hold = int(round(len(srcs) * a.holdout_frac))
    hold_srcs = set(srcs[:n_hold])
    hold = [e for e in entries if e['src'] in hold_srcs]
    train_e = [e for e in entries if e['src'] not in hold_srcs]
    if a.holdout_frac > 0:
        fields = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
                  'wd14', 'verdict', 'boxes', 'edited', 'note']
        with io.open(a.holdout_out, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for e in hold:
                w.writerow(e['row'])
        print('留出 %d 个源图 / %d 条（写进 %s）；训练用 %d 条'
              % (len(hold_srcs), len(hold), os.path.relpath(a.holdout_out, PROJ), len(train_e)))
    else:
        train_e = entries

    for out, spec in TARGETS.items():
        root = os.path.join(DATA, out)
        if a.clear and os.path.isdir(root):
            shutil.rmtree(root)
        img_dir, lbl_dir = os.path.join(root, 'images'), os.path.join(root, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_img = n_box = n_neg = 0
        for e in train_e:
            stem = os.path.splitext(e['row']['crop'])[0]
            lp = os.path.join(e['root'], 'labels', stem + '.txt')
            ip = os.path.join(e['root'], 'images', stem + '.png')
            if not (os.path.exists(lp) and os.path.exists(ip)):
                continue
            keep = []
            for line in io.open(lp, encoding='utf-8'):
                p = line.split()
                if len(p) >= 5 and int(p[0]) in spec['map']:
                    keep.append('%d %s' % (spec['map'][int(p[0])], ' '.join(p[1:5])))
            for rep in range(max(1, a.repeat)):
                name = stem + ('' if rep == 0 else '__r%d' % rep)
                dst_i = os.path.join(img_dir, name + '.png')
                if os.path.exists(dst_i):
                    continue
                try:
                    os.link(ip, dst_i)
                except Exception:
                    shutil.copy(ip, dst_i)
                io.open(os.path.join(lbl_dir, name + '.txt'), 'w', encoding='utf-8',
                        newline='\n').write(('\n'.join(keep) + '\n') if keep else '')
                n_img += 1
                n_box += len(keep)
                n_neg += 0 if keep else 1
        base = {'ds_det_fix': os.path.join(DATA, 'ds_det'),
                'ds_hand_fix': os.path.join(DATA, 'ds_hand'),
                'ds_foot_fix': os.path.join(DATA, 'ds_foot')}[out]
        y = os.path.join(root, spec['yaml'])
        with io.open(y, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('path: %s\n' % root.replace('\\', '/'))
            fh.write('train:\n  - %s\n  - %s\n' % (os.path.join(base, 'images', 'train').replace('\\', '/'),
                                                   img_dir.replace('\\', '/')))
            fh.write('val: %s\n' % os.path.join(base, 'images', 'val').replace('\\', '/'))
            fh.write('names:\n' + ''.join('  %d: %s\n' % (i, c) for i, c in enumerate(spec['names'])))
        print('%-12s 新增 %3d 图 / %3d 框 / %d 张纯负样本（repeat=%d）→ %s'
              % (out, n_img, n_box, n_neg, max(1, a.repeat), os.path.relpath(y, PROJ)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
