"""半监督姿态数据集：**人工真值高权重 + 检测修正预标注低权重**。

为什么这么搭（用户 2026-10-13 的要求）：
  * 人工真值（data/pose_gt，69 张）是干净的，但太少 —— 要**高权重**；
  * 其余图的预标注来自 detection 修正过的 `data/prelabel_face`（面点用 M2b face/eye 合成），
    量大但不可全信 —— **低权重**。

ultralytics 没有按样本加权的接口，所以权重用**按样本重采样**实现：训练列表里人工图标重复 K 次、
自动图各 1 次（同配置下等价于给人工样本 K 倍权重，且不动磁盘上的标签文件）。
两臂共享同一份图像/标签/验证集，**唯一变量就是 train 列表**：
  pose_semi.yaml      train = 人工×K + 自动        （半监督臂 B1）
  pose_semi_ctrl.yaml train = 只有自动             （对照臂 B0，等于"不标人工能到哪"）
验证集固定为**按源图留出的人工图标**（人工真值是唯一可评的基准；留出集不参与任何训练列表）。
"""
import argparse
import glob
import io
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export_labels import COCO17_FLIP, COCO17_NAMES                        # noqa: E402

DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
GT = os.path.join(DATA, 'pose_gt')
LBL17 = os.path.join(DATA, 'labels_semi', 'pose', 'body17')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA, 'ds_pose17_semi'))
    ap.add_argument('--human-repeat', type=int, default=10, help='人工图标在训练列表里重复几次（权重）')
    ap.add_argument('--human-val', type=int, default=15, help='留出多少张人工图标做验证（按源图）')
    ap.add_argument('--seed', type=int, default=20261013)
    a = ap.parse_args()
    out = a.out
    for sub in ('images/train', 'labels/train', 'images/val', 'labels/val'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    human = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(GT, '*.json')))
    rng = random.Random(a.seed)
    rng.shuffle(human)
    val = sorted(human[:a.human_val])
    human_tr = sorted(human[a.human_val:])
    auto = []
    for p in sorted(glob.glob(os.path.join(PRE, '*.json'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in set(human):
            auto.append(stem)

    n = {'train_h': 0, 'train_a': 0, 'val': 0, 'no_img': 0, 'no_lbl': 0}
    lists = {'human': [], 'auto': [], 'val': []}
    for tag, stems in (('human', human_tr), ('auto', auto), ('val', val)):
        for stem in stems:
            rec = json.load(io.open(os.path.join(GT, stem + '.json'), encoding='utf-8')) if tag != 'auto' \
                else json.load(io.open(os.path.join(PRE, stem + '.json'), encoding='utf-8'))
            src = rec.get('src') or ''
            lp = os.path.join(LBL17, stem + '.txt')
            if not (src and os.path.exists(src)):
                n['no_img'] += 1
                continue
            if not os.path.exists(lp):
                n['no_lbl'] += 1
                continue
            sub = 'val' if tag == 'val' else 'train'
            ext = os.path.splitext(src)[1] or '.png'
            dst_img = os.path.join(out, 'images', sub, stem + ext)
            if not os.path.exists(dst_img):
                shutil.copy2(src, dst_img)
            shutil.copy2(lp, os.path.join(out, 'labels', sub, stem + '.txt'))
            if tag == 'val':
                lists['val'].append(dst_img)
                n['val'] += 1
            elif tag == 'human':
                lists['human'].append(dst_img)
                n['train_h'] += 1
            else:
                lists['auto'].append(dst_img)
                n['train_a'] += 1

    semi = lists['human'] * a.human_repeat + lists['auto']
    rng.shuffle(semi)                                    # 打散，避免人工样本总是集中在每个 epoch 前部
    with io.open(os.path.join(out, 'train_semi.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(semi) + '\n')
    with io.open(os.path.join(out, 'train_auto_only.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lists['auto']) + '\n')
    with io.open(os.path.join(out, 'val_human.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lists['val']) + '\n')
    for name, train in (('pose_semi.yaml', 'train_semi.txt'), ('pose_semi_ctrl.yaml', 'train_auto_only.txt')):
        with io.open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('path: %s\ntrain: %s\nval: val_human.txt\nnames:\n  0: person\n'
                     'kpt_shape: [17, 3]\nflip_idx: %s\nkpt_names: %s\n'
                     % (out.replace('\\', '/'), train, COCO17_FLIP, COCO17_NAMES))
    json.dump({'human_train': n['train_h'], 'human_repeat': a.human_repeat, 'auto_train': n['train_a'],
               'samples_per_epoch_semi': len(semi), 'samples_per_epoch_ctrl': len(lists['auto']),
               'val_human': n['val'], 'val_stems': val, 'seed': a.seed,
               'missing_img': n['no_img'], 'missing_label': n['no_lbl']},
              io.open(os.path.join(out, 'semi_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('人工 %d 张（训练） ×%d + 自动 %d 张 ⇒ 每轮 %d 样本（对照臂 %d）'
          % (n['train_h'], a.human_repeat, n['train_a'], len(semi), len(lists['auto'])))
    print('验证 = 人工留出 %d 张（按源图，不进任何训练列表）：%s' % (n['val'], ', '.join(x[:28] for x in val[:3]) + ' …'))
    print('缺原图 %d / 缺标签 %d（若不为 0 说明导出没跑或路径变了）' % (n['no_img'], n['no_lbl']))
    print('数据集：%s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
