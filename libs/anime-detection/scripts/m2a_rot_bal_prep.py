# -*- coding: utf-8 -*-
"""task-10 追补：0° 加权数据集（单模型，不加伪标签）。

做法：把 ds_det_rot 里 45 张 0° 原图各再复制 2 份（`__b2` / `__b3`，硬链接 + 同名标签），
⇒ 0° 在这些图上的权重从 1× 变 3×（不改变其它角度的权重，也不复制基础集）。
产出 data/ds_det_rot/bal/{images,labels} 与 data/ds_det_rot/det_rot_bal.yaml。
"""
import io
import os
import shutil

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(SUITE, 'data')
ROT = os.path.join(DATA, 'ds_det_rot')
DST = os.path.join(ROT, 'bal')
COPIES = ('__b2', '__b3')
CLS = {'person': 0, 'hand': 1, 'foot': 2, 'head': 3}


def boxes_by_cls(lab_path):
    out = {}
    for line in io.open(lab_path, encoding='utf-8'):
        c = line.split()
        if len(c) >= 5:
            out[int(c[0])] = out.get(int(c[0]), 0) + 1
    return out


def main():
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
    zero = [f for f in sorted(os.listdir(os.path.join(ROT, 'images'))) if f.endswith('__a0.jpg')]
    n_img = n_lab = 0
    boxes = {v: 0 for v in CLS.values()}
    for f in zero:
        stem = f[:-4]
        lab = os.path.join(ROT, 'labels', stem + '.txt')
        assert os.path.exists(lab), lab
        for suf in COPIES:
            for ext in ('.jpg', '.txt'):
                src = os.path.join(ROT, 'images' if ext == '.jpg' else 'labels', stem + ext)
                dst = os.path.join(DST, 'images' if ext == '.jpg' else 'labels', stem + suf + ext)
                assert not os.path.exists(dst), dst
                if ext == '.jpg':
                    os.link(src, dst)   # 硬链接，不占额外空间
                    n_img += 1
                else:
                    shutil.copyfile(src, dst)
                    n_lab += 1
        b = boxes_by_cls(lab)
        for k, v in b.items():
            boxes[k] += v * len(COPIES)

    yaml = os.path.join(ROT, 'det_rot_bal.yaml')
    train = [os.path.join(DATA, 'ds_det', 'images', 'train'),
             os.path.join(DATA, 'ds_det_fix', 'images'),
             os.path.join(ROT, 'images'),
             os.path.join(DST, 'images')]
    lines = ['path: %s' % ROT, 'train:'] + ['  - %s' % t for t in train]
    lines += ['val: %s' % os.path.join(DATA, 'ds_det', 'images', 'val'), 'names:']
    lines += ['  %d: %s' % (v, k) for k, v in CLS.items()]
    io.open(yaml, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    print('0° 源图 %d 张，每张再复制 %d 份 ⇒ 新增 images %d / labels %d' % (len(zero), len(COPIES), n_img, n_lab))
    print('新增标签框（×副本）: %s' % {k: boxes[v] for k, v in CLS.items()})
    n_base = len(os.listdir(os.path.join(DATA, 'ds_det', 'images', 'train')))
    n_fix = len(os.listdir(os.path.join(DATA, 'ds_det_fix', 'images')))
    n_rot = len(os.listdir(os.path.join(ROT, 'images')))
    print('训练集张数: ds_det %d + ds_det_fix %d + rot %d + bal %d = %d'
          % (n_base, n_fix, n_rot, n_img, n_base + n_fix + n_rot + n_img))
    print('wrote', yaml)
    print(io.open(yaml, encoding='utf-8').read())


if __name__ == '__main__':
    main()
