"""从 4 类 det 标签里抽出**单类**数据集（给"专用检测器"用）：默认 foot，可换 hand。

动机（2026-10-11 实测）：4 类模型里 foot 是最弱的一类（960 版 mAP50 0.577 / R 0.541），而产线上"脚趾数量"
恰恰最要紧；单类模型不用和 person/hand/head 抢容量。**并且不同类别的模型"错误的分布"不同 ⇒ 与 4 类取并集能补召回**。

图像用**硬链接**（同卷 NTFS 支持），不额外占磁盘；不支持时回落为拷贝。

用法：python scripts/export_foot_only.py [--cls foot|hand]
"""
import argparse
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
SRC = os.path.join(DATA, 'ds_det')                 # 已有 images/labels + train/val 划分
CLASSES = ['person', 'hand', 'foot', 'head']        # 与 export_labels.DET_CLASSES 一致


def link_or_copy(src, dst):
    if os.path.exists(dst):          # 重跑时目标已存在：先删（硬链接会让 copy 报 SameFileError）
        try:
            os.remove(dst)
        except OSError:
            pass
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', default='foot', choices=['foot', 'hand'])
    ap.add_argument('--no-clean', action='store_true',
                    help='保留目标目录里的旧文件。**默认会先清空**：源数据集重建后，上一版导出的文件'
                         '不再满足条件（无该类/换了划分），留在目录里就是"旧标签混新标签"（2026-10-14 实测踩到）')
    a = ap.parse_args()
    idx = CLASSES.index(a.cls)
    out_dir = os.path.join(DATA, 'ds_' + a.cls)
    if not a.no_clean and os.path.isdir(out_dir):
        for sub in ('train', 'val'):
            for kind in ('images', 'labels'):
                d = os.path.join(out_dir, kind, sub)
                if os.path.isdir(d):
                    for fn in os.listdir(d):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass
    n = {'train': 0, 'val': 0, 'boxes': 0, 'skip': 0}
    for sub in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', sub), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', sub), exist_ok=True)
        img_dir = os.path.join(SRC, 'images', sub)
        lbl_dir = os.path.join(SRC, 'labels', sub)
        for fn in os.listdir(img_dir):
            stem = os.path.splitext(fn)[0]
            lp = os.path.join(lbl_dir, stem + '.txt')
            if not os.path.exists(lp):
                continue
            keep = []
            for line in io.open(lp, encoding='utf-8'):
                p = line.split()
                if len(p) >= 5 and p[0] == str(idx):
                    keep.append('0 ' + ' '.join(p[1:5]))     # 重编号成 0 类
            if not keep:
                n['skip'] += 1
                continue
            link_or_copy(os.path.join(img_dir, fn), os.path.join(out_dir, 'images', sub, fn))
            with io.open(os.path.join(out_dir, 'labels', sub, stem + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(keep) + '\n')
            n[sub] += 1
            n['boxes'] += len(keep)
    with io.open(os.path.join(out_dir, '%s.yaml' % a.cls), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: %s\n'
                 % (out_dir.replace('\\', '/'), a.cls))
    print('%s 单类数据集：%s（train %d / val %d，框 %d；无该类的图跳过 %d）'
          % (a.cls, out_dir, n['train'], n['val'], n['boxes'], n['skip']))


if __name__ == '__main__':
    main()
