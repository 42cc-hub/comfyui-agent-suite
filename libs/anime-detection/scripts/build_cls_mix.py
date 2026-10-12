"""构造 `data/cls_bin_mix`：同一批已标注裁图的**两种裁图口径**都进 train，用来训一个口径无关的 M3。

动机（R39/R42）：M3 的判定对裁图口径极其敏感 —— 同一只脚，"旧口径"（`prelabel`：姿态脚点框 + 0.35）
与"产线口径"（`locate.py` 默认：检测器框 + 0.10）下 p(bad) 可以是 1.00 vs 0.00。
现在的两条路都要"选一个口径"（改推断 `--m3-foot-crop pose`，或按产线口径重训）。
第三条路：**训练时两种口径都喂**，让模型学会忽略"取景"这个 nuisance 变量。

    train = cls_bin_v8/train（旧口径 354） + cls_bin_prod/train（产线口径 350）= 704
    val   = cls_val_common（旧口径 50）—— 与两个来源的 train 均无同名重叠（已核对为 0）

产线口径的副本加 `__prod` 后缀，避免与旧口径同名互相覆盖（分类任务不看文件名）。

用法：python scripts/build_cls_mix.py
输出：data/cls_bin_mix/{train,val}/{ok,bad}/*.png + 屏幕计数
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
OLD = os.path.join(DATA, 'cls_bin_v8')
PROD = os.path.join(DATA, 'cls_bin_prod')
VAL = os.path.join(DATA, 'cls_val_common')
OUT = os.path.join(DATA, 'cls_bin_mix')


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    cnt = {}
    for lab in ('ok', 'bad'):
        os.makedirs(os.path.join(OUT, 'train', lab), exist_ok=True)
        os.makedirs(os.path.join(OUT, 'val', lab), exist_ok=True)
        for src, suf in ((os.path.join(OLD, 'train', lab), ''), (os.path.join(PROD, 'train', lab), '__prod')):
            n = 0
            for f in sorted(os.listdir(src)):
                if not f.lower().endswith('.png'):
                    continue
                stem, ext = os.path.splitext(f)
                shutil.copy(os.path.join(src, f), os.path.join(OUT, 'train', lab, stem + suf + ext))
                n += 1
            cnt[('train', lab, suf or 'old')] = n
        n = 0
        for f in sorted(os.listdir(os.path.join(VAL, lab))):
            if f.lower().endswith('.png'):
                shutil.copy(os.path.join(VAL, lab, f), os.path.join(OUT, 'val', lab, f))
                n += 1
        cnt[('val', lab, 'old')] = n
    for k in sorted(cnt):
        print('  %-6s %-4s %-5s %d' % (k[0], k[1], k[2], cnt[k]))
    tr = sum(v for k, v in cnt.items() if k[0] == 'train')
    va = sum(v for k, v in cnt.items() if k[0] == 'val')
    print('%s：train %d / val %d' % (OUT, tr, va))


if __name__ == '__main__':
    main()
