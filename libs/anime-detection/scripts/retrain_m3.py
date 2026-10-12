"""用户判完队列后，**一条命令**走完 M3 的重训与验收（2026-10-12 重写，把本窗口的纪律固化进去）。

与旧版的区别（都是这轮吃过的亏）：
  1. 标签**从所有队列汇总**（`data/cls_queue*/index.csv`），不再只读一个队列；
  2. 切分**按源图分组**（R39/R43：按裁图随机切会让 val 里 37%~81% 的裁图其实在 train 里）；
  3. 不再默认"增强扩充少数类"（R44：3 个种子下与不增强**无差别**）；
  4. 指标给**按源图 5 折折外**（R46：单次切分在这个数据量下折间方差 ±0.165，不可比）；
  5. 验收跑在**交付口径**上（`--imgsz 1600`，R45：1280→1600 让 M3 主判定 3/4→4/4）。

用法：
  python scripts\\retrain_m3.py                    # 全部默认
  ... --epochs 2 --no-oof --no-accept                                      # 冒烟（约 3 分钟）
  ... --tag v2                                                    # 产物名后缀
流程：汇总标签 → 建 `data/cls_bin_<tag>`（按图切分）→ 三种子训练 `m3_<tag>_s<seed>` →
      （默认）折外 5 折 → 交付图验收（1600，三种子并集）。
"""
import argparse
import csv
import glob
import io
import os
import random
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
CROPS = os.path.join(DATA, 'crops_todo')
PY_M = r'python'
PY_S = r'python'
SEEDS = [20261010, 20262010, 20263010]
BAD = ('bad_count', 'bad_shape', 'not_foot', 'not_hand')      # 二元任务：这些都算"不能用"


def log(msg):
    print(msg, flush=True)


def collect():
    """汇总所有队列里已人工标注的裁图 → {crop: 'ok'|'bad'}。"""
    lab = {}
    for p in sorted(glob.glob(os.path.join(DATA, 'cls_queue*', 'index.csv'))):
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            v = (r.get('label') or '').strip()
            if not v or v == 'skip':
                continue
            lab[r['crop']] = 'ok' if v == 'ok' else ('bad' if v in BAD else None)
    lab = {k: v for k, v in lab.items() if v}
    log('汇总标签：%d 张（ok %d / bad %d）' % (len(lab), sum(1 for v in lab.values() if v == 'ok'),
                                           sum(1 for v in lab.values() if v == 'bad')))
    return lab


def src_of(name):
    m = re.match(r'^(.*?)_(hand|left_foot|right_foot|foot)\d*', name)
    return m.group(1) if m else os.path.splitext(name)[0]


def build(lab, tag, val_frac):
    """按**源图**分组切分（同源图只落一边），产线口径的旧裁图直接用 `crops_todo` 里的原图。"""
    out = os.path.join(DATA, 'cls_bin_%s' % tag)
    if os.path.isdir(out):
        shutil.rmtree(out)
    by_img = {}
    for crop, y in lab.items():
        by_img.setdefault(src_of(crop), []).append((crop, y))
    imgs = sorted(by_img)
    random.Random(20261010).shuffle(imgs)
    target = int(round(len(lab) * val_frac))
    val_imgs, n = set(), 0
    for im in imgs:
        if n >= target:
            break
        val_imgs.add(im)
        n += len(by_img[im])
    cnt = {}
    for crop, y in sorted(lab.items()):
        sub = 'val' if src_of(crop) in val_imgs else 'train'
        src = os.path.join(CROPS, crop)
        if not os.path.exists(src):
            continue
        d = os.path.join(out, sub, y)
        os.makedirs(d, exist_ok=True)
        shutil.copy(src, os.path.join(d, crop))
        cnt[(sub, y)] = cnt.get((sub, y), 0) + 1
    # 泄漏自检：val 的源图绝不能在 train 出现（按目录实际内容查，而不是相信上面的分组逻辑）
    def srcs(sub):
        s = set()
        for y in ('ok', 'bad'):
            d = os.path.join(out, sub, y)
            if os.path.isdir(d):
                s |= {src_of(f) for f in os.listdir(d)}
        return s

    leak = srcs('train') & srcs('val')
    log('数据集 %s：%s；源图泄漏 %d（必须为 0）' % (out, dict(cnt), len(leak)))
    return out, cnt


def run(cmd, tag, keep=()):
    log('\n=== %s ===' % tag)
    r = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True, encoding='utf-8', errors='replace')
    for line in (r.stdout or '').splitlines():
        if not keep or any(k in line for k in keep):
            print('  ' + line.rstrip())
    if r.returncode != 0:
        log('  !! 退出码 %d：%s' % (r.returncode, (r.stderr or '')[-400:]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v9', help='产物后缀（数据集 data/cls_bin_<tag>、权重 m3_<tag>_s<seed>）')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--no-oof', action='store_true', help='跳过折外 5 折（省约 10 分钟）')
    ap.add_argument('--no-accept', action='store_true', help='跳过交付图验收')
    ap.add_argument('--imgsz', type=int, default=1600, help='交付验收用的检测分辨率（R45：1600）')
    a = ap.parse_args()

    lab = collect()
    if len(lab) < 60:
        raise SystemExit('标注太少（%d 张），先跑 label_cls.py 标一批' % len(lab))
    d, _ = build(lab, a.tag, a.val_frac)
    ws = []
    for sd in SEEDS:
        name = 'm3_%s_s%d' % (a.tag, sd)
        run([PY_M, '-X', 'utf8', os.path.join(HERE, 'train.py'), '--task', 'cls', '--data', d,
             '--model', 'yolo11n-cls.pt', '--epochs', str(a.epochs), '--imgsz', '224',
             '--name', name, '--seed', str(sd), '--batch', '32'], '训练 %s' % name,
            keep=('Results saved', 'all '))
        w = os.path.join(PROJ, 'runs', name, 'weights', 'best.pt')
        if os.path.exists(w):
            ws.append(w)
        else:
            log('  !! %s 没有产出权重（GPU OOM？看 runs/_logs）' % name)
    if not ws:
        raise SystemExit('三个种子都没出权重')

    if not a.no_oof:
        run([PY_M, '-X', 'utf8', os.path.join(HERE, 'cv_m3.py'), '--src', d, '--folds', '5',
             '--seeds', '2', '--imgsz', '224', '--epochs', str(a.epochs),
             '--out', os.path.join(DATA, 'cv_%s_oof.csv' % a.tag),
             '--work', os.path.join(DATA, 'cv_%s' % a.tag)],
            '按源图 5 折折外（阈值曲线 + 折间方差）',
            keep=('折', 'OOF', '单模型', '并集', '0.', '阈值'))

    if not a.no_accept:
        run([PY_S, '-X', 'utf8', os.path.join(HERE, 'ablate_m3_crop.py'), '--modes', 'det',
             '--imgsz', str(a.imgsz), '--m3-weights'] + ws + ['--tag', 'retrain_%s' % a.tag],
            '交付图验收（imgsz=%d，%d 个权重并集）' % (a.imgsz, len(ws)), keep=('[', '→', '|'))
    log('\n完成。新权重：%s' % ', '.join(os.path.basename(os.path.dirname(os.path.dirname(w))) for w in ws))


if __name__ == '__main__':
    main()
