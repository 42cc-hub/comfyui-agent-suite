"""消融 harness：跑「配置 × 种子」矩阵，每个单元训练后用**同一套指标**评估，结果追加到 CSV（可续跑）。

为什么需要它（2026-10-11 用户给的 8 小时窗口）：
  1. 单次训练的方差实测可达 ±0.026 F1 ⇒ **同配置必须跑 ≥3 个种子**，报 mean±std，否则"更好"无从判断；
  2. 只报 val 不够 —— 本项目反复踩过"val 更好但交付图更差"（R29/R31），所以每个单元都同时评：
       · val：top1 / bad 召回（用 data/cls_bin_v8 的**按图切分**，无泄漏）
       · 交付图：8 张上"已知缺陷报出 x/4"（主判定 ∪ 收紧旗标）
       · 池外报出率：crops_todo 里没标过、且过门限的裁图，模型判 bad 的比例（衡量是否滥报）
  3. 训练可能被别的 GPU 作业挤死（本机 6.4 GB、多个 agent 共用）⇒ 每个单元写独立日志，
     失败/中断的单元会被记为 `failed` 而不是静默消失，重跑时自动跳过已完成的单元。

用法：
  python scripts/ablate.py --list                       # 只打印要跑的单元
  python scripts/ablate.py --only aug_on                # 按 tag 过滤
  python scripts/ablate.py --seeds 3                    # 覆盖种子数
结果：data/ablation_m3.csv（每个单元一行）+ runs/_logs/ablate_<tag>_s<seed>.log
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
LOGS = os.path.join(PROJ, 'runs', '_logs')
PY_M = r'python'
PY_A = r'python'
PY_S = r'python'
OUT_CSV = os.path.join(DATA, 'ablation_m3.csv')
FIELDS = ['tag', 'seed', 'status', 'val_top1', 'bad_recall', 'delivery_main', 'delivery_union',
          'pool_fire', 'secs', 'data', 'extra']

# 矩阵：每个 tag 一个配置（data 目录、模型、imgsz、epochs、额外参数）
# ⚠ 本机 6.4 GB 且与 rope 模块共用 ⇒ 只用 yolo11n + 224/288 + 固定小 batch，避免把别人的作业挤 OOM
CONFIGS = [
    # 基线：修过泄漏的按图切分 + 增强扩充（当前生产口径）
    dict(tag='base_aug', data='cls_bin_v8', model='yolo11n-cls.pt', imgsz=224, epochs=80, extra=['--batch', '32'], val_dir='cls_val_common'),
    # 消融：关掉增强扩充（用户 2026-10-11 提的"堆数据 = 增强"那一档到底值多少）
    dict(tag='no_aug', data='cls_bin_v8_noaug', model='yolo11n-cls.pt', imgsz=224, epochs=80, extra=['--batch', '32'], val_dir='cls_val_common'),
    # 消融（R39 的核心问题）：把已标注裁图按**产线口径**重切后再训 —— 训练/推断口径一致值多少
    #   ⚠ 用**公共 val 集**（crops_todo 与 crops_prod 都有的那些 val 裁图）评，否则两个 val 集不同不可比
    dict(tag='crop_prod', data='cls_bin_prod', model='yolo11n-cls.pt', imgsz=224, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
    # ---- 网络侧（第二批）：输入分辨率与骨干容量 ----
    # 动机：M3 判的是"指/趾多一根"这种**极细粒度**差异，224 输入下手指可能只剩十几个像素；
    #       同时 yolo11n 的容量是否够也是未知数。两者都只改网络，数据固定用第一批的**赢家**
    #       （`no_aug`：2 个种子 top1 都是 0.84 > base_aug 0.82，且池外报出更低）。
    dict(tag='na_res288', data='cls_bin_v8_noaug', model='yolo11n-cls.pt', imgsz=288, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
    dict(tag='na_res320', data='cls_bin_v8_noaug', model='yolo11n-cls.pt', imgsz=320, epochs=80,
         extra=['--batch', '24'], val_dir='cls_val_common'),
    dict(tag='na_s224', data='cls_bin_v8_noaug', model='yolo11s-cls.pt', imgsz=224, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
    dict(tag='na_s288', data='cls_bin_v8_noaug', model='yolo11s-cls.pt', imgsz=288, epochs=80,
         extra=['--batch', '24'], val_dir='cls_val_common'),
    # 消融：**口径不变性** —— train 里同时放旧口径与产线口径的同一批裁图（`build_cls_mix.py`，704 张），
    # 让"取景"变成 nuisance 变量；若它在两种口径下都不差，就不必再"选口径"（R39 的正解）。
    dict(tag='mix_conv', data='cls_bin_mix', model='yolo11n-cls.pt', imgsz=224, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
    # ---- 类别口径（第三批）：什么算 bad ----
    # 背景（2026-10-12 实测）：现产线的二元定义把 `not_hand`/`not_foot`（框根本不是该部位）也算 bad，
    # 而这类样本占了 bad 的 **126/174 = 72%**（真正的"数错/畸形"只有 48 张！）
    # ⇒ 模型大半在学"像不像手/脚"，而不是"指/趾数对不对"。
    # `tax_parts` = 只留 {bad_count, bad_shape}（40 训练 / 8 val）；`tax_3cls` = ok/bad_count/bad_shape。
    # ⚠ 两者都用**公共 val**（含 not_* 裁图）评 —— 在这个 val 上 tax_parts 会"掉召回"，但那是**定义不同**
    # 的必然结果（它没学过 not_*）。真正可比的数字来自 `cv_m3.py` 在同一批"真部位"裁图上的折外曲线。
    dict(tag='tax_parts', data='cls_tax_parts', model='yolo11n-cls.pt', imgsz=224, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
    dict(tag='tax_3cls', data='cls_tax_3cls', model='yolo11n-cls.pt', imgsz=224, epochs=80,
         extra=['--batch', '32'], val_dir='cls_val_common'),
]


def free_vram_mb():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=15)
        used, total = [int(x) for x in r.stdout.strip().split(',')]
        return total - used
    except Exception:
        return 9999


def wait_for_vram(need_mb=1500, max_wait=1800):
    """起跑前等显存：本机是和 rope 模块共用的，抢显存会把两边都拖死/OOM。"""
    t0 = time.time()
    while free_vram_mb() < need_mb and time.time() - t0 < max_wait:
        print('   等显存：可用 %d MB < %d MB，30 s 后重试' % (free_vram_mb(), need_mb), flush=True)
        time.sleep(30)
    return free_vram_mb()


def read_done():
    done = set()
    if os.path.exists(OUT_CSV):
        with io.open(OUT_CSV, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if r.get('status') == 'ok':
                    done.add((r['tag'], r['seed']))
    return done


def append(row):
    new = not os.path.exists(OUT_CSV)
    with io.open(OUT_CSV, 'a', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def val_metrics(weights, data_dir):
    """调 `eval_m3.py`（单一实现）算 val top1 / bad 召回 + 池外报出率。

    `data_dir` 可以是数据集根（含 train/val），也**可以直接是 val 目录**（如公共 val 集 `cls_val_common`）。
    """
    vd = data_dir
    if not (os.path.isdir(os.path.join(vd, 'ok')) or os.path.isdir(os.path.join(vd, 'bad'))):
        vd = os.path.join(data_dir, 'val')
    r = subprocess.run([PY_M, '-X', 'utf8', os.path.join(HERE, 'eval_m3.py'), '--weights', weights,
                        '--val-dir', vd, '--pool-n', '120'],
                       capture_output=True, text=True, cwd=PROJ, encoding='utf-8', errors='replace')
    import json
    for line in (r.stdout or '').splitlines()[::-1]:
        line = line.strip()
        if line.startswith('{'):
            d = json.loads(line)
            d['rate'] = d.get('pool_rate')
            return d
    print('   !! eval_m3 解析失败：%s' % ((r.stderr or '')[-200:]), flush=True)
    return {}


def delivery_metrics(weights):
    """8 张交付图：主判定报出几张 / 主判定∪收紧旗标报出几张（真值 4 张有缺陷）。"""
    out_dir = os.path.join(PROJ, '.runtime', 'ablate_delivery')
    src = r'<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final'
    r = subprocess.run([PY_S, '-X', 'utf8', os.path.join(HERE, 'locate.py'), '--src', src,
                        '--out', out_dir, '--imgsz', '1280', '--m3-weights', weights],
                       capture_output=True, text=True, cwd=PROJ, encoding='utf-8', errors='replace')
    known = {'04_escape_recaptured_2000.png', '05_escape_restraint_start_2000.png',
             '07_escape_restraint_tightened_2000.png', '08_escape_feather_tickle_2000.png'}
    import glob
    import json
    main = union = 0
    for p in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        d = json.load(io.open(p, encoding='utf-8'))
        if os.path.basename(d['image']) not in known:
            continue
        bad = [x for x in d['flags'] if x.get('type') == 'part_check' and x.get('verdict') not in ('ok', 'skip', None)]
        tight = [x for x in d['flags'] if x.get('type') == 'foot_tight_check' and x.get('verdict') != 'ok']
        main += 1 if bad else 0
        union += 1 if (bad or tight) else 0
    return {'delivery_main': main, 'delivery_union': union}


def pool_fire(weights):
    """池外报出率：还没标过、且过门限（conf/size/edge）的裁图里，模型判 bad 的比例。"""
    code = (
        "import csv, io, os, json, glob\n"
        "from PIL import Image\n"
        "from ultralytics import YOLO\n"
        "man = {r['crop']: r for r in csv.DictReader(io.open(os.path.join(r'%s','data','crops_todo','manifest.csv'), encoding='utf-8'))}\n"
        "done = set()\n"
        "for q in ('cls_queue','cls_queue_mined','cls_queue_next','cls_queue_all2'):\n"
        "    p = os.path.join(r'%s','data',q,'index.csv')\n"
        "    if os.path.exists(p):\n"
        "        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):\n"
        "            if (r.get('label') or '').strip(): done.add(r['crop'])\n"
        "m = YOLO(r'%s')\n"
        "bad = [i for i, n in m.names.items() if n == 'bad']\n"
        "import numpy as np\n"
        "n = 0; fire = 0\n"
        "for c, r in sorted(man.items()):\n"
        "    if c in done: continue\n"
        "    p = os.path.join(r'%s','data','crops_todo',c)\n"
        "    if not os.path.exists(p): continue\n"
        "    im = Image.open(p)\n"
        "    if min(im.size) < 64: continue\n"
        "    a = np.asarray(im.convert('L').resize((128,128), Image.BILINEAR), dtype=np.float32)\n"
        "    gy, gx = np.gradient(a)\n"
        "    if float(np.mean(np.hypot(gx,gy))) < 4.0: continue\n"
        "    n += 1\n"
        "    pr = m.predict(p, verbose=False)[0].probs.data.tolist()\n"
        "    fire += 1 if max(pr[j] for j in bad) >= 0.5 else 0\n"
        "    if n >= 200: break\n"
        "print(json.dumps({'n': n, 'fire': fire, 'rate': fire / max(1, n)}))\n"
        % (PROJ, PROJ, weights, PROJ)
    )
    out = subprocess.run([PY_M, '-X', 'utf8', '-c', code], capture_output=True, text=True,
                         cwd=PROJ, encoding='utf-8', errors='replace')
    for line in (out.stdout or '').splitlines()[::-1]:
        line = line.strip()
        if line.startswith('{'):
            import json
            return json.loads(line)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='只跑 tag 里含该子串的配置')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--eval-only', default=None, help='只评估已有权重（给 runs/<name>/weights/best.pt 的 name）')
    a = ap.parse_args()

    if a.eval_only:
        w = os.path.join(PROJ, 'runs', a.eval_only, 'weights', 'best.pt')
        print('val  :', val_metrics(w, os.path.join(DATA, 'cls_bin_v8')))
        print('交付 :', delivery_metrics(w))
        print('池外 :', pool_fire(w))
        return

    cfgs = [c for c in CONFIGS if a.only is None or a.only in c['tag']]
    if a.list:
        for c in cfgs:
            print('  %-12s data=%s model=%s imgsz=%d epochs=%d' % (c['tag'], c['data'], c['model'],
                                                                  c['imgsz'], c['epochs']))
        return
    os.makedirs(LOGS, exist_ok=True)
    done = read_done()
    for c in cfgs:
        for s in range(a.seeds):
            seed = 20261010 + s * 1000
            if (c['tag'], str(seed)) in done:
                print('跳过（已完成） %s seed=%d' % (c['tag'], seed), flush=True)
                continue
            name = 'ab_%s_s%d' % (c['tag'], seed)
            logp = os.path.join(LOGS, 'ablate_%s_s%d.log' % (c['tag'], seed))
            cmd = [PY_M, '-X', 'utf8', os.path.join(HERE, 'train.py'), '--task', 'cls',
                   '--data', os.path.join(DATA, c['data']), '--model', c['model'],
                   '--epochs', str(c['epochs']), '--imgsz', str(c['imgsz']), '--name', name,
                   '--seed', str(seed)] + c['extra']
            print('=== %s (seed=%d) → %s' % (c['tag'], seed, os.path.basename(logp)), flush=True)
            print('   可用显存 %d MB' % wait_for_vram(), flush=True)
            t0 = time.time()
            with io.open(logp, 'w', encoding='utf-8') as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
            secs = round(time.time() - t0, 1)
            w = os.path.join(PROJ, 'runs', name, 'weights', 'best.pt')
            row = dict(tag=c['tag'], seed=str(seed), status='ok' if (r.returncode == 0 and os.path.exists(w))
                       else 'failed', secs=secs, data=c['data'],
                       extra=' '.join(c['extra']))
            if row['status'] == 'ok':
                row.update(val_metrics(w, os.path.join(DATA, c.get('val_dir') or c['data'])))
                row.update(delivery_metrics(w))
                row.update(pool_fire(w))
                print('    val=%s 交付=%s/%s 池外=%s%% 用时%.0fs'
                      % (row.get('val_top1'), row.get('delivery_main'), row.get('delivery_union'),
                         round(100 * (row.get('rate') or 0), 1), secs), flush=True)
            else:
                print('   !! 失败（returncode=%s），日志见 %s' % (r.returncode, logp), flush=True)
            append(row)


if __name__ == '__main__':
    main()
