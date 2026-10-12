"""M2b-gag（口塞/眼罩）消融：**多种子重训 + 工作点（阈值）扫描**。

为什么是这两件（2026-10-11，用户给的 8 小时窗口、无新数据）：
  · 交付图上"口塞检出 4/6"这个数字来自**单次训练 + 单阈值**；本项目已多次见到单次运行的方差能翻转单图结论
    （M2b v1→v2 只差 1 张图，07 就从漏检变检出）⇒ 必须先量**种子方差**；
  · v2 与 v4 是"并集使用"的，而两个权重的打分分布不同（v4 更保守）⇒ 阈值到底取 0.5 还是别的，
    要用"交付图召回 vs 池外报出率"这两条曲线一起看，而不是凭一个点。

数据（无新增）：`data/ds_face_gag`（580 张 head 全覆盖：239 gag + 176 blindfold 框，已排除交付图 head）。
评测：① 交付图 8 张里"真有口塞的 6 张"报出几张（真值由用户在 2026-10-11 确认）；
      ② 池外报出率——WD14 说"有 gag/blindfold"、但**从没标过**的 head（`data/m2b_tags_merged.csv` 里剩下的）。

用法：
  python scripts/ablate_gag.py --seeds 3                 # 重训 N 个种子并评估
  python scripts/ablate_gag.py --eval-only runs/m2b_gag_v2  # 只评估已有权重（含阈值扫描）
结果：data/ablation_gag.csv + 控制台阈值扫描表
"""
import argparse
import csv
import io
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
LOGS = os.path.join(PROJ, 'runs', '_logs')
OUT = os.path.join(DATA, 'ablation_gag.csv')
FIELDS = ['tag', 'seed', 'status', 'thr', 'delivery_hits', 'pool_fire', 'pool_n', 'secs']
PY_M = r'python'
SRC = r'<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final'
# 用户 2026-10-11 确认的真值：01/08 无口塞无眼罩；02/03/04/05/06/07 有口塞；8 张全无眼罩
GAG_TRUTH = ['02_escape_corridor_2000.png', '03_escape_nurse_pursuit_2000.png',
             '04_escape_recaptured_2000.png', '05_escape_restraint_start_2000.png',
             '06_escape_restraint_start_b_2000.png', '07_escape_restraint_tightened_2000.png']

EVAL = r'''
import csv, io, json, os, random, sys
from PIL import Image
from ultralytics import YOLO
PROJ = r"%s"
SRC = r"%s"
GAG = ["gag","gagged","ball gag","cloth gag","tape gag","bit gag","cleave gag","improvised gag","covered mouth","mouth hold"]
BL = ["blindfold","black blindfold","eye mask","bandage over one eye","bandage on face","covering face"]
truth = %r
weights = %r
thrs = %r
GIMGSZ = %r          # gag 模型自己的推理分辨率（0 = 用 ultralytics 默认 640）
head = YOLO(os.path.join(PROJ, "runs", "m2a_det_996img_960", "weights", "best.pt"))
mods = [YOLO(w) for w in weights]
# ---- 1) 交付图：每个阈值下"6 张真有口塞的图"报出几张 ----
crops = []
for f in sorted(os.listdir(SRC)):
    if not f.endswith(".png"):
        continue
    p = os.path.join(SRC, f)
    img = Image.open(p).convert("RGB")
    r = head.predict(p, conf=0.3, imgsz=1280, verbose=False)[0]
    for hb in [b for b in r.boxes if head.names[int(b.cls[0])] == "head"]:
        x0, y0, x1, y1 = [float(v) for v in hb.xyxy[0]]
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        c = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(img.width, int(x1 + pad)), min(img.height, int(y1 + pad)))
        cp = os.path.join(PROJ, ".runtime", "_tmp_ag_" + str(len(crops)) + ".png")   # 每个 head 一个文件，别共用同一个路径（会被后一张覆盖）
        img.crop(c).resize((256, 256), Image.LANCZOS).save(cp)   # 与 locate.py 的"标准画布"一致（漏了这步 ⇒ 交付图 0/6）
        crops.append((f, cp))
out = []
for thr in thrs:
    hit = set()
    for f, cp in crops:
        for m in mods:
            res = m.predict(cp, conf=thr, verbose=False, **({} if not GIMGSZ else dict(imgsz=GIMGSZ)))[0]
            if len(res.boxes):
                hit.add(f)
    out.append({"stage": "delivery", "thr": thr, "hits": len([t for t in truth if t in hit])})
# ---- 2) 池外：WD14 说有、但没标过的 head ----
inq = set()
for q in ("m2b_queue","m2b_queue2","m2b_queue3"):
    p = os.path.join(PROJ, "data", q, "index.csv")
    if os.path.exists(p):
        inq |= {r["crop"] for r in csv.DictReader(io.open(p, encoding="utf-8-sig"))}
tags = {r["crop"]: (r["tags"] or "").split("|") for r in csv.DictReader(io.open(os.path.join(PROJ,"data","m2b_tags_merged.csv"), encoding="utf-8"))}
pool = sorted(c for c in tags if c not in inq and ((set(GAG)|set(BL)) & set(tags[c])))
pick = random.Random(7).sample(pool, min(80, len(pool)))
for thr in thrs:
    fire = 0
    miss = 0
    for c in pick:
        p = os.path.join(PROJ, "data", "prelabel", c)
        if not os.path.exists(p):
            miss += 1
            continue
        for m in mods:
            if len(m.predict(p, conf=thr, verbose=False, **({} if not GIMGSZ else dict(imgsz=GIMGSZ)))[0].boxes):
                fire += 1
                break
    out.append({"stage": "pool", "thr": thr, "hits": fire, "n": len(pick) - miss, "miss": miss})
print(json.dumps(out, ensure_ascii=False))
'''


def run_eval(weights, thrs, gimgsz=0):
    code = EVAL % (PROJ, SRC, GAG_TRUTH, weights, thrs, gimgsz)
    r = subprocess.run([PY_M, '-X', 'utf8', '-c', code], capture_output=True, text=True,
                       cwd=PROJ, encoding='utf-8', errors='replace')
    for line in (r.stdout or '').splitlines()[::-1]:
        line = line.strip()
        if line.startswith('['):
            import json
            return json.loads(line)
    print('   !! 评估失败：%s' % ((r.stderr or '')[-300:]), flush=True)
    return []


def append(row):
    new = not os.path.exists(OUT)
    with io.open(OUT, 'a', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--eval-only', default=None, help='只评估已有权重（给 runs/<name>）')
    ap.add_argument('--union', action='store_true', help='评估时把 v2 与 v4 并集使用')
    ap.add_argument('--gag-imgsz', type=int, default=0,
                    help='gag 模型的推理分辨率（0 = 保持 locate.py 现在的行为，即 ultralytics 默认 640；'
                         '模型是 256 画布训练的 ⇒ 640 等于把它上采样 2.5×）')
    a = ap.parse_args()
    thrs = [0.3, 0.4, 0.5, 0.6, 0.7]

    if a.eval_only:
        ws = [os.path.join(PROJ, 'runs', a.eval_only, 'weights', 'best.pt')]
        if a.union:
            ws.append(os.path.join(PROJ, 'runs', 'm2b_gag_v2', 'weights', 'best.pt'))
        print('评估 %s（阈值扫描，gag imgsz=%s）' % (
            [os.path.basename(os.path.dirname(os.path.dirname(w))) for w in ws], a.gag_imgsz or '默认640'))
        for row in run_eval(ws, thrs, a.gag_imgsz):
            if row['stage'] == 'delivery':
                print('   阈值 %.1f → 交付图口塞检出 %d/6' % (row['thr'], row['hits']))
            else:
                print('   阈值 %.1f → 池外报出 %d/%d（缺图 %d）' % (row['thr'], row['hits'], row['n'],
                                                              row.get('miss', 0)))
            # eval-only 也落 CSV：这一轮只做"评估"（不重训），但结果同样要可复算
            append(dict(tag='eval:%s|imgsz=%s' % ('+'.join(
                os.path.basename(os.path.dirname(os.path.dirname(w))) for w in ws), a.gag_imgsz or 640),
                seed='-', status='ok', thr=row['thr'],
                delivery_hits=row.get('hits') if row['stage'] == 'delivery' else '',
                pool_fire=row.get('hits') if row['stage'] == 'pool' else '',
                pool_n=row.get('n', ''), secs=''))
        return

    os.makedirs(LOGS, exist_ok=True)
    for s in range(a.seeds):
        seed = 20261010 + s * 1000
        tag, name = 'gag_v4', 'ab_gag_v4_s%d' % seed
        logp = os.path.join(LOGS, 'ablate_%s.log' % name)
        cmd = [PY_M, '-X', 'utf8', os.path.join(HERE, 'train.py'), '--task', 'det',
               '--data', os.path.join(DATA, 'ds_face_gag', 'gag.yaml'), '--model', 'yolo11n.pt',
               '--epochs', str(a.epochs), '--imgsz', '256', '--name', name, '--seed', str(seed)]
        print('=== %s seed=%d → %s' % (tag, seed, os.path.basename(logp)), flush=True)
        t0 = time.time()
        with io.open(logp, 'w', encoding='utf-8') as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ)
        secs = round(time.time() - t0, 1)
        w = os.path.join(PROJ, 'runs', name, 'weights', 'best.pt')
        ok = (r.returncode == 0 and os.path.exists(w))
        print('   %s (%.0fs)' % ('OK' if ok else 'FAILED', secs), flush=True)
        if not ok:
            append(dict(tag=tag, seed=str(seed), status='failed', secs=secs))
            continue
        ws = [w, os.path.join(PROJ, 'runs', 'm2b_gag_v2', 'weights', 'best.pt')]   # 与生产一致：并集
        for row in run_eval(ws, thrs):
            rr = dict(tag=tag, seed=str(seed), status='ok', thr=row['thr'],
                      delivery_hits=row.get('hits') if row['stage'] == 'delivery' else '',
                      pool_fire=row.get('hits') if row['stage'] == 'pool' else '',
                      pool_n=row.get('n', ''), secs=secs)
            append(rr)
            print('   阈值 %.1f → %s %s' % (row['thr'], row['stage'], row.get('hits')), flush=True)


if __name__ == '__main__':
    main()
