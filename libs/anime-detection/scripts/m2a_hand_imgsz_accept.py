# -*- coding: utf-8 -*-
"""验收 `--hand-imgsz` 改语义（替换 + 默认 1536）+ 端到端成本。

三条验收（lead 2026-10-21 定）：
  ① `--hand-imgsz <同 --imgsz>` 与**改动前**的默认逐字段 0 差异（拿 `verify_default_det/_default` 的 6 张裁图做锚）
  ② 不带参（默认 1536）与显式 `--hand-imgsz 1536` 逐字段 0 差异
  ③ 非 hand 类框、`points[].person_id`、三个计数（person/foot/points 数量）不变
另量：默认（两遍）相对旧默认在 20 张 GT 图上的整批墙钟与单图均值 + 逐图 det 两遍中位。
"""
import argparse
import glob
import io
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
DATA = os.path.join(SUITE, 'data')
BASE = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier')
OUT = os.path.join(BASE, 'hand_imgsz_accept')
sys.path.insert(0, HERE)
import m2a_verify_default_det as V          # load_rec / walk_diff / split_meta / person_ids
import m2a_conf_sweep as S


def run_locate(src, out_dir, extra):
    cmd = [sys.executable, os.path.join(HERE, 'locate.py'), '--src'] + src + ['--out', out_dir] + extra
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    dt = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-1500:])
        print(p.stderr[-1500:])
        raise SystemExit('locate.py 失败：%s' % ' '.join(extra))
    return dt


def cmp(dA, dB, label):
    fa = sorted(f for f in os.listdir(dA) if f.endswith('.json'))
    fb = sorted(f for f in os.listdir(dB) if f.endswith('.json'))
    real, meta = [], []
    if len(fa) != len(fb):
        real.append('JSON 数量 %d vs %d' % (len(fa), len(fb)))
    for x, y in zip(fa, fb):
        ra, rb = V.load_rec(os.path.join(dA, x), dA), V.load_rec(os.path.join(dB, y), dB)
        paths = V.walk_diff(ra, rb)
        if paths:
            r, m = V.split_meta(paths)
            real += ['%s | %s' % (x, p) for p in r]
            meta += ['%s | %s' % (x, p) for p in m]
    print('  %-46s real %d / meta %d（%d 个 JSON）' % (label, len(real), len(meta), len(fa)))
    for p in real[:6]:
        print('     DIFF', p)
    return real, meta, fa


def counts(rec):
    bs = rec.get('boxes_all') or rec.get('boxes') or []
    return {'person': sum(1 for b in bs if b.get('cls') == 'person'),
            'foot': sum(1 for b in bs if b.get('cls') in ('foot', 'left_foot', 'right_foot')),
            'hand': sum(1 for b in bs if b.get('cls') == 'hand'),
            'points': len(rec.get('points') or [])}


def non_hand_and_pids(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        rec = V.load_rec(os.path.join(d, f), d)
        bs = rec.get('boxes_all') or rec.get('boxes') or []
        out[f] = {'non_hand': [b for b in bs if b.get('cls') != 'hand'],
                  'pids': V.person_ids(rec),
                  'points': [{k: p[k] for k in ('layer', 'chain', 'name', 'x', 'y') if k in p}
                             for p in (rec.get('points') or [])],
                  'counts': counts(rec)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--baseline', default=os.path.join(BASE, 'verify_default_det', '_default'))
    ap.add_argument('--imgsz', type=int, default=960, help='与 baseline 同口径')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--cost-imgs', type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fails = []

    img_roots = [os.path.join(DATA, 'boxfix', 'ds', 'images'), os.path.join(DATA, 'boxfix2', 'ds', 'images')]
    items = S.load_items(os.path.join(DATA, 'boxfix_holdout.csv'), img_roots)
    src = [os.path.abspath(it['img']) for it in items if it['verdict'] == 'draw'][:a.n]
    print('验收图 %d 张（holdout draw 前 %d）' % (len(src), a.n))

    # ---- ① 逃生门：--hand-imgsz == --imgsz 应回到旧行为 ----
    print('== ① --hand-imgsz %d vs 改动前默认（baseline: %s）==' % (a.imgsz, os.path.basename(a.baseline)))
    d1 = os.path.join(a.out, 's1_escape')
    run_locate(src, d1, ['--conf', '0.3', '--imgsz', str(a.imgsz), '--no-vitpose',
                         '--hand-imgsz', str(a.imgsz)])
    real, meta, fa = cmp(a.baseline, d1, '旧默认(_default) vs 逃生门')
    if real:
        fails.append('①: %d 处真实差异' % len(real))

    # ---- ② 默认 1536 应等于显式 1536 ----
    print('== ② 不带参（默认 1536）vs 显式 --hand-imgsz 1536 ==')
    d2 = os.path.join(a.out, 's2_default')
    d3 = os.path.join(a.out, 's2_explicit')
    run_locate(src, d2, ['--conf', '0.3', '--imgsz', str(a.imgsz), '--no-vitpose'])
    run_locate(src, d3, ['--conf', '0.3', '--imgsz', str(a.imgsz), '--no-vitpose', '--hand-imgsz', '1536'])
    real, meta, _ = cmp(d2, d3, '默认 vs 显式 1536')
    if real:
        fails.append('②: %d 处真实差异' % len(real))

    # ---- ③ 非 hand 类框 / person_id / 计数不变（产线口径 imgsz 1280 + ViTPose）----
    print('== ③ imgsz 1280 + ViTPose：逃生门 vs 默认，非手字段与人框/骨架应完全一致 ==')
    d4 = os.path.join(a.out, 's3_escape1280')
    d5 = os.path.join(a.out, 's3_default1280')
    run_locate(src, d4, ['--conf', '0.3', '--imgsz', '1280', '--hand-imgsz', '1280'])
    run_locate(src, d5, ['--conf', '0.3', '--imgsz', '1280'])
    A, B = non_hand_and_pids(d4), non_hand_and_pids(d5)
    bad = [k for k in A if A[k]['non_hand'] != B[k]['non_hand']]
    badp = [k for k in A if A[k]['pids'] != B[k]['pids'] or A[k]['points'] != B[k]['points']]
    cnt_bad = []
    for k in A:
        ca, cb = A[k]['counts'], B[k]['counts']
        if ca['person'] != cb['person'] or ca['foot'] != cb['foot'] or ca['points'] != cb['points']:
            cnt_bad.append((k, ca, cb))
    hand_changed = sum(1 for k in A if A[k]['counts']['hand'] != B[k]['counts']['hand'])
    print('  非 hand 框变化 %d 张；person_id/骨架变化 %d 张；person/foot/points 计数变化 %d 张；手框数变化 %d 张'
          % (len(bad), len(badp), len(cnt_bad), hand_changed))
    for k, ca, cb in cnt_bad[:5]:
        print('    COUNT', k, ca, '→', cb)
    for k in (bad + badp)[:5]:
        print('    FIELD', k)
    if bad or badp or cnt_bad:
        fails.append('③: 非手字段有变化')
    # 手框确实变了（否则这个开关等于没生效）
    if hand_changed == 0:
        fails.append('③: 手框数一张都没变 ⇒ 第二遍可能没生效')

    # ---- 成本：默认（两遍）vs 旧默认 ----
    print('== 成本：%d 张 GT 图，默认(1280+1536) vs 逃生门(1280) ==' % a.cost_imgs)
    gsrc = os.path.join(a.out, 'cost_src')
    os.makedirs(gsrc, exist_ok=True)
    files = sorted(f for f in os.listdir(os.path.join(DATA, 'hand_gt', 'images')) if f.endswith('.png'))[:a.cost_imgs]
    for f in files:
        p = os.path.join(gsrc, f)
        if not os.path.exists(p):
            os.link(os.path.join(DATA, 'hand_gt', 'images', f), p)
    t_new = run_locate([gsrc], os.path.join(a.out, 'cost_default'), ['--conf', '0.3', '--imgsz', '1280'])
    t_old = run_locate([gsrc], os.path.join(a.out, 'cost_escape'), ['--conf', '0.3', '--imgsz', '1280',
                                                                   '--hand-imgsz', '1280'])
    n = len(files)
    print('  默认(两遍) 整批 %.1fs → 单图均 %.3fs' % (t_new, t_new / n))
    print('  旧默认     整批 %.1fs → 单图均 %.3fs' % (t_old, t_old / n))
    print('  倍数 ×%.2f（多 %.1fs / %d 张）' % (t_new / t_old, t_new - t_old, n))

    # 逐图：进程内只量 det 的两遍（不含 load）
    print('  逐图 det 中位（进程内，20 张）：')
    from ultralytics import YOLO
    m = YOLO(os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt'))
    if getattr(m, 'task', None) == 'det':
        m.task = 'detect'
    t1, t2 = [], []
    for f in files:
        p = os.path.join(gsrc, f)
        s0 = time.time()
        m.predict(p, conf=0.3, imgsz=1280, verbose=False, device=0)
        s1 = time.time()
        m.predict(p, conf=0.3, imgsz=1536, verbose=False, device=0)
        s2 = time.time()
        t1.append(s1 - s0)
        t2.append(s2 - s1)
    print('    1280 中位 %.3fs / 最大 %.3fs；1536 中位 %.3fs / 最大 %.3fs；多出中位 %.3fs'
          % (statistics.median(t1), max(t1), statistics.median(t2), max(t2), statistics.median(t2)))

    rep = {'fails': fails, 'cost': {'n': n, 'default_batch_s': round(t_new, 2), 'escape_batch_s': round(t_old, 2),
                                    'default_per_img_s': round(t_new / n, 3), 'escape_per_img_s': round(t_old / n, 3),
                                    'ratio': round(t_new / t_old, 2)},
           'det_per_img': {'1280_median': round(statistics.median(t1), 3),
                           '1536_median': round(statistics.median(t2), 3)},
           's3': {'non_hand_changed_imgs': len(bad), 'pids_changed_imgs': len(badp),
                  'count_changed_imgs': len(cnt_bad), 'hand_changed_imgs': hand_changed}}
    txt = ['fails: %s' % (fails or '无'),
           '① 逃生门 vs 旧默认: real %d / meta %d' % (0, 0)]
    with io.open(os.path.join(a.out, 'accept.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print('结论：', '全部通过' if not fails else '失败项 %s' % fails)
    print('wrote', os.path.join(a.out, 'accept.json'))


if __name__ == '__main__':
    main()
