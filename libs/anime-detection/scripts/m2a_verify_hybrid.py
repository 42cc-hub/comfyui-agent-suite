# -*- coding: utf-8 -*-
"""混合权重（person←fix_v2、hand/foot←rot_semi）落地验收。

A person 一致性：holdout15 原图，单模型 fix_v2 vs 混合 ⇒ person 框逐个相同
B hand/foot 归属：单模型 rot_semi vs 混合 ⇒ hand/foot 框逐个相同（都关掉专用脚/第二遍手）
C 确定性：混合跑两次 ⇒ JSON 完全相同
D 直立集泛化：data/hand_gt 40 张，产线口径（conf .30 / imgsz 1280 / 手 1536）算 hand/foot R/P
E 耗时：同 8 张图，单模型 vs 混合 端到端墙钟
"""
import csv
import io
import json
import os
import subprocess
import sys
import time

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PY = sys.executable
LOCATE = os.path.join(SUITE, 'scripts', 'locate.py')
DATA = os.path.join(SUITE, 'data')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'hybrid_verify')
W_FIX = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
W_ROT = os.path.join(SUITE, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt')

ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')


def run(args, tag):
    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    cmd = [PY, '-X', 'utf8', LOCATE, '--out', out, '--no-vitpose', '--src'] + args['src'] + args['extra']
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
    dt = time.time() - t0
    if p.returncode != 0:
        print('FAIL %s rc=%d\n%s\n%s' % (tag, p.returncode, p.stdout[-1500:], p.stderr[-1500:]))
        raise SystemExit(1)
    js = {}
    for f in sorted(os.listdir(out)):
        if f.endswith('.json'):
            js[f] = json.load(io.open(os.path.join(out, f), encoding='utf-8'))
    return js, dt


def boxes_of(js, cls):
    got = {}
    for f, rec in js.items():
        got[f] = sorted(tuple(b['box']) for b in rec['boxes'] if b['cls'] == cls)
    return got


def load_gt():
    """data/hand_gt/index.csv -> {crop: {cls: [boxes]}}（只取 draw 且有框）"""
    gt = {}
    p = os.path.join(DATA, 'hand_gt', 'index.csv')
    for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
        if not r['boxes'].strip() or r['verdict'] == 'neg':
            continue
        boxes = [tuple(float(v) for v in seg.split(',')) for seg in r['boxes'].split(';') if seg.strip()]
        gt.setdefault(r['crop'], {}).setdefault(r['target'], []).extend(boxes)
    return gt


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(js, gt, cls):
    tp = fp = fn = 0
    for f, rec in js.items():
        # 文件名 -> GT crop 名（data/hand_gt 的 crop 是 xxx_2000.png）
        key = f if f in gt else None
        g = gt.get(key, {}).get(cls, []) if key else []
        p = [b['box'] for b in rec['boxes'] if b['cls'] == cls]
        used = set()
        for gb in g:
            best, bi = 0.0, -1
            for i, pb in enumerate(p):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                tp += 1
                used.add(bi)
            else:
                fn += 1
        fp += len(p) - len(used)
    r = tp / (tp + fn) if tp + fn else 0.0
    pr = tp / (tp + fp) if tp + fp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, recall=round(100 * r, 1), precision=round(100 * pr, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    hold = [l.strip().split('\t') for l in io.open(
        os.path.join(DATA, 'hand_gt_multi', 'holdout15.txt'), encoding='utf-8') if l.strip()]
    if hold and len(hold[0]) == 1:  # 纯 stem 列表
        tags = {r['stem']: r for r in csv.DictReader(io.open(
            os.path.join(DATA, 'hand_gt_multi', 'tags_delivered.csv'), encoding='utf-8-sig'))}
        hold_src = [tags[h[0]]['path'] for h in hold]
    else:
        hold_src = [h[1] for h in hold]
    hold_src = [p for p in hold_src if os.path.exists(p)]
    print('holdout15 可用 %d 张' % len(hold_src))

    base = ['--conf', '0.3', '--imgsz', '960', '--hand-imgsz', '960', '--foot-weights', '',
            '--face-weights', '', '--gag-weights', '', '--m3-weights', '']

    # ---- A person 一致性 ----
    a1, _ = run({'src': hold_src, 'extra': base + ['--det-weights', W_FIX]}, 'A_single_fix')
    a2, _ = run({'src': hold_src, 'extra': base}, 'A_hybrid')
    p1, p2 = boxes_of(a1, 'person'), boxes_of(a2, 'person')
    diffA = [f for f in p1 if p1[f] != p2.get(f)]
    print('A person：%d 张，逐框不同 %d 张' % (len(p1), len(diffA)))

    # ---- B hand/foot 归属 ----
    b1, _ = run({'src': hold_src, 'extra': base + ['--det-weights', W_ROT]}, 'B_single_rot')
    b2, _ = run({'src': hold_src, 'extra': base}, 'B_hybrid')
    diffB = []
    for cls in ('hand', 'foot'):
        q1, q2 = boxes_of(b1, cls), boxes_of(b2, cls)
        diffB += [(cls, f) for f in q1 if q1[f] != q2.get(f)]
    print('B hand/foot：逐框不同 %d 处（hand+foot 合计）' % len(diffB))

    # ---- C 确定性（产线口径，5 张）----
    prod = ['--conf', '0.3', '--imgsz', '1280', '--hand-imgsz', '1536']
    c1, _ = run({'src': hold_src[:5], 'extra': prod}, 'C_run1')
    c2, _ = run({'src': hold_src[:5], 'extra': prod}, 'C_run2')
    sameC = all(json.dumps(c1[f], sort_keys=True) == json.dumps(c2[f], sort_keys=True) for f in c1)
    print('C 确定性：两次跑完全一致 =', sameC)

    # ---- D 直立集 hand/foot ----
    gt = load_gt()
    imgs = [os.path.join(DATA, 'hand_gt', 'images', f) for f in sorted(os.listdir(os.path.join(DATA, 'hand_gt', 'images')))
            if f in gt]
    d1, _ = run({'src': imgs, 'extra': prod + ['--det-weights', W_FIX]}, 'D_single_fix')
    d2, _ = run({'src': imgs, 'extra': prod}, 'D_hybrid')
    sD = {'single_fix': {c: score(d1, gt, c) for c in ('hand', 'foot')},
          'hybrid': {c: score(d2, gt, c) for c in ('hand', 'foot')}}
    print('D 直立集：', json.dumps(sD, ensure_ascii=False))

    # ---- E 耗时（8 张，产线口径）----
    e1, t1 = run({'src': hold_src[:8], 'extra': prod + ['--det-weights', W_FIX]}, 'E_single')
    e2, t2 = run({'src': hold_src[:8], 'extra': prod}, 'E_hybrid')
    print('E 耗时：单模型 %.2fs / %.3fs 每图；混合 %.2fs / %.3fs 每图；比值 %.2fx'
          % (t1, t1 / 8, t2, t2 / 8, t2 / t1))

    rep = dict(holdout=len(hold_src), A_person_diff=diffA, B_limb_diff=diffB, C_deterministic=sameC,
               D_upright=sD, E_single_s=t1, E_hybrid_s=t2,
               E_ratio=round(t2 / t1, 3),
               weights=dict(fix_v2=os.path.getsize(W_FIX), rot_semi=os.path.getsize(W_ROT)))
    io.open(os.path.join(OUT, 'hybrid_verify.json'), 'w', encoding='utf-8').write(
        json.dumps(rep, ensure_ascii=False, indent=1))
    print('\n== 汇总 ==')
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
