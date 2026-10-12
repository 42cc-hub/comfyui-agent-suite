"""建"眼睛 + 嘴"的补充标注队列（**复用已经标注过 gag/blindfold 的那批 head**）。

用户 2026-10-13 的决定：
  · 目标输出只要 **眼 + 嘴**（二次元的鼻子不明显，可由眼/嘴推断 ⇒ 不标鼻子）；
  · **并进 gag/blindfold 那个检测器**（同一张 256 head 画布，共享面部结构知识）⇒ 四类：eye / mouth / gag / blindfold；
  · 为省数据，**在已经人工判过 gag/blindfold 的 581 张 head 上补标眼和嘴**，而不是新开一批图；
  · 允许**预填**（本轮用户已认可）：预填来自现役 `m2b_face_v3_3983` 的 face/eye 框 + 几何推嘴
    （`face_geom.mouth_from_face_eyes`，眼线定朝向）⇒ 人只需确认/微调。

分层（尽量让"嘴可见/被口塞挡/眼被眼罩挡"三类都够）：
  干净（无 gag 无 blindfold）:  ~150 张   ← 用户强调的部分
  有 gag（嘴被挡）:             ~75 张
  有 blindfold（眼被挡）:       ~75 张
每张 head 出 **两轮**（`target=eye` 与 `target=mouth`），所以 300 张 = 600 轮。

用法：
  python scripts/build_mouth_eye_queue.py --n 300
输出：`data/m2b_queue4/index.csv`（labeler 直接读）+ `data/m2b_queue4/sheet_*.png` 联系表
"""
import argparse
import collections
import csv
import glob
import io
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
OUT = os.path.join(DATA, 'm2b_queue4')
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
          'wd14', 'verdict', 'boxes', 'edited', 'note']
QUEUES = ('m2b_queue', 'm2b_queue2', 'm2b_queue3')


def coverage():
    """→ {crop: (gag?, blindfold?)}，只取"四个目标都判过"的 head。"""
    cov = {}
    p = os.path.join(DATA, 'labels', 'face_extra', '_coverage.csv')
    for r in csv.DictReader(io.open(p, encoding='utf-8')):
        cov[r['crop']] = r
    # 已判内容从各队列读（verdict=draw 且有 boxes ⇒ 有；neg ⇒ 没有）
    have = collections.defaultdict(set)
    for q in QUEUES:
        p = os.path.join(DATA, q, 'index.csv')
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            v = (r.get('verdict') or '').strip()
            if v == 'draw' and (r.get('boxes') or '').strip():
                have[r['crop']].add(r['target'])
    return cov, have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--model', default=os.path.join(PROJ, 'runs', 'm2b_face_v3_3983', 'weights', 'best.pt'))
    ap.add_argument('--per-page', type=int, default=12)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, HERE)
    from face_geom import mouth_from_face_eyes
    from ultralytics import YOLO

    cov, have = coverage()
    pool = {'clean': [], 'gag': [], 'blind': []}
    for crop, r in sorted(cov.items()):
        if not os.path.exists(os.path.join(PRE, crop)):
            continue
        t = have.get(crop, set())
        if 'gag' in t:
            pool['gag'].append(crop)
        elif 'blindfold' in t:
            pool['blind'].append(crop)
        else:
            pool['clean'].append(crop)
    print('候选池：干净 %d / 有 gag %d / 有 blindfold %d（共 %d）' % (
        len(pool['clean']), len(pool['gag']), len(pool['blind']), sum(len(v) for v in pool.values())))
    want = {'clean': int(a.n * 0.5), 'gag': int(a.n * 0.25), 'blind': int(a.n * 0.25)}
    rng = random.Random(20261013)
    pick = []
    for k, w in want.items():
        lst = sorted(pool[k])
        rng.shuffle(lst)
        pick += [(c, k) for c in lst[:min(w, len(lst))]]
    # 不足的名额用其它池补
    if len(pick) < a.n:
        rest = [c for k in pool for c in pool[k] if c not in {x[0] for x in pick}]
        rng.shuffle(rest)
        pick += [(c, 'fill') for c in rest[:a.n - len(pick)]]
    print('选中 %d 张：%s' % (len(pick), dict(collections.Counter(k for _, k in pick))))

    m = YOLO(a.model)
    names = m.names
    face_i = [i for i, n in names.items() if n == 'face']
    eye_i = [i for i, n in names.items() if n == 'eye']
    rows, missing = [], 0
    for i, (crop, kind) in enumerate(pick, 1):
        p = os.path.join(PRE, crop)
        r = m.predict(p, conf=0.25, verbose=False)[0]
        fb, eb, confs = [], [], []
        for b in (r.boxes or []):
            k = int(b.cls[0])
            bx = [float(v) for v in b.xyxy[0].tolist()]
            if k in face_i:
                fb.append((bx, float(b.conf[0])))
            elif k in eye_i:
                eb.append(bx)
                confs.append(float(b.conf[0]))
        if not fb:
            missing += 1
            continue
        face = max(fb, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))[0]
        mb, m_src = mouth_from_face_eyes(face, eb)
        eye_prop = ';'.join(','.join(str(int(round(v))) for v in b) for b in eb)
        rows.append(dict(crop=crop, source=kind, head='', target='eye', prop=eye_prop,
                         prop_src='m2b_face_v3_3983 eye 框', prop_conf=round(sum(confs) / max(1, len(confs)), 3),
                         wd14='', verdict='', boxes='', edited='', note=''))
        rows.append(dict(crop=crop, source=kind, head='', target='mouth',
                         prop=','.join(str(int(round(v))) for v in mb), prop_src=m_src,
                         prop_conf=round(max([c for _, c in fb]), 3),
                         wd14='', verdict='', boxes='', edited='', note=''))
    print('产出 %d 轮（%d 张 head）；没有 face 框跳过 %d 张' % (len(rows), len(rows) // 2, missing))

    # 把**已经判过的 gag/blindfold** 也抄进同一个队列文件 ⇒ 这个 index.csv 就是四类数据集的唯一来源
    # （否则导出时得跨队列拼，容易漏）。这些行带 verdict，用户用 `--only-unlabeled` 时就只会看到新的眼/嘴轮次。
    carried = 0
    for q in QUEUES:
        p = os.path.join(DATA, q, 'index.csv')
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
            if r['crop'] not in {x['crop'] for x in rows}:
                continue
            v = (r.get('verdict') or '').strip()
            if not v:
                continue
            rows.append(dict(r, source='carried:%s' % q, note='（已判，从 %s 抄入；不用重判）' % q))
            carried += 1
    print('抄入已判的 gag/blindfold %d 轮 ⇒ 队列共 %d 轮（其中 %d 轮待判）' % (
        carried, len(rows), sum(1 for r in rows if not (r.get('verdict') or '').strip())))
    os.makedirs(OUT, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['idx'] = i
    with io.open(os.path.join(OUT, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    print('→ %s' % os.path.join(OUT, 'index.csv'))

    # 联系表（把预填画在图上，先看一眼质量再开标）
    if a.per_page:
        from PIL import Image, ImageDraw
        tiles = []
        for r in rows:
            if r['target'] != 'mouth':
                continue
            im = Image.open(os.path.join(PRE, r['crop'])).convert('RGB').resize((256, 256))
            d = ImageDraw.Draw(im)
            for spec, col in ((r['prop'], (0, 230, 230)),):
                if spec:
                    bx = [float(v) for v in spec.split(',')]
                    d.rectangle(bx, outline=col, width=2)
            for rr in rows:
                if rr['crop'] == r['crop'] and rr['target'] == 'eye' and rr['prop']:
                    for part in rr['prop'].split(';'):
                        d.rectangle([float(v) for v in part.split(',')], outline=(255, 160, 0), width=2)
            d.text((4, 4), r['source'], fill=(255, 255, 0))
            tiles.append(im)
        per = a.per_page
        cols = 4
        for pi in range(0, len(tiles), per):
            page = tiles[pi:pi + per]
            rws = (len(page) + cols - 1) // cols
            sheet = Image.new('RGB', (cols * 256, rws * 256), (12, 12, 12))
            for k, im in enumerate(page):
                sheet.paste(im, ((k % cols) * 256, (k // cols) * 256))
            f = os.path.join(OUT, 'sheet_%02d.png' % (pi // per + 1))
            sheet.save(f)
        print('联系表 %d 页 → %s/sheet_*.png（青=嘴预填、橙=眼预填）' % (
            (len(tiles) + per - 1) // per, OUT))


if __name__ == '__main__':
    main()
