"""建 M2b **四类覆盖补全**队列 v5：给"从没人判过"的 head 一把补 eye/mouth/gag/blindfold。

用户 2026-10-13："接下来有整块时间标数据"；13 日末合并后的四类模型用 287 张 head 训出了第一版
（m2b_face4_v1，交付图上 eye 8/8 张、blindfold 0 误报，但 gag 只认到 3/6 张 —— 数据量是瓶颈）。
本队列用**现有 head、不新开图**把训练集从 287 张扩到 ~670 张。

为什么四类必须一起判（2026-10-14 实测，无来源重叠的 57 张 val，conf 0.4 / IoU 0.5）：
    eye 召回 0.906 ｜ mouth 0.608 ｜ gag 0.591 ｜ blindfold 0.846
所以**不能**拿模型伪标注去补 mouth/gag —— 漏掉的框会变成"这里没有嘴/没有口塞"的负样本，正好教反。
能进训练集的只有**人判过四目标全覆盖**的 head（build_face4.py 里就是这条门槛）。

分档（判定成本随档下降；档位只用 gag/眼罩 的两个信号，因为 eye/mouth 每个 head 本来就有）：
  A = WD14 说"有 gag/眼罩" **且** 现役 gag 模型也报框  ← 最可能是真目标，看一眼确认（队列最前）
  B = WD14 说有、但现役模型没报        ← 模型漏检的难例，值得人看
  C = WD14 说没有、但模型报了          ← 可能是 WD14 漏（实测 1~2%），也可能是模型误报
  （两边都说没有 ⇒ **自动判 neg**，不入队；依据：287 张人判样本上 WD14"说没有"时真没有的比例
   gag 98.9% / blindfold 98.0%）

预填：gag/blindfold 用现役 gag 模型（v2∪v4）；eye/mouth 用四类模型 m2b_face4_v1（降 conf 到 0.25 换召回，
反正人只要按一下 `A` 采纳整张 head）。B 档没有 gag 框可填，只把 WD14 命中标签写进 `wd14` 列当提示。

⚠️ 自动 neg 的 ~544 张**只能当误报测试集**，不能进四类训练集：它们没有 eye/mouth 标签，
   进训练等于把"有嘴"教成背景。当测试集用是干净的 —— 两个信号都说没有 ⇒ gag/眼罩真没有的概率 98~99%。

用法：python scripts/build_m2b_cover_queue.py [--limit 400]
输出：data/m2b_queue5/index.csv + _tiers.csv + auto_neg.csv + sheet_*.png
      然后：python scripts/label_face.py --queue data/m2b_queue5/index.csv
            （一张 head 四个目标排在一起，按 `A` 一次全采纳，只在没预填时停下）
"""
import argparse
import collections
import csv
import glob
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
PRE = os.path.join(DATA, 'prelabel')
OUT = os.path.join(DATA, 'm2b_queue5')
TAGS_CSV = os.path.join(DATA, 'm2b_tags_merged.csv')
OLD_GAG = [os.path.join(PROJ, 'runs', n, 'weights', 'best.pt') for n in ('m2b_gag_v2', 'm2b_gag_v4')]
FACE4 = os.path.join(PROJ, 'runs', 'm2b_face4_v1', 'weights', 'best.pt')
QUEUES = ['m2b_queue', 'm2b_queue2', 'm2b_queue3', 'm2b_queue4']
TARGETS = ['eye', 'mouth', 'gag', 'blindfold']
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
          'wd14', 'verdict', 'boxes', 'edited', 'note']
GAG_TAGS = {'gag', 'gagged', 'ball gag', 'cloth gag', 'tape gag', 'bit gag', 'cleave gag',
            'improvised gag', 'covered mouth', 'mouth hold'}
BL_TAGS = {'blindfold', 'black blindfold', 'eye mask', 'bandage over one eye', 'bandage on face',
           'covering face'}
COLOR = {'eye': (255, 165, 0), 'mouth': (0, 220, 220), 'gag': (255, 60, 60), 'blindfold': (170, 0, 255)}


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def dedup(boxes):
    """同一目标被两个权重各报一次 ⇒ 并集里会出现两个几乎重合的框，人是按 `A` 全采纳的，会变成重复真值。
    这里按 IoU≥0.5 去重留高分那个（eye/mouth 天然多框，只有真重合才动）。"""
    out = []
    for b, cf in sorted(boxes, key=lambda t: -t[1]):
        if not any(_iou(b, o) >= 0.5 for o, _ in out):
            out.append((b, cf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只出前 N 张 head（0 = 全出）')
    ap.add_argument('--conf', type=float, default=0.5, help='gag/眼罩预填的置信度阈值')
    ap.add_argument('--face-conf', type=float, default=0.25, help='eye/mouth 预填的置信度阈值（低一点换召回）')
    ap.add_argument('--sheet-per-page', type=int, default=24)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    tags = {r['crop']: set(filter(None, (r.get('tags') or '').split('|')))
            for r in csv.DictReader(io.open(TAGS_CSV, encoding='utf-8'))}
    judged = set()                      # 已经判过的 (crop,target)：本队列不再重复问
    for q in QUEUES:
        p = os.path.join(DATA, q, 'index.csv')
        if os.path.exists(p):
            for r in csv.DictReader(io.open(p, encoding='utf-8-sig')):
                if (r.get('verdict') or '').strip():
                    judged.add((r['crop'], r['target']))
    done_by_crop = collections.defaultdict(set)
    for c, t in judged:
        done_by_crop[c].add(t)
    dsf = set()
    for sub in ('train', 'val'):
        d = os.path.join(DATA, 'ds_face', 'images', sub)
        if os.path.isdir(d):
            dsf |= set(os.listdir(d))
    # 入队门槛 = **四目标还不齐**（不是"任一目标没判"）：老队列里剩的单目标缺口也要在这里补齐，
    # 否则那张 head 永远进不了 build_face4.py 的全覆盖门槛。
    pool = sorted(c for c in dsf if c in tags and os.path.exists(os.path.join(PRE, c))
                  and (set(TARGETS) - done_by_crop[c]))
    print('候选 head %d（ds_face %d 张 − 四目标已齐的 %d 张）'
          % (len(pool), len(dsf), len(dsf) - len(pool)))

    gag_mods = [YOLO(w) for w in OLD_GAG]
    face4 = YOLO(FACE4) if os.path.exists(FACE4) else None
    if face4 is None:
        print('⚠ 没有 %s ⇒ eye/mouth 只能空手画' % FACE4)
    tiers, rows, auto_neg = collections.defaultdict(list), [], []
    for c in pool:
        tg = tags[c]
        w_gag, w_bl = bool(GAG_TAGS & tg), bool(BL_TAGS & tg)
        p = os.path.join(PRE, c)
        pre = collections.defaultdict(list)          # target -> [(box, conf)]
        for m in gag_mods:                           # gag/blindfold 预填 + 分档信号，都只看现役 gag 模型
            r = m.predict(p, conf=a.conf, verbose=False)[0]
            for b in (r.boxes or []):
                nm = m.names[int(b.cls[0])]
                if nm in ('gag', 'blindfold'):
                    pre[nm].append(([round(float(v)) for v in b.xyxy[0].tolist()], float(b.conf[0])))
        if face4 is not None:                        # eye/mouth 预填只看四类模型（它的 gag/blindfold 不掺进来）
            r = face4.predict(p, conf=a.face_conf, verbose=False)[0]
            for b in (r.boxes or []):
                nm = face4.names[int(b.cls[0])]
                if nm in ('eye', 'mouth'):
                    pre[nm].append(([round(float(v)) for v in b.xyxy[0].tolist()], float(b.conf[0])))
        if not done_by_crop[c] and not (w_gag or w_bl) and not pre['gag'] and not pre['blindfold']:
            auto_neg.append(c)
            continue
        tier = 'A' if ((w_gag or w_bl) and (pre['gag'] or pre['blindfold'])) else \
               ('B' if (w_gag or w_bl) else 'C')
        tiers[tier].append(c)
        hint = '|'.join(sorted((GAG_TAGS | BL_TAGS) & tg))
        for tgt in TARGETS:
            if (c, tgt) in judged:
                continue
            bx = dedup(pre[tgt])
            prop = ';'.join(','.join(str(v) for v in b) for b, _ in bx)
            rows.append(dict(crop=c, source='cover%s' % tier, head='', target=tgt,
                             prop=prop,
                             prop_src=('m2b_face4_v1 预填' if tgt in ('eye', 'mouth') and prop
                                       else ('m2b_gag_v2∪v4 预填' if prop else '')),
                             prop_conf=(round(max(cf for _, cf in bx), 3) if bx else ''),
                             wd14=hint, verdict='', boxes='', edited='', note=''))
        if a.limit and len(tiers['A']) + len(tiers['B']) + len(tiers['C']) >= a.limit:
            break
    order = {'A': 0, 'B': 1, 'C': 2}
    rows.sort(key=lambda r: (order[r['source'][-1]], r['crop'], TARGETS.index(r['target'])))
    os.makedirs(OUT, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['idx'] = i
    with io.open(os.path.join(OUT, 'index.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    with io.open(os.path.join(OUT, '_tiers.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'tier'])
        for t in ('A', 'B', 'C'):
            for c in tiers[t]:
                w.writerow([c, t])
    with io.open(os.path.join(OUT, 'auto_neg.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['crop', 'why'])
        for c in auto_neg:
            w.writerow([c, 'WD14 说没有 且 现役 gag 模型没报 ⇒ 只当误报测试集，不进训练'])
    n_head = len({r['crop'] for r in rows})
    n_pre = sum(1 for r in rows if r['prop'])
    n_done = len(dsf) - len(pool)        # ds_face 里四目标已齐的 head
    print('档位：A %d / B %d / C %d 张 head；自动 neg %d 张（误报测试集）' % (
        len(tiers['A']), len(tiers['B']), len(tiers['C']), len(auto_neg)))
    print('待判 %d 轮 / %d 张 head（%d 轮有预填）；判完 ds_face 四目标齐的 head %d → ≈ %d 张' % (
        len(rows), n_head, n_pre, n_done, n_done + n_head))
    print('按 `A` 一张 head 一下 ≈ %.0f~%.0f 分钟；逐目标按 1 ≈ %.0f~%.0f 分钟' % (
        n_head * 1.5 / 60.0, n_head * 3 / 60.0, len(rows) * 2.5 / 60.0, len(rows) * 4 / 60.0))
    print('→ %s' % os.path.join(OUT, 'index.csv'))

    per = a.sheet_per_page
    cols = 6
    src_of = dict((r['crop'], r['source']) for r in rows)
    tiles = []
    for c in sorted({r['crop'] for r in rows}, key=lambda c: (order[src_of[c][-1]], c)):
        im = Image.open(os.path.join(PRE, c)).convert('RGB').resize((256, 256))
        d = ImageDraw.Draw(im)
        for r in rows:
            if r['crop'] == c and r['prop']:
                for part in r['prop'].split(';'):
                    d.rectangle([float(v) for v in part.split(',')], outline=COLOR[r['target']], width=2)
        d.text((4, 4), '%s %s' % (c.split('_head')[0][:14], c.split('_')[-2]), fill=(255, 255, 0))
        tiles.append(im)
    for pi in range(0, len(tiles), per):
        page = tiles[pi:pi + per]
        rws = (len(page) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * 256, rws * 256), (12, 12, 12))
        for k, t in enumerate(page):
            sheet.paste(t, ((k % cols) * 256, (k // cols) * 256))
        sheet.save(os.path.join(OUT, 'sheet_%02d.png' % (pi // per + 1)))
    print('联系表 %d 页（橙=eye、青=mouth、红=gag、紫=blindfold，都是预填）'
          % ((len(tiles) + per - 1) // per))


if __name__ == '__main__':
    main()
