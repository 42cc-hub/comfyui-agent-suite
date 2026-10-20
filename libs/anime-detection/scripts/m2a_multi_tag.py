# -*- coding: utf-8 -*-
"""task-7 第 2 步（v2）：WD14 打标并**保留人数标签的概率**，便于离线改阈值。

为什么不直接用 wd14_gate.Tagger.run：它只回"过阈值的标签"，没有概率，
一旦发现 3 人图被漏（`2girls+1boy` 里某个计数标签概率 0.3），就得整批重跑。
这里直接复用 Tagger 的 onnx session，把 girls/boys 两族的**最高概率计数标签**连同概率存下来。

人数解析（**人数代理，不是交互度**）：
  girls = 概率最高的 {1girl..6+girls, multiple_girls}（multiple_girls 记 2）
  boys  同理；total = girls + boys
  `1other` / `no humans` 单独记。

用法：
  python scripts/m2a_multi_tag.py --pool delivered --out data/hand_gt_multi/tags_delivered.csv
  python scripts/m2a_multi_tag.py --pool candidates --out data/hand_gt_multi/tags_candidates.csv
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
DATA = os.path.join(SUITE, 'data')
TASK = r'<你的 ComfyUI 项目>\task'
HAND_GT = os.path.join(DATA, 'hand_gt', 'index.csv')
SKILL = r"<你的用户目录>\.dsh\skills\comfyui-image-generation\scripts"

GIRL = re.compile(r'^([1-9])(\+)?girls?$')
BOY = re.compile(r'^([1-9])(\+)?boys?$')
FIELDS = ['sha256', 'pool', 'stem', 'project', 'path', 'dups', 'w', 'h',
          'rating', 'girls_tag', 'girls_p', 'boys_tag', 'boys_p', 'multi', 'no_humans',
          'girls', 'boys', 'total', 'labeled_in_hand_gt', 'tags']


def sha_of(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, TASK)


def project_of(p):
    return rel(p).split(os.sep)[0]


def pref_rank(p):
    r = rel(p).lower().replace(os.sep, '/')
    for i, key in enumerate(('/deliver/', '/uploads_img/', '/upload/', '/final', '/refined', '/art/')):
        if key in r:
            return i
    return 9


def collect(pool):
    out = []
    for root, dirs, files in os.walk(TASK):
        if pool == 'candidates' and os.path.basename(root) != 'candidates':
            continue
        for f in files:
            low = f.lower()
            if not low.endswith(('.png', '.webp', '.jpg')):
                continue
            if pool == 'delivered' and '_2000' not in os.path.splitext(low)[0]:
                continue
            out.append(os.path.join(root, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', choices=['delivered', 'candidates'], default='delivered')
    ap.add_argument('--out', default='')
    ap.add_argument('--threshold', type=float, default=0.35, help='写 tags 列的门槛；人数两族单独存概率')
    a = ap.parse_args()

    labeled = set()
    if os.path.exists(HAND_GT):
        for r in csv.DictReader(io.open(HAND_GT, encoding='utf-8-sig')):
            labeled.add(r['crop'])

    files = collect(a.pool)
    uniq = {}
    for p in files:
        s = sha_of(p)
        uniq.setdefault(s, []).append(p)
    reps = []
    for s, paths in uniq.items():
        best = sorted(paths, key=pref_rank)[0]
        reps.append({'sha': s, 'path': best, 'dups': len(paths), 'stem': os.path.splitext(os.path.basename(best))[0]})
    reps.sort(key=lambda r: (project_of(r['path']), r['stem']))
    print('%s: 文件 %d → 唯一 %d 张（副本 %d 份）' % (a.pool, len(files), len(reps), len(files) - len(reps)))

    sys.path.insert(0, SKILL)
    from _libs import add_lib; add_lib("wd14-gate")
    from wd14_gate import Tagger
    tg = Tagger()
    tagset = tg.tags
    gi, ci = tg.general_index, tg.character_index

    rows = []
    for i, r in enumerate(reps, 1):
        im = Image.open(r['path']).convert('RGB')
        w, h = im.size
        ratio = float(tg.size) / max(im.size)
        nw, nh = tuple(int(x * ratio) for x in im.size)
        im2 = im.resize((nw, nh), Image.LANCZOS)
        sq = Image.new('RGB', (tg.size, tg.size), (255, 255, 255))
        sq.paste(im2, ((tg.size - nw) // 2, (tg.size - nh) // 2))
        arr = np.array(sq).astype(np.float32)[:, :, ::-1][None, ...]
        out = tg.sess.run([tg.sess.get_outputs()[0].name], {tg.input.name: arr})[0][0]
        probs = out.tolist()
        rating = max([(tagset[k], probs[k]) for k in range(gi)], key=lambda x: x[1])[0]
        gtag, gp = '', 0.0
        btag, bp = '', 0.0
        multi = ''
        no_humans = ''
        gen_tags = []
        for k in range(gi, ci):
            t = tagset[k]
            p = probs[k]
            if p <= a.threshold:
                continue
            gen_tags.append(t)
            if GIRL.match(t) or t == 'multiple_girls':
                if p > gp:
                    gtag, gp = t, p
            elif BOY.match(t) or t == 'multiple_boys':
                if p > bp:
                    btag, bp = t, p
            elif t == 'multiple_others':
                multi = t
            elif t == 'no humans':
                no_humans = t
        # 人数：允许用同一族的"最高概率"标签，即使它没到 0.35（阈值离线可调）
        best_g = max([(tagset[k], probs[k]) for k in range(gi, ci)
                      if GIRL.match(tagset[k]) or tagset[k] == 'multiple_girls'], key=lambda x: x[1])
        best_b = max([(tagset[k], probs[k]) for k in range(gi, ci)
                      if BOY.match(tagset[k]) or tagset[k] == 'multiple_boys'], key=lambda x: x[1])

        def num(tag):
            m = GIRL.match(tag) or BOY.match(tag)
            if m:
                return int(m.group(1))
            if tag in ('multiple_girls', 'multiple_boys'):
                return 2
            return 0

        rows.append({'sha256': r['sha'], 'pool': a.pool, 'stem': r['stem'], 'project': project_of(r['path']),
                     'path': r['path'], 'dups': r['dups'], 'w': w, 'h': h, 'rating': rating,
                     'girls_tag': best_g[0], 'girls_p': round(best_g[1], 4),
                     'boys_tag': best_b[0], 'boys_p': round(best_b[1], 4),
                     'multi': multi, 'no_humans': no_humans,
                     'girls': num(best_g[0]), 'boys': num(best_b[0]),
                     'total': num(best_g[0]) + num(best_b[0]),
                     'labeled_in_hand_gt': int(os.path.basename(r['path']) in labeled),
                     'tags': ';'.join(gen_tags)})
        if i % 40 == 0:
            print('  %d/%d' % (i, len(reps)), flush=True)

    out = a.out or os.path.join(DATA, 'hand_gt_multi', 'tags_%s.csv' % a.pool)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote', out, len(rows))


if __name__ == '__main__':
    main()
