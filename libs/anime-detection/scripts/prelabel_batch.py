"""各模块的**预标注批处理**（CPU，不抢 GPU）：M1 骨架 / M2a 身体组 / M2b 面部组 / M3 手·脚裁图。

设计依据（README「体系设计 v0.2」+「统一化策略」）：
  - 现成开源模型只当**标注器**，人只做修正；输出统一到我们的点列契约与目录规范；
  - M2b 是 M2a 的**下游**：head 框裁出 → 归一化 256×256 再检脸/眼（尺度固定 ⇒ 好训好标）；
  - M3 的输入是**裁图**（手：`detect_hands`；脚：M1 的踝/趾点包围盒），人只点一下标 ok/bad。

用法：
  python scripts/prelabel_batch.py --escape                     # 逃出收容所那 8 张
  python scripts/prelabel_batch.py --escape --sample 32         # 再抽 32 张语料
  python scripts/prelabel_batch.py --src <目录或文件...> [--limit N]
"""
import argparse
import csv
import io
import json
import os
import random
import shutil
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')     # 见 README 坑 0：Xet 走不通
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
from PIL import Image, ImageDraw

from imgutils.detect import (detect_eyes, detect_faces, detect_hands, detect_heads,
                             detect_person, detect_with_booru_yolo)
from imgutils.pose import dwpose_estimate

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
IMAGES = os.path.join(DATA, 'images')
PRE = os.path.join(DATA, 'prelabel')
VIS = os.path.join(PRE, 'vis')
CROPS = os.path.join(DATA, 'crops_todo')
ESCAPE = r"<你的 ComfyUI 项目>\task\escape_containment_20261010\references\originals"
TEMPLE = r"<你的图片目录>"
TEMPLE_POOLS = ['_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4']

# OpenPose18 的 18 个身体点（顺序即模型输出顺序）
BODY18 = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'r_ear', 'l_eye', 'l_ear']
HAND21 = ['wrist',
          'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
          'index_mcp', 'index_pip', 'index_dip', 'index_tip',
          'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
          'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
          'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
FOOT3 = ['ankle', 'big_toe', 'small_toe']
SCORE_VIS = (0.5, 0.3)          # >=0.5 → vis 2；>=0.3 → vis 1；否则 0


def vis_of(score):
    return 2 if score >= SCORE_VIS[0] else (1 if score >= SCORE_VIS[1] else 0)


def points_from_kp(kp, chain):
    """把 dwpose 的一个实例拆成我们契约里的 points[]。"""
    out = []
    for part, names in (('body', BODY18), ('left_hand', HAND21), ('right_hand', HAND21),
                        ('left_foot', FOOT3), ('right_foot', FOOT3)):
        pts = getattr(kp, part, None)
        if pts is None:
            continue
        a = np.asarray(pts, dtype=float)
        if a.size == 0:
            continue
        if a.ndim == 1:
            a = a.reshape(1, -1)
        for i, row in enumerate(a):
            x, y, sc = float(row[0]), float(row[1]), float(row[-1])
            out.append({'layer': 'M1', 'name': '%s.%s' % (part, names[i] if i < len(names) else i),
                        'cls': part, 'x': round(x, 1), 'y': round(y, 1),
                        'prob': round(sc, 3), 'vis': vis_of(sc), 'chain': chain})
    return out


def crop_and_label(img, box, pad_frac, min_side, path):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pad = int(pad_frac * max(w, h))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
    if min(x1 - x0, y1 - y0) < min_side:
        return None
    img.crop((x0, y0, x1, y1)).save(path)
    return [x0, y0, x1, y1]


def sample_temple(n, rng):
    """从 5 个池**分层**抽样（只读源目录）。返回 [(路径, 池名)]。"""
    picked = []
    per = max(1, n // len(TEMPLE_POOLS))
    for pool in TEMPLE_POOLS:
        d = os.path.join(TEMPLE, pool)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        rng.shuffle(files)
        for f in files[:per]:
            picked.append((os.path.join(d, f), pool))
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--escape', action='store_true')
    ap.add_argument('--src', nargs='*', default=None)
    ap.add_argument('--sample', type=int, default=0, help='从 <你的图片目录> 分层抽 N 张')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-booru', action='store_true')
    a = ap.parse_args()
    for d in (IMAGES, PRE, VIS, CROPS):
        os.makedirs(d, exist_ok=True)

    items = []                                     # (显示名, 路径, 来源)
    if a.escape:
        for f in sorted(os.listdir(ESCAPE), key=lambda s: int(''.join(c for c in s if c.isdigit()))):
            if f.lower().endswith('.png'):
                items.append((f.replace(' ', '_').replace('(', '').replace(')', ''), os.path.join(ESCAPE, f), 'escape'))
    for p in (a.src or []):
        if os.path.isdir(p):
            items += [(os.path.basename(x), x, 'src') for x in
                      sorted(os.listdir(p)) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        else:
            items.append((os.path.basename(p), p, 'src'))
    if a.sample:
        items += [('temple_%s_%s' % (pool, os.path.splitext(os.path.basename(f))[0]),
                   os.path.join(TEMPLE, pool, f), 'temple')
                  for f, pool in sample_temple(a.sample, random.Random(20261010))]
    if a.limit:
        items = items[:a.limit]
    if not items:
        print('没有输入（用 --escape / --src / --sample）')
        return

    manifest, rows = [], []
    for name, path, origin in items:
        t0 = time.time()
        img = Image.open(path).convert('RGB')
        rec = {'image': name, 'origin': origin, 'src': path, 'size': '%dx%d' % img.size,
               'points': [], 'boxes': [], 'chain': [], 'errors': {}}
        # 可视化画在副本上：**裁图与脸/眼检测必须用干净的原图**，否则框线会被烘进训练数据
        vis = img.copy()
        drw = ImageDraw.Draw(vis)

        # ---- M1 骨架（同时是 M3 脚部裁图的来源）----
        kps = []
        try:
            kps = dwpose_estimate(img)
            rec['chain'].append('M1.dwpose')
            for pi, kp in enumerate(kps):
                rec['points'] += points_from_kp(kp, 'M1.dwpose#%d' % pi)
            rec['n_people'] = len(kps)
        except Exception as e:
            rec['errors']['M1'] = '%s: %s' % (type(e).__name__, e)

        # ---- M2a 身体组 ----
        dets = []
        for fn, cls in ((detect_person, 'person'), (detect_hands, 'hand')):
            try:
                r = fn(img)
                rec['chain'].append('M2a.%s' % cls)
                dets += [(b, cls, float(c)) for b, _l, c in r]
            except Exception as e:
                rec['errors']['M2a.%s' % cls] = '%s: %s' % (type(e).__name__, e)
        try:
            r = detect_heads(img)
            rec['chain'].append('M2a.head')
            dets += [(b, 'head', float(c)) for b, _l, c in r]
        except Exception as e:
            rec['errors']['M2a.head'] = '%s: %s' % (type(e).__name__, e)
        if not a.no_booru:
            try:
                r = detect_with_booru_yolo(img)
                rec['chain'].append('M2a.booru_yolo')
                dets += [(b, 'booru:' + l, float(c)) for b, l, c in r]
            except Exception as e:
                rec['errors']['M2a.booru'] = '%s: %s' % (type(e).__name__, e)
        for b, cls, c in dets:
            rec['boxes'].append({'layer': 'M2a', 'cls': cls, 'box': [int(v) for v in b], 'prob': round(c, 3)})
            col = {'person': (0, 255, 0), 'hand': (255, 80, 80), 'head': (80, 160, 255)}.get(cls, (200, 200, 0))
            drw.rectangle([int(v) for v in b], outline=col, width=3)

        # ---- M2b 面部组：head 框裁 256×256 再检脸/眼 ----
        heads = [b for b, cls, _c in dets if cls == 'head']
        for hi, hb in enumerate(heads):
            x0, y0, x1, y1 = [int(v) for v in hb]
            pad = int(0.15 * max(x1 - x0, y1 - y0))
            hx0, hy0 = max(0, x0 - pad), max(0, y0 - pad)
            hx1, hy1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
            head = img.crop((hx0, hy0, hx1, hy1)).resize((256, 256), Image.LANCZOS)
            hp = os.path.join(PRE, '%s_head%d_256.png' % (name, hi))
            head.save(hp)
            sx, sy = (hx1 - hx0) / 256.0, (hy1 - hy0) / 256.0
            for fn, cls in ((detect_faces, 'face'), (detect_eyes, 'eye')):
                try:
                    r = fn(head)
                    rec['chain'].append('M2b.%s' % cls)
                    for b, _l, c in r:
                        bx = [int(hx0 + b[0] * sx), int(hy0 + b[1] * sy),
                              int(hx0 + b[2] * sx), int(hy0 + b[3] * sy)]
                        rec['boxes'].append({'layer': 'M2b', 'cls': cls, 'box': bx,
                                             'prob': round(float(c), 3), 'from': 'head%d_256' % hi})
                        drw.rectangle(bx, outline=(255, 0, 255), width=2)
                except Exception as e:
                    rec['errors']['M2b.%s' % cls] = '%s: %s' % (type(e).__name__, e)

        # ---- M3 裁图队列：手（检测器）+ 脚（M1 踝/趾点包围盒）----
        for j, (b, cls, c) in enumerate([(b, cls, c) for b, cls, c in dets if cls == 'hand']):
            cp = os.path.join(CROPS, '%s_hand%02d.png' % (name, j + 1))
            bx = crop_and_label(img, [int(v) for v in b], 0.10, 48, cp)
            if bx:
                manifest.append({'crop': os.path.basename(cp), 'source': name, 'kind': 'hand',
                                 'x0': bx[0], 'y0': bx[1], 'x1': bx[2], 'y1': bx[3],
                                 'conf': round(c, 3), 'origin': origin, 'label': ''})
        for pi, kp in enumerate(kps):
            for side in ('left_foot', 'right_foot'):
                pts = getattr(kp, side, None)
                if pts is None:
                    continue
                arr = np.asarray(pts, dtype=float)
                if arr.size == 0 or (arr[:, -1] >= 0.3).sum() < 2:      # 至少两个点可信才算"看见脚"
                    continue
                ok = arr[arr[:, -1] >= 0.3]
                bx = [int(ok[:, 0].min()), int(ok[:, 1].min()), int(ok[:, 0].max()), int(ok[:, 1].max())]
                cp = os.path.join(CROPS, '%s_%s_p%d.png' % (name, side, pi))
                real = crop_and_label(img, bx, 0.35, 48, cp)
                if real:
                    manifest.append({'crop': os.path.basename(cp), 'source': name, 'kind': side,
                                     'x0': real[0], 'y0': real[1], 'x1': real[2], 'y1': real[3],
                                     'conf': round(float(ok[:, -1].mean()), 3), 'origin': origin, 'label': ''})

        # ---- 落盘 ----
        with io.open(os.path.join(PRE, '%s.json' % name), 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        # ---- 把骨架点画到可视化上（只画 vis>=1 的，避免一堆"推断点"糊住画面）----
        BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                 (1, 11), (11, 12), (12, 13), (0, 14), (14, 15), (0, 16), (16, 17)]
        by_chain = {}
        for pt in rec['points']:
            if pt['cls'] == 'body':
                by_chain.setdefault(pt['chain'], {})[pt['name'].split('.', 1)[1]] = pt
        for pts in by_chain.values():
            for ia, ib in BONES:                     # 注意别用 a/b 当循环变量：会遮蔽 argparse 的命名空间 a
                na, nb = BODY18[ia], BODY18[ib]
                if na in pts and nb in pts and pts[na]['vis'] >= 1 and pts[nb]['vis'] >= 1:
                    drw.line([pts[na]['x'], pts[na]['y'], pts[nb]['x'], pts[nb]['y']],
                             fill=(0, 255, 255), width=3)
            for pt in pts.values():
                if pt['vis'] >= 1:
                    x, y = pt['x'], pt['y']
                    drw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 0))
        for pt in rec['points']:
            if pt['cls'] in ('left_hand', 'right_hand', 'left_foot', 'right_foot') and pt['vis'] >= 2:
                x, y = pt['x'], pt['y']
                drw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 128))

        vis.save(os.path.join(VIS, '%s_vis.png' % name))
        rows.append({'image': name, 'origin': origin, 'people': rec.get('n_people', -1),
                     'points': len(rec['points']), 'boxes': len(rec['boxes']),
                     'sec': round(time.time() - t0, 1), 'errors': len(rec['errors'])})
        print('[%-28s] people=%-2s points=%-4d boxes=%-3d %.1fs %s'
              % (name, rec.get('n_people', -1), len(rec['points']), len(rec['boxes']),
                 rows[-1]['sec'], ','.join(rec['errors']) if rec['errors'] else ''), flush=True)

    with io.open(os.path.join(CROPS, 'manifest.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop', 'source', 'kind', 'x0', 'y0', 'x1', 'y1',
                                           'conf', 'origin', 'label'])
        w.writeheader()
        w.writerows(manifest)
    print('\n=== 汇总（%d 张）===' % len(rows))
    print('%-30s %-8s %-7s %-7s %s' % ('image', 'origin', 'people', 'points', 'boxes'))
    for r in rows:
        print('%-30s %-8s %-8s %-7s %-7s' % (r['image'], r['origin'], r['people'], r['points'], r['boxes']))
    print('\n裁图（待人工判 ok/bad）%d 张 → %s' % (len(manifest), CROPS))
    print('点列/框 JSON → %s；可视化 → %s' % (PRE, VIS))


if __name__ == '__main__':
    main()
