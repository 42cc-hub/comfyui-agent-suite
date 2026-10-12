"""用 dghs-imgutils 的二次元检测模型在 escape 8 张上跑基线。

目的：量出"现成件能到哪一步"—— 我们要知道 person / hand / dwpose 这三件在二次元绳缚图上
到底能检出什么，再决定哪一层真的需要自己标数据训模型。

同时把手部检测框裁成小图导出（`data/crops_todo/`），作为"手/脚合理性分类"任务 A 的待判清单。

用法：python scripts/baseline_imgutils.py [--task-dir <escape 任务目录>] [--limit N]
"""
import argparse
import csv
import io
import json
import os
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')   # Xet 走不通：经本机代理 6132 握手超时、直连 401（探针见 .runtime/probe_hf_proxy.py）
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
# 网络事实（2026-10-10 实测，用户提醒后测的）：
#   huggingface.co  **必须**走系统代理 127.0.0.1:6132（直连 10060 超时；走代理 1 MB / 2.8 s）
#   cas-server.xethub.hf.co（Xet） 经代理 SSL 握手超时 ⇒ 只能关掉 Xet 走普通 LFS
#   hf-mirror.com  **不能**与代理同用（308），是不走代理时的备用路：设 HF_ENDPOINT 并临时清空 HTTP(S)_PROXY
# 所以默认不设 HF_ENDPOINT：让 huggingface_hub 自己吃系统代理，只关 Xet。

import numpy as np
from PIL import Image

from imgutils.detect import detect_hands, detect_person
from imgutils.detect.visual import detection_visualize
from imgutils.pose import dwpose_estimate
from imgutils.pose.visual import op18_visualize

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RUNTIME = os.path.join(PROJ, '.runtime', 'baseline_imgutils')
CROPS = os.path.join(PROJ, 'data', 'crops_todo')
DEFAULT_TASK = r"<你的 ComfyUI 项目>\task\escape_containment_20261010"
PARTS = ['body', 'face', 'left_hand', 'right_hand', 'left_foot', 'right_foot']


def part_stat(pts, thr=0.3):
    """返回 (总数, 超过阈值的点数)。pts 是 numpy 数组（x,y,score 或 x,score）——**不能用 `if not pts`**（数组真值歧义）。"""
    if pts is None:
        return 0, 0
    a = np.asarray(pts, dtype=float)
    if a.size == 0:
        return 0, 0
    if a.ndim == 1:
        a = a.reshape(1, -1)
    score = a[:, -1] if a.shape[1] >= 2 else np.zeros(a.shape[0])
    return int(a.shape[0]), int((score >= thr).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-dir', default=DEFAULT_TASK)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    for d in (RUNTIME, CROPS):
        os.makedirs(d, exist_ok=True)

    srcs = sorted([f for f in os.listdir(os.path.join(a.task_dir, 'references', 'originals'))
                   if f.lower().endswith('.png')],
                  key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
    if a.limit:
        srcs = srcs[:a.limit]

    rows = []
    manifest = []
    for name in srcs:
        tid = int(''.join(ch for ch in name if ch.isdigit()))
        path = os.path.join(a.task_dir, 'references', 'originals', name)
        img = Image.open(path).convert('RGB')
        rec = {'image': name, 'size': '%dx%d' % img.size}
        t0 = time.time()

        try:
            persons = detect_person(img)
            rec['person_n'] = len(persons)
            rec['person_boxes'] = [[list(b), round(float(c), 3)] for b, _l, c in persons]
        except Exception as e:
            persons = []
            rec['person_n'] = -1
            rec['person_err'] = '%s: %s' % (type(e).__name__, e)

        try:
            hands = detect_hands(img)
            rec['hand_n'] = len(hands)
            rec['hand_boxes'] = [[list(b), round(float(c), 3)] for b, _l, c in hands]
        except Exception as e:
            hands = []
            rec['hand_n'] = -1
            rec['hand_err'] = '%s: %s' % (type(e).__name__, e)

        try:
            kps = dwpose_estimate(img)
            rec['pose_people'] = len(kps)
            rec['pose_parts'] = []
            for kp in kps:
                d = {}
                for part in PARTS:
                    pts = getattr(kp, part, None)
                    n, ok = part_stat(pts)
                    d[part] = '%d/%d' % (ok, n)
                rec['pose_parts'].append(d)
        except Exception as e:
            kps = []
            rec['pose_people'] = -1
            rec['pose_err'] = '%s: %s' % (type(e).__name__, e)

        rec['sec'] = round(time.time() - t0, 1)
        rows.append(rec)

        try:
            if persons or hands:
                detection_visualize(img, persons + hands, fp=os.path.join(RUNTIME, 'det_test%d.png' % tid))
        except Exception as e:
            print('  visualize det fail: %s' % e, flush=True)
        try:
            if kps:
                op18_visualize(img, kps).save(os.path.join(RUNTIME, 'pose_test%d.png' % tid))
        except Exception as e:
            print('  visualize pose fail: %s' % e, flush=True)

        for j, (box, label, conf) in enumerate(hands):
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = int(0.10 * max(x1 - x0, y1 - y0))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            cp = os.path.join(CROPS, 'test%d_hand%02d.png' % (tid, j + 1))
            img.crop((x0, y0, x1, y1)).save(cp)
            manifest.append({'crop': os.path.basename(cp), 'source': name, 'kind': 'hand',
                             'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'conf': round(float(conf), 3),
                             'label': ''})

        print('[%s] person=%s hand=%s pose=%s  %.1fs%s%s%s' %
              (name, rec.get('person_n'), rec.get('hand_n'), rec.get('pose_people'), rec['sec'],
               ('  personERR=%s' % rec['person_err']) if rec.get('person_err') else '',
               ('  handERR=%s' % rec['hand_err']) if rec.get('hand_err') else '',
               ('  poseERR=%s' % rec['pose_err']) if rec.get('pose_err') else ''), flush=True)

    with io.open(os.path.join(RUNTIME, 'baseline.json'), 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    with io.open(os.path.join(CROPS, 'manifest.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop', 'source', 'kind', 'x0', 'y0', 'x1', 'y1', 'conf', 'label'])
        w.writeheader()
        w.writerows(manifest)

    print('\n=== 汇总 ===')
    print('%-16s %-8s %-8s %-8s %s' % ('image', 'person', 'hand', 'pose人', '手掌/手关键点(阈值0.3)'))
    for r in rows:
        parts = r.get('pose_parts') or []
        ph = ','.join('%s:%s' % (k, v) for d in parts for k, v in d.items() if 'hand' in k or 'foot' in k) or '-'
        print('%-16s %-8s %-8s %-8s %s' % (r['image'], r.get('person_n'), r.get('hand_n'),
                                           r.get('pose_people'), ph))
    print('\n裁出待判手部小图 %d 张 → %s' % (len(manifest), CROPS))


if __name__ == '__main__':
    main()
