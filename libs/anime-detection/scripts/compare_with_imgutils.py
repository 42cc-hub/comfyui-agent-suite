"""我们自己的权重 vs 现成件（imgutils）在**同一批图**上的对比（目标②要求"可对比指标"）。

两个后端要在各自的 conda 环境里跑（本机两个环境各有半边）：
    python    compare_with_imgutils.py --backend ours
    python compare_with_imgutils.py --backend imgutils
各写一份 CSV 到 `.runtime/compare/`，然后手动并排看（或用 --show 把两份都打出来）。

对比口径（只看**能力**，不看 mAP）：
  · 检出人数（pose 层）        · 手框数 / 头框数（det 层）
  · 每张耗时
用法：python scripts/compare_with_imgutils.py --src <目录> --backend ours|imgutils
"""
import argparse
import csv
import io
import os
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, '.runtime', 'compare')
DEF_SRC = r"<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEF_SRC)
    ap.add_argument('--backend', required=True, choices=['ours', 'imgutils'])
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = [os.path.join(a.src, f) for f in sorted(os.listdir(a.src))
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if a.backend == 'ours':
        from ultralytics import YOLO
        pose = YOLO(os.path.join(PROJ, 'runs', 'm1_pose_coco17', 'weights', 'best.pt'))
        det = YOLO(os.path.join(PROJ, 'runs', 'm2a_det_996img_960', 'weights', 'best.pt'))
        for m in (pose, det):
            if getattr(m, 'task', None) == 'det':
                m.task = 'detect'
        rows = []
        for f in files:
            t0 = time.time()
            rp = pose.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            rd = det.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            cls = [rd.names[int(b.cls[0])] for b in (rd.boxes or [])]
            rows.append({'image': os.path.basename(f), 'backend': 'ours',
                         'people': len(rp.keypoints or []),
                         'hand': cls.count('hand'), 'head': cls.count('head'),
                         'foot': cls.count('foot'), 'person': cls.count('person'),
                         'sec': round(time.time() - t0, 1)})
    else:
        from PIL import Image
        from imgutils.detect import detect_hands, detect_heads, detect_person
        from imgutils.pose import dwpose_estimate
        rows = []
        for f in files:
            img = Image.open(f).convert('RGB')
            t0 = time.time()
            try:
                people = len(dwpose_estimate(img))
            except Exception:
                people = -1
            try:
                nh = len(detect_hands(img))
            except Exception:
                nh = -1
            try:
                nhd = len(detect_heads(img))
            except Exception:
                nhd = -1
            try:
                np_ = len(detect_person(img))
            except Exception:
                np_ = -1
            rows.append({'image': os.path.basename(f), 'backend': 'imgutils',
                         'people': people, 'hand': nh, 'head': nhd, 'foot': '-', 'person': np_,
                         'sec': round(time.time() - t0, 1)})

    p = os.path.join(OUT, 'compare_%s.csv' % a.backend)
    with io.open(p, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['image', 'backend', 'people', 'person', 'hand', 'foot',
                                           'head', 'sec'])
        w.writeheader()
        w.writerows(rows)
    print('%-40s %-6s %-6s %-6s %-6s %s' % ('image', 'people', 'person', 'hand', 'foot', 'head'))
    for r in rows:
        print('%-40s %-6s %-6s %-6s %-6s %s' % (r['image'][:40], r['people'], r['person'],
                                                r['hand'], r['foot'], r['head']))
    print('\n→ %s' % p)


if __name__ == '__main__':
    main()
