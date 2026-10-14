"""用（微调后的）ViTPose 生成预标注，作为 `prelabel_face_full` 之外的一条新预标注源。

为什么需要：现在的预标注源是 dwpose（躯干好、面部全错、覆盖率只有 ~55%）。ViTPose 微调后覆盖率高、
面点由人工监督，所以它更适合当"给人改的初稿"。产物与既有 prelabel **同 schema**（points/boxes/chain/
src），所以标注器 `--pred-dir`、`export_labels.py` 的 `ADS_PRE`、以及评测脚本都能直接吃。

约定：
  * 关节点名字用我们的 BODY18 缩写（body.l_eye 等），chain 名 `M1.vitpose#i`；
  * `prob` = ViTPose 的 score；`vis`：score>=0.5 ⇒ 2（可见）、>=0.2 ⇒ 1（遮挡/待确认）、否则 0（不可见）；
  * 人框沿用原 prelabel 的 M2a `person` 框（与训练/评测口径一致）；
  * 只写 body 的 17 点（手/脚细点不在本模型能力内），原始的人框/头框等 boxes 原样带过来。

用法：python scripts/prelabel_vitpose.py --model models/vitpose_ft_v1 --out data/prelabel_vitpose [--limit N]
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export_labels import COCO17_NAMES                                      # noqa: E402

PRE = os.path.join(PROJ, 'data', 'prelabel_face_full')
B2C = {'nose': 'nose', 'l_eye': 'left_eye', 'r_eye': 'right_eye', 'l_ear': 'left_ear', 'r_ear': 'right_ear',
       'l_shoulder': 'left_shoulder', 'r_shoulder': 'right_shoulder', 'l_elbow': 'left_elbow',
       'r_elbow': 'right_elbow', 'l_wrist': 'left_wrist', 'r_wrist': 'right_wrist',
       'l_hip': 'left_hip', 'r_hip': 'right_hip', 'l_knee': 'left_knee', 'r_knee': 'right_knee',
       'l_ankle': 'left_ankle', 'r_ankle': 'right_ankle'}
C2B = {v: k for k, v in B2C.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(PROJ, 'models', 'vitpose'))
    ap.add_argument('--out', default=os.path.join(PROJ, 'data', 'prelabel_vitpose'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', nargs='*', default=None, help='只做这些 stem')
    ap.add_argument('--score-visible', type=float, default=0.5)
    ap.add_argument('--score-occluded', type=float, default=0.2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from PIL import Image
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    proc = VitPoseImageProcessor.from_pretrained(a.model)
    model = VitPoseForPoseEstimation.from_pretrained(a.model).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    files = sorted(f for f in os.listdir(PRE) if f.endswith('.json'))
    if a.only:
        want = set(a.only)
        files = [f for f in files if f[:-5] in want]
    if a.limit:
        files = files[:a.limit]
    n_img = n_person = n_pt = 0
    for fn in files:
        rec = json.load(io.open(os.path.join(PRE, fn), encoding='utf-8'))
        boxes = [b['box'] for b in (rec.get('boxes') or []) if b.get('cls') == 'person']
        points = []
        if boxes and rec.get('src') and os.path.exists(rec['src']):
            img = Image.open(rec['src']).convert('RGB')
            with torch.no_grad():
                inputs = proc(images=img, boxes=[boxes], return_tensors='pt')
                inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
                out = model(**inputs)
            res = proc.post_process_pose_estimation(out, boxes=[boxes])
            for i, inst in enumerate(res[0]):
                kp = inst['keypoints'].cpu().numpy()
                sc = inst['scores'].cpu().numpy()
                n_person += 1
                for j, cname in enumerate(COCO17_NAMES):
                    s = float(sc[j])
                    if s < a.score_occluded:
                        continue
                    vis = 2 if s >= a.score_visible else 1
                    points.append({'layer': 'M1', 'name': 'body.' + C2B[cname], 'cls': 'body',
                                   'x': round(float(kp[j][0]), 1), 'y': round(float(kp[j][1]), 1),
                                   'prob': round(s, 3), 'vis': vis, 'chain': 'M1.vitpose#%d' % i,
                                   'src': 'ViTPose-ft'})
                    n_pt += 1
        out_rec = dict(rec)
        out_rec['points'] = [q for q in (rec.get('points') or []) if q.get('cls') != 'body'] + points
        out_rec['vitpose_prelabel'] = {'model': os.path.basename(os.path.normpath(a.model)),
                                       'persons': len(boxes), 'points': len(points)}
        io.open(os.path.join(a.out, fn), 'w', encoding='utf-8', newline='\n').write(
            json.dumps(out_rec, ensure_ascii=False, indent=1))
        n_img += 1
    print('写出 %d 张 → %s（%d 个人框 / %d 个点）' % (n_img, a.out, n_person, n_pt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
