"""自训练刷新：用当前最好的权重，把**自动池**（没有人工真值的那 927 张）的预标注重新预测一遍。

动机：自动池的标签是 dwpose 的输出（体点尚可、面点没有），它既是"蒸馏老师"也是噪声源。
既然我们现在有了一个在人工真值上训过的强模型，就可以用它当**新老师**重打标签（self-training），
再训一轮 —— 这是半监督里还没试过的杠杆。

产物与既有 prelabel **同 schema**（points/boxes/chain/src），所以 `--pred-dir`/导出/评测都能直接用。

用法：python scripts/refresh_autolabels.py --model models/vitpose_ft_v3_h5auto --out data/prelabel_vitpose_selftrain
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export_labels import COCO17_NAMES                                        # noqa: E402

PRE = os.path.join(PROJ, 'data', 'prelabel_face_full')
GT = os.path.join(PROJ, 'data', 'pose_gt')
C2B = {'nose': 'nose', 'left_eye': 'l_eye', 'right_eye': 'r_eye', 'left_ear': 'l_ear', 'right_ear': 'r_ear',
       'left_shoulder': 'l_shoulder', 'right_shoulder': 'r_shoulder', 'left_elbow': 'l_elbow',
       'right_elbow': 'r_elbow', 'left_wrist': 'l_wrist', 'right_wrist': 'r_wrist',
       'left_hip': 'l_hip', 'right_hip': 'r_hip', 'left_knee': 'l_knee', 'right_knee': 'r_knee',
       'left_ankle': 'l_ankle', 'right_ankle': 'r_ankle'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--score-visible', type=float, default=0.5)
    ap.add_argument('--score-occluded', type=float, default=0.20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from PIL import Image
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    proc = VitPoseImageProcessor.from_pretrained(a.model)
    model = VitPoseForPoseEstimation.from_pretrained(a.model).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    files = [f for f in sorted(os.listdir(PRE)) if f.endswith('.json')
             and not os.path.exists(os.path.join(GT, f))]
    if a.limit:
        files = files[:a.limit]
    n_img = n_pt = 0
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
                for j, cname in enumerate(COCO17_NAMES):
                    s = float(sc[j])
                    if s < a.score_occluded:
                        continue
                    points.append({'layer': 'M1', 'name': 'body.' + C2B[cname], 'cls': 'body',
                                   'x': round(float(kp[j][0]), 1), 'y': round(float(kp[j][1]), 1),
                                   'prob': round(s, 3), 'vis': 2 if s >= a.score_visible else 1,
                                   'chain': 'M1.self#%d' % i, 'src': 'self-train'})
                    n_pt += 1
        out_rec = dict(rec)
        out_rec['points'] = [q for q in (rec.get('points') or []) if q.get('cls') != 'body'] + points
        out_rec['selftrain_prelabel'] = {'model': os.path.basename(os.path.normpath(a.model)),
                                        'persons': len(boxes), 'points': len(points)}
        io.open(os.path.join(a.out, fn), 'w', encoding='utf-8', newline='\n').write(
            json.dumps(out_rec, ensure_ascii=False, indent=1))
        n_img += 1
        if n_img % 100 == 0:
            print('  已刷 %d 张 / %d 个点' % (n_img, n_pt), flush=True)
    print('自训练标签写出 %d 张 → %s（共 %d 个点）' % (n_img, a.out, n_pt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
