"""在**人工真值**上比姿态模型：半监督臂 / 对照臂 / 现役基线。

口径（与面点 A/B 一致，便于横向比较）：
  * 基准 = `data/pose_gt/<stem>.json` 的人工点；`--split val` 用留出集（从不参与训练），
    `--split train` 用人工训练集 —— 后者是**记忆检查**（半监督臂若只在 train 上好看，就是背下来了）；
  * 预测实例按"与真值框 IoU 最大"一对一配对（全 0 则中心最近）；
  * 只比真值 vis>=1 的关节；预测该关节没出（conf<0.2）算**漏检**，单独报覆盖率
    —— 否则"少预测就少犯错"，误差会被系统性美化；
  * 报：覆盖率、平均/中位/p90 像素误差、>60px 占比、命中率（≤5% 图高）、**逐图误差中位**、逐关节中位。

用法：
  python scripts/pose_eval_vs_human.py --weights A.pt B.pt C.pt --split val
  python scripts/pose_eval_vs_human.py --weights A.pt B.pt C.pt --split train --json .runtime/semi_train.json
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
from export_labels import COCO17_NAMES                                    # noqa: E402

C2B = {'nose': 'nose', 'left_eye': 'l_eye', 'right_eye': 'r_eye', 'left_ear': 'l_ear', 'right_ear': 'r_ear',
       'left_shoulder': 'l_shoulder', 'right_shoulder': 'r_shoulder', 'left_elbow': 'l_elbow',
       'right_elbow': 'r_elbow', 'left_wrist': 'l_wrist', 'right_wrist': 'r_wrist',
       'left_hip': 'l_hip', 'right_hip': 'r_hip', 'left_knee': 'l_knee', 'right_knee': 'r_knee',
       'left_ankle': 'l_ankle', 'right_ankle': 'r_ankle'}
SCORE = 0.2


def gt_instances(stem):
    """人工真值可能有多个实例（多人物）——全部返回，各自带自己的点集与框。"""
    p = os.path.join(PROJ, 'data', 'pose_gt', stem + '.json')
    if not os.path.exists(p):
        return []
    g = json.load(io.open(p, encoding='utf-8'))
    by = {}
    for q in g['points']:
        if q.get('cls') == 'body':
            by.setdefault(q['chain'], {})[q['name'].split('.')[-1]] = (q['x'], q['y'], q.get('vis', 0))
    out = []
    iw, ih = [int(v) for v in g['size'].split('x')]
    for ch, pts in by.items():
        vis = [v for v in pts.values() if v[2] >= 1]
        if len(vis) < 2:
            continue
        xs, ys = [v[0] for v in vis], [v[1] for v in vis]
        out.append({'pts': pts, 'box': (min(xs), min(ys), max(xs), max(ys)),
                    'size': (iw, ih), 'src': g.get('src'), 'stem': stem, 'chain': ch})
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def predict(model, img):
    r = model(img, imgsz=768, conf=0.20, verbose=False)[0]
    out = []
    if r.keypoints is None or r.keypoints.xy is None:
        return out
    xy = r.keypoints.xy.cpu().numpy()
    cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
    bx = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((len(xy), 4))
    for i in range(len(xy)):
        d = {C2B[nm]: (float(xy[i][j][0]), float(xy[i][j][1]))
             for j, nm in enumerate(COCO17_NAMES) if cf[i][j] >= SCORE}
        if len(d) >= 2:
            out.append({'pts': d, 'box': tuple(float(v) for v in bx[i])})
    return out


def run_weight(model, gts):
    per_img, joint, cov_joint = {}, {}, {}
    n_gt = n_pred = 0
    for g in gts:
        preds = predict(model, g['src'])
        p = None
        if preds:
            best, bi = 0.0, None
            for k, q in enumerate(preds):
                v = iou(g['box'], q['box'])
                if v > best:
                    best, bi = v, k
            if bi is None:
                gc = ((g['box'][0] + g['box'][2]) / 2, (g['box'][1] + g['box'][3]) / 2)
                dd = [(np.hypot((q['box'][0] + q['box'][2]) / 2 - gc[0],
                                (q['box'][1] + q['box'][3]) / 2 - gc[1]), k) for k, q in enumerate(preds)]
                bi = min(dd)[1]
            p = preds[bi]
        errs = []
        for nm, (x, y, v) in g['pts'].items():
            if v < 1:
                continue
            n_gt += 1
            cov_joint[nm] = cov_joint.get(nm, [0, 0])
            cov_joint[nm][1] += 1
            if p is None or nm not in p['pts']:
                continue
            q = p['pts'][nm]
            d = float(np.hypot(x - q[0], y - q[1]))
            errs.append(d)
            joint.setdefault(nm, []).append(d)
            cov_joint[nm][0] += 1
            n_pred += 1
        if errs:
            per_img[g['stem'] + '#' + g['chain']] = float(np.median(errs))
    return {'per_img': per_img, 'joint': joint, 'cov_joint': cov_joint, 'n_gt': n_gt, 'n_pred': n_pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--manifest', default=os.path.join(PROJ, 'data', 'ds_pose17_semi', 'semi_manifest.json'))
    ap.add_argument('--split', choices=['val', 'train'], default='val')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    man = json.load(io.open(a.manifest, encoding='utf-8'))
    if a.split == 'val':
        stems = man['val_stems']
    else:
        # 半监督训练列表里出现 K 次的就是人工图（权重用重采样实现）—— 这是"当时到底训练了哪 54 张"的
        # 唯一可靠还原：pose_gt 会随着人继续标注而变大，不能拿"现在的人工真值 − 留出集"当训练集。
        tp = os.path.join(os.path.dirname(a.manifest), 'train_semi.txt')
        cnt = {}
        for line in io.open(tp, encoding='utf-8'):
            line = line.strip()
            if line:
                cnt[line] = cnt.get(line, 0) + 1
        reps = sorted(k for k, v in cnt.items() if v >= 2)
        stems = sorted(os.path.splitext(os.path.basename(k))[0] for k in reps)
        print('从 train_semi.txt 还原人工训练图：%d 张（重复 %d 次）；训练列表共 %d 行 / %d 张不同图'
              % (len(stems), max(cnt.values()) if cnt else 0, sum(cnt.values()), len(cnt)))
        if len(stems) != man.get('human_train'):
            print('  ⚠ 与 manifest 记的 human_train=%s 不一致' % man.get('human_train'))
    gts = [g for s in stems for g in gt_instances(s)]
    print('人工真值（%s 集）：%d 张图 / %d 个实例 / %d 个 vis>=1 关节'
          % (a.split, len(stems), len(gts), sum(1 for g in gts for v in g['pts'].values() if v[2] >= 1)))
    if a.split == 'train':
        print('  注意 train 集是半监督臂的训练图，这里看的是"有没有背下来"')
    from ultralytics import YOLO
    H = float(np.median([g['size'][1] for g in gts]))
    res = {}
    print('\n%-26s %7s %7s %7s %7s %7s %8s' %
          ('模型', '覆盖', '平均', '中位', 'p90', '>60px', '命中率'))
    for w in a.weights:
        if not os.path.exists(w):
            print('%-26s 缺权重 %s' % ('?', w))
            continue
        name = os.path.basename(os.path.dirname(os.path.dirname(w)))
        r = run_weight(YOLO(w), gts)
        allerr = np.array([d for v in r['joint'].values() for d in v])
        perimg = np.array(list(r['per_img'].values()))
        cov = 100.0 * r['n_pred'] / max(1, r['n_gt'])
        if len(allerr):
            print('%-26s %6.0f%% %7.1f %7.1f %7.1f %6.0f%% %7.0f%%'
                  % (name[:26], cov, allerr.mean(), np.median(allerr),
                     np.percentile(allerr, 90), 100 * (allerr > 60).mean(),
                     100 * (allerr <= 0.05 * H).mean()))
        else:
            print('%-26s %6.0f%% 没有可比对关节' % (name[:26], cov))
        if len(perimg):
            print('%-26s   逐图误差中位：中位 %.1f / 平均 %.1f（%d 个实例）'
                  % ('', np.median(perimg), perimg.mean(), len(perimg)))
        res[name] = {'coverage': cov, 'n_gt': r['n_gt'], 'n_pred': r['n_pred'],
                     'mean': float(allerr.mean()) if len(allerr) else None,
                     'median': float(np.median(allerr)) if len(allerr) else None,
                     'p90': float(np.percentile(allerr, 90)) if len(allerr) else None,
                     'per_image_median': float(np.median(perimg)) if len(perimg) else None,
                     'per_joint_median': {k: float(np.median(v)) for k, v in r['joint'].items()}}
    # 逐关节对照表
    names = [n for n in res if res[n]['per_joint_median']]
    if len(names) >= 1:
        print('\n逐关节中位像素误差：')
        print('%-13s %s' % ('关节', ' '.join('%9s' % n[:9] for n in names)))
        for j in COCO17_NAMES:
            b = C2B[j]
            row = ['%9.1f' % res[n]['per_joint_median'].get(b, float('nan')) for n in names]
            print('%-13s %s' % (b, ' '.join(row)))
    if a.json:
        io.open(a.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps({'split': a.split, 'h': H, 'result': res}, ensure_ascii=False, indent=1))
        print('\n写入 %s' % a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
