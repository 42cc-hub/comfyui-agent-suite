# -*- coding: utf-8 -*-
"""端到端（`locate.py` 真实输出）对 `data/hand_gt` 手框打分。

与 `m2a_hand2pass_score.py`（模型直跑）不同，这里只读 locate.py 落盘的 JSON，
用来闭掉"接线路径 vs 评分口径是否有差"这条环。conf 0.30 / IoU 0.5 贪心一对一。
"""
import argparse
import io
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m2a_hand2pass_score as H


def load_pred_dir(d, cls='hand', conf=0.30):
    """→ {crop_name: [{'box': [x0,y0,x1,y1], 'conf': ..}]}"""
    out, files = {}, 0
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        files += 1
        rec = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        boxes = rec.get('boxes_all') or rec.get('boxes') or []
        preds = []
        for b in boxes:
            if b.get('cls') != cls:
                continue
            sc = float(b.get('prob', b.get('conf', 0)))
            if sc < conf:
                continue
            xy = b.get('box') or b.get('xyxy') or b.get('bbox')
            if not xy or len(xy) != 4:
                print('  ⚠ 无法定位坐标键：%s keys=%s' % (f, sorted(b)))
                continue
            preds.append({'box': [float(v) for v in xy], 'conf': sc})
        crop = os.path.splitext(f)[0] + '.png'
        out[crop] = preds
    return out, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--gt', default=os.path.join(SUITE, 'data', 'hand_gt'))
    ap.add_argument('--conf', type=float, default=0.30)
    a = ap.parse_args()
    gt_map, gt_boxes = H.load_gt(a.gt)
    labels = a.labels or [os.path.basename(os.path.normpath(x)) for x in a.dirs]
    print('GT: %d 张（hand 行）、%d 个手框、conf %.2f / IoU 0.5' % (len(gt_map), gt_boxes, a.conf))
    for d, lab in zip(a.dirs, labels):
        pred_map, nf = load_pred_dir(d, 'hand', a.conf)
        miss = [c for c in gt_map if c not in pred_map]
        res = H.score(gt_map, pred_map, 0.5)
        hp = sum(len(v) for v in pred_map.values())
        print('%-22s JSON %2d 张 预测手框 %3d  TP %2d FN %2d FP %2d  R %5.1f  P %5.1f  medIoU %.3f  全中 %d/%d%s'
              % (lab, nf, hp, res['tp'], res['fn'], res['fp'],
                 res['recall50'], res['precision50'], res['median_iou'],
                 res['img_full'], res['n_img'], ('  ⚠ 缺 JSON %d 张' % len(miss)) if miss else ''))


if __name__ == '__main__':
    main()
