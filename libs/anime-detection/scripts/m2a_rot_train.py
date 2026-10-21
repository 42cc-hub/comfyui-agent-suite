# -*- coding: utf-8 -*-
"""task-10 训练入口：复用 det_v2 的 ultralytics 训练配方（data/ds_det_fix/det_body_fix.yaml 同款 4 类 + imgsz 960），
只换数据 yaml / run 名 / 时间上限。不使用 val（评测统一在 holdout 上做）。

用法：python m2a_rot_train.py --run rot --time 0.33
      （--time 单位小时，ultralytics 的 time 参数同样按小时）
"""
import argparse
import os

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(SUITE, 'data')
RUNS = os.path.join(SUITE, 'runs')
INIT = os.path.join(RUNS, 'm2a_det_fix_v2', 'weights', 'best.pt')
YAML = {
    'rot': os.path.join(DATA, 'ds_det_rot', 'det_rot.yaml'),
    'rot_semi': os.path.join(DATA, 'ds_det_rot', 'det_rot_semi.yaml'),
    'semi': os.path.join(DATA, 'ds_det_rot', 'det_semi.yaml'),
    'rot_bal': os.path.join(DATA, 'ds_det_rot', 'det_rot_bal.yaml'),
}
NAME = {'rot': 'm2a_det_rot_only', 'rot_semi': 'm2a_det_rot_semi', 'semi': 'm2a_det_semi_only',
        'rot_bal': 'm2a_det_rot_bal'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', choices=['rot', 'rot_semi', 'semi', 'rot_bal'], required=True)
    ap.add_argument('--time', type=float, default=0.33)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr0', type=float, default=0.004)
    ap.add_argument('--model', default=INIT)
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.model)
    m.train(
        data=YAML[a.run], epochs=a.epochs, time=a.time, patience=50,
        imgsz=960, batch=a.batch, workers=4, device='',
        project=RUNS, name=a.name or NAME[a.run], exist_ok=True,
        val=False, plots=False, seed=20261010, deterministic=True,
        pretrained=True, optimizer='auto', lr0=a.lr0, cos_lr=False,
        close_mosaic=10, amp=True, cache=False, verbose=True,
    )


if __name__ == '__main__':
    main()
