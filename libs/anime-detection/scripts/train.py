"""统一训练入口（ultralytics YOLO）。三个任务共用一套参数与记录方式。

6 GB 显存的保守配方（见 docs/training.md）：
  model=yolo11n-*.pt（从 COCO 预训练迁移）、imgsz=640→768、batch=-1（框架按 ~60% 显存自测、OOM 自动减半）、
  amp=True、cache=False、workers=2；**从 COCO 权重迁移**比从零训收敛快得多。

用法：
  python scripts/train.py --task det  --data data/ds_det/det_body.yaml   --model yolo11n.pt       --epochs 100
  python scripts/train.py --task pose --data data/ds_pose/pose_body.yaml --model yolo11n-pose.pt  --epochs 150
  python scripts/train.py --task cls  --data data/cls                    --model yolo11n-cls.pt   --epochs 80
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RUNS = os.path.join(PROJ, 'runs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['det', 'pose', 'cls'], required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', default='-1', help='-1 = 自动按显存；正整数 = 固定 batch')
    ap.add_argument('--name', default=None)
    ap.add_argument('--seed', type=int, default=20261010,
                    help='随机种子。做消融时**同配置跑 ≥3 个种子**再报 mean±std —— 实测单次运行方差可达 ±0.026 F1')
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--resume', action='store_true')
    # ---- 数据增强（2026-10-13 用户提的关键点）----
    ap.add_argument('--aug', choices=['default', 'face'], default='default',
                    help='增广预设。`face` = **关掉 mosaic/切片**（用户指出：二次元的嘴常常就是"一条线"，'
                         '切片拼接后不可识别 ⇒ 污染训练），同时 erasing 归零（随机擦除可能正好盖住眼/嘴）、'
                         '缩放与平移调小；`default` = ultralytics 默认（mosaic=1.0, erasing=0.4, scale=0.5…）')
    for _k, _t in (('mosaic', float), ('close_mosaic', int), ('scale', float), ('translate', float),
                   ('degrees', float), ('fliplr', float), ('flipud', float), ('erasing', float),
                   ('mixup', float),
                   # 微调用（2026-10-14 加）：从已训好的权重做"修一处"的微调时，**默认 lr0=0.01 太猛**，
                   # 会把旧权重整体推走（实测：val 全面掉分、最佳轮=epoch 1）。微调建议 lr0 5e-4 ~ 1e-4，
                   # 或者 freeze 前 N 层只训 head。
                   ('lr0', float), ('lrf', float), ('warmup_epochs', float), ('freeze', int)):
        ap.add_argument('--%s' % _k.replace('_', '-'), dest=_k, type=_t, default=None,
                        help='覆盖增强预设（不填则用预设值）')
    a = ap.parse_args()

    # 预设 → 实际传给 ultralytics 的增强参数（键名与官方 cfg 一致；cls 任务用不到的键会被忽略）
    aug = {'mosaic': 1.0, 'close_mosaic': 10, 'scale': 0.5, 'translate': 0.1,
           'degrees': 0.0, 'fliplr': 0.5, 'flipud': 0.0, 'erasing': 0.4, 'mixup': 0.0}
    if a.aug == 'face':
        aug.update(dict(mosaic=0.0, close_mosaic=0, scale=0.25, translate=0.05, erasing=0.0))
    for k in list(aug):
        v = getattr(a, k)
        if v is not None:
            aug[k] = v
    print('[aug] 预设=%s → %s' % (a.aug, ', '.join('%s=%s' % (k, aug[k]) for k in sorted(aug))), flush=True)

    model = a.model or {'det': 'yolo11n.pt', 'pose': 'yolo11n-pose.pt', 'cls': 'yolo11n-cls.pt'}[a.task]
    # 坑：ultralytics 训练时接受短名 'det'，但会把 checkpoint 的 task 存成 'det'，
    # 之后 predict/val 只认 'detect' ⇒ 训练时就把名字规范成官方全名。
    task_full = {'det': 'detect', 'pose': 'pose', 'cls': 'classify'}[a.task]
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('还没装 ultralytics：用 sdenv 或 me2 环境跑（实测这两个都有 ultralytics 8.4.x；'
                         'anime_det 里**没有** ultralytics，见 docs/training.md）')

    # batch 要传 int 给 DataLoader（float 会报 "batch_size should be a positive integer"）
    batch = -1 if str(a.batch).strip() in ('-1', 'auto') else int(float(a.batch))

    YOLO(model).train(
        task=task_full, data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=batch,
        amp=True, cache=False, workers=2, pretrained=True, resume=a.resume,
        project=RUNS, name=a.name or ('%s_%s' % (a.task, os.path.splitext(os.path.basename(a.data))[0])),
        exist_ok=True, patience=a.patience, seed=a.seed, deterministic=True,
        **aug)


if __name__ == '__main__':
    main()
