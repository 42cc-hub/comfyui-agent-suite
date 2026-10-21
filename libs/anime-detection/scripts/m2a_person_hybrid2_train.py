"""hybrid2 训练：24 epochs 写死、无 time 参数。窗口紧时由外部 watcher 提前停止。"""
import os, sys, time

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
WARM = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')


def main():
    from ultralytics import YOLO
    m = YOLO(WARM)
    t0 = time.time()
    m.train(data=os.path.join(D, 'det_person_hybrid2.yaml'), epochs=24, patience=40,
            imgsz=960, batch=8, workers=4, device='', project=os.path.join(SUITE, 'runs'),
            name='m2a_det_person_hybrid2', exist_ok=True, val=False, plots=False,
            seed=20261024, deterministic=True, pretrained=True, optimizer='auto',
            lr0=0.004, cos_lr=False, close_mosaic=10, amp=True, cache=False)
    print('[train] done %.1f min' % ((time.time() - t0) / 60))
    best = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid2', 'weights', 'best.pt')
    print('[train] best exists=%s size=%s' % (os.path.exists(best), os.path.getsize(best) if os.path.exists(best) else -1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
