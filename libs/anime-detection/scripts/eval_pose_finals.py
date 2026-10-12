"""在**交付图**上评一个 pose 权重（人数检出），用来选产线权重。

为什么单列一个脚本：README 里那条教训（val 上更好 ≠ 交付图上更好）需要**每次都按同一口径量交付图**，
不能再靠截图目测。用法：
    python scripts/eval_pose_finals.py --weights runs/m1_pose_coco17/weights/best.pt [...]
"""
import argparse
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FINALS = r"<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--src', default=FINALS)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.3)
    a = ap.parse_args()
    from ultralytics import YOLO
    files = sorted(glob.glob(os.path.join(a.src, '*.png')))
    print('交付图 %d 张，imgsz=%d conf=%.2f' % (len(files), a.imgsz, a.conf))
    for w in a.weights:
        m = YOLO(w)
        cnt = []
        for f in files:
            r = m.predict(f, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            cnt.append(len(r.keypoints or []))
        hit = sum(1 for c in cnt if c > 0)
        print('%-46s 检出人 %d/%d  逐图=%s' % (os.path.basename(os.path.dirname(os.path.dirname(w))),
                                             hit, len(files), cnt))


if __name__ == '__main__':
    main()
