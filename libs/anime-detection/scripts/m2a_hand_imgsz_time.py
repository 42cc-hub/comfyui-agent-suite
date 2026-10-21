# -*- coding: utf-8 -*-
"""`--hand-imgsz` opt-in 的真实成本：20 张 GT 图，四档配置。

四档（全部靠命令行/权重驱动，不改 locate.py 默认值）：
  A 默认产线      `locate.py` 不带 `--hand-weights`（手模型关，imgsz 1280）
  B 手开@1280     `--hand-weights m2a_hand_v3`（手模型开，仍跟随 --imgsz 1280）
  C 手开@1536     `--hand-weights m2a_hand_v3 --hand-imgsz 1536`（新开关）
  D 手双跑        B 与 C 的手那部分都跑（1280 + 1536 各一次）

两类数：
  1) 端到端：每档 `locate.py --src <20 张目录>` 跑一次，量整批墙钟；单图 = 整批/20（模型每批只加载一次，
     生产就是每批加载一次），同时每 0.4 s 采一次 `nvidia-smi` 记**峰值显存 − 空闲**。
  2) 逐图：一个进程内加载一次模型，对 20 张图逐张计时（det 1280 + foot 1280 + 手 1280/1536），
     给**单图墙钟中位数与最大值**，把加载成本从单图成本里剥掉；B/C/D 相对 A 的倍数也由此给出。
"""
import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
PY = sys.executable
LOCATE = os.path.join(HERE, 'locate.py')
W_DET = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
W_FOOT = os.path.join(SUITE, 'runs', 'm2a_foot_v1', 'weights', 'best.pt')
W_HAND = os.path.join(SUITE, 'runs', 'm2a_hand_v3', 'weights', 'best.pt')
OUT_DEFAULT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'hand_imgsz_time')


def vram_used():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, stop):
        super().__init__(daemon=True)
        self.stop_ev, self.peak, self.n = stop, 0.0, 0

    def run(self):
        while not self.stop_ev.is_set():
            v = vram_used()
            if v is not None:
                self.peak = max(self.peak, v)
                self.n += 1
            time.sleep(0.4)


def run_locate(src, out, extra):
    cmd = [PY, LOCATE, '--src', src, '--out', out] + extra
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    idle = vram_used()
    stop = threading.Event()
    s = Sampler(stop)
    t0 = time.time()
    s.start()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    dt = time.time() - t0
    stop.set()
    s.join(timeout=3)
    peak = (s.peak - idle) if (idle is not None and s.n) else None
    return {'extra': ' '.join(extra) if extra else '(默认)', 'batch_wall_s': round(dt, 2),
            'per_img_s': None, 'rc': p.returncode,
            'peak_delta_mb': None if peak is None else round(peak, 0),
            'tail': (p.stdout or '')[-200:] if p.returncode == 0 else (p.stderr or p.stdout or '')[-500:]}


def per_image_timing(images, rounds=1):
    """进程内一次加载，逐图计时；返回各档单图秒数（含 predict，不含加载）。"""
    import torch
    from ultralytics import YOLO
    det, hand = YOLO(W_DET), YOLO(W_HAND)
    foot = YOLO(W_FOOT) if os.path.exists(W_FOOT) else None
    for m in (det, foot, hand):
        if m is not None and getattr(m, 'task', None) == 'det':
            m.task = 'detect'
    keys = ('A_default', 'B_hand1280', 'C_hand1536', 'D_hand_both', 'hand1280', 'hand1536', 'det1280', 'foot1280')
    times = {k: [] for k in keys}
    for _ in range(rounds):
        for f in images:
            t0 = time.time()
            det.predict(f, conf=0.3, imgsz=1280, verbose=False, device=0)
            t1 = time.time()
            if foot is not None:
                foot.predict(f, conf=0.15, imgsz=1280, verbose=False, device=0)
            t2 = time.time()
            hand.predict(f, conf=0.15, imgsz=1280, verbose=False, device=0)
            t3 = time.time()
            hand.predict(f, conf=0.15, imgsz=1536, verbose=False, device=0)
            t4 = time.time()
            times['det1280'].append(t1 - t0)
            times['foot1280'].append(t2 - t1)
            times['hand1280'].append(t3 - t2)
            times['hand1536'].append(t4 - t3)
            times['A_default'].append(t2 - t0)
            times['B_hand1280'].append(t3 - t0)
            times['C_hand1536'].append(t4 - t0)
            times['D_hand_both'].append((t4 - t0) + (t3 - t2))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    stats = {}
    for k, v in times.items():
        if not v:
            continue
        stats[k] = {'n': len(v), 'median_s': round(statistics.median(v), 3),
                    'max_s': round(max(v), 3), 'mean_s': round(sum(v) / len(v), 3)}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(SUITE, 'data', 'hand_gt', 'images'))
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--skip-end2end', action='store_true')
    a = ap.parse_args()

    imgs = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                  if f.lower().endswith(('.png', '.jpg', '.webp')))
    n = len(imgs)
    os.makedirs(a.out, exist_ok=True)
    print('图 %d 张' % n)

    end2end = {}
    if not a.skip_end2end:
        for name, extra in (('A_default', []),
                            ('B_hand1280', ['--hand-weights', W_HAND]),
                            ('C_hand1536', ['--hand-weights', W_HAND, '--hand-imgsz', '1536'])):
            r = run_locate(a.images, os.path.join(a.out, name), extra)
            r['per_img_s'] = round(r['batch_wall_s'] / n, 3) if n else None
            end2end[name] = r
            print('%-11s 整批 %7.2fs  单图均 %6.3fs  峰值ΔVRAM %s MB  rc=%d' % (
                name, r['batch_wall_s'], r['per_img_s'] or -1, r['peak_delta_mb'], r['rc']), flush=True)

    per_img = per_image_timing(imgs, a.rounds)
    print('逐图（中位/最大，秒）：')
    for k, v in per_img.items():
        print('  %-11s %6.3f / %6.3f' % (k, v['median_s'], v['max_s']))

    ratio = {'hand_1536_over_1280': round(per_img['hand1536']['median_s'] / per_img['hand1280']['median_s'], 2)
             if per_img.get('hand1280') else None}
    if per_img.get('A_default'):
        for tag, key in (('B_over_A_total', 'B_hand1280'), ('C_over_A_total', 'C_hand1536'),
                         ('D_over_A_total', 'D_hand_both')):
            ratio[tag] = round(per_img[key]['median_s'] / per_img['A_default']['median_s'], 2)
    report = {'images': n, 'end2end': end2end, 'per_image': per_img, 'ratio': ratio,
              'notes': ['end2end.per_img_s = 整批墙钟 / 张数（每批只加载一次模型）',
                        'per_image = 进程内加载一次后逐图计时（不含加载），中位/最大是 20 张图的分布',
                        'peak_delta_mb = 运行中 nvidia-smi 采样峰值 − 启动前空闲；共享单例 ComfyUI 的占用会污染该值',
                        'B = 手模型开仍 1280；C = 手头 1536；D = 手双跑（1280+1536）；三者与 A 的差只来自手那一路']}
    with io.open(os.path.join(a.out, 'timing.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print('倍数', json.dumps(ratio, ensure_ascii=False))
    print('wrote', os.path.join(a.out, 'timing.json'))


if __name__ == '__main__':
    main()
