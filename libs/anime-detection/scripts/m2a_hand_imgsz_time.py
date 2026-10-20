# -*- coding: utf-8 -*-
"""量 `--hand-imgsz` opt-in 的真实成本：三档配置跑同 20 张 GT 图，报**单图墙钟秒数中位数 + 峰值显存**。

三档（都不改任何默认值，全部通过命令行参数驱动）：
  A base1280   默认产线：不带 `--hand-weights`（专用手模型关）
  B hand1280   `--hand-weights m2a_hand_v3`（手模型开，但仍 1280）
  C hand1536   `--hand-weights m2a_hand_v3 --hand-imgsz 1536`（新开关）

口径：
  * 每档跑 `locate.py --src <图目录>` 一次，量**整批墙钟**；单图 = 墙钟 / 张数（含一次性模型加载，生产就是每批加载一次）。
  * 另跑"单图重复" 3 次量进程启动+加载开销，把 predict 与加载拆开，避免把加载成本算成手模型成本。
  * 峰值显存：跑的同时每 0.4 s 采一次 `nvidia-smi --query-gpu=memory.used`，报**峰值 − 空闲**（MB）。
  * 不删不改 locate.py 的产物目录：全部落在 --out 下。
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
W_HAND = os.path.join(SUITE, 'runs', 'm2a_hand_v3', 'weights', 'best.pt')
LOCATE = os.path.join(HERE, 'locate.py')


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
        self.stop_ev = stop
        self.peak = 0.0
        self.n = 0

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
    return {'cmd': ' '.join(extra) if extra else '(默认)', 'wall': round(dt, 2),
            'rc': p.returncode, 'peak_delta_mb': None if peak is None else round(peak, 0),
            'tail': (p.stdout or '')[-300:] if p.returncode == 0 else (p.stderr or p.stdout or '')[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(SUITE, 'data', 'hand_gt', 'images'))
    ap.add_argument('--out', default=os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'hand_imgsz_time'))
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张（0=全部）')
    ap.add_argument('--skip-load-probe', action='store_true')
    a = ap.parse_args()

    imgs = sorted(f for f in os.listdir(a.images) if f.lower().endswith(('.png', '.jpg', '.webp')))
    if a.limit:
        imgs = imgs[:a.limit]
    n = len(imgs)
    os.makedirs(a.out, exist_ok=True)
    print('图 %d 张：%s' % (n, a.images))

    cfgs = [
        ('A_base1280', []),
        ('B_hand1280', ['--hand-weights', W_HAND]),
        ('C_hand1536', ['--hand-weights', W_HAND, '--hand-imgsz', '1536']),
    ]
    res = []
    for name, extra in cfgs:
        r = run_locate(a.images, os.path.join(a.out, name), extra)
        r['name'] = name
        r['per_img'] = round(r['wall'] / n, 3) if n else None
        res.append(r)
        print('%-12s wall %7.2fs  per_img %6.3fs  峰值ΔVRAM %s MB  rc=%d' % (
            name, r['wall'], r['per_img'] or -1, r['peak_delta_mb'], r['rc']), flush=True)
        if r['rc'] != 0:
            print('  !! rc!=0 tail:', r['tail'])

    load = None
    if not a.skip_load_probe:
        # 单图 ×3：进程启动 + 模型加载占大头，用它把"加载成本"从单图成本里剥出来
        walls = []
        for k in range(3):
            extra = ['--hand-weights', W_HAND, '--hand-imgsz', '1536']
            cmd = [PY, LOCATE, '--src', os.path.join(a.images, imgs[0]), '--out',
                   os.path.join(a.out, 'load_probe_%d' % k)] + extra
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            t0 = time.time()
            subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
            walls.append(time.time() - t0)
        load = {'runs': [round(w, 2) for w in walls], 'median': round(statistics.median(walls), 2)}
        print('单图×3 墙钟（含加载）', load['runs'], '中位', load['median'])

    report = {'images': n, 'configs': res, 'single_image': load,
              'notes': ['per_img = 整批墙钟 / 张数（含每批一次模型加载）',
                        '峰值ΔVRAM = 运行中 nvidia-smi 采样峰值 − 启动前空闲值，含共享单例 ComfyUI 的占用误差',
                        'B vs A 的差 = 专用手模型多跑一次推理；C vs B = 手那一路 1280→1536 的增量']}
    with io.open(os.path.join(a.out, 'timing.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print('wrote', os.path.join(a.out, 'timing.json'))


if __name__ == '__main__':
    main()
