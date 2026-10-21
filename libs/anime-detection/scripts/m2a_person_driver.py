# -*- coding: utf-8 -*-
"""阶段 B 驱动：等 GPU 票 → 训 person 专项候选 → 在 batch-2 图上评测 → 还票。

不碰 locate.py 默认值；权重只落 runs/m2a_det_person_v1。
用法：python -X utf8 scripts/m2a_person_driver.py [--now]
"""
import json
import os
import subprocess
import sys
import threading
import time

SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PY = sys.executable
CLAIM = os.path.abspath(os.path.join(SUITE, '..', '..', 'team', 'gpu_claim.py'))
REP = os.path.abspath(os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier'))
LOG = os.path.join(REP, 'person_gt2', 'driver.log')
YAML = os.path.join(SUITE, 'data', 'ds_person', 'person.yaml')
WARM = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
NAME = 'm2a_det_person_v1'
EPOCHS, TIME_H = 14, 0.30
LEVEL, TASK = 3, 'person-specialized det candidate (task-14 stage B)'
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def claim():
    for i in range(12):
        p = subprocess.run([PY, CLAIM, 'acquire', '--owner', 'm2a', '--level', str(LEVEL),
                            '--task', TASK, '--minutes', '40', '--wait', '240'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
        log('acquire#%d rc=%d %s' % (i, p.returncode, (p.stdout or p.stderr).strip()[:200]))
        if p.returncode == 0:
            return True
        time.sleep(30)
    return False


def heartbeat(stop):
    while not stop.wait(60):
        subprocess.run([PY, CLAIM, 'heartbeat', '--owner', 'm2a'], capture_output=True, env=ENV)


def release():
    subprocess.run([PY, CLAIM, 'release', '--owner', 'm2a'], capture_output=True, env=ENV)
    log('released')


def free_mb():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        return int(p.stdout.strip().splitlines()[0])
    except Exception as e:
        log('nvidia-smi 失败 %s' % e)
        return None


def external_busy():
    """用户的 ComfyUI 出图作业（anima_txt2img.py）在跑就别训：6GB 卡会被它吃掉 ~5GB。"""
    try:
        p = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                            "Select-Object -ExpandProperty CommandLine"],
                           capture_output=True, text=True, timeout=40)
        return any('anima_txt2img' in ln for ln in (p.stdout or '').splitlines())
    except Exception as e:
        log('进程检查失败 %s（按"忙"处理）' % e)
        return True


def queue_busy():
    """ComfyUI /queue：running 或 pending 非空就算忙（比看进程可靠——进程退了队列里可能还有活）。"""
    try:
        import urllib.request
        d = json.load(urllib.request.urlopen('http://127.0.0.1:7187/queue', timeout=5))
        return bool(d.get('queue_running')) or bool(d.get('queue_pending'))
    except Exception as e:
        log('queue 查询失败 %s（按"忙"处理）' % e)
        return True


def wait_idle_window(need_mb=3500, timeout_s=3600, stable_s=60):
    """等真空窗：队列空 + 无 anima 进程，连续稳定 stable_s 秒，且空闲显存 ≥ need_mb。

    ComfyUI 的池子（cudaMallocAsync）出完图不还给驱动，所以"队列真的空"之后通常要 lead 重启 7187
    才能把显存放出来；这段时间脚本只等、不动 ComfyUI。
    """
    t0 = time.time()
    idle_since = None
    while time.time() - t0 < timeout_s:
        busy = queue_busy() or external_busy()
        free = free_mb()
        if busy:
            if idle_since is not None:
                log('空窗中断（又有活）')
            idle_since = None
        else:
            if idle_since is None:
                idle_since = time.time()
                log('检测到队列空 + 无外部进程，开始计时 %ds（free=%sMiB）' % (stable_s, free))
            elif time.time() - idle_since >= stable_s and free is not None and free >= need_mb:
                log('真空窗确认 %ds（free=%dMiB ≥ %d）→ 开始训练' % (stable_s, free, need_mb))
                return True
            elif time.time() - idle_since >= stable_s:
                log('队列已空 %ds，但 free=%sMiB < %d —— 等 7187 重启放池' %
                    (stable_s, free, need_mb))
        time.sleep(20)
    return False


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if not claim():
        log('没拿到票，退出')
        return 2
    stop = threading.Event()
    th = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    th.start()
    try:
        if not wait_idle_window():
            log('被外部作业阻塞：60 分钟没等到真空窗，release 票并退出')
            return 3
        from ultralytics import YOLO
        m = YOLO(WARM)
        if getattr(m, 'task', None) == 'det':
            m.task = 'detect'
        log('开始训练 %s epochs=%d time=%.2fh' % (NAME, EPOCHS, TIME_H))
        m.train(data=YAML, epochs=EPOCHS, time=TIME_H, patience=50, imgsz=960, batch=6,
                workers=4, device='', project=os.path.join(SUITE, 'runs'), name=NAME,
                exist_ok=True, val=False, plots=False, seed=20261022, deterministic=True,
                pretrained=True, optimizer='auto', lr0=0.004, cos_lr=False, close_mosaic=10,
                amp=True, cache=False)
        best = os.path.join(SUITE, 'runs', NAME, 'weights', 'best.pt')
        log('训练完成 best=%s exists=%s' % (best, os.path.exists(best)))
        if os.path.exists(best):
            p = subprocess.run([PY, '-X', 'utf8', os.path.join(SUITE, 'scripts', 'm2a_person_gt2_eval.py'),
                                '--run', '--weights', best,
                                '--pred-dir', os.path.join(REP, 'person_gt2', 'preds_cand')],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
            log('评测 rc=%d' % p.returncode)
            log((p.stdout or '')[-3000:])
            if p.returncode != 0:
                log((p.stderr or '')[-2000:])
    finally:
        stop.set()
        release()


if __name__ == '__main__':
    sys.exit(main())
