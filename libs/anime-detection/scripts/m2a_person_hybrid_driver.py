"""person hybrid 训练 driver：L3 票 → 空闲守卫 → 训练（epochs=40，无 time 截断）→ 打印耗时。

用法: python m2a_person_hybrid_driver.py [--epochs 40]
评测在训练完成后由 m2a_person_hybrid_eval.py 单独跑。
"""
import argparse, json, os, subprocess, sys, threading, time

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
CLAIM = os.path.join(SUITE, '..', '..', 'team', 'gpu_claim.py')
YAML = os.path.join(D, 'det_person_hybrid.yaml')
WARM = os.path.join(SUITE, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
OUT = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid')
OWNER = 'm2a'

STOP = threading.Event()


def sh(args, timeout=None):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, errors='replace')


def claim(minutes):
    for i in range(8):
        r = sh([sys.executable, CLAIM, 'acquire', '--owner', OWNER, '--level', '3',
                '--minutes', str(minutes), '--task', 'person hybrid retrain (mixed labels, 3x person)',
                '--wait', '240'])
        print('[claim %d] rc=%s %s' % (i + 1, r.returncode, (r.stdout or r.stderr).strip()[-300:]))
        if r.returncode == 0:
            return True
        if r.returncode == 2:
            return False
    return False


def heartbeat():
    while not STOP.wait(60):
        r = sh([sys.executable, CLAIM, 'heartbeat', '--owner', OWNER])
        print('[heartbeat] rc=%s %s' % (r.returncode, (r.stdout or '').strip()[:80]))


def release():
    r = sh([sys.executable, CLAIM, 'release', '--owner', OWNER])
    print('[release] rc=%s %s' % (r.returncode, (r.stdout or '').strip()[:120]))


def gpu_free_mib():
    r = sh(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
    try:
        return int(r.stdout.strip().split('\n')[0])
    except Exception:
        return -1


def comfy_busy():
    try:
        import urllib.request
        with urllib.request.urlopen('http://127.0.0.1:7187/queue', timeout=3) as f:
            q = json.loads(f.read().decode('utf-8'))
        return len(q.get('queue_running', [])) + len(q.get('queue_pending', [])) > 0
    except Exception:
        return False


def anima_running():
    r = sh(['powershell', '-NoProfile', '-Command',
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"])
    return 'anima_txt2img' in (r.stdout or '')


def wait_idle(max_min=20, need=3500):
    t0 = time.time()
    ok_since = None
    while time.time() - t0 < max_min * 60:
        free = gpu_free_mib()
        busy = comfy_busy() or anima_running()
        if not busy and free >= need:
            ok_since = ok_since or time.time()
            if time.time() - ok_since >= 60:
                print('[guard] idle confirmed free=%dMiB' % free)
                return True
        else:
            if busy or free < need:
                print('[guard] busy=%s free=%dMiB' % (busy, free))
            ok_since = None
        time.sleep(20)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--minutes', type=int, default=45)
    a = ap.parse_args()
    if not claim(a.minutes):
        print('[blocked] no GPU ticket')
        return 2
    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()
    try:
        if not wait_idle():
            print('[blocked] external GPU job / low VRAM for 20min')
            return 3
        from ultralytics import YOLO
        m = YOLO(WARM)
        t0 = time.time()
        m.train(data=YAML, epochs=a.epochs, patience=60, imgsz=960, batch=6, workers=4,
                device='', project=os.path.join(SUITE, 'runs'), name='m2a_det_person_hybrid',
                exist_ok=True, val=False, plots=False, seed=20261023, deterministic=True,
                pretrained=True, optimizer='auto', lr0=0.004, cos_lr=False, close_mosaic=10,
                amp=True, cache=False)
        print('[train] done %.1f min' % ((time.time() - t0) / 60))
        best = os.path.join(OUT, 'weights', 'best.pt')
        print('[train] best.pt exists=%s size=%s' % (os.path.exists(best), os.path.getsize(best) if os.path.exists(best) else -1))
        return 0
    finally:
        STOP.set()
        release()


if __name__ == '__main__':
    sys.exit(main())
