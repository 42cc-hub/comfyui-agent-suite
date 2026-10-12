"""G13 · "最弱的一类能不能学会"：对"红绳贴红底/低对比"这类图做留一法。

问题：累计标了 30 张新图，标准留出集（s01~s05）F1 反而从 0.927 掉到 0.909；
而在这一类图上，没见过它们的旧模型只有 F1 0.17~0.53。到底是"数据不够"还是"这类学不会"？

做法（公平对照）：
  * 基线 = 旧模型（AB_G3_rot20，训练时没有任何 x/y/z/aa/bb 图）在这些图上的 F1；
  * 留一 = 对每张难图 h：训练集 = 全部标签 − (s01~s05 探针) − x05 − h，只测 h。
    这样 h 由"见过同类其它 6 张"的模型预测 ⇒ 直接回答"同类样本有没有教会这一类"。

    python scripts\\rope_loo_class.py
"""
import glob
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_T = r'python'
PY_C = r'python'
W = os.path.join(ROOT, '.runtime', 'rope_cv')
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
ORIG = os.path.join(W, 'ps_work', '_orig')
PROBE = ['s01', 's02', 's03', 's04', 's05', 'x05']
HARD = ['x04', 'y04', 'y05', 'aa04', 'aa05', 'bb04', 'bb05']
BASE_W = os.path.join(ROOT, 'data', 'masks', 'rope', 'weights', 'rope_unet_AB_G3_rot20_s0_s01.pt')
BASE = ['--steps', '1200', '--encoder', 'resnet18', '--freeze-enc', '0', '--enc-lr-scale', '0.1',
        '--mid', '128', '--pos-w', '20', '--loss', 'bce', '--in', 'rgb', '--rot', '20']


def all_labels():
    return [os.path.basename(p)[:-len('_labels.png')]
            for p in sorted(glob.glob(os.path.join(RAW, '*_labels.png')))]


def gpu_free():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
        return int((o.stdout or '0').strip().splitlines()[0])
    except Exception:
        return 99999


def wait_gpu(min_free=1500, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        if gpu_free() >= min_free:
            return
        print('  显存不足，等 30s', flush=True)
        time.sleep(30)


def agg(jp):
    d = json.load(io.open(jp, encoding='utf-8'))
    tp = sum(x['TP'] for x in d); fp = sum(x['FP'] for x in d); fn = sum(x['FN'] for x in d)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {'P': P, 'R': R, 'F1': 2 * P * R / max(P + R, 1e-9),
            'strokes': sum(x['est_strokes'] for x in d)}


def baseline_on(names, tag='AB_G13_base_on_hard'):
    """旧模型（没见过这些图）在难图上的 F1 —— 公平的"之前"。"""
    src = os.path.join(W, 'ps_work', '_g13_src')
    os.makedirs(src, exist_ok=True)
    for n in names:
        f = sorted(glob.glob(os.path.join(ORIG, n + '_*')))
        if f:
            import shutil
            shutil.copy2(f[0], os.path.join(src, n + os.path.splitext(f[0])[1]))
    out = os.path.join(W, tag)
    subprocess.run([PY_T, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_predict.py'),
                    '--src', src, '--out', out, '--weights', BASE_W,
                    '--overlay-dir', os.path.join(out, '_ov')], cwd=ROOT, capture_output=True)
    jp = os.path.join(out, 'eval.json')
    subprocess.run([PY_C, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_eval.py'),
                    '--pred', out, '--json', jp], cwd=ROOT, capture_output=True, text=True)
    return json.load(io.open(jp, encoding='utf-8')) if os.path.exists(jp) else []


def main():
    rows = []
    print('=== 基线：旧模型（AB_G3_rot20，从未见过这些图）===', flush=True)
    base = baseline_on(HARD)
    for x in base:
        print('  %-5s F1 %.4f  P %.3f R %.3f' % (x['image'], x['F1'], x['P'], x['R']), flush=True)
    print('\n=== 留一：模型见过同类其它 6 张 ===', flush=True)
    for h in HARD:
        tag = 'AB_G13_hardLOO_%s' % h
        out = os.path.join(W, tag)
        tr = [n for n in all_labels() if n not in PROBE and n != h]
        if not all(os.path.exists(os.path.join(out, h + '_conf.png')) for _ in [0]):
            wait_gpu()
            cmd = [PY_T, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_unet.py'),
                   '--only', ','.join(tr), '--test-only', h, '--seed', '0', '--out', out] + BASE
            t0 = time.time()
            with io.open(os.path.join(W, 'ps_work', tag + '.log'), 'w', encoding='utf-8') as fh:
                r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            print('  [%s] 训练 exit=%d %.1f min' % (h, r.returncode, (time.time() - t0) / 60), flush=True)
        jp = os.path.join(out, 'eval.json')
        subprocess.run([PY_C, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_eval.py'),
                        '--pred', out, '--json', jp], cwd=ROOT, capture_output=True, text=True)
        if os.path.exists(jp):
            a = agg(jp)
            rows.append(dict(image=h, **a))
            print('  %-5s 留一 F1 %.4f  P %.3f R %.3f 笔数 %d' % (h, a['F1'], a['P'], a['R'], a['strokes']),
                  flush=True)
    lines = ['# G13 · 最弱一类（红绳贴红底/低对比）的留一法', '',
             '基线 = 旧模型（没有任何新数据）在这些图上的表现；留一 = 见过同类其它 6 张的模型。', '',
             '| 图 | 基线 F1 | 留一 F1 | 基线 P/R | 留一 P/R | 留一笔数 |', '|---|---|---|---|---|---|']
    bmap = {x['image']: x for x in base}
    for r in rows:
        b = bmap.get(r['image'], {})
        lines.append('| %s | %.4f | **%.4f** | %.3f/%.3f | %.3f/%.3f | %d |'
                     % (r['image'], b.get('F1', 0), r['F1'], b.get('P', 0), b.get('R', 0),
                        r['P'], r['R'], r['strokes']))
    if rows and bmap:
        mf = sum(r['F1'] for r in rows) / len(rows)
        mb = sum(bmap[r['image']]['F1'] for r in rows if r['image'] in bmap) / len(rows)
        lines += ['', '平均：基线 %.4f → 留一 %.4f（%+.4f）' % (mb, mf, mf - mb)]
    io.open(os.path.join(W, 'table_G13.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
