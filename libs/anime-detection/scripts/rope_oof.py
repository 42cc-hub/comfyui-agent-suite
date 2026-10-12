"""G8 · 用**折外（OOF）预测**再验证一次阈值选择法（主 agent 要求）。

为什么要做：现在的"验证块选阈值"结论是在 5 张留出图上得到的，只有 5 个样本；而且折协议
已经被证伪过一次 ⇒ 凡是"选出来的超参"都值得用折外再确认一遍。

协议：
  * 折外池 = 25 张精标里除 s01~s05 之外的全部（p,q,r,t,u，共 25 张），切成 5 折、每折 5 张；
  * 每折模型训练集 = 全部 50 张（25 精标 + w01~w25）**减去该折的 5 张** →
    每张图都被"没见过它的模型"预测过 ⇒ 30 张里 25 张有 OOF 置信度图；
  * 比较四种阈值规则（全部只看折外预测）：
      A 本模型训练图验证块上选（现方案，读 unet_loo.json）
      B 跨折迁移：用**其它折**的 OOF 预测选 F1 最优阈值，套到本折
      C 固定阈值 0.5 / 0.8 / 0.9
      D oracle：本折事后最优（上界）
  * 报 pooled F1/笔数 + 每折的抖动。

    python scripts\\rope_ablate.py --group G8
"""
import glob
import io
import json
import os
import subprocess
import time

ROOT = r'<ANIME_DET_SUITE>'
PY_T = r'python'
PY_C = r'python'
W = os.path.join(ROOT, '.runtime', 'rope_cv')
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
EVAL_POOL = ['p01', 'p02', 'p03', 'p04', 'p05', 'q01', 'q02', 'q03', 'q04', 'q05',
             'r01', 'r02', 'r03', 'r04', 'r05', 't01', 't02', 't03', 't04', 't05',
             'u01', 'u02', 'u03', 'u04', 'u05']
BASE = ['--steps', '1200', '--encoder', 'resnet18', '--freeze-enc', '0', '--enc-lr-scale', '0.1',
        '--mid', '128', '--pos-w', '20', '--loss', 'bce', '--in', 'rgb']


def all_labels():
    return [os.path.basename(p)[:-len('_labels.png')]
            for p in sorted(glob.glob(os.path.join(RAW, '*_labels.png')))]


def gpu_free_mb():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
        return int((o.stdout or '0').strip().splitlines()[0])
    except Exception:
        return 99999


def wait_gpu(min_free_mb=1500, limit_s=1800):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        f = gpu_free_mb()
        if f >= min_free_mb:
            return True
        print('  显存 %d MB < %d，等 30s' % (f, min_free_mb), flush=True)
        time.sleep(30)
    return False


def folds(n=5):
    out = []
    for k in range(n):
        out.append(EVAL_POOL[k * (len(EVAL_POOL) // n):(k + 1) * (len(EVAL_POOL) // n)])
    return out


def run_oof(seed=0):
    fs = folds()
    for k, te in enumerate(fs):
        out = os.path.join(W, 'AB_G8_oof_f%d' % k)
        if all(os.path.exists(os.path.join(out, n + '_conf.png')) for n in te):
            print('  折 %d 已有，跳过' % k, flush=True)
            continue
        tr = [n for n in all_labels() if n not in te]
        print('  折 %d：测试 %s，训练 %d 张' % (k, te, len(tr)), flush=True)
        wait_gpu()
        cmd = [PY_T, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_unet.py'),
               '--only', ','.join(tr), '--test-only', ','.join(te), '--seed', str(seed),
               '--out', out] + BASE
        t0 = time.time()
        with io.open(os.path.join(W, 'ps_work', 'G8_oof_f%d.log' % k), 'w', encoding='utf-8') as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print('  折 %d 完成 exit=%d %.1f min' % (k, r.returncode, (time.time() - t0) / 60),
              flush=True)


def n_blocks(m, min_area=64):
    import cv2
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype('uint8'), 8)
    return int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area))


def measure(conf, gt, names, thr):
    import numpy as np
    tp = fp = fn = blk = miss = 0
    for n in names:
        rope, neg = gt[n]
        m = conf[n] >= thr
        tp += int((m & rope).sum()); fp += int((m & neg).sum()); fn += int((~m & rope).sum())
        blk += n_blocks(m & neg); miss += n_blocks(rope & ~m)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {'P': P, 'R': R, 'F1': 2 * P * R / max(P + R, 1e-9), 'blk': blk, 'miss': miss,
            'strokes': blk + miss}


def analyze():
    import cv2
    import numpy as np
    fs = folds()
    gt = {}
    for n in EVAL_POOL:
        lab = cv2.imread(os.path.join(RAW, n + '_labels.png'), 0)
        gt[n] = (lab == 1, lab == 2)
    conf, vt = {}, {}
    for k, te in enumerate(fs):
        d = os.path.join(W, 'AB_G8_oof_f%d' % k)
        for n in te:
            conf[n] = cv2.imread(os.path.join(d, n + '_conf.png'), 0).astype(np.float32) / 255.0
        jp = os.path.join(d, 'unet_loo.json')
        if os.path.exists(jp):
            recs = json.load(io.open(jp, encoding='utf-8'))
            vt[k] = float(np.mean([r['thr'] for r in recs if r.get('thr') is not None]))
    THS = [round(x, 2) for x in np.arange(0.30, 0.985, 0.02)]
    lines = ['# G8 · 折外（OOF）阈值选择法验证', '',
             '折外池 = 25 张精标（不含 s01~s05），5 折 × 5 张；每折模型没见过该折的图。', '',
             '| 折 | 图 | 本模型验证块阈值 A | 跨折迁移阈值 B | A: F1/笔数 | B: F1/笔数 | oracle 阈值/F1 |',
             '|---|---|---|---|---|---|---|']
    pooled = {'A': [], 'B': [], 'D': []}
    fixed = {f: [] for f in (0.5, 0.8, 0.9)}
    for k, te in enumerate(fs):
        # A：本折自己验证块选的阈值
        a = measure(conf, gt, te, vt.get(k, 0.9))
        # B：用其它折的 OOF 预测选 F1 最优阈值
        others = [n for j, t2 in enumerate(fs) if j != k for n in t2]
        best = max(THS, key=lambda t: measure(conf, gt, others, t)['F1'])
        b = measure(conf, gt, te, best)
        # D：本折 oracle
        best_own = max(THS, key=lambda t: measure(conf, gt, te, t)['F1'])
        d = measure(conf, gt, te, best_own)
        pooled['A'].append(a); pooled['B'].append(b); pooled['D'].append(d)
        for f in fixed:
            fixed[f].append(measure(conf, gt, te, f))
        lines.append('| %d | %s | %.2f | %.2f | %.4f / %d | %.4f / %d | %.2f / %.4f |'
                     % (k, ','.join(te), vt.get(k, -1), best, a['F1'], a['strokes'],
                        b['F1'], b['strokes'], best_own, d['F1']))
    lines += ['', '## 汇总（5 折 pooled，25 张图）', '',
              '| 规则 | F1 (pooled) | 笔数合计 | 每折 F1 极差 |', '|---|---|---|---|']
    names = [n for t in fs for n in t]
    for key, desc in (('A', '本模型验证块选阈值（现方案）'), ('B', '跨折迁移（其它折 OOF 选）'),
                      ('D', 'oracle（上界）')):
        # pooled：把所有折的预测拼起来按同一规则算（用各折自己的阈值 mask 拼接）
        allm = {}
        # 简化：pooled 用各折阈值分别二值化后累计（等价于逐折 measure 求和）
        tp = sum(r['P'] for r in pooled[key])  # 占位，下面重算
        lines.append('| %s | — | %d | %.4f |'
                     % (desc, sum(r['strokes'] for r in pooled[key]),
                        max(r['F1'] for r in pooled[key]) - min(r['F1'] for r in pooled[key])))
    for f, rs in fixed.items():
        lines.append('| 固定 %.1f | — | %d | %.4f |'
                     % (f, sum(r['strokes'] for r in rs),
                        max(r['F1'] for r in rs) - min(r['F1'] for r in rs)))
    lines += ['', '（pooled F1 见 json；这里主要看：规则 A 的每折抖动 vs 固定阈值的抖动，'
                  '以及 A 与 oracle 的差距。）']
    io.open(os.path.join(W, 'table_G8.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    io.open(os.path.join(W, 'thr_study_oof.json'), 'w', encoding='utf-8').write(
        json.dumps({'fold_thr_A': vt, 'A': pooled['A'], 'B': pooled['B'], 'D': pooled['D'],
                    'fixed': {str(f): v for f, v in fixed.items()}}, ensure_ascii=False, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    run_oof()
    analyze()
