#!/usr/bin/env python
"""多 seed 消融跑批器（8 小时窗口的主力工具）。

设计要点（对应主 agent 的硬约束）：
  * 协议：`--only`（训练）/`--test-only`（留出）分离，脚本内部**只训一次**（已修）；
    每个配置 × 多个 seed，报 **mean±std**，区间重叠就明说"无差别"。
  * 输入：默认 RGB（`--in rgb` 写死在命令里，避免以后默认值改动影响可比性）。
  * 主 KPI：预计笔数，且**必须在同等召回下比**——跑完一组自动重算 thr_sweep2 并抓取
    R≈0.64/0.70/0.75/0.80 四个工作点的笔数。
  * 已存在的输出目录（5 张留出图 + eval json 齐全）默认跳过，不重复烧 GPU。
  * 每组落盘：`.runtime/rope_cv/ablation_<组>.jsonl` + `.md`

    python scripts\\rope_ablate.py --group G0
"""
import argparse
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
PROBE = ['x05']   # 永久探针（大场景/暗细绳类），不进训练，只用来看"新数据有没有教会这一类"
TEST = ['s01', 's02', 's03', 's04', 's05']
R1 = ['p01', 'p02', 'p03', 'p04', 'p05', 'q01', 'q02', 'q03', 'q04', 'q05',
      'r01', 'r02', 'r03', 'r04', 'r05']
BASE = ['--steps', '1200', '--encoder', 'resnet18', '--freeze-enc', '0', '--enc-lr-scale', '0.1',
        '--mid', '128', '--pos-w', '20', '--loss', 'bce', '--in', 'rgb']
MATCH_R = (0.64, 0.70, 0.75, 0.80)

# 组定义：conf 名 -> 额外 CLI。BASE 之上的覆盖写在 extras 里（argparse 后者生效）
GROUPS = {
    'G0': dict(  # 修正协议 + RGB 下的骨干基线表
        split=15, seeds=[0, 1, 2],
        confs={
            'r18_ft': [],
            'r18_frz': ['--freeze-enc', '1'],
            'yolo_frz': ['--encoder', 'yolo11n_seg', '--freeze-enc', '1'],
            'yolo_ft': ['--encoder', 'yolo11n_seg', '--freeze-enc', '0', '--enc-lr-scale', '0.1'],
            'mbv3_ft': ['--encoder', 'mobilenet_v3_small', '--freeze-enc', '0',
                        '--enc-lr-scale', '0.1'],
        }),
    'G1': dict(  # 当前数据（25 精标 + 25 负样本 = 50 张）参考基线
        split=50, seeds=[0, 1, 2],
        confs={'ref50': []}),
    'G2': dict(  # 损失/采样，在 RGB + 冠军骨干下重测
        split=50, seeds=[0, 1, 2],
        confs={
            'costmap': ['--costmap', '1'],
            'cldice': ['--loss', 'bce+cldice'],
            'hardneg': ['--hard-neg', '1'],
            'coh': ['--coh-w', '0.1'],
            'costmap_coh': ['--costmap', '1', '--coh-w', '0.1'],
        }),
    'G3': dict(  # 增强（需要 rope_unet.py 的 --rot / --copy-paste）
        split=50, seeds=[0, 1, 2],
        confs={
            'rot20': ['--rot', '20'],
            'cpaste': ['--copy-paste', '0.3'],
            'rot_cpaste': ['--rot', '20', '--copy-paste', '0.3'],
        }),
    'G5': dict(  # 结构：解码器宽度
        split=50, seeds=[0, 1, 2],
        confs={'mid192': ['--mid', '192'], 'mid256': ['--mid', '256']}),
    'G6': dict(  # 训练瓦片尺寸（预测瓦片固定 512）
        split=50, seeds=[0, 1, 2],
        confs={'tile1024': ['--tile', '1024', '--batch', '2'],
               'tile384': ['--tile', '384']}),
    'G7': dict(  # 重跑 yolo 不冻结（G0 里的 yolo_ft 是无效行：ultralytics 权重默认 requires_grad=False，
        #            修好后 unfrozen 才真的放开 3.10M 参数），用与 G0 相同的 15 张切分以便对比
        split=15, seeds=[0, 1, 2],
        confs={'yolo_ft_fixed': ['--encoder', 'yolo11n_seg', '--freeze-enc', '0',
                                 '--enc-lr-scale', '0.1']}),
    'G9': dict(  # 新数据（batch b1：x01~x04 暗/细/低对比绳）加入后，重跑当前最优组合 rot20
        split=50, seeds=[0, 1, 2],
        confs={'rot20': ['--rot', '20']}),
    'G10': dict(  # 假设：x01~x04 里 99% 是背景，而 pos_weight 封顶 20 导致正权被稀释
        #            → 放开正权（40）再测一次，看 s 留出集与 x05 探针能不能回来
        split=50, seeds=[0, 1, 2],
        confs={'rot20_pw40': ['--rot', '20', '--pos-w', '40']}),
    'G11': dict(  # 人工新数据全部入库后（x/y/z/aa 共 20 张 + w01~w25 负样本）复核最优组合
        split=50, seeds=[0, 1, 2],
        confs={'rot20': ['--rot', '20']}),
    'G12': dict(  # bb 批（再 +5 张）入库后复核同一组合
        split=50, seeds=[0, 1, 2],
        confs={'rot20': ['--rot', '20']}),
    'G14': dict(  # 收尾：全部人工数据入库后（85 个标签文件）的最终复核
        split=50, seeds=[0, 1, 2],
        confs={'rot20': ['--rot', '20']}),
    'C1': dict(  # 路线 C1：Swin-T（分层 attention）替换 resnet18（判据见 docs/rope_C_prereg.md）
        split=50, seeds=[0, 1, 2],
        confs={'swin_t': ['--rot', '20', '--encoder', 'swin_t', '--freeze-enc', '0',
                          '--enc-lr-scale', '0.1']}),
    'C2': dict(  # 路线 C2：ViT-B/16 冻结 + 薄解码器（位置编码固定 224 ⇒ --tile 224）
        split=50, seeds=[0, 1, 2],
        confs={'vit_b_16_frz': ['--rot', '20', '--encoder', 'vit_b_16', '--freeze-enc', '1',
                                '--tile', '224']}),
}

# 折外阈值验证（主 agent 要求）走单独脚本，队列照用 `--group G8`
SPECIAL = {'G8': 'scripts/rope_oof.py'}


def all_labeled():
    names = [os.path.basename(p)[:-len('_labels.png')]
             for p in sorted(glob.glob(os.path.join(RAW, '*_labels.png')))]
    # x05（大水族馆场景）**永久留作"暗/细绳"类的探针**：不进训练，只用来评
    # "新数据有没有教会这一类"（G9 在它上面从 0.834 掉到 0.796，需要干净的对照）。
    return [n for n in names if n not in TEST and n not in PROBE]


def split_of(name):
    return R1 if name == 15 else all_labeled()


def tag_of(group, conf, seed):
    return 'AB_%s_%s_s%d' % (group, conf, seed)


def gpu_free_mb():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
        return int((out.stdout or '0').strip().splitlines()[0])
    except Exception:
        return 99999


def wait_gpu(min_free_mb=1500, limit_s=1800):
    """GPU 共用：显存不足就先等（主 agent 的 M3 也在跑，谁都不许把对方挤 OOM）。"""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        f = gpu_free_mb()
        if f >= min_free_mb:
            return True
        print('  显存只剩 %d MB（< %d），等 30s 再看' % (f, min_free_mb), flush=True)
        time.sleep(30)
    return False


def run_one(group, conf, extras, split, seed, force=False):
    tag = tag_of(group, conf, seed)
    out = os.path.join(W, tag)
    ev = os.path.join(W, 'eval_%s.json' % tag)
    tr = split_of(split)
    done = all(os.path.exists(os.path.join(out, n + '_conf.png')) for n in TEST)
    if done and os.path.exists(ev) and not force:
        return tag, True
    wait_gpu()
    cmd = [PY_T, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_unet.py'),
           '--only', ','.join(tr), '--test-only', ','.join(TEST), '--seed', str(seed),
           '--out', out] + BASE + extras
    t0 = time.time()
    with io.open(os.path.join(W, 'ps_work', tag + '.log'), 'w', encoding='utf-8') as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0
    if ok:
        subprocess.run([PY_C, '-X', 'utf8', os.path.join(ROOT, 'scripts', 'rope_eval.py'),
                        '--pred', out, '--json', ev], cwd=ROOT, capture_output=True, text=True)
    print('  %-28s seed %d  %s  %.1f min' % (conf, seed, 'OK' if ok else 'FAIL',
                                             (time.time() - t0) / 60), flush=True)
    return tag, ok


def agg_eval(path):
    d = json.load(io.open(path, encoding='utf-8'))
    tp = sum(x['TP'] for x in d)
    fp = sum(x['FP'] for x in d)
    fn = sum(x['FN'] for x in d)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return {'P': P, 'R': R, 'F1': 2 * P * R / max(P + R, 1e-9),
            'blk': sum(x['FP_blocks'] for x in d), 'miss': sum(x['miss_segments'] for x in d),
            'strokes': sum(x['est_strokes'] for x in d)}


def agg_ms(out):
    p = os.path.join(out, 'unet_loo.json')
    if not os.path.exists(p):
        return None
    d = json.load(io.open(p, encoding='utf-8'))
    v = [x.get('ms_per_Mpx') for x in d if x.get('ms_per_Mpx')]
    return sum(v) / len(v) if v else None


def mean_std(v):
    if not v:
        return 0.0, 0.0
    m = sum(v) / len(v)
    if len(v) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, var ** 0.5


def summarize(group, tags, confs, seeds):
    # 只重算本组的目录（全量重扫要十几分钟，会一直挡着下一组起跑）
    subprocess.run([PY_C, '-X', 'utf8', os.path.join(W, 'ps_work', 'thr_sweep2.py'),
                    '--only-tags', 'AB_%s_' % group], cwd=ROOT, capture_output=True, text=True)
    sw = json.load(io.open(os.path.join(W, 'thr_sweep2.json'), encoding='utf-8'))['sweep']
    lines = ['# 消融组 %s（%s）' % (group, time.strftime('%Y-%m-%d %H:%M')), '',
             '协议：训练集/留出集分离、单模型、固定 seed、输入 RGB；留出 s01~s05。',
             '主 KPI：预计笔数，且下面给出**同等召回**下的笔数（各自阈值下的笔数不可横比）。', '',
             '| 配置 | F1 (mean±std) | P | R | 各自阈值笔数 | R≈0.64 | ≈0.70 | ≈0.75 | ≈0.80 | ms/Mpx |',
             '|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    for conf in confs:
        f1, st, pp, rr, strokes, ms, mr = [], [], [], [], [], [], {}
        for seed in seeds:
            tag = tag_of(group, conf, seed)
            ev = os.path.join(W, 'eval_%s.json' % tag)
            if not os.path.exists(ev):
                continue
            a = agg_eval(ev)
            f1.append(a['F1']); st.append(a['strokes'])
            pp.append(a['P']); rr.append(a['R'])
            m = agg_ms(os.path.join(W, tag))
            if m:
                ms.append(m)
            if tag in sw:
                for x in MATCH_R:
                    near = min(sw[tag], key=lambda r: abs(r['R'] - x))
                    mr.setdefault(x, []).append(near['strokes'])
            rows.append(dict(group=group, conf=conf, seed=seed, tag=tag, **a))
        mF1, sF1 = mean_std(f1)
        cells = []
        for x in MATCH_R:
            v = mr.get(x, [])
            cells.append('%.0f±%.0f' % mean_std(v) if v else '-')
        lines.append('| %s | %.4f±%.4f | %.3f | %.3f | %.0f±%.0f | %s | %.0f |'
                     % (conf, mF1, sF1, mean_std(pp)[0], mean_std(rr)[0], mean_std(st)[0],
                        mean_std(st)[1], ' | '.join(cells), mean_std(ms)[0] if ms else 0))
    with io.open(os.path.join(W, 'ablation_%s.jsonl' % group), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with io.open(os.path.join(W, 'ablation_%s.md' % group), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--seeds', default='')
    a = ap.parse_args()
    if a.group in SPECIAL:                    # G8：折外阈值验证，跑专用脚本
        subprocess.run([PY_T, '-X', 'utf8', os.path.join(ROOT, SPECIAL[a.group])],
                       cwd=ROOT)
        return
    g = GROUPS[a.group]
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else g['seeds']
    print('组 %s：split=%s seeds=%s confs=%s'
          % (a.group, g['split'], seeds, list(g['confs'])), flush=True)
    for conf, extras in g['confs'].items():
        for seed in seeds:
            tag, ok = run_one(a.group, conf, extras, g['split'], seed, force=a.force)
            if not ok and tag:
                print('  !! %s 失败，详见 .runtime/rope_cv/ps_work/%s.log' % (tag, tag), flush=True)
    summarize(a.group, None, list(g['confs']), seeds)


if __name__ == '__main__':
    main()
