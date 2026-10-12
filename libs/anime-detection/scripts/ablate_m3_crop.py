"""M3 裁图口径消融（R39/R40 那条线的**推断侧**对照）。

背景：M3 训练裁图是"姿态脚点包围盒 + 0.35 外扩"（紧贴远端脚），产线 `locate.py` 默认给的是
"检测器脚框 + 0.10"。同一只脚两种口径下 p(bad) 可以是 1.00 vs 0.00 ⇒ 交付图 08（趾多）三模型全判 ok。
`locate.py --m3-foot-crop pose` 用踝/膝锚点复现训练口径。本脚本在**同一批交付图**上跑
`det`（老口径）与 `pose`（训练口径）两遍，逐图列出报出的部位与旗标数，回答：

  · 已知 4 张缺陷图（04/05/07/08）各被报出几张（主判定 / 主判定∪收紧旗标）
  · 另外 4 张（01/02/03/06）被误报多少 —— 只看召回不看误报会把产线推向滥报

用法：
  python scripts/ablate_m3_crop.py --modes det pose
  python scripts/ablate_m3_crop.py --src <图目录> --modes pose --imgsz 1280
输出：<out-root>/<mode>/*.json（locate.py 原样产物）+ <out-root>/summary.csv + 屏幕表格
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PY_S = r'python'
DEF_SRC = r'<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final'
# 交付图上人工判定的 M3 真值（用户 2026-10-11 给）：04 手指数多 / 05 趾多+指畸形 / 07 趾畸形 / 08 趾多
DEFECTS = {'04_escape_recaptured_2000.png', '05_escape_restraint_start_2000.png',
           '07_escape_restraint_tightened_2000.png', '08_escape_feather_tickle_2000.png'}


def run_mode(mode, src, out_root, imgsz, extra, tag=None):
    out = os.path.join(out_root, tag or mode)
    os.makedirs(out, exist_ok=True)
    cmd = [PY_S, '-X', 'utf8', os.path.join(HERE, 'locate.py'), '--src', src, '--out', out,
           '--imgsz', str(imgsz), '--m3-foot-crop', mode] + extra
    print('  $ locate.py --m3-foot-crop %s %s → %s' % (mode, ' '.join(extra), out), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJ, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('  !! locate.py 退出码 %s: %s' % (r.returncode, (r.stderr or '')[-400:]), flush=True)
    return out


def read_flags(out):
    rows = {}
    for p in sorted(glob.glob(os.path.join(out, '*.json'))):
        d = json.load(io.open(p, encoding='utf-8'))
        name = os.path.basename(d['image'])
        bad = [f for f in d.get('flags', [])
               if f.get('type') == 'part_check' and f.get('verdict') not in ('ok', 'skip', None)]
        tight = [f for f in d.get('flags', [])
                 if f.get('type') == 'foot_tight_check' and f.get('verdict') != 'ok']
        rows[name] = {
            'bad': ['%s#%s:%s' % (f.get('part'), f.get('index'), f.get('verdict')) for f in bad],
            'tight': ['%s' % f.get('verdict') for f in tight],
            'n_flags': len(bad) + len(tight),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEF_SRC)
    ap.add_argument('--modes', nargs='+', default=['det', 'pose'])
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--out-root', default=os.path.join(PROJ, '.runtime', 'ablate_m3_crop'))
    ap.add_argument('--no-foot-tight', action='store_true', help='关掉并行收紧旗标（只比主判定）')
    ap.add_argument('--m3-weights', nargs='+', default=None, help='默认用 locate.py 的生产并集')
    ap.add_argument('--m3-conf', type=float, default=None,
                    help='M3 的 p(bad) 阈值（见 locate.py --m3-conf；默认不传 = locate.py 默认 0.5）')
    ap.add_argument('--tag', default=None, help='输出子目录名（默认用 mode；扫阈值时用来区分）')
    a = ap.parse_args()
    extra = ([] if a.m3_weights is None else ['--m3-weights'] + a.m3_weights)
    if a.m3_conf is not None:
        extra += ['--m3-conf', str(a.m3_conf)]
    if a.no_foot_tight:
        extra.append('--no-m3-foot-tight')

    res = {}
    for mode in a.modes:
        key = a.tag or mode
        res[key] = read_flags(run_mode(mode, a.src, a.out_root, a.imgsz, extra, key))

    names = sorted(set().union(*[set(v) for v in res.values()]))
    modes = list(res)
    w = max(len(n) for n in names) if names else 10

    def cell(m, n):
        r = res.get(m, {}).get(n, {'bad': [], 'tight': []})
        return ','.join(r['bad']) or '-'

    print('\n%-*s | %-6s | %s' % (w, 'image', 'defect', ' || '.join('%-24s %s' % ('main flags(%s)' % m,
                                                                                 'tight(%s)' % m)
                                                                   for m in modes)))
    print('-' * 100)
    summ = {m: {'hit': 0, 'hit_union': 0, 'fp': 0, 'flags': 0} for m in res}
    for n in names:
        is_def = n in DEFECTS
        for m in res:
            r = res[m].get(n, {'bad': [], 'tight': [], 'n_flags': 0})
            summ[m]['hit'] += 1 if (is_def and r['bad']) else 0
            summ[m]['hit_union'] += 1 if (is_def and (r['bad'] or r['tight'])) else 0
            summ[m]['fp'] += 1 if ((not is_def) and r['n_flags']) else 0
            summ[m]['flags'] += r['n_flags']
        print('%-*s | %-6s | %s' % (w, n, 'DEFECT' if is_def else '-',
                                    ' || '.join('%-24s %s' % (cell(m, n),
                                                              ','.join(res[m].get(n, {}).get('tight', [])) or '-')
                                                for m in modes)))
    print('-' * 100)
    for m, s in summ.items():
        print('[%s] 已知缺陷主判定 %d/4，主判定∪收紧 %d/4；正常图被报 %d/4；总旗标 %d'
              % (m, s['hit'], s['hit_union'], s['fp'], s['flags']))

    csv = os.path.join(a.out_root, 'summary_%s.csv' % (a.tag or '-'.join(a.modes)))
    with io.open(csv, 'w', encoding='utf-8', newline='') as fh:
        fh.write('image,defect,mode,main_flags,tight_flags\n')
        for n in names:
            for m in res:
                r = res[m].get(n, {'bad': [], 'tight': []})
                fh.write('%s,%d,%s,"%s","%s"\n' % (n, int(n in DEFECTS), m,
                                                   '|'.join(r['bad']), '|'.join(r['tight'])))
    print('→ %s' % csv)


if __name__ == '__main__':
    main()
