"""M2a（人/手/脚检测）**推断分辨率**消融：交付图上的检出覆盖率。

为什么做这条：交付验收的真值里最弱的一环是**手**（8 张里只有 4 张检到手），而手是 M3 判定的入口
（没检到就没有裁图、更谈不上数手指）。改动成本最低的一档就是推断分辨率 —— `locate.py --imgsz`
现在默认 1280（之前 640 时人 5/8 vs 1280 时 7/8，已经证明分辨率在这一档很值钱）。
本脚本在同一 8 张交付图上跑 960 / 1280 / 1600（可加 1920），报每张图检到的 person/hand/foot 数，
以及**相对于默认 1280 的增减**（是否严格不差）。

用法：
  python scripts/ablate_m2a_imgsz.py                          # 960 1280 1600
  python scripts/ablate_m2a_imgsz.py --imgsz 1280 1600 1920
输出：.runtime/ablate_m2a_imgsz/<imgsz>/*.json + summary.csv + 屏幕表格
"""
import argparse
import glob
import io
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PY_S = r'python'
DEF_SRC = r'<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final'


def run(imgsz, src, root, extra=None, tag=None):
    out = os.path.join(root, ('%d_%s' % (imgsz, tag)) if tag else str(imgsz))
    os.makedirs(out, exist_ok=True)
    cmd = [PY_S, '-X', 'utf8', os.path.join(HERE, 'locate.py'), '--src', src, '--out', out,
           '--imgsz', str(imgsz)] + (extra or [])
    print('  $ locate.py --imgsz %d %s → %s' % (imgsz, ' '.join(extra or []), out), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJ, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('  !! 退出码 %s: %s' % (r.returncode, (r.stderr or '')[-300:]), flush=True)
    return out


def counts(out):
    rows = {}
    for p in sorted(glob.glob(os.path.join(out, '*.json'))):
        d = json.load(io.open(p, encoding='utf-8'))
        bx = d.get('boxes', [])
        cls = [b.get('cls') for b in bx]
        bad = [f for f in d.get('flags', []) if f.get('type') == 'part_check'
               and f.get('verdict') not in ('ok', 'skip', None)]
        tight = [f for f in d.get('flags', []) if f.get('type') == 'foot_tight_check'
                 and f.get('verdict') != 'ok']
        rows[os.path.basename(d['image'])] = dict(
            person=sum(1 for c in cls if c == 'person'),
            hand=sum(1 for c in cls if c == 'hand'),
            foot=sum(1 for c in cls if c in ('foot', 'left_foot', 'right_foot', 'foot(pose)')),
            bad=len(bad), tight=len(tight))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', nargs='+', type=int, default=[960, 1280, 1600])
    ap.add_argument('--src', default=DEF_SRC)
    ap.add_argument('--base', type=int, default=1280, help='算增减的基准')
    ap.add_argument('--reuse', action='store_true', help='输出目录已存在就只解析、不重跑 locate.py')
    ap.add_argument('--extra', default=None,
                    help='透传给 locate.py 的额外参数，一条字符串（如 "--hand-weights runs/m2a_hand_v2/weights/best.pt"）')
    ap.add_argument('--tag', default=None, help='输出子目录加后缀，便于与基线对比（如 with_hand）')
    ap.add_argument('--out-root', default=os.path.join(PROJ, '.runtime', 'ablate_m2a_imgsz'))
    a = ap.parse_args()
    extra = a.extra.split() if a.extra else None
    res = {}
    for z in a.imgsz:
        out = os.path.join(a.out_root, ('%d_%s' % (z, a.tag)) if a.tag else str(z))
        if a.reuse and glob.glob(os.path.join(out, '*.json')):
            print('  复用 %s' % out, flush=True)
        else:
            out = run(z, a.src, a.out_root, extra, a.tag)
        res[z] = counts(out)
    names = sorted(set().union(*[set(v) for v in res.values()]))
    if a.base not in res:
        res[a.base] = counts(os.path.join(a.out_root, str(a.base))) if os.path.isdir(
            os.path.join(a.out_root, str(a.base))) else {}

    hdr = '%-40s | %s' % ('image', ' | '.join('%-14s' % ('sz%d' % z) for z in a.imgsz))
    print('\n' + hdr)
    print('-' * len(hdr))
    for n in names:
        cells = []
        for z in a.imgsz:
            r = res[z].get(n, {})
            mark = ''
            b = res.get(a.base, {}).get(n)
            if b:
                d = (r.get('hand', 0) + r.get('foot', 0) + r.get('person', 0)) - \
                    (b.get('hand', 0) + b.get('foot', 0) + b.get('person', 0))
                mark = ('+%d' % d) if d > 0 else ('%d' % d if d < 0 else '=')
            cells.append('%-14s' % ('P%d H%d F%d %s' % (r.get('person', 0), r.get('hand', 0),
                                                        r.get('foot', 0), mark)))
        print('%-40s | %s' % (n, ' | '.join(cells)))
    print('-' * len(hdr))
    KNOWN = {'04_escape_recaptured_2000.png', '05_escape_restraint_start_2000.png',
             '07_escape_restraint_tightened_2000.png', '08_escape_feather_tickle_2000.png'}
    print('覆盖率 / 旗标：')
    for z in a.imgsz:
        r = res[z]
        n = len(r)
        hit = sum(1 for k, v in r.items() if k in KNOWN and (v['bad'] or v['tight']))
        print('  imgsz=%-5d person %d/%d  hand %d/%d  foot %d/%d | 框 %d | 已知缺陷(主判定∪收紧) %d/4 '
              '| 主判定旗标 %d 收紧旗标 %d' % (
                  z, sum(1 for v in r.values() if v['person']), n,
                  sum(1 for v in r.values() if v['hand']), n,
                  sum(1 for v in r.values() if v['foot']), n,
                  sum(v['person'] + v['hand'] + v['foot'] for v in r.values()),
                  hit, sum(v['bad'] for v in r.values()), sum(v['tight'] for v in r.values())))
    csv = os.path.join(a.out_root, 'summary.csv')
    with io.open(csv, 'w', encoding='utf-8', newline='') as fh:
        fh.write('image,imgsz,person,hand,foot,m3_bad,m3_tight\n')
        for n in names:
            for z in a.imgsz:
                r = res[z].get(n, {})
                fh.write('%s,%d,%d,%d,%d,%d,%d\n' % (n, z, r.get('person', 0), r.get('hand', 0),
                                                     r.get('foot', 0), r.get('bad', 0), r.get('tight', 0)))
    print('→ %s' % csv)


if __name__ == '__main__':
    main()
