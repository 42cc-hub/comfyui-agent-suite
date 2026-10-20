r"""验收：`locate.py` 的 **默认** `--det-weights` 是否真的已经指向修好的权重。

为什么不能只验 override 路径：`--det-weights <新权重>` 能跑出好数字，只能证明权重要求存在；
用户/下游不带参数跑的时候用的是 `DEF_DET`，那条路必须在同一份输出上得到**全字段相同**的结果。

三段检查：
  A. 读 `locate.py` 的 `DEF_DET`，打印并判断文件名是否含 fix_v2；
  B. 用 `DEF_DET` 这个默认值在 holdout 上量 per-class 指标，与接线前的验收数字对表
     （整体 R 62.5% / P 84.2% / median IoU 0.681 / 脚 R 52.9%、P 100%、手 R 67.7%；conf 0.30 / imgsz 960）；
  C. 同一批裁图跑三次 `locate.py`：
       run1 = 不带 --det-weights（走 DEF_DET 默认路径）
       run2 = 显式 --det-weights <DEF_DET>
       run3 = 显式同一个值**再跑一次**（确定性控制组）
     然后 **整份 JSON 逐字段比对**（含 `points[].person_id`）。要证明的是"接默认值这件事本身
     没改变任何东西"，所以不能只比 hand/foot 框。run3 存在的意义：若 run2≠run3，任何 run1≠run2
     都不能归因于接线，而是模型/后处理本身不确定。

退出码：0 = 全过；1 = 有检查不过；2 = **person_id 单独不一致**（按下达时的约定：停手、不上报结论）。
另：若差异只出现在"运行元数据"（输出目录路径、时间戳之类），脚本会把它标成 META 并单独打印，
不当作接线失败——但那几条必须在汇报里逐条写出来，不许悄悄忽略。
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
sys.path.insert(0, HERE)

EXPECT = {'recall50': 62.5, 'precision50': 84.2, 'median_iou': 0.681, 'foot_recall50': 52.9,
          'foot_precision50': 100.0, 'hand_recall50': 67.7}
TOL = {'recall50': 0.6, 'precision50': 0.6, 'median_iou': 0.02, 'foot_recall50': 0.6,
       'foot_precision50': 0.6, 'hand_recall50': 0.6}


# ---------------------------------------------------------------- 工具
def load_rec(path, out_dir=None):
    """读 locate.py 的 JSON；若记录里嵌了输出目录路径，先归一化，避免把路径差异当行为差异。"""
    with io.open(path, encoding='utf-8') as fh:
        txt = fh.read()
    if out_dir:
        for v in {out_dir, out_dir.replace('\\', '/'), out_dir.replace('/', '\\')}:
            txt = txt.replace(v, '<OUT>')
    return json.loads(txt)


def walk_diff(a, b, prefix=''):
    """递归列举两份 JSON 的差异路径（值都为标量/容器），返回 list['路径: 左 → 右']。"""
    out = []
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return ['%s: 类型 %s → %s' % (prefix or '<root>', type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            p = '%s.%s' % (prefix, k) if prefix else k
            if k not in a:
                out.append('%s: 缺于左，右=%r' % (p, b[k]))
            elif k not in b:
                out.append('%s: 缺于右，左=%r' % (p, a[k]))
            else:
                out += walk_diff(a[k], b[k], p)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append('%s: 长度 %d → %d' % (prefix, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += walk_diff(x, y, '%s[%d]' % (prefix, i))
    else:
        if isinstance(a, float) or isinstance(b, float):
            if abs(float(a) - float(b)) > 1e-9:
                out.append('%s: %r → %r' % (prefix, a, b))
        elif a != b:
            out.append('%s: %r → %r' % (prefix, a, b))
    return out


META_RE = re.compile(r'(^|\.)((out|out_dir|src|src_dir|path|file|filepath|image|elapsed|ms|time|'
                     r'timestamp|date|host|pid))($|\.|\[)')


def split_meta(paths):
    meta = [p for p in paths if META_RE.search(p.split(':')[0])]
    real = [p for p in paths if p not in meta]
    return real, meta


def person_ids(rec):
    return [pt.get('person_id') for pt in rec.get('points', [])]


def run_locate(src, out_dir, extra, tag):
    cmd = [sys.executable, os.path.join(HERE, 'locate.py'),
           '--src'] + src + ['--out', out_dir, '--conf', '0.3', '--imgsz', '960', '--no-vitpose'] + extra
    print('  [%s] $ %s' % (tag, ' '.join('"%s"' % c if ' ' in c else c for c in cmd)))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit('locate.py 失败（tag=%s, returncode %d）' % (tag, p.returncode))
    return p.stdout


def cmp_dirs(dA, dB, src, label):
    """整份 JSON 逐字段比对两个目录，返回 (real_diffs, meta_diffs, n_cmp)。"""
    fa = sorted(f for f in os.listdir(dA) if f.endswith('.json'))
    fb = sorted(f for f in os.listdir(dB) if f.endswith('.json'))
    print('  %s: 左 %d 个 JSON / 右 %d 个 JSON（请求 %d 张）' % (label, len(fa), len(fb), len(src)))
    real, meta = [], []
    if len(fa) != len(fb) or len(fa) != len(src):
        real.append('JSON 数量不符：%d / %d / 请求 %d' % (len(fa), len(fb), len(src)))
    for x, y in zip(fa, fb):
        ra = load_rec(os.path.join(dA, x), dA)
        rb = load_rec(os.path.join(dB, y), dB)
        paths = walk_diff(ra, rb)
        if paths:
            r, m = split_meta(paths)
            real += ['%s | %s' % (x, p) for p in r]
            meta += ['%s | %s' % (x, p) for p in m]
        # person_id 单独看一眼，便于汇报
        pa, pb = person_ids(ra), person_ids(rb)
        if pa != pb:
            print('   ⚠ %s 的 person_id 序列不同：\n     左 %s\n     右 %s' % (x, pa, pb))
    return real, meta, len(fa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=6)
    ap.add_argument('--src', nargs='+', default=None,
                    help='给 C 段换一批图（默认用 holdout 前 6 张裁图；换产线整图可验证多人 person_id）')
    ap.add_argument('--tag', default='verify',
                    help='子目录名后缀，便于把"holdout 裁图"和"产线整图"两轮结果分开存')
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(PROJ)), 'team', 'reports', 'm2a-verifier', 'verify_default_det'))
    a = ap.parse_args()
    fails, meta_notes = [], []

    # ---- A. 默认值本身 ----
    print('== A. locate.py 的 DEF_DET ==')
    import locate
    det = locate.DEF_DET
    print('  DEF_DET = %s' % det)
    print('  文件存在: %s' % os.path.exists(det))
    if 'm2a_det_fix_v2' not in det:
        fails.append('A: DEF_DET 仍指向 %s' % os.path.basename(os.path.dirname(det)))
    if not os.path.exists(det):
        fails.append('A: DEF_DET 路径不存在')

    # ---- B. 用默认值在 holdout 上量指标 ----
    print('== B. 用 DEF_DET 默认值跑 holdout（conf 0.30 / imgsz 960）==')
    import numpy as np
    from ultralytics import YOLO
    import m2a_conf_sweep as S
    img_roots = [os.path.join(DATA, 'boxfix', 'ds', 'images'), os.path.join(DATA, 'boxfix2', 'ds', 'images')]
    items = S.load_items(os.path.join(DATA, 'boxfix_holdout.csv'), img_roots)
    per, fps = S.holdout_metrics(YOLO(det), items, S.DET_CLS, 0.30, 960)
    g_gt = sum(v['gt'] for v in per.values())
    g_hit = sum(v['hit'] for v in per.values())
    g_pred = sum(v['pred'] for v in per.values())
    g_ph = sum(v['pred_hit'] for v in per.values())
    g_ious = [x for v in per.values() for x in v['ious']]
    got = {'recall50': round(100.0 * g_hit / max(1, g_gt), 1),
           'precision50': round(100.0 * g_ph / max(1, g_pred), 1),
           'median_iou': round(float(np.median(g_ious)), 3),
           'foot_recall50': round(100.0 * per['foot']['hit'] / max(1, per['foot']['gt']), 1),
           'foot_precision50': round(100.0 * per['foot']['pred_hit'] / max(1, per['foot']['pred']), 1),
           'hand_recall50': round(100.0 * per['hand']['hit'] / max(1, per['hand']['gt']), 1)}
    print('  （命中 %d/%d 框，预测 %d 框；负样本 strict 误报 %s）' % (g_hit, g_gt, g_pred, fps))
    for k, want in EXPECT.items():
        g = got[k]
        ok = abs(g - want) <= TOL[k]
        print('  %-16s 期望 %7.3f  实测 %7.3f  %s' % (k, want, g, 'OK' if ok else ' FAIL'))
        if not ok:
            fails.append('B: %s 期望 %.3f 实测 %.3f' % (k, want, g))

    # ---- C. 默认路径 vs 显式 override vs 确定性控制组 ----
    print('== C. locate.py 整份 JSON 比对（含 points[].person_id）==')
    src = a.src if a.src else [it['img'] for it in items if it['verdict'] == 'draw'][:a.n_images]
    src = [os.path.abspath(p) for p in src]
    print('  C 段输入 %d 张：%s' % (len(src), ', '.join(os.path.basename(p) for p in src)))
    dD = os.path.join(a.out_dir, a.tag + '_default')
    dO = os.path.join(a.out_dir, a.tag + '_override')
    dC = os.path.join(a.out_dir, a.tag + '_override_ctl')
    for d in (dD, dO, dC):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith('.json'):
                os.remove(os.path.join(d, f))

    print('  run1：默认路径（不带 --det-weights）…')
    run_locate(src, dD, [], 'run1-default')
    print('  run2：显式 --det-weights 同一个权重 …')
    run_locate(src, dO, ['--det-weights', det], 'run2-override')
    print('  run3：显式同一个值再跑一次（确定性控制组）…')
    run_locate(src, dC, ['--det-weights', det], 'run3-ctl')

    real_ctl, meta_ctl, _ = cmp_dirs(dO, dC, src, 'run2 vs run3（控制组）')
    if real_ctl:
        fails.append('C-控制组：override 自身两次不一致 %d 处 ⇒ 后续差异不可归因于接线' % len(real_ctl))
        for x in real_ctl[:12]:
            print('   控制组差异 %s' % x)
    else:
        print('  控制组一致：同一权重两次运行逐字段相同 ⇒ 比对是确定性可用的')

    real, meta, n_cmp = cmp_dirs(dD, dO, src, 'run1 vs run2（默认 vs 显式）')
    meta_notes += meta
    for x in real:
        fails.append('C: %s' % x)
    print('  比对图片数 %d / 请求 %d；差异 %d 处（其中运行元数据 %d 处）'
          % (n_cmp, len(src), len(real), len(meta)))
    for x in meta[:10]:
        print('   元数据差异（不算行为差异，但必须上报）%s' % x)

    print('')
    if fails:
        only_pid = all(('person_id' in f) for f in fails if f.startswith('C'))
        print('== 未通过（%d 项）==' % len(fails))
        for f in fails:
            print('  - %s' % f)
        if only_pid:
            print('\n**差异只在 person_id**：按下达约定停手，不在这里下结论，立刻上报 lead。')
            return 2
        return 1
    print('== 全部通过：A 默认值指向 m2a_det_fix_v2；B holdout 数字与验收表一致；'
          'C 默认路径与显式传值逐字段相同（含 person_id），且控制组自身一致 ==')
    if meta_notes:
        print('注意：另有 %d 处运行元数据差异（见上），已单独列出。' % len(meta_notes))
    return 0


if __name__ == '__main__':
    sys.exit(main())
