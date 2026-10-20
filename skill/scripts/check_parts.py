#!/usr/bin/env python
"""check_parts.py —— 把 anime_detection_suite 的自检入口（locate.py）包成"别的 agent 直接能调"的一个命令。

回答的问题：这张（批）图里 **人/手/脚框、指趾数是否可疑、口塞/眼罩、绳子掩版** 有没有问题；
并且可以直接产出**逐人掩版**（做"逐人掩版重绘"时把某个角色的属性单独改掉，互不串扰）。

模型与代码**不复制**，全部按绝对路径调用现成套件（`--suite` 可覆盖，见 SKILL.md 的路径表）。

用法（任意 cwd 都行）：
  python check_parts.py --src G:\\path\\img.png
  ... --src <图目录> --mask                       # 手/脚掩版
  ... --person-masks                              # **逐人掩版**（SAM 框提示 → 该人的 silhouette）
  ... --recipe rec                                # 实测更好的配方（imgsz 1600 + 无泄漏三种子 M3）
  ... --json                                      # 只输出一行机器可读 JSON

逐人掩版的两份产物（每次每人一对，写在 `--mask-dir`，默认 <out>/mask）：
  <name>_person<i>.png        灰度：**255 = 要重绘的区域**（人），0 = 保留
  <name>_person<i>_rgba.png   RGBA：**重绘区 alpha=0**，可直接喂 ComfyUI LoadImage
                              （ComfyUI 的 MASK = 1 − alpha ⇒ 该人的像素正好变成重绘区）

退出码：0 = 跑完（**"可疑"不等于失败**，它只是"交给人复核"）；非 0 = 环境/调用出错
"""
import argparse
import collections
import glob
import io
import json
import os
import subprocess
import sys
import time

def _suite_def():
    """套件根目录：① 环境变量（下面 argparse 里优先取）② `_libs` 解析出的子库 ③ 仓库相对位置。"""
    try:
        from _libs import find_lib
        p = find_lib('anime-detection')
        if p:
            return p
    except Exception:                                   # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', '..', 'libs', 'anime-detection'),
                 os.path.join(here, '..', 'libs', 'anime-detection')):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return '<ANIME_DET_SUITE>'


SUITE_DEF = _suite_def()
PY_DEF = sys.executable or 'python'          # 产线推荐：ComfyUI 自己的 env（有 torch+ultralytics）
SAM_CKPT_DEF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'models', 'sams', 'sam_vit_b_01ec64.pth')
REC_IMGSZ = 1600
REC_M3 = ['m3_clean_v1_s20261010', 'm3_clean_v1_s20262010', 'm3_clean_v1_s20263010']
_SAM = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def die(msg, code=2):
    print('check_parts: %s' % msg, file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------- 调套件
def build_cmd(a):
    locate = os.path.join(a.suite, 'scripts', 'locate.py')
    if not os.path.exists(locate):
        die('找不到套件入口 %s（用 --suite 指向 anime_detection_suite 根目录）' % locate)
    if not os.path.exists(a.python):
        die('找不到解释器 %s：用 --python 指定一个装了 torch+ultralytics 的 python'
            '（见 libs/anime-detection/requirements.txt）' % a.python)
    # 子进程的 cwd 是套件目录 ⇒ 路径必须先转绝对，否则相对路径会指向错地方（探针实测踩到）
    cmd = ([a.python, '-X', 'utf8', locate, '--src'] + [os.path.abspath(s) for s in a.src]
           + ['--out', os.path.abspath(a.out)])
    cmd += ['--imgsz', str(a.imgsz if a.imgsz else (REC_IMGSZ if a.recipe == 'rec' else 1280))]
    if a.mask:
        cmd.append('--mask')
    if a.m3_conf is not None:
        cmd += ['--m3-conf', str(a.m3_conf)]
    if a.foot_crop:
        cmd += ['--m3-foot-crop', a.foot_crop]
    if a.no_foot_tight:
        cmd.append('--no-m3-foot-tight')
    if a.hand_weights:
        cmd += ['--hand-weights', a.hand_weights]
    weights = list(a.m3_weights or [])
    if not weights and a.recipe == 'rec':
        weights = [os.path.join(a.suite, 'runs', n, 'weights', 'best.pt') for n in REC_M3]
        missing = [w for w in weights if not os.path.exists(w)]
        if missing:
            die('配方 rec 需要的无泄漏权重缺失：%s' % missing)
    if weights:
        cmd += ['--m3-weights'] + weights
    return cmd


def expand_srcs(srcs):
    """把 --src 展开成具体图片文件列表（目录 → 目录下的图片），用于给逐人掩版找回原图。"""
    out = []
    exts = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
    for s in srcs:
        if os.path.isdir(s):
            for e in exts:
                out += sorted(glob.glob(os.path.join(s, e)))
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------- 汇总
def summarise(out_dir, ignore=(), seconds=None):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        if os.path.basename(p) == 'summary.json':
            continue
        d = json.load(io.open(p, encoding='utf-8'))
        flags = d.get('flags', [])
        boxes = d.get('boxes_all') or d.get('boxes', [])
        bad = ['%s#%s' % (f.get('part'), f.get('index')) for f in flags
               if f.get('type') == 'part_check' and f.get('verdict') not in ('ok', 'skip', None)
               and f.get('part') not in ignore]
        skipped = sum(1 for f in flags if f.get('type') == 'part_check' and f.get('verdict') == 'skip')
        tight = ['#%d' % (i + 1) for i, f in enumerate(
            f for f in flags if f.get('type') == 'foot_tight_check' and f.get('verdict') != 'ok')]
        gag = [{'cls': b['cls'], 'prob': b.get('prob'), 'box': [int(v) for v in b['box']],
                'person_id': b.get('person_id', -1)}
               for b in boxes if b.get('cls') in ('gag', 'blindfold') and b['cls'] not in ignore]
        parts = [f for f in flags if f.get('type') == 'part_check']
        n_hands = sum(1 for f in parts if f.get('part') == 'hand')
        n_feet = sum(1 for f in parts if f.get('part') == 'foot')
        no_hand = any(f.get('type') == 'no_hand_visible' for f in flags)
        no_person = any(f.get('type') == 'no_person' for f in flags)
        rope_path = os.path.join(os.path.dirname(out_dir), 'data', 'masks', 'rope', '%s_mask.png'
                                 % os.path.splitext(os.path.basename(d.get('image') or p))[0])
        rows.append(dict(
            image=os.path.basename(d.get('image') or p),
            json=os.path.relpath(p, os.path.dirname(out_dir)),
            n_people=d.get('n_people'), n_persons=d.get('n_persons'), n_skeletons=d.get('n_skeletons'),
            suspicious=bad, tight_flags=tight, gag_blindfold=gag,
            n_hands=n_hands, n_feet=n_feet,
            hands_state=('not_visible' if no_hand else ('zero_detected' if n_hands == 0 else 'visible')),
            feet_state=('not_visible' if (no_person and n_feet == 0) else
                        ('zero_detected' if n_feet == 0 else 'visible')),
            # rope 的语义：本入口**不做绳子检测**，只能报"rope 模块有没有留下掩版"
            rope={'mask': 'present' if d.get('ropes') else 'missing',
                  'path': (d.get('ropes') or {}).get('mask'),
                  'expected_path': rope_path,
                  'note': 'mask_missing = 本入口没发现 rope 模块的产物，**不代表图里没有绳**'},
            persons=[{'person_id': b.get('person_id'), 'box': [int(v) for v in b['box']],
                      'prob': b.get('prob')} for b in boxes if b.get('cls') == 'person'],
            skipped_parts=skipped, note=([('无人' if no_person else '')] if no_person else []) +
                                       (['看不到手'] if no_hand else [])))
    if seconds is not None and rows:
        per = round(seconds / len(rows), 1)
        for r in rows:
            r['seconds_total_run'] = round(seconds, 1)
            r['seconds_per_image_est'] = per
    return rows


# ---------------------------------------------------------------- 逐人掩版
COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def skeleton_mask(rec, chain, kps, shape, h, head_box=None):
    """**骨架膨胀**掩版：按 COCO-17 骨骼连线 + 关节画粗线（半径 ~0.045×身高，躯干用肩→髋四边形）。

    为什么不用凸包：凸包会把"骨架之间的空隙"整块吞掉（站姿两腿之间一直填到地面、臂与躯干之间也是），
    而且形状与人体不像；骨架膨胀贴着四肢走，天然不填这些空隙。
    头：有 head 框就按框画椭圆（关键点只有眼鼻耳，凸包盖不到头骨/头发）；没有就按鼻点画圆。
    """
    import cv2
    import numpy as np
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    if not kps:
        return m.astype(bool)
    by = {}
    for p in rec.get('points', []):
        if p.get('chain') == chain and p.get('vis', 0) >= 1:
            by[p.get('name', '').split('.')[-1]] = (int(p['x']), int(p['y']))
    r = max(8, int(0.045 * h))
    for a_, b_ in BONES:
        pa, pb = by.get(COCO17[a_]), by.get(COCO17[b_])
        if pa and pb:
            # 头/脸那几根（鼻-眼-耳）不用画粗线，交给下面的头部圆
            if a_ <= 4 and b_ <= 4:
                cv2.line(m, pa, pb, 1, thickness=max(3, r // 2))
            else:
                cv2.line(m, pa, pb, 1, thickness=2 * r)
    for _, pt in by.items():
        cv2.circle(m, pt, r, 1, -1)
    # 躯干：用"肩宽 → 髋宽"的四边形（比一条粗线更像人）
    sh = [by.get('left_shoulder'), by.get('right_shoulder')]
    hp = [by.get('left_hip'), by.get('right_hip')]
    if all(sh) and all(hp):
        quad = np.array([sh[0], sh[1], hp[1], hp[0]], np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(quad), 1)
        cv2.polylines(m, [quad], True, 1, thickness=max(6, int(0.03 * h)))
    # 头：优先用 head 框的内接椭圆（关键点凸包在头上只覆盖眼鼻耳那一小块）
    if head_box:
        import cv2 as _cv2
        hx = (head_box[0] + head_box[2]) / 2.0
        hy = (head_box[1] + head_box[3]) / 2.0
        _cv2.ellipse(m, (int(hx), int(hy)),
                     (max(4, int((head_box[2] - head_box[0]) * 0.56)),
                      max(4, int((head_box[3] - head_box[1]) * 0.56))), 0, 0, 360, 1, -1)
    else:
        nose = by.get('nose')
        if nose:
            cv2.circle(m, nose, max(14, int(0.075 * h)), 1, -1)
    return m.astype(bool)


def compose_person_mask(parts, kps, shape, a, box):
    """按覆盖策略把零件合成一张掩版，返回 (mask, 说明文字, 被剪掉的像素数)。

    模式（`--person-mask-cover`）：
      tight    = SAM ∩ 凸包护栏(10%)          —— 贴边、不碰道具
      skeleton = SAM ∪ 骨架膨胀                —— 贴着体型，不填骨架间空隙（**推荐，默认 auto 首选**）
      hull     = SAM ∪ 凸包(+3%)               —— 保证整条骨架在内，但会吃掉空隙
      box      = SAM ∪ 骨架膨胀 ∪ 检测框        —— 最粗（含道具），agent 做中/大特征时可用
      auto     = 依次试 skeleton → hull，取第一个"关键点覆盖 ≥0.95"的；都不行就用 box
    """
    import cv2
    import numpy as np
    from scipy import ndimage
    H, W = shape
    sam = parts.get('sam')
    if sam is None:
        sam = np.zeros((H, W), bool)
    skel, hull, bx, head = parts.get('skel'), parts.get('hull'), parts.get('box'), parts.get('head')

    def uni(*ms):
        out = None
        for m in ms:
            if m is None:
                continue
            out = m if out is None else (out | m)
        return out

    def cov(m):
        if not kps or m is None:
            return 0.0
        return sum(1 for x, y in kps if 0 <= y < H and 0 <= x < W and m[y, x]) / len(kps)

    base = None
    if len(kps) >= 3:
        base = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(base, cv2.convexHull(np.array(kps, np.int32)), 1)
        base = base.astype(bool)
    mode = a.person_mask_cover
    if mode == 'tight' and base is not None:
        guard = ndimage.binary_dilation(base, iterations=max(20, int(0.10 * (box[3] - box[1]))))
        if head is not None:                    # 头必须进护栏，否则贴边模式会把脑袋剪掉
            guard = guard | head
        out = sam & guard
        return out, 'tight', max(0, int(sam.sum()) - int(out.sum()))
    cand = [('skeleton', uni(skel, head)), ('hull', uni(hull, head)), ('box', bx)]
    if mode == 'auto':
        for nm, mm in cand:
            if mm is not None and cov(sam | mm) >= 0.95:
                return sam | mm, 'auto->%s' % nm, 0
        return sam | (bx if bx is not None else uni(skel, head)), 'auto->box(fallback)', 0
    pick = {'skeleton': uni(skel, head), 'hull': uni(hull, head), 'box': bx}.get(mode)
    if mode == 'box':                       # box 模式把骨架也带上（框里没骨架证据的部分也不吃亏）
        out = sam | (bx if bx is not None else False)
        return uni(out, skel, head), 'box', 0
    if pick is None:
        return sam, 'sam-only', 0
    return sam | pick, mode, 0


def feature_masks(rec, a, mask_dir, shape, img=None):
    """**特征掩版**：直接由"可信的检测框"生成常用的局部重绘掩版（比手搓更省事、也更容易复现）。

    来自 2026-10-13 第 2 份使用报告：`手框 / 脸框` 是可信的，而 **M1 的鼻/嘴细点在这类图上不可信**
    （鼻点落在嘴上、嘴位落在下巴）⇒ 重绘嘴不要用关键点推，**用脸框的下半部分**。
    产出（`<mask-dir>/feature/`）：
      hand0/foot0/head0/face0/gag0/blindfold0/eye0 … = 对应检测框（可 `--grow` 外扩）
      mouth0 … = **脸框下 42%**（横向内缩 12%）—— 由框推导，不依赖关键点
    每个还带 `source`（来自哪个框）写进 JSON，便于追溯。
    """
    import numpy as np
    from PIL import Image
    H, W = shape
    boxes = rec.get('boxes_all') or rec.get('boxes', [])
    out, counters = [], collections.Counter()
    want = ('hand', 'foot', 'left_foot', 'right_foot', 'head', 'face', 'gag', 'blindfold', 'eye')
    fdir = os.path.join(mask_dir, 'feature')
    os.makedirs(fdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(rec.get('image') or 'img'))[0]

    def emit(cls, i, box, source, rgba_too):
        x0, y0, x1, y1 = [int(max(0, min(v, d))) for v, d in zip(box, (W, H, W, H))]
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = 1
        if a.grow:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
            m = cv2.dilate(m, k, iterations=1)
        gp = os.path.join(fdir, '%s_%s%d.png' % (name, cls, i))
        Image.fromarray((m * 255), 'L').save(gp)
        rp = None
        if rgba_too and img is not None:
            rp = os.path.join(fdir, '%s_%s%d_rgba.png' % (name, cls, i))
            Image.fromarray(np.dstack([img, np.where(m > 0, 0, 255).astype(np.uint8)]), 'RGBA').save(rp)
        return dict(cls=cls, index=i, box=[x0, y0, x1, y1], mask=gp, rgba=rp,
                    area_frac=round(float((m > 0).mean()), 4), source=source)

    for b in boxes:
        cls = b.get('cls')
        if cls not in want:
            continue
        counters[cls] += 1
        r = emit(cls, counters[cls] - 1, b['box'], 'box:%s(person_id=%s)' % (cls, b.get('person_id')), a.feature_rgba)
        if r:
            out.append(r)
        if cls == 'face':                      # 由**检测框**推嘴（不用 M1 关键点：实测关键点会散落一脸）
            x0, y0, x1, y1 = [float(v) for v in b['box']]
            w, h = x1 - x0, y1 - y0
            # ① 有两只眼框 ⇒ 用**眼线角度**定朝向，把"眼往下 0.38 脸高"沿脸的正下方推
            ex = [e for e in boxes if e.get('cls') == 'eye' and
                  x0 - 0.2 * w <= (e['box'][0] + e['box'][2]) / 2 <= x1 + 0.2 * w and
                  y0 - 0.2 * h <= (e['box'][1] + e['box'][3]) / 2 <= y1 + 0.2 * h]
            if len(ex) >= 2:
                ex = sorted(ex, key=lambda e: (e['box'][0] + e['box'][2]) / 2)[:2]
                c = [((e['box'][0] + e['box'][2]) / 2, (e['box'][1] + e['box'][3]) / 2) for e in ex]
                import math
                th = math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0])
                vx, vy = -math.sin(th), math.cos(th)
                if vy < 0:                      # "下"要指向脸框底部那一侧
                    vx, vy = -vx, -vy
                mx = (c[0][0] + c[1][0]) / 2 + vx * 0.38 * h
                my = (c[0][1] + c[1][1]) / 2 + vy * 0.38 * h
                mw, mh = 0.44 * w, 0.22 * h
                mb = [mx - mw / 2, my - mh / 2, mx + mw / 2, my + mh / 2]
                src_txt = 'eye_line+face(θ=%.0f°)' % math.degrees(th)
            else:
                mb = [x0 + 0.12 * w, y0 + 0.58 * h, x1 - 0.12 * w, y1]
                src_txt = 'face_box_lower42%(只有一只眼框，退回几何)'
            r = emit('mouth', counters['mouth'], mb, src_txt, a.feature_rgba)
            counters['mouth'] += 1
            if r:
                out.append(r)
    rec['feature_masks'] = out
    return out


def _sam_predictor(ckpt):
    global _SAM
    if _SAM is None:
        from segment_anything import SamPredictor, sam_model_registry
        import torch
        if not os.path.isfile(ckpt):
            die('找不到 SAM 权重 %s（本 skill 自带 models/sams/sam_vit_b_01ec64.pth；'
                '或用 --person-mask-model hull 走关键点兜底）' % ckpt)
        sam = sam_model_registry['vit_b'](checkpoint=ckpt)
        sam.to('cuda' if torch.cuda.is_available() else 'cpu')
        sam.eval()
        _SAM = SamPredictor(sam)
    return _SAM


LAST_PID_PATH = []   # 诊断：最近一次 _instances() 每条链走的路径（'exact' / 'greedy'），不参与输出


def _instances(rec, a):
    """把"一个角色"定义清楚 —— **以 M1 骨架链为准**（一条链 = 一个角色），
    再补上"没有任何链、但置信度够高且不与别的框嵌套"的 person 框。

    为什么不能只按 M2a 的 person 框来：实测同一张图 M2a 会给 3 个框（0.90 / 0.44 / 0.31），
    低分的两个是重叠的伪框，而 M1 的两条骨架链才是真正的两个角色；按框逐个人做掩版，
    会得到重复且互相咬的掩版，第三条还是"没有骨架、只有框"的垃圾掩版。
    被丢掉的框会连同原因写进返回值，便于下游复核。
    """
    boxes = rec.get('boxes_all') or rec.get('boxes', [])
    persons = [b for b in boxes if b.get('cls') == 'person']
    chains = collections.defaultdict(list)
    cpid = collections.defaultdict(set)
    for p in rec.get('points', []):
        if p.get('vis', 0) >= 1 and p.get('chain'):
            chains[p['chain']].append((int(p['x']), int(p['y'])))
            v = p.get('person_id')
            if isinstance(v, int):
                cpid[p['chain']].add(v)
    inst, dropped = [], []
    used = set()
    LAST_PID_PATH[:] = []
    have_chain = any(len(k) >= 3 for k in chains.values())
    # ① 精确路径：locate 的 ViTPose 分支按"链 i ← 第 i 个人框"裁图，点上 `person_id` 写的就是它
    #    （套件 locate.py:290/629）。链上 vis>=1 的点口径一致、合法、且没被别的链占住 → 直接用。
    pid2idx = {}
    for i, b in enumerate(persons):
        v = b.get('person_id', i)
        if isinstance(v, int) and v not in pid2idx:
            pid2idx[v] = i
    pick, path = {}, {}
    for ch in sorted(chains):
        if len(chains[ch]) < 3 or len(cpid.get(ch, ())) != 1:
            continue
        idx = pid2idx.get(next(iter(cpid[ch])))
        if idx is not None and idx not in used:
            pick[ch], path[ch] = idx, 'exact'
            used.add(idx)
    # ② 兜底（老 JSON / person_id=-1 / 与别的链撞同一个 pid）：一对一几何贪心。
    #    多人身体重叠时 person 框会互相罩住对方的质心：逐链各自"取含心框里 prob 最高的"会把
    #    两条链判给同一个框（2026-10-20 双人重叠 20 张实测：17/44 条链错配、15/20 张图有碰撞，
    #    下游做出"两个人一根骨架"）。改成**一对一**：候选越少的链先抢，一个框只归属一条链
    #    （同 docs/runs.md:942 头框"最大且在框内"撞车的修法）。
    hits = collections.OrderedDict()
    for ch, kps in sorted(chains.items()):
        if len(kps) < 3:
            continue
        cx = sum(x for x, _ in kps) / len(kps)
        cy = sum(y for _, y in kps) / len(kps)
        hits[ch] = [i for i, b in enumerate(persons)
                    if b['box'][0] <= cx <= b['box'][2] and b['box'][1] <= cy <= b['box'][3]]
    for ch in sorted(hits, key=lambda c: (len(hits[c]),
                                         -max([persons[i].get('prob', 0) for i in hits[c]] or [0]))):
        if ch in pick:                 # ① 已经精确命中，不再参与抢
            continue
        for i in sorted(hits[ch], key=lambda i: -persons[i].get('prob', 0)):
            if i not in used:
                pick[ch] = i
                used.add(i)
                path[ch] = 'greedy'
                break
    LAST_PID_PATH[:] = [path.get(ch, 'none') for ch in sorted(chains) if len(chains[ch]) >= 3]
    for ci, (ch, kps) in enumerate(sorted(chains.items())):
        if len(kps) < 3:
            continue
        bi = pick.get(ch)
        if bi is not None:
            used.add(bi)
        inst.append(dict(chain=ch, person_id=(persons[bi].get('person_id') if bi is not None else -1),
                         box=[int(v) for v in (persons[bi]['box'] if bi is not None else
                                               (min(x for x, _ in kps), min(y for _, y in kps),
                                                max(x for x, _ in kps), max(y for _, y in kps)))],
                         kps=kps, src='M1%s' % ('+M2a' if bi is not None else '')))
    for i, b in enumerate(persons):
        if i in used:
            continue
        prob = float(b.get('prob') or 0)
        if prob < a.person_min_prob:
            dropped.append(dict(box=[int(v) for v in b['box']], prob=prob, why='low_prob'))
            continue
        big = False
        for j, o in enumerate(persons):
            if j == i:
                continue
            x0, y0, x1, y1 = o['box']
            a0 = max(0, min(x1, b['box'][2]) - max(x0, b['box'][0])) * \
                 max(0, min(y1, b['box'][3]) - max(y0, b['box'][1]))
            if a0 / max(1.0, (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])) > 0.7:
                big = True
                break
        if big:
            dropped.append(dict(box=[int(v) for v in b['box']], prob=prob, why='nested_in_other'))
            continue
        # **没有骨架支撑的 person 框默认不产出掩版**：实测餐厅那张图里，一个 prob=0.31 的伪框
        # （box 覆盖左侧整面墙）被做成了"一个人的掩版" —— 纯错标。有人脸/身体证据才值得出一张掩版。
        # 例外：整张图**一条链都没有**（pose 层失败）时仍然放行，否则工具在这类图上什么都给不出。
        if not it_ok_box_only(a, have_chain):
            dropped.append(dict(box=[int(v) for v in b['box']], prob=prob, why='no_skeleton'))
            continue
        inst.append(dict(chain=None, person_id=b.get('person_id', i),
                         box=[int(v) for v in b['box']], kps=[], src='M2a'))
    # 给每个实例配一个 head 框（M2a 的 head）：掩版的"头"零件与裁剪护栏都要用。
    # ⚠ 必须**一对一**按"离实例框中心最近"贪心分配 —— 两人靠在一起时 person 框与 head 框都会互相重叠，
    # 之前按"框最大且在实例框内"选，四个实例全都拿到了同一个头（2026-10-13 双人靠肩那张实测的坑）。
    heads = [b for b in boxes if b.get('cls') == 'head']
    pairs = []
    for ii, it in enumerate(inst):
        x0, y0, x1, y1 = it['box']
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for hi, h in enumerate(heads):
            hcx, hcy = (h['box'][0] + h['box'][2]) / 2.0, (h['box'][1] + h['box'][3]) / 2.0
            # 只考虑"头框中心大致在这个人的框里"的组合，再按距离排序
            if (x0 - 0.2 * (x1 - x0) <= hcx <= x1 + 0.2 * (x1 - x0)
                    and y0 - 0.2 * (y1 - y0) <= hcy <= y1 + 0.2 * (y1 - y0)):
                pairs.append(((hcx - cx) ** 2 + (hcy - cy) ** 2, ii, hi))
    pairs.sort()
    taken_i, taken_h = set(), set()
    for _, ii, hi in pairs:
        if ii in taken_i or hi in taken_h:
            continue
        taken_i.add(ii)
        taken_h.add(hi)
        inst[ii]['head_box'] = [int(v) for v in heads[hi]['box']]
    # 顺手把该角色的 face 框也记上（掩版的"头"用它推椭圆，比 head 框紧）
    faces = [b for b in boxes if b.get('cls') == 'face']
    for it in inst:
        x0, y0, x1, y1 = it['box']
        hbx = it.get('head_box') or it['box']
        cand = [f for f in faces
                if hbx[0] - 0.2 * (hbx[2] - hbx[0]) <= (f['box'][0] + f['box'][2]) / 2 <= hbx[2] + 0.2 * (hbx[2] - hbx[0])
                and hbx[1] - 0.2 * (hbx[3] - hbx[1]) <= (f['box'][1] + f['box'][3]) / 2 <= hbx[3] + 0.2 * (hbx[3] - hbx[1])]
        it['face_box'] = ([int(v) for v in max(cand, key=lambda f: (f['box'][2] - f['box'][0])
                                               * (f['box'][3] - f['box'][1]))['box']] if cand else None)
    for it in inst:
        it.setdefault('head_box', None)
    return inst, dropped


def it_ok_box_only(a, have_chain):
    """无骨架的 person 框是否放行：默认"图里有链就跳过它"，`--allow-box-only-persons` 强制放行。"""
    return bool(a.allow_box_only_persons) or not have_chain


def person_masks(img_path, rec, a, mask_dir):
    """每个"角色"一条 silhouette（见 `_instances` 的划分规则）。"""
    import cv2
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    inst, dropped = _instances(rec, a)
    src = Image.open(img_path).convert('RGB')
    img = np.asarray(src)
    H, W = img.shape[:2]
    name = os.path.splitext(os.path.basename(img_path))[0]
    pred = _sam_predictor(a.sam_ckpt) if (a.person_mask_model == 'sam' and inst) else None
    if pred is not None:
        pred.set_image(img)
    raw = []
    for it in inst:
        kps = it['kps']
        x0, y0, x1, y1 = it['box']
        method = a.person_mask_model
        prompt = 'box'
        m = None
        if pred is not None:
            # 提示策略（2026-10-13 实测）：只用**检测框**当 box 提示时，SAM 会把"人坐着的椅子/雪橇"
            # 一起算进人（面积 10% vs 6%）；改成 **骨架外接框 + 躯干正样本点** 后，椅子被正确排除。
            pos = []
            if a.person_mask_prompt == 'pose' and len(kps) >= 3:
                want = ('neck', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'nose')
                pos = [(int(p['x']), int(p['y'])) for p in rec.get('points', [])
                       if p.get('chain') == it['chain'] and p.get('name', '').split('.')[-1] in want
                       and 0 <= int(p['y']) < H and 0 <= int(p['x']) < W]
                hx, hy, hw, hh = cv2.boundingRect(cv2.convexHull(np.array(kps, np.int32)))
                pad = int(0.10 * max(hw, hh))
                pbox = [max(0, hx - pad), max(0, hy - pad), min(W, hx + hw + pad), min(H, hy + hh + pad)]
                prompt = 'pose'
            else:
                pbox = [x0, y0, x1, y1]
            kwargs = dict(box=np.array(pbox, np.float32)[None, :], multimask_output=True)
            if pos:
                kwargs['point_coords'] = np.array(pos, np.float32)
                kwargs['point_labels'] = np.ones(len(pos), np.int32)
            masks, scores, _ = pred.predict(**kwargs)
            best, bkey = None, None
            for mm, s in zip(masks, scores):
                ok = all(bool(mm[y, x]) for x, y in pos) if pos else True
                # 规则：优先"把提示点都包住"的候选，其中取面积最小者（避免把背景/道具一起框进来）；
                #       都不满足则退回 SAM 自己的分数最高者
                key = (1, -int(mm.sum()), float(s)) if ok else (0, float(s), 0)
                if bkey is None or key > bkey:
                    best, bkey = mm, key
            m = np.asarray(best, bool)
            # 只保留与提示点/框中心相连的那一块，避免粘一块背景
            lab, n = ndimage.label(m)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            c = lab[cy, cx] if 0 <= cy < H and 0 <= cx < W else 0
            if c == 0:
                for (x, y) in (pos or kps):
                    if 0 <= y < H and 0 <= x < W and lab[y, x]:
                        c = lab[y, x]
                        break
            if c == 0:
                sizes = ndimage.sum(m, lab, range(1, n + 1))
                c = int(np.argmax(sizes)) + 1 if n > 1 else 0
            m = (lab == c)
            if m.sum() < 0.02 * max(1, (x1 - x0) * (y1 - y0)):   # 太小 ⇒ 认输，退回关键点凸包
                method, m = 'hull(sam_failed)', None
        if m is None:                                            # 兜底：关键点凸包
            pts = np.array(kps, np.int32) if len(kps) >= 3 else np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
            if len(kps) < 3:                                     # 连关键点都不够 ⇒ 只能用检测框
                m = np.zeros((H, W), np.uint8)
                m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1
                method, prompt = 'box', 'box'
            m = m.astype(bool)
            grow_min = int(0.12 * (y1 - y0))
            if a.grow < grow_min:
                m = ndimage.binary_dilation(m, iterations=grow_min)
        # ---- 掩版"零件"（2026-10-13：用户要 agent-in-the-loop，所以不再只给一张成品）----
        #   sam    = SAM 的轮廓（最贴边，但可能漏掉下垂的头发/被挡的肢体）
        #   skel   = **骨架膨胀**（按骨骼连线 + 关节画粗线，比凸包更贴体型：不填腿间/臂间空隙）
        #   hull   = 关键点凸包 +3% 外扩（保证整条骨架在内，但会吃掉骨架间空隙）
        #   box    = M2a 的 person 框（最粗，含道具；agent 做中/大特征时可用）
        parts = {'sam': m}
        base = None
        if len(kps) >= 3:
            base = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(base, cv2.convexHull(np.array(kps, np.int32)), 1)
            parts['hull'] = ndimage.binary_dilation(base, iterations=max(10, int(0.03 * (y1 - y0)))).astype(bool)
        parts['skel'] = skeleton_mask(rec, it['chain'], kps, (H, W), y1 - y0, it.get('head_box'))
        # head 单独一个零件：**关键点凸包只包住"眼鼻耳"那三点，根本盖不到头骨/头发**
        # ⇒ 逐人掩版必须显式把头框（M2a 的 head）并进来，否则"改这个人的脸/发型"会漏掉脑袋
        # （2026-10-13 实测：双人靠肩那张，头框在掩版里的覆盖率只有 0.34/0.44）。用头框的内接椭圆，比矩形像人。
        hb = it.get('head_box')
        if hb:
            # 头部掩版 **一律用 M2a 头框的内接椭圆**（用户 2026-10-12 定的口径）：
            # 头部检测框实测很准，而原来优先用的 M2b face 框（更紧）在**强重叠姿态**（膝枕/拥抱）下
            # 会被配到错的人身上（实测：face 框落到了躺着那位的头上）⇒ 用头框更稳。
            # 半径严格内切（0.50×框），再与"该角色自身轮廓膨胀一点"求交 —— 否则头框偏大时
            # 椭圆会鼓出成一圈"光环"，两人靠近时光环交叉（用户 2026-10-13 报过"交叉区域很奇怪"）。
            bx0, by0, bx1, by1 = hb
            he = np.zeros((H, W), np.uint8)
            hx, hy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            ax, ay = (bx1 - bx0) * 0.50, (by1 - by0) * 0.50
            cv2.ellipse(he, (int(hx), int(hy)), (max(4, int(ax)), max(4, int(ay))), 0, 0, 360, 1, -1)
            he = he.astype(bool)
            ref = parts.get('sam') if parts.get('sam') is not None and parts['sam'].any() else parts.get('hull')
            if ref is not None and ref.any():
                he &= ndimage.binary_dilation(ref, iterations=max(10, int(0.03 * (y1 - y0))))
            parts['head'] = he
            it['head_mask_src'] = 'M2a_head_box_inscribed_ellipse'
        bm = np.zeros((H, W), bool)
        bm[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        parts['box'] = bm
        raw.append(dict(it, m=m, parts=parts, method=method, prompt=prompt))

    # ---- 合并"其实是同一个角色"的实例 ----
    # 实测（餐厅那张）：M1 会把一个角色**拆成两条链**（上身一条、腿一条），两条链的掩版大量重叠
    # ⇒ 按"重叠 > 较小掩版的 50%"合并成一个人，否则同一角色会出两份互相咬的掩版。
    parent = list(range(len(raw)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merges = []
    if a.merge_overlap:
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                mi, mj = raw[i]['m'], raw[j]['m']
                inter = int((mi & mj).sum())
                if not inter:
                    continue
                # 身份门槛：只有"同一个人"才允许合并（M1 把一个角色拆成上身/腿两条链时 pid 相同；
                # 两个贴身的人 pid 不同）。少了这道门槛，两人轮廓 95%+ 重叠就会被并回一个人 ——
                # 改前就是这样：4/6 张双人图默认只出 1 个掩版。两边都 -1（老 JSON / 无 pid）时仍按
                # 纯几何合并，保持旧行为。
                if raw[i].get('person_id', -1) != raw[j].get('person_id', -1):
                    continue
                if inter > 0.5 * min(int(mi.sum()), int(mj.sum())):
                    a_, b_ = find(i), find(j)
                    if a_ != b_:
                        parent[b_] = a_
                        merges.append([raw[i]['chain'] or 'M2a%d' % i, raw[j]['chain'] or 'M2a%d' % j])
    groups = collections.OrderedDict()
    for i, r in enumerate(raw):
        groups.setdefault(find(i), []).append(r)

    parts_dir = os.path.join(mask_dir, 'parts')
    built = []
    for gi, members in enumerate(groups.values()):
        # 每个零件按组取并集，再按覆盖策略**在组这一级**合成（合并后关键点更多，策略可能变化）
        p = {}
        for r in members:
            for k, v in r['parts'].items():
                p[k] = (p[k] | v) if k in p else v
        kps = [pt for r in members for pt in r['kps']]
        mbox = [min(r['box'][0] for r in members), min(r['box'][1] for r in members),
                max(r['box'][2] for r in members), max(r['box'][3] for r in members)]
        m, covermode, clipped = compose_person_mask(p, kps, (H, W), a, mbox)
        if not a.no_fill_holes:
            m = ndimage.binary_fill_holes(m)
        if a.grow > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
            m = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)
        tag = 'person%d' % gi
        built.append(dict(tag=tag, m=m, parts=p, kps=kps, box=mbox, covermode=covermode,
                          clipped=clipped, members=members))

    # ---- 两人**贴在一起**时，把重叠像素判给"更深在谁身体里"的那一位 ----
    # 实测（2026-10-13 双人靠肩那张）：两个 person 框重叠 116 px，用户只能自己"按重叠中点切一刀"。
    # 第一版我照抄了"垂直平分线"（= 直线切），用户指出**接缝看起来很奇怪**：两人身体交错时，
    # 直线会把胳膊/头发切出一道横贯的十字。改成按**掩版内部深度**（到各自边界的距离变换）判归属，
    # 接缝自然沿两人身体的中介线走，像人手工描的分界线。
    overlaps_resolved = []
    if a.exclusive_persons and len(built) > 1:
        for i in range(len(built)):
            for j in range(i + 1, len(built)):
                mi, mj = built[i]['m'], built[j]['m']
                inter = mi & mj
                n_inter = int(inter.sum())
                if not n_inter:
                    continue
                if a.overlap_split == 'bisector':       # 旧行为（保留作对照）
                    ci = ((built[i]['box'][0] + built[i]['box'][2]) / 2.0,
                          (built[i]['box'][1] + built[i]['box'][3]) / 2.0)
                    cj = ((built[j]['box'][0] + built[j]['box'][2]) / 2.0,
                          (built[j]['box'][1] + built[j]['box'][3]) / 2.0)
                    ys, xs = np.nonzero(inter)
                    vx, vy = cj[0] - ci[0], cj[1] - ci[1]
                    den = vx * vx + vy * vy
                    if den < 1e-6:
                        mi &= ~inter
                    else:
                        t = ((xs - ci[0]) * vx + (ys - ci[1]) * vy) / den
                        to_j = t >= 0.5
                        mi[ys[~to_j], xs[~to_j]] = False
                        mj[ys[to_j], xs[to_j]] = False
                    how = 'perpendicular_bisector'
                else:                                    # 默认：按内部深度判归属
                    di = ndimage.distance_transform_edt(mi)
                    dj = ndimage.distance_transform_edt(mj)
                    ys, xs = np.nonzero(inter)
                    # 谁那里"更靠里"（到边界的距离更大）就归谁；平手时归先来的那个
                    to_j = dj[ys, xs] > di[ys, xs]
                    mi[ys[~to_j], xs[~to_j]] = False
                    mj[ys[to_j], xs[to_j]] = False
                    how = 'inside_depth'
                overlaps_resolved.append({'a': built[i]['tag'], 'b': built[j]['tag'],
                                          'px': n_inter, 'how': how})

    out = []
    for b in built:
        m, tag, members = b['m'], b['tag'], b['members']
        kps = b['kps']
        kp_in = sum(1 for x, y in kps if 0 <= y < H and 0 <= x < W and m[y, x])
        cover = round(kp_in / len(kps), 2) if kps else None
        os.makedirs(mask_dir, exist_ok=True)
        gp = os.path.join(mask_dir, '%s_%s.png' % (name, tag))
        Image.fromarray((m * 255).astype(np.uint8), 'L').save(gp)
        rp = os.path.join(mask_dir, '%s_%s_rgba.png' % (name, tag))
        Image.fromarray(np.dstack([img, np.where(m, 0, 255).astype(np.uint8)]), 'RGBA').save(rp)
        part_files = {}
        if a.save_parts:
            os.makedirs(parts_dir, exist_ok=True)
            for k, v in b['parts'].items():
                pf = os.path.join(parts_dir, '%s_%s_%s.png' % (name, tag, k))
                Image.fromarray((v * 255).astype(np.uint8), 'L').save(pf)
                part_files[k] = pf
        out.append(dict(
            tag=tag, mask=gp, rgba=rp, parts=part_files or None,
            area_frac=round(float(m.mean()), 4), pose_coverage=cover,
            prompt=members[0]['prompt'], method=members[0]['method'],
            cover_mode=b['covermode'], clipped_to_pose_px=b['clipped'], supported_by_pose=bool(kps),
            person_id=(members[0]['person_id'] if len(members) == 1 else
                       [r['person_id'] for r in members]),
            chain=(members[0]['chain'] if len(members) == 1 else [r['chain'] for r in members]),
            source=members[0]['src'], box=members[0]['box'],
            grows_px=a.grow, filled_holes=not a.no_fill_holes,
            merged_from=([r['chain'] or 'M2a' for r in members] if len(members) > 1 else None)))
    if len(out) > 1:
        ms = [np.asarray(Image.open(o['mask']), np.uint8) > 0 for o in out]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                inter = int((ms[i] & ms[j]).sum())
                if inter:
                    out[j].setdefault('overlap_with', {})[out[i]['tag']] = inter
    rec['persons_masks'] = out
    rec['person_boxes_dropped'] = dropped
    rec['person_instances_merged'] = merges
    rec['person_overlaps_resolved'] = overlaps_resolved
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--src', nargs='+', required=True, help='图片文件或目录（可多个）')
    ap.add_argument('--out', default=None, help='输出目录（默认 ./part_check_out）')
    ap.add_argument('--recipe', choices=['prod', 'rec'], default='prod',
                    help='prod=现产线口径（imgsz 1280 + 现役 M3 双权重）；rec=实测更好（imgsz 1600 + 无泄漏 M3 三种子）')
    ap.add_argument('--imgsz', type=int, default=None, help='覆盖检测分辨率')
    ap.add_argument('--m3-conf', type=float, default=None, help='M3 的 p(bad) 阈值（默认 0.5）')
    ap.add_argument('--foot-crop', choices=['det', 'pose'], default=None)
    ap.add_argument('--hand-weights', default=None, help='可选：专用手检测器（runs/m2a_hand_v2/weights/best.pt）')
    ap.add_argument('--m3-weights', nargs='+', default=None, help='覆盖 M3 权重（绝对路径）')
    ap.add_argument('--no-foot-tight', action='store_true', help='关掉脚的收紧复核旗标')
    ap.add_argument('--mask', action='store_true', help='额外导出手/脚掩版 PNG（套件行为）')
    ap.add_argument('--person-masks', action='store_true', dest='person_masks',
                    help='导出**逐人掩版**（灰度 255=重绘区 + ComfyUI 语义 RGBA）')
    ap.add_argument('--person-mask-model', choices=['sam', 'hull'], default='sam',
                    help='sam（默认，box 提示 + 关键点校验）；hull=只用 M1 关键点凸包（不需要 SAM）')
    ap.add_argument('--person-mask-prompt', choices=['pose', 'box'], default='pose',
                    help='pose（默认）= 骨架外接框 + 躯干正样本点（实测能把椅/雪橇排除）；'
                         'box = 只用检测器 person 框（会把坐着的人下面的道具一起包进来）')
    ap.add_argument('--person-mask-cover', choices=['auto', 'tight', 'skeleton', 'hull', 'box'],
                    default='auto',
                    help='掩版覆盖策略（默认 auto：依次试 骨架膨胀→凸包，取第一个"关键点覆盖≥0.95"的，'
                         '都不行就用检测框）。skeleton=骨架膨胀（推荐，贴体型、不填骨架间空隙）；'
                         'hull=凸包；box=连检测框一起（最粗）；tight=贴边不碰道具。'
                         '这套流程的使用者多是 agent、改的是中/大尺寸特征，"只改半个人"比"多盖一点"糟得多')
    ap.add_argument('--no-save-parts', action='store_false', dest='save_parts', default=True,
                    help='不把掩版零件（sam/skel/hull/box）单独存盘（默认存：agent 可自己组合）')
    ap.add_argument('--feature-masks', action='store_true', dest='feature_masks',
                    help='导出**特征掩版**：hand/foot/head/face/gag/blindfold/eye 的检测框 + '
                         '由脸框推出来的 mouth（下 42%%）—— 局部重绘常用，比手搓掩版可复现')
    ap.add_argument('--feature-rgba', action='store_true', dest='feature_rgba',
                    help='特征掩版也写一份 ComfyUI 语义的 RGBA（默认只写灰度）')
    ap.add_argument('--person-min-prob', type=float, default=0.5,
                    help='低于该置信度的 person 框**不做掩版**（默认 0.5：实测伪框给到 0.44/0.31，'
                         '做了也是重复的、互相咬的掩版）；被丢的框与原因写在 JSON 的 person_boxes_dropped')
    ap.add_argument('--no-merge-overlap', action='store_false', dest='merge_overlap', default=True,
                    help='不合并"掩版大量重叠"的实例（默认会合并：M1 有时把一个角色拆成两条链，'
                         '不合并就会出两份互相咬的掩版）')
    ap.add_argument('--no-exclusive-persons', action='store_false', dest='exclusive_persons', default=True,
                    help='不把"两人贴在一起"的重叠像素分给更近的那位（默认会分：按两人框中心的垂直平分线，'
                         '等价于用户手动的"按重叠中点切一刀"，保证**逐人掩版互不重叠**）')
    ap.add_argument('--overlap-split', choices=['depth', 'bisector'], default='depth',
                    help='重叠怎么分：depth（默认）=按**掩版内部深度**（到各自边界的距离变换）判归属，'
                         '接缝沿两人身体的中介线走；bisector=两人框中心的垂直平分线（直线切，接缝会横贯身体）')
    ap.add_argument('--allow-box-only-persons', action='store_true', dest='allow_box_only_persons',
                    default=False,
                    help='连"没有骨架支撑的 person 框"也出掩版（默认不出：实测这类框会把墙壁/布景'
                         '当成一个人 ⇒ 纯错标；整张图没有骨架时仍会自动放行）')
    ap.add_argument('--mask-dir', default=None, help='掩版输出目录（默认 <out>/mask）')
    ap.add_argument('--grow', type=int, default=15, help='掩版外扩像素（默认 15，给重绘留接缝余量）')
    ap.add_argument('--no-fill-holes', action='store_true', dest='no_fill_holes',
                    help='不填内部孔洞（**不建议**：人身上的洞会把该处冻结住）')
    ap.add_argument('--no-clip-to-pose', action='store_false', dest='clip_to_pose', default=True,
                    help='不把掩版裁到骨架轮廓（默认会裁：坐着的人下面的椅子/雪橇、手里道具通常不该跟着重绘）')
    ap.add_argument('--ignore-cls', nargs='*', default=[], help='汇总时忽略这些类（如 blindfold）')
    ap.add_argument('--sam-ckpt', default=os.path.abspath(SAM_CKPT_DEF))
    ap.add_argument('--json', action='store_true', help='只打印一行机器可读 JSON')
    ap.add_argument('--suite', default=os.environ.get('ANIME_DET_SUITE', SUITE_DEF))
    ap.add_argument('--python', default=os.environ.get('ANIME_DET_PYTHON', PY_DEF))
    a = ap.parse_args()
    a.out = os.path.abspath(a.out or os.path.join(os.getcwd(), 'part_check_out'))
    a.mask_dir = os.path.abspath(a.mask_dir or os.path.join(a.out, 'mask'))
    os.makedirs(a.out, exist_ok=True)
    cmd = build_cmd(a)
    if not a.json:
        print('$ %s' % ' '.join('"%s"' % c if ' ' in c else c for c in cmd), flush=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=a.suite, env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print((r.stdout or '')[-2000:], file=sys.stderr)
        die('locate.py 退出码 %d：%s' % (r.returncode, (r.stderr or '')[-400:]), 1)

    if a.person_masks:
        src_map = {os.path.basename(p): p for p in expand_srcs(a.src)}
        for p in sorted(glob.glob(os.path.join(a.out, '*.json'))):
            if os.path.basename(p) == 'summary.json':
                continue
            rec = json.load(io.open(p, encoding='utf-8'))
            img_name = rec.get('image')
            src_img = src_map.get(os.path.basename(img_name or ''))
            if src_img is None or not os.path.exists(src_img):
                print('  !! 找不到原图 %s，跳过逐人掩版' % img_name, file=sys.stderr)
                continue
            rec['persons_masks'] = person_masks(src_img, rec, a, a.mask_dir)
            with io.open(p, 'w', encoding='utf-8') as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)

    if a.feature_masks:
        import numpy as _np
        from PIL import Image as _Im
        for p in sorted(glob.glob(os.path.join(a.out, '*.json'))):
            if os.path.basename(p) == 'summary.json':
                continue
            rec = json.load(io.open(p, encoding='utf-8'))
            src_map = {os.path.basename(x): x for x in expand_srcs(a.src)}
            sp = src_map.get(os.path.basename(rec.get('image') or ''))
            if not sp:
                continue
            im = _np.asarray(_Im.open(sp).convert('RGB'))
            feature_masks(rec, a, a.mask_dir, im.shape[:2], im)
            with io.open(p, 'w', encoding='utf-8') as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)

    elapsed = time.time() - t0
    rows = summarise(a.out, set(a.ignore_cls), seconds=elapsed)
    with io.open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(recipe=a.recipe, imgsz=a.imgsz or (REC_IMGSZ if a.recipe == 'rec' else 1280),
                       suite=a.suite, mask_dir=a.mask_dir if (a.person_masks or a.feature_masks) else None,
                       seconds_total=round(elapsed, 1), images=rows), fh, ensure_ascii=False, indent=1)
    if a.json:
        print(json.dumps(dict(out=a.out, mask_dir=a.mask_dir if a.person_masks else None,
                              seconds=round(elapsed, 1), images=rows), ensure_ascii=False))
        return
    print('\n%-34s %-4s %-20s %-10s %-14s %s' % ('image', '人', '可疑部位(主判定)', '收紧复核', '口塞/眼罩', '备注'))
    print('-' * 112)
    for x in rows:
        extra = list(x['note'])
        if x['rope']['mask'] == 'present':
            extra.append('绳掩版:有')
        if x['skipped_parts']:
            extra.append('数不了%d处' % x['skipped_parts'])
        if x['gag_blindfold']:
            extra.append('口塞/眼罩: ' + ','.join('%s(%.2f)' % (g['cls'], g['prob'] or 0)
                                                 for g in x['gag_blindfold']))
        if x.get('persons_masks'):
            extra.append('逐人掩版 ' + ','.join('p%d %.0f%%' % (m['person_id'], 100 * m['area_frac'])
                                              for m in x['persons_masks']))
        print('%-34s %-4s %-20s %-10s %-14s %s' % (
            x['image'][:34], x['n_persons'] if x['n_persons'] is not None else x['n_people'],
            ','.join(x['suspicious']) or '-', ','.join(x['tight_flags']) or '-',
            x['hands_state'] if x['n_hands'] == 0 else '%d 手' % x['n_hands'], ' '.join(extra)))
    print('-' * 112)
    print('%d 张，用时 %.1fs（%.1fs/张）；主判定可疑 %d 张、收紧复核另有 %d 张' % (
        len(rows), elapsed, elapsed / max(1, len(rows)),
        sum(1 for x in rows if x['suspicious']),
        sum(1 for x in rows if x['tight_flags'] and not x['suspicious'])))
    print('契约 JSON 与汇总：%s%s（可疑 = 交给人复核，不是"判定失败"）' % (
        a.out, '；逐人掩版：%s' % a.mask_dir if a.person_masks else ''))


if __name__ == '__main__':
    main()
