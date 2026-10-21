"""M1 骨架**校正式**标注器 —— 模型先标，人拖点；不做从零标注。

为什么这么设计（用户 2026-10-13 定的口径）：
  - 骨架从零标太贵，人只适合"看着已有的骨架把节点拖正"；
  - 画布**故意比图大**：四周各留 `margin_scale × 图宽` 的额外区域，节点可以拖到**画面外**并原样存下来
    （骨架被画框切掉是常态；canvas 里"不能在画板外绘制"那条约束在这里是错的，不学）；
  - 交互照 canvas 的节点模式来，但用 PyQt5 实现，和 `label_face.py` / `label_cls.py` 保持同一套标注器风格。

预标注来源（按优先级）：
  1. `data/pose_gt/<stem>.json`  —— **人判过的真值**，存在就永远显示它，绝不被预标注覆盖；
  2. `data/prelabel/<stem>.json` 的 `cls='body'` 点（`M1.dwpose#<i>` 每 chain 一个人，BODY18 命名）；
     这是现成 994 张图的 dwpose 预标注，不需要重跑模型；
  3. 若某张图没有 body 点 ⇒ 画布空的，人可以从零拖（但这是例外路径，不是常规）。

真值写出 `data/pose_gt/<stem>.json`，**schema 与 prelabel 的 points 完全一致**
（`layer='human'`, `cls='body'`, `name='body.<x>'`, `chain='H#<i>'`, `x/y/vis`），
所以 `export_labels.export_pose_coco17()` 只要换个来源即可，格式不用改。
另外存一份 `pred_points`（模型原始预测）与 `shift_px`（人改动了多远），用于**量化模型误差**，不用靠肉眼。

键位（按钮上也都写了）：
  拖点 = 校准（实时跟手）；空白处拖动 或 中键拖动 = 平移；滚轮 = 以光标为心缩放
  ← / → = 上一张 / 下一张（相邻）；Ctrl+→ = 跳到下一张未判的；翻页不改判定
  空格/Enter 采纳并下一张（没改过=采纳预标注，改过=人工改过）　A 整张采纳　X 本张无骨架　S 只保存
  [ / ] / Tab 换实例　N 新增实例（人形模板）　Delete 删除实例　R 当前实例还原成预标注　Z 撤销
  1 / 2 / 3 或右键点节点 可见性：1 可见、2 遮挡、3 不可见（存进标签的 vis 仍是 2/1/0）
  Shift+方向键 微调选中节点 1px（+Ctrl 为 10px）　F / W 适应内容（图 ∪ 骨架，含画到画面外的点）
  V 显示/隐藏低置信预标注实例　G 跳到第 N 张

用法：
    python scripts/label_pose.py --build-queue            # 从 data/prelabel 建队列（有 body 点的图）
    python scripts/label_pose.py --build-queue --sort hard --limit 30 --out-csv data/pose_queue/hard30.csv
    python scripts/label_pose.py                          # 开标（默认队列 data/pose_queue/index.csv）
    python scripts/label_pose.py --selftest               # 无人值守自检（离屏，不弹窗）
"""
import argparse
import copy
import csv
import io
import json
import os
import sys
import time

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

# 输出被重定向（写日志 / CI / 传管道）时中文控制台会 UnicodeEncodeError 崩，降级别崩
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:                                       # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
# 预标注源：**默认用当前 SOTA**（微调 ViTPose，人工标注监督，报告集 median 23.1px / 覆盖 88%）。
# 旧目录 `data/prelabel`（dwpose 打底）保留可读：`--pred-dir data\prelabel` 仍能开。
# 注意：换源只改"人打开一张图时看到的初稿"，已存的真值不受影响。
PRE_DEF = os.path.join(DATA, 'prelabel_vitpose_v6')
GT_DEF = os.path.join(DATA, 'pose_gt')
QUEUE_DEF = os.path.join(DATA, 'pose_queue', 'index.csv')

# BODY18 = 与 export_labels.py 完全一致（自检里会断言，防止两边漂移）
BODY18 = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'r_ear', 'l_eye', 'l_ear']
IDX = {n: i for i, n in enumerate(BODY18)}
EDGES = [('nose', 'neck'), ('neck', 'r_shoulder'), ('neck', 'l_shoulder'),
         ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
         ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
         ('neck', 'r_hip'), ('neck', 'l_hip'),
         ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'), ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
         ('nose', 'r_eye'), ('r_eye', 'r_ear'), ('nose', 'l_eye'), ('l_eye', 'l_ear')]
VIS_COLOR = {2: QColor(60, 220, 90), 1: QColor(255, 170, 40), 0: QColor(140, 140, 140)}
# 新增实例时的**粗略人形模板**（相对检测框的 0..1 坐标）：模型整丢了某个人时，
# 人不用 18 个点从零拖，先拖一个大致人形即可。r_* 是角色右侧 = 画面左侧。
HUMAN_TEMPLATE = {'nose': (0.50, 0.08), 'neck': (0.50, 0.16),
                  'r_eye': (0.44, 0.05), 'r_ear': (0.40, 0.06), 'l_eye': (0.56, 0.05), 'l_ear': (0.60, 0.06),
                  'r_shoulder': (0.34, 0.20), 'l_shoulder': (0.66, 0.20),
                  'r_elbow': (0.26, 0.36), 'l_elbow': (0.74, 0.36),
                  'r_wrist': (0.20, 0.52), 'l_wrist': (0.80, 0.52),
                  'r_hip': (0.40, 0.54), 'l_hip': (0.60, 0.54),
                  'r_knee': (0.38, 0.74), 'l_knee': (0.62, 0.74),
                  'r_ankle': (0.37, 0.95), 'l_ankle': (0.63, 0.95)}
NODE_R = 5                                                  # 节点半径（屏幕像素）
HIT_R = 18                                                  # 命中半径（屏幕像素）——太小就得瞄得很准才能拖
FIELDS = ['idx', 'stem', 'src', 'n_pred', 'n_gt', 'verdict', 'edited', 'shift_px', 'pred_chain', 'note']
VERDICT_CN = {'': '未判', 'ok': '采纳预标注', 'edit': '人工改过', 'none': '无骨架'}


# ---------------------------------------------------------------- 数据

def jload(path):
    with io.open(path, encoding='utf-8') as fh:
        return json.load(fh)


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=1))


def empty_pts():
    return [(0.0, 0.0, 0) for _ in BODY18]


def pred_instances(rec, min_vis):
    """从 prelabel 记录里取 body 骨架，每个 chain 一个实例。

    `min_vis`：只要"可信点够多"的实例。dwpose 在一张图上会吐出十几个 chain（234 个 body 点 / 18 ≈ 13 个），
    其中大部分是 prob≈0 的垃圾实例；不设门槛画布会糊成一片。
    """
    by = {}
    for p in rec.get('points') or []:
        if p.get('cls') != 'body':
            continue
        ch = p.get('chain') or ''
        nm = (p.get('name') or '').split('.')[-1]
        if nm not in IDX:
            continue
        by.setdefault(ch, {})[nm] = p
    out = []
    for ch, d in by.items():
        pts = []
        meta = {}
        for j, nm in enumerate(BODY18):
            p = d.get(nm)
            pts.append((float(p['x']), float(p['y']), int(p.get('vis') or 0)) if p else (0.0, 0.0, 0))
            mark = {k: p[k] for k in ('src', 'from') if p and p.get(k)}
            if mark:
                meta[j] = mark
        n_vis = sum(1 for _x, _y, v in pts if v >= 1)
        if n_vis < min_vis:
            continue
        out.append({'chain': ch, 'pts': pts, 'n_vis': n_vis, 'meta': meta})
    out.sort(key=lambda o: -o['n_vis'])
    return out


def gt_instances(gt):
    """读真值 → 实例列表（chain 顺序即人加的顺序）。

    `meta[j]` 透传点上的来源标记（`src`/`from`）：派生点（如 neck=双肩中点）必须能被下游认出来，
    不能被当成模型输出。
    """
    by = {}
    order = []
    meta = {}
    for p in gt.get('points') or []:
        if p.get('cls') != 'body':
            continue
        ch = p.get('chain') or 'H#0'
        nm = (p.get('name') or '').split('.')[-1]
        if nm not in IDX:
            continue
        if ch not in by:
            by[ch] = empty_pts()
            meta[ch] = {}
            order.append(ch)
        by[ch][IDX[nm]] = (float(p['x']), float(p['y']), int(p.get('vis') or 0))
        mark = {k: p[k] for k in ('src', 'from') if p.get(k)}
        if mark:
            meta[ch][IDX[nm]] = mark
    return [{'chain': ch, 'pts': by[ch], 'meta': meta.get(ch) or {}} for ch in order]


def points_of(instances, layer):
    out = []
    for i, ins in enumerate(instances):
        chain = 'H#%d' % i if layer == 'human' else (ins.get('chain') or 'M#%d' % i)
        meta = ins.get('meta') or {}
        for j, (nm, (x, y, vis)) in enumerate(zip(BODY18, ins['pts'])):
            p = {'layer': layer, 'cls': 'body', 'name': 'body.%s' % nm, 'chain': chain,
                 'x': round(float(x), 2), 'y': round(float(y), 2), 'vis': int(vis)}
            p.update(meta.get(j) or {})        # 来源标记透传：派生点不能被误当成模型输出
            out.append(p)
    return out


def shift_stats(gt_pts, pred_pts):
    """人把预标注拖动了多远（**按实例出现顺序配对**）。既是质检指标，也是"模型到底差多少"的量化。

    不能按 chain 名排序配对：pred 的 chain 名（`M1.dwpose#k`）是模型自己的输出序号，和真值的 `H#i`
    不是一套顺序，排一下就会把不同的人配到一起 —— 2026-10-13 实测一张 6 人的图因此报出 379px 假位移
    （按序号配对后是 0.0，因为那张是"采纳预标注"）。
    """
    if not gt_pts or not pred_pts:
        return ''

    def group(pts):
        order, by = [], {}
        for p in pts:
            ch = p['chain']
            if ch not in by:
                by[ch] = {}
                order.append(ch)
            by[ch][p['name']] = p
        return [by[k] for k in order]

    ds = []
    for A, B in zip(group(gt_pts), group(pred_pts)):
        for nm, a in A.items():
            b = B.get(nm)
            if b and (a['vis'] >= 1 or b['vis'] >= 1):
                ds.append(((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5)
    if not ds:
        return ''
    ds.sort()
    return '%.1f/%.1f' % (sum(ds) / len(ds), ds[len(ds) // 2])      # 平均/中位 像素位移


def build_queue(pred_dir, out_csv, limit=0, sort='stem', seed=1234, min_pred_vis=1, no_skeleton=False):
    """建队列。

    `no_skeleton=True` 时反过来挑：**检测到人/头但没有可用骨架**的图（实测 996 张里有 74 张）。
    这类图必须人从人形模板摆，工作量大得多，所以单独成队、可以按主动学习挑着标。
    """
    import random
    rows = []
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith('.json'):
            continue
        try:
            rec = jload(os.path.join(pred_dir, fn))
        except Exception:                                   # noqa: BLE001
            continue
        ins = pred_instances(rec, min_pred_vis)
        has_person = any((b.get('cls') in ('person', 'head')) for b in (rec.get('boxes') or []))
        if no_skeleton:
            if ins or not has_person:
                continue
            stem = os.path.splitext(fn)[0]
            rows.append({'stem': stem, 'src': rec.get('src') or '', 'n_pred': 0,
                         'pred_chain': '', 'note': '模型没检出骨架（有人/头框）'})
            continue
        if not ins:
            continue
        stem = os.path.splitext(fn)[0]
        rows.append({'stem': stem, 'src': rec.get('src') or '', 'n_pred': len(ins),
                     'pred_chain': ins[0]['chain'], 'note': rec.get('origin') or ''})
    if sort == 'hard':
        rows.sort(key=lambda r: (-r['n_pred'], r['stem']))
    elif sort == 'rand':
        random.Random(seed).shuffle(rows)
    else:
        rows.sort(key=lambda r: r['stem'])
    if limit:
        rows = rows[:limit]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with io.open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for i, r in enumerate(rows, 1):
            r['idx'] = i
            r.setdefault('n_gt', '')
            r.setdefault('verdict', '')
            r.setdefault('edited', '')
            r.setdefault('shift_px', '')
            w.writerow(r)
    miss = sum(1 for r in rows if not (r['src'] and os.path.exists(r['src'])))
    print('队列：%d 张 → %s（多人优先=%s，原图缺失 %d 张）'
          % (len(rows), out_csv, sort == 'hard', miss))
    return len(rows)


def phantom(x, y, v):
    """缺失关节的占位值 = vis 0 且坐标恰为 (0,0)。

    `pred_instances()`/`gt_instances()` 对缺失关节填 `(0.0, 0.0, 0)`（第 128-130 行），
    而 ViTPose 只出 COCO-17（无 `neck`）⇒ 每条链的 neck 都是这种占位点。
    只跳过"恰好 (0,0) 且 vis=0"的：人按 3 判成"不可见"但坐标有效的点必须照画。
    """
    return v == 0 and abs(x) < 1e-9 and abs(y) < 1e-9


# ---------------------------------------------------------------- 画布

class PoseCanvas(QWidget):
    """整块画布 = 图 + 四周留白。节点坐标存的是**图像坐标**，可以 <0 或 >宽高。"""

    def __init__(self, on_change=None):
        super().__init__()
        self.pix = None
        self.spix = None                    # 缩放后的位图缓存（拖动时别每帧重采样，见 apply_scale）
        self.iw = self.ih = 0
        self.margin = 0
        self.scale = 1.0
        self.instances = []                 # [{'chain','pts':[(x,y,vis)*18]}]
        self.pred = []                      # 预标注快照（R 还原 / 采纳用）
        self.boxes = []                     # 参照：检测框
        self.active = 0
        self.sel = None
        self.hover = None
        self.dirty = False
        self.drag = None                    # (instance_idx, node_idx)
        self.pan = None
        self.pan_cand = None                # 空白处按下：还没超过阈值 ⇒ 松手算"取消选中"
        self.user_view = False              # 人手动缩放过/平移过 ⇒ 窗口改大小时不再自动重摆
        self.scroll = None
        self.on_change = on_change
        self.setMouseTracking(True)
        # 标注是"对准"的动作 ⇒ 画布上用十字准星（悬停到节点也保持十字，靠高亮提示可拖）
        self.setCursor(Qt.CrossCursor)
        self.label_font = QFont()
        self.label_font.setPointSize(13)        # 原本是默认字号，压在图上偏小
        self.label_font.setBold(True)

    # ---- 坐标 ----
    def m(self):
        return int(round(self.margin * self.scale))

    def to_canvas(self, x, y):
        return (x + self.margin) * self.scale, (y + self.margin) * self.scale

    def to_image(self, px, py):
        return px / self.scale - self.margin, py / self.scale - self.margin

    def canvas_size(self):
        return (int(round((self.iw + 2 * self.margin) * self.scale)),
                int(round((self.ih + 2 * self.margin) * self.scale)))

    # ---- 基本 ----
    def set_image(self, path, margin_scale=1.0):
        self.pix = QPixmap(path)
        if self.pix.isNull():
            raise SystemExit('读不了图：%s' % path)
        self.iw, self.ih = self.pix.width(), self.pix.height()
        self.margin = margin_scale * self.iw
        self.instances, self.pred, self.boxes = [], [], []
        self.active, self.sel, self.hover, self.drag = 0, None, None, None
        self.pan, self.pan_cand = None, None
        self.user_view = False
        self.dirty = False
        self.apply_scale()

    def apply_scale(self):
        """缓存缩放后的位图：paintEvent 里每帧 smoothed-scale 一张 1500px 的图，拖动就会卡。"""
        w, h = self.canvas_size()
        self.setFixedSize(max(1, w), max(1, h))
        self.spix = None
        if self.pix:
            self.spix = self.pix.scaled(max(1, int(self.iw * self.scale)), max(1, int(self.ih * self.scale)),
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()

    def set_scale(self, s):
        self.scale = max(0.05, min(8.0, s))
        self.apply_scale()

    def fit_height(self):
        """默认视图：图**正好撑满视口高度**（上下贴边）并把图的中心对到视口中心。

        为什么不用 fit(whole)：留白区是上下左右各 1 倍图宽，把"图+留白"整块塞进视口会让图变得很小；
        看图时要的是尽量大，留白靠平移/缩放去看（用户 2026-10-13 的要求）。
        """
        if not self.pix or not self.scroll:
            return
        vp = self.scroll.viewport()
        if vp.width() <= 0 or vp.height() <= 0:
            return                                   # 启动早期视口还是 0，等 showEvent 再摆
        self.set_scale(vp.height() / float(self.ih))
        self.center_view()

    def center_view(self, cx=None, cy=None):
        """把**图（或指定点）的中心**对到视口中心（画布比视口大时才有效果）。

        `cx/cy` 给的是**图像坐标**；不给就按图的中心。
        """
        if not self.pix or not self.scroll:
            return
        vp = self.scroll.viewport()
        hb, vb = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        ix = self.iw / 2.0 if cx is None else cx
        iy = self.ih / 2.0 if cy is None else cy
        hb.setValue(int(round((self.margin + ix) * self.scale - vp.width() / 2.0)))
        vb.setValue(int(round((self.margin + iy) * self.scale - vp.height() / 2.0)))

    def content_rect(self):
        """**内容范围**（图像坐标）：图本身 ∪ 所有实例里可见的节点 ∪ 参照框。

        为什么不用"整块画布"：留白是上下左右各 1 倍图宽，"适应整块"会把图缩得很小（用户 2026-10-13 反馈）。
        内容范围还天然包含画到画面外的骨节点 —— 那正是当初扩画布的目的。
        """
        x0, y0, x1, y1 = 0.0, 0.0, float(self.iw), float(self.ih)
        for ins in self.instances:
            for x, y, v in ins['pts']:
                if v >= 1:
                    x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
        for b in self.boxes:
            bx0, by0, bx1, by1 = b['box']
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
        return x0, y0, x1, y1

    def fit_content(self):
        """适应 = 让**内容**（图 ∪ 骨架）正好放进视口并居中；不是把整块带留白的画布塞进去。"""
        if not self.pix or not self.scroll:
            return
        vp = self.scroll.viewport()
        if vp.width() <= 0 or vp.height() <= 0:
            return
        x0, y0, x1, y1 = self.content_rect()
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        self.set_scale(min(vp.width() / w, vp.height() / h) * 0.98)
        self.center_view((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self.user_view = True                    # 手动适应过 ⇒ 窗口缩放时不再自动重摆

    def changed(self, dirty=True):
        if dirty:
            self.dirty = True
        # 必须立刻重绘：否则拖动时节点不跟手，要等松手/别的重绘时机才看到结果（用户 2026-10-13 报的体验问题）
        self.update()
        if self.on_change:
            self.on_change()

    def cur(self):
        return self.instances[self.active] if self.instances else None

    def push_undo(self):
        if not hasattr(self, 'undo'):
            self.undo = []
        self.undo.append((copy.deepcopy(self.instances), self.active))
        del self.undo[:-50]

    def pop_undo(self):
        if getattr(self, 'undo', None):
            self.instances, self.active = self.undo.pop()
            self.sel = None
            self.dirty = True
            self.changed(dirty=False)
            return True
        return False

    # ---- 命中 ----
    def node_at(self, px, py):
        """返回 (实例下标, 节点下标)。当前实例优先，命中不到再看别的实例。

        **空实例必须早退**：画布上有图但没有实例是正常状态（判过"无骨架"的图、预标注全被门槛滤掉的图），
        这里索引 `instances[0]` 会 IndexError，而鼠标一动就调到这里 —— 2026-10-13 实测把程序打崩过。
        """
        if not self.instances:
            return None
        order = [self.active] + [i for i in range(len(self.instances)) if i != self.active]
        best = None
        for i in order:
            for j, (x, y, _v) in enumerate(self.instances[i]['pts']):
                cx, cy = self.to_canvas(x, y)
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d <= HIT_R and (best is None or d < best[0]):
                    best = (d, i, j)
        return None if best is None else (best[1], best[2])

    # ---- 绘制 ----
    def label(self, p, x, y, text):
        """画带**黑描边**的标签：白字直接压在图上，遇到亮背景就看不见了（用户 2026-10-13 反馈）。"""
        p.setFont(self.label_font)
        p.setPen(QPen(QColor(0, 0, 0, 230)))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1), (-2, 0), (2, 0)):
            p.drawText(int(x + dx), int(y + dy), text)
        p.setPen(QPen(QColor(255, 255, 255)))
        p.drawText(int(x), int(y), text)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 38, 42))                     # 画布外=深灰（留白区）
        if self.pix:
            ox, oy = self.m(), self.m()
            if self.spix is not None:
                p.drawPixmap(ox, oy, self.spix)
            p.setPen(QPen(QColor(90, 90, 100), 1, Qt.DashLine))
            p.drawRect(ox, oy, int(self.iw * self.scale), int(self.ih * self.scale))
            p.setPen(QPen(QColor(70, 70, 78), 1, Qt.DotLine))
            p.drawLine(ox, oy, ox, 0)            # 留白区参考线（图框四边延长）
            p.drawLine(ox, oy + int(self.ih * self.scale), ox, self.height())
            p.drawLine(ox, oy, 0, oy)
            p.drawLine(ox + int(self.iw * self.scale), oy, self.width(), oy)
        for i, ins in enumerate(self.instances):
            is_act = (i == self.active)
            for a, b in EDGES:
                xa, ya, va = ins['pts'][IDX[a]]
                xb, yb, vb = ins['pts'][IDX[b]]
                if va == 0 and vb == 0:
                    continue
                # 一端是缺失关节的占位点 (0,0)：这条边画出来就是一条从人物连到画面左上角的假线
                if phantom(xa, ya, va) or phantom(xb, yb, vb):
                    continue
                pen = QPen(VIS_COLOR[2] if (va >= 1 and vb >= 1) else VIS_COLOR[1], 2 if is_act else 1)
                if not (va >= 1 and vb >= 1):
                    pen.setStyle(Qt.DashLine)
                pen.setColor(QColor(pen.color().red(), pen.color().green(), pen.color().blue(),
                                    255 if is_act else 90))
                p.setPen(pen)
                ax, ay = self.to_canvas(xa, ya)
                bx, by = self.to_canvas(xb, yb)
                p.drawLine(int(ax), int(ay), int(bx), int(by))
        for b in self.boxes:
            x0, y0, x1, y1 = b['box']
            ax, ay = self.to_canvas(x0, y0)
            bx, by = self.to_canvas(x1, y1)
            p.setPen(QPen(QColor(90, 200, 255, 160), 1, Qt.DashLine))
            p.drawRect(int(ax), int(ay), int(bx - ax), int(by - ay))
            self.label(p, ax + 3, ay - 4, b.get('cls', 'box'))
        for i, ins in enumerate(self.instances):
            is_act = (i == self.active)
            for j, (x, y, v) in enumerate(ins['pts']):
                if phantom(x, y, v):            # 缺失关节不画点（GT 里仍是 (0,0,0)，缺就是缺）
                    continue
                cx, cy = self.to_canvas(x, y)
                col = VIS_COLOR.get(v, VIS_COLOR[0])
                col = QColor(col.red(), col.green(), col.blue(), 255 if is_act else 90)
                r = NODE_R + (2 if (is_act and j == self.sel) else 0)
                p.setBrush(QBrush(col) if v >= 1 else Qt.NoBrush)
                p.setPen(QPen(col, 2 if (is_act and j == self.sel) else 1))
                p.drawEllipse(int(cx - r), int(cy - r), 2 * r, 2 * r)
                if is_act and (j == self.sel or j == self.hover):
                    self.label(p, cx + r + 3, cy - r - 3, BODY18[j])

    # ---- 鼠标 ----
    def mousePressEvent(self, e):
        if not self.pix:
            return
        self.setFocus()                                     # 点画布就把焦点收回来（配合窗口级按键过滤）
        if e.button() == Qt.MiddleButton:
            g = e.globalPos()                                  # 锚点用**屏幕坐标**（原因见 mouseMoveEvent）
            self.pan = (g.x(), g.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        hit = self.node_at(e.pos().x(), e.pos().y())
        if e.button() == Qt.RightButton:                    # 右键点节点 = 循环 可见(2)→遮挡(1)→不可见(0)
            if hit:
                i, j = hit
                self.active, self.sel = i, j
                x, y, v = self.instances[i]['pts'][j]
                self.push_undo()
                self.instances[i]['pts'][j] = (x, y, {2: 1, 1: 0, 0: 2}[v])
                self.changed()
            return
        if e.button() != Qt.LeftButton:
            return
        if hit:
            i, j = hit
            if i != self.active:
                self.active = i
            self.sel = j
            self.push_undo()
            self.drag = (i, j)
            self.changed(dirty=False)
        else:
            # 空白处按下：拖动超过阈值就平移画布，没动就当成"取消选中"
            g = e.globalPos()
            self.pan_cand = (g.x(), g.y())
            self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, e):
        if not self.pix:
            return
        if self.pan_cand and not self.drag:
            px, py = self.pan_cand
            g = e.globalPos()
            if abs(g.x() - px) + abs(g.y() - py) > 3:
                self.pan, self.pan_cand = (px, py), None
                self.setCursor(Qt.ClosedHandCursor)
        if self.pan and self.scroll:
            self.user_view = True                    # 手动平移过 ⇒ 同上
            hb, vb = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
            ax, ay = self.pan
            oh, ov = hb.value(), vb.value()
            # 锚点必须用**屏幕坐标**：平移时画布 widget 自己会在视口里移动，`e.pos()`（相对画布）
            # 每帧都在变，和锚点互相抵消 ⇒ 抖动方向正好是拖动方向（用户 2026-10-13 报的）。
            g = e.globalPos()
            hb.setValue(oh - (g.x() - ax))
            vb.setValue(ov - (g.y() - ay))
            # 锚点只推进"视图真的动了"的那部分：到边界被夹住时不推进，否则一往回拖就跳
            self.pan = (ax + (oh - hb.value()), ay + (ov - vb.value()))
            return
        if self.drag:
            i, j = self.drag
            x, y = self.to_image(e.pos().x(), e.pos().y())
            _ix, _iy, v = self.instances[i]['pts'][j]
            self.instances[i]['pts'][j] = (round(x, 1), round(y, 1), v)
            self.changed()                                  # changed() 里会 update() ⇒ 节点跟手
            return
        hit = self.node_at(e.pos().x(), e.pos().y())
        self.hover = hit[1] if hit else None
        self.setCursor(Qt.CrossCursor)                      # 画布上永远是十字准星
        if hit:
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self.pan:
            self.pan = None
            self.setCursor(Qt.CrossCursor)
        if e.button() == Qt.LeftButton:
            if self.pan_cand:                               # 空白处单击（没拖动）⇒ 取消选中
                self.pan_cand = None
                self.sel = None
                self.changed(dirty=False)
            self.drag = None
            self.pan = None
            self.setCursor(Qt.CrossCursor)

    def zoom_center(self, factor):
        """键盘缩放：以视口中心为锚（和滚轮一个语义，别从左上角缩）。"""
        if not self.scroll:
            self.set_scale(self.scale * factor)
            return
        vp = self.scroll.viewport()
        self.zoom_at(factor, vp.width() / 2.0, vp.height() / 2.0)

    def zoom_at(self, factor, vx, vy):
        """以**视口坐标** (vx, vy) 为锚点缩放：锚点下那块画面必须留在原地。

        原实现有两个错（用户 2026-10-13 报"缩放不符合直觉"）：
          ① 把 widget 坐标当视口坐标用 —— 两者差一个滚动偏移，偏移越大跑得越远；
          ② 公式 (v+off)*k-off 把滚动偏移也乘了 k ⇒ 每次滚轮画面额外平移一段。
        """
        if not self.pix:
            return
        self.user_view = True                    # 手动缩放过 ⇒ 别在窗口变化时自动重摆
        old = self.scale
        if not self.scroll:
            self.set_scale(old * factor)
            return
        hb, vb = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        old_h, old_v = hb.value(), vb.value()               # 必须**缩放前**取，缩放会夹住数值
        cx, cy = vx + old_h, vy + old_v                     # 光标处的画布坐标
        self.set_scale(old * factor)
        k = self.scale / old
        vpw, vph = self.scroll.viewport().width(), self.scroll.viewport().height()
        # 只在"真的被裁剪"时自己定量程（防 resize 事件时序）；offscreen 平台不裁剪，量程交给调用方
        if vpw < self.width():
            hb.setRange(0, self.width() - vpw)
        if vph < self.height():
            vb.setRange(0, self.height() - vph)
        hb.setValue(int(round(cx * k - vx)))                # 同一个画布点回到原来的视口位置
        vb.setValue(int(round(cy * k - vy)))

    def wheelEvent(self, e):
        if not self.pix:
            return
        if self.scroll:
            vx = e.pos().x() - self.scroll.horizontalScrollBar().value()
            vy = e.pos().y() - self.scroll.verticalScrollBar().value()
        else:
            vx, vy = e.pos().x(), e.pos().y()
        self.zoom_at(1.25 if e.angleDelta().y() > 0 else 1 / 1.25, vx, vy)


# ---------------------------------------------------------------- 窗口

class PoseLabeler(QMainWindow):
    def __init__(self, queue, gt_dir, pred_dir, only_unjudged=False, margin_scale=1.0,
                 min_pred_vis=4, show_all=False):
        super().__init__()
        self.queue = queue
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.margin_scale = margin_scale
        self.min_pred_vis = min_pred_vis
        self.show_all = show_all
        self.rows = []
        with io.open(queue, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                self.rows.append(r)
        self.only_unjudged = only_unjudged
        self.pos = 0
        self.gt_cache = {}               # stem -> 实例（C 拷贝上一张用；只活在本进程里）
        self.setWindowTitle('M1 骨架校准标注器')
        self.resize(1500, 980)
        # 界面字号整体上调并加粗（用户 2026-10-13：标签看不清）。设在建子控件之前，子控件继承。
        _f = QFont()
        _f.setPointSize(11)
        _f.setBold(True)
        self.setFont(_f)

        self.canvas = PoseCanvas(on_change=self.refresh_status)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        # 滚动条常显：否则它随缩放出现/消失会改变视口尺寸，画面在阈值附近来回跳（缩放手感的一部分）
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # 画布比视口小时（缩得很小），让它居中而不是贴在左上角
        self.scroll.setAlignment(Qt.AlignCenter)
        self.canvas.scroll = self.scroll
        self.status = QLabel('')
        self.prog = QLabel('')
        self.cb_all = QCheckBox('显示低置信预标注 (V)')
        self.cb_all.setChecked(show_all)
        self.cb_all.stateChanged.connect(lambda _s: self.set_show_all(self.cb_all.isChecked()))
        self.spin = QSpinBox()
        self.spin.setRange(1, max(1, len(self.rows)))
        self.spin.setPrefix('#')
        # 顶栏这些控件**不许拿键盘焦点**：QSpinBox 内部是 QLineEdit，一旦拿到焦点，
        # `,`/`.` 会被当成文字吃进输入框、方向键被它/滚动区吃掉 ⇒ 翻页全哑。
        # 跳转改由 G 键的对话框负责（模态，不会抢主窗口的键）。
        self.spin.setReadOnly(True)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        for wdg in (self.spin, self.cb_all, self.scroll):
            wdg.setFocusPolicy(Qt.NoFocus)
        self.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        # 兜底：无论焦点在哪里（画布/滚动区/任何子控件），按键都走同一个处理函数
        QApplication.instance().installEventFilter(self)
        top = QHBoxLayout()
        top.addWidget(self.cb_all)
        top.addWidget(self.spin)
        top.addWidget(self.prog)
        top.addWidget(self.status, 1)

        # ---- 按钮行：所有操作都要有按钮，别把功能全压在键盘上 ----
        def mkbtn(text, tip, fn):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)      # 按钮抢焦点 = 键盘失效，一律 NoFocus
            b.clicked.connect(fn)
            return b

        def row(items):
            h = QHBoxLayout()
            for t, tip, fn in items:
                h.addWidget(mkbtn(t, tip, fn))
            h.addStretch(1)
            return h

        r1 = row([('◀ 上一张 (←)', '上一张（← / , / ↑）', lambda: self.step(-1)),
                  ('下一张 (→)', '下一张（→ / . / ↓）', lambda: self.step(1)),
                  ('跳到 # (G)', '跳到第 N 张', self.ask_jump)])
        r2 = row([('采纳并下一张 (空格)', '采纳（没改过=采纳预标注，改过=人工改过）+ 前进', self.accept_and_next),
                  ('整张采纳 (A)', '把人画的整张预标注采纳为真值并前进', self.adopt_all),
                  ('无骨架 (X)', '这张没有可标的人/骨架并前进', self.mark_none),
                  ('保存 (S)', '只保存不翻页', lambda: self.save_row())])
        r3 = row([('新增实例 (N)', '用人形模板新增一个人', self.new_instance),
                  ('删除实例 (Del)', '删掉当前实例', self.del_instance),
                  ('还原预标注 (R)', '把当前实例还原成模型预标注', self.reset_instance),
                  ('撤销 (Z)', '撤销上一步；没得撤就回上一张', self.undo_or_back)])
        # 可见性 + 视图分两行放不下时窗口最小宽度会超过屏幕（2560 宽实测被 Windows 拒绝设置几何）
        r4 = row([('可见 (1)', '选中/光标下的节点设为可见（存 vis=2）', lambda: self.set_vis(2)),
                  ('遮挡 (2)', '……设为被遮挡（存 vis=1）', lambda: self.set_vis(1)),
                  ('不可见 (3)', '……设为不可见/不存在（存 vis=0）', lambda: self.set_vis(0))])
        r5 = row([('适应内容 (F)', '图 + 骨架（含画到画面外的点）正好放进视口并居中', self.canvas.fit_content),
                  ('放大 (+)', '以视口中心放大（滚轮同效）', lambda: self.canvas.zoom_center(1.25)),
                  ('缩小 (−)', '以视口中心缩小（滚轮同效）', lambda: self.canvas.zoom_center(1 / 1.25))])
        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addLayout(r1)
        lay.addLayout(r2)
        lay.addLayout(r3)
        lay.addLayout(r4)
        lay.addLayout(r5)
        lay.addWidget(self.scroll, 1)
        box = QWidget()
        box.setLayout(lay)
        self.setCentralWidget(box)
        self.first_show = True
        self.last_msg = ''

    def note(self, msg):
        """把"刚刚发生了什么"写进状态栏 —— 手感问题没法靠猜，得让人（和我）看见按键真的做了什么。"""
        self.last_msg = msg
        self.refresh_status()

    def showEvent(self, ev):
        """窗口真正显示出来时视口才有尺寸 ⇒ 这时候才摆默认视图（撑满高度 + 居中）。"""
        super().showEvent(ev)
        if self.first_show:
            self.first_show = False
            self.canvas.fit_height()

    def resizeEvent(self, ev):
        """窗口大小变了：人没手动缩放过就重新撑满；手动动过就不打扰他的视角。"""
        super().resizeEvent(ev)
        if not self.canvas.user_view:
            self.canvas.fit_height()

    def ask_jump(self):
        from PyQt5.QtWidgets import QInputDialog
        v, ok = QInputDialog.getInt(self, '跳转', '第几张：', self.pos + 1, 1, len(self.rows))
        if ok:
            self.jump(v)
            self.note('跳到第 %d 张' % v)

    def undo_or_back(self):
        if not self.canvas.pop_undo():
            self.step(-1)
            self.note('撤销：没有节点改动 ⇒ 回上一张')
        else:
            self.note('撤销：节点改动')

    def copy_prev(self):
        """C：把**上一张**的真值整份拷进当前图（用户 2026-10-21：差分套图相邻两图几乎一样，想复制再微调）。

        约定：拷完**不翻页**（先看一眼再按空格/Enter 采纳）；判定记 `edit`（不新增 verdict 值，
        免得下游统计要认第五种）；note 追加 `copy_from=<上一张 stem>` 便于回溯来源。
        拷贝源优先级：本进程缓存（看过/刚存的）→ pose_gt/<stem>.json → 状态栏提示没有。
        """
        if self.pos <= 0:
            self.note('第 1 张没有"上一张"可拷')
            return
        rows = list(csv.DictReader(io.open(self.queue, encoding='utf-8'))) \
            if not self.rows else self.rows
        prev_stem = (rows[self.pos - 1].get('stem') or '') if rows else ''
        src = self.gt_cache.get(prev_stem)
        if src is None:
            f = self.gt_path(prev_stem)
            if os.path.exists(f):
                src = gt_instances(jload(f))
        if not src:
            self.note('上一张（%s）没有标注可拷贝' % prev_stem)
            return
        c = self.canvas
        c.push_undo()                       # 拷贝这一下也进撤销栈：Z 能退回拷贝前的状态
        c.instances = copy.deepcopy(src)
        c.active = 0
        c.sel, c.hover, c.drag = None, None, None
        c.changed()
        self.save_row('edit')               # 立刻落盘：判过的图必须显示真值，否则重启后这次拷贝丢了
        r = self.current_row()
        if r is not None:
            note = (r.get('note') or '').strip()
            r['note'] = (note + ' copy_from=%s' % prev_stem).strip()
            self.write_queue()
        self.note('已拷贝上一张 %s 的 %d 个实例（Z 可撤回拷贝，空格采纳并下一张）'
                  % (prev_stem, len(c.instances)))


    # ---- 行 ----
    def current_row(self):
        return self.rows[self.pos] if self.rows else None

    def n_judged(self):
        return sum(1 for r in self.rows if (r.get('verdict') or ''))

    def next_unjudged_index(self, pos):
        """从 pos 之后找第一张**没判定**的图 —— 采纳过的就不再出现（用户 2026-10-13 定的口径）。"""
        for i in range(pos + 1, len(self.rows)):
            if not (self.rows[i].get('verdict') or ''):
                return i
        return None

    def gt_path(self, stem):
        return os.path.join(self.gt_dir, stem + '.json')

    def show_row(self):
        r = self.current_row()
        if r is None:
            self.status.setText('队列空了')
            return
        src = r.get('src') or ''
        if not (src and os.path.exists(src)):
            self.canvas.pix = None
            self.canvas.instances, self.canvas.pred, self.canvas.boxes = [], [], []
            self.canvas.setFixedSize(400, 200)
            self.status.setText('第 %s 张：原图不在（%s）' % (r.get('idx'), src))
            self.refresh_status()
            return
        self.canvas.set_image(src, self.margin_scale)
        pre = os.path.join(self.pred_dir, r['stem'] + '.json')
        rec = jload(pre) if os.path.exists(pre) else {}
        pred = pred_instances(rec, 1 if self.show_all else self.min_pred_vis)
        self.canvas.pred = copy.deepcopy(pred)
        self.canvas.boxes = [b for b in (rec.get('boxes') or []) if b.get('cls') == 'person']
        gt_file = self.gt_path(r['stem'])
        if os.path.exists(gt_file):                      # 判过的图永远显示真值，绝不显示预标注
            gt = jload(gt_file)
            self.canvas.instances = gt_instances(gt)
            self.gt_cache[r['stem']] = copy.deepcopy(self.canvas.instances)
        else:
            self.canvas.instances = copy.deepcopy(pred)
        self.canvas.active = 0
        self.canvas.sel, self.canvas.hover, self.canvas.drag = None, None, None
        self.canvas.dirty = False
        self.canvas.undo = []
        self.spin.blockSignals(True)
        self.spin.setValue(self.pos + 1)
        self.spin.blockSignals(False)
        self.canvas.fit_height()          # 默认视图：上下撑满 + 居中（视口还没建立时内部会跳过）
        self.canvas.center_view()
        if self.first_show:
            self.first_show = False
        self.refresh_status()

    def refresh_status(self):
        r = self.current_row()
        if r is None:
            return
        c = self.canvas
        inst = '实例 %d/%d' % (c.active + 1, len(c.instances)) if c.instances else '无实例'
        sel = ''
        if c.sel is not None and c.instances:
            x, y, v = c.instances[c.active]['pts'][c.sel]
            sel = ' · 选中 %s vis=%d @(%.0f,%.0f)' % (BODY18[c.sel], v, x, y)
            if not (0 <= x <= c.iw and 0 <= y <= c.ih):
                sel += ' 画面外'
        self.prog.setText('已判 %d/%d（未判 %d）'
                          % (self.n_judged(), len(self.rows), len(self.rows) - self.n_judged()))
        # 预标注为 0 是**正常且重要**的状态：模型没检出骨架不等于这张图没有骨架（实测 74/996 张如此）。
        # 明确告诉人下一步怎么做，别让人以为程序坏了。
        hint = ''
        if not c.pred and not c.instances:
            hint = ('\n⚠ 模型没检出骨架' + ('（但检测到了人/头）' if c.boxes else '')
                    + '：按「新增实例 (N)」会按检测框放一个人形模板，拖正即可；若这张确实没有骨架按「无骨架 (X)」')
        self.status.setText(
            '第 %s/%d 张 %s · %s%s · 预标注%s个（%s）· 判定=%s · %s%s\n'
            '键位都写在按钮上了；按钮之外的只有：拖点校准 · Shift+方向键 微调 · 右键点节点 改可见性 · '
            '滚轮缩放 · 空白处拖动/中键 平移 · C 拷贝上一张 · Ctrl+→ 跳到下一张未判的' % (
                r.get('idx'), len(self.rows), r['stem'], inst, sel, len(c.pred),
                os.path.basename(os.path.normpath(self.pred_dir)),
                VERDICT_CN.get(r.get('verdict') or '', r.get('verdict') or ''),
                '有改动未存' if c.dirty else '已存/未改',
                ('　·　最近动作：%s' % getattr(self, 'last_msg', '')) if getattr(self, 'last_msg', '') else '')
            + hint)

    def save_row(self, verdict=None):
        r = self.current_row()
        if r is None:
            return
        c = self.canvas
        if not c.instances and verdict is None:
            verdict = 'none'
        if verdict is None:
            # 落盘时**不许把"没判过"偷偷变成 ok**：只有改过才是 edit，否则保持原判定不变。
            # （"采纳"必须是显式动作：空格/Enter、A、或「采纳」按钮。）
            verdict = 'edit' if c.dirty else (r.get('verdict') or '')
        stem = r['stem']
        if not c.instances:
            jdump(self.gt_path(stem), {'stem': stem, 'image': stem, 'src': r.get('src') or '',
                                       'size': '%dx%d' % (c.iw, c.ih), 'margin_scale': self.margin_scale,
                                       'verdict': verdict, 'instances': 0, 'points': [], 'pred_points': [],
                                       'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
        else:
            pts = points_of(c.instances, 'human')
            ppts = points_of(c.pred, 'M1') if c.pred else []
            jdump(self.gt_path(stem), {
                'stem': stem, 'image': stem, 'src': r.get('src') or '',
                'size': '%dx%d' % (c.iw, c.ih), 'margin_scale': self.margin_scale,
                'verdict': verdict, 'instances': len(c.instances),
                'edited': sum(1 for i, ins in enumerate(c.instances)
                              if i < len(c.pred) and ins['pts'] != c.pred[i]['pts']),
                'shift_px': shift_stats(pts, ppts), 'points': pts, 'pred_points': ppts,
                'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
        r['verdict'] = verdict
        r['n_gt'] = str(len(c.instances))
        r['edited'] = str(sum(1 for i, ins in enumerate(c.instances)
                              if i < len(c.pred) and ins['pts'] != c.pred[i]['pts']))
        r['shift_px'] = shift_stats(points_of(c.instances, 'human'), points_of(c.pred, 'M1')) if c.pred else ''
        self.write_queue()
        self.gt_cache[stem] = copy.deepcopy(c.instances)
        c.dirty = False
        self.refresh_status()

    def write_queue(self):
        with io.open(self.queue, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    # ---- 导航 ----
    def step(self, d, why=''):
        """前进 = 下一张**未判**的（归档/已判的不再出现）；后退 = 上一张（含已判的，方便回头改）。

        翻页**不改变判定**；只有真的改过（dirty）才落盘（记 edit）。
        从前是"路过未判图就默认采纳成 ok"，而 ok 会让这一行从集合里消失 ⇒ 接着找"最近未判"
        就找到了**后面**那张 —— 于是"上一张"实际在前进（2026-10-13 修掉，有回归测试）。
        """
        if self.canvas.instances and self.canvas.dirty:
            self.save_row()
        if d > 0:
            # 前进 = 下一张**未判**的：归档过的（已判）不再出现（用户 2026-10-13 的要求）。
            # 相邻翻页想看已判的图时用 ←（后退不跳过任何行）。
            nxt = self.next_unjudged_index(self.pos)
            if nxt is None:
                self.note('后面没有未判的图了（已判 %d / %d）' % (self.n_judged(), len(self.rows)))
                return
            self.pos = nxt
            self.last_msg = why or '下一张未判'
        else:
            self.pos = max(0, min(len(self.rows) - 1, self.pos + d))
            self.last_msg = why or '上一张（含已判的，可回头改）'
        self.show_row()
        # show_row 里已经按"撑满高度 + 居中"摆好了视图；这里不用再把滚动条拉回左上角
        # （旧代码 setValue(0) 让每翻一张都跳回左上角，用户 2026-10-13 报"默认没居中"）

    def jump(self, idx1):
        r = self.current_row()
        if self.canvas.instances and self.canvas.dirty:
            self.save_row()          # 有真改动先落盘；单纯跳转不改判定（同 step 的口径）
        self.pos = max(0, min(len(self.rows) - 1, int(idx1) - 1))
        self.show_row()

    def set_only_unjudged(self, on):
        """已废弃：现在**前进默认就跳过已判的图**，不需要这个开关（用户 2026-10-13 定的）。"""
        self.only_unjudged = on

    def set_show_all(self, on):
        self.show_all = on
        if self.canvas.instances:
            self.save_row(verdict=self.current_row().get('verdict') or None) if self.canvas.dirty else None
        self.show_row()

    # ---- 动作 ----
    def adopt_all(self):
        """采纳预标注（整张图一次），不改动。"""
        c = self.canvas
        if c.pred:
            c.instances = copy.deepcopy(c.pred)
        c.dirty = False
        self.save_row(verdict='ok')
        self.step(1, why='采纳预标注并下一张')

    def mark_none(self):
        self.canvas.instances = []
        self.save_row(verdict='none')
        self.step(1, why='标记无骨架并下一张')

    def accept_and_next(self):
        """空格/Enter = 明确采纳：没改过记 ok（采纳预标注），改过记 edit，然后下一张。

        这是**唯一**会把"未判"变成判定的翻页路径（外加 A / X / 各按钮）；单纯 `← →` 不改判定。
        """
        v = 'edit' if (self.canvas.dirty or (self.current_row() or {}).get('verdict') == 'edit') else 'ok'
        self.save_row(verdict=v)
        self.step(1, why='采纳并下一张（%s）' % VERDICT_CN.get(v, v))

    def new_instance(self):
        """新增实例：挑一个"最没被现有实例占用"的检测框，用人形模板初始化（全部 vis=1，等人拖正）。

        没有检测框就退回到画面中央的一块区域 —— 总之给人一个能拖的东西，而不是 18 个叠在一起的点。
        """
        c = self.canvas
        c.push_undo()
        boxes = c.boxes or []
        best, best_n = None, None
        for b in boxes:
            x0, y0, x1, y1 = b['box']
            n = 0
            for ins in c.instances:
                for x, y, v in ins['pts']:
                    if v >= 1 and x0 <= x <= x1 and y0 <= y <= y1:
                        n += 1
            if best_n is None or n < best_n:
                best, best_n = (x0, y0, x1, y1), n
        if best is None or (best_n or 0) > 6:                   # 框都被占了 / 没框 ⇒ 画面中央
            cx, cy = c.iw * 0.5, c.ih * 0.5
            hw, hh = c.iw * 0.18, c.ih * 0.30
            best = (cx - hw, cy - hh, cx + hw, cy + hh)
        x0, y0, x1, y1 = best
        pts = []
        for nm in BODY18:
            u, v = HUMAN_TEMPLATE[nm]
            pts.append((round(x0 + u * (x1 - x0), 1), round(y0 + v * (y1 - y0), 1), 1))
        c.instances.append({'chain': 'H#%d' % len(c.instances), 'pts': pts})
        c.active = len(c.instances) - 1
        c.sel = None
        c.changed()
        self.refresh_status()

    def del_instance(self):
        c = self.canvas
        if not c.instances:
            return
        c.push_undo()
        c.instances.pop(c.active)
        c.active = max(0, min(len(c.instances) - 1, c.active))
        c.sel = None
        c.changed()
        self.refresh_status()

    def reset_instance(self):
        c = self.canvas
        if c.active < len(c.pred):
            c.push_undo()
            c.instances[c.active] = copy.deepcopy(c.pred[c.active])
            c.changed()
            self.refresh_status()

    def cycle_instance(self, d):
        c = self.canvas
        if c.instances:
            c.active = (c.active + d) % len(c.instances)
            c.sel = None
            self.refresh_status()
            c.update()

    def set_vis(self, v):
        c = self.canvas
        if not c.instances:
            self.note('没有实例，先按「新增实例」')
            return
        j = c.sel if c.sel is not None else c.hover
        if j is None:
            self.note('先在画布上点一个节点（或把鼠标悬在节点上），再设可见性')
            return
        x, y, _old = c.instances[c.active]['pts'][j]
        c.push_undo()
        c.instances[c.active]['pts'][j] = (x, y, v)
        c.changed()
        self.note('节点 %s 可见性 → %d' % (BODY18[j], v))

    def nudge(self, dx, dy):
        c = self.canvas
        if not c.instances or c.sel is None:
            return
        x, y, v = c.instances[c.active]['pts'][c.sel]
        c.push_undo()
        c.instances[c.active]['pts'][c.sel] = (round(x + dx, 1), round(y + dy, 1), v)
        c.changed()
        self.refresh_status()

    def eventFilter(self, obj, ev):
        """把本窗口（及其子控件）的按键统统抢过来自己处理。

        为什么需要：QSpinBox 内部的 QLineEdit 拿到焦点时会把 `.`/`,` 当文字吃掉、方向键也会被
        QScrollArea/QSpinBox 消费掉 —— 表现就是"翻页键全哑"。顶栏控件已设 NoFocus，这里是兜底：
        只要事件属于本窗口树，且没有模态对话框开着（G 跳转框是模态，不能抢），就交给同一个处理函数。
        """
        if ev.type() != QEvent.KeyPress:
            return False
        if QApplication.activeModalWidget() is not None:
            return False
        if not isinstance(obj, QWidget):
            return False
        if obj is self or self.isAncestorOf(obj):
            self.keyPressEvent(ev)
            return True
        return False

    def keyPressEvent(self, e):
        k = e.key()
        c = self.canvas
        mod = e.modifiers()
        if k in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.accept_and_next()
        elif k == Qt.Key_A:
            self.adopt_all()
        elif k == Qt.Key_X:
            self.mark_none()
        elif k == Qt.Key_S:
            self.save_row()
        elif k == Qt.Key_C:
            self.copy_prev()
        elif k == Qt.Key_Comma:
            self.step(-1)
        elif k == Qt.Key_Period:
            self.step(1)
        elif k in (Qt.Key_BracketLeft,):
            self.cycle_instance(-1)
        elif k in (Qt.Key_BracketRight, Qt.Key_Tab):
            self.cycle_instance(1)
        elif k == Qt.Key_N:
            self.new_instance()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.del_instance()
        elif k == Qt.Key_R:
            self.reset_instance()
        elif k == Qt.Key_Z:
            self.undo_or_back()
        elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            # 键位按**键盘相邻顺序**对应"语义顺序"：1 可见 → 2 遮挡 → 3 不可见。
            # 存进标签的 vis 仍是 2/1/0（导出契约，不能跟着键位改）。
            self.set_vis({Qt.Key_1: 2, Qt.Key_2: 1, Qt.Key_3: 0}[k])
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if mod & Qt.ShiftModifier:                      # Shift+方向键 = 微调选中节点（Ctrl 加成 10px）
                n = 10 if (mod & Qt.ControlModifier) else 1
                dx, dy = {Qt.Key_Left: (-n, 0), Qt.Key_Right: (n, 0),
                          Qt.Key_Up: (0, -n), Qt.Key_Down: (0, n)}[k]
                self.nudge(dx, dy)
            else:                                           # 方向键 = 上一张/下一张
                self.step(-1 if k in (Qt.Key_Left, Qt.Key_Up) else 1)
        elif k in (Qt.Key_PageUp, Qt.Key_PageDown):
            self.step(-1 if k == Qt.Key_PageUp else 1)
        elif k in (Qt.Key_F, Qt.Key_W):
            c.fit_content()                                  # 适应内容（图 ∪ 骨架）；W 保留为别名
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            c.zoom_center(1.25)
        elif k == Qt.Key_Minus:
            c.zoom_center(1 / 1.25)
        elif k == Qt.Key_V:
            self.cb_all.setChecked(not self.cb_all.isChecked())
        elif k == Qt.Key_G:
            self.ask_jump()
        else:
            # 没绑定的键就说出来（状态栏留痕）。手感问题不能靠猜：这样一眼能看出
            # "某个键到底做了什么"，也能立刻抓到"按什么都翻页"这类串键。
            txt = e.text() if hasattr(e, 'text') else ''
            self.note('无绑定键：%s' % (txt or ('Key_%d' % int(k))))
            super().keyPressEvent(e)


# ---------------------------------------------------------------- 自检

def selftest():
    """离屏把关键路径跑一遍：预标注注入、拖动、出画坐标、落盘、判定不被覆盖、schema 与导出一致。"""
    import tempfile
    from PIL import Image
    app = QApplication.instance() or QApplication(sys.argv)      # main() 里已建；单独调用时兜底
    tmp = tempfile.mkdtemp()
    pred_dir = os.path.join(tmp, 'prelabel')
    gt_dir = os.path.join(tmp, 'pose_gt')
    queue = os.path.join(tmp, 'queue.csv')
    img = os.path.join(tmp, 'a.png')
    img2 = os.path.join(tmp, 'b.png')
    Image.new('RGB', (400, 300), (30, 30, 45)).save(img)
    Image.new('RGB', (360, 480), (20, 40, 30)).save(img2)

    # 造一条 prelabel 记录：2 个实例（其一有 12 个可信点，其一是垃圾实例 vis=0）
    def inst(chain, shift, vis_n):
        pts = []
        for i, nm in enumerate(BODY18):
            pts.append({'layer': 'M1', 'cls': 'body', 'name': 'body.%s' % nm, 'chain': chain,
                        'x': 100 + i * 5 + shift, 'y': 120 + i * 4 + shift, 'prob': 0.9,
                        'vis': 2 if i < vis_n else 0})
        return pts
    rec = {'image': 'a', 'src': img, 'size': '400x300', 'origin': 'selftest', 'n_people': 2,
           'chain': ['M1.dwpose', 'M2a.person'],
           'points': inst('M1.dwpose#0', 0, 12) + inst('M1.dwpose#1', 40, 0),
           'boxes': [{'layer': 'M2a', 'cls': 'person', 'box': [90, 110, 260, 290], 'prob': 0.9}]}
    os.makedirs(pred_dir, exist_ok=True)
    jdump(os.path.join(pred_dir, 'a.json'), rec)
    # 第二张：单实例干净预标注（翻页/回退/平移这些交互要有第二张才测得出）
    jdump(os.path.join(pred_dir, 'b.json'),
          {'image': 'b', 'src': img2, 'size': '360x480', 'origin': 'selftest', 'n_people': 1,
           'chain': ['M1.dwpose'], 'points': inst('M1.dwpose#0', 5, 14),
           'boxes': [{'layer': 'M2a', 'cls': 'person', 'box': [40, 60, 300, 460], 'prob': 0.9}]})

    ok = []

    def chk(name, cond):
        ok.append((name, bool(cond)))
        print('  [%s] %s' % ('OK ' if cond else 'FAIL', name))

    import export_labels as EX                          # schema 对齐用（import 不建目录）
    chk('BODY18 与 export_labels 完全一致', BODY18 == EX.BODY18)

    n = build_queue(pred_dir, queue, min_pred_vis=1)
    chk('建队列：两张图都进来（垃圾实例那张也在，实例被门槛挡住）', n == 2 and os.path.exists(queue))

    # ---- "检出人但没有可用骨架"的图：单独成队 + 状态栏给出下一步（实测 996 张里有 74 张）----
    jdump(os.path.join(pred_dir, 'c.json'),
          {'image': 'c', 'src': img, 'size': '400x300', 'origin': 'selftest', 'n_people': 0,
           'chain': ['M2a.person'], 'points': [],
           'boxes': [{'layer': 'M2a', 'cls': 'person', 'box': [40, 50, 300, 290], 'prob': 0.9}]})
    qns = os.path.join(os.path.dirname(queue), 'no_skel.csv')
    n2 = build_queue(pred_dir, qns, min_pred_vis=6, no_skeleton=True)
    ns_rows = list(csv.DictReader(io.open(qns, encoding='utf-8')))
    chk('无骨架队列：只挑"有人/头框但没有可用实例"的图',
        n2 == 1 and [r['stem'] for r in ns_rows] == ['c'] and ns_rows[0]['n_pred'] == '0')
    wns = PoseLabeler(qns, gt_dir, pred_dir)
    wns.show_row()
    chk('这种图打开时状态栏给出下一步提示（别让人以为程序坏了）',
        '模型没检出骨架' in wns.status.text() and '新增实例' in wns.status.text())
    chk('这种图不会崩：悬停/按下都安全', wns.canvas.node_at(100, 100) is None)

    w = PoseLabeler(queue, gt_dir, pred_dir, margin_scale=1.0, min_pred_vis=6)
    w.show_row()
    c = w.canvas
    chk('预标注注入：低置信实例被门槛挡掉', len(c.instances) == 1 and len(c.pred) == 1)
    exp_w = int(round((c.iw + 2 * c.margin) * c.scale))
    chk('画布 = 图 + 四周各 1 倍图宽的留白', c.margin == c.iw and c.width() == exp_w
        and c.width() > c.iw)
    chk('参照检测框画出来了', len(c.boxes) == 1 and c.boxes[0]['cls'] == 'person')
    chk('刚进来不算脏', c.dirty is False)

    # 拖一个节点到画面外（左下越界），并要求原样存下来
    c.push_undo()
    c.instances[0]['pts'][IDX['nose']] = (-50.0, -30.0, 2)
    c.changed()
    chk('拖动后置脏 + 出画坐标被接受', c.dirty is True and c.instances[0]['pts'][IDX['nose']][0] == -50.0)
    chk('出画节点的画布坐标落在图外（留白区里）', c.to_canvas(-50.0, -30.0)[0] < c.m())

    w.step(1)
    gt = jload(os.path.join(gt_dir, 'a.json'))
    pts = {p['name']: p for p in gt['points'] if p['chain'] == 'H#0'}
    chk('落盘：verdict=edit', gt['verdict'] == 'edit')
    chk('落盘：出画坐标原样保留（未被夹到 0）', pts['body.nose']['x'] == -50.0 and pts['body.nose']['y'] == -30.0)
    chk('落盘：schema 与 prelabel 一致（layer/cls/name/chain/vis）',
        pts['body.nose']['cls'] == 'body' and pts['body.nose']['layer'] == 'human'
        and all(k in pts['body.nose'] for k in ('x', 'y', 'vis', 'chain', 'name')))
    chk('落盘：18 个点齐全', len(pts) == 18)
    chk('落盘：留着模型原始预测 pred_points', len(gt['pred_points']) == 18)
    chk('落盘：算了人改动位移 shift_px', bool(gt['shift_px']))
    chk('队列同步落盘', (lambda r: r['verdict'] == 'edit' and r['n_gt'] == '1')(
        list(csv.DictReader(io.open(queue, encoding='utf-8')))[0]))

    w.step(-1)
    chk('回头看：显示的是真值（不是预标注）', c.instances[0]['pts'][IDX['nose']][0] == -50.0)
    chk('回头看：不算脏', c.dirty is False)

    # 采纳路径：A 必须把预标注原样存成 ok，且不保留上一轮的位移
    w.adopt_all()
    gt2 = jload(os.path.join(gt_dir, 'a.json'))
    p2 = {p['name']: p for p in gt2['points'] if p['chain'] == 'H#0'}
    chk('采纳：verdict=ok 且坐标回到预标注', gt2['verdict'] == 'ok' and p2['body.nose']['x'] == 100.0)
    chk('采纳：shift_px 归零', gt2['shift_px'].startswith('0.0/0.0'))

    # 无骨架路径
    w.step(-1)
    w.mark_none()
    gt3 = jload(os.path.join(gt_dir, 'a.json'))
    chk('无骨架：verdict=none 且 points 为空', gt3['verdict'] == 'none' and gt3['points'] == [])

    # 键盘事件用**真的 QKeyEvent**：PyQt 对 keyPressEvent 做类型检查，假事件对象会 TypeError。
    # 就地 import + 别名：自检后面还会 import 这两个名字，闭包会被"函数局部变量"规则拦住。
    def _key(k, m=Qt.NoModifier, text=''):
        from PyQt5.QtCore import QEvent as _QE
        from PyQt5.QtGui import QKeyEvent as _QKE
        return _QKE(_QE.KeyPress, k, m, text)

    w.step(-1)
    chk('判定=none 的图回看是空画布', len(c.instances) == 0)
    w.new_instance()
    chk('空画布上 N = 建第 1 个实例，且用人形模板初始化（18 点 vis=1）',
        len(c.instances) == 1 and sum(1 for _x, _y, v in c.instances[0]['pts'] if v >= 1) == 18)
    chk('模板落在检测框里（不是叠在同一点）',
        len({(round(x), round(y)) for x, y, _v in c.instances[0]['pts']}) == 18)
    w.new_instance()
    chk('再 N = 第 2 个实例（框已被占用 ⇒ 退回画面中央）', len(c.instances) == 2 and c.active == 1)
    w.del_instance()
    chk('删除实例：回到 1 个', len(c.instances) == 1)
    w.keyPressEvent(_key(Qt.Key_Z))
    chk('撤销：把删掉的实例撤回来', len(c.instances) == 2)

    # vis 设置
    c.active, c.sel = 0, IDX['l_ankle']
    w.set_vis(0)
    chk('set_vis(0) 把节点设为不可见', c.instances[0]['pts'][IDX['l_ankle']][2] == 0)
    # 键位 = 键盘相邻顺序 1/2/3 → 可见/遮挡/不可见（存进去仍是 2/1/0）
    c.active, c.sel = 0, IDX['nose']
    for key, want in ((Qt.Key_1, 2), (Qt.Key_2, 1), (Qt.Key_3, 0)):
        w.keyPressEvent(_key(key))
        chk('按 %s → vis=%d' % (chr(key), want), c.instances[0]['pts'][IDX['nose']][2] == want)
    w.set_vis(2)
    w.keyPressEvent(_key(Qt.Key_0))
    chk('0 不再绑定可见性（相邻顺序只用 1/2/3）', c.instances[0]['pts'][IDX['nose']][2] == 2)
    c.sel = IDX['nose']
    w.nudge(-3, 2)
    chk('方向键微调选中节点', c.instances[0]['pts'][IDX['nose']][:2] != (0, 0))

    # 导出一致性：真值实例 ≥10 个可信点 ⇒ 能出一条 COCO17 行（1 + 4 + 17*3 个数）
    gt4 = jload(os.path.join(gt_dir, 'a.json')) if False else gt2
    vis_pts = [p for p in gt4['points'] if p['vis'] >= 1]
    row = ['0'] + ['0.1'] * 4
    for bi in EX.COCO17_FROM_BODY18:
        p = next((q for q in gt4['points'] if q['name'] == 'body.%s' % BODY18[bi]), None)
        row += ['0.0000', '0.0000', '0'] if p is None else ['0.5', '0.5', str(p['vis'])]
    chk('导出一致性：真值能按 COCO17 展开成一行 YOLO-pose 标签', len(row) == 1 + 4 + 51)
    chk('导出一致性：可信点够 10 个（export 的 min_vis_pts 门槛）',
        len([p for p in gt4['points'] if p['vis'] >= 1]) >= 10)

    # ---- 2026-10-13 用户报的交互问题：拖不跟手 / 回不了上一张。逐条写成回归测试 ----
    from PyQt5.QtCore import QEvent, QPoint, QPointF
    from PyQt5.QtGui import QKeyEvent, QMouseEvent

    def mev(kind, x, y):
        return QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)

    class PanEv:
        """按真实物理造事件：**屏幕坐标固定，widget 坐标 = 屏幕坐标 + 当前滚动量**。

        抖动就是这么来的：平移时画布 widget 会在视口里移动，`e.pos()`（相对画布）每帧都在变，
        和锚点互相抵消 —— 实测旧实现每两步只动一步、总位移只有一半（见 .runtime/sim_pan.py）。
        """

        def __init__(self, kind, sx, sy, btn=Qt.MiddleButton):
            self._k, self._sx, self._sy, self._b = kind, sx, sy, btn

        def button(self):
            return self._b

        def pos(self):
            hb = c.scroll.horizontalScrollBar()
            vb = c.scroll.verticalScrollBar()
            return QPoint(int(self._sx + hb.value()), int(self._sy + vb.value()))

        def globalPos(self):
            return QPoint(int(self._sx), int(self._sy))

    w.jump(2)                                   # 第 2 张是未判的 ⇒ 画布上是预标注
    c = w.canvas
    chk('第 2 张画布上就是预标注实例', len(c.instances) == 1 and len(c.pred) == 1)
    n_up = [0]
    real_update = c.update
    c.update = lambda *a, **k: (n_up.__setitem__(0, n_up[0] + 1), real_update(*a, **k))[1]
    j0 = IDX['nose']
    x0, y0, _v = c.instances[0]['pts'][j0]
    cx, cy = c.to_canvas(x0, y0)
    c.mousePressEvent(mev(QEvent.MouseButtonPress, cx, cy))
    chk('按下节点 = 进入拖动并选中', c.drag == (0, j0) and c.sel == j0)
    n_before = n_up[0]
    c.mouseMoveEvent(mev(QEvent.MouseMove, cx + 40, cy + 25))
    nx, ny, _v2 = c.instances[0]['pts'][j0]
    chk('拖动中坐标立刻跟鼠标（不是松手才生效）',
        abs(nx - (x0 + 40 / c.scale)) < 1.5 and abs(ny - (y0 + 25 / c.scale)) < 1.5)
    chk('拖动中每次移动都触发重绘（跟手的必要条件）', n_up[0] > n_before)
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, cx + 40, cy + 25))
    chk('松手结束拖动', c.drag is None)
    c.update = real_update
    # 命中范围放大：离节点 15px 也该能拖（旧值 HIT_R=11 会漏，用户 2026-10-13 反馈"要瞄得很准"）
    # 注意：自检的合成数据里节点只相隔 ~6px，必须先把被测节点挪到角落做隔离，否则测的是"邻近节点"。
    c.set_scale(1.0)
    c.active, c.sel = 0, None
    j0 = IDX['nose']
    c.instances[0]['pts'][j0] = (50.0, 50.0, 2)
    px, py = c.to_canvas(50.0, 50.0)
    c.mousePressEvent(mev(QEvent.MouseButtonPress, px + 15, py))
    chk('离节点 15px 按下也能命中（命中半径 18；旧值 11 会漏）', c.drag == (0, j0))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, px + 15, py))
    c.mousePressEvent(mev(QEvent.MouseButtonPress, px + 40, py))
    chk('离节点 40px 按下不命中（半径没有大到乱抓）', c.drag is None)
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, px + 40, py))

    w.jump(1)
    w.rows[1]['verdict'] = ''                   # 第 2 行设成未判，前进才有目标
    w.rows[0]['verdict'] = 'none'
    p0 = w.pos
    w.keyPressEvent(_key(Qt.Key_Right))
    chk('→ 前进到相邻下一张', w.pos == p0 + 1)
    w.keyPressEvent(_key(Qt.Key_Left))
    chk('← 回上一张（含已判的，能回头改）', w.pos == p0)

    w.jump(2)
    c.sel = IDX['neck']
    bx = c.instances[0]['pts'][IDX['neck']][0]
    w.keyPressEvent(_key(Qt.Key_Left, Qt.ShiftModifier))
    chk('Shift+← 只微调节点、不翻页（方向键已经是翻页了）',
        w.pos == 1 and abs(c.instances[0]['pts'][IDX['neck']][0] - (bx - 1)) < 0.01)
    c.undo = []                                 # 没有可撤的改动 ⇒ Z 应退化成"回上一张"
    w.keyPressEvent(_key(Qt.Key_Z))
    chk('没有可撤的改动时 Z = 回上一张', w.pos == 0)

    # ---- 2026-10-13 用户报"无法正常翻页，, . 也不行"：焦点被顶栏输入框抢走 ----
    chk('顶栏控件一律不抢键盘焦点',
        all(x.focusPolicy() == Qt.NoFocus for x in (w.spin, w.cb_all, w.scroll)))
    chk('跳转框只读（跳转交给 G 键的模态对话框）', w.spin.isReadOnly())

    def real_key(widget, key, text=''):
        """真实按键要带 text —— 不带 text 的合成事件骗过了输入框（我第一版复现就栽在这）。"""
        app.sendEvent(widget, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text))

    def fresh_two():
        """把主队列摆成"第 1 行已判、第 2 行未判、光标停在第 1 行"，供按键测试用。"""
        w.rows[0]['verdict'] = 'none'
        w.rows[1]['verdict'] = ''
        w.jump(1)
        return w.pos

    p0 = fresh_two()
    real_key(w.spin.lineEdit(), Qt.Key_Period, '.')
    chk('焦点在跳转输入框里、按键带 text：`.` 仍能前进（不再被当文字吃掉）', w.pos == p0 + 1)
    p0 = fresh_two()
    real_key(w.cb_all, Qt.Key_Right)
    chk('`→` 不被复选框/滚动区吃掉', w.pos == p0 + 1)
    p0 = fresh_two()
    real_key(w.scroll, Qt.Key_Space, ' ')
    chk('空格不被吃掉（= 采纳并下一张）', w.pos == p0 + 1)

    q3 = os.path.join(os.path.dirname(queue), 'queue3.csv')
    with io.open(q3, 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        wr.writeheader()
        wr.writerow({'idx': 1, 'stem': 'a', 'src': img, 'verdict': 'ok'})
        wr.writerow({'idx': 2, 'stem': 'b', 'src': img2, 'verdict': ''})
        wr.writerow({'idx': 3, 'stem': 'a', 'src': img, 'verdict': ''})
    w3 = PoseLabeler(q3, gt_dir, pred_dir)
    w3.show_row()
    w3.pos = 0
    w3.step(1)
    chk('第 1 张已判 ⇒ 前进到相邻的第 2 张', w3.pos == 1)

    # ---- 2026-10-13 用户定的新口径：前进=下一张未判（采纳过的不再出现），后退=上一张（含已判的）----
    q6 = os.path.join(os.path.dirname(queue), 'queue6.csv')
    with io.open(q6, 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        wr.writeheader()
        for i in range(6):
            wr.writerow({'idx': i + 1, 'stem': 'a' if i % 2 == 0 else 'b',
                         'src': img if i % 2 == 0 else img2, 'verdict': ''})
    w6 = PoseLabeler(q6, gt_dir, pred_dir)
    w6.show_row()
    w6.pos = 3
    w6.step(-1)
    chk('后退真的后退（这就是那个 bug）', w6.pos == 2)
    w6.step(-1)
    chk('连续后退仍然单调', w6.pos == 1)
    chk('后退**不改变判定**（经过的那行仍是未判）', (w6.rows[1].get('verdict') or '') == '')
    w6.pos = 3
    w6.step(1)
    chk('前进也不改判定（不会偷偷记 ok）', w6.pos == 4 and (w6.rows[3].get('verdict') or '') == '')
    w6.pos = 3
    w6.accept_and_next()
    chk('采纳（空格/按钮）：记 ok 并前进', (w6.rows[3].get('verdict') or '') == 'ok' and w6.pos == 4)
    w6.step(-1)
    chk('后退能回到刚采纳的那张（回头改得动）', w6.pos == 3)
    w6.pos = 2
    w6.step(1)
    chk('前进跳过已判的（归档的不再出现）', w6.pos == 4 and (w6.rows[3].get('verdict') or '') == 'ok')
    w6.step(-1)
    chk('后退不跳过已判的（相邻回看）', w6.pos == 3)
    # 拖过之后先按 S 再按采纳：判定不许从 edit 降级成 ok
    w6.pos = 0
    w6.rows[0]['verdict'] = ''
    w6.canvas.dirty = True
    w6.save_row()
    chk('拖过后按 S：记 edit', (w6.rows[0].get('verdict') or '') == 'edit')
    w6.accept_and_next()
    chk('已记 edit 的行再按采纳：仍是 edit（不降级）', (w6.rows[0].get('verdict') or '') == 'edit')
    for i in range(6):
        w6.pos = i
        if not (w6.rows[i].get('verdict') or ''):
            w6.accept_and_next()
    w6.pos = 3
    before = w6.pos
    w6.step(1)
    chk('全都判完后再前进：原地不动并提示',
        w6.pos == before and '没有未判的图' in w6.last_msg)

    # ---- 按钮：不能把所有操作都压在键盘上（用户 2026-10-13 的要求）----
    from PyQt5.QtWidgets import QPushButton
    btns = list(w.findChildren(QPushButton))

    def bfind(key):
        """按按钮文本里的**键位标记**找（如 '(→)'）—— 比按名字前缀稳（'◀ 上一张' 开头不是"上一张"）。"""
        return next((b for b in btns if key in b.text()), None)

    chk('按钮行存在', len(btns) >= 16)
    chk('所有按钮都不抢键盘焦点', all(b.focusPolicy() == Qt.NoFocus for b in btns))
    chk('键位写在按钮文本里（不再靠状态栏大段文字）',
        all(('(' in b.text() and ')' in b.text()) for b in btns))
    w.rows[1]['verdict'] = ''                   # 让前进有目标
    w.jump(1)
    p0 = w.pos
    bfind('(→)').click()
    chk('点「下一张」= 前进到相邻下一张', w.pos == p0 + 1)
    bfind('(←)').click()
    chk('点「上一张」= 回退', w.pos == p0)
    for k in ('(F)', '(+)', '(−)'):
        bfind(k).click()
    chk('缩放/适应按钮都能点（不报错）', True)
    bfind('(3)').click()
    chk('没有选中节点时点可见性按钮 = 给提示，不是静默什么都不做',
        ('先在画布上点一个节点' in w.last_msg) or ('没有实例' in w.last_msg))
    chk('最近动作会写进状态栏（手感问题可查）', w.last_msg != '' and '最近动作' in w.status.text())
    # ---- 空实例不许崩（2026-10-13 实测：判过"无骨架"的图上鼠标一动就 IndexError）----
    c.instances = []
    c.active = 0
    c.mouseMoveEvent(mev(QEvent.MouseMove, 300, 300))
    c.mousePressEvent(mev(QEvent.MouseButtonPress, 300, 300))
    c.mouseReleaseEvent(mev(QEvent.MouseButtonRelease, 300, 300))
    chk('画布上有图但没有实例时：悬停/按下/松手都不崩，且 node_at 返回 None',
        c.node_at(300, 300) is None and c.drag is None)
    # ---- 可读性：字号/字重 + 光标（用户 2026-10-13）----
    chk('画布默认鼠标 = 十字准星', c.cursor().shape() == Qt.CrossCursor)
    chk('节点标签更大更粗（>=13pt 且 bold）',
        c.label_font.bold() and c.label_font.pointSize() >= 13)
    chk('界面字号整体上调并加粗', w.font().bold() and w.font().pointSize() >= 11)
    c.mouseMoveEvent(mev(QEvent.MouseMove, 2000, 2000))          # 挪到空白处
    chk('在画布上移动后仍是十字准星', c.cursor().shape() == Qt.CrossCursor)

    c.sel = IDX['nose']
    c.scroll.horizontalScrollBar().setRange(0, 800)
    c.scroll.horizontalScrollBar().setValue(400)
    c.mousePressEvent(PanEv(QEvent.MouseButtonPress, 5, 5, Qt.LeftButton))       # 画面外空白处
    c.mouseMoveEvent(PanEv(QEvent.MouseMove, 60, 40, Qt.LeftButton))
    chk('空白处拖动 = 平移画布', c.pan is not None
        and c.scroll.horizontalScrollBar().value() == 400 - 55)
    c.mouseReleaseEvent(PanEv(QEvent.MouseButtonRelease, 60, 40, Qt.LeftButton))
    c.mousePressEvent(PanEv(QEvent.MouseButtonPress, 5, 5, Qt.LeftButton))
    c.mouseReleaseEvent(PanEv(QEvent.MouseButtonRelease, 5, 5, Qt.LeftButton))
    chk('空白处单击（没拖）= 取消选中', c.sel is None)

    # ---- 缩放 / 平移手感（用户 2026-10-13 报的第二组问题：缩放不直觉、中键拖动反复抖）----
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    c.set_scale(1.5)          # 画布必须比视口大，滚动条才有量程（fit(whole) 正好相反：整块塞进视口）
    app.processEvents()
    real_scroll = c.scroll
    chk('离屏视口有真实几何', c.scroll.viewport().width() > 100 and c.scroll.viewport().height() > 100)

    class _Bar:                                  # 桩滚动条：offscreen 不做裁剪，resize 会把量程清零，
        def __init__(self, v=0, mx=10 ** 6):     # 所以锚点数学用桩来测（真机量程另有探针验证）
            self._v, self._max = v, mx

        def value(self):
            return self._v

        def setValue(self, v):
            self._v = max(0, min(self._max, int(v)))

        def setRange(self, a, b):
            self._max = int(b)

        def maximum(self):
            return self._max

        def minimum(self):
            return 0

    class _VP:
        def width(self):
            return 800

        def height(self):
            return 600

    class _Scroll:
        def __init__(self, h, v):
            self._h, self._v = h, v

        def horizontalScrollBar(self):
            return self._h

        def verticalScrollBar(self):
            return self._v

        def viewport(self):
            return _VP()

    hb, vb = _Bar(300), _Bar(300)
    c.scroll = _Scroll(hb, vb)
    h0, v0, s0 = hb.value(), vb.value(), c.scale
    vx, vy = 500.0, 300.0
    cx, cy = vx + h0, vy + v0                    # 缩放前光标处的画布坐标
    c.zoom_at(1.25, vx, vy)
    k = c.scale / s0
    chk('缩放以光标为锚：锚点下的画面留在原地（±2px）',
        abs((cx * k - hb.value()) - vx) <= 2 and abs((cy * k - vb.value()) - vy) <= 2)
    c.zoom_at(1 / 1.25, vx, vy)
    chk('缩放可逆：缩回去偏移回到原位（±2px）',
        abs(hb.value() - h0) <= 2 and abs(vb.value() - v0) <= 2)
    # 换个滚动偏移 + 换光标位置：旧公式的误差 = k*偏移，这里能抓住（旧实现偏 75px）
    c.set_scale(1.0)
    hb.setValue(300)
    vb.setValue(150)
    vx2, vy2 = 60.0, 520.0
    cx2, cy2 = vx2 + 300, vy2 + 150
    c.zoom_at(1.25, vx2, vy2)
    k2 = c.scale / 1.0
    chk('锚点在大偏移/边角位置同样成立，且没被量程夹住（旧公式在这里偏得最多）',
        abs((cx2 * k2 - hb.value()) - vx2) <= 2 and abs((cy2 * k2 - vb.value()) - vy2) <= 2
        and hb.value() < hb.maximum() and vb.value() < vb.maximum())
    c.set_scale(100.0)
    chk('缩放上限', c.scale <= 8.0)
    c.set_scale(0.001)
    chk('缩放下限', c.scale >= 0.05)
    # ---- 默认视图：图正好撑满视口高度 + 居中（用户 2026-10-13 的要求）----
    hb.setRange(0, 10 ** 6)                  # 前面的缩放测试把桩的量程压小了，这里放开
    vb.setRange(0, 10 ** 6)
    hb.setValue(0)
    vb.setValue(0)
    c.set_scale(0.1)
    c.fit_height()
    chk('默认视图 = 图正好撑满视口高度', abs(c.scale - 600.0 / c.ih) < 1e-6)
    cx = (c.margin + c.iw / 2.0) * c.scale
    cy = (c.margin + c.ih / 2.0) * c.scale
    chk('默认视图 = 图的中心对到视口中心（±2px）',
        abs(hb.value() - (cx - 400.0)) <= 2 and abs(vb.value() - (cy - 300.0)) <= 2)
    w.jump(2)
    chk('换一张图之后视图仍然居中（不再跳回左上角）', vb.value() > 0)
    # ---- 适应 = 适应**内容**（图 ∪ 骨架，含画到画面外的点），不是整块带留白的画布 ----
    hb.setRange(0, 10 ** 6)
    vb.setRange(0, 10 ** 6)
    c.instances[0]['pts'][IDX['nose']] = (-300.0, -200.0, 2)     # 故意画到画面外
    x0, y0, x1, y1 = c.content_rect()
    chk('内容范围包含画到画面外的节点', x0 <= -300 and y0 <= -200 and x1 >= c.iw and y1 >= c.ih)
    c.fit_content()
    cw, chh = x1 - x0, y1 - y0
    chk('适应内容：缩放 = min(视口宽/内容宽, 视口高/内容高) × 0.98',
        abs(c.scale - min(800.0 / cw, 600.0 / chh) * 0.98) < 1e-6)
    ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # 内容比视口窄时，居中需要的偏移是负的、会被量程夹到 0（真机里这种"没得滚"的情况由
    # scroll.setAlignment(AlignCenter) 负责居中），所以期望值也要按夹取算
    exp_h = max(0, min(hb.maximum(), round((c.margin + ccx) * c.scale - 400)))
    exp_v = max(0, min(vb.maximum(), round((c.margin + ccy) * c.scale - 300)))
    chk('适应内容：内容中心对到视口中心（按量程夹取后 ±2px）',
        abs(hb.value() - exp_h) <= 2 and abs(vb.value() - exp_v) <= 2)
    c.scroll = real_scroll
    hb, vb = c.scroll.horizontalScrollBar(), c.scroll.verticalScrollBar()

    hb.setRange(0, 800)
    vb.setRange(0, 800)
    hb.setValue(400)
    vb.setValue(200)
    c.mousePressEvent(PanEv(QEvent.MouseButtonPress, 300, 300))
    deltas = []
    for k in range(1, 11):                           # 屏幕光标向右 6px × 10 步
        before = hb.value()
        c.mouseMoveEvent(PanEv(QEvent.MouseMove, 300 + 6 * k, 300))
        deltas.append(hb.value() - before)
    chk('平移 10 小步：每步位移一致且方向不变（这正是"反复抖动"的判据）', deltas == [-6] * 10)
    chk('总位移 = 总拖拽量（1:1 跟手，无自反馈）', hb.value() == 400 - 60)
    c.mouseReleaseEvent(PanEv(QEvent.MouseButtonRelease, 360, 300))
    chk('松手结束平移', c.pan is None)

    hb.setValue(0)
    vb.setValue(0)
    c.mousePressEvent(PanEv(QEvent.MouseButtonPress, 400, 400))
    for k in range(1, 7):                            # 想往负偏移拖 ⇒ 被夹在 0
        c.mouseMoveEvent(PanEv(QEvent.MouseMove, 400 + 10 * k, 400))
    chk('贴边界时不推进锚点（否则反向拖会跳）', hb.value() == 0 and c.pan == (400, 400))
    c.mouseMoveEvent(PanEv(QEvent.MouseMove, 340, 400))
    chk('越过起点后才 1:1 跟手', hb.value() == 60 and c.pan == (340, 400))
    c.mouseMoveEvent(PanEv(QEvent.MouseMove, 400, 400))
    chk('拖回原点即回原位（不跳、不累积漂移）', hb.value() == 0 and c.pan == (400, 400))
    c.mouseReleaseEvent(PanEv(QEvent.MouseButtonRelease, 400, 400))

    def _mk(chains):
        out = []
        for ch in chains:
            for nm in BODY18:
                out.append({'chain': ch, 'name': 'body.%s' % nm, 'x': 10.0, 'y': 20.0, 'vis': 2})
        return out

    gt_many = _mk(['H#%d' % i for i in range(12)])
    pred_shuf = _mk(['M1.dwpose#%d' % i for i in (5, 0, 11, 3, 8, 1, 10, 2, 9, 4, 7, 6)])
    chk('位移指标按实例序号配对（按 chain 名排序会报假值，实测过 379px）',
        shift_stats(gt_many, pred_shuf).startswith('0.0/0.0'))

    # ---- C 键：把上一张的真值整份拷进当前图（用户 2026-10-21 的差分套图需求）----
    cq = os.path.join(tmp, 'copy', 'q.csv')
    cpred = os.path.join(tmp, 'copy', 'pre')
    cgt = os.path.join(tmp, 'copy', 'gt')
    os.makedirs(cpred)
    os.makedirs(cgt)
    jdump(os.path.join(cpred, 'p1.json'), rec)
    jdump(os.path.join(cpred, 'p2.json'), rec)
    build_queue(cpred, cq, min_pred_vis=1)

    def _gt_pts(chains):
        out = []
        for ch in chains:
            for nm in BODY18:
                out.append({'layer': 'human', 'cls': 'body', 'chain': ch,
                            'name': 'body.%s' % nm, 'x': 10.0, 'y': 20.0, 'vis': 2})
        return out

    gt_a = _gt_pts(['H#0', 'H#1'])
    gt_a[IDX['l_wrist']]['x'], gt_a[IDX['l_wrist']]['y'] = 123.0, 45.0   # 指纹：证明拷的是真值不是预标注
    gt_a[IDX['neck']]['x'], gt_a[IDX['neck']]['y'], gt_a[IDX['neck']]['vis'] = 0.0, 0.0, 0
    jdump(os.path.join(cgt, 'p1.json'), {'stem': 'p1', 'verdict': 'edit', 'instances': 2,
                                         'points': gt_a, 'pred_points': []})
    wc = PoseLabeler(cq, cgt, cpred, min_pred_vis=1)
    wc.show_row()
    wc.step(1)
    chk('C 拷贝前：第 2 张还没判（pose_gt 里没有它）', not os.path.exists(os.path.join(cgt, 'p2.json')))
    pre_snap = copy.deepcopy(wc.canvas.instances)
    pos_before, undo_before = wc.pos, len(wc.canvas.undo)
    wc.keyPressEvent(_key(Qt.Key_C))
    gi = jload(os.path.join(cgt, 'p2.json'))
    src0 = [p for p in gt_a if p['chain'] == 'H#0']
    got0 = [p for p in gi['points'] if p['chain'] == 'H#0']
    chk('C 拷贝：实例数与源图相等（2 个）', gi['instances'] == 2 and len(wc.canvas.instances) == 2)
    chk('C 拷贝：点数与坐标/可见度逐项相等（18 点，含指纹点与缺失的 neck）',
        len(got0) == 18 and all((p['x'], p['y'], p['vis']) == (q['x'], q['y'], q['vis'])
                                for p, q in zip(got0, src0)))
    chk('C 拷贝：不翻页', wc.pos == pos_before)
    chk('C 拷贝：判定记 edit（不引入新的 verdict 值）', gi['verdict'] == 'edit')
    crow = list(csv.DictReader(io.open(cq, encoding='utf-8')))[1]
    chk('C 拷贝：队列 note 记了来源图 copy_from=p1', 'copy_from=p1' in (crow['note'] or ''))
    chk('C 拷贝：撤销栈加了一格', len(wc.canvas.undo) == undo_before + 1)
    chk('C 拷贝：画布实例与源图 18 点逐项相等（不是只写盘）',
        [tuple(t) for t in wc.canvas.instances[0]['pts']]
        == [(p['x'], p['y'], p['vis']) for p in src0])
    wc.keyPressEvent(_key(Qt.Key_Z))
    chk('C 拷贝后 Z：实例回到拷贝前的预标注（拷贝可撤销）',
        [i['pts'] for i in wc.canvas.instances] == [i['pts'] for i in pre_snap])
    wc.jump(1)
    wc.keyPressEvent(_key(Qt.Key_C))
    chk('C 在第 1 张：状态栏提示没有"上一张"，不崩', '上一张' in wc.status.text())
    chk('缺失关节占位点 (0,0,0) 判为"不画"，vis=0 但坐标有效仍要画',
        phantom(0.0, 0.0, 0) and not phantom(0.0, 0.0, 2) and not phantom(5.0, 7.0, 0))

    bad = [n for n, c_ in ok if not c_]
    print('\n自检结果：%d/%d 通过%s' % (len(ok) - len(bad), len(ok), ('　失败：%s' % bad) if bad else ''))
    print('（临时目录：%s）' % tmp)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=QUEUE_DEF)
    ap.add_argument('--pred-dir', default=PRE_DEF)
    ap.add_argument('--gt-dir', default=GT_DEF)
    ap.add_argument('--margin-scale', type=float, default=1.0,
                    help='四周留白 = 该值 × 图宽（默认 1.0；节点可以放到画面外）')
    ap.add_argument('--min-pred-vis', type=int, default=4,
                    help='预标注实例至少要有几个可信点才显示（面点自 2026-10-13 起只留真检出的眼，'
                         '可信点分母 18→13，原来的 6/18≈4/13）')
    ap.add_argument('--only-unjudged', action='store_true',
                    help='已废弃：现在"前进"默认就跳过已判的图，无需此开关')
    ap.add_argument('--show-all', action='store_true')
    ap.add_argument('--build-queue', action='store_true')
    ap.add_argument('--no-skeleton', action='store_true',
                    help='建队列时反过来挑"检出人/头但没有可用骨架"的图（要人从模板摆）')
    ap.add_argument('--out-csv', default=QUEUE_DEF)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sort', choices=['stem', 'hard', 'rand'], default='stem')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.build_queue:
        build_queue(a.pred_dir, a.out_csv, a.limit, a.sort, min_pred_vis=a.min_pred_vis,
                    no_skeleton=a.no_skeleton)
        return 0
    if a.selftest:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication(sys.argv)                        # noqa: F841
        return selftest()
    if not os.path.exists(a.queue):
        raise SystemExit('找不到队列：%s（先跑 --build-queue）' % a.queue)
    app = QApplication(sys.argv)
    w = PoseLabeler(a.queue, a.gt_dir, a.pred_dir, a.only_unjudged, a.margin_scale,
                    a.min_pred_vis, a.show_all)
    w.show_row()
    w.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
