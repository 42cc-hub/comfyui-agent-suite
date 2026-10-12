"""M2b 面部组（gag / blindfold / **eye / mouth**）标注器 —— 画框，或者按一下说"没有"。

口径演变：
  - **一轮只处理一个目标**；同一张 head 有多个目标时拆成连续多轮（队列里就排在一起）；
  - 2026-10-11：一度取消预标注 —— 第一版预标注是"WD14 标签 + face/eye 框几何推的"，位置普遍偏，
    用户判了 41 轮**一轮都没采纳**（原话「与其让我再点一次不通过，不如不要给预标注了」）；
  - **2026-10-13：重新给预标注，但改成"可采纳"**。前提是这轮预标注换了来源、且逐图验过：
    眼框来自现役 `m2b_face_v3_3983`（实测可信），嘴框由 **眼线朝向 + 脸框** 推（14/14 落在嘴上，含侧躺）。
    未判轮次预标注直接落在画布上 ⇒ **按 1 采纳、拖角微调、右键删掉重画、按 3 = 没有这个目标**。
    ⚠️ **只有未判轮次才注入**（已判轮次永远显示 CSV 里的真值）—— 10-11 那个"预标注覆盖人工框"的护栏不动。
  - `verdict` 取值：`draw`（人画的/采纳的框）/ `neg`（没有这个目标）；
    老口径留下的 `pass`/`fail` 仍然按记录正常显示与导出（向后兼容）。

  - **2026-10-14：加 `A` = 整张 head 一次采纳**。覆盖补全队列里一张 head 有四个目标（eye/mouth/gag/blindfold），
    逐目标按 1 是 4 次按键/张；预填对的时候按一下 `A` 就把这张 head 全落盘，只在该目标**没有预填**时停下让人画
    （没预填 = 可能漏检，绝不替人猜"没有"）。已判轮次永不覆盖。

键位：1 或 → 画好了/下一轮（没画框时 = 没有这个目标）　3 没有这个目标　**A 采纳本张 head 的全部预填**
      ←/→ 上下轮　Z 撤销最后一个框　C 清空本轮的框　右键点框 = 删掉那个框
      F 适应窗口　+/- 缩放　S 保存　U 只看未判　G 跳到第 N 轮
画框：空白处拖 = 新建框；框内拖 = 移动；四角拖 = 改大小。

用法：python scripts/label_face.py --queue data/m2b_queue/index.csv
      python scripts/label_face.py --selftest        # 无人值守自检（离屏，不弹窗）
"""
import argparse
import csv
import io
import os
import sys

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PRE = os.path.join(PROJ, 'data', 'prelabel')
FIELDS = ['idx', 'crop', 'source', 'head', 'target', 'prop', 'prop_src', 'prop_conf',
          'wd14', 'verdict', 'boxes', 'edited', 'note']
TARGET_CN = {'gag': '口塞 gag', 'blindfold': '眼罩 blindfold', 'eye': '眼睛 eye', 'mouth': '嘴 mouth'}
HANDLE = 9                      # 角的抓取半径（屏幕像素）


def parse_box(s):
    try:
        v = [float(x) for x in str(s).split(',')]
        return v if len(v) == 4 else None
    except Exception:
        return None


def fmt_boxes(boxes):
    return ';'.join(','.join(str(int(round(v))) for v in b['r']) for b in boxes)


def parse_boxes(s):
    """把 CSV 里存的 `x0,y0,x1,y1;x0,y0,x1,y1` 解析回框列表（人画的真值）。"""
    out = []
    for part in str(s or '').split(';'):
        b = parse_box(part)
        if b:
            out.append(b)
    return out


class Canvas(QWidget):
    """自己画的画布（不用 QGraphicsView）：整张 256 图按 scale 缩放，框的坐标始终是图像坐标。

    `boxes` = **会被保存的真值**；`ref` = 只画不存的参照（原始预标注），标成虚线青、不参与命中测试。
    `dirty` 只在"人真的动过"时置位 —— 用户 2026-10-11 报的 bug 就出在这里：
    回头看已判轮次时画布只重画了预标注，再翻页就把预标注当成人画的结果写回去，覆盖了人的记录。
    """

    def __init__(self, on_change=None):
        super().__init__()
        self.pix = None
        self.scale = 1.0
        self.boxes = []                 # [{'r': [x0,y0,x1,y1], 'kind': 'prop'|'user'}]
        self.ref = None                 # 参照框（原始预标注），永不保存
        self.drag = None
        self.edited = False
        self.dirty = False
        self.on_change = on_change
        self.setMouseTracking(True)

    # ---- 基本 ----
    def set_image(self, path):
        self.pix = QPixmap(path)
        self.boxes = []
        self.ref = None
        self.drag = None
        self.dirty = False
        self.apply_scale()

    def apply_scale(self):
        if self.pix:
            self.setFixedSize(max(1, int(self.pix.width() * self.scale)),
                              max(1, int(self.pix.height() * self.scale)))
        self.update()

    def set_scale(self, s):
        self.scale = max(0.2, min(8.0, s))
        self.apply_scale()

    def prop_index(self):
        for i, b in enumerate(self.boxes):
            if b['kind'] == 'prop':
                return i
        return None

    def changed(self, edited=None, dirty=True):
        if edited is not None:
            self.edited = edited
        if dirty:
            self.dirty = True
        if self.on_change:
            self.on_change()

    # ---- 画 ----
    def paintEvent(self, _e):
        p = QPainter(self)
        if self.pix is not None and not self.pix.isNull():
            p.drawPixmap(self.rect(), self.pix)
        if self.ref:                                    # 参照：原始预标注（虚线青，只画不存）
            x0, y0, x1, y1 = [v * self.scale for v in self.ref]
            p.setPen(QPen(QColor(0, 200, 235), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))))
        for b in self.boxes:
            x0, y0, x1, y1 = [v * self.scale for v in b['r']]
            if b['kind'] == 'prop':
                p.setPen(QPen(QColor(0, 220, 255), 2, Qt.DashLine))
            else:
                p.setPen(QPen(QColor(90, 255, 130), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))))
            if b['kind'] == 'user':
                p.setBrush(QBrush(QColor(90, 255, 130)))
                p.setPen(QPen(QColor(90, 255, 130)))
                for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                    p.drawRect(int(cx - 3), int(cy - 3), 6, 6)

    # ---- 鼠标 ----
    def _hit(self, pos):
        """返回 (框下标, 角) ；角 ∈ tl/tr/bl/br；在框内但不在角上 → 角为 None。"""
        ix, iy = pos.x() / self.scale, pos.y() / self.scale
        for i in range(len(self.boxes) - 1, -1, -1):
            x0, y0, x1, y1 = self.boxes[i]['r']
            if abs(ix - x0) * self.scale <= HANDLE and abs(iy - y0) * self.scale <= HANDLE:
                return i, 'tl'
            if abs(ix - x1) * self.scale <= HANDLE and abs(iy - y0) * self.scale <= HANDLE:
                return i, 'tr'
            if abs(ix - x0) * self.scale <= HANDLE and abs(iy - y1) * self.scale <= HANDLE:
                return i, 'bl'
            if abs(ix - x1) * self.scale <= HANDLE and abs(iy - y1) * self.scale <= HANDLE:
                return i, 'br'
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                return i, None
        return None, None

    def mousePressEvent(self, e):
        if self.pix is None:
            return
        ix, iy = e.pos().x() / self.scale, e.pos().y() / self.scale
        i, corner = self._hit(e.pos())
        if e.button() == Qt.RightButton:
            if i is not None:
                del self.boxes[i]
                self.changed(edited=True)
            return
        if i is not None and corner:
            self.drag = dict(mode='resize', i=i, corner=corner, start=(ix, iy),
                             orig=list(self.boxes[i]['r']))
        elif i is not None:
            self.drag = dict(mode='move', i=i, start=(ix, iy), orig=list(self.boxes[i]['r']))
        else:
            self.drag = dict(mode='new', start=(ix, iy))
            self.boxes.append({'r': [ix, iy, ix, iy], 'kind': 'user'})
            self.drag['i'] = len(self.boxes) - 1
        self.update()

    def mouseMoveEvent(self, e):
        if not self.drag:
            return
        ix, iy = e.pos().x() / self.scale, e.pos().y() / self.scale
        d = self.drag
        if d['mode'] == 'new':
            r = self.boxes[d['i']]['r']
            r[2], r[3] = ix, iy
        elif d['mode'] == 'move':
            dx, dy = ix - d['start'][0], iy - d['start'][1]
            x0, y0, x1, y1 = d['orig']
            self.boxes[d['i']]['r'] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        else:
            x0, y0, x1, y1 = d['orig']
            c = d['corner']
            if 'l' in c:
                x0 = ix
            if 'r' in c:
                x1 = ix
            if 't' in c:
                y0 = iy
            if 'b' in c:
                y1 = iy
            self.boxes[d['i']]['r'] = [x0, y0, x1, y1]
        self.update()

    def mouseReleaseEvent(self, _e):
        if not self.drag:
            return
        mode = self.drag['mode']
        orig = self.drag.get('orig')
        i = self.drag.get('i')
        self.drag = None
        if i is None or i >= len(self.boxes):
            return
        r = self.boxes[i]['r']
        x0, x1 = sorted((r[0], r[2]))
        y0, y1 = sorted((r[1], r[3]))
        r = [max(0.0, min(x0, 255.0)), max(0.0, min(y0, 255.0)),
             max(0.0, min(x1, 255.0)), max(0.0, min(y1, 255.0))]
        if r[2] - r[0] < 3 or r[3] - r[1] < 3:
            del self.boxes[i]                      # 误点：不留退化框
            if mode != 'new':
                self.changed(edited=True)          # 但"把已有框拖没了"算真改动
            self.update()
            return
        self.boxes[i]['r'] = r
        # 只点了一下、没真正移动（例如想把预标注放大看清楚）⇒ 不算修改，
        # 否则一次空点击就会把"预标注"标成"人改过的框"，把通过率算脏
        if mode != 'new' and orig is not None and all(abs(a - b) <= 0.5 for a, b in zip(r, orig)):
            self.update()
            return
        if self.boxes[i]['kind'] == 'prop':
            self.boxes[i]['kind'] = 'user'         # 动过预标注 ⇒ 算"人工改过的框"
            p = None
        else:
            p = self.prop_index()                  # 新画了框 ⇒ 预标注作废
        if p is not None:
            del self.boxes[p]
        self.changed(edited=True)
        self.update()


class Labeler(QMainWindow):
    def __init__(self, queue, only_unlabeled=False):
        super().__init__()
        self.setWindowTitle('M2b 面部组标注器（gag / blindfold / eye / mouth）—— anime_detection_suite')
        self.resize(1180, 900)
        with io.open(queue, encoding='utf-8-sig') as fh:
            self.rows = list(csv.DictReader(fh))
        self.queue = queue
        self.pos = 0
        self.only_unlabeled = only_unlabeled
        self.edited = False
        self._loading = False
        self.fit_scale = 1.0
        self._build_ui()
        self.rebuild_view()

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)

        self.scroll = QScrollArea()
        self.scroll.setStyleSheet('background:#111;')
        self.canvas = Canvas(on_change=self.on_canvas_change)
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        h.addWidget(self.scroll, 3)

        side = QVBoxLayout()
        h.addLayout(side, 1)

        self.head = QLabel('—')
        self.head.setStyleSheet('font-size:17px; font-weight:bold;')
        self.head.setWordWrap(True)
        side.addWidget(self.head)

        self.meta = QLabel('—')
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet('color:#555;')
        side.addWidget(self.meta)

        side.addSpacing(6)
        self.hint = QLabel('')
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('color:#06c; font-weight:bold;')
        side.addWidget(self.hint)

        side.addSpacing(10)
        self.b_done = QPushButton('1 / →　画好了，下一轮')
        self.b_done.clicked.connect(self.do_enter)
        side.addWidget(self.b_done)
        self.b_none = QPushButton('3　没有这个目标')
        self.b_none.clicked.connect(self.do_none)
        side.addWidget(self.b_none)
        self.b_head = QPushButton('A　采纳本张 head 的全部预填')
        self.b_head.clicked.connect(self.adopt_head)
        side.addWidget(self.b_head)

        side.addSpacing(8)
        nav = QHBoxLayout()
        self.b_prev = QPushButton('← 上一轮')
        self.b_next = QPushButton('下一轮 →')
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(+1))
        nav.addWidget(self.b_prev)
        nav.addWidget(self.b_next)
        side.addLayout(nav)

        edit = QHBoxLayout()
        b_undo = QPushButton('撤销框 (Z)')
        b_clear = QPushButton('清空 (C)')
        b_undo.clicked.connect(self.undo_box)
        b_clear.clicked.connect(self.clear_boxes)
        edit.addWidget(b_undo)
        edit.addWidget(b_clear)
        side.addLayout(edit)

        jump = QHBoxLayout()
        jump.addWidget(QLabel('跳到 #'))
        self.spin = QSpinBox()
        self.spin.setRange(1, 99999)
        self.spin.valueChanged.connect(self.jump)
        jump.addWidget(self.spin)
        side.addLayout(jump)

        zoom = QHBoxLayout()
        for txt, s in (('适应(F)', None), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _c, ss=s: self.set_zoom(ss))
            zoom.addWidget(b)
        side.addLayout(zoom)

        self.cb_un = QCheckBox('只看未判（U）')
        self.cb_un.setChecked(self.only_unlabeled)
        self.cb_un.toggled.connect(self.set_only_unlabeled)
        side.addWidget(self.cb_un)

        side.addStretch(1)
        self.status = QLabel('')
        self.status.setStyleSheet('color:#060;')
        self.status.setWordWrap(True)
        side.addWidget(self.status)

    # ---------- 视图 ----------
    def view(self):
        out = []
        for i, r in enumerate(self.rows):
            if self.only_unlabeled and (r.get('verdict') or '').strip():
                continue
            out.append(i)
        return out

    def rebuild_view(self):
        self.v = self.view()
        if not self.v:
            self.head.setText('没有满足筛选条件的轮次')
            self.canvas.set_image('')
            return
        self.pos = min(self.pos, len(self.v) - 1)
        self.show_row()

    def current_row(self):
        return self.rows[self.v[self.pos]]

    def show_row(self):
        """重画一轮。

        两条用户 2026-10-11 定的规矩：
        1. **未判轮次不再画预标注**（"与其让我再点一次不通过，不如不要给预标注了"）——直接画框，或说"没有"；
        2. **已判轮次显示 CSV 里存下来的真值**（绿框），不许因为翻页被原始预标注覆盖（当天报的 bug）。
        老口径判过的轮次（`pass`/`fail`）仍按记录显示，只是把原始预标注降级成虚线青参照（`ref`，永不保存）。
        """
        r = self.current_row()
        self._loading = True
        rounds_here = [x for x in self.rows if x['crop'] == r['crop']]
        which = rounds_here.index(r) + 1
        v = (r.get('verdict') or '').strip()
        saved = parse_boxes(r.get('boxes'))
        mark = '✔已判' if v else '·未判'
        self.head.setText('%s　#%s　%s　%s' % (mark, r['idx'], TARGET_CN.get(r['target'], r['target']),
                                             '本张第 %d/%d 轮' % (which, len(rounds_here))))
        prop = parse_box(r.get('prop'))
        what = {'draw': '有目标 · 人工画的框', 'pass': '有目标 · 采纳了预标注',
                'fail': '有目标 · 否掉预标注后人工画的', 'neg': '没有这个目标'}.get(v, '（还没判）')
        self.meta.setText('%s\n来源: %s ｜ head%s\n记录: %s%s\n%s' % (
            r['crop'], r.get('source', ''), r.get('head', ''),
            ('%s ｜ %d 个框 %s' % (what, len(saved), r.get('boxes', ''))) if v else '（还没判）',
            ('　（原始预标注 (%d,%d)-(%d,%d) 只作对照，不再画出来）' % tuple(prop)) if (prop and v) else '',
            r.get('note', '')))
        self.canvas.edited = False
        self.canvas.set_image(os.path.join(PRE, r['crop']))     # 内部会把 dirty/ref 清干净
        if v and saved:
            self.canvas.boxes = [{'r': [float(x) for x in b], 'kind': 'user'} for b in saved]
            if v in ('pass', 'fail') and prop:
                same = any(all(abs(float(a) - float(c)) <= 1 for a, c in zip(b, prop)) for b in saved)
                self.canvas.ref = [float(x) for x in prop] if not same else None
            self.hint.setText('已记录：%s（%d 个框）%s'
                              % (what, len(saved),
                                 '　虚线青 = 原预标注（对照用，不会被保存）' if self.canvas.ref else ''))
        elif v == 'neg':
            self.canvas.boxes = []
            self.hint.setText('已记录：没有这个目标（要改成"有"，直接画框再按 1）')
        else:
            # 用户 2026-10-13 改的口径：**未判轮次给可采纳的预标注**（青/橙框），
            # 按 1 直接采纳、拖角微调、右键删掉重画、按 3 = 没有这个目标。
            # 只有"未判"才注入 —— 已判轮次永远显示 CSV 里的真值（10-11 那个覆盖 bug 的护栏不动）。
            props = parse_boxes(r.get('prop'))
            if props:
                self.canvas.boxes = [{'r': [float(x) for x in b], 'kind': 'prop'} for b in props]
                self.canvas.dirty = False
                self.hint.setText('预标注 %d 个框（来源：%s）⇒ 按 1 采纳 / **A 整张 head 全采纳** / '
                                  '拖角微调 / 右键删掉重画 / 按 3 = 没有这个目标'
                                  % (len(props), r.get('prop_src') or '未知'))
            else:
                self.canvas.boxes = []                          # 没有预标注：纯手画
                self.hint.setText('鼠标拖出框（= 有目标），或按 1 / 3 = 没有这个目标；画完按 → 下一轮')
        self.canvas.update()
        self.fit()
        self._loading = False
        self.spin.setValue(int(r['idx']))
        n_lab = sum(1 for x in self.rows if (x.get('verdict') or '').strip())
        self.status.setText('进度 %d/%d ｜ 已判 %d/%d ｜ 本轮: %s ｜ 每步即时落盘'
                            % (self.pos + 1, len(self.v), n_lab, len(self.rows), v or '（未判）'))

    def fit(self):
        if self.canvas.pix is None or self.canvas.pix.isNull():
            return
        area = self.scroll.viewport().size()
        self.fit_scale = max(0.2, min(area.width() / max(1, self.canvas.pix.width()),
                                      area.height() / max(1, self.canvas.pix.height())))
        self.canvas.set_scale(self.fit_scale)

    def set_zoom(self, s):
        self.canvas.set_scale(self.fit_scale if s is None else s)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'canvas'):
            self.fit()

    # ---------- 行为 ----------
    def on_canvas_change(self):
        if self._loading:
            return
        self.hint.setText('已画好 —— 按 1 或 → 进入下一轮（按 C 清空重画）')

    def _apply(self, verdict, boxes, advance):
        r = self.current_row()
        r['verdict'] = verdict
        r['boxes'] = fmt_boxes(boxes)
        r['edited'] = '1' if self.canvas.edited else ''
        self.save()
        self.status.setText('✔ 已落盘：第 %s 轮 → %s，%d 个框 %s'
                            % (r['idx'], verdict, len(boxes), r['boxes'] or '（空）'))
        if advance:
            self.step(+1)
        else:
            self.show_row()

    def do_enter(self):
        """`1` / `→`：**画布上有框就存下来当"有目标"，没有框就记"没有这个目标"**。

        口径是用户 2026-10-11 改的：预标注质量太差（他判的 41 轮里 0 轮采纳），
        所以不再给预标注、也不再要"通过/不通过"两步 —— 直接"画框"或"没有"。
        """
        if self._loading or not self.v:
            return
        if self.canvas.boxes:
            self.commit_current(advance=True)
        else:
            self.do_none()

    def do_none(self):
        if self._loading or not self.v:
            return
        self.canvas.boxes = []
        self.canvas.edited = True
        self.canvas.dirty = True
        self.canvas.update()
        self._apply('neg', [], advance=True)

    def commit_current(self, advance=True):
        """把人画的框落到 CSV（`verdict=draw`），并守住"没动过就不覆盖已存记录"的规矩。"""
        r = self.current_row()
        v = (r.get('verdict') or '').strip()
        if self.canvas.boxes:
            if v in ('', 'draw'):
                r['verdict'] = 'draw'
                r['edited'] = '1'
            if self.canvas.dirty or not (r.get('boxes') or '').strip():
                r['boxes'] = fmt_boxes(self.canvas.boxes)
                r['edited'] = '1'
            self.canvas.dirty = False                   # 幂等：翻页时不会重复判定为"有改动"
            self.save()
            self.status.setText('✔ 已落盘：第 %s 轮 → 有目标，%d 个框 %s'
                                % (r['idx'], len(self.canvas.boxes), r['boxes']))
        elif v == '':
            self.status.setText('本轮还没画框也没说"没有" ⇒ 不记录')
        self.save()
        if advance:
            self.step(+1)

    def do_none(self):
        if self._loading or not self.v:
            return
        self.canvas.boxes = []
        self.canvas.edited = True
        self.canvas.dirty = True
        self.canvas.update()
        self._apply('neg', [], advance=True)

    def adopt_head(self):
        """`A`：一次采纳当前 head 的**全部**预填（四个目标），只在该目标没有预填时停下。

        用户 2026-10-14："以后的标注基本都是快速确认"。覆盖补全队列一张 head 排四个目标，
        预填对的时候逐目标按 1 要按四次；这里一次落盘整张 head。
        三条护栏不变：已判轮次永不覆盖；**没预填的轮次绝不替人猜"没有"**（停下让人画或按 3）；
        人已经在当前轮动过框，先把那一条按"人画的"落盘，不让 A 吞掉。
        """
        if self._loading or not self.v:
            return
        r = self.current_row()
        if self.canvas.dirty and self.canvas.boxes:
            self.commit_current(advance=False)
        n_adopt, pending = 0, None
        for x in self.rows:
            if x['crop'] != r['crop'] or (x.get('verdict') or '').strip():
                continue
            props = parse_boxes(x.get('prop'))
            if props:
                x['verdict'] = 'pass'
                x['boxes'] = fmt_boxes([{'r': b} for b in props])
                x['edited'] = ''
                n_adopt += 1
            elif pending is None:
                pending = x
        self.save()
        if pending is not None:
            i = next(i for i, x in enumerate(self.rows) if x is pending)
            if i in self.v:
                self.pos = self.v.index(i)
                self.show_row()
            self.status.setText('✔ 采纳 %d 个预填 ｜ 本张还有"%s"没预填（可能漏检）⇒ 画框或按 3'
                                % (n_adopt, TARGET_CN.get(pending['target'], pending['target'])))
            return
        nxt = None
        for j in range(self.pos + 1, len(self.v)):
            if self.rows[self.v[j]]['crop'] != r['crop']:
                nxt = j
                break
        if nxt is None:
            self.status.setText('✔ 采纳 %d 个预填 ｜ 已是最后一轮' % n_adopt)
        else:
            self.pos = nxt
            self.show_row()
            self.status.setText('✔ 本张 head 全部采纳（%d 个预填）⇒ 下一张' % n_adopt)

    def undo_box(self):
        if self.canvas.boxes:
            self.canvas.boxes.pop()
            self.canvas.edited = True
            self.canvas.dirty = True
            self.canvas.update()

    def clear_boxes(self):
        self.canvas.boxes = []
        self.canvas.edited = True
        self.canvas.dirty = True
        self.canvas.update()

    def step(self, d):
        if not self.v:
            return
        r = self.current_row() if 0 <= self.pos < len(self.v) else None
        if r is not None:
            v = (r.get('verdict') or '').strip()
            if self.canvas.boxes:
                self.commit_current(advance=False)      # 画了框就落盘（没动过则原样保留）
            elif v in ('draw', 'fail'):
                r['verdict'], r['boxes'] = '', ''        # 人把框删干净了 ⇒ 退回未判，不猜
                self.save()
                self.status.setText('第 %s 轮的框被清空 ⇒ 记为未判（确实没有目标请按 3）' % r['idx'])
        npos = self.pos + d
        if 0 <= npos < len(self.v):
            self.pos = npos
            self.show_row()
        else:
            self.status.setText('%s（已是视图%s）' % (self.status.text(), '末尾' if d > 0 else '开头'))

    def jump(self, n):
        for i, r in enumerate(self.rows):
            if int(r['idx']) == n and i in self.v:
                self.pos = self.v.index(i)
                self.show_row()
                return

    def set_only_unlabeled(self, on):
        self.only_unlabeled = on
        self.pos = 0
        self.rebuild_view()

    def save(self):
        with io.open(self.queue, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_1, Qt.Key_Return, Qt.Key_Enter):
            self.do_enter()
        elif k == Qt.Key_2:
            # 旧口径的"不通过"已取消（用户 2026-10-11：预标注质量差 ⇒ 索性不给预标注）。
            # 这里故意**不**把它映射成"没有目标"，免得手快按 2 记错数据。
            self.status.setText('已取消"不通过"这一步：直接画框，或按 1/3 记"没有这个目标"')
        elif k == Qt.Key_3:
            self.do_none()
        elif k == Qt.Key_A:
            self.adopt_head()
        elif k in (Qt.Key_Right, Qt.Key_Space):
            self.step(+1)
        elif k == Qt.Key_Left:
            self.step(-1)
        elif k == Qt.Key_Z:
            self.undo_box()
        elif k == Qt.Key_C:
            self.clear_boxes()
        elif k == Qt.Key_F:
            self.set_zoom(None)
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.canvas.set_scale(self.canvas.scale * 1.5)
        elif k == Qt.Key_Minus:
            self.canvas.set_scale(self.canvas.scale / 1.5)
        elif k == Qt.Key_S:
            self.save()
            self.status.setText('已保存 → %s' % self.queue)
        elif k == Qt.Key_U:
            self.cb_un.setChecked(not self.cb_un.isChecked())
        else:
            super().keyPressEvent(e)


def selftest():
    """无人值守自检：把关键路径都跑一遍（离屏），确认判定/画框/落盘三者一致。"""
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), 'index.csv')
    rows = [
        {'idx': 1, 'crop': 'x_head0_256.png', 'source': 'x', 'head': 0, 'target': 'gag',
         'prop': '100,180,160,220', 'prop_src': 'face+eye', 'prop_conf': '0.9', 'wd14': 'gag',
         'verdict': '', 'boxes': '', 'edited': '', 'note': 't'},
        {'idx': 2, 'crop': 'x_head0_256.png', 'source': 'x', 'head': 0, 'target': 'blindfold',
         'prop': '', 'prop_src': 'none', 'prop_conf': '', 'wd14': '', 'verdict': '', 'boxes': '',
         'edited': '', 'note': 't'},
        {'idx': 3, 'crop': 'y_head0_256.png', 'source': 'y', 'head': 0, 'target': 'gag',
         'prop': '80,170,150,215', 'prop_src': 'face', 'prop_conf': '0.9', 'wd14': 'cloth gag',
         'verdict': '', 'boxes': '', 'edited': '', 'note': 't'},
        {'idx': 4, 'crop': 'z_head0_256.png', 'source': 'z', 'head': 0, 'target': 'gag',
         'prop': '', 'prop_src': 'none', 'prop_conf': '', 'wd14': '', 'verdict': '', 'boxes': '',
         'edited': '', 'note': 't'},
    ]
    with io.open(tmp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # 用真实存在的图，免得 QPixmap 为空
    real = None
    for f in sorted(os.listdir(PRE)):
        if f.endswith('_head0_256.png'):
            real = f
            break
    for r in rows:
        r['crop'] = real
    with io.open(tmp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    w = Labeler(tmp)
    ok = []

    def chk(name, cond):
        ok.append((name, bool(cond)))
        print('  [%s] %s' % ('OK ' if cond else 'FAIL', name))

    w.show_row()
    # ---- 2026-10-13 口径：未判轮次给**可采纳**的预标注（kind=prop、不脏），已判轮次永远显示真值 ----
    chk('未判轮次注入可采纳的预标注', len(w.canvas.boxes) == 1
        and w.canvas.boxes[0]['kind'] == 'prop'
        and [int(v) for v in w.canvas.boxes[0]['r']] == [100, 180, 160, 220]
        and w.canvas.dirty is False)
    chk('未判轮次也不画"对照"', w.canvas.ref is None)
    w.canvas.boxes = [{'r': [110.0, 190.0, 170.0, 230.0], 'kind': 'user'}]
    w.canvas.dirty = True
    w.step(+1)                                   # 画了框 ⇒ 翻页即落盘
    chk('画框后翻页 ⇒ verdict=draw + 记录框', w.rows[0]['verdict'] == 'draw'
        and w.rows[0]['boxes'] == '110,190,170,230')
    w.step(-1)
    chk('回头看：画布显示记录的框', [int(v) for v in w.canvas.boxes[0]['r']] == [110, 190, 170, 230])
    chk('新口径的 draw 轮次不再画预标注参照', w.canvas.ref is None)
    chk('回头看：不算"未保存的改动"', w.canvas.dirty is False)
    w.step(+1)
    w.step(-1)
    chk('来回翻页不覆盖人画的框', w.rows[0]['boxes'] == '110,190,170,230'
        and w.rows[0]['verdict'] == 'draw')
    w.step(+1)
    # 没画框 ⇒ 按 1（= →）就是"没有这个目标"
    chk('翻到第2轮', w.pos == 1)
    w.do_enter()
    chk('没画框时按 1 = 没有这个目标', w.rows[1]['verdict'] == 'neg' and w.rows[1]['boxes'] == '')
    w.jump(2)
    w.do_none()                                  # 3 的路径
    chk('按 3 同样记"没有"', w.rows[1]['verdict'] == 'neg' and w.rows[1]['boxes'] == '')
    # 已判轮次里把框删干净再翻页 ⇒ 退回未判（不猜）
    w.jump(1)
    w.canvas.boxes = []
    w.canvas.dirty = True
    w.step(+1)
    chk('清空框后翻页 ⇒ 退回未判', w.rows[0]['verdict'] == '' and w.rows[0]['boxes'] == '')
    # 老口径判过的轮次仍要能正常显示与导出（向后兼容）：记录与预标注不同 ⇒ 虚线青当对照
    w.rows[0].update({'verdict': 'pass', 'boxes': '90,170,150,210'})
    w.jump(1)
    chk('老口径 pass 仍显示记录', [int(v) for v in w.canvas.boxes[0]['r']] == [90, 170, 150, 210])
    chk('老口径轮次保留预标注对照（ref）', w.canvas.ref == [100.0, 180.0, 160.0, 220.0])
    with io.open(tmp, 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        wr.writeheader()
        for r in w.rows:
            wr.writerow(r)
    with io.open(tmp, encoding='utf-8') as fh:
        back = list(csv.DictReader(fh))
    chk('落盘一致', [r['verdict'] for r in back] == ['pass', 'neg', '', '']
        and back[0]['boxes'] == '90,170,150,210')
    # ---- 2026-10-14 新键 A：整张 head 一次采纳（覆盖补全队列一张 head 四个目标）----
    real2 = next((f for f in sorted(os.listdir(PRE))
                  if f.endswith('_head0_256.png') and f != real), real)
    w.rows[0].update({'verdict': 'pass', 'boxes': '90,170,150,210', 'crop': real})
    w.rows[1].update({'verdict': '', 'boxes': '', 'crop': real})       # 未判且没预填
    w.rows[2].update({'verdict': '', 'boxes': '', 'crop': real})       # 未判但有预填
    w.rows[3].update({'crop': real2, 'prop': '70,160,140,205', 'verdict': '', 'boxes': ''})
    w.jump(1)
    w.adopt_head()
    chk('A 采纳本张有预填的轮次', w.rows[2]['verdict'] == 'pass' and w.rows[2]['boxes'] == '80,170,150,215')
    chk('A 不覆盖已判轮次', w.rows[0]['verdict'] == 'pass' and w.rows[0]['boxes'] == '90,170,150,210')
    chk('A 遇到没预填的轮次 ⇒ 停下不替人猜"没有"',
        w.rows[1]['verdict'] == '' and w.current_row()['idx'] == '2')
    w.rows[1]['prop'] = '60,150,130,200'
    w.adopt_head()
    chk('A 本张判完 ⇒ 跳到下一张 head',
        w.rows[1]['verdict'] == 'pass' and w.current_row()['idx'] == '4')
    w.adopt_head()
    chk('A 在最后一张 head 上照样落盘', w.rows[3]['verdict'] == 'pass'
        and w.rows[3]['boxes'] == '70,160,140,205')
    bad = [n for n, c in ok if not c]
    print('\n自检结果：%d/%d 通过%s' % (len(ok) - len(bad), len(ok), ('　失败：%s' % bad) if bad else ''))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(PROJ, 'data', 'm2b_queue', 'index.csv'))
    ap.add_argument('--only-unlabeled', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication(sys.argv)
        sys.exit(selftest())
    if not os.path.exists(a.queue):
        raise SystemExit('找不到队列：%s（先跑 scripts/m2b_propose.py）' % a.queue)
    app = QApplication(sys.argv)
    w = Labeler(a.queue, a.only_unlabeled)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
