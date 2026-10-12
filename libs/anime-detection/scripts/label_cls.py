"""M3（手/脚合理性）分类标注器 —— PyQt5 单图界面，翻页 + 单选框 + 键盘快捷键。

为什么要有它：59 张裁图用"看联系表 + 回一行文字"还勉强，几百张就不行了；而且**判"指/趾数对不对"必须能放大**。
所以这里做成一页一图：自动铺满/可放大到 4×、显示 WD14 自动给的状态与误检提示、点单选框即落盘并自动翻页。

标签口径（docs/labeling_spec.md §2/§2.1）：
  ok        数量与形状都对
  bad_count 数量不对（多/少指、多/少趾）——本项目最主要的目标错误
  bad_shape 数量说不清或明显畸形（粘成一坨、多一节、扭曲断开）
  skip      太小/糊/被遮/穿鞋穿袜 ⇒ 数不了，**不进训练集**
  not_part  这根本不是该部位（手框里是脚、脚框里是手）⇒ 给 M2a 当负样本；按 kind 自动写成 not_hand/not_foot

用法（在桌面会话里跑）：
  python scripts\\label_cls.py
  快捷键：1~5 = 选标签并自动翻页　←/→ = 上/下一张　+/- = 放大缩小　F = 适应窗口
          U = 只看未标注　B = 只看 bare　G = 跳到第 N 张　S = 立即保存
"""
import argparse
import csv
import io
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
                             QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CROPS = os.path.join(PROJ, 'data', 'crops_todo')
FIELDS = ['idx', 'crop', 'kind', 'source', 'origin', 'w', 'h', 'conf',
          'state', 'mismatch', 'bad_flags', 'label', 'auto_tags']

LABELS = [('ok', '数量与形状都对'),
          ('bad_count', '数量不对（多/少）'),
          ('bad_shape', '畸形/数不清'),
          ('skip', '判不了（小·糊·遮·穿鞋袜）'),
          ('not_part', '根本不是该部位（误检）')]


class Labeler(QMainWindow):
    def __init__(self, queue, only_bare=False, only_unlabeled=False):
        super().__init__()
        self.setWindowTitle('M3 分类标注器 —— anime_detection_suite')
        self.resize(1280, 860)
        with io.open(queue, encoding='utf-8-sig') as fh:      # 容忍 BOM（PowerShell/Excel 存过就会带）
            self.rows = list(csv.DictReader(fh))
        self.queue = queue
        self.pos = 0
        self.zoom = 0.0                 # 0 = 适应窗口
        self._loading = False
        self.only_bare = only_bare
        self.only_unlabeled = only_unlabeled
        self._build_ui()
        self.rebuild_view()

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)

        self.image = QLabel('(no image)')
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumWidth(820)
        self.image.setStyleSheet('background:#111; color:#888;')
        h.addWidget(self.image, 3)

        side = QVBoxLayout()
        h.addLayout(side, 1)

        self.head = QLabel('—')
        self.head.setStyleSheet('font-size:16px; font-weight:bold;')
        self.head.setWordWrap(True)
        side.addWidget(self.head)

        self.meta = QLabel('—')
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet('color:#555;')
        side.addWidget(self.meta)

        side.addSpacing(8)
        self.group = QButtonGroup(self)
        self.buttons = []
        for i, (key, desc) in enumerate(LABELS):
            b = QRadioButton('%d. %s — %s' % (i + 1, key, desc))
            self.group.addButton(b, i)
            # 用 clicked 而不是 toggled：后者在"已经是选中态"时不再发信号（上一张标签粘住时就点不动了）
            b.clicked.connect(lambda _c, k=key: self.on_label(k))
            side.addWidget(b)
            self.buttons.append(b)

        side.addSpacing(10)
        nav = QHBoxLayout()
        self.b_prev = QPushButton('← 上一张')
        self.b_next = QPushButton('下一张 →')
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(+1))
        nav.addWidget(self.b_prev); nav.addWidget(self.b_next)
        side.addLayout(nav)

        jump = QHBoxLayout()
        jump.addWidget(QLabel('跳到 #'))
        self.spin = QSpinBox(); self.spin.setRange(1, 9999)
        self.spin.valueChanged.connect(self.jump)
        jump.addWidget(self.spin)
        side.addLayout(jump)

        self.cb_bare = QCheckBox('只看 state=bare（穿鞋/穿袜的直接跳过）')
        self.cb_bare.setChecked(self.only_bare)
        self.cb_bare.toggled.connect(self.set_only_bare)
        side.addWidget(self.cb_bare)

        self.cb_un = QCheckBox('只看未标注（U）')
        self.cb_un.setChecked(self.only_unlabeled)
        self.cb_un.toggled.connect(self.set_only_unlabeled)
        side.addWidget(self.cb_un)

        zoom = QHBoxLayout()
        for txt, z in (('适应(F)', 0.0), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0)):
            b = QPushButton(txt); b.clicked.connect(lambda _c, zz=z: self.set_zoom(zz))
            zoom.addWidget(b)
        side.addLayout(zoom)

        side.addStretch(1)
        self.status = QLabel('')
        self.status.setStyleSheet('color:#060;')
        self.status.setWordWrap(True)
        side.addWidget(self.status)

    # ---------- 视图 ----------
    def view(self):
        out = []
        for i, r in enumerate(self.rows):
            if self.only_bare and (r.get('state') or '') != 'bare':
                continue
            if self.only_unlabeled and (r.get('label') or '').strip():
                continue
            out.append(i)
        return out

    def rebuild_view(self):
        self.v = self.view()
        if not self.v:
            self.head.setText('没有满足筛选条件的图'); self.image.setText('(空)')
            return
        self.pos = min(self.pos, len(self.v) - 1)
        self.show_row()

    def current_row(self):
        return self.rows[self.v[self.pos]]

    def show_row(self):
        r = self.current_row()
        self._loading = True
        self.head.setText('#%s  %s   %sx%s' % (r['idx'], r['kind'], r['w'], r['h']))
        hint = []
        if r.get('p_bad'):
            try:
                pb = float(r['p_bad'])
                hint.append('模型提示 p(bad)=%.2f%s' % (pb, '（倾向 bad）' if pb >= 0.5 else '（倾向 ok）'))
            except ValueError:
                pass
        if r.get('state'):
            hint.append('WD14 状态: %s' % r['state'])
        if r.get('mismatch'):
            hint.append('⚠ 疑似部位错: %s' % r['mismatch'])
        if r.get('bad_flags'):
            hint.append('WD14 弱先验: %s' % r['bad_flags'])
        self.meta.setText('%s\n来源: %s\n检测置信: %s\n%s' %
                          (r['crop'], r.get('source', ''), r.get('conf', ''),
                           ' | '.join(hint) if hint else '(WD14 无提示)'))
        lab = (r.get('label') or '').strip()
        self._clear_radios()                       # 切换后先重置：上一张的选择不能粘到下一张
        if lab:
            key = 'not_part' if lab.startswith('not_') else lab
            for i, (k, _d) in enumerate(LABELS):
                if k == key:
                    self.buttons[i].setChecked(True)
        self._loading = False
        self.spin.setValue(int(r['idx']))
        self.load_pixmap()
        n_lab = sum(1 for x in self.rows if (x.get('label') or '').strip())
        self.status.setText('进度 %d/%d 已标注 ｜ 当前视图 %d/%d ｜ 本张标签: %s' %
                            (self.pos + 1, len(self.v), n_lab, len(self.rows), lab or '（未标）'))

    def _clear_radios(self):
        """清空单选框。

        坑（用户 2026-10-10 报的）：**互斥 QButtonGroup 里不能直接 setChecked(False) 取消勾选**，
        必须临时 setExclusive(False) 再清空，否则上一张的选项会粘住 —— 切到标签相同的下一张时，
        点击"已经选中"的那个按钮不触发信号，看起来就是"点不动"。
        """
        self.group.setExclusive(False)
        for b in self.buttons:
            b.setChecked(False)
        self.group.setExclusive(True)

    def load_pixmap(self):
        r = self.current_row()
        p = os.path.join(CROPS, r['crop'])
        pm = QPixmap(p)
        if pm.isNull():
            self.image.setText('读不到图: %s' % p); return
        self.pm = pm
        self.render()

    def render(self):
        if not hasattr(self, 'pm'):
            return
        area = self.image.size()
        fit = min(area.width() / max(1, self.pm.width()), area.height() / max(1, self.pm.height()))
        s = self.zoom if self.zoom > 0 else max(0.05, fit)
        s = min(s, 6.0)
        self.image.setPixmap(self.pm.scaled(int(self.pm.width() * s), int(self.pm.height() * s),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status.setToolTip('显示缩放 %.2f×（原图 %dx%d）' % (s, self.pm.width(), self.pm.height()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.render()

    # ---------- 行为 ----------
    def on_label(self, key):
        if self._loading:
            return
        r = self.current_row()
        if key == 'not_part':
            key = 'not_hand' if r['kind'] == 'hand' else 'not_foot'
        r['label'] = key
        self.save()
        self.step(+1)

    def step(self, d):
        if not self.v:
            return
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

    def set_zoom(self, z):
        self.zoom = z
        self.render()

    def set_only_bare(self, on):
        self.only_bare = on; self.pos = 0; self.rebuild_view()

    def set_only_unlabeled(self, on):
        self.only_unlabeled = on; self.pos = 0; self.rebuild_view()

    def save(self):
        with io.open(self.queue, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})

    def keyPressEvent(self, e):
        k = e.key()
        if Qt.Key_1 <= k <= Qt.Key_5:
            self.buttons[k - Qt.Key_1].setChecked(True)
        elif k in (Qt.Key_Right, Qt.Key_Space):
            self.step(+1)
        elif k == Qt.Key_Left:
            self.step(-1)
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.set_zoom(2.0 if self.zoom == 0 else min(6.0, self.zoom * 1.5))
        elif k == Qt.Key_Minus:
            self.set_zoom(max(0.25, (self.zoom or 1.0) / 1.5))
        elif k == Qt.Key_F:
            self.set_zoom(0.0)
        elif k == Qt.Key_U:
            self.cb_un.setChecked(not self.cb_un.isChecked())
        elif k == Qt.Key_B:
            self.cb_bare.setChecked(not self.cb_bare.isChecked())
        elif k == Qt.Key_S:
            self.save(); self.status.setText('已保存 → %s' % self.queue)
        else:
            super().keyPressEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join(PROJ, 'data', 'cls_queue', 'index.csv'))
    ap.add_argument('--only-bare', action='store_true')
    ap.add_argument('--only-unlabeled', action='store_true')
    a = ap.parse_args()
    if not os.path.exists(a.queue):
        raise SystemExit('找不到队列：%s（先跑 scripts/cls_queue_sheet.py）' % a.queue)
    app = QApplication(sys.argv)
    w = Labeler(a.queue, a.only_bare, a.only_unlabeled)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
