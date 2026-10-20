# -*- coding: utf-8 -*-
"""验 `label_face.py` 加了 `person` / `head` 两个中文名之后：显示、画框落盘、hand/foot 未变。

离屏跑真实 Labeler（QT_QPA_PLATFORM=offscreen），队列 3 行 = person / hand / foot，
图用 data/hand_gt/images 里真实存在的交付图。不碰生产队列，全部在临时目录里。
"""
import csv
import io
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_face as LF  # noqa: E402

IMGDIR = os.path.join(os.path.dirname(HERE), 'data', 'hand_gt', 'images')
OK = []


def chk(name, cond):
    OK.append((name, bool(cond)))
    print('  [%s] %s' % ('OK  ' if cond else 'FAIL', name))


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    img = sorted(f for f in os.listdir(IMGDIR) if f.lower().endswith('.png'))[0]
    tmp = os.path.join(tempfile.mkdtemp(), 'index.csv')
    rows = [
        {'idx': 1, 'crop': img, 'source': os.path.splitext(img)[0], 'head': 0, 'target': 'person',
         'prop': '100,80,900,1900;1200,90,1980,1880', 'prop_src': 'det_v2', 'prop_conf': '0.91',
         'wd14': '', 'verdict': '', 'boxes': '', 'edited': '', 'note': 'multi2'},
        {'idx': 2, 'crop': img, 'source': os.path.splitext(img)[0], 'head': 0, 'target': 'hand',
         'prop': '640,700,780,860', 'prop_src': 'model_union', 'prop_conf': '0.72',
         'wd14': '', 'verdict': '', 'boxes': '', 'edited': '', 'note': 'multi2'},
        {'idx': 3, 'crop': img, 'source': os.path.splitext(img)[0], 'head': 0, 'target': 'foot',
         'prop': '900,1700,1100,1940', 'prop_src': 'det_v2', 'prop_conf': '0.55',
         'wd14': '', 'verdict': '', 'boxes': '', 'edited': '', 'note': 'multi2'},
    ]
    with io.open(tmp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=LF.FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 1) 字典本身 ----
    chk('TARGET_CN 有 person', 'person' in LF.TARGET_CN and '人 person' in LF.TARGET_CN['person'])
    chk('TARGET_CN 有 head', 'head' in LF.TARGET_CN and '头 head' in LF.TARGET_CN['head'])
    chk('原有 hand/foot 文案未变',
        LF.TARGET_CN['hand'] == '手 hand（要含手指）' and LF.TARGET_CN['foot'] == '脚 foot（要含脚趾/远端）')

    w = LF.Labeler(tmp, pre=IMGDIR)
    # ---- 2) 侧栏中文名 ----
    w.jump(1)
    t_person = w.head.text()
    chk('person 行显示中文名', '人 person' in t_person and 'person（整个身体' in t_person)
    print('     person 行标题 =', t_person)
    w.jump(2)
    t_hand = w.head.text()
    chk('hand 行仍是原中文名', '手 hand（要含手指）' in t_hand)
    w.jump(3)
    t_foot = w.head.text()
    chk('foot 行仍是原中文名', '脚 foot（要含脚趾/远端）' in t_foot)

    # ---- 3) person 行画框 → 翻页落盘 ----
    w.jump(1)
    chk('person 行预标注被注入成可采纳框', len(w.canvas.boxes) == 2 and w.canvas.boxes[0]['kind'] == 'prop')
    w.canvas.boxes = [{'r': [90.0, 70.0, 910.0, 1890.0], 'kind': 'user'},
                      {'r': [1210.0, 95.0, 1975.0, 1875.0], 'kind': 'user'}]
    w.canvas.dirty = True
    w.step(+1)
    chk('person 行画框后翻页 ⇒ verdict=draw + boxes 落盘',
        w.rows[0]['verdict'] == 'draw' and w.rows[0]['boxes'] == '90,70,910,1890;1210,95,1975,1875')
    # ---- 4) A 键对 person 行同样可用 ----
    w.rows[0].update({'verdict': '', 'boxes': ''})
    w.jump(1)
    w.adopt_head()
    chk('A 键采纳 person 预填', w.rows[0]['verdict'] == 'pass'
        and w.rows[0]['boxes'] == '100,80,900,1900;1200,90,1980,1880')
    # ---- 5) save() 落盘重读 ----
    w.rows[0].update({'verdict': 'draw', 'boxes': '90,70,910,1890;1210,95,1975,1875'})
    w.save()
    with io.open(tmp, encoding='utf-8') as fh:
        back = list(csv.DictReader(fh))
    chk('save() 后重读：列数与 FIELDS 一致', list(back[0].keys()) == LF.FIELDS)
    chk('save() 后重读：person 行 boxes 完整',
        back[0]['boxes'] == '90,70,910,1890;1210,95,1975,1875' and back[0]['target'] == 'person')
    # 注意：上面按过一次 A，而 A 的口径就是「整张 head 的全部预填一次采纳」，
    # 所以 hand/foot 行**应当**被 A 一起采纳（verdict=pass、boxes=自己的预填）——
    # 这里要验的是「A 没有把 person 的框写到 hand/foot 上」，不是「hand/foot 没被 A 动过」。
    chk('save() 后重读：hand/foot 行的预填/目标未变、且没被写入 person 的框',
        back[1]['target'] == 'hand' and back[1]['prop'] == '640,700,780,860'
        and back[2]['target'] == 'foot' and back[2]['prop'] == '900,1700,1100,1940'
        and back[1]['boxes'] in ('', '640,700,780,860') and back[2]['boxes'] in ('', '900,1700,1100,1940')
        and '90,70,910,1890' not in back[1]['boxes'] and '90,70,910,1890' not in back[2]['boxes'])
    print('     hand 行 =', {k: back[1][k] for k in ('target', 'verdict', 'boxes')})
    print('     foot 行 =', {k: back[2][k] for k in ('target', 'verdict', 'boxes')})

    bad = [n for n, c in OK if not c]
    print('\n结果：%d/%d 通过%s' % (len(OK) - len(bad), len(OK), ('　失败：%s' % bad) if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
