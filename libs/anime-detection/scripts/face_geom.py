"""由**检测框**推嘴的几何（不用 M1 关键点：实测 pose 的脸部关键点会散落一脸，用户 2026-10-13 否决）。

规则：脸框给尺度，**两眼框给朝向**，嘴 = 眼线中点沿"脸正下方"偏移 0.38×脸高，尺寸 0.44×脸宽 × 0.22×脸高。
为什么不用"鼻点往下"：那是图像坐标的竖直偏移，头一旋转就错（交付图 04/05/06 是侧躺，θ≈±80°）。
没有两只眼框时退回"脸框下 42%"。

`check_parts.py`（skill 侧）与本文件的公式必须一致；skill 侧会优先 import 这里。
"""
import math


def mouth_from_face_eyes(face_box, eye_boxes, near=0.2):
    """→ (mouth_box, src_text)。

    face_box: [x0,y0,x1,y1]；eye_boxes: [[x0,y0,x1,y1], ...]（允许为空/1 个）
    """
    x0, y0, x1, y1 = [float(v) for v in face_box]
    w, h = x1 - x0, y1 - y0
    ex = []
    for e in eye_boxes or []:
        cx, cy = (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0
        if x0 - near * w <= cx <= x1 + near * w and y0 - near * h <= cy <= y1 + near * h:
            ex.append((cx, cy, e))
    if len(ex) >= 2:
        ex.sort(key=lambda t: t[0])
        (c0x, c0y, _), (c1x, c1y, _) = ex[0], ex[1]
        th = math.atan2(c1y - c0y, c1x - c0x)
        vx, vy = -math.sin(th), math.cos(th)
        if vy < 0:                      # "下"要指向脸框底边那一侧
            vx, vy = -vx, -vy
        mx = (c0x + c1x) / 2 + vx * 0.38 * h
        my = (c0y + c1y) / 2 + vy * 0.38 * h
        mw, mh = 0.44 * w, 0.22 * h
        return [mx - mw / 2, my - mh / 2, mx + mw / 2, my + mh / 2], 'eye_line+face(θ=%.0f°)' % math.degrees(th)
    return [x0 + 0.12 * w, y0 + 0.58 * h, x1 - 0.12 * w, y1], 'face_box_lower42%(只有一只眼框，退回几何)'
