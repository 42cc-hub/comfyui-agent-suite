"""验收新的四类面部检测器（eye/mouth/gag/blindfold）在**交付图**上的表现，并与现役模型对照。

为什么要单独一个脚本：`locate.py` 的 M2b 路径是按"face/eye"+"gag/blindfold 两个权重并集"写的，
新模型是**一个四类权重**，接口不同；而验收要看的三件事必须同时摆在眼前：
  ① **gag 不能退**（现役交付成绩：8 张里 02/03/04/07 检出 = 4/6）；
  ② **blindfold 不能误报**（8 张真值都没有眼罩 ⇒ 目标是 0 个误报）；
  ③ **新增的 eye / mouth 准不准**（画出来给人看，并与"脸框+眼线推的嘴"对照）。

用法：
  python scripts/eval_face4.py --weights runs/m2b_face4_v1/weights/best.pt
  python scripts/eval_face4.py --weights <新权重> --sheet <对照图.png>
输出：屏幕逐图表 + 可选对照图（每张 head 一格：新模型框 vs 现役 gag 框）

`run_eval()` 是**给别人复用的同一份实现**（`ab_face4.py` 并多组权重的表时直接调它，不再抄一遍）。
"""
import argparse
import glob
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SRC = r'<你的 ComfyUI 项目>\task\escape_containment_20261010\art\final'
HEAD_W = os.path.join(PROJ, 'runs', 'm2a_det_996img_960', 'weights', 'best.pt')
OLD_GAG = [os.path.join(PROJ, 'runs', n, 'weights', 'best.pt') for n in ('m2b_gag_v2', 'm2b_gag_v4')]
# 用户 2026-10-11 确认的真值：01/08 无口塞无眼罩；02~07 有口塞；8 张全无眼罩
GAG_TRUTH = {'02', '03', '04', '05', '06', '07'}
COLORS = {'eye': (255, 170, 0), 'mouth': (0, 230, 230), 'gag': (255, 60, 60), 'blindfold': (170, 0, 255)}


def run_eval(weights, src=SRC, head_weights=None, conf=0.4, sheet=None, per_row=4, quiet=False):
    """交付图验收。返回 {'eye': 眼睛总数, 'mouth_imgs': 出嘴的图数, 'gag_hit': [...], 'bf_fp': [...], 'rows': [...]}。"""
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    head_weights = head_weights or HEAD_W
    new = YOLO(weights)
    old = [YOLO(w) for w in OLD_GAG]
    head = YOLO(head_weights)

    hit_gag, fp_bl, rows, tiles = set(), [], [], []
    for f in sorted(glob.glob(os.path.join(src, '*.png'))):
        name = os.path.basename(f)
        r = head.predict(f, conf=0.3, imgsz=1280, verbose=False)[0]
        img = Image.open(f).convert('RGB')
        for hb in [b for b in (r.boxes or []) if r.names[int(b.cls[0])] == 'head']:
            x0, y0, x1, y1 = [float(v) for v in hb.xyxy[0]]
            pad = int(0.15 * max(x1 - x0, y1 - y0))
            c = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
                 min(img.width, int(x1 + pad)), min(img.height, int(y1 + pad)))
            cp = os.path.join(PROJ, '.runtime', '_f4_%s.png' % name[:12])
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            crop = img.crop(c).resize((256, 256), Image.LANCZOS)
            crop.save(cp)
            rn = new.predict(cp, conf=conf, verbose=False)[0]
            got = {}
            for b in (rn.boxes or []):
                got.setdefault(rn.names[int(b.cls[0])], []).append(
                    [round(float(v), 1) for v in b.xyxy[0].tolist()])
            # 现役 gag/blindfold：两个权重并集
            oldg, oldb = 0, 0
            for m in old:
                ro = m.predict(cp, conf=0.5, verbose=False)[0]
                for b in (ro.boxes or []):
                    if m.names[int(b.cls[0])] == 'gag':
                        oldg += 1
                    else:
                        oldb += 1
            key = name[:2]
            if got.get('gag'):
                hit_gag.add(key)
            if got.get('blindfold'):
                fp_bl.append((name, len(got['blindfold'])))
            rows.append((key, {k: len(v) for k, v in got.items()}, oldg, oldb))
            if sheet:
                d = ImageDraw.Draw(crop)
                for k, boxes in got.items():
                    for bx in boxes:
                        d.rectangle(bx, outline=COLORS.get(k, (255, 255, 255)), width=2)
                        d.text((bx[0] + 2, bx[1] + 2), k, fill=COLORS.get(k, (255, 255, 255)))
                d.text((4, 4), '%s  gag/bf(old)=%d/%d' % (key, oldg, oldb), fill=(255, 255, 0))
                tiles.append(crop)

    if not quiet:
        print('\n逐图（新模型 conf=%.2f；gag 真值 = 02~07 有）' % conf)
        print('%-5s %-46s %-14s %s' % ('图', '新模型检出', '现役 gag/bf', '判定'))
        for key, got, oldg, oldb in rows:
            verdict = []
            if key in GAG_TRUTH:
                verdict.append('gag✓' if got.get('gag') else 'gag✗(漏)')
            else:
                verdict.append('gag 误报' if got.get('gag') else 'gag 正确未报')
            if got.get('blindfold'):
                verdict.append('blindfold 误报!')
            print('%-5s %-46s %-14s %s' % (key, str(got), '%d/%d' % (oldg, oldb), ' '.join(verdict)))
        print('\n聚合：gag 检出 %d/6（现役 4/6）；blindfold 误报 %d 张（目标 0）' % (
            len(hit_gag & GAG_TRUTH), len(fp_bl)))
        if sheet and tiles:
            per = per_row
            rowsn = (len(tiles) + per - 1) // per
            sheet_img = Image.new('RGB', (per * 256, rowsn * 256), (12, 12, 12))
            for i, t in enumerate(tiles):
                sheet_img.paste(t, ((i % per) * 256, (i // per) * 256))
            sheet_img.save(sheet)
            print('对照图 → %s（橙=eye 青=mouth 红=gag 紫=blindfold；左上角是现役 gag/blindfold 计数）' % sheet)
    return {'eye': sum(d.get('eye', 0) for _, d, _, _ in rows),
            'mouth_imgs': sum(1 for _, d, _, _ in rows if d.get('mouth')),
            'gag_hit': sorted(hit_gag & GAG_TRUTH), 'bf_fp': fp_bl, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--head-weights', default=HEAD_W)
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--sheet', default=None)
    ap.add_argument('--per-row', type=int, default=4)
    a = ap.parse_args()
    run_eval(a.weights, a.src, a.head_weights, a.conf, a.sheet, a.per_row)


if __name__ == '__main__':
    main()
