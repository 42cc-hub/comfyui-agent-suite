#!/usr/bin/env python
"""m3_crops_from_gen.py —— 把生成图里的手/脚按检测框裁成 M3 输入（224px，外扩 10%），并拼一张抽检面板。

输入：`data/m3_gen/raw/*.png`（`gen_m3_defects.py` 产出，文件名 `<engine>_<cls>_s<seed>.png`）
      `data/m3_gen/loc/<stem>.json`（`locate.py --src ... --out data/m3_gen/loc` 产出，含 boxes[].cls/box）
输出：`data/m3_gen/crops/<cls>/<stem>_hand0.png` …（224×224）
      `data/m3_gen/contact_sheet.png`（带类名/框类型标注，供人抽检）
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJ / "data" / "m3_gen"


NAME = re.compile(r"^(?P<engine>[a-z]+)_(?P<body>.+)_s(?P<seed>\d+)$")


def parse_name(stem):
    """严格按 `gen_m3_defects.py` 的命名解析：`<engine>_<cls>_<style>_s<seed>`；不合规返回 None。

    不做模糊猜测 —— 之前按 `s` 前缀猜 seed，把 `anima_hand_six` 解析成 seed='ix' 直接崩了。
    """
    m = NAME.match(stem)
    if not m:
        return None
    engine, body, seed = m.group("engine"), m.group("body"), int(m.group("seed"))
    # cls 自己可能带下划线（`four_toes`），所以从右往左切：剥 seed 之后，最后一段是 style
    head, _, style = body.rpartition("_")
    if not head or not style.isalpha():
        return None
    return engine, head, style, seed


def center_crop(img, frac):
    """手占满画面 / 手出框时，检测框不可靠 —— 直接取中心区域。"""
    w, h = img.size
    side = int(min(w, h) * frac)
    return img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--classes", default="hand,foot")
    ap.add_argument("--center-frac", type=float, default=0.72, help="中心裁取边长占比（handsonly 风格用）")
    ap.add_argument("--sheet", default="contact_sheet.png")
    args = ap.parse_args()

    root = Path(args.root)
    raw_dir, loc_dir = root / "raw", root / "loc"
    crops_dir = root / "crops"
    wanted = set(args.classes.split(","))

    rows, total, skipped = [], 0, []
    for src_path in sorted(raw_dir.glob("*.png")):
        stem = src_path.stem
        parsed = parse_name(stem)
        if parsed is None:
            skipped.append(src_path.name)
            continue
        engine, cls, style, seed = parsed
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        tiles = []

        if style in ("handsonly", "onehand", "onefoot"):
            crop = center_crop(img, args.center_frac).resize((args.size, args.size), Image.LANCZOS)
            out_dir = crops_dir / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / ("%s_center.png" % stem)
            crop.save(out_path)
            tiles.append(("center", crop))
            total += 1
            rows.append((cls, engine, style, seed, tiles, src_path.name))
            continue

        loc_path = loc_dir / (stem + ".json")
        if not loc_path.exists():
            rows.append((cls, engine, style, seed, [], src_path.name))
            continue
        rec = json.loads(loc_path.read_text(encoding="utf-8"))
        boxes = [b for b in rec.get("boxes", []) if b.get("cls") in wanted]
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = b["box"]
            bw, bh = x1 - x0, y1 - y0
            x0 -= bw * args.pad; x1 += bw * args.pad
            y0 -= bh * args.pad; y1 += bh * args.pad
            box = (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            crop = img.crop(box).resize((args.size, args.size), Image.LANCZOS)
            out_dir = crops_dir / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / ("%s_%s%d.png" % (stem, b["cls"], i)))
            tiles.append((b["cls"], crop))
            total += 1
        rows.append((cls, engine, style, seed, tiles, src_path.name))

    # 抽检面板：每行一个源图，左边源图缩略，右边裁图（最多 4 张）
    cell, label_h = 132, 18
    max_tiles = max((len(r[4]) for r in rows), default=0)
    cols = 1 + min(max_tiles, 4)
    sheet = Image.new("RGB", (cols * cell, max(1, len(rows)) * (cell + label_h)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for row_i, (cls, engine, style, seed, tiles, name) in enumerate(rows):
        top = row_i * (cell + label_h)
        draw.text((4, top + 3), "%s | %s/%s | s%d | tiles=%d" % (cls, engine, style, seed, len(tiles)),
                  fill=(230, 230, 230))
        thumb = Image.open(raw_dir / name).convert("RGB").resize((cell, cell), Image.LANCZOS)
        sheet.paste(thumb, (0, top + label_h))
        for t_i, (box_cls, crop) in enumerate(tiles[:4]):
            sheet.paste(crop.resize((cell, cell), Image.LANCZOS), ((1 + t_i) * cell, top + label_h))
            draw.text(((1 + t_i) * cell + 3, top + label_h + 3), box_cls, fill=(120, 230, 120))
    sheet_path = root / args.sheet
    sheet.save(sheet_path)

    print("crops written:", total)
    for cls, engine, style, seed, tiles, _ in rows:
        print("  %-10s %-6s %-10s s%-9d tiles=%d" % (cls, engine, style, seed, len(tiles)))
    print("contact sheet:", sheet_path)
    if skipped:
        print("skipped (命名不合规):", skipped)
    empty = [r for r in rows if not r[3]]
    if empty:
        print("WARNING 没裁到任何手/脚框的源图:", [r[4] for r in empty])


if __name__ == "__main__":
    main()
