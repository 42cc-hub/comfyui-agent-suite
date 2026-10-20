#!/usr/bin/env python
"""m3_review_sheets.py —— 把生成的裁图拼成"抽检表"（每张 3×2，带编号），供人/视觉逐张判缺陷。

输出：
  <root>/review/sheet_<cls>_<k>.png   每张 3 列 × 2 行，格子下方标编号
  <root>/review/index.csv             列：sheet, tile, crop_path, cls, label_by_recipe
用法：
  python scripts/m3_review_sheets.py [--root data/m3_gen] [--per-sheet 6] [--only five,malformed]
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJ / "data" / "m3_gen"

# 配方 -> 标签（bad 类 = 生成时要求了缺陷；ok 类 = 要求正常）
BAD_CLASSES = {"six", "five", "webbed", "malformed", "six_toes", "four_toes"}
OK_CLASSES = {"ok5", "ok5toe"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    root = Path(a.root)
    crops_root = root / "crops"
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)
    wanted = set(x for x in a.only.split(",") if x)

    index_rows = []
    sheet_no = 0
    cols = 3
    for cls_dir in sorted(crops_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        if wanted and cls not in wanted:
            continue
        files = sorted(cls_dir.glob("*.png"))
        for start in range(0, len(files), a.per_sheet):
            chunk = files[start:start + a.per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            label_h = 20
            sheet = Image.new("RGB", (cols * a.tile, rows * (a.tile + label_h)), (20, 20, 24))
            draw = ImageDraw.Draw(sheet)
            for i, path in enumerate(chunk):
                r_, c_ = divmod(i, cols)
                top = r_ * (a.tile + label_h)
                with Image.open(path) as im:
                    sheet.paste(im.convert("RGB").resize((a.tile, a.tile), Image.LANCZOS),
                                (c_ * a.tile, top + label_h))
                draw.text((c_ * a.tile + 4, top + 3), "%s#%d" % (cls, start + i), fill=(240, 240, 240))
                index_rows.append({"sheet": "sheet_%s_%02d.png" % (cls, start // a.per_sheet),
                                   "tile": start + i, "crop": str(path), "cls": cls,
                                   "label_by_recipe": "bad" if cls in BAD_CLASSES else "ok"})
            out = review / ("sheet_%s_%02d.png" % (cls, start // a.per_sheet))
            sheet.save(out)
            sheet_no += 1

    with (review / "index.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "tile", "crop", "cls", "label_by_recipe"])
        w.writeheader()
        w.writerows(index_rows)
    print("sheets:", sheet_no, "| tiles:", len(index_rows), "| dir:", review)


if __name__ == "__main__":
    main()
