"""Build a labelled contact sheet from a list of images (uniform cells, any input sizes).

The point of the matting workflow's verification step: judging 19 cutouts one file at a time is
how a systematically wrong island rule survives. One sheet, one look.

usage:
  python contact_sheet.py OUT.png WxH img1 [img2 ...]
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("cell", help="cell size as WxH, e.g. 260x380")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()

    cw, ch = (int(v) for v in a.cell.lower().split("x"))
    cols = max(1, min(a.cols, len(a.images)))
    rows = (len(a.images) + cols - 1) // cols
    label_h = 26
    sheet = Image.new("RGB", (cols * cw, rows * (ch + label_h)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(a.images):
        im = Image.open(p).convert("RGB").resize((cw, ch), Image.LANCZOS)
        r, c = divmod(i, cols)
        x, y = c * cw, r * (ch + label_h)
        sheet.paste(im, (x, y + label_h))
        draw.text((x + 6, y + 6), p.replace("\\", "/").split("/")[-1][:52], fill=(230, 230, 235))
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
