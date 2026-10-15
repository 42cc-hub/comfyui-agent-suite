#!/usr/bin/env python
"""mosaic_censor.py —— 把指定区域打上马赛克，**只写副本，绝不覆盖原图**。

区域来源二选一（可叠加）：
  --mask MASK.png      掩版图，**非零像素 = 要打码**（可用 --mask-invert 取反）
  --box x0,y0,x1,y1    矩形，可重复

自带两条硬自检（不满足就退出、不产出文件）：
  1. mask 为空的区域 ⇒ 拒绝（绝不产出"看起来打了码其实没盖住"的图）；
  2. mask 之外的像素必须**逐位不变**（max diff 0），否则拒绝。

输出副本 +（可选）--report 写一份 JSON，记录盖了多少像素、用了哪些框、网格大小、源文件 sha256。

例：
  python mosaic_censor.py --src a.png --out a_mos.png --mask m.png --cell 32
  python mosaic_censor.py --src a.png --out a_mos.png --box 120,80,360,300 --box 140,300,340,420
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Windows 控制台常是 GBK：中文与符号会让 print / sys.exit 抛 UnicodeEncodeError
# （产物已写完却报失败，或把乱码丢给调用方）。stdout 与 stderr 一起转 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def dilation(mask, radius):
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(img.filter(ImageFilter.MaxFilter(k))) > 127


def mask_from_file(path, invert):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    mask = arr > 0
    return ~mask if invert else mask


def mask_from_boxes(shape, boxes):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        xa, xb = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
        ya, yb = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
        mask[ya:yb, xa:xb] = True
    return mask


def mask_from_polygons(shape, polygons):
    """多边形掩版（坐标会被夹到画布内）。"""
    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(max(0, min(w - 1, float(x))), max(0, min(h - 1, float(y)))) for x, y in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.asarray(img) > 127


def trapezoid_polygon(x0, y0, x1, y1, inset):
    """上边满宽、下边两侧各内收 inset 比例的梯形（胸罩形状）。"""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    dx = (xb - xa) * inset
    return [(xa, ya), (xb, ya), (xb - dx, yb), (xa + dx, yb)]


def inverted_triangle_polygon(x0, y0, x1, y1):
    """上边满宽、下方收成一点的倒三角（内裤形状）。"""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return [(xa, ya), (xb, ya), ((xa + xb) / 2.0, yb)]


def mosaic_array(arr, cell):
    h, w = arr.shape[:2]
    small = Image.fromarray(arr).resize((max(1, w // cell), max(1, h // cell)), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.NEAREST))


NAMED_FILLS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
}


def parse_fill(text):
    """--fill black | white | grey | R,G,B | #RRGGBB —— 平涂颜色（不给就是马赛克）。"""
    value = str(text).strip().lower()
    if value in NAMED_FILLS:
        return NAMED_FILLS[value]
    hexed = value.lstrip("#")
    if len(hexed) == 6 and all(c in "0123456789abcdef" for c in hexed):
        return tuple(int(hexed[i:i + 2], 16) for i in (0, 2, 4))
    parts = [int(float(v)) for v in value.split(",")]
    if len(parts) != 3 or any(v < 0 or v > 255 for v in parts):
        raise argparse.ArgumentTypeError("fill must be black/white/grey, R,G,B or #RRGGBB")
    return tuple(parts)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_box(text):
    parts = [int(float(v)) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1")
    return tuple(parts)


def parse_trapezoid(text):
    parts = [float(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError("trapezoid must be x0,y0,x1,y1[,inset]")
    inset = parts[4] if len(parts) == 5 else 0.22
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(inset))


def apply_censor(src, out, mask, cell, report_path=None, extra=None, fill=None):
    """写副本并自检。返回 report dict；任何自检失败都 raise SystemExit。"""
    base = load_rgb(src)
    if mask.shape != base.shape[:2]:
        sys.exit(f"mask shape {mask.shape} != image shape {base.shape[:2]}")
    covered = int(mask.sum())
    total = int(mask.size)
    if covered == 0:
        sys.exit("REFUSED: the censor mask is empty — nothing would be hidden")
    if fill is None:
        cover = mosaic_array(base, cell)
    else:
        cover = np.zeros_like(base)
        cover[:, :] = np.asarray(fill, dtype=np.uint8)
    result = base.copy()
    result[mask] = cover[mask]
    outside = ~mask
    if not np.array_equal(result[outside], base[outside]):
        sys.exit("REFUSED: pixels outside the mask changed — refusing to write")
    if Path(out).resolve() == Path(src).resolve():
        sys.exit("REFUSED: --out must not be the source file; write a copy")
    Image.fromarray(result).save(out)
    report = {
        "src": str(src),
        "out": str(out),
        "cell": cell,
        "mode": "fill" if fill is not None else "mosaic",
        "fill": list(fill) if fill is not None else None,
        "covered_px": covered,
        "coverage_pct": round(covered / total * 100, 3),
        "src_sha256": sha256_file(src),
        "out_sha256": sha256_file(out),
        "outside_mask_max_diff": 0,
    }
    if extra:
        report.update(extra)
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--mask-invert", action="store_true")
    ap.add_argument("--box", action="append", type=parse_box, default=[])
    ap.add_argument("--trapezoid", action="append", type=parse_trapezoid, default=[],
                    help="x0,y0,x1,y1[,inset] —— 上宽下窄的梯形（胸罩形状）")
    ap.add_argument("--triangle", action="append", type=parse_box, default=[],
                    help="x0,y0,x1,y1 —— 上宽下尖的倒三角（内裤形状）")
    ap.add_argument("--cell", type=int, default=32, help="马赛克网格边长（16–64 是本项目实测区间）")
    ap.add_argument("--fill", type=parse_fill, default=None,
                    help="平涂模式：black / white / grey / R,G,B（不给则用马赛克）")
    ap.add_argument("--dilate", type=int, default=0, help="掩版外扩像素")
    ap.add_argument("--report")
    args = ap.parse_args()

    if not args.mask and not args.box and not args.trapezoid and not args.triangle:
        sys.exit("need --mask / --box / --trapezoid / --triangle")
    with Image.open(args.src) as im:
        shape = (im.height, im.width)
    mask = mask_from_boxes(shape, args.box)
    polygons = [trapezoid_polygon(*item) for item in args.trapezoid]
    polygons += [inverted_triangle_polygon(*item) for item in args.triangle]
    if polygons:
        mask = mask | mask_from_polygons(shape, polygons)
    if args.mask:
        mask = mask | mask_from_file(args.mask, args.mask_invert)
    mask = dilation(mask, args.dilate)
    report = apply_censor(args.src, args.out, mask, args.cell, args.report,
                          extra={"boxes": args.box, "trapezoids": args.trapezoid,
                                 "triangles": args.triangle, "mask": args.mask, "dilate": args.dilate},
                          fill=args.fill)
    print(json.dumps(report, ensure_ascii=False))
    print("\n下一步（必做）：对副本重跑闸门\n  python libs/wd14-gate/scripts/wd14_gate.py " + args.out)


if __name__ == "__main__":
    main()
