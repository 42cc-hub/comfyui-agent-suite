"""gate_review.py —— 闸门判定 + **人工审核通道**（BLOCK 时走 human_canvas）。

背景（skill: references/output_safety_gate.md）：wd14_gate.py 的 BLOCK 是**防护 agent 自己的
上传预算**，不是对这张画的判词。当**文件所有者本人**看过并愿意授权时，人可以给出比分类器
更高的判据（多模态类任务里人的反馈是最高判据）。

本脚本**不改判定**，只把"BLOCK 之后怎么办"变成一条可执行流程：

    python gate_review.py IMG... [--tags] [--sheet SHEET.png]
      OK/REVIEW 存在 -> 正常可看（照旧）
      存在 BLOCK     -> 打印 human_canvas 交接块：由人涂出要打码/要看的区域，
                        脚本随后按该 mask 出打码图，再重新过闸门并**记录人的授权**。

为什么由人涂而不是自动打码：实测（2026-09-29 矩阵 + 2026-09-30 反向案例）
  - 平涂填充可能**把可看的图变成 BLOCK**，马赛克/人涂过的语义则通常改善判定；
  - 判定依赖**上下文**，不只看被盖住的像素 ⇒ "找哪一小块露骨"是判断，不是计数，归人。
人的涂抹产出的是**可信的授权记录**；之后 agent 看这张（打码后的）图是正当的。

mask 落地的细节（`alpha = 255 - mask` 之类）交给下游 canvas_mask.py：
    python scripts/tools/canvas_mask.py --rope L1.png --keep L2.png \
           --src IMG --out mask.png --overlay overlay.png --dilate 4 --feather 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wd14_gate as g  # noqa: E402


def find_canvas_mask():
    """canvas_mask.py（L1-L2 → mask 的合成器）随本工具一起发布；按就近顺序找。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, "canvas_mask.py"),
        os.path.join(here, "tools", "canvas_mask.py"),
        os.path.join(here, "..", "..", "..", "skill", "scripts", "tools", "canvas_mask.py"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def build_sheet(paths, out, cell=(380, 556), cols=4):
    from PIL import Image, ImageDraw
    cw, ch = cell
    cols = max(1, min(cols, len(paths)))
    rows = (len(paths) + cols - 1) // cols
    lab = 26
    sheet = Image.new("RGB", (cols * cw, rows * (ch + lab)), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((cw, ch), Image.LANCZOS)
        r, c = divmod(i, cols)
        x, y = c * cw, r * (ch + lab)
        sheet.paste(im, (x, y + lab))
        d.text((x + 6, y + 6), os.path.basename(p)[:52], fill=(230, 230, 235))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--tags", action="store_true", help="print all tags")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--sheet", help="BLOCK 多张时：拼带文件名标注的联系表，人一次涂完")
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()

    t = g.Tagger()
    rows, blocked = [], []
    for p in a.images:
        if not os.path.exists(p):
            print("MISSING %s" % p)
            continue
        rating, tags = t.run(p, a.threshold)
        v = g.verdict(rating, tags)
        hit_b = sorted(g.BLOCK_TAGS & set(tags))
        hit_w = sorted(g.WATCH_TAGS & set(tags))
        rows.append((p, v, rating, tags, hit_b, hit_w))
        print("%-8s rating=%-12s %s" % (v, rating, os.path.basename(p)))
        if hit_b:
            print("         BLOCK-tags: %s" % ", ".join(hit_b))
        if hit_w:
            print("         watch-tags: %s" % ", ".join(hit_w))
        if a.tags:
            print("         all: %s" % ", ".join(tags))
        if v == "BLOCK":
            blocked.append(p)

    order = {"OK": 0, "REVIEW": 1, "BLOCK": 2}
    worst = "OK"
    for _, v, *_ in rows:
        if order[v] > order[worst]:
            worst = v
    print("OVERALL=%s" % worst)

    if not blocked:
        print("\n没有 BLOCK 项 —— 无需人工审核通道。")
        return 0 if worst == "OK" else 1

    canvas = find_canvas_mask()
    print("\n" + "=" * 78)
    print("BLOCK 项 %d 个 —— 走人工审核通道（判定不因此改变；改变的是**谁有资格看**）" % len(blocked))
    for p in blocked:
        print("  . " + p)
    print("""
下一步（agent 执行；不要自己决定打哪里）：

1) 把图交给文件所有者，用 human_canvas 涂两层：
     L1「要打码 / 不给 agent 看」= 由他决定盖哪里
     L2「要保留」                = 保护层（可选；mask = L1 - L2）
   多张时先拼一张带文件名标注的联系表，人一次涂完。
2) 读回各层，用 canvas_mask.py 合成遮罩：
     %s
   --dilate/--feather 按**画布像素**给；在缩略图上涂就先换算回原尺寸。
   打码在**副本**上做，绝不覆盖原文件；产物写明 mask 来源。
3) 对打码后的产物**重新过闸门**并记录：改善与否都写进授权记录（可能更差，不影响授权）。
   默认用**马赛克**（16-64px 网格）而不是平涂 —— 实测平涂灰块会把可看的图判成 BLOCK。
4) 授权记录（谁、何时、涂了哪里、打码后判定）随产物留档；此后 agent 看这张打码图属于
   **经所有者授权**，不必重复闸门。

授权来源必须是**文件所有者本人**；"我觉得无害""看起来正常"都不算授权。
""" % (canvas or "<工具目录>/canvas_mask.py  <- 未找到；本仓库在 skill/scripts/tools/ 下自带"))

    if a.sheet and len(blocked) > 1:
        build_sheet(blocked, a.sheet, cols=a.cols)
        print("已生成联系表：%s（%d 张，格子 380x556）" % (a.sheet, len(blocked)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
