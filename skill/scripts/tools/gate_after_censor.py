#!/usr/bin/env python
"""gate_after_censor.py —— 打码审验：判"关键点是否真的被盖住了"（而不是判整图是否变干净）。

放行规则（实测得出，见 references/output_safety_gate.md 规则五·b）：
  1. 原图过 WD14 → 拿到**关键点词汇**（nipples / pussy / penis / anus / vulva …）；
  2. 打码后再过一次 WD14 → 若**关键点词汇全部消失**，即使 `nude` / `completely nude` /
     `censored` / `mosaic censoring` 这类**状态词**仍在 ⇒ **放行**。
  理由：WD14 感知不到"打码"这件事本身，它只会从上下文继续推断 `nude`；真正决定"东西露没露"的是
  **关键点词汇**。要求打码后变成 REVIEW/OK 会让所有打码都过不了。

判 FAIL 的两种情形（都不许看）：
  · 原图的关键点词汇还有残留（说明没盖住）；
  · 打码后**新出现**了原图没有的关键点词汇（说明遮出了新问题，例如平涂把能看的图变糟）。

退出码：0 = PASS；3 = 关键点残留；4 = 出现新关键点；5 = 用法/读取错误。

例：
  python scripts/tools/gate_after_censor.py --src IMG --censored COPY.png [--report r.json]

wd14_gate.py 的定位：本机 `<skill>/scripts/`，公版仓库 `libs/wd14-gate/scripts/` —— 自动找，
也可 `--wd14-dir` 或环境变量 `WD14_GATE_DIR` 指定。
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Windows 控制台常是 GBK：中文与符号会让 print / sys.exit 抛 UnicodeEncodeError
# （产物已写完却报失败，或把乱码丢给调用方）。stdout 与 stderr 一起转 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve()

# 关键点词汇：**可被遮盖**的解剖/行为词 —— 它们必须在打码后消失。
# 这个分层是本脚本的概念（wd14_gate 只有一张 BLOCK 表），所以写在这里。
KEYPOINT_TAGS = {
    "nipples", "puffy nipples", "areolae", "large areolae",
    "pussy", "penis", "anus", "clitoris", "testicles",
    "bare crotch", "cameltoe", "partially visible vulva", "vulva", "labia", "visible clitoris",
    "sex", "cum",
}


def find_wd14_gate(explicit=None):
    """定位 wd14_gate.py：本机是 <skill>/scripts/，公版仓库是 libs/wd14-gate/scripts/。

    不假设布局（与 gate_review.py 找 canvas_mask.py 同一套路），找不到就把试过的路径列出来。
    """
    candidates = []
    if explicit:
        # 显式指定就是明确意图：找不到要立刻报错，不能悄悄回退到自动定位。
        given = Path(explicit)
        if given.is_file() and given.name == "wd14_gate.py":
            return given.parent
        if (given / "wd14_gate.py").is_file():
            return given
        raise SystemExit(f"--wd14-dir 下没有 wd14_gate.py: {given}")
    env = os.environ.get("WD14_GATE_DIR")
    if env:
        candidates.append(Path(env))
    tools_dir = HERE.parent            # <skill>/scripts/tools
    scripts_dir = tools_dir.parent     # <skill>/scripts
    skill_dir = scripts_dir.parent     # <skill>
    repo_dir = skill_dir.parent        # 公版：仓库根
    candidates += [
        scripts_dir,                                   # 本机：<skill>/libs/wd14-gate/scripts/wd14_gate.py
        skill_dir / "scripts",
        repo_dir / "libs" / "wd14-gate" / "scripts",    # 公版：<repo>/libs/wd14-gate/scripts/
        skill_dir / "libs" / "wd14-gate" / "scripts",
        repo_dir / "skill" / "scripts",
    ]
    tried = []
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "wd14_gate.py":
            return candidate.parent
        tried.append(candidate)
        if (candidate / "wd14_gate.py").is_file():
            return candidate
    raise SystemExit("找不到 wd14_gate.py；用 --wd14-dir 指定，或设 WD14_GATE_DIR。\n已试过：\n  "
                     + "\n  ".join(str(c) for c in tried))


def load_gate(explicit=None):
    """返回 (Tagger, block_tags, state_tags)。state = 原 BLOCK ∪ POSE 减去关键点。"""
    sys.path.insert(0, str(find_wd14_gate(explicit)))
    from wd14_gate import BLOCK_TAGS, POSE_TAGS, Tagger  # noqa: E402
    block = set(BLOCK_TAGS)
    state = (block | set(POSE_TAGS)) - KEYPOINT_TAGS
    return Tagger, block, state


def classify(tags, block, state):
    got = {str(t).lower().strip() for t in tags or []}
    return {
        "keypoints": sorted(got & KEYPOINT_TAGS),
        "state": sorted(got & state),
        "obsolete": sorted((got & block) - KEYPOINT_TAGS - state),
    }


def judge(orig_keypoints, cens_keypoints):
    survived = sorted(set(orig_keypoints) & set(cens_keypoints))
    appeared = sorted(set(cens_keypoints) - set(orig_keypoints))
    if appeared:
        return False, "FAIL", f"打码后新出现关键点词汇：{', '.join(appeared)}", survived, appeared
    if survived:
        return False, "FAIL", f"关键点词汇仍有残留：{', '.join(survived)}", survived, appeared
    if not orig_keypoints:
        return True, "PASS", "原图本无关键点词汇（无需遮盖）", survived, appeared
    return True, "PASS", "原图的关键点词汇已全部消失（状态词残留不影响）", survived, appeared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原图")
    ap.add_argument("--censored", required=True, help="打码后的副本")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--wd14-dir", default=None, help="wd14_gate.py 所在目录（默认自动定位）")
    ap.add_argument("--report")
    args = ap.parse_args()

    for path in (args.src, args.censored):
        if not os.path.exists(path):
            sys.exit(f"not found: {path}")
    if os.path.abspath(args.src) == os.path.abspath(args.censored):
        sys.exit("--censored must be a different file from --src")

    Tagger, block, state = load_gate(args.wd14_dir)
    tagger = Tagger()
    o_rating, o_tags = tagger.run(args.src, args.threshold)
    c_rating, c_tags = tagger.run(args.censored, args.threshold)
    o = classify(o_tags, block, state)
    c = classify(c_tags, block, state)
    ok, label, why, survived, appeared = judge(o["keypoints"], c["keypoints"])

    print(f"原图      : rating={o_rating}  keypoints={o['keypoints']}  state={o['state']}")
    print(f"打码副本  : rating={c_rating}  keypoints={c['keypoints']}  state={c['state']}")
    print(f"\nVERDICT={label}  —— {why}")
    if ok:
        print("⇒ 允许 read_image 看这张**打码副本**；原图仍按 BLOCK 处置（除路径外不给别的）。")
        print("  残余风险：骨骼定位不是 100% 可靠；若判定与实际观感不符，以人眼为准并回退到人工掩版。")
    else:
        print("⇒ **不许看**：只报路径，或交人用 human_canvas 涂掩版后重跑本脚本。")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "src": args.src, "censored": args.censored, "threshold": args.threshold,
            "verdict": label, "reason": why,
            "orig": {"rating": o_rating, **o}, "censored": {"rating": c_rating, **c},
            "survived_keypoints": survived, "appeared_keypoints": appeared,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("report:", args.report)
    sys.exit(0 if ok else (3 if survived else 4))


if __name__ == "__main__":
    main()
