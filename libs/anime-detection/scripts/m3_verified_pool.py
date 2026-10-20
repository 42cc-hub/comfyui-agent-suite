#!/usr/bin/env python
"""m3_verified_pool.py —— 把"视觉复核"的结论变成干净的数据池。

输入：`<root>/review/index.csv`（sheet, tile, crop, cls, label_by_recipe）
      `<root>/review/verdicts.csv`（sheet, tile, verdict）verdict ∈ defect / normal / unclear
输出：`<root>/verified/bad/*.png`、`<root>/verified/ok/*.png`（只放**复核过**的）
      `<root>/review/yield.json`（每个配方类的实际命中率 —— 生成失败率必须报出来）

标签规则：**按看到的标，不按配方标**。配方要求缺陷但画成正常手 ⇒ 进 ok（它确实是正常手）；
看不清 ⇒ 丢弃（不进任何一边，避免噪声）。
"""
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJ / "data" / "m3_gen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--verdicts", default="review/verdicts.csv")
    a = ap.parse_args()

    root = Path(a.root)
    index = {}
    with (root / "review" / "index.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            index[(r["sheet"], int(r["tile"]))] = r
    # 复核结论可以按 (sheet,tile) 也可按 crop 文件名给 —— 后者在重拼抽检表后依然有效，优先用它
    verdicts = {}
    by_crop = {}
    vpath = root / a.verdicts
    with vpath.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = r["verdict"].strip().lower()
            if "crop" in r and r["crop"].strip():
                by_crop[Path(r["crop"].strip()).name] = v
            if "sheet" in r and r["sheet"].strip():
                verdicts[(r["sheet"].strip(), int(r["tile"]))] = v

    out_bad, out_ok = root / "verified" / "bad", root / "verified" / "ok"
    for d in (out_bad, out_ok):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    per_class = defaultdict(Counter)
    copied = Counter()
    for key, row in index.items():
        verdict = by_crop.get(Path(row["crop"]).name) or verdicts.get(key)
        cls = row["cls"]
        per_class[cls]["total"] += 1
        if verdict is None:
            per_class[cls]["unreviewed"] += 1
            continue
        if verdict == "defect":
            dst = out_bad / ("%s_%s_%s" % (cls, row["sheet"].replace(".png", ""), Path(row["crop"]).name))
            shutil.copy(row["crop"], dst)
            per_class[cls]["defect"] += 1
            copied["bad"] += 1
        elif verdict == "normal":
            dst = out_ok / ("%s_%s_%s" % (cls, row["sheet"].replace(".png", ""), Path(row["crop"]).name))
            shutil.copy(row["crop"], dst)
            per_class[cls]["normal"] += 1
            copied["ok"] += 1
        else:
            per_class[cls]["unclear"] += 1

    yields = {}
    for cls, c in sorted(per_class.items()):
        seen = c["defect"] + c["normal"]
        yields[cls] = {"total": c["total"], "defect": c["defect"], "normal": c["normal"],
                       "unclear": c["unclear"], "unreviewed": c["unreviewed"],
                       "defect_rate": round(c["defect"] / seen, 3) if seen else None}
    (root / "review" / "yield.json").write_text(json.dumps(yields, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print("verified pool:", dict(copied), "->", out_bad.parent)
    print(json.dumps(yields, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
