#!/usr/bin/env python
"""cv_m3_synth.py —— 在**固定真实评测集**上测"合成缺陷数据能不能提升 M3"。

与 `cv_m3.py` 同一条按源图分组的分折逻辑，但：
  · 评测集**永远是那批人工标注的真实裁图**（`cls_bin_v8_noaug`，764 张 / bad 334）；
  · 合成样本**只进训练折**，绝不进评测折（否则就是把答案抄进考卷）；
  · 按比例扫：`--ratios 0,1,2` = 每折训练集里合成 bad 张数 / 真实 bad 张数。

输出：`<out>` 汇总 JSON + 每个比例的 OOF CSV。取 ratio=0 与基线（acc 0.795 / AUC 0.817 /
bad recall 0.686）对表，就是这套 harness 的自证。

用法：
  python scripts/cv_m3_synth.py --synth-bad data/m3_gen/verified/bad --synth-ok data/m3_gen/verified/ok \
      --ratios 0,1,2 --folds 5 --epochs 50
"""
import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
HERE = PROJ / "scripts"
PY = sys.executable
REAL = DATA / "cls_bin_v8_noaug"


def real_rows(root=REAL):
    out = []
    for sub in ("train", "val"):
        for lab in ("ok", "bad"):
            d = root / sub / lab
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() == ".png":
                    out.append((f, f.name, lab))
    return out


def load_manifest():
    man = {}
    p = DATA / "crops_todo" / "manifest.csv"
    if p.exists():
        with io.open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                man[r["crop"]] = r
    return man


def group_of(name, man):
    r = man.get(name)
    if r:
        return r.get("source") or name
    m = re.match(r"^(.*?)_(hand|left_foot|right_foot|foot)\d*", name)
    return m.group(1) if m else name


def assign_folds(rows, manifest, folds):
    groups = {}
    for p, n, lab in rows:
        groups.setdefault(group_of(n, manifest), []).append((p, n, lab))
    order = sorted(groups.items(), key=lambda kv: -sum(1 for x in kv[1] if x[2] == "bad"))
    fold_of, cnt = {}, [{"bad": 0, "n": 0} for _ in range(folds)]
    for g, items in order:
        k = min(range(folds), key=lambda i: (cnt[i]["bad"], cnt[i]["n"]))
        fold_of[g] = k
        cnt[k]["bad"] += sum(1 for x in items if x[2] == "bad")
        cnt[k]["n"] += len(items)
    return fold_of, cnt


def metrics(rows, threshold=0.5):
    pos = [r["p_bad"] for r in rows if r["label"] == "bad"]
    neg = [r["p_bad"] for r in rows if r["label"] == "ok"]
    acc = sum(1 for r in rows if (r["p_bad"] >= threshold) == (r["label"] == "bad")) / max(1, len(rows))
    tp = sum(1 for x in pos if x >= threshold)
    fp = sum(1 for x in neg if x >= threshold)
    pairs = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    auc = pairs / (len(pos) * len(neg)) if pos and neg else float("nan")
    return {"n": len(rows), "bad": len(pos), "ok": len(neg), "acc@0.5": round(acc, 4),
            "auc": round(auc, 4), "bad_recall": round(tp / max(1, len(pos)), 4),
            "bad_precision": round(tp / max(1, tp + fp), 4),
            "ok_fp_rate": round(fp / max(1, len(neg)), 4)}


def augment_bad(src_png, out_dir, copies, rng):
    """把一张真实 bad 裁图做 k 份增强（标签确定、零标注成本）。

    只做"几何+亮度"级别的增强，不改解剖结构 —— 目的是测"多为 bad 类提供视角多样性"能不能提召回。
    """
    from PIL import Image, ImageEnhance
    made = []
    with Image.open(src_png) as im:
        base = im.convert("RGB")
        for i in range(copies):
            img = base
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = rng.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            scale = rng.uniform(0.85, 1.12)
            w, h = img.size
            nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
            if scale > 1:                      # 放大后裁回中心
                left, top = (nw - w) // 2, (nh - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:                              # 缩小后贴回中心（背景填平均色）
                canvas = Image.new("RGB", (w, h), img.resize((1, 1)).getpixel((0, 0)))
                canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
                img = canvas
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.08))
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
            name = "aug_%s_%d.png" % (Path(src_png).stem, i)
            img.save(out_dir / name)
            made.append(name)
    return made


def build_fold_dir(fold_dir, rows, fold_of, manifest, k, synth_bad, synth_ok, n_synth_bad, n_synth_ok,
                   rng, aug_bad=0):
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    for lab in ("ok", "bad"):
        for sub in ("train", "val"):
            (fold_dir / sub / lab).mkdir(parents=True, exist_ok=True)
    for p, n, lab in rows:
        sub = "val" if fold_of[group_of(n, manifest)] == k else "train"
        dst = fold_dir / sub / lab / n
        if not dst.exists():
            shutil.copy(p, dst)
    aug_made = 0
    if aug_bad > 0:
        for p, n, lab in rows:
            if lab != "bad" or fold_of[group_of(n, manifest)] == k:
                continue
            aug_made += len(augment_bad(p, fold_dir / "train" / "bad", aug_bad, rng))
    picked = []
    for src_list, lab, count in ((synth_bad, "bad", n_synth_bad), (synth_ok, "ok", n_synth_ok)):
        if not src_list or count <= 0:
            continue
        take = rng.sample(src_list, min(count, len(src_list)))
        for i, p in enumerate(take):
            dst = fold_dir / "train" / lab / ("synth_%s_%03d_%s" % (lab, i, Path(p).name))
            if not dst.exists():
                shutil.copy(p, dst)
            picked.append((str(p), lab))
    picked.append(("<aug_bad x%d>" % aug_bad if aug_bad else "<no aug>", str(aug_made)))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(REAL))
    ap.add_argument("--synth-bad", default=str(DATA / "m3_gen" / "verified" / "bad"))
    ap.add_argument("--synth-ok", default=str(DATA / "m3_gen" / "verified" / "ok"))
    ap.add_argument("--ratios", default="0,1,2", help="合成 bad 张数 / 真实 bad 张数（每折训练集）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--work", default=str(DATA / "cv_m3_synth"))
    ap.add_argument("--out", default=str(DATA / "cv_m3_synth_summary.json"))
    ap.add_argument("--only-ratio", type=int, default=-1, help="只跑某个比例（调试用）")
    ap.add_argument("--aug-bad", type=int, default=0,
                    help="每张真实 bad 裁图额外生成 k 份增强（只进训练折）——零标注成本地给 bad 类补多样性")
    a = ap.parse_args()

    from ultralytics import YOLO

    real = real_rows(Path(a.real))
    manifest = load_manifest()
    fold_of, cnt = assign_folds(real, manifest, a.folds)
    print("real crops %d (bad %d) | groups %d | folds %d | per-fold %s" % (
        len(real), sum(1 for x in real if x[2] == "bad"), len(fold_of), a.folds,
        [(c["bad"], c["n"]) for c in cnt]), flush=True)

    synth_bad = sorted(Path(a.synth_bad).glob("*.png")) if Path(a.synth_bad).is_dir() else []
    synth_ok = sorted(Path(a.synth_ok).glob("*.png")) if Path(a.synth_ok).is_dir() else []
    print("synth pool: bad %d / ok %d" % (len(synth_bad), len(synth_ok)), flush=True)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ratios = [int(x) for x in a.ratios.split(",") if x.strip()]
    if a.only_ratio >= 0:
        ratios = [a.only_ratio]

    summary = {"baseline_reference": {"acc@0.5": 0.795, "auc": 0.817, "bad_recall": 0.686,
                                      "bad_precision": 0.815, "n": 764, "bad": 334, "ok": 430},
               "real_crops": len(real), "synth_pool": {"bad": len(synth_bad), "ok": len(synth_ok)},
               "folds": a.folds, "epochs": a.epochs, "ratios": {}}
    for ratio in ratios:
        oof = []
        for k in range(a.folds):
            real_bad_train = sum(1 for p, n, lab in real if lab == "bad" and fold_of[group_of(n, manifest)] != k)
            n_bad = int(round(real_bad_train * ratio))
            n_ok = n_bad  # 类别平衡：合成 ok 与合成 bad 等量
            rng = random.Random(a.seed + k * 17 + ratio * 101)
            fold_dir = work / ("r%d_f%d" % (ratio, k))
            picked = build_fold_dir(fold_dir, real, fold_of, manifest, k, synth_bad, synth_ok, n_bad, n_ok, rng,
                                    aug_bad=a.aug_bad)
            name = "cv_m3_synth_r%d_f%d_aug%d_s%d_e%d" % (ratio, k, a.aug_bad, a.seed, a.epochs)
            weight = PROJ / "runs" / name / "weights" / "best.pt"
            logp = PROJ / "runs" / "_logs" / ("%s.log" % name)
            logp.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, "-X", "utf8", str(HERE / "train.py"), "--task", "cls", "--data", str(fold_dir),
                   "--model", a.model, "--epochs", str(a.epochs), "--imgsz", str(a.imgsz),
                   "--name", name, "--seed", str(a.seed), "--batch", "32"]
            t0 = time.time()
            with io.open(logp, "w", encoding="utf-8") as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJ))
            print("  ratio %d fold %d: train %.0fs rc=%s synth_in_train=%d" % (
                ratio, k, time.time() - t0, r.returncode, len(picked)), flush=True)
            if not weight.exists():
                print("  !! no weights for ratio %d fold %d" % (ratio, k), flush=True)
                continue
            m = YOLO(str(weight))
            bi = [i for i, n in enumerate(m.names.values()) if n == "bad"]
            val_dir = fold_dir / "val"
            for lab in ("ok", "bad"):
                for f in sorted((val_dir / lab).iterdir()):
                    if f.suffix.lower() != ".png":
                        continue
                    pr = m.predict(str(f), verbose=False)[0].probs.data.tolist()
                    oof.append({"crop": f.name, "fold": k, "seed": a.seed, "label": lab,
                                "p_bad": round(max(pr[j] for j in bi), 4)})
        if oof:
            csv_path = work / ("oof_r%d.csv" % ratio)
            with io.open(csv_path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["crop", "fold", "seed", "label", "p_bad"])
                w.writeheader()
                w.writerows(oof)
            m_ = metrics(oof)
            summary["ratios"]["r%d" % ratio] = dict(m_, csv=str(csv_path))
            print("ratio %d: %s" % (ratio, json.dumps(m_)), flush=True)
        else:
            summary["ratios"]["r%d" % ratio] = {"error": "no oof rows"}

    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsummary ->", a.out)
    print(json.dumps(summary["ratios"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
