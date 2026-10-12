#!/usr/bin/env python3
"""基线探针：对一批图跑现成检测器，量化"二次元上到底能不能检到人/手"。

    python det_probe.py <图片目录> [--limit N] [--glob "*.png"] [--out baseline.md]

做两件事：
  ① 经 ComfyUI 跑 `DWPreprocessor` 与 `OpenposePreprocessor`（各输出 POSE_KEYPOINT）；
  ② 解析两者的 JSON，出表：`people 数`、`手部关键点是否有效`（**注意：OpenPose 的手部点是占位假数据**，
     见下方"已知坑"），并可附 WD14 的"手/臂位置"标签（`bound wrists` / `arms behind back` / `straitjacket`
     —— 用来判断这张图**手到底可不可见**）。

已知坑（2026-10-10 实测，别重复踩）：
  * `OpenposePreprocessor` 的坐标是**归一化 0~1**，不是像素；
  * 它的 `hand_*_keypoints_2d` 坐标全 0、分数全 1.0 ⇒ **不能当"检到手了"**；
  * `MeshGraphormer`（HandRefiner）在本机 ImportError，且与别的节点同图会**整条中止** ⇒ 本脚本刻意不带它；
  * `DWPreprocessor` 在蜷/躺姿势上大量 `people: []`（YOLOX 是照片训练的）。
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time

from PIL import Image

PIPE = os.environ.get("DSH_PIPE", r"<你的 ComfyUI 项目>\pipeline")
COMFY_IN = os.environ.get("DSH_COMFY_IN", r"<你的 ComfyUI 安装目录>\input")
COMFY_OUT = os.environ.get("DSH_COMFY_OUT", r"<你的 ComfyUI 安装目录>\output")
PY = sys.executable
RUN = os.path.join(PIPE, "comfy_run.py")


def build_wf(in_name, tag):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": in_name}},
        "2": {"class_type": "DWPreprocessor",
              "inputs": {"image": ["1", 0], "detect_hand": "enable", "detect_body": "enable",
                         "detect_face": "disable", "resolution": 1280,
                         "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384.onnx",
                         "scale_stick_for_xinsr_cn": "disable"}},
        "3": {"class_type": "SavePoseKpsAsJsonFile",
              "inputs": {"pose_kps": ["2", 1], "filename_prefix": "dsh_probe/dw_%s" % tag}},
        "4": {"class_type": "OpenposePreprocessor",
              "inputs": {"image": ["1", 0], "detect_hand": "enable", "detect_body": "enable",
                         "detect_face": "disable", "resolution": 1024,
                         "scale_stick_for_xinsr_cn": "disable"}},
        "5": {"class_type": "SavePoseKpsAsJsonFile",
              "inputs": {"pose_kps": ["4", 1], "filename_prefix": "dsh_probe/op_%s" % tag}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "dsh_probe_op_%s" % tag,
                                                    "images": ["4", 0]}},
    }


def trip(flat):
    return [(float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
            for i in range(0, len(flat) - 2, 3)]


def hand_valid(people):
    """OpenPose 的手部点是否**真的**存在（判据：有非零坐标；分数不可信）。"""
    ok = 0
    for p in people:
        for k in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
            kp = trip(p.get(k) or [])
            if any(abs(x) > 1e-6 or abs(y) > 1e-6 for x, y, _ in kp):
                ok += 1
    return ok


def load_last(prefix):
    fs = sorted(glob.glob(os.path.join(COMFY_OUT, prefix + "*.json")), key=os.path.getmtime)
    if not fs:
        return None
    try:
        return json.load(open(fs[-1], encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--out", default="")
    ap.add_argument("--wd14", action="store_true",
                    help="顺带跑 WD14 取「手/臂位置」标签（判断这张图手可不可见）")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.img_dir, a.glob), recursive=True))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print("没有匹配的图：%s" % a.img_dir); return 1

    tagger = None
    if a.wd14:
        sys.path.insert(0, r"<你的用户目录>\.dsh\skills\comfyui-image-generation\scripts")
        from _libs import add_lib; add_lib("wd14-gate")
        import wd14_gate
        tagger = wd14_gate.Tagger()

    rows = []
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
        in_name = "dsh_probe_%s.png" % tag
        Image.open(f).convert("RGB").save(os.path.join(COMFY_IN, in_name))
        wfp = os.path.join(COMFY_IN, "wf_probe_%s.json" % tag)
        io.open(wfp, "w", encoding="utf-8").write(json.dumps(build_wf(in_name, tag),
                                                            ensure_ascii=False, indent=1))
        t0 = time.time()
        subprocess.run([PY, "-X", "utf8", RUN, wfp, os.path.join(COMFY_IN, "probe_%s.png" % tag),
                        "--timeout", "900"], capture_output=True, text=True)
        dw = load_last("dsh_probe/dw_%s" % tag) or [{}]
        op = load_last("dsh_probe/op_%s" % tag) or [{}]
        n_dw = len((dw[0] or {}).get("people") or [])
        n_op = len((op[0] or {}).get("people") or [])
        hv = hand_valid((op[0] or {}).get("people") or [])
        hint = ""
        if tagger:
            try:
                _, tags = tagger.run(f, threshold=0.12)
                hint = ",".join(t for t in tags if any(k in t for k in
                                ("bound", "behind", "straitjacket", "handcuff", "restrained")))
            except Exception:
                hint = "?"
        rows.append((tag, n_dw, n_op, hv, hint))
        print("%-28s DWpeople=%-2d OPpeople=%-2d 手部有效=%-2d %s  (%.0fs)"
              % (tag, n_dw, n_op, hv, hint, time.time() - t0), flush=True)

    print("\n== 汇总 ==")
    hit_dw = sum(1 for r in rows if r[1] > 0)
    hit_op = sum(1 for r in rows if r[2] > 0)
    hands = sum(1 for r in rows if r[3] > 0)
    print("图 %d 张：DWPreprocessor 检到人 %d 张、OpenposePreprocessor %d 张、**手部关键点有效 %d 张**"
          % (len(rows), hit_dw, hit_op, hands))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("| 图 | DW people | OpenPose people | 手部有效 | 手/臂位置标签 |\n|---|---|---|---|---|\n")
            for r in rows:
                fh.write("| %s | %d | %d | %d | %s |\n" % r)
            fh.write("\n合计 %d 张：DW 命中 %d、OpenPose 命中 %d、手部关键点有效 %d\n"
                     % (len(rows), hit_dw, hit_op, hands))
        print("→ %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
