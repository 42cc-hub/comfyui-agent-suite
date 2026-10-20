#!/usr/bin/env python
"""gen_m3_defects.py —— 为 M3（手/脚好/坏分类器）批量生成"缺陷"样本。

标签由**生成配方**决定（这是本方案的核心假设：配方即标签），不靠人眼：
  bad 类：six fingers / four fingers / webbed fingers / malformed hand / six toes / four toes …
  ok  类：five fingers / five toes（同时给 ok 补量，防类别不平衡）

引擎：`--engine anima`（本机 anima-base-v1.0 + qwen 0.6B 编码器 + qwen_image_vae）
      `--engine il`（waiIllustriousSDXL_v170）
输出：`<out>/<engine>_<cls>_s<seed>.png`，并写一份 `runs.jsonl` 记录每次提交的 prompt/workflow/seed/文件名。

队列礼貌（与 skill 的硬规则一致）：提交前查 `/queue`，有别人在跑就等；**从不 /interrupt、从不清队列**。
出图一律 `PreviewImage`（落 temp，不走 SaveImage 写 output/）。

用法：
  python scripts/gen_m3_defects.py --classes six,four,webbed,ok5 --seeds 1 --size 1024 --steps 24
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("COMFY_BASE", "http://127.0.0.1:7187")
PROJ = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJ / "data" / "m3_gen" / "raw"

QUALITY_NEG = ("low quality, worst quality, lowres, blurry, jpeg artifacts, watermark, text, "
               "signature, cropped, out of frame")
DEFECT_NEG = "six fingers, extra digit, missing finger, webbed fingers, fused fingers, deformed hand, malformed hand"

# cls -> (正向片段, 额外负面)
PROMPTS = {
    "six":     ("spread fingers, six fingers, extra digit", ""),
    "five":    ("spread fingers, four fingers, missing finger", ""),
    "webbed":  ("webbed fingers, fused fingers, skin bridge between fingers", ""),
    "malformed": ("broken fingers, twisted fingers, claw hand, malformed hand", ""),
    "six_toes": ("bare feet, toes in frame, six toes, extra toe", ""),
    "four_toes": ("bare feet, toes in frame, four toes, missing toe", ""),
    "ok5":     ("spread fingers, five fingers, normal hands", DEFECT_NEG),
    "ok5toe":  ("bare feet, toes in frame, five toes, normal feet", DEFECT_NEG),
}

BASE_POS = ("masterpiece, best quality, absurdres, anime style, 1girl, solo, close-up on hands, "
            "hands in frame, palms facing viewer, bare hands, soft lighting, simple background, white background")
BASE_POS_FOOT = ("masterpiece, best quality, absurdres, anime style, 1girl, solo, feet in frame, "
                 "barefoot, soles up, toes in frame, soft lighting, simple background, white background")

# 构图（style）：第一版证明 `1girl + close-up on hands` 会画成"人物全身、手不可见"。
#   handsonly —— 只要手，不要人物；裁图用**中心裁**（绕开检测器）
#   upper     —— 上半身 + 手举到脸前，手大、检测器能找到
#   palms     —— 保留人物，但明确"手掌朝镜头"
STYLES = {
    "handsonly": {
        "pos": ("masterpiece, best quality, absurdres, anime style, hands only, no character, two hands, "
                "extreme close-up of hands, fingers spread, bare hands, soft lighting, "
                "simple background, white background"),
        "neg": "1girl, 1boy, face, head, body, arms, torso, portrait",
        "crop": "center",
    },
    "upper": {
        "pos": ("masterpiece, best quality, absurdres, anime style, 1girl, solo, upper body, "
                "hands up, hands in front of face, fingers spread, bare hands, soft lighting, "
                "simple background, white background"),
        "neg": "",
        "crop": "det",
    },
    "palms": {"pos": BASE_POS, "neg": "", "crop": "det"},
    # 单手特写：`handsonly` 会让模型画"两只手叠在一起"，指头数根本数不清（2026-10-20 实测）
    "onehand": {
        "pos": ("masterpiece, best quality, absurdres, anime style, one hand only, single hand, "
                "back of hand facing viewer, palm facing viewer, fingers spread wide, five fingers, "
                "bare hand, wrist, soft lighting, simple background, white background"),
        "neg": ("1girl, 1boy, face, head, body, arms, torso, portrait, two hands, hands together, "
                "holding hands, multiple hands, interlocked fingers"),
        "crop": "center",
    },
    "onefoot": {
        "pos": ("masterpiece, best quality, absurdres, anime style, one foot only, single bare foot, "
                "sole facing viewer, toes spread, five toes, ankle, soft lighting, "
                "simple background, white background"),
        "neg": ("1girl, 1boy, face, body, legs, two feet, both feet, shoes, socks, toes curled"),
        "crop": "center",
    },
}


def get(path, timeout=15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def queue_len():
    q = get("/queue")
    return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


def wait_idle(max_wait=1800, poll=15):
    """等队列空出来（不 interrupt，不清队列）。等不到就放弃本次提交。"""
    waited = 0
    while waited <= max_wait:
        running, pending = queue_len()
        if running == 0 and pending == 0:
            return True
        time.sleep(poll)
        waited += poll
    return False


def build_workflow(engine, positive, negative, seed, size, steps, cfg):
    if engine in ("anima", "turbo"):
        unet = "anima-turbo-v1.1.safetensors" if engine == "turbo" else "anima-base-v1.0.safetensors"
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "7": {"class_type": "EmptyLatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                                                       "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
                                                       "model": ["1", 0], "positive": ["5", 0],
                                                       "negative": ["6", 0], "latent_image": ["7", 0]}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
        }
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "Illustrious\\anime\\waiIllustriousSDXL_v170.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                                                   "sampler_name": "euler_ancestral", "scheduler": "karras", "denoise": 1,
                                                   "model": ["1", 0], "positive": ["2", 0],
                                                   "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
    }


def fetch_image(record):
    for node_out in (record.get("outputs") or {}).values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "temp")})
            with urllib.request.urlopen(BASE + "/view?" + q, timeout=60) as r:
                return r.read(), img["filename"]
    return None, None


def run_one(engine, cls, seed, out_dir, size, steps, cfg, style="palms", timeout=600):
    piece, extra_neg = PROMPTS[cls]
    conf = STYLES[style]
    base = conf["pos"] if "toe" not in cls else BASE_POS_FOOT
    positive = base + ", " + piece
    negative = ", ".join(x for x in (QUALITY_NEG, conf["neg"], extra_neg) if x)
    wf = build_workflow(engine, positive, negative, seed, size, steps, cfg)
    if not wait_idle():
        return None, "queue busy"
    sub = post("/prompt", {"prompt": wf, "client_id": "m3-gen"})
    pid = sub["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get("/history/" + pid)
        rec = hist.get(pid)
        if rec and (rec.get("outputs") or rec.get("status", {}).get("completed")):
            data, filename = fetch_image(rec)
            if data is None:
                return None, "no image in outputs"
            path = out_dir / ("%s_%s_%s_s%d.png" % (engine, cls, style, seed))
            path.write_bytes(data)
            return path, filename
        time.sleep(2)
    return None, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="six,five,webbed,malformed,ok5")
    ap.add_argument("--seeds", type=int, default=1, help="每类生成几张（seed 递增）")
    ap.add_argument("--start-seed", type=int, default=20261015)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=0, help="0 = 按引擎默认（turbo 10 / anima 24 / il 30）")
    ap.add_argument("--cfg", type=float, default=0.0, help="0 = 按引擎默认（turbo 1.0 / anima 4.5 / il 6）")
    ap.add_argument("--engine", default="turbo", choices=["turbo", "anima", "il"])
    ap.add_argument("--style", default="palms", choices=list(STYLES),
                    help="构图：handsonly（只要手，配中心裁）/ upper（上半身手举脸前）/ palms（人物+手掌朝镜头）")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max", type=int, default=0, help="最多生成几张（0=不限）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "runs.jsonl"
    # 引擎默认步数/CFG（turbo 的元数据：CFG 1、8-12 步）
    steps = args.steps or {"turbo": 10, "anima": 24, "il": 30}[args.engine]
    cfg = args.cfg or {"turbo": 1.0, "anima": 4.5, "il": 6.0}[args.engine]
    todo = [(cls, args.start_seed + i) for cls in args.classes.split(",") if cls in PROMPTS
            for i in range(args.seeds)]
    if args.max:
        todo = todo[: args.max]
    print("planned:", len(todo), "images | engine:", args.engine, "| out:", out_dir)
    ok = 0
    for cls, seed in todo:
        t0 = time.time()
        path, info = run_one(args.engine, cls, seed, out_dir, args.size, steps, cfg, args.style)
        status = "OK" if path else "FAIL"
        print("  %-10s seed=%-9d %s  %5.1fs  %s" % (cls, seed, status, time.time() - t0, info))
        if path:
            ok += 1
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"cls": cls, "seed": seed, "file": path.name, "engine": args.engine,
                                     "style": args.style, "crop": STYLES[args.style]["crop"],
                                     "size": args.size, "steps": args.steps, "cfg": args.cfg}, ensure_ascii=False) + "\n")
    print("done: %d/%d" % (ok, len(todo)))


if __name__ == "__main__":
    main()
