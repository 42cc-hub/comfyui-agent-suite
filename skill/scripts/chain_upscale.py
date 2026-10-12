"""Upscale chain: 4x sharpen -> script-controlled shrink -> 4x sharpen.

The upscale model's factor is fixed at 4x, so the only precise control over the final
size is the shrink in the middle. Going up first (as a sharpen/repaint pass) and then
shrinking gives the headroom needed to land on the delivery size, and finishing with a
model pass instead of a resample keeps the result crisper than 4x-then-downscale.

Delivery convention: 2x the source. For source size S that means
    step1  4S
    step2  4S * 0.125 = 0.5S
    step3  4 * 0.5S   = 2S
"""

import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid

from PIL import Image

BASE = "http://127.0.0.1:7187"
INPUT_DIR = r"<你的 ComfyUI 安装目录>\input"
MODEL = "4x-AnimeSharp.pth"


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))


def upscale4x(src_path, tag):
    """Run 4x-AnimeSharp once; return the resulting PIL image."""
    staged = f"dsh_chain_{tag}.png"
    shutil.copyfile(src_path, os.path.join(INPUT_DIR, staged))
    graph = {
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": MODEL}},
        "2": {"class_type": "LoadImage", "inputs": {"image": staged}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": f"dsh_chain_{tag}", "images": ["3", 0]}},
    }
    res = post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if res.get("node_errors"):
        raise SystemExit(f"node_errors {json.dumps(res['node_errors'])[:600]}")
    pid = res["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 900:
        entry = json.loads(urllib.request.urlopen(f"{BASE}/history/{pid}", timeout=120)
                           .read().decode("utf-8")).get(pid)
        if entry:
            if entry.get("status", {}).get("status_str") == "error":
                raise SystemExit(f"ERROR {json.dumps(entry['status'])[:600]}")
            for out in entry.get("outputs", {}).values():
                for im in out.get("images", []):
                    q = urllib.parse.urlencode({"filename": im["filename"],
                                                "subfolder": im.get("subfolder", ""),
                                                "type": im.get("type", "output")})
                    data = urllib.request.urlopen(f"{BASE}/view?{q}", timeout=300).read()
                    os.remove(os.path.join(INPUT_DIR, staged))
                    tmp = os.path.join(os.path.dirname(src_path) or ".",
                                       f"_chain_{tag}.png")
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                    return Image.open(tmp).convert("RGB"), tmp
        time.sleep(4)
    raise SystemExit("TIMEOUT")


def main():
    src, out = sys.argv[1], sys.argv[2]
    # intermediate size as a multiple of the source. 0.5 lands the delivery at 2x the
    # source; 1.0 keeps the source's full information and lands it at 4x. Going BELOW
    # the source here throws information away before the final model pass, which then
    # invents detail and compounds AnimeSharp's sharpening into artifacts.
    mid_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    steps = []
    s = Image.open(src).convert("RGB")
    steps.append(("source", s.size))

    big, tmp1 = upscale4x(src, "a")
    steps.append(("4x sharpen", big.size))

    target_inter_w = max(8, round(s.width * mid_scale))
    f = target_inter_w / big.width
    mid = big.resize((target_inter_w, max(8, round(big.height * f))), Image.LANCZOS)
    mid_path = os.path.join(os.path.dirname(src) or ".", "_chain_mid.png")
    mid.save(mid_path)
    steps.append((f"shrink x{f:.4f}", mid.size))

    final, tmp2 = upscale4x(mid_path, "b")
    steps.append(("4x final", final.size))
    final.save(out)

    for name, size in steps:
        print(f"  {name:18} {size[0]}x{size[1]}")
    print(f"saved {out} ({os.path.getsize(out)} bytes)")
    for p in (tmp1, tmp2, mid_path):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
