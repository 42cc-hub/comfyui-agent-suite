"""Chunked, resumable download via explicit byte ranges.

Neither huggingface_hub nor curl -C - sustains a transfer here: the library stalled twice at
~3 GB, and curl's own resume averaged 0.26 MB/s while a plain ranged request on the same URL
measured 3.1 MB/s. So instead of relying on either resume path, each 1 GB chunk is fetched
with an explicit `-r start-end` (a fresh connection every time, and below any per-connection
cap), then appended. Progress is per-chunk, so an interrupted run resumes from the last
completed chunk rather than from a byte offset the server may treat badly.
"""

import os
import subprocess
import sys
import time

PROXY = "http://127.0.0.1:6132"
CHUNK = 256 << 20          # 256 MB: a 1 GB request died with CURLE_PARTIAL_FILE (rc=18)
TMP = r"H:\FramePack\chunks"

TARGETS = [
    ("https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
     "ltxv-2b-0.9.8-distilled-fp8.safetensors",
     r"<你的 ComfyUI 安装目录>\models\checkpoints\ltxv-2b-0.9.8-distilled-fp8.safetensors", 4.16),
    ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
     "t5xxl_fp8_e4m3fn_scaled.safetensors",
     r"<你的 ComfyUI 安装目录>\models\text_encoders\t5xxl_fp8_e4m3fn_scaled.safetensors", 4.92),
    ("https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
     "vae/diffusion_pytorch_model.safetensors",
     r"<你的 ComfyUI 安装目录>\models\vae\ltxv_vae.safetensors", 1.56),
]


def head_size(url):
    p = subprocess.run(["curl.exe", "-sIL", "--max-time", "40", "--proxy", PROXY, url],
                       capture_output=True, text=True, errors="replace")
    size = None
    for line in p.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
    return size


def fetch(url, dest, expect_gb):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    total = head_size(url)
    if not total:
        print(f"  !! cannot read size for {url}", flush=True)
        return False
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"  already complete ({total / 2**30:.2f} GB)", flush=True)
        return True

    name = os.path.basename(dest)
    print(f"  {name}: {total / 2**30:.2f} GB (api said {expect_gb} GB), {CHUNK >> 20} MB chunks",
          flush=True)
    done = 0
    part = os.path.join(TMP, name + ".part")
    if os.path.exists(part):
        done = os.path.getsize(part)
        print(f"    resuming from {done / 2**30:.2f} GB of completed chunks", flush=True)
    else:
        open(part, "wb").close()

    t0 = time.time()
    with open(part, "ab") as fh:
        while done < total:
            end = min(done + CHUNK, total) - 1
            p = subprocess.run(
                ["curl.exe", "-sL", "--fail", "--proxy", PROXY,
                 # abort a transfer that stops making progress, rather than blocking
                 # for the full --max-time on a link that goes quiet
                 "--speed-limit", "20480", "--speed-time", "25",
                 "--max-time", "300", "--connect-timeout", "20",
                 "-r", f"{done}-{end}", url],
                capture_output=True)
            if p.returncode != 0 or not p.stdout:
                print(f"    chunk {done} failed (rc={p.returncode}), retry in 5s", flush=True)
                time.sleep(5)
                continue
            fh.write(p.stdout)
            fh.flush()
            done += len(p.stdout)
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate if rate else 0
            print(f"    {done / 2**30:5.2f}/{total / 2**30:.2f} GB  "
                  f"{rate / 2**20:5.2f} MB/s  eta {eta / 60:4.1f} min", flush=True)
            if len(p.stdout) < (end - (done - len(p.stdout)) + 1):
                # short read: server closed early, keep going from where we are
                pass

    if os.path.getsize(part) != total:
        print(f"    size mismatch {os.path.getsize(part)} != {total}", flush=True)
        return False
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(part, dest)
    print(f"    -> {dest}  {os.path.getsize(dest) / 2**30:.2f} GB", flush=True)
    return True


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for url, dest, gb in TARGETS:
        if only and only not in dest:
            continue
        print(f"=== {os.path.basename(dest)}", flush=True)
        ok = fetch(url, dest, gb)
        print(f"    {'OK' if ok else 'FAILED'}\n", flush=True)
