"""Read a remote file size reliably, then finish the LTX downloads.

HEAD is not trustworthy through this CDN -- it returned 1114 bytes for a 4 GB file, and that
bad value let the previous guard treat an empty file as complete and install it. Two fixes:
parse the total out of a 1-byte ranged GET's Content-Range header, and refuse to proceed
when the size is unknown or non-positive.
"""

import os
import shutil
import subprocess
import sys

PROXY = "http://127.0.0.1:6132"
CHUNKS = r"H:\FramePack\chunks"
SUB = 32 << 20

TARGETS = [
    ("https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
     "ltxv-2b-0.9.8-distilled-fp8.safetensors",
     r"<你的 ComfyUI 安装目录>\models\checkpoints\ltxv-2b-0.9.8-distilled-fp8.safetensors"),
    ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
     "t5xxl_fp8_e4m3fn_scaled.safetensors",
     r"<你的 ComfyUI 安装目录>\models\text_encoders\t5xxl_fp8_e4m3fn_scaled.safetensors"),
    ("https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
     "vae/diffusion_pytorch_model.safetensors",
     r"<你的 ComfyUI 安装目录>\models\vae\ltxv_vae.safetensors"),
]


def remote_size(url):
    """Total size from Content-Range of a 1-byte GET; HEAD is unreliable here."""
    p = subprocess.run(["curl.exe", "-sL", "-D", "-", "-o", "NUL", "--max-time", "40",
                        "--proxy", PROXY, "-r", "0-0", url],
                       capture_output=True, text=True, errors="replace")
    for line in p.stdout.splitlines():
        low = line.lower()
        if low.startswith("content-range:"):
            try:
                return int(line.split("/")[-1].strip())
            except ValueError:
                pass
    return None


def fetch_range(url, start, end, tries=8):
    want = end - start + 1
    for i in range(tries):
        p = subprocess.run(
            ["curl.exe", "-sL", "--fail", "--proxy", PROXY,
             "--speed-limit", "10240", "--speed-time", "20",
             "--max-time", "180", "--connect-timeout", "15",
             "-r", f"{start}-{end}", url],
            capture_output=True)
        if p.returncode == 0 and len(p.stdout) == want:
            return p.stdout
        print(f"      {start}-{end} try{i + 1}: rc={p.returncode} {len(p.stdout)}/{want}",
              flush=True)
        if p.returncode == 0 and len(p.stdout) > 0 and len(p.stdout) < want:
            # short body: accept it and advance only by what actually arrived
            return p.stdout
    return None


def finish(url, dest, force=False):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.makedirs(CHUNKS, exist_ok=True)
    total = remote_size(url)
    print(f"  remote size = {total}", flush=True)
    if not total or total <= 0:
        print("    refusing: size unknown (never treat unknown as complete)", flush=True)
        return False

    if os.path.exists(dest) and not force:
        got = os.path.getsize(dest)
        if got == total:
            print(f"    already complete {got / 2**30:.3f} GB", flush=True)
            return True
        print(f"    existing file is {got} != {total}, will re-fetch", flush=True)
        os.remove(dest)

    part = os.path.join(CHUNKS, os.path.basename(dest) + ".part")
    done = os.path.getsize(part) if os.path.exists(part) else 0
    if done > total:
        print(f"    staged part {done} > total, resetting", flush=True)
        open(part, "wb").close()
        done = 0
    print(f"    staged {done / 2**30:.3f} / {total / 2**30:.3f} GB", flush=True)

    with open(part, "ab") as fh:
        stall = 0
        while done < total:
            end = min(done + SUB, total) - 1
            data = fetch_range(url, done, end)
            if data is None:
                stall += 1
                if stall >= 3:
                    print("    too many failures, aborting this file", flush=True)
                    return False
                continue
            stall = 0
            fh.write(data)
            fh.flush()
            done += len(data)
            if done % (4 * SUB) < SUB or done == total:
                print(f"    {done / 2**30:5.3f}/{total / 2**30:.3f} GB", flush=True)

    got = os.path.getsize(part)
    if got != total:
        print(f"    SIZE MISMATCH {got} != {total}", flush=True)
        return False
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(part, dest)
    print(f"    OK -> {dest}  {got / 2**30:.3f} GB", flush=True)
    return True


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for u, d in TARGETS:
        if only and only not in d:
            continue
        print(f"=== {os.path.basename(d)}", flush=True)
        print("   ", "OK" if finish(u, d) else "FAILED", flush=True)
