"""Bailian / DashScope image client: list models, upload a file, run an instruction edit.

Endpoint map, established by probing (each line cost a failed call, so they are recorded):
  text-to-image (wanx*)     POST /api/v1/services/aigc/text2image/image-synthesis      async
  instruction edit (wanx*)  POST /api/v1/services/aigc/image2image/image-synthesis     async
  qwen-image-edit*          POST /api/v1/services/aigc/multimodal-generation/generation
                            SYNCHRONOUS. Sending X-DashScope-Async returns
                            403 "current user api does not support asynchronous calls".
  upload policy             GET  /api/v1/uploads?action=getPolicy&model=...  then a
                            multipart POST to the returned host. POST on the policy URL
                            gives 405; omitting x-oss-object-acl / x-oss-forbid-overwrite
                            from the form gives 403 "Policy Condition failed".

Two facts that decide how this is used for matting:
  - the edit endpoints want an http(s) URL or a base64 data URL; an `oss://` URL from the
    upload endpoint is rejected with "Image must be either a public URL or a Base64 string";
  - the output is a PNG **with no alpha channel**, so a cutout still needs a local key step
    (see chroma_key.py). The output is also re-generated, not a mask of the input: on this
    machine two prompt variants of the same poster agreed with each other at only alpha
    IoU 0.586, and the figure's scale drifted enough to cost 9.3% of the frame in
    over-inclusion against a hand-cut ground truth.

The API key is not in this skill tree. Resolution order: --key-file, then
$DASHSCOPE_API_KEY, then $BAILIAN_KEY_PATH, then ~/.dsh/secrets/config/keys.json,
then a legacy key json sitting next to this script (kept so older callers still work).

usage:
  python bailian_image_edit.py list
  python bailian_image_edit.py edit IN.png OUT.png "instruction" [--model qwen-image-edit-max]
  python bailian_image_edit.py upload IN.png
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid

NATIVE = "https://dashscope.aliyuncs.com"
OPENAI = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CTX = ssl.create_default_context()


SECRETS_KEY = os.path.join(os.path.expanduser("~"), ".dsh", "secrets", "config/keys.json")
LEGACY_KEY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/keys.json")


def _load_key_json(path):
    cfg = json.load(open(path, encoding="utf-8"))
    return cfg.get("api_key") or cfg["DASHSCOPE_API_KEY"]


def api_key(args) -> str:
    # 顺序：显式 --key-file → $BAILIAN_KEY_PATH → $DASHSCOPE_API_KEY → 新默认位置 → 旧位置
    if args.key_file:
        return _load_key_json(args.key_file)
    env_key = os.environ.get("DASHSCOPE_API_KEY")
    if env_key:
        return env_key
    for p in (os.environ.get("BAILIAN_KEY_PATH"), SECRETS_KEY, LEGACY_KEY):
        if p and os.path.exists(p):
            return _load_key_json(p)
    raise SystemExit("no key: --key-file / $BAILIAN_KEY_PATH / $DASHSCOPE_API_KEY / %s" % SECRETS_KEY)


def call(url, key, payload=None, headers=None, method=None, timeout=180, raw=None, content_type=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    h = {}
    if content_type:
        h["Content-Type"] = content_type
    elif payload is not None:
        h["Content-Type"] = "application/json"
    h["Authorization"] = f"Bearer {key}"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            text = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(text)
            except Exception:
                return r.status, {"raw": text}  # OSS answers a good upload with an empty body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def cmd_list(key: str) -> None:
    st, body = call(f"{OPENAI}/models", key)
    ids = [m["id"] for m in body.get("data", [])] if st == 200 else []
    print(f"GET /models -> {st}, {len(ids)} models")
    for i in ids:
        if any(k in i.lower() for k in ("image", "wanx", "wan2", "qwen-image", "z-image")):
            print("   ", i)


def cmd_upload(path: str, key: str, model: str) -> str | None:
    st, pol = call(f"{NATIVE}/api/v1/uploads?action=getPolicy&model={model}", key, method="GET")
    if st != 200:
        print("policy error", st, pol)
        return None
    d = pol["data"]
    key_name = f"{d['upload_dir']}/{uuid.uuid4().hex}{os.path.splitext(path)[1]}"
    boundary = "----dshform" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        blob = fh.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    parts = []
    for name, value in (("OSSAccessKeyId", d["oss_access_key_id"]), ("policy", d["policy"]),
                        ("Signature", d["signature"]), ("key", key_name),
                        ("x-oss-object-acl", d.get("x_oss_object_acl", "private")),
                        ("x-oss-forbid-overwrite", str(d.get("x_oss_forbid_overwrite", "true")).lower()),
                        ("success_action_status", "200")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
    head = "".join(parts).encode()
    tail = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"a{os.path.splitext(path)[1]}\"\r\nContent-Type: {mime}\r\n\r\n").encode()
    raw = head + tail + blob + f"\r\n--{boundary}--\r\n".encode()
    st, body = call(d["upload_host"], key, raw=raw,
                    content_type=f"multipart/form-data; boundary={boundary}", timeout=300)
    print(f"upload -> {st} {str(body)[:160]}")
    return f"oss://{key_name}" if st in (200, 203, 204) else None


def cmd_edit(src: str, out: str, prompt: str, key: str, model: str, negative: str | None,
             use_upload: bool, refs=(), prompt_extend: bool = False) -> None:
    content = []
    for p in [src] + list(refs):
        if p.startswith(("http", "data:")):
            url = p
        elif use_upload:
            url = cmd_upload(p, key, model) or ""
        else:
            url = data_url(p)
        if not url:
            raise SystemExit("no image url for " + p)
        content.append({"image": url})
    content.append({"text": prompt})            # 多图输入：每张参考图一个 image 项，最后是文字指令
    payload = {"model": model, "input": {"messages": [{"role": "user", "content": content}]},
        # prompt_extend=False：不先做提示词改写。官方价 $0.027504/张 → **$0.01434/张**（约 ¥0.10，省一半），
        # 且"分色掩版/纯黑白抠图"这类指令需要**逐字执行**，改写反而会把指令改坏（2026-10-12 实测）。
        "parameters": {"watermark": False, "prompt_extend": bool(prompt_extend)}}
    if negative:
        payload["parameters"]["negative_prompt"] = negative
    t0 = time.time()
    st, body = call(f"{NATIVE}/api/v1/services/aigc/multimodal-generation/generation", key, payload)
    print(f"edit -> {st} in {time.time()-t0:.1f}s")
    if st != 200:
        print(json.dumps(body)[:500])
        return
    try:
        url_out = body["output"]["choices"][0]["message"]["content"][0]["image"]
    except Exception:
        print("unexpected response:", json.dumps(body)[:700])
        return
    req = urllib.request.Request(url_out, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300, context=CTX) as r, open(out, "wb") as fh:
        fh.write(r.read())
    from PIL import Image
    print(f"saved {out} {os.path.getsize(out)} bytes  "
          f"(actual format {Image.open(out).format}, size {Image.open(out).size})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "upload", "edit"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--key-file", default=None,
                    help="密钥 JSON；缺省按 $BAILIAN_KEY_PATH → ~/.dsh/secrets/config/keys.json 回退")
    ap.add_argument("--model", default="qwen-image-edit-max")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--ref", action="append", default=[],
                    help="额外的参考图（可重复）。多图输入：参考图 + 参考图 + 文字指令。")
    ap.add_argument("--prompt-file", default="",
                    help="从文件读指令（多行提示词用这个，避免 shell 转义）")
    ap.add_argument("--use-oss-upload", action="store_true",
                    help="upload first and pass oss:// (rejected by qwen-image-edit*; kept for wanx)")
    ap.add_argument("--prompt-extend", action="store_true",
                    help="让服务端**先改写提示词**再生成（默认关：官方价 $0.027504/张 → $0.01434/张，"
                         "且掩版/分色/纯黑白抠图这类指令必须逐字执行）")
    a = ap.parse_args()
    key = api_key(a)
    if a.cmd == "list":
        cmd_list(key)
    elif a.cmd == "upload":
        print(cmd_upload(a.args[0], key, a.model))
    else:
        prompt = a.args[2] if len(a.args) > 2 else ""
        if a.prompt_file:
            prompt = open(a.prompt_file, encoding="utf-8").read()
        cmd_edit(a.args[0], a.args[1], prompt, key, a.model, a.negative,
                 a.use_oss_upload, a.ref, a.prompt_extend)


if __name__ == "__main__":
    main()
