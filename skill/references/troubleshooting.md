


---

## Troubleshooting


- **`To see the GUI go to ...` never appears** — check the job output tail; likely the sandbox write denial or a db lock (see Start/stop section).
- **`node_errors` non-empty on submit** — the graph references an unknown node or bad enum; re-query `/object_info` for the exact field values.
- **Very slow first run** — the model loads into 6 GB VRAM (async offload); SDXL/Anima at 1024² typically takes ~20–90 s including model load.
- **Garbled / wrong-character result** — check in this order: (1) the LoRA's base-model family matches the pipeline; (2) **the trigger list is complete** (sidecar `civitai.trainedWords` *and* `civitai.images[].meta.prompt`, plus `ss_tag_frequency`) and none of your own tags contradict the LoRA's own words; (3) drop conflicting prompt tags. Do **not** reach for `strength_model`/`strength_clip` first: interpolating a LoRA's strengths is only empirically smooth, not a guaranteed blend, and lowering them silently weakens the very identity you are trying to fix — keep both at **1.0** unless the LoRA visibly damages the image (then prefer another LoRA, and lower strength last).
- **Blender exits 1 at startup** — missing `--factory-startup` (it tries to write the user extension cache under `%APPDATA%\Blender`).
- **Blender exits 1 AFTER saving every file** — same extension cache, denied by the sandbox. Check the files exist rather than trusting the exit code, or set `BLENDER_USER_EXTENSIONS` to a workspace path for a clean exit.
- **Skeleton offset from the clay render** — keypoints projected in armature space; multiply by `arm_obj.matrix_world` first.
- **Depth map is a flat gray slab** — near/far anchored to the whole scene instead of the subject.
- **img2img output is monochrome** — a gray clay init at low denoise; raise denoise to ~0.8.
- **Stacked ControlNets re-frame the subject** — keep the depth weight ≥ 0.7.
- **Any HTTPS download fails with `schannel: AcquireCredentialsHandle failed`** — this host's
  shell has no direct internet and Windows TLS cannot obtain credentials inside the sandbox.
  Do not fight it: use the local proxy `127.0.0.1:6132` with **Python**
  (`scripts/fetch_url.py`), which goes through OpenSSL. `curl` and
  `Invoke-WebRequest` will keep failing no matter what you set `SecurityProtocol` to.
- **Upscale output looks soft** — check which `.pth` is actually loaded. `2x_AnimeJaNai` was
  much softer than `4x-AnimeSharp` (it is **gone** — see Machine facts), and a
  4× result downscaled to 2× beats a native 2× model. If a render still looks soft, the softness
  usually comes from an earlier step (an upscale-then-repaint chain bakes the soft input in),
  not from the last resize.
- **A local repaint changed nothing** — the most common cause is the **inverted mask**:
  `LoadImage`'s MASK is `1 − alpha`, so a mask written into alpha as-is repaints the complement
  (a silent no-op). Write `putalpha(255 - mask)` and prove it with `diff_report.py`: in-mask
  change ≈ 30 %, **out-mask 0.0000 %**. A mask that is empty or reversed shows as in-mask ≈ 0.
- **A CLI flag appears to do nothing** — read the graph that was actually submitted
  (`wf/*.json`). A `--lora-strength` that never reached `LoraLoader.strength_model` (hard-coded
  0.8) is indistinguishable from "denoise is too low" in the output; the fix is in the script,
  not in the numbers.
- **A cloud-edited region does not sit in the picture** (visible rectangle, halo, or a rope that
  no longer lines up) — the tool ignored the mask and/or re-lit the background. Never replace the
  image with its answer: paste **by mask**, use the subject-only merge with a dark guard, match
  the background's median colour offset first, and keep thin structures' original pixels
  (`references/cloud_edit_and_merge.md`).
- **A preprocessor returns `people: []` / an empty result for an image you already uploaded** —
  `LoadImage` + preprocessor results are cached **by filename**. `upload(..., overwrite=true)`
  rewrites the bytes but the node still serves the **old** cached result, so re-using one input
  filename pins whatever the first run produced — including a first-run *failure*, permanently.
  Measured on `OpenposePreprocessor`: same fixed name → `face_keypoints_2d` 0 / `people: []`
  on every repeat; same bytes under a fresh name → **face 70/70, identical bbox**, 3/3 runs.
  Do **not** "fix" this with a content hash — identical content means an identical name, which makes
  a cached empty result deterministic. Upload each detection input under a **unique** name
  (`role_stem_timestamp_rand.png`). A dose sweep then reports "no face found" on every variant after
  the first — indistinguishable from a flaky detector.
- **A script reports "detection failed" but the detector is fine** — on Windows the console defaults
  to GBK, so `print("✗ …")` raises `UnicodeEncodeError` and **kills the process**; the caller sees a
  non-zero exit and no success line. Measured: 3/3 "no face found" without the guard, 3/3 success
  with it, same image. Put `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of
  every script that prints symbols, and invoke child scripts with `python -X utf8`.
- **Deciding a download source is broken** — this host's international routes can drop
  completely (measured): hf-mirror / huggingface.co / github.com / ghproxy /
  jsdelivr / pypi.org all `URLError` (DNS resolves, TCP fails) while baidu / bing /
  modelscope.cn / mirrors.aliyun.com / pypi.tuna are fine. Run that reachability matrix **before**
  blaming the source, and prefer Python `urllib` for it. Partial file downloads are resumed with
  `Range`; a run that raises `IncompleteRead` is truncated, and only a clean read-to-EOF counts as
  complete — never decide completeness by comparing to a **guessed** size.
- **`insightface` raises `InvalidProtobuf` / `Protobuf parsing failed` on a model you never
  asked for** — `FaceAnalysis` loads **every** `.onnx` in the pack directory; `allowed_modules`
  does not stop the load. Put only the pieces you need in a separate pack dir (a minimal
  `buffalo_l_min/` with `det_10g` + `w600k_r50` is enough for **detection + recognition**, i.e.
  ArcFace identity similarity, without waiting for the big `1k3d68`). Also check the EP list:
  this host's `onnxruntime 1.29.0` has **no CUDA provider**, so FaceID runs on CPU.

