---
name: comfyui-image-generation
description: Generate images through a local ComfyUI server via its HTTP API, including the Illustrious (SDXL anime) and Anima (qwen-image) model pipelines, LoRA selection/trigger-word lookup, submit/poll/save automation, Blender-authored ControlNet/LLLite control maps (depth, OpenPose pose with hands, img2img), cutting the background out of renders or existing illustrations (local SAM matting, chroma key, or a hosted image-edit model) with measured scoring against a ground truth, and self-checking a finished render with a trained detector suite (people/hands/feet boxes, finger-and-toe count suspicion, gag/blindfold, rope mask) via `scripts/check_parts.py`. Use when rendering, iterating on, or cutting out images instead of using a hosted image API, or when a render needs its hands/feet/digits verified before delivery.
allowed-tools: Read Write Edit Bash
license: MIT
compatibility: Requires a running ComfyUI server (<你的 ComfyUI 安装目录>, conda env sdenv, port 7187, NVIDIA GPU). Uses `node` (Node 18+) for the API calls; Blender 5.1 at `<你的 Blender 安装目录>` for the control-map pipeline; `ffmpeg` optional only if audio->image conversion is needed elsewhere. Matting additionally needs `pip install segment-anything` (small pure-Python package) and the SAM ViT-B checkpoint that ships with this skill at `models/sams/sam_vit_b_01ec64.pth`.
metadata:
  version: "1.12"
  skill-author: local (machine-specific operational notes)
---

# ComfyUI Image Generation

Drive a **local ComfyUI** instance over its HTTP API to generate images. This skill covers the
request/response flow, the two model pipelines on this machine — **Illustrious (IL, SDXL anime)**
and **Anima (qwen-image)** — and everything that happens *after* a picture exists: ControlNet /
Blender-authored control maps, masked local repaint (smart crop, local and cloud-assisted),
upscaling and final-size reduction, matting, comic-page tiling, cloud API editing, human-in-the-loop
masking/Photoshop handoff, image localization/measurement, and the safety gate that decides whether a picture may enter the agent's context at all.

## How these docs are written

Rule, trigger, threshold, anti-pattern, tool pointer — nothing else: no incident stories, no chronology,
no dates/attributions on a rule, no migration notes, no conclusion told twice; quantitative evidence stays,
one line, tagged `[measured]`/`[corpus]`/`[rule]`. Every rule must stand alone for a mid-file reader.

## Working mode — **default: semi-auto**（用户没说的不做）

Unless the user hands over the decision ("你来定 / 自行决定 / 不用问我"), run in **semi-auto**: do exactly the
action asked — **no extra steps, no unrequested model/parameter/prompt change, no unasked retry or multi-round
loop**; show each result (one panel call) before continuing; when you cannot judge the picture yourself (is the
hand broken? may it ship?) **stop and hand it to the human** — "I think it's fine" is not an acceptance result;
report **facts** (path, seed, size, seconds, detector output), never impressions; money/batch/irreversible
actions get announced first. Contract + write-permission boundary + self-check checklist:
**`references/modes_and_discipline.md`**.
## When to use
- You want to generate images on the local GPU (not a hosted API) from text prompts.
- You need to apply a LoRA to a base model and want its activation/trigger tags.
- You are iterating: change prompt / seed / LoRA weight / sampler and regenerate quickly.
- Pose, framing or perspective must be **exact** rather than prompt-guessed → author it in Blender and feed ControlNet/LLLite (see `references/blender_control_pipeline.md`).
- You need the **background removed / an alpha cutout** from a render or an existing illustration → see `references/matting_and_cutout.md`; difficulty differs by an order of magnitude depending on whether the image is ours (flat generated background) or someone else's (poster).
- You need to **fix a detail only a few pixels wide** on a finished image (a shoelace, an eyelet) without disturbing the rest → SmartImageCrop + `SetLatentNoiseMask` + SmartImageStitcher; see `references/upscaling.md` and `references/local_repaint_bigcanvas.md`.
- The deliverable is a **comic page** — N images tiled into one image with dividers and lettering, in reading order → see `references/comic_page_layout.md`.
- The local stack is structurally weak at the task — **text inside the image** (Chinese calligraphy, signage), adding objects on top of a hand-edited base → call a **cloud** model and paste the answer back by mask: see `references/cloud_edit_and_merge.md`.
- **A human must supply rough 2D input** (a painted mask, a few points/edges) or do real Photoshop work on this host → see the Human-in-the-loop section and `references/human_in_the_loop.md`. **When handing a file to Photoshop (`ps_edit`): ONE file per round — opening a second file while one is already open FREEZES Photoshop on this host (measured; the process must be killed and unsaved work is lost).**
- You must **know exactly where something is** before masking it (which pixels are the rope, where the face sits, which of two strengths is better) → see the Locating-things section and `references/grid_localization.md`. **要给某块区域做掩版时，先回答四个问题（形状/有没有遮挡/周围复杂吗/能不能上传云端），再在六条路里选一条** —— ① 检测框 / 骨架膨胀 / 头框内接椭圆（本机免费，默认先试）② **SAM 提示式分割**（给框+点，1~2 秒，边界最贴形状）③ 云端视觉模型分色（只传**切片**、先过 WD14 闸门，语义身份最清楚但花钱）④ 网格数格子 → 矩形或**内接椭圆**（圆/方物体两条命令）⑤ **切片 + colorblock 加减法**（周围简单、目标突出时最省；边界模糊处一律"**先外扩目标、再减掉确定的非目标**"）⑥ 画布让人涂（最稳但要人）。**几条路是零件、可以拼**（检测给粗位置、SAM 给边界、色块给同类、最后 `mask_compose --union/--sub` 组合）：取舍、实测数字、可抄的命令与交付前核对见 **`references/localization_and_masks.md`**。
- You need a **body-part mask without asking a human to paint one** → the local OpenPose node + `pose_keypoints.py` give per-part boxes (feet/legs/face/arms/hands/torso) and can write the mask: see the auto-localization section of `references/local_repaint_bigcanvas.md`. Reliable on standing/kneeling; **extreme poses (all fours, suspension) lose joints** → ask for a rough stroke or use `SAM3_Detect`.
- **Before any image enters your own context** — every vision read uploads the picture to a model provider → run the local WD14 gate first: see the Safety-gate section and `references/output_safety_gate.md`.
- A figure is **complex** (two subjects in contact, a container/rig, a crowd, an unusual camera, a foreground foot/face) and you must decide **what to sample, what to program, what can be added later** → see the Planning section and `references/planning_and_decomposition.md`.
- You are **starting a picture (or a set) and need the macro order** — when to sample, when to let a human edit, when to upscale, when to switch models → see the **Production pipeline** section and `references/production_pipeline.md`. It also carries the two-count disciplines: **many samples only at step 1**, and **upscaling happens more than once** because fine objects need pixels.
- The picture has **two or more characters** and you need them to stay distinct (who is who, what each is doing, no feature bleed) → step 1 and step 4 of `references/production_pipeline.md` (wide canvas, per-character blocks, region repaint with only that character's LoRA, mask-paste when the engine changes).
- You have a **reference illustration of an interior** and need *controlled* depth/line maps, or that image's **camera** (to place a character at the right scale), or you are compositing a character into a generated background → see `references/perspective_and_scene_geometry.md`.
- You **composited something** (a program-drawn rope/chain, a Blender part, a matted cut-out, a UI overlay) and it reads as **pasted** → see the Harmonizing section and `references/harmonization_and_embedding.md`; run `scripts/harmonize.py`.
- You are **starting or resuming a multi-image project** (a set, a series, a client job) and need the disk/metadata discipline — what to keep, how humans and agents refer to the same picture, how to avoid a growing pile of near-duplicate helper scripts → see **`references/project_management.md`**. It is the cross-project layer: object model (Task/Scene/**Image**/Artifact/Decision/Publication), ID rules, the five retention levels, the operation primitives, GC by reachability, and the agent's default read boundary. Two rules from it are load-bearing: **default ephemeral → explicit promote**, and **one canonical implementation per operation — prefer the engine's native nodes over a new Python script** (e.g. a masked repaint pastes back with `ImageCompositeMasked`, not a hand-written merge).

## Machine facts (this host)
| Thing | Value |
|---|---|
| ComfyUI dir | `<你的 ComfyUI 安装目录>` |
| Launch script | `<你的 ComfyUI 安装目录>\comfyUI.bat` (`--enable-manager --port 7187`) |
| Python | `python` |
| Server base URL | `http://127.0.0.1:7187` |
| GPU | RTX 3060 Laptop, 6 GB (vram `NORMAL_VRAM`, async offload) |
| Blender | `<你的 Blender 安装目录>\blender.exe` (5.1, headless needs `--factory-startup`) |
| SAM (matting) | `models/sams/sam_vit_b_01ec64.pth` inside this skill + `pip install segment-anything` |
| Upscalers | `models/upscale_models/`: `4x-AnimeSharp.pth` (**default**, best for illustration), `RealESRGAN_x4plus_anime_6B.pth`, `4xhfa2kludvaeswinir_v1.safetensors`. **`2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` was REMOVED** — it is visibly softer than the 4x models; it now sits in `<你的 ComfyUI 安装目录>\agent_work\_disabled_upscalers\` (out of ComfyUI's model list). Measured on a 2000×1368 illustration: |Laplacian| mean 6.08 / P99 86 (2x-AnimeJaNai) vs **9.56 / 160** (4x-AnimeSharp then Lanczos half). **Deliver with `4x-AnimeSharp → ImageUpscaleWithModel → Lanczos to the target width`**, never a native 2x model. |
| Outbound network | **The shell has no direct internet.** HTTP times out; HTTPS via curl/.NET dies with `schannel: AcquireCredentialsHandle failed`. Only working path: local proxy `127.0.0.1:6132` + **Python (OpenSSL)** → `scripts/fetch_url.py` |
| **NEVER TOUCH** | **`<你的 ComfyUI 安装目录>\output\`** — a **mixed** directory: the user's own files live there next to renders (notes, downloaded reference pngs, `output\漫画委托\` = the whole commissioned-project material). Deletion here is **unrecoverable** (`os.remove` skips the Recycle Bin). **Never delete, move, or reorganize anything under `output/`** — file-by-file, by explicit name only. |

## Cleaning up after a run (hard rules)

**Workspace boundary: work inside `<你的 ComfyUI 安装目录>\agent_work\` only.** Everything outside it (`output\`,
`input\`, `models\`, the ComfyUI tree, Desktop, Downloads) is the user's territory: **never delete, move,
rename or reorganize anything there unless the user names that exact file.** Agent products live under
`agent_work\` so human-made and machine-made files cannot be confused.

Cleaning **cannot be undone**: `os.remove`/`Remove-Item` bypass the Recycle Bin and there is no recovery path.
Last resort for a lost name: the session transcript `~/.dsh/sessions/<project>/<session>/session.v3.jsonl.zstd`
still holds the real filenames — `tools/recover_names_from_session.py`.

1. **List first, delete second.** Print every path + size + mtime before removing anything — even when the
   request is a casual "clean up the intermediate files".
2. **Whitelist by filename.** Never delete by "directory + age" or "directory + size": coarse conditions
   sweep the user's files.
3. **Mixed directories are off-limits**: `<你的 ComfyUI 安装目录>\output\`, `<你的 ComfyUI 安装目录>\input\` (except the
   staging subdir below), Desktop, Downloads. Touch them only file-by-file, by explicit name.
4. **Prove "it is a duplicate" before deleting it**: sha256, or a pixel diff of **0.0**.
5. **`input/dsh_edit/` is the one safe-to-empty directory** — it is pure staging; every script re-copies its
   inputs from the project dirs before each run.
6. **Never delete**: the user's hand-made files (PS handoffs, hand-painted mask layers), authored sources,
   `final/` dirs, exported pages, or any file whose name looks like a note / script / reference.

**Production form — never write into `output\`.** Generate with **`PreviewImage`**, not `SaveImage`: the result
lands in `temp\` and comes back via `GET /view?filename=..&type=temp`; `temp\` is **wiped on restart**, so
anything to keep is **explicitly re-saved** into the workspace/task dir as a delivery step. `output\` stays a
by-filename whitelist (§3).

## Safety gate: check before you look (hard rule)

**Reading an image with a vision tool uploads that image to the model provider.** On an account
that must not carry sensitive material, "do I look at it?" is a controlled action, not a reflex.

```bash
python libs/wd14-gate/scripts/wd14_gate.py out/*.png            # local WD14 tagger → tags + rating
python libs/wd14-gate/scripts/wd14_gate.py out/*.png --tags     # with the full tag list, for a human read
python libs/wd14-gate/scripts/nsfw_gate.py out/*.png            # pixel-statistics pre-filter (coarse)
```

| Verdict | Meaning | Action |
|---|---|---|
| `OK` | general | view it |
| `REVIEW` | sensitive / `questionable` **without** a key-point tag / **`explicit` without any key-point tag**（`escalated`）/ **only a pose tag**（`pose-level`，如 `spread legs`）/ watch-tag hit | viewable |
| `BLOCK` | **命中关键点标签**（`nipples`、`pussy`、`penis`、`anus`、`crotch`、`nude`、`topless`、`bottomless`、`presenting`、`clitoris`、`sex`、`cum`、`cameltoe`、`partially visible vulva`、`vulva`、`labia`、`visible clitoris` …）；或 `explicit` **且**命中关键点/姿势标签 | **never view** — see below |

> **A `BLOCK` image gets a path in the reply, nothing else.** Not `read_image`, and **not `dsh_show_media`**
> either — its result *does* re-enter the agent's context (measured). Review one with
> **`dsh_show_gallery`** (folder or `paths`): only a handle enters the tool result ⇒ **zero** context cost.
>
> **两次放宽：硬边界是关键点，既不是 rating，也不是姿势。**
> 依据（实测）：9 张纯面部特写（blush / shy / rolling eyes / angry+咬牙 / **orgasm face(ahegao)** /
> torogao / empty eyes / crying / naughty tongue）rating 最高只到 `questionable`，**无一到 explicit**；
> 而"旗袍 + crotch rope + 用皮带绑在大腿内侧的按摩棒"那张 rating=explicit，**却一个裸露标签都没有**；
> 横一字马那张**衣着完整、rating 仅 `sensitive`**，却因为一个 `spread legs` 被判 BLOCK
> （用户复核："没露，尺度甚至比同批其它几张还小"）⇒ **姿势被当成关键点是过杀**。
> 因此：`spread legs` 移出关键点表、进入**姿势表 `POSE_TAGS`**（单独出现 ⇒ REVIEW ＋
> `pose-level` 标记；与关键点同现、或 rating=explicit ⇒ BLOCK）；`cameltoe / partially visible vulva /
> vulva / labia / visible clitoris` 反而**提为硬 BLOCK**（它们是真关键点，不该只在 explicit 时才拦）。
> 输出里会多打两行标记，人工过目时一眼可辨"题材过杀 vs 真露点"：
> `escalated: rating=explicit but no key-point tags -> REVIEW` 与 `pose-level: spread legs (...)`。
> 阈值提醒：`cameltoe` 这类标签的置信度常落在 0.30–0.35 之间，**`--threshold 0.30` 会比默认 0.35 多拦一批**。
> 细则与完整证据：`references/output_safety_gate.md`。**风险仍由用户承担** ✓。

Full rules, the measured evidence and the "make images safe by construction" tricks (negating a
LoRA's default outfit, writing `opaque fabric, fully covered`): `references/output_safety_gate.md`.
**Censoring a BLOCK image to look at it** `[measured]`: mosaic the **whole panel** (16–64 px cells; 32–64 often reaches `OK`) — *not* the one suspicious spot, and never a flat colour fill (a flat
patch reads as a censor bar and can rate worse). The verdict depends on **context**, not only on
the pixels you covered: a panel whose middle 50% was cut out stayed `explicit` with the same tags
(`vibrator under clothes`, `vibrator cord`), and scene-level tags (`exhibitionism`, `public
indecency`, `bdsm`) survive any mosaic without blocking. So gate at the granularity you intend to
act on — a crop that gates REVIEW can still make the full frame explicit, and a collage of four
REVIEW panels can itself be BLOCK.

## Planning a complex illustration — decompose before you render

Do this **before** writing a prompt or building a control map; it is the layer that decides whether a
figure is cheap or hopeless.

- **Delete test** — remove an element from the picture: if the rest still makes sense, it **can be added
  later** (off-frame hands, a leash leading out of frame, wooden stocks, a case frame, plain geometry);
  if not, it **must be generated together with the subject** (anything hanging, and anything that leaves a
  **mark**: rope indentation, cloth deformation, contact shadow).
- **Interaction has a direction** — the side that acts on / looks at the other cannot be produced alone;
  the acted-upon side can be produced first; a symmetric pair (mutual gaze + contact) needs one generation.
- **Order layers so the uncontrollable goes first** — AI sampling cannot be aimed ⇒ sample the person,
  then build the machine/container in Blender **against that person**. (The reverse — structure first — is
  for architecture, where the room *is* the subject.)
- **"Would a human spot it?"** — countable / evenly spaced / symmetric / regularly grouped things (keyboards,
  chains, fences, grids, type) and **everyday objects** (wardrobes, desks, doors, furniture legs) are
  verified at a glance ⇒ program or Blender them. Same for **anatomy** (proportions, limb symmetry).
- **Macro vs micro** — orientation, proportions, perspective, contact, composition, and anything large in
  frame must be right **at generation time**; rope detail, text, light patches, wetness, composites are
  repairable afterwards. Filter on the macro list first.
- **Copy before props** — write the line first, then reverse-engineer which props must be visible.
- **Hiding is a layer problem, not a prompt problem** (rope that must stay inside a collar): build the
  hidden thing, then put the coat on top.

Full playbook, measured model-capability table (IL vs Anima, shibari prior, POV/selfie, text, toxic LoRAs),
the account's content rules and the layered-production checklist: **`references/planning_and_decomposition.md`**.

## Recovering perspective — a human's line annotation → camera → control maps
人手画几条线 → 求相机 → 出控制图。三步：**先把轴定死**（图像 x 右/y 下/z 进屏，世界 x 前/y 左/z 上，
旋转用**固定基线 R0 上的旋转向量**而不是自由欧拉角）→ 用人画的线做鲁棒拟合 → 用求出的相机出
depth/line 控制图。完整方法与实测数字：**`references/perspective_and_scene_geometry.md`**。

## Harmonizing a composite — stop it looking pasted

Whenever a layer is added to a finished image (program-drawn rope/chain, Blender render, matted cut-out,
UI overlay, an AI-generated extra), it reads as a sticker unless **tone, sharpness, grain and edge behaviour**
are matched to its surroundings. Use **`scripts/harmonize.py`**:

```powershell
python scripts\harmonize.py BASE PATCH --mask MASK.png --out OUT.png `
    --match lab --feather 8 --shrink 2 --ao 0.25 --ao-width 18 --ao-dir left --report
```

- `BASE` = finished illustration, `PATCH` = the layer (same canvas), `MASK` = white where the patch shows.
- `--match lab` (different-source patch) / `--match luma` (patch already carries the right colour, e.g. anime
  flat art) / `--match hist` (**only** when both sides are flat-colour with the same dynamic range).
- `--report` prints a **seam step** (mean |ΔL| across the mask border) before/after — optimise that number.
- Measured on a real case: naive paste seam **60.2** → after `--match lab` **26.0** / `--match luma` **25.4**
  (−57 %), with the patch's sharpness/noise profile brought onto the base's; `--match hist` made it *worse* (54.2).

Practical order: **classical match → masked low-denoise repaint of the seam (if the light is wrong) → classical
match again**. Full checklist (PS-side tricks, anime-specific notes, what classical matching cannot do):
**`references/harmonization_and_embedding.md`**.

## Start / stop ComfyUI
Launch the server in the background from the ComfyUI dir. **It must run with full file access** — otherwise the sandbox denies writes to its database/settings/output dirs (outside the workspace) and it exits with `[sandbox: file access denied under workspace-write mode]`.

**First-time install (repo/tag, torch+CUDA incl. the Blackwell/sm_120 branch, model dir layout, verification): `references/installing_comfyui.md`.**

```powershell
# clear any stale db lock, then launch (needs danger-full-access for the ComfyUI root and AppData writes)
Remove-Item '<你的 ComfyUI 安装目录>\user\comfyui.db-wal','<你的 ComfyUI 安装目录>\user\comfyui.db-shm' -ErrorAction SilentlyContinue
& 'python' 'main.py' '--port' '7187' '--listen' '127.0.0.1' 2>&1
```
Run that as a background job with `sandbox_permissions: "danger-full-access"` (justification: ComfyUI writes its db/settings/output outside the workspace). Wait for the ready line in the job output:
```
[INFO] To see the GUI go to: http://127.0.0.1:7187
```
Confirm with `Get-NetTCPConnection -LocalPort 7187 -State Listen` or `GET /`.

Common startup blockers:
- **db lock** — remove `comfyui.db-wal` / `comfyui.db-shm` (as above).
- **sandbox write denial** — relaunch with `danger-full-access`.
- Ignores a single custom-node import warning (`NODE_CLASS_MAPPINGS is empty ...` / `'gbk' codec`); core pipeline still works.

## The HTTP API recipe
All calls go to the server base. Node graph format: every input that references another node is `["<nodeId>", <outputIndex>]`.

1. **Discover nodes / enum values**
   `GET /object_info/<NodeClass>` → `.input.required.<field>[0]` is the enum.
   e.g. `GET /object_info/CheckpointLoaderSimple` → `ckpt_name`; `GET /object_info/LoraLoader` → `lora_name`; `GET /object_info/CLIPLoader` → `type`.
2. **Submit a workflow**
   `POST /prompt` with `{"prompt": { "<id>": {"class_type": "...", "inputs": {...}}, ...}, "client_id": "..."}` → `{prompt_id, node_errors}`.
3. **Poll for completion**
   `GET /history/<prompt_id>` until `history[prompt_id].outputs` is non-empty. Check `status.status_str === 'error'` to bail early.
4. **Save the result**
   Take the first image in any output node: `outputs[<saveNodeId>].images[0] = {filename, subfolder, type}`. Fetch `GET /view?filename=..&subfolder=..&type=output` and write bytes to a file.

A reusable script lives at `scripts/comfy_generate.mjs`:
```bash
node scripts/comfy_generate.mjs references/anima_workflow.json out.png
```
It reads the node-graph JSON, submits, polls, and writes the first output image to the given path. Set `COMFY_BASE` to override the URL (default `http://127.0.0.1:7187`).

**Before a heavy run, empty the queue** — a stale job from an interrupted session will otherwise
take the GPU and your submit just waits behind it. Use the project script
(`tools/clear_queue.py`) or this one-liner (single quotes inside, so it
survives PowerShell):

```bash
python -c "import json,urllib.request as u;[print(p, u.urlopen(u.Request('http://127.0.0.1:7187'+p, data=json.dumps(d).encode(), headers={'Content-Type':'application/json'}), timeout=20).status) for p,d in (('/interrupt',{}),('/queue',{'clear':True}))]"
```

**Hard limits on a 6 GB card** (not negotiable):
sampling **side ≤ 2400**, stay **≤ 1536 native** when upscaling a crop for sampling, and
**downscale any 4× intermediate before it enters a sampler** (≈52 MP ceilings are unreachable
here). The upscale model runs on the 4× intermediate *outside* the sampler, which is fine.

## Illustrious (IL) workflow — SDXL anime
IL is an SDXL-family anime base. Use `CheckpointLoaderSimple` (single node gives MODEL+CLIP+VAE), add optional LoRAs, then standard txt2img.

Reference graph: `references/illustrious_workflow.json`. Key nodes:

- **CheckpointLoaderSimple** `ckpt_name` = one of:
  - `Illustrious\anime\waiIllustriousSDXL_v170.safetensors` (default anime)
  - `SD\architecturerealmix_v11.safetensors`
  - `SD\deliberateCyber_v60.safetensors`
  Outputs: `[0]=MODEL, [1]=CLIP, [2]=VAE`.
- **LoraLoader** (optional) `model=ckpt[0]`, `clip=ckpt[1]`, `lora_name=<path>`, `strength_model`, `strength_clip`. Outputs `[0]=MODEL, [1]=CLIP`.
- **CLIPTextEncode** ×2 → `clip = (lora?lora[1]:ckpt[1])`.
- **EmptyLatentImage** → e.g. `1024x1024`, batch 1.
- **KSampler** `model=lora[0]|ckpt[0]`, `positive/negative`, `latent_image`, `seed`, `steps≈28`, `cfg≈6`, `sampler_name=euler_ancestral`, `scheduler=karras`, `denoise=1`.
- **VAEDecode** `samples=ksampler[0]`, `vae=ckpt[2]`.
- **PreviewImage**（**默认**；落 `temp\` 再按文件名取回，不写 `output\`）. 老工作流里的 `SaveImage` 只在"用户明确要求落盘到某个目录"时用，`filename_prefix` 指向工作区而不是 `output\`.

### Image sizes & hi-res for video
- Landscape/anime usually `1024x1024`. For a **vertical (9:16) video** use portrait SDXL ratios like `832x1216` or `896x1152`; `1080x1920` itself is valid but slow on 6 GB.
- A 832-wide source upscaled to `1080x1920` looks soft. Generate higher (e.g. `1216x1792`) or add a **hi-res fix** (a second low-denoise KSampler pass / hires-fix / upscale LoRA) for crisper video output. Keep the subject framed with headroom so a Ken-Burns crop doesn't clip it.
- **Size the canvas to the character, not to the aspect ratio** — the figure needs ≥ `700×1200` px
  or the face goes off-model. See "Subject pixel area" below before picking a resolution.

### Sampling resolution decides the *failure mode*, not just the framing

Measured (Illustrious, no LoRA, `cfg 8` / 20 steps, paired seeds, hands counted by a
human; full data in the pixiv project's `experiments/scale_vs_fingers/`):

| canvas | hands with **extra** fingers (≥6) | hands with **missing** fingers (≤4) |
|---|---|---|
| 832×1216 (1.01 MP) | **0 / 40** | 4 / 40 |
| 1232×1800 (2.22 MP) | **9 / 40** (Fisher p = 0.0024) | 0 / 40 |

- **Sampling above the training resolution makes the model invent detail.** At 1232×1800 hands
  gained extra fingers — and the rate does **not** depend on how large the hand is drawn: on the
  same large canvas, hands drawn large gave 4/20 and hands drawn tiny (full-body shot) gave 5/20,
  **Fisher p = 1.0**. It is the **canvas**, not the object's apparent size. So framing tricks
  (`close-up`, `full body`) cannot fix it — only changing the canvas can.
- **The missing-finger direction is the opposite, and *is* local:** on the small canvas, hands drawn
  small gave 4/20 missing while the same canvas framed as a close-up gave 0/20. There the object's
  pixel count is the variable — **draw the hand bigger and the merging goes away.**
- **Too small is unusable:** at 208×312 (0.065 MP) even the framing stops being controllable
  (per-seed `1girl` / `rope` tag σ up to 0.31; figures cropped at random). Usable floor ≈ 0.4 MP.
  Even at 0.065 MP the rope *content* was requested correctly — it is control that is lost.
- **Large structures only benefit from size:** a rope harness got monotonically cleaner
  (`rope` tag 0.28 → 0.89 → 0.97 across 0.065 / 0.39 / 2.97 MP).
- **Practice:** keep sampling near ~1 MP (SDXL's native) and **upscale afterwards** — the same rule
  as "upscaling ≠ re-rendering at high resolution", now with a mechanism behind it.

### Subject pixel area: the character box must not be smaller than ~700×1200

A second, independent rule. It is about **how many pixels the character
gets**, not how big the canvas is — so it governs a different failure mode from the finger-invention
result above and the two do not contradict each other:

| what you control | the failure it causes |
|---|---|
| **canvas** size (above) | sampling above the training resolution ⇒ invented detail (extra fingers) |
| **subject** pixel area (this rule) | too few pixels on the figure ⇒ mushy render, and **features — the face above all — drift off-model** |

- **Rule: the character's bounding rectangle should be at least `700×1200` px.** Below that the
  figure comes out blurry and the face in particular stops looking like the character.
- **Scene shots are the dangerous case.** A character standing in a landscape occupies a much
  smaller share of the frame than a portrait does, so a canvas that is fine for a 立绘 is not fine
  for a scene. **Raise the generation resolution to match**: if the character will only cover a
  third of the frame's height, the canvas has to be ~3× taller to still give her 1200 px.
  (Measured in this project: the 1280×720 event CGs put the whole figure in ~700 px of height, and
  her face in well under 100 px — fine at a glance, soft on inspection.)
- **Iteration vs. delivery.** Low resolution is the efficient way to explore composition and pose
  (fast, cheap, throw the misses away). But anything kept for delivery should be **upscaled AND
  have the character region repainted** — upscaling alone does not restore a face that was never
  resolved. Region repaint: `references/local_repaint_bigcanvas.md`
  (`SmartImageCrop` → `SetLatentNoiseMask` → `SmartImageStitcher`, cost tracks the crop — and the graph is
  **bit-exact outside the mask** (`max Δ 0`) as long as the crop is **not rescaled** (`force_divisibility=8`,
  `min_resolution` already cleared by the mask bbox) and `crop_mask` is wired into the stitcher's `mask`.
  Drift is **not** the stitcher's fault: measured (cases P1–P7) it comes from `SmartImageCrop` resizing the crop
  (0.0542 % even with no sampling at all) and from leaving the stitcher's `mask` unwired (0.3352 %).)

Practical recipe: trial at low res to settle framing → re-render the winner at a canvas big enough
that the character box clears `700×1200` → face/hand region repaint if still soft.

## Anima workflow — qwen-image
Anima is a newer (qwen-image) pipeline. It is NOT a standard checkpoint: load the UNET, the qwen CLIP, and the qwen VAE separately, then a LoRA. Reference graph: `references/anima_workflow.json`.

- **UNETLoader** `unet_name=anima-base-v1.0.safetensors`, `weight_dtype=default` → `[0]=MODEL`.
- **CLIPLoader** `clip_name=qwen_3_06b_base.safetensors`, `type=stable_diffusion`, `device=default` → `[0]=CLIP`.
  (Reference templates use `stable_diffusion`; the `qwen_image` type also exists but the saved workflows use `stable_diffusion`.)
- **VAELoader** `vae_name=qwen_image_vae.safetensors` → `[0]=VAE`.
- **LoraLoader** `model=unet[0]`, `clip=clip[0]`, `lora_name=<path>`, `strength_model≈0.85`, `strength_clip≈0.8` → `[0]=MODEL, [1]=CLIP`. (Those are just the values found in the reference templates — see "Finding a LoRA" for why **1.0/1.0** is the default to reach for.)
- **CLIPTextEncode** ×2 → `clip = lora[1]`.
- **EmptyLatentImage** → `1024x1024`, batch 1.
- **KSampler** `model=lora[0]`, `sampler_name=euler`, `scheduler=simple`, `steps≈30`, `cfg≈8`, `denoise=1`.
- **VAEDecode** `samples=ksampler[0]`, `vae=vae_loader[0]`.
- **PreviewImage**（**默认**；落 `temp\` 再按文件名取回，不写 `output\`）. 老工作流里的 `SaveImage` 只在"用户明确要求落盘到某个目录"时用，`filename_prefix` 指向工作区而不是 `output\`.

### Anima LoRA: DeepSeek 鲸鱼娘
- `lora_name = character\E9B2B8E9B1BCE5A898Fix.wTK8.safetensors` (card title `鲸鱼娘_fix`)
- Base: `anima` (trained on `animayume_v04`), VAE `qwen_image_vae`, network module `networks.lora_anima`.
- Character trigger tags (from `ss_tag_frequency`): `1girl, blue hair, very long hair, wavy hair, ahoge, sidelocks, hair ornament, hair flower, blindfold, covered eyes, cetacean tail, fish tail, fins, head fins, animal ears, white dress, frilled dress, off-shoulder dress, high-low skirt, choker, frilled choker, blue bow, arm garter, thigh strap, barefoot`.

### 模型选型：何时用哪个
- **Anima (qwen-image)**：**自然语言** prompt，适合**描述性/示意性**内容——图示、漫画、概念插画（如"上下两个图层、虚线分隔"这类构图）。文字类/示意图效果好。
- **Illustrious (SDXL anime)**：**tag 式** prompt，适合**动漫角色 + LoRA**（触发词 + 特征 tag），二次元立绘。
- 若内容是"图解/概念/信息图"选 Anima；"角色/立绘/具体人物"选 Illustrious+LoRA。

### 尺寸参考（按用途）
- **图示/信息图/漫画**：`768×768`（方，简洁）
- **幻灯片/横屏**：`1280×720` 或 `1344×768`
- **动漫头像**：`1024×1024`
- **竖屏视频(9:16)**：`832×1216` / `896×1152`（SDXL 竖屏比）；`1080×1920` 合法但在 6GB 上慢
- **高分辨率**：`1216×1792` 再降采样，避免放大发软；或加 hi-res fix 二次低 denoise 采样
- ⚠️ **以上是画布尺寸，不是人物尺寸**。选定前先算一下人物能占多少像素：
  **人物矩形必须 ≥ `700×1200`，否则脸会走形**。带场景的图里人物占比小，
  画布就要按比例放大（人物只占画面高度的 1/3 → 画布要 3 倍高才够她 1200px）。
  低分辨率适合快速试构图，**成品必须放大并对人物区域重绘**
  （`references/local_repaint_bigcanvas.md`）。详见上文 "Subject pixel area"。

### 系列 / 批量生成（一致性）
做一套图（如各章配图）要像"同一系列"：
- **固定**一个**风格前缀**（如 `flat vector illustration, clean simple minimal, soft pastel, centered`）+ 每张只在主题词上变化；
- **种子基**：同一系列用**一组连续/固定种子**（或 `seed = base + i`），避免风格漂移；
- **批量**：一次提交多个 workflow（或循环 submit→poll→save），失败重跑；用 `comfy_generate.mjs` 或循环脚本。
- **画风统一度**看预览是否一致；不统一就调风格前缀/seed。

### 经典 / 知名角色：可能不需要 LoRA
对于**经典、传播广的角色**（如原神人物、名配角），模型通常**已记住它们**，往往**不用 LoRA**，直接写提示词即可。做法：
1. **先搜官方设定/提示词**：去网上（wiki / 萌娘百科 / Character Prompt 站点）查该角色的**准确外观描述**（发色发型、瞳色瞳形、服装、饰品、体型）。
2. 若没有现成提示词：**找几张该角色的人物图**，**看图自己写外观描述词**（逐项：发色/发饰/瞳色/服装配色与纹样/配饰）。
3. 用 **`角色英文名 + 外观描述`** 作为人物提示词（Anima 用自然语言、Illustrious 用 tag），如
   `Nahida, short white hair with green accent braid, big green eyes with leaf-shaped pupils, pointed elf ears, white dress with green and gold details, white bloomers, mostly bare legs, white stirrup shoes`。
- **别瞎猜/硬编码服饰细节**（我在某次就在角色服装上写错过"黑裙黑袜"，应先查设定再写——这直接决定像不像）。
- **LoRA 留作保底/强化特定画风**：先试纯提示词，效果不符或要统一画风再上 LoRA。

## Production pipeline — the macro order (先画什么 → 什么时候编辑 → 什么时候放大)

**一张图从立项到交付的 6 步，唯一出处：`references/production_pipeline.md`**（单人也适用；
原先只讲双人的 `duo_and_multi_character.md` 已并入其中）。摘录：

1. **抽卡定底图** —— **Anima，小尺寸 `800×1200` 挡位，跑 3–5 张**，挑结构对的（人数/关系/姿势/机位/遮挡）。
   结构不对就继续抽，**不要在错的画面上打磨**。多人：画布要宽（双人 `1216×832`）、每人一块外观词、
   负面加 `1boy, male`；**LoRA 绝不跨基底**。
2. **大尺度结构性修正** —— 胳膊腿位置、朝向、谁在哪儿：**人工编辑 + 重绘**（`SetLatentNoiseMask` 0.5–0.6）。
   手改稿就是画布，改动要写进提示词。**这一步的输出是后面所有步骤的底。**
   多人分区重绘的 4 条纪律（**只挂当前那人的 LoRA**、对方简化成 `another girl …`、不跨基底、同引擎 pass 连跑）
   与"**提示词只写画布上还看得见的东西**"（crop 型 vs 全画布 mask 型）：见该文档第 2 步。
3. **放大到 `1200×1800` 挡位 → 整图 `denoise 0.3`** 丰富细节（⚠ 整图路线蒙版外非逐位不变）。
4. **模型分流** —— 要 **IL 的质感** ⇒ 提边线（canny/softedge）+ **IL img2img `0.8` + ControlNet**；
   有 **IL 专属角色 LoRA** ⇒ **IL img2img `0.6`**。换引擎必须回贴。
5. **细节修正** —— **PS + 局部重绘**：手指、脚趾、`vibrator`、rope（一处一趟，0.2–0.4）。
6. **放大到发布尺寸** ⇒ 能结束就结束；还要瞳孔这类更细的 ⇒ **SmartCrop 工作流**。

**四条纪律**：⓪ **每一步的产物都要用 dsh_show_media 嵌进对话给用户看**（只报路径＝用户看不到；要和旧版对照就用 compare_with）；① **多张只属于第 1 步** —— 编辑阶段一次一张（小 denoise 样本间差别极小），
**只有 denoise > 0.6（比如转 IL）才多抽几张**；② **"放大"要发生不止一次**，
因为手指/脚趾/瞳孔这类东西**没有足够像素就画不好**，每次放大对应下一层细节的尺度；
③ **denoise 决定"读"还是"重写"**：0.2–0.4 材质 / 0.5–0.6 比例与换引擎 / 0.8–0.9 结构拓扑
（绳"连到床框"那种拓扑 0.4–0.75 出不来，0.9 才出来）/ 1.0 重抽；换材质颜色靠提示词，不靠加 denoise。


## Writing the prompt — structure, negation, costume swaps
结构（分组、空行、否定、换装）、**简洁（一个意图一处说，§2b：长度=稀释）**、Anima 专属调整、A/B 实测、536 条语料统计，全部正文在
**`references/prompt_writing.md`**；可直接抄的正向前缀、四套负面模板、8 条骨架模板与词表在
**`references/prompt_cheatsheet.md`**（写前先开；固定缺陷块见 §2、服装词见 §3、逐姿势 tag 表见 §4）。
**提交前必做 tag review（硬步骤）**：先写"场景清单"再逐 tag 对照查互斥与残留（来路非本场景从头推，或
服装·姿势·道具·地点·视角·画幅任一变化 ⇒ 必做）；只通读不算做过（**token ≠ 意识到 token 矛盾**）。
全表与步骤：**`references/tag_review.md`**。

## Blender-driven control maps — depth / pose / hands
When pose, framing, focal length or foreshortening must be exact, author the scene in Blender and
render control maps instead of describing it in words；depth 与 pose 走**同一相机、同一骨架**，可以叠。

**入口四步**：`blender_build_scene.py`（必须 `--factory-startup`，出 depth + clay + kps）→
`blender_draw_openpose.py` → `blender_overlay_check.py`／`blender_crop_hands.py`（**先验证骨架压在
几何上再花 GPU**）→ `comfy_control.mjs`。另有四条二次入口：抄现成骨架、多角色摆台、人在 GUI 里改完
读回、局部重绘（见下）。

**完整流程、scenes.json 格式、坑与全部实测数字（denoise 分带表、手/脸修复边界、FaceID 三个权重文件、
同视角才可比的 87% 现象、6 GB 上的耗时）：`references/blender_control_pipeline.md`。**

## Cutting the background out (matting / alpha)
按**输入类型**选路线，难度差一个数量级：自家渲染且提示词写了 `plain white background` →
`matte_flat_bg.py`；别人的插画/海报 → `matte_from_scribbles.py`（要人给 FG/BG 涂鸦）或 `SAM3_Detect`；
纯色幕布 → `chroma_key.py`。完整选路表、实测数字与评分方式：**`references/matting_and_cutout.md`**。

## Upscaling (2x / 4x) — after generation, before delivery
链路：**`UpscaleModelLoader` → `ImageUpscaleWithModel` → `PreviewImage`**（默认进 `temp\`；要留就显式转存到工作区），不过扩散、秒级完成。
默认 `4x-AnimeSharp` + Lanczos 收到目标宽度；交付尺寸与清晰度证明见
**`references/upscaling.md`**（含模型下载地址、四种放大器的实测对比与"2x 模型已移除"的决定）。

## Comic page — tiling N panels into one page
交付物是**页**而不是单图时（N 格、直/斜分格线、格内不留白边、阅读顺序）：`libs/comic-layout/scripts/comic_layout.py`
做分格搜索与渲染，`libs/comic-layout/scripts/letter_page.py` 做气泡、文字框搜索与锚点求解。设计要点（递归一刀切一个凸
胞，任意角度都成立、并集恰好是整页）与全部参数：**`references/comic_page_layout.md`**。

## Cloud APIs — generation and editing (and how to merge the answer back)
本地 ComfyUI 是默认；只有在本地栈**结构性不行**的地方才上云：图内文字（中日文书法/招牌）、
在手改过的底图上加物件、场景级清理。选哪家、key 从哪来、参考图怎么给、答案怎么按掩版贴回本地，
全部见 **`references/cloud_edit_and_merge.md`**（含"参考图必给"与逐字执行指令这两条用户强调的规矩）。

## Human in the loop — `human_canvas` (rough 2D) and `ps_edit` (real Photoshop)

The human owns aesthetics and intent; the agent owns measurement, mechanics and verification. Both
channels are **non-blocking**: open the task, say exactly what to do, **yield the turn**, read back
when they reply. Use the lighter channel — canvas for anything you can refine programmatically,
Photoshop only when the job genuinely needs it.

- **Ask for two layers, not one:** `L1 = repaint`, `L2 = keep`. The keep layer is the human's
  protection switch (fingers, a hair ornament, an already-good region); mask = `L1 − L2`.
- **Paint on a zoomed crop** (2–3×) when the target is small — the same stroke lands 2–3× more
  precisely in original coordinates.
- **Look at the mask yourself before spending GPU time** (tint it over the picture, report area %
  and bbox). That catches "they painted the whole head, not the eye" in one round.
- **`ps_edit` is verified by content, not by the clock:** `open` returns mtime + sha256, the human
  edits and saves **the same path**, `check --since_sha256` decides. **Archive the human's
  hand-edited file** — and its difference against the pre-edit base is the *exact* mask
  (`local_repaint_bigcanvas.md` §"Get the shaped mask for free"). Never rebuild a hand-edited base
  from a pre-edit version.

Playbook, the canvas read-back schema, the PS handoff steps and the division of labour:
`references/human_in_the_loop.md`.

## Locating things — grids, ASCII maps, compare sheets, colour masks

"Where exactly is it?" is a **measurement**. Guessing a bbox wastes a GPU run at best and eats a
neighbouring part (a finger, a hair strand) at worst. All of these are CPU-only and take seconds:

| Tool | Gives you |
|---|---|
| `crop_view.py IMG out.png x0,y0,x1,y1 --scale 1.5 --grid 100` | a crop with a grid **labelled in original coordinates** — hand it to a human so "around x 880, y 810" is actionable |
| `ascii_map.py IMG x0,y0,x1,y1 --step 4` | the region as text (R = target, S = skin, `.` = background) — shows which pixels are the rope vs. the fingers without opening a viewer |
| `sheet3.py out.png x0,y0,x1,y1 --labels a,b,c IMG...` | N versions of the **same region at the same scale** in one sheet — the only honest way to choose between denoise 0.3 and 0.5 |
| `red_mask.py IMG mask.png --hue-max 16 --sat-min .6 --drop …` | a colour mask by **HSV hue** + area/position filtering |
| `diff_report.py A B --mask M` | in-mask vs out-mask change — the number that proves an edit is confined |

Two traps with measured consequences: **`R − max(G,B)` is not a redness test** (a warm brown floor
and a dark-red cushion both pass it — 3.86 % of the frame "red"); and **"dark = ink" fails on a
night scene** (the paper's own median luminance is 50). Use HSV hue with saturation/value floors,
then filter connected components by area and position. Full detail:
`references/grid_localization.md`.

## Checking a render — 人/手/脚/指趾/口塞/掩版 自检（anime_detection_suite）

**已训好、可直接调用**。这套检测工具**不复制**进本 skill：脚本按绝对路径调用套件本体（代码与权重都在那里，
套件一更新这里立刻生效）。它回答：**人/手/脚框找到了吗、指趾数可疑吗、有没有口塞/眼罩、这张图有没有绳子掩版**。

```bash
# 任意 cwd；单图或目录都行
python "<你的用户目录>\.dsh\skills\comfyui-image-generation\scripts\check_parts.py" ^
  --src <图或目录> --out <输出目录> [--mask] [--person-masks] [--recipe rec] [--json]
```

| 参数 | 说明 |
|---|---|
| `--src` | 图片文件或目录（可多个） |
| `--out` | 输出目录（默认 `./part_check_out`）：套件原始契约 JSON + `summary.json` |
| `--recipe prod\|rec` | `prod`（默认）= 现产线口径（imgsz 1280 + 现役 M3 双权重）；`rec` = 实测更好（imgsz 1600 + 无泄漏 M3 三种子，**尚未设为产线默认**） |
| `--imgsz` / `--m3-conf` / `--foot-crop` / `--hand-weights` / `--m3-weights` / `--no-foot-tight` | 透传给套件入口（`locate.py --help` 有一手说明） |
| `--mask` | 手/脚掩版 PNG（套件行为） |
| **`--person-masks`** | **逐人 silhouette**（多角色"逐人重绘"用；见下） |
| `--person-mask-model` | `sam`（默认，本 skill 自带 SAM ViT-B）/ `hull`（只用 M1 关键点凸包，不需要 SAM） |
| `--person-mask-prompt` | `pose`（默认）= 骨架外接框 + 躯干正样本点；`box` = 只用检测器 person 框 |
| `--person-mask-cover` | `auto`（默认，**宁可多盖**）/ `tight`（贴边、不碰道具）/ `full`（保证整条骨架在内） |
| `--grow N` / `--no-fill-holes` / `--no-clip-to-pose` / `--no-merge-overlap` / `--person-min-prob` | 掩版的四个后期开关（见下面的经验值） |
| `--ignore-cls` | 汇总时忽略这些类，例如角色设定本来就戴的眼罩：`--ignore-cls blindfold` |
| `--json` | 只打印一行机器可读 JSON（agent 解析用） |

**怎么读结果**：`可疑部位` = 需要**人复核**的部位，不是"渲染失败"；`收紧复核` 是脚趾的加严旗标（低优先级）；
`看不到手 / 无人` 是构图提示；`数不了N处` = 穿鞋穿袜（指趾数不了，正常）。
`summary.json` 逐图给出：`suspicious / tight_flags / gag_blindfold(含 box+prob) / n_hands / n_feet /
hands_state(visible|not_visible|zero_detected) / rope / persons / persons_masks / seconds*`。

### 逐人掩版（多角色属性串扰的正解）
同一张图里"谁被绑/谁自由、谁黑发/谁白衣"这类**属性独占**需求，提示词压不住；结构性解法是逐人
掩版重绘（一人一次 pass，另一人像素冻结）。一条命令出掩版：
`python scripts/check_parts.py --src BASE.png --out OUT --person-masks --json`
产物在 `OUT/mask/`：`<name>_person0.png`（255=重绘区）与 `<name>_person0_rgba.png`
（**重绘区 alpha=0**，因为 ComfyUI 的 `LoadImage` MASK = 1 − alpha）。掩版怎么定（重叠自动切分、
头并进来、`person_*` 字段含义）与逐人 pass 的完整做法：**`references/local_repaint_bigcanvas.md`**。

### 局部特征重绘 + 打底（underpaint）
① `check_parts.py --feature-masks` 直接给手/脚/头/脸/口塞/眼罩/眼 的**框掩版**，外加
`mouth*` = 脸框下 42%（横向内缩 12%）——**COCO-17 根本没有嘴点**，重绘嘴只能用脸框推。
（M1 换成微调 ViTPose 后鼻/眼/耳已经可信：报告集面部中位误差 15.9px；但**背面/侧脸视图下脸点仍会飘**，
所以"嘴/脸这类细部位一律用脸框推、不用关键点"这条规矩不变。）
② `mask_compose.py` 是 agent-in-the-loop 的掩版组合器（图元 ∪/∖ 已有掩版），不必一次给对。
③ 打底（underpaint）三条路与掩版规矩见 **`references/underpaint_and_stencil.md`**。

### 两个"模型不认识的新角色"同框、且不许反左右（标准解）
1. 先出**不挂任何角色 LoRA 的版式底图**，方向一律写**画面坐标系**，负面点名 `mirrored composition`、
`swapped positions`；2. `check_parts.py --person-masks` 拿每人一块互不重叠的掩版；3. **每次只挂一个人
的 LoRA**（1.0/1.0 + sidecar 完整触发词表）`denoise 0.65~0.75`，串行两次；4. 每人的重绘提示词里都
再写一遍接触关系。完整四步与证据：**`references/prompt_writing.md`**。

## Finding a LoRA and its trigger words
**检索与下载**（civitai 用 `?query=` 而不是 `?search=`、shell 直连 403、十六进制文件名、base 匹配、
校验与放置）见 **`references/lora_sourcing.md`**；下面只讲"拿到之后怎么读它的触发词"。
- **Keep `strength_model` and `strength_clip` at 1.0.** Interpolability in
  latent space is *empirical, not a strict property*: a LoRA is a trained perturbation, and
  scaling its strengths does not guarantee a smooth blend — it can quietly degrade the identity,
  costume or rendering while a thumbnail still looks "close enough". Only deviate when the LoRA
  **demonstrably damages the picture** (burned colour, broken anatomy, style collapse), and even
  then: fix the prompt first (a conflicting tag can be what the LoRA is fighting), then try another
  LoRA, and lower strength last. `[measured]` the same character LoRA at 0.85/0.8 *plus* an
  incomplete trigger list produced the wrong costume/hair/eyes; at **1.0/1.0 with the sidecar's
  full word list** it matched.
- **Use the sidecar's *whole* trigger list, not the few words you remember.** A missing descriptor
  is a wrong picture, and it is indistinguishable from "the LoRA is weak": the same session's
  first pass omitted `pink pupils`, `forehead mark`, `sleep mask on head`, `purple dress` from the
  LoRA card and the eyes/costume came out off-model; the model then fights your substitute words
  (`white dress` vs the LoRA's `purple dress`) instead of following them.
- **List available LoRAs**: `GET /object_info/LoraLoader` → `required.lora_name[0]` is the enum (paths relative to `models/loras`). Exact `lora_name` must match, e.g. `character\E9B2B8E9B1BCE5A898Fix.wTK8.safetensors`.
- **Card / title / base model**: the LoRA-Manager sidecar `<name>.metadata.json` next to the `.safetensors` holds `model_name`, `base_model`, `preview_url` (this is what LoRA-Manager shows).
- **Trigger words / tags**: read the safetensors header metadata — first 8 bytes are the **little-endian uint64** JSON-header length; parse that JSON, then `__metadata__`:
  - `ss_tag_frequency` → the training tag vocabulary (build the prompt from these).
  - `ss_sd_model_name` / `ss_base_model_version` → which base model it needs.
  - `ss_vae_name` → the VAE it expects.
  - `ss_network_module` (e.g. `networks.lora_anima`) → whether it needs a special pipeline.
  - `modelspec.title` → display name.
  - **`ss_tag_frequency` CAN be empty** — many Civitai character LoRAs keep the trigger elsewhere. Check the LoRA-Manager sidecar `<name>.metadata.json` → **`civitai.trainedWords`** (and `civitai.model.trainedWords`); that holds the real trigger + descriptor tags (e.g. `hu_tao_genshin_impact, brown hair, hat, long hair, ...`).
  - **Best source of all: `civitai.images[].meta.prompt`** — the sidecar stores the **generation prompt of each preview image**, i.e. the author's own tag table for that character, and it is more complete than `trainedWords`. A character's chest badge was absent from *both* `trainedWords` variants and showed up only there, as `lanyard`. One line to read it:
    `for im in (m.get("civitai") or {}).get("images", []): print(im["meta"].get("prompt"))`
  - **Finding the word is not the same as being able to suppress it.** Negating that badge (`lanyard, id card, name tag, badge, nameplate, tag, neck strap, cord around neck, hanging ornament`) did **not** remove it — **character priors outrank the negative prompt**. Changing the outfit did remove it. If a negative produces no visible change, the element is baked into the LoRA's default look: change something else (costume, pose, crop) or edit the pixels.
  - **Author-recommended sampler/steps/CFG** often live in the same sidecar's `civitai.model.description` (e.g. "Steps 30 / CFG 5-7 / DPM++ 2M SDE Karras"). Prefer those over the generic defaults below.

  **Match the LoRA's base pipeline**: a LoRA trained on Anima (`anima`) goes in the Anima workflow; one trained on Illustrious/SD goes in the IL workflow. Applying a LoRA to the wrong base yields poor output.

## Troubleshooting
ComfyUI 提交/加载、Blender 退出码、LoRA 串味、放大与云 API 的常见故障与排查顺序，见
**`references/troubleshooting.md`**（每条都带"看到什么 → 查哪里"）。红线相关的两条留在上面
「Safety gate」与「Cleaning up after a run」里，不在那份文档内。
