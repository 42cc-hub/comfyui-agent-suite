---
name: comfyui-image-generation
description: Generate images through a local ComfyUI server via its HTTP API, including the Illustrious (SDXL anime) and Anima (qwen-image) model pipelines, LoRA selection/trigger-word lookup, submit/poll/save automation, Blender-authored ControlNet/LLLite control maps (depth, OpenPose pose with hands, img2img), cutting the background out of renders or existing illustrations (local SAM matting, chroma key, or a hosted image-edit model) with measured scoring against a ground truth, and self-checking a finished render with a trained detector suite (people/hands/feet boxes, finger-and-toe count suspicion, gag/blindfold, rope mask) via `scripts/check_parts.py`. Use when rendering, iterating on, or cutting out images instead of using a hosted image API, or when a render needs its hands/feet/digits verified before delivery.
allowed-tools: Read Write Edit Bash
license: MIT
compatibility: Requires a running ComfyUI server (<你的 ComfyUI 安装目录>, conda env sdenv, port 7187, NVIDIA GPU). Uses `node` (Node 18+) for the API calls; Blender 5.1 at `E:\\Program File\\blender` for the control-map pipeline; `ffmpeg` optional only if audio->image conversion is needed elsewhere. Matting additionally needs `pip install segment-anything` (small pure-Python package) and the SAM ViT-B checkpoint that ships with this skill at `models/sams/sam_vit_b_01ec64.pth`.
metadata:
  version: "1.11"
  skill-author: local (machine-specific operational notes)
---

# ComfyUI Image Generation

Drive a **local ComfyUI** instance over its HTTP API to generate images. This skill covers the
request/response flow, the two model pipelines on this machine — **Illustrious (IL, SDXL anime)**
and **Anima (qwen-image)** — and everything that happens *after* a picture exists: ControlNet /
Blender-authored control maps, masked local repaint (smart crop, local and cloud-assisted),
upscaling and final-size reduction, matting, comic-page tiling, cloud API editing, human-in-the-loop
masking/Photoshop handoff, image localization/measurement, and the safety gate that decides whether
a picture may enter the agent's context at all.

## When to use
- You want to generate images on the local GPU (not a hosted API) from text prompts.
- You need to apply a LoRA to a base model and want its activation/trigger tags.
- You are iterating: change prompt / seed / LoRA weight / sampler and regenerate quickly.
- Pose, framing or perspective must be **exact** rather than prompt-guessed → author it in Blender and feed ControlNet/LLLite (see the Blender section).
- You need the **background removed / an alpha cutout** from a render or an existing illustration → see the matting section; difficulty differs by an order of magnitude depending on whether the image is ours (flat generated background) or someone else's (poster).
- You need to **fix a detail only a few pixels wide** on a finished image (a shoelace, an eyelet) without disturbing the rest → SmartImageCrop + `SetLatentNoiseMask` + SmartImageStitcher; see the Upscaling section and `references/local_repaint_bigcanvas.md`.
- The deliverable is a **comic page** — N images tiled into one image with dividers and lettering, in reading order → see the Comic page section and `references/comic_page_layout.md`.
- The local stack is structurally weak at the task — **text inside the image** (Chinese calligraphy, signage), adding objects on top of a hand-edited base → call a **cloud** model and paste the answer back by mask: see the Cloud section and `references/cloud_edit_and_merge.md`.
- **A human must supply rough 2D input** (a painted mask, a few points/edges) or do real Photoshop work on this host → see the Human-in-the-loop section and `references/human_in_the_loop.md`. **When handing a file to Photoshop (`ps_edit`): ONE file per round — opening a second file while one is already open FREEZES Photoshop on this host (measured 2026-10-06; the human had to kill the process and unsaved work was lost).**
- You must **know exactly where something is** before masking it (which pixels are the rope, where the face sits, which of two strengths is better) → see the Locating-things section and `references/grid_localization.md`. **要给某块区域做掩版时，先在四条路里选一条** —— ① 检测框（本机、免费、最常用）② 视觉模型分色（云端 ~10 秒，语义身份最清楚）③ 网格+坐标估计 ④ 画布让人涂（最稳但要人）：取舍、实测数字与命令见 **`references/localization_and_masks.md`**。
- You need a **body-part mask without asking a human to paint one** → the local OpenPose node + `pose_keypoints.py` give per-part boxes (feet/legs/face/arms/hands/torso) and can write the mask: see the auto-localization section of `references/local_repaint_bigcanvas.md`. Reliable on standing/kneeling; **extreme poses (all fours, suspension) lose joints** → ask for a rough stroke or use `SAM3_Detect`.
- **Before any image enters your own context** — every vision read uploads the picture to a model provider → run the local WD14 gate first: see the Safety-gate section and `references/output_safety_gate.md`.
- A figure is **complex** (two subjects in contact, a container/rig, a crowd, an unusual camera, a foreground foot/face) and you must decide **what to sample, what to program, what can be added later** → see the Planning section and `references/planning_and_decomposition.md`.
- You are **starting a picture (or a set) and need the macro order** — when to sample, when to let a human edit, when to upscale, when to switch models → see the **Production pipeline** section and `references/production_pipeline.md`. It also carries the two-count disciplines: **many samples only at step 1**, and **upscaling happens more than once** because fine objects need pixels.
- The picture has **two or more characters** and you need them to stay distinct (who is who, what each is doing, no feature bleed) → step 1 and step 4 of `references/production_pipeline.md` (wide canvas, per-character blocks, region repaint with only that character's LoRA, mask-paste when the engine changes).
- You have a **reference illustration of an interior** and need *controlled* depth/line maps, or that image's **camera** (to place a character at the right scale), or you are compositing a character into a generated background → see the Perspective section and `references/perspective_and_scene_geometry.md`.
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
| Blender | `E:\Program File\blender\blender.exe` (5.1, headless needs `--factory-startup`) |
| SAM (matting) | `models/sams/sam_vit_b_01ec64.pth` inside this skill + `pip install segment-anything` |
| Upscalers | `models/upscale_models/`: `4x-AnimeSharp.pth` (**default**, best for illustration), `RealESRGAN_x4plus_anime_6B.pth`, `4xhfa2kludvaeswinir_v1.safetensors`. **`2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` was REMOVED (2026-09-25, user decision)** — it is visibly softer than the 4x models; it now sits in `<你的 ComfyUI 安装目录>\agent_work\_disabled_upscalers\` (out of ComfyUI's model list). Measured on a 2000×1368 illustration: |Laplacian| mean 6.08 / P99 86 (2x-AnimeJaNai) vs **9.56 / 160** (4x-AnimeSharp then Lanczos half). **Deliver with `4x-AnimeSharp → ImageUpscaleWithModel → Lanczos to the target width`**, never a native 2x model. |
| Outbound network | **The shell has no direct internet.** HTTP times out; HTTPS via curl/.NET dies with `schannel: AcquireCredentialsHandle failed`. Only working path: local proxy `127.0.0.1:6132` + **Python (OpenSSL)** → scripts/fetch_url.py |
| **NEVER TOUCH** | **`<你的 ComfyUI 安装目录>\output\`** — a **mixed** directory: the user's own files live there next to renders (notes, downloaded reference pngs, `output\漫画委托\` = the whole commissioned-project material). One date-based sweep by this agent destroyed 17 of the user's files (2026-09-25, unrecoverable: `os.remove` skips the Recycle Bin). **Never delete, move, or reorganize anything under `output/`.** |

## Cleaning up after a run (hard rules — 2026-09-25)

**Workspace boundary (user rule, 2026-09-25): work inside `<你的 ComfyUI 安装目录>\agent_work\` only.**
Everything outside it (`output\`, `input\`, `models\`, the ComfyUI tree, the Desktop/Downloads) is the user's
territory: **never delete, move, rename or reorganize anything there unless the user spells it out for that
exact file.** The agent's products live under `agent_work\` precisely so human-made and machine-made files do
not get mixed — mixing is what makes an agent mistake a user's note for its own leftover.

Cleaning is the one operation that **cannot be undone** here: `os.remove`/`Remove-Item` bypass the Recycle Bin,
`vssadmin` needs admin rights we do not have, so there is **no recovery path**.
If a name is lost, one partial rescue exists: the session transcript under
`~/.dsh/sessions/<project>/<session>/session.v3.jsonl.zstd` (`zstd -d`, then JSON-decode each line) still
holds the real filenames — .

1. **List first, delete second.** Print every path + size + mtime before removing anything — even when the
   request is a casual "clean up the intermediate files".
2. **Whitelist by filename.** Never delete by "directory + age" or "directory + size": coarse conditions
   sweep user files (that is exactly how the accident above happened).
3. **Mixed directories are off-limits**: `<你的 ComfyUI 安装目录>\output\`, `<你的 ComfyUI 安装目录>\input\` (except the
   staging subdir below), Desktop, Downloads. Touch them only file-by-file, by explicit name.
4. **Prove "it is a duplicate" before deleting it**: sha256, or a pixel diff. (Example: `puffy_binding/out14/**`
   vs `final/**` diff = **0.0** ⇒ true copies, safe; identical diff to two different candidates ⇒ they are copies.)
5. **`input/dsh_edit/` is the one safe-to-empty directory** — it is pure staging; every script re-copies its
   inputs from the project dirs before each run.
6. **Never delete**: the user's hand-made files (PS handoffs, hand-painted mask layers), authored sources,
   `final/` dirs, exported pages, or any file whose name looks like a note / script / reference.

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

> **A `BLOCK` image gets a path in the reply, nothing else.** Not `read_image`, and **not
> `dsh_show_media`** either — that tool's result *does* re-enter the agent's context (measured
> 2026-09-24), so it is not a safe review channel. Text path only; the human opens it locally.
>
> **两次放宽（用户同意，2026-09-24 与 2026-10-07）：硬边界是关键点，既不是 rating，也不是姿势。**
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
**Censoring a BLOCK image to look at it** `[measured]`: mosaic the **whole panel** (16–64 px cells;
32–64 often reaches `OK`) — *not* the one suspicious spot, and never a flat colour fill (a flat
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

Full playbook, measured numbers and the toolchain: **`references/perspective_and_scene_geometry.md`**.
The shape of it, in the order that matters:

- **Agree the axes first.** image: x right / y down / z into the screen (`u = cx + f·X/Z`); world:
  x forward / y left / z up (ROS); `p_cam = R(p_world − C)`; families `x`=depth, `y`=across,
  `z`=vertical. Parameterise the rotation as a **rotation vector on a fixed baseline `R0`**, never
  as free Euler angles (they let an optimiser report `pitch = −424°` and call it a fit).
- **The human labels each line with its family — that is what makes a hard constraint safe.**
  Human lines: plain L2, weight 1. Machine-detected lines: Cauchy via IRLS, dropped above ~5–12°.
  A Cauchy loss on *human* lines lets the optimiser abandon one for free (seen: a 220 px labelled
  line parked at a 37° residual at cost ≈ 0); and priors must weigh like a line, or f drifts to
  the bound to buy a marginally better line fit (seen: f = 285 px, h-fov 122°).
- **Residual** = angle between the line and the direction from its midpoint to that family's
  projected VP. Guard the **VP-at-infinity** case (direction ⊥ optical axis) explicitly, or the
  division by ~0 poisons the whole solve.
- **Report observability, not just a camera**: rank + null direction of the Jacobian, and a **focal
  sweep**. Measured: a single **wall corner** (3 lines, one per family) fits *exactly* at every
  focal from 30° to 90° — f completely unobservable, only pitch (≈ +28°) is; **one extra line in any
  family** pins that VP and makes f observable; two near-parallel lines of one family add almost
  nothing (their VP is at infinity), so ask for lines that fan out.
- **Assume the drawing is not a projection, and prove it with numbers.** In the session's
  illustration the case's shelf-edge families imply a 48° down-pitch while the fixture verticals
  imply ≈ 0°; with 40 detected lines the robust fit still pinned roll/pitch at the parameter bounds
  with a 12° median residual. When the input is squishy, deliver **largest consistent subset +
  outlier list + a per-line residual overlay** (green/yellow/red), never a forced camera.
- **Build the scene as a wall corner** (floor `z=0`, one wall `{y=Y_L}` parallel to the aisle, one
  `{x=X_R}` perpendicular). Take each wall's offset from the **bottom boundary of the human's paint
  mask** (ray → floor, median of the world coordinate): 796 samples gave `Y_L=1.94 m`; the occluded
  wall's number is biased far (12.6 m) and is **the knob**. Lines *above* the floor are not base
  lines — assuming they lie on `z=0` gave a nonsense 45 m.
- **Repaint the depth per painted region** from each plane's equation (and fill the character's
  region with the background behind her); also emit a plane-id colour map — humans judge that far
  better than grey.
- **Pair the control model with its family.** Anima has **no depth ControlNet** (only LLLite
  pose/scribble/inpaint; the depth `.safetensors` in that dropdown is SDXL/NoobAI). Depth →
  **Illustrious + `noob_sdxl_controlnet_depth.safetensors`** (832×1216, strength ≈0.85, 111 s,
  visibly better than Anima+scribble here). Control images must match the sampling canvas; stage
  them in `input/dsh_ctrl/` and pass `dsh_ctrl/<file>` (a bare filename = HTTP 400); `LoadImage`'s
  MASK is output index 1 and equals **1 − alpha**.
- **Two-stage compositing must pin ONE scale anchor.** The character's pixel size comes from the
  human's mask, the background's scale from your geometry — nothing ties them, and the session got
  a **2.30× giant**. Fix: `d` = ray through her feet ∩ floor; `h_expect = f·H_char/d`;
  `factor = h_expect/h_mask`; zoom the background about her feet (default) or shrink the mask, and
  **print the factor**.
- **Finisher**: one whole-image img2img at **denoise 0.6, no ControlNet** (30 s) removed the
  two-stage seam and unified the lighting. Run on our composite it gave the best image *of our own
  pipeline*; run on the reference it gave the best-looking image of the session but that is the
  **ceiling, not a reconstruction** — at 0.6 the reference's own layout and props survive.

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

```powershell
# clear any stale db lock, then launch (needs danger-full-access for G:\ and AppData writes)
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
 or this one-liner (single quotes inside, so it
survives PowerShell):

```bash
python -c "import json,urllib.request as u;[print(p, u.urlopen(u.Request('http://127.0.0.1:7187'+p, data=json.dumps(d).encode(), headers={'Content-Type':'application/json'}), timeout=20).status) for p,d in (('/interrupt',{}),('/queue',{'clear':True}))]"
```

**Hard limits on a 6 GB card** (from the 2026-09-24 incident; they are not negotiable):
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
- **SaveImage** `filename_prefix`, `images=vae_decode[0]`.

### Image sizes & hi-res for video
- Landscape/anime usually `1024x1024`. For a **vertical (9:16) video** use portrait SDXL ratios like `832x1216` or `896x1152`; `1080x1920` itself is valid but slow on 6 GB.
- A 832-wide source upscaled to `1080x1920` looks soft. Generate higher (e.g. `1216x1792`) or add a **hi-res fix** (a second low-denoise KSampler pass / hires-fix / upscale LoRA) for crisper video output. Keep the subject framed with headroom so a Ken-Burns crop doesn't clip it.
- **Size the canvas to the character, not to the aspect ratio** — the figure needs ≥ `700×1200` px
  or the face goes off-model. See "Subject pixel area" below before picking a resolution.

### Sampling resolution decides the *failure mode*, not just the framing

Measured 2026-09-21 (Illustrious, no LoRA, `cfg 8` / 20 steps, paired seeds, hands counted by a
human; full data in our own measurements):

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

A second, independent rule (user-provided, 2026-09-23). It is about **how many pixels the character
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
  A 2026-10-03 note here blamed the stitcher for 4.31 % out-of-mask drift; **that attribution was wrong and is
  retracted** — measured 2026-10-04 (cases P1–P7): the drift comes from `SmartImageCrop` resizing the crop
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
- **SaveImage** `filename_prefix`, `images=vae_decode[0]`.

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

**四条纪律**：⓪ **每一步的产物都要用 dsh_show_media 嵌进对话给用户看**（只报路径＝用户看不到；要和旧版对照就用 compare_with）—— 用户 2026-10-03 反复要求过；① **多张只属于第 1 步** —— 编辑阶段一次一张（小 denoise 样本间差别极小），
**只有 denoise > 0.6（比如转 IL）才多抽几张**；② **"放大"要发生不止一次**，
因为手指/脚趾/瞳孔这类东西**没有足够像素就画不好**，每次放大对应下一层细节的尺度；
③ **denoise 决定"读"还是"重写"**：0.2–0.4 材质 / 0.5–0.6 比例与换引擎 / 0.8–0.9 结构拓扑
（绳"连到床框"那种拓扑 0.4–0.75 出不来，0.9 才出来）/ 1.0 重抽；换材质颜色靠提示词，不靠加 denoise。


## Writing the prompt — structure, negation, costume swaps

Rules distilled from the local `comfyui_PromptManage` plugin
(`custom_nodes\comfyui_PromptManage`: `README.md` §LLM 提示词生成规则,
`web/llm-templates.json`, `prompt_example/LLM_analyze_prompt.md`), Anima-specific
adjustments, hard-won A/B measurements, **and a 536-prompt corpus harvested from
`H:\FILE\temple` (2026-09-28/29)**.

**写提示词时先开速查表:`references/prompt_cheatsheet.md`** —— 可抄的正向前缀（IL/Anima）、
四套负面模板、8 条骨架模板（单人/双人 `{}` 块/容器内/悬吊/Anima）、词表（绳索按部位、
口球、腿几何、状态、镜头）、if→then 规则表、参数表。**展开的规则与证据**
（每条数字的来源、A/B 实测）：`references/prompt_writing.md`。语料统计复现：
`experiments/prompt_organize/analyze_corpus.py` + `analyze_extra.py`
。

**硬性要求：一级用空行（`\n\n`）分区块，二级用单换行（`\n`）分要素。** JSON 里写 `\n\n` 就是
ComfyUI 界面里的空行。理由：提示词是给人**手动改**的，人的注意力有限——2026-09-21 交付过一版把
六组压成一行的提示词，用户当场要求返工（"人的注意力是有限的，处理不了那么多黏着的滔滔不绝的
文本"）。语料实测：**85% 的提示词在区块内部用单换行，中位 9 行 / 4 个空行区块**，所以正确粒度是
**行**——姿势一行、缚法一行、表情一行、服装一行；只有 4% 是一条到底。

**分组顺序（经验值，来自 536 条实际提示词的平均位置）：**

```
风格/质量块  →  场景/镜头  →  主体+姿势+缚法+表情  →  角色身份+外观+服装  →（可选）风格块
0.06–0.17      0.35–0.45      0.49–0.61               0.71–0.75
```

**角色身份块放最后三分之一**（284 条带 `名字 \(系列\)` 的提示词里平均位置 0.71，69% 落在后 1/3）。
好处是**换角色只改一块**，不动场景和缚法。位置本身不敏感，但**同一系列内保持一致**。
`[plugin]` 规则 4 的"质量 → 主体 → 服装 → …"是另一种排布，同样可用。

**风格块是固定的、可整段复制的**：语料里 **57% 的提示词开头 18 个 tag 逐字相同**，
即 `masterpiece, best quality, absurdres, ultra detailed, high resolution, newest, highly
aesthetic, intricate details, professional artwork, trending on ArtStation, digital illustration,
painterly style, soft gradient shading, professional digital painting, extreme details`
（Anima 系换成风格 LoRA 的**触发整句** + `score_9/8/7` + `year 2025`）。
`masterpiece, best quality, absurdres` 本身不是风格——风格在那串尾巴上。
**风格块只写一次**；重要特征给 2–4 个同义强化词。

**IL vs Anima — the one real difference:**

| content | Illustrious (SDXL) | Anima (qwen-image) |
|---|---|---|
| scene / mood / lighting / event | tag phrases | **a sentence** (clauses, semicolons OK) |
| constituent elements (clothing, body, props, style) | tag phrases | **comma phrases** |

Mnemonic: **"what happens" in sentences, "what is there" in phrases.**

**Never put negation in the positive prompt.** The text encoder does not bind `no`,
`without`, `nothing`, `except` to the noun that follows — `no apron` reads as *apron*.
Write the affirmative (`plain white summer dress`), and move the unwanted thing
(`apron, navy blue maid dress, black skirt`) into the **negative** prompt. Same for
quantitative constraints: `only the very edges, nothing in the middle` is read as
*edges* + *middle*, so the middle gets drawn anyway. (Corpus check: only 2% of tag-style
prompts contain `no <noun>`, so the convention agrees.)

**负向是"第二块画布"，不是只放缺陷。** 语料里负向的用途分布：材质/风格纠正 55%
（`brush, comic` 压水彩 LoRA；`flat colors, cel shaded, flat shading, hard edges, low contrast`
锁厚涂风）、角色服装残留清除 14%、裸露压制 11%、**多人压制 10%**（单人图加 `people, person, girl`）、
**姿势/器具压制 6%**（`hanging rope, suspension, ribbon, satin ribbon` —— `red rope` 的默认读法
之一就是悬吊和缎带）。原则：**"这个词会带来的不想要的读法"进负向**。
固定缺陷块也在 `references/prompt_writing.md` §2（三条可直接抄的模板）。负面同样要瘦身。

**Costume / character swaps need an explicit replacement clause.** A LoRA carries its
own outfit, and merely naming a new garment loses to it. Write
`plain white summer dress instead of a maid outfit`, **and** put
`navy blue maid dress, maid outfit, apron` in the negative — that combination took the
white dress from 0/4 to 4/4. Target garment alone in the positive: swap fails.

**短动作短语胜过几何描述 —— 但"拓扑"可以用句子。** `pull down the dress on crotch
with both hand` 一次就中；一段描述"中间那块压平、外侧两角高于胯线、大腿外侧露出一条窄边"
的段落被 8 张图全部无视。语料的修正：**绳路这种"什么连着哪里"的拓扑，句子反而更清楚**
（14% 的语料用整句：`Her upper body is tied in shibari style by large amounts of red ropes,
and her arms are tied behind her back.` ×7）。判据：**句子里出现相对量/像素量就停手，
去 `references/blender_control_pipeline.md` 或遮罩。**

**绳索要写部位。** 不要只写"绑住腿"：语料用 `bound ankles / bound knees / bound thighs /
bound legs / bound arms / bound wrists / bound feet` 逐个部位写，配 `shibari, red rope,
crotch rope, breast rope, skindentation`（绳勒进皮肤）、`shibari over clothes`（隔衣缚，
防自动脱衣）。完整词表 + 规范/非规范写法之辨：`references/prompt_writing.md` §3。

**整体缚法挑"有公认英文名的那个"** —— 14 种姿势实测（Anima，绿✅9 / 半⚠️5 / 认不出 0）：
最好使的是 `box tie / hogtie / gyaku ebi / frogtie / strappado / crotch rope / ippon tsuri /
human dog`，日文名单独写没有控制力。**绳的艺术编花（菱形 karada、龟甲 kikkou）出不来** ⇒ 要控制图或局部重绘；
`suspension` 默认读作**倒吊**，要立吊写 `single rope, bound wrists above head`；`spread eagle` 只是**摆开**、没绑四角；
双人并绑用 `bound together, back to back, ropes joined, 2girls` 一次就成。
逐姿势 tag 表与证据：`references/prompt_cheatsheet.md` §4「整体缚法姿势」。

**权重幅度跟底模走，且本管线未做 A/B。** 语料：IL 用权重时中位 **1.20**、只有 5% 到 1.5 以上；
Anima 中位 **1.50**、**51%** 在 1.5 以上（出现 `(shibari:4.0)`）。IL 上写 `:4.0` 没有依据，
会海报化。**先减词，再调权重。**

**Check the tag's real meaning before writing it** (esp. Japanese net/fashion terms —
在线百科 / 萌娘百科). One theme word was drawn literally as a prop for 12 images
before the definition was checked; the correct reading was a garment silhouette. A
wrong reading costs a whole batch.

**提示词不是"越多越好"，正负都要适度 —— 写"承重的那几个"，不是写"齐全"。**（2026-09-21，用户纠正）

- **机制**：词表越长，每个词的相对权重越低（词表长度本身就是稀释）。堆到几十个词之后，
  新增的词不再增加控制力，只是把已有的词摊薄。
- **实测（负向）**：一版把 60+ 个缺陷词塞进负向、还给鞋类词加了 `(shoes:1.45)` → 鞋子纹丝不动，
  反而因负向过载把整张图渲染成绿/红海报化；另一版只用 20 个词、鞋类只留 `shoes, high heel`，
  配合 `cfg 8 / 20 步` → 鞋直接消失。
- **实测（正向）**：同一时期把角色 + 服装 + 姿势 + 表情 + 质量词全堆成一行长句的版本，
  反而不如用户自己写的"短句 + 分组空行"版本听话（那次短版直接把鞋去掉了）。
- **做法**：每个内容组只留**承重的那几个词**；分组空行（见本文件 §Writing the prompt 的硬性要求）
  让你能**整组删掉而不伤别的组**。
- **判据**：**加一个词如果没让画面可辨地改变，它就是噪声，删掉。** 先减词，再调权重 ——
  加权是最后手段，不是第一手段。

## Blender-driven control maps — depth / pose / hands
When pose, framing, focal length or foreshortening must be exact, build the scene
in Blender and render control maps instead of describing it in words. Depth and
pose maps are projected through the **same camera from the same armature**, so
they can be stacked without contradicting each other.

**Full workflow, scene-spec format, pitfalls and measured settings:
`references/blender_control_pipeline.md`.**

```bash
# 1. scene -> depth + clay preview + keypoints (--factory-startup is REQUIRED)
"E:\Program File\blender\blender.exe" --background --factory-startup \
    --python libs/blender-controlnet/scripts/blender_build_scene.py -- --spec scenes.json --out out
# 2. keypoints -> OpenPose skeleton (COCO-18 body + 2x21 hand points)
python libs/blender-controlnet/scripts/blender_draw_openpose.py out/s1_kps.json out/s1_pose.png
# 3. verify the skeleton sits on the geometry BEFORE spending GPU time
python libs/blender-controlnet/scripts/blender_overlay_check.py out/s1_clay.png out/s1_pose.png out/s1_overlay.png
python libs/blender-controlnet/scripts/blender_crop_hands.py s1        # hands are ~20 px; zoom before judging
# 4. generate: ControlNet (SDXL) / LLLite (Anima) / img2img from the clay render
node scripts/comfy_control.mjs references/blender_control_jobs_sdxl.json

# 5. optional: INGEST an existing 2D skeleton instead of authoring a pose by hand
python libs/blender-controlnet/scripts/blender_extract_kps.py ref/skeleton.png ref/target_kps.json --overlay ref/check.png
"E:\Program File\blender\blender.exe" --background --factory-startup \
    --python libs/blender-controlnet/scripts/blender_fit_pose.py -- --target ref/target_kps.json \
    --spec scenes.json --scene s1 --out scenes_fitted.json
python libs/blender-controlnet/scripts/blender_pose_error.py ref/target_kps.json out_fitted/fitted_kps.json

# 6. multi-character tableau (+ per-character/env silhouette masks, + editable .blend)
"E:\Program File\blender\blender.exe" --background --factory-startup \
    --python libs/blender-controlnet/scripts/blender_stage_multi.py -- --spec scenes_battle.json --out out --masks --save-blend scene.blend
# 7. human drags camera/characters/props in the Blender GUI, Ctrl+S, then read it back
"E:\Program File\blender\blender.exe" --background --python libs/blender-controlnet/scripts/blender_extract_blend.py -- \
    --blend scene.blend --spec scenes_battle.json --out scenes_from_blend.json
# 8. local repaint: prior mask -> GrabCut-aligned mask -> latent noise mask
python libs/blender-controlnet/scripts/blender_make_inpaint_mask.py base.png out/s_mask_keqing.png mask_keqing.png --grow 16 --feather 6
node scripts/comfy_inpaint.mjs inpaint_jobs.json
```

Key facts:
- Poses are authored as **world-space bone directions** in `scenes.json`; hands as
  `{"L": {"curl": 0..1, "spread": 0..1}}`. Fingers are real bones (15/hand), so the
  mesh, depth map and OpenPose hand skeleton share one pose.
- **Multi-character** staging is `blender_stage_multi.py` (§6 of the reference):
  N armatures, props parented to bones, per-character keypoints, and `--masks`
  silhouettes. Hand the `.blend` to the human when hand-authored poses still read
  wrong; `blender_extract_blend.py` reads the GUI edits back (round-trip fidelity
  measured at max 27/255, mean 0.019). Mesh-level edits (subdivide/join) do NOT
  survive — only object transforms do.
- **Local repaint** beats re-rolling a whole image when one element is wrong
  (§8): Blender silhouette → GrabCut aligned on the *generated* image → `VAEEncode`
  + `SetLatentNoiseMask` + `KSampler(denoise)`. **Simplify the prompt outside the
  repaint region** (reduce the other character to `1girl at right side`) or the
  model rewrites them.
- **Choose denoise by WHAT you are changing, not by how wrong it looks.** The band
  names the task; a band that is too high throws away exactly the structure you
  meant to keep. (2026-09-20: "fix the tail's scales" was run at 0.85 and the tail
  came back detached from her hips with stray fins floating in the water — the task
  was band 1, the pass was band 4.)

  | denoise | the task it is for |
  |---|---|
  | **0.2 – 0.4** | fixing texture and linework |
  | **0.5 – 0.6** | fixing proportions and expressions |
  | **0.7** | large-scale restructuring |
  | **0.8 – 1.0** | discarding the original structure — a fresh roll |

  Two consequences that are easy to get wrong:

  - **Style and colour are prompt-driven, not denoise-driven.** Keeping the prompt
    unchanged keeps the original look at *any* band, so raising denoise to change
    an element's material or palette just breaks its structure instead. "Make this
    element look like X while keeping its shape" = band 1–2 **plus a prompt that
    describes X**.
  - Over a mask, denoise does not control *how much* changes, only whether the
    structure survives — the masked region is repainted whole at every band
    (`out = out * mask + latent * (1 - mask)`; see `local_repaint_bigcanvas.md`).
- **Repairing hands / faces (the "detailer" family) — measured boundaries (2026-10-03).**
  The community recipe is always *locate → crop → upscale ×2 → feathered mask → low-denoise
  repaint → paste back*; what decides success is **whether the defect is a texture problem or
  a topology problem**:

  | target | defect type | what works | what does not |
  |---|---|---|---|
  | face | **resolution loss** (small face, was upscaled) | `denoise 0.35–0.45` — eyelashes, pupil highlights, hair strands appear; in-mask change ≈ 13–14 %, out-mask max Δ = **2** | — |
  | face | **structure destroyed** (smudged/smeared) | nothing at this scale — even `0.55` moves only **5 %** of in-mask pixels; `0.7` only guesses at the mouth | low denoise (it faithfully reproduces "blurry") |
  | hand | small / soft (skeleton still trustworthy) | hand-skeleton as LLLite, low denoise | — |
  | hand | **fingers fused / six fingers** (skeleton itself wrong) | nothing reliable: `0.45–0.92` moves ≈ 23–30 % of pixels but the **detail energy goes DOWN** (it smooths, not rebuilds); `1.00` finally separates the fingers cleanly **and** grows a new hand beside her face | LLLite on the bad skeleton (it anchors the error); mid denoise (looks like it changed, is only blurring) |

  Three rules that follow:
  1. **Never measure a repaint by "how many pixels changed".** Read the *direction*: a
     smoothing pass gives a large in-mask change with **negative** detail energy. Pair
     `changed_pct_inside_mask` with a Laplacian-energy delta.
  2. **A repaint has no placement constraint.** It can fix local topology but cannot know
     *where* the hand should go; supplying a corrected skeleton/dense map is the precondition
     for high denoise, not an optional extra.
  3. **Paste-back cleanliness is cheap to prove**: out-mask max Δ ≈ 2 means the feathering is
     fine and any remaining artifact is inside the mask. Report it with every repair.
- **Character replacement without a face-swap model** (`swapchar.py`): HED edge map →
  `ControlNetApplyAdvanced` (softedge_hed, 0.9) + **grayscale** base at img2img `denoise 0.8`
  + character LoRA 0.85 keeps pose / rope topology / framing and swaps identity, hair and
  outfit. Prove it is the LoRA and not a re-roll by running the **same seed with the original
  character's LoRA** as a control — the control must reproduce the original character.
  (See also the §"云 API" note: never hand the model the original colours if the outfit must change.)
- **True identity transfer needs FaceID, and FaceID needs two weight sets that are NOT on this
  host** (checked 2026-10-03). Do not reach for plain IPAdapter and call it identity transfer:
  `ip-adapter-plus-face_sdxl_vit-h` is a **CLIP-vision image prompt**, so it changes face shape
  and eye shape but does **not** carry the reference's hair or iris colour, and at
  `weight 1.0` it destroys the image outright (severe colour cast, features dissolve) with or
  without a character LoRA — the usable band is ≈0.5 and it still is not identity.
  `FACEID PLUS V2` on SDXL in turn requires **all three** of:
  `ip-adapter-faceid-plusv2_sdxl.safetensors` (matched by regex `faceid.plusv2.sdxl.(safetensors|bin)`
  in `comfyui_ipadapter_plus/utils.py::get_ipadapter_file` — a FaceID **LoRA alone is not enough**),
  `ip-adapter-faceid-plusv2_sdxl_lora.safetensors`, and an `insightface` pack with
  `det_10g.onnx` + `w600k_r50.onnx`. Missing the first one fails as
  `Exception("IPAdapter model not found.")` at `IPAdapterUnifiedLoaderFaceID`, which is easy to
  misread as an insightface problem.
  For the objective ArcFace identity metric, a **minimal pack** (`buffalo_l_min/` holding only
  `det_10g` + `w600k_r50`, see `verify_arcface.py`) is sufficient — it skips the 143 MB `1k3d68`,
  and `FaceAnalysis` must be pointed at that dir because it loads **every** onnx it finds there.
- To **copy a pose that already exists as a skeleton image**, use the ingest trio
  (`blender_extract_kps.py` → `blender_fit_pose.py` → `blender_pose_error.py`).
  Fitting runs inside Blender so the FK/projection cannot drift from the renderer,
  and it is fast: **27.0 % → 6.9 % RMS in 11 s** on an arabesque target. The
  residual floor is the mannequin's fixed proportions, not the pose.
- To **get a pose out of a photo**, use ComfyUI's own detector —
  `LoadImage → DWPreprocessor → SavePoseKpsAsJsonFile` — then
  `blender_dwpose_to_kps.py` to convert the JSON (already COCO-18, no colour
  extraction needed). DWPose beats `OpenposePreprocessor` on small input; keep the
  `confidence` field, since low-score joints carry garbage coordinates and both the
  metric and the fitter drop them below 0.3.
- **`blender_pose_error.py` is same-view only**: the same 3D pose scores **87 %**
  against itself across two camera angles. ~5–10 % means a good same-view fit;
  50–90 % across viewpoints says nothing about the pose.
- Control models on this host: `noob_sdxl_controlnet_depth.safetensors` (4.66 GB)
  and `openpose_pre.safetensors` (2.33 GB) on Illustrious;
  `anima-lllite-pose-1.safetensors` (0.02 GB) on Anima via
  `AnimaLLLiteApply_sdscripts`.
- Timings at 832×1216 on 6 GB: depth CN 85–105 s, pose CN 47–70 s, LLLite ~60 s,
  img2img 41–49 s, depth+pose stacked 160–171 s (weights exceed VRAM; ComfyUI offloads).
- `comfy_control.mjs` uploads control images to `input/dsh_ctrl` (delete after) and
  reads results via `PreviewImage`/`type=temp`, so **nothing lands in `output/`**.
- Hand keypoints fixed "mitten hands" in an A/B at identical seeds — but only
  because the control model was trained with hands; A/B it, never assume.

## Cutting the background out (matting / alpha)

Full playbook with measured numbers: **`references/matting_and_cutout.md`**. Pick by input type —
the difficulty differs by an order of magnitude:

| Input | Route | Measured |
|---|---|---|
| Our own render, prompt said `plain white background` | `matte_flat_bg.py "out/**/*.png" --out DIR` | SAM score 0.961 mean over 19 images, 1–3 islands each, no decisions |
| Someone else's illustration / poster | `matte_from_scribbles.py IMG --fg FG.png --bg BG.png [--subtract mask.npy]` | IoU 0.737 against a hand-cut GT (0.779 when the best SAM proposal is hand-picked via `--pick`) |
| Flat chroma background | `chroma_key.py IN.png OUT.png` | background-coloured pixels left inside: 0.000% |
| Need props/effects **removed** or occluded areas **rebuilt** | `bailian_image_edit.py edit IN.png OUT.png "<instruction>"` then `chroma_key.py` | one 21 s call, but it re-renders: two prompt variants agreed at only alpha IoU 0.586 |

Key facts that decide the approach (each cost a failed attempt):

- **SAM, not colour.** `models/sams/sam_vit_b_01ec64.pth` ships with this skill; install with
  `pip install segment-anything`. GrabCut fails on white hair over a white background, and a
  lineart-dam flood fails on posters whose background is dark and textured.
- **The proposal choice matters more than the prompt.** With a box over the whole subject, SAM's
  highest-scoring mask is often everything in the box. `sam_object_mask.segment` ranks proposals
  by IoU against the rough foreground / the human's scribble, and `matte_from_scribbles.py` prints
  every proposal so `--pick` can override.
- **Do not invent island rules.** On a poster, "keep only the largest island", "islands right of
  x=500 are flames" and "pale islands are flames" respectively deleted the long hair sweep + lower
  leg, the red wing ornament, and the entire character. Use the human's scribble per island: it
  agreed with the hand-cut ground truth on **7/7** islands.
- **Hosted edit models return no alpha** and re-render rather than segment, so a cutout from them is
  "draw on chroma → key locally", and the pixels are the model's, not the original's.
- **Verify, don't trust.** `score_matte.py GT.png mine.npy` (IoU / FP / FN / edge + error clusters)
  plus `compare_matte.py` (red = missed, blue = added) before claiming a matte is good.

## Upscaling (2x / 4x) — after generation, before delivery

Chain: **`UpscaleModelLoader` → `ImageUpscaleWithModel` → `SaveImage`**. No diffusion pass,
so it is fast and deterministic — 1216×832 → 4864×3328 takes seconds, not minutes.
Reference graph: `references/upscaling_workflow.json`. Model sources, the verified download
URLs and the full comparison: **`references/upscaling.md`**.

**Measured 2026-09-17** — same source, same 200×200 face crop, all four shown at 4×:

| model | verdict |
|---|---|
| `4x-AnimeSharp.pth` | **best** — cleanest line art, hair strands separate, no sharpening halos. **Use as default.** |
| `RealESRGAN_x4plus_anime_6B.pth` | good, marginally harder / more aggressive sharpening |
| `4xhfa2kludvaeswinir_v1.safetensors` | general purpose, not anime-tuned |
| `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` | **REMOVED 2026-09-25** (moved to `agent_work/_disabled_upscalers/`) — it was **noticeably softer than either 4x model**, and using it was the actual cause of "upscale looks blurry" reports. Use `4x-AnimeSharp` then Lanczos-halve instead. |

Rules:

- **Install the model; don't blame the node.** A weak upscale almost always means a weak or
  missing `.pth`, not that ComfyUI's upscaling is bad. The default install shipped with only
  a 2x anime model and a general-purpose SwinIR, which is why it looked poor.
- **Deliver at 2×.** 4× from a 1216-wide source is 4864 px / 16 MP. Keep the 4× result and
  downscale it by 2 with LANCZOS — a 4×-then-downscale beats a native 2× model.
- **Upscaling is not a substitute for framing.** If the subject is too small in frame, crop to
  it first and upscale that, rather than enlarging the whole canvas.
- **Only ESRGAN-architecture `.pth` loads in `UpscaleModelLoader`.** Real-CUGAN is a different
  architecture and needs its own runtime (ncnn) — see `references/upscaling.md`.
- **Fixing a few pixels inside a big image** is a different job from upscaling: cut the region
  out with `SmartImageCrop`, repaint it, paste it back with `SmartImageStitcher`, and limit the
  repaint with `SetLatentNoiseMask` on a **shaped** mask. Cost tracks the crop, not the canvas
  (a 3328×4864 page took **44 s** against 80 s for its 832-wide source). Full playbook:
  **`references/local_repaint_bigcanvas.md`**. Note the measured negative result there: the
  "4x → shrink → 4x" chain does not work out, because the model's factor is fixed at 4x.
  The same reference now also has the **no-custom-node route** (PIL crop → sample → paste by mask,
  with the `LoadImage MASK = 1 − alpha` trap that turns a repaint into a silent no-op), the
  `min-sample` rule for tiny targets, and when to **split one diagonal region into two passes**.
  **The stitcher itself is bit-exact; the crop's *rescaling* is what moves pixels (2026-10-04, corrected).**
  A previous note here claimed `SmartImageStitcher` "does not limit its change to the mask" (4.31 % out-of-mask,
  `max Δ 83`, and attributed it to the stitcher + its feather). **That claim was wrong** — the user challenged it
  (the node only cuts and pastes back) and the isolation experiment confirms it: with the crop **not rescaled**,
  a pure passthrough (`crop_image → SmartImageStitcher`, **no sampler at all**) reproduces the original
  **bit-for-bit** — out-of-mask `>2/255` = **0.0000 %**, `max Δ 0` — with `Box Feather 32` and with or without the
  stitcher's `mask` input. The **full sampled** graph is bit-exact too once the crop is not rescaled and the mask is
  wired (out-of-mask **0.0000 %**, `max Δ 0`, in-mask 62–73 % changed).
  The drift comes from **`SmartImageCrop` resizing the crop** — `force_divisibility=128` / `min_resolution=768`
  force a resize even with no sampling (measured 0.0542 % out-of-mask, `max Δ 46` just from crop→paste), and if the
  stitcher's `mask` is *not* wired the whole crop box carries that difference (0.3352 %, `max Δ 229`).
  So: **set `force_divisibility=8` (and a `min_resolution` the mask bbox already clears) so no resize happens, and
  wire `crop_mask` into the stitcher's `mask`** — then either route is exactly 0 outside. Report `max Δ outside
  mask` plus the `>2/255` percentage. Measurement script + raw numbers: this session's
  `task/lumine_hospital_20261004/verify_stitcher.py` (cases P1–P7).

### Shrinking to the delivery size, and proving sharpness

- **Downscaling is part of the recipe, not an afterthought**: `ImageUpscaleWithModel` → `ImageScale`
  (`lanczos`) to the target long side. Delivering the 4× output directly is 16 MP of no extra
  information.
- **A plain resize is not a substitute for the upscale model.** Same source, same target size
  (3000 px long side): plain LANCZOS gives `|Laplacian|` mean **3.04** / P99 44; 4×-AnimeSharp then
  Lanczos gives **5.17 / 94**, against the 2000 px original's 5.53 / 89 — i.e. the model preserves
  the source's edge energy while a bare resize loses ~40 % of it.
- **Score sharpness at the same pixel size, on the luminance channel**, using the 3×3 Laplacian's
  mean absolute value and P99 . A
  57 % gap is real; 5 % is noise. This is also how to settle "which upscaler" and "did my
  re-sharpen pass help" without arguing from impressions.

## Comic page — tiling N panels into one page

When the deliverable is a **page** rather than separate images: one image, N cells, straight
or slanted dividers, no white margin inside any cell, key content always visible, reading
order preserved. Two scripts:

| Script | Role |
|---|---|
| `libs/comic-layout/scripts/comic_layout.py` | layout search + renderer. `comic_layout.py OUT.png IMG... --page 1400x3200 --beam 12 --weights 1,0.6,1,0.6,1,1,1` |
| `libs/comic-layout/scripts/letter_page.py` | balloon shapes, text-box search, anchor solving (imports `comic_layout`) |

Design in one line: **recursively cut one convex cell with one straight line** — arbitrary
angles become legal, the union stays exactly the page, and because a cut hands range
`[lo..k]` to one side and `[k+1..hi]` to the other, reading order is structural. Each cell is
then filled with a `cover`-scale crop of its image (so it can never letterbox) slid to keep
that image's **key region** inside.

Full playbook — the objective terms, the two ways of getting reading order wrong (pairwise
ordering, and banding by vertical overlap), the blank-area probe, and the balloon geometry:
**`references/comic_page_layout.md`**.

Three things worth knowing before starting:

- **Ask the human for two annotations, not one.** Key regions *and* balloon anchors are
  semantic judgements — for a panel holding two people, tile statistics cannot say which is
  which. `human_canvas` with a labelled contact sheet (or pre-seeded nodes the human only
  drags) makes this cheap.
- **Probe before changing a constraint.** The search left 13.8% blank against an 8% target;
  raising the penalty 700x reached 7.7%, so it was a trade, not an infeasibility. Reworking
  the constraints would have been wasted work.
- **Every objective term is a normalised ratio**, so energy is comparable across page sizes
  and sweeping them is meaningful — on a 7-panel set a 1:2.29 strip scored 8.26 against
  14.95 for the next best, because portrait panels stack naturally on a tall page.

## Cloud APIs — generation and editing (and how to merge the answer back)

Local ComfyUI is the default; a hosted model earns its place where the local stack is
structurally weak: **text inside the image** (in-image Chinese/Japanese lettering, calligraphy,
signage), adding/refining objects on top of a hand-edited base, and scene-level cleanup.

| Service | Route | Facts |
|---|---|---|
| 百炼 / DashScope `qwen-image-edit-max` | `scripts/bailian_image_edit.py edit IN.png OUT.png "<instruction>" [--ref R.png] [--prompt-file F]` | synchronous, 20–50 s, model re-renders (no alpha, its own output size) |
| GPT-image (OpenAI) | hand the human a **package** (crop + mask + prompt) — no local script | best at **legible text**; mask convention is **transparent = editable** |

**① 参考图是必给的，不是可选（用户 2026-10-03 强调）。** `--ref` 里的图是**内容参考**：
画**某个角色**时，云端不会照你的文字去认人 —— 先在**本地出该角色的三视图**（正/侧/背，干净背景）
或找一张该角色的图当参考；有**具体场景**需求时同理（环境/道具/光线也要给参考图）。
**不给参考图 ⇒ 环境和人物都会对不上**（模型按自己的先验编）。

```
# 角色：先本地出一张干净的三视图，再拿它当参考
... 本地 Anima/IL 出 char_sheet.png（front / side / back，plain background）
python scripts/bailian_image_edit.py edit IN.png OUT.png "<instruction>" --ref char_sheet.png
# 场景：再把环境参考也一起给（可重复 --ref）
python scripts/bailian_image_edit.py edit IN.png OUT.png "<instruction>" --ref char_sheet.png --ref room.png
```

其余五条（都是实测，细节在 `references/cloud_edit_and_merge.md`）：

- **参考图是"内容参考"，不是控制图** —— 所以它**指定"是什么"，不指定"怎么摆"**：
  只给骨架图 ⇒ 姿势错、还多出一个人；**姿势要用文字描述**。两张图的分工是
  **`--ref` 定人物与环境，文字定动作与构图**。
- **Hosted models refuse R-18 edits outright** (measured 2026-10-02): DashScope/百炼 answered
  **HTTP 400 `DataInspectionFailed`** — *"Input data may contain inappropriate content"* — to an
  instruction that asked to **add rope bondage**. The filter reads the **instruction text**, so no
  amount of wording craft gets it through, and **rewording to slip past it is evading a provider's
  safety control — don't do it**. Split the job instead: the innocent siblings of the same edit went
  through fine (*"摆成太极式中心对称躺姿"* ✓, *"把床单上的图案换成普通床单被子枕头"* ✓) ⇒
  **innocent steps → cloud, rope/BDSM/undress steps → local**.
  2026-10-03 补两条：**连"不要 XX"都不能写** —— 指令里写了 *"画面里不要出现绳子"* 本身就吃了 400
  （过滤器匹配**关键词**，"禁止"也算提到；同"正向写否定无效"那个坑）；以及**输入图也会被审** ——
  一张正面朝观众的坐姿＋抬腿（穿着完整、无绳）被拒，而同角色的站姿四视图能过 ⇒ **同一姿势的侧视/背面更容易过**。
- **They do not respect the mask.** GPT-image's answer differed from the sent crop by >12 in
  **3.57 %** of the *outside-mask* pixels, with a ±6 px alignment sweep choosing (0,0) as best —
  tonal drift, not a shift. Bailian repainted a night sky **13 levels brighter** and moved a rope's
  entry point **56 px**. So: **paste back by mask, never replace the image.**
- **Fine-grained content rules get ignored** — "write only the six visible characters" came back as
  the whole couplet, with a mis-written character and one inserted. List the characters one by one,
  and prefer the model that got 6/6 right.
- **Never let a cloud model own a thin structure** (rope, leash, cord) that must line up with the
  rest of the picture: keep those pixels from the original in every merge mode.

Merge patterns that work: **mask paste** (tool respected the mask) · **subject-only** (mask =
old∪new subject silhouette — background stays bit-exact; needed when the tool also re-lit the
background) · **rect / vertical-split** (leave the seam to the human in PS, or combine two
strengths: body 0.3 + sky 0.5 with a 30 px ramp). Two guards: a **dark guard** in the feather ring
(`min(base, result)`, kills the halo of a lighter cloud sky) and a **sky match** (add the median
per-channel offset measured on non-subject pixels). Verify with
`diff_report.py base.png merged.png --mask m.png`: in-mask ≈ 30 % changed, **out-mask 0.0000 %**.

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
同一张图里"谁被绑/谁自由、谁黑发/谁白衣"这类**属性独占**需求，提示词经常压不住（实测 8~10 次失败：
模型会按颜色把道具配对、同类并置结构互相传染）。**结构性**解法是逐人掩版重绘 —— 一人一次 pass，
另一人的像素在这一次里被冻结：

```bash
# 1) 定位 + 出逐人掩版（默认 pose 提示 + SAM）
... check_parts.py --src BASE.png --out OUT --person-masks --json
# 2) 掩版在 OUT/mask/：<name>_person0.png（灰度 255=重绘区）与 <name>_person0_rgba.png（重绘区 alpha=0）
# 3) 每人一次：LoadImage(RGBA) → VAEEncode → SetLatentNoiseMask → KSampler(denoise 0.65~0.70) → VAEDecode
#    串行跑，每个 pass 只碰一个人
```

**实测经验（踩过的坑，照做能省几次废图）**
1. **ComfyUI 的 `LoadImage` MASK = 1 − alpha** ⇒ 要重绘的地方 **alpha 必须为 0**。`_rgba.png` 已经按这个约定写好；
   用灰度 `_personN.png` 自己接时，记住 255 = 重绘区。
2. **别为了"保住某根细绳/某个道具"在掩版里挖洞**：框外的像素本来就不会被重绘，绳的连续性自动保住；
   在**人身上**挖洞会把该处冻结住，"把绳从她身上去掉"这类减法就永远做不到（实测被用户一眼看出孔洞）。
3. **掩版边缘羽化 8~16 px**（或收尾补一次整图 denoise 0.25~0.3）：硬边掩版会在潜空间 8× 下采样后留下
   一条淡色竖缝（雪地这种平坦背景最容易看出来）。掩版外的变化应 <1%（实测 0.5~0.8%，就是那条晕环）。
4. `--grow`（默认 15）就是留接缝余量；**别用检测框直接当掩版**（会把坐着的人下面的椅子/雪橇一起重绘）。

**逐人掩版是怎么定的（可查、可反驳）**：`persons_masks` 每个实例给出`tag / person_id / chain / source(M1 或 M1+M2a 或 M2a) / prompt / area_frac / pose_coverage / merged_from`，
外加 `person_boxes_dropped`（低置信或嵌套的 person 框及原因）与 `person_instances_merged`（哪些链被合并）。
- **角色以 M1 骨架链为准**（一条链 = 一个人）；M2a 的 person 框只用来给链配 `person_id` 和置信度过滤。
  实测同一张图 M2a 会给出 3 个框（0.90 / 0.44 / 0.31，后两个是重叠伪框），而 M1 的两条链才是真正的两个角色。
- **没有骨架支撑的 person 框不出掩版**（默认；`--allow-box-only-persons` 可强制，整张图一条链都没有时也会自动放行）。
  这条是用户复核后加的：餐厅那张图里 `prob=0.31` 的伪框正好覆盖**左侧整面墙**，给它出掩版就是**纯错标**。
  被丢掉的框与原因都写在 `person_boxes_dropped`（`low_prob` / `nested_in_other` / `no_skeleton`），
  每个产出的掩版还带 `supported_by_pose`（有没有骨架证据）。
- **`pose_coverage` < 0.7 就该人工看一眼**（该角色的可见关键点有多少落在掩版里）；掩版完全错时它会很低。
- **逐人掩版保证互不重叠**（`--no-exclusive-persons` 可关）：两人贴在一起时，重叠像素按**掩版内部深度**
  （各掩版到自身边界的距离变换）判给"更深在谁身体里"的那一位 —— 接缝沿两人身体的中介线走，像手工描的分界。
  旧的直线切法仍可用（`--overlap-split bisector`），但它会横贯身体切出一道直边（实测：两人靠肩那张，
  直线切会把靠肩那位的脸和另一位的头发各切一条竖线）。被处理的重叠写在 `person_overlaps_resolved`。
  实测：自动分配 57k 争议像素 ⇒ **最终重叠 0 px**，两种切法归属不同的像素 12.6k（就是接缝那一条）。
- **头一定在掩版里，但不鼓成"光环"**：M1 的关键点凸包只包住"鼻/眼/耳"那一小块，盖不到头骨与头发 ⇒ 掩版显式
  并入该角色头部的椭圆；椭圆优先按 **M2b 的 face 框**（比 M2a 的 head 框紧）算，并与该角色自己的轮廓求交，
  否则两人靠一起时两圈"头光环"交叉，看起来就是一团奇怪的十字。
- **`pose_coverage` < 0.7 就该人工看一眼**（该角色的可见关键点有多少落在掩版里）；掩版完全错时它会很低。
- **覆盖策略（`--person-mask-cover`，默认 `auto`）**：用户 2026-10-13 定调 —— **宁可多盖**。
  理由：用这套流程的基本都是 agent，做的是**中等/大尺寸特征**的重绘；"只改了一半个人、接缝卡在躯干上"
  比"多盖一点（含骨架之间的空隙，甚至蹭到一点道具）"糟得多；**细小特征本来就不指望 agent 做，人工来**。
  `auto` = 先算贴边掩版，关键点覆盖 < 0.9 就补上骨架凸包（外扩 3% 身高）——两张真实多角色图上都触发了，
  覆盖从 0.59/0.76 变成 **1.00**；`tight` 留给"绝不能碰到道具"的场合。
- 实测对照（同一张坐姿图）：`tight` 4.4%（她下半身在桌下漏掉）→ `auto` 12.0%（整条骨架都在内）。
- `merged_from` 非空 = M1 把一个角色拆成了两条链，工具按"掩版重叠 > 较小者的 50%"合并了。
- 两张真实多角色图上的实测：`sled`（2 人，坐姿）→ 2 个互不重叠掩版（8.2% / 5.4%，椅子被排除）；
  `rest`（2 人 + 桌椅）→ 合并后 2 个掩版（11.9% / 4.4%），桌椅都不在掩版里。
- 成本：`--person-masks` 比不加慢 ~10 s/张（SAM ViT-B 在本机 GPU 上 1~2 s/张 + 每角色一次 predict）。

### 局部特征重绘 + 打底（underpaint）—— 第 2 份使用报告打出来的经验
**① 特征掩版**：`--feature-masks` 直接给 `hand*/foot*/head*/face*/gag*/blindfold*/eye*` 的框掩版，
外加一个 **`mouth*` = 脸框下 42%（横向内缩 12%）**。为什么要单独给 mouth：报告实测 **M1 的鼻/嘴细点不可信**
（鼻点落在嘴上、嘴位落在下巴），而**框是可信的** ⇒ **重绘嘴用脸框推，不要用关键点推**。
（可靠度分层：`M2a 手框 / M2b 脸框` 可信 > `M1 细点` 不可信。）

**② 掩版组合器 `mask_compose.py`（agent-in-the-loop）**：不必要求一次给对掩版 —— 检测已经落盘了可靠的东西，
剩下"这次要改哪一块"由 agent 自己拼。图元（`--box` / `--circle` / `--poly`）+ 已有掩版（`--union` / `--sub`）
+ `--grow` / `--feather`，秒级、不需要 GPU：
```bash
# 指尖一个小圆 + 脸框下半；羽化 8px（给潜空间 8× 下采样留余量）
python libs/sam-matting/scripts/mask_compose.py --like base.png --circle 430,352,74 \
  --union out/mask/feature/base_mouth0.png --grow 4 --feather 8 --out mask.png --rgba mask_rgba.png
# 只要这个人：person0 减掉 person1
python libs/sam-matting/scripts/mask_compose.py --like base.png --union out/mask/base_person0.png \
  --sub out/mask/base_person1.png --out only_p0.png
```
（它**拒绝**把 `--rgba`/`--out` 指到输入图上，避免把原图改写成带透明区的 RGBA。）

**③ 打底（underpaint）—— 加"原本不存在的物件"时必须先做**：局部重绘是**基于已有像素**去改的，
那块本来什么都没有，重绘结果就不可控。做法：**先用程序/PS 在目标位置粗画出那个物件**（哪怕只是色块/椭圆），
再掩版重绘（denoise ~0.70）——**打底给"形"，重绘给"质"**。报告实测：程序粗画的红球 → 一次重绘变成
有高光/柄/叶/抓握关系的苹果，而**直出两次都失败**（模型会为了让"手能拿到苹果"而**解放被绑的肢体**：
道具动作与拘束结构互相抢 ⇒ **先定结构、再局部补道具**）。

  **形的三个来源（按图案复杂度选）**：① **程序粗画**（`scripts/tools/pattern_fill.py`：纯色/条纹/格子/点阵 + 描边 + 外扩；或 `mask_compose.py` 的图元）；② **人手绘**（PS 或画布 —— **复杂/有机/细致的图案建议走这条，人画一次比返工三次便宜**）；③ **画布给区域 + 脚本填充**（人涂一层 → `pattern_fill.py --mask 那层`）。三条路的命令、掩版规矩（外扩 2~8 px、**绝不压结构件**）与反例见 **`references/underpaint_and_stencil.md`**。

**④ 掩版绝不要压到结构件**：同一份报告里颈环/腕铐/横杆被画坏，唯一原因就是掩版放大到压上了它们；
改成"指尖小圆（离金属杆 90 px）+ 脸框下半（底边停在颈环之上 4 px）"后结构件毫发无损。
⇒ 凡是重绘区旁边有**道具/家具/拘束具**，先 `mask_compose --sub` 掉它们，或用小图元贴着特征走。

**⑤ 提示词自相矛盾会毁掉重绘**：报告里两次失败各有自己的矛盾句（`the top of her fingers is bare` vs 要画苹果、
`pouting` vs `her mouth is open`）—— 模型会挑其中一条执行。重绘前把整段提示词通读一遍。

### 两个"模型不认识的新角色"同框、且不许反左右（2026-10-13 第 3 份报告的标准解）
1. **先出一张不挂任何角色 LoRA 的"版式底图"**，把布局与关系做对；方向一律写**画面坐标系**
   （`the girl on the left side of the image`、`leans sideways to her left`），负面点名
   `mirrored composition / swapped positions`；
2. `check_parts.py --person-masks` 拿**每人一块、互不重叠**的掩版（见上：重叠会自动切分、头会自动并进来）；
3. **每次只挂一个人的 LoRA**（1.0/1.0 + sidecar 的完整触发词表），`denoise 0.65~0.75`，串行两次；
4. **每人的重绘提示词里都写一遍接触关系**（"the other girl leans her head on her shoulder"），否则边界处的姿态会被改坏。
> 中文角色名在 LoRA 文件名里常搜不到，但写在 sidecar 的 `model_name` 里（遍历匹配）；最全的触发词在
> sidecar 的 `civitai.images[].meta.prompt`。

**模型与代码位置（只给路径，不复制）**

| 东西 | 路径 |
|---|---|
| 套件根 | `<ANIME_DET_SUITE>`（`--suite` 或环境变量 `ANIME_DET_SUITE` 可覆盖） |
| 产线入口 | `<套件>\scripts\locate.py` |
| 推理环境 | `python`（ComfyUI 自己的 env，有 torch+ultralytics；`--python` 可换） |
| 权重 | `<套件>\runs\<name>\weights\best.pt`；每层的默认权重写在 `locate.py` 顶部的 `DEF_*` |
| M1 骨架 | `runs/m1_pose_coco17` + `runs/m1_pose17_1129`（取检出人多的） |
| M2a 框 | `runs/m2a_det_996img_960`（person/hand/foot/head）+ `runs/m2a_foot_v1`（单类脚，取并集）；可选 `runs/m2a_hand_v2`（单类手） |
| M2b 面部 | `runs/m2b_face_v3_3983`（输入是 head 裁剪的 256×256） |
| M2b-gag | `runs/m2b_gag_v2` + `runs/m2b_gag_v4`（并集；口塞/眼罩） |
| M3 指趾判定 | 现役 `runs/m3_cls_aug_v1` + `runs/m3_review_v1`（**val 有训练集泄漏**，见套件 R43）；推荐 `runs/m3_clean_v1_s2026{1,2,3}010` |
| M4 绳子 | **不在本入口**：由 rope 模块产出 `data\masks\rope\<name>_{mask,conf}.png`，`locate.py` 只按约定路径引用 |

**已知限制（别把"没报错"当成"一定没问题"）**
1. M3 的诚实指标（按源图 5 折折外）：召回 **0.74（单模型）/ 0.79（双权重并集）**，ok 误报 14%~19%；**手（0.61）明显差于脚（0.85）**。
2. 脚召回靠三级兜底（4 类 ∪ 专用脚 ∪ 踝点外推）；契约 JSON 的 `boxes` **不含**兜底补的部位（见 `boxes_all`）。
3. 分辨率：`--imgsz 1600` 明显好于 1280（手覆盖 4/8→5/8，08 那种"趾多"从只被收紧旗标报出变成进主判定），代价约 1.5~2× 时间；本机 1536×1024 的图在 1280 下约 17 s/张、加 `--person-masks` 再 +10 s。
4. 上面的验收数字都来自**同一批 8 张图**，小样本；换风格/题材请自己再复核几张。
5. `rope` 只表示"**rope 模块有没有留下掩版文件**"（`present|missing`），**不代表图里没有绳** —— 本入口不做绳子检测。
6. 更细的接口与实验记录在 `<套件>\docs\`：`integration.md`（接口与降级）、`runs.md`（R1~R49 实验）、`handover_20261012.md`（现状与待定项）。

**契约 JSON 里给下游用的字段（2026-10-13 新增，都是加法式的）**
- `points[].person_id` / `boxes[].person_id`：把点/框挂到具体某个角色上（多角色图做逐人处理时必需）。
- `n_persons`（M2a 的 person 框数）与 `n_skeletons`（M1 链数）分开报；旧的 `n_people` 仍是骨架数（兼容）。
- `boxes_all`：`boxes` **加上**兜底补的部位（专用脚 / 踝点外推 / 专用手），每个框带
  `origin: det | fallback`。做掩版请用 `boxes_all`，否则会漏掉兜底找回来的脚。
- `persons_masks` / `person_boxes_dropped` / `person_instances_merged`：见上面"逐人掩版是怎么定的"。

**想更新模型**：把新权重放到 `<套件>\runs\<name>\weights\best.pt`，然后改 `locate.py` 的 `DEF_*`，
或调用时用 `--m3-weights/--hand-weights` 指定；`--recipe rec` 就是"实测更好但还没转正"的那一套。

## Finding a LoRA and its trigger words
- **Keep `strength_model` and `strength_clip` at 1.0** (user rule, 2026-09-29). Interpolability in
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
  - **Best source of all: `civitai.images[].meta.prompt`** — the sidecar stores the **generation prompt of each preview image**, i.e. the author's own tag table for that character, and it is more complete than `trainedWords`. In the 2026-09-21 session a character's chest badge was absent from *both* `trainedWords` variants and showed up only there, as `lanyard`. One line to read it:
    `for im in (m.get("civitai") or {}).get("images", []): print(im["meta"].get("prompt"))`
  - **Finding the word is not the same as being able to suppress it.** Negating that badge (`lanyard, id card, name tag, badge, nameplate, tag, neck strap, cord around neck, hanging ornament`) did **not** remove it — **character priors outrank the negative prompt**. Changing the outfit did remove it. If a negative produces no visible change, the element is baked into the LoRA's default look: change something else (costume, pose, crop) or edit the pixels.
  - **Author-recommended sampler/steps/CFG** often live in the same sidecar's `civitai.model.description` (e.g. "Steps 30 / CFG 5-7 / DPM++ 2M SDE Karras"). Prefer those over the generic defaults below.

  **Match the LoRA's base pipeline**: a LoRA trained on Anima (`anima`) goes in the Anima workflow; one trained on Illustrious/SD goes in the IL workflow. Applying a LoRA to the wrong base yields poor output.

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
  (scripts/fetch_url.py), which goes through OpenSSL. `curl` and
  `Invoke-WebRequest` will keep failing no matter what you set `SecurityProtocol` to.
- **Upscale output looks soft** — check which `.pth` is actually loaded. `2x_AnimeJaNai` was
  much softer than `4x-AnimeSharp` (it is **gone** as of 2026-09-25 — see Machine facts), and a
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
  Measured 2026-10-03 on `OpenposePreprocessor`: same fixed name → `face_keypoints_2d` 0 / `people: []`
  on every repeat; same bytes under a fresh name → **face 70/70, identical bbox**, 3/3 runs.
  Do **not** "fix" this with a content hash — identical content means an identical name, which makes
  a cached empty result deterministic. Upload each detection input under a **unique** name
  (`role_stem_timestamp_rand.png`). This also silently ruined a dose sweep: every variant after the
  first reported "no face found", which looks exactly like a flaky detector.
- **A script reports "detection failed" but the detector is fine** — on Windows the console defaults
  to GBK, so `print("✗ …")` raises `UnicodeEncodeError` and **kills the process**; the caller sees a
  non-zero exit and no success line. Measured: 3/3 "no face found" without the guard, 3/3 success
  with it, same image. Put `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of
  every script that prints symbols, and invoke child scripts with `python -X utf8`.
- **Deciding a download source is broken** — this host's international routes can drop
  completely (measured 2026-10-03): hf-mirror / huggingface.co / github.com / ghproxy /
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
