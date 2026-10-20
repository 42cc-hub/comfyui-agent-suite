---
name: comfyui-image-generation
description: Generate images through a local ComfyUI server via its HTTP API, including the Illustrious (SDXL anime) and Anima (Cosmos-Predict2-2B anime) model pipelines, LoRA selection/trigger-word lookup, submit/poll/save automation, Blender-authored ControlNet/LLLite control maps (depth, OpenPose pose with hands, img2img), cutting the background out of renders or existing illustrations (local SAM matting, chroma key, or a hosted image-edit model) with measured scoring against a ground truth, and self-checking a finished render with a trained detector suite (people/hands/feet boxes, finger-and-toe count suspicion, gag/blindfold, rope mask) via `scripts/check_parts.py`. Use when rendering, iterating on, or cutting out images instead of using a hosted image API, or when a render needs its hands/feet/digits verified before delivery.
allowed-tools: Read Write Edit Bash
license: MIT
compatibility: Requires a running ComfyUI server (<你的 ComfyUI 安装目录>, conda env sdenv, port 7187, NVIDIA GPU). Uses `node` (Node 18+) for the API calls; Blender 5.1 at `<你的 Blender 安装目录>` for the control-map pipeline; `ffmpeg` optional only if audio->image conversion is needed elsewhere. Matting additionally needs `pip install segment-anything` (small pure-Python package) and the SAM ViT-B checkpoint that ships with this skill at `models/sams/sam_vit_b_01ec64.pth`.
metadata:
  version: "1.13"
  skill-author: local (machine-specific operational notes)
---

# ComfyUI Image Generation

Drive the local ComfyUI at `http://127.0.0.1:7187` over HTTP. Two pipelines: **Illustrious (IL, SDXL anime)** and **Anima (Cosmos-Predict2-2B anime)**. Everything that happens after a picture exists (control maps, masked repaint, upscaling, matting, tiling, cloud edit, human handoff, localization, safety gate) is a separate document under `references/`.

## Working mode — **semi-auto, default**（硬规则 1：用户没说的不做）

- Run the action asked and nothing else: no extra steps, no unrequested model/parameter/prompt change, no unasked retry or multi-round loop.
- Show each result with one `dsh_show_media` call, then continue.
- Cannot judge the picture yourself (is the hand broken? may it ship?) ⇒ **stop and hand it to the human**. "I think it's fine" is not an acceptance result.
- Report **facts** — path, seed, size, seconds, detector output — never impressions.
- Announce money/batch/irreversible actions before doing them.
- Self-direction requires an explicit grant ("你来定 / 自行决定 / 不用问我").

Contract, write-permission boundary, self-check list: **`references/modes_and_discipline.md`**.

## When to use

- Generate images on the local GPU (not a hosted API) from text prompts; iterate on prompt / seed / LoRA weight / sampler.
- Apply a LoRA to a base model and need its activation/trigger tags.
- Pose, framing, focal length or foreshortening must be **exact** rather than prompt-guessed → author it in Blender: `libs/blender-controlnet/references/blender_control_pipeline.md`.
- Need the **background removed / an alpha cutout** from a render or an existing illustration → `libs/sam-matting/references/matting_and_cutout.md`. Difficulty differs by an order of magnitude: our own flat generated background vs. someone else's poster.
- Fix a detail only a few pixels wide (a shoelace, an eyelet) without disturbing the rest → `SmartImageCrop` + `SetLatentNoiseMask` + `SmartImageStitcher`: `references/upscaling.md`, `references/local_repaint_bigcanvas.md`.
- The deliverable is a **comic page** — N images tiled into one page with dividers and lettering, in reading order → `libs/comic-layout/references/comic_page_layout.md`.
- The local stack is structurally weak at the task — **text inside the image** (Chinese calligraphy, signage), adding objects on top of a hand-edited base → **cloud** model, paste back by mask: `references/cloud_edit_and_merge.md`.
- **A human must supply rough 2D input** (a painted mask, a few points/edges) or do real Photoshop work on this host → Human-in-the-loop section + `references/human_in_the_loop.md`. **Photoshop handoff: ONE file per round — opening a second file while one is open FREEZES Photoshop on this host** (`[measured]`, n=1; the process must be killed and unsaved work is lost).
- Must **know exactly where something is** before masking it → Locating-things section + `references/grid_localization.md`. Answer four questions first (shape / occluded? / busy surroundings? / may it go to the cloud?) then pick one of six routes: ① detection box, skeleton dilation, inscribed ellipse in the head box (free, try first) ② **SAM prompted segmentation** (box + points, 1–2 s, best-fitting boundary) ③ cloud vision model colour separation (send **crops only**, gate through WD14 first, costs money) ④ grid count → rectangle or inscribed ellipse ⑤ crop + colourblock add/subtract (cheapest when surroundings are simple; at fuzzy boundaries always **grow the target first, then subtract confirmed non-target**) ⑥ canvas for the human to paint (safest, needs a human). The routes are parts and compose (`mask_compose --union/--sub`): trade-offs, measured numbers, copyable commands in **`references/localization_and_masks.md`**.
- Need a **body-part mask without asking a human to paint one** → local OpenPose node + `pose_keypoints.py` give per-part boxes (feet/legs/face/arms/hands/torso) and can write the mask: auto-localization section of `references/local_repaint_bigcanvas.md`. Reliable on standing/kneeling; **extreme poses (all fours, suspension) lose joints** → ask for a rough stroke or use `SAM3_Detect`.
- **Before any image enters your own context** — every vision read uploads the picture to a model provider → run the local WD14 gate first: Safety-gate section + `references/output_safety_gate.md`.
- A figure is **complex** (two subjects in contact, a container/rig, a crowd, an unusual camera, a foreground foot/face) and you must decide **what to sample, what to program, what can be added later** → Planning section below.
- Starting a picture or a set and need the **macro order** (when to sample, when a human edits, when to upscale, when to switch models) → Production-pipeline section below. Two counting disciplines: **many samples only at step 1**, and **upscaling happens more than once** because fine objects need pixels.
- Have a **reference illustration of an interior** and need controlled depth/line maps, that image's **camera**, or are compositing a character into a generated background → `references/perspective_and_scene_geometry.md`.
- **Composited something** (program-drawn rope/chain, Blender part, matted cut-out, UI overlay) and it reads as **pasted** → Harmonizing section + `references/harmonization_and_embedding.md`; run `scripts/harmonize.py`.
- Starting or resuming a **multi-image project** (set, series, client job) and need disk/metadata discipline → two load-bearing rules: **default ephemeral → explicit promote**, and **one canonical implementation per operation — prefer the engine's native nodes over a new Python script** (a masked repaint pastes back with `ImageCompositeMasked`, not a hand-written merge).

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
| Upscalers | `models/upscale_models/`: `4x-AnimeSharp.pth` (**default**), `RealESRGAN_x4plus_anime_6B.pth`, `4xhfa2kludvaeswinir_v1.safetensors`. `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` **removed** (visibly softer; now in `<你的 ComfyUI 安装目录>\agent_work\_disabled_upscalers\`). **Deliver with `4x-AnimeSharp → ImageUpscaleWithModel → Lanczos to the target width`**, never a native 2x model. Comparison: `references/upscaling.md`. |
| Outbound network | **The shell has no direct internet.** HTTP times out; HTTPS via curl/.NET dies with `schannel: AcquireCredentialsHandle failed`. Only working path: local proxy `127.0.0.1:6132` + **Python (OpenSSL)** → scripts/fetch_url.py |
| **NEVER TOUCH** | **`<你的 ComfyUI 安装目录>\output\`** — a **mixed** directory: the user's own files live there next to renders (notes, downloaded reference pngs, `output\漫画委托\` = the whole commissioned-project material). Deletion here is **unrecoverable** (`os.remove` skips the Recycle Bin). **Never delete, move, or reorganize anything under `output/`** — file-by-file, by explicit name only. |

## Cleaning up after a run（硬规则 2：出图不落 `output\`）

**Workspace boundary: work inside `<你的 ComfyUI 安装目录>\agent_work\` only.** Everything outside it (`output\`, `input\`, `models\`, the ComfyUI tree, Desktop, Downloads) is the user's territory: never delete, move, rename or reorganize anything there unless the user names that exact file. Cleaning **cannot be undone** — `os.remove`/`Remove-Item` bypass the Recycle Bin. Last resort for a lost name: the session transcript `~/.dsh/sessions/<project>/<session>/session.v3.jsonl.zstd` still holds the real filenames — .

1. **List first, delete second** — print every path + size + mtime before removing anything, even for a casual "clean up the intermediate files".
2. **Whitelist by filename** — never by "directory + age" or "directory + size"; coarse conditions sweep the user's files.
3. **Mixed directories are off-limits**: `<你的 ComfyUI 安装目录>\output\`, `<你的 ComfyUI 安装目录>\input\` (except the staging subdir below), Desktop, Downloads — file-by-file, by explicit name.
4. **Prove "it is a duplicate" before deleting it**: sha256, or a pixel diff of **0.0**.
5. **`input/dsh_edit/` is the one safe-to-empty directory** — pure staging; every script re-copies its inputs before each run.
6. **Never delete**: the user's hand-made files (PS handoffs, hand-painted mask layers), authored sources, `final/` dirs, exported pages, or any file whose name looks like a note / script / reference.

**Generate with `PreviewImage`, never `SaveImage`.** The result lands in `temp\` and comes back via `GET /view?filename=..&type=temp`; `temp\` is **wiped on restart**, so anything to keep is **explicitly re-saved** into the workspace/task dir as a delivery step. `output\` stays a by-filename whitelist (§3). `SaveImage` only when the user names an exact destination directory, with `filename_prefix` pointing into the workspace.

## Safety gate: check before you look（硬规则 3）

**Reading an image with a vision tool uploads that image to the model provider.** "Do I look at it?" is a controlled action, not a reflex.

```bash
python libs/wd14-gate/scripts/wd14_gate.py out/*.png            # local WD14 tagger → tags + rating
python libs/wd14-gate/scripts/wd14_gate.py out/*.png --tags     # with the full tag list, for a human read
python libs/wd14-gate/scripts/nsfw_gate.py out/*.png            # pixel-statistics pre-filter (coarse)
```

| Verdict | Meaning | Action |
|---|---|---|
| `OK` | general | view it |
| `REVIEW` | sensitive / `questionable` **without** a key-point tag / **`explicit` without any key-point tag**（`escalated`）/ **only a pose tag**（`pose-level`，e.g. `spread legs`）/ watch-tag hit | viewable |
| `BLOCK` | **key-point tag hit**（`nipples`、`pussy`、`penis`、`anus`、`crotch`、`nude`、`topless`、`bottomless`、`presenting`、`clitoris`、`sex`、`cum`、`cameltoe`、`partially visible vulva`、`vulva`、`labia`、`visible clitoris` …）；或 `explicit` **且**命中关键点/姿势标签 | **never view** — see below |

**The hard boundary is a key point — not the rating, not the pose.**

- A `BLOCK` image gets a path in the reply, nothing else. Not `read_image`, and **not `dsh_show_media`** either — its result *does* re-enter the agent's context (`[measured]`). Review one with **`dsh_show_gallery`** (folder or `paths`): only a handle enters the tool result ⇒ **zero** context cost.
- `spread legs` belongs in the **pose table `POSE_TAGS`**: alone ⇒ REVIEW + `pose-level` mark; together with a key point, or rating=explicit ⇒ BLOCK.
- `cameltoe / partially visible vulva / vulva / labia / visible clitoris` are hard BLOCK — they are real key points and must not be gated only under explicit.
- Gate output prints the two marks that separate topic over-kill from real exposure: `escalated: rating=explicit but no key-point tags -> REVIEW` and `pose-level: spread legs (...)`.
- Threshold: `cameltoe`-class confidences sit at 0.30–0.35, so **`--threshold 0.30` blocks more than the default 0.35**.
- Evidence for the key-point boundary, and where the pose table came from: `references/output_safety_gate.md` §"单次观察". **Risk stays with the user.**

**Censoring a BLOCK image to look at it** `[measured]`:

- Mosaic the **whole panel** (16–64 px cells; 32–64 often reaches `OK`) — *not* the one suspicious spot, and never a flat colour fill (a flat patch reads as a censor bar and can rate worse).
- Tools: `scripts/tools/mosaic_censor.py` (human-specified mask/boxes) and, **only in fully automatic mode with no human available**, `scripts/tools/pose_censor.py` (skeleton → chest + crotch band). Both refuse to write a copy that hides nothing, and the censored copy **must** be re-gated before you look.
- **Censoring can backfire**: covering the underwear of a `REVIEW` image with a rectangle or mosaic (any colour) flips it to `BLOCK` — the clothing evidence goes away and `nude` appears. Painting garment *shapes* does not rescue it; on a nude base all fill colours still gate `BLOCK`. A gate verdict is not a measure of how much you actually hid. **To make an image safer, add clothing.**
- **Pass rule for your own look at a censored copy**: `scripts/tools/gate_after_censor.py --src IMG --censored COPY.png` — passes when the *key-point* words disappear (`nipples/pussy/penis/…`) even though `nude`/`censored` remain; fails if any key point survives or a new one appears.
- **If you must cover, cover the whole band.** If only the human needs to see the picture, `dsh_show_gallery` — no censoring needed.

Gate at the granularity you intend to act on: a crop that gates REVIEW can still make the full frame explicit, and a collage of four REVIEW panels can itself be BLOCK (`[measured]`, single case — see `references/output_safety_gate.md`).

## Planning a complex illustration — decompose before you render

Decide this **before** writing a prompt or building a control map; it decides whether a figure is cheap or hopeless.

- **Delete test** — remove an element: if the rest still makes sense it **can be added later** (off-frame hands, a leash leaving frame, wooden stocks, plain geometry); if not it **must be generated with the subject** (anything hanging, and anything that leaves a **mark**: rope indentation, cloth deformation, contact shadow).
- **Interaction has a direction** — the acting/looking side cannot be produced alone; the acted-upon side can be produced first; a symmetric pair (mutual gaze + contact) needs one generation.
- **Order layers so the uncontrollable goes first** — sampling cannot be aimed ⇒ sample the person, then build the machine/container in Blender **against that person**. Structure-first is for architecture, where the room *is* the subject.
- **"Would a human spot it?"** — countable / evenly spaced / symmetric / regularly grouped things (keyboards, chains, fences, grids, type) and **everyday objects** (wardrobes, desks, doors, furniture legs) are verified at a glance ⇒ program or Blender them. Same for **anatomy** (proportions, limb symmetry).
- **Macro vs micro** — orientation, proportions, perspective, contact, composition, anything large in frame must be right **at generation time**; rope detail, text, light patches, wetness and composites are repairable afterwards. Filter on the macro list first.
- **Copy before props** — write the line first, then reverse-engineer which props must be visible.
- **Hiding is a layer problem, not a prompt problem** (rope that must stay inside a collar): build the hidden thing, then put the coat on top.

## Recovering perspective — human line annotation → camera → control maps

Three steps: **fix the axes** (image x right / y down / z into screen; world x forward / y left / z up; rotation as a **rotation vector on a fixed baseline R0**, not free Euler angles) → robust fit on the human's lines → render depth/line maps from the solved camera. Method and measured numbers: **`references/perspective_and_scene_geometry.md`**.

## Harmonizing a composite — stop it looking pasted

A layer added to a finished image (program-drawn rope/chain, Blender render, matted cut-out, UI overlay, an AI-generated extra) reads as a sticker unless **tone, sharpness, grain and edge behaviour** match its surroundings. Use **`scripts/harmonize.py`**:

```powershell
python scripts\harmonize.py BASE PATCH --mask MASK.png --out OUT.png `
    --match lab --feather 8 --shrink 2 --ao 0.25 --ao-width 18 --ao-dir left --report
```

- `BASE` = finished illustration, `PATCH` = the layer (same canvas), `MASK` = white where the patch shows.
- `--match lab` (different-source patch) / `--match luma` (patch already carries the right colour, e.g. anime flat art) / `--match hist` (**only** when both sides are flat-colour with the same dynamic range).
- `--report` prints a **seam step** (mean |ΔL| across the mask border) before/after — optimise that number.
- Measured on a real case: naive paste seam **60.2** → `--match lab` **26.0** / `--match luma` **25.4**; `--match hist` made it worse (54.2).

Practical order: **classical match → masked low-denoise repaint of the seam (if the light is wrong) → classical match again**. Checklist (PS-side tricks, anime-specific notes, what classical matching cannot do): `references/harmonization_and_embedding.md`.

## Start / stop ComfyUI

Launch from the ComfyUI dir as a background job. **It must run with full file access** — otherwise the sandbox denies writes to its database/settings/output dirs (outside the workspace) and it exits with `[sandbox: file access denied under workspace-write mode]`.

First-time install (repo/tag, torch+CUDA incl. the Blackwell/sm_120 branch, model dir layout, verification): `references/installing_comfyui.md`.

```powershell
# clear any stale db lock, then launch (needs danger-full-access for G:\ and AppData writes)
Remove-Item '<你的 ComfyUI 安装目录>\user\comfyui.db-wal','<你的 ComfyUI 安装目录>\user\comfyui.db-shm' -ErrorAction SilentlyContinue
& 'python' 'main.py' '--port' '7187' '--listen' '127.0.0.1' 2>&1
```

Run it with `sandbox_permissions: "danger-full-access"` (justification: ComfyUI writes its db/settings/output outside the workspace). Wait for the ready line:

```
[INFO] To see the GUI go to: http://127.0.0.1:7187
```

Confirm with `Get-NetTCPConnection -LocalPort 7187 -State Listen` or `GET /`.

Startup blockers: **db lock** (remove `comfyui.db-wal` / `comfyui.db-shm`, above); **sandbox write denial** (relaunch with `danger-full-access`); ignore a single custom-node import warning (`NODE_CLASS_MAPPINGS is empty ...` / `'gbk' codec`) — the core pipeline still works.

## The HTTP API recipe

All calls go to the server base. Every input that references another node is `["<nodeId>", <outputIndex>]`.

1. **Discover nodes / enum values** — `GET /object_info/<NodeClass>` → `.input.required.<field>[0]` is the enum. e.g. `CheckpointLoaderSimple` → `ckpt_name`; `LoraLoader` → `lora_name`; `CLIPLoader` → `type`.
2. **Submit** — `POST /prompt` with `{"prompt": {"<id>": {"class_type": "...", "inputs": {...}}, ...}, "client_id": "..."}` → `{prompt_id, node_errors}`.
3. **Poll** — `GET /history/<prompt_id>` until `history[prompt_id].outputs` is non-empty; bail early when `status.status_str === 'error'`.
4. **Save** — first image in any output node: `outputs[<nodeId>].images[0] = {filename, subfolder, type}`; fetch `GET /view?filename=..&subfolder=..&type=output` and write the bytes.

Reusable script `scripts/comfy_generate.mjs` (reads the graph JSON, submits, polls, writes the first output image; `COMFY_BASE` overrides the URL):

```bash
node scripts/comfy_generate.mjs references/anima_workflow.json out.png
```

**Before a heavy run, empty the queue** — a stale job from an interrupted session takes the GPU and your submit just waits behind it. Use , or this one-liner (single quotes inside, so it survives PowerShell):

```bash
python -c "import json,urllib.request as u;[print(p, u.urlopen(u.Request('http://127.0.0.1:7187'+p, data=json.dumps(d).encode(), headers={'Content-Type':'application/json'}), timeout=20).status) for p,d in (('/interrupt',{}),('/queue',{'clear':True}))]"
```

**Hard limits on a 6 GB card** (not negotiable): sampling **side ≤ 2400**; stay **≤ 1536 native** when upscaling a crop for sampling; **downscale any 4× intermediate before it enters a sampler** (≈52 MP ceilings are unreachable here). The upscale model runs on the 4× intermediate *outside* the sampler, which is fine.

## Illustrious (IL) workflow — SDXL anime

SDXL-family anime base. `CheckpointLoaderSimple` (single node gives MODEL+CLIP+VAE), optional LoRAs, then standard txt2img. Reference graph: `references/illustrious_workflow.json`.

- **CheckpointLoaderSimple** `ckpt_name` = `Illustrious\anime\waiIllustriousSDXL_v170.safetensors` （默认 anime） / `SD\architecturerealmix_v11.safetensors` / `SD\deliberateCyber_v60.safetensors`. Outputs `[0]=MODEL, [1]=CLIP, [2]=VAE`.
- **LoraLoader** (optional) `model=ckpt[0]`, `clip=ckpt[1]`, `lora_name=<path>`, `strength_model`, `strength_clip` → `[0]=MODEL, [1]=CLIP`.
- **CLIPTextEncode** ×2 → `clip = (lora?lora[1]:ckpt[1])`.
- **EmptyLatentImage** → e.g. `1024x1024`, batch 1.
- **KSampler** `model=lora[0]|ckpt[0]`, `positive/negative`, `latent_image`, `seed`, `steps≈28`, `cfg≈6`, `sampler_name=euler_ancestral`, `scheduler=karras`, `denoise=1`.
- **VAEDecode** `samples=ksampler[0]`, `vae=ckpt[2]`.
- **PreviewImage** (default). `SaveImage` only on an explicit "write to this directory" request, `filename_prefix` inside the workspace.

### Image sizes & hi-res for video

- Landscape/anime usually `1024x1024`. For a **vertical (9:16) video** use portrait SDXL ratios `832x1216` or `896x1152`; `1080x1920` is valid but slow on 6 GB.
- An 832-wide source upscaled to `1080x1920` looks soft. Generate higher (`1216x1792`) or add a **hi-res fix** (second low-denoise KSampler pass / hires-fix / upscale LoRA). Keep headroom so a Ken-Burns crop doesn't clip the subject.
- **Size the canvas to the character, not to the aspect ratio** — the figure needs ≥ `700×1200` px or the face goes off-model (see below).

### Sampling resolution decides the *failure mode*, not just the framing

`[measured]` Illustrious, no LoRA, `cfg 8` / 20 steps, paired seeds, hands counted by a human; full data in our own measurements:

| canvas | hands with **extra** fingers (≥6) | hands with **missing** fingers (≤4) |
|---|---|---|
| 832×1216 (1.01 MP) | **0 / 40** | 4 / 40 |
| 1232×1800 (2.22 MP) | **9 / 40** (Fisher p = 0.0024) | 0 / 40 |

- **Sampling above the training resolution makes the model invent detail.** At 1232×1800 hands gained extra fingers, and the rate does **not** depend on how large the hand is drawn: same large canvas, hands drawn large 4/20 vs hands drawn tiny (full-body) 5/20, **Fisher p = 1.0**. It is the **canvas**, not the object's apparent size — framing tags (`close-up`, `full body`) cannot fix it; only the canvas can.
- **The missing-finger direction is the opposite, and *is* local:** small canvas with hands drawn small gave 4/20 missing, the same canvas as a close-up gave 0/20. There the object's pixel count is the variable — draw the hand bigger and the merging goes away.
- **Too small is unusable:** at 208×312 (0.065 MP) framing stops being controllable (per-seed `1girl`/`rope` tag σ up to 0.31). Usable floor ≈ 0.4 MP. The rope *content* was still correct at 0.065 MP — control is what is lost.
- **Large structures only benefit from size:** a rope harness got monotonically cleaner (`rope` tag 0.28 → 0.89 → 0.97 across 0.065 / 0.39 / 2.97 MP).
- **Practice:** keep sampling near ~1 MP (SDXL's native) and upscale afterwards.

### Subject pixel area: the character box must clear ~700×1200

Independent of canvas size — this governs a different failure mode, so the two do not contradict each other:

| what you control | the failure it causes |
|---|---|
| **canvas** size (above) | sampling above the training resolution ⇒ invented detail (extra fingers) |
| **subject** pixel area (this rule) | too few pixels on the figure ⇒ mushy render, and **features — the face above all — drift off-model** |

- **Rule: the character's bounding rectangle ≥ `700×1200` px.** Below that the figure is blurry and the face stops looking like the character.
- **Scene shots are the dangerous case:** a figure standing in a landscape occupies a much smaller share of the frame than a portrait, so a canvas that is fine for a 立绘 is not fine for a scene. Raise the generation resolution to match — character covering a third of the frame height ⇒ canvas ~3× taller to still give her 1200 px.
- **Iteration vs. delivery:** low resolution is the efficient way to explore composition and pose; anything kept for delivery must be **upscaled AND have the character region repainted** — upscaling alone does not restore a face that was never resolved. Repaint: `references/local_repaint_bigcanvas.md`. The graph is **bit-exact outside the mask** (`max Δ 0`) as long as the crop is **not rescaled** and `crop_mask` is wired into the stitcher's `mask`; drift comes from `SmartImageCrop` resizing the crop (0.0542 % with no sampling at all) and from leaving that `mask` unwired (0.3352 %) (`[measured]`, cases P1–P7).

Recipe: trial at low res to settle framing → re-render the winner at a canvas big enough that the character box clears `700×1200` → face/hand region repaint if still soft.

## Anima workflow — Cosmos-Predict2-2B (anime)

Anima is a **2B** model from CircleStone Labs × Comfy Org, architecturally derived from **Cosmos-Predict2-2B-Text2Image** (it reuses the **Qwen-Image VAE** + **Qwen3-0.6B** text encoder; it is **not** qwen-image architecture). Not a standard checkpoint: UNET, qwen CLIP and qwen VAE load separately, then LoRAs. Reference graph: `references/anima_workflow.json`. **Weight tiers (base / aesthetic / turbo), the turbo draw route and all tier timing: `references/anima_turbo_route.md`.**

- **UNETLoader** `unet_name=anima-base-v1.0.safetensors`, `weight_dtype=default` → `[0]=MODEL`.
- **CLIPLoader** `clip_name=qwen_3_06b_base.safetensors`, `type=stable_diffusion`, `device=default` → `[0]=CLIP`. (Reference templates use `stable_diffusion`; the `qwen_image` type exists but the saved workflows use `stable_diffusion`.)
- **VAELoader** `vae_name=qwen_image_vae.safetensors` → `[0]=VAE`.
- **LoraLoader** `model=unet[0]`, `clip=clip[0]`, `lora_name=<path>`, `strength_model≈0.85`, `strength_clip≈0.8` → `[0]=MODEL, [1]=CLIP`. (Values from the reference templates — see "Finding a LoRA" for why **1.0/1.0** is the default to reach for.)
- **CLIPTextEncode** ×2 → `clip = lora[1]`.
- **EmptyLatentImage** → `1024x1024`, batch 1.
- **KSampler** `model=lora[0]`, `sampler_name=euler`, `scheduler=simple`, `steps≈30`, `cfg≈8`, `denoise=1`.
- **VAEDecode** `samples=ksampler[0]`, `vae=vae_loader[0]`.
- **PreviewImage** (default); `SaveImage` only on explicit request.

**Picking a tier — pick by constraint, not by an assumed quality order:** many compositions to try ⇒ **turbo** (`anima-turbo-v1.1.safetensors`, 8–12 steps, **CFG 1**); training a LoRA ⇒ base; few parameter tweaks ⇒ aesthetic. **CFG 1.0 = the uncond branch is never computed, so the negative prompt has zero effect** (mechanism fact; same seed, standard vs empty negative decode pixel-identical, MAD 0.0000, 2/2 seeds) — `worst quality…` boilerplate does nothing there. **CFG 1.1–1.5 only changes the picture, never toward the negative words** (attribute probabilities unmoved, +49 % time per image); to actually delete content use **CFG≈2.0** and accept a flatter look. Tier-for-tier "quality" is a single-observation result, not a rule: gate the specific delivery image with your eyes (`references/anima_turbo_route.md` §单次观察 · §CFG × 负面).

**Anima LoRA: DeepSeek 鲸鱼娘**
- `lora_name = character\E9B2B8E9B1BCE5A898Fix.wTK8.safetensors` (card title `鲸鱼娘_fix`)
- Base `anima` (trained on `animayume_v04`), VAE `qwen_image_vae`, network module `networks.lora_anima`.
- Trigger tags (`ss_tag_frequency`): `1girl, blue hair, very long hair, wavy hair, ahoge, sidelocks, hair ornament, hair flower, blindfold, covered eyes, cetacean tail, fish tail, fins, head fins, animal ears, white dress, frilled dress, off-shoulder dress, high-low skirt, choker, frilled choker, blue bow, arm garter, thigh strap, barefoot`.

**Model choice:** **Anima** — tag-style prompt (may mix natural language), anime/illustration, diagram-like compositions hold; **text in image is weak** (single words, occasionally a phrase) ⇒ cloud for lettering (`references/cloud_edit_and_merge.md`); turbo for draws and composition tests. **Illustrious** — tag-style prompt, anime characters + LoRA (trigger + feature tags), 2D 立绘. Content that is 图解/概念/信息图 ⇒ Anima; 角色/立绘/具体人物 ⇒ Illustrious + LoRA.

**Canvas sizes by use** — 图示/信息图/漫画 `768×768`; 幻灯片/横屏 `1280×720` or `1344×768`; 头像 `1024×1024`; 竖屏 9:16 `832×1216` / `896×1152` (`1080×1920` valid, slow on 6 GB); 高分辨率 `1216×1792` then downsample. ⚠ **These are canvas sizes, not figure sizes** — compute the character's pixel share first: **人物矩形必须 ≥ `700×1200`**, else the face drifts (2 lines above). Full tables, prompt prefixes, negative templates: `references/prompt_cheatsheet.md`.

### Series / batch generation (consistency)

- **Fix one style prefix** (e.g. `flat vector illustration, clean simple minimal, soft pastel, centered`) and vary only the subject words per image.
- **Seed base:** use a fixed/contiguous seed set for the series (`seed = base + i`) to prevent style drift.
- **Batch:** submit several workflows, or loop submit→poll→save and rerun failures; `scripts/comfy_generate.mjs` or a loop script.
- **Check unity on the previews**; if they drift, adjust the style prefix or the seed.

### Classic / well-known characters: a LoRA may be unnecessary

1. **Find the official design / prompt first** — Danbooru-style tag sources (the model's native vocabulary; spelling and renderability follow it) + 萌娘百科 / Pixiv 百科 / 百度百科 / 维基百科 for setting facts (costume, props, aliases, romanisation) + the LoRA sidecar. Take the accurate appearance description (hair colour/style, eye colour/shape, costume, accessories, build). Source ladder and 资料→tag translation rules: `references/character_features.md`.
2. No ready prompt ⇒ **find a few images of the character**, gate them through `libs/wd14-gate/scripts/wd14_gate.py` **before looking** (`BLOCK` ⇒ `dsh_show_gallery` for the human, or ask the user to describe it), then write the description yourself item by item (hair / hair ornament / eyes / costume colours and pattern / accessories).
3. Use **`English name + appearance description`** (natural language for Anima, tags for Illustrious), e.g. `Nahida, short white hair with green accent braid, big green eyes with leaf-shaped pupils, pointed elf ears, white dress with green and gold details, white bloomers, mostly bare legs, white stirrup shoes`.
- **Never guess or hard-code costume details** — check the setting first; a wrong costume decides "does it look like her".
- **Keep a LoRA as the fallback / style anchor**: try prompt-only first, add the LoRA when the result does not hold or the style must be unified.

## Production pipeline — the macro order（先画什么 → 何时编辑 → 何时放大）

1. **Draw the base** — **Anima, small `800×1200` tier, 3–5 images**, keep the one with the right structure (count / relation / pose / camera / occlusion); turbo (10 steps CFG 1) saves 4.7× time at this step. **Do not polish a wrong picture** — keep drawing. Multi-person: wide canvas (duo `1216×832`), one appearance block per character, negative `1boy, male`; **a LoRA never crosses bases**.
2. **Large structural fixes** — limb positions, orientation, who is where: human edit + repaint (`SetLatentNoiseMask` 0.5–0.6). The hand-edited file *is* the canvas; changes go into the prompt. **This output is the base for every later step.** Per-region repaint discipline (only the current character's LoRA, simplify the other to `another girl …`, no cross-base, same-engine passes back to back) and "**prompt only what is visible on the canvas**" (crop-type vs full-canvas mask-type): `references/local_repaint_bigcanvas.md` §"The prompt must describe only what is inside the crop".
3. **Upscale to the `1200×1800` tier → whole-image `denoise 0.3`** for detail. ⚠ the whole-image route is not bit-exact outside the mask.
4. **Model split** — want **IL's look** ⇒ edge map (canny/softedge) + **IL img2img `0.8` + ControlNet**; have an **IL-only character LoRA** ⇒ **IL img2img `0.6`**. Changing engine always requires pasting back.
5. **Detail fixes** — **PS + local repaint**: fingers, toes, `vibrator`, rope (one issue per pass, 0.2–0.4).
6. **Upscale to delivery size** ⇒ ship, or to the **SmartCrop workflow** for pupil-level detail.

**Four disciplines**: ⓪ **embed every step's product in the conversation with `dsh_show_media`** (a path alone is invisible to the user; use `compare_with` to show the old version beside it); ① **many images only at step 1** — one per pass afterwards (small-denoise samples differ only marginally), and only `denoise > 0.6` (e.g. switching to IL) justifies several; ② **"upscale" happens more than once**, because fingers/toes/pupils cannot be drawn without pixels — each upscale serves the next detail scale; ③ **denoise decides "read" vs "rewrite"**: 0.2–0.4 material / 0.5–0.6 proportion and engine switch / 0.8–0.9 structural topology (a rope "connected to the bed frame" does not appear at 0.4–0.75, only at 0.9) / 1.0 redraw. Material *colour* changes via prompt, not via more denoise.

## Writing the prompt（硬规则 5：从简）— structure, negation, costume swaps

Structure (grouping, blank lines, negation, costume swaps), **concision (one intent said once, §2b: length dilutes)**, Anima-specific adjustments, A/B results and the 536-prompt corpus statistics: **`references/prompt_writing.md`**. Copy-ready positive prefixes, four negative templates, 8 skeleton templates and word lists: **`references/prompt_cheatsheet.md`** (open it before writing; fixed-defect block §2, costume words §3, per-pose tag table §4).

**Hard rule 4 — tag review before every submit:** write the scene manifest first, then check every tag against it for contradictions and residue. New subject matter is checked from scratch; any change to costume / pose / prop / location / viewpoint / aspect ⇒ mandatory. A read-through does not count as a review (**token ≠ awareness of token contradiction**). Full table and steps: **`references/tag_review.md`**.

## Blender-driven control maps — depth / pose / hands

When pose, framing, focal length or foreshortening must be exact, author the scene in Blender and render control maps instead of describing it in words. Depth and pose come from **the same camera and the same armature**, so they stack.

**Entry, four steps:** `libs/blender-controlnet/scripts/blender_build_scene.py` (needs `--factory-startup`; emits depth + clay + kps) → `libs/blender-controlnet/scripts/blender_draw_openpose.py` → `libs/blender-controlnet/scripts/blender_overlay_check.py` / `libs/blender-controlnet/scripts/blender_crop_hands.py` (**verify the skeleton sits on the geometry before spending GPU**) → `scripts/comfy_control.mjs`. Four secondary entries: copy an existing skeleton, multi-character staging, read back a GUI edit, local repaint.

Full flow, `scenes.json` format, pitfalls and all measured numbers (denoise band table, hand/face repair limits, the three FaceID weight files, the same-view-only 87% phenomenon, 6 GB timings): `libs/blender-controlnet/references/blender_control_pipeline.md`.

## Cutting the background out (matting / alpha)

Pick the route by **input type** — difficulty differs by an order of magnitude: our own render with `plain white background` in the prompt → `libs/sam-matting/scripts/matte_flat_bg.py`; someone else's illustration/poster → `libs/sam-matting/scripts/matte_from_scribbles.py` (needs human FG/BG scribbles) or `SAM3_Detect`; solid-colour backdrop → `libs/sam-matting/scripts/chroma_key.py`. Route table, measured numbers and scoring: `libs/sam-matting/references/matting_and_cutout.md`.

## Upscaling (2x / 4x) — after generation, before delivery

Chain: **`UpscaleModelLoader` → `ImageUpscaleWithModel` → `PreviewImage`** (lands in `temp\`; move it into the workspace explicitly to keep it). No diffusion, seconds. Default `4x-AnimeSharp` then Lanczos to the target width; delivery sizes and the sharpness proof: `references/upscaling.md` (model download URLs, four-amplifier comparison, the "2x model removed" decision).

## Comic page — tiling N panels into one page

When the deliverable is a **page** (N panels, straight or diagonal dividers, no white margin inside a panel, reading order): `libs/comic-layout/scripts/comic_layout.py` searches and renders the panels, `libs/comic-layout/scripts/letter_page.py` handles bubbles, caption-box search and anchor solving. Design notes (recursive one-cut per convex cell — valid at any angle, union is exactly the page) and every parameter: `libs/comic-layout/references/comic_page_layout.md`.

## Cloud APIs — generation and editing (and merging the answer back)

Local ComfyUI is the default; go to the cloud only where the local stack is **structurally** incapable: text inside the image (CJK calligraphy/signage), adding objects on a hand-edited base, scene-level cleanup. Which vendor, where the key lives, how to pass reference images, how to paste the answer back by mask: `references/cloud_edit_and_merge.md` (including the two rules the user insists on: **always supply the reference image**, and **follow the instructions verbatim**).

## Human in the loop — `human_canvas` (rough 2D) and `ps_edit` (real Photoshop)

The human owns aesthetics and intent; the agent owns measurement, mechanics and verification. Both channels are **non-blocking**: open the task, say exactly what to do, **yield the turn**, read back when they reply. Use the lighter channel — canvas for anything you can refine programmatically, Photoshop only when the job genuinely needs it.

- **Ask for two layers, not one:** `L1 = repaint`, `L2 = keep`. The keep layer is the human's protection switch (fingers, a hair ornament, an already-good region); mask = `L1 − L2`.
- **Paint on a zoomed crop** (2–3×) when the target is small — the same stroke lands 2–3× more precisely in original coordinates.
- **Look at the mask yourself before spending GPU time** (tint it over the picture, report area % and bbox). That catches "they painted the whole head, not the eye" in one round.
- **`ps_edit` is verified by content, not by the clock:** `open` returns mtime + sha256, the human edits and saves **the same path**, `check --since_sha256` decides. **Archive the human's hand-edited file** — its difference against the pre-edit base is the *exact* mask (`local_repaint_bigcanvas.md` §"Get the shaped mask for free"). Never rebuild a hand-edited base from a pre-edit version.

Playbook, canvas read-back schema, PS handoff steps, division of labour: `references/human_in_the_loop.md`.

## Locating things — grids, ASCII maps, compare sheets, colour masks

"Where exactly is it?" is a **measurement**. Guessing a bbox wastes a GPU run at best and eats a neighbouring part (a finger, a hair strand) at worst. All CPU-only, all seconds:

| Tool | Gives you |
|---|---|
| `crop_view.py IMG out.png x0,y0,x1,y1 --scale 1.5 --grid 100` | a crop with a grid **labelled in original coordinates** — so "around x 880, y 810" is actionable for a human |
| `ascii_map.py IMG x0,y0,x1,y1 --step 4` | the region as text (R = target, S = skin, `.` = background) — shows which pixels are rope vs. fingers without opening a viewer |
| `sheet3.py out.png x0,y0,x1,y1 --labels a,b,c IMG...` | N versions of the **same region at the same scale** in one sheet — the only honest way to choose between denoise 0.3 and 0.5 |
| `red_mask.py IMG mask.png --hue-max 16 --sat-min .6 --drop …` | a colour mask by **HSV hue** + area/position filtering |
| `diff_report.py A B --mask M` | in-mask vs out-mask change — the number that proves an edit is confined |

Two traps with measured consequences: **`R − max(G,B)` is not a redness test** (a warm brown floor and a dark-red cushion both pass it — 3.86 % of the frame "red"); **"dark = ink" fails on a night scene** (the paper's own median luminance is 50). Use HSV hue with saturation/value floors, then filter connected components by area and position. Detail: `references/grid_localization.md`.

## Checking a render — 人/手/脚/指趾/口塞/掩版 self-check (anime_detection_suite)

Pretrained, callable directly. The suite is **not** copied into this skill: the script calls the suite by absolute path (code and weights live there, so a suite update takes effect here immediately). It answers: **people/hands/feet boxes found, digit counts suspicious, gag/blindfold present, rope mask present.**

```bash
# any cwd; single image or directory
python "<你的用户目录>\.dsh\skills\comfyui-image-generation\scripts\check_parts.py" ^
  --src <图或目录> --out <输出目录> [--mask] [--person-masks] [--recipe rec] [--json]
```

| Argument | Meaning |
|---|---|
| `--src` | image file or directory (several allowed) |
| `--out` | output dir (default `./part_check_out`): the suite's raw contract JSON + `summary.json` |
| `--recipe prod\|rec` | `prod` (default) = current production setting (imgsz 1280 + current M3 dual weights); `rec` = measured better (imgsz 1600 + the three leak-free M3 seeds, **not yet the production default**) |
| `--imgsz` / `--m3-conf` / `--foot-crop` / `--hand-weights` / `--m3-weights` / `--no-foot-tight` | passed through to the suite entry (`locate.py --help`) |
| `--mask` | hand/foot mask PNG (suite behaviour) |
| **`--person-masks`** | **per-person silhouette** (multi-character per-person repaint; see below) |
| `--person-mask-model` | `sam` (default, this skill's SAM ViT-B) / `hull` (M1 keypoint convex hull only, no SAM) |
| `--person-mask-prompt` | `pose` (default) = skeleton bounding box + torso positive points; `box` = detector person box only |
| `--person-mask-cover` | `auto` (default, **cover generously**) / `tight` (edge-hugging, does not touch props) / `full` (guarantees the whole skeleton inside) |
| `--grow N` / `--no-fill-holes` / `--no-clip-to-pose` / `--no-merge-overlap` / `--person-min-prob` | the four mask post-processing switches |
| `--ignore-cls` | ignore classes in the summary, e.g. a blindfold the design requires: `--ignore-cls blindfold` |
| `--json` | one machine-readable JSON line (for agent parsing) |

**Reading the result:** `可疑部位` = parts needing **human review**, not "the render failed"; `收紧复核` is the tightened flag for toes (low priority); `看不到手 / 无人` is a composition hint; `数不了N处` = shoes/socks (digit counts impossible — normal). `summary.json` reports per image: `suspicious / tight_flags / gag_blindfold (box+prob) / n_hands / n_feet / hands_state(visible|not_visible|zero_detected) / rope / persons / persons_masks / seconds*`.

### Per-person masks (the structural fix for multi-character attribute bleed)

"Who is bound / who is free, who has black hair / white clothes" cannot be held by a prompt. The structural answer is per-person masked repaint — one pass per person, the other person's pixels frozen. One command produces the masks:

```bash
python scripts/check_parts.py --src BASE.png --out OUT --person-masks --json
```

Products in `OUT/mask/`: `<name>_person0.png` (255 = repaint region) and `<name>_person0_rgba.png` (**repaint region alpha=0**, because ComfyUI's `LoadImage` MASK = 1 − alpha). How the mask is decided (overlap auto-split, head included, `person_*` field meanings) and the full per-person pass: `references/local_repaint_bigcanvas.md`.

### Local feature repaint + underpaint

- `check_parts.py --feature-masks` gives box masks for hands / feet / head / face / gag / blindfold / eyes, plus `mouth*` = lower 42 % of the face box (12 % inset horizontally) — **COCO-17 has no mouth point**, so a mouth repaint can only be derived from the face box. (With M1 on the fine-tuned ViTPose, nose/eye/ear are trustworthy — median face error 15.9 px on the report set — but **face points still drift in back and profile views**, so the rule "derive mouth/face from the face box, never from keypoints" stands.)
- `libs/sam-matting/scripts/mask_compose.py` is the agent-in-the-loop mask combiner (set union/difference against an existing mask) — you do not have to get it right in one shot.
- The three underpaint routes and their mask discipline: `references/underpaint_and_stencil.md`.

### Two unknown characters in one frame that must not swap sides (the standard solution)

1. Draw the **layout base with no character LoRA at all**; state orientation in **screen coordinates** only; name `mirrored composition`, `swapped positions` in the negative. 2. `check_parts.py --person-masks` for one non-overlapping mask per person. 3. **Attach one person's LoRA per pass** (1.0/1.0 + the sidecar's full trigger list) at `denoise 0.65~0.75`, twice serially. 4. Every repaint prompt restates the contact relation. Four steps and evidence: `references/prompt_writing.md`.

## Finding a LoRA and its trigger words

Search, download, base matching, verification and placement (civitai wants `?query=` not `?search=`, direct shell access returns 403, hex filenames): `references/lora_sourcing.md`. Below is only how to read the trigger words once you have the file.

- **Keep `strength_model` and `strength_clip` at 1.0.** Interpolability in latent space is *empirical, not a strict property*: a LoRA is a trained perturbation, and scaling its strengths does not guarantee a smooth blend — it can quietly degrade identity, costume or rendering while a thumbnail still looks "close enough". Deviate only when the LoRA **demonstrably damages the picture** (burned colour, broken anatomy, style collapse), and then in this order: fix the prompt first (a conflicting tag can be what the LoRA is fighting), try another LoRA, lower strength last.
- **Use the sidecar's *whole* trigger list, not the few words you remember.** A missing descriptor is a wrong picture, and it is indistinguishable from "the LoRA is weak" — the missing tag silently loses, and the model then fights your substitute words (`white dress` vs the LoRA's `purple dress`) instead of following them.
- **List available LoRAs**: `GET /object_info/LoraLoader` → `required.lora_name[0]` (paths relative to `models/loras`). The `lora_name` must match exactly, e.g. `character\E9B2B8E9B1BCE5A898Fix.wTK8.safetensors`.
- **Card / title / base model**: the LoRA-Manager sidecar `<name>.metadata.json` beside the `.safetensors` holds `model_name`, `base_model`, `preview_url`.
- **Trigger words / tags**: read the safetensors header metadata — the first 8 bytes are the **little-endian uint64** JSON-header length; parse that JSON, then `__metadata__`: `ss_tag_frequency` (the training tag vocabulary — build the prompt from these), `ss_sd_model_name` / `ss_base_model_version` (required base), `ss_vae_name` (expected VAE), `ss_network_module` (e.g. `networks.lora_anima` ⇒ special pipeline), `modelspec.title` (display name).
- **`ss_tag_frequency` CAN be empty** — many Civitai character LoRAs keep the trigger elsewhere. Check the sidecar's **`civitai.trainedWords`** (and `civitai.model.trainedWords`); that holds the real trigger + descriptor tags (e.g. `hu_tao_genshin_impact, brown hair, hat, long hair, ...`).
- **Best source of all: `civitai.images[].meta.prompt`** — the sidecar stores the generation prompt of each preview image, i.e. the author's own tag table for that character, and it is more complete than `trainedWords`. One line to read it:
  `for im in (m.get("civitai") or {}).get("images", []): print(im["meta"].get("prompt"))`
- **Finding the word is not the same as suppressing it.** Negating a character's baked-in prop (`lanyard, id card, name tag, …`) did not remove it — **character priors outrank the negative prompt**. If a negative produces no visible change, the element is baked into the LoRA's default look: change something else (costume, pose, crop) or edit the pixels.
- **Author-recommended sampler/steps/CFG** often live in the same sidecar's `civitai.model.description` (e.g. "Steps 30 / CFG 5-7 / DPM++ 2M SDE Karras"). Prefer those over the generic defaults.
- **Match the LoRA's base pipeline**: a LoRA trained on Anima (`anima`) goes in the Anima workflow; one trained on Illustrious/SD goes in the IL workflow. The wrong base yields poor output.

## Troubleshooting

ComfyUI submit/load failures, Blender exit codes, LoRA bleed, upscaling and cloud-API failures, with the diagnosis order: **`references/troubleshooting.md`** (each entry is "what you see → where to look"). The two red-line cases stay above, in Safety gate and Cleaning up after a run.
