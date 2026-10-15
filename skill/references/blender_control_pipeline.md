# Blender-driven control pipeline (depth / pose / hands)

Author a pose and camera in **3D**, render the control maps, and let ControlNet /
LLLite carry that structure into the generated image. Pose, framing, focal length
and foreshortening stop being prompt-word guesses and become values you set.

Verified end-to-end on this host (Blender 5.1 + ComfyUI 0.24 + RTX 3060 6 GB).

## Why use it

- **Perspective that prompts cannot express.** A 28 mm low-angle shot, a
  wide-lens floor convergence, a foreshortened forearm pointing at the camera —
  all of it is decided by the camera transform and survives into the render.
- **Pose is exact, and reproducible.** The pose lives in a JSON file, not in a
  seed lottery.
- **Depth and pose maps cannot disagree.** Both are projected through the same
  camera from the same armature, so stacking them is safe.
- **Hands.** A body-only OpenPose map leaves the model to invent fingers, which
  is where anime generations usually fail. Adding the 21-point hand skeleton per
  hand measurably fixes it (see "Did the hands help?").

## Machine facts

| Thing | Value |
|---|---|
| Blender | `<你的 Blender 安装目录>\blender.exe` — 5.1.0, bundled Python 3.13 + numpy |
| Headless invocation | `blender.exe --background --factory-startup --python <script> -- <args>` |
| PIL (skeleton drawing) | `python` (PIL 12.2, numpy 2.4) |
| ComfyUI | `http://127.0.0.1:7187` (see SKILL.md for launch) |

**Always pass `--factory-startup`.** Without it Blender loads user extensions and
dies on a permission error writing `...\Blender\5.1\extensions\user_default\...`.

Even with `--factory-startup`, headless Blender still tries to write its extension
cache to `%APPDATA%\Blender Foundation\...\extensions\.cache\compat.dat`. Under the
file sandbox that write is denied, so **Blender prints its renders, saves every
file, and then exits 1**. Do not treat that exit code as failure — check that the
expected files exist. To get a clean exit code, redirect the cache into the
workspace:

```powershell
$env:BLENDER_USER_EXTENSIONS = "<你的 ComfyUI 安装目录>\agent_work\blender_ext"
```

## Pipeline

```
scenes.json
   │  blender_build_scene.py --spec scenes.json --out out
   ├──► <scene>_depth.png   depth, near = white
   ├──► <scene>_clay.png    flat-shaded preview (human verification only)
   └──► <scene>_kps.json    COCO-18 + 2x21 hand keypoints, image space
              │  blender_draw_openpose.py <kps.json> <pose.png>
              └──► <scene>_pose.png   OpenPose skeleton
                        │
                        ▼
   comfy_control.mjs jobs.json ──► ControlNet / LLLite / img2img ──► final PNGs
```

## 1. Author the scene — `references/blender_scenes_example.json`

### Bones are world-space directions, not Euler angles

`bones` maps a bone name to the direction its **tail should point from its head**,
in world space. The script solves the local basis via `pose_bone.matrix`, walking
parents first. This is readable, roll-independent, and needs no inverse
kinematics: to raise an arm, write the direction it should point.

```json
"upper_arm.L": [0.45, -0.55, 0.70],
"forearm.L":   [0.30, -0.50, 0.81]
```

Unlisted bones keep their rest orientation (they still inherit the posed parent),
so a scene only specifies what differs.

Character convention (do not change casually — keypoint mapping depends on it):
character faces **-Y**, up is **+Z**, so the person's **left is +X** and `.L`
bones map straight onto OpenPose `left_*` without mirroring.

### Hands

```json
"hands": { "L": { "curl": 0.40, "spread": 0.0 }, "R": { "curl": 0.25 } }
```

`curl` 0→1 is relaxed→full fist (per-phalanx flexion angles live in `_HAND_CURL`).
`spread` is applied only at the knuckle, otherwise every phalanx twists.
Hands are **real bones** (15 per hand: thumb + 4 fingers × 3 phalanges), so the
mesh, the depth render and the OpenPose hand skeleton all follow one pose.
Finger bones are rolled with `align_roll(palm_normal)`, which makes local +X the
flexion axis — curling is a single `rotation_euler.x`.

### Camera / environment / root

- `camera: {location, target, lens}` — `look_at` aims it. `lens` in mm on a 36 mm
  sensor; portrait framing is covered by `2 * distance * 18 / lens` metres, so
  pick the lens from the subject height rather than guessing.
- `env: {floor, floor_size, boxes: [{loc, scale}]}` — **per scene**, because a
  stool belongs under a seated pose and not under a standing one. Props are
  rebuilt per scene; stale props would leak into the depth map.
- `root: [x,y,z]` — translates the whole character (sitting, crouching).
- `resolution: [832, 1216]` — **must equal the generation resolution**; the
  control map is fed 1:1.

## 2. Control maps

### Depth — an emission material, not a Z pass

The depth map is produced by an `Emission` shader driven by
`ShaderNodeCameraData → View Z Depth → MapRange(near..far → 1..0)`. Rationale:
no compositor / render-pass plumbing (which churns across Blender versions), and
the range is an explicit number you control.

**Anchor the range to the subject, not the scene.** `world_bounds_z()` is called
with the mannequin only: with a 24 m floor in range, the character collapses into
a flat mid-gray slab. Also set `view_transform = 'Standard'`, or the view
transform gamma-chews the gradient.

### Keypoints — COCO-18 body + 2×21 hands

`collect_keypoints()` projects posed bone head/tail positions through
`world_to_camera_view` into image space. Hand format is OpenPose standard:
21 points = wrist + (base, joint1, joint2, tip) × 5 fingers, which falls directly
out of the 3-phalange bone chains.

**Invariant:** pose_bone.head/tail are ARMATURE-space. Multiply by
`arm_obj.matrix_world` before projecting, or any scene that sets `root` drifts.
This bug is silent whenever root is zero.

`blender_draw_openpose.py` draws the body with the standard 18-color limb palette
and the hands with the standard 20 edges at half stroke width.

## 3. Verify BEFORE generating

Cheap checks that catch geometry bugs the generation would only blur:

```bash
python scripts/blender_overlay_check.py out/s1_clay.png out/s1_pose.png out/s1_overlay.png
python scripts/blender_crop_hands.py s1        # -> out/s1_hands_zoom.png
```

- `overlay_check` blends the skeleton onto the clay render. Any offset is a
  projection bug, not a pose difference.
- `crop_hands` crops ±150 px around the wrist keypoints and upscales 3×. Hands
  are ~20 px wide in a full-body 832×1216 frame; at 1:1 you cannot tell a correct
  hand from a broken one.

## 4. Generate — `scripts/comfy_control.mjs`

```bash
node scripts/comfy_control.mjs references/blender_control_jobs_sdxl.json
```

Job file: `{pipeline, base|unet/clip/vae, outDir, jobs[]}`. Each job carries
prompt, seed, size, sampler settings, `controls[]`, and optional
`init: {image, denoise}` for img2img. The script uploads control images to
ComfyUI's `input/dsh_ctrl` subfolder (delete it afterwards) and pulls results
through `PreviewImage → /view?type=temp`, so **nothing is written to ComfyUI's
`output/`** and the workspace stays clean.

### Available control models on this host

| Model | Pipeline | Size | Use |
|---|---|---|---|
| `noob_sdxl_controlnet_depth.safetensors` | SDXL / Illustrious | 4.66 GB | **perspective + volume, best overall** |
| `openpose_pre.safetensors` | SDXL / Illustrious | 2.33 GB | exact joints, incl. hands |
| `anima-lllite-pose-1.safetensors` | Anima (`AnimaLLLiteApply_sdscripts`) | 0.02 GB | light, native to Anima |
| `anima-lllite-scribble-1.safetensors` | Anima | 0.01 GB | edge/scribble maps |

SDXL base: `Illustrious\anime\waiIllustriousSDXL_v170.safetensors`.
Anima base: `waiANIMA_v10Base10.safetensors` + `waiANIMA_v10Base10_txt.safetensors`
(CLIPLoader `type=stable_diffusion`) + `qwen_image_vae.safetensors`.

### Measured settings (832×1216, RTX 3060 6 GB)

| Mode | Weights | Time |
|---|---|---|
| SDXL + depth CN | depth 0.8 | 85–105 s |
| SDXL + pose CN | pose 0.9 | 47–70 s |
| Anima + LLLite pose | strength 1.0 | ~60 s |
| img2img from clay | denoise 0.82 | 41–49 s |
| depth + pose stacked | depth 0.7 / pose 0.7 | 160–171 s |

Two 4.66 GB-class control nets plus SDXL exceed 6 GB VRAM; ComfyUI offloads and
it still works, just slowly.

### img2img from the clay render

Feeding `<scene>_clay.png` as the init image reproduces the 3D composition almost
exactly. **Denoise matters:** at 0.62 the gray clay palette survives and the
output is monochrome; 0.82 regenerates color while keeping the composition. Add
`mannequin, doll, 3d render, plastic` to the negative prompt.

## Did the hands help?

Same seed, only the control map changed (body-only vs +hand keypoints), 4 cases:

| Case | Body-only | + hand keypoints |
|---|---|---|
| hand on hip | blob, fingers merged | four fingers articulated on the skirt, thumb tucked |
| hand hanging | merged into the skirt | joints and thumb readable |
| open hand raised | roughly right, angle off | matches the skeleton orientation |
| low angle, far hand | **featureless mitten** | five fingers separated |

The worst case went from unusable to correct. Note this is a property of the
control model: it only helps if the ControlNet was trained with hand keypoints.
Always A/B it rather than assuming.

## 5. Ingest an existing pose (2D skeleton → 3D)

Use when the target pose already exists as a 2D skeleton (an OpenPose render, a
photo run through a pose preprocessor, another tool's output) instead of a JSON of
bone directions. Three scripts, all CWD-relative:

```bash
# a) skeleton image -> 18 keypoints (+ visual check)
python scripts/blender_extract_kps.py ref/skeleton.png ref/target_kps.json --overlay ref/check.png

# b) hand-author an initial 3D pose in scenes.json, then fit it to the target
blender --background --factory-startup --python scripts/blender_fit_pose.py -- \
    --target ref/target_kps.json --spec scenes.json --scene dance_front --out scenes_fitted.json

# c) score any candidate (rendered kps or fitted kps) against the target
python scripts/blender_pose_error.py ref/target_kps.json out_fitted/fitted_kps.json --overlay cmp.png
```

**Why extraction is easy:** most OpenPose renderers draw limbs at ~60 % palette
brightness and joint dots at full brightness. So an exact-colour match on each
palette entry isolates exactly one dot; take the largest connected component and
its centroid. Verified on a 1600×1600 COCO-18 render: 18/18 joints, one component
each, sub-pixel clean.

**Why the metric is meaningful:** `blender_pose_error.py` Umeyama-aligns the two
point sets (scale, rotation, translation free) and reports RMS as a percentage of
the target's radius plus a per-joint breakdown. Framing is therefore free and only
pose shape is penalised.

### Getting the target from a photo — use ComfyUI, don't hand-roll it

If the pose source is a *photograph* rather than a skeleton image, ComfyUI already
ships the detector. No colour extraction is involved:

```
LoadImage → DWPreprocessor(detect_body/hand/face = enable, resolution = 512)
          → SavePoseKpsAsJsonFile(pose_kps)   # writes output/<prefix>_0000N.json
          → PreviewImage                      # rendered skeleton
python scripts/blender_dwpose_to_kps.py output/dsh_pose_photo_00003.json ref/target_kps.json
```

`pose_keypoints_2d` is already COCO-18 in the same order this pipeline uses, so the
conversion is a straight repack; the JSON also carries 70 face points and both
hands. `blender_draw_kps_on_image.py <photo> <kps.json> <out.png>` draws a detection
back over the photo — that is how you tell a bad detection from a bad metric in one
look.

- **Which detector:** on a 172×270 stage photo `OpenposePreprocessor` (OpenCV DNN)
  returned `people: []` even at `resolution=1024`; `DWPreprocessor` found the person
  and 17/18 joints. Prefer DWPose for small/low-res input. It falls back to CPU here
  (onnxruntime has no GPU provider) but a 512 px image still takes only seconds.
- **Confidence is not optional.** DWPose reported `left_ankle` with score 0.00 and a
  garbage off-frame coordinate for the leg hidden behind the tutu. Both
  `blender_pose_error.py` and `blender_fit_pose.py` drop joints below 0.3, so carry
  `confidence` through the conversion (the script does).

**⚠ 2D comparison is same-view only.** The metric measures image-plane shape.
Control test: the *same* 3D pose rendered from the front and from behind scores
**87.1 % against itself**. Therefore:

| Reading | Meaning |
|---|---|
| ~5–10 % | same viewpoint, good fit |
| 50–90 % | different viewpoints — says nothing about the pose |

To compare poses across viewpoints, fit both to 3D and compare bone directions
instead. Worked example on this host: the photo's DWPose pose scored 56.8 % against
a *back*-camera render of the fitted pose but 87.1 % against a *front*-camera render
of that same pose — i.e. the back camera is the better viewpoint match, and the
number itself was never evidence about the pose.

**Why fit inside Blender:** the fitter uses the *same* armature, `aim_bone` FK and
`world_to_camera_view` projection that render the control maps, so a fitted pose
renders exactly as scored — no reimplemented FK to drift out of sync. It is also
fast enough that reimplementation would be pointless: **~5 ms per evaluation**.

Search = greedy coordinate descent over per-bone (azimuth, elevation) plus camera
distance, step shrinking ×0.72. Measured on an arabesque target: **27.0 % → 6.9 %
in 11 s** (39 parameters, 14 passes).

Depth is ambiguous from a single 2D view, so the hand-authored init is the prior:
the fit refines a plausible pose instead of flipping it inside out. Give it a
sensible init or it will happily return an anatomically impossible mirror.

**Residual floor = rig proportions, not pose.** The remaining 6.9 % sat almost
entirely on hips/shoulders/head (worst 48/38/52 px) because the mannequin's torso
and arms are short relative to a real dancer's. If a target figure has different
proportions, either accept the residual or parameterise bone lengths in
`_BONES`/`_RADII` — the mesh is generated from the same table, so both follow.

**Camera choice is free.** A front camera plus a fitted body azimuth can express any
reference viewpoint; the figure ends up rotated rather than the camera, which is
fine for control maps. Pick the *second* scene's camera afterwards to match a
photographic reference (e.g. a back-3/4 stage shot).

## 6. Multi-character staging — `scripts/blender_stage_multi.py`

`blender_build_scene.py` stages one mannequin. A two-character tableau needs N
armatures with their own root/pose/facing, props rigidly attached to bones, and
per-character keypoints so the OpenPose map can be multi-person. This script adds
exactly that and reuses everything else from `blender_build_scene` (rig, `aim_bone`
FK, projection, depth material) — one implementation of the geometry.

```bash
blender.exe --background --factory-startup --python scripts/blender_stage_multi.py -- \
    --spec scenes_battle.json --out out --masks [--save-blend scene.blend]
```

- `characters[]` — `{name, root, root_rot, bones, hands, facing}`; `facing` only
  feeds the synthesized face keypoints, so a character seen in profile or lying down
  does not get its face on the wrong side.
- `props[]` — `{owner, bone, offset, rot, size}`: a box parented to the owner's
  armature, so dragging the character in the GUI keeps the rifle in hand.
- `--masks` — renders `<scene>_mask_<char>.png` (white = that character + its props)
  and `<scene>_mask_env.png` (white = env boxes only, **floor excluded**). These are
  the priors for local repaint (§8).
- Keypoints output is `{resolution, characters: [{name, keypoints}]}`; `blender_draw_openpose.py`
  accepts both that and the old flat shape and draws every skeleton into one map.

## 7. Human-in-the-loop in the Blender GUI

When hand-authored poses still read wrong, hand the *scene* to the human instead of
guessing again. The loop:

```
agent: build + --save-blend → open the .blend in the Blender GUI
human: drag camera / characters / props (G/R/S), Ctrl+S
agent: scripts/blender_extract_blend.py → scenes_from_blend.json → rebuild + re-render
```

`blender_extract_blend.py --blend X.blend --spec Y.json --out Z.json` reads back the
camera (location + look-at target recovered from the view direction × 5 m, lens,
resolution), each `rig_<name>` (root location, `root_rot`, and every non-finger bone
direction in armature space) and every `env_box_*` (loc/scale).

Round-trip fidelity measured: rebuild from the extracted spec renders **max pixel
delta 27/255, mean 0.019** — i.e. identical modulo rounding.

**What the round-trip cannot carry: mesh-level edits.** Subdivision, joins, modifiers
and other mesh-data changes are invisible to an object-transform dump — a subdivided
"rounder sandbag" comes back as a plain box of the same size. Either accept that or
extend the extractor to dump mesh data.

## 8. Local repaint — fix one element without disturbing the rest

Whole-image regeneration re-rolls everything; when only one character's outfit or
one background element is wrong, repaint just that region.

```
libs/blender-controlnet/scripts/blender_make_inpaint_mask.py <base.png> <prior_mask.png> <out_mask.png> --grow 16 --feather 6
scripts/comfy_inpaint.mjs <jobs.json>
```

**Mask: prior → aligned to the generated image.** The Blender silhouette will not
line up pixel-exactly with the generation (pose and outline drift), so the script
erodes/dilates the prior into a GrabCut trimap, runs GrabCut **on the generated
image** (the boundary snaps to the real silhouette there), intersects with a dilated
prior to stop colour-similar leakage, then keeps the largest component, fills holes
and feathers. The prior's blind spots are inherited: the mannequin has no hair, so
hair is never inside a character mask.

**Repaint: latent noise mask, not img2img.** `LoadImage(base) + LoadImage(mask) →
VAEEncode → SetLatentNoiseMask → KSampler(denoise)`. Only the masked latent is
noised, so everything outside comes back untouched.

**Prompt simplification is the anti-pollution trick.** Describe only the region being
repainted; reduce everything else to an anchor tag:

| Repainting | Prompt | Measured |
|---|---|---|
| one girl's outfit (denoise 0.7) | her full description + `1girl at right side` | outfit changed, other girl byte-identical |
| empty sky → sandbag wall (denoise 0.8) | `sandbag wall, stacked sandbags, burlap sacks` + no character text | wall appeared, both girls and the stone wall untouched |

Denoise has a sweet spot: at 0.9 the sandbags came back as coiled hay-rolls; 0.8 held
the shape. Raise it only as far as the deviation demands.

**The prior cannot locate what the model re-invented.** A 3D sandbag wall
projected to the right edge, but the model rendered a large stone wall across the
upper frame — no 3D mask could cover it. For "the model reinterpreted the scene"
cases, the human paints the region on the generated image (`human-canvas`,
`dsh-man-in-loop`) and the same GrabCut + latent-mask path takes it from there.

## Known pitfalls

1. **Keypoints in armature space** — multiply by `matrix_world` (see above).
2. **Depth range over the whole scene** — anchor it to the subject.
3. **Relative `--out` resolves against `C:\`** in headless Blender; the script
   anchors relative paths to the spec file's directory.
4. **BOM breaks `json.load`** — PowerShell's `Set-Content -Encoding UTF8` writes a
   BOM; the spec is read with `encoding="utf-8-sig"`.
5. **Stacked ControlNets drift the framing** — keep depth ≥ 0.7, or the weaker
   constraint lets the model re-frame the subject.
6. **Gray init ⇒ gray output** — raise img2img denoise.
7. **`--factory-startup` is not optional** — see machine facts.
8. **`json.load(fh)` inside a comprehension** re-reads the file per iteration and
   fails on the second element — load once, then index.
9. **The floor is part of `build_env`** — masking "env" therefore masks the whole
   lower frame; filter to `env_box_*` when the mask means "props only".
10. **Two characters swap roles easily.** SDXL binds colour to the *nearest* pose
    tags, so keeping each character's colour + pose in one clause works, while
    negative prompts like "purple clothes on blonde girl" flip the assignment.
11. **Blender piped through PowerShell can silently skip the render** — write the
    log to a file (`*> log`) and grep it instead of piping stdout; also check the
    output file's mtime against the spec's before believing a render.

## Extending

- **Normal map / lineart** for LLLite scribble or softedge HED control.
- **Face keypoints** (OpenPose 70-point) — same projection path, add to
  `collect_keypoints`.
- **Per-finger orientation** — the fingertips currently carry position only, so
  the control map cannot express finger twist.
- **Sweep the camera** — `scenes.json` is data; generate one scene per lens /
  height and batch it.
- **Parameterise rig proportions** — the single largest source of fit residual
  (see §5).
- **Mesh-level round-trip** — dump mesh data (or a rebuild recipe) so human mesh
  edits survive the .blend → spec → rebuild path (see §7).


---

## Blender-driven control maps — depth / pose / hands


When pose, framing, focal length or foreshortening must be exact, build the scene
in Blender and render control maps instead of describing it in words. Depth and
pose maps are projected through the **same camera from the same armature**, so
they can be stacked without contradicting each other.

**Full workflow, scene-spec format, pitfalls and measured settings:
`references/blender_control_pipeline.md`.**

```bash
# 1. scene -> depth + clay preview + keypoints (--factory-startup is REQUIRED)
"<你的 Blender 安装目录>\blender.exe" --background --factory-startup \
    --python scripts/blender_build_scene.py -- --spec scenes.json --out out
# 2. keypoints -> OpenPose skeleton (COCO-18 body + 2x21 hand points)
python scripts/blender_draw_openpose.py out/s1_kps.json out/s1_pose.png
# 3. verify the skeleton sits on the geometry BEFORE spending GPU time
python scripts/blender_overlay_check.py out/s1_clay.png out/s1_pose.png out/s1_overlay.png
python scripts/blender_crop_hands.py s1        # hands are ~20 px; zoom before judging
# 4. generate: ControlNet (SDXL) / LLLite (Anima) / img2img from the clay render
node scripts/comfy_control.mjs references/blender_control_jobs_sdxl.json

# 5. optional: INGEST an existing 2D skeleton instead of authoring a pose by hand
python scripts/blender_extract_kps.py ref/skeleton.png ref/target_kps.json --overlay ref/check.png
"<你的 Blender 安装目录>\blender.exe" --background --factory-startup \
    --python scripts/blender_fit_pose.py -- --target ref/target_kps.json \
    --spec scenes.json --scene s1 --out scenes_fitted.json
python scripts/blender_pose_error.py ref/target_kps.json out_fitted/fitted_kps.json

# 6. multi-character tableau (+ per-character/env silhouette masks, + editable .blend)
"<你的 Blender 安装目录>\blender.exe" --background --factory-startup \
    --python scripts/blender_stage_multi.py -- --spec scenes_battle.json --out out --masks --save-blend scene.blend
# 7. human drags camera/characters/props in the Blender GUI, Ctrl+S, then read it back
"<你的 Blender 安装目录>\blender.exe" --background --python scripts/blender_extract_blend.py -- \
    --blend scene.blend --spec scenes_battle.json --out scenes_from_blend.json
# 8. local repaint: prior mask -> GrabCut-aligned mask -> latent noise mask
python libs/blender-controlnet/scripts/blender_make_inpaint_mask.py base.png out/s_mask_keqing.png mask_keqing.png --grow 16 --feather 6
node scripts/comfy_inpaint.mjs inpaint_jobs.json
```

Key facts:
- Poses are authored as **world-space bone directions** in `scenes.json`; hands as
  `{"L": {"curl": 0..1, "spread": 0..1}}`. Fingers are real bones (15/hand), so the
  mesh, depth map and OpenPose hand skeleton share one pose.
- **Multi-character** staging is `blender_stage_multi.py` (§6):
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
  meant to keep. (measured: "fix the tail's scales" at 0.85 came back with the tail
  detached from her hips and stray fins floating in the water — a band-1 task run at
  band 4.)

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
- **Repairing hands / faces (the "detailer" family) — measured boundaries.**
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
  (See also `references/cloud_edit_and_merge.md`: never hand the model the original colours if the outfit must change.)
- **True identity transfer needs FaceID, and FaceID needs two weight sets that are NOT on this
  host** (measured). Do not reach for plain IPAdapter and call it identity transfer:
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

