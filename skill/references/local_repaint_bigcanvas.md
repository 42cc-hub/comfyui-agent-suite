# Editing a small region of a large image (SmartImageCrop / SmartImageStitcher)

For when a detail is only a few pixels wide on the finished panel — a shoelace knot, an
eyelet, a ring — and a whole-image pass cannot resolve it: at low denoise the detail stays
blurry, and at high denoise the entire picture drifts. Verified on a 1400×3200 comic page
and its 832×1216 panels, ComfyUI 0.24, RTX 3060 6 GB.

## The shape of the solution

`ComfyUI-Smart-Image-Crop-and-Stitch` (already installed) provides:

| Node | What it does |
|---|---|
| `SmartImageCrop` | crops to the MASK's bbox, then scales so the long side lands in `[min_resolution, max_resolution]`. Outputs `crop_image`, `crop_mask`, `stitcher_info`, `preview_overlay` |
| `SmartImageStitcher` | pastes the processed crop back using `stitcher_info`, with `feather_pixels`, `blend_mode`, `enable_color_match` |

**These exist to edit huge images without blowing VRAM** — cut the region out, process it at
its own scale, paste it back. Using them the other way round (small image, magnified crop)
throws away the point.

### Cost depends on the crop, not the canvas

| Canvas | Wall time |
|---|---|
| 832×1216 | 80 s |
| 3328×4864 (4x) | **44 s** |

The second is faster: only the 932×779 crop is denoised. **Editing a big image costs the
same as editing a small one.**

## Two masks, two jobs

Do not use one mask for both:

* **crop mask** — generous. It sets the crop box, hence the magnification and how much
  surrounding context the model gets to see.
* **noise mask** — tight, around the thing actually being repainted.

Then `VAEEncode(crop_image) → SetLatentNoiseMask(mask=noise_mask) → KSampler(denoise)`.

## The noise mask's *width* is a knob for contact, not just a boundary

`SetLatentNoiseMask` freezes every pixel outside the mask, so the model can only add texture
*inside* it. A tight noise mask can give a rope surface detail but can never make it cast a
shadow on the hair or push hair aside — those pixels are locked, so the rope reads as pasted on.

Measured on a hand-drawn rope crossing a face (Illustrious, cfg 8, 20 steps, diff-derived mask):

| noise-mask dilate | denoise | outcome |
|---|---|---|
| 28 (13% of frame) | 0.4 | rope lies flat, **no shadow on the hair** |
| 45 (18%) | 0.4 | still a smooth flat band, no braid |
| 45 (18%) | **0.5** | **braid texture appears**, the rope reads round |
| 70 (25%) | 0.4 | shadow and occlusion present, **but the eyes got repainted too** |

The two knobs are independent: **mask width buys contact (shadow, occlusion); denoise buys
texture (twist, volume)**. The combination that works is a mask **narrow enough to leave eyes
and teeth outside, wide enough to ring the target by ~45 px, at denoise 0.5**.

Failure mode to watch: widening the mask to include the face lets the model "correct" features
you meant to keep — the run that fixed the rope also moved the eye and the cheek.

### A noise mask cannot remove what the LoRA always draws

If an element belongs to the LoRA's default look, painting over it and re-denoising brings it
back. **Diagnostic:** measure the pixels the repaint *added* relative to the hand edit. When they
return **shattered into hundreds of small blobs** (measured: 3438 px across 1442 blobs, largest
191 px), the prompt is not asking for a rope — the prior is pulling every fragment back. Lowering
denoise only shrinks it (0.3 → 3438 px, 0.2 → 3151 px). Either exclude that region from the mask
or fix it by hand.

Corollary for tiny hand edits: a mask under ~1% of the frame at denoise 0.2 (12 steps ⇒ 2.4
effective steps) changes almost nothing — the edit survives untouched, which is a fine outcome,
but do not expect the pass to add anything either. Passes earn their cost when what must appear
is *texture*, not geometry.

## Feather hides a problem you should not have

Drawing with the whole rectangular crop re-noised means the crop's border can never match
its surroundings, so a seam appears and `feather_pixels` is used to disguise it. That is the
wrong fix. `SetLatentNoiseMask` limits the renoising to the mask, so everything outside comes
back untouched and **no seam can form**.

Measured on the same edit (changed pixels vs the source, threshold 8/255):

| Approach | changed px | % of frame | mean diff |
|---|---|---|---|
| feather 128, whole crop re-noised | 351,236 | 2.17% | 0.967 |
| rectangular `SetLatentNoiseMask` | 155,113 | 0.96% | 0.503 |
| **shaped** `SetLatentNoiseMask` | **74,979** | **0.46%** | **0.214** |

A smaller rectangle is still a rectangle — it keeps re-noising the subject and the
background inside it. **The mask has to follow the shape.**

### Get the shaped mask for free

If a human edited the image on top of a known original, the mask is already implied:

```python
diff = ImageChops.difference(before, after).convert("L")
mask = diff.point(lambda v: 255 if v > 24 else 0)      # threshold the anti-aliasing noise
mask = mask.filter(ImageFilter.MaxFilter(5))           # dilate so the soft edge is inside
```

No extra masking work, and it is exact by construction.

### Residual difference outside the mask

**The stitcher is a pure cut-and-paste; it does not change anything by itself (measured 2026-10-04).**
With the crop **not rescaled**, a passthrough (`crop_image → SmartImageStitcher`, **no sampler**) reproduces
the original **bit-for-bit**: out-of-mask `>2/255` = **0.0000 %**, `max Δ 0` — and that holds with
`Box Feather 32`, and with or without the stitcher's `mask` input. The **full sampled** graph is bit-exact
too once the crop is not rescaled and the mask is wired (**0.0000 %** out-of-mask, `max Δ 0`, in-mask 62–73 %
changed at denoise 0.45).

What *does* move out-of-mask pixels is **`SmartImageCrop` resizing the crop**, plus leaving the stitcher's
`mask` unwired:

| configuration | out-of-mask `>2/255` | `max Δ` |
|---|---|---|
| passthrough, crop not rescaled (`force_divisibility=8`) | **0.0000 %** | **0** |
| passthrough, defaults (`min_resolution=768`, `force_divisibility=128`, `mask_grow=32`) | 0.0542 % | 46 |
| sampled 0.45, crops rescaled, `mask` wired | 0.3352 % | 229 |
| sampled 0.45, crop **not** rescaled, `mask` wired | **0.0000 %** | **0** |

So to eliminate it: **set `force_divisibility=8` and a `min_resolution` the mask bbox already clears** (no
resize), and pass the **same shaped mask to `SmartImageStitcher`'s `mask` input**, so the paste-back region
equals the repaint region exactly. `feather_pixels` is not a cause (it blends two identical images in the
passthrough → no change). Script + raw numbers: `task/lumine_hospital_20261004/verify_stitcher.py` (P1–P7).

> ⚠️ An earlier version of this section said this route "is not bit-exact outside the mask (0.23 % Box Feather /
> 0.53 % Mask Feather)", and `SKILL.md` went further, blaming `SmartImageStitcher` itself for 4.31 %.
> **Both attributions were wrong** — the numbers were real but they came from the crop resize and the unwired
> mask, not from the stitcher, and the "0.23 %"/"4.31 %" figures are not properties of this route.

## 预算怎么分：**抽卡修结构，重绘修微观**（用户定，2026-09-30）

> **多轮抽卡是留给"出底图"的；重绘不需要多轮。** 理由不是重绘不好用，而是**两者能修的东西不同**：

| 手段 | 能改 | 改不了 |
|---|---|---|
| **换种子重抽（抽卡）** | 结构、人数、两人关系、机位、谁在做什么 —— 即**宏观层** | 小细节（每次重抽都会换一批新的小错） |
| **局部重绘** | 纹理、绳路密度、道具、接缝 —— 即**微观层** | 任何宏观错误（姿势、朝向、遮挡、构图）—— 它们被 mask 外的像素**锁死** |

必然推论：

- **"这个结构不对"＝继续重绘是浪费** —— 去抽卡（或改控制图），别在错的画面上打磨。
- **结构一旦成立，重绘的边际收益就很小** —— 布局固定、角色身份由 LoRA/提示词钉住、外观已符合时，
  一遍就够；第二遍不会产生可辨的差异，而且**每一遍都在对 mask 外做一次 VAE 往返**（见上节），
  还会把已经对的部分重新置于风险里。**实测（2026-09-30）**：同一张双人底图、4 个种子各做一遍
  区域重绘，四张几乎没有可辨差异；mask 外的差异 mean 1.6 / p99 12（肉眼不可见），
  mask 内 mean 55–62（真改动）——**收益集中在"选定的一块"上，不在"多跑几轮"上**。
- 所以：**底图的抽卡预算按"宏观失误模式"给**（双人易互换/人数错/合并、多人易污染 ⇒ 多抽几张），
  **重绘按"要修的那一块"给一次**，然后**在成品上验收**，而不是反复重绘。

## 多人图里的分区重绘：只重绘"当前这个人"（用户定，2026-09-30）

对一张已经成立的多人图做重绘时（全画布 + `SetLatentNoiseMask`，不动别处）：

1. **mask 只框当前这个人**（人画的区域或按人拆的层），**denoise 0.6** 起步；
2. **LoRA 只挂当前这个人的那一个**，不要同时挂两个人的 —— 两个角色 LoRA 会在 latent 里互相拉扯；
3. **提示词仍写整个场景，但其他人一律简化**成占位词：
   `another girl sitting beside her`（必要时加两三个不会污染的颜色词，如 `blue hair`），
   **不要**把另一个人的完整外观块写进去 —— 否则模型会按你的文字重画她、或把她的特征带到当前这个人身上。
   实测（2026-09-30，Anima 双人）：这样写之后当前人物拿到自己的身份，另一个人保持原样、人数不变。
4. **不能跨基底挂 LoRA**：这一轮想给第二个角色也挂 LoRA 时发现她的 LoRA 是 SDXL
   （`modelspec.architecture = stable-diffusion-xl-v1-base/lora`、tensor 键 `lora_te1_*`），
   而底图是 Anima（qwen-image）—— **挂不上**。要么整条流水线换成该 LoRA 的基底，
   要么这个人**只用文字描述**（并在记录里写明这是妥协）。

> 与上一节 §"The prompt must describe only what is inside the crop" 的关系：那条针对**裁切型**
> 重绘（画布上只剩裁片，写别的等于让模型画它看不见的东西）；本节针对**全画布 mask 型**重绘
> （整幅都在画布上，所以"写整个场景"才是对的）。**判据 = 画布上还剩多少上下文**。


`feather_pixels` was fine at 32 on an 832-wide canvas; on a 3328-wide one that is a quarter
of its relative reach and left a **visible rectangular seam**. 128 removed it. The same
applies to `mask_grow_pixels`: **any parameter in pixels must be scaled with the canvas**,
or its behaviour changes silently.

## Denoise sets how many pixels wide an edit can reach — and that scales with the image

> 宏观层面这条决定"这一步该给多少 denoise"：**0.2–0.4 材质 / 0.5–0.6 比例与换引擎 /
> 0.8–0.9 结构拓扑 / 1.0 重抽**。整条主线见 `references/production_pipeline.md`。
> 实测补充（2026-10-03，Anima 2000² 整图）：**绳"连到床框"这种拓扑，0.4/0.6/0.75 全出不来，0.9 才出来**；
> 而 0.4~0.75 这一档只改材质与颜色——**换材质/颜色靠提示词，不靠加 denoise**。

`denoise` is a fraction of the sampler schedule, so the model's reach is a fraction of the
**latent grid**, not a fixed number of pixels:

> **The lower the denoise, the narrower the band it can actually change — and the same numeric
> denoise reaches proportionally further on a larger image.**

Measured on this host (Illustrious, `cfg 8`, 20 steps, diff-derived mask ≈ 1 % of a 1232×1800 frame,
a hand-drawn rope across a face):

| denoise | what changed |
|---|---|
| 0.2 | almost nothing — one highlight softened; the flat rope stayed flat |
| 0.4 | rope kept its shape and gained slight shading — still a flat band |
| 0.5 | **braid texture appeared**; the rope read as round |

Consequences:

- To repaint something only a few pixels wide on the canvas, **low denoise cannot touch it**, however
  tight the mask is. Either **crop to it** so it becomes a large fraction of the canvas (see the
  crop-to-mask rule in `SKILL.md`), or raise denoise — and raising denoise costs structure.
- **Re-tune denoise when the canvas changes.** 0.2 on an 832×1216 canvas reaches roughly half the
  pixel width of 0.2 on a 1664×2432 one. A denoise that "did nothing" is usually mis-scaled, not wrong.
- **"Raise denoise until it moves" is the wrong reflex.** The cheaper fix is almost always
  *crop closer, keep denoise low*; that is also what keeps the untouched area bit-for-bit intact.

## The prompt must describe only what is inside the crop

After cropping, the model sees nothing else. Naming the park, the legs, or the character
when the crop holds only two shoes is the same error as naming an occluded limb: it asks the
model to draw something it cannot see.

Narrowing the mask therefore means narrowing the prompt with it.

## The 4x → shrink → 4x chain: measured, and rejected

The idea is attractive — the upscale model's factor is fixed at 4x while the shrink is
script-controlled, so going up, shrinking, and going up again should give both a sharpening
pass and a controlled final size. It does not work out:

* delivery = **4 × intermediate**, so a 2x-source delivery forces the intermediate to
  **0.5× the source** — information destroyed before the final pass, and AnimeSharp's
  sharpening compounds into halos and banding (measured: crunchier hair, stepped rope,
  banding in the sky)
* keeping the intermediate ≥ the source forces a **4x-source delivery** (dimension explosion)

The existing rule already *is* the intended thing: **a single 4x-AnimeSharp pass is the
sharpening pass, and the LANCZOS halving is the precise size control.** The extra 4x only
re-applies sharpening.

> A first attempt at this chain multiplied the wrong way and produced 6656×9728 / 47 MB from
> an 832px panel. When chaining fixed-factor models, write the dimension math down and check
> the output size before looking at the picture.

## 分区定 denoise：IL 只治人物，别碰场景结构

**结论（2026-09-24 实测，代价见下）**：拿 IL 去"整体提升质感"是错的 —— 它会**重画结构**，
而 AI 生成图的结构（棋盘格线、棋子、建筑、文字）经不起重画。正确做法是**按区域分两次 inpaint**：

| 区域 | denoise | 目的 |
|---|---|---|
| **环境 / 道具 / 背景** | **≈ 0.2** | 只补一点肌理与光色；结构几乎不动 |
| **人物** | **≈ 0.4–0.6** | 材质、发丝、皮肤层次可以明显改 |

即：**环境一刀、人物一刀**（各自蒙版 + `SetLatentNoiseMask`），**不要整图一刀**。

依据（同一张 2000×1016，IL，无 LoRA，same prompt）：

- 整图 **0.45** ⇒ 人物质感确实变好，但**棋盘格线扭曲、白马化成蜡烛、马头消失** ✗（放大棋盘区逐格对比可见）。
  这不是可以调掉的副作用：**denoise 是采样进度的一部分，"哪里该动"它无从知道**。
- 加**结构控制**（`noob-sdxl-controlnet-softedge_hed`，自造 DoG 边缘图即可，占边 ~15%）：
  结构基本被钉住 ✓（棋盘格线笔直、棋子形状在 ✓），**但低信息区域会凭空多出物件** ✗
  （实测桌上多出一个金壶＋橙杯），因为控制只约束"边缘在哪"，不约束"那里有什么"。
- 再叠一层 **inpaint 控制**（`NoobAI_Inpainting_ControlNet` + `SolidMask(0)` → `InpaintPreprocessor`，
  strength 0.6）能把内容拉回原图 ⇒ **多余物件消失** ✓，质感仍比原图好 ✓；但 **0.9 会把画面压回接近原图**
  （没有余量改质感 ✗），且两个 4.66 GB 控制叠加在 6 GB 卡上要 **461 s/张**（单控制 ~110 s ✗）。

⇒ 所以最省的组合不是"堆控制"，而是**分区**：环境压到 0.2（几乎不动结构、也不需要控制网），
人物给 0.4–0.6（放开改）。**控制网是兜底手段，不是首选。**

## The no-custom-node route (PIL: crop → sample → paste by mask)

When SmartImageCrop/Stitcher is unavailable — or when you want the arithmetic explicit and the
outside guaranteed bit-exact — the same job is ~60 lines of PIL. Measured: a 1024×1024 crop
upscaled to 1280 and sampled takes **48 s** on a 3000×1830 canvas; outside-mask change
**0.0000 %**. The recipe:

1. **crop box = mask bbox + `pad`** (120 px is plenty of context), then **grown** so the short
   side clears `min-side` (832–1024). A crop barely larger than the mask gives the sampler no
   context and it invents surroundings.
2. **Small target ⇒ upscale the crop before sampling** (`min-sample` 768–1280). A 120×170 px
   face is only ~20 k pixels; sampling it at 1280 gives the model enough pixels to draw an eye,
   and the answer is downscaled back. This is the "crop closer, keep denoise low" rule from
   §"Scale-dependent parameters", made numeric.
3. **A large diagonal region must be split into two passes.** Measured: a rope running from a
   forearm to the far corner of a 3000 px image has a 1805×1690 bbox; with `max-side 1536` the
   crop is downscaled 0.85× and the strand texture blurs away. Split it at a vertical line into
   1392×848 + 1024×1064 (both native) and run them in sequence on the same canvas — the masks are
   disjoint, so the second pass cannot touch the first's pixels.
4. **Rescale picture and mask with the same factor**; `LANCZOS` for the picture, `NEAREST` (or
   `LANCZOS`) for the mask. A mask resized with the wrong filter grows a fuzzy border that
   repaints the neighbour.
5. **Alpha carries the inverted mask** — see the trap below.
6. Paste only inside the crop box: `out = base * (1 - m) + repainted * m`. Everything else is the
   original file, bit for bit.

### The `LoadImage` trap: `MASK = 1 − alpha`

`LoadImage`'s MASK output is **`1 − alpha`** (`nodes.py:1744-1745`). Write your mask into alpha
as-is and `SetLatentNoiseMask` receives its complement: the pass repaints exactly what you meant
to freeze and the edit becomes a **silent no-op** (measured: 0.20 % of the "masked" area changed,
0.00 % outside — nothing happened). Write the **inverted** mask (`rgba.putalpha(255 - mask)`) and
verify with `diff_report.py` before reporting any result. In an earlier project two runs were
recorded as successful repaints when the visible change had actually come from the user's own
Photoshop work.

### An ignored flag looks exactly like a reversed mask

Both surface as "the picture did not change". Diagnose by **reading the graph that was actually
submitted** (`wf/*.json`) rather than re-tuning numbers: a `--lora-strength` that never reached
`LoraLoader.strength_model` (hard-coded at 0.8) made "the LoRA is too weak at denoise 0.3" look
like a denoise problem — feeding the documented strength produced the missing texture at the
*same* denoise.

### Cloud-assisted inpaint through the same crop

The crop does not have to be sampled locally: send crop + mask to a hosted editor, paste the
answer back by the same mask. The crop/mask/scale arithmetic is unchanged; the extra work is
proving the answer respected the mask and protecting structures that must line up with the world
outside the crop — both in `cloud_edit_and_merge.md`.

## Runnable template, and the wire-up traps (smoke-tested 2026-09-26)

`references/smartcrop_repaint_workflow.json` is a complete graph — IL checkpoint + LoRA, two
`CLIPTextEncode`, and:

```
LoadImage(base) ─┬──────────────────────────────────────────────┐
LoadImage(mask) → ImageToMask(red) → SmartImageCrop ─┬─ crop_image → VAEEncode → SetLatentNoiseMask → KSampler → VAEDecode ┐
                                                     ├─ crop_mask ────────────────────────────────────────────────────────┤
                                                     └─ stitcher_info ────────────────────────────────────────────────────┤
                                                                        SmartImageStitcher(original, processed, info, mask=crop_mask) → SaveImage
```

Placeholders to edit: `lora_name` (or delete the LoraLoader), the positive prompt (it must
describe **only** the masked region), `denoise`, `seed`, and the two input PNG paths in
`<你的 ComfyUI 安装目录>\input\dsh_edit\`.

Three things the smoke test (1200×800 base, a 260×260 white square, denoise 0.5, 28 s) settled:

1. **The mask PNG must be readable as a mask.** `LoadImage`'s own MASK output is `1 − alpha`, so a
   plain white-on-black PNG (or the grayscale L masks the project scripts write) arrives as *all
   zeroes*. `ImageToMask(channel="red")` reads the picture's value instead and works for both
   grayscale and white-on-black masks. (Equivalently: keep the alpha route and add `InvertMask`.)
2. **The stitcher's optional `mask` input must be the *crop* mask, not the full-image mask.**
   Passing the full-image mask made the paste-back fight the repaint: the measured in-region change
   collapsed from **11.1 % to 1.5 %** — a "the repaint did nothing" symptom caused entirely by the
   wire-up. (`SmartImageStitcher` gets the placement from `stitcher_info`, so it wants the mask in
   the crop's own coordinates.)
3. **This route is bit-exact outside the mask — if the crop is not rescaled and the stitcher's `mask` is
   wired** (`max Δ 0`, measured 2026-10-04). Earlier this bullet recorded 0.23 %/0.53 % drift and blamed the
   route; that was a **misattribution** — the drift is `SmartImageCrop`'s resize (`force_divisibility=128` /
   `min_resolution=768`) plus the un-masked paste-back, and it disappears at `force_divisibility=8` with
   `crop_mask` fed to the stitcher's `mask`. See §"Residual difference outside the mask". The PIL route is
   still worth preferring when you want the arithmetic explicit, but "0.0000 % vs 0.23 %" is no longer the
   reason.

## 自动定位部位——不必每次都请人画蒙版（实测 2026-10-02）

这台 ComfyUI **已经装好**姿态与分割节点，可以在**我们自己生成的二次元图**上自动定位部位：

| 节点 | 输出 | 实测 |
|---|---|---|
| `OpenposePreprocessor` | 骨架图 ＋ **`openpose_json`** | 在一张自出二次元图上正确估出 15 个身体关节（肩/肘/腕/髋/膝/踝…）。**坐标是归一化 0~1**，要乘原图尺寸 |
| `SDPoseKeypointExtractor` / `SDPoseDrawKeypoints` | `POSE_KEYPOINT` / 渲染图 | 需 MODEL+VAE，备用 |
| `SAM3_Detect` | `MASK` ＋ `BOUNDING_BOX` | 按提示分割；与上面的部位框配合可做精细蒙版 |
| `SAMPreprocessor` / `AnimalPosePreprocessor` / `DensePosePreprocessor` | → IMAGE | 备用 |

**拿关节点的正确姿势**（踩过的坑都在这条路上）：

1. `POSE_KEYPOINT` 这类自定义输出**不在保存的图片里**，只能从 `/history/<prompt_id>` 取；
2. 输出键名不是 `pose_keypoint`，而是 **`openpose_json`**，值是 **`["<json 字符串>"]`（单元素列表）**，
   JSON 内容是标准 OpenPose 布局 `{"people":[{"pose_keypoints_2d":[x,y,score, ...]}]}`（每人 18 组）；
3. 坐标**归一化**，直接 `%d` 打印会全是 0（这个坑很像"检测失败"）；
4. 有了关节坐标 ⇒ 按关节组取外接框（feet/legs/face/arms/torso）当**部位蒙版的初框**。

已封装成 （上传→跑节点→取 history→换算像素→输出部位框，
`--mask legs=out.png` 还能直接写出该部位的蒙版 PNG），
实测：`legs bbox=(309,805,513,1102)`、`face bbox=(300,307,563,361)`、`torso bbox=(243,489,620,806)`。
**已知空白**：手部/脸部精细关键点在这类图上常缺失（这张图没出手点），必要时用 SAM 按框补分割。
**准确度边界（实测）**：站姿/跪姿够用；**四肢着地、吊缚这类极端姿势**关节大量缺失，
框会只覆盖到近侧肢体（`human_dog` 那张只框到大腿）⇒ 极端姿势仍需人粗涂一笔或用 SAM 补。

## Files

| Script | Role |
|---|---|
| `references/smartcrop_repaint_workflow.json` | runnable SmartImageCrop → SetLatentNoiseMask → SmartImageStitcher graph (smoke-tested; edit the LoRA, prompt, denoise and the two input paths) |
| `scripts/chain_upscale.py` | 4x → script-controlled shrink → 4x, with the dimension arithmetic in comments (kept as the measured negative result) |
| `libs/blender-controlnet/scripts/blender_make_inpaint_mask.py` | builds a prior mask from a Blender silhouette when there is no hand edit to diff against |

See also `references/upscaling.md` for model choice and `references/prompt_writing.md` §2 for
why naming an invisible thing distorts the result.
