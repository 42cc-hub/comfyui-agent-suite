# Upscaling — models, sources, measured comparison

Why this file exists: "the upscaling is bad" is almost always "the right model isn't installed" —
the stock install ships only two upscale models and produces visibly soft results. Everything
below is measured.

## The node chain

```
UpscaleModelLoader   { model_name: "<file>.pth" }          -> UPSCALE_MODEL
LoadImage            { image: "dsh_edit/<name>.png" }      -> IMAGE
ImageUpscaleWithModel{ upscale_model, image }              -> IMAGE (enlarged)
SaveImage            { filename_prefix, images }
```

Template: `upscaling_workflow.json`. Run it with the normal driver:

```bash
node comfy_generate.mjs references/upscaling_workflow.json out.png
```

Notes:

- `LoadImage` resolves relative to **`<你的 ComfyUI 安装目录>\input\`** — copy (and convert to 8-bit RGB
  PNG) the source there first.
- No sampler and no seed: the operation is **deterministic**, so re-running is pointless unless
  you change the model.
- Fast: 1216×832 → 4864×3328 takes seconds. The upscale is never the bottleneck; the diffusion
  pass is.

## Model inventory on this host (`models/upscale_models/`)

| file | size | arch | verdict |
|---|---|---|---|
| `4x-AnimeSharp.pth` | 63.9 MB | ESRGAN | **default** — cleanest illustration result |
| `RealESRGAN_x4plus_anime_6B.pth` | 17.1 MB | ESRGAN | good; slightly harder sharpening |
| `4xhfa2kludvaeswinir_v1.safetensors` | 16.0 MB | SwinIR | general purpose, not anime-tuned |
| `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` | 0.36 MB | — | **much softer than either 4x model** — this is why local upscaling looked bad |

The folder also contains the placeholder `put_esrgan_and_other_upscale_models_here`, i.e. an
otherwise-empty install.

## How the comparison was measured

Take the **same source**, the **same 200×200 crop** (a face), and display every candidate at the
**same magnification (4×)** — with a plain LANCZOS 4× as the baseline. Matching magnification is
the whole point: putting a 2× model's output next to a 4× model's output and eyeballing them
compares zoom levels, not models.

Result, best → worst: **`4x-AnimeSharp`** → `RealESRGAN_x4plus_anime_6B` → `2x_AnimeJaNai`.
AnimeSharp keeps line art crisp and separates hair strands without the white halos that
over-sharpening produces.

## Downloading more models

**This host's shell has no direct internet.** Plain HTTP times out; HTTPS through `curl` and
`Invoke-WebRequest` fails with
`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS` — the Windows TLS stack cannot
get credentials inside the sandbox. There *is* a local forward proxy on **`127.0.0.1:6132`**, and
Python's OpenSSL backend works through it.

```bash
# scripts/fetch_url.py
python fetch_url.py --head <url>                                  # size probe only
python fetch_url.py <url> <你的 ComfyUI 安装目录>\models\upscale_models\<file>
```

Verified reachable (HEAD → 200):

| model | size | url |
|---|---|---|
| 4x-AnimeSharp | 63.91 MB | `https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth` |
| 4x_foolhardy_Remacri | 63.92 MB | `https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_foolhardy_Remacri.pth` |
| 4x-UltraSharp | 63.86 MB | `https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth` |
| RealESRGAN_x4plus | 63.94 MB | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth` |
| RealESRGAN_x4plus_anime_6B | 17.11 MB | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth` |
| realcugan-ncnn-vulkan (Windows) | 43.85 MB | `https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-windows.zip` |

A downloaded ESRGAN `.pth` is a zip container — check the first four bytes are `PK\x03\x04`.
If the file starts with anything else, the download was an error page.

## Real-CUGAN vs Real-ESRGAN

They are different architectures, and `UpscaleModelLoader` only accepts the ESRGAN family:

- **Real-ESRGAN / AnimeSharp / Remacri / UltraSharp** → drop the `.pth` in `upscale_models/` and
  it works immediately, no node changes.
- **Real-CUGAN** → needs its own runtime (`realcugan-ncnn-vulkan`, above) or a dedicated ComfyUI
  node. Exporting the weights as `.pth` does **not** make them loadable.

## The browser tool at `tool.hoothin.com/zh-CN/image-upscaler`

Investigated because its "Anime+ (4x)" looked better than the local result.

- It runs **entirely in the browser**: a TensorFlow.js graph model with a WebGL/WebGPU backend.
  The image is never uploaded; only the weights are fetched once from `/models/...`.
- The models are the **Real-CUGAN / Real-ESRGAN family** — the same families installable
  natively, where they run faster and can be scripted.
- Its own artifact is a **TF.js graph model** (`convertedBy: TensorFlow.js Converter v4.21.0`,
  float16 weights, 64×64 input tiles → 256×256 output). That is a browser-only format;
  converting it back to something ComfyUI can load is work with no payoff.
- **Conclusion: don't port it — install the native model.** The "Anime+" slot maps to the
  Real-ESRGAN anime model; the "RealCUGAN 降噪版 (4x)" slot maps to `up4x-conservative` in the
  ncnn-vulkan build.

Caveat: the site's models are downloaded to the browser cache, so after the first
load it works offline-ish, but a page load still needs the network for HTML/JS. It is a fine tool;
it is simply not automatable, which is the only thing ncnn buys you.

## Delivery

- **放大在流程里会发生不止一次，而且每次对应"下一层要处理的细节"的尺度** —— 手指、脚趾、瞳孔这类小物件
  **没有足够像素就画不好**，所以是"小尺寸抽结构 → 中间尺寸补细节 → 交付尺寸收尾"，不是一次放大到底。
  见 `references/production_pipeline.md`（第 3 步与第 6 步）。
- 4× from a 1216-wide source = 4864×3328 (16 MP). Ship **2×**: keep the 4× result and downscale
  by half with LANCZOS. That beats feeding the source to a native 2× model.
- 2432×1664 lands around 3–4 MB as PNG — comfortable for a gallery upload.
- **Keep the un-upscaled original.** Re-upscaling with a better model later is cheap;
  re-generating the image is not.

## 放大之后再"整图重绘"会让画质变差 —— 要用线控把它钉住（实测）

**现象**：把刚放大完的图再做一次**整图 0.5 去噪重绘**，锐度会掉，而且内容会被重解。

| 步骤（同一张图） | 清晰度 mean / P99（\|Laplacian\|） |
|---|---|
| 底图 768×1104 | 8.22 / 135 |
| **放大后（4x-AnimeSharp → Lanczos，1256×1800）** | **7.58 / 148** |
| 整图 0.5 重绘 ×3（无控制） | **6.63 ~ 6.80 / 123 ~ 130** |

**机理**：放大造出的高频"偏硬但确实存在"；0.5 去噪会重新加噪采样，而模型**没有原图高频细节的记忆**，
只能在那个尺度上重新生成"看起来合理"的细节 ⇒ 细节被换掉（蓝玫瑰→蝴蝶结）、边缘被平滑。
**重绘不是锐化工具**：它用来**改内容**，不是用来"再清晰一点"。要做锐化，用放大链而不是采样。

**对策（实测有效，+10% 锐度）**：给重绘加**线控**把结构钉住。本机 anima 基座**没有 canny 控制模型**，
等价物是 LLLite 的线稿控制：

```text
LoadImage(放大后的图) → Canny(low 0.2, high 0.5) → AnimaLLLiteApply_sdscripts(
    model = UNETLoader → LoraLoader, lllite_name = "anima-lllite-scribble-1.safetensors",
    strength = 1.0, start_percent = 0.0, end_percent = 1.0)  → KSampler(denoise 0.5)
```

| 变体（固定种子，1256×1800，d0.5） | 清晰度 mean / P99 |
|---|---|
| 无控制 | 6.71 / 123 |
| LLLite s0.8 ＋ canny 0.2/0.5 | 7.26 / 137 |
| **LLLite s1.0 ＋ canny 0.2/0.5** | **7.37 / 139** |
| LLLite s0.8 ＋ canny 0.4/0.8 | 6.89 / 130 |
| LLLite s1.0 ＋ canny 0.4/0.8 ＋ d0.4 | 6.84 / 130 |

两条反直觉但实测的结论：**① 线越密越好**（`0.2/0.5` 明显优于 `0.4/0.8`）—— 钉结构靠的是线多，
不是去噪小；**② 降低 denoise 救不回来**（0.4/0.8 那组降到 0.4 仍 6.84），因为瓶颈是"线不够"。

**⚠ 一个工作流只留一个 `SaveImage`**：为了顺手看 canny 线稿而加第二个 `SaveImage`，会让
`comfy_run` 把**二值线图**当成产出抓走（96 KB、清晰度 34.75 —— 看着像"突然变超清"，其实是线）。
要看中间产物就单独跑一次。


---

## Upscaling (2x / 4x) — after generation, before delivery



Chain: **`UpscaleModelLoader` → `ImageUpscaleWithModel` → `SaveImage`**. No diffusion pass,
so it is fast and deterministic — 1216×832 → 4864×3328 takes seconds, not minutes.
Reference graph: `references/upscaling_workflow.json`. Model sources, the verified download
URLs and the full comparison: **`references/upscaling.md`**.

**Measured** — same source, same 200×200 face crop, all four shown at 4×:

| model | verdict |
|---|---|
| `4x-AnimeSharp.pth` | **best** — cleanest line art, hair strands separate, no sharpening halos. **Use as default.** |
| `RealESRGAN_x4plus_anime_6B.pth` | good, marginally harder / more aggressive sharpening |
| `4xhfa2kludvaeswinir_v1.safetensors` | general purpose, not anime-tuned |
| `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth` | **REMOVED** (moved to `agent_work/_disabled_upscalers/`) — it was **noticeably softer than either 4x model**, and using it was the actual cause of "upscale looks blurry" reports. Use `4x-AnimeSharp` then Lanczos-halve instead. |

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
  The same reference also has the **no-custom-node route** (PIL crop → sample → paste by mask,
  with the `LoadImage MASK = 1 − alpha` trap that turns a repaint into a silent no-op), the
  `min-sample` rule for tiny targets, and when to **split one diagonal region into two passes**.
  **The stitcher itself is bit-exact; the crop's *rescaling* is what moves pixels.** With the crop **not
  rescaled**, a pure passthrough (`crop_image → SmartImageStitcher`, **no sampler at all**) reproduces the
  original **bit-for-bit** — out-of-mask `>2/255` = **0.0000 %**, `max Δ 0` — with `Box Feather 32` and with or
  without the stitcher's `mask` input; the **full sampled** graph is bit-exact too once the crop is not rescaled
  and the mask is wired (out-of-mask **0.0000 %**, `max Δ 0`, in-mask 62–73 % changed). An out-of-mask **4.31 %**
  / `max Δ 83` reading is a *rescaled* crop, not a stitcher limit. The drift comes from **`SmartImageCrop`
  resizing the crop** — `force_divisibility=128` / `min_resolution=768` force a resize even with no sampling
  (measured 0.0542 % out-of-mask, `max Δ 46` just from crop→paste), and if the stitcher's `mask` is *not* wired
  the whole crop box carries that difference (0.3352 %, `max Δ 229`). So: **set `force_divisibility=8` (and a
  `min_resolution` the mask bbox already clears) so no resize happens, and wire `crop_mask` into the stitcher's
  `mask`** — then either route is exactly 0 outside. Report `max Δ outside mask` plus the `>2/255` percentage.
  Measurement script + raw numbers: `task/lumine_hospital_20261004/verify_stitcher.py` (cases P1–P7).

### Shrinking to the delivery size, and proving sharpness

- **Downscaling is part of the recipe, not an afterthought**: `ImageUpscaleWithModel` → `ImageScale`
  (`lanczos`) to the target long side. Delivering the 4× output directly is 16 MP of no extra
  information.
- **A plain resize is not a substitute for the upscale model.** Same source, same target size
  (3000 px long side): plain LANCZOS gives `|Laplacian|` mean **3.04** / P99 44; 4×-AnimeSharp then
  Lanczos gives **5.17 / 94**, against the 2000 px original's 5.53 / 89 — i.e. the model preserves
  the source's edge energy while a bare resize loses ~40 % of it.
- **Score sharpness at the same pixel size, on the luminance channel**, using the 3×3 Laplacian's
  mean absolute value and P99 (script in the pixiv project: `outfit_prompt_lab/sharpness.py`). A
  57 % gap is real; 5 % is noise. This is also how to settle "which upscaler" and "did my
  re-sharpen pass help" without arguing from impressions.

