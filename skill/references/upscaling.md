# Upscaling — models, sources, measured comparison

**"The upscaling is bad" 通常是"对的模型没装"** —— 原装只有两个 upscale 模型，出图明显发软。
下面的数字都是实测。

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

## 浏览器 upscaler（`tool.hoothin.com/zh-CN/image-upscaler`）—— 不要移植

它跑在浏览器里（TF.js 图模型 + WebGL/WebGPU 后端，图片不上传），模型就是 Real-CUGAN / Real-ESRGAN
同族，但产物是浏览器专用 TF.js 格式（float16，64×64 tile → 256×256），转回 ComfyUI 可加载格式没有收益。
**结论：装原生模型**——"Anime+" 槽 = Real-ESRGAN anime；"RealCUGAN 降噪版 (4x)" = ncnn-vulkan 版的
`up4x-conservative`。它不可自动化，而可自动化正是 ncnn 的唯一价值。

## Delivery

- **放大在流程里会发生不止一次，而且每次对应"下一层要处理的细节"的尺度** —— 手指、脚趾、瞳孔这类小物件
  **没有足够像素就画不好**，所以是"小尺寸抽结构 → 中间尺寸补细节 → 交付尺寸收尾"，不是一次放大到底。
  见 `（未随仓库发布的内部文档）`（第 3 步与第 6 步）。
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


## Sharpness at delivery size

Judge sharpness at the size you ship, not at the 4x size: mean |Laplacian| of the luma
channel (3x3 kernel), image at 2000 px long edge — plain LANCZOS straight from the 4x
render **3.04** (P99 44); 4x-AnimeSharp then LANCZOS down **5.17** (P99 94); a 2000 px
original as the ceiling **5.53** (P99 89). Compare only at identical pixel dimensions;
a ~57 % gap is real, ~5 % is noise.
