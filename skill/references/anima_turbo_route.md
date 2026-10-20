# Anima 权重三档（base / aesthetic / turbo / 社区合并）

`Anima` 是 CircleStone Labs × Comfy Org 的 **2B** 模型，架构上是
**NVIDIA Cosmos-Predict2-2B-Text2Image 的衍生**（不是 qwen-image；它只复用
**Qwen-Image VAE** + **Qwen3-0.6B** 文本编码器）。定位：动漫/插画，**不做写实**。

## 三档权重（哪一档干什么）

| 权重 | 文件 | 用途 | 采样 |
|---|---|---|---|
| base v1.0 | `anima-base-v1.0.safetensors` | **训 LoRA 用这个**；最灵活、多样性最高、默认风格最平 | 30–50 步，CFG 4–5 |
| aesthetic v1.1 | `anima-aesthetic-v1.1.safetensors` | 审美微调版，默认出图质量最好（作者："大多数人会更喜欢"）；**评分 tag 别再加** | 30 步，CFG 3–5 |
| turbo v1.1 | `anima-turbo-v1.1.safetensors` | 蒸馏快版，**最省时间的一档**；也能直接当出图档，交付前自己看那张 | **8–12 步，CFG 1** |
| 社区合并 | `waiANIMA_v10Base10.safetensors` | base v1.0 的社区合并版（风格更讨喜，不是质量保证） | 30 步，CFG 5 |

- 三者都是 `UNETLoader` + `qwen_3_06b_base.safetensors` + `qwen_image_vae.safetensors`，
  **换权重只改 `unet_name` 和 steps/CFG**，节点图不动；文本编码器三档通用。
- turbo 的 sha256 `fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb`（3.90 GB）；
  Civitai 侧文件名是 `anima_turboV11.safetensors`，HF 在 `split_files/diffusion_models/`。

## 本机实测（800×1200，seed 424242，er_sde/simple，同一 prompt，无 LoRA）[measured]

| 权重 | 步数 / CFG | 单张耗时 |
|---|---|---|
| anima-base-v1.0 | 30 / 5.0 | 49.4 s |
| waiANIMA_v10Base10 | 30 / 5.0 | 47.5 s |
| anima-turbo-v1.1 | 10 / 1.0 | **10.5 s** |

- turbo 快 **4.7×**，差距来自两处：步数 ÷3 + CFG 1 只有单分支。
- **不要按型号预设质量结论**：三档同 prompt 同 seed 出图的差别是**风格/质感差异，不是固定的好坏排序**。
  turbo 出图**不必然**更脏/更平，base/wai 也不必然更好 —— 交付前逐张看那张具体图。 [user]
- 选档的唯一稳定判据是**约束**，不是"画质口碑"：要多试构图 ⇒ turbo；要训 LoRA ⇒ base；
  要少调参直接出 ⇒ aesthetic；要特定风格 ⇒ 社区合并。

## turbo 的三条硬规矩

1. **CFG 1.0 ⇒ 负面提示词对结果零影响**（机制事实：CFG=1 时 uncond 分支不计算）。同 seed、空负面 vs
   标准负面解码后逐像素完全相同（**MAD 0.0000，2/2 seed**）。别再写 `worst quality, low quality, score_1...`
   指望它生效，负面前缀在 turbo 上等于不存在；控制手段只剩正向 tag、seed、画布。
   **别用文件 sha256 判"两张不同"** —— ComfyUI 把 prompt 写进 PNG 的 tEXt 块，sha 一定变，要比重解码后的像素。
2. **多样性与风格被蒸馏压过**：默认风格很强、出图之间更像。要风格多样性就回 base/aesthetic。
3. **不要给 turbo 挂"为 base 训的"角色 LoRA**（LoRA 必须与基底匹配）；turbo 挂 LoRA 的效果**未验证**。

## CFG × 负面：1.1–1.5 是在"变画面"，不是在"删内容" `[measured, 1 prompt × 2 seeds]`

条件：Anima turbo，800×1200，1 prompt × 2 seeds，paired（同 seed 只改 CFG），耗时取干净窗口的中位数。

| CFG | 现象 | 数字 |
|---|---|---|
| **1.0** | 基线；负面零影响 | 逐像素 MAD 0.0000（2/2 seed） |
| **1.1–1.5** | 画面在变，但**不朝负面词变**：负面直接写 `glasses, blonde hair` 去否正向明确要求的属性 | 像素 MAD 7.8–23.1，属性概率几乎不动（0.970→0.961、0.921→0.914，4/4 同向）；缺陷标签质量两 seed 反而略升 +0.014…+0.088；**唯一稳定效果是降饱和** |
| **≈2.0** | 真删东西 | 缺陷分 0.105→0.044、0.075→0.048，主要抹掉报纸上的 `english text`；代价 = 画面变平 |
| 成本 | CFG>1 每步多算一次 uncond | 单张 **16.5 s vs 11.1 s = ×1.49**；1.1 与 2.0 一样贵 |

- **结论**：默认 **CFG 1.0**；要删内容走 **CFG≈2.0** 并接受变平；**不要用 1.1–1.5**（多 50% 时间、指标不动）。
- **负向词只能压掉非绑定要素**（伪英文 / 水印这类模型自造的附加物）；角色自带的配饰（如荧头上的花）四档 CFG（1.1 / 1.25 / 1.5 / 2.0）× 4 seed **全无效（0/16）**，且每张多花 50–60% 时间 —— 要删只能改正向描述、逐人掩版重绘或后期处理。
- 证据强度：1 prompt × 2 seeds、paired、计时取中位数 ⇒ 只支持上表这些结论，**不要外推成"CFG 与画质"的普适关系**。

## 可选路线：turbo 抽卡 → 放大 → 换权重提细节

**不是必走流程。** turbo 抽到的图直接交付也成立；只有当你认为那张图的细节/质感还不够时，
才往下走这一条。反过来说，同一构图想要多试几个 seed 时，turbo 的单张成本只有 1/4.7。

```
Step 1  turbo 抽卡      800×1200, steps 10, CFG 1, er_sde/simple, 跑 3–5 张选构图   [10.5 s/张]
Step 2  放大            4x-AnimeSharp → ImageScale(lanczos) 到 1200×1800
Step 3  提质感          base / aesthetic / waiANIMA  img2img, denoise 0.30–0.40
                        (已验证 0.35, 25 步 CFG 5, 同一 prompt)
```

- **Step 3 的 denoise 分带**（与 `（未随仓库发布的内部文档）` 一致）：`0.30–0.40` 只换材质与细节、
  不动构图；`0.5–0.6` 才开始改比例；`>0.8` 改结构拓扑。想保构图就别超过 0.4。
- **精修提示词写画布上还看得见的东西**：turbo 底图里没有的元素写了也不该出现，写了反而稀释。
- 整条路线（turbo 底图两段合计，含换模型加载）**114.8 s 产出 1200×1800 成图** [measured]。
- **成本账要算对**：这条路线省的是**抽卡**（10.5 s vs 47.5 s 一张），
  **最后一版精修的代价 ≈ 直接生成一张**。所以：要多试构图才划算；只有一张、构图已定，
  直接按目标权重重新生成一次即可，不用绕这条路线。
- 换权重必须重新加载 3.9 GB 模型，单次加载已含在上面 114.8 s 里；连续跑同一权重不回退。

## 已验证 / 未验证

- 已验证：三档速度对比（800×1200）；`turbo 800×1200 → 4x → Lanczos 1200×1800 → waiANIMA 0.35`
  全程在 6 GB 上不 OOM。
- 未验证：精修尺寸再往上（1600×2400）；turbo 挂角色 LoRA；turbo 与 Illustrious 组合。
- **放大别一步跳到精修尺寸以外**：精修在 `1200×1800` 做完之后，再按交付尺寸放大
  （`upscaling.md`），不要先放大到 4× 再进采样器。

## 可抄的图

- 精修段（Step 2+3）成品图：`anima_turbo_route_workflow.json`（API 格式，直接喂
  `scripts/comfy_generate.mjs`）。用之前把选中的 turbo 底图复制成
  `input/dsh_edit/turbo_draft.png`（staging 目录，随便清），或改 `LoadImage` 的 `image` 字段。
  图里有两个 PreviewImage：节点 5 = 只放大、节点 14 = 放大+精修，方便对照。
- 抽卡段（Step 1）就是标准 Anima 图：`anima_workflow.json` 改成
  `unet_name=anima-turbo-v1.1.safetensors`、`steps=10`、`cfg=1`。
