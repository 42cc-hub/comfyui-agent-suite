# 安装成功探针：胡桃 + 芙宁娜「膝枕」双人图（Anima 直出 + 逐人掩版重绘）

这篇既是**第一次跑通的例子**，也是**安装自检**：它把"出图 → 自检 → 只重绘其中一个人"这条闭环走完。
如果它能跑完，说明你的 ComfyUI、模型、脚本路径、检测子库都接对了。

**为什么选"膝枕"**：两人**强重叠**（一个头枕在另一个腿上）是最能体现**分区可控性**的场景 ——
底图只负责把"谁在哪、什么姿势"做对，细节由**逐人掩版重绘**分别打磨，互不干扰。

- **不需要角色 LoRA**：胡桃、芙宁娜这类经典角色，近两年的二次元底模基本都认识（本探针实测一次成功）。
- **管线**：`anima-base-v1.0` + `qwen_3_06b_base` + `qwen_image_vae`（`euler_ancestral` / `beta` / 30 步 / cfg 6.0 / 1024×1536）。
- **场景要求**：胡桃（坐）穿**水手服**、看着镜头微笑；芙宁娜（躺）穿**死库水**、闭眼睡着、头枕在胡桃腿上；**两人都不戴帽子**（帽子写进反向）。

```
① 底图（不挂 LoRA，只把姿势/关系/服装做对）
      ↓
② 自检 + 逐人掩版（check_parts.py --person-masks）
      ↓
③ 掩版"谁是谁"核对（mask_overlay.py）← 必做，编号/JSON 都不可盲信
      ↓
④ 只重绘坐着那位（denoise 0.45，提示词只写她）
      ↓
⑤ 只重绘躺着那位（底图换成④的结果，提示词只写她）
      ↓
⑥ 自检：人数 / 谁看镜头 / 谁闭眼 / 服装 / 手
```

---

## 0. 前置

| 项 | 说明 |
|---|---|
| ComfyUI | 在跑（默认 `http://127.0.0.1:8188`；本仓库脚本用 `COMFY_BASE` 覆盖） |
| 模型 | `models/unet/anima-base-v1.0.safetensors`、`models/text_encoders/qwen_3_06b_base.safetensors`、`models/vae/qwen_image_vae.safetensors` |
| Node | 18+（`skill/scripts/*.mjs` 只用内置模块，无 npm 依赖） |
| Python | `Pillow` + `numpy`（第 2、3、6 步还需要 `libs/anime-detection`：`torch`+`ultralytics`；逐人掩版用 SAM，`libs/sam-matting/scripts/fetch_sam.py` 自动下载） |
| 自训权重 | `python skill/scripts/fetch_weights.py --from-dir <你的 runs 目录>`（清单见 `weights/manifest.json`） |

命令都在**仓库根目录**下执行；`<skill>` 指 `skill/`。

## 1. 出底图（姿势与关系在这一步定死）

```bash
node <skill>/scripts/comfy_generate.mjs <skill>/examples/workflows/pair_base.json out/pair_base.png
```

底图提示词的三条要点（见 `pair_base.json`）：

1. **位置用画面坐标系**：`the girl on the left side of the image …` / `the right girl lies with her head on the left girl's lap`；
2. **服装/表情写全**：水手服（`sailor uniform, sailor collar, pleated skirt`）与死库水（`school swimsuit, one-piece swimsuit`）、
   `she looks at the viewer, she is smiling` / `her eyes are closed, she is sleeping peacefully`；
3. **不想要的东西写反向**：`hat, cap, beret, headwear, top hat`（帽子与这两套服装不搭），
   以及 `mirrored composition, swapped positions, the sitting girl lies down`（防镜像/防姿势互换）、`one girl only, 1boy, 3girls`（防人数漂移）。

> 想要更多候选就换 `KSampler.seed` 多跑几张，用 `skill/scripts/tools/sheet3.py` 拼成联系表挑一张。

## 2. 自检 + 逐人掩版

```bash
python <skill>/scripts/check_parts.py --src out/pair_base.png --out out/parts --json \
       --person-masks --person-mask-cover auto
```

输出：

- `out/parts/<name>.json`：人数、各自 person 框、眼/嘴/手/脚、M3 质量标记（`suspicious`）；
- `out/parts/mask/<name>_person<i>.png`（灰度，白=重绘区）与 `_person<i>_rgba.png`（**RGBA，重绘区 alpha=0**，直接喂 ComfyUI）。

## 3. 掩版"谁是谁"核对（**这一步不能省**）

```bash
python <skill>/scripts/tools/mask_overlay.py out/pair_base.png out/parts/mask --out out/mask_check.png
```

**为什么必须核对**：`persons_masks[]` 里的 `person_id` / `box` 取自该实例的 `members[0]`，
**两人强重叠时两条记录可能重复**；文件名里的 `person<i>` 又来自构建顺序 ⇒ **两者都可能与你的直觉相反**。
我实测踩过：掩版喂反，用"另一个人闭眼睡觉"的提示词去重绘这一位的脸，**眼睛颜色当场串味**。

判据（叠加图上一眼可辨）：

| 观察 | 结论 |
|---|---|
| 外接框靠上/靠里、覆盖脸 | 这是**坐着/靠前**那位（本例：胡桃，水手服） |
| 外接框靠下、覆盖枕在腿上的头 | 这是**躺着**那位（本例：芙宁娜，死库水） |
| 两块在 x 上左右分开 | 左块=左位、右块=右位 |

确认后把掩版改成**角色名**，避免后面再混：

```bash
cp out/parts/mask/<name>_person<?>_rgba.png out/mask_sitting.png   # 坐着那位
cp out/parts/mask/<name>_person<?>_rgba.png out/mask_lying.png     # 躺着那位
```

## 4. 上传底图与两块掩版

```bash
python <skill>/scripts/tools/comfy_upload.py out/pair_base.png out/mask_sitting.png out/mask_lying.png \
       --sub dsh_ctrl --base $COMFY_BASE
```

（上传器带 `overwrite` —— 否则 ComfyUI 会把同名文件存成 `xxx (1).png`，而工作流引用的是固定名字，
会**静默用到上一轮的旧图**。这个也实测踩到过。）

## 5. 逐人重绘（两遍，串行）

```bash
node <skill>/scripts/comfy_generate.mjs <skill>/examples/workflows/repaint_sitting.json out/pair_step1.png
python <skill>/scripts/tools/comfy_upload.py out/pair_step1.png --sub dsh_ctrl --base $COMFY_BASE
node <skill>/scripts/comfy_generate.mjs <skill>/examples/workflows/repaint_lying.json out/pair_final.png
```

三条硬规矩（本探针的全部价值就在这三条）：

1. **提示词只写"这一块里的人"**：写她的名字、发色、瞳色、服装、表情；
   mask 外的另一位**只用泛称**（`another girl is lying on her lap, only that girl's body is visible`），
   **绝不写她的名字和外貌** —— 否则模型会把她的特征往这块里带。
2. **反向里点名另一位（和另一位身上的一切）的特征**：
   重绘胡桃时反向写 `blue hair, blue eyes, blue streaks, school swimsuit`；
   重绘芙宁娜时反向写 `brown hair, twintails, red eyes, sailor uniform`。这是防串味最有效的一招。
3. **denoise 0.40~0.55**：低了改不动，高了会把这一块的脸重画（实测 0.55 就开始"换脸"）。

> ⚠️ **不要做"把重绘结果按掩版贴回原图"**（`ImageCompositeMasked` 那类）。
> `SetLatentNoiseMask` 本身是干净的：**掩版干净时，掩版外不会串味**（本探针用"胡桃的眼睛颜色"验证过）。
> 贴回属于旁门左道，边缘一定出问题。

## 6. 自检（别用眼睛当唯一判据）

```bash
python <skill>/scripts/check_parts.py --src out/pair_final.png --out out/final_check --json
python <skill>/scripts/tools/sheet3.py out/compare.png out/pair_base.png out/pair_step1.png out/pair_final.png \
       --labels "底图,坐着那位重绘,躺着那位重绘(终)" --cell 430x645
python <skill>/libs/wd14-gate/scripts/gate_review.py out/pair_final.png      # 出图闸门
```

验收清单：

| 约束 | 合格标准 |
|---|---|
| 人数 | `n_persons == 2` |
| 谁看镜头 / 谁睡 | 坐着那位**睁眼看镜头微笑**；躺着那位**闭眼**、表情放松 |
| 姿势 | 躺着那位的头**枕在**坐着那位的腿上（不是反过来、不是并排） |
| 服装 | 一个水手服、一个死水着（死库水）；**两人都没有帽子** |
| 五官 | 每张脸 2 眼 1 嘴；手指数正常（`suspicious` 里不该有 `hand#*`） |
| 分区可控 | 只重绘 A 时，B 的脸/眼**不该变**（本探针用眼睛颜色当哨兵） |

## 坑（本探针实测踩到的，按疼的程度排）

| 坑 | 现象 | 解法 |
|---|---|---|
| **掩版喂反/拿错** | 用"闭眼睡觉"的提示词重绘了另一个人的脸 ⇒ 眼睛颜色串味 | 第 3 步的叠加核对；掩版按**角色名**重命名 |
| `persons_masks[]` 的 id/box 重复 | 强重叠姿态下两条记录指向同一个 person | 别信 JSON 的 id，**看像素外接框**（`mask_overlay.py` 会打印） |
| 上传重名 | 工作流引用 `pair_base.png`，实际存成了 `pair_base (1).png` ⇒ 用了旧图 | 上传器 `overwrite=true`；或换子目录 |
| 提示词写了两个人 | 另一位特征被带进来（发色/服装/瞳色） | 只写目标那位；另一位用泛称；反向点名另一位特征 |
| denoise 过高 | "重绘"变成"换脸" | 0.40~0.55 |
| 只写"左/右" | 容易被镜像 | 画面坐标系措辞 + 反向 `mirrored composition` |

## 这套打法还能怎么用

同一个骨架换概念即可：**加物件**（打底 + 小区域重绘）、**修手修脚**（用 `check_parts.py` 的
`hand#*`/`foot#*` 定位后局部重绘）、**两个新角色同框**（那时才需要逐人挂各自的 LoRA ——
见 `libs/anime-detection/README.md` 的评测协议与 `references/human_in_the_loop.md`）。

---

**推荐配置（作者本机，仅作参考，不是硬要求）**：Windows + `conda` 两环境（一个：`torch`+`ultralytics`+
`segment_anything`+`opencv`+`PyQt5`；一个：新版 `torch`），Node 18+，Blender 5.x（仅 `libs/blender-controlnet` 需要），
ComfyUI 本机 7187 端口。**最小可用**：ComfyUI + Node + `Pillow`/`numpy` 就能跑第 1、4、5 步；
第 2、3、6 步需要 `libs/anime-detection`（AGPL-3.0）。
