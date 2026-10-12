# sam-matting

**抠图 / 掩版合成**：用 SAM（提示式分割）把主体或局部抠出来，再合成、评分、对比。

- `scripts/sam_object_mask.py` —— 点/框提示 → SAM 掩版（可 `--grow` / 羽化）。
- `scripts/matte_flat_bg.py` / `matte_from_scribbles.py` —— 纯色背景 / 手涂涂鸦两种抠图路线。
- `scripts/chroma_key.py` —— 色键抠图（更快、边缘更硬）。
- `scripts/mask_compose.py` —— 离线掩版合成（框/圆/多边形 / 并集 / 差集 / 缩放 / 羽化），拒绝覆盖输入。
- `scripts/score_matte.py` / `compare_matte.py` —— 与真值对比评分（抠图质量要量化，不要目测）。

**模型自动下载**：`python scripts/fetch_sam.py`（SAM ViT-B，Apache-2.0）。仓库不含权重。

> **抠图质量对照**：实测在"双人膝枕、长发交错"的图上，本地 SAM 掩版的**内部空洞（漏发缝/漏块）**
> 合计 29.2k px（最大一块 9.2k px），而**生成式分色抠图**（DashScope `wan2.7-image` 输出纯黑白块 → 阈值）
> 只有 3.5k px（最大 353 px），覆盖也更全（38.8% vs 28.9%）。
> ⇒ **要质量且能联网/付费**走那条路（见 skill 的 `references/matting_and_cutout.md` 路线一B）；
> **要离线、要批量、或只要"保原像素的硬边"**用本库的 SAM。
