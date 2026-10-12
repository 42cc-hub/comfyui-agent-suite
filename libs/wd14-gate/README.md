# wd14-gate

**出图安全闸门**：用 WD14 tagger 给图打标 → 按关键词表判 `OK / REVIEW / BLOCK`。

- `scripts/wd14_gate.py`：分类器本体（onnxruntime，CPU 就够）。
- `scripts/nsfw_gate.py`：更轻的前置筛。
- `scripts/gate_review.py`：**BLOCK 之后的人工授权通道** —— BLOCK 不是对作品的判词，而是
  "agent 不许把这张读进上下文"的护栏；由**文件所有者本人**在 `human-canvas` 上涂出要打码/要保留的区域，
  合成遮罩、在**副本**上打码、重新过闸门，并留一份授权记录。之后 agent 看打码图才算获授权。
- **模型自动下载**：`python scripts/fetch_wd14_model.py`（默认下到 `models/`，可用 `--out` 改）。
  仓库不含模型权重。

## 依赖

`onnxruntime`、`numpy`、`Pillow`、`pandas`。WD14 模型（`wd-v1-4-convnextv2-tagger-v2`）有它自己的许可，
首次使用前请阅读模型页面的说明。
