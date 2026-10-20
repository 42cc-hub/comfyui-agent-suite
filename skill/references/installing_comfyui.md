# 从零装一台 ComfyUI（本机实测口径）

> SKILL.md 只写了"启动 / 停止"；这一节补**首次安装**。**按本机实测写，不要搬别人的清单。**

## 1. 取仓库与版本

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git <你的 ComfyUI 安装目录>
cd <你的 ComfyUI 安装目录> && git log --oneline -1     # 固定 commit/tag 再动手，别在浮动分支上装依赖
```

* 自定义节点放 `custom_nodes\`，**逐个装、装完即验证**（版本互斥是主要坑）；
* 本机在用的节点：`comfyui-wd14-tagger`（安全闸门依赖它的模型目录）、其余按任务需要再加；
* 插件装完先 `git submodule status`/各节点自己的 `requirements.txt` 过一遍，别一次装五个再排查。

## 2. Python 环境与 torch / CUDA

* 本机环境：`python`（torch + ultralytics + opencv + PyQt5 + transformers + segment_anything）；
* **新卡分支（Blackwell / sm_120）**：默认 `pip install torch` 装到的 wheel 可能不含对应 arch ⇒
  必须换 `TORCH_COMMAND` / 指定 cu12x 的索引源（本机有实证案例，症状是"能装能跑但 kernel 报 no kernel image"）；
* 装完自检：`python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。

## 3. 模型目录约定

| 类型 | 位置 | 说明 |
|---|---|---|
| 主模型 | `models\checkpoints\` | 本机主链是 `UNETLoader + CLIPLoader + VAELoader`（Anima / qwen 路线），不一定走 `CheckpointLoaderSimple` |
| LoRA | `models\loras\<类别>\` | 命名规范与触发词读取见 `references/lora_sourcing.md` |
| 放大 | `models\upscale_models\` | 2x/4x；**upscale 会发生不止一次**（细物件要像素） |
| SAM | `models\sams\sam_vit_b_01ec64.pth` | 本 skill 自带；抠图/掩版用 |
| WD14 | 见 wd14-gate 的模型目录 | 闸门用；BLOCK 的图不许进 agent 上下文 |

## 4. 启动参数与端口（本机实况）

```bash
python main.py --port 7187 --listen 127.0.0.1
```

* 本机 ComfyUI 跑在 **7187**；`vram NORMAL_VRAM`、async offload 这类按显存取舍（6 GB 卡上别开大 batch）；
* 同机可能有别的 agent 在用 ⇒ 见 §6 的排队礼貌。

## 5. 两个必写的启动坑（都有实证）

1. **db 锁**：启动前删 `user\comfyui.db-wal` / `user\comfyui.db-shm`，否则可能起不来或写不进历史；
2. **沙箱写权限**：**必须以 full access 运行**，否则写库/写设置被拒、进程直接退出，
   错误形如 `[sandbox: file access denied …]` —— 这是策略拒绝，不是程序 bug，别去改代码。

## 6. 排队礼貌（同机多 agent）

* **不 `/interrupt`、不清队列**；
* 只 `POST /prompt` 后轮询 `/history/<id>`；要取消就取消**自己**那一条；
* 一次只排自己需要的量，别把队列塞满。

## 7. 验证安装（三步都过才算装上）

1. `GET /system_stats` → 200，且能看到 GPU；
2. `GET /object_info/LoraLoader` → 200（证明 LoRA 路径配置生效）；
3. 跑一张 **1 图工作流**（`PreviewImage` 结尾，别用 `SaveImage`，见 SKILL.md 的 Cleanup 硬规则），
   再用 `GET /view?filename=..&type=temp` 取回，确认图能落、能取、能看（**先过 WD14 gate 再看**）。
