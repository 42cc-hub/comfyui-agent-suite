# comfyui-agent-suite

让 agent 学会"**用 ComfyUI 出图 / 局部重绘 / 与人协作**"的一套工具。核心是 `skill/`（agent 读的说明书 + 脚本），
`libs/` 下是若干**可独立成库**的组件 —— 按需装，不强迫你复制作者的环境。

- **不含数据集、不含任何原图**；模型权重走 Release 下载。
- **`skill/` 本体几乎没有依赖**：驱动脚本是 Node（只用内置模块），Python 侧只要 `Pillow` + `numpy`。
- 需要"检测/抠图/闸门"这些重活时，再装对应的 `libs/*`（各自声明自己的依赖）。

## 模块成熟度（**先看这张表再用**）

有些模块是打磨过的，有些还在开发中 —— 说清楚，免得你按"全都可靠"来用：

| 模块 | 状态 | 说明 |
|---|---|---|
| **检测（M2a 人/手/脚/头、M2b 面部四类）** | ✅ **相对可靠** | 按源图切分 + 多种子验证过；头部框尤其准（掩版的"头"就用它的**内接椭圆**） |
| 局部质量分类（M3） | ⚠️ 可用但样本少 | 只对"结构错/糊/缺件"这类粗判可靠；缺陷样本量是瓶颈 |
| **骨架（M1 关键点）** | ❌ **不可靠，开发中** | 一直没专门做过：关节会飘、细点（鼻/嘴）落错位置、强重叠姿态下人↔骨架配对会失败。**别用 M1 关键点推掩版**，要用检测框 |
| **绳子（rope）** | ❌ **不可靠，开发中** | 低对比/暗场景漏检明显，正样本像素占比低导致训练信号被稀释；每轮结论见 `docs/` |
| 人在回路画布 / 闸门 / 抠图 / 控制图 | ✅ 可用 | 见各 `libs/*/README.md` |

**下一步要做的**（欢迎一起做）：
1. **骨架数据集**：M1 需要一批**人工修正过**的骨架标注才能修 —— 这是当前最值得投的一块数据；
2. 绳子模块：更难的评测集（现在留出集只有 5 张，看不出 1~2 个点的改进）+ 采样/正权侧改动；
3. 头/手/脚的**局部重绘掩版**继续用检测框推（不依赖关键点）。

## 组件与依赖

| 组件 | 作用 | 依赖 | 许可证 |
|---|---|---|---|
| `skill/` | **核心**：SKILL.md + 参考资料 + ComfyUI 驱动 + 自检/放大/小工具 | Node 18+；Python: `Pillow`,`numpy` | MIT |
| `libs/human-canvas/` | 人在回路画布（+ 可选 Photoshop 通道、Windows 托盘提醒） | Node 18+（DSH 插件形态）；PS 可选 | MIT |
| `libs/wd14-gate/` | 出图安全闸门（WD14 打标 / NSFW 判定 / BLOCK 的人工授权通道） | `onnxruntime`,`numpy`,`Pillow`,`pandas` | MIT（模型另有许可） |
| `libs/sam-matting/` | 抠图 / 掩版合成（SAM 提示式分割） | `torch`,`opencv-python`,`Pillow`,`numpy` | MIT（SAM: Apache-2.0） |
| `libs/blender-controlnet/` | Blender 生成 depth / OpenPose / 手部控制图 | Blender 3.x+（脚本用其自带 Python） | MIT |
| `libs/comic-layout/` | 漫画分格排版与嵌字 | `Pillow` | MIT |
| `libs/anime-detection/` | 二次元人体/部件/面部/质量检测与评估（训练+评测+推理） | `torch`,`ultralytics`,`opencv-python`,`Pillow`,`numpy`（标注 GUI 另需 `PyQt5`） | **AGPL-3.0** |

## 安装

```bash
git clone --recursive <this-repo> comfyui_agent_suite
python install.py                 # 把 skill 装到 DSH skills 目录 + 生成 config/paths.json
python install.py --with-models   # 顺便下 SAM 与 WD14 模型（可选）
```

`install.py` 只做四件事：装 skill（`--dest` 可改，默认 `~/.dsh/skills/comfyui-image-generation`）、
生成 `config/paths.json`（子库绝对路径 + ComfyUI 地址）、按需下模型、打印自检命令。
**密钥不进仓库**：`config/keys.example.json` 是模板，复制成 `keys.json` 后填（已在 `.gitignore` 里）。

子库的依赖按需装：

```bash
pip install -r requirements.txt                     # skill 本体（Pillow / numpy）
pip install -r libs/wd14-gate/requirements.txt      # 只装你要用的
pip install -r libs/anime-detection/requirements.txt
```

## 第一次跑：安装成功探针

**`skill/examples/hutao_furina_two_girls.md`** —— 给一段提示词，出一张
"胡桃 + 芙宁娜同框、且左右与靠肩方向不许反"的图：Anima 底图 → 人体定位 → 按重叠中点切两份互不重叠掩版 →
逐人掩版重绘 → 自检。**跑通即说明安装正确**，而且这套打法（位置交给掩版、身份交给提示词/ LoRA）本身就是
这套工具的主线思路。

## 推荐配置（作者本机，只是参考）

Windows + `conda` 两个环境（一个跑检测/训练：`torch`+`ultralytics`+`segment_anything`+`opencv`+`PyQt5`；
一个跑训练的新版 torch），Node 18+，Blender 5.x，ComfyUI 本机 7187 端口。
**最小可用**是另一回事：只要 ComfyUI + Node + `Pillow`/`numpy`，`skill/` 的出图与重绘路线就能跑。

## 许可证与合规

各组件按目录各自许可（见上表与各库 `LICENSE`）；整体组合请遵守其中最严格的 **AGPL-3.0**
（`libs/anime-detection` 依赖 Ultralytics；闭源商用需另行获取 Ultralytics 商业许可）。
本仓库**不包含任何训练数据或原图**；`libs/anime-detection` 的权重在私有数据集上训练，
使用者自行负责其使用场景的合规性。
