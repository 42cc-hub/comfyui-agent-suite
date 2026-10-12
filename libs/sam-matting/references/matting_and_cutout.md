# 抠图 / 去背景（Cutout & Matting）

给生成图或现成插画**去背景、拿 alpha 通道**。和本 skill 其它部分的关系：出图用 ComfyUI，
抠图用这里的两条路线 —— **本地 SAM**（保原像素）和 **托管图像编辑**（保语义、会重绘）。

## 先判断输入类型，这决定难度（差一个数量级）

| 输入 | 难度 | 原因 | 用哪条路线 |
|---|---|---|---|
| 自己生成的图（提示词里写了 `plain white background`） | **易** | 单一物体 + 平坦底色 | `matte_flat_bg.py`，批处理，零人工决策 |
| 官方海报 / 他人插画 | **难** | 背景杂乱（文字/贴纸/纹理）+ 道具 + 半透明特效 | `matte_from_scribbles.py`（要人涂个粗轮廓）或托管编辑模型 |
| 纯色（绿幕/蓝幕）底的图 | **易** | 底色可解析 | `chroma_key.py` |

**实测数字**（同一台机器）：

| 输入与路线 | 结果 |
|---|---|
| 19 张 Anima 生成图 → `matte_flat_bg.py` | SAM 得分均值 **0.961**，最终连通域 **1~3 个**（均值 1.5），逐张目视干净 |
| 官方海报 → 本地 SAM + 人涂轮廓 → 对人工 GT | **IoU 0.737**（自动选提案）；手工挑提案最好 **0.779**；误留 0.49%、边界误差 4.7px |
| 官方海报 → 百炼 `qwen-image-edit-max` 重绘 + `chroma_key.py` | IoU 0.62~0.76，**误留最多 9.3%**；但去剑/去火焰/补遮挡一次完成 |

## 路线一：本地 SAM（保原像素，首选）

模型：`models/sams/sam_vit_b_01ec64.pth`（Meta SAM ViT-B，357 MB，已放在本 skill 目录内）。
依赖：`pip install segment-anything`（几十 KB 的纯 Python 包；权重不需要联网下载）。

为什么是 SAM：它按**物体性**分割，不在乎"白头发和白背景同色"。同一条任务上试过的两条
颜色/边缘路线都失败了，原因记下来避免重走：

- **GrabCut**：颜色模型把白色头发判成背景（白底白物），并会把细发丝整段删掉；
- **线稿堤坝（从画布边缘灌水、遇线稿停）**：思路对（半透明火焰没有线稿会被自动排掉），但这张
  海报背景里有大面积深色（PHOENIX 大字、黑胶唱片、噪点），堤坝占 52% 的像素、灌水被切碎。
  若改用**人涂的背景层**当水源则可行，但复杂海报上仍不如 SAM 直接。

```bash
# A. 生成图批处理（平坦背景）
python libs/sam-matting/scripts/matte_flat_bg.py "out/**/*.png" --out matte_out

# B. 海报：先让人在画布上涂"人物"和"背景"两层，再跑
python libs/sam-matting/scripts/matte_from_scribbles.py poster.png \
    --fg fg_scribble.png --bg bg_scribble.png \
    --subtract sword_mask.npy --out matte_out
# 打印每个 SAM 提案的面积/IoU，自动选择不满意时用 --pick N 覆盖

# C. 单独取一个物体（例如先把剑抠出来再当 --subtract 传入）
python libs/sam-matting/scripts/sam_object_mask.py --image poster.png --out sword.npy \
    --box 130,460,420,1400 --pos "245,900;265,1000" --neg "285,800"
```

### 三条把人带沟里的"启发式"（都有实测代价，别再用）

在海报这张图上，凡是**绕过人的涂抹、自己发明规则**的做法都删掉了真实身体部分：

| 错误规则 | 代价 |
|---|---|
| "只保留最大连通域" | 删掉左侧长马尾（1.87%）+ 左小腿（0.40%）——它们只是因为剑的掩码把它们与躯干切开了 |
| "质心 x>500 的连通域都是火焰" | 删掉腰侧**红色羽翼饰板**（0.55%）——它是不透明服装件，不是火焰 |
| "饱和度低的连通域都是火焰" | 把整个**人物**删了（白裙子本来就低饱和） |

**正确做法**：**逐连通域的取舍交给人涂的那层轮廓**。实测它与人工 GT 的判断在**每一个**连通域上
都一致（7/7）。脚本就是按这个规则写的。

### 另一个真实坑：悬空件与原图缺像素

剑的心形护手压在长发与头部的连接处，**那段像素在原图里根本不存在**。任何抠图算法都只能得到
"悬空的一缕头发"。要么接受，要么用生成模型补（见路线二），要么让人在 PS 里接。

## 路线一B：生成式分色抠图（DashScope `wan2.7-image`）—— **实测比 SAM 干净，不发缝**

思路：不要求模型"输出 alpha"，而是让它**输出一张纯黑白分割图**，再阈值化成 matte。

```bash
python scripts/bailian_image_edit.py edit IN.png matte_block.png \
  "把图中两个女孩（包含头发、发丝边缘）的像素全部涂成纯白色 #FFFFFF；其余一切（背景、沙发、灯、家具）
   涂成纯黑色 #000000。输出一张纯黑白的分割图：不要保留任何细节、不要阴影、不要渐变、不要灰色过渡、不要文字。" \
  --model wan2.7-image
# 阈值化（>200 当主体；80~200 的灰阶可以直接当**软 alpha**，边界自带羽化）
python scripts/tools/mask_from_colors.py matte_block.png --out-dir out/matte \
  --rgba-base IN.png --colors "white=#FFFFFF" --tol 60
```

**实测对比**（2026-10-12，同一张双人膝枕图，1024×1536；"内部空洞"= 不接触画面边界的背景连通块，
也就是常说的**发缝/漏块**）：

| 掩版来源 | 覆盖 | 内部空洞 | 最大空洞 |
|---|---|---|---|
| **Wan 黑白分色** | **38.8%**（含发丝外扩） | **53 个 / 合计 3.5k px** | 353 px |
| 本地 SAM 两层并集 | 28.9% | **96 个 / 合计 29.2k px** | **9.2k px** |
| 检测掩版并集（合成轮廓，非 matte） | 37.0% | 0 | — |

⇒ **SAM 的空洞面积是 Wan 的 8.4 倍**（还有一块 9.2k px 的大洞），这正是用户说的"**SAM 会漏发缝**"；
Wan 的覆盖更高、空洞小一个数量级。**代价**：按次计费、输出常被放大（要**最近邻**缩回原尺寸）、
边界是生成的 ⇒ 抠完**务必叠一张预览核对**（`mask_from_colors.py --preview`），
关键交付物仍建议与 SAM 结果**求交/求并**后人工过一眼。

**价格**（官方价，2026-10-12）：默认（服务端改写提示词）**$0.027504/张 ≈ ¥0.20**；
加 `prompt_extend=false`（**本工具现在的默认**）**$0.01434/张 ≈ ¥0.10**。
关改写对抠图是**又便宜又听话**（"涂纯白/纯黑"这种指令不该被改写）：同图实测空洞
3,469 → 5,373 px，两模式 IoU **0.95**（同一质量档）。要旧行为加 `--prompt-extend`。

**什么时候还该用本地 SAM**：完全离线、批量几百张、或者只要"保原像素的硬边"（生成式抠图会重绘边缘几个像素）。

## 路线二：托管图像编辑（百炼 / DashScope），保语义但会重绘

适合"**去道具、去半透明特效、补全被遮挡区域**"这类**语义**需求——这是本地抠图做不到的。

```bash
export DASHSCOPE_API_KEY=...        # 或 --key-file path/to/key.json
python scripts/bailian_image_edit.py list
python scripts/bailian_image_edit.py edit poster.png green.png \
    "把图中的人物抠出来放到纯绿色背景#00FF00上，不要剑，不要火焰光效，保留人物所有细节" \
    --model qwen-image-edit-max
python libs/sam-matting/scripts/chroma_key.py green.png cutout.png     # 出 alpha
```

**必须知道的三件事**（都会让你白跑一趟）：

1. **它是重绘，不是分割。** 同一张输入只换提示词，两次输出彼此 alpha IoU 只有 **0.586**，
   人物尺寸都会变（保守提示词下多留 9.3% 的画幅）。要保原像素就别用它。
2. **不返回 alpha。** 输出 PNG 是 RGB，所以必须"要纯色底 → 本地 key"。key 质量很好：
   背景色残留 **0.000%**、despill 后无绿边。
3. **接口细节**（每条都是踩过的）：
   - `qwen-image-edit*` 是**同步**接口，带 `X-DashScope-Async` → `403 current user api does not support asynchronous calls`；
   - 图像参数要 **http(s) 或 base64 data URL**；`/uploads` 得到的 `oss://` 会被拒（`Image must be either a public URL ... or a Base64`）；
   - `/api/v1/uploads?action=getPolicy` 要用 **GET**（POST → 405）；OSS 表单必须带
     `x-oss-object-acl`、`x-oss-forbid-overwrite`（否则 `403 Policy Condition failed`）；
   - 输出尺寸会被改写（879×1442 → 800×1312，等比 0.91），落盘后**按内容**判断格式，别信扩展名。

## 路线三：纯色底 key

`chroma_key.py`：数边框取背景色 → 距离阈值出软 alpha → **despill**（把背景主通道压到另两通道
均值附近）→ 保留主连通域 + 填内洞 → 距离变换重建 1px 软边。输出旁边会存一张洋红底预览。

## 验收：别只报一个数

```bash
python libs/sam-matting/scripts/score_matte.py GT.png mine.npy baseline.npy   # IoU / 误留 / 漏检 / 边界误差 + 误差簇
python libs/sam-matting/scripts/compare_matte.py out.png base.png GT.png mine.npy   # 红=漏、蓝=多、绿/洋红=两条边界
python scripts/contact_sheet.py sheet.png 260x380 *.png      # 批量目视
```

`score_matte.py` 会把误差**聚类**并给出 bbox/方位，因为"少了 6%"可能是"整个左小腿没了"，也可能
是"边缘差 1px"，两者要采取的行动完全不同。抠图结论**必须**配 `compare_matte.py` 那张图一起看。

**做人工 GT 时**：让对方把图另存一份、**画布尺寸与人物位置保持不变**、存成带透明的 PNG。
只有对齐了才能逐像素比较（尺寸不一致时脚本会拒绝比较而不是猜）。

## 快速决策

1. 图是自己生成的、且提示词写了纯色背景 → `matte_flat_bg.py`，直接用。
2. 图是别人的/海报 → 先请人涂两层轮廓 → `matte_from_scribbles.py`；需要去道具/特效/补遮挡 → 走百炼，然后 `chroma_key.py`。
3. 任何"看起来很干净"的结果，都用 `score_matte.py` + `compare_matte.py` 对着 GT 验一遍再下结论。
