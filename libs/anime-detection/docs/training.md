# 训练手册（anime_detection_suite）

> 2026-10-10 起。环境＝conda `anime_det`（`E:\conda\envs\anime_det`）；**不用 venv、不用 docker**（用户定的）。
> 目前 env 里**还没有 torch/ultralytics**（只装了 dghs-imgutils + onnxruntime 用于预标注）。

## 1. 装依赖（一次性）
```powershell
conda activate anime_det
nvidia-smi                       # 先看驱动支持的 CUDA 版本，再决定下面装哪个 cuXXX
pip install ultralytics
# 若需要 CUDA 版 torch（Windows，按 nvidia-smi 选）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
- 只想先跑通流程：CPU 版 torch 也能训（慢），但 `imgsz` 要降到 512、`epochs` 降到 10 做冒烟。
- **显存只有 6 GB ⇒ 训练时不要同时跑 ComfyUI 渲染**（会 OOM）。

## 2. 数据怎么来（全部已就绪）
```powershell
python scripts\prelabel_batch.py --escape --sample 32   # 预标注：点列/框/裁图 + 可视化
python scripts\wd14_autotag.py                          # WD14 自动 state/mismatch/bad_*
python scripts\export_labels.py                         # det/pose 导出 + cls 切分（cls 需先有人工 label）
python scripts\build_dataset.py                         # 生成 ultralytics 数据集 + yaml
```
目录：
| 用途 | 路径 |
|---|---|
| 检测（M2a/M2b） | `data/ds_det/`（images/labels + `det_body.yaml`，类别 person/hand/foot/head） |
| 姿态（M1） | `data/ds_pose/`（`pose_body.yaml`，`kpt_shape=[18,3]`，类别 person） |
| 分类（M3） | `data/cls/{train,val}/{ok,bad_count,bad_shape}/`（由 `index.csv` 的人工 `label` 列切分） |

## 3. 训练（6 GB 显存配方）
```powershell
python scripts\train.py --task det  --data data/ds_det/det_body.yaml   --model yolo11n.pt      --epochs 100
python scripts\train.py --task pose --data data/ds_pose/pose_body.yaml --model yolo11n-pose.pt --epochs 150
python scripts\train.py --task cls  --data data/cls                    --model yolo11n-cls.pt  --epochs 80
# **头/脸类小目标检测**必须加 `--aug face`（见下）
python scripts\train.py --task det --data data/ds_face_gag/gag.yaml --model yolo11n.pt --epochs 150 --imgsz 256 --aug face
```

### 3.1 增强：头/脸类小目标要关掉 mosaic（2026-10-13 用户提的）
用户原话的意思：面部 YOLO 训练一般带**切片形式的数据增强**（mosaic：四张图拼一格），
但**二次元的嘴常常就是"一条线"**，一切片就不可识别了 ⇒ 这种增强会**污染**训练。
`--aug face` 就是为它准备的预设：`mosaic=0, close_mosaic=0, erasing=0, scale=0.25, translate=0.05`
（`erasing` 一并归零：随机擦除可能正好盖住眼/嘴）。单项可用 `--mosaic/--erasing/--scale/...` 覆盖。

⚠️ **已训练完的模型都是开着 mosaic 的**（`runs/*/args.yaml` 实测：`mosaic: 1.0, close_mosaic: 10,
erasing: 0.4`，包括 256 画布的 `m2b_face_v3_3983` / `m2b_gag_v2` / `m2b_gag_v4`）⇒ 在 256 的头裁图上
mosaic 会把 4 张脸拼成一小格、每张只有 ~128 px，正好是"把一条线的嘴切碎"的最坏情形。
**后续任何 M2b 重训都用 `--aug face`**，并安排一次 A/B（同数据同种子，开/关 mosaic）看交付图指标。

**2026-10-12 补：再加 `--degrees 60`（三种子定档）。** 用户报的"侧躺看不到眼睛"实测是**方向**问题
（同一张 head 裁图测试时转 ±30°/60°/90°，0° 只出 1 只眼的侧躺图每个角度都变 2 只）。
在 661 张 head 上跑了 `--degrees 0|30|60` × **各 3 个种子**，验收口径（交付图）与聚合口径又打架一次：

| 档 | mAP50-95 | val 嘴 R | 交付 eye（满 16） | 交付 gag（3 种子 × 6 图 = 18） |
|---|---|---|---|---|
| 0° | **0.513±0.008** | **0.85** | 15.0 | 10/18 |
| 30° | 0.474±0.006 | 0.81 | 16.0 | 14/18 |
| **60°（采用）** | 0.478±0.002 | 0.83 | **16.3** | **17/18** |

**交付 gag：0° vs 60° 的 Fisher 双侧 p = 0.018（显著）**，0° vs 30° p = 0.29、30° vs 60° p = 0.34；
60° 的种子波动还最小（mAP ±0.002）。0° 唯一更强的是聚合定位 mAP50-95 —— 但它交付口塞只有 10/18。
配方：**`--aug face --degrees 60`**（mosaic/erasing 已归零）。详见 `docs/runs.md` R55/R56。
⚠️ 单种子会把结论带偏两次（"30° 提升 gag 召回"两 seed 时成立、三 seed 后消失），**M2b 定档必须 ≥3 种子 + 交付图**。

配方依据（调研，见 `docs/research_2026-10-10.md`）：官方**没有**公开显存规格表 ⇒ 用 `batch=-1` 让框架按 ~60% 显存自测
（首 epoch OOM 自动减半重试）；`imgsz 640→768`、`amp=True`、`cache=False`、`workers=2`，并**从 COCO 预训练权重迁移**。
产物落 `runs/<task>_<data>/`。

## 3.5 首轮实测（2026-10-10，30 张预标注图，GPU=RTX 3060 Laptop 6.4GB）
环境：**`me2`**（torch 2.13.0+cu130 / ultralytics 8.4.128；`sdenv` 也有同样的 torch，但那是 ComfyUI 的 env，**别用**）。

| 跑什么 | 命令要点 | 实测 |
|---|---|---|
| det 冒烟 | `--task det --epochs 3 --imgsz 512` | 显存 **1.64 GB**；推理 7.6 ms/张；mAP 垃圾级（数据太少，仅验证管线） |
| pose pilot | `--task pose --epochs 20 --imgsz 768` | **20 epoch 只用 75 秒**；bbox P 0.54 / R 1.00 / mAP50 0.82，**但 pose 关键点 P/R/mAP = 0** |

结论（写下来免得重复踩）：
1. **6 GB 完全够**：768 分辨率、`batch=-1`（自动）在 30 张数据上没有任何 OOM；
2. **框学得快、关键点学不动**：20 epoch 就能把"人在哪"学出来，而 18 点骨架在 5 个验证实例上完全没起来
   ⇒ ① 关键点要的是**更多图 + 更多 epoch**；② 更要紧的是**我们的 pose 标签本身来自 dwpose 预标注**（未人工修正），
   噪声上限摆在那儿 —— **M1 的瓶颈是标注质量，不是训练配置**；
3. 所以真实顺序是：**先修标注（人）→ 再谈训练**；在此之前所有训练都只当"管线验证 + 自动标注器种子"。

## 4. 评测协议（**别按像素/按框随机切**）
- **按图划分**：`build_dataset.py` 已保证同一张图的样本只出现在一个 split；逃出收容所那 8 张里**固定 3 张进 val**，
  它们是我们唯一有"已知缺陷 ground truth"的图（1 趾少/指多、4 趾多/指畸形、6 趾畸形、7 趾多）。
- 指标：
  - M1/M2：`P / R / F1`（召回优先）+ 每张图的**误检连通块数**；
  - M3：混淆矩阵 + **"4 张已知畸形图是否全部判 bad"**（这是硬指标）；
  - M4（绳子，另见 `docs/rope_prelabel_cv.md`）：Dice / 边界 F + 误检连通块数（细长结构**不用 IoU**）。
- 每一轮结果写进 `docs/runs.md`（表格：日期 / 数据量 / 参数 / 指标 / 结论），**别只留在终端里**。

## 5. 已知限制（写清楚，别自欺）
1. 目前 38 张、样本极少 ⇒ 只能验证"管线通不通 + 预标注质量"，**不能**指望模型能用；
2. 手/脚被藏起来的图占多数（绳缚题材）⇒ M3 的负样本天然多、正样本少，训练时要**按类别重采样**；
3. `unknown` 状态偏多（WD14 在紧裁图上上下文不足）⇒ 要么外扩裁图，要么只把 `state` 当参考特征；
4. M2b（面部组）目前只有"检测框预标注"，**还没进数据集**（要先把 head 裁剪 256×256 的框映射回原图这一步跑成批处理）。
