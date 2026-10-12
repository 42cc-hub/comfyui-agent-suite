# anime-detection

二次元插画的**人体 / 部件 / 面部 / 质量**检测与评估套件：训练、消融、评测协议、推理入口。
用途是给"出图 → 自检 → 修图"这条闭环提供**可量化的眼睛**：人/手/脚/头有没有画错、口塞/眼罩在不在、
局部画得是不是"糊/结构错"。

> **权重与数据都不在本仓库**：数据集私有，权重走 Release（`skill/scripts/fetch_weights.py` 或手动下载）。

## 成熟度（**用之前先看**）

| 模块 | 状态 | 怎么用 |
|---|---|---|
| **M2a 检测：人 / 手 / 脚 / 头** | ✅ 相对可靠 | 掩版一律由**检测框**推：头用**头框的内接椭圆**，手/脚/头/脸用框（`check_parts.py --feature-masks`） |
| **M2b 面部四类（eye/mouth/gag/blindfold）** | ✅ 相对可靠 | 单权重；gag 阈值取 0.4、eye/mouth 取 0.25；gag/blindfold 按"每张 head 有无"计数，不数框 |
| M3 局部质量分类 | ⚠️ 可用、样本偏少 | 只信"结构错/糊/缺件"的粗判；缺陷样本量还在攒 |
| **M1 骨架 / 姿态** | ❌ **不可靠，开发中** | 关节会飘、鼻/嘴细点位置错、强重叠姿态下人↔骨架配对会失败（实测出现"某个 person 框因 no_skeleton 被丢弃"）。**不要用关键点推掩版** |
| **rope 绳索** | ❌ **不可靠，开发中** | 暗场景 / 红底低对比漏检明显；正样本像素占比 1~3% 导致信号被稀释 |
| 评测协议（下节） | ✅ 这部分才是本库最值钱的 | 按源图切分、多种子、交付图优先、先建负样本集 |

### 路线（roadmap）
1. **骨架数据集**：M1 要修必须有**人工修正过的骨架标注**；计划做成一套"检测框 → 骨架初值 → 人修正 → 回训"的小工具链。当前最优先。
2. **rope**：先把评测集做大（现在留出集 5 张，±0.04，看不出 1~2 点改进），再动采样/正权（`pos_weight` 单独调过无效 ⇒ 要试"每批固定正样本比例"）。
3. **人↔骨架配对**：强重叠（膝枕/拥抱）时的一对一分配要改成"按骨架框与 person 框的几何一致性打分"，而不是贪心。
4. 掩版：头/手/脚继续走检测框（头=头框内接椭圆）；关键点只做参考，不做掩版依据。

## 组件与任务

| 代号 | 任务 | 入口 |
|---|---|---|
| M1 | 人体关键点（COCO17） | `scripts/locate_limbs.py`、`runs/*pose*` |
| M2a | 人 / 手 / 脚 / 头 检测（含专用手、脚模型） | `scripts/locate.py`、`scripts/ablate_m2a_imgsz.py` |
| M2b | 面部四类：**eye / mouth / gag / blindfold**（一个权重） | `scripts/build_face4.py`、`scripts/eval_face4.py`、`scripts/ab_face4.py` |
| M3 | 局部质量分类（结构错 / 糊 / 缺件…） | `scripts/eval_m3.py`、`scripts/cv_m3.py`、`scripts/sweep_m3_thresh.py` |
| rope | 绳索像素级检测（独立任务，与人体检测只共享评测框架） | `scripts/rope_*.py` |

**推理总入口**是 `scripts/locate.py`：给一张图，输出人物、骨架、各部件框、口塞/眼罩、M3 质量标记的 JSON
（`<skill>/scripts/check_parts.py` 是它的 agent 友好封装）。

## 评测协议（这个套件最值钱的部分，不是模型）

1. **按源图切分**（同源图的裁图只落一遍）—— 早期漏了这条，val mAP 0.99 是假的（77% 泄漏）。
2. **多种子 mean±std**：单次运行方差可达 ±0.026。
3. **交付图验收优先于聚合指标**：mAP 更高不等于交付更好（本套件实测多次"聚合指标与验收口径打架"）。
4. **先建负样本测试集再谈误报**：用"两个弱信号都说没有"的样本当负集，其标签可信度先在人判样本上量过。
5. 数据构建脚本要**自带硬校验**（图/标签数不匹配或 train/val 源图重叠 ⇒ 直接判数据集不可用并非 0 退出）。

详见 `docs/training.md`、`docs/integration.md`、`docs/runs.md`（实验日志，含每次结论与踩过的坑）。

## 快速开始

```bash
pip install ultralytics torch opencv-python pillow numpy
python scripts/train.py --task det --data <你的 dataset.yaml> --model yolo11n.pt --imgsz 256 --aug face
python scripts/locate.py --src IMG.png --out out/            # 推理：出 JSON + 可视化
```

## 许可证

**AGPL-3.0**（见 `LICENSE`）—— 因为依赖 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）。
若你需要闭源商用，请向 Ultralytics 购买商业许可，或把训练/推理后端换掉。
