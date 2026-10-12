# 产线集成说明（anime_detection_suite）

> 给"产线那边"（也就是 ComfyUI 出图后、投稿前的自检/精修环节）看的接口文档。
> 2026-10-11 起生效。**改动接口前先改这份文档。**

## 1. 一个入口：`scripts/locate.py`
```powershell
python scripts\locate.py --src <图或目录> --out <输出目录> [--mask] [--conf 0.3]
```
产出**每张图一个 JSON**（统一点列契约）+ 控制台汇总；`--mask` 时另出 `mask/<name>_hands_feet.png`。

## 2. 输出契约（`<out>/<name>.json`）
```jsonc
{
  "image": "07_escape_restraint_tightened_2000.png", "size": "1368x2000",
  "chain": ["M1.ours", "M2a.ours"],          // 实际跑过的层，可追溯
  "n_people": 1,                             // = M1 骨架链数（旧字段，保留）
  "n_persons": 1, "n_skeletons": 1,          // 2026-10-13 新增：M2a person 框数 / M1 链数（二者本就会不一致）
  "points": [ {"layer":"M1","chain":"M1#0","name":"body.left_wrist","cls":"body","x":812.5,"y":402.0,
               "prob":0.91,"vis":2,"person_id":0}, … ],
  "boxes":  [ {"layer":"M2a","cls":"foot","box":[x0,y0,x1,y1],"prob":0.58,"person_id":0,"origin":"det"}, … ],
  "boxes_all": [ … ],                        // 2026-10-13 新增：boxes ∪ 兜底补的部位（origin: det|fallback）
  "flags":  [ {"type":"no_hand_visible"},
              {"type":"part_check","part":"foot","index":1,"state":"bare","verdict":"bad","prob":0.97,"box":[…]} ],
  "ropes":  {"mask":"data/masks/rope/<name>_mask.png","conf":"…_conf.png","source":"M4(rope module)"}   // 若该图有
}
```
- 坐标一律**图像绝对像素**；`vis`：2 可见 / 1 遮挡但可判 / 0 无。
- **`person_id`（2026-10-13 新增）**：把点/框挂到具体某个角色上 —— M1 的每条链按关键点包围盒归属，
  其余框按"中心落在哪个 person 框内、否则 IoU 最大"归属（-1 = 没归上）。
  下游做**逐人掩版重绘**（多角色属性串扰的结构性解法）就靠它；`boxes`/`boxes_all`/`points` 都有这个字段。
- **`n_persons` / `n_skeletons`**：`n_people` 只是 M1 链数，实测会出现"M2a 给 3 个 person 框、M1 只给 2 条链"
  （其中 2 个框是重叠伪框，0.44/0.31）这类不一致 ⇒ 下游要按**链**来认角色，person 框只用来配 id / 过滤置信度。
- **`boxes_all` + `origin`**：`boxes` 一直**不含**兜底补的部位（专用脚模型 / 踝点外推 / 专用手），
  做掩版请用 `boxes_all`（`origin=fallback` 的就是这些）。`boxes` 本身没变（老行为）。
- `flags` 是给产线看的**判定**：`part_check.verdict` ∈ `ok` / `bad` / `skip`（`skip` 由 WD14 的 `state=covered/shod` 直接给出，
  表示"穿鞋穿袜数不了趾"）；`no_hand_visible` / `no_person` 是构图提示；`*_unavailable` 表示本环境缺该层模型。
- **`foot_tight_check`（2026-10-11 新增，默认开，`--no-m3-foot-tight` 可关）**：脚再按"下 40% + 宽 1.2×高"收紧裁一次
  单独判一遍的**并列旗标**（`verdict` / `votes` / `box` 同 `part_check`）。它**不改变** `part_check` 的语义，
  所以下游可以选择性地使用：只认主判定 ⇒ 已知缺陷 3/4；**主判定 ∪ 它 ⇒ 4/4**（交付图 08 的"趾多"只被它报出）。
  代价：它比主判定更爱开火（实测 4 张语料图 17 条 vs 主判定 7 条），所以**建议把它当"低优先级复核"**而不是等同于 `bad`。
  根因见 `docs/runs.md` R39/R40：M3 的训练裁图是"紧贴远端脚"，产线给的是"整只脚+腿"。
  ⚠️ **2026-10-12 补充（R45）**：把 `--imgsz` 提到 **1600** 时，检测器先找到了脚，08 的"趾多"**进了主判定**
  ⇒ 主判定 3/4 → **4/4**（不再依赖上面这条在验收图上扫出来的收紧规则）。代价是旗标 16 → 20（`--m3-conf 0.8` 可压到 20 以内）。

## 3. 分层与降级（每层都可单独替换）
| 层 | 本仓库默认权重（2026-10-11 复核） | 缺失时的回落 | 备注 |
|---|---|---|---|
| M1 骨架 | `runs/m1_pose_coco17` **+** `runs/m1_pose17_1129`（取检出人数多者） | imgutils `dwpose` | 用 OpenPose18 schema 训关键点学不动（mAP50 0.01 vs COCO-17 0.835） |
| M2a 框 | `runs/m2a_det_996img_960`（person/hand/foot/head）+ `runs/m2a_foot_v1`（单类脚，取并集） | imgutils `person/hand/head` | 检测侧已不是瓶颈：交付图人 8/8、脚框 7/8<br>**手是最弱项**（1280 下 4/8、1600 下 5/8）⇒ 2026-10-12 加了可选的 `--hand-weights`（单类手检测器，与 4 类取并集；**默认关**，等 `runs/m2a_hand_v2` 按交付图实测后再决定是否开） |
| M2b 面部 | `runs/m2b_face_v3_3983`（**输入是 head 裁剪的 256×256**） | imgutils `face/eye` | mAP50 0.993；固定画布是关键 |
| M2b-gag | `runs/m2b_gag_v2` **+** `runs/m2b_gag_v4`（取并集，同目标 IoU>0.6 去重） | 无 | v2 保 gag 交付图 4/6；v4（含 pixiv 眼罩 176 框）让 blindfold val mAP50 0.967 |
| M3 判定 | `runs/m3_cls_aug_v1` + `runs/m3_review_v1`（**任一 p(bad) ≥ `--m3-conf` 即可疑**，默认 0.5 = 原行为） | 无（记 `unknown`） | 先用 WD14 state 过滤，再判 ok/bad；脚另有 `foot_tight_check` 并列旗标。<br>⚠️ **这两个权重的 val 训练集泄漏 42%/58%（R43）**，它们的 val 数字不可引用；诚实指标见本文 §6。 |
| M4 绳子 | **不在这里** —— 由 rope 模块单独跑，产物按约定路径被本入口引用 | — | `data/masks/rope/<name>_{mask,conf}.png` |

## 4. 运行环境（2026-10-11 更新：**产线可以直接用 ComfyUI 自己的 env**）
| 用途 | 环境 | 说明 |
|---|---|---|
| **产线跑 `locate.py`（推荐）** | **`E:\conda\envs\sdenv`** | **实测可用、不需要装任何东西**：它有 torch 2.13+cu130 / ultralytics 8.4.123 / onnxruntime 1.29，而我们的权重**不依赖 imgutils**（软依赖，缺了只是回落不可用）⇒ 在 ComfyUI 同环境里直接调用最省事。<br>⚠️ 只是**用它的解释器跑脚本**，不往那个环境里装东西（避免污染 ComfyUI）。 |
| 开发/训练 | `E:\conda\envs\me2` | torch+ultralytics，训练与验收都在这里 |
| 预标注（CPU） | `E:\conda\envs\anime_det` | dghs-imgutils + onnxruntime + PyQt5；**没有 torch**（3.14 装不上 dghs-imgutils，两边各有半边） |

```powershell
# 产线调用示例（ComfyUI 所在环境，无需新装）
python <...>\scripts\locate.py --src <图或目录> --out <输出目录> --mask
```
（`--imgsz 1280` 是默认值，不要调小 —— 640 时交付图上人数检出会从 8/8 掉到 5/8。
**2026-10-12 实测（R45）：交付提取建议用 `--imgsz 1600`** —— 手覆盖率 4/8 → 5/8、总框 18 → 29、
M3 主判定 3/4 → 4/4；代价是推理更慢、旗标更多。详见 `docs/runs.md` R45。）

**待办（不急）**：把 5 个权重导出 ONNX ⇒ 产线可以在**完全没 torch** 的环境里跑（现在不必要，因为 sdenv 已有 torch）。

## 5. 建议的产线用法（自检 → 人工 → 精修）
1. 出完图，跑 `locate.py --mask`；
2. 读 JSON：`flags` 里有 `part_check.verdict=bad` 或 `no_person` 的 → **交人看**（这一步替代现在"人眼逐张数手指"）；
3. 需要局部重绘时，用 `mask/<name>_hands_feet.png`（手/脚）与 rope 模块的 `<name>_mask.png`/`_conf.png`（绳子）
   一起喂 `SetLatentNoiseMask`；`conf` 图是 0~255 灰度，可当**软蒙版**直接省掉羽化；
4. 人工修完的结论回填到 `data/cls_queue/index.csv` 的 `label`（下轮训练用）。

## 6. 已知限制（写清楚，别让产线误信）
1. **脚召回仍是最弱项**（mAP50 0.577 / R 0.541）：脚底朝镜头、被绳缠、2000px 大图上都会漏 ⇒ 别把"没报错"当"脚一定没问题"；
2. M3 的诚实指标：**按源图分组的 5 折折外（382 张 / 167 bad）** —— 单模型召回 **0.74**（阈值 0.5，
   ok 误报 0.14）、2 模型并集 **0.79**（误报 0.19）；**折间方差 ±0.165**，所以别信单次切分的 val。
   ⚠️ 现在默认的 `m3_cls_aug_v1` / `m3_review_v1` 的训练集与公共 val 重叠 **42% / 58%**（R43）——
   它们的 val 数字（召回 0.94）**不可引用**；干净替身是 `runs/m3_clean_v1_s2026{1,2,3}010`（见 `docs/handover_20261012.md`）。
3. 手常被藏（绳缚题材 6/8）⇒ `no_hand_visible` 是**正常**状况，不是错误；
4. M4 绳子学的是"**红绳**"（白绳/黑带在标注里是负样本）——这条来自 rope 模块自己的失败模式记录。
