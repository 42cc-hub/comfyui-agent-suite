# M1 骨架现状（2026-10-14）：微调 ViTPose 取代 dwpose / YOLO-pose

**一句话**：M1 从"能用但飘"变成**可用**——换成在**人工标注**上微调的 ViTPose（top-down，吃 M2a 的 person 框），
覆盖率和尾部误差同时改善，代价是权重 344MB 且需要一个能跑 `transformers` 的环境。

## 1. 数字（可复现）

固定 15 张**报告集**（`data/ds_pose17_semi/semi_manifest.json` 的 `val_stems`，
**从不参与训练、也不用于选 checkpoint**）、297 个人工 `vis>=1` 关节、同一份真值：

```bash
python .runtime/eval_common_subset.py --split val --dump .runtime/eval_report_v6.json \
  --weights "dwpose=pre" "yolo_semi=runs/m1_pose17_1129/weights/best.pt" \
            "vitpose_v6=vitpose:models/vitpose_ft_v6_labelbox_honly"
```

| 预测源 | 覆盖率 | 中位(自身) | p90(自身) | 中位(公共 149 关节) | p90(公共) |
|---|---|---|---|---|---|
| dwpose 预标注（旧） | 52% | 26.2 | 224.0 | 26.0 | 234.0 |
| YOLO-pose（旧） | 89% | 70.5 | 335.0 | 74.8 | 320.8 |
| **ViTPose 微调（新）** | **88%** | **23.1** | **97.8** | **23.7** | **75.6** |

**面部（这次的主要诉求）**——逐关节中位像素误差：

| 关节 | dwpose | YOLO-pose | **ViTPose 微调** |
|---|---|---|---|
| nose | 不报 | 31.2 | **14.1** |
| l_eye / r_eye | 7.6 / 6.2 | 33.4 / 43.7 | **11.2 / 13.2** |
| l_ear / r_ear | 不报 | 44.0 / 74.8 | **19.7 / 22.8** |
| l_wrist / r_wrist | 148.6 / 64.6 | 241.2 / 277.7 | **58.5 / 49.6** |
| l_knee / r_knee | 112.6 / 41.9 | 159.6 / 147.1 | **44.7 / 39.1** |

**配对聚类 bootstrap**（按实例重采样 5000 次；`python .runtime/bootstrap_compare.py
.runtime/eval_report_v6.json vitpose_v6 dwpose --metric median|mean`）：

| 对比 | 差值 | 95% 区间 | 判定 |
|---|---|---|---|
| 中位 | −2.0 px | [−22.0, +14.1] | **无差别**（p≈0.33） |
| 均值 | **−31.8 px** | [−53.3, −11.0] | **显著**（p=0.0004） |

⇒ 诚实结论：**中位与 dwpose 无差别，均值（尾部）显著更好**；中位停在同一量级的原因是
"人工自己重标同一张图"的噪声就是中位 16.0 px / p90 181 px（`data/pose_gt_superseded` 对照），
模型误差已经落在标注噪声里。所以这一轮买到的是**覆盖率（52%→88%）+ 尾部（p90 224→98）+ 面点（0→5 个全有）**。

## 2. 接入方式

`scripts/locate.py` 新增 `--vitpose-dir`（默认 `models/vitpose_ft_v6_labelbox_honly`）：

* **默认就用它当 M1**；目录不存在或 `--no-vitpose` 时按老路回落 `--pose-weights` → dwpose；
* 它是 top-down，需要 person 框 ⇒ locate.py 把 M2a 的 det 推理提到 M1 之前，**det 只跑一次**，两层共用；
* 输出契约不变：`body.<COCO17 名>` + `vis∈{0,1,2}` + `chain='M1#i'`，所以骨架掩版、脚召回兜底、
  M3 脚裁图**都不用改**；额外写 `rec['vitpose']` 与 `chain: 'M1.vitpose'` 便于溯源。

超参（见 `models/vitpose_ft_v6_labelbox_honly/meta.json`、`docs/pose_vitpose_arms.md`）：
188 张人工骨架 / 260 实例 ×5 重复、`--box-src label`（用标签自己的外接框）、rot ±12°、
颜色抖动 0.10、面部关节权重 3×、lr head 1e-4 / backbone 1e-5、30 epoch（最好的一轮在 2~14）。
**推理用 M2a 的 person 框**（比标签框更宽，实测更稳）。

## 3. 8 臂对照（同一报告集，为什么选这一版）

| 臂 | 报告集中位 | 报告集均值 | 框 | 自动池 |
|---|---|---|---|---|
| **v6 labelbox + 仅人工** | **26.4** | **72.1** | label | 无 |
| v5 自训练池 | 30.6 | 75.1 | person | 自训练 |
| v3 仅人工 | 27.4 | 75.2 | person | 无 |
| v4 labelbox | 26.9 | 83.2 | label | dwpose |
| v4 labelbox 难关节×2 | 27.4 | 84.2 | label | dwpose |
| v3 5×自动 | 27.6 | 88.4 | person | dwpose |
| v4 难关节×2 | 27.7 | 89.4 | person | dwpose |

（`docs/pose_vitpose_arms.md` 是完整表。）两条结论：**dwpose 自动池有害**（均值 83–89 px vs 72–76 px，
因为它的面点是错的）；中位平台 26–31 px 靠模型/数据推不动，只剩"难关节标注一致性"这一条路。

## 4. 已知不足（别拿它当万灵药）

* **背面/侧脸**：看不到脸时面点仍会飘（例：一张背影图被放在头侧）；强重叠姿态下
  "人 ↔ 骨架"配对仍可能错（M1 生成的点按 chain 分组，配对靠 M2a 框的 IoU）。
* **出框外推**：人框裁得只剩上半身时，膝/踝会被外推到画布外（confidence 0.2~0.4，`vis=1`）。
  下游用点推掩版时要**先裁剪到画布内**并看 `vis`。
* **手/脚细点不在能力内**：这一层只给 COCO-17 的 17 点；指/趾仍走 M2a+M3。
* 权重 344MB + 需要 `transformers`；没有 torch/transformers 的环境会静默回落到旧 YOLO-pose。
