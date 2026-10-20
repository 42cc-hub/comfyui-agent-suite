# 工具链清单（套件内 scripts\ 全部脚本按线归类）

**用途**：这个套件的 `scripts\` 里有 100 多个 `.py`，其中一半只回答过一个问题就该归档。本文按"线"归类，并给每个脚本标 **入口**（日常/产线会反复用，能一条命令复现某个结论）或 **一次性诊断**（为某个具体问题跑过、结论已落进 docs/，留作证据、不必再跑）。

**数量是移动目标，别引用本文的张数**：实测 2026-10-21 03:0x 为 100 个、03:5x 为 **102** 个 —— 写本文的 40 分钟里 `dualpose_pick.py` / `dualpose_viz.py` / `dualpose_probe.py` / `m2a_eval_holdout.py` 被陆续加进来（M1 双人诊断线 + M2a holdout 复核）。要当下准确数就现跑：

```powershell
(Get-ChildItem <ANIME_DET_SUITE>\scripts -Filter *.py -File).Count
```

本文按**条目**维护（哪个脚本属于哪条线、是不是入口），不按张数。

---

## 1. 模块与权重现状（一行一个模块，谁在用、能不能用）

| 模块 | 交付级 | 现役权重（`locate.py` 默认） | 关键数字 / 状态 | 状态文档 |
|---|---|---|---|---|
| **M1 骨骼** | **是** | `DEF_POSE=[runs/m1_pose_coco17, runs/m1_pose17_1129]`；`DEF_VITPOSE=models/vitpose_ft_v6_labelbox_honly`（行 54-58） | ViTPose 覆盖 88% / 中位 23.1px / p90 97.8；YOLO-pose 89% / 70.5 / 335.0（15 张报告集、297 关节） | `docs\pose_m1_status.md` |
| **M2a 人/手/脚/头框** | 是（默认值还是旧的） | `DEF_DET=runs/m2a_det_996img_960`（行 68） | 4 类 person .984 / hand .914 / foot .694 / head .978（旧口径）；**新权重 v2 实测 0.373→0.695（脚 0.000→0.604）、误报 0/9，但 `DEF_DET` 还没指过去** | `docs\m2a_boxfix.md` |
| M2a 专用脚/手 | 脚=否，手=待接 | `DEF_FOOT=runs/m2a_foot_v1`；`--hand-weights` 默认 **None** | 专用脚 `m2a_foot_v1` 在修框验收池全 0.000（v1 命中 14.6%）⇒ 别指望；专用手 v3 中位 0.633→**0.810**（验证 0.8870→0.8757）**但没接默认** | `docs\m2a_boxfix.md` §5 |
| M2a 脚远端 | 否（已判失败） | — | `m2a_foot_distal_v1` 不达交付 | `docs\m2a_boxfix.md` |
| **M2b 脸/眼罩/口塞/手/脚** | 是 | `DEF_FACE=runs/m2b_face_v3_3983`；`DEF_GAG=[m2b_gag_v2, m2b_gag_v4]`（行 78-80） | face4 三 seed val mAP50-95 0.476/0.475/0.487；`ds_face_gag` 干净口径 gag 0.842 / blindfold 0.967；⚠️ `m2b_face_v3_3983` 的 val 0.993 **因泄漏不可引用**（R52，见 §4） | `docs\m2b_status.md` |
| **M3 局部质量分类** | 是（但指标不涨） | `DEF_M3=[runs/m3_cls_aug_v1, runs/m3_review_v1]`（行 71-72，并集：任一判 non-ok 即可疑） | 固定 382 集 acc 0.7801 / AUC 0.8225 / bad 召回 0.6587；⚠️ 现役两权重 val 被污染（R43）；无泄漏替代 `runs/m3_clean_v1_s2026{1,2,3}010` **未复核** | `docs\m3_status.md` |
| **M4 绳子** | **是** | `data/masks/rope/weights/rope_unet_resnet18_g14.pt`（thr 0.90） | holdout F1 0.9277±0.0117 / 召回 0.9812 / **precision 0.862 是瓶颈**；无 torch 兜底 `rope_mlp_k9_4img.npz` | `docs\rope_module.md` §0 |
| 线稿/色块/线棍 | 否（探索线） | — | 见 `docs\colorblock_lineart_assessment.md` / `docs\colorblock_linewand.md` / `docs\colorblock_sam_or_wan.md` | 同左 |

---

## 2. 100 个脚本：按线归类

标记：**【入口】** = 日常/产线会反复用；**【一次性】** = 问题已回答完，可归档（文档留数字，脚本留证据）；**【私有】** = 明确不发公版（`release_build.py` 第 53-54 行）。

### 2.1 总入口与产线编排（1 + 2）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\locate.py` | **【入口】** | 整套的**唯一总入口**：`--image <图> --classes hand,foot` → JSON 契约（M1 骨骼 + M2a/M2b 框 + M3 质量）；改默认权重的唯一地方 |
| `scripts\pipeline_datasets.py` | **【入口】** | 一条命令跑完"预标注 → 数据集"，参数与流程写在文件调用处 |
| `scripts\train_detached.py` | **【入口】** | `Start-Process` 起独立进程训练，避免 agent 会话中断带走训练（配 `train_status.py`） |

### 2.2 标注 GUI（4 个，都要人上手，都是入口）

| 脚本 | 标记 | 作用与键位 |
|---|---|---|
| `scripts\label_pose.py` | **【入口】** | M1 骨骼标注器（81KB，最大的一个）；`--queue data/pose_queue` |
| `scripts\label_face.py` | **【入口】** | M2b 六类（gag/blindfold/eye/mouth/hand/foot）+ **M2a 修框画框**（`--pre data/boxfix/crops`）；`--selftest` 现 **19/19**；clamp bug 已修（`docs\m2a_boxfix.md` 第 34-39 行） |
| `scripts\label_cls.py` | **【入口】** | M3 二分复核器；`--queue data/cls_review` |
| `scripts\prefill_queue.py` | **【入口】** | 把预标注结果预填进队列 index.csv，人只改不画 |
| `scripts\locate_limbs.py` | 【入口】 | 按关键点裁手脚放大图，核对时看大图不去翻原图 |

### 2.3 预标注 → 数据集（10 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\prelabel_batch.py` | **【入口】** | 批预标注（det/pose → queue），产线前置 |
| `scripts\prelabel_vitpose.py` | **【入口】** | 用 ViTPose 跑骨骼预标注 |
| `scripts\build_dataset.py` | **【入口】** | 队列 → YOLO 数据集（det/cls 分折）；公版与本地内容已不同（发布差异里有它） |
| `scripts\export_labels.py` | **【入口】** | 预标注标签导出/切分（23.7KB，参数最多） |
| `scripts\build_boxfix_queue.py` | **【入口】** | M2a 修框队列构建（把 miss/偏移的框裁出来等人画） |
| `scripts\export_boxfix.py` | **【入口】** | 修框结果 → 训练集增量 |
| `scripts\mix_boxfix.py` | **【入口】** | 把修框增量与原始集按比例混合（"纯修框集会把别的类打坏"的反面证据在这） |
| `scripts\export_foot_only.py` | **【入口】** | 单类数据集导出（脚/手专用模型用） |
| `scripts\export_foot_distal.py` | 【一次性】 | 脚远端专门集导出，路线已判失败 |
| `scripts\refresh_autolabels.py` | 【一次性】 | 自训练刷新伪标签 |

### 2.4 M2b（脸/遮物）队列与数据集（8 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\m2b_propose.py` | **【私有】** | 用公开脸检测器提议 M2b 候选框（含第三方私有路径，不发） |
| `scripts\m2b_complete_coverage.py` | 【一次性】 | 补全覆盖队列（face_v1 漏检的脸补出来） |
| `scripts\build_face4.py` | 【一次性】 | 组 face4 四类数据集（产物已在盘上） |
| `scripts\build_m2b_cover_queue.py` | 【一次性】 | 组 `m2b_queue5`（1950 行；已判 1242 / **未判 708**） |
| `scripts\build_mouth_eye_queue.py` | 【一次性】 | 嘴/眼专用队列 |
| `scripts\export_face_extra.py` | 【一次性】 | 脸类增量导出 |
| `scripts\face_geom.py` | 【一次性】 | 用眼线推嘴位置（几何先验实验） |
| `scripts\find_gag_ballonly.py` | 【一次性】 | 找"只有口塞球没有带子"的难例 |
| `scripts\eval_face4.py` | **【入口】** | M2b 交付图验收（4 类逐图对账，选 `DEF_FACE` 指哪个就靠它） |

### 2.5 M2a 评测与探针（3 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\eval_boxfix.py` | **【入口】** | M2a 验收：留出集 40 图/45 条（36 正 9 负）上算命中率/中位 IoU/误报，**这口径不是 val mAP** |
| `scripts\m2a_eval_holdout.py` | **【入口】** | 同一套 holdout（`data/boxfix_holdout.csv`，conf 0.25 / imgsz 960、同一批裁图）上补 **per-class recall/precision**；precision 是**下界**、误报有 strict/loose 两口径，读数字前先看文件头的口径说明 |
| `scripts\ablate_m2a_imgsz.py` | 【一次性】 | 输入尺寸消融 |
| `scripts\det_probe.py` | 【一次性】 | 检测器单图探针（看某个框为什么出/不出） |

### 2.6 M1 骨骼（6 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\train_vitpose.py` | **【入口】** | ViTPose 微调（8 臂实验的主入口，22.4KB） |
| `scripts\eval_pose_finals.py` | **【入口】** | 交付图选权重（决定 `DEF_VITPOSE` 指哪个） |
| `scripts\pose_eval_vs_human.py` | 【入口】 | 与人工关键点的差距（bootstrap 置信区间的来源） |
| `scripts\pose_semi_build.py` | 【一次性】 | 半监督数据集构建（w10/ctrl 两臂） |
| `scripts\dualpose_pick.py` | 【一次性】 | 挑"双人 + 外接框重叠"图，供双人归并诊断（CPU） |
| `scripts\dualpose_probe.py` | 【一次性】 | 双人归并三层归因探针（检测 NMS 0.7/0.5/0.3 → 框本身合并 / 姿态 inst_sep / 归属错配），复刻 `locate.py` 默认口径 |
| `scripts\dualpose_viz.py` | 【一次性】 | 双人诊断的两张图：候选 contact sheet + 叠加图 |

### 2.7 M3 分类与阈值（24 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\cv_m3.py` | **【入口】** | 固定 382 张集**按源图分组 5 折 OOF**，输出 `data\cv_m3_oof.csv`（0.7801/0.8225/0.6587 的来源） |
| `scripts\eval_m3.py` | **【入口】** | 单个权重的同一套指标（消融与人工报告共用，避免两套口径） |
| `scripts\sweep_m3_thresh.py` | **【入口】** | 阈值扫描（产线并集口径、纯离线）—— 0.40 那个决策项的表就从这条命令重扫 |
| `scripts\acceptance_m3.py` | **【入口】** | 交付图验收：4 处已知缺陷是否都报出来 |
| `scripts\retrain_m3.py` | **【入口】** | M3 重训一条命令（防手敲参数） |
| `scripts\cls_active_queue.py` | 【入口】 | 主动学习挑样（也是 **R43 泄漏的发现处**，文件头有记录） |
| `scripts\cv_m3_synth.py` | 【一次性】 | 合成数据对照实验（路线已判死，9 张注入无效果） |
| `scripts\eval_m3_convention.py` | 【一次性】 | 口径交叉评估（验证"两套口径不能混用"） |
| `scripts\cls_queue_sheet.py` | 【一次性】 | 队列做联系表 |
| `scripts\build_cls_all.py` / `build_cls_mix.py` / `build_cls_taxonomy.py` | 【一次性】 | 三个数据集构建器（all / mix / 层级）。**注意别把 `cls_bin_mix`（754 张）当评测基准**：`data\cv_m3_oof.csv` 的 764 条是 382 张 × 2 seed，与它无关（详见 `docs\m3_status.md` §1） |
| `scripts\compare_taxonomy_oof.py` | 【一次性】 | 层级 vs 二分 OOF 对比 |
| `scripts\ablate_m3_crop.py` | 【一次性】 | 裁图范围消融 |
| `scripts\add_pred_to_queue.py` | 【一次性】 | 把预测回填进队列 |
| `scripts\recut_crops.py` / `recut_crops_prod.py` | 【一次性】 | 按新框重裁（两个版本，prod 是改过的那个） |
| `scripts\gen_m3_defects.py` | 【一次性】 | Anima-Turbo 生成坏样本（58 张里只有 17.3% 可用） |
| `scripts\m3_crops_from_gen.py` | 【一次性】 | 从生成图切裁图 |
| `scripts\m3_review_sheets.py` | 【一次性】 | 做人工复核联系表（10/18/30 那张表的读图工具） |
| `scripts\m3_verified_pool.py` | 【一次性】 | 复核过的池子汇总（bad 9 / ok 17） |
| `scripts\pixiv_collect.py` | **【私有】** | Pixiv 采集 |
| `scripts\wd14_autotag.py` | **【私有】** | WD14 自动打标 |
| `scripts\tag_cls_queue.py` | **【私有】** | 队列打标 |
| `scripts\mine_candidates.py` | **【私有】** | 难例挖掘 |

### 2.8 M4 绳子（16 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\rope_predict.py` | **【入口】** | M4 产线入口（固定 CLI，`--weights/--thr/--tile/--out`，产三件套 mask.png/overlay.png/result.json） |
| `scripts\rope_train.py` | **【入口】** | MLP 版训练（无 torch 兜底那条线） |
| `scripts\rope_unet.py` | **【入口】** | U-Net 训练（交付权重出自这里，52.5KB） |
| `scripts\rope_eval.py` | **【入口】** | 评测 KPI："人擦几笔"= 误检块 + 漏检段 |
| `scripts\rope_clean_labels.py` | **【入口】** | 标注卫生，每次标完必跑 |
| `scripts\rope_mask_to_label.py` | **【入口】** | 人标 mask → 训练标签 |
| `scripts\rope_neg_pick.py` | **【入口】** | 负样本挑样（`--confirm` 才写盘） |
| `scripts\rope_prelabel_ps.py` | **【入口】** | PS 预标注的导出/升格 |
| `scripts\rope_prelabel_cv.py` | 【一次性】 | 传统 CV 预标注 v3（被神经网络取代，52.8KB 留着当对照） |
| `scripts\rope_refiner.py` | 【一次性】 | SegRefiner 路线实验 |
| `scripts\rope_make_coarse.py` | 【一次性】 | 造"粗标注→精标注"训练对 |
| `scripts\rope_thr_study.py` | 【一次性】 | 阈值研究（纯离线，结论已进 rope_module.md） |
| `scripts\rope_oof.py` | 【一次性】 | OOF 评测 |
| `scripts\rope_loo_class.py` | 【一次性】 | 留一类（LOO）实验 |
| `scripts\rope_ablate.py` | 【一次性】 | 消融（G3/G11/G12/G13/G14 那一批） |
| `scripts\rope_label_audit.py` | 【一次性】 | 标签审计 |

### 2.9 工具库：SAM / 色块 / 线稿（9 个）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\colorblock.py` | **【入口】** | 色块分块工具（`--mode compete`，18.8KB）；公版与本地内容已不同 |
| `scripts\sam_point_mask.py` | **【入口】** | 点选 → SAM 单张蒙版 |
| `scripts\sam_multi.py` | **【入口】** | 多点/多框批量 SAM |
| `scripts\lineart_anime_local.py` | 【入口】 | 本地 AnimeLineart 出线（不依赖云端） |
| `scripts\build_colorblock_ds.py` | 【一次性】 | 色块数据集构建 |
| `scripts\colorblock_targets.py` | 【一次性】 | 色块 → 线稿目标 |
| `scripts\lineblock.py` | 【一次性】 | S0 实验 |
| `scripts\train_linewand.py` / `eval_linewand.py` | 【一次性】 | 线棍（line wand）训练/评测，探索线 |

### 2.10 训练基建（2 个，配 2.1 的 train_detached）

| 脚本 | 标记 | 作用 |
|---|---|---|
| `scripts\train.py` | **【入口】** | 统一训练入口（ultralytics；6GB 保守配方注释在文件里）；公版与本地不同 |
| `scripts\train_status.py` | **【入口】** | 训练进度/曲线/剩余时间一眼看 |
| `scripts\release_build.py` | **【入口】** | 发布构建（幂等覆盖 `F:\comfyui_agent_suite`） |

### 2.11 一次性探针与外部对照（12 个，全部可归档）

`ab_face4.py`（face4 数据量臂）、`ablate.py`（消融 harness）、`ablate_gag.py`、`baseline_imgutils.py`（imgutils 外部基线）、`compare_with_imgutils.py`（与外部基线同图对比）、`hair_dp_parse.py`、`hair_dp_twostage.py`、`hair_sam_probe.py` + `hair_sam_probe2/3/4.py`（头发 SAM 6 件套，多类头发已搁置）。

**小计（逐条数过，合计 102 = 03:5x 实测张数）**：**【入口】43** / **【一次性】54** / **【私有】5**。口径：一个脚本只归一行，多用途归最主要那条线；"入口"= 产线与日常流程会反复用，"一次性"= 结论已进 docs/、脚本留作证据。

**结论**：**53 个脚本（一半以上）本身可以归档**，但**别删**——它们是 `docs\` 里那些数字的唯一可复现证据；真正的成本不在磁盘（都是几 KB~几百 KB），而在"新人打开 `scripts\` 看到 100 个文件不知道从哪进"。所以对外只暴露 5 个入口即可：

```
locate.py          # 产线：一张图 → 定位 + 质量 JSON
label_pose.py / label_face.py / label_cls.py   # 人要画的东西
cv_m3.py + eval_boxfix.py + rope_eval.py + eval_face4.py   # 四个模块的验收
release_build.py   # 发布
```

---

## 3. 哪里需要人标注（现状，不是计划）

| 位置 | 还差什么 | 量（实测） |
|---|---|---|
| `data\boxfix` + `boxfix2` | M2a 手脚框修正 | **已判完**（24 + 200 行） |
| `data\m2b_queue5` | 脸/眼罩/口塞/手/脚 | 1950 行，**未判 708**（queue1-4 已判完，合计 3554 行已判） |
| `data\pose_queue` | M1 骨骼 | 922 行，已判 184（ok 46 / edit 138），**未判 738** |
| `data\pose_gt` | 骨骼真值 | 218 个文件（另有 47 个 superseded） |
| `data\cls_bin_all` | M3 二分复核 | 394 张（ok 221 / bad 173） |
| `data\cls_bin_mix` | M3 混合池 | 754 张（ok 384 / bad 370） |
| M4 绳子 | 已标 85 张（79 训 / 5 留出） | 留出只有 **5 张**，这是 M4 最大缺口 |
| 头发多类 | 已搁置 | — |

---

## 4. 已知泄漏三处（引用指标前必须看）

| 编号 | 位置 | 事实 | 出处 |
|---|---|---|---|
| **R39** | M3 数据集切分 | 旧 `cls_bin_*` 的 val 与 train 有同源裁图 ⇒ 一切 val 指标偏乐观；改用 `cv_m3.py` 按源图分组 OOF | `docs\runs.md`、`docs\handover_20261012.md` |
| **R43** | `m3_cls_aug_v1` / `m3_review_v1` | 这两个现役权重的 **val 被污染：42% / 58% 的评测裁图就在它们 train 里** ⇒ **它们的 val 指标一律不能引用** | `scripts\cls_active_queue.py` 文件头 |
| **R52** | `ds_face` | train/val 有 **343/444 张同一批裁图**（抽查 5 张 sha256 全一致）⇒ `m2b_face_v3_3983` val mAP50 **0.993 不可引用** | `docs\runs.md` 第 921/929 行 |

---

## 5. 真"乱"在哪（给整理时的落点）

1. **三套标注器**（`label_pose` / `label_face` / `label_cls`）界面逻辑各写一遍 ⇒ 可收敛成一个 `--task {pose,face,cls}`；但 `label_face.py` 已经吃下 M2a 修框，收敛时要先把它拆出"画框模式"。
2. **实验权重与交付权重都在 `runs\`**，靠目录名区分（`*_v1/_v2`、`s2026xxxx`）⇒ 交付权重建议在 `runs\<名>\DELIVER` 落一个空标记文件。
3. **队列目录 30+ 个**（`cls_queue_*` 16 个、`m2b_queue1-5`、`boxfix*`、`pose_queue`），过期的该并进 `_archive`。
4. **三处泄漏要显眼**（见 §4）——本文和每个状态文档开头都要提。
5. **`rope_*` 15 篇、`pose_*` 7 篇**文档可以各归并成一篇 + 附录，现状数字进"状态"文件，实验过程进"result"文件。
6. `data\masks\rope\weights\` 里躺着 **1.6GB** 的消融权重（含 3 个 332MB 的 ViT-B）——它们只在正文里回答过一次对比问题，属可归档。
