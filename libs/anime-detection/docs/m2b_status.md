# M2b 面部组（eye / mouth / gag / blindfold）现状（2026-10-21）

**一句话**：M2b 的**标注工具已到可用状态**（画框器 + 6 个目标 + 五个队列），
**模型现状本次未复核** —— 下面表格里的指标全部标注了来源与复核状态，没有一项是本次实测的。

## 1. 工具现状：`scripts\label_face.py`（本次实测）

`scripts\label_face.py`，798 行 / 35.4 KB，PyQt5，队列驱动（CSV 一行一轮）。

| 项 | 现状 | 依据 |
|---|---|---|
| 交互 | 空白处拖 = 新建框；框内拖 = 移动；四角拖 = 改大小；`1`=画好/下一轮（没画框即"没有这个目标"）；`3`=没有；`A`=整张 head 全部预填一次采纳；`Z` 撤销、`C` 清空、右键删框、`S` 保存、`U` 只看未判、`G` 跳轮 | `scripts\label_face.py` 文件头 §键位（第 18–21 行） |
| 目标类别 | `TARGET_CN` 六类：`gag` / `blindfold` / `eye` / `mouth` / **`hand`** / **`foot`**（后两类是 2026-10-14 起把同一个画框器借去修 M2a 手脚框，见 `docs\m2a_boxfix.md` §做了什么） | 同上第 42–44 行 |
| `--pre` | 裁图目录，默认 `data/prelabel`；修 M2a 手脚框时指 `data/boxfix/crops` | 同上 `--pre` help（第 779 行） |
| 其他参数 | `--queue`（默认 `data/m2b_queue/index.csv`）、`--title`、`--only-unlabeled`、`--selftest` | 同上第 778–783 行 |
| 预标注护栏 | **只有未判轮次才注入预标注**；已判轮次永远显示 CSV 里的真值；`A` 遇到"该目标没有预填"的轮次会**停下让人画**，绝不替人猜"没有" | 文件头第 7–16 行 |
| clamp bug | **已修**：`mouseReleaseEvent` 原把框坐标硬夹到 `[0,255]`（给 M2b 256×256 脸裁图写的常量），换到 M2a 后裁图/预填框 >255 ⇒ 拖有动画、松手被打回。现夹到**图像尺寸** | 修复与回归验证：`docs\m2a_boxfix.md` §途中修的工具 bug（第 34–39 行）；第 2 批实测 5 条坐标 >255 生效 |
| 离屏自检 | **19/19 通过** | 2026-10-21 本机实跑，命令见下；旧记录是 12/12（`docs\runs.md` 第 372 行），之后加了 `A` 键等 7 项检查 |

复现自检（离屏、不弹窗、不占 GPU）：

```bash
python scripts\label_face.py --selftest
# 末行：自检结果：19/19 通过   （exit 0；stderr 会刷 "QFontDatabase: Cannot find font directory"，不影响结果）
```

## 2. 队列现状（本次实测行数）

统计方式：`Import-Csv <queue>\index.csv` 后按 `verdict` 列分组（`draw`/`neg` = 已判；空 = 未判；`pass`/`fail` = 10-11 前的旧口径，仍按已判计）。

| 队列 | 总行 | 已判 | 未判 | 已判构成 |
|---|---|---|---|---|
| `data\m2b_queue` | 578 | 578 | 0 | draw 160 / neg 385 / fail 33 |
| `data\m2b_queue2` | 198 | 198 | 0 | draw 69 / neg 129 |
| `data\m2b_queue3` | 386 | 385 | 0(空值 1) | draw 151 / neg 234 |
| `data\m2b_queue4` | 1151 | 1151 | 0 | draw 616 / neg 513 / fail 22 |
| `data\m2b_queue5` | 1950 | 1242 | **708** | draw 914 / neg 328 |
| 合计 | 4263 | 3554 | 709 | — |

⇒ 只有 `m2b_queue5`（四类覆盖补全队列）还有 **708 行没判**，其余四个队列已判完。

## 3. 模型现状：**未复核**

`locate.py` 的 M2b 默认权重（`scripts\locate.py` 第 78–80 行）：

```python
DEF_FACE = runs/m2b_face_v3_3983/weights/best.pt        # 256 固定画布
DEF_GAG  = [runs/m2b_gag_v2/weights/best.pt, runs/m2b_gag_v4/weights/best.pt]   # 并集
```

磁盘事实（本次只核了文件在不在、多大、什么时候写的）：

| 权重 | 文件 | 大小 | 写盘日期 |
|---|---|---|---|
| `runs\m2b_face_v3_3983\weights\best.pt` | 在 | 20.7 MB | 2026-10-11 |
| `runs\m2b_face4_v3_rot60_s20261013\weights\best.pt` | 在 | 5.2 MB | 2026-10-12 |
| `runs\m2b_face4_v3_rot60_s20262013\weights\best.pt` | 在 | 5.2 MB | 2026-10-12 |
| `runs\m2b_face4_v3_rot60_s20263013\weights\best.pt` | 在 | 5.2 MB | 2026-10-12 |
| `runs\m2b_gag_v2\weights\best.pt` | 在 | 5.2 MB | 2026-10-11 |
| `runs\m2b_gag_v4\weights\best.pt` | 在 | 5.2 MB | 2026-10-11 |

**指标数字（全部是引用，不是本次实测）**：

| 引用值 | 来源 | 复核状态 |
|---|---|---|
| face4 四类 val mAP50-95 **0.476~0.487**（三个 seed） | `docs\toolchain_inventory.md` 第 15 行 | **未复核** |
| `ds_face_gag` 的 val 干净：gag **0.842** / blindfold **0.967** | `docs\toolchain_inventory.md` 第 16 行 | **未复核** |
| `m2b_face_v3_3983` val mAP50 **0.993** | `docs\handover_20261011.md` 第 17 行、`docs\integration.md` 第 51 行 | **已作废，不要引用**：`docs\runs.md` R52（第 921–929 行）查出 `ds_face` 的 train/val 有 **343/444 张是同一批裁图**（抽查 5 张 sha256 全一致）⇒ 0.993 是被泄漏撑起来的 |
| face4 现役三 seed 里 s20263013 最好（交付图 gag 6/6，另两个 5/6） | `weights\manifest.json` 对应条目（已发 `weights-v2`） | **未复核** |

复核 M2b 三件事的命令（都还没跑，任何一条跑完都能把上表换成实测值）：

```bash
# ① 交付图验收（现有脚本可直接用）
python scripts\eval_face4.py --weights runs\m2b_face4_v3_rot60_s20263013\weights\best.pt
# ② 多权重并成一张表
python scripts\ab_face4.py
# ③ 端到端（M1→M2a→M2b→M3 一条链，输出 JSON 契约）
python scripts\locate.py --image <图> --classes face,gag,blindfold
```

⚠️ 跑 ①②③ 都要占 GPU：先持票（`<你的 ComfyUI 安装目录>\agent_work\team\gpu_claim.py acquire --owner keeper --level 1 ...`）。

## 4. 未验证项 / 风险

1. **合成缺陷/污染数据未评估**：`data\m3_gen`、`data\colorblock_ds` 与 M2b 无关，但 M2b 自己的**难例**（侧脸、遮挡、只框住球的口塞）没有量化过；`scripts\find_gag_ballonly.py` 只做过"口径混杂"的定位，没做过"口径混杂对指标的影响"。
2. **face4 与 face/eye 两代权重的关系没有对照实验**：`locate.py` 仍用 `DEF_FACE`（旧 face/eye）+ `DEF_GAG` 并集，**face4 三 seed 完全没接线**（`weights\manifest.json` 里发了，代码里没用）。这是 M2b 最大的缺口，且与 M2a 的"权重修好但没接线"是同一类问题。
3. **`m2b_queue5` 708 行未判**：这决定 face4 能不能补数据的最后一块，判完才能谈"要不要换掉旧 face/eye"。
4. 本文件的工具部分（§1、§2）可在无 GPU 环境复现；§3 的全部指标在复核前**只能当参考**。
