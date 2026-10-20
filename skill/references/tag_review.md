# Tag review —— 提交前的标准步骤 `[rule]`

> **规则**：把提示词交给 `POST /prompt` 之前，**逐 tag 对照"场景清单"复核一遍**。
> 触发条件命中时（§1）这一步不许跳；它不是"再读一遍"，而是一次**逐项对照**。

## 0. 判据：模型不会自己发现 token 之间的矛盾

- 它把每个 tag 都当独立条件挂上，遇到互斥就忽略其中一个、或画出四不像（`barefoot` + `black pantyhose` ⇒ 一只脚穿袜一只脚光着）。**"我写的时候想清楚了"不成立**：写的那一刻脑子里有场景，token 里没有。
- **成本**：一轮出图 ≈ 2 分钟（批量 ×N，还要人工确认）；逐 tag review ≈ 几十秒 ⇒ **一律做**，包括"看起来很简单"的那次。跳过的代价不是"没想清楚"，而是"没有一步专门用来发现自己没想清楚"。

## 1. 什么时候必须做（任一命中即做）

1. 提示词的**来路不是为当前场景从头推出来的**：LoRA 触发词、抄来的工作流、"点子"参考图、上一版 prompt、**别的角色/别的 task 的 prompt**；
2. **场景需求变了任意一项**：角色 / 服装（含内衣、袜、鞋）/ 暴露度 / 姿势动作 / 道具 / 地点 / 时间天气 / 视角构图 / 画幅；
3. 换了**底模或 LoRA**（自带风格块与 outfit 词会一起变）；
4. **批量生成前**（一次查清，省 N 轮）；
5. 用户 brief 本身含互斥或省略项（"穿裤袜但不穿鞋"、"还是那套衣服但把外套脱了"）——必须**落成显式向/否向**，不许听天由命；
6. 任何**暴露度相关** tag 改动之后 ⇒ 顺带**重跑 WD14 gate**（`libs/wd14-gate/scripts/wd14_gate.py`；gate 的判据就是标签与像素，改标签会改判定）。

## 2. 判据：先写"场景清单"，再逐 tag 对照

清单写的是**事实**（不是提示词）：

```
角色/身份 · 服装(上衣/下装/内衣/袜/鞋) · 暴露度要求 · 姿势动作 · 道具与手持物 ·
地点场景 · 时间天气 · 视角与构图 · 画幅 · 一致性锚点(seed/参考图/掩版)
```

**做法**：把 prompt 里每个 tag 归到清单的某一项。**归不进去的 = 可疑项** —— 要么删掉，要么补进清单并说明它为什么该在。这一步是"逐 tag"，不是"通读"。

## 3. 冲突检查表（本项目高发，按类别查）

| 类别 | 互斥 / 需同现 | 典型冲突与写法 |
|---|---|---|
| 脚部 | `barefoot`/`bare feet` **⟂** `pantyhose`/`thighhighs`/`socks`/`shoes`/`boots`；`no shoes` ⟂ `shoes` | 要"只穿裤袜不穿鞋" ⇒ 正向只写 `black pantyhose`，`no shoes` 只走否定面，负向加 `shoes, boots, sneakers`，**并删掉 `barefoot`** |
| 腿部 | `bare legs` ⟂ 裤袜/长袜 | 两者同写时底模随机选一个 |
| 上下装 | `shirt` ⟂ `no shirt`/`topless`；`dress` ⟂ `skirt`+`shirt` | 换装必须**显式替换从句**（`prompt_writing.md` §6） |
| 暴露/遮挡 | `nude`/`topless`/`bottomless` ⟂ 同时写的衣着 | **这组直接改变 gate 判定**，改完必须重跑 gate |
| 姿势 | `standing` ⟂ `sitting`/`lying`/`on back`；`arms up`/`hands up` ⟂ `hands in pockets`/`arms crossed`；`spread legs` ⟂ `legs crossed`；`walking`/`running` ⟂ `lying` | 姿势互斥通常表现为"画面像另一个姿势" |
| 视角 | `full body` ⟂ `portrait`/`close-up`/`upper body`；`from above` ⟂ `from below`；`facing viewer` ⟂ `from behind`/`facing away` | 与尺寸/主体像素面积联动（SKILL.md 尺寸节） |
| 场景 | `indoors` ⟂ `outdoors`；`in bed` ⟂ `standing`；`rain` ⟂ `clear sky`/`sunny`；`night` ⟂ `day` | 背景常被"随机"处理，正是漏检区 |
| 道具 | 同手两物（`holding rope` + `holding cup`）；道具与姿势冲突（`suspended` 但 `on bed`） | 绳类拓扑见 `prompt_writing.md` §3 |
| 遮挡 vs 要求露出 | 道具/绳/头发遮住关键部位，而清单要求可见（或反之） | 决定 gate 是"故意遮蔽"还是"穿帮" |
| 质量/风格 | `masterpiece, best quality` 与 LoRA 自带质量块重复堆叠；`simple background` ⟂ 复杂场景；`sketch`/`monochrome` ⟂ 彩色 | 风格块只出现一次（§11） |
| **触发词体检** | LoRA 触发词常自带 **outfit/发色/职业/场景**，与清单冲突 | 必须在正向**显式覆盖**（`opaque fabric, fully covered` 那套），**不要指望负向压过去** |
| 画幅/构图 | `wide shot` 双人 vs 竖构图 832×1216；`full body` 但主体小于 ~700×1200 | 冲突时改画幅或改分镜，不是硬堆 tag |
| 否定面 | 负向里出现清单**要求**的词（负向写 `shoes` 但清单要穿鞋；负向写 `long hair` 但角色是长发） | 负向是"第二块画布"，也要对照清单 |

## 4. 执行步骤（可照抄）

1. 写**场景清单**（一行一项；落进任务日志 / workflow JSON 备注）；
2. 逐 tag 归类 ⇒ 得到**可疑项列表**；
3. 用 §3 表查冲突 ⇒ 每命中一处：删 / 换 / 落否向，并写一行"为什么"；
4. **触发词体检**：LoRA 触发词逐个对照清单（尤其自带 outfit 的）；
5. 检查**否定面**是否有与清单冲突的词；
6. 暴露度相关有改动 ⇒ 重跑 `python libs/wd14-gate/scripts/wd14_gate.py IMG...`；
7. 提交 `POST /prompt`；把"本轮改掉的 tag + 原因"记进 workflow JSON 备注，供下一轮复用。

## 5. 反模式

- ❌ "我写的时候看过一遍了" —— **通读不是 review**；必须逐 tag 对照清单（token ≠ 意识）；
- ❌ "这是上一版跑通的 prompt，应该没问题" —— 场景一变，**跑通的历史就是最大的误导源**；
- ❌ "负向里有就够了" —— 正向写了互斥词时负向往往**压不住**，先删正向；
- ❌ 用"再通读一遍"代替"逐项对照" —— 漏掉的恰恰是自己刚写下的那个词；
- ❌ 发现 brief 自身互斥时**自己替用户改需求** —— 该问就问（半自动模式：用户没说的不做，发现了要说出来）。

## 6. 与其它规则的关系

- **Working mode（半自动）不变**：tag review 是"用户要求的这件事"内部的必要工序，不是自作主张扩大范围；brief 自相矛盾时**提问**，不擅自改需求。
- **WD14 gate**（`references/output_safety_gate.md`）：标签是 gate 的输入之一，暴露度相关 review 之后**必须重跑**。
- **`prompt_writing.md` §11**：tag review 是那份提交前 checklist 的**第 0 项**（先对照清单，再逐条过 checklist）。
- **一致性**：同一 task 内清单要**一字不差地复用**（发型/瞳色/服装锚点），改一项就是改场景 ⇒ 回到 §1 触发条件 2。
