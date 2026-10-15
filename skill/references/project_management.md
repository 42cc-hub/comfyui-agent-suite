# 项目管理：原则与操作原语（通用）

> **适用**：任何"人出意图、agent 执行、产物是文件"的创作项目（生图只是其中一种）。
> **定位**：只写**跨项目不变**的东西 —— 原则、对象、编号、生命周期、操作原语、读取边界。
> 具体项目的额外规则（投稿平台、标签口径、引擎参数）写在该项目的 `STANDARD.md` 里，不写在这里。

---

## 0. 四条原则

1. **默认不保存，保存必须晋升。** agent 生成的一切先是临时产物；只有经过明确 `promote` 才进入项目资产。
2. **图（对象）是核心，磁盘只是 artifact store。** 目录树适合放文件，不适合表达创作关系；关系写在 `status.json`。
3. **事实只从文件来，不从 agent 记忆来。** 人物设定、尺寸、池子大小这类"事实"，要么指到证据文件，要么当场重测，否则只能标"未验证"。
4. **agent 自主决定"怎么做到"，人决定"什么值得永久留下"。** 因此删除永远是受控操作，不是顺手动作。
5. **同一个操作只有一个实现，且优先用引擎自带的。** 常用操作先规范化成"操作目录"里的那一个实现（§6.1）；
   能用 ComfyUI 原生节点做完的，就不要在 Python 里再写一遍。**每期新写一个功能相似的脚本 = 项目负债。**

---

## 1. 对象模型

| 对象 | 是什么 | 例 |
|---|---|---|
| **Task** | 一个项目单位＝一个目录 | `odette_dancer_20261006/` |
| **Scene** | 一个画面/构图目标 | `s2` |
| **Image** | **人和 agent 对话的逻辑对象**，不是某个固定文件 | `s2-03` |
| **Artifact** | Image 在磁盘上的具体文件，带保留等级 | `art/s2/s2-03/final.png` |
| **Batch** | 一次生成 N 个候选，供人挑 | `b001` |
| **Decision** | 人做过的关键选择（采纳/淘汰/修改） | `pick b001 → s2-03c` |
| **Publication** | 一组 Image 作为一个对外交付单元 | `g1` |

**不要混三套状态**：Task 状态、Image 状态、Publication 状态互相独立。`image=final` 与 `task=producing` 同时成立是合法的。

---

## 2. 目录骨架

```text
task/{name}_{YYYYMMDD}/
  status.json          机读入口（唯一）
  README.md            自然语言入口（固定小节，见 §8）
  decisions.jsonl      压缩历史（追加式；不是 agent 的上下文）

  art/s{n}/s{n}-{nn}/  每个 Image 一个目录，目录名＝ID
    base.png           working
    work/              人可达的往返件（PS 稿、画布、mask、control）——按内容命名
    final.png          canonical
    workflow.json      canonical（与 final 绑定）

  references/          外部资料        evidence/  证据
  uploads_img/  uploads_wf/  uploads_job/{g1,g2}/     对外交付
  .runtime/{scratch,cache,logs,trash}                 agent 私有工作台
```

**项目空间 vs 运行空间**：`art/ references/ evidence/ uploads_*` ＝ 项目的一部分；`.runtime/` ＝ agent 为完成当前任务产生的计算垃圾，**不是项目历史**。

**⚠ 关键区分（不要照搬"什么都进 .runtime"）**：`.runtime/` 只放 **agent 私有**的东西（未选候选、对照表、`.runtime/logs`）。
**人在上面反复编辑的中间稿必须人可达**（双击能开 PS、能在资源管理器里看到），所以它们住 `art/.../work/`，**不是垃圾**。

---

## 3. 保留等级（Artifact 级）

| 等级 | 例子 | 默认保留 | 回收时机 |
|---|---|---|---|
| `ephemeral` | 未选候选、对照表、临时脚本、`.runtime/logs` | 否 | batch 解决后进 trash，TTL 到期真删 |
| `working` | `work/` 里的 PS 往返、mask、control | 否（随 Image 存活） | Image 被 dropped 且无人保留时 |
| `canonical` | `final.png`、绑定的 `workflow.json`、复现必需的 mask | **是** | 不自动回收 |
| `delivery` | `uploads_img/ uploads_wf/ uploads_job/` | **是** | 永不 |
| `external` | `references/ evidence/` | **是** | 永不 |

两套东西别混：**Image 的状态**（`candidate → accepted → final → delivered` / `dropped`）≠ **Artifact 的保留等级**。

---

## 4. 编号（硬规则）

- Scene `s{n}`；Image `s{n}-{nn}`；Batch 内候选 `s{n}-{nn}{a..z}`（**临时引用**，batch 结束即失效）。
- 全局引用 `{task}/s2-03`。**目录名＝ID**，文件名＝角色（`base/final/workflow`）。
- 1) agent 展示任何图，必须同时给 **ID ＋ 路径**；2) 编号分配后**永不复用**；3) 废弃 Image 保留编号、状态置 `dropped`；
  4) **不许从文件名猜 ID**；5) ID→路径以 `status.json` 为准；6) 同 ID 对应多文件必须**报歧义**，不许自己挑；
  7) 禁止用"这张/刚才第二张/右边那个"作为唯一引用。

---

## 5. `status.json` 的职责

**是**：当前状态的机器入口 —— 有哪些 Scene/Image、哪些被采纳、有哪些正式 Artifact、人物设定与证据、标准与口径、下一步。
**不是**：历史记录（那是 `decisions.jsonl`）；也不是可以任由 agent 手改的自由文本。

- 写入**只走操作原语**（§6），由工具同时改元数据与文件系统。
- `status.json` 与磁盘冲突时：**磁盘是真相，`status.json` 是可重建的索引**；`check_status` 负责报差。
- agent 开工**只读** `status.json` ＋ README 的「当前进度/当前问题/下一步」＋ 当前 Scene 的 canonical 资产。

字段骨架（项目可加，不可减）：

```jsonc
{ "schema": 3, "task": "…", "title": "…", "created": "YYYY-MM-DD",
  "status": "design|producing|ready|publishing|published|dropped", "phase": "…",
  "dirs": { "art": "art", "img": "uploads_img", "wf": "uploads_wf", "job": "uploads_job" },
  "characters": [ { "name": "…", "canon": "…", "evidence": ["evidence/x.png"],
                    "prompt_do": [], "prompt_dont": [] } ],
  "scenes": [ { "key": "s2", "name": "…", "images": [
      { "id": "s2-03", "state": "final", "group": "g1", "artifacts": [
          { "id": "a1", "path": "art/s2/s2-03/final.png", "retention": "canonical" } ] } ] } ],
  "groups": [ { "key": "g1", "job": "uploads_job/g1/job.json", "images": ["s2-03"],
                "status": "staged|publishing|published|aborted", "artwork_ids": [], "urls": [] } ],
  "standards": {}, "keep": [], "known_gaps": [], "note": "…", "updated_at": "…" }
```

---

## 6. 操作原语（agent 只用这些，不手改文件/JSON）

### 6.1 反重复原则（"操作目录"）

**同一个操作，全项目只允许一个实现。** 出现第二次"功能相似但文件不同"的脚本时，正确动作是**把参数抽出来**，
而不是复制一份改改。判断顺序：

1. **引擎原生能不能做？** —— 能就用它（例：局部重绘的"按 mask 贴回"＝ ComfyUI 的 `ImageCompositeMasked`，
   不是自己写 Python 合成；羽化＝`FeatherMask`）。
2. **skill 里已经有脚本？** —— 有就用/扩它，不改。
3. **只有参数不同？** —— 把参数写进 **recipe（数据）**，脚本不动。
4. **确实只有这个任务才需要，且以后还要复现？** —— 才允许留一个脚本，**并作为 canonical artifact 正式登记**；
   否则它属于 `ephemeral`，用完即回收。

**禁止**：`gen_xxx.py` / `make_xxx.py` / `upscale_xxx.py` 这类按任务复制出来的脚本进项目空间。
项目空间只留 `metadata ＋ recipe ＋ result ＋ decision`。

### 6.2 原语表

每个原语都是 **输入对象 → 产物 → 记一条 decision → 更新状态** 的原子动作。

| 原语 | 干什么 | 关键副作用 |
|---|---|---|
| `create_task(name, date)` | 建目录＋最小 status.json/README | — |
| `add_scene(task, name)` | 加一个 Scene | — |
| `add_image(scene, n)` | 分配 ID、建 `art/s{n}/s{n}-{nn}/` | ID 永不复用 |
| `add_batch(image, n)` | 生成 N 个候选到 `.runtime/scratch/b{nnn}/` | 记 `candidate_batch` |
| `show(id)` | 给人看：**必须带 ID＋路径** | — |
| `pick(batch, id, by)` | 人挑定 | 记 `pick`；未选 → trash |
| `promote(artifact, level)` | 提升保留等级 | 只有 §7 的四种理由 |
| `discard(id)` | 淘汰 | 记 `discard`；进 trash |
| `set_state(id, state)` | 改 Image 状态 | 状态机校验 |
| `bind_workflow(id, path)` | 把最终 workflow 绑到 final | 一起 canonical |
| `add_group(images)` | 组一次对外交付 | 分配 `g{n}` |
| `publish(group)` | 交付 | 记 `publication`；成为 GC root |
| `gc(task, dry_run)` | 可达性回收 | 见 §7 |
| `check_status(task)` | 元数据 ↔ 磁盘对账 | 报差，不改 |

---

## 7. 晋升与 GC

**只有四种理由可以 promote**：① 人采纳；② 成为后续正式工作的输入；③ 人明确要求保留；④ 复现最终结果确实需要（`reproducibility_required`）。

**GC 是可达性判断，不是"看着像垃圾"**：
- **Roots**：canonical 资产、delivery 资产、已发布记录、references、evidence、`keep` 点名。
- 沿 `derived_from` ＋ `reproducibility_required` 建闭包；**闭包外的 ephemeral/working 才进回收**。
- `delete` 实际是 `move → .runtime/trash/{时间戳}_…`，TTL 到期才真删 ⇒ 犯错便宜。
- 第一次跑或删敏感内容**必须 `--dry-run` 列出"将删除/将保留"**，人点头才执行。
- **TTL 只是兜底**，不是主要机制。
- `keep` 是**少量明确例外**，不是"整个目录都留下"的保险箱。

**⚠ 别做过头**：不要"batch 一解决就立刻删未选结果"。跨批次回头比是常态（换 denoise 重挑、隔天反悔）；未选结果进 trash ＋ TTL 就够。

---

## 8. README 的固定小节（自然语言入口）

```markdown
# 当前任务
## 目标        ## 当前进度     ## 当前问题     ## 下一步
## 已确认事实（带证据路径）      ## 已知坑      ## 设计备注
```
agent 每轮默认只读 **status.json ＋「当前进度/当前问题/下一步」**；要查历史才去读 `decisions.jsonl` 或 `.runtime/logs`（on-demand，不是 always-on）。

---

## 9. 一次任务的标准顺序

```text
建 Task → 落盘 characters/evidence → 建 Scene → 生成 Batch(.runtime/scratch)
   → 人比较 → 记 Decision → promote 被选 → 修改/重绘(work/)
   → Final → Deliver → Upload/Publish → GC
```

关键在中间那四步：**生成 → 临时 → 选择 → 正式**。而不是"生成 → 全留 → 以后再整理"。

---

## 10. 上下文纪律（与磁盘整洁是同一个问题）

文件越多 ⇒ agent 读到的有效信息比例越低 ⇒ 越容易误读/幻觉 ⇒ 为保险又存更多 ⇒ 文件更多。
所以：**项目历史应该是经过编辑的知识（事实/决策/结论/依赖），不是文件系统的尸体。**

- 不许"全量读取 task 目录"或"遍历历史候选"。
- 不许用模型记忆补全事实；没有证据的进 `known_gaps`。
- 数字要么带**对照**，要么标"未验证"（例：断言"mask 外没动"必须给匹配对照，而不是一个没有区分力的 `max diff`）。
