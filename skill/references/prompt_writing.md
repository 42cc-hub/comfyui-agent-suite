# Prompt writing — 结构、否定、换装

> 要**可抄的块**（风格前缀 / 负面模板 / 骨架模板 / 词表）：`references/prompt_cheatsheet.md`。
> 本文件只放**规则与判据**，以及支撑它们的来源级别。

## 0. 来源级别：哪条结论能当规则用

| 标 | 来源 | 能证明什么 |
|---|---|---|
| `[measured]` | 本机自己的 A/B 出图 | 因果：X 导致 Y。**条数少，是单次观察** |
| `[corpus]` | `<你的图片目录>` 里 536 条已发布的提示词（2026-09-28/29 抽取），原始数据 `<你的 ComfyUI 项目>\experiments\prompt_organize\{wf,analysis}` | **只证明惯例**：词汇、形状、大家实际在发什么。**频次 ≠ 效果**——其中很大一部分是同一条模板被复制改的（最高频那条负面串出现 136 次），高频词可能只是传统 |
| `[plugin]` | 本地 `comfyui_PromptManage`（`README.md` §「LLM 提示词生成规则」、`web/llm-templates.json`） | 该插件的家规，按 IL 系 tag 提示词写的 |

**`[measured]` 与 `[corpus]` 冲突时以 `[measured]` 为准**；语料是词库和形状参照，不覆盖跑过的实验。
`[measured]` 条目若没写次数，一律按单次观察读，**不要据此推通用规律**。

语料形状：544 条记录（232 × `waiIllustriousSDXL_v160`、90 × v170、124 Anima 系、105 条只放大的图、其余 novaAnimeXL），
80% 是绳缚内容；IL 中位 **44 tag / 666 字符 / 4 个空行块（9 行）**，Anima 59 tag / 1052 字符。

---

## 1. 结构硬规则：一行一组，段落之间空行

1. **空行（`\n\n`）分隔区块**（场景 / 人物 / 风格）。用途是**给人手改**：改一条只定位一个区块。
   `[measured]` 六组压成一行的版本被当场要求返工。
2. **单换行（`\n`）分隔要素组**：同一个区块里一个语义要素占一行（姿势 / 缚法 / 表情 / 服装各一行）。

`[corpus]` 第 2 条是实际写法：**85% 在区块内部用单换行**，中位 9 行 / 4 个空行块，只有 4% 一条到底。
正确粒度是**行**：全堆一行难改，每个要素都空行又太散。

### 分组顺序 `[corpus]`（类别平均相对位置，0 = 首组，1 = 末组）

| 位置 | 内容 | 依据 |
|---|---|---|
| 0.06–0.17 | **风格 / 质量块**（只出现一次） | `score` 0.06、`style` 0.13、`quality` 0.15 |
| 0.35–0.45 | 场景 / 镜头（`indoor, wooden floor, full body, front view`） | `lighting` 0.35、`environment` 0.40、`camera` 0.45 |
| 0.49–0.61 | 主体 + 姿势 + 缚法 + 表情 | `bondage` 0.49、`pose_action` 0.60、`expression` 0.61 |
| 0.71–0.75 | **角色身份 + 外观 + 服装** | `appearance` 0.75、`clothing` 0.73 |

**角色身份块放最后三分之一**：284 条带 `名字 \(系列\)` 的记录，身份块平均相对行位置 **0.71**（中位 0.75），
**69% 落在后三分之一**，只有 3 条放第一行。

- 放最后是社区惯例，好处是**改角色不动场景**（corpus 里 `xtll_*`、`mh*` 系列就这么批量产）。
- 放前面是 LoRA 训练的常见排布，`[measured]` 换装实验里放前面也有效。
- **硬约束只有一条**：角色触发词 + 外观的**完整性**比位置重要。位置随你，同一系列内保持一致。

真实形状（`[corpus]` 逐字，`ayaka_rope (1)`，见 `prompt_cheatsheet.md` §3 T2）：姿势 / 缚法 / 道具 / 表情
各占一行，角色块在最后两行，风格块用 `;` 隔开放末尾。风格块也可放开头，**但只能出现一次**。

---

## 2. 风格不是质量词给出来的 `[measured]`

`masterpiece, best quality, absurdres` **本身不构成风格**——风格来自跟在它们后面的那一串
（`[plugin]` 说"开头写质量词"，`[measured]` 说质量词不是风格，两句都对）。
语料量化：**304 / 536（57%）**开头 18 个 tag 逐字相同；307 条风格块在第一行、53 条在最后一组、
3 条前后都放；173 条完全没有 `trending on artstation`（多为 Anima / pony 系）。

三套可直接抄的 IL 块（通用 / 厚涂 / 电影光长版）与 Anima 触发句都在 `prompt_cheatsheet.md` §1，
**不要在这里再抄一遍**。要用的规则是：

- **IL 的 18-tag 块不是 Anima 的块，跨底模复用 = 白烧一批图。** `[corpus]`：Anima 记录里 91% 用自然语言句子、
  93% 带 `@画师`、63% 用权重（中位 1.45）；而**我们自己那批 Anima 记录 33% 直接贴了 IL 的 18-tag 块**、88% 没写句子。
- `score_9, score_8_up, score_7_up` 只在 pony 底模上用；IL 上写 `score_*` 是噪声（语料 9%）。

### 想要"无描边半写实"的四个杠杆（缺一个就退回平涂）`[measured]`

1. **换 UNET 到 `anima-aesthetic-v1.1.safetensors`**（不是 `anima-base-v1.0`）。本机**唯一**真正管用的风格杠杆；
   base 版怎么写词都出平涂 + 描边。
2. 正向去描边：`(semi-realistic:1.4), (no lineart:1.4), (lineless:1.3), (soft shading:1.3), realistic skin texture,
   subsurface scattering, airbrushed, painterly rendering, soft rim light, cinematic lighting, depth of field`
3. 负向压平涂：`(lineart:1.4), (outlines:1.3), (black outlines:1.3), (cel shading:1.3), (flat color:1.3),
   anime coloring, (thick lines:1.2), sticker, vector art, comic`
4. **40 步 + CFG 5.5**（低 CFG + 高步数才把它推离卡通；Anima 默认 30/8 会拉回平涂）。

- 单次观察：同题面下 aesthetic 版出的是无描边半写实（布料褶皱、皮肤次表面、麻绳扭转、背景景深），base 版是平涂描边。
- 本地**没有**伪写实画风 LoRA（`gpt-image-2_anima-base1_v*`、`anima_quality`、`anima-highres-aesthetic-boost`），
  这条路是"finetune + 题面"；装上那些 LoRA 还能再稳一档。
- **近景/广角里被推向镜头的部位会称霸画面**：`(soles towards viewer:1.2)` 让脚掌变成主体 ⇒ 该类描述要降权或不写。
- 脚本：`experiments/repro_146351817/run_realistic.py`（`--unet/--steps/--cfg`）。

---

## 3. 负面是第二块画布，不是缺陷清单

`[plugin]` 规则 9/10 只说"负面列缺陷、一到两行"。`[corpus]` 显示负面承担**五种**功能，中位 234 字符 / 20 tag（Anima 系 737 字符）：

| 用途 | 占比 | 例（逐字） |
|---|---|---|
| 固定缺陷块 | 几乎每条 | 见 cheatsheet 模板 |
| **材质/风格纠正** | 55% | `brush, comic`（水彩 LoRA）；`flat colors, cel shaded, hard edges, bold outlines, simple shading, 2d anime screencap`（锁厚涂） |
| 角色服装残留清除 | 14% | `dress, hat`；`detached sleeves, tassel, high heels` |
| 裸露 / 情色压制 | 11% | `nsfw, nude, open mouth` |
| **多人压制** | 10% | `people, person, girl`；`double head, 2girl` |
| **姿势/器具压制** | 6% | `hanging rope, suspension`（要落地绳）；`ribbon, satin ribbon`（绳别变缎带） |

**判据：想要什么写正向，"这个词会带来的不想要的读法"写负向。** `red rope` 的默认读法之一是悬吊与缎带，
所以 `hanging rope, suspension, ribbon, satin ribbon` 是最常见的搭配修正——比正向写 `not hanging` 有效（§6）。

- Anima 形体纠正系把 `ungagged` 排在第一位是**故意的**：压的是"口球 LoRA 在没写口球时也画口球"的过冲。
- 经典 danbooru 系负面在 corpus 里只占 23% ⇒ **负面词跟底模的语料走**：IL 吃 aesthetic 系、pony 吃 `score_1..3, source_pony, anthro, furry`、Anima 吃形体/材质系。
- **负面要瘦身**：单次观察——60+ 缺陷词 + `(shoes:1.45)` 那一版鞋子纹丝不动、整张图海报化（绿/红偏色）；
  20 个词（`shoes, high heel`）配 CFG 8 / 20 步，鞋直接消失。

---

## 3b. 简洁：长度本身就是成本 —— 一个意图一处说

> 条件按 token 竞争注意力：词越多，每个词分到的相对权重越低 ⇒ 你真正想要的 tag 被稀释，
> "什么都写了" = "什么都没到位"。写得长不等于更保险，只等于更平均。

**必须删的两种写法**

1. **正负双写同一件事**：正向 `no shoes` + 负向 `shoes, boots` + 一句 `not wearing shoes` —— 一个意图说三遍，
   正向里的否定模型还读不懂（§6）。**想排除 ⇒ 只在负向写最准的那个词。**
2. **正向同义堆叠**：`long hair, very long hair, flowing long hair` —— 第 2–4 个不提供新信息，只摊薄其余要素。
   **留最准的那一个**（先查它指什么，§10）。

**操作规则**

- 每个内容组只留承重的词；空行分组让"整组删掉"不伤别的组；
- 负向 ~30 tag 是**上限不是目标**；通用缺陷块别抄两遍；
- 画面不对时第一反应是**删词**，不是加词或加权；
- 长度自查：正向 9 行上下；超了先问"删哪一个"，不是"还缺什么"；
- 判据：**加一个词如果没让画面可辨地改变，它是噪声，删掉。**

**三个维度互不替代**：§1 管结构（一行一组）· 本节管总量（多不多 / 有没有重复）·
tag review 管对不对（互斥、残留、场景冲突，`references/tag_review.md`）。

---

## 4. 绳索 / 束缚词表 `[corpus]`（df = 出现该 tag 的提示词数）

| tag | df | 说明 |
|---|---|---|
| `shibari` | 266 | 主词 |
| `red rope` | 253 | 材质 + 颜色，与 `shibari` 同写 |
| `ball gag` | 122 | 口球主词 |
| `crotch rope` | 107 | 股绳 |
| `mouth gag` | 107 | 通用口部束缚 |
| `arms behind back` | 93 | **规范 tag，优先用它** |
| `bound legs` / `bound ankles` / `bound knees` / `bound thighs` / `bound arms` / `bound wrists` / `bound feet` | 79 / 23 / 16 / 15 / 35 / 6 / 6 | **按部位逐个绑**，别只写"绑住腿" |
| `breast bondage` / `breast rope` | 78 / 27 | 胸绳 |
| `bondage` | 76 | 泛词 |
| `arms bondaged behind back` | 63 | 非规范但大量使用（见下） |
| `legs bondaged` | 47 | 同上 |
| `blindfold` / `black blindfold` | 32 / 9 | |
| `suspension` | 29 | 单独写容易变倒吊。**悬挂的正确搭配是 `suspension` + `hang up`**：只写"吊"的散文常落成"坐在床上/被托着"，`hang up` 才能读出"身体离地" |
| `harness ball gag` / `ring gag` / `gag harness` / `cleave gag` / `wiffle gag` | 28 / 22 / 12 / 5 / 2 | 口球细分 |
| `shibari over clothes` | 22 | 隔衣缚（避免自动脱衣） |
| `skindentation` | 15 | 绳勒进皮肤（肉感全靠它） |
| `bitchsuit` / `straitjacket` / `hogtie` / `spreader bar` | 15 / 13 / 14 / 5 | 拘束衣类 |
| `M-legs` / `W-legs` / `wariza` / `folded legs` | 16 / 13 / 12 / 6 | 腿部几何 |
| `nipple clamps` / `vibrator cord` / `egg vibrator bulge` / `leash` | 11 / 13 / 11 / 9 | 附件 |

**规范 vs 非规范**：`bondaged`、`arms bondaged behind back`、`legs bondaged` 都不是 danbooru tag，
但 corpus 里合计 100+ 次并产出可用图（`bondaged` 含 `bondage` 子词，编码器能接上）。规则：

- **默认用规范 tag**（`arms behind back`, `bound arms`）——只有规范写法带训练先验；
- 规范写法渲染失败时**加**非规范变体（两者同写不冲突）；
- 拼错到脱离原词就是噪声（`ctorch rope`、`amrs bondaged behind back`、`tembling`）——**优先规范拼写**。

### 绳的拓扑句子（14% 的提示词用整句，且高度复用）

```
She is tied up by red ropes in a kikkou shibari style.                        (×10)
Her legs are tied together by large amounts of red ropes.                     (×11)
Her upper body is tied in shibari style by large amounts of red ropes,
  and her arms are tied behind her back.                                      (×7)
Her hands are tied behind her back. She is wearing a harness ball gag,
  and has a blush and tears in eyes.                                          (×15)
the gag's straps are black, and this strap goes up over the nose,
  meets at the top, and then goes around the back of the head
  to form a harness gag.                                                      (×6)
A ball gag with harness and leather straps is secured in her mouth.
Her other leg is suspended, raised high above her head.                       (×4)
Her dress is rolled up by the crotch rope, revealing her white panties.       (×2)
```

| 要描述的东西 | 写法 | 例 |
|---|---|---|
| **拓扑**：什么连什么、绑在哪个部位、绕几圈 | **句子更清楚** | `Her upper body is tied in shibari style by red ropes, and her arms are tied behind her back.` |
| **属性**：颜色、材质、款式、状态 | tag 短语 | `red rope, shibari, crotch rope, skindentation` |
| **像素级几何**：布抬几厘米、哪条边在上 | **别写，去做遮罩** | `the centre panel pressed flat, the outer corners above the hip line` —— 8 张图全无视 |

`[measured]` "短句胜过长几何描述"的失败案例属第三行，corpus 里生效的绳句属第一行。
**判据：句子里出现"相对量/像素量"就停手**，转 `libs/blender-controlnet/references/blender_control_pipeline.md` 或遮罩。

---

## 5. 多主体：标签 + `;` 分块

| 模式 | 条数 `[corpus]` | 例 |
|---|---|---|
| `front girl` / `other girl` | 29 | `front girl, lying,` … `other girl, sitting beside,` |
| `left girl` / `right girl` / `meddle girl` | 18 | 三人及以上并排 |
| `girl below` / `girl above` / `1girl over` | 6 | **上下叠压构图必须写**，否则角色互换 |
| `sub girl` / `dominatied girl` / `dominator` | 5 | 按关系角色命名 |
| `first girl:` / `girl A:` + `{}` 包块 | 3 / 1 | 显式块结构 |
| `2girl` / `2girls` | 60 / ~10 | 先写数量 |
| `1boy` / `1man` | 38 | 含男女同框 |

**规则（`[measured]` + `[corpus]` 一致）**

1. 先写数量；
2. 每个角色一块，标签写块首（屏幕位置或关系角色）；
3. 每个角色**自己**的姿势 + 状态必须写全——只写全局动作会让两人做同一件事；
4. `;` 或空行分块，别让两套姿势词混行；
5. **`left/right/below/above` 的绑定是防"角色互换"最强的单点修正**；
6. 负面钉人数：单人 `multiple girls`，要三人 `2girl`。

逐字样例（`belle_vvn_handup (1)` 的完整块）见 `prompt_cheatsheet.md` §3 T3。

**Anima 的多主体**：同一套块结构但写句子，`;` 独立成行（场景句 / 全局动作句 / `left <名字>:` / `right <名字>:` / 风格尾）。
`[measured]` 同底模同参数只改文本：块结构 → 角色、身高关系、动作全对；tag 汤 + 一句场景（4 个种子）→
**四张全错**（两人都躺地上、缚法被读成胸带、服装不分、谁是谁逐种子变化）。

---

## 6. 永不在正向里写否定 `[measured]`

文本编码器不会把否定词绑到后面的名词：`no apron` 贡献的是 *apron*，`nothing at all in the middle` 贡献 *middle*，
`the rest still hidden` 最多给一个 *hidden* 的风格暗示。`no / without / nothing / except` 都没有可靠算子。

**规则**：正向写**肯定目标**（`plain white summer dress`），要排除的全进负面：

```
NEG: navy blue maid dress, maid outfit, apron, frilled apron, black skirt,
     low quality, worst quality, lowres, blurry, jpeg artifacts, bad anatomy, ...
```

定量约束同理："裙子吹起但只露外侧一点点"不能写 `only the very edges, nothing in the middle`——
模型读到名词会把中间也画出来。要么正向描述"能看见的部分"，要么用遮罩 / ControlNet 控制。

`[corpus]` 交叉验证（399 条纯 tag 提示词）：含 `no <名词>` 的只有 **2%**、含 `only` 的 2%
（含 `no` 的 7% 集中在中文改图指令句里）⇒ 惯例一致，但**因果仍来自 `[measured]`**。
`[plugin]` 没有这条规则，最接近的是 `llm-templates.json` 规则 9 的弱表述。

---

## 7. 换装 / 换角色要显式替换从句 `[measured]`

角色 LoRA 自带服装且权重高，只点名新衣服**会输给 LoRA**。两处必须同时改：

```
POS: ... she wears a plain white summer dress instead of a maid outfit: white dress, ...
NEG: navy blue maid dress, maid outfit, apron, frilled apron, black skirt, ...
```

DeepSeek whale-girl LoRA（训练数据是海军蓝女仆装）上的实测：

| 正向 | 白裙出图率 |
|---|---|
| 只写 `white dress`，无替换从句、无负面 | 0 / 4 |
| `plain white summer dress instead of a maid outfit, no apron` + 服装进负面 | **4 / 4** |
| 有替换从句但负面漏掉原服装 | 部分——袖子/围裙残留 |

**换装时把原服装 tag 从角色块里删掉**（`navy blue maid dress`、`deepseek_maid_outfit`、`white frilled apron`），
留着就是和替换从句对抗。**身份类 tag 全保留**（发色、瞳色、鳍耳、尾巴、头饰形状）。

---

## 8. 角色块：位置、LoRA 触发词、`<lora:>` 写法

`[corpus]` 三种写法都在：LoRA 触发词打头（`sprkamiaya, light blue hair, ...`）、
规范 `名字 \(系列\)`（`escoffier \(genshin impact\), ...`）、散文式（`Alice Thymefield from Zenless Zone Zero, ...`）。

- **`名字 \(系列\)` 是主导写法**（284 条；`\\(` 转义出现在 51% 的记录、60% 带系列名）。
  特殊字符必须转义，否则括号被当权重语法。
- **LoRA 的 trainedWords 照抄，不要自己编**——包括**整句**触发词和带权重的艺术家触发词
  `@[himitsu (hi mi tsu 2):0.8]`、`@[wlop::0.6]`（各 54 条）。`@name` 式风格 tag 在 19% 的记录里，
  常见组合：`@houkisei, @rokuwata tomoe, @atdan, @[himitsu (hi mi tsu 2):0.8], @wlop, @m4me, @n4g1style`。
- `<lora:name:weight>` 写进文本的只有 15 条——本管线走 `LoraLoader` 节点，**不要内联**。
- 27 条用 `__char__` / `${char}` 占位符批量套模板：**这正是把角色块放最后的好处**，换人只改一块。

---

## 9. 权重：可以用，幅度跟底模走；本管线未做 A/B

`[corpus]` 24% 的 IL、51% 的 Anima 提示词用 `(tag:w)`：

| 底模族 | 用权重的记录 | 加权 tag 数 | 中位 w | p90 w | 最大 w | w ≥ 1.5 |
|---|---|---|---|---|---|---|
| IL / SDXL | 62 | 136 | **1.20** | 1.30 | 2.50 | **5%** |
| Anima | 69 | 327 | **1.50** | 2.00 | 8.00 | **51%** |

**IL 上权重是微调**（1.1–1.3，极少到 2.5）；**Anima 上权重是主控制手段**（一半 ≥1.5，出现 `(shibari:4.0)`）。
在 IL 上写 `:4.0` 没有 corpus 支持，只会把画面拉成海报化。`@[name::0.8]`（双冒号）是触发词内嵌权重，
与 `LoraLoader` 的 `strength` 等价但写在文本里。

⚠ `[pipeline]` 记录显示 **`(keyword:1.2)` 在本管线的正向效果尚未验证成功**。规则：**先减词，再调权重**；
要用就用 1.1–1.3，一次只动一个，否则分不清是权重还是词长在起作用。

### 与语料原题面的实测对照（5 例同图复现）

脚本 `experiments/compare_prompts.py` → `experiments/prompt_style_compare.md`：

| 特征（均值） | 原题面 | 本次我写的 |
|---|---|---|
| 字符数 | 1044 | 1635 |
| 行 / 块 | 8 / 5 | 8 / 5（一致 ⇒ 行级 + 分块结构没问题） |
| 带权重 token | **3** | **18** |
| 最高权重 | 1.5–3.0 | 1.4–1.8 |
| `@画师` | **6** | **0** |
| 负向字符 / 行 | 827 / 1 | 659 / 4 |

四条要改的：

1. **权重的"数量"要减，"幅度"不要加。** 原题面平均 3 个权重而我写了 18 个；
   但它们把峰值拉到 2.0–3.0（甚至 `(shibari:4.0)`）**不是目标**——权重加大容易崩坏，除非确需不要超过 ~1.5。
   要让某样东西"必须出现"，正确手段是**把词命名准**（原生 tag / 独占一行 / 位置词）+ **负向压掉它的替代品**。
2. **画风不是质量词能替代的。** 原题面 6/6 都带同一组六人 `@画师` 块，LoRA 栈里**总**有画风 LoRA
   （`qiandai_yiyu_style_v2` / `NAGI` / `anima_m4me` / `BlueArchiveStyleB1`，权重 0.2–0.8）。
   `very aesthetic, absurdres, year 2025` 只提质量、**提不出画风**；复现语料图之前先查有没有画风 LoRA，
   没有就如实说"风格这一层复现不了"，不要用质量词假装。
3. **正向头部保留一个"评级 token"**（`safe` / `questionable` / `explicit` / `nsfw`，pony 系用 `score_9, score_8_up, score_7_up`）——
   它是显式的内容尺度旋钮。
4. **负向要贴语料观感就直接用它们那条审美模板**：约 800 字符的审美/质量串**挤在 1 行**（互相抄袭，同一条出现 136 次），
   内容类另起一行。抄的时候注意 `BDF525D9` 那类原题面头部混进了 PromptManage 的 UI 串（`? (false), ?? (both), append, yes, …`），
   语料 26% 有这种泄漏。

---

## 10. 写之前先查这个词到底指什么

日本网络 / 服装 / 同人词汇，写前先查（`dic.pixiv.net/a/<tag>`、萌娘百科）。
角色特征 tag 从哪来、资料站描述怎么翻成 tag：`references/character_features.md`。
意译外来语和黑话经常是错的，一个错读烧一整批——某主题词被当字面道具画了 12 张，实际含义是服装轮廓。

`[corpus]` 两个活例：

- `pillory`（枷）——本写成"厚木板 + 圆孔"，出来是奇怪的板子；换原生 tag `pillory` 后器具直接对了
  （见 `experiments/ganyu_cangue/README.md`）。
- `locker` 1789 帖、`locker_room` 706 帖可用；`metal_locker` / `storage_locker` / `school_locker`（日式鞋柜）/
  `footlocker` **都不是 tag**，写了等于没写。

---

## 11. 语料里的坏习惯（别学）

- **插件 UI 字符串泄漏**：`Select the Wildcard to add to the text`、`Select the LoRA to add to the text`、
  `reproduce`、`否 (false), 两者 (both), append, yes` —— PromptManage 的按钮文字被写进条件。
  `[corpus]` 29 条（26% 的"他人下载"组）含这类字符串；我们自己的记录里也见过 `否 (false), 两者 (both)`。
- **插件 token JSON 泄漏**：一条记录的 `CLIPTextEncode.text` 里塞了 PromptManage 整个 token 数组
  （`[{"id":"token_1_1779355449532","text":"medieval dungeon",...`），单条 17,965 字符——占满上下文窗口。
  **提交前看一眼文本首尾。**
- `extreme details,,,` 类重复逗号 41%（拼接痕迹，本身无害；我们的提示词由脚本生成，不该有）；尾逗号 73%（无影响）。
- `2girl` 与 `2girls`、`(series)` 与 `\(series\)` 混用——各自内部统一，别在一条里混。
- **多人图里 `1girl` + `2girl` 并存就是自相矛盾**（单人图里没坏处）。

## 12. 提交前 checklist

**第 0 项：tag review**——逐 tag 对照"场景清单"，见 `references/tag_review.md`。
提示词来路不是为当前场景从头推的（LoRA 触发词 / 抄来的工作流 / 上一版 / 别的 task），
或场景任一要素变了（服装、姿势、道具、地点、视角、画幅）⇒ **不许跳**；只通读一遍不算做过。

- [ ] 区块空行、要素各占一行（9 行上下）？
- [ ] 风格块**只出现一次**，且质量词后面跟的是**真风格**？
- [ ] 正向里没有 `no / without / nothing / except / only`？
- [ ] 没有正负双写同一意图？没有同义堆叠？（§3b）
- [ ] 要排除的（原服装、不想要的读法、多余的人、材质）在负面且 ≤ 30 tag？
- [ ] 场景语汇按底模走：IL 用 tag、Anima 用句子？
- [ ] 换装？→ 正向 `instead of X` + 负面含 X + 角色块里删掉 X？
- [ ] 绳索写明**部位**？
- [ ] 拓扑类（谁连谁）用句子、像素几何类去做遮罩/ControlNet？
- [ ] 多人？→ 数量 + 每角色块 + 位置/关系标签 + 各自状态 + 负面钉人数？
- [ ] 角色块位置在本系列内一致？主题词汇查过 tag 库？
- [ ] 权重数量少、幅度 ≤ ~1.5、一次只动一个？（先减词，再调权重）
- [ ] 重绘？`denoise` **按这张画布重新标定**（低 denoise 覆盖像素少，"什么都没发生"通常是尺度不对，
      见 `references/local_repaint_bigcanvas.md`）。

## 附录：这些数字怎么复现

语料来自 `<你的图片目录>`（**只读**，只拷副本）。脚本在
`<你的 ComfyUI 项目>\experiments\prompt_organize\`：

```powershell
cd <你的 ComfyUI 项目>\experiments\prompt_organize
python organize_workflows.py    # 扫描 11,154 张图，带工作流的拷进 ref/（1,362 → 去重 544）
python extract_wf.py            # PNG 内嵌 workflow -> wf/<name>.json（保留 \n\n）
python analyze_corpus.py        # -> analysis/corpus_stats.md（分组、词频、位置、负面模板）
python analyze_extra.py         # -> analysis/extra_stats.md（身份块位置、权重幅度、负面用途）
python sample_prompts.py        # -> analysis/sample_prompts.txt（52 条最远点采样）
```

- **分析结构前先确认提取器没吃掉结构**：`extract_wf.py` 的 `clean_text()` 保留换行；此前的 `norm_text()`
  把 `\n\n` 压成空格，得出过"所有提示词都只有 1 组"的假结论。
- `wf/*.json` 字段：`image, src, size, positive, negative, checkpoint, loras, sampler, prompt_graph`；
  `negative` 原样文本也在 `prompt_graph` 里可回溯。快照 544 条（536 正向 / 528 负面），采集于 2026-09-28/29。

---

## 两个"模型不认识的新角色"同框、且不许反左右（标准解）

1. **先出一张不挂任何角色 LoRA 的"版式底图"**，把布局与关系做对；方向一律写**画面坐标系**
   （`the girl on the left side of the image`、`leans sideways to her left`），负面点名
   `mirrored composition / swapped positions`；
2. `check_parts.py --person-masks` 拿**每人一块、互不重叠**的掩版（见 `references/local_repaint_bigcanvas.md` 的「逐人掩版」节）；
3. **每次只挂一个人的 LoRA**（1.0/1.0 + sidecar 完整触发词表），`denoise 0.65~0.75`，串行两次；
4. **每人的重绘提示词里都写一遍接触关系**（"the other girl leans her head on her shoulder"），否则边界处姿态会被改坏。

> 中文角色名在 LoRA 文件名里常搜不到，但写在 sidecar 的 `model_name` 里（遍历匹配）；最全的触发词在
> `civitai.images[].meta.prompt`。

**模型与代码位置（只给路径，不复制）**

| 东西 | 路径 |
|---|---|
| 套件根 | `<ANIME_DET_SUITE>`（`--suite` 或 `ANIME_DET_SUITE` 覆盖） |
| 产线入口 | `<套件>\scripts\locate.py` |
| 推理环境 | `python`（`--python` 可换） |
| 权重 | `<套件>\runs\<name>\weights\best.pt`；默认权重在 `locate.py` 顶部 `DEF_*` |
| M1 骨架 | **首选** `models/vitpose_ft_v6_labelbox_honly`（微调 ViTPose，top-down 吃 M2a 的 person 框）；缺目录或 `--no-vitpose` 时回落 `runs/m1_pose_coco17` + `runs/m1_pose17_1129`（取检出人多的）→ dwpose |
| M1 实测 | 固定 15 张报告集（279 关节）：ViTPose-ft 覆盖 88% / median 23.1px / p90 97.8；回落那套 71% / 66.0 / 341.6。配对聚类 bootstrap 与 dwpose 比 median 无差别（−2.0px）、mean 显著 −31.8px [−53.3,−11.0] |
| M2a 框 | 主检 `runs/m2a_det_fix_v2`（person/hand/foot/head）；手第二遍 `runs/m2a_det_rot_semi`（`--hand-imgsz` 默认 1536，**整类替换**主检的手框，`--hand-imgsz 1280` 回旧行为）；专用脚 `runs/m2a_foot_v1`（与四类取并集）；`runs/m2a_hand_v3` 只建议诊断用（并集会掉精度） |
| M2b 面部 | `runs/m2b_face_v3_3983`（输入是 head 裁剪的 256×256） |
| M2b-gag | `runs/m2b_gag_v2` + `runs/m2b_gag_v4`（并集；口塞/眼罩） |
| M3 指趾判定 | 现役 `runs/m3_cls_aug_v1` + `runs/m3_review_v1`（**val 有训练集泄漏**，见套件 R43）；推荐 `runs/m3_clean_v1_s2026{1,2,3}010` |
| M4 绳子 | **不在本入口**：rope 模块产出 `data\masks\rope\<name>_{mask,conf}.png`，`locate.py` 只按约定路径引用 |

**已知限制（"没报错"≠"没问题"）**

1. M3 诚实指标（按源图 5 折折外）：召回 **0.74（单模型）/ 0.79（双权重并集）**，ok 误报 14%~19%；**手（0.61）明显差于脚（0.85）**。
2. 脚召回靠三级兜底（4 类 ∪ 专用脚 ∪ 踝点外推）；契约 JSON 的 `boxes` **不含**兜底补的部位（见 `boxes_all`）。
3. `--imgsz 1600` 明显好于 1280（手覆盖 4/8→5/8，08 那种"趾多"从只被收紧旗标报出变成进主判定），
   代价约 1.5~2× 时间；本机 1536×1024 的图 1280 下约 17 s/张、加 `--person-masks` 再 +10 s。
4. 上述验收数字都来自**同一批 8 张图**，小样本；换风格/题材自己再复核几张。
5. `rope` 只表示"rope 模块有没有留下掩版文件"（`present|missing`），**不代表图里没有绳**。
6. 更细的接口与实验记录在 `<套件>\docs\`：`integration.md`（接口与降级）、`runs.md`（R1~R49）、`handover_20261012.md`。

**契约 JSON 给下游用的字段（都是加法式的）**

- `points[].person_id` / `boxes[].person_id`：把点/框挂到具体角色上（多角色逐人处理必需）。
- `n_persons`（M2a 的 person 框数）与 `n_skeletons`（M1 链数）分开报；旧的 `n_people` 仍是骨架数（兼容）。
- `boxes_all`：`boxes` **加上**兜底补的部位，每个框带 `origin: det | fallback`。做掩版请用 `boxes_all`，否则漏掉兜底找回来的脚。
- `persons_masks` / `person_boxes_dropped` / `person_instances_merged`：见 `references/local_repaint_bigcanvas.md` 的「逐人掩版」节。

**更新模型**：新权重放 `<套件>\runs\<name>\weights\best.pt`，改 `locate.py` 的 `DEF_*`，
或调用时用 `--m3-weights/--hand-weights`；`--recipe rec` 是"实测更好但还没转正"的那一套。
