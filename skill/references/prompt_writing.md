# Prompt writing — structure, negation, costume swaps, and what a working prompt looks like

> **只想快速写一条提示词？用 `references/prompt_cheatsheet.md`** —— 那里是可抄的前缀/负面/骨架模板
> 与词表。本文件是**规则与证据**：每条结论的来源、语料统计、A/B 实测。

## Provenance of every claim in this file

Three sources, and they do **not** have equal authority:

| tag | source | what it can prove |
|---|---|---|
| `[measured]` | our own A/B renders on this host — each row cost a batch of images | causality: X caused Y |
| `[corpus]` | **536 published prompts** harvested from `<你的图片目录>` (workflows extracted 2026-09-28/29). Raw data: `<你的 ComfyUI 项目>\experiments\prompt_organize\{wf,analysis}` | convention only: vocabulary, shape, what people actually ship. **Frequency is not efficacy** — a large share of these prompts is one shared template copied and edited (e.g. the top negative string appears 136×), so a frequent word may be tradition rather than a working trick |
| `[plugin]` | local `comfyui_PromptManage` — `README.md` §「LLM 提示词生成规则」, `web/llm-templates.json`, `prompt_example/LLM_analyze_prompt.md` | the house rule set, written for IL-family tag prompts |

**When `[measured]` and `[corpus]` disagree, `[measured]` wins.** The corpus is a vocabulary
mine and a sanity check on shape; it never overrides a render we actually ran.

Corpus shape in one line: 544 records (232 × `waiIllustriousSDXL_v160`, 90 × v170, 124 Anima-family,
105 upscale-only graphs with no checkpoint, the rest novaAnimeXL), 80% of it rope/restraint
content, **median 44 tags / 666 chars / 4 groups (9 lines)** for IL, 59 tags / 1052 chars for Anima.

---

## 0. 硬性要求：一行一组，段落之间空行

**两组要求，别混：**

1. **空行（`\n\n`）分隔"区块"。** 场景块 / 人物块 / 风格块之间空一行。这是给人**手动改**用的：
   改一条时只需定位一个区块。`[measured]` 把六组压成一行的版本会当场被要求返工
   （人的注意力有限，处理不了黏着的长文本）。
2. **单换行（`\n`）分隔"要素组"。** 同一个区块里，**一个语义要素占一行**：姿势一行、缚法一行、
   表情一行、服装一行。

`[corpus]` 第 2 条不是我的偏好，是实际写法：**85% 的提示词在区块内部用单换行**，
中位数 **9 行 / 4 个空行区块**。只有 4% 是一条到底。所以正确粒度是**行**，不是区块——
把六个要素堆成一行同样难改，把每个要素都用空行隔开又太松散。

### 分组的实际顺序 `[corpus]`

经验顺序，按"类别平均出现的相对位置"（0 = 第一组，1 = 最后一组）排出来：

| 位置 | 内容 | 依据 |
|---|---|---|
| 0.06–0.17 | **风格 / 质量尾**（固定词块，见 §1） | `score` 0.06、`style` 0.13、`quality` 0.15 |
| 0.35–0.45 | 场景 / 镜头（`indoor, wooden floor, full body, front view`） | `lighting` 0.35、`environment` 0.40、`camera` 0.45 |
| 0.49–0.61 | 主体 + 姿势 + 缚法 + 表情 | `bondage` 0.49、`expression` 0.61、`pose_action` 0.60 |
| 0.71–0.75 | **角色身份 + 外观 + 服装**（角色名 \(系列\)、发色瞳色、衣着、裸露状态） | `appearance` 0.75、`clothing` 0.73 |

**最重要的一条：角色身份块放在最后三分之一。** `[corpus]` 284 条带
`名字 \(系列\)` 的提示词里，身份块的平均相对行位置 **0.71**（中位 0.75），
**69% 落在后三分之一**，只有 3 条放在第一行。

这**不是**插件规则里"质量 → 主体 → 服装 → …"的顺序（那个顺序要求角色开头）。
两种都成立，区别在于代价：

- 角色块**放最后**是社区惯例，好处是**改角色不动场景**——同一套姿势/场景/缚法提示词，
  换最后一行的角色块就能换人（corpus 里大量 `xtll_*`、`mh*` 系列就是这么批量产的）。
- 角色块**放前面**是 LoRA 训练时的常见排布，`[measured]` 换装实验里角色块在前也是有效的。
- **一条硬约束**：角色触发词 + 外观描述的完整性比位置重要。位置随你，但同一个系列内保持一致。

### 真实形状（`[corpus]` 逐字，`ayaka_rope (1)`）

```
facing to viewer, view from front,
<空行>
1girl sitting on floor, wariza, W-legs,
arms bondaged behind her back, red rope,
pantyhose pull down on knee, wet thighs,
egg vibrator controller bind on her thigh with tape, vibrator cord from her crotch,
legs bondaged,
shy, black blindfold, orgasm face mouth gag, ball gag,
ayaka \(springbloom missive\), light blue hair, butterfly hair ornament, hair flower,
brown headwear, braid, hair up, mole under eye, blue eyes, white collar, blue dress,
long sleeves, white pantyhose, high heel boots, blue butterfly ring,
<空行>
;
masterpiece, best quality, absurdres, ultra detailed, ... extreme details,
```

注意：姿势/缚法/道具/表情**各占一行**，角色块压到最后两行，风格块在末尾——
同一份文件里多数提示词把风格块放在开头（§1），两者都行，**但风格块只出现一次**。

---

## 1. 固定风格块（可直接抄）`[corpus]`

### 伪写实 / 无描边（"3D 渲染感"）`[measured]`

想要**没有描边、靠渐变和真实光照塑形**的半写实画面（参考图 `146351817_p0` 那种），
本机可用的杠杆有四个，缺一个都会退回平涂：

1. **换 UNET：`anima-aesthetic-v1.1.safetensors`**（不是 `anima-base-v1.0`）。这是本机**唯一**
   真正管用的风格杠杆；base 版无论如何写词都出平涂+描边。
2. 正向加去描边块：
   `(semi-realistic:1.4), (no lineart:1.4), (lineless:1.3), (soft shading:1.3), realistic skin texture, subsurface scattering, airbrushed, painterly rendering, soft rim light, cinematic lighting, depth of field`
3. 负向压平涂/描边：
   `(lineart:1.4), (outlines:1.3), (black outlines:1.3), (cel shading:1.3), (flat color:1.3), anime coloring, (thick lines:1.2), sticker, vector art, comic`
4. **40 步 + cfg 5.5**（低 cfg + 高步数把它推离卡通；Anima 默认 30/8 会拉回平涂）。

`[measured]` 实测一次到位：同题面下 aesthetic 版出来的是无描边半写实（布料褶皱、皮肤次表面、
麻绳扭转、背景带景深），base 版是平涂描边。

附带两条：

- 本地**没有**朋友那套伪写实画风 LoRA（`gpt-image-2_anima-base1_v*`、`anima_quality`、
  `anima-highres-aesthetic-boost`），所以这条路是"**finetune + 题面**"；装上那些 LoRA 还能再稳一档
  （正好是 §8b "画风靠 LoRA、质量词替代不了"那条规则的反面用法）。
- **近景下被推向镜头的那块身体会称霸画面**：同一次实测里 `(soles towards viewer:1.2)` 让脚掌变成
  主体（参考里脚显眼但不主导）。近景/广角构图里，把某个部位推向镜头要**降权或干脆不写**。

脚本：`experiments/repro_146351817/run_realistic.py`（`--unet/--steps/--cfg` 可换着试）。

`[plugin]` 规则 6 说"开头写质量词"，`[measured]` 说
`masterpiece, best quality, absurdres` **本身不构成风格**——两句都对，因为真正的风格来自
跟在质量词后面的那一串。corpus 把这件事量化了：

- **304 / 536（57%）**的开头 18 个 tag **逐字相同**；
- 307 条把风格块放在**第一行**，53 条放在**最后一组**，只有 3 条前后都放；
- 173 条完全没有 `trending on artstation`（多为 Anima / pony 系）。

**IL 主用（无 airbrush 版，最通用）：**

```
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest,
highly aesthetic, intricate details, professional artwork, trending on ArtStation,
digital illustration, painterly style, soft gradient shading, professional digital painting,
extreme details,
```

**IL 厚涂版**（把 `airbrush rendering` 加回 `soft gradient shading` 后面，73 条逐字使用）。

**IL 电影光长版**（`ht_glby*` / `xtll_glby` 系列，25 个 tag，把光位也写进风格块）：

```
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest,
highly aesthetic, intricate details, professional artwork, trending on ArtStation,
digital illustration, painterly style, soft gradient shading, gentle color transitions,
airbrush rendering, soft cinematic lighting, volumetric lighting, rim lighting,
subsurface scattering on skin, gentle highlights and shadows, depth of field, bokeh,
dreamy atmosphere, highly detailed texture, semi-realistic anime illustration,
professional digital painting, extreme details,
```

**Anima 版**：风格块的等价物是**风格 LoRA 的触发句 + 分数 tag + 年份 tag**，逐字照抄：

```
An aesthetic digital illustration masterpiece that is anime-based with realistic skin,
hair, clothing textures, physics, anatomy, and lighting while keeping the anime style face.
masterpiece, best quality, very aesthetic, amazing quality, absurdres, newest,
year 2025, year 2024, score_9, score_8, score_7, detailed background, depth of field,
photorealistic textures,
```

`[corpus]` 该触发句出现在 15+ 条里，是 `anima-highres-aesthetic-boost` / `anima_m4me` 系的
trained words（见 §7）。`score_9, score_8_up, score_7_up` 是 pony 系写法（17% 的记录），
**只在 pony 底模上用**；IL 上出现 `score_*` 只是噪声（9%）。

**反面案例（我们自己的历史数据）**：语料按作者切分后，**Anima 上的写法和 IL 上的写法几乎正交**——
Anima 语料里 91% 用自然语言句子、93% 带 `@画师` 风格触发词、63% 用权重（中位 1.45），
而**我们自己那批 Anima 记录里 33% 直接把 IL 的 18-tag 风格块贴了上去**，88% 没有句子。
IL 的风格块不是 Anima 的风格块；跨底模复用风格块 = 白烧一批图。

---

## 2. 负面提示词是"第二块画布"，不是缺陷清单

`[plugin]` 规则 9/10 只说"负面列缺陷、一到两行"。corpus 显示负面承担了**五种**功能，
中位长度 234 字符 / 20 个 tag（Anima 系 737 字符）：

| 用途 | 占比 `[corpus]` | 例子（逐字） |
|---|---|---|
| 固定缺陷块 | 高（几乎每条） | 见下面模板 A/B |
| **材质/风格纠正** | **55%** | `brush, comic`（水彩 LoRA）；`flat colors, cel shaded, flat shading, hard edges, bold outlines, thick lines, simple shading, no gradients, low contrast, 2d anime screencap`（锁厚涂风） |
| 角色服装残留清除 | 14% | `dress, hat`；`detached sleeves, tassel, backless dress, bridal gauntlets, high heels`；`hat, toque blanche, tail`；`shoes, high heel` |
| 裸露 / 情色压制 | 11% | `nsfw, nude, open mouth` |
| **多人压制** | 10% | `people, person, girl`（主体是具名角色时，防止多画一个路人）；`double head, 2girl` |
| **姿势/器具压制** | 6% | `hanging rope, suspension`（想要"落地绳"而不是吊缚）；`ribbon, satin ribbon`（防止绳子被画成缎带）；`crotch rope, legs bondaged`（反向：不要某条绳） |

**要点**：想要什么写正向，**"这个词会带来的不想要的读法"写负向**。
`red rope` 的默认读法之一是悬吊和缎带 —— 所以 `hanging rope, suspension, ribbon, satin ribbon`
是最常见的搭配修正。这比在正向里写 `not hanging` 有效（§5）。

**模板 A**（`[corpus]` 136 条共用，偏"通用缺陷"）：

```
low quality, worst quality, lowres, blurry, jpeg artifacts, bad art, poorly drawn, low detail,
bad composition, signature, watermark, text, error, cropped, extra finger, wrong body, bad hands,
multiple girls, nsfw, nude, open mouth, hanging rope, suspension, ribbon, satin ribbon
```

**模板 B**（136 条共用，偏"aesthetic"系；`pixelated/oversaturated/underexposed` 是它的标志）：

```
low quality, worst quality, lowres, blurry, pixelated, jpeg artifacts, bad art, poorly drawn,
low detail, oversaturated, underexposed, bad composition, signature, watermark, text, error,
cropped, extra finger, wrong body, brush, side view, wrong hand
```

**模板 C**（Anima "形体纠正"系，101 tags 的那一族）：`ungagged, shiny skin, oiled skin,
glistening skin, oily, greasy, plastic texture, voluptuous, thick thighs, wide hips, fleshy,
heavy proportions, chubby, simple background, white background, border, frame, white margin,
polaroid, letterboxed, comic panel, text, watermark` ——
**注意到 `ungagged` 排第一**：它压制的是"口球 LoRA 在没写口球时也画口球"的过冲。

经典 danbooru 系（`bad anatomy, bad hands, missing fingers, extra digit, fused fingers,
long neck, poorly drawn face...`）在 corpus 里只占 23%，说明**负面词也要跟底模的语料走**：
IL v1.x/v2.x 吃 aesthetic 系，pony 吃 `score_1..3, source_pony, source_furry, anthro, furry`，
Anima 吃形体/材质系。

**长度**：`[measured]` 一版把 60+ 缺陷词塞进负向、还给鞋类词加 `(shoes:1.45)` →
鞋子纹丝不动，反而因负向过载把整张图渲染成绿/红海报化；另一版只用 20 个词
（`shoes, high heel`）配合 cfg 8 / 20 步 → 鞋直接消失。**负面也要瘦身。**

---

## 2b. 简洁：长度本身就是成本 —— **一个意图一处说** `[rule]`

> **好提示词一定是简明扼要的，不是一篇论文。** 条件是按 token 竞争注意力的：词越多，每个词分到的
> 相对权重越低 ⇒ **你真正想要的那几个 tag 被稀释掉**，于是"什么都写了"＝"什么都没到位"，
> 根本弄不出想要的东西。写得长不等于更保险，只等于更平均。

**必须删掉的两种写法**

1. **正负双写同一件事**（最常见，也最浪费）：正向 `no shoes`，负向 `shoes, boots, sneakers`，
   再补一句 `not wearing shoes` —— 同一个意图说了三遍，既占预算又互相打架
   （正向里的否定模型读不懂，§5 已实测）。**想排除 ⇒ 只在负向写最准的词；正向里不许出现否定。**
2. **正向同义堆叠**：`long hair, very long hair, flowing long hair, beautiful long hair` ——
   第 2–4 个不提供任何新信息，只是把同句里的服装/姿势/场景摊薄。**留最准的那一个**
   （先查它到底指什么，§9）。

**操作规则（可照抄）**

- 每个内容组只留**承重的那几个词**；分组空行（§0 硬性要求）让"整组删掉"不伤别的组；
- 负向 ~30 tag 是**上限不是目标**；模板里的通用缺陷块**别重复抄两遍**（§2）；
- 画面不对时第一反应是**删词**，不是加词或加权（§8、§11：**先减词，再调权重**）；
- 长度自查：正向要素 9 行上下（§0）；超了先问"**删哪一个**"，而不是"还缺什么"；
- 判据沿用：**加一个词如果没让画面可辨地改变，它就是噪声，删掉。**
  （文末"不是越多越好"那节有正/负两组实测。）

**三个检查维度，互不替代**：§0 管**结构**（一行一组）· 本节管**总量**（多不多 / 有没有重复）·
tag review（`references/tag_review.md`）管**对不对**（互斥、残留、与当前场景冲突）。

**反模式**

- 把"更保险"理解成"再补两个同义词"；
- 把用户的每句形容词都翻成一个 tag —— 用户给的是**意图**，不是 tag 清单；
- 只增不减：上一版少写了什么，这版就加上去，从不回头看该删哪个（prompt 会单调膨胀，
  直到想要的 tag 全部被稀释）。

---

## 3. 绳索 / 束缚词表（本项目的主战场）`[corpus]`

80% 的 corpus 是绳缚内容，这部分是它最有价值的产出。按使用频次排（df = 出现该 tag 的提示词数）：

| tag | df | 说明 |
|---|---|---|
| `shibari` | 266 | 主词，一半的提示词都写 |
| `red rope` | 253 | 材质 + 颜色，和 `shibari` 一起写 |
| `ball gag` | 122 | 口球主词 |
| `crotch rope` | 107 | 股绳 |
| `mouth gag` | 107 | 通用口部束缚 |
| `arms behind back` | 93 | **规范 tag**，优先用它 |
| `bound legs` / `bound ankles` / `bound knees` / `bound thighs` / `bound arms` / `bound wrists` / `bound feet` | 79 / 23 / 16 / 15 / 35 / 6 / 6 | **按部位逐个绑**是标准做法，别只写一句"绑住腿" |
| `breast bondage` / `breast rope` | 78 / 27 | 胸绳 |
| `bondage` | 76 | 泛词 |
| `arms bondaged behind back` | 63 | 非规范写法，但**大量使用**（见下） |
| `legs bondaged` | 47 | 同上 |
| `blindfold` / `black blindfold` | 32 / 9 | |
| `suspension` | 29 | 吊缚；**单独写容易变成倒吊**，配合 `leg suspension` / `other leg bent` 约束。**悬挂的正确搭配是 `suspension` + `hang up`**：只写"吊"的散文（`hanging in midair` 之类）常常落成"坐在床上/被托着"，而 `hang up` 能把"被绳子吊起、身体离地"这层读出来 |
| `harness ball gag` / `gag harness` / `ring gag` / `cleave gag` / `wiffle gag` | 28 / 12 / 22 / 5 / 2 | 口球细分；`harness` 是"带头带"的那款 |
| `shibari over clothes` | 22 | 隔衣缚（避免模型自动脱衣的关键词） |
| `skindentation` | 15 | 绳勒进皮肤（danbooru 原生 tag，肉感全靠它） |
| `bitchsuit` / `straitjacket` | 15 / 13 | 拘束衣类 |
| `hogtie` | 14 | 反绑跪伏 |
| `spreader bar` | 5 | 撑杆 |
| `folded legs` / `M-legs` / `W-legs` / `wariza` | 6 / 16 / 13 / 12 | 腿部几何词，`wariza`（鸭子坐）比 `M-legs` 更常被正确渲染 |
| `nipple clamps` / `leash` / `vibrator cord` / `egg vibrator bulge` | 11 / 9 / 13 / 11 | 附件 |

**规范 vs 非规范写法。** `bondaged`、`arms bondaged behind back`、`legs bondaged` 都**不是**
danbooru tag，但在 corpus 里合计出现 100+ 次并产出了可用的图（`bondaged` 含 `bondage` 子词，
文本编码器能接上）。结论不是"可以乱写"，而是：

- **默认用规范 tag**（`arms behind back`, `bound arms`, `bound ankles`）——只有规范写法带训练先验；
- 规范写法渲染失败时，**加**非规范变体（两者同写不冲突，corpus 里常见 `bondage, bound, bound arms` 三连）；
- 拼错到脱离原词就纯属噪声：corpus 里的 `ctorch rope`、`amrs bondaged behind back`、`tembling`
  只能靠旁边的正确词救场。**优先规范拼写。**

### 绳的"拓扑句子"——本项目该学的部分

`[corpus]` 14% 的提示词用**整句**描述绳路（不是 tag 堆），且这些句子高度复用：

```
She is tied up by red ropes in a kikkou shibari style.                       (×10)
Her legs are tied together by large amounts of red ropes.                    (×11)
Her upper body is tied in shibari style by large amounts of red ropes,
  and her arms are tied behind her back.                                     (×7)
Her legs are bound together from thighs to ankles by red ropes.
Her hands are tied behind her back. She is wearing a harness ball gag,
  and has a blush and tears in eyes.                                         (×15)
the gag's straps are black, and this strap goes up over the nose,
  meets at the top, and then goes around the back of the head
  to form a harness gag.                                                     (×6)
A ball gag with harness and leather straps is secured in her mouth.
Her other leg is suspended, raised high above her head.                      (×4)
Her dress is rolled up by the crotch rope, revealing her white panties.      (×2)
```

**这和 §8「短动作短语胜过几何描述」不矛盾，两者管的是不同的东西：**

| 要描述的东西 | 写法 | 例子 |
|---|---|---|
| **拓扑**：什么和什么相连、绑在哪个部位、绕几圈 | **句子可以，而且更清楚** | `Her upper body is tied in shibari style by red ropes, and her arms are tied behind her back.` |
| **属性**：颜色、材质、款式、状态 | tag 短语 | `red rope, shibari, crotch rope, skindentation` |
| **像素级几何**：某块布料抬高几厘米、哪条边在上 | **别写，去做遮罩** | `the centre panel pressed flat, the outer corners above the hip line` —— 8 张图全无视 |

`[measured]` "短句胜过长几何描述"的失败案例属于第三行；corpus 里那些生效的绳句属于第一行。
**判据：句子里出现"相对量/像素量"就停手，去 `references/blender_control_pipeline.md` 或遮罩。**

---

## 4. 多主体：标签 + `;` 分块

corpus 的双人/多人写法（`[corpus]` 计数）：

| 模式 | 条数 | 例 |
|---|---|---|
| `front girl` / `other girl` | 29 | `front girl, lying,` … `other girl, sitting beside,` |
| `left girl` / `right girl` / `meddle girl` | 18 | 三人及以上并排时用 |
| `girl below` / `girl above` / `1girl over` | 6 | **上下叠压构图**（骑乘、叠躺）——不写就角色互换 |
| `sub girl` / `dominatied girl` / `dominator` / `dominant female` | 5 | 按关系角色命名 |
| `first girl:` / `girl A:` + `{}` 包块 | 3 / 1 | 显式块结构 |
| `2girl` / `2girls` | 60 / ~10 | 先写数量 |
| `1boy` / `1man`（含男女同框） | 38 | |

`[corpus]` 结构最完整的一例（`belle_vvn_handup (1)`，逐字）：

```
indoor, black sofa,
2girl, yuri,
;
first girl:{
  1girl sitting on sofa, left leg raise up, vertical split, left foot hang up,
  red rope, arms bondaged behind back,
  wet crotch,
  mouth gag, tape gag, blush, shy, orgasm face,
  belle (ZenlessZoneZero),
  crossed bangs, hairclip, ahoge, blue hair, green eyes,
  bare body, breast, bare neck, nipples,
  white pantyhose, single thigh strap,
}
;
second girl:{ ... }
```

**规则（`[measured]` + `[corpus]` 一致）：**

1. 先写数量（`2girl`）；
2. 每个角色一个**带标签的块**，标签写在块首（屏幕位置或关系角色）；
3. 每个角色**自己**的姿势 + 状态（表情/情绪）必须写全——只写全局动作会让两人做同一件事；
4. `;` 或空行分块，别让两套姿势词混在一行；
5. **`left` / `right` / `below` / `above` 的绑定是防"角色互换"的最强单点修正；**
6. 负面加 `multiple girls`（单人时）或 `2girl`（要三人时）来钉住人数。

### Anima 的多主体形态 `[measured]`

Anima（qwen-image）用**同一套块结构但写句子**，`;` 独立成行：

```
<场景句>: where / weather / ground
;
<全局动作句>: 一句话点名两人的关系
;
left <名字>: <角色触发 + tag>, <该角色自己的姿势与状态>
;
right <名字>: <角色触发 + tag>, <该角色自己的姿势与状态>
;
<风格尾>
```

同一底模、同一参数、**只改提示词文本**：用上面的块 → 一个女孩蹲下系鞋、另一个站着脸红
低头手被红绳反绑，角色、身高关系、动作全对；写成 tag 汤 + 一句场景（4 个种子）→ **四张全错**，
两人都躺地上、缚法读成胸带、服装不分、谁是谁逐种子变化。参见 §3 的 Anima 表格。

---

## 5. 永远不要在正向里写否定 `[measured]`

文本编码器不会把否定词绑到它后面的名词。`no apron` 贡献的是 *apron*；
`nothing at all in the middle` 贡献 *middle*；`the rest still hidden` 最多贡献一个 *hidden* 的风格暗示。
`no / without / nothing / except` 都没有可靠的算子。

**规则**：正向写**肯定目标**（`plain white summer dress`、`white bloomers`），
要排除的全部写进负面：

```
NEG: navy blue maid dress, maid outfit, apron, frilled apron, black skirt,
     low quality, worst quality, lowres, blurry, jpeg artifacts, bad anatomy, ...
```

定量约束同理。"裙子被吹起但只露外侧一点点"不能写成
`only the very edges, nothing in the middle`——模型读到名词，把中间也画出来。
要么正向描述"能看见的部分"，要么用遮罩 / ControlNet 控制。

`[corpus]` 交叉验证：在**纯 tag** 提示词里（399 条），含 `no <名词>` 的只有 **2%**、
含 `only` 的 2%（含 `no` 的 7% 集中在中文改图指令句里）。社区写法与这条规则一致，
但这是**惯例**，不是实验证据——因果仍然来自 `[measured]`。

*注*：`[plugin]` 文档**没有**这条规则。最接近的是 `llm-templates.json` 规则 9
（"正向描述你要的，负向列你要避免的"），是更弱的表述。

---

## 6. 换装 / 换角色需要显式替换从句 `[measured]`

角色 LoRA 自带服装且权重很高。只在正向点名新衣服**会输给 LoRA**。两处必须同时改：

```
POS: ... she wears a plain white summer dress instead of a maid outfit: white dress, ...
NEG: navy blue maid dress, maid outfit, apron, frilled apron, black skirt, ...
```

DeepSeek whale-girl LoRA（训练数据是海军蓝女仆装）上的实测：

| 正向提示词 | 白裙出图率 |
|---|---|
| 只写 `white dress`，无替换从句、无负面 | 0 / 4 |
| `plain white summer dress instead of a maid outfit, no apron` + 服装进负面 | **4 / 4** |
| 有替换从句但负面漏掉原服装 | 部分——袖子/围裙残留 |

换装时**把原服装 tag 从角色块里删掉**（`navy blue maid dress`、`deepseek_maid_outfit`、
`white frilled apron`），留着就是和替换从句对抗。**身份类 tag 全部保留**
（发色、瞳色、鳍耳、尾巴、头饰形状）。

---

## 7. 角色块：位置、LoRA 触发词、`<lora:>` 写法

`[corpus]` 实测（§0）：身份块平均在第 0.71 的位置。三种写法都出现在 corpus 里：

```
sprkamiaya, light blue hair, blue eyes, blunt bangs, mole under eye, braid, hair rings,
brown headwear, butterfly hair ornament, blue dress, frilled dress, long sleeves,      # LoRA 触发词打头
escoffier \(genshin impact\), green eyes, orange hair, ahoge, long hair, bow, tail,   # 规范 (系列) 写法
Alice Thymefield from Zenless Zone Zero, with animal ears, black pantyhose, ...       # 散文式（Anima）
```

- **`名字 \(系列\)` 是 corpus 的主导写法**（284 条；`\\(` 转义出现于 51% 的记录，
  60% 的记录带系列名）。特殊字符必须转义，否则括号会被当作权重语法。
- **LoRA 的 trainedWords 照抄，不要自己编**——包括**整句**触发词（§1 的 aesthetic 句）
  和带权重的艺术家触发词 `@[himitsu (hi mi tsu 2):0.8]`、`@[wlop::0.6]`（各 54 条，
  是同一批"多画师风格平均"写法的一部分）。`@name` 式风格 tag 出现在 19% 的记录里，
  常见组合是 `@houkisei, @rokuwata tomoe, @atdan, @[himitsu (hi mi tsu 2):0.8], @wlop, @m4me, @n4g1style`。
- `<lora:name:weight>` **写进提示词文本**的写法有 15 条——我们的管线走 LoraLoader 节点，
  不要用这种内联写法。
- `[corpus]` 里 27 条含 `__char__` / `${char}` 占位符，是批量套模板用的（同一套场景/缚法
  配不同角色）。**这正是把角色块放最后的好处**：换人只改一块。

---

## 8. 权重语法：可以用，但**幅度跟底模走**，且本管线未做 A/B

`[corpus]` 24% 的 IL 提示词、51% 的 Anima 提示词用了 `(tag:w)`：

| 底模族 | 用权重的记录 | 加权 tag 数 | 中位 w | p90 w | 最大 w | w ≥ 1.5 的比例 |
|---|---|---|---|---|---|---|
| IL / SDXL | 62 | 136 | **1.20** | 1.30 | 2.50 | **5%** |
| Anima | 69 | 327 | **1.50** | 2.00 | 8.00 | **51%** |

读法：**IL 上权重是微调**（几乎都在 1.1–1.3，极少数到 2.5）；**Anima 上权重是主控制手段**
（一半的权重 ≥1.5，出现 `(shibari:4.0)`、`(4 toes:4.0)` 这种量级）。在 IL 上写 `:4.0`
没有 corpus 支持，只会把画面拉成海报化。

另一类写法是**反向**的：`@[name::0.8]`（注意双冒号）表示"这个风格 LoRA 只用 0.8"，
属于触发词内嵌权重，与 `LoraLoader` 的 `strength` 等价但写在文本里。

⚠️ `[pipeline]` 记录显示 **`(keyword:1.2)` 在本管线的正向效果尚未验证成功**。
所以：**先减词，再调权重**（见 checklist 最后两条）；要用就用 1.1–1.3，并且一次只动一个，
否则分不清是权重还是词表长度在起作用。

### 8b. 与语料原题面的实测对照（5 例同图复现）

把同一批图的**原题面**与**本次为复现写的题面**放在一起量（脚本
`experiments/compare_prompts.py` → `experiments/prompt_style_compare.md`）：

| 特征（均值） | 原题面 | 本次我写的 |
|---|---|---|
| 字符数 | 1044 | 1635 |
| 行 / 块 | 8 / 5 | 8 / 5（一致，说明行级+分块结构没问题） |
| 带权重 token | **3** | **18** |
| 最高权重 | 1.5–3.0 | 1.4–1.8 |
| `@画师` | **6** | **0** |
| 负向字符 / 行 | 827 / 1 | 659 / 4 |

四条要改的写法：

1. **权重的"数量"要减，"幅度"不要加。** 原题面平均只 3 个权重而我写了 18 个——这条要学；
   但它们把峰值拉到 2.0–3.0（甚至 `(shibari:4.0)`）**不是要跟的目标**：
   **权重加大画面容易崩坏，除非确实需要，否则不要超过 ~1.5。**
   要让某样东西"必须出现"，正确手段是**把词命名准**（原生 tag / 独占一行 / 位置词）+
   **在负向压掉它的替代品**，而不是把权重往上推。
2. **画风不是质量词能替代的。** 原题面 6/6 都带同一个六人 `@画师` 块，且 LoRA 栈里**总**有画风
   LoRA（`qiandai_yiyu_style_v2` / `NAGI` / `anima_m4me` / `BlueArchiveStyleB1` …，权重 0.2–0.8）。
   `very aesthetic, absurdres, year 2025` 只能提质量、**提不出画风**；复现语料图之前先查有没有画风
   LoRA，没有就如实说明"风格这一层复现不了"，不要用质量词假装。
3. **正向头部保留一个"评级 token"**（`safe` / `questionable` / `explicit` / `nsfw`，或 pony 的
   `score_9, score_8_up, score_7_up`）。原题面都有，我全丢了——它是显式的内容尺度旋钮。
4. **负向要贴语料观感就直接用它们那条审美模板**：原题面把 ~800 字符的审美/质量串**挤在 1 行**
   （且互相抄袭，同一条出现过 136 次），内容类负向另起一行。注意 `BDF525D9` 的原题面头部混进了
   PromptManage 的 UI 串（`? (false), ?? (both), append, yes, …`）——语料里 26% 的记录有这种泄漏，
   抄它们的负向时别把 UI 垃圾一起抄进来。

---

## 9. 写之前先查这个词到底指什么

日本网络 / 服装 / 同人词汇，写进提示词前先查（pixiv 百科 `dic.pixiv.net/a/<tag>`、萌娘百科）。
意译外来语和网络黑话经常是错的，一个错读要烧一整批：
某主题词被当成字面道具画了 12 张，实际含义是服装轮廓，根本不涉及道具。

`[corpus]` 的两个活例：

- `pillory`（枷）—— 我本写成"厚木板 + 圆孔"这类结构描述，画出来是奇怪的板子；
  换成原生 tag `pillory` 后器具直接对了（见 `experiments/ganyu_cangue/README.md`）。
- `locker` 1789 帖、`locker_room` 706 帖可用；`metal_locker` / `storage_locker` /
  `school_locker`（= 日式鞋柜）/ `footlocker` **都不是 tag**，写了等于没写。

与 `SKILL.md` §"经典 / 知名角色"同一条纪律：**先确认，再写**。

---

## 10. corpus 里见到的坏习惯（别学）

- **插件 UI 字符串泄漏进提示词**（比上一条常见得多）：`Select the Wildcard to add to the text`、
  `Select the LoRA to add to the text`、`reproduce`、`否 (false), 两者 (both), append, yes` ——
  这些是 `PromptManage` 界面按钮/选项文字被写进了条件。`[corpus]` 里 **29 条（26% 的"他人下载"组）**
  含这类字符串，纯噪声；我们自己的记录里也见过 `否 (false), 两者 (both)`。
- **插件 token JSON 泄漏**：1 条记录的 `CLIPTextEncode.text` 里塞了
  `PromptManage` 的整个 token 数组（`[{"id":"token_1_1779355449532","text":"medieval dungeon",...`），
  单条 17,965 字符——占满整个上下文窗口，把真正的提示词挤出去。
  **提交前看一眼文本首尾**，别把插件内部数据编进条件。
- **`extreme details,,,` 这类重复逗号**：41% 的记录有 `,,`（拼接多段时留下的），
  本身无害，但它是"手工拼提示词"的痕迹；我们的提示词由脚本生成，不该有。
- **尾逗号** 73% —— 无影响。
- **两种写法混用**：`2girl` 与 `2girls`、`(series)` 与 `\(series\)` 同时存在。
  各自内部统一即可，别在一条提示词里混。
- **裸 `1girl, solo` 与具名角色并存**（80% 写 `1girl`、39% 写 `solo`）：
  单人图里没坏处；**多人图里 `1girl` + `2girl` 并存就是自相矛盾**，corpus 里确实有这种。

---

## 11. 提交前 checklist

**第 0 项（先做）：tag review —— 逐 tag 对照"场景清单"，见 `references/tag_review.md`。**
提示词来路不是为当前场景从头推的（LoRA 触发词 / 抄来的工作流 / 上一版 prompt / 别的 task），
或者场景变了任意一项（服装、姿势、道具、地点、视角、画幅）⇒ 这一步**不许跳**；
只通读一遍不算做过了（模型输出 token ≠ 意识到 token 之间的矛盾）。

- [ ] 区块之间空行、要素各占一行（9 行上下），不是一坨？
- [ ] 风格块**只出现一次**，且质量词后面跟的是**真风格**（不只是 `masterpiece, best quality`）？
- [ ] 正向里没有 `no / without / nothing / except / only`？
- [ ] **没有正负双写**同一意图？（想排除 ⇒ 只在负向说一次，正向不许出现否定，§2b/§5）
- [ ] **没有同义堆叠**？（同一要素写了两遍以上 ⇒ 只留最准的那一个，§2b）
- [ ] 要排除的东西（原服装、不想要的读法、多余的人、不想要的材质）在负面里，且负面 ≤ 30 tag？
- [ ] 场景语汇按底模走：IL 用 tag、Anima 用句子？
- [ ] 换装？→ 正向 `instead of X` **且** 负面含 X **且** 角色块里删掉了 X？
- [ ] 绳索写明**部位**（`bound ankles, bound knees, ...`）而不是只有一句"绑住"？
- [ ] 姿势涉及**拓扑**（谁连谁）→ 句子可以；涉及**像素几何** → 去做遮罩/ControlNet？
- [ ] 多人？→ 数量 + 每角色块 + 位置/关系标签 + 每角色自己的状态 + 负面钉人数？
- [ ] 角色块位置在本系列内一致？
- [ ] 主题词汇查过词典/tag 库？
- [ ] 权重（如果用了）**数量很少**且幅度不超过 ~1.5（见 §8b：加大容易崩坏），一次只动一个？
- [ ] **正负都瘦身**：加一个词没让画面可辨地改变，它就是噪声，删掉；
      **先减词，再调权重**——加权是最后手段，不是第一手段。
- [ ] 重绘？`denoise` **按这张画布重新标定**——denoise 越低覆盖的像素越少，
      "什么都没发生"通常是尺度不对，不是数值不对（`references/local_repaint_bigcanvas.md`）。

---

## 附录：这些数字怎么复现

语料来自 `<你的图片目录>`（**只读**，只拷副本）。脚本在
`<你的 ComfyUI 项目>\experiments\prompt_organize\`：

```powershell
cd <你的 ComfyUI 项目>\experiments\prompt_organize
python organize_workflows.py    # 扫描 11,154 张图，把带工作流的拷进 ref/（1,362 → 去重后 544）
python extract_wf.py            # PNG 内嵌 workflow -> wf/<name>.json（保留 \n\n；见 clean_text）
python analyze_corpus.py        # -> analysis/corpus_stats.md（分组、词频、位置、负面模板）
python analyze_extra.py         # -> analysis/extra_stats.md（身份块位置、权重幅度、负面用途）
python sample_prompts.py        # -> analysis/sample_prompts.txt（52 条最远点采样 + 分层样本，供精读）
```

- `extract_wf.py` 的 `clean_text()` 会**保留换行**；此前的 `norm_text()` 把 `\n\n` 压成空格，
  导致"所有提示词都只有 1 组"的假结论——**分析结构时先确认提取器没吃掉结构**。
- `wf/*.json` 字段：`image, src, size, positive, negative, checkpoint, loras, sampler, prompt_graph`。
- `negative` 的原样文本也在 `prompt_graph` 里，可回溯。
- corpus 快照：544 条（536 条正向、528 条负面），采集于 2026-09-28/29。


---

## Writing the prompt — structure, negation, costume swaps



Rules distilled from the local `comfyui_PromptManage` plugin
(`custom_nodes\comfyui_PromptManage`: `README.md` §LLM 提示词生成规则,
`web/llm-templates.json`, `prompt_example/LLM_analyze_prompt.md`), Anima-specific
adjustments, hard-won A/B measurements, **and a 536-prompt corpus harvested from
`<你的图片目录>` (2026-09-28/29)**.

**写提示词时先开速查表:`references/prompt_cheatsheet.md`** —— 可抄的正向前缀（IL/Anima）、
四套负面模板、8 条骨架模板（单人/双人 `{}` 块/容器内/悬吊/Anima）、词表（绳索按部位、
口球、腿几何、状态、镜头）、if→then 规则表、参数表。**展开的规则与证据**
（每条数字的来源、A/B 实测）：`references/prompt_writing.md`。语料统计复现：
`experiments/prompt_organize/analyze_corpus.py` + `analyze_extra.py`
（在 `agent_work\pixiv_publishing\` 下）。

**硬性要求：一级用空行（`\n\n`）分区块，二级用单换行（`\n`）分要素。** JSON 里写 `\n\n` 就是
ComfyUI 界面里的空行。理由：提示词是给人**手动改**的，人的注意力有限 —— 六组压成一行会被要求返工。
语料实测：**85% 的提示词在区块内部用单换行，中位 9 行 / 4 个空行区块**，所以正确粒度是
**行**——姿势一行、缚法一行、表情一行、服装一行；只有 4% 是一条到底。

**分组顺序（经验值，来自 536 条实际提示词的平均位置）：**

```
风格/质量块  →  场景/镜头  →  主体+姿势+缚法+表情  →  角色身份+外观+服装  →（可选）风格块
0.06–0.17      0.35–0.45      0.49–0.61               0.71–0.75
```

**角色身份块放最后三分之一**（284 条带 `名字 \(系列\)` 的提示词里平均位置 0.71，69% 落在后 1/3）。
好处是**换角色只改一块**，不动场景和缚法。位置本身不敏感，但**同一系列内保持一致**。
`[plugin]` 规则 4 的"质量 → 主体 → 服装 → …"是另一种排布，同样可用。

**风格块是固定的、可整段复制的**：语料里 **57% 的提示词开头 18 个 tag 逐字相同**，
即 `masterpiece, best quality, absurdres, ultra detailed, high resolution, newest, highly
aesthetic, intricate details, professional artwork, trending on ArtStation, digital illustration,
painterly style, soft gradient shading, professional digital painting, extreme details`
（Anima 系换成风格 LoRA 的**触发整句** + `score_9/8/7` + `year 2025`）。
`masterpiece, best quality, absurdres` 本身不是风格——风格在那串尾巴上。
**风格块只写一次**；重要特征给 2–4 个同义强化词。

**IL vs Anima — the one real difference:**

| content | Illustrious (SDXL) | Anima (qwen-image) |
|---|---|---|
| scene / mood / lighting / event | tag phrases | **a sentence** (clauses, semicolons OK) |
| constituent elements (clothing, body, props, style) | tag phrases | **comma phrases** |

Mnemonic: **"what happens" in sentences, "what is there" in phrases.**

**Never put negation in the positive prompt.** The text encoder does not bind `no`,
`without`, `nothing`, `except` to the noun that follows — `no apron` reads as *apron*.
Write the affirmative (`plain white summer dress`), and move the unwanted thing
(`apron, navy blue maid dress, black skirt`) into the **negative** prompt. Same for
quantitative constraints: `only the very edges, nothing in the middle` is read as
*edges* + *middle*, so the middle gets drawn anyway. (Corpus check: only 2% of tag-style
prompts contain `no <noun>`, so the convention agrees.)

**负向是"第二块画布"，不是只放缺陷。** 语料里负向的用途分布：材质/风格纠正 55%
（`brush, comic` 压水彩 LoRA；`flat colors, cel shaded, flat shading, hard edges, low contrast`
锁厚涂风）、角色服装残留清除 14%、裸露压制 11%、**多人压制 10%**（单人图加 `people, person, girl`）、
**姿势/器具压制 6%**（`hanging rope, suspension, ribbon, satin ribbon` —— `red rope` 的默认读法
之一就是悬吊和缎带）。原则：**"这个词会带来的不想要的读法"进负向**。
固定缺陷块也在 `references/prompt_writing.md` §2（三条可直接抄的模板）。负面同样要瘦身。

**Costume / character swaps need an explicit replacement clause.** A LoRA carries its
own outfit, and merely naming a new garment loses to it. Write
`plain white summer dress instead of a maid outfit`, **and** put
`navy blue maid dress, maid outfit, apron` in the negative — that combination took the
white dress from 0/4 to 4/4. Target garment alone in the positive: swap fails.

**短动作短语胜过几何描述 —— 但"拓扑"可以用句子。** `pull down the dress on crotch
with both hand` 一次就中；一段描述"中间那块压平、外侧两角高于胯线、大腿外侧露出一条窄边"
的段落被 8 张图全部无视。语料的修正：**绳路这种"什么连着哪里"的拓扑，句子反而更清楚**
（14% 的语料用整句：`Her upper body is tied in shibari style by large amounts of red ropes,
and her arms are tied behind her back.` ×7）。判据：**句子里出现相对量/像素量就停手，
去 `references/blender_control_pipeline.md` 或遮罩。**

**绳索要写部位。** 不要只写"绑住腿"：语料用 `bound ankles / bound knees / bound thighs /
bound legs / bound arms / bound wrists / bound feet` 逐个部位写，配 `shibari, red rope,
crotch rope, breast rope, skindentation`（绳勒进皮肤）、`shibari over clothes`（隔衣缚，
防自动脱衣）。完整词表 + 规范/非规范写法之辨：`references/prompt_writing.md` §3。

**整体缚法挑"有公认英文名的那个"** —— 14 种姿势实测（Anima，绿✅9 / 半⚠️5 / 认不出 0）：
最好使的是 `box tie / hogtie / gyaku ebi / frogtie / strappado / crotch rope / ippon tsuri /
human dog`，日文名单独写没有控制力。**绳的艺术编花（菱形 karada、龟甲 kikkou）出不来** ⇒ 要控制图或局部重绘；
`suspension` 默认读作**倒吊**，要立吊写 `single rope, bound wrists above head`；`spread eagle` 只是**摆开**、没绑四角；
双人并绑用 `bound together, back to back, ropes joined, 2girls` 一次就成。
逐姿势 tag 表与证据：`references/prompt_cheatsheet.md` §4「整体缚法姿势」。

**权重幅度跟底模走，且本管线未做 A/B。** 语料：IL 用权重时中位 **1.20**、只有 5% 到 1.5 以上；
Anima 中位 **1.50**、**51%** 在 1.5 以上（出现 `(shibari:4.0)`）。IL 上写 `:4.0` 没有依据，
会海报化。**先减词，再调权重。**

**Check the tag's real meaning before writing it** (esp. Japanese net/fashion terms —
pixiv 百科 / 萌娘百科). One theme word was drawn literally as a prop for 12 images
before the definition was checked; the correct reading was a garment silhouette. A
wrong reading costs a whole batch.

**提示词不是"越多越好"，正负都要适度 —— 写"承重的那几个"，不是写"齐全"。**

- **机制**：词表越长，每个词的相对权重越低（词表长度本身就是稀释）。堆到几十个词之后，
  新增的词不再增加控制力，只是把已有的词摊薄。
- **实测（负向）**：一版把 60+ 个缺陷词塞进负向、还给鞋类词加了 `(shoes:1.45)` → 鞋子纹丝不动，
  反而因负向过载把整张图渲染成绿/红海报化；另一版只用 20 个词、鞋类只留 `shoes, high heel`，
  配合 `cfg 8 / 20 步` → 鞋直接消失。
- **实测（正向）**：同一时期把角色 + 服装 + 姿势 + 表情 + 质量词全堆成一行长句的版本，
  反而不如用户自己写的"短句 + 分组空行"版本听话（那次短版直接把鞋去掉了）。
- **做法**：每个内容组只留**承重的那几个词**；分组空行（见本文件 §Writing the prompt 的硬性要求）
  让你能**整组删掉而不伤别的组**。
- **判据**：**加一个词如果没让画面可辨地改变，它就是噪声，删掉。** 先减词，再调权重 ——
  加权是最后手段，不是第一手段。



---

## 两个"模型不认识的新角色"同框、且不许反左右（标准解）


1. **先出一张不挂任何角色 LoRA 的"版式底图"**，把布局与关系做对；方向一律写**画面坐标系**
   （`the girl on the left side of the image`、`leans sideways to her left`），负面点名
   `mirrored composition / swapped positions`；
2. `check_parts.py --person-masks` 拿**每人一块、互不重叠**的掩版（见 `references/local_repaint_bigcanvas.md` 的「逐人掩版」节）；
3. **每次只挂一个人的 LoRA**（1.0/1.0 + sidecar 的完整触发词表），`denoise 0.65~0.75`，串行两次；
4. **每人的重绘提示词里都写一遍接触关系**（"the other girl leans her head on her shoulder"），否则边界处的姿态会被改坏。
> 中文角色名在 LoRA 文件名里常搜不到，但写在 sidecar 的 `model_name` 里（遍历匹配）；最全的触发词在
> sidecar 的 `civitai.images[].meta.prompt`。

**模型与代码位置（只给路径，不复制）**

| 东西 | 路径 |
|---|---|
| 套件根 | `<你的 ComfyUI 安装目录>\agent_work\experiments\anime_detection_suite`（`--suite` 或环境变量 `ANIME_DET_SUITE` 可覆盖） |
| 产线入口 | `<套件>\scripts\locate.py` |
| 推理环境 | `python`（ComfyUI 自己的 env，有 torch+ultralytics；`--python` 可换） |
| 权重 | `<套件>\runs\<name>\weights\best.pt`；每层的默认权重写在 `locate.py` 顶部的 `DEF_*` |
| M1 骨架 | **首选** `models/vitpose_ft_v6_labelbox_honly`（微调 ViTPose，HF 目录，top-down 吃 M2a 的 person 框）；缺目录或 `--no-vitpose` 时回落 `runs/m1_pose_coco17` + `runs/m1_pose17_1129`（取检出人多的）→ dwpose |
| M1 实测 | 固定 15 张报告集（279 关节）：ViTPose-ft 覆盖 88% / median 23.1px / p90 97.8；回落那套 71% / 66.0 / 341.6。配对聚类 bootstrap：与 dwpose 比 median「无差别」(−2.0px)、**mean 显著** −31.8px [−53.3,−11.0] |
| M2a 框 | `runs/m2a_det_996img_960`（person/hand/foot/head）+ `runs/m2a_foot_v1`（单类脚，取并集）；可选 `runs/m2a_hand_v2`（单类手） |
| M2b 面部 | `runs/m2b_face_v3_3983`（输入是 head 裁剪的 256×256） |
| M2b-gag | `runs/m2b_gag_v2` + `runs/m2b_gag_v4`（并集；口塞/眼罩） |
| M3 指趾判定 | 现役 `runs/m3_cls_aug_v1` + `runs/m3_review_v1`（**val 有训练集泄漏**，见套件 R43）；推荐 `runs/m3_clean_v1_s2026{1,2,3}010` |
| M4 绳子 | **不在本入口**：由 rope 模块产出 `data\masks\rope\<name>_{mask,conf}.png`，`locate.py` 只按约定路径引用 |

**已知限制（别把"没报错"当成"一定没问题"）**
1. M3 的诚实指标（按源图 5 折折外）：召回 **0.74（单模型）/ 0.79（双权重并集）**，ok 误报 14%~19%；**手（0.61）明显差于脚（0.85）**。
2. 脚召回靠三级兜底（4 类 ∪ 专用脚 ∪ 踝点外推）；契约 JSON 的 `boxes` **不含**兜底补的部位（见 `boxes_all`）。
3. 分辨率：`--imgsz 1600` 明显好于 1280（手覆盖 4/8→5/8，08 那种"趾多"从只被收紧旗标报出变成进主判定），代价约 1.5~2× 时间；本机 1536×1024 的图在 1280 下约 17 s/张、加 `--person-masks` 再 +10 s。
4. 上面的验收数字都来自**同一批 8 张图**，小样本；换风格/题材请自己再复核几张。
5. `rope` 只表示"**rope 模块有没有留下掩版文件**"（`present|missing`），**不代表图里没有绳** —— 本入口不做绳子检测。
6. 更细的接口与实验记录在 `<套件>\docs\`：`integration.md`（接口与降级）、`runs.md`（R1~R49 实验）、`handover_20261012.md`（现状与待定项）。

**契约 JSON 里给下游用的字段（新增字段，都是加法式的）**
- `points[].person_id` / `boxes[].person_id`：把点/框挂到具体某个角色上（多角色图做逐人处理时必需）。
- `n_persons`（M2a 的 person 框数）与 `n_skeletons`（M1 链数）分开报；旧的 `n_people` 仍是骨架数（兼容）。
- `boxes_all`：`boxes` **加上**兜底补的部位（专用脚 / 踝点外推 / 专用手），每个框带
  `origin: det | fallback`。做掩版请用 `boxes_all`，否则会漏掉兜底找回来的脚。
- `persons_masks` / `person_boxes_dropped` / `person_instances_merged`：见 `references/local_repaint_bigcanvas.md` 的「逐人掩版」节。

**想更新模型**：把新权重放到 `<套件>\runs\<name>\weights\best.pt`，然后改 `locate.py` 的 `DEF_*`，
或调用时用 `--m3-weights/--hand-weights` 指定；`--recipe rec` 就是"实测更好但还没转正"的那一套。

