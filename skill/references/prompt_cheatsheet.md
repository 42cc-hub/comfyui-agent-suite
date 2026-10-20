# Prompt cheat sheet — 照抄、替换、提交

给写提示词时用的速查。**展开的规则与证据**在 `references/prompt_writing.md`；
本表只放**可直接抄的块**和**查表用的词**。每条模板后面标了它的来源记录
（`wf/<name>.json`，语料库在 （我们的实验））。

30 秒流程：

1. 选底模 → 抄 §1 的**正向前缀**；
2. 选骨架 → 抄 §3 的**模板**（单人/双人/容器内/悬吊/Anima）；
3. 填内容 → 查 §4 的**词表**（绳索按部位、口球、腿部几何、状态、镜头）；
4. 抄 §2 的**负面**，按需加"不想要的读法"；
5. 过一遍 §7 的 10 秒检查。

---

## 1. 正向前缀（按底模）

### IL / SDXL 标准（语料 57% 逐字使用）

```
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest,
highly aesthetic, intricate details, professional artwork, trending on ArtStation,
digital illustration, painterly style, soft gradient shading, professional digital painting,
extreme details,
```

厚涂版：在 `soft gradient shading,` 后面插 `airbrush rendering,`。

### IL 电影光长版（要光位/景深时）

```
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest,
highly aesthetic, intricate details, professional artwork, trending on ArtStation,
digital illustration, painterly style, soft gradient shading, gentle color transitions,
airbrush rendering, soft cinematic lighting, volumetric lighting, rim lighting,
subsurface scattering on skin, gentle highlights and shadows, depth of field, bokeh,
dreamy atmosphere, highly detailed texture, semi-realistic anime illustration,
professional digital painting, extreme details,
```

### Anima（qwen-image）

风格 LoRA 的**触发整句**必须逐字照抄（这是 `anima-highres-aesthetic-boost` / `anima_m4me` 的
trained words）：

```
An aesthetic digital illustration masterpiece that is anime-based with realistic skin,
hair, clothing textures, physics, anatomy, and lighting while keeping the anime style face.
masterpiece, best quality, very aesthetic, amazing quality, absurdres, newest,
year 2025, year 2024, score_9, score_8, score_7, detailed background, depth of field,
photorealistic textures,
```

⚠️ **别把 IL 的 18-tag 风格块贴到 Anima 上**——33% 的 Anima 记录这么干过，
而 Anima 语料的惯例是"句子 + `@画师` 触发词 + 权重 1.5–2.0"。
⚠️ `score_9, score_8_up, score_7_up` 只在 pony 底模上用；IL 上写 `score_*` 是噪声。

---

## 2. 负面模板

### IL「aesthetic」系（**本项目默认**；语料里 94% 的自有记录用它）

```
low quality, worst quality, lowres, blurry, pixelated, jpeg artifacts, bad art,
poorly drawn, low detail, oversaturated, underexposed, bad composition, signature,
watermark, text, error, cropped, extra finger, wrong body, brush,
```

单人具名角色加一行 **多人压制**（语料里的做法，效果好）：

```
people, person, girl,
```

### 经典 danbooru 系（**给 Anima / 老底模用**，IL 上收益低）

```
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits,
cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark,
username, blurry, deformed, mutated, ugly, disfigured, extra limbs, malformed limbs,
fused fingers, too many fingers, long neck, poorly drawn face, poorly drawn hands,
missing arms, extra arms, bad proportions, cloned face, gross proportions
```

### pony 系（只在 pony 底模用）

```
score_1, score_2, score_3, source_pony, source_furry, source_cartoon, worst quality,
(worst aesthetic), bad quality, bad anatomy, (artistic error), old, mammal, anthro, furry,
feral, semi-anthro, lowres, (bad), off-topic, multiple views, comic, very displeasing,
chromatic aberration, (extra digits), fewer digits, jpeg artifacts, (artist name:2),
signature, twitter username, logo, watermark, censoring, mosaic censoring
```

### Anima「形体纠正」系（压制过冲；`ungagged` 排第一是**故意的**）

```
ungagged, shiny skin, oiled skin, glistening skin, oily, greasy, plastic texture,
voluptuous, thick thighs, wide hips, fleshy, heavy proportions, chubby,
simple background, white background, border, frame, white margin, polaroid,
letterboxed, comic panel, text, watermark,
```

### 追加块（按需拼在上面任意一套后面）

| 想要的效果 | 往负面里加 |
|---|---|
| 锁厚涂/半写实风，别出平涂动画 | `flat colors, cel shaded, flat shading, hard edges, bold outlines, thick lines, simple shading, no gradients, low contrast, 2d anime screencap` |
| 用水彩/画师风格 LoRA 时 | `brush, comic` |
| 想要**落地绳**，别出悬吊 | `hanging rope, suspension` |
| 绳子别被画成缎带 | `ribbon, satin ribbon` |
| 单人图别多画人 | `people, person, girl` / `multiple girls` |
| 别出双头/双人 | `double head, 2girl` |
| 清掉角色原服装 | 原服装的组成 tag（`dress, hat, apron, detached sleeves, tassel, high heels, shoes, ponytail`） |
| 不要裸露 | `nsfw, nude, open mouth` |
| 手部别崩 | `wrong hand, extra fingers, six fingers` |
| 某条绳不要 | 直接写进负面，可加权：`(crotch rope:1.1)` |

**负面也要瘦身**：60+ 词的负面实测会过载（画面海报化），20 词左右才听话。
**正向同样不许重复**：一个意图只说一次，想排除就只在负面写一遍（正向写 `no shoes` 等于白写）；
同一要素别堆同义词（`long hair, very long hair, flowing long hair`）—— 长度本身就是稀释，
多写的词会把你真正想要的那几个 tag 摊薄（`references/prompt_writing.md` §2b）。

---

## 3. 骨架模板（照抄后替换角色块）

### T1 · IL 单人 · 四段式（**最常用**）— 源 `wf/004.json`

```
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest, highly aesthetic, intricate details, professional artwork, trending on ArtStation,
digital illustration, painterly style, soft gradient shading, airbrush rendering, professional digital painting,extreme details,

indoor, wooden floor,

1girl, standing, push opening a door,
<LoRA 触发词>,<发色瞳色发型>, <服装配色部件>

NEG: <aesthetic 块>, people, person, girl,
```

改角色只改最后一块；改场景只改第三块。

### T2 · IL 单人 · 行级要素（要素多时用）— 源 `wf/ayaka_rope (1).json`

```
facing to viewer, view from front,

1girl sitting on floor, wariza,W-legs,
arms bondaged behind her back, red rope,
pantyhose pull down on knee, wet thighs,
egg vibrator controller bind on her thigh with tape, vibrator cord from her crotch,
legs bondaged,
shy,black blindfold, orgasm face mouth gag, ball gag,
ayaka \(springbloom missive\), light blue hair,butterfly hair ornament,hair flower,brown headwear,braid,hair up, mole under eye,blue eyes, white collar, blue dress, long sleeves, white pantyhose,high heel boots, blue butterfly ring,

;
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest, highly aesthetic, intricate details, professional artwork, trending on ArtStation, digital illustration, painterly style, soft gradient shading, dramatic lighting, volumetric light, extreme details,
```

要点：**姿势 / 缚法 / 道具 / 表情各占一行**，角色块压到最后两行，风格块放末尾（用 `;` 隔开）。

### T3 · IL 双人 · `{}` 块 + `;`（最不容易换人）— 源 `wf/belle_vvn_handup (1).json`

```
indoor, black sofa,
2girl,yuri,
;
first girl:{
1girl sitting on sofa, left leg raise up, vertical split, left foot hang up, red rope, arms bondaged behind back,
wet crotch,
mouth gag, tape gag, blush, shy, orgasm face,

<角色A触发词 \(系列\)>,
crossed bangs,hairclip,ahoge,blue hair, green eyes,
bare body, breast, bare neck, nipples,
white pantyhose, single thigh strap,
}
;

second girl:{
1girl sitting beside, left hand on first girl waist, right hand on crotch, fingering, kissing on first girl cheek,
<角色B触发词 \(系列\)>,purple hair,multicolored hair,red eyes,pointy ears,long hair,drill hair,hairband,white dress,white shirt,black bow,black pantyhose,
}
;
masterpiece, best quality, absurdres, ultra detailed, high resolution, newest, highly aesthetic, intricate details, professional artwork, trending on ArtStation, digital illustration, painterly style, soft gradient shading, dramatic lighting, volumetric light, extreme details,
```

要点：**每块的"该角色自己的动作"必须写全**，否则两人做同一件事。

> ⚠️ **`[measured]`（本机 IL，2 个种子）**：`2girl` + 两个 `1girl` 块**不足以保证**第二个角色是女性 ——
> 叠压构图（`straddling` / `girl on top`）下 2 个种子里有 1 个把下方角色画成男性
> （WD14 读出 `1boy, hetero, clothed female nude male, girl on top`）。做法：必写 `2girl, yuri`
> **且**每个块里各自写 `1girl` + 外观，**逐种子验证**（加 `yuri` 修好了另一个种子，失败的那个仍失败）；
> 仍出男性就往负面加 `1boy, male, penis, hetero`（待验证）。
> 若不需要叠压关系，用 corpus 那种更安全的并列构图（`sitting side by side`, `kissing`）。

### T4 · IL 双人 · 关系标签（更短）— 源 `wf/xtll_lumi (5).json`

```
bathroom, full body,
2girl, yuri, kissing,

sub girl, dominatied girl,standing,
red rope ,arms bondaged behind her back,
crotch rope, arch back,
blush, shy, struggling,

<角色A \(系列\)>, long hair, braid, blue eyes, pink hair, bare shoulders, twin braids, hair ornament, facial mark, hair intakes, multicolored hair, headgear,
naked, bare body, breast, nipples,

other girl, dominator,
standing beside, bondaged,
<角色B \(系列\)>, naked, smile,
```

标签词：`sub girl` / `dominator` / `dominatied girl`、`front girl` / `other girl`、
`left girl` / `right girl`、`girl below` / `girl above`、`first girl:` / `girl A:`
（上下叠压构图**必须**用 `below`/`above`，否则角色互换）。

### T5 · 人在容器 / 器具里 — 源 `wf/lanyan_case2 (2).json`、`wf/akf_pillory (1).json`

容器（蜷起来塞进箱子）：

```
outdoor,luggage box,

1girl, lying in luggage box, on side, legs together,bent legs,body curl up,
eyes closed, blush, shy, mouth gag, ball gag,black blindfold,
dildo in pussy,tembling, struggling,
red rope, shibari, amrs bondaged behind, legs bondaged, breast rope,

<角色 \(系列\)>, long hair, hair ornament,
naked, bare body,breast, nipples, dildo in pussy, wet crotch,
```

器具（枷、木架等 —— **用原生 tag**，别自己描述结构）：

```
wooden floor, front view, full body,

1girl standing on floor and bent down,
upper body restraint in pillory,pillory,stocks,
blush, shy, looking at viewer, wet thighs,

<角色 \(系列\)>,
green eyes, orange hair, ahoge,long hair,bow, tail, apron only,
bare back, bare crotch, bare legs, bare feet,

;
<风格块>
```

### T6 · 悬吊但**别翻倒** — 源 `wf/nxd_reserve (1).json`

```
white background, simple background,

1girl, solo, full body,
reserve, up side down, hang up,
leg suspension, other leg bent,
arms bondaged behind back, red rope,
blush, shy, struggling,

<角色 \(系列\)>,dress hanging down, white underpanties, small breast,
```

要点：**不要写 `midair`**（实测会变成倒挂）；用 `leg suspension, other leg bent` +
`the rope takes part of her weight` 这类话把体重交代清楚。要"坐/跪在容器底"就写
`knees on the floor of the locker`。

### T7 · Anima 单人（句子 + `@画师` + 权重）— 源 `wf/099A2CA200DC439D27CDB971961C68B0.json`

```
masterpiece, best quality, newest, absurdres, highres, score_9, score_8, nsfw, 1girl,

@houkisei, @rokuwata tomoe, @atdan, @yuuraku yuraku, @[himitsu (hi mi tsu 2):0.8] , @[wlop::0.6], @n4g1style,

<角色名 from 系列>, with animal ears, black pantyhose, blonde hair, hair ornament, heterochromia, long hair, low twintails, rabbit ears, white shirt, sphere hair ornament, dark green skirt, pleated skirt, yellow eyes,

red wiffle gag, shibari, breast bondage, ball gag, bound legs, arms behind back, kneeling on one leg, suspension, leg folded, crotch rope, pussy juice stain, wet panties, orgasm,

She is embarrassed and nervous.

A red wiffle gag with harness and leather straps is secured in her mouth. Her legs are bound together from thighs to ankles by red ropes. Her arms, torso and breasts are bound tightly in a shibari style by red ropes.

She is kneeling on one leg on stage with spotlight, with the other leg suspended by ropes and bound folded, showing her crotch.
```

要点：**逐部位绑**（`bound legs, bound arms, bound wrists, bound thighs, bound ankles, bound knees`）、
**权重 1.5–2.0**、`@画师` 触发词成组出现、句子写"发生了什么"。

### T8 · Anima 绳路段落（可整段复用）— 源 `wf/077B59A5A8C33C090066C4DF8D4D9143.json`

```
A close shot of the character. She is lying on back, with her legs together. She is tied by an extremely large amount of red ropes and is tied a crotch rope. Her legs are tied together firmly by an extremely large amount of red ropes. Her upper body is tied in shibari style by an extremely large amount of red ropes, and her arms are tied behind her back. She is wearing a metal collar, and the viewer is holding a chain leash connecting to her collar. Pussy juice is dripping onto the ground under her, and forms a pussy juice puddle.
```

（原文那条记录本身是**坏例子**：整块重复了两次，还混进了
`<lora:Nicole Reeyn 2680078:0.8>` 内联标签和 `Select the LoRA to add to the text`；
只取这段绳路文字用。）

其他可复用句（逐字，来自语料）：

```
She is tied up by red ropes in a kikkou shibari style.
Her upper body is tied in shibari style by large amounts of red ropes, and her arms are tied behind her back.
Her legs are bound together from thighs to ankles by red ropes.
A ball gag with harness and leather straps is secured in her mouth.
the gag's straps are black, and this strap goes up over the nose, meets at the top, and then goes around the back of the head to form a harness gag.
Her other leg is suspended, raised high above her head.
Her dress is rolled up by the crotch rope, revealing her white panties.
```

---

## 4. 词表速查

### 绳索 / 束缚（按部位逐个写，别只写"绑住"）

| 部位 | tag | 语料频次 |
|---|---|---|
| 全身泛写 | `shibari`, `bondage`, `bound` | 266 / 76 / 27 |
| 材质 | `red rope`, `rope` | 253 / 13 |
| 手臂 | `arms behind back`, `bound arms`, `bound wrists`, `arms bondaged behind back` | 93 / 35 / 6 / 63 |
| 腿脚 | `bound legs`, `bound ankles`, `bound knees`, `bound thighs`, `bound feet`, `legs bondaged` | 79 / 23 / 16 / 15 / 6 / 47 |
| 胸 | `breast bondage`, `breast rope` | 78 / 27 |
| 股 | `crotch rope` | 107 |
| 隔衣 | `shibari over clothes` | 22 |
| 勒痕 | `skindentation` | 15 |
| 吊缚 | `suspension`, `leg suspension`, `hogtie` | 29 / 2 / 14 |
| 拘束衣 | `bitchsuit`, `straitjacket`, `corset`, `spreader bar` | 15 / 13 / 7 / 5 |
| 器物 | `pillory`, `stocks`, `handcuff`, `chain`, `leash`, `collar` | 用原生 tag，见 §3 T5 |

### 整体缚法姿势（14 种实测，Anima 直出 832×1216、2 种子/姿势）

`✅` 一眼认得出、绳路合理 · `⚠️` 方向对但关键细节不到位（**0 个认不出**）
语料：（我们的实验）（提示词、28 张图、联系表）

| 日文名 | 英文 | 该抄的 tag | 结果 |
|---|---|---|---|
| 後手縛り | box tie | `box tie, bound arms, arms behind back, bound wrists, bound elbows` | ✅ 最标准 |
| 海老縛り | hogtie | `hogtie, shrimp tie, bound arms, bound legs, bound ankles, on stomach` | ✅ |
| 逆海老 | gyaku ebi | `gyaku ebi, hogtie, on back, knees to chest` | ✅ |
| 胡座縛り | frogtie | `frogtie, bound legs, bound calves, bound thighs, knees apart` | ✅ |
| 一本吊り | ippon tsuri | `suspension, single rope, bound wrists above head, feet off the ground` | ✅ 立吊要这样写 |
| 後ろ手吊り | strappado | `strappado, arms behind back, arms up behind, bent over` | ✅ |
| 股縄 | crotch rope | `crotch rope, rope between legs, rope harness` | ✅ |
| 並缚 | bound together | `bound together, back to back, ropes joined, 2girls` | ✅ 一次成 |
| 人犬 | human dog | `human dog, pet play, all fours, collar, leash` | ✅ |
| 菱縄縛り | karada | `shibari, rope harness, karada, diamond pattern rope, bound arms` | ⚠️ 菱形编花不出 |
| 亀甲縛り | kikkou | `kikkou, tortoise shell rope pattern, rope harness, arms behind back` | ⚠️ 六角纹不出 |
| 逆拝み | reverse prayer | `reverse prayer, bound arms, palms together` | ⚠️ 合掌不出 |
| 吊り縛り | suspension | `suspension, hanging, suspended, arms above head, feet off the ground` | ⚠️ 出来是**倒吊** |
| 磔 | spread eagle | `spread eagle, arms bound apart, legs bound apart, x pose` | ⚠️ 只摆开、没绑四角 |

**规律（直接复用）**
1. **有公认英文名的姿势最好使**（box tie / hogtie / strappado / frogtie）；日文名单独写没用，英文名才是控制词。
2. **绳的"艺术编花"（菱形 karada、龟甲 kikkou）纯提示词出不来** ⇒ 要参考图＋ControlNet/LLLite，或局部重绘；文字只能保证"大概有绳"。
3. **`suspension` 默认读作倒吊**；要立吊/半吊必须写 `single rope, bound wrists above head, feet off the ground`。
4. **`spread eagle` 只是摆姿势**，要绑在四角得补 `bound to bedposts / attached to corners` 之类。
5. **`bound together` ＋ `2girls` ＋ `back to back` ＋ `ropes joined`** 一次就成 —— 双人并绑类优先用这个组合。

### 口球 / 头部

`ball gag` 122 · `mouth gag` 107 · `harness ball gag` 28 · `gag harness` 12 ·
`ring gag` 22 · `cleave gag` 5 · `wiffle gag` 1 · `blindfold` 32 · `black blindfold` 9 ·
`tape gag` 4 · `white cleave gag`（语料写法，计数 <3）

### 腿部几何

`wariza` 12（鸭子坐）· `M-legs` 16 · `W-legs` 13 · `folded legs` 6 ·
`leg folded`（Anima 写法）· `bent legs` 35 · `legs together` 55 · `spread legs` 14 ·
`legs open` 36 · `vertical split` · `body curl up` · `cross ankles` · `lotus position`

### 状态 / 表情（被动方一定要给一个）

`shy` 231 · `blush` 318 · `struggling` 87 · `flushed face` 64 · `orgasm` 64 ·
`tear in eyes` 25 · `tears` 50 · `saliva drip` 48 · `trembling` 13（语料里也写成 `tembling`）·
`helpless` 3 · `embarrassed` 14 · `angry` · `looking at viewer` 117 · `eyes closed` /
`eyes half closed` / `rolling eyes` 17 · `ahegao`（**本语料 0 次，别指望它**）

### 镜头 / 取景

`full body` 168 · `cowboy shot` · `upper body` · `close-up` · `side view` 42 ·
`from above` 19 · `from behind` · `three-quarter view` · `depth of field` 71 · `bokeh` 47 ·
`centered character` · `white background` 68 · `simple background` 58 · `front view`

### 环境

`indoor` 77 · `bedroom` 50 · `lying on bed` 37 · `wooden floor` 23 · `on bed` 16 ·
`classroom` 6 · `bathroom` 3 · `library` / `dungeon` / `stone wall` 各 1

（数字是**本语料**的 df；环境词在本语料里多数以句子出现，
`locker room`（0）、`bamboo forest`（0）、`stone wall`（1）这类 **tag 计数低 ≠ 不能用**——
它们是 danbooru 原生 tag，只是这批提示词把它们写进了句子或没提。
选环境词时按"这个词在 danbooru 有没有帖"判断，别按本表计数。）

### 风格 LoRA 触发词（`@` 前缀，19% 的语料在用）

`@houkisei` 74 · `@atdan` 73 · `@rokuwata tomoe` 60 · `@m4me` 56 · `@wlop` ·
`@himitsu (hi mi tsu 2)` · `@n4g1style` · `@yuuraku yuraku` · `@BlueArchStyle` ·
`@gpt-image-2`；带权重写 `@[name::0.8]`

---

## 5. 规则速查（if → then）

| 想要 | 怎么做 |
|---|---|
| 换角色 / 换装 | 正向 `X instead of Y` **＋** 负面含 `Y` **＋** 角色块里删掉 `Y` 的 tag（实测：0/4 → 4/4） |
| 两人不互换 | 每角色一块 + `left/right` 或 `below/above` 或 `front/other` 标签 + 各自的动作与状态 |
| **两人都是女性** | `2girl, yuri` + 每块写 `1girl` + 各自外观；**堆叠构图（`straddling`/`girl on top`）逐种子验证**，出男性就往负面加 `1boy, male, penis, hetero`（`[measured]` §3 T3） |
| **器具/木枷/架子要落地** | 器具写**原生 tag**（`pillory, stocks`）进 tag 段，别只写在句子里；句子只做补充（`[measured]`：只写句子时 WD14 读不出任何器物） |
| 单人图别多画人 | 负面加 `people, person, girl` |
| 落地绳不要悬吊 | 负面加 `hanging rope, suspension` |
| 绳子别变缎带 | 负面加 `ribbon, satin ribbon` |
| 不要裸露 | 负面加 `nsfw, nude, open mouth`；正向别写 `naked` |
| 精确的像素级几何（布料抬多高、只露一条边） | **别写词**，去做遮罩 / Blender ControlNet |
| 绳路拓扑（什么连着哪、绕几圈） | **用句子写**，比 tag 堆更清楚 |
| 角色/器具/服装的正确 tag | 先查（`tagcheck.py` / `dic.pixiv.net/a/<tag>`），别硬描述结构 |
| 不知道某件衣服叫什么 | 查一个近似 tag，读它的 post 的 tag 串，"组成 tag"就在里面 |
| 多人里的某条绳不要 | 负面写它，可加权 `(crotch rope:1.1)` |
| 要"同一个系列" | 固定风格前缀 + 固定种子基 + 只换角色块 |

---

## 6. 参数速查

| 项 | IL / SDXL | Anima (qwen-image) |
|---|---|---|
| 采样器 | `dpmpp_2m` + `karras` | `euler` + `simple`（或 `er_sde`+`simple`） |
| 步数 | 本管线 **28**；语料中位 22 | **30**（turbo LoRA 时 12） |
| CFG | **6.5**；语料中位 8.0 | **8**；语料中位 6.0（1.0–8.0，turbo LoRA 会到 1） |
| 权重幅度 | 1.1–1.3（几乎不超过 1.5） | 1.5–2.0（可到 4.0） |
| 画布 | 832×1216 / 1200×1600 / 1200×1200 | 832×1216 / 1024×1536 |
| 分辨率本身是取景开关 | 见 `references/local_repaint_bigcanvas.md` | 同 |

LoRA：`LoraLoader` 节点给强度（`strength_model` / `strength_clip`），
**不要**把 `<lora:name:0.8>` 写进提示词文本（语料里有 15 条这么干，我们不这么用）。

---

## 7. 提交前 10 秒检查

1. 空行分区块、要素各占一行？
2. 风格块只有一次，且是**对应底模**的那一块？
3. 正向里没有 `no / without / nothing / except / only`？
4. 要排除的东西在负面里（原服装、不想要的读法、多余的人、不想要的材质）？负面 ≤ ~30 tag？
5. 绳索写明**部位**了？被动方有状态词？
6. 多人：数量 + 每角色块 + 位置/关系标签 + 各自状态 + 负面钉人数？
7. 权重：`(x:w)` 只在需要时用，一次只调一个，幅度按底模？
8. 提示词文本首尾没有插件 UI 字符串（`Select the Wildcard to add to the text`、`reproduce`、
   `否 (false), 两者 (both)`）或 token JSON？
9. 加一个词没让画面可辨地改变 → 删掉。**先减词，再调权重。**
10. 重绘？`denoise` 按这张画布重新标定（`references/local_repaint_bigcanvas.md`）。

---

## 8. 反面样例（语料里真实存在的坑）

- **UI 字符串泄漏**：`Select the Wildcard to add to the text` / `Select the LoRA to add to the text` /
  `reproduce` —— 语料 29 条（26% 的"他人下载"组）含这些，纯噪声。
- **正面里写否定**：`Her legs aren't bound.`（`wf/ComfyUI_00012_.json`）——
  文本编码器不会把 `aren't` 绑到 `legs` 上。
- **整块重复**：同一段绳路文字在一条提示词里出现两次（`wf/077B59A5….json`），
  不增加控制力，只稀释。
- **token JSON 泄漏**：一条记录里塞进插件整个 token 数组，单条 17,965 字符。
- **超长负面**：60+ 词实测会过载、画面海报化。
- **拼错的 tag**：`ctorch rope`、`amrs bondaged behind back`、`tembling` ——
  靠旁边正确词救场；**优先规范拼写**。

---

## 9. 从哪来 / 怎么复现

- 所有"频次"= document frequency，全表数字取自 `analysis/tag_df.csv`
  （`analyze_corpus.py` 生成，与 `corpus_stats.md` 同源），语料 536 条有提示词的记录。
- 统计脚本：（我们的实验）
  的 `analyze_corpus.py` / `analyze_extra.py` / `compare_groups.py`；
  单 tag 查值：`check_cheatsheet_tags.py`。
- **频次只是惯例，不是效果**；带"实测"的条目来自我们自己的对照出图，
  两者冲突时以实测为准（见 `references/prompt_writing.md` 开头）。
