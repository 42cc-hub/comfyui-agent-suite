# 出图安全闸门：看图是受控动作

**风险模型**：agent 用视觉工具看图 = 把这张图上传给模型服务商；画面命中敏感元素 ⇒ 账号风险。
⇒"看不看"是**受控动作**，不是顺手动作。

## 规则一 · 先过本地闸门，再决定看不看

```bash
python libs/wd14-gate/scripts/wd14_gate.py out/*.png            # 语义版（首选）
python libs/wd14-gate/scripts/wd14_gate.py out/*.png --tags     # 附带全部标签，便于人工判读
python libs/wd14-gate/scripts/nsfw_gate.py out/*.png            # 像素统计版（粗筛，阈值已收紧）
```

| 判定 | 含义 | 动作 |
|---|---|---|
| `OK` (general) | 安全 | 可以看 |
| `REVIEW` (sensitive / **questionable 但无关键点** / **explicit 但无关键点（escalated）** / **仅命中姿势表（pose-level）** / 命中关注标签) | 可能擦边 | 标签温和（`bare legs`、`bare shoulders`、`cleavage`、`navel`）时可看 |
| `BLOCK` (**命中关键点**；explicit **且**命中关键点或姿势表) | 不可看 | **绝对不看**，走规则二 |

- **硬边界是关键点，不是 WD14 整体 rating。** 无关键点的 `questionable` 归 REVIEW（可看，留痕）；
  `explicit` 同样：**命中关键点才 BLOCK**，否则降 REVIEW 并加一行
  `escalated: rating=explicit but no key-point tags`（人工区分"题材过杀 vs 真露点"）。**风险由用户承担。**
- **姿势不是关键点。** `spread legs` 属姿势表：衣着完整时单独出现 ⇒ REVIEW + `pose-level: spread legs`；
  与关键点同现或 `rating=explicit` ⇒ BLOCK。
- **真关键点不等 explicit 才拦**：`cameltoe` / `partially visible vulva` / `vulva` / `labia` /
  `visible clitoris` 直接进 `BLOCK_TAGS`。

**关键点表**（命中即 BLOCK，即 `BLOCK_TAGS`）：`nipples`、`puffy nipples`、`pussy`、`penis`、
`anus`、`bare crotch`、`crotch`、`completely nude`、`nude`、`topless`、`bottomless`、
`presenting`、`large areolae`、`areolae`、`clitoris`、`testicles`、`sex`、`cum`、
`cameltoe`、`partially visible vulva`、`vulva`、`labia`、`visible clitoris`。

**姿势表**（`POSE_TAGS`，**不是**关键点）：`spread legs`。

**explicit 复核扩展表**：`NUDITY_TAGS = BLOCK_TAGS | {bare butt, exposed butt, no panties,
exposed crotch, exposed breasts, large breasts, huge breasts}`。`explicit` ＋命中 `NUDITY_TAGS`
或 `POSE_TAGS` ⇒ BLOCK；两者都不命中 ⇒ REVIEW ＋ `escalated`。

- **"皮肤占比"不能当主判据**：隔着布料的敏感部位颜色被布料改写 ⇒ 像素规则抓不到。
  单次观察（半透明白裙单图，n=1）：肤色仅 9%，像素判定 SAFE，WD14 一次给 `questionable`。
  故 `nsfw_gate.py` 只作粗筛。
- 已知阈值副作用（单次观察，n=1）：`cameltoe` 置信度落在 0.30–0.35 ⇒ `--threshold 0.35` 判 REVIEW、
  `0.30` 判 BLOCK。`cameltoe` 是否也改条件式：**待用户决定**。

## 规则二 · 人工审核通道 = 只给路径，不嵌入

> **命中 BLOCK 的图：既不要 `read_image`，也不要 `dsh_show_media`。**
> `dsh_show_media` 会把图块带进 agent 上下文（实测，与其工具说明相反），不能当"安全通道"。
> 用 **`dsh_show_gallery`**：只把句柄放进结果，图由浏览器自取，**模型上下文增量为零** ⇒ 唯一合规审阅通道。
> 面板没装时退回：回复里只给**文件路径**（纯文本），人在本地打开。
> 命中 REVIEW：仅当关注标签温和（`bare legs`、`navel`、`collarbone`）时才可看，否则走上面两条通道。

用法：`dsh_show_gallery(folder="<目录>", pattern="*.png", sort="name-desc", limit=100)`，或 `paths=[...]`。
代价是人多一步操作，换来账号风险为零。

## 规则三 · 边界针对关键点，不针对裸露程度

- 挡的是关键点（`nipple` / `pussy` 这类）；**不挡裸露本身**——比基尼、及膝短裙、露腿露肩都正常。
- 绳缚元素一般问题不大（rating 多落 `sensitive` ⇒ REVIEW，可看）。
- ⇒ 负面词要**点对点**，不要写一片保守词把画面压成粽子。

## 规则四 · 让图天生安全的两招

1. **把角色 LoRA 的默认服装写进负面**。作者通常在模型页给这条技巧；
   单次观察（同一构图，n=1）：把 `detached sleeves, black skirt, thighhighs, corset` 全写进负面后，
   rating 从 `questionable` → `general`。**这是单次观察，不是保证。**
2. **正向写"厚实、不透"**：`modest nightgown, long sleeves, opaque fabric, fully covered`；
   配"及膝短裙"级别的适度即可，不必长裙。

## 打码能换来"可看"吗

四块同级面板 × 6 种变体共 24 张、一次闸门调用（单次观察，条件：同一组面板，n=24 张）：

| 源 | 原图 | mos16 | mos32 | mos64 | 纯色填平 | 挖空中间 50% |
|---|---|---|---|---|---|---|
| BLOCK 面板 A | BLOCK explicit | REVIEW | REVIEW | REVIEW | REVIEW | REVIEW（仍带 `exhibitionism, public indecency`） |
| REVIEW 面板 B | REVIEW | **OK** | OK | OK | OK | REVIEW |
| BLOCK 面板 C | BLOCK explicit | **OK** | OK | OK | REVIEW | **仍 BLOCK explicit**，标签几乎不变 |
| REVIEW 面板 D | REVIEW | OK | OK | OK | REVIEW | REVIEW |

按可用性排序的结论：

1. **整块面板打马赛克有效且廉价**：16px 网格足以把 BLOCK 翻成 REVIEW，32/64px 常到 `OK`。
   「看不到」不是问题——**找"露骨的那一小块"才是问题**。
2. **判定依赖上下文，不只看被涂掉的像素。** 面板 C 挖空中间 50% 仍 `explicit`，标签
   （`vibrator under clothes`、`vibrator cord`、`remote control vibrator`）原样保留——判定从**洞外可见的结构**
   推出；面板 A 的 `exhibitionism / public indecency` 是**场景级**推断。
   ⇒ **单块裁片的判定 ≠ 整图的判定**：裁片单独 REVIEW 的那块放回整图可让整图变 `explicit`，
   所以按格子扫"露骨点"会系统性漏判。
3. **纯色填平有时比马赛克更糟**（C/D：flat=REVIEW，马赛克=OK）：平整灰块像"打过码的条"，反而更可疑。
4. **场景级标签不会挡**：`bdsm / bondage / shibari / exhibitionism / public indecency` 打码后仍在，也不触发
   BLOCK（BLOCK 只看 `explicit` 与关键点标签）⇒ 别追着它们改。
5. **遮住"让这张图还算正常的东西"会反噬：REVIEW → BLOCK。** 单次观察（同一张内衣立绘 832×1216，
   骨骼定位后处理胸+胯整段，n=1 图 × 3 种处理）：

   | 处理方式 | 判定 | rating | 关键标签 |
   |---|---|---|---|
   | 原图（未处理） | `REVIEW` | `sensitive` | `underwear, panties, white bra, cleavage, navel, breasts` |
   | 矩形 · 32px 马赛克 | `BLOCK` | `questionable` | `nude, completely nude, mosaic censoring`；衣着标签**全消失** |
   | 矩形 · 纯黑 / 灰 / 白平涂 | `BLOCK` | `questionable` | `nude, bar censor` |

   矩形/马赛克盖的是**整段**（肩→髋、髋→+0.22 躯干高），必然包含内衣本身 ⇒
   **遮盖抹掉"穿着"的证据，分类器补出 `nude` ⇒ BLOCK；换颜色只改 censor 标签名，救不回来。**
   与上面那张 BLOCK→REVIEW 的表不矛盾：那里盖掉的是**露骨内容本身**（有效），这里盖掉的是**让图合规的衣物**（反噬）。
   **判据：你要盖的是"过敏源"还是"遮羞布"？前者盖，后者别盖。**

> ⚠️ **已撤回的结论（别再用）**：曾据"在**已有内衣**的图上画内衣形状（梯形+倒三角）仍判 REVIEW"写成
> "形状能保住判定"。该实验设计与归因都不成立（底图本来有内衣，画上去的形状比矩形小、也没验证是否真盖住那件内衣）。
> 补做的决定性实验：同一套形状画在**全裸**底图上——白 / 黑 / 皮肤色 `#E8BEAC` 三种填色**全部 BLOCK**
> （白色那版连 `nipples` 都还在），与黑色矩形全覆盖一致（单次观察，n=1 图 × 3 填色）。
> ⇒ **画"衣物形状"不能让裸图过关；判定标签也不能当"遮盖程度"的度量。**
> 工程结论：要遮就**遮全（矩形全覆盖最稳）**，形状会留缝；`pose_censor.py --garment …` 只作示意/排版工具，
> **不要**当"骗过闸门"的手段。

操作规则：

- **定位要打码的地方时，只认真正的标记**：`rating=explicit` 加 key-point / 体液类标签
  （`pussy juice`、`nipples`、`pussy`、`nude`、`spread legs`、`vaginal object insertion`、`areola` …）。
  **位置/场景类描述词不是标记**——`crotch rope`、`shibari`、`bondage`、`bdsm`、`exhibitionism`、
  `public indecency`、`sex toy`/`vibrator` 本身都不触发 BLOCK。把它们算进"要糊的地方"会**过判**：
  单次观察（n=1）把 `crotch` 列入条件后单格被标记、连带 3×3 邻域一起糊，整图糊掉 49.9%，图也就没用了。
- 要**看**一张 BLOCK 参考图：按**你要用的粒度**（整图 / 整块面板）打马赛克，网格 16–64px，
  **不要纯色填**，也不要只涂"可疑的那一小块"。
- **闸门必须在你要动手的粒度上跑**：要切面板就在面板粒度判定，别拿裁片判定去推断整图。
- **合成图（拼图、贴人物后的整图）按整图重新过闸门**——四块都 REVIEW 的拼图整体仍可能 BLOCK。

脚本：（我们的实验）（本实验矩阵）、
同目录 `censor_panels.py`（分块打码 + 单次闸门）、`auto_censor.py`（按格子扫描——**反例，勿用**：
判定依赖上下文，逐格扫必然漏判）。

---

## 反例 · 本来能看的图，平涂打码后反而 BLOCK

同一构图两个文件（588×721，同一角色同一张插画，差异 13.0% 且集中在两块矩形 —— 是同一张图的局部编辑）：
`161.png` 角色**穿**白色比基尼上衣＋半透明纱笼 ⇒ `sensitive` / REVIEW；
`161 - 副本.png` 把衣服擦掉换成两块平涂灰色矩形（胸 y234–321、裆 y375–471）⇒ `questionable` + `nude` + `censored` ⇒ **BLOCK**。

1. **衣着是强 `safe` 先验**：抹掉衣服换中性色块 = **移除**该先验，分类器只能读出裸露解剖结构。
   **"打码"不是安全操作，它可能把可看的图变成不可看的图。**
2. **BLOCK 是"某个文件"的属性，不是"某张画"的属性**：换版本（重存、裁切、修图）重过闸门是正当的，且可双向翻转。
3. **不要把"我打了码"当成把 `nude` 图带回上下文的通道**：分类器对平涂打码版可能判得更重。
4. **文件属于人、人已看过时，由人拍板**：闸门保护 agent 的上传预算与账号，不是覆盖所有者的判断；
   授权来自**所有者**，不来自"它看起来无害"的自我判断。

## 规则五 · BLOCK 的处置 = 人工打码审核通道

BLOCK 不是对这张画的判词，而是"**agent 自己不许看**"的保护动作。**多模态任务里人的反馈是最高判据**：
文件所有者本人看过、并指定盖哪里之后，agent 看那张**打码产物**是正当的，**不需要**闸门认可
（判定不因此改变：打码后可能仍 BLOCK，甚至更差）。

**先问一句：需要 agent 自己看吗？** 只是要人过目 ⇒ 用 `dsh_show_gallery`（规则二），**不必打码**。
打码是"agent 自己要看"时的手段，不是默认手段。

流程（`libs/wd14-gate/scripts/gate_review.py` **不改判定**，只把 BLOCK 之后的处置变成可执行步骤）：

```
python libs/wd14-gate/scripts/gate_review.py IMG...                # 判定 + （若有 BLOCK）打印处置流程
python libs/wd14-gate/scripts/gate_review.py IMG... --tags         # 附带全部标签便于人判读
python libs/wd14-gate/scripts/gate_review.py IMG... --sheet s.png  # BLOCK 多张 -> 拼带文件名标注的联系表
```

1. **agent 不得自己决定盖哪里** —— 把图交给所有者，用 `human_canvas` 涂两层：
   `L1`「要打码 / 不给 agent 看」、`L2`「要保留」（保护层，mask = `L1 − L2`）。
   多张时先在**缩略图粒度**拼联系表（`--sheet`），人一次涂完；缩略图上涂要把 `--dilate/--feather` 换算回原尺寸。
2. 读回各层 → `canvas_mask.py`（`--rope L1.png --keep L2.png --src IMG --out mask.png --overlay o.png`）
   合成遮罩 → **在副本上**打码，**绝不覆盖原文件**：
   `python scripts/tools/mosaic_censor.py --src IMG --out COPY.png --mask mask.png --cell 32`
   （掩版非零即打码；**掩版为空、或掩版外像素有任何变化 ⇒ 脚本直接拒绝出图**）。
3. **默认马赛克（16–64px 网格），不要平涂**：平涂灰块会把可看的图判成 BLOCK，马赛克常能改善判定。
4. 对打码产物**重新过闸门**并记录：改善 / 不变 / 更差都写进授权记录（谁、何时、涂了哪里、之后判定），随产物留档。
5. **授权只能来自文件所有者本人。** "我觉得无害""它看起来正常"都不是授权；也不能用"我先自己打码"给自己开路。

### 规则五·b · 无人可提供掩版时的自动通道（仅全自动模式）

前提两条缺一不可：**用户已授权自动跑**（见 `modes_and_discipline.md`），**且确实没有人工能涂掩版**。
用骨骼关键点自动框两个区域再打码：

```
python "<套件>\scripts\locate.py" IMG --out pose.json   # 检测套件出 COCO-17 关键点（套件路径见 prompt_writing.md 的套件表）
python scripts/tools/pose_censor.py --src IMG --pose pose.json --out COPY.png --cell 32
python libs/wd14-gate/scripts/wd14_gate.py COPY.png                  # 必做：对副本重跑闸门
```

- 自动框两区：**胸部**（肩 5/6 → 髋 11/12 之间的躯干带）、**胯部**（髋线向下约 1/3 躯干高）；
  `--margin` 按框尺寸外扩，默认 0.15。
- **硬约束**（脚本已实现，违反即拒绝出图）：关键点缺失/不可见 ⇒ 拒绝；覆盖 <0.5% 或 >60% ⇒ 拒绝；
  `--out` 等于 `--src` ⇒ 拒绝；掩版外像素非逐位不变 ⇒ 拒绝。
- **放行判据 = 关键点词汇是否消失**（`python scripts/tools/gate_after_censor.py --src IMG --censored COPY.png`）：

  | 原图 → 打码副本 | 判定 |
  |---|---|
  | 原图有关键点词汇（`nipples / pussy / penis / anus / vulva / clitoris / labia / cameltoe / sex / cum` …），打码后**全部消失** | **PASS** —— 状态词（`nude / completely nude / topless / crotch`）与 `censored / mosaic censoring` 残留**不算失败** |
  | 关键点词汇**还有残留** | **FAIL**（没盖住，多半是框偏了） |
  | 打码后**新出现**关键点词汇 | **FAIL**（遮出了新问题，例如平涂把能看的图变糟） |
  | 原图本无关键点词汇 | PASS（本来就不需要遮） |

  退出码 `0 / 3 / 4`；`--report` 把两侧标签与判定一起存档。
- **为什么不用"判定变 REVIEW/OK"当判据**：WD14 **感知不到"打码"这件事本身**，只会从上下文继续推断 `nude`
  ⇒ 拿"必须变干净"当判据等于**所有打码都过不了**。真正决定"东西露没露"的是**关键点词汇**。
- **残余风险（明确接受）**：骨骼定位不是 100% 可靠，个别关键点可能没被框住 ⇒ 本通道**必须**经
  `gate_after_censor.py` 复核、**必须**留痕（模式/路线/框/两侧标签/判定）；不能因审核模型不完美就一律拒绝。
- **必须留痕**：模式（自动）、路线（pose-chest-crotch / garment）、用了哪些框、打码前后判定，写进产物旁记录。
  自动通道是"agent 自己给自己开路"的**唯一例外**，留痕是它的前提条件；**半自动模式下不许用**。

### mask 自查的两个陷阱

- **不要打开"整图叠色"的 mask overlay 自查**：那类图是**整幅原图**加半透明色，看它等于把**未打码原图**
  送进上下文，正好绕过这套机制。允许的自查只有两种：① **数字**（面积 %、bbox、连通块形状/数量）；
  ② **只裁 mask 那一小块**再裁图看（`crop_view.py`），绝不整图叠色。
- **打码脚本要按 mask 回贴、且保持画布尺寸**：`out = base`；只在 bbox 内 `crop → 缩 → NEAREST 放大`
  再按 mask `paste` 回去。验证口径是 `mask==0` 的像素**逐位不变**（max diff 0）——那是"只动了该动的地方"的唯一硬证据。

规则二与规则五不冲突：规则二管"未经授权时默认只看路径"，规则五管"人授权 + 打码后怎么把图安全带回上下文"。
合起来 = **默认不给看；人授权＋打码之后可以看**。

## 相关脚本

| 脚本 | 作用 |
|---|---|
| `libs/wd14-gate/scripts/wd14_gate.py` | 语义闸门（本地 WD14 → 标签 + rating → OK/REVIEW/BLOCK） |
| `libs/wd14-gate/scripts/nsfw_gate.py` | 像素统计粗筛（肤色占比 + 最大连通块） |
| `libs/wd14-gate/scripts/gate_review.py` | 闸门 + BLOCK 的人工审核通道（打印 human_canvas / canvas_mask 处置流程，`--sheet` 拼联系表） |

模型与标签表：`<你的 ComfyUI 安装目录>\custom_nodes\comfyui-wd14-tagger\models\wd-v1-4-convnextv2-tagger-v2.{onnx,csv}`
