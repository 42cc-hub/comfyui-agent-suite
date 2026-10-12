# 出图安全闸门：读图前自检 + 人工审核通道

**风险模型**：agent 用视觉工具看图 = **把这张图上传到模型服务商**。画面一旦命中敏感元素，
账号有被封的风险（用户 2026-09-24 明确要求）。所以"看不看"必须是一个**受控动作**，不是顺手动作。

## 规则一：先过本地闸门，再决定看不看

```bash
python libs/wd14-gate/scripts/wd14_gate.py out/*.png            # 语义版（首选）
python libs/wd14-gate/scripts/wd14_gate.py out/*.png --tags     # 附带全部标签，便于人工判读
python libs/wd14-gate/scripts/nsfw_gate.py out/*.png            # 像素统计版（粗筛，阈值已收紧）
```

`wd14_gate.py` 用**本地** WD14 tagger（ComfyUI 那个 custom node 里已装的 onnx ＋ csv，
预处理与官方节点一致），给出 booru 标签与 **rating**：

| 判定 | 含义 | 我该做什么 |
|---|---|---|
| `OK` (general) | 安全 | 可以直接看 ✓ |
| `REVIEW` (sensitive / **questionable 但无关键点** / **explicit 但无关键点（escalated）** / **仅命中姿势表（pose-level）** / 命中关注标签) | 可能擦边 | 标签温和（`bare legs`、`bare shoulders`、`cleavage`、`navel` 这类）时可看 ✓ |
| `BLOCK` (**命中关键点**；explicit **且**命中关键点或姿势表) | 不可看 | **绝对不看**，走规则二 |

> **2026-09-24 更新（用户同意）**：硬边界是**关键点**，不是 WD14 的整体 rating。
> 实测一圈"绳缚＋强迫＋居家薄衣"的图全被判 `questionable`，但**关键点标签一个都没有**、
> 人眼看也正常 ⇒ 分类器对 BDSM 语境偏保守是已知特性 ✗。
> 故 `questionable` 不再一刀切 BLOCK；无关键点的 `questionable` 归 REVIEW（可看，仍留痕）。

> **2026-10-07 更新（用户同意）**：同一逻辑延伸到 `explicit`。证据：连测 9 张**纯面部特写**
> （blush / shy / rolling eyes / angry＋咬牙 / **orgasm face(ahegao)** / torogao / empty eyes /
> crying / naughty tongue），rating 最高只到 `questionable`，**没有任何一张到 explicit**；
> 反例是"旗袍 ＋ crotch rope ＋ 用皮带绑在大腿内侧的按摩棒"那张：rating=explicit，
> **却连一个裸露标签都没有**（tags 只有 `crotch rope / vibrator under clothes / hitachi magic wand`
> 一类题材标签）。⇒ rating 这道闸拦的是**题材**，不是关键点。
> 所以 `explicit` 不再一刀切 BLOCK：**命中关键点才 BLOCK**，否则降级 REVIEW，并在输出里多打一行
> `escalated: rating=explicit but no key-point tags` —— 人工过目时一眼可辨"题材过杀 vs 真露点"。
> **风险仍由用户承担** ✓。

> **2026-10-07 第二次更新（用户同意）——姿势不是关键点**：四组姿势实测（横一字马 / 纵向一字马 /
> 扎马步＋分腿杆 / M 字开脚，全部**衣着完整**）：
> · 横一字马那张 rating **只有 `sensitive`**，却因一个 `spread legs` 标签被判 **BLOCK**；
>   用户亲眼复核："没露，尺度甚至比同批其它几张还小" ⇒ **姿势被当成关键点是过杀** ✗。
> · 纵向一字马（`standing split / leg up`）与 M 字开脚（`legs up`，侧视）**都没有**触发 `spread legs`。
> · 分腿杆（`spreader bar`）模型认识：WD14 直接打出 `spreader bar`，画面也是标准横杆＋脚镣形态 ✓。
> 改动：`spread legs` 从 `BLOCK_TAGS` **移出**，进入新的**姿势表 `POSE_TAGS`** ——
> 单独出现（衣着完整）时归 REVIEW 并在输出里标 `pose-level: spread legs`；
> **只有与关键点同现、或 rating=explicit 时**才升级 BLOCK。
> 同时把 `cameltoe / partially visible vulva / vulva / labia / visible clitoris` 从"靠 rating 兜底"
> **提升为硬 BLOCK**（它们是真关键点，不该只在 explicit 时才拦）。
> ⚠️ 注意阈值副作用：纵向一字马那张的 `cameltoe` 置信度落在 0.30～0.35 之间 ——
> 默认 `--threshold 0.35` 时它是 REVIEW，一旦用 `--threshold 0.30` 就会因 `cameltoe` 变 BLOCK。
> 该图人眼复核同样"没露"，所以 `cameltoe` 是否也该降为条件式（像 `spread legs`），留待用户决定。

**关键点表**（命中即 BLOCK，即 `BLOCK_TAGS`）：`nipples`、`puffy nipples`、`pussy`、`penis`、
`anus`、`bare crotch`、`crotch`、`completely nude`、`nude`、`topless`、`bottomless`、
`presenting`、`large areolae`、`areolae`、`clitoris`、`testicles`、`sex`、`cum`、
`cameltoe`、`partially visible vulva`、`vulva`、`labia`、`visible clitoris`。

**姿势表**（`POSE_TAGS`，**不是**关键点）：`spread legs`。衣着完整时单独出现 ⇒ REVIEW ＋
`pose-level` 标记；与关键点同现或 rating=explicit ⇒ BLOCK。

另有一张用于 explicit 场景复核的扩展表
`NUDITY_TAGS = BLOCK_TAGS | {bare butt, exposed butt, no panties, exposed crotch, exposed breasts,
large breasts, huge breasts}`。逻辑：`explicit` ＋命中 `NUDITY_TAGS` 或 `POSE_TAGS` ⇒ BLOCK；
`explicit` ＋两者都不命中 ⇒ REVIEW ＋ `escalated`。

**为什么不用"皮肤占比"当主判据**：隔着布料的敏感部位颜色会被布料改变 ⇒ 像素规则抓不到 ✗。
实测一张"半透明白裙"只有 9% 肤色、被判 SAFE，而 WD14 一次就给出 `questionable` ✓
（见 `nsfw_gate.py` 保留作粗筛的缘由）。

## 规则二：人工审核通道 = **只给路径**，不要嵌入 ✗

⚠️ **实测更正（2026-09-24）**：原本按工具说明（"This is for the human, not the model"）推断
`dsh_show_media` 不会把图带进 agent 上下文 —— **实际会** ✗：把一张 BLOCK 级的图用
`dsh_show_media` 贴出来后，**工具结果把图片送进了 agent 上下文**（agent 确实"看到"了它）。
所以 **`dsh_show_media` 不能当"安全的人工审核通道"** ✓。

正确做法：

> **命中 BLOCK 的图：既不要 `read_image`，也不要 `dsh_show_media`。**
> 只在回复里**给文件路径**（纯文本），由人**自己在本地打开**判断 ✓✓ —— 这才是真正不进模型上下文的通道。
> 命中 REVIEW：仅当关注标签温和（`bare legs`、`navel`、`collarbone` 这类）时才可看 ✓；
> 否则同样**只给路径** ✓。

这条的代价是人的操作多一步（自己去开文件 ✓），换来的是**账号风险为零** ✓ —— 值得。

## 规则三：边界针对**关键点**，不针对裸露程度

用户明确的边界：

- **要挡的是关键点**：`nipple` / `pussy` 这类 ✗
- **不挡裸露本身**：比基尼是正常的沙滩服饰 ✓；及膝短裙、露腿、露肩都正常 ✓
- 绳缚元素**一般问题不大** ✓（rating 一般落在 `sensitive`，属 REVIEW，可看）

所以负面词要**点对点**，不要写一大片保守词把画面压成"粽子" ✗。

## 规则四：让图天生安全的两招（实测有效）

1. **角色 LoRA 会拖出它的默认服装** ✗。作者通常会在模型页给出技巧：
   「若默认服装残留，就把它的部件写进负面」——实测把默认战斗服的部件
   （`detached sleeves, black skirt, thighhighs, corset` …）全写进负面后，
   同一构图 rating 从 `questionable` → `general` ✓✓。
2. **正向写"厚实、不透"**：`modest nightgown, long sleeves, opaque fabric, fully covered` ✓
   （再配合"及膝短裙"级别的适度即可，不必长裙 ✗）。

## 相关脚本

| 脚本 | 作用 |
|---|---|
| `libs/wd14-gate/scripts/wd14_gate.py` | 语义闸门（本地 WD14 → 标签 + rating → OK/REVIEW/BLOCK） |
| `libs/wd14-gate/scripts/nsfw_gate.py` | 像素统计粗筛（肤色占比 + 最大连通块） |
| `libs/wd14-gate/scripts/gate_review.py` | 闸门 + **BLOCK 的人工审核通道**（打印 human_canvas / canvas_mask 处置流程，`--sheet` 拼联系表） |

模型与标签表位置：`<你的 ComfyUI 安装目录>\custom_nodes\comfyui-wd14-tagger\models\wd-v1-4-convnextv2-tagger-v2.{onnx,csv}`

---

## 打码能不能换来"可看"？实测（2026-09-29）

对四块同级面板做 6 种变体、共 24 张、一次闸门调用（`[measured]`）：

| 源 | 原图 | mos16 | mos32 | mos64 | 纯色填平 | 挖空中间 50% |
|---|---|---|---|---|---|---|
| BLOCK 面板 A | BLOCK explicit | REVIEW | REVIEW | REVIEW | REVIEW | REVIEW（仍带 `exhibitionism, public indecency`） |
| REVIEW 面板 B | REVIEW | **OK** | OK | OK | OK | REVIEW |
| BLOCK 面板 C | BLOCK explicit | **OK** | OK | OK | REVIEW | **仍 BLOCK explicit**，标签几乎不变 |
| REVIEW 面板 D | REVIEW | OK | OK | OK | REVIEW | REVIEW |

结论，按可用性排序：

1. **整块面板打马赛克是有效且廉价的**：16px 网格就足以把 BLOCK 翻成 REVIEW，32/64px 常常到 `OK`。
   「看不到」不是问题——**找"露骨的那一小块"才是问题**。
2. **判定依赖上下文，不只看被涂掉的像素。** 面板 C 把中间 50% 挖空填平后仍然 `explicit`，标签
   （`vibrator under clothes`、`vibrator cord`、`remote control vibrator`）几乎原样保留——判定是从
   **洞外还看得见的结构**推出来的；面板 A 挖空后 `exhibitionism / public indecency` 依然在，那是
   **场景级**推断（夜里被绑的少女），根本不是某个像素。
   ⇒ 推论：**单块裁片的判定 ≠ 整图的判定**。裁片单独过闸门时是 REVIEW 的那块，放回整图上下文里可以
   让整图变 `explicit`，所以按格子扫"露骨点"会系统性漏判。
3. **纯色填平有时比马赛克更糟**（C/D 的 flat = REVIEW，马赛克 = OK）：平整灰块像"打过码的条"，
   反而更可疑；马赛克像纹理，读起来更安全。
4. **场景级标签不会挡**：`bdsm / bondage / shibari / exhibitionism / public indecency` 打码后依然存在，
   它们不触发 BLOCK（BLOCK 只看 `explicit` 与 key-point 标签），所以别追着它们改。

操作规则：

- **定位要打码的地方时，只认真正的标记**：`rating=explicit` 加上 key-point / 体液类标签
  (`pussy juice`、`nipples`、`pussy`、`nude`、`spread legs`、`vaginal object insertion`、`areola` …)。
  **位置/场景类描述词不是标记** —— `crotch rope`、`shibari`、`bondage`、`bdsm`、`exhibitionism`、
  `public indecency`、`sex toy`/`vibrator` 本身都不触发 BLOCK。把它们算进"要糊的地方"会**过判**：
  一次实测把 `crotch` 列入条件后，单个格子被标记、连带 3×3 邻域一起糊，整图糊掉 49.9%，图也就没用了。
- 要**看**一张 BLOCK 参考图：按**你要用的粒度**（整图 / 整块面板）打马赛克，网格 16–64px 都行，
  **不要用纯色填**，也不要只涂"可疑的那一小块"。
- **闸门必须在你要动手的粒度上跑**：要切面板就在面板粒度上判定，别拿裁片的判定去推断整图。
- 反过来，**合成图（拼图、贴人物后的整图）要按整图重新过闸门**——四块都 REVIEW 的拼图整体仍可能是 BLOCK。

脚本：（我们的实验）（本实验的矩阵）、
同目录 `censor_panels.py`（一次通过的分块打码 + 单次闸门）、`auto_censor.py`（按格子扫描——**已被本实验证伪**，保留作反例）。

---

## 反向案例：**本来能看的图，平涂打码后反而 BLOCK**（2026-09-30 实测）

同一构图的两个文件（588×721，同一角色的同一张插画），只差两处局部编辑：

| 文件 | 内容 | 闸门判定 |
|---|---|---|
| `161.png` | 角色**穿**白色比基尼上衣 + 半透明纱笼 | **`sensitive` / REVIEW**（可看） |
| `161 - 副本.png` | 同上，但把那两件衣服**擦掉**，换成**两块平涂灰色矩形**（胸 y234–321、裆 y375–471） | **`questionable` + `nude` + `censored` ⇒ BLOCK** |

"是同一张图 + 两处局部编辑"的像素证据（不是两张不同的图）：差异 13.0%，且集中在
**两个大矩形**（28,845 px 胸带、26,072 px 髋带），其余像素几乎逐位相同（连通块中位数仅 4 px）。

结论（与上一节 §打码能不能换来"可看" 第 3 条同源，但这次是**方向相反**的完整对照）：

1. **衣着是强 `safe` 先验**；把衣服抹掉换成中性色块，等于**移除**了这个先验，分类器只能读出裸露的解剖结构 ⇒
   衣着原图 REVIEW、平涂打码版 BLOCK。**"打码"不是安全操作，它可能把可看的图变成不可看的图。**
2. **BLOCK 是"某个文件"的属性，不是"某张画"的属性。** 换一个版本（重存、裁切、修图）重新过闸门是正当的，
   而且可以双向翻转 —— 所以别把某次判定当成对这张画的永久结论。
3. **不要把"我打了码"当成把 `nude` 图带回上下文的通道**：分类器对平涂打码版可能判得更重（本例即如此）。
4. **文件属于人、人已经看过时，由人拍板。** 闸门的目的是保护 **agent 的上传预算与账号**，不是覆盖文件所有者的判断；
   本例中用户看过并明确授权后，agent 才看那张副本 —— 授权来自**所有者**，而不是来自"它看起来无害"的自我判断。

---

## 规则五：BLOCK 的处置 = **人工打码审核通道**（2026-09-30 定案）

BLOCK 不是对这张画的判词，而是"**agent 自己不许看**"的保护动作。**多模态类任务里人的反馈是最高判据** ——
文件所有者本人看过、并由他指定盖哪里之后，agent 看那张**打码产物**是正当的，**不需要**再靠闸门认可。
（判定不因此改变：打码后可能仍然 BLOCK，甚至更差 —— 那不影响授权本身。）

流程（脚本：`libs/wd14-gate/scripts/gate_review.py`，它**不改判定**，只把 BLOCK 之后的处置变成可执行步骤）：

```
python libs/wd14-gate/scripts/gate_review.py IMG...              # 判定 + （若有 BLOCK）打印处置流程
python libs/wd14-gate/scripts/gate_review.py IMG... --tags       # 附带全部标签便于人判读
python libs/wd14-gate/scripts/gate_review.py IMG... --sheet s.png  # BLOCK 多张 -> 拼带文件名标注的联系表
```

1. **agent 不得自己决定盖哪里** —— 把图交给所有者，用 `human_canvas` 涂两层：
   `L1`「要打码 / 不给 agent 看」（人指定区域）、`L2`「要保留」（保护层，mask = `L1 − L2`）。
   多张时先在**缩略图粒度**拼一张联系表（`--sheet`），人一次涂完；缩略图上涂就把
   `--dilate/--feather` 换算回原尺寸。
2. 读回各层 → `canvas_mask.py`（`--rope L1.png --keep L2.png --src IMG --out mask.png --overlay o.png`）
   合成遮罩 → **在副本上**打码，**绝不覆盖原文件**。
3. **默认用马赛克（16–64px 网格），不要平涂**：实测平涂灰块会把可看的图判成 BLOCK，而马赛克常能改善判定。
4. 对打码产物**重新过闸门**并记录：改善 / 不变 / 更差都写进授权记录（谁、何时、涂了哪里、之后判定如何），
   随产物留档。此后 agent 看这张打码图属于**经所有者授权**。
5. **授权只能来自文件所有者本人**。"我觉得无害""它看起来正常"都不是授权；agent 也不能用"我自己先打码"
   来给自己开路 —— 打哪里是人的判断，不是自动扫描能算出来的（见上一节：判定依赖上下文，不只看被盖住的像素）。

### ⚠️ mask 自查的两个陷阱（2026-09-30 首次实操踩到）

- **不要打开"整图叠色"的 mask overlay 来自查。** 那类图是**整幅原图**加一层半透明色，看它就等于把
  **未打码的原图**送进上下文，**正好绕过这套机制要防的东西**（首次实操就这么犯了一次）。
  允许的自查只有两种：① **数字**（面积 %、bbox、连通块形状/数量）；② **只裁 mask 那一小块**再裁图看
  （`crop_view.py`），绝不整图叠色。
- **打码脚本要按 mask 回贴、且保持画布尺寸**：`out = base`；只在 bbox 内做 `crop → 缩 → NEAREST 放大`
  再按 mask `paste` 回去。验证口径是 `mask==0` 的像素**逐位不变**（max diff 0）——那是"只动了该动的地方"的唯一硬证据。

这条与规则二（BLOCK 只给路径）**不冲突**：规则二管"agent 未经授权时的默认动作＝只看路径"，规则五管
"人授权之后怎么把图安全地带回上下文"。两条合起来＝**默认不给看；人授权＋打码之后可以看**。



