# 打底（underpaint）：先给"形"，再让重绘给"质"

**为什么必须打底**：局部重绘（`SetLatentNoiseMask`）是 **denoise 0.65~0.8**，它**基于这一块已有的像素**去改。
那块本来什么都没有（要加一个原本不存在的物件）时，**重绘出来是什么样完全不可控**。
所以：**先在目标位置把物件的"形"画出来（改像素），再掩版重绘** —— 这是**控制手段**，不是省事技巧。

> 单次观察（n=1，未复验）：直出两次都失败（模型为了让手拿到苹果，**顺手把拘束具拆了** —— 道具动作与
> 拘束结构互相抢）；改成"结构底图 → 程序粗画一个红球 → 掩版重绘 denoise 0.70"后一次就成了，
> 而刑具毫发无损。保守方向：**加原本不存在的物件一律先打底，不要指望直出**。

## 形的三个来源（按"图案复杂度"选）

| 复杂度 | 用哪种 | 怎么做 |
|---|---|---|
| **规则/简单**（球、盒、杆、带子、绑带、纯色块） | **① 程序粗画** | `pattern_fill.py`（纯色/条纹/格子/点阵 + 描边 + 外扩）或 `mask_compose.py` 的图元（`--box/--circle/--poly`） |
| **复杂/有机/细致**（花纹、褶皱、蕾丝、复杂机械细节、要贴合已有形体的图案） | **② 人手绘（推荐）** | 在 **PS** 里画（`ps_edit`，本机有 PS 时），或直接在**画布**上涂；模型与程序都很难把这类图案画准，**人画一次比返工三次便宜** |
| **位置与范围只有人知道，但图案本身可程序化** | **③ 画布给区域 → 脚本填充** | 人用 `human_canvas` 涂出区域 → `pattern_fill.py --mask 那层 --pattern stripes --angle 35 ...` 填充 |

### ① 程序粗画

```bash
# 纯色打底 + 描边 + 掩版一起出（掩版会自动外扩 6px，别贴死物件边缘）
python scripts/tools/pattern_fill.py --base base.png --mask region.png --out under.png \
  --color '#C8321E' --pattern flat --grow 2 --outline '#5A1410@3' --mask-out under_mask.png

# 带走向的物件（绑带/绳子）：条纹，角度写走向
python scripts/tools/pattern_fill.py --base base.png --mask band.png --out under_band.png \
  --color '#B03030' --pattern stripes --period 22 --thickness 9 --angle 35 --mask-out band_mask.png
```

### ② 人手绘（复杂图案建议这条）

- **PS 路线**：`ps_edit(open)` 把图交给用户，人在原文件上画出打底（**同一路径 Ctrl+S 保存**），
  agent `ps_edit(check)` 读回；再 `mask_compose.py --box ...` 圈出"要重绘的那块"（通常比人画的范围**略大**）。
- **画布路线**：`human_canvas(action='open', image=..., seed_json={'layers':[{'name':'打底','label':'paint','color':'#ff5050'}]})`
  → 让出回合 → 人涂 → `human_canvas(action='status')` 读回。
- 人画的**边界不必精确**：重绘会把它吃掉；真正重要的是**位置、比例、走向**。

### ③ 画布给区域 → 脚本填充

```bash
# 1) 人用画布涂出"这里要有一条带子"（一层即可）
#    human_canvas(action='open', image=base.png, task='underpaint', seed_json={'layers':[{'name':'带子','label':'band','color':'#ff5050'}]})
# 2) 读回图层 → 合成掩版 → 脚本填充
python scripts/tools/canvas_mask.py --paint band_L1.png --src base.png --out band_region.png --dilate 2
python scripts/tools/pattern_fill.py --base base.png --mask band_region.png --out under.png \
  --color '#B03030' --pattern stripes --angle 35 --mask-out band_mask.png
```


## 特征掩版：用检测框，不要用细点

- `check_parts.py --feature-masks` 输出 `feature/<cls>_<i>.png`，类名 glob：`hand*` `foot*` `head*` `face*` `gag*` `blindfold*` `eye*`。
- `mouth*` 不能直接取关键点：M1 的鼻点会落在嘴上、嘴点落到下巴 ⇒ 由 `face*` 框推——**纵向取脸框下 42%，横向左右各内缩 12%**，再靠 `--grow/--feather` 放宽。
- 可靠度：M2a 手框、M2b 脸框 > M1 细点；框定尺寸，点只定位置。
- 拼掩版：`mask_compose.py --box WxH+X+Y` / `--circle cx,cy,r` / `--poly x1,y1,...`，组合 `--union --sub --grow N --feather N`：
  `--circle 430,352,74 --union out/mask/feature/base_mouth0.png --grow 4 --feather 8`；两人相邻时 `--sub out/mask/base_person1.png`。
- 它拒绝把 `--rgba/--out` 指到输入图（防覆盖原图）。


## 打底之后的掩版重绘（四条硬规矩）

1. **掩版只贴特征，绝不压到结构件**：掩版放得太大、压上颈环/腕铐/横杆就会把它们画坏。
   正确的量：**指尖一个小圆（r=74，离金属杆 90 px）** 与
   **脸框下半（底边停在颈环之上 4 px）** ⇒ 结构件毫发无损。
   重绘区旁边有道具/家具/拘束具时，先 `mask_compose.py --sub` 把它们挖掉。
2. **打底的掩版要比物件略大**（`--grow 2~8` + `--feather 1~4`）：给潜空间 8× 下采样留余量，
   否则打底色块的硬边会留在成品上。
3. **denoise 0.65~0.75**，并在提示词里**同时**写"物件长什么样"（材质/高光/柄叶）
   与**被它影响的接触关系**（"她的手指抓着苹果"）—— 只写物件，模型可能把它画成悬空的。
4. **提示词自相矛盾会毁掉重绘**：实测两次失败各有自己的矛盾句（`the top of her fingers is bare` vs 要画苹果、
   `pouting` vs `her mouth is open`）—— 模型会挑其中一条执行。重绘前把整段提示词通读一遍。

## 反例（别这么干）

- **拿掩版硬刷一个空区域**：结果不可控（实测：直出与重绘各失败一次）。
- **指望"先用低 denoise 试一次"来代替打底**：低 denoise 改不动"无中生有"，只会把边缘糊一圈。
- **打底画得比掩版还大**：露在掩版外的打底像素会**原样留下**（重绘不会修它）。
- 用**云端分色掩版**（见 `localization_and_masks.md` 路 2）来做**要反复返工**的打底迭代 —— 那是按张计费的，
  迭代期先用本地路 1 试，定型后再花那份钱。
