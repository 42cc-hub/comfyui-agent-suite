# human-canvas —— 人在画布上输入

> 取代旧的 `mask-editor`。它不只是"画蒙版"：把**人在图上/图结构上的输入**统一成一份结构化、
> 带语义标注的产物，交给 agent 去理解和使用。

## 为什么需要它

agent 的短板是**感知**：它能算，但看不清。人恰好相反——看得清，但不想算。
关键差别在于**人的输入精度是可变的**：

| 人花的时间 | 给出的东西 | agent 该怎么做 |
|---|---|---|
| 几秒 | 粗糙几笔，圈个大概 | 结合图像边缘 / 颜色 / 已有检测，把粗描细化成明确边界 |
| 几分钟 | 精确描出语义边界 | 直接用，不必再猜 |

纯逻辑脚本做不到这种"模糊输入 + 推理补全"。所以本工具**只负责忠实记录人的输入**，
不做任何自动"智能"处理——怎么用是 agent 的判断。

## 两种输入形态

### 栅格（多图层涂抹）
- 每个图层一个语义（「裙摆」「左腿」「天空」），颜色区分，**每层可填 label**。
- 工具：画笔(半径/不透明度/柔边)、橡皮、魔棒、套索、磁性套索、矩形、椭圆、多边形、贝塞尔、撤销。
- **左键 = 加选，右键 = 减选**（2026-10-03）：等于给**所有**工具都配了橡皮 ——
  矩形/椭圆/套索/磁性套索/魔棒右键都是"把这块从当前图层减掉"，画笔右键就是擦。
  多笔画工具（多边形/贝塞尔）的加减**由第一笔决定**，该笔全程跟随（`Delete` 退掉上一个点、Esc 放弃、双击/回车闭合）。
  右键原先的"退点"已让位，**撤销统一走顶栏「撤销」或 Ctrl+Z**。
- **中键按住拖动 = 平移画面**（放大后精细操作时用；滚轮缩放保持不变）。
- 输出：每层一张 RGBA PNG，**RGB = 该层颜色，A = 覆盖率**（既能直接看，也能当 mask 用）。
- 典型用法：人粗描边缘 → agent 用边缘检测 + 人的先验求出精确边界 → 送 inpainting / 抠图 / 分区域重绘。

### 矢量（图结构）
- 点 + 边，可拖动定位，**每个点和每条边都能填 label**。
- 工具：「点」空白处点击加点 / 拖动移动；「边」从一点拖到另一点连线；Delete 删除选中。
- 典型用法：**修正 / 补充骨骼关键点**——agent 先把自动检测到的点作为 seed 灌进来，
  人只改错的那几个、补上丢的那几个，label 直接写 `left_ankle` 这类语义名。
  也可用于标定其他图结构（拓扑、连线关系、任意节点图）。

### 备注
面板底部一个自由文本框：人可以直接告诉 agent 自己干了什么
（"左腿整个没检出，我补了三个点；裙摆我粗描了一圈"）。

## agent 用法（非阻塞，与 ps 工具同一套约定）

```js
import { openHumanCanvas, canvasStatus, resetHumanCanvas, closeHumanCanvas } from '@dsh-external/human-canvas'

// 1) 开画布（立即返回，不等人）；可以把自动检测结果当 seed 灌进去
const { url, task } = await openHumanCanvas({
  image: 'G:/work/photo.png',
  seed: { nodes: [{ id: 'k13', x: 1162, y: 530, label: 'left_ankle' }], edges: [] },
})
// 2) notifyHuman({message:'需要你在画布上补一下左腿'}) + ask_user_question 闸门
// 3) 【让出回合】：告诉人「画完点提交，然后回复我」
// 4) 人回复后恢复：
const st = await canvasStatus(task)
if (st.done) { const { annotation, outDir } = st /* 每层 PNG 在 outDir 里 */ }
```

- **不要**在这里阻塞等人；服务跑在 detached 子进程里，位置记在 `canvas-state.json`，
  所以 agent 下一轮用 `canvasStatus(task)` 就能读回。
- `resetHumanCanvas()` 在开新一轮前调用，避免上一轮结果被当成本轮。
- `closeHumanCanvas()` 人不再需要时调用。

## 数据契约

`<outDir>/annotation.json`：

```json
{
  "tool": "human-canvas", "version": 1,
  "task": "t-xxxx", "image": "G:/work/photo.png", "size": [832, 1216],
  "submitted_at": 1757000000000,
  "raster": { "layers": [
    { "id": "L1", "name": "图层1", "label": "裙摆(粗描)", "color": "#ff5050",
      "opacity": 1, "visible": true, "file": "layer_L1.png", "empty": false }
  ]},
  "vector": {
    "nodes": [{ "id": "n1", "x": 10, "y": 20, "label": "left_ankle" }],
    "edges": [{ "id": "e1", "a": "n1", "b": "n2", "label": "left shin" }]
  },
  "notes": "左腿没检出，我补了三个点"
}
```

图层文件 `layer_<id>.png` 与图等尺寸，alpha 即覆盖率。

## agent 侧怎么用（以抠图为例）

人只涂粗糙的两层，agent 负责把它变成精确边界。实测配方：

1. **两层 alpha → GrabCut trimap**：重涂（α>0.85）= 确定前/背景，轻涂（α>0.15）= 可能，
   都没涂 = 可能背景；两层重叠处让"涂得更多"的那层降级为"可能"。
2. `cv2.grabCut(img, mask, None, bgd, fgd, 5, GC_INIT_WITH_MASK)` —— 用图像颜色建 GMM，边界吸附到真实边缘。
3. 后处理：最大连通域（去碎块）→ `binary_fill_holes`（内部异色如水印不会被打穿）→
   形态学 close/open 平滑 → ~1px 高斯羽化。
4. 输出 `alpha.png` + 透明 `cutout.png`，并生成对照图供人复核。

**人的先验值多少（实测，397×270 街景人像）**：同样的算法、同样的后处理，只换初始化——

| 初始化 | 结果 |
|---|---|
| 自动（居中方框，无人的输入） | 把右上角一片街景整个吞进来（多选 2560 px），头发边缘也漏 |
| 人涂的两层 trimap | 干净轮廓，多选/漏选各 ~2600 px 全部消掉，IoU 82.7% → 目视无残留 |

> 注意 OpenCV 5 去掉了 `GC_DEF_BGD/GC_DEF_FGD` 别名，"确定"就是 `GC_BGD(0)`/`GC_FGD(1)`。

## 服务端

| 端点 | 说明 |
|---|---|
| `GET /` | 编辑器 |
| `GET /task` | 当前 task + 图片 URL + seed |
| `GET /img?p=` | **白名单内**的图片（注册的图 + `--allow` 目录），其余 403 |
| `GET /status?task=` | 是否已提交；**task 不匹配返回 done:false** |
| `POST /submit` | 收一次提交（单个 JSON，含各层 dataURI），落盘拆文件 |
| `POST /reset` | 清空上一轮产物 |
| `POST /config` | 复用已起的服务开新一轮（换图/换 task/换输出目录） |
| `GET /health` | 身份探针（`{"tool":"human-canvas"}`） |

监听 `127.0.0.1`，端口从 8140 起顺延找空闲口。

## 相对旧 mask-editor 修掉的三处

1. `/img?p=<任意绝对路径>` 无白名单 → 现在只服务注册过的图/目录（旧版本机任意文件可被读取）。
2. `done.txt` 是全局单例、服务端从不清理 → 现在 done 里带 **task token**，串轮次会被拒。
3. 服务与 agent 进程生命周期耦合 → 现在以 detached 子进程运行，位置写 `canvas-state.json`。

## 测试

```bash
node test/human-canvas/smoke.mjs --image <png> [--kps <kps.json>]
```

覆盖：起服务、`/health`、`/task`、白名单命中/拒绝、过期 task 拒收、错误 token 拒收、
提交→读回（图层/点/边/备注全字段 round-trip）、图层 PNG 落盘、reset 生效。
