# human-canvas

**人在回路**的默认交互组件：agent 遇到"只能由人给的粗信息"（要重绘哪块、骨骼点在哪、掩版怎么分）时，
把图交给这个本地网页画布，人在上面涂几层 / 点几个点，agent 读回结构化结果继续干活。

- 两种任务：**掩版层**（多层、带标签、每层一种颜色）与**矢量图**（节点 + 边，如骨架点）。
- 非阻塞：agent 打开画布后**让出回合**，人画完回一句话，agent 再读回（`action=status`）。
- 位置：`lib/tools/human-canvas/`，作为插件挂载（`cordis.patch.yml` 已带 `insert` 行）。
- `lib/tools/ps/` 是**可选**的 Photoshop 通道（需要本机装 PS）；没装 PS 时只用画布即可。
- `lib/notify.mjs` + `notify/win-balloon.ps1`：Windows 托盘气泡提醒（人不在屏幕前时用）。

## 脱离 DSH 使用

`server.mjs` 是一个独立 HTTP 服务（`python -m http.server` 风格的零依赖 Node 服务），
只要能把浏览器指到它的端口，就能拿它当"人给粗标注"的通用小工具；`index.mjs` 里的前端协议是纯 HTTP+JSON。
