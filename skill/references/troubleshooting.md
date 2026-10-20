# Troubleshooting —— 看到什么 → 查哪里 → 怎么办

红线相关的两条不在本文件，在 SKILL.md 的 Safety gate 与 Cleaning up。

## ComfyUI / 提交

- **`To see the GUI go to ...` 一直不出现** — 看 job 输出尾部；多半是沙箱写拒绝或 db 锁（见 SKILL.md 的 Start/stop）。
- **提交时 `node_errors` 非空** — 图里引用了未知节点或坏枚举；重新 `GET /object_info` 拿准确字段值。
- **第一次跑特别慢** — 模型要进 6 GB VRAM（async offload）；SDXL/Anima 1024² 含加载约 20–90 s。
- **CLI flag 看起来没生效** — 去读**实际提交的图**（`wf/*.json`）。`--lora-strength` 没接到 `LoraLoader.strength_model`（脚本里硬编码 0.8）时，输出与"denoise 太低"不可区分；要改的是脚本，不是参数。
- **下载源被判"坏了"** — 先跑连通性矩阵：本机国际线路会整体掉（`[measured]`：hf-mirror / huggingface.co / github.com / ghproxy / jsdelivr / pypi.org 全 `URLError`——DNS 通、TCP 失败；baidu / bing / modelscope.cn / mirrors.aliyun.com / pypi.tuna 正常）。断点续传用 `Range`；`IncompleteRead` = 截断，只有读到 EOF 才算完整，**不许拿猜的字节数当完整性判据**。
- **HTTPS 下载一律 `schannel: AcquireCredentialsHandle failed`** — 本机 shell 没有直连 internet，Windows TLS 在沙箱里拿不到凭据。别硬试：走本地代理 `127.0.0.1:6132` + **Python**（scripts/fetch_url.py，OpenSSL）。`curl` 和 `Invoke-WebRequest` 改 `SecurityProtocol` 也没用。

## 提示词 / LoRA

- **串味、画错角色** — 按顺序查：(1) LoRA 的底模家族与管线是否匹配；(2) **触发词表是否完整**（sidecar 的 `civitai.trainedWords` *和* `civitai.images[].meta.prompt`，加 `ss_tag_frequency`），以及自己的 tag 有没有与 LoRA 自带词冲突；(3) 删掉冲突 tag。**不要先动 `strength_model`/`strength_clip`**：插值只是经验上平滑、不是保证的混合，调低会悄悄削弱你正想修的那个身份——除非 LoRA 明显损坏画面，否则都保持 **1.0**（要降也是最后一步，先换 LoRA）。

## 掩版 / 重绘

- **局部重绘什么都没变** — 最常见是**反了掩版**：`LoadImage` 的 MASK = `1 − alpha`，把 mask 原样写进 alpha 就是重绘补集（静默 no-op）。写 `putalpha(255 - mask)`，并用 `diff_report.py` 证明：in-mask 变化 ≈ 30 %、**out-mask 0.0000 %**。掩版为空或反了表现为 in-mask ≈ 0。
- **预处理器返回 `people: []` / 已上传的图却检测不到** — `LoadImage` + 预处理器的结果**按文件名缓存**。`upload(..., overwrite=true)` 只改字节，节点仍返回**旧**缓存结果，所以复用同一个输入文件名会把第一次的结果（包括第一次的**失败**）永久钉住。`[measured]` `OpenposePreprocessor`：固定文件名 → 每次都 `people: []` / `face_keypoints_2d` 0；同样字节换新名 → **face 70/70、bbox 一致**，3/3 次。**不要**用内容哈希"修"它——内容相同则名字相同，等于把空结果变成确定行为。每个检测输入都用**唯一**名字（`role_stem_timestamp_rand.png`）；否则一次 dose sweep 从第二个变体起全部报"没找到脸"，与检测器不稳无法区分。

## Blender

- **启动即 exit 1** — 缺 `--factory-startup`（它会去写 `%APPDATA%\Blender` 下的扩展缓存）。
- **写完所有文件后才 exit 1** — 同一个扩展缓存被沙箱拒绝。**看文件是否存在**，别信退出码；或把 `BLENDER_USER_EXTENSIONS` 指到工作区路径换取干净退出。
- **骨架与 clay 渲染错位** — 关键点是在 armature 空间投影的；先乘 `arm_obj.matrix_world`。
- **depth 图是一片灰** — near/far 绑到了整个场景而不是主体。
- **img2img 出来是单色** — 灰 clay init + 低 denoise；提到 ~0.8。
- **叠 ControlNet 后主体被重新取景** — depth 权重保持 ≥ 0.7。

## 放大 / 抠图 / 云端编辑

- **放大结果发软** — 先确认真正加载的是哪个 `.pth`。`2x_AnimeJaNai` 明显比 `4x-AnimeSharp` 软（**已移除**，见 SKILL.md 的 Machine facts），且 4× 降到 2× 胜过原生 2× 模型。仍软则多半软在更早的一步（"先放大再重绘"的链会把软输入烘进去），不是最后一次 resize。
- **云端编辑的区域贴不进画面**（可见矩形、光晕、绳子对不上）— 工具忽略了掩版和/或重打了背景光。**永远不要用答案替换原图**：按掩版贴回、用带暗边的 subject-only merge、先对齐背景中位色偏、细结构保留原像素（`references/cloud_edit_and_merge.md`）。
- **脚本报"检测失败"但检测器是好的** — Windows 控制台默认 GBK，`print("✗ …")` 抛 `UnicodeEncodeError` 直接**杀掉进程**，调用方只看到非零退出。`[measured]` 无 guard 3/3 报"没找到脸"、加 guard 3/3 成功（同一张图）。每个打印符号的脚本顶部放 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，子脚本用 `python -X utf8` 调。
- **`insightface` 报 `InvalidProtobuf` / `Protobuf parsing failed`（那个模型你根本没要求）** — `FaceAnalysis` 会加载 pack 目录里**每一个** `.onnx`，`allowed_modules` 拦不住加载。只把需要的放独立目录（最小集 `buffalo_l_min/` = `det_10g` + `w600k_r50` 就够检测+识别，即 ArcFace 身份相似度，不必等大的 `1k3d68`）。再看 EP 列表：本机 `onnxruntime 1.29.0` **没有 CUDA provider**，FaceID 走 CPU。
