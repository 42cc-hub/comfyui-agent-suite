# 找与装 LoRA（civitai → 本机）

## 一、检索：用 API，不要翻页面

```bash
# 镜像站（本机可直连）
https://civitai.red/api/v1/models?query=<关键词>&types=LORA&limit=5
https://civitai.red/api/v1/models?query=<关键词>&types=LORA&limit=10&cursor=10   # 翻页
```

**参数是 `?query=`，`?search=` 无效**（会返回一堆无关热门模型）。

返回里要看的字段：

| 字段 | 用途 |
|---|---|
| `baseModels` / `modelVersions[].baseModel` | **必须与你的栈匹配**（见第三节） |
| `modelVersions[].trainedWords` | 触发词，直接拿来写提示词 |
| `modelVersions[].files[].sizeKB` | 体积，先挑小的试 |
| `modelVersions[].downloadUrl` | 形如 `https://civitai.red/api/download/models/<versionId>` |
| `description` | 作者常在这里写用法：强度、负面建议、**默认服装的否定技巧** |

**不要用视觉工具去抓模型页/示例图**（那里常有 NSFW，等于把图上传给模型服务商）。只看 JSON 字段就够，
`trainedWords` 已经把外观描述全给了。

**没有 LoRA、或 sidecar 写得不全**时，特征来源阶梯见 `references/character_features.md`。

## 二、下载：shell 会被 403，走浏览器

- **shell 直连 `downloadUrl` ⇒ `HTTP 403 Forbidden`**（civitai 要登录态；`fetch_url.py` 也一样）。
- **可行路线**：用**浏览器**（带用户登录态）导航到 `downloadUrl` ⇒ 文件落到 `<你的用户目录>\Downloads\`
  ⇒ 再用脚本归位改名。
- 其他源（HuggingFace 等）走 `pipeline/fetch_url.py <url> <dest>`——依赖本机代理 `127.0.0.1:6132`（时有时无，
  失败先重试一次再下结论）。

## 三、底模必须对得上

| LoRA 的 `baseModel` | 能用在哪 |
|---|---|
| `Anima` | Anima 管线（`waiANIMA_v10Base10` + qwen VAE） |
| `Illustrious` / `SDXL` / `NoobAI` | IL 管线（`waiIllustriousSDXL_v170`） |
| `Pony` / `SD 1.5` / `Krea 2` / `Other` | **本机两个栈都用不了** |

装好后**从 safetensors 头部复核，不信文件名**：

```python
import json, struct
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n).decode('utf-8'))
md = hdr.get('__metadata__') or {}
print(md.get('ss_base_model_version'), md.get('ss_network_dim'), md.get('ss_network_alpha'))
```

## 四、归位与命名

- 目录：`<你的 ComfyUI 安装目录>\models\loras\<类别>\`（角色放 `character\`）。
- 命名：**`{作品}_{人物英文名}_{人物中文名}[-底模].safetensors`**，
  例 `绝区零_Claret_克拉蕾-Anima.safetensors`、`绝区零_Claret_克拉蕾-Illustrious.safetensors`
  （同角色多版本用底模后缀；游戏名里的冒号在 Windows 非法 ⇒ 去掉）。
- **有的下载器把中文名写成十六进制**（`E79FA5E69BB4E9B89FE699.YnQa.safetensors` 其实是「知更鸟…」）
  ⇒ 用 `pipeline/rename_hex_loras.py` 批量规范化（它同时改元数据里的 `civitai.name` 和 workflow 里的旧引用）。
- 同组的 `.metadata.json` / `.jpeg` 一起改名（同一 stem）。

## 五、在 ComfyUI 里调用

- **`LoraLoader.lora_name` 必须是相对 `models/loras/` 的完整路径**（含子目录、文件名一字不差）。
  写错直接 400，报错是 `value_not_in_list`——而 `comfy_run.py` 不打印校验正文，要自己 POST `/prompt` 才看得到。
- 强度先按作者建议（常见 0.9–1.0）。
- **角色 LoRA 会拖出默认服装** ⇒ 作者一般在模型页给了"把默认服装部件写进负面"的技巧。
  单次观察：加负面后同一构图 rating 从 `questionable` → `general`（n=1，别当保证）。

## 六、装完先跑一张探针

一张 832×1248 的简单构图（正面/半身），确认三件事：**像不像**、**服装对不对**、
**rating 是否安全**（过 `libs/wd14-gate/scripts/wd14_gate.py`）。不像就换版本/作者；服装串了照第五节加负面；
rating 偏高先压尺度再谈出图。

## 七、淘汰：不值得留的移出 `models\loras\`，别让它继续误导

**判据「加了不如不加」**：同一提示词同一栈，换上某个角色 LoRA 后质感反而变硬（塑料感）、
或服装被拖回默认设定、或**底模本来就认识这个角色** ⇒ 留着只会误导后续（下一步会把锅算到栈头上）。

**处理**：`.safetensors` + `.metadata.json` + 同名预览一起**移出 `models\loras\` 树**
——**移进子目录没用**（ComfyUI 与 LoRA Manager 都递归扫描）；确认无用后再硬删。
同一角色的不同作者版本**分别判断**。删除后用 `/object_info/LoraLoader` 的列表长度复核（214 → 213）。

## 八、批量更新已有 LoRA / checkpoint：走 LoRA Manager 自己的接口

装了 `custom_nodes\comfyui-lora-manager` 就不必手工比对版本（它自己跟 civitai 同步），
且**路由挂在 ComfyUI 同一端口**（它只是插件，不是独立服务）：

```bash
# 列出"有更新"的（read-only，先看清单）
GET  http://127.0.0.1:7187/api/lm/loras/list?update_available_only=true
GET  http://127.0.0.1:7187/api/lm/checkpoints/list?update_available_only=true

# 让它自己下（DB/cache 由扩展维护，别手工下载，否则状态不一致）
POST http://127.0.0.1:7187/api/lm/download-model
```

数据（SQLite + settings + 模型根目录配置）在 `<你的用户目录>\AppData\Local\ComfyUI-LoRA-Manager\`。
注意 `py\routes\update_routes.py` 是**插件自身**的 Git/ZIP 自更新，不是模型更新
（模型版本逻辑在 `py\services\model_update_service.py`）。

**两条纪律**

1. **替换件验证通过前绝不删旧文件**（体积合理 + safetensors 头部可解析）。
2. **不同分支不算"旧版本"**：本地 `anima-base-v1.0` 与 `aesthetic-v1.1` / `turbo-v1.1` 是三个分支，
   base 还是 **LoRA 的训练基底**（官方原文 "LoRAs should be trained using this version"）且被 workflow
   按文件名引用 ⇒ **只增不删**。`turbo` 用法也不同（CFG 1、8–12 步），不能套 base 的 CFG 8 / 30 步。
