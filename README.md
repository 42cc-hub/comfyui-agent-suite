**English** | [中文说明](README.CN.md)

# comfyui-agent-suite

Tools that teach an agent to "**generate with ComfyUI / repaint locally / collaborate with a human**".
The core is `skill/` (the instructions + scripts an agent reads); `libs/` holds components that **stand alone as
libraries** — install only what you need, and you don't have to replicate the author's environment.

- **No datasets, no source images.** Model weights come from the Release.
- **`skill/` itself is nearly dependency-free**: the drivers are Node (built-ins only); the Python side needs nothing but `Pillow` + `numpy`.
- For the heavy jobs ("detection / matting / safety gate"), install the matching `libs/*` — each declares its own dependencies.

## Module maturity (**read this table before you use anything**)

Some modules are polished, some are still under development. Spelled out, so you don't assume "it's all reliable":

| Module | Status | Notes |
|---|---|---|
| **Detection (M2a person/hand/foot/head, M2b four face classes)** | ✅ **Relatively reliable** | Validated with per-source-image splits and multiple seeds; head boxes are especially accurate (the mask's "head" is their **inscribed ellipse**) |
| Local quality classification (M3) | ⚠️ Usable, few samples | Only coarse calls like "broken structure / blurry / missing part" are reliable; defective samples are the bottleneck |
| **Skeleton (M1 keypoints)** | ❌ **Unreliable, WIP** | Never got a dedicated effort: joints drift, fine points (nose/mouth) land in the wrong place, and person↔skeleton pairing fails under strong overlap. **Don't derive masks from M1 keypoints** — use detection boxes |
| **Rope** | ❌ **Unreliable, WIP** | Obvious misses in low-contrast and dark scenes; the positive-pixel ratio is so low that the training signal gets diluted. Per-round conclusions live in `docs/` |
| Human-in-the-loop canvas / gate / matting / control maps | ✅ Usable | See each `libs/*/README.md` |

**What's next** (contributions welcome):
1. **Skeleton dataset**: M1 needs a batch of **human-corrected** skeleton annotations before it can be fixed — the most worthwhile data to invest in right now;
2. Rope: a harder eval set (the hold-out set is only 5 images — too small to see a 1–2 point improvement) plus sampling / positive-weight changes;
3. Keep deriving **head/hand/foot repaint masks** from detection boxes (no keypoints needed).

## Components and dependencies

| Component | What it does | Dependencies | License |
|---|---|---|---|
| `skill/` | **Core**: SKILL.md + references + ComfyUI drivers + self-check / upscale / small tools | Node 18+; Python: `Pillow`, `numpy` | MIT |
| `libs/human-canvas/` | Human-in-the-loop canvas (+ optional Photoshop channel, Windows tray notifications) | Node 18+ (as a DSH plugin); PS optional | MIT |
| `libs/wd14-gate/` | Output safety gate (WD14 tagging / NSFW verdict / a manual approval path for BLOCK) | `onnxruntime`, `numpy`, `Pillow`, `pandas` | MIT (models licensed separately) |
| `libs/sam-matting/` | Matting / mask composition (promptable SAM segmentation) | `torch`, `opencv-python`, `Pillow`, `numpy` | MIT (SAM: Apache-2.0) |
| `libs/blender-controlnet/` | Blender-generated depth / OpenPose / hand control maps | Blender 3.x+ (scripts use its bundled Python) | MIT |
| `libs/comic-layout/` | Comic panel layout and lettering | `Pillow` | MIT |
| `libs/anime-detection/` | Anime body / part / face / quality detection and evaluation (train + eval + inference) | `torch`, `ultralytics`, `opencv-python`, `Pillow`, `numpy` (the labeling GUI also needs `PyQt5`) | **AGPL-3.0** |

## Install

```bash
git clone --recursive https://github.com/42cc-hub/comfyui-agent-suite comfyui_agent_suite
python install.py                 # installs the skill into the DSH skills dir + writes config/paths.json
python install.py --with-models   # also fetch the SAM and WD14 models (optional)
```

`install.py` does exactly four things: install the skill (`--dest` to change it; defaults to
`~/.dsh/skills/comfyui-image-generation`), write `config/paths.json` (absolute paths to the sub-libraries plus the
ComfyUI address), fetch models on demand, and print self-check commands.
**Keys never enter the repo**: `config/keys.example.json` is a template — copy it to `keys.json` and fill it in
(it is already in `.gitignore`).

Install sub-library dependencies only as needed:

```bash
pip install -r requirements.txt                     # the skill itself (Pillow / numpy)
pip install -r libs/wd14-gate/requirements.txt      # only what you actually use
pip install -r libs/anime-detection/requirements.txt
```

## Model weights (Release download)

The repo **ships no weights**. 11 self-trained models, 70.3 MB in total (person/hand/foot/head detection, four
face classes, local quality classification, COCO17 skeleton). Provenance, config, metrics and sha256 all live in
`weights/manifest.json`:

```bash
python skill/scripts/fetch_weights.py --list                    # see the manifest and sizes first
python skill/scripts/fetch_weights.py --from-release 42cc-hub/comfyui-agent-suite@weights-v1
python skill/scripts/fetch_weights.py --from-dir D:/my_runs     # if you trained your own: find best.pt by weight name
```

They land in `libs/anime-detection/runs/<name>/weights/best.pt` (the default lookup path for the suite's scripts)
and every file is checked against its sha256: **a mismatch is reported loudly, never silently used**; if something
can't be fetched, the script prints the manifest and exits instead of failing quietly.

## First run: the install-succeeded probe

**`skill/examples/hutao_furina_two_girls.md`** — from a single prompt, produce an image where
"Hu Tao + Furina share the frame, and left/right plus the shoulder-lean direction must not flip":
Anima base render → body localization → split into two non-overlapping masks at the overlap midpoint →
per-person masked repaint → self-check. **If it runs, your install is correct** — and this very approach
(position comes from masks, identity comes from the prompt / LoRA) is the main idea behind the whole toolkit.

![Probe, three panels: base → per-person masked repaint → final](docs/images/probe_pair.jpg)

Left: the Anima base (they share the frame, but the pose/orientation is only "accidentally right"); middle: the two
characters split by detection boxes, each given its own non-overlapping mask and repainted separately; right: the
final — Furina (school swimsuit) asleep on her side on Hu Tao's (sailor uniform) lap, Hu Tao looking at the camera.
At no point is anything **pasted back** from the original render.

## Recommended setup (the author's machine, for reference only)

Windows + two `conda` environments (one for detection/training:
`torch`+`ultralytics`+`segment_anything`+`opencv`+`PyQt5`; one with a newer torch for training), Node 18+,
Blender 5.x, ComfyUI on local port 7187.
**The minimum viable setup is a different thing**: ComfyUI + Node + `Pillow`/`numpy` is enough to run the
generation and repaint routes in `skill/`.

## License and compliance

Each component is licensed per directory (see the table above and each library's `LICENSE`): the top-level
`LICENSE` is **MIT**, covering `skill/` and the `libs/*` marked MIT; `libs/anime-detection` is separately
**AGPL-3.0**. **When you distribute a combination that includes it, comply with the strictest of those —
AGPL-3.0** (that library depends on Ultralytics; closed-source commercial use requires a separate Ultralytics
license). This repo contains **no training data and no source images**; the `libs/anime-detection` weights were
trained on a private dataset, and users are responsible for the compliance of their own use case.
