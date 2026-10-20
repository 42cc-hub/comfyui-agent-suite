# Human in the loop — `human_canvas` (rough 2D) and `ps_edit` (real Photoshop)

The human owns **aesthetics and intent**; the agent owns **measurement, mechanics and
verification**. Both channels below are **non-blocking**: open the task, say exactly what the
human must do, and **yield the turn** — never busy-wait, never re-open the same task twice in a
round. If the human may be away, ping with a tray balloon (`man_in_loop_notify`).

Pick the lighter channel:

| Channel | Use for | Human effort |
|---|---|---|
| **`human_canvas`** | rough masks, labels, points/edges — anything the agent can **refine programmatically** (GrabCut trimaps, fitting, colour masks, feathering) | paint a blob / drag a point |
| **`ps_edit`** | the task genuinely needs Photoshop on this host: retouching, layers/channels, painting structure the models cannot do | real image editing |

## `human_canvas` — rough 2D input

```
human_canvas action=open image=<path> out_dir=<dir> seed_json={"layers":[{...},{...}]}
  → returns a task token; YIELD, tell the human what to paint
human_canvas action=status task=<token>     # when the human replies
  → annotation.raster.layers[].file  = one PNG per layer (alpha = what was painted)
```

Rules that make this cheap:

- **Ask for two layers, not one:** L1 「repaint」 and L2 「keep」. `keep` is the human's
  *protection switch* — fingers, a hair ornament, a region that is already good. Mask =
  `L1 − L2`. Without it, any colour-based mask you derive will nibble at a boundary the human
  cared about, and you will be re-running passes.
- **Let the human paint on a zoomed crop when the target is small.** A 2–3× nearest/lanczos crop
  of the region (with some context) means the strokes land 2–3× more precisely in original
  coordinates; map the returned layer back by the same factor and offset. On a full 3000 px
  canvas a 120×170 px face is not paintable by hand.
- **Check the human's mask yourself before spending GPU time:** build the mask, tint it over the
  picture, *look at it*, and report the painted area % and bbox. This catches the "they painted
  the whole head, not the eye" class of misunderstanding in one round instead of one GPU run.
- **The canvas has zoom / pan / brush size** — say so, and give coordinates
  (e.g. "the face is around x 840–960, y 780–950" from a grid overlay, see
  `grid_localization.md`). A human with a bare image and no coordinates will guess.
- **Read every layer back, don't trust the JSON's `empty` flag** (a layer can be reported
  non-empty while another one carries the paint).
- **Pre-seed** the layers (`seed_json`), and for graph work pre-seed the nodes so the human only
  drags — asking for two annotations (key regions *and* anchors) beats asking for one
  ambiguous one.

Deriving the mask from a canvas layer:

```bash
python tools/canvas_mask.py --rope L1.png --keep L2.png \
       --src BASE.png --out mask.png --overlay overlay.png --dilate 4 --feather 3
```

`--dilate/--feather` are in **pixels of the canvas you painted on** — if that canvas is a 2×
crop, halve them for the original.

## `ps_edit` — hand a file to Photoshop, verify by content

> ### ⛔ 硬规则：**一次只能开一个文件**（`[measured, n=1]` 未复验；后果不可逆 ⇒ 按硬规则执行）
> **绝对不要连着调两次 `ps_edit action=open`**，也不要在 PS 已打开文件时再 open 另一个：同时开两个文件会让
> **Photoshop 直接卡死**，人只能强杀进程，未保存的手改全丢。要改第二张时**先让人保存并关掉当前这张**。
> 一次交接 = 一个文件 = 一轮对话。

```
1. copy the file into the project's ps/ dir      (never hand over the only original)
2. ps_edit action=open  image=ps/xxx.png         → returns mtime + sha256  (record them)
3. YIELD: "edit and Ctrl+S the SAME path, then reply"
4. ps_edit action=check image=... since_mtime=<mtime> since_sha256=<sha256>
   → changed: true/false  (hash decides; the clock alone is not evidence)
```

- **Hash, not mtime, is the evidence.** A save that only touched metadata, or an edit saved
  under a new name/folder, will fool a clock check. Ask for the filename if the human may have
  saved elsewhere.
- **Archive the human's hand-edited file** — it is the only record of what the human changed.
  Its difference against the base is also the **best possible mask** (exact, free): threshold
  the diff, dilate, feather (see `local_repaint_bigcanvas.md` §"Get the shaped mask for free").
- **Never re-generate over a hand-edited base.** If a repaint must build on the human's edit, the
  *canvas* has to be their file, not the pre-edit base — using the base silently reverts their
  work inside the mask. Conversely, when you must keep their work outside the mask, paste by mask
  so the outside comes from their file.
- If the human **cropped or shifted the canvas**, realign before using it as a base (match a
  feature or use a stored box) — a canvas-size mismatch must fail loudly, not resize silently.
- The human may hand you a **layer** instead of a flattened edit (e.g. a PNG of just the edited
  region with transparency): then paste it **in place** onto a same-size base
  (`Ctrl+Shift+V` in PS, or paste-with-mask locally). Ship both files (base + layer) when that is
  easier for them than blending a flat composite.

## Division of labour（一圈只做一种活）

- The agent proposes the **region** (measured, with a grid overlay), the human corrects it by
  painting; the agent turns paint into a precise mask, runs the model, and shows a **1:1
  before/after of that region** — never the whole image scaled down, where the difference is
  invisible.
- Ask for **one kind of work per round**. "Paint the mask, then also tell me the caption" wastes
  a round and gets both jobs half-done.
- When the human says "I'll fix it in PS myself", stop generating: hand over the *exact* file
  (with base + a layer file when useful), and switch to the verification role.
- Report what you changed as **measurements** ("outside-mask change 0.0000 %", "face mask 2.09 %")
  — that is what lets the human trust an edit they cannot see at conversation scale.
