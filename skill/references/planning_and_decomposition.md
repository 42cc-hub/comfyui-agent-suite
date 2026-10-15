# Planning & decomposition — decide *what* to render, in *what order*, *before* generating

Scope: this is the layer **above** prompting and control maps. It answers four questions that decide
whether a complex illustration is cheap or hopeless:

1. **Can the picture be split into layers — and which layer must be generated together with which?**
2. **What must be drawn by program / Blender instead of by a model?**
3. **Which errors must be caught at generation time (they cannot be repaired later)?**
4. **What has to be decided (copy, props, outfit) before a single image is rendered?**

Evidence base: 75 figures planned one-by-one (60 random library samples + 15 hand-picked, 2026-09-27).
The raw, per-image records and the numbered rule list live in
`<你的 ComfyUI 项目>\experiments\plan_practice\PLANS.md`
and the condensed table in `...\plan_practice\SUMMARY.md` — cite them when a judgement here is questioned.

---

## 0. Standing account rules (this user's account)

- **Theme**: 「拘束 → 照顾 → 爱」 — restraint is framed as being cared for, never as violence.
  **No blood / serious injury** (use everyday marks: grazes, bruises, dirt). Keep `blood, gore` in the negative.
- **Never expose key points** (nipples/genitalia) — "R-18 with the sexual content implied" is the house style;
  ropes, tape, clothing, hands, metal cups, framing are the tools that keep it inside that line.
- **Titles / tags / captions contain no AI or technical wording**; the AI declaration is the submission checkbox.
- **Reference images give ideas only**: never base a *published* image on someone else's game screenshot,
  animation frame, or illustration (copyright). The user's **own** older works are safe to reuse and improve.
- **Character tags** must be checked against pixiv's hot spelling before publishing.

---

## 1. The four-step decision, in order

### Step 1 — Classify the reference (and hunt its mistakes)

Decide what kind of reference it is: **technique** (how to draw), **concept** (what to draw),
**data/structure** (proof a pose works), or **the user's own old work**.
Then read it **both ways**: what to borrow, and **what is wrong with it** → write a
**"do-not-inherit list"** (wrong proportions, mushy props, gibberish text, watermarks, platform UI).
Reference images are not standards — the goal is to be **better** than them.

### Step 2 — The delete test: can an element be added afterwards?

Remove the element from the picture and ask: **does the rest still look natural / make sense?**

- **Yes → it can be added later** (a hand from off-frame, a leash that just leads off-frame, a wooden stock,
  a glass-case frame, simple geometry). This is usually **cheaper and more controllable**: draw the person
  first, then place the program/PS element against it.
- **No → it must be generated together with the subject** (a hanging body that needs the rope to explain it;
  anything that leaves a **mark** — rope indentation, cloth deformation, contact shadow).

⚠️ **Marks count as part of the element.** Tight rope leaves indentations ⇒ if you remove the rope you must
also remove the indentation ⇒ the body no longer looks right ⇒ generate together.

### Step 3 — Interaction has a direction (do not treat "interaction ⇒ together" as a law)

Ask the two sides separately:

- **The dependent side** (the one acting on / looking at the other) **cannot be produced alone** —
  its pose and gaze are defined by the partner.
- **The acted-upon side**, whose pose does **not** visibly depend on the partner, **can be produced first**
  (e.g. a standard all-fours pose with only a hand on the head: generate it first).
- **Symmetric interaction** (mutual gaze + mutual contact, both leaning in) ⇒ **only one generation works**.

Weak interaction plus a straight-on camera is not rare: a two-person figure with a **simple background and
side-on stances** can be produced by plain sampling on IL (measured: several such figures came out directly).

### Step 4 — Order layers so that the **uncontrollable** one goes first

- **AI sampling is uncontrollable** — pose, facing, angle cannot be dictated precisely.
- **Blender / a program is controllable** — position, angle, scale can be tuned exactly.
⇒ **Produce the uncontrollable layer first, then make the controllable layer fit it.**
For "person + machine" (a scooter rig, a set of stocks, a binder device):
**sample the person first, then build the machine in Blender against that person's perspective** and merge.
(The opposite order — structure first — is for **architecture/rooms**, where the structure *is* the subject
and the person is a small later addition.)

**Exception** — if the "controllable" element **decides the composition**
(a giant robot arm that fixes where the character stands, where the hand reaches, where the visual weight sits),
the two are a **strongly coupled system**: generate a **cheap rough blockout of both** first, lock the composition,
*then* spend money on each side's detail (`rough character + rough robot → composition lock → final character /
final robot`). Cheap first, expensive second — but **both** rough versions must exist before either is finished.

**Refinement of Step 2 (dependency test)** — replacing a bare "delete test" with an explicit checklist:
remove element A, then verify **one by one** that the background, the pose, the lighting, the occlusion and the
gaze still hold. **All five hold ⇒ A is a late-bind object** (UI, decorative rope, mechanical parts, particles,
small background props). **Any of them fails ⇒ A must be generated with the subject.**

**Refinement of Step 3 (formalised)** — write the dependency as a direction: if **`Pose(A)` affects `B`**
(a hand gripping a rope; a gaze aimed at a screen), then **`B` depends on `A`** ⇒ generate `A` first, then `B`.

**Refinement of the routing rule** — the "human-eye falsifiability" test is best stated as:
**discrete constraints go to programs, continuous aesthetics go to the model.**
*Program*: chain link counts, gear teeth, screws, evenly spaced UI, chessboards, fences, perspective guides,
symmetric structures, mechanical joints, repeating textures. *Model*: faces, hair, cloth folds, atmosphere,
lighting feel, irregular ornament, overall taste.

**And one more check — topology, not just "does it look right"**: the model's most damaging
error is rarely "one screw missing", it is that **the connection relationship is wrong** (a rope drawn *through*
the hand where it should pass *around* it). Check the **occlusion graph, the contact graph and the attachment
graph** explicitly, especially for rope ⇄ body ⇄ prop.

---

## 2. "Would a human spot the error?" — the test that routes work to programs

Route an element to **program / Blender** (not sampling) when it is **countable, evenly spaced, symmetric,
or regularly grouped**, because a viewer verifies those **at a glance**:

- piano keyboards (2-3 black-key grouping), chains, spear fences, lattice/shoji grids, tile grout,
  fence railings, net/fishnet, plaid, evenly spaced rope coils, UI chrome, type.

Related, and stricter:

- **Human anatomy is a high-sensitivity object** — a too-long waist, too-long legs, two legs of different
  lengths, or a wrong joint is spotted instantly. **Check explicitly**: head-to-body ratio, waist:leg,
  left-right symmetry, joint positions, centre of gravity. (Do this even when the reference looks fine —
  references themselves carry proportion errors.)
- **Everyday objects are high-sensitivity objects** — wardrobes, desks, doors, windows, furniture legs,
  wheels, keyboards: people see them daily, so **they tolerate distortion far less** than they do an
  unfamiliar machine. Get the **functional parts** right (lock, handle, hinges, drawer gap) or program them.

Also: **the local stack cannot draw text** (Chinese calligraphy, signage, logos, UI labels) — treat any
in-image text as a program task (`letter_page.py`), or blur/abstract it away; a native reader spots fake
glyphs instantly. AI-generated calligraphy is **pseudo-text**, not a quotation.

---

## 3. Macro vs micro: what must be decided at generation time

**Macro (cannot be repaired later; the fix would be a repaint):**
body orientation/twist, proportions, perspective & camera, contact & occlusion, overall composition,
**and anything that occupies a large fraction of the frame** (a face in a close-up; feet in the foreground).

**Micro (repairable afterwards):** rope/knot detail, buckles, text, light patches, sweat/wetness,
background clutter, a composited UI layer, dropped-in program geometry.

⇒ When you sample, **filter on the macro list first**. Do not spend the budget patching a macro error.

> **预算规则：多轮抽卡是留给"出底图"的，重绘不多轮。**
> 抽卡能换掉的是**宏观层**（结构、人数、两人关系、谁在做什么）；局部重绘只能改**微观层**
> （纹理、绳路、道具、接缝）—— 宏观错误被 mask 外的像素锁死，重绘修不动。
> 所以：**结构不对就继续抽卡，不要在错的画面上打磨**；**结构一旦成立，重绘一遍就够** ——
> 第二遍不会产生可辨差异，还要再付一次 mask 外的 VAE 往返代价。
> 实测（双人底图 ×4 种子各做一次区域重绘）：四张几乎无可辨差异。
> 细节与量化数字见 `references/local_repaint_bigcanvas.md` §预算怎么分。

---

## 4. Precision is decided by purpose, not by habit

- **Background information without identity** (crowds, haze, clutter, distant silhouettes) may be
  sloppy — even wrong — as long as it does not steal attention.
- **Foreground / identity-bearing things** (the character, a crest, a hairstyle, a knot that must read)
  are worth control maps and retouching.
- **Costume complexity is a choice**: game characters (especially miHoYo's) carry so much trim that manual
  correction is uneconomical — either accept "close enough" (viewers rarely notice) or **switch to a simpler
  outfit** (pyjamas, shirt + shorts). Changing the design is usually cheaper than repairing detail.
- **Hiding a difficulty does not remove it** (a hand covering the face just makes the *hand* the focus) —
  ask whether the part you shifted attention to is one you are more confident about.

---

## 5. Decide the copy first, then the props

Write the title / the line / the caption first, then reverse-engineer **what must be visible** for that line
to land (a desk and a keyboard for "stop working and rest"). **Props may be drawn roughly but must not be
absent** — and the narrative prop needs its place reserved at composition time.
**Accessories need a reason** (a cat-ear headset is not a standard part: the character must have a reason to
wear one).

---

## 6. Model-capability notes (measured on this host)

| Observation | Consequence |
|---|---|
| **IL is much faster than Anima** | Sample/explore on **IL**; keep Anima for what IL cannot do. |
| IL is weak at **environment + light** as the subject, and at **complex composition** (multi-subject, unusual perspective, scale differences, crowds) | those go to **Anima**; a **two-person, side-on, simple-background** scene is *not* "complex" and IL can still deliver with several samples |
| The models' **shibari prior is decent** | sample the rope structure, then fix crossings/knots/occlusion alignment/contact shadows programmatically (`rope_studio.py`) |
| **Structural objects** (piano keyboards, chains, railings…) — even GPT-image fails | program/Blender them; do not burn samples |
| **POV / selfie / disembodied hands / a leash held off-frame** are separate difficulties | probe first (cheap small run) before committing to control maps |
| **Text is the single most "breaks the illusion" element** | typeset it yourself, or keep it non-readable / out of focus |
| LoRAs named **`nbe11`** (and friends) smell like game-screenshot/MMD training | test texture on a small sample first — they drag a 3D/plastic look into the illustration |
| **Hiding "inside" a layer is a layer problem, not a prompt problem** (rope must stay inside the collar) | build the thing to be hidden first, then put the coat on top; or paint only the visible sliver inside the opening |

---

## 7. Composition & camera

- **Whole-image composition beats detail composition**: perspective and figures first; small floor props
  belong to the "PS + repaint later" tier.
- **Camera is inferred from evidence**: which side of the body do we see, and what is nearer/larger —
  **not** from a pose summary ("head low, hips high" can simply be a high camera in front).
- **A standalone split needs two things**: a **shared layout** (camera, horizon, scale, stance, light
  direction) and a **flat foreground** (the figure sitting in one depth plane). A foreground that spans
  depth (hand/foot thrust at the lens) cannot be split.
- **Multi-subject scenes**: block the room in Blender → depth (+ line art) → ControlNet/cloud, then add the
  figures **from far to near**, avoiding repeated local repaints of distant figures.

---

## 8. Layered production checklist (what this repo already gives you)

| Need | Tool |
|---|---|
| Rope / chain / glowing "magic line" along an authored path, with correct over-under and contact shadow | `rope_studio.py` (one rope = one bezier; **Z per segment** ⇒ self-knotting; painter's algorithm; global light dial; knot presets; exports layer/mask/SVG/scene JSON) |
| Procedural strand texture along a path extracted from a mask/finished image | `rope_texture.py` (`path` / `render` / `auto`; `--drop-blobs` to reject blob-shaped red regions) |
| Splitting an existing rope image into segments, classifying crossings/knots | `rope_segments.py` |
| Typeset anything (dialogue, captions, UI, multi-language variants) | `letter_page.py` |
| Two (or N) mannequins staged for depth/OpenPose control maps | `blender_stage_multi.py`, `blender_extract_blend.py` (round-trip GUI edits) |
| Local masked repaint of one region (fix a hand, a foot, a knot) | `patch_local.py` (diff-mask) / `redraw_whole.py` |
| Fitting a pose to a reference image and measuring the error | `blender_extract_kps.py` → `blender_fit_pose.py` → `blender_pose_error.py` |
| Harmonizing a composited layer so it stops looking pasted (tone/sharpness/grain/edge) | **`scripts/harmonize.py`** — see `harmonization_and_embedding.md` for the checklist, the measured seam metric and the anime-specific notes |

Rule of thumb: **if it is geometry, author it; if it is flesh, sample it; if it is type, set it.**

---

## 9. Known traps found while planning (do not re-learn these)

- **WD14 hallucinates on pure-text pages**: a novel screenshot was tagged `monochrome, comic, greyscale,
  silent comic` ⇒ the gate and tag-based de-duplication are **unusable for text-only images**; classify those
  by eye. A "text page filter" (high text density / low colour count) is worth adding to any sampling script.
- **Reference libraries are not pure illustration**: phones screenshots, real photographs, 3D/MMD renders,
  other people's doujinshi pages. Decide what the file *is* before borrowing from it.
- **A round-trip sheet of sample/compare images beats words** when reporting "which candidate is usable".
- **The system prompt's scene/light rules are per-stack**: Anima takes *sentences* for scenes and *phrases*
  for elements, with the camera sentence first; IL takes booru tags.
