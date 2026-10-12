# Harmonization — make a composited element stop looking pasted

Problem this solves: a program-drawn rope/chain, a Blender-rendered machine part, a UI overlay, a matted
cut-out or an AI-generated extra person is placed into a finished illustration and **reads as a sticker**.
That happens because four things disagree across the seam:

1. **Tone / white balance** (the patch is warmer, cooler, darker, more saturated than its surroundings)
2. **Sharpness / detail scale** (AI or Blender output is soft; program geometry is razor-sharp)
3. **Grain / noise** (illustration JPEG noise vs. a clean synthetic layer)
4. **Edge behaviour** (hard cut, halo of the source background, missing contact shadow)

`scripts/harmonize.py` fixes all four numerically. Measured on a real case (a program-drawn chain pasted onto
an AI classroom render, 832×1216):

| metric | naive paste | after `--match lab` | after `--match luma` | `--match hist` |
|---|---|---|---|---|
| **seam step** (mean \|ΔL\| across the mask border, lower = better) | **60.22** | **26.04** | **25.43** | 54.19 ❌ |
| patch HF profile (Laplacian var / residual σ) | 743.9 / 2.86 | 11077 / 10.61 | 12330 / 11.16 | 11944 / 11.01 |
| base's own HF profile (for reference) | 11811 / 10.96 | — | — | — |
| added grain σ | 0 | 3.84 | 3.84 | 3.84 |

⇒ **tone match alone cut the seam step by ~57 % and brought the patch's sharpness/noise profile onto the base's.**

### Second case — a **bright** object (calligraphy sheet) hung on a warm wall (2026-09-27)

Generated an IL interior (Chinese study, warm lantern light) and pasted a photo-derived white calligraphy
sheet onto the wall, three ways. **The lesson is that `--match lab` is wrong for a bright object, and that the
seam metric must not be measured on the side where a cast shadow legitimately falls:**

| variant | seam (all edges) | **seam (top edge only — no shadow there)** |
|---|---|---|
| naive paste (hard edge, no shadow) | 41.7 | 42.3 |
| `--match lab` (paper dragged to the wall's luminance) | 52.7 | 42.3 |
| **`--match chroma` + `--cast` (paper keeps its brightness, gets the room's cast + a cast shadow)** | 55.6 | **34.7** |

Visual verdict: naive = a sticker; `lab` = the white paper turns into an old amber scroll (tone fits, identity
lost); **chroma + cast = reads as a sheet hanging in that warm room** (bright ivory paper, warm cast, subtle
shadow under the right/bottom edge). What is still **not** natural in all three: the sheet's **plane** does not
match the wall's perspective and it has no frame/thickness — i.e. the residual failure is **geometry**, exactly
the category the external review put above tone.

### Third case — hang it properly: a **wall** + a **program-drawn frame** (2026-09-27)

Pasting a bare sheet onto a room's back wall proved a poor test (the user's point: a bare sheet on a wall is not
something you can judge). Redone: generate **a wall** (IL, frontal plaster wall, warm lamp lower-left, wainscot),
give the calligraphy a **program-drawn frame** (mat + dark-wood frame + inner gold line + bevel highlight), hang it,
then composite. Seam measured on the **top and left** edges (the cast shadow falls lower-right, so those two edges
are the fair ones):

| variant | seam top | seam left |
|---|---|---|
| naive paste | 59.9 | 79.1 |
| **`--match chroma --cast 0.40 --ao 0.10`** | **42.9** | **55.6** |
| + `--lowfreq 80` | 42.9 | 55.6 (**no change**) |

- The framed picture with chroma-cast + cast shadow **reads as hung on that wall** (it takes the wall's warm cast,
  gets a soft lower-right shadow, and stops being a hard sticker). **−28 % / −30 % seam.**
- **`--lowfreq` did nothing here**, and that is the correct behaviour: on a nearly uniform wall there is no
  low-frequency gradient to inherit across the object. It only earns its keep when the base has a **visible light
  ramp across the object** (e.g. hung close to the lamp) — a good follow-up micro-test.
**Lesson: put the object where the scene's structure is (a wall, a floor, a desk), otherwise the composite cannot
be judged at all.**

### Fourth case — **replacing a rope's texture** with our procedural rope (2026-09-27)

Chain: IL renders a bound figure → `rope_texture.py path/render` extracts the rope's centreline and redraws it with
our own strand texture → `harmonize.py` adjusts the replaced region. Measured:

| step | result |
|---|---|
| extraction on a **red** rope in a **warm wooden** room | ❌ the colour test caught the wood/skin too ⇒ stray "rope" drawn on her hair; `--region` helps but is not enough |
| extraction on a **deliberately blue** rope (separable colour), mask → `path --source gray` | ✅ 63 paths; **IoU vs the blue rope 0.51, precision 0.975, recall 0.52** |
| `render --strands 3 --pitch 3.2 --light-dir=ul` | draws the strand texture along the path, light side up-left |
| `harmonize --match chroma --ao 0.28` | tool's own border metric 61.5 → **37.9**; visually the contact shadow along the rope edge is what sells it |

Lessons: **(1) make the rope's colour separable at generation time** (blue/green/purple), because a central
colour test is unreliable in a warm scene — or supply the mask yourself (`--source gray` with a mask as `src`).
**(2) The tool draws a centreline + width**, so it covers ~half of the original rope *band*: widen with
`--width-scale` / `--max-paths` or the old rope shows through. **(3) For thin structures a ring-band seam metric
needs small kernels** (3/9 px), otherwise the band is empty and reports 0. **(4) Do not luminance-match a rope**
(it has its own material) — `--match chroma` (+AO) is the right mode.

### Fifth case — a **chain** along a hand-painted path (`chain_texture.py`, 2026-09-27)

The human paints where the chain should go in `human_canvas` (one raster layer), then:
`rope_texture.py path <layer.png> --source alpha` → centreline → `chain_texture.py render <base.png> --path …` →
`harmonize.py`. Measured on the blue-rope figure (chain from her neck across the wooden wall to the frame edge):

| step | result |
|---|---|
| link proportions Lo/Wo 1.28, wire/Wo 0.31, **pitch = Lo − 1.5·wire** | ✅ interlocked (see the linking number below) |
| pitch = Lo − wire (the textbook "inner length + wire") | ❌ **degenerate**: the neighbour's wire lands exactly on the inner boundary ⇒ Lk ≈ 0.07, i.e. the rings only *touch*, they are not linked |
| rotating each link's long axis 90° in the image plane (v1) | ❌ wrong rotation axis (see lesson 1) |
| splitting the edge-on link into far/near wires and painting in 3 passes (v2) | ❌ still stacked profiles — a painter's algorithm cannot express interpenetration (lesson 2) |
| **3-D tubes + per-pixel z-buffer** (v3) | ✅ Gauss linking number **Lk = 0.99** for every adjacent pair (19/19 on the blank test) |
| `harmonize --match luma --ao .35 --cast .30` | border metric 61.5 → **41.4**; better than `--match chroma`, which drains the metal's own colour |
| renderer's own `--shadow` **and** harmonize `--cast` | ❌ double shadow — leave the drop shadow to `harmonize` |

Lessons (this entry went through three revisions; the first two were geometrically wrong and the user caught
both by writing out a cross-section):

**(1) A chain's links are not rotated in the image plane; each link rotates about the CHAIN TANGENT.** Every
link's long axis stays along the chain. One lies in the image plane (a full oval ring with a hole); the next is
seen nearly edge-on, i.e. **compressed along its short axis** — its two wires project on top of each other, so it
reads as a narrow bar (a sliver of hole only if the projected offset exceeds the wire diameter, which needs a big
viewing tilt). Model this with a single `--tilt` (≈12°) applied to *all* links, not with a manual squash.

**(2) "Hooked through each other" is a topological fact — a 2-D painter's algorithm cannot express it.** The user's
test: cut a profile through a single ring ⇒ `ABCDCBA` (A outer edge, B wire, C inner edge, D hole); two rings that
are *hooked* ⇒ the wires of the two rings alternate (`…1…2…1…2…`); the painter's version produced
`A1B1C1D1C1B1A1` then `A2B2C2D2C2B2A2` — each ring painted as one complete closed shape, just stacked in depth.
The fix is not a better z-order rule but a **real z-buffer**: build each link as a 3-D tube (a planar stadium
centreline swept with a sphere of radius t/2), project orthographically, and per pixel solve the ray/centreline
closest point analytically (depth = z_center − √(ρ²−d²), normal = the perpendicular component of the hit vector
w.r.t. the local tangent). Nearest wins — occlusion is *computed*, never ordered. Cost: ~0.3 s for a 10-link chain
in pure numpy.

**(3) Verify the topology numerically, don't squint at it.** The discrete **Gauss linking number** of two adjacent
link centrelines is ≈ ±1 when they are genuinely linked and ≈ 0 when they merely overlap; it caught the degenerate
pitch immediately. Keep it as a post-generation self-check (`chain_texture.py --report` prints Lk per pair, and
warns when a pair is not linked — e.g. across a stitched hand-drawn mask junction).

**(4) The geometry inequality that makes interlocking possible**: hole width (Wo − 2t) ≥ wire diameter t, i.e.
**Wo ≥ 3t**; and the pitch must sit strictly inside `t < p < Lo − t` — in practice `p = Lo − 1.5t` gives 0.5 t of
clearance on each side. The textbook `p = Lo − t` is the boundary case (zero clearance) and is numerically
degenerate.

**(5) Swept-sphere tubes need the true tube normal.** Using the per-sphere radial direction leaves a ~20° normal
crease at every segment joint, which cel quantisation amplifies into stripes all over the wire. Take the
perpendicular component of (hit − closest centreline point) w.r.t. the local tangent instead; also shade *after*
rasterising (store the normal buffer, blur it mildly, then quantise) so that rasterisation noise does not become
banding. **(6) Cel-quantise with thresholds, not a linear ramp** — a ramp makes the highlight far too wide and the
metal reads as plastic.

**(7) A hand-painted mask is multi-stroke and its skeleton forks**: when stitching segments, orient A so it *ends*
at the chosen endpoint and B so it *starts* at its own; the inverted rule draws the chain twice — the signature is
"two parallel chains that converge" (and the `tangent` field is `(dy, dx)`, so reading it as `(dx, dy)` stands the
whole chain up 90°). Smooth over ≈1 link width afterwards. **(8) Test on a blank canvas first** — a chain judged
only on top of a photo hides geometry errors behind the background's texture.

Implementation: `pipeline/chain_texture.py` (`render` subcommand; `--tilt`, `--wire-ratio`, `--length-ratio`,
`--pitch-ratio`, `--smooth`, `--report`), blank-canvas + topology harness in `experiments/chain_probe/blank.py`
(`main`, `main3d`, `profile_check`).

### Sixth case — the **fast path** and where the time actually goes (2026-09-27, at the user's request)

The user's point: a chain joint only ever has two structures, so a **precomputed joint** should beat a 3-D render.
Built as `--fast`: rasterise one link in each of its two poses **once** (the sprite = coverage + normal field +
near/far split by the sign of the hit depth), then rotate/translate it per link with `cv2.warpAffine`, composing in
three passes (all far wires → all oval rings → all near wires). Same placement math as the 3-D path, so the two can
be compared pixel by pixel.

| what | measured (832×1216, 20 links) |
|---|---|
| 3-D tube + per-pixel z-buffer | 160 ms |
| `--fast` (2 sprites + ~30 warps) | **64 ms (2.5×)**; on big links (Wo 90) up to **3.3×** |
| agreement vs the 3-D reference | differences sit in a 1 px band along every edge/outline (AA-level); the linking number per pair is identical (0.99) |

**The lesson is where the time was NOT.** The first `--fast` version came out only 1.2× faster, and profiling showed
the cost was **not** the geometry: it was the *post-pass* — four Gaussian blurs, a normal normalisation, a
`digitize` + palette gather, AO and outline ops — all running over the full 832×1216 canvas for a chain that covers
**0.5 %** of it. Cropping every post-pass to the chain's bounding box (see `_finish()`), shared by both renderers,
took the 3-D path from 420 ms to 160 ms by itself. So: **before optimising the algorithm, check that the work is
proportional to the object, not to the canvas.**

---

## 1. What a human does in Photoshop (the checklist to imitate)

1. **Shrink + feather the mask** before compositing. Pulling the mask *in* by 1–2 px removes the halo of the
   source background (decontamination); feathering 4–10 px kills the razor edge. Never hard-paste.
2. **Match tone locally, not globally.** Sample the *surroundings* of the paste region (a ring band), then use
   Curves/Levels (or `Color Balance`) so the patch's black/white points and cast match that ring — not the whole image.
3. **Match grain.** Add `Filter ▸ Noise` (or Camera Raw grain) sized to what the surroundings have; on a
   *flat* area measure the noise, don't eyeball it.
4. **Match sharpness.** If the patch is softer, `Filter ▸ Sharpen`/High-Pass overlay; if it's sharper than the
   illustration, a light Gaussian (0.3–1.0 px) is usually right.
5. **Contact shadow / AO.** A dark band hugging the contact edge (mask painted outward, Multiply, soft brush),
   direction opposite the key light. This is what sells "it is standing on / inside something".
6. **Light wrap** on the edge (a thin rim of the base's light color over the patch border) for bright scenes.
7. **Blend gradients, not opacity**: where the patch's own illumination drifts, a gradient mask beats a uniform one.

## 2. The programmatic equivalents (what `harmonize.py` does)

| PS step | Code |
|---|---|
| shrink + feather mask | `erode` by `--shrink` px, then Gaussian feather `--feather` px |
| local tone match | LAB mean/σ transfer from the mask's **outer ring**: `--match lab` (all channels) / `--match luma` (L only) / **`--match chroma` (a/b only — keeps the object's own brightness)** / `--match hist` |
| sharpness match | compare Laplacian variance (ring vs patch) → unsharp or small Gaussian, clamped 0.4–2.5× |
| grain match | robust σ on **flat pixels only** (lowest-25 % gradient percentile) → add Gaussian noise, capped (`--grain-max`, default 5) |
| contact shadow | `--ao 0.25 --ao-width 18 [--ao-dir left]`: darken a falloff band just outside the mask |
| **cast shadow** | `--cast 0.38 --cast-len 14 --cast-blur 7 --cast-dir down-right`: offset the mask along the light's opposite direction and multiply the base — **this is what a hanging/standing object needs and what AO alone cannot give** |
| report | `seam_step` before/after = mean \|ΔL\| across the mask border (the number to optimise) |

```powershell
python scripts\harmonize.py BASE PATCH --mask MASK.png --out OUT.png `
    --match chroma --feather 3 --shrink 1 --ao 0.10 --ao-width 8 `
    --cast 0.38 --cast-len 14 --cast-blur 7 --cast-dir down-right --report
# BASE  = the finished illustration
# PATCH = the layer to embed (program render, matted cut-out, UI, AI extra), same canvas as BASE
# MASK  = white where PATCH should show (on the BASE canvas)
```

**Which `--match` for which object** (measured, see §5):
`lab` for a cut-out/photo/AI extra that must fully adopt the scene's look · **`chroma` for anything that is
*legitimately* brighter or darker than its surroundings (white paper, chrome, a lit screen, hair highlights) —
dragging its luminance to the wall turns white paper into an old scroll and makes the seam metric worse** ·
`luma` when the patch already carries the right colour (anime flat fills) · `hist` almost never.

## 3. Two lessons measured here

- **`--match hist` is wrong for a photographic/textured surround** (seam got *worse*: 54 vs 26). Histogram
  matching re-maps the patch's tonal distribution, which for a flat synthetic layer injects banding and a
  bright rim. Use it only when *both* sides are flat-colour artwork with the same dynamic range.
- **Grain must be estimated from flat areas.** Estimating σ from the whole ring gave 15.05 (the "noise" was
  actually desk-leg edges) and visible over-noising; the flat-percentile estimator gives 3.84 and a clean result.

## 4. What we do NOT try to do classically (and what to reach for instead)

- **Re-lighting the patch to a new light direction** — classical matching only transfers *statistics*; it cannot
  move a specular. For that: Blender (render the element under the scene's key light in the first place) or a
  low-denoise **masked repaint** (our `local_repaint_bigcanvas` / `patch_local.py` route: img2img at
  denoise 0.25–0.45 restricted to the seam) — the model re-draws the boundary with its own statistics.
  **Suggested order: classical match → masked low-denoise repaint → classical match again** (the last pass
  re-fixes the grain/sharpness the model changed).
- **Structural mistakes** (wrong perspective, wrong occlusion, wrong contact) — no harmonizer fixes those; see
  `planning_and_decomposition.md` (macro vs micro).

## 5. Anime-specific notes (our house style)

- Anime illustration is **flat fill + soft gradients**. That means:
  - **match luminance, keep the patch's own chroma** (`--match luma`) when the patch already carries the right
    local colour (e.g. a rope drawn in the scene's palette); use `--match lab` when the patch comes from a
    *different* image (cut-out, AI extra).
  - **gradients are the risk**: a flat dark patch in a soft gradient background shows a step. Prefer a gradient
    (non-uniform) weight map over a uniform feather when the base has a large luminance ramp.
  - **chroma noise is invisible, luminance noise is not**: if grain must be added, add it to L only.
  - keep line art **crisp**: cap the sharpness match so an inked element is not blurred into the painting
    (`--no-sharp` if the base is softer than the line work and you *want* the line to stay crisp).
- For a **UI overlay** (live-stream frame, dialog boxes): match tone/sharpness/grain the same way, but **do not
  feather the outer frame edge** — UI is meant to sit on top crisply; only the *illustration-facing* inner edges
  (e.g. a translucent panel) need softening.

## 6. External review — what a larger model said to change (ChatGPT, 2026-09-27)

Put to it: this pipeline, the measured numbers (seam 60.2 → 25.4, histogram worse at 54.2), and four questions.
Its verdict in one line: **the remaining "pasted" feeling is not a colour problem any more — it is
*painting language*, *lighting*, *edge style* and *layer topology*; keep tuning LAB/σ/Laplacian and returns fall off.**

**Priority order it proposes** (higher first): reconstruction of the **painting language** → **local lighting** →
**edges/contours** → **layer relations** → statistical harmonization → global noise/sharpness.

### The upgrades, with the parameters it gave

| # | Upgrade | How | Numbers |
|---|---|---|---|
| 1 | **Value-band (colour-step) matching — not mean/σ** | cluster the surrounding L into 3–5 bands (KMeans k=4 ⇒ shadow/midtone/base/light), then remap the patch with a **piecewise LUT** instead of `aL+b` | e.g. 0→0.02, .25→.18, .5→.48, .7→.68, .9→.86, 1→0.93 |
| 2 | **Edge-language rebuild** | treat α-seam and edge treatment as **two separate things**: extract the contour, then give it the background's edge style (hard / soft / **line art**) | edge width 0.5–1.5 px, opacity 0.3–0.8, blur 0.2–0.7 px; **do not feather 2–3 px on cel art** (it smears) |
| 3 | **Local colour bleed** | tint only a 1–4 px outer ring with the background's hue | strength 0.05–0.20, edge only |
| 4 | **Split AO into three** | **contact shadow** (obeys the key light!) + **cast shadow** + **ambient occlusion** | contact w 1–3 px / op 0.15–0.35; cast len 3–15 px / blur 1–5 px; AO r 2–6 px / op 0.08–0.25 |
| 5 | **Environment-colour field** | blur the background 30–100 px into a low-frequency colour map, add `k · env` to the patch | k = 0.03–0.12 (keep it subtle for anime) |
| 6 | **Low/high-frequency split** | match **low freq** (colour/lighting) separately from **high freq** (line/texture/grain) so a Blender part's real geometry is not smeared away | — |
| 7 | **2D relighting, quantized** | pseudo-normal from L gradients (`N = normalize(-∂xL, -∂yL, 1/s)`), then **quantize** `x = dot(N, L)` into light/base/shadow (+ smoothstep) — reads like an animator repainted the light, unlike continuous Lambert | thresholds .65/.15, or .7→1.0 / .2–.7→0.85 / <.2→0.60 |
| 8 | **Light-direction map / master lighting profile** | one 2D `L(x,y)` (or a YAML profile: azimuth, elevation, temperature, intensity, shadow_strength, shadow_softness) that **every** program element (rope, robot, UI, repairs) consumes | light ×1.08 +hue 5°, shadow ×0.72 −hue 8° |
| 9 | **Estimate the key light from the character** | mark 2–5 highlight/shadow points on the figure, fit a 2D lighting field, make all inserted elements follow it | — |

### Corrections on the learned-harmonizer question

- **The paper I was thinking of is the wrong one**: *Harmonizing Attention* (WACV 2025) is **texture-aware
  geometry transfer**, not foreground/background image harmonization.
- The right family: **FreePIH** (ACM MM 2024, *training-free painterly image harmonization*) and **TF-GPH**
  (AAAI 2025, *training-and-prompt-free general painterly harmonization*, official repo, **SD2.x, ~20–23 GB VRAM**
  ⇒ not runnable on this 6 GB laptop). It also warns that their benchmarks are "painting + photographic object",
  which is **not** our target ("looks like *this* anime illustration"), so they may not beat the deterministic route.
- **Cheapest thing to actually try: PIH** (Adobe, CVPR 2023, semi-supervised parametric harmonization) —
  there is a **ComfyUI-PIH** custom node: `Background + Foreground + Mask → harmonize`, strength adjustable.
- Suggested experiment matrix: `A` raw composite / `B` our classical pipeline / `C` FreePIH / `D` TF-GPH /
  `E` classical+FreePIH / `F` classical+TF-GPH, judged by **edge seam, low-frequency colour error, edge energy,
  human preference and identity preservation** — and with the hard constraint that **a harmonizer must not alter
  the structure inside the mask** (screws/gears/connections must survive).
- Its conclusion: prefer **deterministic harmonization + optional learned harmonization** over "a learned
  harmonizer does everything".

### Recommended pipeline order (harmonizer last, not central)

```
master lighting profile
  → geometry / topology check        (occlusion, contact, attachment graphs)
  → edge language
  → palette / value bands
  → low-frequency lighting
  → local environment colour
  → contact shadow → cast shadow
  → high-frequency / line treatment
  → noise / grain
  → composite
  → [optional learned harmonizer: PIH → FreePIH]
  → human check
```

### Benefit / cost ranking it gave

| tier | change | benefit | cost |
|---|---|---|---|
| S | unified key-light direction & colour temperature | ★★★★★ | very low |
| S | value-band / shading-structure matching | ★★★★★ | low |
| S | edge-language rebuild | ★★★★★ | low |
| A | split contact + cast shadow | ★★★★☆ | low |
| A | environment-colour field | ★★★★☆ | low |
| A | low/high-frequency split | ★★★★☆ | medium |
| A | pseudo-normal 2D relighting | ★★★★☆ | medium |
| B | PIH (ComfyUI node) / FreePIH / TF-GPH | ★★★☆☆–★★★★☆ | very low / medium |
| C | Diff-Harmonization | ★★☆☆☆ | high |
| C | histogram matching | ★☆☆☆☆ | already measured as worse |

**Closing point worth keeping**: "pasted" decomposes into four independent problems — **geometry** not of this
world (topology/perspective/contact), **light** not of this world (master lighting/relighting/shadows),
**painting language** not of this world (edge/value bands/palette/frequency), **material** not of this world
(texture/highlight/micro-detail). The classical pipeline above only touches the last third; the next real
investment is a **unified painting-language + lighting language** pass (i.e. our own *anime composite harmonizer*
taking FG + BG + mask + lighting profile).

## 7. Open items

- ~~Consult a larger model~~ **done 2026-09-27** (section 6). Next: implement the S-tier items (value bands,
  edge language, unified lighting profile) in `scripts/harmonize.py`, then A/B on the same classroom+chain case
  and re-check the seam metric; only then evaluate PIH as an optional last stage.
