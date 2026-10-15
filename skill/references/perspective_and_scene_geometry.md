# Perspective recovery & scene geometry (from a human's line annotation to control maps)

For the job "I have a reference illustration of an interior and I want **controlled** depth/line
control maps, or the camera of that reference so a second object can be placed at the right
scale". Every number below is tagged `[measured]` (our own runs) or `[observed]`, and comes from
one convenience-store illustration set.

## 1. Conventions (agree them with the human before writing code)

| | axes |
|---|---|
| **image** | x right, y down, z into the screen — `u = cx + f·X/Z`, `v = cy + f·Y/Z` |
| **world** | x forward (down the aisle), y left, z up (ROS body frame) |
| **pose** | `p_cam = R·(p_world − C)`, `R = expm([r]×)·R0`, `R0` maps world (fwd,left,up) → camera (right,down,fwd) |
| **families** | `x` = depth, `y` = across, `z` = vertical |

Parameterise the rotation as a **rotation vector** applied to `R0`, not as free Euler angles:
it removes the mod-360 equivalent branches that let an optimiser report `pitch = −424°`.
Report the pose as the camera's forward/up vectors in world plus yaw/pitch/roll — those are
readable; raw Euler triples are not.

## 2. Fitting the camera from annotated lines — the general solution

The human draws **any number** of line segments and labels each with its family (that label is
an *assertion*, which is exactly what makes a hard constraint safe). Unknowns are
`[log f, r1, r2, r3]` (principal point weakly prior-ed to the centre).

Residual per line = the angle between the image line and the direction from its midpoint to
that family's projected vanishing point `v_k = K·R·e_k`. Two properties make it the right
residual: it is scale-free (a VP thousands of px outside the frame behaves), and it stays
finite when the direction is perpendicular to the optical axis (**VP at infinity** — the normal
case for a zero-pitch camera's verticals). Guard that case explicitly:

```python
if abs(D[2]) > 1e-6: w = vp_xy - midpoint          # finite VP: use its position
else:                w = D[:2]                     # VP at infinity: use the direction
r = atan2(cross(d, w), dot(d, w)); r = (r + pi/2) % pi - pi/2   # lines are undirected
```

Without the guard the division by ~0 produces NaN and a Jacobian singular value of `7e8`
`[measured]`; with it, the same fit is clean.

**Weights and losses — the split that matters** `[measured]`:
- **Human lines: plain L2** (weight 1 each). With a Cauchy loss the optimiser *abandons* one of
  them for free — a 220 px labelled line settles at a 37° residual while the cost
  stays ~0, because there is no other data to pull against.
- **Machine-detected lines: Cauchy via IRLS** (drop them if their best residual stays above
  ~5–12°). They are the noisy majority and must never outvote the human's constraints: with
  length weighting alone, a long detected edge (600 px) counts ~2.7× a short human line (220 px).
- **Priors must be weighted like a line.** With the focal prior scaled by 0.02 the solver drifted
  to `f = 285 px` (h-fov 122°) to buy a slightly better line fit `[measured]`; at weight 1 a
  0.35 log-sigma prior holds.

**Observability is the headline output, not the camera.** Two cheap reports:
- SVD of the residual Jacobian → rank and the **null direction** (which parameter combination
  the annotation does not determine).
- A **focal sweep**: pin f at a grid of values, re-solve the rotation, print the cost.
  A flat curve = f is not observable; the useful by-product is how far the rotation slides.

`[measured]`, one scene:
- **One wall corner** (vertex + vertical + two perpendicular horizontals = 3 lines, one per
  family): exact fit (0.00° on all three) at **every** focal from 30° to 90° — rank 3 of 4,
  cost ~1e−30 across the sweep. Only the pitch was observable (≈ +28° down); roll and yaw slid
  (+9.5°…−1.9° and −30°…−39°). A single corner also has a **discrete mirror branch** (multi-start
  landed in it at 45°/80°), and the two branches can differ by 0.01% in cost.
- **Adding one more line in any family** pins that family's VP and makes f observable (rank 4/4).
- **Two nearly-parallel lines of the same family buy almost nothing**: their "VP" is at infinity,
  so the pair adds ~1 weak constraint. Ask for lines that **fan out in the image** (finite VP);
  for a floor, a near seam and a far seam, not two adjacent tiles.

## 3. Assume the drawing may not be a projection — and say so with numbers

`[measured]`, the same illustration: the case's two shelf-edge families (2280 px at −34°,
1179 px at −24°) intersect at **(1657, −363)** — off-frame, above it. If those are horizontal
3D lines their VP is on the horizon, so the horizon would sit 363 px above the frame ⇒ pitch
≈ 48° down. But the fixture verticals (11 segments, −85.5°…−86.7°) are parallel within 4° ⇒
pitch ≈ 0. Both cannot hold. Feeding 40 machine-detected lines into the robust fit did not
rescue it: the best solutions pinned roll/pitch at the parameter bounds and still left a
**median residual of 12°** (6–14 of 40 lines within 5°).

So the deliverable when the input is squishy is **"largest consistent subset + explicit
outliers + per-line residuals"**, never a single forced camera. Two practices make that usable:
- **Colour the human's lines by residual** on top of their image (green < 3°, yellow < 12°,
  red above) and hand that back — it is the actionable form of "your annotation disagrees with
  itself", and it tells them *which* line to re-trace or re-label.
- **Let a category be an outlier**: in this scene the cold-case shelf bands were labelled `x`
  (semantically right — they do run along the aisle) but the drawing's angles contradict the
  verticals, so keeping them in forces a compromise camera that satisfies nothing. Dropping
  them (`--drop e4,e5`) took the verticals from ~7° to ~4° and made the render usable.

## 4. Geometry → control maps (the part that actually drew the picture)

Model these interiors as a **wall corner**: floor `z=0`, a wall **parallel** to the aisle
`{y = Y_L}` (the refrigerated case, its base line a depth line) and a wall **perpendicular** to
it `{x = X_R}` (the produce shelf, its base line an across line). The two meet at the corner;
the camera sits inside the quadrant `{x < X_R, y < Y_L}`. Ray-cast the three surfaces (nearest
positive hit, per-pixel plane id) → depth (near = white) **and** a plane-id colour map, which a
human reads far more easily than grey.

Getting `Y_L` / `X_R` out of the human's painted region masks `[measured]`: take the **bottom
boundary** of a region mask, run each pixel's ray down to `z=0`, and take the median of the
resulting world coordinate — `y` for a y-wall, `x` for an x-wall.
- Left case: **y = +1.94 m** from 796 junction samples — usable directly.
- Right shelf: **x = 12.6 m** from 105 samples, but that boundary sits mid-shelf (the base is
  hidden behind the character), so the number is biased far. Cross-checks disagreed in the same
  direction: the human's `y`-labelled line, read as a floor line, implies 45 m and the corner
  vertex 9.5 m. ⇒ treat the perpendicular wall's distance as **the one tunable knob**, and check
  the rendered wall region against the painted mask before spending GPU time.

Two more lessons from this step:
- **A line above the floor is not a floor line.** A shelf edge at z ≈ 1 m looks plausible in the
  annotation and projects to a nonsense distance (45 m here) if you assume it lies on `z=0`.
  Say which lines are *base* lines in the annotation brief; they are the ones worth drawing.
- **Repaint the depth per painted region** from each plane's equation (every pixel inside a
  painted region carries exactly that plane's depth) instead of using the raw render; it removes
  the un-hit black zones and makes the control map agree with what the human drew. Fill the
  character's region with the **background behind her** — that is what a background pass needs.

## 5. Control routing — pair the control model with its own family

- **Anima has no depth ControlNet.** Its LLLite set is pose / scribble / inpainting
  (`anima-lllite-*`). The `noob_sdxl_controlnet_depth` entry that appears in the same
  `lllite_name` dropdown belongs to the SDXL/NoobAI family and is meaningless on Anima's
  qwen-image stack. `[observed]` Anima + LLLite scribble (our lineart, strength 0.8) did produce
  a store, but read the layout as a long corridor and with much less detail.
- **Depth route on this host = Illustrious + `noob_sdxl_controlnet_depth.safetensors`**:
  `[measured]` 832×1216, strength 0.85, dpmpp_2m/karras, 28 steps, cfg 6.5, **111 s** — visibly
  cleaner and far more detailed than the Anima scribble pass, and it followed the depth map's
  near/far structure.
- **Control images must match the sampling canvas.** Resize the control to the exact W×H you
  sample at (the 1024×1536 depth was resized to 832×1216). SDXL anime likes ~1 MP; do not sample
  a 1.57 MP portrait just because the depth map is that size.
- **`LoadImage` paths are relative to ComfyUI's `input/`.** Stage the control into
  `input/dsh_ctrl/` and pass `dsh_ctrl/<file>`; a bare filename is an HTTP 400 `[measured]`.
  `LoadImage`'s MASK is output **index 1** and equals **1 − alpha** — write the mask into alpha
  inverted or the repaint silently does the complement.

## 6. Two-stage compositing must pin ONE common scale anchor (the giant bug)

`[measured]` The character's pixel size came from the human's mask (traced on the reference);
the background's implied scale came from the solved geometry. Nothing forced them to agree, and
the result was a **2.30× giant**: at her feet the floor was 2.01 m away, a squatting 0.95 m
figure should paint `f·H/d = 391 px`, the mask painted 901 px.

The fix is three lines of arithmetic and one flag:

```
d       = ray_through_feet ∩ floor   (metres, from the solved camera)
h_expect = f_work * H_char / d        (H_char = the human's number: 0.95 m squat)
factor   = h_expect / h_mask          -> zoom the background about her feet by 1/factor
```

Zooming the **background** (about her feet, keeping that pixel fixed) is the default: it makes
the scene scale grow so she becomes the right size *in it*, and it tightens the framing toward
the reference's (a close-up). The alternative — shrinking her mask — keeps the wide background.
Whichever you pick, print the measured factor; it self-corrects for any mask/base pair.

## 7. The finisher: whole-image img2img at denoise 0.6, no ControlNet

`[measured]` One 30 s pass unified a two-stage composite: the inpaint seam disappeared and the
lighting/colour came together. The same pass run on the reference itself looked even better —
but that is the *ceiling*, not a reconstruction: at 0.6 the reference's own
layout and props survive (its shopping basket, its framing), so it proves only that "layout right
+ 0.6 repaint" is a strong finisher. Use it as the **last** step, after the layout is settled, and
do not mistake it for recovery.

## 8. Toolchain (all under `agent_work\pixiv_publishing\experiments\bg_control\`)

| script | role |
|---|---|
| `measure_lines.py` | Canny + HoughLinesP, angle histogram, LSQ vanishing point, detection overlay |
| `annotate_vp.py` | draw the measured families extended to their VP (the "this drawing is not a projection" evidence figure) |
| `grid_overlay.py` | labelled coordinate grid, for reading structure off an image |
| `manhattan_solve.py` | family classification (angle), hard/soft solve, EM re-assignment, global focal sweep, inlier scoring |
| `solve_render.py` | the solver in the ROS convention (hard human lines, weighted verticals, roll/pitch priors) + wall-corner ray render (depth + plane id) + residual-coloured fit overlay |
| `mask_control.py` | repaint depth per painted region, draw the structural lines clipped per surface |
| `run_bg.py` / `run_bg_il.py` | background passes (Anima+LLLite scribble / IL+depth ControlNet) |
| `run_char.py` | character pass with the scale self-check (`--fix bg|char|none`, `--char-h`) |
| `run_img2img.py` | whole-image img2img finisher (`--denoise 0.6`) |
| `submit_diag.py` / `wait_hist.py` | print the server's 400 body; wait for a queued prompt and fetch the image |

Everything renders through `PreviewImage` (`type=temp`), so nothing lands in `output/`.

## 9. Checklist

1. Agree the axes; parameterise rotation as a rotation vector.
2. Ask for annotated lines **with family labels**, and for **≥2 lines in at least one family that
   fan out** — that is what makes the focal length observable.
3. Human lines hard, machine lines robust, priors weighted like a line.
4. Report rank/null space + focal sweep + per-line residual map → hand the map back to the human.
5. Build the scene as a wall corner; take wall offsets from the mask's **bottom boundary**, and
   treat the occluded wall's distance as a knob.
6. Repaint the depth per region; fill the character's region with the background behind her.
7. Pair the control model with its family (Anima: scribble/pose/inpaint; depth → Illustrious).
8. **Before compositing a character, match the scales** with the anchor arithmetic; print the factor.
9. Finish with one whole-image img2img at 0.6 if the stages need unifying.


---

## Recovering perspective — a human's line annotation → camera → control maps



Full playbook, measured numbers and the toolchain: **`references/perspective_and_scene_geometry.md`**.
The shape of it, in the order that matters:

- **Agree the axes first.** image: x right / y down / z into the screen (`u = cx + f·X/Z`); world:
  x forward / y left / z up (ROS); `p_cam = R(p_world − C)`; families `x`=depth, `y`=across,
  `z`=vertical. Parameterise the rotation as a **rotation vector on a fixed baseline `R0`**, never
  as free Euler angles (they let an optimiser report `pitch = −424°` and call it a fit).
- **The human labels each line with its family — that is what makes a hard constraint safe.**
  Human lines: plain L2, weight 1. Machine-detected lines: Cauchy via IRLS, dropped above ~5–12°.
  A Cauchy loss on *human* lines lets the optimiser abandon one for free (a 220 px labelled
  line can park at a 37° residual at cost ≈ 0); and priors must weigh like a line, or f drifts to
  the bound to buy a marginally better line fit (f = 285 px, h-fov 122°).
- **Residual** = angle between the line and the direction from its midpoint to that family's
  projected VP. Guard the **VP-at-infinity** case (direction ⊥ optical axis) explicitly, or the
  division by ~0 poisons the whole solve.
- **Report observability, not just a camera**: rank + null direction of the Jacobian, and a **focal
  sweep**. Measured: a single **wall corner** (3 lines, one per family) fits *exactly* at every
  focal from 30° to 90° — f completely unobservable, only pitch (≈ +28°) is; **one extra line in any
  family** pins that VP and makes f observable; two near-parallel lines of one family add almost
  nothing (their VP is at infinity), so ask for lines that fan out.
- **Assume the drawing is not a projection, and prove it with numbers.** The case's shelf-edge
  families imply a 48° down-pitch while the fixture verticals
  imply ≈ 0°; with 40 detected lines the robust fit still pinned roll/pitch at the parameter bounds
  with a 12° median residual. When the input is squishy, deliver **largest consistent subset +
  outlier list + a per-line residual overlay** (green/yellow/red), never a forced camera.
- **Build the scene as a wall corner** (floor `z=0`, one wall `{y=Y_L}` parallel to the aisle, one
  `{x=X_R}` perpendicular). Take each wall's offset from the **bottom boundary of the human's paint
  mask** (ray → floor, median of the world coordinate): 796 samples gave `Y_L=1.94 m`; the occluded
  wall's number is biased far (12.6 m) and is **the knob**. Lines *above* the floor are not base
  lines — assuming they lie on `z=0` gave a nonsense 45 m.
- **Repaint the depth per painted region** from each plane's equation (and fill the character's
  region with the background behind her); also emit a plane-id colour map — humans judge that far
  better than grey.
- **Pair the control model with its family.** Anima has **no depth ControlNet** (only LLLite
  pose/scribble/inpaint; the depth `.safetensors` in that dropdown is SDXL/NoobAI). Depth →
  **Illustrious + `noob_sdxl_controlnet_depth.safetensors`** (832×1216, strength ≈0.85, 111 s,
  visibly better than Anima+scribble here). Control images must match the sampling canvas; stage
  them in `input/dsh_ctrl/` and pass `dsh_ctrl/<file>` (a bare filename = HTTP 400); `LoadImage`'s
  MASK is output index 1 and equals **1 − alpha**.
- **Two-stage compositing must pin ONE scale anchor.** The character's pixel size comes from the
  human's mask, the background's scale from your geometry — nothing ties them, so a missing anchor
  gave a **2.30× giant**. Fix: `d` = ray through her feet ∩ floor; `h_expect = f·H_char/d`;
  `factor = h_expect/h_mask`; zoom the background about her feet (default) or shrink the mask, and
  **print the factor**.
- **Finisher**: one whole-image img2img at **denoise 0.6, no ControlNet** (30 s) removed the
  two-stage seam and unified the lighting. Run on our composite it gave the best image *of our own
  pipeline*; run on the reference it looked even better but that is the
  **ceiling, not a reconstruction** — at 0.6 the reference's own layout and props survive.

