# Locating things on an image — grids, ASCII maps, compare sheets, colour masks

"Where exactly is it?" is a **measurement**, not a judgement. Guessing a bbox costs a GPU run
at best and silently eats a neighbouring part (a finger, a hair strand) at worst. The whole
toolkit below is cheap: seconds of CPU, no model.

## 1. Grid overlay — get coordinates you can act on and hand to a human

Crop the region of interest, scale it 1–2.5×, and draw a labelled grid **in original
coordinates**:

```bash
python tools/crop_view.py（薄封装，自带） IMG.png out/grid.png 700,600,1250,1080 --scale 1.5 --grid 100
```

Rules that make it useful:

- **Label with original-image numbers**, never with panel-local pixels (the script does this).
  A label that needs mental arithmetic gets mis-read by both the agent and the human.
- A 100 px grid on a 1.5× view is the sweet spot: fine enough to name a spot, coarse enough to
  count at a glance.
- Use it **before** asking a human to paint: "the face is around x 840–960, y 780–950" turns a
  vague request into a two-stroke job.

## 2. ASCII map — see the *structure* of a region as text

When you need to know which pixels belong to the target and which to its neighbour, dump the
region as a classified character map (R = target colour, S = skin, `.` = background, `b` = blue
night, `?` = other):

```bash
python tools/ascii_map.py（薄封装，自带） IMG.png 740,380,1000,560 --step 4
```

This is how you discover, without opening an image viewer: that a "rope region" actually runs
`x 860–930` while the fingers start at `x 820`, that the wrist is 40 px further left than you
assumed, or that a colour rule is picking up the floor. `--step 4–6` is a good compromise
between resolution and readable width.

## 3. Compare sheets — judge a parameter ladder at a glance

```bash
# same region, N images stacked, each labelled (中文 labels OK)
python tools/sheet3.py（薄封装，自带） out/arm_3way.png 690,440,910,620 \
       --labels '原图,0.3,0.5' --scale 3.2 a.png b.png c.png
```

- **Same box, same scale, one sheet.** Comparing two full images by eye at conversation scale
  hides exactly the differences you are deciding on.
- `scripts/contact_sheet.py` (in this skill) does the many-thumbnails version for candidate
  batches.
- Keep the sheet as **evidence** in the project dir: it is what lets a later session see *why* a
  parameter was chosen. Delete the per-pass full-size images instead.

## 4. Colour masks — extract a region by channel separation

For "the red rope", "the gold trim", "the blue cloth" there is no need to ask a human:

```bash
python tools/red_mask.py（薄封装，自带） IMG.png mask.png \
       --hue-max 16 --sat-min 0.60 --val-min 0.35 --min-area 400 \
       --drop 1200,1150,1920,1700 --overlay overlay.png
```

**Do not use `R − max(G,B)`** (the obvious "redness" score). Measured on a night scene: a warm
brown wooden floor (R≈140 G≈100 B≈60) and a dark-red cushion both pass it — 3.86 % of the frame
was "red", including two ~30 k-pixel blobs of floor. **Use HSV**: hue near 0°/360° with
saturation and value floors, then filter connected components by **area** and by **position**
(`--drop` boxes for the cushion, the floor, a calico cat's orange patch). Result: 1.69 % →
0.697 %, exactly the rope's segments.

Two more rules learned the same way:

- **"Dark = the ink" fails on a night scene.** The paper itself is dark (median luminance 50
  inside the scroll), so a brightness threshold selects the paper along with the ink. Separate by
  *where* (inside the paper's quadrilateral) or by a tighter threshold plus a paper mask.
- **A colour mask's dilation decides whether a boundary structure survives.** Rope over a sleeve
  needs `dilate 6`; a mask that reaches 18 px into a hand will repaint the fingers. When the human
  cares about the neighbour, use the two-layer canvas instead (`human_in_the_loop.md`).

## 5. Detector boxes (YOLO / OCR) as *candidates*, never as ground truth

`vision_detect` returns pct/pixel boxes for text and objects. Useful to shortlist where something
is; **confirm with a 1:1 crop or an ASCII map before masking**, because a detector box for "the
swallow" happily includes the rope loop next to it, and OCR boxes for a stylised calligraphy
stroke are unreliable at small sizes.

## 6. Verify with numbers, not impressions

| Question | Measurement |
|---|---|
| Did anything outside the mask change? | `diff_report.py base.png out.png --mask m.png` → expect `out-mask >12 = 0.0000 %` |
| Did the edit actually do something? | same command, in-mask % (0 ⇒ the mask was reversed or empty) |
| Is version A sharper than B? | Laplacian energy: `|lap|` mean and P99 at the **same size** (e.g. 2000×1368: 6.08/86 vs 9.56/160 — a 57 % difference is real; a 5 % one is not) |
| Did the cloud result drift from the input? | 3.57 % of out-of-mask pixels >12, best alignment at (0,0) ⇒ tonal drift, not a shift |
| Is the mask the right size? | painted area %, bbox, and the overlay image — quote them in the reply |

These numbers are what make an edit reviewable by a human who cannot see 3-pixel differences in
a chat preview, and they are how you catch a silent no-op (a reversed mask, a flag that was
ignored, a crop that was in the wrong place).
