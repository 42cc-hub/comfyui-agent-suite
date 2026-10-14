# Laying N panels out as one comic page

Tiling a set of finished images into a single page that tells the story in order, with
straight or slanted dividers, no white margins inside any cell, and the important content
always visible. `libs/comic-layout/scripts/comic_layout.py` does the layout; `libs/comic-layout/scripts/letter_page.py` adds the
text. Verified on a 7-panel page, 1400×3200.

## The model

The page is subdivided by **recursively cutting one convex cell with one straight line**.
That single choice buys three things at once:

* arbitrary angles are legal — slanted dividers need no special case
* the union is exactly the page — no gaps, no overlap, ever
* reading order falls out of the index ranges

**Placement inside a cell**: `cover` scale (`max(bw/iw, bh/ih)`) makes the image fill the
cell completely, so there is never a white margin; whatever overflows the polygon is clipped
away. The crop window is then slid so the key region sits inside it, centred as far as the
image edges allow.

**Key regions**: the region that must stay visible, per image. A sidecar
`<stem>.keymask.png` overrides automatic derivation.

> **Automatic derivation is unreliable and must be eyeballed.** Tile variance alone fails on
> a park photo — trees, grass and path are all high-variance, so their combined bounding box
> spans nearly the whole frame and the constraint becomes vacuous. Taking the largest
> connected cluster of high-variance tiles fixes that, but **for a panel holding two people
> it still cannot say which is which**, because that is a semantic question, not a
> computational one. Ask the human.

## Reading order is the trap

The layout can tile perfectly and still tell the story out of order. Two wrong ways to check:

1. **Pairwise ordering at each cut is not enough.** Ordering the two halves of every cut
   gives contiguous index ranges, which feels sufficient and is not:
   ```
   vertical cut first  -> left [0..3] | right [4..6]
   then horizontal cuts -> row1 = left-top(0,1) + right-top(4,5)
                           row2 = left-bottom(2,3) + right-bottom(6)
   reads 0,1,4,5,2,3,6          <- contiguous pairs, scrambled page
   ```
2. **Banding by vertical overlap passes wrong layouts.** A cell at y 1982–2800 overlaps a
   tall cell at 1125–2800 by more than half the smaller height, so the two get merged into
   one band and the sequence looks fine — while a reader, going by where each cell *starts*,
   sees a different order. **Band by TOP EDGE**, with a tolerance of ~15% of page height.

Then require, walking bands in order and cells left-to-right within a band, that each cell's
index range ends before the next one begins.

## Objective terms

| term | why |
|---|---|
| key region does not fit the visible window | heaviest; the soft form of a hard constraint |
| key region off the cell centre | `placement` can only centre it as far as the image edges allow, so a key region near a border comes out visibly off-centre |
| fraction of the image cropped away | how much of the picture is thrown away |
| cover scale over fit scale | over-zooming |
| sliver cells | extreme aspect ratios read as mistakes |
| **deviation from each panel's own target area share** | panels are not equally important; the two shoe close-ups are inserts and read fine at a fraction of the establishing shot's size |
| **reserved empty area** | room for dialogue and narration |
| edges away from four | 3- and 5-sided cells are welcome, not free |

Two objective-shape traps, both measured:

* **Averaging the area deviation lets a local disaster hide.** One badly-sized cell among
  good ones averages away — use the SUM.
* **A one-sided blank-area cost drives it to zero.** Use a target band: pay below it (or
  there is nowhere for text) and pay much more above it, with the slope steepening
  (`w*over + quad*over²`) so "slightly over" is not a cheap buy.

### Reserved space must touch the page edge

A blank cell in the middle reads as a **misprint (a hole)**; the same area along an edge
reads as a **text margin**, which is what comic pages actually do. Enforce it as a
constraint on the blank cell, not a penalty.

### Distinguish "lazy" from "infeasible" with a probe

The optimiser left 13.8% blank against an 8% target. Before rewriting the objective, raise
the blank penalty ~700x and re-run: **it reached 7.7%**, so a near-gapless layout existed and
13.8% was a *trade* the search had chosen, costing only ~0.4 energy. The hypothesis "the
constraints make it infeasible" was wrong, and reworking the reading-order rule would have
been wasted work.

**Do this probe before changing a constraint.** "I think it is impossible" is a claim to
falsify, not to act on.

## Page aspect is part of the design

Every term is a normalised ratio (overflow, waste, zoom, thinness, centring offset, area
share, blank fraction), so **the energy is comparable across page sizes** — which is what
makes sweeping them meaningful.

No single best ratio, but the sweep is decisive (7 panels: three portrait 2:3, three square,
one landscape):

| page | energy | blank |
|---|---|---|
| **1400×3200 (1:2.286 strip)** | **8.26** | 7.7% |
| 2080×2560 (1:1.231) | 14.95 | 7.7% |
| 2000×2600 (1:1.300) | 15.71 | 6.6% |
| 2100×2970 (A4) | 16.88 | 7.7% |
| 2300×2300 (square) | 21.55 | 5.5% |

The tall strip wins because the portrait panels stack along the page and each cell's aspect
ends up close to its image's, so both crop waste and zoom drop. A near-square page forces
portrait images into squarer cells and pays for it. **The right answer depends on where the
page is read** — a tall strip is a webtoon, not a printed page.

## Text: balloons are shapes, not decorations

### Build the balloon as ONE union mask

Draw the ellipse and the tail into the same filled mask (with the **tail's base INSIDE the
ellipse**), then derive the border from that mask by morphology:

```python
border = mask.filter(ImageFilter.MaxFilter(5))
inner  = mask.filter(ImageFilter.MinFilter(5))
edge   = ImageChops.subtract(border, inner)
```

Two separately-outlined shapes can never join, so the tail appears detached and has to be
drawn long to reach the speaker. A union cannot come apart, and the tail can be short.

### Supersample the outline

A thresholded edge band is a staircase. Build the geometry at 2x and downsample with
LANCZOS; the outline then carries a real alpha edge. Only the geometry needs it — draw the
text at page scale.

### An ellipse only contains a rectangle ~0.71 of its extent

For an ellipse `(A,B)` to contain a text box `(w,h)`: `(w/2A)² + (h/2B)² ≤ 1`. With
`A = w/2·infl`, `B = h/2·infl` this reduces to **`infl ≥ √2 ≈ 1.414`, independent of aspect
ratio**. Anything less and a two-line box always spills, no matter the tuning. A cloud whose
radius dips to `(1−k)` needs `√2/(1−k)`.

### A thought cloud is a cycloid, and the cusp must be at the SMALLEST radius

`r = 1 − k·(1 − |cos(nθ/2)|^p)` — round bulges outward, sharp re-entrant notches between
them. Small harmonics keep it from reading as a machined gear; an anisotropic stretch fits
it to the text box so it stays a cloud instead of becoming a circle.

> Using `|sin|` instead puts the derivative discontinuity (the cusp) where `|sin| = 0`, which
> is the **largest** radius — spiky bulges with smooth notches. The shape's whole character
> is inverted by picking the wrong extreme, and it is not obvious until it is rendered.

### Anchors beat an ordering rule

Each balloon belongs next to its speaker. Make the anchor a **soft target** (ideal position =
just above the anchor) rather than a fixed start, so the search can trade it against
containment and key coverage instead of being pinned.

With anchors, **order needs no rule at all**: the balloons sit where the characters are, and
the characters are already in reading order. The global "balloon tops must not decrease"
rule that this replaced was in permanent conflict with the geometry — a tall cell's balloon
lands below the next panel's entire cell, so tightening it made that panel impossible and
loosening it let captions invert. **No amount of tuning fixes a rule that fights the layout.**

Supply anchors from a human, and label the drawn text with no speaker name — the position
already carries that, and the name only makes every balloon wider.

## Working with a human on the annotations

Both key regions and balloon anchors are **semantic judgements the machine cannot make**.
`human_canvas` handles them efficiently: put every panel on one labelled contact sheet
(recording each panel's rect so painted pixels map back), or seed the vector nodes with
computed guesses so the human only has to *drag* rather than create from scratch.

The split that worked: **the human says who is where and what must not be cropped; the
machine does the coordinate mapping, the solving and the rendering.**

## Files

| Script | Role |
|---|---|
| `libs/comic-layout/scripts/comic_layout.py` | layout search + renderer. CLI: `comic_layout.py OUT.png IMG... --page WxH --beam N --weights 1,0.6,...` |
| `libs/comic-layout/scripts/letter_page.py` | balloon shapes, text-box search, anchor solving. Imports `comic_layout`; reads `candidates/anchors/anchors.json` and `<image>.keymask.png` sidecars |

## Process note

The layout was built through ~10 numbered revisions, and most of the bugs came from the
**patch scripts used to apply the changes**, not from the algorithm:

* `str.replace` and slice concatenation **fail silently**. A patch that changed nothing still
  compiles and still prints "patched".
* A slice with `START`/`END` taken from two `index()` calls that are in the **wrong order**
  re-appends the region it meant to delete. The result defined the NEW renderer first and the
  OLD one second, and Python takes the last definition — the fix was live in the file and
  dead in the program, and the "no visible change" was reported as a rendering result.
* **Assert every substitution.** The one patch that did (`assert old in t`) succeeded first try.

Verify a patch by its effect, never by its exit status.


---

## Comic page — tiling N panels into one page（自入口 SKILL.md 搬入 2026-10-13）

> 来源：`SKILL.md` 的同名章节，2026-10-13 入口瘦身时**整段**搬来，正文未改（只调了标题层级与交叉引用）。


When the deliverable is a **page** rather than separate images: one image, N cells, straight
or slanted dividers, no white margin inside any cell, key content always visible, reading
order preserved. Two scripts:

| Script | Role |
|---|---|
| `libs/comic-layout/scripts/comic_layout.py` | layout search + renderer. `comic_layout.py OUT.png IMG... --page 1400x3200 --beam 12 --weights 1,0.6,1,0.6,1,1,1` |
| `libs/comic-layout/scripts/letter_page.py` | balloon shapes, text-box search, anchor solving (imports `comic_layout`) |

Design in one line: **recursively cut one convex cell with one straight line** — arbitrary
angles become legal, the union stays exactly the page, and because a cut hands range
`[lo..k]` to one side and `[k+1..hi]` to the other, reading order is structural. Each cell is
then filled with a `cover`-scale crop of its image (so it can never letterbox) slid to keep
that image's **key region** inside.

Full playbook — the objective terms, the two ways of getting reading order wrong (pairwise
ordering, and banding by vertical overlap), the blank-area probe, and the balloon geometry:
**`references/comic_page_layout.md`**.

Three things worth knowing before starting:

- **Ask the human for two annotations, not one.** Key regions *and* balloon anchors are
  semantic judgements — for a panel holding two people, tile statistics cannot say which is
  which. `human_canvas` with a labelled contact sheet (or pre-seeded nodes the human only
  drags) makes this cheap.
- **Probe before changing a constraint.** The search left 13.8% blank against an 8% target;
  raising the penalty 700x reached 7.7%, so it was a trade, not an infeasibility. Reworking
  the constraints would have been wasted work.
- **Every objective term is a normalised ratio**, so energy is comparable across page sizes
  and sweeping them is meaningful — on a 7-panel set a 1:2.29 strip scored 8.26 against
  14.95 for the next best, because portrait panels stack naturally on a tall page.

