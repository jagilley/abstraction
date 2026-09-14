# tiles_twin — design record

**Up**: [`../SPEC.md`](../SPEC.md) (plant) · **Files**: [FILES.md](FILES.md)
**DGP**: [`canvas/tiles.py`](../../tiles.py), used unmodified — the hidden-grid tile grammar
with an exact oracle, kept from the start as the calibration twin for
"misalignment between a hidden grid and the VQ grid, measured exactly"
([`canvas/CLAUDE.md`](../../CLAUDE.md)).
**Pipeline**: `canvas.plant.plant` reused **unchanged** for all five core readouts. This node
adds a corpus and the oracle columns; it changes nothing the agent sees.
**Idea doc**: [`style_practice_substrate`](../../../../ideas/style_practice_substrate.md) §3a,
§4, §5, §9 (all three risks are what `pl0` hit).
**Status (2026-08-25)**: complete. Corpora built (36 schools × 128 swatches × 2 conditions,
96 MB each); `twq_*` + `tw_*` + `oracle_run` + `panel_dump` all landed for both conditions;
numbers reduced by [`analyze.py`](analyze.py) and [`../twosided.py`](../twosided.py).
**No README** — numbers are with the coordinator for discussion.

## The question

[`pl0`](../SPEC.md) measured three preconditions on the GLSL library and all three came back
negative *in the same direction*:

1. 2×2 code-block recurrence tracked per-style **code entropy** (Spearman −0.836), not part
   structure; the designed part-y/texture-y control did not separate at all (0.130 vs 0.202,
   wrong way round), and the styles that recurred most were the translation-invariant ones
   (`wood_grain` 0.677, `sediment_strata` 0.613) while visibly-part-y tilings sat at the floor
   (`truchet_meander` 0.013, `girih_stars` 0.021).
2. Search bought nothing: 32× the forward passes moved exact code recovery by ≤ 0.013, and at
   full-grid masks the plant matched a style-marginal-argmax filler exactly (0.119 vs 0.124).
3. The mean-NLL grade of record passed the plant's own fills at 0.997–1.000 — *above* genuine
   held-out exemplars at 0.869 — and got worse as the grader got better.

Two candidate causes, and this node separates the first from the substrate:

- **Phase misalignment.** `lib0` swatches are crops at random pixel offsets, so the same motif
  spells differently every time and no code tuple can recur unless the style is translation
  invariant. If that is the cause, an *aligned* corpus should recover part structure and a
  *misaligned* one should reproduce `pl0`'s null — same styles, same everything else.
- **The grader's shape.** Mean per-token NLL rewards the marginal mode. That is
  [`../twosided.py`](../twosided.py)'s question, answered offline on `pl0` itself, and it is
  re-asked here against *known* validity.

## The one variable

A swatch is a 256×256 crop of a 320×320 render of a 10×10 tile grid at 32 px/tile. The two
conditions share the same 36 schools, the same tilings, the same renders, the same seed bands
and the same split. They differ only in where the crop starts:

| condition | crop offset | consequence |
|---|---|---|
| **aligned** | a multiple of 32 px | the 16-px code grid **nests** in the tile grid: a tile is exactly a 2×2 code block, a 2×2-tile motif exactly 4×4 — `tiles.py`'s own level structure, made literal |
| **misaligned** | drawn so `offset mod 16 ∈ [4,12]` on **both** axes | no code cell coincides with a tile; every tile straddles four of them. `lib0`'s condition, by accident |

The misaligned offset is drawn from a restricted set rather than uniformly on purpose: a
uniform draw would land tile- or code-aligned a sixteenth of the time and blunt the contrast.
That choice is the whole treatment, so it is stated rather than hidden.

## The corpus

**36 styles = 4 preset tilesets × 9 schools**, matching `lib0`'s style count exactly so the
plant's capacity, the grader's corpus ladder and every step count are unchanged. A `School` is
a demand-concentration over the tileset's valid set (memo §6, literally: weights, neighbour
affinities built from sampled 2×2 motifs, palette, stroke geometry). Nine schools on one
tileset is **style drift on a fixed truth** and a different tileset is **truth drift** — both
are in the corpus for later nodes and neither is used here.

Splits reuse `canvas.plant.corpus.SPLITS` verbatim (seed bands 0–63, 2 crops each,
128 swatches/school), so `plant_train`/`gA_train`/`gB_train` mean the same thing as in `pl0`.

## The oracle, written down and never fed to the agent

- Per swatch: the crop offset, and (aligned only) the exact 8×8 tile ids of the window.
- Per school: the tileset's edge table, the demand weights, and a 32-px render of **every tile
  in the catalogue** — which is what lets a decoded completion be classified back to tiles.
- Per `plant_test` tiling, at each of the three nested levels (1×1, 2×2, 4×4 tiles = 2×2, 4×4,
  8×8 codes when aligned), `tiles.damage` gives two classes with **known** validity:

  | class | what it is | oracle |
  |---|---|---|
  | `seam` | region re-sampled ignoring the surround: every tile legal, every interior edge agreeing, the inconsistency only in the parent region | **invalid** (measured: 0/... valid) and locally unsuspicious |
  | `offstyle` | region re-sampled respecting the surround under a uniform demand | **valid** but atypical — the exact `alt_valid` class |

  757 instances across the 36 schools, rendered from the same 320×320 pipeline and cropped with
  the *same two offsets*, so the panel is paired across conditions.
- `tiles.describe()` per school ([`describe_schools.py`](describe_schools.py)): edge-pattern
  coverage, single-tile repairability and greedy dead-end rate per level — the DGP's own depth
  margins, which the cost-to-depth curve is read against.

## What is measured

The five core readouts come from `plant.run` unchanged, once per condition. On top:

1. **Recurrence against a known answer** (`twin.oracle_run`). In the aligned condition every
   tile position carries both a true tile id and the code block that landed on it, so two
   purities say whether the codebook resolved the grammar's unit: `block→tile` (of the
   occurrences of one code block at support, what fraction share a tile id) and `tile→block`.
   A codebook that recovered the tile grid scores ≈1 on both. The same one level up, over
   2×2-tile motifs, against the school's own motif set. This readout is **aligned-only by
   construction** — a misaligned crop straddles tiles, so no code block has a tile id to be
   pure with respect to. The control for it is the *learned* recurrence measured identically
   in both conditions by `plant.run` (T[2] mass at support, T[3] all-halves-at-support), which
   needs no oracle.
2. **Grader calibration against known validity**: pass rates on `seam` (should be rejected) and
   `offstyle` (should be passed), one-sided and two-sided, per level. `lib0` could only guess
   which of `shuffle`/`marginal` was which.
3. **The plant's fills, validated**: decode a completion to pixels, classify each 2×2 code
   block to the nearest rendered tile, run the edge-agreement test. The direct gaming readout —
   whether what the grader passes at ≈1.00 is actually in the grammar. A clean swatch put
   through the same round trip is the ceiling, and is reported beside it.

## Scope conditions

- Single seed. Ranks, signs and the aligned–misaligned contrast are the currency.
- The tile classifier is nearest-neighbour in pixel space against the school's own atlas; its
  agreement with truth on the *unmasked surround* is reported as its own ceiling, so a low
  validity number cannot be read as a classifier artefact without checking that column.
- `n_violations` is computed from the tileset's edge table directly rather than through
  `Tileset`, so nothing on the agent's path imports `tiles.py`.
- The misaligned condition has no per-swatch tile grid (a crop straddles tiles), so readout 1
  is aligned-only by construction; readouts 2 and 3 exist in both.

## Tail re-analysis of the grade of record (`tailgrade.py`, 2026-08-25)

An offline re-scoring of the same known-validity panel with **tail statistics of the per-token
NLL** instead of its mean. Nothing was retrained and nothing was re-sampled: `twin.panel_tokens`
replays `oracle_run`'s panel and calibration slices against the saved weights and writes out the
per-token array that `grader.score` averages away. `np.nanmean` of what it dumps reproduces
every `nll` in `oracle.json` to five decimals, which is the check that the replay is the same
object.

### Why

The grade of record passes the **invalid** class more often than the **valid** one at L2/L3
(aligned, gA24, 8-forward: `seam` 0.639 / `offstyle` 0.604, then 0.542 / 0.264). The standing
reading is a shape mismatch: validity here is a **conjunction** — every shared edge must agree —
and a mean is an **average**. `seam` is locally legal everywhere except the region boundary, so
a few tokens are surprising and the mean barely moves; `offstyle` is a valid fill under a
uniform demand, so every token is somewhat surprising and the mean moves a lot. If that is the
whole story, a tail statistic of the same array should read truth where the mean reads style.

### What changed, and what did not

Only the statistic. The reader, the panel, the buckets and the thresholding rule are untouched:
τ is still the q = 0.90 quantile of **that statistic** on genuine exemplars of the same
(style, nominal mask side), so every column is as oracle-free as the grade of record. Nine
statistics, the mean carried through as the baseline:

| statistic | what it is |
|---|---|
| `mean` | the grade of record |
| `max`, `top2`, `top4`, `top8` | mean of the k largest per-token NLLs, k clipped to the item's token count |
| `frac_hi` | fraction of the item's tokens above a per-bucket token-surprise level (itself the q = 0.90 quantile of the pooled per-token NLL of that bucket's genuine exemplars — oracle-free by the same argument as τ) |
| `max_dev`, `top4_dev` | `max − mean`, `top4 − mean`. **Demand-normalised**: a uniform shift in how atypical the demand is cancels, a few bad tokens in an otherwise fine region do not. Added because the plain tails inherit the mean's level and therefore inherit exactly the sensitivity the re-analysis is trying to remove. |
| `ring_gap` | mean NLL on the region's **outer ring** minus mean NLL in its interior. The one statistic that knows *where* to look; oracle-free because the region rectangle **is** the mask the grader is handed. Undefined at L1 (a 2×2 region has no interior). |

Separation is reported as the AUC of the **rule that is actually used**: each item's statistic is
mapped to its position inside its own bucket's genuine-exemplar distribution (pass at q *is*
`position ≤ q`), so sweeping q traces the real ROC and its AUC is that rule's AUC. The plant's
own fills are additionally split by **oracle validity** — `panel_tokens` dumps the per-item
`n_violations` of the decoded fill that `oracle_run` keeps only per-school means of — with a
second column restricted to swatches whose clean round trip is itself violation-free, so the
tile classifier's 0.69 ceiling cannot be charged to the plant.

### Caveats, stated rather than hidden

- Chain-rule per-token NLLs are conditioned on the surround **plus the cells revealed earlier in
  a random reveal order**, averaged over 2 orders. Tail statistics of that column mix "how
  surprising this cell is" with "how early it was revealed", so the 1-forward independent form
  (surround only, no reveal-order nuisance) is reported beside it throughout. The two agree on
  every conclusion below.
- Buckets are keyed by the **nominal** mask side of the level (L1/L2/L3 → 2/4/8), exactly as
  `oracle_run` does. Aligned, that is the true token count (4/16/64). **Misaligned**, a tile
  region's covering rectangle is larger (9/25/81), so misaligned panel items are compared against
  calibration items with fewer tokens. The mismatch is inherited, not introduced; it depresses
  all misaligned *pass rates* uniformly and leaves the seam-vs-offstyle *AUCs* untouched (both
  classes carry identical masks). Two columns are degenerate under it and are not read:
  misaligned L1 `top4_dev` (calibration items have 4 tokens, so the statistic is identically 0
  there) and misaligned L1 `ring_gap` (no calibration item at side 2 has an interior, so τ falls
  back to the side-4/8 median).
- At L1 an item has only 4 tokens, so `top4` = `top8` = `mean` by construction.
- Single seed, one panel. Ranks and signs are the currency.

### Scope

Both conditions, both graders, both forward counts, three levels, one-sided and two-sided. The
two-sided cut moves nothing that matters here (it costs clean and plant fills a few points and
leaves every seam/offstyle ordering intact), so the one-sided column is the one to read.

## The adjacency-support grader (`adjacency.py`, 2026-08-25)

The tail re-analysis above ended with a shape claim: **validity is a relation between tokens,
typicality is a property of tokens**, so no statistic of per-token surprise reads a pairwise
constraint on an aligned alphabet. `critic/`'s grade on RHM was never a likelihood either —
possible-set success is a **support test**. This is that instrument's learned analogue on the
code grid, and it takes **zero forward passes**.

### The rule

- **Support.** From genuine exemplars only, the set of **oriented** adjacent code pairs that
  occur: `H = (code(r,c), code(r,c+1))` and `V = (code(r,c), code(r+1,c))`, kept as two
  different relations because the tile grammar's edge table is not symmetric.
- **Verdict.** A candidate fill passes iff **every** adjacent pair it touches is in the
  support — pairs inside the hole *and* pairs between a hole cell and the surround. The
  conjunction is expressed as a conjunction.
- **Oracle-free** by the same argument as τ: the support comes from held-in exemplars of the
  grader's own split (`gA_train`, and `gA_train + gB_train` for saturation), the mask is what
  the grader is handed, and no shader, seed, offset or held-out pixel is read.

### The variants, and why each is there

| variant | what it asks |
|---|---|
| `per_style` / `pooled` | support within one school, or over all 36. The strict reading trades false accepts for false rejects; the loose one does the reverse. |
| `per_tileset*` | **diagnostic only, and starred everywhere it appears.** The 9 schools sharing a tileset share a valid set, so this is the closest a support test gets to the DGP's own answer. The grouping is a fact about the generator, so it is a ceiling, not a runnable rule. |
| `mc1 / mc2 / mc4` | a pair counts as supported only if seen ≥ that often — how much of the test rests on singletons. |
| soft columns | `frac_unsup` and `n_unsup`, so there is a ROC rather than one binary point, plus `pass_tau`: the **same** oracle-free q = 0.90 quantile rule the NLL grader uses, applied to `frac_unsup` on the grader's own `gA_val` exemplars. |

### What is reported

Per condition × variant × level: pass rates on `clean` / `seam` / `offstyle` / `plant_fill`
(binary and τ-thresholded), mean unsupported-pair count out of touched pairs, AUC on
seam-vs-offstyle and seam-vs-clean over `frac_unsup`, the plant's fills **split by oracle
validity** (with the clean-round-trip-valid subset beside it, so the tile classifier's 0.689
ceiling cannot be charged to the plant), and **where the unsupported pairs sit** — the fraction
of them that cross the hole boundary against the crossing pairs' share of all touched pairs,
which is the relation-level form of the boundary readout.

Support coverage is reported first, because it is what the whole test rests on: distinct
oriented pairs per style and pooled, and the **false-reject floor** — the fraction of pairs in
genuinely held-out (`plant_test`) swatches that are in support, and the fraction of whole
held-out swatches with no unsupported pair at all.

### Scope conditions

- Single seed, one codebook (K = 512), one geometry.
- The false-reject floor is the binding constraint, not the discrimination: a clean L3 region
  touches ~140 pairs, so even a 0.3–0.7 % per-pair miss rate costs a third of clean exemplars
  the binary verdict. Every `clean` and `offstyle` number below the seam/offstyle separation
  should be read against that floor, which is why it is printed on every table.
- `per_tileset*` is oracle-informed and never enters a comparison presented as a runnable rule.
- The misaligned condition is the control and is reported in full, not summarised away.
