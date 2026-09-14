# plant — the canvas substrate description: the alphabet has to nest, and validity is a relation

**Up**: [`../README.md`](../README.md) (canvas) · **Design records**: [`SPEC.md`](SPEC.md) (the
GLSL-library plant, `pl0`), [`tiles_twin/SPEC.md`](tiles_twin/SPEC.md) (the aligned/misaligned
twin, the tail re-analysis, the adjacency-support grader) · **Files**: [`FILES.md`](FILES.md),
[`tiles_twin/FILES.md`](tiles_twin/FILES.md)
**Idea doc**: [`style_practice_substrate`](../../../ideas/style_practice_substrate.md) §3a, §4, §5,
§8–§10 (its §11 records what this node did to it) · **Grader donor**:
`rhm/practice/critic/`[^private] · **Vocabulary reference**:
[`rhm/practice/ratchet/macros.py`](../../rhm/practice/ratchet/macros.py)
**Runs**: `q0` + `pl0` (GLSL library, 92 min on one L4) · `twq_*` + `tw_aligned` / `tw_misaligned`
(tiles twin, 95 min each, 2-wide) · `oracle_run`, `panel_dump`, `panel_tokens` (replays against
saved weights, ≤ 3 min each) · `twosided.py`, `tailgrade.py`, `adjacency.py` (offline), all
2026-08-25. **Ranks, signs and the aligned–misaligned contrast are the
claims.** One orchestrated conversation; two implementer agents (the second picked up after the
first lost its session at the tail re-analysis).

## The question

This is the first node on canvas and it is **not a practice loop**. Before one can run on image
styles, the *plant* — the learner the loop acts on — has to exist and be described. On RHM the
hierarchy was in the data by construction; on images it is in the generating program, and whether
it survives tokenization into a codebook the first regime built is
[the memo](../../../ideas/style_practice_substrate.md)'s §9 risk 1, "the alphabet may not
factor." Alongside it, `critic/`'s learned grade of record had only ever run with the exact grade
logged beside it; images are where it has to run unchecked. So the node measures the
preconditions the arc has been shown to need — an alphabet whose tuples nest, a depth ladder,
depth unaffordable to the primitive action, and a grader that can be calibrated without an
oracle — in the spirit of `tiles.py:describe()` and `rhm/practice/tall/`'s preflight, rather than
assuming them.

## What was built

Three things, in order (full design reasoning in [`SPEC.md`](SPEC.md)):

1. **An alphabet.** k-means over 16×16-px patches (K = 512 of record; 256–2048 laddered), decoder
   a lookup, so the reported floor is the alphabet's and nothing else's. A 256×256 swatch becomes
   a 16×16 code grid — `macros.py`'s `T[1]`.
2. **A plant.** One any-order masked token model (MaskGIT-style, ≈ 5 M params) over the grid,
   trained on all styles with held-out *seeds* per style (two crops of one seed overlap in
   pixels, so a swatch-level split leaks). The style token is dropped 10 % of the time so the
   same weights give a conditioned and a surround-only readout. **The mask-size ladder is the
   depth ladder**: eras of small holes → half the image.
3. **A grader.** `critic/`'s protocol: a second reader of the same form on a disjoint split,
   verdict = mean per-token NLL of the fill (chain-rule over random reveal orders; a 1-forward
   independent form beside it), thresholded at the oracle-free q = 0.90 typicality quantile,
   corpus size by the self-manufactured-damage rule. Two forced departures: τ per (style,
   mask-size), and the chain-rule verdict so `shuffle` damage is visible.

Every damage class is a rearrangement of codes the agent already holds — no program, seed or
held-out pixel is read anywhere on an agent-consumed path. Eight bit-level gates
(`plant.selfcheck`) pin the apparatus, including that the 1-forward verdict *is* the independent
per-cell NLL and that `recur.py` reproduces `macros.py`'s ratchet constraint (uniform noise
collapses to a single OOV block at level 3).

## Findings

### Round 1 — the GLSL library (`pl0`): three preconditions fail in the same direction

1. **The floor is shallow and the codebook is per-style.** 8× the codebook buys 23 % of the
   error (NMSE 0.355 → 0.273 over K = 256 → 2048); held-out ≈ train at every K. **Each style
   uses 17–183 of 512 codes**, largely disjoint from the others.
2. **The plant learns marginals, then only local conditionals — two phases, not
   coarse-after-fine.** To step ~1400 every mask size improves together (6.03 → ~3.9 nats: code
   marginals and style identity). After that only the small holes keep improving (area 1: 2.98
   → 2.10); the whole-grid case gains nothing after step ~4k and drifts up; overfit onset is
   monotone in mask size. The surround alone determines the style (+0.03–0.06 nats without the
   id at every size that has a surround).
3. **Recurrence tracks code entropy, not parts.** The designed part-y / texture-y control is
   null (T[2] mass at support 0.130 vs 0.202, unlabelled 0.169). What orders the styles is
   per-style code perplexity (Spearman −0.84; −0.28 against NMSE): the top of the table is
   `wood_grain` 0.677 and `sediment_strata` 0.613 — the translation-invariant stripes — while
   visibly part-y tilings sit at the floor (`truchet_meander` 0.013, `girih_stars` 0.021) and
   `brain_coral`, `mosaic_tesserae` have **no** 2×2 block recurring 8× in 8192, so their T[2] is
   empty and T[3] is all OOV.
4. **Search buys nothing.** 32× the forward passes moves exact code recovery by ≤ 0.013 at every
   mask size; more search is slightly *less* true and more typical under the reporting grader;
   at area 256 the plant equals a style-marginal-argmax filler exactly (0.119 vs 0.124).
5. **No corpus size qualifies, and the grader is gamed by the marginal mode.** Damage falls
   0.686 → 0.327 across the corpus ladder and plateaus; clean sits at 0.87. `other_style` and
   `uniform` are caught (0.039, 0.000); `shuffle` (0.609) and `marginal` (0.333) are not, both
   strongly mask-size dependent; the valid-alternative class `same_style_other` passes 0.594,
   which is the grader behaving correctly. **The plant's own fills pass at 0.997–1.000 — above
   genuine exemplars — with lower NLL, and the effect grows as the grader improves** (0.931 →
   0.998 across the ladder). Visually the beam fills a truchet hole with blank paper: the modal
   code. The homogeneous gA/gB pair lands on `endo_expansion` §7.6's measured floor exactly
   (top-1 disagreement 0.036, rank corr 0.991); r(score, −d_truth) is +0.18–0.74 by class —
   *not* the near-orthogonal 0.01–0.04 `critic/` measured on RHM.

   *Offline follow-up (`twosided.py`, from a replay of the panel against saved weights):* the
   plant is not in the low tail of the genuine-exemplar distribution, it is **off the bottom of
   it** — mean percentile 0.12–0.14, and 45–49 % of its fills are more typical than *any* of
   the 64 real exemplars in their bucket (clean sits at 0.48). A two-sided typical-set rule with
   a 1 % lower cut halves the plant's pass rate (0.997 → 0.493) for 7 points of clean and 2 of
   `same_style_other` — which survives because valid alternatives sit at percentile 0.70, far
   from the bland end.

The three failures read as one fact with a proposed mechanism: crops land at random pixel
offsets, so the same motif spells differently in code every time and only translation-invariant
styles can recur; the plant learns what survives that (local conditionals); and a
typicality grade is maximised by the marginal mode, which is off-style. This is exactly the
"misalignment between a hidden grid and the VQ grid" case `tiles.py` was kept as a calibration
twin for.

### Round 2 — the tiles twin: alignment is decisive

Same 36 schools (4 preset tilesets × 9), same tilings, same 320×320 renders, same seed bands;
the only variable is where the 256×256 crop starts — **aligned** (a multiple of 32 px, so a
tile is exactly a 2×2 code block) or **misaligned** (`offset mod 16 ∈ [4, 12]` on both axes, so
every tile straddles four cells; `lib0`'s condition by accident). The same `plant.py` produces
the five readouts unchanged; `twin.oracle_run` adds the withheld oracle as reporting columns.

| readout | aligned | misaligned | lib0 |
|---|---|---|---|
| quantizer NMSE at K = 512 (K = 2048) | **0.107** (0.015) | 0.239 (0.147) | 0.325 (0.273) |
| depth ladder, held-out NLL/token at area 1 → 121 → 256 | **0.026 → 1.600 → 2.224**, monotone | **1.543 → 1.551 → 3.307**, flat to 0.03 nats across a 121× range of hole area | 2.133 → 3.686 → 4.074 |
| T[2] mass at support · \|T[2] at support\| | **0.999** · 31 | 0.543 · 220 | 0.166 · 58 |
| T[3] occurrences with all halves at support (the ratchet constraint one level up) | **0.996** | 0.213 | 0.032 |
| exact code recovery, 1 forward, area 1 / 64 / 256 | 0.986 / 0.492 / 0.249 | 0.454 / 0.470 / 0.344 | 0.352 / 0.212 / 0.119 |
| what 32 forwards buy over 1 | +0.03–0.04 at mid sizes | ≤ 0.005 | ≤ 0.013 |

6. **The miner recovers the tile catalogue as its level-2 vocabulary** (aligned; oracle
   column). Per tileset — knot (29 tiles) 32 blocks at support, block→tile purity 0.937,
   tile→block 0.954; circuit (37) 33, 0.978 / 1.000; meander (28) 20, 0.897 / 0.992; full (47)
   40, 0.930 / 0.974. T[3] at support is ~18 motifs per school out of ~1200 true ones, 88 %
   pure — the affinity-boosted ones recur, the rest is tail.
7. **Depth is unaffordable to the primitive action here, by the grammar's sparsity.**
   `tiles.describe()` over all 36 schools: single-tile repairability 1.00 / 0.076 / 0.019 at
   levels 1/2/3, greedy dead-end rate 0 / 0.30 / 0.53. `lib0` could not establish this at all.
8. **Aligned is also easier, and this design cannot separate that from "the hierarchy is in the
   tokens."** The subagent flagged it and it stands on the record. Our reading: for a
   fixed-capacity learner these are the same fact seen from two sides, and RHM had both by
   construction; what the twin adds is that a *learned* alphabet can have them too, if the
   tokenization respects the DGP's lattice. The recurrence and purity numbers are the ones that
   are not a restatement of the floor.

### Round 3 — the grader against known validity: typicality reads demand, and a conjunction needs a support test

The twin's damage panel has known validity: `seam` (region re-sampled ignoring the surround —
every tile legal, every interior edge agreeing, the inconsistency only at the region boundary;
**invalid**, locally unsuspicious) and `offstyle` (re-sampled respecting the surround under a
uniform demand; **valid** but atypical — the exact `alt_valid` class). Pass rates for gA24,
one-sided / two-sided:

| condition | level | clean | seam (invalid) | offstyle (valid) | plant fill |
|---|---|---|---|---|---|
| aligned | L1 | .93 / .91 | .62 / .62 | .69 / .69 | .93 / .91 |
| aligned | L2 | .81 / .78 | .64 / .64 | .60 / .57 | .86 / .77 |
| aligned | L3 | .72 / .65 | **.54** / .53 | **.26** / .26 | .87 / .70 |
| misaligned | L1 | .94 / .75 | .47 / .44 | .81 / .62 | .97 / .58 |
| misaligned | L2 | .90 / .69 | .51 / .51 | .88 / .75 | .96 / .46 |
| misaligned | L3 | .78 / .52 | .49 / .47 | .77 / .65 | .97 / .26 |

9. **In the aligned condition the grade of record passes invalid `seam` more than valid
   `offstyle`** (L2 .64 vs .60; L3 .54 vs .26), and the two-sided fix stops helping because the
   plant's fills there are no longer bland (percentile 0.41; 12–14 % below bucket minimum vs
   45–49 % on `lib0`) — just typical and about half invalid. Decoding the plant's aligned fills
   back to tiles and running the edge test: valid 0.497 against a 0.689 classifier ceiling (tile
   classifier accuracy 0.870; a clean round trip is violation-free 1.000), while the grader passes
   them at 0.87–0.93. **Mean per-token NLL is a demand gauge, not a truth gauge, and a better
   local reader makes it a sharper demand gauge** — a seam violation lifts a few boundary tokens
   by ~0.09 nats while an off-style fill is lifted ~0.37 nats at *every* token.
10. **No statistic of per-token surprise fixes it** (`tailgrade.py`, from a per-token replay of
    the panel). AUC(seam vs offstyle), aligned gA24 8-forward, L1/L2/L3: mean .548 / .469 /
    .302; max .527 / .471 / .424; top-2 .558 / .492 / .407; top-4 .548 / .493 / .388; fraction
    above a surprise quantile .482 / .494 / .335; demand-normalised tails .43–.51. Tails move
    seam and offstyle up *together*. The only statistic above chance knows *where* to look:
    `ring_gap` (mean NLL on the region's outer ring minus its interior; oracle-free, the mask
    is the region) reaches .603 / .651 at L2 / L3 and finally orders the classes correctly at
    q = 0.90 (clean .878 > offstyle .854 > plant .837 > seam .757) — and it is at chance on the
    plant's own fills, whose invalidity is distributed through the hole rather than at its
    boundary (there the *mean* carries the weak signal: AUC .626 / .656, still passing 85 % of
    invalid fills).
11. **The mechanism is geometric, and it puts the alphabet result and the grader result in
    direct tension.** In the *misaligned* condition the same likelihood grader works
    (mean .769 / .806 / .693; top-4 .824 / .848 / .868; `ring_gap` .892 / .951 at L2 / L3; at L3
    top-4 drops seam to .104 while offstyle stays .681; seam's top-4 tokens sit on the ring at
    0.844 vs a 0.395 share) because a code cell *straddles* the tile boundary, so an edge
    disagreement lands inside one patch and yields a code no exemplar contains. In the *aligned*
    condition the disagreement falls exactly *between* two cells: each code is individually
    legal and the violation exists only as a relation between neighbours, which a per-token
    reader barely sees (seam's top-4 on the ring 0.531 vs a 0.438 share, lift 1.21). **The
    condition that gives you a vocabulary is the condition that blinds a likelihood grader to
    validity.**
12. **The matched instrument is a support test, and it works** (`adjacency.py`, zero forward
    passes). From genuine exemplars only (gA's 24 swatches/style), record every oriented
    adjacent (code, code) pair; a fill passes iff every pair it touches — inside the hole and
    across the hole boundary — is in support. It is a conjunction by construction and reads
    truth, not demand. Aligned, pooled support, min-count 1:

    | level | clean | seam (invalid) | offstyle (valid) | plant fill | AUC seam vs offstyle | AUC seam vs clean |
    |---|---|---|---|---|---|---|
    | L1 | .984 | **.069** | .929 | .962 | **.958** | .964 |
    | L2 | .915 | **.007** | .784 | .671 | **.987** | .993 |
    | L3 | .701 | **.007** | .451 | .153 | **.972** | .990 |

    96 % of the seam's unsupported pairs cross the region/surround interface against a 20 %
    chance share at L3 (.992 / .984 / .961 vs .653 / .383 / .200 by level) — the violation is
    found where the DGP put it; clean and offstyle sit at or below chance on 8–25× fewer
    unsupported pairs. **The direct truth test**, the plant's own fills split by oracle validity
    (aligned L3, restricted to swatches whose clean round trip is itself violation-free,
    n = 146): **rejects 96.6 % of invalid fills and passes 60.0 % of valid ones**, AUC .933
    (pooled .889) — and 0.60 is essentially the clean-exemplar ceiling (.701 binary), so false
    accepts are ~nil and the residual is coverage, not a plant fault. The binding limit is the
    **false-reject floor**: 99.3 % of held-out pairs are in support, but an L3 region touches
    ~140 pairs, so 35 % of clean swatches fail the binary verdict. More exemplars fix that
    (pooling gA + gB: held-out pairs 0.9988, clean L3 .701 → .854, AUC .972 → .984); min-count
    thresholds do not (mc4 buys specificity by destroying coverage). Misaligned is the control:
    support 2.8× larger and unsaturated (0.860 of held-out pairs; 3.5 % of clean swatches
    fully supported), the binary verdict rejects nearly everything, AUC .69–.84, and no boundary
    localisation (.394 / .337 / .270 vs shares .488 / .317 / .184).

| AUC(seam vs offstyle), gA24 | aligned (alphabet nests, purity .93–.98) | misaligned (vocabulary null) |
|---|---|---|
| mean-NLL grade of record | **.30–.55** (inverted at L2–L3) | .69–.81 |
| best per-token tail statistic | .65 (`ring_gap` only) | .95 |
| **adjacency support, 0 forwards** | **.86–.99** | .65–.84 |

## Interpretation (discussed with Jasper 2026-08-25 — argued, not measured)

- **(a) The alphabet has to nest in the DGP's lattice, and a learned alphabet can.** RHM's
  tokens are the grammar's leaves by construction; on images the same condition has to be
  earned by the tokenization, and when it is, mining over a reconstruction-trained codebook
  recovers the generating unit (finding 6) — the memo's §1 justification 1, answered.
  Without it, none of the arc's preconditions hold (findings 1–5, reproduced as the misaligned
  column), and the `lib0` library, whose styles have no lattice, lives permanently in that
  regime.
- **(b) Validity is a relation between tokens; typicality is a property of tokens.** No
  functional of per-token surprise reads a pairwise constraint on an alphabet aligned so that
  constraints fall between tokens (findings 9–11). RHM's grade of record was never a
  likelihood: possible-set success is a support test — a conjunction over rule lookups — and
  `critic/` could not have seen this because at uniform demand the possible set is flat, so
  there was no mode to collapse into and no demand shift to confuse with a violation. On
  canvas the learned analogue of possible-set success is adjacency support (finding 12), with
  the honest scope condition that validity be pairwise-local — true of the tile grammar by
  construction, true of RHM at each level, not true of the GLSL library.
- **(c) The two currencies now have two organs on the grader side.** `typed_gaps/` found the
  conditioning gap is typed — chunks store demand-concentration, not truth, one organ per
  currency of change. The panel shows the same split on the grader: the support test reads
  truth and passes rare-but-legal `offstyle`; the typicality reader reads demand and passes
  in-style-but-broken `seam`. The typicality reader is therefore not a failed truth gauge but
  the *taste* organ the memo's §6 said neither prior substrate had — a grader with preferences
  inside the valid set — and it dissociates from truth exactly where it should.
- **(d) Tiles is the substrate; the GLSL library is a taste venue for later.** With alignment,
  aligned tiles carries the full instrument set the arc ran with on RHM — a nesting alphabet, a
  graded depth ladder, depth unaffordable to single-tile repair, a truth gauge, style drift on
  fixed truth (nine schools per tileset) and truth drift (four tilesets), a held-and-withheld
  oracle, and a `given` arm (the tile catalogue and motif set are the true T[2]/T[3]) — plus a
  taste gauge RHM never had. `lib0` has neither a nesting alphabet nor a pairwise-local
  validity, so only the demand gauge exists there; its real demand and human-authored styles
  are still the reason to return to it once taste is the question.

## Caveats

- **One library, one tile geometry, one panel.** Ranks, signs and the
  aligned–misaligned contrast are the currency; no number here is a benchmark.
- **The aligned/misaligned manipulation is one variable but not one difficulty** (finding 8).
- **Both readers are homogeneous** (`heterogeneous_graders` §9): the gA/gB pair catches sampling
  noise, not a shared failure mode; the VLM A/B probe in the teacher slot is out of scope here.
- 8-forward per-token NLLs are conditioned on earlier reveals in a random order; the 1-forward
  independent column has no such nuisance and agrees on every conclusion in round 3.
- Bucket keys in the twin panel are the nominal level side; misaligned covering rectangles are
  larger, which depresses misaligned *pass rates* uniformly and leaves AUCs untouched.
- The tile classifier used for fill validity is nearest-neighbour in pixel space (accuracy
  0.870); every validity number is reported beside the clean-round-trip ceiling.
- bf16 autocast + TF32 were added after the first smoke; all NLLs are fp32 numbers and the
  gates were re-run under autocast.

## Runs on disk

| tag / artifact | node | what |
|---|---|---|
| `q0`, `pl0` | `plant/` | GLSL library: quantizer ladder; plant + 7 graders + all five readouts |
| `pl0/panel_dump.npz` | `plant/` | replay of the panel against saved weights, per-item NLLs (`twosided.py`) |
| `twq_aligned`, `twq_misaligned` | `tiles_twin/` | quantizer ladders per condition |
| `tw_aligned`, `tw_misaligned` | `tiles_twin/` | the five readouts per condition |
| `tw_*/oracle.json` | `tiles_twin/` | tile recovery, known-validity calibration, fill validity |
| `tw_*/panel_tokens.npz` | `tiles_twin/` | per-token NLLs, per-item fill validity, panel code grids (`tailgrade.py`, `adjacency.py`) |

Volume `canvas-data` (profile `chromatic`): `/data/corpora/{plant0,tiles_aligned,tiles_misaligned}/`,
`/data/plant/<tag>/`. Reduced JSONs are committed under each node's `results/`; figures
(`figures/*.png`) and `.npz` dumps are gitignored and regenerate from the commands below.

## Reproduce

```bash
cd experiments            # conda glp; MODAL_PROFILE=chromatic

# round 1 — the GLSL library
python -m canvas.plant.corpus --out canvas/corpora/data/plant0 --tar
modal volume put canvas-data canvas/corpora/data/plant0.tar corpora/plant0.tar
modal run -m canvas.plant.plant::selfcheck
modal run -m canvas.plant.plant::unpack
modal run -m canvas.plant.plant::quantize_ladder --tag q0
modal run -m canvas.plant.plant::run --tag smoke --smoke 1                     # ~3 min
python3 canvas/plant/launch_detached.py --fn run --tag pl0 --qtag q0 --k 512   # ~90 min
python3 canvas/plant/analyze.py --tag pl0 --qtag q0 --k 512 --fetch --figures
modal run -m canvas.plant.plant::panel_dump --tag pl0                          # for twosided.py
python3 canvas/plant/twosided.py --tag pl0

# round 2 — the tiles twin
python -m canvas.plant.tiles_twin.corpus --tar
python3 canvas/plant/tiles_twin/describe_schools.py
for c in aligned misaligned; do
  modal volume put canvas-data canvas/corpora/data/tiles_$c.tar corpora/tiles_$c.tar
  modal run -m canvas.plant.tiles_twin.twin::unpack_tiles --tar corpora/tiles_$c.tar
  modal run -m canvas.plant.plant::quantize_ladder --tag twq_$c --corpus tiles_$c
  python3 canvas/plant/launch_detached.py --fn run --tag tw_$c --qtag twq_$c --k 512
done
# round 3 — oracle columns and the grader re-analyses (after both runs land)
for c in aligned misaligned; do
  modal run -m canvas.plant.tiles_twin.twin::oracle_run   --tag tw_$c --qtag twq_$c --corpus tiles_$c
  modal run -m canvas.plant.tiles_twin.twin::panel_tokens --tag tw_$c --qtag twq_$c
done
python3 canvas/plant/tiles_twin/analyze.py --fetch --figures
python -m canvas.plant.tiles_twin.tailgrade --figures
python -m canvas.plant.tiles_twin.adjacency --figures
```

Gates, per-decision reasons and volume layouts: [`SPEC.md`](SPEC.md), [`FILES.md`](FILES.md),
[`tiles_twin/SPEC.md`](tiles_twin/SPEC.md), [`tiles_twin/FILES.md`](tiles_twin/FILES.md).

## Next steps (queued, not started)

*Update 2026-08-26: the first item ran — [`../practice/ratchet/`](../practice/ratchet/README.md).*

The first practice-flavoured node on canvas — fork `rhm/practice/ratchet/ratchet.py` +
`macros.py` verbatim onto **aligned tiles**, with the support test as the grade of record for
the value target, mining gate and audition, the typicality reader logged as taste from day one,
`at_support` next-level yield logged from day one, ranks against `never_base` and (now) a `given`
arm from the tile catalogue. Design choices to settle first: the damage ladder (`tiles.damage`'s
nested seam levels, repair-shaped, vs masks, inpainting-shaped); whether `given` runs as an arm
or stays reporting-only; whether PR #69's endogenous arrival pair runs in the first round ·
Support coverage for the grade of record (gA + gB pooling, or more exemplars) so the false-reject
floor at L3 is not the binding term · `critic/`'s pending README gains this node's finding (b)
as its addendum · The GLSL library returns when taste is the question (the `taste/clicker.py`
gauge has zero clicks) · A translation-equivariant alphabet is the sibling if a non-lattice
image substrate is ever wanted.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
