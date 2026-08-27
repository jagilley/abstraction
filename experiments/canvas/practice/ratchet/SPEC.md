# ratchet on canvas — SPEC (2026-08-26)

**Up**: [`../../README.md`](../../README.md) (canvas) · **Donors**: [`rhm/practice/ratchet/`](../../../rhm/practice/ratchet/README.md)
(`ratchet.py` loop shape, `macros.py` tables — fork with a fork notice, donors untouched) ·
**Substrate**: [`canvas/plant/`](../../plant/README.md) (the plant, the alphabet, the two graders;
its `tiles_twin/` for corpora + oracle) · **Arc context**: [`rhm/practice/README.md`](../../../rhm/practice/README.md)
(read `ratchet`, `ear`, `native`, `spiral`, `census` entries) · **Idea docs**:
[`style_practice_substrate`](../../../../ideas/style_practice_substrate.md) §8 node 1 + §11,
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md).

This is the first practice node on canvas and the first of three planned:
**(1) `ratchet`** — earn a level-indexed vocabulary; **(2) `native`** — consolidate it into the
plant, with a third port form canvas makes natural (*mint a tile token into the plant's own input
stream*, `native/SPEC.md:173`'s unrun sibling) beside RHM's routing head; **(3) `spiral`** —
re-earn level 3+ over the minted plant. Node 1 is this spec. Leave hooks for node 2 where cheap
(a block-level read on the plant trunk, arms as pure overrides), but don't build it.

## The question

On aligned tiles, can practice earn a level-indexed code-tuple vocabulary (T[2] ≈ tiles, T[3] ≈
motifs) over an inpainting depth ladder under a priced forward-pass budget, graded by an
oracle-free support test — and how much of what `given` (the tile catalogue) buys does it recover,
at what cost-to-depth, against `never_base`? Same claim shape as `rr_s0`: ranks and the
earned-vs-given fraction, single seed, floors measured in-tag.

## What exists (distilled — don't re-derive)

- **Corpus + plant**: `tiles_aligned` on volume `canvas-data` (`/data/corpora/tiles_aligned/`), 36
  schools = 4 tilesets × 9; splits by *seed band* (`plant/corpus.py:SPLITS`, held-out held out of
  the codebook too). Quantizer `twq_aligned` (K=512 of record), plant + grader weights under
  `/data/plant/tw_aligned/`. A swatch is a 16×16 code grid; a tile is exactly a 2×2 block, a
  2×2-tile motif 4×4. `plant/model.py` (any-order masked model, style token = the request, 10 %
  dropped), `plant/sampler.py:decode` (the primitive action; cost = steps × width forward passes).
- **The vocabulary machinery**: `macros.py` tables are substrate-agnostic (entries = s-tuples of
  lower-entry indices; here s=4 in 2×2 spatial arrangement). `plant/recur.py` already builds T[2]
  candidates as non-overlapping 2×2 blocks and T[3] as 2×2 blocks of those with sub-support → OOV
  — that *is* the ratchet constraint. Mining on canvas has no reader to learn: the alphabet is the
  parse (blockify), so tuples come straight from solved fills.
- **Truth gauge (grade of record)**: `plant/tiles_twin/adjacency.py` — oriented adjacent code
  pairs from genuine exemplars; a fill passes iff every pair it touches, incl. across the hole
  boundary, is in support. Zero forwards; AUC .96–.99 seam-vs-offstyle. **Binding limit is
  coverage**: on `gA_train` alone, 35 % of clean L3 swatches false-reject; pooling gA+gB lifts
  clean L3 .701 → .854. Min-count thresholds do not help.
- **Taste gauge**: `plant/grader.py` typicality (mean per-token NLL under the gA reader, q=0.90
  per style × mask size). Reads demand, not truth. Logged, never decides.
- **Oracle** (experimenter-held, reporting columns only): `tiles_twin/twin.py` — decode a fill to
  pixels, `classify_tiles` against the school's atlas (accuracy 0.870; report beside the
  clean-round-trip ceiling), `Tileset.validate`; per-swatch tile ids on disk; `tiles.describe()`
  gives single-tile repairability 1.00 / 0.076 / 0.019 and greedy dead-end 0 / 0.30 / 0.53 at
  L1/L2/L3 — depth unaffordable to the primitive action, by the grammar.
- **The loop shape** (`rhm/practice/ratchet/ratchet.py:run_arm`): eras of nested damage; per cycle
  practice (closed-loop beam at declared budget, every tip's trajectory labelled by terminal
  grade → value), mining from the beam's own chosen answers on solved instances (`mine_cap`,
  `mine_support`), the candidate-macro shadow audition `A` on a held-out set, the certificate
  (δ-silence on `A` + descent precondition), commit → macros instantiated at every node of their
  level, frozen; metering `e_k` on a fixed held-out set per era; the plant fine-tunes on its own
  solved configurations with replay. `ear/` added provisional commitment at the era boundary.

## Design choices settled (Jasper 2026-08-26 — these are the picks, reasons given so you can push back)

1. **Piece = `mask` mode, nested regions** (`tiles.level_regions`: 2×2 / 4×4 / 8×8 codes, era
   k+1's hole contains era k's). Inpainting is what the plant was trained on and what the sampler
   prices; `seam` repair adds a detection sub-problem that would confound the vocabulary claim.
   Log `seam` as a reporting ladder if cheap.
2. **`given` runs as an arm**: T[2] = the tileset's catalogue as code blocks, T[3] = its
   affinity-boosted motifs (the true tables `recur.py`'s purity numbers were scored against).
   Canvas has the ceiling RHM's memo node 1 lacked; use it.
3. **A move commits one block to one table entry**, scored by the plant's per-cell log-probs at
   the block's cells given the current partial fill — one forward gives all cells at once, so a
   level-ℓ macro costs about one grounding. **Price in forward passes.** Declared per-solve budget
   G; each arm runs the widest beam its own action-set size affords (`fit_width`), exactly
   `rr_s0`'s matched-pricing rule. Moves restricted to blocks intersecting the mask.
4. **Support is incremental and free** — every placed pair is checkable mid-fill, so the grade of
   record may serve as a dense signal in the beam (a value target and/or a pruning term), not just
   a terminal verdict. Say which you did. Note the consequence: `never_base` then measures whether
   search + support alone afford depth at G, a stronger control than RHM's.
5. **Coverage before anything decides.** Build support from `gA_train + gB_train` pooled (per
   tileset). Log the clean false-reject rate per level every cycle beside `at_support` and
   typicality. If the L3 floor is still the binding term, more exemplars (not thresholds) is the
   lever; say so in the reduction rather than tuning around it.
6. **Scope for the first run**: one tileset (`full`, 47 tiles — the largest catalogue) × its 9
   schools, one table set. Tables keyed per tileset if you run all four; the style token already
   names the school (and thus tileset), so that's the request, not the oracle.
7. **Arms**: `rr_s0`'s five (`never_base`, `given`, `practice_gated`, `practice_early`,
   `practice_late`) plus `ear/`'s `practice_prov` if it's a pure override. `at_support` next-level
   yield logged from day one (PR #69's endogenous arrival pair is node 3's, not this node's).

Open to you: G, era lengths, `mine_cap` / `mine_support`, the value function's form (the RHM MC
value vs. something support-shaped), certificate constants (start from `rr_s0`'s and recalibrate
on the smoke — `rr_s0`'s README has the calibration record), how the plant fine-tunes (replay
against the clean pool, as `calp_s0`), beam width ladder, held-out set sizes.

## Instruments, never acted on

Oracle validity of every audited/metered fill (with the round-trip ceiling); tile-purity of mined
entries; `A_true` (the catalogue table on the same instances) and `rand_k` (matched-size random
subsets, 3 draws — concentration vs coverage); `held`/`live` (frozen committed vs live
vocabulary — the recert counterfactual); width ladder; plant guard (held-out NLL by mask size,
flat or not); typicality of every fill; clean false-reject of the support grader per level.

## Gates (bit-for-bit, in a `selfcheck`)

Fork fidelity: with all vocabulary machinery off, the arm reproduces `sampler.decode`'s fill and
cost exactly on a fixed batch. The given table's entries decode to catalogue tiles 1:1 through
the oracle. Truncating T[2] drops the rebuilt T[3] (`ratchet` C-R). `macros.make_table` with s=4
round-trips `recur.py`'s block ids. Pricing: counted forward passes equal the sampler's reported
cost. Grader: clean pass ≥ 0.85 and `seam` reject ≥ 0.90 at every level under the pooled support
before the loop runs (`describe`-style preflight — report, don't tune).

## Ways of working

- Invoke `/run-experiment-on-modal` before touching Modal and `/subagent-instructions` before any
  wait. App `canvas`, volume `canvas-data`, profile `chromatic`, L4. Results to
  `/data/canvas_practice_ratchet/<tag>/<arm>/results.json` + `setup.json`; reduced JSON committed
  under `results/`, figures under `figures/` (gitignored). Launch via a copy of
  `plant/launch_detached.py` with `--logdir` outside the package.
- Smoke (`--quick`, foreground, ≤ 8 min) before the main run. Main run detached, **one seed**.
  Halt with the handle; halt again with the reduction (`analyze.py`: per-era `e_k` table, priced
  time, earned-vs-given fractions, cost-to-depth, commit events, the oracle columns; figures).
- Add rows to `canvas/FILES.md` (a `practice/` child) and a `practice/FILES.md`. **No README** —
  results are discussed first, per repo policy. Record the decisions you made and why in
  `FILES.md`'s design-record section (offbook's convention).
- Don't touch git branches or state; work only inside this worktree.
