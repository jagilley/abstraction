# canvas/practice/ratchet — files

**Up**: [`../FILES.md`](../FILES.md) · **Design record**: [`SPEC.md`](SPEC.md) (the question and
the picks Jasper settled) and the *decisions I made* section below · **Substrate**:
[`../../plant/README.md`](../../plant/README.md) · **Donors**:
[`rhm/practice/ratchet/`](../../../rhm/practice/ratchet/README.md) (`ratchet.py` loop shape,
`macros.py` tables — forked, donors untouched), [`rhm/practice/ear/`](../../../rhm/practice/ear/README.md)
(`practice_prov`).

**No README yet** — results get discussed before anything is written up (repo policy).

## Code files

| file | purpose |
|---|---|
| [`macros.py`](macros.py) | **Fork of `rhm/practice/ratchet/macros.py`**, `s = 4` in a 2×2 spatial arrangement. The donor's table algebra verbatim (`base_table`, `make_table`, `_flatten`, `Miner`, `grade_table`) over a new substrate: T[1] = the K = 512 codes, T[2] = a 2×2 code block (= one tile, aligned), T[3] = a 2×2 block of those (= a 2×2-tile motif). Adds the geometry the 2-D arrangement needs — `zcells` (a block's cells in the quadtree order `_flatten` produces, which is exactly `plant/recur.py:_blockify`'s), `pair_slots` / `boundary_slots` / `cell_neighbours` (the adjacency slots a move touches), `internal_unsup` (an entry's own out-of-support pair count, a property of table × support so it is computed once), and `chain_scores` (the donor's max-sum DP) beside `flat_scores` (the gather the loop runs) for the equivalence gate. |
| [`ratchet.py`](ratchet.py) | The node. `support_matrices` / `grade_support` (the grade of record, `plant/tiles_twin/adjacency.py` made incremental and dense), `nested_regions` / `region_mask` / `make_instances` (the inpainting depth ladder), `solve` (the closed-loop beam — the whole action space and pricing), `audition`, `build_given` (the DGP's own vocabulary read off genuine exemplars through the withheld tile ids), `run_arm`, and the Modal entrypoints `run`, `ladder` (the budget calibration) and `selfcheck` (the gates). |
| [`analyze.py`](analyze.py) | **local only.** Fetches `<tag>/` off the volume and reduces it: per-era `e_k` beside the grader's clean false-reject floor, priced time in forward passes, earned-vs-given, cost-to-depth, commit events, the unit-LP audition series with `given`/`rand_k` beside them, the oracle and taste columns, the width ladder and the miner's at-support yield; three figures. |
| [`launch_detached.py`](launch_detached.py) | `canvas/plant/launch_detached.py`, retargeted. Use `--logdir` outside `canvas/` — Modal's local-source mount aborts if a file under the package changes during the build. |

## Decisions I made (the SPEC left these open; reasons, so they can be argued with)

**The beam has no learned value head; the support test is the selector.** The SPEC's design pick
4 allows the grade of record to serve as a dense signal. It is free (zero forwards) and every
placed pair is checkable mid-fill, so the beam scores a partial fill by
`LP + γ·h − λ·U`: `LP` is the log-probability of the cells written, `h` an optimistic completion
bound over the still-masked cells, `U` the adjacent code pairs left out of support (recomputed
exactly from the grid each step). λ = 8 nats makes a violation dominate any per-cell
log-probability difference, i.e. near-hard pruning with a fallback. Nothing else in the loop
learns except the plant, so **the entire descent is carried by the action space** — cleaner than
the donor, where the value was the only thing practice could move. The consequence the SPEC
flags is real and is the headline control: `never_base` measures whether search + support alone
afford depth at the declared budget.

**γ = 0.9, and why `h` exists at all.** Cumulative log-probability is not comparable across
moves of different sizes — a 4-cell macro always "spends" more of it than a 1-cell move, so a
plain-`LP` beam would never take a macro. Adding the optimistic bound `h` over the cells not yet
written makes the comparison a *regret* against the per-cell greedy bound. Discounting it by
γ < 1 preserves, among moves of the same size, the confidence order `sampler.decode` uses —
which is what makes the fork-fidelity gate exact.

**The budget buys coordination, not just hypotheses.** First smoke (`smoke0`, 6 arms, quick):
every arm sat at the grader's clean false-reject floor at every era (e = 0.00 / 0.13 / 0.25
against floors 0.04 / 0.13 / 0.21). The cause was structural: with one move per step the base
arm always reached full per-cell conditioning whatever the budget, so `G` set only the beam
width and nothing was ever unaffordable. Fixed by keeping `sampler.decode`'s **reveal
schedule**: at a declared budget of G forwards the arm runs `steps = G / width` forwards and
must commit `ceil(left / steps_left)` cells at each one. A tight budget therefore forces many
cells to be committed from a *single* forward — independent per-cell argmaxes for base moves,
one joint table entry for a macro. That is the cost-to-depth contrast, and it also strengthens
gate F: the base arm now reproduces `decode` bit-for-bit at *any* (steps, width = 1), not only
one cell per step.

**Width, and matched pricing.** `fit_width` transposed: `width = clamp(G / min_steps, 1, 8)`
where `min_steps = ceil(|M| / 4^(ℓmax−1))` is the fewest moves the arm's own action set needs to
cover the hole; then `steps = G / width`. Every arm spends the same G forwards, split
differently between coordination and hypotheses.

**A move applies only to a block that is entirely masked.** The hole is exactly tiled by macro
blocks at every level ≤ era+1 (a level-k tile region is a 2^k × 2^k code square at a multiple of
2^k), so a move never overwrites the exemplar surround and never needs an entry-matching rule
for partially filled blocks.

**`given` is read off genuine exemplars at the SAME support threshold the miner uses (3).**
T[2] = every 2×2 code block that some (school, tile id) renders to at least 3 times, via the
withheld tile ids — 364 entries, block→tile purity 0.962, all 47 catalogue tiles covered. T[3] =
every 2×2-tile motif at the same threshold, as 2×2 arrangements of T[2] *entries* (so the
ratchet constraint holds for `given` too) — 538 entries, 98 % of which map to exactly one true
motif. Sizes at other thresholds are logged (`n_t2_at`, `n_t3_at`: T[2] is flat in the threshold,
371 → 346 over 1 → 8, because the codebook resolves tiles crisply; T[3] falls 10752 → 2380 → 538
→ 239 → 37). Using one threshold for both `given` and the miner makes the *only* difference
between them the source of the tuples: genuine exemplars plus the oracle's tile ids, versus the
agent's own solved fills.

**Mining is uncapped and `n_pr = 128`.** The donor capped mined instances at 8 because a
14-entry table saturated in one cycle and would have made the certificate vacuous. Here the
table has ~364 entries, each solved instance yields exactly ONE level-2 observation at era 1,
and `mine_cap = 16` starved the miner to zero entries in the smoke. Saturation is nowhere near,
so the cap is off and practice is widened instead.

**`practice_prov` is a pure override.** `ear/`'s provisional commit (commit at the era boundary
without paying for the pre-commit audition) is a change to the commit rule alone. Its live
recert — which *swaps* a committed table — is not, because it would break the frozen ratchet
that makes the poison test a test; the recert counterfactual is logged (`held` vs `live`) and
never acted on, exactly as in the donor.

**Splits.** Support is built from `gA_train + gB_train` pooled over the 9 schools of the tileset
(360 exemplars). Practice draws from `plant_train`, metering from `plant_test`, the shadow
audition and the commit confirmation from `plant_val` with different region draws — so the
vocabulary is mined on one split and metered on another, and neither touches the grader's.

**Certificate constants** start from `rr_s0`'s calibrated values (α 0.2, c 0.06, c_v 0.15,
W 5, hold 2, min_cycle 6, `lp_min_drop` 0.10).

## Gates (`selfcheck`, all bit-for-bit, all passing)

| gate | what it holds |
|---|---|
| **F** fork fidelity | with the vocabulary off and λ = 0, the arm reproduces `sampler.decode`'s fill *and* its cost exactly, at every (era, steps ∈ {2, 8, \|M\|}, width 1) — 8 cells. The one thing that had to change to get there: alt 0 is chosen with `max`, not `topk`, because `decode` picks the code by argmax and topk's tie order is not argmax's. |
| **C-M** the operator | the donor's max-sum DP over the table chain equals the flat gather the loop runs, at L2 (exactly) and L3 (1.4e-6). `given` and `practice_*` are the same machinery differing only in which tuples are in the table. |
| **C-B** the flattening | `make_table` with s = 4 round-trips `plant/recur.py:_blockify`'s block ids at both levels. |
| **C-R** the ratchet | truncating T[2] 43 → 10 drops the rebuilt T[3] 498 → 0. |
| **G** the given table | its entries decode to catalogue tiles through the oracle: purity 0.962, all 47 tiles covered. |
| **P** the grade of record | clean pass 0.984 / 0.938 / 0.828 at L1/L2/L3 (the **false-reject floor** — no arm's `e` can go below it, and it is logged in-tag every cycle); `seam` pass 0.222 / 0.000 / 0.000. **L1 is a gate shortfall and is reported, not tuned**: min-count thresholds buy specificity by destroying coverage, so the lever is more exemplars. Era 1 is the bootstrapping era; the vocabulary claim lives at eras 2–3, where the gate holds. |

## Runs on disk

| tag | what |
|---|---|
| `smoke0` | the quick 6-arm smoke that found the pricing flaw and the starved miner |
| `cal0` | the depth ladder under the *first* (broken) pricing — the run that showed the vocabulary arms strictly worse everywhere, and diagnosed why |
| `cal1` | the depth ladder under the corrected pricing, 144 held-out instances per era, fixed width 2: what picked `G = 8` |
| `cr_s0` | **the main run** — 6 arms, seed 0, 3 eras × 24 cycles = 72 cycles, complete, 1147 s |

### Operating-point facts to carry into any reading of `cr_s0`

- **The width ladder is inverted at this budget.** At G = 8 the probe gives, for every arm and
  era, `w1 ≤ w2 ≤ w4` in error (e.g. `never_base` L3: 0.701 / 0.826 / 0.910 at 8 forwards).
  Because the budget is split as `steps = G / width`, width costs conditioning, and
  conditioning is what matters here. The run was metered at width 2 for every arm, so ranks
  and the earned-vs-given fraction are unaffected, but every absolute `e` is above what the
  same budget buys at width 1. `cal1` was measured at the same width, so the budget choice is
  internally consistent — but the operating point is dominated, and a rerun should use
  width 1 (or make width part of what an arm may choose).
- **The era-1 denominator is 0.0028.** `e_never − e_given` at era 1 is inside the metering
  noise (1/144 = 0.007), so era-1 earned-vs-given fractions are noise over noise and carry no
  information. This is the substrate's own statement — `tiles.describe` puts single-tile
  repairability at 1.00 — and era 1 is the bootstrapping era where the vocabulary is *earned*,
  not a test of whether it is needed.
- **The level-3 certificate fired at the earliest legal cycle.** `practice_gated`'s era-2 `A`
  series ran 0.927 → 0.594 monotonically over 24 cycles; the certificate fired at c_in_era 6
  (= `sil_min_cycle`) with `A = 0.823`, i.e. mid-descent, on a 16-entry table (recall 0.022),
  where `practice_late` committed 81 entries at c_in_era 21. `lp_min_drop = 0.10` was met by
  the first six cycles and the silence window then read a slow monotone descent as quiet
  because `scale = ref − emin` was still small. The era-1 firing is the textbook case by
  contrast: flat at 0.073 for ten cycles before firing at c21. The constants are `rr_s0`'s and
  were not recalibrated on this substrate; that is the obvious next calibration.

Volume `canvas-data` (profile `chromatic`): `/data/canvas_practice_ratchet/<tag>/setup.json` and
`<tag>/<arm>/results.json`. `analyze.py --fetch` mirrors that into `results/<tag>/` (gitignored)
and writes the committed reduction to `results/reduced_<tag>.json`; figures go to
`figures/<tag>/` (gitignored).

## Reproduce

```bash
cd experiments            # conda glp; MODAL_PROFILE=chromatic
modal run -m canvas.practice.ratchet.ratchet::selfcheck
modal run -m canvas.practice.ratchet.ratchet::ladder --tag cal1 --budgets 2,4,8,16,32,64
modal run -m canvas.practice.ratchet.ratchet::run --tag smoke0 --quick 1
python3 canvas/practice/ratchet/launch_detached.py --tag cr_s0 --logdir /tmp/crlogs
python3 canvas/practice/ratchet/analyze.py --tag cr_s0 --fetch --figures
```
