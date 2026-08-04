# The level-indexed allocation ladder: levels are an action-space axis, not an allocation-space axis

**Status**: built and run — four gates (three passing, one failing in a locatable way), the
allocation prize decomposed at 3 seeds, and one controlled action-space contrast. The cut this
was built to run (a satiety-gated climb over a level-ordered budget) was **not** run, because
the privileged version of it measures zero. **Date**: 2026-08-04.
**Up**: [../README.md](../README.md) (level_moves) · **Node**: [../../../../README.md](../../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Builds**: [`../../README.md`](../../README.md) (full_loop) open item 2 — *"a level-ordered
allocation space, so §12's satiety-gated 'recruit ℓ+1' has an upward direction to recruit in.
Currently it recruits sideways into a distractor."*
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../../../ideas/adaptive_core_and_hierarchy_climb.md)
§12 (satiety as the climbing mechanism), which this supplies the missing axis for and then
finds the axis inert.

---

## One-liner

Open item 2's allocation space now exists: a budget spent over (channel, level) cells, with a
level-indexed forward model, planner and ballistic grader, and a damage model whose error
**deepens across the run** so that climbing is necessary rather than merely available. Every
upstream gate passes — the exact DP's best move level climbs 1 → 2 → 3 with the damage depth
(G-L), and a privileged per-cell allocator follows it (mean level of tree spend 2.16 → 2.42 →
2.68, slope **+0.287 ± 0.038, t = +13.15**, 3/3 seeds). And it buys **nothing**: holding the
channel allocation fixed at 99.4% tree, the level index is worth **+0.00016 ± 0.00226
(t = +0.13)** in tree FM error, against a total uniform→oracle allocation prize of **+0.0217**
that a level-*blind* oracle already recovers **99.2%** of. The same node's other contrast comes
back very large: switching the *action space* from block-only to level-indexed, at matched task
and matched allocation, moves ballistic control **0.121 → 0.424 (3.51×, +0.303 ± 0.051,
t = +10.40**, 3/3 seeds). **Where to spend is worth zero; what you can do is worth 3.5×.**

---

## 1. What this was built to test, and why the obvious design was not it

[`../../README.md`](../../README.md) §3 records that satiety on a channel-indexed budget
*"recruits laterally into structA"* rather than upward, and names the fix: give the budget a
level index. [`../README.md`](../README.md) built the level-indexed **action** space and the
**readout**, and stopped there.

The obvious cut — give the budget a level index and see whether the value climbs it — was
already answered in the negative by [`../README.md`](../README.md) §10: the value's preference
*slope* against move level does not resolve (+0.060 ± 0.108, t = +0.97). What **does** resolve
(§11) is that its abstraction premium tracks *how deep the error is*. So the value supplies
**error-depth matching, not a depth ordering**: it can say *"a level-ℓ commitment is worth it
here"*, never *"go higher"*.

So the design does not ask the drive to supply "up". It makes the **world** supply it and asks
whether the allocator follows — [`../../climb.py`](../../climb.py) §6's logic transplanted one
axis over:

| | the argument that failed | why | the fix |
|---|---|---|---|
| `climb` | deep levels are more invariant → a learner climbs | surface re-fit repaired **91%** of each event, so climbing was never *necessary* | starve samples/event; the depth advantage migrates (t = −5.44) |
| **here** | deep moves explain more → the value prefers them | shallow moves were always *sufficient* — `corrupt_tree` leaves **29.4%** of tree blocks off-grammar, so every error is repairable one block at a time | put the error *at* a level and move it deeper across the run |

## 2. What got built

Four pieces, each with its own gate. All default off or exactly reducible, so every prior
result on this node stays reachable.

**`SpanLatentFM`.** [`../README.md`](../README.md) recorded a span FM as the standing blocker
(`forecast_visits` rolls `_fm_chunked(fm, z, kk)` indexed by block, so it cannot index a
level-ℓ node). It turned out small: `BlockLatentFM`'s **target** is already the whole
`(n_blocks, D)` latent delta — only the action *conditioning* is per-block, one scattered
marker. The span version marks every block in the move's span and adds a level embedding,
subtracting `weight[1]` so a level-1 marker is **exactly** `action_embedding[k]` whatever the
embedding learns.

> **C4 (asserted).** On the flat move set the span FM is **bit-identical** to `BlockLatentFM`,
> max |diff| **0.0**, tested with a *perturbed* level embedding so the reduction is structural
> rather than an artifact of the zero init. The published FM is this FM's `max_level=1`
> restriction, not an analogue. `selfcheck` also asserts the FM is not blind to the level index
> above level 1.

**The cell index.** `build_cells` gives one cell per (channel, level) — 23 moves over 10 cells
at L=4. `_draw_moves` samples a **cell** from `w` and then a node inside it uniformly, which is
what makes this a budget over (channel, level) rather than over moves: drawing moves in
proportion to `w` would hand `tree|L1` (8 nodes) eight times the weight of `tree|L4` (1 node).
[`../../ladder.py`](../../ladder.py) makes the same choice one level up.

**Move-indexed taps, planner and grader.** `forecast_visits_cells`, `open_loop_beam_moves` /
`_imagine_plan_moves`, `per_cell_error`. The ballistic grader is load-bearing rather than
cosmetic: a block-only planner cannot *execute* a level-ℓ commitment, so under hierarchical
damage its ceiling would be set by the damage rather than by the FM (§5 measures exactly this).

**The damage schedule.** `parse_schedule("1,2,3")` splits the run into three equal blocks and
moves the error one level deeper in each, on top of [`../README.md`](../README.md) §11's
`corrupt_tree_hier`.

### The one deviation from the published ladder, and why it is forced

`n_corrupt` defaults to **2** here, not 3. `corrupt_tree_hier` damages
`min(n_corrupt, s**(depth-level))` nodes and level 3 has only 2 of them, so at `n_corrupt=3`
the entire tree is a wrong derivation, there is no clean context left to condition on, and
every move helps roughly equally. Measured, not assumed (`gate`, 3 damage depths × 3 values):

| `n_corrupt` | argmax tree level, damage depth 0 → 3 | `d*==0` at depth 3 | verdict |
|---|---|---|---|
| 1 | 1, 1, **2, 3** | 0.142 | finest ladder, but 14% of rows carry no damage |
| **2** | 1, 1, **3, 3** | 0.038 | clean, and passes [`../README.md`](../README.md)'s own G-D threshold |
| 3 (published) | **3, 3, 3, 3** | 0.038 | **swamped** — the necessity structure is invisible |

Had the published default been inherited, the ladder would have returned an uninterpretable
null. This is the node's cheapest lesson and it generalises: **`n_corrupt` is not a free
parameter once damage is hierarchical, because it interacts with the node count at the damage
level.**

## 3. The gates, in the order they must pass

**G-D — the damage is on-grammar at every depth.** 1.0000 / 1.0000 / 1.0000 at depths 1/2/3,
`d*` 2.79 / 3.26 / 3.97, `d*==0` on 0.9% / 3.2% / 3.8% of rows. Inherited from
[`../README.md`](../README.md) `certify_damage`, asserted in-run.

**G-L — the world makes depth necessary.** The exact DP's best available Δ`d*` per tree cell,
by where the error lives (in-run, `n_corrupt=2`, 512 frozen states):

| damage depth | `tree\|L1` | `tree\|L2` | `tree\|L3` | `tree\|L4` | argmax |
|---|---|---|---|---|---|
| 1 | **+1.613** | +1.453 | +1.373 | +0.502 | L1 |
| 2 | +1.225 | +1.697 | **+1.702** | +0.943 | L3 |
| 3 | +1.008 | +1.447 | **+1.920** | +1.582 | L3 |

`tree|L1` falls monotonically (+1.613 → +1.008) while `tree|L3` rises (+1.373 → +1.920) and
`tree|L4` more than triples (+0.502 → +1.582). The implied privileged mean level of tree spend
is **2.16 → 2.42 → 2.68**. The world orders the levels, and it does so by an exact quantity
rather than a learned one.

**G-E — can any endogenous drive see it?** Six per-cell drives computable from the trained
value, scored against G-L. 3 seeds, no FM, no loop (`drive_check`):

| drive | implied mean level, d=1 → d=3 | slope per damage depth (per seed) |
|---|---|---|
| **ORACLE** | 2.117 → 2.397 → 2.692 | **+0.29** |
| `abs_dv_mean` (raw \|ΔV\|) | 2.945 → 2.956 → 3.031 | +0.044 `[+0.030, +0.045, +0.056]` |
| `dv_best` | 1.808 → 1.817 → 1.851 | +0.022 `[+0.039, +0.004, +0.022]` |
| `dv_best_span` | 1.450 → 1.467 → 1.493 | +0.021 `[+0.028, +0.012, +0.024]` |
| `prem_random` | 1.797 → 1.651 → 1.649 | inconsistent `[−0.066, −0.256, +0.101]` |
| `prem_lazy` | 2.944 → 3.063 → 3.376 | inconsistent `[+0.373, nan, +0.059]` |

Three drives (`abs_dv_mean`, `dv_best`, `dv_best_span`) have **3/3-consistent positive slopes**
— the value's level preference does move the right way as the error deepens — at **7–15% of the
oracle's rate**. The shortfall localises to one cell: at `tree|L4` the value scores a root
commitment at **−0.87 to −0.94** where the oracle says **+0.49 → +1.60**. Not conservative;
wrong-signed, at every depth, in the cell whose true relevance triples. This is
[`../README.md`](../README.md) §12's structural point (a root move masks every tree token, so
the commitment is uninformed) plus §9's span confound, appearing as an *allocation* signal
rather than as a readout.

> **A retraction inside this node.** `prem_lazy` was read as passing G-E on seed 1 (a clean
> monotone 2.254 → 2.463 → 3.000). It is one seed of three; across seeds the slopes are
> `[+0.373, nan, +0.059]` with mean-level sd of ±0.53–0.98 against a ~0.3 effect, and the d=3
> value of exactly 3.000 is `tree|L2` flipping slightly negative so all mass lands on
> `tree|L3`. **The floor that mattered was the across-seed spread (0.03–0.46), not the
> within-run off-tree spread (~0.005) that was measured first.** Same shape as
> [`../../partial_hetero/shared_surface/`](../../partial_hetero/shared_surface/README.md) §4,
> and the same fix — measure the instrument before reading its cells.

## 4. The result: the level index is worth nothing, with the channel allocation held fixed

Four arms, 12 rounds, damage schedule `1,1,1,1,2,2,2,2,3,3,3,3`, 3 seeds. Every arm forks from
the same warm span FM on a bit-identical world sequence and pays the same monitor charge, so
the arms differ only in the allocation rule.

| arm | tree FM err ↓ | ballistic ↑ | tree share | mean level of tree spend |
|---|---|---|---|---|
| `uniform` | 0.4275 ± 0.0041 | 0.374 ± 0.029 | 0.400 | 2.500 (flat) |
| **`oracle_channel`** | **0.4059 ± 0.0044** | 0.424 ± 0.023 | 0.994 | 2.500 (flat) |
| **`oracle_level`** | **0.4058 ± 0.0066** | 0.418 ± 0.018 | 0.994 | 2.12 → 2.41 → **2.69** |
| `channel_only` | 0.4107 ± 0.0039 | 0.411 ± 0.031 | 0.831 | 2.500 (flat) |

`oracle_channel` is the published `oracle` rung in cell coordinates — all budget to the tree,
**uniform across its levels**. It is the only control that isolates the level index, and adding
it changed the reading:

| contrast | what varies | tree FM error |
|---|---|---|
| `uniform` → `oracle_level` | everything | **+0.0217 ± 0.0027, t = +13.68** |
| `uniform` → `oracle_channel` | channel selection only | **+0.0215 ± 0.0016, t = +23.98** |
| **`oracle_channel` → `oracle_level`** | **level index only** | **+0.00016 ± 0.00226, t = +0.13** (signs −/−/+) |

**A level-blind oracle recovers 99.2% of the entire allocation prize.** Ballistic agrees
(+0.0063 ± 0.0388, t = +0.28). The paired within-run floor is **±0.0009**, so an effect a sixth
of the channel prize would have resolved.

> **The intermediate reading that did not survive, recorded because it is the trap.**
> `channel_only − oracle_level` = +0.0049 ± 0.0029, t = +2.99, 3/3 seeds — which reads as a
> real level-index prize and is **confounded**: `channel_only` puts 83.1% on the tree against
> `oracle_level`'s 99.4%, so the difference bundles better *channel* selection with better
> *level* selection. `oracle_channel` holds the channel allocation fixed and the effect goes to
> zero. **A control that varies one thing is what a t = +3.0 needs before it is a finding.**

### This is a null about the geometry, not about the instrument

The distinction matters and this node can make it, which several siblings could not:

- the world demonstrably orders the levels (G-L, exact DP, argmax 1 → 3);
- the allocator demonstrably follows it (slope +0.287 ± 0.038, t = +13.15, 3/3 seeds);
- allocation demonstrably pays on this substrate (+0.0217, t = +13.7);
- the comparison is **paired within-run**, floor ±0.0009.

Climbing correctly, on a world that rewards climbing, with an exactly-correct allocator, buys
zero.

**The mechanism we believe, stated as a hypothesis.** Channels are **disjoint sets of
positions**; levels are **nested** on the same positions. The tree occupies blocks 0–7 whatever
the level, so a level-3 move and a level-1 move rewrite overlapping blocks and train the same
FM parameters — the level changes *which distribution of rewrites* the model sees, not *which
parameters receive data*. Nothing can be starved by spending at the wrong level, so there is no
level-specific data need for a budget to allocate over. This is structurally the same discovery
[`../../README.md`](../../README.md) §3 made one axis up (*"this geometry has no
visited-but-irreducible cell"*, so `value_red` and `visits_only` tie by construction). It is an
argument fitted to the null, not an independent measurement — see §7 for the test that would
separate it from the alternatives.

## 5. The contrast that is large: the level ACTION space

Same task, same damage schedule, same allocation rule (`oracle_channel`), 3 seeds. The only
thing that varies is the move set — `--max-level 1` gives the published block-only action space
(14 moves, 5 cells) against the level-indexed one (23 moves, 10 cells).

| | block-only moves | level moves | gain |
|---|---|---|---|
| ballistic control | 0.121 ± 0.028 | **0.424 ± 0.023** | **+0.303 ± 0.051, t = +10.40, 3/3 seeds (3.51×)** |

The block-only arm lands at 0.121, inside the published ladder's ~0.10 regime — so the
`max_level=1` restriction does reproduce the existing apparatus, and the contrast is against a
faithful baseline rather than a crippled one.

> **What must NOT be compared across these two runs.** `tree_err` averages over whichever cells
> the run's own move set contains, and normalised MSE **falls with span** because the target
> norm grows: in the level run the per-cell nmse is `tree|L1` 0.807, `tree|L2` 0.528, `tree|L3`
> 0.261, `tree|L4` 0.048. So the level run's 0.406 against the block run's 0.706 is a different
> mix of cells, not an improvement, and an earlier draft of this section quoted it as
> `+0.300, t = +12.2`. **Retracted.** Ballistic is unaffected — identical frozen eval states,
> identical damage, and terminal possible-set success is a ground-truth task metric independent
> of the move set.

**And the like-for-like FM comparison goes the other way.** On `tree|L1`, the one cell both
runs share, the block-only FM is **better**: 0.706 vs 0.812, **−0.106 ± 0.058, t = −3.14**,
3/3 seeds. It spends its whole capacity and budget on block transitions.

So the level arm's forward model is **worse on the dense proxy and 3.51× better on the task**.
That is [`../../README.md`](../../README.md) §6's own methodological claim — *"our research
process advanced by grader disagreement"* — arriving as a result rather than as a method, and
it is the sharpest instance this node produced.

## 6. What this establishes — and what it does not

**Establishes:**

1. **A level-indexed allocation space, forward model, planner and ballistic grader**, with the
   span FM exactly reducing to the published block FM (C4, max |diff| 0.0). The durable output.
2. **A damage schedule that makes abstraction necessary**, certified on-grammar at every depth
   and verified against the exact DP to order the levels (G-D, G-L) — and the measurement that
   `n_corrupt` is not a free parameter once damage is hierarchical.
3. **The level index is inert as an allocation axis on this geometry**: +0.00016 ± 0.00226
   against a paired floor of ±0.0009, with a level-blind oracle recovering 99.2% of the prize.
   This is a *geometry* null with every upstream gate passing, not an instrument null.
4. **The level index is large as an action axis**: 3.51× ballistic control at matched task and
   matched allocation, 3/3 seeds, against a baseline that reproduces the published regime.
5. **A quantified grader disagreement**: the same swap makes the FM worse on its dense proxy
   (−0.106, t = −3.14) and 3.51× better on the evaluative one.
6. **G-E as a reusable instrument** — any value head can be scored per cell against the exact
   oracle in ~3.5 minutes, with no FM, loop or allocator.

**Does not establish:**

- **Nothing about §12's satiety mechanism.** The satiety arm was built and never run at scale,
  because the *privileged* version of its allocation question measures zero. §12 is untested
  here, not refuted — what is measured is that the axis it wanted has no prize on it.
- **Nothing about whether a fixed value head could climb.** G-E says the current one tracks at
  7–15% of the oracle's rate; §4 says closing that gap would buy ~0 *on this axis*. The pooling
  question ([`../README.md`](../README.md) open item 6) is untouched and is still live for the
  readout, just no longer blocking allocation.
- **The disjoint/nested mechanism in §4 is an argument fitted to the null**, not a measurement.
- **One geometry**: L=4, s=2, one tree plus four distractors, one damage schedule, 3 seeds.
- **The action-space contrast is at one allocation rule** (`oracle_channel`). Whether the 3.51×
  survives under a weaker allocator is unmeasured.

## 7. Open items

1. **Separate the disjoint/nested account from its rivals.** The prediction it makes is
   testable: give the levels **separate FM capacity** (per-level trunks, or a per-level adapter
   large enough to starve) and the level index should stop being inert, because spending at the
   wrong level would then leave real parameters untrained. If it stays inert under separated
   capacity, the account in §4 is wrong and the null is about something else. This is the single
   most informative follow-up and it is cheap on this apparatus.
2. **Does the 3.51× survive a weaker allocator?** §5 holds the allocation at `oracle_channel`.
   The endogenous rung (`visits_level` on the published tap) is one command.
3. **A geometry where levels are disjoint.** If §4's account is right, the way to make level
   allocation matter is to stop nesting — e.g. channels of *different depths* competing, where
   spending on a shallow channel genuinely starves a deep one. That is closer to
   [`../../partial_hetero/`](../../partial_hetero/README.md)'s axis than to this one.
4. **The root cell, still.** `tree|L4` is where the value is most wrong (G-E) and where the
   oracle's relevance grows fastest (G-L). [`../README.md`](../README.md) open item 5's
   partial-mask deep move is the fix and is unbuilt.
5. **`premium_level` and `satiety_level` were built and are unrun at scale.** They are wired,
   priced and smoke-tested; they were not run because §4 removed the prize they would compete
   for. One command each if §7 item 1 restores it.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                    # the gate
modal run rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::selfcheck_remote   # C4

# G-D + G-L, and the n_corrupt sweep that picks the operating point
modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::gate \
    --tag g1 --n-corrupt-sweep 1,2,3

# G-E -- the endogenous drive gate, 3 seeds
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::drive_check \
      --tag e1_s$s --seed $s
done

# S4 -- the allocation prize, with oracle_channel isolating the level index
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::level_ladder \
      --tag price2_s$s --seed $s --policies uniform,oracle_channel,oracle_level,channel_only --taps fc
done

# S5 -- the action-space contrast: identical everything, block-only move set
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::level_ladder \
      --tag act1_s$s --seed $s --max-level 1 --policies uniform,oracle_channel --taps fc
done

python3 rhm/directed_sculpting/full_loop/level_moves/level_ladder/aggregate.py --drives 'drive_e1*'
python3 rhm/directed_sculpting/full_loop/level_moves/level_ladder/aggregate.py --pattern 'ladder_price2_*'
```

`--pattern` is **required**: results are keyed by seed, so `ladder_price2_s1` and
`ladder_act1_s1` are the same seed on different layouts and would silently overwrite each other
— the bug [`../aggregate.py`](../aggregate.py) had to fix mid-node. Results JSON on the
`rhm-scaling-data` volume under `directed_sculpting/level_ladder/`, mirrored to
[`figures/`](figures/).
