# ratchet — earning a tile vocabulary on canvas: earned ≈ given, the descent is vocabulary-carried, and the two gauges disagree in-loop

**Up**: [`../README.md`](../README.md) (canvas practice) · **Spec**: [`SPEC.md`](SPEC.md) (the
question and the design picks, 2026-08-26) · **Files + design record**: [`FILES.md`](FILES.md)
(the implementer's open calls with reasons, gates, operating-point facts) ·
**Substrate**: [`../../plant/README.md`](../../plant/README.md) · **Donors**:
[`rhm/practice/ratchet/`](../../../rhm/practice/ratchet/README.md) (loop shape, `macros.py`, forked
with `s = 4`), [`rhm/practice/ear/`](../../../rhm/practice/ear/README.md) (`practice_prov`) ·
**Idea docs**: [`style_practice_substrate`](../../../../ideas/style_practice_substrate.md) §8 node 1,
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md).
**Runs**: `smoke0`, `cal0`, `cal1` (calibration), **`cr_s0`** (main: 6 arms × 3 eras × 24 cycles,
seed 0, 1147 s on one L4), 2026-08-26. **Single seed; ranks, signs and in-tag contrasts are the
claims.** One orchestrated conversation, one Opus implementer.

## The question

The first practice node on canvas. On RHM, [`ratchet`](../../../rhm/practice/ratchet/README.md)
showed a level-indexed macro vocabulary mined from the agent's own successful repairs recovers
68–98 % of what the DGP's true tables buy, and that the descent is carried by the action space.
Can the same loop earn a vocabulary on images — T[2] ≈ tiles (2×2 code blocks), T[3] ≈ 2×2-tile
motifs — over an inpainting depth ladder, priced in forward passes, with the oracle-free
adjacency-support test as the grade of record, and how much of `given` does it recover against
`never_base`?

## What was built (full record in [`FILES.md`](FILES.md))

Aligned tiles, tileset `full` (47 tiles) × its 9 schools. The **piece** is `mask`-mode inpainting on
nested regions: era k's hole is a 2^k × 2^k code square (one tile, one motif, four motifs). A
**move** commits one fully-masked block to one table entry, scored by the plant's per-cell
log-probs from one forward; the base action commits single cells. **Price** = forward passes, G = 8
per solve for every arm (`cal1` picked it; width 2, steps 4, `sampler.decode`'s reveal schedule so
the budget binds). **Grade of record** = adjacency support built from `gA_train + gB_train` (360
exemplars), used incrementally in the beam (`LP + γ·h − λ·U`, λ = 8 nats) and as the mining gate,
audition and meter. **No learned value head**; the plant fine-tunes on its own solved fills.
**Mining** from the beam's chosen fills on solved instances, `mine_support = 3`, uncapped.
**`given`** is read off genuine exemplars through the withheld tile ids at the same threshold —
364 T[2] entries (purity 0.962, all 47 tiles), 538 T[3] — so the only difference from an earned
table is the source of the tuples. Arms: `never_base`, `given`, `practice_gated` (unit-LP
certificate, `rr_s0`'s constants), `practice_early` (commit at era cycle 1), `practice_late`
(cycle 21), `practice_prov` (boundary commit, `ear/`). Logged, never acted on: oracle validity of
every fill beside the round-trip ceiling, `A_true`/`rand_k`, `held`/`live`, the width ladder, the
plant guard, typicality, `at_support`, and the grader's clean false-reject per level.

Gates (`selfcheck`, bit-for-bit): fork fidelity against `sampler.decode` at every (steps, width 1);
the donor's DP ≡ the flat gather; `make_table(s=4)` round-trips `recur._blockify`; truncating
T[2] 43 → 10 drops T[3] 498 → 0; grader clean pass .984 / .938 / .828, `seam` pass .222 / .000 /
.000 (L1 is a gate shortfall, reported not tuned).

## Findings

**Per-era e = 1 − support pass on 144 held-out instances (last-5-cycle means), 8 forwards/solve:**

| arm (T[2] / T[3] entries at end) | era 1 | era 2 | era 3 | earned-vs-given, eras 2 / 3 |
|---|---|---|---|---|
| `never_base` | .058 | .472 | .808 | — |
| `given` (364 / 538) | .056 | .260 | .608 | 1 |
| `practice_prov` (203 / 96) | .057 | .238 | .581 | 1.11 / 1.14 |
| `practice_late` (191 / 81) | .063 | .286 | .619 | 0.88 / 0.94 |
| `practice_gated` (192 / 16) | .056 | .278 | .639 | 0.92 / 0.85 |
| `practice_early` (11 / 2) | .056 | .385 | .715 | 0.41 / 0.47 |
| clean false-reject floor | .042 | .069 | .174 | |

Denominators (`e_never − e_given`) 0.003 / 0.213 / 0.200; era 1 is below the 1/144 metering
resolution and carries no information (single-tile repairability is 1.00 — the DGP's own statement).

1. **Earning the vocabulary recovers what `given` buys.** At eras 2–3 the three sensibly-timed
   practice arms sit in one cluster with `given` (fractions 0.85–1.14, inside `recital`'s ±0.15),
   `practice_early` at ~0.45, `never_base` 0.2–0.35 worse. Priced time is within 3 % across arms.
2. **`practice_prov` ≥ `given` is a tie at the top, on a strict subset.** Its 203 T[2] entries are
   all in `given`'s 364 (precision 1.000); the extra entries buy nothing. Cycle-to-cycle spread
   inside an era is ±0.05, prov's edge 0.02–0.03 on the last-5 window and ~0 on era means — a
   rank tie, not a win. Mined tables beat a random same-size subset of `given` in every audition
   cell (`cand < rand_k`, both levels, all eras) — concentration over coverage, as on RHM.
3. **The descent is purely vocabulary-carried.** There is no value head, and the plant guard is
   flat (held-out NLL 0.08 / 0.22 / 0.37 at mask sides 2 / 4 / 8, every arm, every era); within
   an era `never_base` and `given` are flat to noise. `never_base` is therefore search plus a
   free truth grader at the same budget, and it loses at every deep era.
4. **The grade of record held up in-loop.** Oracle fill validity (beside the clean-round-trip
   ceiling .885 / .688 / .406) orders the arms identically to support at every era: era 3
   `never_base` .135 → `given`/`late` .198.
5. **The level-3 vocabulary is structurally present and buys almost nothing at this operating
   point.** `cal1`: L3 base .861 → +T[2] .653 → +T[2]+T[3] .639; gated's 16-entry T[3] lands within
   noise of late's 81. The L3 auditions locate it: `given`'s T[3] nails a single-motif 4×4 hole
   (.37 vs candidates .7+) but on the 8×8 hole every T[3] table sits at .73–.88, and there the
   mined tables edge `given` (.73–.88 vs .83–.85). T[3] at support 3 is 538 of 10 752 at support 1;
   committed recall 0.02–0.10. Coverage is the binding term one level up.
6. **`practice_early` is foreclosed but not worse-than-never** (RHM: −0.37). T[2] 11 entries →
   T[3] 2; the ratchet constraint bites. Width is fixed across arms here, so a bad table costs
   nothing in rent — RHM's early penalty was foreclosure *plus* its pricing inversion. A design
   difference, not a contradiction.
7. **The taste gauge rises as truth falls.** Typicality pass .86 → .90 → .98 across eras, all arms
   alike, while validity drops to ~.2. Large holes are filled with the most common tiles; a fill
   of common tiles is maximally typical whether or not its edges agree.
8. **Certificate**: era-1 firing is textbook (`A` flat at .073 for ten cycles, fires at c21, 192
   entries); the era-2 L3 firing came at the earliest legal cycle mid-descent (`A` .927 → .594
   monotone, fired at .823 on 16 entries). `rr_s0`'s constants, not recalibrated — moot while
   T[3] is inert, the obvious calibration when it isn't.

## Interpretation (discussed with Jasper 2026-08-26 — argued, not measured)

- **(a) The ratchet's headline reproduces on a second substrate with a stronger control.** RHM's
  descent had a learning value and an inert plant; canvas has neither learning, so the whole
  earned-vs-given contrast is about which tuples are in the table, and the base arm's failure to
  buy depth with search + a free truth grader is the cleanest statement yet of "depth is
  unaffordable to the primitive action."
- **(b) Finding 2 is the third sighting of the same fact** (after `transpose`'s stale-table-beats-
  current and `ear`'s `practice_prov`): the vocabulary the agent writes for itself from its own
  successful work is the *used* part of the complete list, and the complete list's extra entries
  are legal-but-rare spellings that never help. Chunks store demand-concentration.
- **(c) Finding 7 is the two-organ dissociation, now in-loop.** Validity is a relation between
  tokens, typicality a property of tokens (`plant/` finding 11); had the memo's original plan —
  typicality as the grade of record — been followed, the loop would have graded its worst fills as
  its best and could not have separated the vocabulary arms from `never_base`.
- **(d) Level 3 is where the next question lives, and it is a coverage question.** The nesting
  works (gates, foreclosure); what the L3 table lacks is entries. Exemplar count is a knob on
  canvas, and `reread/lm`'s ~10×-per-half-level wall is directly testable in node 3.

## Caveats

- **Single seed, one tileset, one panel.** Ranks and signs are the currency.
- **The operating point is dominated**: at G = 8 the width ladder is inverted for every arm
  (w1 < w2 < w4 in error; `never_base` L3 .701 / .826 / .910) because `steps = G / width` and
  conditioning is what the budget buys on a masked model. All arms were metered at width 2, so
  ranks stand and the vocabulary gap survives at width 1 (`given` .49 vs `never_base` .70 at L3),
  but absolute `e` is above what the same budget buys at width 1. A rerun should use width 1.
- **`given` is a more-data ceiling, not the DGP-table ceiling.** It is read off 360 exemplars at
  support 3: at L2 that is the catalogue (all 47 tiles), at L3 the recurring subset. The
  `valid_blocks` table was not run; the distinction matters once T[3] carries weight.
- **The L1 grader gate falls short** (`seam` pass .222 against the ≤ .10 rule) — a 2×2 hole
  touches too few pairs; eras 2–3 hold.
- The plant's fine-tune ran and moved nothing measurable; whether a plant *can* move here is
  untested (RHM's `handle/` / `native/` question).

## Runs on disk

| tag | what |
|---|---|
| `smoke0` | quick 6-arm smoke — found that the budget did not bind and the miner was starved by the donor's cap |
| `cal0` | depth ladder under the first (broken) pricing: vocabulary arms strictly worse everywhere; diagnosed γ and the width rule |
| `cal1` | depth ladder under corrected pricing, 144 instances/era, width 2 — picked G = 8 |
| `cr_s0` | **the main run** — 6 arms, seed 0, 72 cycles, complete |

Volume `canvas-data` (profile `chromatic`): `/data/canvas_practice_ratchet/<tag>/setup.json`,
`<tag>/<arm>/results.json`. The committed reduction is [`results/reduced_cr_s0.json`](results/reduced_cr_s0.json);
figures (`figures/cr_s0/{cost_to_depth,audition,earned_vs_given}.png`) are gitignored and regenerate below.

## Reproduce

```bash
cd experiments            # conda glp; MODAL_PROFILE=chromatic
modal run -m canvas.practice.ratchet.ratchet::selfcheck
modal run -m canvas.practice.ratchet.ratchet::ladder --tag cal1 --budgets 2,4,8,16,32,64
modal run -m canvas.practice.ratchet.ratchet::run --tag smoke0 --quick 1
python3 canvas/practice/ratchet/launch_detached.py --tag cr_s0 --logdir /tmp/crlogs
python3 canvas/practice/ratchet/analyze.py --tag cr_s0 --fetch --figures
```

## Next steps (queued, not started)

**Node 2 — `native/` on canvas, the minting node**: consolidate `practice_late`/`prov`'s L2 table
(~200 tiles, precision 1.0) into the plant three ways as arms — a routing head π(entry | surround)
(RHM's port 1), and the port canvas makes natural, a minted tile token in the plant's own input
stream, as *substitution* (the four cells replaced) and as *annotation* (the token added beside
them; `native/SPEC.md:173`'s unrun sibling) — with `nf_s0`'s battery (table deletion after
consolidation, π's mass on a block's own cell spelling, the poison twin). **Node 3 — `spiral/`**:
re-earn L3+ over the minted plant with exemplar count as the variable, and `census/`'s arrival
pair · Rerun at width 1 for the record · Recalibrate the certificate once T[3] carries weight ·
The `valid_blocks` true-table `given` beside the exemplar-read one.
