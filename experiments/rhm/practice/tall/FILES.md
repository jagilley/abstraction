# tall — File Index

The pacing question at the repo's deep RHM setting (64-token sequences). **No `README.md` by
deliberate policy**, as for [`../ear/`](../ear/FILES.md) and [`../recital/`](../recital/FILES.md)
— the writeup is held for the joint multi-round writeup.

**Status: feasibility only.** The main run is not designed or launched. This file records what
the calibration established, because one of its findings changes the round's specification.

## Why the round exists

`recital`'s ladder topped out at level 3 of a 4-level grammar — simultaneously the deepest level
the agent could *earn* and one below the grammar's own ceiling. Every endogenous-signal failure
the arc found at level 3 is therefore confounded: **boundary artifact of the grammar's top, or a
real frontier effect?** At depth 6 the two come apart — the damage ladder runs to level 5 while
the earnable range stops at 3, so "the frontier" and "the grammar's ceiling" are different
places for the first time.

## The finding that changes the spec: **m = 4 is inadmissible for this substrate**

The round was specified at the repo's language-like LM setting, **L=6, m=4** (per
[`../../CLAUDE.md`](../../CLAUDE.md)). That setting is calibrated for *autoregressive scaling*
experiments. On the *sculpting* substrate it destroys the quantity the whole arc measures.

**1. The depth ladder collapses.** The oracle's repair distance `d*` over the nested ladder,
broken-only instances, measured on CPU:

| setting | L1 | L2 | L3 | L4 | L5 | gradient |
|---|---|---|---|---|---|---|
| depth 4, m=2 *(the arc's current setting)* | 1.58 | 2.25 | 2.67 | — | — | **1.69×** |
| depth 4, m=4 | 1.13 | 1.15 | 1.17 | — | — | 1.04× |
| **depth 6, m=2** | **1.71** | **2.37** | **3.23** | **4.52** | **5.54** | **3.25×** |
| depth 6, m=3 | 1.33 | 1.42 | 1.33 | 1.52 | 1.73 | 1.30× |
| depth 6, m=4 | 1.09 | 1.12 | 1.12 | 1.17 | 1.15 | 1.06× |

The 2×2 localises the collapse to **m, not depth**: depth 4 at m=4 collapses too, and depth 6 at
m=2 gives the *steepest* ladder the arc has ever had. Mechanism: at m=4 every feature has four
synonymous expansions, so some single commitment almost always re-derives the observed subtree —
deeper damage is not harder, and "cost-to-depth" has nothing to measure.

**2. The macro levels wall off.** `entries(l) = v·e(l)`, `e(l) = m·e(l-1)^s` — doubly
exponential, verified against `MC.true_tables` at levels 2–4:

| m | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| 2 | 16 | 64 | 1,024 | 262,144 |
| 3 | 24 | 216 | 17,496 | 114,791,256 |
| 4 | 32 | 512 | 131,072 | 8,589,934,592 |

At m=4 nothing above level 3 is representable — so the interior levels this round exists to test
could not be earned at all. (Building m≥3's L5 is also what OOM-killed the calibration host; the
sizes are now computed in closed form and only small tables are ever materialised.)

**Verdict: run at depth 6, m=2.** It is also the `m` the rest of the arc uses, so depth is the
only variable that moves and the depth-4 results stay directly comparable.

## What the earnable range implies for the design

Mining feasibility follows the same doubly-exponential law — the level-`l` miner needs
`≥ 3·entries(l)` observations at `mine_support=3`, i.e. at `mine_cap=8` spans/cycle:

| m=2 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| cycles to cover | 6 | 24 | **384** | 98,304 |

So **levels 2 and 3 are earnable and level 4 is not** (at any budget this round can afford),
even at m=2. The damage ladder (5 levels) is therefore deliberately **longer than the earnable
range (2–3)**, and eras 3–5 are pure consumption. That asymmetry is the round's instrument, not
a defect — but it does break one inherited gate's assumption (see G-S below).

## Code files

| File | Purpose |
|---|---|
| `tall.py` | The Modal app. `gates()` runs the arc's inherited gates at the tall ladder plus this port's own **T-1…T-4**; `feasibility()` times the real substrate — setup, references, reader/plant exactness, per-cycle cost — and writes `setup.json` to the volume. The arm loop is `recital.run_arm`, imported and unmodified. `_tall_cfg` is `ear`'s config with only the grammar substituted |
| `analyze_tall.py` | Reduction. Leads with the **operating-regime check** (where competence landed between `stale` and `floor`), because every other readout presupposes the loop opened that range; then the discriminator, the matched-budget exam with the spread compared against stream noise, the two mechanism instruments, deep-era separation, boundary placements and the instrument suite. `--figures` writes fig1-fig4 |
| `launch_detached.py` | `recital`'s session-isolated launcher, retargeted at this node. **Note it opens the log with `"w"`** — a relaunch overwrites the previous log, so that file is a live view, not an archive |

## Gates

| Gate | What it asserts | Status |
|---|---|---|
| C-M / C-R / G-D | the arc's: macro bit-identity to the true level move, the ratchet bites, nested damage is on-grammar | **pass at depth 6, m=2** |
| G-N / G-S / G-R | `ear`'s: nested ladder, on-grammar self-manufactured auditions, live recert undoes the cap | **pass on the earning prefix** (see below) |
| **T-1** | the ladder nests at all five levels and every node index is in range (`s^(depth-l)` nodes at level `l`) | pass |
| **T-2** | damage is 100% on-grammar at **every** ladder level, with the rejection acceptance rate the port pays (0.90–1.00) | pass |
| **T-3** | the oracle's repair distance is **monotone in ladder depth** — the cost-to-depth the arc measures | pass, **gradient 3.31×** |
| **T-4** | the action set and its grounding price at 32–62 moves | pass |

**G-S needed a scope fix, and the reason is load-bearing.** `ear.gate_manufacture` builds an
audition for the level each era *earns* (`era.level + 1`), so it is only defined where that
level is ≤ `max_macro_level`. At depth 4 that held for every era because the ladder was exactly
as long as the earnable range. Here it deliberately is not, so the inherited gate runs on the
**earning prefix** (`1:25,2:12`) and the consumption eras are covered by T-2/T-3. The gate was
not weakened — its domain was made explicit.

## Measured at depth 6, m=2 (`gates_remote`)

Ladder `1:25 → 2:12 → 3:6 → 4:3 → 5:1` (nested, verified).

| era | on-grammar | accept | `d*` mean | `d*` max | P(`d*`>4) | P(`d*`>6) | P(`d*`>8) | P(`d*`>10) |
|---|---|---|---|---|---|---|---|---|
| L1n25 | 1.000 | 0.996 | 1.71 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| L2n12 | 1.000 | 0.926 | 2.32 | 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| L3n6 | 1.000 | 0.977 | 3.30 | 7 | 0.21 | 0.004 | 0.00 | 0.00 |
| L4n3 | 1.000 | 0.949 | 4.58 | 10 | 0.48 | 0.16 | 0.016 | 0.00 |
| L5n1 | 1.000 | 0.898 | 5.67 | 13 | 0.65 | 0.33 | 0.13 | 0.039 |

**Action set** (`build_move_set`): 32 base moves; 48 / 56 / 60 / 62 with macro levels 2 / 3 / 4 / 5
— against 8 / 12 / 14 at depth 4.

**Pricing** (`beam_ground(n, budget, width)`): `n=32, b=8` → 257 (w1) / 482 (w2);
`n=56, b=8` → 449 (w1) / 842 (w2). Depth 4 used `g_budget=58`.

## Measured on GPU (`feas0`)

| quantity | depth 4 (measured) | depth 6 (measured) |
|---|---|---|
| setup (`_shared_plus`) | 565 s | **513 s** |
| per-cycle | 1.39 s | *(in flight)* |
| priced cost / solve | 58 | 482 (8.3x) |

**Setup does not scale with sequence length** — 513 s at 64 tokens against 565 s at 16 tokens,
i.e. 3-5x better than the structural projection. The substrate nets are small enough that fixed
overheads dominate; the 4x token count is nearly free at this size. Whether the same holds for
the arm loop (which pays 8x the beam evaluations per solve) is what the probe measures.

References at depth 6, m=2 (all five eras, healthy): on-grammar 1.000 everywhere, `d0` monotone
1.71 -> 5.51, `floor` 0.10 -> 0.82.

**`preflight` exists because `feas0` crashed twice on interface drift**, both times *after* the
~510 s setup had been paid: first `shared["generator"]` (the key is `generator0`), then
`plant_probe`'s own `shared["probe_clean"]`, which every round's *entrypoint* populates but
`build_shared`/`_shared_plus` do not. Fixing one key at a time costs a full setup per attempt,
so the interface was instead audited in one pass — every `shared[...]` read across
`ratchet.py`/`ear.py`/`recital.py` diffed against what the builders produce — which found
`probe_clean` to be the *only* remaining gap. `_tall_shared` now closes it and asserts the
complete key set, and `preflight` calls every imported function once at tiny sizes so drift is
caught in ~2 minutes instead of after a paid setup. Volume persistence for setup was considered
and not taken: with one gap found by audit and the preflight in place, a third re-pay is
unlikely, and a stale-checkpoint bug on the run everything else is sized from is the worse risk.

**Detail of the first crash** at `plant_probe(shared["generator"], ...)`
— the shared dict's key is `generator0` (each arm deepcopies it; there is no `generator`). Setup
and references had already completed and logged healthy. Fixed, and `read_acc` is now recorded
too: it is the quantity that compounds as `acc**span` when mining at level `l`, so it is the
direct empirical check on whether `max_macro_level=3` is reachable.

## Result: `tl_s0` is voided by a substrate failure, and the failure is diagnosed

**The loop did not sculpt at depth 6.** Competence recovered only **3.5-16.5%** of the range
between the two references the setup measured (`stale` 0.896 -> `floor` 0.102 in era 1), against
roughly 50-80% at depth 4, and within several eras `e` *rose*. Every arm finished near the
never-commit baseline, so the schedule-shape and pacer-quality questions are unanswerable from
this run: the spread of arm means is **0.0255, below the +/-0.034 stream-position noise** the
depth-4 arc measured.

**Root cause: the value head is signal-starved, not the controller, reader, or plant.**

| | depth 4 | depth 6 |
|---|---|---|
| controller final acc | 0.793 | **0.820** (fine) |
| reader `read_acc` | — | **1.000** (exact) |
| plant parse/infill | flat | flat (0.67-0.70 / 0.71-0.76) |
| **stale value buffer terminal success** | **0.296** | **0.076** |

At 64 tokens with 32 base moves, stale rollouts succeed 7.6% of the time, so the value model
trains on a ~4x sparser positive signal and cannot rank states. The width ladder confirms the
policy is not search-limited: `e` at beam width 1 / 4 / 16 is 0.80 / 0.78 / 0.81.

**This is a sizing failure, not a design failure**, and it is the thing the feasibility gates
did not test. They covered trainability, reader exactness, on-grammar damage, the ladder
gradient, and pricing — but never *does the policy descend toward the floor in a realistic
cycle budget*. `cal0` already contained the warning (era-1 `e` went 0.836 -> 0.867 over 35
cycles, i.e. worse) and it was reported without being flagged. The lesson for the next attempt
is that **descent toward the floor belongs in the feasibility gate set**, and that the value
buffer needs denser positives at 64 tokens (easier damage during collection, a larger
collection budget, more `value_episodes`, or a curriculum).

### What survives the failure

Two findings are regime-independent and worth carrying:

1. **Endogenous pacers cannot traverse a ladder longer than the earnable range.** `pace_cert`
   advanced at c16 and c9, then sat in era 3 for 66 cycles (75.8% of `T`); `pace_vocab` at c24
   and c28, then 40 cycles. Neither reached eras 4-5. Era 3 earns level 4 > `max_macro_level`,
   so both the certificate and the admission rate are undefined and no advancement signal
   exists. Structural, not statistical — and invisible at depth 4, where the ladder ended
   exactly at the earnable range.
2. **The mined table cannot churn.** The entry-identity instrument records **zero removals** in
   every arm at both levels — and this is true *by construction*: miner counts only increase, a
   support threshold on a monotone count is monotone, and the ratchet filter can only drop an
   entry if the lower table shrinks, which it cannot. So `recital`'s open mechanism question
   ("same table or different?") has an a priori answer — always the same table plus additions —
   and reduces to "does the increment matter?", which the recert counterfactual already answers
   (0 swaps here across 45 recerts, 0 swaps in every depth-4 seed). The question should be
   retired rather than carried into the writeup.

### What is confounded

The level-3 certificate **does** fire at interior level 3 — non-provisionally, in every arm
given >= 28 cycles in era 2 (`sched_earnable` c15, `pace_vocab` c18, `pace_cert` c9), while
arms with 16 cycles committed provisionally. At depth 4 it needed 35-40 cycles and usually never
fired. That is the round's discriminator pointing at *boundary artifact*, but it **cannot be
claimed**: the two settings are in different operating regimes, and a detector going quiet in a
loop that is not learning is not evidence that it went quiet because learning finished. The
`cal0`-based prior recorded at checkpoint 2 is retracted.

### Cost

23,893 s actual against 4.8 GPU-h projected. 565 cycles at **42.3 s/cycle** against the 22.0
s/cycle measured probe-free: **probes were ~47% of runtime**. The projection applied a 40%
*overhead* to cycle time, but the probe block runs 13 beam evaluations (3 width-ladder + 5
all-eras + 5 fixed-reference — the last five added this round) and fires 24-28 times per arm
once boundary, matched-budget and terminal triggers are counted, at ~68 s each.

## `dens0`: the value-densification test and the DESCENT GATE

**Sweep** (in-run `3/8` control reproduces `tl_s0`, so no cross-run inference is needed):

| n_corrupt / collect budget | terminal success | |
|---|---|---|
| **3 / 8** (the `tl_s0` config, control) | **0.070** | reproduces `tl_s0`'s 0.076 |
| **1 / 8** | **0.133** | best — 1.9x the control |
| 1 / 12 | 0.110 | |
| 2 / 12 | 0.068 | |
| 1 / 16 | 0.109 | |
| *depth 4 reference* | *0.296* | the target |

**`n_corrupt` is the whole lever; collection budget is not — and slightly hurts** (1/8 > 1/12 >
1/16). That half of the search-difficulty diagnosis is **retracted**: with `explore_eps=0.3`,
extra rollout steps are as likely to break an already-correct configuration as to fix a broken
one, so more attempts do not buy more successes. Fewer targets does buy them, as predicted.
Densification closed roughly half the gap to depth 4 and did not reach it.

**Descent gate** (22 cycles of era 1, densified value head, probes minimal):

| | value |
|---|---|
| `stale` / `floor` | 0.883 / 0.102 |
| e first-5 / last-5 | 0.7005 / 0.6781 |
| **recovered fraction (last-5)** | **0.262** |
| recovered fraction at cycle 1 | 0.233 |
| best sustained (c12-c21, pre-commit) | ~0.30-0.35 |
| pre-commit slope | -0.004 / cycle |
| *`tl_s0` era 1* | *0.035-0.165* |
| *depth 4* | *0.50-0.80* |

Better than `tl_s0`, still short of depth 4 — and the decomposition matters more than the
headline: **0.233 of the 0.262 was already present at cycle 1**, supplied by the densified value
head rather than by the loop. The loop does now descend (it did not in `tl_s0`) but at
-0.004/cycle, so era 1 alone would need ~55 cycles to reach depth 4's 0.50.

### The finding that matters most for any re-run: **committing costs more than it buys**

| cycle | n_moves | width | g/solve | e |
|---|---|---|---|---|
| c20 | 32 | 2 | 482 | 0.638 |
| c21 | 48 (L2 committed) | 2 | 482 | **0.625** (best of the run) |
| c22 | 48 | **1** | 385 | **0.805** |

The L2 commit at c21 was *good* — non-provisional, 16 entries, recall 0.786. But committing
grows the action set 32 -> 48, and at `g_budget=482` `fit_width` then drops the beam from width
2 to width 1. The resulting jump is **+0.180 against a mean cycle-to-cycle |delta| of 0.019 and
a maximum of 0.039** — a 4.6x outlier, not noise.

At depth 4 the same width transition occurred (`g_budget=58`, n=8 w2 -> n=12 w1) and the macro
was worth more than the width. At depth 6 the action set is 4x larger, so beam width buys more
and each macro covers a smaller fraction of the sequence, and the sign flips. **Under this
pricing the tall setting penalises the very act the arc exists to study.** Holding width 2 after
commit needs `g_budget >= 722` (n=48) or `>= 842` (n=56) -- ~1.5-1.75x the priced cost per
solve. This must be fixed before any re-run, or the vocabulary-earning question is confounded by
a width penalty.

## Runs on disk

| tag | what |
|---|---|
| `tl_s0` | **the main run**: 6 arms at matched `T=5.6e7`, five-era ladder, `budget=8`. Voided by the substrate failure above |
| `cal0` | calibration: one clock-paced `practice_climb`, 35 cycles era 1 + 35 era 2 |
| `dens0` | value-densification sweep + the descent gate |
| `feas0` | the feasibility probe: full setup, references, plant, reader exactness, 6 cycles of `practice_climb` at depth 6, m=2, `max_macro_level=3`, `budget=8`, `g_budget=482` |

Volume (`rhm-scaling-data`): `/data/rhm_practice_tall/<tag>/`.

## Calibration (`cal0`): measured firing cycles

One clock-paced `practice_climb` arm, 35 cycles in era 1 then 35 in era 2. Both pacers are
logged in every arm regardless of which is read, so one arm measures all three detectors.

| era (earning) | cert (mfg/pol) | vocab (admission) | task (metered e) | table at end |
|---|---|---|---|---|
| era 1 (L2) | **c12** | **c23** | c13 | 13 entries, recall 0.643 |
| era 2 (L3) | **c22** | **c20** | c18 | 17 entries, recall 0.071 |

**The level-3 certificate fires, at c22.** At depth 4 the L3 certificate needed 35-40 cycles and
usually never fired inside the 30-cycle clock; the L3 commit here is *non-provisional*. Level 3
is interior at depth 6 and top-of-earnable at depth 4, so this is the round's discriminator
firing on the boundary-artifact side — from calibration alone, n=1 arm, clock-paced, and to be
confirmed by the main run.

Costs: **22.0 s/cycle over 70 cycles** (the 6-cycle `feas0` figure of 36 s/cycle was all
width-2 pre-commit and overestimated by 64%); priced cost per cycle is **not** constant —
501,593 in era 1 and 684,063 in era 2, rising with `n_moves` as macros commit.

## Approved design for the main run

`depth 6, m=2, max_macro_level=3`, ladder `1:25,2:12,3:6,4:3,5:1`, single seed, commit policy
fixed across arms (certify-else-provisional-at-boundary). Six arms — `never_base` is cut as a
full arm, its depth-necessity role moving to a pre-loop calibration ladder plus the in-arm
width-ladder instrument:

| arm | advancement policy |
|---|---|
| `given` | the DGP's own tables — the ceiling reference |
| `sched_uniform` | equal priced share per era |
| `sched_bottom` | bottom-heavy (the shape that won at depth 4) |
| `sched_earnable` | **new to this setting**: time concentrated in eras 1-2, where vocabulary can actually be earned; eras 3-5 are consumption. Only askable because the damage ladder is now longer than the earnable range |
| `pace_cert` | the unit-LP certificate |
| `pace_vocab` | the novel-tuple admission rate |

Instrumentation added this round, both requested by `recital`'s mechanism check: **per-cycle
mined-table entry-identity sets** (so table *membership* churn is observable, not just size and
recall) and a **fixed-reference policy probe** (policy quality on a set that does not move with
the era, which the depth-4 logs could not separate from era-relative competence).

Era lengths and total budget come from offline detector replay on `feas0`'s own series at
predicted-firing + ~50% margin — deliberately not generous padding, and explicitly applying the
30-cycle-truncation lesson from `ratchet`/`ear`, where the fixed clock cut the level-3
certificate about five cycles short. `budget=8` by default; `budget=10` is restored if the
measured per-cycle cost keeps the full run within ~5 GPU-hours, since that is the only trim with
real epistemic cost (13% vs 3.9% of L5 instances above the oracle's reach).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/tall/tall.py::gates_remote

python3 rhm/practice/tall/launch_detached.py --fn feasibility --tag feas0 \
    --cycles 6 --max-macro-level 3 --budget 8 --g-budget 482 \
    --eras "1:25,2:12,3:6,4:3,5:1" --arm practice_climb
```

## Inherited, not copied

`../recital/` (`run_arm`, `ARMS`, the three pacing detectors), `../ear/` (config, shared setup,
grader, gates), `../ratchet/` (substrate, macros, pricing, references), `../crystallize/units.py`
(action space, hierarchical damage, exact-DP oracle). Nothing in those folders is modified.
