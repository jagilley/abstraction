# transpose — File Index

Complete file-by-file reference for this node. **No `README.md` by deliberate policy** — as for
[`../ear/`](../ear/FILES.md), [`../recital/`](../recital/FILES.md) and
[`../tall/`](../tall/FILES.md), the writeup is held for a joint multi-round writeup and for
discussion first. This file plus the code comments are the record; it documents **machinery and
calibration only**, not results.

## What the round installs

Every round of the arc so far has measured commitment policy in a regime where **post-commit
invalidation is structurally impossible**. The substrate has a *boundary* conditioning gap
(hidden ancestors, so launch-time observation is informative — what state-conditioned commitment
buys) and an *epistemic within-unit* gap (a committed macro forgoes priced groundings that would
have revealed facts about the grammar), but no **news** gap: the grammar is static, damage
recurs i.i.d. within an era, and the plant is inert. The arc's own instruments say so in four
independent places — recert fired **0/24** (`ear`) and **0/288** (`recital`); `tall` proved the
mined table **cannot churn**; the étude's post-commit drift was exactly **0.0000**;
`crystallize` found "nothing to certify".

This round installs the missing gap — **within-run grammar drift** — and re-runs the
commit-policy comparison on top of it. Nothing else changes: the substrate, the pricing, the
detector knobs, the era clock and the grader are `ear`/`recital`'s, imported.

**Why grammar drift and not the alternatives.** Surface masking and stochastic execution inside
a committed unit both make *launch* a bet; neither makes already-committed content *become
wrong*. A committed level-*l* macro entry is a level-1 tuple that **was** derivable at level *l*;
resampling a level-*l* rule cell can make it un-derivable, at which point applying the macro
writes an off-grammar span and the repair fails. That is the only lever considered that delivers
post-commit invalidation.

**Why not `full_loop`'s OU drift.** `directed_sculpting/full_loop`'s `advance_drift` /
`rhm_drift.calibrate_sigma_event` drift the **synonym mixture weights** — which of the *m* rules
is chosen — so the drift is soft and measured in nats of KL. It cannot invalidate a committed
macro: every entry stays grammatical, only its frequency moves. Hard support drift is what this
substrate needs, and its honest currency is a **fraction** (of the level's vocabulary surviving
the event), not nats, because the support changes and the KL is infinite. What is kept from that
machinery is the idea that matters: a drift **event** of *measured* magnitude, calibrated to a
target rather than guessed, with per-event and cumulative magnitudes reported separately.

**Drift is unannounced.** No arm reads a drift signal. Re-mining, count decay and recert all run
on their own clocks; the news has to arrive through the error stream. That is what makes the gap
a conditioning gap rather than a scheduled event.

**Drift is confined to levels ≥ 2**, which leaves the bottom layer (`rules[depth-1]`, level-1
feature → leaf tuple) untouched — so `canon`, the inverse maps, the reader and the generator's
block head all stay exact and the effect is isolated to the vocabulary rather than confounded
with a broken parse. The entrypoint asserts this.

## Code files

| File | Purpose |
|---|---|
| `drift.py` | The drift primitives, kept out of the app so they are auditable and gate-testable with no substrate. `resample_cells` is one drift event: *n* (feature, rule) cells of the level-*l* layer resampled, preserving `generate_rules_distinct`'s within-feature distinctness, every other layer bit-identical. `table_survival` / `corpus_survival` / `half_life_cycles` are the magnitude currencies (agent-independent: the world's motion, not any arm's luck). `epoch_of` / `n_epochs` / `rules_trajectory` are the schedule, drawn once from a dedicated `drift_seed` stream. **`DecayMiner`** is `MC.Miner` with geometric count decay — the structural build item `tall` forced (see below) — and is `MC.Miner` *exactly* at `decay = 1.0`. `entry_set` / `churn` are `tall`'s entry-identity instrument. `simulate_miner` is the offline payback-bracket calibration |
| `transpose.py` | The Modal app. Forks **only `recital.run_arm`**; `build_shared`/`_shared_plus`, `beam_moves`, `context_instances`, `finetune_generator`, `value_steps`, `measure_refs`, `plant_probe`, `macros`, `grader`, `node_at`, `step_detector`, `vocab_rate` and `write_results` are all imported. `build_world` precomputes **every grammar the run passes through** and everything derived from it (rules, `rules_t`, true tables, all three held-out sets per era, the clean probe pool, magnitudes, and `epoch_refs`) — once, at setup, so **every arm lives in literally the same world at literally the same cycle** and a difference between arms cannot be a difference in the draws. `epoch_refs` carries the exact-DP floor at *every* epoch (`tall`'s lesson: descent toward the floor belongs in the gate set, and a moving world moves the denominator). New per-cycle log keys: `epoch`, `stale`, `churn`, `bank_fresh`; new event kind `world`. Entrypoints: `transpose` (main), `cal_descent` (the GPU descent gate at a ladder of drift specs), `cal_drift{,_remote}` (the offline magnitude / payback-bracket / forgetting calibration), `selfcheck{,_remote}` |
| `analyze_transpose.py` | Reduction. `--fetch` pulls from the volume; `--fidelity` compares against `ear/figures/er_s0` cycle for cycle over ten log fields. The default report gives the world's motion, the commit-policy grade (per era and on the terminal all-eras exam, with earned-vs-given measured against **both** forks of `given`), the **post-commit unit error trajectory**, committed-table **staleness**, **recert** firings and swaps, vocabulary **churn**, and the instrument suite. `--figures` writes fig1–fig4 |
| `launch_detached.py` | Session-isolated detached launcher (`start_new_session=True`). Opens its log with `"a"`, not `"w"` — `tall`'s launcher overwrote the log on relaunch, so that file was a live view rather than an archive |

## The structural build item: a vocabulary that can lose entries

`tall` established that the arc's miner **cannot churn** — counts only increase, a support
threshold on a monotone count is monotone, and the ratchet filter can only drop an entry if the
lower table shrinks, which it cannot. Under a static grammar that is harmless (a stale entry is
a contradiction in terms). Under drift it is fatal: an entry legal when it was mined stays in
the table forever.

The mechanism chosen is the cheapest one that is also **agent-internal**: `DecayMiner` ages every
count by `mine_decay` once per cycle and forgets a tuple once its count falls under `mine_floor`.
A tuple observed at rate λ settles at count λ/(1−γ), so it survives while λ > (1−γ)·support and is
forgotten about log(support/count)/log(γ) cycles after the world stops producing it. No oracle,
no drift signal, no re-grading — recency-weighted mining on its own clock. It is carried by **one
arm** (`climb_decay`), so "can the vocabulary forget?" is a measured contrast rather than a
substrate change, and `decay = 1.0` is the parent miner exactly (gate T-4), which is what makes
the fidelity replay structural rather than lucky.

## Arms

`vocab` ∈ {base, true (frozen initial tables), true_live (tracking), earned}; everything else is
`ear`'s. **The `given` arm forks under drift because the true table does**: a frozen-table arm
bounds what a perfect but static handover is worth, a tracking arm bounds what perfect tracking
is worth, and the gap between them prices the news the agent has to buy for itself.

| arm | vocabulary | commit policy | recert | miner |
|---|---|---|---|---|
| `never_base` | base level-1 moves forever | — | — | monotone | 
| `given` | the DGP's tables from cycle 1, **frozen** | — | — | monotone |
| `given_live` | the DGP's tables, **re-read every epoch** | — | — | monotone |
| `practice_gated` | earned | unit-LP certificate (`ratchet`'s control) | off | monotone |
| `practice_prov` | earned | provisional at the era boundary (`ear`'s rescue) | **on** | monotone |
| `prov_norecert` | earned | provisional at the era boundary | **off** | monotone |
| `practice_climb` | earned | certify-else-provisional (the settled policy) | **on** | monotone |
| `climb_decay` | earned | certify-else-provisional | **on** | **decaying** |

`recital`'s eight inherited arms (`practice_late`, `practice_self`, `taught`, the pacers, …) are
carried unchanged so the fidelity replay can run them. `gated_decay` exists for the
certificate × decay cell and is not in the default set.

**Recert reaches every committed level under drift** (`recert_all`, ascending so a swapped
level-2 table lets level 3 rebuild over it, which is gate G-R's mechanism). `ear`/`recital`
recertify only the era's own level; under a static grammar that is equivalent, but under drift
level-2 content keeps going stale through era 3 and a channel that could not reach it would
decide the round by construction.

## Gates

| Gate | What it asserts | Where |
|---|---|---|
| C-M / C-R / G-D / G-N / G-S / G-R | the arc's and `ear`'s | inherited via `recital.selfcheck` |
| G-P | `recital`'s: the pacers are one detector, and the descent precondition is load-bearing | inherited |
| **T-1** | a level-*l* resample touches **only** the level-*l* layer, changes exactly `n_cells` cells, and preserves within-feature distinctness — in particular the bottom layer is bit-identical at every drift level ≥ 2 | `transpose.gate_drift` |
| **T-2** | drift is **level-indexed in the vocabulary too**: a level-2 resample changes T2 *and* T3; a level-3 resample changes T3 and leaves T2 **bit-identical**. That alignment is the round's instrument | `gate_drift` |
| **T-3** | **post-commit invalidation is real**: a table that was perfect before the event is not perfect after it — the precondition the whole arc has lacked | `gate_drift` |
| **T-4** | the decaying miner is `MC.Miner` **exactly** at decay 1.0 (tables bit-identical on the same stream), and **does** forget at decay < 1 | `gate_drift` |
| **T-5** | with `drift_every = 0` the trajectory has exactly one epoch and it is the pristine grammar — drift off is a no-op by construction, not by care | `gate_drift` |

Measured by `selfcheck_remote` at (v=8, s=2, depth=4, m=2, rule_seed=0), all passing:

- T-2, one cell: a **level-2** resample leaves L2 vocabulary survival 0.929 and L3 0.857; a
  **level-3** resample leaves L2 at **1.000** and L3 at 0.929. The level indexing is exact.
- T-3: a frozen perfect L2 table drops to precision **0.929** after one level-2 event (1 of 14
  entries invalidated) — the flat baseline is broken by construction.
- T-4: 24 tuples forgotten by a γ=0.8 miner after the stream stops; 0 at γ=1.0, tables identical.

Note the true tables carry **14** distinct level-2 tuples and **56** level-3 tuples (not 16/64):
`generate_rules_distinct` guarantees distinctness only *within* a feature, so a few tuples
collide across features. `MC.grade_table` works on sets, so every survival/precision figure here
is over the distinct sets.

## Fidelity

With `--drift-every 0` the drift epoch is 0 forever, every added instrument is gated behind
`drift_on`, and no new code consumes randomness — `build_world` is numpy-local plus deterministic
`torch.no_grad()` beams, so it cannot shift the global torch stream. `tf_s0` therefore runs
`ear`'s own eight arms, in `ear`'s own order, and must reproduce `er_s0` cycle for cycle
(`recital`'s `rf_s0` convention, extended from `e`/`t_cum` to ten log fields).

**Why all eight**: `ratchet`/`ear` do not reset torch's global RNG between arms, so arm *k*'s
stream depends on how many cycles arms 1..*k*−1 ran.

## Pre-run offline calibration (`cal_drift`, zero GPU)

Three measurements, all on the DGP's own tables so nothing depends on a trained substrate.

### (1) Magnitude — the world's motion per event

Per-event survival for one resampled cell, and the resulting **half-life of a frozen table** in
cycles at each candidate period (rule_seed 0; the cell drawn matters, so survival varies by draw):

| drift level | cells | L2 survival | L3 survival | corpus survival | half-life @P=3 | @P=5 | @P=10 |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.929 | 0.964 | 0.941 | 28.1 | 46.8 | 93.5 |
| 2 | 2 | 0.857 | 0.839 | 0.632 | 13.5 | 22.5 | 45.0 |
| 2 | 4 | 0.714 | 0.554 | 0.335 | 6.2 | 10.3 | 20.6 |
| 3 | 1 | **1.000** | 0.929 | 0.938 | 28.1 | 46.8 | 93.5 |
| 3 | 2 | **1.000** | 0.857 | 0.818 | 13.5 | 22.5 | 45.0 |
| 3 | 4 | **1.000** | 0.714 | 0.650 | 6.2 | 10.3 | 20.6 |

`corpus_survival` (the fraction of pristine derivations still parsing to their own root) is far
more sensitive than vocabulary survival, because a derivation needs *every* node legal: a
depth-4 tree has 4 level-2 nodes but only 2 level-3 nodes, so level-3 drift is structurally
gentler on the corpus at equal vocabulary damage. Cell-to-cell variance is large (a frequently
used rule breaks more derivations), which is why the run's own `world` block reports the realised
trajectory rather than this expectation.

### (2) The payback bracket — where compiling is worth anything

`simulate_miner` replays the mining stream against a drifting grammar with no substrate at all,
at the mining rate the parent run actually paid (below), and reads the steady-state quality of
the built table **against the current truth**. Tail means over the last 30 of 90 cycles,
level 2, one cell per event, `mine_support = 3`:

| period | decay 1.0 (the arc's monotone miner) | 0.98 | 0.95 | 0.90 | 0.80 |
|---|---|---|---|---|---|
| P=2 | recall .942 / prec **.420** | .907/.489 | .870/.737 | .754/.906 | .323/.972 |
| P=3 | .942 / **.479** | .922/.528 | .873/.764 | .733/.963 | .353/1.000 |
| P=5 | .927 / **.594** | .900/.648 | .877/.771 | .793/.948 | .292/.978 |
| P=10 | .967 / **.807** | .960/.854 | .960/.906 | .838/.953 | .298/.981 |
| P=30 | 1.000 / .867 | 1.000/.950 | 1.000/1.000 | .910/1.000 | .431/1.000 |

The monotone miner's failure mode under drift is **precision, not recall** — it never loses a
tuple, so it accumulates stale ones. Decay trades recall for precision monotonically, and
**over-forgetting (γ=0.80) collapses recall to ~0.3 at every period**, which is the "churn faster
than a unit pays back" edge on the decay axis rather than the drift axis. A table frozen at c25
ends the 90-cycle run at precision 0.29–0.31 / recall 0.29 at P=5, i.e. it has lost roughly
70% of its validity.

This simulation is the **optimistic limit** (a perfect parser reading a uniform sample of the
current true table); the real miner reads through the agent's own reader on solved
configurations, which is measurably more starved — see (3).

### (3) The measured mining stream (from `ear/er_s0`'s own logs, no new compute)

| level | obs/cycle | distinct by end of its era | at support 3 | implied λ per tuple |
|---|---|---|---|---|
| 2 | **8.00** (`mine_cap` binds in every arm, every cycle) | 12–13 of 14 true | 10 | ≈ 0.67 |
| 3 | 8.00 | 26 of 56 true | 13 | ≈ 0.31 |

This is what sets the **forgetting horizon**, whose closed form is
`cycles-below-support = log(support·(1−γ)/λ)/log(γ)`:

| γ | steady count at λ=0.67 (L2) | at λ=0.31 (L3) | cycles below support (L2) |
|---|---|---|---|
| 0.98 | 33.5 | 15.5 | 119 |
| 0.95 | 13.4 | 6.2 | 29 |
| 0.90 | 6.7 | **3.1** | 7.6 |
| 0.80 | 3.4 | **1.6** | 1.4 |

Level 3 is the binding constraint: at γ ≤ 0.90 a level-3 tuple's steady count sits *at or below*
`mine_support`, so the level-3 table would be gutted for a bookkeeping reason — and `T3 ⊆ T2 × T2`
means an over-forgotten level-2 table forecloses level 3's *representation*, which is
`ratchet`'s C-R failure mode. **γ = 0.95** is the largest decay that keeps the sparsest level's
steady count clear of the support threshold (2.1×) while still shedding a level-2 tuple within
about one era of the world dropping it. It is carried into the GPU descent gate against γ = 0.90
rather than asserted.

### (4) The GPU descent gate (`cal_descent`)

`tall` was voided because its feasibility gates covered trainability, exactness, on-grammar
damage, the ladder gradient and pricing, and never asked *does the policy descend toward the
floor in a realistic cycle budget*. Drift is exactly the kind of change that can break that: the
controller, generator and value are all trained on the pristine grammar, so a world that moves
too fast degrades the substrate rather than testing the vocabulary. `cal_descent` therefore runs
one setup, one era, the same arms at a ladder of drift specs (`every/cells/level`), with an
in-run `0/1/2` control so no cross-run inference is needed, and reads the **recovered fraction
against that epoch's own moving `stale`/`floor`**.

**The finding that set the round's specification: drift at level 2 is INADMISSIBLE; drift at
level 3 is admissible.** The readout is arm separation — `e(never_base) − min e(vocabulary
arms)`, on the tail of each epoch, against `recital`'s measured ±0.034 stream-position noise:

| spec | ep1 | ep2 | ep3 | ep4 | ep5 | ep6–9 | end-of-run `stale` |
|---|---|---|---|---|---|---|---|
| `0/1/2` (no drift, control) | — | — | — | — | — | **+0.127** @c40 | 0.331 |
| `4/1/2` | +0.003 | +0.003 | +0.017 | +0.016 | +0.010 | +0.027 / −0.009 / +0.056 / +0.050 | 0.948 |
| `8/1/2` | +0.004 | +0.056 | +0.035 | +0.054 | −0.037 | — | 0.810 |
| **`8/1/3`** | **+0.095** | **+0.161** | **+0.190** | **+0.111** | **+0.115** | — | 0.596 |

Under **level-2** drift the arms stop separating **within one or two events** — separation sits
at or below the noise floor from ep1 on and goes negative twice — while `stale` runs away to
0.81–0.95 and the exact-DP `floor` follows it from 0.195 to 0.61. That is the substrate failing,
not the vocabulary being tested, and it is `tall`'s voided-run signature caught before the fact.
Under **level-3** drift at the same event count the loop stays alive and discriminating at every
epoch, at times separating *more* than the undrifted control.

The mechanism is the one the offline magnitude table predicts structurally: a depth-4 tree has
**four** level-2 nodes but only **two** level-3 nodes, so a level-2 rule cell participates in
twice as many derivations (measured per-event corpus survival **0.71** vs **0.86**), and the
level-2 layer is what the base level-1 moves' rendering and the generator's infill depend on
most directly. Drifting level 2 degrades the **executor**; drifting level 3 moves the
**vocabulary**.

**Consequences for the main run's design, and what they cost.**

- `drift_level = 3`. Gate T-2 then guarantees the level-2 vocabulary is *bit-identical*
  throughout, which buys a **paired within-arm control**: the same arm holds two committed
  tables, one that provably cannot go stale and one that can, in the same world at the same
  cycle. `held − true` (frozen content vs perfectly tracking content, on the same set) is
  therefore readable at both levels, and the level-2 series is the "the world got harder"
  baseline the level-3 series has to beat.
- The cost is that all of the round's news is about **level-3** committed content, which the arc
  earns thinly (11–12 entries, recall 0.14–0.18) and consumes only in era 3.
- `drift_start = 33`, `drift_every = 8`: era 1 is left pristine, so the level-2 vocabulary is
  earned in a clean world and the era-1 cell stays directly comparable to `ear`/`recital`; the
  event density is the 8-cycles-per-event the gate verified; and no event lands on an era
  boundary (events at c33/41/49/57/65/73/81/89, boundaries at c30/c60), so "the world moved" is
  never confounded with "the era advanced".
- `mine_decay = 0.95` by the offline forgetting-horizon rule. The gate could not separate γ=0.95
  from γ=0.90 (differences of 0.002–0.020, inside noise) because it ran era 1 only, where under
  level-3 drift nothing the miner holds can go stale — there is nothing to forget yet. The knob
  is therefore set by (3) above, whose binding case *is* this round's drifting level: at λ=0.31
  a γ=0.90 miner's steady count sits at the support threshold itself.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | attached `--quick` smoke, 8 arms, drift on (every 3 cycles, 1 cell, level 2), decay 0.90 |
| `cal0` | the GPU descent gate: 4 drift specs × 4 arms × 40 cycles, one era. 1259 s. **Set the round's specification** (level 2 inadmissible; see above) |
| `tf_s0` | the fidelity replay: `ear`'s eight arms, `ear`'s order, `--drift-every 0`. 1304 s. **PASSES: all eight arms bit-identical to `ear/er_s0` across all 90 cycles**, over ten log fields (`e`, `t_cum`, `n_moves`, `level`, `era`, `width`, `succ`, `dres`, `e_practice`, `g_per_solve`) — max\|delta\| = 0.00e+00 everywhere |
| `tp_s0` | the main run: 8 arms, seed 0, 3 eras × 30 cycles, drift level 3, 1 cell every 8 cycles from c33 (8 events), `mine_decay` 0.95 |

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_transpose/<tag>/<arm>/results.json`, with
`setup.json` beside them (which carries the whole `world` block: per-epoch magnitudes, the cells
that changed, and the moving references). Fetched copies live in `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/transpose/transpose.py::selfcheck_remote
modal run rhm/practice/transpose/transpose.py::cal_drift_remote

python3 rhm/practice/transpose/launch_detached.py --fn cal_descent --tag cal0 \
    --specs "0/1/2,8/1/2,4/1/2,8/1/3" --cycles 40 --eras "1:6" \
    --arms "never_base,practice_climb,climb_decay:mine_decay=0.95:nm=climb_d95,climb_decay:mine_decay=0.90:nm=climb_d90"

# fidelity replay of `ear` (drift off == `recital` in its `ear` mode == `ear`)
python3 rhm/practice/transpose/launch_detached.py --fn transpose --tag tf_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_late,practice_self,taught,practice_prov,practice_climb" \
    --drift-every 0 --t-budget 0 --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 \
    --n-aud 192 --mfg-source child --mfg-render canon --mfg-min-bank 32 \
    --recert-every 5 --recert-margin 0.05 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

python3 rhm/practice/transpose/analyze_transpose.py --tag tf_s0 --fetch --fidelity
python3 rhm/practice/transpose/analyze_transpose.py --tag tp_s0 --fetch --figures
```

## Seeds

`--seed` sets damage draws, metering sets, mining and probe RNG; `--rule-seed` / `--train-seed`
set the DGP draw and the substrate, defaulted to `ratchet`'s (`0`/`0`/`1`) so every prior run
reproduces. **`--drift-seed` is a separate stream**, so the world's motion is independent of
every arm's randomness and of the DGP draw. A replicate varies the triple together as
`seed = k, rule_seed = k, train_seed = k + 1` (`recital`'s convention).

## Inherited, not copied

`../recital/` (`run_arm`, `ARMS`, `step_detector`, `vocab_rate`, `parse_arms_nm`, the gates),
`../ear/` (config, shared setup, grader, `node_at`, `write_results`), `../ratchet/` (substrate,
`macros.py`, pricing, beam, reader, online plant and selector, references),
`../crystallize/units.py` (the level-indexed action space, hierarchical damage, exact-DP oracle).
Nothing in those folders is modified.
