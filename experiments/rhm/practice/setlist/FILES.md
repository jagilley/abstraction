# setlist — File Index

Complete file-by-file reference for this node. **No `README.md` by deliberate policy** — as for
[`../ear/`](../ear/FILES.md), [`../recital/`](../recital/FILES.md), [`../tall/`](../tall/FILES.md)
and [`../transpose/`](../transpose/FILES.md), the findings go into the joint writeup at
[`../typed_gaps/README.md`](../typed_gaps/README.md). This file documents **machinery and
calibration only**.

## What the round installs

[`../transpose/`](../transpose/FILES.md) installed a live conditioning gap by resampling rule
cells — **truth-news**, "what is so has changed" — and measured that committed chunks cannot
consume that currency. This round supplies the currency the same reading says they *can*:
**the grammar is fixed and the consumption distribution drifts**. Nothing becomes false; the ask
moves.

**What "demand" is here.** A task instance is (r\*, damaged configuration): a target root and a
node that must be re-derived. Which vocabulary entry is *demanded* at that node is fixed by the
**clean derivation** the damage was applied to — by r\* and by the rule choices on the path down
to the node. So the demand *is* the generative distribution over clean derivations, and drifting
it re-weights which chunks get asked for while every chunk stays perfectly legal:

- the **root prior** over r\* (v-way) — literally the request;
- the **mixture weights** over which of the m synonymous rules each feature expands with, on the
  rule layers **above** the mined vocabulary (`demand_levels = "0/1"`, root-ward indexing, i.e.
  `rules[0]` root→L3 and `rules[1]` L3→L2).

**Why the level restriction is load-bearing: the level-2 cell is a null by construction.**
Re-weighting the mixture at level 2 would change which *synonym* of a level-2 feature the world
produces — and writing **either** synonym still derives that feature, so the repair succeeds
either way and the whole manipulation is a measured zero. Demand has to move which **feature** is
asked for at the node, and that is decided further up. This is the demand-side analogue of
`transpose`'s level-2 inadmissibility: in both rounds the lever has to be aimed, and in both the
aiming was forced by the substrate rather than chosen.

**Everything else is untouched**, and that is the round's admissibility gate: `corrupt_hier`, the
era ladder's (level, node) cells, the true tables and every grading path. So the full true table
is exactly correct at every epoch, no entry ever becomes illegal, and the damage operator's
difficulty is unchanged.

**Why `rhm_drift`'s OU machinery is the right tool here, having been the wrong one there.**
`transpose` explicitly rejected `full_loop`'s `advance_drift` / `calibrate_sigma_event` because
it drifts the synonym mixture weights: *"it cannot invalidate a committed macro — every entry
stays grammatical, only its frequency moves."* That is exactly the property this round requires.
Because the support never changes, the KL is **finite and nats are the honest currency again**;
in `transpose` the support changed and the only honest currency was a survival fraction. What is
adapted: the OU form (mean-reverting, so the shift-generating process is itself stationary),
`weights_from_theta`, `sample_derivations_weighted`, the calibrate-to-a-target-*event*
discipline, and the prewarm warning. What is **not** adapted is the magnitude functional —
`rhm_drift.drift_kl` measures KL per *sequence* over the whole tree, whereas the quantity this
round is about is the demand over the **level-l vocabulary at the nodes the era damages**.
`demand_kl` measures that directly, so the knob is calibrated in the currency the experiment
reads.

**Drift is unannounced.** No arm reads a demand signal; re-mining, count decay and recert all run
on their own clocks.

## Code files

| File | Purpose |
|---|---|
| `demand.py` | The demand primitives, kept out of the app so they are auditable and gate-testable with no substrate. `new_demand` / `demand_step` are the OU over (root logits, rule logits at `demand_levels`); `root_prior` / `rule_weights` read it out. `sample_pool_demand` and `context_instances_demand` are `_sample_pool` and `ratchet.context_instances` with the clean derivation drawn from the current demand and **`corrupt_hier` untouched**. `demand_hist` is the demand over a level's vocabulary (pooled over every node of that level, matching `ratchet.macro_moves`' position-independence); `demand_kl` / `demand_entropy` / `coverage` are the currencies. `demand_table` builds the demand-matched oracle table through `MC.Miner` so the ratchet nesting is applied by the parent's own code. `typical_demand` is the representative-start fix (below). `calibrate_sigma_demand` / `event_demand_kl` / `simulate_demand` are the offline calibration |
| `setlist.py` | The Modal app. Forks **`transpose.run_arm`**; `epoch_refs`, `DecayMiner`, `entry_set`/`churn` and the arm dict are imported from `transpose`, and `build_shared`/`_shared_plus`, `beam_moves`, `context_instances`, `finetune_generator`, `value_steps`, `measure_refs`, `plant_probe`, `macros`, `grader`, `node_at`, `step_detector`, `vocab_rate`, `write_results` from further up the arc. `build_world` precomputes every **demand epoch** once at setup — `rules`, `rules_t` and `truth` are the *same objects* at every epoch, which is what makes "nothing becomes false" true by construction rather than by care. `ctx` dispatches the instance draw: with no demand state it is **the parent function itself**, which is what makes the drift-off replay bit-identical rather than merely equivalent. `_world_gate` prints the admissibility gate before any arm runs. New per-cycle log keys beyond `transpose`'s: `cand_cover`, `live_cover`, `dem` (the tracking oracle's audition), `cover_now` / `cover_init`, and `churn[...]["cover"]`. Entrypoints: `setlist` (main), `cal_gate` (admissibility + descent at a ladder of demand specs), `cal_demand{,_remote}` (the offline σ×κ sweep), `selfcheck{,_remote}` |
| `analyze_setlist.py` | Reduction. `--fetch` pulls from the volume; `--fidelity` compares against `ear/figures/er_s0` over ten log fields. The default report leads with the **admissibility gate** (demand must move while the world does not), then the grade and the **concentration premium**, the post-commit unit error with its typed decomposition (`held − dem` = frozen vs perfectly demand-tracking content, `held − true` = frozen vs full coverage), the **typed pair** (precision-against-truth beside coverage-against-demand), recert with frozen-vs-live margins and per-swap coverage, churn, grader staleness, and the instrument suite. `paid_grader` prices only the cells an arm actually buys (validated in `transpose` against `ear`'s published 29.2% for `never_base`). `--figures` writes fig1–fig4 |
| `launch_detached.py` | `transpose`'s session-isolated launcher (`start_new_session=True`), retargeted. Opens its log with `"a"`, so a relaunch appends and the file stays an archive |

## Arms

`vocab` ∈ {base, true, dem_frozen, dem_live, earned}; everything else is `ear`'s.

| arm | vocabulary | commit policy | recert | miner |
|---|---|---|---|---|
| `never_base` | base level-1 moves forever | — | — | monotone |
| `given` | the DGP's **full** table — pure COVERAGE, demand-invariant | — | — | monotone |
| `demand_frozen` | perfect CONCENTRATION on **epoch-0** demand, never updated | — | — | monotone |
| `demand_live` | perfect CONCENTRATION on **current** demand, re-read every epoch | — | — | monotone |
| `practice_gated` | earned | unit-LP certificate | off | monotone |
| `practice_prov` | earned | provisional at the era boundary | **on** | monotone |
| `prov_norecert` | earned | provisional at the era boundary | **off** | monotone |
| `practice_climb` | earned | certify-else-provisional | **on** | monotone |
| `climb_decay` | earned | certify-else-provisional | **on** | **decaying** (γ = 0.95) |

**`given_live` is dropped and replaced by the oracle bracket.** In `transpose` the true table
could go stale, so a frozen and a tracking copy were different arms. Here nothing becomes false,
so the truth cannot be re-read — the fork that *does* exist is over **concentration**.
`demand_frozen` and `demand_live` are built by the same construction (`demand_table`: the
smallest entry set covering `demand_cover = 0.8` of the demand mass, nested through `MC.Miner` so
`T[l]` is defined over `T[l-1]` entries), from the epoch-0 demand and from the current demand
respectively. Same construction, same size, **only tracking differs** — which makes the pair a
direct oracle bracket on what demand-tracking is worth.

`transpose`'s and `recital`'s inherited arms are carried unchanged so the fidelity replay can run
them. Recert reaches every committed level (`recert_all`), ascending, as in `transpose`.

## Gates

| Gate | What it asserts | Where |
|---|---|---|
| C-M / C-R / G-D / G-N / G-S / G-R / G-P / T-1…T-5 | the arc's, `ear`'s, `recital`'s and `transpose`'s | inherited via `transpose.selfcheck` |
| **D-1** | at uniform demand the weighted sampler is **distributionally** the parent's. It is *not* stream-identical (the parent draws `rng.integers`, the weighted sampler `rng.random`), which is exactly why the runner keeps the parent function itself on the drift-off path | `setlist.gate_demand` |
| **D-2** | **nothing becomes false**: the true tables are the same object at every epoch, drawn instances stay 100% on-grammar, and any table's precision against the truth is invariant to any amount of demand drift — the property `transpose` could not have, and what makes the two rounds a typed pair | `gate_demand` |
| **D-3** | demand actually moves: per-event KL > 0 at both mined levels and a concentration frozen at epoch 0 loses coverage | `gate_demand` |
| **D-4** | difficulty is held: `d0` and the on-grammar rate are stationary across epochs, so demand-drift is not difficulty-drift in disguise | `gate_demand` |
| **D-5** | the OU is **prewarmed to stationarity** — a cold start begins at maximum-entropy, *tasteless* demand and then acquires taste, so any "decay since the start" reading would be measuring the warm-up | `gate_demand` |
| **D-6** | the **demand-invariant denominator**: the full true table's coverage of demand does not move with the drift, which is what licenses `given` as the concentration-premium reference | `gate_demand` |

Measured at (v=8, s=2, depth=4, m=2, rule_seed=0), all passing:

- **D-1**: `d0` 1.544 (parent) vs 1.559 (weighted); on-grammar 1.000 both.
- **D-2**: a frozen table's precision against the truth is **exactly 0.800 at all 8 epochs**;
  true tables bit-identical; on-grammar 1.000 throughout.
- **D-3**: KL-from-init up to **0.974 nats**; frozen coverage 0.869 → 0.639.
- **D-4**: `d0` spread 0.234, slope **−0.011/epoch** (no trend).
- **D-5**: cold-start entropy begins at **2.511** (the maximum) and trends down (slope −0.052,
  start − later-mean **+0.360**); prewarmed begins at 1.846 with slope +0.010 and start −
  later-mean −0.327.
- **D-6**: true-table coverage of demand **0.827–0.863, spread 0.036**.

Note the demand histogram is over **parsed** spans (the space the miner reads), so the full true
table's coverage is a constant just below 1 (≈0.85 at L2, ≈0.67 at L3) rather than exactly 1:
`generate_rules_distinct` guarantees distinctness only *within* a feature, so a few leaf tuples
collide across features and the exact inverse map's "last writer wins". Being constant is what
D-6 needs; being 1 is not required.

## Fidelity

With `--demand-every 0` the demand state is `None`, `ctx` calls `ratchet.context_instances`
itself, the world has one epoch, and every added instrument is gated behind `dem_on`. `sf_s0`
therefore runs `ear`'s own eight arms in `ear`'s own order and **reproduces `ear/er_s0`
bit-for-bit — all 8 arms, all 90 cycles, ten log fields, max|Δ| = 0.00e+00.**

## Pre-run offline calibration (`cal_demand`, zero GPU)

### The lesson: σ and κ are different knobs, and the objective is non-monotone

**Binary-searching σ against a target per-event KL — the `calibrate_sigma_event` recipe that
worked for `transpose`'s magnitude — FAILS here and runs to the bound.** As σ grows the softmax
saturates toward one-hot and a fixed logit perturbation stops changing the argmax, so per-event
KL rises and then *falls*. The two knobs have to be **swept, not solved**:

- **σ sets the CONCENTRATION** of demand (the OU logit scale, hence the entropy of what the world
  asks) — what a chunk has to exploit, and hence what it has to lose.
- **κ sets the MIXING RATE** (mean reversion), hence how fast a frozen concentration goes out of
  fashion. The *relative* size of one event is `sqrt(1 − (1−κ)²)` of the stationary spread and
  **does not depend on σ at all**.

Swept 5 σ × 4 κ at period 4 over 90 cycles (`cover` 0.8, level 2; `half-life` = cycles for a
frozen concentration to lose half its coverage):

| σ | κ | H(L2) | KL/event L2 | KL/event L3 | cov@0 | cov@45 | cov@90 | half-life | true_cov |
|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.15 | 2.508 | 0.0061 | 0.0239 | 0.853 | 0.875 | 0.884 | > run | 0.852 |
| 1.00 | 0.05 | 2.064 | 0.0170 | 0.0544 | 0.803 | 0.713 | 0.631 | > run | 0.854 |
| **1.00** | **0.15** | **2.326** | **0.0494** | **0.1468** | **0.807** | **0.797** | **0.643** | **> run** | **0.858** |
| 1.50 | 0.15 | 2.135 | 0.0768 | 0.2034 | 0.847 | 0.707 | 0.466 | 70 | 0.855 |
| 2.50 | 0.15 | 1.856 | 0.1136 | 0.2741 | 0.866 | 0.676 | 0.390 | 46 | 0.850 |
| 2.50 | 0.30 | 2.015 | 0.3225 | 0.7276 | 0.820 | 0.611 | 0.668 | 46 | 0.853 |
| 2.50 | 0.50 | 2.126 | 0.3729 | 0.9052 | 0.851 | 0.536 | 0.737 | 46 | 0.855 |

`true_cover` is flat at 0.85 across the whole sweep — D-6 holding independently of the knobs.
Note the coverage decay is **non-monotone at large κ**: the OU is mean-reverting, so a
concentration can come *back* into fashion. That is honest behaviour, not noise, and it has a
consequence for reading results (below).

### The prewarm fix: `typical_demand`

Prewarming to stationarity fixes the cold-start transient (D-5) but replaces it with a subtler
version of the same artefact: the start is **one** draw from the stationary law, and one draw can
be unrepresentative. `cal0` caught exactly that. At seed 0 the prewarmed demand had entropy
**1.856 against a process median of 2.250** — atypically concentrated — and the OU then relaxed
toward typical demand. Because the *same* prewarm draw seeds every cell of a sweep, every cell
inherited the same relaxation, and quantities that follow demand concentration (`stale`, the
exact-DP `floor`, the true macro's ceiling) trended monotonically **in the epoch index in cells
with completely different σ** — a signature no property of the drift itself can produce.

`typical_demand` draws `demand_n_cand = 48` stationary candidates, measures each one's demand
entropy, and starts from the one closest to the median; the candidate spread is logged so the
lottery that was avoided is on the record rather than assumed away.

| σ | chosen entropy | median | candidate range | entropy-path slope after the fix |
|---|---|---|---|---|
| 1.0 | 2.415 | 2.420 | 2.030 – 2.598 | −0.006/epoch |
| 2.5 | 2.247 | 2.250 | 1.552 – 2.620 | −0.026/epoch |

The per-epoch exact-DP floor was also widened from 128 to `n_floor = 256` instances to cut its
noise.

## The GPU admissibility gate (`cal0`)

`transpose`'s lesson applied in the opposite direction. There the risk was that the world got
harder and the arms stopped separating; here the world should **not** get harder, and the risk is
the mirror image — that demand moves and nothing downstream notices. `cal_gate` runs one setup,
one era, the same arms at a ladder of `period/sigma/kappa` specs with an **in-run no-drift
control**, so no cross-run inference is needed. Readout: arm separation
`e(never_base) − min e(vocabulary arms)` on the last 5 cycles, against `recital`'s measured
±0.034 stream-position noise.

| cell | never_base | practice_climb | climb_decay | given | **separation** | recerts |
|---|---|---|---|---|---|---|
| `0/0.0/0.0` (control) | 0.365 | 0.273 | 0.244 | 0.177 | **0.121** | 0/0 |
| **`4/1.0/0.15`** | 0.268 | **0.145** | 0.182 | 0.164 | **0.123** | 0/6, 0/5 |
| `4/2.5/0.15` | 0.287 | 0.266 | 0.294 | 0.155 | **0.021** | 0/1 |

**σ = 2.5 is inadmissible**, for the same reason level-2 grammar drift was in `transpose`: the
arms stop separating (0.021, below the noise floor; `climb_decay` finishes *worse* than
`never_base`). **σ = 1.0 preserves separation at the no-drift level** (0.123 vs 0.121). This is
what set the main run's σ.

`cal0` is also where the prewarm artefact was found: `floor` fell 0.180 → 0.047 (σ=1.0) and
0.164 → 0.039 (σ=2.5) along near-identical trajectories, which is what identified the cause as
epoch-indexed rather than demand-driven.

## Dead and confounded instruments — do not trust these

Two instruments in this node's logs do not measure what their names suggest.

1. **`log["bank_fresh"]` is dead.** It asks whether a span read off the agent's own derivation
   bank is in the current demand's **support** — and the support is nearly everything, so it
   reads **1.000 for every arm at every cycle**. It carries no information. The intended quantity
   was the demand *mass* the bank's spans carry, which is not reconstructible from the logs
   because the bank itself is not serialised. A future round wanting bank staleness must log the
   mass (or the bank) at write time.
2. **The grader-staleness signal (`mfg_pol − real_pol` against the manufactured set's age) is
   confounded with era position.** The signal is there — the gap runs from ≈−0.01 at age 0–7 to
   +0.10…+0.14 at age 24–29, correlation +0.28…+0.41 — but the manufactured set is rebuilt at
   every era boundary and never within an era, so **age is nearly collinear with `c_in_era`**,
   and `never_base`, which holds no vocabulary at all, shows the same pattern (+0.125, corr
   0.38). Ageing and within-era learning cannot be separated in this design. Testing grader
   staleness properly needs the manufactured set rebuilt on a clock **independent** of the era
   boundary.

A third caveat is about reading results rather than about an instrument: because the OU is
**mean-reverting**, a concentration frozen at epoch 0 can return to favour. In `sl_s0` its
coverage fell to 0.674 at its trough and was back to **0.903 at cycle 90** — so the terminal
all-eras exam is taken at a moment when the epoch-0 demand happens to be back in fashion, and it
**understates** erosion. Per-epoch trajectories, not endpoints, are the honest readout for
anything demand-relative.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | attached `--quick` smoke, 9 arms, demand every 3 cycles at σ=2.5/κ=0.15, `mine_decay` 0.9 |
| `cal0` | the admissibility gate + descent check: 3 demand specs × 4 arms × 40 cycles, one era, in-run no-drift control. 1043 s. **Set the round's σ** (2.5 inadmissible; see above) and surfaced the prewarm artefact |
| `sf_s0` | the fidelity replay: `ear`'s eight arms, `ear`'s order, `--demand-every 0`. 1394 s. **PASSES: bit-identical to `ear/er_s0` on all 8 arms across all 90 cycles**, ten log fields, max\|Δ\| = 0.00e+00 |
| `sl_s0` | the main run: 9 arms, seed 0, 3 eras × 30 cycles, σ=1.0 / κ=0.15 / one demand event every 4 cycles from c6 (23 epochs), representative start, `mine_decay` 0.95, `recert_all`. 1722 s |

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_setlist/<tag>/<arm>/results.json`, with
`setup.json` beside them (carrying the whole `world` block: per-epoch demand KL, entropy, true
and oracle coverage, and the moving references). `cal0` nests one level deeper,
`/data/rhm_practice_setlist/cal0/<spec>/<arm>/`. Fetched copies live in `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/setlist/setlist.py::selfcheck_remote
modal run rhm/practice/setlist/setlist.py::cal_demand_remote

# the admissibility gate + descent check (sets sigma)
python3 rhm/practice/setlist/launch_detached.py --fn cal_gate --tag cal0 \
    --specs "0/0.0/0.0,4/2.5/0.15,4/1.0/0.15" --cycles 40 --eras "1:6" \
    --arms "never_base,given,practice_climb,climb_decay"

# fidelity replay of `ear` (demand off == `transpose` with drift off == `recital` == `ear`)
python3 rhm/practice/setlist/launch_detached.py --fn setlist --tag sf_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_late,practice_self,taught,practice_prov,practice_climb" \
    --demand-every 0 --t-budget 0 --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 \
    --n-aud 192 --mfg-source child --mfg-render canon --mfg-min-bank 32 \
    --recert-every 5 --recert-margin 0.05 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

# the main run
python3 rhm/practice/setlist/launch_detached.py --fn setlist --tag sl_s0 --seed 0 \
    --arms "never_base,given,demand_frozen,demand_live,practice_gated,practice_prov,prov_norecert,practice_climb,climb_decay" \
    --demand-every 4 --demand-sigma 1.0 --demand-kappa 0.15 --demand-levels "0/1" \
    --demand-start 6 --demand-seed 0 --demand-n 8192 --demand-cover 0.8 --demand-n-cand 48 \
    --n-floor 256 --mine-decay 0.95 --mine-floor 0.5 --recert-all \
    --t-budget 0 --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 \
    --max-macro-level 3 --n-pr 64 --n-rt 384 --n-score 512 --n-aud 192 --n-grad 4 \
    --value-lr-online 3e-5 --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 --mfg-source child --mfg-render canon \
    --mfg-min-bank 32 --recert-every 5 --recert-margin 0.05 --prov-offset 0 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

# reduction
python3 rhm/practice/setlist/analyze_setlist.py --tag sf_s0 --fetch --fidelity
python3 rhm/practice/setlist/analyze_setlist.py --tag sl_s0 --fetch --figures
```

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_competence_gate.png` | Competence per cycle with commits marked; beside it **the gate** — demand moving (the epoch-0 concentration's coverage) against the world not moving (`stale`, `floor`, the true table's coverage) |
| `figures/<tag>/fig2_postcommit.png` | `held − dem` per level: frozen committed content against perfectly demand-tracking content on the same set at the same cycle |
| `figures/<tag>/fig3_typed_pair.png` | Coverage of the current demand (what drifts) beside precision against the truth (what cannot move here) |
| `figures/<tag>/fig4_recert.png` | `e(frozen) − e(live)` per recert event against the swap margin |

## Seeds

`--seed` sets damage draws, metering sets, mining and probe RNG; `--rule-seed` / `--train-seed`
set the DGP draw and the substrate, defaulted to `ratchet`'s (`0`/`0`/`1`) so every prior run
reproduces. **`--demand-seed` is a separate stream**, so the audience's motion is independent of
every arm's randomness and of the DGP draw. A replicate varies the triple together as
`seed = k, rule_seed = k, train_seed = k + 1` (`recital`'s convention). Single seed by default;
absolute error levels are not comparable across rule draws, so orderings, signs and recovery
fractions are the reported quantities.

## Inherited, not copied

`../transpose/` (`run_arm`'s shape, `epoch_refs`, `DecayMiner`, `entry_set`/`churn`, `ARMS`, the
launcher), `../recital/` (`step_detector`, `vocab_rate`, `parse_arms_nm`, the gates),
`../ear/` (config, shared setup, grader, `node_at`, `write_results`), `../ratchet/` (substrate,
`macros.py`, pricing, beam, reader, online plant and selector, references),
`../crystallize/units.py` (the level-indexed action space, hierarchical damage, exact-DP oracle),
and `rhm/rhm_drift.py` (`sample_derivations_weighted`, `weights_from_theta`, and the OU form).
Nothing in those files is modified.
