# recital — File Index

Complete file-by-file reference for this node. **No `README.md` by deliberate policy** — the
writeup for this round is being held for a multi-round writeup later, exactly as for
[`../ear/`](../ear/FILES.md). This file plus the code comments are the record.

## What the round removes

`ratchet` and `ear` handed the agent an **era clock**: a fixed 30-cycle block per depth era,
externally imposed. This round takes it away. Each arm holds **one total priced-time budget**
for the whole run and decides for itself when to move from era k's damage distribution to era
k+1's, on the same nested ladder (`L1n6 ⊂ L2n3 ⊂ L3n1`). Everything else is `ear` as built,
imported rather than copied, so `er_s0` stays reproducible and this fork's `ear`-path arms stay
comparable to it cycle for cycle.

**The commit policy is fixed across every practice arm** (`ear`'s `delta_prov`: certify where
the unit-LP certificate can, commit provisionally at the moment of advancement otherwise).
**The advancement policy is the only variable** between the self-paced arms. Both pacing
detectors are the *same* detector (`step_detector`) on two different series, at knobs held
bit-identical to `ratchet`/`ear` (c 0.06, c_v 0.15, W 5, hold 2, `lp_min_drop` 0.10,
`sil_min_cycle` 6). **Advancement is monotone** — no dropping back down the ladder.

### Agent-legal readouts (may be acted on)

- the arm's own metered task error `e` on the **current** era's held-out set — priced feedback
  it is already buying;
- the unit-LP audition series on the grader cell it reads (`mfg`/policy: contexts the agent
  manufactures from its own solved derivations, graded by its own performance policy) — priced;
- its own **mining stream** for the level it is earning — `n_obs` (level-1 spans read off its
  own solved configurations), `n_distinct` (distinct tuples ever seen) and `n_at_support`.
  These are the agent's own counters; recall/precision against the DGP's table stay
  instruments;
- its own **total priced-time budget** and the priced time it has spent — a clock it owns;
- the number of levels on the ladder (`max_macro_level`) — task specification, already known to
  `ear` through `node_at`.

### Instruments (measured every cycle, logged, never acted on)

The rest of the 3×2 grader grid (including `real`, which needs the oracle), `A_true`, `rand_k`,
the `held`/`live` counterfactual, `grader.context_stats`, the width ladder, the **all-eras
probe** (it would leak the future ladder — it is the unpriced matched-budget final exam), and
the plant guard.

## Code files

| File | Purpose |
|---|---|
| `recital.py` | The Modal app. Forks **only `ear.run_arm`** — the era `for`-loop becomes one `while`-loop whose stopping rule is the priced-time budget and whose era index advances when the arm's own policy says so; every other piece of the substrate (`build_shared` via `ear._shared_plus`, `beam_moves`, `context_instances`, `finetune_generator`, `value_steps`, `measure_refs`, `plant_probe`, `macros`, `grader`, `node_at`, `write_results`) is imported. `step_detector` factors `ratchet`'s unit-LP detector out so the certificate and the task-progress pacer are literally the same code on two series; it consumes no randomness, which is what keeps the fixed-clock arms bit-identical to `ear`. `vocab_rate` derives the novel-tuple admission rate from the agent's own mining counters — also pure arithmetic, so the third pacer costs no randomness either. `ARMS` adds one field, `pace` ∈ {`cycles`, `frac`, `cert`, `task`, `comp`, `vocab`}, and carries `ear`'s eight arms unchanged so the fidelity replay can run them. New per-cycle log keys: `c_in_era`, `tcert` and `vcert` (the task and vocabulary pacers' state, logged in every arm whether or not it is read); new event kind `advance`. Probes additionally fire at matched-budget checkpoints (`probe_t_n` per run) and on the terminal cycle. Entrypoints: `recital` (main), `selfcheck{,_remote}` (`ear`'s C-M / C-R / G-D / G-N / G-S / G-R plus this round's **G-P**) |
| `analyze_recital.py` | Reduction. `--fetch` pulls from the volume; `--fidelity` compares against `ear/figures/er_s0` cycle for cycle. The default report gives the **matched-budget grade** (the terminal all-eras probe: every arm's error on *every* era's held-out set at the same total priced time, plus the same exam at 25/50/75/100% of budget), **boundary placements** (cycle, priced time, %-of-budget, reason, and what the arm was holding when it advanced), **regret** against the best fixed schedule in the sweep, per-era competence actually visited, earned-vs-given on the terminal ladder, cost-to-depth, commits and recerts, `predict` (the offline replay of *both* pacers on *every* arm's own series, against what the arm actually did), grader-cost accounting, the grader grid, the oracle check, the seam, mined-vs-random and the plant guard. `--figures` writes fig1–fig4 |
| `launch_detached.py` | Session-isolated detached launcher (`start_new_session=True`), so a harness signal cannot cancel the remote input mid-run. `--fn` selects the entrypoint; logs to `results/launch_<tag>.log` |

## Gates

| Gate | What it asserts | Where |
|---|---|---|
| C-M / C-R / G-D | `ratchet`'s: the macro with the DGP's table is bit-identical to the true level move; the `T3 ⊆ T2 × T2` ratchet bites; the nested damage ladder is on-grammar | inherited via `ear.selfcheck` |
| G-N / G-S / G-R | `ear`'s: the ladder is nested; a self-manufactured audition context is on-grammar, confined and correctly levelled; a live recert undoes the ratchet's cap | inherited via `ear.selfcheck` |
| **G-P** | **this round's**: the pacers are one detector (P-4); it never fires mid-descent (P-1); it does fire on descend-then-flat, and not before `sil_min_cycle` (P-2); it does **not** fire on a series that is flat from the start (P-3) or rising (P-3b) — so the descent precondition is what makes LP starvation consequential rather than instantly-advancing; and `pace_comp`'s deadline carve partitions the total budget (P-5) | `recital.gate_pacers` |
| **G-P** (ext.) | **the vocabulary pacer**: `vocab_rate` is the admission rate it claims to be — reads only the agent's own counters, takes the era's virtual zero as the window origin, stays in [0, 1], starts at 1 when every span is new and reaches exactly 0 once nothing is admitted (P-6); and on a real-shaped mining stream it fires once growth saturates, never while the table is still filling, and not at all on a stream that never grew (P-7a/b/c) | `recital.gate_pacers` |

## Fidelity

With `--t-budget 0` this file **is** `ear`: the loop stops after the last era's `era_cycles`
cycles and every RNG stream is untouched. `rf_s0` therefore runs `ear`'s own eight arms, in
`ear`'s own order, and must reproduce `er_s0` cycle for cycle.

**Why all eight.** `ratchet`/`ear` do not reset torch's global RNG between arms, so arm *k*'s
stream depends on how many cycles arms 1..*k*−1 ran. Reproducing `practice_climb` (arm 8)
therefore requires running all eight arms for their full 90 cycles — which is what `rf_s0`
does, and why the main run's `fixed_climb` is a *comparability* anchor rather than a
bit-identical one.

## Inherited, not copied

`../ear/` (`ear.py`'s `run_arm` is the direct parent of `recital.py`'s; `grader.py` supplies
`manufacture_damage`, `policy_e`, `context_stats` and the G-N/G-S/G-R gates),
`../ratchet/` (`macros.py`'s earned vocabulary and max-sum macro operator; `ratchet.py`'s depth
ladder, matched pricing, beam, reader, online plant and selector, references), and
`../crystallize/units.py` (the level-indexed action space, hierarchical damage, exact-DP
oracle). Nothing in those folders is modified.

## Arms

| arm | vocabulary | advancement policy |
|---|---|---|
| `never_base` | base level-1 moves forever | the external clock (30 cycles/era), then out the budget in the last era |
| `given` | the DGP's own tables from cycle 1 | the external clock |
| `fixed_climb` | earned, `ear`'s `practice_climb` grader + commit | the external clock — the arm to beat |
| `pace_cert` | earned, same | **advance when the unit-LP certificate fires** (may never) |
| `pace_task` | earned, same | **advance when the same detector goes silent on the metered task error** |
| `pace_comp` | earned, same | **advance on the certificate, or on a self-set deadline at era *j*'s share of the total budget — whichever is first** |
| `pace_vocab` | earned, same | **advance when the same detector goes silent on the agent's own novel-tuple ADMISSION RATE** — i.e. when its vocabulary stops growing |
| `sched_a..g` | earned, same | fixed boundaries at declared **fractions of the total budget** (`sf1`, `sf2`) — the reference sweep regret is measured against |

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_pacing.png` | Competence against priced time with each arm's *own* era boundaries and commits marked; beside it, the curriculum each policy actually chose (era index against cycle) |
| `figures/<tag>/fig2_signals.png` | The two signals a self-paced agent can read — the unit-LP audition (mfg/policy) and the metered task error — per arm, with advancements marked |
| `figures/<tag>/fig3_regret.png` | The matched-budget final exam over the whole ladder, and regret against the best fixed schedule in the sweep |
| `figures/<tag>/fig4_vocab_cost.png` | Earned-table recall against the DGP's vocabulary, and the cumulative priced cost of the whole grader grid |

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | attached `--quick` smoke, 7 arms, all five advancement policies exercised |
| `rf_s0` | the fidelity replay: `ear`'s eight arms, `ear`'s order, `--t-budget 0`; must equal `er_s0` |
| `rc_s0` | the main run: 10 arms at a matched total priced budget of 7.0e6, seed triple 0 |
| `rc_s1`, `rc_s2` | independent replicates of `rc_s0` at seed triples 1 and 2 |
| `smoke1` | attached `--quick` smoke of `pace_vocab` and the extended sweep cells |
| `rv_s0` | the vocabulary pacer and the extended schedule sweep, seed triple 0 (same world as `rc_s0`): `given`, `sched_d2` (.50/.75, the champion re-run in-run), `pace_vocab`, `sched_e` (.60/.80), `sched_f` (.70/.85), `sched_g` (.80/.92) |

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_recital/<tag>/<arm>/results.json`, with
`setup.json` beside them.

## Seeds

`--seed` alone sets only `cfg["seed"]` (damage draws, metering sets, mining and probe RNG); the
DGP draw and the substrate are `rule_seed` and `train_seed`, which `ratchet`/`ear` hardcode.
This node exposes all three, defaulted to `ratchet`'s (`0`/`0`/`1`) so every prior run
reproduces. **A replicate varies all three together** as the triple
`seed = k, rule_seed = k, train_seed = k + 1` — a genuinely independent world (new grammar, new
substrate, new damage). Seed 0 is the k = 0 member. Substrate quality varies materially across
rule draws (controller accuracy 0.72–0.89 over k = 0, 1, 2), so **absolute error levels are not
comparable across seeds** — only orderings, signs and recovery fractions are.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/recital/recital.py::selfcheck_remote
modal run rhm/practice/recital/recital.py::recital --quick --tag smoke0 \
    --arms "never_base,given,fixed_climb,pace_cert,pace_task,pace_comp,sched_frac:sf1=0.2:sf2=0.5:nm=sched_a"

# fidelity replay of `ear` (t_budget 0 == the era clock restored)
python3 rhm/practice/recital/launch_detached.py --fn recital --tag rf_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_late,practice_self,taught,practice_prov,practice_climb" \
    --t-budget 0 --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 \
    --n-aud 192 --mfg-source child --recert-every 5 --recert-margin 0.05 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

# the main run
python3 rhm/practice/recital/launch_detached.py --fn recital --tag rc_s0 --seed 0 \
    --arms "never_base,given,fixed_climb,pace_cert,pace_task,pace_comp,sched_frac:sf1=0.15:sf2=0.40:nm=sched_a,sched_frac:sf1=0.25:sf2=0.55:nm=sched_b,sched_frac:sf1=0.33:sf2=0.67:nm=sched_c,sched_frac:sf1=0.50:sf2=0.75:nm=sched_d" \
    --t-budget 7000000 --max-cycles 300 --probe-t-n 12 \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --pr-width 16 --g-budget 58 \
    --max-macro-level 3 --n-pr 64 --n-rt 384 --n-score 512 --n-aud 192 --n-grad 4 \
    --value-lr-online 3e-5 --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 --mfg-source child --mfg-render canon \
    --mfg-min-bank 32 --recert-every 5 --recert-margin 0.05 --prov-offset 0 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

# reduction
python3 rhm/practice/recital/analyze_recital.py --tag rf_s0 --fetch --fidelity
python3 rhm/practice/recital/analyze_recital.py --tag rc_s0 --fetch --figures
```

## Pre-run offline calibration (zero GPU, from `er_s0`'s own logs)

Replaying both pacers over `er_s0`'s recorded series — the only era in which a replay is valid,
since trajectories diverge after the first advancement — predicted, before the run:

| pacer | era 1 (earning L2) | era 2 (earning L3) |
|---|---|---|
| `pace_cert` (mfg/policy unit-LP) | fires **c21–c26** across the climb-family arms (`er_s0`'s own `practice_climb` fired at c26, `practice_self` at c14) | **does not fire** in any climb-family arm: the level-3 audition's span is 0.06–0.17 against `lp_min_drop` 0.10 |
| `pace_task` (same detector on the metered task error) | descends by 0.16–0.20 but does not reach silence inside 30 cycles once the certificate-driven commit lands late in the era — so it should fire **shortly after c30**, i.e. later than `pace_cert` | the task error **drifts upward** within era 2 in every arm (span 0.00–0.02), so the descent precondition is not met |

Recorded as a prediction, not as an interpretation. What the arms actually did is in
`analyze_recital.predict`, which runs the same replay on the run's own series.
