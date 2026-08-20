# fingering — file index

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Siblings**: [`../etude/`](../etude/README.md) (the parent; its finding 7 is this node's mandate) ·
[`../../../rhm/practice/crystallize/`](../../../rhm/practice/crystallize/README.md) (the round
being ported: state-conditioned commitment beats state-independent 1.8–3.0×)

No findings README yet — round 0 is measurement, and results get discussed before any writeup.

## Code files

| file | purpose |
|---|---|
| `world.py` | The substrate both rounds share: the piece (approach → drilled passage), the plant + hard region, the on-policy diet, the metered `Body` traversal, CEM-MPC, the compile ops (`select_fixed` / `select_library` / `fit_bc` / `mean_unit`), the audition, and the priced-time `Ledger`. |
| `gates.py` | **Round 0** — the admissibility gates G1–G4 plus CAL-P (planner sizing) and CAL-D (descent clock). One Modal container per *world* (`--worlds "<label>:<curl_b>:<push_a>"`), so the difficulty knob is swept rather than solved. |
| `fingering.py` | **Round 1** — the arm comparison: `never` / `fixed` / `library` / `regress` / `mean`, one Modal container each. Arms are bit-identical up to the commit cycle; commitment is provisional at a fixed cycle with the δ-silence certificate logged as an instrument, plus one capped recert. |
| `analyze_gates.py` | Fetches the round-0 results off the volume and prints the gate report (one table per gate across worlds, plus the calibration record). |
| `analyze_fingering.py` | Reduces a round-1 run: headline (error × priced time, window-meaned), commit record, seam-law optimism gap, post-commit drift, candidate-pool geometry, itemised priced time, and the payback crossing against `never` at matched budget. |
| `launch_detached.py` | Session-isolated detached launcher (`--fn gates|fingering`), the `plasticity_gain` gotcha fix. |

## The world, and how each knob was set

| knob | value | set by |
|---|---|---|
| plant | 3-link planar arm, `(0.4, 0.4, 0.3)` m, masses `(1.0, 1.0, 0.6)`, `joint_damping=0.5`, `gear=8.0`, `frame_skip=10` | `arm_substrate` P5/P7's n=3 curl design point, byte-identical. **n=3 for redundancy, not capacity** — capacity binds only at n≥5 (P1), and this node needs a null space, not a capacity frontier. |
| `q_center` | `(0.4, 1.1, 0.8)` → tip `(0.197, 0.778)`, radius 0.803 | local FK search: keeps the jittered start distribution off the singular full-extension shell (r < 0.91 for the whole `q_jit=0.15` box) where `on_policy`'s `(0.4, 0.8, 0.6)` runs to 1.01 of a 1.10 m arm. |
| piece | `W1 = (0.3063, 0.4778)`, `W2 = (0.7463, 0.4778)`; approach 0.32 m, drilled 0.44 m, ~70° turn | local FK search over (direction, length) subject to: every point on both paths at radius ∈ [0.32, 0.92·L]; a real turn (0.10 ≤ cos ≤ 0.75); and the patch's gate on the approach path minimised. The drilled segment **extends** the arm (r 0.57 → 0.89), so `M(q)` changes along the passage. |
| patch | Gaussian-gated `curl_field` on the drilled midpoint, σ = 0.08 | gate weight at *both* waypoints and at the approach's closest approach = **0.023**, against the étude's 0.044 — so the approach is clean and the passage is local. |
| `curl_b` | **swept** {6, 14, 26} | `typed_gaps`: a difficulty knob must be swept, not solved. |
| `push_a` | 0 by default; −9 in the `b14obs` world | the G2 escalation. `arm_env._apply_push_field` is additive and off by default; whether it is *needed* is a measured result. |
| `h_drill` | 20 | at/beyond the plant's composition horizon (`arm_substrate` P4: 20–23 at n=3), re-measured per world by G4. |
| `sigma_perf` | 0.06 | motor noise live **in the run-through**, not only in practice. The étude's noise-free run-throughs are exactly why its post-commit drift was identically 0.0000. |
| `d_fb`, `dt_ctrl` | 0.10 s, 0.02 s | the étude's pricing model. A reactive drilled segment costs 20 delays; a committed one costs exactly 1 — and here that one is actually *spent*, on the observation that keys the library. |

## Additive changes to shared machinery (off by default, prior results byte-identical)

| file | change |
|---|---|
| `../../arm_env.py` | `push_fields` — a list of Gaussian-gated **radial** tip forces (`_apply_push_field`), i.e. a *soft* obstacle. Position-dependent (acts at rest, unlike the curl), smooth (so still open-loop compensable, unlike a contact). Nothing is written when the key is absent, and `build_xml` is untouched, so `on_policy/verify_backcompat.py`'s checks 1–2 are unaffected. |
| `../../embodied.py` | `Body.reset(..., q0=, qd0=)` — optional explicit start postures. Defaults `None`, and the `None` path consumes RNG exactly as before, so every prior call is byte-identical. Needed because a *piece* is a fixed object: every arm must start its run-throughs from the same held-out geometry, and the `null` start mode draws postures along the arm's self-motion manifold rather than from a box. |

## Modal volume layout

```
/data/practice_fingering/<tag>/<world>/results.json     # round 0 (one dir per world)
/data/practice_fingering/<tag>/<arm>/results.json       # round 1 (one dir per arm)
```

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# round 0 -- the gate sweep (one container per world)
python3 mjc/practice/fingering/launch_detached.py --fn gates --tag g0 --seed 0 \
    --worlds "b6:6:0,b14:14:0,b26:26:0,b14obs:14:-9"
python3 mjc/practice/fingering/analyze_gates.py --tag g0 --fetch

# round 1 -- the OP question (defaults are this run's config)
python3 mjc/practice/fingering/launch_detached.py --fn fingering --tag f0 --seed 0 --curl-b 14.0 \
    --arms "never,fixed,library_kmeans,library_proj,regress,mean"

# round 1b -- TIMING x MAINTENANCE, op held at library_kmeans
python3 mjc/practice/fingering/launch_detached.py --fn fingering --tag f1b --seed 0 --curl-b 14.0 \
    --n-cycles 91 --commit-early-cycle 20 --commit-mastery-cycle 51 --interleave-period 3 \
    --reselect-every 15 --reselect-n-cand 32 --reselect-n-score 48 \
    --arms "never,commit_early,commit_mastery,commit_early_il,commit_mastery_il,commit_mastery_il_reselect"
python3 mjc/practice/fingering/analyze_fingering.py --tag f1b --fetch

# round 1c -- LIVE CONTENT under committed routing (completes the op taxonomy)
python3 mjc/practice/fingering/launch_detached.py --fn fingering --tag f1c --seed 0 --curl-b 14.0 \
    --n-cycles 91 --commit-early-cycle 20 --commit-mastery-cycle 51 --interleave-period 3 \
    --reselect-every 15 --reselect-n-cand 32 --reselect-n-score 48 \
    --arms "never,plan_launch_early,plan_launch_early_il,plan_launch_mastery_il,commit_mastery_il_reselect"
```

## The op taxonomy (what each arm commits)

| op | routing | content | deliberations per traversal |
|---|---|---|---|
| `never` | live | live | `H_drill` |
| `plan_launch` | **committed** | **live** | 1 |
| `fixed` / `library_*` / `mean` | committed | frozen | 0 |
| `regress` | committed | frozen function of the hand-over | 0 |

`plan_launch` spends the boundary feedback event that every committed arm is already charged and
that the étude never actually spent. It runs no audition — there is nothing to select — so it is
cheap in priced time and expensive in deliberation, which is the whole point of the `d_plan` axis.

## Deliberation pricing: counted, never charged

`Ledger` records `plans` per kind alongside `steps` and `fb`; `d_plan` defaults to **0**, so
`t_priced` is byte-unchanged and every earlier number stays comparable. The reducer sweeps
`d_fb × d_plan` post-hoc and reports the smallest `d_plan` at which each arm overtakes `never`.
The load-bearing asymmetry: an audition rollout replays *stored* content, so it costs `fb` but
**zero** deliberation — frozen content's big up-front grading bill is free on the `d_plan` axis
while `never` pays `H_drill` deliberations on every traversal, forever.

`f0`'s defaults are deliberately left at what `f0` ran (`n_cycles=60`, `commit_cycle=20`,
`n_cand=96`, `n_score=96`), so the round-1 command above still reproduces it; every round-1b
change is an added knob or an added arm, never a changed default.

## Round-1b calibrations (both read off round 1, both recorded in the run config)

| knob | value | measured from |
|---|---|---|
| `commit_mastery_cycle` | **51** | `never`'s held-out **ballistic** error, trailing-3-probe mean, plateau 0.0384 ± 0.0026 over the last 5 probes; first sustained entry within 1 sd (≤0.0410) is c51. Reading the *reactive* curve instead would have said c12 — open-loop competence is the later of the two clocks, and round 1's mistake was reading the earlier one. |
| `interleave_period` | **3** | round-1 diet rent, as each arm's ballistic error ÷ `never`'s at matched cycle: `library_kmeans` 0.85–0.92× for the first 7 post-commit cycles, 2.04× by +10, saturating 2–3×; `fixed` broke at +4. Rot onset ≈ 7–10 consecutive pure-committed cycles, so a period of 3 (≤2 in a row) keeps a ~3× margin. |
| `reselect_every` / sizes | 15 / 32×48 | a maintenance audit, priced as one: 768 s each against the initial audition's 4608 s. |

## Gotchas (each cost a run)

- **Launcher logs must live outside the mounted package.** `shared.py` mounts `mjc` with
  `add_local_python_source("mjc")` and Modal hashes the whole directory, so a detached launcher
  appending to `mjc/practice/<node>/results/launch_<tag>.log` makes *every other* `modal run`
  started meanwhile die with `ExecutionError: <path> was modified during build process`. This
  node's launcher writes to `experiments/.launch_logs/` instead. The failure names the log file,
  not the run that is actually broken, so it reads as unrelated.
- **Size the CEM to the action-sequence dimension** (`arm_substrate` P3). The first smoke ran the
  drilled segment at the pusher-tuned `k_shoot=256, cem_iters=4` on a 60-dim sequence (H=20 x 3
  joints) and G3's usable range came out at 0.0032 against a 0.0062 noise floor — a failing gate
  that was planner noise, not an absent axis. CAL-P measures the stale/ceiling gap at three planner
  sizes precisely so that this is visible rather than inferred.
- **A "ceiling" FM collected on a broad pool is not the ceiling for a passage.** The same smoke's
  ceiling model had seen ~180 in-region transitions and was only 1.8x better than stale in-region.
  The ceiling here is corridor-matched (`n_corridor` traversals of the piece itself), per
  `bridge_assembly`'s spatial-matching law and the etude's `cal_s1`.
