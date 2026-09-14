# rubato — file index and design record

**Up**: [`../../README.md`](../../README.md) (practice) · [`../../../README.md`](../../../README.md) (mjc)
**Contract**: [`SPEC.md`](SPEC.md) — the orchestrator's prompt, verbatim, as in `accelerando/`,
`prestissimo/`, `solo/` and `acappella/`.
**Donor (forked, never edited)**: [`../accelerando/`](../accelerando/FILES.md) — `piece.py`,
`world.py` and `t1_pins.py` are copy-forks; `accelerando/` is untouched and `t1`, `p1`, `c1`,
`fsmoke_a`, `fsmoke_a2` and `fsmoke_b` stay byte-reproducible. Through it:
[`../prestissimo/`](../prestissimo/FILES.md) (the level ladder, the weld, `app=fixed`),
[`../../solo/`](../../solo/FILES.md) (the member library, the parity gate),
[`../../acappella/`](../../acappella/README.md) (the Δ ladder, the playability guard),
[`../../etude/`](../../etude/README.md) (the plant and the piece constants),
[`../../legato/`](../../legato/README.md) (F4: measured content carries no composition error).
**Read, not forked**: [`../../accompanist/`](../../accompanist/FILES.md) (the efference-copy operator
and the confound this node is built to avoid) · [`../../../jacobian_teacher/`](../../../jacobian_teacher/README.md)
(a learned `f(s,u)` on an mjc body; the one-step map is near-perfect at every capacity while the
composed map is not — the sizing fact Phase 3 has to survive).

**No findings README** — numbers get discussed before any writeup (repo policy, and the SPEC's
own instruction). This file is the record of what was decided and why.

## What this node is

`accelerando/` measured that a posture-conditioned head which emits **forces** matches or beats
the verbatim tape at every practiced tempo (L4 0.0056 vs tape 0.0069 at 48 ms notes against a
re-fit reflex at 0.0690) and transfers to **no** unpracticed tempo at any step the body can
express (0/6 slots at L3–L4 even at 1.14×). Forces are not tempo-invariant on any body. What is
tempo-invariant is the **path**. This node stores a committed unit as **position and velocity
against phase, with the clock outside the program**, and puts a **body model in the executor** —
on the learner's side of the meter — to convert it to commands. The reflex stays model-free and
re-fit per tempo; the plant model is never handed to the incumbent as a planner (the
`accompanist` confound). ROADMAP §2.2, §7.1.3.

## Status

| phase | what | tags | state |
|---|---|---|---|
| 0 | plumbing + every gate end to end, tiny config | `ksmoke0` | **run 2026-09-06**, 53 s, all gates pass |
| 1 | `kin_oracle` — the ceiling, exact inverse dynamics, no training anywhere | `ksmoke1`, `ksmoke2` | **run 2026-09-06**. 9/9 gates. NO GO on `k1`: the ceiling does not transfer |
| 1d | the diagnostic round — why it does not transfer (decisions 19–21) | `kdsmoke0`, `kdsmoke1` | **run 2026-09-06**. Located a defect in the resampler (decision 22) and the Phase 1 ceiling result |
| 1r | Phase 1's RUN OF RECORD | `krsmoke0`, `k1` | **run 2026-09-06**, 1601 s, 9/9 gates, P-T1 125 checks at 0.000e+00 |
| 2 | `kin_learned` — an inverse model fitted on slow-tempo data only, two families | `klsmoke0`, `klsmoke1` | **smoked 2026-09-06**, 1077 s. Ceiling 9 in-band cells, MLP 6, linear 0 |
| 2c | the refit control (decision 28): is the linear family's failure identifiability? | `klsmoke2` {32,16}, `klsmoke3` {8} | **run 2026-09-06**. Identifiability fired; `klsmoke1`'s reading withdrawn |
| 3 | `kin_corrected` — the read ladder, two forward models, P-E re-asked | `kcsmoke0`, `kc1` | **smoked 2026-09-06**, 1729 s |
| 2 | `kin_learned` — an inverse model fitted on slow-tempo data only, two families | — | not built |
| 3 | `kin_corrected` — a forward model, online correction, P-E re-asked | — | not built |

## Code files

| file | what |
|---|---|
| `piece.py` | `accelerando/piece.py` forked **verbatim**, unchanged. The fast body (mass 0.06, gear 10, damping 2; τ = 30 ms), the tempo ladder H ∈ {32,16,8,4,2}, `R*` and the band, the body-scaled gain grid, the ported practice noise. |
| `world.py` | `accelerando/world.py` forked verbatim, plus: `RolloutPool.run_path` / `World.rollout_path` (a grounding that records the full realised state path); `resample_path` / `kin_cmds` / `kin_library` / `measure_paths` (THE KINEMATIC UNIT); `body_consts` / `fwd_zoh` / `inv_zoh` / `inv_ct` (the executor's two schemes and the forward step); `World.obs_predict` + `World.fm_step` and the efference-copy branch of `traverse`'s `obs()` (inert by default, for the `reflex_ec` reference row only). |
| `t1_pins.py` | `accelerando/t1_pins.py` copied **verbatim**: `t1`'s R\*, tempo ladder, per-(tempo, Δ) gain table, 75 construction scores, 50 sweep numbers, P-S gains and the configuration they were produced under. Gate P-T1 asserts every one at max\|Δ\| = 0. |
| `kin.py` | the Phase 1 runner: gates P-F0/P-TAU/P-ZOH/P-G/P-T1/P-N/P-R/P-KN/P-K, the per-tempo library rebuild, the path measurement, the scaled and kinematic libraries, and the sweep at the fixed Δ. |
| `nets.py` | Phase 2's learned cerebellum: `triples_from_pool` (the practice traversals replayed for their velocities), `split_triples` (+ gate P-FH), `fit_linear` / `linear_report` (the minimal family and its recovered coefficients against the analytic ones), `fit_mlp`, `inv_fitted` (a fitted model wrapped as an inverse-dynamics scheme with `inv_zoh`'s signature), `heldout_mse`, and `coverage` — which reports how far outside the fit each tempo's query lands, so "extrapolation" is measured and not asserted. |
| `analyze_kin.py` | the reduction (pure python, local) + five figures (rendered remotely). |

## Decisions taken, with reasons

1. **`accelerando` is forked, not extended.** `piece.py` and `t1_pins.py` are byte copies;
   `world.py` differs only by additions that are inert unless switched on. Gate **P-T1** asserts
   at max\|Δ\| = 0 that with every new arm off this fork reproduces `t1`'s library-construction
   scores and its pinned sweep rows, which is what makes `t1`'s own P-F1 (vs `acappella/b1`, 14
   checks) and P-F2 (vs `prestissimo/a1g`, 16 checks) inherited rather than re-paid for.

2. **The kinematic unit is the REALISED PATH of the unit's own tape, re-measured, not the
   harvest's stored `traj`.** Every member's tape is executed on the plant from the launch key it
   was cut at, and the full `(x, v)` at every control step is recorded (`measure_paths`). Three
   reasons, in order of weight. (a) The harvest's `traj` records **position only**; the executor
   needs velocity, and differencing an interpolated position would manufacture a derivative the
   body never produced. (b) `build_level_cell` stores a welded member's `traj` as the
   *concatenation of its two parents' original trajectories*, and that is **not** what the weld
   does — the second half is launched from wherever the first actually ended. Re-measuring makes
   a level-ℓ unit's path the path of the level-ℓ unit. (c) It is measured content in `legato`
   F4's sense: the body produced it, so it carries no composition error. Cost: one grounding per
   member, charged to the ledger and reported (`path_ledger`).

3. **The path is expressed against PHASE over the unit's own span, and the clock is outside it.**
   `phi_j = j / N_src`; at a destination tempo the same curve is sampled at `phi_n = n / N_dst`
   and its velocities scale by `s = H_src / H_dst`, because `v = (dx/dphi) / T_span` and only
   `T_span` moved. There is no tempo in the stored object.
   **The derivative scheme, stated because it matters**: the stored pair is `(x, v)` as the body
   realised them, so velocity is **measured, never differenced from an interpolant**, and the
   only numerical operation is linear interpolation in phase. The acceleration the executor
   consumes is the destination grid's own one-step velocity difference — the quantity the exact
   inverse is exact for, and the quantity a model fitted on real `(v_n, a_n, u_n)` triples was
   fitted on. A continuous-time derivative of an interpolant would be a different object that
   the discrete plant never sees.

4. **Two inverse-dynamics schemes, both reported, `zoh` the scheme of record.** The plant holds
   its command constant across a control step, so its exact discrete map is
   `v_{n+1} = alpha v_n + (1 - alpha) v_term u_n` with `alpha = exp(-dt_ctrl / tau)`.
   - `zoh` — the exact inverse of that map.
   - `ct` — the brief's literal `u = (m a + c v) / gear`, i.e. the first-order scheme.
   Written as `u = c_a a + c_v v`, the two **agree exactly on drag** (`c_v = 1/v_term = 0.2`) and
   differ only on acceleration: `dt/((1-alpha) v_term) = 0.008717` against `1/A = 0.006000`, a
   ratio of **1.4528**. That factor is not a rounding: `dt_ctrl / tau = 0.8` on this body, so the
   control step is **not** small against the body and the two schemes are different functions.
   Both are free, so which one the body wants is a measurement. Gate **P-ZOH** checks that the
   discretisation the oracle inverts is the one the plant runs (`ksmoke0`: measured alpha
   0.44932909 vs analytic 0.44932896, rel 2.9e-07).

5. **The oracle knows the BODY, not the WORLD.** `fwd_zoh` / `inv_zoh` are built from the free-
   flight constants and contain nothing about the piece's command-rotation region. That is not an
   oversight but `acappella`'s standing choice, ported: the hard passage is left **unmodelled**
   rather than wrongly modelled (étude's `pretrain_mode=exclude`, model-free analogue). Measured
   cost, `ksmoke0`: the same forward step's one-step residual is `|dv|` rms **2.2e-07** on the
   clean world and **2.4e-01** on the rotated one. Inverting the rotation too is available as an
   instrument and is deliberately not the arm of record.

6. **A kinematic library is a `LevelLibrary` with converted content and nothing else changed.**
   Cells, slot ids, parents, poison twins, scores and **keys** are carried over verbatim, exactly
   as `resample_library` does, so `LadderDecider` plays a kinematic arm with no changes at all
   and the comparison against `rec` / `scl0` / `scl2` is a one-variable comparison. Keys are the
   **source** tempo's launch states and are not re-fit: a path cut at one tempo knows nothing
   about the hand-over distribution at another, and re-fitting them would be the kinematic arm
   quietly consuming information it has not paid for. `key` mode therefore selects on stale keys,
   `audit` mode on the plant, and both are reported at every cell.

7. **Four kinematic arms, because the transfer question needs its own control.**
   `kzs` / `kcs` carry the **slowest** tempo's paths to the asked tempo (the transfer question);
   `kzo` / `kco` convert the **asked** tempo's own paths at `s = 1` (the CONVERSION-COST
   control). `kzo / rec` is what the executor costs; `kzs / kzo` is what the tempo change costs
   once the executor is paid for. Without `kzo` a bad `kzs` cannot be attributed to either.
   `kzs` and `kzo` are identical by construction at the source tempo, which is a free consistency
   check on every run.

8. **`reflex_ec` and `reflex_ec0` are REPORTED reference rows and are never headlined.** Now that
   the learner holds a body model, the incumbent is only honest if the same model is offered to
   it; `accompanist` d3b's operator does that by carrying the Δ-stale read forward along the raw
   commands already issued. Two rows, because one alone is a strawman in either direction:
   `reflex_ec` keeps the Δ-fit gains (the same controller, a better read) and `reflex_ec0` plays
   the Δ = 0 gains (the controller an un-delayed read licenses). The reflex arm of record stays
   **model-free and re-fit per tempo**, exactly as in `accelerando`, and the model is never
   handed to the incumbent as a *planner* — it never proposes a command, only a read.

9. **The efference copy uses the RAW (pre-noise) command.** That is what the agent knows it asked
   for; the motor noise on top of it is exactly what the copy cannot contain (`offbook` round 7b,
   verbatim). At the sweep's `explore = 0` the two coincide, so this is a statement about the
   operator and not about any number in this node.

10. **P-K, the oracle identity, is gated OUTSIDE the rotation region and reported as a curve.**
    A member's kinematic unit converted back at its own tempo must reproduce its own commands.
    Tolerance **1e-3 of full scale, fixed before the gate was read**: `accelerando`'s P-E measured
    a 5% command error already spending the whole band at the deepest rung, so a residual 50×
    below the smallest command error that node could see in execution cannot move anything here.
    The physical floor it is held against is RK4 at 0.002 s on a 0.03 s time constant plus one
    float32 round-trip on the launch state. **The gate is reported as a residual-vs-gate-weight
    table**, so "the residual is the rotation and nothing else" is something anyone can check
    rather than a claim. `ksmoke0`, over 5120 steps at two tempi:

    | the world's gate weight over the step | steps | rms \|du\| | max \|du\| |
    |---|---|---|---|
    | < 1e-6 | 3434 | 2.33e-08 | 1.19e-07 |
    | 1e-6 – 1e-4 | 350 | 1.90e-08 | 8.94e-08 |
    | 1e-4 – 1e-2 | 446 | 2.68e-04 | 1.45e-03 |
    | 1e-2 – 1 | 890 | 8.18e-02 | 5.04e-01 |

    Out-of-region max 1.19e-07 against a 1e-3 tolerance — four orders inside it, and at the
    integrator's own floor.

11. **The region weight is the MAX over the step's two endpoints, not its start — corrected after
    `ksmoke0` measured an apparatus fact.** The env re-evaluates its Gaussian gate at every 2 ms
    **physics** substep; at the fastest tempo the body covers 0.06 m in one 24 ms control step
    against a region sigma of 0.0244 m, i.e. **2.5 sigma per step**. Classifying a step by its
    start state alone therefore put genuinely in-region steps in the out-of-region half and P-K
    read `out_max` 4.8e-03 against an `out_rms` of 1.8e-04 — a tail, not a floor. Taking the max
    over both endpoints bounds the traversal for any step that does not enter and leave between
    them, which at 2.5 sigma it does not. **The threshold itself was not moved** — it is the
    env's own 1e-4 cutoff — and the bucket table above is published precisely so that this
    correction can be audited rather than trusted: the residual collapses to 2.3e-08 wherever the
    gate is off and rises monotonically with it.

12. **`P-KN`: the phase resampler splits at a span's midpoint at max\|Δ\| = 0.** On a dyadic
    ladder the destination sample counts are exact multiples at every rung, so this is an identity
    and not a tolerance. It is what licenses reading a kinematic level-ℓ unit as two kinematic
    level-(ℓ−1) units in sequence — the kinematic counterpart of P-N, and the reason a kinematic
    library can be nested at all. Note what it does **not** say: a welded unit's *path* is not the
    concatenation of its parents' paths (decision 2b), so P-KN is about the resampler, and the
    content difference is a property of the weld that decision 2 measures rather than hides.

13. **`app = fixed` only, for the Phase 1 smoke.** `accelerando` decision 11 made it the arm of
    record and measured `arm` to blow the lead-in up on this body (hand-over 3.3–5.3 legs).
    Running one mode halves the sweep. `arm` is available behind `--app-modes arm,fixed` and is
    the obvious thing to add at the run of record if the orchestrator wants it.

14. **The saturation fraction is an instrument reported at every (arm, tempo, level).** The
    ceiling of the factored architecture is an actuator ceiling before it is anything else:
    `accelerando`'s P-T found feasibility binding at `R* = 0.160` with the schedule itself asking
    `u_turn = 0.87` at 48 ms notes. A converted path that asks for more than the actuator has is
    not a failure of the factoring, and the clip fraction is what separates the two.

15. **Phase 1 has NO learned component at all.** No torch, no head, no fitted model anywhere. The
    reflex law, executed content, the resampler, the exact inverse dynamics as an
    experimenter-side oracle, and the plant as the only judge. The learned cerebellum is Phase 2.

## Gates

| gate | what it asserts | result |
|---|---|---|
| **P-F0** | the 13 donor constants, `ast`-read out of `etude/etude.py` | `ksmoke0`: 13/13 — **pass** |
| **P-T1** | with every new arm off, this fork reproduces `accelerando/t1` at max\|Δ\| = 0 | `ksmoke1`: **75 checks, max\|Δ\| = 0.000e+00, applicable=True** — pass |
| **P-G** | one pinned per-(tempo, Δ) gain cell re-fit from the full 63-cell grid | `ksmoke1`: cell (H=8, Δ=5) re-fit **[1.2, 0.12]** vs pinned [1.2, 0.12] — pass |
| **P-TAU** | τ measured from a step response on both bodies | `ksmoke0`: fast 30.00 ms vs 30.00 (rel 1.7e-04); donor 499.9 vs 500.0 — **pass** |
| **P-ZOH** | the discretisation the oracle inverts is the one the plant runs | `ksmoke0`: α measured 0.44932909 vs analytic 0.44932896, rel **2.9e-07** — **pass** |
| **P-R** | the command resampler is the identity at s = 1 | `ksmoke0`: 0.000e+00 over 120 members — **pass** |
| **P-N** | the nesting op is exact at every tempo | `ksmoke1`: 504 spelled, 0 bad, weld 6.02e-08 — **pass** |
| **P-KN** | the phase resampler splits at the span midpoint | `ksmoke1`: **0.000e+00** over 504 members — **pass** |
| **P-K** | THE ORACLE IDENTITY, outside the rotation region | `ksmoke1`: out max **9.37e-05** (tol 1e-03), rms 1.09e-06 over 56652 steps — **pass** |
| **P-P** | parity per slot against the same-tempo recording (decision 16) | `ksmoke2`: see the run record |

## Arms (Phase 1)

| arm | content | groundings |
|---|---|---|
| `reflex` | the reflex law, gains re-fit per tempo on the clean world. **Model-free.** | 0 |
| `reflex_ol` | INSTRUMENT: the same law read once at the drilled launch, replayed open-loop | 0 |
| `reflex_ec` / `reflex_ec0` | REFERENCE ROWS, never headlined: the incumbent given the same body model through the delay (efference copy), at the Δ-fit and at the Δ = 0 gains | 0 |
| `key_Lℓ` / `aud_Lℓ` | `rec` — this tempo's own recording. The ORACLE content at an unpracticed tempo | 0 / `n_slot × m_member` per decision |
| `key_all` / `aud_all` | `rec`, every rung legal at every aligned seam | as above |
| `key_s0_Lℓ` / `aud_s0_Lℓ` | `scl0` — the slowest tempo's tape, time-resampled | as above |
| `key_s2_Lℓ` / `aud_s2_Lℓ` | `scl2` — the same × `s²` | as above |
| `key_kzs_Lℓ` / `aud_kzs_Lℓ` | **THE HEADLINE** — the slowest tempo's PATHS, exact (`zoh`) inverse at the asked tempo | as above |
| `key_kzo_Lℓ` / `aud_kzo_Lℓ` | the CONVERSION-COST control — this tempo's own paths, `zoh`, `s = 1` | as above |
| `key_kcs_Lℓ` / `aud_kcs_Lℓ` | the slowest tempo's paths, continuous-time (`ct`) inverse | as above |
| `key_kco_Lℓ` / `aud_kco_Lℓ` | this tempo's own paths, `ct`, `s = 1` | as above |

Every arm plays the same shared approach at the same Δ; the approach ledger is subtracted.
62 arms per (tempo, app mode).

## Smokes and sizing

| tag | what | wall |
|---|---|---|
| `ksmoke0` | `--quick`, tempi {8, 2}, tiny config: plumbing and every config-independent gate end to end | **53 s** |
| `ksmoke1` | `t1`'s knobs, tempi **{32, 8, 2}** (the source at 768 ms notes, the target at 48 ms — steps of 4× and 16×), `app=fixed`, so P-T1 and P-G are applicable | **825 s** (library 362 · kin 0 · sweep 353) |
| `ksmoke2` | `ksmoke1` re-run with the parity gate (decision 16) added; nothing else changed | launched detached 2026-09-06, `ap-UFzxYI4p937HCIPvOqgF9L` |
| `kdsmoke0` | `--quick --arm-set diag --diagnostics`: the two new instruments and the three filtered arms end to end | **58 s** |
| `kdsmoke1` | the DIAGNOSTIC smoke: `t1`'s knobs, tempi **{32, 16, 8, 2}** so the dyadic ladder's own 2× step is on the record beside 4× and 16×, `--arm-set diag --diagnostics` | launched detached 2026-09-06, `ap-8WtjjzEDub0PjhlFGENYqr` |
| `k1` | Phase 1 of record: 5 tempi, `app=fixed`, plus `app=arm` at the fastest tempo only (decision 17) | **NO GO** (2026-09-06) — the ceiling smoke is not the result the factoring needed |

## Known limitations, stated not hidden

- **The oracle is an oracle.** `kin_oracle` reads the body's true constants. It is labelled as an
  experimenter-side oracle everywhere, exactly as `rec` at an unpracticed tempo is, and its whole
  purpose is to be a ceiling. Nothing about Phase 1 says a learner could reach it; Phase 2 asks.
- **The body model does not contain the command-rotation region** (decision 5), so every
  kinematic arm is systematically wrong at drilled segment 3 and P-K measures by exactly how
  much. That is a property of the design, not a defect, but it is a floor under every kinematic
  number in the node and no kinematic arm should be expected to reach `rec` at `s = 1`.
- **Any tag at a mass other than 1.0 leaves the donor plant** (`accelerando`'s standing note): no
  cross-tag control against `acappella`, `solo`, `etude` or `prestissimo` applies to treatment
  numbers here, and none is claimed.
- **The library's slot partition is degenerate at four of five tempi** on this substrate (`t1`'s
  P-S: 1.000×, 1 of 6 distinct argmins), so `key`-mode selection has little to select on. Carried
  from `accelerando`; both modes are reported.

16. **A parity gate, per slot per tempo, against the same-tempo recording (`P-P`).** `solo`'s
    gate with the tempo added to the index and with **both sides content** rather than a head and
    a tape: a slot is OPEN at a tempo iff the kinematic arm's own members, executed on the plant
    from that slot's held-out launch states, beat the same-tempo recording's **per-state audition
    winner** (not a fixed member) on at least τ = 0.75 of them. It was not in `ksmoke1` and the
    numbers below are therefore missing it; it is built and populates in `ksmoke2`, whose config
    is otherwise `ksmoke1`'s exactly.
    *Held out means held out twice over*: the launch states come from a traversal at the
    **deployment** condition (Δ = 5, `app = fixed`, level-1 keyed) rather than the harvest
    condition the construction audition ran at — which already makes them different states at
    every drilled seam past the first, but **not at drilled seam 0**, where `app = fixed` plays
    the same Δ = 0 lead-in at the same harvest gains and the hand-over is numerically identical
    to `score_set`'s. So the `[n_score:]` slice is kept on top of it. Both guards, not either.
    The reference side is an instrument and is not charged; the arm side is charged.

17. **Phase 1's run of record is `app = fixed` at all five tempi, with `app = arm` kept at the
    fastest tempo only, as the continuity row.** From the orchestrator, 2026-09-06. The delayed
    lead-in was measured as an artifact in `prestissimo` (decision 20, `zero` withdrawn as a gain
    effect wearing a read effect's name) and again in `accelerando` (flag 1: `arm` hand-over 3.3
    legs at H = 8 and 5.3 at H = 2, against 0.18 / 0.25 under `fixed`). It does not need five
    tempi again; one row at the fastest tempo keeps the series comparable with `t1`'s.

18. **The Phase 3 correction fork is a LADDER, not an endpoint — recorded now, built later.**
    From the orchestrator, 2026-09-06, in answer to a fork I raised and did not resolve. Neither
    "read every step" (`accompanist` d3b's operator, one feedback event per step, which gives up
    the level's whole one-read-against-sixteen economy) nor "read once at launch and dead-reckon"
    (the strongest cerebellar claim) is the arm of record on its own. The arm of record is the
    **ladder between them**: the state read every 1, 2, 4, 8 and 16 steps with the forward model
    dead-reckoning between reads, and the headline is **the fewest reads per span at which the
    corrected unit stays in band**. Every-step is then the row directly comparable with
    `reflex_ec`; once-at-launch is the far end. That puts the level's economy on the meter's own
    coordinate instead of choosing a side of the fork by taste. **Nothing for Phase 3 is built
    until Phases 1 and 2 have their go.**

## Run record — `ksmoke1`, 2026-09-06 (Phase 1's smoke, 825 s)

16 CPUs, seed 0, `t1`'s knobs, R\* = 0.160, tempi {32, 8, 2} = 768 / 192 / 48 ms notes,
Δ = 5 (120 ms), `app = fixed`, band 0.0611, do-nothing 0.3603 at every tempo.
**9 of 9 gates pass**, P-T1 and P-G both applicable.

**P-K's residual against the world's own rotation-gate weight**, 56652 out-of-region steps and
15924 in-region:

| gate weight w over the step | steps | rms \|du\| | max \|du\| |
|---|---|---|---|
| < 1e-6 | 52080 | 3.92e-08 | 1.79e-07 |
| 1e-6 – 1e-4 | 4572 | 3.84e-06 | 9.37e-05 |
| 1e-4 – 1e-2 | 6958 | 7.80e-04 | 7.89e-03 |
| 1e-2 – 1 | 8966 | 1.58e-01 | 1.19e+00 |

**The transfer table** (`aud`, `e_piece`, band 0.0611, `*` = in band). `kzo` is the same-tempo
conversion (s = 1, the executor's own cost); `kzs` is the same executor carrying the 768 ms
paths to the asked tempo.

| H | L | rec | kzo | kzs | scl0 | kzo/rec | kzs/kzo | kzs/scl0 | sat(kzs) |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0454\* | 0.0408\* | 0.0408\* | 0.0454\* | 0.90× | 1.00× | 0.90× | 0.000 |
| 32 | 2 | 0.0266\* | 0.0399\* | 0.0399\* | 0.0266\* | 1.50× | 1.00× | 1.50× | 0.000 |
| 32 | 3 | 0.0259\* | 0.0326\* | 0.0326\* | 0.0259\* | 1.26× | 1.00× | 1.26× | 0.000 |
| 32 | 4 | 0.0263\* | 0.0213\* | 0.0213\* | 0.0263\* | 0.81× | 1.00× | 0.81× | 0.000 |
| 8 | 1 | 0.0312\* | 0.0312\* | 0.1611 | 0.1093 | 1.00× | 5.17× | 1.47× | 0.000 |
| 8 | 2 | 0.0289\* | 0.0243\* | 0.2340 | 0.1412 | 0.84× | 9.64× | 1.66× | 0.000 |
| 8 | 3 | 0.0075\* | 0.0183\* | 0.3144 | 0.1742 | 2.43× | 17.20× | 1.80× | 0.000 |
| 8 | 4 | 0.0123\* | 0.0226\* | 0.2944 | 0.1461 | 1.84× | 13.04× | 2.01× | 0.000 |
| 2 | 1 | 0.0453\* | 0.0222\* | 0.3999 | 0.2168 | 0.49× | 18.04× | 1.84× | 0.000 |
| 2 | 2 | 0.0298\* | 0.0343\* | 0.2626 | 0.2323 | 1.15× | 7.65× | 1.13× | 0.000 |
| 2 | 3 | 0.0334\* | 0.0501\* | 0.1968 | 0.2356 | 1.50× | 3.93× | 0.84× | 0.000 |
| 2 | 4 | 0.0069\* | 0.0578\* | 0.3184 | 0.2452 | 8.38× | 5.51× | 1.30× | 0.000 |

At `H = 32` the two kinematic columns are identical to the last digit at every level, which is
the free consistency check decision 7 named: at the source tempo `kzs` and `kzo` are the same
object.

**Saturation is 0.000 at every (arm, tempo, level)** — `kzs`, `kzo`, `kcs`, `kco` alike. The clip
never binds anywhere in this tag, so nothing here is an actuator ceiling. The schedule's own
`u_turn` at the fastest tempo is 0.87 (decision 14's guard, unchanged from `t1`).

**The derivative scheme** (`ct` against `zoh`, same content, same source):
`kco/kzo` runs 0.68–1.40× at H = 32, 0.59–1.16× at H = 8, 1.30–2.29× at H = 2;
`kcs/kzs` runs 0.65–1.17× at H = 8 and 0.90–1.17× at H = 2. Neither scheme dominates.

**The reference rows** (never headlined): reflex / `reflex_ec` / `reflex_ec0` = 0.0037 / 0.0252 /
0.0013 at H = 32, 0.0174 / 0.0661 / 0.0079 at H = 8, 0.0690 / 0.1219 / 0.0320 at H = 2.

**The era table** (best arm at each tempo, and best among the DEPLOYABLE arms — those whose
content exists without practising at the asked tempo):

| H | best (any) | e | in band | best deployable | e | in band | reflex |
|---|---|---|---|---|---|---|---|
| 32 | key_L3 | 0.0175 | yes | key_s0_L3 | 0.0175 | yes | 0.0037 |
| 8 | aud_L3 | 0.0075 | yes | aud_kcs_L1 | 0.1059 | **no** | 0.0174 |
| 2 | aud_L4 | 0.0069 | yes | aud_kzs_L3 | 0.1968 | **no** | 0.0690 |

**Open diagnostic, not answered by this tag and not speculated about here**: `kzo` is in band at
every tempo and level, so the executor works and the conversion is not the failure; saturation is
zero, so the actuator is not the failure; and `kzs` is out of band at every level at both
unpractised tempi. What separates them is only `s`. The smallest step this tag can express is 4×
(32 → 8); the run of record's five-rung ladder adds 2× (32 → 16), which is the nearest step the
dyadic ladder has, and `accelerando/fsmoke_b` measured the force-level program's cliff to sit at
the practiced/unpractised boundary rather than at a step size.

## Figures — `ksmoke1`

`figures/ksmoke1/`: `f1_transfer.png` (every content source across tempo, one panel per level) ·
`f2_decompose.png` (conversion cost `kzo/rec` against transfer cost `kzs/kzo`) ·
`f3_pk.png` (P-K's residual against the world's own gate weight) ·
`f4_saturation.png` · `f5_grades.png` (both grades, shallowest and deepest rung).

## `ksmoke2` — `ksmoke1` plus parity (decision 16), 2026-09-06

Config identical to `ksmoke1`; every gate and every sweep row reproduces it. **Parity per slot
against the same-tempo recording**, open / checked at τ = 0.75:

| arm | H32 L1 | L2 | L3 | L4 | H8 L1 | L2 | L3 | L4 | H2 L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kzs` | 28/48 | 15/24 | 6/12 | 0/6 | **6/48** | **2/24** | **0/12** | **0/6** | **8/48** | **5/24** | **0/12** | **0/6** |
| `kzo` | 28/48 | 15/24 | 6/12 | 0/6 | 28/48 | 8/24 | 6/12 | 0/6 | 14/48 | 7/24 | 0/12 | 0/6 |
| `kcs` | 26/48 | 7/24 | 0/12 | 0/6 | 7/48 | 1/24 | 0/12 | 0/6 | 11/48 | 7/24 | 0/12 | 0/6 |
| `kco` | 26/48 | 7/24 | 0/12 | 0/6 | 19/48 | 3/24 | 0/12 | 0/6 | 6/48 | 0/24 | 0/12 | 0/6 |

At H = 32 `kzs` and `kzo` are the same object and their rows are identical, as they must be.
Mean executed error on the held-out states, arm / reference — `kzs` at H = 8 runs
0.1239/0.0448 (L1) to **0.5183/0.0065** (L4), and at H = 2 0.0815/0.0352 to 0.2822/0.0062;
`kzo` at the same cells runs 0.0476/0.0448 to 0.0253/0.0065 and 0.0392/0.0352 to 0.0389/0.0062.
**No arm opens a single L4 slot at any tempo, including `kzo` at the tempo it was cut at**, and
`rec`'s own L4 reference error there is 0.0062–0.0194 — the tightest cell in the table.

## The diagnostic round — decisions 19–21

*From the orchestrator, 2026-09-06, after reading `ksmoke1`: **no go on `k1`**. `kzo` is in band
everywhere and `kzs` is out of band at every level at both unpractised tempi with zero
saturation, so before anything scales, one diagnostic smoke that locates why, with each reading
pre-fixed. Tempi {32, 16, 8, 2}, so the 2× step the dyadic ladder actually has is on the record
beside 4× and 16×.*

19. **PATH REPRESENTABILITY, instrumented before it is treated.** The 768 ms path was shaped by a
    controller with 32 control steps per note; at 48 ms notes there are 2. A corner that exists in
    the slow path may not be expressible by any 2-step-per-note command sequence, and the exact
    inverse of an unrepresentable path produces something else. The instrument is a **shape
    distance**: the mean-over-phase distance between LAUNCH-ALIGNED position curves — the slow
    path resampled to the target grid against the target tempo's own realised paths in the same
    cell — reported per level beside `kzs`'s error. Both the **nearest** destination path (is
    there anything like this here at all?) and the **mean** over all of them are kept, so
    `nearest` is not a cherry-pick. The scale is supplied by the destination library **against
    itself** (`d_within`: each member to the nearest other member in its cell), because a cross
    distance in metres means nothing until it is read against the spread the destination's own
    content already has. **Pre-fixed reading**: does `d_near / d_within` track `kzs`'s error
    across tempi and levels? If it does, the invariant is not the path but the path up to the
    resolution the tempo permits.

20. **The pre-fixed treatment: low-pass the stored path before conversion. TWO widths, and the
    reason both are run.** The filter is a centred boxcar with edge replication — `_boxcar`, the
    listener's-grade kernel, reused so the node has one filter — applied on the **source** grid to
    position and velocity with the SAME kernel (a boxcar is linear and time-invariant, so `v` and
    `dx/dt` stay consistent). Edge replication moves the unit's own endpoints; that is inherent to
    the treatment and is said here rather than hidden.
    - `note` (**the treatment**, `w = H_src` source samples = one note): removes every feature
      finer than a note, i.e. exactly the structure a destination tempo with few steps per note
      cannot express. Arm `kzsf`, with `kzof` (own source, same filter) as **the filter's own cost
      control** — the same discipline as decision 7, so a `kzsf` that fails cannot be blamed on
      the filter without checking what the filter costs content that already worked.
    - `aa` (**an apparatus check, not a treatment**, `w = round(H_src/H_dst)` = one destination
      control step): phase resampling from 32 to 2 steps a note is a 16× decimation by point
      sampling, and without anti-aliasing it folds whatever the source path carries above the
      destination's Nyquist into the destination band. That is a possible defect in **my own
      resampler**, not a hypothesis about the body, and it has to be excluded before the treatment
      is read. Arm `kzsa`.
    **Pre-fixed reading**: does `kzsf` enter the 0.0611 band at any (tempo, level) where `kzs` is
    outside it, and does `kzsa` move at all?

21. **LAUNCH MISMATCH, instrumented and given its own rollout.** The stored path's initial
    velocity, scaled, is the SLOW lead-in's; the hand-over the target tempo actually delivers is
    the FAST lead-in's. On a 30 ms body a velocity error decays in about one control step but
    leaves a position error of order `|Δv|·τ`, which at 2.5 m/s is 0.075 m — a quarter leg
    against a 0.0611 m band. Two measurements, neither predicted: (a) the scaled initial `(x, v)`
    of every unit logged beside the actual hand-over `(x, v)` per tempo, with `dx`, `dv` and
    `dv·τ`; (b) the unit's converted commands executed from **both** launches — the actual
    hand-over (what the sweep does) and the path's own scaled initial state (the diagnostic) —
    as the mean waypoint error over the unit's own span, which at the top rung is exactly
    `e_piece`. **Pre-fixed reading**: if the own-launch row is in band where the actual-launch row
    is not, it is the launch; if it is not in band either, it is the path.
    Both diagnostics consume the **same held-out launch states parity uses** (decision 16) and
    neither is charged: they are instruments the experimenter reads, not decisions any arm makes.

    **The arm set is trimmed to pay for them.** `--arm-set diag` drops the `scl2` arms — `ksmoke1`
    measured them worst at every unpractised cell (0.31–0.99 against a 0.0611 band) — and the two
    `ct` families, since `ksmoke1` measured neither inverse scheme to dominate (`kco/kzo`
    0.59–2.29×). That is 16 arms back, which buys the three filtered arms and the fourth tempo at
    the same arm count as the ceiling tag: 62.

    **The listener's grade is reported for `kzs` at all four levels and all three widths** beside
    `e_piece`, not only at the shallowest and deepest rung. A corner-cut path that stays in the
    right place would pass the listener's clock and fail the waypoint grade, and that disagreement
    is what the two grades were pre-fixed to read (`accelerando` decision 8).

## Run record — `kdsmoke1`, 2026-09-06 (the diagnostic smoke)

16 CPUs, seed 0, `t1`'s knobs, R\* = 0.160, tempi **{32, 16, 8, 2}** = 768 / 384 / 192 / 48 ms
notes (steps of **2× / 4× / 16×** from the source), Δ = 5, `app = fixed`, band 0.0611,
`--arm-set diag --diagnostics`. **9 of 9 gates pass**; P-T1 **100 checks at max\|Δ\| = 0.000e+00**,
P-G applicable and exact.

### Decision 19's reading — the shape distance

`d_near / d_within`, per level (the scale-free form; `d_near` and `d_within` in metres are in the
report):

| H | none | aa | note | `kzs` error L1→L4 |
|---|---|---|---|---|
| 32 | 0.0 / 0.0 / 0.0 / 0.0 | same | 0.9 / 1.3 / 1.7 / 1.4 | 0.0408 · 0.0399 · 0.0326 · 0.0213 |
| 16 | 1.7 / 2.3 / 2.6 / 3.7 | 1.4 / 2.1 / 2.6 / 3.7 | 1.2 / 1.7 / 2.6 / 3.7 | 0.1227 · 0.1321 · 0.1478 · 0.2940 |
| 8 | 2.9 / 3.5 / 4.0 / 5.3 | 2.1 / 3.0 / 3.9 / 5.2 | 1.5 / 2.4 / 3.9 / 5.3 | 0.1611 · 0.2340 · 0.3144 · 0.2944 |
| 2 | 14.1 / 18.9 / 16.9 / 11.2 | 9.0 / 14.8 / 15.3 / 11.1 | 9.0 / 14.7 / 16.3 / 12.0 | 0.3999 · 0.2626 · 0.1968 · 0.3184 |

`d_near` in absolute terms is nearly FLAT across tempi (0.0253–0.0313 at L1, 0.0275–0.0295 at L4);
what moves by 7× is `d_within`, the destination library's own spread (0.0155 → 0.0022 at L1 from
H = 32 to H = 2). The ratio's rise is therefore mostly its denominator, and the two series do not
order together across the whole grid — `d_near/d_within` at L4 rises 3.7 → 5.3 → 11.2 while
`kzs`'s L4 error runs 0.2940 → 0.2944 → 0.3184.

### Decision 20's reading — the two filters

`aud`, `e_piece`, band 0.0611, `*` = in band:

| H | L | `kzs` | `kzsa` (anti-alias) | `kzsf` (note LP) | `kzo` | `kzof` (LP cost control) | `scl0` |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 0.1227 | 0.0836 | 0.0745 | 0.0404\* | 0.0775 | 0.0677 |
| 16 | 2 | 0.1321 | **0.0426\*** | **0.0463\*** | 0.0253\* | 0.0726 | 0.0623 |
| 16 | 3 | 0.1478 | **0.0426\*** | **0.0451\*** | 0.0416\* | 0.0356\* | 0.0858 |
| 16 | 4 | 0.2940 | **0.0446\*** | **0.0506\*** | 0.0271\* | 0.0405\* | 0.0951 |
| 8 | 1 | 0.1611 | 0.0868 | 0.1700 | 0.0312\* | 0.0413\* | 0.1093 |
| 8 | 2 | 0.2340 | 0.0838 | 0.0807 | 0.0243\* | 0.0666 | 0.1412 |
| 8 | 3 | 0.3144 | **0.0529\*** | 0.1316 | 0.0183\* | 0.0278\* | 0.1742 |
| 8 | 4 | 0.2944 | **0.0550\*** | 0.0640 | 0.0226\* | 0.0407\* | 0.1461 |
| 2 | 1 | 0.3999 | 0.3498 | 0.3035 | 0.0222\* | 0.0877 | 0.2168 |
| 2 | 2 | 0.2626 | 0.2033 | 0.1612 | 0.0343\* | 0.0509\* | 0.2323 |
| 2 | 3 | 0.1968 | 0.1210 | 0.0726 | 0.0501\* | 0.0275\* | 0.2356 |
| 2 | 4 | 0.3184 | 0.0660 | 0.0957 | 0.0578\* | 0.0629 | 0.2452 |

**`kzsa` moves every unpractised cell and puts six of them in band, four at the deep rungs.**
The apparatus check was supposed to exclude a defect in my own resampler and did not: it found
one. Phase resampling from 32 steps a note to 2 by point sampling is a 16× decimation with no
anti-aliasing, and every `kzs` number in `ksmoke1`, `ksmoke2` and this tag's `kzs` column is a
number about an aliased path. **`ksmoke1`'s `kzs` column and its "kzs/kzo 3.9–18.0×" reading are
therefore retracted as a measurement of tempo transfer**; they measure a resampler.
`kzsa` is what `kzs` should have been from the start, and decision 22 makes it the arm of record.
The note-period filter (`kzsf`) is neither uniformly better nor worse than the anti-alias width,
and its own cost control `kzof` is NOT free — at H = 32 it takes L4 from 0.0213 to 0.0461 (2.2×)
and at H = 8 L2 from 0.0243 to 0.0666 (2.7×).

Parity, open/checked at τ = 0.75: `kzsa` H16 **19/48 · 5/24 · 0/12 · 0/6**, H8 **22/48 · 4/24 ·
0/12 · 0/6**, H2 13/48 · 4/24 · 1/12 · 0/6, against `kzs` H16 6/48 · 0/24 · 0/12 · 0/6 and H8
6/48 · 2/24 · 0/12 · 0/6. **No arm opens an L4 slot at any tempo**, `kzo` included.

### Decision 21's reading — the launch

Mean over ALL members of a cell (the sweep's `aud` figure is a per-state **min** over members, so
these two are not the same statistic and are not comparable digit for digit).

| arm | H | L4: e from own launch | e from actual | dx | dv | dv·τ |
|---|---|---|---|---|---|---|
| `kzs` | 16 | 0.5932 | 0.5986 | 0.0042 | 0.826 | 0.0248 |
| `kzs` | 8 | 0.5807 | 0.5946 | 0.0043 | 1.652 | 0.0495 |
| `kzs` | 2 | 0.3178 | 0.3316 | 0.0077 | 6.606 | 0.1982 |
| `kzsa` | 16 | 0.0734 | 0.0696 | 0.0038 | 0.219 | 0.0066 |
| `kzsa` | 8 | **0.0493\*** | **0.0535\*** | 0.0028 | 0.883 | 0.0265 |
| `kzsa` | 2 | 0.0802 | 0.0758 | 0.0055 | 3.333 | 0.1000 |

**The pre-fixed reading fires on its second branch for `kzs`**: the own-launch row is out of band
at every unpractised cell (0.2110–0.5932), so `kzs`'s failure was not the launch. For `kzsa` the
two rows are within 8% of each other at every cell, so the launch is not the residual there
either. `dv` is large everywhere (0.22–8.50 m/s) and `dv·τ` reaches **0.2549 m at H = 2**, 4.2×
the band — the launch gap is physically able to matter and is measured not to be what is
happening.

### Decision 21's third reading — the two grades do not disagree

`kzs`'s pass fractions at ½ / ⅓ / ¼ leg are **0.00 / 0.00 / 0.00 under BOTH grades at every
unpractised cell, all four levels**. Where `kzsa` enters the band it reads 1.00 / 0.00 / 0.00
under both. The corner-cut-but-in-place signature the listener's clock was pre-fixed to detect
(`accelerando` decision 8) does not appear anywhere in this tag.

22. **`kzsa` — the anti-aliased conversion — becomes the kinematic arm of record, and `kzs` is
    kept beside it as the aliased control.** Not a preference: `resample_path` decimates by point
    sampling, which is a defect in the resampler and not a property of the body or of the
    factoring, and it was found by the apparatus check that was written down to exclude it. The
    unfiltered arm stays in every table so the size of the defect remains on the record and
    `ksmoke1`/`ksmoke2` remain readable as what they were. The `note` filter stays as a treatment
    with `kzof` as its cost control; it is not folded into the arm of record, because `kzof`
    measures it to cost 2.2–2.7× on content that already works.

## Figures — `kdsmoke1`

`figures/kdsmoke1/`: `f1_transfer.png` (now including `kzsa`, `kzsf`, `kzof`) ·
`f2_decompose.png` · `f3_pk.png` · `f4_saturation.png` · `f5_grades.png`.

## THE PHASE 1 CEILING RESULT (on `kzsa`, the arm of record) — `kdsmoke1`, 2026-09-06

*This is the reading of record for Phase 1. Every `kzs` figure before decision 22 measured an
aliased resampler and is retained only as the control.*

**A unit stored as a path at 768 ms notes, converted by the exact inverse dynamics at a faster
tempo, plays IN BAND (0.0611) at:**

| step | tempo | in band at | `kzsa` L1 → L4 | `kzs` (aliased control) |
|---|---|---|---|---|
| 2× | H = 16, 384 ms | **L2, L3, L4** | 0.0836 · **0.0426\* · 0.0426\* · 0.0446\*** | 0.1227 · 0.1321 · 0.1478 · 0.2940 |
| 4× | H = 8, 192 ms | **L3, L4** | 0.0868 · 0.0838 · **0.0529\* · 0.0550\*** | 0.1611 · 0.2340 · 0.3144 · 0.2944 |
| 16× | H = 2, 48 ms | none (L4 at **0.0660** against 0.0611) | 0.3498 · 0.2033 · 0.1210 · 0.0660 | 0.3999 · 0.2626 · 0.1968 · 0.3184 |

**Never at L1, at any step.** The in-band set is a wedge: it opens at the deep rungs and closes
from the shallow end as the step grows. Against the same-tempo recording (`rec`, the oracle
content, not deployable) and the naive resample (`scl0`) at the same cells: at H = 8 L4, 0.0550
against `rec` 0.0123 and `scl0` 0.1461.

**The defect and the mechanism are the same object, and both should be said.** The anti-alias
width is `H_src / H_dst` — it is *by construction* a **tempo-dependent low-pass**. So the
resampler that decision 22 corrected is, read from the other side, exactly decision 19's
hypothesis: the invariant is not the path but **the path up to the resolution the tempo
permits**, and the correct resampling operation is the one that enforces that resolution. The
apparatus fix and the scientific reading coincide. That is why `kzsf` (the note-period filter,
a *coarser* tempo-independent low-pass) does not beat it: the right width is the one the tempo
ratio names, not the one the score does.

### Named finding — the deep rung is bounded by something no executor can fix

**No arm opens a single L4 parity slot at any tempo — `kzo` and `kzsa` included, at the tempo
they were cut at.** Parity is execution reproduction on the plant from held-out launch states
against the same-tempo recording's own per-state audition winner; the L4 reference error there is
0.0062–0.0194, the tightest cell in the table. Forces reconstructed from a measured path do not
reach the verbatim tape at the deep rung, even when the reconstruction is exact and even at
`s = 1`. This is `accelerando`'s P-E exchange rate — a 5% command error spending the whole band
at the deepest rung, because the span is many τ long and nothing re-grounds — appearing a third
time, and it **bounds what any executor can do with a path**, learned or oracular. It is not a
statement about tempo transfer: it holds where there is no transfer at all.

23. **Decision 19's denominator is `d_near / band`, in metres, with the ratio to `d_within` kept
    as a column.** From the orchestrator, 2026-09-06. `d_within` collapses 7× across the ladder
    (0.0155 → 0.0022 at L1) because the fast library is nearly one path — `accelerando`'s
    degenerate slot partition, P-S 1.000× with 1 of 6 distinct argmins at four of five tempi — so
    the ratio rises largely for its denominator and is not the readout it looked like.
    Against the band the picture is flat and legible instead: `d_near` at L4 is **0.028 m at
    every tempo against a 0.061 band**, i.e. 0.46 of the band, which is the same wedge `kzsa`'s
    landing in band at the deep rung describes.

24. **The stale hand-over is logged per seam, for every arm.** A level-ℓ arm re-decides at
    `K / 2^(ℓ-1)` drilled seams — **8 / 4 / 2 / 1** at L1 / L2 / L3 / L4 — and every one is a
    launch from a Δ-old read. `accelerando/fsmoke_a2` measured a deep-best / shallow-worst
    ordering it explicitly could not explain, and `kzsa`'s in-band wedge has the same shape. The
    candidate mechanism is that **depth amortises the stale read**, and the number that tests it
    is the read's own error at each seam (`read_dx`, `read_dv` — arm-independent, a property of
    the delay and the tempo) beside the arm's realised per-seam error (`e_seg`). It was not on
    `kdsmoke1`'s log; it is collected from `klsmoke1` onward for every arm at no extra plant cost
    (`collect_obs` was already allocated).

## Phase 2 — the learned cerebellum: decisions

25. **The fitted family's target is the true inverse's own functional form, and that is the
    point.** The exact inverse is `u = c_a a + c_v v` with `c_a = 0.008717` and `c_v = 0.200000`
    — LINEAR in `(v, a)` with two free numbers. So the minimal family has exactly the right form
    and the MLP does not, which is what makes the inductive-bias question sharp rather than a
    matter of capacity: a linear map fitted anywhere on the `(v, a)` plane extrapolates to the
    whole plane by construction, and an MLP fitted on the slow tempi' corner of it has no reason
    to. A faster tempo scales `|v|` by `s` and `|a|` by `s²`, so it is literally a request for a
    region of `(v, a)` slow practice never visited, and `coverage` reports how far outside the
    fit each tempo's query actually lands rather than leaving "extrapolation" an adjective.
    **The linear family is fitted as a general 2×4 map with a bias** — the body's isotropy and
    the which-channel-depends-on-what are NOT given to it, because handing a family the answer's
    structure would make its success a statement about the experimenter. The recovered
    coefficients are reported against the analytic ones at every fit.

26. **The training data is the practice traversals, re-executed for their velocities, at ONE
    tempo.** `build_ladder`'s harvest already runs the reflex law with motor noise and records
    `(s0, cmds)` per drilled seam; `rollout_path` replays exactly those and returns the full
    `(x, v)`, so the triples are the body's own executed data and the replay is deterministic and
    bit-identical to the harvest. It is charged as a grounding anyway, because replaying a
    rendition is a trial.
    **Fitted on H = 32 alone**, the slowest rung and the source of the stored paths, so that
    *every* target tempo is an extrapolation. Fitting on {32, 16} would put the 2× rung — the
    cell where the oracle just succeeded — inside the training set and contaminate the cleanest
    reading; the variant is one flag (`--fit-tempos 32,16`) and is not the smoke of record.
    Gate **P-FH** asserts the train/hold split is disjoint rather than assuming it.

27. **Pre-fixed reading for Phase 2, written before the smoke ran: how much of the ORACLE's
    in-band set survives, per family per tempo.** `kzsa` is the ceiling and `kfl` / `kfm` sit on
    exactly its resampler, so the only thing that differs is where the inverse came from. The
    summary statistic is the count of in-band (tempo, level) cells: ceiling vs linear vs MLP, with
    the per-cell ratios `kfl/kzsa` and `kfm/kzsa` beside them, and the held-out command MSE at
    every tempo beside the execution numbers so a family that fits and still fails is
    distinguishable from one that does not fit.

## `klsmoke0` / `klsmoke1` — Phase 2's smokes

| tag | what | wall |
|---|---|---|
| `klsmoke0` | `--quick --arm-set learn --fit-tempos 8`: the fit, both families, P-FH, and the two fitted arms through the sweep and parity end to end | **87 s** |
| `klsmoke1` | `t1`'s knobs, tempi {32, 16, 8, 2}, `--arm-set learn --fit-tempos 32 --fit-max-rend 48` | launched detached 2026-09-06, `ap-wUvFbz3lqbu6UHiHdTwrkq` |

`klsmoke0`'s plumbing numbers (tiny config, H_src = 8, **not readable as results**): P-FH 614
train / 154 hold, overlap 0; the linear family recovered `c_v` 0.203536, 0.191925 against
0.200000 and `c_a` 0.008770, 0.008626 against 0.008717, with off-diagonal max 2.6e-02 and bias
max 8.0e-03.

## Run record — `klsmoke1`, 2026-09-06 (Phase 2's smoke, 1077 s)

`t1`'s knobs, tempi {32, 16, 8, 2}, `app = fixed`, `--arm-set learn --fit-tempos 32
--fit-max-rend 48`. **P-T1 100 checks at max\|Δ\| = 0.000e+00**; P-G exact; P-FH 9830 train /
2458 hold, overlap 0. Timings: library 394 · fit **5** · parity 146 · sweep 497.

### Decision 27's pre-fixed reading — how much of the ceiling survives

**In-band (tempo, level) cells of 16: ceiling `kzsa` 9, linear `kfl` 0, MLP `kfm` 6.**

| H | L | band | `kzsa` (ceiling) | `kfl` (linear) | `kfm` (MLP) | kfl/kzsa | kfm/kzsa |
|---|---|---|---|---|---|---|---|
| 32 | 1–4 | 0.0611 | .0408\* .0399\* .0326\* .0213\* | .1053 .0656 .1105 .1003 | **.0362\*** **.0458\*** .0631 .0845 | 2.58 1.65 3.39 4.70 | 0.89 1.15 1.94 3.96 |
| 16 | 1–4 | 0.0611 | .0836 .0426\* .0426\* .0446\* | .0898 .1017 .1111 .1016 | **.0601\*** .0689 **.0568\*** .0910 | 1.07 2.39 2.61 2.28 | 0.72 1.62 1.33 2.04 |
| 8 | 1–4 | 0.0611 | .0868 .0838 .0529\* .0550\* | .0909 .0753 .1119 .0966 | .0748 **.0534\*** **.0531\*** .0989 | 1.05 0.90 2.11 1.76 | 0.86 0.64 1.00 1.80 |
| 2 | 1–4 | 0.0611 | .3498 .2033 .1210 .0660 | .2885 .2424 .1255 .1440 | .2306 .1832 .1263 .1869 | 0.82 1.19 1.04 2.18 | 0.66 0.90 1.04 2.83 |

**The MLP keeps two thirds of the ceiling's in-band set and the linear family keeps none** — the
opposite ordering to the one decision 25 set up. `kfm` is also in band at three cells the oracle
is NOT (H = 32 L1, H = 16 L1, H = 8 L2), and at the deepest rung both families lose it
everywhere (`kfm/kzsa` 3.96× at H = 32 L4). Parity, open/checked: `kfm` L1 8/48 · 8/48 · 15/48 ·
4/48 across the four tempi against `kzsa`'s 28 · 19 · 22 · 13; **L4 is 0/6 for every arm at every
tempo**, as it is for the oracle.

### Why the linear family fails, measured rather than inferred

The recovered coefficients, fitted as a general 2×4 map with a bias on H = 32 alone:

| | fitted (x, y) | analytic |
|---|---|---|
| drag `c_v` | **0.104180, 0.086787** | 0.200000 |
| accel `c_a` | 0.007578, 0.007415 | 0.008717 |
| off-diagonal max | 3.02e-02 | 0 |
| bias max | 3.67e-03 | 0 |

**The drag coefficient comes back at half its true value.** The family has exactly the right
functional form and two free numbers and still misses one of them, at the tempo it was fitted on.
The same fit run on H = 8 in `klsmoke0` recovered 0.203536, 0.191925 — so this is a property of
**fitting where the body barely moves**, not of the family. At H = 32 the schedule asks
`u_turn = 0.04` while the practice noise is σ = 0.0628, so the drag term `c_v·v ≈ 0.032` sits
*below* the exploration noise that dominates the command; the coefficient is weakly identified
there and the fit says so. That is a statement about **where a cerebellum can be trained**, and
it was not anticipated: decision 25 predicted the linear family would win *because* it has the
right form.

### The extrapolation axis is VELOCITY, not acceleration — and that inverts decision 25's premise

| H | fitted? | lin mse | mlp mse | v reach | a reach | v frac out | a frac out |
|---|---|---|---|---|---|---|---|
| 32 | yes | 4.049e-03 | 2.456e-03 | 1.34× | 1.16× | 0.007 | 0.004 |
| 16 | no | 5.145e-03 | 5.514e-03 | 1.69× | 0.98× | 0.023 | 0.000 |
| 8 | no | 4.781e-03 | 4.625e-03 | 1.06× | 0.85× | 0.002 | 0.000 |
| 2 | no | 4.945e-02 | 7.018e-02 | 1.62× | **1.15×** | **0.398** | **0.102** |

Decision 25 argued that a faster tempo scales `|a|` by `s²` and is therefore a request for a
region of `(v, a)` slow practice never visited. **The acceleration reach never exceeds 1.16× at
any tempo and is BELOW 1.0 at two of them.** The reason is on the record already: the practice
noise is a *command* noise of σ = 0.0628 on a body with `A = 167 m/s²`, i.e. ~10.5 m/s² of
noise-driven acceleration at every tempo, which dwarfs the schedule's own. Slow practice on a
noisy actuator already covers the acceleration axis. What it does not cover is **velocity** —
39.8% of the H = 2 query lies outside the fit's 99.5th percentile. The framing in decision 25 was
wrong in its specifics and is corrected here rather than quietly dropped; the instrument that
caught it (`coverage`) was built because the framing might be wrong.

### Decision 24's column, first reading

`aud_kzsa`, mean read error at the launches and `e_seg` across the eight drilled seams:

| arm | launches | H32 read_dx | H16 | H8 | H2 | H2 `e_seg` peak |
|---|---|---|---|---|---|---|
| L1 | 8 | 0.0186 | 0.0432 | 0.0907 | **0.3780** | 0.518 |
| L2 | 4 | 0.0192 | 0.0454 | 0.0910 | 0.3053 | 0.395 |
| L4 | 1 | 0.0207 | 0.0347 | 0.0711 | **0.2352** | 0.111 |

**The read error is NOT arm-independent, contrary to what this instrument's own first draft
asserted** (the analyzer text is corrected). It is the distance the body travels in Δ steps, so
it depends on where the arm has put the body and how fast it is going: at H = 2 it differs by
1.6× between L1 and L4. Both it and the realised per-seam error order with depth at the fastest
tempo, which is the shape the amortisation candidate predicts — but the two are not independent
measurements of it, since a worse arm produces a worse read, and this tag cannot separate them.

## Figures — `klsmoke1`

`figures/klsmoke1/` (five, as for the other tags).

28. **The refit control: `--fit-tempos {32,16}` and `--fit-tempos {8}`, both families, as a
    smoke.** From the orchestrator, 2026-09-06, after `klsmoke1`. The linear family's failure —
    `c_v` recovered at 0.104/0.087 against 0.200 while the same family on H = 8 in `klsmoke0`
    recovered 0.204/0.192 — is a **candidate identifiability effect**: at H = 32 the schedule
    asks `u_turn = 0.04` while the practice noise is σ = 0.0628, so the drag term `c_v·v ≈ 0.032`
    sits *below* the noise dominating the command, and a coefficient the data cannot see is not
    a statement about the family. This is the control that separates the two.
    - `{32, 16}` — the pianist practising at two slow tempi. Adds range on the velocity axis
      without reaching the tempi the program is asked at, except H = 16, which stops being an
      unpractised rung for the executor and is reported as such.
    - `{8}` — the identifiability check at the tempo where the coefficient came back right,
      accepting that H = 8 is no longer "slow" and that the arm is then a different claim.
    **Reported per fit set**: the recovered `c_v` and `c_a` against the analytic 0.200000 and
    0.008717, decision 27's in-band-cell count per family, held-out MSE at every tempo, and the
    `coverage` instrument — **velocity is the axis** (`klsmoke1`: `v` reach 1.62× with 39.8% of
    the H = 2 query outside the fit, against an `a` reach that never exceeds 1.16×), so the
    reading is how much of the velocity axis each fit set actually covers.
    **Fix carried in with it**: the per-tempo held-out row now uses *that tempo's own* held-out
    slice rather than the pooled one. With a single fit tempo the two are the same object and
    `klsmoke1` is unaffected; with two they are not, and the pooled version would have made the
    H = 32 and H = 16 rows the same number under different headers.

## Run record — `k1`, 2026-09-06. **PHASE 1 OF RECORD**, 1601 s

16 CPUs + L4 (idle: no fit in this tag), seed 0, `t1`'s knobs, R\* = 0.160, five tempi
H ∈ {32, 16, 8, 4, 2} = 768 / 384 / 192 / 96 / 48 ms notes, Δ = 5 (120 ms), band 0.0611,
do-nothing 0.3603, `app = fixed` as the arm of record with `app = arm` at H = 2 only.
`--arm-set record --diagnostics`. Timings: library 648 · parity 189 · diagnostics 75 · sweep 649.

**Gates, 9 of 9.** P-F0 13/13 · **P-T1 125 checks at max\|Δ\| = 0.000e+00** · P-G re-fit
[1.2, 0.12] = pinned · P-TAU 30.00 ms vs 30.00 (donor 499.9 vs 500.0) · P-ZOH α 0.44932909 vs
0.44932896 · P-R 0.000e+00 over 405 members · P-N 840 spelled, 0 bad, weld 6.02e-08 · **P-KN
0.000e+00 over 840** · **P-K out-of-region max 9.37e-05 (tol 1e-03), rms 9.02e-07 over 83020
steps**, with the residual collapsing to 3.80e-08 where the world's rotation gate is off.

### The headline: the in-band set is a wedge that closes from the shallow end

`kzsa` (`aud`, `e_piece`), the exact executor on the anti-aliased resampler, carrying the 768 ms
paths:

| step | H | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 1× (source) | 32 | 0.0408\* | 0.0399\* | 0.0326\* | 0.0213\* |
| **2×** | 16 | 0.0836 | **0.0426\*** | **0.0426\*** | **0.0446\*** |
| **4×** | 8 | 0.0868 | 0.0838 | **0.0529\*** | **0.0550\*** |
| **8×** | 4 | 0.2298 | 0.1230 | 0.0886 | **0.0555\*** |
| **16×** | 2 | 0.3498 | 0.2033 | 0.1210 | 0.0660 |

**Never at L1 at any step; L4 survives to 8× and misses at 16× by 1.08×** (0.0660 against
0.0611). Against the same cells: `rec` (the same-tempo recording, the ORACLE content, not
deployable) L4 0.0093 / 0.0123 / 0.0104 / 0.0069; `scl0` L4 0.0951 / 0.1461 / 0.1839 / 0.2452;
the re-fit reflex 0.0103 / 0.0174 / 0.0267 / 0.0690; `kzs` (the aliased control) L4 0.2940 /
0.2944 / 0.2588 / 0.3184.

**The frozen key beats the plant audition at three of four unpractised tempi.** `key_kzsa` reads
0.0339 at H = 16 L2, 0.0498 at H = 8 L4 and 0.0486 at H = 4 L4, against `aud_kzsa`'s 0.0426,
0.0550 and 0.0555 — with the SOURCE tempo's keys, which decision 6 declined to re-fit. Not
anticipated; the era table's best deployable arm is `key_kzsa` at H = 16, 8 and 4.

**Era table, best DEPLOYABLE arm against the re-fit reflex**: H = 16 `key_kzsa_L2` 0.0339
(**0.30×**), H = 8 `key_kzsa_L4` 0.0498 (**0.35×**), H = 4 `key_kzsa_L4` 0.0486 (**0.55×**),
H = 2 `aud_kzsa_L4` 0.0660 (**1.05×** — the one cell where the incumbent wins). At one feedback
event per traversal against 16–256.

**Saturation is 0.000 at every (arm, tempo, level)**, all four arms. No clip binds anywhere in
this tag; the schedule's own `u_turn` at the fastest tempo is 0.87.

**Both grades agree everywhere.** `kzsa` L4 `e_listen` runs 0.0214 / 0.0441 / 0.0485 / 0.0499 /
0.0568 against the `e_piece` column above; pass fractions at ½ / ⅓ / ¼ leg are 1.00 / 0.00 / 0.00
at every in-band unpractised cell under both grades. The corner-cut-but-in-place disagreement the
listener's clock was pre-fixed to detect (`accelerando` decision 8) does not appear in this tag
either.

**Parity — the named finding, confirmed at five tempi.** No arm opens a single L4 slot at any
tempo, `kzo` at s = 1 included; the L4 reference error is 0.0062–0.0194. `kzsa` L1 open/checked:
28/48 · 19/48 · 22/48 · 4/48 · 13/48 across the ladder; `kzs` 28 · 6 · 6 · 1 · 8.

**The shape instrument, in metres against the band** (decision 23's form, `lp = aa`): `d_near` is
**0.41–0.50 of the band at L2–L4 and 0.32–0.34 at L1, FLAT across all four unpractised tempi**
(0.0198–0.0308 m). It does not order with `kzsa`'s error, which varies 7× across the same cells.

**The launch diagnostic** (`kzsa` L4, own-launch vs actual): 0.0655/0.0716 · 0.0734/0.0696 ·
0.0493/0.0535 · 0.0472/0.0548 · 0.0802/0.0758 — **within 8% at every tempo**, so the residual is
not the launch, at five tempi as at four. `dv·τ` reaches 0.1000 m at H = 2.

**The stale-read column** (decision 24), mean `read_dx` at the arm's own launches:
L1 (8 launches) 0.0186 / 0.0432 / 0.0907 / 0.2493 / 0.3780; L2 (4) 0.0192 / 0.0454 / 0.0910 /
0.1511 / 0.3053; L4 (1) 0.0207 / 0.0347 / 0.0711 / 0.1339 / 0.2352. It orders with depth from
H = 16 down, by 1.6× at H = 2 — but it is a joint property of the delay, the tempo AND the arm
(a worse arm puts the body somewhere worse), so this tag cannot separate amortisation from that.

**The continuity row** (`app = arm`, H = 2 only): `e_app` 0.1139, reflex 0.1020, `aud_kzsa_L4`
0.1154, `aud_L4` 0.1447, `aud_s0_L4` 0.1898, `aud_kzs_L4` 0.2367 — the lead-in blows up as
`prestissimo` decision 20 and `accelerando` flag 1 measured, and every arm degrades with it.

**Reference rows, never headlined**: `reflex_ec` 0.0252 / 0.0434 / 0.0661 / 0.0884 / 0.1219 and
`reflex_ec0` 0.0013 / 0.0027 / 0.0079 / 0.0183 / 0.0320 against the reflex's 0.0037 / 0.0103 /
0.0174 / 0.0267 / 0.0690.

**Two reduction bugs found and fixed while reducing this tag, both in the analyzer only** (no run
was affected): the report defaulted to `app_modes[0]`, which is `arm` here, so the first pass
printed the continuity row as the whole table; and the era table's "deployable" set and the
saturation table's arm list were hard-coded from the ceiling round, which silently excluded
`kzsa` — the arm of record — from the era table. Both are now derived from the arms actually
present.

## Figures — `k1`

`figures/k1/`: `f1_transfer.png` · `f2_decompose.png` · `f3_pk.png` · `f4_saturation.png` ·
`f5_grades.png`.

## `klsmoke2` / `klsmoke3` — decision 28's refit control

| tag | fit set | handle |
|---|---|---|
| `klsmoke2` | H ∈ {32, 16} — the pianist practising at two slow tempi | `fc-01M1TMGYEDCHHKQM80F92P4ZM9` |
| `klsmoke3` | H ∈ {8} — the identifiability check, where `klsmoke0` recovered `c_v` at 0.20 | `fc-01M1TMHDND4AKJ707NTERY2X74` |

Both `--arm-set learn --tempos 32,16,8,2 --fit-max-rend 48`, launched 2026-09-06.

## Run record — `klsmoke2` / `klsmoke3`, 2026-09-06: decision 28's refit control

`klsmoke2` 1516 s (fit {32, 16}, P-FH 14746/3686, overlap 0) · `klsmoke3` 1506 s (fit {8},
P-FH 2458/614, overlap 0). Both `t1`'s knobs, tempi {32, 16, 8, 2}, `--arm-set learn`.
**P-T1 100 checks at max\|Δ\| = 0.000e+00 in both.**

### Decision 28's reading: IDENTIFIABILITY fired, not the family

| fit set | `c_v` (x, y) vs 0.200000 | `c_a` (x, y) vs 0.008717 | in-band cells: ceiling / linear / MLP |
|---|---|---|---|
| **{32}** (`klsmoke1`) | 0.104180, 0.086787 — **52% / 43%** | 0.007578, 0.007415 | 9 / **0** / 6 |
| **{32, 16}** (`klsmoke2`) | 0.168617, 0.159768 — **84% / 80%** | 0.008360, 0.008296 | 9 / **7** / 4 |
| **{8}** (`klsmoke3`) | 0.199206, 0.195891 — **99.6% / 98%** | 0.008751, 0.008721 | 9 / **6** / 6 |

**The linear family's in-band count tracks its recovered drag coefficient and nothing else**:
0.10 → 0 cells, 0.17 → 7 cells, 0.199 → 6 cells. The family was never the problem. Decision 28's
first branch fires: at H = 32 the schedule asks `u_turn = 0.04` while the practice noise is
σ = 0.0628, so the drag term `c_v·v ≈ 0.032` sits below the noise dominating the command and the
coefficient is not identifiable there. Give the fit one more slow tempo (adding H = 16, i.e.
doubling the velocity range) and it recovers 84% of `c_v` and **7 of the ceiling's 9 in-band
cells** — more than the MLP gets from any fit set. `klsmoke1`'s reading, that the MLP beats the
minimal family, is **withdrawn**: it was a statement about where the fit was taken, not about
inductive bias. Decision 25's *premise* (the linear family has the right functional form and two
free numbers) stands; its *prediction* failed only because one of the two numbers was invisible
in the data it was given.

**The MLP does not benefit the same way** — 6 / 4 / 6 across the three fit sets, roughly flat and
never better than the best linear fit. Nothing here explains that.

### Held-out command MSE per tempo (per-tempo slices; `fitted` rows use that tempo's own hold-out)

| fit set | H32 lin/mlp | H16 | H8 | H2 |
|---|---|---|---|---|
| {32} | **4.049e-03** / 2.456e-03 | 5.145e-03 / 5.514e-03 | 4.781e-03 / 4.625e-03 | 4.945e-02 / 7.018e-02 |
| {32,16} | **4.786e-03** / 2.385e-03 | **1.243e-03** / 1.507e-03 | 1.158e-03 / 1.268e-03 | 1.262e-02 / 2.390e-02 |
| {8} | 5.574e-03 / 5.775e-03 | 1.234e-03 / 1.222e-03 | **1.298e-03** / 1.146e-03 | 6.984e-03 / 7.154e-03 |

(bold = a fitted tempo.) The H = 2 held-out MSE falls **4×** from fit {32} to fit {32,16} and
**7×** to fit {8}, at a tempo none of them practised.

### Coverage — velocity is the axis (decision 28's reporting requirement)

| fit set | v reach @ H2 | v frac outside @ H2 | a reach @ H2 | a frac outside @ H2 |
|---|---|---|---|---|
| {32} | 1.62× | **0.398** | 1.15× | 0.102 |
| {32,16} | 1.53× | **0.328** | 1.19× | 0.109 |
| {8} | 1.46× | **0.266** | 1.56× | 0.172 |

Adding velocity range to the fit set monotonically reduces the fraction of the H = 2 query that
falls outside it (0.398 → 0.328 → 0.266), and the H = 2 held-out MSE falls with it. The
acceleration axis stays covered throughout, as `klsmoke1` first measured.

### An observation this tag CANNOT resolve, stated as such

At fit {8} both fitted families beat the **exact** executor at the deep rung at three of four
tempi: `kfl` L4 reads 0.0316 (H = 8) and 0.0435 (H = 16) and `kfm` L4 reads 0.0309, 0.0324 and
0.0336, against `kzsa`'s 0.0550, 0.0446 and 0.0213. A fitted model beating the oracle it
approximates has an obvious candidate — the fit is taken on data from the ROTATED world, so it
can absorb some of the command rotation the body model deliberately does not contain (decision 5)
— but that candidate is **not tested here**, and there is a competing one that this tag cannot
rule out: `klsmoke3`'s linear coefficients match the analytic ones to under 2% while its bias
term is 8.9e-03, which on a schedule asking `u_turn = 0.17` at H = 8 is a ~5% command
perturbation, and `accelerando`'s P-E measured 5% command error to spend the whole band at the
deepest rung. A perturbation that size can move a deep-rung number either way. Both readings are
on the record and neither is adopted.

29. **`klsmoke1`'s inductive-bias reading is withdrawn.** It said the MLP keeps two thirds of the
    ceiling's in-band set while the minimal family keeps none, and framed that as the opposite of
    decision 25's prediction. `klsmoke2` and `klsmoke3` show it was a property of the fit set:
    the linear family reaches 7 of 9 cells once its drag coefficient is identifiable. What
    survives from `klsmoke1` is the measurement that produced the retraction — the recovered
    coefficient printed beside the execution number at every fit — which is why the reading could
    be withdrawn on evidence rather than argued about.

## Phase 3 — the corrected executor: decisions

30. **The read ladder is the arm's own coordinate, per decision 18.** The executor produces
    feedforward from the kinematic program and corrects it online against a FORECAST:
    ```
    if h % R == 0:            # a READ, charged as one feedback event
        s_hat <- the Delta-stale true read, rolled forward through the commands THIS EXECUTOR
                 has already issued  (the efference copy, on the learner's side of the meter)
    u = ff[h] + kp (x*(h) - s_hat_x) + kd (v*(h) - s_hat_v)
    s_hat <- fm(s_hat, u)     # dead-reckon to the next step
    ```
    `R ∈ {1, 2, 4, 8, 16}` control steps. `R = 1` reads every step and is the row directly
    comparable with `reflex_ec`; `R ≥ span` reads only at launch and is the strongest cerebellar
    claim; the ladder between them puts the level's economy on the meter's own coordinate instead
    of choosing a side of the fork. **Every read is charged as one feedback event**, including
    the launch read, so the ladder appears in the ledger exactly as the reflex's per-step reads
    do, and `reads_per_span` is reported beside every row.
    **The efference buffer starts EMPTY at the unit's launch.** For the first Δ steps of a span
    the copy has fewer than Δ commands to roll forward, because the commands issued before the
    launch belong to the approach or to the previous unit and this executor did not issue them.
    That is the honest boundary condition for a learner's own copy, and it is the same
    disadvantage every other arm faces at a launch.

31. **Both forward models are ONE BODY MODEL with the inverse, not a second fit.** The exact pair
    (`inv_zoh`, `fwd_zoh`) is two directions of the same two constants, `α` and `v_term`. The
    fitted forward model is derived from the fitted LINEAR inverse by algebraic inversion of its
    2×2 acceleration block — `a = W_a⁻¹(u − W_v v − b)`, `v' = v + dt·a` — and **introduces no
    new parameters**. Position in both is the trapezoidal integral of the velocity channel, which
    is kinematics and not dynamics: a learner that did not know position is the integral of
    velocity would have no kinematic program to execute in the first place. So the exact and the
    fitted executor are the same KIND of object and the comparison between them is a comparison
    of one body model's accuracy. (The MLP inverse is deliberately NOT given a forward direction:
    inverting it would need a numerical solve per step and would stop being the same object.)

32. **The PD rule, fixed before the grid is read.** With a correct forecast the correction loop
    carries no delay, so the position-error dynamics are `ë + (A·kd + 1/τ)ė + A·kp·e = 0`. The
    rule asks for **critical damping at a natural frequency of one NOTE** — the correction
    settles inside the unit's smallest musical division, which is the only time-scale the piece
    supplies — giving `ω = 2π/T_note`, `kp = ω²/A`, `kd = (2ω − 1/τ)/A`. Apparatus only: `A`, `τ`
    and the note duration; no arm is read.
    The grid searched around it is **centred on that value** (`kp × {¼, ½, 1, 2, 4}`,
    `kd × {½, 1, 2, 4}`), so an edge means "the body wanted something 4× from the analytic value"
    — a legible statement rather than a grid artefact — and both edges are reported per tempo.
    The fit is taken **once per tempo on the CEILING configuration** (exact forward model, ORACLE
    feedforward, `R = 1`, level 1) on the **clean world**, which is `accelerando` decision 15's
    discipline for the reflex's own gains. **The resulting table is then used by every treatment
    row** — every read count, both forward models, both feedforward sources — so no arm can move
    its own gains.

33. **Feedforward is the {32,16} linear inverse, with oracle feedforward as a control row.**
    `klsmoke2` made the {32,16} linear fit the learned cerebellum of record (7 of the ceiling's 9
    in-band cells, `c_v` at 84% of true). `kzsa`-oracle feedforward is run at `R ∈ {1, 16}` as
    the control that says how much of any correction result is the feedforward rather than the
    correction. **KEY mode only** for every corrected arm, and the reason is measured: `k1` found
    `key_kzsa` to BEAT `aud_kzsa` at three of four unpractised tempi, so frozen source-tempo keys
    are not a handicap — and auditioning a closed-loop policy would let the arm pay groundings
    the open-loop wedge it is compared against never paid.
    **Pre-fixed readings**: (a) the fewest reads per span at which the corrected unit is in band,
    per tempo and level, against `k1`'s open-loop wedge; (b) P-E with and without correction at
    each read count — the uncorrected row is the same executor with no reads past launch, so the
    two sides differ in exactly the correction — reading off the ε at which the band is spent per
    level, against `accelerando`'s 5% bar; (c) the `reflex_ec` row beside the `R = 1` row, which
    is the comparable pair.

34. **The clean-world fit (`kflc`), the free discriminator for `klsmoke3`'s open observation.**
    The same harvested renditions replayed on the world with the **rotation region off**, and the
    linear family refitted on that. If the clean-world fit's bias and off-diagonals collapse while
    its coefficients stay put, the rotated-world fit's bias IS the absorbed rotation; if the bias
    survives, it is not. One extra `rollout_path` batch per fit tempo — it is free, which is the
    condition it was authorised under — plus four key-mode arms to execute it.

## Run record — `kc1`, 2026-09-06 (Phase 3's smoke, 1729 s)

`t1`'s knobs, tempi {32, 16, 8, 2}, `app = fixed`, `--arm-set corr --fit-tempos 32,16
--fit-clean --corrected --no-parity`. **P-T1 100 checks at max\|Δ\| = 0.000e+00**; P-K
out-of-region max 9.37e-05; P-FH 14746 / 3686, overlap 0. Timings: library 574 · fit 9 ·
corrected 560 · sweep 548.

### Decision 34's discriminator, at real config — it fires

| fit | `c_v` | `c_a` | off-diag max | bias max |
|---|---|---|---|---|
| rotated world | 0.168617, 0.159768 | 0.008360, 0.008296 | **1.84e-02** | **3.58e-03** |
| **clean world** | **0.200000, 0.200000** | **0.008717, 0.008717** | **9.90e-11** | **3.68e-11** |

On clean data the linear family recovers the analytic constants to ten significant figures with
zero off-diagonal and zero bias. The rotated-world fit's off-diagonal and bias terms **are the
rotation**. `klsmoke3`'s open observation now has one of its two candidates confirmed as present:
a fitted executor does absorb the command rotation the exact one is denied by decision 5. Whether
that is what makes it beat the oracle at the deep rung is still not established — the bias is
also of P-E-sensitive size — but the "bias term is an artefact" reading is no longer available.

### The PD gains (decision 32), per tempo, with their edges

| H | analytic kp | analytic kd | chosen | kp | kd | e_piece | kp edge | kd edge |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.4016 | 1e-06 | ×2 / ×16 | 0.8032 | 1.6e-05 | 0.0024 | no | **EDGE** |
| 16 | 1.606 | 1e-06 | ×2 / ×16 | 3.213 | 1.6e-05 | 0.0142 | no | **EDGE** |
| 8 | 6.426 | 0.1927 | ×4 / ×2 | 25.7 | 0.3854 | 0.0138 | **EDGE** | no |
| 2 | 102.8 | 1.371 | ×0.0625 / ×2 | 6.426 | 2.742 | 0.1063 | **EDGE** | no |

**The analytic centre is a poor one at three of four tempi, and at the two slowest the kd rule is
DEGENERATE.** `kd = (2ω − 1/τ)/A` goes negative when `2ω < 1/τ`, i.e. at any tempo slower than
`T_note > 4πτ = 377 ms` — H = 32 and H = 16 — where it is clamped to 1e-06 and the multiplier
grid is then multiplying zero, so the ×16 "edge" there is an edge on a degenerate centre and
means nothing. The body's own drag already over-damps the correction loop at slow tempi. At
H = 8 the grid wants 4× the analytic kp and at H = 2 **1/16 of it**, so the one-note bandwidth
target is wrong in both directions on either side. Reported, not fixed: the rule was fixed before
the grid was read and a rule changed after reading a grid is not the same rule.

### (a) The fewest reads per span at which the corrected unit is in band

`x` = never in band at any read count on the ladder. **The ladder is capped at R = 16 control
steps**, so at slow tempi its coarsest rung is still many reads per span (16 at H = 32 L4);
those entries are the LADDER's floor, not a measured minimum. The genuine one-read points are the
open-loop columns.

| H | L | exact FM | fitted FM | open `kfl` (1 read) | open `kzsa` (1 read) | `k1` aud_kzsa |
|---|---|---|---|---|---|---|
| 32 | 1 | **2** (R=16) 0.0148 | **2** 0.0204 | 0.0522\* | 0.0399\* | 0.0408\* |
| 32 | 2 | **4** (R=16) 0.0433 | **4** 0.0486 | 0.0997 | 0.0872 | 0.0399\* |
| 32 | 3 | **8** (R=16) 0.0373 | **8** 0.0438 | 0.0604\* | 0.0803 | 0.0326\* |
| 32 | 4 | **16** (R=16) 0.0214 | **16** 0.0246 | 0.0627 | 0.0839 | 0.0213\* |
| 16 | 1 | **1** (R=16) 0.0582 | **1** 0.0551 | 0.1202 | 0.0776 | 0.0836 |
| 16 | 2 | **2** (R=16) 0.0609 | **2** 0.0423 | 0.0694 | 0.0339\* | 0.0426\* |
| 16 | 3 | **4** (R=16) 0.0571 | **4** 0.0566 | 0.0673 | 0.0498\* | 0.0426\* |
| 16 | 4 | **8** (R=16) 0.0368 | **8** 0.0519 | 0.0821 | 0.0719 | 0.0446\* |
| 8 | 1 | **2** (R=4) 0.0429 | **8** (R=1) 0.0433 | 0.1420 | 0.1285 | 0.0868 |
| 8 | 2 | **4** (R=4) 0.0402 | **4** 0.0476 | 0.0564\* | 0.0545\* | 0.0838 |
| 8 | 3 | **2** (R=16) 0.0610 | **2** 0.0589 | 0.0555\* | 0.0587\* | 0.0529\* |
| 8 | 4 | **4** (R=16) 0.0540 | **4** 0.0595 | 0.0574\* | 0.0498\* | 0.0550\* |
| 2 | 1 | **x** | **x** | 0.2950 | 0.2919 | 0.3498 |
| 2 | 2 | **x** | **x** | 0.1260 | 0.0846 | 0.2033 |
| 2 | 3 | **x** | **x** | 0.1398 | 0.1167 | 0.1210 |
| 2 | 4 | **x** | **x** | 0.1160 | 0.0938 | 0.0660 |

**At H = 2 the corrected unit is out of band at every read count and every level**, including
R = 1 (16 reads per L4 span, the same count the reflex pays) — 0.1105 / 0.1639 / 0.1401 / 0.1137
against the open-loop `key_kzsa`'s 0.2919 / 0.0846 / 0.1167 / 0.0938 and `k1`'s best deployable
0.0660. The fitted forward model tracks the exact one closely everywhere except H = 8 L1, where
it needs 8 reads against the exact model's 2.

### (c) `reflex_ec` beside the R = 1 row, and the oracle-feedforward control

| H | reflex (fb) | reflex_ec | reflex_ec0 | cor R=1 L1 (fb) | cor R=1 L4 (fb) | orc-ff R=1 L1 | L4 |
|---|---|---|---|---|---|---|---|
| 32 | 0.0037 (256) | 0.0252 | 0.0013 | 0.0146 (256) | 0.0223 (256) | 0.0044 | 0.0232 |
| 16 | 0.0103 (128) | 0.0434 | 0.0027 | 0.0164 (128) | 0.0293 (128) | 0.0173 | 0.0288 |
| 8 | 0.0174 (64) | 0.0661 | 0.0079 | 0.0288 (64) | 0.0262 (64) | 0.0367 | 0.0265 |
| 2 | 0.0690 (16) | 0.1219 (16) | 0.0320 (16) | 0.1105 (16) | 0.1137 (16) | 0.1063 | 0.1025 |

At equal feedback the corrected unit beats `reflex_ec` at every tempo (0.0146 vs 0.0252 at
H = 32; 0.1105 vs 0.1219 at H = 2) and loses to both the plain reflex and `reflex_ec0` at every
tempo. Oracle feedforward and the {32,16}-fitted feedforward are within 20% of each other at
R = 1 everywhere except H = 8 L1.

### (b) P-E, with and without correction — the ε at which the band is spent

| H | L | R=1 | R=4 | R=16 | open (1 read) |
|---|---|---|---|---|---|
| 32 | 1 | 0.179 | 0.176 | 0.173 | **0.096** |
| 32 | 4 | 0.172 | 0.169 | 0.166 | **0.123** |
| 16 | 1 | **>0.2** | **>0.2** | 0.049 | 0.049 |
| 16 | 4 | **>0.2** | **>0.2** | **>0.2** | already out |
| 8 | 1 | **>0.2** | 0.082 | already out | already out |
| 8 | 4 | **>0.2** | **>0.2** | **>0.2** | already out |
| 2 | 1 | already out | already out | already out | already out |
| 2 | 4 | already out | already out | already out | already out |

**Where the arm is in band at all, correction relaxes the bar well past `accelerando`'s 5%.**
At H = 8 L4 the open-loop row is already outside the band at ε = 0 (0.1073) while every corrected
row tolerates the full ε = 0.20 ladder without leaving it (0.0262 → 0.0290 at R = 1). At H = 32
the open-loop row spends the band at ε ≈ 0.10–0.12 and the corrected rows at ε ≈ 0.17. At H = 2
no row is in band at ε = 0, so the question is undefined there.

### The ledger, feedback events per traversal (drilled)

`reflex` 256 / 128 / 64 / 16 across H = 32 / 16 / 8 / 2; `key_kzsa_L4` **1** at every tempo;
the corrected L4 rows R = 1 / 2 / 4 / 8 / 16 pay 256 / 128 / 64 / 32 / 16 at H = 32 and
16 / 8 / 4 / 2 / 1 at H = 2. **The corrected arm buys its accuracy with reads at a rate the
level's economy was built to avoid**: at H = 32 L4 correction takes 0.0839 (1 read) to 0.0214
(16 reads), a 3.9× gain for 16× the feedback.

35. **The read ladder does not reach the once-at-launch end at slow tempi, and that is a design
    limitation of the ladder as specified, not a result.** `R` is in control steps and caps at
    16, while a level-4 span at H = 32 is 256 steps; the coarsest rung there is still 16 reads.
    The comparison that IS clean is the corrected rows against the open-loop rows in the same
    table, which are the genuine one-read points. A ladder in **reads per span** rather than
    control steps would ask the intended question at every tempo, and is the obvious fix if this
    line continues.

## Figures — `kc1`

`figures/kc1/` (five, as for the other tags).

36. **A new PD rule, posed in DISCRETE time: place both eigenvalues of the exact error system at
    `rho`.** Written down, with its derivation and its reasons, before its grid was read.
    With the exact inverse as feedforward and a state estimate, the tracking error
    `e = (x* − x, v* − v)` obeys a two-state **linear discrete** system, exactly and not to first
    order — writing `g = (1−α)v_term`, `h = v_term(dt − τ(1−α))`:

    ```
    e_{n+1} = [[1 − h·kp,   τ(1−α) − h·kd],
               [  −g·kp,        α − g·kd  ]] e_n
    ```

    the plant's own update and the feedforward's definition cancel the tempo, the piece and the
    path out of the error dynamics entirely. Placing both eigenvalues at `ρ` gives `trace = 2ρ`
    and `det = ρ²`; **the `kp·kd` terms cancel in the determinant**, so both conditions are linear
    in `(kp, kd)` and the solution is closed form:

    ```
    kp = (1 − ρ)² / ((1 − α)·v_term·dt)        [using h + g·τ = v_term·dt, exact]
    kd = (1 + α − 2ρ − h·kp) / g
    ```

    **Why this rule and not the previous one** — four properties of the rules, not of any outcome:
    (1) it is **discrete**, where `pd_analytic` was a continuous-time critical-damping condition
    with no notion of the control period at all, on a body where `dt/τ = 0.8` — the same reason
    `inv_ct` is not `inv_zoh`; (2) it is **exact**, so eigenvalue placement is a placement rather
    than an approximation; (3) it is **body-only and tempo-free**, which is what the error
    dynamics are — the old rule tied `ω` to the note duration, making the gains tempo-dependent
    by construction, and that is exactly what drove `kd = (2ω − 1/τ)/A` negative and clamped it
    to zero at every tempo slower than `T_note > 4πτ = 377 ms` (`kc1`: H = 32 and H = 16 both
    degenerate) while demanding 4× the gain at H = 8 and 16× too much at H = 2; (4) the free
    parameter is **one number with an operational meaning** — the per-control-step decay of the
    tracking error — so the search is a **1-D grid over `ρ`** and an edge says something specific
    instead of naming a corner of a 2-D multiplier box.
    `ρ = 0` is deadbeat and is deliberately not the rule: deadbeat has no robustness to model
    error and this world contains a command rotation the body model is denied by decision 5. Grid
    `ρ ∈ {0, 0.1, …, 0.9}`; beyond `ρ ≈ 0.75` the placement needs `kd < 0`, which is a valid
    placement and is **allowed and reported rather than clipped**, so the grid's top end is not
    silently a different rule. **If this rule also lands on edges it is reported and the grid is
    not widened a second time.**
    *Fit per (tempo, R) as asked, with a prediction written down first*: the brief poses the loop
    as carrying an effective delay of the read interval `R`. That holds when the estimate is HELD
    between reads; this executor **dead-reckons every step**, so with an exact forward model on
    the clean world the estimate equals the true state at every step whatever `R` is, and the
    loop carries no lag — predicting an **R-invariant** `ρ*`. The fit is therefore taken at both
    ends of the ladder (R = 1 and R = 16) and the prediction is measured. **`kcsmoke1` already
    falsifies it at one tempo** (H = 8: `ρ* = 0.8` at R = 1 against 0.4 at R = 16; H = 2
    invariant at 0.7), at a tiny `n_cal = 6`; `kc2` re-asks it at `n_cal = 16`. The per-R table
    is reported either way and the R = 1 fit is the one frozen for the treatment rows.

37. **The read ladder is in READS PER SPAN, not control steps.** `kc1`'s ladder was `R` in steps
    capped at 16, so at slow tempi its coarsest rung still cost 16 reads on a 256-step level-4
    span and the fewest-reads question was **never asked there at all** (decision 35's
    limitation). Reads-per-span `n ∈ {1, 2, 4, 8, 16}` gives the same set at every (tempo,
    level): `n = 1` is **launch only**, the once-at-launch end of decision 18's fork, and
    `R = ceil(span/n)`. `n` is capped at the span so a 2-step span cannot be asked for 16 reads,
    duplicate `R` values are run once, and the realised `reads_per_span` is reported beside every
    row.

## `kcsmoke1` / `kc2` — the re-posed Phase 3 smoke

| tag | what | wall |
|---|---|---|
| `kcsmoke1` | `--quick`, decisions 36 + 37 end to end | **73 s** |
| `kc2` | `t1`'s knobs, tempi {32, 16, 8, 2}, reads/span {1,2,4,8,16}, exact + fitted FM, oracle-ff control at {1,16}, P-E at L1/L4 × {1,4,16} × ε{0,.02,.05,.20}, `--fit-clean` | launched 2026-09-06, `fc-01M1TS2VYHJ074ZMZR9BKCYMFK` |

Three plumbing bugs fixed on the way, all in this node's own new code and none affecting a prior
run: a Modal flag whose name carried an uppercase letter (`read_ladder_fit_R` → `read_fit_r`);
a JSON cycle from storing the ρ grid on the grid's own winning entry; and `arm_table` emitting a
kinematic arm whose library had not been built when its fitted inverse was not requested.

## Run record — `kc2`, 2026-09-06: the re-posed Phase 3 smoke, 1656 s

Tempi {32, 16, 8, 2}, `app = fixed`, feedforward from the {32,16} linear inverse, decisions 36
(the discrete ρ rule) and 37 (reads per span). **P-T1 100 checks at max\|Δ\| = 0.000e+00**;
P-K out-of-region max 9.37e-05. Clean-world discriminator unchanged from `kc1` (clean fit
`c_v` 0.200000, 0.200000, `c_a` 0.008717, 0.008717, off-diag 9.90e-11, bias 3.68e-11).

### Decision 36's rule, and the verdict on its prediction

| H | ρ\* (R = 1) | kp | kd | e_piece | edge | ρ\* at R = 16 | R-invariant? |
|---|---|---|---|---|---|---|---|
| 32 | 0.8 | 0.6053 | −0.06294 | 0.0027 | no | 0.6 | **no** |
| 16 | 0.3 | 7.415 | 0.20775 | 0.0119 | no | 0.0 | **no** |
| 8 | **0.0** | 15.13 | 0.32083 | 0.0188 | **EDGE** | 0.8 | **no** |
| 2 | 0.7 | 1.362 | −0.00058 | 0.1671 | no | 0.7 | yes |

**One edge of ten grid points, at H = 8 (ρ\* = 0, deadbeat), against `kc1`'s four edges of
eight** — reported, and **the grid is not widened a second time**, as decision 36 said in
advance. The negative `kd` at H = 32 and H = 2 is the allowed-and-reported branch of the rule,
not a clip.
**The R-invariance prediction is FALSIFIED at three of four tempi.** I derived that with an exact
forward model on the clean world the estimate equals the true state at every step whatever `R`
is, so `ρ*` should not depend on `R`; it does, and by a lot (H = 8: 0.0 at R = 1 against 0.8 at
R = 16). The derivation is wrong somewhere and this tag does not say where. The frozen table is
the R = 1 fit, as stated in advance.

### (a) The fewest READS PER SPAN at which the corrected unit is in band

| H | L | exact FM | fitted FM | open `kfl` | open `kzsa` | `k1` aud_kzsa |
|---|---|---|---|---|---|---|
| 32 | 1 | **1** 0.0434 | **1** 0.0437 | 0.0522\* | 0.0399\* | 0.0408\* |
| 32 | 2 | **1** 0.0497 | **1** 0.0566 | 0.0997 | 0.0872 | 0.0399\* |
| 32 | 3 | **1** 0.0368 | **4** 0.0459 | 0.0604\* | 0.0803 | 0.0326\* |
| 32 | 4 | **1** 0.0239 | **1** 0.0578 | 0.0627 | 0.0839 | 0.0213\* |
| 16 | 1 | **1** 0.0540 | **1** 0.0530 | 0.1202 | 0.0776 | 0.0836 |
| 16 | 2 | **1** 0.0603 | **1** 0.0459 | 0.0694 | 0.0339\* | 0.0426\* |
| 16 | 3 | **1** 0.0568 | **1** 0.0611 | 0.0673 | 0.0498\* | 0.0426\* |
| 16 | 4 | **2** 0.0574 | **16** 0.0349 | 0.0821 | 0.0719 | 0.0446\* |
| 8 | 1 | **2** 0.0531 | **4** 0.0360 | 0.1420 | 0.1285 | 0.0868 |
| 8 | 2 | **4** 0.0354 | **1** 0.0570 | 0.0564\* | 0.0545\* | 0.0838 |
| 8 | 3 | **4** 0.0542 | **4** 0.0590 | 0.0555\* | 0.0587\* | 0.0529\* |
| 8 | 4 | **4** 0.0496 | **4** 0.0599 | 0.0574\* | 0.0498\* | 0.0550\* |
| 2 | 1–4 | **x** | **x** | 0.2950 · 0.1260 · 0.1398 · 0.1160 | 0.2919 · 0.0846 · 0.1167 · 0.0938 | 0.3498 · 0.2033 · 0.1210 · 0.0660 |

**With the ladder posed in reads per span, ONE read — the launch read, the once-at-launch end of
decision 18's fork — is enough at every level at H = 32 and at three of four levels at H = 16.**
The requirement rises with tempo: 1 read at 1×–2×, 2–4 reads at 4×, and **at 16× no read count on
the ladder puts any level in band** (best: L4 0.1073 at 16 reads/span). `kc1`'s ladder could not
express these rows at all.

Full ladder, exact FM, `e_piece` [realised reads/span] — H = 32 L4: 0.0239[1] 0.0210[2] 0.0236[4]
0.0226[8] 0.0215[16]; H = 8 L4: 0.0979[1] 0.0714[2] 0.0496[4] 0.0371[8] 0.0252[16]; H = 2 L4:
0.2566[1] 0.1822[2] 0.1343[4] 0.1160[8] 0.1073[16].

### (b) P-E — the ε at which the band is spent

| H | L | n = 1 (launch only) | n = 4 | n = 16 |
|---|---|---|---|---|
| 32 | 1 | 0.075 | **0.121** | **0.124** |
| 32 | 4 | 0.085 | **0.105** | **0.113** |
| 16 | 1 | 0.119 | **>0.2** | **>0.2** |
| 16 | 4 | already out | **>0.2** | **>0.2** |
| 8 | 1 | already out | **>0.2** | **>0.2** (n = 8) |
| 8 | 4 | already out | **>0.2** | **>0.2** |
| 2 | 1, 4 | already out | already out | already out |

Where the arm is in band at ε = 0, correction moves the bar from **0.075–0.085 to 0.105–0.124**
at H = 32 and from "already outside" to **beyond the whole ε = 0.20 ladder** at H = 16 and H = 8.
`accelerando`'s figure was that a 5% command error spends the band at the deep rung open-loop;
the launch-only rows here spend it at 7.5–8.5% and the corrected rows do not spend it at 20%.

### (c) The equal-feedback comparison

| H | reflex (fb) | reflex_ec | reflex_ec0 | corrected L4, e_piece (fb, reads/span) |
|---|---|---|---|---|
| 32 | 0.0037 (256) | 0.0252 | 0.0013 | 0.0215 (16) · 0.0226 (8) · 0.0236 (4) · 0.0210 (2) · **0.0239 (1)** |
| 16 | 0.0103 (128) | 0.0434 | 0.0027 | 0.0303 (16) · 0.0354 (8) · 0.0375 (4) · 0.0574 (2) · 0.0760 (1) |
| 8 | 0.0174 (64) | 0.0661 | 0.0079 | 0.0252 (16) · 0.0371 (8) · 0.0496 (4) · 0.0714 (2) · 0.0979 (1) |
| 2 | 0.0690 (16) | 0.1219 | 0.0320 | 0.1073 (16) · 0.1160 (8) · 0.1343 (4) · 0.1822 (2) · 0.2566 (1) |

At H = 32 the corrected L4 unit reads 0.0239 **at one feedback event** against `reflex_ec`'s
0.0252 at 256 and the reflex's 0.0037 at 256. At H = 8 it needs 16 reads to reach 0.0252, still
above `reflex_ec0`'s 0.0079. `reflex_ec0` beats every corrected row at every tempo.

**Oracle-feedforward control** (exact FM): H = 32 L4 0.0295 (1 read) → 0.0219 (16);
H = 8 L4 0.0980 → 0.0253; H = 2 L4 0.2454 → 0.0904 — within 10% of the fitted feedforward at
almost every cell, so the read ladder's shape is not a property of which inverse produced the
feedforward.

### The ledger

`reflex` 256 / 128 / 64 / 16 fb across H = 32 / 16 / 8 / 2; `key_kzsa_L4` 1 fb at every tempo;
the corrected L4 rows pay exactly their reads — 16 / 8 / 4 / 2 / 1 fb at H = 32 (one per read on
a 256-step span at R = 16 / 32 / 64 / 128 / 256) down to 16 / 8 / 4 / 2 / 1 at H = 2.
