# accelerando — file index and design record

**Up**: [`../../README.md`](../../README.md) (practice) · [`../../../README.md`](../../../README.md) (mjc)
**Contract**: [`SPEC.md`](SPEC.md) — the orchestrator's prompt, verbatim, as in `acappella/`,
`solo/` and `prestissimo/`.
**Donors (forked, never edited)**: [`../prestissimo/`](../prestissimo/FILES.md) — `piece.py` and
`world.py` are copy-forks; `prestissimo/` is untouched and `a0`/`a0s`/`a1`/`d5`/`d10`/`dH10`/
`a1g`/`dH10g` stay byte-reproducible. Through it: [`../../solo/`](../../solo/FILES.md) (the member
library, the parity gate, the trunk clone — Phase 2), [`../../acappella/`](../../acappella/README.md)
(the Δ ladder, the naive delay operator, the playability guard),
[`../../etude/`](../../etude/README.md) (the plant and the piece constants),
[`../../accompanist/presto/`](../../accompanist/presto/FILES.md) (the fast-piece design),
[`../../legato/`](../../legato/README.md) (nesting, the fusion control, F2's calibration discipline).
**RHM donors (read, not forked)**: [`rhm/practice/ratchet/`](../../../../rhm/practice/ratchet/README.md)
(`mine_from = chosen`, `mine_support`, `n_at_support`) ·
[`rhm/practice/conductor/`](../../../../rhm/practice/conductor/README.md) (the thermostat, dead
zones by null-ABBA, yoked clocks) — both Phase 3.

**No findings README yet — numbers get discussed before any writeup** (repo policy). This file is
the record of what was decided and why.

## Status

| phase | what | tags | state |
|---|---|---|---|
| 1 | the fast body and the tempo ladder, recordings only (`rec`, `rec_scaled`) | `tsmoke0`, `tsmoke1`, `tcal`, `t1` | **run 2026-09-05**; reduction + 6 figures |
| 2 | the program: a posture/slot/tempo-conditioned emitter, gated by parity on the plant | `psmoke`–`psmoke5`, `p1` | **run 2026-09-06**; reduction + 5 figures |
| 3 | the crank: mined nested tables under conductor's thermostat, tempo advance | `csmoke`, `c1` | **run 2026-09-06**; reduction written. Read as WITHIN-TEMPO PACING, not a tempo ratchet |

## Code files

| file | what |
|---|---|
| `piece.py` | pure python (a Modal *local* entrypoint imports it, and the local client has no numpy). `prestissimo/piece.py` forked verbatim, plus: `mass` as a `Piece` field reaching the plant through `dgp()`; the FAST BODY (`FAST_MASS`); `practice_sigma` (the donor's motor noise ported by matching realised velocity spread); `TEMPO_LADDER`, `DELTA_FIX`, `W_LISTEN`, `A_R_LADDER`, `U_TURN_MAX`; `Piece.tau/accel/v_terminal/turn_command`; `accel_piece(R, H)` and `a1g_piece()`. |
| `world.py` | `prestissimo/world.py` forked verbatim, plus `listener_err`/`_boxcar`/`schedule_ref` (the second grade, computed at every cell), `resample_cmds`/`resample_library` (the `rec_scaled` content), and the realised path always recorded. |
| `tempo.py` | the Phase 1 runner: gates P-F0/P-F1/P-F2/P-TAU/P-R/P-T/P-N/P-A/P-S, the per-tempo library builds, the tempo sweep at the fixed Δ, and the Δ instrument sweep. |
| `analyze_tempo.py` | Phase 1's reduction (pure python, local) + six figures (rendered remotely). |
| `t1_pins.py` | `t1`'s own numbers — R\*, the per-(tempo, Δ) gain table, 75 construction scores, 50 sweep numbers, its P-S gains and the config they were produced under. Generated once from `results/t1/tempo.json`, never hand-edited; gate P-T1 asserts every one at max\|Δ\| = 0. |
| `nets.py` | Phase 2's one learned component and the decider that plays it: `phase_feats` (the note-factorised phase basis), `ProgHead`, `build_prog`, `train_prog`, `make_prog_fn`, `ProgBuffer` + `split_code` (the self-imitation store and its deterministic train/hold split), `ref_errors` + `parity` (the gate), `ProgDecider`. |
| `program.py` | the Phase 2 runner: gates P-F0/P-G/P-T1/P-N/P-R/P-H, the per-tempo library rebuild, the self-imitation capture, the two heads, parity per slot per tempo against two references, the tempo sweep, and the Δ instrument at L1/L2. |
| `analyze_program.py` | Phase 2's reduction + five figures. |
| `analyze_crank.py` | Phase 3's reduction. Prints the scope caveat at the top of every report it writes. |
| `crank.py` | the Phase 3 runner: gates P-F0/P-T1/P-P/P-D/P-Y, the shared level-1 vocabulary per tempo, the miner (`mine_from = chosen` over adjacent committed slots played by solved traversals), `commit_level` (mined pairs welded and auditioned), and the crank loop under `conductor`'s policies — `anchor`, `outer_yield`, `outer_ledger` and a yoked-clock control per gauge arm. |

## The question Phase 1 answers

At which **tempo** does a level-ℓ unit stop being playable at a delay that never moves, and does
level ℓ+1 stay playable? Plus: does the fast body make seams carry information, which prestissimo
measured at ≈1.0 and which routing and trust in Phase 3 would have to select on?

## Decisions taken, with reasons

1. **The body is one changed number: `pusher_mass` 1.0 → 0.06 kg.** The plant is
   `a = (gear/m)·u − (c/m)·v`, so the body is `τ = m/c` and `A = gear/m`. Keeping `gear = 10` and
   `damping = 2.0` at the donor's values and moving only the mass gives **τ = 30 ms** (from
   500 ms) and **A = 167 m/s²** (from 10) while leaving the **terminal speed `gear/c` = 5 m/s
   exactly unchanged** — so practice variability, the actuator's speed ceiling and the arena are
   all the donor's, and the two numbers the node is about are the only ones that moved.
   *Why the bottom of the brief's 30–50 ms range:* a stored unit can execute a note blind only if
   τ < T, and the fastest note on the dyadic ladder is 48 ms. At τ = 50 ms that rung has τ > T and
   the band `τ < T < Δ` is empty again at exactly the tempo the node exists for; at τ = 30 ms the
   fastest note is 1.6 τ and the slowest is 25.6 τ. **Measured, not assumed** — gate P-TAU fits τ
   from a step response on both the fast body and the donor body.
2. **Δ = 5 control steps = 120 ms, fixed before any arm was read, held across all three phases.**
   A human proprioceptive loop is ~100–120 ms; prestissimo fixed its *note* at 5 steps for the
   same reason and this node fixes its *delay* there instead, because a pianist's Δ is what does
   not change while T does. The band `τ < T < Δ` is then occupied by H = 4 (96 ms) and H = 2
   (48 ms); H = 8/16/32 are the rungs where feel still has time to work. **The Δ sweep survives
   only as the instrument** that decomposes a result into its feedback and execution halves, run
   at each tempo on the incumbent plus the two ends of the level ladder, under both approach
   modes.
3. **The tempo ladder is dyadic and has five rungs: H ∈ {32, 16, 8, 4, 2} = 768/384/192/96/48 ms.**
   Dyadic because then a level-(ℓ+1) unit at tempo 2× has the same wall-clock span as a level-ℓ
   unit at tempo 1×, so the ladder's rungs and the listener's window commensurate.
   **The slowest rung was added after a measurement, not by taste**: the `tsmoke0` probe
   (2026-09-05, gates off, 2 tempi, 63 s) put the incumbent at `e_piece = 0.153` against a
   `0.053` band already at H = 8 under the fixed 120 ms delay — i.e. an era ladder starting at
   H = 16 might have had no era 0 at all, which is prestissimo `a0`'s failure mode reproduced by
   design. The ladder is extended downward until the incumbent is competent rather than the delay
   being shrunk, because the delay is the constant the node is built around. Cost: the H = 32 rung
   is ~half the node's wall clock.
4. **`k_app = 3` approach notes, prestissimo's, unchanged.** `k0·H > Δ` at every tempo (6 > 5 at
   the fastest), so the read at drilled seam 0 is **never clamped at the reset state** — the
   aliasing prestissimo's diagnostic round located at `k0·H = 15 < 16`. A fourth note would buy
   nothing and cost 9% of every traversal on a five-rung ladder.
5. **The piece is prestissimo's irregular octagon, geometry unchanged** — same angles, same radius
   factors, same command-rotation region (σ = 0.2 × mean leg, φ = 1.2 rad, drilled segment 3),
   same lead-in construction. Only `R` (size) is calibrated and only `H` (tempo) is swept. Holding
   the geometry fixed is what makes this node's P-S readout comparable with prestissimo's 1.03×
   and `d10`'s 1.25×: the body and the tempo move, the piece does not. P-S is measured per tempo
   and per level rather than assumed, as the brief requires.
6. **Practice variability is PORTED, not carried.** `EXPLORE_SIGMA = 0.25` is a donor constant
   (P-F0 reads it out of `etude/etude.py`) and it is a *command* noise; what practice variability
   is, though, is the spread of realised motion, and on `v_{n+1} = a v_n + (1−a) v_term u_n` that
   is `σ · v_term · sqrt((1−a)/(1+a))`. `v_term` is unchanged between the two bodies, so matching
   the donor's realised velocity spread is a pure rescaling of σ: **0.25 → 0.0628**. Carrying 0.25
   unchanged would make practice ~4× more variable in realised motion than the donor's, which on
   legs of 0.12 m would leave the rendition pool with nothing recognisable in it. The rule involves
   **the two bodies only** — no piece, no tempo, no arm — so it cannot be moved by an outcome, and
   it returns exactly 0.25 on the donor body, which is what keeps P-F1 and P-F2 exact.
7. **The R calibration: three guards, all naming only the incumbent or the apparatus, all fixed
   before the grid was read** (legato F2, prestissimo decision 8).
   - *feasibility* — `u_turn_max(H_fast) ≤ 0.9`. `u_turn` is the **schedule's own** peak actuator
     demand: at a seam the schedule asks for `Δv` within one note (`Δv/T`) on top of holding speed
     against drag (`|v|/τ`), and the command that buys both is `(Δv/T + |v|/τ)/A`. Above 1 the
     figure at that tempo is outside `|u| ≤ 1` and *no* controller, stored or felt, can play it —
     the tempo axis would be measuring the actuator's ceiling rather than the consumption of
     feedback. It is an **apparatus** quantity computed by the experimenter at calibration time,
     exactly like the do-nothing floor and the band; no arm reads it, nothing is trained on it,
     and it predicts no state. The estimate is conservative twice over (it adds turn and drag
     *magnitudes* instead of composing the vectors, and reads `|u|₂` against a limit that is
     really `|u|∞ ≤ 1`, a square), so the only open question is how much headroom to leave for
     correction; 0.9 leaves 10%, and at 1.0 the design point would be playable only by an
     open-loop optimum. Binding value: `u_turn(H=2) = 5.43 R`, so the guard binds at R = 0.166.
   - *competence* — `e_reflex(H_slow, Δ_fix, gains re-fit on the CLEAN world) ≤ band(R)`: there
     must be an era 0 to leave.
   - *demand* — `e_reflex(H_fast, Δ_fix) > band(R)`: the tempo axis must have bite (offbook d0's
     measured failure was "playable for everyone, then unplayable for everyone").
   - `R* =` the **largest** R passing all three. Fallback: largest R passing feasibility and
     competence, said so in the record.
   Only the two **guard tempi** are gridded in P-T; every rule names the slowest or the fastest
   rung and nothing else, and the incumbent's number at the intermediate tempi is re-measured by
   the main run's own per-tempo re-fit and reported there. On a five-rung ladder whose slowest
   note is 768 ms, gridding all five would double the node's wall clock to produce numbers the
   run already produces.
8. **Two grades, both pre-fixed, both reported at every cell, neither privileged.**
   - `e_piece` — prestissimo's per-waypoint drilled error with the `max(½ mean leg, ref_play ×
     mean_leg / 0.8)` band and pass fractions at ¼ / ⅓ / ½ mean leg. Every prior number in this
     arc is in these units.
   - `e_listen` — **the listener's clock**: the executed path and the schedule's own reference are
     low-passed with the *same* boxcar of fixed wall-clock width (16 control steps = **384 ms**)
     and compared at every drilled control step. At H = 16 the window is one note and the grade is
     a mildly smoothed tracking error; at H = 2 it spans eight notes, so within-phrase wiggle and
     timing slop are forgiven while the phrase's position is not. Same units, same three widths.
   *Which arms each grade favours, stated before the grid*: the per-waypoint grade rewards arms
   that resolve each waypoint — per-seam auditions, and feel wherever feel still has time; the
   listener's grade rewards arms whose gross path is right and should favour whole-figure units at
   fast tempi. Where they disagree, the disagreement is the readout.
   *Rejected alternative, recorded*: grading only at window **endpoints**. The figure is closed, so
   at the fastest tempo there is exactly one window whose endpoint is the start vertex — the
   do-nothing policy would score ≈ 0. The sliding low-pass keeps the sample count and a real
   do-nothing floor at every tempo, and **both floors are reported at every tempo** so the grade's
   non-degeneracy is on the record rather than argued (measured on `tsmoke0`: do-nothing 0.427
   under `e_piece` and 0.418–0.424 under `e_listen`).
9. **`W_LISTEN = 16` steps is a constant of the ROOM, not of the score.** It is the note length at
   H = 16 — the rung where the two grades are most nearly the same instrument — and roughly the
   width of the auditory present. It did not change when the ladder gained the H = 32 rung; there
   the window is half a note.
10. **Three content sources in Phase 1, all in every comparison.**
    - `rec` — a library harvested **at its own tempo**, built once at Δ = 0 with the harvest gains
      frozen at that tempo's Δ = 0 fit (acappella's discipline, prestissimo decision 10). One
      library per tempo, prestissimo's construction unchanged.
    - `scl0` — the **slowest** tempo's library, time-resampled note-by-note to the asked tempo
      (linear interpolation at centre-aligned positions). "Play the tape faster."
    - `scl2` — the same with amplitude × `(H_src/H_dst)²`, the dimensional-analysis correction for
      an inertial body. On a body with drag the exponent that holds a *speed* is 1 and the one
      that buys an *acceleration* is 2, and a real traversal needs both, so neither is right —
      which is the measured question, and both are free.
    Neither scaled arm reads a state, so neither is an inverse model of the body (that would be
    `u = (m a + c v)/gear`, computed from the recorded trajectory — available, and deliberately
    not used, because it is exactly the `f(s,u)` this arc forbids).
    Resampling is **note-wise**, so a level-ℓ member's resampled tape is exactly the weld of its
    parents' resampled tapes and gate P-N holds on the scaled libraries too. The scaled libraries
    carry the **source tempo's keys and construction scores**, which are not valid at the
    destination: `key` mode therefore selects on stale keys and `audit` mode on the plant, and both
    are reported. Pretending otherwise would be the scaled arm quietly consuming information it
    has not paid for.
11. **`app = fixed` is the arm of record, `app = arm` is reported beside it** (prestissimo decision
    20, and the brief's instruction). Under `fixed` the lead-in reads at Δ = 0 *and* plays the
    Δ = 0 gains, so it is one pre-computed plan (presto decision 3) and the hand-over into drilled
    seam 0 does not move with Δ; it is also exactly the hand-over the library was harvested from,
    since the harvest runs at Δ = 0 with those gains. `zero` is **not** carried: prestissimo
    measured it to be a gain effect wearing a read effect's name.
12. **Two fork-fidelity gates, not one.** P-F1 is prestissimo's, verbatim: the fork runs the DONOR
    SQUARE and reproduces `acappella/b1`'s library build and its Δ = 0 / Δ = 8 rows at
    max|Δ| = 0.000e+00 with the gains re-derived. But the donor square has no approach, no level
    ladder, no weld and no fast piece, so it cannot see most of this fork's surface. **P-F2** runs
    `prestissimo/a1g`'s piece and configuration — R = 0.20, k_app = 3, H = 5, mass 1, damping 2,
    a1g's own pinned per-Δ gains — through this node's `build_ladder` and `LadderDecider` and
    asserts its 8 level-1 and 7 level-(ℓ>1) construction scores, its nesting identity, its `e_app`
    and 12 of its `app=fixed` sweep rows at max|Δ| = 0.000e+00. Like P-F1 it pins the donor's own
    configuration inside the gate, so it is defined at every treatment config including `--quick`.
13. **P-A is measured on `app = arm`,** as prestissimo's was, plus two structural checks
    (`read_grows`: the read at drilled seam 0 is stale at every Δ > 0; `never_clamped`: `k0·H > Δ`
    at every tempo). Under `fixed` the deepest *key* arm is Δ-invariant **by construction**
    whenever its slot choice is stable — the lead-in does not move and one decision covers the
    figure — which is a property of the instrument and not the artifact acappella finding 5 named.
    Both modes are reported at both ends of the level ladder.
15. **The incumbent's gain grid is PORTED to the body, exactly as the practice noise is.** The
    reflex law's `u` drives an acceleration `A·u` with `A = gear/m`, so every closed-loop property
    the grid is meant to span is a property of `A·kp` (stiffness, hence bandwidth `~sqrt(A·kp)`
    and ramp error `v/(v_term·kp)`) and `A·kd` (the velocity-loop gain) — never of `kp` and `kd`
    alone. `A` is 16.7× larger here, and the consequence is not cosmetic: the delayed velocity
    loop is stable only for `A·kd·Δ ≲ 1`, i.e. **kd ≲ 0.05** at Δ = 120 ms, while prestissimo's
    grid *floor* is kd = 0.25. **Measured** on `tsmoke1` with the smoke's coarse grid (kd floor
    1.0): the incumbent scored **0.2425 at H = 16 against a 0.0534 band and a 0.3334 do-nothing
    floor** — barely better than not moving, at a tempo where the delay is a third of a note. The
    unscaled grid cannot express a stable controller on this body, so a run using it would have
    measured the grid's floor and called it the delay (prestissimo flag 1's disease, in the other
    direction). Both grids are therefore multiplied by `m/m_donor = 0.06`, which holds `A·kp` at
    5–1600 s⁻² and `A·kd` at 2.5–160 s⁻¹ — prestissimo's own spans, exactly — and brackets both
    stability limits (kd ≲ 0.05 falls between rungs 0.03 and 0.06; kp ≲ 0.42 between 0.3 and 0.6).
    The rule involves the two bodies only and is the identity on the donor, so P-F1 and P-F2 stay
    exact. Raw gains and `A·kp` / `A·kd` are both reported at every cell.
16. **The P-T fallback, revised before the grid was read, when competence fails at every R.** The
    first draft took the smallest R. That is wrong here for a measured reason: `e_reflex / band`
    is only weakly R-dependent and what dependence it has runs the *wrong way* for shrinking
    (`tsmoke1`: 4.5 at R = 0.14 against 3.1 at R = 0.20), so shrinking the figure makes the piece
    slower and no more playable — prestissimo decision 8's reversal in this node's units.
    Competence failing at every R is a statement about **the delay and the body**, not about the
    figure's size. The revised fallback keeps the **largest R passing feasibility and demand** and
    records that era 0 does not exist on the ladder.

14. **Phase 1 has no learned component at all** — no trunk, no π, no torch, and no forward model
    anywhere. The reflex law, executed content, the resampler, and the plant as the only oracle.
    The program head enters in Phase 2.

## Gates

| gate | what it asserts | result |
|---|---|---|
| **P-F0** | the 13 donor constants, `ast`-read out of `etude/etude.py`, never imported | pending `t1` |
| **P-F1** | the fork, run on the **donor square**, reproduces `acappella/b1` (14 checks) | `tsmoke1`: **14 checks, max\|Δ\| = 0.000e+00, gains_match=True** — pass |
| **P-F2** | the fork, run on **prestissimo `a1g`'s fast piece**, reproduces its construction scores, nesting and `app=fixed` sweep rows | `tsmoke1`: **16 checks, max\|Δ\| = 0.000e+00**; nesting 168/168 spelled, 0 bad, exec 1.1047e-07 (a1g's own recorded value) — pass |
| **P-TAU** | τ measured from a step response matches `m/c` on the fast body and the donor body | `tsmoke0`: 30.00 ms vs 30.00 (rel 2e-4); donor 499.9 vs 500.0 (rel 2e-4); v∞ 4.9998 vs 5.00 — **pass** |
| **P-R** | the resampler is the identity at `s = 1` | `tsmoke0`: 0.000e+00 over 120 members — **pass** |
| **P-N** | the nesting op: spelling exact, execution within the float32 hand-over round-trip | `tsmoke0`: 42/42 spelled, 0 bad, 5.3e-08 / 5.8e-08 — **pass** |
| **P-T** | the R calibration, incumbent + apparatus only, rules fixed before the grid | `tcal`/`t1`: **R\* = 0.160**, all three guards, binding guard = feasibility |
| **P-A** | the approach removed acappella finding 5's artifact, at every tempo | `t1`: spread half **passes** (`arm` key_L4 1.45× / 1.91× / 12.39× / 9.84× / 41.48×), read stale at every Δ>0 — but the `never_clamped` half **FAILS**; see limitation 5 |
| **P-S** | seam information per level per tempo | instrument |

## Arms (Phase 1)

| arm | decision rule | groundings |
|---|---|---|
| `reflex` | the reflex law, gains re-fit per tempo (and per Δ in the instrument) on the clean world | 0 |
| `reflex_ol` | INSTRUMENT: the same law read once at the drilled launch, replayed open-loop | 0 |
| `key_Lℓ` / `aud_Lℓ` | `rec` (this tempo's library), frozen posture key / plant audition | 0 / `n_slot × m_member` per decision |
| `key_all` / `aud_all` | `rec`, every rung legal at every aligned seam | as above |
| `key_s0_Lℓ` / `aud_s0_Lℓ` | `scl0` (slow tape, time-resampled) | as above |
| `key_s2_Lℓ` / `aud_s2_Lℓ` | `scl2` (slow tape, resampled × s²) | as above |

Every arm plays the same shared approach at the same Δ; the approach ledger is subtracted.

## Smokes and sizing

| tag | what | wall |
|---|---|---|
| `tsmoke0` | `--quick --no-pf1 --no-pf2`, 2 tempi, plumbing + P-TAU/P-R/P-N/P-S end to end | 63 s |
| `tsmoke1` | `--quick`, all gates including P-F1 and P-F2 (P-F1 116 s, P-F2 55 s of it) | 265 s |
| `psmoke`–`psmoke5` | Phase 2's five smokes: plumbing; the phase-basis rebuild; capacity and GPU; P-E and the held-out mse; the `progall` control | 102–161 s each |
| `p1` | Phase 2 of record: 5 tempi, 3 heads, seed 0, 16 CPUs + L4 | **2920 s** (libraries 622 · capture 76 · training 1210 · sweep 388 · Δ instrument 540 · parity 42 · P-G 25 · P-E 1) |
| `csmoke` | Phase 3 `--quick`: the crank end to end, P-P 12/12, P-Y 0.000e+00, mining and committing live | 134 s |
| `c1` | Phase 3, 5 tempi, 5 arms, 80 cycles, seed 0, `fc-01M1T2GC9TYDQD5NMFRHR14FC3` | launched detached 2026-09-06 00:39. **Read as a data point on within-tempo pacing, not as a tempo ratchet**: with a per-tempo table and the program never executed, a tempo advance is a change of piece rather than a level crossing, so `c1` tests whether one thermostat paces commit/hold inside a tempo — not whether a ratchet survives the era knob. |
| `fsmoke_a` | fix round (a): capture at the deployment condition, `t1` knobs, tempi 16/8/2, no Δ instrument | **721 s**, P-T1 75 checks at 0.000e+00 |
| `fsmoke_a2` | decision 35: `--capture-at deploy --posture obs` — train and deploy posture the same object | **750 s**, P-T1 75 checks at 0.000e+00; the reading's first branch fires |
| `fsmoke_b` | fix round (b): the metronome ladder H ∈ {8,7,6,5,4}, gains re-fit, `--posture obs --capture-at deploy` | **1091 s**, P-T1 50 checks at 0.000e+00; the pre-fixed statistic is undefined (no level exceeds 0.5 open at any rung) |
| `tcal` | `--cal-only --no-pf1 --no-pf2` at the FULL gain resolution, the body-scaled grid, 5 R × the two guard tempi — prestissimo's `pcal` step | 216 s (P-T alone) |
| `t1` | Phase 1 of record: 5 tempi, 5 R, seed 0, 16 CPUs, no GPU | **2072 s** (P-F1 80 · P-F2 33 · P-T 172 · libraries 913 · sweep 387 · Δ instrument 402 · P-S 51) |

## P-T, as measured (`tcal`, 2026-09-05, before any treatment number existed)

The calibration was run as its own tag, `--cal-only`, exactly as prestissimo ran `pcal`: the rule
is fixed, the grid is read once, and no committed arm's number exists yet, so the rule cannot be
moved by an outcome. Full gain resolution (9 kp × 7 kd, body-scaled), n_cal = 16, Δ = 5, the two
guard tempi only.

| R | mean leg | band | H32: v | u_turn | e_reflex | e_listen | H2: v | u_turn | e_reflex | e_listen | do-nothing | range | feas | comp | demand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.08 | 0.0611 | 0.0305 | 0.08 | 0.02 | 0.0019 | 0.0023 | 1.27 | 0.43 | 0.0550 | 0.0504 | 0.193 | 29.3× | ✓ | ✓ | ✓ |
| 0.11 | 0.0840 | 0.0420 | 0.11 | 0.03 | 0.0026 | 0.0032 | 1.75 | 0.60 | 0.0732 | 0.0661 | 0.259 | 28.4× | ✓ | ✓ | ✓ |
| 0.14 | 0.1069 | 0.0534 | 0.14 | 0.03 | 0.0033 | 0.0041 | 2.23 | 0.76 | 0.0916 | 0.0819 | 0.324 | 27.9× | ✓ | ✓ | ✓ |
| **0.16** | **0.1222** | **0.0611** | 0.16 | 0.04 | **0.0038** | 0.0047 | 2.55 | 0.87 | **0.1039** | 0.0925 | 0.368 | 27.7× | ✓ | ✓ | ✓ |
| 0.20 | 0.1527 | 0.0764 | 0.20 | 0.05 | 0.0047 | 0.0059 | 3.18 | **1.09** | 0.1285 | 0.1138 | 0.456 | 27.4× | **✗** | ✓ | ✓ |

**R\* = 0.160** — all three guards; the largest figure the incumbent can still play. The binding
guard is **feasibility**: at R = 0.20 the schedule itself asks for |u| = 1.09 at the fastest
tempo, i.e. more than the actuator has. Competence and demand pass at every R on the ladder, with
the incumbent **16× inside the band at the slowest tempo (0.0038 vs 0.0611)** and **1.7× outside
it at the fastest (0.1039)**, against a do-nothing floor of 0.368. The usable range —
`e_reflex(H_fast)/e_reflex(H_slow)` — is **27.7×**, against prestissimo's 9.5× on its Δ axis.

The same table with the *unscaled* (prestissimo-numbered) grid is what `tsmoke1` measured: the
incumbent at 0.2425 at H = 16, i.e. 4.5× outside the band and barely inside the do-nothing floor.
The whole difference between "no era 0 exists" and "era 0 is 16× inside the band" is decision 15.

## Flags carried forward from the smoke (unsmoothed, to be re-read at full scale)

1. **`app = arm` blows the lead-in up on this body.** `tsmoke0` hand-over under `arm`: |x − V0|
   = 3.26 legs at H = 8 and 5.29 legs at H = 2, with |v| = 5.9 m/s (the diagonal terminal speed)
   against a schedule tempo of 0.46 / 1.84 m/s, and a read-error in velocity of 12.4 m/s. On a
   body with A = 167 m/s² a delayed stiff PD oscillates violently; the gain grid still picks it
   because it is fit on the **drilled** error only. Under `fixed` the same hand-over is 0.18 / 0.25
   legs. This is why `fixed` is the arm of record and why both columns are reported.
2. **Even under `fixed` the hand-over speed overshoots the schedule** (|v| 1.93 vs 0.46 at H = 8).
   The Δ = 0 gain fit minimises waypoint *position* error and is free to arrive fast on an agile
   body — `legato` l1/F3's "hit waypoints while ignoring hand-over velocity", from the controller
   side. It is arm-neutral (shared lead-in, and the library is harvested from exactly this
   distribution) and `e_app` is reported at every cell.
3. **Seam information is present on the fast body, where prestissimo measured ≈1.0.** `tsmoke0`
   at H = 8, Δ = 0 (3 slots only): L1 **1.72×**, L2 1.25×, L3 1.07×, L4 1.11×, falling to 1.36× /
   1.14× / 1.15× / 1.11× at Δ = 5 and to 1.27× / 1.13× / 1.06× / 1.03× at H = 2, Δ = 0. Against
   prestissimo's 1.03× at seg/τ = 0.24 and `d10`'s 1.25× at 1.20. To be re-read at 6 slots.
4. **The incumbent's ceiling on this body is a stiffness-vs-delay trade, and it is analytic.**
   Tracking a ramp at speed `v` with the reflex law leaves a steady-state lag `v/(v_term·kp)`, so
   staying inside the band needs `kp ≥ v/(v_term·band)`; the delayed loop is stable only up to
   `ω_c ≈ 1/Δ`, i.e. `kp ≲ 1/(A·Δ²)`. On this body those two meet at a speed of roughly
   `v_term·band/(A·Δ²)`, which is why the ladder was extended downward rather than the delay
   shrunk. The run measures where they actually cross; the algebra is recorded here so the
   measurement can be read against it.
5. **The two grades already disagree at smoke scale.** `tsmoke0`: the incumbent fails the band at
   H = 8 under `e_piece` (0.1475 > 0.0534) but passes under `e_listen` (0.0409), and fails both at
   H = 2. So "the tempo at which the reflex fails" is 8 on one grade and 2 on the other.

## Run record — Phase 1 (`t1`), 2026-09-05

16 CPUs, no GPU, seed 0, 2072 s. R\* = 0.160 (mean leg 0.1222 m, band 0.0611, do-nothing 0.360),
five tempi at Δ = 5 (120 ms), both approach modes, `fixed` the arm of record.

**Gates.** P-F0 13/13 · **P-F1 14 checks at max|Δ| = 0.000e+00, `gains_match=True`** ·
**P-F2 16 checks at max|Δ| = 0.000e+00**, nesting 168/168 spelled, 0 bad, exec 1.1047e-07 (a1g's
own recorded value) · **P-TAU τ_fit 29.995 ms vs m/c 30.000 (rel 1.7e-4), donor 499.9 vs 500.0,
v∞ 4.9998 vs 5.00** · P-R 0.000e+00 over 405 members · P-N 168/168 at every tempo
(5.0e-08–6.0e-08) · **P-A partially fails, see limitation 5**.

### The headline, uninterpreted: the tempo axis orders feel against stored content

`app=fixed`, Δ = 5, the best `rec` arm against the per-tempo re-fit reflex, and the same on the
listener's clock. Feedback events per drilled traversal beneath.

| | H = 32 (768 ms) | H = 16 (384) | H = 8 (192) | H = 4 (96) | H = 2 (48) |
|---|---|---|---|---|---|
| reflex e_piece | **0.0037**\* | 0.0103\* | 0.0174\* | 0.0267\* | 0.0690 |
| best rec e_piece | 0.0175\* (key_L3) | 0.0087\* (key_L4) | 0.0075\* (aud_L3) | 0.0090\* (key_L4) | **0.0069\*** (aud_L4) |
| ratio reflex/best | 0.21× | **1.18×** | 2.32× | 2.96× | **10.02×** |
| same, e_listen | 0.23× | 1.15× | 1.53× | 2.15× | 9.24× |
| reflex fb events | 256 | 128 | 64 | 32 | 16 |
| L4 fb events | 1 | 1 | 1 | 1 | 1 |
| ol/cl (`fixed`) | 474.5 | 128.1 | 45.6 | 15.1 | 2.18 |

(\* = inside the 0.0611 band.) The crossing is at **H = 16** and the advantage is monotone in
tempo from there. The incumbent is never a tape by the pre-fixed ≤ 1.10 reading, at any tempo, in
either mode.

### The era table (`fixed`, Δ = 5): era₁ exists, which it never did in prestissimo

At H = 2 the incumbent leaves the ½-leg band (0.0690 > 0.0611, pass fraction 0.06) while **every**
`rec` level is inside it: key_L1 0.0399, key_L2 0.0181, key_L3 0.0123, **key_L4 0.0074**, all at
pass 1.00 under both grades. `piece`: key era₁ = 2, era₃ = 32; aud era₁ = 2, era₃ = 4; era₂ and
era₄ null. Under `app=arm` era₁ is **null** — at H = 2 the shared lead-in is already 1.09 legs off
with |v| 2.48 against a 2.10 schedule, and no committed arm is in band (0.134–0.156).
**The two grades disagree about era 0**: on `e_listen` the incumbent never leaves the band (0.0472
at H = 2, pass 1.00), so `reflex_fails` is null and era₁ does not exist on that grade.

Ordering margins on e_piece (`fixed`, positive = the deeper rung wins), H = 32 → 2:
key m12 −0.0280/+0.0361/+0.0036/+0.0028/+0.0218 · m23 +0.0667/+0.0047/+0.0234/+0.0220/+0.0058 ·
m34 +0.0000/+0.0055/+0.0126/+0.0042/+0.0048; aud m34 −0.0004/+0.0371/−0.0048/+0.0079/+0.0265.
m34 is positive at 4 of 5 tempi under `key` and 3 of 5 under `aud`; m23 at 5 of 5 under `key`.

### A recording does not survive a tempo change

`scl0` / `scl2` against `rec` at the same tempo, audition, ratio (1.00 at H = 32 is the `s = 1`
identity that gate P-R checks):

| level | H = 32 | H = 16 | H = 8 | H = 4 | H = 2 |
|---|---|---|---|---|---|
| L1 `scl0` | 1.00× | 1.81× | 3.51× | 1.43× | 4.79× |
| L2 `scl0` | 1.00× | 1.55× | 4.88× | 2.39× | 7.79× |
| L3 `scl0` | 1.00× | 1.85× | 23.20× | 9.70× | 7.06× |
| **L4 `scl0`** | 1.00× | **10.22×** | **11.92×** | **17.67×** | **35.58×** |
| L4 `scl2` | 1.00× | 20.02× | 34.99× | 20.98× | 59.97× |

No scaled arm is inside the band at any tempo but its source, at any width, under either grade.
The amplitude-corrected variant (`scl2`, × s²) is 2–2.6× **worse** than the plain resample
everywhere — the inertial correction over-drives a drag-dominated body, as decision 10 said it
might. One exception worth its own line: at H = 2 under the **listener's** grade `aud_s0` reads
0.084–0.102 against the same-tempo recording's 0.140–0.144 and the incumbent's 0.0472 — i.e. the
slow tape played fast keeps the phrase in roughly the right place while missing every waypoint.

### Seam information (P-S), per tempo

| H | Δ | L1 | L2 | L3 | L4 | distinct argmins (L1) |
|---|---|---|---|---|---|---|
| 32/16/8 | 0 and 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 / 6 |
| **4** | **0** | **1.978** | **1.291** | **1.198** | **1.149** | **5.0 / 6** |
| 4 | 5 | 1.143 | 1.080 | 1.106 | 1.149 | 2.5 / 6 |
| 2 | 0 and 5 | 1.000 | 1.000 | 1.000 | 1.001 | 1.0 / 6 |

Against prestissimo's 1.03× at Δ = 0 and 1.00× under delay, and `d10`'s 1.25×. At four of five
tempi the six level-1 slots score identically at every performer state and one slot is the argmin
everywhere — the k-means partition over content found nothing to partition, which with the ported
practice noise (σ = 0.063) and a harvest controller sitting at the grid top (kp = 9.6 at every
tempo) is a statement about the homogeneity of the rendition pool, not about the seam. H = 4 is
the exception and it is a large one.

### The Δ instrument, per tempo — does a longer delay open more rungs?

The reflex's own e_piece under `fixed`, with `X` marking outside the band:

| H | Δ=0 | Δ=2 | Δ=5 | Δ=8 | Δ=12 |
|---|---|---|---|---|---|
| 32 | 0.0005 | 0.0023 | 0.0037 | 0.0097 | 0.0063 |
| 16 | 0.0012 | 0.0059 | 0.0103 | 0.0232 | 0.0146 |
| **8** | 0.0025 | 0.0123 | 0.0174 | 0.0397 | 0.0264 |
| **4** | 0.0040 | 0.0206 | 0.0267 | 0.0458 | **0.0720 X** |
| 2 | 0.0112 | 0.0427 | **0.0690 X** | **0.1487 X** | **0.1710 X** |

**At H = 8 the incumbent never leaves the band over the whole Δ ladder** (max 0.0397 at Δ = 8,
non-monotone because the gains are re-fit per Δ), while aud_L4 sits flat at 0.0123 and key_L4 at
0.0185 from Δ = 5 on — so a longer delay does not open a rung there. **At H = 4 it leaves at
Δ = 12** (0.0720), with every level still inside: key_L4 0.0090, aud_L4 0.0104, key_L1 0.0283,
aud_L1 0.0554. Under `fixed` the deepest arms are flat in Δ by construction (one decision from a
lead-in that does not move); the L1 arms are not (aud_L1 at H = 32: 0.0174 → 0.0969 over the
ladder), which is where the Δ axis still reads.

### The hand-over log

| app | H | \|x−V0\| (legs) | \|v\| | v_sched | read err pos / vel | read@ | e_app |
|---|---|---|---|---|---|---|---|
| fixed | 32 | 0.0005 (0.00) | 0.159 | 0.131 | 0.0191 / 0.000 | 91/96 | 0.0005 |
| fixed | 16 | 0.0010 (0.01) | 0.318 | 0.262 | 0.0382 / 0.000 | 43/48 | 0.0009 |
| fixed | 8 | 0.0020 (0.02) | 0.637 | 0.525 | 0.0763 / 0.008 | 19/24 | 0.0024 |
| fixed | 4 | 0.0036 (0.03) | 0.959 | 1.049 | 0.1525 / 0.938 | 7/12 | 0.0051 |
| fixed | 2 | 0.0081 (0.07) | 2.547 | 2.099 | 0.3415 / 0.803 | 1/6 | 0.0094 |
| arm | 2 | 0.1334 (1.09) | 2.483 | 2.099 | 0.2274 / 1.530 | 1/6 | 0.1139 |

With the body-scaled gain grid the lead-in is clean at every tempo under `fixed` (≤ 0.07 legs,
`e_app` ≤ 0.0094) — smoke flag 2, the 4× velocity overshoot, was the unscaled grid and is gone.
The read error at drilled seam 0 grows with tempo exactly as `v·Δ` (0.019 m at H = 32 to 0.342 m —
2.8 legs — at H = 2), which is the delay doing its work.

### Acquisition

30 336 groundings per performer per tempo, 71 453 s priced, for a 105-slot ladder (L1 56, L2 28,
L3 14, L4 7) at each of five tempi — the whole build is 913 s wall at 16 processes, 44% of the
run. Harvest gains land at the kp grid **top** (9.6) at every tempo; the gain-grid edge flags fire
at (H, Δ) = (32,0), (32,2), (16,0), (16,2), (8,0), (4,0), (2,0) and **nowhere at the headline
Δ = 5**.

## Phase 2 — decisions taken, with reasons

17. **The head is a PHASE-CONDITIONED emitter, not a fixed-width vector.** solo's `SpanHead`
    emitted `max_span × 2` values at once, which is a recording with a learned index: one output
    slot per control step, hence one output shape per tempo. A level-ℓ unit here is `2^(ℓ−1)·H`
    steps and `H` is the era knob, so a fixed-width head could not be asked the same question at
    two tempi. `ProgHead` emits `u(φ)` for a scalar phase and is called once per step, so the
    *same* head answers at every tempo and the sample count is the tempo's business. That is the
    program/recording distinction made structural rather than asserted.
18. **The phase basis is factorised the way the piece is — and the smoke forced it.** The first
    build used one Fourier basis over the SPAN phase; `psmoke` (6 epochs) measured mse 0.0293
    against a target mean-square of 0.0324 and `prog_L4` came back **20× worse than the tape it
    was cloning**. A level-4 unit is eight notes long, so a span-only basis needs frequencies past
    the note rate before it can represent one turn per note. The basis is now
    `[sin/cos(φ_note)×n_freq, φ_note, sin/cos(φ_span)×n_freq, φ_span, note_frac, log2(H),
    log2(span)]` — a within-note curve shared across notes (where the body's own turn-and-settle
    shape lives), a span phase (which leg), and the note index.
19. **No trunk read, and no score.** solo fed its span head the behaviour-cloned trunk's hidden
    state; that question was interference, this one is transfer, so Phase 2 has exactly ONE
    learned component and it takes only what the brief names — posture, slot, tempo, phase. The
    schedule's own target at each phase is deliberately **not** an input, although the reflex has
    it and it would make tempo-generalisation far easier: giving the head the score would confound
    "the program transfers" with "the score transfers", and the whole point is whether a *stored
    unit* survives a tempo change.
20. **The practiced set is a measured axis, not a cell.** `prog` practices H ∈ {32, 16, 8} and
    extrapolates one and two rungs; `prog2` practices {32, 16} and extrapolates three. Both are
    minted from independent generators and trained on the same buffer, so the only difference is
    which tempi were legal targets, and the extrapolation *distance* is read off the pair.
21. **Launch states are captured at EVERY tempo; targets only at the practiced ones.** The parity
    gate needs held-out launch states wherever it is asked, and a learner playing at a new tempo
    necessarily has them — arriving at a seam is not practice. What it does not have is a library
    there, so no target from an unpracticed tempo is ever a training row (`train_prog(tempi=…)`).
22. **Two parity references, both reported.** solo's gate is "at least as good as the audition's
    own chosen member", so the reference is the per-state audition winner (`ref_errors`), not a
    fixed member. At a practiced tempo the same-tempo recording is that reference — solo's gate
    verbatim, and solo finding 6's question. At an unpracticed tempo **there is no same-tempo
    recording for a learner**, so `scl0` is the *deployable* reference and `rec` is the *oracle*
    one. `proggate` respects the deployable gate and falls back to the `scl0` tape below parity,
    so the gate can never introduce drift; `prog` is ungated and measures transfer rather than
    adoption. τ = 0.75, solo's, with continuous fractions logged.
23. **`rec` at an unpracticed tempo is an ORACLE row, labelled as such.** It is what practice
    there would have bought, and no `prog` arm can see it. It is the ceiling, never a baseline
    the head is entitled to.
24. **The Δ sweep is read at L1 and L2 only.** Under `app=fixed` a level-4 arm decides once from
    a lead-in that does not move with Δ, so it is Δ-flat *by construction* — `t1` measured key_L4
    spreads of 1.00×–1.09×. L1 re-decides at all eight drilled seams and L2 at four, so those are
    the rungs where the Δ axis reads anything about feedback at all.
25. **σ is NOT changed, and the pool is logged instead.** Phase 1 measured the slot partition
    degenerate at four of five tempi (P-S 1.000×, 1 of 6 distinct argmins) and the head trains on
    that same pool. Changing σ would buy diversity at the cost of comparability with `t1`, so the
    pool's rendition spread is logged per tempo — `cmd_sd`, `cmd_rms`, `traj_sd`, the launch-state
    sd, and the harvest's own error mean/sd — beside the head's per-tempo training loss and two
    interpretable baselines: `mse_zero` (predicting no command; the target's own mean square) and
    `mse_slotmean` (each cell's mean tape, i.e. everything except the posture-conditioning). A
    head that learned the pool sits near `slot_mean`; one that learned the posture sits below it;
    one that learned nothing sits at `zero`. That is the instrument for "locate whether it is the
    pool or the head".
26. **Three heads, and the third is a ceiling control.** `prog` (practiced 32/16/8) and `prog2`
    (32/16) differ only in extrapolation distance. **`progall`** practices *every* tempo and is
    not a transfer arm at all: it separates "the head cannot extrapolate to this tempo" from "the
    head cannot represent this tempo at all". It was added after `psmoke4` measured the held-out
    command mse at **0.0007 inside the practiced set and 0.346 one rung outside it** — the
    failure was already located on the tempo axis, and `progall` turns that inference into a
    measurement. It is an oracle, exactly like `rec` at an unpracticed tempo, and is labelled as
    one everywhere.
26b. **P-E, the exchange rate between command error and execution error.** `psmoke3` measured the
    head fitting its training targets to mse 0.0013 — *below* the exploration noise's own floor
    of σ² = 0.0040, i.e. memorising them — and still executing 15× worse than the tape at a tempo
    it had practiced. That is neither a capacity nor a transfer result: it says an open-loop unit
    on a body with τ = 30 ms is exquisitely sensitive to command error, because the span is many
    τ long and nothing re-grounds. P-E takes the library's own best member, perturbs its commands
    by ε, executes from held-out launch states and reads the piece error — per tempo and per
    level. It converts any measured command RMSE into the execution error the body charges for
    it, and it is what separates "the head is bad" from "no approximation could work here".
    Smoke reading (`psmoke4`, band 0.0611): at H = 16, L4 the tape goes 0.0291 → 0.0435 → 0.0613
    → 0.2376 at ε = 0 / 0.02 / 0.05 / 0.20, so **a 5% command error already spends the whole
    band at the deep rung**, while at H = 2, L1 the same ladder is 0.0187 → 0.0189 → 0.0150 →
    0.0244, i.e. nearly flat. The level axis is the sensitivity axis.
26c. **Capacity and epochs sized against a measured floor, on a GPU.** The exploration noise is
    i.i.d. and unlearnable, so the target's own noise floor is σ² = 0.00395; `psmoke2` put the
    head at 0.0126 (3.2× the floor), i.e. a systematic command error of ≈ 0.09 RMS, which on
    A = 167 m/s² is enough to lose a leg inside one note. The head is therefore given hidden 512 ×
    4 layers and 1200 epochs with a cosine schedule — ~2 ms a step on an L4 against ~50 ms on
    these cores — and the capture set is doubled to 48 launch states per cell. The GPU idles
    through the ~900 s of plant work; that is the cheaper mistake. Whether the head reaches the
    floor is a result, and the parity-vs-`rec` column **at a practiced tempo** is the instrument
    that separates "cannot transfer" from "cannot reproduce a tape at all".
27. **Gate P-T1 replaces re-paying for P-F1 and P-F2.** Phase 2 adds no code to `world.py` or
    `piece.py` — the head and its decider live in `nets.py` — so rebuilding `t1`'s five libraries
    from the same seeds and reproducing its 75 construction scores and 50 sweep numbers at
    max|Δ| = 0 certifies the shared substrate has not moved, and `t1`'s own P-F1 (14 checks vs
    `acappella/b1`) and P-F2 (16 checks vs `prestissimo/a1g`) are inherited through it. The
    libraries have to be rebuilt anyway, so the gate is free. **P-G** covers the one thing P-T1
    cannot: the per-(tempo, Δ) gain table is pinned rather than re-fit (1073 s of `t1` saved), so
    one cell is re-fit from the full 63-cell grid and must reproduce `t1`'s choice exactly.
    **P-H** asserts the train/held-out split is disjoint rather than assuming it.

## Phase 3 — decisions taken, with reasons

28. **The crank runs over RECORDINGS, and the program is a logged instrument.** `p1` measured the
    program's parity gate at **0/90 open at H = 4 and 0/54 at H = 2** against the recording — it
    never opens off the practiced tempi — and where it does open, against `scl0`, the resulting
    arm is still 0.15–0.24 against a 0.0611 band because `scl0` is itself out of band everywhere
    but its source tempo. A ratchet commits content it will then deploy; at every tempo the
    content that is valid there is the recording harvested there, and the program is not. So
    `rec` is the executor. The head is not discarded: it is logged at each era boundary as the
    reading "would this table have survived the advance the loop is about to make?".
29. **π over slots is NOT learned, and the reason is measured.** `t1`'s P-S is 1.000× with 1 of 6
    distinct argmins at four of five tempi, and `p1`'s capture found `n_distinct_member` = 1.00 at
    the same four. A learned router would have nothing to select on. The decider stays `key` /
    `audit` as in Phase 1, and H = 4 (P-S 1.978×) is where a router would first have something to
    do — recorded as the place to put one if this line continues.
30. **`conductor/policy.py` is IMPORTED, not forked.** It is pure stdlib and deliberately lives
    outside the donor's Modal app, so `QuietPolicy`, `SchedulePolicy`, `YokePolicy`, `null_abba`
    and `policy_gate` are the donor's own objects running in this node's process. Gate **P-P**
    runs the donor's 12 offline checks in-process (`csmoke`: 12/12 pass), which is a stronger
    fidelity statement than any fork could make.
31. **A table is per tempo, and the committed level does not survive an advance.** On a tempo
    advance the loop's committed level resets to 1 and has to be re-earned by mining at the new
    tempo. That is not a conservatism: `t1` measured a recording to be worth 10–36× less at any
    tempo but its own, so a level carried across an advance would be a claim about content that
    no longer exists. Putting tempo on the era axis is exactly this exposure, and the crank has
    to survive it.
32. **The dead zones are measured on the anchor's own series, and a degenerate floor is reported
    rather than substituted.** `null_abba` on the anchor's `at_support` and `e_piece` series with
    the action cycles skipped; the dead zone for the decision statistic is `sd(N)/2`, policy.py's
    own derivation. If a read's floor comes back exactly 0 — its series never moved over any
    admissible window — that read is **not** given a substitute: its arm is flagged
    `floor_degenerate` and reported as "this currency never armed in this tag", which is
    conductor's own verdict on its raw `endo` read. `csmoke` hit exactly this at 14 cycles, which
    is why the run of record uses 80.

## Phase 2, the fix round — decisions and their PRE-FIXED readings

*Process, from 2026-09-06: every run of record now needs a smoke with numbers shown and an
explicit go before it. Both entries below are smokes. Nothing here is launched at full scale.*

33. **Capture at the DEPLOYMENT condition (`--capture-at deploy`).** `p1` captured the
    self-imitation buffer where the LIBRARY was built — a level-1-keyed traversal at Δ = 0 under
    the harvest gains — and deployed at Δ = 5 under `app=fixed`. A tape is invariant to that; a
    posture-conditioned head is not, and `p1` measured the size of it: `progall` at H = 2 matched
    the tape on held-out capture states (**head 0.0082 vs tape 0.0081**, 18/54 slots open) and
    read **0.7823** in the sweep. The flag captures at Δ = `delta_fix` under `app=fixed` instead,
    i.e. the sweep's own condition; everything else is `p1`. It is additive and defaults to
    `harvest`, so `p1` stays reproducible.
    **Pre-fixed reading, stated before the smoke ran**: does `progall` — which practiced H = 2 —
    reach the 0.0611 band in the sweep at H = 2, where `p1` had it at 0.7823 against a tape at
    0.0074? If it does, the capture/deployment mismatch was the cause. If it stays out of band
    **while its capture-state parity still matches the tape**, then P-E's exchange rate is the
    binding constraint — an open-loop unit on a τ = 30 ms body cannot be approximated at all —
    and that is what the record will say.
34. **The metronome ladder (`--head-mode incremental`).** `p1` asked the head for a whole octave
    in one jump (H = 8 → 4 → 2) and the gate never opened. A pianist does not do that: the next
    increment is small and the head is refit on its own executed traversals at every step. In
    incremental mode head `inc_i` practices `tempos[:i+1]` and is asked at `tempos[i+1]`, so the
    readout is **how far ahead of the practiced tempo the parity gate opens against the
    same-tempo recording** when the step is a quarter- or half-octave instead of a full one. The
    ladder is in integer control steps because that is what the body quantises to — H ∈
    {8, 7, 6, 5, 4} is 192 / 168 / 144 / 120 / 96 ms, steps of 2^(1/4) rounded to the clock — and
    the per-(tempo, Δ) gains for the rungs `t1` never ran are **re-fit from the full 63-cell grid
    under `t1`'s own rule**, never interpolated (`--refit-gains`; the re-fit rows are reported).
    **Pre-fixed reading**: the largest `tempos[i+1]/tempos[i]` at which `inc_i`'s parity against
    the same-tempo recording is open on a majority of slots at any level. `p1`'s answer at a step
    of 2.0 is zero slots; anything above zero here locates a step size at which a program does
    transfer, and zero everywhere says the tempo axis defeats it at any step.
    **Conditional, agreed before running**: this smoke is only worth running if (33) opens.

## The fix round's smoke (a), and what it located — `fsmoke_a`, 2026-09-06

721 s, `t1`'s knobs, tempi 16/8/2, capture at Δ = 5 under `app=fixed`, no Δ instrument.
**P-T1 75 checks at max|Δ| = 0.000e+00**, so the capture change disturbs neither the libraries nor
the recording arms.

**Neither branch of decision 33's pre-fixed reading fires, and the reason is a third mechanism.**
`progall`, which practiced H = 2, has a held-out command mse of **7e-06** there, and its parity
against the same-tempo recording at **level 4** is **head 0.0064 vs tape 0.0062** — equal. Its
sweep row at the same tempo and level is **0.4048**, against `aud_L4` 0.0069 and a 0.0611 band.
Both evaluations play the same 8-note span open-loop from a launch at drilled seam 0, so P-E's
exchange rate cannot be what separates them, and tempo extrapolation cannot be either.

What separates them is **the posture the head is handed**. `ProgBuffer.store` records
`okey["seam"][k]` — the TRUE seam state — while `ProgDecider.__call__` is handed `obs()`, the
Δ-stale read. `t1`'s hand-over log measures that read error at 0.0191 m at H = 32 and **0.3415 m,
2.8 legs, at H = 2**, and the failure orders with it: at H = 8 (0.62 legs) `progall_L1` is 0.1002
against `aud`'s 0.0312, a 3.2× gap; at H = 2 (2.8 legs) it is 59×. Capture-at-deploy halved
`progall_L4` at H = 2 (0.7823 → 0.4048) and `progall_L1` (0.3969 → 0.3254) — it changed the
DYNAMICS the states were generated under — but it still stored the true state, so it did not touch
the mismatch that matters.

The named next fix, **not run and not to be run without a go**: store the head's input as the
OBSERVED state, so training posture and deployment posture are the same object. It is a one-line
change to the capture and it would make the head a function of what the agent can actually see —
which is what `rec` is invariant to and a posture-conditioned program is not.

Per-level parity against the same-tempo recording, `progall` (open / checked, head vs tape):

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| `p1` H = 2 | 4/24 0.0076/0.0075 | 7/12 0.0063/0.0063 | 4/12 0.0123/0.0121 | 3/6 0.0063/0.0062 |
| `fsmoke_a` H = 2 | 9/42 0.0394/0.0394 | 2/24 0.0370/0.0369 | 7/12 0.0207/0.0207 | 0/6 0.0064/0.0062 |
| `fsmoke_a` H = 8 | 15/42 0.0484/0.0484 | 6/18 0.0538/0.0537 | 5/12 0.0336/0.0336 | 1/6 0.0067/0.0065 |

**Smoke (b), the metronome ladder, is built and NOT RUN** — it was agreed as conditional on (a)
opening, and (a) did not open. The machinery is in place behind `--head-mode incremental` and
`--refit-gains`.

35. **Store the head's input as the OBSERVED state (`--posture obs`).** Written down before the
    smoke ran. `ProgBuffer` recorded `okey["seam"][k]` — the TRUE seam state — while
    `ProgDecider` is handed `obs()`, the Δ-stale read, so every `prog` number in `p1` and
    `fsmoke_a` was produced by a head asked to condition on a posture it had never been trained
    on. The size of that mismatch is `t1`'s hand-over log: the read error at drilled seam 0 is
    **0.0191 m at H = 32 and 0.3415 m — 2.8 legs — at H = 2**, and the failure orders with it,
    **3.2× at 0.62 legs (H = 8) and 59× at 2.8 legs (H = 2)**. The fix stores the observed state
    as the head's input while the target stays the audition's winner computed from the TRUE state
    (a plant rollout cannot start from a fictitious posture), so the head learns "given what you
    can see, emit what was best given where you actually were". The buffer therefore carries both
    postures; parity and `ref_errors` keep rolling out from the true one. Additive, default
    `true`, so `p1` and `fsmoke_a` stay reproducible.
    **Pre-fixed reading**: does `progall` at H = 2 reach the 0.0611 band in the sweep **at any
    level** once train and deploy posture are the same object, with its capture-state parity
    reported beside it? If it does, capture consistency was the whole of the practiced-tempo
    failure and `p1`'s reading is **retracted in the record**. If it stays out of band while
    capture-state parity still matches the tape, then the program cannot regenerate deep content
    at the accuracy P-E demands even with consistent inputs, and that is the finding.
    This is the third attempt at one fix and also the first time the question is asked cleanly,
    which is why it is worth one smoke and no more.

## `fsmoke_a2` — decision 35's smoke, and the RETRACTION it forces

750 s, `t1`'s knobs, tempi 16/8/2, `--capture-at deploy --posture obs`.
**P-T1 75 checks at max|Δ| = 0.000e+00**, so the libraries and every recording arm are `t1`'s and
`p1` / `fsmoke_a` stay reproducible.

**The first branch of decision 35's pre-fixed reading fires.** `progall`, practiced at H = 2, is
**inside the 0.0611 band at every level there** — L1 0.0136, L2 0.0285, L3 0.0184, **L4 0.0056** —
at pass fractions 1.00 / 1.00 / 1.00 across ½, ⅓ and ¼ mean-leg for L4. Its L4 row is **better
than the tape it was cloned from** (`aud_L4` 0.0069, `key_L4` 0.0074) and **12.3× better than the
re-fit incumbent** (0.0690), at one feedback event against sixteen. The same row read 0.7823 in
`p1` and 0.4048 in `fsmoke_a`: **140× and 72×**.

At the tempi `prog` itself practiced the same thing holds without the oracle: **prog_L4 0.0086 at
H = 16** (against `aud_L4` 0.0093, `key_L4` 0.0087, reflex 0.0103 — the best arm at that tempo)
and **0.0063 at H = 8** (against `aud_L4` 0.0123 and reflex 0.0174 — **2.0× better than the tape**
and 2.8× better than the incumbent). That is `solo` finding 6 — the head beats the tape it was
trained on — reproduced on this substrate.

**RETRACTED, in the record.** `p1`'s reading that the program "reaches the band only at the
slowest practiced tempo" and that a second, unidentified limiter sat on top of tempo
extrapolation is withdrawn. There was one bug and it was mine: the head was trained on the TRUE
seam state and deployed on the Δ-stale read. `p1`'s located cause (i) — tempo extrapolation —
**stands and is now measured cleanly**; its located cause (ii) — a "capture/deployment
distribution mismatch" — was correctly detected but wrongly identified as the Δ = 0-vs-Δ = 5
*condition* (which `fsmoke_a` tested and which bought only 1.9×) rather than the true-vs-observed
*posture* (which buys 72× on top of it). Every `prog`, `proga`, `proggate`, `prog2` and `progall`
number in `p1` and `fsmoke_a` is a number about a head asked to condition on a posture it had
never seen, and should be read only as that.

**What survives, and is now the clean transfer result.** `prog` practiced {16, 8} and asked at
H = 2 is 0.2688 (L1) to 0.3158 (L4), with parity against the same-tempo recording **0/42, 0/24,
0/12, 0/6 open** at L1–L4 and a held-out command mse of **0.30756** against 0.0 at the practiced
H = 8. The tempo axis defeats the head at a full-octave step even when its inputs are consistent.

Per-level parity against the same-tempo recording (open/checked, head vs tape):

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| `progall` H = 16 | 11/24 0.0443/0.0341 | 7/12 0.0369/0.0349 | 4/6 0.0070/0.0071 | 4/6 0.0092/0.0092 |
| `progall` H = 8 | 14/42 0.0485/0.0484 | 2/18 0.0539/0.0537 | 2/12 0.0338/0.0336 | 0/6 0.0067/0.0065 |
| `progall` H = 2 | 15/42 0.0394/0.0394 | 10/24 0.0370/0.0369 | 7/12 0.0207/0.0207 | 1/6 0.0061/0.0062 |
| `prog` H = 8 | **30/42** 0.0484/0.0484 | **13/18** 0.0537/0.0537 | 6/12 0.0335/0.0336 | 0/6 0.0065/0.0065 |
| `prog` H = 2 | 0/42 0.1229/0.0394 | 0/24 0.1726/0.0369 | 0/12 0.2526/0.0207 | 0/6 0.3218/0.0062 |

**One thing the numbers say that nothing anticipated**: the head is best at the DEEPEST rung and
worst at the shallowest (`prog` at H = 8: L4 0.0063 against L1 0.0858), which is the opposite of
the ordering P-E's sensitivity curve would predict. A level-4 arm makes one decision from one
posture; a level-1 arm makes eight, each from a different stale read. Not pursued here.

36. **The metronome ladder, run (`--head-mode incremental`).** Written down before the smoke ran.
    `p1` and `fsmoke_a2` both asked the head for a full octave in one jump and the gate never
    opened: `prog` practiced {16, 8} and asked at H = 2 is **0/42, 0/24, 0/12, 0/6** open at
    L1–L4 with a held-out command mse of 0.30756 against 0.0 at the practiced tempo. A pianist
    does not move an octave at once. The ladder is **H ∈ {8, 7, 6, 5, 4}** — 192 / 168 / 144 /
    120 / 96 ms, steps of 1.14× / 1.17× / 1.20× / 1.25×, in integer control steps because that is
    what the body quantises to — and head `inc_i` practices `tempos[:i+1]` and is asked at
    `tempos[i+1]`, refit at every step on its own executed traversals. Gains for the rungs `t1`
    never ran are **re-fit from the full 63-cell grid under `t1`'s own rule**, never interpolated.
    Everything else is `fsmoke_a2`: `--posture obs --capture-at deploy`, since every prog number
    predating decision 35 is a number about a head conditioning on a posture it never saw.
    **Pre-fixed reading**: for each `i`, the fraction of slots at each level at which `inc_i`'s
    parity against the **same-tempo recording** at `tempos[i+1]` is open, and the sweep error
    there against the band and against `aud_L*`. The summary statistic is **the largest `i` at
    which any level exceeds 0.5 open** — i.e. how far along the metronome the program still
    regenerates its own content. `p1`'s answer at a step of 2.0 is zero slots at every level;
    anything above zero here locates a step size at which a program does transfer, and zero
    everywhere says the tempo axis defeats it at any step this body can express.
    **Carried into the reduction as a column, per level**: parity open-fraction and sweep error
    beside **the number of stale-read decisions that level makes per traversal — 8 / 4 / 2 / 1 at
    L1 / L2 / L3 / L4** — so `fsmoke_a2`'s unanticipated deep-best/shallow-worst ordering
    (`prog` at H = 8: L4 0.0063 against L1 0.0858) is on the record beside the quantity that would
    explain it.

## `fsmoke_b` — the metronome ladder, run. 1091 s

H ∈ {8, 7, 6, 5, 4}, `--head-mode incremental --refit-gains --posture obs --capture-at deploy`.
Gains re-fit from the full grid under `t1`'s rule reproduce `t1`'s pin exactly at the one cell
both ran ((H = 8, Δ = 5) → (1.2, 0.12)), and **P-T1 is 50 checks at max|Δ| = 0.000e+00** at the
two rungs `t1` measured. P-N 168/168 at all five tempi; P-H overlap 0.

**Decision 36's pre-fixed statistic is UNDEFINED: no level at any rung exceeds 0.5 open.** The
largest open fraction at a head's next rung is **5/48 = 0.104** (`inc2` at L1), and **L3 and L4
open zero slots at every rung**. The step sizes are 1.14× / 1.17× / 1.20× / 1.25×.

| head | practiced | asked at | step | L1 (8 dec) | L2 (4) | L3 (2) | L4 (1) |
|---|---|---|---|---|---|---|---|
| `inc0` | 8 | 7 | 1.14× | 0/36 | 0/18 | 0/12 | 0/6 |
| `inc1` | 8,7 | 6 | 1.17× | 0/48 | 0/24 | 0/12 | 0/6 |
| `inc2` | 8,7,6 | 5 | 1.20× | **5/48** | 2/24 | 0/12 | 0/6 |
| `inc3` | 8,7,6,5 | 4 | 1.25× | 3/48 | 3/24 | 0/12 | 0/6 |

**The cliff is at the practiced/unpracticed boundary, not at a step size.** Held-out command mse:
`inc0` 0.000000 (H = 8) → **0.0433** (H = 7); `inc1` 1.3e-05 (H = 7) → **0.2796** (H = 6);
`inc2` 8.6e-04 (H = 6) → **0.2611** (H = 5); `inc3` 9.0e-04 (H = 5) → **0.4165** (H = 4). Every
head fits every tempo it practiced to ≤ 9e-04 and loses two to three orders at the very next rung,
at 1.14× as much as at 1.25×.

The sweep says the same thing in the piece's units — `inc3_L4` is **0.0078 / 0.0054 / 0.0086 /
0.0070** across its four practiced tempi (all inside the 0.0611 band, beating `aud_L4` at H = 8)
and **0.2029** at the first tempo it did not practice.

**The cheapest possible program beats the learned one at every unpracticed rung.** `aud_s0_L4` —
the H = 8 tape, time-resampled, zero training — is **0.0272\* at H = 7 and 0.0530\* at H = 6**
(in band) against `prog_L4`'s 0.3459 and 0.2051, and `aud_s0_L1` at H = 7 is 0.0180\*, better than
the same-tempo recording's 0.0386. It fails at 1.6× and 2.0× (0.1021, 0.1058), so on a fine ladder
the resample carries two rungs and the head carries none.

**The deep-best / shallow-worst ordering inverts off the practiced set.** At a practiced tempo the
head is best at L4 (1 stale-read decision) and worst at L1 (8): `inc3` at H = 5 reads L4 0.0070
against L1 0.0550. At the first unpracticed tempo the ordering reverses — L1/L2 open a few slots,
L3/L4 open none, and `inc3` at H = 4 reads L4 0.2029 against L1 0.3427 in the sweep but 0/6
against 3/48 in parity. The `n_dec` column is on the record; nothing here explains the inversion.

**Unsmoothed anomaly**: `inc1` (practiced 8, 7) is catastrophic at H = 6 — L4 0.6771, held-out mse
0.2796 — worse than `inc0`, which practiced strictly less. And `inc0`'s aggregate parity at H = 5
(29/90) is higher than at the nearer H = 7 (0/72). Neither is monotone in practice or in step
size, and neither is explained here.

**Gain-grid flag**: the Δ = 0 re-fit sits at the kp grid **top** (9.6) at every one of the five
tempi (`at_edge` [True, False]); the Δ = 5 fits do not (1.2, 0.12, interior). The harvest gains
are therefore an upper bound at every tempo in this tag, as they were in `t1`.

## Known limitations, stated not hidden

- **`c1`'s within-level gauge is constant BY CONSTRUCTION on this substrate, so conductor's
  refuser cannot be reproduced here — only its impossibility shown.** The plant is deterministic,
  the per-tempo library is static, and there is no executor plasticity, so the arm's own metering
  error cannot move except when the loop acts. `c1` measured exactly that: null-ABBA over 48
  windows gave `ledger` sd(N) = sd(D) = 0 and a dead zone of exactly 0. conductor's `outer_ledger`
  refused because its currency quieted early; here it has nothing to quiet from.
- **`c1`'s yield thermostat never armed**: `at_support` rises monotonically through every era and
  does not plateau within 80 cycles, so all 12 of `outer_yield`'s actions were cap-forced. That
  makes `outer_yield` a **cap schedule with a different period**, not a driven loop, and its
  advantage over the anchor (8 commits / 4 advances vs 5 / 3, ending H = 2 L3 vs H = 4 L2) is an
  advantage of period, not of reading. Both of these are apparatus properties of this substrate,
  not results about the rule, and nothing should be built on `c1`.


- **32 shared eval geometries** in the sweep and 16 in the
  calibration, so the R choice is made on a different start set than the numbers it selects for.
- **The gain grid hits edges** at some cells (`at_kp_edge` / `at_kd_edge` reported at every
  (tempo, Δ)), as in acappella, solo and prestissimo.
- **P-A's `never_clamped` half fails, and the failure is in the gate's own scope, not the piece.**
  `k0·H > Δ` holds at the headline Δ = 5 at every tempo (6 > 5 at the fastest), which is what
  decision 4 sized `k_app` for. The gate as written asserts it over the whole **instrument** Δ
  ladder, which reaches Δ = 12; at H = 2 the lead-in is only 6 control steps, so the Δ = 8 and
  Δ = 12 instrument cells at that one tempo read the reset state. Those two cells are the
  aliasing prestissimo named and should be read as such; nothing at Δ = 5 is affected, and the
  spread half of the gate passes at every tempo (1.45×–41.48× under `arm`).
- **The library's slot partition is degenerate at four of five tempi** (P-S 1.000×, 1 of 6
  distinct argmins). Routing, trust and `π` over slots — Phase 3's machinery — would have nothing
  to select on at those tempi. H = 4 is the exception at 1.978× (L1, Δ = 0).
- **P-S's consumption distribution** is taken from an L1-keyed traversal at every level, so a
  level-ℓ cell's seam states are a proxy for the ones an actual level-ℓ arm would produce.
  acappella and prestissimo did the same; flagged.
- **Any tag at a mass other than 1.0 LEAVES THE DONOR PLANT.** No cross-tag control against
  `acappella`, `solo`, `etude` or `prestissimo` applies to its treatment numbers and none is
  claimed. P-F1 and P-F2 are unaffected — both build donor-plant pieces whatever the treatment
  cell is — and must still pass at 0.000e+00 in every tag.
