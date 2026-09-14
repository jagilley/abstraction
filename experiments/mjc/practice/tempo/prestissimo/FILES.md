# prestissimo — file index and design record

**Up**: [`../../README.md`](../../README.md) (practice) · [`../../../README.md`](../../../README.md) (mjc)
**Contract**: [`SPEC.md`](SPEC.md) — the orchestrator's prompt, verbatim, as in `acappella/` and
`solo/`.
**Donors (forked, never edited)**: [`../../solo/`](../../solo/FILES.md) — `world.py` is a copy-fork made
piece-parameterised; `solo/` is untouched and `s0`/`s1q` stay byte-reproducible.
[`../../acappella/`](../../acappella/README.md) — the Δ ladder, the naive delay operator, the pre-fixed
niche gate, the playability guard. [`../../accompanist/presto/`](../../accompanist/presto/FILES.md) — the
fast-piece design and its twelve recorded decisions. [`../../legato/`](../../legato/README.md) — nesting,
the fusion control (F4), the level-(k+1) law (F5), and F2's calibration discipline.
[`../../etude/`](../../etude/README.md) — the plant, the piece constants, the compile op.
**RHM donors (read, not forked)**: [`rhm/practice/ratchet/`](../../../../rhm/practice/ratchet/README.md)
(`mine_from = chosen`, `mine_support`, `n_at_support`, the nested build that drops entries whose
halves are not committed) · [`rhm/practice/conductor/`](../../../../rhm/practice/conductor/README.md)
(the thermostat rule, the reads, the measured dead zones, the mandatory yoked-clock control).

**No findings README yet — numbers get discussed before any writeup** (repo policy). This file is
the record of what was decided and why.

## Status

| phase | what | tags | state |
|---|---|---|---|
| A | the piece, the static ladder, the Δ sweep — the era calibration | `psmoke`, `pcal`, `a0`, `a0s` | **run 2026-09-05**; reduction + 4 figures per tag |
| B | the crank: mined nested tables under conductor's outer loop | — | **not started** — held pending discussion of A's reduction |

## Code files

| file | what |
|---|---|
| `piece.py` | pure python. The DONOR square (étude's constants, verbatim) **and** the fast figure generator, plus the `Piece` object `world.World` is parameterised on and the pre-fixed `band()`. No numpy — a Modal *local* entrypoint imports it. |
| `world.py` | `solo/world.py` forked verbatim and made piece-parameterised, plus `LadderLayout` / `LevelLibrary` / `build_level_cell` (the nesting op) / `LadderDecider`. |
| `ladder.py` | the Phase A runner: gates P-F0/P-F1/P-T/P-N/P-A, the static ladder build, the Δ sweep, P-S. |
| `analyze_ladder.py` | the reduction (pure python, local) + the four figures (rendered remotely). |

## The question Phase A answers

At which Δ does a level-ℓ unit stop being playable, and does level ℓ+1 stay playable? That
reading is the **era calibration** Phase B's crank is paced against, and it is the first thing
this piece can say — including the possibility that the deep rungs are not readable at all.

## Decisions taken, with reasons (the things a reader could disagree with)

1. **Substrate: option (a) — a fast piece on `solo/`'s pusher, not option (b), the port onto
   `presto/`'s arm.** Three measured reasons, in order of weight. (i) The standing constraint is
   *no forward model anywhere*; `presto/world.py` is a 110 kB fork of `offbook/`'s in which the FM
   is the incumbent's planner, the audition, the delay bridge (`obs_predict`) and the content
   source. Porting it means either carrying a model (violating the constraint) or gutting the
   file, and what would be left is not presto's world. (ii) `presto`'s reflex is a
   Jacobian-transpose PD in tip space that reads fixed kinematics — arguably still model-free in
   purpose, but the trunk clone, the plant audition and every gate would have to be rebuilt on a
   new plant before the ladder question could be asked at all. (iii) Option (a) keeps a gate no
   other construction can give: the fork runs the **donor square** through the *same* `world.py`
   and reproduces `acappella/b1` at 0.000e+00, so the piece is provably the only variable
   (gate P-F1). The brief's own reading — "(a) reaches the ladder question faster; (b) is the
   biology-facing piece" — is adopted with (b) recorded as the escalation if the ladder turns.
2. **`H_seg = 5` = 120 ms, fixed by argument, not swept.** A human proprioceptive loop is ~100 ms
   = 4.2 control steps at the donor's `dt_ctrl = 0.024 s`, so a 5-step segment makes closed-loop
   correction *within* a segment physically impossible rather than forbidden (`offbook` Round 4
   rejected a feedback cap as necessity-by-fiat; presto moved the tax into the tempo). 120 ms is
   also presto's own segment duration exactly (its `H_seg = 6` at `dt_ctrl = 0.02`), so the two
   fast pieces are matched in wall-clock even though the plants differ.
3. **`K_drill = 8`, so the ladder has four rungs and the top one is the whole drilled figure.**
   `s = 2` as in RHM. presto's five does not tile. Entries are **aligned** (a level-ℓ entry sits
   at drilled seam `k ≡ 0 mod 2^(ℓ−1)`): RHM's are position-free because a macro applies wherever
   its span fits, but a piece is not translation-invariant, and aligning makes
   `T[ℓ] ⊆ T[ℓ−1] × T[ℓ−1]` a *tiling* — every level-ℓ unit is exactly two committed level-(ℓ−1)
   units and the top rung is the piece. Recorded as a choice; every-seam entries would give more
   mining data and a ragged top rung.
4. **`K_app = 3` approach segments, played closed-loop by the primitive, excluded from the
   comparison.** This is the whole reason a new piece exists. On étude's square the loop starts
   from rest at seam 0, so a committed arm reads the TRUE state there at every Δ and is exactly
   delay-invariant — `acappella` finding 5, flagged before its launch and never a defect of
   depth. Three segments put the first drilled seam 15 control steps (360 ms) into the traversal.
   Gate **P-A** measures that the artifact is gone rather than assuming it. The approach uses the
   **same per-Δ re-fit gains as the reflex arm** — every advantage to the incumbent, applied to a
   shared lead-in, so it is arm-neutral (presto decision 3); `e_app` is reported at every cell so
   the lead-in's own cost is on the record. The whole approach ledger (`k0 × H` feedback events
   per performer) is **subtracted** from the rent table, which is why `traverse` now splits the
   ledger at the first drilled seam (additive, inert at `k0 = 0`, which is what keeps P-F1 exact).
5. **An irregular octagon, not a scaled grand tour and not a regular one.** Cumulative vertex
   angles 0/38/90/131/186/225/275/318° with radius factors spread 0.90–1.08, giving legs that
   differ ≈1.46× and turns of 22.7–65.3°. A regular polygon at constant speed is a translation
   wearing a piece's name and a keyed library over identical segments is trivially degenerate
   (presto decision 2). The difficulty is the **turn rate** — ~45° every 120 ms against
   `offbook/`'s ~90° every 400 ms, a 1.7× rate — and it is smooth and global (`ballistic/` cut
   4b), on a plant whose velocity time constant is `m/c = 0.5 s = 21 control steps`, i.e.
   momentum-dominated, so braking must be anticipated.
6. **The command-rotation region is KEPT (scaled), where presto turned its patch off.** presto's
   `curl_b = 0` reasoning is that a localised **force** needle is open-loop *incompensable* —
   even a perfect model cannot counteract a strong local kick feedforward, so it saturates every
   ballistic arm at "fail". A command **rotation** is the opposite: it is exactly invertible, so a
   tape recorded through it carries the compensation and replaying it reproduces the motion on a
   deterministic plant. Keeping it (σ = 0.2 × mean leg, φ = 1.2 rad, étude's ratio; on drilled
   segment 3, mid-figure) preserves the model-free analogue of étude's `pretrain_mode=exclude`
   — the reflex's gains are fit on the world *without* the region, so the hard passage is
   unmodelled rather than wrongly modelled — which is the one thing besides delay that gives
   committed content anything to hold. The turn rate stays the global difficulty, as presto
   requires.
7. **The band — the `solved` criterion and the playability guard, one quantity.**
   `band = max(½ × mean drilled leg, ref_play × mean_leg / 0.8)` with `ref_play = 0.1066`
   (étude's `never` at performance tempo, 3-seed mean, published blind before this arc existed).
   The first term is presto decision 8's task-anchored component — the waypoints are still
   resolved, i.e. the figure is recognisable; the second is acappella's own published guard
   ported scale-free to a smaller figure. On the donor square the first term is 0.4 against the
   second's 0.1066 and acappella declined it as **vacuous**; on a fast piece the same
   figure-relative generosity is exactly what Jasper's note asks for (error at a fast tempo is
   partly a byproduct of chunking, and the listener's clock does not speed up with the notes).
   Both components are reported at every cell, and **pass fractions at ¼ / ⅓ / ½ mean-leg are
   logged for every arm at every Δ**, so the whole ladder can be re-read at another band without
   re-running anything (acappella's "continuous fractions logged" discipline).
   *Rejected alternative, recorded*: grading the executed trajectory on a fixed wall-clock
   listener grid. Over a 40-step drilled figure a listener period long enough to matter
   (≥ 200 ms) leaves ~6 grading points, i.e. barely coarser than the 8 waypoints it replaces, and
   it would need its own reference and break comparability with every number in the arc. The
   scale-free band achieves the same generosity with an instrument the record already reads.
8. **The tempo calibration rule, and its one reversal.** `R` (the circumradius) is the speed knob;
   `H` and the angles fix the turn rate, so **the turn rate is invariant in R** — calibrating R
   does not soften the piece's defining property, it only moves the figure's size. Rules, both
   naming *only the incumbent*:
   - *competence* — `e_reflex(Δ = 0, gains re-fit on the CLEAN world) ≤ band(R)`: the incumbent
     must be inside the playability band at zero delay, or there is no era 0 to leave. presto's
     own guard (`≤ ½ × the do-nothing floor`) is computed and reported beside it as an
     independent second read.
   - *demand* — `e_reflex(Δ = 8) > band(R)`: at the arc's published niche delay the incumbent
     must be out of the band, or the delay axis has no bite (`offbook` d0's measured failure was
     "playable for everyone, then unplayable for everyone").
   - `R* =` the **largest** R passing both — the fastest tempo feel can still play.

   **The reversal, with the grid it was made on (2026-09-05, tag `pcal`).** The first draft
   selected `argmax usable range = e_reflex(16)/e_reflex(0)`. The calibration grid — the
   incumbent alone, before any committed arm existed and before any treatment number was
   computed — showed that criterion is maximised by making the piece *easy*, because the range is
   a ratio to the incumbent's own floor and grows as the floor shrinks:

   | R | mean leg | v (m/s) | band | do-nothing | e_reflex@0 | (per leg) | @8 | @16 | range | comp | presto comp | demand |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | 0.08 | 0.0611 | 0.51 | 0.0305 | 0.204 | 0.0100 | 0.16 | 0.179 | 0.413 | 41.4× | ✓ | ✓ | ✓ |
   | 0.11 | 0.0840 | 0.70 | 0.0420 | 0.267 | 0.0165 | 0.20 | 0.186 | 0.578 | 35.1× | ✓ | ✓ | ✓ |
   | 0.14 | 0.1069 | 0.89 | 0.0534 | 0.331 | 0.0301 | 0.28 | 0.189 | 0.685 | 22.8× | ✓ | ✓ | ✓ |
   | 0.20 | 0.1527 | 1.27 | 0.0764 | 0.462 | 0.0576 | 0.38 | 0.199 | 0.691 | 12.0× | ✓ | ✓ | ✓ |
   | 0.28 | 0.2138 | 1.78 | 0.1069 | 0.637 | 0.1607 | 0.75 | 0.336 | 0.759 | 4.7× | ✗ | ✓ | ✓ |

   Range would have selected **R = 0.08**: a figure whose legs (0.061 m) are half the pusher's own
   radius (0.12 m) at 0.51 m/s — half étude's linear speed — i.e. it selects against the whole
   reason this node exists. The largest-R rule selects **R = 0.20**: legs 0.153 m, mean tip speed
   1.27 m/s (1.3× étude's), 120 ms segments, 45° turns, with the incumbent at 0.0576 just inside
   its 0.0764 band and already 2.6× outside it by Δ = 8. `usable_range` is still computed and
   reported at every cell. This is `legato` F2's discipline applied to itself: the first
   criterion did not preserve the axis. presto decision 10's precedent — a reversal recorded *as*
   the record.
9. **Two tempo tags, DoP 2** — `a0` at the rule's design point R* = 0.20 and
   `a0s` at R = 0.11 (a comfortably-competent cell: e_reflex@0 = 0.20 leg). The second is a
   robustness read on whether the ladder's readability depends on tempo, not a seed replicate and
   not a search over R: `a0` is the tag of record and the rule chose it. Flagged to the
   orchestrator on the halt.
10. **The library is built ONCE, at Δ = 0, with the harvest gains frozen at the Δ = 0 fit** —
   acappella's discipline (`a0`'s gains were frozen through the whole of `b1`'s sweep) and the
   brief's own reading of the era knob: a tape stays valid across delays but not across tempi, so
   the vocabulary persists across eras the way `T[ℓ]` persists across RHM's damage eras. The
   launch distribution at the first drilled seam *does* move with Δ, because the shared approach
   is played closed-loop; every arm inherits the same shift and `e_app` is reported.
11. **The nesting op is the weld, and it is EXACT.** A level-ℓ member's commands are the
   concatenation of two committed level-(ℓ−1) members'. Gate **P-N** asserts both halves: the
   *spelling* (bit-equality of the command arrays against the parents' members — binding, must be
   0 bad) and the *execution* (playing the weld from a state equals playing the halves in
   sequence). The execution half carries a stated tolerance of 1e-6 and cannot be exact, because
   `World.rollout` casts start states to float32 (the donor's line, untouched), so re-grounding at
   the internal seam round-trips the hand-over through float32 while the weld keeps it in float64
   inside MuJoCo — the smoke measured 1.0e-07, which *is* that round-trip. That is the point:
   on a deterministic plant, welding measured content is free (`legato` F4 measured −0.004 on the
   arm). **What a level changes is not execution but when the second half was chosen** — at the
   first seam, with the hand-over known at construction time, instead of read Δ-old at run time.
12. **A level-ℓ slot IS a pair of level-(ℓ−1) slots.** `parents = (a, b)`, members = the
   auditioned welds of their members. This is what makes `T[ℓ] ⊆ T[ℓ−1] × T[ℓ−1]` literal and
   what gives `native/`'s can't-decompose readout a form on this substrate for the first time
   (π's mass on a unit's own two parent slots) — `solo` finding 9 recorded that it had none at
   segment span. It also unifies Phase A and Phase B: Phase A's slots are the top pairs by
   construction audition, Phase B's are the mined pairs at support; the *object* is the same.
13. **The build walks the piece in order at every level** (presto decisions 6 and 9): both the
   score set and the candidate pool are drawn from the configuration that will deploy the unit —
   level-1 cells from a prefix that plays committed level-1 units at earlier seams, level-ℓ cells
   from a prefix that plays already-built level-ℓ units at earlier aligned seams and level-(ℓ−1)
   elsewhere. `d1`'s measured failure (a reactive-harvested pool left seams 1–2 4× worse at the
   per-state-oracle level) and legato F5 (the lower level's library funds the upper level's
   addressable variation) are both about exactly this. **Harvest late**: all harvesting happens
   after the full warm-up.
14. **Cross-level selection compares the mean over the span's waypoints** — `offbook`'s
   cross-level convention, carried unchanged so `aud_all` stays comparable with acappella's
   `lib_all`. acappella finding 6 says this selector is the most delay-fragile thing in the
   system; `aud_all`'s `mean_level` per Δ is that instrument here rather than a defect of the
   decider.
15. **Phase A has no learned component at all** — no trunk, no π, no torch. The reflex law,
   executed content, and the plant as the only oracle. The behaviour-cloned trunk enters in
   Phase B, where the primitive must be a legal choice at every seam for `no_table` to have a
   fallback and adoption to have meaning (`solo`'s design).

## Diagnostic round (decisions 17–19), 2026-09-05

Phase A's flags 1–3 each sit on a mechanism that is testable with one flag. All three are
**instruments, not arms**: incumbent- and library-side only, no learned component. Every rule below
was fixed before any grid was read, as the R rule was.

17. **The approach hand-over.** The three approach segments are played by the primitive under the
    *same* delay as the arms, so from Δ ≈ 6 up every committed arm launches the drilled figure from
    a state the delayed reflex has already lost. That is `acappella` finding 5 traded for its
    mirror: from rest the chain was delay-invariant; from a delayed approach every rung inherits
    the incumbent's collapse before it plays a note. presto decision 3 had the lead-in as a shared
    *pre-computed* plan precisely so it could not be a variable. `traverse` gains `obs_delay_app`
    (additive; `None` reproduces the old behaviour exactly, which is what keeps P-F1 untouched),
    the sweep runs every arm under both `app=arm` and `app=zero`, and a **hand-over log** is
    written once per (mode, Δ) — the approach's landing error `|x − V0|`, the hand-over speed and
    its deviation from the schedule's own tempo velocity, the agent's read error at drilled seam 0
    in position and velocity, and the read's source step. `app=arm` remains the arm of record.
    *A separate matter the same log settles*: the read at drilled seam 0 is clamped at the
    traversal start whenever `k0·H < Δ`. Here `k0·H = 15`, so **Δ = 16 reads the reset state** —
    the candidate explanation for `a0s`'s Δ = 16 reversal is aliasing, not a niche, and `clamped`
    is logged per cell so it can be read off rather than argued. `dH10` (`k0·H = 30`) carries the
    unclamped comparison.
18. **The plant's time constant against the segment.** Flag 3 says a 120 ms tape on a τ = m/c =
    0.5 s plant carries its hand-over error instead of damping it, so P-S reads 1.03× and keying
    has nothing to select on. `damping` is a plant knob the piece-parameterised world already
    takes, so the sweep is four cells of the ratio **seg/τ = H·dt_ctrl/(m/c)**:

    | tag | damping | H | seg/τ | terminal speed |
    |---|---|---|---|---|
    | `a1` | 2 | 5 | 0.24 | 5.0 m/s |
    | `dH10` | 2 | 10 | 0.48 | 5.0 m/s |
    | `d5` | 5 | 5 | 0.60 | 2.0 m/s |
    | `d10` | 10 | 5 | 1.20 | 1.0 m/s |
    | (étude's square, for scale) | 2 | 34 | **1.63** | 5.0 m/s |

    The R calibration is **re-run per cell under the unchanged rule** — necessary as well as
    principled, since terminal speed is `gear/damping` and at damping 10 the ladder's top rungs are
    kinematically infeasible. Pre-declared readouts: P-S's per-state-oracle ratio, the L1 tape's
    error relative to the reflex at Δ = 0, and the era table with Phase A's margins.
    **Flagged: any cell at a damping other than 2.0 LEAVES THE DONOR PLANT** — no cross-tag control
    against `acappella`, `solo` or `etude` applies to its treatment numbers and none is claimed.
    P-F1 is unaffected, because it builds the *donor piece*, which carries damping 2.0 whatever the
    treatment cell is, and it must still pass at 0.000e+00 in every tag.
19. **The reflex as a velocity tape.** `reflex_cmd` is `kp·(tgt − x) + kd·(vtg − v)` and `vtg` is
    the schedule's own feedforward velocity for the whole figure, so at `kp = 2` on a smooth
    octagon the incumbent is playing the *top-rung unit's content* open-loop with a velocity servo
    on top — which is why the re-fit sits on the grid floor and the ladder penalty is 9.5× and not
    `acappella`'s 134×. `FrozenReflexDecider` (`reflex_ol`) is the instrument: the same re-fit law,
    evaluated **once** at the drilled launch and played open-loop to the end, paying the one
    feedback event every committed arm pays there. **Pre-fixed reading, stated before the grid**:
    the closed-loop incumbent is called *a tape* at a Δ when `ol/cl ≤ 1.10`, i.e. the 40 feedback
    events it is charged for buy less than 10%; the continuous ratio is reported at every cell so
    another threshold can be read off. The kp grid also gains **two rungs below `a0`'s floor**
    (0.5, 1.0) — two rather than one so the floor has headroom to be found rather than merely
    moved — and every cell reports the argmin **restricted to kp ≥ 2.0**, `a0`'s own grid, so the
    two are comparable inside one tag.

20. **The clean approach variant — the lead-in as one pre-computed plan.** Decision 17's
    `app=zero` was measured **not to isolate the read**: it removed the approach's observation
    delay but kept the *per-Δ re-fit gains*, so the lead-in was played by a controller fitted for
    a stale read while consuming a fresh one. It under-drives, and the hand-over got **worse** at
    every Δ ≥ 3 in all four diagnostic cells (`a1` Δ = 8: 0.71 → 1.49 legs; `d5`: 0.40 → 1.22;
    `d10`: 0.53 → 1.46; `dH10`: 0.26 → 0.99), with the hand-over speed falling below the
    schedule's tempo. The log says why in its own columns: under `zero` the read-error term
    `rd dv` goes to 0.00–0.07 at Δ ≤ 6 while `|x − V0|` keeps growing — a **gain** effect wearing
    a **read** effect's name.
    `app=fixed` is the clean variant: the approach reads at Δ = 0 **and plays the Δ = 0 gains**,
    so the lead-in is a single pre-computed plan (presto decision 3, verbatim in intent),
    identical at every rung of the sweep. `World` gains `kp_app` / `kd_app` — additive, `None` by
    default, and structurally inert at `k0 = 0`, which is what keeps P-F1 exact. Verified on the
    smoke: the hand-over into drilled seam 0 is **numerically identical at Δ = 0, 4, 8 and 16**
    (`|x − V0|` 0.1000, `|v|` 1.273, `|v − v*|` 0.979), so the only thing that varies across the
    sweep is the drilled figure's own read — which is what the sweep is supposed to isolate.
    Two cells on the DONOR PLANT only, `a1g` (= `a1`'s config) and `dH10g` (= `dH10`'s), each
    keeping `app=arm` beside it as the arm of record.

## Gates

| gate | what it asserts | result |
|---|---|---|
| **P-F0** | the 13 donor constants, `ast`-read out of `etude/etude.py`, never imported | 13/13, smoke and both runs |
| **P-F1** | the fork, run on the **donor square**, reproduces `acappella/b1`'s library build (4 segment + 3 chain construction scores) and its Δ = 0 and Δ = 8 rows for `key_seg` / `lib_seg` / `lib_all` / `key_chain` / `aud_chain` / reflex, and re-derives b1's per-Δ gains | **14 checks, max\|Δ\| = 0.000e+00, gains_match=True** (`psmoke`) |
| **P-T** | the tempo calibration, incumbent only, rules fixed before the grid was read (decision 8) | R* = 0.20 (`pcal`) |
| **P-N** | the nesting op: spelling exact, execution within the float32 hand-over round-trip | 168/168 spelled, 0 bad; 1.105e-07 |
| **P-A** | the approach removed acappella finding 5's artifact — the deepest committed arm is *not* delay-invariant | pass, 11.5× spread |
| **P-S** | seam information under the delayed read, per level per Δ (per-state oracle vs the best fixed slot) | instrument |

`P-F1` pins b1's **own** configuration (`DCFG`, `D_KP_GRID`, `D_KD_GRID`) inside the gate rather
than inheriting this node's knobs, so it is *defined at every treatment config* — including
`--quick`, where it is the smoke's strongest single check. Rollouts are deterministic and chunking
across the process pool is bit-identical by construction, so `n_proc` cannot move it either.
This is stricter than `solo`'s S-F1, which declared itself inapplicable whenever the run's config
differed from b1's.

## Arms (Phase A)

| arm | decision rule | groundings |
|---|---|---|
| `reflex` | the reflex law, gains re-fit per Δ on the clean world | 0 |
| `key_Lℓ` | frozen posture key over level-ℓ slots, then the nearest member key | 0 |
| `aud_Lℓ` | plant audition of every member of every level-ℓ slot | `n_slot × m_member` per decision |
| `key_all` / `aud_all` | every rung legal at every aligned seam | as above |

Every arm plays the same shared approach at the same Δ; the approach ledger is subtracted.

## Sizing and smokes

| tag | what | wall |
|---|---|---|
| `psmoke` | `--quick`, full chain, every gate exercised; P-F1 defined and passing at 0.000e+00 | 157 s (P-F1 116 s of it) |
| `pcal` | `--cal-only`, the P-T grid at full gain resolution (7 × 7), 5 R × 3 Δ | 78 s |
| `a0` | Phase A at R* = 0.20 — the tag of record | 328 s |
| `a0s` | Phase A at R = 0.11 (the tempo robustness read) | 330 s |

## Run record — Phase A (`a0`, `a0s`), 2026-09-05

Both 16 CPUs, no GPU, seed 0, 328 s / 330 s wall. `a0` is the tag of record (R\* = 0.20, the
calibration rule's design point); `a0s` is the tempo-robustness read at R = 0.11 (`--r-force`).

**Gates.** P-F0 13/13. **P-F1 14 checks at max|Δ| = 0.000e+00, `gains_match=True`** in both tags —
the fork run on the donor square reproduces `acappella/b1`'s four segment and three chain
construction scores and its Δ = 0 and Δ = 8 rows for `key_seg`/`lib_seg`/`lib_all`/`key_chain`/
`aud_chain`/reflex, and re-derives b1's per-Δ gains. **P-N** 168/168 level-(ℓ>1) members bit-spelled
by their two parents, 0 bad; weld vs halves-in-sequence 1.105e-07 (the float32 hand-over
round-trip, tol 1e-6). **P-A passes**: the deepest committed arm spreads 11.5× over the Δ ladder
(`key_L4` 0.1043 → 1.1356) against acappella finding 5's exactly-invariant chain arms — the
artifact the piece exists to remove is gone, measured rather than assumed.

**Acquisition.** 30 336 groundings per performer, 13 724 s priced, for a 105-slot ladder
(L1 56, L2 28, L3 14, L4 7 — 6 slots + 1 poison per cell × 4 members). The weld auditions
dominate; the whole build is 25 s wall at 16 processes.

**The headline, uninterpreted.** At R\* = 0.20 **no committed arm is inside the band at any Δ, at
any of the three band widths** (½ / ⅓ / ¼ mean-leg), and the reflex is better than every committed
arm at every Δ. The era ladder is all `null`. At R = 0.11 committed arms are in-band at Δ ≤ 4 and
the era ladder has entries (`key_era_3 = 1`, `aud_era_3 = 1`, `aud_era_4 = 4`), but the reflex is
still better than every committed arm at every Δ ≤ 12, so era 1 (reflex fails, L1 passes) is
`null` in both tags. `acappella` b1's niche — stored content beating a per-Δ re-fit reflex from
Δ = 8 — **does not reproduce on this piece at either tempo**.

**Flags, unsmoothed.**

1. **The per-Δ re-fit lands on the kp grid MINIMUM (kp = 2.0) at 8 of 9 delays in `a0` and 6 of 9
   in `a0s`.** On a fast piece the best "feel" controller is a near-pure feedforward velocity
   tracker with almost no positional feedback — which is structurally delay-robust, because it
   barely consumes the delayed read. The incumbent's delay penalty over the whole ladder is 9.5×
   here against acappella's **134×** on the slow piece. The grid was widened *upward* for the fast
   schedule (decision 8) and the binding edge turned out to be the bottom; extending it downward
   is the named cheap check if this cell becomes load-bearing.
2. **Nothing above Δ ≈ 8 is interpretable.** Every arm (reflex included) is above the do-nothing
   floor at Δ = 12 in both tags — 0.462 at R = 0.20, 0.266 at R = 0.11 — i.e. diverged, not
   degraded. `a0s`'s Δ = 16 cell is non-monotone (committed arms 1.00 → 0.12 between Δ = 12 and
   Δ = 16, the only cells in either tag where committed content beats the reflex); the Δ = 16 gain
   fit sits at the **kd grid minimum**, so the candidate cause is the gain fit collapsing to
   near-do-nothing, and the cell is not read as a niche.
3. **Seam information is nearly absent, at every level.** Per-state oracle over the best fixed
   slot: mean **1.03×** at Δ = 0 in `a0` (range 1.01–1.07) and **1.20×** in `a0s` (1.00–1.65),
   against `solo`'s 1.52–2.40× and `acappella`'s 1.22–2.35× on the slow piece. At Δ = 8 and 16 it
   is **1.00× with 1 of 6 distinct argmins at every level** — under delay, keying picks the same
   slot for every performer and all slots score identically. Candidate mechanism, argued not
   measured: a 5-step segment is 0.24 of the plant's velocity time constant (m/c = 0.5 s = 21
   steps), so an open-loop tape cannot damp a hand-over error inside its own span; it carries it.
4. **The per-seam audition is WORSE than the frozen key at low Δ** — `a0` at Δ = 0: `aud_L1`
   0.1650 vs `key_L1` 0.0939, while the per-state oracle at each seam is ≈ 0.081. Greedy
   per-seam selection nails its own waypoint and hands the next segment an unplayable state:
   `legato` l1/F3's measured disease ("hit waypoints while ignoring HANDOFF VELOCITY"),
   reproduced on a fast piece from the *content-selection* side rather than the controller side.
   The effect vanishes at the top rung, where one decision covers the whole figure and there is
   no per-seam greed to commit: `aud_L4` 0.0927 beats `key_L4` 0.1043.
5. **The depth ordering is present and monotone at low Δ, in the predicted direction.** `a0`
   `aud`: L4 < L3 < L2 ≈ L1 at Δ ∈ {0, 1, 2, 3}, with `aud_L1 / aud_L4` = 1.78× / 1.94× / 1.60× /
   2.11× at **8× fewer feedback events** (1 vs 8 per drilled traversal). It inverts at Δ ≥ 8
   (0.91× / 0.93×), inside the diverged range. `a0s` shows the same shape (1.42× / 1.43× / 1.20× /
   1.82× / 3.46× at Δ ≤ 4).
6. **Level-ℓ construction scores degrade along the piece** (`a0` L2: 0.0776 / 0.1210 / 0.1533 /
   0.2582 at aligned seams 0/2/4/6), the greedy nested prefix accumulating error — legato's own
   ordering effect, on the record rather than smoothed.

**The band, in one paragraph.** The form fixed before any run is
`max(½ × mean drilled leg, ref_play × mean_leg / 0.8)` with `ref_play = 0.1066` (étude's `never`,
published blind); the second term never binds, so the operative band is ½ × mean leg — 0.0764 at
R\* = 0.20 and 0.0420 at R = 0.11. It is not vacuous: the incumbent sits just inside it at Δ = 0 by
construction of the calibration rule, and is 2.6× outside it by Δ = 8. **`a0`'s ladder is not
readable at any of the three widths** — the pass fraction is 0.00 for every committed arm at every
Δ at ½, ⅓ and ¼ mean-leg, with two exceptions, both at Δ = 0: `key_L1` 0.219 and `aud_L4` 0.094 at
the ½ width only. That is the "band admits nothing" outcome, at full scale, and widening the band
does not rescue it, because the reflex beats every committed arm at every Δ by 1.6–2.8× — the
failure is the *ordering*, not the threshold. `a0s`'s ladder **is** readable at ½ and partly at ⅓
(e.g. Δ = 4: `aud_L4` 0.94 / 0.50 / 0.00 at ½ / ⅓ / ¼), which is why it is reported beside `a0`.

## Run record — the diagnostic round (`a1`, `d5`, `d10`, `dH10`), 2026-09-05

Four cells, DoP 4, 16 CPUs, seed 0, no learned component; 504–678 s each. **P-F0 13/13, P-F1
14 checks at max|Δ| = 0.000e+00 with `gains_match=True`, P-N 168/168 members bit-spelled by their
parents with 0 bad and execution 3.3e-08–1.3e-07, P-A pass — in all four tags**, `d5`/`d10`/`dH10`
included, because P-F1 builds the donor piece at damping 2.0 whatever the treatment cell is.

### The kp floor as found on the extended grid (decision 19)

The floor moved from `a0`'s 2.0 to 0.5 and **still binds**: at Δ ∈ {6, 8, 12, 16} in `a1`, {8, 12,
16} in `dH10`, {12, 16} in `d5`, and **nowhere** in `d10`. The optimum is below the extended grid
wherever it binds, so the incumbent's floor is measured only as an upper bound at those cells.
Every cell also carries the argmin restricted to kp ≥ 2.0, which reproduces `a0`'s reflex column
**exactly** — so the size of the clipping is on the record inside one tag:

| Δ | `a1` restricted (kp ≥ 2, = `a0`) | `a1` extended | chosen gains |
|---|---|---|---|
| 0 | 0.0769 | 0.0769 | kp 2 / kd 16 |
| 6 | 0.1371 | 0.1096 | kp 0.5 / kd 1 |
| 8 | 0.2066 | 0.1705 | kp 0.5 / kd 0.5 |
| 12 | 0.5460 | **0.2692** | kp 0.5 / kd 0.25 |
| 16 | 0.6877 | **0.3249** | kp 0.5 / kd 0.25 |

The committed arms move further than the incumbent does, because the shared approach is played
with these gains: `aud_L4` at Δ = 16 goes **1.1471 (`a0`) → 0.1752 (`a1`), 6.5×**, and no arm in
`a1` is above the do-nothing floor (0.4620) at any Δ. Phase A flag 2 — "nothing above Δ ≈ 8 is
interpretable, everything is diverged" — was **the clipped grid on the approach, not the plant**.

### `a1` hand-over log (leg 0.1527, schedule tempo 1.273 m/s, k0·H = 15)

| Δ | app=arm \|x−V0\| | (legs) | \|v\| | read@ | app=zero \|x−V0\| | (legs) | \|v\| |
|---|---|---|---|---|---|---|---|
| 0 | 0.0928 | 0.61 | 1.264 | 15/15 | 0.0928 | 0.61 | 1.264 |
| 4 | 0.0366 | 0.24 | 1.197 | 11/15 | 0.1364 | 0.89 | 1.181 |
| 8 | 0.1082 | 0.71 | 1.634 | 7/15 | 0.2270 | 1.49 | 0.986 |
| 12 | 0.2531 | 1.66 | 1.158 | 3/15 | 0.3007 | 1.97 | 0.788 |
| 16 | 0.2528 | 1.66 | 1.173 | **0/15 clamped** | 0.3007 | 1.97 | 0.788 |

**`app=zero` makes the hand-over worse, not better, at every Δ ≥ 3, in all four cells** (`d5` at
Δ = 8: 0.40 → 1.22 legs; `d10`: 0.53 → 1.46; `dH10`: 0.26 → 0.99), and the hand-over *speed* falls
below the schedule's tempo under `zero` while it stays at or above it under `arm`. Stated plainly:
**the instrument as implemented does not isolate what decision 17 aimed at.** `app=zero` plays the
*per-Δ re-fit gains* with an undelayed read — a controller fitted for a stale read consuming a
fresh one, which under-drives, and at Δ = 16 those gains are kp 0.5 / kd 0.25. The residual drift
under `zero` is therefore a GAIN effect, not a READ effect, and it is directly readable in the log
(`rd dv` → 0.00–0.07 under `zero` at Δ ≤ 6 while `|x−V0|` still grows). The clean variant — the
Δ = 0 approach played at the **Δ = 0 gains**, i.e. presto's shared pre-computed lead-in — is a
one-line follow-up and was not run.

### The aliasing verdict for Δ = 16

Confirmed as a fact: at k0·H = 15 the drilled-seam-0 read is `read@step 0/15` at Δ = 16 — the
agent reads the **reset state** — in `a1`, `d5` and `d10`, under both approach modes. `dH10`
(k0·H = 30) is the unclamped comparison: its Δ = 16 read is `read@step 14/30`.
**But aliasing is not what produced `a0s`'s Δ = 16 reversal.** With the kp floor unclipped, `a1`'s
Δ = 12 → 16 is monotone (`key_L4` 0.1575 → 0.1606, `aud_L3` 0.1449 → 0.1420) and the 10× jump
`a0s` showed does not reproduce. The reversal was the clipped gain fit; the clamping is real,
independent, and still present.

### The damping table (decision 18) — off damping 2.0 these cells LEAVE THE DONOR PLANT

| tag | c | H | seg/τ | R\* | leg | band | P-S gain Δ0 | Δ8 arm | Δ8 zero | reflex@0 | key_L1/rf | aud_L1/rf | key_L4/rf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a1` | 2 | 5 | 0.24 | 0.20 | 0.1527 | 0.0764 | 1.029 | 1.004 | 1.002 | 0.0726 | 1.29× | 2.27× | 1.44× |
| `dH10` | 2 | 10 | 0.48 | 0.28 | 0.2138 | 0.1069 | 1.000 | 1.003 | 1.050 | 0.0132 | 4.50× | 3.24× | 2.84× |
| `d5` | 5 | 5 | 0.60 | 0.20 | 0.1527 | 0.0764 | 1.012 | 1.006 | 1.000 | 0.0771 | 2.15× | 1.68× | 1.76× |
| `d10` | 10 | 5 | 1.20 | 0.11 | 0.0840 | 0.0420 | **1.252** | 1.036 | 1.002 | 0.0121 | 1.64× | 1.33× | 1.34× |

Seam information rises with seg/τ only at the top cell, and only there: `d10` at Δ = 0 reads
L1 1.34× / L2 1.20× / L3 1.10× / L4 1.07×, against 1.00–1.03× everywhere else and `solo`'s
1.52–2.40×. It collapses to 1.01–1.05× by Δ = 4 in every cell. No committed arm beats the reflex
at Δ = 0 in any cell (the ratios above are all > 1).

### Era tables (band-crossing) and the niche (committed beats the re-fit reflex)

| tag/mode | eras (key) | eras (aud) | committed < reflex |
|---|---|---|---|
| `a1`/arm | — | — | Δ8 1.38× · Δ12 1.87× · **Δ16 2.29×** |
| `a1`/zero | — | — | Δ6 1.06× |
| `dH10`/arm | era₃ = 3 | era₂ = 2, era₃ = 4, era₄ = 6 | **Δ16 3.15×** |
| `dH10`/zero | era₂ = 1, era₃ = 2 | era₂ = 2, era₃ = 3 | Δ12 1.15× · Δ16 1.18× |
| `d5`/arm | — | — | Δ16 1.03× |
| `d5`/zero | — | — | Δ12 1.04× |
| `d10`/arm | era₂ = 12 | era₂ = 2, era₄ = 8, era₃ = 12 | Δ6 1.06× · Δ8 1.10× · Δ12 1.80× |
| `d10`/zero | — | era₂ = 1 | — none |

**era₁ (the reflex fails the band and L1 passes) is null in every cell and every mode.**
The cleanest column is `dH10`/arm/aud: L4 is inside the band at Δ ∈ {0, 3, 4, 6, 8, 16} while L1
is outside from Δ = 2 and L2 from Δ = 3, with m34 positive at all nine Δ (+0.0061 … +0.1151) and
m23 positive at seven of nine. Its Δ = 16 cell — `aud_L4` 0.1025 inside the 0.1069 band while the
reflex reads 0.3227 — is the **only cell in the whole round where a committed arm is both in-band
and ahead of the incumbent**, by 3.15× at 1 feedback event against 80.

### The open-loop-reflex column (decision 19)

`ol/cl`, the re-fit reflex against its own read-frozen replay. Pre-fixed reading: ≤ 1.10 ⇒ the
incumbent is a tape at that Δ.

| tag/mode | Δ0 | Δ1 | Δ2 | Δ3 | Δ4 | Δ6 | Δ8 | Δ12 | Δ16 |
|---|---|---|---|---|---|---|---|---|---|
| `a1`/arm | 13.58 | 10.27 | 8.44 | 11.77 | 15.38 | 9.78 | 2.53 | 1.75 | 1.65 |
| `a1`/zero | 13.58 | 10.47 | 10.16 | 9.95 | 9.81 | 5.37 | 2.68 | 1.51 | 1.47 |
| `dH10`/arm | 118.98 | 66.65 | 58.66 | 81.83 | 96.14 | 18.89 | 20.40 | 10.51 | 2.91 |
| `d10`/arm | 34.26 | 25.75 | 16.50 | 13.81 | 11.99 | 10.47 | 3.86 | 2.95 | 1.42 |
| `d10`/zero | 34.26 | 22.17 | 14.92 | 15.15 | 11.77 | 7.66 | 2.14 | 1.60 | **1.10\*** |

By the pre-fixed reading the incumbent is a tape in **exactly one cell of the round** — `d10`/zero
at Δ = 16, at the threshold. Everywhere else it is not, including every cell where kp sits on the
grid floor (`a1` Δ ≥ 6 runs 9.78 → 1.65). The ratio is monotone in Δ in every tag. At `a1` Δ = 0,
where `a0`'s flag 1 located the mechanism, the closed loop is **13.6×** better than its own
open-loop replay.

## Run record — decision 20's clean approach variant (`a1g`, `dH10g`), 2026-09-05

Two cells on the **donor plant only**, DoP 2, seed 0, 593 s / 1016 s. Each runs `app=arm` (the arm
of record) beside `app=fixed`. Gates in both: P-F0 13/13 · **P-F1 14 checks at max|Δ| = 0.000e+00,
`gains_match=True`** · P-N 168/168 spelled, 0 bad, execution 1.10e-07 / 1.33e-07 · P-A pass. The
extended kp floor (0.5) still binds at Δ ∈ {6, 8, 12, 16} in `a1g` and {8, 12, 16} in `dH10g`.

### The hand-over log — `fixed` does what it was built to do

| | Δ0 | Δ2 | Δ4 | Δ8 | Δ12 | Δ16 |
|---|---|---|---|---|---|---|
| `a1g` arm `\|x−V0\|` (legs) | 0.61 | 0.43 | 0.24 | 0.71 | 1.66 | 1.66 |
| `a1g` **fixed** | **0.61** | **0.61** | **0.61** | **0.61** | **0.61** | **0.61** |
| `dH10g` arm | 0.06 | 0.10 | 0.06 | 0.26 | 0.43 | 0.42 |
| `dH10g` **fixed** | **0.06** | **0.06** | **0.06** | **0.06** | **0.06** | **0.06** |

Under `fixed` the hand-over into drilled seam 0 is **numerically identical at all nine Δ** —
`a1g` `|x−V0|` 0.0928, `|v|` 1.264, `e_app` 0.0933; `dH10g` 0.0125, 0.891, 0.0111 — so the sweep
isolates the drilled figure. The only thing that varies is the arm's own read staleness at that
seam (`rd dx` 0.0000 → 0.3746 in `a1g`, → 0.3428 in `dH10g`). Δ = 16 is still clamped at the reset
state in `a1g` (k0·H = 15) and unclamped in `dH10g` (k0·H = 30, read@14/30).

### Delay flatness (max/min over the Δ ladder) — `offbook` finding 7's law with the lead-in removed

| tag / mode | reflex | key L1 | L2 | L3 | L4 | aud L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|---|---|
| `a1g` arm | 6.45× | 2.45 | 1.59 | 2.37 | 2.14 | 1.52 | 1.44 | 1.60 | 1.97 |
| `a1g` **fixed** | 5.70× | 2.03 | 1.23 | **1.02** | **1.04** | 1.25 | 1.07 | 1.22 | 1.25 |
| `dH10g` arm | 24.37× | 4.76 | 5.78 | 5.30 | 5.30 | 9.43 | 9.18 | 7.37 | 9.50 |
| `dH10g` **fixed** | 22.14× | 3.34 | 5.81 | **1.73** | **1.73** | 11.83 | 9.19 | 4.72 | 3.69 |

### `a1g` (c = 2, H = 5, seg/τ = 0.24, R\* = 0.20, band 0.0764, do-nothing 0.4620)

`fixed`/key is flat from Δ = 3 on — `key_L3` 0.1539 and `key_L4` 0.1082 at every Δ from 3 to 16
(one decision, same slot) — against the reflex's 0.0619 → 0.3407. Margins under `fixed`/key: m34
+0.0442 … +0.0572 at all nine Δ, m23 +0.0136 … +0.0510 at all nine; under `fixed`/aud m23 positive
at all nine (+0.0374 … +0.0714), m34 at eight of nine.

| Δ | reflex | key L4 | ratio |
|---|---|---|---|
| 8 | 0.1535 | 0.1082 | 1.42× |
| 12 | 0.2735 | 0.1082 | **2.53×** |
| 16 | 0.3407 | 0.1082 | **3.15×** |

(`arm`: 1.38× / 1.87× / 2.29×.) **eras are null in both modes.** Pass fractions: the only committed
cells that ever pass are `aud_L4` 0.03–0.09 at Δ ≤ 6 and `key_L1` 0.22 at Δ = 0, all at ½ leg; at
⅓ and ¼ leg **every committed arm is 0.00 at every Δ**, and so is the incumbent except reflex 0.16
at Δ = 4 (⅓). Reflex's own ½-leg pass fraction runs 0.59 / 0.69 / 0.69 / 1.00 / 0.94 / 0.16 / 0.00
/ 0.00 / 0.00. So at this tempo the band admits essentially nothing at any width, the incumbent
included, and the ⅓ / ¼ columns do not rescue the ladder.

### `dH10g` (c = 2, H = 10, seg/τ = 0.48, R\* = 0.28, band 0.1069, do-nothing 0.6392)

**eras[fixed]**: key era₂ = 1, era₃ = 2 · aud era₂ = 2, era₃ = 4, era₄ = 6.
**eras[arm]**: key era₃ = 3 · aud era₂ = 2, era₃ = 4, era₄ = 6. era₁ null in both.

`fixed`/aud, piece error (\* = in band), with the ½-leg pass fraction beneath:

| Δ | reflex | ol | L1 | L2 | L3 | L4 | m23 | m34 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0132\* | 1.5754 | 0.0429\* | 0.0411\* | 0.0255\* | 0.0194\* | +0.0156 | +0.0061 |
| 2 | 0.0299\* | 1.2530 | 0.1383 | 0.0367\* | 0.0438\* | 0.0534\* | −0.0071 | −0.0096 |
| 4 | 0.0138\* | 1.3020 | 0.1248 | 0.1691 | 0.0650\* | 0.0338\* | +0.1041 | +0.0312 |
| 6 | 0.0629\* | 1.1062 | 0.3070 | 0.1708 | 0.1203 | 0.0290\* | +0.0505 | +0.0914 |
| 8 | 0.0603\* | 1.0650 | 0.3103 | 0.1708 | 0.0333\* | 0.0338\* | +0.1375 | −0.0004 |
| 12 | 0.1209 | 1.0328 | 0.3843 | 0.3375 | 0.0333\* | 0.0290\* | +0.3041 | +0.0044 |
| 16 | 0.2932 | 0.7184 | 0.4459 | 0.3375 | 0.1203 | 0.0715\* | +0.2171 | +0.0488 |

`aud_L4`'s ½-leg pass fraction is **1.00 at every Δ from 0 to 16** (1.00 at ⅓ too for Δ ≤ 12;
0.25 at ⅓ and 0.09 at ¼ at Δ = 16). `aud_L1` leaves at Δ = 2, `aud_L2` at Δ = 4, `aud_L3` at
Δ = 6 (returning at 8 and 12). **The incumbent leaves the band at Δ = 12** (0.1209 > 0.1069, pass
0.00) while `aud_L3` and `aud_L4` stay at 1.00 — the playability crossing exists, at levels 3 and
4 rather than at level 1, which is why era₁ is null while era₂/₃/₄ are not.
Committed beats the re-fit reflex under `fixed` at Δ 6 (2.17×), 8 (1.81×), **12 (4.18×)** and
**16 (4.51×)**, against `arm`'s single cell (Δ16, 3.15×). m23 positive at seven of nine Δ, m34 at
eight of nine.

### The open-loop-reflex column

| tag / mode | Δ0 | Δ2 | Δ4 | Δ6 | Δ8 | Δ12 | Δ16 |
|---|---|---|---|---|---|---|---|
| `a1g` arm | 13.58 | 8.44 | 15.38 | 9.78 | 2.53 | 1.75 | 1.65 |
| `a1g` fixed | 13.58 | 10.26 | 11.44 | 6.71 | 3.04 | 1.76 | 1.97 |
| `dH10g` arm | 118.98 | 58.66 | 96.14 | 18.89 | 20.40 | 10.51 | 2.91 |
| `dH10g` fixed | 118.98 | 41.87 | 94.66 | 17.59 | 17.68 | 8.54 | 2.45 |

**The pre-fixed ≤ 1.10 reading fires nowhere in either tag, under either mode.** `reflex_ol` is
above the do-nothing floor at every Δ in both tags and its pass fraction is 0.00 at all three
widths everywhere — it is an instrument, as declared, not a viable arm.

## Known limitations, stated not hidden

- **32 shared eval geometries** in the sweep and 16 in
  the calibration (so the R choice is a design decision made on a different start set than the
  numbers it selects for — the choice is not a result).
- **The gain grid hits edges** at several R cells (`at_kp_edge` / `at_kd_edge` reported at every
  Δ). acappella's b1 had the same and reported it.
- **P-S's consumption distribution** is taken from an L1-keyed traversal at every level, so a
  level-ℓ cell's seam states are a proxy for the ones an actual level-ℓ arm would produce.
  acappella did the same; flagged.
- **δ_perf against the tape is not logged in Phase A.** `traj` is carried on every member (a
  level-ℓ member's is the concatenation of its parents' measured trajectories — what the halves
  were measured doing, i.e. the intention reference), and the instrument is wired in Phase B,
  where it is consumed by nothing, as the brief directs.
- **Timing has no certifier**, as on `solo` (finding 8). Deliberately left that way this round.

## Reproduce

```bash
cd experiments/                      # MODAL_PROFILE=chromatic
modal run mjc/practice/prestissimo/ladder.py::prestissimo_ladder --quick --tag psmoke
modal run mjc/practice/prestissimo/ladder.py::prestissimo_ladder --cal-only --no-pf1 --tag pcal
modal run --detach mjc/practice/prestissimo/ladder.py::prestissimo_ladder --spawn --tag a0 --seed 0
modal run --detach mjc/practice/prestissimo/ladder.py::prestissimo_ladder --spawn --tag a0s \
    --seed 0 --r-force 0.11
python3 mjc/practice/prestissimo/analyze_ladder.py --tag a0 --fetch --figures
```

Volume `mujoco-control-data`: `/data/practice_prestissimo/<tag>/ladder.json`; fetched copies and
`reduction.txt` under `results/<tag>/`, figures under `figures/<tag>/`.
