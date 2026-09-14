# accelerando — tempo as the era ladder, units as programs, on a body fast enough for the question

You are implementing a new node under `experiments/mjc/practice/`, suggested name `accelerando/`.
Save this prompt verbatim as the node's `SPEC.md`. Work in `experiments/` with
`MODAL_PROFILE=chromatic`; read `/run-experiment-on-modal` before launching anything and
`/subagent-instructions` before waiting on anything. Do not touch git. Fork donors, never edit
them; every prior result must stay byte-reproducible. **No forward model anywhere** — nothing
imports, trains or evaluates an `f(s,u)`; the resettable plant is the only oracle.

This node has three phases. Work through all three, halting with a launch handle and again
with a reduction at each; I resume you between halts. Stop early only if a phase's result makes
the next one ill-posed, and say why.

## Why this node exists

`prestissimo/` (2026-09-05, read its `FILES.md` first — 20 decisions and four run records) put
a four-rung execution-span ladder on `solo/`'s pusher and found the mapping's core prediction
holds model-free: rungs leave the playable band in strict order of feedback consumption, the
whole-figure unit beats the re-fit reflex 2–4.5× from 144 ms of delay while staying in band to
384 ms, and welding two units into one is exact. It also found three things that shape this
node:

1. **The band τ < T < Δ was empty.** τ is the body's velocity time constant (mass/damping,
   0.5 s on the pusher), T the note length, Δ the feedback delay (0.19 s at the niche). A stored
   unit can execute a note blind only if τ < T; feel fails only if T < Δ. With τ > Δ no tempo
   opens the band, so prestissimo inflated Δ to 384 ms to reach it. That is a diagnostic, not the
   phenomenon: a pianist's Δ is fixed and it is T that changes.
2. **Rungs 1–2 never paid on their own.** The reflex outlasted them; demand jumped from feel
   to the half- or whole-figure unit. The rungs that paid were those whose span exceeds τ.
3. **Seam information was ≈ 1.0 under delay in every cell.** Which unit you launch from a
   given posture did not matter, so routing and trust would have had nothing to select on.

And one structural fact from `solo/` finding 6: what internalizes is a posture-conditioned
program, not a recording — the corridor head beat the tapes it was trained on. A recording is
a command sequence in absolute time and is wrong the moment the tempo changes; a program can
be asked for the same trajectory faster. Only a program lets tempo be the era knob without
erasing the library at every rung, which is what a ratchet needs.

The design, agreed with Jasper: make the band exist physically (a fast body), fix Δ at a human
value, sweep tempo as the era ladder, store units as programs with recordings as the control,
and grade on the listener's clock. Keep the delay sweep as the instrument that decomposes any
result into its feedback and execution halves.

## Read first

- `prestissimo/FILES.md`, `SPEC.md`, `piece.py`, `world.py`, `ladder.py`, `analyze_ladder.py`
  — the piece-parameterised world, the nesting op (`build_level_cell`), the gates
  (P-F0/P-F1/P-N/P-A/P-S/P-T), the calibration rules, and the apparatus lessons: the kp grid
  must reach 0.5 or below; the lead-in must be played at the Δ = 0 read **and** the Δ = 0
  gains (`app=fixed`); the reset-state aliasing at Δ = k0·H; the reflex re-fit per condition
  with every advantage to the incumbent; pass fractions at three band widths for every arm
  including the reflex and its open-loop replay.
- `solo/README.md`, `FILES.md`, `solo.py` and `offbook/nets.py` (`SpanHead`, `build_span`,
  the parity gate as execution reproduction on the plant, the trunk clone).
- `acappella/README.md` + `SPEC.md` Phase B1 (the delay operator and the niche gate).
- `rhm/practice/ratchet/README.md` + `ratchet.py` (the miner, `mine_support`, `n_at_support`);
  `conductor/README.md` + `FILES.md` (the thermostat, dead zones by null-ABBA, yoked clocks).
- `mjc/pusher_env.py` — the plant knobs. The point-mass dynamics are `a = (gear/m)·u − (c/m)·v`,
  so the body is two numbers: τ = m/c and the acceleration scale gear/m. `dt_ctrl` = 0.024 s
  on 0.002 s substeps.
- `ideas/practice_manufactures_its_own_credit.md` §1–2; `ideas/two_climbings.md` §7, §9.

## The design, held loosely

**The body.** Choose τ in 30–50 ms, well under the fastest note you intend to play, and an
acceleration scale that makes the figure reachable at the fastest tempo (mass, gear and
damping trade off; pick the pair, state why). **Measure τ from a step response as a gate**;
do not assume it. The donor plant (mass 1, gear 10, damping 2) stays as the fork-fidelity
reference: the new world must reproduce `prestissimo/a1g`'s rows on its config bit-for-bit,
the way prestissimo reproduced `acappella/b1`.

**The delay.** Fixed, human-like: Δ in 4–6 control steps (96–144 ms), a constant of the body,
chosen before any arm is read and held across all three phases. The Δ sweep survives as an
instrument run at each tempo, never as the era knob.

**Tempo.** The same waypoints at every tempo, note length stepping down. Dyadic tempi are
natural here — H ∈ {16, 8, 4, 2} steps (384, 192, 96, 48 ms) — because then a level-(ℓ+1)
unit at tempo 2× has the same wall-clock span as a level-ℓ unit at tempo 1×, and the
listener's-clock window can be one slow note. Hold the exact ladder loosely; calibrate as
prestissimo calibrated R, with the rule fixed before the grid is read. The reflex is re-fit
per tempo (and per Δ in the instrument sweep), every advantage to the incumbent.

**Units as programs.** The table keeps identity: a level-ℓ slot is a pair of level-(ℓ−1) slots
at a seam (prestissimo's `LadderLayout`). Content comes from one of three sources, all three
arms in every comparison:
- `rec`: a recording harvested at the current tempo (prestissimo's construction). The control:
  what a tape is worth when the tempo it was cut at is the tempo asked for.
- `rec_scaled`: the slowest tempo's recording resampled to the asked tempo. The cheapest
  possible program; expected to fail for dynamical reasons and must be measured, not argued.
- `prog`: a head conditioned on posture, slot and tempo — a phase-conditioned emitter
  `u(φ, z, slot, tempo)` is one natural form, since the span's step count changes with tempo —
  trained by self-imitation on the body's own executed traversals at the tempi practiced so
  far, and gated per slot per tempo by parity on the plant (solo's gate: execution reproduction
  from held-out seam states). At a tempo never practiced, the head extrapolates and the gate
  decides whether it is trusted. Recordings at that tempo enter only after practice there.

**The grade.** Two, both pre-fixed and reported everywhere: prestissimo's per-waypoint error
with the ½-leg band, and a **listener's-clock** error — the executed trajectory against the
figure on a fixed wall-clock window (one slow note is the natural width), so at fast tempi
several notes fall inside one grading unit. Say which arms each grade favours; the two will
disagree somewhere and that disagreement is a readout.

**The piece.** A closed irregular figure with a few sharp turns and a command-rotation region
sized to the fast body, so that hand-over posture can matter. Seam information (P-S) is a
calibration readout at every tempo, not an assumption. Prestissimo's octagon is the starting
point; change it if P-S says it is smooth enough to be posture-free.

## Phases

**1 — the fast body and the tempo ladder, recordings only.** Plant gates (τ measured; fork
fidelity), the piece calibrated, static level libraries per tempo (`rec`), the reflex re-fit
per tempo, the tempo sweep at fixed Δ with both grades and pass fractions at three widths, the
era table by tempo (the smallest tempo at which level ℓ fails and ℓ+1 does not, and the one at
which the reflex fails), P-S per tempo, and the Δ sweep at each tempo as the decomposition
instrument. Also `rec_scaled` here, since it needs no training. **Halt with the reduction.**
This is the calibration everything else reads from: whether tempo alone opens a rung-by-rung
ladder on a fast body, and whether seams carry information.

**2 — the program.** `prog` trained on executed traversals at the slow tempi, gated at each
faster one, against `rec` and `rec_scaled` at every tempo and level. The transfer question:
at which tempo does a recording break and a program not, per level; and does the head beat the
tape it was trained on, as it did in `solo`. Δ sweep on the winning arm as the instrument.
**Halt with the reduction.**

**3 — the crank.** Mined nested tables (`mine_from = chosen` over adjacent committed slots
played by solved traversals, entering at `mine_support`; `at_support` per level as the free
one-level-up gauge), `conductor`'s thermostat deciding commit/hold per level and **tempo
advance**, dead zones by null-ABBA on this node's own series, arms in conductor's shape: the
scheduled crank, `outer_yield`, the within-level reader (the arm's own error, the one that
should refuse — and on a ladder whose lower rungs may be scaffolding, the one with nothing to
see), and a yoked-clock control per gauge arm. π over slots and the program head as in `solo`;
the address-book battery per level; can't-decompose at level 2. Single seed. **Halt with the
reduction.** If Phase 1 shows no era structure at all, or seams carry nothing at any tempo,
halt after Phase 1 and say so rather than building this.

Sizing: prestissimo's Phase A cells ran 5–17 min on 16 CPUs, no GPU; `solo/s0` was 56 min.
Budget each phase at a few multiples of that. Single seed; DoP at most 4; no multi-seed runs
without authorisation. Smoke on Modal before every launch; launch detached.

## Record

`FILES.md` from the start: decisions with measured reasons, gates, smokes, sizing, run records
with flags unsmoothed — `prestissimo/FILES.md` is the template. Reductions and figures under
`results/<tag>/` and `figures/<tag>/`. A `ROADMAP_PROGRESS.md` entry per phase, numbers only.
**No README** — numbers are discussed before any interpretation is written. Outcomes are not
pre-interpreted here on purpose: if tempo alone opens no ladder, if the program does not
transfer, if the crank refuses, that is the finding, with its reason located.
