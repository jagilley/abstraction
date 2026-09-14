# rubato — the factored program: a kinematic chunk in the main model, a body model in the executor

You are implementing a new node under `experiments/mjc/practice/`, suggested name `rubato/`
(stolen time: a program whose tempo is not its own). Save this prompt verbatim as the node's
`SPEC.md`. Work in `experiments/` with `MODAL_PROFILE=chromatic`; read `/run-experiment-on-modal`
before launching anything and `/subagent-instructions` before waiting on anything. Do not touch
git. Fork donors, never edit them; every prior result must stay byte-reproducible.

**Process rule from Jasper, in force:** every step is a smoke first. Halt with the smoke's
numbers; nothing runs at full scale until I say go after reading them.

## Why this node exists

`accelerando/` (read its `FILES.md` first — 36 decisions, six run records, two retractions)
established, on a pusher with a 30 ms time constant at a fixed 120 ms delay, that tempo is the
right era knob: the stored unit's advantage over a per-tempo re-fit reflex rises monotonically
with tempo to 10× at 48 ms notes at one feedback event against sixteen, and the segment rung has
its own niche. It then asked whether a unit can be a *program* rather than a recording, so that
the library survives a tempo change. The answer on that body, in two smokes (`fsmoke_a2`,
`fsmoke_b`): a posture-conditioned head trained on the posture it can see matches or beats the
verbatim tape at every practiced tempo and level (L4 0.0056 vs tape 0.0069 vs reflex 0.0690 at
48 ms), and transfers to no unpracticed tempo at any step the body can express — 0/6 slots at
L3–L4 even at 1.14× — with held-out command error *worsening* as more tempi are practiced (0.04
after one, 0.42 after four). The time-resampled tape, with zero training, carries two fine rungs.

Jasper's reading, which this node tests: the head was asked to emit **forces**, and forces are
not tempo-invariant on any body (the drag part scales with speed, the inertial part with speed
squared, in a ratio set by the body's time constant — `accelerando/t1` measured both naive
scalings failing in opposite directions). What *is* tempo-invariant is the **path**: position and
velocity against phase, with the clock outside the program. A main model whose chunk is a path
has no tempo in it at all; converting the path to forces at a given tempo is inverse dynamics,
and doing it online with correction against a forecast is the cerebellum's job. The single-model
design forced the main model to learn forces, and forces carry the body's timing inside them.

So this node **lifts the no-forward-model rule of `solo`/`prestissimo`/`accelerando`, in one
specific way**: a body model (inverse and forward) may sit in the **executor**, on the learner's
side of the meter, converting a kinematic program to commands and correcting online. It may
**not** be handed to the incumbent as a planner — that is the `accompanist` confound that
invalidated `offbook`. The reflex stays model-free and re-fit per tempo. One reported reference
row, `reflex_ec` (the reflex with efference copy through the delay, `accompanist` d3b's
operator), keeps the incumbent honest now that the learner holds a model; it is reported, never
headlined. Records the roadmap's position: the plant model belongs on the learner's side of the
meter or not at all (ROADMAP §2.2, §7.1.3).

## Read first

- `accelerando/FILES.md`, `SPEC.md`, `piece.py`, `world.py`, `tempo.py`, `program.py`,
  `nets.py`, `analyze_*.py`, `t1_pins.py` — the fast body (mass 0.06, gear 10, damping 2; τ = 30
  ms measured), the tempo ladder H ∈ {32,16,8,4,2}, the piece and its calibration (R* = 0.160,
  band 0.0611), the level libraries and the weld, `app=fixed`, the body-scaled gain grid, the
  P-E sensitivity instrument, both grades, the resampler and its identity gate, the parity gate
  (execution reproduction on the plant from held-out seam states, head asked with the
  *observed* posture and rolled out from the true one — decision 35).
- `accompanist/README.md` and `FILES.md` (the efference-copy operator; the confound).
- `jacobian_teacher/` (README, `core.py`) — a learned `f(s,u)` on an mjc body, its action
  Jacobian, and the capacity ladder finding that the one-step map is near-perfect at every
  capacity while the composed map is not.
- `mjc/pusher_env.py` — the dynamics: `a = (gear/m)·u − (c/m)·v`, `|u| ≤ 1`, `dt_ctrl` 0.024 s
  on 0.002 s substeps, motor noise as configured in the practice line.
- `ROADMAP.md` §2.2 (the two FMs), §4.5, §7.1.3; `ideas/practice_manufactures_its_own_credit.md`
  §1–2; `ideas/two_climbings.md` §7.

## The design, held loosely

**The kinematic unit.** A committed level-ℓ unit is the **realized path** of its tape when played
at the tempo it was harvested at — position and velocity resampled onto phase φ ∈ [0, 1] over
the span (measured content, `legato` F4: produced by the body, so it carries no composition
error). Nesting stays as in `accelerando`: a level-ℓ path is its two level-(ℓ−1) paths in
sequence. At a target tempo T′ the program is x(φ) with the clock outside it: v = x′(φ)/T′,
a = x″(φ)/T′², from the stored path's derivatives. State the derivative scheme; it matters.

**The executor**, three forms, each a phase:

1. `kin_oracle` — the exact inverse dynamics `u = (m·a + c·v)/gear`, clipped to [−1, 1], from
   the body's true constants: an **experimenter-side oracle**, the way RHM uses exact DP, giving
   the ceiling of the factored architecture with a perfect cerebellum. Report the saturation
   fraction per tempo and level; `accelerando`'s tempo calibration found feasibility binding at
   48 ms notes, so saturation is a real ceiling, not an artifact.
2. `kin_learned` — an inverse model fitted on the body's **own executed data at the slow tempi
   only** (the practice traversals: (v, a) → u triples). Two families, both reported: the minimal
   one, linear in (v, a), and a generic MLP. This asks whether a cerebellum trained where the
   learner practiced extrapolates to the forces a faster tempo needs, and whether that depends
   on its inductive bias. Held-out fit at every tempo, beside execution.
3. `kin_corrected` — the learned inverse model plus online correction: a learned forward model
   `f(s,u)` from the same data rolls the issued commands forward from the Δ-stale read (the
   efference copy, now on the **learner's** side), and a PD term on the forecast state against
   the path corrects the feedforward command. Also with the exact forward model as the ceiling.

**The comparison**, at every tempo and level, all arms present in every table: the kinematic
unit from the **slowest** tempo's paths converted to each target tempo, against the same-tempo
recording (`rec`, the oracle content), the resampled tapes (`scl0`, `scl2`), the model-free
reflex re-fit per tempo, and `reflex_ec` reported. Both grades, pass fractions at three widths,
parity per slot against the same-tempo recording, the era table by tempo. The **transfer
question** is the headline: does a path stored at 768 ms notes play in band at 48 ms notes
through an executor that knows the body, where every force-level program failed.

**P-E, re-asked.** Perturb the executor's command by ε with and without online correction and
read the band cost per level. `accelerando` measured a 5% error spending the band at the deep
rung open-loop; whether correction relaxes that bar is the third phase's own readout.

## Phases — each a smoke first, then a run of record only on my go

**1 — the ceiling.** `kin_oracle` on the tempo ladder. No training anywhere. Smoke on three
tempi; halt with the numbers. If the kinematic unit does not transfer through a perfect
cerebellum, the factoring is not the answer on this body and that is the finding.

**2 — the learned cerebellum.** `kin_learned`, both families, trained on slow-tempo data only.
The readout is how much of Phase 1's ceiling survives, per tempo, and whether the two families
differ. Smoke, halt.

**3 — correction.** `kin_corrected` with the learned and the exact forward model; P-E with and
without correction. Smoke, halt.

## Gates, before any number is read

Fork fidelity: with every new arm off, the world must reproduce `accelerando/t1`'s rows
bit-for-bit (its P-T1 idiom). The oracle identity: the kinematic unit of a tape, converted back
at the tape's own tempo, must reproduce the tape's own commands up to the motor noise and
clipping the tape was recorded under — state the tolerance and its reason. The resampler and
weld identities as before. τ measured. Every calibration rule pre-fixed before its grid is read.

## Record

`FILES.md` from the start (`accelerando/FILES.md` is the template): decisions with measured
reasons, gates, smokes, run records with flags unsmoothed. Reductions under `results/<tag>/`,
figures under `figures/<tag>/`. A `ROADMAP_PROGRESS.md` entry per landed run, numbers only. **No
README.** Outcomes are not pre-interpreted here: if the ceiling transfers and the learned
cerebellum does not, if saturation caps the fast end, if correction buys nothing, that is the
finding, with its reason located.
