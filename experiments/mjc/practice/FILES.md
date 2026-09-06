# FILES — practice

**Up**: [README.md](README.md) · [../README.md](../README.md) (mjc)

## Code files

None directly at this node — `practice/` is an index and run record. Shared substrate code lives
at the mjc level (`../pusher_env.py`, `../shared.py`); the corrected-δ assembly every child forks
lives in `../bridge_assembly/bridge_assembly.py` (unmodified).

## Children

| child | what it varies |
|---|---|
| [`estimability/`](estimability/FILES.md) | Temporal concentration of practice at fixed per-context sample count (massed → spaced → interleaved), over a stream of 1000 context-pure batches whose *order* is the only thing that changes. Carries a passive benchmark-estimator panel (net at 4–5 lrs, per-context tabular EWMA at 3–4 α, global scalar, frozen) riding one common FM trajectory, scored against a measured oracle error field recomputed every 4 cycles. |
| [`priced_plasticity/`](priced_plasticity/FILES.md) | FM capacity (`fm_hidden` 256→32), replay (`n_replay` 4→0), and total plasticity spend (factored into relative allocation × spend, carried on the learning rate). Graded against a swept uniform-learning-rate Pareto frontier rather than a single matched-budget point. |
| [`aleatoric_flip/`](aleatoric_flip/FILES.md) | Flip amplitude f ∈ {0, 3, 10, 30} on an already-mastered region, with its rotation φ held. Includes an oracle-mask arm given the true flipped-region identity, and ensemble-disagreement and conjunction arms. |
| [`etude/`](etude/README.md) | **Sequence structure**, not per-step difficulty: a fixed K-segment "piece" whose hard passage is a localized command rotation. Varies *when* a mastered segment is compiled into a committed ballistic unit (δ-silence gate vs early/late schedule vs never vs deliberately-sloppy traces), graded on cumulative practice traversal time priced at a feedback delay × waypoint error at performance tempo. No per-sample δ gain anywhere — δ is used only as a detector. |
| [`fingering/`](fingering/FILES.md) | The compile-op taxonomy on a boundary that carries information (the n=3 arm's hand-over posture): frozen ops (fixed/keyed/regressed/averaged) vs **live content under committed routing**, timing × maintenance, priced deliberation (the plans counter). |
| [`legato/`](legato/FILES.md) | Committed **span** against the plant's composition horizon: segment vs phrase granularity × live vs measured content on a closed 4-leg loop whose phrase is ~3× the horizon, with a double-sided fusion control and a seam-cost calibration chosen by a pre-fixed neutral criterion. |
| [`offbook/`](offbook/FILES.md) | **The port back** (`native/`'s consolidation on the motor substrate): seam-time audition as the enumeration analogue, π-routing over library slots, a parity-gated span head, the address-book battery and poison twin — then three canary rounds on why the chain level is never adopted (credit currency, exposure, and an observation delay on the reflex loop). **Suspect** (2026-08-27): the incumbent plans for free in a near-perfect FM and the ports are built from it — see `accompanist/`. Rounds 5–7b (`d1`–`d3b`) live here as additive flags on its delay gate. |
| [`accompanist/`](accompanist/FILES.md) | **The super-writeup of the follow-up** (2026-08-26 → 27): library construction (content ladder, legato's nesting, the model adapting) and the delay operator (naive vs efference copy) on offbook's piece (`offbook/` Rounds 5–7b) and on a fast 120 ms-segment piece (child `presto/`). The pivot: the incumbent's free forward model, not the piece, refuses the deep unit. |
| [`acappella/`](acappella/FILES.md) | **The model-free port** (written up 2026-08-27): `etude/`'s substrate and piece with no forward model anywhere. Phase A ran RHM's grounding economy in the body's currency (priced real rollouts on a resettable plant copy, budget ladder to G=4096, reflex-law reference) and halted at its own pre-fixed gate — search never beats the reflex. The amendment re-sited Phase B on the feedback/delay axis: an unbridgeable observation delay Δ over reflex vs keyed/auditioned library arms, where a segment-span niche opens at 192 ms. |
| [`span/`](span/FILES.md) | **The composition horizon itself, as a trajectory.** `legato`'s practice loop with nothing committed and the metering removed, snapshotting the forward model dense-early/sparse-late and measuring at each snapshot (a) the planner-free imagination horizon along a fixed two-lap executed command sequence and (b) the executed span of one live open-loop phrase plan — against `e_react`, which is at plateau from cycle 0 on this piece. |
| [`tempo/`](tempo/README.md) | **Super-writeup** (2026-09-05 → 06) over three implementer nodes — `prestissimo/` (the four-rung execution-span ladder under delay on the donor body), `accelerando/` (a fast body, τ = 30 ms, at a fixed 120 ms delay with tempo as the era ladder; the force-level program; the crank), `rubato/` (the factored program: a path on phase through an exact, then learned, then forecast-corrected body model). Levels as execution span; delay and tempo as two knobs on Δ/T; a chunk is a program at the address and a recording in its content. Children carry `SPEC.md` + `FILES.md` only. |

## Related, living elsewhere

| node | what it varies |
|---|---|
| [`../bridge_assembly/difficulty_sweep/`](../bridge_assembly/difficulty_sweep/FILES.md) | Recovery difficulty, b(s)'s clock (`--bench-lr`), the frontier's clock (`--adapt-lr`), and the aleatoric decoy's amplitude, on the parent runner with defaults otherwise untouched. Lives under `bridge_assembly/` because it is that node's runner with swept knobs. |
