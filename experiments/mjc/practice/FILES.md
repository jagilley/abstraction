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
| [`span/`](span/FILES.md) | **The composition horizon itself, as a trajectory.** `legato`'s practice loop with nothing committed and the metering removed, snapshotting the forward model dense-early/sparse-late and measuring at each snapshot (a) the planner-free imagination horizon along a fixed two-lap executed command sequence and (b) the executed span of one live open-loop phrase plan — against `e_react`, which is at plateau from cycle 0 on this piece. |

## Related, living elsewhere

| node | what it varies |
|---|---|
| [`../bridge_assembly/difficulty_sweep/`](../bridge_assembly/difficulty_sweep/FILES.md) | Recovery difficulty, b(s)'s clock (`--bench-lr`), the frontier's clock (`--adapt-lr`), and the aleatoric decoy's amplitude, on the parent runner with defaults otherwise untouched. Lives under `bridge_assembly/` because it is that node's runner with swept knobs. |
