# endo_expansion — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md).

The environment, the belief-internalization ladder and the three FM taps live one level up in
[`../channel_env.py`](../channel_env.py); the 2×2 this extends is [`../expansion.py`](../expansion.py).
Two strictly additive changes were made to `channel_env.py` for this node and every existing caller
stays bit-identical: `open_loop_beam(return_final=)` (hands back the terminal sequences so one beam
can be read by both the paid and the reported grader) and `belief_update(pstates=, proots=)` (lets
the plan term train on a filtered state pool while the dense term keeps the full one).

## Code files

| File | Purpose |
|---|---|
| `endo_expansion.py` | **E4** — the nine-arm harness. `frozen`/`dense`/`evaluative` reproduced from the published 2×2 as in-run anchors, plus `endo_rollout` (the missing cell), its two content controls `endo_shuffled` and `endo_random`, the reported-currency arm `endo_value`, and the `alloc_kstar`/`alloc_uniform` pair that tests endogenous allocation of an external target. One DP pass per round in every teacher arm — fed to the loss only in `evaluative`/`alloc_*`, used everywhere else as the `agree_dp` diagnostic — so the teachers are compared without a state-draw confound and the instrument charge is identical across arms. Re-seeds the global torch RNG identically at every arm, which is what makes splitting arms across parallel jobs exact |
| `endo_graders.py` | The endogenous teachers and the two new instruments. `collect_rollout_moves` (Monte-Carlo terminal task success per candidate move, with random tie-breaking and an informative mask), `collect_value_moves` (the reported-currency teacher), `grounded_candidates`/`metered_target` (the DP table and its `visits`-allocated restriction), `forecast_visits_per_block` (the ladder's `p` tap at block granularity), `move_scores`/`grader_disagreement` (five move-scorers of different type and all pairwise top-1 / rank-corr / magnitude — `heterogeneous_graders.md` §9's load-bearing test, first half), and `value_calibration`/`beam_belief_vs_truth` (the wirehead pair; see README §5 for why the first of these is invalid on CE-teacher arms) |
| `aggregate.py` | Merges the arm-groups per seed — refusing to if their round-0 beliefs differ, so the job split is checked rather than assumed — and prints the five tables: expansion with paired per-seed recovery of the DP teacher's lift, teacher quality against the `k*` it never sees, the disagreement instrument against its homogeneous floor, the wirehead pair, and the meter |

## Auxiliary docs

| File | What |
|---|---|
| `PREREGISTRATION.md` | The eight predictions, the falsifiers, and the stated limits, **written before the first launch**. Carries a dated appendix recording three implementation fixes a `--quick` smoke forced before the real run — including a lowest-index tie-break that was handing `alloc_uniform` a free on-target teacher — because one of them materially affects an arm's prediction and hiding it in a diff would be dishonest |

## Results

| Path | What |
|---|---|
| `figures/endo_expansion_e4a_s{1,2,3}/results.json` | The core question: `frozen` / `dense` / `evaluative` / `endo_rollout` |
| `figures/endo_expansion_e4b_s{1,2,3}/results.json` | The controls and the allocation cut: `endo_shuffled` / `endo_value` / `alloc_kstar` / `alloc_uniform` |
| `figures/endo_expansion_e4c_s{1,2,3}/results.json` | `endo_random` — the zero-information floor, added after the first full run showed `endo_shuffled` recovering 113% of the DP teacher's PR lift with ballistic control at the no-loop floor. **This is the arm that establishes belief PR is anti-informative** (145% of the lift, agreement with `k*` at chance) |
