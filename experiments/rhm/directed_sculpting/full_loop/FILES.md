# full_loop — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md).

Primitives live two levels up at [`../../`](../../FILES.md) because they are shared across RHM experiments
(per [`STRUCTURE.md`](../../../../STRUCTURE.md) — code lives at the lowest node that shares it):
[`../../rhm_channels.py`](../../rhm_channels.py) (the distractor DGP),
[`../../rhm_drift.py`](../../rhm_drift.py) (support-fixed drift, per-event σ calibration, the stationary
anchor), [`../../rhm_repair_cost.py`](../../rhm_repair_cost.py) (the meter and the homeostatic readout).
The sculpting apparatus this extends is [`../../rhm_sculpt_latent.py`](../../rhm_sculpt_latent.py) and
[`../../rhm_sculpt_internalize.py`](../../rhm_sculpt_internalize.py).

## Code files

| File | Purpose |
|---|---|
| `channel_env.py` | The multi-channel sculpting **environment** — what turns the DGP into a control task. Per-block rendering tables (each channel edits through its own grammar; noise blocks redraw iid), mixture rendering so drift changes the edit *dynamics* rather than only the data distribution, tree-only corruption, channel-aware generator/value/FM trainers, the three FM taps (`counterfactual_lp` = explore, `forecast_visits` = relevance, the two beams = control), depth/PR probes on the tree slice, the allocation policies, world snapshot/restore for cross-world grading, and Stage-5's belief-internalization ladder ported onto the layout |
| `env_check.py` | **E1–E6** certifying that environment before the loop runs on it: edit determinism under `canon`, the per-channel **irreducible floor** measured by redrawing the same `(x,k)`, FM learnability against that floor, DP solvability, the ballistic **commit-length** sweep that sets `chunk_len=3`, and the learned value's ΔV per channel against the exact DP Δ`d*`. Also `fm_bisect`: reference → tree-only → full-canon → full-mixture, one variable at a time — the back-compat gate for the port |
| `ladder.py` | **E1** — E3's allocation ladder over channels under continuous drift, ten policies. Carries the `e` tap in **two estimators run side by side** so the estimator itself is a controlled variable: the counterfactual fit (`lprog_only`/`value`/`value_satiety`) and floor-corrected reducibility (`reducible_only`/`value_red`/`value_red_satiety`). Both are computed every round for every arm, so the monitor charge is identical and only the allocation rule differs. Graded by tree-channel FM error (sighted) and ballistic beam success (behavioural), with the reactive beam as the near-blind control |
| `climb.py` | **E2** — repeated drift events with a *trainable* belief, against a `nodrift` matched-compute arm, on per-event σ matching and the entropy-matched control. Two sweeps: **which level drifts** (`surface`/`mid`/`root`) and — the one that produced the positive — **samples per event** at fixed KL/event, each budget carrying its own matched control. Reads repair-per-event against the per-level depth probe |
| `expansion.py` | **E3** — the grader-type × drift 2×2 (`frozen`/`dense`/`evaluative` × `static`/`drift`), nested so `evaluative − dense` isolates grounding. Reads the belief's PR and depth *trajectory*, plus fresh-FM transferability and ballistic control graded in three worlds (`base` / `novel` / `own`) |

## Results

| Path | What |
|---|---|
| `figures/fm_bisect_l4_v1/results.json` | The back-compat gate: reference 0.486 / tree-only **0.492** / full-canon 0.360 / full-mixture 0.310 `delta_cos` |
| `figures/env_check_l4_v2/results.json` | E1–E6 at L=4 — irreducible floors, floor-relative headroom, DP solvability, the commit-length sweep, the value's ΔV per channel |
| `figures/ladder_fix_s{1,2,3}/results.json` | **E1 with the repaired `e` tap**, 3 seeds, ten policies — the run the writeup quotes. Both estimators computed every round for every arm, so the monitor charge is identical and only the allocation rule differs |
| `figures/ladder_l4_v1/results.json` | E1 on the inverted counterfactual-fit tap, single seed. **Retained as the diagnosis**: mean LP came out exactly inverted (tree +0.044 < noise +0.066) and per-round tree-share sd was 0.309 |
| `figures/climb_nec_s{1,2,3}/results.json` | **The necessity sweep**, 3 seeds × 4 samples-per-event budgets × {nodrift, surface} — the deep-vs-surface migration (t = −5.44) |
| `figures/climb_em_v1/results.json` | **E2**, both drift fixes — the run the writeup quotes. Event KL matched to 7% across levels (0.539 / 0.498 / 0.566) |
| `figures/climb_l4_v1/results.json` | E2 on the pre-fix instrument (stationary σ, uniform-anchored control). **Retained as instrument history**: its per-event magnitudes were 0.78 / 1.00 / 3.49, a 4.5× spread inside a matched-magnitude sweep. Its depth null agrees with `climb_em_v1`, which is the point — the corrections pointed *toward* a climb and it still did not appear |
| `figures/expansion_em_s{1,2,3}/results.json` | **E3**, 3 seeds, entropy-matched base + per-event σ + cross-world grading — **the run that answers drift-vs-static**. Drift advantage is +0.011 ± 0.046 with the sign flipping across seeds |
| `figures/expansion_xw_s{1,2,3}/results.json` | E3 + cross-world grading on the uniform anchor, 3 seeds. **Retained as the demonstration the artifact was real**: the drift arm's own world has a ~18% lower stochasticity floor, and its apparent ballistic edge (+0.024, t=2.15) falls to +0.008 (t=0.71) once both arms are graded in a common world |
| `figures/expansion_l4_v1/results.json` | The first E3 pass — own-world grading only, single seed. Superseded for any drift-vs-static comparison; its grader-type numbers agree with the 3-seed runs |

## Children

None.
