# FILES — `fsm_part3/dispositional_grammar/`

Design decisions and their reasons live in [DESIGN.md](DESIGN.md); the parent index is
[../../../rhm/confabulation/FILES.md](../../../rhm/confabulation/FILES.md). The writeup (`README.md`) is written after the results are
discussed and is not in this folder yet.

## Code files

| File | Purpose |
|---|---|
| `dispositional.py` | The whole sub-experiment. Forks `rhm_confabulation.confabulation_test`'s `ntp_aux_cl` wake verbatim (helpers **imported**, not copied) and adds a checkpoint schedule, so the report head can be trained across M's own training trajectory. Five Modal functions, driven by one CPU coordinator: **`dispositional_run`** (coordinator — wake → instruments → targets → cells → summary, each stage resumable from its own volume part, independent stages fanned out over containers); **`wake_trajectory`** (GPU — the parent's wake with `torch.save` at 16 geometric checkpoints; its final val loss is the harness-validation gate against 2.3829); **`instrument_at_checkpoint`** (GPU, one per checkpoint — fresh instrument FMs at two capacities × 3 seeds with the parent's `train_fresh_fm` recipe, the two junk-residual guards `ens_cos` and hierarchy-η² under *both* estimators, and the cached report-set state `a0`/`a6`/`x7`/`x7_confab`/logits/NLL/entropy every later stage reads); **`build_targets`** (GPU — the shared k-means codebook on per-checkpoint-centered residuals, all eight targets, the level/checkpoint/position confound diagnostics, the instrument-reliability readings, and **the autocorrelation gate that chooses the horizon before any head is trained**); **`battery_cell`** (GPU, one per target × instrument — the self report head, the confabulator, and the observer ladder `O_input`/`O_io`/`O_io_half`/`O_act`/`O_hist`/`O_acthist`, each scored on held-out sequences *and* on a held-out checkpoint window); **`control_battery`** (GPU, one per target — paper 2's occurrent IMPL/BEHAV/ENT/WORLD rows at the final checkpoint with the parent's conventions verbatim, the "battery shown alive on this model" gate). |
| `analyze_dispositional.py` | CPU, local, no Modal. Three modes. **`--summary`** reduces a run's `<tag>_summary.json` to `figures/<tag>_reduction.txt` (nine tables: harness validation, guards along the trajectory, per-level NLL trajectory, the gate, the target confounds, the headline advantage per target per direction, the same within level, the observer ladders, the continuous readout) and to the figures below. Reduction only — no interpretation. **`--compare label=path ...`** emits the cross-arm / cross-horizon reduction (harness validation per arm, headline advantage over all runs, within-level advantage, the gate per arm, the guards per arm, per-level NLL start/end, the continuous readout, an `O_hist` table) plus two comparison figures. **`--followups <summary> --base _k1=... _k3=...`** emits the three-readings reduction (each reading's magnitude and floor share by level, advantage across readings pooled and within level, and the continuous readout by level) plus a readings figure. **`--compare ... --ceilings`** emits the within-level breakdown against *every* rung rather than only `O_io`: self / best `O_io` / `O_hist` / `O_act` / `O_acthist` / confabulator per level, per-level test `n`, and the advantage against `max(O_io, O_hist)` on both held-out axes — all read from the cells, which store each rung's per-level accuracy. |
| `add_excess_targets` (in `dispositional.py`) | Re-analysis, no retraining: adds `REORG_{PRO,RETRO}_excess` to an existing targets file — the fresh-instrument reorganization minus the per-position same-checkpoint no-change floor — so the grammar carries all three readings (raw fresh / excess / fixed) on the same within-(level, checkpoint) binning. Also records each reading's per-level magnitude, the floor's share of the fresh reading, and the rank correlation between the fresh and excess targets (DESIGN.md §18). |
| `followups` (in `dispositional.py`) | CPU coordinator for the two OL follow-ups: builds the excess targets at each horizon, then fans out four full-ladder cells (fresh × 2, excess × 2) and four `cont_only` cells over containers, and writes `<tag>_followups_summary.json`. |
| `floor_by_level` (in `dispositional.py`) | Re-analysis, no retraining: the per-position, per-level no-change floor `1 − cos(r^{s_i}_c(p), r^{s_j}_c(p))` — the disagreement between independent fresh FMs at the *same* checkpoint — raw and input-centered, from the saved instrument FMs and cached state. It is the floor for the *fresh*-instrument reorganization reading only; the headline `REORG` rows use the fixed-instrument reading, whose floor is exactly 0 (DESIGN.md §17). |
| `DESIGN.md` | Every design decision and its reason: fork discipline, why the checkpoint schedule is geometric, the 2 × 2 of targets and why PRO/RETRO are the same statistic read both ways, the fixed-vs-fresh instrument readings, the within-(level, checkpoint) binning that kills the level confound, per-checkpoint input centering, the two held-out axes, the two new observer rungs, the disjoint-window autocorrelation gate, the capacity guards, and the design's known limitations. |
| `FILES.md` | This file. |

Helpers inside `dispositional.py` worth knowing by name: `ckpt_schedule` (geometric in steps, so a backward horizon window spans the same step ratio as a forward one — the symmetry the PRO/RETRO comparison rests on); `level_of_pos` (s-adic valuation of `p+1`, the `PER_LEVEL_LOSS` convention); `_gate_stats` (autocorrelation across checkpoints, reported at lag 1 **and** at the disjoint lag `k+1` — adjacent difference windows share an endpoint with opposite signs, which biases lag 1 negative at `k=1` whatever the signal is, so only the disjoint reading feeds the horizon rule); `_bin_within` (quantile bins computed inside each (level, checkpoint) cell, which removes both confounds by construction) against `_bin_pooled` (kept for the record); `_var_share` / `_cat_lift` (how much of a target is just "which level" / "which checkpoint" / "which position" — the committee_head Phase C diagnostic); `_spearman`.

## Reproduce

```bash
cd experiments/

# wiring smoke (minutes; numbers meaningless -- M trained 300 steps, the junk regime,
# which this smoke reproduces on purpose: CTRL IMPL reads +0.097 there)
modal run a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run \
    --tag smoke --smoke --max-dop 10

# the run (one detached coordinator; every stage resumes from its volume part)
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run \
    --tag main --max-dop 6

# the closed-loop arm's second horizon: reuses ckpt/, fm/ and parts/, writes
# targets_k3.npz and cells/<cell>_k3.json beside the first (see DESIGN.md section 15)
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run \
    --tag main --out-suffix _k3 --horizon 3 --no-auto-k --no-ctrl --max-dop 6

# the open-loop arm, both horizons in one launch (see DESIGN.md section 16). The control
# battery runs once and is shared across horizons; only build_targets + cells repeat.
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run \
    --tag ol --cond ntp_aux --horizons "1,3" --max-dop 5

# reduction + figures (CPU, local; once per horizon)
modal volume get rhm-scaling-data \
  /rhm_confabulation/v16_s2_L6_m4_distinct/dispositional/main/main_summary.json .
python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional \
    --summary main_summary.json --tag main
python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional \
    --summary main_k3_summary.json --tag main_k3

# the per-level no-change floor for the fresh-instrument reading (GPU, minutes)
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::floor_by_level

# the OL follow-ups: the fresh and excess-over-floor readings through the cells, and
# the rank readout re-scored within level with predictions saved (both horizons)
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::followups \
    --tag ol --max-dop 6

# the cross-arm / cross-horizon reduction (one table set over all four runs)
python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional --name arms --compare \
    CL-k1=main_summary.json CL-k3=main_k3_summary.json \
    OL-k1=ol_k1_summary.json OL-k3=ol_k3_summary.json

# the within-level ceiling breakdown (self against every rung, per-level n)
python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional \
    --name ol_ceilings --ceilings --compare \
    OL-k1=ol_k1_summary.json OL-k3=ol_k3_summary.json

# the follow-ups reduction
python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional \
    --name ol_followups --followups ol_followups_summary.json \
    --base _k1=ol_k1_summary.json _k3=ol_k3_summary.json
```

Arms live in sibling tag directories: `dispositional/main/` is `ntp_aux_cl` (closed loop,
val 2.3829) and `dispositional/ol/` is `ntp_aux` (open loop, val 1.5447). They share
nothing on disk, so either can be re-run alone.

Individual stages can be invoked directly when only one needs redoing, e.g.
`...::build_targets --tag main` (recomputes every target and the gate from the existing
instrument parts, no retraining) or `...::battery_cell --cell IMPL_PRO --target IMPL_PRO_cls`.
`battery_cell`, `control_battery` and `instrument_at_checkpoint` skip when their part
already exists — delete the part to force a recompute.

## Data layout

Volume `rhm-scaling-data`, under `/rhm_confabulation/v16_s2_L6_m4_distinct/dispositional/<tag>/`:

| path | contents |
|---|---|
| `ckpt/ntp_aux_cl_step<step>_seed42.pt` | M's weights at each of the 16 scheduled checkpoints |
| `fm/<cap>_step<step>_s<j>.pt` | the fresh instrument FMs (2 capacities × 3 seeds × 16 checkpoints) |
| `parts/step<step>.npz` | that checkpoint's cached report-set state (`a0`, `a6`, `x7`, `x7_confab`, `logit` as fp16; `nll`, `ent`, `correct`, `tok`) |
| `parts/step<step>_meta.json` | that checkpoint's guards: FM cosine, `‖r‖`, `ens_cos`, hierarchy η² (reference last-token *and* pooled), rule η² |
| `targets<suffix>.npz` | every target at that horizon, both binnings, the level map, the split indices |
| `targets_meta<suffix>.json` | the gate table, class persistence, instrument reliability, target confounds, the per-level NLL trajectory, the chosen horizon and the held-out checkpoint window |
| `cells/<cell><suffix>_contpred.npz` | per-position predictions of the four rank models (self, `O_io`, `O_hist`, `O_act`) on both eval sets, with the z-scored target and the level map — so rank-based questions need no retraining |
| `excess_meta<suffix>.json` | each reorganization reading's per-level magnitude, the floor and its share, and the fresh-vs-excess rank correlation |
| `<tag>_followups_summary.json` | the follow-up cells, per horizon |
| `cells/<cell><suffix>.json` | one battery cell: self, confabulator, the full observer ladder, advantage pooled and per level, both held-out axes |
| `cells/CTRL_<target>.json` | one occurrent control row at the final checkpoint |
| `<tag><suffix>_summary.json` | everything above, aggregated — the only file the analyzer needs |

## Figures

`figures/<tag>_reduction.txt` plus `<tag>_guards.png` (FM cosine, `ens_cos`, d6 η² along the
trajectory, with the junk-regime lines marked), `<tag>_level_trajectory.png` (which levels M
learns and when), `<tag>_gate.png` (the autocorrelation gate against horizon), 
`<tag>_advantage.png` (advantage per target per direction, on both held-out axes),
`<tag>_advantage_by_level.png`, `<tag>_ladders.png` (the observer ladder for every target,
with the self-report, `O_act` and `O_hist` lines).
