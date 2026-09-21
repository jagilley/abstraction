# Files — logit_reading/frontier

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

Imports the parent's `flat_oracle`, `calibration` and `altitude.frontier` unchanged, and replays
`rhm/practice/reread/lm/lm_reread.py`'s training loop unchanged (the fidelity gate asserts it).

## Code files

| file | purpose |
|---|---|
| `common.py` | The panel's primitives: `log_softmax`/`ent`/`xent`/`kl`; `detrend`, `template_scores`, `offset_means`, `leaf_templates` and `nested_phase` (the bottom-up period detector, `template` or `meanprof` mode); `endo_levels` (one estimated boundary column → every position's level by its `s`-adic residue) and `oracle_levels`; `level_panel` (excess, trend, `KL`, `H` by level), `confusion`, `detector_block`. Also the Modal image `_ignore` that skips `results/` trees and `.log` files. |
| `panel.py` | **Q1.** `panel_ckpt` reads one `stepNNNNNN_calibration.npz` and emits the oracle-labelled and endogenous-labelled panels plus the detector block; `panel_sweep` (CPU, ~40 s) runs all 13 checkpoints and writes `frontier_panel.json`. |
| `wall.py` | **Q2.** `build_refs` (GPU, once): the exact oracle predictives for the held-out venue V and the model-independent templates. `wall_arm` (L4, one container per arm): `lm_reread`'s loop verbatim with the panel logged at 23 checkpoints on venues V and C, checkpoints saved at 13 of them. `wall_sweep` (CPU coordinator) fans the arms out and runs `fidelity_gate` against `lm0`; `gate_only` re-runs the gate alone. |
| `analyze.py` | Local reduction: `--fetch` pulls `frontier_panel.json`, `refs_w0.json` and `w0/` off the volume into `results/raw/`; writes every table in [`results/tables.md`](results/tables.md) and, with `--figs`, `figs/{q1_panel,q1_endogenous,q2_wall,q2_by_level}.png`. Reads only JSON. |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | 36 tables, facts only, with the reproduction header. |
| `results/raw/*.json` | The fetched inputs to the reduction (the Q1 panel, the wall arms, the altitude JSONs Q1d re-reads, the `lm0` JSONs the gate compares against). |
| volume `/data/v16_s2_L6_m4_distinct/logit_reading/frontier/` | `frontier_panel.json`; `refs_w0.{npz,json}`; `w0/<arm>.json`, `w0/<arm>_stepNNNNNN.pt` (13 per arm), `w0/summary.json`. Two smoke artefacts (`refs_smoke.*`, `smoke/`) were left in place. |
