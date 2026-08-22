# Files — `dress_rehearsal`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `submission.py` | **The port, v3.** One self-contained competition submission (~28 KB, passes `one-layer validate`) exporting `SUBMISSION`. Bidirectional prompt encoder pooling a state `h₀` and a rule vector `r` (`variable_modulus`'s `cond` arm); a tied operator applied **exactly `T` times**, `T` parsed from the prompt by walking left over digit tokens from the last real token (loop-on-`T`; legality reading in the docstring); a decoder emitting a right-aligned digit grid scattered to where `target_positions` reads; wall-clock LR schedule. Two arms selected by the `ARM` constant: `control` (terminal CE) and `closure` (plus `ballistic_depth` §2's label-free cycle + re-entry terms as a `token_training_loss` over tensors handed out through `auxiliary`). In v3 the cycle term re-encodes **through the real encoder** — decoded soft digits written in place, right-aligned into the `x` field's span, terms masked to width-matching rows — and the aux ramp is **gated on an EMA of train exact crossing 0.5** (clock fallback 60%). v1 (fixed 64-unroll + learned rung selector) and v2 (`ReEncoder`, clock-gated ramp) are superseded and identified only by SHA in `results/hosted/*/submission.json`. |
| `emit_arm.py` | Rewrites `submission.py`'s single `ARM = "…"` line into `<out_dir>/<arm>/submission.py`, so the two arms cannot drift and each keeps the filename upstream requires. Checks the 256 KiB limit. |
| `hard_configs.py` | The plausible-Hard dataset neighbourhood around public `m5` — one generator knob moved per config (`hB_width_down`, `hC_width_up`, `hD_t_shallow`, `hE_t_deep`, `hF_data_25x`; all **prompt-grouped**, whereas Hard is modulus-grouped — see README §1–2) — plus `manifest()` (an evaluator manifest in upstream's exact shape), `cpu_smoke_manifest()`, and `PUBLIC_DATA_ROOTS` aliases (`e1`, `e5`, `m5`, `m6`–`m10`). CLI: `generate`, `manifest`. Documents the arithmetic ceiling capping `examples_per_setting` near 28k for any 12-bit cell. |
| `audit_dataset.py` | Reads a generated dataset and reports what the manifest never states: **distinct training moduli per bit width** (6/21/14/57/42/169/148 valid RSA moduli at 10–16 bits), **per-modulus base coverage**, the **analytic no-reduction floor at every ladder rung** (`x^(2^T) < N`), the periodicity leak (prompts whose rung-`T` answer repeats a lower rung's), and per-modulus first depth-repeat. CLI over one or more `data_root`s, optional `--json`. Its outputs for `m6`–`m9` are in `results/audit/`. |
| `modal_rehearsal.py` | **The Modal harness.** App `old-dress-rehearsal`, volume `old-dress-rehearsal-data` — deliberately separate from the research node's `shared.py`. Image = `debian_slim(3.13)` + upstream's pinned deps + a clone of upstream at `4ceff95`. Functions: `generate_public` (runs `scripts/generate_datasets.sh` verbatim onto the volume), `generate_hard`, `audit`, `evaluate` (**H100**; shells out to `python -m benchmark.runner` with `CUDA_VISIBLE_DEVICES=0`, captures `RESULT_JSON=` and the depth-profile lines; `--arm baseline` runs upstream's reference submission), `evaluate_l4` (cheap training checks; tags prefixed `l4_` so they are never mistaken for tier-faithful numbers), and `collect`. Only smoke budgets (≤120 s) were run here; all real-budget numbers are hosted. |

## Results

| path | contents |
|---|---|
| `results/hosted/<tier>_<dataset>_<arm>[_vN]/` | One directory per hosted run (15: `easy_e5_*` ×4, `medium_m5_*` ×5, `medium_m6_*` ×3, `medium_m7_control_v2`, `hard_h1_*` ×2). `status.json` = `one-layer status --json` (full seven-rung profile on both profiles, per-split counts and losses, steps, timings, model state); `metrics.jsonl` = the `log_every=100` training curve; `submission.json` = submission id, SHA-256 of the exact file, and a change note. `medium_m6_closure_v3/` also holds a copy of the v3 file. |
| `results/audit/m6_m9_fixed_n.json` | `audit_dataset.py` output for the four fixed-`N` Medium sets used for the fresh-`x` readout (cohort sizes, floors, first depth-repeat). |

## Children

None.
