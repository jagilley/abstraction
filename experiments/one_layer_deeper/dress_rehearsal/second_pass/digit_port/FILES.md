# Files — `digit_port`

**Up**: [NOTES.md](NOTES.md) (this node) · [../README.md](../README.md) (`second_pass` — the writeup) · [../../README.md](../../README.md) (`dress_rehearsal`)

## Code files

| file | purpose |
|---|---|
| `submission.py` | **The per-digit port.** One self-contained competition submission exporting `SUBMISSION`. The recurrent state is a fixed-width **digit grid** (slot `j` = the `j`-th digit from the right, zero-padded left, each slot a distribution over the vocabulary); the initial state is `x` right-aligned; a tied operator (4 layers of slot self-attention + cross-attention to an `N`-field-only encoder + MLP) maps grid to grid and is applied exactly `T` times, `T` parsed from the prompt tail; the grid after `T` steps is scattered to where `target_positions` reads. Abacus positional scheme (field id + within-field index from the right). Operator residual branches zero-initialised and the digit head tied to the state embedding, so the step starts as a near-identity copy. Loss = terminal CE + a zero-padding convention term on grid slots above the answer width. Four arms on the `ARM` line: `digit` (gather-seeded), `digit_learned` (learned slot-query seed instead of the gather), `digit_closure` (+ label-free sharpness and hard-requantized re-entry), `digit_carry` (+ a continuous residual channel — the ablation that says whether the digit bottleneck is doing the work). Legality reading in the docstring; the aggressive part is flagged in `NOTES.md`. |
| `emit_arm.py` | Rewrites `submission.py`'s single `ARM = "…"` line into `<out_dir>/<arm>/submission.py` and prints the SHA-256 of the emitted bytes, so arms cannot drift and the exact submitted file is identifiable. Checks the 256 KiB limit. |
| `probe.py` | Local CPU checks against real generated prompts: that the layout parse and right-aligned gather recover `(N, x, T)` exactly, that the grid→`target_positions` scatter round-trips, and that the operator is a near-identity copy at initialisation (the `head_scale` calibration). Not part of any submission. |
| `save_run.py` | Persists one hosted run into `results/hosted/<tag>/` — `status.json` (full seven-rung profile on both depth profiles), `metrics.jsonl`, `submission.json` (id, SHA-256 of the exact file, change note). Same layout as the parent node's. |
| `read_run.py` | Reduces saved `status.json`s to the deciding numbers: per-split counts and losses, both seven-rung profiles, steps, final train loss, timings. |

## Results

| path | contents |
|---|---|
| `results/hosted/<tier>_<dataset>_<arm>[_vN]/` | One directory per hosted run. |

## Children

None.
