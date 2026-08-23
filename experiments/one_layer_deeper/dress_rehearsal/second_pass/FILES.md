# Files — `second_pass`

**Up**: [README.md](README.md) (this node — the writeup) · [../README.md](../README.md) (dress_rehearsal) · [../../README.md](../../README.md) (one_layer_deeper)

## Code files

None at this level. Code lives in the children; the third track's code lives with its donor (below).

## Children

| child | one-liner |
|---|---|
| [`digit_port/`](digit_port/NOTES.md) | **Track A — the per-digit port.** `submission.py` (digit-grid carrier, loop-on-`T`, `N` as per-slot digit embeddings; arms `digit` / `digit_learned` / `digit_closure` / `digit_carry`; `STATE_MODE` ∈ soft/residual/norm/highway, `OP_CROSS`, `CTX_SCOPE`), `emit_arm.py`, `probe.py` (layout/gather/scatter/identity-at-init checks), `save_run.py` / `read_run.py` (hosted records), `modal_sweep.py` (600 s Medium reproduction on our H100, app `old-digit-port`). `NOTES.md` is the run-by-run record of 11 hosted runs and 6 sweeps; [`FILES.md`](digit_port/FILES.md) the code index; `results/hosted/<tier>_<dataset>_<arm>/` the 11 hosted run records (force-tracked, as the parent's are). |
| [`board_note/`](board_note/README.md) | **Track C — the externally-readable note**: the 6/768 floor, the dense 12-bit cell (58% coverage), periodicity, the `eps ≲ 9e-4/T` ladder; `discord.md` is the ≤300-word version. Board-and-task-arithmetic only. Not posted. |
| [`../../rule_acquisition/staged_reduce/terminal_only/`](../../rule_acquisition/staged_reduce/terminal_only/NOTES.md) *(linked, not a child directory)* | **Track B — the staged forward pass under terminal-only supervision.** `terminal_only.py` (verbatim fork of `staged_reduce.py` with a diff table), `reprobe.py` (depth / restart / stagefn / crossprobe), `analyze.py`, `NOTES.md`, [`FILES.md`](../../rule_acquisition/staged_reduce/terminal_only/FILES.md), `results/<tag>/`. Kept under its donor because it imports it by absolute module path and is that node's question. |

## Results layout

Hosted: `digit_port/results/hosted/<run>/{status.json,metrics.jsonl,submission.json}`. Local sweeps: Modal volume `old-dress-rehearsal-data:/digit_results/<tag>/` (not fetched into the repo; the reads are in `digit_port/NOTES.md`). Track B: volume `one-layer-deeper-data:/staged_reduce/terminal_only/<tag>/` with local copies under `terminal_only/results/<tag>/`.
