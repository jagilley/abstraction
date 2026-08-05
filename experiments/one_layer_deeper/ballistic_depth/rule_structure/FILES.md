# Files — `rule_structure`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (ballistic_depth)

## Code files

| file | purpose |
|---|---|
| `dlog_probe.py` | **The probe, on Modal; loads checkpoints and trains nothing.** Owns `group_coords` (brute-force discrete log per CRT factor, asserted to be a bijection onto the grid *and* to make squaring the doubling map — a silently wrong coordinate would produce a confident null), the four readouts (`_fourier_conc` over the full group and the QR sublattice, `_translation_r2`, the single-linear-map fit `Enc(x²) ≈ M·Enc(x)` scored on held-out bases, and the model's own operator on held-out bases), and the three calibrations: a permutation null, an untrained-encoder control, and `_synthetic_group_emb` — the positive control that says what "found the group" scores at a range of mode counts and SNRs. The positive control is load-bearing: against the permutation null alone a functionally nil effect clears z > 30. |
| `dlog_probe_table.py` | Local, no GPU. Cross-tag summary — one row per (training tag, arm, variant) with modulus and reachable-state count so structure can be read against scale, the synthetic control ladder, and the multiple by which the best observed run falls short of the weakest detectable group signal. Reads `../results/<tag>/dlog_probe_seed*.json`. |

Results live in the parent cut's [`../results/`](../README.md#reproduce) tree, keyed by the
*training* tag whose checkpoints they were read from — this node adds no training runs of its own.
Model construction is imported from [`../ballistic_depth.py`](../ballistic_depth.py) and the task
from [`../../squaring_mod.py`](../../squaring_mod.py), per
[STRUCTURE.md](../../../../STRUCTURE.md) — code lives at the lowest node that shares it.
