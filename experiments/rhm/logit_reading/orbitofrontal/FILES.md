# orbitofrontal — file index

**Up**: [`../FILES.md`](../FILES.md) (logit_reading) · **Writeup**: [`README.md`](README.md)

The super-node for the 2026-09-17 value-side round. No code lives at this level; every child carries its own
`FILES.md` with the commands of record, app ids, wall clock, peak RSS and volume paths. The interpretation of all
six is in this node's README; the children carry pointer READMEs only.

## Children

| folder | what it built | pointers |
|---|---|---|
| [`adaptation/`](adaptation/README.md) | The norm as a running estimate, within-subject: a diet as a row order over its mask, five online fitters, phases `lo→mid` / `hi→mid` / `mid→mid` in two families plus interleaved controls, read on norm's fixed rows and twins at 29–37 checkpoints, two order seeds. Donor `../striatum/norm/`. | [`FILES.md`](adaptation/FILES.md), `results/tables.md`[^omitted] |
| [`projection/`](projection/README.md) | Belief or representation: the critic refit on the model's output distribution with dimension-matched controls; the priced twin (a fixed map from the forecast difference to the revision difference); the critic's direction against the output layer's row space at steps 0 / 8k / 64k. Donor `../striatum/norm/`. | [`FILES.md`](projection/FILES.md), [`results/tables.md`](projection/results/tables.md) |
| [`shaped/`](shaped/README.md) | A trunk fine-tuned on the actor's goal in three arms (task, task + next-token, next-token only), then every striatum and junction readout, Part 1's altitude and Part 2a's detection on each. Donor `../striatum/`. | [`FILES.md`](shaped/FILES.md), [`results/tables.md`](shaped/results/tables.md) |
| [`regime/`](regime/README.md) | A world where a violation is evidence about a persistent hidden state that predicts future cost: a two-state Markov corruption regime, one trajectory per world with an i.i.d. control, the exact running filter as reference, striatum's actor and critic, natural and surprisal-matched twins. Donor `../striatum/`. | [`FILES.md`](regime/FILES.md), [`results/tables.md`](regime/results/tables.md) |
| [`abstain/`](abstain/README.md) | The consumer: an abstain-or-commit choice at a swept price, gated by each value-side object and each surprise-shaped scalar, coeruleus's step-function fit with floors, ceilings, four conditionings and basalis's screen; the anchor venue (CPU) and the stream venue (every position). Donor `../striatum/`. | [`FILES.md`](abstain/FILES.md), [`results/tables.md`](abstain/results/tables.md), [`results/tables_stream.md`](abstain/results/tables_stream.md) |

## Written up here, living elsewhere

| where | what |
|---|---|
| [`../../practice/voicing/tessitura/`](../../practice/voicing/tessitura/README.md) | The same three questions asked of the practice learner's own judge, on `voicing`'s machinery (default-off knobs, G-F 0.000e+00, byte-identical dumps): the judge's level per cycle, a fixed per-(slot, era) panel, the structural label per row; a CPU refit under outcome diets on the fixed trunk; two learner seeds. Lives with its machinery, per the arc's convention for [`perception/`](../../practice/perception/README.md). |
| [`../striatum/results/NOTES_2026-09-17_rereads.md`](../striatum/results/NOTES_2026-09-17_rereads.md) | The re-reads: the across-level contrast on the MLP critic and inside `(j, k*)` strata ([`tables_mlp_across.md`](../striatum/results/tables_mlp_across.md)); the norm round on trajectory seeds `a1_s43` and `a1_s44` with three-seed side-by-sides ([`norm/results/tables_s43.md`](../striatum/norm/results/tables_s43.md), [`tables_s44.md`](../striatum/norm/results/tables_s44.md)); a mean-and-variance critic ([`norm/results/tables_varhead.md`](../striatum/norm/results/tables_varhead.md)). |

## Edits to donors made by this round (all additive, all gated)

| file | change | gate |
|---|---|---|
| `../striatum/norm/task.py` | a `var_target` knob (`""` = off), threaded through `norm_sweep` | the banked `swap65k` 64k cell re-run with the knob off is identical on all 250 JSON leaves |
| `../striatum/norm/analyze.py` | `--mode varhead`, `--banked-actor` (another seed's actor column or `none`), `--outfile` | defaults reproduce the banked tables |
| `../striatum/analyze.py` | `--mlp-across` mode with `(j, k*)` cells; `addendum()` skips its banked-head columns when no head sits beside the checkpoint | the banked reduction reproduces line for line |
| `../striatum/addendum.py` | a `hexcess_sweep` coordinator | `hexcess_ckpt` unchanged |
| `../../practice/voicing/voicing.py` | the `# [tessitura]` hunks: three default-off knobs and `ts_gates_cpu`; `VO_MEMORY_MB` cut from 32768 to 16384 on a measured 7.0 GB peak | G-F 0.000e+00; TS-1 0.000e+00 at both seeds; TS-1b byte-identical dumps |

[^omitted]: Omitted from the mirror for size (a raw results dump over the per-file cap). Available on request.
