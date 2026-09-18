# regime — a world where a rule violation is evidence about a hidden state that predicts future cost

**Up**: [`../README.md`](../README.md) (orbitofrontal) · **Written up in**: the parent's [§2b](../README.md#2b-where-a-violation-predicts-future-cost-the-reader-reads-the-hidden-state-priced-by-its-belief)
· **Machinery, gates, runs and gotchas**: [`FILES.md`](FILES.md) · **Every number**:
[`results/tables.md`](results/tables.md) · **Donors**: [`../../striatum/`](../../striatum/README.md),
[`../../altitude/train_noisy.py`](../../altitude/train_noisy.py), [`../../basalis/`](../../basalis/README.md)

A two-state Markov corruption regime along the stream and an i.i.d. control at the same marginal rate; one
next-token trajectory per world; the exact running regime filter as the reference, the same reader applied to
both worlds; striatum's actor and critic on each; natural and surprisal-matched twins. This folder carries the
world, the trainer, the readout, the tables and the figures; the interpretation is the parent's.
