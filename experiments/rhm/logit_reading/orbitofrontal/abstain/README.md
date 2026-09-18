# abstain — the consumer: an abstain-or-commit choice gated by the value reading

**Up**: [`../README.md`](../README.md) (orbitofrontal) · **Written up in**: the parent's [§3](../README.md#3-what-consumes-it-the-expectation-gates-away-from-the-events)
· **Machinery, gates, runs and gotchas**: [`FILES.md`](FILES.md) · **Every number**:
[`results/tables.md`](results/tables.md) (the anchor venue, CPU on the banked striatum cells) and
[`results/tables_stream.md`](results/tables_stream.md) (the stream venue, every position of held-out windows) ·
**Donors**: [`../../striatum/`](../../striatum/README.md), [`../../coeruleus/gainloop.py`](../../coeruleus/gainloop.py),
[`../../basalis/gates2.py`](../../basalis/gates2.py)

At a query the actor answers, returning the outcome, or abstains at a fixed price; each gate's policy is a
quantile step function fitted on half the rows and read on the other half, against a per-stratum constant floor
and the realised outcome as ceiling, under four conditionings, with pure-noise gates and the binned out-of-sample
R² screen beside every AUC. This folder carries the code, the tables and the figures; the interpretation is the
parent's.
