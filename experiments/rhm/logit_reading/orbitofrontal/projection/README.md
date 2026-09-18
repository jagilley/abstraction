# projection — does the value projection read the belief or the representation?

**Up**: [`../README.md`](../README.md) (orbitofrontal) · **Written up in**: the parent's [§4](../README.md#4-belief-or-representation-the-norm-is-public-the-response-is-private)
· **Machinery, gates, runs and gotchas**: [`FILES.md`](FILES.md) · **Every number**:
[`results/tables.md`](results/tables.md) (six cells; the across-checkpoints sections carry the readout span and
the priced twin's floor) · **Donor**: [`../../striatum/norm/`](../../striatum/norm/README.md)

The critic refit on the model's output distribution in place of the residual stream, with dimension-matched
controls; a fixed linear map from the model's forecast difference on same-prefix twins to the critic's revision
difference; the critic's direction against the output layer's row space at steps 0, 8k and 64k. This folder
carries the code, the tables and the figures; the interpretation is the parent's.
