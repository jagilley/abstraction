# shaped — a trunk fine-tuned on the goal, in place of the frozen next-token trunk

**Up**: [`../README.md`](../README.md) (orbitofrontal) · **Written up in**: the parent's [§2a](../README.md#2a-shaping-the-trunk-on-the-goal-sharpens-the-cost-reading-and-leaves-legality-alone)
· **Machinery, gates, runs and gotchas**: [`FILES.md`](FILES.md) · **Every number**:
[`results/tables.md`](results/tables.md) and the per-arm tables under [`results/per_arm/`](results/per_arm/) in
the banked striatum and junction formats · **Donor**: [`../../striatum/`](../../striatum/README.md)

Three arms of continued training from the 64k checkpoint on bit-identical data (task only, task plus next-token,
next-token only), then every striatum and junction readout, Part 1's altitude and Part 2a's detection on each.
This folder carries the code, the tables and the figures; the interpretation is the parent's.
