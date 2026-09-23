# grokking/basis — the endogenous basis mint (pointer)

**Up**: [`../README.md`](../README.md), which writes this node up in **§2 "Basis (`b1`): the basis mints itself from the net's
own function, with a label-free certificate"** and carries it into "The update" items 3 and 4. This folder holds the run, not a
separate writeup.

What runs here: on the banked `m1` snapshots, the net's own function a ↦ net(a, g) is read as an operator on the 97 unordered
symbols at six generators and eigendecomposed; the real invariant subspaces are the candidate units; cross-generator agreement
is the endogenous certificate; overlap with the true DFT characters and the eigenvalue phase are the oracle, logged and never
consumed; then the parent's walk and compiles run over the recovered basis. No retraining.

- Decisions, gates, designed checks, falsifiers and defects: [`NOTES.md`](NOTES.md)
- File index: [`FILES.md`](FILES.md)
- Table of record: [`figures/basis_b1_table.txt`](figures/basis_b1_table.txt) · figure: `figures/basis_b1_read.png`
- Commands of record: [`results/RUN_basis.sh`](results/RUN_basis.sh)
