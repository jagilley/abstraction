# timbre — pp1, pp3 and pp5 re-read on richer reader forms of the same pooled state

**Up**: [`../README.md`](../README.md) (preplay, whose claims this re-reads) · **Written up in**:
[`../../rubato/README.md`](../../rubato/README.md) §2 and its One-liner (the session's super-writeup, 2026-09-23 → 24) ·
**Decisions, gates, defects**: [`NOTES.md`](NOTES.md) · **Machinery record**: [`FILES.md`](FILES.md) · **Every
number**: [`figures/timbre_reduction.txt`](figures/timbre_reduction.txt) (the cross-reader tables of record, tag
`tb1`) and `figures/rr1_<reader>_reduction.txt` / `figures/rr5_<reader>_reduction.txt` (pp1's and pp5's own reducers,
unchanged, on each reader's columns).

The banked pp1, pp3 and pp5 fires re-read with the ridge of record, the plant's block-head log-softmax appended, a
128-unit MLP on the pooled state, an MLP on the belief alone, a belief-only ridge and a per-slot ridge, each refit on
pp1's shared bank subsample through the shaped, frozen and twin trunks, arm 0 bit-identical to the banked columns
in-container. Offline, nothing paid in the loop. This folder carries the code and the facts; the interpretation is in
the rubato writeup.
