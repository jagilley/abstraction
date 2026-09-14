# Authorship — file & reproduction index

Node: `experiments/a2a_forward/fsm_part3/authorship/`.
Parent: [`../../confabulation/README.md`](../../confabulation/README.md) / [`../../confabulation/FILES.md`](../../confabulation/FILES.md).
Design decisions: [`DESIGN.md`](DESIGN.md).

Can a transformer tell which parts of its context it wrote itself? A standalone
mini-project on the parent's battery: the report target is per-position **authorship**
(`MINE`) rather than the parent's forward-model residual, and the parent's `IMPL` /
`IMPL_COS` targets are carried along on the same sequences and positions as the positive
control.

## Code files

| File | Purpose |
|---|---|
| `authorship.py` (`authorship_test`) | **The whole job.** Loads (or trains) the parent's `ol` wake checkpoint; writes a sampling loop for M (`a2a_forward/model.py` has none); builds three held-out report sets that mix M-written and corpus-written spans; re-reads them through M in an ordinary teacher-forced pass; and runs the battery — report heads `SELF_mlp` / `SELF_seq`, the capacity-swept observer ladder `O_input` / `O_lik` / `O_io` / `O_io_nolik` with a half-data control, the `O_act` activation-access ceiling, and the `LIK_pos` likelihood floor — on `MINE`, plus `IMPL` / `IMPL_COS` with their `ens_cos` and residual-structure guards. Imports the parent's `confabulation.py` helpers (k-means, report heads, residual diagnostics) rather than copying them |
| `reduce.py` | Local reduction of one run: reads `results.json` + `predictions.npz` from the volume and writes `figures/<tag>_reduction.txt` plus four PNGs (the battery per set; the `MINE` vs `IMPL` dissociation on the same positions; AUC against how much of a span has been seen; M's own log-probability of the token by author) |
| `__init__.py` | Package marker so `a2a_forward.fsm_part3.authorship` imports under Modal's `add_local_python_source("a2a_forward")` |

## Reproduce

```bash
cd experiments/                      # the chromatic workspace holds /data/tokens

# wiring check (~3 min; numbers meaningless)
modal run a2a_forward/fsm_part3/authorship/authorship.py::authorship_test --smoke

# the run (~85 min on one L4 including 10k wake steps; peak RSS 11.5 GB.
# Trains the wake model only if the parent's `ol` checkpoint is not on the volume)
modal run --detach a2a_forward/fsm_part3/authorship/authorship.py::authorship_test \
    --tag main

# reduce (local)
modal volume get language-reduction-data \
    /a2a_forward/confabulation/authorship/main /tmp/authorship_main
python a2a_forward/fsm_part3/authorship/reduce.py --dir /tmp/authorship_main --tag main
```

Results: `language-reduction-data:/data/a2a_forward/confabulation/authorship/<tag>/`
(`results.json`, `predictions.npz`). The wake checkpoint is the parent's, at
`/data/a2a_forward/confabulation/wake_ckpt/ol_L4H4D256_P10000000_T128_s10000_post_block0topost_block2_inj1_fm2x64x2_seed42.pt`.

## Notable flags

| Flag | Default | Why you would change it |
|---|---|---|
| `--temperature` | `1.0` | The honest case: M's tokens are drawn from M's own distribution. Lower values make M's text trivially recognizable by likelihood |
| `--temperature-lo` | `0.7` | The second (light-ladder) mix set, run for the record |
| `--prefix-len` | `24` | Corpus context M generates from. Excluded from scoring in every set — it is always corpus and always at the front |
| `--span-lens-str` | `8,12,16,24,32` | Span-length distribution, shared by both classes so length never predicts the class |
| `--n-report-sequences` | `3000` | The parent's report-set size |
| `--inst-cap-str` | `16:0.5` | The instrument FM for the `IMPL` control (the parent's default capacity, 10.6% of the predicted blocks). Only one capacity: the sweep is the parent's question |
| `--skip-impl` | off | Drops the `IMPL` / `IMPL_COS` positive control and the instrument entirely (~40% of the job) |
| `--observer-caps` | `1:64,2:128,4:256` | The ladder; the last point matches M's own architecture |
| `--cond` | `ol` | The arm. `cl` would need the parent's injected re-read to be meaningful; nothing here turns on the loop |
| `--refresh-wake` | off | Retrain the wake model even if the checkpoint exists |

## Gotchas

- **`O_io` is handed per-token likelihood features here**, unlike the parent's. Without
  them the observer would have to join M's distribution at *t−1* with the identity of the
  token at *t* through attention it was never trained for, and the resulting shortfall
  would be an input-format artifact rather than a fact about access. `O_io_nolik` keeps
  the parent's version in the same run so the size of the deviation is a reported number.
  See [`DESIGN.md`](DESIGN.md) §3.2.
- **`SELF_seq` exists because `MINE` is a span-level target.** Comparing the parent's
  per-position MLP against bidirectional sequence observers would rig the comparison.
  `SELF_mlp` is kept so the `IMPL` rows stay comparable to the paper.
- The prefix is unscored, and the `IMPL` control is scored on **exactly the same
  positions** as `MINE`, so the two rows are a within-run contrast rather than a
  comparison across data.
- The wake checkpoint key is the parent's. For the `ol` arm the co-trained forward model
  has no gradient path into M, so any `ol` checkpoint at this config holds the identical
  M; the checkpoint is written via a temp file plus `os.replace` because a sibling job may
  be writing the same path.
- Peak RSS in the smoke was 5.8 GB at `N=300`; the job requests 16 GB, sized from the
  N-dependent activation caches at `N=3000` (the run used 11.5 GB).
- **`predictions.npz`'s saved `_offset` is superseded.** It counts class runs from
  position 0 and so does not reset at the prefix boundary, which makes offset partly
  predict the class (a corpus span starting right after the always-corpus prefix inherits
  the prefix's run length). `reduce.py` recomputes the offset inside the scored region
  from the saved labels and uses that; `_offset` is kept only for provenance.
