# FILES — `duplex`

The machinery record: every code file, every knob, every gate, every arm, every run. Decisions and
withdrawn diagnoses are in [`DESIGN.md`](DESIGN.md); the brief is [`SPEC.md`](SPEC.md). The writeup is the parent's, [`../README.md`](../README.md); [`README.md`](README.md) here is a pointer.

**Up**: [`../FILES.md`](../FILES.md) (`aliquot`) · [`../../FILES.md`](../../FILES.md)
(`sotto_voce`) · [`../../../FILES.md`](../../../FILES.md) (`voicing`)

## Code files

| file | purpose |
|---|---|
| `duplex.py` | the job. `probe_seed` (L4) loads one banked dump (`vo_heads.pt`, `vo_rows.npz`, `vo_rows_meta.json`, `results.json`), reproduces `overtone::post_write_probe` on the banked and on a never-trained trunk (gates G0/G1), shapes nine more trunks, and re-reads the probe, the world-model diagnostics and the transfer split on the identical rows. `sweep` is the CPU coordinator that `.starmap`s the two seeds across two containers. Nothing in `voicing.py`, `sotto_voce.py`, `aliquot.py` or `overtone/analyze_dump.py` is touched |
| `analyze_duplex.py` | the local reduction -> `figures/<tag>_reduction.txt`: (G0) the estimator gate, (G1) the reproduction gate against the banked overtone dump cell by cell, (a) the probe per trunk and diet, (b) the read of record per level, (c) the world-model diagnostics, (d) the transfer split, (e) the shaping's own trajectory and the cost |

## Docs

| file | purpose |
|---|---|
| `SPEC.md` | the orchestrator's brief, verbatim |
| `DESIGN.md` | the decisions, the two gates, the cost, and every withdrawn diagnosis beside its correction |
| `FILES.md` | this file |

## Imported, unchanged

| from | what |
|---|---|
| `rhm/practice/voicing/voicing.py` | `ov_irls` (gate G0's reference estimator), `build_critic` (the banked critic's architecture) |
| `rhm/practice/native/span/span_net.py` | `trunk` (the pooled read), `slot_count` |
| `rhm/practice/ratchet/macros.py` | `parse_features` (the level-1 parse with the `mask_block` idiom), `exact_features` (the oracle target), `true_tables` / `make_macro` / `to_device` / `macro_features` (the nested DP parse) |
| `rhm/rhm_generative_planner.py` | `_build_generator` (the plant), and `_train_generator`'s loss form (reimplemented inline so the fresh-window stream and the mask plan can be fixed and matched across arms; the form is verbatim) |
| `rhm/rhm_data.py` | `generate_rules_distinct`, `build_inverse_maps` |
| `rhm/rhm_sculpt_planner.py` | `_sample_pool` (the fresh corpus) |
| `rhm/shared.py` | `image`, `volume`, `DATA_DIR`, `NumpyEncoder` |

## The knobs (module constants in `duplex.py`; nothing is a CLI flag but the tag, the smoke and the seed list)

| constant | value | meaning |
|---|---|---|
| `SEEDS` | `s0 -> ov_s0b/ovt_comp_pr_sh`, `s2 -> ov_s2/ovt_comp_pr_dis` | the two banked dumps, and nothing beyond them (SPEC) |
| `OV_RAND_SEED` | `20260915` | **overtone's** control-two init seed, so the `rand` row IS `rand_slot`. `aliquot`'s in-loop twin uses `20260918`; that is a different draw and is deliberately not used here |
| `PROBE_RIDGE`, `PROBE_ITERS` | `1e-2`, `40` | `overtone::_fit_probe`'s literals, unchanged |
| `SHAPE_LR` | `1e-4` | the run's own `gen_lr` — the rate at which the loop continues this plant |
| `SHAPE_STEPS` | `(100, 400, 1600)` | 2.5% / 10% / 40% of the paid arm's ~4,000-step plant budget |
| `SHAPE_SLOT_BATCH` | `32` | outcome rows per buffer per step; 64 per slot across the two doors, the critic's own batch |
| `SHAPE_INFILL_BATCH` | `256` | the run's own `batch_size` |
| `SHAPE_CLIP` | `1.0` | `_train_generator`'s clip, on every arm alike |
| `CORPUS_SEED_TRAIN`, `CORPUS_SEED_EVAL` | `777001`, `777002` | the fresh corpus draws; distinct from `train_seed = 1` and from `seed + 31337` |
| `N_CORPUS_TRAIN`, `N_CORPUS_EVAL` | `32768`, `8192` | pool sizes |
| `XFER_CAP` | `20000` | the transfer split's matched fit size per group |
| `WM_INFILL_BATCHES`, `WM_INFILL_BATCH` | `16`, `512` | the fixed held-out infill plan |
| `WM_PARSE_N`, `WM_DP_N` | `4096`, `1024` | the parse and nested-DP diagnostics' window counts |
| `WM_SEED` | `909` | the diagnostic plan's seed — one plan, reused on every trunk |
| `--shape-levels` | flag, default `""` | empty = the outcome loss hears every slot (`du0`, `du1`); `"2,3"` = it hears the L2/L3 slots only (`du2`, DESIGN §5a). Touches the shaping diet and nothing else |
| `--modes`, `--steps` | flags, default `""` | which of `both,out,infill` and which step counts to run; empty = all three and `SHAPE_STEPS` |
| `--dedup` | flag on `sweep`/`probe_seed` | off for `du0`, on for `du1`. Drops from the shaping diet **and** from the probe's own fit every training row whose post-write configuration appears among the held-out rows of any buffer (DESIGN §2.5). The overlap is measured and logged on **every** run whether or not the filter is applied |

## The trunks (11 per seed)

| name | what |
|---|---|
| `frozen` | the banked trunk, `load_state_dict` and no optimiser. Its `post` column must equal overtone's `post_slot` (G1) |
| `rand` | never trained, same architecture, `manual_seed(20260915)` after the trained core is loaded, in overtone's order. Its `post` column must equal overtone's `rand_slot` (G1) |
| `both@100/400/1600` | outcome BCE + masked-infill CE, weights 1 and 1 |
| `out@100/400/1600` | outcome BCE alone |
| `infill@100/400/1600` | masked-infill CE alone, matched steps |

## The reads (4 per trunk per buffer)

| name | overtone's name | what |
|---|---|---|
| `pre` | `pre_slot` | `[pooled(obs).mean ; pooled(obs)[span].mean]` — the write not in the input |
| `pmean` | `post_mean` | `pooled(obs+write).mean` |
| `post` | `post_slot` | `[pooled(obs+write).mean ; pooled(obs+write)[span].mean]` — **the read of record** |
| `postm` | — | `post` with the first block outside the span masked (`parse_features`' idiom, `aliquot`'s read) |

plus `mlp` (the banked critic recomputed through *this* trunk) and `dp` (the free prior from the
dump, trunk-independent by construction and therefore also a consistency check: it must be
identical on every trunk row).

## Gates

| gate | what it asserts | how it is read |
|---|---|---|
| **G0** | the torch IRLS is `voicing::ov_irls` | `max |Δw|`, `max |Δscore|`, `Δ AUC` on one real buffer, section (G0) of the reduction. Smoke: 2.6e−11 / 2.9e−6 / 0.0 |
| **G1** | `frozen` and `rand` reproduce the banked overtone dump cell for cell | `analyze_duplex.py` parses `overtone/figures/ov_*_dump.txt`'s S1b table and reports `n`, `max |Δ|` and `mean |Δ|` per (trunk, read). Smoke: every compared cell agreed to the printed 3 decimals |
| **matched streams** | `both@N` / `out@N` share their outcome rows; `both@N` / `infill@N` share their windows and masks | the two arms' logged `infill_ce` at step 1 agree to 4 decimals (smoke), and the RNGs are keyed on `N` alone (DESIGN §3.4) |
| **G1b** | the same gate at the median | this node's `frozen`/`rand` medians against the `MEDIAN over slots` lines overtone printed itself. `du0`: **max |Δ| 0.0012 / 0.0015 over all 36 median cells**, 22 of 24 overtone printed matching exactly |
| **dp invariance** | the prior column is trunk-independent | section (a): the `dp` cell must be identical on every trunk row of a seed. Holds on both seeds and all 11 trunks |
| **cross-tag infill identity** | `infill@N` is the same trunk under every tag | the infill stream is keyed on the step count alone and the outcome diet never enters it, so `du0`/`du1`/`du2` `infill@400`, `infill@1600`, `frozen` and `rand` must have identical fingerprints. **Spread 0.00e+00 on all 8 rows** (section (f)) |
| **overlap instrument** | the exact-duplicate channel is measured, not assumed | `overlap` in the JSON and the container log. `du0`/`du1` agree with an independent numpy measurement on the local dumps to the row: filed 5994/22937 and 5542/22474, probe 53/19774 and 127/19558 |

## Runs

| tag | what | app | cost |
|---|---|---|---|
| `dusm` | smoke, seed 0, 8 buffers, 20 steps | `ap-oPxj29JDalYiXhbDqhzUGU` | 23 s, 1 L4 |
| `du0` | the run of record, both seeds, 11 trunks each; the protocol as overtone and the loop have it | `ap-Af1vhccrAyTwl6IlSKLOOI` | 766 s / 732 s, 2 L4 in parallel |
| `dusm2` | smoke of the `--dedup` path | (same app pattern) | 25 s, 1 L4 |
| `du1` | the same with DESIGN §2.5's exact-duplicate filter on | `ap-IuKBShaEHbn7tFiIxGin4K` | 780 s / 743 s, 2 L4 in parallel |
| `du2` | §5a: the outcome loss restricted to the L2/L3 slots, `both`+`infill` at 400/1600, dedup on | `ap-8HUhm9otsf7XY2S5RjOmU8` | 365 s / 379 s, 2 L4 in parallel |

## Artefacts

| path | what |
|---|---|
| volume `/data/rhm_practice_duplex/du0/{s0,s2}.json` | every number, per buffer per trunk: the four reads' held-out AUC, `mlp`, `dp`, the draw split (`probe/u`, `probe/d`), the world-model diagnostics, the transfer split, the shaping trajectory, the fingerprints |
| volume `/data/rhm_practice_duplex/du0/{s0,s2}.log` | the container's own log |
| `figures/du0_reduction.txt` | the reduction for the protocol as overtone has it — carries gates G0, G1, G1b |
| `figures/du1_reduction.txt` | the reduction with the exact-duplicate filter on — the filed column to read |
| `figures/du2_reduction.txt` | the reduction with the outcome loss heard at L2/L3 only — section (d) there is the transfer question as posed, and section **(f)** is the three-tag side-by-side (built with `--vs du1,du0`) |
| `results/launch_du{0,1,2}.log` | the launch logs |

## Gotchas

- **A filed row's post-write configuration is its trajectory's FINAL configuration**, while
  `hold_code` hashes the PRE-write context. One trajectory that wrote at several slots therefore
  contributes several rows with the same post-write input and the same verdict on *opposite sides*
  of the audit split — 26.1% / 24.7% of held-out filed rows at the two seeds. Anything trained or
  fitted on the post-write input inherits that. See DESIGN §2.5; `--dedup` closes it exactly for
  filed rows and `du1` is the tag with it closed.
- **`vo_heads.pt` and `vo_rows.npz` are on the volume only** — gitignored, ~3.2 MB and ~3.5 MB per
  arm. The local `figures/ov_s0b/...` mirror in `overtone/` carries `results.json` and
  `vo_rows_meta.json` but not those two, so the job has to run where the volume is mounted (which
  it needs the GPU for anyway).
- **The level-1 parse accuracy's ceiling is ~0.65, not 1.0**, because `_train_generator` computes
  its loss on masked blocks only and the head is never trained to report a block it can see
  (`calr_s0`: 0.63). Read it as a between-trunk contrast, never as an absolute competence.
- **The nested DP parse at level 5 allocates `batch × 262144` floats**, so its batch is sized from
  the true table's entry count (`16e6 // n_entries`, floor 8) and the loop is wrapped so an OOM at
  the top rung skips that rung rather than failing the arm.
- **Do not background a waiter with an inner `&` inside a `run_in_background` Bash call.** The
  harness then tracks the wrapper, which exits immediately, and the completion notification fires
  seconds after launch with the job still running. The loop itself is what must be backgrounded by
  the tool.
- **Moving the launch log while the run is in flight breaks a path-based waiter** (the writer keeps
  its file descriptor, so the file grows at the new path while the waiter greps the old one). This
  node moved `duplex/` under `aliquot/` mid-run and had to re-arm.
