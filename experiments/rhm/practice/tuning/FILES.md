# tuning — File Index

Summarized in [README.md](README.md); design record in [SPEC.md](SPEC.md) (original spec +
the 2026-08-30 Gate-1 reshaping addendum). This is the complete listing.

## Code files

| File | What |
|---|---|
| `burst.py` | Span-confined uniform burst construction (per-step consumed draws, fixed shadow draws; RNG never touches the batch sampler) + the noise-aware exact-BP oracle (softened leaf evidence + output mixture) giving the model-free irreducible floor per ρ. Gate B-1…B-5. |
| `drift_ev.py` | Drift events for Gate 1: wraps `transpose.drift.resample_cells` (imported verbatim) — permanent cumulative level-3 rule-cell resampling on pool-refresh boundaries, drift ladder + null rung (fresh undrifted draw) for the matched-surprisal discipline. Gates D-1…D-4. |
| `tune_lm.py` | Gate 0: `wall`+`no_wall` co-resident with per-reader RNG isolation on the burst-augmented stream; the uncharged panel (per-view NLLs under clean/rotation/burst-ladder counterfactuals, `D_pair`/`D_self`/mirror control, shadow batches) and the FM stack (4 online lrs riding one forward pass + `frozen_prev`/`frozen_event`) at every 125-step checkpoint. `::gate` = the CPU structural gate. |
| `gate1_lm.py` | Gate 1 wave 1: the full three-event schedule (rotations + calibrated bursts + drifts), T\*-typed op maps on the online read (last-8-offered-batches vs own EWMA, dead zones at in-tag floors), reversible ops (`continue`/`merge`/`skip`), per-epoch eval/probe/exact-BP references, fp16 reader+FM checkpointing (stride 16 + event micro-grids). Known deviation: binds `pn_leaf` at epoch 0 (see README caveats). |
| `gate2_lm.py` | Gate 1 wave 2: re-derived dead zones, `rot_requires_out=False`, zero-lag drift micro-grid (epoch turns before the checkpoint), FM sign/lag grading (post-`g2c` fix: FM learns through skips), ABBA trial meter with read-charging, the priced 16-offset re-key search (+ dead mirror), `basis_sep` logged on every arm, `nll_by_shift` persisted unconditionally. |
| `gate1g.py` | Gate 1G Modal entrypoints over the banked checkpoints: per-layer ‖Δθ‖ + event×event alignment cosines (CPU), and the representation reads (RSA, PC/Procrustes angles, the address-matched sharp test and map-swap bracket control) on the panel's own eval batches, per epoch grammar. |
| `analyze_tuning.py` | Reduction for `tn0`/`g1a`/`g2a`/`g2c`: matched-surprisal Factor-T sections (bracketing discipline, pooled groups, sign counts), section 2b (counterfactual-vs-realized cross-check), 2c (mirror control), 2d (the movement decomposition, added retroactively), Factor-M rows, ledgers, per-event costs in floor multiples, figures. |
| `analyze_g1g.py` | Gate 1G reduction: the weight null (‖Δθ‖ ~ n^α fit + the direct n=125 datum), the two alignment floors (random-vector + post-merge empirical), the rep null, verdicts (reorient/gain/rebuild), re-index recovery, figures. Null designs documented in its module docstring. |
| `transparency.py` | Gate 0 gates: `--shadow`-vs-`--no-shadow` and FM-vs-no-FM trajectory identity; `--fidelity` = the five-instrument max\|Δ\| comparison against fwlm0/fwlm1/tsdB/ey0 over shared prefixes. |
| `transparency_g1.py` | Wave-1 fidelity + the per-arm first-op report (prints first-op steps rather than asserting, so a pre-event fire reads as a result, not a gate crash). |
| `transparency_g2.py` | Wave-2 cross-wave prefix licenses (each arm vs wave-1 `track`; `self_v2` vs wave-1 `self`; `g2c` vs `self_v2` divergence point) + donor twins. |
| `launch_detached.py` | Session-isolated detached launcher for `tune_lm` (donor pattern). |
| `launch_g1.py` | Gate 1 launcher: wave A (4 workers) then wave B chained in one detached session, DoP ≤ 4. |
| `launch_g2.py` | Wave 2 launcher: batch A then the rekey batch; also the `g2c` single-arm relaunch. See README's launcher-hazard note (Modal retry duplicates; last-writer-wins). |

## Runs

| Tag | What | Volume |
|---|---|---|
| `tn_calib` | Rotation-spike-vs-ρ calibration + donor fidelity check (17k steps, no bursts, no panel) | `rhm_practice_tuning/tn_calib/` |
| `tn0` | Gate 0 shadow panel: `pair` (wall+no_wall) + `skip` workers, uncharged everything | `rhm_practice_tuning/tn0/` |
| `g1_calib` | Drift calibration (`n_cells*` per maturity; null rung) + burst re-confirmation | `rhm_practice_tuning_g1/g1_calib/` |
| `g1a` | Gate 1 wave 1: `self`, `pair`, `sur_merge`, `sur_skip` + `track`/`sched`/`rand`/`deadpair` | `rhm_practice_tuning_g1/g1a/` (+ `ckpt/`) |
| `g2a` | Wave 2: `self_v2`, `self_fm`, `sur_fm`, `self_abba`, `self_rekey`+`rekey_dead` | `rhm_practice_tuning_g2/g2a/` |
| `g2c` | The fix round: `self_fm` with the FM learning through skips | `rhm_practice_tuning_g2/g2c/` |
| `g1g` | Offline checkpoint analytics (weights + representations), ~0.03 GPU-h | `rhm_practice_tuning_g1g/g1g/` |

Fetched copies + figures under `figures/<tag>/`; every tag's exact config in its `setup.json`.
Figures: `tn0/fig1–fig6` (competence / Factor T / ladder / Factor M / FM-lr / movement),
`g1a/fig1_typing_and_floors`+`fig2_ledger`, `g2a/fig1_wave2`+`fig2_two_wave_ledger`+
`fig3_gate1_complete`, `g2c/fig1_gate1_complete`, `g1g/fig1_absorption_locus`+
`fig2_remap_alignment`+`fig3_representation`+`fig4_sharp_reindex`.

## Gate records

All PASS at max|Δ| = 0.0 unless noted: panel-moves-nothing and FM-moves-nothing (`tn0` smokes,
22 ckpts × 3 arms × 15 instruments); donor fidelity vs fwlm0/fwlm1/tsdB/ey0 (`tn0`, `g1a`,
`g2a`/`g2c`; the one apparent tsdB failure was a mis-set comparison cap — tsdB's first ABBA
collapse begins at s1000 — verified by exact divergence at s1125); cross-worker identity
(`wall_skip`-vs-`wall`, `self_rekey`-vs-`track`); cross-wave prefixes (each wave-2 arm ≡ wave-1
`track` to its first op; `self_v2` ≡ wave-1 `self` through s8050 in weights); post-fix
`gate2_lm` ≡ pre-fix on `self_v2` (FM-only change); structural gates B-1…B-5 (burst) and
D-1…D-4 (drift); the drift null rung at −0.0000.
