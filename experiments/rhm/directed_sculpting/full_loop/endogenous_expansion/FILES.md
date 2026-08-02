# Files — `endogenous_expansion/`

Node README: [README.md](README.md) · Parent: [`../README.md`](../README.md) ·
Parent files: [`../FILES.md`](../FILES.md)

## Code files

| File | Purpose |
|---|---|
| [`endo_expansion.py`](endo_expansion.py) | The target-source ladder. Two Modal entrypoints: `calibrate` (the gate — scores every candidate target source against the exact DP on a frozen move probe, and sets both control corruption rates) and `endo_expansion` (7 arms × 10 rounds, one variable: where `belief_update`'s `gkstar` comes from). Carries `critic_target` (the endogenous MC-critic target on materialised next states), `policy_top1` (the `mirror` arm's target, and the wirehead discriminator when scored on the probe), `make_move_probe` / `score_moves` (the frozen exact-Δ`d*` instrument, held out of every loss), `beam_with_confidence` (ballistic success alongside what the loop *believes* it achieved), `refresh_critic` (policy-iteration refresh of the grading critic from paid rollouts), and `frontier_probe` (the §8 residual decomposition, absorbed by the fresh FM, on its own RNG stream so it cannot perturb the run). |
| [`aggregate.py`](aggregate.py) | Local, CPU-only. Four panels: EXPANSION (all four readouts, paired per-seed against the same seed's `frozen` floor, plus reproduction of the published 2×2 rows), BRACKET (the critic against both fidelity-matched external controls, with the fidelity match verified as delivered), WIREHEAD (the pre-registered self-vs-world discriminator, cross-world grading, and the metering ledger), READOUT BAKE-OFF (every candidate representation-side metric rank-correlated against the known functional ordering — restricted to the four target arms, since across all seven the frozen/dense floor manufactures a positive correlation and hides the inversion). Accepts runs by **schema, not filename**: a sibling session writes a different design under the same `endo_expansion_*` tag prefix on the shared volume, and those files glob identically. |

## Results

| Path | What |
|---|---|
| `figures/endo_expansion_endo_s{1,2,3}/` | The ladder, 3 seeds — the primary run (§4–§7). |
| `figures/endo_expansion_frontier_s{1,2,3}/` | Independent replicate of the same 3 seeds, adding the §8 residual decomposition. Doubles as the run-to-run noise measurement (§9). |
| `figures/endo_calibrate_cal_v1/` | Calibration gate output (`--quick`). |
| `figures/AGGREGATE.txt` | `aggregate.py` over the primary run. |
| `figures/AGGREGATE_frontier.txt` | `aggregate.py` over the replicate, including the readout bake-off. |

## Load-bearing gotchas

- **The `kstar_agree` ceiling is 0.915, not 1.0.** Mixture rendering is stochastic, so two runs of
  the same exact DP disagree a few percent of the time on the same state. Read every agreement
  number against that; `gain_frac` is the robust currency and is what the controls are matched on.
- **Evaluative-mode arms are not bit-reproducible across containers.** `frozen`/`dense` are; the
  four target arms diverge from round 1 (§9). Run-noise on PR is comparable to across-seed sd.
- **`own` == `base` by construction** in a static run — the informative cross-world columns are the
  two held-out `novel` realisations.
- **`belief_success` / `wirehead_index` are unreliable** for arms whose in-loop value saturated at
  0, and are inflated for `frozen` whose value was never updated. Use the frozen move probe's
  realised Δ`d*` as the wirehead discriminator instead (§7).
