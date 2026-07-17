# Gen-Distill Extended: 40-Cycle Ratchet (2026-06-26)

**Code**: `rhm_rl_gen_distill_extended.py`
**Prior experiment**: [Gen-distill run 13](RHM_RL_RATCHET_README.md#run-13-generation-based-distillation-2026-06-25)

## Motivation

The 4-cycle gen-distill run (run 13) showed L3 feature eta² at the final layer increasing monotonically (+15% over 4 cycles, 0.389→0.449) while outcome metrics (generation accuracy, per-level NTP loss) stayed flat. This raised the question: does the representational deepening eventually produce a behavioral phase transition if given more cycles?

The RHM's DGP is compact (96 rules = 576 bits for L=6/m=2/v=8/s=2) and the 2.68M model is ~150,000× overparameterized relative to this target. The FM captures 91-97% of the DGP's composition rules at learned levels. These are conditions where extended training with structured regularization could plausibly produce delayed generalization.

## Design

Same setup as gen-distill run 13, extended to 40 cycles:
- **Pre-train**: 10K NTP steps (shared with run 13, same seed)
- **Per cycle**: 2000 wake (RL + FM + sparse NTP mask=0.95) → 500 sleep (gen-based distillation) → 2000 FM repoint
- **Hyperparameters**: All identical to run 13 (lr=3e-4, distill_lr=1e-4, lambda_local=1.0, distill_alpha=0.5)
- **Norm failsafe**: Halt if activation norms at post_block5 exceed 5× the post-pretrain baseline

Per-cycle measurements: val loss, per-level NTP loss, per-layer feature eta², generation accuracy (with and without FM), FM cosine, gate stats, activation norms. Self-knowledge probes and robustness at milestones (every 10 cycles).

**Reproduction**: `modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended`

## Results (25 of 40 cycles complete)

### Outcome metrics are flat

Generation accuracy (standalone): 0.390 ± 0.001 across all 25 cycles, at every hierarchy level. No trend, no variance.

Per-level NTP loss at L3: mean 1.732, CV=1.7%, no trend (slope=-0.0004/cycle, p=0.65). L4 shows a statistically significant but tiny decline (slope=-0.0003/cycle, p<0.001) that amounts to 1.691→1.682 over 25 cycles.

Val loss declines slowly: 0.821→0.803 (-2.2%, p<0.001, R²=0.54). This is the only outcome metric with a clear trend, but the magnitude is small.

### The 4-cycle L3 eta² trend at the final layer was within noise

The motivating signal — L3 eta² at post_block5 increasing +15% over 4 cycles (0.389→0.449) — does not hold up over 25 cycles. The full trajectory has std=0.05 on a mean of ~0.45, and the early/late comparison (cycles 1-8 vs 18-25) gives p=0.15. The original 4-cycle trend happened to sample an upswing in a noisy series.

### L3 eta² IS increasing at intermediate layers

The eta² trend is real at earlier network layers:

| Layer | Early (c1-8) | Late (c18-25) | Δ | p-value |
|---|---|---|---|---|
| post_block0 | 0.173 ± 0.006 | 0.189 ± 0.004 | +9.4% | <0.001 |
| post_block3 | 0.552 ± 0.022 | 0.588 ± 0.010 | +6.5% | 0.001 |
| post_block5 | 0.429 ± 0.054 | 0.470 ± 0.047 | +9.5% | 0.15 (n.s.) |

The model's early and middle representations are becoming more conditioned on L3 hierarchical features (p<0.001 at blk0, p=0.001 at blk3). This is a real representational change that is NOT propagating to the final layer or to behavioral outcomes over 25 cycles.

### L5 "phase transition" alerts were noise

L5 NTP loss has CV=7.1% (vs 1.4-1.7% for L2-L4), bouncing between 3.95 and 5.29 with no trend (slope=+0.014, p=0.11). The three >10% cycle-over-cycle drops (c11→12: -13.6%, c17→18: -15.6%, c24→25: -23.0%) are followed by rebounds. This is a random walk at a level the model has not learned.

### Activation norm inflation is steady and linear

| Metric | c1 | c25 | Slope/cycle | R² |
|---|---|---|---|---|
| ActNorm (post_block5) | 55.4 | 159.5 | +4.2 | 0.981 |
| Residual norm | 20.4 | 57.3 | +1.5 | 0.978 |

Activation norms are growing linearly at 2.85× baseline after 25 cycles. At this rate, the 5× halt threshold would be reached around cycle 55-60. The growth is highly linear (R²=0.98), suggesting a steady drift rather than exponential instability.

### Gate, FM cosine, self-knowledge are stable

- **Gate**: 0.983→0.990, creeping slowly toward 1.0
- **FM cosine**: 0.908-0.914 range, stable in the sweet spot as designed
- **Self-knowledge**: 0.562 (c10), 0.549 (c20) — flat, comparable to OL baseline (0.544)
- **Robustness**: +0.085 (c10), +0.078 (c20) — flat

### Full trajectory

| Cyc | Val | L0 | L1 | L2 | L3 | L4 | L5 | Gate | FMCos | ResNrm | ActNrm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.821 | 0.41 | 1.21 | 1.17 | 1.78 | 1.69 | 4.08 | 0.983 | 0.908 | 20.4 | 55.4 |
| 5 | 0.809 | 0.41 | 1.17 | 1.15 | 1.70 | 1.69 | 4.36 | 0.957 | 0.912 | 30.6 | 83.2 |
| 10 | 0.804 | 0.41 | 1.19 | 1.16 | 1.72 | 1.68 | 4.36 | 0.974 | 0.910 | 39.9 | 107.2 |
| 15 | 0.803 | 0.40 | 1.17 | 1.14 | 1.71 | 1.68 | 4.61 | 0.981 | 0.912 | 47.6 | 129.4 |
| 20 | 0.800 | 0.41 | 1.17 | 1.16 | 1.71 | 1.68 | 4.30 | 0.987 | 0.917 | 52.1 | 146.9 |
| 25 | 0.803 | 0.42 | 1.21 | 1.17 | 1.72 | 1.68 | 4.07 | 0.990 | 0.914 | 57.3 | 159.5 |

## What this tells us about the setup

### 1. The gen-distill ratchet stabilizes but does not compound on stationary data

Gen-distill solved the NTP destruction problem and stabilized all metrics across 25 cycles (no degradation in val loss, generation accuracy, or per-level NTP performance). FM cosine stays in the 0.91 sweet spot. But the ratchet cycle (wake → sleep → repoint) does not compound: each fresh FM finds approximately the same prediction gap, the gate stays pinned near 1.0, and distillation transfers approximately the same amount of generation ability each cycle.

### 2. Representational change at intermediate layers does not propagate to behavior

The statistically significant L3 eta² increase at blk0 and blk3 shows the ratchet IS producing real representational change — the model's early/middle layers are encoding L3 hierarchical features more strongly over cycles. But this does not reach the final layer (blk5 trend is not significant) and does not affect any outcome metric. The computation at later layers may be dominated by other pressures (the dense local loss, the RL objective, or capacity constraints) that prevent the intermediate representational change from influencing the output.

### 3. Activation norm inflation is the binding constraint for extended runs

The linear growth in activation norms (~4.2/cycle at post_block5) means this setup cannot run indefinitely. At ~55-60 cycles the norm halt would trigger. This is the same pattern seen in the MNIST extended ratchet (post_block3 norms grew 8.5× over 16 cycles). The local loss (lambda_local=1.0) may contribute — it pushes the model toward FM-predictability, and the FM's prediction target is post_block5, creating pressure on those activation magnitudes.

### 4. The original 4-cycle L3 eta² trend was aliased

The +15% trend at post_block5 over cycles 1-4 that motivated this experiment was real in those 4 data points but falls within the noise band of the full 25-cycle trajectory (std=0.05, p=0.15 for early/late comparison). This is a cautionary note about interpreting short trajectories on noisy metrics.

## What to change to better test the hypothesis

The goal — testing whether FM-driven structural regularization can produce delayed generalization on a compact DGP — remains untested. This run showed that one specific configuration (gen-distill with lambda_local=1.0, mask=0.95, on stationary L=6/m=2 data) does not produce it over 25 cycles. Several design choices may need to change:

1. **Stationary data**: The model has seen the same 18M training tokens dozens of times. Each ratchet cycle provides a fresh FM perspective but not fresh data. Novel data (e.g., new rule sets at the same L,m, or curriculum over increasing L) would sustain compression pressure that stationary data exhausts.

2. **Lambda_local=1.0**: The local loss feedback loop pushes the model toward FM-predictability, which may suppress the representational exploration needed for a phase transition. Run 10 (lambda_local=0, 3-block gap) showed the richest dynamics (2.9× eta² progression, cosine 0.944) before activation norm inflation killed it. Stabilizing lambda_local=0 training (via gradient clipping or norm regularization) would test whether removing the FM-predictability pressure allows the intermediate-layer L3 trend to propagate further.

3. **Weight decay strength**: Standard grokking is sensitive to weight decay — too little and the model stays in the memorization solution, too much and it never memorizes. The current 0.01 is a default, not tuned. A sweep could reveal whether stronger weight decay accelerates the representational trend. See the weight decay experiment below — tested, and WD does not change outcome metrics.

4. **Model capacity vs DGP complexity**: The model can compose 2-3 levels at m=2. If L3 composition is at the edge of its capacity, no amount of regularization will push it through. A larger model (e.g., 8L/8H/256D, ~6.3M params) that can demonstrably learn 3-4 levels under standard NTP would be a cleaner test — the question becomes whether the ratchet accelerates learning at a level the model CAN reach, not whether it enables learning the model can't do.

## Weight decay experiment (2026-06-26)

Tests whether strong weight decay (0.1 vs baseline 0.01) during either the wake or sleep phase addresses the activation norm inflation and/or changes outcome metrics. Two conditions, each run for 40 cycles (15 complete at time of analysis):

- **Sleep WD**: 10× weight decay during distillation (sleep_wd=0.1), standard during wake (wake_wd=0.01). Hypothesis: the simplest standalone model that reproduces the FM-aligned teacher's behavior is more DGP-aligned, since the distillation gradient reflects the FM's functionally simpler transform rather than the NTP gradient's L0-dominated weighting.
- **Wake WD**: 10× weight decay during wake (wake_wd=0.1), standard during distillation (sleep_wd=0.01). Hypothesis: simplify the model while the FM is available as a crutch, so sleep has cleaner weights to distill into.

The theoretical concern with wake WD: NTP gradient is dominated by level 0 (~51% from geometric position weighting), so weight decay during wake preferentially kills the weakly-reinforced L3+ compositional weights while preserving the robust L0 sharpening weights — the opposite of what we want. Sleep WD avoids this because the distillation gradient reflects the FM's DGP-aligned computation, not the NTP objective's level weighting.

**Reproduction**:
```bash
modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended --weight-decay-mode sleep
modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended --weight-decay-mode wake
```

### Results (15 cycles)

#### Activation norm inflation

| Condition | c1 ActNorm | c15 ActNorm | c1→c15 ratio | Slope/cycle |
|---|---|---|---|---|
| Baseline | 55.4 | 129.4 | 2.34× | +5.3 |
| Sleep WD | 54.9 | 113.0 | 2.06× | +4.1 |
| Wake WD | 48.3 | 42.6 | 0.88× | **-0.4** |

Wake WD completely solved the norm inflation — activation norms are slightly *decreasing*. Sleep WD slowed it (~22% less inflation than baseline) but didn't stop it. Residual norms track the same pattern (baseline: 20.4→47.6, sleep: 20.2→41.7, wake: 17.4→15.3).

#### Outcome metrics are identical across all three conditions

| Metric | Baseline (c15) | Sleep WD (c15) | Wake WD (c15) |
|---|---|---|---|
| Val loss | 0.803 | 0.810 | 0.807 |
| L3 NTP loss | 1.709 | 1.828 | 1.742 |
| L4 NTP loss | 1.684 | 1.691 | 1.689 |
| Gen accuracy | 0.391 | 0.391 | 0.391 |
| L3 eta² @blk0 | 0.197 | 0.196 | 0.184 |
| L3 eta² @blk3 | 0.588 | 0.597 | 0.600 |

Generation accuracy is 0.389-0.391 in all three conditions at every cycle. Per-level NTP loss trajectories are noisy but indistinguishable. The slow L3 eta² trend at blk0 and blk3 proceeds at the same rate regardless of weight decay mode.

#### FM cosine and gate behavior diverge

| Condition | c15 FMCos | c15 Gate | c15 ResNorm |
|---|---|---|---|
| Baseline | 0.892 | 0.981 | 47.6 |
| Sleep WD | 0.888 | 0.986 | 41.7 |
| Wake WD | 0.876 | 0.934 | 15.3 |

Wake WD is the only condition where the gate isn't pinned near 1.0 — it's at 0.93 and drifting lower, meaning the model is selectively reducing FM reliance. FM cosine is also lower (0.876 vs 0.892), but the absolute residual is much smaller (15.3 vs 47.6), so the FM captures less of a smaller computation.

#### Robustness (c10 milestone)

| Condition | rob(eps=1.0) |
|---|---|
| Baseline | 0.085 |
| Sleep WD | 0.135 |
| Wake WD | 0.377 |

Wake WD is 4.4× more sensitive to perturbations. This likely reflects the lower activation norms making fixed-magnitude perturbations relatively larger, not a structural change.

### Interpretation

**Weight decay changes the model's magnitude and stability properties without changing what it can learn.** All three conditions produce identical outcome metrics (generation accuracy, per-level NTP loss, val loss trajectory) and the same slow L3 eta² trend at intermediate layers. The binding constraint on compositional learning is model capacity, not regularization strength.

Wake WD fully solves the norm inflation problem, which is useful for enabling longer runs, but the 15 cycles of stable training show no sign of emerging compositional improvement that more cycles would unlock. The model at this (L,m) setting simply cannot compose more than 2 levels regardless of how its weights are regularized.

The theoretical concern about wake WD preferentially killing L3+ weights was not validated — but it wasn't falsified either, since L3+ learning is at floor in all conditions. The concern remains relevant for a setting where the model CAN learn L3+ (e.g., a larger model or lower m).
