# RHM FOMAML Ratchet Experiments (2026-06-26)

**Code**: `rhm_fomaml_ratchet.py` (NTP wake), `rhm_fomaml_rl_ratchet.py` (RL wake)
**Prior experiments**: [MNIST gated ratchet](../../a2a_forward/GATED_RATCHET_README.md), [RHM gen-distill](RHM_RL_RATCHET_README.md#run-13-generation-based-distillation-2026-06-25), [RHM gen-distill extended](RHM_RL_GEN_DISTILL_EXTENDED_README.md)

## Motivation

The MNIST gated ratchet showed that FOMAML bilevel meta-learning and first-order NTP-trained unified gates produce equivalent compounding val loss improvement (48% gap vs OL over 4 cycles). Multiple first-order approaches on RHM (sparse ratchet, RL ratchet, gen-distill) failed to compound. The question: is the failure method-specific (first-order approximation loses something on harder DGPs) or domain-specific (the mechanism doesn't work on RHM regardless of optimization)?

Running FOMAML on RHM directly answers this. If FOMAML compounds on RHM, the first-order methods are losing something. If not, the problem is the domain.

## Design

Three experiments sharing the same architecture and evaluation:

- **Model**: 6L/6H/192D GPT (~2.68M params) on L=6/m=4/v=8/s=2 (seq_len=64)
- **FM**: 1L/1H/16D causal (50.4K params), post_block0 -> post_block3 (3-block gap)
- **Inject after block 1**
- **Pre-train**: 10K NTP steps (shared across all conditions)
- **Per cycle**: 2000 wake + 500 sleep (gen-based distillation) = 2500 steps, 4 cycles
- **FM retrain**: 2000 steps per repoint (frozen model)
- **Conditions**: WS_LG (FOMAML-gated local loss), WS (injection only), OL (continuous NTP baseline)

The LearningGate is a 37.1K-param MLP (384->64->192) outputting per-dimension sigmoid weights. On MNIST it takes CLS token activations; here it takes mean-pooled activations across positions (no CLS in autoregressive models). The gate persists across cycles; the CerebellarGate (injection) is reinitialized each cycle.

All experiments use gen-based distillation for the sleep phase (standard KL distillation destroys NTP on RHM; see [gen-distill run 13](RHM_RL_RATCHET_README.md#run-13-generation-based-distillation-2026-06-25)).

## Experiment 1: Dense NTP wake, NTP outer

**Code**: `rhm_fomaml_ratchet.py`

Wake: NTP + CerebellarGate injection + FOMAML-gated local loss. Inner loss = NTP + lambda * gated_local. Outer objective = NTP loss under virtual params (same batch, same as MNIST design).

### Results

| Cycle | WS_LG | WS | OL |
|---|---|---|---|
| 1 | 1.3842 | 1.3830 | 1.3849 |
| 2 | 1.3795 | 1.3893 | 1.3774 |
| 3 | 1.3755 | 1.3811 | 1.3740 |
| 4 | 1.3743 | 1.3771 | **1.3713** |

**Gate trajectory**: 0.002 -> 0.017 -> 0.025 -> 0.021 (97-100% sparse). The gate closes immediately and stays closed.

**Interpretation**: Dense NTP gives gradient at all 64 positions x 192 dimensions. The local loss (FM prediction error at intermediate layers) is redundant — it provides the same dimensionality of gradient that NTP already provides, with a less task-aligned signal. The bilevel optimization correctly identifies this and shuts the gate. On MNIST, classification gives gradient only at CLS (128 dims), so the local loss at 50 positions x 128 dims is a 50x information advantage — the gate opens because the local loss is genuinely useful.

**Notable positive**: WS produces 1.6x higher L3 feature eta^2 at the final layer than OL (0.094 vs 0.057). The gen-distill ratchet produces representational deepening that doesn't translate to NTP improvement.

## Experiment 2: RL wake, RL outer

**Code**: `rhm_fomaml_rl_ratchet.py` (before the per-level outer edit)

The dense NTP result motivated switching to sparse supervision. RL (REINFORCE on suffix generation) provides sparse, global supervision analogous to MNIST classification: one scalar reward per sequence. Wake: RL generation + sparse NTP (95% masked, ~3 positions) + FOMAML-gated local loss. Outer objective = RL loss under virtual params (REINFORCE log-prob of the same generated tokens, scored under virtual parameters).

### Results

| Cycle | WS_LG | WS | OL |
|---|---|---|---|
| 1 | 1.4212 | 1.4284 | 1.3849 |
| 2 | 1.4153 | 1.4345 | 1.3774 |
| 3 | 1.4124 | 1.4182 | 1.3740 |
| 4 | 1.4200 | 1.4195 | **1.3713** |

**Gate trajectory**: 0.499 -> 0.500 -> 0.501 -> 0.501 (0% sparse). The gate doesn't move from initialization (sigmoid(0) = 0.5).

**Generation accuracy**: WS_LG and WS both reach 0.234 (vs OL 0.160, +47%). The RL + FM injection helps generation substantially, but the FOMAML-gated local loss adds nothing beyond what WS already provides. With gate stuck at 0.5, WS_LG is effectively WS + uniform local loss at half weight.

**Interpretation**: The gate is stuck because the REINFORCE outer objective is too noisy for the meta-gradient to find a direction. The meta-gradient flows through: `rl_loss' -> virtual_params -> inner_gradient -> gate_w -> LearningGate`. The RL loss includes `advantage * log_prob`, where the advantage is a noisy scalar (reward minus running baseline). Each step's meta-gradient is dominated by whether this particular batch generated good or bad suffixes, not by whether the local loss was useful. The sign of the gate update flips randomly, and updates cancel over 8000 steps.

The gate at 0.5 (no movement) is qualitatively different from the NTP gate at 0.002 (decisive closure). With NTP, the meta-gradient has a consistent sign ("local loss doesn't help") and the gate converges quickly. With RL, the meta-gradient has no consistent sign (noise > signal) and the gate stays at initialization.

## Experiment 3: RL wake, per-level NTP outer

**Code**: `rhm_fomaml_rl_ratchet.py` (current version)

Keeps the sparse RL inner loop but replaces the noisy RL outer with a deterministic, compositionally-targeted objective: NTP loss evaluated only at hierarchy levels >= 2 (15 positions out of 63, selected by s-adic valuation of position indices). This has the MNIST structure: inner is sparse (~3 NTP positions + 1 RL scalar), outer measures something the inner mostly misses (15 level-2+ positions), and local loss bridges the gap (64 positions x 192 dims).

The outer uses sequence-aligned data (complete RHM sequences where the s-adic valuation is meaningful), separate from the inner RL batch. It evaluates NTP with injection under virtual parameters. The s-adic valuation is a function of position index and the branching factor s (a known hyperparameter), not of the DGP rules — analogous to how real-world RL data disproportionately tests higher-level compositional reasoning.

### Results

| Cycle | WS_LG | WS | OL |
|---|---|---|---|
| 1 | 1.4281 | 1.4284 | 1.3849 |
| 2 | 1.4310 | 1.4345 | 1.3774 |
| 3 | 1.4140 | 1.4182 | 1.3740 |
| 4 | 1.4177 | 1.4195 | **1.3713** |

**Gate trajectory**: 0.000 at every cycle (100% sparse). The gate closes harder than the dense NTP outer (0.000 vs 0.002).

**Interpretation**: The per-level outer gave the gate a clean, deterministic signal — and the signal said "local loss does not help compositional NTP." The gate closes decisively, confirming that the Experiment 2 result (gate stuck at 0.5) was indeed a variance problem, not an alignment problem. But fixing the variance revealed that the underlying answer is still "no": the FM's local loss does not improve level-2+ NTP quality after one FOMAML inner step.

WS_LG = WS across all metrics (generation 0.234 vs 0.234, val loss, self-knowledge, robustness). The gate at 0.000 means the local loss contributes nothing.

## Summary: three outer objectives, same conclusion

| Outer objective | Gate | Signal quality | Answer |
|---|---|---|---|
| Dense NTP (all 63 pos) | -> 0.002 | Clean (deterministic, dense) | "Local loss redundant — NTP covers everything" |
| RL (REINFORCE) | stuck at 0.5 | Noise (high-variance REINFORCE chain) | "Can't tell" |
| Per-level NTP (15 pos, L>=2) | -> 0.000 | Clean (deterministic, sparse) | "Local loss doesn't help composition" |

The per-level outer resolves the ambiguity left by the RL outer. The FM's prediction error at intermediate layers does not provide a useful gradient signal for higher-level compositional learning, even under sparse supervision where the information advantage should be maximal.

## Why FOMAML works on MNIST but not RHM

On MNIST, three properties align:

1. **Sparse task supervision**: Classification gives gradient at 1 position (CLS), so local loss at 50 positions is a 50x information advantage.
2. **Low-rank, class-discriminative residual**: The FM's 18-dimensional residual cleanly separates digit classes. Compressing along these dimensions directly helps classification.
3. **Single-step benefit**: One gradient step with local loss visibly improves the 10-class classification boundary.

On RHM, none hold:

1. **Dense NTP redundancy**: Even with sparse NTP masking (5%), the RL + NTP inner gradient already covers the relevant computation. The local loss adds dimensionality but not information the model can't get elsewhere within one step.
2. **High-rank, diffuse residual**: The FM's ~100-dimensional residual doesn't cleanly separate hierarchical features. Compressing in FM error directions doesn't directly help level-2+ composition.
3. **Multi-step composition gap**: Improving level-2+ NTP requires the model to learn deeper compositional rules — a representational change that takes many gradient steps to manifest. One FOMAML inner step is too short for the local loss effect to propagate from intermediate-layer FM-predictability to output-layer compositional accuracy.

The MNIST equivalence (FOMAML = unified gate = uniform local loss) was specific to a regime where local loss and the task objective were naturally aligned through a low-rank, class-discriminative residual. This is a property of the task, not a general property of the architecture.

## What remains untested

1. **Multi-step FOMAML**: The single inner step may be too short for the local loss to affect compositional NTP. A K-step inner loop (K=10-50) with the per-level outer would test whether the local loss helps with more compute — expensive but diagnostic.

2. **Evolution Strategies on the gate**: Derivative-free meta-optimization in 192-d gate space, immune to both REINFORCE variance and single-step limitations. Each gate perturbation trains for K steps before evaluation.

3. **FM-based gate heuristics**: Skip bilevel entirely. Set gate weights from FM error statistics (per-dimension variance, hierarchy-conditioned eta^2). Tests whether the FM's error structure contains information about useful compression directions, even if FOMAML can't extract it through a single inner step.

4. **Novel data**: All experiments used static data. The ratchet may need non-stationary data to sustain compression pressure — the FM's perspective is genuinely new only when the data distribution shifts.

## Reproduction

```bash
cd experiments/

# Experiment 1: Dense NTP wake + NTP outer
modal run --detach -m rhm.ratchet.rhm_fomaml_ratchet::rhm_fomaml_ratchet

# Experiment 2: RL wake + RL outer (original rhm_fomaml_rl_ratchet before edits)
# Not directly reproducible from current code; see git history

# Experiment 3: RL wake + per-level NTP outer (current code)
modal run --detach -m rhm.ratchet.rhm_fomaml_rl_ratchet::rhm_fomaml_rl_ratchet
```
