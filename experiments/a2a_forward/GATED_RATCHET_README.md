# MNIST Multi-Cycle Gated Ratchet (2026-06-17)

**Code**: `mnist_gated_ratchet.py`
**Prior experiments**: [MNIST local loss + learning gate](MNIST_LOCAL_LOSS_README.md), [MNIST wake-sleep comparison](MNIST_DISTILLATION_README.md#multi-cycle-comparison-with-compute-matched-baselines-2026-06-13)
**Idea docs**: [local_prediction_error_learning.md](../../ideas/local_prediction_error_learning.md), [cerebellar_abstraction_ratchet.md](../../ideas/cerebellar_abstraction_ratchet.md)

## Goal

Test whether multi-cycle CL_LG (injection + bilevel-gated local loss) with periodic distillation and FM reinitialization produces compounding improvement beyond standard wake-sleep. The learning gate persists across cycles, accumulating a meta-learning policy that must generalize across FM reinitializations. This is the first test of the full ratchet mechanism with task-informed gating.

## Design

Four conditions, all compute-matched on main-model gradient steps (8400 total = 4 cycles × (1500 wake + 600 sleep)):

| Condition | Wake phase | Sleep phase | FM reinit? | Gate persists? |
|---|---|---|---|---|
| **WS_LG** | CL_LG (injection + bilevel-gated local loss) | KL + CE distillation | yes (each cycle) | yes |
| **WS** | CL (injection only) | KL + CE distillation | yes (each cycle) | N/A |
| **CL_LG** | CL_LG continuous (no cycles) | N/A | no | yes (continuous) |
| **OL** | Open-loop continuous | N/A | no | N/A |

Architecture: 4L/4H/128D ViT (0.80M params), 1L/1H/32D bidirectional FM (83K params), post_block0 → post_block3, injection after block 1. Learning gate: 256→64→128 MLP (24.8K params), bilevel-optimized via FOMAML. Identical seed (42), lr (3e-4), initial weights across all conditions. 1500 retrain steps per FM reinitialization (frozen model, does not count toward main-model steps). Data: MNIST, 60K train images = ~18 epochs over 8400 steps.

## Results

### Val loss / accuracy: compounding improvement, widening gap

| Cycle | WS_LG | WS | CL_LG | OL | WS_LG vs OL |
|---|---|---|---|---|---|
| 1 (2100) | 0.1035 / 97.5% | 0.0945 / 97.2% | 0.1275 / 96.4% | 0.1559 / 95.2% | −34% |
| 2 (4200) | 0.0677 / 98.1% | 0.0631 / 98.3% | 0.0898 / 97.8% | 0.1059 / 96.6% | −36% |
| 3 (6300) | **0.0560** / 98.3% | 0.0677 / 97.3% | 0.0751 / 97.5% | 0.1050 / 97.7% | −47% |
| 4 (8400) | **0.0522** / **98.6%** | 0.0628 / 98.6% | 0.0895 / 98.0% | 0.1005 / 97.5% | **−48%** |

WS_LG achieves the best final val loss (0.0522) and ties for best accuracy (98.6%). Its val loss drops **monotonically** across all 4 cycles, while WS stalls from cycle 2 onward (0.0631 → 0.0677 → 0.0628) and CL_LG oscillates. OL plateaus by cycle 2.

The WS_LG → OL gap widens monotonically from 34% to 48%. The compounding requires both components: distillation alone (WS) stalls at 0.063 val loss; gated local loss alone (CL_LG) stalls at 0.090. Neither component in isolation produces the compounding trajectory.

### Robustness: improves across cycles but injection-only (WS) still dominates

| Cycle | WS_LG | WS | CL_LG | OL |
|---|---|---|---|---|
| 1 | +0.363 | +0.185 | +0.251 | +0.104 |
| 2 | +0.136 | +0.036 | +0.144 | +0.044 |
| 3 | +0.086 | +0.004 | +0.092 | +0.015 |
| 4 | +0.073 | **+0.004** | +0.015 | +0.003 |

WS_LG starts very brittle (worse than OL at cycle 1 — the local loss creates perturbation sensitivity, replicating the MNIST local loss finding). But it improves 5× across 4 cycles (0.363 → 0.073), suggesting the ratchet partially mitigates the brittleness. WS achieves the best robustness at every cycle, consistent with the prior finding that robustness requires injection experience without local-loss-induced compression.

CL_LG (continuous, no distillation) reaches near-OL robustness by cycle 4 (0.015 vs 0.003), outperforming WS_LG (0.073). Without distillation resetting the model each cycle, the gate has more time to learn perturbation absorption within a single continuous training run.

### Self-knowledge probes: negative (measurement artifact)

| Cycle | WS_LG | WS | CL_LG | OL |
|---|---|---|---|---|
| 1 | −0.014 | 0.007 | −0.058 | 0.011 |
| 2 | −0.103 | 0.038 | −0.171 | 0.000 |
| 3 | −0.229 | 0.016 | −0.354 | 0.005 |
| 4 | −0.323 | 0.027 | −0.282 | −0.001 |

All R² values for WS_LG and CL_LG at post_block0 are negative and worsening. This is a measurement artifact: the probes use a **fresh FM** trained on each model's final activations, not the FM from co-training. The single-cycle CL_LG R² = 0.83 (from [MNIST_LOCAL_LOSS_README.md](MNIST_LOCAL_LOSS_README.md)) was measured against the co-training FM — the model encoded self-knowledge about *that specific FM's* residual. After distillation reshuffles the model's representations and a fresh FM finds completely different residual structure, the model has no reason to linearly predict this new FM's errors.

This confirms the prior finding that self-knowledge is FM-specific. A more informative measurement would probe during each wake phase against the current co-training FM, but this is a property of the experimental design, not a negative result about the mechanism.

### Gate dynamics: opens and differentiates, does not close

| Phase | Mean | Std | Sparse (<0.1) | Sparse (<0.2) |
|---|---|---|---|---|
| Post-wake C1 | 0.308 | 0.250 | 0.0% | 0.0% |
| Post-wake C2 | 0.513 | 0.367 | 0.0% | 4.7% |
| Post-wake C3 | 0.688 | 0.373 | 4.7% | 10.2% |
| Post-wake C4 | 0.773 | 0.360 | 4.7% | 7.0% |

The gate weight repoint values are essentially unchanged from post-wake values (e.g., post_wake_c4 mean=0.773 vs post_repoint_c4 mean=0.763), confirming the gate's policy is stable across FM reinitializations.

The gate **opens** across cycles rather than closing: mean weight rises from 0.31 to 0.77. This is the opposite of the naive "plasticity closes with maturity" prediction. However, selectivity does increase: some dimensions reach near-zero weight (sparse <0.1 grows from 0% to 4.7%), and std increases relative to mean (0.250/0.308 = 0.81 at C1 vs 0.360/0.773 = 0.47 at C4 — the coefficient of variation drops, but absolute differentiation grows).

**Why the gate opens rather than closes**: at ~18 epochs on MNIST, the model has seen every training example ~18 times. There is no genuinely novel data — the entire input distribution is routine. The bilevel signal asks "does compressing this dimension help val loss after one gradient step?" and the answer is always "yes" because more regular computation generalizes better *within the same distribution*. The gate has no reason to close because there is nothing to protect from being overwritten. The model endlessly refines its solution to become maximally legible to a compressed version of its own computation, and this refinement is always beneficial on a fixed distribution.

The gate would only close in a regime where the bilevel outer loop evaluates on a distribution that differs from the inner loop — i.e., where over-compression of existing representations would damage performance on novel inputs. On MNIST at 18 epochs, inner and outer distributions are identical.

### Residual structure: divergent trajectories

| Cycle | WS_LG eff rank | WS eff rank | CL_LG eff rank | OL eff rank |
|---|---|---|---|---|
| 1 | 27.8 | 23.9 | 23.1 | 20.6 |
| 2 | 27.3 | 16.5 | 24.9 | 18.6 |
| 3 | 28.0 | 13.4 | 27.0 | 18.6 |
| 4 | **30.2** | **13.5** | 24.5 | 19.7 |

WS_LG and WS diverge dramatically: WS_LG's effective rank **increases** (27.8 → 30.2) while WS's **decreases** (23.9 → 13.5). Without the local loss, standard distillation concentrates the residual into fewer dimensions (the "grokking direction" — computation becoming more structured/low-rank). With the gated local loss, the model's computation stays high-rank because the FM is being pushed toward accuracy across ALL dimensions, preventing concentration.

WS_LG's digit discriminability (eta²) also increases across cycles (0.049 → 0.083) while WS's decreases (0.054 → 0.035). The gated ratchet produces a residual that becomes MORE digit-specific over cycles — the fresh FM keeps missing class-conditional computation that the model continues to develop.

### Innovation migration: moderate, increasing overlap

| Cycle boundary | Subspace overlap |
|---|---|
| C1 → C2 | 0.627 |
| C2 → C3 | 0.632 |
| C3 → C4 | 0.656 |

Successive FMs' top-5 PC directions overlap ~63%, increasing slightly across cycles. This is moderate migration — the model's computation is changing with each cycle, but the gate stabilizes what kind of computation the model performs. Compare with single-cycle MNIST distillation where PCs were near-orthogonal (cos 0.04–0.33); the gated ratchet produces more continuity across cycles.

## Interpretation: the gated ratchet as implicit regularization

The central finding is that WS_LG produces monotonically compounding val loss improvement (34% → 48% gap vs OL, widening at every cycle) from a mechanism that requires both components — the bilevel-gated local loss AND the distillation ratchet. Neither alone sustains improvement past cycle 2.

**What the gated local loss provides that distillation alone doesn't**: dense, direction-specific supervision at intermediate layers. NTP gives gradient only through the classification loss (effectively 128-d at [CLS]). The gated local loss gives gradient at all 50 positions × 128 dimensions, filtered by the bilevel optimization to include only directions where compression helps classification. This is implicit regularization — it selects among the many parameter configurations that achieve high accuracy for the one whose intermediate computation is most compressible by a self-model.

**What distillation provides that continuous gated training doesn't**: a discrete compression event that forces the model to internalize the FM's contribution into its weights alone (sleep phase), followed by a fresh FM perspective on the reorganized model (re-point phase). The fresh FM discovers new residual structure that the gate must adapt to, preventing co-adaptation between gate and FM. CL_LG continuous (no distillation) stalls because the gate and FM settle into a stable equilibrium — the distillation periodically breaks this equilibrium and forces the system to re-optimize.

**Why it compounds**: each cycle, the distillation absorbs the FM's compressed perspective into the model's weights (making computation more regular), and the fresh FM discovers what's still irregular about the newly reorganized model. The gate selects which directions of the new irregularity are worth compressing. Over cycles, the model's computation becomes progressively more regular — more legible to a compressed version of itself — and this regularity is genuine generalization (val loss, not just train loss), at least within the training distribution.

The analogy to human expertise: a domain expert doesn't just know the right answers; they've reorganized their reasoning to be compressible — explainable in simple terms. The gated ratchet produces models that find solutions which are maximally legible to a compressed self-model, in a way that the task loss endorses. This is deep understanding as opposed to surface-level pattern matching.

## The gate-closing prediction and its failure

The developmental trajectory hypothesis predicted the gate would close across cycles — becoming more selective as the model "matures" and has less to learn. Instead, the gate opens (mean weight 0.31 → 0.77).

This is explained by the data regime. On MNIST at 18 epochs, the model has seen every training example many times. There is no novel data — the entire input distribution is routine. The bilevel optimization's outer loop evaluates on the same distribution as the inner loop, so more compression always helps. The gate correctly concludes: compress everything.

The gate would close only in a regime where:
1. The model has accumulated knowledge worth protecting (existing competencies that over-compression would damage)
2. Novel inputs arrive that require representational capacity the model hasn't yet allocated
3. The bilevel outer loop evaluates on a distribution that includes this novelty

On a fixed dataset, condition (2) never holds. The gate-closing prediction requires a continual learning or OOD evaluation setup — specifically, one where the meta-objective is "help on the NEXT task" rather than "help on the SAME task." This is disanalogous to human development, where the outer loop is implicitly "the rest of your life, including things you've never seen."

[**Update (2026-06-18):** OOD gate experiments ([OOD_GATE_README](../a2a_forward/OOD_GATE_README.md)) tested conditions (1-3) directly with digit shifts (0-6 → 0-9) and domain shifts (MNIST → Fashion-MNIST). Conditions (1-3) are necessary but not sufficient for *selective* closing. A fourth condition is required: the FM's error dimensions must decompose along the boundaries the gate needs to discriminate. On the digit shift (same visual manifold), the FM's error decomposition remained valid for novel digits and the gate was input-selective (novel digits got +0.13-0.17 higher gate weights). On the Fashion shift (different visual manifold), the FM's error dimensions didn't align with domain boundaries, so the gate closed *globally* rather than selectively — it retreated uniformly when the combined task was hard enough that compression hurt both domains. The gate's selectivity is per-dimension of FM error, not per-input; it can only discriminate distinctions that the FM's error structure encodes. More broadly, the gate's meta-learning operates over the model's *computational* structure as decomposed by the FM, not over data domains.]

The biological analog to the gate opening on routine data may be related to the phenomenology of expertise and boredom. The dopamine system may track the *derivative* of the FM residual — rapid decrease in residual (novel thing becoming routine) = rewarding; residual already near zero (already routine) = boring; high static residual (confused but not progressing) = frustrating. The gate opening on MNIST at 18 epochs corresponds to the model in an environment where everything is routine and there is no boredom signal to redirect attention.

## Reproduction

```bash
cd experiments/

# Full experiment (~3-4 hours on L4)
modal run --detach a2a_forward/mnist_gated_ratchet.py::a2a_mnist_gated_ratchet
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/mnist_gated_ratchet/
└── vit_4L_4H_128D/post_block0_to_post_block3/
    ├── wslg_model.pt, wslg_fm.pt, wslg_lgate.pt
    ├── ws_model.pt, ws_fm.pt
    ├── cllg_model.pt, cllg_fm.pt, cllg_cgate.pt, cllg_lgate.pt
    ├── ol_model.pt
    └── results.json
```

## Extended ratchet: 16 cycles with 10% FM (2026-06-17)

**Code**: `mnist_extended_ratchet.py` | **Analysis**: `extended_ratchet_analysis.py`, `gate_structure_analysis.py`

Tests the absorbing state prediction: does multi-cycle WS_LG converge to a state where the model's computation is fully FM-predictable (residual → 0)? Runs WS_LG only (+ lightweight OL reference) for 16 cycles with the same 10% FM (83K params) as the 4-cycle experiment. Identical seed/lr/init — the first 4 cycles reproduce the 4-cycle results.

### Two-phase dynamics: compression then norm inflation

| Cycle | res_norm | fwd_cos | eff_rank | gate_m | val_loss | rob(1.0) |
|---|---|---|---|---|---|---|
| 1 | 0.580 | 0.991 | 27.8 | 0.314 | 0.1035 | +0.363 |
| 2 | 0.524 | 0.996 | 27.3 | 0.510 | 0.0677 | +0.136 |
| 3 | 0.449 | 0.998 | 28.0 | 0.688 | 0.0560 | +0.086 |
| 4 | 0.433 | 0.998 | 30.2 | 0.763 | 0.0522 | +0.073 |
| **5** | **0.370** | **0.999** | 33.3 | 0.737 | 0.0470 | +0.047 |
| 6 | 0.454 | 0.999 | 34.5 | 0.728 | 0.0274 | +0.029 |
| 7 | 0.500 | 0.999 | 35.9 | 0.692 | 0.0597 | +0.017 |
| 8 | 0.566 | 0.999 | 35.8 | 0.719 | 0.0456 | +0.023 |
| 9 | 0.646 | 0.999 | 32.9 | 0.738 | 0.0407 | +0.018 |
| 10 | 0.725 | 0.999 | 34.5 | 0.727 | 0.0328 | +0.009 |
| 11 | 0.962 | 0.999 | 30.5 | 0.602 | 0.0581 | +0.014 |
| 12 | 0.877 | 0.999 | 31.8 | 0.564 | 0.0433 | +0.015 |
| 13 | 0.986 | 0.999 | 32.9 | 0.653 | 0.0428 | +0.007 |
| 14 | 0.991 | 0.999 | 33.5 | 0.622 | 0.0444 | +0.012 |
| 15 | 1.151 | 0.999 | 30.6 | 0.592 | 0.0563 | +0.002 |
| 16 | 1.136 | 0.999 | 31.1 | 0.628 | 0.0513 | +0.001 |

The absorbing state prediction (residual → 0) fails. The residual norm decreases during cycles 1-5 (0.58 → 0.37), then reverses and grows 3× through cycle 16 (0.37 → 1.14). The reversal coincides precisely with the FM cosine plateauing at 0.999 (cycle 5), and with the best val loss (cycle 6, 0.0274). Val loss oscillates around 0.04-0.06 afterward.

### The activation norm explosion

**Analysis**: `extended_ratchet_analysis.py`

The growing residual is caused by activation norm inflation at post_block3 (the FM's prediction target). Comparing the 16-cycle WS_LG model with the 4-cycle model:

| Layer | WS_LG growth (16c/4c) | OL growth (16c/4c) |
|---|---|---|
| post_embed | 1.7× | 1.9× |
| post_block0 | 3.6× | 2.1× |
| post_block1 | 3.2× | 2.2× |
| post_block2 | 3.4× | 2.2× |
| **post_block3** | **8.5×** | **2.2×** |

Post_block3 mean activation norm reaches 79.7 in WS_LG_16c vs 32.2 in OL_16c (2.5× ratio). At 4 cycles, WS_LG had *smaller* activations than OL (ratio 0.64). The inflation is specific to the WS_LG mechanism and concentrated at the FM prediction target.

**The raw robustness improvement is a measurement artifact.** With fixed-size perturbations (eps=1.0), WS_LG_16c shows +0.001 vs OL's +0.003 — apparently 3× more robust. But with perturbations scaled proportionally to activation norm:

| rel_eps | WS_LG_16c | OL_16c |
|---|---|---|
| 0.05 | +0.018 | +0.001 |
| 0.10 | +0.248 | +0.026 |
| 0.20 | +1.032 | +0.260 |

WS_LG is 10-16× MORE sensitive than OL at matched relative perturbation strength. The local-loss-induced brittleness (from the single-cycle experiment) persists and is masked by growing activation norms.

Note: the 4-cycle robustness was genuine (WS_LG had smaller activations than OL and was validated by Hessian trace analysis). The transition from genuine robustness to norm-inflated robustness occurs around cycle 5, when the FM hits its compression ceiling.

### Mechanism: FM compression ceiling

The FM cosine plateaus at 0.999 by cycle 5. At 83K params (10% of the 0.80M ViT), the FM is at the saturation point identified by the scaling sweep — it can represent essentially all of the main model's computation, including irregular, input-specific patterns. Once the FM captures everything, the gated local loss gradient becomes uninformative: "be predictable to the FM" is vacuous when the FM already predicts everything. Without meaningful consolidation pressure, activation norms drift upward under the classification loss, with nothing constraining them.

The timing is a smoking gun: FM cosine plateau, residual reversal, best val loss, and gate peak all coincide at cycle 5-6. Everything after cycle 5 is post-ceiling artifact.

### Smaller FM (1.6%): ceiling hit earlier, same failure mode

**Code**: `mnist_extended_ratchet.py` with `--fwd-d-head 8 --fwd-mlp-mult 0.25`

To test whether a smaller FM would sustain compression pressure longer (lower ceiling → more room for the ratchet), reran with a 13.1K FM (1.6% of main model, d_head=8, mlp_hidden=32) for 10 cycles.

| Cycle | res_norm | fwd_cos | eff_rank | gate_m | val_loss |
|---|---|---|---|---|---|
| 1 | 0.542 | 0.988 | 41.5 | 0.456 | 0.1040 |
| 2 | 0.466 | 0.994 | 37.6 | 0.887 | 0.0730 |
| 3 | 0.480 | 0.995 | 38.0 | 0.963 | 0.0677 |
| 4 | 0.500 | 0.996 | 33.4 | 0.953 | 0.0633 |
| 5 | 0.504 | 0.997 | 34.9 | 0.923 | 0.0485 |
| 6 | 0.529 | 0.998 | 34.4 | 0.727 | 0.0384 |
| 7 | 0.782 | 0.997 | 26.8 | 0.746 | 0.0477 |
| 8 | 1.259 | 0.997 | 20.0 | 0.601 | 0.0543 |
| 9 | 1.816 | 0.996 | 15.5 | 0.421 | 0.0458 |
| 10 | 2.130 | 0.994 | 16.8 | 0.375 | 0.0517 |

The smaller FM made the problem worse, not better:

1. **The FM hits a lower ceiling earlier** (0.994-0.997, plateauing by cycle 3 vs 0.999 at cycle 5 for the 10% FM). The residual reversal happens at cycle 2 and is more dramatic (2.13 by cycle 10 vs 1.14 by cycle 16).

2. **The gate opens then closes dramatically** (0.46 → 0.96 → 0.38, vs 0.31 → 0.76 → 0.63 for the 10% FM). The bilevel optimization initially endorses strong compression, then discovers the small FM's local loss is too noisy to be useful and shuts the gate.

3. **Effective rank collapses** (41.5 → 16.8, vs stable 28-36 for the 10% FM). The tiny FM fails in the same few ways every time, concentrating the residual into fewer dimensions.

The failure is not that the FM is too large (capturing irregularity), but that the FM's error structure interacts with the bilevel optimization: high FM error → noisy local loss gradient → gate closes to protect the model from bad gradients → compression pressure vanishes. A smaller FM has higher error, so the gate closes faster.

### Gate structure analysis: what the gate is selective for

**Code**: `gate_structure_analysis.py`

Analyzed the gate's per-dimension and per-digit selectivity at two points: the 4-cycle peak (10% FM, gate mean 0.76) and the 10-cycle closing (1.6% FM, gate mean 0.38).

**The gate is highly digit-selective.** Digit identity explains 57% of gate weight variance at peak and 82% at closing. At closing, clear digit groups emerge: digits 0/6/7 stay open (gate ~0.52-0.56) while digits 2/3/4/5/8/9 close (gate ~0.27-0.30).

**The gate anti-correlates with FM error.** corr(gate_weight, FM_error_variance) = -0.55 at peak, -0.79 at closing. The gate closes for dimensions where the FM has high error. This is rational from the bilevel optimization's perspective (noisy FM predictions → noisy gradients → better to ignore), but means the gate avoids exactly the dimensions where compression would be most beneficial for the ratchet.

**The gate is mostly a static per-dimension mask.** 72% of variance is across-dimension (which dims are open/closed), only 28% within-dimension (input-dependent variation). At closing, gate weights are nearly binary: 59/128 dims < 0.2 and 33/128 dims > 0.8.

**The two runs learned unrelated dimension policies.** corr(peak_dims, closing_dims) = -0.07. Because the FMs have different architectures and different error structures, the gates adapted to different selectivity patterns.

### Interpretation: the FM capacity sweet spot

The extended ratchet reveals a narrow viable regime for the FM capacity:

- **FM too large** (10% / 83K): captures all computation including irregularity. FM cosine saturates at 0.999, consolidation pressure vanishes, activation norms inflate. The ratchet works for ~5 cycles then enters an artifact regime.

- **FM too small** (1.6% / 13K): high FM error makes the local loss noisy. The bilevel gate rationally closes to protect the model from bad gradients, killing compression pressure even faster. The ratchet works for ~2 cycles.

- **Sweet spot** (untested): an FM capacity where the FM is accurate enough to provide clean gradients (gate stays open) but limited enough that there's persistent residual (genuine compression pressure). Likely in the 3-5% range for this architecture.

However, this analysis applies specifically to the fixed-dataset regime (MNIST at 60K images, ~18+ epochs per cycle). The FM capacity sweet spot is narrow because the data distribution is exhausted — there's no novel data to sustain the ratchet. In a continual learning setting, novel data would continually generate computation that even a well-matched FM can't predict, sustaining compression pressure regardless of FM capacity. The biological analog — a cerebellum that's permanently capacity-limited relative to cortical computation, but operating on an ever-changing experience stream — sidesteps the fixed-dataset problem entirely.

## Reproduction

```bash
cd experiments/

# Original 4-cycle experiment (~3-4 hours on L4)
modal run --detach a2a_forward/mnist_gated_ratchet.py::a2a_mnist_gated_ratchet

# Extended 16-cycle with 10% FM (~4-5 hours on L4)
modal run --detach a2a_forward/mnist_extended_ratchet.py::a2a_mnist_extended_ratchet \
  --n-cycles 16

# Extended 10-cycle with 1.6% FM (~3 hours on L4)
modal run --detach a2a_forward/mnist_extended_ratchet.py::a2a_mnist_extended_ratchet \
  --n-cycles 10 --fwd-d-head 8 --fwd-mlp-mult 0.25

# Activation norm analysis (loads saved checkpoints)
modal run --detach a2a_forward/extended_ratchet_analysis.py::analyze_activation_norms

# Gate structure analysis (loads saved checkpoints)
modal run --detach a2a_forward/gate_structure_analysis.py::analyze_gate_structure
```

## Next steps

1. **Language gated ratchet**: Language's full-rank residual (200/256 dimensions) and rich behavioral decomposition (delimiter tracking, distributed attention, focused retrieval) would make the gate's selectivity much more interpretable. On language, the local loss dynamics may differ because each token already provides a dense NTP signal (unlike MNIST where only [CLS] gets classification gradient). The gate might develop per-behavioral-category selectivity — compressing delimiter tracking while leaving semantic composition alone — which MNIST's 10-class structure can't distinguish.

2. ~~**OOD gate-closing test**~~: *Done* — see [OOD_GATE_README](OOD_GATE_README.md). The gate opens more for novel digits (diff +0.13-0.17) rather than closing for known digits. On a harder cross-domain shift (MNIST → Fashion-MNIST), the gate closes globally rather than selectively. The gate's meta-learning operates over computational structure as decomposed by the FM, not over data domains.

3. **OOD adaptation post-ratchet**: Freeze WS_LG and OL models after 4 cycles. Fine-tune on rotated MNIST or Fashion-MNIST. Measure adaptation speed and forgetting. The "maximally regular computation" from the gated ratchet should produce better zero-shot OOD (from regularity) and potentially better adaptation (from organized representations). The MNIST adaptation experiment's three-way dissociation (zero-shot tracks distillation, forgetting tracks CL) predicts WS_LG should win on both axes.
