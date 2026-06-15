# MNIST OOD Adaptation Speed Experiment (2026-06-14)

**Code**: `mnist_adaptation.py`
**Prior experiments**: [MNIST wake-sleep comparison](MNIST_DISTILLATION_README.md#multi-cycle-comparison-with-compute-matched-baselines-2026-06-13), [OOD robustness](OOD_ROBUSTNESS_README.md), [Jacobian analysis](JACOBIAN_ANALYSIS_README.md)

## Motivation

The OOD robustness experiment (language models) showed that forward prediction produces distribution-invariant perturbation robustness but no zero-shot OOD loss benefit. Two possible explanations: (1) the 29M language models are too underparameterized for anything generalizable to port between language distributions; (2) zero-shot loss is the wrong metric — adaptation speed or adaptation cost might better capture the functional advantage of organized representations.

The MNIST wake-sleep comparison provides ideal checkpoints to test this: four compute-matched conditions (WS, CL, OL, KD — all at 8400 main-model gradient steps, identical init/seed) with cleanly dissociated mechanisms (distillation in WS+KD, CL co-training in WS+CL). The MNIST domain has stronger self-knowledge signals (3–6× language), dramatic robustness effects, and a right-sized model for the task.

## Design

**OOD domain**: Rotated MNIST at 15°, 30°, 45°, 60°, 90°. Rotation preserves the underlying task (digit classification) while shifting the input distribution. The model needs the same computational structure (edge detection, curve following, spatial integration) applied to transformed inputs. Five angles provide a dose-response curve from near-ID (15°) to severe shift (90°).

**Protocol**: Load the four checkpoints from the wake-sleep comparison. For each condition × angle:
1. Zero-shot evaluation on rotated test set (10K images, full)
2. Fine-tune all parameters for 500 steps on rotated training set (lr=1e-4, AdamW, same batch ordering across conditions)
3. Evaluate on rotated test set every 10 steps (full 10K test set)
4. Post-adaptation evaluation on original (unrotated) test set to measure catastrophic forgetting

**Conditions** (from the wake-sleep comparison, all at 8400 main-model gradient steps):

| Condition | Wake phase | Sleep teacher | FM/injection |
|---|---|---|---|
| **WS** | CL co-training | Self (CL model + FM + gate) | Yes |
| **CL** | CL co-training | None (continuous) | Yes |
| **OL** | OL training | None (continuous) | No |
| **KD** | OL training | External (independent model) | No |

All models evaluated and fine-tuned standalone (no injection active).

**Key comparisons**:
- WS vs KD: isolates the self-referential component (both have distillation)
- CL vs OL: isolates the closed-loop effect (no distillation)
- WS vs OL: combined effect

## Results

### ID performance (unrotated MNIST test set)

| Condition | Loss | Accuracy |
|---|---|---|
| WS | 0.061 | **98.1%** |
| KD | 0.061 | **98.1%** |
| OL | 0.096 | 97.3% |
| CL | 0.109 | 97.0% |

Same ordering as the wake-sleep comparison: {WS, KD} >> OL > CL. CL's standalone accuracy is lowest due to the unresolved dependency gap.

### Zero-shot OOD accuracy

| Angle | WS | KD | OL | CL |
|---|---|---|---|---|
| 15° | **96.1%** | 95.4% | 94.5% | 92.8% |
| 30° | **84.7%** | 82.4% | 78.6% | 76.8% |
| 45° | **61.4%** | 59.0% | 48.8% | 50.5% |
| 60° | 33.6% | **34.3%** | 25.7% | 27.0% |
| 90° | **12.9%** | 11.1% | 12.4% | 12.1% |

The ordering is WS ≈ KD >> OL ≈ CL. At 45°, WS is 12.6pp above OL. This tracks **distillation** (val loss quality), not self-knowledge or robustness. The models with soft-target training generalize better to rotated inputs without any fine-tuning, consistent with distillation producing smoother decision boundaries. CL is the worst at small angles due to its dependency gap degrading standalone performance.

At 90° all models converge to near-chance (10–13%), since a 90°-rotated digit is unrecognizable without adaptation.

### Adaptation speed (null result)

| Angle | WS | CL | OL | KD |
|---|---|---|---|---|
| 15° | 0 | 0 | 0 | 0 |
| 30° | 10 | 10 | 10 | 10 |
| 45° | 40 | 40 | 40 | 40 |
| 60° | 70 | 70 | 70 | 70 |
| 90° | 150 | 130 | 130 | 140 |

(Steps to recover 90% of each condition's own ID accuracy.)

All conditions adapt at the same rate. The loss landscape differences that produce 2–34× perturbation robustness do not translate into different gradient descent speeds at this learning rate. This makes sense: perturbation robustness measures response to *random* displacement, while gradient descent follows a *structured* direction (the loss gradient). A flatter landscape reduces the damage from random perturbations but doesn't necessarily accelerate optimization along the gradient.

### Adaptation AUC (time-weighted mean accuracy during fine-tuning)

| Angle | WS | CL | OL | KD |
|---|---|---|---|---|
| 15° | **0.979** | 0.978 | 0.974 | 0.975 |
| 30° | **0.961** | 0.958 | 0.956 | 0.957 |
| 45° | **0.939** | 0.932 | 0.930 | 0.932 |
| 60° | **0.913** | 0.906 | 0.905 | 0.910 |
| 90° | **0.866** | 0.859 | 0.862 | 0.859 |

WS has the highest AUC at every angle, but the differences are small (0.5–1pp). The AUC advantage is driven primarily by the zero-shot gap (WS starts higher), not by faster adaptation.

### Post-adaptation accuracy (500 steps)

| Angle | WS | CL | OL | KD |
|---|---|---|---|---|
| 15° | 98.1% | **98.2%** | 97.8% | 97.7% |
| 30° | 97.2% | **97.3%** | 96.8% | 96.9% |
| 45° | **96.7%** | 96.4% | 96.1% | 96.0% |
| 60° | 95.2% | **95.4%** | 95.3% | 95.0% |
| 90° | 93.6% | 94.0% | **94.1%** | 93.3% |

All conditions converge to approximately the same accuracy after 500 steps of fine-tuning. The starting-point differences wash out. CL often reaches the highest post-adaptation accuracy despite starting from the worst zero-shot performance — consistent with its representations being structurally well-organized for adaptation even though its standalone performance is degraded by the dependency gap.

### Catastrophic forgetting (positive result)

ID accuracy after fine-tuning on rotated data, measured on the original (unrotated) test set:

| Angle | WS | CL | OL | KD |
|---|---|---|---|---|
| 15° | -0.9pp | **+0.4pp** | -0.2pp | -1.0pp |
| 30° | -6.5pp | -5.7pp | -5.3pp | -6.1pp |
| 45° | -20.4pp | -20.5pp | -20.9pp | -20.9pp |
| 60° | **-38.6pp** | -39.3pp | -40.5pp | -40.0pp |
| 90° | **-48.9pp** | -52.0pp | -58.4pp | -58.1pp |

The forgetting gap scales with rotation angle and is largest at 90°, where WS retains 9.5pp more ID accuracy than OL. Absolute post-adaptation ID accuracy at 90°:

| Condition | Post-adapt ID acc | ID acc retained |
|---|---|---|
| WS | 49.2% | 50.2% of original |
| CL | 45.0% | 46.4% |
| OL | 38.9% | 40.0% |
| KD | 40.0% | 40.8% |

The ordering is **WS > CL > KD ≈ OL**. This tracks the **robustness** mechanism (CL co-training), not the distillation mechanism. KD — which has distillation but no CL — is indistinguishable from OL on forgetting, despite being tied with WS on zero-shot OOD accuracy. CL — which has CL co-training but no distillation — is second-best on forgetting despite being worst on zero-shot accuracy.

At 15°, CL actually *improves* ID accuracy after fine-tuning on rotated data (+0.4pp), suggesting the mild distribution shift acts as regularization for the CL model.

## Interpretation

### Three mechanisms, three metrics

The experiment cleanly dissociates three metrics into three mechanisms:

| Metric | Best conditions | Mechanism | Prior evidence |
|---|---|---|---|
| Zero-shot OOD | WS ≈ KD >> OL > CL | **Distillation** (soft-target regularization) | Val loss ordering from wake-sleep comparison |
| Adaptation speed | All equal | Neither | — |
| Forgetting resistance | WS > CL >> OL ≈ KD | **CL co-training** (loss landscape flatness) | Perturbation robustness, Hessian trace |

**Zero-shot OOD accuracy tracks distillation, not self-knowledge.** The KD baseline (external teacher, no self-reference) matches WS. Distillation produces smoother decision boundaries that generalize better to rotated inputs. This is a well-known property of knowledge distillation (soft targets provide richer supervision than one-hot labels) and is not specific to the cerebellar architecture.

**Adaptation speed is uninformative.** Despite 2–34× differences in perturbation robustness, all models reach the same gradient-descent speed. Perturbation robustness measures response to random displacement; gradient descent follows a structured direction. The Hessian flatness makes the model robust to noise at the injection point but doesn't change the gradient geometry for the adaptation task.

**Forgetting resistance tracks CL co-training.** The WS and CL models preserve more ID knowledge while adapting to OOD data. The KD baseline — with distillation but no CL — shows the same forgetting as OL, confirming that distillation alone does not produce this property. This is a direct consequence of the flatter loss landscape: gradient updates from OOD fine-tuning cause less collateral damage to existing representations. The model can move toward the new distribution without climbing over hills that would destroy its original solution.

### Connection to the robustness story

The forgetting result is the adaptation-speed experiment's contribution: it extends the perturbation robustness finding from random noise to structured gradient updates. The Jacobian analysis showed the CL model has 0.45× the Hessian trace at the injection point. The forgetting result shows this flatness has functional consequences beyond perturbation resistance — it reduces catastrophic interference during continual learning. The CL model's representations are organized in a way that OOD adaptation and ID preservation are less conflicting.

### Why WS > CL on forgetting despite CL > WS on perturbation robustness

Perturbation robustness measures raw loss-landscape flatness (where CL is best). Forgetting measures the *joint* outcome of adaptation quality + knowledge retention. WS combines CL's flatness with distillation's better starting point (98.1% vs 97.0% ID accuracy). The 1.1pp ID accuracy gap at baseline propagates through the entire adaptation trajectory, so WS retains more absolute accuracy even though CL has the flatter landscape. The WS model gets both mechanisms — organized representations from CL co-training AND smoother decision boundaries from distillation.

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/mnist_adaptation.py::a2a_mnist_adaptation
```

Results saved to `language-reduction-data` volume at `/data/a2a_forward/mnist_adaptation/vit_4L_4H_128D/`.
