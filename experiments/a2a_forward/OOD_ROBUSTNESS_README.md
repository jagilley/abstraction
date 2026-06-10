# OOD Robustness: What Kind of Self-Knowledge Survives Distribution Shift?

**Prior experiments**: [BASELINE_BATTERY_README.md](BASELINE_BATTERY_README.md) (checkpoints + ID robustness comparison), [JACOBIAN_ANALYSIS_README.md](JACOBIAN_ANALYSIS_README.md) (Hessian trace mechanism), [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md) (self-knowledge probes).

**Code**: `calibration_transfer.py` (Experiment A), `ood_robustness.py` (Experiment B).

## Motivation

The paper shows the model "knows what it knows": closed-loop co-training makes every layer encode where the model's computation will surprise a compressed model of itself. Two follow-up questions, both answered by taking the trained baseline-battery models out of distribution:

1. **Does the model know what it *doesn't* know?** (Experiment A) Epistemic self-knowledge — calibration about one's own competence — cannot be memorized: the boundary of your competence is by definition not in your training data. If closing the loop produced *epistemic* self-knowledge, competence probes should transfer across distribution shift better for the forward condition (cf. Kadavath et al. 2022, whose behaviorally-trained self-knowledge collapses OOD, Brier 0.15 → 0.43).

2. **Is forward prediction's robustness advantage made of something different from the baselines'?** (Experiment B) The paper's conclusion (§6) claims the forward model encodes the main model's *computational function* (weight-determined, distribution-invariant) while the autoencoder encodes the *activation manifold* (jointly determined by weights and data). On ID data this is unfalsifiable — forward 0.41× vs autoencoder 0.64× loss degradation is consistent with both "different kinds of flatness" and "more vs less of the same regularizer." The paper explicitly flags the test: *"If this account is correct, the gap should widen on novel inputs."* This experiment runs that test.

Both experiments reuse the baseline battery checkpoints (identical lr/seed/init, differing only in what was injected during training). No retraining.

## OOD corpora

2M GPT-2 tokens each, cached to the volume by `cache_ood_tokens` (`/data/ood_tokens/`). Shift severity is graded and measurable by the open-loop model's base loss:

| Corpus | What shifts | OL base loss | OL top-1 acc |
|---|---|---|---|
| ID test (FineWeb held-out) | nothing | 5.35 | 0.22 |
| Wikipedia-EN | register/format (still English prose) | 5.69 | 0.21 |
| Python code (codeparrot) | "language" statistics: indentation, brackets, rigid syntax | 6.48 | 0.30 |
| open-web-math | domain: LaTeX-heavy mathematical text | 6.99 | 0.16 |
| French Wikipedia | the natural language itself | 7.64 | 0.08 |
| Shuffled FineWeb | same ID tokens permuted — sequential structure destroyed, unigram stats controlled | 11.45 | 0.01 |

All corpora use the same tokenizer and window length: this is distribution shift over token sequences, not a modality change — the right kind of shift for a claim about the *activation manifold*. Code is the load-bearing corpus: strongly novel statistics but still computable structure (model accuracy actually rises while loss rises), and the corpus where perturbations hurt the open-loop model most (Δloss 2.6× its ID value).

## Experiment A: Calibration transfer (negative result)

**Design.** Linear competence probes: each layer's activations → the model's own forthcoming per-token LM loss. The probe target involves no forward model (it is the main model's own error); the forward model enters only as the training pressure that shaped the weights. Probes are closed-form ridge regressions trained on FineWeb val (~325K positions), then evaluated **frozen** on each OOD corpus. Primary metric: Spearman ρ between probe prediction and actual loss (scale-free, robust to the OOD loss mean-shift). Retention = mean OOD ρ / ID ρ (excluding shuffled). Conditions: the 5 battery conditions plus `forward_noinj` (forward weights, injection off).

**Retention by layer × condition:**

| Layer | open_loop | forward | fwd_noinj | shifted | random_proj | autoencoder |
|---|---|---|---|---|---|---|
| post_embed | 0.43 | 0.65 | 0.65 | 0.37 | 0.65 | 0.61 |
| post_block0 | 0.29 | 0.39 | 0.38 | 0.56 | 0.38 | 0.36 |
| post_block1 | 0.32 | 0.37 | 0.36 | 0.42 | 0.45 | 0.44 |
| post_block2 | 0.25 | 0.32 | 0.29 | 0.28 | 0.34 | 0.30 |
| post_block3 | 0.35 | 0.33 | 0.31 | 0.34 | 0.35 | 0.36 |

**Findings:**

1. **In-distribution, all conditions are identical** (ρ ≈ 0.32–0.38 at every layer). Closing the loop does not make the model's own future errors more linearly legible.
2. **OOD transfer shows a generic injection effect, not a forward effect.** Open-loop is generally the worst transferer, but forward ≈ random_proj ≈ autoencoder at every layer. This is the signature of the generic early-layer reorganization from the baseline battery (any used injection), with none of the forward-specific depth-uniformity that distinguishes the self-knowledge probes.
3. **Output entropy beats every activation probe**, both ID (ρ = 0.49 vs ≤ 0.38) and in retention (0.64–0.74 vs ≤ 0.65). At this scale the model's behavioral confidence is its best "knows what it doesn't know" signal.
4. **Code shows negative transfer** (ρ down to −0.28): ID-learned competence heuristics actively invert on code — except at post_embed for conditions whose gates opened during training.
5. **forward ≈ forward_noinj everywhere**: whatever is there is in the weights.

**Conclusion: cerebellar co-training does not produce epistemic self-knowledge.** The self-knowledge demonstrated in the paper is not calibration about competence. (A leading indicator was always present: the forward residual never correlated with LM loss.)

## Experiment B: OOD robustness (positive result — confirms the paper's prediction)

**Design.** Identical instrument to the baseline battery / Jacobian analysis, with the evaluation distribution as the new manipulated variable. 16 fixed random unit directions added to the residual stream entering block 2 (the injection point) at fixed absolute scale s × b1_std(ID, open-loop) — so all conditions and corpora receive *identical* perturbations. Conditions tested bare-weights (robustness is in the weights, per the battery); a `forward_inj` variant (natural mode) checks the real-time mirror. Per (condition, corpus): empirical Δloss per direction, response norm/cosine at post_block3, and Hessian trace at the injection point (Hutchinson, 10 Rademacher vectors × 10 batches, vectors shared across conditions).

**Δloss ratio vs open_loop (s=2.0; lower = more robust):**

| Condition | ID | Wikipedia | Code | Math | French |
|---|---|---|---|---|---|
| forward | 0.43 | 0.49 | **0.51** | 0.33 | −0.04 |
| forward_inj | 0.47 | 0.49 | 0.56 | 0.30 | 0.17 |
| shifted | 0.49 | 0.47 | **1.11** | 0.14 | 0.02 |
| random_proj | 0.68 | 0.65 | 0.69 | 1.00 | 0.37 |
| autoencoder | 0.65 | 0.70 | **0.93** | 0.57 | 0.49 |

**Hessian trace ratio vs open_loop:**

| Condition | ID | Wikipedia | Code | Math | French | Shuffled |
|---|---|---|---|---|---|---|
| forward | 0.38 | 0.38 | **0.45** | 0.22 | 0.38 | 0.54 |
| forward_inj | 0.47 | 0.44 | 0.59 | 0.34 | 0.48 | 0.53 |
| shifted | 0.51 | 0.49 | 0.51 | 0.37 | 0.44 | 0.55 |
| random_proj | 0.75 | 0.77 | 0.77 | 0.78 | 0.83 | 0.72 |
| autoencoder | 0.71 | 0.76 | **1.02** | 0.65 | 0.86 | 0.71 |

**Paired forward vs autoencoder (same 16 directions, s=2.0):**

| Corpus | fwd Δloss | ae Δloss | ae/fwd | fwd<ae dirs |
|---|---|---|---|---|
| ID test | 0.0048 | 0.0073 | 1.51 | 13/16 |
| Wikipedia | 0.0055 | 0.0078 | 1.42 | 10/16 |
| Code | 0.0148 | 0.0269 | **1.81** | 9/16 |
| Math | 0.0016 | 0.0028 | 1.74 | 10/16 |
| French | −0.0003 | 0.0044 | — | 9/16 |

**Findings:**

1. **The forward model's robustness is distribution-invariant.** It takes roughly half the open-loop model's damage on every corpus, and its loss landscape is ~0.4× as curved everywhere (tr(H) ratio 0.38–0.45 across English, code, French; 0.22 on math). The flatness travels with the weights.
2. **The autoencoder's robustness collapses off-manifold.** On code its Δloss ratio rises to 0.93 and its Hessian trace ratio to **1.02** — curvature indistinguishable from a model trained with no injection at all. Its ID flatness was a property of the activation manifold it learned, and off the manifold it is gone.
3. **The forward/autoencoder gap widens with shift** (ae/fwd degradation ratio 1.51 ID → 1.81 code; on French, forward takes zero average damage while the autoencoder still takes a hit), exactly as the paper's §6 prediction requires. Two independent measurements (empirical perturbation and Hutchinson curvature) agree.
4. **Random projection patterns with the autoencoder** (manifold-bound: tr(H) ratio 0.75–0.83 everywhere, advantage 1.00× on math), not with forward.
5. **The shifted condition is unstable**: robust on ID/Wikipedia/math but actively *worse than open-loop* on code (1.11×). The one injection the model learned to ignore has the least principled robustness profile.
6. **Robustness is in the weights OOD too**: forward_inj ≈ forward at every corpus.
7. **Hessian prediction quality**: E[ΔL] = ε²/(2d)·tr(H) matches empirical Δloss within ~25% on ID/Wikipedia/shuffled but underpredicts ~2× on code for all conditions — higher-order landscape terms matter more there. The tr(H) *ordering* across conditions tracks the empirical ordering everywhere.
8. Curiosity: on shuffled tokens, tr(H) is negative and perturbations *reduce* loss for every condition — structureless input sits at a qualitatively different part of the landscape.

**Conclusion.** Before this experiment we knew forward prediction produces the most ID robustness (a ranking on one distribution). Now we know forward's robustness is a constant of the weights while the autoencoder's was conditional on inputs staying on the training manifold. This is the first direct evidence for the computational-function-vs-activation-manifold distinction, converting the paper's §6 hedge ("the gap should widen on novel inputs") into a demonstrated result.

## Joint interpretation

The two experiments locate the paper's self-knowledge on the map: **computational, not epistemic; distribution-invariant, not data-bound.** The model does not transferably know *where it will fail* (Experiment A) — what it knows about itself is *its own computational structure*, and that knowledge is precisely the part that survives leaving the training distribution, uniquely for forward prediction among all injection types (Experiment B).

## Caveats

- The per-direction paired sign test (fwd < ae) is strong ID (13/16) but ~10/16 OOD: the widening mean gap OOD is driven by magnitude on some directions, not by forward winning more directions. The tr(H) gap is the more convincing OOD evidence.
- French and math have small open-loop Δloss denominators; their ratios are noisy. Code is the load-bearing corpus.
- Manifold displacement is inferred (from input shift + the base-loss gradient), not measured directly. A cheap direct check: the autoencoder's reconstruction MSE per corpus should rise with shift severity and peak on code.
- Calibration probes at 29M params / 10M tokens are heavily frequency-dominated; the calibration null might look different at scale.

## Modal volume

```
/data/ood_tokens/{wikipedia_en,french,code_python,math}.npy   # 2M tokens each
/data/a2a_forward/calibration_transfer/post_block0_to_post_block3/inject1/P_10000000/results.json
/data/a2a_forward/ood_robustness/post_block0_to_post_block3/inject1/P_10000000/results.json
```

## Reproduction

```bash
cd experiments/

# One-time: cache OOD corpora (idempotent)
modal run --detach a2a_forward/calibration_transfer.py::cache_ood_tokens

# Experiment A: calibration transfer
modal run --detach a2a_forward/calibration_transfer.py::a2a_calibration_transfer

# Experiment B: OOD robustness
modal run --detach a2a_forward/ood_robustness.py::a2a_ood_robustness
```
