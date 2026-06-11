# Representational Divergence Analysis: Open-Loop vs Closed-Loop (2026-06-02)

**Code**: `representational_divergence.py`
**Prior experiment**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md) (Run 6 — controlled retrain establishing the self-knowledge probe gap)

## Motivation

The controlled retrain (Run 6) showed that the closed-loop model encodes the forward model's residual dramatically better than the open-loop model (Δ R² ~ +0.18 at every layer). But the R² gap was roughly uniform across layers, which seemed to rule out layer-specific structure. This left an open question: is the reorganization the same everywhere, or do different layers change for different reasons?

Linear probes measuring a single target (residual R²) can't distinguish between "this layer reorganized *for* self-knowledge" and "this layer reorganized for other reasons and self-knowledge came along for the ride." We need to look at the full representational geometry.

## Method

Load the controlled retrain checkpoints (identical lr=3e-4, seed=42, differing only in whether the cerebellar loop is closed). Run both models on the same 327,680 eval positions. Three analysis steps:

1. **CKA** between open-loop and closed-loop activations at each layer — where do the representations diverge?
2. **PCA on the activation difference** (closed - open) at each layer — is the divergence concentrated or diffuse? What directions?
3. **Alignment between divergence directions and self-knowledge directions** — did the model reorganize *along* the dimensions that encode what the forward model misses?

Step 3 is the key test. If the divergence directions carry self-knowledge (ratio > 1), the model literally reorganized its representations along self-knowledge-relevant dimensions. If ratio ~ 1, the self-knowledge is incidental to whatever else changed.

## Results

### Step 1: CKA — monotonic divergence through the network

| Layer | CKA |
|---|---|
| post_embed | 0.972 |
| post_block0 | 0.882 |
| post_block1 | 0.819 (injection point) |
| post_block2 | 0.752 |
| post_block3 | 0.727 |

The representations diverge progressively from input to output. Even post_block0 (pre-injection, changes only via backprop) is substantially different (CKA 0.88). By post_block3, CKA is 0.73 — the two models encode substantially different geometric structure by the final layer.

### Step 2: PCA — early-layer change is concentrated, late-layer change is diffuse

| Layer | Eff rank | Top-10 frac | Relative diff norm | Rank for 50% |
|---|---|---|---|---|
| post_embed | 225.3 | 0.106 | 0.35 | 79 |
| post_block0 | **164.6** | **0.167** | 0.75 | 47 |
| post_block1 | 189.5 | 0.155 | 0.88 | 54 |
| post_block2 | 218.6 | 0.116 | 0.89 | 72 |
| post_block3 | 226.6 | 0.106 | 0.88 | 79 |

Post_block0 has the **most concentrated** divergence: lowest effective rank (164.6), highest top-10 fraction (16.7%), and only 47 PCs needed for 50% of the variance. The difference at this layer is also massive — 75% the magnitude of the activations themselves.

Later layers have more diffuse divergence (eff_rank 219-227), approaching the full-rank character of the forward model residual itself.

### Step 3: The key result — alignment grows through the network

**Residual variance captured by top divergence PCs** (ratio vs random baseline):

| Layer | Top-5 ratio | Top-10 ratio | Top-20 ratio |
|---|---|---|---|
| post_block0 | 1.10x | 1.07x | 1.07x |
| post_block1 | 1.36x | 1.32x | 1.23x |
| post_block2 | **1.87x** | **1.69x** | **1.55x** |
| post_block3 | **2.42x** | **2.19x** | **1.96x** |

At post_block3, the top-5 divergence directions capture **2.42x more** forward-model residual variance than random directions. At post_block0, the ratio is barely above 1.

**Extra self-knowledge overlap** (overlap between the *extra* probe weight directions — closed minus open probe SVD — and the divergence PCs):

| Layer | Rank-5 ratio | Rank-10 ratio | Rank-20 ratio |
|---|---|---|---|
| post_block0 | 0.73x | **0.49x** | **0.48x** |
| post_block1 | 1.65x | 1.12x | 0.84x |
| post_block2 | 1.76x | 1.35x | 1.20x |
| post_block3 | 1.65x | 1.28x | 1.28x |

At post_block0, the extra-SK directions are **below random** (0.49x at rank-10). The directions where the early layer changed most are *orthogonal* to the directions that carry its self-knowledge. At later layers (post_block1 onward), the overlap is above random.

**Projection probe confirmation** (R² predicting residual from acts projected onto diff PCs vs random directions):

| Layer | R² diff PCs | R² random | Δ | (at rank 50) |
|---|---|---|---|---|
| post_block0 | 0.154 | 0.128 | +0.026 | |
| post_block1 | 0.173 | 0.133 | +0.040 | |
| post_block2 | 0.195 | 0.150 | +0.045 | |
| post_block3 | 0.232 | 0.166 | **+0.067** | |

Consistent with the variance ratio results: diff PCs are increasingly better than random for residual prediction as you go deeper.

### Residual probe R² (replication of Run 6)

| Layer | Open R² | Closed R² | Δ R² |
|---|---|---|---|
| post_block0 | 0.024 | 0.210 | +0.187 |
| post_block1 | 0.061 | 0.263 | +0.202 |
| post_block2 | 0.157 | 0.344 | +0.187 |
| post_block3 | 0.259 | 0.422 | +0.163 |

Replicates the Run 6 finding: Δ R² is roughly uniform across layers (~+0.18). But this analysis reveals that the *nature* of the representational change underlying that uniform Δ R² is qualitatively different at early vs late layers.

## Interpretation

Two distinct phenomena drive the closed-loop model's representational reorganization:

**1. Early layers (post_block0): Large, concentrated, non-self-knowledge change.** The biggest, most structured representational change happens at the first transformer layer. The divergence is concentrated (eff_rank 164.6), large (75% of activation norm), but NOT aligned with self-knowledge directions (extra-SK overlap 0.49x, below random). This layer reorganized extensively via backprop to produce better inputs for the downstream injection-processing layers. The self-knowledge it encodes (Δ R² = +0.19) lives in *quieter* directions, orthogonal to the dominant change.

**2. Late layers (post_block2, post_block3): Diffuse, self-knowledge-aligned change.** At later layers, the divergence is more diffuse (eff_rank 219-227) but strongly aligned with self-knowledge (2.4x ratio at post_block3). The representations reorganized primarily along directions that encode what the forward model misses. These layers directly process the injected prediction, so their reorganization is driven by the need to evaluate and act on it.

This resolves an apparent contradiction from Run 6. The Δ R² was uniform across layers, which seemed to suggest the same reorganization happened everywhere. In fact, different layers changed for different reasons:
- Block 0 changed *a lot* but for general-purpose reasons; self-knowledge is a side-effect in low-variance directions
- Blocks 2-3 changed less concentratedly, but the change *is* the self-knowledge

The controlled retrain's linear probes couldn't distinguish these because they only measured a single target (residual R²). The representational divergence analysis reveals the *geometry* of the change — not just whether self-knowledge is present, but whether the dominant representational change *is* the self-knowledge.

### Biological parallel

This maps onto the distinction between cortical reorganization driven by cerebellar input vs. metacognitive encoding:
- Early processing stages (analogous to sensory/association cortex) change their representations to better serve the cerebellar prediction loop, but this change is primarily about input formatting, not self-awareness
- Later processing stages (analogous to prefrontal/executive areas) reorganize specifically around the error structure — their dominant change is metacognitive

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/representational_divergence.py \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```

---

# Prediction Trust: what form the self-knowledge takes (2026-06-10)

**Code**: `prediction_trust.py` (`run_prediction_trust.sh`)
**Builds on**: this analysis (the closed-loop model reorganizes *along* the forward model's residual directions at late layers) and `JACOBIAN_ANALYSIS_README.md` (directional-sensitivity machinery).

## Motivation

The divergence analysis above established *that* the closed-loop model reorganizes along self-knowledge (residual) directions, especially late in the network. It did not establish *what form* that self-knowledge takes — what the model is actually encoding about its own computation, and whether it is functionally used direction-by-direction.

Treat the forward model as a compressed model of the main model's own computation over blocks 1–3. Two facts let us turn "self-knowledge" into a closed-form, falsifiable question:

1. The injection is **additive** (`h ← h + G·p`), so perturbing the prediction `p` along a direction `v` is identical, within one forward pass, to perturbing the residual stream along `G·v`. The model cannot tell the two apart. The system's per-direction "usage gain" therefore factorizes exactly into a **gate** factor `‖Gv‖` (static, learned) and a **downstream** factor `S_down` (blocks 2–3's output sensitivity along `Gv`, learned).
2. The forward model's per-direction reliability is computable in closed form from its error statistics: the Wiener (optimal-linear) gain `Cov(v·tgt, v·pred) / Var(v·pred)`, the SNR `Var(v·tgt)/Var(v·res)`, and the raw error variance `Var(v·res)`.

So we can ask: **do the gains the model actually implements track the forward model's reliability spectrum?** A discounting/Wiener account predicts the model *suppresses* the directions where its self-model is unreliable. An error-monitoring account predicts the opposite — the model is *most* responsive there, because that is where it must supply its own complementary computation.

## Method

All measurements on the controlled-retrain checkpoints (identical lr=3e-4, seed=42, init; differ only in whether the loop is closed). No retraining. Three conditions: **CL+M** (closed-loop, injection active), **CL−M** (closed-loop weights, injection removed), **OL** (open-loop). OL and CL−M are measured along the *same* `G·v` directions taken from the closed-loop gate, so any CL-specific structure is attributable to training with the channel.

- **Phase 1 — reliability statistics.** Collected on the injection-active forward pass (the distribution the forward model trains against). Per-direction Wiener/SNR/var_res computed in the residual-covariance eigenbasis (`res_eig`) and a random orthonormal basis (null).
- **Phase 2 — measured usage gains.** `‖Gv‖` read directly from the gate weights; `S_down` measured by central-difference directional Jacobians of the logits (RMS logit movement per unit ε, 256 directions × 3 conditions). Linearity verified (max relative deviation between ε and ε/2: 0.0000).
- **Phase 3 — correlations** between measured gains and normative reliability, with partial correlations controlling for prediction/target variance and permutation p-values.
- **Phase 4 — counterfactual error injection (Test B).** Matched-norm corruptions along the gate-images of the top-16 residual eigenvectors (habitual-error directions), bottom-16 (trusted directions), and 16 random directions, at three scales; measure Δloss and KL.

## Results

### The forward model is unbiased but heteroscedastic

| Basis | Wiener gain range | var_res range | SNR range |
|---|---|---|---|
| res_eig | 0.955 – 1.016 | 0.038 – 0.577 (**15.4×**) | 4.4 – 14.0 |
| random | 0.981 – 1.016 | 0.156 – 0.206 (1.3×) | 5.8 – 8.9 |

The forward model's prediction is essentially the **unbiased conditional mean** of the main model's later computation (Wiener gain ≈ 1 in every direction — there is nothing for an optimal linear filter to correct). All the informative structure is in error *magnitude*, which spans 15× across the residual eigenbasis. The residual `r = a_j − f(a_i)` is, to good approximation, an **innovation**: the directionally-structured part of the main model's own computation that a compressed model of it cannot anticipate.

### The model amplifies its self-model's blind spots (error-monitoring, not Wiener filtering)

| Measure (res_eig basis) | Correlation | perm p |
|---|---|---|
| gate `‖Gv‖` vs Wiener gain | **−0.340** | 0.0005 |
| gate `‖Gv‖` vs log SNR | −0.385 | — |
| gate `‖Gv‖` vs **var_res** | **+0.602** | — |
| gate `‖Gv‖` vs Wiener, partialling out var_pred & var_tgt | −0.212 | — |

The gate's learned gain points its amplification *at* the directions where the forward model habitually errs (var_res r = +0.602) and *away* from where it is reliable (Wiener r = −0.340). The naive discounting account is falsified by sign. In the random basis every correlation collapses (gate vs Wiener r = −0.054, p = 0.39), confirming the structure is specific to the residual eigenbasis.

Downstream sensitivity tells the same story, with a clean CL-specific increment:

| Condition | S_down vs var_res (r) | (Spearman ρ, p) |
|---|---|---|
| CL+M | +0.368 | +0.422, 0.0005 |
| CL−M | +0.379 | +0.435, 0.0005 |
| OL | +0.250 | +0.310, 0.0005 |

There is a generic landscape component (OL already responds more along high-variance directions, r = +0.250), but the closed-loop model is differentially steeper: the **CL/OL sensitivity-ratio profile correlates with var_res at r = +0.293 (p = 0.0005)**, null in the random basis (−0.032, p = 0.61). As with every prior in-the-weights result, runtime injection is irrelevant (CL−M ≈ CL+M).

### Counterfactual error injection confirms it behaviorally

Matched-norm corruptions; KL is output movement (higher = bigger effect):

| Direction set | mean `‖Gd‖` | var_res | KL @ s=1.0 (CL+M) |
|---|---|---|---|
| error_top (top-16 res eigvecs) | 0.219 | 0.420 | 0.00036 |
| trusted_bottom (bottom-16) | 0.188 | 0.073 | 0.00027 |
| random | 0.186 | 0.175 | 0.00026 |

**KL(trusted)/KL(error) ratio**, constant across all three corruption scales:

| Condition | ratio |
|---|---|
| CL+M | 0.742 |
| CL−M | 0.732 |
| OL | 0.768 |

Corrupting along habitual-error directions moves the output **more** than along trusted directions (ratio < 1), and the closed-loop model pushes this further than open-loop (0.73–0.74 vs 0.77) — the error-monitoring geometry, sharpened by closing the loop.

### The two instruments agree quantitatively (the gauge-symmetry-grade result)

The small-ε Jacobian gains (Test A) predict the finite-ε behavioral ratios (Test B) within ~5%:

| Condition | (S_full bot/top)² [predicted KL ratio] | measured KL ratio |
|---|---|---|
| CL+M | 0.702 | 0.742 |
| CL−M | 0.695 | 0.732 |
| OL | 0.729 | 0.769 |

And the differential decomposes cleanly across the two learned factors. At the spectrum extremes the **gate carries most of it** (gate-gain bottom/top = 0.858, squared = 0.737 ≈ the whole effect), with downstream layers adding the smaller CL-specific increment (S_down bottom/top: 0.973 CL vs 0.990 OL). The trust policy is concretely localized — readable off the gate's weight spectrum, with a measurable echo in blocks 2–3.

## Interpretation: the innovation map

Because the forward model is the unbiased conditional mean of the main model's later computation, the residual is the **innovation** — and its covariance is directionally structured (15× spread), with the high-variance directions being the genuinely hard computations (delimiter matching, multi-source attention; cf. the behavioral-residual analysis in the main README). By organizing its gate spectrum and downstream sensitivity around `Cov(r)`, the main model encodes the **second-moment structure of its own innovation**: a directional map of which components of its own layer-to-layer transformation are compressible/routine versus incompressible/novel.

The *sign* is the tell. If the injection were a noisy answer, the optimal move is to suppress the high-error directions (Wiener filtering). The model does the opposite. That is only rational if the injection along blind-spot directions is a **reference to compute against**, not an answer to trust: to difference your true computation against a reference, you want the reference at high gain exactly where you will need to correct it. The innovation along those directions is either coherent-and-interesting or incoherent noise, and the model rationally dedicates downstream capacity to deciphering which. "Amplify where the self-model is blind" is the functional signature of *complementing* a predictor rather than *deferring* to it — the directional, causal counterpart of the division-of-labor account (cheap model handles the predictable bulk; the main model concentrates its own sensitivity on the residual). It is also why net LM loss stays flat while reliance grows: the model relocates the seam between "offloaded" and "mine," and the seam's location — drawn along the forward model's error spectrum — *is* the self-knowledge.

This unifies with the divergence result above: the same residual eigenstructure the **activations** come to encode (late-layer reorganization along residual directions; the Δ R² probe gap) is the structure the **weights** come to amplify (gate + downstream gains along high-var_res directions). Representation plus use — two readouts of one reorganization, in two different objects.

This is **predictive-coding-shaped** (Rao–Ballard / Friston): the forward model supplies a prediction and the downstream layers behave like error-units, most responsive on unexplained variance. The wiring differs (additive injection into the same stream, not a separate error channel), so it is an analogy, not an identity. It also reframes the calibration-transfer null (`OOD_ROBUSTNESS_README.md`): the self-knowledge lives in **computation-space** (which parts of my transformation are novel), not **outcome-space** (will I get the next token right). A scalar loss-calibration probe was looking in the wrong space.

## Caveats

1. **Causal arrow / equilibrium ambiguity.** Co-training gives the fixed point. "Encodes the error covariance and allocates sensitivity to the innovation" is a functional reading of a correlation-plus-control, not a demonstrated mechanism — amplifying a direction could itself raise its residual variance. The downstream CL/OL-ratio result is somewhat insulated (per-unit-norm, OL control); the gate↔var_res headline is fully exposed. Clean test: gate-spectrum surgery (flatten per-direction gains at eval, preserving total norm, and measure where LM loss degrades).
2. **var_res vs var_tgt.** The partial correlation controlling for prediction/target variance weakens but keeps the sign (−0.212). In a full-rank diffuse residual these do not separate cleanly, so state it as "organized around innovation magnitude," not a pristine dissociation from activation magnitude.
3. **Untested mechanistic prediction this generates.** If the amplified injection along blind-spot directions is a reference to difference against, there should be a **comparator** in blocks 2–3 whose output is a function of `(actual − injected)` concentrated in the high-var_res directions. We now know which directions to look in.

## Reproduction

```bash
cd experiments/
bash a2a_forward/run_prediction_trust.sh
# or:
modal run --detach a2a_forward/prediction_trust.py::a2a_prediction_trust \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
