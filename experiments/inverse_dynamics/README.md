# Inverse-Dynamics Self-Models Are Ill-Posed — and Grokking Makes Them Worse

**Conceptual parent:** the forward self-model program — [`experiments/a2a_forward/README.md`](../a2a_forward/README.md) and `papers/forward_self_models_paper1.md`[^private].
**Grokking recipe borrowed from:** `fer/experiments/zipfian_grokking/cnb_self_regulation/modal_app.py`[^private] (the `GrokMLP` on `(a+b) mod p`).
**Motivating development:** the July 2026 falsification of the Jacobian Conjecture.

> **One line:** A *forward* self-model (predict later activations from earlier ones) is always well-posed, because a layer is a function. An *inverse* self-model (predict earlier activations from later ones — the self-model analog of an inverse-dynamics world model) is ill-posed exactly to the degree the forward computation destroyed information. In a grokking MLP we watch this happen causally: **as the model groks, its representation compresses onto the task variable, and its inverse self-model collapses from R²≈0.88 to R²≈0.15 — while the forward self-model stays pinned at R²≈1.0 the whole time.** The ill-posedness tracks *representational compression*, not test accuracy: it keeps deepening after the model has already reached 100% test accuracy.

---

## 1. Motivation

### 1.1 The Jacobian Conjecture, and why it bears on world models

The Jacobian Conjecture (Keller, 1939) asserts that **local invertibility implies global invertibility** in the cleanest possible category: a polynomial map $F:\mathbb{C}^n\to\mathbb{C}^n$ whose Jacobian determinant is a nonzero constant must have a polynomial inverse. On 2026-07-19, Levent Alpöge (using Claude Fable) announced an explicit counterexample in $\mathbb{C}^3$: a polynomial map with **$\det JF \equiv -2$ everywhere** that is nonetheless **globally non-injective** — three distinct points collapse to one output,
$$F(0,0,-\tfrac14)=F(1,-\tfrac32,\tfrac{13}{2})=F(-1,\tfrac32,\tfrac{13}{2})=(-\tfrac14,0,0).$$
Every point is locally invertible (the inverse function theorem applies everywhere); the global inverse is a multivalued relation, not a function. It is the sharpest available statement that *a map can look invertible under every local diagnostic and still have no well-posed inverse* — and it needs no Jacobian degeneracy to do it ($|J|$ is bounded away from zero).

This is a prior against **inverse-dynamics models** in world modeling. An inverse-dynamics model bets that the transition determines the cause: learn $g$ with $a_t = g(s_t, s_{t+1})$, i.e. that the forward dynamics can be inverted to recover the action/latent. The JC result says invertibility can fail even when everything looks locally fine, and the failure is *silent* — regressing a single-valued $g$ onto a multivalued relation returns a smooth, confident, mode-averaged answer that is wrong on the collisions.

### 1.2 The self-model transposition

The `a2a_forward` program trains *forward* self-models: small nets predicting a main model's later-layer activations from earlier ones, learning an empirical approximation of the layer's computational function. Forward prediction is well-posed by construction — the layers compute a function, one input to one output.

The mirror-image object is an **inverse self-model**: predict *earlier* activations from *later* ones ("what internal state produced this?"). This is the self-model analog of inverse dynamics, and the JC argument predicts it should be ill-posed *exactly to the degree the forward computation is non-injective* (collapses a preimage fiber). This experiment tests that prediction and asks **what controls the degree of non-injectivity**.

---

## 2. Setup

### 2.1 Why modular addition, and why an MLP

Modular addition is the cleanest ground-truth analog of the JC collision. A model computing $(a+b)\bmod p$ maps all $p$ pairs sharing a sum to the **same** output class. The preimage fiber of output $c$ is exactly $\{(a,b): a+b\equiv c \pmod p\}$, size $p$ — **the collision structure is known in closed form** ($p$ fibers, each of size $p$; here $p=97$, so $9409$ pairs collapse to $97$ sums).

We deliberately use a **non-residual MLP** (the `GrokMLP` from the cnb grokking work), not a transformer. A transformer's residual stream ($a_j = a_i + \sum\Delta$) is *engineered* to be approximately invertible — the early state is additively preserved in the late state — which would *hide* the effect. A plain MLP has no such highway, so the collapse of the fiber is real. (See §5.3 for why this cuts the other way for real transformers.)

- **Main model:** `GrokMLP`, `layer0: 2p→128`, `layer1: 128→128`, `head: 128→p`, ReLU. One-hot $(a,b)$ input. Recipe verbatim from cnb: 30% train split, full-batch AdamW, `lr=1e-3`, `weight_decay=1.0`, 40k epochs. It groks: test acc 1.0, grok onset ≈ epoch 15k.
- **Cached representations** (all $p^2$ pairs): `h1 = relu(layer0(x))` (early, pair-specific), `h2 = relu(layer1(h1))` (middle), `logits = head(h2)` (late, collapses onto the sum).

### 2.2 The self-models and the metrics

For a layer pair `(early, late)` we train, with identical architecture, a **forward** self-model (`early→late`) and an **inverse** self-model (`late→early`) — a ReLU MLP, capacity swept over hidden widths `{8,32,128,512,2048}`, MSE loss, evaluated on a held-out 20% of the $p^2$ pairs. We measure:

- **Test R² / cosine** (forward vs. inverse; the asymmetry).
- **Capacity scaling** — forward should saturate toward ~1.0 (as in the paper's §3.2); an ill-posed inverse should *plateau, or degrade*, with capacity.
- **Fiber-mean baseline R²** — the R² of a predictor that sees *only* the sum $c$ and outputs the target's per-fiber mean. This equals the **between-fiber variance fraction** of the target. If `inverse R² ≈ fiber-mean R²`, the inverse recovers *nothing beyond the preimage centroid* — the JC mode-averaging failure.
- **Model-free fiber collapse** — for each representation, the **within-fiber variance fraction** (fraction of variance surviving after conditioning on $c$; 0 = fully collapsed onto $c$). This is our proxy for representational compression.
- **Effective-rank collapse** of the predictions vs. the true target.

---

## 3. Results

Runs are on `p=97`, seed 42 (replication across seeds 0–2: §4). Everything runs locally in the `glp` conda env (MPS); no Modal. Grokking + full analysis is ~3 min.

### 3.1 The grokked model: forward is perfect everywhere; the inverse breaks at the readout

**Model-free fiber collapse** (grokked model) — how much of each representation survives conditioning on $c$:

| representation | within-fiber (pair-specific) | between-`c` (collapsed) | eff. rank |
|---|---|---|---|
| input | 1.00 | 0.00 | 192/194 |
| `h1` (early) | **0.89** | 0.11 | 88/128 |
| `h2` (middle) | 0.085 | 0.91 | 17/128 |
| `logits` (late) | 0.060 | 0.94 | 14/97 |

**Forward vs. inverse, capacity sweep** (test R²):

| width → | 8 | 32 | 128 | 512 | 2048 |
|---|---|---|---|---|---|
| forward `h1→logits` | 0.68 | **1.00** | 1.00 | 1.00 | 1.00 |
| **inverse `logits→h1`** | 0.10 | 0.24 | **0.30** | 0.28 | 0.25 |
| forward `h1→h2` | 0.61 | 0.99 | 1.00 | 1.00 | 1.00 |
| inverse `h2→h1` | 0.15 | 0.47 | 0.79 | 0.85 | 0.85 |

All three predicted signatures of ill-posedness appear on `logits→h1`:

1. **Asymmetry.** Forward saturates to R²≈1.0 by width 32; the inverse never exceeds ~0.30 with 60× more parameters.
2. **Capacity doesn't help — it hurts.** The inverse *peaks at width 128 then degrades* (0.30 → 0.28 → 0.25). A hard-but-well-posed problem improves with capacity; an ill-posed one overfits within-fiber noise — its prediction effective rank balloons (7 → 59) while R² falls. This is the cleanest fingerprint separating ill-posedness from mere difficulty.
3. **Mode-averaging.** The inverse recovers only ~0.13 beyond the fiber-centroid baseline (0.124), i.e. it is largely returning the mean over the preimage set. Its predictions are rank-collapsed: eff. rank ~59 vs. the true `h1`'s ~95, even at maximum capacity.

### 3.2 The nuance that matters: information loss, not variance collapse

The effect is **graded by information loss, and information loss is not the same as variance collapse.** Look at `h2→h1`: `h2` has **91% of its variance explained by $c$**, yet the inverse still recovers **R²=0.85**. The pair information survives in `h2`'s *low-variance* directions; a single ReLU layer only loses ~15%. The genuine destruction happens at the **readout** (`head`: `h2→logits`), and that is the only place the inverse truly collapses.

So "the representation has collapsed onto $c$ (in variance)" does **not** imply "the map is non-invertible." Only actual information loss does. This is itself the JC lesson one level down: *local/statistical diagnostics do not certify global invertibility — you have to measure the achievable inverse.* It also means our model-free within-fiber-variance metric is a **necessary-not-sufficient** signal for inverse failure, which we state rather than paper over.

### 3.3 The causal result: grokking induces the ill-posedness

![Grokking induces inverse ill-posedness](results/p97_trajectory/trajectory.png)

Snapshotting the model along the grokking curve and running the full forward/inverse analysis at each (widest self-model shown):

| epoch | test acc | `logits` within-fiber | forward `h1→logits` R² | **inverse `logits→h1` R²** | inverse `h2→h1` R² | beyond fiber-centroid |
|---|---|---|---|---|---|---|
| 5,000 | 0.02 (**memorized only**) | 0.71 | 1.00 | **0.88** | 0.96 | +0.82 |
| 12,000 | 0.69 | 0.29 | 1.00 | **0.82** | 0.95 | +0.75 |
| 18,000 | 0.99 | 0.11 | 1.00 | **0.68** | 0.93 | +0.58 |
| 25,000 | 1.00 | 0.06 | 1.00 | **0.30** | 0.86 | +0.18 |
| 40,000 | 1.00 | 0.05 | 1.00 | **0.15** | 0.79 | **+0.015** |

Three things to read off this:

- **A clean memorization control (epoch 5,000).** The model has fully *memorized* (train acc 100%) but not *generalized* (test acc 2%). Its inverse works (R²=0.88): a memorizing model keeps its internal state approximately invertible because it stores per-pair information. Then as it groks, the inverse collapses in lockstep to 0.15, while the forward stays at 1.00. **Same architecture, same data — the only thing that changed is that the model learned to generalize, and that alone destroyed invertibility.**
- **The mechanism is compression** (right panel of the figure). Inverse R² falls monotonically with the readout's within-fiber variance fraction — as the representation compresses onto $c=a+b$ and discards *which pair* produced it.
- **It tracks compression, not accuracy.** The inverse keeps collapsing *after generalization is complete*: test accuracy saturates by ~epoch 18k, but between 18k and 40k the inverse still falls 0.68 → 0.15 (and readout within-fiber drops 0.11 → 0.05). That is the weight-decay "cleanup" phase of grokking — the model keeps *purifying* its representation long after it is already correct. The ill-posedness is a **representational** phenomenon, not a performance one. And it is a clean dose-response by depth: the hidden-state inverse `h2→h1` degrades mildly (0.96 → 0.79), the readout inverse `logits→h1` severely (0.88 → 0.15).

---

## 4. Robustness

Replicated across seeds {0, 1, 2} (plus canonical seed 42), using each run's best (highest test-acc) grokked checkpoint and the widest self-model. Every seed groks (test acc ≥ 0.996, grok onset epoch 14k–16k) and shows the same asymmetry:

| seed | forward `h1→logits` R² | inverse `logits→h1` R² | inverse `h2→h1` R² |
|---|---|---|---|
| 0 | 0.998 | 0.214 | 0.799 |
| 1 | 0.998 | 0.420 | 0.895 |
| 2 | 0.998 | 0.251 | 0.865 |
| 42 | 0.998 | 0.277 | 0.847 |
| **mean ± sd** | **0.998 ± 0.000** | **0.29 ± 0.08** | **0.85 ± 0.04** |

The forward self-model is essentially seed-invariant at R²≈1.0. The readout inverse (`logits→h1`) is severe and well below forward at every seed — with a fiber-mean baseline of ≈0.12–0.13, it recovers only ~0.15 beyond the preimage centroid on average. The hidden-state inverse (`h2→h1`) is consistently intermediate (~0.85). These are best-checkpoint values; the fully-compressed epoch-40k endpoint (§3.3) drives `logits→h1` lower still (0.15), consistent with the compression story. The effect is not a single-seed artifact.

---

## 5. Interpretation

### 5.1 What it says about the forward self-model program

The paper's central claim is a **dissociation between representation and computation**: forward self-models compress a layer's *computational function* cheaply because they are handed the representation and only have to model the forward map. This experiment supplies the mirror-image claim and a mechanism:

> Forward self-models are well-posed because a layer is a function; inverse self-models have no such guarantee, and **the better the main model generalizes — i.e. the more it compresses its representation onto the task-relevant variable — the worse its inverse self-model becomes.**

That reframes the forward framing from *convenient* to *principled*. It also predicts a hazard for any future "inverse self-model" variant (predict earlier activations from later, or infer a latent "what did this layer do" code from input/output pairs): it inherits this ill-posedness by default, and it degrades precisely as the host model gets better.

### 5.2 What it says about the Jacobian-Conjecture argument for world models

This is the JC lesson made concrete on a neural network. Inverse dynamics bets on invertibility; here the forward computation becomes non-injective *as a direct consequence of learning the clean, compressed dynamics you actually want*. The inverse-dynamics model degrades exactly as the world-model improves. Local diagnostics (variance structure — the analog of Jacobian conditioning) fail to warn you: `h2` looks 91%-collapsed yet inverts fine, while the genuine collapse is elsewhere. You have to measure the achievable inverse.

### 5.3 The residual stream is the missing 2×2 cell (next experiment)

We chose a non-residual MLP so the collapse would be visible. Real transformers use a residual stream, which is *engineered* to preserve the early state additively — so the same forward self-models that this program relies on transfer cleanly, and an inverse self-model over a residual gap should be **much less** ill-posed. The natural completion is the 2×2 (info-destroying vs. info-preserving computation) × (forward vs. inverse): rerun this battery over a language model's `post_block_i ↔ post_block_j` (adapting `a2a_forward/scaling_sweep.py`). Predicted result: forward ≈ inverse, both fine — the residual architecture is itself a partial mitigation of the JC problem, which is *why* forward self-models work in the paper.

---

## 6. Limitations

- **One data-generating process.** Modular addition is the cleanest collision structure but also the most extreme. The residual-stream contrast (§5.3) is needed to show the effect is modulated by architecture, not just present in one toy.
- **Readout vs. hidden-state.** The *severe* effect is at the readout (`logits→h1`); the hidden-state inverse (`h2→h1`) is only mildly ill-posed because this shallow MLP concentrates its information destruction at the head. A deeper grokking MLP would give a hidden→hidden gap that collapses, letting the "inverse *dynamics*" claim stand without leaning on the readout.
- **MLP, not transformer.** The `GrokMLP` groks without learning the clean Fourier algorithm a transformer would (per the cnb notes). The fiber-collapse result is task-defined (the output *is* $c$) so it holds regardless, but the internal geometry differs from a transformer's.

---

## 7. Reproduction

Runs locally in the `glp` conda env (torch 2.7.0, MPS). No Modal.

```bash
cd experiments/inverse_dynamics

# Grokked-model forward-vs-inverse capacity sweep (single model, full width sweep):
conda run -n glp python grokking_fwd_vs_inv.py \
    --p 97 --n-epochs 40000 --sm-steps 4000 --widths 8,32,128,512,2048 \
    --out results/p97_seed42
# (or: bash train.sh)

# Trajectory: analyze forward/inverse at snapshots along the grokking curve:
conda run -n glp python grokking_fwd_vs_inv.py \
    --p 97 --n-epochs 40000 --snapshots 5000,12000,18000,25000 \
    --sm-steps 4000 --widths 32,128,512 --out results/p97_trajectory

# Figure:
conda run -n glp python plot_trajectory.py

# Multi-seed replication:
for s in 0 1 2; do conda run -n glp python grokking_fwd_vs_inv.py \
    --p 97 --seed $s --widths 32,128,512,2048 --out results/p97_seed$s; done
```

## 8. Files

| file | purpose |
|---|---|
| `grokking_fwd_vs_inv.py` | Grok a `GrokMLP` on `(a+b) mod p`; cache activations; model-free fiber diagnostics; forward/inverse self-model capacity sweep; optional grokking-trajectory mode (`--snapshots`). |
| `plot_trajectory.py` | 2-panel figure: inverse ill-posedness vs. epoch, and vs. readout compression. |
| `train.sh` | Canonical grokked-model run, tees output to `logs/`. |
| `results/p97_seed42/` | Grokked-model full-width capacity sweep (§3.1). |
| `results/p97_trajectory/` | Grokking-trajectory analysis + `trajectory.png` (§3.3). |
| `results/p97_seed{0,1,2}/` | Replication seeds (§4). |
| `logs/` | Captured run output. |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
