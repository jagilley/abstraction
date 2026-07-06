# Synthetic-input experiment: can a forward self-model learn a layer's *program* off-manifold?

**Prior README**: [README.md](README.md) (main A2A forward model index)
**Code**: `synthetic_input.py` (Gaussian ladder + perturb + real), `synthetic_input_manifold.py` (manifold-matched rungs: GMM, mixup)
**Origin**: a friend's "Idea A" — if you train a forward self-model of a layer on *off-manifold* inputs (random noise, or hand-crafted vectors), does it still predict correctly for *on-manifold* inputs? I.e. can you learn the "program" the layers compute without ever visiting the activation manifold?

## Question

The forward self-model is trained on pairs `(a_i, a_j)` where `a_j = f(a_i)` and `f` is a frozen block (or stack of blocks) of the main model. Normally `a_i` is a **real** activation — one the main model actually produces on text. But `f` is just a deterministic function; you can evaluate it on *any* `(B, T, C)` tensor. So: replace the training input distribution with synthetic samples, train the forward model on `input → f(input)`, and **evaluate every model on the same held-out real `(a_i, a_j)` pairs**. How much of the on-manifold ceiling can an off-manifold-trained self-model recover?

## Setup

- **Frozen main model**: reused 29M checkpoint (`a2a_forward/transformer/P_10000000`, 4L/4H/256D). Never trained here.
- **The program `f`**: the frozen block(s) mapping `predict_from → predict_to`.
  - 1-block: `f = block1` (`post_block0 → post_block1`), forward model = 1-layer transformer (330K params). Ceiling matches the main README's baseline (full cosine 0.972).
  - 3-block: `f = block1∘block2∘block3` (`post_block0 → post_block3`), forward model = 2-layer transformer. The deeper "program."
- **Forward model**: fresh `TransformerForwardModel`, identical init across conditions (same seed), trained 10K steps on MSE against `f(input)`.

### The ladder of input distributions

Mean is matched for every Gaussian variant, so the ladder isolates *how much manifold structure the training inputs must carry*:

| condition | what it is |
|---|---|
| `gaussian_iso` | matched mean + scalar variance (isotropic noise — crudest) |
| `gaussian_global` | matched global mean + full covariance |
| `gaussian_perpos` | matched **per-position** mean + covariance (best per-position marginal) |
| `gmm_global` | fitted mixture of Gaussians (K=64 K-means + per-cluster full cov) — multimodal marginal |
| `mixup` | convex combinations of pairs of **real** activation *sequences* (near-manifold; preserves cross-position joint structure) |
| `perturb_1.0`, `perturb_0.5` | real activation + isotropic noise at `α·‖a‖` (near-manifold) |
| `real` | trained on real activations — the ceiling |

### Why update-space metrics

A transformer block is near-identity (`block(x) = x + small update`), so full-activation cosine is lenient — predicting identity already scores high (identity `cos(a_i, a_j)` = **0.598** at 1 block, **0.319** at 3 blocks). We therefore report metrics on the **update** `Δ = a_j − a_i` (the block's genuine contribution): `update_cosine` and `update_r2`. These isolate whether the model learned the *computation*, not the residual-stream carry-through, and are the metrics to read.

### Comparison validity

`synthetic_input_manifold.py` runs **only** the new manifold conditions but reuses the identical eval pipeline (same seed, `n_stat_batches`, `n_eval_batches`). Because `real_eval` depends only on CPU `torch.randint` token indices + deterministic model forward, it reproduces byte-for-byte. This is verified at runtime by an `identity_full_cosine` **gate** against the saved run — both depths matched exactly (0.5976, 0.3189), so the new rungs slot directly into the saved ladder.

## Results

### 1-block program (`f = block1`) — identity full-cos 0.598

| condition | full_cos | upd_cos | **upd_r2** | full_r2 | gen→iso | gen→perpos |
|---|---|---|---|---|---|---|
| `gaussian_iso` | 0.906 | 0.865 | 0.738 | 0.816 | 0.950 | 0.884 |
| `gaussian_global` | 0.943 | 0.920 | 0.849 | 0.894 | 0.912 | 0.930 |
| `gaussian_perpos` | 0.953 | 0.934 | 0.876 | 0.912 | 0.911 | 0.944 |
| `gmm_global` | 0.954 | 0.936 | 0.878 | 0.914 | 0.896 | 0.921 |
| `mixup` | 0.972 | 0.961 | **0.926** | 0.948 | 0.858 | 0.917 |
| `perturb_1.0` | 0.961 | 0.945 | 0.897 | 0.927 | 0.941 | 0.932 |
| `perturb_0.5` | 0.971 | 0.959 | 0.922 | 0.945 | 0.927 | 0.929 |
| `real` (ceiling) | 0.974 | 0.964 | **0.931** | 0.951 | 0.847 | 0.912 |

### 3-block program (`f = block1∘block2∘block3`) — identity full-cos 0.319

| condition | full_cos | upd_cos | **upd_r2** | full_r2 | gen→iso | gen→perpos |
|---|---|---|---|---|---|---|
| `gaussian_iso` | 0.793 | 0.785 | 0.616 | 0.629 | 0.899 | 0.826 |
| `gaussian_global` | 0.853 | 0.847 | 0.718 | 0.728 | 0.825 | 0.878 |
| `gaussian_perpos` | 0.884 | 0.880 | 0.776 | 0.784 | 0.824 | 0.907 |
| `gmm_global` | 0.879 | 0.875 | 0.767 | 0.775 | 0.804 | 0.866 |
| `mixup` | 0.931 | 0.929 | **0.866** | 0.871 | 0.755 | 0.858 |
| `perturb_1.0` | 0.895 | 0.892 | 0.798 | 0.805 | 0.869 | 0.875 |
| `perturb_0.5` | 0.913 | 0.911 | 0.831 | 0.837 | 0.832 | 0.864 |
| `real` (ceiling) | 0.937 | 0.935 | **0.877** | 0.881 | 0.739 | 0.849 |

`gen→iso` / `gen→perpos` = the same trained model's `update_cosine` evaluated on held-out **off-manifold** (isotropic / per-position Gaussian) inputs — a probe of how globally vs locally each model learned `f`.

## Key findings

**1. The program is largely recoverable off-manifold, and degrades gracefully with depth — no cliff.**
Trained on *pure isotropic noise* (activations the model never remotely visits), the forward model reaches update-R² **0.738** (1-block) / **0.616** (3-block) on real inputs. Matched per-position statistics (still *zero* real data) reach **0.876 / 0.776** — 94% / 88% of the `real` ceiling. The friend's worry ("if inputs are off-manifold, does it still predict on-manifold?") answers **yes, strongly**.

**2. Depth is the axis that matters.** The `perpos → real` gap **doubles** with depth (0.055 → 0.101 update-R²) and recovery fraction drops (94% → 88%). Composition amplifies the manifold's nonlinear structure — the part synthetic samplers miss — so off-manifold learning gets relatively harder the deeper the program. Predicts the effect keeps growing for full N-layer programs.

**3. The residual gap is *joint cross-position structure*, not the marginal density.**
- `gmm_global` (a strictly more expressive marginal — 64-component mixture) buys **~nothing**: +0.002 at 1-block, and **−0.009** at 3-block (slightly worse than a single per-position Gaussian). Better density modeling of *where activations sit* is not the missing ingredient.
- `mixup` closes **~90% of the gap at both depths** (0.050/0.055 and 0.090/0.101; ~99% of the absolute ceiling). The one thing `mixup` does that no Gaussian/GMM sampler does: it draws **whole real sequences** and interpolates them, preserving the **joint structure across positions**. Every parametric sampler here draws each position independently and destroys it.
- **Conclusion: what makes an input "on-manifold" for learning a transformer's layer-program is the relational/joint structure across the sequence, not the per-token marginal.**

**4. A real bias/coverage tradeoff — `real` is the most manifold-*specialized*.**
Reading the generalization columns: the `real`-trained model is the **best on-manifold but the worst off-manifold** (`gen→iso` = 0.847 at 1-block, **0.739** at 3-block — lowest in the table), and this specialization *widens* with depth. Noise-trained models are flatter and more global; `mixup` inherits `real`'s specialization (its `gen→iso` is also lowest-tier), confirming it recovers performance by *matching the manifold*. On-manifold training buys sharpness where the data lives at the cost of global fidelity — a quantified, in-network version of the OOD "computational function vs ID activation manifold" result (see [OOD_ROBUSTNESS_README.md](OOD_ROBUSTNESS_README.md)).

## Reproduction

```bash
cd experiments/

# Gaussian ladder + perturb + real (trains 6 forward models per gap)
modal run --detach a2a_forward/synthetic_input.py::synthetic_input_experiment \
  --predict-from post_block0 --predict-to post_block1 --fwd-n-layer 1 --n-steps 10000
modal run --detach a2a_forward/synthetic_input.py::synthetic_input_experiment \
  --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --n-steps 10000

# Manifold-matched rungs (GMM + mixup); reuses the eval pipeline, gate-verified
modal run --detach a2a_forward/synthetic_input_manifold.py::manifold_experiment \
  --predict-from post_block0 --predict-to post_block1 --fwd-n-layer 1 --n-steps 10000
modal run --detach a2a_forward/synthetic_input_manifold.py::manifold_experiment \
  --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --n-steps 10000
```

Results: `a2a_forward/synthetic_input/{gap}/fwd_L{n}/results.json` (ladder) and `results_manifold.json` (new rungs, with the identity gate).

## Open threads

- **Idea B (recursive residual):** train a second small model to predict the `gaussian_perpos`/`gmm` residual on real inputs — does boosting recover the ~0.10 gap that `mixup` recovers via manifold matching? Tests whether the joint structure is learnable as a correction rather than supplied via the input distribution.
- **Push depth:** full `post_block0 → post_block4` and the 77M model, to confirm the gap keeps widening with program depth.
- **Why does GMM independence hurt at depth?** Directly measure how much cross-position correlation the frozen block's attention induces, and whether a *sequence-level* generative model (vs per-position) is the minimal sampler that matches `mixup`.
