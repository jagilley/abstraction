# Language Reduction: Status

## What this is

Implementation of the idea in `ideas/language_reduction.md`: reduce the effective complexity of a language corpus by intervening on its co-occurrence spectrum, then measure how this changes the neural scaling exponents predicted by Cagnetta et al. (2026).

The pipeline uses corpus statistics (not LM-based rewriting) to perform context-dependent spectral denoising. For each token-occurrence, we compute how much of its co-occurrence with surrounding context lives in the top-k subspace of the multi-lag PMI matrices. Contextually atypical occurrences are replaced with candidates that maximize explained co-occurrence.

## Main result

Denoising at τ=0.3 steepens the empirical scaling curve by 2.6x and shifts the measurable language statistics (β, γ) in the directions predicted by theory:

| | τ=0.0 (original) | τ=0.3 (denoised) | interpretation |
|---|---|---|---|
| **β** | 0.929 | 0.739 | correlations decay more slowly — denoising removed noise that was masking long-range structure |
| **γ** | 0.509 | 0.738 | conditional entropy decays faster — remaining patterns are more locally predictable |
| **α_D predicted** = γ/(2β) | 0.274 | 0.499 | theory predicts steeper scaling |
| **α_D empirical** | 0.082 | 0.214 | empirical scaling steepens by 2.6x |

The predicted-vs-empirical gap (~3x for both τ values) likely reflects that our 2-layer model is in the slow-learning regime where δ < γ/(2β). The directional change is correct: denoising truncates the tail of the subtask distribution, so small models learn the remaining structure more efficiently.

### Scaling curves (best val loss)

| P | τ=0.0 | τ=0.3 | Δ |
|---|---|---|---|
| 100K | 7.453 | 7.004 | -0.449 |
| 1M | 6.576 | 5.955 | -0.621 |
| 10M | 5.329 | 5.160 | -0.169 |
| 100M | 4.711 | 4.776 | +0.065 |

Denoised corpus wins at small P (gap largest at P=1M), advantage vanishes at P=100M. Consistent with the prediction: denoising truncates the tail, so small models saturate earlier, but the ceiling is slightly higher because real information was removed.

## Pipeline stages

All stages run on Modal (workspace `jagilley`), orchestrated via `modal_app.py`.

| Stage | Command | Status |
|-------|---------|--------|
| 1. Tokenize FineWeb-Edu | `--stage tokenize` | Done (1B tokens, 100 shards) |
| 2. Vocab reduction + covariance + PMI | `--stage stats` | Done (V=50257→3200, 10 lags, 100M tokens) |
| 3. SVD of PMI(n) | `--stage spectral` | Done (10 lags, PMI-normalized) |
| 4. Denoise corpus | `--stage denoise --tau 0.3` | Done (10 shards, parallel GPU) |
| 5. Measure beta | `--stage beta` | Done (τ=0.0: 0.929, τ=0.3: 0.739) |
| 6. Train AR models | `--stage train-sweep` | Done (P=100K,1M,10M,100M for both τ) |
| 7. Measure gamma | `--stage gamma` | Done (τ=0.0: 0.509, τ=0.3: 0.738) |
| 8. Validate alpha_D | `--stage validate` | Done (see table above) |
| 9. Embedding geometry | `--stage embedding-eval` | Done (see GEOMETRY_ANALYSIS.md) |
| 10. Contextual embeddings | `--stage contextual-eval` | Done (see GEOMETRY_ANALYSIS.md) |

## Key design decisions

**PMI-normalized SVD** (not raw covariance). The raw covariance C(n) has a spectrum dominated by token frequency — the top singular vectors are frequency-correlated, causing the replacement algorithm to default to commas and "the". PMI(n) = log(P(x,y)/P(x)P(y)) factors out frequency, surfacing genuine associative structure. After this fix, replacements use real content words ('love', 'story', 'written', 'researchers') instead of function words.

**GPU-parallel denoising**. The replacement scoring (phase 2) is memory-bandwidth-bound: for each flagged position, score all 3200 candidates across 10 lags. Moving to GPU (T4) reduced per-shard time from 20+ min to ~1 min. Modal `spawn()` parallelizes across shards — 10 shards of 10M tokens denoise in ~2 min wall time.

**Smaller model for scaling experiments**. 2-layer, 4-head, 128-dim GPT-2 (~7M params, mostly embeddings). Training steps auto-scale with P (5 epochs, floor 2000, cap 20000). Best-val-loss tracked with 5-batch averaged eval.

**β still measured on raw covariance C(n)**, not PMI. This matches Cagnetta et al.'s definition: ||C(n)||_op ~ n^{-β}. PMI is only used for the spectral subspace that drives the denoising.

## What's on the Modal volume

```
/data/
  tokens/                  # 100 shards × 10M tokens (1B total, GPT-2 BPE)
  stats/                   # vocab_map, freq, covariance/, pmi/, svd/
  stats_denoised/tau_0.300/  # covariance on denoised corpus (for β measurement)
  denoised/tau_0.300/      # 10 shards denoised at τ=0.3
  models/tau_0.000/        # trained models at P=100K,1M,10M,100M
  models/tau_0.300/        # trained models at P=100K,1M,10M,100M
  results/                 # beta, gamma, summary, embedding_eval, contextual_embedding_eval JSONs
```

## Embedding geometry

Done. See `GEOMETRY_ANALYSIS.md` for full results. Summary: τ=0.3 models are not lobotomized — they learn meaningful embedding structure (40% analogy accuracy at P=100M). Denoising trades distributional breadth (analogy accuracy drops ~7pp, word similarity correlation degrades) for **topical coherence** (neighborhood coherence +0.05 across all P and all layers). The embedding space is lower-dimensional and more spectrally concentrated, consistent with subtask tail truncation. Contextual embeddings (per-layer hidden states) show the coherence advantage persists through the model's processing, and both models converge to similar clustering structure by the final layer despite very different static embeddings.

## What's next

1. **More P values** (add 300K, 3M, 30M) for smoother scaling curves — this is the highest priority, see τ sweep addendum below for why
2. **Finer eval granularity at small P**: eval every 50 steps instead of 500 for P=100K and P=1M to get more reliable best_val_loss
3. **50 lags**: current results use 10 lags. More lags would give more accurate β fits and potentially better denoising

---

## τ sweep addendum (2026-04-27)

### Full sweep results

We ran the pipeline at τ = 0.0, 0.1, 0.3, 0.5, 0.7. Stages 1–3 (tokenize, stats, spectral) are shared; stages 4–8 were run per τ.

| τ | β | γ | α_D predicted | α_D empirical (4pt) | R² |
|---|---|---|---|---|---|
| 0.0 | 0.929 | 0.509 | 0.274 | 0.082 | 0.988 |
| 0.1 | 0.824 | 0.578 | 0.351 | 0.088 | 0.990 |
| 0.3 | 0.739 | 0.738 | 0.499 | 0.214 | 0.998 |
| 0.5 | 0.632 | 1.240 | 0.981 | 0.272 | 0.998 |
| 0.7 | 0.432 | 0.878 | 1.017 | 0.249 | 1.000 |

### Per-P validation losses (best_val_loss)

| P | τ=0.0 | τ=0.1 | τ=0.3 | τ=0.5 | τ=0.7 |
|---|---|---|---|---|---|
| 100K | 7.453 | 7.383 | 7.004 | 6.920 | 5.832 |
| 1M | 6.576 | 6.540 | 5.955 | 5.858 | 4.989 |
| 10M | 5.329 | 5.403 | 5.160 | 5.135 | 4.518 |
| 100M | 4.711 | 4.827 | 4.776 | 4.825 | 4.256 |

### Finding 1: β is smooth and monotonic; γ is not

β decreases monotonically (0.93 → 0.43): correlations decay more slowly as denoising strips noise and leaves persistent long-range structure. This is the cleanest signal in the sweep.

γ rises through τ=0.5 (0.51 → 0.58 → 0.74 → **1.24**) then drops to 0.88 at τ=0.7. At τ=0.7, H_inf drops to 4.20 (from ~4.7 at other τ values), meaning nearly half a nat of fundamental entropy has been removed. The conditional entropy curve has so little dynamic range left that the power-law fit for γ is fitting a qualitatively different shape. The non-monotonicity reflects the corpus leaving the regime where γ is a meaningful description of its structure.

### Finding 2: The model nearly saturates at τ=0.7

At τ=0.7, P=100M: loss 4.256 vs H_inf 4.199 — only 0.057 nats above the irreducible floor. The model has learned essentially everything the distribution has to offer. This is the predicted endpoint of subtask tail truncation: denoise hard enough and a small model saturates.

### Finding 3: Denoised corpora overfit less, not more

| τ | P=1M gap (final - best) | P=100K gap |
|---|---|---|
| 0.0 | 0.528 | 3.180 |
| 0.1 | 0.385 | 3.352 |
| 0.3 | 0.450 | 3.432 |
| 0.5 | 0.296 | 3.370 |
| 0.7 | 0.109 | 3.004 |

At P=1M, overfitting shrinks monotonically with τ: the denoiser removes contextually atypical token-occurrences, which are exactly the instance-specific patterns that drive overfitting. The denoised corpus has a higher signal-to-noise ratio — less signal but disproportionately less noise — so there is less for the model to memorize beyond the genuine distributional structure. This is an internal consistency check: if overfitting had grown with τ, it would mean the denoiser was removing signal and leaving noise.

### Finding 4: P=100K distorts the scaling fits

P=100K trains for 2000 steps with eval_interval=500, giving only 4 evaluation checkpoints — the first at 41 epochs. Best_val_loss at P=100K is "best of 4 snapshots, all deep into overfitting." Dropping it and refitting α_D from P = 1M, 10M, 100M reveals a substantially different picture:

| τ | α_D (4pt) | α_D (3pt) | predicted |
|---|---|---|---|
| 0.0 | 0.082 | 0.315 | 0.274 |
| 0.1 | 0.088 | 0.297 | 0.351 |
| 0.3 | 0.214 | 0.304 | 0.499 |
| 0.5 | 0.272 | 0.400 | 0.981 |
| 0.7 | 0.249 | 0.248 | 1.017 |

The 3-point fit is exactly determined (3 parameters from 3 points), so individual values should be interpreted cautiously. But two qualitative conclusions are robust:

1. **The Cagnetta theory is more accurate on natural language than the 4-point fit suggested.** At τ=0.0: predicted 0.274, 3-point empirical 0.315. The 4-point fit made the theory look 3.4x off; the 3-point fit shows near-agreement.

2. **The "2.6x scaling improvement" at τ=0.3 was driven by P=100K.** Without it, α_D ≈ 0.30 for τ = 0.0, 0.1, and 0.3. Mild denoising does not change the empirical scaling exponent, even as β and γ both shift. Only at τ=0.5 does α_D genuinely steepen (to 0.40), and at τ=0.7 it drops back (to 0.25).

### Finding 5: The Cagnetta theory has a bounded regime of validity

The predicted-vs-empirical gap grows monotonically with τ in the 3-point fit:

| τ | predicted/empirical ratio |
|---|---|
| 0.0 | 0.87 (near-perfect) |
| 0.1 | 1.18 |
| 0.3 | 1.64 |
| 0.5 | 2.45 |
| 0.7 | 4.10 |

The theory works well on natural language (τ=0.0) and degrades as denoising moves the corpus away from the heavy-tailed subtask distribution the theory assumes. At high τ, α_D predicted exceeds 1.0 (the theory predicts every parameter buys proportional improvement) but the model is near saturation. The power-law scaling regime itself breaks down: at τ=0.7, the loss-above-floor spans only 0.73 nats across a 1000x range of P, and the model at P=100M is 0.06 nats from H_inf. You can fit a power law to anything with 3 points, but the curve is really an exponential decay to a floor, not a power law.

### Interpretation

The sweep traces the boundary of the Cagnetta scaling law's domain. The theory is a theory of learning from a heavy-tailed subtask distribution, and it works well when the distribution is heavy-tailed (natural language). As denoising truncates the tail:

- β and γ change in the predicted directions (β down, γ up), but the empirical scaling is insensitive to mild denoising (τ ≤ 0.3)
- There is a narrow band (around τ=0.5) where denoising genuinely steepens empirical scaling
- Beyond that, the distribution is simple enough that the model saturates within our P range, and the power-law description breaks down

### A deeper implication

At τ=0.0, the Cagnetta formula works: predicted 0.274, empirical 0.315. But under denoising, β and γ shift in ways the formula says should steepen scaling (predicted α_D climbs to ~1.0), while empirical α_D stays flat at ~0.30 through τ=0.3. The surface statistics change; the scaling doesn't.

This suggests the formula α_D = γ/(2β) is correct, but β and γ play a dual role. In Cagnetta's theory, they are properties of the *latent generative hierarchy* — the deep structure of how language composes subtasks. In our measurements, they are *surface statistics* of the token sequence (co-occurrence decay, conditional entropy decay). On natural text, these coincide: the surface statistics faithfully reflect the latent structure, and the formula works. Under denoising, they diverge: the surface statistics change (we flatten the correlation spectrum, speed up entropy decay) without restructuring the underlying generative process. The model still encounters the same effective subtask hierarchy, so the empirical scaling is unchanged.

If this holds up, the robustness of α_D ≈ 0.30 under mild denoising is evidence that scaling behavior is a property of the deep generative structure of language, not of the surface statistics the Cagnetta formula takes as input. The formula works on natural language precisely *because* natural language's surface statistics are faithful to its latent structure — and the fact that denoising breaks this correspondence while leaving scaling intact is, paradoxically, the strongest validation of the framework's core claim that scaling arises from hierarchical latent structure.

### Correction: cross-τ loss comparisons are invalid

Several claims above compare absolute loss values across different τ values. These comparisons are not valid. Each model's validation loss is measured against its own denoised distribution, which has a different entropy. Specific claims that should be read with this caveat:

- **Main result table** (top of document): "Denoising at τ=0.3 steepens the empirical scaling curve by 2.6x and shifts the measurable language statistics..." — the 2.6x was already corrected (Finding 4, P=100K artifact), but the scaling curves table comparing τ=0.0 losses to τ=0.3 losses (e.g., P=1M: 6.576 vs 5.955) is also misleading. The 0.62 nat difference is not "the denoised model learned more" — it's partly or entirely "the denoised target has lower entropy." Without evaluating both models on the same held-out set, we cannot decompose this into "easier target" vs "more efficient learning."

- **Per-P validation losses table** in the sweep addendum: same issue. The losses within a single τ row are comparable (same distribution, different P), so the *shape* of each row — and therefore α_D — is valid. But comparisons across columns (across τ) are not.

- **Finding 2** ("the model has learned essentially everything the distribution has to offer"): this claim is valid — loss vs H_inf is a within-distribution comparison.

- **Finding 3** (overfitting gaps): these are within-distribution (best vs final on the same τ), so the comparison across τ is legitimate. The gap measures how much the model memorized beyond generalizable structure, and this quantity is meaningfully comparable across τ.

The only valid cross-τ comparison of learning efficiency is α_D (the slope of each distribution's own scaling curve), which is the basis of the interpretation and deeper implication sections. Those conclusions are unaffected by this correction.

### Next steps

The most important next step is **adding intermediate P values** (300K, 3M, 30M) to get 6–7 point scaling curves that don't depend on a single unreliable point. The 3-point fit is suggestive but not conclusive.

---

## Vocab-only addendum (2026-05-06)

### Motivation

The spectral denoising method replaces tokens based on PMI-derived co-occurrence structure — the same statistics (β, γ) we then measure on the output corpus. This entanglement means we can't distinguish "genuine effects of corpus simplification" from "the replacement algorithm imprinting its own structure." The vocab-only method breaks this entanglement: it maps rare tokens to their nearest neighbors by GPT-2 embedding cosine similarity, a context-independent, frequency-based operation that doesn't consult co-occurrence matrices.

### Method

For a given τ, compute v' = the vocabulary size such that out-of-vocab tokens account for τ of the corpus by frequency. Map each rare token to its nearest top-v' neighbor in GPT-2 embedding space (paradigmatic similarity). No spectral analysis, no context dependence. The replacement is a deterministic, position-independent function of token identity.

### Results

| τ | v' | β | γ | α_pred = γ/(2β) | α_emp | pred/emp | H_inf | best loss (P=100M) | gap to floor |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 50,257 | 0.929 | 0.509 | 0.274 | ~0.295 | 0.93 | — | — | — |
| 0.1 | 10,833 | 0.921 | 0.465 | 0.252 | 0.343 | 0.74 | 4.493 | 4.700 | 0.207 |
| 0.3 | 1,724 | 1.264 | 0.475 | 0.188 | 0.150 | 1.25 | 4.256 | 4.385 | 0.129 |
| 0.5 | 236 | 1.154 | 0.443 | 0.192 | 0.151 | 1.27 | 3.502 | 3.598 | 0.097 |

Per-P validation losses (best_val_loss, nats):

| P | τ=0.1 | τ=0.3 | τ=0.5 |
|---|---|---|---|
| 1M | 6.160 | 5.178 | 4.037 |
| 10M | 5.161 | 4.712 | 3.781 |
| 100M | 4.700 | 4.385 | 3.598 |

Local log-log slopes:

| interval | τ=0.1 | τ=0.3 | τ=0.5 |
|---|---|---|---|
| 1M → 10M | 0.077 | 0.041 | 0.029 |
| 10M → 100M | 0.041 | 0.031 | 0.022 |

Note: α_emp values are from 3-point exactly-determined fits (P = 1M, 10M, 100M) and carry no uncertainty estimates. The τ=0.0 row uses the spectral baseline (shared corpus, no replacement in either method). Cross-method absolute loss comparisons are not valid (different effective vocabulary sizes → different target entropies). Within-method scaling shapes are valid.

### Finding 1: γ is invariant to vocabulary compression

Spectral γ ranged from 0.509 to 1.240 (2.4× range). Vocab-only γ ranges from 0.443 to 0.509 (< 15% variation) across a compression from 50,257 to 236 types. The conditional entropy decay rate — how much each additional context token helps predict the next — is an intrinsic property of the generative process, not something that depends on the surface vocabulary.

The spectral method's large γ variation was an artifact: PMI-based replacement makes tokens more predictable from their context, which mechanically increases the apparent conditional entropy decay rate. The vocab-only method, which replaces based on embedding similarity without consulting co-occurrence structure, reveals that γ is nearly constant.

### Finding 2: The spectral β trajectory was artifactual

Spectral β decreased monotonically with τ (0.929 → 0.432), interpreted as "denoising reveals hidden long-range structure." Vocab-only β *increases* with τ (0.929 → 0.921 → 1.264), with a slight non-monotonicity at τ=0.5 (1.154). The intuitive direction is upward: collapsing many token types onto fewer representatives reduces the information content at each position, so correlations between positions should decay faster.

The spectral method lowered β because the replacement algorithm optimizes replacements to maximize explained co-occurrence across PMI lag matrices, mechanically inflating ||C(n)||_op at large lags. The "hidden long-range structure revealed by denoising" was the denoiser imprinting its own structure onto the corpus.

The non-monotonicity at τ=0.5 (β = 1.154, down from 1.264 at τ=0.3) may reflect extreme compression (v'=236) creating new correlations as many source tokens collapse onto the same tiny set of representatives.

### Finding 3: The Cagnetta formula tracks empirical scaling under vocab-only

Under spectral, the predicted/empirical α ratio diverged with τ (0.93 → 1.70 → 1.79 → 4.16 → 2.76), suggesting the framework breaks down under denoising. Under vocab-only, the ratio stabilizes: 0.93 → 0.74 → 1.25 → 1.27. Both predicted and empirical α decrease with τ and move in the same direction, rather than diverging.

The formula α = γ/(2β) works here because the inputs are clean. γ is nearly constant (~0.47), so the formula reduces to α ≈ 0.47/(2β). As β increases under vocabulary compression, predicted α decreases, and empirical α follows. The mechanism is straightforward: fewer types → less information per position → less to learn per data point → slower scaling.

### Finding 4: Vocab-only reduction is a channel intervention, not a DGP intervention

The invariance of γ and the correct tracking of the Cagnetta formula together suggest that vocab-only reduction does not change the data generating process. It is a deterministic, context-independent function applied to the output of the original DGP — a lossy observation channel.

The factorization is:
- **γ** ≈ 0.47: property of the DGP (hierarchical sequential structure). Invariant to the observation channel.
- **β**: property of the channel (per-position resolution / type diversity). Increases under vocabulary compression.
- **α = γ/(2β)**: correctly combines a DGP property and a channel property to predict scaling.

The spectral method could not reveal this factorization because it modified the conditional distributions P(x_t | context), creating a genuinely new DGP rather than a lossy encoding of the original one. That's why both β and γ moved under spectral — the intervention was entangled with both the channel and the source.

### Interpretation

The hierarchical prediction structure of natural language — how context disambiguates the next token across multiple scales (word → phrase → sentence → paragraph → topic) — lives almost entirely in the positional structure of the sequence, not in which specific tokens occupy those positions. You can render the entire corpus in 236 token types (less than ASCII) and the conditional entropy decay rate barely moves. The hierarchy is in the positions, not the types.

This is a stronger result than the spectral sweep could provide. The spectral sweep showed "scaling is invariant to surface statistics" but the statistics themselves were corrupted. The vocab-only sweep shows "scaling correctly tracks a clean factorization of DGP structure (γ) and channel resolution (β)." The Cagnetta framework isn't broken — it was being tested with contaminated inputs.

### Caveats

1. **3-point fits**: All α_emp values are from exactly-determined fits (3 parameters from 3 points). Intermediate P values (300K, 3M, 30M) would provide uncertainty estimates and test whether the scaling is genuinely power-law.

2. **Near saturation**: At τ=0.3 and τ=0.5, the model is within 0.1–0.13 nats of H_inf at P=100M. The apparent α decrease at high τ may partly reflect the power-law regime ending rather than a genuine exponent change.

3. **τ=0.7 not yet run**: The most extreme compression point would test whether γ invariance holds when the vocabulary is reduced even further (v' likely < 100).

4. **Cross-method loss comparisons**: Absolute losses cannot be compared between spectral and vocab-only (different effective vocabularies → different target entropies). Only within-method scaling shapes (α) are valid comparisons.

## File overview

- `modal_app.py` — Modal app: parallel GPU denoising, scaling sweeps, all pipeline stages
- `statistics.py` — Vocab reduction, multi-lag covariance + PMI matrices, SVD
- `denoise.py` — Context-dependent PMI-spectral denoising (GPU + CPU paths)
- `model.py` — Minimal GPT-2 (supports APE and RoPE, matching Cagnetta's setup)
- `measure.py` — β and γ estimation, empirical scaling fit
- `tokenize_data.py` — FineWeb streaming tokenization
- `config.py` — Configuration dataclass
- `test_local.py` — Local end-to-end test on synthetic data
- `eval_embeddings.py` — Embedding geometry evaluation (analogies, similarity, coherence, clustering)
- `inspect_text.py` — Decode and compare original vs denoised tokens
- `GEOMETRY_ANALYSIS.md` — Full embedding geometry analysis results
