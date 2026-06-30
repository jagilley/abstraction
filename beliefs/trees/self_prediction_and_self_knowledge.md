# Self-Prediction and Self-Knowledge

*Domain: representation learning, self-modeling, mechanistic interpretability*
*Last updated: 2026-06-29*


## Co-training a system with predictions of its own future computation produces computational self-knowledge
*Confidence: strong*

- Closed-loop models (receiving FM predictions of post_block3 from post_block0) encode the FM's residual vector dramatically better than open-loop models at every layer (R²=0.42 vs 0.26 at post_block3, R²=0.21 vs 0.02 at post_block0), replicated in controlled retrain with identical lr/seed/init — [controlled retrain](../../experiments/a2a_forward/CONTROLLED_RETRAIN_README.md)
- The self-knowledge is distributed across all layers via backpropagation, including layers that never directly see the injection — early-layer R² gap (0.21 vs 0.02) is the cleanest evidence — same source
- Calibration transfer experiment showed zero epistemic self-knowledge: competence probes (activations → own per-token loss) trained ID and evaluated OOD are identical across all conditions, and output entropy beats every activation probe — [OOD robustness](../../experiments/a2a_forward/OOD_ROBUSTNESS_README.md)
- Cross-domain replication: MNIST ViT shows 3× stronger self-knowledge probes (Δ R²=+0.52 at post_block0) with low-rank, digit-discriminative residual structure — [MNIST experiment](../../experiments/a2a_forward/MNIST_README.md)

### Self-knowledge is encoded directionally (what kind of computation was surprising) rather than as scalar magnitude (how surprising)
*Confidence: strong*

- Vector probe Δ R² (+0.18) is 6× scalar residual-norm probe Δ R² (+0.03) in language — [controlled retrain](../../experiments/a2a_forward/CONTROLLED_RETRAIN_README.md)
- Causal steering along the residual-norm direction has zero novelty-specific effect on injection reliance — [causal probes](../../experiments/a2a_forward/CAUSAL_PROBES_README.md)
- Injection help is organized by residual direction clusters (η²=0.0017), not magnitude octiles (η²=0.0003) — 5× more explanatory power — same source
- Directional causal steering confirms partial functional use: focused-attention error cluster has selectivity 2.41, the strongest direction-specific causal effect — [directional steering](../../experiments/a2a_forward/DIRECTIONAL_STEER_README.md)


## The residual of a capacity-limited self-model reflects the computational complexity of the modeled system
*Confidence: moderate*

- Residual effective rank tracks DGP complexity across four settings: grokking (mod addition) ~15/128, MNIST ~18/128, language 29M ~200/256, language 77M ~235/512 — [MNIST](../../experiments/a2a_forward/MNIST_README.md), [model scale](../../experiments/a2a_forward/MODEL_SCALE_README.md)
- MNIST residual is low-rank and digit-discriminative (top-5 PCs all discriminate digit identity, η²=0.14–0.23); language residual is full-rank and diffuse (top-1 PC explains only 2.4%) — the structure mirrors task structure — same sources
- The capacity bottleneck must be in the forward model's *capacity*, not its *structural ability to see the input*: a per-position MLP residual captures "attention exists" (structural blindness), while a transformer FM residual captures "computation too complex for this capacity" (genuine novelty) — [open-loop analysis](../../experiments/a2a_forward/OPEN_LOOP_ANALYSIS_README.md)
- Scaling sweep confirms: at 10% capacity the FM captures 99.9% (cosine 0.999) and computational-novelty behavioral effects shrink (delimiter d drops from +0.78 to +0.30), but the residual remains inherently high-rank in language (eff rank >235/256 at all capacity points) — [scaling sweep](../../experiments/a2a_forward/SCALING_SWEEP_README.md)
