# The Locality of Manifold Structure

*April 2026*

## The empirical picture

GLP residuals — the difference between what a language model actually produces and what the GLP considers "typical" — are semantically meaningful. When you negate a residual, you get something recognizable as the semantic opposite. The geometry of residuals correlates with the geometry of their behavioral effects (CKA ~0.5). Individual residuals capture what is *novel* about a particular activation in a particular context.

But residuals do not transfer. Applying prompt A's residual to prompt B produces effects that depend on the target, not the source. The result is a clean null: 9.5% probe accuracy on 10-way classification vs. 10% chance. The manifold is curved. Parallel transport changes meaning.

You *can* extract transferable structure by averaging: group prompts into categories, compute centroid residuals, take SVD. This yields directions that transfer at 33% on 5-way (p < 10^-7), with two directions reaching 50%. But this is coarse — the two directions that work correspond roughly to "factual vs. creative prompt type." Finer-grained clustering (k=11 semantic clusters) performs comparably to human-defined categories, not better.

## The interpretation

The GLP manifold has rich local geometry and coarse global geometry. Each point on the manifold encodes meaningful, interpretable deviations from typicality — but the encoding is context-dependent. The same concept (e.g., emotional intensity) is realized through different geometry in different regions. This is consistent with the Fractured Entangled Representations (FER) picture: same semantics, different circuitry.

Global structure exists but is categorical, not fine-grained. When you average enough residuals within a group, you recover what *type* of prompt produced them. Whether finer transferable structure is hidden by Llama 1B's behavioral limitations (repetitive loops dominate at steering strength alpha=2) or whether the manifold is genuinely only globally structured at a coarse level — this remains open.

## The deeper point

It may be misguided to try hard to extract globally transferable structure at high resolution. Both humans and neural networks seem to naturally localize knowledge. A doctor's understanding of "inflammation" and a mechanic's understanding of "overheating" may share abstract structure, but the operational knowledge — the part that does work — is embedded in context-specific representations. Whether this is a feature or a bug is not obvious.

If it's a feature, then the right unit of analysis for the GLP isn't "find universal directions" but rather "characterize the local geometry at each point and understand why it takes the shape it does." The residual is a local object. Trying to make it global may be forcing a frame that the geometry doesn't support.

If it's a bug — or at least a limitation of scale — then larger models with less fractured representations might show cleaner global structure. The path to testing this is straightforward: repeat the transfer experiments with Llama 8B.

Either way, the verbalization pipeline (extract residual, steer, describe behavioral effect in language) remains the right decontextualization strategy. Language is the natural coordinate system for comparing across contexts, precisely because it abstracts over the context-dependent geometry.

## Summary of evidence

| Experiment | Result | Implication |
|---|---|---|
| Residual semantics (negation test) | 36-78% accuracy | Residuals are locally meaningful |
| Residual semantics (CKA) | 0.45-0.55, p < 0.001 | Residual geometry tracks behavioral effect geometry |
| Residual transfer | 9.5% vs 10% chance | Individual directions do not transfer across contexts |
| PC verbalization (naive PCA, diverse) | 21% on 5-way (chance) | Within-category variance is not transferable |
| PC verbalization (centroid SVD) | 33.1% on 5-way (p < 10^-7) | Between-category structure transfers, but coarsely |
| Semantic cluster transfer | 12.3% on 11-way (p = 0.004) | Data-driven clusters comparable to human categories |
| Nonlinear epistemic probe (subset test) | Velocity uplift shrinks from +0.10 (full) to +0.03 (category-controlled subset) | Coarse between-category structure > fine within-category structure, even through nonlinear readout |
