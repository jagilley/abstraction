# Compositional Learnability in the Random Hierarchy Model

*Domain: controlled-DGP experiments (RHM), compositional learning under self-supervision*
*Last updated: 2026-06-29*


## In the Random Hierarchy Model, deep compositional structure is recoverable and representable, but how deeply plain next-token prediction learns it is gated by the data-generating process, not by model capacity
*Confidence: strong*

- The RHM is a tree-structured DGP with known ground-truth latents, so per-level recovery, an optimal-inference ceiling (belief propagation), and privileged supervision are all directly computable — the transparency that makes every claim below testable — [deep-composition arc](../../experiments/rhm/RHM_DEEP_COMPOSITION_README.md)
- A standard 8L/256D causal transformer, probed for the true latent ancestor at each hierarchy level, lands far below the recoverable ceiling at high levels under plain NTP — that gap is the phenomenon these beliefs explain — [occupancy-frontier sweep](../../experiments/rhm/rhm_occupancy_frontier.py)


### The deep structure is recoverable in principle and representable by a standard transformer; at m≥4 the binding constraint on learning it is the training signal, not capacity or architecture
*Confidence: strong*

- Exact tree belief propagation with known rules recovers high-level latents well (especially at low occupancy) — the information is present in the leaves, so the structure is recoverable — [BP ceiling](../../experiments/rhm/rhm_local_signal.py)
- Supplying the latent signal as an auxiliary loss lets the *same* architecture reach the BP ceiling at every level including the root (oracle-aux, Exp 4d) — capacity/architecture suffice; plain NTP just never supplies the gradient — [deep-composition arc](../../experiments/rhm/RHM_DEEP_COMPOSITION_README.md)
- No local self-supervised objective tried — NTP, MLM, masked-span, or contrastive/DINO/Sinkhorn invariance — reaches deep composition at m=4; the high-level signal is intrinsically diluted in the data (Exp 5) — same source
- Scope note: this "not self-supervised-learnable at depth" conclusion was established entirely at m=4; it does **not** hold at m=2, where plain NTP reaches the root given adequate model depth (see sibling belief).


### Recoverability and learnability are governed by separate DGP knobs: occupancy m/v^(s-1) sets the information ceiling; synonymic multiplicity m sets the learnable frontier depth under NTP
*Confidence: strong*

- m sets the depth the bottom-up learning wave reaches under NTP: m=2 reaches the root (d6 = 0.93 = BP exactly, all lower levels 1.00), m=4 stalls ~d4, m=8 stalls ~d3 (8L/8H/256D, L=6, s=2, 40k steps, distinct rules) — [occupancy-frontier sweep](../../experiments/rhm/rhm_occupancy_frontier.py)
- At fixed m=4, lowering occupancy 0.50→0.0625 (v8→v64) raised the d4 frontier 0.43→0.93 and the BP root ceiling 0.40→0.96, but left d5/d6 near chance despite BP(d5/d6)≈1.0 and perfectly clean d1–d3 — occupancy improves quality at the frontier and raises the ceiling, but does not advance frontier *depth* — same source
- Collapse-check kill shot: v16m2 and v32m4 share occupancy 0.125 and BP root ceiling ~0.92, yet learned root recovery is 0.93 vs 0.07 — identical recoverability, opposite learnability, so m (not occupancy) is the learnability control — same source
- Refines the earlier "occupancy law" reading that attributed the scaling result "m dominates ~3:1" purely to occupancy/recoverability: m carries a learnability effect over and above its recoverability effect — [scaling sweep](../../experiments/rhm/README.md)
- Caveats: single rule seed for the m=2-solves headline (1–2 more seeds wanted); the m=4 wall is confirmed asymptotic only at occ 0.25 (100k → d5 plateaus ~0.28), so a 100k low-occupancy run is pending to confirm d5 never starts there.

#### Constructive implications: curriculum over m (not occupancy) is the lever, and m=2 is a ready single-task deep-hierarchy substrate
*Confidence: speculative*

- Lowering occupancy at fixed m does not deepen the frontier, so curriculum-over-occupancy (varying v) is ruled out as a route to deeper composition — [occupancy-frontier sweep](../../experiments/rhm/rhm_occupancy_frontier.py)
- Curriculum over m — train where NTP reaches the root (m=2), transfer toward the regime where it cannot (m=4) — is the live, untested constructive route to deep composition at hard m — same source
- v16m2 is a substrate where plain NTP fully encodes the hierarchy to the ceiling, making it the intended testbed for whether a forward self-model's residual becomes hierarchy-legible (the A2A amplifier-vs-source question) — same source
