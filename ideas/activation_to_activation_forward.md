# Activation-to-Activation Forward Modeling

**Status**: Core mechanism validated (grokking self-regulation, LLM co-training, Llama scale-up); self-knowledge demonstrated; internalization untested
**Date**: 2026-05-24 (updated 2026-06-02)
**Supersedes**: [conditional_novelty_bottleneck.md](conditional_novelty_bottleneck.md)
**Builds on**: GLP residual semantics, VPD (Goodfire/Sharkey et al., 2026), CLS theory, CNB grokking experiments, cerebellar neuroscience
**Validated component**: Activation-matching self-regulation prevents Sisyphean collapse (see fer/experiments/zipfian_grokking/cnb_self_regulation/)

## One-liner

A small auxiliary model learns to predict a larger model's activations from its current state; feeding predictions back into the main model causes all layers to develop representations encoding the auxiliary model's error structure — a form of learned self-knowledge, distributed via backprop.

## The problem, progressively stated

### The real limitation: unconditional manifold modeling

The original CNB doc diagnosed the GLP's problem as a lack of weight-space grounding — activation residuals tell you "this is far from typical" but not "this deviates from a specific computational mechanism." That diagnosis was partially right about the symptom but wrong about the cure.

The GLP models P(activations) — the unconditional distribution of activations across inputs. This has three problems:

1. **It's enormous.** The Llama 1B GLP has ~3B parameters. This is suspicious — the hippocampus is tiny relative to the cortex. A novelty detector should be smaller than the thing it monitors.

2. **It doesn't distinguish expected from unexpected variation.** It knows what's statistically typical, but not what the model *should* compute for *this specific input*. An unusual-but-correct activation and a wrong-but-plausible activation look the same to the manifold.

The right question isn't "what do activations typically look like?" It's "what should this model compute given this input?" That's a forward prediction problem, not a density estimation problem.

### Why forward prediction is intrinsically cheaper

By the chain rule, H(activations | input, current_state) ≤ H(activations). Conditioning can only reduce entropy. Since a transformer's activations are a deterministic function of its inputs, the conditional entropy is in principle zero — the forward model's job is function approximation of a known target, not density estimation over an unknown manifold. This explains the size asymmetry: a forward model can be much smaller than an unconditional model of the same activation space, because the input does most of the work.

The cerebellum is biologically consistent: architecturally much simpler than the cortex it models, using a relatively homogeneous supervised-learning circuit to approximate a vastly more complex recurrent system. It works because conditioning on cortical state eliminates most of the variance.

### The grounding problem, resolved

The original CNB doc argued that novelty must be grounded in weight-space — that you need to decompose the model's weights (via VPD) to anchor predictions in computational structure rather than activation statistics. The grokking experiments and the cerebellar analogy both point to a different resolution.

**The experiment**: SVD-based activation prediction prevented Sisyphean collapse completely, but the weight decomposition wasn't load-bearing. Rank 128 (full rank, no truncation at all) worked nearly as well as rank 64. The bottleneck was the *checkpoint* — a frozen snapshot of the model at its peak — not the rank truncation. Even without any information bottleneck, any faithful representation of the peak model's computation provided a clean reference for self-regulation.

**The biology**: The cerebellum doesn't receive weight decompositions. It receives activations from the cortex via the pontine nuclei and learns to predict cortical dynamics from observation. Its own synaptic weights encode what it's learned about how the cortex behaves. No explicit weight inspection is needed.

**The resolution**: Grounding comes from *feeding predictions back into the main model*. The cortex processes cerebellar predictions through its own connection weights — the same weights it uses to process any other input. The main model's forward pass *is* the grounding function. Over co-training, the main model develops representations that are informed by the forward model's predictions, which are themselves informed by the main model's dynamics. The grounding is the co-adaptation, not the weight decomposition.

## The biological parallel: cerebellar forward models

### The cerebellum as a meta-model of cortical dynamics

The cerebellum contains roughly half the neurons in the brain, uses a remarkably uniform supervised-learning architecture (granule cell expansion → Purkinje cell readout → climbing fiber error signal), and has massive bidirectional connectivity with the cortex via the thalamus. Its classical role is motor prediction — predict the sensory consequences of motor commands, flag discrepancies, issue corrections — but this extends to cognition. The lateral cerebellar hemispheres (massively expanded in humans, disproportionately so relative to brain size) connect primarily to prefrontal and temporal association cortex, not to motor cortex.

The computational principle: given the current cortical state (received via efference copies through pontine relays), predict the next cortical state. When the prediction fails, the error signal (climbing fiber → Purkinje cell LTD) updates the cerebellar model and the corrective output (deep nuclei → thalamus → cortex) adjusts cortical processing.

Key properties:
- **Conditional prediction**: conditions on current cortical activations, not on cortical weights
- **Supervised learning**: explicit error signal (climbing fiber complex spikes) rather than unsupervised density estimation
- **Sparse error signals**: climbing fibers fire at 1-4 Hz vs. 50-100+ Hz tonic Purkinje rate — each error is rare and highly informative
- **Thalamic transformation**: the error signal passes through the thalamus, which applies temporal derivative filtering (amplifying transitions, suppressing steady-state) and is gated by cortical feedback — the cortex partially controls which cerebellar signals it listens to

### Representational homogeneity enables native ingestion

The neocortex uses remarkably uniform architecture everywhere — six layers, similar columnar microcircuitry, similar population coding. This means cortical areas speak the same representational language. A signal arriving from the cerebellum via the thalamus is processed by the same cortical machinery that handles sensory input, hippocampal replay, or internally generated representations. No format conversion is needed.

The cortex doesn't need the cerebellar signal to be pre-formatted in "cortical language." It just needs it to arrive at the right laminar address (cerebellar returns target middle layers, the same as feedforward sensory input), and cortical learning handles the rest.

For transformers: the residual stream has uniform dimensionality throughout. A side-channel prediction can enter the stream directly — concatenated, added, or cross-attended — and the transformer processes it using the same weights it applies to any other input. The representational homogeneity condition is satisfied by construction.

### The self-map: co-training produces learned self-knowledge

This is the key emergent property. Because the cerebellum constantly sends predictions about cortical states back to the cortex, and the cortex processes these through its own weights, over time the cortex develops representations that are *about its own computational states*. The cerebellar return signal is statistically informative about cortical dynamics, and the cortex is a good enough learner to extract that structure.

**What the experiments show**: Co-training a main model with an auxiliary predictor causes all layers — including layers before the injection point — to develop representations encoding the auxiliary model's error structure. The self-knowledge is directional (the model encodes *what kind* of computation was missed, not *how much*) and is partially causally used (the model discriminates the error type where the injection is most useful). This is a plausible mechanism for how a system could develop forward-pass-embedded knowledge of its own computational structure — "self-interpretability" — without requiring explicit introspection.

The representational divergence analysis (REPRESENTATIONAL_DIVERGENCE_README.md) revealed that the uniform Δ R² ~+0.18 across layers masks qualitatively different reorganizations at early vs late layers:

- **Early layers (post_block0)**: Large, concentrated change (75% of activation norm, effective rank 164.6). But this dominant change is *orthogonal* to self-knowledge (extra-SK overlap 0.49x, *below* random). The layer reorganized extensively via backprop for general-purpose input formatting. Self-knowledge lives in quieter, orthogonal directions — a distinct signal, not a byproduct of the dominant change.
- **Late layers (post_block2, post_block3)**: More diffuse change, but strongly *aligned* with self-knowledge (2.42x ratio at post_block3). The dominant representational change at late layers IS the self-knowledge — these layers reorganized primarily along directions encoding what the auxiliary model misses.

This functional hierarchy — input formatting at early layers, self-representation at late layers — is a natural prediction of the self-knowledge framework and a less obvious prediction of pure gradient-mediated co-adaptation. It suggests the self-knowledge at late layers is not incidental to some other reorganization; it is the primary axis of change.

**What remains hypothetical**: Whether this representation is *manipulable* — whether the model can flexibly compose or query its self-knowledge for downstream reasoning, as opposed to using it only for credit assignment over the injection signal. The biological analogy suggests this should be possible (fast metacognitive judgments, approximate self-knowledge, flexible recombination), but the current experiments test only the credit-assignment form of self-knowledge use.

For AI systems, co-training with a self-predictor gives the main model representations encoding its own computational structure as a learnable, forward-pass-accessible signal. Whether this scales from "one useful discrimination about focused-attention positions" to a rich, general-purpose self-model is an empirical question — but the mechanism that would produce it (backprop-mediated reorganization in response to processing self-predictions) is demonstrated, and the representational divergence results show that late-layer reorganization is dominated by self-knowledge rather than being a side-effect of other changes.

### The abstraction ratchet

Initially, a cognitive operation requires full cortical elaboration — slow, step-by-step. The cerebellum learns to predict the trajectory. Once accurate, the cerebellar prediction arrives before full elaboration completes, pre-activating the end state. The cortex can accept the predicted conclusion without fully recomputing it.

The compressed cerebellar output — capturing the input-output mapping but not the intermediate dynamics — gets written into cortical association areas as a new stable pattern. What was a multi-step derivation becomes a single retrievable primitive. This primitive can then serve as input to higher-order operations, which the cerebellum in turn models, compresses, and feeds back. Each cycle produces a layer of abstraction without anyone designing the hierarchy.

This maps onto the expertise literature (chess masters recognize chunks of chunks) and explains why experts often can't decompose their knowledge back into primitives — the intermediate steps are genuinely no longer explicitly represented.

### Good novelty versus bad novelty

Not all prediction errors are worth learning from. The brain distinguishes them through multiple mechanisms:

**Precision weighting** (predictive coding framework): every prediction comes with estimated confidence. Errors are weighted by precision before driving updates. High-precision errors (violations of confident predictions) are amplified. Low-precision errors (deviations within expected noise) are attenuated. Engineering analog: normalize residuals by estimated variance.

**Structural analysis**: expected uncertainty (known unknowns — acetylcholine) vs. unexpected uncertainty (the model itself is wrong — norepinephrine). The signature of unexpected uncertainty is that errors are *structured* — correlated across time or features in ways noise wouldn't be. Engineering analog: low-rank, temporally persistent errors are informative; diffuse, transient errors are noise.

**Learned valence tagging** (amygdala, dopamine): fast classification of error patterns as informative/rewarding vs. meaningless/aversive, based on lifetime experience with similar errors. Engineering analog: a small auxiliary classifier predicting whether a given error will prove *reducible* over subsequent training. This is recursive — training a system to predict which of its own prediction errors are worth learning from — but it's exactly what the brain appears to do.

**Status relative to experiments**: The LLM experiments (Runs 2, 5) showed the residual is full-rank and diffuse at every capacity point (effective rank >235/256), with no low-rank structure corresponding to identifiable mechanisms. The prescription "low-rank residuals are more informative than diffuse residuals" does not apply in the language setting — the residual is inherently high-rank because language computation is distributed across all dimensions. The residual does have meaningful structure when conditioned on *behavioral* categories (delimiter tracking, focused attention, sentence boundaries), but this structure lives in the residual's direction, not its rank. The precision weighting and learned valence tagging ideas remain untested. Whether richer residual structure emerges at wider layer gaps or larger scale is an open question.

## The experimental proof of concept: Zipfian grokking

### What we tested

A 2-layer MLP trained on (a+b) mod 97 discovers a Fourier solution (grokking), then collapses under Zipfian loss weighting (Sisyphean collapse) as memorization of high-weight samples overwrites the general solution. We froze an SVD-based activation reference from the model's peak (epoch 25k, 99.4% accuracy) and added an MSE penalty encouraging h1 activations to match the reference during continued training.

### What worked

The activation-matching penalty completely prevented collapse. Both β=0.1 and β=1.0 maintained >99% test accuracy for 25,000 additional epochs, while the unregulated baseline plunged to 56.2%. The model didn't just maintain performance — it improved to 99.7%, spending training time on Fourier refinement rather than fighting memorization pressure. Fourier energy was preserved (0.955→0.956) vs. degraded in the baseline (0.951→0.702). The mechanism was insensitive to β — the bottleneck itself provides the right scale of regularization.

The residual converged *toward* the reference (1.59→0.16), meaning the model became more aligned with the SVD approximation over time. The mechanism is self-reinforcing: under regulation, the Fourier solution only gets stronger, so a rolling reference would create a virtuous cycle.

### What wasn't load-bearing

**Weight decomposition**: Rank 128 (full rank, no truncation) prevented collapse nearly as well as rank 64 (99.1% vs 99.7%). The decomposition coarseness wasn't filtering memorization — it was providing slack for continued learning. The bottleneck was the checkpoint, not the rank.

**VPD**: Input-dependent binary routing was the wrong bottleneck for the Fourier solution's structure (universal low-rank, not input-dependent sparse). VPD's CI values drifted downward over training, destroying early bimodal structure. The failure illustrates that explicit decomposition algorithms bake in assumptions about sparsity structure.

**The router, the information bottleneck, the conditional architecture**: The finite input space (9,409 pairs) allowed precomputation of the reference as a lookup table. No router, no compression, no conditional prediction — just cached targets. Every component the original CNB design argued was essential was bypassed.

### What this validates and what it doesn't

**Validated**: The activation-matching signal works for self-regulation. A frozen reference from a clean checkpoint anchors the model to its discovered structure. The mechanism requires zero domain knowledge — no labels, no memorization direction, no Fourier analysis.

**Not validated**: The router (needs infinite input space), the information bottleneck (needs a contaminated reference or a model where memorization is low-rank enough to survive truncation), co-training dynamics (reference was frozen, not learned), the self-map property (no loopback into the main model).

The grokking experiment proves the signal works. The LLM setting is where the architecture gets tested.

## The proposal: cerebellar forward models for neural networks

### Core architecture

1. **The main model** ("cortex"): a standard or looped transformer that processes inputs and produces outputs.

2. **The forward model** ("cerebellum"): a small auxiliary network that takes cheap features (token embeddings, early-layer activations, or current recurrence state) and predicts activations at later/deeper layers of the main model.

3. **The feedback loop**: the forward model's predictions are fed back into the main model as native inputs — added to the residual stream, concatenated, or cross-attended. The main model processes them through its own weights, the same way it handles any other input.

4. **The residual**: actual activation minus predicted activation. This is the novelty signal — what the main model computes that the forward model didn't anticipate.

### Training

Co-train both models. The forward model's loss is prediction error on the main model's activations. The main model's loss is the task loss, optionally plus an activation-matching penalty (for self-regulation) or the novelty signal fed back as input (for self-knowledge).

Over co-training:
- The forward model learns the main model's computational dynamics
- The main model learns to incorporate forward-model predictions as useful inputs
- The main model develops representations of its own computational states (the self-map)

### Why the bottleneck lives in the forward model's capacity, not in an explicit decomposition

The forward model must be small — otherwise it's just distilling the main model, which is redundant and expensive. This capacity constraint forces the forward model to approximate the main model's computation under a parameter budget.

In grokking, where the main model's computation is low-rank (Fourier), the forward model's approximation naturally captures that structure. In language, the scaling sweep (Run 5) showed the opposite: the forward model is "slightly worse everywhere" — uniform approximation, not mechanism discovery. The residual is full-rank and diffuse (effective rank >235/256) at every capacity point tested. The capacity constraint doesn't force discovery of discrete mechanisms; it forces uniform compression. Whether this changes at larger scale (wider layer gaps, more complex computation to approximate) is an open question.

The important point is that the forward model doesn't need to discover mechanisms for the co-training loop to produce useful self-knowledge. What matters is that the forward model's error structure varies systematically by input type (it's better at focused-attention positions, worse at delimiter tracking), and this variation is what the main model learns to represent.

### The looped transformer as natural home

A looped transformer applies the same block of layers repeatedly, refining representations through recurrence. This provides a native slot for the forward model's predictions: inject the prediction at each recurrence step. The transformer can:
- **Accept the prediction** (fast convergence, few loops) when the forward model is well-calibrated — the prediction pre-activates the right attractor basin
- **Override the prediction** (more loops) when the forward model is wrong — continued recurrence refines past the prediction

Adaptive compute falls out naturally: the delta between successive loop iterations measures how much the prediction helped. Easy inputs (well-predicted) converge in 1-2 loops. Novel inputs (poorly predicted) need many. No separate difficulty estimator required.

### Explicit decomposition as initialization, not architecture

The original CNB proposal's VPD-based design isn't wrong — it's a valid initialization strategy for the forward model. Pre-decompose the main model's weights, use the components as a starting basis for the forward model, then let co-training refine from there. This bootstraps the forward model with structural knowledge rather than forcing it to discover everything from scratch.

But the decomposition shouldn't be *architectural* — the forward model should be free to adapt its implicit decomposition as the main model's weights change. Locking the forward model to a fixed external decomposition creates staleness (the decomposition lags the model) and fragility (the wrong decomposition algorithm breaks the system, as VPD did for grokking).

### The frozen reference for self-regulation

For the specific task of preventing collapse (as validated in grokking), a frozen forward model from a "peak" state provides the regulatory anchor. Any drift from the peak's computational structure registers as increased residual, regardless of whether the drift is toward memorization or toward some alternative solution. The frozen reference is the worst case — a rolling reference that updates online would be self-reinforcing, since the capacity bottleneck prevents memorization artifacts from entering the forward model's predictions.

For ongoing novelty detection (the richer use case), the forward model updates online and the residual itself is the novelty signal rather than a loss term.

## What this gives you

Progressing from validated to speculative:

1. **Self-regulation** (validated in grokking): prevent collapse under distributional pressure by penalizing deviation from a frozen reference. No domain knowledge required. The mechanism is insensitive to hyperparameters.

2. **LM improvement from self-prediction** (validated in LLM co-training): the injection consistently reduces LM loss (-0.05 to -0.48 nats over training), and the benefit grows monotonically even as the forward model's prediction accuracy degrades. The model uses the prediction as a structured reference frame, not a literal preview. The forward model's approximation error varies systematically by input type (focused attention vs delimiter tracking vs distributed computation), and this position-dependent reliability is what the main model learns to exploit.

3. **Learned self-knowledge** (validated, directional form): co-training causes all layers — including layers before the injection point — to develop representations encoding the forward model's error structure, distributed via backprop (R²=0.21 vs 0.02 at post_block0 in controlled retrain). The self-knowledge is directional (vector probe gap 6× scalar gap) and partially causally used (focused-attention cluster selectivity = 2.41 in directional steering). Whether this representation is *manipulable* — usable for flexible reasoning about what the model knows — is untested.

4. **Adaptive compute** (architecturally plausible in looped transformers): prediction quality gates the number of recurrence loops. Well-predicted inputs converge fast; poorly-predicted inputs need more computation.

5. **Progressive abstraction** (speculative): the compression ratchet from forward-model predictions enables hierarchical concept formation. Multi-step computations get compressed into single primitives that serve as inputs to higher-order operations. Requires wake-sleep consolidation (internalization of offloaded computation) which is untested.

## Key design questions

1. **What does the forward model predict?** Full activation vectors are high-dimensional. It might work better to predict a lower-dimensional target — the top-k principal components of activations, or which subspace the solution lives in — and let the main model fill in the details.

2. **What serves as input to the forward model?** Token embeddings? Shallow activations? Current recurrence state? The forward model needs cheap features that are informative about what the main model will compute. If it needs the full forward pass to make predictions, novelty detection isn't cheaper than just running the model.

3. **Online learning dynamics**: the forward model must learn fast (track model changes) but not forget (maintain stable reference). The cerebellum solves this with multiple plasticity sites at different timescales — fast Purkinje cell LTD for moment-to-moment correction, slower deep nuclear plasticity for consolidation. An engineering analog might be fast-and-slow learning rates, or a mixture of online and periodic batch updates.

4. **The lag problem**: after the main model learns something new, the forward model's predictions are temporarily wrong — spurious novelty signals. This may be phenomenologically recognizable as the feeling of familiar things seeming strange after an insight. Feature or bug? Probably both: the lag is informative (the model changed in a way not yet captured by the forward model) but could be noisy if updates are frequent.

5. **Residual structure engineering**: The residual is full-rank and diffuse in language (effective rank >235/256 at all capacity points). The original expectation of low-rank residuals corresponding to discrete missed mechanisms did not materialize. However, the residual has meaningful structure when conditioned on behavioral categories (delimiter tracking, focused attention), and the closed-loop model's directional self-knowledge captures this. Open questions:
   - **Precision weighting**: normalize residuals by estimated variance, either learned or running-average. Untested.
   - **Learned error classification**: a small auxiliary network predicting whether a given error will prove reducible over subsequent training. Untested.
   - Whether richer residual structure (low-rank, mechanism-specific) emerges at wider layer gaps or larger scale.

6. **How does the delta-residual interact with the prediction residual?** At coarse forward-model capacity, some of the main model's computation is inherently unpredictable — it falls outside what the forward model can represent. Inputs that rely heavily on this unpredictable remainder are themselves a novelty signal.

7. **Wake-sleep consolidation for the abstraction ratchet**: The closed-loop model (Run 4) relies on the forward model's injection at inference time — the main model learns to *use* predictions but not to *internalize* them. The brain solves this with regime alternation: the cerebellum provides predictions during waking, and the cortex consolidates them during sleep via offline replay (thalamocortical spindles, without real-time cerebellar correction). Two distinct consolidation timescales are at play: (a) the cortex slowly consolidating cerebellar predictions into its own weights (slow, Hebbian, requires many repetitions and possibly sleep), and (b) the cerebellum rapidly retraining to catch up to the cortex's changed representations (fast, supervised via climbing fiber errors). This asymmetry creates a natural cycle where the cerebellum is usually ahead and the cortex is still consolidating. Engineering analog: interleave closed-loop training (with injection — the model uses predictions and develops novelty awareness) and open-loop training (without injection — gradient pressure forces the main model to internalize whatever it was relying on the prediction for). Could anneal injection strength over cycles. If consolidation works, this enables the abstraction ratchet: the forward model predicts routine computation → the main model offloads it → frees capacity for new computation → the forward model catches up → the main model consolidates and offloads again → each cycle pushes toward more compressed, abstract representations.

## Relationship to the original CNB proposal

This doesn't invalidate the original CNB idea — it refines its core arguments and corrects its architectural commitments.

**What carries over**:
- The size constraint as a meaningful architectural choice — a small auxiliary model that approximates the main model's computation under a capacity budget
- The conditional framing — predict activations conditioned on current state, not model the unconditional activation distribution (validated: 200× cheaper than GLP at Llama scale)
- The CLS motivation — AI needs a structurally distinct fast-learning complement
- The recognition that the GLP's unconditional manifold modeling is the wrong approach

**What changes**:
- Weight-space decomposition is an optional initialization, not the core architecture
- Grounding comes from co-training and feedback, not from decomposing weights
- The forward model conditions on activations (like the cerebellum), not on weight decompositions
- The primary biological analog shifts from hippocampus to cerebellum
- The proposal gains the self-map property — the main model develops learned self-knowledge — which the original didn't anticipate
- The looped transformer provides a natural architectural home that the original lacked

**Why the original's weight-space commitment was wrong**:
The original argued that activations are "symptoms of computation, not the computation itself" and that grounding required accessing "where the model's knowledge actually lives" (weights). The cerebellum demonstrates that you can build an excellent forward model of a system without inspecting its parameters — you just observe it operating and learn the dynamics. The main model's weights are accessed implicitly every time it processes the forward model's predictions through its own forward pass.

## Intellectual lineage

- **CLS theory** (McClelland et al.): hippocampus/neocortex division as fast/slow learning. Motivates the need for a distinct novelty detection system.
- **Cerebellar forward models** (Schmahmann, Ramnani, Ito): the cerebellum as a general-purpose prediction engine for cortical dynamics, not just motor control. Primary biological analog for this proposal.
- **Predictive coding** (Friston, Rao & Ballard): hierarchical prediction error as the organizing principle of cortical computation. Provides the theoretical framework for precision-weighted novelty.
- **GLP** (this project): activation-space meta-modeling via flow matching. Demonstrates semantic residuals but limited by unconditional modeling.
- **VPD** (Bushnaq, Sharkey et al., 2026): adversarial parameter decomposition. Provides interpretable weight-space structure; useful as initialization but wrong as architectural commitment.
- **CNB grokking experiments** (this project): proof of concept for activation-matching self-regulation. Validates the signal; the architecture is what's new here.
- **Looped transformers / Universal Transformers / DEQ**: recurrent depth via weight sharing. Provides the native architectural slot for prediction injection.
- **Fast weight systems / TTT** (Schmidhuber; Sun et al.): online-updating auxiliary models. Related approaches to the forward model's learning regime.
- **Conditional rate-distortion / information bottleneck** (Tishby et al.): compress conditioned on what you already know — the formal frame for "transmit only surprise."
- **Cerebellar evolution** (Neubauer et al.): cerebellar globularization as one of the last anatomical changes in H. sapiens (100k-35k years ago), coinciding with behavioral modernity. Suggestive of the cerebellum's role in enabling recursive abstraction and metacognition.
