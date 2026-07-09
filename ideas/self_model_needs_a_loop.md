# Self-Knowledge Needs a Loop: the map–model distinction, and why strong self-reference is ill-posed feedforward

**Status**: Hypothesis (theory + one supporting α=0 control on MNIST). No looped experiment run yet.
**Date**: 2026-07-08
**Builds on**: [activation_to_activation_forward.md](activation_to_activation_forward.md), [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md), [local_prediction_error_learning.md](local_prediction_error_learning.md)
**Key experiments**: [a2a_forward/README.md](../experiments/a2a_forward/README.md) (whole arc), [FORWARD_MODEL_SWAP_README.md](../experiments/a2a_forward/FORWARD_MODEL_SWAP_README.md), [DISTILLATION_README.md](../experiments/a2a_forward/DISTILLATION_README.md), [MNIST_LOCAL_LOSS_README.md](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md), [rhm/RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md), [rhm/RHM_FM_REGULARIZER_README.md](../experiments/rhm/RHM_FM_REGULARIZER_README.md)

## One-liner

The "object-level self-knowledge" we thought closed-loop training + distillation produced is mostly **legibility** (an absorbed *map*: M computes what a compressed model of it predicted) plus re-derivation from data — not a **self-model** (a separable, causally-used *forecast* of M's own computation M can run and check). Strong self-referential self-knowledge is structurally **ill-posed and unpressured in a feedforward net**, and becomes **forced (locally) in a weight-shared looped model with forward-model injection**, where a converged operator's resting state must be self-consistent with a forecast of itself.

## The two-line vocabulary this doc introduces

- **Map (legibility / object-level absorption)**: M's weights come to *compute* what the forward model (FM) predicted. Amortized — per input, M represents *nothing* about the FM; it just produces the forecast answer. Changes *what M outputs*. This is what distillation gives, and it is re-derivable from data.
- **Model (self-reference / strong self-knowledge)**: M carries a *separable, causally-used forecast* of its own computation, held as a distinct object it can consume, check, and run counterfactually. Adds a *channel*; doesn't merely change the output. This is what we have **never** built.

The distinction we had been using — **meta vs object** ([MNIST_LOCAL_LOSS_README.md](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md), ported to RHM in [RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md)) — is a *probe-level* decomposition of one activation snapshot. Map-vs-model is orthogonal to it and is about **causal role**, not probe target. A meta probe can read a *correlate* of self-knowledge off activations; it cannot tell whether M *uses* a forecast as a model.

## The empirical trigger: the α=0 distillation control on MNIST (2026-07-08)

The whole a2a arc read distillation as the step that converts closed-loop meta-knowledge into internalized object-level self-knowledge ([DISTILLATION_README.md](../experiments/a2a_forward/DISTILLATION_README.md), [GEOMETRY_README.md](../experiments/a2a_forward/GEOMETRY_README.md)). But the language distillation loss was `0.5·KL(student‖teacher) + 0.5·CE_NTP` — **dense NTP included, no α=0 control**. The RHM line ([RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md), findings 5 + sparse-KL section) had shown on RHM that the teacher KL transfers *nothing* — val and representation re-derive identically with the KL removed — because RHM's compact grammar makes withheld supervision recoverable ("distillation is load-bearing only when the withheld supervision carries information the student can't reconstruct from the retained supervision"). It explicitly flagged the cross-domain follow-up: **run α=0 + the cross-model internalization probe on MNIST/language.**

We ran it on MNIST (`mnist_distillation.py`, `--distill-alpha {0.5, 0.0}`, alpha-tagged save paths; results on the `language-reduction-data` volume under `/a2a_forward/mnist_distillation/vit_4L_4H_128D/post_block0_to_post_block3/alpha_{0.5,0.0}/`). The α=0 arm is a pure injection-off CE fine-tune — the re-derivation control language never had. It split our "object-level internalization" evidence into two very different signals:

| What distillation was thought to do | α=0 control verdict |
|---|---|
| Close the dependency gap (competence) | **Re-derivation** — α=0 closes it *slightly better* (211% vs 204%) |
| Test 2: FM-function linearly *decodable* from M | **Re-derivation** — Δ vs OL identical across α (~+0.18 @block3, gap ~0.00) |
| Retain combined-system robustness | **Inherited from CL wake + preserved by re-derivation** — the KL *hurts* it (α=0 Dist/OL = 0.12 at eps=1.0 vs α=0.5's 0.61; α=0 is even more robust than CL) |
| Test 3: M's computation *agrees with* the FM (cosine) | **KL-specific** — +0.023 with KL (M moves toward FM) vs −0.029 without (M drifts away). A clean sign flip. |

Reading:

1. **Test 3 is a legibility measure** ("how legible / agreeable is M to a compressed model of itself"), *not* self-knowledge. We proved this dissociation earlier and forgot it: the **LL condition** ([MNIST_LOCAL_LOSS_README.md](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md)) *directly optimizes* this legibility (`λ·MSE(FM(post_block0), post_block3)`), reaches FM cosine **0.997**, and has **zero** self-knowledge (SK R² ≈ −0.03 to −0.05). Maximizing exactly what Test 3 measures produces no self-knowledge.
2. So the KL's *only* unique contribution on MNIST is to **buy FM-legibility at the cost of robustness** — an FM-regularization-shaped effect (cf. [RHM_FM_REGULARIZER_README.md](../experiments/rhm/RHM_FM_REGULARIZER_README.md), which showed functional simplification needs no self-knowledge apparatus at all). It is *not* "absorbing the combined system's self-knowledge."
3. **The genuine combined-system property (robustness) is generated during wake (the loop), not transferred during sleep.** α=0 preserves it; the KL erodes it. This matches RHM from the other side ("injection generates, sleep transfers nothing"). Both domains now say: **the loop does the work; distillation only preserves or erodes.**
4. Legibility and robustness are **anti-correlated everywhere**: LL (max legibility) is 13× brittle; KL distillation (raises legibility) trades away robustness; α=0 (doesn't chase FM legibility) keeps it.

Net: our MNIST/language object-level self-knowledge evidence, once controlled, reduces to **legibility + re-derivation**. We have not demonstrated a self-*model*.

Caveats: single seed (the sign flips are the robust part, not the magnitudes; user chose not to seed-replicate for now). Robustness Δloss values are small. This warrants a revision of DISTILLATION_README's "robustness partially retained" reading and MNIST_DISTILLATION's internalization claims — **pending discussion + a seed replicate before those READMEs are rewritten** (per repo convention: discuss results before writing a README).

## Why strong self-knowledge is ill-posed feedforward (not a capacity claim)

A feedforward transformer *can* represent information about its own state at t+k (autoregressive models demonstrably encode future-token info). Capacity is not the obstruction. The obstruction is **causal separability and pressure**:

- In feedforward M, `h_{t+k}` is produced by the intervening layers from `h_t`. A probe that decodes `~h_{t+k}` from `h_t` finds a *correlate* read off the very activations that *constitute* the computation. **Nothing downstream consumes "the forecast of h_{t+k}" as an object distinct from "the actual h_{t+k}".** Even a perfect probe shows the info is *present*, never that it *functions as a model*. Predictor and predicted are the same physical tensor from the same weights.
- **Unpressured**: a feedforward net never *has to* forecast its own future — it just runs the layers and finds out. The only forecast pressure we ever created was the injection, artificially. (Confound floor: even OL gets R²≈0.56 on block0→block3 self-predictability — smoothness, not a self-model. [distillation_probes Test 1](../experiments/a2a_forward/DISTILLATION_README.md).)

This is why we kept sliding into legibility: legibility is well-defined feedforward precisely *because* it requires no extra representation — it's a property of the map. A self-model requires a separate, redundant channel encoding what the computation will do, held apart from doing it. Feedforward has no home for it.

Reframe (from discussion): **strong self-knowledge ≈ being able to run a rough forward pass on yourself, off to the side.** A counterfactual is a simulated forward pass; epistemic honesty is a rough forward pass aware of its own boundaries. The operative word is *simulated* — you must execute a cheap model of your computation **without committing to the real computation.** Feedforward can't: to know what block 3 says, you run the layers. The separately-runnable cheap operator is *literally what the FM is* (a 1% model you can execute in parallel).

## The mechanism: weight-sharing forces producer = consumer

In the feedforward closed-loop model the two halves of a would-be self-model live in **different weights** (the [FORWARD_MODEL_SWAP_README.md](../experiments/a2a_forward/FORWARD_MODEL_SWAP_README.md) injection-point crossover):

- **pre-injection layers = producers** of the FM's input → develop FM-*agnostic* (object-ish) self-knowledge (small +).
- **post-injection layers = consumers** of the FM's output → develop FM-*specific* meta-knowledge (the residual).

Because these are different weights, the producer never has to *be* the consumer — it's shaped (amortized, via backprop through the injection path) to emit good FM-inputs, but has no reason to *represent* what the consumer will do with them. The self-model is split across real estate and never reconciled inside one operator.

**Weight-sharing collapses this.** In a looped model injecting `FM(h_t)` each step, the same operator `g` must simultaneously *produce* the state the FM reads and *consume* the FM's forecast — applied to its own previous output. The injection is both upstream and downstream of every application of `g`. Producer and consumer are forced into one function, evaluated on its own output. This is the "backwards-and-forwards causality localized to a layer" a self-model needs, and weight-sharing is what forces it. (Feedforward would need N separate self-models, one per layer-pair, with nothing to consolidate into; the loop has one reusable operator to model — a genuine candidate for "information about my own weights, since the weights are one operator.")

## The payoff: the fixed point forces local self-consistency (this is the seed)

Write the loop as `h_{t+1} = g(h_t + gate·FM(h_t))`, FM forecasting the next state. Two equilibria:

1. **Ignore the forecast** (gate→0): `h* = g(h*)`, an ordinary fixed point, no self-consistency, no self-knowledge — the trivial corner.
2. **Use the forecast** (gate open): `h* = g(h* + gate·FM(h*))`. For `h*` to be a stable fixed point `g` actually reaches, the forecast **cannot push `g` away from where it was going** — else it isn't a fixed point. So **convergence forces the forecast to agree with the actual resting state.**

What selects corner 2 is **usefulness** — the gate opens only because the forecast helps (as it did feedforward: gate norm grew 0.08→3.0, [CLOSED_LOOP_README.md](../experiments/a2a_forward/CLOSED_LOOP_README.md)). So the load-bearing statement is:

> **(forecast is useful → gate opens) + (weight-sharing → producer = consumer) + (convergence → fixed point) ⟹ `g` is an operator whose actual resting state agrees with a forecast of that resting state.**

This is more than "a trained predictor is accurate": the agreement is a **constraint of any converged solution**, not something optimized in. The forecast is **self-fulfilling** — `g` has restructured so that being told where it's going is *consistent with going there*, and it *exploits* the expectation of its own future to get there. An operator *helped by and stable under a forecast of itself* has structured itself around a model of its own trajectory. **Forced, not permitted.** No analog exists feedforward (no fixed point, no equation `g` must satisfy against its own forecast).

**Honest boundary**: this pins agreement *at and near the resting state* — **local** self-consistency. It says nothing about the forecast being accurate far from where `g` settles. It is the *seed* of self-knowledge, likely necessary but not obviously sufficient for the full counterfactual-capable kind.

## Two corrections from discussion (important — they change the experiment)

1. **FM-generality is neither forced nor the right target.** Nothing in the looped setup pressures agreement with FMs it never saw; the self-consistency is with *its own* co-trained FM — one apparatus. Expect looped self-knowledge to be **FM-specific**, like feedforward. And FM-specific is **correct, not a defect**: human self-knowledge is calibrated to your one actual cognitive apparatus, not a universal self-model ([FORWARD_MODEL_SWAP_README.md](../experiments/a2a_forward/FORWARD_MODEL_SWAP_README.md) says this; we then wrongly treated FM-specificity as disqualifying). **The right discriminator is map-vs-model (separability + causal use), NOT specific-vs-general.** The swap test measured the wrong axis to tell them apart.
2. **Near-manifold, not off-manifold.** The fixed-point argument only guarantees local consistency, and humans don't run counterfactuals far off-manifold anyway. The realistic target is **near-manifold** self-simulation (plausible small perturbations: "what if this input/token were slightly different"). Theory and realism agree.

## The experiment this implies

Build a **looped transformer with FM injection at each step** (self-model target = next iterate or fixed point). Compare against the feedforward closed-loop model. Discriminators — chosen to be **FM-specific-friendly** (causal/behavioral, not cross-FM transfer):

1. **Causal necessity**: ablate the anticipatory / injected-forecast component → does *convergence* break (loop fails to settle / degrades), in a way the feedforward absorbed-map does not?
2. **Near-manifold self-counterfactual**: can M use its self-forecast to answer "what would I output if this input were slightly perturbed" better than an OL/feedforward baseline — a behavior the absorbed map should fail and a real self-model should pass?
3. **Fixed-point self-consistency**: measure whether the converged `g`'s trajectory agrees with the FM's forecast of it (the resting-state self-consistency the theory predicts), and whether it's *stable* under injecting the forecast.

Positive result = the loop produces a **separable, causally-used, FM-specific, locally-scoped self-forecast (a model)**, while the feedforward model produces only the **absorbed map (legibility)** distillation gives. That is the whole thesis, demonstrated.

Cheaper feedforward pre-check (optional, mainly as a baseline): a probe testing whether M can run its self-forecast on a *near-manifold* perturbation it wasn't trained on — the absorbed map should fail it. Expect this to keep bumping into the feedforward degeneracy; its value is as the contrast the looped result beats.

## Biology (not serendipity — the same structure)

The cortico-cerebellar circuit *is* a loop (cortex → pons → cerebellum → thalamus → cortex), and cortex is massively recurrent with dense feedback. The thalamic return of the cerebellar forecast lands on cortical circuitry that is *both upstream and downstream* of that forecast — the same populations that generate the state the cerebellum predicts *from* also receive the prediction *back*. That is **producer = consumer in wetware**. Our feedforward port was the tractable approximation; the biology was signaling the loop is load-bearing all along. This also predicts the injection sits *inside* the recurrence (before and after the cortical injection point are the same recurrent circuit), not bolted onto a feedforward stack.

## The injection as a new self-referential input modality (efference copy) — appended 2026-07-08

A separable line from the discussion, worth persisting: **injecting an activation predicted end-to-end in an activation-to-activation sense (the FM) rather than a token-to-token sense (NTP) may introduce a qualitatively new *class of input* to the model — not new information, but a new modality.**

To not fool ourselves: information-theoretically the injection adds **zero** Shannon information. `FM(post_block0)` is a deterministic function of an activation the model already holds. If "new data" meant "new information about the world," the answer is no. The interesting sense is about **type**, along three axes that don't reduce to information content:

1. **Reference / aboutness.** Every other input the model receives is *about the world* (tokens/pixels) or a deterministic consequence of it (its own activations, downstream of data). The injected forecast is *about the model itself* — a prediction of its own future state. First input whose referent is its own computation, not the environment.
2. **Abstraction-level provenance (and gauge).** The FM's entire input-output signature lives in activation-space; it never touched a token. So the signal is native to representation-space and carries a **different inductive bias / gauge** than anything token-derived. This is not just an analogy: the FM-swap result showed the FM finds a *different parameterization of the same function* (zero weight cosine, near-perfect attention-pattern cosine — gauge symmetry; see [FORWARD_MODEL_SWAP_README.md](../experiments/a2a_forward/FORWARD_MODEL_SWAP_README.md) and the structure analysis in [a2a_forward/README.md](../experiments/a2a_forward/README.md)), and **the cerebellum and cortex are directly observed to hold different gauges for the same representations** — so the injected signal genuinely arrives in a different coordinate system than the cortex's own, exactly as in wetware. Crucially the FM's abstraction level is **endogenous** — it rises with the model — whereas a token signal is exogenous, pinned to abstraction level 0 forever (the RHM/KFW point). The injection is the model's first exposure to a signal whose abstraction height *tracks its own frontier*.
3. **Graph/temporal placement.** It relocates a forecast of a *later* computation to an *earlier* point — no new information, but different *accessibility* (like a derivative or Fourier transform: same information, qualitatively different representation for the operations you want to perform).

So: **not new data (no information), but a new modality — a self-referential, activation-native channel.**

**The channel-type is new unconditionally; the content-level is target-dependent.** Whether this modality carries genuinely *higher-abstraction* structure or just a copy of what the model already reached depends on the FM's target — exactly the latent-vs-token result ([RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md)): token-FM injection carries only the shallow already-reached frontier (RHM-token shallow; MNIST/language deep because the task supervises deep), while latent-FM injection carries frontier structure. Same new modality, different payload — resolving why token-CL on RHM produced FM-specific noise-meta while latent-CL produced FM-general frontier-meta.

**Injection as an implicit inductive bias toward FM-regularization *and* DGP-legibility.** Because the injected signal lives at an endogenous, representation-native abstraction level, feeding it back plausibly biases the main model *toward being FM-legible* (the [RHM_FM_REGULARIZER_README.md](../experiments/rhm/RHM_FM_REGULARIZER_README.md) "be legible to a compressed self" pressure, but arriving through the injection channel rather than an explicit MSE term) and, via the FM's endogenous abstraction level, *toward DGP-alignment* — consistent with the observed increased FM↔DGP alignment under the latent target on RHM (the FM residual becomes near-ceiling FM-invariant / DGP-aligned, ens_cos ~0.98). This is a candidate mechanism by which the *modality itself* — not a separate regularizer — nudges representations up the abstraction ladder. It reopens, with the right substrate, the original "self-knowledge as implicit regularizer / abstraction-lift" intuition that feedforward legibility only approximated.

**Reframe this enables.** If the injection is a new *modality* rather than "extra compute help," then **self-knowledge is what you'd expect: the model building representations to *interpret a new class of input*.** You add a novel channel; the model reorganizes to make sense of it. The dependency we kept measuring is then not a bug but a *slot* allocated for an expected input modality.

**Biological grounding — this is efference copy / corollary discharge.** Nervous systems categorically distinguish internally-generated predictions of the consequences of their own activity from externally-caused sensory input — different pathways, used to cancel self-generated sensation, attribute agency, and stabilize perception. The cerebellar→thalamic→cortical prediction *is* that class of signal. So biology answers the question with a definite yes: an internally-generated forecast of one's own state is a categorically distinct input class, not more sensory data — and it is the substrate of self-modeling. Our FM injection is the artificial efference copy.

**Testable edge (feedforward-runnable, not degenerate).** If the injection is truly a distinct modality, the model should treat injected vs. genuinely self-computed activation *differently* — the corollary-discharge signature is *cancellation/attribution*, not summation. Test: does a trained closed-loop model distinguish "this activation was injected" from "this was self-computed," and use that distinction functionally (e.g., discount self-generated components, as corollary discharge cancels self-caused sensation)? If injection were just noisy extra data there'd be no such asymmetry. Runnable on existing closed-loop checkpoints, and — unlike the self-prediction probes — not confounded by feedforward degeneracy, because it tests *differential treatment of two input sources*, not "does M encode its own future."

## Connections to beliefs / prior results

- **Legibility ≠ self-knowledge ≠ robustness.** Three dissociated axes: LL gives legibility with zero self-knowledge and *negative* robustness; the loop (wake) gives robustness; distillation-KL gives legibility at robustness's expense. [dimensionality_expansion](../beliefs/dimensionality_expansion.md)'s health-triple should carry a *causal-role* label (map vs model), not just probe-target labels.
- **RHM latent target** ([RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md)): the one place we produced *FM-general* meta — but via a privileged latent target that manufactures a deep moving frontier. MNIST gets that frontier free (the class label *is* the root of the DGP; classification is necessary-and-sufficient to characterize it, so supervision is deep-by-construction — plausibly why MNIST's ratchet/meta-learning compounded where RHM-token stalled). General principle: **every non-MNIST DGP needs a latent target to synthesize a deep moving frontier, *and then* the loop to climb it** — complementary halves.
- **The ratchet** ([cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md)): if sleep transfers nothing (re-derivation) and the loop does the generative work, the ratchet's "click" must come from wake, not sleep — reframes what the sleep phase is for (preservation/consolidation, not transfer).

## Context pointers for a future agent (how to reach these conclusions)

1. Read [a2a_forward/README.md](../experiments/a2a_forward/README.md) end-to-end for the closed-loop / self-knowledge / distillation arc.
2. [FORWARD_MODEL_SWAP_README.md](../experiments/a2a_forward/FORWARD_MODEL_SWAP_README.md) — the pre/post-injection crossover (FM-agnostic producers vs FM-specific consumers) is the empirical seed of the producer=consumer argument.
3. [MNIST_LOCAL_LOSS_README.md](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md) — LL = pure legibility, FM cos 0.997, zero self-knowledge; the meta-vs-object probe machinery (`mnist_local_loss_probes.py`), and legibility→brittleness.
4. [DISTILLATION_README.md](../experiments/a2a_forward/DISTILLATION_README.md) + `distillation_probes.py` — the internalization Tests 1/2/2b/3 whose interpretation the α=0 control revises.
5. The α=0 result: rerun `modal run --detach a2a_forward/mnist_distillation.py::a2a_mnist_distillation --distill-alpha {0.5,0.0}` (alpha-tagged paths added 2026-07-08). Compare `internalization.test3_fm_prediction_quality` (CL→Dist cosine step: +0.023 with KL, −0.029 without) and `robustness` (α=0 more robust than α=0.5).
6. [rhm/RHM_LATENT_LOOP_README.md](../experiments/rhm/RHM_LATENT_LOOP_README.md) — the α=0 control on RHM (KL inert), the recoverability ladder, and FM-general meta via the latent target; [rhm/RHM_FM_REGULARIZER_README.md](../experiments/rhm/RHM_FM_REGULARIZER_README.md) — functional simplification needs no self-knowledge apparatus.
7. Parent idea: [activation_to_activation_forward.md](activation_to_activation_forward.md); looped architecture appears as a next-step there and in the a2a README ("Looped transformer") — this doc supplies the *reason* it's load-bearing (self-knowledge becomes measurable/forced only there), not just "more depth."

## Open questions

- Is the fixed-point self-consistency (local, necessary) *sufficient* for usable near-manifold counterfactuals, or only the seed? (The core empirical unknown.)
- Does the loop's self-forecast survive removal of the external FM (internalization), so M runs its *own* rough forward pass — the "imagine off to the side" capability — and if so, is *that* the distillation step that finally does something a plain CE fine-tune can't?
- Does an internal self-model, once present, act as the implicit regularizer / abstraction-lift the FM-regularizer intuition predicted — the thing feedforward legibility approximated but didn't deliver? (Reopens the "self-knowledge as implicit regularizer" thread with the right substrate.)
- Does the model treat injected (efference-copy) activations as a **categorically distinct input class** — cancellation/attribution rather than summation (the corollary-discharge signature)? If so, the injection is a genuine new modality, not noisy extra data. (Feedforward-runnable on existing CL checkpoints; not degenerate.)
- Does the injection *channel itself* bias the main model toward FM-legibility and DGP-alignment (an implicit regularizer arriving through the modality, not an explicit MSE term)? Testable by comparing FM↔DGP alignment / activation-manifold rank of a plain closed-loop model against the explicit FM-regularizer at matched knowledge.
