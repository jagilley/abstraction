# Cancellation, not summation: the forward-model injection as efference copy / predictive coding

**Status**: Architectural proposal — **tested on the looped ViT (2026-07-13): primary claim confirmed.** See [experiments/a2a_forward/CANCELLATION_README.md](../experiments/a2a_forward/CANCELLATION_README.md) and the "Experimental status" section below. This **deepens the efference-copy thread appended to [self_model_needs_a_loop.md](self_model_needs_a_loop.md) (2026-07-08)** from a *diagnostic* ("do summation-trained closed-loop models happen to cancel their self-generated component?") into a *prescription* ("wire the loop so it cancels — subtract the forecast, propagate the residual").
**Date**: 2026-07-12
**Builds on**: [self_model_needs_a_loop.md](self_model_needs_a_loop.md), [activation_to_activation_forward.md](activation_to_activation_forward.md), [local_prediction_error_learning.md](local_prediction_error_learning.md), [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md)
**Key experiments**: a2a_forward closed-loop arc — [CLOSED_LOOP_README](../experiments/a2a_forward/CLOSED_LOOP_README.md), [EXTENDED_TRAINING_README](../experiments/a2a_forward/EXTENDED_TRAINING_README.md) (Run 7b, net-zero benefit), [BASELINE_BATTERY_README](../experiments/a2a_forward/BASELINE_BATTERY_README.md), [DISTILLATION_README](../experiments/a2a_forward/DISTILLATION_README.md); the looped transformer [LOOPED_README](../experiments/a2a_forward/LOOPED_README.md); RHM sculpting Stage 3b/3c ([RHM_SCULPTING_README](../experiments/rhm/RHM_SCULPTING_README.md)) — the re-grounding + Kalman-filter patches this generalizes.

## One-liner

Our forward-model loop currently **adds** the FM's prediction to the residual stream (`h + gate·p` — *summation*, a side-channel) and computes the novelty residual `r = actual − p` off to the side as a loss/probe target. It should instead **subtract** the prediction and propagate the residual (`f(h − gate·p) + p` — *cancellation*, the forward path): the efference-copy / predictive-coding wiring. Summation treats the FM as **optional help**, which produces *entangled* dependency, a runaway gate, and rollout drift. Cancellation treats the FM as a **permanent, modular component**: it makes the novelty residual the tensor the model actually computes on (giving the separable self-model that feedforward otherwise has no home for), adds a restoring force that self-corrects drift, and self-regulates the gate by forecast quality.

## Experimental status (2026-07-13, looped ViT — single seed, Fashion-MNIST)

Built cancellation in as a **one-flag strict generalization** on the looped ViT (`inject_mode=cancel`: `s_{t+1} = G(s_t − gate·inj) + gate·inj`, zero-init gate → identical to the plain loop at init), everything else held fixed against the summation baseline (whose re-run reproduces the [LOOPED](../experiments/a2a_forward/LOOPED_README.md) headline exactly). Full writeup: [CANCELLATION_README.md](../experiments/a2a_forward/CANCELLATION_README.md).

- **Payoff 3 (restoring force) — CONFIRMED three ways.** The perturbation `error_correction` **flips −0.18 → +0.05** (summation amplifies a mid-loop perturbation; cancellation corrects it). The **mechanism is a directional cancellation of the forecast** — the net effect of the injection on the next state goes from `cos(Δ,inj)=+0.89` (summation, amplify) to `+0.095` (cancellation) — and it is **topological, not learned**: every operator, cancellation ones included, responds to a raw `+inj` identically (`cos ≈ 0.89`); only the subtract-then-re-add topology cancels ("restoring force by construction", as predicted). And on **OOD inputs** the summation injection turns actively harmful while cancellation's stays helpful.
- **Payoff 4 (benefit tracks forecast quality) — MECHANISM CONFIRMED.** Corrupting the forecast directly (interpolate toward a garbage FM, magnitude-matched), cancellation's benefit **falls monotonically and flips negative** (+0.136 → −0.148) as forecast quality degrades, while summation stays positive (+0.176 → +0.037). This is the loss-landscape asymmetry that would drive gate self-closing (a wrong forecast *hurts* under cancellation, is *benign* under summation). The gate *dynamics* (whether a trainable input-conditioned gate actually closes) remains untested — only the pressure that would drive it is established.
- **Payoff 1 (modular dependency) — SUBSTRATE-LIMITED / inconclusive.** The fresh-FM swap costs identical accuracy for both wirings, but only because the low-rank loop's FM is near-unique, so a "different but equally-good" FM barely exists to swap in — too weak a perturbation to test entanglement. Needs a gauge-free substrate (RHM / language).
- **Cost — none.** Task accuracy is unchanged (cancel 0.850 vs sum 0.852), as the doc predicted ("no net loss benefit is expected").
- **Refinement (honest):** the mechanism cancellation is **directional** (`cos(Δ,inj) → 0.095`), *not* a magnitude null — the operator amplifies (gain ~2–4), so cancellation *orthogonalizes* the forecast out of the forward path rather than zeroing its magnitude. The directional axis is the meaningful one (the a2a self-knowledge is directional throughout). Cleanest in the `update` (deviation) form; `next_state` cancels only partially.

## The two-line vocabulary this doc introduces

- **Summation (side-channel injection)**: forward path is `h + gate·p`. The forecast `p` is an *extra addend* to the main signal; the residual `r = actual − p` is computed *off to the side* (the local-loss target in [MNIST_LOCAL_LOSS](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md); the self-knowledge probe target). The main computation flows through the full, prediction-boosted signal. **This is what the whole a2a arc has done.**
- **Cancellation (forward-path residual)**: the forecast is *subtracted* — `e = h − gate·p` — and **the residual `e` is what propagates downstream**; the forecast is re-added at read-out — `output = gate·p + f(e)`. Downstream capacity is spent entirely on the part the FM couldn't predict. This is **Rao–Ballard predictive coding** / **efference copy**: top-down prediction subtracted, only the prediction *error* transmitted forward.

The distinction is orthogonal to map-vs-model and meta-vs-object (the other two axes in the a2a program). Those are about *what a probe reads off one activation snapshot*; this is about **how the forecast is wired into the forward computation** — a sign (+ vs −) and a topology (side-branch vs main path).

## Why this matters: the pathology summation produces

The a2a loop helps, but three failure modes recur across every closed-loop experiment, and all three trace to the *additive* wiring:

1. **Entangled dependency** ([CLOSED_LOOP](../experiments/a2a_forward/CLOSED_LOOP_README.md): +0.11 nats worse without the injection; [EXTENDED_TRAINING](../experiments/a2a_forward/EXTENDED_TRAINING_README.md) Run 7b: benefit and dependency exactly cancel). Adding `gate·p` hands downstream blocks a *free preview of the answer*. The NTP gradient rewards using it, so the predictable computation gets **distributed into the interaction between the injected preview and the downstream weights**. Remove `p` at test and you've removed a load-bearing term those weights silently rely on. The reliance is *implicit and entangled* — that is the dependency bug.
2. **Runaway gate.** On a fixed distribution *more free preview always lowers loss within-distribution*, so the gate opens monotonically and never closes ([CLOSED_LOOP](../experiments/a2a_forward/CLOSED_LOOP_README.md): 0.08→3.03; the MNIST gated-ratchet gate "opens rather than closes," 0.31→0.77). Selective closing only appears under an explicit distribution shift + a bilevel/FOMAML gate ([OOD_GATE](../experiments/a2a_forward/OOD_GATE_README.md)). The additive wiring has no endogenous pressure for the gate to close where the forecast is *wrong*.
3. **Rollout drift.** Rolling the FM forward generatively (`h_{t+1} = FM(h_t)`) compounds error with no restoring force (the Stage-2 latent-planner collapse; the open-loop latent rollout in [RHM_SCULPTING](../experiments/rhm/RHM_SCULPTING_README.md) Stage 3b). We patched this twice — **Dreamer-style re-grounding** (never roll the FM more than one step from truth, Stage 3b) and a **Kalman-style filter** (observed block → re-encode, occluded → keep FM prediction, Stage 3c). Both are *local instances of a corrector the additive loop lacks by construction.*

## The reframe: subtract the forecast, propagate the residual, recombine at read-out

The residual stream is a **common d-dimensional space across layers**, so a prediction `p` of a *later* layer's residual-stream state is commensurate with an *earlier* layer's state — which is exactly why additive injection at an intermediate site is well-posed. Subtraction is equally well-posed. Concretely, for the feedforward a2a setup (FM predicts `post_block_L` from `post_block0`, injected after block 1):

- **Summation (current)**: `h₁' = h₁ + gate·p`;  `y = Blocks_{2..L}(h₁')`  (downstream rebuilds the full `h_L`, with a head start).
- **Cancellation (proposed)**: `e = h₁ − gate·p`;  `y = Blocks_{2..L}(e)`;  `h_L^eff = y + gate·p`  (downstream transports and elaborates the *deviation from the forecast*; the forecast is restored at read-out).

The blocks are trained end-to-end on the prediction-*error* stream — that is the whole point. A **zero-init gate** makes it degrade gracefully to the ordinary forward pass, so it is a strict generalization of the current model, learned into rather than imposed.

For the **looped** transformer ([LOOPED](../experiments/a2a_forward/LOOPED_README.md)), `s_{t+1} = g(s_t + gate·FM(s_t))` becomes:

```
s_{t+1} = gate·FM(s_t)  +  g( s_t − gate·FM(s_t) )
```

- `gate→0`: `s_{t+1} = g(s_t)` — the plain loop.
- `gate→1`: the operator processes the deviation `s_t − FM(s_t)` and the forecast is added back — a predictive-coding loop.

(Where to subtract and where to re-add is a design axis; the invariant is *forecast subtracted from the forward path, residual propagated, forecast restored at read-out*.)

## Payoff 1 — dependency becomes modular instead of entangled (this is the core fix)

*Status (2026-07-13): **substrate-limited / inconclusive** on the looped ViT — the fresh-FM swap couldn't test it because the low-rank loop's FM is near-unique. Needs a gauge-free substrate (RHM / language).*

Under cancellation the predictable computation lives **entirely and explicitly in `p`** (added back cleanly at read-out); downstream weights are shaped to compute *only the correction* `f(e)`. So:

- The FM **owns** the predictable part; downstream **owns** the residual; they recombine at read-out. The reliance is **modular and explicit**, not entangled and implicit. Permanent reliance on the FM is now *correct* — the FM is a component of the model (its cerebellum), not optional help — and the pathology to guard against is no longer "can't run without it" but *instability*, which Payoff 3 addresses.
- The FM **cannot over-help.** It is not handing downstream an answer to get addicted to; it is removing a baseline. The better it predicts, the *smaller* the residual and the *cleaner* the novelty signal downstream must model. The division of labor is **structural** (enforced by the subtract-then-add-back topology), not *incentivized-and-hoped-for* (as in summation, where we hoped for specialization and got offloading).

This reframes "dependency" itself. The additive setup treats dependency as a bug because it treats the FM as optional. Cancellation makes dependency *the design* — echoing the [self_model_needs_a_loop](self_model_needs_a_loop.md) line "the dependency we kept measuring is not a bug but a *slot* allocated for an expected input modality." **Cancellation is the architecture that makes that slot real.**

## Payoff 2 — the novelty residual becomes a causal object (resolves the feedforward-separability problem)

The central obstruction in [self_model_needs_a_loop](self_model_needs_a_loop.md): feedforward, a probe can *decode* the residual as a *correlate*, but **nothing downstream consumes "the forecast" as an object distinct from the thing forecasted** — predictor and predicted are the same physical tensor from the same weights, so you can show the info is *present*, never that it *functions as a model*.

Cancellation resolves this *architecturally*, and cheaply: subtracting `p` makes the forecast (`p`) and the deviation-from-forecast (`e = h − p`) **two physically distinct tensors with distinct downstream roles**. The residual is no longer "encoded somewhere in the activations" — it is *literally the input to downstream computation*, hence definitionally consumed as an object. This is the **separable, causally-used forecast** the idea doc says feedforward has no home for — and cancellation gives it a home *without* requiring the full weight-shared loop. The two are complementary: the loop forces *producer = consumer*; cancellation forces *forecast held apart from residual*. You likely want both.

Note this is the *constructive* counterpart to the diagnostic test the parent doc proposed (does a summation-trained model *happen* to treat injected vs. self-computed activation differently — cancellation/attribution rather than summation?). Rather than test whether cancellation emerges from additive training, **build it in.**

## Payoff 3 — a restoring force that self-corrects drift (generalizes the re-grounding/Kalman patches)

*Status (2026-07-13): **confirmed** on the looped ViT — perturbation `error_correction` flips −0.18 → +0.05; the forecast is directionally cancelled from the forward path (cos +0.89 → +0.095), topological not learned.*

A predictive-coding loop is a **predictor–corrector**: predict the next state, then correct by `(observation − prediction)`. That is a Kalman filter — *exactly* what we hand-built in Stage 3c ("observed → re-encode; occluded → keep FM prediction") and the general form of the Stage-3b re-grounding. Cancellation gives the restoring force **by construction**: any drift of the state from what is actually true produces a larger residual `e`, which drives a stronger correction pulling it back. Additive injection has no such term — it just keeps adding prediction, so nothing pulls a drifting state home. **Rollout drift is fixed structurally instead of patched per-experiment.**

## Payoff 4 (prediction) — the gate self-regulates by forecast quality

*Status (2026-07-13): **mechanism confirmed** on the looped ViT — a wrong forecast hurts cancellation (benefit +0.136 → −0.148 as the forecast degrades) but is benign under summation (+0.176 → +0.037), the loss-landscape asymmetry that would drive gate-closing. The gate **dynamics** themselves are untested (needs an input-conditioned gate).*

Under summation, a *bad* added preview is cheap to ignore — downstream can set its effective weight low at near-zero cost — so there is no pressure for the gate to close on inputs the FM predicts poorly (hence the runaway). Under cancellation the asymmetry flips: subtracting a *wrong* forecast **actively corrupts the forward path** (downstream must both undo the bad subtraction *and* compute the real signal, from a higher-variance residual). So gate-opening is beneficial exactly where the forecast is accurate and *harmful* where it is not.

**Prediction**: with cancellation, the gate should **close on inputs the FM predicts poorly** (novel/OOD/high-surprise) and open where the FM predicts well — the selective-closing behavior [OOD_GATE](../experiments/a2a_forward/OOD_GATE_README.md) needed bilevel/FOMAML meta-learning to obtain, arising *for free from the wiring*. This is the sharpest falsifiable consequence.

## The fixed point strengthens the self-consistency seed into an attractor

[self_model_needs_a_loop](self_model_needs_a_loop.md) argues the looped fixed point *passively* forces the forecast to agree with the resting state (convergence ⟹ agreement). Under cancellation this becomes an **active attractor**: the natural fixed point of a predictive-coding loop is *minimal prediction error*, and a wrong forecast injects error the operator is penalized for, so the dynamics *descend toward* the self-consistent state `s* ≈ FM(s*)` rather than merely being consistent with it if they happen to converge there. Self-consistency stops being a consequence you hope for and becomes the quantity the loop minimizes. (Plausibility argument, not a proof — and note the resting error is only zero for *predictable* inputs; a genuinely hard input keeps `e` nonzero, which is precisely "process the deviation.")

## What you give up, and why it is cheap

Cancellation removes the *free preview* from the forward path, so it likely forfeits the additive loop's raw LM-loss help. But that help is close to illusory at the margin: [EXTENDED_TRAINING](../experiments/a2a_forward/EXTENDED_TRAINING_README.md) Run 7b found the **net val benefit is ~zero — pure computation relocation** (injection benefit exactly offset by dependency), and [BASELINE_BATTERY](../experiments/a2a_forward/BASELINE_BATTERY_README.md) found **LM loss and robustness are dissociated** (the autoencoder gave the *largest* LM benefit and the *worst* robustness). So you are trading a net-zero preview for structural stability, a self-regulating gate, and a genuinely separable self-model. By the program's own findings, a good trade.

## Biological grounding (this is efference copy / corollary discharge)

Nervous systems route the forward model's output as a **subtraction**, not an addition. A motor command's efference copy drives a forward model whose predicted sensory consequences (corollary discharge) are *subtracted* from incoming afference; only the **unpredicted** part propagates. This is why you cannot tickle yourself (self-generated, hence predicted, touch is cancelled/attenuated), why the visual world stays stable across saccades (predicted retinal shift is cancelled), and how agency is attributed. The cortico–cerebello–thalamo–cortical loop *is* this circuit; the thalamic return lands on cortex that is both upstream and downstream of the forecast (producer = consumer, per the parent doc). **Our additive injection is the tractable-but-wrong port; cancellation is the wiring biology actually uses**, and it is the wiring that solves exactly the dependency/stability problems the additive port kept hitting. The parent doc already noted "the corollary-discharge signature is *cancellation/attribution, not summation*" — this doc makes that the build target.

## Connections to predictive coding / efficient coding

- **Rao–Ballard predictive coding**: hierarchical top-down prediction, only prediction *error* transmitted forward — cancellation is this, with the FM as the top-down predictor.
- **Barlow efficient coding / redundancy reduction**: transmitting the residual instead of the full signal is a decorrelating/whitening transform. Downstream capacity is not wasted re-representing the predictable part — the same reason video codecs send *frame differences*. This is the concrete sense in which the injection is "same information, a representation better-suited to the operations you want" (the derivative/Fourier analogy in the parent doc's modality thread).
- **[local_prediction_error_learning.md](local_prediction_error_learning.md)**: the local loss uses `‖FM(h₀) − h_L‖²` as a *training signal*; cancellation uses the *same error* as the *forward signal*. Sibling ideas — one shapes weights toward legibility, the other routes the error through computation. Worth asking whether they compose (LL to make the forecast good, cancellation to route its residual).

## The experiment (small, RHM-instrumentable, strict generalization of existing code)

Re-wire the injection from `h + gate·p` to `f(h − gate·p) + p` on an existing setup (looped transformer or the RHM sculpting FM), zero-init gate, everything else fixed. Watch three things:

1. **Dependency**: the standalone-removal gap should stop being the natural metric / behave qualitatively differently (removing a *component* vs. removing *optional help*). Test whether the predictable computation is now *modular* — localizable in the FM — rather than entangled in downstream weights (e.g. swap in a fresh FM; cancellation should degrade more gracefully than summation because downstream only ever owned the residual).
2. **Gate dynamics**: under cancellation the gate should **self-limit** and, on a mixed-difficulty / mildly non-stationary stream, **close on poorly-predicted inputs** — without any bilevel machinery. (Direct test of Payoff 4.)
3. **Drift**: multi-step latent rollout should **self-correct** without the Stage-3b re-grounding crutch — the restoring-force prediction (Payoff 3).

RHM is the right substrate: ground-truth latents let us check whether downstream genuinely carries the *residual* structure and whether the recombined read-out matches the true `h_L`.

**Done (2026-07-13) — on the looped ViT, not RHM.** (1) *Dependency*: the fresh-FM swap was **inconclusive** — the low-rank loop's FM is near-unique, so the swap is too weak a perturbation to separate modular from entangled (the RHM substrate, with a gauge-free FM, is still the right place for this). (2) *Gate dynamics*: the loss-landscape asymmetry is confirmed (a wrong forecast hurts cancellation, is benign under summation — Payoff-4 mechanism), but the **gate-closing dynamics themselves are untested** (the scalar gate can't express input-dependence; needs an input-conditioned gate). (3) *Drift/restoring force*: **confirmed** — the perturbation `error_correction` flips sign (−0.18 → +0.05), and the forecast is directionally cancelled from the forward path (cos +0.89 → +0.095). The RHM port (ground-truth-latent residual check + it generalizes the Stage-3b re-grounding / Kalman patches) is the natural next substrate.

## Predictions / falsification

- **Falsified if** cancellation reproduces the same runaway gate and entangled dependency as summation (would mean the sign/topology is not what drives the pathology). → *Not falsified: the sign flip changes the mechanism (cos +0.89 → +0.095) and reverses the perturbation response, confirming the topology drives it. (The scalar gate's resting value is unchanged at ~0.017 under both — but that is the input-independent gate; the input-dependent closing pressure is present, see next.)*
- **Falsified if** the drift/self-correction and gate-self-closing predictions fail — i.e. the restoring force is not load-bearing. → *Restoring force **confirmed** (error_correction −0.18 → +0.05). Gate-self-closing pressure **confirmed** (benefit flips negative on a bad forecast) but the gate **dynamics** are untested (needs an input-conditioned gate).*
- **Confirmed strongly if** the OOD_GATE selective-closing behavior appears from the wiring alone (no bilevel), and rollout stabilizes without re-grounding. → *Partially: the loss-landscape asymmetry the OOD_GATE selective-closing rests on is present from the wiring alone; whether a trainable gate exploits it (no bilevel) is the open follow-up.*
- **Open magnitude question**: how much raw LM-loss help is forfeited, and whether robustness/self-model quality gains dominate (Run 7b + BASELINE_BATTERY predict the trade is favorable). → *Resolved favorably on the looped ViT: **task accuracy is unchanged** (0.850 vs 0.852), so no help was forfeited, while the robustness/restoring-force gain is real.*

## Connections to beliefs / prior results

- **[dimensionality_expansion](../beliefs/dimensionality_expansion.md)** frames the residual as the frontier (`R_res` = the un-absorbed slice of `R_act`). Cancellation makes `R_res` the *forward-propagating signal*, not just a measured quantity — the strongest possible version of "the model processes its frontier."
- **[self_model_needs_a_loop](self_model_needs_a_loop.md)**: cancellation is the missing *sign/topology* of the injection; the loop is the missing *recurrence*. The efference-copy thread there is the seed; this is its architectural cash-out. The parent's diagnostic test (does the model cancel?) becomes this doc's build target (make it cancel).
- **[conditional_novelty_bottleneck.md](conditional_novelty_bottleneck.md)**: its conditional-information-bottleneck reframe — "compress the activation conditioned on what the model already knows, so the bottleneck only transmits the *surprise*" — is the **same principle** as cancellation. The distinction is *use*: that doc builds a small (weight-space-grounded, via VPD) meta-model to **detect/code** novelty as a read-out signal; cancellation **routes the residual through the forward computation** as the propagating tensor. Detector vs. forward-path — two applications of "transmit only the conditional surprise." The grounding question it raises (novelty relative to the model's *concepts*, not its activation *statistics*) transfers directly: cancellation's residual is only as conceptually-grounded as the FM's forecast, which is the Stage-4 belief-depth lever ([RHM_SCULPTING](../experiments/rhm/RHM_SCULPTING_README.md)).
- **[cerebellar_abstraction_ratchet](cerebellar_abstraction_ratchet.md)**: if the loop (wake) does the generative work and cancellation stabilizes it, the ratchet's "click" may need the corrector to avoid the drift/norm-inflation that killed the extended (16-cycle) ratchet.

## Open questions

- Where should the subtraction and re-addition sit (single intermediate site vs. per-layer predictive-coding stack)? Per-layer is closer to Rao–Ballard but a larger change.
- Does cancellation compose with the **weight-shared loop** to give *both* producer=consumer *and* forecast-separability — the full self-model — or do they interfere?
- Does the whitening/decorrelation effect lift the abstraction level of downstream computation (the "modality nudges representations up the ladder" thread from the parent doc), now that downstream sees only the surprising part?
- Does cancellation change the *content* of self-knowledge from *legibility* (map) toward a genuine *model*, per the map-vs-model axis — i.e. is routing the residual through computation sufficient to make the self-forecast causally-used-as-a-model even feedforward?
- Interaction with **internalization**: once the FM is internalized (endogenous self-forecast, per [REACHING_INTERNAL](../experiments/a2a_forward/REACHING_INTERNAL_README.md)), cancellation subtracts the model's *own* forecast — the purest efference copy. Does internalized-cancellation behave differently from external-FM cancellation?

## Context pointers for a future agent

1. Read [self_model_needs_a_loop.md](self_model_needs_a_loop.md) end-to-end — especially the 2026-07-08 efference-copy append; this doc is its constructive continuation.
2. [activation_to_activation_forward.md](activation_to_activation_forward.md) + the a2a README for the additive injection as built, and the residual = actual − predicted definition.
3. [CLOSED_LOOP](../experiments/a2a_forward/CLOSED_LOOP_README.md) (dependency + runaway gate), [EXTENDED_TRAINING](../experiments/a2a_forward/EXTENDED_TRAINING_README.md) Run 7b (net-zero benefit → the trade is cheap), [OOD_GATE](../experiments/a2a_forward/OOD_GATE_README.md) (selective closing needed bilevel — the thing cancellation predicts for free).
4. [RHM_SCULPTING](../experiments/rhm/RHM_SCULPTING_README.md) Stage 3b/3c — the re-grounding and Kalman-filter patches cancellation generalizes; the cleanest first place to test the drift-self-correction prediction.
5. The experiment is a strict generalization (zero-init gate → identical to current model at init), so it can be run as an ablation flag on the looped transformer or the sculpting FM, not a rewrite.
