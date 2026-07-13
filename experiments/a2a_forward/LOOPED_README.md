# Looped Transformer with Forward-Model Injection

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md)
**Parent experiment**: [README.md](README.md) (the feedforward a2a arc)
**Status**: In progress. The headline is single-seed; the baseline-battery control and the representational imprint probes (matched-subspace / position-specificity / cancellation controls) are **run and analyzed** (2026-07-09). A near-manifold self-map extrapolation test (runnable-simulator probe) is also run — **negative on MNIST** (no forecast-specific extrapolation; consistent with the strong "self-execution" capability being a scale-plus-domain phenomenon). Discriminators 2–3 partially probed; a self-referential-domain port pending.
**Date**: 2026-07-09

## Goal

Test the central claim of [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md): a *separable, causally-used* self-forecast (a self-**model**, not merely absorbed legibility) is ill-posed in a feedforward net and becomes **forced** in a **weight-shared looped** model with forward-model (FM) injection at each step — because weight-sharing makes the same operator `g` both the *producer* of the state the FM reads and the *consumer* of the FM's forecast, and convergence forces the resting state to agree with a forecast of itself.

Concretely: build a looped ViT (`h_{t+1} = g(h_t + p + gate·FM(h_t))`), where the FM predicts a future iterate, and test whether the injected self-forecast becomes **causally necessary** — in a way the feedforward closed-loop model's absorbed map is not.

## Architecture (`looped_vit.py`)

`LoopedViT` — a weight-shared, recurrent-depth ViT with the same I/O contract as `vit.py`:

- **Shared operator `G`**: `n_loop_layers` pre-norm ViT blocks applied `T` times with weight sharing (default 1 block × T=8). This is the theory's reusable `g`.
- **Input re-injection** (Universal-Transformer / DEQ style): `s_0 = 0`, `s_{t+1} = G(s_t + p + inject_t)`, `p = prelude(embed(x))`. Re-injecting the input each step makes the fixed point input-dependent, and no per-step timestep embedding is used (that would make `g_t ≠ g` and break the "one operator to model" premise).
- **Optional non-shared bookends** `prelude` / `coda` (the "looparound"): with them, the FM setup is a near-exact analog of the feedforward a2a — FM reads a mid-stack recurrent state (≈ `post_block0`), predicts a future state (≈ the `post_block` target), injects into the recurrent stream (≈ inject-after-block1), and coda+head separate readout (≈ block3→head). **These turned out to be a confound (see trajectory) and default to 0 in the confound-free runs.**
- **Knobs added over the trajectory**: `readout_all_steps` (per-step logits for deep supervision), `step_inject_fn(t, operand, state)` hook (carries the gated FM forecast; exposes the raw state so the caller can inject the *update* `FM−s`), `predict_k` (FM predicts `s_{t+k}`), `injection_form` (`next_state` vs `update`), `gate_type` (`proj` vs bounded `scalar`), `damping` (`s += α·(G−s)` for convergence without deep supervision).

## The trajectory — what we tried and what each step taught

This experiment moved through a sequence of failures that were each individually informative. The short version: **the loop's self-forecast is inert or destabilizing under the obvious/naive settings, and only becomes a clean, load-bearing self-referential channel once four separate confounds are removed.**

| # | Setup | Result | Lesson |
|---|---|---|---|
| 1 | **Plain looped ViT**, no injection (`mnist_looped.py`) | Matches 4-layer FF at ¼ params (0.21M); but the timed-loop **overshoots** — acc peaks at train-T (97.3%) and *degrades* past it (94.7% @T=20). | The plain loop learns a *timed trajectory*, **not a fixed point** (rel-Δ still 17% at T=8). No stable resting state → nothing for the fixed-point argument to bite on. |
| 2 | **+ Deep supervision** (readout loss at every step) | Overshoot **gone** — loss flat to 4 decimals T=15→20, a genuine attractor. | Deep supervision manufactures a fixed point. *But it forces every iterate toward the answer (high-abstraction supervision) — later identified as a confound that makes the task easier than it should be.* |
| 3 | **Naive injection** `gate·FM(a_t)`, FM predicts next state (2×2, w/ prelude/coda + deep-sup) | **Unstable.** Trained fine until the gate opened past ~2.5, then the loop diverged (cl_last acc 0.94→0.56, gate ran 0.08→3.7). FM cosine *fell* 0.999→0.95 as it destabilized. | Injecting a full next-state-magnitude forecast **compounds** over 8 steps. The gate feels the same greedy upward pressure as feedforward (0.08→3.0) but the loop can't absorb it. *(This also explains the "0.97 cosine" — it was instability, not a bottleneck.)* |
| 4 | **Update injection** `gate·(FM(a_t) − s_t)` | **Stable but inert.** cl_last 0.972; injection benefit ≈ 0 (−0.015); no-inj ≈ inj; convergence/self-consistency identical to OL. | The update form vanishes at the fixed point (`FM(s*)≈s*`), so it's stable — but with a 1-step-ahead target it carries nothing the loop doesn't already have. Self-consistency achieved *trivially*. |
| 5 | **Longer horizon** `k=3` (predict `s_{t+3}`) | **Still inert.** fm_cos still 0.999; benefit −0.007. | The one-block loop's k-step trajectory is *still* trivially predictable. Not a horizon problem. |
| 6 | **FM-capacity sweep** (`mnist_looped_fm_sweep.py`) | Even a **1%-capacity FM (5.4K) predicts the loop's *update* at cos 0.92**; a full-size FM at 0.996. No FM in the family gives cos 0.5–0.7. | The loop's per-step transformation is **intrinsically low-complexity** (residual eff-rank ~20/128). But 0.92 is the same operating point where feedforward's 1% FM *was* useful — so "FM too accurate" isn't the whole story. |
| 7 | **Confound diagnosis** | The pure phase-1 loop *did* need its depth (T=1 68% → T=8 97%). Adding prelude/coda (a 3-block FF path) + deep-sup made MNIST look feedforward-solvable. | **Two of our own confounds** — the prelude/coda feedforward shortcut and deep supervision — were masking the loop's role. And **MNIST classification barely needs iterative computation** (the answer, not just the supervision, must require the loop). |
| 8 | **Fashion-MNIST** (harder perception, same DGP type) | Loop far more load-bearing: T=1 **40%** → T=8 85% (vs MNIST 68%→97%). | A harder task makes the loop's iterations genuinely necessary — a better substrate for testing whether the self-forecast matters. |
| 9 | **Confound-free** (pure loop, no deep-sup, small FM, Fashion), **unbounded gate** | Injection **helped a lot early** (−0.69 nats, acc 0.53 @ gate 0.43) — *then* the unbounded gate ran away (0.43→1.9) and collapsed the model to chance. | The injection **is** useful when the loop is load-bearing — but the projection gate is an **over-relaxation coefficient** and, unbounded, extrapolates past the stable region. Signal present; training unstable. |
| 10 | **+ Bounded scalar gate** (mixing coeff ∈ [0,1]) | **Stable and strongly causally necessary** — see headline. | Bounding the over-relaxation coefficient fixes the instability while keeping the confound-free pure loop. |

## Headline result (confound-free, bounded gate — single seed)

Pure loop (prelude=coda=0), no deep supervision, small 5.4K FM (update-cos ≈ 0.92), `predict_k=3`, update-form injection, bounded scalar gate ∈ [0,1]:

| task | OL acc @T=8 | CL **with inj** | CL **inj ablated** | dependency (Δacc) | inj benefit (Δloss) | final gate |
|---|---|---|---|---|---|---|
| **MNIST** (loop ~idle) | 0.973 | 0.964 | 0.925 | 0.039 | −0.16 | 0.015 |
| **Fashion** (loop load-bearing) | 0.852 | 0.847 | **0.667** | **0.180** | **−0.86** | 0.016 |

**The injected self-forecast is a real, load-bearing channel, and its load scales with how load-bearing the loop is.** Ablating the injection from the Fashion CL model (same weights, gate zeroed) collapses accuracy 0.847 → 0.667 (+0.86 nats). On MNIST, where the loop is nearly idle, ablation costs only 0.04. The dependency is **~5× larger on the task that actually needs the loop** — the opposite of the inert behavior every confounded setup showed.

### The self-regulated gate (the striking part)

The gate tells a clean story across the trajectory:

- **Unbounded projection gate** → runs away (0.08→3.7) → collapse. No self-regulation.
- **Bounded scalar gate** → **self-regulates to a small, stable value (~0.016) and holds it** (it even drifts *down* 0.018→0.015 during training, actively staying out of the over-relaxation regime) — while the model becomes **strongly dependent** on that tiny signal. A ~2% anticipatory injection, present at every step, becomes load-bearing. The model settles on a small self-chosen dose of its own forecast and organizes its computation around it.

## Interpretation

**No net loss benefit is expected, not a disappointment.** CL-with-injection ≈ OL in accuracy (0.847 vs 0.852 on Fashion). We have no reason to expect the loop's self-forecast to *lower task loss* — the injection genuinely adds a new stream of information the model must parse *in addition to* the task, so equal or even slightly-worse loss is the natural outcome. This connects directly to the idea doc's ["injection as a new self-referential input modality" / efference-copy](../../ideas/self_model_needs_a_loop.md) section: the dependency is not a bug but a **slot the model allocates for an expected input modality** it has learned to interpret. The right question is not "did loss go down" but "is the self-forecast a separable, causally-used channel" — and the ablation says yes.

**Why the confounds mattered.** Each confound independently made the self-forecast look inert: prelude/coda gave a feedforward path around the loop; deep supervision forced intermediate states to already *be* the answer; an over-capacity FM made the forecast redundant; MNIST barely needs the loop at all. Only with all four removed — pure loop, last-step supervision, bottlenecked FM, load-bearing task — does the causal necessity appear. This is itself a finding: **the loop's self-forecast is causally used precisely when (and to the degree that) the loop's iterative computation is load-bearing.**

## Caveats

1. **Ablation alone is weak evidence.** *Any* signal a model trains with becomes load-bearing when removed. The non-trivial part is the **scaling with loop-necessity** (Fashion 5× MNIST) — but that could partly reflect task difficulty (Fashion's higher baseline loss leaves more headroom for any auxiliary signal). **The baseline battery is the decisive control.**

   **Resolved (2026-07-09) — see the "Baseline battery" and "Representational imprint probes" sections below.** Bottom line: random_proj (a frozen random-init FM) creates *equal-or-greater absolute* dependency than the genuine forecast, so ablation alone certifies nothing. But only forward's dependency *scales* with loop-necessity (3.6–5.4× MNIST→Fashion vs 1.5× random, 0.5× shifted), and only forecast-shaped channels leave a representational imprint concentrated in their own directions (3.3–4.1× vs random_proj's channel-avoidant 0.4×). This supports a **veridical, channel-specific self-map** — richer than the feedforward absorbed map, short of a runnable self-simulator — not the strong "self-model" claim outright.
2. **Discriminators 2 & 3 partially probed.** Control 2 (position-specific future decodability) is a near-manifold self-counterfactual analog and *does* separate forward from shifted; Control 3 tests the fixed-point / efference-copy mechanism (finds summation, not cancellation). A *behavioral* near-manifold counterfactual (2) and a full fixed-point self-consistency *structure* analysis (3) remain.

## Baseline battery — the dependency is not forecast-specific, but its *scaling* is (2026-07-09)

**Code**: `mnist_looped_injection.py` (`--baseline-type random_proj|shifted`). The confound-free bounded-gate setup with three injected channels, compared against OL, same seed/init/data:

- **forward** — `gate·(FM(s_t)−s_t)`, FM trained to forecast `s_{t+k}` (the real self-forecast).
- **random_proj** — same form but the predictor is a *frozen random-init* FM (a consistent, structured, but untrained channel — not a forecast).
- **shifted** — the forward forecast rolled by half the positions (forecast-shaped, but position-misaligned).

Dependency = accuracy drop when the injection is ablated (gate zeroed) at eval, measured as `val_acc − val_acc_no_inj` at the final step; scaling = Fashion ÷ MNIST (loop-idle → loop-load-bearing):

| channel | MNIST dep (Δacc) | Fashion dep (Δacc) | **scaling** | Fashion acc (w/ inj) | inj benefit (Δloss, Fashion) |
|---|---|---|---|---|---|
| forward | 0.050 | 0.180 | **3.6×** (5.4× in loss) | 0.852 | −0.86 |
| random_proj | 0.148 | 0.225 | 1.5× | **0.878** | −0.66 |
| shifted | 0.069 | 0.037 | **0.5× (inverted)** | 0.853 | −0.15 |

1. **Absolute causal necessity is NOT forecast-specific.** random_proj creates equal-or-greater absolute dependency than the genuine forecast on both tasks. Any consistent, position-aligned self-referential channel becomes load-bearing once removed — ablation alone cannot certify a self-model (caveat 1 was right).
2. **Only forward's dependency scales with loop-necessity.** forward's dependency grows 3.6× (5.4× in loss) from loop-idle MNIST to loop-load-bearing Fashion; random_proj barely scales (1.5×) and shifted *inverts* (0.5× — less dependent on the harder task). The forecast-specific signature is the **interaction (channel × loop-necessity)**, not a main effect. This is the surviving distinctive behavioral result.
3. **Forecast content does not buy task performance.** random_proj is the best-performing condition on both datasets (Fashion 0.878 > OL 0.869 > forward 0.852). Consistent with the modality/slot framing: the injection adds an input to parse, not free task help.
4. **Robustness inverts in the loop.** On Fashion, forward's error-correction is *negative* at every ε (−0.05 at ε=1.0, −0.18 at ε=2.0): in operating mode the injection *amplifies* perturbation damage. The feedforward arc's headline robustness win (flatter landscape, distribution-invariant) does **not** carry over — the loop recirculates a corrupted forecast every step, a positive feedback on error the one-shot feedforward injection structurally couldn't have.

## Representational imprint probes — what each channel leaves in the weights (2026-07-09)

**Code**: `mnist_looped_probes.py`. Eval-only on the four checkpoints. Motivation (the user's reframe): *any* consistently-injected channel should leave a representational imprint, so random_proj is not a failed condition — it **defines the generic imprint**, and the self-knowledge signature is whatever `forward` imprints *beyond* it. Because all conditions share init + data order + seed, the channel-off divergence from OL, `D_t = s_t^{ablated} − s_t^{OL}`, is attributable to the channel (controlled-retrain design). All probes are linear.

**Probe 2 — future self-decodability is degenerate (confirms the doc's warning).** `R²(s_t → s_{t+k})` is ~0.94–0.98 for *every* condition including OL, with live ≈ ablated. The loop's low-rank, slowly-varying states make the future trivially linearly accessible everywhere. Decodability is a dead metric here — exactly the "feedforward degeneracy" [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md) predicts. Self-knowledge must be read from causal role and imprint *structure*, not decodability.

**Probe 3 + Control 1 — representations organize around forecast-shaped channels only.** Fraction of the weight-imprint `D` living in the channel's own top-10 output subspace `U`, vs a random 10-dim subspace of the task manifold (top-40 future PCs). Ranges are MNIST–Fashion:

| channel | imprint in own `U` | in random task-10-d | **U/rand** | `U`↔future align (chance ≈ 0.28) |
|---|---|---|---|---|
| forward | 0.72–0.83 | ~0.22 | **3.3–3.8×** | 0.66–0.74 |
| shifted | 0.83–0.84 | ~0.21 | **3.9–4.1×** | 0.69–0.75 |
| random_proj | 0.08–0.09 | ~0.21 | **0.4×** | 0.27–0.29 (= chance) |

Forecast-shaped channels (forward, shifted) concentrate their imprint in their *specific* directions 3.3–4.1× more than generic task reorganization predicts; random_proj is *below* chance (0.4×) — it reorganizes **away** from its own channel. So the natural intuition "any consistent injection leaves representations organized around it" is **false as stated: consistency isn't enough — forecast-shape is what earns the concentration.** The random channel's output points in random directions (U↔future = 0.28 = the √(10/128) chance baseline; the metric is calibrated), so the model routes around it; a forecast's output is task/future-aligned (0.66–0.75), so the model organizes within it. Control 1 kills the "both are just task-aligned" confound: the concentration is channel-*specific*, not generic-task.

**Control 2 — position-specificity separates forward from shifted (representationally).** Decode the *aligned* future vs the *position-rolled* future (rolled by the same half-window that defines the shifted channel), on Fashion:

| channel | aligned R² | rolled R² | aligned advantage |
|---|---|---|---|
| OL | 0.947 | 0.386 | 0.562 |
| forward | 0.949 | 0.433 | 0.516 |
| random_proj | 0.955 | 0.420 | 0.536 |
| **shifted** | 0.955 | **0.553** | **0.402** |

shifted makes the position-scrambled future anomalously decodable (0.553 vs ~0.39–0.43) — its self-map encodes the cross-position structure of the channel it consumed. **The imprint carries the specific position-structure of its channel**, and forward's map is *veridical* (points at the real future, position-locked like OL) while shifted's is a *faithful map of a wrong future*. Both build a self-map; only forward's matches reality. (Clean only on Fashion — on MNIST forward's imprint is extremely low-rank (eff-rank 4.9, top-1 PC 40%), so rolled decodability is inflated for everyone; the loop-load-bearing task is the right regime.)

**Control 3 — no cancellation; the loop summates/amplifies.** Per-step injection "gain" (‖Δnext-state‖ / ‖injection‖) is ~4–5× with `cos(effect, injection)` ≈ **+0.79 to +0.89** for all channels — the next state moves *in the direction of* the injection, amplified. **No corollary-discharge cancellation**, and forward is not absorbed more than the others. A clean null for the efference-copy *cancellation* prediction, coherent with the robustness inversion above: the loop uses its self-forecast **additively/expansively**, not subtractively.

### What the probes change about the map-vs-model picture

The map-vs-model distinction from the idea doc now **localizes across four separable layers** — richer than the doc's binary:

1. **The slot (map), forecast-generic** — forecast-shaped channels get representations concentrated in their directions (Control 1); random channels don't. Shared by forward and shifted. This is the doc's "new self-referential modality / slot," now directly demonstrated *and* shown to be forecast-specific.
2. **Channel-identity, forecast-specific** — the imprint encodes the *position-structure of the specific forecast consumed* (Control 2); forward's map is veridical, shifted's faithfully maps a wrong dynamics.
3. **Usefulness (the model proper), correctness-specific and behavioral** — forward beats shifted only in *causal load-bearing* (the baseline battery's 3.6–5.4× dependency scaling), not in imprint magnitude.
4. **Mechanism — additive, not cancelling** (Control 3).

**Net revision to the doc's intuitions:**

- **Confirmed strongly**: the self-referential *modality / slot* framing, and the *decodability-is-degenerate* warning. The dependency really is "a slot allocated for an expected input modality."
- **Sharpened**: map-vs-model is not binary. The loop produces something richer than the feedforward "absorbed map" — a **separable, channel-specific, veridical self-map** of the model's own future dynamics. Real progress toward a self-model, but still a self-*map* (a static internal picture), not yet a self-*simulator* runnable counterfactually off to the side. That gap is now the sharp next target.
- **Adjusted**: the **fixed-point self-consistency** mechanism is only half-right. There is a faint signature — the aligned forecast is the one that nearly vanishes at rest (inj rel-norm 0.012–0.017 vs shifted's 0.026) — but the loop never settles (≈15% from a fixed point) and the causal action lives in the *transient*. The forecast behaves more like a *trajectory the model organizes around* than a *fixed point it settles into agreement with*.
- **Challenged**: the **efference-copy cancellation** prediction is not borne out — this architecture summates and amplifies the forecast rather than cancelling it (Control 3), and correspondingly amplifies rather than damps perturbations. The biology analogy holds at "distinct modality with its own slot" but breaks at "…that gets subtracted out."

## Near-manifold self-map extrapolation — is the self-map a runnable *simulator*? (2026-07-09)

**Code**: `mnist_looped_extrapolation.py`. Motivation: the strongest sense of self-knowledge in the idea doc is "execute a rough forward pass on yourself, off to the side" — a *runnable self-simulator*, not just a static self-*map*. Framing (from discussion): split "self-execution" into **(i) acceleration/halting** (use a rough self-forecast to reach your answer with less compute — the bounded gate is literally an over-relaxation coefficient, so a primitive form is already present) and **(ii) counterfactual query** ("what would I output if this input were slightly different?", run off to the side without committing — the strong sense). This test is the cheapest MNIST proxy for (ii): does the model's self-forecast stay valid on **near-manifold inputs it never trained on** (a simulator *extrapolates*; a static lookup *collapses*)?

Four looped conditions (forward / OL / random_proj / shifted), all sharing init/seed/data with the FM co-trained in every one (FM-as-approximator held constant). Perturbations unseen in training: **rotation** (near-manifold) and **gaussian pixel noise** (graded off-manifold stress). Two metrics per condition × level:
- **A (representational)** — off-manifold self-consistency `cos(FM(a'_t), s'_{t+k})` on the intrinsic (channel-off) perturbed run; retention = value(ε)/value(clean). A runnable simulator degrades slowly; a lookup collapses.
- **B (behavioral)** — from the bare embedded input `a'_0 = prelude(embed(x'))`, FM → readout → label; agreement with the model's *actual* final output on x' ("can I predict my own output from a rough forward pass?").

**Result: negative for the runnable-simulator hypothesis on MNIST — structurally, not noisily.**

1. **Metric A is flat and non-discriminative.** Self-consistency retention stays 0.98–1.05 out to 45° / σ=0.75 for *every* condition, and forward is **not** better than OL (if anything slightly worse: MNIST σ=0.75 retention 0.978 vs OL 1.003, random_proj 1.050). The loop's per-step transformation is intrinsically low-rank (a 1% FM already predicts the update at cos 0.92), so self-consistency is a generic property of the dynamics that holds on *and* off manifold for all conditions — there is nothing for "simulator quality" to vary along. This is the clean, unconfounded part: **no extra off-manifold self-forecastability that forward earned over OL.**
2. **Metric B is confounded by output-collapse and shows no forward advantage.** On-manifold agreement is weak for all (~0.15–0.27, near the majority-class rate) — the rough-forward-pass-from-raw-input is too lossy at this scale. Under heavy noise the metric inverts degenerately: OL/random/shifted agreement *rises* (retention 2.8–5.5×) because both the FM shortcut and the actual output collapse to the same majority class (agreement by mutual collapse), while forward's still-differentiated live output diverges from the collapsed shortcut. The only faint positive — Fashion under small rotations, injection-trained models retain agreement better than OL (rot_15 retention forward 1.18 / random 1.09 vs OL 0.56) — is the familiar *any-injection* effect, **not forecast-specific**.

**Reading.** The cheapest test that could have revealed a runnable self-simulator didn't, and it failed for a *structural* reason: MNIST's loop is so low-rank that self-consistency is trivially maintained by everyone, so a simulator (if one existed) would be invisible — and MNIST classification applies no pressure to build one (cheap real computation, no early-commitment payoff, no continuation to rough out). The negative is consistent with **"MNIST cannot show the runnable-simulator (ii) capability," not with "it doesn't exist."** Net boundary: the looped model has a veridical, channel-specific self-*map* (imprint probes) but shows **no sign of a runnable self-*simulator*** on MNIST — the map→simulator gap is real and MNIST is where it taps out. Priors on the strong hypothesis are essentially unmoved; the question needs a domain where the loop's computation is expensive and self-reference is meaningful (RHM / looped language). Both the user and the analysis went in expecting (ii) to be a scale-plus-domain phenomenon; this result is consistent with that and does not disconfirm it.

## Next steps

1. ~~**Baseline battery on Fashion**~~ *Done (2026-07-09)* — see the two sections above. The dependency is not forecast-specific; its *scaling* (behavioral) and the *imprint concentration* (representational) are.
2. **Cross-decodability matrix** — decode the aligned future in the shifted model and the shifted future in the forward model, to sharpen "veridical vs faithful-map-of-a-wrong-future" into a clean 2×2.
3. ~~**Chase the cancellation signature.**~~ *Done — see [CANCELLATION_README.md](CANCELLATION_README.md).* Rather than hope corollary-discharge *emerges* from summation training, we **built it in** as a one-flag strict generalization (`inject_mode=cancel`: `s_{t+1}=G(s_t−inj)+inj`). It produces the cancellation signature (net effect on the next state flips from `cos(Δ,inj)=+0.89` to `+0.095`) and **flips the robustness inversion** (perturbation error_correction −0.18 → +0.05) — and the cancellation is *topological, not learned* (every operator responds to a raw `+inj` identically). Confirms the summation/amplification here was the additive *wiring*, not an intrinsic property of the loop.
4. **Domain port for the runnable-simulator (ii) question (the priority for the strong hypothesis).** The MNIST self-map extrapolation test came back negative *because MNIST cannot pressure a simulator*, not because none can exist (see section above). Move to a substrate where the loop's computation is expensive and self-reference is meaningful: RHM (recursive DGP, existing latent-loop machinery, and the idea doc's point that non-MNIST DGPs need a latent target to synthesize a deep moving frontier) as the controlled bridge, then a looped-language port. Optional first: salvage a collapse-controlled Metric B on MNIST/Fashion to make the negative airtight (expected to stay negative, since the unconfounded Metric A is structurally flat).
5. **Discriminator 2 (behavioral near-manifold counterfactual)** and **damping vs bounded-gate convergence** — whether the FM injection alone (bounded) provides the endogenous convergence that deep supervision faked (the "scrap deep-sup once self-knowledge suffices to halt" thread).

## Reproduction

```bash
cd experiments/

# 1. Plain looped ViT (phase 1): overshoot; + deep supervision -> attractor
modal run --detach a2a_forward/mnist_looped.py::a2a_mnist_looped
modal run --detach a2a_forward/mnist_looped.py::a2a_mnist_looped \
  --prelude-layers 1 --coda-layers 1 --deep-sup --deep-sup-frac 0.5 --no-train-ff-baseline

# 2. Injection 2x2 (per-condition container; run all four in parallel).
#    injection-form: update (stable) | next_state (naive, unstable)
for c in ol_last ol_deep cl_last cl_deep; do
  modal run --detach a2a_forward/mnist_looped_injection.py::train_condition \
    --condition $c --injection-form update --predict-k 3 & done; wait
modal run a2a_forward/mnist_looped_injection.py::aggregate --injection-form update --predict-k 3

# 3. FM-capacity sweep (state-cos vs update-cos vs FM size), confound-free pure loop
modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep --dataset mnist
modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep --dataset fashion_mnist

# 4. HEADLINE: confound-free + bounded gate, MNIST vs Fashion
for d in mnist fashion_mnist; do for c in ol_last cl_last; do
  modal run --detach a2a_forward/mnist_looped_injection.py::train_condition \
    --condition $c --dataset $d --prelude-layers 0 --coda-layers 0 \
    --fwd-d-head 1 --fwd-mlp-mult 0.125 --predict-k 3 --injection-form update \
    --gate-type scalar & done; done; wait

# 4b. Baseline battery: random_proj + shifted channels (Fashion + MNIST)
for d in mnist fashion_mnist; do for b in random_proj shifted; do
  modal run --detach a2a_forward/mnist_looped_injection.py::train_condition \
    --condition cl_last --dataset $d --baseline-type $b \
    --prelude-layers 0 --coda-layers 0 --fwd-d-head 1 --fwd-mlp-mult 0.125 \
    --predict-k 3 --injection-form update --gate-type scalar & done; done; wait

# 5. Representational imprint probes (eval-only; loads all 4 conditions per dataset,
#    runs Probes 1-3 + Controls 1 matched-subspace / 2 position-specificity / 3 cancellation)
modal run --detach a2a_forward/mnist_looped_probes.py::probe_battery --dataset mnist
modal run --detach a2a_forward/mnist_looped_probes.py::probe_battery --dataset fashion_mnist

# 6. Near-manifold self-map extrapolation (runnable-simulator probe; eval-only)
modal run --detach a2a_forward/mnist_looped_extrapolation.py::extrapolation --dataset mnist
modal run --detach a2a_forward/mnist_looped_extrapolation.py::extrapolation --dataset fashion_mnist
```

## Files

- `looped_vit.py` — `LoopedViT` (weight-shared recurrent-depth ViT; injection hook, prelude/coda, deep-sup readout, damping-compatible).
- `mnist_looped.py` — phase 1: plain loop, convergence dynamics, acc-vs-T sweep, deep-supervision, FF baseline.
- `mnist_looped_injection.py` — phase 2: the injection 2×2 (`train_condition`) + `aggregate`. Supports injection form, `predict_k`, dataset (mnist/fashion), damping, gate type, FM capacity.
- `mnist_looped_fm_sweep.py` — FM-capacity sweep on a frozen confound-free loop (state-cos vs update-cos vs size; loop-necessity).
- `mnist_looped_probes.py` — representational imprint probes (eval-only): future self-decodability, divergence imprint (`D = s^{ablated} − s^{OL}`), channel-subspace alignment, + Control 1 (matched-subspace null), Control 2 (position-specificity: aligned vs rolled-future decodability), Control 3 (injection absorption / cancellation). Reconstructs the frozen `random_proj` predictor by seed (not saved).
- `mnist_looped_extrapolation.py` — near-manifold self-map extrapolation / runnable-simulator probe (eval-only): off-manifold self-consistency retention (Metric A) and predict-your-own-output agreement (Metric B) under unseen rotation / noise perturbations, all four looped conditions. Negative on MNIST (see section).
- `forward_model.py::BoundedScalarGate` — bounded over-relaxation gate ∈ [0, max_gate].

Modal volume: `/data/a2a_forward/mnist_looped*/`; probe JSON at `/data/a2a_forward/mnist_looped_injection/probes/{mnist,fashion_mnist}_{imprint_probes,extrapolation}.json`.
