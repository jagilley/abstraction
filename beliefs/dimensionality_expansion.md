# Continual Learning as Dimensionality Expansion

**Date:** 2026-07-03 · *extended 2026-07-25 with §"What grades an expansion?"; decomposition corrected 2026-07-27; scoped 2026-07-27*
**Status:** Core belief intact and **still untested**; its **scope now confirmed by measurement** on a motor plant ([`mjc/expansion/`](../experiments/mjc/expansion/README.md), 2026-07-27: drift ≠ expansion, ±0.08 directions against a +0.72 calibration); its arithmetic operationalization **falsified and replaced** (see §"The decomposition, corrected"), and its domain of applicability now **bounded** (see §"Scope"). Measured across three domains in [`rhm/residual_decomposition/`](../experiments/rhm/residual_decomposition/README.md).
**Related:**
- [Self-prediction and self-knowledge](./trees/self_prediction_and_self_knowledge.md)
- [Cerebellum and cognitive architecture](./trees/cerebellum_and_cognitive_architecture.md)
- [Metacognitive novelty learning](./metacognitive_novelty_learning.md)
- [Heterogeneous graders](../ideas/heterogeneous_graders.md) — supplies the grader this file's expansion drive left unspecified
- [A2A forward model](../experiments/a2a_forward/README.md)

## The belief

The objective of continual learning is not to minimize error on any task. It is to **grow the number of directions the base weights can coherently represent** (`R_act`), so that any individual task occupies a progressively smaller, more sharply characterized subspace of a larger whole. Novelty is sought because it is the only unbounded source of new representable directions. A persistent, nonzero self-model residual is the *signature of health*, not a failure to converge — the goal was never a residual of zero.

This reframes the recurring villain of the a2a_forward ratchet work — the absorbing state / narrow FM-capacity sweet spot on a fixed dataset — as the predicted consequence of running a compressor with no expansion drive.

## The decomposition, corrected (2026-07-27)

> **Resolved.** The partition claimed here was measured directly and **falsified**, across three domains and 40+ conditions. Full writeup: [`rhm/residual_decomposition/README.md`](../experiments/rhm/residual_decomposition/README.md). **The belief's core — continual learning as growing representable directions, a persistent frontier as the signature of health — was never tested and is untouched.** What died is this arithmetic operationalization of it.

Fix a layer, run the data distribution, and split each activation `A` by what the self-model (forward model) compresses:

```
A  =  FM(earlier_layer)  +  Residual
```

The old claim was that this partitions activation space — `R_act ≈ R_comp + R_res`, with `R_res` "the slice of `R_act` the self-model hasn't absorbed yet". It does not, for three measured reasons:

1. **`R_comp` is a redundant readout of `R_act`.** 72.7 vs 72.7 on RHM, 207.0 vs 207.0 on language, 47.8 vs 47.8 on MNIST. A good self-model reproduces the activations it predicts, so `rank(FM output) → rank(A)` by construction. There is no independent "absorbed directions" counter, so there is no conservation for the flows to run along.
2. **`R_res` > `R_act` in every condition** (120.4 vs 72.7). A slice cannot exceed what it slices.
3. **`R_res` is blind to magnitude.** Relative residual 0.0509 → 0.0030 (17×) while `R_res` goes 116.0 → **120.4** — it *rises* as the residual vanishes.

The subadditivity argument was the error: `rank(X+Y) ≤ rank(X)+rank(Y)` is a theorem about **hard** rank, which the entropy effective rank used everywhere in this repo does not obey. (Hard rank is degenerate here regardless: N ≫ D makes all three exactly D.)

### What is actually there: a graded shadow, not a partition

The FM does not absorb some directions and miss others. It absorbs **every** direction partially, in a proportion that follows a power law:

> # **res_var(i) ∝ act_var(i)^β**
>
> across the main model's principal directions `i` — **R² = 0.95–0.99**, in RHM, language, and vision alike.

Since β < 1, the leftover shrinks more slowly than the signal, so the unabsorbed fraction goes as `act_var^(β−1)`: the model's loudest directions are absorbed nearly perfectly (ρ ≈ 0.95), its quietest barely at all (ρ ≈ 0.32). A photocopier reproducing bold lines and losing faint pencil marks. **Naive residual rank was counting the faint pencil marks.** There is no boundary anywhere, so there is nothing to partition — but the frontier is real, and it is graded: not a *set of directions* but a *depth*, how far down the model's own priority list the self-model has reached.

### The instrument that replaces the triple

Two numbers, not three, and they move independently:

| | what it is | behaviour |
|---|---|---|
| **β** | the **shape** — how absorption falls off across the model's directions | invariant to a 4× FM-capacity range (±0.01); a property of *the model's computation* |
| **`R_res_participation`** | the **frontier's dimensionality** — how many of the model's own working directions still carry unexplained computation | counted in the model's own basis, weighted by the computation actually done there |

Plus **frontier mass** (fraction of activation variance unexplained) for size. `R_res_participation` reads **7** on RHM where naive rank reads 84, because the naive count was dominated by directions the model barely uses. That gap *is* the failure of the old instrument.

### The historical negatives this explains

Rank had failed as an instrument in this repo three times, for one reason now measured — each was reading a **saturated FM's noise floor**, where the FM drives relative residual to ~0, β collapses to 0.11–0.17, and rank inflates to 94–96% of `d_model`:

- **"Language's residual is full-rank 200/256 yet the FM captures it to cosine 0.97."** Measured at a 1-block gap. On language, naive `R_res` reads **244.3 in the pure-noise regime and 244.7 in the genuinely structured regime** — it cannot tell them apart at all. That figure was a noise floor and is retired.
- **[`mjc` cut #1](../experiments/mjc/contact_residual/README.md): contact eff-rank 3.60 > free 2.52**, which retired "residual rank ∝ DGP complexity". Removing structured error and leaving diffuse error mechanically *raises* rank. The negative was the instrument, not the hypothesis.
- **[`RESIDUAL_RANK_README`](../experiments/rhm/RESIDUAL_RANK_README.md) Exp. 3: a 17× residual-norm range leaves rank flat at 90–96%.** Entropy rank reads shape, not size.

### What survives

The intuition that the residual maps the model's own structure is **confirmed, at 5–6× chance**: at a wide prediction gap, residual variance in the model's top-16 directions is 0.66–0.80 against a chance baseline of 0.125. The residual does live in the directions the base weights use. It just does so by *grading* them, not by *partitioning* them.

## The flows: measured, and they do not happen

The claim was that compression drains `R_res → R_comp` with `R_act` unchanged, running to `R_res → 0` on fixed data, while novelty refills `R_res` from the top. A 2×2 factorial on RHM — `{open-loop, wake-sleep} × {fixed DGP, new rules per cycle}`, 6 cycles, measured each cycle on a **fixed held-out probe** so `R_act` can only move if the model's computation moved:

- **Wake-sleep ≈ open-loop on every metric** (ΔR_act +52.7 vs +51.9; ΔR_res −5.9 vs −4.1). The compression loop has no measurable effect on the triple.
- **`R_act` rises in all arms including open-loop** (58.2 → 110.1). Its growth is ordinary continued training — not compression, not novelty, and not the gated local loss.
- **`R_res` does not drain toward 0** — flat at 135–143 over 6 cycles. The absorbing collapse does not reproduce.

**The novelty arm was confounded and settles nothing.** New rule sets at matched (v,s,L,m) share no structure with the old, so the model fully relearns and fully forgets (val loss on cycle-0 rules 1.41 → 4.51 while val on current rules holds at ~1.44). That is task switching, not novelty. **Whether novelty expands representable directions remains untested** — it needs an intervention with shared structure across cycles.

## Two routes to expansion — one bounded, one not

- **Internal (self-legibility reorganization):** the gated local loss spreads existing computation across more, more-orthogonal directions without new data — but is **capped**, since you can only re-express fixed content so far.
- **External (novelty):** new data adds genuinely new content. **Unbounded.** This is the channel the human intuition is about.

⚠️ The a2a evidence once cited for the internal channel saturating — WS_LG's residual rank growing 27.8 → 30.2 before hitting activation-norm inflation — is a **rank number measured with the falsified instrument**, and the direction of a rank change is exactly what that instrument gets wrong. The *behavioural* observation (the ratchet stops compounding by 16 cycles; activation norms inflate) stands on its own and is what the claim now rests on. The rank mechanism is withdrawn pending re-measurement with β and `R_res_participation`.

## Scope: which domains should expand at all (2026-07-27)

This file has read as universal. It should not. **Expansion needs a domain with hierarchical structure to expand *into*** — where new data carries genuinely new compositional content, so there is more to represent than before. RHM and language are such domains by construction. **A motor plant with fixed degrees of freedom is not.** An arm is a fixed map from commands to state; a drift moves the target function without enlarging the space of functions worth representing, so a well-fit FM should show a **flat frontier**, and on this belief's own terms that is *health*, not a failed compressor.

The biology agrees and is the reason to take it seriously: real animals do not visibly expand their motor representations and remain extremely capable. A body never grows a limb. The conditions under which a motor system *must* expand are the ones that change its degrees of freedom — **development, injury, and tool use** — and tool use is the sharp one, since a tool extends the body schema.

Two consequences:

1. **Drift ≠ expansion.** Drift moves the *target function*; expansion grows the *space of representable functions*. A one-parameter walk (friction slowly changing) does the first and structurally cannot do the second. They converge **only under support-growing drift** — which is the useful form of the "drifting dynamics is isomorphic to expansion" intuition, and the boundary that makes the distinction operational rather than semantic.
2. **The motor-domain test must be a support-growing intervention** — *DOF accretion* (joints unlocking) is what was run; *region accretion* and tool acquisition remain untried. Writeup: [`mjc/expansion/`](../experiments/mjc/expansion/README.md).

> ### ✅ Run, and it came out as predicted (2026-07-27) — [`mjc/expansion/`](../experiments/mjc/expansion/README.md)
>
> A fixed-DOF n=5 arm under 8 rounds of perpetual support-fixed drift — one parameter, and separately four localized regions — holds the frontier **flat**: `R_res_participation` moves **±0.08 directions**, de-confounded by a 2×2 that separates *how many parameters drift* from *how hard the operator is*. The **same plant under DOF accretion opens it by +0.72 ± 0.42** (3/3 seeds), so the drift null is ~10× below a calibrated real effect and the third possibility — the instrument not reading on control substrates at all — is excluded. **Drift ≠ expansion is now a measurement with a unit attached, not an argument.**
>
> Two things this buys beyond the scoping claim. First, read through [heterogeneous_graders](../ideas/heterogeneous_graders.md) §4b it is an existence proof that **continuous, non-stationary, genuinely-difficult prediction pressure produces no expansion** — exactly what "expansion needs an evaluative grader, because opening a direction makes prediction worse before it makes it better" predicts. Second, it retires the [`physical_control_substrate.md`](../ideas/physical_control_substrate.md) motivation that drifting dynamics is a *novelty generator*; drift remains essential for **realism**, just not for representational growth.
>
> **The instrument caveat this run adds, and it is load-bearing for this file.** **β is unusable at low state dimension**: on a 10-dim motor state it fails its own capacity-invariance control by **6–59×** (against ±0.01 on rhm's domains), with fit R² 0.26–0.51 against 0.95–0.99, because it is a log-log slope over 10 points. `R_res_participation` works but is the noisiest readout there, and its *direction* is domain-dependent — a better model reads **higher** on the arm (absorbing the loud directions flattens what remains, pushing the count above `R_act`) and **lower** on RHM (7 against `R_act` ≈ 73). **The count is a shape descriptor, not a quality metric.** Both point the same way: **the shape half of this belief needs a high-dimensional domain.**

**Instrument note carried over.** FM overcapacity does not bias the readout — `R_res_participation` weights by computation actually done, so idle directions contribute ~nothing (the 7-vs-84 gap on RHM is precisely this working). But **β** is a power-law fit across *participating* directions, so a low-dimensional prediction target leaves ~6 points to fit rather than ~200. Frontier mass and `R_res_participation` survive that; β does not. On control substrates this argues for predicting a **learned latent** rather than simulator state — which is independently the right move, since predicting `qpos/qvel` is privileged access to the DGP's own coordinates that no embodied learner has.

## Supporting evidence (from a2a_forward)

- ⚠️ *Withdrawn:* "naive wake-sleep on fixed data collapses residual rank (23.9 → 13.5); the gated ratchet grows it (27.8 → 30.2)" — [GATED_RATCHET_README](../experiments/a2a_forward/GATED_RATCHET_README.md). Both numbers come from the falsified instrument, and the RHM 2×2 above finds no such drain when the triple is measured properly. The behavioural phenomena those runs also recorded (compounding stops; norms inflate) are unaffected.
- The FM-capacity sweet spot is narrow on a fixed dataset, and the authors independently conclude "a continual learning setting would sidestep this by providing novel data that sustains compression pressure" — [README §extended ratchet](../experiments/a2a_forward/README.md).
- The gate becomes input-selective (opens on novel, closes on routine) only under distribution shift; on a stationary set it opens uniformly — [OOD_GATE_README](../experiments/a2a_forward/OOD_GATE_README.md).
- Closed-loop self-knowledge is *computational and distribution-invariant, not epistemic* — the residual encodes what kind of computation is un-compressed, consistent with it being a frontier map — [OOD_ROBUSTNESS_README](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md).

## What grades an expansion? (2026-07-25)

This file specifies the two flows and never says **what decides which novelty** — "novelty raises `R_act`" leaves the selector unnamed. The [heterogeneous-graders frame](../ideas/heterogeneous_graders.md) supplies it, and the answer is that compression and expansion are gradeable by *different kinds of signal*, necessarily.

**The activation-energy argument.** Copernicus threw away degrees of freedom and got *worse* predictions than Ptolemy; on a description-length metric at the moment of the rotation, epicycles win ([contra-long-horizon-benchmarks](../ideas/contra-long-horizon-benchmarks.md) §I–II). So compression is the **endpoint, not the process** — you often must expand first, tolerate a worse fit, and cross a barrier to reach the basis in which the compression is available at all. A learner that monotonically descends a compression objective cannot cross that barrier by construction, which is the same wall this file already measures from the inside (`R_res → 0` on fixed data).

| | graded by | availability | payoff |
|---|---|---|---|
| **compression** (`R_res → R_comp`) | prediction error | dense, every step, free | immediate |
| **expansion** (`R_act ↑`) | *cannot be prediction error* | sparse, slow | **deferred, initially negative** |

Opening a direction makes prediction worse before it makes it better, so no dense predictive signal can drive it; its grader must tolerate deferred and initially-negative payoff. That is what "evaluative" means, and the underlying impossibility is that **no single signal is both dense and evaluative** — dense requires being free (self-supervised on what happened), evaluative requires referencing outcomes you would rather not sample ([cerebellum tree](trees/cerebellum_and_cognitive_architecture.md#no-single-learning-signal-can-be-both-dense-and-evaluative--so-the-two-teacher-structure-is-derived-not-designed)). **Compression/expansion is that dichotomy viewed from the representation side rather than the signal side.**

**This says what the selector is: learning progress is the expansion grader.**
> ⚠️ **Challenged 2026-07-29, in the direction the frame itself predicted.** With a *working* (floor-corrected) reducibility estimator, LP-alone recovers only **18%** of the uniform→oracle allocation prize on [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md)'s ladder, against **84%** for an outcome-trained relevance tap — a 4.7× asymmetry the "no visited-but-irreducible cell" caveat does *not* explain (that caveat explains only why the *product* ties relevance alone). [heterogeneous_graders](../ideas/heterogeneous_graders.md) §10 had already flagged why: LP *"is a functional of the dense signal and is evaluative only about epistemics, not about the world's rewards."* The dense/evaluative frame is vindicated; **this specific nomination is not.** See [meta_learning_under_metered_data.md](../ideas/meta_learning_under_metered_data.md) §8. Caveat: the ladder grades *allocation*, not expansion — LP has never been tested as an expansion grader directly. LP is ~0 when mastered (dark room), ~0 when irreducible (noisy TV), and peaks at moderate-and-*falling* error — a band-pass shape recorded in [two_timescale_value_loop](../ideas/two_timescale_value_loop.md) as an empirical property of a curiosity drive and validated in [curiosity Phase 1](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md). Read through this frame the shape is a **specification**, not a curiosity: it is what a grader of "is this direction worth opening?" has to look like. The experiment proposed below (residual-guided data selection) is therefore not one option among many — it is the expansion grader, wired.

**Two consequences worth recording.**

1. **The LLM reading.** Weight decay, width, depth, data mix, when to stop — every one is an expansion decision graded on deferred payoff. LLM training does not *lack* an expansion loop; **it has one, implemented in humans**, running on a wall-clock of weeks. That is the mechanism behind LLMs expanding only slowly, coarsely, and exogenously: the expansion grader was outsourced to the researcher. It also predicts that an intrinsic expansion grader is worth more than a better compressor.
2. **The abandoned flagship.** *Dimensionality expansion under drifting dynamics* was [`physical_control_substrate.md`](../ideas/physical_control_substrate.md) cut #5, billed in its own text as *"the single biggest open question of the whole program,"* and never run — the drift machinery built to make it free was consumed by the value-loop program instead ([`mjc/HISTORY.md`](../experiments/mjc/HISTORY.md) §Planned and never run). Two independent routes have now arrived back at it. **Verified unrun 2026-07-25**: no rank or spectral measurement exists anywhere in `mjc/` except cut #1's, and nothing was started-and-abandoned in git history.

   **The instrument caution this file needs to carry — and which it under-called.** Cut #1 ran a participation ratio on the 8-dim *state residual* of one FM on a *stationary* pusher, got a negative (contact eff-rank 3.60 > free 2.52), and retired *"residual rank ∝ DGP complexity"* ([contact_residual](../experiments/mjc/contact_residual/README.md)); a2a independently found `rank ⊥ noise` — language's residual is full-rank 200/256 yet the FM captures it to cosine 0.97. So **rank had already failed as an instrument here twice**, and this file's triple inherited that prior. The defense written here on 2026-07-25 — *those were absolute and static, the triple is used differentially against a matched control* — **did not hold**, because the triple's components are not independent of each other (see §"The decomposition, corrected"); a matched control does not rescue that. **All three failures now have one measured cause** — each was reading a saturated FM's noise floor — and a replacement instrument (β, `R_res_participation`) that does respond to magnitude. The generalized lesson, which is the part worth keeping: **before designing around a rank-shaped instrument, check that its components are independent and that it responds to magnitude, not only to shape.** Both design constraints derived here are unaffected — the drift must have **growing support** (a one-parameter walk demands re-fitting, not new directions) and capacity must actually **bind** (2-link arms leave it slack; n ≥ 5). Current status of the cut, including a gap-width calibration that now gates it: [`mjc/README.md`](../experiments/mjc/README.md) §Next steps #3.

   The `R_res` type ambiguity raised in [RHM_LATENT_LOOP](../experiments/rhm/RHM_LATENT_LOOP_README.md) turns out to have been the early warning, and it was under-read: the response recorded here was "report all three together," which cannot work when two of the three are redundant. The corrected version of that response is to report **shape and level separately** — β and frontier mass — which the capacity sweep shows move independently.

## The experiment this predicts (self-model curiosity)

Standard curiosity rewards *world*-model prediction error. The cerebellar residual is prediction error about the model's *own computation* — a curiosity signal computed for free. Make it active: **let the FM residual select what to train on next** (high residual = frontier = where new representable dimensions live).

- **Primary readout:** **β** and **`R_res_participation`** across cycles, measured on a fixed held-out probe — *not* the (R_act, R_comp, R_res) triple, which does not decompose. Two prerequisites, both learned the hard way: keep the prediction gap wide enough that the FM does not **saturate** (a saturated FM yields β ≈ 0.13 and near-full rank in every domain, i.e. pure noise), and match the FM's architecture to the block it predicts — head count *and* attention mode.
- **Prediction:** residual-guided data selection sustains a nonzero frontier where uniform selection lets it collapse. State it in frontier mass and `R_res_participation`; `R_act` alone will not discriminate, since it rises under ordinary training in every condition.
- **Contrast condition:** dedicated-practice ratchet on a fixed target vs. the same compute spent on residual-selected novelty — tests whether "lateral" beats "direct". The novelty arm must share structure across cycles (same rules, deeper level; partial rule swap), because a full rule swap produces catastrophic forgetting and measures nothing.

## Caveat

R_act is an *activation* rank, hence data-dependent, not a pure weight property — it mixes "base-weight capacity" with "how much this data excites it." The measurement fix is a **fixed held-out probe** held constant across cycles and conditions, so R_act can only move if the model's computation moved; the RHM 2×2 uses this and it is what revealed that R_act rises identically in open-loop. A weight-native measure (Jacobian rank, or count of active dictionary features) would isolate base capacity more cleanly still.

Two further cautions from the re-measurement, both general to rank-shaped instruments:

- **Check magnitude-sensitivity and component independence before designing around one.** This is the generalized lesson from three failures; a matched control does not rescue a readout whose components are not independent.
- **`R_act`/`d_model` is a confound in any cross-model comparison.** Relative naive `R_res` falls with model width (RHM 75.3% → 46.3%), which looks like the residual becoming lower-rank — but measured against the model's *own* active dimensionality it **rises** (1.82 → 2.18). The denominator moved. Meanwhile frontier mass is scale-invariant when the FM is held at a fixed fraction of a block: **the frontier's level is set by the capacity ratio, not absolute capacity.**
