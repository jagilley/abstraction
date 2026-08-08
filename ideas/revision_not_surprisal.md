# Revision, not surprisal: the conditioning gap, the aleatoric null, and why the residual needs an oracle

**Status**: Conceptual. Nothing built. The measurement it implies is specified at
[`rhm/conditional_revision/SPEC.md`](../experiments/rhm/conditional_revision/SPEC.md).
**Date**: 2026-08-07
**Supersedes**: [the_forecast_needs_a_lead.md](the_forecast_needs_a_lead.md) (retained, marked
outdated). That doc's §4 is the surviving object; its §1–§3 mechanism and its §5 are revised here.
**Builds on**: [self_model_needs_a_loop.md](self_model_needs_a_loop.md),
[efference_copy_cancellation.md](efference_copy_cancellation.md),
[heterogeneous_graders.md](heterogeneous_graders.md) §4/§4b,
[activation_to_activation_forward.md](activation_to_activation_forward.md)
**Evidence re-read**: [`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/README.md)
(Gate 0 + two nulls) · [`a2a_forward/OOD_ROBUSTNESS_README.md`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md)
· [`rhm/residual_decomposition/`](../experiments/rhm/residual_decomposition/README.md) (β) ·
[`mjc/arity_torque/`](../experiments/mjc/arity_torque/README.md) ·
[`mjc/ballistic/`](../experiments/mjc/ballistic/README.md) 4b ·
[`a2a_forward/reaching/ACTIVE_VISION_README.md`](../experiments/a2a_forward/reaching/ACTIVE_VISION_README.md)
· §8 re-reads the local-loss arc: [`MNIST_LOCAL_LOSS`](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md),
[`GATED_RATCHET`](../experiments/a2a_forward/GATED_RATCHET_README.md),
[`RHM_FM_REGULARIZER`](../experiments/rhm/RHM_FM_REGULARIZER_README.md), against
[local_prediction_error_learning.md](local_prediction_error_learning.md)
**Reading**: Ruffini, Lopez-Sola, … Friston (2025), *Cross-Frequency Coupling as a Neural Substrate
for Prediction Error Evaluation*[^private]
— §2.3 and §4.2 only.
**Attribution**: the depth≈time observation and the original challenge are Jasper's (see the parent
doc). In this round Jasper supplied the three challenges that produced most of the revision: *"is
this the same as what we called a physics state FM — which have we used for ballistic control?"*,
*"isn't the prediction always ahead and therefore conditioned on strictly less?"*, and *"does that
mean we're overloading the residual with an extra independent variable?"*, plus the CFC paper. The
conditioning-gap reframe, the compression/revision decomposition, the aleatoric-null claim, the
backprop correction, and the BP-oracle design came out of the exchange. §8 was prompted by Jasper's
question of whether this explains the local-loss nulls, and whether the cerebello-thalamic injection
was serving a Friston-esque role the port dropped.

---

## One-liner

**Felt surprise is how much a word changes your model, not how improbable it was.** Token surprisal
bundles *reducible* surprise (the word told me something about the world's latent structure) with
*irreducible* surprise (the word was one of several equivalent realizations of a structure I already
had). NTP teaches from the sum and therefore keeps paying gradient on the irreducible part forever —
it lacks an **aleatoric null**. The separating quantity is the **belief revision** the token induces,
and on RHM it is *exactly computable*, so this stops being an argument and becomes a measurement.

## Vocabulary

- **Conditioning gap** — what the predictor's conditioning set omits that causally determines the
  target. Replaces "lead". The gap's *content type* — world information vs. the predictor's own
  omitted context — is what sets the residual's meaning.
- **Revision** — how much a token moved a forecast of one's own next state, holding the forecaster
  fixed. Directional (a vector), not a magnitude.
- **Compression term** — the forecaster's shortfall given everything, including the token. Permanent
  and desirable; see §6.
- **Aleatoric null** — a learner correctly not updating on *irreducible* surprise. Distinct from the
  trivial null (already-predicted input gives ~0 loss and ~0 gradient), which NTP does have.

## 1. The variable is the conditioning gap, not depth vs. time

The parent doc's information-set argument is correct — a forecaster holding all of the target's
information cannot be mistaken, only short — but it names the wrong variable and inverts a word.

**Wrong variable.** `post_block0[t] → post_block6[t+1]` is depth-wise *and* has a lead. Arity-2 in a
deterministic recurrent net is temporal and has *none* (the target is exactly determined). Depth is
neither necessary nor sufficient. What determines the residual's meaning is only ever *what sits in
the conditioning gap*: omit world information not yet arrived and the residual is world-facing; omit
internal context the predictor could have had and it is a compression residual.

**Inverted word.** The parent doc uses *epistemic* = "wrong about the world" and *computational* =
"lacked capacity." Standard usage is the reverse — epistemic uncertainty **is** the reducible,
capacity-limited kind. This matters because it broke the doc's headline retro-explanation:
`OOD_ROBUSTNESS` Experiment A tested *competence calibration*, which under the doc's own taxonomy is
exactly what a capacity-shortfall residual should carry. So §1 predicted the depth FM should be
**good** at that task; the null is mildly against the framing, not explained by it.

**The correct reading of that null is already in the repo, and it is stronger.** The depth residual
measures the FM's failure to compress M — a property of the **(M, FM) pair**, not of M's relation to
the world. Gate 0 measured this directly: `corr(rres, nll) = −0.336`, and the two are *anti*-localised
against the known hierarchy. That is the `MNIST_LOCAL_LOSS` lesson one axis over — **legibility is not
self-knowledge** (LL directly optimizes FM agreement to cos 0.997 and has SK R² ≈ −0.03). Four results
support this; the information-set argument needs none of them.

## 2. Timing governs *use*; conditioning governs *content*

The natural objection: in wetware the forecast always arrives ahead of the signal, so it is
*necessarily* conditioned on less. Two corrections.

**The implication doesn't run backwards, and it doesn't fix content.** Arriving ahead is sufficient
for conditioning on less but not necessary — just don't feed the forecaster the token. And arriving
ahead says nothing about *what kind* of thing is missing: commit early inside a single forward pass
and all you lack is your own downstream computation.

**And the cerebellar forecast is not conditioned on strictly less.** The efference copy hands it the
motor command — information the afferent signal has not yet reflected. Its information set is
**incomparable** to the observation's, not smaller: it has the command, it lacks the world's noise
realization. That is arity-2. **Biology already runs the two-forecast structure**, which is support
for §3's decomposition rather than for the lead as mechanism.

So: **timing is an architectural constraint on causal *use*** — a forecast that gets cancelled,
planned on, or gated by must exist before the thing it is about is computed, which is exactly what
`mjc/ballistic` 4b cashes out (FM quality transmits at slope **+1.07** ballistic vs **+0.35**
reactive) — **and conditioning is what fixes the residual's *content***. The parent doc merges them.

## 3. The decomposition

With `p⁻ = FM₁(s_t)` (forecast of my own next state *before* the token) and `p⁺ = FM₂(s_t, x_{t+1})`
(*after*), the arity-1 residual splits as an identity:

```
s_{t+1} − p⁻   =   (s_{t+1} − p⁺)   +   (p⁺ − p⁻)
arity-1 residual    COMPRESSION          REVISION
```

This dissolves the parent doc's §4 tension. Arity-2 strips exogenous surprise; a lead supplies it;
they "pull opposite ways" only if you must pick one. You don't — you take the **difference of two
forecasts that differ by exactly one token of conditioning**, which is a controllable variable and
already our idiom (`arity_torque`: *"identical data — only the input differs"*).

**In belief space the arity-1 forecast collapses to the identity.** If `b_t` is a posterior, the
tower property gives `E[b_{t+1} | x_{≤t}] = b_t` — beliefs are a martingale. So the Bayes-optimal
`p⁻` *is* `b_t`, and revision becomes `b_{t+1} − b_t`: the token's **Bayesian surprise**, computed on
the model's own belief. Two consequences: the object has a name and a literature, and any measured
departure of a learned `p⁻` from `b_t` is itself a readout of how miscalibrated the model's belief
trajectory is. (This holds for beliefs, not for arbitrary activations — `E[h_{t+1}] ≠ h_t` in a
residual stream. It is a reason to work in belief space, not a free lunch.)

## 4. What revision is *for*: the aleatoric null

Not "what ideas the word provoked." The defensible claim:

> Where the only uncertainty is irreducible — RHM's `m` synonymous production rules — the true latent
> posterior does not move, so revision ≈ 0 while `nll` is maximal. Where a token disambiguates a
> higher-level feature, revision is large while `nll` may be small.

And on RHM this is **exact, not analogical**. For latents `z` truncated at abstraction level ℓ:

```
E[ B_ℓ ]  =  H(x_{t+1} | x_{≤t})  −  H(x_{t+1} | z_{≤ℓ}, x_{≤t})
             total surprisal        irreducible-given-structure
```

Expected Bayesian surprise about structure = surprisal minus the entropy of the realization given
that structure. Both terms are computable by belief propagation on the known parse tree. So the
aleatoric/epistemic split is **an identity indexed by the DGP's own abstraction ladder**, not a hoped-for
correlation. That ladder is the same one every RHM readout in this repo already uses.

This is what NTP lacks. The parent doc's "NTP has no null" is wrong — a predicted token gives ~0 loss
and ~0 gradient. What NTP lacks is the *aleatoric* null: it pays gradient on irreducible surprise
forever. `endogenous_teacher` measured the consequence without naming it — **ordering by token
surprisal was the worst assignment of all** (`nll` erased `uniform`'s val gain entirely), because on
RHM the high-`nll` positions are the irreducible ones. That is the noisy TV arriving through the
teaching signal.

## 5. The null is two-factor and two-timescale, not local plasticity

The parent doc's §5 claims global backprop forbids a null because "a global scalar objective injects
gradient into every parameter regardless of local activity." **This is false.** `∂L/∂W = δ ⊗ a`: zero
incoming activation gives zero gradient however global the loss. Backprop is already gated on the
forward side. What actually blocks a null in a transformer is the **skip connection**, which is why
`CANCELLATION_README` found *directional* cancellation (cos +0.89 → +0.095) and explicitly not a
magnitude null.

The standard answer, and the better one, is **precision**: the update is `Π · (I − P)`, and the null
comes from `Π → 0`, not from the error vanishing. Ruffini et al. locate the two factors on two
timescales (fast SEC for content, slow EEC for precision) for exactly our reason —

> *"the estimation of precision requires the pooling of instantaneous prediction errors over time in
> the same way that a statistician would pool prediction errors over multiple observations to
> estimate the standard error."*

You cannot tell reducible from irreducible from one sample. **The aleatoric null is definitionally a
slow-timescale quantity**, which is why §5 was looking in the wrong place — it is not a property of
the wiring or the plasticity rule. This is implementable under backprop as a per-position weight, and
it retro-explains `endogenous_teacher`'s sign: that cut weighted by the *instantaneous residual
magnitude* (`|E|`, i.e. the noisy TV) where the theory calls for `Π` (its opposite).

**The trap.** Precision is by construction a scalar, and we have measured scalar gating to be the weak
lever three times — `EMOTION_INJECTION` (*"'this will be hard' doesn't tell the model **what** to
compute differently — only **how much**"*; cerebellar gate 3.0 vs emotion gate 0.15), the
directional-vs-scalar belief node (vector probe ΔR² +0.18 vs scalar +0.03), and `endogenous_teacher`
itself. Ruffini allows a precision *operator* in one parenthetical and never uses it. That empty cell
is the only version worth building.

## 6. Compression never leaves, so it needs an oracle rather than removal

The decomposition introduces a second forecaster and therefore a second capacity term: with
`p⁻ = E[s_{t+1}|s_t] + ε₁` and `p⁺ = E[s_{t+1}|s_t,x_{t+1}] + ε₂`, we measure **revision + (ε₂ − ε₁)**.
Differencing already helps — matched, correlated forecasters cancel most of it — and the residue is
checkable by the `residual_decomposition` discipline (β was invariant to ±0.01 across a 4× FM-capacity
range while the level moved 1.8×): sweep FM capacity, and the revision estimate must be flat.

But compression should not be designed away. **The cerebellum represents vastly less than the cortex
it forecasts; compression is the organ's function, not our small-FM artifact.** A forecaster with the
forecastee's capacity is not a forecaster, it is a copy. So the move is to obtain a **compression-free
oracle for the same quantity** and calibrate against it:

| | quantity | compression |
|---|---|---|
| `B_t` | exact belief revision from BP on the known parse tree | **zero by construction** |
| `Δ_t` | the model's own revision | present |
| `B_t − Δ_t` | how much of the true revision the model registered | **the compression term, measured** |

The contaminant becomes a second readout — a DGP-legibility measure on the axis the RHM belief-depth
line already runs. This is the `mjc/expansion` discipline (calibrate → measure → calibrate), which is
the one place in this repo a rank-shaped instrument survived contact with a null.

## 7. What we have and have not built (by codomain, not by axis)

| | predicts | axis | codomain | has a world-facing gap? |
|---|---|---|---|---|
| a2a depth FM | `post_block0[t] → post_block6[t]` | depth | own activations | **no** |
| `mjc/*` — the **bare-state physics FM** | `f(s_t,u_t) → Δs_t` | time | **the world's state** | **yes** |
| `reaching/ACTIVE_VISION`, `REACHING_INTERNAL` | `FM(s_t,u_t) → Δ_t` on the looped ViT | time | own activations | partially |
| `RHM_SCULPTING` 3b | belief rolled one step | time | own latent | partially |

Two things fall out. First, **`mjc/` is the one place in this program where a forward model has a
genuine world-facing gap**, and the parent doc does not notice it — the target is produced by MuJoCo,
a process the FM is not and does not contain. Second, that charge comes from the **exogenous
codomain**, not from commitment: the physics FM has both, and the parent doc's §2 credits the wrong
one. Ballistic control is evidence the *forecast's accuracy* is load-bearing, not that the *residual*
is world-facing.

Transposed to reading, only the gap comes along: `h[t+1] = g(h_t, x_{t+1})` is still produced by the
model's own weights applied to an exogenous input. The reading-axis residual therefore sits **between**
depth (fully endogenous, zero world content) and physics (fully exogenous target) — a genuine mixture,
which is precisely why it cannot be read raw and must be decomposed.

`ACTIVE_VISION` also supplies the build constraint: predicting `s_{t+1}` scored **0.94 by echoing
carried state**; you must predict the *update*. Belief space makes this structural rather than a
patch — §3's martingale property says the echo *is* the correct arity-1 forecast, so the update is
what remains.

## 8. Why the local-loss / predictive-coding port came back null

[`local_prediction_error_learning.md`](local_prediction_error_learning.md) proposed using the FM's
error as a *training* signal rather than a forward signal —
`L = L_NTP + λ‖sg(FM(h₀)) − h_L‖²` — for three payoffs: dense high-dimensional supervision per
example, depth-localized credit assignment, and automatic routine/novel partitioning, the last
precision-weighted as `L_local = Σ_d r_d²/σ_d²` and cited there as *"the predictive coding formulation
(Rao–Ballard / Friston)"*. The biological anchor was the cerebello-thalamo-cortical loop, with the
learning gate mapped onto reticular thalamic gating of the teaching signal.

**The mechanism this doc supplies**: `h₀` and `h_L` are both deterministic functions of the same input
through the model's own weights, and the FM is stop-gradded. The only degree of freedom is to move
`h_L` into the image of a low-capacity FM. So —

> **With an endogenous target and identical information sets, "minimize prediction error" is not
> "model the world better." It is definitionally "be simpler."** Rao–Ballard's error falls when your
> hypothesis explains the data better; ours falls when your computation becomes more compressible.
> Same equation, opposite content, and the difference is entirely what sits in the conditioning gap.

Four separately-recorded results are one fact:

| result | where | reading |
|---|---|---|
| *"functional simplification needs no self-knowledge apparatus at all"* | [`RHM_FM_REGULARIZER`](../experiments/rhm/RHM_FM_REGULARIZER_README.md) | the degeneracy stated as a finding |
| LL reaches FM cos **0.997** with SK R² ≈ **−0.03** | [`MNIST_LOCAL_LOSS`](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md) | the parent idea doc's 2026-06-15 revision saw the shape (*"eliminates the residual and therefore eliminates self-knowledge... nothing to precision-weight because nothing is surprising"*) but read it as a **balance** problem to fix with a λ curriculum. It is not tunable — it is forced by the conditioning. |
| **13.4×** brittleness, *"geometric (compressing ~117/128 dimensions...)"* | same | low-rank collapse is what a simplicity objective produces |
| the gate **opens** 0.31 → 0.77 instead of closing — *"inner = outer, so more compression is always beneficial"* | [`GATED_RATCHET`](../experiments/a2a_forward/GATED_RATCHET_README.md) | the gate is correctly measuring a degenerate objective |

(`RHM_LATENT_LOOP`'s `λ_local`, ΔSK −0.06 → −1.54, is the same shape on the second substrate.)

**Why the precision half failed specifically.** The 2026-06-16 revision recorded the reason as *"the
Friston framing doesn't apply here: precision weighting distinguishes signal from noise, but FM errors
aren't noise — they're structured capacity limits."* Correct, and §4 says why it **had** to be:
precision is the ratio of aleatoric to epistemic variance, and with identical information sets the
residual is **100% capacity shortfall by construction**. Precision is therefore uninformative
*everywhere*, not merely in the directions measured. **Friston's machinery was applied to a signal
engineered to contain zero of the thing precision weighs.** This closes a dependency chain with §5:
*no working precision term without an aleatoric component, and no aleatoric component without an
exogenous conditioning gap.* Both halves of the local-loss program failed for the same upstream
reason.

**What biology was actually supplying.** Two versions of one thing.

- **Rao–Ballard does not escape via exogenous targets at every level** — level L predicts level L−1's
  *activity*, endogenous too. It escapes because the stack is **jointly settling** against exogenous
  drive: level L−1 is pushed bottom-up by data the top-down hypothesis does not yet explain, so its
  error is *"the part of the drive I fail to account for."* The world stays in the loop at every level
  **through the recurrent dynamics**. Feedforward, `h_L` is fully determined by `h₀` — there is no
  independent bottom-up drive to fail to explain. **We ported the topology and dropped the dynamics
  that made the topology meaningful.**
- **The cerebellar case is the same, through time.** The cerebrocerebellum's forecast of cortical
  dynamics has an endogenous codomain, but the cortex it forecasts is a nonstationary,
  externally-driven system — the forecast is of a state co-determined by afferent input arriving
  *during the delay*. Our feedforward port froze the world by construction: one input, one pass, no
  interval during which anything arrives. This is the parent doc's *"the port kept the compression and
  silently dropped the anticipation"* with a mechanism instead of an assertion.

**The architecture was never the problem.** The `mjc` bare-state physics FM is entirely conventional
and is behaviorally load-bearing at slope **+1.07** (§7). Don't replace the FM — **replace what it
predicts.**

**What this does not explain**, and should not be collected as a null:

- **LL's learning-speed win is real and orthogonal** — +2.5pp over OL at step 500, 31% lower final val
  loss, from 6400-d supervision against `[CLS]`'s 128-d. A simplicity regularizer with dense
  supervision can genuinely speed optimization. That payoff stands.
- **The bilevel learning gate's R² = 0.83** (the highest self-knowledge ever measured here) was a **CL**
  condition — injection present, so a residual exists. The degeneracy above is about LL-alone.
- Whether an exogenous-gap local loss works at all is **untested**.

**The constructive consequence, and the degeneracy it swaps in.** A non-degenerate local loss needs a
target conditioned on something the predictor lacks: `h_ℓ[t] → h_ℓ[t+1]`, co-determined by `x_{t+1}`.
That is this doc's revision object used as a *training signal* rather than a measurement — so the
[SPEC](../experiments/rhm/conditional_revision/SPEC.md)'s gated follow-up is also the fix for the
local loss. But it trades one degeneracy for another: minimizing `‖FM(h[t]) − h[t+1]‖²` collapses by
making `h[t+1]` **insensitive to `x_{t+1}`** — ignore the input. That is data2vec's known failure mode,
and we already built anti-collapse machinery for it ([`rhm_sculpt_data2vec.py`](../experiments/rhm/rhm_sculpt_data2vec.py)).

> **An endogenous target makes the objective a simplicity regularizer. An exogenous conditioning gap
> makes it a predictability regularizer whose degenerate solution is input-invariance. Neither is
> epistemic on its own. What kills both is grounding — a target the model does not control.**

Which re-derives [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md) §5 —
*grounded evaluative grader expands, endogenous "be-predictable" caps, "grounding is the pivot"* —
from the conditioning structure rather than from observation. The payoff of this section is a
**mechanism** for a regularity we already had empirically, plus a prediction about which failure mode
replaces which.

## What would kill this

- **Revision turns out monotone in token surprisal under the constructed contrast** (nll held fixed by
  design, not merely regressed out). Then this document is bookkeeping. This is the primary kill and
  it is Gate B of the spec.
- **Cheap early version of the same kill.** The minimal instantiation of §1 is a one-line change to the
  existing FM — shift the target by one position instead of six blocks, `FM(h₆[≤t]) → h₆[t+1] − h₆[t]`,
  frozen main model, open-loop. `endogenous_teacher` Gate 0 measured the *depth* residual at
  `corr(rres, nll) = −0.336`, anti-localised against the hierarchy. The temporal residual should flip
  positive and align. **Two-sided kill**: `corr ≈ 0` means the axis change did nothing; `R² > 0.9`
  means the residual is surprisal re-expressed in state space. Half a day, no new machinery, and it
  gates everything else — [SPEC](../experiments/rhm/conditional_revision/SPEC.md) Gate 0.
- **The revision estimate moves with FM capacity.** Then we are measuring `ε₂ − ε₁`, not revision.
- **The model's belief revision is orthogonal to the oracle's** given `nll`. Then no forecast of that
  belief can carry the signal, and the substrate is wrong before any FM is trained.
- **A precision-weighted teaching signal beats `uniform` only as much as any scalar reweighting does.**
  Then the aleatoric null is real as a measurement and inert as a lever — which is still worth knowing,
  and is the fourth scalar-gating null.
- **§8 specifically**: a local loss with an exogenous conditioning gap (`h_ℓ[t] → h_ℓ[t+1]`, LL-alone)
  reproduces the *same* signature as the depth version — FM cos → ~1, SK → 0, brittleness of the same
  order. Then the conditioning gap is not the operative variable and §8's degeneracy reading is wrong;
  the simplicity collapse would be a property of any auxiliary "be predictable" term. This is a cheap,
  direct test and it should be run before the teaching intervention.

**Standing prior against.** Four independent replications say endogenous targets cap — `LL` (FM cos
0.997, SK ≈ 0), `λ_local` (ΔSK −0.06 → −1.54), `data2vec` (+23% vs grounded MLM's +44%), `fm_cotrain`
(*"grounding is the pivot"*). None was conditional-revision, so the cap does not transfer directly.
But note what the exception looks like: `full_loop` §5's expansion came from a **grounded** evaluative
grader (external, precomputed DP-best-move), and LP-alone was separately the *weak* tap (18% vs 84%).
The oracle `B_t` here is grounded in the same sense. If revision only works when read against the
oracle and not from the model's own estimate, that is the same finding again, and we should say so
rather than discover it afterward.

## Connections

- **[heterogeneous_graders](heterogeneous_graders.md) §4/§4b** — revision is a functional of the dense
  signal, so it is **not** a counterexample to "no signal is both dense and evaluative"; it has the
  same status §10 already assigns LP. The honest claim is that it is a better-formed *epistemic*
  allocator than surprisal, not a second grader.
- **[efference_copy_cancellation](efference_copy_cancellation.md)** — the CFC paper supplies a third
  answer to the separability problem we had not considered: **multiplex**. Predictions ride slow alpha,
  errors ride the fast gamma envelope; they sum on one wire and stay separable because they occupy
  different bands. Our own "Refinement (honest)" — that cancellation *orthogonalizes* rather than
  nulls — may be the biologically correct mechanism rather than a shortfall.
- **[`residual_decomposition`](../experiments/rhm/residual_decomposition/README.md)** — `res_var(i) ∝
  act_var(i)^β`, β<1, loud directions absorbed at ρ≈0.95 and quiet at 0.32. Ruffini's comparator
  operates on the *coarse-grained* envelope by design. We recorded graded absorption as a fact; that
  is what a comparator should look like. (Also: **β is unusable below ~100 directions** and RHM's
  `R_res_participation` is 7 — do not build a readout on β at this scale.)
- **[operators_not_footprints](../beliefs/trees/operators_not_footprints.md)** — a conditional forward
  model compresses the transition operator `g(s, x)`; the revision term is the part of that operator
  the exogenous input controls.

## Open questions

- Does `B` decompose cleanly by *level*, or does the coarse-graining choice (which `ℓ` truncates `z`)
  dominate the answer? The identity in §4 is exact for each ℓ; whether the ordering across signals is
  stable in ℓ is unknown.
- Is there a **precision operator** — a directional, low-rank `Π` — that expresses "I am unreliable
  *in these directions*"? This is the only cell in §5 that is not already a measured null.
- Does the martingale property (§3) hold empirically in a trained model's belief trajectory, and is
  the departure from it a useful calibration readout in its own right?
- Does any of this require the forecast to exist *causally* before the token is processed, or only to
  be *conditioned* differently? Measurement needs only conditioning (§2); anything that gates or
  cancels needs the timing, and that is a real architectural fork we have not chosen.
- If revision works only against the grounded oracle and not from the model's own estimate, is that a
  ceiling or a curriculum — i.e. can the oracle be withdrawn after training, as `full_loop`'s
  internalization ladder asks?
- **From §8**: is there a formulation in which the two degeneracies — collapse-toward-simplicity
  (endogenous target) and collapse-toward-input-invariance (exogenous gap) — trade off continuously, so
  that some intermediate conditioning is optimal? Or does grounding dominate both, making the whole
  axis a distraction? The second answer is the one `full_loop` §5 currently supports.
- **Also from §8**: LL's learning-speed win (+2.5pp at step 500, 31% lower val loss) survives the
  degeneracy reading intact and is unexplained by it. Is dense auxiliary supervision simply a good
  optimizer-level trick that has nothing to do with self-models — and if so, is it the *only* thing the
  local-loss arc actually bought?

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
