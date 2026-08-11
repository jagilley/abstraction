# The bridge carries a performance error: a baseline-subtracted, agency-gated forward-model error is what the value system reads

**Status**: Proposal, informed by a literature pass (2026-08-11). Nothing new run. The afferent direction it depends on **already has a positive result in this repo** ([`drift_value_loop`](../experiments/mjc/drift_value_loop/README.md) Cut 3) and a working biological instance with the right control (Gadagkar et al. 2016). What is new is the *specific form* of the signal and the experiment that separates it from a sensory response.
**Date**: 2026-08-11
**Builds on**: [two_timescale_value_loop.md](two_timescale_value_loop.md) (the FM↔value interface, named there as the open bridge), [efference_copy_cancellation.md](efference_copy_cancellation.md), [the_forecast_needs_a_lead.md](the_forecast_needs_a_lead.md) (arity), [heterogeneous_graders.md](heterogeneous_graders.md) (the blind-grader argument), [physical_control_substrate.md](physical_control_substrate.md)
**Supersedes**: an earlier draft of this idea (`cancellation_is_the_credit_gate.md`, same session) which claimed the FM *masks* self-caused content out of value's view. That sign was inverted — see §3. Nothing was run against it.

## One-liner

The cerebellum→value bridge does not carry a reward and does not carry a mask. It carries a **performance error**: the
forward model's own prediction error, **subtracted from a running benchmark of itself**, and **gated on the existence of an
efference copy**. Positive when you did better than your recent self. Self-supervised — no extrinsic reward appears
anywhere in it.

```
        a_t ──────────────┐
         │                │  efference copy
         ▼                ▼
     [ world ]        [ FM₂(s,a) ] ──┐
         │                │          │
         │            [ FM₁(s) ] ────┤ g = ‖p₂ − p₁‖   (agency: what my action explains)
         ▼                           │
      s_{t+1} ─── e = ‖s_{t+1}−p₂‖ ──┤
                        │            │
                   b ← EWMA(e)       │
                        │            │
                        ▼            ▼
                  δ = (b − e) · σ(g/θ)        ← the bridge signal
                        │
                        └──► gain on FM learning rate + policy/allocation
```

## 1. The claim

Three properties, each taken from a specific result, each fixing something this program already observed.

```
e_t     = ‖s_{t+1} − FM₂(s_t, a_t)‖_V      # value-relevant slice of the arity-2 residual
b_{t+1} = (1−α)·b_t + α·e_t                 # running benchmark of your own error
g_t     = ‖FM₂(s_t, a_t) − FM₁(s_t)‖        # arity gap = how much my action explains
δ_t     = (b_t − e_t) · σ(g_t/θ)            # performance error
```

| term | where it comes from | what it fixes |
|---|---|---|
| `e` on the **value-relevant slice** | [`drift_value_loop`](../experiments/mjc/drift_value_loop/README.md) Cut 3 | control is a near-blind grader; FM error is not |
| `b_t −` (baseline subtraction) | Gadagkar's "flexible performance benchmark" | Cut 3's **noisy self-tuning** — this is the actor-critic baseline trick applied to a signal whose landscape is already known good |
| `σ(g/θ)` (agency gate) | the passive-playback control (Gadagkar Fig. 4) | makes this a *cerebellar→value* connection rather than a sensory one |
| fast clock on `δ` | 58–81 ms latencies, a separate projection-defined population | keeps performance error off the reward channel's slow trace — which is where the one direct negative lives (§6) |

The efferent return is a **gain**, not an addend: `δ` multiplies the FM's learning rate (and policy/allocation), it is not
summed into any representation.

## 2. Why we think this works

Six independent lines, four of them from our own prior runs.

**(a) The afferent direction already worked here.** [`drift_value_loop`](../experiments/mjc/drift_value_loop/README.md)
Cut 3 found that grading the meta-loop by control is near-blind ("a replanning controller reaches goals about as well
with a stale model as a fresh one"), but grading it by the **value-relevant forward-model prediction error** — which that
README already calls "the literal cerebellum→VTA prediction-error messenger, which reads the FM's error, not the
downstream reward" — yields a **clean interior optimum the loop can climb** (landscape clean, 3 seeds). The one defect
was that self-tuning came out *directional-but-noisy*. We are not proposing a new mechanism; we are proposing the
variance fix for one that already has the right landscape.

**(b) The post-mortem of the failed version says the missing piece was the teacher.**
[`online_value_loop`](../experiments/mjc/online_value_loop/README.md) built the fully-online loop two ways
(`meta_curiosity_loop.py` afferent, `meta_value_online.py` efferent) and was obstructed on both levers. Its own
conclusion, sharpened by Cut 3: *"The missing piece was never the value structure (two values, self-tuned); it was the
teacher."* We now have a specific better teacher.

**(c) The efferent half is reproduced three ways here and once independently in humans.**
[`value_shaping`](../experiments/mjc/value_shaping/README.md) (value→FM as capacity re-allocation),
[`meta_adapt`](../experiments/mjc/meta_adapt/README.md) §#4d/#4e (the re-allocation *caused by reward*, with reward
rediscovering the value support), and [REACHING_LOOKAHEAD](../experiments/a2a_forward/reaching/REACHING_LOOKAHEAD_README.md)
Finding 2 (value-shaped FM ≻ a perfect simulator for control). Kim, Parvin & Ivry (2019) find the same arrow in humans
and describe it in the same terms — task outcome as **"a gain on implicit adaptation."**

**(d) Everything agrees the return is multiplicative.** [EMOTION_INJECTION](../experiments/a2a_forward/EMOTION_INJECTION_README.md)
found an additive value scalar in the residual stream nearly inert; multiplicative gates worked throughout the arc; Kim
et al. say *gain*. Three independent routes to the same architectural constraint.

**(e) There is a working biological instance, with the discriminating control already run.** Gadagkar et al. (2016) —
see §5.

**(f) It does not require a long-range FM-error projection.** The one thing the literature explicitly does *not* show is
transmission of a cerebellar prediction error to VTA/SNc that shapes dopaminergic RPE. Hull (2020) makes that
unnecessary: climbing fibers already carry TD-like signals, granule cells already carry efference copies *and*
reward predictions, and cerebellar-nuclei output already tracks the predicted-vs-actual mismatch for self-generated
movement. Every ingredient for this computation sits inside one structure. The gating can be local.

## 3. Agency is a precondition for credit, not a veto

This is the conceptual correction that reorganized the whole idea, and it came from one control in the songbird paper.

The earlier draft claimed: self-caused ⇒ predicted ⇒ cancelled ⇒ **invisible to value**. That is wrong, and it was
contradicted from two directions at once. Empirically, Tricomi, Delgado & Fiez (2004) find caudate responds to reward
*only* under perceived action-outcome contingency — self-causation **amplifies** the striatal response. Mechanically,
the residual `s_{t+1} − FM₂(s_t,a_t)` cancels everything predictable from state-and-action and has no way to represent
*which* of the two supplied the predictability — so it could not implement an agency gate even in principle.

The correct reading: **no efference copy → no prediction → no residual → nothing to evaluate.** Cancellation does not
remove the self-generated component from value's view; it *converts* it into a performance error, which is the most
useful thing value can read. Agency **enables** credit rather than vetoing it, which reconciles the Tricomi result
instead of fighting it.

This also repairs the anti-wireheading argument, which was broken in the earlier draft. You cannot farm reward from
self-generated states because the residual goes to **zero** when your prediction is good — and you cannot manufacture a
deviation from your own model by acting, since acting is exactly what the model predicts. That is a stronger and more
structural veto than [RHM_EDIT_CONTROL](../experiments/rhm/RHM_EDIT_CONTROL_README.md)'s hand-installed on-manifold
check or [META_ADAPT §#4e](../experiments/mjc/meta_adapt/README.md)'s fixed weight budget.

## 4. Agency is the arity gap, not the residual

Native to this program's own vocabulary ([the_forecast_needs_a_lead](the_forecast_needs_a_lead.md) §3), with one
precision that doc did not state: **the second argument selects which side you keep.**

| condition on | residual is | use |
|---|---|---|
| the world's input (`x_{t+1}`) | your own revision | the reading tap — [`conditional_revision`](../experiments/rhm/conditional_revision/) |
| your own action (`a_t`) | the external cause | the control tap |
| **the gap `p₂ − p₁`** | **what your action explains** | **the agency term — this doc** |

`e = s_{t+1} − p₂` is *novelty*. `g = p₂ − p₁` is *agency*. They are orthogonal, and the earlier draft conflated them
throughout. The arity-1/arity-2 machinery already exists on this substrate
([`arity_torque`](../experiments/mjc/arity_torque/README.md)).

## 5. The biological instance

**Gadagkar, Puzerey, Chen, Baird-Daniel, Farhang & Goldberg (2016), *Science* 354:1278–1282** — "Dopamine neurons encode
performance error in singing birds." Local copy:
`reading/cerebellum/Dopamine neurons encode performance error in singing birds.pdf`[^private]

VTA neurons projecting to Area X (the songbird basal-ganglia nucleus required for song learning) were recorded while
auditory feedback was distorted on a target syllable:

- **Suppressed** after distorted syllables (worse than predicted): latency 58 ± 13 ms, duration 86 ± 35 ms, ~75% rate
  reduction, on 94% of distorted trials.
- **Activated** at the precise moment a predicted distortion did *not* occur (better than predicted): latency 81 ± 20 ms,
  duration 62 ± 27 ms.
- **Scaled by expectation** — response magnitude depended on distortion probability; syllables are "evaluated against an
  estimate of syllable quality that is diminished by a memory of recent trials (i.e. a **flexible performance
  benchmark**)." This is the `b_t` term.
- **A distinct, projection-defined population**: 17 of 125 recorded neurons formed a separate error-responding cluster
  (p<0.001), spatially intermingled with the other 108; **13 of 14** antidromically-identified Area-X-projecting neurons
  encoded performance error.
- **The control that matters most**: during nonsinging periods, the same neurons did **not** differentially respond to
  playback of distorted vs. undistorted renditions of the bird's own song (normalized rate 1.0 ± 0.1 vs 1.1 ± 0.1,
  p > 0.3). Identical acoustics, no error signal. The paper's reading: "there is nothing intrinsically 'good' or 'bad'
  about these sounds according to the performance-monitoring system," and performance error "is not derived from sensory
  feedback of intrinsic reward or reward-predicting value."

That last item is the experiment we should copy directly (§7, Experiment 2). It is the thing that separates "value reads
the forward model" from "value reads the senses."

## 6. The one direct negative, and why we think it constrains rather than refutes

**Parvin, McDougle, Taylor & Ivry (2018), *J. Neuroscience* 38(19):4521–4530** — "Credit Assignment in a Motor Decision
Making Task Is Influenced by Agency and Not Sensory Prediction Errors." (Not downloaded.) They explicitly hypothesized
that cerebellar SPEs attenuate value updating in the basal-ganglia RL system, manipulated SPE strength, and found **no
effect on choice behavior**; belief about causal structure did the work.

Three reasons we read this as a constraint on *where* rather than *whether*:

1. Their positive finding is that **agency** governs credit. That is this doc's §3, and `g` is a computational form of
   it. What they ruled out is *residual magnitude* as the gating variable — which §3 independently concedes.
2. They probed cross-talk into the **generic reward-RL** channel. Gadagkar's architecture lives in a separate,
   projection-defined **performance** channel. Cut 3 found the same split from the other side: grade by FM error, *not*
   by downstream reward.
3. The same lab found the converse arrow one year later: **Kim, Parvin & Ivry (2019)**, *eLife* 8:e39882 — local copy
   `reading/cerebellum/The influence of task outcome on implicit motor learning.pdf`[^private].
   With clamped feedback holding SPE invariant and only target size varying, **hitting the target attenuates implicit
   adaptation by ~35%** (late learning F(2,45)=4.44, p=0.016, η²=0.17; replicated at a second clamp angle). Their
   conclusion: task outcome serves as **"a gain on implicit adaptation."** Notably the effect was **categorical** — full
   hit only, not straddle.

**This is the load-bearing bet, stated plainly**: we are betting that performance error works as a *separate channel*
rather than as a modulation of reward learning. If it only helps by displacing reward learning, Parvin generalizes.

## 7. Two clocks — the part nobody has done

The forecast leads and its evidence lags, so the prediction must be **held** until the evidence arrives. That is the
Smith predictor (Miall, Weir, Wolpert & Stein 1993) — a 33-year-old published architecture, not a new derivation — and
its biological realization is **Suvrathan, Payne & Raymond (2016), *Neuron* 92:959–967**, where flocculus
parallel-fiber/Purkinje plasticity is tuned to **~120 ms, precisely matching that circuit's error-return delay**, while
the vermis (serving many functions) shows a *range* of tunings across cells. Eligibility-trace timing is
**circuit-specific and matched to the circuit's own delay.**

The dopaminergic trace is a different clock: **Yagishita et al. (2014), *Science* 345:1616–1620** — spine enlargement
only in a **0.3–2 s** window after glutamatergic input.

Nobody has aligned the two. And Gadagkar supplies a datapoint that reframes the question: the performance-error
latencies are **58–81 ms with 62–86 ms durations** — the *cerebellar* regime, not the reward-RPE regime — in a
functionally and anatomically **distinct dopaminergic population**. So the two-clock problem may be solved by having two
populations rather than one reconciled trace. That is testable (§7, Experiment 4) and it is our flag.

Note this cuts one way we did not expect: because traces are tuned *per circuit*, **latency is compensable** — a slow
channel simply gets a slow trace. Any argument that slow channels cancel worse has to rest on **resolution** and
**delay variance** (you can tune a window to a fixed delay, not a stochastic one), not on mean latency.

## 8. Experiments

Ordered by cost. All on the MuJoCo substrate ([`pusher_env.py`](../experiments/mjc/pusher_env.py),
[`shared.py`](../experiments/mjc/shared.py)).

### Experiment 1 — the variance fix (cheapest; tests the central claim)

Retrofit the `b_t` benchmark onto Cut 3's existing teacher in
[`drift_value_loop/online_value_loop.py`](../experiments/mjc/drift_value_loop/README.md). Everything else held fixed.

**Prediction**: self-tuning goes from *directional-but-noisy* to clean, with the interior optimum **unchanged in
location**. Report the variance of the self-tuning trajectory across seeds, not just its direction.

This is a few lines of code against a result we already have, and it is the sharpest single test of whether the
songbird prescription buys anything.

### Experiment 2 — the playback control (the Gadagkar replication; the one that makes this a *bridge* result)

Same sensory sequence, two conditions:

- **ACT** — the agent produces the trajectory (efference copy present, `g > 0`)
- **PLAYBACK** — the identical trajectory replayed as observation, agent passive (`FM₂ ≡ FM₁`, so `g = 0`)

**Prediction**: `δ` fires in ACT and is silent in PLAYBACK.

Why this design and not the SELF/EXT-distractor 2×2 the earlier draft specified: **predictability is identical across
these two conditions**, so a positive result cannot be confounded with "prediction error gates learning," which is
textbook TD. The songbird found the clean manipulation for us. Report `δ` magnitude and `g` separately so the gate's
contribution is dissectible.

### Experiment 3 — noisy-TV discrimination (on data we already have)

Run the three regimes from [`curiosity_control`](../experiments/mjc/curiosity_control/README.md) /
[CURIOSITY_DRIVE](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md) against two formulations that
**disagree in one cell**:

| regime | cost model `−α‖e‖ + β(−d‖e‖/dt)` | benchmark model `b − e` |
|---|---|---|
| dark room | 0 | 0 |
| **noisy TV** | **negative** (frustrating) | **zero** (the benchmark adapts to the noise) |
| frontier | positive | positive |

Parsimony favors the benchmark model; [GATED_RATCHET](../experiments/a2a_forward/GATED_RATCHET_README.md)'s own note
("high static residual = **frustrating**") favors the cost term. This decides whether a separate cost term is needed at
all. No new runs required.

A second, cheap payoff if the benchmark model wins: `b − e` is computable **online from a running average**, replacing
the fragile per-step derivative `−d‖e‖/dt` that
[two_timescale_value_loop](two_timescale_value_loop.md) flags as open question #2.

### Experiment 4 — two clocks

Under delayed reward with fast FM feedback, sweep the performance-error window `τ_fast` × the FM's feedback delay
`d_fm`, with the reward channel on a separate slow window `τ_slow`.

**Predictions**: (i) performance peaks on the diagonal `τ_fast ≈ d_fm` (Suvrathan's per-circuit matching); (ii) a
**two-channel** architecture beats a **single shared-window** channel at matched capacity (Gadagkar's separate
population). The second is the novel one.

### Sequencing

1 → 2 are independent and can run in parallel; 3 needs no compute; 4 is worth building only if 1 and 2 both land.

## 9. Risks

Held loosely — these are the places we would expect to learn something, not conditions we have committed to treat as
falsifiers.

- **Baseline subtraction may not clean up Cut 3's self-tuning.** If it doesn't, the variance diagnosis is wrong, but the
  landscape result stands and the noise has some other source worth finding.
- **`δ` might fire in PLAYBACK.** That would mean the arity gap isn't functioning as an agency signal on this substrate —
  possibly because `FM₁` is too good, which is itself informative about the task's action-dependence.
- **Two channels might not beat one at matched capacity.** The population split could be a fact about birds rather than
  an architectural principle. Worth knowing either way; it does not touch §1–§4.
- **Parvin might generalize.** If performance error only helps by displacing reward learning, the separate-channel bet
  is wrong. This is the biggest one.
- **The noisy-TV cell might come out negative**, which would mean the benchmark model is incomplete and the cost term is
  real. That is a refinement, not a loss — and it would connect §1 back to the compression-cost literature (§10).
- **The agency gate might turn out to be decorative** — i.e. `b − e` alone performs as well as `(b − e)·σ(g/θ)`. Then the
  contribution narrows to the benchmark term, which is still worth having.

## 10. Papers

**Downloaded** (in `reading/cerebellum/`[^private]):
- Gadagkar et al. 2016, *Science* 354:1278–1282 — performance error in singing birds[^private]. The architecture, with the playback control.
- Hull 2020, *eLife* 9:e54073 — Prediction signals in the cerebellum: beyond supervised motor learning[^private]. Why the gating can be local; also lists "how does cerebellar learning modify output to the mesolimbic dopamine system during goal-directed behaviors?" as an open question.
- Kim, Parvin & Ivry 2019, *eLife* 8:e39882 — The influence of task outcome on implicit motor learning[^private]. Outcome → FM as a gain; categorical.
- Manto et al. 2024, *The Cerebellum* 23:2169–2192 — Consensus Paper: Cerebellum and Reward[^private] (abstract only). Review framing, not demonstration — treat its TD/RPE claims accordingly.

**Not downloaded — worth having:**
- **Parvin, McDougle, Taylor & Ivry 2018**, *J. Neurosci.* 38(19):4521–4530. The one direct negative (§6). Highest priority to actually read.
- **Suvrathan, Payne & Raymond 2016**, *Neuron* 92:959–967 — circuit-matched eligibility-trace timing (~120 ms flocculus). Second priority; §7 leans on it.
- **Khamassi, Lallée, Enel, Procyk & Dominey 2011**, *Front. Neurorobotics* 5:1 — efference-copy-gated credit assignment on the *action* side. The nearest prior art; this doc is its sensory-side counterpart.
- Yagishita et al. 2014, *Science* 345:1616–1620 — the 0.3–2 s dopaminergic eligibility window.
- Miall, Weir, Wolpert & Stein 1993, *J. Motor Behavior* — the Smith predictor.
- Brooks et al. 2015 — cerebellar-nuclei output tracking predicted-vs-actual sensory consequences of self-generated head movement (via Hull 2020).
- Ohmae & Medina 2015, *Nat. Neurosci.* 18:1798–1803; Heffley & Hull 2019, *eLife* 8:e46764; Kostadinov et al. 2019, *Nat. Neurosci.* 22:950–962; Wagner et al. 2017, *Nature* 544:96–100 — reward/TD signals inside the cerebellum.
- Carta, Chen, Schott, Dorizan & **Khodakhah** 2019, *Science* 363:eaav0581 — monosynaptic cerebellum→VTA. (Note: [two_timescale_value_loop](two_timescale_value_loop.md) currently miscites the last author as Regehr.)
- Tricomi, Delgado & Fiez 2004, *Neuron* 41:281–292 — agency amplifies caudate reward response (§3).
- Takahashi et al. 2017, *Neuron* 95:1395–1405 — dopamine signals errors in predicted *sensory features*, not only value.
- Muller, Zhang & Sawtell 2023, *Neuron* — mormyrid negative-image cancellation; the canonical statement that cancellation exists *in order to* make external causes detectable.
- Gerstner, Lehmann, Liakoni, Corneil & Brea 2018, *Front. Neural Circuits* 12:53 — the three-factor/eligibility-trace review (surveys striatum, cortex, hippocampus; **not** cerebellum — hence the gap in §7).
- Press, Kok & Yon 2020, *TiCS* 24:13–24 — the perceptual prediction paradox; predicted does not always mean attenuated.
- Zénon, Solopchuk & Pezzulo 2019, *Neuropsychologia* 123:5–18; Piray & Daw 2021, *Nat. Commun.* 12:4942; Gershman & Lak 2025, *J. Neurosci.* 45:e1756242024 — the compression-cost literature, relevant if Experiment 3 says the cost term is real.

## 11. Honest status

Nothing here has been run. §1's signal is an assembly of parts that each have support; the assembly does not.

The strongest ground is **§2(a)** — the afferent teacher already has a clean landscape on this substrate at 3 seeds, and
the proposed change is a standard variance reduction, not a new mechanism. The weakest ground is **§6's bet** that
performance error is a separate channel rather than a modulation of reward learning; that is the one assumption whose
failure would take most of the doc with it.

§3's reframe (agency as precondition) is a correction of our own earlier error, forced by one control figure. It is the
part we are most confident in and the part that most changes what to build.

The claims about which sensory channels are prone to spurious value binding, which motivated the earlier draft, have
been **removed rather than revised** — the literature pass found the supporting anchors individually weak and the
specific empirical claim unmade by anyone. If we want that thread back it needs its own doc and its own evidence.

---

## 12. What we ran (2026-08-11), and where it landed

**Source of truth**: [`experiments/mjc/drift_value_loop/teacher_snr/README.md`](../experiments/mjc/drift_value_loop/teacher_snr/README.md).
This is a pointer, not a summary of record — read that node for method, tables, caveats, and reproduction.

We ran §8's Experiment 1 the same day this doc was written. Three things came back, and two of them
correct claims made above.

**(a) §8 Experiment 1 was a no-op as specified.** The baseline term — the songbird's "flexible performance
benchmark", `b_t − e_t` — was **already implemented** in Cut 3 as the standard REINFORCE baseline
(`adv = R − base`, EWMA). §1's table credits that term with fixing Cut 3's noisy self-tuning; it cannot,
because it was already there.

**(b) §2(a) is falsified as an argument.** The claim was: the afferent teacher already has a clean landscape,
so the proposal only needs a variance fix. The landscape does reproduce (bowl 0.00397, argmin b=0.5) — but
it is **shallower than the per-epoch nuisance** (0.00497), giving SNR 0.80, and across 42 gradient samples
the perturbation explains 0.0% of the advantage variance. The obstruction was never estimator variance. A
dense teacher (averaging the M+1 per-round samples the drive already computes) does not help either:
**74% of the spread is between-epoch state variation** that within-epoch averaging structurally cannot
touch, capping the idea at a 14% reduction before it starts. Removing world motion collapses the absolute
noise ~8× and lifts landscape SNR to 1.40, but leaves that structure unchanged and the loop still does not
climb — the local gradient at the operating point is ~0.6× the noise. On this substrate the obstruction is
the **search procedure** (a score-function estimator on one scalar parameter with expensive noisy
evaluations), not the teacher's form.

**(c) The "fast clock = sample rate" reframe is wrong as stated, and narrows.** Density in the songbird
comes from ~50k renditions of a *repeated trial against a stable target*, sampled fast **relative to how
slowly competence changes** — not from a higher measurement rate. Measuring a still-learning forward model
more often within an epoch measures the same moving state more times. Untested and worth stating as a
hypothesis rather than a property: a performance-error teacher may need measurement rate decoupled from
learning rate.

**What this does not touch.** §3 (agency as precondition), §4 (the arity gap), §5, §6, and §7 were not
tested — nothing here bears on them. **Experiment 2, the playback control, is unaffected and is now the
most informative remaining test**: it asks whether δ discriminates ACT from PLAYBACK, which is a *signal*
test and never requires the outer loop to converge.

**One framing correction, unrelated to these runs.** §2 and §7 lean at points on non-stationarity as the
condition under which a second loop pays. [two_timescale_value_loop](two_timescale_value_loop.md) amended
that on 2026-07-29: the `full_loop` 2×2 separates world motion from grader type and finds an evaluative
grader expands the belief *identically* in static and drifting worlds. The operative conditions are grader
**type** and **metered sample price**, not world motion. Any rewrite should not lean on drift.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
