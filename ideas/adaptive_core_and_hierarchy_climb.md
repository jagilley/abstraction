# The adaptive core and the hierarchy climb: satiety as the expansion mechanism

**Status**: conceptual synthesis from a 2026-07-28 discussion. No new experiment. Every number is re-read from an existing node; the frame, the §7 resolution, and the §11–§13 synthesis are new and unbuilt.
**Built and partly amended 2026-07-29** → [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md) built §4's loop; [meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) is the follow-on synthesis. Three corrections that matter for reading this file: **§6's homeostatic argument survives only in its *necessity* form** (invariance was free — the surface arm repaired 91% of damage in-round — so drift alone induced no climb; starving *samples per event* at fixed magnitude is what migrates the gain deep); **§12's satiety-as-climbing-mechanism has no axis** on a channel-indexed budget and recruits *sideways* into a highly-reducible-but-irrelevant distractor, costing more than the entire allocation prize; and **§10's satiety prediction is confirmed** (irreducible leak 5.3% → 0.8% of the learned budget). §7b's barrier-vs-dilution discriminator remains unrun and is still the cheapest open experiment here.
**Date**: 2026-07-28
**Prompt**: a conversation that started as *"we've demonstrated the full inner/outer loop plus forward model for the first time on MuJoCo, but we've gone off the deep end hand-crafting environments — what translates back to the symbolic domain?"* and ended up somewhere else: a homeostatic account of why hierarchy-climbing is the optimal response to drift, and a mechanism (satiety) for making an agent do it.
**Builds on**: [heterogeneous_graders.md](heterogeneous_graders.md) (§4/§4b dense-vs-evaluative, compression-vs-expansion), [two_timescale_value_loop.md](two_timescale_value_loop.md) (the two-teacher interface, Type-1/Type-2 non-stationarity, the somatic marker), [physical_control_substrate.md](physical_control_substrate.md), [efference_copy_cancellation.md](efference_copy_cancellation.md), [self_model_needs_a_loop.md](self_model_needs_a_loop.md)
**Key experiments**: mjc — [E3](../experiments/mjc/on_policy/directed_on_policy/README.md), [on_policy](../experiments/mjc/on_policy/README.md) E0–E2, [expansion](../experiments/mjc/expansion/README.md) (cut #5), [dynamics_shift](../experiments/mjc/dynamics_shift/README.md), [drift_value_loop](../experiments/mjc/drift_value_loop/README.md), [curiosity_control](../experiments/mjc/curiosity_control/README.md), [COLLECTION_REALISM](../experiments/mjc/on_policy/COLLECTION_REALISM.md) · rhm — [sculpting](../experiments/rhm/RHM_SCULPTING_README.md) (esp. Stages 4–5), [sculpt-continual](../experiments/rhm/RHM_SCULPT_CONTINUAL_README.md), [specialization](../experiments/rhm/specialization/README.md), [residual_decomposition](../experiments/rhm/residual_decomposition/README.md), [ACTIVE_RHM](../experiments/rhm/ACTIVE_RHM_README.md), [RHM_EDIT_CONTROL](../experiments/rhm/RHM_EDIT_CONTROL_README.md)
**Attribution**: the homeostatic framing (§6), the cortical-control-of-value proposal (§7), the both-graders/tuning point (§9), the hardcode-until-language scoping (§10), the satiety observation (§11), and the adaptive-core-vs-climb consolidation (§12) are Jasper's, quoted at the head of each section. The repo-side reconciliations and the §13 synthesis came out of the exchange.

---

## One-liner

**Drift on a non-hierarchical task buys you an adaptive core and nothing more — which is exactly what
`mjc/` measured, calibrated, over eight months of work. Drift on a *hierarchical* task can in
principle buy you hierarchy-climbing, because higher levels are more invariant under shift and
therefore cheaper to maintain.** What turns "can in principle" into "does" is a small hardcoded
outer-loop prior — *satiate on the current level, prefer the next one up* — and the missing piece we
have never built is the **satiety half**: our learning-progress drives decay to zero on a mastered
region, which produces indifference, where the biology goes actively negative, which produces a
push. Satiety is not a detail of the drive's shape; it is the mechanism that converts polishing the
current level into climbing to the next.

---

## 1. Where this started

> *We've learned a lot from running experiments of this nature on the MuJoCo substrate, and for the
> first time we've demonstrated the full inner/outer loop plus forward model system. But I think
> we've also gone off the deep end a bit in terms of needing to hand-craft environments that closely
> mirror the real world without cheating. Suppose we were to go back to the discrete, symbolic
> domain, on e.g. RHM. What have we learned that'd translate, such that we could have fresh eyes for
> a new approach? What would be the direct RHM version of the E3 full-loop run?*

The complaint has receipts. [E3](../experiments/mjc/on_policy/directed_on_policy/README.md)'s own
gotchas are three environment-geometry bugs in a row: on/off-reach had to be redefined as gate-summed
occupancy of an FK'd joint sweep after a corridor test *and* an angle test both mislabelled regions;
regions had to sit in a hand-found "extended/tame radius band" or their fast dynamics gave ~10× the FM
error; and noise `amp` had to be held `< gear` or the distractor destroyed the target's reducibility
(A's ceiling 0.14 → 0.37). And [`expansion/`](../experiments/mjc/expansion/README.md) states the
cheating problem outright: **the prediction target is `qpos/qvel`, the simulator's own generalized
coordinates — privileged access no embodied learner has.**

Note which half was expensive. The metering (`Body`, no `set_state`, every step charged) is small and
clean. The *geometry* ate the runs. RHM deletes the geometry cost and keeps the metering question
intact — the partition, the visitation measure, the reducibility label and the per-region difficulty
are all lookups in a known DGP.

**What ports** (each with its RHM translation):

| mjc lesson | evidence | RHM form |
|---|---|---|
| Two graders of different type; never trust one | control flat to 0.1% where FM-error has a clean optimum (drift_value_loop Cut 3); ladder resolves only on region-A FM error (E3) | already owns a better-instrumented pair — token CE (dense/blind) vs ground-truth ancestor recovery (sighted). Already caught disagreeing: SUBTREE gets val 1.561→1.553 with depth Δ ≤ 0.013 |
| Metered measurement makes "where should I practise" a question at all | 22× subsidy → unmeasurable; 1.84× → positive | RHM's probes are free **and** privileged. Allocation signals must be label-free functionals of the model's own loss and generations; ancestry probes are grader-only |
| Local drift is the lever; global drift is free | E1 global re-adapts under every mode (5.5/4.8/4.9×); E2 local costs 2.5× and broad teleport *never* repairs | RHM's novelty arm failed for E1's reason inverted — wholesale rule-swap is task switching (val on old rules 1.41→4.51), i.e. maximally global |
| Commitment is what makes the FM load-bearing | ballistic transmits FM quality 3.0–3.3×, re-adapts 4.3–5.2× | **already re-derived on RHM and never connected**: sculpting length-gen Round 3, closed-loop w64 at c=1 gives arity-2/arity-1 = 0.672/0.672 (erased); open-loop gives 0.439/0.178 |
| The noisy-TV control is mandatory | `error-only` burns 47% of budget on irreducible regions | RHM makes irreducibility *exact* rather than tuned |

## 2. A correction that relocates the experiment

The obvious port — allocate a metered budget over (domain, level) rule cells and see whether
`lprog × visits` beats uniform on plain NTP — **is already ruled out, with a reason.**
[two_timescale_value_loop.md](two_timescale_value_loop.md) records a substrate correction dated
2026-07-17: when that line went looking for a home for exactly this loop it considered RHM and chose
active vision instead, because the [specialization](../experiments/rhm/specialization/README.md) line
shows *the RHM depth frontier is not moved by allocation* — so an allocation-driven signal has no
lever and an `LP ≈ uniform` result would be a **confounded null**. The receipt: supervising subtree A
alone lifts B's `d4` from 0.108 → 0.408, and broad beats narrow on A's own domain (0.581 vs 0.440).

The disciplines survive; the venue does not. What relocates it is the forward model.

## 3. The forward model: physics-state → activation, still arity-2

> *We'd be likely going back from the physics-state FM to the activation FM here, ideally still
> arity-2.*

This constraint picks the venue, because **the repo already has an arity-2 activation FM on RHM and
it lives in sculpting**: `FM(z, k) → Δz` predicts the controller's *own* per-block latent given that
latent plus the edit command. Activation-space (it models the model, not a plant) and
command-conditioned. The arity-1 control is built too — `_build_block_fm_arity1`, whose MSE optimum is
exactly `E_k[Δz]`, the command-averaged "received wisdom" floor, with `rank_corr ≡ 0` by construction.

**A second construction, if we want the *cross-layer* a2a FM rather than the latent-transition one.**
The a2a FM is arity-1 by construction — predict layer `j` from layer `i` at fixed input, nothing to
condition on. The way to give it a command on an autoregressive model is to make the command **the
token the model itself just emitted**: `FM(h_t, y_t) → h_{t+1}` against an arity-1
`FM(h_t) → E_y[h_{t+1}]`. That is the literal efference copy — the self-generated action whose
consequences on one's own state should be cancellable — and it has two RHM-specific properties:

1. **It only bites in free-running generation.** Teacher-forced, you observe the true token and
   re-ground every step, so arity is moot. That is Round 3's closed-vs-open result at the token level.
2. **The arity gap should scale with `m`.** Synonymic multiplicity sets how broad the model's own
   next-token distribution is. At low `m` the emitted token is nearly determined and `E_y[h_{t+1}]`
   ties; at high `m` it carries a lot and arity-1 floors. A knob-controlled dose-response predictable
   before the run.

It also lands on a pending thread: cancellation's Payoff 1 (does subtracting the forecast make the
dependency *modular* rather than entangled?) is logged **"substrate-limited / inconclusive — needs a
gauge-free substrate (RHM / language)"**, and sculpt-continual already runs the diagnostic that would
settle it (fresh FM stays +0.15 above co-trained — no privatization).

## 4. The two loops

**Inner — dense grader, drives compression.** Controller `M` encodes a config into per-block latents
`z`; the arity-2 activation FM predicts `Δz` under command `k`, trained on self-supervised prediction
error over transitions actually taken — free, every step, no goal. It proposes on-manifold moves,
ranks by `value(b + FM(z,k))`, commits, and either re-grounds (reactive) or rolls its own prediction
forward (ballistic). Optionally wired as **cancellation** rather than summation. Its native tendency
is compression: absorb the predictable, shrink the residual. On a fixed task it terminates; under
drift it re-fits the same directions forever.

**Outer — evaluative grader, hypothesised to drive expansion.** It reads the FM twice and writes back
once:

- **`e`-tap (explore)**: learning progress on the *reducible* residual, `−d‖e‖/dt`, filtered by
  fresh-FM-ensemble invariance so it rejects the noisy TV.
- **`p`-tap (relevance)**: roll the FM forward to forecast where the planner will actually go.
- The product allocates a metered budget over start-state classes. **The meter is native**:
  sculpting already counts materializations (token beam `W·n_regions`/step, latent beam `W` — the ~8×
  claim). Charging the monitor from that pool is a config change, not a build.
- **The write-back is what matters for expansion**: the grounded planner CE against the DP best move
  `k*`, the term that reshaped the belief in Stage 5. Allocation decides *where* `k*` is computed —
  which is sculpt-continual's own untested next step #3, flagged as possibly strengthening the weak
  ratchet *or collapsing it*.

The FM is read three ways from one object — forward for control, residual-derivative for explore,
rolled-forward for relevance — which is what makes this one system rather than two bolted together.
That is E3's finding #4 verbatim.

## 5. The 2×2 that makes it mean something

`heterogeneous_graders` §4b carries a caveat added 2026-07-27: the mjc expansion null shows dense
pressure is *insufficient*, **"not that an evaluative grader is *sufficient*. Whether an evaluative
grader actually expands anything is still untested."**

It is not untested. **Sculpting Stage 5 ran it and labelled it something else.** Two *nested* rungs
differing only by the grounded term:

| grader | what it is | belief PR | transferable plannability |
|---|---|---|---|
| `fm_cotrain` — "be predictable" | pure FM-predictability pressure = **dense** | 6.7 → **6.8** | top1 0.357 → **0.304** (down) |
| `planner` — DP-best-move CE | references outcomes = **evaluative** | 6.7 → **17.0** | top1 0.357 → **0.521** |

|  | fixed task | locally drifting task |
|---|---|---|
| **dense grader** | terminates — measured (PR 6.8) | re-fits forever, no expansion — **cut #5, needing the RHM replication in a domain that has a hierarchy** |
| **evaluative grader** | expands *once*, then consolidates — measured (6.7→17.0, then 12.8→10.1; continuous re-internalization only +0.02) | **unrun** |

Three cells have anchors. The fourth is the question. It matters because *"compounding requires a
moving frontier"* was discovered three separate times here — the ratchet arc, the latent loop,
sculpt-continual — and **every one was measured with a dense grader in the loop.** The exhaustion law
may be a law about *compression*, not about learning.

Read out with the corrected instrument (β and `R_res_participation`), which is the one thing that
gets *better* going back: β is capacity-invariant to ±0.01 at d=128/256 with R² 0.95–0.99, and cut #5
found it unusable at `d_state = 10`, failing by 6–59×. Cut #5's own conclusion was that shape
questions need a high-dimensional domain and that expansion needs a hierarchy to expand into.

## 6. The homeostatic argument, and why it replaces the instrument

> *Suppose you do have a drifting task that is also hierarchical. Then, from a purely homeostatic
> perspective, the optimal move is simply to move up that hierarchy in understanding, so that you
> have to expend less energy processing the world around you.*

This is derivable, not merely intuitive. If drift hits the surface while deep structure is invariant,
a learner holding only surface statistics pays re-fitting cost on *every* drift event, forever; a
learner holding the deep invariants pays once and then patches a thin surface layer. Integrated cost
over the drift sequence is minimised by climbing. Same content as two_timescale's *"LP is a bet on
non-stationarity — invest in reducible structure now because invariants pay off under future shift"*,
but denominated in energy rather than in wager, which is the more useful currency.

**It changes what we measure.** This program has been burned by rank-shaped instruments three times
and then found β unusable at low dimension. The homeostatic reading says: stop counting dimensions,
**count the cost of the next repair.** If climbing is real, drift event #8 costs less than drift event
#1 *at matched drift magnitude*; if it is not, repair cost is flat forever — exactly what "drift
demands re-fitting existing directions, forever" predicts. Magnitude-honest, no degenerate ends, and
already mjc's native currency (E1's 5.5/4.8×, E2's 2.5×).

Two design consequences:

- **The drift must be surface-local and deep-invariant, or the argument doesn't bite.** If deep rules
  drift too, climbing buys nothing. That is not a nuisance, it is the **knob**: sweep *which level*
  drifts and predict the climbing payoff scales with how deep the invariance goes. A dose-response the
  DGP hands you, which the physical substrate structurally could not.
- **Read it against the depth probe, not alone.** "Fewer rounds to recover" could just be "warmer
  model, more data" — any model gets faster at fitting as it trains. But the discriminator does not
  have to be one clean control, because RHM hands us a **second instrument of a different type**:
  per-level ancestor recovery `d1…d6` reads climbing *directly*. Climbing says repair cost falls
  **and** depth rises; ordinary continued training says repair cost falls and depth stays put. That is
  the arc's own grader-disagreement discipline — two instruments, and the reading lives in whether
  they agree — rather than one instrument leaning on a control. Keep matched drift magnitude and the
  frozen probe (same experiment every round; only world and model differ) as hygiene, and heed
  sculpt-continual's `r0→r1` confound (value-iteration conflated with a collector switch) as the
  warning about moving two things at once.

## 7. Cortical control of the value system — the tension, and the way out

> *It may be tempting to try to fit local noise, even where that fit is spurious — a local,
> memorization-driven dense-type grader. But a truly intelligent agent ought to be able to control its
> machinery to make learning higher levels of the DGP its preferred value solution, to better process
> the full distribution of what it sees. This cortical control of the value system ought to put the
> global loss minimum for the cortex itself at the DGP-optimal solution, in principle. (I'm not sure
> we actually have the correct wiring for the inner loop to change the outer loop yet.)*

The repo has measured against the naive version of this four times:

- `fm_cotrain` (endogenous "be predictable") **caps**: PR flat at 6.8, transferable plannability down.
- data2vec (EMA self-teacher) **caps** at ~20% of the depth gap; mlm (grounded in true tokens)
  recruits ~40%.
- [Edit-control](../experiments/rhm/RHM_EDIT_CONTROL_README.md): a value collapsed onto the belief
  scalar `P(root=r*)` is **gamed off-manifold** — planning drives `P(r*) → 1` while ~99% of sequences
  go off-grammar. Fixed only by an on-manifold self-consistency veto.
- mjc #4e: the reward loop **games loss-scale rather than allocation** unless the weight budget is
  normalized.

And sculpt-continual is explicit that Stage 5 worked *because* the DP `k*` teacher is external and
precomputed — "a stable ground-truth teacher, not self-distillation, so the loop can't wirehead its
own value." So "the cortex shapes what its value rewards" is structurally the configuration that has
capped or been gamed every time we have built it.

**The proposal survives with one constraint the homeostatic frame supplies.** What makes an
endogenous value gameable is that the agent *reports* it — a belief scalar, a competence estimate, a
loss magnitude. A homeostatic value is denominated in something the agent **pays**: transitions
spent, materializations burned, re-fit steps consumed after a drift event. You cannot inflate *"I had
to do less work this round"*, because the work is imposed by the world's drift rather than asserted by
the model.

> **The design rule this yields: an endogenous value must be denominated in a currency the agent pays,
> not one it reports.**

This also retroactively explains why metering is load-bearing beyond E3's reason. We adopted it
because free surveys make "where should I look" a non-question. The deeper reason is that **a metered
currency is the only endogenous reward with no wireheading surface** — and therefore the only shape in
which self-controlled value is safe to build.

The honest hole: pure cost-minimisation is the dark room — doing nothing costs nothing. It has to be
*minimise integrated cost subject to holding competence on the graded distribution*. Constrained, not
free.

### 7b. Barrier, or just flat?

The "global minimum ought to be at the DGP-optimal solution" clause points at a mechanism different
from the one §4b assumes, and RHM can tell them apart.

§4b argues expansion needs an evaluative grader via **activation energy**: Copernicus got worse
predictions before better, so monotone descent cannot cross the barrier. But KFW says learning level ℓ
from a token target costs `vm^(ℓ+2)` — exponentially diluted — and specialization then found that
reweighting that diluted target toward depth is **monotonically harmful** (d4 0.340 → 0.286 → 0.160).
Together those describe not a barrier but a **flat**: the minimum is where the argument says it is, and
the gradient toward it is exponentially small.

That changes what the evaluative grader is *for*. Under the barrier story its job is to tolerate a
worse fit long enough to cross. Under the dilution story its job is to **supply gradient the dense
signal structurally does not carry** — *signal, not objective*.

**The discriminating measurement** (cheap, and I don't think we've ever looked): interpolate between
the NTP solution and the oracle-aux solution and watch token loss along the path. Barrier → it rises
then falls. Dilution → it is flat, and the deep structure is free on the landscape but invisible to
the gradient. The answer changes the fix.

## 8. Not either/or — and the mix has an interior optimum

> *I don't think it's an either-or between the dense grader and the evaluative grader for the full
> online loop with both systems. You presumably need both, because they play complementary roles. And
> in deployment you might have to fine-tune your agent to get the mix right. I don't think humans are
> at all good at tuning their dense versus evaluative machinery, and we're often quite bad at not
> succumbing to just doing whatever the evaluative loop says.*

Two anchors say the mix is a real knob with an interior optimum:

- [`curiosity_control/`](../experiments/mjc/curiosity_control/README.md): pure curiosity has a
  noisy-TV pathology on low-dim control; **grounding fixes it** by running explore and exploit as two
  additive drives, with an **intermediate-balance optimum**.
- [`drift_value_loop/`](../experiments/mjc/drift_value_loop/README.md) Cut 3: that balance has a clean
  interior optimum at `b = 0.5` — **visible only to the FM-error grader**. Control was dead flat there
  (0.1% spread).

Which is the structural version of the human observation: the mix has an optimum, and **the grader you
tune it with determines whether the optimum is visible at all.** A system tuning its own mix
necessarily judges with one of the two things it is balancing. That is the same failure as §8's
LLM-as-judge point — checker and checked go blind together — and it is why we should expect *any*
system, biological or not, to be bad at this.

One amendment to the human claim: the somatic-marker section reads the pathologies (phobia, addiction,
superstition) as the **dense cache going spurious** with the evaluative loop failing to override it,
not the reverse. Both directions of imbalance exist, and both are only diagnosable under shift — which
is another reason drift is in the design.

## 9. Scoping: hardcode the outer loop until language

> *My suspicion — hard to establish from comparative anatomy — is that humans have more
> inner-loop-to-outer-loop control than other animals because we're the only animal in a domain where
> things change fast enough that it's required for fitness. Simple organisms get by fine with
> hard-coded outer-loop heuristics, which is something Ilya says almost verbatim. So for motor
> learning, and probably for RHM too, hard-coded outer-loop heuristics are mostly fine; we'd only need
> the reverse pathway if and when we port this to language.*

This resolves §7's tension rather than dodging it, and it can be stated as a condition. A frozen prior
is optimal when the **distribution of shifts** is stationary, even though the world shifts. The
two-timescale doc splits non-stationarity into Type-1 (epistemic; DGP fixed, knowledge moves,
self-terminating) and Type-2 (environmental; the DGP changes, unbounded). The argument needs a third
level:

> **The reverse pathway earns its keep exactly when the shift-generating process is itself
> non-stationary — i.e. when a hardcoded prior can go stale within a lifetime.**

That is precisely the failure the somatic-marker section names — snake fear, hunger-in-abundance:
*"not never causal, but out-of-distribution now."* Simple organisms live in a Type-2 world with a
stationary drift distribution; hardcode and you are done. Human culture changes its own rate and kind
of change within a generation.

Consequences for us: **on mjc and RHM we hardcode**, and the frame predicts the reverse pathway buys
**nothing** there — a null we can check cheaply if we ever want to, which turns the comparative-anatomy
hunch into something falsifiable without the anatomy. It also matches the known knife-edge: the
two-timescale doc's open magnitude question is exactly *how frozen* the value head should be — fully
frozen is robust but cannot fit; **fully plastic collapses into the inner loop**, at which point there
is one grader again and the whole frame is void. The plasticity knob and the mix knob are the same
knob.

## 10. Satiety: we have the decay, we have never built the sign flip

> *The human/animal outer loop seems to be negative-feedback in a way I'm not sure we've ever
> implemented. Eating food tastes great when you're hungry, and then at some point it flips over to
> being actively anti-dopaminergic if you're too full.*

Our learning-progress drives already have the **shape** — ~0 when mastered (dark room), ~0 when
irreducible (noisy TV), peak at moderate-and-falling error — and curiosity Phase 1 describes LP as
riding the reducible frontier and then **releasing** it. But release is decay-*to*-zero: indifference.
Satiety is decay *through* zero: aversion. They come apart exactly where we keep getting bitten.

- **E3's `value` policy still leaks 29% of its budget into the irreducible noise regions.** With
  decay-to-zero, a mastered or irreducible region has near-zero pull — and near-zero plus estimator
  noise still attracts sampling. A negative lobe actively pushes out. One-line change to the drive,
  directly testable on an existing ladder: does the leak fall?
- **It makes anti-overfitting endogenous.** "Don't keep fitting the local thing" currently has to be
  imposed as a regulariser or a capacity limit. Satiety says: once a region is extracted, further
  fitting there is *bad*, not neutral. That is the spurious-local-fit worry of §7, answered by the
  drive rather than bolted on beside it.

The hazard to design against, symmetric to the dark room: a satiating drive can be gamed the other way
— never master anything, so nothing ever goes negative (perpetual novelty-seeking, dilettantism). Same
fix as §7's cost-minimisation: constrain by competence on the graded distribution.

## 11. The consolidation: adaptive core vs. hierarchy climb

> *Drift over a non-hierarchical task just leads to retaining an "adaptive core" of representations
> plus re-adapting to whatever the current shift is — the simple-animal condition. Drift over a
> hierarchical task in principle leads to some degree of hierarchy climbing, perhaps indefinitely if
> the agent takes over its own value system strongly enough. That requires hardcoding "higher in the
> hierarchy = better" into the outer loop — which is kind of spurious, but is the kind of thing
> evolution has done countless times ("more food eaten = better") — and then if your language model
> wants to strengthen or change that prior, that's its prerogative. It feels like this connects to
> specialization, where we previously found it doesn't help so long as your objective is next-token
> prediction, but which in principle must work somehow if your value system is literally putting its
> finger on the loss landscape.*

**The adaptive core is not a hypothesis — it is what `mjc/` measured.** Cut #3: the FM re-adapts from
~50 reward-free transitions because the shift corrupts only the world-model factor — invariant core
retained, drifted factor repaired. Cut #5: eight rounds of perpetual drift move the frontier ±0.08
directions against a +0.72 ± 0.42 calibration, and the writeup itself says a flat frontier there is
*health, not a failed compressor*.

So the honest reading of cut #5 is not "the flagship came back negative." It is **the complete,
calibrated characterization of the simple-animal condition** — adaptive core plus repair, no climbing,
on a substrate with nothing to climb. Which makes it the **control arm** for the RHM experiment rather
than a disappointment, and fills in the 2×2's dense/drifting cell with a calibration attached.

**And it reconciles the specialization negative.** Everything specialization tested was a
*reweighting of the token loss*: restrict breadth (inert, Δ ≤ 0.013), skew gradient toward deep
positions (harmful and monotone, d4 0.340 → 0.286 → 0.160), concentrate the direct target on a subtree
(hurts — broad beats narrow, because the shared grammar means A's depth *is* B's depth). Exp 2's
mechanism is general: deep composition is built on the shallow substrate, so starving the low levels
removes the foundation.

But *"the value system putting its finger on the loss landscape"* is not a reweighting. It is a
preference over **which target to pursue** — and in the KFW frame target choice is the difference
between `vm^(ℓ+2)` and a depth-independent `vm³`. That is the only intervention that has ever moved
the frontier (root 0.08 → 0.80), and we already have the non-privileged proof that target selection
works without cheating: mlm recruits ~40% of the oracle's depth with zero DGP knowledge (and survives
a fully-agnostic span-masking ablation), while grounded internalization buys the rollability mlm
cannot.

> **Specialization-by-reweighting is dead — measured, both axes. Specialization-by-target-selection
> with an intrinsic selector has never been tried.** Those are different claims and only the first
> has been falsified.

## 12. The synthesis: satiety *is* the climbing mechanism

The naive form of "higher = better" is a static skew toward deep levels — which is exactly what
specialization Exp 2 ran, and it is harmful for a principled reason. The prior has to be **gated on
consolidation**: not *always prefer up*, but *once level ℓ stops paying, recruit ℓ+1*. That is the
specialization README's own reframe ("a learner that already holds level ℓ recruits ℓ+1 from a small
marginal sample"), and it is the only form that does not starve the substrate.

What tells you level ℓ has stopped paying? **Satiety.** The negative lobe is what makes polishing ℓ
actively unrewarding, which is what forces the move to ℓ+1. Decay-to-zero leaves the agent content to
keep polishing — which is memorisation of the current level, the spurious local fit of §7, one rung
down.

So the hardcoded outer-loop prior is small and evolution-shaped, exactly as §11 says it should be:

> **Satiate on the current level; prefer the next one up.**

Two lines. Everything else — where to collect, what to model, when to commit — is downstream. §10's
satiety and §11's climbing prior are not two ideas; they are the same mechanism seen from the drive
side and the objective side.

**Scope, so we don't over-claim.** RHM is finite-depth, so climbing terminates at the root — Type-1
self-termination, which is *why* the ratchet exhausted every time we ran it. On RHM we would measure
*does it climb at all, and does climbing show up as declining repair cost per drift event* — not
unboundedness. The unbounded version needs a hierarchy that keeps being built, which is the
culturally-constructed case, which is the language port, which is where §9's reverse pathway comes
back in. The pieces line up in the same order.

## Predictions / falsification

- **Falsified if** the evaluative/drifting cell also exhausts in one pass. Then expansion is one-shot
  regardless of grader type and §4b's "expansion needs an evaluative grader" loses most of its content
  — the grader type would not be what does the work.
- **Falsified if** repair cost per drift event is flat in the hierarchical/evaluative arm at matched
  drift magnitude. That is the homeostatic claim's whole content — and equally falsified, in the other
  direction, if repair cost falls while the per-level depth probe stays put, which is the
  ordinary-continued-training reading rather than a climb.
- **Predicts** the climbing payoff scales with how deep the drift-invariance goes (sweep which level
  drifts; drift at `L0` only → large payoff, drift at the root → none).
- **Predicts** a satiating drive reduces E3's 29% budget leak into irreducible regions, where a
  decay-to-zero drive does not. Cheap, on existing apparatus.
- **Predicts** the arity gap of the emitted-token activation FM scales with `m`, and is ~0 under
  teacher forcing at every beam width.
- **Predicts** the reverse (inner → outer plasticity) pathway buys nothing on mjc or RHM, and only
  pays where the shift-generating process is itself non-stationary.
- **Discriminates** barrier from dilution: interpolate NTP → oracle-aux and watch token loss. Rises
  then falls = barrier. Flat = the gradient is missing, not blocked.

## What this does not establish

- **Belief PR is not the frontier.** Stage 5's PR expansion may be the latent-loop's "grounded target
  feeds gradient to idle capacity" signature rather than a frontier opening. If belief-PR and
  `R_res_participation` dissociate, the headline has to be restated.
- **The instrument window binds.** Cut #5 found the readout degenerate at *both* ends — saturation
  below, diverged rollout above — so everything must sit inside `0.02 < frontier mass < 0.90`, and
  sculpting's open-loop rollout is exactly what diverges past ~6 steps.
- Every RHM sculpting number above is **single rule-seed and directional**; the mjc numbers are 3-seed.
- The satiety mechanism, the metered-endogenous-value rule, and the consolidation-gated climbing prior
  are **arguments with anchors, not measurements**. Nothing here has been run.
- The comparative-anatomy claim in §9 is a **suspicion with a formalization**, not evidence.

## Open questions

- Repair-cost-per-drift-event is confoundable with ordinary continued training, and the per-level
  depth probe resolves the *sign* (§6). The open part is **quantitative**: how much of a falling cost
  curve is attributable to the climb rather than to being further along in training? Sign agreement
  between two instruments does not apportion the effect, and apportioning it may need the matched-data
  arm after all.
- Does satiety need to be a *state* (a depletable internal variable, as hunger is) or does a
  sign-flipped function of LP suffice? The biology is opponent-process, not a reshaped scalar.
- What is the RHM analog of the competence constraint that stops satiety collapsing into
  dilettantism — and does it have to be extrinsic, reintroducing the grounding problem?
- Does the on-manifold veto (edit-control's wireheading fix) compose with a cost-denominated value, or
  is it made redundant by it?
- If specialization-by-target-selection works, does it produce *selective* depth (A deep, B shallow)
  on a shared grammar — or does the two-way transfer still flood the complement, meaning the
  intrinsic selector needs cut-3's heterogeneous DGP to express?

## Context pointers for a future agent

1. [heterogeneous_graders.md](heterogeneous_graders.md) §4/§4b end-to-end — this doc is its
   continuation, and §4b's "untested" caveat is what §5 answers.
2. [two_timescale_value_loop.md](two_timescale_value_loop.md) — especially the 2026-07-17 substrate
   correction (§2 here), the Type-1/Type-2 split (§9), the two-afferent-taps interface (§4), and the
   somatic-marker section (§7, §8).
3. [`mjc/expansion/`](../experiments/mjc/expansion/README.md) — read as the *control arm* per §11, and
   for the calibrate → measure → calibrate discipline and the instrument's two-ended degeneracy.
4. [`rhm/RHM_SCULPTING_README.md`](../experiments/rhm/RHM_SCULPTING_README.md) Stages 4–5 and
   [`RHM_SCULPT_CONTINUAL_README.md`](../experiments/rhm/RHM_SCULPT_CONTINUAL_README.md) — the
   substrate, the arity-2 activation FM, and the dense-vs-evaluative rungs of §5's table.
5. [`rhm/specialization/README.md`](../experiments/rhm/specialization/README.md) — both negatives and
   the 07-18 reframe; §11's reconciliation depends on reading Exp 2's mechanism, not just its verdict.
