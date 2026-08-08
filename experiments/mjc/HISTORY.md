# HISTORY — the mjc node

*A big-picture read of the arc, not a changelog. See [`README.md`](README.md) for the current-state
bullet list and [`FILES.md`](FILES.md) for the lookup index; this document is about how the thinking
moved.*

## Why this substrate exists

`mjc` was founded as a deliberate rung between RHM (known-latent, discrete, language-like) and real
language — a controllable-dynamics DGP with genuine continuous contact physics, built with the same
"sweep the knobs, ship a dissociation" discipline as RHM rather than as an RL benchmark
([`README.md`](README.md) §What this is). That founding discipline — *policies obtained cheaply, every
experiment a knob sweep with a vanilla baseline, no PPO-maxxing* — is the one constraint that never
relaxed across the whole arc below, and it's what makes the later results (many of them negative)
legible instead of just noisy.

## The foundational trilogy: confound-killing becomes house doctrine

Cut #1 ([`contact_residual/`](contact_residual/README.md)) ported an a2a claim — residuals concentrate
at surprising events — onto contact-rich physics with no RL in the loop yet. The interesting move came
*after* the headline: the residual spiked and decayed in ~4 steps while contact itself persisted for
20+, forcing a reframe from "contact is a hard *state*" to "contact-onset is a hard *event boundary*."
Two devices born here under real friction — Huber over MSE, cosine similarity over raw residual ratios
— became permanent fixtures downstream.

Cut #2 ([`arity_torque/`](arity_torque/README.md)) deliberately restricted itself to free-flight,
*using* Cut #1's finding to design around a confound rather than let contact noise contaminate the
arity comparison, and added its own confound-kill (i.i.d. commands, measured not assumed) that recurs
as a template.

Cut #3 ([`dynamics_shift/`](dynamics_shift/README.md)) is the trilogy's pivot: an early per-step
replanning design came back near-null because per-step feedback lets search substitute for the model,
so a stale forward model never bites. The fix — commit to an open-loop horizon (`replan_every`) before
re-grounding — is what made the world-model genuinely load-bearing, and it seeded two things that
outlived the cut: the "control/replanning as a blind grader" idea that keeps resurfacing through the
whole value↔FM arc, and the biological reading (`replan_every` ≈ how ballistic a movement is) that
later became its own arc.

## The first pivot: from verifying atoms to phenomenon-first

[`value_shaping/`](value_shaping/README.md) and [`directed_readapt/`](directed_readapt/README.md) sit
as siblings probing Cut #3's two layers, and both come back as **instructive negatives**.
Value-shaping's efficiency "teeth" turn out to be a data-scarcity artifact. Directed_readapt runs two
clean nulls: a global shift gives undirected collection nothing to beat (no scarcity to exploit), and a
localized patch shift shows the disagreement drive is *blind to confident-wrongness* — it flags
uncertainty, not staleness, so it under-visits exactly the region that needs it. Read together, these
overturn the working assumption that mechanism (which drive, how value re-allocates) should be verified
atom-by-atom before building anything bigger, and the doc names the pivot explicitly: toward
*phenomenon-first* work — get compounding re-adaptation actually working across a sequence of shifts,
then back-translate mechanism from what works. Everything from `meta_adapt/` onward is downstream of
that pivot.

## The value↔FM interface arc: one dissociation, re-derived four times

[`meta_adapt/`](meta_adapt/README.md) reproduces RHM's "meta collapses to multitask" floor on purpose,
as an anchor, then opens a real gap only under actuator-rotation *conflict* (#4). #4b's context latent
`z` produces the arc's load-bearing dissociation — **identifiability and adaptation-benefit are
orthogonal** (`z` always decodes the task; it only pays off under conflict) — which recurs in every
later node under a new name. #4c expects value-directed identification to help and gets a clean
negative: for a 1-D parameter, max-magnitude collection is already maximally informative, which
reframes VoI as something with a *difficulty floor* rather than a universal win. #4d/#4e land the
capacity-competition result the arc needed: value-shaping is real but conditional on capacity
competition existing at all.

[`curiosity_control/`](curiosity_control/README.md) expects the active-vision prior (disagreement
rejects noise) to transfer and gets an inversion instead — on low-dimensional control, ensemble members
fit *different noise realizations* and disagree on aleatoric noise rather than converging past it (the
noisy-TV pathology). The fix is additive grounding, not a smarter filter, and it reproduces #4b's
identifiability⊥value-relevance split on the explore side. Its "ties under static drift, engages under
moving drift" result is the first hint that value's leverage is about *adaptation speed*, not converged
competence.

[`online_value_loop/`](online_value_loop/README.md) takes that hint literally, tries to close both taps
fully online, and gets obstructed on both — CEM-MPC is robust to the exploration drive's real tracking
gains (control as blind grader, again), and the value-shaping gain turns out to be a slow,
commit-dependent transient invisible to any online probe. The obstruction is read as **confirmation**,
not failure: it explains *why* #4e had to be graded offline, and it forces the reframe that value's
payoff is adaptation-speed under drift, which only matters if drift is perpetual — motivating
[`drift_value_loop/`](drift_value_loop/README.md). There, compounding turns out to live in the *memory*
architecture, not the value-carving, and the online loop's obstruction gets re-diagnosed one level
deeper: the missing piece was the **teacher** (grade by value-relevant FM error, the
cerebellum→VTA signal, not by control) — swapping the grader alone produces the clean interior optimum
the prior node couldn't find. "Control is a near-blind grader" is by this point standing doctrine, not
a per-experiment caveat.

## The ballistic arc, and a retraction handled in the open

[`ballistic/`](ballistic/README.md) resolves *why* the FM→behavior bridge kept failing to transmit: it
was never about value, it was about control mode — a committed, feedforward controller is the regime
where a forward model is behaviorally load-bearing, matching Cut #3's biological reading. Its own
caveats named the next question (can a value loop direct *where* to collect), and
[`ballistic/directed/`](ballistic/directed/README.md) built S0→S2 to answer it.

S2 originally reported a real null. That explanation was retracted the same day it was written, after
an internal audit found the null was manufactured: the grading metric was itself a blind control
metric (the very failure `drift_value_loop` had already diagnosed one node earlier), 62% of the loop
score accrued after recovery was already complete, one degenerate seed flipped the aggregate, and — the
decisive bug — every round spent ~2,240 free teleported monitoring transitions against a real budget of
100, a **22× subsidy** that made the claim structurally unmeasurable in either direction. Nothing was
deleted; the wrong claim is struck through in place with the mechanism spelled out, and the file states
plainly that the transferable lesson is the mistake itself — reading a belief downgrade off an
instrument already known to be blind. That became the arc's standing discipline (prefer
value-relevant-error teachers over control metrics, distrust asymptotic scores, charge measurement to
budget before trusting a directed-collection result), and it is cited by name downstream in
[`arm_substrate/`](arm_substrate/README.md) and is the entire reason the [`on_policy/`](on_policy/README.md)
line exists.

## Cut #5: an idea-doc motivation, tested and overturned

[`expansion/`](expansion/README.md) tested the founding idea doc's claim that "drifting dynamics is a
novelty generator" — that perpetual drift alone should grow a forward model's representable-function
space. Rather than risk one null-prone rank measurement (which had already failed three times on this
substrate), the cut calibrated both ends first: an instrument-health check, then a known-real effect
(unlocking degrees of freedom genuinely grows the frontier, +0.72 ± 0.42), then support-fixed drift
measured against that yardstick — landing at ±0.08, ~10× below the calibrated real effect, reproduced
across drift geometries. What survives is the distinction itself, now with a unit: *drift moves the
target function; expansion grows the space of representable functions*, and only the second is
expansion — [`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md) §Scope.
What's explicitly retired is the informal reading that non-stationarity alone is sufficient. The belief
doc's actual core claim (does *novelty* grow representable directions?) was never tested here — a
fixed-DOF plant has no hierarchy to expand into — and is deferred to RHM's hierarchical domain.

[`arm_substrate/`](arm_substrate/README.md), built alongside this, is characterization rather than a
cut: a second task family whose whole point is retiring the single-family caveat and replacing the
pusher's *manufactured* capacity competition with competition intrinsic to the plant (a passive tool,
morphology-driven). It inherits the ballistic-arc discipline by name and pre-commits to *not* fixing
acquisition realism, deliberately leaving that as the next node's job.

## The collection-realism arc: the substrate's last standing unrealism

[`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md), written directly off the S2 retraction,
names the arc's last uncontrolled variable: every FM to this point trained on **teleported** data —
free, omniscient, discontinuous. [`on_policy/`](on_policy/README.md) built the alternative as a flag,
not a second substrate, and pried the claim apart in the order the evidence forced:

- **E0** expected embodiment itself to explain on-policy's apparent efficiency win, and found instead
  that most of it was a **mistuned teleport knob** — an oracle teleporter told where to look closes
  almost the whole gap. What *is* structurally embodied is command-state entanglement, caused by
  continuity itself — which quietly retires Cut #2's i.i.d.-command premise as a teleport-only luxury.
- **E1** tested the natural follow-on hypothesis (on-policy coverage should stretch Cut 4c-arm's
  step-like recovery into a gradual one) and got it **refuted backwards** — the only arm that looks
  gradual is the mistuned default teleporter; a globally-learnable drift re-adapts from any motion.
- **E2** made the drift local, and only then did embodiment become load-bearing: on-policy costs 2.5×
  more transitions, and broad teleport fails outright. The mechanism is bootstrap data-quality (data
  gathered while the model is wrong is itself wrong-distributed), not the coverage-growth story the
  memo had guessed.
- **E3** ([`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md)) is the retracted S2
  cut, re-attempted where the fatal subsidy is structurally gone (the survey itself is metered, ratio
  1.84× not 22×). Both halves of the ladder reproduce this time — the retracted claim was true, it just
  couldn't be measured honestly until acquisition itself was embodied.

The net revision to the substrate's self-model: embodiment's cost is *conditional on structure* — free
under global drift, expensive under local drift — and that conditionality, not embodiment per se, is
now the load-bearing fact every acquisition claim in this tree must state.

## Cross-substrate porting: what travels between mjc and RHM, and what doesn't

[`on_policy/metered_repair/`](on_policy/metered_repair/README.md) (E4/E5) is the newest node and the
first explicit attempt to port two claims wholesale from
[`rhm/directed_sculpting/full_loop/`](../rhm/directed_sculpting/full_loop/README.md) onto this
substrate — a live test of how substrate-general the interface arc's findings actually are. One ports
cleanly: RHM's finding that a fixed-budget counterfactual-fit reducibility tap is *inverted*
(rewards data-starved channels, not reducible ones) reproduces here exactly, and the repaired
(floor-corrected) tap fixes the process metrics — but **the repair doesn't cash out** in outcome, unlike
on RHM. What lands instead is a contrast neither substrate alone could run: with a genuinely
visited-but-irreducible region placed, relevance-only becomes the *worst* arm in the ladder, and only
the conjunction of relevance and reducibility beats it — the two substrates turn out to have
mirror-image degenerate geometries (RHM never has a visited-irreducible cell; mjc never has a
visited-but-irrelevant one), so neither alone could ever show both terms are necessary. E5's necessity
question — does repair migrate from a shared/deep cause to independent surface ones — returns a clean,
mechanically-explained negative: a fixed-DOF plant with spatially-gated fields gives the forward model
no representational route to shared-parameter inference at all, sharpening (not contradicting) the
expansion cut's conclusion that this substrate has no hierarchy to climb.

## Where the priors sit now

The arc's accumulated, load-bearing beliefs, in the order they were forced: **contact is an event, not
a state**; **arity beats resolution under a killed confound**; **committing to a horizon is what makes
a world-model matter, and control that re-plans too fast is a near-blind grader of model quality**;
**identifiability and adaptation-benefit are orthogonal**; **value's leverage is adaptation-speed under
drift, not converged competence, and needs the right teacher (FM error, not control) to show up**;
**directed collection claims are unmeasurable until the looking itself is charged to budget**;
**embodiment's cost is conditional on whether the thing being learned is globally or locally
structured**; and **drift alone does not grow a model's representational frontier — only added degrees
of freedom do, and this plant has none to give**. The live thread is the two open ends of
`metered_repair` — an unexplained allocation result (§4d, no process column predicts the outcome
column) and a necessity question now explicitly deferred to a substrate with a real hierarchy — both
pointing the same direction the expansion cut already pointed: back toward RHM.
