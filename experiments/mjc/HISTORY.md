# HISTORY — experiments/mjc

A big-picture pass over how thinking evolved in the `mjc` node, read primarily from
[`README.md`](README.md), [`FILES.md`](FILES.md), each child's own README, and the founding idea/belief
docs. Day-to-day detail lives in those files; this traces the shape of the arc and the moments priors
actually moved. (Git history for this node is shallow-cloned back only to 2026-08-08, so the timeline
below is reconstructed from dates and status notes embedded in the docs themselves, not from commit
history.)

## Why this substrate exists

[`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md) (2026-07-16) founded
`mjc` to fill a gap in the substrate ladder: RHM is discrete with a fully known DGP, language is opaque,
and nothing sat between them offering continuous, contact-rich dynamics with a real command channel. The
founding discipline — stated once and never relitigated — is that MuJoCo is a **knob-sweep DGP
instrument**, not an RL-to-SOTA robotics benchmark: policies are obtained the cheap way, every experiment
reports a slope or dissociation against a vanilla baseline, never a single leaderboard number. That
discipline is why the whole arc below reads as a sequence of *controlled comparisons* rather than a
capability chase. The doc explicitly frames `mjc` as testing a2a_forward/RHM claims (arity, veridicality-
vs-control, residual-as-DGP-complexity) on physics instead of toy data — this node was never meant to
stand alone.

## Foundations: establishing the substrate is trustworthy (Cuts #1–#3)

[`contact_residual/`](contact_residual/README.md), [`arity_torque/`](arity_torque/README.md), and
[`dynamics_shift/`](dynamics_shift/README.md) spent the substrate's first capital proving the physics
behaves the way the toy substrates predicted before building anything on top: residual concentrates at
contact events (not contact *states*), arity beats resolution once the generator confound is killed, and —
the pivotal move — **operator intervention** (shifting the transition function while holding the task
fixed) lets a model-based agent recover from a handful of reward-free transitions while a motor-program
policy needs ~240× more reward-labeled data. Cut #3's `replan_every` ablation, tying recovery benefit to
open-loop commitment length, is the seed of everything the `ballistic/` line later becomes. This is the
substrate every later node still calls "the base."

## The value↔FM interface arc: from a stationary control to a forced architecture

[`ideas/two_timescale_value_loop.md`](../../ideas/two_timescale_value_loop.md) (2026-07-17/18) supplied
the hypothesis this arc spends the rest of the node testing: that value is a forward model read out
through a reward dimension, and the missing piece of the program was a slow, non-stationary outer loop
shaping *what the FM models*. Its own revision trail is instructive — an early "high-rank = noise" mapping
was retracted as a category error, non-stationarity was split into Type-1 (epistemic, self-terminating)
vs. Type-2 (environmental, unbounded), and a later addendum retracted the claim that meta-learning strictly
*requires* world motion (the real gate turned out to be grader type and sample price).

The empirical arc tracks that hypothesis maturing under contact with data:
- [`value_shaping/`](value_shaping/README.md) established the **stationary control** — value re-allocates
  a capacity-limited FM away from a reducible-but-irrelevant factor. Nothing moving yet; this is the
  yardstick everything non-stationary below is measured against.
- [`directed_readapt/`](directed_readapt/README.md) produced two informative negatives (a global shift has
  no scarcity to exploit; disagreement-driven exploration is blind to a confident-but-wrong prior) that
  explicitly **motivated a pivot to phenomenon-first work** rather than chasing a single directed-collection
  win.
- [`meta_adapt/`](meta_adapt/README.md) (#4→#4e) is the arc's spine: the damping floor collapses (an RHM
  anchor) while an actuator-rotation conflict opens a meta-learning gap monotonically; a context latent
  decodes the task *without* that decoding predicting adaptation benefit (system-ID ⊥ usefulness); value-
  directed identification is a robust scarcity-gated negative; value-shaping and meta-conditioning turn out
  to be **separable capacity levers**; and closing the reward loop (#4e) both prefers the informative
  regime and rediscovers value's support on its own, with a wireheading gotcha (normalize the weight
  budget) as the price of admission.
- [`curiosity_control/`](curiosity_control/README.md) added the *afferent* complement — an intrinsic
  surprise drive that tracks a moving reducible frontier — and surfaced a **noisy-TV pathology** on
  low-dim control (a substrate inversion of the a2a active-vision result) that only grounding (explore +
  exploit as two additive drives) fixes.
- [`online_value_loop/`](online_value_loop/README.md) tried to build the fully-online two-timescale loop
  the idea doc named, on both levers, and got **obstructed on both** — CEM-MPC is robust to the drive's
  tracking effect, and the value-shaping benefit turned out to be a pre-convergence transient, not a
  converged-competence gain. This negative result is treated as a *positive confirmation*: value is a
  slow/committed quantity, so the two-timescale split is forced by the data, not a design choice.
- [`drift_value_loop/`](drift_value_loop/README.md) took that obstruction three cuts deeper under
  perpetual drift and found the fix wasn't the value structure at all — it was **the teacher**. Grading the
  meta-loop on value-relevant FM prediction-error (the literal cerebellum→VTA signal) rather than control
  gives the explore/exploit balance a clean interior optimum. This is the hinge that hands off directly to
  the performance-error-bridge arc below.
- [`ballistic/`](ballistic/README.md) resolved *why* the FM→behavior bridge kept failing to transmit under
  reactive control: replanning is a blind grader that re-grounds past a stale model every step. A
  feedforward, committed controller is not a variant worth trying — it is the regime that makes the FM
  behaviorally load-bearing at all, and its child [`directed/`](ballistic/directed/README.md) lands the
  directed-collection ladder the earlier negatives couldn't.

## The realism axis: what the substrate had been quietly assuming

Two nodes exist because the substrate's early convenience turned out to be hiding real content.
[`arm_substrate/`](arm_substrate/README.md) added a second task family specifically to retire the
"single-family" caveat on the ballistic claims and to get capacity competition that comes from the
*plant's morphology* rather than a hand-tuned force field. [`on_policy/`](on_policy/README.md), grounded in
the standing memo [`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md), audited the substrate's
default **teleport** collection (`set_state` to anywhere, free and omniscient) and sorted every prior claim
into safe / inflated / unaskable tiers. The finding that reorganized the convention: most of on-policy's
apparent advantage was a *mistuned teleport knob*, not embodiment per se — but under **local** drift,
embodiment becomes genuinely load-bearing (on-policy needs 2.5× more transitions; broad teleport never
repairs the region). That dissociation — embodiment is nearly free when structure is global, expensive
when local — is now the field's working answer to "does it matter how you collect," and on-policy
collection is the adopted default for new cuts (teleport stays the code default, deliberately, for
backward compatibility).

## Cut #5: the flagship that closed by falsifying its own premise

[`expansion/`](expansion/README.md) was billed in the founding memo as "the single biggest open question
of the whole program" and sat unrun for months. When finally run (calibrate→measure→calibrate, after three
prior rank-shaped instruments had failed here), the result was unambiguous: drifting a plant's physics does
**not** open the forward model's representational frontier (±0.08 directions against a genuine +0.72±0.42
calibration, ~10× below a known-real effect). This is not a local negative — it forced a scope correction
in [`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md) itself: representational
expansion needs *hierarchical* structure to expand into, which a fixed-DOF motor plant structurally lacks,
so the null is health, not failure. The founding memo's "drift is a novelty generator" motivation is
explicitly retired going forward; drift remains essential for *realism* (it is what the whole
`drift_value_loop`/`on_policy` line depends on), just not for representational growth. This is the
clearest instance in the node of a belief doc and an experiment correcting each other in both directions
in the same week.

## The performance-error bridge: naming the signal, then testing it three ways

[`ideas/performance_error_is_the_bridge.md`](../../ideas/performance_error_is_the_bridge.md) (2026-08-11)
picked up where `drift_value_loop`'s "the teacher, not the structure" finding left off, naming the bridge
signal biologically (Gadagkar et al.'s songbird VTA performance-error neurons: a baseline-subtracted,
agency-gated FM error). The doc's own construction is unusually self-correcting in real time — it
supersedes an earlier draft with an inverted sign, and three same-day rounds of runs falsify or narrow
specific sub-claims while confirming the core two-channel architecture. Three nodes tested it same-day:
[`agency_gate/`](agency_gate/README.md) showed the agency gate is load-bearing (self-produced vs. passive
playback, ~369× difference) with a real bug caught and fixed (the gate needs centering);
[`plasticity_gain/`](plasticity_gain/README.md) showed the gain actually works as a per-sample
consumption mechanism, with its real content in *allocation*, not ceiling performance; and
[`curiosity_control/benchmark_vs_cost/`](curiosity_control/benchmark_vs_cost/README.md) showed a standing
cost term is unsupported — "frustration" falls out of the signal as a transient rather than needing to be
built in. [`two_clocks/`](two_clocks/README.md) then confirmed the idea doc's riskiest structural bet
(performance-error and reward are non-redundant channels that *destructively interfere* if shared), and
[`bridge_assembly/`](bridge_assembly/README.md) assembled the corrected signal live, landing a
transmission-inversion law (localized FM differences reach behavior through ballistic control, diffuse
differences through reactive control) as the arc's most portable output.

## Practice as a third, separate mechanism

[`practice/`](practice/README.md) and its child [`practice/etude/`](practice/etude/README.md) opened a
third axis late in the arc: not what to collect or how to grade it, but how a *sequence* of skill gets
compiled from practice into performance. The étude sub-arc (11 runs, each correcting the last) landed on
"compilation is selection, not averaging" — committing a single well-chosen realized trace beats fitting
an average of many — and that assembling segments in piece order, scored against seam-matched hand-over
states, collapses an optimism gap that a winner's-curse-style naive selection had introduced. Two
intuitive mechanisms tested here — signal fusion across compiled segments, and dedicated seam drills — came
back null, which the writeup treats as informative rather than embarrassing. δ appears in this arc only as
a compile-trigger detector, a deliberate narrowing from its plasticity-gain role elsewhere in the node.

## Recurring shape of the thinking, across the whole arc

A few patterns repeat often enough to be worth naming as the node's actual epistemic style, more than any
single result:
- **Instructive negatives redirect rather than stall.** `directed_readapt`, `online_value_loop`, and
  `expansion` are all "the obvious thing didn't work," and each time the response was to ask what the
  negative result was itself evidence of, not to retry harder.
- **Convenience gets audited, not assumed.** Teleport collection was the default for most of the node's
  life and was never treated as free until `COLLECTION_REALISM.md` explicitly priced it — a pattern of
  going back to check what a founding simplification cost.
- **Belief docs and experiments correct each other bidirectionally.** `dimensionality_expansion.md` and
  `expansion/` is the cleanest case, but `two_timescale_value_loop.md`'s and `heterogeneous_graders.md`'s
  own dated revision logs show the same loop running throughout — an idea doc making a prediction, an
  experiment landing, and the doc being edited in place rather than superseded quietly.
- **Mechanisms outlive the motivations that launched them.** Drift is still central to the substrate after
  `expansion/` retired the reason it was originally introduced; the ballistic/reactive control-mode
  distinction outlived the specific value-shaping question it was built to answer and became the node's
  general transmission law.

## Where the thread stands

Per [`README.md`](README.md) §Next steps, the live question is whether a directed-collection loop's
relevance term pays off once collection is genuinely embodied and metered (re-attempting the retracted
`ballistic/directed` S2 question on the local-drift on-policy substrate), and whether ensemble disagreement
— which `online_value_loop` found blind because every ensemble member had data everywhere — recovers its
detection job once on-policy collection creates real unvisited territory. The node also carries two
explicitly unimported findings worth remembering: the belief that novelty grows representable directions
has *still* never been directly measured anywhere (RHM's novelty arm swapped rule sets wholesale, which
measures forgetting, not expansion), and the performance-error bridge's transmission-inversion law has not
yet been back-translated into the belief tree.
