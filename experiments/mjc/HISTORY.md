# HISTORY — experiments/mjc

A qualitative, "big history" pass over this node's git-visible record and its README/FILES
hierarchy. Rewritten fresh each run — see [FILES.md](FILES.md) for the file-by-file index and
[README.md](README.md) for the current-state summary this document does not duplicate.

## Why this substrate exists

[`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md) (2026-07-16)
opens with a ladder whose middle rung was empty: RHM has a known DGP and true latents but
*constructed* commands; language has real continuity but no dial to turn. MuJoCo was picked as
the first substrate where a command genuinely controls the state as a physical fact, not by
authorial fiat — the precondition a prior "sensing is a trap, you need control" pivot demanded.
The doc is unusually candid about the price: no latent tree, no DP-optimal policy, so this is "a
*second* substrate, not a migration" from RHM. Its own subtlest hedge — that MuJoCo's realism
"is smaller than it looks *for our questions*" — turned out to be load-bearing twice over, in the
teleport reckoning and the expansion cut below.

## The three foundations, and one reframe

Cuts #1–#2 confirmed as pitched: contact produces an *event*, not a *state*, signature in FM
residual ([`contact_residual/README.md`](contact_residual/README.md), Huber-over-MSE the
governing gotcha), and under i.i.d. commands arity beats resolution outright
([`arity_torque/README.md`](arity_torque/README.md)). Cut #3 — intervening on the transition
operator itself while holding the task fixed — is where the program found its real center of
gravity: an MB agent recovers from ~50 reward-free transitions where a committing MF agent needs
~240× more reward-labeled data ([`dynamics_shift/README.md`](dynamics_shift/README.md)). It was
also **reframed mid-flight**, from a degradation-slope claim to a factorization claim, once
per-step replanning made the original framing a near-null — the first instance of a pattern that
recurs across the whole node: a result that looked like a null was actually pointing at the
*control mode* as the missing variable, not at the phenomenon being absent.

## The value↔FM interface arc: a six-node argument with itself

[`value_shaping/`](value_shaping/README.md) established the stationary control — value
re-allocates a capacity-limited FM away from a reducible-but-irrelevant factor — before anything
non-stationary was attempted. [`directed_readapt/`](directed_readapt/README.md) then produced two
*instructive* negatives rather than one clean win: a global shift has no scarcity to exploit, and
a disagreement-based drive is blind to a confident-*wrong* region under a local one. That
motivated an explicit pivot to phenomenon-first work: [`meta_adapt/`](meta_adapt/README.md)'s
#4→#4e arc is the node's most-revised single thread — a 1-D floor collapse anchored against RHM,
an actuator-rotation conflict that opens the gap monotonically, a context latent that decodes the
task perfectly yet is *orthogonal* to adaptation-benefit, a value-directed identification drive
that is a robust scarcity-gated negative, and only at #4d do value-shaping and meta-conditioning
resolve into two separable capacity levers rather than one confused signal.
[`curiosity_control/`](curiosity_control/README.md) supplied the afferent complement and found its
own pathology — disagreement chasing a noisy TV, a substrate-inversion of the a2a active-vision
result — cured only by grounding explore against exploit as two additive drives.

The keystone the idea docs had been pointing at — a fully online, two-timescale value loop — was
then built and **obstructed on both levers** in
[`online_value_loop/`](online_value_loop/README.md): CEM-MPC is robust to the tracking drive, and
the value-shaping benefit turns out to be a pre-convergence transient, not a converged-competence
effect. Read straight, this looks like a negative result. Read as the arc reads it, it is a
positive confirmation — value is a *slow, committed* quantity, so the two-timescale split is
forced by the phenomenon, not a design choice — and it named the next experiment.
[`drift_value_loop/`](drift_value_loop/README.md) ran that experiment three cuts deep and located
the actual missing piece: not the value *structure*, but the *teacher* — grading the meta-loop by
FM prediction-error (the literal cerebellum→VTA signal) rather than by control gives the
explore/exploit balance a clean interior optimum where grading by control did not.

## The ballistic pivot

The recurring "control is a blind grader" note from cuts #3 and drift_value_loop crystallizes in
[`ballistic/README.md`](ballistic/README.md): replanning re-grounds past a stale model on every
step, so reactive control structurally cannot see what the FM is worth. A feedforward, committed
controller — the biologically forced regime, since nothing can replan faster than sensorimotor
delay — makes the FM ~3× more behaviorally load-bearing, and after a drift, online reward-free
re-adaptation restores ballistic competence ~4.3× more than reactive. The reading that survives
into everything downstream: the cerebellar FM is the reward-free-maintainable asset *and* the
feedforward-critical asset, because they are the same object. This is also where the substrate's
composition horizon — the step count past which a plan stops tracking the plant — first becomes
a load-bearing measured quantity rather than a design parameter, foreshadowing the practice arc.

## The confound the node built for itself

By late July, every FM in this node had trained on **teleported** data — `set_state` to an
arbitrary posture, apply an arbitrary command, record one triple: free, discontinuous, omnisciently
covering. [`on_policy/COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md) is the node
auditing its own method rather than a new experiment's negative result, and it did so honestly: it
sorted every prior claim into three tiers — dissociations stay safe (the subsidy cancels on both
arms), absolute sample counts are upper bounds not estimates, and allocation questions were
outright unaskable (one retracted null had spent a 22× measurement subsidy without knowing it). The
fix was a flag, `collection_mode`, backed by a metered `Body` that cannot expose `set_state` — one
substrate, not two — adopted 2026-07-23 as the default for new cuts while the code default stayed
teleport so nothing already finished lost reproducibility. Pricing it (E0–E3,
[`on_policy/README.md`](on_policy/README.md)) found the honest shape of the cost: embodiment is
nearly free when the drift is global (Cut 4c-arm reproduces at 4.8–5.5× under every mode) and
expensive when it is local (2.5× more transitions, because bootstrap data quality — not coverage —
is what a wrong model corrupts). [`arm_substrate/`](arm_substrate/README.md) then retired the
single-plant-family caveat this whole arc had been carrying, confirming ballistic's asymmetry and
the capacity results transfer to a second, morphologically different plant.

## Cut #5: a founding motivation, narrowed by its own belief tree

[`expansion/`](expansion/README.md) tested the substrate memo's own flagship pitch — that drifting
dynamics is a free novelty generator, and novelty should grow the FM's representable space — and
had to build a calibrate→measure→calibrate instrument because naive rank readouts had already
failed three times on this substrate. The result: drift moves the target function
(±0.08 directions) but does not open the frontier, ~10× below a genuine support-growth calibration
(+0.72 ± 0.42). This confirms, rather than merely cites,
[`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md)'s own §Scope
amendment — expansion needs hierarchical structure to expand *into*, and a fixed-DOF plant has
none — and it retires the founding memo's "drift as novelty generator" claim by name, while
explicitly preserving drift's independent role as *realism* for everything the value-loop and
on-policy lines depend on. The belief's actual core claim — does novelty grow representable
directions — is left, correctly, untested here: it needs a domain with structure to climb, which
this substrate was never going to have.

## The performance-error bridge, tested against its own idea doc

[`ideas/performance_error_is_the_bridge.md`](../../ideas/performance_error_is_the_bridge.md)
supplied its own falsification and rescue in the same document: §12 shows Experiment 1 was a
no-op (the benchmark term already existed as a REINFORCE baseline; the real obstruction was the
search procedure), while §13–14, run the same session, show the *gate* is not decorative.
[`agency_gate/README.md`](agency_gate/README.md) found δ fires ~369× more under self-produced than
passively-replayed identical sensory sequences — the gate, not the sensory stream, carries the
discrimination — but also that the doc's own σ(g/θ) leaks at g=0 and needs centering.
[`plasticity_gain/README.md`](plasticity_gain/README.md) then found δ's real content is in
allocation and retention, not raw adaptation speed, and only works as a benchmark once it is
context-conditional. [`two_clocks/README.md`](two_clocks/README.md) landed the idea doc's §6 bet
in a stronger form than pitched: sharing an eligibility window between performance-error and
reward is not merely suboptimal, it is *destructive interference* — every shared variant loses to
both single channels. [`bridge_assembly/README.md`](bridge_assembly/README.md) assembled the
corrected signal live and surfaced the arc's transferable law: which controller transmits an FM
difference depends on the difference's *spatial structure* (localized → ballistic, diffuse →
reactive) — the "blind grader" pattern from cuts #3 and ballistic, now stated as a matching
condition rather than a failure mode.

## The practice arc — the live thread (2026-08-12 → 08-20)

This is where git activity concentrates and where the thinking is currently moving.
[`practice/README.md`](practice/README.md) opened as an audit of δ-as-consumption and mostly
produced facts about its own apparatus — Adam is invariant to global loss rescaling (so a "matched
budget" claim resting on a weight normalizer never bound), the calibrated gate is binary rather
than graded, and a Pareto-hull comparison flipped sign on grid coarseness alone. The verdict
carried forward: **δ's surviving role is detection, not consumption.**

With consumption demoted, the question shifted from *which samples to learn from* to *what to
commit*. [`practice/etude/README.md`](practice/etude/README.md) tried three successive forms of
the compile operator — regression, then selection ("the songbird crystallises a rendition; it
does not average the babble"), then expected-performance selection under the consumption
distribution — each correction forced by a measured failure (a winner's-curse effect, a seam-state
shift). Sequential assembly with seam-matched selection beat never-compiling on both axes, 3/3
seeds, but the arc's real yield was a degeneracy: fusion was provably vacuous because committed
units were state-independent, giving the mandate that shaped everything after — **hierarchy is
only meaningful over boundaries that carry information.**

[`practice/fingering/README.md`](practice/fingering/README.md) ported the question to a redundant
arm where arrival posture makes seams informative, reproduced the RHM practice-arc laws, and then
inverted the étude's own conclusion: content frozen at commit time rots, but *live* content under
committed *routing* wins — the lesson was never that committing is corrosive, only that committing
frozen bytes is. [`practice/legato/README.md`](practice/legato/README.md) found that law is
horizon-local: past the plant's own composition horizon, frozen measured chains win because they
were produced by the body and carry no composition error, while live plans win only inside the
model's reach. [`practice/span/README.md`](practice/span/README.md) then re-read the horizon
itself as a trajectory rather than a fixed number — the FM's representable reach keeps expanding
under practice while a live plan's actual reach does not follow it (believed vs. true plan error
correlate at r≈0.01) — closing the loop back to the expansion cut's distinction between moving a
target and growing a frontier, this time inside a single controller's own planning horizon rather
than across an FM's training.

## What actually changed across this record

Read end to end, the node's standing epistemic habit is not "run the experiment," it is
"distrust the instrument before trusting the result": contact residual needed a directional
control before AUC meant anything; expansion needed three failed instruments before a calibrated
one; on-policy discovered its own teleport subsidy; the practice arc discovered its own gate was
binary and its own hulls were grid-sensitive. The substrate's founding bet — that a controllable
physical DGP would let control-oriented claims (arity, composition, robustness) be tested with the
same variable-discipline RHM affords — held up, but nearly every specific hypothesis it was built
to test (novelty-driven expansion, disagreement-driven exploration, teleport realism, δ as a
learning-rate gain) survived only in a *narrower, more structural* form than first pitched, with
"the control mode is a blind grader" as the single recurring reframe underneath most of them.
