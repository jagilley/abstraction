# History of `experiments/mjc`

A big-picture pass over this node's arc, read from its READMEs and git history, not day-to-day.
Written fresh; see [README.md](README.md) and [FILES.md](FILES.md) for the standing synthesis this
draws on.

## Why this substrate exists

Founded from [`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md)
(2026-07-16) to fill a gap in the existing ladder: RHM/MNIST-reaching gives known latents but
discrete, constructed dynamics; language gives realism but no controllable DGP. MuJoCo sits between
them — continuous, contact-rich, a *real* command channel — but the memo drew one hard line at
founding that the whole node has held to since: **a controllable-dynamics DGP whose knobs get swept
against a vanilla baseline, never an RL-to-SOTA benchmark.** No PPO-maxxing. Every cut below is a
slope or a dissociation, not a leaderboard number.

## Phase 1 — does physical realism survive the toy results? (contact_residual, arity_torque, dynamics_shift)

The first three cuts were a stress test: do the RHM/a2a claims still hold on real contact and torque,
or do they wash out once dynamics stop being hand-constructed? They held, cleanly.
[`contact_residual/`](contact_residual/README.md) found the forward-model residual concentrates
**8.1×** at contact and is an *event* detector, not a *state* detector. [`arity_torque/`](arity_torque/README.md)
reproduced the arity-1 floor under real torque, with the received-wisdom confound (i.i.d. commands)
explicitly killed. Then [`dynamics_shift/`](dynamics_shift/README.md) — Cut #3 — did more than confirm:
by intervening on the *transition operator* rather than sensing or reward, it found reward-free
re-adaptation recovers to the oracle ceiling from ~50 transitions while a matched model-free policy
needs ~240× more reward-labeled data. This is the node's real foundation stone — nearly every later
node builds its substrate on Cut #3's shift, and its `replan_every` ablation (the benefit peaks at the
FM's composition horizon, then decays) is the first hint of the **ballistic vs. reactive** distinction
that would not get named for another dozen nodes.

Early on, one instrument was tried and abandoned: contact_residual's rank-based reading of the
residual did not hold, and that thread stayed live — see the retraction below.

## Phase 2 — the value↔FM interface arc, and a long detour through blind graders

The center of mass of this node's work is the question the idea doc barely poses: *how does a reward
signal shape what a forward model represents, and does it help?* [`value_shaping/`](value_shaping/README.md)
set the **stationary** control first — a value signal re-allocates a capacity-limited FM away from a
reducible-but-irrelevant force field, with an irreducible contact impulse as the foil. Robust core,
fragile "teeth" (capacity efficiency only shows under data scarcity).

[`directed_readapt/`](directed_readapt/README.md) then delivered two negatives that set the tone for
everything downstream: a global shift gives no scarcity to exploit (null), and under a local shift the
disagreement-based exploration drive is **blind to a confidently-wrong region** — it doesn't visit
where it's needed. This is the arc's first "blind grader" — a signal that looks like it should work but
is structurally unable to see the thing that matters — and it triggered an explicit pivot away from
single dissociations toward phenomenon-first work.

[`meta_adapt/`](meta_adapt/README.md) (Cut #4→#4e) is the arc's spine. The 1-D damping floor collapses
(the RHM anchor — meta buys nothing when there's nothing to disambiguate) while an actuator-rotation
*conflict* opens the meta-vs-pooled gap monotonically — the same "which confound is doing the work"
discipline the arity cut used. #4b showed a context latent decodes the task parameter perfectly, yet
system-ID is orthogonal to adaptation-benefit — knowing the parameter and using it are different
achievements. #4c is another scarcity-gated negative (VoI-directed identification can't beat a trivial
heuristic). #4d isolates value-shaping and meta-conditioning as two *separable* capacity levers, only
paying off under capacity competition. #4e closes the loop end to end and surfaces a wireheading
gotcha: an unnormalized weight budget lets the outer loop game loss-scale instead of allocation.

[`curiosity_control/`](curiosity_control/README.md) supplied the afferent half — where to collect, as
a reducible frontier *drifts*. It tracks the moving frontier and helps specifically because the target
is non-stationary (ties on a static one), but on its own it inverts the a2a active-vision result:
disagreement chases aleatoric noise (a noisy-TV pathology) on low-dimensional control. The fix,
grounding with the exploit signal as an additive second drive, is the arc's first constructive resolution
rather than a negative — afferent and efferent wired as one system.

[`online_value_loop/`](online_value_loop/README.md) tried to build the fully-online keystone the idea
doc names, and got obstructed on *both* levers at once: CEM-MPC control absorbs the curiosity drive's
tracking effect, and the value-shaping benefit turns out to be a slow, pre-convergence transient, not a
converged-competence gain. Read as confirmation rather than failure — value is a **slow, committed**
quantity, which is why the two-timescale split is forced rather than a design choice.
[`drift_value_loop/`](drift_value_loop/README.md) took that finding three cuts deeper under perpetual
drift and located the actual missing piece: not the value structure, but the **teacher** — grading the
meta-loop by value-relevant FM prediction-error (not control) gives the explore/exploit balance a clean
interior optimum. Control, again, was the blind grader.

[`ballistic/`](ballistic/README.md) is where the recurring "FM → behavior doesn't transmit" pattern
finally gets a *mechanism*: replanning re-grounds past a stale model every step, so reactive control is
structurally blind to FM quality. A ballistic (feedforward, committed) controller makes the FM ~3×
more behaviorally load-bearing, and reward-free re-adaptation restores ballistic competence ~4.3× more
than reactive after a drift. This reframes several earlier "obstructed" results as artifacts of control
mode, not of the value machinery — and its lead reading (the cerebellar FM as both the
reward-free-maintainable *and* the feedforward-critical asset) is the node's clearest biological payoff.
Its child, [`ballistic/directed/`](ballistic/directed/README.md), tried to land a directed-collection
ladder on top of this — but its S2 cut had to be **retracted**: the per-region monitoring survey was
free and global, a 22× measurement subsidy that made the "where to collect" question unmeasurable. That
retraction, not a positive result, is what set up the next phase.

## Phase 3 — making realism itself a variable (arm_substrate, on_policy, metered_repair)

Two axes had been fixed all along: the plant (one pusher) and how experience gets acquired (free
teleportation). Both got turned into movable knobs. [`arm_substrate/`](arm_substrate/README.md) built a
second task family — an N-link arm — specifically to retire the "single family" caveat and to get
capacity competition intrinsic to the plant rather than manufactured; it verified every precondition
the arc needed (capacity binds only at n≥5, ballistic transmits FM quality 6.0×, composition horizon
14–23 steps) before any cut was ported.

[`on_policy/`](on_policy/README.md), seeded by [`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md),
is the more consequential move: it names every prior result's teleportation as a form of unrealism and
sorts every claim into safe/inflated/unaskable tiers. Its E0–E3 arc is the node's best example of an
instrument correcting itself in public. **E0** found that most of on-policy's apparent advantage was
not embodiment at all but a *mistuned teleport knob* — an oracle teleporter told where to look closed
almost the whole gap; what's genuinely structural is command-state entanglement from continuity itself,
which quietly retires Cut #2's i.i.d.-command premise as a teleport-only luxury. **E1** showed a
*global* drift is repaired equally well under any collection mode — the memo's core hypothesis
("on-policy will stretch the recovery") **refuted**, and the earlier "gradual" recovery diagnosed as
the same mistuned knob showing up in a live cut. **E2** made the drift *local* and finally found the
real cost: on-policy needs 2.5× more transitions, broad teleport never repairs the region at all, and
the mechanism is bootstrap data-quality (bad-model data is wrong-distributed), not the "coverage grows"
story the memo guessed. **E3** ([`directed_on_policy/`](on_policy/directed_on_policy/README.md)) is the
payoff: the retracted `ballistic/directed` S2 cut, re-run with the survey itself metered (1.84× not
22×) — both halves of the ladder reproduce for real, and the on-policy substrate dissolves one of the
two decoys (you only go where you reach, so "visited-but-irrelevant" territory doesn't exist).

[`on_policy/metered_repair/`](on_policy/metered_repair/README.md) (E4/E5, the most recent work, still
warm) is the clearest instance of cross-pollination with [`rhm/`](../rhm/) in this node's history. It
ports RHM's diagnosis that a fixed-budget counterfactual learning-progress tap is *inverted* (measures
data-starvation, not reducibility) — and the diagnosis **ports exactly** (separation flips sign,
matching RHM's finding). But the repair that fixed it on RHM **does not cash out** on mjc (no
significant gain over the unrepaired tap). What the two substrates' complementary blind spots — RHM has
no visited-but-irreducible cell, mjc has no visited-but-irrelevant one — jointly enable is a genuinely
new result neither could show alone: once a visited-but-irreducible region is placed, relevance *alone*
becomes the ladder's *worst* arm, and adding reducibility is what rescues it. E5's necessity question
returns a clean architectural negative: with spatially-disjoint drift, repairing one region does not
repair an identically-drifting other — an MLP over a fixed-DOF plant has no route to shared-parameter
inference. This sharpens, rather than repeats, [`expansion/`](expansion/README.md)'s conclusion below.

## Phase 4 — Cut #5, the "ambitious extension," run and closed

The founding idea doc's most speculative bet was that drifting dynamics is a free novelty generator —
that non-stationarity alone should push a forward model to expand its representable directions.
[`expansion/`](expansion/README.md) finally ran this, after three earlier instrument failures forced a
calibrate→measure→calibrate design rather than a single rank-shaped readout. The result: drift does
**not** open the frontier (±0.08 directions against a +0.72±0.42 calibration from a genuine support
increase — ~10× smaller). The idea doc's own motivating paragraph has since been edited in place to
mark this **retired**: drift moves the target function, expansion grows the space of representable
functions, and a fixed-DOF motor plant never has the latter to offer. What survives is the *realism*
argument for drift (Type-2 non-stationarity, load-bearing for the whole `drift_value_loop`/`on_policy`
line) — only its special status vis-à-vis representational growth is gone.

## Recurring shapes, visible only at this distance

- **Blind graders, everywhere.** Disagreement blind to a confident-wrong shift
  ([`directed_readapt/`](directed_readapt/README.md)), control blind to converged FM quality
  ([`online_value_loop/`](online_value_loop/README.md), [`drift_value_loop/`](drift_value_loop/README.md)),
  replanning blind to a stale model by construction ([`ballistic/`](ballistic/README.md)), and ballistic
  control itself blind in [`metered_repair/`](on_policy/metered_repair/README.md) E4 (near-saturated,
  forcing every conclusion onto the FM-error grader alone). The node's single most repeated move is
  swapping the grader, not the mechanism, to unblind a result.
- **Mistuned defaults masquerading as findings, twice.** [`on_policy/`](on_policy/README.md) E0 and E1
  both found that what looked like a structural embodiment effect was a teleport sampling knob nobody
  had tuned to the task. The fix (`teleport_matched`) is now a standing recommendation for any future
  teleport cut.
- **Rank/triple readouts don't survive contact with this substrate.** contact_residual's early rank
  angle failed; the `(R_act, R_comp, R_res)` triple written into the original Cut #5 spec was formally
  retracted (2026-07-26, [see commit `40a6826`](../../experiments/mjc)) once direct measurement on
  a2a/RHM showed `R_comp` isn't independent of `R_act`. Both times the fix was to fall back to a
  behavioral or gap-width readout rather than patch the rank instrument.
- **Duplication over edits, consistently.** New cuts are copy-and-modify descendants of their parent
  script (`metered_repair`'s scripts vs. `directed_on_policy`'s) rather than in-place edits — the
  node's applied version of "keep prior results reproducible."
- **Cross-substrate pollination is now routine**, not incidental: `expansion/` imports the RHM frontier
  instrument directly; `metered_repair/` ports both a diagnosis and a necessity design from
  `rhm/directed_sculpting/full_loop/`. The two DGPs are increasingly read as one program with two
  instruments, each catching what the other's degenerate geometry cannot.

## Where the live thread stands

The open edge, per [README.md](README.md) "Next steps" and `metered_repair/`'s own next steps: explain
why E4's allocation outcome isn't predicted by any process column (a scheduled-burst control is
proposed to test whether the ladder has been measuring timing rather than reducibility); give the FM a
route to a genuinely shared parameter before re-asking E5's necessity question; and back-translate the
interface arc's headline findings (value is slow/committed; the teacher, not the structure, was
missing; control mode gates transmission) into the belief tree, which has not yet happened.
