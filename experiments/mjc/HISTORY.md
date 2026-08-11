# HISTORY: `experiments/mjc/` — the MuJoCo control substrate

*A big-picture reconstruction from README prose, not git log — this directory's entire tree landed
in one squashed commit (`ea372ae`, 2026-08-02), so the story below is read out of the documents
themselves, which is itself evidence of how disciplined the write-as-you-go convention has been here.*

## Why this substrate exists

[`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md) (2026-07-16),
prompted by a look at Sunday Robotics' ACT-2 preview, proposed a rung between RHM and language:
continuous dynamics, real contact, no ground-truth latents, but still a **controllable DGP** whose
knobs (friction, mass, gear, arena) can be swept the way RHM sweeps `(v,s,L,m)` — explicitly *not*
a robotics benchmark to RL-to-SOTA. The doc names five candidate cuts and, notably, already flags
cut #5 (drifting dynamics → representational expansion) as "the single biggest open question of the
whole program." That framing — written before a line of `mjc` code existed — turns out to matter a
lot for how the arc eventually closes (see Expansion, below). The doc also hedges honestly about
failure modes (low-dim fully-observed state letting search substitute for a model); that particular
worry never materialized, but other surprises did, in places the doc didn't anticipate.

## Foundational cuts: proving the substrate is honest (Cuts #1–#3, mid-July)

[`contact_residual/`](contact_residual/README.md) (Cut #1) ported the a2a residual-concentration
story onto physics and sharpened it into something more specific than expected: not "residual is
higher near contact" but an **event detector** — it spikes at the free→contact transition and decays
within a handful of steps even while contact persists. A companion hypothesis ("contact residual is
lower-rank") was cleanly falsified (contact eff-rank *higher* than free, not lower) and dropped
without ceremony. The load-bearing lesson was methodological: MSE lets contact's heavy tail starve
free-flight learning entirely; Huber is required just to see the effect. [`arity_torque/`](arity_torque/README.md)
(Cut #2) moved the arity thread from RHM's synthetic actuators onto a real one, and its main labor
was killing a confound rather than discovering a phenomenon: commands had to be made genuinely
i.i.d. of state (`max|corr(u,s)|=0.003`), or an arity-1 model could silently reconstruct `u` from `s`
and fake the gap. [`dynamics_shift/`](dynamics_shift/README.md) (Cut #3) is where the project first
pivots away from an older, shakier frame — the a2a "static degradation slope" — toward re-adaptation
*efficiency* as the dissociation that matters. An early per-step-replanning run came back a near-null;
the fix (committing MPC to `replan_every`-step open-loop segments) is what made the world-model
load-bearing at all, and is explicitly named as the seed of the entire later ballistic arc.

## First pivot: two negatives end "atom verification" (2026-07-18)

[`value_shaping/`](value_shaping/README.md) needed two redesigns before it worked — a naively heavy,
frequently-hit puck taught the lesson that *energy is not prediction cost*, and that an irreducible
foil (Cut #1's contact aliasing) can't be "reallocated away from" because no capacity was ever spent
on it in the first place. Once fixed, the capacity-reallocation effect was real but its "teeth" (the
efficiency payoff) turned out to be a data-scarcity artifact, reported as such rather than inflated
into a headline. [`directed_readapt/`](directed_readapt/README.md) then delivered two clean negatives
for disagreement-directed exploration: a global shift gives the drive nothing to exploit (null,
expected), but a *localized* shift produces a sharper and more interesting failure — ensemble
disagreement is **blind to a confident-prior shift**, because an ensemble that never visited the
patch agrees (wrongly, confidently) there, so the drive that should chase disagreement under-visits
exactly the region that's broken. The README states the resulting decision directly: two negatives in
a row on isolated mechanistic atoms, with diminishing returns, triggered an explicit switch to
**phenomenon-first** work — stop verifying atoms in isolation, go build the actual meta-loop
[`meta_adapt/`](meta_adapt/README.md) and see what happens.

## The value↔forward-model interface arc (Cuts #4–#4e, 2026-07-18 to 21)

This arc opens skeptical, not hopeful — a prior RHM result had already shown meta-learning collapsing
to multitask pooling absent real task conflict, so #4 tracks "meta − multitask gap" as a falsifiable
quantity. The floor condition (1-D damping) **confirms** the skeptical prior (gap ≈ 0) and is read not
as failure but as diagnosis: task conflict, not architecture, was the missing ingredient — which #4's
actuator-rotation condition supplies, opening the gap monotonically with conflict Φ. #4b's context
latent `z` was added specifically because Reptile's task memory is hidden in the weights and
unprobeable; the payoff was a genuine dissociation — `z` decodes the task almost perfectly even where
the adaptation benefit is ~0, i.e. knowing the task can be *true but useless*, a distinction that
recurs for the rest of the arc. #4c is a second robust negative (no VoI-directed identification drive
beats trivial max-magnitude collection on this low-dim system-ID problem), explicitly scoped as "VoI
is over-engineering here." #4d/#4e are where the causal question — does value actually shape the FM,
or only correlate with it — finally gets answered, but not on the first try: an early design collapsed
to a vacuous loss, and even the redesign showed only a neutral "wash" on control until a genuinely
competing force field forced real capacity competition, at which point a small but 4-seed-robust
control benefit appeared. #4e closes the loop with a reward-driven outer weight and catches a
wireheading bug (an unnormalized weight lets the optimizer win by gaming loss scale, not by
reallocating capacity) — a second instance of a metric almost producing a false conclusion.

[`curiosity_control/`](curiosity_control/README.md) supplies the arc's afferent half and inverts an
earlier precedent: active-vision's ensemble disagreement had *rejected* noise; here, on low-dimensional
control, disagreement instead **chases** a noisy-TV distractor. Grounding curiosity with the exploit
signal as an *additive*, not gating, drive fixes the pathology. [`online_value_loop/`](online_value_loop/README.md)
is the arc's sharpest turn: both attempts to make the interface fully online — self-tuning
explore/exploit, self-tuning capacity allocation — get obstructed (CEM-MPC is robust to the drive's
tracking signal; the value-shaping benefit is a slow, commit-dependent transient invisible to fast
local probes), and this obstruction is explicitly read as **confirming** the two-timescale hypothesis
rather than refuting the whole program. The project's target metric pivots here, from converged
competence to *adaptation speed*. [`drift_value_loop/`](drift_value_loop/README.md), the direct sequel,
delivers the arc's final settled belief: compounding across drift is real, but its source is the fast
context-latent memory, not the slow value-carving (which turns out capacity-gated and control-robust —
a null). The deciding move is diagnosing that the meta-loop's earlier inertness was a **teacher**
problem, not a value-structure problem: grading the outer loop by FM prediction error rather than
control performance produces a clean interior optimum where grading by control was flat.

## Ballistic control and the collection-realism reckoning (2026-07-22 to 24)

[`ballistic/`](ballistic/README.md) resolves the "replanning is a blind grader" puzzle as a fact
about *control mode*: a feedforward, committed controller makes FM quality ~3× more behaviorally
load-bearing than a reactive one, and reward-free re-adaptation restores ballistic competence ~4.3×
more. Its first attempt at directing collection (Cut 4a) failed from a geometry confound and was
explicitly dropped rather than patched — a decision later reread, in the memo below, as "the warning
from the one time we tried an embodied drive and abandoned it by going disembodied instead of fixing
the geometry." [`ballistic/directed/`](ballistic/directed/README.md)'s S2 experiment is the arc's most
consequential epistemic event: it first reported a clean null (a relevance-weighted collection signal
never beat plain reducibility-chasing), then a self-audit (2026-07-24) **retracted the null outright**
— the grader used was one the tree had already shown to be blind, most of its score accrued after
recovery was already complete, the ranking flipped under a sighted metric, and every policy had been
quietly subsidized by ~2,240 free teleported probe transitions per round. The corrected status was
downgraded from "relevance doesn't pay" to simply **untested**.

That audit produced [`on_policy/COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md), the closest
thing this node has to a constitutional turning point: every FM in the substrate had been trained on
teleported (`set_state`) data — free, discontinuous, omnisciently covering — properties no real body
has, and the memo sorts every prior claim into three tiers: **safe** (dissociations/ratios, where the
coverage advantage cancels across both arms of a comparison), **inflated** (absolute sample counts,
which are upper bounds, not estimates), and **unaskable** (anything whose dependent variable is the
allocation of experience itself, since free looking has no cost to weigh against). Rather than
retrofitting every past cut, the memo's fix — a `collection_mode` flag, a metered `Body` with no
`set_state` exposed, teleport kept bit-identical and default for reproducibility — became the new
convention for cuts going forward, not a repudiation of what came before.

[`on_policy/`](on_policy/README.md)'s own experiments (E0–E2) then repeatedly **overturn the memo's
own guesses about itself**: E0 found most of on-policy's apparent advantage was a *mistuned teleport
knob*, not embodiment; E1 explicitly refutes the memo's live hypothesis that on-policy would smooth
Cut 4c-arm's step-like recovery into something graded (it doesn't, under global drift); E2 then
narrows the true effect to something sharper than either guess — embodiment is nearly free when drift
is global and expensive (2.5× more transitions, and unrepairable under broad teleport) only when
drift is local, via a bootstrap data-quality mechanism rather than a coverage story. [`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md)
(E3) finally resolves the S2 retraction on ground where the question is askable: with the monitor
survey itself metered, the relevance term the disembodied version could never validate now wins in
3/3 seeds — the original intuition was right, it just needed a substrate honest enough to test it on.

## Cut #5 closes: expansion, and retiring its own founding premise (2026-07-27)

[`expansion/`](expansion/README.md) finally runs the question the founding idea doc called the
program's biggest open question, unrun for months. Because rank-shaped instruments had already failed
three times in this repo, the cut is built as a deliberate **calibrate → measure → calibrate**
sandwich rather than a single readout. The result: drift does *not* open the forward model's
representable frontier (±0.08 directions, ten times below a genuine, calibrated expansion signal of
+0.72±0.42), de-confounded against a drift-geometry difficulty confound the team caught and fixed
mid-cut. The parent README is explicit that this isn't a defeat of "expansion" as a concept but a
**scope correction** to the founding doc's own 2026-07-16 framing: drift moves the target function a
fixed-DOF plant already spans; expansion needs a domain with hierarchical structure to expand
*into*, which a motor plant structurally lacks. That belief-level correction is now written back into
[`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md) §Scope, and the
open question itself is redirected toward [`rhm/residual_decomposition/`](../rhm/residual_decomposition/README.md),
a domain that actually has the hierarchy this substrate doesn't.

[`arm_substrate/`](arm_substrate/README.md), built alongside, retires the ballistic arc's
single-family caveat and replaces the pusher's *manufactured* capacity competition with competition
intrinsic to a real plant — itself full of small corrected assumptions (capacity does not bind at 2
links, contrary to the obvious guess, only at n≥5; an apparent substrate failure at one probe turned
out to be an under-sized planner, not the substrate). It inherits the teleport-collection problem
unchanged and says so plainly rather than claiming to have fixed it.

## The shape of the whole arc

Read end to end, the through-line isn't any single result — it's a discipline of **treating negative
and obstructed results as diagnostic rather than terminal**: the meta-learning floor, the two
directed-collection negatives, the online-loop's double obstruction, and the S2 retraction each
redirected the next piece of work rather than ending the thread. The substrate's live self-correction
apparatus (COLLECTION_REALISM.md's tiering, the expansion cut's belief-scope correction, the repeated
"our own prior guess was wrong" admissions in `on_policy/`) is arguably as much the product of this
period as any individual finding. The standing open threads, per [`README.md`](README.md)'s own
next-steps: re-attempting the directed-collection ladder on the local-drift on-policy substrate where
it's finally askable, and testing whether ensemble disagreement — shown blind to confident-prior shifts
under teleport — recovers its job once on-policy collection creates genuinely unvisited territory.
