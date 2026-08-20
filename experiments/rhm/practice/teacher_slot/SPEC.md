# SPEC — teacher_slot: can a second loop fund the crossing, and can the learner supply its own currency?

**Status**: spec, unbuilt (2026-08-20). Written for an orchestrating agent who will delegate each
rung to a subagent. Operationalizes the named next step of
[`../fourwall/lm/`](../fourwall/lm/README.md) (the decision round) and extends it two rungs.
**Up**: [`../README.md`](../README.md) (practice arc) · **Substrate donor**:
[`../fourwall/lm/`](../fourwall/lm/FILES.md) — inherit `wall.py` / `wall_lm.py` / the per-arm-worker
design verbatim; do not modify it (its results must stay byte-reproducible).
**Parents**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§18 (the "open port for an already-climbed evaluation layer") ·
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) §9 ·
[`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md) (a value signal as a
drive over policy, reward = learning progress, acting on a low-dimensional *direction*).
**Attribution**: the teacher-slot framing — that a verbal learner might occupy the port the arc keeps
re-deriving, where a termite-fishing chimp has only "did I get a termite" — is Jasper's (2026-08-20),
as is the reading that the sculpting nodes' inert plant is unsurprising because the earned
vocabulary had no backprop handle. The rung decomposition came out of that exchange.

## Why this experiment

`fourwall/lm` measured, on a real NTP transformer, that gradients supply *track* and not *merge*:
handed a free spurious key, the reader takes it instantly, re-maps it after every rotation, and
never quotients — ending at Bayes-level task error with its latent-derived pathway threefold
suppressed (d4 0.27–0.30 vs 0.83–0.84), invisible to the loss. Supplied exogenously, the merge op
repairs that at ~zero price (+0.10–0.17 nats, gone in ~100 steps). Its joint interpretation split
the deficit in two, and this node exists to test the two halves separately:

- **The capability gap.** The op transient is a locally-worse region a local rule will not cross.
  But at the instant of the op, *the pathway readout is already climbing while task NLL is spiked* —
  a quantity exists that improves inside the dip. Will an outer loop that reads it **choose** the
  merge and **hold** through the dip? (Rung A.)
- **The incentive gap.** The within-level lifetime ledger *rewards* never merging (`true_wall`
  1.42 < `wall` 1.46 < `merge_13000` 1.47 < … < `no_wall` 1.53), and the withheld pathway is
  denominated in next-level currency. The arc's line — *the factory cannot manufacture the price of
  its own next level* — is why every ladder in this arc has topped out at a teacher (lend the
  grader, schedule the recital, hold at the bottom, schedule the decorrelation). Can the learner
  supply that currency itself — numerically from its own next-level yield (Rung A½), or verbally
  (Rung B)?

Rung A isolates *will a loop fund the crossing, given the currency*. A½ and B ask *can the learner
find the currency*. Keep them separate: if B fails, you want to know whether it failed at funding
or at finding.

## Reading list (ordered; first two load-bearing)

1. [`../fourwall/lm/README.md`](../fourwall/lm/README.md) then its [`FILES.md`](../fourwall/lm/FILES.md)
   — the world, the instruments glossary, the bit-identity design (9a), the floors, and the
   `wall_fast` checkpoint-grid correction (read that before trusting any single-checkpoint number).
2. [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
   §18 — distribution is forced; introspection is level-bounded; the teacher thread.
3. [`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md) — the value signal
   must be a drive over policy/allocation (not a residual-stream scalar), read off a value-relevant
   *direction*; reward = learning progress on the reducible component. Its §Results (CURIOSITY_DRIVE
   2b) is the nearest built instance.
4. [`../ear/README.md`](../ear/README.md) — grading is priced (5–29% of budget); the `taught`
   paradox; "the concert date is what commits the piece."
5. [`../recital/README.md`](../recital/README.md) finding 9 and [`../merge/README.md`](../merge/README.md)
   round 1 finding 3 — the next-level currency measured (level-k+1 table; `|T3|` withheld by the
   index). These are what A½'s readout must be denominated in.
6. [`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md)
   §5 (the merge op's constraints) and §9.
7. For Rung B only: [`../../../../beliefs/trees/self_prediction_and_self_knowledge.md`](../../../../beliefs/trees/self_prediction_and_self_knowledge.md)
   and [`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md) §3 — what a
   text-trained model does and does not carry about itself and about ratchet-stage priors.

## Substrate (inherited, not rebuilt)

`fourwall/lm`'s world exactly: v16/s2/L6/m4, rule_seed 0; a free prefix token `w` bijective with the
level-2 latent at node 0; the `wall` schedule (identity to 8000, rotate +4 mod 16 every 2000,
identity return at 14000); 8L/8H/256D GPT, 20k steps × 64; one worker per arm with identical
seeds, so trajectories are bit-identical until schedules diverge. **This last property is
load-bearing** — it is the only reason cross-arm contrasts are licensed in that node, and every
arm here must keep it. The merge op is fwlm1's: from some step, `w` → the neutral filler in the
input stream, no parameter touched.

Two readouts carry the whole design, both already built: the **task readout** (indexed-span NLL,
consumed condition) and the **pathway readout** — key-anchor d4 under the randomised/neutral wall,
and its probe-free twin (indexed-span excess over exact Bayes per arrival level under the neutral
condition). The probe-free twin needs no trained probe and is the more endogenous of the two; both
cost an eval pass, and that pass should be priced (ear's convention). Floors: 0.0012 seed /
0.0039 placebo on indexed-span NLL; the dead pair is the replicate.

## Rung A — the decision round (held loosely; the instrument list is the commitment)

**What's new**: an outer loop that, at each checkpoint, chooses the input condition for the next
interval — `w` present or collapsed. Make it **reversible** per checkpoint: fwlm1's `merge_s` arms
are then the step-function special case, which is also the fidelity gate (a policy hard-coded to
"collapse from s, never restore" must reproduce `merge_s` bit-for-bit). Reversibility is what makes
"hold through the dip" a decision rather than a foregone conclusion — a loop that reads task NLL
*can* bail out at M+25 when the spike is at its worst. Keep the learner minimal (a tabular/bandit
value over the readout trajectory, or even a stated rule); the point is what it reads, not how
clever it is. It holds a budget: probing the pathway readout costs steps.

Arms (first pass): `outer_task` (reads task NLL only) · `outer_path` (reads the pathway readout) ·
`outer_path_lp` (reads learning progress on the pathway readout — two_timescale's form) · the
exogenous bracket `merge_1000 / merge_8000 / merge_13000` and `wall`, `no_wall`, re-run in-tag for
bit-identity. An **unrotated** control arm (the `true_wall` schedule under `outer_path`) is worth
having: does it merge with no rotation ever occurring, on pathway evidence alone?

Readouts: whether and when each loop first collapses `w`; whether it restores during the transient
(bail-out count); terminal pathway wholeness (d4 rand) and terminal task NLL; the lifetime task
integral; probe spend as a fraction of budget; timing against the exogenous bracket.

**Metered sub-round (A-m, optional, same machinery)**: price every feedback/token event and re-run
the loops. fwlm's prediction is that a meter makes the *rent* half visible (holding the scaffold now
costs something the ledger can see) and leaves the *pathway* half invisible at any meter setting.
Whether that prediction holds is a measurement.

## Rung A½ — an endogenous cross-level drive

Replace the hand-built pathway probe with the learner's own **next-level yield**: a readout
denominated one level above the keyed latent (reread/lm's level-ordered probes are the instrument;
merge round 1's `|T3|` minability is the sculpting-side form). The arc's most seed-stable fact is
that the value of level-k practice is expressed in level k+1's currency — so this is the most
natural drive that isn't a probe we wrote to the answer. It is still numeric, still not verbal.

Gate first: reread/lm found d5/d6 with essentially no dynamic range at 8L/256D over 81.9M tokens. If
the next level above the keyed node has no range here either, that is a scope result to record
(consistent with `tall`: signals undefined beyond the earnable range), and the remedies are a larger
model, a shallower grammar, or moving A½ onto a sculpting twin where `|T3|` is readable. Do not
force it.

## Rung B — the learner occupies its own teacher slot (design after A/A½ land)

The same decision, made through a **verbal channel**. Two forms, in order of how much they bet on:

- **B1 — a reasoner reading the instruments.** A language model is given, at each decision point, a
  compact description of the learner's state (what its predictions are keyed on, the task and
  pathway trajectories as A's loops see them) and asked to name what the learner is keying on,
  whether that key is stable, and to choose the input condition. The interesting cell is the
  **pre-rotation** one: before any rotation, nothing in the within-level evidence says the key is
  spurious — it is a perfect correlate. If a verbal reasoner merges *anyway*, on the grounds that a
  free, perfectly-correlated prefix token is the kind of thing that is a scaffold, then what the
  teacher slot carries is a **prior about what kinds of keys are scaffolds** — and a text-trained
  model may hold that prior because text is post-ratchet. Vary what it's shown (task only; task +
  pathway; plus the rotation history once rotations begin) and log its stated justification.
- **B2 — the learner's own self-report.** Same question, asked of the learner itself through a
  self-report channel rather than of an external reasoner reading its instruments. This rides on
  introspective access the repo has measured as limited in places (self-knowledge in
  computation-space, not outcome-space; the calibration null), so it can fail for a reason
  unrelated to the teacher-slot question. That is why it comes last.

Readouts for both: merge timing relative to the first rotation; hold vs bail-out; terminal pathway
wholeness; the justifications (qualitative, kept).

## Adjacent and separable — the empty cell (`handle/`)

Every earned-vocabulary node in the arc kept the vocabulary *outside* weights (frozen tables over
GD-given level-1 features; macros are controller-side actions the generator never predicts or
conditions on — see `../ratchet/macros.py`'s header), and every endogenous twin had no earning op.
The cell *earned vocabulary living in weights as a backprop handle* is empty, and the sculpting
nodes' inert plant is exactly what you'd expect if the new tokens had no gradient path. The cheap
probe: on the ratchet substrate, give each committed macro entry (or a per-level macro symbol) a
slot in the generator's own vocabulary so it predicts and conditions on it, and read whether
clean-config parse/infill moves, whether next-level yield changes, and whether the earned-vs-given
fraction shifts. This is [`language_reduction_continual_learning`](../../../../ideas/language_reduction_continual_learning.md)'s
"token as backprop handle" on sculpting. It runs on a different substrate from A–B and can go in
parallel; it does not gate them.

## What the outcomes might mean (predictions, not constraints)

Recorded per repo norms as the theory's guesses, held loosely: `outer_task` never merges or bails
out in the dip; `outer_path` merges and holds; A½'s yield drive is slower than the pathway probe or
has no range at this scale; B1's pre-rotation cell is genuinely open and is the cell worth the most.
Outcomes that would revise the arc are at least as valuable — e.g. `outer_task` merging (the
within-level ledger is less hostile than the integral suggests, perhaps on local LP), or `outer_path`
failing to hold under a budget (the barrier is taller than +0.10–0.17 nats once probing is priced).
Bring the numbers back for discussion before writing any README; idea-doc revisions are queued,
not applied.

## Practicalities and orchestration

- Invoke `/run-experiment-on-modal` first; profile `chromatic`; smoke before every detached launch;
  halting procedure for anything >5 min. Single seed first (`experiments/CLAUDE.md`); ranks, signs,
  trajectories, and floor-multiples are the claims; the dead pair supplies the floors.
- **Fidelity gates before any arm**: the hard-coded step-function policy reproduces `merge_8000`
  bit-for-bit (fwlm1's 9a discipline, max|Δ| = 0.0); the unassisted arms reproduce `fwlm0`/`fwlm1`.
  Inherit the per-arm-worker / zero-global-RNG design; if the outer loop consumes RNG, give it its
  own generator.
- Structure per `STRUCTURE.md`: this folder is the node; rungs are children (`decision/`, `yield/`,
  `verbal/`, and `handle/` if run), each with its own `README.md` post-discussion and `FILES.md`;
  this `SPEC.md` stays as the record of what was asked. Import `../fourwall/lm/` machinery; don't
  copy or modify it.
- **Delegation**: one subagent per rung, owning the whole build loop — script, debug, smoke, launch,
  reduce to a summary and figures — per `/write-spec-or-prompt`. You keep the wait and the
  interpretation. Rungs are sequential (A → A½ → B) because each one's design reads the last one's
  numbers; `handle/` may run alongside A. Expect two halts per rung (launch handle, then reduced
  results); attach a Monitor on launch; resume a subagent by name rather than respawning.
  Subagents do not touch git.
- Out of scope here, named so nobody reaches for it by accident: re-key and retire as supplied ops;
  the strong-form scaffold (true key not cheaply representable); the grokking-window long run.
  All are `fourwall/lm`'s own queued items and stay there.
