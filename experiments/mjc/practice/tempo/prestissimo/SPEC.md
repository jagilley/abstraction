# The level ladder on a fast piece, model-free — the motor instance of Track A

You are implementing a new node under `experiments/mjc/practice/`. Suggested name
`prestissimo/` (faster than `accompanist/presto/`, which this escalates); rename if you find a
better one. Save this prompt verbatim as the node's `SPEC.md` so the record matches
`acappella/` and `solo/`. Work in `experiments/` with `MODAL_PROFILE=chromatic`; read
`/run-experiment-on-modal` before launching anything and `/subagent-instructions` before
waiting on anything. Do not touch git. Fork donors, never edit them; every prior result must
stay byte-reproducible.

## Why this node exists

The RHM practice arc climbs a ladder of DGP levels: a level-ℓ macro is a pair of committed
level-(ℓ−1) entries (`T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]`), mined from the learner's own solved trajectories,
and Track A put a hand-written outer loop over that crank that reads the learner's own
next-level minability (`at_support`) to decide commit/hold and era advance — and the earnable
range extended with each turn (`conductor` → `maestro` → `crescendo`, ROADMAP §4.1 and §7.1).

The motor side has never had a ladder. Jasper's mapping, which the record already adopted in
`solo/`: **an RHM level is an automatic execution span.** Level 1 is a segment played open-loop
from one feedback event; level 2 is two segments played from one; and so on. Being able to
execute a long chunk without re-grounding *is* the level jump, and the piano analogy says the
value of the jump is that you can now play at a tempo or a delay where feel cannot keep up.

What the motor row has, model-free, at the segment rung (`solo/`, 2026-09-05): address (the
table deletable after consolidation), trust (π's per-level mass, poison quarantined tenfold),
routing, corridor, and adoption that tracks the meter (π routes to the primitive at Δ = 0 and to
memory at Δ = 8). What it does not have — the six gaps, in order of how much this node is about
them:

1. **No nesting op.** A "chain" in `offbook`/`acappella`/`solo` is a whole-tail command stream
   harvested in the segment-committed configuration, keyed by (span, seam), not by the lower-level
   slots it is made of. Nothing is mined from routed successes at level 2, so no `at_support`,
   so nothing for an outer loop to read. `legato/`'s fusion control is the one tuple op on the
   plant: welding measured segment tapes cost −0.004 (free) while welding live plans cost 2.2×.
2. **No era ladder.** RHM's nested damage schedule forces level k+1 at era k+1. `acappella/b1`'s
   delay sweep is a static version of the motor demand ladder (segment tapes beat a per-Δ re-fit
   reflex from Δ = 8 = 192 ms, widening to 5.6× at 384 ms); no run has paced it.
3. **The piece.** `etude/`'s square starts from rest, so chain arms decide at seam 0 with the
   true state and are exactly delay-invariant (`acappella` finding 5). The pre-declared
   escalation is a fast piece whose first seam is not its start — `accompanist/presto/`.
4. **The currency.** The RHM grounding economy does not run on a plant (search by real trial
   never beats the reflex; `acappella` finding 3); the motor meter is feedback under delay. And
   committed arms do not win raw error: `never` wins at every price in `fingering`/`legato`, and
   `solo` finding 7 says learning bought cost and delay-robustness, not error. So what a level
   buys here is **playability where the lower level is not playable**, plus cost and trust —
   never lower error at Δ = 0.
5. The primitive is continuous, so reflex → segment is crystallization and can't-decompose has
   no segment-span form (`solo` finding 9). At level 2 it does: a unit's spelling is its two
   level-1 slots. This node is the first place that readout can exist on mjc.
6. Timing has no certifier on mjc (`solo` finding 8). δ_perf against the tape is logged and
   consumed by nothing; leave it that way this round, as an instrument.

This node closes 1–3, gives 4 its readout, and makes 5 measurable. It is the motor A1: the
crank with the thermostat over it. A learned rule (A2) and the signature (A3) come after, if
this turns.

## Read first

- `solo/README.md`, `SPEC.md`, `FILES.md`, `solo.py`, `world.py` — the model-free stack you
  are extending (cloned reflex trunk, Ports 1+2, parity on the plant, battery, gates, sizing).
  Its standing constraint is yours: **no forward model anywhere**, not even as a control arm.
- `acappella/README.md` + `SPEC.md` Phase B1 — the delay operator (naive, nothing bridges it;
  honest because no predictor exists), the pre-fixed niche gate, the playability guard, the
  ordering margins reported per Δ.
- `accompanist/presto/FILES.md` + `piece.py` — the fast piece and its twelve recorded decisions
  (tempo against a ~100 ms loop, turn rate as the difficulty, `curl_b = 0`, greedy nested score
  sets, harvest late, the two-component guard). Its world is the **arm** and its incumbent
  plans in an FM with efference copy; neither carries over here. The piece design does.
- `legato/README.md` — nesting, the fusion control (F4), the level-(k+1) law (F5).
- `rhm/practice/ratchet/README.md` + `ratchet.py` (the miner: `mine_from = chosen`,
  `mine_support`, `n_at_support`, the nested build that drops entries whose halves are not
  committed); `conductor/README.md` + `FILES.md` ("The rule", the reads, the null-ABBA dead
  zones, the yoked-clock control that `census` finding 4 made mandatory); `spiral/README.md`
  (the three clocks); `native/README.md` (the battery, can't-decompose).
- `ROADMAP.md` §4.1, §4.5, §7.1–7.2; `ideas/practice_manufactures_its_own_credit.md` §1–2
  (§2(b): delay variance forces chunk boundaries — ballistic within, evaluated at the boundary);
  `ideas/two_climbings.md` §7 and §9; `ideas/sparse_one_rung_up.md` §3.

## The design, held loosely

**Substrate.** Two honest options; decide after reading both worlds and record the decision.
(a) A fast piece on `solo/`'s pusher: keep the entire model-free stack and gate chain, add a new
piece module (short segments, an approach leg so seam 0 is not from rest, a closed figure).
(b) Port the stack onto `presto/`'s arm: the piece is already designed against the reflex
delay and is momentum-dominated, but there is no model-free reflex law there (a Jacobian-
transpose PD in tip space uses fixed kinematics, not an `f(s,u)`; whether that honours the
constraint's purpose — nothing imagines the future — is yours to judge), and the trunk clone,
plant audition and gates all have to be rebuilt on a new plant. (a) reaches the ladder question
faster; (b) is the biology-facing piece. Either way, calibrate the piece the way presto did
(tempo × Δ sweep, the design point read off the sweep, pre-fixed rules for every shared knob —
`legato` F2).

**Levels.** Take s = 2 like RHM. K segments; level ℓ spans 2^(ℓ−1) of them. K = 8 gives four
rungs and the top one is the whole piece; K = 4 gives three. Presto's five does not tile. A
level-ℓ entry at seam k is the pair of committed level-(ℓ−1) slots at seams k and k + 2^(ℓ−2),
materialised as their fused tape (one feedback event at launch, the internal seam's feedback
dropped — that is what the level buys). Whether entries may sit at every seam or only aligned
ones is your call; RHM's are position-free, the piece is not.

**Mining.** `mine_from = chosen`: a level-(ℓ+1) candidate is an adjacent pair of level-ℓ slots
that a *solved* traversal actually played; it enters the table at `mine_support` observations.
`at_support(ℓ+1)` is the count of such pairs. This is the free one-level-up gauge.

**Solved.** RHM has exact solved/not. Here a traversal is solved if it is inside a pre-fixed,
arm-neutral band. Jasper's note on the band: **be generous at fast tempi.** Error is partly a
byproduct of chunking (gap 4), and the listener's perceptual clock does not speed up with the
notes — a smudge inside a 120 ms segment is not heard the way the same metres are heard over
800 ms. Two candidate forms, both pre-fixed and both reported: grade the executed trajectory on
a fixed wall-clock grid (resample at a listener period rather than per waypoint), and/or carry
presto's two-component guard `max(ref, ½ × mean leg)`. Pick, state why, never choose it by an
arm's outcome.

**Eras.** Delay Δ is the era knob, not tempo: a tape stays valid across delays but not across
tempi, so the vocabulary persists across eras the way `T[ℓ]` persists across damage eras. Era
ℓ+1 is a Δ at which level-ℓ units fail the band and level-(ℓ+1) units need not — read off the
static sweep, as acappella read Δ\*. Whether the piece admits three distinct eras is a result,
not an assumption.

**The outer loop.** Port `conductor`'s thermostat: commit/hold per level and era advance, read
from `at_support` one level up, dead zones measured by null-ABBA on this node's own series,
caps so a refusing arm terminates. Arms, in `conductor`'s shape: the scheduled crank (the
comparator), `outer_yield`, the within-level reader (the arm's own error — the one that should
refuse), and a **yoked-clock control per gauge arm** (same realised commit and advance cycles,
no gauge). `fid`-style twins at 0.000e+00 where the machinery allows.

**Readouts.** The three clocks per level (certificate, coverage of the table, value = the
playable niche at the era's Δ plus fb and groundings per traversal); π's per-level mass over
cycles (trust); the battery at every level (`no_table` / `no_prim` / restored; untrained heads
as the negative control); can't-decompose at level 2 (π's mass on a unit's own two slots);
`at_support` per level per cycle; the poison twin. δ_perf against the tape logged per slot.

## Phases

**A — the piece and its ladder.** Calibrate; build the level libraries statically (every
level, legato's nested construction, harvest late); run the Δ sweep with per-Δ re-fit reflex,
each level's keyed and auditioned arm, and report acappella B1's table per Δ with margins. This
is the era calibration and the first thing this piece can say — including whether the deep
rungs are readable at all. **Halt with the reduction** before building B's arms; B's era ladder
is read from it.

**B — the crank.** The mined nested tables under the outer loop, arms as above, one seed.
Smoke on Modal, launch detached, halt with the handle, halt again with the reduced summary and
figures. Sizing: `solo/s0` was 8 arms × 120 cycles in 56 min on 16 CPUs, no GPU; expect a few
multiples of that. Single seed; no multi-seed or high-DoP runs without authorisation.

## Record

`FILES.md` from the start: every decision with its measured reason, gates, smokes, sizing,
flags unsmoothed — `solo/FILES.md` is the template. Reduction and figures under
`results/<tag>/` and `figures/<tag>/`. **No README** — numbers are discussed before any
interpretation is written; the halts above are where that discussion starts. Outcomes are not
pre-interpreted here on purpose: if the crank refuses, if no era opens, if the band admits
everything or nothing, that is the finding, with its reason located.
