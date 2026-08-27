# SPEC — a cappella: the port back on the motor substrate, with no forward model anywhere

**The question in one sentence**: does `rhm/practice/native/`'s consolidation — a committed motor
vocabulary proposed and run as one unit by a routing policy, its members no longer enumerated, with
trust formed by time-in-use — happen on a plant when the incumbent it competes with has to **pay for
every look and every trial in the world**, the way RHM's incumbent did?

**Status**: spec, 2026-08-27. Nothing run. **Parent arc**: [`../README.md`](../README.md) (mjc/practice).
**Why this node exists**: [`../accompanist/README.md`](../accompanist/README.md) — the pivot. Every
mjc practice node since `fingering/` handed the reactive incumbent a near-perfect forward model at
zero price, and `offbook/` (PR #72) then built the ports themselves out of that model. Six further
rounds (`offbook/` d1–d3b, `accompanist/presto/`) found that stored motor content reaches the bar on
both pieces once it is built the way `etude/`/`legato/` built it, and that the deep unit wins only
against an incumbent that is not allowed to imagine — because any delay shorter than the model's
composition horizon is bridged by the model. That is RHM's condition in negative: on RHM search was
*real and priced* (materialise-and-re-encode on the plant, a declared grounding budget), and depth
became unaffordable to primitives at the budget. **`offbook/` is marked suspect for this reason.**
**The working precedent**: [`../etude/`](../etude/README.md) — the one MuJoCo node where committed
units beat the incumbent on both axes, 3/3 seeds (0.077 vs 0.107 at 34–53% less priced time).
Read it first: its compile op (selection of executed renditions on consumption-distribution states,
never regression), its metering (feedback events × `d_fb`, plan counts logged separately), and its
piece are the starting point here.
**The RHM shape to reproduce**: [`../../../rhm/practice/ratchet/README.md`](../../../rhm/practice/ratchet/README.md)
(the declared budget G, "depth is unaffordable to base moves, not merely harder", and §"the declared
budget sets the headline… the choice is ours") ·
[`../../../rhm/practice/native/README.md`](../../../rhm/practice/native/README.md) (Port 1 routing,
the address-book battery, the poison twin) ·
[`../../../rhm/practice/spiral/README.md`](../../../rhm/practice/spiral/README.md) (the budget at
which enumeration *cannot* keep width past a commit).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3 (compilation is self-imitation of one's own traces), §3½ (held loosely here — the FM as holding
structure is exactly what this node removes), §6 (the conditions must be manufactured);
[`two_climbings`](../../../../ideas/two_climbings.md) §7, §9.
**Attribution**: the suspicion that the FM was the confound and that this test should be model-free
"even if suboptimal in quality" is Jasper's (2026-08-27); the diagnosis of *how* the model enters
each node, and the étude-as-precedent framing, came out of the exchange that followed.

## What went wrong before, in one paragraph

On RHM the incumbent's search is a closed-loop re-grounded beam: every candidate move is
materialised on the plant, each materialisation is a grounding, and a per-solve grounding budget
G is declared. A level-2 macro costs one grounding to play. At G = 58 the base action space's
privileged exact-DP oracle reaches 0.449 while one macro at width 1 reaches 0.293 — depth is
*unaffordable*, and the budget was chosen on purpose. On the arm, the reactive incumbent is CEM-MPC
over a learned forward model: it re-plans every 20 ms in imagination, at no price, and re-grounds on
a full-state observation, and it sits at ~0.01 on every piece. `never` won outright in `fingering/`,
`legato/` and `offbook/`. `offbook/` then measured its units with the same model (seam-time
audition, blind at chain span), priced its "rent" in FM rollout-steps (which buy nothing — G-K(a)),
and taxed feedback with an observation delay that the model bridges (`accompanist/presto/`,
`offbook/` d3b). The one MuJoCo node where chunks won, `etude/`, is the one where the model was the
*bottleneck* (capacity-limited, interfering across segments) — committed executed traces were
immune to the model's rot. So the substrate has never run the RHM economy: the incumbent's search
has never been real, and never priced.

## The design

**No forward model anywhere.** Not in the incumbent, not in the audition, not in the delay
operator, not in the ports. Everything the agent knows about the plant it knows by having executed
on it.

**Substrate and piece: `etude/`'s, unchanged.** Puck-free corridor world (`../../pusher_env.py`,
gear 10, damping 2.0, `frame_skip` 12), the closed square loop of four 34-step segments with the
localized rotation on segment 1, boundaries re-grounded from the true achieved state, metering by
boundary error at performance tempo, `d_fb = 0.10 s`, motor noise on practice. Fork `etude.py`'s
world verbatim and gate the fork with a bit-identity assertion on a fixed command sequence (the
`offbook/` G-F idiom) before changing anything. Three seeds of committed-beats-incumbent exist on
this piece; a null here cannot be blamed on the piece. If the étude's piece turns out too easy for
the *new* incumbent (see the first gate below), `legato/`'s arm piece is the fallback, with the
same fork discipline — but change one thing at a time.

**The incumbent: real-rollout search under a declared budget.** Replace CEM-over-the-FM with CEM
(or a beam) whose rollouts are **executed on a resettable copy of the plant** from the current
state — the motor analogue of RHM's materialise-and-re-encode. Every rollout is a **grounding**,
priced on the ledger; a declared per-decision budget G (rollouts) is the economy, and each arm runs
the widest search its own action set affords at G, exactly as `ratchet/` did. Sizing G is the
calibration this node has to do honestly and record: `ratchet/` chose the point where the base arm
spends the most; `spiral/` chose the budget at which enumeration cannot keep width past a commit.
Pick by a pre-fixed neutral rule, on an arm-neutral reference, before any treatment runs (legato
F2), and report the ladder around it so the dependence stays visible.

**A model-free reference incumbent with no search at all**: a fixed feedback law toward the
waypoint schedule (PD on tip or joints, gains calibrated once on the stale plant). It pays a
feedback event per step and nothing else. This is "closed-loop by feel" in its purest form and the
biologically honest reflex — keep it as the `never`-style reference so the priced-search
incumbent's cost is visible against something that plans nothing.

**The library: executed tapes, selected on consumption states, nested in piece order.** The
`etude/` compile op as corrected there (uniformly-drawn executed renditions scored on held-out
hand-over states, never ranked by their own outcome — E-3b's winner's curse), auditioned **on the
plant** (priced groundings), keyed at launch by the observed hand-over (`legato/`'s
`select_library`), built seam by seam **in the already-committed configuration** so candidates for
seam k launch from the frozen predecessor's arrival distribution (`offbook/` d2's finding: this
nesting, not the audition op, is what carries legato's content — per-state oracle 4× worse
without it), harvested late. Two levels as in `offbook/`: segment tapes and chains to the end of
the piece. Chain cells grown over segment-slot spellings (legato F5) — with a reactive-sourced
chain pool as the declared control, since `presto/` found F5 did not pay at 120 ms seams.

**Port 1, routing: `offbook/`'s π over library slots, unchanged in shape** — π(unit-slot | seam
posture, seam index) gates which slots get *materialised* (top-k of the cell), trained online by
self-imitation on traversals the body graded competent, ε upstream of π's gate for exposure
(`offbook/` O3's lesson), poison twin pinned to a reserved slot. What changes is the currency: an
audition here is a **plant rollout**, so the O(K) → O(k) cut routing buys is a cut in groundings —
the actual `native/` finding 1, now against a real price. No Port 2 in this round: the corridor
port has no trunk to sit on without a model, and that is a separate question.

**Delay, if used, is a decision constraint on the reflex law and the search's initial state only**
(`offbook/` round 4's one-variable construction) — and there is no predictor to bridge it. The
naive delayed operator, which `offbook/` d3b and `presto/` p0 showed is a strawman *when a model
exists*, is the honest one here. Δ stays a swept knob with Δ = 0 as the baseline
(`ballistic/`'s discipline: a slope, never a single success number).

## Gates before any treatment (Phase A)

1. **Fork fidelity**: the forked world reproduces an `etude/` traversal bit-for-bit on a fixed
   command sequence.
2. **The incumbent reaches**: the priced-rollout planner at the calibrated budget plays the piece
   at Δ = 0 to within the `etude/`'s `never` range (0.10–0.11) or better; the reflex law plays it at
   all. If the priced planner at any affordable budget is *worse than the reflex law*, say so and
   stop — the search is then not an incumbent worth beating.
3. **The rent is real**: a materialisation ladder (k = 1 … all) in groundings and priced time, so
   that "the library pays rent" is a statement about cost *and* outcome in the same currency.
4. **Seam information**: posture spread ÷ repeat noise per seam (`fingering/` G1 / `offbook/`
   G-S) — keying is only meaningful where this is > 1; `presto/` found it below 1 at 120 ms seams.
5. **Audition calibration on the plant**: chosen vs realised on held-out states, both levels — the
   thing `offbook/` could only do in imagination.

## Phase B: the port, then trust

The arm ladder of `offbook/` O1 with the currency changed: `never_reflex` · `never_search` (priced
rollouts, full width at G) · `key_frozen` · `audit_all` (plant audition of every slot, at G) ·
`audit_prop_k` (π top-k) · `fid` (π wired and shut, must ≡ `audit_all` exactly). Then the address
-book battery (`no_table`, `no_prim`, `no_table_no_plan`) and the poison twin, as in `offbook/`.
Read: does routing consolidate (fewer groundings at equal-or-better error — `native/` finding 1),
does the chain level get adopted (`frac_chain` at performance tempo, which was 0.000 in every
`offbook/` round), does trust track exposure, does the quarantine hold. The readouts and the
per-cycle probe ladder are `offbook/`'s; reuse `nets.py` and the ledger idioms rather than
re-deriving them.

If adoption occurs, the trust-formation instruments queued in `rhm/practice/census/` (arrival vs
rehearsal, the exposure-vs-credit control at `pi_fb_rep = 1`) finally have a substrate. If it does
not, the by-arm ladder says where — and this time the answer cannot be "the model".

## Held loosely

- Whether a resettable plant rollout is the right motor analogue of a grounding. On RHM the plant
  materialises a proposed edit; here the body executes a proposed command sequence and is reset.
  Both are real, priced trials in the world with reversibility. If this bothers you, the reflex
  law is the incumbent with no reset at all, and the comparison of the two incumbents is itself a
  readout.
- The budget G "sets the headline" on RHM and will here too. That is the design, not a flaw —
  `ratchet/` said so — but the ladder around G must be reported every time.
- `etude/`'s units were state-independent and its boundaries did not carry information (its
  finding 7). Keying and nesting were built for the arm. On the pusher they may be inert; that is
  a finding, not a failure, and the seam-information gate says which in advance.
- Practice-time learning with no model is selection over executed renditions plus π's routing.
  Whether anything else *should* learn (a value over (state, slot), as RHM's value did) is open;
  start without it.

## Discipline

`MODAL_PROFILE=chromatic`; `/run-experiment-on-modal` before anything; launch runners only via a
`--spawn` path (the `offbook/` gotcha cost two runs); smoke every gate before every launch; volume
`mujoco-control-data` under `/data/practice_acappella/<tag>/`; single seed first, seeds only if the
effect is small enough to need them (`experiments/CLAUDE.md`). Keep a factual `FILES.md` as you go;
discuss numbers before writing a README.

---

## Amendment, 2026-08-27 — Phase B is re-sited on the feedback/delay axis

**Attribution**: the re-site decision is Jasper's, on reading Phase A (`a0`). The gate-2 halt
below is **not retracted**; it is the finding the re-site is a response to.

### What Phase A settled, and what it retired

Gate A-I's pre-fixed halt **fired**. The priced-rollout planner does reach — 0.0612 at G\* = 4096,
inside étude's `never` band (0.10–0.11) and better — but **0 of 15 cells** of the whole incumbent
family (9 budgets × performance tempo, 6 matched-cost tempo cells, 3 reactive rungs) beat the
model-free reflex law on piece error, and none dominate it: best search cell 0.0607 at 15011 s
priced against the reflex's 0.0045 at 16.9 s — 13.5× worse error at 888× the cost. Per this SPEC,
"the search is then not an incumbent worth beating", and Phase B **as originally specified** (an arm
ladder whose incumbent is `never_search`, graded in groundings) is not licensed and will not be run.

**Therefore the grounding economy is retired as this node's headline.** RHM's economy does not
transfer to this plant, for a measured reason: a real motor trial costs the time to perform it
(0.916 s here), while a feedback event costs `d_fb` = 0.10 s, so search is ~250× the price of feel.
Two Phase A numbers say where the economy actually lives instead — A-R's rent table separates the
arms by **feedback events** (reflex 136, segment library 4, whole-piece chain 1), not by
groundings (routing's O(K)→O(k) cut was 13 of ~16400, 0.08%); and A-S found the noise-free seam
spread under the reflex collapsing to 2.1e-05 and 0.0 at the last two seams.

### Phase B1 — the Δ ladder on étude's piece, unchanged

**The question**: does the Δ-slope open a **playable niche where committed content is strictly
best**? This is the meter's native form on a plant, and it is the regime the biology claim was
always conditional on — closed-loop feel failing while stored content still passes.

**The operator** is `offbook/` round 4's, one variable: an observation delay Δ on every agent-side
feedback consumer (the reflex law's per-step read, the library key at launch, the seam-time
audition's rollout start, the search's initial state); experimenter-side instruments and the whole
ledger stay true-state. **Nothing bridges it, and nothing may be added.** `accompanist/` showed the
naive operator is a strawman precisely when a predictor exists (`d3b`: efference copy recovered
6–11× of the naive penalty); with no forward model anywhere it is the honest operator. The
no-FM constraint applies to the delay operator above all.

**Arms**, ordered by feedback consumption: `reflex` (136 fb, gains **re-fit per Δ** on the clean
world — every advantage to the incumbent) · `key_seg` (4 fb, 0 groundings) · `lib_seg` (4 fb, 32) ·
`lib_all` (≤4 fb, 49) · `key_chain` (1 fb, 0) · `aud_chain` (1 fb, 8) · `search` at Δ = 0 only, as
the record rung and explicitly not the headline.

**Pre-fixed, before launch** (implemented in `delay_gate.py`, recorded in `FILES.md` §B1):

- **The gate.** Δ\*<sub>niche</sub> = the smallest Δ in the declared ladder at which the best
  committed-content arm is **strictly** better than the reflex law on piece error, subject to that
  arm being ≤ `ref_play`.
- **The guard.** `ref_play` = **0.1066** — étude's `never` piece error at performance tempo,
  3-seed mean; the donor node's ballistic-per-segment incumbent under a stale model, undelayed.
  This is `offbook/` d0's `ref_stale` construction on this very piece, arm-neutral here because no
  arm in this node produces it. presto's second term (½ × leg = 0.4) is **reported beside it but
  not used**: on this piece it would make the guard vacuous, which is d0's own guard-revision
  failure in the opposite direction.
- **The depth test.** d0's 3-term ordering `e_chain ≤ e_seg ≤ e_reflex` (`aud_chain`, `lib_seg`,
  `reflex`), reported per Δ with both ordering margins, the guard margin in metres and as a
  fraction, and the chain's own distance to the guard (presto decision 11). It distinguishes a
  **segment-span** niche (`offbook/` finding 7d) from a **chain-span** one.
- **The converse halt.** If the reflex law is best at every Δ, B1 reports no niche and B2 is not
  licensed, regardless of how the priced ledger looks.
- **The range is fixed.** Δ ∈ {0,1,2,3,4,6,8,12,16} control steps (0–384 ms at `dt_ctrl` = 0.024 s;
  d0's range). If the gate fails the ladder is **not** extended or re-tuned (`legato/` F2); the
  pre-declared escalation is the fast piece.

**Structural note that must accompany every chain number**: `aud_chain` and `key_chain` decide once,
at seam 0, where the piece starts from rest and there is no stale history — so their read is the
true state at every Δ and they are *exactly* delay-invariant. That is a property of **this piece**
(étude's loop starts at the first waypoint; `offbook/`'s had an approach leg before its first seam),
not a property of depth.

**Pre-declared escalation**: if B1's slope opens no niche, the next round forks
`accompanist/presto/`'s 120 ms-segment world (`t0`/`p0` donor) under the same fork-fidelity
discipline and the same criteria. The two pieces are not blended in one round.

### Phase B2 — only if a niche opens

The port/trust round at Δ\*: `offbook/` O1–O3's shape (π over library slots, trust formation,
exposure-ε upstream of π's gate, the poison twin), with a reduction read between B1 and B2.
