# SPEC — solo: re-internalization on the motor substrate, with no forward model anywhere

**The question in one sentence**: does [`rhm/practice/native/`](../../../rhm/practice/native/README.md)'s
consolidation — a committed motor vocabulary *proposed* by the learner's own routing head, *emitted*
by a corridor head on the learner's own executor, the table deletable, trust formed by use — happen
on a plant with nothing imaginary in the loop, at the delay where committed content is measured to
pay?

**Status**: spec, 2026-09-04. Nothing run. **Parent arc**: [`../README.md`](../README.md).
**Substrate and machinery donor**: [`../acappella/`](../acappella/README.md) — world, piece, reflex
law, nested library, plant audition, the naive delay operator, all model-free; fork `delay_gate.py` /
`world.py`, donor untouched, bit-identity gate against `b1` before changing anything.
**Port donor**: [`../offbook/nets.py`](../offbook/nets.py) (`SlotLayout`, π, `select_slots`,
`PropTrainer`, `build_span`, `SpanBuffer`) and [`../offbook/FILES.md`](../offbook/FILES.md) §"The
mapping", §"What Phase B should be run with", §"Why exploration had to be added at the ACTION",
§"The battery and the poison twin". Read those four sections before designing anything.
**Trunk donor**: the behavior-cloned motor program lineage —
[`../../ballistic/ballistic_readapt.py`](../../ballistic/ballistic_readapt.py) (`ballistic_bc`) and
[`../../jacobian_teacher/core.py`](../../jacobian_teacher/core.py) (`clone_policy`).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§1 (the three components), §3½ (identity → planner as routing; corridor → executor; identity never
into weights) · [`two_climbings`](../../../../ideas/two_climbings.md) §9 (a climb = address + trust;
use is only earnable) · [`sparse_one_rung_up`](../../../../ideas/sparse_one_rung_up.md) §3 (the
command face; the intention reference is exact DP on RHM and the FM elsewhere — this node tests a
third option: the committed tape itself).
**Attribution**: the framing is Jasper's (2026-09-04): re-internalization of new macros into the
policy is the goal; the multi-level ladder is set aside as possibly ill-constructed on this
environment; the FM's suspected role is as the *certifier* of a macro (error / δ-silence), so the
test is whether re-internalized macros go far with no FM, consulting an oracle temporarily where
required. The roles table, the trunk-without-a-model design, and the tape-as-intention-reference
reading came out of the exchange (session `0176ruxEDeDbCGGBxEMnNXkE`).

## Why this node exists, in one paragraph

Re-internalization ran once on mjc, in `offbook/`, and that node is marked suspect: the incumbent
planned for free in a near-perfect forward model and both ports were built out of it (seam-time
audition as FM rollout, the span head on the FM trunk). Port 2 never cleared parity there (0/20
slots — untested, not refuted), and the π-level facts that stand (trust tracks exposure; the poison
quarantine) were measured under that confound. `acappella/` then removed the model, found the RHM
grounding economy cannot run on a plant (the reflex law beats every priced search cell), re-sited
the meter on feedback under delay, and found the niche: at Δ = 8 (192 ms) stored segment tapes at
4 feedback events beat a per-Δ re-fit reflex at 136 by 2.0×, widening to 5.6× at 384 ms. It
deferred Port 2 in one sentence — "the corridor port has no trunk to sit on without a model" — and
queued B2 (π and trust at Δ = 8) behind a licensing condition that Jasper's framing now supersedes.
This node supplies the trunk without a model and runs both ports plus the address-book battery at
the niche.

## What the forward model did on this substrate, and what replaces it

| FM role | where it sat | model-free form here |
|---|---|---|
| the incumbent's planner | `fingering` → `offbook` | the reflex law (acappella's) |
| audition from the realized seam state | `offbook` | plant audition on the resettable copy (a priced grounding), or the frozen posture key — both built |
| bridging the observation delay | `accompanist` d3b | nothing; the naive operator is the honest one |
| content of the unit | `fingering`'s live plan | the executed tape (span F2/F4) |
| the trunk Port 2's span head reads | `offbook` | **a behavior-cloned reflex** — the learner's own primitive executor, cloned from the reflex law's traversals |
| intention reference for a committed unit's execution error | never built on mjc | **the tape's own stored realized trajectory** from harvest — free and exact, as the committed table was on RHM |
| intention reference for a live primitive action | `bridge_assembly` onward | out of scope by Jasper's framing; not needed to re-internalize a committed unit |
| δ-silence as the compile certificate | `etude` E-gate | the body's grade (band pass) plus rendition repeatability — logged as instruments this round, consumed by nothing |

Three facts from the record say the model-free regime is the natural one, not a handicap:
chunks pay only where the incumbent cannot imagine (`accompanist` finding 3); RHM's
re-internalization ran on an inert plant (`ratchet`: the descent was vocabulary-carried), which a
fixed reflex plus a frozen library reproduces; and nothing modelled should enter a library anyway
(`span` F2/F4).

## The design

**Standing constraint, verbatim from acappella**: no forward model anywhere in this node, not even
as an extra control or reference arm. Nothing in this folder imports, trains or evaluates an
`f(s,u)`. If the design ever seems to demand one, halt and report.

**Operating point: Δ = 8 (192 ms), acappella's sustained niche.** The reflex's per-Δ gain fit is
interior there (its edge cells are Δ ∈ {3,4,6,12,16}). Δ = 0, where the reflex wins outright and
consolidation should have nothing to buy, is worth a control row if cheap; the implementer decides.
Everything else about the world, piece, metering, library construction (nested, harvested late,
harvest reflex frozen at a0's gains) and the delay operator is acappella's, unchanged.

**The primitive is an option at every seam.** acappella's committed arms ran `with_prim=False`.
Here the reflex (through the trunk) must be a legal choice at each seam, or `no_table` has no
fallback and adoption has no meaning. Under delay the primitive is the worse option, so `frac_prim`
over cycles at performance tempo is the arc's plainest adoption readout: does the learner stop
playing by feel where memory pays.

**The trunk: a behavior-cloned reflex.** Clone the reflex law's own closed-loop behaviour into a
small policy net (delayed observation, seam index, phase within segment → command), on the
`ballistic_bc` / `clone_policy` pattern. Gate it: the clone must reproduce the reflex law's piece
error within a stated tolerance at Δ = 0 and at Δ = 8, on the clean and the rotated world, before
any port is wired. A trunk that plays worse than the PD law it clones would confound the primitive
and every battery row; if the gate fails, say so and stop. Whether the trunk stays frozen or adapts
through practice is the implementer's call with a stated reason — native's encoder was frozen; the
plant guard (the trunk's own primitive error, `e_react`'s analogue) is the interference readout
either way, and offbook's `regress` warns what unconditioned regression does to renditions.

**Port 1, routing.** π(unit-slot | delayed seam posture, seam index) → top-k materialised, trained
online by self-imitation on traversals the **body** graded competent, never on the audition score.
Exploration at the action, upstream of π's gate (`eps_act`, `explore_eps`, `force_window` — offbook
O3's lesson, with the reasons in `offbook/FILES.md`). The poison twin pinned to its reserved slot.
Currency: an audition is a grounding, so the O(K) → O(k) cut is a cut in groundings; acappella A-R
found that cut invisible against the search's budget, but here there is no search — the honest
rent is feedback events and groundings per traversal against the reflex's 136 / 0.

**Port 2, corridor.** The span head (slot, delayed posture, trunk read) → the unit's measured
commands, behind a per-slot parity gate; below parity the tape plays verbatim. Parity is
**execution reproduction on the plant** — a grounding — on held-out seam states split by a
deterministic code of the state (offbook decision 7), not the model-based check offbook used.
Every selected slot feeds its buffer on the metering traversal, not only the launched one
(offbook decision 6). The within-slot question the arm can ask and RHM could not: does the
conditioned head beat verbatim playback on off-key seams.

**The arm ladder** is offbook O1's in acappella's currency, which acappella's SPEC already wrote:
`reflex` (the reference; the reflex through the trunk) · `key_seg` (frozen posture key) ·
`audit_all` (plant audition of every legal slot) · `audit_prop_k` (Port 1) · `route_native`
(Ports 1 + 2) · `fid` (both ports wired and shut, must ≡ `audit_all` at 0.000e+00) ·
`audit_prop_kN` (π live at k = every legal slot, must ≡ `audit_all`). Pre-commit, every arm plays
the same stream (legato's `sched_late` property), which is what makes one seed readable.

**The battery**, on the final trained state of every arm with a library: `no_table` (the library
deleted; π and the head must serve, the trunk as fallback), `no_prim` (the primitive ablated),
`no_table_no_prim`, `restored`; `fid`'s untrained heads under the same forced-open policy as the
negative control. The poison twin's mass and launch fraction over the run. Whether native's
can't-decompose readout has a meaningful form at segment span — π's mass on a unit's own spelling,
where a segment's spelling is per-step reflex commands — is for the implementer to judge; it may
not, and saying so is fine.

**Levels.** `SlotLayout` populates segments and chains; leave both, since the machinery is there,
but spend nothing on the chain question. acappella finding 5 stands: on this piece the chain arms
decide at seam 0 from rest and are exactly delay-invariant, so no chain number here is evidence
about depth. The question is segment-span consolidation.

**The metering component, restored without a model — as instruments.** For every launched
committed unit, log the execution error `e` against the tape's own stored trajectory (waypoint
and per-step), a per-slot benchmark `b(s)` (a running estimate per slot, the `bridge` line's
form), δ_perf = (b − e)·gate with the gate on the body's grade, and each slot's δ-silence over
cycles. Log the audition's optimism (realised ÷ chosen) beside it, as acappella did. Nothing
consumes these this round; they sit next to trust (π's per-level and per-slot mass), parity, and
adoption so that the certifying role Jasper suspects the FM plays can be read against model-free
candidates on the same run. Consuming δ_perf as a plasticity gain on the span head is
`two_deltas`' regime-dependent question and is held for a later round.

**Where an oracle may be consulted, explicitly.** The resettable plant for audition and parity
(already the grounding). True-state grading of renditions (instruments and the ledger stay
true-state; agent-side reads are delayed, as in acappella). Optionally, a `given` library built
from the per-state oracle argmin or a high-budget plant-CEM harvest, to give mjc its first
earned-vs-given reading — cheap (acappella's build was 3840 groundings, 88 s) and the
implementer's call. None of these is a forward model.

## Gates before any treatment

1. **Fork fidelity**: with both ports shut and the primitive disabled, the fork reproduces
   acappella `b1`'s library build and its Δ = 0 and Δ = 8 rows for `key_seg` / `lib_seg` /
   `lib_all` / reflex at 0.000e+00 (acappella B-F1's cross-tag idiom, declared inapplicable
   rather than silently passed under any non-`b1` config).
2. **The trunk reproduces the reflex** within the stated tolerance, both worlds, both Δ. Report
   the tolerance and the numbers; stop if it fails.
3. **`fid` ≡ `audit_all`** and **π at k = N ≡ `audit_all`** bit-for-bit (G-P's precondition holds
   by construction in `select_slots`; assert it anyway).
4. **Parity is measured on the plant**, split by state code; continuous fractions logged beside
   the verdict so another τ can be read off the record.
5. **Seam information at Δ = 8**: acappella A-S found the spreads marginal (1.07–1.13) with
   per-state oracle gains of 1.22–2.35×; re-report under the delayed read, since π and the key
   both read a Δ-old posture (acappella flag 6: the coarse key degraded less than the fine
   audition — that comparison at the level of π is this node's).

## Readouts

Per cycle and per arm: piece error at performance tempo; feedback events and groundings per
traversal; `frac_prim` / `frac_seg` / `frac_chain` at performance; π's mass per level and per slot
(the trust series, offbook's probe ladder); parity fraction per slot; the plant guard; the poison
series. End of run: the battery deltas, the audition-optimism series, the δ_perf / δ-silence
instruments per slot, and the rent table against the reflex. Ranks, signs, bit-identity twins and
pre-fixed gates are the claims; single seed.

## Held loosely

- Whether the trunk should adapt through practice at all, and if so on what data (its own
  traversals only, or every executed rendition).
- Whether a value over (state, slot) should learn, as RHM's value head did; acappella held this
  open and started without it. Start without it here too unless the ladder needs it.
- Whether the Δ = 0 control row earns its cost.
- Whether a `given` library is worth building this round.
- What parity's τ should be on continuous commands; offbook's 0.9 never cleared and was never
  swept.

## Discipline

`MODAL_PROFILE=chromatic`; `/run-experiment-on-modal` before anything; launch only via a `--spawn`
path (offbook's gotcha cost two runs); never import a sibling runner (one Modal `app` for the
package — read donor constants with `ast`, as acappella does); entrypoint names unique across the
package; CPU only unless the heads genuinely need a GPU (acappella ran 16 processes, no CUDA);
volume `mujoco-control-data` under `/data/practice_solo/<tag>/`; smoke every gate before every
launch; single seed; keep a factual `FILES.md` as you go; **no README** — numbers get discussed
before interpretation is written; per repo norms no outcome is interpreted in this SPEC.

---

## Amendment, 2026-09-05 — the imitation filter, and a probe before a second run

**Attribution**: Jasper's, on reading `s0` with the figures. The decision to probe briefly before
committing to a second full run is also his.

**What `s0` showed that this responds to.** Both routed arms start at 0.0955 (cycle 0, where the
untrained π amounts to the construction-order top-2), rise to ~0.15 over cycles 2–20 as π first
moves off that prior, and settle at 0.12–0.13; the donor 8-tape audition reads 0.0838 on the same
pool in the same run. Learning bought cost and delay-robustness (4× fewer groundings, 3.8× less
priced time at ~6% error against enumeration) and not error.

**The candidate mechanism, argued not measured.** π's imitation targets are the better half of
each practice cycle's own traversals by realized piece error (`good = ep <= median(ep)`), a
*relative* filter. `offbook/`'s "competence band" turns out to be the same construction
(`np.nanmedian(pr["e_piece"])`), so an absolute band is new to this lineage rather than a return
to it. Under delay the body's grade is only weakly a function of the slot chosen, so a relative
filter admits posture luck as competence and π imitates it.

**The probe** (`s1q`): `--pi-filter band` with `good = ep <= band`, two band values pre-fixed from
published, arm-neutral references on this piece at Δ = 8 — **0.1066** (`ref_play`, étude's `never`
at performance tempo, acappella's blind playability guard) and **0.0838** (the donor 8-tape
`lib_seg` at Δ = 8, acappella `b1`, re-derived at 0.000e+00 in `s0`). Neither is chosen by
outcome. `audit_prop_k` and `route_native` at each band, 30 cycles, Δ = 8 only, the `median`
twin of `audit_prop_k` as the cross-tag identity gate against `s0`'s first 30 cycles. Per-cycle
band pass fraction and π target-row count are logged because starvation is the failure mode; a
band that admits nothing is a finding, not a knob. A second full run is licensed only by what the
probe shows, and per repo norms nothing about that is interpreted here. Jasper's framing for the
record: `s0` stands as a result whether or not the band changes anything.
