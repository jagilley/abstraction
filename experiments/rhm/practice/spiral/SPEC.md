# SPEC — spiral: the live re-earning spiral on the depth-6 substrate

**The question in one sentence**: when the vocabulary is re-earned *live inside the ported loop* —
earn level k, consolidate it into the learner's own planner and executor, then earn level k+1
natively over the routed policy — does the crank accelerate, hold, or decay per turn, and which
wall bites first?

**Status**: spec, 2026-08-22. Nothing run. **Parent arc**: [`../README.md`](../README.md)
(rhm/practice). **Direct parents**: [`../native/`](../native/README.md) (the ports; its next-steps
queue names this round first) and [`../tall/`](../tall/README.md) (the depth-6 substrate; closed as
a diagnosed apparatus negative with the re-run recipe recorded in
[`../tall/FILES.md`](../tall/FILES.md)). `native/SPEC.md` explicitly scoped live re-earning *out*
of that round; this is that round. Single seed, per standing policy — ranks, signs, and multiples
of measured floors are the claims.
**Attribution**: the question is the arc's queued next step (native README §Next steps, discussed
with Jasper 2026-08-21); the depth-6 venue is forced by the confound `tall/` was built to break.

## Why this experiment, and why it became possible

`native/` showed a fixed, already-earned vocabulary consolidates: routing (identity's *use*) into
the planner beats enumeration outright at matched price, the corridor consolidates into the
executor without `handle/`'s interference, and the table survives as the address book — needed for
growth, not performance. But every claim there is one turn of the crank with the vocabulary held
fixed, at depth 4, where level 3 is simultaneously the deepest earnable level and one below the
grammar's ceiling. The **rate version** — era k+1 certifies *faster* given native era k — is named
unclaimed in both `ratchet/` and `native/`; and native finding 7 (native L2 routing raises L3
minability: chosen trajectories built of L2 calls are better mining substrate) is exactly the
mechanism a live spiral would compound, measured only in its one-turn form.

`tall/` is the venue because only there does the spiral have room to turn more than once with the
frontier and the ceiling in different places: the damage ladder runs to level 5 while the earnable
range stops at 3, so eras 3–5 are pure consumption and L2/L3 are both interior. `tall/` closed
with three diagnosed failures, and the reason to re-open it now is that `native/` dissolves one of
them **by construction**: the pricing inversion ("committing grows the action set 32→48 and
`fit_width` drops the beam from width 2 to 1; +0.180, a 4.6× outlier") does not exist under top-k
routing, where cost depends on k, not on action-set size — the very act the tall setting penalised
is the act routing makes free. The second failure (value starvation at 64 tokens) was half-fixed
by `dens0` (`n_corrupt=1` is the whole lever; loop descends at −0.004/cycle, ~4–8× slower than
depth 4) and routing bears on it too, in a way nobody has measured: freed groundings buy width at
fixed G, macros shorten repairs, and both should densify the positive signal the value head
starves for. Whether that is enough is Phase A's question, and a clean "no" is a finding about
what consolidation cannot rescue, not a failure of the round.

## Substrate (measured facts, not choices)

Depth 6, m=2 — never m≥3 (measured inadmissible: synonymy flattens the ladder to 1.06–1.30× vs
m=2's 3.25×; everything above L3 unrepresentable). v=8, s=2, 64 tokens, 32 base moves (48 after
L2 commit, 56 after L3), `max_macro_level=3`. Nested damage ladder `1:25,2:12,3:6,4:3,5:1`,
gradient 3.31×. Earnable range at `mine_cap=8`: L2 needs ~6 cycles to cover, L3 ~24, L4 ~384
(not earnable at any affordable budget). Hard prerequisites from `tall/FILES.md`, all measured:
`n_corrupt=1` collection; the **descent gate** (competence descends toward the floor in a
realistic cycle budget) as a precondition for any main run; probes re-priced (they were 47% of
`tl_s0`'s runtime); `preflight` before any paid setup (two interface-drift crashes each cost a
~510 s setup before it existed). Pricing: `beam_ground(32,8)` → 257 (w1) / 482 (w2); holding
width 2 post-commit under **enumeration** needs `g_budget ≥ 722` — an enum arm run below that is
width-crippled at commit and its earning trajectory confounded, so pick the g_budget with reasons
on the record; under routing, width refits under effective branching k (`native/prop`'s idiom).

## Design (held loosely — the instrument list is the commitment, not the interpretation)

**The loop.** Live mining from the beam's own chosen trajectories (`mine_from="chosen"`), commit
policy fixed across arms: certify-else-provisional-at-boundary (`ear/`'s rescue, `tall/`'s
approved design; recert stays live). The natural live form of consolidation, stated as the default
and revisable with reasons: π trains online by self-imitation from cycle 1 over whatever action
set exists; at a level-ℓ commit the entries enter the table, π's action space grows, and span
slots open for the new entries behind the per-macro parity gate (τ = 0.95, re-checked every
cycle). "Consolidation" is then not an event but the commit plus the heads catching up — and the
*lag* between commit and parity/routing-adoption is itself a readout, per level.

**Arms** (first pass — a treated arm and its twin bit-identical until treatment; final list is
the builder's with reasons on the record):

| arm | what it is | the question |
|---|---|---|
| `given` | true tables, enum | the arc's fixed ceiling reference |
| `given_native` | true tables, both ports | the native ceiling — what consolidation buys when nothing must be earned |
| `spiral` | live earn → consolidate → earn natively, both ports | the treatment |
| `spiral_route` | routing only, no span | which port carries the effect |
| `enum_live` | live earning, no ports | the non-native crank — `tall/`'s original question, run in the regime it needed |
| `fid` | both ports wired, both shut | the in-tag bit-for-bit assertion (`given_fid`'s role) |

`never_base`'s depth-necessity role moves to the pre-loop calibration ladder plus the in-arm
width-ladder instrument, per `tall/`'s approved design. If budget forces a cut, `given_native` is
the most cuttable (its content is a depth-6 replication of `native/`), and say so out loud.

**Readouts that make this a spiral claim rather than a replication**:

- **The rate claim, per turn.** Cycles-to-certification at L2 and at L3 (and provisional-vs-cert
  at each boundary), native vs enum arms. Accelerate / hold / decay is read here. `tall/cal0`'s
  clock-paced reference: L2 cert c12, L3 cert c22 — but from a loop that was not learning, so
  retracted as a prior; measure it fresh.
- **Per-turn crank health.** |T3| candidates at `mine_support`, committed recall/precision at both
  levels, the freed-budget-buys-width readout at each commit, and the observation-stream count
  (distinct level-shaped tuples at support in chosen trajectories — the table-free key from
  `native/full`'s battery) per era: the diet wall would show here first.
- **The wall autopsy, dissociated by instrument.** Diet: stream thinning per level (recurrence
  thins ~s× per level; `reread/lm`'s corpus wall). Grader: audition calibration at L3 (the
  consumption-matched audition from `ear/`; the arc's most-repeated blocker). Corridor: span
  parity trajectory per level (L3 heads mostly never cleared τ at depth 4 — an L3 span is 8 leaves
  at both depths, so depth 6 dissociates span length from coverage fraction, which the chunk-rent
  observation entangled).
- **The can't-decompose signature per level** (π's mass on a proposed chunk's own spelling) — does
  it stack as levels consolidate, per native interpretation (d)?
- **Consumption eras 3–5**: cost-to-depth on the priced ledger — what the earned-and-consolidated
  stack is worth where nothing more can be earned.
- **Plant guard every probe** (parse/infill on clean configurations, flat is the pass), fidelity
  twins in-tag, per-arm streams, every new RNG consumer on its own generator.

**Phases.** Phase A (~1 GPU-h): inherited gates at depth 6 (tall's T-1…T-4 pass stands; re-run
cheap), native's fidelity gates on this substrate (k = n_moves ≡ enum bit-for-bit; ports-shut ≡
donor), then the **descent gate with routing live** — era-1 descent rate and the commit-cycle
pricing readout, `n_corrupt=1`, vs `dens0`'s −0.004/cycle. Task-matched collection damage is the
one untested densification lever `tall/` named; Phase A is where it is cheap to try if descent
needs it. Orchestrator reads Phase A reduced results before any main launch — sizing (era
lengths, budget, probe cadence) comes from Phase A's own measured cycle cost, not from
projections. Phase B: the main run, single seed, target ≤ ~6 GPU-h with probes re-priced.

## Integration surface (the known risk, named)

Three lineages meet here: `native/full/full.py` (both ports + live miners already composed — the
donor to fork, per the `handle/` convention, fork notice at top, donors untouched), `tall/tall.py`
(the depth-6 substrate transplant: `_tall_cfg`/`_tall_shared`'s interface-audit lessons,
`preflight`), and `ear/`→`recital/`'s commit policy (certify-else-provisional-at-boundary; the
G-S scope fix — manufactured auditions are only defined on the earning prefix `1:25,2:12`).
`full.py`'s run loop is ratchet-lineage with a fixed in-tag commit schedule; the boundary-
provisional policy must be transplanted, not assumed present. The safety net is the arc's
fork-fidelity discipline: before anything at depth 6, `spiral.py` with the tall transplant
switched off must reproduce a `native/full` arm **bit-for-bit** at depth 4 (`fidelity_check.py`
in `critic/` is the model), and in-tag twins carry the floors thereafter.

## What the outcomes might mean (predictions, not constraints)

Held loosely, so the next agent can disagree: the reading the arc points to is a crank that at
least *holds* — native L2 makes L3 cheaper to earn (finding 7's mechanism, compounded), the L3
cert fires interior as `cal0` hinted, and the consumption eras show the flattened cost-to-depth
`ratchet/` measured at depth 4. Acceleration would be the strong form of "A becomes input to B."
Decay is at least as informative *if the autopsy assigns it*: to the diet (stream thins before
the table covers), the grader (audition calibration degrades at L3), or the corridor (parity
never clears at depth). A descent gate that fails even with routing live closes `tall/` harder —
consolidation does not rescue value starvation, sharpening the substrate-redesign requirement —
and stops the round at Phase A having learned that. Live provisional commits may also re-ask
native finding 6 unprompted: if a boundary commit lands a bad entry, does π quarantine it in-run?
Bring the numbers back for discussion before writing any README; idea-doc revisions stay queued.

## Practicalities and orchestration

- Invoke `/run-experiment-on-modal` first; profile `chromatic`; L4 GPU; smoke before every
  detached launch; halting procedure for anything >5 min; single seed, no multi-seed or DoP >4
  without orchestrator authorization.
- Structure per `STRUCTURE.md`: this folder is the node; `spiral.py`, `launch_detached.py`,
  `analyze_spiral.py`, `FILES.md`; this `SPEC.md` stays as the record of what was asked. Donors
  (`native/`, `tall/`, `ear/`, `recital/`, `ratchet/`) are never modified.
- **Delegation**: one implementer subagent owning the whole build loop — script, preflight,
  selfchecks, smoke, launch, reduce to summary + figures — per `/write-spec-or-prompt`; it reads
  `/subagent-instructions` before waiting on anything; two halts per phase (launch handle, then
  reduced results); the orchestrator keeps the wait and the interpretation. Subagents do not
  touch git.
