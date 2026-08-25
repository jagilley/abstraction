# SPEC — census: can an endogenous gauge buy the coverage the certificate can't see?

**The question in one sentence**: the spiral's certificate is satisfied by concentration — it
certifies a level at <10% coverage while mining runs ahead of the freeze — so can an endogenous,
label-free, coverage- or yield-shaped gauge at the frontier buy the coverage whose value is
already measured, and at what price in cycles, priced time, and foregone early nativization?

**Status**: spec, 2026-08-23. Nothing run. **Parent arc**: [`../README.md`](../README.md).
**Direct parent**: [`../spiral/`](../spiral/README.md) — findings 5 and 6 are the two measured
facts this round exists to connect: (6) the L3 certificate fired on 9–11 of 56 entries in every
earning arm, with live recall 0.14 vs frozen 0.054–0.089 and 75–77 audition answers that a
live-table swap would change; (5) the complete vocabulary beats the earned sliver by **+0.3–0.4
recovered fraction in eras 4–5** at matched ports — the value of coverage beyond the demand
distribution the sliver was earned on.
**Gauge donors**: [`../teacher_slot/`](../teacher_slot/README.md) (the scarce thing is *which
gauge is consulted*; a one-level-up gauge makes the call the same-level gauge refuses) and
`../teacher_slot/endo_yield/`[^private] (the label-free form of the
next-level-yield read). Within-level saturation gauges: [`../recital/`](../recital/README.md)'s
admission-rate pacer (behaved exactly as predicted there — and was still early: saturation ≠
sufficiency, which is precisely the distinction this round prices).
**Substrate**: `../spiral/spiral.py` forked (the `handle/` convention; donors untouched) —
depth 6, m=2, `sp_s0`'s exact configuration and era ladder unless Phase 0 gives a measured
reason to resize era 2. Single seed; ranks, signs, floor-multiples are the claims.

## Why this experiment

The spiral's grader wall is not noise; it is a type error in the gauge. The certificate audits
*the quality of what is held* and cannot see *how much of the level exists to hold* — so the loop
sincerely concludes level 3 is finished at 5–9 entries of 56, freezes, and pays for it exactly
where demand outruns the sliver (eras 4–5). Whether that price must be paid is not obvious in
either direction: the arc's own prior lessons say concentration dominates coverage *within*
demand (`merge/`, `setlist/`, spiral finding 5 eras 1–3), and the ratchet's timing law says
holding a commit open is itself catastrous when it delays nativization (`ratchet/`: one cycle
early is worse than never; the spiral measured what early native width buys). So the round is
not "gates good, certificates bad" — it is a *pricing* question: what does bought coverage cost,
what does it return in deep consumption, and does any endogenous signal price it correctly?

Two structurally different ops can buy coverage, and they should not be conflated:

- **Gate-later**: hold the level-3 commit open until a coverage/saturation signal quiets. Buys
  coverage at the price of delayed nativization (no routing width for the level while open) —
  the op `recital/`'s pacers implemented within-level, now aimed at the frontier.
- **Commit-then-extend**: commit at the certificate as now (early native width, whose value the
  spiral measured), and let the recert step **extend** the frozen table with post-commit mined
  entries instead of only offering whole-table swaps (which never fire). The miner is monotone
  and the ratchet filter structural (`tall/`: the table cannot churn), so extension is additive
  by construction. Buys coverage at ~zero timing price; its risk is exactly the selection risk
  recert exists to grade.

The second op is the spec's guess for the winner, held loosely — it is also the reading under
which the address book stays *open* rather than merely surviving deletion. The gauges that could
license either op, all label-free: the **admission rate** at the frontier level (new distinct
tuples at support per cycle — already logged), the **frozen-vs-live divergence** (recert's
`n_diff`, already logged, currently consumed by nothing), and the **next-level yield** read
(does the candidate stream one level up — T4-shaped tuples at support, observable even though L4
is unearnable — respond to holding more of L3? `endo_yield`'s currency, one level down).

## Phase 0 — the free experiment: replay every gauge over `sp_s0`'s logs (no GPU)

Every candidate gauge is a function of per-cycle series `sp_s0` already logs (admission rate,
distinct-at-support, `n_diff`, live/frozen table sizes, T4-shaped stream counts if the miner's
observation keying extends one level — check; if not, this one alone needs the live run).
**Before any paid run**: replay each gauge offline over all four earning arms' logs and report,
per gauge × arm × level — when it would have fired, how many entries the table held at that
cycle, and how far behind the live table the frozen one would have been. This is the census
taken retroactively. Constants are then chosen a posteriori from measured behavior (the
`tall/`-era-sizing precedent), and any gauge that is degenerate on the logs (never fires, or
fires with the certificate) is dropped from the live round with the replay as the record.

## Phase 1 — the live round (single seed, target ≤ ~5 GPU-h)

Arms (first pass — final list is the builder's with reasons on the record; every arm carries the
full spiral instrument set including the shadow certificate, `committed_grade`, `aud_oracle`,
and commit pricing events):

| arm | what it is | the question |
|---|---|---|
| `spiral_route` | `sp_s0`'s incumbent, re-run in-tag | the reference; also the in-tag fidelity anchor (bit-identical to treated arms until their gate diverges) |
| `given_native` | true tables, both ports | the coverage ceiling (eras 4–5: +1.048/+0.858) |
| `census_extend` | commit at cert; recert extends the frozen table with post-cert mined entries | the zero-timing-price op |
| `census_gate` | L3 commit held open until the Phase-0-chosen coverage gauge quiets | the gate-later op |
| `yoked_delay` | commits at the same cycle `census_gate` did, by clock, not signal | the timing control — dissects criterion from delay |
| `census_yield` | gate or extend licensed by the next-level-yield read | only if Phase 0 shows the yield gauge is non-degenerate and *different* from the coverage gauge — otherwise cut, with the replay as the reason |

Routing-only ports throughout (`spiral_route`'s configuration): finding 9 (the both-ports
consumption gap) is unresolved, and letting it into this round would confound the coverage
readout with the span question. Say so in the node's caveats.

**Readouts**: end-of-run L3 coverage (entries/recall/precision) and the frozen-vs-live gap per
arm — did the op buy coverage at all; **eras 4–5 recovered fraction against the
`spiral_route` → `given_native` bracket** — what the bought coverage returned, the round's money
readout; the T4-shaped observation stream under fuller vs sliver L3 — the growth direction,
observable without earning L4; cycles-to-native and width trajectory for `census_gate` vs
`yoked_delay` vs `census_extend` — the timing price, on the ledger; grader-cost accounting
(extra auditions/recerts the extension op pays); plant guard, twins, per-arm streams as always.

## What the outcomes might mean (predictions, not constraints)

Held loosely: `census_extend` closes some real fraction of the eras-4–5 gap at ~zero timing
price (the address book takes new addresses; selection-before-regression keeps extension safe —
the poison-quarantine mechanism is already measured); `census_gate` buys similar coverage but
pays visibly in delayed width, with `yoked_delay` showing the delay alone is a cost (the
ratchet's timing law, re-measured at the frontier). The informative failures: extension buys
coverage that *doesn't* move eras 4–5 (concentration was sufficient after all, and the
certificate was right to be satisfied — the "benign sliver" reading, which would demote the
grader-wall claim to a measurement note); no endogenous gauge separates from the certificate on
the logs (the census cannot be taken from inside — the teacher-slot thread's strongest form);
or the yield gauge alone works where coverage gauges don't (sufficiency is readable only one
level up — `teacher_slot/`'s rung A½ landing at the frontier). Bring the numbers back before
any README; the growth-direction reading stays conditional on the T4-stream instrument being
mechanically sound (check before claiming).

## Practicalities and orchestration

- Invoke `/run-experiment-on-modal`; profile `chromatic`; L4; preflight before paid setup; smoke
  before detached launches; halting procedure; single seed; DoP ≤ 4; no multi-seed without
  orchestrator authorization.
- Phase 0 is CPU-side analysis of already-fetched `figures/sp_s0/` logs — no Modal spend; its
  report is a decision document the orchestrator reads before Phase 1 is sized.
- Fork discipline as always: fork `../spiral/spiral.py`, fork notice at top, `# [census]`
  markers at insertions; `spiral/` and everything upstream untouched. G-F-style gate: with the
  census ops off, this file replays `spiral.py`'s `spiral_route` arm bit-for-bit in-tag.
- The extension op must go through the existing recert grading path (selection, not append):
  an entry enters the frozen table only by passing the same audition machinery a commit passes.
  Extension events, entries added, and their grades are logged per cycle.
- Structure per `STRUCTURE.md`: this folder is the node (`census.py`, `analyze_census.py`,
  `launch_detached.py`, `FILES.md`); this `SPEC.md` is the record of what was asked. One
  implementer subagent owns the whole build loop per `/write-spec-or-prompt`; it reads
  `/subagent-instructions`; two halts per phase; the orchestrator keeps the wait and the
  interpretation; subagents do not touch git.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
