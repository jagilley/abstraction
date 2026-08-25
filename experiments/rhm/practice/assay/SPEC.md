# SPEC — assay: which ingredient of the complete vocabulary is the currency?

**The question in one sentence**: the census showed bought coverage doesn't close the deep-era
gap and the forensics showed why it might not have — junk admissions de-fund the whole level's
routing (π vets levels, not entries) while the one precision-preserving addition helped — so
this round performs oracle surgery on the committed table itself, crossing **amount × truth ×
arrival time**, to measure which of them the deep-era bracket actually pays for.

**Status**: spec, 2026-08-23. Nothing run. **Parent arc**: [`../README.md`](../README.md).
**Direct parents**: [`../census/`](../census/SPEC.md) (`cs_s0` + the forensics pass —
`census/figures/forensics/report.txt` is required reading and carries every number cited here) ·
[`../spiral/`](../spiral/README.md) (the substrate and the eras-4–5 coverage measurement).
**Substrate**: `../census/census.py` forked (the `handle/` convention; donors untouched) —
`cs_s0`'s exact configuration: depth 6, m=2, routing-only ports, G=482, ladder
`1:25:48,2:12:40,3:6:12,4:3:9,5:1:7`, certify-else-provisional commit policy. Single seed;
ranks, signs, floor-multiples are the claims; if a key contrast lands small, a seed pair is the
orchestrator's call afterward, not part of this round.

## The measured state of the question

Three facts from `cs_s0` + forensics set up the design:

1. **The precision split.** `census_gate` added +2 L2 / +3 L3 entries at *unchanged* precision
   and beat the anchor in 4 of 5 eras; `census_extend` added +4 L2 / +12 L3 at *degraded* L3
   precision (0.333 → 0.250) and lost eras 3–4. Same op family, opposite signs, split on purity.
2. **Level-granularity quarantine.** π's action space is one action per (level, node) slot —
   which table entry serves a call is the executor DP's unlogged choice — so π cannot starve a
   junk entry, only the level. Measured: the arm with the most L3 entries ended with the lowest
   L3 proposal mass (0.158 vs anchor 0.233), lowest argmax share (0.099 vs 0.234), and fewest
   macro expansions. Pollution de-funds the level, true entries included.
3. **The arrival-time trace.** 80.3% of `given_route`'s era-4 advantage over the earning arms is
   present at c16 — before any commit — and the gap grows to c116 (era-5: 40.8% at c16). Partly
   definitional (an arm holding true macros probes deep cells better whenever it holds them);
   what surgery can recover of it has never been measured.

## Design — the surgery arms (held loosely; the instrument list is the commitment)

At each level's own commit cycle, the arm commits a **constructed** table instead of (or in
addition to) its mined one; everything else — policy, pricing, schedule, streams — is the
anchor's. Twins are bit-identical until the first surgery (the L2 commit), which is the in-tag
fidelity gate. The true tables come from `MC.true_tables` (oracle surgery is the round's method,
stated openly; nothing endogenous is claimed for the ops here — this round measures *what the
bracket pays for*, not *how a loop could buy it*).

| arm | table committed (both levels, consistently) | axis isolated |
|---|---|---|
| `anchor` | own mined table (`spiral_route` replica) | the reference |
| `strip` | own mined, junk removed at commit | truth ↑ at amount ↓ — does purity alone restore π's trust in the level? |
| `complete` | own mined ∪ all missing true entries | amount ↑ and truth ↑ (L3 ends 56 true + own junk, precision ≈ 0.875) — pure coverage's return with junk retained |
| `exact` | the full true table, nothing else | the complete-dictionary *content* at commit-time **arrival** — vs `given_c1` this is the arrival-time axis at identical content |
| `junk_dose` | own mined ∪ K plausible junk entries, K matched to `complete`'s addition count | amount ↑ at truth ↓ — the poison dose, predicted to reproduce the L3 mass-drain |
| `given_c1` | true tables from cycle 1 (`given_route` replica) | the c1-arrival ceiling, in-tag |

Junk for `junk_dose` should be *realistic* junk — candidates actually mined-but-false somewhere
in `cs_s0`/`sp_s0` (the kind selection produces), not uniform random tuples; say what was used.
If six arms strain the budget, `strip` is the most cuttable (its content is a subset of what
`complete` vs `junk_dose` brackets), and say so out loud. Estimated cost at `cs_s0`'s measured
10.4 s/cycle: 6 × 116 ≈ **2.1–2.6 GPU-h**.

## Instruments (the two the forensics could not compute are the round's additions)

- **Per-execution entry identity**: log which table entry the executor DP selects at every macro
  execution (entry id per call; truth keyed offline in the reduction). This splits the poison
  mechanism cleanly: if the DP already filters junk at execution time (picks true entries from
  impure tables), the damage is *trust-damage* (π-level) and not *execution-damage* — the
  forensics could not distinguish these because the choice was unlogged.
- **π per-level proposal mass, argmax share, and macro-expansion counts at every probe** — the
  mass-drain signature, now watched prospectively in a controlled dose design.
- The standard set: shadow certificate per level (surgery arms still log what the certificate
  *would* have done), `committed_grade`, eras-3–5 bracket on recovered fraction and raw e
  (`anchor` → `given_c1`), the c16-style all-eras probe series from c1 (the arrival-time trace,
  per arm), G-Y, plant guard, commit pricing events, per-arm streams.

## What the outcomes might mean (predictions, not constraints)

Held loosely, and the forensics' own guesses stated so the run can contradict them:
`complete` and `exact` close a large fraction of the deep bracket (pure coverage is currency);
`junk_dose` reproduces the L3 proposal-mass drain and loses eras 3–4 (level-granularity
quarantine as *mechanism*, prospectively); `strip` helps more than its −amount would suggest
(purity restores the level's funding). `exact` vs `given_c1` is the round's most open cell:
near-equality says the deep gap is table-content and arrival hardly matters (surgery could in
principle recover it — the "lifetime" reading demoted to definitional); a large residual says
when the vocabulary arrives matters beyond what it contains (trajectory/value shaping is real
currency no table surgery reaches — the census's reading B, finally isolated). The DP
entry-choice instrument decides whether junk hurts by being *executed* or by being *carried*.
Informative failures welcome: `complete` failing to close the bracket despite 0.875 precision
would say even pure coverage isn't the currency and push the whole question to reading B.
Bring the numbers back before any README; census + forensics + assay are one epistemic unit and
will be written up together.

## Practicalities and orchestration

- `/run-experiment-on-modal`; chromatic; L4; preflight before paid setup; smoke before detached
  launch; halting procedure; single seed; DoP ≤ 4; no multi-seed without orchestrator
  authorization.
- Fork `../census/census.py` (fork notice, `# [assay]` markers); `census/`, `spiral/` and all
  donors untouched. G-F: with surgery off, `assay.py` replays `census.py`'s `spiral_route`
  bit-for-bit in-tag (the anchor arm doubles as the full-scale carrier, as `cs_s0` did via the
  shared-prefix check). Surgery tables constructed once at setup, logged verbatim in
  `setup.json` (entries, truth split) so the treatment is auditable.
- Structure per `STRUCTURE.md`: this folder is the node (`assay.py`, `analyze_assay.py`,
  `launch_detached.py`, `FILES.md`); this SPEC is the record of what was asked. One implementer
  subagent owns the whole build loop per `/write-spec-or-prompt`; `/subagent-instructions`
  before waiting; two halts (launch handle, reduced results); the orchestrator keeps the wait
  and the interpretation; subagents do not touch git.
