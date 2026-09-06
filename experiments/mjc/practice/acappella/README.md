# a cappella — the model-free port: the grounding economy halts at its own gate, and the meter's native feedback axis opens a segment-span niche at biological delay

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Contract**: [`SPEC.md`](SPEC.md) (2026-08-27, plus the same-day amendment re-siting Phase B) ·
**Files**: [`FILES.md`](FILES.md) — every decision with its measured reason, the pre-fixed criteria,
and the flags, unsmoothed.
**Why this node exists**: [`../accompanist/README.md`](../accompanist/README.md) — the FM was the
incumbent's free accompanist; every prior mjc win/loss was conditional on who held the model.
**Substrate donor**: [`../etude/`](../etude/README.md) (world, piece, metering; donor untouched;
gate A-F asserts bit-for-bit fidelity, both noise regimes). **Machinery donors**:
[`../offbook/`](../offbook/FILES.md) (delay-operator idiom, `SlotLayout`),
[`../legato/`](../legato/README.md) (nesting, launch keying).
**Roadmap**: `ROADMAP.md`[^private] §4.5 Track M — this node is the motor
de-confound; it runs beside the spine and blocks nothing.
**Runs**: `a0` (Phase A, 3383 s) · `b1` (Phase B1, 694 s), both seed 0, 16 CPUs, **no GPU and no
forward model anywhere** — nothing in this folder imports, trains, or evaluates an `f(s,u)`.
**Attribution**: the no-FM constraint and the re-site decision are Jasper's (2026-08-27); the
pre-fixed halt that forced the re-site fired exactly as the SPEC committed it to.
**Ranks, signs, bit-identity gates and pre-fixed criteria are the
claims.** One orchestrated conversation, one implementer agent.

## The question, and what happened to it

The SPEC asked whether `rhm/practice/native/`'s consolidation happens on a plant when the
incumbent **pays for every look and every trial in the world** — RHM's economy (a declared
per-decision grounding budget, every candidate materialised on the plant) ported into the body's
currency. Phase A answered a prior question the SPEC had gated on: that economy **cannot run on
this plant**, for a measured reason, and the pre-fixed halt stopped the node's original Phase B.
The amendment then re-sited the economy on the meter's native motor axis — priced feedback under
an honest observation delay — and Phase B1 found the niche the grounding economy could not
express. Both halves are the de-confound Track M asked for.

## Findings

1. **The apparatus is clean and the confound is gone.** Gate A-F: 13/13 donor constants
   (`ast`-read, never imported) and the forked traversal reproduces étude's bit-for-bit
   (max|Δ| = 0.000e+00 on states, commands, per-segment error, fb count and priced time, noise-free
   and under motor noise). B-F1: the delayed code path reproduces `a0` exactly at Δ = 0
   (0.000e+00, six checks), so Δ is the only variable in B1. A grounding = one open-loop rollout
   on a resettable plant copy, 0.87 ms at 16 processes (measured; threads are GIL-bound and
   slower than single-threaded). (`a0`, `b1`)
2. **The grounding economy does not transfer to the plant — the pre-fixed halt fired.** The
   budget ladder descends monotonically to 0.0612 at G = 4096 (16,384 real rollouts per
   traversal) without saturating — 16× étude's own *imaginary* budget — so G\* = 4096 is
   ladder-limited. The search reaches étude's `never` band (0.10–0.11) and beats it; but the
   reflex law (PD on the waypoint schedule, gains fit once on the clean world) plays the piece at
   **0.0045 for 16.9 s priced**, and **0 of 15 cells** of the whole search family beat it on
   error — best cell 0.0607 at 15,011 s, 13.5× worse at 888× the price. Per the SPEC's clause,
   the priced planner "is not an incumbent worth beating", and Phase B as originally specified
   was not run. The reason is structural, not a tuning failure: a motor trial costs the time to
   perform it (0.916 s priced) while a feedback event costs 0.10 s, and on this piece feel is
   sufficient — on RHM a grounding was cheap in absolute terms (an edit has no duration) and
   mandatory (no continuous feel exists). (`a0`, gate A-I)
3. **The meter's native motor form is feedback pricing.** The rent table separates arms by
   feedback events, not groundings: reflex 136 fb / 0 g · keyed segment tapes 4 / 0 · nested
   library 3.4 / 49 · whole-piece chain 1 / 0–8. The library alone beats the search on both axes
   (0.0352 at 49 groundings vs 0.0612 at 16,384); routing's grounding cut — `native/` finding 1's
   currency — would be 13 of ~16,400 (0.08%), invisible. The glossary's generalization ("the
   meter: priced feedback, generalized to any pressure against enumeration") runs backwards on a
   plant: the generalized form has no bite, the original form does. (`a0`, gate A-R)
4. **Under honest delay, a model-free segment-span niche opens at biological latency.** With an
   observation delay Δ on every agent-side feedback consumer — and, for the first time in this
   arc, *nothing to bridge it* (`accompanist/` d3b showed the naive operator is a strawman
   exactly when a predictor exists) — the pre-fixed gate passes: Δ\*_niche = 4 (96 ms), a fragile
   +0.0054 single point that closes again at Δ = 6, and a **sustained niche from Δ = 8 (192 ms)**:
   stored segment tapes at 4 fb beat the reflex — re-fit at every Δ, every advantage given — by
   2.0× (0.0838 vs 0.1697), widening to 5.6× at 384 ms (0.107 vs 0.600), all inside the
   playability guard (`ref_play` = 0.1066, étude's `never`, fixed blind). Degradation over the
   sweep is ordered by feedback consumption — reflex **134×**, segment arms 1.65–3.10×, chains
   1.00× — `offbook/` finding 7's law, reproduced with the confound removed, and its 7d
   ("memory first pays one level down, at the segment span") now model-free. (`b1`)
5. **The chain question is structurally unanswerable on this piece — pre-committed, not
   discovered.** Both chain arms decide once at seam 0, where étude's loop starts from rest, so
   their observation is the true state at every Δ and they are *exactly* delay-invariant (0.0852 /
   0.1184 to four decimals). The depth-ordering pass at Δ = 12/16 is this artifact, flagged
   before launch: a property of the piece (offbook's had an approach leg before its first seam;
   this one does not), not of depth. No chain number in this node is evidence about chunking.
   (`b1`, pre-registered in the SPEC amendment)
6. **The audition is the most delay-fragile selector in the system.** At Δ = 0, plant audition is
   nearly calibrated (construction gaps 1.07–1.23 at segment span, 0.92–1.12 at chain span,
   against étude E-3b's 2.7–3.4× optimism — executing the audition dissolves the scorer-optimism
   confound that clouded `offbook/`), and the winner's curse still reproduces in construction
   (counterfactual 0.697 vs chosen 0.068 in one cell), so selection hygiene is substrate-general.
   Under delay the picture inverts: audition calibration walks 1.00 → 1.77; within the same 4-fb
   tier the coarse frozen key degrades *less* than the fine audition (1.65× vs 2.90×); and
   `lib_all` — the arm free to choose chains — selects them least exactly where they pay most
   (`frac_chain` 0.394 at Δ = 8 → 0.040 at Δ = 16, while the pure chain arm is flat-best from
   Δ = 12). The op that would select depth is the first thing delay breaks. Suggestive, not
   measured: routing (π) reads a posture and needs no fresh rollout per decision, so trust has a
   structural edge over re-grounding selectors under delay — that comparison is B2's, queued.
   (`a0` A-C, `b1` flags 6–7)
7. **Smaller banked instruments.** Étude's finding 2 (the mean of valid sequences is not a valid
   sequence) reproduces on real rollouts (elite mean vs best sample, 2.3× at G = 4096). Seam
   information on this piece is marginal (matched spread ratios 1.07–1.13), as the SPEC's
   held-loosely anticipated, while per-state keying still buys 1.22–2.35× over the best fixed
   slot. `frac_chain` is nonzero (0.09–0.39 across arms/Δ) — the first chain consumption in the
   mjc arc, read lightly. (`a0` A-S/A-B, `b1`)

## Interpretation (discussed with Jasper 2026-08-27 — argued, not measured)

The de-confound is complete in both directions. The RHM grounding economy cannot run on this
plant — not because the port was built wrong (the fork is exact, the ladder honest, the rule
pre-fixed) but because embodiment prices trials in performance-currency time and this piece is
steerable by feel, so search-by-real-trial is dominated by an option RHM structurally lacks. And
the economy that does exist on a plant — feedback under delay — gives committed content its first
model-free niche, at latencies in the biological reflex band, one level down from the chain:
the regime where ballistic units pay in animals ("impossible closed-loop, possible from memory")
reached from the incumbent's side with nothing imaginary anywhere in the loop. The motor row can
now carry its claims in the roadmap; what it says so far is segment-span, and says nothing about
depth.

## Caveats

- 24 shared eval geometries; the b1 sweep uses n_eval = 32.
- **The reflex gain grid hits edges** at Δ ∈ {3, 4, 6} (kp min) and {12, 16} (kp min and kd max).
  The Δ = 8 fit is interior, so the sustained-niche point is clean; the fragile Δ\* = 4 margin
  (+0.0054) could plausibly flip with a wider grid, and the Δ = 16 gap (7×) could not. The Δ = 3
  cell fit clean *better* than Δ = 0 (0.0026 vs 0.0030) — read as a coarse-grid oddity, not an
  effect.
- **G\* is ladder-limited** (last rung; 2048 misses the 5% band by 3.8%): "the search never
  saturates below the reflex" is bounded at G ≤ 4096, not proven in the limit. A warm-started
  planner was deliberately not built (the gap it addresses is 13.5× at 888× cost).
- **The chain arms' delay-invariance** (finding 5) caps what B1 can say about depth at exactly
  the segment level. This is the piece's property; the fast-piece escalation is the fix.
- The delay operator is naive by design and nothing bridges it; an arm with plant access *could*
  bridge Δ by re-executing its own issued commands on a resettable copy (efference copy without a
  model). Recorded as deliberately-not-taken — for the reflex and the frozen key the naive
  operator is a necessity, for the auditioning arms a choice.

## Runs on disk

| tag | what |
|---|---|
| `asmoke` | `--quick` smokes of the full Phase A chain, every gate exercised before launch |
| `a0` | Phase A: A-F fork fidelity, the reflex calibration, the G ladder + tempo instrument, gate A-I (the halt), the nested library, A-C audition calibration, A-R rent, A-S seam information, A-D cycle costs |
| `b1` | Phase B1: the Δ ladder — B-F0/B-F1 identity gates, per-Δ reflex re-fit, 6 arms × 9 Δ, the pre-fixed niche gate and depth test, the record search rung at Δ = 0 |

## Reproduce

```bash
cd experiments/                      # MODAL_PROFILE=chromatic
modal run mjc/practice/acappella/profile_cost.py::acappella_profile
modal run mjc/practice/acappella/gates.py::acappella_gates --quick --tag asmoke
modal run --detach mjc/practice/acappella/gates.py::acappella_gates --spawn --tag a0 --seed 0
python3 mjc/practice/acappella/analyze_gates.py --tag a0 --fetch
modal run --detach mjc/practice/acappella/delay_gate.py::acappella_delay --spawn --tag b1 --seed 0
python3 mjc/practice/acappella/analyze_delay.py --tag b1 --fetch
```

Volume `mujoco-control-data`: `/data/practice_acappella/{a0,b1}/`; fetched copies and reports
under `results/{a0,b1}/`.

## Next steps (queued, not started — each with its licensing condition)

- **B2 — the π/trust round at Δ = 8**, `offbook/` O1–O3's shape at the niche delay, segment
  span. Licensed if Track F's RHM/canvas instruments produce a trust law that needs a
  physical-plant, biology-facing test (the roadmap's "if adoption occurs" conditional is now
  met at segment span). Deliberately not run on momentum: it would largely re-confirm the
  standing offbook π-facts one level down, and the depth half is out of this piece's reach.
- **The presto escalation** — fork `../accompanist/presto/`'s 120 ms-segment world (a piece
  whose first seam is not its start, killing finding 5's artifact) under the same fork-fidelity
  discipline and the same pre-fixed criteria. Licensed if the chain/depth claim becomes
  load-bearing for the biology-facing story; it is the prerequisite for any deep-unit question
  on this substrate.
- `/update-beliefs` for the unit: the meter's motor form is feedback, not groundings; the
  delay-fragility of re-grounding selectors; the segment-span niche as the model-free form of
  offbook 7d.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
