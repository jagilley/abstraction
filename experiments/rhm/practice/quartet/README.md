# quartet — four instruments for the unification node: what survives absorption, and what forms trust

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: the 2026-08-31 roadmap
synthesis (ROADMAP §7.1–§7.2 read against PRs #82/#84): before the two-currency unification node
runs, which introspective signals are actually readable, and what does trust formation respond to?
**This is the joint writeup** (the `typed_gaps`/`two_deltas` pattern) for one conversation's four
parallel lanes — the member machinery carries `FILES.md` records and no READMEs by policy:
[`../woodshed/`](../woodshed/FILES.md) (F1: the trust-formation-rate instrument + the rehearsal
arm) · [`../ostinato/`](../ostinato/FILES.md) (the δ_perf benchmark-estimability sweep) ·
[`../tuning/`](../tuning/FILES.md) `wledger.py` (the weight-ledger online typer) ·
[`../continuo/`](../continuo/FILES.md) PASS 2 (`corridor*.py` — E2 on the live executor).
**Direct donors** (untouched): [`../two_deltas/`](../two_deltas/README.md) (the round this
follows; `intonation`'s live executor is the substrate for two lanes) ·
[`../tuning/`](../tuning/README.md) + [`absorption_blinds_the_evaluator`](../../../../ideas/absorption_blinds_the_evaluator.md)
(the banked checkpoints and the §4.3 prediction the typer tests) ·
[`../census/`](../census/README.md)+[`../assay/`](../assay/README.md) (arrival-dominates — the
bracket `woodshed` re-opens) · `papers/forward_self_models_paper2.md`[^private]
(guards and observer-twin discipline).
**Runs**: `wd_smoke`/`wd_smoke2`/`wd_s0`/`wd_s1` (3.90 GPU-h) · `os_gf`/`os_s0` (3.34 + ~0.7 lost
to one Modal preemption) · `co_gf`/`co_s0` (≈1.45) · `wl0` (CPU) — **≈ 9.9 GPU-h**, 2026-08-31.
One orchestrating conversation; four delegated implementer agents; every fork gated bit-identical
at 0.000e+00 against its donor with its knobs off.
**Attribution**: the four lanes were scoped in the roadmap-synthesis discussion (Jasper's
RHM-first call; the real-FM variable deliberately confined to the E2 lane); the question-port
reframe that closed the round's discussion is Jasper's (`OPEN_QUESTIONS` #3, now a QUEUE entry).
The `wd_s0` dose bug and the `os` premise correction were found by the implementers' own
instruments, not by design.

## The questions

The unification node (QUEUE) will put both currencies — outcome and execution — over the crank
at once. Four inputs to that node were unmeasured: (1) is the weight ledger readable *online*,
where `tuning` showed the stream ledger drains (the absorption doc's predicted instrument)?
(2) does the executor-side consolidation state decode off a sub-saturation FM's residual once
pass 1's novelty confound is designed out (E2 pass 2)? (3) at what rate does trust form after a
vocabulary arrives, and can rehearsal of a *received* vocabulary compress the clock that gift
could not skip (F1)? (4) is δ_perf's benchmark regime-dependence really §13(c)'s estimability —
does thinning the benchmark's evidence alone reproduce scarcity's reversal (`two_deltas`
finding 6's candidate mechanism)?

## Findings

1. **The weight ledger types events online, and it does not drain** (`tuning/wledger.py`, `wl0`,
   CPU over 328 banked fp16 checkpoints, 8 arms). Three scale-free per-layer-group ‖Δθ‖
   log-ratios (`c_emb`, `c_gain`, `c_comp`), z-scored against each arm's own quiet intervals,
   type rotation/burst/drift **12/12 by argmax** on the consume-everything arm (min margin
   +0.65σ; 90/90 cross-type rank-consistent; 11/12 fire correctly under the arm's own quiet
   gate), readable at 25-step lag for burst/drift and 125 for rotation — inside the op map's
   +50…+125 decision window. The load-bearing contrast with [`tuning`](../tuning/README.md)
   finding 2: across rotations 1→6 the weight read holds ×0.79 (+6.45 → +5.11σ) **where the
   stream read decayed +0.090 → −0.004 over the same six events**; burst ×0.98; drift ×0.65
   (the most-learnable event, once under floor). Event-matched nulls are clean — arms whose
   wall never rotates fire 0/6 at rotation instants while firing 6/6 on their real bursts and
   drifts, and merge kills rotation news on this ledger too (+5.4 pre-merge → +0.3 after). The
   directional read earns exactly one column: the burst/ln_f reflex template (cos +0.955,
   honest margin +0.125 over a 0.828 quiet floor); rotation and drift directions sit at their
   floors. And the meter-type law reappears on the weight side: skipped windows have Δθ
   **exactly 0** — the ledger is silent precisely where the arm declined to learn.

2. **The executor-side instruction/data bit still does not decode — with the age confound
   designed out rather than controlled for** (`continuo` PASS 2, `co_s0`, 2 arms × 110
   per-cycle span-head snapshots; snapshot hook certified inert at 0.000e+00 on all 13 series).
   The command/data object was re-founded on the **parity gate** — open = `SpanHead.emit`
   materializes the span in one pass (compiled), closed = `apply_any` enumerates it — which
   varies within level, within slot over time (re-closures included), and is decoupled from
   mint age, so pass 1's fatal "command ≈ young" collinearity cannot recur. The decode fails
   anyway: pooled self-over-observer advantage has within-cell sd ≈ **4.7× the largest mean**
   with signs flipping across draws and projection dims; the I/O observers win the
   per-checkpoint decode outright (−0.12…−0.19); the guard panel reads calendar-and-slot, not
   content (`cycle` η² +0.22…+0.29 ≈ `slot_id`, grammar ≤ 0.05); the `ens_cos` junk band
   reproduces at H=16 (all four reductions) with the deterministic ridge again the readable
   instrument; and the raw `open` label loses its negative class under age floors (every slot
   eventually opens — a lost class, not a decayed effect; `open_tau` carries the grid). The
   δ_perf-gain arm is *not* the arm that keeps a bit — the contrast sits inside draw spread.
   **One cell is sign-stable across both arms × both draws**: `cmass_R ~ at_support@L4`
   residualized on cycle+era, +0.11…+0.23 — pass 1's surviving cell, reproduced on the
   executor side.

3. **Trust has a rate law, and credited rehearsal of a received vocabulary re-opens the
   arrival bracket** (`woodshed`; the offline instrument over 13 tags / 101 (arm, level)
   cells, then `wd_s1`). The instrument (a lift normalization forced by masked-softmax
   semantics — a level's slots re-enter carrying untrained logits ≈ the uniform share)
   reproduces five cross-tag anchor replays bit-identically and census finding 6's headline
   exactly, for free. The rate law: every **earned** arrival is driven *below chance* within
   one probe interval (median trough lift 0.13–0.24) before climbing back, and the
   crossing-back accelerates with depth (median cycles-to-chance-share 70 / 30 / 13 at
   L2/L3/L4); the **gift** family never dips and the two families are disjoint at every
   matched τ. `audiation` 8(ii) decomposes: the per-datum arm's weaker consolidation is
   entirely an L2 effect (argmax 0.000, mass declining) with L3 at or above its anchor.
   Offline, executions add nothing once the clock is partialled out (median partial r +0.038)
   — which is exactly the collinearity the rehearsal arm breaks. At corrected dose (`wd_s1`;
   `wd_s0` is retained on the record as a measured near-null-dose run — its `corrupt_hier`
   all-nodes bug posed 8-simultaneous-corruption questions, solve rate 0.003 vs 0.151
   corrected, and *nothing cleared the in-tag noise handle*): targeted rehearsal of the
   received L3 orders **credit > exposure > none on every trust statistic** (argmax
   0.555 / 0.375 / 0.240; the one bit separating the treated arms is whether π imitates the
   rehearsal trajectories that *solved* or an equal number drawn at random from the same
   episodes) **and on deep-era value** — closing 0.70/0.69 of the anchor→gift bracket at
   eras 4/5, vs 0.29/0.34 exposure-matched and 0.18/0.28 untreated — with the L3 movement
   3.7×/2.1× the unrehearsed-L2 divergence handle. Arm-order invariance was verified
   empirically across the two tags (identical terminal entry selections with `exact` run 4th
   vs 2nd).

4. **The benchmark's evidence requirement is real, √-law-shaped — and its value curve is a
   valley, not a crossover** (`ostinato`, `os_s0`, 6 arms; `os_r100`/`os_log` double as
   full-scale bit-identity twins of `in_s0`, 0.000e+00 over 13 series × 131 cycles). The
   premise correction came first, from the donors' own logs: `ma_s0`'s scarcity did **not**
   remove context recurrence (calls per slot 14.85 → 14.26 at era 5) — it halved the
   performances behind each benchmark update (159 → 75 rows; ρ_eff 0.47). The knob is
   therefore `perf_bench_frac` ρ: a uniform subsample of each execution's rows feeding
   `b(s)`'s EWMA, with volume, difficulty, diet, and the update instants all held. The ladder
   moves benchmark precision exactly as designed (signal/SE 6.79 → 3.51 → 1.60 → 1.32 across
   ρ ∈ {1, 0.2, 0.04, 0.008}; measured sd(b) tracks the √-law to ~10%). On value, the answer
   is not the hypothesized interior optimum: deep-era recovered fraction vs the budget-matched
   raw arm is **negative at every rung including full evidence** (ρ=1: −0.09/−0.11 at eras
   4/5, ~1.0–1.3× the earning floor) and **non-monotone with its worst cell in the interior**
   (ρ=0.04: −0.43/−0.61, 5.0×/7.0× floors; the one-row-per-update rung recovers to
   −0.09/−0.04). Meanwhile the benchmarked form keeps the best L4 trust (argmax 0.281 vs raw
   0.258 vs uniform 0.185) and the §H counterfactual has every benchmarked rung *delaying* the
   L4 crossing (+3/+14/+7 cycles) where raw and uniform hold the clock. The 2×2 stays
   populated at every rung (ill-executed solves 13–19% of the diet; P(solve|intended) 1.3–1.6×
   P(solve|ill)).

## Joint interpretation (discussed with Jasper 2026-08-31 — argued, not measured)

- **(a) Two morals carry the round, and they are the arc's old morals in sharper form.**
  *Durable evaluator signals live in what learning creates or preserves, never in the stream it
  consumes*: finding 1 is the absorption claim's predicted positive instrument, measured
  (the ledger absorption *writes* stays loud exactly where the ledger it *drains* went quiet),
  and finding 2 is the same law from the failing side — a stream-of-states read whose residual
  is dominated by the conditioning gap's boring drivers (calendar, slot, age). And
  *selection-by-success is the active ingredient*: matched exposure without credit is nearly
  inert (finding 3's one-bit split), the E2 lineage's only survivors live under per-datum
  credit or as the `at_support` aggregate, and the `wd_s0`→`wd_s1` dose contrast shows even
  credit is inert when the questions posed are unsolvable — the band-pass shape the belief
  tree assigns to learning progress, observed as an accident.
- **(b) `two_deltas` finding 6 likely decomposes.** With the raw arm budget-matched, raw ≥
  benchmarked on deep-era *value* in both regimes (finding 4 at abundance; `rawx` at
  scarcity), while benchmarked wins *trust/head* statistics in both — so "the regime reverses
  the gain ordering" reads instead as the old raw arm's 0.40× budget confound plus a stable
  **value-vs-trust dissociation** in what the benchmark buys. And the estimability condition
  is real but inverted in consequence: a half-estimated benchmark is *worse* than either a
  good one or effectively none — it injects its own noise into every judgment — so §13(c)
  should be read as a licensing condition for consuming `b(s)` at all, not as a dial to
  optimize. The `two_deltas` §(a) "executor plasticity: benchmarked where contexts recur"
  seat should be read with this correction pending its own test.
- **(c) Arrival dominates — softened, constructively.** `census`/`assay` established that
  content transfers and trust does not; `woodshed` shows trust is nonetheless
  *manufacturable*: gift + credited rehearsal recovers most of the from-birth advantage that
  gift alone leaves on the table. This is the receiver-side of the instruction-tuning framing
  (ROADMAP §7.1.3's "F1 as E′'s other half"): a received vocabulary is someone else's commands
  without the issuing policy, and issuing them yourself under feedback — with credit — is what
  forms the trust receipt didn't. It is also the candidate recipe for `native/lm` (deferred):
  pretrained structure converted to wielded structure by manufactured practice.
- **(d) The E-side inputs to the unification node are now settled by measurement.** The veto
  seat reads the weight ledger (finding 1; free at train time in its Δθ form); the value seat
  reads `at_support`; per-token self-decodes enter nowhere (finding 2, twice over). The FM's
  surviving-seat list (`two_deltas` §(d)) is unchanged in membership but its
  implementation-fact seat now has a positive form (the weight ledger) beside the repeatedly
  negative one (residual decodes of consolidation state).
- **(e) The round's instruments are post-hoc by construction, and the discussion's close named
  the remaining exogenous seat** (Jasper): demand. Every node in the arc takes its questions
  from outside; naive self-demand is a measured negative (`merge`/`fourwall` β=2), the
  allocation loop is a measured positive (`full_loop`), and finding 3's dose contrast shows
  question *quality* alone can move bracket closure by ~0.4. Question-choice as an outer-loop
  action — with an oracle bisection ceiling and Δ`at_support`-per-priced-sample as the
  endogenous question-judge — is queued (`OPEN_QUESTIONS` #3 → QUEUE "the question port").

## Caveats

- The rehearsal result rests on one corrected-dose run against an in-tag divergence handle
  (n=3 unrehearsed-L2 cells); **no independent floor for π mass exists anywhere in the arc**,
  and building one is part of any verification pass. Rehearsal is also not compute-matched
  (recorded in `g`; ~2× practice compute on treated arms) — the claims are trust-per-cycle.
- `os_r001`'s realised ρ is 0.0148, not 0.008: the one-row-per-update floor binds, so the
  bottom rung is "one performance per update," and the interior-valley shape is one rung deep
  at its extreme. The arms are clock-yoked; §H is a counterfactual, not a realised pacing
  difference.
- The typer is a trailing-window read (windows must contain the onset; +125-start windows
  collapse), its three axes are hand-specified from Gate 1G's subspaces (a generic 12-group
  template does worse on merged arms), and checkpoint deltas understate event excess
  (quiet-side coherence exponents 0.52–0.65) — the exact train-time gradient form is the
  queued instrument.
- `co_s0`'s four L4 corridors are trivial (`e` ≡ 0.0000), the FM sees the head's global read
  rather than its per-offset input (chosen to avoid re-importing a level signature), and
  occupancy TV drift is tie-dominated at this probe size.
- `wd_s0` (near-null dose) and the `os` preemption/no-resume-guard behavior are recorded in
  the member `FILES.md`s as apparatus lessons, with fixes specified.

## Runs on disk

| lane | node | tags | what |
|---|---|---|---|
| typer | `tuning/` | `wl0` | per-layer-group Δθ ratios over the g1a/g2a/g2c checkpoints, CPU |
| E2 pass 2 | `continuo/` | `co_gf`, `co_s0` | span-head snapshots on the live executor; 2 arms × 2 draws reduced |
| F1 | `woodshed/` | `wd_smoke(2)`, `wd_s0`, `wd_s1` | offline trust-rate reduction + the rehearsal/exposure pair |
| estimability | `ostinato/` | `os_gf`, `os_s0` | the ρ ladder + rung-invariant comparators |

Volumes: `rhm-scaling-data:/data/rhm_practice_{tuning_wl,corridor,woodshed,ostinato}/<tag>/`.
Reductions and figures under each member's `figures/`.

## Reproduce

Each member's `FILES.md` carries its full command set (preflight, G-F/gate, smoke, main,
reduce): [`../woodshed/FILES.md`](../woodshed/FILES.md) ·
[`../ostinato/FILES.md`](../ostinato/FILES.md) · [`../tuning/FILES.md`](../tuning/FILES.md)
(§wledger) · [`../continuo/FILES.md`](../continuo/FILES.md) (§PASS 2). Fork lineage (each fork
gated bit-identical with its knobs off): `woodshed.py → assay.py`;
`ostinato.py → intonation.py`; `corridor.py` monkeypatches `span_net.parity` over
`intonation`'s run body; `wledger.py` imports `gate1g`'s loaders verbatim.

## Next steps (queued, not started)

The question port (QUEUE; needs the question-parameterization spec) · F2 with F1's instrument
and the `at_support@L4` cross as candidate gauge · the unification node (QUEUE; findings 1/4
set its veto and plasticity seats) · the train-time gradient form of the typer · the
benchmark-licensing test implied by (b) (consume `b(s)` only above a measured signal/SE
threshold) · `wd_s1`'s verification pass alongside a π-mass floor instrument · the
`/update-beliefs` sweep carrying findings 1–4 (the 2026-08-31 sweep predates this round's
reductions).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
