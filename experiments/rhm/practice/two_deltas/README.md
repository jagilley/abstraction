# two_deltas — Track E′ round 2: two δs got one name, and only one of them was ever zero here

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: `ROADMAP.md`[^private]
§7.1.3 (Track E′), re-pivoted mid-round by the two-δ diagnosis (below).
**This is the joint writeup** (the `typed_gaps` pattern) for one conversation's five pieces —
the member nodes carry `FILES.md` machinery records and no READMEs by policy:
[`tacet/`](../tacet/FILES.md) (E3′, the outcome-δ gate) ·
[`../audiation/`](../audiation/README.md) `fit.py` §9 (the finding-8(i) re-encode check) ·
[`continuo/`](../continuo/FILES.md) (E2, consolidation off the residual) ·
[`intonation/`](../intonation/FILES.md) (δ_perf: the live executor) ·
[`caesura/`](../caesura/FILES.md) (δ-silence as the commit decision).
**Direct donors** (untouched): [`../crescendo/`](../crescendo/README.md) (the A-track stack every
node here forks, via `tacet`) · [`../native/span/`](../native/span/FILES.md) (Port 2: the span
head and parity gate `intonation` un-gates) ·
[`../../../mjc/two_clocks/`](../../../mjc/two_clocks/README.md) and
[`ideas/performance_error_is_the_bridge.md`](../../../../ideas/performance_error_is_the_bridge.md)
§1/§13–14 (δ_perf's corrected form and the separate-channels law) ·
[`../../confabulation/temporal/epistemics/`](../../confabulation/temporal/epistemics/README.md) +
`papers/forward_self_models_paper2.md`[^private]
(`continuo`'s guards and observer-twin discipline).
**Runs**: `tc_s0` (2.85 GPU-h), `au_s1/fit` §9 (CPU), `continuo` pass 1 + follow-up chain (CPU),
`in_s0` (3.01), `ca_s0` (2.82), `ma_s0` (3.38), plus smokes/probes ≈ 1.1 — **≈ 13.2 GPU-h**,
2026-08-29→31. **Ranks, signs, and multiples of measured floors
are the claims.** One orchestrating conversation; four delegated implementer agents.
**Attribution**: the two-δ diagnosis and the live-executor pivot arrived from Jasper via a
second agent's research pass (2026-08-31), verified here against the record before anything was
built; the strongly-metered hypothesis is Jasper's; `tacet`/`continuo`/the re-encode check ran
*before* the pivot and are re-read by it.

## The question, and the diagnosis that re-formed it mid-round

Track E′ (ROADMAP §7.1.3) asked whether a per-trajectory scalar gate on decision-time signals —
δ and the deliberation state, no forward model anywhere — changes what the table contains, what
trust forms, and whether the range extends. Round 1 (`tacet`, `continuo`, and the queued
re-encode check) ran that program as written. The diagnosis then landed: **the arc had two δs
under one name.** The bridge line's δ is a **performance error** —
`δ_perf = (b(s) − e)·σ((g−g₀)/θ)`, where `e` is the mismatch between what the learner *meant*
to execute and what it *did* execute, benchmarked per context and gated on agency — practice's
own metering component ([`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§1, component (2)). What `audiation`, `tacet`, and ROADMAP §7.1.2 called δ is
`grade − v(s)` — an **outcome surprise**. [`two_clocks`](../../../mjc/two_clocks/README.md)
(3 seeds) showed credit factorizes into *precision* (which experiences — δ_perf's content) ×
*value* (toward what end — the grade's content) and that sharing one channel is destructive
interference. And on this stack δ_perf was **zero by construction**: the A-track composition
executes macros by `macros.apply_any`, an exact DP over the committed table, so intention and
realization coincide, `e ≡ 0`. Round 2 (`intonation`, `caesura`, `ma_s0`) un-gated
`native/span`'s head below its parity threshold so realization became fallible, and `e` went
live on RHM for the first time — with the intention reference **exact and free** (the DP's own
spelling; no learned FM, no oracle: the reference is the learner's own committed table).

## Findings

1. **The outcome-δ gate re-derives the shared-channel result — content is real, the channel is
   wrong** (`tacet`, tc_s0, 6 arms clock-yoked to one bit). Ranking the learner's diet by any
   decision-time content beats a random gate at exactly matched volume (δ-directions at era 3:
   1.69×/1.17× the earning floor; δ-lo at era 5: 1.19× pooled / 1.00× in-node), the two δ
   directions select **disjoint** halves (kept-set Jaccard 0.000), and the deliberation margin
   selects nearly orthogonally to both (corr 0.11–0.13) — but every gated arm loses to
   learning-from-all-successes above every floor at eras 3–5 and forms roughly half its L4
   trust (mass 0.057–0.131 vs 0.152). Yoke caveat on the record: each gate arm's own gauge
   would have committed 3–17 cycles away from the baseline's optimum (§H), so diet and pacing
   mismatch are not fully separated.

2. **`audiation` finding 8(i) does not survive its queued control — and the inversion flips
   back.** The symmetric re-encode grid reproduces all five published Q4 cells exactly, then
   shows the observer's 8× advantage on spillover direction was format plus leak: at matched
   sparse basis the ordering reverses (`z` 0.304 vs `O_x` 0.289); the incumbent event-level
   split leaks instance identity (`(cycle, inst) ↔ (cycle, x0)` is a bijection; a group-mean
   predictor with **no inputs** reads 0.399, above every cell); under an instance-grouped split
   every state code sits ≤ 0 and the **only surviving source in the grid is the learner's own
   per-datum free scalars** — δ, grade, log‖Δᵢ‖ — essentially untouched (0.062 → 0.060,
   capacity-flat). Finding 8(i) should not be cited as an observer inversion anywhere.

3. **The E2 residual's "instruction/data bit" is mostly a novelty tag — except under per-datum
   credit** (`continuo`, CPU-only on `audiation`'s 117 head snapshots; guards per paper 2, all
   held). Pass 1's +0.052 self-over-observer decode advantage decays monotonically to zero by a
   20-cycle token-age floor with the scalar-norm negative in lockstep, in all three
   draws — what the residual knows about a command token is that it is *new* — and draw
   variance exceeds the headline (+0.033…+0.075). Two cells survive every perturbation: the L2
   commit's occupancy elevation (top 21% of drift, 3/3 configurations) and
   `cmass_R ~ at_support@L4` (+0.37…+0.41). The arm contrast required an instrument fix (at
   matched `H` the per-datum arm's FM sat in the `ens_cos` junk band; re-read at the
   deterministic ridge rung): **the less-consolidated `perdatum` arm keeps the bit after the
   age control** (+0.083 at floor 20, +0.042 at 50, permutation p = 0.011) **while the anchor's
   dies** (−0.006, p = 0.125). Sub-saturation on this substrate means < ~6% of the predicted
   span — any future E2-shaped fit here must stay there.

4. **δ_perf exists, and the 2×2 the arc could never produce is populated in every cell**
   (`intonation`, in_s0, 5 arms; misfire 5.6–8.8%, all 28 slots firing, G-F and meter-inert
   both 0.000e+00). Honest failures (executed-as-intended, unsolved) outnumber as-intended
   successes **3.4×** in every arm; **13–22% of all solves are ill-executed** — the
   lucky-success cell the grade-only mining rule cannot see is a fifth of its diet; executing
   as intended raises P(solve) 1.27–1.57×. The agency structure is nearly binary: on 55–62% of
   executed rows the command explains nothing beyond the slot-free argmax (`g = 0`) and there
   `e` ≈ 0.001; where the command carries content, `e` ≈ 0.13–0.21. The ACT/PLAYBACK
   dissection reproduces the bridge's §13(a) signature (ungated `b−e` fires in both columns;
   the centered gate separates, ~30× at smoke scale). As consumption: the δ_perf **gain** on
   the executor's own training orders δ ≻ uniform ≻ raw on misfire and end parity with 1.52×
   the baseline's era-5 L4 argmax share (task value below floors; the raw arm's budget
   confound, realised weight 0.40×, is recorded and retired in finding 6); the δ_perf **gate**
   mines 1.84× fewer ill-executed answers than ungated at matched volume — and still loses the
   deep eras exactly as the outcome gate does. Un-consumed observation: the live executor alone
   moved the whole crank 25 cycles earlier (L4 at c104 vs c129).

5. **δ-silence, wired to the commit op it was designed for, out-paces the one-level-up yield
   gauge — from one rung behind the frontier** (`caesura`, ca_s0, 4 arms, deliberately
   un-yoked: pacing is the variable; the 201-cycle schedule arm is the lifetime control).
   Structural fact first, found by gate D before any GPU: the gauge is computed over open
   slots and slots are minted *by* a commit, so **δ-silence cannot license the first
   crossing** — `dsil_read` bootstraps L2 on the arc's default rule (logged, D-5-gated,
   strictly pre-gauge). Above the bootstrap, δ-silence *choosing* the crossings commits L3
   c71 / L4 c97 against the yield thermostat's c72 / c104 and holds **+0.304 / +0.388 /
   +0.415** over it at eras 3–5 (3.7× / 2.6× / 1.2× the given-family floors), forms the most
   L4 trust in the tag (0.367 vs 0.168), and **beats the 201-cycle schedule ceiling on 25
   fewer cycles** (+0.171/+0.191/+0.240, 2.0–2.8× the earning floor) — so the lifetime
   confound, though real, does not carry the result. The veto form (`dsil_and`) is fragile
   where the pacer form is not: its single one-cycle deferral of L3 crossed an era boundary
   and the rung was never committed at all — one event, not a rate, and recorded as such.
   Instrument tension for the record: by movement-to-floor `dsil` reads as *dead* (0.24,
   beside the refusing ledger's 0.27, far under yield's 1.78) — whatever D/floor measures, it
   did not predict this gauge's usefulness. In-tag floor 0.00492; the run was governed by the
   offline 0.00321 (1.53× tighter — the narrower claim is in-tag).

6. **Scarcity removes gating's cost and reverses the gain ordering** (`ma_s0`, 5 arms, regime
   certified in-run: 0.54–0.58× the mining cap, cap binding 4.6–5.9% of cycles vs 95.4% under
   abundance, π's buffer never full; below the cap the mining subsample is never taken, so
   **the mining channel is un-gateable and the gates act on π alone**). The gates' deep-era
   cost collapses from 1.9–2.7× the given-family floors under abundance to ~0.1× under
   scarcity, and the δ_perf gate now beats the outcome gate at eras 2–3 (1.31×/1.08× the
   earning floor) — selection stops hurting when data is dear, but still doesn't pay above
   floor against learning-from-everything. The gain trio **reverses sign against abundance**:
   with the raw arm budget-matched at last (`rawx`, realised mean weight 0.999 vs the broken
   0.402; the `perf_mean` coverage fix took gate ties 48.7% → 1.5%), the un-benchmarked raw-`e`
   gain posts the only above-floor value cells in the tag (+0.109/+0.148 at eras 4–5,
   1.25×/1.70× the earning floor) and 2.6× the baseline's L4 trust, while the benchmarked δ
   gain — abundance's winner on the head — drops to the worst L2 parity. Candidate mechanism,
   discussed and *not* measured: §13(c)'s estimability condition — `b(s)` is meaningful only
   under repetition per context, and scarcity is the removal of repetition.

## Interpretation (discussed with Jasper 2026-08-31 — argued, not measured)

- **(a) The metering signal has two seats with positive, floor-clearing evidence, and the gate
  seat is not one of them.** As the **commit pacer** (finding 5) — the compile-trigger seat
  [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
  §1 designed it for — and as **executor-side plasticity** (findings 4, 6), in a
  regime-dependent form: benchmarked where contexts recur, raw where they don't. As a diet
  gate on mining/π it fails in both regimes with both δs, now for a mapped reason
  (`two_clocks`' channel law under abundance; no leverage under scarcity) rather than as a
  bare negative.
- **(b) The within-level type law may be about currency, not level.** The within-level *ledger*
  refuses crossings optimally (`teacher_slot`); within-level *δ-silence* paced them better than
  the one-level-up yield gauge (finding 5). If this holds more generally, "a within-level
  signal cannot price the crossing" sharpens to "an **outcome-currency** within-level signal
  cannot" — execution-currency signals speak to *when* the current rung is compiled, which is
  its own kind of crossing information. Held loosely; verification deferred to later runs.
- **(c) No privacy claim anywhere in this round.** δ_perf is computable from public quantities
  by design and `continuo`'s advantage outside the per-datum cell is a novelty tag; the one
  place a self-side quantity uniquely survived its controls is the re-encode grid's per-datum
  scalars (finding 2) — which is consonant with the round's larger shape: the cheap scalars on
  the wire keep beating every richer construction *except* where execution itself is fallible,
  which is where δ_perf's two seats live.
- **(d) What the pivot retired and what it kept.** Retired: the FM as self-update forecaster
  (already, `audiation`), the outcome-δ gate as E3′'s object, and 8(i)'s observer inversion.
  Kept and sharpened: the FM's outcome-forecast seat (the value head), E2's per-datum cell,
  and — restored from the mjc line — the metering component, now measured on the substrate
  where every other practice op was calibrated. The mjc-side port with a *learned* intention
  reference (`acappella` lineage) is the designed next test of whether these seats survive
  estimation error.

## Caveats

- **One substrate throughout.** `caesura` is un-yoked by design and its deep-era deltas conflate
  pacing with practice time (the schedule arm bounds this); `dsil_and`'s rung loss is one
  event; `dsil_read`'s L2 is a bootstrap, not a δ-silence commit.
- The four L4 corridors are trivial (`e` ≡ 0.0000 exactly), so all live performance error is
  L2/L3 — the meter cannot bite where the corridor is degenerate.
- Under scarcity the mining channel is un-gateable (finding 6), so `ma_s0`'s gate cells are
  π-diet effects only; and `rawx`'s 2×2 dissociation collapses to 1.03× (its harder-trained
  head solves ill-executed trajectories at nearly the intended rate) — recorded, unexplained.
- The agency gate's self-calibration is degenerate (g₀ = 0) in most arms — the ACT/PLAYBACK
  exclusion is structural (an unexecuted row is never scored), not gate-carried.
- `in_s0`'s gate arms carried the 57% Σδ_perf = 0 tie fraction; fixed (`perf_mean`) only from
  `ma_s0` on. `in_s0`'s `perf_raw` is budget-confounded (0.40×) and superseded by `ma_s0`'s
  `rawx`; quote only the latter.

## Runs on disk

| node | tag | what |
|---|---|---|
| `tacet` | `tc_smoke`, `tc_s0` | outcome-δ gates, 6 arms yoked, live L4 frontier |
| `audiation` | `au_s1/fit` §9 | the re-encode grid (`--e1b-reencode`), CPU |
| `continuo` | pass 1 + `agefloor`/`fmseed777`/`seed1`/`ridge`/`perdatum(_ridge)` | E2 off `au_s0`/`au_s1` snapshots, CPU |
| `intonation` | `in_gf`, `in_smoke`, `in_s0`, `ma_probe`, `ma_s0` | δ_perf: the live executor; then the metered regime |
| `caesura` | `ca_gf`, `ca_s0` | δ-silence as the commit decision |

Volumes: `rhm-scaling-data:/data/rhm_practice_{tacet,intonation,caesura}/<tag>/` and
`/data/rhm_practice_audiation/au_s1/`. Reductions and figures under each node's `figures/`.

## Reproduce

Each node's `FILES.md` carries its full command set (preflight, G-F, smoke, main, reduce);
`continuo/FILES.md` for the CPU chain; `audiation/FILES.md` §fit for `--e1b-reencode`. Fork
lineage (each fork gated bit-identical at 0.000e+00 with its knobs off):
`caesura.py → intonation.py → tacet.py → crescendo.py → maestro.py → conductor.py`.

## Next steps (queued, not started)

Verification by later runs rather than seed pairs now (Jasper's call) · the
benchmark-estimability sweep (repetition per slot as the variable — finding 6's candidate
mechanism, directly testable) · a yoked `caesura` variant if commit *content* needs separating
from pacing · E3′'s free-paced outcome-δ arm (demoted by the diagnosis; still the clean
completion of `tacet`'s caveat) · the mjc port with a learned intention reference
(`acappella` lineage) · the ROADMAP §7.2 append recording the two-δ split · the
`/update-beliefs` sweep now carries findings 2, 5, and 6.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
