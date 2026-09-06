# ear — climbing the evaluation layer: the grader grid, LP starvation, and the provisional-commit rescue

**Up**: [../README.md](../README.md) (rhm/practice) · [../../README.md](../../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1's third component — re-instantiation of the evaluation layer, the last never built; §14's
Next steps (i) is this round's charter; §15 points here)
**Parents**: [`../ratchet/`](../ratchet/README.md) (the earned vocabulary, and the level-3 refusal
this round exists to fix) · [`mjc/practice/etude/`](../../../mjc/practice/etude/README.md) (the
compile op and the seam law)
**Child / successor**: [`../recital/`](../recital/README.md) — removes the era clock; its
fidelity replay (`rf_s0`) reproduces this round bit-for-bit and its findings revise one of this
round's interpretations (noted in place below).
**Status**: written up 2026-08-16 (results discussed with Jasper 2026-08-15). Five runs: one
mechanism calibration, three detector calibrations, and the 8-arm main run.
File index: [FILES.md](FILES.md).

## One-liner

The ratchet's certificate refused level 3 because its audition was mis-levelled — the grader had
not climbed with the vocabulary. This round climbs it, on two coordinates at once: audition
**context** (era-k damage / self-manufactured level-k+1 / real level-k+1) × **evaluator**
(single action / in-policy). Climbing fixes the **measurement** — the consumption-matched
in-policy audition is calibrated to 0.97 — and does **not** fix the decision: every certificate
arm still refused level 3, because arms already holding a level-2 vocabulary leave a candidate
level-3 unit almost no headroom to demonstrate. What rescued era 3 was **committing provisionally
at the era boundary and letting consumption do the grading**: `practice_prov` and
`practice_climb` are the only arms in the arc to that point that beat `given` itself, and the
live recert they carry as a safety net fired **zero times in 24 recerts** — boundary commitment
is safe because mining is already selection-filtered.

## Findings

The fidelity gate comes first because everything else stands on it.

1. **Fidelity**: `never_base`, `given` and `practice_gated` are **bit-identical to
   `ratchet/rr_s0` across all 90 cycles** (max|Δe| = 0, max|Δt_cum| = 0). The evaluation layer
   did not leak into the substrate; the mis-levelled-grader control is a literal re-run.
2. **The seam law is now three-coordinate, and the climbed grader closes it.** Seam ratio
   (realised competence ÷ audition) at level 3: the ratchet's era-k/single-action cell reads
   **0.53–0.62** (pessimistic — the refusal that cost `rr_s0` its era); the self-manufactured
   in-policy cell reads **1.3–2.1** (optimistic); the real-context in-policy cell reads
   **0.97**. Audition must match consumption in **state** (étude), **level** (ratchet), and
   **evaluator** (this round). At level 2 all graders sit at 0.79–0.93.
3. **Climbing buys certification speed where certification works.** All four level-2 certificate
   firings landed within one cycle of their offline replay predictions; the climbed grader
   (`practice_self`, mfg/policy) fired at **c14** against the ratchet grader's c20.
4. **Learning progress starves at the frontier.** Arms holding a level-2 vocabulary have level-3
   audition spans of **0.07–0.17** where a from-scratch probe had 0.30–0.37 — compounding
   competence eats the headroom a new unit can demonstrate, so "descend, then go silent" never
   coincides and every certificate arm refused level 3 (accumulated descent at silence 0.068–0.172
   against `lp_min_drop` 0.10). The margin counterfactual (`pol_cand − pol_base`, logged every
   cycle) would have refused everywhere too: the unit's marginal value peaks mid-era and erodes
   as the closed-loop policy improves.
   *Revised by [`../recital/`](../recital/README.md)*: with self-paced time, the level-3
   certificate **does** fire given ≥35 cycles in era 2 — the 30-cycle era clock was cutting it
   ~5 cycles short. The corrected statement is that LP at depth is *slow and was
   clock-truncated*, not dead; the refusal-at-30-cycles reproduces exactly (`fixed_climb`), and
   the arms that did earn non-provisional L3 certificates still lost to a bottom-heavy schedule
   committing provisionally — certification is not what sets the grade.
5. **The `taught` paradox: best instrument, worst outcome.** `taught` reads the best-calibrated
   grader in the run (seam 0.97) and finished with the worst earned era-3 error (0.463) — a
   truthful audition of a genuinely low-marginal-value unit truthfully reports no learning
   progress, and the gate refuses. A perfect measurement fed to a starved decision rule loses to
   a cheap decision rule with no measurement at all.
6. **Provisional commitment rescues era 3.** `practice_prov` (commit at the boundary, grade only
   in consumption, priced recert): era-3 e = **0.321**, earned-vs-given **1.067**, and the only
   negative cost-to-depth slope in the arc to that point (**−0.036** — it got better as damage
   deepened). `practice_climb` (certificate where LP is alive, provisional fallback at the
   boundary): **0.303**, earned-vs-given **1.128**. Both beat `given` (0.341); the ratchet
   control `practice_gated` sits at 0.438.
7. **The recert is a clean negative that reassigns credit.** 0 swaps in 24 recerts
   (frozen−live mean −0.0004, max +0.0000, against a 0.05 margin): frozen tables never fell
   behind live ones. Boundary commitment is safe not because re-grading protects it but because
   **mining is selection-filtered** — only solved repairs enter the table, at a support
   threshold — so quality control happens upstream at mining time. The ratchet's poison
   (`practice_early`) was a *too-early* problem, not a no-certificate problem.
8. **Evaluation is expensive: 4.8–29.2% of priced time** (a policy audition is a beam;
   the pre-run estimate of ~1–2% is retracted). The cheapest grading policy — grade only in
   consumption, 4.8% — finished second. The most expensive arm (`practice_climb`, 6.22M ≈
   `given`'s 6.14M) finished first.
9. **The manufactured-audition oracle check, with one retraction.** Self-manufactured contexts
   track real ones at the calibration's coverage-ladder scale (corr +0.95 to +0.99), and
   `calm_s0`'s carried claim "manufacturing is faithful" holds in-loop at level 2 (corr +0.75 to
   +0.98) but **collapses at level 3** (several arms ≈ 0): across cycles where the candidate
   barely moves, noise dominates; the paired bias/mad readout is what survives (L2 bias −0.024
   to −0.089; L3 bias up to −0.248). Two measured sources of the mfg cells' optimism: seeding
   tests from the agent's **own solved** derivations biases the test easy by −0.066 (L2) /
   −0.158 (L3), and the `node_empty` signature mismatch (0.61–0.77 vs real 0.000) persisted from
   the gate into the loop.

## Results — `er_s0` (seed 0, 8 arms, 3 eras × 30 cycles, complete, 1564 s)

`e` = mean over each era's last 5 cycles; `t` = era priced time. Denominators for the
earned-vs-given fraction `(e_never − e_arm)/(e_never − e_given)`: 0.148 / 0.290 / 0.301.

| arm | grader read | era 1 | era 2 | era 3 | t_cum | e-v-g era 3 |
|---|---|---|---|---|---|---|
| `never_base` | — | 0.349 | 0.530 | 0.642 | 4.21M | — |
| `given` | — | **0.201** | **0.240** | 0.341 | 6.14M | 1.000 |
| `practice_gated` (ratchet ctrl) | era/act | 0.223 | 0.288 | 0.438 | 4.95M | 0.678 |
| `practice_late` | era/act | 0.282 | 0.273 | 0.355 | 5.25M | 0.953 |
| `practice_self` | mfg/pol | 0.249 | 0.258 | 0.439 | 5.70M | 0.676 |
| `taught` | real/pol | 0.211 | 0.271 | 0.463 | 6.00M | 0.597 |
| `practice_prov` | none (prov + recert) | 0.356 | 0.288 | **0.321** | 5.41M | **1.067** |
| `practice_climb` | mfg/pol + prov fallback | 0.221 | 0.289 | **0.303** | 6.22M | **1.128** |

Matched priced time vs `never_base`'s own trajectory at era 3: `practice_prov` +0.320 ·
`practice_climb` +0.315 · `given` +0.294 · `practice_late` +0.271 · `practice_gated` +0.224 ·
`practice_self` +0.211 · `taught` +0.193.

### Certificate firings vs offline prediction

| arm | grader | era | lvl | fired | predicted | entries | recall | prec |
|---|---|---|---|---|---|---|---|---|
| `practice_gated` | era/act | 1 | 2 | c20 | c19–20 | 8 | 0.500 | 0.875 |
| `practice_self` | mfg/pol | 1 | 2 | **c14** | c13 | 7 | 0.429 | 0.857 |
| `taught` | real/pol | 1 | 2 | c21 | c20 | 10 | 0.643 | 0.900 |
| `practice_climb` | mfg/pol | 1 | 2 | c26 | c13–c23 | 9 | 0.571 | 0.889 |
| `practice_self` | mfg/pol | 2 | 3 | **REFUSED** | fire c19 | — | — | — |
| `taught` | real/pol | 2 | 3 | **REFUSED** | no-fire | — | — | — |
| `practice_climb` | mfg/pol | 2 | 3 | c60 prov fallback | fire c19 | 11 | 0.143 | 0.727 |
| `practice_prov` | — | 1,2 | 2,3 | c30, c60 prov | — | 10, 12 | 0.643, 0.179 | 0.900, 0.833 |

`taught`'s refusal was predicted at launch (its L3 span for an arm holding L2 is 0.099 <
`lp_min_drop`); `practice_self`'s and `practice_climb`'s were not — the calibration's
from-scratch probe had 3× the span of a vocabulary-holding arm, which is finding 4.

### Grader-cost accounting

| arm | priced total | grader paid | share |
|---|---|---|---|
| `never_base` | 4.21M | 1.23M | 29.2% |
| `given` | 6.14M | 1.38M | 22.5% |
| `practice_gated` | 4.95M | 1.27M | 25.6% |
| `practice_self` | 5.70M | 0.78M | 13.7% |
| `taught` | 6.00M | 1.27M | 21.2% |
| `practice_prov` | 5.41M | **0.26M** | **4.8%** |
| `practice_climb` | 6.22M | 1.05M | 16.8% |

### Instruments

Mined-vs-random reproduces at L2 in every arm (−0.17 to −0.22, 29–30/30) and at L3 for the
provisional arms (−0.11 to −0.18); plant guard flat in all 8 arms across 90 cycles (the descent
is vocabulary-carried, as in the ratchet); reader 1.000 throughout.

## Interpretation (discussed with Jasper 2026-08-15 — argued, not measured)

- **(a) Measurement and decision came apart.** Grader climbing was specified as the fix for the
  level-3 refusal; it fixed the audition (0.97) and the refusal remained, because the LP gate is
  starved by the very competence the vocabulary compounds. Crystallize gave the certificate its
  first scope condition (practice must move something the unit depends on); this round adds the
  second: **an LP gate is informative only while the policy is far from the unit's value** — at
  the frontier, commitment must come from the boundary, not the certificate. (See finding 4's
  revision note: recital sharpens "starved" to "slow and clock-truncated", and shows the
  conclusion — certification does not set the grade — survives the sharpening.)
- **(b) The teacher thread gains its second half.** Interpretation (e) of the ratchet read
  teaching as transfer of the grader. `taught` shows a lent, already-climbed grader is real and
  the best-calibrated instrument in the run — and still loses if the decision rule starves. The
  thing that transferred era 3 was the *boundary*: the recital, not the ear. Phenomenological
  anchor, recorded as anchor: performers report you never feel ready at the frontier; the
  concert date is what commits the piece.
- **(c) Practice-room optimism is measurable.** The self-manufactured audition is easier than
  the real thing by a measured −0.07 to −0.16 (self-seeding) plus a signature mismatch
  (`node_empty`), and its optimism (1.3–2.1×) is the stage-vs-practice-room effect in numbers.
- **(d) Where to grade is an allocation question.** The evaluation layer competes for the same
  priced budget as learning (finding 8); grading only in consumption was nearly free and nearly
  best. Component (1) of the idea doc's definition applies to the grader too.

## Calibration record

| run | measured | change it forced |
|---|---|---|
| `selfcheck` | ratchet's C-M/C-R/G-D + **G-N** (ladder nested: era k's enclosing level-(k+1) node is era k+1's cell), **G-S** (manufactured contexts on-grammar 1.000, confined to the node span; `node_empty` 0.56–0.66 vs real 0.000), **G-R** (live recert restores the L3 table the ratchet's cap truncated, 14 → 56) | design admissible; the `node_empty` mismatch became a calibration cell rather than an assumption |
| `calm_s0` | manufactured-vs-real audition corr +0.95 to +0.99 on every variant/evaluator; whole-node damage fixes `node_empty` but breaks `frac_broken` and does not improve agreement; self-seeding bias −0.066 (L2) / −0.158 (L3); in the realistic 1–8-entry regime the action evaluator is flat-to-backwards at L3 while the policy evaluator descends 0.688 → 0.547 | `mfg_source=child`, `mfg_render=canon`; the policy evaluator is the load-bearing coordinate; the self-seeding bias is a measured cost, not a bug |
| `cald_s0` | fork fidelity (bit-identical, 30 cycles); grader-grid span/noise — the mfg cells at L2 were **0.9 / 1.8** because the self-manufactured test waited for a full derivation bank and did not exist until c5, after the descent | `mfg_min_bank` = 32 with the set sized to what is banked — without this, `practice_self` would have failed level 2 for a bookkeeping reason and then found level 3 unrepresentable (`T3 ⊆ T2 × T2`), a structural artifact masquerading as a result. Also fixed: the teacher's audition set was keyed to the successor era existing rather than to the level being earned, so it went missing exactly in the last era |
| `cald3_s0` | with the fix, L2 `mfg/act` span 0.026 → 0.773 (s/n 0.9 → 9.8); every L2 cell 8.1–11.5 | admissible |
| `cald2_s0` | L3 grid: `era/pol` **3.1** (worst — the mis-levelling reproduces in the policy evaluator), `mfg/pol` 5.8–6.5, `real/pol` 5.7–5.9; and for an arm holding L2 the mfg *base* drifts 0.323 — larger than the candidate series' own span | `n_aud` = 192; the drift confound is logged every cycle for every cell |
| offline replay | drop 0.05 and 0.10 fire identically in 15/16 cells; where they differ, 0.05 fires mid-descent; margin-based reading refuses L3 on every cell for an arm holding L2 | detector kept **bit-identical to the ratchet's** (c 0.06, c_v 0.15, W 5, hold 2, `lp_min_drop` 0.10) for all arms — the grader is the only variable between arms; the margin counterfactual logged, reconstructible offline |

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | attached `--quick` smoke |
| `calm_s0` | calibration 1: is a manufactured test a faithful test; action vs policy evaluator range |
| `cald_s0` | detector calibration + fork-fidelity gate (superseded `cald_s0` variant kept for the record) |
| `cald2_s0` | L3 detector calibration with the corrected teacher set |
| `cald3_s0` | L2 detector calibration with the readiness fix |
| `er_s0` | the main run: 8 arms, seed 0, 3 eras × 30 cycles |

Volume (`rhm-scaling-data`): `/data/rhm_practice_ear/<tag>/<arm>/results.json` + `setup.json`,
`cal_mfg.json` for the calibration entrypoint. Fetched copies live in `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/ear/ear.py::selfcheck_remote
python3 rhm/practice/ear/launch_detached.py --fn cal_mfg --tag calm_s0

# the main run (flags as in recital/FILES.md's rf_s0 replay, which reproduces it bit-for-bit)
python3 rhm/practice/ear/launch_detached.py --fn ear --tag er_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_late,practice_self,taught,practice_prov,practice_climb" \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 \
    --n-aud 192 --mfg-source child --mfg-render canon --mfg-min-bank 32 \
    --recert-every 5 --recert-margin 0.05 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4

python3 rhm/practice/ear/analyze_ear.py --tag er_s0 --fetch --figures --fidelity
```

## Caveats

- **The era-3 ordering (`prov`/`climb` > `given`) has margins (0.020–0.038) that sit near the
  ±0.034 stream-position noise recital later measured.** The certificate behaviour is
  reproduced in 3 worlds by recital's `rc_s0/s1/s2` (which carry `fixed_climb` — this round's
  `practice_climb`). The qualitative rescue
  (0.30–0.32 against the mis-levelled grader's 0.44) is well clear of that floor; the
  beats-`given` reading specifically should be held as suggestive.
- The mfg-vs-real correlation collapse at L3 (finding 9) means the manufactured audition's
  fidelity at depth is certified by paired bias, not by correlation.
- `practice_prov` pays era 1 (no vocabulary until its c30 boundary commit), so its era-1 cell
  reads like `never_base`; its grade is an era-2/3 statement.
- The recert's clean negative is a statement about **this substrate's oracle-verified mining**
  ("solved" = terminal possible-set success). Where success labels are noisy or self-assessed,
  selection-filtering is weaker and re-grading may earn its keep.

## Next steps

Consumed by [`../recital/`](../recital/README.md): with the grader climbed and the commit
policy settled (certify-else-provisional), the last handed-over piece of curriculum was the era
clock itself — remove it and ask whether any signal internal to the agent can place the
boundaries.
