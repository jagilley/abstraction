# antiphon — the question port: selection moves the mining, the pacer converts it, and the ask directs the search rather than authoring the answer

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: [`SPEC.md`](SPEC.md)
(OPEN_QUESTIONS #3 → QUEUE "the question port" — the five threads it ties, the two payoffs, the
two failure poles; written 2026-08-31 to hand this conversation the design step) · **Files**:
[`FILES.md`](FILES.md) (main lane: parameterization §1–§3, gates §5, runs §6, the pacer
marginals §6b) · [`provenance/FILES.md`](provenance/FILES.md) (shape P: the offline pass, P′,
both P″ rounds, and the strict-vs-restricted gate question).
**Direct donors** (untouched): [`../crescendo/`](../crescendo/README.md) (fork base;
`anchor_long` and `outer_yield_m4` are the baselines, and its Phase-0 offline-sizing discipline
is the method here) · [`../woodshed/`](../woodshed/FILES.md)+[`../assay/`](../assay/FILES.md)
(P′/P″ fork base; the credit > exposure cell this round re-reads) ·
[`../quartet/`](../quartet/README.md) (the round that queued this; its finding 3's dose
contrast was the strongest prior).
**Runs**: `an_gf`(×2) · `an_smoke` · `an_s0` (7 arms) · `an_s1` (pacer-only) ≈ 5.99 GPU-h;
`ap_s0` (P′, logging-only) 1.61; `pp_smoke3`/`pp_s0`/`pp_s1` (P″ twice) ≈ 1.72; two CPU-only
offline passes (the shape-P premise check; the circularity check) — **≈ 12.3 GPU-h,
2026-09-01**, one orchestrating conversation, two delegated implementer lanes. Every fork gate
in the round at **0.000e+00**: G-F on two donor arms, the full-scale `q_exo` ≡
`cr3_s0/anchor_long` replay (13 series × 201 cycles), the `an_s1` ≡ `cr3_s0/outer_yield_m4`
certification (162 = 162), `ap_s0`'s recorder-on replay of all four `wd_s1` arms, and both P″
twin gates in the corrected first-consumed-cycle form.
**Attribution**: the question, the two-payoff framing, and the two failure poles are Jasper's
(OPEN_QUESTIONS #3; the SPEC). The parameterization, the three premise corrections, and the
certification design are the implementing lanes'. The follow-up asks (the endo era-4 read, the
`exaff_prop` circularity check, the pacer attribution) were relayed from a second agent's
review; the **director-not-author** synthesis in interpretation (c) came out of that review's
close and was adopted in the 2026-09-01 discussion.

## The questions

(1) Does question *quality* move the climb at matched priced budget — bisection vs the
exogenous ladder? (2) Can the learner's own value signal grade questions rather than answers?
(3) Does a question-judge steer between the two failure poles (the aleatoric nerdsnipe; the
β=2 comfort collapse)? (P) Is self/other provenance computable from inside — efference copies
of one's own asks — and does provenance-gated credit change what forms trust?

## Design in brief

**A question is a posed repair instance, selected from a menu.** Each cycle the world offers
K = 2048 candidates drawn from the era's own damage cell; the arm's selector picks the 64 it
practises on. The lever is exact by construction: the damage cell is one half of the
level-(ℓ+1) span that gets mined, so the selector controls half of every mined key. Rejected
with reasons: mining-side selection (a post-consumption diet gate, closed by `two_deltas`) and
node-choice (moves the demand support, violating the SPEC's one hard norm).

**Three premise corrections, all offline before GPU** (the `ostinato` discipline): (i) under
this parameterization the exogenous ladder *is* a uniform draw over cell instances, so
"exogenous" and "random questions" are one arm — which doubles as the round's load-bearing
fidelity gate; (ii) **the aleatoric trap is natively installed** — the grammar's own parse
ambiguity puts junk mass 0.24/0.58/0.82 at the L2/L3/L4 mining nodes (why every donor table
sits at precision 0.19–0.41), so A3′ needs no burst machinery: a novelty judge with the guard
removed is the nerdsnipe cell, free; (iii) **the claim is a clock claim, not an endpoint
claim** — the incumbent coupon-collects to L3 recall ≈ 0.34 by c201 anyway, but the table
freezes at commit (realized recall 0.054–0.179 there), and the r² law makes 2× L3 recall ≈ 4×
the L4 ceiling. It also reframes the donor: `crescendo`'s thermostat buys coverage *with time*
(17 extra cycles for 2.4× the observations); the port buys the same quantity with selection.

**Controls** (what the question knob does *not* move is the whole experiment): per-cycle d\*
difficulty quota from the menu's head, filled bin-for-bin by every pinned arm (held 201/201
everywhere, selected mean 2.752 vs menu 2.757); volume (`mine_cap` asserted per cycle); priced
budget (t_cum spread 0.02% across the six 201-cycle arms); selection compute (the reader and
value forwards over all 2048 run in **every** arm, charged in the ledger); oracle containment.
Delivered-vs-designed dose is a first-class instrument: non-null and era-bounded (0.37–0.54 in
era 1 → ~0 by era 4, where the cell swallows the span), matching Phase 0's calibrated δ ≈ 0.44.

| arm | selector | job |
|---|---|---|
| `q_exo` | the menu's head, donor's own RNG | exogenous ladder ≡ random; **bit-identical replay of `cr3_s0/anchor_long`** |
| `q_bisect` | exact clean-derivation key, prefer true-not-at-support, r²-aware | the oracle ceiling (grades/chooses; never consultable by other arms) |
| `q_endo` | half-key novelty vs own at-support set × posed-vs-landed delivery ledger | Δ`at_support`/priced sample — the endogenous judge |
| `q_novel` | same, guard removed | the nerdsnipe pole |
| `q_comp` | highest own v(x0) within stratum | comfort pole at matched difficulty |
| `q_bisect_loop` | oracle selector under `outer_yield_m4`'s pacer (180 cycles) | the composition cell |
| `q_comp_free` | highest own v(x0), difficulty free | the honest β=2 pole (its d\* deviation −0.364 is its definition, measured) |
| `an_s1/outer_yield_m4` | none (port off) | pacer-only; doubles as the fork certification |

**Provenance resolves into two grains on this substrate.** The answer-level efference copy
(the miner's own `keys_at_support`) was built and *rejected offline* — it anti-correlates with
trust (τ_b −0.29…−0.47) because it is downstream of use. The question-level copy is **π's
proposal slot**: exafference = an execution the DP served from an entry π did not propose.
`ap_s0` adds that per-execution join as a logging-only recorder (an element-wise refinement of
`assay`'s `entry.hist`, 0 mismatches); P″ then gates rehearsal credit on it, twice (strict:
every macro π-proposed; level-restricted: the rehearsed level's macros π-proposed, with an
eligible-pool count-matched control).

## Findings

1. **Question quality moves the mining, ordered exactly as designed.** End-of-run L4 true keys
   at support (of 816): `q_bisect` **50** · `q_novel` 39 · `q_endo` 35 · `q_comp` 27 ·
   `q_comp_free` 26 · `q_exo` **22** (matched-priced-spend brackets within 2 everywhere). The
   oracle aims 0.844/0.629 of its questions at true / needy-true keys vs 0.39–0.42/0.13–0.15
   for every non-oracle arm; L2 recall reaches 1.000 only in the two bisect arms.

2. **Mining converts to deep-era value through the pacer — the composition *is* the effect.**
   All three lanes measured against one certified shared trajectory
   (`an_s0/q_exo` ≡ `cr3_s0/anchor_long`, identical to t_cum = 50,688,282), with the
   pacer-alone lane in-tag after `an_s1`'s 0.000e+00 certification:

   | recovered fraction, Δ to baseline | era 4 | era 5 |
   |---|---|---|
   | pacer-alone (`an_s1`) | +0.091 | −0.060 |
   | selection-alone (`q_bisect`) | −0.085 | +0.322 |
   | composed (`q_bisect_loop`) | **+0.285** | **+0.536** |
   | **composed − sum of singles** | **+0.279** | **+0.273** |

   The interaction term is roughly the whole deep-era effect. Same pattern on trust: π's L4
   beam share +0.26–0.27 composed vs +0.10–0.15 and ≈0 for the singles; the composed arm
   commits L4 at c121 vs c180. Framing that must travel with the table: loop arms are not
   lifetime-matched (composed −11.0% priced spend — its positives are *conservative*;
   pacer-alone −19.6% — its negatives are suspect toward time cost), so the interaction's
   **sign** rests on matched conditions while its magnitude is not lifetime-clean. The loop
   arm is bit-identical to the schedule until c18 — the anchor's own first commit — so
   divergence begins exactly at the loop's first action.

3. **The endogenous judge is real but partial.** `q_endo` mines mid-pack (35), posts era-4
   +0.130 (1.5× the earning-family floor, 4.3× the in-node era-4 displacement floor) and the
   tag's best era-5 delta, +0.716 (1.75× the thin in-node era-5 floor — rank/sign territory).
   Its delivery ledger: 1,433 half-keys posed, 574 landed, mean weight 0.607.

4. **The comfort pole is the dangerous one on this grammar; the noise pole is mild.** Pinned
   `q_comp` is the tag's slowest climber (L3 recall 0.196 — lowest; L2 commit c60 vs c18);
   unpinned `q_comp_free` abandons the difficulty mix (−0.364) and posts the tag's worst
   deep-era values (−0.103/−0.311). `q_novel`, sitting in a natively junk-dominated
   observation stream (0.82 junk mass at L4), still mines 39 true keys and lands ≈ `q_exo` on
   value — the SPEC §4 asymmetry (on a constrained grammar nearly every answer is worth
   something; the corruption payoff is small, the ordering payoff decisive), measured.

5. **Provenance is computable from inside at the proposal grain, and it moves with trust —
   non-circularly.** `exaff_prop` (of π-proposed executions, the share served by a spelling
   never self-derived) orders credit > exposure > none (0.154/0.112/0.079) at 1.68× the
   unrehearsed-L2 handle — and the circularity concern is retired on the banked join: the
   ask-independent pooled read keeps the order (1.55×), every volume/ramp-matched read raises
   the multiple (1.87–3.56×), the rate-vs-weight decomposition puts the proposal-weight term
   at −1…−4% of each gap, and the schedule-forced window is *flat* across arms (spread 109×
   smaller than the π-asked spread) — the provenance differences live entirely in what π chose
   to ask. The gift arm runs at exafference 1.0000 for 44 contiguous cycles before its
   `mine_support` crossing — receipt-of-someone-else's-answers as a wire-level trace — and
   credited rehearsal expands the entries π actively proposes (12 → 17) while matched exposure
   moves it not at all.

6. **Per-step authorship gates dose-collapse, and the ledger says why.** The strict gate
   (every macro in a solving rehearsal trajectory π-proposed) passes 4.9% of solves, 38% of
   them macro-free — because macro work inside solving rehearsal trajectories decomposes as
   **31.5% π-proposed / 6.8% forced / 61.7% exploration-drawn**. The level-restricted round
   (`pp_s1`) delivered 0.085 vs the independence-model's 0.125 (L3 application sources are
   positively correlated within a trajectory) and **does not resolve**: the L3 trust contrast
   is 1.09× its handle with the sign against the treatment, era-4 value is below the stream
   floor, and the one separable cell (era 5, 1.9×) favors the control and is explained by a
   named defect — the control was count-matched only at the first consumed cycle and delivered
   48% more pairs (fix specified: the `yoked_*` idiom; recorded, not run).

## Interpretation (discussed with Jasper 2026-09-01 — argued, not measured)

- **(a) The port's two payoffs separate, and value is their product.** Selection buys table
  content everywhere it can reach (finding 1, every arm); the pacer buys the clock (the commit
  59 cycles earlier); and deep-era value appears where the two compose — the earlier-committed
  book is also the fuller one, and π forms trust on it while the eras that pay are still ahead.
  This sharpens `crescendo`'s reframe: coverage-with-time and coverage-with-selection are
  separable levers, and the crank wants both in one loop.
- **(b) The two poles are asymmetric on a constrained grammar, as the SPEC predicted.** The
  measured hazard of naive self-demand here is comfort (difficulty collapse), not noise; the
  nerdsnipe pole should be re-tested where answers can be worthless (language, the world)
  before the guard is judged unnecessary in general.
- **(c) The ask is the director of the search, not the author of the answer.** Three facts
  jointly force this: the credit > exposure effect is real and circularity-proofed (finding 5);
  the provenance differences localize entirely to what π chose to ask (the 109× cut); and the
  bulk of macro work inside solving trajectories is exploration-drawn (finding 6). Under that
  reading π's proposals steer which neighborhoods get searched, exploration does most of the
  work there, and success-filtered imitation credits what the search *found* — so a per-step
  authorship gate must dose-collapse, because it discards precisely the found-but-not-authored
  work the ask existed to license. The trajectory-grain provenance experiment — which episodes
  you chose to pose, filtered by success — has already been run and already worked: it is
  `woodshed`. P″ tested a sharper authorship theory than the mechanism uses. Status: supported
  and unfalsified — the "must" is an argument from the decomposition, and `pp_s1`'s null is
  volume-confounded — but it is the one reading consistent with all three facts, and it answers
  the strict-vs-restricted gate question by dissolving it: the right grain for provenance is
  the episode, not the step.
- **(d) The headline answer to OPEN_QUESTIONS #3**: yes — question-choice can occupy the
  demand seat the roadmap reserves for the teacher, and the value system can grade questions
  (thermostat-grade, partially). But it enters as a *composed* action: the loop that paces the
  crank should also choose the questions, because either lever alone leaves most of the
  deep-era value unclaimed.

## Caveats

- The lifetime mismatch on loop arms is structural (they leave eras early); the per-row
  conservative/suspect framing in finding 2 is the honest read, and no gating run fixes it.
- The era-5 floors are thin (in-node 0.410), so era-5-only cells — `q_endo`'s crown among them
  — are rank/sign claims.
- The question knob's grip is era-bounded: delivered dose ≈ 0 in eras 4–5, where the damage
  cell swallows the span. Question-construction machinery for deep eras does not exist yet;
  findings 1–2 are about what selection buys *before* the frontier reaches it.
- `pp_s1`'s control over-delivered pairs by 48% (defect named, `yoked_*` fix specified);
  no floor for π mass or for any provenance statistic exists anywhere in the arc;
  `mine_support` = 3 is a fixed parameter of the provenance match, not sweepable offline; the
  probe phase of `entry.hist` mixes in audition rows (donor-lineage fact — two probe columns
  demoted, reducers now guard).

## Runs on disk

| tag | what | GPU-h |
|---|---|---|
| `an_gf` (×2) | G-F fork-vs-donor replay + Q-11 on-substrate aiming check | 0.45 |
| `an_smoke` | 7 arms at `--quick`; dose instrument shakedown (uninformative at quick scale, recorded) | 0.75 |
| `an_s0` | the main run: 7 arms, 201 cycles (loop arm 180) | 4.12 |
| `an_s1` | pacer-only arm; the fork certification | 0.67 |
| `ap_s0` | P′: the proposal-grain recorder over `wd_s1`'s four arms | 1.61 |
| `pp_smoke3`/`pp_s0` | P″ strict gate (arm 2 cut at the lane's own dose ledger) | 0.82 |
| `pp_s1` | P″ level-restricted pair | 0.90 |

Volumes: `rhm-scaling-data:/data/rhm_practice_antiphon/{an_s0,an_s1}/` and
`/data/rhm_practice_antiphon_p/{ap_s0,pp_s0,pp_s1}/`. Fetched mirrors, reductions, and figures
under `figures/` and `provenance/figures/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# Phase 0 (offline, no GPU), then the main tag and the pacer-only certification arm
python3 rhm/practice/antiphon/phase0_question.py
python3 rhm/practice/antiphon/launch_detached.py --fn antiphon_run --tag an_s0 \
    --arms "q_exo,q_bisect,q_endo,q_novel,q_comp,q_bisect_loop,q_comp_free" --seed 0
python3 rhm/practice/antiphon/launch_detached.py --fn antiphon_run --tag an_s1 \
    --arms "outer_yield_m4" --ref-tag an_s0 --seed 0
python3 rhm/practice/antiphon/analyze_antiphon.py --tag an_s0 --merge-tag an_s1 --fetch --figures
python3 rhm/practice/antiphon/pacer_marginal.py

# Shape P: offline premise check, the P' recorder round, the circularity check, P'' rounds
python3 rhm/practice/antiphon/provenance/prov_tag.py --figures
python3 rhm/practice/antiphon/provenance/launch_detached.py --fn antiphon_p_run --tag ap_s0 \
    --arms "exact_reh,exact,exact_exp,given_c1" --ap-rec --seed 0
python3 rhm/practice/antiphon/provenance/analyze_ap.py --tag ap_s0 --fetch --figures
python3 rhm/practice/antiphon/provenance/analyze_pp.py --tag pp_s1 --fetch --figures
```

Full flag sets, gate tables, and per-run configurations: [`FILES.md`](FILES.md) and
[`provenance/FILES.md`](provenance/FILES.md).

## Next steps (queued, not started)

The learned question-judge inside the composed loop (the A2 move over this round's action set)
· question-construction machinery for the deep eras, where the delivered dose is currently ~0
· the yoked P″ pair (`pp_s2`) as the clean completion of that record — demoted by
interpretation (c), not a live hypothesis · an `/add-belief` pass for director-not-author,
read against `performance_error_is_the_bridge` §3's trajectory-grain agency law · a seed pair
on the interaction cell if it becomes load-bearing externally.
