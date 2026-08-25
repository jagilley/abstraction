# spiral — the live re-earning spiral: the crank holds, narrows, and the certificate can't see the narrowing

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (the record
of what was asked, 2026-08-22) · **Files**: [`FILES.md`](FILES.md) (machinery, gates, the Phase-B
decision table with the measured reason for each choice)
**Direct parents**: [`../native/`](../native/README.md) (the two ports; its next-steps queue named
this round first) · [`../tall/`](../tall/README.md) (the depth-6 substrate, closed 2026-08-16 as a
diagnosed apparatus negative; this round satisfies its recorded re-run recipe).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§17 (the depth-6 confound), §18, §20 ·
[`cerebellar_abstraction_ratchet`](../../../../ideas/cerebellar_abstraction_ratchet.md) §2–§4.
**Runs**: `pa0` (Phase A: gates + descent feasibility, 0.40 GPU-h), `sp_s0` (Phase B: 7 arms ×
116 cycles, 4.31 GPU-h), plus `fid_d4`/`fid_d4b` donor-fidelity gates and smokes, 2026-08-22→23.
**Single seed on every treatment; ranks, signs, and multiples of measured floors are the claims.**
One orchestrated conversation; both phases built end-to-end by one delegated implementer agent
(machinery details in `FILES.md`).

## The question

`native/` showed a fixed, already-earned vocabulary consolidates into the learner's own planner
(routing) and executor (corridor), with the table surviving as the address book. Every claim there
was one turn of the crank, vocabulary held fixed, at depth 4 — where level 3 is simultaneously the
deepest earnable level and one below the grammar's ceiling. This round runs the loop the arc had
never run: **earn level k live, consolidate it, earn level k+1 natively over the routed policy** —
in the depth-6 world, where the damage ladder (5 levels) outruns the earnable range (2–3), so the
frontier and the ceiling are different places and eras 3–5 are pure consumption. The stated
readouts: does the crank **accelerate, hold, or decay** per turn, and **which wall bites first**
(the diet — the observation stream thinning with level; the grader — the arc's most-repeated
blocker; or the corridor — span parity at depth)?

## Design in brief

Depth 6, m=2 (m≥3 measured inadmissible in `tall/`), v=8, s=2, 64 tokens, 32 base moves,
`max_macro_level=3`, nested damage ladder `1:25, 2:12, 3:6, 4:3, 5:1` at era lengths
32/56/12/9/7 (sized by offline detector replay on Phase A's own series — depth-6 certification
runs 1.6–1.75× later than `cal0`'s clock-paced prediction). `spiral.py` forks
`native/full/full.py` verbatim (both ports + live miners), transplants `tall/`'s depth-6 recipe
(`n_corrupt=1` collection, preflight, probe re-pricing) and `ear/`→`recital/`'s commit policy
(certify-else-provisional-at-boundary, live recert every 5 cycles), and holds `g_budget=482` — the
budget at which enumeration *cannot* keep width 2 past the L2 commit, because that inversion is
the thing routing is claimed to dissolve. π trains online by self-imitation from cycle 1; at a
commit the entries enter the table, π's action space grows (with a 2-cycle forced-exploration
window for the new slots), and span slots open behind the per-macro parity gate (τ=0.95,
`span_min_hold=128`). Arms: `given` (true tables, enum), `given_native` (true tables, both ports),
`spiral` (live earn + both ports), `spiral_route` (live earn + routing only), `enum_live` (live
earn, no ports), `enum_live_g722` (enum at the budget that protects the L2 commit, width-capped to
the matched arm pre-commit), `fid` (both ports wired and shut). **Gates all bit-for-bit**: the
fork replays the donor at depth 4 exactly (max|Δ| = 0.0, self-replay control 0.0, re-verified
after every addition); `fid` ≡ `given` over all 116 cycles; every pre-treatment twin window exact.

## Findings

1. **Routing rehabilitates the depth-6 substrate — the descent gate `tall/` failed now passes.**
   (Phase A, `pa0`.) The enum arm reproduces `dens0`'s broken reference in-run (slope
   −0.0054/cycle vs −0.004; recovered fraction 0.24–0.31); the routed arms reach **0.50–0.59
   recovered fraction — inside the depth-4 target band** (0.50–0.80). The decomposition is
   honest: the routed gain arrives as a *step* when the proposal filter switches on (e
   0.750→0.477 at c5), i.e. width bought at fixed price (18 vs 2 at G=482), not a steeper
   per-cycle learning slope.
2. **The pricing inversion that voided `tall/` dissolves live, and its two halves have different
   time-signatures.** Enum's L2 commit penalty reproduces `dens0`'s and is **permanent** (+0.172;
   width 1 to end of run). The routed commit jump is larger (+0.336) but strictly **transient** —
   exactly the 2-cycle forced-exploration window (k_eff 4→20, width 18→3), then back to width 18
   and better-than-pre-commit error. The insurance control priced the alternative: `enum_live_g722`
   protects the L2 commit and buys **nothing measurable after era 1** — identical certs, tables,
   and e from era 2 on — because the L3 commit (n=56, needs G≥842) collapses it anyway.
3. **The spiral turns, twice, on its own certificates.** Every earning arm certified both levels
   (L2 at c18–19, L3 at c48–52); the boundary-provisional fallback never fired; L2 corridors
   opened 13–16 of 16 slots at parity 0.98–1.00 and served natively.
4. **The rate question has three answers, one per clock — the disagreement is the finding.**
   *Certificate clock*: the crank **holds, mildly accelerating** — net cycles-to-cert (first
   candidate → fire) 18→16 (`enum_live`), 17→15 (`spiral_route`), 17→19 (`spiral`), for a level
   4× larger. *Coverage clock*: it **decays hard** — committed recall falls 0.57–0.79 at L2 to
   **0.054–0.089 at L3** (9–11 entries of 56) in every earning arm. *Value clock*: it **holds
   inside the earnable range** (finding 5). The L2 rate deltas between arms are 1–2 cycles —
   within plausible stream noise; the cross-level and cross-clock structure is the claim.
5. **Earned ≥ given inside the earnable range, natively, at roughly half enumeration's priced
   time — and coverage's value is measured just beyond it.** Recovered fraction of each era's own
   stale→floor range, eras 1–3: `spiral_route` **+0.505/+0.661/+0.690** vs `given_native`
   +0.287/+0.574/+0.647 and enum-with-true-tables (`given`) +0.036/+0.237/+0.297 — at era-1
   priced time 8.2M vs enum's 15.1M. In eras 4–5, where demand outruns what was earned,
   `given_native` (complete true L2/L3) pulls away: +1.048/+0.858 vs +0.676/+0.563. The gap
   between a complete vocabulary and an earned sliver, at matched ports, is **the measured value
   of coverage beyond the demand distribution the sliver was earned on.**
6. **The wall that bites first is the grader, in a sharper form than the arc has stated it: the
   certificate is *satisfiable by concentration*.** It certified L3 tables covering <10% of the
   level in every earning arm, and the freeze then holds while mining runs ahead — live L3 recall
   0.14 vs frozen 0.054–0.089 everywhere; in the enum arms a live-table swap would change 75–77
   audition answers (recert `n_diff`). Not mis-measurement: the priced audition, the unpriced
   oracle audition, and the true-macro ceiling agree to 0.02–0.04 at both levels. The gauge reads
   *quality of what is held*, and holding little that is good satisfies it.
7. **The two walls predicted to bite did not.** *Diet*: the L3 observation stream **grows** —
   distinct tuples at support rise L2→L3 and era 2→3 in every arm (spiral 15/36→16/37), and the
   routed arms make the better mining substrate (L3 committed precision 0.44–0.46 vs enum's 0.27
   at matched size — `native/` finding 7 reproduced live at depth 6). *Corridor*: 5 of 8 L3 span
   slots cleared τ at parity 0.99–1.00 (`spiral`; `given_native` 3–7), where depth-4 L3 corridors
   mostly never cleared — an L3 span is 8 leaves at both depths, so depth dissociated span length
   from coverage fraction and the corridor tracked span length.
8. **The address-book claim survives on a live-earned vocabulary.** End-of-era battery: deleting
   the committed table costs `spiral` ~nothing at the current level (a_full 0.654 vs b_table
   0.661, era 3) while deleting the corridor too collapses it (b_span 0.867); same pattern in
   `given_native`. π sharpens monotonically in every routed arm (top-1 0.25→0.31–0.34, entropy
   falling) — recorded as a proxy; `native/`'s decomposition-mass instrument was not re-run.
9. **One cell is now a pattern: the both-ports arm underperforms routing-alone in consumption.**
   `spiral` vs `spiral_route`, era 3: e 0.743 vs 0.611 (era 5: 0.916 vs 0.892) — `native/`'s
   unresolved `practice_late_native` +0.058 cell, reproduced at depth 6 and larger, despite
   parity ≥0.99 on every serving slot and a span-bypass battery that reads the head's firing as
   locally neutral. Unresolved, deliberately not sanded down.

## Interpretation (discussed with Jasper 2026-08-23 — argued, not measured)

- **(a) The crank does not stall — it narrows, and the narrowing is arithmetic before it is
  anything else.** The space of possible units roughly squares per level (16 → 56 usable → 1024)
  while the experience stream is fixed (`mine_cap=8` sightings/cycle drawn from the agent's own
  chosen trajectories). Each turn therefore earns a shrinking *fraction* of a widening floor, at
  roughly constant cycles per turn. Units broaden (each L3 chunk covers more of the world);
  coverage narrows (the agent holds less of the space of such chunks) — the same hierarchy fact
  seen from the token side and the agent side.
- **(b) The grader wall matures from "noisy at depth" to "satisfied by concentration."** The
  audition-shaped gauge cannot distinguish a well-selected sliver from an earned level, so the
  loop sincerely concludes level 3 is finished. This is the in-vivo form of the teacher-thread
  claim (`teacher_slot/`): the scarce thing is *which gauge is consulted*, and the loop's own
  gauge votes wrongly-by-type at the frontier.
- **(c) Concentration dominates coverage exactly as far as demand reaches, and no further.**
  Within the demand distribution the sliver was earned on, earned ≥ given (finding 5, eras 1–3 —
  `merge/`/`setlist/`'s lesson measured on the earning side for the first time); beyond it, the
  eras-4–5 gap is what coverage buys. Whether the sliver is *benign* (all value needs) or a
  *foreclosure* (a thin frozen L3 bounds any L4 forever, by `T4 ⊆ T3×T3`) is not measurable at
  this depth — L4 is unearnable at any affordable budget — and is the named question of the next
  round, not this one.
- **(d) `tall/` is retro-diagnosed: its third failure was not a substrate defect but the absence
  of consolidation.** The width rent it measured is what enumeration pays for a growing action
  set; a planner that proposes does not pay it. The re-run recipe's other two levers
  (`n_corrupt=1`, task-matched collection) contributed 1.2–1.9× each; routing contributed the
  regime change.

## Caveats

- **Single seed, one rule draw, one damage ladder.** The consumption-era rank gaps (finding 5)
  are large multiples of the depth-4 stream-noise floor (±0.034) and carry the claims; the L2
  cert-cycle deltas (1–2 cycles) and the `spiral`-vs-`spiral_route` era-5 gap do not.
- `given_native`'s era-4 recovered fraction is 1.048: the "floor" is an exact-DP rollout over
  **base moves only**, which a true-vocabulary native arm can legitimately beat — era 4–5
  normalisations are not a [0,1] scale for native arms; ranks there were read on raw e.
- `enum_live_g722` protects the L2 commit only (L3 needs G≥842); both enum arms run width 1 in
  the consumption eras, so the g722 contrast is an era-1–2 (earning) instrument by design.
- One recert reading (`spiral` L2 at c52, 0.89 vs neighbours 0.71) is a measurement-ordering
  artifact: it ran inside the post-L3-commit forced-exploration window. The enum arm observed 17
  distinct L2-shaped tuples against a true table of 16 — at least one off-grammar tuple survives
  at support (committed precision 0.733 is consistent).
- The span heads' L3 parity was reached under `span_min_hold=128` with τ re-checked every cycle;
  the corridor claims are conditional on that gate, and finding 9 stands unexplained beside them.

## Runs on disk

| tag | what |
|---|---|
| `pa0` | Phase A: depth-6 gates, descent gate with routing live, commit pricing, in-tag twin, task-matched collection measurement |
| `sp_s0` | Phase B: the main run — 7 arms × 116 cycles, five-era ladder, certify-else-provisional fixed |
| `fid_d4`, `fid_d4b` | G-F donor fidelity at depth 4, before and after the Phase-B additions (both 0.000e+00 against 0.000e+00 controls) |
| `smokeB` | `spiral_run --quick`, all seven arms end-to-end |

Volume `rhm-scaling-data:/data/rhm_practice_spiral/<tag>/`; fetched copies and figures under
`figures/<tag>/` (`fig1_competence`, `fig2_rate`, `fig3_walls`; PNGs are gitignored — regenerate
with `--figures`).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/spiral/spiral.py::gates_remote
modal run rhm/practice/spiral/spiral.py::fidelity_d4

python3 rhm/practice/spiral/launch_detached.py --fn spiral_run --tag sp_s0 \
    --eras "1:25:32,2:12:56,3:6:12,4:3:9,5:1:7" \
    --arms "given,fid,enum_live,spiral_route,spiral,enum_live_g722,given_native" \
    --budget 8 --g-budget 482 --max-macro-level 3 --n-corrupt 1 --mine-cap 8 \
    --span-min-hold 128 --span-tau 0.95 --collect-task-matched --tm-episodes 8192 \
    --prov-offset 0 --recert-every 5 --n-aud 192 \
    --probe-every 8 --probe-widths "1,2,4" --n-pr 64 --n-rt 384 --n-score 256 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --prop-warmup 4 --prop-new-cycles 2 --seed 0 --rule-seed 0 --train-seed 1

python3 rhm/practice/spiral/analyze_spiral.py --tag sp_s0 --fetch --figures
```

Full command set (Phase A, gates, preflight) and the per-decision measured reasons:
[`FILES.md`](FILES.md).

## Next steps (queued, not started)

The **coverage-gated frontier** (`census/`, spec in preparation): the certificate is satisfied by
concentration (finding 6) and coverage's value beyond demand is measured (finding 5, eras 4–5) —
can an endogenous coverage- or yield-shaped gauge (`teacher_slot/endo_yield/`'s currency) buy that
coverage, at what price, with the commit-timing confound controlled · the `spiral`-vs-
`spiral_route` consumption cell (finding 9, now a two-node pattern) · a seed pair if any
headline effect thins under scrutiny · `/update-beliefs` revisions (this round's, plus
`native/`'s still-queued §3½/§18 items) · `enum_live_g842` if the consumption eras ever need
de-confounding at enum's expense.
