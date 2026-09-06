# census + assay — what the complete vocabulary has that the earned one lacks: amount, truth, and arrival, crossed — and arrival wins

**Up**: [`../README.md`](../README.md) (practice arc) · **Specs**: [`SPEC.md`](SPEC.md) (census,
2026-08-23) and [`../assay/SPEC.md`](../assay/SPEC.md) (assay, 2026-08-23) — the records of what
was asked at each step · **Files**: [`FILES.md`](FILES.md) (census machinery) and
[`../assay/FILES.md`](../assay/FILES.md) (assay machinery, incl. the stream-twin mechanics)
**Direct parent**: [`../spiral/`](../spiral/README.md) — findings 5 (the eras-4–5 coverage gap)
and 6 (the certificate satisfied by concentration) are the two measured facts this unit connects.
**Runs**: `cs_s0` (census, 6 arms, 2.31 GPU-h) · the forensics pass (`forensics.py`, no GPU) ·
`as_s0` (assay, 6 arms, 2.29 GPU-h + 0.6 aborted) · `as_s1` (stream-displaced twins, 0.91 GPU-h),
2026-08-23→25. **Ranks, signs, and multiples of the *measured depth-6 stream floor* are the
claims** — this unit measured its own noise floor (finding 7), and
several of its early readings are demoted by it below. One orchestrated conversation; one
implementer agent built all four steps; `census/` and `assay/` are siblings because each is a
proper node with its own spec and tags — this README is the single writeup for the unit.

## The question

The spiral ended with a diagnosis and a gap: its certificate certifies level 3 on a
demand-concentrated sliver (<10% of the level), and the complete true vocabulary beats the earned
sliver by +0.3–0.4 recovered fraction exactly where demand outruns what was earned (eras 4–5).
The obvious reading — *the book is too thin; fill it in and the deep eras improve* — is a causal
claim, and this unit tested it in three steps: **census** (can an endogenous gauge buy coverage,
and does bought coverage pay?), **forensics** (when it didn't pay, why not — from the existing
logs, free), and **assay** (oracle surgery on the committed table, crossing **amount × truth ×
arrival** so the ingredient of `given`'s advantage is isolated rather than argued), plus a
stream-displaced-twin pair that measures the depth-6 noise floor every claim is read against.

## Design in brief

All on the spiral's depth-6 substrate (routing-only ports, G=482, ladder era-cycles 48/40/12/9/7
after a census re-balance), forked in a chain — `census.py` from `spiral.py`, `assay.py` from
`census.py` — with the arc's fidelity discipline at every link (fork ≡ donor bit-for-bit with
ops off; every treated arm ≡ its twin until treatment; cross-tag anchor replays exact over all
116 cycles). **Census** (`cs_s0`): `census_gate` (L2 commit held open until an admission-rate
gauge quiets, as a conjunction with the certificate), `yoked_delay` (same hold by clock — the
criterion/timing dissection), `census_extend` (commit at cert; recert extends the frozen table
through the same audition machinery), vs `spiral_route`/`given_route`/`given_native`. A Phase-0
offline replay over `sp_s0`'s logs sized everything first and found the L2 freeze caps buildable
L3 at frozen-recall² — which moved the gate from L3 to L2 before any GPU was spent. **Assay**
(`as_s0`): at each level's own commit cycle, commit a *constructed* table — `anchor` (own),
`strip` (junk removed), `complete` (own ∪ all missing true), `exact` (the full true table,
commit-time arrival), `junk_dose` (own ∪ realistic mined-but-false junk), `given_c1` (true tables
from cycle 1) — everything else identical; plus a per-execution **entry-identity instrument**
(which table entry the executor DP actually serves, bit-identical to the donor path). **Twins**
(`as_s1`): `given_c1` and `exact` re-run with only their RNG stream position displaced (a logged
1024-draw burn on both generators), on an in-job-asserted identical substrate — the direct
measurement of what stream position alone moves.

## Findings

1. **Coverage is buyable, and the r² ratchet works in both directions.** The census gate's
   17-cycle hold bought +0.072 L2 recall (20% of what the Phase-0 replay said was on offer — the
   gauge certified a genuine pause, not the ceiling); extension bought L2 0.571→0.786 and 2.5×
   the L3 coverage (recall 0.071→0.179, 12→31 entries), running at both levels in all five eras
   once the recert loop was widened. In the assay, completing L2 first made all 48 missing L3
   entries buildable (unbuildable = 0) — the freeze-caps-the-next-level mechanism, working in
   the arm's favour when reversed. (`cs_s0`, `as_s0`)
2. **Bought coverage did not close the deep-era gap, and the failure was not an artifact.** The
   census's 2.5×-coverage arm closed ≤0.08 of the eras-4–5 bracket and fell below the anchor in
   era 4; the forensics then refuted both mechanical suspects — extension events never trigger
   the forced-exploration window (0/12, commits 8/8 as positive control), and 9 of 12 admitting
   events were followed by error *decreases* (the one excursion was an era boundary). The
   deficit predated the nearest event by four cycles and originated in era 3: carried-state
   damage, not event damage. (`cs_s0`, forensics)
3. **π vets levels, not entries — pollution de-funds the level, true entries included.**
   Structural: the proposal head's action space is one action per (level, node) slot; which
   entry serves a call is the executor DP's choice, invisible to π. Measured: the arm holding
   the *most* L3 entries ended with the *lowest* L3 proposal mass of the earning arms (0.158 vs
   anchor 0.233), lowest argmax share (0.099 vs 0.234), and fewest macro expansions. The sign
   split in the same round: a precision-preserving addition (+3 L3 at unchanged 0.333) beat the
   anchor in 4 of 5 eras; the precision-degrading addition (+12 L3 at 0.250) lost eras 3–4.
   (forensics; `junk_dose`'s mass drain reproduced prospectively in `as_s0`)
4. **Gate-later is dead at this substrate, cheaply.** `census_gate` and `yoked_delay` are
   bit-identical over all 116 cycles — the gauge's criterion contributed nothing beyond a cycle
   number — and the 17-cycle hold cost no priced time, no width, and ~0.01 recovered fraction.
   With the spiral's transient commit penalty, commit timing under routing is cheap in *both*
   directions; the ratchet's timing law reads as an enumeration-era fact. (`cs_s0`)
5. **The executor DP is a junk filter, so truth matters much less than the census suggested.**
   Per-execution entry identity: a 6.2%-true L3 table yields **71–76% true executions** (every
   impure arm's true-selection share exceeds its table precision). Consistent with that,
   `junk_dose` sat at-or-above the anchor in eras 4–5 while its π L3 mass drained exactly as
   finding 3 predicts — junk arriving *at commit time* costs little; the census's damage came
   from junk dripping in mid-flight. (`as_s0`; dose caveat below)
6. **The headline: arrival dominates.** `exact` holds content identical to `given_c1` (the full
   true table, precision 1.000, verified entry-for-entry) and differs only in arriving at the
   commit cycles instead of cycle 1. The residual — what content equalisation does *not*
   recover — is 72–82% of the `anchor → given_c1` span in eras 4–5, and **certifies at era 4 at
   3.24× the measured stream floor** (era 3: 1.67×, marginal; era 5: 1.52×, not separable —
   see finding 7). The mechanism is trust with its own slow clock: `given_c1` ends with π's L3
   mass at 0.580 and argmax share 0.831, executing macros 2.4× more than any commit-time arm;
   *every* late-arrival arm — the perfect `exact` included — caps at ≤0.276 mass / ≤0.331
   argmax. The separation survives stream displacement on both sides (displaced c1-arrival
   0.448/0.625 vs commit-time max 0.287/0.344, the families moving toward each other and still
   not touching). Pure coverage buys a real but minority share: `complete` − `anchor` is the
   unit's largest certified surgery effect (**3.56× floor**, +0.23/+0.31 in eras 4–5);
   `exact` − `anchor` 2.38×. (`as_s0` + `as_s1`)
7. **The depth-6 stream floor, measured — and it retro-scopes three rounds.** Displacing only
   stream position moves eras-3–5 recovered fractions by up to **0.087** (`exact` family) and
   **0.083/0.148/0.344** (`given` family — the floor *grows with era depth*). Under
   displacement, the eras-4–5 money-table rank order is **invariant** while eras 1–3 orderings
   shuffle and `exact` − `anchor` flips sign there; bracket-fraction denominators are
   twin-dependent (e.g. `complete` closes 0.425 or 0.803 of era 5); and L3 certification cycles
   carry an **11–14 cycle** floor, so every arm-order reading of L3 cert cycles in `sp_s0`,
   `cs_s0`, and `as_s0` (differences of 3–5 cycles) was never readable. Ranks and signs in the
   deep eras are the currency; fractions and shallow-era orderings are not — `recital/`'s
   depth-4 lesson, now measured at depth 6 and sharper with depth. (`as_s1`)
8. **Sighted along the way, filed**: competence concentrates trajectories — `given_c1` executes
   macros 2.4× more and observes the *fewest* distinct L4-shaped tuples (279–318 vs 482–542
   across earning arms, ordering stable under displacement); the `merge/` index-withholding
   rhyme, first measurement of a new instrument, not interpreted. The census's admission
   audition was bit-flat on 8 of 12 admitting events (admission by tie-break at `tol=0`): an
   audition under present demand cannot price an entry whose value lives under future demand —
   `ear/`'s consumption-matching law, recursed into the op that was meant to fix its last
   appearance.

## Interpretation (discussed with Jasper 2026-08-24→25 — argued, not measured)

- **(a) The value of a vocabulary is mostly not in the table — it is in the policy that grew up
  with the table.** Amount, truth, and even perfect content transfer recover a certified
  minority of the deep-era gap; the majority rides on arrival — practice-time during which the
  planner's trust in the level forms. §3½'s decomposition gains a third member: content is
  derivable (the corridor), identity is storable (the table), but **use is only earnable**, on
  its own clock, and was not shortcut by gift anywhere in this unit.
- **(b) The gauge blindness recursed.** The spiral found the certificate satisfied by
  concentration; the census's extension audition was then satisfied by harmlessness-under-
  present-demand. Both are the same type error: every audition-shaped gauge prices what current
  demand exercises, and coverage-for-the-future is precisely what it cannot price.
- **(c) The benign-sliver reading matures.** The earned sliver plus the DP's execution-time
  filtering is close to what in-loop experience can actually vet and cash; what `given` has
  beyond it is mostly not a better book but a lifetime of leaning on one. Quarantine has level
  granularity (finding 3), so the one real vocabulary hazard the loop faces is *dilution of a
  level's reliability*, not individual bad entries.
- **(d) Held loosely, the natural next instrument**: the trust-formation rate itself — π's
  per-level mass growth after arrival — as a first-class readout, and whether targeted
  rehearsal of a received vocabulary can compress the clock that gift could not skip.

## Caveats

- **The stream twins bound stream-position noise.** Era-5 magnitudes are consistent with the
  headline but not separable from the measured floor.
- Demoted by the floor, on the record: `strip`'s entire eras-3–5 profile (1.19× floor — its
  apparent era-2 result is a fraction of a degenerate +0.041 span, i.e. ~0.03 absolute);
  the era-5 `complete` > `exact` ordering (flips under displacement); all L3 cert-cycle arm
  orderings; all precise bracket fractions (double-valued under twins).
- `junk_dose`'s L3 dose under-filled by design collision (K=20 vs `complete`'s 48 — its own
  polluted L2 capped buildability), so finding 5's "junk at commit is cheap" is measured at a
  weaker dose than intended. The `exact`/`given_c1` contrast bounds but does not eliminate the
  stream-family confound (no constructible pair differs in stream alone when control flow
  differs — the twins measure the floor instead).
- The census gate's constants certified a pause in the admission stream, not its ceiling
  (`sp_s0`'s stream later reached 19 where the gate stopped at 13–15): gate-shaped ops were
  priced here, not optimized.
- One apparatus lesson, generalised in `assay.py` and worth carrying: a selfcheck that builds
  models before setup **perturbs the substrate stream it certifies** unless RNG-sandboxed with
  the restore asserted — caught by a fourth-decimal stale-buffer mismatch, cost 0.6 GPU-h.

## Runs on disk

| tag / artifact | node | what |
|---|---|---|
| `cs_s0` | `census/` | 6 arms × 116 cycles: gate, yoked, extend, anchors and ceilings |
| `figures/phase0/`, `figures/forensics/` | `census/` | the offline gauge replay (pre-GPU sizing; moved the gate L3→L2) and the forensics pass (T/V/B checks over existing logs) |
| `as_s0` | `assay/` | the surgery six: anchor, strip, complete, exact, junk_dose, given_c1 + entry-identity instrument |
| `as_s1` | `assay/` | stream-displaced twins of `given_c1` and `exact`; merged reduction with substrate-identity gates |

Volumes: `/data/rhm_practice_census/<tag>/`, `/data/rhm_practice_assay/<tag>/`; fetched copies
and figures under each node's `figures/<tag>/` (PNGs regenerate via `--figures`).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# census
python3 rhm/practice/census/phase0_replay.py
python3 rhm/practice/census/launch_detached.py --fn census_run --tag cs_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "spiral_route,census_gate,yoked_delay,census_extend,given_route,given_native" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 \
    --gate-win 12 --gate-theta 0 --extend-cap 8 --extend-tol 0.0 \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
python3 rhm/practice/census/analyze_census.py --tag cs_s0 --fetch --figures
python3 rhm/practice/census/forensics.py

# assay
python3 rhm/practice/assay/launch_detached.py --fn assay_run --tag as_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "anchor,complete,exact,junk_dose,given_c1,strip" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
# stream twins
python3 rhm/practice/assay/launch_detached.py --fn assay_run --tag as_s1 --ref-tag as_s0 \
    --arms "given_c1_j,exact_j" --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7"   # + as_s0 flags
python3 rhm/practice/assay/analyze_assay.py --tag as_s0 --merge-tag as_s1 --fetch --figures
```

Gates, per-decision measured reasons, and volume layouts: [`FILES.md`](FILES.md) and
[`../assay/FILES.md`](../assay/FILES.md).

## Next steps (queued, not started)

The trust-formation-rate instrument (π per-level mass growth post-arrival as a first-class
readout; can targeted rehearsal compress the clock gift could not skip) · a seed pair scoped to
era-5 magnitudes, if and when they become load-bearing · the spiral's `spiral`-vs-`spiral_route`
span-port consumption cell (still open, three sightings) · `/update-beliefs` sweep for the
spiral + this unit (§3½'s third component; the recursed gauge law; the measured stream-floor
methodology note) · the census gate-shaped ops only if a future round needs them optimized
rather than priced.
