# recital — self-paced eras: can practice manufacture its own recitals?

**Up**: [../README.md](../README.md) (rhm/practice) · [../../README.md](../../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1's first component — allocation — wired into the arc for the first time; §16 points here)
**Parents**: [`../ear/`](../ear/README.md) (the grader grid, the commit policy, and the substrate,
imported rather than copied — the fidelity replay `rf_s0` reproduces `er_s0` bit-for-bit) ·
[`../ratchet/`](../ratchet/README.md) (vocabulary, pricing, detector)
**Successor**: [`../tall/`](../tall/README.md) — the depth-6 port attempt this round's open
question motivated (voided in the loop; two of its findings act on this node and are noted in
place).
**Status**: written up 2026-08-16 (results discussed with Jasper 2026-08-15). One fidelity
replay, one 10-arm main run, two independent-world replicates, one follow-up run
(`pace_vocab` + extended sweep), and an offline mechanism check. File index: [FILES.md](FILES.md).

## One-liner

The arc's rounds all handed the agent an era clock; this round removes it. Each arm holds one
total priced budget and decides itself when to advance the depth ladder, with the commit policy
fixed (`ear`'s certify-else-provisional) so **advancement policy is the only variable**. The
answer, stable across three independent worlds: **no internal signal we built prices what time
at the bottom of the ladder actually buys.** The unit-LP certificate advances earliest and
finishes bottom-half everywhere (it reads commit-readiness, which is not level-exhaustion); the
task-progress pacer starves at the bottom in 2 of 3 worlds and degenerates to never-advancing;
the vocabulary-growth pacer — built after the main run named the mechanism — behaves exactly as
predicted (uniform across depth) and still advances too early, because **vocabulary saturation
is not vocabulary sufficiency**. Meanwhile a fixed bottom-heavy schedule (50% of budget in era
1) is **rank 1 of 8 on both the deep and mean cells in all three worlds**, and in the two
independent worlds beats the DGP's own vocabulary outright. The transportable law is the *first*
boundary: corr(first boundary as %T, mean error) = −0.94 / −0.69 / −0.79.

## Findings

Three independent worlds (seed triples 0/1/2 — new grammar, new substrate, new draws per
replicate; absolute errors are **not** comparable across worlds, so every claim below is an
ordering, a sign, or a recovery fraction). The stability table is the round's spine.

| claim | s0 | s1 | s2 | verdict |
|---|---|---|---|---|
| corr(**1st** boundary, mean err) < 0 | −0.94 | −0.69 | −0.79 | **holds** |
| corr(**2nd** boundary, mean err) < 0 | −0.47 | **+0.41** | −0.86 | **flips in s1 — narrowed to the 1st boundary** |
| `sched_d` (.50/.75) rank 1 of 8, deep **and** mean | 1 / 1 | 1 / 1 | 1 / 1 | **holds, 3/3** |
| `sched_d` deep-cell recovery ≈ 1 | 0.99 | 1.21 | 1.27 | **holds, strengthens** |
| cert-pacers advance early, finish poor | yes | yes | yes | **holds** |
| `pace_task` era-3 transfer, rank 2 | rank 2 | rank 7 | rank 8 | **retracted** |
| recert swaps = 0 | 0/106 | 0/90 | 0/92 | **holds** |
| plant inert | ✓ | ✓ | ✓ | **holds** |

1. **Time at the bottom is the law; the deeper boundaries are not.** First-boundary lateness
   predicts the matched-budget grade in every world, both cells, parametric and rank
   (Spearman −0.93 / −0.46 / −0.86). The second boundary's correlation flips sign in s1, so the
   transportable statement is *stay at the bottom longer*, not *delay every boundary*.
2. **A well-placed fixed schedule closes the whole gap to the true vocabulary.** `sched_d`
   (advance at 50%/75% of budget) is rank 1 in every world; deep-cell recovery
   `(e_never−e_arm)/(e_never−e_given)` = 0.99 / 1.21 / 1.27 — in both independent worlds it
   **beats `given`**. In world 0 the beats-`given` reading fails in *both* measurements (rc_s0
   +0.003, rv_s0 −0.060 in `given`'s favour... see finding 7): the rank claim is the robust one.
   Mechanically, `given`'s full true table is even *worse than no vocabulary* on s1's shallowest
   era — coverage costs beam width at matched pricing, and shallow damage does not repay it —
   while the mined table's task-shaped concentration wins at depth (the ratchet's
   concentration-over-coverage, at the schedule level).
3. **The certificate is the wrong curriculum signal, everywhere.** `pace_cert` produces the
   earliest first boundary in every world (11.0 / 14.1 / 12.1 %T against medians of 22–24) and
   finishes rank 5–8 of 8. It is well-calibrated to "this unit is ready to commit" and
   mis-calibrated to "this level is exhausted". `pace_comp`'s deadline never bound (both its
   boundaries came from the certificate), so the composed policy degenerated to `pace_cert`.
4. **`pace_task` starves at the bottom — the seed-0 transfer result is retracted.** In s0 it
   advanced once at c33 and posted the second-best deep cell (0.926 recovery) without ever
   visiting era 3. In s1 and s2 the era-1 task error never dropped by `lp_min_drop` (max drops
   0.042 / 0.055), the descent precondition correctly refused for the entire budget, and it
   finished ranks 7–8. What survives is weaker and real: an arm that never left era 1 still
   recovers **0.70–0.75** of `given`'s deep-cell advantage — **vocabulary-carried transfer is
   robust; competitive transfer was a seed-0 accident.**
5. **The LP-starvation reading of [`../ear/`](../ear/README.md) is revised: it was
   clock-truncated.** Every arm that gave itself ≥35 cycles in era 2 fired a genuine,
   non-provisional level-3 certificate (spans 0.151–0.167 over the 0.10 threshold);
   `fixed_climb`, at exactly 30 cycles, refused — reproducing `ear` and locating the ratchet/ear
   era clock ~5 cycles short of the level-3 certificate. The sharper point survives: the three
   arms with earned L3 certificates still lost to `sched_d` committing provisionally.
   **Certification is not what sets the grade; vocabulary quality is.**
6. **`pace_vocab`: the mechanism-named signal, predicted exactly, and still early** (`rv_s0`).
   Offline prediction 3/3: later than the certificate in era 1 (c26 vs c21), earlier in era 2
   (certificate: never), and *exactly uniform across depth* (26 and 26 cycles). The series is
   textbook (0.500→0.000, 0.875→0.025, margins 5–8× the precondition). It still produced the
   run's earliest first boundary (19.5%T) and finished rank 4 of 5, deep recovery 0.750.
   **The table stops growing long before the level stops paying** — measured directly by the
   mechanism check (finding 8).
7. **Stream-position noise bounds what a single run can claim** (`rv_s0`'s drift guards). Two
   identical arm specs on the same world at different RNG stream positions moved in opposite
   directions (deep cell ±0.023–0.034), swinging the `sched_d`-vs-`given` gap by ~0.057 and a
   recovery fraction by ~0.15. **Rank orderings are the currency; fractions at n=1 carry
   ±0.15.** (This also bounds `ear`'s beats-`given` margins, noted in that node's caveats.)
8. **The extended sweep: the optimum is at or near .50/.75.** Within-run: `sched_d2` .50/.75 →
   deep 0.333; `sched_e` .60/.80 → 0.359; `sched_f` .70/.85 → 0.409; `sched_g` .80/.92 → 0.339.
   Later-than-half does not help (the non-monotone `sched_g` recovery is read as n=1 noise);
   bottom-heavy, not maximally bottom-heavy.
9. **The mechanism check: what does late buy? Not the L2 table, not any logged policy
   quantity; the deep grade tracks the L3 table.** Between vocabulary saturation (~c25) and
   `sched_d`'s boundary (~c64) — 60% of its era-1 stay: novel tuples flat by construction, the
   live table gains ≤1 entry (in s1 a *false* one), and the frozen-vs-live counterfactual prices
   the post-saturation accumulation at nothing (0 recert swaps anywhere). Era-1 policy
   quantities move inconsistently across seeds (in s0 the metered era-1 error *worsens* over the
   window). What is seed-stable: corr(L3 entries, deep err) < 0 in 3/3 (−0.33 / −0.34 / −0.77),
   30/40 within-seed arm pairs order as the L3-table story predicts, and `sched_d` ≥ `sched_a`
   on L3 entries and better on the deep cell in 3/3. The chain from era-1 time to the better L3
   table has a hole the logs cannot fill (s2: identical L2 tables, L3 still 17 vs 15; support
   concentration and era-3-harm both tested and rejected on sign flips).
   *Acted on by [`../tall/`](../tall/README.md)*: the "same table or different?" half of this
   question is **retired a priori** — the miner is structurally monotone (entries are never
   removed), so late-vs-early tables always differ by additions only, and the recert
   counterfactual already prices the additions at zero. What remains open is which unlogged
   quantity (plausibly value/policy quality expressed only on the *next* level's mining yield)
   carries the benefit — the fixed-reference probe built in `tall` is the right instrument.
10. **Instruments.** Recert: 0 swaps in 288 events across all worlds. Plant inert everywhere
    (the descent is vocabulary-carried, as in the ratchet). Mined beats matched-size random
    subsets of the *true* table at L2 in every arm and world. Measurement-validity note: the
    earned-vs-given denominator is healthy **only on the deep cell** (the L1 denominator
    collapses to −0.044…+0.008 in s1/s2), so the seed-0 mean-cell recovery of 1.037 is
    withdrawn — deep-cell recoveries are the reported quantity.

## Results — `rc_s0` (terminal all-eras exam at matched T = 7.0e6, seed triple 0)

| arm | 1st boundary (%T) | L1n6 | L2n3 | **L3n1** | mean |
|---|---|---|---|---|---|
| `never_base` | 20.1 | .3255 | .5130 | .6589 | .4991 |
| `given` | 29.2 | .2240 | .2318 | **.3073** | .2543 |
| **`sched_d`** (.50/.75) | **50.5** | .2214 | .2057 | .3099 | **.2457** |
| `pace_task` | 25.2 | .2448 | .2995 | .3333 | .2925 |
| `sched_b` (.25/.55) | 25.3 | .2552 | .2708 | .3411 | .2891 |
| `sched_c` (.33/.67) | 33.7 | .2656 | .2422 | .3568 | .2882 |
| `pace_cert` | **11.0** | .2656 | .2812 | .3594 | .3021 |
| `fixed_climb` | 22.7 | .2474 | .3073 | .3620 | .3056 |
| `pace_comp` | **10.3** | .3047 | .2917 | .3698 | .3220 |
| `sched_a` (.15/.40) | 15.4 | .2500 | .3073 | .3854 | .3142 |

Grader cost 12.6–27.2% of budget for the practice arms (`pace_task` highest, 68 cycles of L3
auditions); `never_base`/`given` pay 0. Fidelity: `rf_s0` (`--t-budget 0`) reproduces
`ear/er_s0` **bit-for-bit on all eight arms across all 90 cycles**, extended past `e`/`t_cum`
to `n_moves`, `level`, `era`, `e_task`, the certificate-state arrays, every commit event and the
terminal ladder.

Advancement-vs-prediction: the within-run offline replay (`analyze_recital.predict`) reproduces
every self-paced arm's actual firing point exactly (`pace_cert` 15/39, `pace_comp` 14/40,
`pace_task` 33). The *pre-run* replay from `er_s0`'s logs got era-1 firing direction right and
the cycle wrong (predicted c21–c26, fired c15) — arm order changes the RNG stream, so replay
timing transports only within-run.

## Interpretation (discussed with Jasper 2026-08-15/16 — argued, not measured)

- **(a) Three internal signals, three distinct measured failures.** Commit-readiness fires too
  early (blind to what the level still yields); task progress starves at the bottom (no descent
  to detect in 2 of 3 worlds); vocabulary growth saturates before the level's value is
  extracted. Each honestly reads its own quantity; none reads *what time at the bottom buys* —
  which finding 9 locates downstream, in the next level's mining yield, a place no within-level
  signal looks.
- **(b) The teacher's second function.** `ear` measured teaching as lending a calibrated grader;
  this round adds **holding the student at the bottom**. Every self-pacer advanced too early in
  every world; the externally imposed bottom-heavy schedule was rank 1 everywhere. That the
  value of "more scales" is invisible to every first-person signal we built is a candidate
  first-principles account of why curriculum patience is externally enforced across practice
  traditions — recorded as suggestive, one substrate.
- **(c) "Can practice manufacture its own recitals?" — split verdict.** The recital *deadline*
  half barely mattered here (`pace_comp`'s deadline never bound; commits at boundaries were
  routinely fine). The *advancement* half is where external structure earned its keep. Whether a
  fourth signal (something value-model-shaped, per finding 9's residue) could close the gap is
  open, not foreclosed.
- **(d) Concentration-over-coverage climbs a level.** `given` losing to an earned, bottom-heavy
  schedule in 2 of 3 worlds — and losing to *no vocabulary* on one world's shallow era — extends
  the ratchet's interpretation (d): the consumption prior beats the exhaustive grammar not just
  at fixed table size but at the level of what a schedule should optimize for.

## Runs on disk

| tag | what |
|---|---|
| `smoke0`, `smoke1` | attached `--quick` smokes (first: 7 arms, all policies; second: `pace_vocab` + extended cells) |
| `rf_s0` | fidelity replay of `ear`'s 8 arms — must equal `er_s0`, and does |
| `rc_s0` | the main run: 10 arms at matched T = 7.0e6, seed triple 0 |
| `rc_s1`, `rc_s2` | independent-world replicates (seed triples 1, 2) |
| `rv_s0` | `pace_vocab` + extended sweep + in-run drift guards, seed triple 0 |

Volume: `/data/rhm_practice_recital/<tag>/<arm>/results.json` + `setup.json`; fetched copies in
`figures/<tag>/`; full reduction reports in `results/*_report.txt`.

## Reproduce

Full CLIs (fidelity replay, main run, replicate seed-triple convention) are in
[FILES.md](FILES.md) §Reproduce and §Seeds. The follow-up run:

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 rhm/practice/recital/launch_detached.py --fn recital --tag rv_s0 --seed 0 \
    --arms "given,sched_frac:sf1=0.50:sf2=0.75:nm=sched_d2,pace_vocab,sched_frac:sf1=0.60:sf2=0.80:nm=sched_e,sched_frac:sf1=0.70:sf2=0.85:nm=sched_f,sched_frac:sf1=0.80:sf2=0.92:nm=sched_g" \
    --t-budget 7000000 --max-cycles 300 --probe-t-n 12 \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --pr-width 16 --g-budget 58 \
    --max-macro-level 3 --n-pr 64 --n-rt 384 --n-score 512 --n-aud 192 --n-grad 4 \
    --value-lr-online 3e-5 --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 --mfg-source child --mfg-render canon \
    --mfg-min-bank 32 --recert-every 5 --recert-margin 0.05 --prov-offset 0 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 \
    --lp-min-drop 0.10 --late-offset 3 --probe-every 4
python3 rhm/practice/recital/analyze_recital.py --tag rv_s0 --fetch --figures
```

## Caveats

- **Absolute error levels are not comparable across worlds** (substrate quality spans controller
  accuracy 0.72–0.89 over the three rule draws); everything reported is an ordering, sign, or
  recovery fraction, and finding 7 bounds fractions at ±0.15 for a single stream position.
- `rv_s0` (the vocab pacer and the sweep turnover) is one world; its two headline statements
  are an exact 3/3 offline prediction and a within-run ordering, the forms that survived
  seeding elsewhere, but they have not themselves been replicated.
- `pace_comp` never exercised its deadline, so the composed *advancement* policy is untested in
  the regime where the deadline would bind (it would need a certificate that never fires —
  which s1/s2's `pace_task` shows exists, but that arm read the task series).
- The mechanism residue (finding 9) is unresolved by design of the logging, not by data volume:
  the missing quantities (per-cycle entry identity, fixed-reference policy quality) were built
  as instruments in [`../tall/`](../tall/README.md) and await a substrate where the loop learns.

## Next steps

Consumed by [`../tall/`](../tall/README.md): every level-3 statement in this arc is confounded
by level 3 being both the deepest earnable level and one below the grammar's ceiling — the
depth-6 port was scoped to pull those apart (and was voided in the loop; see that node for what
survives). The remaining open threads from this node: a fourth advancement signal shaped like
value-model progress on the *next* level's mining yield, and the composed
certify-else-deadline advancement policy in a regime where the deadline binds.
