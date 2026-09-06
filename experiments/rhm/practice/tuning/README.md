# tuning — one model can type the news; its own learning then eats the evidence

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (what was
asked, 2026-08-29, plus the Gate-1 reshaping addendum of 2026-08-30 — the reshaping itself came
out of Gate 0's data and a parallel-session note on Garcia-Garcia et al., bioRxiv
2026.03.03.709240) · **Files**: [`FILES.md`](FILES.md) · **Substrate donors**:
[`../fourwall/lm/`](../fourwall/lm/README.md) (`wall.py`/`wall_lm.py` imported verbatim,
untouched; its world, floors and bit-identity design), [`../teacher_slot/`](../teacher_slot/README.md)
(shadow-panel and priced-read idioms), [`../transpose/`](../transpose/FILES.md)
(`resample_cells`, imported verbatim), [`../../conditional_revision/`](../../conditional_revision/README.md)
(the `1L/8H/16d` FM idiom). **Idea doc produced**:
[`absorption_blinds_the_evaluator`](../../../../ideas/absorption_blinds_the_evaluator.md).
**Runs**: `tn0`/`tn_calib` (Gate 0, 2026-08-29→30), `g1a`+calib (Gate 1 wave 1), `g2a`/`g2c`
(wave 2 + fix round), `g1g` (offline checkpoint analytics), 2026-08-30→31; ~29 GPU-h total.
**Ranks, signs, trajectories and floor-multiples are the claims.**
One orchestrated conversation; every gate built end-to-end by one delegated implementer agent.

## The question

Three kinds of OOD news raise indexed-span surprisal identically and call for three different
responses: a **rotation** of the spurious key's meaning (correct op: merge, or track at the price
of staying hollow), an irreducible **noise burst** on the span (skip — the aleatoric null as an
action), and a **grammar drift** one level up (track). A single-reader surprisal gauge cannot
separate them by construction. The spec crossed two candidate instruments: **Factor T**, a
two-basis typing gauge read at the event instant, and **Factor M**, a forward self-model of the
reader whose benchmark-subtracted residual grades the op over the following window. Bursts are
span-confined and magnitude-matched to a rotation's spike, so nothing separates the events for
free — that discipline (per-checkpoint counterfactual rotation + six-rate burst ladder on the
same weights, matched surprisal by interpolation) is the node's measurement backbone.

## Design in brief

`fourwall/lm`'s world exactly (v16/s2/L6/m4, rule_seed 0, key = level-2 node 0, 8L/8H/256D,
20k × 64). Readers co-reside per worker with **per-reader global-RNG isolation**, licensed by the
gates: the uncharged panel moves nothing, the attached FM moves nothing, and every arm reproduces
fwlm0/fwlm1/tsdB/ey0/tn0 twins at **max|Δ| = 0.0 on the five instruments** over every shared
prefix, all tags. Burst calibration is two-sided (a noise-aware exact-BP oracle gives the
irreducible floor model-free; a 17k-step model-side run matches ρ per onset — the rotation spike
grows with scaffold age, 0.88–0.97 realized match). Drift = `transpose` level-3 resampling at
`n_cells = 20` (maturity-invariant match), with a null rung proving reads clean of draw variance.
Ops per 125-step window: `continue`, `merge` (input-stream collapse, latched), `skip` (no
step), `rekey` (priced 16-offset search on the observed token, wave 2). Op maps run on an
**online** read — the three views on the last 8 offered batches against the arm's own EWMA, dead
zones at in-tag mature-p99 floors — not on the counterfactual panel, which no learner has.
Gate 0 logged everything uncharged; Gate 1 ran 14 policies in two waves plus a fix round; the
Gate 1G analytics ran on ~180 banked fp16 checkpoints with per-event micro-grids.

## Findings

**1. The KL magnitudes were the wrong contraction; the movement triple types all three events
from one model.** Gate 0 as specced read `D_pair`/`D_self` and got a thin 1.1–1.3× separation
(`D_pair`) and an anti-separation (`D_self` responds ~2.4× *more* to a matched burst, 97% of
checkpoints; its counterfactual parse basis is measurably hollow — the mirror control sits at
~60% of `D_pair`'s clean level, beside d4-rand 0.27). Re-contracted as **movement under the
event per view** — T\* = (Δkey, Δself, Δout) — the same logged panel separates perfectly:
Δself exactly 0 under rotation at every checkpoint (by construction: a key-free view is blind to
a key permutation) vs +0.596 median / +0.269 min under a matched burst; with drift in-world
(`g1a`, never-merged arm, n=61) the three events occupy three corners, **100% sign-consistent on
every pairwise comparison** — rotation (0, +0.012), burst (+0.246, −0.000), drift (+0.175,
+0.128) on (Δself, Δout). The addendum's sharpest question answered yes: **the hollow key-free
basis does register drift** (min +0.058, never zero). And the two-reader version buys nothing:
`pair` ≡ `self` at max|Δ| = 0.0 across all 191 checkpoints — the second basis does not need a
second system on this world.

**2. The absorption wall: online, the learner erases the evidence at its own learning rate.**
The reader absorbs 86% of a +0.65-nat rotation shock in 50 steps, accelerating across events
(fwlm's 250→100→50). In-tag online floors run 1.1–3.6× the static calibration's (an acting arm
in a moving world has no quiet checkpoints), and event signatures at the op map's +50…+125-step
lag sit at or under them for everything the learner can learn: drift — the *most* learnable
event — is nearly invisible even at zero lag (2 of 3 under floor), while the same drift reads
+0.559 at 7–20× floors for seven straight checkpoints on an arm whose charged budget had expired
134 steps before onset (the accidental frozen-reader control: instrument fine, evidence
absorbed). Online rotation detection degrades across the run for the same reason (dk +0.090 at
rotations 1–2 → −0.004 by rotation 6). Only the burst — a sustained irreducible *condition*,
not a transient — stays reliably visible (typed 3/3 online in every typing arm, both waves).

**3. Supplied index ops are outrun by the native track, not refuted by it.** The priced re-key
search, correctly typed and fired at two rotations, **chose shift 0 both times**: at +50 steps,
restoring the old addresses is worse than the re-map already underway. The s14000 identity
return pays full price (+0.683, `stale_idx` flip) — fwlm0's no-mothball finding reproduced on a
new arm. Gate 1G supplies the mechanism: **there is no map-keyed rotation organ** (rotations
8000 steps apart applying the *identical* permutation share cos −0.06 at the embedding; alignment
decays with separation regardless of map). Tracking accelerates because rotations reduce to
**purer re-indexing over a stabilizing embedding table** — re-index recovery 0.599 → 0.874
monotone; the address-matched sharp test (pre-rotation h6 under the old map vs post-absorption h6
under the new map, same latent) recovers RSA to 0.60–0.88 where the naive read gives 0.11–0.43
and equals the map-swap control. The native mechanism converges toward *being* the re-key op,
inside one window.

**4. The learner carves a reflex for what it cannot learn.** Burst×burst weight alignment is a
fixed ln_f direction at cos **0.957**, non-decaying across the run (empirical floor 0.305) — a
"back off the span" gain reflex, carved once and reused verbatim: a native soft-skip, for
exactly the event class whose correct op is skip. Representationally the burst is a **gain
reorientation with no structural loss** (h3 RSA 0.911; h6 moves 3.36× null with separability
*rising* 1.33× — under span corruption the model amplifies its key code: noise entrenches the
crutch, fwlm's "available is taken" with a mechanism). Drift is the one true **rebuild**,
localized to the composition path (h1/h2 weight peak, PC angles 29–37°, no vocabulary or gain
news). The three event types occupy **three disjoint weight subspaces** (rotation: wte 4.1×;
burst: ln_f 4.3×; drift: h1–h2 1.8×; cross-type cosines at floor) — the typing information
absorption drains from the stream is written, cleanly localized, into the ledger absorption
writes. (Predicted instrument, unbuilt: per-layer-group update-norm ratios as a free online
weight-ledger typer.)

**5. Typing bought nothing on the ledger once merge timing is controlled — its measured value
is the veto.** Wave 1's apparent `self`-over-`sched`/`rand` edge (0.0095, 1.2× placebo floor)
was merge lateness: seven merging arms span six merge steps in-tag, corr(merge step, integral)
= −0.53, and at matched timing (`self_v2` 8050 vs `sched` 8000) typing reads 0.009 *worse*.
Where the per-event ledger is unconfounded it is decisive the other way: `sur_merge` — the
surprisal twin that cannot type — merged on the first burst at **+12.5× floor** (+14.0× on the
drift), exactly the failure the spec predicted. Typing's job on this world is guarding the
irreversible op, not choosing among ops. Relatedly: **merge destroys the typing instrument
itself** — `basis_sep` collapses to exactly 0.0000 on every merged arm (and the whole triple
shrinks: dk +0.062 vs +0.559 at the same drift) while never-merged arms keep it alive and
deepening. The op a typer most consequentially fires is the one that blinds it.

**6. The threshold fixes worked exactly as replayed from logged reads — and the drift residue
is structural.** Wave 2's re-derived dead zones + dropped `rot_requires_out` conjunction
recovered the missed s8000 merge at precisely s8050 and cleaned the burst typing, with no
re-run needed to know it (the replay predicted both). The zero-lag drift read (epoch turn +
micro-grid) lifted drift signatures 3–4× and bought the program's first correctly-typed drifts —
and the two remaining misses sit 0.0005–0.013 under a floor that is itself the honest limit:
the event whose correct op is "just learn it" self-erases (finding 2).

**7. Factor M: the meter's type must match the op it grades.** The FM-alone arm (`sur_fm`)
never fired a merge — pre-stated SNR ≈ 1 in this drifting world (Gate 0's clean sign separation
was measured on a static reader), confirmed as a clean negative: self-modeling did not
substitute for the second basis at the instant. The FM-as-grader arm failed twice, identically:
pre-fix via a coding fault (skips froze the FM; `dg` = −1.000 exactly on 5/9 reverts) and
post-fix (0/8 degenerate) via the located design fault — **a reader-movement statistic is
systematically small precisely when the op under test suppresses reader movement**, so the
fail-open rule ("revert unless confirmed") undid nine of ten correctly-typed skips; the fix
moved the outcome 0.05× floor (identical merge and divergence steps). The one live call in both
runs was the FM correctly detecting the drift's input shock and the op map converting it to the
wrong op. Wave 1's loud burst lag-signatures (+1.3…+8.5) belong to arms that *consumed* their
bursts — values a skipping arm can never reproduce. The trial meter (`self_abba`) reached the
same merge verdict as T\* 450 steps later at 1091 charged step-equivalents and the worst
merged-arm pathway (0.480). Design consequences on the record, unvalidated: fail-closed burden;
input-side statistics (the shadow-batch residual) for grading skips; movement statistics for
movement-permitting ops.

**8. The dissociation held against all fourteen policies.** corr(within-level lifetime
integral, terminal d4-rand) = **+0.754**; merge status is the axis (merged 1.5171 ± 0.0109 /
0.660 ± 0.070; never-merged 1.4915 ± 0.0383 / 0.396 ± 0.224). Full ledger:

| arm | wave | merged | integral ↓ | d4(rand) ↑ | skips |
|---|---|---|---|---|---|
| `track` | w1 | — | **1.4419** | 0.265 | 0 |
| `sur_fm` | w2 | — | 1.4522 | 0.260 | 6 |
| `self_rekey` | w2 | — | 1.4802 | 0.219 | 13 |
| `sur_skip` | w1 | — | 1.4951 | 0.214 | 58 |
| `self` | w1 | 10050 | 1.4983 | 0.677 | 7 |
| `sched` | w1 | 8000 | 1.5078 | 0.708 | 0 |
| `rand` | w1 | 8000 | 1.5080 | 0.678 | 6 |
| `self_v2` | w2 | 8050 | 1.5167 | 0.668 | 13 |
| `sur_merge` | w1 | 5025 | 1.5227 | 0.713 | 0 |
| `self_fm_fixed` | w2c | 6025 | 1.5244 | 0.679 | 6 |
| `self_fm` | w2 | 6025 | 1.5248 | 0.675 | 8 |
| `self_abba` | w2 | 8500 | 1.5341 | 0.480 | 13 |
| `pair_parse` | w1 | — | 1.5363 | **0.752** | 0 |
| `rekey_dead` | w2 | — | 1.5432 | 0.668 | 0 |

In-tag floors: 0.0023 seed / 0.0081 placebo p90 (dead pair, `track` adjacent-checkpoint).

## Joint interpretation (agreed 2026-08-31; crystallized in the idea doc)

Think of three ledgers an evaluator could read. The **stream ledger** (event-evidence in inputs)
decays at the learner's absorption rate, which practice accelerates — that is the wall, and it
is why event-typing, though solved offline by T\* from one model, buys only the veto online.
The **weight ledger** is what absorption *creates* — the FM's sign signatures and 1G's disjoint
subspaces both live there. The **standing counterfactual ledger** (capability state on fixed
shadow reads) moves slowly and is readable at leisure — `teacher_slot`'s successful loops read
it and committed pre-rotation, no event needed. The two gauge families that have worked across
the arc are precisely the two absorption cannot drain. Full statement, the edge-relative
inner/outer terminology (a deliberate fence against
[`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md)'s slot usage), the
biology reading (RPE self-erasure; adaptation masking; cerebellum→VTA as plumbing the value
system to the adaptation ledger — Gate 1's FM-as-grader being that circuit in silico, with the
two-node/three-node caveat), and the open tests:
[`absorption_blinds_the_evaluator`](../../../../ideas/absorption_blinds_the_evaluator.md).
Gate 3 (endogenise) is deliberately **not** run as specced — its premise (fold in the second
reader; the mirror risk) was dissolved by finding 1 and re-posed by findings 2–5; it gets
re-specced in a separate conversation.

## Corrections and methods exports (on the record)

- **Matched surprisal by per-checkpoint counterfactual ladders** (evaluate rotation + burst
  ladder on the same weights, interpolate on the keyed reader's spike) removes calibration error
  from every comparison and is validated in-tag (counterfactual vs realized rotation within
  0.03–0.05 nats, same sign 6/6).
- **A typing gauge should be read as per-view movement, not between-view divergence** — the KL
  contraction cost Gate 0 its result until the retroactive re-read; the panel's uncharged
  everything-logged design is what made the recovery free.
- **Static calibration under-floors an acting learner** (in-tag mature-p99 1.1–3.6× the
  event-free calibration's) — dead zones must be re-derived in-tag.
- **The fail-open/fail-closed asymmetry for graders of inaction** (finding 7) — exported as a
  design rule, unvalidated.
- Launcher hazard for reproducers: Modal can retry a completed-but-undelivered worker;
  five bit-identical duplicate spawns raced last-writer-wins on one volume path (`g2c`), and the
  launcher client silently stopped capturing new containers' stdout. Per-attempt output paths or
  an `app list` check belong in any future launcher; the record self-healed here because writes
  were bit-identical.

## Caveats

- One world, one key level/node/loudness, weak-form scaffold regime — inherited
  from the donor.
- The 1G quiet-interval nulls admit absorption tails: every event excess there is a lower bound.
- `g1a`'s own panel never rebinds `pn_leaf` after a drift, so its post-drift panel rows read
  pre-drift sequences (1G re-read each epoch on its own grammar; wave-2's epoch-turn fix
  addressed the eval switch). Treat `g1a` post-drift offline drift rows with that in mind.
- Burst longitudinal comparisons are confounded by design (ρ is calibrated per onset, so later
  bursts are larger); drifts are cumulative (survival 0.688/0.500/0.281 — by era 3 the world is
  substantially not the starting world; per-epoch eval sets and exact-BP references keep the
  denominator honest).
- Ops fire one window after onset by construction; charged arms' budgets expire before 20000
  (`self_abba` s18969, `self_rekey` s18866 — the latter is what accidentally produced the
  frozen-reader control).
- d4-anchored next-level readout, as throughout the arc; post-merge, `basis_sep` and the triple
  read the abandoned circuitry's absence, not competence.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# Gate 0 (structural gate, then the shadow panel):
modal run -m rhm.practice.tuning.tune_lm::gate
python3 rhm/practice/tuning/launch_detached.py --fn tune_lm --tag tn0
python3 rhm/practice/tuning/analyze_tuning.py --tag tn0 --fetch --figures   # incl. section 2d (movement decomposition)
# Gate 1 (wave 1 + baselines; then wave 2; then the fix round):
python3 rhm/practice/tuning/launch_g1.py --tag g1a
python3 rhm/practice/tuning/transparency_g1.py --tag g1a --fidelity
python3 rhm/practice/tuning/launch_g2.py --tag g2a
python3 rhm/practice/tuning/launch_g2.py --tag g2c --arms self_fm   # post-fix FM round
python3 rhm/practice/tuning/transparency_g2.py --tag g2a
# Gate 1G (offline, on the banked checkpoints):
modal run -m rhm.practice.tuning.gate1g       # two entrypoints; see FILES.md
python3 rhm/practice/tuning/analyze_g1g.py
```

Every tag's exact config is in `setup.json` beside its results. Volumes (`rhm-scaling-data`):
`/data/rhm_practice_tuning/{tn0,tn_calib}/`, `/data/rhm_practice_tuning_g1/{g1a,g1_calib}/`,
`/data/rhm_practice_tuning_g2/{g2a,g2c}/` (each with `ckpt/` — ~2.7 GB fp16 reader+FM
checkpoints), `/data/rhm_practice_tuning_g1g/g1g/`. Fetched copies + figures under
`figures/<tag>/`.

## Next steps (queued, not started)

The offline weight-ledger typer check (free, on the banked checkpoints — do per-layer-group Δθ
ratios separate the events at online-available lag?) · the fail-closed + input-side-statistic M
rule as a single validation arm · the Gate 3 re-spec around the three ledgers (separate
conversation) · the absorber-that-also-reports question (an FM with a compensatory channel —
possibly its own node) · geometry reads as an evaluator channel where movement is
absorption-limited · the `/update-beliefs` sweep carrying this node's items · the seed pair on
external citation.
