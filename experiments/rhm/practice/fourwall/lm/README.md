# fourwall/lm — the index ops asked of an endogenous reader: track is free, the whittling is not

**Up**: [`../README.md`](../README.md) (fourwall — the exogenous parent whose shape this node
ports) · **Files**: [`FILES.md`](FILES.md) (machinery, arms, gates, calibration records, caveat
lists — read its caveats before trusting any single number) · **Template sibling**:
[`../../reread/lm/`](../../reread/lm/README.md) (the first exogenous→endogenous port; this node
reuses its NTP-vs-exact-BP-oracle pattern and reproduces its probe ceilings) · **Licensing-condition
sibling**: [`../../merge/`](../../merge/README.md) (`mg_s0`: paced rotation licenses the merge op) ·
**Idea docs**: [`recurrence_manufactures_confounds`](../../../../../ideas/recurrence_manufactures_confounds.md)
(§§2–9; §9 names the exposure this node was built to make well-posed) and
[`practice_manufactures_its_own_credit`](../../../../../ideas/practice_manufactures_its_own_credit.md) (§18).
**Runs**: `fwlm0` (necessity) and `fwlm1` (sufficiency), 2026-08-18; **rank orderings,
signs, and multiples of the measured floors are the reported quantities.**

## The question

Every index-op finding in the arc — scaffold, debt, merge/re-key/retire, the withheld next level —
was measured on **exogenous machinery**: a hand-built lookup-table library, ops we ran on the
learner's behalf. The port question: are these real joints for a learner that maintains its own
representations, or artifacts of our scaffolding? The question only this substrate makes well-posed
(the idea doc's §9 calls it its deepest exposure): **is there a discrete index event, or does dense
learning under varied demand smear its way to the same quotient** — in which case §7's belief-limit
collapses into "generalization"? Round 1 (`fwlm0`) asks what the bare inner loop does; round 2
(`fwlm1`) supplies the missing op exogenously and prices it. Deliberately **no** outer loop, meter,
or value learner anywhere: the point was to measure the unassisted gradient reader first.

## Design in two paragraphs

The library is the model's learned grammar; the index is what its early predictions are keyed on.
Every sequence gets a free prefix token `w` at position 0, a bijection of the true level-2 feature
at **node 0** (governing the first 16 of 64 leaves — the three unindexed level-2 nodes give a
within-sequence admissibility control for free). A rotation cyclically permutes the `w`↔`z` map
(+4 mod 16, orbit 4), unannounced; grammar and derivation distribution are untouched, so the news
is index-only by construction. fourwall's ops became **readouts, not arms**: merge = the
wall-conditional behaviours collapse; re-key = the implied routing table migrates; stale = it keeps
the old addresses. `fwlm0` arms: `true_wall` (never rotates — oracle-index anchor), `wall` (identity
to step 8000, then every 2000; identity return at 14000 = the c121 analogue), `wall_fast` (every 250
— `mg_s0`'s licensing-condition rate), `dead_wall` (uninformative key — binding null + admissibility
control), `no_wall`, `dead_wall_b` (the seed/stream floor). `fwlm1` arms: `merge_1000` / `merge_8000`
/ `merge_13000` (the timing axis on the `wall` schedule), `merge_fast_8000`, `merge_dead_8000` (the
op's own admissibility control), `no_wall` re-run in-tag. **The merge op = collapse `w` to the
neutral filler in the input stream from `merge_at` on** — no parameter touched, and the collapse
target is the same token `no_wall` consumes from step 0, so post-merge a merge arm and `no_wall` see
token-for-token identical batches: every later difference is the inherited weights alone.

Two structural properties carry the inference. **(1) Exact oracles**: Gate 0 is model-free — a right
index is worth 0.1520 nats/token on the indexed span (10.5% of the 1.4503 Bayes level) and 0.0028
outside it, concentration 54.6×; the token-derived pathway's exact probe ceiling is d4 = 0.921
without the wall (1.000 given), so the shortcut-vs-inference race is live; last-anchor ceilings
reproduce `reread/lm`'s published values on a fresh eval draw. **(2) Bit-identity**: arms run as
separate workers with identical seeds and training consumes zero global torch RNG, which removes
the arc's standing stream-position confound — `fwlm1` merge arms are identical *in every printed
digit* to their `fwlm0` twins at all shared pre-merge checkpoints (gate 9a, max|Δ| = 0.0 on five
instruments), so cross-arm and cross-tag contrasts are licensed here, unlike `fw_s0–s3`, and
post-fork divergence is attributable to the op. Floors, measured on the dead pair: 0.0012 nats
(seed) / 0.0039 (placebo p90) on indexed-span NLL. All gates: [`FILES.md`](FILES.md).

## Findings — `fwlm0` (necessity: what the bare gradient reader does)

**1. Available is taken — fourwall's binding null inverts.** The key-anchor probe reads 1.000 with
the wall present at the *first checkpoint* (step 250; 0.161 randomised), and the keyed arms ride the
key to the full exact bracket (oracle capture 1.02–1.03 by ~step 8000). A free spurious key is a
full-value loan and the gradient reader takes it instantly — the sculpting substrate's "available,
not taken" was substrate-scoped exactly as suspected. Representation-level binding precedes
prediction-level routing by ~4000 steps.

**2. The debt is real, flat, and the old address book is destroyed, not shelved.** Each rotation
costs ~0.63 nats on the indexed span (290–580× the seed floor; every control at floor). The cost
does not shrink across six events. The **c121 analogue fails to reproduce**: the identity return
costs +0.68 — as much as any other rotation — because the abandoned map degrades monotonically
(`e_old` 1.36 → 2.01) as the new one is written. Weights pay rent on one map at a time; there is no
mothball to revive.

**3. Track is the one op gradients supply, and it accelerates.** Full re-mapping in 250 → 100 → 50
steps across successive rotations; NLL recovers to near-baseline within 50–100 steps of each event.
The re-mapping is within the wall basis — the model never migrates off it.

**4. No merge, at any rate — including the rate that licenses it.** Index cardinality stays 16 at
every threshold and checkpoint for every keyed arm; wall-conditional behavioural spread rises
monotonically. `wall_fast` keeps a fully-formed index pinned permanently one era stale
(`map_match_old` 1.000, `map_match_cur` 0.000). Where `mg_s0` showed the exogenous op *refuses
without* paced decorrelation, the endogenous reader *does not quotient with* it: the licensing
condition is necessary for the op and not sufficient without one.

**5. The withheld pathway — the arc's index-blindness claim, endogenously.** With the wall
randomised (forcing the token-derived inference route), the control arms climb to d4 = 0.83–0.84
against the 0.921 ceiling; every keyed arm sits at 0.27–0.30 — **including `true_wall`, which never
experienced a rotation** — and the gap widens over training. Probe and probe-free instruments agree
(rank corr +0.700). A model at Bayes-level task error on the indexed span, with the pathway that
would make it index-independent threefold suppressed, invisible in the only currency the loss
reads. The within-sequence control scopes it: at the unindexed node the contrast is within ~1.5×
seed spread, so the licensed claim is span-local.

## Findings — `fwlm1` (sufficiency: supplying the op, and pricing it)

**1. Attribution: the op repairs the exact named failure, cheaply.** Every merged arm's pathway
climbs to `no_wall` parity by budget end (terminal d4 0.81–0.86 vs 0.841; probe-free twin agrees to
≤0.004 nats of excess). The foreclosure deficit is visible *at the fork* (merged arms below
`no_wall` at matched step: 0.386 vs 0.433 at s8000) and then closes. `merge_dead_8000` prices the
token swap itself at +0.0032 (at floor): ~97% of the op transient is genuine index deletion, and the
whole transient decays within 25–125 steps, 12× concentrated on the indexed span. A weight-free edit
to the input stream suffices — the op lives on the conditions of learning, as §18 specifies.

**2. Timing: a crutch, not a trap, within this budget — and stickiness is rewarded.** Even
`merge_13000` (three rotations deep) recovers to within 1–3× the placebo floor of `no_wall` at
terminal, leaving a −0.031 d4 residual that may or may not close past the horizon (untested). The
lifetime integral is monotone in lateness (1.4729 < 1.4991 < 1.5304): the scaffold pays oracle-grade
rent while held and the rebuild is cheap, so **later merging dominates earlier over this horizon**.
fourwall's "a good scaffold raises the bar for its own replacement" appears here as measured
lifetime economics. Scope: this world's rebuild is cheap; whether the loan acquires a deadline
where rebuilding is expensive is exactly what this design cannot say.

**3. A correction to `fwlm0`, on the record.** `fwlm0`'s finding that `wall_fast` runs +0.149 nats
worse than `no_wall` was **~8× overstated by a checkpoint-grid artifact**: its 250-step grid sampled
that arm only at era boundaries — zero steps into each new map, its maximum-debt instant. On
`fwlm1`'s offset 125-step grid the fast rotator is 0.100 nats *better* than `no_wall` mid-era, and
era-averaged the net price of tracking at that speed is **+0.019** (~5× placebo floor). The
mechanism (always exactly one era stale, never quotients) stands; its price does not. This brings
the endogenous learner into agreement with `mg_s0`: **rotation-tracking and merge are near-substitutes
on task error** — the case for the op never lived in that currency. Against the corrected baseline,
`merge_fast_8000` still posts the tag's best terminal NLL (1.4648) and highest pathway recovery
(d4 0.859).

**4. The lifetime trade: the within-level meter votes against merging.** Lifetime integral ordering:
`true_wall` 1.4202 < `wall` 1.4619 < `merge_13000` 1.4729 < `merge_8000` 1.4991 < `merge_1000`
1.5304 ≈ `no_wall` 1.5311 ≈ `merge_dead` 1.5312 < `merge_fast` 1.5338 < `wall_fast` 1.6922. On task
error over this horizon, never merging is optimal — while the never-merge arms end with the
inference pathway at 0.27–0.30 (hollow) and the merged arms end whole (~0.84) at a terminal task
cost of ±0.004, i.e. nothing distinguishable. Whether to merge is a bet on demand futures — §7's
demand-weighted MDL as an operational decision, and fourwall's mothball-vs-delete bet at the level
of whether to whittle at all.

**5. The abandoned circuitry evaporates — and it is rent, not decay.** Post-merge, wall-conditional
behavioural spread collapses ~200× (0.09 → 0.0004), fastest in the first 250 steps; the formerly
right wall ends slightly *worse* than the neutral token (−0.0137). Weight decay (0.01, on) is
arithmetically far too slow to account for this; the rate tracks ongoing learning pressure (largest
during post-merge re-equilibration). Unused function in weights pays rent in gradient traffic —
which is simultaneously why mothball is impossible endogenously and why the op's garbage collection
came free. Retire-as-delete is a property of the medium, not a hyperparameter.

## What the two rounds jointly say (agreed interpretation, 2026-08-18/19; idea-doc revision queued)

1. **On this substrate, §9 resolves: the quotient is not what dense learning does under varied
   demand.** No discrete index event, and no smooth convergence to invariance either — the learner
   converges to a stale index rather than to the quotient, at any rotation rate tested. Gradients
   natively supply **track**; merge, re-key-onto-an-earned-basis, and mothball are absent. The
   whittling ops look like genuinely missing organs, measured on an actual NTP transformer rather
   than a stand-in.
2. **Both the op and its justification must come from outside the within-level loop.** The
   capability gap: the op transient (+0.10–0.17 nats) is the measured height of the barrier between
   the tracked solution and the quotient basin — a locally-worse region a local rule will not cross,
   though it recovers within ~100 steps once crossed. (Named untested escapes: grokking-style late
   transitions at horizons ≫20k steps; capacity pressure — this model was comfortably sized.) The
   incentive gap: in this deliberately unmetered regime the lifetime ledger *rewards* keeping the
   scaffold, and the deeper half — the withheld pathway — is denominated in a currency (next-level
   extraction) that no within-level signal reads at any meter setting. §18's line measured twice
   over: the factory cannot manufacture the price of its own next level.
3. **The two currencies dissociate at the barrier itself**: at the instant of the op, task NLL is
   spiked while the pathway readout is already climbing. A quantity exists that is improving inside
   the dip — which is what an outer loop (a value learner holding a budget and an invariance probe)
   would need to read to fund the crossing. This round deliberately contained no such loop;
   building the minimal one is the named next rung.

## Caveats

- **The dead pair is the only replicate and supplies the floors**
  (0.0012 seed / 0.0039 placebo). Terminal differences of ±0.004 among merged arms and
  `no_wall` are unrankable; the claims ride on trajectories, integrals, transients, and 10×+ floor
  multiples. The per-arm-worker design removes stream-position noise from every cross-arm contrast.
- **The timing axis brackets, it does not locate** (1000/8000/13000; no `merge_4000`), and
  `merge_13000` confounds lateness with rotation stress (no unrotated-late control).
- **One lever throughout**: perfect `w`↔`z` correlation, one key level and node, one rotation
  family, one loudness. The weak-form scaffold regime only (the true key is representable and
  cheaply extractable — Gate 0's 0.921); partial correlation, the strong form, capacity pressure,
  long-horizon (grokking-window) runs, metered variants, and truth-/demand-news crossings are all
  unbuilt, as are partial merge (16→4), randomise-instead-of-collapse, and re-key/retire as
  supplied ops.
- **Post-merge, `binding`/`map_match`/capture read the abandoned circuitry, not competence**; all
  recovery claims rest on indexed-span NLL vs `no_wall` at matched steps (an exact comparison) and
  the d4 probe/probe-free pair, as pre-stated at launch.
- d5/d6 have little dynamic range at this model size (as in `reread/lm`); this is a d4-anchored
  story. Full caveat list, including instrument granularities and the local-log truncation notes:
  [`FILES.md`](FILES.md).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run -m rhm.practice.fourwall.lm.wall_lm::gate
python3 rhm/practice/fourwall/lm/launch_detached.py --fn wall_lm --tag fwlm0 \
    --arms "true_wall,wall,wall_fast,dead_wall,no_wall,dead_wall_b"
python3 rhm/practice/fourwall/lm/analyze_wall_lm.py --tag fwlm0 --fetch --figures
python3 rhm/practice/fourwall/lm/launch_detached.py --fn wall_lm --tag fwlm1 \
    --arms "no_wall,merge_1000,merge_8000,merge_13000,merge_fast_8000,merge_dead_8000" \
    --ckpt-every 125
python3 rhm/practice/fourwall/lm/analyze_wall_lm.py --tag fwlm1 --ref-tag fwlm0 --fetch --figures
```

Every tag's exact config is in `setup.json` beside its results (volume `rhm-scaling-data`,
`/data/rhm_practice_fourwall_lm/<tag>/`; fetched copies under `figures/<tag>/`). Figures:
`fig1–fig5` per tag (competence / capture + stale attachment / index instruments / probe grid /
per-latent routing rasters) and `fwlm1`'s `fig6_merge` (the sufficiency answer figure).

## Next steps (queued, not started)

The decision round — a minimal outer loop holding a budget and the invariance probe, asked whether
it will *choose* the merge and hold through the dip (the convergence point with the two-timescale
value-loop line) · a metered variant (does pricing feedback flip the incentive on the rent half,
as the arc predicts, while leaving the withheld-pathway half invisible) · the loudness /
partial-correlation axis and the strong-form scaffold regime · partial merge and
randomise-instead-of-collapse · a long-horizon run through the grokking window · the seed pair for
the floor-adjacent readouts (last-anchor spillover; merge_13000's terminal residual) · the idea-doc
revision (§5's triad gains "track is native; merge/re-key/retire are supplied"; §7's belief-limit
gains "the quotient must be bought"; the fwlm0→fwlm1 `wall_fast` correction propagated).
