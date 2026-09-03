# committee_head — a committee of forward models as the reader, a trained head as the judge

**Up**: [../README.md](../README.md) (mjc) · **Design**: [SPEC.md](SPEC.md) · **Files**: [FILES.md](FILES.md)
**Substrate parent**: [`../on_policy/directed_on_policy/`](../on_policy/directed_on_policy/README.md) (E3), forked.
**Program**: [`ideas/two_timescale_value_loop.md`](../../../ideas/two_timescale_value_loop.md) · the atom is
[`ideas/activation_to_activation_forward.md`](../../../ideas/activation_to_activation_forward.md)
§"Learned valence tagging" — *a small auxiliary classifier predicting whether a given error will prove
reducible over subsequent training* — proposed in May and unbuilt until now.
**Status**: Phases A + B run and clean; Phase C run and resolved as an incompatibility. **Date**: 2026-09-03.

---

## One-liner

A small head trained **reward-free** on the *second-moment signature* of a committee of forward models
learns the three-way split — mastered / novel-and-aleatoric / novel-and-learnable — at **AUROC 0.857**,
and **generalises to regions it never trained on** (leave-one-region-out **0.882**). Committee
disagreement is the best single reward-free reducibility reader (**0.716**), beating both the incumbent
counterfactual survey it would replace (`lprog`, 0.618) and a *privileged* reader given a matched-FM
ceiling (0.618). Reading the split did **not** change allocation, for a structural reason E3 already
named. Getting there required fixing two substrate defects — one of which is inherited from E3 and
re-scopes its published ladder — and the port to RHM turned out to be **not askable in that framing**:
there the supervision and the ground truth decompose along orthogonal axes.

## The question, and why a committee

Prediction error bundles *"I have not learned this yet"* with *"this is inherently random"*. The record
says a single FM cannot separate them: its first moment is the unbiased conditional mean (Wiener gain
≈ 1), its error *magnitude* is the reader's output entropy (R² 0.90) with revision R² 0.0001
([`a2a/conditional_revision`](../../a2a_forward/conditional_revision/README.md) Gate D), and all the
structure is in the error's **second** moment (15× across directions,
[`REPRESENTATIONAL_DIVERGENCE`](../../a2a_forward/REPRESENTATIONAL_DIVERGENCE_README.md)). So the reader
must pool over repeats — a committee. Separately, *"is it learnable"* ≠ *"does it matter"*: relevance is
not in the epistemic signature ([`curiosity_control/`](../curiosity_control/README.md) Finding 2) and has
to come from rolling the plan through the FM.

Every composition in the record was **hand-specified** (`lprog × visits`, `grounded@b` swept, `b(s)`
fixed in form). This cut asks whether the composition can be *learned*.

**Design.** E3's substrate forked (arm; one on-reach reducible target A; off-reach reducible
distractors; off-reach noise regions; reflecting OU walk on the curl gains; on-policy metered collection
*and* survey; sighted grader = region-A FM error). The single `net` becomes a K-member random-prior
committee whose **mean plans**. Per region per round the loop reads a reward-free signature — committee
error `e`, disagreement `d`, benchmarked error `b(s)−e`, agency gap `g`, plan occupancy `v`, plus lags —
and a head **shared across regions** maps it to (predicted reducibility, predicted relevance).

Two properties make this a measurement rather than an inference from allocation. A **shadow head trains
on every arm** without allocating, so "did it learn the split" is separable from "did its allocation
help". And **fork fidelity is a gate**: at `--k-members 1 --rpf-beta 0` the `value` arm reproduces E3
bit-identically (21/21 checks, `array_equal`, 14/14 rounds). The one non-exact quantity — a region centre
differing by 1.11e-16, 0.5 ulp — reproduces when E3 is run against *itself* on a second container, so it
is host round-off, not fork divergence.

## Phase A — two substrate defects, found and fixed

**(1) Four of six regions were never entered.** On E3's *own published* `ladder_s0` `value` arm:

| region | eval-path visitation | collection `in_share` |
|---|---|---|
| A | 0.833 | 0.462 |
| Boff1 | 0.084 | **0.413** ← what an off-reach distractor should be |
| Boff2 / Boff3 / Doff1 / Doff2 | 0.000–0.072 | **0.000** |

Their metered survey therefore fell through the `min_in` guard onto out-of-region transitions, so every
per-region quantity for them described somewhere else. E3's headline survives this (`lprog × visits` is
carried by `visits`, which is rolled through the FM and needs no in-region measurement); a *per-region
signature* does not. This is [`metered_repair/`](../on_policy/metered_repair/README.md) §7.2's named hole,
and `Boff1` proves the two properties are separable rather than one. The cause is `_farthest_points`,
which maximises mutual separation and so selects *for* unenterability. `--reachable-off` applies this
node's own gotcha #1 discipline — classify by **actual** visitation, never geometry — to *collection*
reachability, probing each candidate with an independent field-free instrument.

**(2) The budget bought unequal amounts of usable data.** `collect_on_policy` fills `n` transitions by
stepping `n_par` parallel episodes **forward in time**, so it samples timesteps `0 .. ceil(n/n_par)` of an
`ep_len` reach. At the shipped `n_par=16`, **nothing completes a reach**: the survey saw 4 timesteps of
14, and a region allotted 120/6 = 20 transitions saw 2 — the body still at the start posture, nowhere
near the region it was aiming at. Charged steps were unmatched exactly as the arithmetic predicts
(uniform `6·⌈20/16⌉·16 = 192` vs burst `1·⌈120/16⌉·16 = 128`; measured 192 and 128).

The consequence was a ladder that ranked arms by **yield**, not targeting: `corr(A-err, in-region
transitions) = −0.940`, and the all-or-nothing `burst` timing control came first — precisely
[`metered_repair/`](../on_policy/metered_repair/README.md) §7.1's stated worry.

`--episode-budget` denominates the budget in **complete reaches** (`n_par=1`, `need = n_eps·ep_len`), so
charged steps are identical for every policy by construction and every transition comes from a finished
reach. `--gate-invariance` verifies it and reproduces the defect in its control arm:

| denomination | size 1 | 2 | 4 | 8 | ρ(size, yield) | steps/txn spread |
|---|---|---|---|---|---|---|
| **episode** | 0.700 | 0.779 | 0.575 | 0.721 | **−0.023 → FLAT** | **0.000** |
| transition | 0.286 | 0.300 | 0.357 | 0.527 | **+0.669 → SIZE-DEPENDENT** | 0.143 |

After the fix, `mon:coll` is **1.50 for all 12 arms** and `burst` goes from best (0.2098) to worst
(0.2949) — timing is dead as an explanation.

**(3) Signature separability, read-only, under a neutral allocation:**

| channel | reducible vs noise | A vs rest |
|---|---|---|
| `e` committee error | **0.066** | 0.056 |
| `d` disagreement | **0.716** | 0.631 |
| `b−e` benchmarked | 0.613 | 0.572 |
| `g` agency gap | 0.670 | 0.701 |
| `v` plan occupancy | **0.522** | **1.000** |
| `lprog` (the incumbent) | 0.618 | 0.521 |
| `excess*` (**privileged**) | 0.618 | 0.518 |

`e` is *strongly inverted* — committee error is far higher on the noise regions, the noisy-TV structure
read straight off the signature, and why `error-only` sends **71%** of its budget there (E3: 47%). `v` is
0.522 on the epistemic split and **1.000** on relevance across all 12 arms — Finding 2 about as cleanly
as it comes out. And `d`, which is reward-free, beats `excess*`, which is allowed to consume the
matched-FM ceiling.

Committee health is worth recording as a substrate fact: `ens_cos ≈ 0.93` (members leave largely the same
residual), and the **RPF prior buys nothing measurable here** — `d(red)/d(noise)` and the loop-level
numbers are the same with `--rpf-beta 0.6` and `0`. `curiosity_control`'s rationale (distinct frozen
priors make members disagree where data is *sparse*) does not transfer, plausibly because the metered
survey visits every region every round, so nothing stays off-data long enough for the prior to bite.

## Phase B — the head learns the split

The head initially failed (AUROC 0.459) and the diagnostic that explains it is the one worth keeping:

| channel | AUROC vs class | ρ with the head's target `lprog(t+1)` |
|---|---|---|
| `d` | 0.716 | **−0.057** |
| `b−e` | 0.613 | 0.126 |
| `g` | 0.670 | **−0.024** |

**No channel predicted the target.** `lprog`'s round-to-round autocorrelation is **+0.007** — at
single-round resolution the label is white noise about a per-region mean (between/within = 0.29) — while
`visits`' is **+0.681**, which is why relevance was learned easily and reducibility not at all. Pooling
many rounds is how `lprog` scores 0.618 against the class while being unpredictable per round.
*Probe-able does not imply learnable **from a given target**.*

Three methods, each aimed at that diagnosis:

| arm | AUROC | A-err |
|---|---|---|
| baseline (`lprog(t+1)`, huber, MLP) | 0.459 | 0.2718 |
| + smoothed target (mean over 8 rounds) | 0.496 | 0.2700 |
| + rank loss (RankNet within round) | 0.548 | 0.2656 |
| + linear head | 0.650 | 0.2746 |
| **all three** | **0.857** | 0.2730 |

0.857 exceeds the best raw channel (0.716), which is what a learned combination should do — `d`, `b−e`
and `g` each carry partial information. Singles rank linear (+0.19) > rank (+0.09) > smooth (+0.04) and
the combination far exceeds their sum. The linear head being the largest single lever supports the
capacity reading: ~1850 params against ~180 rows.

**It generalises.** Leave-one-region-out (ridge refit offline on the logged signature, every region held
out in turn) gives **0.882** (in-sample 0.759), with all four reducible regions ranked above both noise
regions on held-out data — so this is a rule about signatures, not six memorised constants.

**Allocation is unchanged** (A-err 0.2656–0.2746 across all four arms). The likely reason is structural
and SPEC.md named it: allocation is `pred_red × pred_rel`, relevance is near-perfect and concentrated on
region A, and the on-policy arm has **no visited-but-irreducible region** — E3's own finding that you only
go where you reach — so relevance alone nearly determines where to go.

## Phase C — RHM, and why the question is not askable there

RHM has what the arm lacks: an **exact** reducible/irreducible split by belief propagation on the known
parse tree, with 27.6% of positions purely irreducible *while still carrying surprisal* (mean 0.393). The
same head shape was trained the same reward-free way and scored against it. Three findings, in order of
how much they cost to establish:

- **Nothing in the committee signature beats the base model's own NLL** on the entropy-free
  reducible-share question (best lift −0.133, worst −0.321), and **`d` is robustly inverted** — 0.377 /
  0.320 / 0.366 under three normalisations (prediction-norm, unnormalised, target-norm), *deepening* with
  training, and inverted at every individual tree level (0.438, 0.462, 0.433, 0.370, 0.352). The arm's
  winning head configuration does not rescue it (0.411 vs 0.439).
- **The scoring target needed correcting twice.** `I_red = H_tot − H_irr` is a *component* of total
  entropy, so any channel tracking entropy tracks it — which is why the oracle's own surprisal, the
  quantity this cut was built to show is blind, comes out best (0.895 pooled, 0.910 depth-stratified).
  The entropy-free question is the reducible **share** `I_red/H_tot`, split within level; and the baseline
  is the base model's NLL, not chance.
- **The label and the ground truth decompose along orthogonal axes**, which is the resolution:

  | quantity | between-level share of per-position variance |
  |---|---|
  | epistemic share (target) | 0.469 |
  | `I_red` (target) | 0.282 |
  | **realised drop (label)** | **0.830** |

  The label is *reliable* (split-half **0.934** across disjoint sequence halves) — it is simply 83% "which
  tree level is this", while 53–72% of the targets' variance is *within* level. Conditioning on level
  preserves the target's variance and destroys the label's; not conditioning is confounded precisely
  because the label ≈ depth. There is no scoring granularity at which the two meet.

Stated positively: **in RHM, FM learnability is organised by tree level (the type of prediction problem)
while epistemic content is organised within level (which particular prefix).** The per-position framing
cannot pose the reducible-vs-irreducible question. This is a fact about the substrate, and it explains
every symptom above.

A by-product worth keeping: an exact law-of-total-variance decomposition of the temporal FM's own target,
by enumerating all `v` continuations through the frozen base. It gives **aleatoric_fraction 0.653**
against the **0.665** published in
[`aleatoric_fraction/`](../../rhm/conditional_revision/aleatoric_fraction/README.md) — an independent
route to that number, and a validated floor for future cuts.

## An inherited confound: E3's ladder under matched steps

Defect (2) is E3's, not this fork's, so E3's published ladder was re-run at its published config with
**only `n_par` changed** (3 seeds):

| policy | A-err (`n_par=1`) | published |
|---|---|---|
| error-only | 0.2464 ± 0.0144 | 0.457 |
| lprog-only | 0.2519 ± 0.0159 | 0.447 |
| value | 0.2521 ± 0.0262 | 0.350 |
| uniform | 0.2592 ± 0.0173 | 0.475 |
| visits-only | 0.2718 ± 0.0204 | 0.390 |
| oracle | 0.2806 ± 0.0173 | 0.372 |

The ladder spread collapses from 0.125 to **0.034**, with per-arm standard errors of 0.014–0.026 —
comparable to the whole spread. `value < lprog-only` holds 2/3 seeds (published 3/3) and
`value < error-only` 1/3 (published 3/3).

**What this does and does not say.** `n_par=1` changes more than the metering — it also makes reaches
complete — so this is a different collection regime, not purely a confound removal, and every arm's error
drops ~2× because the data is better. The supported claim is that **E3's ordering is specific to a regime
in which different allocations bought unequal amounts of usable data**, not that the relevance and
reducibility claims are false. Re-establishing them on the corrected substrate is a live question and
this node's machinery is set up for it.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# gate first: the fork's incumbent must BE E3's incumbent
modal run mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --quick --policies value --tag fid_e3
modal run mjc/committee_head/committee_head.py::committee_head --quick --policies value --k-members 1 --rpf-beta 0 --tag fid_ch
python3 mjc/committee_head/check_fidelity.py

# Phases A + B on the corrected substrate (add --gate-invariance on one job)
F="--spawn --episode-budget --budget-eps 12 --mon-eps 3 --oracle-mode excess --ceil-all-regions \
   --reachable-off --reach-shortlist 40 --off-reach-min 0.20 --min-in 5 --k-members 4 --rounds 30 --seed 0"
modal run --detach mjc/committee_head/committee_head.py::committee_head $F --gate-invariance --policies "uniform,error-only,disagree-only" --tag c_base_s0
modal run --detach mjc/committee_head/committee_head.py::committee_head $F --policies "value,oracle,steady,burst" --tag c_inc_s0
modal run --detach mjc/committee_head/committee_head.py::committee_head $F --head-target lprog_smooth --head-window 8 --head-loss rank --head-hidden 0 --policies head_sup --tag m_all
python3 mjc/committee_head/committee_head_agg.py --tags c_base_s0 c_inc_s0 m_all

# Phase C (needs conditional_revision gate0's frozen base on the rhm volume)
modal run --detach mjc/committee_head/committee_head_rhm.py::phase_c --tag pc_full2 --seed 42
```

**`--spawn` is not optional from a restartable container.** `modal run --detach` on a *local entrypoint*
does not survive the client dying — four in-flight jobs were cancelled mid-round by a container restart,
with `Received a cancellation signal` timestamped to it. `.spawn()` queues the call server-side; the
durable artifact is then the volume copy, not the local mirror. E3's entrypoint gained the same flag
(additive, default off, prior paths byte-identical).

## Caveats

- **The arm's headline rests on the sighted grader, not control.** Ladder A-err spans 0.2632–0.2949 and
  is not resolvable arm-to-arm; the split-quality readout is what carries the result, which is why the
  shadow head exists.
- **`excess*` is privileged and is a reference line only** — it consumes the matched-FM ceiling and is
  never a head input.
- **The oracle needed repair to be a valid ceiling.** E3's `stale_mask × per_err × visits` is
  error-scale-sensitive; with off-reach regions at ~10× region A's error it spent ⅔ of its budget
  off-reach and landed 0.018 from uniform, which makes gap-closed figures unstable. `--oracle-mode excess`
  targets the excess over each region's own matched-FM floor, with `--ceil-all-regions` so that floor is
  not itself undertrained.
- **One task family**, and the arm has no visited-but-irreducible region by construction.

## Next steps

1. **`rhm/directed_sculpting/full_loop` is the right home for the transfer question**, for a now-precise
   reason: it has cells under independent support-fixed OU drift, so reducible-vs-irreducible varies
   *within a repeated context class* — which the arm has and per-position RHM does not — and its label is
   cell-level repair rather than per-position drop, so label and target would share an axis.
2. **Re-establish E3's ladder on the corrected substrate.** Its claims are not refuted, only re-scoped;
   the episode-denominated loop is the setup in which they can be re-asked cleanly.
3. **Place an on-reach noise region.** E3 called it unplaceable on-policy; the reachability probe built
   here is arguably the instrument that would place one, which would let the head's split-reading actually
   change allocation.
4. **The reward-driven head (`head_rl`)** is implemented and unrun — SPEC.md's shape (2), to be read only
   after shape (1), which is now read.
