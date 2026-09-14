# DESIGN — the dispositional battery: what was decided, and why

**Parent:** [`../../../rhm/confabulation/README.md`](../../../rhm/confabulation/README.md) (the occurrent confabulation battery) ·
**Grandparent:** [`../../../rhm/README.md`](../../../rhm/README.md) ·
**Code index:** [`FILES.md`](FILES.md) ·
**Substrate:** [`../../../rhm/RHM_LATENT_LOOP_README.md`](../../../rhm/RHM_LATENT_LOOP_README.md)

This file records design decisions and their reasons. It deliberately contains **no
results and no interpretation** — those live in the reduction under `figures/` and in
the `README.md` that gets written after the results are discussed.

## The question this node is built to answer

Paper 2's report target is **occurrent**: the direction class of M's forward-model
residual at one position, at one training snapshot. Reports about it carry a first-person
advantage over every capacity-matched observer; reports about behaviour (own correctness,
own entropy) do not. This node asks whether the same privilege extends to **dispositional**
facts about M's own dynamics — how its computation is changing, has changed, and will
change. Those are still implementation facts, so the privilege may survive; they are slow
by nature, so the head has to be trained across a training trajectory rather than at one
snapshot.

## 1. Fork discipline

The wake is `rhm_confabulation.confabulation_test`'s Phase 1 **verbatim** — same DGP, same
8L/8H/256D model, same `ntp_aux_cl` recipe, same `UnifiedGate`, same losses, same RNG
streams, same `init` — with `torch.save` calls inserted at the checkpoint schedule. Nothing
else about the wake changed, so the final checkpoint is the parent's model and its val loss
is a **harness-validation gate** against the published 2.3829.

Everything importable is imported from the parent rather than copied:
`_generate_with_traces`, `_kmeans_fit`, `_kmeans_assign`, `_balance`, `_ensemble_cos`,
`_make_head`, `_fit_head`, `_head_acc`, and the reference `_compute_hierarchy_eta2` from
`rhm_latent_loop`. Copies would drift; the guards have to mean the same thing here as
there.

`ntp_aux_cl` and not `ntp_cl`: latent-loop showed RHM's token-only target has *inverted*
self-knowledge (fresh-FM SK −0.67), so a report battery there tests the channel on a model
with nothing to report. The latent auxiliary target is not optional.

## 2. The checkpoint schedule is geometric, and starts at 400

16 checkpoints, `400 · (50)^(i/15)` for i = 0..15, ending exactly at 20000:

```
400 519 674 875 1135 1474 1913 2483 3222 4183 5429 7046 9146 11871 15409 20000
```

Two reasons for geometric rather than "denser early then linear":

- **Direction symmetry.** The dispositional targets are read at a horizon of *k checkpoint
  indices*. Under geometric spacing a backward window `[c−k, c]` spans the same *step
  ratio* as a forward window `[c, c+k]`, so the prospective and retrospective rows are
  matched in training-time units. Under any other spacing the two directions would differ
  in how much training they cover, and a prospective-vs-retrospective asymmetry could be
  that rather than the direction of time.
- **Denser early where the dynamics are**, which geometric spacing gives for free.

Step 0 is excluded. At initialisation M computes almost nothing, a fresh FM reads cosine
≈ 0.996, and the residual is the noise-floor artifact
([`trajectory/README.md`](../../../rhm/residual_decomposition/trajectory/README.md), Result 3) —
there are no dynamics there to report on, and the guards would fail. Starting at 400 keeps
the earliest checkpoints in view *with their guards attached* rather than assumed clean;
the reduction reports guards per checkpoint so early checkpoints can be excluded post hoc
on evidence.

## 3. Targets: the 2 × 2, and why they are the same statistic read both ways

At checkpoint `c`, position `p`, horizon `k` in checkpoint indices:

| target | what it says | family | direction |
|---|---|---|---|
| `IMPL_PRO` | 8-way class of the direction of `r_{c+k}(p)` | implementation | prospective |
| `IMPL_RETRO` | 8-way class of the direction of `r_{c−k}(p)` | implementation | retrospective |
| `IMPL_NOW` | 8-way class of the direction of `r_c(p)` | implementation | occurrent anchor |
| `REORG_PRO` | within-level quartile of `1 − cos(r^{(c)}_c, r^{(c)}_{c+k})` | implementation | prospective |
| `REORG_RETRO` | within-level quartile of `1 − cos(r^{(c)}_{c−k}, r^{(c)}_c)` | implementation | retrospective |
| `BEHAV_PRO` | within-level quartile of `ℓ_c(p) − ℓ_{c+k}(p)` | behaviour | prospective |
| `BEHAV_RETRO` | within-level quartile of `ℓ_{c−k}(p) − ℓ_c(p)` | behaviour | retrospective |
| `ENT_NOW` | within-level quartile of M's output entropy at `c` | behaviour | occurrent anchor |

**PRO and RETRO are the same statistic read forward and backward from `c`.** This is the
load-bearing symmetry: any difference between the two directions is then about the
direction of time, not about the target's functional form. The brief's alternative
retrospective target ("time since the residual last reorganised") was not used as primary
because it is a hazard-style variable with no prospective twin, and the mirror form *is*
integrated past reorganisation over the window `[c−k, c]`.

**Why the class, not the cosine, is the primary implementation readout.** Every scalar
collapse of the residual in this program has nulled; the standing belief is that the
absorbed signal is directional (*what kind of computation*), not scalar (*how much*).
`IMPL_PRO/RETRO` are 8-way direction classes — the same readout paper 2 uses on the grammar
substrate. `REORG_PRO/RETRO` are the scalar reorganisation-amount rows, kept because the
brief names them and because a scalar/directional dissociation inside the implementation
family is itself informative. The continuous residual-**direction** regression (paper 2's
*language* variant, scored by cosine) was dropped: on the grammar substrate paper 2 itself
uses the categorical class, and carrying a 256-dim target for 12 checkpoints costs ~2 GB
per target for no new axis.

## 4. The instrument problem: two readings of "reorganisation"

`r_c` and `r_{c+k}` come from *different* fresh FMs, so a naive reorganisation measure
mixes instrument variation with M's change. Both readings are computed:

- **Fixed instrument (primary for `REORG_*`).** `FM_c` — the fresh FM trained on `M_c` — is
  applied to *both* checkpoints: `r^{(c)}_{c+δ} = a6_{c+δ} − FM_c(a0_{c+δ})`. The instrument
  is identical on both sides, so the no-change floor is **exactly 0** by construction and
  none of the measured change is the instrument's. `FM_c` being stale for `M_{c+k}` is not a
  confound: it is stale *because* M moved, which is the quantity of interest.
- **Fresh instrument (`REORG_*_fresh`, secondary).** `1 − cos(r_c, r_{c+k})` with each
  checkpoint's own fresh FM. Its no-change floor is `1 − ens_cos` at fixed `c`, which is
  measured at every checkpoint and reported alongside.

For the categorical `IMPL_PRO/RETRO`, the target is the class of the **fresh-instrument**
residual at `c ± k` — each checkpoint's residual defined exactly as paper 2 defines it,
with one shared codebook. The instrument's contribution to *that* label is measured
directly: the same codebook is applied to the residual left by FM seed 1, and the
class-agreement rate is reported (`cls_instrument_agreement`). The reorganisation targets
get the matching reading — the correlation between the seed-0 and seed-1 versions of the
target.

## 5. The codebook is fit on per-checkpoint-centered residuals

M's representation drifts across training, so a codebook fit on raw pooled residuals could
encode "which checkpoint" rather than "which direction". Residuals are therefore centered
per checkpoint (per-dimension mean over the report set) before the spherical k-means fit
and before assignment. Spherical k-means already removes scale; centering removes the
per-checkpoint offset. The codebook is fit on **train sequences at train checkpoints only**,
so the label definition never sees either held-out axis.

The occurrent control battery (§9) keeps the parent's **uncentered** convention, so its
numbers stay comparable to the published ones.

## 6. Reading within level, and the confounds that forces

In RHM, age of acquisition is nearly the same thing as hierarchy level, and level is an
*input* fact any observer reads off the token index (paper 2's WORLD row). committee_head
Phase C found the realised-drop label to be **83% between-level** on this substrate while
53–72% of the epistemic target's variance is *within* level. So:

- **Level** of report position `p` is the s-adic valuation of `p+1`, capped at `L−1`
  (`PER_LEVEL_LOSS_README`): 32 positions at L0, 16 at L1, 8, 4, 2, 1.
- **Continuous targets are quantile-binned within (level, checkpoint)** on train-sequence
  rows. Balancing inside each cell makes the label carry *no* information about level or
  about checkpoint identity, by construction. Pooled-quantile variants are computed and
  reported for the record, together with their level-conditional baselines, so the size of
  the confound is visible rather than assumed away.
- Every row is additionally reported **per level**, and the reduction carries the
  per-level NLL trajectory so levels the model never learns can be noted rather than
  averaged in.
- Two baselines accompany every row: the majority-class baseline and the
  **position-conditional** baseline (the best predictor that knows only `p`). Per-position
  constants are free to the observer *and* to the self-report, so the position baseline is
  the honest floor.

## 7. Inputs are centered per checkpoint, symmetrically

The report state `x7 = post_block{n−1}`, M's logits, and M's `a0` are each mean-centered per
checkpoint using train-sequence rows, for **self, confabulator and every observer alike**.
Centering by a known per-checkpoint constant is information-preserving *within* a
checkpoint; what it removes is the cross-checkpoint offset, i.e. the cue "which snapshot am
I looking at". Since the targets are already balanced within (level, checkpoint), that cue
could only buy a shared constant, and removing it symmetrically keeps the battery about
*states* rather than about snapshot recognition.

## 8. Two held-out axes

One head per (target, instrument), trained on **train sequences × train checkpoints**, is
scored on two disjoint sets:

- `held_seq` — held-out sequences at trained checkpoints (paper 2's split).
- `held_both` — held-out sequences at a **held-out contiguous checkpoint window** (3 of the
  12 target checkpoints, taken from the middle so the window is interpolated rather than
  extrapolated).

A rule about states survives both; a memorised per-checkpoint constant survives only the
first. Observers get exactly the same two splits and the same training rows.

## 9. The observer ladder, and two new rungs

| rung | sees | why |
|---|---|---|
| `O_input` | tokens | the input-only floor; climbs when the fact is in the input |
| `O_io` | tokens + `M_c`'s full output distribution | everything the I/O map offers — **the advantage is defined against the best of these** |
| `O_io_half` | the same at half the sequences | data-budget control |
| `O_act` | `M_c`'s own `post_block0` | activation-access ceiling |
| `O_hist` | tokens + logits at `c` **and** at `c−k` | the historian: the natural ceiling for the retrospective rows |
| `O_acthist` | `a0` at `c` and at `c−k` (implementation rows) | activation-access historian |

`O_hist` is the sharpest control in the design: it can compute `BEHAV_RETRO` almost
*exactly* (it holds both checkpoints' logits and the tokens, hence both losses), while it
holds nothing that yields `r_{c−k}`. A single observer that trivially solves the behavioural
retrospective row and fails the implementation one is a within-battery dissociation that no
capacity argument is needed for. The historian is given the *past* for the prospective rows
too — an outside observer at time `c` may have recorded history, but not the future.

`O_input` is swept at 2 capacities on the dispositional rows and 4 on the control battery
(where it is the "the ladder can climb" control); `O_io` is swept at all 4 everywhere.

## 10. The gate before the heads

committee_head Phase B: a head on FM-derived signals sat at AUROC 0.459 because its target's
round-to-round autocorrelation was +0.007 — *probe-able does not imply learnable from a
given target*. So the per-position autocorrelation of every continuous target across
checkpoints is measured **before any head is trained**, at k = 1..4, within (checkpoint,
level).

One subtlety the first implementation got wrong and this one fixes: the targets are
**differences over a k-checkpoint window**, so at k = 1 adjacent windows share an endpoint
with opposite signs and the lag-1 autocorrelation is biased *negative* whatever the signal
is. The gate therefore reads the **disjoint-window** autocorrelation at lag `k+1` (no shared
term, no induced correlation); lag 1 is reported for the record with the artifact flagged.

**Horizon rule** (stated before the run, applied to the gate and to nothing else): take the
smallest `k` whose four disjoint within-level autocorrelations are all ≥ 0.10; if none
qualifies, take the `k` that maximises that minimum. The rule reads the target's own
statistics, never any advantage, so it cannot select for an outcome. For the categorical
`IMPL` family, the analogous reading is class persistence `P(class_c = class_{c+k})` against
its chance rate, reported at every `k`.

## 11. Capacity, the guards, and what could counterfeit this

A saturated instrument manufactures the headline: the report head sits downstream of `a_j`
and can report noise perfectly, while no observer can predict noise from tokens and logits.
The parent produced that artifact deliberately (`ens_cos` 0.65, d6 η² 0.003, and a +0.118
"advantage"). So:

- **Both guards are computed at every checkpoint, at both instrument capacities** —
  `ens_cos` across 3 independent fresh FMs, and hierarchy η² under the reference last-token
  estimator (imported from `rhm_latent_loop`, so it is exactly comparable to the published
  0.322/0.295) as well as the pooled estimator.
- The battery itself is run at **two capacities** for the two implementation rows most
  exposed to the artifact (`h16m1`, the parent's default at 5.6% of the predicted blocks,
  and `h4m0.25` at 1.4%), because paper 2's sharpest check is that the advantage *rises* as
  the instrument shrinks, in lockstep with `ens_cos`.
- Head count is fixed at M's 8 throughout (`RESIDUAL_RANK`: a head-count mismatch dominates
  the residual on its own).

Guards are expected to fail at the earliest checkpoints — that is a fact about the
trajectory, not a flaw, and the reduction reports rows both over all target checkpoints and
with the guard column in view.

## 12. The occurrent control battery

Paper 2's four rows (IMPL / BEHAV / ENT / WORLD) are re-run at the **final** checkpoint with
the parent's conventions verbatim — 12000 report sequences, one sequence-level split,
uncentered residual, fresh `h16m1` instrument, full capacity ladder — so the battery is
shown alive on this model before any dispositional row is read. Published reference:
+0.096 / −0.001 / −0.035 / −0.038 at val 2.3829.

## 13. Sizing

One CPU coordinator drives four GPU stages and fans the independent ones out over
containers (`max_dop` 6): the wake (one container), 16 instrument jobs, one target-building
job, then 10 dispositional cells + 4 control cells. This spreads arms already in the design
— it adds no GPU-hours, only wall clock. Every stage writes its own part to the volume and
skips on re-run, so a crash costs one stage rather than the run. Report set: 3000 sequences
× 63 positions × 12 target checkpoints = 2.3M pooled rows (paper 2 used 12000 sequences at
one checkpoint = 756k rows). Memory requests were set from the smoke's peak RSS rather than
inherited.

## 14. Known limitations of this design

- **One seed, one rule seed, one DGP regime**, as in the parent.
- **The report head is trained.** This is a claim about grounding, not spontaneity.
- **The horizon is a single k** chosen by the gate. The targets exist at other horizons and
  the gate table shows what they look like there, but the battery was run at one.
- **Head/observer budget is 5000 steps** on a pooled set 12× more diverse than the parent's
  single-checkpoint set; self and observers get identical budgets and the half-data rung
  tests the data axis, but nothing here tests the *step* axis.
- **`REORG_*` are scalar collapses** of a directional quantity, and this program's record on
  scalar collapses is a string of nulls. They are included as the brief's named target and
  as the scalar arm of a scalar/directional contrast, not as the primary implementation row.
- **The checkpoint-held window is interpolated**, not extrapolated: nothing here tests
  generalisation to training stages beyond those seen.

## 15. Amendment (made after the first run's gate, before its results were read)

The gate rule in §10 reads the **disjoint-window** autocorrelation at lag `k+1`. On the
first run it selected `k = 1`. A closer look at the differencing algebra shows that rule was
stricter than it needed to be, in a way that matters:

windows `[c, c+k]` and `[c+lag, c+lag+k]` share an index **only when `lag == k`**. So the
sign-cancellation artifact contaminates lag 1 *only at k = 1*; at every `k ≥ 2` the lag-1
autocorrelation is already clean of it. Reading lag `k+1` at every `k` therefore compares
different actual separations across horizons, and at larger `k` it reads a separation far
enough out that mean reversion dominates.

Under the clean reading, the two families separate sharply: the reorganisation targets are
persistent at every horizon, while the behavioural targets have essentially no persistence
at `k = 1` (their only uncontaminated statistic there is the lag-2 reading, ≈ −0.05) and
substantial persistence from `k = 2` upward (lag-1, clean, +0.28 to +0.53). A battery run at
`k = 1` therefore tests the behavioural rows on a target that is close to white noise at
that resolution — the committee_head Phase B situation, where a null says nothing about
privilege because nothing was learnable from the target.

So the battery was run at **two horizons**: `k = 1` (what the stated rule selected) and
`k = 3` (the smallest horizon at which all four continuous targets are solidly persistent on
the clean statistic). Both are reported. This is still a gate decision — it reads only the
targets' own autocorrelations, never any advantage — and the second arm reuses the same
checkpoints and the same instrument parts, so the two horizons differ in nothing but `k`.

Mechanically: `build_targets` and `battery_cell` take an `out_suffix`, so a second horizon
writes `targets_k3.npz` / `cells/<cell>_k3.json` / `main_k3_summary.json` beside the first
and shares `ckpt/`, `fm/` and `parts/`. `--no-ctrl` on the second arm: the occurrent control
battery is a single-checkpoint measurement and does not depend on `k`.

## 16. The open-loop arm (added after the first arm's reduction, on the coordinator's call)

Table 3 of the first reduction shows the closed-loop model's **standalone** NLL *rising*
over training at every level except L1 (L0 +0.63, val 2.116 at step 400 → 2.383 at 20000).
That is injection dependence, not degradation: `ntp_aux_cl` is trained with the forward
model's forecast injected after block 1, and the battery — following the parent verbatim —
reads standalone forward passes. So on the CL arm the behavioural targets `ℓ_c(p) −
ℓ_{c±k}(p)` are partly *"my dependence on the injection is growing"* rather than *"my answer
is improving"*, which makes the BEHAV rows hard to read as learning facts.

The **`ntp_aux` (open-loop) arm** fixes this: with no injection, the standalone loss *is* the
model's behaviour, so a loss change is a learning fact. It is also the like-for-like
comparison with the language sibling, which is open-loop. Paper 2's own grammar OL row is
the harness README's `ntp_aux` line: val **1.5447** is the validation gate, and the occurrent
IMPL advantage there was smaller but still positive (+0.06) with the three control rows at
about zero — so the arm is known to carry the effect this battery is built to track, in
weaker form.

Everything else is held: same DGP, same rule seed, same 16-checkpoint geometric schedule,
same instrument capacities and seeds, same guards, same report set, same splits, same head
and observer budgets, single seed. Both horizons (`k = 1` and `k = 3`) are run on both arms,
which is cheap once the instrument parts exist — only `build_targets` and the cells repeat.
The occurrent control battery is shared across horizons within an arm (it is a
single-checkpoint measurement and does not depend on `k`).

Mechanically this added a `horizons` parameter to the coordinator: it runs the control
battery once and then loops `build_targets` + cells per horizon, writing
`<tag>_k<h>_summary.json` per horizon over one shared `ckpt/`, `fm/` and `parts/`. The CL
arm's two summaries keep their original names (`main_summary.json` for k=1,
`main_k3_summary.json` for k=3) so the first arm's artifacts are not renamed after the fact.

**What this arm can and cannot settle.** It makes the behavioural rows readable as learning
facts and puts the grammar substrate on the same footing as the language sibling. It does
*not* separate "the loop amplifies dispositional privilege" from "the loop changes what
there is to report on" — the parent established that CL and OL residuals are about equally
DGP-structured (d6 η² 0.322 vs 0.295 on the reference estimator) while the CL model *encodes*
that structure better, so an arm difference here is expected to track the self-knowledge gap
rather than a difference in the residual itself.

## 17. Which instrument reading the REORG rows are (answer to a question from the coordinator)

Stated plainly, because the two readings are easy to confuse and the language sibling's
result turns on the difference:

- **The headline `REORG_PRO` / `REORG_RETRO` rows are the FIXED-instrument reading.** The
  cells are run on `REORG_*_bin`, which bins `reorg_fixed(c, ±k, fms0)` — `FM_c`, the fresh
  instrument trained on `M_c`, applied to *both* sides:
  `1 − cos(a6_c − FM_c(a0_c), a6_{c±k} − FM_c(a0_{c±k}))`. The same network sits on both
  sides of the cosine, so **the no-change floor is exactly 0 by construction** and none of
  the measured reorganisation is instrument variation. There is consequently no
  excess-over-floor correction to make on these rows: they are already the corrected
  quantity.
- **The `REORG_*_fresh` variants are the two-fresh-FM reading**, `1 − cos(r_c, r_{c±k})`
  with each checkpoint's own instrument. They appear only in the confound diagnostics
  (reduction table 5) and in the per-level target trajectory. **No cell was ever run on
  them**, so no advantage number in any table is a fresh-instrument number.
- The **categorical** `IMPL_PRO` / `IMPL_RETRO` rows are different again: their target is the
  class of the fresh-instrument residual at `c ± k` under one shared codebook, which is how
  paper 2 defines "the residual" at a checkpoint. Their instrument contribution is bounded
  directly by the class-agreement reading (same codebook, FM seed 1): OL 0.684, CL 0.796.

For the fresh reading, the matching floor is the per-position disagreement between
independent fresh FMs at the *same* checkpoint, `1 − cos(r^{s_i}_c(p), r^{s_j}_c(p))`,
averaged over seed pairs. `ens_cos` is the pooled, input-centered version of that number;
`floor_by_level` recomputes it per position and per level, raw and centered, from the saved
instrument FMs and cached state (no retraining), so the fresh reading can be read as an
excess over its floor.

## 18. Follow-ups on the OL arm: the three instrument readings, and the rank readout by level

Two additions, both re-analyses over artifacts already on the volume — no new wake, no new
instruments, no new checkpoints.

**(1) All three reorganization readings through the cells.** §17 established that the
headline `REORG` rows were the *fixed*-instrument reading only. The language sibling's
prospective advantage vanishes under both the excess-over-floor and the fixed readings, so
the grammar needs all three on the same footing before the two substrates can be compared
on that row:

| reading | target | no-change floor |
|---|---|---|
| raw fresh | `1 − cos(r_c, r_{c±k})`, each checkpoint's own fresh FM | the instrument's own disagreement |
| excess | the same, minus the per-position same-checkpoint floor | removed per position |
| fixed | `1 − cos(r^{(c)}_c, r^{(c)}_{c±k})`, one FM on both sides | 0 by construction |

The floor is `floor_c(p) = mean over independent fresh-FM pairs of
1 − cos(r^{s_i}_c(p), r^{s_j}_c(p))`, taken at checkpoint `c` rather than averaged with
`c ± k`, because the no-change counterfactual is exactly "M did not move, so the far
instrument is just another fresh FM on `M_c`". `ens_cos` is the pooled, input-centered
version of the same quantity.

All three are quantile-binned within (level, checkpoint) on train rows, so the three targets
are directly comparable and none of them carries level or checkpoint information. The floor
subtraction is not a rank-preserving no-op — the floor varies per position, so it reorders
rows inside each (level, checkpoint) cell; the reduction reports the rank correlation
between the fresh and excess targets so the size of that reordering is visible. Each reading
gets the full cell battery: the same head, the same ladder including `O_hist` and `O_act`,
scored pooled and within level, on both held-out axes.

**(2) The continuous rank readout within level.** The first run stored only the pooled
Spearman for the continuous models, which made the deep-level question unanswerable for that
readout. `battery_cell` gained a `cont_only` mode that skips the categorical battery and
re-runs just the four rank models (self, `O_io@8:256`, `O_hist@8:256`, `O_act@8:256`),
scoring Spearman per level and **saving the per-position predictions** to
`cells/<cell><suffix>_contpred.npz`, so any further rank-based question can be answered
without retraining anything.

Both run at both horizons. Cells are named `<row>_fresh` / `<row>_excess` / `<row>_cont` so
nothing overwrites the first run's cells; the `fixed` column in the reduction is read from
the original `REORG_PRO` / `REORG_RETRO` cells. Single seed, everything else unchanged.

**What this cannot settle.** The three readings differ in their instrument treatment and in
nothing else, so a difference between them is attributable to the instrument — but the raw
fresh and excess readings share a definition in which *both* endpoints carry instrument
variation, while the fixed reading has none at either. A reading-to-reading difference
therefore bounds the instrument's contribution; it does not decompose it into "variation at
`c`" and "variation at `c ± k`".
