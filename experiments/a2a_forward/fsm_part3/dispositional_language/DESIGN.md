# Design notes — the dispositional battery

**Up**: [`../../confabulation/README.md`](../../confabulation/README.md) (the occurrent confabulation battery this forks) ·
**Files**: [`FILES.md`](FILES.md) · **Code**: [`dispositional.py`](dispositional.py)

This file records the design decisions and why they were made. It is not the writeup;
results and their interpretation live in `README.md`, written after the numbers are
discussed.

---

## The question this node is built to answer

Paper 2's target is **occurrent**: *what am I computing right now*, at one position, at one
training snapshot. The question here is whether the same first-person advantage extends to
**dispositional** facts — how M's computation is changing, has changed, and will change.

Three things from the record shape the design:

1. **Calibration and entropy are public** (paper 2 §5; `OOD_ROBUSTNESS_README` Exp A).
   So the battery must carry behavioural rows in *both* time directions as the built-in
   dissociation, exactly as paper 2 does for the occurrent case.
2. **`committee_head` Phase B**: a head trained on FM-derived signals to predict future
   learning progress sat at AUROC 0.459 because the per-round target's autocorrelation was
   +0.007. *Probe-able does not imply learnable from a given target.* Hence the
   autocorrelation gate, run before any head is trained, and the horizon sweep and
   smoothed variants computed unconditionally so a widen-k fallback needs no relaunch.
3. **`rhm/residual_decomposition/trajectory`**: along training the residual goes from
   noise-shaped to computation-shaped while the FM's own cosine falls. That reorganization
   is the raw material of the IMPL targets — and it is also why the earliest checkpoints
   are the junk-residual risk here (at step 0 a fresh FM reads cosine ~0.996).

Every scalar collapse of the residual has nulled in this program, so the targets are
directional (cosine between residual directions at two snapshots) rather than magnitude.

---

## 1. Substrate and wake

**OL arm only, the parent's recipe verbatim.** Paper 2 found CL and OL indistinguishable on
language, and OL has no injection to complicate checkpoint semantics (the parent's own
assert exists because `intermediates[post_block{i}]` is recorded *before* the injection is
added). 4L/4H/256D GPT on 10M FineWeb-Edu tokens, `T=128`, 10k steps,
`post_block0 → post_block2`, report from `post_block3`.

**Bit-faithful to the parent.** `wake_trajectory` replicates `confabulation_test`'s OL path
down to the RNG consumption order — the throwaway `_init` GPT under `manual_seed(seed)`, the
re-seed, then the forward-model construction, then a separate `Generator` for batches. The
model has no dropout, so the trajectory depends only on that generator. The final state is
written to the harness's standard `wake_ckpt/{ck_key}.pt` with the parent's key format and
content shape (`{"model":…, "fm":…}`), so a sibling running the parent battery on the OL arm
at this seed can load it instead of retraining. TF32 is deliberately left off (the parent
does not enable it); enabling it would produce a *different* model and break that promise.

**Checkpoint schedule: 16 log-spaced steps from 300 to 10000** —
`300, 380, 479, 605, 764, 965, 1219, 1540, 1946, 2458, 3105, 3923, 4956, 6261, 7910, 10000`.

- *Log, not linear*, because learning is front-loaded: equal **ratios** of steps buy roughly
  equal amounts of learning per interval, which is what makes "horizon k in checkpoints" a
  comparable unit across the trajectory. The consequence — that "one checkpoint" is 80 steps
  early and 2090 steps late — is the point, not a defect, but it means the retrospective
  "age" targets are ages in log-training-time.
- *Starting at 300, not 0*, because a barely-trained M is nearly trivial to predict and the
  instrument saturates; the trajectory node above measured cosine 0.996 at step 0. The guard
  table is printed per checkpoint so early rows can still be dropped after the fact.

---

## 2. The fixed report set

The **same** N = 2000 held-out val sequences at every checkpoint, drawn once from
`report_seed`. This is essential: the targets are per-position across time, so the position
index has to mean the same thing at every snapshot. N is below the parent's 3000 because the
aggregator holds every checkpoint's report set resident at once (~2 GB per full-width fp16
tensor across 16 checkpoints); 2000 sequences still gives 2.64 M pooled training rows and
152 k strict-eval rows.

Splits are **sequence-level and shared** by heads, observers and target-quantile
boundaries — 80/20 — so a head never trains on positions from a sequence it is tested on.

---

## 3. Per checkpoint

At every checkpoint: M's report-site state `a_c(p)`, M's early state `a_0`, the O_io summary
(top-64 ids and probabilities **plus the four full-distribution scalars** — the parent's
gotcha: without them the observer cannot compute M's entropy and the behavioural controls
show a fake advantage), per-position loss `ℓ_c(p)`, entropy, correctness, and M's own
`ln_f` parameters.

**Instruments**: `ens_n = 3` freshly trained FMs at each of two capacities, `16:0.5`
(default, ~10.6 % of the predicted blocks) and `4:0.25` (~4.9 %), `fresh_fm_steps = 3000`,
the parent's recipe. Guards per checkpoint per capacity: `ens_cos`, `eta2_norm` / `eta2_dir`
/ `eta2_vec`, per-category Cohen's *d* on |r|, FM cosine, |r|.

**One efficiency choice, and it is not an approximation.** The M forward pass is shared
across the two instrument capacities. The parent battery gives every capacity the same
`fm_seed`, hence the same data generator, hence the **identical batch stream** — so one
forward genuinely serves both. Ensemble members *within* a capacity still get fully
independent data, which is what `ens_cos` is about.

---

## 4. The targets

Per position *p*, checkpoint *c*, horizon *k* in checkpoints, window *w* = 4.

| target | direction | family | definition |
|---|---|---|---|
| `IMPL_PROSP_k` | prospective | implementation | `1 − cos(r_c(p), r_{c+k}(p))` |
| `BEHAV_PROSP_k` | prospective | behaviour | `ℓ_c(p) − ℓ_{c+k}(p)` |
| `IMPL_RETRO_w` | retrospective | implementation | `Σ_{i=1..w} (1 − cos(r_{c−i}, r_{c−i+1}))` |
| `BEHAV_RETRO_w` | retrospective | behaviour | `Σ_{i=1..w} max(0, ℓ_{c−i} − ℓ_{c−i+1})` |

Horizons `k ∈ {1, 2, 4}`. The four rows above at `k = 1` / `w = 4` are the **primary**
battery and get the full observer ladder; everything else gets a light ladder or, for the
middle horizons, only the autocorrelation reading.

**Secondary targets**, all computed:

- `IMPL_AGE_w` / `BEHAV_AGE_w` — checkpoints since the most recent *event*, censored at
  `w+1`. An event is a one-interval change in that interval's top quartile (threshold fit on
  train sequences). The "how long ago did it settle" formulation the brief asks for, with no
  integration.
- `*_PROSPSM` — the prospective target averaged over `k = 1..4`. `committee_head`'s smoothing
  fallback, computed unconditionally so a null on `k = 1` does not need a second launch.
- `*_RANK` — continuous variants of the four primary rows, trained with a pairwise rank loss
  and scored by Spearman ρ. Every scalar collapse of the residual has nulled in this program,
  so the categorical rows are primary; the rank rows say whether a null is about k-means.

### The instrument-variation subtlety, and the four things done about it

`r_c` and `r_{c+k}` come from **different** freshly trained FMs, so a naive
`1 − cos(r_c, r_{c+k})` mixes M's own change with instrument variation. Three readings, all
computed at `k = 1`:

| reading | definition | what it controls |
|---|---|---|
| `IMPL_PROSP` | `1 − cos(r_c^{(0)}, r_{c+k}^{(0)})` | nothing — the naive quantity |
| `IMPL_PROSPEXC` | the above **minus** `floor_c(p)` | `floor_c(p)` is the mean over the 3 fresh FMs at the *same* checkpoint of `1 − cos(r_c^{(m)}, r_c^{(m′)})`: the per-position **no-change floor**, i.e. what two instruments disagree by when M has not moved at all |
| `IMPL_PROSPFIX` | `1 − cos(r_c^{FM_c}, a_j(c+k) − FM_c(a_0(c+k)))` | the instrument is *literally held fixed*: the FM trained at `c` is applied at `c+k`, so both sides come from one instrument |

The fourth control is **structural and applies to every target**: the categorical labels are
**within-checkpoint quantile classes**, with boundaries fit on that checkpoint's train
sequences. So anything constant across positions at a given `(c, k)` — residual-stream basis
drift between snapshots, a global instrument-variation level, the global learning rate — is
removed by construction. This is also what stops a head pooled across checkpoints from
scoring by memorising a per-checkpoint constant, and it is why the quantile construction is
primary rather than a convenience.

**A stated assumption**: comparing `r_c` and `r_{c+k}` by cosine treats M's residual-stream
coordinates as the same basis at both snapshots. Gradient descent moves weights continuously,
so this is the usual checkpoint-comparison assumption; the within-checkpoint quantile
construction removes any *uniform* component of basis drift but not a position-dependent one.

---

## 5. The report head

**Reads `a_c(p)` at checkpoint `c` only** — the occurrent state, with no memory of earlier
snapshots. That is the question: does M's *current* state encode facts about its own
dynamics. Trained **pooled across checkpoints**.

**Input transform.** Primary is `ln_f(a_c(p))`, M's own final LayerNorm applied to the report
site. Rationale: activation *scale* drifts a great deal along a training trajectory, and a
pooled head reading raw activations could locate itself in training time by the norm alone;
`ln_f` is the transform M itself applies before its output head, so this is still M's state.
The raw variant is run alongside for the primary rows as a sensitivity column, and the
occurrent control battery uses raw throughout so it stays parent-faithful.

**Architecture.** The parent's MLP (`d → 256 → n_out`, GELU), 3000 steps, batch 4096. A
**linear** head is additionally fit for the primary rows, since `committee_head` found linear
+ rank loss + smoothed target was the combination that worked when an MLP head failed on a
low-autocorrelation target; if the MLP struggles here, the linear column says whether
capacity was the problem.

**Two evaluations, both on held-out sequences:**

- *held-out sequences @ train checkpoints* — the ordinary generalisation number;
- *held-out sequences @ a held-out contiguous CHECKPOINT BLOCK* (indices 8, 9, 10 of 16;
  steps 1946, 2458, 3105) — the strict number, and the one the summary table reports. A
  contiguous interior block rather than scattered singletons, because the interesting claim
  is a rule about states that transfers to a stretch of training the head never saw.

Observers are trained on exactly the same checkpoint split and evaluated on both.

---

## 6. The observers

The parent's ladder (`O_input`, `O_io`, capacity-swept `1:64 / 2:128 / 4:256`, plus a
half-data control on the strongest `O_io`), `O_act` as the activation-access ceiling, and two
additions:

**A checkpoint-index embedding for every observer.** Without it `O_input` is degenerate: the
report tokens are identical at every checkpoint, so a tokens-only observer could not even
tell which snapshot it is looking at, and its "score" would be the position's average rank
across training. An outsider watching a training run obviously knows which snapshot it sees.
Giving it the index is the generous choice, and the whole argument rests on the third party
being given every advantage we can afford.

**`O_hist`, a history observer**: tokens + M's own output summary at checkpoints
`c, c−1, …, c−4` (lag-embedded and masked at the start of the trajectory). `hist_lags = 4`
deliberately matches the retrospective window, so `O_hist` sees exactly the stretch of
behaviour the retrospective targets integrate over. It is the natural **ceiling** for those
rows — it asks whether the fact is knowable from behavioural history *at all*. If `O_hist`
beats the self-report on a retrospective row, that row is public in exactly the sense paper
2's `ENT` row is public. It is reported as a ceiling and is **not** folded into the advantage
statistic, which stays `self − best O_io` as in the paper.

**Matched exposure.** `obs_steps = head_steps = 3000`. An observer step sees
`obs_bs × (T−1) = 32 × 127 = 4064` positions and a head step `head_bs = 4096` rows, so equal
step counts give the two sides nearly identical exposure to the same training positions. The
half-data control and the capacity sweep are the other two axes on which the observer is
checked for starvation.

Observers are **bidirectional** by default, as in the parent — strictly more generous to the
third party.

---

## 7. The autocorrelation gate

Run **before** any head is trained, on train sequences only, for every target with a
continuous form:

- `r1` — lag-1 Pearson across consecutive checkpoints, computed over positions and averaged;
- `rho1` — its rank version;
- `between_frac` — the fraction of the target's variance that is a per-position constant.

This is `committee_head` Phase B's diagnosis turned into a gate: a per-position target that
is white noise across checkpoints cannot be learned at any capacity, and a target that is
almost entirely a per-position constant is being learned as a static fact rather than a
dispositional one. Both readings go in the reduction next to every advantage.

---

## 8. Controls carried over from paper 2

The occurrent battery (`IMPL`, `IMPL_COS`, `BEHAV`, `ENT`, `WORLD`) is run at the **final
checkpoint** with the parent's recipe — raw report-site input, spherical k-means on the
residual direction fit on train rows, the same ladder. Its job is to show the battery alive
on *this* model, so the dispositional rows are read against a known-good occurrent baseline
rather than against the published numbers of a different training run.

Not ported: matched-KL steering and the channel-ablation columns. They are instrument-level
causal tests of the occurrent report and unchanged from the parent; the question here is
whether a dispositional fact is *reportable and private*, which is Test 1's shape.

---

## 9. Compute shape

A CPU coordinator runs three phases, each independently resumable from the volume:

1. `wake_trajectory` — one GPU container, ~30 min, writes 16 checkpoints + the standard
   wake checkpoint.
2. `checkpoint_probe` — one GPU container **per checkpoint**, `max_dop = 8`, ~10 min each.
   These are independent by construction, so spreading them costs no extra GPU-hours and
   turns ~2.5 h of serial instrument work into ~25 min of wall clock. Each writes one part
   file and caches its fresh-FM weights, so a rerun skips training.
3. `analyze` — one GPU container: loads the parts, folds in the cross-checkpoint
   quantities through a sliding window (never more than `max(k)+1` residual tensors
   resident), runs the gate, the battery and the occurrent controls.

Single seed. Memory requests are sized from the smoke's peak RSS and the linear scaling in
`N × n_ckpt`; `peak_rss_gb` is recorded in `results.json` so the next launch can be sized
from the run rather than from the donor.
