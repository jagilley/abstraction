# Conditional revision on language: does a reader's state distinguish "this token changed what I believe" from "this token was surprising"?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../ideas/revision_not_surprisal.md)
**Sibling / predecessor**: [`../../rhm/conditional_revision/`](../../rhm/conditional_revision/README.md)
— the same question on RHM. Its README's tracking appendix is what set this cut's design.
**Date**: 2026-08-09 · **Status**: substrate built and self-tested; Gates 0, A, B1–B3, D, E run.
One reader (`Llama-3.2-1B`, frozen), one regime.

## Goal

The idea doc's one-liner is *"felt surprise is how much a word changes your model, not how
improbable it was."* RHM answered a version of this exactly, but its own audit flagged the limit:
*"oracle revision is zero exactly when the prefix already determined the structure, so 'prefix
uncertainty predicts informativeness' is much less surprising on RHM than it would be on language"* —
and indeed exact prefix entropy separated its families at **0.73–0.80**, better than the belief
readout's 0.690. The contrast was close to definitional on that substrate.

This cut asks the question on English, with a frozen pretrained reader, on a substrate built so that
prefix uncertainty is **not** the family label, and so that oracle surprisal is pinned *by
construction* rather than by stratification.

## The substrate — "Chronicle" ([`corpus.py`](corpus.py))

A latent scene `z = (person, object, place, hour)`, each slot uniform over 4 single-token English
words, so `|Z| = 256`. A story is 14 clauses of ordinary (if stilted) English. The **clause plan is
sampled from prefix-determined eligibility only** — never from `z` — and the emissions are the only
z-dependent part. That factorisation keeps the posterior a product of per-slot uniforms over
shrinking live sets, so

```
B_t = KL( P(z | x_<=t+1) || P(z | x_<=t) )
```

is a closed form rather than an inference problem. Sample:

> Late that night the house on Wexley Street stood empty… A small note said the missing item was not
> the compass. All evening the fog had not let up. The catalogue narrowed the missing item to either
> the ledger or the diary. The hallway was quiet and old. A little note said the missing item was
> not the telegram.

Three self-tests, all exact and all in `python3 -m a2a_forward.conditional_revision.corpus`:
the factorised posterior support against brute-force enumeration over all 256 scenes (**0
mismatches**), `H_post` against `log |support|` (**0**), and the idea doc's §4 identity
`E_x[B] = H(x_t | x_<t) − E_z[H(x_t | z, x_<t)]` **per position** (max error **1.7e-16**).
Reconstructing `B` from the per-slot posteriors inside the gates agrees to **1.1e-16**.

### Families, and why the negation clause is built the way it is

| family | `B` | oracle surprisal | what it is |
|---|---|---|---|
| `synonym` | 0 | log 4 | one of 4 equiprobable stylistic variants |
| `offtopic` | 0 | log 4 | a 4-way fact about a variable *outside* `z` |
| `mention` | 0 | log 4 | a slot's value word, carrying no information about `z` |
| `reveal` | log 4 | log 4 | states `z_k` with all 4 values live |
| `reveal_narrow` | log L | log L | states `z_k` with L < 4 live |
| `negate_live` | log(L/(L−1)) | log 4 | "…was **not** the compass", compass still live |
| `negate_dead` | 0 | log 4 | same sentence, compass already eliminated |
| `echo` | 0 | 0 | restates an already-revealed `z_k` |
| `deduced` | 0 | 0 | states `z_k` after negations already forced it |

The negation clause names a **dead** value with probability `p = d/4` and otherwise a live non-true
one. With `d = 4 − L` that makes the *marginal* emission exactly `1/4` per value for every state, so
`negate_live` vs `negate_dead` is a contrast **inside one identical template, at one identical clause
position, with oracle surprisal pinned to `log 4` on both sides and only `B` differing**. That is
Gate B1, and it is the thing RHM has no analogue of.

## Two construction failures that were the substrate's real cost

Both were found by running, not by thinking, and both are the reason to keep the smoke discipline.

**1. Deadness was perfectly confounded with prior mention.** In the first version, negation was the
only elimination mechanism, so a value was dead *only because an earlier clause had named it*. The
frozen reader separated the two families at **AUC 0.96 on token surprisal alone** and at **1.00**
from a linear readout of its own state. Both were reading an induction head. The fix is the `narrow`
clause — *"the catalogue narrowed the missing item to either the ledger or the diary"* — which
eliminates values by naming the **survivors**, so the values it kills are never mentioned. That
fills the two cells of (dead/live) × (mentioned/not) that were empty by construction, and drops the
raw `nll` gap from 1.41-vs-6.73 to 3.46-vs-6.11. It is also what puts the idea doc's actual claim
into the corpus; see the sign reversal below.

**2. `narrow`'s own tokens cannot be analysis positions.** Between "either the ledger" and "or the
diary" the true posterior is momentarily non-uniform (reading the first disjunct puts 1/2 on it),
which breaks the uniform-on-a-live-set invariant the oracle relies on. After the clause completes it
is exactly uniform on the named pair again — verified by brute force — so every analysis position
stays exact. The two tokens are marked `analysis=False`.

## Gate 0 — the instrument

3000 stories, 60794 analysis tokens, **0 alignment failures**. The reader reads: `echo` sits at
`nll` 1.18 and `reveal` at 6.88, i.e. it knows a restated value.

| readout | person | object | place | hour |
|---|---|---|---|---|
| **probe** / exact Bayes ceiling | 0.657 | 0.665 | 0.655 | 0.528 |
| **prompted** / exact Bayes ceiling | **0.828** | 0.756 | 0.758 | 0.700 |

The prompted readout — appending *"Q: Which item is missing? A: the"* and reading the model's own
distribution over the four values — beats the trained probe at every slot **with no training and no
ground-truth latents**. RHM left this open (*"whether the probe can be withdrawn after training is
untested"*); on this substrate the answer is that the ungrounded readout is the better one.

RHM's Gate-0 statistic transfers in sign but not in magnitude: `corr(r_temporal, nll)` = **+0.193**,
`R² = 0.037` (RHM: **+0.652 / 0.425**), FM variance explained 0.762. Neither side of the inherited
two-sided kill fires, but the temporal residual is far less surprisal-loaded here than on RHM.

## Gate B1 — the load-bearing contrast

`negate_live` vs `negate_dead`, restricted to states where both families are reachable, on held-out
stories (n = 974 / 655). Columns are progressively stronger matchings; **STRICT** fixes the exact
arriving word, the syntactic frame, the slot's live-set size, the prior-mention count of that word,
and the model's own surprisal.

| readout | raw | `nment` | FULL | STRICT |
|---|---|---|---|---|
| oracle surprisal | **0.500** | **0.500** | **0.500** | **0.500** |
| model `nll` | 0.640 | 0.614 | 0.533 | 0.524 |
| output entropy | 0.466 | 0.434 | 0.406 | 0.390 |
| oracle prefix entropy | 0.551 | 0.545 | 0.493 | 0.472 |
| `delta_norm` (no FM) | 0.512 | 0.489 | 0.468 | 0.502 |
| FM residual norm `r_norm` | 0.421 | 0.401 | 0.385 | 0.349 |
| probe `M` | 0.469 | 0.465 | 0.511 | 0.494 |
| probe `Mpm` | 0.480 | 0.482 | 0.472 | 0.558 |
| prompted `M` | 0.615 | 0.550 | 0.515 | — |
| `h_before_dir` (prefix state) | 0.569 | 0.583 | 0.585 | 0.554 |
| **`h_after_dir` (post-token state)** | **0.997** | **0.997** | **0.992** | **0.993** |
| `r_dir` (FM residual direction) | 0.997 | 0.997 | 0.995 | 0.996 |
| *guard*: `M_shuffled` | 0.494 | 0.480 | 0.522 | 0.502 |
| *guard*: `M_swap` | 0.506 | 0.518 | 0.501 | 0.468 |
| *guard*: **`h_swap_dir`** | **0.506** | **0.506** | **0.493** | **0.472** |

*(pairs: 637970 / 278057 / 839 / 269; the intermediate `tokid × frame × nlive × nment` column has
3032 pairs and reads `h_after_dir` 0.994, `h_before_dir` 0.529, `h_swap_dir` 0.512, `nll` 0.583,
probe `M` 0.491.)*

**Oracle surprisal reads exactly 0.500 in every column of every contrast** — the design pins it,
stratification is not doing the work.

Two things sit side by side here. The distinction is **linearly decodable from the post-token
residual stream at ~0.99** while the *prefix* state reads 0.55; and **no belief-derived readout
expresses it** — probe `M`, `Mpm`, prompted `M` and prefix entropy are all within noise of chance,
as is the model's own surprisal once matched. In the prompted arm, `M_prompt` reads 0.530 with its
own swap guard at 0.531, i.e. nothing.

`r_dir` (0.996) is not better than `h_after_dir` (0.993), and `r = h_after − f(prefix)`, so the
forward model is **inheriting** this signal rather than producing it.

### The guard that matters, and two that don't

`h_swap_dir` is the decisive control: identical fitting procedure, identical labels, identical
capacity, but the state vectors are permuted within (frame × split), so they carry the right
distribution and the wrong content. It reads **0.472–0.512** everywhere. The 0.99 is not
manufactured by the fitting.

The two label-shuffle guards are **not** trustworthy here, and this is worth not rediscovering:
permuting labels within strata reads 0.62–0.83, and permuting them globally still reads 0.52–0.62.
When the target feature is near-perfectly linearly decodable, the finite-sample residual correlation
a permutation leaves behind is enough to recruit the dominant direction — the guard degrades
precisely when the signal is strong. **Permute the inputs, not the labels.**

## The sign reversal — the sharpest result in the run

Splitting `negate_dead` by *how the value died*, with the AUC stratified inside each cell:

| dead value was… | n live/dead | `nll` | `h_after_dir` | `h_before_dir` | `h_swap_dir` | probe `M` |
|---|---|---|---|---|---|---|
| **named** by an earlier negation | 974 / 330 | **0.891** | 0.997 | 0.530 | 0.527 | 0.525 |
| killed **silently** by a `narrow` | 974 / 325 | **0.164** | 0.994 | 0.510 | 0.471 | 0.466 |

Both cells are `B = 0`. In the first, the non-revising token is the *least* surprising thing in the
story (it is a repeat). In the second it is the *most* surprising (a word the narrowing had ruled out
without ever naming). **Token surprisal does not merely fail to track revision here; its sign depends
on how the information arrived**, while the post-token state decodes the same distinction at
0.994–0.997 in both cells with the swap guard at chance.

Pooled, the two mechanisms partly cancel into a weak apparent positive (`nll` 0.640 raw). A corpus
with only the negation mechanism would have shown 0.96 and looked like a clean surprisal result; a
corpus with only the narrow mechanism would have shown a clean *anti*-surprisal result. Owning the
DGP is what makes both visible at once.

## Gate D — the variance budget of the forward model's target

Per family, on held-out stories, with `Δ_t = h[t] − h[t−1]` and `r = Δ_t − FM(h[<t])`:

| family | oracle `B` | mean `nll` | `H_out` | `E‖Δ‖²` | `E‖r‖²` |
|---|---|---|---|---|---|
| `echo` | 0 | 1.29 | 2.92 | 2705 | **307** |
| `deduced` | 0 | 2.41 | 3.05 | 2502 | 444 |
| `mention` | 0 | 6.55 | 3.30 | 2922 | 553 |
| `negate_live` | **0.39** | 6.11 | 3.74 | 3360 | 571 |
| `reveal_narrow` | **0.86** | 4.40 | 3.69 | 2979 | 639 |
| `negate_dead` | 0 | 3.61 | 2.98 | 2960 | 655 |
| `reveal` | **1.39** | 6.98 | 4.35 | 3523 | 747 |
| `synonym` | 0 | 6.61 | 4.37 | 2806 | 796 |
| `offtopic` | 0 | 6.90 | 6.21 | 4000 | **1499** |

Across the nine families, `E‖r‖²` is explained by

| by | R² |
|---|---|
| the reader's **output entropy** | **0.901** |
| model `nll` | 0.391 |
| the oracle's mean belief revision | **0.0001** |

`reveal` (`B` = log 4) sits *below* `synonym` (`B` = 0), and `negate_dead` (`B` = 0) sits *above*
`negate_live` (`B` > 0). On this substrate the temporal FM residual's **magnitude** is a measurement
of how unpredictable the token was and carries no information about how much the token moved the
posterior. A scalar precision applied to it would weight positions by exactly the quantity an
aleatoric null is meant to discount — not as a matter of tuning, but of what the object is. The
residual *direction* separates at 0.996, which is the third instance of this repo's
directional-not-scalar pattern and the one cell the idea doc's §5 calls empty.

## Gate A — the model's revision against the oracle's

Partial `R²` on positions where `B` is defined, with two nulls:

| readout | `R²(M ~ B \| nll)` | `R²(M ~ nll \| B)` | rank version |
|---|---|---|---|
| probe `M` | 0.137 | 0.057 | 0.104 |
| probe `Mpm` | 0.001 | 0.073 | 0.001 |
| prompted `M` | 0.033 | **0.316** | 0.061 |
| `r_norm` | 0.019 | 0.172 | 0.009 |
| **null: fully random** | **0.000** | | |
| **null: shuffled within frame** | **0.097** | | |

Against a fully random null the probe's 0.137 looks like a result; against the frame-preserving null
it is **1.4×**, where RHM's equivalent was 20–30× its shuffled control. The honest reading is that
`M` carries little oracle structure beyond the syntactic frame. `R²(B ~ nll)` = 0.081, so the two
references are well separated on this substrate — the failure is not a lack of headroom.

## The other contrasts, and why their state rows are not interpretable

`h_before_dir` is the diagnostic: if the *prefix* state predicts family membership, the contrast is
degenerate for any state-based readout.

- **B2** (`reveal` vs `synonym`/`offtopic`/`mention`) and **B2b** (`reveal` vs `mention`): oracle
  surprisal is *exactly* equal by construction, which is the purest form of the doc's one-liner —
  but the syntactic frame determines the family (the frame-matched column has **0 usable pairs**) and
  `h_before_dir` reads **1.000**. Probe `M` reads 0.79–0.85 here and this should not be read as
  revision-tracking.
- **B3** (`mention` vs `synonym`/`offtopic`, all three with `B = 0`) is the specificity guard, and it
  fires: probe `M` separates two *zero-revision* families at **0.621**, with its own swap guard at
  0.608 and prefix entropy at 0.639. So B2's number is content and frame, not revision. `h_swap_dir`
  is also 1.000 here — the swap is within frame, and when frame determines family the swap guard is
  degenerate too. It is only meaningful in B1.
- **E** (`deduced` vs `echo`, oracle `B` = 0 *and* oracle surprisal 0 on both sides) is
  prefix-degenerate rather than frame-degenerate: `deduced` requires a slot already narrowed to one
  value, so `h_before_dir` reads 0.995. `r_norm` = 0.781 and `nll` = 0.656 there are suggestive of
  the model's own inference gap — a value it *could* have deduced arriving as news — but they are not
  attributable on this contrast. Worth a purpose-built design; not evidence yet.

## What this establishes, and what it does not

**Establishes** (one reader, one regime):

- On a language substrate with an exact oracle, oracle surprisal can be pinned at AUC 0.500 by
  construction while belief revision varies — the reducible/irreducible split is realisable in English,
  not only on RHM.
- **Token surprisal's relationship to belief revision reverses sign** depending on whether the
  information arrived by naming or by elimination. This is stronger than "surprisal is a noisy proxy".
- The distinction is **linearly decodable from the post-token residual stream** (0.99) and not from
  the prefix state (0.55), at matched word, frame, live-set size, mention count and surprisal, with a
  swap-state control at chance.
- The **model's own belief readouts do not express what its state contains** — probe and prompted `M`,
  `Mpm` and prefix entropy all at chance in B1, and Gate A at 1.4× a frame-preserving null. This is
  [`MNIST_LOCAL_LOSS`](../MNIST_LOCAL_LOSS_README.md)'s *legibility is not self-knowledge* on a new
  axis.
- The temporal FM residual's **magnitude** tracks output entropy (R² 0.901) and not revision
  (R² 0.0001), which is a measured reason — rather than an assertion — that a scalar precision on this
  object cannot express an aleatoric null.
- The **ungrounded prompted readout beats the trained probe** at every slot (70–83% vs 53–67% of the
  exact Bayes ceiling), which answers RHM's open question about withdrawing the probe in the
  affirmative for the readout, if not for the result.

**Does not establish:**

- That the 0.99 is *self-knowledge*. It is a supervised decode trained on the oracle label, so it
  measures legibility. The model's own posterior is the thing that reads at chance.
- Anything about *using* any of this. Everything here is open-loop measurement against a frozen reader.
- That the belief-readout null is structural rather than a capacity limit of a 1B reader. The state
  contains the distinction and the readout does not; whether that gap closes with scale is the single
  most informative untested question, and it is RHM's *"belief depth is the binding constraint"*
  restated.
- The sign-reversal table is the result most worth replicating.
- Anything about naturalistic text. Chronicle is fluent but templated, and the frame degeneracy in
  B2/B2b/B3 is a direct consequence of that.

## Reproduction

```bash
cd experiments

# oracle self-tests -- local, CPU, no Modal, no tokenizer, ~30 s
python3 -m a2a_forward.conditional_revision.corpus

# the full run, ~30 min on an L4 the first time
modal run --detach -m a2a_forward.conditional_revision.gates::gates \
    --n-stories 3000 --n-clauses 14 --probe-steps 6000 --fm-steps 4000 \
    --n-prompt-stories 700 --prompt-batch 64 --disc-steps 1200 --tag v3
```

Results land on the **`reading-data`** volume (`chromatic` workspace) at
`/data/chronicle/gates_v3_seed42.json`, with reader activations and the prompted readout cached under
`/data/chronicle/cache/`. The cache makes analysis-only re-runs ~6 min; the prompted readout alone is
1390 s of the 1749 s cold run. The Modal app and volume keep the name the cut was built under
(`reading`) even though the node is now `conditional_revision`, because every result is keyed to it.

## Gotchas worth not rediscovering

- **Permute the inputs, not the labels.** Both label-shuffle guards read 0.52–0.83 against a feature
  that a swap-state control puts at 0.50. A near-perfectly decodable feature gets recruited by the
  residual correlation a permutation leaves behind.
- **`h_before_dir` is the contrast-validity check.** If the prefix state predicts the family, every
  state-based row in that contrast is reading the template. It reads 1.000 in B2/B2b/B3 and 0.995 in E.
- **The swap guard is only meaningful when frame does not determine family.** Swapping within frame
  cannot decouple a label that frame already fixes, so `h_swap_dir` reads 1.000 in B3.
- **Elimination must not always name the eliminated value**, or "dead" means "already mentioned" and
  an induction head solves the task. This is the single most important property of the substrate.
- **Matching strata must be atom-aware and finely binned.** Oracle surprisal is a single atom here by
  design; 8 quantile bins left the `nll` self-matched guard at 0.62, and 24 bins bring it to ~0.50.
- **`logits_to_keep=1` is load-bearing in the prompted readout**, not an optimisation: the full
  `(B, T, 128k)` logit tensor is 7.5 GB at batch 128 and OOMs an L4.
- **Cache files holding numpy arrays need `weights_only=False`** under torch ≥ 2.6.
