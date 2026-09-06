# Aleatoric fraction: how much irreducible content is the state carrying, and how much of it is discardable?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Sibling**: [`../synonym_retention/README.md`](../synonym_retention/README.md) — the same question with a
distance axis, which is the axis this cut lacks
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) §8
**Date**: 2026-08-08 · **Status**: run. One regime (m4), one checkpoint, frozen — nothing is
trained here.

## Goal

[`local_loss/`](../local_loss/README.md) swapped the depth target for a temporal one and found the same
signature rather than a different one, but that comparison is hard to read: its own Finding 1 shows ~90%
of the local-loss term's dynamic range is the target shrinking in *scale*, a gauge every downstream
reader of `post_block6` discards. Both arms had that exit available, so the null does not clearly bear on
the conditioning gap.

This cut asks the prior question, with no training and no forward model: **is there anything for an
aleatoric filter to remove?** [`sculpt_slip/`](../sculpt_slip/README.md) is the cautionary case — it ran a
precision-operator intervention and only afterwards established that the prize it was chasing was ~0.03.

## The object

§4's identity splits surprisal by belief propagation:

```
H(x_{t+1}|x_≤t)  =  I(x_{t+1}; z_≤D | x_≤t)  +  H(x_{t+1} | z_≤D, x_≤t)
                     reducible                   irreducible  (= H_irr[D])
```

The same BP quantities give the law-of-total-variance split of the **state update**:

```
Var( h[t+1] | x_≤t )  =  Var_z( E[h[t+1] | z_≤D, x_≤t] )  +  E_z[ Var(h[t+1] | z_≤D, x_≤t) ]
                         EPISTEMIC (E_D)                     ALEATORIC (A_D)
```

Two properties make this exact and cheap rather than Monte-Carlo:

- **Conditional on the prefix, a causal model's `h[t+1]` takes only `v = 16` values**, one per possible
  arriving token. Both conditional variances are 16-term weighted sums over counterfactual activations.
- **The weights are already computed by [`../oracle.py`](../oracle.py) and discarded.** `H_tot` and
  `H_irr[D]` are the entropies of exactly the two distributions needed. A default-off
  `return_leaf_posteriors` kwarg emits them; prior callers are untouched and the self-test still passes.

An observed sequence's true latent assignment is itself a draw from `P(z | x_≤t)`, so averaging the inner
term over the dataset is an unbiased one-sample estimator of `E_z[·]` — no enumeration or sampling of `z`.
Substituting a token that makes the sequence an illegal production is harmless: BP gives it probability
zero and it drops out of both sums.

**The readout is `A_D/(A_D+E_D)`, which is scale-free** — the one thing the gauge collapse cannot
masquerade as.

`A_D` splits again into `A_prot` (the model is *forced* to carry it, because downstream next-token
prediction reads it) and `A_free` (droppable at zero NTP cost). `A_free/Total` is the headroom any
aleatoric filter could ever win. The split comes from an exact local criterion: two arriving tokens are
NTP-interchangeable iff `down_u ⊗ #{r : rule matches}` is proportional, where `down_u` is the root-side BP
message into the token's leaf-parent. **A first version of this criterion was wrong** — it ignored `down_u`
and over-separated — and the in-run check against direct BP caught it. That check is kept: same-class
tokens agree bit-identically on every latent node, clique and next-token distribution (max Δ = 0.0),
different-class tokens differ by ≥ 0.20.

## Instrument checks

Oracle identity; emitted-posterior consistency 3.3e-16; criterion-vs-BP 1.1e-16; local-criterion-vs-oracle
2.2e-16; substitution-vs-plain-forward 2.1e-04, i.e. fp32 noise at ~1e-7 relative. That last check caught a
block-stacking reshape that had scrambled the block and token axes.

## The number

Frozen m4 base, `post_block6`. n = 1500 sequences × 63 positions = 94,500 cells; 18.4% have only one legal
arriving token and drop out of numerator and denominator together. Bootstrap over sequences, 400 draws.

| `D` | level | `A/(A+E)` | boot 95% | sd | NTP floor | **prize** `A_free/Tot` | ceiling |
|---|---|---|---|---|---|---|---|
| 0 | d6 | 0.9821 | [0.9814, 0.9828] | 0.00036 | 0.9819 | **0.00784** | 0.0448 |
| 1 | d5 | 0.9726 | [0.9716, 0.9733] | 0.00043 | 0.9723 | 0.00785 | 0.0448 |
| 2 | d4 | 0.9538 | [0.9528, 0.9547] | 0.00052 | 0.9535 | 0.00784 | 0.0448 |
| 3 | d3 | 0.9141 | [0.9129, 0.9155] | 0.00065 | 0.9135 | 0.00784 | 0.0447 |
| 4 | d2 | 0.8288 | [0.8271, 0.8305] | 0.00087 | 0.8275 | 0.00780 | 0.0447 |
| 5 | d1 | 0.6651 | [0.6636, 0.6668] | 0.00081 | 0.6625 | 0.00779 | 0.0445 |

"Ceiling" is model-free: the prize an isotropic token-identity code would offer under the same equivalence
classes.

The aleatoric fraction is large and tracks `H_irr[D]` as it should — **the conditioning gap does give the
residual a genuine aleatoric component, and a big one.** Two thirds of the ideal arity-1 residual's
variance is irreducible even at the tightest reading. What the table also says is that **the measured
value sits on the floor at every rung** (gap 0.0002–0.0026), and the headroom is 0.0078 against a
substrate cap of 0.045. Estimator variance is not the limiting factor.

Note this measures the *Bayes-optimal* arity-1 residual: the best `p⁻` is `E[h[t+1] | x_≤t]`, so the
forecaster's own capacity shortfall `ε₁` is removed by construction. That sidesteps SPEC's Gate C rather
than answering it — capacity invariance was never swept, the FM was replaced by the exact posterior.

## The prize is `D`-invariant, which we did not expect

To three decimals across the whole ladder. The reason is structural: the NTP-equivalence classes do not
depend on `D`, so lowering `D` adds *between*-class variance to both `A` and `A_prot` and leaves `A_free`
untouched.

If that reading is right, the `D` ladder indexes **how much protected content you agree to call
irreducible**, not how much is discardable, and those are different quantities. §4's identity is exact at
every `ℓ` and this does not disturb it; what it bears on is the use of the ladder as a knob for the
*intervention* side. We have not tested this beyond observing the invariance.

## The floor is not zero, and it had to be measured

Re-realising a completed constituent from the same feature leaves the exact-Bayes next token **exactly**
unchanged in only 55–87% of cases, with mean TV 0.002–0.099 otherwise. The mechanism is parse ambiguity:
`generate_rules_distinct` draws each feature's `m` tuples independently across features, so the tuples are
not injective (56 distinct of 64 at every level) and two realisations of one feature leave different
posteriors *over that feature*. With more left context the ambiguity shrinks sharply — at d3, mean TV
0.022 → 0.003 when the constituent is second rather than first.

Split by what the arriving token does (`D = 5`):

| | positions | share of total var | `A/(A+E)` | floor | free/A | prize ceiling |
|---|---|---|---|---|---|---|
| **opens** a constituent | 31 | 0.811 | 0.8090 | 0.8077 | 0.008 | 0.016 |
| **completes** one | 32 | 0.189 | 0.0465 | **0.0338** | 0.283 | — |

At completing positions the floor is 0.034 rather than 0 — that is the parse ambiguity, measured. At lower
`D` the completing-position floor rises to 0.71 (`D=3`) and 0.94 (`D=0`), because there the compatible
tokens span different level-5 features.

**98.7% of the aleatoric variance sits at opening positions** (`H_irr[5]` = 1.306 nats there vs 0.108 at
completions) — which is where discarding is forbidden, since the synonym choice is what determines the
sibling. Stated on the time axis: on this substrate the interval between "this content is revealed" and
"this content stops mattering" is very short. Content stops being uncertain at close to the same moment it
stops being load-bearing. Whether that generalises past `s=2` is treated below.

The one place a sizeable local prize exists is high-level completions: arrival levels 0–2 have free/A of
0.94–0.97 and a local prize of 0.242, but that is 4 of 63 positions and 0.07% of total variance.

## A result we did not go looking for: the depth profile

`free/A` is scale-free within a block, so this comparison is legitimate where the raw prize is not:

| block | `A/(A+E)` | prize | `free/A` |
|---|---|---|---|
| `post_embed` | 0.6220 | 0.0433 | 0.0696 |
| `post_block0` | 0.5533 | 0.0342 | 0.0617 |
| `post_block2` | 0.4536 | 0.0224 | 0.0494 |
| `post_block4` | 0.5534 | 0.0129 | 0.0233 |
| `post_block6` | 0.6651 | 0.0078 | 0.0117 |
| `post_block7` | 0.6930 | 0.0067 | 0.0097 |

The embedding keeps 7.0% of its aleatoric budget as NTP-free content; block 6 keeps 1.2%. On this
checkpoint, **plain NTP discards ~83% of the discardable aleatoric content by block 6, with no local loss
anywhere.** This is close to tautological in direction — a sufficient statistic discards nuisance — but the
magnitude is what determines whether an added objective has room, and [`synonym_retention/`](../synonym_retention/README.md)
tests the same claim causally and on a distance axis.

Pooled `A/(A+E)` = 0.665 sits slightly *above* the token-identity reference (gini 0.630, embedding 0.622),
so `h6` carries arriving-token identity fully. That is consistent with the parent README's
next-token-sufficient-statistic finding rather than in tension with it.

## Subspace separability: negative, with one caveat

§5's open question asks whether a *directional* precision operator could express "I am unreliable in these
directions." That needs `A` and `E` to occupy separable subspaces.

| | k=4 | k=16 | k=64 |
|---|---|---|---|
| A's variance inside E's top-k | 0.232 | 0.849 | 0.981 |
| A's own top-k (**ceiling**) | 0.457 | 0.950 | 0.993 |
| random-subspace (**null**) | 0.016 | 0.062 | 0.250 |
| mean cos² principal angles | 0.186 | 0.881 | 0.723 |

At k=16, `E`'s directions capture 89% of what `A`'s own best directions capture, against a 6% null. The
ceiling-and-null framing is the load-bearing part; participation ratios (A 12.6, E 12.2 of 256) are
reported in the JSON but this repo has found that metric unreliable and nothing here rests on it.

The one hint of separation is k=4 — 0.232 against a 0.457 ceiling, about half. That is exactly the regime
[`residual_decomposition`](../../residual_decomposition/README.md) records β as unusable in (below ~100
directions, and RHM's `R_res_participation` is 7), so we did not build on it. **Open rather than closed.**

## Would any RHM variant give headroom? An analytic survey

A follow-up pass enumerated ~60 modifications this program has made to vanilla RHM — the partially
heterogeneous splices, hierarchical and canonical damage, sculpting, minting, root restriction, rule-set
multiplicity, partial observability, active query, OU rule drift, the channel constructions, and every
training-objective variant — and assessed each against nonzero `A_free`.

**This section is analysis, not measurement.** The one anchored piece is that `A_free` on RHM is governed
by tuple-space occupancy `m/v^(s−1)` (0.25 at v16/m4), which reproduces the 98.7%-at-openings number
analytically. The rest is reasoning from the DGP and should be treated accordingly.

The survey's conclusion is that no variant creates non-degenerate headroom, for a structural reason:
`A_free` is the non-injectivity of (arriving token) → (posterior over the future), and **on every RHM here
the leaf token *is* the level-0 feature** — the emission map is the identity. Restructuring the latent tree
cannot touch that. Three consequences worth recording:

- **Raising `s` strictly reduces `A_free`** — larger `s` makes prefix-sharing exponentially rarer under the
  same occupancy formula. `s = 2` is the best case for this quantity, not the worst.
- **Raising occupancy to buy headroom spends frontier quality.** At fixed `m=4`, moving occupancy
  0.0625 → 0.50 drops d4 recovery 0.93 → 0.43 and the BP root ceiling 0.96 → 0.40
  ([`RHM_FRONTIER_AND_LEGIBILITY`](../../RHM_FRONTIER_AND_LEGIBILITY_README.md) finding 2). Estimated
  ceiling from that route is ~0.03 — the prize size `sculpt_slip` already nulled on. `gini_free_reference`
  needs no model and no GPU, so a `(v, m, rule_kind)` sweep would settle this in minutes and has not been run.
- **Noise channels in [`../../rhm_channels.py`](../../rhm_channels.py) are the one construction with
  certainly-nonzero, ~10× headroom**, since nothing downstream reads those positions. That is the
  degenerate solution, and its value would be as a positive control — `sculpt_slip` could not separate "no
  prize" from "filter cannot find it," and a substrate with a known prize would. It has never met NTP:
  those functions are used only inside the sculpting control loop.

**A defect in the criterion, found by the survey and worth stating plainly.** The NTP-equivalence test is
*instantaneous* — it asks whether two tokens induce identical futures from the moment of arrival. Content
that is load-bearing for `w` steps and then exactly stops is fully protected at arrival and scores zero.
So `A_free` cannot register the shape most of interest, and the survey's most promising candidates
(interleaved structured channels, partial observability, active query) all satisfy the intuition and read
exactly 0 for this reason. Related: OU rule drift sends `A_free` to exactly 0 under an arbitrarily weak
dependence, so it is the ε=0 point of a rate–distortion curve `A(ε)` and the least robust point on it.
[`synonym_retention/`](../synonym_retention/README.md) exists to supply the missing axis.

## What this establishes, and what it does not

**Establishes** (one regime, one checkpoint):

- The temporal conditioning gap gives the residual a large genuine aleatoric component — 0.665 of the
  ideal arity-1 residual's variance at the tightest reading, against 0 by construction for the depth
  target. §1's claim about the residual's composition holds.
- On this checkpoint the measured aleatoric fraction sits on its NTP-protected floor at every rung, with
  headroom 0.0078 against a substrate cap of 0.045.
- The NTP-protected floor is nonzero even at constituent-completing positions (0.034), from parse
  ambiguity in `generate_rules_distinct`.
- `A` and `E` are not separable above k≈16 against a proper ceiling and null.

**Does not establish:**

- Anything about content that is load-bearing and *then* stops — the criterion is instantaneous, by
  construction. This is the largest scope limit and it motivated the sibling cut.
- That the estimate is robust to relaxing exact equivalence. An ε-relaxed criterion (merge near-equivalent
  tokens for small NTP cost) would grow both prize and ceiling, the vectors are cached, and the sweep was
  not run.
- Anything under a whitened metric. The readout uses the Euclidean trace in raw activation space,
  deliberately, because that is the metric `‖FM(h) − h_tgt‖²` uses; high-variance directions dominate.
- That the floor is exact for block 6 specifically — NTP could route a distinction through a lower block,
  so `A_prot` over-counts and the prize is a lower bound. The depth profile partly addresses this.
- Seed robustness, transfer off this regime, or anything about the m2 substrate `local_loss/` ran on.

Aligned eval sequences rather than the training distribution of random-offset windows, so position 0 is a
context the model rarely saw; every measurement in this folder shares that.

## Reproduction

```bash
cd experiments

# DGP-only checks + the criterion proof (local, CPU, ~2 min, no Modal, no model)
python3 -m rhm.conditional_revision.aleatoric_fraction.aleatoric_fraction
python3 -m rhm.conditional_revision.oracle          # still 2.2e-16 / 3.3e-15

# the run (~13 min on an L4; the base loads from cache and is never trained)
modal run --detach -m rhm.conditional_revision.aleatoric_fraction.aleatoric_fraction::aleatoric \
    --n-seq 1500 --n-swap 1500 --n-verify 8 --n-boot 400 \
    --chunk-gpu 64 --chunk-oracle 250 --tag af1
```

Output: `aleatoric_fraction_af1_seed42.json` under
`/data/v16_s2_L6_m4_distinct/conditional_revision/` on the `rhm-scaling-data` volume
(**`chromatic` workspace**).

## Gotchas worth not rediscovering

- **The NTP-equivalence criterion needs the root-side message.** A criterion built from rule-matching
  counts alone over-separates. The in-run check against direct BP is what caught it; keep it.
- **`A_free` is instantaneous.** It cannot see content that is load-bearing for a while and then stops, and
  it is the ε=0 point of a rate–distortion curve — an arbitrarily weak dependence sends it to exactly 0.
- **Check substituted forward passes against plain ones.** A block-stacking reshape scrambled the block and
  token axes and nothing else caught it.
- **The prize is `D`-invariant**, so there is no favourable rung to pick. Do not sweep `D` hoping for one.
- **Participation ratio is not the evidence** for subspace overlap. Use the ceiling-and-null comparison.
