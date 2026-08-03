# Variable modulus — the arity-2 version of the ballistic-depth cut

**Up**: [../README.md](../README.md) (one_layer_deeper) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 5 arms × 3 seeds, a re-projection cut, a coverage sweep, a 500k-step
grokking probe, and an arity/capacity sweep. **Date**: 2026-08-02.

---

## One-liner

[`ballistic_depth/`](../ballistic_depth/README.md) held `N = 893` fixed, so its tied operator
never had to be conditioned on the rule. Sampling `N` per example makes the true operator
arity-2. The headline number from that cut — a label-free closure constraint moving the
composition horizon **13 → 51** — does not reproduce here: the same constraint installs closure
just as completely (cos **0.29 → 0.97**) and moves the horizon **10 → 12**.

Measuring why produced the cut's main result, a **double dissociation between two independent
limits on composition depth**:

| knob | closure@1 | raw horizon | per-restart rate `p` | re-projected @T=28 |
|---|---|---|---|---|
| **cycle** off→on *(coverage 0.90)* | 0.29 → **0.97** | 10 → **12** | 0.894 → 0.895 | 0.613 → 0.632 |
| **state coverage** 0.50→0.98 *(`fold`)* | 0.26 → 0.28 | 9.9 → 9.7 | 0.47 → **0.98** | 0.048 → **0.908** |

Each knob moves one factor and leaves the other flat. Separately, **no arm ever learned modular
squaring**: held-out performance sits at or below the accuracy obtainable with no rule knowledge
at all, and 500k steps under grokking conditions did not move it.

Four of six pre-registered predictions failed. They are recorded as such below.

---

## The DGP, and a tension it forced

`ModulusFamily` (in [`../squaring_mod.py`](../squaring_mod.py)) builds a *family* of moduli under
three constraints, each load-bearing:

1. **Uniform digit width**, so answer width is constant and exact-match never tangles answer
   *length* with answer *value* — the same reason the fixed-`N` cut zero-pads.
2. **`depth_first_repeat > eval range` for every member.** `x^(2^T)` is eventually periodic in
   `T`; one leaky modulus would let the model pass depth extrapolation on those examples by
   discovering a cycle rather than iterating.
3. **`min(p, q) >= min_factor`**, excluding semiprimes with a tiny factor whose unit group is
   degenerate.

**The tension.** Constraints (1) and (2) fight each other at small scale. Only **21** 3-digit
moduli have a margin above 20, which is why the fixed-`N` cut sat at `N = 893` (margin 67) — it
spent its entire state budget on one rule to buy a 60-deep measurable range. Widening to 4 digits
gives 325 moduli at margin > 60 but a much harder one-step map. **Cut1's family** is the
compromise: 3-digit, margin > 25 → **33 moduli (26 train / 7 held-out)**, `N ∈ [177, 995]`,
minimum first-repeat **29**, so eval runs to **T = 28**.

**`reachable_states` is the quantity that matters for learnability.** At depth ≥ 1 the state
collapses into each modulus's quadratic-residue subgroup (`φ(N)/4`, since squaring is 4-to-1 on
the units of a product of two odd primes), and those subgroups are disjoint across moduli. So the
tied operator must realise `n_train_moduli` separate permutations totalling **3520** points,
against **207** in the fixed-`N` cut. It is logged at startup.

## Design

One encoder / one tied operator / one decoder, as in `ballistic_depth`. Arms differ in **where
the rule lives**, not whether the model has it:

| arm | encoder sees `N` | operator gets `r` | cycle |
|---|---|---|---|
| `blind` | ✗ | ✗ | ✗ |
| `fold` | ✓ | ✗ (rule *carried* in the state) | ✗ |
| `cond` | ✓ | ✓ (rule re-injected every step) | ✗ |
| `fold_cyc` / `cond_cyc` | ✓ | ✗ / ✓ | ✓ (w=3) |

`r` comes from a small **rule encoder over `N`'s digits alone**, deliberately not a readout from
the prompt encoder — a bidirectional encoder mixes every field, so a prompt-derived rule vector
would also carry `x_0`, and re-injecting it each step would leak the initial state into the
rollout. Digit-based also makes held-out `N` conceivable at all.

Train `T ∈ 1..6`, eval `T ∈ 1..28`, 60k steps, `d_model` 256, operator `d_ff` 8192. Cycle weight
3.0 and re-entry 0.0 are **transferred from `ballistic_depth` §4/§7, not re-swept here.**
Three OOD axes are reported throughout: held-out depth, held-out `x`, held-out `N`.

**New instruments** beyond the prior cut: `inrange@t` (fraction of decoded `x_t` that are `< N` —
does the state still know its own world) and `ruleswap@T` (roll `cond` with a different modulus's
rule vector — a causal check replacing a parameter-matched control).

---

## 1. Pre-registered predictions

| | prediction | outcome |
|---|---|---|
| **P1** | `blind` floors at every depth | ✅ ID 0.070, horizon 1 |
| **P2** | `cond` horizon > `fold` | ⚠️ real but small: **12 vs 10** |
| **P3** | cycle helps `fold` more than `cond` (sub-additive) | ✅ `fold` 10→12, `cond` 12→12 |
| **P4** | rule retention decays for `fold`, flat for `cond` | ❌ instrument confounded — see §5 |
| **P5** | held-out `N` — open | ❌ no generalisation at all (§3) |
| **P6** | re-projection helps `fold` more than `cond` | ❌ identical: 0.613 vs 0.610 |

Main table, 3 seeds, train `T ≤ 6`, eval to `T = 28`:

| arm | horizon (per seed) | ID | OOD (T>6) | closure@1 | held-out `x` | held-out `N` |
|---|---|---|---|---|---|---|
| `blind` | 1, 1, 1 | 0.070 | 0.005 | 0.214 | 0.020 | 0.015 |
| `fold` | 10, 10, 10 | 1.000 | 0.158 | 0.293 | 0.017 | 0.015 |
| `cond` | 11, 12, 12 | 1.000 | 0.222 | 0.154 | 0.025 | 0.012 |
| `fold + cycle` | 12, 12, 11 | 1.000 | 0.218 | **0.969** | 0.025 | 0.013 |
| `cond + cycle` | 12, 12, 12 | 1.000 | 0.251 | **0.966** | 0.041 | 0.005 |

**P1.** `blind` cannot do the task, confirming the rule is load-bearing. Its 0.070 is not noise:
for `x < √N`, `x² < N` and no modular reduction happens, so those cases are answerable without
the rule. The analytic no-reduction fraction for this family is **0.041**; `blind` sits somewhat
above it, plausibly because `x < N` is itself weak evidence about `N`.

**P6 is the informative failure.** Re-projection was expected to help `fold` more, because the
snap `h ← Enc(N, x̂)` re-supplies the *rule* as well as the state, and `cond` already has the
rule. It helps both identically (§4). The reading we take: **a static rule is cheap to carry** —
`fold` holds `N` in the rolled state for 28 steps without meaningful loss. This is a genuine
disanalogy with [`mjc/arity_torque`](../../mjc/arity_torque/README.md) and
[`RHM_SCULPTING`](../../rhm/RHM_SCULPTING_README.md) Round 3, where the *command changes every
step* and arity was decisive. Whether that distinction is the operative one is untested here.

**The rule-swap control is unambiguous**: `cond` goes 1.000 → **0.016** when rolled with a
different modulus's rule vector (`cond_cyc`: 1.000 → 0.011). The operator is fully conditioned on
`r`; there is no spare-capacity explanation for `cond`'s advantage.

## 2. Two independent limits on depth (the main result)

The cycle term reproduces the fixed-`N` mechanism *mechanically* — closure 0.29 → 0.969, and the
§8 signature holds: at zero rollout steps and T=20, `fold` scores 0.022 (cannot decode its own
encoder's output) while `fold_cyc` scores 0.882. But the horizon moves only 10 → 12.

Test-time re-projection (`h ← Enc(N, decode(h))` every `k` steps, no retraining) separates why.
3 seeds; the `k=0` condition is asserted against each run's stored rollout before anything else
is read.

| arm | none @T=10 | none @T=28 | self k=5 @T=28 | **oracle k=5 @T=28** |
|---|---|---|---|---|
| `fold` | 0.399 ±.032 | 0.014 | 0.613 ±.011 | **0.894 ±.002** |
| `cond` | 0.763 ±.034 | 0.017 | 0.610 ±.002 | **0.894 ±.002** |
| `fold + cycle` | 0.773 ±.072 | 0.004 | 0.632 ±.001 | **0.895 ±.001** |
| `cond + cycle` | 0.860 ±.026 | 0.007 | 0.646 ±.005 | **0.903 ±.003** |

**The oracle ceiling is 0.894–0.903 across arms whose closure spans 0.15 → 0.97.** An oracle snap
*is* a cold start, so this measures the per-restart rate `p` directly, and it is insensitive to
the intervention that moves closure 6×.

The coverage sweep manipulates `p` instead of inferring it, by varying what fraction of each
modulus's units are training bases (everything else fixed, 1 seed):

| coverage | `fold` horizon | `fold_cyc` horizon | closure@1 (`fold`) | **oracle `p`** | self @T=28 |
|---|---|---|---|---|---|
| 0.98 | 9.7 | 10.9 | 0.282 | **0.982** | 0.908 |
| 0.90 | 9.5 | 11.1 | 0.304 | **0.894** | 0.613 |
| 0.70 | 10.1 | 10.7 | 0.287 | — | — |
| 0.50 | 9.9 | 11.9 | 0.258 | **0.473** | 0.048 |

**`p ≈ coverage`, close to the identity.** The chain model `p^⌈T/k⌉` from `ballistic_depth` §9
holds at every point — `0.982⁶ = 0.897` vs 0.908 observed, `0.894⁶ = 0.510` vs 0.613,
`0.473⁶ = 0.011` vs 0.048 — with the same systematic slight under-prediction §9 reported.

So the two limits are governed by different variables and can be moved independently:

- **The raw rollout is drift-limited.** How far the model gets unaided (~10) is flat across a 2×
  swing in coverage and a 19× swing in re-projected depth.
- **The re-projected rollout is coverage-limited.** Once snapped back on-manifold every `k`
  steps, depth is `p^⌈T/k⌉` and `p` tracks state coverage.

This also gives an account of the fixed-`N` contrast that is consistent with both cuts, though
not tested by manipulating the fixed-`N` setup: its cycle arm reached `p = 1.000` on a 207-state
set, where `1.000ⁿ` is flat in depth; ours sits at 0.894 on 3520 states, where `0.894ⁿ` decays.

**Re-projection buys depth, never knowledge.** Held-out `N` under re-projection: 0.008 → 0.007,
0.006 → 0.005, 0.006 → 0.006, 0.002 → 0.001. Zero, every arm and seed.

## 3. No arm learned modular squaring

Held-out `x` and held-out `N` sit at **0.005–0.041** in every configuration, against an analytic
no-reduction floor of **0.041** for this family — i.e. at or *below* the accuracy obtainable with
no rule knowledge whatsoever. Held-out `N` is never worse than held-out `x`: a new modulus is no
harder than a new base, and both are simply unmemorised.

A dedicated grokking probe tested whether this is a training-length artifact — 8 moduli, **50% of
bases held out** so lookup cannot cover the test set, constant LR, **500k steps** (10× past
memorisation), at `wd ∈ {0.1, 1.0}` and both depth-1-only and depth-1..6:

| | train | held-out `x` | held-out `N` |
|---|---|---|---|
| step 25k → 500k, all cells | 1.000 throughout | 0.027 – 0.051, no trend | 0.023 – 0.055, no trend |

Analytic floor for that family: **0.0430**; observed means **0.038** (`x`) and **0.042** (`N`).
**No phase transition, at any setting.** Within the scales tested, the model represents these as
per-modulus lookup tables. This bounds what depth results on this substrate can mean — including
the fixed-`N` cut's 1.000 at T=60, which was likewise measured on seen bases.

## 4. Replications of `ballistic_depth`

Three of the prior cut's results reproduce unchanged on a different DGP:

- **The oracle plateau is flat in depth** (0.898 / 0.898 / 0.896 at T=14/20/28), independently
  re-deriving §8's finding that the operator computes correctly at every depth and the failure is
  the rollout leaving its own input domain.
- **The chain model `p^⌈T/k⌉`** tracks, under-predicting by a near-constant ~0.09.
- **The snap must be complete.** At α=0.5 every arm collapses at depth — `fold` scores 0.068 at
  T=20 against α=1.0's 0.749. A half-snap leaves the state between manifolds.

## 5. Instruments and claims retracted

Recorded because both were reported before being checked properly.

**`rule_separation` is confounded and should not be quoted.** It measures
`cos(h_t(N), h_t(N'))` for the same `x_0`. The cycle arms read ~0.90 (apparently "worse rule
separation") while scoring better, because pinning states to the encoder manifold puts them where
the *encoder* separates moduli weakly. It measures which manifold the state is on, not whether
the rule survived. `inrange@t` is the clean readout and does show `fold` losing its world fastest
(0.910 of decodes in-range at t=20 vs `cond`'s 0.987). P4 is unresolved, not answered.

**"More moduli → shorter horizon" was retracted.** An early reading of a fixed depth (T=10)
across an arity sweep showed 0.874 → 0.179 and looked like a large monotone effect. Converted to
horizons it is not monotone and not outside noise:

| train moduli | 8 | 8 | 16 | 16 | 24 | 47 |
|---|---|---|---|---|---|---|
| operator `d_ff` | 1024 | 8192 | 4096 | 8192 | 8192 | 8192 |
| reachable states | 857 | 857 | 1918 | 1918 | 3121 | 5903 |
| **horizon** | 11.1 | 9.9 | 11.2 | 11.6 | 10.6 | 8.6 |

8.6–11.6 across a 7× range in state count and 8× in operator width, single seed, against the
fixed-`N` cut's documented ±2 nondeterminism. **T=10 sat on the steep part of a sigmoid**, where
a ~3-depth shift reads as a 0.7 accuracy swing. The horizon metric exists to prevent exactly
this. What is supported is *invariance*, not decline — and more width made the 8-modulus case
mildly worse (11.1 → 9.9), the same direction as `ballistic_depth`'s iso-compute control.

## Honest caveats

- **The coverage sweep is single-seed**, as is the arity sweep. `cut1` and the re-projection cut
  are 3 seeds each, with tight spreads (`fold` horizon 10/10/10).
- **The cycle weight was transferred, not re-swept.** §7 of the prior cut found the fixed-`N`
  optimum at w≈10 with a collapse by w=30; we used w=3 on a different DGP. Whether the small
  10→12 gain would grow at a re-swept weight is untested. Given §2's finding that the cycle term
  does not move `p`, we would not expect it to change the ceiling, but that is an inference.
- **4-digit moduli were abandoned after failing at 15k steps**, before the 60k-step budget that
  makes 3-digit work was known. They may well be learnable; this was not re-tested, so "3-digit
  is required" is *not* established — only that 3-digit sufficed.
- **`p ≈ coverage` is measured at three points on one axis**, one seed each, all at 26 moduli.
- **Nothing here tests the actual benchmark.** No submission was built, `T` is deliberately
  withheld from the recurrent arms' prompt (upstream always supplies it), and the answer encoding
  is fixed-width — so "depth extrapolation" here is a strictly harder object than upstream's.

## Reproduce

```bash
cd experiments/
# main cut: 5 arms, 3 seeds, eval to T=28
for s in 0 1 2; do
  MODAL_PROFILE=chromatic modal run --detach \
    one_layer_deeper/variable_modulus/variable_modulus.py::variable_modulus \
    --tag cut1 --arms "blind,fold,cond,fold_cyc,cond_cyc" --steps 60000 \
    --eval-max-depth 28 --eval-cap 1024 --n-digits 3 --min-margin 25 --min-factor 3 \
    --d-ff 8192 --consist-cycle 3.0 --consist-reentry 0.0 --seed $s --save-ckpt
done

# test-time re-projection over cut1's checkpoints (trains nothing)
for s in 0 1 2; do
  modal run --detach one_layer_deeper/variable_modulus/reprojection.py::reprojection \
    --tag cut1 --arms "fold,cond,fold_cyc,cond_cyc" --seed $s --eval-cap 1024
done

# coverage sweep — the p manipulation (cut1 supplies the 0.90 point)
for tf in 0.02 0.30 0.50; do
  modal run --detach ...::variable_modulus --tag cov$tf --arms "fold,fold_cyc" \
    --steps 60000 --eval-max-depth 28 --n-digits 3 --min-margin 25 --min-factor 3 \
    --d-ff 8192 --test-x-fraction $tf --seed 0 --save-ckpt
done
for t in cov0.02 cov0.50; do
  modal run ...::reprojection --tag $t --arms "fold,fold_cyc" --seed 0
done

# grokking probe — 500k steps, half the bases held out, constant LR
for wd in 1.0 0.1; do
  modal run --detach ...::variable_modulus --tag grok_d1_wd$wd --arms fold \
    --steps 500000 --batch-size 128 --n-digits 3 --min-margin 20 --min-factor 3 \
    --max-moduli 8 --d-ff 1024 --train-depths "1" --weight-decay $wd \
    --test-x-fraction 0.5 --const-lr --seed 0
done

# arity / capacity sweep
modal run --detach ...::variable_modulus --tag arity_m16 --arms fold --steps 60000 \
  --eval-max-depth 20 --n-digits 3 --min-margin 20 --min-factor 3 --max-moduli 16 --d-ff 8192

# fetch + analyse
MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
  /variable_modulus/<tag>/results_seed<N>.json \
  one_layer_deeper/variable_modulus/results/<tag>/results_seed<N>.json --force
python3 one_layer_deeper/variable_modulus/analyze.py --tag cut1
```

Modal volume `one-layer-deeper-data`, results at `/variable_modulus/<tag>/results_seed<N>.json`,
checkpoints at `/variable_modulus/<tag>/ckpt/<arm>_seed<N>.pt`.

**Gotcha.** `modal volume get` without `--force` silently skips an existing local file, and
downloading a directory into a non-existent local path writes a *file* rather than a tree. Both
produced stale/partial local results during this cut. Always pass `--force` and a per-tag path.

## Next steps

1. **Raise `p` and `closure` together.** §2 moved each alone. Nothing here tests a run with high
   coverage *and* the cycle term at a re-swept weight, which is where the fixed-`N` result would
   predict the horizon finally moves.
2. **Seeds on the coverage sweep**, which currently carries the cut's central manipulation on n=1.
3. **Re-test 4-digit moduli at 60k steps.** The abandonment was made on a budget now known to be
   too small.
4. **A substrate where the rule changes per step.** P6's failure suggests the arity axis needs a
   *non-static* rule to bite, which is the condition under which `mjc`/RHM found it decisive.
