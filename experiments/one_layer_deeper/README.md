# One Layer Deeper — repeated modular squaring as a ballistic forward-model substrate

**Up**: [../CLAUDE.md](../CLAUDE.md) (experiments) · **Files**: [FILES.md](FILES.md)
**Upstream**: [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) —
an architecture-and-optimizer competition from Core Automation × Tilde Research.
**Status**: one cut complete. **Date**: 2026-08-02.

## What this is

The upstream benchmark asks for `y = x^(2^T) mod N` given `(N, x, T)`, with `N`'s factorisation
withheld — so without a shortcut the only route is `T` serial squarings. It fixes the data and
the outer loop and hands the participant the architecture, the optimizer, **and the training
loss**, with depth deliberately unconstrained.

We use it the way we use [RHM](../rhm/README.md) and [`mjc/`](../mjc/README.md): as a
**controllable DGP whose knobs we set and sweep**, not as a leaderboard to climb. No submission
has been built and none is required for the science. What the substrate buys that ours do not:

- **It is purely ballistic.** The model commits at step 0 and never observes an intermediate
  state. Every controller in `mjc/` needed commitment installed as a knob, because reactive
  control re-grounds each step and is therefore a *near-blind grader* of forward-model quality
  (4b: transmission slope **+0.35** reactive vs **+1.07** ballistic). Here exact-match at
  held-out depth is a fully sighted grader by construction.
- **Ground truth is exact and free at every step**, so latent veridicality is measurable against
  a known `x_t` — and the per-step error amplification of the true dynamics is known
  *analytically* (squaring is `a ↦ 2a` in the discrete-log coordinate: the doubling map,
  Lyapunov exponent `ln 2`). Our other substrates had to manufacture the FM-quality axis and
  grade it with proxies.
- **Depth is an unbounded, scored axis** — which is the composition-horizon question with a
  scoreboard attached.

**The DGP knob that governs everything**: `x^(2^T)` is eventually *periodic* in `T`, so a badly
chosen modulus lets a model pass "depth extrapolation" by discovering a cycle rather than
iterating. `squaring_mod.py` computes the margin (`tail + period`, verified by brute force over
all units) and asserts against it.

### How our knobs sit against the upstream tiers

The manifests under `benchmark/manifests/` encode their config in the `data_root` path — e.g.
`..._easy_bidirectional_fixed_n_323_t123` is fixed `N=323`, train `T ∈ {1,2,3}`. The tiers vary
along **three** axes, not one: `fixed_n` (one modulus, vary `T`), `fixed_t` (one `T`, vary the
modulus — the arity axis), and `variable` (both). The `bNNNN` field is the *bit width* of `N`.
Evaluation is a common ladder, `T = 1, 2, 4, 8, 16, 32, 64`.

| | modulus | bits | first depth-repeat | train `T` |
|---|---|---|---|---|
| Easy `e1` (fixed_n) | 323 = 17×19 | 9 | **10** | 1,2,3 |
| Medium `m1` (fixed_n) | 10403 = 101×103 | 14 | **42** | 4,8,16 |
| Easy `e5` (variable) | 10–11 bits | 10–11 | — | 1,2,3 |
| Medium `m5` (variable) | 12/14/16 bits | 12–16 | — | 2,4,8 |
| **ours** | **893 = 19×47** | **10** | **67** | 1..6 |
| ours, next rung | 9853 = 59×167 | 14 | 1149 | — |

Two things follow. **Our modulus sits at Easy's scale but is deliberately much deeper**: 10 bits,
same as Easy's variable range, with a periodicity margin of 67 against Easy's 10. And **the
upstream ladder's deep rungs are partly degenerate on its own moduli** — for `N=323` (tail 4,
period 6) the maps at `T=16`, `32`, `64` are *exactly* the maps at `T=4`, `8`, `4`, so nothing
past `T=8` requires further composition; for `N=10403` (tail 2, period 40) `T=64` reproduces
`T=24`. This is a property of the moduli, not a criticism of the benchmark — the tiers that carry
the real difficulty are the `fixed_t` and `variable` ones, where the modulus is the held-out axis
and depth is short.

## Cuts

### [`ballistic_depth/`](ballistic_depth/README.md) — what actually extends a learned operator's composition horizon

A tied recurrent operator trained on terminal cross-entropy alone (train `T ≤ 6`) has a
composition horizon of **T ≈ 13**; a non-recurrent untied stack of matched depth is perfect at
every trained depth and **at chance one step past it** (1.000 → 0.009) — `operators_not_footprints`
with a scoreboard. Adding one **label-free** constraint — *the state you rolled into must be one
your own encoder could have produced* — moves the horizon to **T ≈ 51** (3.9×) at its swept
optimum, perfect out to T=20 and 0.98 at T=30, and lifts held-out-x generalisation from
**0.001 to 0.32**.

The mechanism is **manifold closure, not error suppression**: the base model's rolled state is
nearly orthogonal to the encoder's representation of the same residue (cos **0.19**) from step 1
while decoding it perfectly — a private trajectory, not a closed operator — and the constraint
takes that to **0.99**, with closure tracking accuracy across a 10× depth range. This reproduces
[`RHM_SCULPTING`](../rhm/RHM_SCULPTING_README.md) Stage 5b's *rollability ≠ depth* dissociation on
an unrelated substrate.

A follow-up micro-cut sharpens the mechanism into a behavioural claim. **Neither operator is
broken at depth**: hand the base model the true residue `x₅₅` and it rolls the last five steps of
a T=60 problem at **0.873**, while its own rollout to the same target scores **0.000**. What fails
is the state, not the map — the entire composition-horizon failure is the rollout leaving its own
input domain. The tightest statement is at zero rollout steps, where base cannot decode its own
encoder's output (**0.008**) and the constrained arm can (**1.000**). Across 72 runs closure
orders the horizon at Spearman **+0.94** — including through a knob reversal where the loss weight
stops predicting the horizon and closure does not — but it does *not* transfer on a calibrated
scale, and `re-entry-only` buys depth without buying closure at all.

Acting on that diagnosis works. Snapping the rolled state back onto the encoder manifold
*at test time*, using the model's own decode and **no retraining**, takes the ungrounded model
from 0.000 to 0.377 at T=60 and the grounded one to **1.000 at every depth out to T=60** — past
where `N = 893` can measure, since it repeats in depth at T=67. The intervention turns one rollout
into a chain of restarts whose accuracy goes as `p^⌈T/k⌉`, so what the training-time constraint
buys is the per-restart rate `p`: 1.000 for the grounded arm, 0.871 for the ungrounded one, which
decays exponentially in depth no matter how the period is tuned.

The cut's *predicted* mechanism — that the rollout fails by amplifying its own error, so
discretising the state is mandatory — is **falsified five ways**, with the instrument calibrated
along an on-manifold direction before the null was read. Hard quantisation failed to learn the
task twice while demonstrably achieving the error correction it was built for (per-step
contraction 0.889, accuracy 0.048): *error correction, achieved and useless.* Two further
results: the label-reusing half of the consistency term is **dose-dependently harmful** above
w≈0.1 (horizon 34.7 → 31.0 → 22.3), inverting the dense/evaluative complementarity the cut was
designed around; and the compute confound is dead in the useful direction — base at 2.5× the
gradient steps is *worse*, reproducing
[`metering_sweep`](../rhm/directed_sculpting/full_loop/metering_sweep/README.md)'s within-round
overfitting on a new substrate.

## Shared machinery (lives at this node)

**`squaring_mod.py`** — the task, vendored from upstream. Token ids, decimal digit encoding,
field markers and the trapdoor label path (`pow(x, 2^T mod φ(N), N)`) are verbatim; the ~900
lines of split machinery are dropped. Two documented departures, both stated because they are
the kind of thing that silently confounds a depth experiment:

1. **It emits the full trajectory** `x_0 … x_T`. The evaluator never gives you this. It is
   *research-only instrumentation* — no arm trains on an intermediate residue — and it is what
   makes latent veridicality measurable per step.
2. **Fixed-width zero-padded answers**, so exact-match does not tangle answer *length* with
   answer *value*.

`TaskSpec.describe()` reports the depth-periodicity margin, the reachable-state count, and the
unit count; `build_trajectories` cross-checks columns against the independent trapdoor path.

**`shared.py`** — Modal app (`one-layer-deeper`), image, volume (`one-layer-deeper-data`),
`NumpyEncoder`.

## Reproduce

```bash
cd experiments/
MODAL_PROFILE=chromatic modal run --detach \
  one_layer_deeper/<cut>/<script>.py::<entrypoint> --tag <tag> --seed <n>
```

Use `--detach` for anything over ~2 minutes and invoke the function explicitly, not the local
entrypoint. Each cut's README carries its exact commands. Modal volume layout:

```
/ballistic_depth/<tag>/results_seed<N>.json
```

## Next steps

1. **Scale `N`.** Everything so far is `N = 893`. `N = 9853 = 59 × 167` (2407 reachable states,
   first depth-repeat 1149) is already characterised in `squaring_mod.py`.
2. **Held-out modulus** — the arity-2 cut. Fixed-`N` never forces the operator to be *conditioned
   on the rule*; a rule-blind operator can only predict the `N`-averaged next state. This is the
   axis that connects to the length-gen finalizer's *"width amplifies arity in an open loop"*.
3. ~~**Does closure predict the horizon across arms?**~~ — done; see `ballistic_depth/` §6–§7.
   Kept below for the original framing. Twelve-plus configurations are already run;
   closure-at-fixed-`t` against horizon would turn the mechanism claim from a two-arm contrast
   into a slope.
