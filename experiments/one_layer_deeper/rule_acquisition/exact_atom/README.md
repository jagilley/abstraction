# Exact atom — how far the one-step map is from being *exactly* right

**Up**: [../README.md](../README.md) (rule_acquisition) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 27 arms across 5 cuts. **Date**: 2026-08-05.

---

## One-liner

The parent cut measured the one-step map against an *accuracy* target. This node re-reads the
same object against an **exactness** target, because that is what the benchmark's Hard tier
gates on, and because [`ballistic_depth/`](../../ballistic_depth/README.md) §9 has already made
depth the cheap axis. Under test-time re-projection at `k=1`, certifying ladder rung `T` needs
one-step error `eps <~ 9e-4 / T`, so **each rung is worth one factor of two in `eps`** and the
whole ladder `T = 1..64` is 64×.

Three things move under that reading:

1. **The multiply does not generalise** — 0.068 at 3 digits on a clean split, against the
   parent's reported 0.920. The parent split unit groups *per modulus* while `x -> x^2` does
   not depend on `N`, so a held-out `x` under one modulus was a trained pair under another.
2. **The reduce does**, and now measured from inside the composed task rather than in
   isolation: fed the true `Enc(DIV, N, x^2)`, the operator and decoder return `x^2 mod N` on
   **held-out `x` at 0.953**. The parent's `redmod` read 0.022 on those same inputs. This
   converts its §3 "coverage, not reachability" inference into a measurement.
3. **The multiply has a scaling path.** At 5 digits with 27.4M parameters it reaches **0.909**
   held-out on an exhaustive 50k pool, still falling at a tail slope of −3.70.

No cell in this node certifies rung `T=1`. Two of our own pre-registered claims failed and are
recorded as such, including a mechanism this node was built to confirm.

---

## Why this node exists, and the arithmetic that scopes it

[`ballistic_depth/`](../../ballistic_depth/README.md) §9 showed a depth-`T` rollout can be
turned into `T` independent depth-1 problems at test time, with no retraining, giving
`accuracy(T) ≈ p^⌈T/k⌉`; [`variable_modulus/`](../../variable_modulus/README.md) §2 measured
`p ≈ coverage` at three points. Since `k` is free, setting `k=1` collapses the depth axis onto
a single number — the one-step rate on the states the rollout actually visits.

Hard's ladder advances only when *every* example at the current `T` is correct, and the
leaderboard's 1/768 granularity fixes the count. Writing `eps = 1 - p`:

`P(certify T) ≈ (1-eps)^(768·T) ≥ ~0.5` ⟺ **`eps ≲ 9.0e-4 / T`**

| rung | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| tolerable `eps` | 9.0e-4 | 4.5e-4 | 2.3e-4 | 1.1e-4 | 5.6e-5 | 2.8e-5 | **1.4e-5** |

`certifiable_T(eps) = ln2 / (768·eps)` is printed next to every number below, so an accuracy is
never quoted without its consequence for the ladder. Two caveats on the formula itself: it
assumes per-step errors are independent across a trajectory, which is false in detail because
the error set is state-indexed rather than random; and it inherits `ballistic_depth`'s
re-projection result, which was measured at fixed `N` on seen bases.

**The instrument.** Every readout is exact-match error on a held-out pool that is
**exhaustively enumerated** wherever the problem space allows, logged ~24 times during training
so the shape of `eps` vs budget is visible rather than inferred from endpoints. Pool size and
its resolution `1/n` are reported with every number — a claim about `eps ~ 1e-5` cannot be made
on the 4096-example pool the parent cut used. Held-out splits are by an arbitrary hash of the
*problem*, so dense sampling can never make the held-out half leak.

---

## 1. The multiply is memorised, and the parent's split was contaminated

All arms 300k steps, `x` sampled uniformly over `[0, 10^w)`, split on `x` alone, held-out pool
exhaustive.

| arm | space | train inputs | seen `x` | held-out `x` | tail slope |
|---|---|---|---|---|---|
| `mul3` | 10^3 | 500 | **1.0000** (0 err) | 0.068 | −0.03 |
| `mul4` | 10^4 | 5,000 | **1.0000** (0 err) | 0.109 | −0.02 |
| `mul5` | 10^5 | 50,000 | 0.913 | **0.614** | **−1.44** |
| `mul5_sparse` | 10^5 | 500 | **1.0000** (0 err) | **0.00003** (3/99,499) | 0.00 |

`mul5_sparse` is the control that separates the two explanations: same ~500 training inputs as
`mul3`, 100× larger space, and transfer collapses from 0.068 to essentially nothing. So
`mul3`'s 0.068 is mostly the small space, not generalisation.

**Against the parent's `sqnomod` at 0.920.** That arm drew `x` from the union of eight moduli's
unit groups and split by `rng.permutation(u.size)` *inside* the per-modulus loop. Its target
`x^2` does not depend on `N`, so an `x` held out under modulus A was a training input under
modulus B with probability `~1 - 0.5^k` for the `k` unit groups containing it. The `(x, x^2)`
pair was therefore usually seen. §4's `sqpad_div_seamcos_gx` reruns the composed arm under the
clean split and reproduces the same picture from the other direction, so this is not an
artefact of also having changed the sampling distribution.

This does not touch the parent's `redmod_q*` numbers, which split on `(N, y)` — a key the
target does depend on. It does mean the parent's §2 decomposition ("both halves generalise on
their own") holds for the reduce half and not the multiply half.

## 2. The memorisation-boundary prediction is falsified; what is left is scaling

§1's one mechanism candidate — *the only arm that generalised was the only arm that failed to
memorise* — was pre-registered here as a prediction that inverts the parent's entire remedy
list: more capacity should make transfer **worse**, by putting the training set back inside
tabling range. All at `w=5` (space 10^5), matched 300k steps, `mul5` as the shared centre.

| arm | params | train inputs | seen `eps` | memorised? | held-out acc | slope |
|---|---|---|---|---|---|---|
| `memb_d128` | 0.55M | 50,000 | 1.40e-1 | no | 0.091 | −0.08 |
| `mul5` | 2.1M | 50,000 | 8.66e-2 | no | 0.614 | −1.44 |
| `memb_d512` | 8.5M | 50,000 | 3.12e-3 | no | 0.468 | −0.66 |
| **`memb_d512L8`** | **27.4M** | 50,000 | 3.10e-3 | no | **0.909** | **−3.70** |
| `mul5_sparse` | 2.1M | 500 | 0 | **yes** | 0.000 | 0.00 |
| `memb_n5k` | 2.1M | 5,000 | 1.96e-1 | no | 0.001 | 0.00 |
| `memb_n20k` | 2.1M | 20,000 | 0 | **yes** | 0.026 | −0.01 |

**The prediction fails in both directions.** Capacity buys transfer rather than costing it
(0.614 → 0.909 from 2.1M to 27.4M), and `memb_n5k` failed to memorise while generalising at
0.001 — so memorisation-infeasibility is not sufficient either. The reading in §1 that `mul5`
was special *because* tabling was unavailable does not survive its own test.

What the grid supports instead is the ordinary one: **training-set size dominates, with a sharp
threshold, and capacity multiplies it.** At fixed 2.1M parameters, 500 → 0.000, 5k → 0.001,
20k → 0.026, 50k → 0.614. Nothing in the 50k row is converged — every slope is still steeply
negative, which is
[`horizon_convergence/`](../../ballistic_depth/horizon_convergence/README.md)'s lesson
recurring — so the capacity *ordering* at 300k (2.1M above 8.5M) is a budget artefact and
should not be read as non-monotonicity.

`memb_d512L8` is the strongest generalisation result anywhere in this program: `eps = 9.1e-2`
on exhaustive held-out 5-digit multiplication, `certifiable_T = 0.010`. Extrapolating its tail
slope — optimistically, since it assumes the power law continues over two decades — puts T=1 at
~1.1e6 steps and T=64 at ~3.4e6, i.e. 3.7× and 11× the budget it has had. `mul5` at slope −1.44
needs 1.9e7 for the same rung, so the slope is doing most of the work in that estimate and it
rests on a single run.

## 3. The reduce: where `eps` closes with budget and where it does not

600k steps, `eps` logged 24×. The modulus-uniform column is a checkpoint re-evaluation
(`reprobe.py`) that matches training's sampling and the parent cut's; the training-time pool
enumerates every `(N, y)` pair and so weights a modulus by its own space size, which at
`q_cap=inf` is `N^2` and spans 32× across a 3-digit family.

| arm | seen `y` | held-out `y` | modulus-uniform | per-modulus spread | held-out `N` | slope |
|---|---|---|---|---|---|---|
| `divq8` | 2.0e-3 | 5.2e-3 | 4.7e-3 | 0 – 9.2e-3 | 0.909 | **+0.09** |
| `divq64` | 1.7e-3 | **2.4e-3** | 2.4e-3 | 3.4e-4 – 5.6e-3 | 0.988 | **−1.45** |
| `divqfull` | 4.9e-1 | 5.5e-1 | 5.2e-1 | **0.025 – 0.998** | 0.999 | −0.20 |
| `divqfull_densen` | 9.8e-1 | 9.8e-1 | 9.7e-1 | 0.80 – 0.998 | 0.992 | — |

`divq64` is the best cell in the node: `eps = 2.4e-3`, `certifiable_T = 0.38`, still falling,
extrapolating to T=1's 9e-4 at ~1.1M steps. `divq8` has strictly *more* budget per unit of
problem space and its error **rose** across the last half of training (slope +0.09) — the same
over-training penalty `ballistic_depth` gotcha 4 and
[`metering_sweep`](../../../rhm/directed_sculpting/full_loop/metering_sweep/README.md) record,
now on a third substrate.

`divqfull`'s per-modulus spread is the informative cell: 0.025 for the smallest modulus and
0.998 for the largest, so it learned division for some members of the same family and not
others, at equal sample redundancy. That is a difficulty gradient in the quotient range rather
than a coverage gradient, and §5 turns it into a clean statement.

**Two results here are not clean.** `divqfull` does **not** replicate the parent's
`redmod_qfull` (0.773 held-out at 150k; ours reads 0.482 modulus-uniform at 600k, and was
already worse at the 150k mark, so over-training alone does not account for it). The parent flagged that arm as unconverged; we have not identified the
difference. And `divqfull_densen` — the arm meant to attack the **rule** axis by exploiting the
fact that division needs neither a factorisation nor a periodicity margin, so `N` can be drawn
from all 900 three-digit integers instead of ~178 semiprimes — **never trained** (`ce_div` 1.64
at the end). Widening the modulus set made the task harder rather than the rule axis easier.
That lever is untested, not falsified.

## 4. The composed atom, and a seam constraint that had to be fixed

The parent's next-step 2, built as a 2×2. A dense division auxiliary alone should do nothing,
because nothing forces the `sq` path to route through the division circuit; the intervention
that forces it is `ballistic_depth`'s closure constraint applied at the multiply/reduce **seam**
— require `Enc(SQ, N, x) ≈ Enc(DIV, N, x^2)`. Both terms need only `(N, x)`, no trapdoor, so
the construction is legal for a real submission. 300k steps, 3-digit, floor 0.039 on held-out
`x`.

| arm | seam loss | cos seen `x` | cos held-out `x` | composed held-out | **oracle reduce, held-out** |
|---|---|---|---|---|---|
| `sqpad` | — | — | — | 0.0251 | — |
| `sqpad_div` | — | 0.013 | 0.010 | 0.0233 | 0.877 |
| `sqpad_seam` | mse | 0.181 | 0.193 | 0.0239 | 0.004 |
| `sqpad_div_seam` | mse | 0.099 | 0.081 | 0.0193 | **0.939** |
| `sqpad_seamcos` | cos | 0.9998 | 0.9997 | 0.0210 | 0.004 |
| **`sqpad_div_seamcos`** | cos | **0.972** | **0.642** | 0.0035 | **0.953** |
| `sqpad_div_seamcos_w10` | cos | 0.989 | 0.686 | 0.0053 | 0.927 |
| `sqpad_div_seamcos_gx` | cos | 0.970 | 0.625 | 0.0018 | 0.851 |

**The `mse` arms did not install the constraint.** MSE is not scale-free and the encoder's
final LayerNorm gain is shared by both branches, so shrinking it drives the loss towards zero
without aligning anything, and every downstream module re-normalises so nothing pushes back.
Measured: seam MSE **0.0010 at cosine 0.081**. The `mse` arms are kept verbatim rather than
edited away, because "a closure term reported as installed on the strength of a
scale-dependent loss" is the kind of thing worth having on the record. `1 - cos` installs it:
0.013 → **0.972** on seen `x`, comparable to `ballistic_depth`'s cycle term at 0.19 → 0.99.

**`sqpad_seamcos` is the control that makes the rest readable.** With no division auxiliary the
target branch is unanchored, so the encoder satisfies the seam trivially — cos **0.9997 even on
held-out `x`** — while its oracle reads 0.004. A high seam cosine means nothing unless the
target branch is independently anchored. This is `ballistic_depth` §10's *closure is only
interpretable conditional on in-distribution competence* in a new form, and the general version
is worth carrying: **a closure statistic is only interpretable conditional on its target being
independently grounded.**

**The oracle probe is the cut's main measurement.** `reprobe.py` feeds the operator and decoder
the *true* `Enc(DIV, N, x^2)` — `ballistic_depth` §8's cold-start probe moved from the rollout
seam to the multiply/reduce seam. On held-out `x`, `sqpad_div_seamcos` reads:

| | |
|---|---|
| composed map, own `h_0` | 0.0035 |
| **operator+decoder fed the true `Enc(DIV,N,x^2)`** | **0.953** |
| same, held-out `N` | 0.003 |

The parent's `redmod`, trained on `y = x^2`, read **0.022 (below floor)** on those inputs and
was read as the general reduction failing. Trained on uniform `y` instead, the same reduce
reads 0.953 on exactly the inputs the squaring path supplies. The parent's own caveat list
asked for this ("nothing here reads the reduce out from inside a trained `sq`").

So the atom decomposes into three measured quantities rather than one null: the reduce works
(0.953), the encoder reaches the state it needs at cos 0.642, and the composition reads 0.0035.
Note this is **sub**-multiplicative — 0.068 × 0.953 ≈ 0.065 is what the factors predict. A
state at cos 0.64 is worse than being wrong 93% of the time, which is the same behaviour as
`ballistic_depth` §9's α=0.5 half-snap collapse.

**The split control.** `sqpad_div_seamcos_gx` is the same arm under §1's clean global `x` split:
composed 0.0018 against 0.0035, and seam cosine 0.970 → 0.625 against 0.972 → 0.642. So the
per-modulus split was inflating the composed number by ~2×, and the multiply's failure at 3
digits is not a split artefact.

## 5. A scissors: no digit width tested where both halves work at once

§2 says the multiply needs a large input space; 3-digit moduli give `x` fewer than 1000 values.
So the composed atom was rerun at 4 digits (~5,000 training `x`), with the reduce measured
alone at the same width — without which a composed null cannot be assigned to a half.

| arm | seen | held-out `y` | held-out `N` |
|---|---|---|---|
| `div4_q64` (quotient ≤ 64) | 0.986 | **0.986** | 0.007 |
| `div4_qfull` (full range) | 0.0005 | **0.0004** | 0.0003 |

| arm | seen `x` | held-out `x` | floor | seam cos held-out | oracle |
|---|---|---|---|---|---|
| `sq4_plain` | 0.982 | 0.0004 | 0.014 | — | — |
| `sq4_div_seamcos` | 0.957 | 0.0003 | 0.014 | **0.945** | 0.0005 |

**The 4-digit composed atom failed on the reduce, not the multiply.** Its division branch reads
0.0004 and its oracle 0.0005, so there was nothing to hand off to. And the failure is not digit
width as such — `div4_q64` at bounded quotient reads 0.986 with no memorisation gap
(seen ≈ held-out). It is the **quotient range**, which `x^2` for `x < N` spans in full by
construction, so the composed task cannot avoid the hard case.

Read with §1 and §2, the two widths tested bracket the problem from opposite sides:

- **3 digits** — reduce works (oracle 0.953); multiply has <1000 possible `x` and reads 0.068.
- **4 digits** — multiply improves exactly as §2 predicts (seam cos on held-out `x` rises
  0.625 → **0.945**, i.e. the encoder does start computing `x^2` for unseen inputs); full-range
  reduce reads 0.0004.

Whether a width or a budget exists where both hold simultaneously is open — nothing here rules
it out, and both halves were still improving in their own best cells.

---

## Predictions that failed

Recorded because they were made before the runs.

1. **"More capacity should make transfer worse"** (§2). The mechanism proposed in §1 — that
   `mul5` generalised because tabling was unavailable — was tested directly and failed in both
   directions: 27.4M parameters gives the best transfer in the node, and an arm that failed to
   memorise (`memb_n5k`) generalised at 0.001.
2. **The `{div aux} × {seam}` 2×2 was expected to move the composed map** (§4). Every corner
   reads at or below floor. The oracle probe shows why, but the 2×2 as posed is null.
3. **Dense-`N` division was expected to attack the rule axis** (§3). Sampling `N` over all 900
   three-digit integers made the task harder and the arm did not train.
4. **The seam was reported installed on a scale-dependent loss** (§4), at MSE 0.0010 and cosine
   0.081. Caught by a checkpoint probe rather than by the training-time instrument.

## Honest caveats

- **Nothing in §2's 50k row is converged**, so the capacity ordering is a budget artefact and
  the `memb_d512L8` extrapolation is a power-law fit over half of one run — an optimistic
  bound, not a prediction.
- **`divqfull` does not replicate the parent's `redmod_qfull`** (0.482 vs 0.773) and the
  difference is unexplained. §3 states it rather than resolving it.
- **The ladder arithmetic assumes independent per-step errors**, which is false in detail: the
  error set is state-indexed, so a trajectory that enters it once tends to stay wrong. It also
  inherits `ballistic_depth` §9's re-projection result, measured at fixed `N` on seen bases.
- **The multiply arms carry `N` in the prompt as a nuisance field** and draw it from arbitrary
  `w`-digit integers rather than the semiprime family, since their target does not depend on
  it. This differs from `sqnomod`, which drew `x` from unit groups.
- **Two widths, one substrate.** §5's scissors is 3 and 4 digits at one budget each.
- **No submission was built.** As in every cut in this experiment, `T` never enters the prompt
  and the answer encoding is fixed-width, so these are not upstream-comparable scores.

## Reproduce

```bash
cd experiments/
E=one_layer_deeper/rule_acquisition/exact_atom/exact_atom.py::exact_atom
R="MODAL_PROFILE=chromatic modal run --detach"

# §3 — the reduce vs budget (eps logged 24x on exhaustive pools)
$R $E --tag divq8     --arms "divq8"            --steps 600000 --seed 0
$R $E --tag divq64    --arms "divq64"           --steps 600000 --seed 0
$R $E --tag divqfull  --arms "divqfull"         --steps 600000 --seed 0
$R $E --tag divdensen --arms "divqfull_densen"  --steps 600000 --seed 0

# §1 — the multiply
$R $E --tag mul_a --arms "mul3,mul4"        --steps 300000 --seed 0
$R $E --tag mul_b --arms "mul5,mul5_sparse" --steps 300000 --seed 0

# §4 — the composed atom. `sq_a`/`sq_b`/`sq_c` are the mse-seam arms, kept for the record.
$R $E --tag sq_a     --arms "sqpad,sqpad_div"                 --steps 300000 --seed 0
$R $E --tag sq_b     --arms "sqpad_seam,sqpad_div_seam"       --steps 300000 --seed 0
$R $E --tag sq_c     --arms "sqpad_div_seam_densen"           --steps 300000 --seed 0
$R $E --tag sq_cos   --arms "sqpad_div_seamcos,sqpad_seamcos" --steps 300000 --seed 0
$R $E --tag sq_cos10 --arms "sqpad_div_seamcos_w10"           --steps 300000 --seed 0
$R $E --tag sq3gx    --arms "sqpad_div_seamcos_gx"            --steps 300000 --seed 0

# §2 — the memorisation boundary (matched 300k against mul5)
$R $E --tag memb_cap  --arms "memb_d128,memb_d512" --steps 300000 --seed 0
$R $E --tag memb_cap8 --arms "memb_d512L8"         --steps 300000 --seed 0
$R $E --tag memb_n    --arms "memb_n5k,memb_n20k"  --steps 300000 --seed 0

# §5 — 4 digits, with the reduce measured alone at the same width
$R $E --tag div4a --arms "div4_q64"        --steps 600000 --seed 0
$R $E --tag div4b --arms "div4_qfull"      --steps 600000 --seed 0
$R $E --tag sq4a  --arms "sq4_div_seamcos" --steps 600000 --seed 0
$R $E --tag sq4b  --arms "sq4_plain"       --steps 600000 --seed 0

# checkpoint probes: modulus-uniform eps (§3) and the seam/oracle decomposition (§4).
# Trains nothing; runs attached in ~1 min per arm.
MODAL_PROFILE=chromatic modal run \
  one_layer_deeper/rule_acquisition/exact_atom/reprobe.py::reprobe \
  --tag sq_cos --arms "sqpad_div_seamcos,sqpad_seamcos"

# fetch + analyse
for t in divq8 divq64 divqfull divdensen mul_a mul_b sq_a sq_b sq_c sq_cos sq_cos10 \
         sq3gx memb_cap memb_cap8 memb_n div4a div4b sq4a sq4b; do
  MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
    /exact_atom/$t/results_seed0.json \
    one_layer_deeper/rule_acquisition/exact_atom/results/$t/results_seed0.json --force
done
python3 one_layer_deeper/rule_acquisition/exact_atom/analyze.py --tags <comma-separated>
```

Modal volume `one-layer-deeper-data`, results at `/exact_atom/<tag>/results_seed<N>.json`,
checkpoints at `/exact_atom/<tag>/ckpt/<arm>_seed<N>.pt`, probes at
`/exact_atom/<tag>/reprobe_seed<N>.json`.

## Next steps

1. **Converge `memb_d512L8`.** Its slope is the only thing in the program pointing at an
   exactly-correct learned map, and it is a fit over half a run. 1M–3M steps at `w=5` would
   either produce the first `eps` inside the ladder or flatten the extrapolation.
2. **Long division as an explicit decomposition.** §5 localises the reduce's failure to the
   *quotient range*, and `div4_q64` shows bounded quotient is fine at 4 digits. A staged reduce
   — one quotient digit at a time, each stage a bounded-quotient problem — is the decomposition
   that fact implies, and needs no trapdoor.
3. **The rule axis remains untouched.** Every held-out-`N` readout in this node is at or below
   floor, including under an oracle state. Nothing tried across three cuts has moved it, and
   §3's dense-`N` attempt did not train.
4. **A width where both halves hold.** §5 brackets it from two sides at one budget each;
   whether the bracket is empty is not established.
