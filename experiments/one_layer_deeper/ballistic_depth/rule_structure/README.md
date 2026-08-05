# Rule structure — is the representation organised by the group, or is it a lookup table?

**Up**: [../README.md](../README.md) (ballistic_depth) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 6 trained models across 2 moduli and 3 configs, each against a permutation
null, an untrained-encoder control, and a synthetic positive control. **Date**: 2026-08-03.
Pure re-analysis: loads checkpoints, trains nothing.

---

## One-liner

[`variable_modulus/`](../../variable_modulus/README.md) §3 found that no arm learns modular
squaring. Read at *depth 1* that is not a claim about composition at all — it says the model
cannot compute a **single** squaring on an unseen input, so every depth result in this cut sits
on top of a memorised atom. This probe asks which kind of failure that is: the algorithm is
**unattractive** (reachable, but lookup is cheaper — remedy: capacity economics) or
**unreachable** (SGD cannot get there from a decimal-digit representation — remedy: the
representation itself).

The answer, within the models tested, is **unreachable**. In the coordinate where the question
is exact, six trained encoders spanning two moduli, an 11.6× state-space range and a 4×
parameter range show **no group organisation beyond a permutation null** — and all of them sit
*below the weakest synthetic group signal that the same instrument can still detect*. Scaling
the state set 11.6× moved the structure **down**, not up, which is the opposite of what
"lookup is merely cheaper" predicts.

## The coordinate, and why it is exact

For `N = p·q`, the CRT gives `(Z/N)* ≅ Z/(p−1) × Z/(q−1)`, with a primitive root in each
factor. In that coordinate squaring is exactly

```
(a, b)  ->  (2a mod p−1,  2b mod q−1)
```

the doubling map, per factor. This is not an assumption about the substrate: the *image* of that
map has `(p−1)/2 · (q−1)/2` points, which is the `reachable_states_depth_ge_1` field
[`TaskSpec.describe()`](../../squaring_mod.py) already reports — 9·23 = **207** for `N=893`,
29·83 = **2407** for `N=9853`. Our own reachable-state counts *are* the group structure. We can
compute the coordinate because we hold the trapdoor `(p, q)`; the model never sees it.
`group_coords` asserts both properties at run time — that the map is a bijection onto the grid,
and that squaring is the doubling map in it — because a silently wrong coordinate would produce
a confident null.

The readout is generator-independent: changing the primitive root relabels `a → k⁻¹a`, which
permutes frequencies, and every statistic below is invariant to a permutation of frequencies.

## Four readouts, three calibrations

| readout | what it asks |
|---|---|
| `fourier` | 2D DFT of the embedding grid over `Z/(p−1) × Z/(q−1)`. A group-organised representation concentrates power in few modes; a table spreads it. Also run on the **QR sublattice** (even `a`, even `b`) alone — the reachable set every rollout past `t=1` actually lives on. |
| `translation` | `cos(Enc(x), Enc(y))` should depend only on the *difference* `(Δa, Δb)` if the representation respects the group. Measured as the R² of predicting the full pairwise-cosine matrix from a difference-indexed table. |
| `linear_op` | fit **one** linear map `M` with `Enc(x²) ≈ M·Enc(x)` on training bases, score on held-out bases. In the group coordinate the true operator *is* linear, so this is satisfiable in principle. |
| `heldout_op` | the model's own tied operator applied to a held-out base's encoding — §2's `on_manifold_cos` restricted to bases never trained on, plus whether it decodes. |

- **`perm` null** — the same embeddings with the `x → (a,b)` assignment shuffled. Preserves the
  point cloud exactly and destroys only the group.
- **`init` control** — an untrained encoder at the same config, catching structure that is
  architectural rather than learned.
- **`synthetic` positive control** — an embedding built from `n_freq` random Fourier modes over
  the same grid at a given SNR: what "found the group" scores on this instrument.

## 1. The positive control is what makes the result readable

Against the permutation null alone, the trained `base` model at `N=893` reads **z = +33**. That
is a 30-sigma effect, and it is functionally nothing — the permutation null has tiny variance, so
significance is cheap. The synthetic control supplies the missing end of the scale:

| representation | `fourier` top-1% | `translation` R² |
|---|---|---|
| synthetic, clean group (2 modes, SNR 10) | **0.990** | **0.990** |
| synthetic, 8 modes, SNR 10 | 0.522 | 0.955 |
| synthetic, 8 modes, SNR 1 | 0.266 | 0.820 |
| synthetic, **weakest still detectable** (8 modes, SNR 0.3) | 0.052 | 0.294 |
| — | | |
| `base` `N=893` (trained) | 0.031 | 0.040 |
| `+cycle` w=1 `N=893` (trained) | 0.033 | 0.041 |
| `+cycle` w=10 `N=893` (trained) | 0.037 / 0.034 *(seeds 0/1)* | 0.035 / 0.034 |
| permutation null | 0.015–0.016 | 0.019–0.025 |

Every trained model sits **below a group signal buried under 3× its own amplitude in noise**. On
the discriminating statistic they are at 0.03–0.04 where that weakest synthetic case still reads
0.294 — roughly 7× short. Reporting the z-score without this row would have recorded
"statistically significant group structure" for a null result.

## 2. On the set the rollout lives on, there is nothing at all

The QR sublattice — the 207 (resp. 2407) reachable states, where everything past `t=1` happens:

| model | QR `fourier` z |
|---|---|
| `base` `N=893` | **+0.0** |
| `+cycle` w=1 `N=893` | −2.3 |
| `+cycle` w=10 `N=893` | −1.6 / −1.4 |
| `base` `N=9853` (wide, ID 1.000) | −1.1 |

Not weak — absent, and mildly negative as often as positive. Whatever faint whole-group signal
the full grid carries does not survive restriction to the states the operator actually iterates
on.

## 3. The cycle term's generalisation gain is not group-based

§2 of the parent cut reports the one intervention in this program that ever moved the
generalisation axis: the label-free cycle term takes held-out-`x` from **0.001 to 0.32**,
unpredicted at the time. If that were the model partially finding the group, `consist` should
separate from `base` here. It does not:

| model | `fourier` | `translation` R² | own operator on held-out `x`: cos / decode |
|---|---|---|---|
| `base` | 0.031 | 0.040 | 0.080 / **0.000** |
| `+cycle` w=1 | 0.033 | 0.041 | 0.766 / **0.349** |
| `+cycle` w=10 | 0.037 | 0.035 | 0.841 / **0.349** *(seed 1: 0.830 / 0.229)* |

Identical on every structure readout, 0.000 vs 0.349 on the behaviour. The probe independently
re-derives §2's held-out-`x` number from checkpoints (0.349 against 0.323) and shows it is **not
algebraic**. A plausible reading, *not tested here*: unseen `x` lands near seen `x` in embedding
space and a smooth operator carries it — interpolation rather than rule.

## 4. A single linear operator does not fit, which kills a proposed next term

Fitting one `M` with `Enc(x²) ≈ M·Enc(x)` on training bases and scoring held-out ones:

| model | held-out cos | held-out **decode** |
|---|---|---|
| `base` `N=893` | 0.644 | 0.000 |
| `+cycle` w=1 | 0.708 | 0.024 |
| `+cycle` w=10 | 0.815 / 0.811 | 0.084 / 0.060 |
| `base` `N=9853` (wide) | 0.479 | 0.000 |

The cosines look respectable and the decodes are ~zero, so the operator is not a linear map in
the encoder's coordinate. This probe was built partly to license or kill a `Enc(x²) ≈ M·Enc(x)`
*training* constraint as the next term to add after the cycle term; it kills it.

## 5. Structure does not grow with the size of the table

The capacity-economics test, and the one route by which "unattractive" could still have been
true — if lookup is merely *cheaper*, a table 11.6× larger should shift the balance:

| model | states | ID (T≤6) | `fourier` (null) | `translation` R² (null) |
|---|---|---|---|---|
| `base` `N=893` | 207 | 1.000 | 0.031 (0.015) | 0.040 (0.025) |
| `base` `N=9853` narrow | 2407 | 0.958 | 0.015 (0.014) | 0.015 (0.015) |
| **`base` `N=9853` wide** | 2407 | **1.000** | **0.017 (0.016)** | **0.007 (0.006)** |

Structure fell to exactly the null, and the wide model rules out under-training as the
explanation — it is ID-perfect at 8× the operator width. Within this range, **a bigger lookup
table does not push the model toward the algorithm.**

## 6. The instrument detects representational collapse — and caught a real one

The `init` control reads held-out cos **0.99** on the linear fit and **0.96** on the model
operator, with decode 0.000, because an untrained encoder maps every input to nearly the same
point and cosine on a collapsed representation is uninformative. That calibration then paid off
immediately: the `+cycle` w=10 arm at `N=9853` reads linear-op cos **0.999**, model-op cos
**0.997**, decode **0.000**, `translation` R² 0.002 = null — every embedding is the same point.
See [`../README.md`](../README.md) §10; the arm has closure 0.998 and in-distribution accuracy
0.001.

**The consequence for the parent cut's main instrument**: `on_manifold_cos` is maximised by
collapse, so **closure is only interpretable conditional on in-distribution competence**. Every
closure claim in §2/§6/§7 is measured where ID = 1.000 and stands, and §6 already restricted its
correlation to the 60 ID-competent runs — but the failure mode was hypothetical there and is now
documented.

## Honest caveats

- **This measures the encoder's readout at the `ANS` position.** Group structure could in
  principle live inside the operator's hidden layer, which the probe does not see. The `linear_op`
  and `heldout_op` readouts constrain the operator behaviourally but not representationally.
- **A null on these statistics is not a proof of absence.** `fourier` and `translation` detect
  structure that is *linear in the group characters*; a representation encoding the group through
  some other nonlinear code would read as null here. What is established is that the specific,
  strong form of group organisation — the one that makes the operator a shift — is absent.
- **Nothing here says the algorithm is unlearnable in principle**, only that these six trained
  encoders do not encode it and that neither the cycle term nor an 11.6× larger state set moved
  it. The remedies this points at (digit representation, positional scheme, intermediate
  supervision on the multiply and the reduce) are untested here.
- **Single seed for most configs.** `w_cyc10.0` was run at two seeds and agrees to within 0.003
  on `fourier` and 0.001 on `translation`; the gap to the weakest positive control is ~7×, so
  seed variation does not bear on the sign.
- **The permutation null is resampled per run**, so null columns vary by ~0.001 between tables
  above; the z-scores should be read as "large and meaningless", not compared with each other.

## Reproduce

```bash
cd experiments/
# one job per (training tag, seed); loads that tag's checkpoints, trains nothing
for t in coldstart w_cyc10.0 scale9853 scale9853_ff8192; do
  MODAL_PROFILE=chromatic modal run \
    one_layer_deeper/ballistic_depth/rule_structure/dlog_probe.py::dlog_probe \
    --tag $t --seed 0 --arms "base,consist"
done

MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
  /ballistic_depth/<tag>/dlog_probe_seed0.json \
  one_layer_deeper/ballistic_depth/results/<tag>/dlog_probe_seed0.json --force
python3 one_layer_deeper/ballistic_depth/rule_structure/dlog_probe_table.py
```

Requires `--save-ckpt` on the training run. Results land beside the parent cut's, at
`/ballistic_depth/<tag>/dlog_probe_seed<N>.json` on volume `one-layer-deeper-data`, because they
are keyed by the training tag they were read from.

**Gotcha.** `modal volume get` into a local directory that does not exist writes a *file* at that
path rather than a tree, and a path relative to a `cd`-ed working directory nests silently — both
happened while assembling this cut. Always pass `--force` and a full per-file destination path.

## Next steps

1. **Probe the operator's hidden layer**, not just the encoder readout — the one place the
   caveats above leave open.
2. **Run on `variable_modulus` checkpoints.** The group coordinate is per-modulus and the probe
   assumes a single `(p, q)`; extending it to a family asks whether the *shared* structure across
   moduli is any more organised than the per-modulus one, which is the arity version of this
   question.
3. **If a representation change is ever tried** (reversed digits, place-value markers), this
   probe is the readout that says whether it worked, independent of accuracy.
