# Rule acquisition — why the one-step map is memorised, and what makes it generalise

**Up**: [../README.md](../README.md) (one_layer_deeper) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 16 arms across 7 tags, single seed. **Date**: 2026-08-03.
**Extended 2026-08-05** by [`exact_atom/`](exact_atom/README.md), which re-reads this cut
against an *exactness* target rather than an accuracy one. It **retracts §2's multiply result**
(see the note there), converts §3's central inference into a direct measurement, and finds the
reduce and the multiply fail at different digit widths.

---

## One-liner

[`variable_modulus/`](../variable_modulus/README.md) §3 found that no arm learns modular
squaring, and [`ballistic_depth/rule_structure/`](../ballistic_depth/rule_structure/README.md)
read that as **unreachable** — gradient descent cannot get to the algorithm from a decimal-digit
representation. This cut tests that node's remedy list and finds a different account.

Nine arms sit at or below the analytic no-reduction floor on held-out `x`, including a 21M-parameter
arm stacking every intervention at once. But **both halves of the atom generalise on their own**:
`x -> x^2` reads **0.920** and `x^2 -> x^2 mod N` reads **0.773–0.996** — provided the reduce is
shown its own input space. Trained on the inputs that `x^2` actually supplies, the *same* reduce
reads **0.022**, below floor.

The reduce's input space is `N` times larger than the multiply's and both see the same number of
examples, so inside `x^2 mod N` the second stage is sampled ~`N` times more sparsely. Within the
scales tested, what makes the atom memorisation-only is that quantity, not the reachability of the
algorithm. The surviving barrier is a different axis: generalising the reduce across **moduli**.

**The multiply half of that claim is retracted** — see §2's note and
[`exact_atom/`](exact_atom/README.md) §1. The 0.920 is a split artefact; on a clean split the
multiply reads **0.068**. The reduce half stands and is strengthened: measured from inside a
trained composed model rather than in isolation, the reduce reads **0.953** on held-out `x`.

---

## Why this node exists

Everything in [`ballistic_depth/`](../ballistic_depth/README.md) and
[`variable_modulus/`](../variable_modulus/README.md) is about *composing* an operator. Read at
depth 1, `variable_modulus` §3 says there is nothing to compose — the model cannot compute a
single modular squaring on an unseen input, so every depth result in this program sits on top of a
memorised atom. The parent README's next-step 2 names this directly: *the axis the benchmark is
actually stuck on is rule acquisition, not depth.* The public leaderboard agrees — stuck at `T=1`,
best 6/768.

`rule_structure` narrowed the question to **unattractive** (lookup is merely cheaper; remedy:
capacity economics) vs **unreachable** (remedy: the representation) and answered *unreachable*,
leaving its remedy list — representation, compute, pressure — untested. This cut tests it, and
adds an axis that list did not contain: **which part of the atom is hard**.

## Protocol

Fixed across every arm, and matched to the prior probe so `sq` is a direct replicate:

- **Train at depth 1 only.** Not a composition experiment.
- **Half of every modulus's bases held out** (`test_x_fraction=0.5`), so lookup cannot cover the
  test set by construction.
- 3-digit moduli, 8 train / 12 held-out unless an arm says otherwise; `d_model` 256, 150k steps,
  batch 256, constant LR 3e-4, wd 0.1. The prior 500k probe found held-out `x` flat at
  wd ∈ {0.1, 1.0}, so only 0.1 is run here.
- The readout is held-out-`x` exact match against the **no-reduction floor computed on each arm's
  own eval pool** — the fraction of examples answerable with no knowledge of `N` at all. Floors
  differ by pool and by task (0.031/0.055/0.041 across the three pools for `sq`), which the prior
  cut's single quoted 0.041 did not distinguish.
- Held-out `x` is tracked **during** training at 30 points, because grokking is a transition and
  an endpoint-only readout cannot separate "never generalised" from "generalised late".

## 1. Every remedy on `rule_structure`'s list is null

| arm | change from `sq` | params | train | ID@1 | held-out `x` | floor |
|---|---|---|---|---|---|---|
| `sq` | — (`x -> x^2 mod N`) | 2.12M | 1.000 | 1.000 | 0.019 | 0.041 |
| `binary` | `N`, `x`, answer in base 2 | 2.12M | 1.000 | 1.000 | 0.015 | 0.041 |
| `abacus` | shared place-value embedding | 2.12M | 1.000 | 1.000 | 0.014 | 0.041 |
| `enc8` | 2 → 8 encoder layers | 6.86M | 1.000 | 1.000 | 0.022 | 0.041 |
| `inner8` | operator applied 8× per step | 2.12M | 1.000 | 1.000 | 0.023 | 0.041 |
| `wide4k` | `d_ff` 1024 → 4096 | 6.85M | 1.000 | 1.000 | 0.022 | 0.041 |
| `manymod` | 8 → 142 train moduli | 2.12M | 0.980 | 0.976 | 0.050 | 0.049 |
| `auxprod` | + head decoding `x^2` off the encoder state | 2.14M | 1.000 | 1.000 | 0.034 | 0.041 |
| **`stack`** | **all of the above at once** | **21.0M** | 0.961 | 0.938 | **0.051** | 0.049 |

Every one at or below floor, and every training trace flat across 150k steps — no rising tail
anywhere. `stack` is the capstone: base-2 encoding, 8 encoder layers, 8 serial operator
applications per step, 4096-wide FFN and 142 moduli together, at 10× the parameters, buy
**+0.002** over floor.

Two of these were pre-registered as the strongest candidates and both failed:

- **`binary` was the representation arm with real teeth** — in base 2 every partial product is an
  AND, so the digit convolution becomes additive. It reached ID 1.000 *faster* than `sq` and
  generalised *worse* (0.015 vs 0.019). Digit-order interventions (reversal, interleaving) were
  deliberately **not** run: the encoder is bidirectional over a 9-token prompt with learned
  absolute positions, so they are relabelings of free parameters and near-vacuous by construction.
  `abacus` is the weak-prior arm included so `binary` reads against something rather than nothing.
- **`manymod` is the capacity-economics test done with training-time pressure**, where
  `rule_structure` §5 could only vary the state-set size. 17.75× more rules makes the lookup table
  17.75× more expensive — ID drops 1.000 → 0.976, so the pressure is real — and generalisation
  moves +0.001.

`enc8`/`inner8`/`wide4k` close a confound this cut was written to address: calling the null
"unreachable" is not safe when 2 encoder layers plus one MLP is very little serial compute for a
3-digit multiply *and* a division. 4× the depth, 8× the per-step serial compute and 3.2× the
parameters move it by nothing.

## 2. Both halves of the atom generalise (the decomposition)

The Hilbert move — find the special case. `sqnomod` keeps the input and drops the reduction;
`redmod` keeps the reduction and is handed the product, on the same base split. Composing
`sqnomod` then `redmod` gives back exactly `sq`.

| arm | map | held-out `x` | floor | lift |
|---|---|---|---|---|
| `sq` | `x -> x^2 mod N` | 0.019 | 0.041 | −0.022 |
| `sqnomod` | `x -> x^2` | **0.920** | 0.000 | +0.920 |
| `mul2` | `x -> 2x mod N` | **0.924** | 0.486 | +0.438 |
| `redmod` | `x^2 -> x^2 mod N` | 0.022 | 0.041 | −0.019 |

**Retracted (2026-08-05): `sqnomod`'s 0.920 is a split artefact.** The base pools above are
split by a permutation *inside* the per-modulus loop, but `sqnomod`'s target `x^2` does not
depend on `N` at all — so an `x` held out under one modulus is a training input under another
with probability `~1 - 0.5^k` for the `k` unit groups containing it, and the `(x, x^2)` pair was
usually seen. [`exact_atom/`](exact_atom/README.md) §1 splits on `x` alone and reads **0.068**
at the same digit width, with seen-`x` at exactly 1.000 — memorisation of ~500 squares. A
matched-count control across a 100× larger space reads 0.00003, so even the 0.068 is mostly the
small space. The multiply *does* generalise at larger scale (0.909 at 5 digits, 27.4M
parameters), but not here, and **"the multiply is the easy half" does not survive.** `mul2` and
the `redmod_*` arms are unaffected: their targets depend on `N`, so the per-modulus split is a
real split for them.

`sqnomod` reaches 0.924 by step 5k and is flat thereafter — read now as converged
memorisation rather than as easy generalisation. `mul2`'s reduction is a single conditional
subtract (`2x < 2N`), so it isolates
only the easy case; its floor is 0.486 precisely because half its examples need no reduction, and
it takes 0.438 of the remaining 0.514.

`redmod` is the informative cell, and on its own it says the general reduction — dividing a
6-digit number by a 3-digit one — is what fails, handed the product outright. §3 shows that
reading is wrong.

## 3. The reduce generalises when it is shown its own input space

`redmod` trains on `y = x^2` for `x` in the train half of the units: ~214 problems per modulus. The
reduce's input space is `[0, N^2)`, so that is ~0.1% coverage, against the multiply's ~43% of
`[0, N)`. Same sample count, and the ratio is exactly `N`. The ladder samples `y` **uniformly**
instead, with the quotient as the knob and the input width fixed at 6 digits throughout so only the
quotient distribution moves. Train/test split by an arbitrary hash of `(N, y)`, so neither half is
a learnable predicate and dense sampling cannot make the split leak.

| arm | reduce input | train | held-out | floor | lift |
|---|---|---|---|---|---|
| `redmod` | `y = x^2`, ~214/modulus | 1.000 | 0.022 | 0.041 | −0.019 |
| `redmod_q8` | uniform, quotient ≤ 8 | 1.000 | **0.996** | 0.102 | +0.893 |
| `redmod_q64` | uniform, quotient ≤ 64 | 0.984 | **0.982** | 0.021 | +0.962 |
| `redmod_qfull` | uniform, `y ∈ [0, N^2)` | 0.891 | **0.773** | 0.003 | +0.770 |

Identical architecture, parameter count and step budget across all four. The only thing that
changes is how the reduce's input space is sampled, and it moves held-out accuracy from below
floor to 0.773–0.996.

Two features worth recording. Division degrades **gracefully** with quotient range
(0.996 → 0.982 → 0.773), not off a cliff. And for every dense arm ID ≈ held-out (0.983/0.982,
0.872/0.773) — with fresh draws on both sides there is no memorisation gap to have, which is
exactly the condition the sparse arms cannot reach.

**What is manipulated and what is inferred.** The manipulated fact is that coverage of the
reduce's input space controls whether the reduce generalises, holding everything else fixed. The
*inference* is that this accounts for `sq`: `redmod` is `sq`'s second stage in isolation, trained
on exactly the input distribution `x^2` supplies, and `x^2` reaches only ~`N` of `N^2` possible
reduce-inputs. That is a strong inference but not a direct measurement — nothing here reads the
reduce out from inside a trained `sq`.

It does, however, retro-explain §1: representation, depth, width, serial compute, capacity
pressure and staged supervision all leave coverage untouched. `auxprod` is the sharpest case —
handing the model the two-stage structure cannot help when stage two is the starved one.

## 4. The surviving barrier is the rule axis, not the arithmetic

Held-out **`N`** across the ladder, and with the modulus count raised:

| arm | moduli | held-out `x` | held-out `N` | floor | train |
|---|---|---|---|---|---|
| `redmod_q8` | 8 | 0.996 | 0.096 | 0.102 | 1.000 |
| `redmod_q64` | 8 | 0.982 | 0.009 | 0.021 | 0.984 |
| `redmod_qfull` | 8 | 0.773 | 0.003 | 0.003 | 0.891 |
| `redmod_q64_many` | 142 | 0.689 | **0.058** | 0.015 | 0.684 |
| `redmod_qfull_many` | 142 | 0.043 | 0.005 | 0.002 | **0.031** |

At 8 moduli the model learns eight specialised divide-by-`N` routines: held-out `N` is at or below
floor even where held-out problems read 0.982. Raising to 142 moduli moves it from below floor to
~3.8× floor — directionally right, and the one place in this cut where modulus count does
anything — but 0.058 is not competence.

Both many-moduli arms are **undertrained, not saturated**: `redmod_q64_many` was still rising at
300k (train 0.684), and `redmod_qfull_many` reads train **0.031** — it did not fit the training
data at all, where the same task at 8 moduli reached 0.891. So `qfull_many`'s 0.043 records
"did not fit in this budget", not a ceiling.

## 5. A group-structure null does not license "no algorithm"

`rule_structure` already carried this as a stated caveat; this cut supplies a demonstration.
Reusing that node's two discriminating statistics on these encoders:

| arm | held-out `x` | fourier (null) | translation R² (null) |
|---|---|---|---|
| `sqnomod` | **0.920** | 0.0250 (0.0142) | 0.0920 (0.0843) |
| `sq` | 0.019 | 0.0518 (0.0159) | 0.1201 (0.0667) |

The arm that generalises reads **lower** on both statistics than the arm that does not, and sits
essentially at its permutation null. This is unsurprising once stated — a model computing `x^2` by
place-value arithmetic has no reason to organise `x` by the discrete-log coordinate — but it means
the inference from "no group organisation" to "the algorithm was not learned" does not go through.
What `rule_structure`'s null establishes is the absence of *group-character-linear* organisation,
which is what that node's own caveats say and what its headline compresses away.

Consequently the **unreachable** verdict should be read as scoped to the group-coordinate probe
and to the models it examined, not as a claim about learnability: §3 exhibits the same model class
learning full-range modular reduction to 0.773 on unseen problems.

## Child cuts

### [`exact_atom/`](exact_atom/README.md) — how far the one-step map is from being *exactly* right

**Goal**: re-read this cut's object against the target the benchmark actually gates on. Under
[`ballistic_depth/`](../ballistic_depth/README.md) §9's test-time re-projection at `k=1`,
certifying Hard ladder rung `T` needs one-step error `eps <~ 9e-4/T`, so each rung is worth one
factor of two in `eps` and the whole ladder `T=1..64` is 64× — depth is the cheap axis and rung
zero is the expensive one.

**Finding**: the atom decomposes into three measured quantities instead of one null. Fed the
true `Enc(DIV, N, x^2)` from inside a trained composed model, the operator and decoder return
`x^2 mod N` on **held-out `x` at 0.953** where this cut's `redmod` read 0.022 — §3's
"coverage, not reachability" inference becomes a measurement. The multiply is the half that
fails (§2's retraction), and it fails for an ordinary reason: at 3 digits `x` has fewer than
1000 values. Given 50k training inputs and 27.4M parameters it reaches **0.909** and is still
improving steeply. A pre-registered mechanism — *generalisation appears where memorisation
becomes infeasible* — was tested directly and **failed in both directions**. The two widths
tested bracket the problem from opposite sides: at 3 digits the reduce works and the multiply
does not; at 4 digits the multiply improves exactly as predicted and the **full-range** reduce
collapses to 0.0004, while a bounded-quotient reduce at the same width reads 0.986 — so the
reduce's difficulty is the *quotient range*, which `x^2` spans in full by construction.
Also: a closure loss must be scale-free (an MSE seam reached 0.0010 at cosine **0.081**), and
a closure statistic is only interpretable if its target branch is independently grounded.

## Predictions that failed

Recorded because they were made before the runs, and both were mine:

1. **The multiply was predicted to be the hard half**, on the strength of the arithmetic
   length-generalisation literature. It is the easy half (0.920, converged by step 5k); the reduce
   is where the difficulty is.
2. **"General reduction does not generalise"** was reported from `redmod` alone, before the
   coverage control had run. `redmod_q8`/`q64`/`qfull` falsify it. The coverage confound had been
   flagged in the same breath as the claim and should have blocked it.
3. **`binary` was predicted to be the representation intervention with teeth.** It is null, and
   slightly worse than baseline.

## Honest caveats

- **Single seed throughout.** The load-bearing contrasts are `sqnomod`/`redmod` (§2) and
  `redmod` vs `redmod_q*` (§3); neither is replicated.
- **`redmod_qfull` had not converged** (train 0.891 at 150k), so 0.773 is a lower bound on what
  full-range division reaches, not a level.
- **Both many-moduli arms are undertrained**, as above. §4's barrier is real but its height is
  unmeasured.
- **3-digit moduli only.** Whether the coverage account survives at 4 digits — where the reduce's
  space grows as `N^2` again — is untested.
- **Nothing here reads the reduce out from inside a trained `sq`.** §3's account of §1 is an
  inference from an isolated stage, as flagged above.
- **The group readout is only comparable across arms sharing a modulus set.** `manymod` and
  `stack` probe 5 different moduli, so their numbers (translation R² 0.325/0.182) are not
  commensurable with the 8-modulus arms' — the nulls move with the grid.
- **The group probe covers 3 of 8 moduli** for the small family: the coordinate needs `p-1, q-1 ≥ 4`
  and `min_factor=3` admits `p=3`.
- **No submission was built.** As in both parent cuts, `T` never enters the prompt and the answer
  encoding is fixed-width, so these are not upstream-comparable scores.

## Reproduce

```bash
cd experiments/
# one job per knob family — each tag is a separate output path, see the gotcha below
MODAL_PROFILE=chromatic modal run --detach \
  one_layer_deeper/rule_acquisition/rule_acquisition.py::rule_acquisition \
  --tag cut1_task --arms "sq,mul2,sqnomod,manymod" --steps 150000 --seed 0
#   ... --tag cut1_repr    --arms "abacus,binary"
#   ... --tag cut1_compute --arms "enc8,inner8,wide4k"
#   ... --tag cut1_stack   --arms "stack"
#   ... --tag cut2_decomp  --arms "redmod,auxprod"
#   ... --tag cut3_quotient --arms "redmod_q8,redmod_q64,redmod_qfull"
#   ... --tag cut4_manydiv --arms "redmod_q64_many,redmod_qfull_many" --steps 300000

# fetch + analyse
for t in cut1_task cut1_repr cut1_compute cut1_stack cut2_decomp cut3_quotient cut4_manydiv; do
  MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
    /rule_acquisition/$t/results_seed0.json \
    one_layer_deeper/rule_acquisition/results/$t/results_seed0.json --force
done
python3 one_layer_deeper/rule_acquisition/analyze.py \
  --tags cut1_task,cut1_repr,cut1_compute,cut1_stack,cut2_decomp,cut3_quotient,cut4_manydiv
```

**Gotcha.** Every job writes `results_seed<N>.json` under its own tag, so **parallel jobs must use
different tags** or they silently overwrite each other; `analyze.py --tags` reassembles them. As in
both parent cuts, `modal volume get` needs `--force` and a full per-file destination path.

## Next steps

1. **Seeds on `sqnomod`, `redmod`, `redmod_q64`** — the whole account rests on that contrast and
   it is n=1. Deferred by agreement, not by oversight.
2. ~~**`sq` trained with an auxiliary dense-division loss.**~~ — done; see
   [`exact_atom/`](exact_atom/README.md) §4. The auxiliary installs a reduce that reads 0.953
   on the inputs `x^2` supplies, but the composed map stays at floor because the *multiply*
   cannot construct the state to hand it. The `{div aux} × {seam closure}` 2×2 is null as posed.
3. **Longer budgets on the many-moduli arms**, which is the only way to put a height on §4's
   barrier rather than a lower bound. Still open, and still the axis nothing has moved:
   `exact_atom/` §3's attempt to widen the rule set — legal because division needs neither a
   factorisation nor a periodicity margin, so `N` can be drawn from all 900 three-digit
   integers rather than ~178 semiprimes — did not train.
4. ~~**Read the reduce out from inside a trained `sq`**~~ — done; see
   [`exact_atom/`](exact_atom/README.md) §4's oracle-seam probe, which is
   `ballistic_depth` §8's cold-start probe moved to the multiply/reduce seam.
5. **Seeds.** `exact_atom/` is single-seed throughout, as is this cut.
