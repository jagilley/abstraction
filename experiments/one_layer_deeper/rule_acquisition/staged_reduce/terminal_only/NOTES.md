# NOTES — `staged_reduce/terminal_only`

Running log. No `README.md` until the numbers have been discussed (repo convention).

## The question

`staged_reduce`'s result is not portable to the competition, because the single-stage training
distribution (uniform `y' ∈ [0, R·N)`, label `y' mod N`) is **self-made**. The only supervision
the benchmark hands you is the terminal label, plus label-free self-consistency of the model's
own states. Every composed arm in this program trained terminal-only sits at floor — but every
one of those had a *monolithic* reduce.

So: **can a staged forward pass self-organise the decomposition from terminal labels alone,
given breadth of moduli?**

## Design

Task identical to `sr3_mono`: `(N, y) → y mod N`, `y ~ U[0, N²)`, loss on the final remainder
only. Forward pass is `S = 6` applications of one tied stage map on the donor's prompt, with
the remainder re-grounded at each seam through the model's own decode. Because `R = 10^m`,
`r·R + c` is `r`'s digit string followed by `c`'s, right-aligned in the `2w` field, so the
re-encode is exact and in place at the digit level.

The seam is a genuine bottleneck in the `st` mode: the *only* channel from one stage to the
next is `w = 3` decimal digits (1000 values), against a prefix `y // R^i` that can be five
digits. It cannot be a copy of the prefix, so whatever the model carries has to compress the
prefix — the question is whether it compresses it to the residue, to a relabelling of the
residue, or to something else that still works.

Architecture, parameter count (2,122,782), family, hash split, budget and **eval pools** are
the donor's, verified bit-identical at launch:

| | 8 moduli | 142 moduli |
|---|---|---|
| stage `heldout_y` / `seen_y` / `heldout_n` | 23688 / 23692 / 42826, floors 0.1001 / 0.0999 / 0.1001 | 262144 / 65536 / 106528, floors 0.0997 / 0.1006 / 0.1000 |
| chain `heldout_y` / `seen_y` / `heldout_n` | 65536 / 32768 / 65536, floors 0.0018 / 0.0017 / 0.0016 | 65536 / 32768 / 65536, floors 0.0027 / 0.0029 / 0.0020 |

which match `sr3_r10` / `sr3_mono` and `sr3_r10_many` / `sr3_mono_many` exactly.

## The cells this reads against

| | 8 moduli | 142 moduli |
|---|---|---|
| monolithic, terminal label (`sr3_mono*`) | 0.4841 / **0.0012** | 0.2527 / **0.0025** |
| + 6 inner ops, no re-grounding (`sr3_mono_inner6`) | 0.7754 / 0.0017 | — |
| staged, **single-stage supervision**, chained at test time (`sr3_r10*`) | 0.9968 / 0.0042 | 0.9943 / **0.9826** |

(held-out `y` / held-out `N`; `sr3_r10`'s 8-modulus held-out-`N` cell is `sr3_r10_m8`'s 0.0042
on the family-matched draw, 0.0025 on `min_margin=20`.) `sr3_r10_many`'s stage map reads
**0.9975** held-out `y` and **0.9918** held-out `N` on the bounded-quotient pools — the number
the `stage_all` probe here is read against.

## Runs

### Round 1 — the 2×2, 3 digits, `R=10` (`S=6`), 600k steps, seed 0 (launched 2026-08-22)

| tag | arm | moduli | re-ground | Modal app |
|---|---|---|---|---|
| `to3a` | `to3_st10` | 8 | straight-through hard | `ap-c3jokgQJ41eoLquGgMtor4` |
| `to3b` | `to3_soft10` | 8 | soft | `ap-Z2zw5B5VKby8EahyYjfwUk` |
| `to3c` | `to3_st10_many` | 142 | straight-through hard | `ap-jym9TFJJg9SwmFUgEp6JpN` |
| `to3d` | `to3_soft10_many` | 142 | soft | `ap-T1nsfKJtVuuk6rTF902CCr` |

~22 step/s on an L4 (the donor's `S=1` arms run ~110), so ~7.5–8 h per arm at 600k.

Smoke (`smoke_to`, `smoke_to2`, both discarded): all four code paths run; the seam-`cos` path
runs at 10.7 step/s (it pays a second encoder pass per seam) and is not in round 1.

**Reads (2026-08-22).** All four arms **at floor, and flat from the first log point (25k)**.

| arm | moduli | mode | chain held-out `y` | floor | chain held-out `N` | floor | `stage_all` | floor |
|---|---|---|---|---|---|---|---|---|
| `to3_st10` | 8 | st | 0.0034 | 0.0018 | 0.0013 | 0.0016 | 0.0025 | 0.100 |
| `to3_soft10` | 8 | soft | 0.0033 | 0.0018 | 0.0019 | 0.0016 | 0.0025 | 0.100 |
| `to3_st10_many` | 142 | st | 0.0052 | 0.0027 | 0.0022 | 0.0020 | 0.0035 | 0.100 |
| `to3_soft10_many` | 142 | soft | 0.0053 | 0.0027 | 0.0026 | 0.0020 | 0.0036 | 0.100 |

Against `sr3_mono` 0.4841 / 0.0012, `sr3_r10` 0.9968 / 0.0025, `sr3_mono_many` 0.2527 / 0.0025,
`sr3_r10_many` 0.9943 / **0.9826**. The soft arms' `chainhard_*` reads are identical to their
native reads (0.0033 / 0.0052), so nothing was hiding in a continuous seam.

**The task was not learned at all, and the staged forward is worse than the monolithic control
on the identical task.** Training CE is flat at 2.07 (8 moduli) / 1.93 (142) from 25k to 600k
against `ln 10 = 2.303`, where `sr3_mono` descends 1.467 → 0.541 and `sr3_mono_many`
1.921 → 0.766 on the same data. Every tail slope is 0.00 — this is a dead optimisation, not an
undertrained one.

**The seam collapsed to a tiny codebook.** Per-stage diagnostics: the decoded intermediate takes
**12 distinct values** across the whole 8-modulus pool (~90 across 142 moduli, i.e. well under
one per modulus) against the ~600 a residue needs; `true_match` 0.002–0.027 at every stage;
purity barely above its within-modulus permutation null (0.977 vs 0.935); `inv_purity` 0.033 vs
null 0.026. Not a relabelled residue — an almost contentless channel. The `st` and `soft` arms
have near-identical CE curves to three decimals at 142 moduli, which is what you expect once the
seam carries nothing: the gradient through it vanishes and both reduce to the same
`(N, c_0)`-only predictor.

**It is the stage map that is broken, not the propagation.** `reprobe` on `to3c`: `restart` is
flat in `j` (0.0054 → 0.0050 across `j = 0…5`), so injecting the *true* partial remainder
immediately before the last stage buys nothing; `depth` is flat (`S+0` 0.0054 → `S+4` 0.0054),
so the extra leading-zero stages are neither no-ops nor harmful because nothing is happening;
and `stagefn` is flat at 0.0036 across **every** quotient digit `q = 0…9`, including `q = 0` —
the map does not even return `y'` when `y' < N`, the free no-reduction case the whole program
floors against. Teacher-forced stage accuracy is 0.0028–0.0114.

**Scope note.** The re-encode is provably positionally exact — in `st` mode the soft overwrite is
a bit-level no-op against the hard token ids (`digits_of(r·10+c, 6)` puts `r`'s digits at field
indices 2,3,4 = absolute 9,10,11 = `r_slots=[9,12)`), so the null is not a prompt-construction
bug. `crossprobe` was not run: the pools are already bit-identical to the donor's by
construction and verified at launch, so it could add nothing to the comparison.

## Reproduce

```bash
cd experiments/
E=one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/terminal_only.py::terminal_only
R="MODAL_PROFILE=chromatic modal run --detach"

$R $E --tag to3a --arms "to3_st10"        --steps 600000 --seed 0
$R $E --tag to3b --arms "to3_soft10"      --steps 600000 --seed 0
$R $E --tag to3c --arms "to3_st10_many"   --steps 600000 --seed 0
$R $E --tag to3d --arms "to3_soft10_many" --steps 600000 --seed 0

# checkpoint probes; train nothing
MODAL_PROFILE=chromatic modal run \
  one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/reprobe.py::reprobe \
  --tag to3c --arms "to3_st10_many"
MODAL_PROFILE=chromatic modal run \
  one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/reprobe.py::crossprobe \
  --tag to3c --arm to3_st10_many --ref-tag srb1 --ref-arm sr3_r10_many

# fetch + analyse
for t in to3a to3b to3c to3d; do
  MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
    /staged_reduce/terminal_only/$t/results_seed0.json \
    one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/results/$t/results_seed0.json --force
done
python3 one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/analyze.py \
  --tags to3a,to3b,to3c,to3d
```
