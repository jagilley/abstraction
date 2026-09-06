# Staged reduce — decomposing the reduce until its stages are coverable

**Up**: [../README.md](../README.md) (rule_acquisition) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 15 arms across 4 cuts; the headline contrast is 2 seeds. **Date**: 2026-08-10.

---

## One-liner

[`exact_atom/`](../exact_atom/README.md) §5 localised the reduce's failure to the **quotient
range**, which `x^2` spans in full by construction, and showed a bounded-quotient reduce is fine
at both widths tested. This node applies the decomposition that fact implies: schoolbook long
division, so each stage is a bounded-quotient reduce, chained **at test time only** with the
remainder re-grounded through the model's own decode. That is
[`ballistic_depth/`](../../ballistic_depth/README.md) §9's re-projection one level down, and
because the remainder is a digit string the snap is exact rather than a projection onto a
learned manifold.

Three things follow.

1. **The decomposition works, at both widths.** Step-matched at 600k, 3-digit: staged
   **0.9968** against monolithic 0.4841. At 4 digits — `exact_atom` §5's scissors cell, where
   the monolithic control reads **0.0005** and so reproduces `div4_qfull`'s 0.0004 — staged
   reads **0.9899**.
2. **It is generalisation, not coverage.** On chains where *no* stage query was ever trained
   on (`k = 0`, rejection-sampled), the staged arm reads **0.9925** against 0.9971 overall.
3. **The rule axis moves for the first time in this program** — but only as an *interaction*.
   Held-out `N` goes from at-or-below floor to **0.9826 / 0.9790** across two seeds. Staging
   alone does nothing for it (8 moduli, same family: **0.0042** against floor 0.0022) and
   breadth alone does nothing (142 moduli, monolithic: 0.0025). Only both together move it.

No cell certifies Hard ladder rung `T=1`: the best `eps` is 3.2e-3 against the 9.0e-4 that rung
needs, i.e. `certifiable_T = 0.29`. One intermediate reading — that a 774-modulus arm
generalised *worse* than a 142-modulus one — is **retracted** as an artefact of a degenerate
held-out selector; see §4.

---

## Why this node exists

[`../README.md`](../README.md) §3 established that coverage of the reduce's input space controls
whether the reduce generalises, and [`exact_atom/`](../exact_atom/README.md) §5 narrowed the
residual failure to the quotient range: `div4_q64` reads 0.986 at 4 digits while `div4_qfull`
reads 0.0004 at the same width and budget. The composed task cannot avoid the hard case, because
`x^2` for `x < N` spans the full quotient range by construction.

Long division is the decomposition that fact implies, and it needs no trapdoor — radix digits and
intermediate remainders come from `(N, y)` alone, so the construction is legal for a real
submission. `exact_atom`'s next-step 2 named it; this node runs it.

The second motivation is the **rule** axis. Monolithic divide-by-`N` is a different function for
every modulus, which is a candidate explanation for why 8 moduli yield 8 specialised routines. A
single quotient-digit stage is closer to "compare against a small multiple of `N`, subtract" —
much more nearly the *same* function across moduli. That was pre-registered as a prediction
before the runs.

## Design

Writing `y` in radix `R` as `c_{S-1} … c_0` and carrying a remainder `r < N`:

```
r ← (r · R + c_i) mod N,   i = S-1 … 0
```

Every stage is a reduce whose dividend is `< R·N`, i.e. **quotient ≤ R−1** — exactly the
bounded-quotient problem that already generalises. The stage's input space is `R·N` per modulus
instead of `N^2`; at `R=10, w=3` that is 100× smaller.

Four properties of the setup are load-bearing:

- **Test-time only.** The model is trained *solely* on the single-stage distribution (uniform
  `y' ∈ [0, R·N)`, labels `y' mod N`). No intermediate supervision, no architecture change, no
  retraining for the chain. This isolates the decomposition from any extra training signal.
- **The monolithic control is the `S=1` endpoint of the same sweep.** `radix=0` runs the same
  code path over `[0, N^2)` with identical prompt layout, architecture, parameter count
  (2,122,782 at w=3), budget and **bit-identical eval pools**. It is not a separately configured
  baseline, and it reproduces `exact_atom`'s `divqfull` / `div4_qfull` at matched budget.
- **A compute-matched control.** `*_inner6` / `*_inner8` applies the tied operator `S` times per
  query with no re-grounding, separating "more serial compute" from "re-grounding at the seam".
- **Modulus-uniform readouts.** Chain pools draw `y ~ U[0, N^2)` with equal draws per modulus,
  matching training's weighting rather than the exhaustive enumeration's `N^2` weighting —
  `exact_atom`'s gotcha, sidestepped by construction rather than corrected afterwards.

`radix` sweeps the stage-count/per-stage-difficulty tradeoff, the direct analogue of §9's
re-projection period `k`.

## 1. The decomposition works at both widths

Chain accuracy on held-out `(N, y)`, `n = 65536`, 8 moduli, step-matched at 600k except where
noted. `certifiable_T = ln2/(768·eps)` is the ladder-rung consequence.

| arm | `S` | steps | chain held-out `y` | `eps` | `certifiable_T` | floor |
|---|---|---|---|---|---|---|
| `sr3_r10` | 6 | 600k | **0.9968** | 3.16e-3 | 0.29 | 0.0018 |
| `sr3_r100` | 3 | 600k | 0.9937 | 6.26e-3 | 0.14 | 0.0018 |
| `sr3_r2` | 20 | 125k *(partial)* | 0.9907 | 9.32e-3 | 0.10 | 0.0018 |
| `sr3_mono_inner6` | 1 | 600k | 0.7754 | 0.225 | 0.004 | 0.0018 |
| `sr3_mono` | 1 | 600k | 0.4841 | 0.516 | 0.002 | 0.0018 |

At 4 digits, `exact_atom` §5's scissors cell:

| arm | `S` | steps | chain held-out `y` | `eps` | floor |
|---|---|---|---|---|---|
| `sr4_r10` | 8 | 200k | **0.9899** | 1.01e-2 | 0.0002 |
| `sr4_r100` | 4 | 200k | 0.7947 | 0.205 | 0.0002 |
| `sr4_mono` | 1 | 600k | 0.0005 | 0.9995 | 0.0002 |

`sr4_mono` reproducing `div4_qfull`'s 0.0004 at matched budget is what makes the w=4 row
readable: the control fails exactly as the parent cut's independently-configured arm did, so the
staged number is being read against a known quantity.

**The compute control is not null, and the result should not be stated as though it were.** Six
serial operator applications without re-grounding move `eps` 0.516 → 0.225 (2.3×). Re-grounding
then moves it a further 71× to 3.16e-3. The defensible claim is that re-grounding *dominates*
compute, not that compute does nothing. Two caveats cut in opposite directions and are worth
recording: `inner6` gets 6× operator applications but only *one* encoder pass, where the staged
chain pays 6 full encode+operator passes; and its 6× applies at training time too, so it is not
FLOP-matched at equal gradient steps. It is best read as a lower bound on what compute alone
buys, which is the conservative direction.

**Both monolithic controls over-train.** `sr3_mono` peaks at 0.5146 (250k) and ends at 0.4841;
`sr4_mono` peaks at 0.0006 (100k) and ends 0.0005. This reproduces `ballistic_depth` gotcha 4,
`exact_atom` §3's `divq8`, and
[`metering_sweep`](../../../rhm/directed_sculpting/full_loop/metering_sweep/README.md) on a
further substrate. Peak-vs-peak does not change any ordering here.

**More stages is better within the range tested**, monotonically at matched budget
(R=100 → R=10: 0.9937 → 0.9968), which is what a coverage account predicts. `sr3_r2` (S=20) was
cancelled at 125k for compute and is not comparable; it is reported as a partial.

## 2. The result is generalisation, not coverage

About half of a staged chain's stage queries land in the arm's own training half
(`stage_seen_frac ≈ 0.49`). That is genuinely ambiguous rather than simply a confound — if the
claim is *decompose until every atom's input space is coverable*, stage queries landing inside the
covered set is the mechanism, not a cheat — but it means "staged beats monolithic on held-out
`y`" is not clean as stated. `reprobe.py` resolves it by stratifying on `k`, the number of stage
queries on the true stage path falling in the training half. `k` is a property of `(N, y)` and the
hash alone, so it is model-independent.

| arm | `k=0` (rejection-sampled) | overall | `k=0` floor | `k=0` pool |
|---|---|---|---|---|
| `sr3_r10` | **0.9925** | 0.9971 | 0.0141 | 65536 (7.3M draws) |
| `sr3_r10_many` | **0.9950** | 0.9942 | 0.0229 | 65536 (8.4M draws) |
| `sr3_r100` | 0.9937 | 0.9938 | 0.0041 | 65536 |
| `sr4_r100` | 0.8156 | 0.7941 | 0.0006 | 65536 |
| `sr3_mono` | 0.4812 | 0.4825 | 0.0020 | its whole pool (`S=1`) |
| `sr3_mono_inner6` | 0.7750 | 0.7736 | 0.0020 | its whole pool |

The `k`-gradient is small at 8 moduli (0.9921 at `k=0` rising to 1.000 at `k=6`) and **absent** at
142 moduli (0.9939–0.9962, non-monotone). For the monolithic arms `S=1`, so their ordinary
held-out pool *is* their `k=0` pool and the comparison is apples to apples. `P(k=0) = 2^-S`, so
this is reachable at `S ≤ 8` and not at `S = 20`.

**The chain beats its own `p^S` prediction when `p` is high and loses to it when `p` is low.**
`sr3_r10` reads 0.9971 against stage`^S` = 0.9867; the sign is the same on `sr3_r100`
(0.9938/0.9854), `sr3_r10_many` (0.9942/0.9853) and `sr4_r100` (0.7941/0.7717). But on held-out
`N`, where the per-stage rate is poor, `sr3_r10` reads 0.0025 against `p^S` = 0.0239 — 10× *worse*
than independent failure. Both directions have the same explanation: the stage inputs a chain
actually visits are not the uniform stage distribution it trains on, and a single early stage
error puts the remainder out of range and poisons everything downstream. So the chain model is a
useful bookkeeping device, not a calibrated predictor, and its error grows as `p` falls.

## 3. The rule axis: an interaction, not a main effect

Chain accuracy on held-out **`N`** — moduli never trained on — at 600k, step-matched within each
column.

| | 8 moduli | 142 moduli |
|---|---|---|
| **monolithic** (full quotient) | 0.0012 *(floor 0.0016)* | 0.0025 *(floor 0.0020)* |
| **staged** (`R=10`, quotient ≤ 9) | 0.0042 *(floor 0.0022)* | **0.9826** *(floor 0.0020)* |

Two seeds on the headline cell: **0.9826** (seed 0) and **0.9790** (seed 1), per-modulus minimum
0.889 and 0.851 across the 36 held-out moduli, median 0.993 and 0.987. It works on every unseen
modulus in the pool, not a favourable subset. The prior best on this axis anywhere in the program
is `redmod_q64_many`'s 0.058 against floor 0.015.

**Neither factor alone does anything.** `sr3_r10_m8` is the control that establishes this: eight
moduli subsampled from the *same* 178-modulus family the `_many` arms use, so only the count
varies. It reads **0.0042** on held-out `N` — at floor, like the original 8-modulus arm — while
reading **0.9943** on held-out `y`. Same model, same family, same budget: it learns staged
division essentially perfectly and still cannot transfer it to an unseen modulus. Symmetrically,
breadth without staging moves held-out `N` only 0.0012 → 0.0025.

This control was run because the original 8-vs-142 contrast confounded modulus *count* with the
family: the 8-modulus arms draw from `min_margin=20` (59 moduli, range 177–995) and the `_many`
arms from `min_margin=2` (178 moduli, range 111–995). The margin is a repeated-squaring constraint
and should be irrelevant to division, but that was an assumption rather than a measurement. It
now is one.

The reading these three cells support — offered as an interpretation, not as something the design
isolates — is that staging does not teach the rule. It changes the function being learned into one
that *has* a shared form across moduli, and breadth is what supplies the evidence that the form is
shared. Neither half of that is sufficient alone.

**One caveat on the comparison**: `max_moduli=8` also subsamples the held-out pool, so
`sr3_r10_m8` scores on 32 held-out moduli against `_many`'s 36 (floors 0.0022 vs 0.0020). Same
family and overlapping draw, but not an identical pool.

## 4. The dense-modulus arm, and a retracted reading

`sr3_r10_dense` trains on 774 three-digit moduli drawn from all integers rather than the ~178
semiprimes — legal because division needs neither a factorisation nor a periodicity margin, and
the attack on the rule axis that `exact_atom` §3's `divqfull_densen` failed to run (it never
trained). Here it trains fine (`chain held-out y` 0.9912) and reads **0.2907** on held-out `N`.

**An intermediate reading of that number — that 774 moduli generalise worse than 142, i.e. that
rule-set breadth has a ceiling — is retracted.** Its per-modulus distribution is sharply bimodal
(median 0.0097, max 1.000) and its chain beats its own `p^S` by 7.4×, both signatures of a
population split rather than uniform mediocrity. `dense_probe` in `reprobe.py` scores the
dense-trained and the semiprime-trained model on one shared pool:

| | the 36 semiprimes | the other 90 | all 126 |
|---|---|---|---|
| `sr3_r10_dense` (774 moduli) | **0.9888** | 0.0094 | 0.2892 |
| `sr3_r10_many` (142 moduli) | 0.9822 | 0.0064 | 0.2852 |
| floor | 0.0020 | 0.0026 | 0.0024 |

On identical moduli the 774-modulus arm is **marginally better** than the 142-modulus one.
Breadth did not hurt.

**The cause is a degenerate held-out selector.** The dense split uses
`(N * 2246822519) % 10 == 0`, intended as a pseudo-random 10% hash. Since `2246822519 ≡ 9 (mod 10)`
and 9 is invertible mod 10, that predicate is *exactly* `N ≡ 0 (mod 10)`. So 90 of the 126
held-out moduli are the one final-digit class the training set could never contain, and
`(36 × 0.989 + 90 × 0.009)/126 = 0.289` reproduces the headline number exactly. **This is scoped to
the `dense_n` arms**: every other arm takes its held-out moduli from `ModulusFamily.split()`, and
the `(N, y)` problem split uses a different, non-degenerate hash.

What the bug exposed is more interesting than what it broke. Transfer is **uniform across every
structural axis that varies inside the trained family** — minimum prime factor 3 / 5 / 7 / ≥11
reads 0.989 / 0.992 / 0.992 / 0.985, and magnitude is flat (0.222 / 0.349 / 0.285 across
`N < 400` / `400–700` / `≥ 700`, entirely tracking how many multiples of ten fall in each band) —
and collapses only on the digit class absent from training. The dense arm *did* train on even
moduli (102, 104, … everything except multiples of ten), so this is not "cannot divide by even
numbers": it fails specifically where the modulus's final digit is one it never saw. Within these
scales the learned rule generalises across arithmetic structure while remaining keyed on surface
digit patterns tightly enough that an unseen final digit breaks it. Single arm, one
digit class — suggestive, not established.

## Predictions

Recorded because they were made before the runs.

1. **Staging would beat monolithic at both widths** — held. The margin at w=4 (0.9899 vs 0.0005)
   is larger than at w=3.
2. **Staging would transfer across moduli better than monolithic** — held, but the prediction as
   stated was wrong about *why*. It was expected to work because each stage is nearly the same
   function across moduli; that is necessary but not sufficient, since the 8-modulus staged arm at
   the same family is at floor. The effect needs breadth as well, which was not predicted.
3. **More moduli would help monotonically** — this was the reading of the dense arm before the
   probe, and it is neither confirmed nor refuted. On matched moduli 774 ≥ 142; whether breadth
   keeps helping past 142 is untested, because the dense arm's held-out pool cannot answer it.
4. **The compute control was expected to be near-null.** It is not — `inner6` recovers roughly a
   third of the gap in `eps` terms.

## Honest caveats

- **No rung is certified.** Best `eps` is 3.16e-3 against rung `T=1`'s 9.0e-4, so
  `certifiable_T = 0.29`. Depth remains the cheap axis and rung zero the expensive one; this node
  moves the rule axis, not the ladder.
- **Three arms are partial** and are reported as such: `sr3_r2` (125k), `sr4_r10` in tag `sr4a`
  (150k, superseded by the complete `sr4a3` at 200k), `sr3_r10_dense` in tag `srb2` (100k,
  superseded by `srb2r` at 600k). All were stopped deliberately for compute.
- **The w=4 staged arms ran 200k against the monolithic control's 600k.** The asymmetry
  disadvantages the staged arm, so it cannot manufacture the effect, but the cells are not
  step-matched. The w=3 cells are.
- **`sr3_r10_m8` scores on 32 held-out moduli against `_many`'s 36**, as above.
- **The chain model is not calibrated** (§2), so `p^S` should not be used to project a
  per-stage rate onto a composed one without measuring the composed number.
- **The dense selector is degenerate** (§4) and is not fixed in the code; the finished runs are
  left reproducible rather than edited. Any new `dense_n` arm needs a real hash first.
- **One substrate, two digit widths, one architecture.** Whether the decomposition-plus-breadth
  interaction is a fact about long division or something more general is untested here.
- **No submission was built.** As in every cut in this experiment, `T` never enters the prompt and
  the answer encoding is fixed-width, so these are not upstream-comparable scores.

## Reproduce

```bash
cd experiments/
E=one_layer_deeper/rule_acquisition/staged_reduce/staged_reduce.py::staged_reduce
R="MODAL_PROFILE=chromatic modal run --detach"

# §1 — H1 at 3 digits, step-matched at 600k
$R $E --tag sr3a  --arms "sr3_r10,sr3_r100"          --steps 600000 --seed 0
$R $E --tag sr3b  --arms "sr3_mono,sr3_mono_inner6"  --steps 600000 --seed 0
$R $E --tag sr3c  --arms "sr3_r2"                    --steps 600000 --seed 0   # stopped at 125k

# §1 — 4 digits, exact_atom §5's scissors cell
$R $E --tag sr4a3 --arms "sr4_r10"   --steps 200000 --seed 0
$R $E --tag sr4a2 --arms "sr4_r100"  --steps 200000 --seed 0
$R $E --tag sr4b  --arms "sr4_mono"  --steps 600000 --seed 0

# §3 — the rule axis, and the interaction
$R $E --tag srb1     --arms "sr3_r10_many,sr3_mono_many" --steps 600000 --seed 0
$R $E --tag srb1r_s1 --arms "sr3_r10_many"               --steps 600000 --seed 1
$R $E --tag sr3m8    --arms "sr3_r10_m8"                 --steps 600000 --seed 0

# §4 — the dense-modulus arm
$R $E --tag srb2r --arms "sr3_r10_dense" --steps 600000 --seed 0

# checkpoint probes; train nothing, ~1 min each
MODAL_PROFILE=chromatic modal run \
  one_layer_deeper/rule_acquisition/staged_reduce/reprobe.py::reprobe \
  --tag sr3a --arms "sr3_r10,sr3_r100"
MODAL_PROFILE=chromatic modal run \
  one_layer_deeper/rule_acquisition/staged_reduce/reprobe.py::dense_probe \
  --tag srb2r --arm sr3_r10_dense --ref-tag srb1 --ref-arm sr3_r10_many

# fetch + analyse
for t in sr3a sr3a2 sr3b sr3c sr3m8 sr4a2 sr4a3 sr4b srb1 srb1r_s1 srb2r; do
  MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
    /staged_reduce/$t/results_seed0.json \
    one_layer_deeper/rule_acquisition/staged_reduce/results/$t/results_seed0.json --force
done
python3 one_layer_deeper/rule_acquisition/staged_reduce/analyze.py --tags <comma-separated>
```

Modal volume `one-layer-deeper-data`; results at `/staged_reduce/<tag>/results_seed<N>.json`,
checkpoints at `/staged_reduce/<tag>/ckpt/<arm>_seed<N>.pt`, probes at
`/staged_reduce/<tag>/reprobe_seed<N>.json` and `/staged_reduce/<tag>/dense_probe_seed<N>.json`.

**Gotchas.** Arms run *sequentially* inside a job, so a two-arm 600k job is ~3h against ~90min for
singles, and killing a job mid-run forfeits any arm that has not started. `persist()` writes
results, checkpoint and a `volume.commit()` at **every** log point with a `complete` flag, so
stopping early keeps the full `eps`-vs-budget curve — partial arms are recoverable, not lost. Each
tag rewrites its own `results_seed<N>.json` from the in-process dict, so relaunching a second arm
under an existing tag overwrites the first arm's curve; use a new tag and reassemble with
`analyze.py --tags`. Local launcher shells are reaped well before these jobs finish: use
`--detach` and detect completion by fetching the result file. `modal volume get` needs `--force`
and a full per-file destination path.

## Child — [`terminal_only/`](terminal_only/NOTES.md): the decomposition is not discoverable from terminal labels

Run 2026-08-22/23 for the competition question (written up in
[`dress_rehearsal/second_pass/`](../../dress_rehearsal/second_pass/README.md) §3): the same task,
pools and parameter count as `sr3_mono*`, but the forward pass is `S=6` applications of one tied
stage map with the remainder re-grounded through the model's own decode, and the loss is on the
final remainder only. All four cells (8/142 moduli × straight-through/soft seam) sit at floor with
CE flat from 25k to 600k where `sr3_mono` descends on identical data; the seam collapses to ~12
distinct values; `restart`/`depth`/`stagefn` probes are flat, including at `q=0`. A degenerate
fixed point: the staging has to be *installed* (by the single-stage distribution), and this node's
interaction (staging × breadth) does not self-organise from the terminal label.

## Next steps

1. **Fix the dense selector and re-ask the breadth question.** `(N * 2246822519) % 10 == 0` is
   `N ≡ 0 (mod 10)`. With a real hash, "does breadth keep helping past 142 moduli" becomes
   answerable; right now §4's arm cannot answer it in either direction.
2. **The final-digit result deserves its own cut.** Holding out a digit class is a sharp probe of
   what the learned rule is keyed on, and §4 found it by accident on one arm. Held-out last digit,
   held-out leading digit and held-out length are the obvious three.
3. **Where does breadth become sufficient?** The interaction is established at 8 vs 142; a ladder
   (24, 48, 96) would distinguish a threshold from a gradient, which bears directly on whether
   "breadth supplies evidence that the form is shared" is the right reading.
4. **Close the exactness gap.** `certifiable_T = 0.29` at `eps = 3.16e-3`, and the staged arms
   were still improving. Whether a budget or a radix reaches 9.0e-4 is untested — though
   `exact_atom`'s over-training results, reproduced here on both controls, are a reason to expect
   this to need more than steps.
5. **Compose it.** Everything here is the reduce in isolation. `exact_atom` §5's scissors had two
   blades; this node closes the reduce one at 4 digits, and whether the composed atom now works at
   that width is the direct follow-up.
