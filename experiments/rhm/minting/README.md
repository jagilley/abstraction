# Minting against a non-mirror verifier: a support oracle is not a density oracle

**Status**: built and run. Headline contrast is **3 seeds, paired within seed**; the regime
diagnostics and the two failed-dose runs are reported as design evidence, not
results. **Date**: 2026-07-30.
**Up**: [../README.md](../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/breadth_as_grader_heterogeneity.md`](../../../ideas/breadth_as_grader_heterogeneity.md)
— this builds its §7 and tests its §5; it does **not** test its §4 (see [§7](#7-what-this-does-not-establish)).
**Anchors**: [`../specialization/README.md`](../specialization/README.md) Exp 1 (the `FULL`/`SUBTREE`
breadth control, the NTP-floor/oracle-aux ladder) — reproduced here as an external reference, not
used as an internal control.

---

## One-liner

A learner seeded on a narrow slice of the tree mints its own sequences and folds the accepted ones
back into its curriculum. **Acceptance type matters a lot and buys nothing.** Filtering by the true
rule table is worth **+0.044 ± 0.008 at d5 against no filter** (t = 5.47, 3/3 seeds) and
**+0.034 ± 0.011 against own-likelihood** (t = 3.22, 3/3) — but the range those contrasts span runs
from *harmful* to *break-even*, not from worse to better. Every unverified minting arm lands
significantly **below** the no-minting control, the verified arm lands on it (+0.006 ± 0.004, t=1.7),
and real held-out data at identical volume lands **+0.106 ± 0.005** above it. §7's own falsifier —
*"falsified if the verifier arm from a narrow seed does not exceed the narrow static control"* — is
**met**.

The mechanism is the part worth carrying. The verified pool is 100% legal **and** 45–50% of it has
root sets lying entirely outside the seed's roots, so support genuinely expands — and capability does
not follow. **The rule table answers "is this in the support of the true distribution", not "is it
drawn with the right probability."** Next-token training is density matching, so a support oracle can
delete out-of-support mass (which is exactly why it prevents the damage the other arms take) and
cannot correct in-support density error (which is exactly why it produces no gain).

---

## 1. What was built

§7 specifies a loop: seed a learner on a bounded corpus, have it **mint** candidate sequences, accept
some, fold them into the training pool, repeat. The load-bearing variable is *what accepts*. The arms
were designed as an ordered axis rather than §7's two-way contrast, because §5 states the
mirror criterion as a dichotomy while its own open question asks whether it admits degrees:

| arm | acceptance rule | what it is |
|---|---|---|
| `static` | — | bounded seed, no minting. The floor. |
| `mirror` | the learner's own mean token log-likelihood | perfect mirror; zero information beyond the dense loss |
| `mirror_dedup` | mirror, exact repeats dropped first | separates mirror *geometry* from mere replay |
| `peer` | a frozen model trained on **disjoint** roots | learned, unprivileged, sighted where the learner is blind |
| `verifier@4` | parses only to level 4 | a graded non-mirror grader |
| `verifier` | the exact rule table, to the root | privileged — it *is* the DGP |
| `peer_verifier` | parses **and** top-scoring under the peer | legality composed with location |
| `random` | uniform | matched volume, no grader at all |
| `real_data` | — (genuine held-out sequences at the same quota) | the ceiling |
| `broad:static` | — | the seed-breadth reference (`FULL` roots) |

**Controlled**: one rule table, one initialisation, one warm checkpoint per breadth (all arms fork
from it), one total step budget (8000 warmup + 8 × 1500 = 20000, matching Exp 1), one candidate
budget, one acceptance quota. Pools therefore grow by the same amount and only their **contents**
differ. The generative probe runs on a **frozen** held-out prefix set with its own per-round RNG, so
it is arm-comparable and cannot move because an arm's pool moved.

Two arms exist only because of confounds and both earned their place — see [§5](#5-two-controls-that-changed-the-conclusion).

### An exact parser was required

`rhm_data.parse_leaves` builds inverse maps last-writer-wins. At v=16, m=4 about 8 of 64 tuples per
level collide, and the wrong parent propagates upward: measured, it **false-rejects 99.98% of genuine
RHM sequences**. It could not have served as the verifier. `rhm_data.possible_set_parse` (added here,
purely additive) carries the full *set* of features that could have produced each node — a CYK parse
for this fixed-arity grammar — so validity is exact. Validation: genuine sequences **1.0000** valid,
uniform-random strings **0.00000**, single-token corruptions caught **98.5%** of the time.

### Minting is prefix-conditioned

Training is phase-diverse (random windows over the concatenated corpus, exactly Exp 1's objective), so
the model cannot know it is at sequence position 0. A prefix supplies the phase; `prefix_frac` is a
controlled variable shared by every arm, set to 0.5 so half the sequence is generated and the root
stays genuinely under-determined. At `prefix_frac` 0.75 yield roughly doubles but the prefix pins the
root, which would suppress the support expansion being measured.

## 2. Choosing the regime, because the specced one does not work

§7 specifies Exp 1's regime (v=16, s=2, L=6, m=4). It fails two independent requirements, and the
diagnostics are worth recording because they also say something about Exp 1.

**Mint yield.** At L6/m4 a *fully trained* model on Exp 1's unmetered 200k-sequence pool
(20k steps, val NTP **1.5586** against Exp 1's 1.553/1.561 — the harness reproduces the anchor)
mints at **4.7%** at half-prefix, and its **mean parse depth tops out at 3.28 of 6**. That is the
~d3.5 stall, visible for the first time in a *second, independent instrument*: self-generation
validity rather than probing. On a bounded pool the same regime overfits so badly that val NTP hits
3.61 — worse than uniform (2.77) — and yield collapses to 0.2%.

**Regime scorecard** (narrow `static` arm, L=6, 20000 steps):

| m | pool | d4 | d5 | d6 | val drift r0→r7 | val comp−S | mint yield |
|---|---|---|---|---|---|---|---|
| 2 | 65536 | 1.000 | 0.991 | 0.837 | +0.009 | **+0.206** | 0.478 |
| 3 | 32768 | 0.820 | 0.385 | 0.116 | **+0.349** | +0.043 | 0.061 |
| **3** | **65536** | **0.937** | **0.566** | **0.185** | **−0.001** | +0.038 | **0.144** |
| 4 | 65536 | 0.387 | 0.137 | 0.078 | +0.004 | −0.003 | 0.042 |

m=3 / pool 65536 is the only cell with depth headroom, no overfit drift, and enough yield for the
quota to bind. m=2 pins d1–d4 at ceiling; m=4 sits at the floor and starves the verifier; m=3/32768
overfits. This uses m=3 against `rhm/CLAUDE.md`'s "m=2 is too easy / m=8 barely learns" guidance,
which was written for the FM-residual measurements; here *the learner must be able to generate a
parseable sequence at all*, which is `specialization/README.md`'s own standing control ("run at an
(L, m, capacity) where the model demonstrably reaches the target level").

**Consequence: Exp 1's anchors do not transfer.** The static controls are therefore re-run in-harness
at matched compute rather than read off Exp 1, and the floor/ceiling pair (root 0.08 → 0.80) is cited
as an external reference only.

## 3. The dose brackets the design space

Two runs failed before the headline one, in opposite directions, and the bracket is a reusable
constraint rather than a wasted pair.

| dose | mechanism | outcome |
|---|---|---|
| minted pool = 11% of pool, uniform sampling | quota 1024 × 8 vs 65536 seed | acceptance type **invisible**; all arms within ±0.015; even `real_data` moves d5 by only +0.034 |
| minted pool gets 50% of the **gradient** | `mint_weight=0.5` on a ≤8192-sequence pool | **every arm degrades equally**, `real_data` included: d5 −0.177, d6 −0.302, arm spread ≤0.03 |
| minted pool at **parity by size** | quota 8192 × 8 = 65536, pooled sampling | `real_data` **+0.106**; arms separate |

The middle row is the informative failure: at 50% gradient share, *genuine held-out data* damages the
learner as much as 90%-junk data does. The harm is **distribution narrowing** from concentrating half
the gradient on a small pool, not autophagy and not data quality. So for minted data to carry real
gradient share without concentration, **the minted pool must rival the seed in size** — and since the
seed cannot shrink below ~65536 at m=3 without overfitting, the quota must rise 8×. Cranking a
weight cannot substitute for it.

## 4. Result — 3 seeds, m=3, pool 65536, quota 8192 pooled

Paired within seed against `narrow_static`, MLP best-over-blocks on the frozen full-tree probe.
d1–d3 saturate (≥0.978 everywhere) and are omitted.

| arm | Δd4 | Δd5 | Δd6 |
|---|---|---|---|
| `real_data` (ceiling) | +0.0119 ± 0.0020 | **+0.1060 ± 0.0046** | +0.0357 ± 0.0045 |
| `peer_verifier` | −0.0021 ± 0.0032 | +0.0104 ± 0.0025 | −0.0010 ± 0.0023 |
| `verifier` | −0.0004 ± 0.0008 | +0.0061 ± 0.0035 | +0.0004 ± 0.0030 |
| `peer` | −0.0044 ± 0.0013 | −0.0201 ± 0.0070 | −0.0161 ± 0.0034 |
| `mirror` | −0.0040 ± 0.0022 | −0.0276 ± 0.0110 | −0.0117 ± 0.0036 |
| `random` | −0.0037 ± 0.0010 | −0.0378 ± 0.0098 | −0.0135 ± 0.0016 |

### 4a. Acceptance type matters — from harmful to break-even

| contrast | Δd5 | t | seeds | Δd6 | t |
|---|---|---|---|---|---|
| `verifier − random` | **+0.0439 ± 0.0080** | 5.47 | 3/3 | +0.0139 ± 0.0026 | 5.40 |
| `verifier − mirror` | **+0.0337 ± 0.0105** | 3.22 | 3/3 | +0.0121 ± 0.0019 | 6.43 |
| `real_data − verifier` | +0.0999 ± 0.0051 | 19.41 | 3/3 | +0.0353 ± 0.0053 | 6.63 |
| `peer_verifier − verifier` | +0.0043 ± 0.0060 | 0.72 | 1/3 | −0.0014 ± 0.0053 | −0.26 |

The verifier arm is **conservative**: its pool ends at 125,759 against the others' 131,072 because
the quota bound at ~91% as validity dipped in later rounds, so it wins while training on ~4% *less*
data.

### 4b. Recovery against the ceiling

Share of `real_data`'s gain each grader buys back, at d5: `peer_verifier` **9.8%**, `verifier`
**5.8%**, `peer` −19.0%, `mirror` −26.1%, `random` −35.6%. At d6: `verifier` 1.2%, everything else
negative.

### 4c. The mechanism: support expands, capability does not

Final-round accepted pool, mean over 3 seeds:

| arm | rule-violation rate | support expansion (root set **entirely** outside S) |
|---|---|---|
| `real_data` | 0.000 | 0.000 *(genuine S-rooted data cannot leave S)* |
| `verifier` | 0.000 | **0.454** |
| `peer_verifier` | 0.000 | **0.497** |
| `peer` | 0.726 | 0.168 |
| `mirror` | 0.664 | 0.166 |
| `random` | 0.897 | 0.049 |

The verified pool is legal and genuinely reaches parts of the tree the seed never contained — nearly
half of it — and this converts to ~nothing at the probes. **§2's "minting defeats the cap" holds at
the level of support and fails at the level of capability.**

The reading the numbers force: the rule table is a **support oracle**, not a **density oracle**. It
certifies membership in the support of the true distribution; it says nothing about the probability
with which a sequence should occur. NTP is density matching. So the filter can delete out-of-support
mass — which is why the zero-violation arms sit at break-even while the 66–90%-violation arms sit
0.02–0.04 below — and cannot add the mass the model is missing. Training on your own accepted samples
is approximately a fixed-point operation regardless of filter quality, because **a filter only ever
removes mass**. Real data is a density sample. That is the 17×.

This **sharpens** [metered-data](../../../ideas/meta_learning_under_metered_data.md) §3 rather than
confirming it: "the scarce resource is verified tokens" is not quite right, because verified
*self-generated* tokens are worth ~nothing here. What is scarce is information the model does not
already carry, and a support oracle cannot manufacture it. §7 calls the rule table *"does the code
run"* — and does-the-code-run is a support oracle, which is a statement that should survive leaving
RHM.

## 5. Two controls that changed the conclusion

- **`random` (matched volume, no grader).** §7's own confound note warns that mint-and-accept drifts
  toward the easy part of the band regardless of grader type. In the very first run `random` —
  accepting 39% rule-violating garbage — got the *best* val NTP of any narrow arm (2.0007 vs static's
  2.5891, better than the verifier's 2.0910). Without it, minting's val-loss gain would have read as
  verification working, when it was raw volume anti-correlated with validity.
- **`real_data` (the ceiling).** Added as a *quality* ceiling, it earned its keep twice as a **dose**
  diagnostic. At `mint_weight=0.5` it degraded as hard as the junk arm; without it that run would
  have read as "minting harms the learner and verification does not save it", which is exactly
  backwards.

**A retraction.** An earlier single-seed run at a bounded pool showed the mirror arm collapsing to
pure replay — duplication → 1.000 by round 3, 114 new sequences from 65,536 candidates generated — and
that was reported as §5 confirmed. It was not. `mirror` and `mirror_dedup` come out **identical, bit
for bit**, at a non-memorising pool, because duplication is then ~0 and the dedup step removes
nothing. The collapse was **overfitting**, not mirror geometry. §5's criterion is supported here by
the *type* contrasts in §4a, not by that collapse.

## 6. Open steps

1. **The density half is unbuilt, and it is the direct next experiment.** Everything above says a
   support oracle cannot supply density information. The test: an acceptance rule that carries
   density information — e.g. importance-weighting accepted sequences toward the *true* conditional,
   or accepting on a statistic of the true rule-choice distribution rather than mere legality. If
   §4c's reading is right, a density-aware grader should recover materially more of `real_data`'s
   +0.106 than the verifier's 5.8%. If it does not, the ceiling is about self-generation itself and
   not about what the filter certifies.
2. **A learned verifier.** The rule table is the exact DGP — the strongest support oracle possible,
   which §7's own caveats flag. 5.8% recovery is therefore an *optimistic* bound on what a learned
   verifier buys. Worth measuring before the number is quoted anywhere.
3. **The breadth half needs cut-3, not a better run** (see §7 below).
4. **Push the dose past parity.** The bracket in §3 is 11% / 50%-concentrated / parity-by-size. The
   parity cell is the first with dynamic range; whether the verifier's break-even becomes a gain, or
   the whole system degrades, at 2× or 4× the seed is unmeasured.
5. **Composition is unresolved rather than refuted.** `peer_verifier − verifier` is +0.004 ± 0.006
   (1/3 seeds) at d5, so [heterogeneous_graders](../../../ideas/heterogeneous_graders.md) §10's
   redundancy-vs-heterogeneity falsifier is **not** answered here. It also has the highest support
   expansion of any arm (0.497), so the composite does what it was designed to do at the pool level
   and the probes cannot see it. More seeds, or a regime where location matters more, would settle it.
6. **The peer/verifier crossover, seen once and not replicated.** In the quota-1024 runs
   the support-expansion ordering **inverted** with validity rate: at m=3 (validity 14%) `verifier`
   0.42–0.54 ≫ `peer` 0.13; at m=2 (validity 48%) `peer` 0.12 ≫ `verifier` 0.02, with
   `verifier ≈ random`. The suggested reading — the verifier gates *legality*, the peer gates
   *location*, and legality stops binding once it is cheap — is a hypothesis from two
cells, not a result.

## 7. What this does not establish

- **The breadth axis of §7 was never testable on this substrate.** `broad_static` is *below*
  `narrow_static` on the depth probes at m=3 (Δd5 −0.062, Δd6 −0.030), and the breadth
  gap appears only in val NTP. The cause is structural, not statistical: the probe is *fitted on the
  probe set*, so when deep structure is carried by shared rules the probe decodes it **even for roots
  the model never saw**. Root-restriction cannot produce a depth-probe deficit in homogeneous RHM —
  the same mechanism as cut-2a's "A's deep target floods into B through the shared grammar" and
  Exp 1's rider 2. So §7's prediction that *narrow+verifier lands between narrow-static and
  broad-static* is untestable here and needs cut-3's partially-heterogeneous DGP.
  **It also suggests, without establishing, that Exp 1's inertness may be partly instrument-limited**
  — the readout it used is blind to root-breadth by construction. That is a claim about Exp 1 which
  this experiment did not set out to test and should not be treated as settled.
- **§4 of the idea doc is untested.** This tests §5 (grader type), as the doc itself says §7 does.
  "Distance ⇒ decorrelated blind spots" remains asserted; the `peer` arm was built to measure it and
  the depth probes could not resolve it.
- **One rule seed** throughout, per repo convention.
- **The verifier is privileged.** It is the exact generative process — the same class of privileged
  access this repo criticises elsewhere. It makes the type contrast clean and the absolute recovery
  numbers optimistic.
- **The depth probes are the only instrument that reports.** Val NTP moves in a *different* order
  from the probes in every run (at m=2 `random` is best and `real_data` worst), which is consistent
  with val NTP rewarding pool diversity while the probes reward structure — but it means the two
  instruments have never agreed, and only one is being read.
- **d1–d3 saturate** at m=3, so the entire result lives at d4–d6, and d4's effects are ~0.004.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic; L4 throughout
# 0. the parser gate (local, no GPU)
python3 -c "from rhm.rhm_data import *; import numpy as np; \
  r=generate_rules_distinct(16,2,6,4,seed=0); s=generate_sequences_batched(r,2000,seed=1); \
  print(possible_set_parse(s,r)['valid'].mean())"        # must be 1.0

# 1. mint-yield diagnostic (pick prefix_frac / temperature before spending on arms)
modal run rhm/minting/mint_loop.py::prep --breadth narrow --sweep --depth 6 --m 3 --seed-pool 65536

# 2. the headline run, 3 seeds (7 arms each; seed 42 also carries verifier@4 / mirror_dedup / broad)
ARMS="narrow:static,narrow:mirror,narrow:peer,narrow:verifier,narrow:random,narrow:peer_verifier,narrow:real_data"
for sd in 42 1 2; do
  modal run --detach rhm/minting/mint_loop.py::minting --tag q8m3s$sd --arms "$ARMS" \
    --depth 6 --m 3 --seed-pool 65536 --n-candidates 73728 --accept-quota 8192 \
    --prefix-frac 0.5 --mint-weight -1 --seed $sd
done

# 3. read it back
PYTHONPATH=. python3 rhm/minting/aggregate.py \
    rhm/minting/figures/q8_m3_s42 rhm/minting/figures/q8_m3_s1 rhm/minting/figures/q8_m3_s2
PYTHONPATH=. python3 rhm/minting/analyze.py rhm/minting/figures/q8_m3_s42   # one run, all tables
```

`--mint-weight -1` selects pooled sampling (minted share = size share); `>= 0` fixes the gradient
share instead and is what §3's middle row shows to be harmful. Results JSON on the `rhm-scaling-data`
volume under `rhm_minting/`, mirrored to [`figures/`](figures/) so the two scripts above run without
volume access.

**Note on monitoring**: `modal run --detach` log streaming died ~1h before the jobs did on two
occasions here. Poll the volume for expected output files rather than trusting a log tail.
