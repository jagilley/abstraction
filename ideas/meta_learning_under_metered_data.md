# Meta-learning is optimal under *metered* data, not under drift

**Status**: conceptual synthesis from a 2026-07-29 discussion. No new experiment. Every number is
re-read from an existing node; the reframe, the two-geometries table (§5), the curation mapping (§6),
and the cut-3 scoping (§9) are new and unbuilt.
**Date**: 2026-07-29
**Prompt**: *"we previously thought meta-learning was the optimal learning strategy under conditions of
drift. I wonder if the right answer is more like: it's the optimal strategy under conditions of being
at the frontier — whenever your capabilities are at the point where you can no longer learn any more
from free, dense data."*
**Builds on**: [heterogeneous_graders.md](heterogeneous_graders.md) (§4/§4b dense-vs-evaluative),
[adaptive_core_and_hierarchy_climb.md](adaptive_core_and_hierarchy_climb.md) (the climb frame this
tests), [two_timescale_value_loop.md](two_timescale_value_loop.md) (Type-1/Type-2, the claim this doc
amends), [dimensionality_expansion.md](../beliefs/dimensionality_expansion.md)
**Key experiments**: [`rhm/directed_sculpting/full_loop/`](../experiments/rhm/directed_sculpting/full_loop/README.md)
(PRs #15/#16 — the ladder, the expansion 2×2, the necessity sweep) ·
[`rhm/specialization/`](../experiments/rhm/specialization/README.md) (both negatives) ·
[`mjc/expansion/`](../experiments/mjc/expansion/README.md) (the drift-≠-expansion calibration) ·
[`rhm/ratchet/RHM_META_LEARNING_README.md`](../experiments/rhm/ratchet/RHM_META_LEARNING_README.md)
(the null this doc re-diagnoses)
**Attribution**: the frontier reframe (§1), the expert-scientist scenario (§7), the
fantasy-football refinement that separates *superficially* from *deeply* unrelated (§5), the
sample-efficiency reading of the curation question (§6), and the framing of cut-3 as *realism* rather
than a dataset change (§9) are Jasper's. The price-vs-stage correction, the grader-homogeneity
diagnosis, and the two-geometries reconciliation came out of the exchange.

---

## One-liner

**Drift was never the condition. Grader *type* is what makes a second loop pay, and *price per sample*
is what makes it necessary.** The RHM full loop measured both halves: an evaluative grader expands the
belief identically in a static and a drifting world (ΔPR +4.63 ± 0.46 vs +4.89 ± 0.73), so
non-stationarity contributes nothing; and starving samples-per-event at exactly-matched drift magnitude
is what migrates the learner's gain from the surface to the deep levels (t = −5.44, 3/3 seeds). Being
"at the frontier" is one way to make data expensive, but it is not the operative variable — the
operative variable is the meter, and embodiment and expert attention turn it on without any frontier.
This reconciles the Scaling-Era answer ("train on all of it") as **the correct degenerate solution to
an allocation problem whose meter is off**, and it predicts that the cheap kind of data curation is the
weak kind.

---

## 1. The claim being amended

[two_timescale_value_loop.md](two_timescale_value_loop.md) §(c) states it flatly:

> Meta-learning is necessary **iff** the world is non-stationary: on a stationary objective it provably
> manufactures nothing.

The evidence was [RHM_META_LEARNING](../experiments/rhm/ratchet/RHM_META_LEARNING_README.md) — five
meta-learners (FOMAML/Reptile) on a stationary objective, all collapsing to plain multitask — plus
[RHM_SCULPT_CONTINUAL](../experiments/rhm/RHM_SCULPT_CONTINUAL_README.md)'s value saturating in one
pass on fixed data.

**That null was diagnosed correctly at the time and attributed to the wrong variable.** Its own
post-mortem says *"the outer loop is identical to the inner loop."* Stationarity was riding along
confounded with grader homogeneity, and the full loop's expansion 2×2 separates them.

## 2. Grader homogeneity, and the 2×2 that de-confounds it

A **grader** is a source of learning signal that scores the model. Two graders are *homogeneous* when
they are the same **kind** of signal, however different their data, timescale, or seed.

FOMAML's outer objective *is* the inner loss evaluated a few steps later — same currency (token CE),
different point on the trajectory. It can only ever prefer what the inner loop already prefers, sooner.
It has no license to disagree. The same is true of `fm_cotrain` ("make your own latents predictable to
your own FM"), which is a functional of the model's own dense prediction error.

The evaluative arm has that license: CE against the DP best move `k*` references **outcomes** — does
this edit actually reduce the exact ancestor-recovery cost `d*`? It can prefer a move that makes token
likelihood worse.

[`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md) §5, 3 seeds, entropy-matched
base, graded in three worlds, evaluative − frozen **in the same world**:

|  | static world | drifting world |
|---|---|---|
| **homogeneous** (dense) | caps *below* the no-loop floor (PR 6.61 vs frozen 6.93) | same (PR 6.62 vs 6.84) |
| **heterogeneous** (evaluative) | **ΔPR +4.63 ± 0.46, Δd4 +0.193 ± 0.016** | **ΔPR +4.89 ± 0.73, Δd4 +0.187 ± 0.015** |

**Rows differ enormously; columns do not differ at all.** The old experiment moved the world, held the
grader type fixed, and read the null as being about the world. It was about the row.

Two riders that matter:

- **The evaluative payoff did not wait for the dense channel to exhaust.** Round-0 belief is
  PR 7.04 ± 0.05; `evaluative_static` is already at 9.03 by round 1 and ends at 11.56 with slope still
  +0.080. It was available immediately, from a learner a third of the way up.
- **Dense pressure at that same point actively degrades** — PR 6.61 against frozen's 6.93, ballistic
  0.007 against 0.024. Under a saturation story dense would asymptote to frozen *from below*. It goes
  past it downward. **The gap is a type gap, not a quantity gap**, which is what rules out "an LLM
  simply hasn't saturated yet, so it never reaches the regime."

## 3. Price, not stage: what the necessity sweep actually moved

The prompt's phrase has two words doing work — *"can no longer learn any more from **free**, dense
data."* **`free` is load-bearing; `no longer` is not.**

[`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md) §6 holds drift magnitude
exactly fixed (0.600 nats per event, via `calibrate_sigma_event`) and holds competence roughly fixed,
and sweeps only samples-per-event:

| samples/event | deep gain (d1,d2) | surface gain (d4) | deep − surface |
|---|---|---|---|
| 4096 | −0.0049 ± 0.0060 | **+0.0252 ± 0.0076** | −0.0301 ± 0.0115 |
| 64 | **+0.0142 ± 0.0108** | +0.0077 ± 0.0243 | **+0.0064 ± 0.0196** |

Monotone across four budgets; slope −0.00585 ± 0.00186 per octave, t = −5.44, 3/3 seeds.

The §6 diagnosis is the right one and it is not about being at a frontier: the surface arm **repaired
91% of its damage inside the round**, so invariance was free and nothing priced it. ***Invariance ≠
necessity.*** What produced the climb was how expensive repair was, not how much the learner knew.

> **The reframe this licenses: meta-learning is the optimal move under *metered* data.** Being at the
> frontier is one way to make data expensive (nothing cheap left to learn). Embodiment is another (each
> sample costs a rollout, and some rollouts kill you). Expert attention is a third (each hour costs an
> hour). None requires the others.

The prompt's own motor example is already stated economically — *"you're on your own to bootstrap your
success from very few subsequent samples."* That is small `S`, not high competence.

## 4. Drift is the clock, not the growth

Drift is not demoted to nothing. `climb.py`'s round opens with `advance_drift(...)`, and the
experimental unit is **damage → budget → repair**. Without events there is no "budget *per event*" to
starve — **drift supplies the denominator.**

The wrong story was *"drift → must adapt → adaptation is meta-learning"*, which is what came back null
in [`mjc/expansion/`](../experiments/mjc/expansion/README.md) (±0.08 directions against a +0.72 ± 0.42
calibration) and again in the 2×2 above. The right one is *"drift → a sequence of repair episodes →
the price per episode selects surface vs deep."*

Drift is also not the only event generator. `evaluative_static` gets its episodes from a moving
*epistemic* frontier (Type-1) instead. Drift's distinction is that it **does not self-terminate** —
which is why it stays the right realism condition for the physical substrates, exactly as
[`dimensionality_expansion`](../beliefs/dimensionality_expansion.md) already concluded (*"drift remains
essential for realism, just not for representational growth"*).

## 5. Two geometries, opposite verdicts — and the one in between is unbuilt

The repo contains two RHM setups that differ in **whether the off-task data shares deep structure with
the task**, and they answer "should I restrict to what's relevant?" with opposite signs:

| geometry | relationship | verdict |
|---|---|---|
| [**specialization**](../experiments/rhm/specialization/README.md) — one shared ruleset, A and B are subtrees | *A's depth **is** B's depth* | **inert.** Breadth restriction Δ ≤ 0.013; broad beats narrow on A's own domain (0.581 vs 0.440); supervising A alone lifts B's d4 **0.108 → 0.408** |
| [**full_loop**](../experiments/rhm/directed_sculpting/full_loop/README.md) — independent rule tables per channel (`rhm_channels.py:83`, `generate_rules_distinct`, `rule_seed=101`) | nothing shared | **pays 0.063** in tree FM error (uniform 0.7371 → oracle 0.6742) |

This is the sharpest available statement of what curation is worth, and it says the answer is **set by
the geometry, not by the learner**.

It also re-scopes §7's satiety result below: `structA` is not a subtree of the tree, it is an
independent grammar with its own seed. So the harm measured there is an upper bound taken in a world
with **literally zero transfer available** — lateral recruitment cannot pay under any future task
drift, because the deep structure is unreachable by construction.

**The interesting case is neither.** *Superficially* unrelated, *deeply* shared — different surface
grammar over a partly-common substrate. That is
[specialization](../experiments/rhm/specialization/README.md)'s **cut-3** (§9 below), specced and never
built, and it is the only geometry in which an intrinsic relevance selector faces a real decision
rather than a stipulated one.

## 6. Why "all of it" was right — and why the cheap curation is the weak curation

The ladder's two ends *are* the Scaling-Era dichotomy: `uniform` allocates evenly over all five
channels (20% tree / 40% struct / 40% noise) — **"train on all of it"**. `oracle` is told the
ground-truth labels and puts 96% on the tree — **"a perfectly curated dataset."** The gap is 0.063 in
tree FM error at a metering ratio of 1.78×.

**First, decompose curation**, because two operations share the word and this repo measures opposite
signs on them:

- **Noise filtering** (dedup, quality classifiers, dropping junk) — uncontested. `error_only` is the
  ladder's *worst* arm (0.736, below uniform's 0.737) precisely because it pours ~40% of budget into
  the irreducible channels. The noisy-TV trap, reproduced.
- **Relevance filtering** (drop the structured-but-off-domain) — verdict set entirely by §5's geometry.

"All of it" is a claim about the second under the **shared-substrate** geometry, and language is far
closer to specialization's single-ruleset world than to full_loop's independent-grammar world. Physics
and fantasy football are written in one grammar, by one species, about one world. **That is why it is
right.**

**Second, the sample-efficiency reading is right, with the mechanism inverted.** KFW says a token
target delivers gradient to level ℓ at `vm^(ℓ+2)` — exponentially diluted — so recovering deep
structure from surface targets takes an exponentially large sample, and a metered learner must have the
deep target *delivered* instead (which is what teaching is). But §3's sweep says the abundant learner
does not pay that tax, it **avoids the bill**: at S=4096 the entire gain is surface, at S=64 it is
deep. Abundance lets surface re-fit suffice forever. *(Cross-substrate analogy, not a measurement.)*

**Third, and the sharpest: current LLM data curation is the epistemic tap.** Perplexity filtering,
educational-value classifiers, dedup, quality scoring — every one asks *"is this data learnable /
high-signal?"* That is reducibility. That is the `e` tap.

> `reducible_only` recovers **18%** of the uniform→oracle prize. `visits_only` recovers **84%**.

The relevance tap is not absent from LLM training — it is **data-mixture ablation**: train N models on
N mixtures, read downstream evals, pick. That is literally sampling outcomes to estimate relevance,
which is what makes it evaluative, which is why it costs weeks of wall-clock and runs on a human's
calendar. It is [heterogeneous_graders](heterogeneous_graders.md) §4b's *"the expansion grader was
outsourced to the researcher"* in concrete form. `forecast_visits` computes the same quantity
endogenously, in one round, with **zero environment interaction and zero labels**.

**This closes back onto metering rather than adding a mechanism.** Curation *is* allocation, and
allocation only pays when looking costs something. A human's samples are metered; a pretraining
corpus's are not. Every place LLM training *does* meter — post-training rollouts, eval budget, ablation
compute — is exactly where relevance-shaped selection has reappeared.

## 7. Satiety says *when*; only an outcome-referencing signal says *where*

[`adaptive_core`](adaptive_core_and_hierarchy_climb.md) §12 nominates satiety as the climbing
mechanism. The full loop implemented it and got both halves of the answer:

- **§10 confirmed.** The leak into irreducible channels falls 7.3% → 2.8% against a hard 2.0% floor —
  the *learned* leak drops 5.3% → **0.8%**.
- **§12 exposed as missing an axis.** The satiating arm is **worse overall, +0.0691 ± 0.0107** — larger
  than the *entire* uniform→oracle prize of 0.063.

The mechanism is the part worth carrying:

| channel | reducible fraction | relevance (forecast visits) |
|---|---|---|
| tree | 0.856 | 0.673 |
| **structA** | **0.845** | **0.164** |
| noiseA/B | 0.421 / 0.432 | 0.064 / 0.034 |

Satiated on the tree, the drive walks into structA — **genuine, learnable, richly-structured, useless**.
It correctly avoids the *noise*; what it cannot see is irrelevant *structure*, because reducibility is
purely epistemic (a functional of the model's own error) while relevance is not. This is a far more
insidious failure than chasing noise, and it is the exact shape of "expert stops learning physics,
becomes deeply invested in fantasy football."

> **A stop-signal without a direction produces sideways motion.** The frontier framing supplies the
> *when* cleanly and leaves the *where* open, and the repo has now measured twice that the *where* is
> the hard part and cannot come from the dense side (here; and specialization's reweighting negatives).

## 8. LP is not the expansion grader

[`dimensionality_expansion`](../beliefs/dimensionality_expansion.md) states it directly — *"learning
progress is the expansion grader"* — and [heterogeneous_graders](heterogeneous_graders.md) §4b says the
curiosity line had been building it without calling it that. The repaired ladder disagrees, and in the
direction the frame itself predicted:

| tap | kind | recovery of the oracle prize |
|---|---|---|
| `reducible_only` (LP, floor-corrected) | epistemic, dense-derived | **18%** |
| `visits_only` (relevance) | outcome-trained value | **84%** |

The structural caveat kills one claim and not the other. It **does** explain why the product
(`value_red`) only ties `visits_only` (+0.0019 ± 0.0030) — this geometry has no visited-but-irreducible
cell. It **does not** explain why the epistemic tap is weak *in isolation*, which is a 4.7× asymmetry
in how much of the available signal each carries alone.

[heterogeneous_graders](heterogeneous_graders.md) §10 already flagged the reason: LP *"is a functional
of the dense signal and is evaluative only about epistemics, not about the world's rewards."* By the
frame's own logic LP should not have qualified. **The frame is vindicated; its nominated instantiation
is not.**

## 9. The named next step: cut-3, the partially-heterogeneous DGP

**Not a dataset change for its own sake — the missing realism condition.** §5 shows the repo has only
measured the two degenerate ends of the shared-structure axis: total sharing (one ruleset) and zero
sharing (independent rule tables). Real domains are neither, and *every* question in this doc that
matters becomes trivial at the ends:

- **Relevance selection is stipulated, not decided.** In full_loop, ground-truth best Δ`d*` is *exactly*
  0.000 for every non-tree channel. There is no judgment call for a selector to get right.
- **"All of it" has no interesting answer.** Shared → inert; independent → curate. Only in between is
  the mixture question real.
- **Satiety's lateral recruitment cannot be scored.** With zero transfer, sideways is always wrong; with
  total transfer, sideways is always fine.
- **§7's dilettantism hazard has no test bed.** The competence constraint that stops satiety collapsing
  into novelty-seeking needs a world where some lateral moves pay and others don't.

The construction (from [specialization](../experiments/rhm/specialization/README.md) cut-3 and
[`adaptive_core`](adaptive_core_and_hierarchy_climb.md)'s closing open question): **share the deep
levels, independently seed the shallow ones.** Two channels drawing from a common level-≥ℓ rule table
with distinct level-<ℓ tables gives a tunable sharing depth ℓ — superficially unrelated, deeply shared,
with the depth of the sharing as the swept knob. `generate_rules_distinct` already takes a per-level
seed structure, so this is a rule-construction change rather than new machinery.

**The prediction it is built to test**: relevance-weighted allocation should beat uniform by an amount
that *falls monotonically with sharing depth*, hitting specialization's inert result at full sharing
and full_loop's 0.063 at zero. That curve is the quantitative form of "when is curation worth it," and
it is the same shape as the metering sweep in §Predictions — two independent routes to the same number.

## Predictions / falsification

- **Falsified if** the uniform→oracle gap does *not* shrink toward zero as the monitor:collect ratio
  goes to zero. That is the whole content of the metering claim, it is cheap (the meter is a config
  change on an existing ladder, currently 1.78–1.89×), and it is the load-bearing test.
- **Falsified if** the necessity sweep's migration depends on the learner's *starting competence* at
  fixed `S`. Price-not-stage predicts a cold and a warm belief climb alike; the frontier-as-stage
  reading predicts only the warm one does.
- **Predicts** the cut-3 curve: allocation payoff falls monotonically with sharing depth (§9).
- **Predicts** that an endogenous relevance tap wired into the *expansion* setup reproduces some of the
  DP teacher's lift. Nothing has tested this — see "What this does not establish."
- **Predicts**, from §2's type-gap reading, that scaling a dense objective does not close the gap an
  evaluative grader opens, at any budget.
- **Discriminates barrier from dilution** (inherited from [`adaptive_core`](adaptive_core_and_hierarchy_climb.md)
  §7b, still unrun and still cheap): interpolate NTP → oracle-aux and watch token loss. Rises-then-falls
  = barrier, so the evaluative grader does something no dense signal can do at any scale. Flat =
  dilution, and "you need an evaluative grader" softens to "you need an undiluted target." **This
  decides which version of §2's claim we hold.**

## What this does not establish

- **The two positives do not overlap, and this is the central gap.** `visits` is endogenous but was
  never tested for expansion — `expansion.py` has no allocation policy, no oracle, and no `visits` tap
  at all; its arms differ only by `collect_grounded_moves`. Conversely the DP `k*` teacher expands but
  is external and precomputed, and [sculpt-continual](../experiments/rhm/RHM_SCULPT_CONTINUAL_README.md)
  says that is *why* it works (*"so the loop can't wirehead its own value"*). **No run is both
  endogenous and expanding.** Reading the ladder's relevance result as an expansion result is a
  misreading of the node.
- **Every climbing claim is half-instrumented.** [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md)
  §6's two-instrument requirement has never been met — the repair readout has failed four times, most
  recently with *negative* damage. The migration rests on the depth probe alone, at magnitudes of
  0.006–0.030.
- **The LLM claims in §6 are structural arguments about training setups**, not measurements on any
  model we have run, and the S-sweep→pretraining step is an analogy across substrates.
- **PR is not certified as "the frontier"**, per the belief file's own warning. It is reported
  alongside depth throughout.
- **A numeric correction to the source node.** `full_loop`'s headline quotes *76%* of the oracle's
  advantage for `visits_only`; that traces to the single-seed `l4_v1` table (0.727 / 0.688 / 0.676).
  The repaired 3-seed re-run gives **84%** on the means (81% using the paired per-seed oracle gap). The
  headline is the conservative figure.

## Open questions

- Is the metering ratio the *only* axis, or does the **shape** of the meter matter — a hard budget vs a
  per-sample cost vs a risk of catastrophic loss? Motor learning's meter is the third kind (some
  rollouts kill you) and nothing here prices irreversibility, which
  [heterogeneous_graders](heterogeneous_graders.md) §11 already flags as an unranked sixth condition.
- **What is "up"?** §7 says satiety needs a level-ordered allocation space. Cut-3 supplies an ordering
  by *sharing depth*, which may or may not be the same ordering as hierarchy level. If they differ, which
  one does an intrinsic selector need?
- Does the relevance tap survive when relevance is **not stipulated**? Its 13–18× separation was
  measured where irrelevance is exactly zero by construction. Cut-3 is where it gets tested against a
  graded answer.
- If curation's value is set by geometry rather than by the learner, is there a *measurable* proxy for
  sharing depth in a real corpus — i.e. can you estimate, before training, whether your domain filter
  will be inert or worth 0.063?
- Does "price per sample" subsume the frontier reading entirely, or is there a residual sense in which
  an expert's situation differs from a beginner's at matched sample cost? The cold-vs-warm necessity
  sweep (§Predictions) is the cheapest probe.

## Context pointers for a future agent

1. [`rhm/directed_sculpting/full_loop/README.md`](../experiments/rhm/directed_sculpting/full_loop/README.md)
   end-to-end — §3 (ladder + satiety), §5 (the 2×2), §6 (necessity). **Read §3 and §5 as separate
   experiments**; conflating them is the easiest available error (see "What this does not establish").
2. [heterogeneous_graders.md](heterogeneous_graders.md) §4/§4b/§10 — the dense/evaluative frame this
   doc confirms, and the LP parenthetical in §10 that §8 here cashes out.
3. [adaptive_core_and_hierarchy_climb.md](adaptive_core_and_hierarchy_climb.md) §7b, §10–§12 — the
   barrier/dilution discriminator (still the cheapest open experiment) and the satiety mechanism §7
   here scopes.
4. [`rhm/specialization/README.md`](../experiments/rhm/specialization/README.md) — both negatives and
   the 07-18 reframe. §5 and §9 here depend on reading Exp 1's *mechanism* (shared grammar), not just
   its verdict.
5. [two_timescale_value_loop.md](two_timescale_value_loop.md) §(b)/§(c) — the Type-1/Type-2 split and
   the "iff non-stationary" claim §1–§2 here amend.
