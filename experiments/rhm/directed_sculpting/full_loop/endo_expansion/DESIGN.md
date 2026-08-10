# Pre-registration — can an ENDOGENOUS evaluative grader expand?

**Written**: 2026-08-01, **before any arm was run.** Nothing below is edited after the first
launch; corrections are appended to the README as "what the pre-registration got wrong".

> **Appended 2026-08-01, after a `--quick` smoke and before the real launch.** No prediction
> changed. Three implementation fixes the smoke forced, recorded here because one of them
> materially affects P7's arm and hiding it in a diff would be dishonest:
>
> 1. **Tie-breaking in the metered teacher.** With lowest-index tie-breaking, `alloc_uniform`
>    reported `tree_share` **1.000** while the DP was only allowed to look at blocks {9, 4, 13}
>    — one tree block of three. Every state where no allowed block improved `d*` was silently
>    taught "regenerate block 0", which at L=4 is a tree block: a free on-target teacher
>    manufactured out of an arbitrary tie-break and handed preferentially to the arm that
>    deserves it least. Metered and endogenous teachers now break ties uniformly and mark
>    no-improvement states **uninformative**, the same convention as the rollout teacher. The
>    published `evaluative` arm keeps the lowest-index convention (it is the reproduced anchor)
>    and its own no-improvement rate is now reported, so the size of the artefact on that arm is
>    measured rather than assumed small.
> 2. **Arm independence.** Arms were inheriting the previous arm's global torch RNG state, so a
>    trajectory depended on arm *ordering*. The global seed is now reset identically at each
>    arm's start — matched minibatch order across arms, and a result that depends on `seed`
>    alone, which is what lets the eight arms split across parallel jobs.
> 3. **`teacher_rolls` 2 → 3.** At 2 rolls the rollout teacher had an opinion on only 38% of
>    states (binary terminal success gives `R̂ ∈ {0, ½, 1}`). Raising it to 3 buys resolution at
>    1.5× teacher cost. This deliberately *helps* the endogenous arm: the question is whether an
>    endogenous evaluative grader can expand, not whether a stingy sample budget can. The meter
>    records the price.

**Parent**: [`../README.md`](../README.md) §5 (the expansion 2×2) and §3 (the ladder, where
`visits` lives) — read as **separate** experiments.
**The gap being attacked**: [`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md)
"What this does not establish" bullet 1 — *"No run is both endogenous and expanding."*
**The frame under test**: [`ideas/heterogeneous_graders.md`](../../../../../ideas/heterogeneous_graders.md)
§4 (no signal is both dense and evaluative), §4b (expansion needs a grader that tolerates
deferred, initially-negative payoff), §8 (the mirror-grader criterion), §9 (heterogeneous vs
homogeneous grader disagreement — named there as *the* load-bearing test of the frame, never
built).
**The design rule held to**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md)
§7 — *an endogenous value must be denominated in a currency the agent **pays**, not one it
**reports**.*

---

## 1. Which outer-loop action each arm gets (the §(i) requirement)

There are two distinct outer-loop actions and they are **not** the same experiment. Every arm
below does exactly one of them.

| outer-loop action | what it decides | arms here |
|---|---|---|
| **choose WHAT** | writes back a per-state target the inner loop then fits | `dense`, `evaluative`, `endo_rollout`, `endo_shuffled`, `endo_value` |
| **choose WHERE** | allocates a metered budget over *where the target gets computed* | `alloc_kstar`, `alloc_uniform` |

No arm does both.

## 2. The arms (all `static` world; nested; matched update budget; one warm start)

Drift is dropped on the published finding that it is orthogonal (evaluative − frozen is
+4.63 ± 0.46 PR static vs +4.89 ± 0.73 drifting, statistically identical, §5). That halves the
compute and is the controlled choice, not a shortcut.

| arm | teacher / target | currency | endogenous? |
|---|---|---|---|
| `frozen` | root-CE only | — | (no-loop floor) |
| `dense` | asymmetric FM local loss ("be predictable") | own dense error | yes, **homogeneous** |
| `evaluative` | hard CE against exact-DP `k*` | privileged DP | **no** — external, precomputed |
| **`endo_rollout`** | hard CE against `k̂ = argmax_k R̂(x,k)`, `R̂` = Monte-Carlo terminal task success after executing `k` then behaving | **paid** (materialisations) | **YES ← the missing cell** |
| `endo_shuffled` | the *same* `k̂` targets, permuted across states | paid, content destroyed | control for loss shape |
| `endo_value` | hard CE against `argmax_k V(z_true_next)` | **reported** (the value's own scalar) | yes — the wirehead arm |
| `alloc_kstar` | `k*` computed on only the top-`dp_blocks` blocks by `forecast_visits` | privileged DP, endogenously *allocated* | allocation only |
| `alloc_uniform` | `k*` on a uniformly-drawn block set of the same size | privileged DP, uniformly allocated | control for the above |

**What makes `endo_rollout` endogenous.** It never touches `dp_cost`, the rule tables, channel
labels, or any per-move label. It touches exactly one thing: the environment's own terminal
reward (`possible_set_success` — "did I reach a valid r\* configuration"), sampled by actually
rolling out, which is the same and only signal the `visits` MC value has ever been trained on.
That is what *evaluative* means (§4: "referencing outcomes, and the informative outcomes are
the ones you would rather not sample"). It is also what makes it **expensive**, and the meter
records that.

**What makes `endo_value` the wirehead arm.** Its target is the value's *report* about a state
rather than a sampled outcome. Per §7's design rule this is the configuration that has capped
or been gamed every time this repo has built it, most sharply in
[edit-control](../../../RHM_EDIT_CONTROL_README.md) (`P(r*) → 1` while ~99% of sequences go
off-grammar). It is built deliberately, as the contrast that makes the paid/reported
distinction a measurement rather than an assertion.

## 3. The grader-disagreement instrument (the §(ii) requirement)

Computed **every round for every arm** on one frozen probe of states, so the monitor charge is
identical across arms and only the loss differs (the ladder's own discipline). Five move-scorers
over all 14 candidate blocks at each probe state:

| scorer | what it is | type |
|---|---|---|
| `fm_pred` | −(acted-block squared FM error) — literally what the `dense` arm's loss rewards | **dense / inner** |
| `fm_pred_fresh` | the same, under the independently-trained fresh FM (grading rounds only) | **dense, homogeneous partner** |
| `belief` | Δ log P(root=r\*) under the controller — the edit-control gameable scalar | belief-report |
| `dp` | −Δ`d*` (privileged; **diagnostic only**, never in a loss outside `evaluative`/`alloc_*`) | external evaluative |
| `roll` | `R̂` — MC terminal success | endogenous evaluative |
| `value` | Δ`V` on the *materialised* next state | reported evaluative |

Reported per ordered pair: `top1_disagree` (argmax differs), `rank_corr` (mean per-state
centred correlation of the two score vectors), and `cost` (the normalised amount of the
*second* scorer the first one asks you to give up at its own argmax) — the magnitude half.

### The calibration, stated both ways

- **By construction**: the `dense` arm's outer objective *is* a functional of `fm_pred`, so any
  ranking it induces is a monotone map of the dense ranking and its disagreement is **exactly
  zero**. There is nothing to measure.
- **Empirically**, the honest floor is `top1_disagree(fm_pred, fm_pred_fresh)` — two graders of
  the *same type*, different init and different data. This is
  [heterogeneous_graders](../../../../../ideas/heterogeneous_graders.md) §9's homogeneous
  seed-ensemble baseline, which the S1 negative says is what a heterogeneous grader has to beat.
  It is free here because the grading rounds already train a fresh FM.

## 4. Predictions, written down before the run

**P1 — instrument.** `top1_disagree(dp, fm_pred)` is **clearly non-zero** (predict > 0.5 over
14 candidates). The external teacher has the license to disagree; if it does not visibly use it,
the instrument is broken and nothing downstream is readable.

**P2 — the homogeneous floor is genuinely lower.** `top1_disagree(fm_pred, fm_pred_fresh)`
< `top1_disagree(dp, fm_pred)`, and `rank_corr(fm_pred, fm_pred_fresh)` is clearly positive
while `rank_corr(dp, fm_pred)` is near zero or negative.

**P3 — the collapse test, and it is the diagnosis to reach for first.** If `endo_rollout`'s
`top1_disagree(roll, fm_pred)` and `rank_corr(roll, fm_pred)` sit at the **homogeneous** floor,
the two loops have collapsed into one and **any gain the arm shows is an inner-loop gain wearing
an outer-loop label**. That conclusion takes precedence over any headline number.

**P4 — the headline.** `endo_rollout` lands **between** `frozen` and `evaluative`, recovering
**30–60%** of the DP teacher's lift: predict belief PR at r10 in **8.0–10.0** (anchors: frozen
6.93, evaluative 11.56), ballistic **0.10–0.25** (anchors 0.024 / 0.369), fresh-FM top1
**0.28–0.42** (anchors 0.173 / 0.554). Reasoning: `visits` recovers 84% of the oracle's
*allocation* prize, but a 2-sample MC return over 14 candidates is a far noisier per-move signal
than an exact DP, so `teacher_agree_dp` should be well under 1 and the lift should scale with it.

**P5 — attribution.** `endo_shuffled` lands **at or below `frozen`**. If it expands, PR
expansion is an artefact of the CE-over-moves loss *shape* feeding gradient to idle capacity,
and the published 2×2's reading needs revisiting — which would be the more important result.

**P6 — the paid/reported split.** `endo_value` wireheads: `value_bias` (mean reported success
probability − mean realised rollout success) grows monotonically across rounds, mean belief
log P(r\*) on the ballistic beam's terminal states rises, and true ballistic success is **flat
or below `frozen`**. §7's design rule predicts exactly this and it has never been tested
directly.

**P7 — endogenous allocation of an external target.** `alloc_kstar` recovers **most** of
`evaluative`'s lift at 3/14 of the DP budget; `alloc_uniform` recovers clearly less. **Stated
weakness, in advance**: 8 of the 14 blocks are tree blocks, so a uniform block draw is already
57% on-target — this geometry caps how large the contrast can be. If the two arms tie, the
honest reading is *"this cut is uninformative here"*, not *"endogenous allocation does not
work"*.

**P8 — the meter.** The endogenous teacher costs ~2 orders of magnitude more materialisations
per label than the DP teacher costs privileged calls. This is the concrete price of not having
ground truth and is reported as a first-class number, not a footnote.

## 5. What would falsify the frame

- `endo_rollout` expands **and** its disagreement sits at the homogeneous floor → the gain is
  not coming from grader heterogeneity, and §4b's dense/evaluative story does not explain it.
- `endo_shuffled` matches `endo_rollout` → target *content* is irrelevant and the whole
  "grounding" reading of `evaluative − dense` is loss-shape, not grounding.
- `endo_value` expands *and* stays calibrated → the paid/reported design rule (§7) is wrong, or
  at least not load-bearing on this substrate.

## 6. Known limits, stated in advance

- **`endo_rollout` still consumes the environment's terminal reward.** That is what evaluative
  means; it is not privileged per-move ground truth, and it is exactly the signal the ladder's
  `visits` value already runs on. But it is not "zero external information", and the writeup
  will not claim it is.
- **Single geometry** (L=4, the independent-grammar distractor DGP), where off-tree relevance is
  exactly 0.000 by construction. Cut-3 (`../partial_hetero/`) is where a graded answer exists.
- **Static world only.** Licensed by the published orthogonality of drift, not by a fresh check.
- The rollout teacher's behaviour policy is the arm's own controller, so the teacher is
  non-stationary across rounds (policy iteration). Intended; recorded.
