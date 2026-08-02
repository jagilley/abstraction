# Can an endogenous evaluative grader expand? Yes — at a fifth of the teacher, and the limit is fidelity, not endogeneity

**Status**: built and run. 7 arms × 10 rounds × 3 seeds, static world, plus an unplanned second
replicate of all three seeds (6 runs total). One calibration gate, one reproducibility failure
found and diagnosed, one readout retired. **Date**: 2026-08-01.
**Up**: [../README.md](../README.md) (full_loop) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md) —
this closes the gap named in its own "What this does not establish".
**Frame under test**: [`ideas/heterogeneous_graders.md`](../../../../../ideas/heterogeneous_graders.md)
§4b (compression vs expansion), §8 (the mirror-grader criterion).
**Design rule held to**: [`adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md)
§7 — *an endogenous value must be denominated in a currency the agent pays, not one it reports.*

---

## One-liner

The central gap closes: a grader that never sees the DP, the rule tables, or a channel label —
only terminal task success on rollouts it pays for — **does expand the belief**, above the no-loop
floor on all four readouts, 3/3 seeds, in two independent replicates. But it recovers only **19%
of the DP teacher's ballistic lift** (66% of depth, 51% of transfer), and two fidelity-matched
*external* controls show the residual gap is **target fidelity, not endogeneity**: at bracketed
fidelity the self-generated target beats an external one on every functional readout. The
mirror-grader hazard did not materialise. What did materialise is an instrument failure that
inverts the node's headline metric — **belief PR anti-correlates with target quality** (ρ = −0.80
against fidelity; the exact DP teacher has the *lowest* ΔPR of the four target arms) — and the
proposed replacement, `R_res_participation`, is ~1.1× `R_act` and inherits the same inversion.
**Every dimension-count fails; only magnitude measures track function, and they do so with a
compression sign.** We have no measurement that separates *the belief got bigger* from *the
belief got better*.

---

## 1. The gap

[`meta_learning_under_metered_data`](../../../../../ideas/meta_learning_under_metered_data.md)
states it in its own limitations section:

> "`visits` is endogenous but was never tested for expansion — `expansion.py` has no allocation
> policy, no oracle, and no `visits` tap at all. Conversely the DP `k*` teacher expands but is
> external and precomputed, and sculpt-continual says that is *why* it works (*'so the loop can't
> wirehead its own value'*). **No run is both endogenous and expanding.**"

Two experiments were on the table. **(a)** keep the DP `k*` teacher and let an endogenous tap
decide *where* `k*` is computed under a metered budget
([sculpt-continual](../../../RHM_SCULPT_CONTINUAL_README.md) next-step #3). **(b)** replace `k*`
with an endogenous target, so nothing external enters the loop.

**Built (b).** (a) introduces no new grader *type* — the expansion signal stays the DP and the
endogenous signal only samples it, so a positive would be a venue-change on
[§3](../README.md#3-e1--the-allocation-ladder-ladderpy)'s already-published allocation result
(`visits_only` recovers 84% of the oracle prize), not a test of the type gap §4b is about. The
idea doc's own prediction — that an endogenous tap *"wired into the expansion setup reproduces
some of the DP teacher's lift"* — is only tested when the tap **generates** the target. And the
decider: the wirehead hazard is the stated reason to prefer (a), but this substrate is the rare
one where that hazard is *diagnosable*, because the exact `d*` and `possible_set_success` can be
held **out** of the loop and used as instruments. Edit-control's belief gaming was caught exactly
that way. A hazard you can measure is a reason to instrument, not a reason to ask a smaller
question.

## 2. The design — one variable: where the target comes from

Every arm is [`../channel_env.py`](../channel_env.py)'s `belief_update` at a matched budget.
`frozen` and `dense` are the published static rows verbatim. The five target arms are **identical
in loss, optimizer, budget, data and seed** — they differ only in the vector `gkstar`.

| arm | target `k̂` | endogenous? | role |
|---|---|---|---|
| `frozen` | — (root CE only) | — | the no-loop floor |
| `dense` | — (+ endogenous FM local loss) | yes | the homogeneous-grader anchor |
| `evaluative` | `argmin_k d*(regen(x,k))` — exact DP | **no** | the published ceiling |
| **`critic`** | `argmax_k V(state(regen(x,k)))` | **yes** | **the question** |
| `mirror` | `argmax_k V(z + FM(z,k))` — the argmax of the very logits the loss trains | yes | the wirehead control |
| `dp_noised_a` | DP `k*`, uniform-corrupted to match `critic`'s top-1 agreement | no | fidelity control (low) |
| `dp_noised_g` | DP `k*`, tree-corrupted to match `critic`'s realised Δ`d*` | no | fidelity control (high) |

**`critic` is sculpt-continual's loop with the teacher deleted and nothing else changed.** `V` is
the MC critic — trained only on terminal `possible_set_success` of paid rollouts, never shown a
channel label or a `d*` — refreshed every round from fresh rollouts under the *current* belief
(policy iteration) and **frozen while it grades**, so the loss cannot move the thing grading it.
Candidate next states are **materialised** through the generator, not imagined through the FM, so
FM hallucination is not a channel into the target.

**Two controls, because the two fidelity currencies do not match together.** Calibration found
that at equal top-1 agreement (0.26) the critic delivers 0.42 of the DP's realised gain while a
uniformly-corrupted DP delivers only 0.27 — the critic's errors are benign (95% still land on the
tree), uniform corruption's are not. So no single control is "matched accuracy," and the honest
move is to **bracket** from both sides.

**Static world only.** [§5](../README.md#5-e3--expansion-expansionpy-grader-type-is-everything-drift-is-orthogonal)
measured drift orthogonal at 3 seeds (ΔPR +4.63 ± 0.46 static vs +4.89 ± 0.73 drift; Δd4 +0.193 vs
+0.187), so the drift column buys nothing here and its compute buys four more target sources.

### The endogeneity line, stated before the run

- **ALLOWED** — terminal binary goal achievement (`possible_set_success`): the environment telling
  the agent whether it hit the goal it was given. This is **reward**, and the agent *pays* for it
  in rollout transitions — §7's "a currency the agent pays, not one it reports."
- **FORBIDDEN** — the DP `d*`: an exact, dense, per-state, per-move optimal-action label computed
  by a solver holding the rule tables. This is an **oracle**.

`critic` and `mirror` touch only the allowed side.

## 3. The gate, run before the ladder

A null is only interpretable if the endogenous target carries real signal. Measured on a frozen
1024-state probe that never enters a loss:

| target source | top-1 agreement with DP | realised Δ`d*` as a fraction of the DP's |
|---|---|---|
| exact DP, re-drawn (**the instrument ceiling**) | **0.915 ± 0.039** | 0.973 |
| `critic` (round 0) | 0.317 ± 0.024 | **0.439 ± 0.057** |
| `mirror` (round 0) | 0.157 ± 0.025 | 0.229 |
| uniform random move | ~0.09 | ~0.10 |

**Gate passes**: ~4× the random floor. Note the ceiling is **0.915, not 1.0** — mixture rendering
is stochastic, so two runs of the same exact solver disagree a few percent of the time on the same
state. Every `kstar_agree` in this node is read against 0.915, which is why the controls are
matched on `gain_frac` as well.

## 4. The anchors reproduce

Nothing else is readable unless the published rows come back. 3 seeds, cell = this run / published:

| arm | PR r1 | PR r10 | d4 | ballistic | fresh-FM top1 |
|---|---|---|---|---|---|
| frozen | 6.971 / 6.990 | 6.864 / 6.930 | 0.620 / 0.619 | 0.022 / 0.024 | 0.163 / 0.173 |
| dense | 6.338 / 6.310 | 6.659 / 6.610 | 0.617 / 0.626 | 0.005 / 0.007 | 0.130 / 0.200 |
| evaluative | 9.050 / 9.030 | 11.422 / 11.560 | 0.806 / 0.812 | 0.396 / 0.369 | 0.553 / 0.554 |

## 5. The result — the endogenous grader expands, partially

Paired against the *same seed's* `frozen` arm at the same round. Pooled over 6 runs (3 seeds × 2
containers, see [§9](#9-a-reproducibility-failure-and-what-it-bought)):

| arm | delivered fidelity | ΔPR | Δd4 | Δballistic | Δfresh-top1 |
|---|---|---|---|---|---|
| `evaluative` | 0.973 | +4.57 ± 0.43 | **+0.190 ± 0.011** | **+0.381 ± 0.053** | **+0.389 ± 0.025** |
| `dp_noised_g` | 0.437 | +6.57 ± 0.78 | +0.089 ± 0.016 | +0.061 ± 0.024 | +0.178 ± 0.049 |
| **`critic`** | **0.353** | **+7.26 ± 0.48** | **+0.126 ± 0.006** | **+0.071 ± 0.015** | **+0.200 ± 0.048** |
| `dp_noised_a` | 0.338 | +7.43 ± 0.92 | +0.084 ± 0.007 | +0.023 ± 0.006 | +0.054 ± 0.045 |
| `mirror` | 0.229 | +3.50 ± 0.31 | +0.144 ± 0.021 | +0.054 ± 0.035 | +0.181 ± 0.051 |

**A run that is both endogenous and expanding now exists.** `critic` is above the floor on all four
readouts in 3/3 seeds, in both replicates. It recovers **66%** of the teacher's depth gain, **51%**
of its transfer, and **19%** of its ballistic lift.

### The bracket: the deficit is fidelity, not endogeneity

`critic`'s delivered fidelity (0.353) sits **between** the two external controls (0.338 / 0.437),
so the match worked. Per-seed contrasts:

| contrast | Δd4 | Δballistic | Δfresh-top1 |
|---|---|---|---|
| `critic` − `dp_noised_a` (lower fidelity) | +0.041 **3/3** | +0.054 **3/3** | +0.176 **3/3** |
| `critic` − `dp_noised_g` (**higher** fidelity) | +0.044 **3/3** | +0.020 (2/3) | +0.030 (2/3) |

Against the agreement-matched control the endogenous target wins on everything, 3/3. Against the
*gain*-matched control — whose delivered fidelity is **higher** than the critic's — it still wins
on depth 3/3 and ties elsewhere. **There is no structural penalty for being self-generated.**
§8's mirror-grader hazard predicted the opposite and did not materialise.

### The mechanism: a grader is a ceiling, approached from whichever side you start

| arm | policy realised Δ`d*` (fraction of DP), r1 → r10 | its target's fidelity |
|---|---|---|
| `evaluative` | 0.674 → **0.748** (rises) | 0.973 — far *above* the policy |
| `critic` | 0.400 → **0.333** (falls) | ~0.35 — *below* the initial policy |

The loop's move quality converges to its teacher's from whichever side it starts. The critic's
target sat **below where the system already stood**, so distillation pulled the policy down. This
is one number explaining the whole ladder: the arms rank on function by target fidelity, and the
endogenous arm's 19% is what a ~0.35-fidelity ceiling buys.

## 6. No arm climbs

Per-level ancestor recovery, paired vs `frozen` (`d1` = root … `d4` = the block's own feature;
`frozen` absolute: 0.791 / 0.665 / 0.569 / 0.620):

| arm | Δd1 | Δd2 | Δd3 | Δd4 |
|---|---|---|---|---|
| evaluative | **−0.011** | +0.067 | +0.121 | +0.186 |
| critic | **−0.021** | +0.034 | +0.067 | +0.127 |
| dp_noised_g | **−0.025** | +0.015 | +0.044 | +0.082 |
| dp_noised_a | **−0.035** | +0.012 | +0.041 | +0.086 |
| mirror | −0.002 | +0.047 | +0.070 | +0.139 |

**Root recovery is ≤ 0 in every arm, including the DP teacher.** Every gain is at d3/d4 and grows
monotonically toward the surface. So "expands" here must not be read as "moves the deep frontier" —
this is surface/mid expansion, consistent with [§4/§6](../README.md#4-e2--the-hierarchy-climb-climbpy-no-climb-where-surface-repair-suffices)'s
finding that climbing appears only under starved per-event budgets.

## 7. The wirehead panel — no gaming, but the pre-registered index failed

Pre-registered: rising SELF columns with flat WORLD columns is edit-control's signature.

| arm | plan_acc | believed | wire idx | ‖ | policy k*agree | policy Δ`d*` | ballistic | root CE |
|---|---|---|---|---|---|---|---|---|
| | *the loop grading itself* | | | ‖ | *held-out instruments* | | | |
| frozen | — | 0.382 | **+0.360** | ‖ | 0.134 | 0.177 | 0.022 | 0.3389 |
| evaluative | 0.999 | 0.448 | +0.053 | ‖ | **0.572** | **0.748** | **0.396** | 0.3496 |
| critic | 0.997 | 0.000 | −0.096 | ‖ | 0.227 | 0.333 | 0.096 | 0.3551 |
| mirror | 0.999 | 0.038 | −0.033 | ‖ | 0.138 | 0.215 | 0.071 | 0.3471 |

**No wireheading.** The critic's policy is well above the floor on an instrument that never enters
a loss (Δ`d*` 0.333 vs 0.177), root CE is within 0.022 of the floor everywhere (no off-manifold
collapse), and cross-world grading (`base` / two held-out `novel` realisations / `own`) agrees to
within ~0.005 for every arm.

**But the index itself failed as an instrument, and that is worth more than the reassurance.** The
in-loop value's sigmoid collapsed to exactly 0.000 in every degraded-target arm, so belief-vs-truth
is *uninformative* there rather than clean; and `frozen` — whose value was never updated — posts
the *highest* wirehead index in the table (+0.360) purely from staleness. The discriminator that
actually worked was realised Δ`d*` on the frozen move probe. The real failure mode is not gaming
but **drift**: the critic's policy quality declines monotonically (0.400 → 0.333) while its PR
inflates.

## 8. PR inverts — and no dimension-count replaces it

This is the most transferable finding here. Among the **four target arms**, where fidelity actually
varies:

| ordering, best-function first | |
|---|---|
| **ballistic (ground truth)** | evaluative > critic > dp_noised_g > dp_noised_a |
| belief PR | critic > dp_noised_a > dp_noised_g > **evaluative** |
| `R_act` | critic > dp_noised_g > dp_noised_a > **evaluative** |
| `R_res_participation` | critic > dp_noised_a > dp_noised_g > **evaluative** |

ρ(target fidelity, PR) = **−0.80**. A worse target scatters gradient into more directions and
recruits more of them for less function — and the degraded arms are also the *noisiest* on PR
(run-noise sd 0.77 for `dp_noised_g` vs 0.08 for `evaluative`), which is the same fact.

**`R_res_participation` is not the replacement.** It sits at **1.06–1.19 × `R_act`** in all seven
arms and reproduces the inversion exactly. Scored against the ladder (ρ vs ballistic / d4, within
the target arms):

| dimension **counts** — all fail | | | **magnitude** measures — all work | | |
|---|---|---|---|---|---|
| `R_act` (= belief PR) | −0.07 | +0.07 | `frontier_mass` | −0.80 | **−1.00** |
| `R_res_participation` | −0.27 | 0.00 | `absorbed_fraction` | +0.80 | **+1.00** |
| `R_comp_participation` | 0.00 | +0.20 | `rel_residual` | −0.87 | −0.93 |
| `naive_R_res` | −0.47 | −0.33 | `mean_residual_norm` | −0.87 | −0.93 |
| `R_act_H`, `R_res_part/R_act` | −0.47, −0.40 | −0.33, −0.13 | `mean_cosine` | +0.87 | +0.93 |

Counts flip sign across seeds; magnitudes are sign-consistent 3/3 at |ρ| up to 1.00. *(Deflation:
those five magnitudes are ~2 independent quantities — `frontier_mass ≡ 1 − absorbed_fraction`, and
`rel_residual ≈ mean_residual_norm ≈ −mean_cosine`.)*

**The dichotomy is counts vs magnitudes, not activation-space vs FM-residual.** `R_act` *is* an
effective-span measure of the belief itself — no FM involved — and it fails; the best-performing
readouts in the table *are* FM-residual metrics. What fails is dimension-counting wherever it is
computed, because a count cannot separate *recruited-and-useful* from *recruited-and-scattered*.
That predicts a weight-space span measure would fail the same way; it is untested.

**And every magnitude measure tracks function with a *compression* sign** — the arms that expanded
most functionally have the most absorbed, least residual, tidiest beliefs. This runs against §4b's
own prediction that opening a dimension makes prediction worse before better, and it answers
[heterogeneous_graders](../../../../../ideas/heterogeneous_graders.md)' standing open question
(*"is there any representational signature of a grader having gone blind, or is blindness only ever
visible as disagreement between two graders' outputs?"*) in the negative for this ladder.

> **We have no measurement that separates "the belief got bigger" from "the belief got better."**

**Where PR is still valid.** It separates target-present from target-absent cleanly and with no
overlap (no-target 6.76 ± 0.22 vs target 13.32 ± 1.37, gap 6.56). The published 2×2 used PR
entirely inside that range, so **none of this retracts [§5](../README.md#5-e3--expansion-expansionpy-grader-type-is-everything-drift-is-orthogonal)** —
it bounds it. PR is a valid *detector* and an invalid *ranker*.

**β is dead at this scale**: R² 0.52–0.57 against the published 0.95–0.99, confirming §4b's
"unusable below ~100 directions" at our 96-dim belief.

## 9. A reproducibility failure, and what it bought

Adding the frontier probe should have been additive (own RNG stream, no change to any loss or
target). Re-running all three seeds: **seed 2 reproduced bit-for-bit; seeds 1 and 3 did not**
(max ΔPR 2.07 and 1.16).

Diagnosis: `frozen` and `dense` are bit-identical in **all three** seeds; only the four
evaluative-mode arms diverge, and they diverge **at round 1 — before the probe first runs at round
2**. So the probe is not the cause. This is ambient non-determinism in the evaluative path (a
14-way stacked value/FM gradient), and the earlier bit-identity of seed 2 was luck, not
certification.

The accidental replicate is worth more than the failure cost, because it measures run-to-run noise
directly — a component invisible to across-seed error bars:

| readout | run-noise sd (single run) | across-seed sd |
|---|---|---|
| **PR** | 0.07 – **0.77** | 0.31 – 1.29 |
| d4 | 0.007 – 0.010 | 0.006 – 0.025 |
| ballistic | 0.008 – 0.021 | 0.008 – 0.056 |
| fresh-top1 | 0.009 – 0.036 | 0.028 – 0.065 |

**PR's run-noise is comparable to its across-seed sd; the functional readouts' is a fraction of
theirs.** An independent third reason not to read PR alone. Every headline claim above was
re-tested in the replicate and survives 3/3 seeds in both.

## 10. What this establishes

1. **The central gap closes, weakly.** An endogenous evaluative grader — no DP, no rule table, no
   channel label, paying 10.08M rollout transitions for terminal-success labels — expands the
   belief above the no-loop floor on all four readouts, 3/3 seeds, two replicates. It recovers
   66% / 51% / **19%** of the teacher's depth / transfer / ballistic lift.
2. **The residual gap is fidelity, not endogeneity.** Bracketed by two external teachers degraded
   to matched fidelity, the self-generated target wins on every functional readout against the
   lower bracket (3/3) and on depth against the higher one (3/3). The mirror-grader hazard did not
   appear.
3. **A grader is a ceiling.** Policy move-quality converges to the target's from whichever side it
   starts — the DP pulls it up (0.674 → 0.748), the critic pulls it down (0.400 → 0.333).
4. **Belief PR is a valid detector and an invalid ranker**, ρ = −0.80 against target fidelity.
5. **No dimension-count replaces it** — `R_res_participation` is 1.1× `R_act` and inherits the
   inversion; every count fails, every magnitude works, and all of them with a compression sign.
6. **No arm climbs**: root recovery ≤ 0 everywhere, all gain at d3/d4.
7. **The price of endogeneity, metered**: 10,080,000 rollout transitions against the oracle route's
   276,480 DP state-move evaluations — ~36× the interaction for a fifth of the lift.

## 11. Caveats

- **19% is small**, and `critic` − `dp_noised_g` is 2/3 seeds on ballistic and transfer. The
  bracket's lower half is solid (3/3 on everything); its upper half rests on depth alone.
- **The fidelity match is a bracket, not an equality** (0.338 / 0.353 / 0.437). No control is
  exactly matched to the critic.
- **The critic pays a currency the other arms do not.** Its 10M rollout transitions train
  `value_g` only, and `value_g`'s sole output is the target, whose fidelity is measured and
  bracketed — so the extra data's influence is fully mediated by the axis being controlled. Fair,
  but named.
- **The pre-registered wirehead index did not work** (§7). The absence of wireheading rests on the
  Δ`d*` probe and root CE, not on the index it was designed around.
- **`own` == `base` by construction** in a static run; the informative cross-world columns are the
  two held-out `novel` realisations.
- **Evaluative-mode arms are not bit-reproducible across containers** (§9).
- **96 dimensions.** β is already dead here; a higher-dimensional belief may behave differently,
  and §4b says as much.
- **PR is not certified as "the frontier"** — this node is the strongest evidence yet that it is
  not, and reports it alongside depth throughout.

## 12. Open items

1. **A readout that separates size from quality.** Everything that tracks function here is a
   magnitude with a compression sign. Weight-space span is untested, but the counting argument in
   §8 predicts it fails too.
2. **Raise the critic's fidelity and re-read the ladder.** §5's mechanism predicts the endogenous
   arm's function is set by its target's fidelity; a better critic (more rollouts, an ensemble, a
   longer-horizon MC target) should move along the same curve. That is the cheapest direct test of
   claim 2.
3. **The pure-rollout target** (`argmax_k` terminal success of actually executing `k`, no critic in
   between) is the fully-paid version of this arm and was dropped for cost. It brackets the critic
   from above on endogeneity.
4. **Version (a) is still unbuilt** — endogenous *allocation* of the DP teacher under a metered
   budget. A sibling session has run a related design; the two should be reconciled.
5. **A repair readout that reports** — inherited from the parent, still open.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"        # the gate
modal run rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::calibrate --quick
modal run rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::endo_expansion --quick

for s in 1 2 3; do         # the ladder; each seed its own client
  modal run --detach rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::endo_expansion \
      --tag endo_s$s --seed $s
done
python3 rhm/directed_sculpting/full_loop/endogenous_expansion/aggregate.py
```

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/`, mirrored to
[`figures/`](figures/). `endo_s*` and `frontier_s*` are the two replicates; only `frontier_s*`
carries the §8 decomposition. Aggregated output is checked in as
[`figures/AGGREGATE.txt`](figures/AGGREGATE.txt) and
[`figures/AGGREGATE_frontier.txt`](figures/AGGREGATE_frontier.txt).
