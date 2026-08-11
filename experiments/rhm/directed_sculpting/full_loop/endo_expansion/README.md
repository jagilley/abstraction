# E4 — can an ENDOGENOUS evaluative grader expand? Yes, and the readout that said so was wrong

**Status**: built and run. 9 arms × 3 seeds × 10 rounds, static world, one warm start per seed.
Predictions written down before the first launch in [DESIGN.md](DESIGN.md).
**Date**: 2026-08-01.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**The gap this closes**: [`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md)
"What this does not establish" bullet 1 — *"No run is both endogenous and expanding."*
**The frame under test**: [`ideas/heterogeneous_graders.md`](../../../../../ideas/heterogeneous_graders.md)
§4 / §4b / §8, and §9's never-built disagreement cut.
**The design rule under test**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md) §7.
**Companion node**: [`../endogenous_expansion/`](../endogenous_expansion/README.md) — see §0.

---

## 0. Read this with its sibling — two independent attacks on the same gap

[`../endogenous_expansion/`](../endogenous_expansion/README.md) was built in parallel, without
contact, on the same sentence of the same idea doc, and merged first. **The two nodes agree on the
answer and cover different halves of the question**, so neither supersedes the other:

| | `endogenous_expansion/` | **this node** |
|---|---|---|
| the endogenous teacher | `critic` — `argmax_k V(next state)`, the **reported** currency, with `V` **frozen while it grades** | `endo_rollout` — `argmax_k` of sampled terminal task success, the **paid** currency (this node also runs the reported one, as `endo_value`) |
| controls on the target | **fidelity**: two external DP teachers degraded to matched fidelity | **content**: the same teacher re-paired across states, and a uniform random block |
| wirehead control | `mirror` — argmax of the very logits the loss trains | `endo_value`, read on the beam belief-vs-truth pair |
| outer-loop action | choose WHAT only (declined WHERE, with an argument) | choose WHAT **and** choose WHERE, as separate arms |
| grader-disagreement instrument | — | five typed move-scorers, all pairwise, against a homogeneous floor |
| headline recovery of the DP lift | 19% ballistic (their `critic`) | 34% ballistic (`endo_rollout`), 33% (`endo_value`) |

**Where they converge, independently, which is the strongest thing here**: both retire belief PR.
They found it as a rank inversion against target fidelity (ρ = −0.80); this node found it causally
with a zero-information target that tops the PR table (§3). Their reconciliation is right and this
node's data sharpens it: **PR detects the presence of the plan-CE *pathway*, not the quality of the
target flowing through it** — which is why `frozen`/`dense` (no plan term at all) separate cleanly
and §5 stands, and why every arm that *has* the term is unrankable by PR.

**Where they differ, and how to read it**: their `critic` recovers 19% of the ballistic lift and
this node's `endo_value` — the same teacher — recovers 33%. The likely cause is that they freeze
`V` while it grades and refresh it from fresh rollouts each round, whereas this node trains `V`
through the plan CE. Their choice is the cleaner anti-wirehead design and is also why this node hit
the gauge freedom in §5. Untested against each other; treat the two numbers as a range, not a
disagreement. Their fidelity controls license a claim this node cannot make (*the residual gap is
fidelity, not endogeneity*); this node's content controls and disagreement instrument license two
they cannot (*PR is causally content-blind*, and *the loops did not collapse*).

---

## One-liner

**An endogenous evaluative grader expands.** A Monte-Carlo teacher that never touches the DP, the
rule tables, or a channel label — only the environment's own terminal reward, sampled by rolling
out — recovers **34% of the privileged DP teacher's ballistic-control lift and 53% of its
transferable plannability**, at 3 seeds, and its disagreement with the dense inner loop
(rank corr **−0.130**) is indistinguishable from the external teacher's (**−0.220**) against a
homogeneous floor of **+0.946**. So the two loops did not collapse into one, and the central gap
is closed on the positive side. **But the readout the published 2×2 leads with does not survive
its own control**: a *uniform random block* target, agreeing with `k*` at chance, produces the
largest belief-PR expansion in the experiment — **145% of the DP teacher's PR lift** — while its
control sits exactly at the no-loop floor. Across nine arms, Spearman(PR, ballistic) = **+0.27**;
fresh-FM plannability gets **+0.87**. PR was never measuring expansion; it was measuring whether
gradient reached the block latents. Two further results: endogenous **allocation** of an external
target works but is beaten by endogenous **target generation** (choosing *what* > choosing
*where*), and §7's paid-vs-reported design rule comes back a **scoped negative**.

---

## 1. What was missing, and what got built

`../expansion.py` has exactly one teacher — `collect_grounded_moves`, the exact-DP best move
`k*` — and it is external, precomputed and privileged. The ladder's `visits` is endogenous but
only ever allocated. The metered-data doc names the consequence as the program's central hole.

Nine arms, nested, forked from one warm start per seed, matched update budget, matched instrument
charge, static world.† Every arm does **exactly one** of the two outer-loop actions.

| arm | outer action | teacher | endogenous? |
|---|---|---|---|
| `frozen` | — | root-CE only | (no-loop floor) |
| `dense` | choose WHAT | asymmetric FM local loss | yes, **homogeneous** |
| `evaluative` | choose WHAT | hard CE vs exact-DP `k*` | **no** — external |
| **`endo_rollout`** | **choose WHAT** | hard CE vs `argmax_k R̂`, `R̂` = MC terminal task success | **YES** |
| `endo_shuffled` | choose WHAT | the *same* targets, re-paired across states | content control |
| `endo_random` | choose WHAT | a uniform random block | zero-information floor |
| `endo_value` | choose WHAT | `argmax_k V(materialised next state)` | yes — **reported** currency |
| `alloc_kstar` | **choose WHERE** | `k*` on the top-3-of-14 blocks by `forecast_visits` | external target, endogenous allocation |
| `alloc_uniform` | choose WHERE | `k*` on a uniform 3-block draw | control for the above |

† Static only, licensed by the published finding that drift is orthogonal (evaluative − frozen is
+4.63 ± 0.46 PR static vs +4.89 ± 0.73 drifting). Not re-checked here.

**What makes `endo_rollout` endogenous, stated precisely.** It never calls `dp_cost`, never reads
a rule table, and is never told which blocks belong to the tree — it evaluates all 14 candidates.
It touches one thing: `possible_set_success`, the environment's own terminal reward, obtained by
*executing* a candidate move and then behaving to the end of the budget. That is what *evaluative*
means ([heterogeneous_graders](../../../../../ideas/heterogeneous_graders.md) §4: "referencing
outcomes, and the informative outcomes are the ones you would rather not sample"), it is the only
signal the ladder's `visits` value has ever been trained on, and it is denominated in a currency
the agent **pays**. It is *not* zero external information, and this writeup does not claim it is.

### The back-compat check: the three reproduced arms land on the published numbers

| arm | PR r1 → r10 | published | d4 | published | ballistic | published |
|---|---|---|---|---|---|---|
| frozen | 6.96 → 6.96 | 6.99 → 6.93 | 0.625 | 0.619 | 0.024 | 0.024 |
| dense | 6.37 → 6.71 | 6.31 → 6.61 | 0.628 | 0.626 | 0.003 | 0.007 |
| evaluative | 9.26 → 11.52 | 9.03 → 11.56 | 0.817 | 0.812 | 0.363 | 0.369 |

Round-0 belief PR 7.04 ± 0.03 against the published 7.04 ± 0.05. The port is clean, so every new
arm below is comparable to the published 2×2 and not only to itself.

## 2. The answer: yes, and it is not an inner-loop gain in disguise

`endo_rollout` at the last round, against the paired per-seed recovery of the DP teacher's lift:

| readout | frozen | **endo_rollout** | evaluative | recovery |
|---|---|---|---|---|
| belief PR | 6.96 | 12.10 | 11.52 | +114.1% ± 6.8% |
| d4 | 0.625 | 0.746 | 0.817 | +63.3% ± 4.4% |
| **ballistic w16** | 0.024 | **0.135** | 0.363 | **+34.4% ± 7.2%** |
| **fresh-FM top1** | 0.134 | **0.337** | 0.514 | **+53.4% ± 7.6%** |

3/3 seeds same sign on every row (vs frozen: ballistic t = +8.6, top1 t = +7.0). Ballistic climbs
monotonically across grading rounds (0.080 → 0.104 → 0.107 → 0.131 → 0.135) and transfers to a
held-out world realisation no arm ever saw (novel 0.125 vs base 0.135).

**The pre-registered collapse test, which decides whether any of that is readable.** The program's
central negative was diagnosed as *"the outer loop is identical to the inner loop"*. Five
move-scorers of different type on one frozen probe, every round, every arm; `cost` is how much of
the second grader, in its own normalised units, the first asks you to give up at its own preferred
move:

| pair | type | rank corr | cost |
|---|---|---|---|
| dense FM vs **independently-trained** dense FM | **homogeneous floor** | **+0.946 ± 0.006** | 0.036 |
| belief scalar `log P(r*)` vs dense FM | report-derived | +0.176 ± 0.006 | 0.643 |
| **external DP vs dense FM** | heterogeneous | **−0.220 ± 0.011** | 0.658 |
| **endogenous rollout vs dense FM** | heterogeneous | **−0.130 ± 0.010** | 0.479 |
| endogenous rollout vs external DP | teacher quality | +0.446 ± 0.015 | 0.408 |

1. **P1 and P2 confirmed.** The external teacher exercises its licence to disagree, and the
   homogeneous floor is genuinely a floor — two dense graders differing only in init and data
   rank moves at +0.95 and cost each other **0.036**, two orders of magnitude less than any
   heterogeneous pair. This is the first quantitative statement of
   [heterogeneous_graders](../../../../../ideas/heterogeneous_graders.md) §9 on this substrate.
2. **P3 passed — the loops did not collapse.** The endogenous grader sits at −0.130, on the
   external teacher's side of the floor, not the dense side. It asks the belief to give up 73% as
   much dense loss as the DP does. Any gain it shows is *not* an inner-loop gain wearing an
   outer-loop label.
3. **A free confirmation of §8's mirror criterion.** The belief's own scalar is the one signal
   that comes back **positively** correlated with the dense grader (+0.176). A grader built from
   the model's report sits nearer the model's own dense error than a grader built from outcomes
   does — which is exactly what "the same grader in a mirror" predicts.
4. The disagreement numbers barely move across arms (rollout-vs-dense is −0.079 in `frozen`,
   −0.130 in `endo_rollout`). **Heterogeneity is a property of the grader types, not of what the
   arm was trained on** — the strongest form of the type-gap claim, and a sign the instrument is
   reading grader geometry rather than arm-specific artefacts. It also means this instrument
   cannot rank arms; its job was the collapse test.

## 3. The result that matters more: belief PR is anti-informative

`endo_shuffled` (the rollout teacher's own targets, randomly re-paired with states) and
`endo_random` (a uniform random block — agreement with `k*` **0.072** against chance 0.071, tree
share **0.574** against block-uniform 0.571):

| arm | PR (recovery) | d4 (recovery) | ballistic (recovery) | fresh-FM top1 (recovery) |
|---|---|---|---|---|
| frozen | 6.96 | 0.625 | 0.024 | 0.134 |
| evaluative | 11.52 (100%) | 0.817 (100%) | 0.363 (100%) | 0.514 (100%) |
| endo_rollout | 12.10 (+114%) | 0.746 (+63%) | **0.135 (+34%)** | **0.337 (+53%)** |
| endo_shuffled | 12.03 (+113%) | 0.722 (+50%) | 0.011 (**−4%**) | 0.204 (+18%) |
| **endo_random** | **13.52 (+145%)** | 0.704 (+41%) | 0.016 (**−2%**) | 0.129 (**−2%**) |

Paired per seed:

| contrast | PR | d4 | ballistic | fresh-FM top1 |
|---|---|---|---|---|
| endo_rollout − endo_shuffled | +0.07 ± 0.42, t=**+0.2**, signs disagree | +0.025, t=+5.4 | **+0.123, t=+11.8** | **+0.133, t=+6.2** |
| endo_random − frozen | **+6.56, t=+14.7** | +0.079, t=+8.7 | −0.009, t=−1.1 | −0.005, t=−0.3 |
| endo_rollout − endo_random | −1.41, t=−5.6 | +0.042, t=+12.1 | **+0.119, t=+8.4** | **+0.207, t=+6.6** |

**A target carrying literally no information produces the largest belief-PR expansion in the
experiment — 1.45× what the privileged DP teacher produces — and a belief that cannot be planned
in at all.** Ranking the nine arms by each readout against their ballistic ordering:

| readout | Spearman with ballistic control |
|---|---|
| belief PR | **+0.27** |
| d4 | +0.75 |
| fresh-FM top1 | **+0.87** |

**What this does and does not overturn.** It does **not** overturn `../README.md` §5's finding:
that result also rested on ballistic (15× the floor) and fresh-FM plannability (3×), and both are
clean under this control. It overturns the **framing** — PR listed first, and the headline phrase
*"expands the belief ~1.7× in effective dimension."* And it shows the parent's existing guard is
insufficient. That guard reads *"no PR rise with flat depth is read as expansion"*; `endo_random`
has a PR rise **and** a depth rise (+41% of the d4 lift) and produces nothing. The sufficient
readout is transferable plannability.

**The mechanism, and why it was foreseeable.** The plan term is a CE over 14 moves whose logits
run `value((z + FM(z, k)).mean)` — i.e. gradient reaches every block latent through the FM and the
value on every step, regardless of which `k` is labelled correct. That is the
[RHM_LATENT_LOOP](../../../RHM_LATENT_LOOP_README.md) *"grounded target feeds gradient to idle
capacity"* signature, and PR reads exactly the idle capacity being filled. `../README.md`'s own
caveat already said PR *"is not certified as the frontier"*. It is worse than uncertified: on this
substrate it is **anti-correlated with content** among the arms that share its loss shape.

**A bonus, on the parent's open item #5** (*"why does the dense grader land below the no-loop
floor?"*). `dense` is at 0.003 against frozen's 0.024 — but so are `endo_shuffled` (0.011) and
`endo_random` (0.016), and neither is a dense grader. So landing below the floor is **not about
density**; it is what reshaping the belief toward any target that does not track the task does.
Suggestive rather than settled — those two deltas are t = −1.9 and −1.1.

## 4. Choosing WHERE: endogenous allocation of an external target

Version (a) — sculpt-continual's own untested next step #3, with `visits` deciding which 3 of 14
blocks the DP is allowed to evaluate, against a uniform draw of the same size.

| arm | blocks the DP saw that were tree | teacher has an opinion | agree w/ full `k*` | PR | d4 | ballistic | top1 |
|---|---|---|---|---|---|---|---|
| `alloc_kstar` | **0.978** | **0.638** | **0.305** | 8.28 | 0.737 | 0.065 | 0.403 |
| `alloc_uniform` | 0.600 | 0.469 | 0.249 | 7.33 | 0.685 | 0.021 | 0.256 |

Paired: +0.96 PR (t=+4.8), +0.052 d4 (t=+2.7), +0.044 ballistic (t=+3.2), +0.147 top1 (t=+2.5),
**3/3 same sign on all four**. **The endogenous relevance tap does steer where an external teacher
gets computed**, and the mechanism is visible rather than inferred: it spends the budget almost
entirely on tree blocks, which is what raises the fraction of states where the metered teacher has
any opinion at all (0.64 vs 0.47).

But metering the DP to 3/14 blocks costs most of the teacher no matter how well it is allocated —
`alloc_kstar` recovers only 12% of the ballistic lift. And the direct comparison between the two
outer-loop actions, which no prior node could make:

> **`endo_rollout` − `alloc_kstar` = +0.069 ballistic (t = +5.8, 3/3 seeds).** At these budgets,
> generating an endogenous target beats endogenously allocating an external one.

The pre-registered weakness — 8 of 14 blocks are tree, so a uniform draw is already 57% on-target —
turned out not to bind: what separates the arms is the *informative* fraction, and there uniform
loses clearly.

## 5. The wirehead arm: a scoped negative, and an instrument that measured a gauge

P6 predicted `endo_value` (target = `argmax_k V(next state)`, the **reported** currency) would
wirehead, per §7's design rule and the [edit-control](../../../RHM_EDIT_CONTROL_README.md)
precedent where `P(r*) → 1` while ~99% of sequences went off-grammar.

**It did not.** `endo_value` is statistically tied with `endo_rollout` on control
(0.132 vs 0.135, t = −0.15, signs disagree) and *better* on d4 (+0.043, t=+8.0) and fresh-FM top1
(+0.108, t=+7.9). The edit-control pair — the belief's own `P(r*)` on the beam's terminal states
against their true possible-set success — shows no collapse anywhere:

| arm | beam `P(r*)` (reported) | beam true success (paid) | ratio |
|---|---|---|---|
| evaluative | 0.542 | 0.363 | 1.5× |
| endo_rollout | 0.349 | 0.135 | 2.6× |
| **endo_value** | 0.403 | 0.132 | **3.1×** |
| endo_random | 0.328 | 0.016 | 21× |
| frozen | 0.384 | 0.024 | 16× |
| dense | 0.231 | 0.003 | 77× |

The ordering external < paid < reported is there and is in §7's predicted direction, but it is a
ranking among arms that all plan *better* than the floor, not a wirehead. The over-report ratio
tracks how badly an arm plans, which is edit-control's *generic* off-manifold overconfidence, not a
loop gaming its own value.

**Read as a scoped negative.** §7's rule is not falsified in general; it was not load-bearing *in
this configuration*, and the reason is structural: here the value is not the grader of record. The
beam is graded by exact possible-set success, and the belief is anchored by root-CE and the FM
term. Edit-control's collapse happened when the belief scalar **was** the planning objective. Ten
rounds also may not be long enough for a slow drift.

### The instrument failure, recorded so it is not rebuilt

`value_calibration` — reported `sigmoid(V)` against realised rollout success — was the *primary*
wirehead readout and it is **invalid for every arm that trains the value through the plan CE**.
The CE logits are `value(...)/tau` over the 14 candidates, so only *differences* across `k` enter
the softmax: **the value's absolute output level is a gauge freedom.** It drifted accordingly —
`sigmoid(V)` reached 0.000 in `endo_rollout`, `endo_shuffled`, `endo_random` and `endo_value`
against realised success of ~0.20. So `value_bias` measures the gauge, not calibration, and the
`corr` column is contaminated by sigmoid saturation on top of that.

The wirehead conclusion above therefore rests **entirely** on the beam belief-vs-truth pair and on
ballistic control, neither of which touches the value head. A future version needs a
gauge-invariant readout — the CE's own temperature-normalised logit spread, or re-anchoring `V`
with a small BCE term on fresh rollout labels each round.

## 6. The meter: what an evaluative grader costs when nobody hands you `d*`

| arm | materialisations | privileged DP calls | materialisations per label |
|---|---|---|---|
| `evaluative` | 2.46e5 | 2.77e5 | **8** |
| `endo_rollout` / `endo_shuffled` | **3.74e7** | 0 | **1218** |
| `endo_value` | 4.30e5 | 0 | 14 |
| `alloc_kstar` / `alloc_uniform` | 9.22e4 | 1.23e5 | 3 |

**152× more materialisations**, P8 confirmed at ~2 orders of magnitude. This is the concrete price
of not having privileged ground truth, and it is
[meta_learning_under_metered_data](../../../../../ideas/meta_learning_under_metered_data.md) §6's
*"the expansion grader was outsourced to the researcher"* with a number on it.

The uncomfortable half: the **reported**-currency teacher costs 14 per label — 87× less than the
paid one — and performs comparably on every paid readout. On this substrate the cheap self-report
is nearly as good as the expensive sampled outcome, and only the (unreliable) calibration readout
distinguished them.

## 7. What this establishes

1. **The central gap is closed on the positive side.** A grader that is simultaneously endogenous
   and evaluative expands: 34% of the DP teacher's ballistic lift, 53% of its transferable
   plannability, 3/3 seeds, against a content control that gets 0% and 18%.
2. **And it is genuinely a second loop.** Rank-correlation with the dense inner grader is −0.130,
   against a homogeneous floor of +0.946 and the external teacher's −0.220. The pre-registered
   collapse diagnosis does **not** apply.
3. **Belief PR is anti-informative about expansion on this substrate.** A zero-information target
   produces the largest PR rise in the experiment (145% of the DP teacher's) with control at the
   no-loop floor. Spearman(PR, ballistic) = +0.27 across nine arms; fresh-FM top1 gets +0.87. PR
   and depth *together* are still not sufficient.
4. **Endogenous allocation of an external target works** (+0.044 ballistic over uniform, t=+3.2,
   3/3) **and is beaten by endogenous target generation** (+0.069, t=+5.8). Choosing *what* > choosing
   *where*, at these budgets.
5. **§7's paid-vs-reported design rule is a scoped negative here**, on a configuration where the
   value is not the grader of record — with the caveat that the instrument built to test it
   measured a gauge freedom and the conclusion rests on the beam belief-vs-truth pair instead.
6. **The first quantitative homogeneous-vs-heterogeneous disagreement measurement in the repo**
   (§9's load-bearing test, first half): same-type graders cost each other 0.036 of their own
   range; different-type graders cost 0.48–0.69. Two orders of magnitude.
7. **A partial answer to the parent's open item #5**: landing below the no-loop floor is not about
   dense pressure, it is what a contentless reshaping does — two non-dense arms land there too.

## 8. Caveats

- **`endo_rollout` still consumes the environment's terminal reward.** No DP, no rule table, no
  channel labels, no per-move label — but not zero external information.
- **The wirehead instrument is broken** (§5) and the negative rests on a secondary readout.
- **Static world only**, licensed by the published orthogonality of drift rather than re-checked.
- **One geometry**, where off-tree relevance is exactly 0.000 by construction.
  [`../partial_hetero/`](../partial_hetero/README.md) is where a graded answer exists.
- **`endo_shuffled`'s teacher row reports the pre-shuffle teacher** (identical to `endo_rollout`'s
  by construction — that identity is the check that the two arms saw the same object).
  `endo_random` is the true zero-information floor.
- **The `evaluative` arm teaches its lowest-index tie-break on the 3.6% of states where no move
  lowers `d*`.** Measured, small, and inherited from the published teacher. New arms break ties
  uniformly and mask those states — see [DESIGN.md](DESIGN.md)'s appendix, which
  records that and two other smoke-driven fixes made before launch.
- **10 rounds.** `endo_rollout`'s PR was still climbing at r10 (+0.140/round) — but so was
  `endo_random`'s, which is the whole point of §3.

## 9. Next steps

1. **A gauge-invariant wirehead readout** (§5), and then re-run `endo_value` long enough for a slow
   drift to show.
2. **Re-read the arc's other PR-based expansion claims against a content control.** Stage 5's
   6.7 → 17.0 and sculpt-continual's 12.8 → 10.1 consolidation were both read off PR with no
   random-target arm. §3 says that control is mandatory and cheap.
3. **Sweep the DP budget in `alloc_kstar`** from 3/14 to 14/14 — where does endogenous allocation
   of an external target cross endogenous generation of an internal one?
4. **The second half of §9's cut**: use grader *disagreement itself* as the allocation signal. The
   instrument now exists and is calibrated; this node only reads it.
5. **`endo_rollout` on cut-3** ([`../partial_hetero/`](../partial_hetero/README.md)), where
   relevance is graded rather than stipulated.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion --quick
for s in 1 2 3; do         # the core question
  modal run --detach rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion \
      --tag e4a_s$s --seed $s --arms frozen,dense,evaluative,endo_rollout
  modal run --detach rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion \
      --tag e4b_s$s --seed $s --arms endo_shuffled,endo_value,alloc_kstar,alloc_uniform
  modal run --detach rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion \
      --tag e4c_s$s --seed $s --arms endo_random
done
python3 rhm/directed_sculpting/full_loop/endo_expansion/aggregate.py
```

Splitting the arms across jobs is exact, not approximate: `_run_arm` re-seeds the global torch RNG
identically at each arm and derives every other stream from `seed` alone, and `aggregate.py`
refuses to merge two arm-groups whose round-0 beliefs differ. Results JSON on the
`rhm-scaling-data` volume under `directed_sculpting/endo_expansion_*`, mirrored to
[`figures/`](figures/).
