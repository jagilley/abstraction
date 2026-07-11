# Active RHM: turning an autonomous problem into a controlled one, and when that's even possible

**Status**: Done. A clean negative (mean-Δ forward model can't plan epistemic queries), a clean positive (a value-of-information head can, at m=2), and a principled null (m=4) that sharpens the whole question into a measurable boundary.
**Date**: 2026-07-10
**Scripts**: [`rhm_active_query.py`](rhm_active_query.py) (arity test + first planner), [`rhm_active_planning.py`](rhm_active_planning.py) (three-way diagnosis of the null), [`rhm_active_voi.py`](rhm_active_voi.py) (the Bayesian fix)
**Sibling / origin**: the reaching-control arity line in a2a_forward — [ACTIVE_VISION_README.md](../a2a_forward/ACTIVE_VISION_README.md), [REACHING_INTERNAL_README.md](../a2a_forward/REACHING_INTERNAL_README.md) — and the idea doc [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md).

## The question

The standard RHM is an **autonomous** problem: a model passively receives a sequence and predicts. The reaching experiments in a2a_forward are a **controlled** problem: an agent chooses actions, and a query-conditioned forward model `F(b, u)` lets an external planner choose *informative* actions in a way no belief-only model `F(b)` can (the "arity" result). This line asks: **does that controlled/active-inference machinery port back to RHM, and if not, why not?**

### The game (how the code frames it)

A root is expanded by the RHM grammar into a leaf sequence of length `s^L = 16`, partitioned into `n_actions = 8` query blocks of size `s = 2`. An agent sees a **masked** sequence, gets a **budget of 4 reveals**, chooses which blocks to uncover, then predicts the hidden root. A **frozen belief controller** (a small transformer, trained once, then frozen) maps a partial observation to a belief state `b` and a root posterior. Forward models predict the belief update from `b` alone (arity-1) or from `(b, u)` (arity-2). The whole question is whether a planner reading a forward model can choose reveals better than random, approaching an **oracle** that is allowed to peek at the true root to pick reveals.

All comparisons are **within a script** (same frozen controller). The `random`/`oracle` numbers differ slightly across the three scripts because each trains its own controller — do not compare planner accuracies across scripts, only within.

## Phase 1 — the arity test and the first planner (`rhm_active_query.py`)

Built the laboratory: frozen controller, an arity-2 FM `F(b,u)` and arity-1 FMs `F(b)` at 1×/2×/4× capacity, a transition-metric battery, and an external entropy planner (score each candidate reveal by the entropy of `root_logits(b + F(b,u))`, pick the lowest).

**Result (m=2): the arity *structure* ports, the arity *usability* does not.**

| transition metric | arity-2 | arity-1 (all 3 sizes) |
|---|---|---|
| command-conditional cos (query-dependent part of the update) | **0.357** | **0.000** |

Only the arity-2 FM captures any query-conditional structure, and no arity-1 capacity buys it — the arity impossibility ports cleanly. But in **planning** it barely mattered:

| planner | root acc |
|---|---|
| random | 0.570 |
| arity-1 (largest) | 0.565 |
| arity-2 | 0.591 |
| arity-2, **shuffled action labels** (control) | 0.583 |
| oracle | **0.917** |

The arity-2 planner beat random by ~2 points — and the shuffled-action control (feed the FM the wrong query embedding) was nearly as good, so the tiny edge was **not** coming from correctly using query identity. Effectively a null, against a 35-point oracle prize. The task richly rewards good reveals; the FM planner collects almost none of it.

## Phase 2 — three-way diagnosis of the null (`rhm_active_planning.py`)

Ran three hypotheses on a shared controller, at m=2 and m=4.

1. **Fidelity sweep** — arity-2 belief-Δ FMs across steps ∈ {8k,24k,48k} × hidden ∈ {96,384}, recording transition cos *and* planner accuracy.
2. **Posterior target** — an arity-2 FM predicting the change in *root logits* per query (route the signal straight to the decision variable), planned by predicted posterior entropy.
3. **m sweep** — confirm the null and the planning prize aren't m=2 artifacts.

**Findings:**

- **Fidelity is not the bottleneck (falsified cleanly).** Planner accuracy is **flat and uncorrelated (even slightly anti-correlated) with FM transition cos.** At m=2 the *best* planner (0.617) had the *worst* Δcos (0.354); the best Δcos (0.586) gave near-worst planning (0.579). "Train the FM harder/bigger" does nothing. The query-conditional signal `cmd_cos` saturates at ~0.35 (m=2) / ~0.29 (m=4) regardless of steps or capacity.
- **Posterior target doesn't rescue it** — comparable at m=2 (~0.61), *worse* at m=4 (below random).
- **m=4 makes the null total** — planner sits exactly at random (0.20 vs 0.198 random) while the oracle reaches 0.656.

**The diagnosis these three converge on.** A deterministic one-step FM predicting `E[Δbelief | b, u]` **structurally cannot rank queries by informativeness**, because value-of-information lives in the *variance* of the belief update across the query's unknown contents, and a point estimate averages that variance away. This is exactly the autonomous-vs-controlled distinction made concrete:

- In **reaching** (controlled), an action's consequence is *content-independent* ("move fovea to block a" moves it there regardless of contents), so a mean-Δ FM forecasts it near-perfectly and planning works.
- In **RHM** (autonomous/epistemic), the "action" is a *query* whose informativeness depends on *what the grammar hid there* — the very unknown. The MSE-trained FM learns the conditional *mean* update, which pulls the belief toward the average posterior by roughly the same amount for every query, so it barely discriminates. `cmd_spread` is high (realized updates *do* vary by query) precisely because the signal is in the spread the FM discards.

## Phase 3 — the Bayesian fix (`rhm_active_voi.py`)

Don't predict the mean update; predict the **decision variable**. An arity-2 head `g(b,u)` regresses the **expected posterior entropy** after revealing query `u` (target = realized `H(root | b, o_u)` on the actual leaves; MSE regression → `E_{o_u}[H(root | b, o_u)]`, the Bayesian-experimental-design quantity). Since `H(root|b)` is constant across candidates, `argmin g(b,u) = argmax` expected information gain — greedy EIG. Entropy is a **nonlinear** function of the belief, so its conditional mean stays query-discriminative where the mean Δ did not.

Head-to-head on one shared controller per m:

| planner | m=2 | m=4 |
|---|---|---|
| random | 0.560 | 0.210 |
| belief-Δ greedy-entropy (the Phase-2 null) | 0.567 | 0.202 |
| **VoI greedy-EIG (new)** | **0.695** | 0.192 |
| oracle | 0.917 | 0.656 |

**m=2: the fix works.** VoI closes **38%** of the random→oracle gap where belief-Δ closed ~2%. Same controller, same budget — the *only* change is the prediction target. The entropy trace confirms genuinely informative reveals (VoI drives root entropy 1.69→0.67 vs random's →0.95).

**The diagnostic that explains why — and why m=4 fails:**

| diagnostic | m=2 | m=4 |
|---|---|---|
| Pearson(pred, realized entropy), all pairs | 0.737 | 0.727 |
| **per-instance query-rank corr** | **0.303** | **0.066** |
| top-1 argmin agreement (random = 0.143) | 0.270 | 0.176 |

The **aggregate** Pearson is high at both m — but it is dominated by the coarse "how much have I revealed / overall entropy level" signal, useless for choosing *between* candidates in a fixed state. The metric that governs planning is the **per-instance query ranking**, which splits cleanly: **0.30 at m=2, 0.07 at m=4.** Planner gain tracks it exactly.

**m=4 is a principled null, not a fixable failure.** At m=4 the expected-posterior-entropy targets have almost no variance across queries (head MSE collapses to ~0.001; everything sits near max entropy). The queries are *nearly equally informative given the belief state*, so no belief-conditioned planner can rank them. The oracle still wins only because it **peeks at true content** to see which reveal nails the root *for this instance* — a signal that is not a function of the belief and thus unpredictable by any forward model reading the belief.

## The insight (what this arc establishes)

1. **The arity *impossibility* ports; the arity *usability* does not port naively.** Belief-only forward models can't represent query-conditional structure at any capacity (as in reaching). But having that structure in a mean-Δ FM does *not* yield a planner on an epistemic task.

2. **For epistemic actions you must predict the decision variable, not the mean transition.** Value-of-information lives in the spread of possible outcomes; a point-estimate forward model discards it. A value-of-information / expected-posterior-entropy head (greedy Bayesian experimental design) is the right instrument, and it converts a total null into a decisive planner. The fidelity sweep is the proof the mean was the wrong target — more accuracy at predicting the mean bought zero planning.

3. **The controllability boundary is measurable.** An autonomous problem is actively **plannable exactly when the value-of-information of a query is carried by the agent's belief state** rather than only by the hidden content. The instrument is the **per-instance query-rank correlation** (not the aggregate correlation, which is misleadingly high). m=2 is belief-carried (corr 0.30 → planner works); m=4 is content-carried (corr 0.07 → provable null, only the content-peeking oracle wins). This is the autonomous-vs-controlled distinction turned into a number you can compute before planning.

4. **Reaching vs RHM, precisely.** Reaching worked with a plain mean-Δ FM because control actions have content-independent consequences; RHM queries do not. The same apparatus succeeds or fails depending on whether the action's payoff is deterministic-given-belief (control) or variance-in-hidden-content (epistemic).

## What this does and does not show

- **Does show**: (a) the mean-Δ FM planner is a structural null on epistemic RHM queries, robust to FM fidelity, capacity, and prediction-space (belief vs logits); (b) a VoI/EIG head closes a large chunk of the oracle gap at m=2, the first genuine active-planning positive on RHM; (c) a computable diagnostic (per-instance query-rank corr) that predicts *in advance* whether a setting is plannable, validated by the m=2/m=4 split.
- **Does not show**: multi-step / non-myopic planning (all planners here are greedy one-step); that m=4 is *fundamentally* content-bound rather than limited by this particular frozen controller's sufficiency (see next steps); more than a single controller seed per setting.

## Next steps

1. **Sweep m ∈ {2,3,4} (and budget)** to trace the per-instance-corr → planner-gain curve: does controllability degrade smoothly as VoI migrates from belief-carried to content-carried?
2. **Does a stronger controller rescue m=4?** The m=4 belief state may be a weak sufficient statistic; a larger / longer-trained controller might make more of the VoI belief-decodable — testing whether the m=4 null is fundamental (information-theoretic) or controller-limited.
3. **Non-myopic VoI.** Greedy EIG is one-step; a multi-step planner (or a head predicting entropy after several reveals) tests whether look-ahead recovers anything m=4 loses to greedy myopia.

## Reproduction

```bash
cd experiments/
# Phase 1: arity test + first planner
modal run --detach rhm/rhm_active_query.py::active_query
# Phase 2: three-way diagnosis (run both m in parallel)
modal run --detach rhm/rhm_active_planning.py::active_planning --m 2
modal run --detach rhm/rhm_active_planning.py::active_planning --m 4
# Phase 3: the VoI fix (run both m in parallel)
modal run --detach rhm/rhm_active_voi.py::voi_planning --m 2
modal run --detach rhm/rhm_active_voi.py::voi_planning --m 4
```

Results JSON on the `rhm-scaling-data` volume under `rhm_active_query/`, `rhm_active_planning/`, `rhm_active_voi/` (`v8_s2_L4_m{m}_seed0/results.json`). Both `rhm_active_planning.py` and `rhm_active_voi.py` take a `--quick` flag for fast smoke tests.
