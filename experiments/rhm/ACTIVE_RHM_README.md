# Active RHM: turning an autonomous problem into a controlled one, and when that's even possible

**Status**: Done. Phases 1–3: a clean negative (mean-Δ forward model can't plan epistemic queries), a clean positive (a value-of-information head can, at m=2), and a principled null (m=4) that sharpens the whole question into a measurable boundary. **Phase 4** (2026-07-11): the *internalized*-forward-model idea from the sibling reaching arc has **no headroom** here — two cheap diagnostic probes show the belief already saturates the observation-decodable value-of-information regardless of training objective, because active-query is an inference task in disguise where acting already builds the plannable representation (**act ≈ plan**). Single seed per setting.
**Date**: 2026-07-10 (Phases 1–3); 2026-07-11 (Phase 4)
**Scripts**: [`rhm_active_query.py`](rhm_active_query.py) (arity test + first planner), [`rhm_active_planning.py`](rhm_active_planning.py) (three-way diagnosis of the null), [`rhm_active_voi.py`](rhm_active_voi.py) (the Bayesian fix), [`rhm_active_internal.py`](rhm_active_internal.py) (Phase 4 Step 0: ceiling probe), [`rhm_active_headroom.py`](rhm_active_headroom.py) (Phase 4 Step 0.5: headroom probe)
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

## Phase 4 — does the *internalized* forward-model idea port? (`rhm_active_internal.py`, `rhm_active_headroom.py`)

Phases 1–3 used a fully **decoupled** apparatus (frozen controller, external planner) — the same Run-5 discipline the sibling reaching arc later *internalized*. [REACHING_INTERNAL](../a2a_forward/REACHING_INTERNAL_README.md) co-trains an endogenous self-forecast with the operator and finds it **reorganizes the operator to be more plannable**: a model-free operator only partly supports forward-model planning (a fresh external planner scores +0.56 on it), and internalization lifts that to +1.00. The natural next move is to port that here. Rather than build the full co-training first, two cheap diagnostic probes ask whether internalization has any **headroom** — any plannable signal a plain controller leaves on the table for it to capture.

### Step 0 — the ceiling probe (`rhm_active_internal.py`): is the null belief-limited or fundamental?

The m=4 null (Phase 2/3) could be **controller-limited** (the compressed belief is a weak sufficient statistic that discarded decodable VoI) or **fundamental** (the VoI genuinely lives in the hidden content). Decompose it with two VoI heads trained on the **same** target — realized posterior entropy `H(root | controller.state(next_obs))` — differing only in their input: the controller's pooled **belief `b`** vs a higher-capacity transformer reading the **raw revealed observation** (815K params, bypassing the compressed belief). The raw-obs head is the ceiling on what is decodable-from-observation *at all*.

| m | belief-`b` corr | raw-obs **ceiling** corr | gap | random→oracle acc |
|---|---|---|---|---|
| 2 | 0.300 | 0.314 | +0.01 | 0.56 → 0.92 |
| 3 | 0.380 | 0.332 | −0.05 | 0.33 → 0.85 |
| 4 | 0.073 | 0.114 | +0.04 (→ **+1.5%** planning) | 0.20 → 0.66 |

*(per-instance query-rank corr — the Phase-3 plannability diagnostic.)* **The belief already sits at the observation ceiling at every m.** A raw-observation reader with no pooling bottleneck extracts no more query-ranking signal than the compact belief. So the belief-pooling bottleneck is **not** the limiter, and at m=4 the ceiling itself is at the floor (+0.04 corr → only +1.5% of the oracle gap in planning): the residual VoI is **fundamentally content-carried**, reachable only by the content-peeking oracle. (The belief head reproduces Phase 3: corr 0.30 @m2 → 0.07 @m4.)

### Step 0.5 — the headroom probe (`rhm_active_headroom.py`): is VoI-sufficiency a property of the *objective*?

Step 0 rules out the pooling bottleneck but leaves one door ajar: the target is defined by the controller's *own* root head, so a controller with a different objective might carry VoI differently. Reaching's internalization headroom came precisely from its baseline being a **model-free policy** (not a sufficient statistic). So hold the VoI target and the observation-ceiling **fixed** (both from the root-predictor `C_root`) and vary the one thing — train a second controller `C_pol` whose belief is shaped **only by imitating good reveals** (a query policy, never trained to predict the root; the faithful `mf` analog) — and ask whether *its* belief carries less VoI.

| m | `C_root` belief (root-predictor) | `C_pol` belief (model-free policy) | ceiling | ceiling − `C_pol` |
|---|---|---|---|---|
| 2 | 0.326 | 0.300 | 0.298 | ≈0 |
| 3 | 0.370 | 0.355 | 0.351 | ≈0 |
| 4 | 0.034 | 0.118 | 0.108 | ≈0 (all near floor) |

Planning — fraction of the random→oracle gap closed (final root prediction always via `C_root`, so only the reveal-selection signal differs):

| m | `voi_croot` planner | `voi_cpol` planner | **`policy_direct`** (raw policy) |
|---|---|---|---|
| 2 | 36% | 36% | **34%** |
| 3 | 15% | 14% | **11%** |
| 4 | ~0 | ~0 | ~0 |

**No gap opens.** The model-free-policy belief carries essentially as much VoI as the root-predictor belief, and both sit at the observation ceiling — at every m. The cleanest tell is the last column: **the raw model-free policy (`policy_direct`) plans about as well as the explicit VoI forecaster** (34% vs 36% @m2; a small dip at m3, all at floor at m4), with no systematic advantage to having a separable forecast. Changing the objective does nothing; **VoI-sufficiency is objective-independent** — any *competent* belief on this task saturates the observation-decodable VoI. There is no plannable signal left for internalization to reorganize toward.

### Why internalization has no purchase here — act ≈ plan

This explains the whole reaching-vs-RHM difference. **Internalization reorganizes a representation only when *acting* and *forecasting* are distinct computations that a model-free learner can shortcut between.**

- **Control (reaching)** genuinely has two separable objects: a **policy** (state → action) and a **dynamics model** (state, action → next state). Model-free learning can build a *reflex* policy — like an outfielder catching a ball by keeping it at a fixed visual angle — without ever building the dynamics model that planning needs. Internalization is the pressure that grows the missing model, and that gap is the headroom (+0.56 → +1.00).
- **Active-query RHM is an inference task in disguise.** Its only state variable that matters is the belief about the hidden root; queries don't change the world, they only sharpen that one belief. To *act* well you must judge which query is most informative — which **is** the forecast a planner uses (value of information, as in choosing a medical test). There is no reflex shortcut that picks a good query without evaluating queries, so a competent actor already builds the plannable representation. One belief, read by acting and planning alike — nothing to split, nothing to shortcut, nothing to internalize.

So active RHM was never going to showcase internalization, for two independent, stacked reasons: **where it is plannable (m=2), acting already plans** (no headroom); **where it is not (m=4), nobody plans** (no belief-carried signal). The reaching positive was the *exception* — a shortcut-able control task — not the rule.

## The insight (what this arc establishes)

1. **The arity *impossibility* ports; the arity *usability* does not port naively.** Belief-only forward models can't represent query-conditional structure at any capacity (as in reaching). But having that structure in a mean-Δ FM does *not* yield a planner on an epistemic task.

2. **For epistemic actions you must predict the decision variable, not the mean transition.** Value-of-information lives in the spread of possible outcomes; a point-estimate forward model discards it. A value-of-information / expected-posterior-entropy head (greedy Bayesian experimental design) is the right instrument, and it converts a total null into a decisive planner. The fidelity sweep is the proof the mean was the wrong target — more accuracy at predicting the mean bought zero planning.

3. **The controllability boundary is measurable.** An autonomous problem is actively **plannable exactly when the value-of-information of a query is carried by the agent's belief state** rather than only by the hidden content. The instrument is the **per-instance query-rank correlation** (not the aggregate correlation, which is misleadingly high). m=2 is belief-carried (corr 0.30 → planner works); m=4 is content-carried (corr 0.07 → provable null, only the content-peeking oracle wins). This is the autonomous-vs-controlled distinction turned into a number you can compute before planning.

4. **Reaching vs RHM, precisely.** Reaching worked with a plain mean-Δ FM because control actions have content-independent consequences; RHM queries do not. The same apparatus succeeds or fails depending on whether the action's payoff is deterministic-given-belief (control) or variance-in-hidden-content (epistemic).

5. **Internalizing the forward model has no headroom on an epistemic-query task, and the reason is structural (Phase 4).** Active-query RHM is inference in disguise: its only state is the belief-over-root, and acting optimally requires the same value-of-information a planner forecasts, so a plain controller — trained to predict the root *or* to imitate reveals — already saturates the observation-decodable VoI (belief corr = raw-obs ceiling at every m; a raw policy plans as well as a VoI forecaster). The reaching arc's internalization worked because control has a genuine policy-vs-world-model split a model-free learner can shortcut; querying has no such split. **Internalization reorganizes representations exactly when acting ≠ planning (control); it is redundant when acting = planning (inference).** This makes the autonomous-vs-controlled distinction sharper still: the *arity* structure ports (Phase 1), the *usability* needs the right forecast target (Phase 3), but the *internalization* phenomenon needs a control task — one where forecasting a consequence is a different computation from emitting an action.

## What this does and does not show

- **Does show**: (a) the mean-Δ FM planner is a structural null on epistemic RHM queries, robust to FM fidelity, capacity, and prediction-space (belief vs logits); (b) a VoI/EIG head closes a large chunk of the oracle gap at m=2, the first genuine active-planning positive on RHM; (c) a computable diagnostic (per-instance query-rank corr) that predicts *in advance* whether a setting is plannable, validated by the m=2/m=4 split; (d) **(Phase 4)** internalizing the forward model has no headroom here — the belief saturates the observation-decodable VoI regardless of training objective (root-predictor = model-free-policy = raw-obs ceiling), and a raw policy plans as well as a VoI forecaster, i.e. active-query is inference-in-disguise (act ≈ plan), structurally unlike the shortcut-able control task where internalization reorganizes the operator.
- **Does not show / caveats**: all planners here are greedy one-step; a single controller seed per setting per script. The Phase-4 "fundamental at m=4 / no-headroom" reading has one open loophole — the VoI target and the ceiling are both defined against the controller's *own* root posterior, so a fully controller-independent test (the **Bayes-optimal EIG** computed from the known grammar) would settle "fundamental" without that loophole. And the full internalized co-training (an endogenous `int_plan`-style planner) was deliberately **not** built: both diagnostic probes showed no headroom for it, so it was not worth the cost.

## Next steps

1. **Sweep m ∈ {2,3,4} (and budget)** to trace the per-instance-corr → planner-gain curve: does controllability degrade smoothly as VoI migrates from belief-carried to content-carried? (Phase 4 added m=3, showing belief and ceiling converge at every m.)
2. ~~**Does a stronger controller rescue m=4?**~~ *Substantially addressed by Phase 4*: the raw-observation ceiling head is a capacity-unbounded reader and it *also* floors at m=4, and Step 0.5 shows the result is objective-independent — so the null is not a weak-belief artifact. The remaining clean test is the controller-independent **Bayes-optimal EIG ceiling** computed from the known grammar (closes the "controller-defines-the-target" loophole).
3. **Non-myopic VoI.** Greedy EIG is one-step; a multi-step planner (or a head predicting entropy after several reveals) tests whether look-ahead recovers anything m=4 loses to greedy myopia. (Note: at m=2 the oracle is *also* greedy, so the unclaimed gap there is content-peeking, not myopia — lookahead is unlikely to help.)
4. **To see the internalization / self-model phenomenon in the RHM *domain*, it must be a genuine control task (act ≠ plan)** — e.g. an agent that *edits/writes* tokens toward a target root, where a forward model must simulate an edit's downstream effect — not information-gathering. This is a real departure ("reaching with RHM dynamics") and is the only way this domain would exhibit the reaching-style plannability reorganization. → **Being executed in [RHM_EDIT_CONTROL_README.md](RHM_EDIT_CONTROL_README.md)** (WIP, 2026-07-11): editing *is* plannable in belief space (arity usability ports, reversing the query null), but surfaced a *second* obstacle — off-manifold belief-gaming — that a generator-defined on-manifold action space + a cerebellar self-consistency veto largely fixes (gt≈0 → gt≈0.65, learned models only). Internalization itself still pending.
5. **Seeds + crystallize.** Firm up the single-seed Phase-4 numbers with 1–2 more seeds, then crystallize the **act ≈ plan → no-internalization-headroom** characterization as a belief (it unifies the whole reaching-vs-RHM difference).

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
# Phase 4 Step 0: ceiling probe (run m in parallel)
modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 2
modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 3
modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 4
# Phase 4 Step 0.5: headroom probe (stagger launches ~25s apart to avoid the app-create rate limit)
modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 2
modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 3
modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 4
```

Results JSON on the `rhm-scaling-data` volume under `rhm_active_query/`, `rhm_active_planning/`, `rhm_active_voi/`, `rhm_active_internal/`, `rhm_active_headroom/` (`v8_s2_L4_m{m}_seed0/results.json`). `rhm_active_planning.py`, `rhm_active_voi.py`, `rhm_active_internal.py`, and `rhm_active_headroom.py` all take a `--quick` flag for fast smoke tests.
