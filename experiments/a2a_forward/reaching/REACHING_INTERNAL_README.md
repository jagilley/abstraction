# Internalized forecasting on the reaching control task — endogenous self-forecast vs decoupled external planner

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../../ideas/self_model_needs_a_loop.md) (internalization; map-vs-model; the runnable-simulator question)
**Parent experiments**: [ACTIVE_VISION_README.md](ACTIVE_VISION_README.md) (the control-regime port + the EXTERNAL-planner positive this internalizes), [README.md](../README.md) (the feedforward a2a arc)
**Code**: `mnist_reaching_internal.py`, `reaching_vit.py`
**Status**: Done, positive + two rounds of sharpening (single seed per dataset: MNIST + Fashion; both agree). **Diagnostics round 1 (2026-07-10)** corrected "both couplings drift to advisor": the *planner* coupling produces a transferable, task-relevant self-model. **Reading-ladder round 2 (2026-07-10)** corrected "only injection produces an advisor": only *raw-scalar* injection does — injection with the **thalamic-relay gate** (`CerebellarGate`, the gate the prior a2a arc actually used) or an explicit readout is also transferable. See [Collusion vs selectivity](#collusion-vs-selectivity-diagnostics-the-advisor-reading-was-coupling-specific-2026-07-10) and [The reading ladder](#the-reading-ladder-transferability-is-gated-by-how-much-the-forecast-is-read-2026-07-10).
**Date**: 2026-07-10

## The question

The reaching positive ([ACTIVE_VISION_README](ACTIVE_VISION_README.md#causal-use-in-the-control-regime-the-null-flips-when-the-objective-is-endogenous-2026-07-10)) was produced by a **fully decoupled** apparatus (Run-5 discipline): the operator `G` is trained model-free and **frozen**; the forward model `FM` is trained on frozen transitions (pure Δ-prediction MSE); an **external** `argmax` planner reads a frozen `FM` + frozen probe. Nothing about `G`'s representation is ever shaped by the fact that a forecast will be read off it, and the `FM` is never shaped by being *used*. The external `argmax` is a scaffold.

This experiment removes that decoupling — makes the self-forecast **endogenous** and co-trains it with the operator — and asks the one question the decoupled setup cannot: **what reorganizes when the forecast is used from inside the loop, rather than by an external planner reading a frozen module?** The discriminator of interest is map-vs-model (a separable, causally-used *forecast* vs an absorbed *legibility map*), now on a task where the forecast is genuinely answer-relevant (so it isn't confounded by the feedforward degeneracy / content-blindness that muddied the perception-domain versions).

## What we built

Four conditions, **matched init seed and identical random-rollout coverage**, so the only variable is the coupling (both internalization mechanisms built as siblings):

- **`mf`** — model-free policy head only. This *is* the decoupled EXT operator; baseline + CKA identity check.
- **`int_plan`** — **differentiable internal planner**: actions scored by rolling the FM, `logits_a = value(s + FM(s,a))/τ`, imitation-trained end-to-end. Co-trains `G` + FM + value head. The forecast **selects behavior**. Direct upgrade of the external `argmax`.
- **`int_inject`** — **fed-back input modality**: `s_{t+1} = G(s_t + obs + gate·FM(s_t, a_t))`, action still from the policy head, zero-init learned gate. The a2a closed-loop analog on a control task.
- **`int_plan_state`** — arity-1 internal planner (command-blind). The sanity floor.

`fm_aux_lambda = 1.0` anchors the co-trained FM to real Δ (to *try* to keep it veridical). Readout battery, applied identically to every frozen operator: (1) native controller + **decoupled EXT planner using a fresh, independently-trained FM** on this operator; (2) FM veridicality (co-trained FM Δ-cos vs a fresh frozen FM's on the same operator); (3) self-consistency (on-traj cos); (4) reorganization (CKA vs mf, position-probe acc, `cmd_rel_spread`); (5) **map-vs-model**: a linear policy readout trained on the frozen operator (no FM channel), plus the direct ablation (zero the forecast / injection off).

## Headline (both datasets, `norm_progress` = fraction of start-distance closed; 1.0 = reached)

| metric | mf | int_plan | int_plan_state | int_inject |
|---|---|---|---|---|
| native progress (MNIST / Fashion) | +1.0 / +1.0 | +1.0 / +1.0 | **−0.04 / −0.04** | +1.0 / +1.0 |
| **EXT planner (fresh FM) on this operator** | +0.56 / +0.54 | **+1.00 / +1.00** | +0.61 / +0.53 | +0.70 / +0.64 |
| ablate forecast / injection | — | **−0.05 / −0.05** | −0.06 / −0.06 | +0.56 / +0.54 |
| linear-readout progress (map probe) | +1.0 / +1.0 | +0.43 / +0.73 | −0.26 / −0.26 | +0.64 / +0.57 |
| CKA vs mf | 1.00 | **0.06 / 0.25** | 0.05 / 0.04 | 0.43 / 0.34 |
| position-probe acc | 0.67 / 0.71 | 0.73 / 0.69 | 0.98 / 0.98 | 0.65 / 0.68 |
| fresh-frozen FM Δ-cos | 0.86 / 0.87 | 0.73 / 0.69 | 0.98 / 0.98 | 0.85 / 0.89 |
| **co-trained FM Δ-cos** | — | **0.38 / 0.15** | 0.99 / 0.91 | **0.36 / 0.40** |
| `cmd_rel_spread` | 0.51 / 0.65 | 0.72 / 0.86 | 0.02 / 0.02 | 0.12 / 0.09 |
| gate | — | — | — | 0.35 / 0.37 |

## Robust findings (hold on both datasets)

1. **Arity can't be bought, even co-trained end-to-end.** `int_plan_state` sits at the random floor (−0.04) while `int_plan` solves. Internalizing a *command-blind* forecast does not manufacture a planner — the arity impossibility ([ACTIVE_VISION_README](ACTIVE_VISION_README.md)) survives closing the loop.

2. **The headline positive — internalization makes the operator dramatically more *plannable*.** A *fresh, decoupled* external planner (its own independently-trained FM) scores **+0.55 on the model-free operator but +1.00 on the co-trained operator**, and CKA vs mf collapses to **0.06 / 0.25**. This is not the co-trained FM (the external planner doesn't use it) — the **representation itself** reorganized so that forward-model planning over it works. Mechanism: in `int_plan`, gradient from "the planner must rank the oracle action first" flows back into `G`, pressuring it to emit states legible to a forecast-based planner — a pressure the model-free operator never felt.

3. **It's *task-relevant* legibility, not uniform.** Position decodability rises (probe 0.67→0.73) while full-state forward-predictability *falls* (fresh-FM Δ-cos 0.86→0.73). The operator concentrated the value-relevant variable (fovea position) into a cleanly plannable form and let the rest of its dynamics get *more* complex. **Plannability ≠ forward-predictability.**

4. **The endogenous forecast has a low full-vector Δ-cos (0.15–0.40 vs a fresh FM's 0.69–0.89).** *Originally read as "both couplings drift simulator→advisor" — the diagnostics below correct this*: for the **planner** coupling the low cosine is mostly **correct task-relevant selectivity** (the FM predicts what matters, drops the rest) and the forecast is a **transferable self-model**; only for the **injection** coupling is it a genuine advisor. The `fm_aux_lambda=1.0` anchor is overridden by the planner CE either way (so "drift" is under one loss-weighting). See [Collusion vs selectivity](#collusion-vs-selectivity-diagnostics-the-advisor-reading-was-coupling-specific-2026-07-10).

5. **Map and model coexist.** The running loop *depends* on the forecast — ablate → floor for `int_plan` (+1.0→−0.05); injection-off drops `int_inject` (+1.0→+0.54) — a causally-used **model**. Yet a linear readout on the frozen operator also recovers a lot (int_plan +0.43 / +0.73) — legibility co-developed, a latent **map**. These aren't in tension: the readout asks *"is the info present?"* (yes), the ablation asks *"does the system as trained use that route?"* (no — it built itself to depend on the forecast channel). The trained system *runs* as a model; a map is *available* in its representation. Same legibility/causal-use dissociation the feedforward arc found, now on the control task.

6. **Injection reproduces the a2a dependency signature.** The gate opens (|gate| 0→0.35) *and* the operator becomes dependent: with injection off it scores +0.54, **worse than the model-free operator that never had it** (+1.0) — the wake-sleep dependency problem, reproduced in the control regime.

## Collusion vs selectivity diagnostics — the "advisor" reading was coupling-specific (2026-07-10)

The low full-vector Δ-cos (finding 4) is ambiguous: it could mean **(i) collusion** — the FM output is a value-shaped signal untethered from the real state, working only because its own co-trained value head adapted to it — or **(ii) task-relevant selectivity** — the FM faithfully predicts the *value-relevant* subspace and correctly discards the ~120 value-irrelevant dims, which drags the *full-vector* cosine down while the forecast is genuinely veridical on what matters. Two diagnostics separate them (`subspace_delta_cos` + a frozen-independent-consumer planning test in `mnist_reaching_internal.py`):

- **Subspace-resolved Δ-cos** — split the FM's Δ prediction into a *fresh, independent* position probe's direction (`on_dir`, what the value readout uses) vs its orthogonal complement (`off_dir`). Selectivity ⇒ `on_dir` ≫ `off_dir`; collusion ⇒ `on_dir` also low.
- **Frozen-independent-consumer** — run the external `argmax` planner using the **co-trained FM** but a **fresh probe it never co-adapted with**. Veridical-on-the-relevant-subspace ⇒ still plans; collusion ⇒ floor/negative.

| condition | on/off ratio (M / F) | fresh-probe transfer (M / F) | verdict |
|---|---|---|---|
| **`int_plan`** | **1.70 / 2.96** | **+0.67 / +0.51** | task-relevant **self-model** — selective *and* transferable |
| **`int_inject`** (raw scalar gate) | 0.78 / **0.39** | **−0.35 / −0.43** | **advisor** — anti-selective, non-transferable (worse than random). *But this is the crude-gate instantiation; the reading-ladder section shows a proper gate fixes it.* |
| `int_plan_state` | 0.95 / 1.04 | −0.06 / −0.06 | **veridical but useless** — Δ-cos 0.91–0.99, yet plans at the floor |

**Reading (both datasets agree):**

- **The planner forecast is *not* an advisor — it is a transferable, task-relevant self-model.** It predicts the value-relevant direction 1.7–3.0× better than the orthogonal junk, and a *fresh independent* probe plans with it to +0.5–0.7 (floor −0.05). The low full-vector cosine is mostly the FM correctly declining the value-irrelevant dims, plus a modest co-adaptation slack (works a bit better with its own head, +1.0 vs +0.6). This is exactly the "the joint system self-models as well as the task demands" outcome — the forecast is a *separable, queryable-by-anyone* object.
- **The *raw-scalar* injection forecast is an advisor** (only this one). Anti-selective (off > on) and *worse than random* when read by an independent probe — a private useful-perturbation for its own operator. **But this turned out to be an artifact of the crude gate**, not of injection itself — see the reading ladder below.
- **The arity-1 control nails the dissociation.** A *perfectly veridical* forecast (0.91–0.99) that is *useless* for control (floor), because its degenerate operator collapsed to autonomous dynamics (`cmd_rel_spread` 0.02) with no command-conditionality to plan over. **Veridicality ≠ control-usefulness.**

**Hypothesis this opened (tested in the next section):** the *consumption mode* — specifically *how much the forecast is read* — determines whether internalization yields a **model** or an **advisor**. In `int_plan` the forecast is consumed by an *explicit readout* (a value head reads it), so gradient shapes it to be legibly readable → transferable. In raw-scalar `int_inject` the forecast is *summed into the recurrent state* through a single unbounded scalar and consumed implicitly, so it never has to be externally legible → private. But note the prior a2a arc's injection used a *learned projection* gate, not a raw scalar — so the honest test is a **reading ladder**, next.

## The reading ladder — transferability is gated by how much the forecast is read (2026-07-10)

The original `int_inject` used a **raw unbounded scalar** gate (`inject = gate·FM`, `gate∈ℝ`) — the crudest possible consumption, and *cruder than the gate the prior a2a arc actually used* (`CerebellarGate`, a zero-init learned linear projection = the "thalamic relay" that selects/reshapes which forecast directions to inject). That scalar gate is a confound: it strips out the "reading" the projection performs. So we ran a ladder holding the task fixed, varying only *how much the forecast is read* (`mnist_reaching_internal.py`, conditions `int_inject` / `int_inject_proj` / `int_inject_readout`):

| consumption of the forecast | freshprobe transfer (M / F) | on/off selectivity (M / F) | verdict |
|---|---|---|---|
| `int_plan` — readout **drives the action** | +0.67 / +0.51 | 1.70 / 2.96 | model (reference) |
| `int_inject_readout` — explicit readout, still injected | **+0.60 / +0.75** | 1.56 / 1.12 | **transferable** |
| `int_inject_proj` — **`CerebellarGate` thalamic relay** | **+0.40 / +0.46** | 0.13 / 0.47 | **transferable** |
| `int_inject` — **raw scalar** gate | **−0.35 / −0.43** | 0.78 / 0.39 | advisor (only this one) |

**Reading (both datasets agree):**

- **Raw-scalar injection is the *only* advisor.** Both the **thalamic projection gate** (the gate the prior arc used) and the **explicit readout** restore transferability — from worse-than-random (−0.4) up to at/above the `int_plan` model reference (+0.4 to +0.75). So the earlier "injection → advisor" reading was **an artifact of the crude scalar gate**, not a property of injection. *"Injection privatizes the forecast" is retracted.*
- **Transferability tracks the *amount of reading*, not a binary readout-vs-injection.** Minimal read (raw scalar) → advisor; a learned read-then-reinject (`proj`) or read-to-scalar (`readout`) → transferable model. A learned projection is itself enough "reading" to make the forecast a public object.
- **Transferable ≠ task-aligned, though.** The `proj` gate transfers for *planning* (+0.4) but is *not* subspace-selective (on/off 0.13/0.47 — its projection reshapes the raw forecast into a usable-but-different gauge). Only `int_inject_readout` (and `int_plan`) are transferable *and* selective (on/off > 1). So *any* reading restores usability; reading *to an interpretable scalar* additionally restores task-aligned fidelity.
- **Caveat / scope**: this does not retroactively overturn the prior arc's "advisor, not simulator" conclusions — those came from different metrics (FM cosine, self-knowledge R²) on *autonomous perception* tasks, not this transfer test on a control task. What it kills is the "injection-per-se privatizes" story.

## Interpretation — where this leaves the "self-model" question

On the [self_model_needs_a_loop](../../../ideas/self_model_needs_a_loop.md) discriminators, **causal necessity now passes cleanly** in the control regime: an endogenous, causally-load-bearing self-forecast that measurably reorganizes the representation. But the honest sharpening is the payload:

- **The planner path produces a genuine (task-relevant) self-model, not an advisor** (per the diagnostics above): selective on the value-relevant subspace and transferable to an independent consumer. Its low full-vector Δ-cos is correct selectivity, not untethering. So on the [self_model_needs_a_loop](../../../ideas/self_model_needs_a_loop.md) discriminators the planner path yields a *separable, causally-used, queryable* forecast — closer to a "model" than we first credited. (What it has *not* shown is *full-state* veridicality or *fixed-point self-consistency* of the whole operator with its own forecast — the forecast is faithful on what the task uses, silent elsewhere.)
- **This is architecturally the *right* shape, not a shortfall (division of labor).** If the main model perfectly internalized the forward model, the cerebellum would be redundant — biology keeps it. The map-model coexistence we measured *is* that division of labor: the operator's latent linear-decodable plan is the coarse "vague idea of what the FM would say," the separable ablation-critical FM is the high-fidelity thing it calls. What looked like "we didn't fully internalize" is plausibly the target.
- **Where the "true simulator" lives is reframed by the control regime.** The main forward pass runs *one* real trajectory; it cannot answer "what if I took action `a` instead" without taking it. The thing that simulates *alternatives off to the side* is the **separable FM** (queryable without committing — the planner's whole trick), doing something the forward pass structurally can't. The planner's FM *does* do this veridically enough to transfer; the injection's does not.

**Net.** Endogeneity produces large, dataset-robust representational change (plannability jump, CKA collapse, real dependency) and a **causally-used, transferable, task-relevant self-model** whenever the forecast is *read* — via the planner's action-selecting readout, an explicit decode head, *or* the thalamic-relay projection gate. Only a raw-scalar gate that reads nothing yields a private advisor. The consumption-mode question is thus **answered** (transferability ∝ amount of reading). The remaining honest gaps are (a) full-state veridicality / operator-level fixed-point self-consistency (the forecast is faithful only on the value-relevant subspace), and (b) whether the transferable one-step forecast **composes** into a multi-step runnable simulator.

## What this does and does not show

- **Does show**: closing the loop with an endogenous self-forecast reorganizes the operator toward plannability (large, robust across MNIST + Fashion), makes the forecast causally load-bearing (ablation → floor), and reproduces the a2a dependency/gate signature on a control task. Whenever the forecast is *read* (planner readout, explicit decode head, *or* the thalamic-relay projection gate) it is a **transferable, task-relevant self-model**; only a raw-scalar gate that reads nothing yields a private advisor. Both robust across datasets.
- **Does not show**: *full-state* veridicality, operator-level *fixed-point self-consistency*, or a *composable multi-step* runnable simulator. The planner forecast is faithful only on the value-relevant subspace and only tested one-step (greedy planner).
- **Caveats**: reaching is easy enough that `mf` saturates native progress (+1.0 everywhere), so the science is in the *representational / diagnostic* metrics, not the behavioral headline. Single seed per dataset. Each condition trains its **own** operator, and `int_plan_state`'s came out degenerate (cmd-spread 0.02, probe 0.98) — so cross-condition *representational* metrics are muddier than the *behavioral* ones; the native/ablate/fresh-probe results are the trustworthy part. The subspace `on_dir`/`off_dir` metrics use slightly different reductions (scalar-sequence cosine vs mean vector cosine), so read them as a *within-condition* on-vs-off contrast, not an absolute scale; the frozen-independent-consumer planning test is the cleaner, behavioral version of the same question.

## Next steps

1. ~~**Why does consumption mode gate transferability?**~~ **Done** (the reading ladder above): transferability ∝ *amount of reading*; the raw-scalar gate was the confound; a thalamic-relay projection or an explicit readout both restore it. Remaining sub-question: `proj` transfers but is *not* task-aligned (on/off < 1) while `readout` is both — worth a small follow-up on whether task-aligned selectivity specifically requires reading *to an interpretable scalar*.
2. ~~**Multi-step lookahead planner (the live next step).**~~ **Done** → [REACHING_LOOKAHEAD_README.md](REACHING_LOOKAHEAD_README.md). Built the endogenous N-step MPC + composed-veridicality readout on 3 lookahead-forcing geometries. The A/C question **split into two axes**: deep veridical composition IS achievable (horizon 8, pos-acc 0.99) but **only on the looped operator** (outcome A, loop-gated — the "doesn't compose" prior was a one-step-*training* artifact); and **veridicality ⊥ control-usefulness** (the value-shaped co-trained forecast plans best despite worst veridicality, beating even a perfect-simulator control — outcome C for *control*). Maze is planner-bound, not simulator-bound. New live next step there: a stronger planner (CEM/beam + geodesic/learned value) to convert the deep simulator into hard-lookahead behavior.
3. **Seeds + a de-saturated task.** Multiple seeds, and a reaching variant hard enough that `mf` does *not* trivially solve it, so the behavioral headline stops being saturated and the planner's contribution shows up in progress, not only in the diagnostic readouts. (The lookahead task in step 2 doubles as this.)

## Reproduction

```bash
cd experiments/
# Round-1 conditions (planner vs injection vs arity-1 control):
modal run --detach a2a_forward/reaching/mnist_reaching_internal.py::internal_reaching --dataset mnist
modal run --detach a2a_forward/reaching/mnist_reaching_internal.py::internal_reaching --dataset fashion_mnist
# conditions default to mf,int_plan,int_inject,int_plan_state; fm_aux_lambda default 1.0

# Round-2 reading ladder (raw scalar vs thalamic proj vs explicit readout); --tag-suffix
# avoids clobbering the round-1 results:
modal run --detach a2a_forward/reaching/mnist_reaching_internal.py::internal_reaching --dataset mnist \
  --conditions "mf,int_plan,int_inject,int_inject_proj,int_inject_readout" --tag-suffix "_ladder"
```

Results JSON under `/data/a2a_forward/mnist_reaching_internal/{dataset}_content_g7_K14_4H128D{tag_suffix}/results.json` (round-2 ladder under the `_ladder` suffix).

## Files

- `mnist_reaching_internal.py` — this experiment: coupling conditions with matched coverage — round 1: `mf` / `int_plan` / `int_inject` / `int_plan_state`; round-2 reading ladder: `int_inject` (raw scalar) / `int_inject_proj` (`CerebellarGate` thalamic relay) / `int_inject_readout` (explicit position-decode head on the forecast, still injected). Gate type and readout are params of `train_int_inject`. Readout battery (native + decoupled-EXT-planner behavior, FM veridicality co-trained vs fresh-frozen, on-traj self-consistency, CKA vs mf, position probe, `cmd_rel_spread`, linear-readout map probe, forecast/injection ablations) + the **collusion-vs-selectivity diagnostics** (`subspace_delta_cos` = on-dir vs off-dir Δ-cos against a fresh independent probe direction; `cotrained_fm_freshprobe_progress` = co-trained FM read by a fresh independent probe in the external planner). `tag_suffix` param keeps runs from clobbering each other.
- `reaching_vit.py::ReachingLoopedViT` — the shared controller (unchanged from the parent; its `step(..., inject=)` hook carries the injection for `int_inject`).
