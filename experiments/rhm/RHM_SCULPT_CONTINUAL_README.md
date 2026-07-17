# Continual internalization on RHM sculpting: is planning a ratchet on its own substrate?

**Status**: WIP, single seed, one round of the loop done (5 rounds × 3 arms). Clean result: **weak ratchet — value-iteration is the compounding engine; the representational headroom is essentially one-shot.**
**Date**: 2026-07-13
**Script**: [`rhm_sculpt_continual.py`](rhm_sculpt_continual.py) (`sculpt_continual` entrypoint).
**Parent**: [RHM_SCULPTING_README.md](RHM_SCULPTING_README.md) Stage 5 (one-shot internalization — the grounded planner reshapes the belief once, then freeze). This iterates that.
**Cousins**: [a2a REACHING_INTERNAL](../a2a_forward/reaching/REACHING_INTERNAL_README.md) (the internalization move), [RHM_LATENT_LOOP](RHM_LATENT_LOOP_README.md) (the frontier-moving/exhausts prediction), [a2a CANCELLATION](../a2a_forward/CANCELLATION_README.md) (the entanglement/privatization diagnostic).

---

## The question

Stage 5 internalized the planning loop **once**: a grounded differentiable planner reshaped the controller's belief so it became FM-rollable (fresh-FM top1 0.36→0.52, PR 6.7→17, clean-channel gap flipped positive). This asks whether that's a **ratchet**:

> Does a better-planning system produce a more *plannable* belief, which produces a better planner… compounding toward the DP optimum (=1.0) — or did the one-shot reshape already extract the representational headroom, leaving only ordinary policy improvement?

## Definition

**Continuous internalization = iterate the internalization loop so belief, FM, value, and behaviour-policy co-evolve, each round's improved planner supplying the data + targets that reshape the next round's belief.** A warm-started wake–sleep / policy-iteration loop:
- **wake (collect):** run the current latent planner as behaviour policy → rollouts → grounded terminal-success labels on the *visited (improving)* distribution.
- **sleep (consolidate):** continue-train the MC value on the new labels (the policy-iteration engine that breaks Stage-5's "behaviour-policy success 0.34 caps the value" ceiling); then **re-internalize** the belief (grounded planner against the improved *frozen* value + the fixed DP best-move `k*`), shaping belief + FM.

The value stays a **pure MC critic** (improved only by policy iteration); the belief is shaped to be legible to that improving critic in a way that matches ground truth. The `k*` teacher is DP-computed on a fixed broad distribution — a **stable ground-truth teacher, not self-distillation** (the same reason grounding was the pivot in Stage 5), so the loop can't wirehead its own value.

## Design — nested arms sharing round 0, matched per-round budget

Single controlled variable = *how much of the loop touches the belief across rounds*.
- **D** = round-0 only (= Stage-5 frozen floor; recovered as C's round 0).
- **C** value-iteration only: belief = frozen parser; iterate value + data.
- **B** one-shot internalize + VI: internalize the belief **once** (round 0), freeze it; iterate value.
- **A** continuous: re-internalize the belief **every** round + iterate value.

Contrasts: **C−D** = value-iteration effect; **B−C** = one-shot-internalization effect (Stage 5, under VI); **A−B = the continuous-internalization effect** (does re-shaping the belief each round, against the continually-improving value, add compounding beyond one-shot?).

Two new pieces vs Stage 5: `_collect_value_planner` (behaviour policy = greedy value+FM latent step + ε, re-grounded — the policy-iteration data engine) and `_reinternalize_belief` (Stage-5's planner shaping with the **value held frozen**, so value stays a clean MC critic).

## Results (latent beam @ w256; DP=1.0, strong reflex 0.585, Stage-5 one-shot ~0.591)

| arm | r0 | r1 | r2 | r3 | r4 | r5 | PR r0→r5 | fresh_top1 r0→r5 | behSucc |
|---|---|---|---|---|---|---|---|---|---|
| **A** continuous | 0.518 | 0.771 | 0.802 | 0.802 | 0.804 | **0.818** | 12.8→10.1 | 0.60→0.54 | ~0.48 |
| **B** one-shot+VI | 0.518 | 0.786 | 0.795 | 0.802 | 0.785 | **0.795** | 12.8 (frozen) | 0.60→0.48 | ~0.48 |
| **C** value-iter only | 0.476 | 0.643 | 0.601 | 0.635 | 0.630 | **0.645** | 6.7 (frozen) | 0.31→0.36 | ~0.30 |

Final-round width sweep (token/latent/gap @ w256): A 0.817/0.818/**+0.001** · B 0.812/0.795/−0.017 · C 0.667/0.645/−0.022.

### Three findings, in order of magnitude

1. **Value-iteration is the big compounding lever (+0.25 in one round).** Every arm jumps hard at r0→r1 (A 0.52→0.77) when collection switches to the improving planner-behaviour-policy and the value re-fits on higher-success rollouts. This is the lever Stage 5 flagged as remaining, and it lifts the beam from ~0.59 to **~0.82** — closing most of the gap to the DP optimum that one-shot internalization left open.
2. **Belief internalization is a large *one-shot* ceiling-setter (+0.15), not a ratchet.** B−C = +0.150: internalizing the belief *once* raises the whole plateau from ~0.64 (parser belief) to ~0.80, by raising the behaviour-policy-success ceiling (0.30→0.48), which is what lets value-iteration climb higher. C (parser belief) caps at behSucc ~0.30 no matter how many VI rounds you run — the belief sets the value-iteration ceiling.
3. **Continuous re-internalization adds only +0.023 — a weak ratchet.** A−B = +0.023; A creeps up monotonically (0.771→0.818) while B is flat, so it's *real* but an order of magnitude smaller than levers 1–2. Tellingly, A's re-internalization **consolidates rather than expands** (PR 12.8→10.1): after round-0's expansion (6.7→12.8), further re-internalization against the sharper value prunes back toward the value-relevant subspace for a small beam gain. **The plannability frontier saturates in one pass** — RHM_LATENT_LOOP's "compounds while the frontier climbs, then exhausts at the ceiling," now confirmed on a *fixed* task.

### Entanglement watch (the a2a-CANCELLATION diagnostic) — clean

No privatization. The fresh (independent) FM stays **+0.15 to +0.20 better than the co-trained FM at every round** (A: fresh−cotr +0.18→+0.15) — the belief never becomes privately dependent on its own co-trained forecast; an independent consumer always plans it *better*. (The co-trained FM being the *worse* one-step ranker yet driving a fine beam is the REACHING_LOOKAHEAD "value-shaped forecast, worst veridicality, best planner" signature, not a pathology.) So the summation-dependency failure mode CANCELLATION warns about does **not** appear here — the grounded DP teacher keeps the loop honest, and RHM (gauge-free, unlike their low-rank MNIST) is where this diagnostic can actually run.

## Verdict

Planning is **at most a weak ratchet on its own representational substrate.** The representational headroom is extracted in a *single* internalization pass (a large ceiling-setting lever); the actual round-over-round compounding is **value-iteration**, which itself saturates in ~1 round once the belief's behaviour-ceiling is hit. The plateau (~0.82) sits well past the strong reflex (0.585) but ~0.18 below the DP optimum. "One good reshape + one big value-iteration jump," not a self-reinforcing spiral. The useful positive: Stage-5 one-shot internalization was the right and *sufficient* representational move, and the route to the DP optimum from here is **better value/search**, not more re-internalization.

## Caveats & load-bearing gotchas (read before resuming)

- **Single seed, single setting** (L=4, m=2, c=3, R=5). Directional.
- **r0→r1 conflates value-iteration with a collector switch.** Round 0 collects value data with `_collect_value_sculpt` (controller-greedy behaviour); rounds 1+ use `_collect_value_planner` (planner-greedy). So the big r0→r1 jump mixes "better value" with "better collector." All arms share it and C (frozen belief) also jumps, so the VI effect is real — but for a clean VI curve, make round 0 also use the planner collector (needs a bootstrap value first).
- **`fresh_fm_top1` is not a pure belief metric.** `_fm_check`'s top1 couples value+FM+belief; when the value changes across rounds it shifts even with a frozen belief (see B's fresh_top1 declining 0.60→0.48 despite a frozen belief). The robust quantity is the **fresh−cotrained diff** (same value both sides) — that's what the entanglement watch uses.
- **Grounded `k*` buffer is fixed/broad**, not policy-following (a deliberate conservative first cut to avoid distribution collapse). A policy-following buffer (shape the belief on the states the improving planner actually visits) is an untested lever that could strengthen the ratchet — or collapse it.
- **PR is two-phase** (expand in round 0, consolidate thereafter in A); don't read the round-0 PR expansion as ongoing.

## Where to resume / next steps

1. **Non-stationary is the interesting regime.** On a *fixed* task the belief frontier exhausts in one pass (finding 3). RHM_LATENT_LOOP argues sustained compounding needs **abstraction-novelty**, not more i.i.d. data. The sharp test: after the plateau, introduce a deeper `L` / fresh rule set and check whether the ratchet **re-opens** (belief frontier climbs again). This is the sculpting instance of the whole RHM ratchet-vs-plateau program.
2. **Push value/search to the DP optimum.** The plateau is value/search-bound, not belief-bound (findings 1–2). Value-iteration to convergence + a stronger planner (CEM/beam, wider) should climb from ~0.82 toward 1.0 — the "what actually binds" decomposition (beam width vs move-manifold vs value) at the plateau.
3. **Policy-following grounded buffer.** Recompute `k*` on the improving planner's visited distribution (with a collapse guard) — does shaping the belief where good planning goes strengthen the weak ratchet?
4. **Fix the r0 collector confound** (bootstrap value, then use the planner collector from r0) for a clean value-iteration curve.

## Reproduce

```bash
cd experiments/
# smoke (2 rounds, tiny)
modal run rhm/rhm_sculpt_continual.py::sculpt_continual --m 2 --quick
# full: 5 rounds x arms A (continuous) / B (one-shot+VI) / C (value-iter only)
modal run --detach rhm/rhm_sculpt_continual.py::sculpt_continual --m 2
```

Results JSON on the `rhm-scaling-data` volume under `rhm_sculpt_continual/` (tagged `..._R{rounds}_{arms}`); per-round records + final width sweep per arm.
