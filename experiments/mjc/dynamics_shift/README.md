# Cut #3 — Operator intervention: reward-free re-adaptation after a dynamics shift

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/physical_control_substrate.md](../../../ideas/physical_control_substrate.md)
**Cousins**: [OOD_ROBUSTNESS](../../a2a_forward/OOD_ROBUSTNESS_README.md) (the static-slope framing this deliberately moves away from), [REACHING_LOOKAHEAD](../../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md) (the composition horizon that reappears here)
**Code**: `dynamics_shift.py` (env: [`../pusher_env.py`](../pusher_env.py)) · File index: [FILES.md](FILES.md)
**Status**: done. The substrate most later cuts build on. **Date**: 2026-07-16.
**Builds on this**: [`value_shaping/`](../value_shaping/README.md) (the stationary learning-layer control for this substrate) · [`directed_readapt/`](../directed_readapt/README.md) (directed vs this cut's *undirected* reward-free collection) · [`meta_adapt/`](../meta_adapt/README.md) (Cuts #4–#4e: one re-adaptation → a *distribution* of dynamics) · [`curiosity_control/`](../curiosity_control/README.md) (where to collect, as the frontier drifts) · [`online_value_loop/`](../online_value_loop/README.md) (the fully-online two-timescale loop) · [`ballistic/`](../ballistic/README.md) (turns this cut's `replan_every` biological reading into the causal knob)

---

**The move MuJoCo uniquely enables** (vs language/RHM/MNIST): **intervene on the transition operator** (the dynamics) while holding the task fixed. We use it to test the

**Factorization claim**: a model-based agent factorizes control into (world-model: *what the world does*) × (planner/value: *what I want*). A dynamics shift corrupts **only the world-model factor**. So the MB agent re-adapts from **reward-free self-supervised** interaction — every `(s,u,s')` is a dense labeled example of the new dynamics, after which the fixed planner is immediately re-optimal — while a model-free amortized policy `π(s,g)→u` entangles both factors and gets **no signal** from reward-free transitions (it needs reward). This is NOT the static "degradation slope" (a shifted operator makes the stale FM itself wrong — theoretically shaky, and would just replicate a2a [OOD_ROBUSTNESS](../../a2a_forward/OOD_ROBUSTNESS_README.md)); the new, mechanistic result is a **re-adaptation sample-efficiency dissociation**.

**Task**: goal-conditioned, **puck-free, momentum-dominated** pusher reaching (low `joint_damping` → the pusher glides → reaching needs anticipatory braking; state `[px,py,vx,vy]`, command = force, short horizon so overshoot is penalized; `build_xml(with_puck=False)`). **Agents** (both from the same d0 knowledge, fair): MB = arity-2 FM `f(s,u)→Δs` ([Cut #2](../arity_torque/README.md) idiom, self-supervised on random d0 interaction) + a **CEM MPC** rolling the FM (value = running distance-to-goal + terminal-velocity penalty — dynamics-independent, **fixed across d0/d1**); MF = a **committing motor-program policy** `π(s,g)→(re-step action sequence)`, behavior-cloned from the MB planner's committed plans and executed with the **same `replan_every` open-loop commit as MB**. So both agents emit committed k-step motor programs from `(s,g)` under an *identical* feedback protocol — the *only* difference is MB generates the program by rolling an adaptable FM, MF by an amortized net (this removes the commit-vs-reactive confound; a motor-program net needs more capacity/data than a reactive policy to reach d0 competence). **Operator shift** d0→d1: a **friction/drag collapse** (`joint_damping 2.0 → 0.05`, "high-drag → ice") — at d0 the drag does the stopping so the cloned policy barely learns to brake; on ice both committed programs overshoot, but only the FM can be re-fit from reward-free sliding.

**Key design (load-bearing)**: the MB planner **commits to `replan_every`-step open-loop segments** (default 8) rather than re-planning every step. This is essential — with per-step re-grounding, feedback *substitutes* for the model (a stale FM is tolerated — the RHM "search-substitutes-for-the-model" regime), so the shift doesn't bite and there is nothing to recover (an early per-step run was a near-null). Committing makes the world-model **load-bearing**: a stale model's errors accumulate over the open-loop window and it overshoots, so the shift bites and reward-free refit genuinely recovers. Quantified by the ablation below.

## Result (metric = final ‖pusher_pos − goal‖, lower better)

| stage | finding |
|---|---|
| **d0 competence** | MB 0.036, MF 0.036 (both solve; random floor 0.65) |
| **d0 competence** | MB 0.036, MF 0.036 (**identical** — fair BC) |
| **shift bites both, equally** (fig1) | d1 zero-shot MB-stale 0.087, MF 0.087 (**identical** — both are committing motor-program generators, so the shift degrades them equally; the protocol is matched); FM-refit **oracle ceiling 0.021** |
| **reward-free re-adaptation** (fig3 — headline) | **MB recovers to the ceiling from ~50 reward-free transitions**; **MF is flat at 0.087** (reward-free `(s,u,s')` gives a program-generator no target — by construction) |
| **MF needs reward, ~240× more** | reward-driven REINFORCE reaches the ceiling only in the ~10⁴ reward-*labeled*-transition range (~500+ episodes), and noisily |

The factorization made physical: the operator shift corrupted only the world-model factor, and only the agent that *has* that factor can fix it from self-supervision.

## The `replan_every` ablation — the effect is gated by re-groundability (`run_replan_ablation`)

Sweeping the MB planner's open-loop commit length (FMs trained once — they are replan-independent — only the MPC eval varies):

| replan_every | 1 | 2 | 4 | 6 | **8** | 12 |
|---|---|---|---|---|---|---|
| stale FM | 0.022 | 0.029 | 0.048 | 0.068 | **0.093** | 0.084 |
| oracle FM | 0.016 | 0.015 | 0.014 | 0.017 | 0.021 | 0.033 |
| **reward-free benefit** (stale − refit) | **+0.005** | +0.012 | +0.030 | +0.049 | **+0.071** | +0.048 |

The reward-free-adaptation benefit **grows with commitment horizon** (feedback tolerates a stale model when you re-ground every step; committing does not), and the refit **tracks the oracle throughout** (~5000 reward-free transitions recover ~all of a fully-adapted model's value at every commitment length). **Bonus — the turnover at 12**: even the *oracle* degrades (0.033) because no one-step FM composes over a 12-step open-loop rollout — the **~6–8-step composition horizon** from [REACHING_LOOKAHEAD](../../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md) reappearing on physics. The value of an accurate model peaks where the model is maximally load-bearing *and still veridical enough to compose*.

## Caveats

- **Protocol-fair (confound fixed).** MF is a committing motor-program policy with the *same* open-loop commit as MB, so both degrade **identically** at zero-shot (0.087) — the earlier commit-vs-reactive confound (a reactive MF that re-grounded every step) is removed. The agents differ *only* in how they re-adapt. (`figures/dynshift_full_v2/` holds the earlier reactive-MF run for the record; `full_v3` is the fair one.)
- **Reward-driven MF is noisy.** Minimal, un-tuned REINFORCE on the committing (sequence-output) policy has high variance — the reward-driven curve reaches the ceiling in the ~10⁴-transition range but non-monotonically. Only the order-of-magnitude gap is claimed, not a precise multiplier. Per "no RL-maxxing", it wasn't tuned further.
- **Adaptation, not zero-shot robustness.** The result is strictly about *adaptation efficiency* — both agents degrade equally zero-shot; we do not claim MB is more robust zero-shot.

## Biological reading

`replan_every` ≈ **how ballistic the movement is**: fast movements outrun sensory feedback (~50–150 ms) and must run open-loop off an internal forward model (the cerebellum); slow movements are feedback-guided and need only a weak model. Making the FM load-bearing = the ballistic regime, where a stale model overshoots (**cerebellar dysmetria**). Reward-free FM refit = **cerebellar recalibration from sensory prediction error** (available from any movement, no reward); reward-driven MF = policy relearning by reinforcement (basal ganglia). The dissociation mirrors the cerebellum (fast, self-supervised, prediction-error-driven) vs basal-ganglia (slow, reward-driven) division of labor. This reading is the seed of the whole [ballistic arc](../ballistic/README.md), which turns "how ballistic the movement is" into the causal knob.

## Figures (`figures/dynshift_full_v3/`, `figures/dynshift_replan_abl_v1/`)

`fig1_shift_bites` · `fig2_recovery_curves` (reward-free vs reward-driven, split by signal type) · **`fig3_sample_efficiency`** (both agents on one interaction axis — the money figure) · `fig_replan_ablation` (benefit vs commitment horizon).

## Reproduce

```bash
cd experiments/
# ~2 min on L4
modal run mjc/dynamics_shift/dynamics_shift.py::dynamics_shift --quick        # smoke
modal run mjc/dynamics_shift/dynamics_shift.py::dynamics_shift --tag full_v3  # replan_every=8, fair committing MF
modal run mjc/dynamics_shift/dynamics_shift.py::dynamics_shift --tag full_v1 --replan-every 1  # muted (re-groundable) endpoint
modal run mjc/dynamics_shift/dynamics_shift.py::replan_ablation               # commit-horizon sweep
```

Knobs are auto-exposed as CLI flags. The Modal function commits `results.json` + figures to `/data/dynamics_shift/<tag>/` on the `mujoco-control-data` volume; the local entrypoint mirrors them to `figures/dynshift_<tag>/`.

## Open (hardening)

Multi-seed the (noisy) reward-driven REINFORCE slope; add a **mass-shift** operator (vs the damping collapse) as a second, qualitatively different shift.
