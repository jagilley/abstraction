# jacobian_teacher — File & figure index

Full file-by-file reference for the error-based action-gradient node (Phases A/B/C). Summarized
in [README.md](README.md); this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md)); the only shared dependencies are
[`../arm_env.py`](../arm_env.py) and [`../shared.py`](../shared.py), which stay at the `mjc` node
because every experiment there imports them. The substrate is forked from
[`../ballistic/arm/arm_readapt.py`](../ballistic/arm/arm_readapt.py) and the fork is gated by
`phase_a.py`, not by convention.

## Code files

| File | Purpose |
|---|---|
| `SPEC.md` | The pre-registration (2026-09-03, Jasper's prompt): the question, the three teachers, the FM ladder read two ways, the dysmetria control, and the Phase C committee. Kept unchanged so the built node can be read against what was specified. |
| `defaults.py` | Configuration only, and it imports **nothing** — a Modal *local entrypoint* runs on the submitting machine, where `../shared.py`'s standing rule says torch/mujoco may be absent. `fork_defaults()` holds Cut 4c-arm's 38 entrypoint defaults in `cfg` spelling; `teacher_defaults()` holds this node's own 22 knobs. `core.py` re-exports both so a Modal function body can keep importing them from `core`. Splitting this out is not cosmetic: `core.py` imports torch at module scope and the first submission failed on exactly that. |
| `core.py` | The substrate fork plus everything new, in three layers. **(1)** `ArmSubstrate` — Cut 4c-arm's closures hoisted to methods with identical arithmetic and seed offsets (`mlp`, `train_steps`, `mpc_plan`, `eval_geometry`, `rollout`, `clone_policy`, the task probe). Additions are marked: `fk_jac` (analytic ∂fk/∂q, for gate 2a), `readout` (adds the **signed radial** term the fork lacks — only an along-reach sign separates hypermetria from hypometria), `execute` (the open-loop half of `rollout`), `mpc_plan(terminal_only=True)`, and a **direction-banded** `eval_geometry`/`clone_policy` that over-samples candidates vectorised (a direction band admits only ~2.5% of joint-space proposals, so the fork's 40 sequential tries would silently return a mostly off-band probe set). **(2)** The Jacobian layer: `fm_endpoint` composes the one-step FM H times into a differentiable ŷ(u); `endpoint_vjp` takes `eᵀ ∂ŷ/∂u` in one backward pass and applies the dysmetria sign mask; `endpoint_jac` / `onestep_jac` materialise the full operators; `plant_endpoint_jac_fd` / `plant_onestep_jac_fd` are the finite-difference plant oracles; `direction_metrics` and `vjp_direction_metrics` score direction, not magnitude. **(3)** `action_gradient` — **every** teacher, in one function returning one object of one shape, so `mixed@β` is literally eq. 2 and no arm has a private update path. Plus `train_teacher` (the shared loop; identical pool, order and noise draws across arms), `summarize_curve`, `fm_readings`, and the Phase C committee (`RPFMember`, `CommitteeMean`, `build_committee`, `train_committee`, `committee_direction_agreement`). |
| `phase_a.py` | `run_phase_a(cfg)`: the four gates. Fork fidelity re-derives `arm_readapt.py`'s defaults from its **source** with `ast` (no import — that would register a second Modal function) and compares key by key, then hashes a fixed-command plant trajectory. The oracle gate checks the kinematic half against `mj_jacSite` and sweeps the finite-difference step size, because a central difference has no error bar of its own and the plateau is the evidence. **Result**: 38/38 defaults identical, hash `d2ea2114e5076d52`, ∂fk/∂q vs `mj_jacSite` 4.4e-16, FD plateau ≥0.99994 over eps 1e-4…1e-2; matched-ceiling endpoint Jacobian cos 0.9863 vs stale 0.8249; ‖g_ebl‖/‖g_rbl‖ = 1/76; `ebl_imagined` 0.0562 vs `ballistic_bc` 0.0578 = 0.97×. Flags: `--reach-budget`, `--sigma`, `--lr-rbl`, `--lr-ebl`, `--fd-eps`, `--jac-n`. |
| `jacobian_teacher.py` | `run_jacobian_teacher(cfg)`: Phase B. Builds the FM-quality ladder (either Cut 4c-arm's reward-free re-adaptation milestones or, with `--capacity-ladder`, matched-data FMs at increasing `fm_hidden`), clones **one** motor program from a full-capacity pre-drift FM, and trains every teacher through every rung from an identical copy on identical reaches. `rbl` is run once and reused across rungs with an equality **audit** at a second rung (it cannot depend on the FM; the audit is what makes that checkable — it returns bit-identical numbers). **Result**: ~1.5× data efficiency over reward, flat across the transition ladder; the capacity ladder's one-step-vs-composed dissociation (0.989→0.9996 against −0.15→0.98); the β reversal; the dysmetria table. Flags: `--milestones`, `--capacity-ladder`, `--teachers`, `--beta-sweep`, `--beta-rungs`, `--mix-scale-modes`, `--dysmetria-rungs`, `--rbl-rungs`, `--eval-schedule`, `--train-dir`, `--gen-offsets`. |
| `committee.py` | `run_committee(cfg)`: Phase C. The same ladder with a K-member random-prior committee, and the three pooling teachers. Reports, per rung, the oracle-free `dir_agree` beside the oracle's `vjp_cos`, plus the sign-correct rate **conditional on unanimity** — the one thing an oracle can say about a reading whose appeal is that it needs no oracle. **Result**: unanimous entries carry the plant's sign 0.905–0.994 vs 0.659–0.864 where members split; Spearman(dir_agree, vjp_cos) = +0.80 (weighted +0.90); `mixed_agree` best at every rung. Flags: `--k-ens`, `--rpf-beta`, `--milestones`, `--rbl-rungs`. |
| `jacobian_teacher_figure.py` | Local, read-only aggregation for Phase B. Carries four readouts the raw run does not, each forced by something in the data: `plateau_mean` (the arms oscillate under a fixed Adam step, so the final readout samples a phase — one rung read 0.0209…0.1126 over its last six), `divergence` (that instability, reported instead of averaged over), `reaches_to` (executed reaches to a **fixed** threshold — `reaches_to_90` rewards an arm that plateaus high), and `grad_snr_early` (the half-batch cosine is meaningless once the mean gradient is near zero). `spearman` refuses a constant series; without that guard `rbl`, which cannot vary with the FM, read ρ = 0.976. `--tags`, `--out`, `--no-plot`. |
| `committee_figure.py` | The same for Phase C, asking the agreement question three ways: the conditional accuracy (needs no ordering assumption), the rank ordering against the oracle, and the pooling comparison. `--tags`. |

## Figures (`figures/<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| **`agg_transition/`** | **The headline transition-ladder figure** (3 seeds): adaptation curves at the stale and matched rungs, and the generalization curve against direction offset. |
| **`agg_capacity/`** | The same for the capacity ladder — the run where the composed-Jacobian reading separates the arms. |
| `jt_jt_s{0,1,2}/` | Per-run `results.json` for the transition ladder: per-rung FM readings (`fm_err`, one-step and composed Jacobian metrics), incumbents, and every arm's full curve, summary, generalization sweep and aftereffect. |
| `jt_jt_cap_s{0,1,2}/` | The same for `fm_hidden ∈ {8,16,32,64,256}`. |
| `phase_a_pa_s0/` | `phase_a.json` — the gate record, including the finite-difference step-size sweep and the measured action-gradient norms. |
| `jc_jc_s0/` | `committee.json` — per-rung agreement, the conditional sign-correct rates, and the pooling arms. |
| `*_smoke*/` | `--quick` runs. Diagnostics that the code executes; they inverted both gate 2c and gate 4 relative to the full runs and are never results. |

## Modal volume layout (`mujoco-control-data`)

```
/data/jacobian_teacher/pa_s0/          phase_a.json      (Phase A gates)
/data/jacobian_teacher/jt_s{0,1,2}/    results.json      (Phase B, re-adaptation ladder)
/data/jacobian_teacher/jt_cap_s{0,1,2}/results.json      (Phase B, capacity ladder)
/data/jacobian_teacher/jc_s0/          committee.json    (Phase C)
```

Every run commits incrementally, after each rung and after the dysmetria block — a Modal input
cancellation can silently swallow a single post-cancellation `volume.commit()`
([`../plasticity_gain/`](../plasticity_gain/README.md)'s transferable gotcha).
