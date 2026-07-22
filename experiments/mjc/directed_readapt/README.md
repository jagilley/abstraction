# Directed reward-free re-adaptation — when does a value-directed drive earn its keep? (two instructive negatives)

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) (§"Meta-RL two-timescale"; §interface point 3 — ensemble-disagreement as the online reducibility proxy) · **Drive selection**: [../../a2a_forward/reaching/CURIOSITY_CONTROL_README.md](../../a2a_forward/reaching/CURIOSITY_CONTROL_README.md) Step 0 (chose the disagreement magnitude)
**Parent**: [Cut #3](../dynamics_shift/README.md) `dynamics_shift.py` — reward-free re-adaptation from *undirected* collection · **Sibling**: [../value_shaping/README.md](../value_shaping/README.md) (the learning-layer control)
**Code**: `directed_readapt.py` (lives in this folder); file index: [FILES.md](FILES.md) · **Status**: two negatives, both well-understood — **global shift → null** (single seed), **localized shift → drive fails** (3 seeds). **Date**: 2026-07-18.
**Builds on this**: [`meta_adapt/`](../meta_adapt/README.md) (this cut's two negatives are what motivate its phenomenon-first pivot) · [`curiosity_control/`](../curiosity_control/README.md) (extends the confident-prior blindness found here to a *drifting* frontier)

---

## Where this sits

The meta ∩ value-supervision question, rung 2: *does a value-directed exploration drive accelerate inner-loop re-adaptation under non-stationarity?* Cut #3 established a model-based agent re-adapts to a shifted operator from **reward-free** interaction (fine-tune the arity-2 FM on d1 `(s,u,s')` → recover toward the oracle) — but it collected those transitions by **undirected** OU-random exploration. Here we ask whether **disagreement-directed** collection (go where a fresh-FM ensemble disagrees = where the new dynamics are unlearned; Pathak et al. 2019) re-adapts in **fewer** transitions.

Answer, on two shift geometries: **no** — and the *reasons* are the payoff. They compose into a clean rule for when the drive can help.

## Experiment 1 — global shift (Cut #3's drag collapse): NULL

`shift_mode="global"` — drag collapses uniformly everywhere (damping 2.0→0.05). Directed ≈ undirected: **both recover to the oracle ceiling by ~25 transitions** (transitions-to-recover = 25 for both; the whole recovery curve overlaps). The smoke's apparent 25-vs-50 edge was tiny-N noise.

**Why**: a global, low-dimensional shift makes **every transition equally informative** — undirected random collection samples the new dynamics everywhere at once and re-fits globally almost instantly. There is **no scarcity** for a drive to exploit, and re-adaptation is so fast there is no room to beat it. This is the curiosity-line Phase-2b lesson (*the drive needs scarcity*) reproduced on control.

## Experiment 2 — localized shift (a force-jet patch): DRIVE FAILS

To create scarcity, `shift_mode="patch"` adds a **spatially-localized** change: a strong constant force **jet** inside a Gaussian patch (`../pusher_env.py`'s `patch` DGP key, via `qfrc_applied`; additive, off by default → Cuts #1–3 reproducible). A weak drag tweak was too small (a stale FM predicted it fine — stale in-patch R² 0.94); the jet makes the local change genuinely load-bearing: **stale in-patch R² ≈ 0.63 → oracle ≈ 1.0**, and planning degrades (stale ~0.05–0.07 vs oracle ~0.03–0.04). Only in-patch transitions are informative → scarce. Sharp readout: **per-region FM R² (in-patch vs out) on a balanced eval pool** (oracle given guaranteed patch coverage) + **patch visitation** per arm.

**Result (3 seeds).** No re-adaptation benefit — and the visitation metric shows why:

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| patch visitation — **undirected** | 0.100 | 0.139 | 0.112 | **0.117** |
| patch visitation — **directed** | 0.087 | 0.121 | 0.092 | **0.100** |
| in-patch recover (dir vs und) | dir faster | **und** faster | tie | wash |

The directed drive **consistently *under*-visits the patch** (lower in all 3 seeds) and in-patch recovery is a **wash** (seed 0 favors directed, seed 1 undirected, seed 2 ties).

**The mechanism (the valuable part): disagreement is blind to a *confident-prior* shift.** The ensemble members all warm-start from the d0 prior; in the unvisited patch they **agree** on the (wrong) "no-jet" d0 prediction → disagreement is *low* exactly where the FM is most wrong. So the drive never targets the patch — it chases **spurious disagreement** elsewhere (velocity extremes / bootstrap variance on already-learnable free-flight) and ends up sampling the patch *less* than uniform exploration. In one line: **disagreement flags where the model is *uncertain*, not where it is *confidently wrong* — and a dynamics shift creates confident-wrongness.**

## The composed lesson

The two negatives give the conditions a value-directed drive needs to earn its keep:
1. **Scarcity** — a localized informative region (Exp 1 lacked this → null).
2. **A signal that actually flags that region** — **disagreement fails this for a shift** (Exp 2).

This also **flips the curiosity-line drive ranking, by regime**: with *irreducible noise present*, disagreement beats raw surprise/error (it rejects the noisy-TV; Phase 2b / Step 0). With *fully-reducible dynamics + a confident-prior shift* (here), that reverses — actual prediction **error/surprise** should beat disagreement, because error *does* flag the confident-wrong patch and there is no noise to be fooled by. Which drive wins is set by the environment, not absolute.

## Where this leads (the pivot)

These negatives close a chapter of *a-priori mechanistic-atom verification* (disc-4 value-shaping; the specific drive) with diminishing returns — isolating a mechanism kept either killing the phenomenon or leaving the baseline already good enough. The decision (2026-07-18) is to go **phenomenon-first**: stop verifying isolated atoms, and instead **get the actual meta phenomenon working — compounding re-adaptation over a *sequence* of shifts** (the thing stationary meta-RL provably can't manufacture, [RHM_META_LEARNING](../../rhm/ratchet/RHM_META_LEARNING_README.md)) — then back-translate the mechanism from what works. Two guardrails keep it from becoming RL-to-SOTA: (a) the deliverable stays a **dissociation** (the compounding curve vs a no-outer-loop *ablation*), not a leaderboard number; (b) a **factored, dissectible** architecture (adaptable FM + explicit outer memory/modulator) so "what worked" is legible by construction. The scarcity + flag-confident-wrong-regions insight becomes a *candidate mechanism to watch for* in whatever compounds — not a thing to verify first. An **error/surprise-directed** drive (which should flag the patch where disagreement couldn't) is a low-cost check we can fold in opportunistically rather than as a standalone arm.

## Reproduce

```bash
cd experiments/
# global shift (the null) — single seed
modal run mjc/directed_readapt/directed_readapt.py::directed_readapt --tag full_v1
# localized force-jet patch (the drive negative) — per seed
modal run mjc/directed_readapt/directed_readapt.py::directed_readapt --tag patch_v1_s0 --shift-mode patch --seed 0
# quick smokes
modal run mjc/directed_readapt/directed_readapt.py::directed_readapt --quick                    # global
modal run mjc/directed_readapt/directed_readapt.py::directed_readapt --quick --shift-mode patch  # patch
```

Knobs are auto-exposed CLI flags (`--patch-force-y`, `--patch-center-x`, `--patch-sigma`, `--ens-k`, `--ens-pert`, `--dir-warmup`, …). Results + figures commit to `mujoco-control-data` under `/data/directed_readapt/<tag>/` and mirror to `figures/directed_readapt_<tag>/`.

## Figures (`figures/directed_readapt_<tag>/`)
`fig1_recovery_curve` (planning dist vs #reward-free transitions, directed vs undirected) · `fig2_transitions_to_recover` (headline bar) · `fig3_r2_recovery` (global FM fidelity) · **`fig4_inpatch_recovery`** (patch mode: in-patch R² recovery + the patch-visitation bar — the diagnostic that exposes the disagreement-blindness).

## Caveats
- Exp 1 single seed (fine — the null is structural); Exp 2 three seeds. All single rule/dynamics-seed per convention.
- `directed_readapt.py` is retained as the **diagnostic record** for the two negatives + the reusable localized-shift substrate (`patch`), not a positive result.
- The disagreement arm's failure is on a *confident-prior shift*; it does **not** refute disagreement as a drive in its native regime (novel-region exploration with irreducible noise present, where Step 0 selected it).
