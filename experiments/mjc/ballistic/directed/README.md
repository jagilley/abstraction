# Directed collection — does the value drive choose *where to look*, and does it pay?

**Up**: [../README.md](../README.md) (ballistic control) · **Idea doc**: [../../../../ideas/two_timescale_value_loop.md](../../../../ideas/two_timescale_value_loop.md)
**Direct parent**: [../README.md](../README.md) — Cut 4 closed the **efferent** half of the cerebellum↔value bridge (a better forward model reaches behavior, ~3× more under feedforward commitment) and named the missing piece in its own caveats: *"the value loop **directing where to collect** is the piece 4a showed is confounded in this geometry — so 'the value loop chooses collection' is **not** cleanly demonstrated here."* This cut builds that afferent link.
**Code** (lives in this folder, per [STRUCTURE.md](../../../../STRUCTURE.md)): `directed_separability.py` (S0), `directed_ladder.py` (S1), `directed_loop.py` (S2); analysis `directed_loop_figure.py`, `directed_loop_from_logs.py`. Env support: `rot_regions` in [../../pusher_env.py](../../pusher_env.py). File index: [FILES.md](FILES.md).
**Status**: S0 + S1 clean positives (3 seeds each). S2 a **bounded positive with a real null inside it** (4 configurations, 3 seeds). Single-family (corridor reach, command-rotation drift). **Date**: 2026-07-22.

---

## One-liner

Directed collection buys **speed of re-adaptation, not a better final model** — and it is the **reducibility** half of the value signal that does the work. Concentrating a scarce reward-free collection budget beats spreading it by ~2.9× under feedforward control and ~2.3× less under feedback control (the Cut-4 ballistic specificity, reproduced). Chasing raw prediction error is the *worst* way to concentrate (the noisy-TV failure, 1.5–1.9× costlier), so **knowing what is learnable is load-bearing**. But weighting by *where you will actually be* — the relevance term — wins decisively when collection is **one-shot** and contributes nothing measurable once collection is **repeated**, because budget spent somewhere irrelevant is a one-time cost you amortize rather than a permanent loss. Two mechanism findings fall out: **ensemble disagreement cannot detect a drift at all** (every member trained pre-drift agrees, and they are all wrong together), and a **multiplicative value signal needs an explicit no-op** or it normalizes floating-point noise into confident nonsense.

## The design problem this cut had to solve first

Cut 4a's negative was that a scalar explore/exploit drive `b` was **confounded by the corridor geometry** — "exploit" already covered the control-relevant needle and "explore" chased off-corridor aleatoric noise, so the intended roles inverted. Cut 4b fixed the *quality* axis by making it **smooth and global** (damping staleness). But a global axis is, by construction, one where *where you collect cannot matter*: every transition informs it equally. So the Cut-4 substrate cannot host a directed-collection test at all.

Making *where* matter requires a **local** dynamics change (only in-region transitions inform it). And 4b established that the obvious local change — an additive force jet — is **open-loop incompensable**: even a perfect FM cannot counteract a strong local kick feedforward, so it saturates ballistic control at "fail" for every FM quality. Local-and-incompensable is a dead end.

> **The unlock: perturb the COMMAND channel, not the force channel.** A spatially-localized *rotation of the command→motion map* (`PusherEnv.rot_regions`, added here) is **local** — φ_j is only observable from transitions taken inside region j — and **open-loop compensable**, because the planner simply pre-rotates its commands. Biologically it is the canonical cerebellar adaptation paradigm (state-dependent visuomotor rotation), and a wrong φ_j is textbook dysmetria.

This is a reusable design rule, not a substrate detail: *if you need a dynamics change that is both locally-learnable and feedforward-correctable, perturb the input map rather than adding a force.*

## Substrate

Puck-free corridor reach (`d=2.0`, the regime where ballistic control is competent with a matched FM — cf. Cut 4c), K=4 Gaussian-gated regions with near-disjoint gates (σ=0.18, cross-gate < 0.01). Each region carries its own command rotation and/or aleatoric noise. Two controllers throughout: `reactive` (re-plan every step) and `ballistic_cem` (plan once, execute open-loop) — the Cut-4b pair.

**The 2×2 partition** (S1/S2) crosses the two properties the value signal must jointly respect:

| | REDUCIBLE (a stale rotation) | IRREDUCIBLE (aleatoric noise) |
|---|---|---|
| **ON** the reach | **A** ← the right answer | **C** ← raw-surprise decoy |
| **OFF** the reach | **B** ← frontier/novelty decoy | **D** ← neither |

No single naive heuristic wins: error-chasing fixates C/D, reducibility-without-relevance goes to B, relevance-without-reducibility splits A/C. Only the conjunction lands on A alone.

---

## S0 — is directed collection even possible here? (`directed_separability.py`)

Three preconditions, each able to kill the design. **All pass, 3 seeds:**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **compensability** cost, ballistic (want ≈0) | −0.028 | −0.032 | −0.026 |
| **staleness damage** ballistic / reactive | 0.573 / 0.090 | 0.604 / 0.102 | 0.587 / 0.086 |
| **ablation spread** (which region is stale) ballistic / reactive | 0.520 / 0.045 | 0.537 / 0.047 | 0.567 / 0.046 |
| **locality** diag/off-diag of the transfer matrix | 31.3× | 26.7× | 14.8× |
| **directed vs uniform** at equal budget, ballistic / reactive | 0.455 / 0.032 | 0.449 / 0.042 | 0.488 / 0.030 |

1. **Compensable.** A matched FM in the rotated world reaches as well as in the clean world (cost ≈ 0, slightly *negative*). Ballistic with a matched FM lands at 0.085, in line with Cut 4b/4c's ~0.108 ceiling — same regime.
2. **Which region is stale matters ~11× more for ballistic than reactive.** There is something for an allocator to choose.
3. **Collection is genuinely local** — budget spent in region j fixes region j and not the others.
4. **Uniform collection at the same budget does nothing** for ballistic (0.670 vs 0.657 stale): the scarcity regime bites as intended.

**Two designs that had to fail first**, both recorded because they are the kind of thing that silently manufactures a result:
- **Identical φ across regions makes them informationally redundant.** With all four at φ=+1.0, "rotate everywhere by +1.0" fits every region at once, so off-path data taught the on-path rotation and the transfer matrix came out flat at **1.1×**. Distinct, pairwise-far angles → 31×.
- **Fine-tuning on the new in-region data alone smears the rotation globally** (global error rose 0.035 → 0.086 while every region's error fell — fake transfer). Fixed by replaying the out-of-region base data, which is still valid post-drift because the gates are ~0 there.

**Caveat.** Which of the *two* on-path regions dominates the ablation flips across seeds (s0/s2: the +0.30 region at +0.51/+0.54; s1: the −0.30 region at +0.53) — a winner-take-all threshold effect on the median goal-distance. On-path ≫ off-path holds in all three.

## S1 — one-shot allocation: does a reward-free signal pick the right place? (`directed_ladder.py`)

A single scarce budget (400 transitions), allocated by hand-specified rules, then graded. **The value signal is `learning_progress × visitation`**, both reward-free:

- **learning progress** — from a cheap monitoring survey (160 real transitions/region, far too few to repair anything): fit half, measure how much *held-out* error the fit actually removes. Large where error is reducible, ~0 on aleatoric noise. Two-tier: you passively experience a little everywhere, and you *choose* where to practise.
- **visitation** — region occupancy of the ballistic plan rolled through **the FM itself**. Purely internal: the forward model tells you where you are going to be, hence where it is worth looking.

**Ballistic gap-closed toward oracle** (1.0 = matches oracle, 0.0 = no better than uniform):

| arm | s0 | s1 | s2 | where it spent |
|---|---|---|---|---|
| **lprog × visits [VALUE]** | **1.03** | **1.18** | **0.92** | 400/0/0/0 — *the oracle allocation* |
| visitation-only | 0.78 | 0.35 | 0.61 | splits A/C — half into noise |
| lprog-only | 0.70 | 0.60 | 0.61 | splits A/B — half off-reach |
| error-only | 0.14 | 0.15 | 0.06 | spread over all four — drawn to C/D |
| disagreement-only | 0.11 | 0.02 | 0.10 | ~uniform (see below) |

The value arm **recovers the oracle allocation exactly** from signals that never touch reward, and every ablation fails in its *predicted* direction. Allocation matters **10.8× / 4.6× / 13.7×** more for ballistic than reactive.

> **Ensemble disagreement cannot detect a drift.** Measured 0.0016 / 0.0014 / 0.0017 / 0.0025 across stale-relevant, stale-irrelevant, and two pure-noise regions — flat, no discrimination. The reason is structural: every committee member was trained on the **pre-drift** pool, where the drifted regions were unambiguous, so they **agree with each other and are wrong together**. Diversity in fitting the *old* data produces no diversity about the *new* mapping. Disagreement finds where you **lack data**, not where your data went **stale**.

## S2 — the online loop against a moving target (`directed_loop.py`)

Each round: survey → allocate → collect → fine-tune → grade, with one region re-drifted every `drift_every` rounds. Four configurations, 3 seeds each. **The primary metric is `excess damage`** — integrated ballistic goal-distance above a shared floor over the rounds following an on-reach drift, i.e. how much behavioral cost was paid while the model was stale.

| policy | loop (rich) | loopB (scarce) | loopC (drift-dense) | loopD (+ gated arms) |
|---|---|---|---|---|
| uniform | 2.082 ± 0.187 | 2.275 ± 0.315 | 2.629 ± 0.277 | 2.613 ± 0.280 |
| error-only | 0.643 ± 0.097 | 1.075 ± 0.135 | 1.735 ± 0.186 | — |
| oracle | 0.493 ± 0.079 | 0.716 ± 0.059 | 1.311 ± 0.122 | 1.295 ± 0.127 |
| visits-only | 0.450 ± 0.040 | 0.708 ± 0.100 | 1.248 ± 0.068 | — |
| value | 0.583 ± 0.104 | 0.578 ± 0.092 | 1.152 ± 0.121 | 1.136 ± 0.127 |
| value-floor (gated) | — | 0.593 ± 0.060 | — | **1.009 ± 0.068** |
| value-maint | — | — | — | 1.232 ± 0.067 |
| **lprog-only** | **0.373 ± 0.047** | **0.490 ± 0.063** | **0.924 ± 0.060** | **0.908 ± 0.057** |

**What is robust across all four configurations:**
1. **Concentration ≫ spreading** — uniform is 2.3–3.9× worse than any directed policy, always, far outside error bars.
2. **Reducibility-awareness is load-bearing** — `error-only` is consistently the worst *directed* policy (1.5–1.9×). The noisy-TV trap is real and expensive.
3. **The payoff stays ballistic-specific** — 2.2–2.8×, matching Cut 4b's transmission slopes and 4c's recovery gains.

**What is a null:** the conjunction never beats reducibility-chasing alone in a loop. `value-floor` (1.009 ± 0.068) vs `lprog-only` (0.908 ± 0.057) is ~1.3 SEM, in the same direction across every configuration tried. The mechanism: with repeated attempts, budget spent off-reach **permanently repairs a rarely-re-drifted region** — a one-time cost, after which the relevance-blind policy behaves identically to a relevance-weighted one. The visitation term only pays when allocation is genuinely zero-sum, which is exactly S1's regime.

### Two things the loop taught that the ladder could not

**The asymptotic metric is the wrong metric, and it nulls.** The first configuration reported *mean* ballistic distance over rounds and showed **no separation at all** — `error-only`, which spends ~60% of every round on noise regions, closed the whole uniform→oracle gap. A loop that is budget-rich in aggregate (400/round × 24 rounds, persistent buffers) equalizes asymptotic competence across every policy that concentrates anywhere; below ~0.05 region-error, ballistic control stops improving and the metric floors out. **Ask how fast it recovers after a change, not where it ends up.**

**A multiplicative value signal needs an explicit no-op.** `lprog × visits` has no lower bound. Once the on-reach region is repaired, `lprog_A → 0` and the whole product collapses to ~1e-8, at which point normalizing turns floating-point noise into a confident all-in allocation on the region with *zero* visitation. In loopC, `value` sent its **entire** budget to the off-reach region in **19 / 22 / 12 of 72 rounds**, and the count is monotone with its damage (12→0.850, 19→0.929, 22→1.678). So `value` was never implementing the conjunction — for ~25% of rounds it *became* `lprog-only`, arriving there through a dead signal with worse timing. Gating it (`value-floor`, fall back to uniform when max product < 1e-3) removes the seed-1 catastrophe and stabilizes it (0.972 / 1.163 / 0.893 vs 0.860 / **1.697** / 0.850) — **which is why the remaining null is a real result rather than a broken implementation.**

Two hypotheses this ruled *out*, both checkable from stored data: the visitation estimate was **not** degenerate (visits[A] = 0.496 / 0.392 / 0.478, std ≈ 0.03–0.04), and allocation **peakedness** does not explain it either (seed 1 H=0.17 dmg=1.68 vs seed 2 H=0.18 dmg=0.85).

**The `oracle` is not a valid ceiling in a loop.** It loses to `value` in 2 of 3 seeds because it is specified as *repair-when-broken* — it stops collecting once a region's error drops below the floor, and with a 400-capacity buffer that region ages out. Continuous top-up beats episodic repair. But the corresponding no-op choice does **not** follow: `value-maint` (fall back to collecting ∝ visitation) is clearly *worse* than `value-floor` (1.232 vs 1.009), because "keep rehearsing where you act" sends budget into the on-reach *noise* region.

---

## What this says about the idea doc

Three updates, stated at the doc's own vocabulary:

1. **Falsify functional-shape bullet #2 / answer open-Q #2.** The doc proposes ensemble disagreement as the online reducibility instrument (next-piece-to-test #3), citing curiosity Phase 2b's 3.7×→8.5× needle over-sampling. That result and ours are both right, and they are **two different problems the doc currently calls one**: disagreement works when the frontier moves into *never-sampled* territory (missing data) and is **blind** when the world *rewrites already-sampled* territory (stale data). Since the doc's own **Type-2** non-stationarity is precisely the stale case, *the proposed instrument fails on the kind of non-stationarity the doc calls load-bearing.* What worked instead is neither the disagreement magnitude nor the `−d‖e‖/dt` derivative but a **counterfactual fit probe** — fit on a small sample, check whether held-out error drops — which also sidesteps the "re-opening blindness" Phase 2 exposed, because it tests whether error *can* fall rather than waiting for it to fall.
2. **Downgrade the two-taps claim.** "Explore/exploit falls out of which part of the FM's output the value system taps" (`e`-tap vs `p`-tap) is supported **one-shot** and unsupported **in a loop**. The interface *shape* survives; the `p`-tap's contribution is regime-conditional, not structural.
3. **"Compounding" should be "rate."** The meta layer here buys *re-adaptation speed*, not a higher ceiling — which is what Type-2 implies anyway, since a genuinely drifting world has no asymptote to compound toward.

**Not updated (re-confirmed):** the noisy-TV control keeps earning its "not optional" status; non-stationarity remains load-bearing, with the sharper condition that drift must be **dense relative to budget**; scarcity replicates from curiosity Phase 2b into a control substrate; the multiplicative/gating form holds, with the no-op addendum above.

## Honest caveats

- **Single-family**: one arena, one kind of drift (command rotation), 3 seeds. The disagreement finding is the most transferable because it rests on a *structural* argument; the relevance-null is the most contingent.
- **The disagreement result's main threat**: our committee is bootstrap-resampled members of one architecture on one pre-drift pool. A genuinely diverse committee might behave differently. The structural argument (no member saw post-drift data, so none can represent the new mapping) is an argument, not a measurement — attack this before promoting it to a belief.
- **`rounds-to-recover` is retired**, not reported. It was censored in 5–8 of 8 epochs for every policy at these settings, and an earlier version normalized to each policy's *own* epoch minimum, which rewards a uniformly-mediocre policy for "recovering" instantly and produced an incoherent ranking. `excess damage` is the workhorse.
- **loopD's numbers come from parsed logs, not `results.json`** — all three clients were killed at ~91%, and the volume commits only at the end. `directed_loop_from_logs.py` reproduces them, and `--verify-splice` confirms the spliced `lprog-only` arm matches loopC to 5e-5 (print rounding). Re-run loopD if a clean artifact is wanted.
- **`gap-closed toward oracle`** is a valid normalizer in S1 and *not* in S2 (see the oracle note above); S2 tables report raw excess damage.

## Reproduce

```bash
cd experiments/
# S0 — separability preconditions (3 seeds)
for s in 0 1 2; do
  modal run --detach mjc/ballistic/directed/directed_separability.py::directed_separability \
      --tag sep_s$s --seed $s --corridor-r 0.4 --plan-h 34
done

# S1 — the one-shot allocation ladder (3 seeds)
for s in 0 1 2; do
  modal run --detach mjc/ballistic/directed/directed_ladder.py::directed_ladder \
      --tag lad_s$s --seed $s
done

# S2 — the online loop. loopC = drift-dense; loopD adds the gated value arms.
for s in 0 1 2; do
  modal run --detach mjc/ballistic/directed/directed_loop.py::directed_loop \
      --tag loopD_s$s --seed $s --policies "uniform,oracle,value,value-floor,value-maint,lprog-only"
done
python3 mjc/ballistic/directed/directed_loop_figure.py --tags loopC_s0 loopC_s1 loopC_s2 --out loopC
python3 mjc/ballistic/directed/directed_loop_from_logs.py \
    --tags loopD_s0 loopD_s1 loopD_s2 --splice-from loopC --splice-policy lprog-only --verify-splice
```

**Gotchas.** (i) Run `modal run` from `experiments/`, not from `mjc/` — the module path doubles and the file is not found. (ii) Launching 3 detached clients from one shell is fragile: clients get evicted and a killed client loses the local mirror *and* (if killed before the function's final `volume.commit()`) the results.json entirely — the per-round log lines are the fallback. (iii) The earlier configurations (`loop_s*` rich, `loopB_s*` scarce) are retained deliberately: the rich/scarce contrast is the evidence for the asymptotic-metric null.

## Figures (`figures/`)

- **S0**: `directed_separability_sep_s*/` — `fig1_compensability`, `fig2_ablation`, **`fig3_transfer`** (the locality matrix), `fig4_allocation_preview`.
- **S1**: `directed_ladder_lad_s*/` — **`fig1_ladder`** (the arm ladder), `fig2_signals_alloc` (per-region signals + where each arm spent), `fig3_gap_closed`.
- **S2**: `directed_loop_{loop,loopB,loopC}_s*/` per-run; aggregates in `directed_loop_agg_{loopA,loopB,loopC}/` — **`fig1_speed_vs_asymptote`** (the metric that discriminates beside the metric that nulls), `fig2_tracking_vs_speed`, `fig3_rounds`.
