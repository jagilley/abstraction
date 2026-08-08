# Sculpt-slip: the conditioning gap on a control substrate, and whether suppressing the aleatoric part of the residual buys anything

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Substrate**: [`../../RHM_SCULPTING_README.md`](../../RHM_SCULPTING_README.md) Stage 3d
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) §5
**Date**: 2026-08-08 · **Status**: Step 1 positive, Step 2 null, closeout **kills the line**. Single
seed throughout.

## Why this substrate

The parent cut established, on the reading axis, that belief revision is separable from token
surprisal — but everything there was open-loop measurement. A control task can do three things the
reading axis structurally cannot: give a **behavioural** readout, supply an **exact binary aleatoric
label**, and make the noise level a **knob**.

Sculpting's block FM is `FM(z, k) → Δz` on the **move axis**, which is temporal. But the task was
*designed* to make consequences deterministic — [`sculpting_control_task.md`](../../sculpting_control_task.md)
ingredient 2, *"Content-independent, deterministic consequences… so a forward model can forecast
it."* So `(z,k)` fully determine `Δz`, the FM's information set is complete, and its residual can only
mean "I lacked capacity" — **epistemically the same class as the depth FM, on a different axis.** A
temporal-vs-depth A/B here would compare two zero-aleatoric forecasters and isolate the axis, which
the parent's gates say is not the operative variable.

Stage 3d's **slippery actuator** (an edit lands as intended with prob 1−q, else on a uniformly random
feature) opens a gap — but its Design 1 deliberately reuses the deterministic FM and ranks by intended
outcome, so the FM never experiences it. `RHM_SCULPTING_README` names the follow-up twice:

> *"A **Design 2** that retrains the FM on slippery targets — so it predicts the true k-dependent
> expectation `E[Δz|z,k]` rather than the intended outcome"*

**Design 2 is the conditioning gap on this substrate**, and it is exactly the cerebellar arity-2
structure: it has the command, it lacks the world's noise realisation.

## Step 1 — is the slip directionally identifiable in the residual? Yes

Ground-truth families (we own the actuator's RNG), 40k held-out transitions per `q`. Signals are
readouts of `r = Δz_realised − FM(z,k)`, scored raw and **matched on ‖r‖** with atom-aware strata, so
the scalar incumbent is pinned at chance by construction and only direction can win.

| q | FM | ‖r‖ raw | ‖r‖ matched *(guard)* | **direction, matched** |
|---|---|---|---|---|
| 0.1 | Design 1 (intended) | 0.784 | 0.532 | 0.701 |
| 0.1 | Design 2 (realised) | 0.773 | 0.527 | 0.712 |
| 0.25 | Design 1 | 0.764 | 0.505 | 0.701 |
| 0.25 | Design 2 | 0.734 | 0.512 | 0.741 |
| 0.5 | Design 1 | 0.752 | 0.501 | 0.700 |
| 0.5 | Design 2 | **0.675** | 0.510 | **0.750** |

The trend across `q` is the mechanism, not just the headline. **Design 2's norm gets less informative
as slip becomes more common (0.773 → 0.675) while its direction gets more (0.712 → 0.750);** Design 1's
norm stays strong (0.784 → 0.752) and its direction is flat. That is what a mean-predictor should do:
a *non*-slipped transition is also off the prediction — it deviates toward the intended outcome — so
both families sit at similar radius on opposite sides. The slipped/clean norm ratio confirms it,
**1.70 for Design 1 against 1.31 for Design 2**.

`cos_to_intent`, a labelled reference that peeks at the counterfactual deterministic outcome, reads
0.069–0.100, i.e. **0.90–0.93 inverted**. So ~0.93 is the ceiling and the honest directional probe
recovers about three quarters of it.

**Guards.** ‖r‖ matched against itself reads 0.500–0.532. The shuffled-label probe reads **0.520–0.525
raw / 0.514 matched, not 0.500** — a mild probe-overfitting leak (768 parameters, 28k train / 12k test),
so the honest excess is ~0.24 rather than ~0.25. Recorded rather than tuned away.

**One registered prediction was half wrong.** We expected direction to matter mainly for Design 2. It
reads 0.70 for Design 1 too. What the conditioning gap changes is not whether the directional structure
exists but whether the **scalar shortcut to it still works** — the same shape the parent cut found on
the reading axis.

## Step 2 — does suppressing it recover anything? No

Framing that makes the test sharp: under a uniform slip, `E[Δz|z,k] = (1−q)·Δz_intended + q·const` is
**affine** in the intended outcome, so Design 2's population optimum induces the *same ranking over
moves* as Design 1's. **The entire cost of opening the conditioning gap is estimation variance.** That
gives a real ceiling and floor.

Six arm families, one shared data stream, differing only in weighting. `Π` is a low-rank operator from
the accumulated residual covariance in the acted-block frame, no labels, stop-gradded and slowly
refreshed (the two-timescale structure §5 asks for), swept over rank ∈ {8,32} × α ∈ {0.1,0.5}.

**The prize first** — how much perfect noise-blindness is worth at all, at q = 0.25:

| metric | `d2_unweighted` (floor) | `d1_clean` (ceiling) | gap |
|---|---|---|---|
| `value_top1_agree` | 0.3604 | 0.3604 | **+0.0000** |
| `value_rank_corr` | 0.5205 | 0.5553 | +0.0347 |
| beam w64 | 0.2139 | 0.2588 | +0.0449 |

Recovery of that gap (1.0 = matches ceiling, 0.0 = no better than floor):

| arm | rank_corr | beam w16 | beam w64 |
|---|---|---|---|
| `d2_oracle_gate` (gating ceiling, uses labels) | 0.37 | 0.12 | 0.67 |
| `d2_precision_r8_a0.5` (best treatment) | −0.31 | −0.24 | 0.33 |
| **`d2_prec_q0_r8_a0.5` (geometry control)** | **0.01** | **0.24** | **0.61** |
| `d2_absE` | −0.55 | −0.52 | −0.24 |

**Null.** The geometry control — `Π` estimated in a world with *no slip to suppress* — matches or beats
every actual treatment arm. Whatever movement the precision arms show is the loss geometry changing,
not the aleatoric subspace being suppressed. The control existed to catch exactly this.

**One prediction landed:** `d2_absE`, weighting by instantaneous residual magnitude, is the worst arm on
all three readouts. That is [`endogenous_teacher`](../../endogenous_teacher/README.md)'s sign reproduced
as a *prediction* rather than an accident — the theory calls for precision and that cut used its
opposite.

### The more useful finding: the prize was ~0.03, and that was predictable

`top1` gap is **exactly zero**. Given the affine argument, the only cost of the gap is estimation
variance — and every arm got 12k steps × batch 256 = **3.07M samples**, which averages it away. The
experiment had almost nothing to win by construction, and no estimator could have shown anything.

This is a design error worth recording: the repo's directed-sculpting arc is **metered** (`Meter`,
monitor:collect 1.78–1.84×, budgets of 100–400 transitions) precisely because allocation only pays when
data is the scarce resource. This was an unmetered version of a metered question. The closeout below
sweeps the budget to find out whether the prize exists anywhere.

## Closeout — does the prize grow as data shrinks? No. Stop.

Pre-registered two-sided: prize grows as data shrinks → the branch is alive in the scarce regime;
prize flat at ~0.03 → suppressing the aleatoric component buys nothing at any budget, and this line
stops. Three bounding arms across a 60× data range:

| fm_steps | samples | floor `rank_corr` | ceiling `rank_corr` | **prize** | prize `top1` | prize beam w64 |
|---|---|---|---|---|---|---|
| 12000 | 3,072,000 | 0.5205 | 0.5553 | +0.0347 | +0.0000 | +0.0449 |
| 3000 | 768,000 | 0.4134 | 0.4584 | +0.0450 | +0.0078 | +0.0020 |
| 750 | 192,000 | 0.2809 | 0.3040 | +0.0230 | −0.0410 | −0.0234 |
| 200 | 51,200 | 0.1234 | 0.1627 | +0.0393 | −0.0078 | +0.0352 |

**Flat and non-monotone across 60× of data.** The second branch fires: perfect noise-blindness is worth
~0.03 at every budget, so no estimator of the aleatoric component can be worth building here.

**The prediction that motivated this sweep was wrong, and the reason is the useful part.** The argument
was that the only cost of the conditioning gap is estimation variance, which scales as 1/n, so
shrinking data 60× should inflate the prize. It doesn't — because the FM is **bias-dominated, not
variance-dominated, at every budget tested.** Watch the floor column: `rank_corr` climbs 0.12 → 0.28 →
0.41 → 0.52 and is *still climbing* at the largest budget. The FM is nowhere near the regime where
noise in its targets is what limits it; its own approximation shortfall is. Slip-induced target noise
never becomes the binding constraint, so removing it never buys much.

Generalising past this substrate: **an aleatoric filter can only pay when the learner is
variance-limited.** Most forward models in this repo are bias-limited — still improving with more
data or capacity — which is a cheap precondition to check (two arms: clean-target ceiling vs
realised-target floor) before building any machinery to separate reducible from irreducible error.

## Why we did not build a learning-progress estimator

The natural next move — weight by `−d‖e‖/dt`, the derivative of prediction error, which distinguishes
irreducible noise from unlearned structure without assuming anything about variance — is **already
explored in this repo and negative**. Recorded here so it is not re-proposed:

| variant | where | outcome |
|---|---|---|
| `−d‖e‖/dt` EMA, stationary | [`a2a_forward/reaching/CURIOSITY_DRIVE_README.md`](../../../a2a_forward/reaching/CURIOSITY_DRIVE_README.md) | clean positive |
| same, abrupt drift | same | **worse than random** (struct-err 0.082 vs 0.050) — after a change the error *rises*, so `relu(slow−fast)` reads 0: *"structurally blind to a re-opened frontier"* |
| same, gradual drift | same | fixes that, still doesn't beat uniform (a derivative reads ~0 at steady tracking) |
| same, scarcity | same | beaten by ensemble disagreement at every level |
| same, on control | [`mjc/curiosity_control/`](../../../mjc/curiosity_control/README.md) | *"`lp` (the derivative) tracks weakly (the Phase-2 blindness, reconfirmed on control)"* |
| **rate × measured level** `red_delta = max(Δerr,0)·red` | [`mjc/on_policy/metered_repair/`](../../../mjc/on_policy/metered_repair/README.md) §4c | **clean negative with a mechanism**: per-region means target 0.0051, noise 0.0049/0.0064 — *"Flat, with the largest value on a noise region"*, because *"a region whose error is aleatoric has the largest round-to-round \|Δerr\|"* |

What the repo uses instead is a **counterfactual fit** (`full_loop/channel_env.py::counterfactual_lp`)
— fork the FM, fit on half a batch, read the held-out error drop — repaired by a **measured floor**
(`measure_floors`, repeat-execution; `floor_tap.py::matched_pairs_floor`, k-NN), giving
`red = clip((err − floor)/err, 0, 1)`. That tap holds up (RHM repair −0.0542 ± 0.0091, 3/3 seeds; mjc
separation +0.1076 ± 0.0167).

**If this line is picked back up, the estimator to use is a repeat-execution measured `Π`**, not the
eigendecomposition used here: this substrate can re-run the same `(z,k)` many times, so the aleatoric
covariance is *measurable* rather than assumed to be the high-variance subspace. That has no free
parameters, and keeping the covariance rather than its trace gives the **directional** precision
operator §5 calls the only version worth building.

**Scope caveat.** Reducibility is a weak *allocation* tap — 18% of the oracle prize against relevance's
84%, and **4% vs 87% once monitor calls are priced** ([`full_loop`](../../directed_sculpting/full_loop/README.md),
[`metering_sweep`](../../directed_sculpting/full_loop/metering_sweep/README.md) §14). Those ladders ask where to
*spend a collection budget*; this cut asks how to *weight a gradient on data already drawn*, which is a
different question — but the base rate is a warning.

## Reproduction

```bash
cd experiments
# Step 1 -- trains and caches the shared instruments (~25 min on an L4 the first time)
modal run --detach -m rhm.conditional_revision.sculpt_slip.slip_gate::slip_gate1 \
    --slips "0.1,0.25,0.5" --tag step1

# Step 2 -- 12 arms, reuses the cached instruments (~48 min)
modal run --detach -m rhm.conditional_revision.sculpt_slip.slip_gate::slip_gate2 \
    --slip 0.25 --precision-ranks "8,32" --precision-alphas "0.1,0.5" \
    --beam-widths "1,16,64" --tag step2

# closeout budget sweep
modal run --detach -m rhm.conditional_revision.sculpt_slip.slip_gate::slip_budget \
    --slip 0.25 --tag budget
```

Results land on the `rhm-scaling-data` volume (**`chromatic` workspace**) under `/data/rhm_sculpt_slip/`:
`step1_step1_seed1.json`, `step2_step2_q0.25_seed1.json`, `budget_budget_q0.25_seed1.json`, plus cached
`instruments_*.pt` and `fm_d1_*.pt` / `fm_d2_q*.pt`.

## Gotchas worth not rediscovering

- **A geometry control is mandatory for any reweighting of a loss.** Changing how the loss is shaped
  moves results on its own; `d2_prec_q0` (same machinery, `Π` estimated where there is no aleatoric
  part) is what turned an apparent small win into a null.
- **Check the size of the prize before building the estimator.** `d1_clean − d2_unweighted` costs two
  arms and bounds everything downstream. Here it was 0.03, and `top1` was exactly 0.
- **Check bias vs variance before building an aleatoric filter.** It can only pay when the learner is
  variance-limited; if the FM is still improving with data or capacity, its own approximation error is
  the binding constraint and removing target noise buys nothing. Two arms settle it.
- **The affine trap.** When the noise is additive-and-uniform, the mean-predictor ranks identically to
  the intended-outcome predictor, so a conditioning gap costs *variance only*. Any experiment testing
  an aleatoric filter under such noise must be data-limited or there is nothing to win.
- The shuffled-label probe guard did not sit exactly at 0.5 (0.52); with ~768 probe parameters, report
  it rather than assume it.
