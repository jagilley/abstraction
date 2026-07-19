# Curiosity drive — the value-side atom of the two-timescale value loop

**Idea doc**: [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) (the value function as the scalar forward model the a2a program never built; reward = **learning progress**; the missing **non-stationary, reward-driven outer loop**)
**Parent / sibling**: [README.md](README.md) (the reaching control arc — the *forward-model / inner-loop* thread this complements), [ACTIVE_VISION_README.md](ACTIVE_VISION_README.md) (the active-vision apparatus this reuses)
**Code**: `curiosity_reaching.py` (Phase 1), `curiosity_drift.py` (Phase 2), `curiosity_scarcity.py` (Phase 2b)
**Status**: Done, single seed. Phase 1 (drive atom) clean positive; Phase 2 (drift) instructive negative + diagnosis; Phase 2b (scarcity + ensemble) clean positive with an honest nuance.
**Date**: 2026-07-17

---

## The thesis

Every value signal the a2a/rhm program built is **extrinsic** (goal-distance, reach-r\*) and **stationary** — and each "lifts once and plateaus." The genuinely-unbuilt atom is an **intrinsic learning-progress reward** `r = −d‖e‖/dt` that **drives the policy (where to attend)** rather than being injected as a residual-stream scalar. This line builds that atom on an active-vision substrate and asks three things: is the LP drive *distinguishable* from raw surprise and from predictability-seeking (Phase 1); does it keep working under *non-stationarity* (Phase 2); and what does it take to make it *earn its keep over uniform allocation* (Phase 2b)?

**Why active vision and not RHM.** The idea doc floated either substrate. The RHM **specialization line** ([../../rhm/specialization/README.md](../../rhm/specialization/README.md)) shows the RHM depth frontier is **not moved by allocation** (breadth-restriction inert, level-reweighting harmful; only a *direct deep target* moves it) and the root is never NTP-supervised — so an allocation-driven curiosity signal has almost no lever there, and an `LP≈uniform` result would be a *confounded* null. Active vision instead has an **actionable** value-relevant variable (what you glimpse is directly observable, and the world-model's error at a region genuinely drops when you attend there). The price is that it is **stationary** in its dynamics — which is exactly why Phase 2 has to *engineer* the non-stationarity.

**Common apparatus.** A **content-query world model** (WM) predicts the glimpse pixels at a queried fovea location from a positional query + a spatial reveal marker (the active-vision efference copy). A **teleport policy** allocates each glimpse by an intrinsic reward, never re-glimpsing within an episode. Drives differ *only* in the reward: **lp** (`relu(slow−fast EMA of ‖e‖)`), **surprise** (`+‖e‖`), **min-surprise** (`−‖e‖`), **random** (uniform), and — in Phase 2b — **disagree** (cross-model prediction variance over an ensemble). The WM trains online on the policy's own glimpses (the meta-RL coupling in miniature). Drives act on the *true* current intrinsic reward (no learned value head) — a deliberate scope choice: Phase 1 asks *what each drive prefers*, not *whether a head can predict the drive*.

**Load-bearing arena lessons** (each cost an iteration; recorded so we don't repeat them):
1. **Predict content (pixels), not the belief update.** The belief-update MSE is dominated by the deterministic marker/pos term identical across regions, which buries the reducible/irreducible split → LP fixates noise.
2. **The "learnable" region must be *deterministic*** (a fixed/morphing image), because a *fresh* MNIST digit's exact pixels are ~as irreducible as noise.
3. **No content-carrying belief.** A recurrent belief that stores glimpsed content lets the WM "predict" re-glimpsed noise from memory (noise `‖e‖` fell *below* its iid floor 1/12), manufacturing fake LP on noise. A content-query WM with no belief is leak-proof; noise stays at its true floor. (The recurrent belief loop is deferred to a future phase — value-shaping the inner FM, drifting dynamics with a real controller.)

---

## Phase 1 — the drive atom (`curiosity_reaching.py`)

A **stationary** 3-region canvas the fovea navigates: **struct** (a *fixed* MNIST digit → reducible) ∣ **noise** (fresh U(0,1) → irreducible) ∣ **blank** (zeros → trivial). Per-region LP (the layout is fixed and known). Deliberately *slow* WM (lr 1e-4, 1 update/iter) so the reducible frontier lasts long enough to observe.

**Result — the time-course, not a snapshot.** All three discriminators fire:
- **noise floor sits exactly at the irreducible 1/12 ≈ 0.083** and never drops (leak-free); struct → ~0 (reducible); blank → ~0 fast.
- **surprise pins to noise** (occ 0.79→0.96) for the entire run — the noisy-TV pathology.
- **min-surprise pins to blank** (0.76→0.93) for the entire run — the dark room.
- **LP alone navigates**: it briefly rides the noise *mean* (occ ~0.8, it 0–100 — learning the mean is genuine early progress), **abandons noise** once the mean is learned (`d‖e‖/dt→0` — the noisy-TV *departure* surprise never makes), **rides struct** (0.87→0.96, it ~166–350), then **exhausts** it once mastered (LP≈0 → indifferent — the stationary plateau).

So **LP ≠ surprise** (struct vs noise = noisy-TV) and **LP ≠ min-surprise** (struct vs blank = dark-room broken), and LP is the only drive that *releases* a region once it stops paying learning — the defining LP signature. The exhaustion is the stationary plateau that motivates Phase 2.

---

## Phase 2 — non-stationarity (`curiosity_drift.py`)

Changes exactly one variable: the struct region's content **drifts** — resampled (`swap`) or smoothly interpolated (`morph`) every `drift_period`. `drift_period=0` recovers stationary Phase 1. Headline metric: an **unbiased** current-struct-error (each WM evaluated on the current struct regardless of what it sampled).

- **Abrupt `swap` → naive LP fails, worse than random** (struct-err 0.082 vs 0.050 at drift@120). Mechanism, all visible in the time-course: right after a swap the struct error *rises*, so `relu(slow−fast)` reads **0** — LP is *structurally blind to a re-opened frontier* (it can only confirm progress where it's already descending); in that blind window **noise's sampling jitter wins fake-LP**, diverting LP for ~40–80 iters; and LP **forgets** struct while away, whereas random's constant rehearsal doesn't.
- **Continuous `morph` (Jasper's suggestion) fixes the abrupt-jump pathology** (struct-err 0.082→0.052; the noise-distraction largely gone) — but LP still doesn't beat uniform (steady-state last-third: LP occ 0.60 / err 0.030 vs random 0.33 / 0.027). The residual issue is partly fundamental: LP is a *derivative*, so at steady tracking (error held constant against drift) it reads ~0 and can disengage.
- **The diagnosis: no scarcity.** With the reducible region = ⅓ of the space, random's 33% rehearsal *already* tracks it — there is nothing for LP's concentration to buy. The doc's "select harder targets/levels" quietly assumes a *moving frontier you must find among many distractors*. → Phase 2b.

The doc's *core* claim survives: LP's struct-error **sawtooths under drift** (re-descends each segment) vs one descent under stationary — "plateaus on fixed, re-opens on moving" holds for LP's *engagement*; it is the *efficiency-vs-uniform* claim that needs scarcity.

---

## Phase 2b — scarcity + the fresh-FM ensemble (`curiosity_scarcity.py`)

A **scarce** arena: a small **needle** of reducible structure (a full-amplitude, deterministic, morphing pattern) occupying only 5–19% of the canvas, embedded in a large field of fresh **noise** plus a small **blank** strip. `needle_cols` is the scarcity knob. Drives are **per-patch** (no privileged region label — the drive must *discover* where the reducible structure is). Adds **disagree**: cross-model prediction variance over K=4 bootstrapped WMs — the doc's fresh-FM prescription, which rejects the aleatoric noise field by construction (models agree on the mean where content is random, disagree where structure is learnable-but-unlearned).

**Headline (amp=1.0, morph@150, 1000 iters) — unbiased needle-error (lower = better) and needle-occupancy:**

| needle (frac) | disagree | random | lp (naive) | surprise | min-surprise |
|---|---|---|---|---|---|
| 4 col (0.19) err | **0.034** | 0.043 | 0.051 | 0.074 | 0.034 |
| ·           occ | 0.71 (3.7×) | 0.19 | 0.53 | 0.08 | 0.92 |
| 2 col (0.10) err | **0.046** | 0.065 | 0.059 | 0.093 | 0.072 |
| ·           occ | 0.63 (6.5×) | 0.10 | 0.40 | 0.06 | 0.62 |
| 1 col (0.05) err | **0.129** | 0.145 | 0.141 | 0.156 | 0.164 |
| ·           occ | 0.43 (8.5×) | 0.05 | 0.20 | 0.04 | 0.10 |

1. **The ensemble reliably finds the needle**, over-sampling it **3.7× → 6.5× → 8.5×** uniform — a clean *monotone* dose-response on the *finding* axis (the scarcer the needle, the harder it homes in). It is the **best drive at every scarcity level**.
2. **Naive LP is *not* qualitatively broken** (the honest nuance, per Jasper). With enough iterations it *partially* finds the needle (occ 0.53/0.40/0.20 — several× uniform), and at maximal scarcity (5%) it **beats every drive except disagree** (0.141 < random 0.145 < surprise < min-surprise). The stark "LP drowns" is the *short-horizon* regime (in a 300-iter smoke lp occ was 0.008); with 1000 iters it recovers. So the ensemble is a **consistent quantitative improvement, not a strict qualitative necessity** — it finds/tracks the scarce moving frontier faster and more reliably, but a single-model LP signal is not fundamentally incapable.
3. **surprise** chases the noise field (never finds the needle); **min-surprise** is scarcity-dependent — competitive at 19% (big needle + small blank → dark-room seeker overflows onto the learned low-error needle) but **worst** at 5% (blank absorbs it). Using a **full-amplitude** needle was the fix that removed a low-scarcity confound: a low-energy needle is ≈0, so predicting-zero (what a blank-trained model does) scores low needle-error *for free*; a bright needle makes tracking require actually learning it.

**The synthesis.** Non-stationarity **re-exposes the noisy-TV problem** that Phase 1's stationary setting hid, and the requirement for a curiosity drive to beat uniform is **scarcity**. The fresh-FM ensemble (disagreement) is the machinery that finds and tracks a scarce moving reducible frontier in an aleatoric field — the regime the doc wrote it for — validated here as a reliable improvement (though not a qualitative rescue).

---

## What this establishes / caveats

- **Establishes**: (1) an intrinsic LP drive is *distinguishable* from surprise (noisy-TV) and predictability-seeking (dark-room), and uniquely *releases* an exhausted frontier (Phase 1); (2) naive LP struggles under non-stationarity for principled reasons — derivative-blindness to re-opened frontiers, noise fake-LP, forgetting — and continuous (gradual) drift mitigates the abrupt-jump part (Phase 2); (3) LP's advantage over uniform requires *scarcity*, and the fresh-FM ensemble reliably finds/tracks the scarce moving frontier where naive LP is unreliable — but naive LP is not qualitatively broken (Phase 2b).
- **Caveats**: single seed throughout; drives act on the *true* intrinsic reward (no learned value head — deliberate scope); no recurrent belief loop (deferred); the clean *monotone* dose-response lives on the *occupancy/finding* axis (the error-margin is best at 10% — at 5% even disagree tracks only partially, occ 0.43); the min-surprise confound is scarcity-dependent; the ensemble's benefit is a quantitative edge, largest at short horizons.
- **Does not show**: value-shaping of the inner loop by the outer drive (needs a trainable operator), compounding beyond one pass, OOD calibration recovery, or a learned LP value head — all deferred.

---

## Reproduction

```bash
cd experiments/
# Phase 1 — drive atom (stationary 3-region arena), 4 drives
modal run --detach a2a_forward/reaching/curiosity_reaching.py::curiosity_reaching --dataset mnist

# Phase 2 — content drift; swap (default) and morph
modal run --detach a2a_forward/reaching/curiosity_drift.py::curiosity_drift --dataset mnist                    # swap
modal run --detach a2a_forward/reaching/curiosity_drift.py::curiosity_drift --dataset mnist --drift-mode morph # morph

# Phase 2b — scarcity + ensemble (full-amplitude needle), scarcity sweep
modal run --detach a2a_forward/reaching/curiosity_scarcity.py::curiosity_scarcity --dataset mnist \
    --needle-amp 1.0 --n-iters 1000 --needle-cols-sweep "4,2,1"
```

Results JSON on the `language-reduction-data` volume under `/data/a2a_forward/{curiosity_reaching, curiosity_drift, curiosity_scarcity}/…`. `--quick` runs a fast smoke for each.

## Files

| File | Role |
|---|---|
| `curiosity_reaching.py` | Phase 1 — content-query WM + 4 intrinsic drives on the stationary 3-region arena; per-region LP; occupancy/error time-course |
| `curiosity_drift.py` | Phase 2 — adds `drift_mode` (swap/morph) content drift + `drift_period`; unbiased current-struct-error metric |
| `curiosity_scarcity.py` | Phase 2b — scarce needle-in-noise arena, per-patch LP, `disagree` (K-model bootstrapped ensemble) drive; scarcity sweep + unbiased needle-error |
