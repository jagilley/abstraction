# Ballistic control — when the cerebellar forward model becomes behaviorally load-bearing

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md)
**Direct parent**: [../drift_value_loop/README.md](../drift_value_loop/README.md) — this is the *"genuinely ballistic, non-re-groundable controller where the FM gap might transmit to control"* it named as its next step. Also extends Cut #3's biological reading (`replan_every` ≈ how ballistic a movement is; reward-free FM refit = cerebellar recalibration).
**Code** (lives in this folder, per [STRUCTURE.md](../../../STRUCTURE.md)): `ballistic_control.py` (4a), `ballistic_transmission.py` (4b), `ballistic_readapt.py` (4c); figures `ballistic_control_figure.py`, `ballistic_transmission_figure.py`, `ballistic_readapt_figure.py`. The only shared dependencies are [`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py), which stay at the `mjc` node because every experiment there imports them. File index: [FILES.md](FILES.md).
**Children**: [directed/README.md](directed/README.md) — directed collection (S0/S1/S2): does the value drive choose *where* to look, and does it pay? · [`arm/`](arm/) — **Cut 4c-arm**, this arc's second task family (curl-field drift on the planar arm), which is what retires the single-family caveat below; *runs in flight, no README yet*.
**Status**: 4a a confounded-drive negative + diagnostic (3 seeds); 4b + 4c clean multi-seed positives (3 seeds each). Single-family (damping / corridor reach) — mechanism metrics are the trustworthy readouts. **Date**: 2026-07-22.
**Builds on this**: [`directed/`](directed/README.md) (its child — the afferent link this cut named as its missing piece) · [`arm_substrate/`](../arm_substrate/README.md) (the second task family built to retire this arc's single-family caveat; re-derives Cut 4b's smooth quality axis)

---

## One-liner

The idea doc's efferent thread (§"the interface shape": *one efferent gain, value → FM*) leaves a piece untested: the **efferent** side of the cerebellum↔behavior bridge — *does a better forward model actually reach behavior?* The drift-value-loop arc found it did **not** under a CEM-MPC controller, because **replanning is a near-blind grader of the FM** (a controller that re-grounds to the true state every step reaches goals about as well with a stale model as a fresh one). This cut shows the resolution is not about the value *structure* but about the **control mode**: a **ballistic** (feedforward, open-loop, committed) controller — the one biological motor control actually uses, because you cannot replan faster than sensorimotor delay — makes the FM **~3× more behaviorally load-bearing** than a reactive one, and after a drift, **online reward-free FM re-adaptation restores ballistic competence ~4.3× more than reactive**. The forward model is exactly the asset you can keep calibrated **without reward** *and* the asset feedforward control depends on — the same object. Evolution gets fast committed movement and a cheaply-maintained motor model in one.

## The frame — the efferent bridge, and the reward-free / ballistic synergy

Two afferent/efferent halves of the cerebellum↔value bridge:
- **Afferent** (FM → value): grade the value/meta-loop by the FM's prediction error — the cerebellum→VTA "prediction-error messenger." The drift-value-loop [Cut 3](../drift_value_loop/README.md) landed this.
- **Efferent** (FM → behavior): a better FM should make a better controller. The drift-value-loop arc kept finding this **doesn't transmit** — CEM-MPC replanning absorbs the FM gap, so downstream control is *flat* over FM quality (the "blind grader").

The resolution here reframes the blind grader as a fact about the **controller**, not the value: replanning is blind because it re-grounds. The forward model matters *to the degree you must commit open-loop*. And that is not a lab artifact — it is the biological regime:

**The reward-free / ballistic synergy (lead reading).** The forward model is the one motor asset you can keep calibrated **reward-free** — every `(s, u, s′)` you experience is a dense, self-supervised label of the physics, no goal or reward required (and in the wild, *sampling rewards is dangerous* — you cannot afford to fail repeatedly to learn to move). It is *also* exactly the asset that **feedforward/ballistic control depends on**: fast movements outrun sensory feedback (~50–150 ms), so they must run open-loop off an internal predictor, and their accuracy is bounded by that predictor's quality. So the reward-free-maintainable thing and the ballistic-critical thing are **the same object** — the cerebellar forward model. Evolution gets speed (ballistic commitment) and safe, cheap maintenance (reward-free prediction error) at once. This cut makes that synergy concrete: FM quality is behaviorally load-bearing *specifically* under the control mode that is *specifically* maintainable without reward.

## Background — ballistic vs reactive, and the cerebellum

- **Ballistic** = feedforward, open-loop, committed. Plan the whole trajectory once (rolling the FM), execute it without looking at the world again. Like a dart throw or a saccade — once launched, unsteerable; FM errors **compound** with no correction. Biologically necessary because feedback is too slow for fast movement. A wrong internal model → **dysmetria** (overshoot/undershoot). Two variants here: `ballistic_cem` (open-loop CEM — model-*optimal* given the FM) and `ballistic_bc` (a behavior-cloned feedforward **motor program** `π(s,g)→` the whole action sequence — no online optimization, the most faithful "pre-compiled command").
- **Reactive** = closed-loop feedback. Re-plan every step from the *true* observed state; a wrong FM prediction is corrected the next step, so errors never accumulate. Note it still *uses* the FM — it is "model **+** constant feedback," not model-free — so it isolates *how much FM quality reaches behavior*. Biologically, the slow, visually-guided movement a cerebellar patient falls back on.

The cerebellum's classical role is exactly the forward model that makes ballistic control possible; this arc turns "how ballistic the movement is" into a **tunable causal knob** on how much the FM matters.

---

## Cut 4a — the confounded-drive negative + the diagnostic (`ballistic_control.py`)

**The attempt.** Make the *value loop's own* explore/exploit balance `b` (drift-value-loop Cut 3's `grounded@b`) produce the FM-quality differences, and grade identical FMs at a sweep of commitment horizons (`replan_every` reactive→ballistic) on the drifting-corridor substrate. Prediction: control-over-`b` flat when reactive, interior-optimum when ballistic.

**Result — a partial negative that survives to a diagnostic (3 seeds).** Control *is* a sighted grader once the reach is feasible (a shallow interior optimum ≈ `b=0.5–0.75`), but the transmission does **not** grow monotonically with commitment horizon — reactive and ballistic b-spreads are comparable and noisy. The reason, and the load-bearing find:

> **The b-drive is confounded in the corridor geometry.** The reducible needle sits *on* the corridor (y=0) and the aleatoric noise sits *off* it (y=0.9). So "exploit" (start→goal path density) actually **covers the needle** (visits it ~50% of the time, *lowest* needle-error), while "explore" (reducible-surprise) **chases the off-corridor noise** and visits the needle *least* (~3%). The intended explore/exploit roles **invert** — `b` was never cleanly varying the *control-relevant* FM quality.

Two secondary observations, both consistent with the arc: the effect is **drift-speed-dependent** (a slower drift washes out all `b`-differentiation — the value benefit is an adaptation-speed effect that needs sufficient non-stationarity), and online REINFORCE self-tuning of `b` stays **noisy** (reproducing drift-value-loop Cut 3's shallow-bowl caveat). **Conclusion: control variables.** Drop the confounded drive; manufacture the FM-quality axis directly (4b).

## Cut 4b — the clean transmission test (`ballistic_transmission.py`)

**The controlled axis.** Replace the confounded drive with **damping-staleness**: train the FM on `d_train` dynamics, test on fixed `d_test=2.0`; FM prediction-error on the true dynamics grows monotonically with `|d_train − d_test|`. Two design facts that are findings in themselves:
- The quality axis must be **smooth/global, not localized** — a localized needle is *open-loop-incompensable* (even a perfect FM can't counteract a strong local kick feedforward, which is *why* biology uses feedback there), so it saturates ballistic control at "fail" for every FM quality. Momentum/damping is smooth → the reach itself (accelerate → decelerate to stop) requires good modeling and is open-loop-compensable.
- The reach must be **feasible** (so the matched controller reaches) and staleness **one-signed** (over-damped only) to avoid the under/over-force sign confound.

Three controllers, same FM, true-env execution, identical reaches.

**Result (3 seeds, mean; control = median goal-dist, lower better):**

| FM error (staleness →) | 0.001 | 0.028 | 0.061 | 0.107 | 0.163 | slope d(ctrl)/d(FM-err) |
|---|---|---|---|---|---|---|
| **reactive** | 0.010 | 0.014 | 0.025 | 0.043 | 0.066 | **+0.35 ± 0.00** (1.0×) |
| **ballistic_cem** | 0.109 | 0.209 | 0.256 | 0.284 | 0.304 | **+1.07 ± 0.08** (3.0×) |
| **ballistic_bc** | 0.102 | 0.185 | 0.248 | 0.293 | 0.298 | **+1.16 ± 0.03** (3.3×) |

**Ballistic control transmits FM quality ~3–3.3× more than reactive**, and the genuinely-feedforward **BC motor program transmits the *most*** (it can't re-optimize per episode, so it bakes in the FM's staleness). Reactive re-grounds past the stale FM and stays competent (slope +0.35, identical across all 3 seeds). The efferent bridge transmits — in proportion to feedforward commitment. *(Note: reactive is the more accurate controller in absolute goal-distance — open-loop is intrinsically harder. The claim is about the **slope** — how much FM quality reaches behavior — not that ballistic beats reactive on accuracy.)*

## Cut 4c — the end-to-end loop (`ballistic_readapt.py`)

**The integration.** Make the FM-quality axis **endogenous**: a Type-2 damping drift `d0=6 → d1=2` leaves the FM stale; the learning layer re-adapts it **online from reward-free `d1` transitions** (self-supervised — the cerebellum re-learning the new table). Snapshot the FM along the re-adaptation trajectory and grade each snapshot under reactive vs ballistic. (Operate at `d1=2`, where ballistic control *can* be competent with a matched FM — cf. 4b — so re-adaptation can restore competence; drift *from* the stale high-damping FM.)

**Result (3 seeds; drift d0=6→d1=2):**

| reward-free re-adapt transitions | 0 (stale) | 200 | 500 | 1200 | 3000 | 6000 | recovery gain (stale − recovered) |
|---|---|---|---|---|---|---|---|
| **FM error @ d1** | 0.132 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 | — |
| **reactive** | 0.053 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | **+0.043 ± 0.001** (1.0×) |
| **ballistic_cem** | 0.295 | 0.108 | 0.108 | 0.108 | 0.108 | 0.109 | **+0.187 ± 0.014** (4.3×) |
| **ballistic_bc** | 0.295 | — | — | — | — | 0.106 | **+0.190 ± 0.020** (4.4×) |

Three things land together:
1. **Ballistic control recovers** as the FM re-adapts (0.295 → 0.108, ~2.7× more accurate), reaching the **matched-FM ceiling** (a fresh `d1` FM gives the same 0.108) — re-adaptation is behaviorally load-bearing under feedforward commitment.
2. **Reactive is ~flat and already competent** throughout (0.053 → 0.010) — it never needed the re-adaptation. So the **behavioral value of re-adaptation is ballistic-specific** (~4.3× larger).
3. **FM error tracks the ballistic recovery** — the value signal the meta-layer reads (Cut 3's afferent teacher) is a valid proxy for the *behavioral* payoff, and only because control is ballistic (for reactive it over-predicts the need).

The whole thesis in one run: the **learning layer** delivers FM quality (fast, reward-free re-adaptation), the **value/meta signal** (FM error) tracks it, and it **cashes out as behavior** only because motor control is feedforward.

---

## Discussion — the synergy, and what it closes

The reward-free / ballistic synergy is the reading to lead with: the cerebellar forward model is the **reward-free-maintainable** asset *and* the **feedforward-critical** asset — the same object. That is why the biology puts a large, plastic, self-supervised predictor (cerebellum) in charge of fast movement: you can keep it accurate from ordinary experience (no dangerous reward-hunting), and it is exactly what ballistic control consumes. This cut closes the **efferent bridge** the idea doc's stock-take flagged as missing: combined with drift-value-loop Cut 3 (the afferent FM-error teacher) and the learning-layer compounding (Cut 1), the three-part architecture — cortical/cerebellar learning loop, dopaminergic value/reward loop, cerebellar FM as the bridge — now has behavioral teeth in **both** directions.

## Honest caveats

- **Single-family** (damping drift, corridor reach). Mechanism metrics (slopes, recovery gains, FM error) are the trustworthy readouts.
- **The re-adaptation is drive-free** (uniform reward-free collection = the *learning layer*). The value loop *directing where to collect* is the piece **4a showed is confounded** in this geometry — so "the value loop chooses collection" is **not** cleanly demonstrated here; the "value" is the FM-error signal + the ballistic-specific payoff, not directed collection. This is the honest headline: **the efferent bridge now transmits, but *directed collection* remains confounded.**
- **Recovery is fast/step-like** (~200 transitions — a pure damping shift is easy to re-learn). A richer drift (or the factored/compounding FM from drift-value-loop Cut 1) would stretch the recovery into a graded trajectory — the natural next probe if you want the adaptation *curve*, not just its endpoints.
- **Ballistic ≠ more accurate.** Reactive wins raw goal-distance; the ballistic value is speed (unmeasured here) + being the mode where FM quality transmits.

## Reproduce

```bash
cd experiments/
# 4a — the confounded-drive landscape (fast-drift, 3 seeds) + the diagnostic
for s in 0 1 2; do
  modal run --detach mjc/ballistic/ballistic_control.py::ballistic_control --tag land48_s$s --seed $s \
    --task-geom corridor --noise --arms "b0.0,b0.25,b0.5,b0.75,b1.0,random" --replan-sweep "3,12,30" \
    --corridor-r 0.5 --plan-h 30 --plan-hp 30 --rounds 48 --outer-m 6 --n-eval 24 \
    --cem-iters 4 --drift-cycles 2.0 --patch-amp 3.5 --patch-sigma 0.30
done
python3 mjc/ballistic/ballistic_control_figure.py --land-tags land48_s0 land48_s1 land48_s2 \
    --selftune-tags selftune_s1 selftune_s2

# 4b — the clean transmission test (damping-staleness, 3 seeds)
for s in 0 1 2; do
  modal run --detach mjc/ballistic/ballistic_transmission.py::ballistic_transmission --tag trans_s$s --seed $s \
    --controllers "reactive,ballistic_cem,ballistic_bc" --d-test 2.0 --damp-trains "2.0,2.8,3.8,5.2,7.0" \
    --corridor-r 0.4 --plan-h 34 --n-eval 40 --bc-tuples 1500
done
python3 mjc/ballistic/ballistic_transmission_figure.py --tags trans_s0 trans_s1 trans_s2

# 4c — the end-to-end re-adaptation loop (3 seeds)
for s in 0 1 2; do
  modal run --detach mjc/ballistic/ballistic_readapt.py::ballistic_readapt --tag readapt_s$s --seed $s
done
python3 mjc/ballistic/ballistic_readapt_figure.py --tags readapt_s0 readapt_s1 readapt_s2
```

**Gotchas** (this session's): (i) launch the heavy detached runs **independently** and don't pile on competing background tasks — the harness can evict older background clients, and a `--detach` client killed mid-run *before its volume commit* loses the run (pull from the volume with `modal volume get mujoco-control-data <path>` if the local mirror is missing). (ii) Both smoke tests looked like nulls until two regime fixes: the FM-quality axis must be **smooth (damping), not a localized needle** (open-loop-incompensable), and the reach must be **feasible** so the matched controller reaches. (iii) In 4c the drift direction matters — operate where ballistic is competent (`d1=2`) and drift *from* the stale FM (`d0=6`), not the reverse.

## Figures (mirrored to `figures/`)

- **4a**: `ballistic_control_contrast/` — `fig_landscape` (control-over-b per commitment horizon), `fig_transmission` (spread vs horizon), `fig_selftune`.
- **4b**: `ballistic_transmission_contrast/` — **`fig_transmission`** (control vs FM-error per controller — the headline slopes) and `fig_slopes` (transmission slope per controller).
- **4c**: `ballistic_readapt_contrast/` — **`fig_recovery`** (control + FM-error vs re-adaptation transitions; ballistic recovers, reactive flat) and `fig_gain` (recovery gain per controller — ballistic-specific).
