# The two-timescale value loop under perpetual drift — compounding, the capacity-gated null, and the corrected teacher

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md)
**Direct parent**: [../online_value_loop/README.md](../online_value_loop/README.md) — the *first* fully-online loop, which found **value is a slow/committed quantity** (fast-online discovery obstructed on both levers), reframed its benefit as **adaptation speed, not converged competence**, and named the next experiment: *"re-adaptation compounding across a sequence of Type-2 drifts."* This arc is that experiment — and it runs three cuts deep.
**Other parents**: [../meta_adapt/README.md](../meta_adapt/README.md) (#4b context latent `f(s,u,z)`; #4d/#4e the capacity-competition boundary), [../curiosity_control/README.md](../curiosity_control/README.md) (the two value terms — explore + exploit — as collection directors, graded by control), `../online_value_loop/meta_curiosity_loop.py` (self-tune the balance `b` by control reward → wandered).
**Code** (lives in this folder, per [STRUCTURE.md](../../../STRUCTURE.md)): `compounding_drift.py` (cuts 1+2), `online_value_loop.py` (cut 3), `compounding_drift_seeds_figure.py`, `value_carved_drift_capacity_figure.py`, `online_value_loop_figure.py`. The only shared dependencies are [`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py), which stay at the `mjc` node because every experiment there imports them. File index: [FILES.md](FILES.md).
**Status**: cut 1 clean positive; cut 2 capacity-gated/control-robust null; cut 3 the corrected teacher — landscape clean (3 seeds), self-tuning directional-but-noisy. Single-family designs; multi-seed where it matters (mechanism metrics trustworthy, control noisy). **Date**: 2026-07-21.
**Builds on this**: [`ballistic/`](../ballistic/README.md) (builds the *genuinely ballistic, non-re-groundable controller* this arc named as its next step, resolving the blind grader) · [`arm_substrate/`](../arm_substrate/README.md) (independently re-derives Cut 3's value-relevant teacher on a second task family)

---

## One-liner

Under perpetual drift, **the compounding the idea doc predicted is real — but it comes from the fast *memory* (learning layer), not the slow value-carving (meta layer).** The value-carving is capacity-gated and the controller is robust to it. And the *reason* the meta-layer looked inert is a **teacher problem, not a value-structure problem**: you cannot self-tune a value by a metric that cannot see what the value does. **Control is a near-blind grader** (a replanning controller reaches goals about as well with a stale model as a fresh one). Grade the meta-loop by the **value-relevant forward-model prediction error** instead — the literal cerebellum→VTA "prediction-error messenger," which reads the FM's error, not the downstream reward — and the explore/exploit balance has a **clean interior optimum** the loop can climb. The missing piece was never the value *structure* (two values, self-tuned); it was the *teacher*.

## The frame

The first online loop ([../online_value_loop/README.md](../online_value_loop/README.md)) established: value is slow/committed, its benefit is *adaptation speed*, and under perpetual drift (where you never converge) the payoff to look for is **compounding** — each new drift cheaper than the last because the invariant core is already paid for. This arc builds that experiment on the Cut #3 re-adaptation substrate, and finds the compounding is real but *reassigns its source*, then diagnoses and fixes the meta-layer's apparent inertness.

---

## Cut 1 — Compounding isolation (`compounding_drift.py`, `run_compounding_drift`)

**The experiment.** A sequence of related **Type-2 drifts** — the `push_rot` actuator-rotation conflict (φ and φ+π are *opposite* command→motion maps, the input-coupled conflict that makes pooling collapse; **not** noise, which collapses meta-learning to multitask). Measure **transitions-to-recover per drift** (held-out velocity-dim Δs R² vs #reward-free transitions) for:
- **factored** — the context-latent `f(s,u,z)` (a stable invariant core `f` shared across φ + a thin adaptable `z` encoding φ; fast adapt = infer `z` by a forward pass; slow loop consolidates `f` over the growing task bank);
- **monolithic + replay** (pooled = the "veridical, models everything" agent) and **monolithic, no replay** (forgetting);
- references: **offline-meta** (`f` pre-trained on the whole φ distribution → the "already paid for" ceiling) and **scratch** (re-fit per drift, no memory → floor).

**Result (3 seeds, threshold-free few-shot R² at N=20 transitions):**

| arm | few-shot R²@20: early → late | reading |
|---|---|---|
| **factored `f(s,u,z)`** | **0.31 → 0.97** (TTR 253 → 7) | **compounds** — climbs from naive, locks at the ceiling |
| **monolithic + replay (pooled)** | **0.66 → 0.21** (TTR → censored) | **anti-compounds** — a *liability* under conflict |
| monolithic, no replay | 0.78 → 0.44 | forgets |
| offline-meta (ceiling) | ~0.93 flat | "already paid for" reference |
| scratch (no memory) | ~0.92 flat | cheap here — see caveat |

- **Task-conditioned memory compounds**: after the invariant core consolidates (~drift 4), every new φ is a ~7-transition forward pass. The gap vs monolithic *widens* over the sequence — the meta-layer signature, isolated.
- **Pooled memory is a liability under conflict**: few-shot quality *collapses* (0.66→0.21), cumulative ~7× worse than task-conditioned and *worse than memoryless scratch* — the dynamic form of meta_adapt's "pooling is actively harmful."
- **Controls (causal):** the **damping floor** (no conflict) → pooling stays healthy (0.98→**0.90**, no collapse) and the factored advantage vanishes → **conflict is the cause, not exposure** (meta_adapt's "collapse at the floor," now dynamic). The **revisit** probe (cycle a fixed φ bank) → factored remembers (→0.95), both monolithic arms degrade.
- **Carve-sweep refinement** ("nothing more, nothing less"): latent width helps **monotonically, saturating at d≈8** — a 1-D task parameter does *not* imply a 1-D optimal latent (z is a learned *modulation channel*, not a minimal sufficient statistic). The carving knob is *which subspace* is adaptive, not the latent dimension count.
- **Honest caveat**: this invariant core (near-linear free-flight) is *cheap* to re-fit, so memoryless scratch is a strong floor — factored's win is the *trajectory/sign* (compounds vs collapses) + zero-gradient adaptation, not raw transition count.

## Cut 2 — Value-driven carving under a value-irrelevant drift (`compounding_drift.py`, `run_value_carved_drift`)

**The experiment.** Composes #4d/#4e's capacity-competition substrate (puck force field = value-irrelevant capacity sink; `push_rot` φ conflict; context-latent `f(s,u,z)→Δs(8)`; CEM-MPC control) with #1's drift *sequence*. The new ingredient: the value-**irrelevant** subspace **drifts too** (`puck_phase` θ rotates each drift — a backward-compatible knob added to `../pusher_env.py`). A **veridical** FM (matches all 8 dims) must re-learn the drifting puck every drift; a **value-carved** FM (drops the puck via the goal-reaching value's support = pusher dims [0,1,4,5], the 4d/4e hand-derived mask, frozen) tracks only φ. Per **guidance.md #3**, log re-adaptation at **both** FM and CONTROL level.

**Result — a clean, informative null (capacity boundary h∈{24,32,64}; consistent with 4d's multi-seed frontier):**

- **The carving mechanism works at every capacity**: carved puck-R² **−50 to −67** (drops the drifting puck), veridical **+0.2 to +0.4** (spends capacity tracking it).
- **The value-relevant FM benefit is CAPACITY-GATED**: few-shot pusher-vel R² gap (carved−veridical) = **+0.06 at h=32**, →**−0.02 at h=64** (the 4d frontier, now under drift). It appears only where capacity binds.
- **It never reaches CONTROL**: the control gap is ~0.01–0.02 at *every* capacity, and the two regimes never coincide — where carving helps the FM (small h) control is poor overall; where control is good (h=64) carving is neutral. **CEM-MPC replanning absorbs the FM gap even under perpetual drift** (guidance #3 / the afferent robustness, confirmed).

## The dissolution (the discussion that reframed cuts 1–2)

Three realizations turned the cut-2 "null" into a sharper claim:

1. **"Control-robust" = control is a *near-blind grader* of the value.** A replanning controller reaches goals about as well with a stale model as a fresh one, so a better model barely registers as better control. The puck's only channel to matter is *capacity* (the control cost already ignores it) — a narrow, non-transmitting channel.
2. **Compounding comes from the MEMORY, not the value-carving.** Both cut-1 factored and cut-2 veridical are context-latent models, and *both* compound. The value-relevance mask (which-dims-*matter*) adds only a capacity-gated FM refinement. The value that actually drives compounding is *which-dims-**changed*** (≈ learning progress) — and it is **compiled into the architecture**: after a drift the invariant core `f` has ~zero error so it isn't touched, while the changed part shows up as high-but-reducible error the encoder soaks into `z`. That *is* `−d‖e‖/dt` in effect — a learning-progress-shaped carve, realized architecturally, not as an explicit reward.
3. **So the meta-layer's apparent inertness is a teacher problem.** The idea doc's two-value structure (explore = learning-progress-ish, exploit = task-relevance, self-tuned) is right; what was missing is grading it by a teacher that can *see* what the value does.

---

## Cut 3 — The corrected full online loop (`online_value_loop.py`)

**The change.** A copy of `../online_value_loop/meta_curiosity_loop.py`'s two-timescale machinery (RPF-ensemble live FM on a recency FIFO; teleport-region collection scored by `grounded@b` = `b·reducible + (1−b)·exploit`; a slow REINFORCE gradient-bandit self-tuning `b = σ(θ)`) with **one load-bearing change**: the outer loop's **teacher**. Two selectable rewards:
- **`online_front`** — the **value-relevant FM re-adaptation error** (mean forward-model prediction error over the goal *corridor*, a fair, non-privileged region; the cerebellum→VTA messenger — value reads FM error, not reward);
- **`online_ctrl`** — downstream **control** goal-dist (= the parent's near-blind grader).

Substrate: a scarce, drifting `field_patch` crossing the goal corridor + an **off-corridor** aleatoric noise distractor (so the two value terms genuinely trade off). Fixed-`b` arms give the two **b-landscapes**.

**Result (3 seeds):**

| b (explore weight) | 0.0 | 0.3 | 0.5 | 0.7 | 1.0 | reading |
|---|---|---|---|---|---|---|
| **FM-error teacher** (corridor err, ↓) | 0.022 | 0.020 | **0.018** | 0.020 | 0.021 | **interior optimum at b=0.5** (15.6% rel spread) |
| **control teacher** (goal-dist, ↓) | 1.284 | 1.284 | 1.283 | 1.284 | 1.283 | **dead flat** (0.1% spread) — the blind grader |

- **The landscape is the load-bearing result.** The value-relevant FM prediction-error *sees* the explore/exploit balance — a clean **interior optimum** (both value terms needed: pure-exploit and pure-explore are both worse) — where downstream control is *flat*. This proves the conceptual claim and retroactively explains why the parent (`meta_curiosity_loop`, graded by control) wandered: it was grading by the blind teacher.
- **Self-tuning (directional confirmation, with an honest caveat).** From a displaced init (b=0.15), `online_front` **climbs toward the optimum** (0.15→~0.33 mean over 3 seeds; 2/3 reach 0.31/0.60, tight variance, and it avoids the bad extreme), while `online_ctrl` **random-walks** (0.09↔0.74, no gradient). But the **shallow bowl** (0.018 vs 0.021–0.022, ~20%) makes online REINFORCE noisy — the convergence is *directional, not crisp*. Firm-up attempts confirmed this is genuinely hard: **deepening the bowl is delicate** (a scarcer needle shifts the optimum to pure-explore), and more epochs don't overcome shallow-bowl REINFORCE noise. Crisp online convergence is deferred to a **variance-reduced outer optimizer** (a coarse bandit / successive-halving instead of REINFORCE) — optimization engineering, not a conceptual gap.

---

## The synthesis (the thesis)

Across the three cuts:

- **Compounding is delivered by the memory architecture** (amortized context-inference over a consolidated invariant core = the *learning layer*, an implicit learning-progress-shaped carve) — **not** by the value-relevance carving (the *meta layer*), which is capacity-gated and control-robust even under drift.
- **The explore/exploit value balance is real and self-tunable** (a clean interior optimum), **but only once you grade it by the value-relevant FM prediction error, not the task reward.** Control is a blind grader; the FM's prediction error is the messenger the value system must read (the idea doc's cerebellum→VTA pathway, made literal).

This closes the loop the whole arc approached — the missing piece was the *teacher*, not the value *structure* — and maps cleanly onto the idea doc's own learning-vs-meta-layer split: the learning layer does the heavy lifting (fast amortized adaptation over an invariant core); the slow value/meta-layer earns its keep by *selecting where adaptation effort goes*, read off the prediction-error signal.

## Reproduce

```bash
cd experiments/
# --- Cut 1: compounding isolation (headline 3 seeds + controls) ---
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag full_v1                    # seed 0
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag full_s1 --seed 1
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag full_s2 --seed 2
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag floor_v1 --family damping   # no-conflict control
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag revisit_v1 --drift-mode revisit
modal run --detach mjc/drift_value_loop/compounding_drift.py::compounding_drift --tag carve --carve-sweep         # latent-dim optimum
python3 mjc/drift_value_loop/compounding_drift_seeds_figure.py                                                   # headline figures

# --- Cut 2: value-carved drift + the capacity boundary ---
for h in 24 32; do modal run --detach mjc/drift_value_loop/compounding_drift.py::value_carved_drift --tag comp_h$h --n-drifts 20 --fm-hidden $h; done
modal run --detach mjc/drift_value_loop/compounding_drift.py::value_carved_drift --tag comp_v1 --n-drifts 20       # h=64
modal run --detach mjc/drift_value_loop/compounding_drift.py::value_carved_drift --tag static_v1 --n-drifts 20 --static-puck
python3 mjc/drift_value_loop/value_carved_drift_capacity_figure.py --runs 24:comp_h24 32:comp_h32 64:comp_v1

# --- Cut 3: the corrected teacher (landscape 3 seeds + displaced-init self-tuning) ---
for s in 0 1 2; do
  modal run --detach mjc/drift_value_loop/online_value_loop.py::online_value_loop --tag teacher_s$s --seed $s \
      --task-geom corridor --noise --arms "online_front,online_ctrl,b0.0,b0.3,b0.5,b0.7,b1.0,random"
  modal run --detach mjc/drift_value_loop/online_value_loop.py::online_value_loop --tag selftune15_s$s --seed $s \
      --task-geom corridor --noise --b-init 0.15 --arms "online_front,online_ctrl"
done
python3 mjc/drift_value_loop/online_value_loop_figure.py --tags teacher_s0 teacher_s1 teacher_s2 \
    --selftune-tags selftune15_s0 selftune15_s1 selftune15_s2
```
**Gotcha** (this session's): `--detach` runs survive client disconnects and commit to the volume, but launching many via `&` can lose all-but-the-last, and heavy runs can get killed before committing — prefer independent launches and pull from the volume (`modal volume get mujoco-control-data <path>`) if the local mirror is missing.

## Figures (mirrored to `figures/`)

- **Cut 1**: `compounding_drift_seeds/` — **`fig_fewshot`** (threshold-free compounding: factored climbs & locks, pooled collapses), `fig_ttr` (transitions-to-recover), `fig_cumulative` (pooled memory is a liability). Per-run: `compounding_drift_{full_v1,full_s1,full_s2,floor_v1,revisit_v1,carve_d*}/`.
- **Cut 2**: `value_carved_drift_capacity/` — **`fig_capacity_boundary`** (carving is capacity-gated, never reaches control). Per-run: `value_carved_drift_{comp_v1,static_v1,comp_h24,comp_h32}/` (fig1 FM-TTR, fig3 puck-drop, fig4 control-TTR).
- **Cut 3**: `online_value_loop_contrast/` — **`fig_landscape`** (interior optimum in FM-error vs flat control) and **`fig_selftune`** (front climbs, control random-walks). Per-run: `online_value_loop_{teacher_s*,selftune15_s*}/`.

## Caveats / next steps

- **Single-family designs** (`push_rot` conflict + puck/field patches). Mechanism metrics (few-shot R², frontier/corridor error, puck-drop, the b-landscape) are the trustworthy readouts; control is the noisy one.
- **Cut 2 is a null**, not a positive control win — the carving works but is capacity-gated + control-robust (consistent with 4d/4e/the first online loop).
- **Cut 3's self-tuning is directional, not crisp** — the landscape (the conceptual claim) is clean; the online convergence needs a variance-reduced outer optimizer. Expected to be hard (the interior optimum is delicate; REINFORCE on a shallow bowl is genuinely noisy).
- **Back-translate into [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md)**: "the teacher, not the value structure" — grade the meta-loop by the value-relevant FM prediction error (the cerebellum→VTA messenger), and the explore/exploit balance self-tunes; compounding lives in the learning layer. (Offered separately.)
- **Firm up the self-tuning** (a coarse bandit / successive-halving outer optimizer).
- ~~try a genuinely ballistic, non-re-groundable controller where the FM gap might transmit to control~~ **✅ landed → [../ballistic/README.md](../ballistic/README.md)** (Cut 4). A ballistic (feedforward, open-loop) controller makes the FM ~3× more behaviorally load-bearing than reactive (damping-staleness axis, 3 seeds), and online reward-free FM re-adaptation restores ballistic competence ~4.3× more than reactive after a Type-2 drift — the efferent bridge transmits under feedforward commitment. En route: the corridor-geometry explore/exploit drive is *confounded* (needle on-path is covered by exploit), so *directed collection* remains the open piece; the transmission itself was isolated on a controlled FM-quality axis.
