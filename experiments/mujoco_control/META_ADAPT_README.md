# Meta-adaptation over a distribution of dynamics — collapse at the floor, gap opens under conflict

**Idea doc**: [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) (§"Meta-RL: the two-timescale structure"; non-stationarity is the load-bearing part) · **Parent**: [Cut #3](README.md#cut-3--operator-intervention-reward-free-re-adaptation-after-a-dynamics-shift) (single reward-free re-adaptation) · [DIRECTED_READAPT](DIRECTED_READAPT_README.md) (the pivot: phenomenon-first, dissectible)
**Code**: `meta_adapt.py` (+ `meta_adapt_sweep_figure.py`) · **Status**: floor = clean collapse anchor; actuator-conflict sweep = the gap opens monotonically. Single family-seed. **Date**: 2026-07-18.

---

## The question (made non-tautological)

A persistent learner on *any* shifting environment improves with exposure, so "adaptation gets cheaper over a sequence" is near-tautological. The falsifiable claim is the **standard meta-learning** one: meta-train a fast adapter over a *distribution* of dynamics, then measure **few-shot adaptation to HELD-OUT dynamics** against a **multitask/pooled init trained on the same data**. That is exactly the comparison [RHM_META_LEARNING](../rhm/ratchet/RHM_META_LEARNING_README.md) found *collapses* (meta → plain multitask) with no exploitable structure. So we track the **meta − multitask gap as an order parameter**, not a checkbox — and ask what a control-dynamics family needs before learn-to-adapt beats pooling.

## Apparatus (`meta_adapt.py`)

Substrate = Cut #3's puck-free momentum reaching. **Four initializations**, each adapted to held-out tasks by the *identical* inner loop (only the starting weights differ):
- **meta** — first-order **Reptile** ("learn an initialization"), warm-started from the pooled init → the gap isolates purely the *adaptability refinement*, and it dodges the from-scratch-Reptile convergence confound.
- **multitask** — one FM pooled over all train buffers. **The load-bearing control** (same data as meta).
- **single-task** — an FM on one reference task (Cut #3's d0 idiom).
- **scratch** — random init (lower bound).

Readout = **held-out velocity-dim Δs R²** vs #adaptation transitions N (velocity dims carry the task variation; all-dim R² saturates ~1 and is a poor discriminator), aggregated as **median over test tasks**; CEM-MPC planning is the behavioral corroborant. Per-task **oracle** ceiling from a large disjoint pool. Load-bearing stabilizers: gradient clipping (a rare stiff wall-contact transition in a few-shot slice otherwise blows up Adam), `adapt_lr` 5e-4, decaying Reptile `meta_lr`.

## Result 1 — the FLOOR (1-D damping): COLLAPSE (`full_v2`)

Family = `joint_damping ∈ [0.05, 2.0]` log-spaced, disjoint **interior (interpolation)** test dampings (native DGP knob, no env change).

**meta − multitask gap ≈ 0 at every budget** (−0.002 … +0.000; planning gap likewise ~0). Both pooled/meta inits transfer to held-out interior dampings **near-ceiling zero-shot** (0.988 R²): the family is so cross-task-shared there is almost nothing task-specific to learn few-shot, so an adaptability prior has no lever — the RHM collapse, anchored on control. The metric *has* resolution (scratch needs ~80 transitions; single-task, extrapolating from one extreme damping, starts at 0.83 and catches up by ~40), so meta and multitask are *genuinely* identical, not both pinned at a saturated ceiling.

## Result 2 — CONFLICT (actuator rotation): the gap OPENS (`actuator_p16/p2/pi`)

The floor told us what's missing: **task conflict**. Family = an **input-coupled** rotation of the command→motion map (`push_rot` in `pusher_env.py`, additive/off-by-default → Cuts #1–3 byte-identical): for task φ the effective actuation is `gear·R(φ)·u`. This is *not* an additive force-field (a constant output bias any init learns by nudging one term); rotating how commands map to outcomes makes φ and φ+π opposite, so a model pooled over φ∈[−Φ,Φ] averages the command gain to `sin(Φ)/Φ` — **1.0 at Φ=0, 0 at Φ=π (arity-1 degenerate)**. Φ is a clean **conflict dial**.

| Φ (rad) | pooled zero-shot R² | meta−multitask gap (peak few-shot) | meta vs multitask sample-efficiency |
|---|---|---|---|
| 0 (floor) | 0.99 | ≈ 0 | tie |
| π/6 ≈ 0.52 | 0.92 | +0.006 | tie |
| π/2 ≈ 1.57 | 0.49 | **+0.156** | meta **adapt@20** vs multitask **adapt@160** (~8×) |
| π ≈ 3.14 | 0.02 | **+0.369** | meta adapt@160; multitask never reaches threshold |

**The gap opens monotonically with conflict** (`fig_gap_vs_conflict.png`) — the collapse and the separation are the same order parameter crossing a transition in input-coupled conflict. Corroborants:
- **Textbook MAML signature at every Φ**: meta is *slightly worse* zero-shot (gap at N=0 is −0.05 to −0.07) but decisively faster few-shot (gap peaks at N=5–20, decays as N→∞) — it emerges, not designed in.
- **Behavioral dissociation**: the CEM-MPC planning gap is positive and grows with Φ (not an FM-R² artifact).

### Two honest nuances
- **At Φ=π pooling becomes *actively harmful*** (multitask ≈ scratch): the pooled representation has learned "command → ~0 effect" and must be *un*learned. So the max-conflict gap partly reflects "pooling worse than nothing"; **Φ=π/2 is the cleaner meta ≫ multitask ≫ scratch regime.**
- **A clean single-task prior is competitive at medium conflict** (better adaptation substrate than the *conflicted* pooled one). Pooling *over conflict* corrupts the representation in a way one unconflicted prior avoids. Neither touches the headline (meta vs multitask is the fair same-data comparison), but both are real.

### Caveats
Single family-seed (one train/test split + collection seed; the 4-point monotone Φ trend is the reassurance). Meta warm-started from multitask → this isolates adaptability-refinement-on-top-of-pooling, not from-scratch MAML. Reward-free FM-meta (no value in the loop yet — that is a later cut).

## Cut #4b — the dissectible outer memory: an explicit context latent (`meta_context.py`)

The Reptile result opened the gap but hid the outer memory in the weights (not probeable). Cut #4b replaces it with an **explicit context latent** `z` conditioning the FM `f(s,u,z)`: a permutation-invariant (DeepSets) encoder infers `z` from a few recent `(s,u,Δs)` transitions (PEARL / amortized system-ID). Few-shot "adaptation" is then a **forward pass** (encode N context transitions → `z`), no gradient steps. Arms: **context** (`z=E(first N)`) vs **z0** (context severed, `z=0` = family-average) vs a plain-`f(s,u)` per-task **oracle**.

**Result 1 — context opens the gap and is *more* sample-efficient than Reptile.** At Φ=π/2, context recovers to **near-oracle (0.998) in just 5 transitions** (vs Reptile's ~20) while z0 stays flat (0.356) — amortized inference reads φ off a handful of transitions directly.

**Result 2 — `z` decodes the task parameter: system identification.** A linear probe `z → φ` (fit on train-task `z`, tested on held-out-task `z`) hits **R²=0.998** — the outer memory is a near-linear encoding of the rotation angle. This is the dissectible payoff the guardrail wanted: `z` is the low-dim "value-relevant direction" the two-timescale doc predicts, and we can read it.

**The dissociation (the sharpening the floor control gave).** Running the damping floor as a control separates two things that look like one:

| | `z`→param probe R² (system-ID) | context − z0 gap (adaptation benefit) |
|---|---|---|
| floor (damping) | **0.987** | **+0.006** (≈0) |
| conflict (actuator Φ=π/2) | **0.998** | **+0.642** |

**System-ID and adaptation-benefit are dissociable.** The encoder identifies the task in *both* regimes (it always builds a `z` that decodes the parameter), but that identification only *buys adaptation under conflict*. At the floor, `z` decodes the damping perfectly yet the gap is ~0 — knowing the task is *true but useless*, because the shared dynamics mean the family-average already predicts held-out tasks. This is the interface frame made concrete: `z` is the low-dim task-identifying slice of the forecast (always present); whether it is *value-relevant* (helps control) is set by task structure.

## Cut #4c — value in the loop: value-directed identification (a robust negative) (`meta_active.py`)

The first cut with **value** in the loop: does a value-of-information drive **identify** the task (infer `z`) in fewer transitions than passive collection? Value = epistemic — score candidate commands by **disagreement across the codebook of task-latents** `{z_k}` (which span the *real* task manifold — the fix for DIRECTED_READAPT's confident-prior blindness), read off the `z`-conditioned FM. `z` inferred by optimization (robust to collection distribution). Four collection arms: **passive** (OU) · **magnitude** (max‖u‖ control) · **myopic VoI** · **navigating VoI** (an info-MPC that plans to *reach* high-disagreement states).

**Regime gate (confirms DIRECTED_READAPT Exp-1).** The **global** rotation is info-rich — passive IDs φ in ~1 transition (no scarcity) → all arms null. A **localized-rotation patch** (`rot_patch`: only in-patch transitions carry φ) creates scarcity → active collection beats passive.

**The negative — robust over 4 seeds, and *not* a naive-implementation artifact** (`act_patch4{,_s1,_s2,_s3}` + `meta_active_seeds_figure.py`). Under scarcity, value-of-information does **not** beat the trivial max-magnitude heuristic:
- **The navigating info-MPC provably navigates** — patch visitation: passive 0.09 → magnitude/myopic-VoI 0.13 → **navigate 0.28** (3× passive, tight across seeds). The mechanism works.
- **But navigation doesn't buy identification** — on in-patch R² and φ-decode error the active arms are a **wash** (bands overlap), with **magnitude marginally best**, and navigate even *slower* at very low N (it spends early transitions travelling). The single-seed rankings (magnitude≫VoI in the 3-arm run; VoI≫magnitude in one 4-arm seed) were **seed-noise** — the multi-seed shows no VoI arm robustly dominates.

**Why (the boundary condition).** For a rotation the most informative command *is* the largest one, so max‖u‖ **simultaneously** maximizes spatial coverage (finding the scarce patch) *and* per-transition signal — for free. And identifying a **1-D** parameter saturates after a handful of good in-patch transitions, so the navigating drive's 3× extra visits are redundant. Value-of-information's lever should appear only for **harder identification**: a higher-dimensional task parameter, or a family where informative actions are *not* simply "large." On low-dim system-ID, VoI is over-engineering.

**Caveats.** Single family/rule design; 4 collection seeds. The info-MPC rolls out with the mean-codebook `z_ref` (valid for navigation — position dynamics are task-independent up to the patch). In-patch R² and φ-error agree; all-region R² is confounded (out-of-patch is `z`-independent) and is not the headline. `push_rot` + `rot_patch` are additive/off-by-default in `pusher_env.py` → Cuts #1–3 byte-identical.

## Reproduce

```bash
cd experiments/
modal run mujoco_control/meta_adapt.py::meta_adapt --quick                                  # smoke (damping)
modal run mujoco_control/meta_adapt.py::meta_adapt --tag full_v2                             # FLOOR (damping collapse)
modal run mujoco_control/meta_adapt.py::meta_adapt --tag actuator_p2 --family actuator --conflict 1.5708   # conflict Φ=π/2
# sweep: --conflict 0.5236 (π/6), 1.5708 (π/2), 3.14159 (π); then:
python3 mujoco_control/meta_adapt_sweep_figure.py                                            # gap-vs-conflict figure

# Cut #4b — context latent
modal run mujoco_control/meta_context.py::meta_context --tag ctx_p2 --family actuator --conflict 1.5708   # conflict + z-probe
modal run mujoco_control/meta_context.py::meta_context --tag ctx_floor --family damping                   # floor control

# Cut #4c — value-directed identification (4 collection seeds, then aggregate)
for s in 0 1 2 3; do modal run mujoco_control/meta_active.py::meta_active --tag act_patch4${s:+_s$s} --rot-patch --seed $s; done
python3 mujoco_control/meta_active_seeds_figure.py     # multi-seed figure (act_patch4 = seed 0)
```

Results/figures commit to `mujoco-control-data` under `/data/meta_adapt/<tag>/` + `/data/meta_context/<tag>/` and mirror to `figures/meta_adapt_<tag>/` (+ `figures/meta_adapt_sweep/`, `figures/meta_context_<tag>/`).

## Figures
Reptile (`figures/meta_adapt_<tag>/`): `fig1_adaptation_curves` (held-out R² vs N, 4 inits — the MAML crossover) · `fig2_transitions_to_adapt` · **`fig3_meta_vs_multitask_gap`** (the order parameter) · `fig4_planning`. Sweep (`figures/meta_adapt_sweep/`): **`fig_gap_vs_conflict`**. Context latent (`figures/meta_context_<tag>/`): `fig1_context_curve` · `fig2_context_gap` · **`fig3_z_probe`** (z→param system-ID scatter). Value-directed (`figures/meta_active_<tag>/`): `fig1_identification_curve` (in-patch R², arms) · `fig2_phi_error`; multi-seed **`figures/meta_active_seeds/fig_seeds`** (ID quality + the visitation mechanism — the headline).

## Next
The arc is complete on one substrate: **collapse→gap (phenomenon) → dissectible `z` (system-ID) → system-ID ⊥ adaptation-value (dissociation) → value-directed identification (VoI in the loop — a robust, scarcity-gated negative).** The interchange story is: `z` is the legible low-dim task-identifier; whether it helps *adaptation* is set by conflict (4b); a VoI drive to *acquire* `z` faster is over-engineering for low-dim system-ID (4c). Candidate next probes: **(a)** disc-4 value-shaping *caused by* the outer loop (does the `z`-conditioned FM become value-shaped when the meta-objective is task reward, not prediction error?); **(b)** raise identification difficulty (higher-dim task parameter, or informative-actions ≠ large) — the boundary condition where 4c predicts VoI should finally earn its keep; **(c)** back-translate the whole arc into [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) / the belief tree.

---

## Cut #4d — Value-shaping × meta-conditioning: two separable capacity levers, and *when* re-allocation buys control (`meta_value_shaping.py`)

**Date**: 2026-07-19. Takes candidate **(a)** above (disc-4: does a value objective *cause* the FM to become value-shaped?) and fuses it with the conflict/context machinery. **Parents**: [`value_shaping.py`](VALUE_SHAPING_README.md) (stationary value-shaping via a *hand-set* per-dim weight; one task, no meta) and Cut #4b `meta_context.py` (the context latent `z` that holds a per-task conflicted map). **Program**: [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) discriminator 4 + the FM↔value interface ("one efferent gain: value-shaping = capacity **re-allocation**, a priority field over prediction targets"). **Status**: disc-4 landed on the FM side; the *control* payoff is a **multi-seed-validated, capacity-gated positive** (4 seeds), with the easy-pusher run as its matched null. Single family design.

### The question and the 2×2

`value_shaping` showed a hand-set `λ_puck=0` re-allocates a small FM off the value-irrelevant puck (stationary). [REACHING_LOOKAHEAD](../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md) showed value-shaped ≻ veridical for control, but from a *hand-coded* value. Neither showed the shaping is **caused by** a value objective, and neither needed the meta machinery. This cut crosses the two on **one** substrate (puck + `push_rot` φ conflict + a capacity-limited context-conditioned FM `f(s,u,z)→Δs(8)`), as a **2×2**:

|  | **veridical** (match all 8 dims) | **value-shaped** (match pusher dims) |
|---|---|---|
| **context** `z=E(N)` | the plain meta FM | value-shaped, per-task |
| **pooled** `z=0` | the gap-result baseline | value-shaped, family-averaged |

The value-shaped arm matches the **support of the goal-reaching value** `V(s) = −‖pusher_pos − g‖ − β‖pusher_vel‖` (arrive *and* stop, = the CEM cost) — its state-support is `{pusher pos, pusher vel}`, zero on the puck. So "value-shaped" = match the **pusher** dims `[0,1,4,5]`, drop the **puck** dims `[2,3,6,7]`: the same split `value_shaping` hand-set, but here **derived from `V`, not tuned** (if `V` cared about the puck, the puck would be matched). The puck carries a nonlinear force field (`value_shaping`'s value-irrelevant, reducible, capacity-hungry sink); staged like `value_shaping` (FM-side first, planner second).

**Load-bearing false start (what we learned first).** The first design made value-shaped match pusher **position only** over an H-step rollout, so velocity-relevance would *emerge* through composition (velocity errors integrate into position errors). It **collapsed to a vacuous loss**: at the fine control step (`frame_skip=3`) the pusher barely moves over 6 steps, so "match position" is trivially satisfied and teaches nothing about the actuated velocity (the φ-conflicted dim). Lesson: velocity-relevance-through-composition needs a *long real-time* rollout; matching the full value **support** (pos+vel), one-step, is the robust realization. Everything below uses the one-step weighted objective (`value_shaping`'s discipline — only the per-dim loss weight and the context mode differ across arms).

### Stage 1 — FM side (`full_v1`, easy pusher; oracle pusher-vel R² 0.974)

| readout | finding |
|---|---|
| **puck-drop is value-caused** | veridical *increasingly* models the value-irrelevant puck as capacity grows (puck-vel slide R² −0.03 → 0.68 across h 16→256); value-shaped **never** does (R²≪0 at every width). The re-allocation is discovered from `V`'s support, not hand-coded in the readout. |
| **capacity benefit: MUTED** | shaped pusher-vel R² > veridical only at **h=16** (0.856 vs 0.808, +0.05); by h=32 both are ~0.94 — the pusher free-flight is too *easy* to be capacity-bound, so freed capacity has nothing value-relevant to buy. |
| **fusion (clean)** | pooling (`z=0`) collapses on the φ-conflicted pusher for **both** objectives (context ~0.95 vs pooled ≤0.32 at every capacity). Value-shaping does **not** rescue pooling — the conflict lever is orthogonal to and dominates the shaping lever. |
| **system-ID preserved** | z→φ probe R² = 0.983 (verid), 0.989 (shaped). |

### Stage 2 — planning (CEM-MPC goal-reaching), and the boundary condition

The FM-side R² *cannot* reveal whether the re-allocation **matters** (veridicality ⊥ usefulness — [REACHING_LOOKAHEAD](../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md)); only planning can. A fixed CEM planner (value = pusher-goal dist + terminal-vel penalty; cost **ignores the puck**) rolls each arm's FM; committed `replan_every`-step segments make the model load-bearing (Cut #3).

- **`plan_v1` (easy pusher) — control-NEUTRAL.** Context-mode planning is a **wash**: veridical ≈ value-shaped at every capacity (verid marginally *ahead* at small caps). The value-shaped FM is a **drastically worse literal simulator** (all-dim veridicality ≪ 0 — puck dropped) at **equal control** — the *veridicality-unnecessary* half of REACHING_LOOKAHEAD, but **not** the value-shaping-superior half. Diagnosis: on an easy, puck-decoupled task the puck **neither competes for capacity the pusher needs nor misleads the planner**, so dropping it is *free but unrewarded*. This is the **matched null** that makes the next run interpretable.
- **The prediction it sets up:** value-shaping should turn control-*superior* **iff** the value-relevant dynamics are capacity-hungry (so the puck genuinely competes).

### The capacity-competition test (`pfield`, hard pusher; 4 seeds) — the multi-seed positive

Turn on a **pusher force field** (`field_pusher_amp=2.0`, distinct phase — `value_shaping`'s device) so the value-relevant pusher dynamics become capacity-hungry (oracle pusher-vel R² 0.974 → 0.917). Everything else identical to `plan_v1`. Aggregated over **4 seeds** (`plan_pfield_v1` + `_s1/_s2/_s3`; `meta_value_shaping_seeds_figure.py`):

| readout | h=16 | h=32 | h=64 | h=128 | h=256 |
|---|---|---|---|---|---|
| **FM-side** pusher-vel R² gap (shaped−verid) | **+0.112** | +0.059 | +0.024 | −0.010 | −0.007 |
| **planning** gap (verid−shaped; >0 = shaped better), mean±sem | +0.022±0.016 | +0.009±0.014 | **+0.036±0.012** | — | −0.009±0.004 |

- **FM-side capacity frontier: clean & monotone.** Value-shaping frees capacity that lifts the value-relevant pusher-vel R² most where capacity binds (+0.11 at h=16), converging to ~0 by h=128.
- **Planning benefit: robust at the sweet spot.** The behavioral order parameter is **+0.036±0.012 (≈3σ) at h=64** — value-shaped plans ~16% closer to goal — marginal at h=16, null at h=32, and slightly *negative* at saturation (h=256). Pooling collapses behaviorally for both objectives (fusion holds in planning too).
- **The FM-gain / planning-gain mismatch (the interesting bit).** The FM-side gap is *largest* at h=16 (+0.11) but the *planning* gap is largest at **h=64**, where the FM gap is only +0.024. **Small one-step fidelity gains compound over the planning horizon** into meaningful control gains — but only once the FM clears a "good-enough-to-plan-through-the-field" threshold (h=16/32 are too weak for the edge to translate; h≥128 both saturate). The control payoff peaks where capacity **binds** *and* the model is **veridical enough to compose** — the same "load-bearing *and* composable" sweet spot as Cut #3's `replan_every=12` turnover.

### The synthesis (disc-4, closed on MuJoCo)

The `plan_v1` ↔ `pfield` pair is the causal isolation: **value-shaping's re-allocation converts to a control benefit *iff* capacity binds on the value-relevant dynamics** (control-neutral without it, control-superior with it). Put together across all runs, the efferent FM↔value link is demonstrated end-to-end: (1) value **causes** a dissectible re-allocation (puck-drop from `V`'s support, not tuned); (2) it is **orthogonal to and dominated by** the meta/conflict lever (fusion, R² + planning); (3) **system-ID preserved** (probe 0.90/0.94); (4) **veridicality ⊥ usefulness** (far worse simulator, equal-or-better planner); (5) **control-neutral** when the value-irrelevant subsystem is decoupled & capacity ample; (6) **control-superior** under capacity competition (+0.036±0.012, 4 seeds).

### Substrate bug fixed (backward-compatible)

This cut is the first to run **`push_rot` and a pusher force field together**, which exposed a latent clobber in `pusher_env.py`: `_apply_actuator_rot` and the pusher branch of `_apply_fields` both wrote `qfrc_applied[pusher_dof]` with `=`, so the rotation (applied last) silently **overwrote** the field. Fixed by summing the field base into the rotation correction. Verified **byte-identical on the field-off path** (base=0), so Cuts #1–3, `value_shaping`, `meta_context`, `meta_adapt`, and the `rot_patch` cut are unchanged.

### Caveats
- The control benefit is **modest** (~16% relative at h=64) and **capacity-specific** (peaks at h=64; muted where the FM is too weak or saturated). Single family design (the 4 seeds vary the train/test split + collection + init, not the family structure).
- **"Reward" here is a decision-aware, value-*support*-weighted prediction objective** (the shaping is *derived from* `V`), **not** a closed-loop reward-*driven* outer loop backpropagated through the planner. Closing that loop (the true two-timescale meta-objective) is the next step, not this cut.
- The value-shaped puck-vel R² is hugely negative (predicts ~0 Δ for a sliding puck) — "dropped", displayed clipped in figures.

### Reproduce
```bash
cd experiments/
modal run mujoco_control/meta_value_shaping.py::meta_value_shaping --quick --plan            # smoke
modal run mujoco_control/meta_value_shaping.py::meta_value_shaping --tag full_v1             # Stage 1 (FM-side, easy pusher)
modal run mujoco_control/meta_value_shaping.py::meta_value_shaping --tag plan_v1 --plan      # Stage 2, easy pusher -> control-NEUTRAL (the null)
# capacity-competition test (hard pusher), 4 seeds -> the positive:
for s in "" 1 2 3; do modal run --detach mujoco_control/meta_value_shaping.py::meta_value_shaping \
    --tag plan_pfield${s:+_s$s}${s:+} --plan --field-pusher-amp 2.0 --plan-h 100 --seed ${s:-0}; done   # seed 0 tag = plan_pfield_v1
python3 mujoco_control/meta_value_shaping_seeds_figure.py                                    # multi-seed aggregate
```
Results/figures commit to `mujoco-control-data` under `/data/meta_value_shaping/<tag>/` and mirror to `figures/meta_value_shaping_<tag>/`. Use `--detach` for the full/plan runs (they exceed the ~2-min client window; the remote commits to the volume regardless of client connectivity).

### Figures
Per run (`figures/meta_value_shaping_<tag>/`): `fig1_capacity_frontier` (puck-drop + value-relevant frontier) · `fig2_perdim_bars` · **`fig3_fusion_2x2`** (pooling collapses for both objectives) · `fig4_z_probe` (system-ID) · **`fig5_planning`** (goal-dist vs capacity + veridicality-vs-plannability scatter). Multi-seed: **`figures/meta_value_shaping_seeds/fig_seeds`** (planning order parameter + FM-side frontier, 4 seeds — the headline).

### Next
- **Close the loop (the real disc-4):** ✅ **done — Cut #4e below** (`meta_value_learn.py`): a reward-*driven* outer loop (the per-dim FM weight chosen by control performance alone, no support hint) reconstructs the shaping, seed-robust and capacity-gated.
- **Back-translate to [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md):** sharpen the efferent-gain claim with the boundary condition — *value-shaping (capacity re-allocation) converts to a control benefit iff the value-irrelevant subsystem competes for capacity the value-relevant control needs*; veridicality⊥usefulness holds regardless.

---

## Cut #4e — Closing the reward loop: the value-shaping is CAUSED by reward (`meta_value_learn.py`)

**Date**: 2026-07-19. Closes the piece Cut #4d flagged as open ("'reward' in 4d is a value-*support*-weighted PREDICTION objective, NOT a reward-*driven* outer loop"). **Parent**: [Cut #4d](#cut-4d--value-shaping--meta-conditioning-two-separable-capacity-levers-and-when-re-allocation-buys-control-meta_value_shapingpy) `meta_value_shaping.py` (identical substrate). **Program**: [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) **discriminator 4** ("does an outer loop *cause* the shaping"). **Status**: disc-4 closed — both readouts seed-robust (3 seeds) and capacity-gated. Single family.

### The question and the one change

4d hand-set the per-dim loss weight from the goal-reaching value's support (match pusher dims `[0,1,4,5]`, drop puck dims `[2,3,6,7]`) and showed the re-allocation buys control *under capacity competition*. That is "value-shaping helps **if you impose it**." The causal question: does a **reward** signal, flowing back, *discover* that shaping on its own? The only change vs 4d: the per-dim weight `w` is no longer hand-set — it is chosen by an **outer loop whose sole signal is control performance** (−median CEM-MPC goal distance in the real env). Nothing tells the loop the puck is irrelevant. Substrate identical to 4d (φ-conflict family + context latent `z` + conditional FM `f(s,u,z)` + puck/pusher force field + CEM-MPC), fixed at the **h=64** sweet spot.

Two readouts, robust-core → dissectible-upgrade (the `value_shaping` discipline):
- **A1 — reward LANDSCAPE.** Sweep a scalar puck-weight `w_puck` (pusher pinned at 1), train the FM for each, eval control. Run on **both** regimes.
- **A2 — reward-DRIVEN outer loop.** A free per-dim weight `w∈ℝ⁸`, optimized by a derivative-free (CEM) outer loop driven *only* by control reward. Does it rediscover V's support with **no knowledge of the groups**? Weight = `8·softmax(θ)` → a **pure allocation** over a fixed budget.

### Results (3 seeds/regime; `meta_value_learn_seeds_figure.py` → `figures/meta_value_learn_seeds/fig_seeds.png`)

| order parameter | capacity competition (pusher field on) | easy pusher (matched null) |
|---|---|---|
| **A1** reward prefers the puck-drop (verid − puck-drop cost) | **+0.035 ± 0.007** (3/3 > 0) | +0.000 ± 0.003 |
| **A2** converged allocation corr→support | **+0.391 ± 0.044** (3/3 > 0) | +0.068 ± 0.140 |

Both fire **only under capacity competition** — the same `plan_v1↔pfield` boundary as 4d, now for a reward-*driven* loop. Under competition the outer loop drives the puck weight to ~0.13 and the pusher weight to ~1.6–1.9 (concentrating the fixed budget on the value-relevant dims), improving control by up to **+0.065** (uniform 0.255 → 0.190, seed 0) — matching *and exceeding* the hand-set 4d margin (it up-weights the pusher **beyond 1**, a stronger re-allocation than the binary support mask). On the easy pusher the loop wanders (no control gradient to chase). **So a reward-driven outer loop *causes* the value-shaping, robustly, and only where capacity competition makes it pay** — disc-4 closed on MuJoCo.

### Two methodological findings (reusable gotchas)

1. **The scale / grad-clip confound (a concrete wireheading instance).** An *unconstrained* per-dim weight lets the black-box loop win by lowering the overall loss scale — with `grad_clip=1.0`, Adam is not scale-invariant — instead of re-allocating. The first (unnormalized) run improved control with corr→support ≈ 0 (`outer_pfield` tag, kept as the demo). Normalizing to a fixed budget (`8·softmax`) removes the scale DOF and forces the value-relevant re-allocation (corr → +0.39). The fixed-budget normalization is the **"on-manifold veto"** the idea doc's wireheading control asks for — a reward loop games any cheap lever you leave open.
2. **Read the CEM's *converged distribution*, not its single-best candidate.** `w_star` = argmin cost over ~50 noisy planning evals is selection-biased (its weight vector is partly eval noise). Using it, A2 looked *fragile* (+0.20±0.20, one seed anti-support). The **converged distribution mean** (the standard CEM point estimate) is unbiased and shows A2 is **robust** (+0.39±0.04, 3/3). Seed 2 is the tell: single-best corr **−0.28** but converged-mean **+0.31** — the optimizer's central tendency *did* drop the puck; only the cherry-picked best was noise.

### Caveats
- **Not the fully-online loop.** This is a derivative-free outer *optimization over* the shaping (the FM is retrained per candidate) — a faithful realization of "a slow reward signal drives the shaping," but the genuinely-online single two-timescale loop (reward continuously shaping a live FM / backprop-through-planner) is a rung further.
- Single family (actuator-rotation + puck-field); 3 seeds; dense goal-distance reward (not sparse/delayed). The A1 pfield landscape is mildly non-monotone at mid-weights (planning noise); the 0-vs-1 endpoints carry the claim.

### Reproduce
```bash
cd experiments/
modal run mujoco_control/meta_value_learn.py::meta_value_learn --quick --outer            # smoke
# A1 dissociation + A2 outer loop, 3 seeds/regime (seed 0 tags have no suffix):
for s in 0 1 2; do
  modal run --detach mujoco_control/meta_value_learn.py::meta_value_learn --tag outer_pfield_norm${s:+_s$s} --field-pusher-amp 2.0 --outer --seed $s
  modal run --detach mujoco_control/meta_value_learn.py::meta_value_learn --tag outer_easy_norm${s:+_s$s} --outer --seed $s
done
python3 mujoco_control/meta_value_learn_seeds_figure.py                                    # the aggregate (headline)
```
Results/figures commit to `/data/meta_value_learn/<tag>/` and mirror to `figures/meta_value_learn_<tag>/`. Per-run: `fig1_reward_landscape` (A1) · `fig2_outer_loop` (A2 weight discovery). Aggregate: **`figures/meta_value_learn_seeds/fig_seeds.png`** (both order parameters, competition vs easy). The `outer_pfield` tag (unnormalized) is retained as the scale-confound demo.
