# Curiosity → control: drive selection, a substrate dead-end, and the MuJoCo redirection

**Idea doc**: [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) (the value/outer-loop thesis; disc-4 = "does an outer loop *cause* value-shaping"; the merged Step 1+2)
**Parent**: [CURIOSITY_DRIVE_README.md](CURIOSITY_DRIVE_README.md) (the drive atom on active vision — Phases 1/2/2b)
**Cousin (the redirection target)**: [../../mjc/dynamics_shift/README.md](../../mjc/dynamics_shift/README.md) Cut #3 `dynamics_shift.py` — reward-free re-adaptation after a dynamics shift
**Code**: `curiosity_reaching_control.py` (the control-substrate attempt; kept as the diagnostic record)
**Status**: **Step 0 (drive selection) — clean positive.** **v1 (curiosity on the reaching control substrate) — instructive substrate-negative + full diagnosis.** Redirected to MuJoCo for the real next step. Single seed.
**Date**: 2026-07-18

---

## What this session was

The parent curiosity line built the **drive atom** on the active-vision *teleport toy*, where acting == planning (the world model just predicts glimpse pixels), so it **cannot** measure the two things the idea doc actually wants: (disc-4) does an outer loop *cause* the inner FM to become value-shaped, and (compounding) does re-adaptation accelerate under non-stationarity. Those need a **control** task (acting ≠ planning). This session: (0) pick which drive to put in the loop, (1) try to run the merged Step 1+2 on the reaching controller, and — after that substrate fought us — (2) reconcile with the MuJoCo line and pick the real next step.

The net: **one clean result (Step 0), one well-understood dead-end (the reaching-ViT substrate), and a sharpened plan (disc-4 on MuJoCo).**

---

## Step 0 — which drive goes in the loop (clean positive)

Before building a loop, settle the Phase-2 wound: the LP **derivative** `−d‖e‖/dt` is drift-fragile (Phase 2 showed it *worse than random* under abrupt drift — blind to a re-opened frontier). The idea doc conjectured a **magnitude** signal — cross-model **disagreement** — would avoid the re-opening blindness. Test: re-run `curiosity_scarcity.py` (Phase 2b) with `--drift-mode swap` (abrupt) vs `morph` (gradual), full horizon (1000 iters, scarcity sweep), comparing `lp` vs `disagree`.

**Unbiased needle-error (lower = better tracking):**

| scarcity | swap `lp` | swap `disagree` | morph `lp` | morph `disagree` |
|---|---|---|---|---|
| 19% | 0.115 | **0.069** | 0.051 | **0.034** |
| 10% | 0.103 | **0.071** | 0.059 | **0.046** |
| 5%  | 0.165 | **0.143** | 0.141 | **0.129** |

- **`disagree` (reducible-disagreement magnitude) wins needle-error at every scarcity under both drift modes** and is the drive for the loop.
- **It is the only drive robust to abrupt drift**: needle-occupancy under `swap` is actually *higher* than under `morph` (0.75/0.70/0.44 vs 0.71/0.63/0.43) — a fresh swap makes the whole needle unlearned → maximal cross-model disagreement → immediate pull. No derivative, no re-opening lag.
- **The LP derivative confirms the Phase-2b nuance honestly**: over 1000 iters it *finds* the needle even under swap (occ 0.42/0.44/0.28, well above uniform — "not qualitatively broken") but *tracks* poorly (error 0.115 vs 0.069 at 19%) — it re-finds but lags each jump.
- One reproduced wrinkle: `minsurprise` ties `disagree` at 19%-morph (the scarcity-dependent dark-room-overflow confound flagged in Phase 2b) but collapses under `swap` and at 5%. `disagree` is the only drive best-or-tied across *all* conditions.

**Verdict: magnitude, not derivative.** Reproduce: `curiosity_scarcity.py --drift-mode swap --n-iters 1000 --needle-cols-sweep "4,2,1"` (and `morph`).

---

## v1 — curiosity on the reaching control substrate (substrate-negative + diagnosis)

**Plan (merged Step 1+2, approved):** on the reaching controller, replace the hand-coded goal-distance value with the disagreement drive; drift the **dynamics** (not the content — content is answer-irrelevant, the ACTIVE_VISION null); measure FM re-shaping (disc-4) + re-adaptation under drift jointly. Non-stationarity is a *precondition*, not a payoff: a stationary LP loop provably exhausts (Phase 1).

**The dynamics-drift design — visuomotor rotation.** The existing efference marker sits at the *true* next cell, so drifting the map would be invisible to the FM. Fix by splitting the map: **canonical** (fixed) = what the efference encodes (the *intended* move) vs **env** (drifts) = the *realized* move (a permutation of the 4 cardinal controls, redrawn each `drift_period`). The FM must learn `canonical → realized`; a drift re-scrambles it. This is the **prism-adaptation / visuomotor-rotation** paradigm. `slip` = a noisy-TV action (random realized cell). Open arena → scarcity for free (greedy reaching only exercises ~2 goalward actions). Frozen, briefly-pretrained controller so shaping isolates the *FM*, not the rep. `curiosity_reaching_control.py` implements all of this.

**The debugging arc (four bugs, recorded so we don't repeat them):**
1. **Device**: `idx` on GPU indexing a CPU image tensor. (Trivial.)
2. **Efference too weak** (`std 0.02` vs an O(1) state) → the FM can't tell which action was commanded → predicts the command-*averaged* next state = **arity-1 collapse**. Fix: an `eff_scale` knob, strong efference.
3. **Position-illegible frozen operator** → `probe_acc = 0.51–0.76`, throttling the planner, the drive, *and* the `pos_acc` metric (all decoded through the probe). Fix: a **position-legibility aux** during pretrain (make the frozen rep decodable — a fixed substrate property, shared across arms) → `probe_acc = 0.998`.
4. **The marker-copy trap** (the load-bearing one — see below).

**The core finding — the reaching-ViT FM *copies a cue*, it does not *learn* where-you-land dynamics.** With the (strong, legible) setup, the FM learns the **identity** map beautifully (`pos_acc 0.35→0.93` on seg 0) — but on a **permuted** segment it stalls at `pos_acc ≈ 0.31`, which is exactly *P(intended == realized)* under a random permutation of 4 directions. It is *copying the intended-cell marker*, not learning the realized remap. Remove the cell marker (action-identity efference only) and the FM can't learn even the **identity** map (`pos_acc ≈ 0.30`), and **more capacity does not help** (a 265K-param 2-layer/4-head FM is just as stuck). So:

> The reaching-ViT encodes position as *which of 49 patch tokens holds the fovea marker*. Predicting "where the action lands" is therefore a discrete, combinatorial **token-reassignment** over a grid — which a small transformer can't learn from an action command, and which the parent's "arity-2 FM" only ever finessed by *copying an oracle cue placed at the true next cell*. (It genuinely learned the *content* transition; its *where-you-land* prediction was **cued, not learned** — drift is what exposed that.) The substrate cannot give us an FM that **learns *and re-learns*** a control map online — which is precisely what the experiment needs.

**This failure is incidental, not fundamental.** It is a property of *representing position as a discrete fovea-on-a-token-grid*, **not** of "reaching" or "control." A reaching task with a continuous/low-dim position + an MLP FM — which is essentially what MuJoCo pusher-reaching *is* — dissolves it entirely. So would a plain gridworld.

---

## The MuJoCo reconciliation — "can it re-adapt at all" is already done, and better

Reading [../../mjc/dynamics_shift/README.md](../../mjc/dynamics_shift/README.md) Cut #3 (`mjc/dynamics_shift/dynamics_shift.py`) reframed the whole line. It is the **reward-free re-adaptation after a dynamics shift** result, on real continuous physics: an arity-2 FM `f(s,u)→Δs` that **provably learns *and re-learns*** the dynamics (recovers to the oracle ceiling from **~50 reward-free transitions**), a model-free policy that gets **no** signal from reward-free interaction and needs **~240×** more reward-*labeled* data, the **factorization** (a dynamics shift corrupts only the world-model factor), and a cerebellar-recalibration reading. The irony: **the exact thing the reaching-ViT could not do — an FM that learns and re-learns a control map online — MuJoCo already has working.**

So re-deriving "an MB agent re-adapts after a shift" on *any* toy substrate (reaching-ViT or gridworld) would just reproduce Cut #3 on worse footing. **Re-adaptation feasibility is not the open question.**

**What is still ours — the increments Cut #3 set up but did not take:**
1. **The drive** — Cut #3 collected its reward-free transitions by *undirected* interaction. Does **disagreement-directed** exploration re-adapt in *fewer* transitions? (The whole reason Step 0 chose the disagreement magnitude — the value of *directing* allocation.)
2. **Compounding** — does re-adaptation *accelerate* over successive shifts? (The meta layer — the thing stationary meta-RL can't manufacture, [RHM_META_LEARNING](../../rhm/ratchet/RHM_META_LEARNING_README.md).)
3. **disc-4** — does the outer loop *cause* the FM to become **value-shaped**? (The linchpin, still un-done anywhere.)

**Is MuJoCo *fundamentally* different, or just convenient?** Honest answer, split by target:
- For **drive-directed re-adaptation / compounding**: **not fundamental.** A clean gridworld would test it just as well. MuJoCo's edge is practical (already debugged, real physics, the lines were converging — its own next-steps flag "drifting-dynamics" as Cut #4/5).
- For **disc-4**: **genuinely fundamental.** Value-shaping = the FM *re-allocating limited capacity* toward value-relevant directions, which only has teeth under real **capacity pressure** (the dynamics must be complex enough that the model must *choose* what to get right). A toy has none — the FM fits the whole tiny state, nothing to re-allocate. MuJoCo supplies both ingredients a toy can't: **capacity pressure + a clean value-irrelevant split** (stiff contact the FM can't fully fit, plus the **puck** as a value-irrelevant distractor for pusher-reaching), and a **natural reducible/irreducible split** (Cut #1: free-flight reducible at cos 0.99, contact partly *irreducible* at cos 0.82 — the noisy-TV discriminator *physically present*, not an engineered `slip`).

The reaching detour wasn't wasted: it is *why* we can now say precisely where a toy substrate stops being enough.

---

## What this establishes / caveats

- **Establishes**: (1) the outer-loop drive should be the **reducible-disagreement magnitude**, not the LP derivative — drift-robust at every scarcity, both drift modes (Step 0, clean). (2) The reaching-ViT substrate **cannot** host an FM that learns *and re-learns* control dynamics online — its arity-2 FM was *cue-copying*, and this is an **incidental** representational limit (discrete fovea-on-token-grid), not a fact about reaching/control. (3) "Can an MB agent re-adapt reward-free after a shift" is **already done** on MuJoCo (Cut #3); the un-done work is the **drive**, **compounding**, and **disc-4**.
- **Caveats**: single seed throughout; `curiosity_reaching_control.py` is retained as a **diagnostic record**, not a positive result — its numbers (probe_acc, the 0.31 permuted-stall) are the evidence for the substrate verdict, nothing more.
- **Does not show**: any positive value-loop result on control yet — that is the next step, on MuJoCo.

---

## Concrete next step — disc-4 on the MuJoCo pusher

Target the **linchpin** (disc-4) on the one substrate where the choice is principled rather than incidental. On the MuJoCo pusher (goal-conditioned reaching, `mjc/`):

- **Value-irrelevant split**: keep the **puck** in the scene as an answer-irrelevant distractor for a *pusher*-reaching task. The FM `f(s,u)→Δs` predicts the full state (pusher + puck); the value/planner only cares about pusher position.
- **The disc-4 test**: does an outer loop driven by the **disagreement magnitude** (Step 0's drive) + goal-value cause the FM to **re-allocate capacity toward the pusher (value-relevant) dynamics and away from the puck (value-irrelevant)** — versus a raw-`Δs`-MSE FM that spends capacity on both? Measure per-dim FM fidelity (pusher-vel vs puck-vel, the Cut #2 idiom) under the value-shaped vs unshaped FM.
- **Under non-stationarity (the merged claim)**: apply Cut #3's dynamics shift (friction/drag collapse). Does the value-shaped FM **re-orient** its capacity to the newly-reducible pusher directions faster than the unshaped one — and does the disagreement-directed collection re-adapt in fewer reward-free transitions than Cut #3's undirected baseline?
- **Controls inherited physically**: noisy-TV = contact aliasing (Cut #1's irreducible component); the drive must chase free-flight-reducible structure, not contact-aliasing-irreducible. Wireheading deferred (the reaching manifold is the reachable state — low risk, per next_steps).

This turns the MuJoCo `dynamics_shift` flagship from "MB re-adapts reward-free" into "the **value/outer loop shapes and re-orients** the world model" — the disc-4 result the whole two-timescale doc is built around, on a substrate with the complexity to make it real. Spec/build lives in `mjc/`, not here.

---

## Biological grounding + design guardrails for the MuJoCo implementer (2026-07-18)

Added after a design discussion (see [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) and [beliefs/trees/cerebellum_and_cognitive_architecture.md](../../../beliefs/trees/cerebellum_and_cognitive_architecture.md)). Read before building — it both *justifies* the bare-state substrate and pins down two things that would otherwise confound disc-4.

**The bare-state physics FM is not a betrayal of the cerebellar story — it's the ancient cerebellum done faithfully.** The biological cerebellum forward-models *both* raw body/physical state (ancient vestibulo-/spinocerebellum, proprioceptive + vestibular input — the MuJoCo `f(s,u)→Δs` case) *and* cortical activation dynamics (new cerebrocerebellum, cortico-ponto-cerebellar input — the a2a activation case), using the *same* microcircuit; the codomain is set by afferent wiring, not by a different computation. So MuJoCo's raw-physics-state FM and the a2a activation FM are faithful to two physically distinct cerebellar territories running one algorithm — the physics-state choice is principled, not a downgrade.

**Guardrail 1 (load-bearing) — the value-irrelevant distractor must be HIGH-ENERGY.** disc-4 = "the value loop re-allocates FM capacity toward the value-relevant (pusher) dims and away from the value-irrelevant (puck) dims." This is clean only when the value-irrelevant part is *causally separable* **and** *costly to predict*. Cut #2's caveat is the trap: under non-seeking collection the puck is "rarely touched → tiny near-noise Δv," so a capacity-limited FM ignores it *for free* — you then cannot distinguish "the value loop dropped the puck" from "the puck was always negligible." This is the exact analog of curiosity Phase 2b's full-amplitude-needle fix. **Requirement: a frequently-contacted / high-energy puck** (seek-gain on during collection, or a task that routinely engages the puck), so predicting it costs real capacity and dropping it is a genuine re-allocation.

**Guardrail 2 — don't overclaim the port to language.** A positive MuJoCo disc-4 establishes the *separable-factor* form of value-shaping (drop a causally-separable object). Hierarchical language/RHM value-irrelevance is *entangled* — deep composes from shallow, and specialization Exp 2 showed starving the shallow substrate *destroys* depth ([../../rhm/specialization/README.md](../../rhm/specialization/README.md)) — so the operation does **not** port directly. Report the result as separable-factor value-shaping; the entangled/hierarchical version is a distinct, harder claim.

**If you later want the a2a self-model (activation) version on MuJoCo:** wrap a controller and forward-model *its* activations (`f(h,u)→h'`). Know the tradeoff first — this reinherits the high-dim distributed manifold that killed the reaching-ViT and *inverts* the clean separability (pusher/puck info mix into shared hidden units; a goal-trained controller may have *already* dropped the puck, so any "shaping" is inherited, not caused). The biological resolution is *modular* (microzones) — i.e. an object-factored / slot controller (pusher-slot, puck-slot), not a single distributed net. So: **bare-state disc-4 first**; a factored-controller activation-FM only as a deliberate later step (and the honest bridge toward language).

---

## Files & reproduction

| File | Role |
|---|---|
| `curiosity_reaching_control.py` | v1 control-substrate attempt (diagnostic record): visuomotor-remap drift, frozen legible controller, online K-FM disagreement drive, four arms (`extrinsic`/`eps`/`two_tap`/`explore`), `slip` noisy-TV. Documents the reaching-ViT cue-copy limit; **not** a positive result. |

```bash
cd experiments/
# Step 0 — drive selection (reuses Phase-2b script): disagreement magnitude vs LP derivative, abrupt vs gradual
modal run --detach a2a_forward/reaching/curiosity_scarcity.py::curiosity_scarcity --dataset mnist --drift-mode swap  --n-iters 1000 --needle-cols-sweep "4,2,1"
modal run --detach a2a_forward/reaching/curiosity_scarcity.py::curiosity_scarcity --dataset mnist --drift-mode morph --n-iters 1000 --needle-cols-sweep "4,2,1"

# v1 substrate diagnostic (reaching-ViT dead-end; kept for the record)
modal run a2a_forward/reaching/curiosity_reaching_control.py::curiosity_reaching_control --quick
```

Results on the `language-reduction-data` volume under `/data/a2a_forward/{curiosity_scarcity, curiosity_reaching_control}/…`.
