# Embodied collection — building the acquisition model, and pricing what it costs

**Up**: [../README.md](../README.md) (mjc) · **Standing memo**: [../COLLECTION_REALISM.md](../COLLECTION_REALISM.md) · **Idea doc**: [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md)
**Direct parent**: [../ballistic/directed/README.md](../ballistic/directed/README.md) — its S2 audit retracted a where-to-collect claim as *unmeasurable*, because collecting was free teleportation and measuring was free and global (a **22× subsidy**). The memo that audit produced is what this node implements.
**Code** (lives in this folder, per [STRUCTURE.md](../../../STRUCTURE.md)): `verify_backcompat.py` (the gate), `coverage_probe.py` (E0), `readapt_both_ways.py` (E1), `readapt_local.py` (E2), aggregators `*_agg.py`, `train.sh`; **E3 is its own sub-experiment** [`directed_on_policy/`](directed_on_policy/README.md) (**the prize**). Shared machinery at the node: [`../embodied.py`](../embodied.py); the flag + multi-region local fields on [`../arm_env.py`](../arm_env.py) / [`../pusher_env.py`](../pusher_env.py). File index: [FILES.md](FILES.md).
**Status**: machinery built and **backwards-compatibility gated**; E0/E1/E2/E3 each 3 seeds, clean. Single task family (arm, curl drift). **Date**: 2026-07-24.

---

## One-liner

Every FM in this node was trained on **teleported** transitions — isolated `(s,u,s′)` triples snapped to arbitrary states, free, i.i.d., omnisciently covering. This node builds the alternative (**data as a byproduct of behaviour**) as a flag on the existing envs, then prices it. Three findings, in the order they corrected each other: **(E0)** most of on-policy's apparent advantage was *not* embodiment but a **mistuned teleport knob** — the default samples a 12.7-rad/s, wide-posture region the task never enters, and an *oracle teleporter told where to look* recovers the whole gap; what survives as structurally embodied is **command-state entanglement** (|corr(u,s)| ≈ 0.7 vs teleport's 0.03), the luxury Cut #2's i.i.d.-command premise was quietly enjoying. **(E1)** Under the arc's **global** curl drift, Cut 4c-arm reproduces under every collection mode (**5.5× / 4.8× / 4.9×** ballistic-over-reactive), the recovery "step" is real and *sharper* on-manifold, and the memo's "on-policy will stretch it" hypothesis is **refuted** — a globally-learnable drift is re-adaptable from any motion. **(E2)** Make the drift **local** and embodiment finally becomes load-bearing: on-policy needs **2.5×** more transitions to repair the region than a task-matched teleporter, and broad teleport **never** repairs it. The mechanism is not the memo's guessed "coverage grows" but **bootstrap data-quality** — you visit the region at the same rate, but the data you gather while your model is wrong is itself wrong-distributed.

**Net for the substrate**: embodiment is nearly free when structure is global and expensive when it is local. That is the boundary condition on every acquisition claim this tree makes, and **local drift is the regime where "where should I practise?" becomes a real question** — the on-ramp back to the retracted directed-collection cut. **(E3)** That cut, re-attempted: the full inner-loop (reward-free FM re-adaptation + ballistic control) + outer-loop (a value signal `lprog × visits` choosing *where* to collect) + FM-as-bridge, on the on-policy arm — with the per-region learning-progress **survey itself on-policy and metered** (monitor:collect **1.84×**, not S2's 22×). Both halves of the retracted S1/S2 ladder reproduce: `value` beats `lprog-only` (**relevance** pays — 3/3 seeds on the sighted grader) *and* beats `error-only` (**reducibility**-awareness pays — `error-only` burns 47% of budget on the noisy-TV trap and collapses to uniform), and `value` matches the privileged oracle. The retracted claim is now measurable and positive, because collecting *and* looking are both embodied.

## The machinery ([`../embodied.py`](../embodied.py))

`collection_mode: "teleport" | "on_policy"` on `collect_pool`. **One plant, one set of knobs, not a second substrate.** What is enforced in code rather than asked of the experimenter:

- **`Body` does not expose `set_state`.** Teleporting is *unrepresentable* to an on-policy agent, not discouraged. This matters because the 22× subsidy did not happen through carelessness — it happened because `collect_region()` was callable for free.
- **Every `env.step` is charged** against a budget, whatever it was for (collection, probing, monitoring). The memo's §5 advice was *"print the ratio; above ~1 the experiment is subsidised"*; a meter you cannot bypass is strictly better than a ratio you must remember to print.
- **One `MjData` per parallel episode**, so trajectories are genuinely continuous and the pusher's contact warm-start is not silently dropped (the established `rollout()` idiom multiplexes B episodes through one `MjData` via `set_state`; exact for the contactless arm, not in general).
- Transitions carry **episode id + within-episode timestep**, so nothing downstream can treat them as i.i.d. by accident.

**The flag is two flags, deliberately.** The memo folds the behaviour policy into the `on_policy` enum value and then says, correctly, that the choice *"is the explore/exploit question in disguise"*. If it is that, it cannot be a hidden default. So acquisition and behaviour are separate axes and the behaviour rungs are enumerated so they can be **swept**:

| rung | behaviour | FM in the loop? | isolates |
|---|---|---|---|
| **B0** | teleport (unchanged) | no | the baseline |
| **B0m/B0r** | teleport, but sampling the **on-policy state distribution** (marginals / resampled) | no | *"an oracle teleporter told where to look"* — the control that decides what is really embodiment |
| **B1** | OU random torque | no | **continuity** alone |
| **B2** | reaches under a **frozen** reference planner | no (exogenous) | **on-task-ness**, decoupled from model quality |
| **B3** | reaches under the **live** FM | yes | the real thing, incl. **bootstrapping** |

B0-vs-B3 alone confounds four separable effects (where samples sit, whether they chain, whether commands are i.i.d., model↔data coupling). The ladder is what decomposes them.

> **Backwards compatibility is gated, not promised.** `verify_backcompat.py` checks the teleport path is **bit-identical** (`array_equal`, n=37 and n=500) to the pre-flag implementation *and leaves the RNG at the same position* — several scripts reuse the generator downstream, so that mattered. It also checks the plant/knob dict is unmutated, that `Body`'s physics matches the multiplexed idiom (8.1e-6, float32 round-tripping in the *old* path), that on-policy transitions chain while teleport ones do not, and — after E2's edit — that the **global** curl is unchanged and a far-gated curl is identically field-free. **Run it after any edit to `collect_pool` or `embodied.py`.**

---

## E0 — pricing the flag (`coverage_probe.py`, 3 seeds)

Six rungs at **matched environment-step budgets** over a sample-size sweep, everything else held fixed: normalisation stats, both probes, eval geometry, planner config, FM init, training steps. *The agent's acquisition changes; the experimenter's instruments do not.*

**At n=6000 (3 seeds):**

| rung | task err | broad err | \|corr(u,s)\| | speed (rad/s) |
|---|---|---|---|---|
| **B0** teleport | 0.304 ± 0.080 | **0.169 ± 0.021** | 0.029 | 12.7 |
| **B0m** oracle-marginals | **0.083 ± 0.009** | 1.86 ± 0.03 | 0.026 | 4.7 |
| **B0r** oracle-resampled | **0.087 ± 0.011** | 3.63 ± 0.10 | 0.025 | 4.7 |
| **B1** OU | 0.090 ± 0.010 | 2.94 ± 0.23 | 0.715 | 9.8 |
| **B2** frozen planner | **0.071 ± 0.004** | 3.75 ± 0.10 | 0.635 | 4.8 |
| **B3** live FM | 0.081 ± 0.025 | 3.22 ± 0.21 | 0.673 | 5.6 |

**1. The two probes move in opposite directions.** On-policy is **0.25–0.32×** teleport's error on the *task* probe and **18–23×** worse on the *broad* probe. On sample-efficiency: every on-policy rung beats teleport's **best over all budgets** using **400–1000** transitions against teleport's **6000**. Read naively this inverts the memo's Tier-B claim ("teleported counts are upper bounds on sample-efficiency; a body gets drastically less information").

**2. …but the controls say it is not embodiment.** `B0r` is *pure teleportation* — no continuity, i.i.d. commands, |corr| at the 0.025 floor — and it closes essentially the whole task-probe gap (0.087 vs on-policy's 0.071–0.090, vs default teleport's 0.304). **An oracle teleporter told where to look does as well as behaving.** So the honest statement is not "embodiment buys task accuracy" but **"our teleport sampling distribution is mistuned"**: `q_range=0.9, v_explore=8` spends a capacity-limited FM's budget on a wide-posture, 12.7-rad/s region the reach (4.8 rad/s) never enters. B0's task error even *stops improving* past n=1000 (0.267 → 0.304) while its broad error keeps halving. This is a **fixable knob**, and it retro-explains `arm_substrate` P5's non-monotone `fm_err(task)` note.

**3. What IS structurally embodied: command-state entanglement.** The oracle teleporters hold |corr(u,s)| at ~0.025; every on-policy rung sits at **0.64–0.72**. No teleporter reproduces it, because it comes from *how you got there* — velocity is the integral of past commands. Note **B1 (OU random torque, no goal, no planner) has the highest correlation (0.715)**, above goal-directed B2 (0.635): the entanglement is caused by **continuity**, not goal-directedness. Cut #2's arity result stays Tier-A safe, but its premise (max |corr| = 0.003 under i.i.d. commands) is a **teleport-only luxury**; a real body faces ~0.7.

**4. B3's bootstrap tax is real and priced.** At n=150 B3 is worst on everything (task 0.66, ballistic 0.229, speed 2.08 — it barely moves). By n=1000 it matches B2. ≈1000 transitions of tax. This is *why* B2 exists: it is the control that says whether you are looking at the phenomenon or at the spiral.

> **Not measurable here (stated rather than spun):** the ballistic-over-reactive transmission slope. Ballistic control is near-saturated across rungs (0.06–0.09 against a 0.07 reference), so there is no dynamic range; the per-rung slopes scatter 0.26–5.31 with no consistency across seeds. Grading transmission needs Cut 4b's dedicated staleness axis — which is E1.

## E1 — Cut 4c-arm, three collection modes, global drift (`readapt_both_ways.py`, 3 seeds)

The memo's named first experiment. Its Tier-B live hypothesis: 4c-arm's recovery is a **step** (96% by the first milestone) because *"omniscient i.i.d. sampling repairs the model everywhere at once… on-policy coverage grows as the agent explores, which would naturally stretch the step into a trajectory."*

**Two prerequisites this had to fix first.** (i) From `ballistic/arm/`'s own logs the entire step sits *inside* the first non-zero milestone (ballistic 0.373 → **0.086 at m=400** → 0.086 at m=14000), so at the old grid the question is unanswerable in either mode — hence the refined grid `0,50,100,200,400,…`. (ii) E0's knob finding means the teleport arm must not be the only teleport arm, or "embodiment wins" would just re-discover the mistuned default — hence **`teleport_matched`** (E0's `B0r` recipe: resample from a competent-reach reference pool), the honest middle baseline.

**Sanity**: ceiling ballistic **0.0993**, exactly `arm_substrate` P7's — the plant is byte-identical. All three arms grade the *same* stale FM at m=0 and produce identical numbers, so divergence is caused purely by collection.

**Ballistic recovery (fraction of stale→ceiling gap closed):**

| arm | m=50 | m=100 | m=200 | m=400 | first-milestone share |
|---|---|---|---|---|---|
| teleport (default) | 0.57 | 0.81 | 0.95 | 0.97 | **56% ± 16%** |
| teleport_matched | 1.00 | 0.99 | 1.05 | 1.11 | **87% ± 6%** |
| on_policy | 0.90 | 1.00 | 1.04 | 1.09 | **81% ± 10%** |

**1. The memo's hypothesis is refuted, backwards.** On-policy and task-matched teleport both step *harder* (recovering fully in ~50 transitions); the **only** arm that looks gradual is **default teleport** — and that is E0's mistuned knob showing up in a live cut. Its "curve" is not gradual *coverage*, it is gradual *arrival* at the right region.

**2. The diagnostic reason, which sets up E2.** The curl field is **smooth and global**: every moving transition is informative about `b`, so even a stale FM's deflected reaches teach it. On-manifold vs off-manifold coverage barely gates a *globally*-learnable drift — which is exactly why `arm_substrate` chose curl. Sharpened prediction: **on-policy should stretch only under a *local* drift.**

**3. The headline dissociation is robust to collection mode.** Ballistic-over-reactive recovery gain: **teleport 5.53× ± 0.27, teleport_matched 4.76× ± 0.39, on_policy 4.85× ± 0.41** (P7 reported 5.24×). Cut 4c-arm is **not** a teleportation artifact. This is the Tier-A reassurance the memo predicted would hold, now measured.

**4. The memo's intuition is not wrong — it hides in the mechanism metric.** On-policy's *FM task-probe error* climbs gradually and monotonically (28% at m=50 → full by ~m=400) while its ballistic *control* already stepped to ceiling by m=50. That is [`drift_value_loop`](../drift_value_loop/README.md) Cut 3 reproduced: **control is a near-blind grader** — it saturates once the FM is good enough on the reach and cannot see continued FM improvement. (Default teleport's task error even goes *non-monotone negative* early — the P5 velocity-tail effect, another face of the same knob.)

## E2 — the local drift (`readapt_local.py`, 3 seeds)

E1's prediction, tested. New env support: **spatially-gated curl** — `curl_field["center"]/["sigma"]` Gaussian-gates the gain on the *tip's* Cartesian position, so the field acts only where the hand is. It keeps every property the curl was chosen for (smooth, velocity-dependent, open-loop **compensable**, aftereffect-capable) and adds locality. With no `center` the gate is identically 1.0 and the global path is byte-identical (gated).

Region placed by computing the reach-tip distribution directly (`fk` is pure numpy): **32% ± 4%** of matched-reach transitions land inside — genuinely local, with a real out-region control. Eval reaches are random-direction (no fixed corridor), per the memo's 4a warning. **The ceiling here is task-matched, not broad-teleport** — under a local field a broad-teleport FM is doubly mistuned in-region (rare samples, wrong velocities) and scores *worse* in-region than a task-matched FM, which sign-flips the recovery normalisation. E1's broad-teleport ceiling was correct *there* because the field was global; that is the E1↔E2 difference, not an inconsistency.

**In-region FM error — transitions to durable 0.8 recovery:**

| arm | transitions | |
|---|---|---|
| **teleport_matched** | **533 ± 133** | steps |
| **on_policy** | **1333 ± 267** | **2.5× stretch** |
| **broad teleport** | never (3/3 seeds censored at 14000) | fails outright |

**1. The stretch is confirmed.** Making the drift local turns E1's null into a **2.5×** penalty for earning your own data. On-policy's in-region recovery is even **negative for the first ~200 transitions** — early reaches through the region, planned with a stale model, actively make in-region prediction *worse* before helping.

**2. The mechanism is bootstrap data-quality, not coverage growth.** The in-region **collection share is flat (~30%) for every arm including on-policy** — it visits the region at the same rate as the teleporters. What differs is *quality*: reaches through the region under a stale FM get **deflected** by the unmodeled field, so the in-region data is off the competent-reach manifold the probe measures. The memo guessed *"coverage grows as the agent explores"*; the substrate says **"the data you gather while your model is wrong is itself wrong-distributed."** Same model↔data-coupling family, sharper and different mechanism — and it is the one the memo names as the honest cost of §3 ("a stale model moves badly, which gets you worse data, which keeps it stale").

**3. Broad teleport fails completely** — never reaches 0.8 in-region recovery in any seed, and its out-region recovery is censored too. Under a local field the E0 knob is not a 3× inefficiency, it is a **qualitative failure**: wide/fast samples in the region are at velocities the task never uses.

**4. The 5× dissociation survives again** (≈5.4/5.1/5.4× at seed 0), across both drift geometries and all three collection modes.

## E3 — the prize: directed collection on the on-policy arm ([`directed_on_policy/`](directed_on_policy/README.md), 3 seeds)

**Full dedicated writeup: [`directed_on_policy/README.md`](directed_on_policy/README.md).** Next-step #1, delivered. The retracted [`ballistic/directed`](../ballistic/directed/README.md) S2 cut — the full **inner loop** (reward-free FM re-adaptation + ballistic control) + **outer loop** (a value signal `lprog × visits` choosing *where* to collect) + **FM as the shared bridge** — re-attempted where the fatal S2 subsidy is gone: the per-region learning-progress **survey is itself on-policy and metered** (monitor:collect **1.84×**, not 22×). Scarcity-among-distractors ladder: 1 on-reach reducible target A + off-reach reducible (catch `lprog-only`) + off-reach noise (catch `error-only`), under continuous OU drift. Mean region-A FM error over rounds (sighted grader; control is near-saturated):

| | uniform | error-only | lprog-only | visits-only | oracle | **value** |
|---|---|---|---|---|---|---|
| region-A err ↓ | 0.475 | 0.457 | 0.447 | 0.390 | 0.372 | **0.350** |
| budget → noise | 33% | **47%** | 35% | 13% | 0% | 29% |

Both halves of the ladder reproduce: `value` beats `lprog-only` **3/3 seeds** (**relevance** pays, once looking is embodied) and beats `error-only`, which burns 47% of budget on the noisy-TV trap and collapses to uniform (**reducibility**-awareness pays); `value` matches the privileged oracle. The one on-policy twist: the S1/S2 *on-reach noise* decoy is **unplaceable** — visitation concentrates on the target, so there is no "visited-but-irrelevant" territory (you only go where you reach). See the writeup for findings, caveats, and the geometry gotchas.

---

## What the arc says about the substrate

1. **Physics realism and experience realism are independent axes, and now both are movable.** `arm_substrate` moved the plant; this node moves the acquisition model, on the same plant, behind one flag.
2. **Embodiment's cost is conditional on the *structure* of what must be learned.** Global structure → embodiment is nearly free (E1). Local structure → embodiment costs 2.5×, and naive teleport fails outright (E2). Any acquisition claim must state which regime it is in.
3. **Teleport was mistuned, and that masqueraded as a scientific effect twice** — as on-policy's apparent sample-efficiency win (E0) and as default teleport's apparent gradual recovery (E1). The fix (`teleport_matched`) is cheap and belongs in any future teleport cut.
4. **Cut #2's i.i.d.-command premise is a teleport-only luxury.** A body faces |corr(u,s)| ≈ 0.7, caused by continuity itself.
5. **Local drift is the regime where "where should I practise?" is a real question** — allocation becomes zero-sum and place-dependent. That is precisely what [`ballistic/directed/`](../ballistic/directed/README.md) S2 could not have, and it is the on-ramp to re-attempting that retracted cut honestly.
6. **The retracted directed-collection claim is true, once looking is embodied (E3).** Relevance pays *because* on-policy you can only cheaply survey where you actually go — the 22× measurement subsidy that made S2 unmeasurable is not a config to remember to disable but a thing the body structurally cannot buy. And **on-policy dissolves the on-reach-noise decoy entirely**: visitation concentrates on the target, so "visited-but-irrelevant" territory does not exist — the relevance lever is purely the off-reach direction.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/on_policy/verify_backcompat.py::verify            # the gate — run FIRST
modal run mjc/on_policy/coverage_probe.py::coverage_probe --quick   # smoke

# E0 / E1 / E2 — launch each seed as its OWN client (see gotcha below)
for s in 0 1 2; do modal run --detach mjc/on_policy/coverage_probe.py::coverage_probe --tag e0_s$s --seed $s; done
for s in 0 1 2; do modal run --detach mjc/on_policy/coverage_probe.py::coverage_probe --tag e0b_s$s --seed $s --rungs "B0,B0m,B0r"; done
for s in 0 1 2; do modal run --detach mjc/on_policy/readapt_both_ways.py::readapt_both_ways --tag rbw_s$s --seed $s; done
for s in 0 1 2; do modal run --detach mjc/on_policy/readapt_local.py::readapt_local --tag loc_s$s --seed $s; done

# E3 — directed collection, the prize (own sub-experiment). --quick smokes the noisy-TV ladder.
modal run --detach mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --quick --tag ladder_smoke
for s in 0 1 2; do modal run --detach mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --tag ladder_s$s --seed $s --mon-n 40; done

python3 mjc/on_policy/coverage_probe_agg.py --tags e0_s0 e0_s1 e0_s2 e0b_s0 e0b_s1 e0b_s2
python3 mjc/on_policy/readapt_both_ways_agg.py --tags rbw_s0 rbw_s1 rbw_s2
python3 mjc/on_policy/readapt_local_agg.py --tags loc_s0 loc_s1 loc_s2
python3 mjc/on_policy/directed_on_policy/directed_on_policy_agg.py --tags ladder_s0 ladder_s1 ladder_s2
```

**Gotchas.**
- **Do not launch the 3 seeds from one shell** with backgrounded `modal run --detach … &` + `wait` (`train.sh`'s multi-seed targets do this, and it cost E2 an entire run). Detached mode only guarantees the *last* triggered function survives the parent; the siblings evict each other, and a client killed before `volume.commit()` loses `results.json` entirely. Launch each seed as its own independent client. Same fragility is recorded in [`../ballistic/directed/README.md`](../ballistic/directed/README.md).
- If a client dies anyway, `modal volume get mujoco-control-data <path>/results.json …` recovers a clean artifact (this rescued `e0b_s1`); `coverage_probe_agg.py` also parses logs, and merges tags by trailing `_s<N>` so a follow-up run that *adds rungs* to an existing seed folds in — with an explicit **merge check** that shared configurations agree to 0.0.
- `modal volume get` can fail with a transient DNS error and leave a **0-byte** file; check the size and retry.
- **E3 geometry (two smokes' worth of lessons, now in code).** (i) On/off-reach must be defined by **actual reach visitation**, not a geometric proxy: the reach tip-path is widely curved, so a straight-corridor (or angle-from-P0) test mislabels swung-through regions as off-reach and, worse, put an "off-reach" distractor ~0.1 m from the start posture — which *every* reach passes through — sending `value`/`oracle` to chase it instead of A. The fix computes gate-occupancy over the FK'd joint-space reach sweep. (ii) Regions must sit in the **extended/tame** radius band; a folded/inward region's fast dynamics give ~10× the FM error and swamp the curl signal. (iii) Noise `amp` must be **< `gear`** and noise regions well-separated, or aleatoric noise bleeds into A and destroys its reducibility (an early smoke's A ceiling jumped 0.14→0.37).

## Caveats

- **Single task family** (planar arm, curl drift, reaching). E0's knob and entanglement findings are substrate-general arguments; the 2.5× stretch is one geometry, one drift, 3 seeds.
- **Anything measured at B3/`on_policy` is compound, not clean** — model quality and data quality are coupled by construction. Tier-A dissociations stay on teleport and were *not* retrofitted; that is the trade the flag exists to make explicit.
- **`teleport_matched` is privileged by construction** (it reads a pool the agent had to behave to obtain), like `directed_loop`'s `oracle`. That is the point — it is the strongest possible teleport — but it is not a baseline an unaided teleporter could implement.
- **E2's region placement was tuned** (two iterations) to land at ~32% in-region; the effect's dependence on region size/placement is unmeasured.
- **The E2 stretch metric is "transitions to durable 0.8 recovery."** The earlier "first-milestone share" is meaningless across the sign change on-policy's early dip produces, and was replaced rather than reported.
- **Transmission slope is not measurable in E0** (saturation, above), and E1/E2's recovery *gains* are the reliable ballistic readouts.

## Next steps

1. ~~**Re-attempt the retracted directed-collection cut on the local-drift substrate** — the prize.~~ **Done (E3).** Allocation is now zero-sum, place-dependent, and metered; the monitor survey is charged to the same `Body` budget (ratio 1.84×, not 22×); the relevance *and* reducibility halves of the ladder both reproduce. Remaining rungs it opens: (a) the **fully-online learned-value head** (backpropped control performance driving a *live* FM, vs E3's reward-free computed `lprog×visits` over per-round refits — idea-doc discriminator #4's last rung); (b) **sweep region count / drift speed** — E3 is one geometry, and no policy fully repairs A under the current OU rate; a slower drift should let `value` reach the ceiling and sharpen the control (currently blind-grader) readout. **(c) Add a representation readout — E3 grades an expansion drive with compression-only metrics.** `lprog × visits` selects where to *open* new structure, but region-A FM error, ballistic distance and budget-share all ask *how well does the current model predict*; a model that expanded its basis and one that re-fit within it return the same number. That is E3's own blind-grader lesson one rung up (it knew control was blind and moved to FM error; FM error is blind to dimensionality). **The diagnosis stands; the fix named here previously does not.** This rung used to specify the `(R_act, R_comp, R_res)` triple, with the prediction that healthy expansion reads as `R_act ↑` then `R_comp ↑` while the noisy-TV leak reads as `R_act ↑` with `R_comp` flat. **Both withdrawn 2026-07-26** — `R_comp` is not independent of `R_act` (a good FM reproduces the activations it predicts), so the noisy-TV signature cannot occur whether or not expansion does. Do not reinstate the triple; the replacement readout is being established on the a2a/RHM substrate, in a writeup not yet in the tree, and [`../README.md`](../README.md) §Next steps #3 carries the current status plus a **gap-width calibration that gates this whole rung** — a one-step arm FM is the narrowest possible gap, which may be the regime where residual geometry is inert. Two structural gaps this rung must close regardless, both inherited and both unaffected by the retraction: E3's OU walk moves the gains of a **fixed** region set, so its *support* is stationary and nothing ever demands a new direction (rung (b)'s region count is the support-growth knob, repurposed); and the 2-link arm leaves **capacity slack**, so E3 makes *collection* zero-sum while representational capacity never binds (contrast `meta_adapt` #4d/#4e, where the value-shaping benefit appeared only under capacity competition). Requires a re-run — E3 persists only JSON scalars (`directed_on_policy.py:547`, `:762`); `render_pack.json` holds rollouts and geometry, not FM weights.
2. **Test whether ensemble disagreement recovers its job.** S1 found disagreement **cannot detect a drift** — structurally, because every member had data everywhere, so the problem was staleness, never missing data. On-policy kills that premise: there *is* genuinely unvisited territory (E3's off-reach regions have visitation ~0). Predicted split: disagreement flags where you *lack* data, learning-progress flags where data went *stale*. This would turn S1's "the doc calls two problems one" into a measured separation. **Second axis on the same cut** ([`ideas/heterogeneous_graders.md`](../../../ideas/heterogeneous_graders.md) §9): S1's members were *homogeneous* (same objective, different seeds), and structural blindness is shared by every such member — so disagreement between **differently-typed graders** (control vs value-relevant FM error, which already disagree at exactly control's blind spot in E1/E3) is a different instrument that should not inherit the blindness. Make grader disagreement itself the allocation signal, with the seed-ensemble as the honest baseline, under E2's **local** drift, keeping the noisy-TV control (noise also produces disagreement; the discriminator is that noise-disagreement does not *close* when you collect there). Both graders are already computed per round, so this is cheap.
3. **Retune the teleport default for new teleport cuts** (`teleport_matched`-style sampling), and re-check whether `arm_substrate` P5's non-monotone `fm_err(task)` disappears.
4. **Sweep `sigma_u`** to separate command-channel identifiability from state coverage as the price of on-policy data.
5. **Aftereffect + directional generalization under on-policy collection** — the arm's second built-for readout, still unmeasured (`arm_substrate` §Caveats).
