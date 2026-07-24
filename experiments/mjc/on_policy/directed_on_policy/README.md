# E3 — directed collection on the on-policy arm (the retracted S2 cut, delivered)

**Up**: [../README.md](../README.md) (on_policy) · **Node**: [../../README.md](../../README.md) (mjc) · **Memo**: [../../COLLECTION_REALISM.md](../../COLLECTION_REALISM.md) · **Idea doc**: [../../../../ideas/two_timescale_value_loop.md](../../../../ideas/two_timescale_value_loop.md)
**Direct parent**: [../../ballistic/directed/README.md](../../ballistic/directed/README.md) — its S2 audit **retracted** the where-to-collect claim as *unmeasurable*: collecting was free teleportation and measuring was free and global (a **22× subsidy**). This node re-attempts it once looking is embodied.
**Code** (this folder): `directed_on_policy.py` (the loop), `directed_on_policy_agg.py` (cross-seed aggregator). Shared machinery at the node: [`../../embodied.py`](../../embodied.py) (`Body`, no `set_state`), [`../../arm_env.py`](../../arm_env.py) (`curl_fields`/`noise_fields` multi-region local drift). Files: [FILES.md](FILES.md).
**Status**: full loop, 3 seeds, clean. Single task family (planar arm, curl drift, reaching). **Date**: 2026-07-24.

---

## One-liner

The full **inner loop** (reward-free FM re-adaptation + ballistic control) + **outer loop** (a value signal `lprog × visits` choosing *where* to collect) + **FM as the shared bridge**, re-attempted on the on-policy arm — with the per-region learning-progress **survey itself on-policy and metered** (monitor:collect **1.84×**, not S2's 22×). Both halves of the retracted S1/S2 ladder reproduce: `value` beats `lprog-only` (**relevance** pays — 3/3 seeds on the sighted grader) and beats `error-only` (**reducibility**-awareness pays — `error-only` burns 47% of its budget on the noisy-TV trap and collapses to uniform), and `value` matches the privileged oracle. The retracted claim is measurable and positive **because collecting *and* looking are both embodied**.

## Why S2 could not answer it, and what changed

S2 built exactly this loop on the pusher under teleport collection, and the relevance claim came out as an unmeasurable null. The reason was structural: every round each policy got ~2,240 **free** teleported probe/monitor transitions sampled everywhere, to decide where to spend a budget of 100 — a **22× measurement subsidy**. When you can look everywhere for free, "where should I even look?" has no job.

On-policy collection makes both collecting *and* monitoring embodied and metered: to gather or even *check* a region the body must physically reach there, paying steps from the same [`Body`](../../embodied.py) budget. Relevance now sits **upstream** of measurement instead of multiplying it afterwards, exactly as the memo argued.

## Setup — scarcity among distractors, the on-policy 2×2

On this arm the curl is easy to learn (~a round's data repairs a region), so with one target the budget never binds and allocation is not a lever. S1 / curiosity Phase 2b: the value signal beats uniform only when the reducible frontier is **small among many distractors**. So:

- one **on-reach reducible target A** — the genuinely most-visited extended tip point, where control depends on the model;
- **3 off-reach reducible** distractors — learnable but behaviourally irrelevant; catch `lprog-only`, which chases reducible structure without asking whether it matters;
- **2 off-reach noise** distractors — irreducible high error (the noisy-TV trap); catch `error-only`, which chases raw prediction error.

All reducible regions drift under a continuous **OU walk** on their curl gains ([`COLLECTION_REALISM.md`](../../COLLECTION_REALISM.md) §4), so nothing stays repaired and allocation is permanently zero-sum. Regions are classified by **actual path visitation** — gate-occupancy of the FK'd joint-space reach sweep — not a geometric proxy (see gotchas). Graded by the **sighted instrument** (region-A FM error; the S2 audit's lesson that control is a near-blind grader), with ballistic control as the behavioural cash-out.

## Result (3 seeds, mean over rounds, lower = better)

| policy | region-A err ↓ | ballistic | budget → noise |
|---|---|---|---|
| uniform | 0.475 ± 0.071 | 0.095 | 33% |
| error-only | 0.457 ± 0.059 | 0.094 | **47%** |
| lprog-only | 0.447 ± 0.108 | 0.078 | 35% |
| visits-only | 0.390 ± 0.053 | 0.080 | 13% |
| oracle (privileged) | 0.372 ± 0.044 | 0.073 | 0% |
| **value = lprog×visits** | **0.350 ± 0.043** | 0.077 | 29% |

1. **The retracted relevance claim reproduces.** `value` beats `lprog-only` on the sighted grader in **3/3 seeds** (Δ=−0.097). Directing a scarce, metered budget by `lprog × visits` keeps the region you actually reach into best-calibrated; `lprog-only` wastes budget on off-reach regions that never pay behaviourally.
2. **Reducibility-awareness is load-bearing (noisy-TV).** `value`/`lprog` beat `error-only` (`value` 3/3 on grader and control). `error-only` pours **47%** of its budget into the irreducible noise regions and collapses to ~uniform; `value` sends 29%, `oracle` 0%.
3. **The reward-free value signal matches the privileged oracle** (0.350 vs 0.372) — echoing `directed_loop`'s "continuous top-up beats episodic repair": `oracle` repairs when broken, `value` maintains a moving target.
4. **The FM is the bridge, read two ways.** `lprog` is a functional of the FM's own error trajectory (residual/`e` tap = *explore*); `visits` is computed by **rolling the FM itself** to forecast where the ballistic plan will go (forecast/`p` tap = *exploit*) — the idea doc's two afferent taps wired as one allocation signal over a shared FM the inner loop keeps calibrated. The efferent direction closes through **data selection**. (Not yet a backpropped learned-value head — that fully-online rung is still open.)

**An on-policy-specific finding the teleport world could not produce.** S1/S2's 2×2 has an *on-reach noise* region (C) to separate `value` from `visits-only`. On-policy it is **unplaceable**: visitation concentrates on the target itself (A visitation 1.0; everywhere else ~0), so there is no "visited-but-irrelevant" territory — *you only go where you reach*. So `visits-only` sits naturally close to `value`, and the relevance lever lives entirely in the off-reach direction.

## Caveats

- **Control is near-saturated** (~0.08 — the blind grader; `value`-vs-`lprog` is 2/3 on control vs 3/3 on the sighted grader), so the ladder resolves on FM error, as [`../../drift_value_loop/README.md`](../../drift_value_loop/README.md) Cut 3 predicts.
- **No policy fully repairs A** under the continuous drift — `value` closes the most of the stale→matched gap, it does not close it (a slower drift is the natural follow-up).
- `value` still leaks ~29% of budget to the noise regions — the counterfactual-fit LP is a quantitative, not perfect, filter; it wins anyway.
- Single task family, 3 seeds, one region geometry; the per-region off-reach "abandonment" errors are noisy (a near-singular distractor dominates).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/on_policy/verify_backcompat.py::verify                                    # the gate — env changes are byte-identical
modal run --detach mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --quick --tag ladder_smoke   # noisy-TV smoke
for s in 0 1 2; do
  modal run --detach mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --tag ladder_s$s --seed $s --mon-n 40
done
python3 mjc/on_policy/directed_on_policy/directed_on_policy_agg.py --tags ladder_s0 ladder_s1 ladder_s2
```

**Launch each seed as its own client** (not `&`+`wait` in one shell — detached siblings evict each other; see [../README.md](../README.md) §Gotchas).

## Gotchas (the geometry, learned the hard way)

1. **On/off-reach must be defined by actual reach visitation, not geometry.** The reach tip-path is widely curved (FK of a joint sweep), so a straight-corridor / angle-from-P0 test mislabels swung-through regions as off-reach and — worse — once placed an "off-reach" distractor ~0.1 m from the start posture, which *every* reach passes through, sending `value`/`oracle` to chase it instead of A. The fix gate-sums occupancy over the FK'd joint-space reach sweep.
2. **Regions must sit in the extended/tame radius band.** A folded/inward region's fast dynamics give ~10× the FM error and swamp the curl signal (a first smoke's A ceiling was 2.6 vs the tame 0.14).
3. **Noise `amp` must be `< gear`, regions well-separated.** Otherwise aleatoric noise bleeds into A and destroys its reducibility (a smoke's A ceiling jumped 0.14 → 0.37).
