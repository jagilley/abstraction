# Teleport collection — the substrate's standing unrealism, and the fix

**Scope**: every experiment under `experiments/mjc/`, both task families.
**Written**: 2026-07-23, out of the [`ballistic/directed/`](../ballistic/directed/README.md) S2 audit.
**Recommendation**: build **on-policy collection** (§3) and make it the default for new cuts.
**Read this before**: designing any new cut, and before quoting any sample-count number from this node.

**Status (2026-07-23): §3 is built; §4's stopgaps are not needed.** The machinery is
[`embodied.py`](../embodied.py) (`Body` — metered, per-episode `MjData`, **no `set_state`**), the flag
is `collection_mode="teleport" | "on_policy"` on `arm_env.collect_pool` / `pusher_env.collect_pool`,
and the node is [`on_policy/`](FILES.md). Backwards compatibility is **gated, not
promised**: `on_policy/verify_backcompat.py` checks the teleport path is bit-identical to the
pre-flag implementation (it is, `array_equal`, and the RNG is left at the same position), so every
Tier-A cut is untouched. E0 (`on_policy/coverage_probe.py` — the B0–B3 behaviour ladder at matched
step budgets) is running; **no results yet, and nothing in §2's tiering has been revised.**

Three places the memo below is superseded by what the build actually needed:
1. **§3's "one flag" is two.** `collection_mode` (how transitions are obtained) × `behaviour`
   (what drives the trajectory). The memo says the behaviour choice "is the explore/exploit
   question in disguise" and then hides it inside one enum value; if it is that, it cannot be a
   hidden default. Hence the enumerated **B0/B1/B2/B3 ladder**, which also decomposes on-policy's
   cost into coverage-lost-to-continuity vs alignment-gained-by-being-on-task vs the
   model↔data coupling. B0-vs-B3 alone confounds all three.
2. **§5's "print the ratio" is replaced by a meter that cannot be bypassed.** The 22× subsidy did
   not happen because nobody knew better; it happened because `collect_region()` was callable for
   free. `Body` charges every `env.step` and does not expose `set_state`.
3. **§2's "re-run 4c-arm both ways is the first thing to do" needs a finer milestone grid first.**
   From `ballistic/arm/logs/`, the ballistic recovery is 0.373 → **0.086 at m=400** → 0.086 at
   m=14000: the entire step sits inside the first non-zero milestone. At the current grid both arms
   would show a step and the question would be unanswerable in either mode.

---

## 1. What the limitation is

Every FM in this node is trained on transitions gathered by **teleportation**:

```python
env.set_state(q, qd)          # snap the body to an arbitrary configuration and velocity
u = rng.uniform(-1, 1, n_u)   # apply an arbitrary command
s, s2 = env.get_state(), env.step(u, frame_skip)   # record one isolated (s, u, s') triple
```

`pusher_env` and `arm_env` both do this; `arm_env.py:378` (`collect_pool`) states it as the
contract every script in the directory is written against, and `arm_env.py:57` names it as
load-bearing (it is *why* the operating region can be bounded by the sampling distribution rather
than by joint limits, which would be stiff constraints and therefore another discontinuity).

Four properties follow, and they always travel together:

| property | what the code does | what a body does |
|---|---|---|
| **discontinuous** | each sample is an isolated 1-step triple from an arbitrary state | experience is one continuous trajectory; the state you see next is the one you caused |
| **omniscient coverage** | samples are ~i.i.d. across the whole workspace from step one | coverage grows only as you explore, and is shaped by what you can already do |
| **free** | a transition costs a counter decrement; per-round budgets refill | moving costs time, energy, and risk of damage |
| **cheap to measure** | probe/monitor surveys sample everywhere at no charge | you can only measure error where you have actually been |

It was a deliberate variable-control choice, and for the cuts already built it is doing real work
(§2, tier A). The point of this memo is that it has been silently load-bearing well outside the
place it was chosen for, and that the next substrate should not inherit it.

## 2. What it does and does not invalidate

Three tiers. Getting the tier right matters more than the general complaint.

### Tier A — safe, and teleportation is genuinely the control

**Every dissociation and ratio.** Both arms of a comparison are graded on the *same* FM trained on
the *same* data, so the coverage advantage cancels exactly. This covers essentially all of the
node's headline claims: Cut #1's contact-residual structure, Cut #2's arity gap, 4b's 3.0–3.3×
transmission, 4c's 4.3×, [`ballistic/arm/`](../ballistic/arm/)'s 5.24× ± 0.18 and its mirror-signed
aftereffect, and all of `arm_substrate` P0–P6.

For these, uniform teleported collection is what guarantees the only thing varying is model
quality. **Do not retrofit §3 onto them** — it would replace a clean measurement with a compound
one and break reproducibility for no gain. They stay as they are.

### Tier B — inflated, and currently mis-stated

**Every absolute sample-count claim.** Cut #3's *"recovers from ~50 reward-free transitions"*,
4c's *"~200"*, 4c-arm's *"96% of recovery complete by 400"*, and the per-round budgets throughout
[`ballistic/directed/`](../ballistic/directed/README.md). Those are counts of *teleported* samples
spanning the whole workspace at random velocities — near-i.i.d. coverage handed over for free. A
body collecting the same number of transitions along its own trajectories gets drastically less
information. These numbers are **upper bounds on sample-efficiency**, not estimates of it, and
should be quoted that way.

> **A live hypothesis this generates.** 4c-arm's open caveat is that re-adaptation is a **step**
> (96% recovered by the first milestone) rather than the graded curve we expected a harder plant
> to produce. Omniscient i.i.d. sampling repairs the model *everywhere at once*, so a step is
> exactly what it should produce. On-policy coverage grows as the agent explores, which would
> naturally stretch the step into a trajectory. **If §3 gets built, re-running 4c-arm both ways is
> the first thing to do**: it tests this directly and would explain a caveat that changing the
> *plant* could not fix.

### Tier C — unaskable on this substrate

**Anything whose dependent variable is the allocation of experience**: directed collection /
where-to-practise ([`ballistic/directed/`](../ballistic/directed/README.md) S1, S2), curiosity or
explore-exploit drives that must pay for themselves
([`curiosity_control/`](../curiosity_control/README.md),
[`directed_readapt/`](../directed_readapt/README.md)), and the idea doc's `p`-tap (relevance /
"where will I be") as a *lever*.

The failure is not subtle. In `directed_loop.py` every policy spends `probe_n×K + monitor_n×K` =
**2,240 free teleported transitions per round, everywhere in the world**, to decide where to spend
a budget of **100** — a **22× measurement subsidy**. The relevance term exists to answer *where
should I even look*; when looking is free and global it is demoted to a tiebreaker on repair
effort, and repair is cheap. Combined with free teleportation, a refilling budget, K=4 hand-drawn
regions and drift as a scheduled one-region-at-a-time event, off-manifold data becomes **a one-time
purchase in a world with four places**. S2's apparent null about the relevance term was retracted
for exactly this reason; `ballistic/directed/directed_loop_audit.py` reproduces the audit from
stored data.

**The arm does not fix any of this.** It fixes the *plant* (real mechanics instead of six bolted-on
`qfrc_applied` layers; continuous locality instead of Gaussian gates) and inherits the *acquisition
model* unchanged. Physics realism and experience realism are independent axes; `arm_substrate`
moved one. Do not let "we moved to the arm" launder an acquisition question.

## 3. The fix — on-policy collection, as the default

**Data becomes a byproduct of behaviour.** The agent gets the transitions its body actually passed
through; to sample somewhere else it must plan a movement that goes there, paying episode time it
is not spending on the task. No teleport sampler, no per-round transition budget (budget becomes
wall-clock), no hand-drawn region partition — "where to collect" becomes a goal in continuous
space. Under this, everything the value signal is supposed to do becomes structurally
non-optional: relevance sits *upstream* of measurement instead of multiplying it afterwards,
allocation is permanently zero-sum, and explore/exploit has a real price. It is also, simply,
a cerebellum's actual situation.

### Build it as one flag, not two environments

`collection_mode: "teleport" | "on_policy"` on the existing collection helpers
(`arm_env.collect_pool`, the per-script `collect_pool`/`collect_region` idioms). **Not** a second
substrate. This matters for three reasons:

1. **No environment management.** One plant, one set of knobs, one code path to keep in sync. New
   cuts default to `on_policy`; finished cuts keep `teleport` and stay byte-identical.
2. **The flag is itself the decomposition.** Re-running a cut both ways *measures* how much of its
   sample-efficiency was a gift from omniscient sampling. That converts the control we give up
   (below) into a measured quantity rather than a lost one, and it is how every Tier-B number gets
   corrected without re-deriving it.
3. **`teleport` remains the honest baseline** — the coverage ceiling an on-policy learner is
   working against.

### The design decision this forces, which should be made deliberately

On-policy collection needs a **behaviour policy** to do the gathering, and that choice will matter
more than the plumbing. Random walk? Goal-directed reaches with exploration noise? The CEM planner
itself, rolled on the current (possibly stale) FM? This is not an implementation detail — *it is
the explore/exploit question in disguise*, which is precisely the thing Tier C wants to study. Pick
the dumbest defensible default first (goal-directed reaches + noise), get it working, and treat
"what should the gathering policy be" as the research question it is, rather than answering it
accidentally in a helper function.

### The honest cost — this is scientific, not just engineering

On-policy collection **couples model quality to data quality**: a stale model moves badly, which
gets you worse data, which keeps it stale. That is the genuine bootstrapping problem and it is part
of the real phenomenon — but it means a clean "how much does FM quality reach behaviour"
measurement becomes a *compound* effect, and Tier A's dissociations would no longer be clean if
measured this way. **Rung 3 buys realism by spending a control.** That is the right trade for new
work and the wrong one for finished work, which is exactly why the flag exists.

### Do not mutate the existing cuts

New node, own baselines, teleport-based cuts reproducible beside it (repo rule: keep all prior
results reproducible; duplicate rather than mutate).

### The warning from the one time we tried it

[`ballistic/`](../ballistic/README.md) Cut 4a *was* the embodied version — an actual behavioural
explore/exploit drive `b` — and it was abandoned as "confounded by the corridor geometry": the
reducible needle sat *on* the reach and the aleatoric noise *off* it, so the intended
explore/exploit roles inverted. The confound was real, but the fix chosen was to go disembodied,
which removed the phenomenon along with the confound. Fix the **geometry** this time (put the
reducible structure off the default path and the noise on it, or randomize goal direction so there
is no fixed corridor) and keep the embodiment.

## 4. Stopgaps, if you need a result before §3 exists

Both are cheap and neither is the recommendation. Take them only to unblock something specific.

- **Charge measurement against the collection budget.** In a directed-collection loop, draw the
  `probe_n`/`monitor_n` surveys from the *same* budget the policy allocates instead of granting
  them free alongside it. A config change. It removes the 22× subsidy and makes "where should I
  look" a real question, without touching how transitions are obtained. Verify by re-running the
  S1 ladder: if `lprog × visits` beats `lprog`-only by more than it did under free monitoring, the
  relevance term's job was measurement-narrowing all along.
- **Continuous drift instead of a scheduled switch.** Replace "redraw region *j* every
  `drift_every` rounds" (and 4c's `b0 → b1` flip) with an Ornstein-Uhlenbeck walk on the
  perturbation field over the whole workspace. Nothing is ever permanently repaired, so a
  relevance-blind policy pays a recurring tax rather than a one-time cost and allocation stays
  zero-sum. Also removes the K-region partition as a modelling assumption.

## 5. How to tell you are about to hit this

1. You are writing an allocation policy over a **hand-specified partition** of state space that the
   agent did not have to discover.
2. Your per-round diagnostic sampling (`probe_n`, `monitor_n`, ensemble surveys) exceeds the budget
   the policy allocates. **Print the ratio**; above ~1 the experiment is subsidised.
3. You are about to quote an absolute sample count as a sample-efficiency result (Tier B).
4. Your headline metric is a **control** readout. [`drift_value_loop/`](../drift_value_loop/README.md)
   Cut 3 established control is a near-blind grader of FM quality — grade by value-relevant FM
   prediction error and check whether the two instruments agree before believing either.

## 6. Pointers

- The audit that produced this memo: `ballistic/directed/directed_loop_audit.py` and
  [`ballistic/directed/README.md`](../ballistic/directed/README.md) §The S2 audit.
- The contract itself: `arm_env.py:378` (`collect_pool`), `arm_env.py:57`; `PusherEnv.set_state`;
  `directed_loop.py:153` (`collect_region`).
- Why the arm does not fix it: [`arm_substrate/README.md`](../arm_substrate/README.md) §"What this
  substrate does *not* fix"; the step-like-recovery caveat it generates is in §P7.
- The abandoned embodied attempt: [`ballistic/README.md`](../ballistic/README.md) Cut 4a.
