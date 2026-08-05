# The composed loop: an endogenous judge on a level-indexed action space

**Status**: built and run. 12 loop runs + 2 static value probes, **single seed throughout** — no
contrast here is resolvable against seed variance, and several are reported as directional only.
**Date**: 2026-08-05.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Composes**: [`../endo_expansion/`](../endo_expansion/README.md) (the endogenous MC judge),
[`../level_moves/`](../level_moves/README.md) (the level-indexed action space, hierarchical damage),
[`../level_moves/level_ladder/`](../level_moves/level_ladder/README.md) (the span FM, the deepening
damage schedule).
**Idea docs**: [`heterogeneous_graders.md`](../../../../../ideas/heterogeneous_graders.md) §4/§5/§9,
[`adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md) §12.

---

## One-liner

Three results landed within four days and had never been in one run: an endogenous Monte-Carlo judge
**expands** (34% of the DP teacher's ballistic lift), the value's abstraction preference is **real and
error-depth-graded** once damage is hierarchical, and levels are worth **3.51× as an action axis and
~0 as an allocation axis**. This composes them. The headline 2×2 came back with a large negative
interaction — the endogenous judge captures none of the level action space's surplus — but the two
findings that survive their own controls are elsewhere: **the endogenous judge's target level is flat
at ~1.72 while an oracle climbs 2.23 → 2.80, and sharpening the judge does not move it**; and **what
the value's training rollouts explore is worth 45% of a privileged teacher on the no-teacher arm**.

---

## 1. What got built

[`composed_graders.py`](composed_graders.py) is [`../endo_expansion/endo_graders.py`](../endo_expansion/endo_graders.py)
generalised from **blocks** to the level-indexed **move set** — the DP teacher, the MC-rollout teacher,
the five-way disagreement instrument, the wireheading readout. Copied rather than flagged because every
function changes the candidate axis from `n_blocks` to `n_moves`, which changes the shape of every
returned array; `endo_expansion` is a merged node whose numbers must stay reachable.

The one object that is genuinely new is **`belief_update_moves`**. `channel_env.belief_update`'s plan
term scores candidates by rolling the block FM one step:

```
logits[k] = value((z + block_fm(z, k)).mean(1), r) / tau ;  CE(logits, k*)
```

The move-indexed form stacks over **moves** through the span FM, so a teacher can write back *commit to
this level-ℓ feature* rather than only *regenerate this block*. At `max_level=1` it reduces to the
published term exactly.

### The confound that had to be designed out

The dense term needs a move per gradient step, and both natural draws break the contrast:

| draw | what goes wrong |
|---|---|
| uniform over **moves** | tree holds 15/23 moves at L4 but 8/14 at L1 — the two action spaces see different **channel** mixes |
| uniform over **cells** | worse: 4 tree cells of 10 vs 1 of 5, a 2× swing |

Since the distractor geometry *is* this substrate's entire relevance structure, either would make "level
moves help" partly "the level arm saw more tree". `draw_moves_block_matched` picks a **block** uniformly
(the published `torch.randint(0, n_blocks)` convention) then a move covering it. Every block sits in
exactly one node per level, so the channel marginal is unchanged while levels stay flat within a channel.

**Gates** (`selfcheck`, CPU): **CG1** the flat move set is the block set in block order · **CG2** the
matched draw's channel marginal is within **0.0007** (L1) / **0.0013** (L4) of uniform-over-blocks ·
**CG3** every block is covered by exactly one move per available level. Move census — L1: 14 moves;
L4: 23 (`tree` 8/4/2/1, `structA` 2/1, `structB` 2/1, noise 1 each).

### Back-compat with the node it composes

Smoke diagnostics reproduce [`../endo_expansion/`](../endo_expansion/README.md)'s anchors closely:
homogeneous floor **+0.94 to +0.99** against its **+0.946**; `endo_random` on its constructed chance
values at both action spaces (`agree_dp` 0.072 vs chance 0.071 at L1; 0.046 vs 0.043 at L4);
`endo_rollout` `tree_share` **0.645** against uniform 0.571 — the reward-free relevance tilt reproduced.
Meter ratio L4/L1 = **2.66×** against the predicted (23/14)² = 2.7×.

## 2. The 2×2 — the headline, and it is a large negative interaction

Ballistic control on a fresh span FM, last graded round, damage depth 3 (no-loop floor **0.316**,
external ceiling **0.678**):

| judge | block (L1) | level (L4) | level gain |
|---|---|---|---|
| `evaluative` (external DP) | 0.225 | **0.678** | **+0.453** |
| `endo_rollout` (endogenous MC) | 0.096 | 0.285 | +0.189 |
| `frozen` (no loop) | 0.131 | 0.316 | +0.186 |

**Interaction −0.264**, growing monotonically with damage depth (−0.096 / −0.172 / −0.264).

Three readings, in descending confidence:

1. **Most of the level action space's value needs no outer loop.** `frozen` alone goes 0.131 → 0.316
   (2.4×) — planner search plus FM rollout over a richer action set, nothing learned.
2. **A good judge exploits levels well beyond that** (+0.453 against the free +0.186).
3. **The endogenous judge captures none of the surplus.** Its +0.189 is `frozen`'s +0.186.

On the block-only column the endogenous judge recovers **43% raw / 18% above the random-target floor**
of the external teacher, against [`../endo_expansion/`](../endo_expansion/README.md)'s published 34% —
i.e. the port lands in the same band as the node it builds on, which is the back-compat reading that
makes the level column worth reading at all.

**Gates.** Collapse check: `roll_vs_fm_pred` rank correlation sits at −0.02 to +0.07 across every arm
against a homogeneous floor of **+0.77 to +0.95**, so the two loops stayed distinct and no gain here is
an inner-loop gain wearing an outer-loop label. Content control: `endo_random` ballistic **0.068** (L1)
/ **0.033** (L4), far below `frozen`.

## 3. The endogenous judge has no upward direction — and this survives a matched teacher budget

The published damage schedule deepens across the run, so the correct move gets more abstract. Mean
target level, arm against the exact DP on the same states:

| | dmg 1 | dmg 2 | dmg 3 |
|---|---|---|---|
| `evaluative` (oracle) | 2.23 | 2.48 | **2.80** |
| `endo_rollout` | 1.72 | 1.72 | **1.73** |
| `endo_random` (floor) | 1.55 | 1.57 | 1.57 |

The oracle climbs +0.57; the endogenous judge is flat at 1.72, **0.16 above a constructed random floor
of 1.56**.

### The confound, owned and then removed

The first runs used `teacher_rolls=2` / `ground_states=3072→2048`, below
[`../endo_expansion/`](../endo_expansion/README.md)'s published `3` / `3072`. That was a wall-clock
cut, and it confounded "the judge has no direction" with "I gave it a noisy teacher". Re-run at the
published budget:

| | agree_dp | ratio to chance | mean target level | ballistic @dmg3 |
|---|---|---|---|---|
| thin (rolls 2, states 2048) | 0.115 | 2.65× | 1.72 | 0.285 |
| **matched (rolls 3, states 3072)** | **0.137** | **3.16×** | **1.78** | 0.236 |

**Accuracy and direction came apart cleanly.** A 1.5× sampling budget improved *which move* by ~19%
relative (2.65× → 3.16× chance) and `informative_frac` 0.898 → 0.921, while *which altitude* moved
0.06 (1.74 / 1.76 / 1.83 by damage depth, against an oracle at 2.50).

**The mechanism this suggests, stated as an argument rather than a measurement**: terminal task success
is a scalar over a short horizon. More rolls reduce its variance, so the argmax lands on the right move
more often — but nothing in that signal is *about* depth. This is
[`../level_moves/`](../level_moves/README.md) §10's *"error-depth matching, not a depth ordering"*
arriving from the **teacher** side rather than the value-readout side.

Ballistic did not improve at the matched budget (0.285 → 0.236, within ~1–2 SE), which rules out "the
thin teacher was holding the composed loop back".

## 4. The external "aim higher" instruction — it works, and its own control refutes the curriculum reading

If the deficit is directional, supply the direction from outside: mask the teacher's argmax to moves at
level ≥ ℓ_min before it commits. The instruction constrains only what the teacher may **commit to** —
the behaviour policy inside the rollout stays unrestricted, or the estimand changes rather than the
teacher. Disallowed moves are not rolled out, so a tighter instruction is *cheaper*.

| arm | ℓ_min | ballistic @dmg3 | recovery | vs `endo_treefloor` |
|---|---|---|---|---|
| `endo_rollout` (unfloored) | — | 0.285 | −0.09 | −0.002 |
| `endo_treefloor` (relevance only) | — | 0.287 | −0.08 | — |
| `endo_floor_static` | ≡ 2 | 0.305 | −0.03 | +0.018 |
| **`endo_floor`** | tracks damage (1,2,3) | **0.346** | **+0.08** | +0.059 |
| **`endo_floor_anti`** | reversed (3,2,1) | **0.383** | **+0.18** | +0.096 |

`endo_floor` clears the no-loop floor at all three depths (0.355 / 0.389 / 0.346 against 0.338 / 0.350 /
0.316) where the unfloored judge was below it everywhere. The instruction moves mean target level
1.72 → **2.47**.

**`endo_treefloor` is why this is readable and is not an optional extra.** On this geometry the
distractors are *shallower* than the tree (`struct_depths 2,2` vs `tree_depth 4`), so a level floor is
partly a **channel** oracle — at ℓ_min = 3 only tree moves survive, and "aim higher" silently becomes
"act on the tree", which the published ladder already prices at 13–18× relevance separation. Any floor
arm must beat *it*, not just the unfloored arm.

**And `endo_floor_anti` refutes the reading the arms were built for.** The reversed schedule is as good
or better. So *matching the instruction to where the error is* is not what does the work — which was the
entire curriculum story. `endo_floor_static` (constant ℓ_min = 2) buys nothing above the relevance
control, so it is not plain "aim higher" either. What the two working arms share, and static does not,
is that they **vary** ℓ_min across the full range. That is a coherent story and it is **post-hoc**,
generated by looking at this table, and no arm was designed to test it.

At the matched teacher budget `endo_floor` lands at **0.330** against the floor's 0.316 — all three
within ~1 SE. So the honest summary is: **the instruction demonstrably moves the judge's target level;
its effect on control is not resolvable at n=1.**

### The diagnostic that answers "could it rank abstract moves if pointed at them?"

`agree_dp_within_allowed` — among states where the floor left the oracle's own answer reachable, does
the teacher find it?

| arm | agree \| allowed | chance | ratio |
|---|---|---|---|
| `endo_rollout` (unfloored) | 0.115 | 0.043 | **2.67×** |
| `endo_floor` | 0.271 | 0.163 | 1.66× |
| `endo_floor_static` | 0.213 | 0.111 | 1.92× |
| `endo_treefloor` | 0.141 | 0.067 | 2.10× |

**Neither hypothesis.** The teacher ranks at roughly 2× chance *at every altitude* and does not improve
when restricted to abstract moves. It is not that the ranking was fine and only direction was missing,
nor that it cannot rank abstract moves at all. At the matched budget the floored ratio rises 1.66× →
1.85×, consistent with §3: more samples buy accuracy, not direction.

## 5. The prerequisite check, and the value-coverage result it produced

`level_value_probe` with lazy twins, on this node's round-0 value — **the matched-span pair, not a level
profile**, since [`../level_moves/`](../level_moves/README.md) §9 retracted the level-profile reading as
span. Probed at each damage depth separately, because §11's finding is that the premium is *graded* by
whether a commitment at that level can reach the error.

**Instrument validation**: the oracle column lands in the published range (0.543–0.661 against
0.610/0.642), and both distractor rows return **no untied pairs at all** — their Δ`d*` is exactly 0.000,
so ground truth never discriminates. Structural irrelevance confirming itself inside the readout.

`tree|L3` matched-span preference (value / oracle), same damage mixture and seed, differing only in the
action space the value's rollouts explored:

| value trained on | dmg 1 | dmg 2 | dmg 3 | rank corr with DP |
|---|---|---|---|---|
| **flat** actions | **0.595** | **0.634** | **0.610** | +0.535 / +0.548 / +0.528 |
| **level-indexed** actions | 0.537 | 0.530 | 0.520 | +0.288 / +0.210 / +0.075 |
| oracle | 0.610 | 0.656 | 0.569 | — |
| *published anchor* | *0.618* | | | *+0.290 → +0.583* |

Naive SE ≈ 0.019; pairs share states, so the true SE is larger.

**The damage mixture is exonerated and the action space is implicated.** With flat actions the
mixture-trained value reproduces the published premium (0.595–0.634 against 0.618) and tracks the DP at
+0.53 at every depth. Training the value on the **level-indexed** action space halves the premium and
decays rank agreement to +0.075.

**But flat is not simply better, and the same probe says why.** Off its training distribution the
flat-trained value is *confidently wrong* at the root: it prefers the committed level-4 move at
0.565–0.596 where the DP prefers the lazy twin at 0.22. The level-trained value gets that cell right
(0.36 vs 0.21, agreement **0.84**) while losing discrimination in the middle. **Experience with an
action space buys calibration at the extreme and costs discrimination in the middle** — a trade, not a
bug. *(The value is `V(state, root)`, not `Q(state, action)`; the action space enters only through which
states the rollouts visit.)*

### What the value explores is worth 45% of a privileged teacher

Ballistic @dmg3, varying **only** which behaviour policy generated the value's training states:

| arm | `own` (level) | `flat` | `mixed` (half each) |
|---|---|---|---|
| **frozen** (no teacher) | 0.316 | 0.213 | **0.479** |
| `evaluative` (external DP) | 0.678 | 0.645 | 0.676 |
| `endo_rollout` (endogenous) | 0.285 | 0.162 | 0.180 |

**The `frozen` row is exactly controlled**: `frozen` never uses the value in `belief_update`, and its
controller comes out **bit-identical** across all three (rootCE 0.3126, d4 0.6221, PR 6.917). Every
difference in that row is the value alone. The whole external-DP teacher is worth 0.678 − 0.316 = 0.362
on this arm; the value-coverage change captures **45% of that with no teacher, no labels and no extra
loop.**

**Two pre-registered predictions failed, and the instrument built for one of them excluded it.**
`flat` was expected to beat `own` on the probe's evidence — it is the *worst* setting in the loop. And
the hazard `planner_mean_level` was added to catch (a flat value being exploited toward the root) **did
not fire**: 2.11 (flat) vs 2.06 (mixed) while ballistic differs 2.2×. `flat` fails while choosing moves
at the same depth. Calibration bias is consistent with the plainer reading — `frozen/own` +0.017
(nearly calibrated) against flat −0.168 and mixed −0.183.

**The caveat that bounds this.** `endo_rollout` runs the *other* way (`own` 0.285 > `mixed` 0.180), and
there is no account of why. Both teacher arms train the value through the plan CE, which predicts
*insensitivity* (`evaluative`: 0.678 / 0.645 / 0.676) rather than a reversal. So the defensible claim is
narrow: **on the no-teacher arm, under an exactly controlled comparison, what the value explores is
worth 45% of what a privileged teacher is worth.** Everything about the interaction with a teacher is
open.

## 6. Two instrument findings worth carrying

**PR and depth remain anti-informative, more starkly than published.** `endo_random` — imitating
uniformly random moves, zero information by construction — posts the run's **highest PR (13.3 / 13.6)
and highest d4 (0.687 / 0.699)** while producing the **worst** ballistic in the experiment (0.033).
[`../endo_expansion/`](../endo_expansion/README.md) §3 found this once; it reproduces here at both
action spaces. Read on ballistic and fresh-FM top1.

**Level moves cost self-predictability while buying control.** On fresh-FM `value_top1_agree` the level
action space *hurts* the teacher arms (`evaluative` −0.070, `endo_rollout` −0.129) and leaves `frozen`
flat (−0.004), while tripling ballistic. This is
[`../level_moves/level_ladder/`](../level_moves/level_ladder/README.md)'s quantified grader disagreement
(−0.106 on the dense proxy, 3.51× on the task) appearing on a second readout. **Consequence for design,
argued not measured**: a drive that is a functional of forward-model error — learning progress,
reducibility, `fm_cotrain` — would *veto* abstraction here rather than merely fail to find it. That sits
alongside [`../metering_sweep/`](../metering_sweep/README.md)'s independent finding that a priced
reducibility tap recovers ~4% of what relevance recovers.

## 7. What this establishes — and what it does not

**Establishes** (single seed; every number is one draw):

1. A move-indexed belief update, teacher set, and disagreement instrument that reduce exactly to the
   published block-indexed ones at `max_level=1`, gated three ways, with back-compat anchors matching
   [`../endo_expansion/`](../endo_expansion/README.md) to within its own reported spread.
2. **The endogenous MC judge does not climb**, and sharpening it does not make it climb — accuracy and
   direction respond differently to sampling budget.
3. **An external ~14-bit-per-run instruction moves the judge's target level 1.72 → 2.47** and lifts it
   from below to above the no-loop floor at the thin budget; the effect on control is inside noise at
   the matched budget.
4. **The value's exploration distribution is a large lever on the no-teacher arm** (+52% ballistic,
   45% of a privileged teacher's worth), under a bit-identical-controller comparison.
5. Two instrument results reproduced: PR/depth are anti-informative about control; level moves trade
   self-predictability for control.

**Does not establish:**

- **Nothing at n=1 is resolvable against seed variance.** Eval-sampling SE alone is ~±0.021 per cell;
  seed spread is a larger and unmeasured quantity. The 2×2 interaction (−0.264) and the frozen
  value-coverage effect (+0.163) are large relative to sampling error and are still single draws.
- **No claim about *why* the floor arms work.** The curriculum reading is refuted by `endo_floor_anti`;
  the surviving "vary ℓ_min across the range" pattern is post-hoc and untested.
- **No account of the `endo_rollout` reversal** in §5, which runs opposite to the `frozen` row.
- **The `own`-value runs predate `planner_mean_level`**, so the flat-vs-own planner-depth comparison is
  against `mixed` only.
- **Nothing about grader *type* in general.** This measures one endogenous instantiation (MC terminal
  success) on one geometry, and §3's mechanism — that a scalar outcome signal carries no depth
  information — is an argument, not a measurement.
- **The distractors are shallower than the tree here**, which is what forces `endo_treefloor` to exist.
  A depth-matched geometry (`--struct-depths 4,2`, built in
  [`../partial_hetero/`](../partial_hetero/README.md)) would separate altitude from relevance by
  construction, at the cost of re-running every comparator.

## 8. Next steps, in the order they seem worth doing

1. **Seed `frozen` × {own, flat, mixed}.** The cheapest arm, the cleanest comparison in the node (the
   controller is bit-identical), and the largest effect. This is the one result worth promoting or
   retiring first.
2. **A hypothesis for the `endo_rollout` reversal** before any GPU. Both teacher arms overwrite the
   value through the plan CE, which predicts insensitivity, not a sign flip.
3. **Depth-matched distractors** so a level floor is not partly a channel oracle, which would let §4's
   floor arms be read as altitude without needing `endo_treefloor` as a comparator.
4. **A teacher whose signal carries depth information at all** — §3 says terminal success cannot. The
   obvious candidate is a *horizon-varying* rollout (a deep commitment pays off over more steps), which
   is a change to the estimand rather than to its variance.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                    # the standing gate
modal run rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::selfcheck_remote   # CG1-CG3

# the 2x2 -- one job per (action space, arm group); the 2x2 is read ACROSS jobs
for lv in 1 4; do
  modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
      --tag l${lv}a_s1 --seed 1 --max-level $lv --arms "frozen,dense,endo_random" --rounds 9 --grade-every 3
  modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
      --tag l${lv}b_s1 --seed 1 --max-level $lv --arms "evaluative,endo_rollout" --rounds 9 --grade-every 3
done

# the PI instruction (level floors) -- L4 only; a level floor is vacuous at L1
modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
    --tag fl_a_s1 --seed 1 --max-level 4 --arms "endo_floor,endo_floor_static" --rounds 9 --grade-every 3
modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
    --tag fl_b_s1 --seed 1 --max-level 4 --arms "endo_floor_anti,endo_treefloor" --rounds 9 --grade-every 3

# matched teacher budget -- endo_expansion's published rolls/states
modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
    --tag mt_a_s1 --seed 1 --max-level 4 --arms "endo_rollout" --teacher-rolls 3 --ground-states 3072 \
    --rounds 9 --grade-every 3

# the value's exploration distribution
for v in flat mixed; do
  modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \
      --tag vas_${v}_a_s1 --seed 1 --max-level 4 --arms "frozen,evaluative" \
      --value-action-space $v --rounds 9 --grade-every 3
done

# the static matched-span prerequisite probe (no loop, ~3 min)
modal run rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::value_probe \
    --tag vp_l4_s1 --seed 1 --max-level 4 --n-probe 512          # value trained on level actions
modal run rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::value_probe \
    --tag vp_l1_clean_s1 --seed 1 --max-level 1 --n-probe 512    # ... on flat actions

# mirror and read
for t in l1a_s1 l1b_s1 l4a_s1 l4b_s1 fl_a_s1 fl_b_s1 mt_a_s1 mt_b_s1 \
         vas_flat_a_s1 vas_flat_b_s1 vas_mix_a_s1 vas_mix_b_s1; do
  modal volume get --force rhm-scaling-data "directed_sculpting/composed_loop_$t" \
      rhm/directed_sculpting/full_loop/composed_loop/figures/; done
python3 rhm/directed_sculpting/full_loop/composed_loop/aggregate.py --pattern 'composed_loop_*_s1'
```

`--max-level 1 --value-action-space own` is the published block-indexed configuration; `--pattern` is
required on `aggregate.py` for the reason [`../level_moves/aggregate.py`](../level_moves/aggregate.py)
records — mirroring two sweeps into `figures/` and keying by seed silently overwrites cells.
