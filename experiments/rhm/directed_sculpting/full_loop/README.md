# The full loop on RHM: inner + outer + forward model, and what drift actually buys

**Status**: built and run. Three conditions (allocation ladder, hierarchy climb, expansion 2×2) plus a
necessity sweep, one certified environment, and several instrument bugs found and fixed mid-flight. Ladder and
climb sweep are 3 seeds; the expansion 2×2 is 3 seeds × 3 grading worlds × 2 drift anchors. **Date**: 2026-07-28.
**Up**: [../README.md](../README.md) (directed_sculpting) · **Node**: [../../README.md](../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../ideas/adaptive_core_and_hierarchy_climb.md) — this builds its §4 loop, tests §6/§11's climb and §5's missing cell.
**Direct parent**: [`mjc/on_policy/directed_on_policy/`](../../../mjc/on_policy/directed_on_policy/README.md) (E3, PR #4) — the loop being ported.
**Substrate**: [`../README.md`](../README.md) — the distractor DGP, support-fixed drift, and the repair-cost instrument this stands on.

---

## One-liner

The reward-free **relevance** signal ports and is a smoking gun — a value trained only on terminal task
success, never told which channel is which, separates real tokens from distractors **13–18×** and recovers
**76%** of a privileged oracle's advantage over uniform allocation. **Expansion turns out to be a property of
the grader's *type*, not of non-stationarity**: a grounded evaluative grader expands the belief ~1.7× in
effective dimension and ~15× in ballistic control while an endogenous dense grader caps *below the no-loop
floor*, and this holds identically in a static and a drifting world. Both of the idea doc's drift-dependent
predictions come back **negative under instruments we had to repair repeatedly**, each repair removing a bias
that pointed toward the positive result. But the climbing null is *scoped*, not general: it held only because
surface re-fit repaired 91% of each drift event, so invariance was free and nothing priced it. Starve
**samples per event** at fixed drift magnitude and the drifting arm's depth advantage **migrates from the
surface to the deep levels**, monotonically across four budgets (t = −5.44, 3/3 seeds) — small, but the first
positive this arc has produced on the climbing axis. **Invariance ≠ necessity** was the missing step.

---

## 1. What got built

[`../README.md`](../README.md) left the substrate certified and the loop unbuilt: *"No arity-2 block FM, no DP
`k*` teacher, no six-policy ladder on this substrate. Everything here is the DGP and the instruments."* This is
the loop.

[`channel_env.py`](channel_env.py) teaches the sculpting apparatus (controller / generator / block FM / MC
value / beams, from [`../../rhm_sculpt_latent.py`](../../rhm_sculpt_latent.py)) to live on a multi-channel
sequence. Three changes were needed, and each is load-bearing:

1. **Per-block rendering tables.** Each channel edits through *its own* grammar. Noise blocks have none —
   regenerating one redraws its tokens iid, which is what makes them an irreducible target for the FM
   (`verify_distractors` P2, restated in the FM's currency).
2. **Mixture rendering.** The old move rendered a block's inferred feature as its *canonical* synonym. Under a
   frozen generator that is a fixed deterministic map, so support-fixed drift would be pure covariate shift and
   the FM would have nothing to re-adapt to. Drawing the synonym from the cell's *current drifted mixture*
   makes drift bite on the dynamics — `rhm_drift`'s "drift the encoding, not the content" taken literally. `d*`
   is unaffected: every synonym legally realises the same feature.
3. **Corruption confined to tree blocks.** Damage to a distractor cannot move `d*`, so it would be damage the
   task is structurally unable to see.

The **three taps** (§4) are one FM object read three ways, which is what makes this one system:
forward for control (`open_loop_beam`), residual-derivative to explore (`counterfactual_lp`), rolled forward
for relevance (`forecast_visits`).

### The back-compat gate

`env_check.py::fm_bisect` runs the same measurement through four conditions, one variable at a time:

| condition | `delta_cos` | `top1` | `rank_corr` |
|---|---|---|---|
| reference — the literal `rhm_sculpt_latent` pipeline, 8 blocks | 0.486 | 0.361 | 0.528 |
| **tree-only — *this* code path, same task** | **0.492** | **0.402** | **0.535** |
| + distractor channels (14 blocks) | 0.360 | 0.281 | 0.421 |
| + mixture rendering | 0.310 | 0.303 | 0.348 |

The reference reproduces the published 0.49 / 0.357 / 0.527, and this code path matches it. So the port is a
genuine back-compat pass, and the two design choices now carry price tags instead of hopes: the distractor
geometry costs ~27% of FM fidelity, making drift bite costs another ~14%. Both affordable.

## 2. The environment, certified before the loop ran on it

`verify_distractors` certified the DGP at the token-CE level and flagged that *"the FM-level LP the real loop
reads is a separate check."* [`env_check.py`](env_check.py) is that check (E1–E6, L=4 and L=5).

**E2 — the irreducible floor, measured rather than assumed.** Redraw the same `(x, k)` four times; the residual
around the conditional mean is the error no FM can beat. Acted-block, mixture rendering:

| | tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|---|
| **floor** | **0.189** | 0.272 | 0.461 | 0.502 | 0.500 |
| FM after 12k steps | 0.651 | 0.942 | 0.923 | 0.921 | 0.899 |
| headroom used | **+0.42** | +0.01 | +0.06 | +0.08 | +0.12 |

Ordered exactly as the ladder needs (noise > struct > tree), and **exactly 0.000** for the grammar channels
under deterministic rendering — E1 passes. The headroom row *is* the `lprog-only` trap: uniform collection
gives the tree 8/14 of the data and it burns 42% of its available headroom, while the struct channels sit
almost untouched with the most left to chase. Same asymmetry `verify_distractors` P4 found (tree satiates at
step 4000, structs at 7500), now in the currency the outer loop reads.

**E5 — the ballistic commit length.** Open-loop success is cleanly humped in commit length: `c1` 0.045,
**`c3` 0.092**, `c6` 0.043 — confirming the ~6-step veridicality bound from the length-gen arc, and setting
`chunk_len=3`. Reactive (re-grounded) control sits at 0.189 and is flat across policies: the near-blind grader
E3's own reactive control turned out to be.

## 3. E1 — the allocation ladder ([`ladder.py`](ladder.py))

Seven policies × 12 rounds under continuous drift, forking from one warm FM, on a bit-identical world sequence.
Monitor:collect **1.89×** against E3's 1.84×.

| policy | tree FM err ↓ | ballistic ↑ | →noise | →struct | →tree |
|---|---|---|---|---|---|
| oracle (privileged) | **0.676** | 0.110 | 2.0% | 2.0% | 96.0% |
| **visits_only** | **0.688** | **0.105** | 11.3% | 21.4% | 67.3% |
| value_satiety | 0.705 | 0.094 | 21.5% | 32.9% | 45.6% |
| lprog_only | 0.718 | 0.091 | 36.7% | 35.6% | 27.7% |
| value = lprog × visits | 0.720 | 0.086 | 21.5% | 34.0% | 44.4% |
| uniform | 0.727 | 0.083 | 40.0% | 40.0% | 20.0% |
| error_only | 0.736 | 0.078 | 39.8% | 42.8% | 17.4% |

### The smoking gun: the value system devalues distractors

The MC value is trained only on terminal possible-set success. It is never shown a channel label. It
nonetheless separates them:

| | tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|---|
| value sensitivity \|ΔV\| (in-loop, mean over rounds) | **0.431** | 0.024 | 0.027 | 0.029 | 0.033 |
| forecast visits (the `p` tap) | **0.673** | 0.164 | 0.065 | 0.064 | 0.034 |
| ground-truth best Δ`d*` | **+1.702** | 0.000 | 0.000 | 0.000 | 0.000 |

13–18× separation; the value's top-1 move is a tree block **94%** of the time; its move ranking correlates
**+0.31** with the exact DP's. Replicates at L=5 (0.61 vs 0.05, 94.5%, +0.30). And it *pays*: `visits_only`
recovers **76%** of the oracle's advantage over uniform on the sighted grader with zero DGP knowledge, matched
by the behavioural grader (0.105 vs oracle 0.110 vs uniform 0.083). `error_only` is the worst arm and pours 40%
of its budget into the irreducible channels — the noisy-TV trap, reproduced.

### The negative that turned into a repair: our reducibility tap was inverted

E3's headline ordering did **not** reproduce on the first pass — `value` lost to
`visits_only`. The reason was the estimator, not relevance. Mean counterfactual LP per
channel came out **exactly inverted**:

| tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|
| **+0.044** | +0.053 | +0.057 | +0.064 | **+0.066** |

Lowest on the target, highest on the irreducible channels. A fixed-budget counterfactual fit
measures *marginal return*, and early in training marginal return is dominated by **data
starvation**: block-proportional warm-up leaves the 1-block noise channels furthest back on
their learning curve, so probing them yields the biggest held-out drop. The noisy-TV *filter*
had become the noisy-TV *attractor*. And the loss was **variance** more than bias —
`lprog × visits` implied a 61% tree share on its round-averaged taps but realised 44%, with a
per-round tree-share sd of **0.309** against `visits_only`'s 0.059.

**The repair** (`reducible_fraction`, `measure_floors`): replace the flow with a **stock** —
`(err − aleatoric floor) / err`, where the floor is *measured* by re-executing the same
command from the same state several times and reading the residual around the conditional
mean. Metered, label-free, no channel labels anywhere. This is the honest version of §4's
*"filtered by fresh-FM-ensemble invariance so it rejects the noisy TV"*: an ensemble filter
**infers** irreducibility, repeat-execution **measures** it — and exact re-executability is
something RHM hands us that a physical substrate does not. Plus EMA smoothing and frozen
probe states for the variance half.

Re-run at 3 seeds, with both estimators computed every round for every arm so the monitor
charge is identical and only the allocation rule differs (monitor:collect **1.78×**):

| policy | tree FM err ↓ | ballistic ↑ | →noise | →tree | tree-share sd |
|---|---|---|---|---|---|
| oracle | 0.6742 ± 0.0118 | 0.100 | 2.0% | 96.0% | 0.000 |
| **visits_only** | **0.6844 ± 0.0132** | 0.093 | 10.4% | 73.6% | 0.045 |
| **value_red** | **0.6863 ± 0.0129** | 0.092 | 7.3% | 76.7% | **0.047** |
| reducible_only | 0.7259 ± 0.0156 | 0.078 | 28.1% | 26.5% | 0.009 |
| uniform | 0.7371 ± 0.0170 | 0.076 | 40.0% | 20.0% | 0.000 |
| value (old tap) | 0.7406 ± 0.0132 | 0.077 | 14.2% | 38.9% | **0.396** |

1. **The repair is large and consistent**: `value_red − value` = **−0.0542 ± 0.0091**, 3/3
   seeds. Tree-share sd 0.396 → 0.047; tree share 39% → 77%; leak 14.2% → 7.3%. The
   reducibility tap now orders the channels correctly (tree 0.856 ≈ structA 0.845 >
   structB 0.609 > noise 0.421/0.432).
2. **But `value_red` only ties `visits_only`** (+0.0019 ± 0.0030; per-seed +0.005 / +0.001 /
   −0.000), both close to the privileged oracle (+0.0122 ± 0.0055). That is **structural**:
   this geometry has no *visited-but-irreducible* cell, so the channels the planner visits are
   also the reducible ones and reducibility carries no information beyond relevance. E3 hit
   the mirror image — no visited-but-*irrelevant* cell, *"you only go where you reach"*. The
   ladder cannot separate the two by construction; what is new is that this is now shown with
   a **working** estimator rather than a broken one.

### Satiety: §10 confirmed, and §12 found to be missing an axis

With a drive that is no longer inverted, §10's prediction is finally testable — and it holds.
The leak into the irreducible channels falls **7.3% → 2.8%** against a hard 2.0% uniform
floor, i.e. the *learned* leak drops from 5.3% to **0.8%**.

But the satiating arm is **worse overall** (+0.0691 ± 0.0107 tree-err). Once satiety hardens
on the tree it recruits **laterally into structA** — reducibility 0.845 against the tree's
0.856, but 4× less relevant — rather than upward. That is §12's rule behaving exactly as
written, *"once level ℓ stops paying, recruit ℓ+1"*, on an allocation space that **has no ℓ**:
the budget is allocated over *channels*, which are not ordered by level. Satiety is
level-agnostic, so on a flat space it moves sideways into a distractor.

> **§12 names satiety as the climbing mechanism, but climbing needs the allocation space to be
> ordered by level. Ours is not.** That is a design gap in this loop, not a refutation of §12 —
> and it is the standing reason the climb result below cannot be read as a test of §12.

## 4. E2 — the hierarchy climb ([`climb.py`](climb.py)): no climb *where surface repair suffices*

Ten drift events with a *trainable* belief under the evaluative grader, swept over which level
drifts, against a `nodrift` matched-compute arm. Depth gain relative to that control, at
per-event matched magnitude (0.539 / 0.498 / 0.566 nats — within 7%):

| arm | Δd1 | Δd2 | Δd3 | Δd4 |
|---|---|---|---|---|
| surface | +0.005 | −0.005 | −0.011 | −0.026 |
| mid | −0.011 | −0.020 | −0.007 | +0.003 |
| root | −0.008 | +0.005 | −0.023 | −0.013 |

All within ±0.026, no ordering. The depth rise *is* real and large — `nodrift` alone goes
d3 0.645 → 0.711, d4 0.769 → 0.827, PR 9.1 → 11.2 — but it is caused by the evaluative grader,
not by drift.

**This is a much narrower statement than "no climbing", and §6 below shows why.** Measured
after the fact: the surface arm **repaired 91% of its damage inside the round**
(residual/damage 0.09 over the last five) and converged to the `nodrift` arm's root CE every
single round (0.320–0.361 against 0.330–0.345). Invariance was free for everyone, so nothing
ever priced the integrated cost. The result says *drift alone, in a regime where surface
re-fit suffices, does not induce climbing* — see [§6](#6-necessity-what-the-climb-null-was-
actually-missing) for the sweep that removes that regime.

**The `root` arm was separately confounded** and should not be read at all: its end-of-round
root CE falls *monotonically* to 0.091, far **below** `nodrift`'s 0.33, because drifting
level 0 makes the root more identifiable from the sequence. That arm was measuring task
difficulty, not damage.

**Repair-per-event never resolved** (6–8 of 10 rounds in range, per-arm slopes −0.004 /
+0.015 / −0.257, values swinging −3.8 to +1.16) and is reported as out of range.

## 5. E3 — expansion ([`expansion.py`](expansion.py)): grader type is everything, drift is orthogonal

§5's 2×2 with the missing cell filled. 3 seeds, entropy-matched base, graded in three worlds.

| condition | PR r1 → r10 | last-3 PR slope | d3 | d4 | ballistic (base) | fresh-FM top1 (base) |
|---|---|---|---|---|---|---|
| frozen_static | 6.99 → 6.93 | +0.003 | 0.583 → 0.572 | 0.660 → 0.619 | 0.024 | 0.173 |
| dense_static | 6.31 → 6.61 | +0.011 | 0.563 → 0.557 | 0.634 → 0.626 | 0.007 | ~0.20 |
| **evaluative_static** | **9.03 → 11.56** | +0.080 | 0.644 → 0.677 | 0.747 → **0.812** | **0.369** | **0.554** |
| frozen_drift | 6.93 → 6.84 | −0.006 | 0.584 → 0.571 | 0.671 → 0.628 | 0.027 | 0.177 |
| dense_drift | 6.32 → 6.62 | +0.041 | 0.569 → 0.546 | 0.645 → 0.618 | 0.008 | ~0.20 |
| **evaluative_drift** | **8.94 → 11.73** | +0.077 | 0.635 → 0.687 | 0.766 → **0.815** | **0.380** | **0.547** |

Round-0 belief: PR 7.04 ± 0.05, d4 0.665 ± 0.023.

> ⚠️ **Correction to this section's framing, 2026-08-01 — the PR column does not rank targets, and
> two children found this independently.** [`endogenous_expansion/`](endogenous_expansion/README.md)
> found it correlationally: belief PR **anti**-correlates with target quality (ρ = −0.80 against
> fidelity; the exact DP teacher has the *lowest* ΔPR of its four target arms).
> [`endo_expansion/`](endo_expansion/README.md) §3 found it causally, with the content control the
> other node did not run: an arm whose target is a **uniform random block** (agreement with `k*` at
> chance, 0.072 vs 0.071) reaches **PR 13.5 — 145% of the DP teacher's PR lift, the largest in that
> experiment** — with ballistic 0.016 and fresh-FM top1 0.129, i.e. exactly at the no-loop floor.
> Across nine arms, Spearman(PR, ballistic) = **+0.27**, against +0.75 for d4 and **+0.87** for
> fresh-FM top1. Two independent implementations, different arms, same verdict.
>
> **The mechanism the two agree on**: the plan CE feeds gradient to every block latent through the
> FM and the value regardless of which move is labelled correct, so PR reads *the loss pathway
> being present*, not the target being good.
> **The finding below stands** — it also rests on ballistic (15× the floor) and fresh-FM
> plannability (3×), both clean under both controls, and `frozen`/`dense` genuinely have no plan
> term. What does not stand is leading with PR, and §9's caveat *"no PR rise with flat depth is
> read as expansion"* is **not a sufficient guard**: the random-target arm raises PR *and* d4
> (+41% of the lift) and produces nothing.

1. **The grounded evaluative grader expands, hugely.** PR 7.0 → ~11.6 (1.7× the belief's effective dimension),
   d4 +0.15, transferable fresh-FM plannability 0.17 → 0.55 (**3×**), ballistic control 0.024 → 0.37
   (**~15× the no-loop floor, ~50× the dense grader**). Stable at 3 seeds, in all three grading worlds, under
   both drift anchors.
2. **The endogenous dense grader caps *below the floor* and degrades depth.** PR 6.6 against frozen's 6.9;
   d4 falls in both worlds; ballistic collapses to 0.007. This is Stage 5's `fm_cotrain` result (PR flat
   6.7 → 6.8, transferable plannability *down* 0.357 → 0.304) reproduced — and now with a drift arm showing
   drift does not rescue it.
3. **Drift is orthogonal.** Evaluative − frozen lift in the *same* world: ΔPR +4.63 ± 0.46 (static) vs
   +4.89 ± 0.73 (drift); Δd4 +0.193 ± 0.016 vs +0.187 ± 0.015. Statistically identical. §5's stated falsifier —
   *"falsified if the evaluative/drifting cell also exhausts in one pass"* — is met: both cells have the same
   PR slope at r10 (+0.080 vs +0.077) and land at the same place.

**The exhaustion law is about compression, and drift is not the escape.** The three prior discoveries of
"compounding requires a moving frontier" were all measured with a dense grader in the loop, and swapping the
grader type is what moves the frontier. Making the *world* move does not add to it.

## 6. Necessity — what the climb null was actually missing ([`climb.py`](climb.py) sweep)

The §4 null was challenged on its logic, and the challenge is right. Unpacked, the homeostatic
argument is:

1. Higher levels are more invariant under surface drift. *True by construction — we chose which
   level drifts.*
2. Therefore they are cheaper to maintain, integrated over a drift sequence. *Follows from 1.*
3. Therefore a learner will climb. **Does not follow at all.**

Step 3 needs an objective that prices the *integral*. SGD prices instantaneous loss. A learner
that re-fits the surface after each event is locally optimal at every single moment, pays full
price forever, and never sees a gradient telling it so. §4 tested step 3 with no mechanism to
produce step 3.

**And the premise was measurable, and true.** The surface arm repaired **91%** of its damage
inside the round and converged to the `nodrift` arm's root CE every round. Invariance was free
for everyone, so it bought nothing measurable — *the agent that never climbed was not paying
for the deep level, it just did not need one.* **Invariance ≠ necessity.**

So the knob is not drift magnitude (already exact) but **samples per event**: hold KL/event
fixed and starve the repair data, until the cheap repair stops being surface re-fit and becomes
deep-prior-plus-few-samples — which is specialization's own line, *"a learner that already holds
level ℓ recruits ℓ+1 from a small marginal sample."* Every per-event budget scales together,
monitoring included, each budget carrying its own matched `nodrift` control. 3 seeds.

### The advantage migrates from the surface to the deep levels as data is starved

`d1` = root … `d4` = the block's own feature. Gain = (arm's depth change) − (its matched control's).

| samples/event | deep gain (d1,d2) | surface gain (d4) | deep − surface |
|---|---|---|---|
| 4096 | −0.0049 ± 0.0060 | **+0.0252 ± 0.0076** | −0.0301 ± 0.0115 |
| 1024 | +0.0066 ± 0.0018 | +0.0210 ± 0.0132 | −0.0143 ± 0.0115 |
| 256 | +0.0042 ± 0.0036 | +0.0112 ± 0.0049 | −0.0070 ± 0.0070 |
| 64 | **+0.0142 ± 0.0108** | +0.0077 ± 0.0243 | **+0.0064 ± 0.0196** |

Per-seed slope of (deep − surface) against log₂ samples/event: −0.00635 / −0.00741 / −0.00379
per octave. **Mean −0.00585 ± 0.00186, t = −5.44, 3/3 same sign.**

At the abundant end the drifting arm gains only at the **surface** and nothing deep — which is
what "surface re-fit suffices" looks like. At the starved end the gain is **deep** and the
surface gain has collapsed. Monotone across four budgets at exactly-matched drift magnitude.
**This is the first positive this arc has produced on the climbing axis.**

The depth probe is a clean instrument for it: it reads a **frozen** sequence set with fixed
ancestor labels and does not depend on which world an arm currently sits in. And the obvious
alternative — *drift is just augmentation, and augmentation makes better representations* —
predicts a **uniform** lift across levels, not a budget-dependent migration. The migration is
what discriminates.

### What still does not report, for the fourth time

Cross-arm root-CE **damage is negative at every budget** (−0.009 to −0.012), and the drifting
arm ends with *better* root CE than its matched control (−0.038 to −0.045). Damage cannot be
negative if it is damage, so `repaired_fraction` is meaningless here and the `unrepaired`
column is not a number. That is the entropy identity's fourth appearance in this node, and it
means **§6's two-instrument read has still never been achieved** — the migration result rests
on the depth probe alone, which §6 explicitly warns against. A genuinely different repair
readout is needed; the cross-arm root-CE difference has now failed for a different reason each
time it has been tried.

### Honest limits on this result

- **Magnitudes are small** (0.006–0.030 in probe accuracy) and individual cells are 1–3 SE.
  The *trend* is the claim; no single cell is.
- **The drive changed at the same time as the sweep** (`value` → `value_red_satiety`), so the
  S=4096 cell is not comparable to §4's (Δd4 −0.026 → +0.025). Within this run every budget
  shares the drive, so the trend is clean, but that cross-run difference is not attributable.
- **§12's mechanism still has no axis.** Satiety is in the loop now, but it allocates over
  *channels*, which are not ordered by level (§3). So this sweep tests **necessity**; it does
  not test §12.
- **The second route is unbuilt.** The other way to make surface repair insufficient is to make
  drift *aliasing* rather than merely different — destroy identifiability at the surface so the
  level above becomes the disambiguator. The cheapest principled version in RHM is
  context-conditioned bottom-level mixtures (synonym weights indexed by the grandparent
  feature), which keeps the support fixed so `d*` stays invariant but makes the RHM
  context-sensitive and needs the closed-form KL machinery extended past its `(v, m)`
  assumption. Specced, not built.

## 7. Two instrument bugs, and why they matter more than the results

Both were found by chasing an apparent positive, and both had biased toward it.

### The own-world artifact

A drift-trained arm was being graded in the world it had just spent 20 OU steps fitting, while the static arm
was graded in a pristine one. Grading every arm, frozen, in a common world:

| grading world | evaluative_drift − evaluative_static, Δballistic | t |
|---|---|---|
| **own** | **+0.0243 ± 0.0196** | **+2.15** |
| pristine | +0.0081 ± 0.0199 | +0.71 |
| novel (held-out realisation) | +0.0098 ± 0.0214 | +0.79 |

~65% of the apparent advantage was the grading venue.

### The uniform anchor

The OU walk anchored at `θ = 0` — the uniform mixture, which is the **maximum-entropy point of the simplex**.
Since `KL(w‖uniform) = log m − H(w)` exactly, a drift event destroys precisely as much rendering entropy as the
KL it creates. So every drifted world has fewer synonyms in play per edit, a lower stochasticity floor, and is
**mechanically easier to plan in** — measured at a fixed controller, tree floor 0.160 (drifted) vs 0.195
(uniform), ~18%. A static arm parked at uniform is not a control; it is a handicap.

*The sign is an artifact of the anchor, not a property of drift.* Anchor at uniform and drift can only reduce
synonymity; anchor near a deterministic mixture and it could only increase it. Drawing the pre-drift world from
the same stationary law the walk lives in (`init="stationary"`, mechanically a σ=0 drift state) makes the two
arms' worlds exchangeable in entropy, so only the *motion* differs.

With both removed, at 3 seeds:

| grading world | Δballistic | Δtop1 |
|---|---|---|
| base | +0.0107 ± 0.0458 (t=0.40) | −0.0070 (t=−1.03) |
| novel | +0.0080 ± 0.0433 (t=0.32) | −0.0099 (t=−1.13) |

Per-seed ballistic deltas `[+0.017, +0.053, −0.038]` — the sign flips. Zero. And the symmetric question no
earlier design asked: a *static*-trained agent does not degrade in a novel drifted world either (base→novel
costs static −0.024, drift −0.027). **Drift-training bought no transfer in either direction.**

### The third appearance of the same identity

`KL(w‖uniform) = log m − H(w)` has now bitten this node three times: it is why raw CE is *exactly* blind to
support-fixed drift (`../README.md` Trap 1), why the drifted world is easier to control in, and why a
uniform-anchored control is not a control. It is worth treating as a standing property of this drift primitive
rather than a recurring surprise.

### And a bug in the readout rather than the design

`calibrate_sigma` matches KL from uniform *at stationarity* — accumulated displacement, not the size of an
event. The climb sweep, whose entire purpose was holding magnitude fixed, first ran at per-event magnitudes of
0.78 / 1.00 / **3.49** nats across levels. `calibrate_sigma_event` targets the event directly and lands all
four levels at 0.600. Both climb corrections removed biases pointing *toward* a climb, so §4's null survived a
thumb on the scale in its disfavour.

## 8. What this establishes

1. **A reward-free relevance signal ports from mjc to RHM and is strong.** The learned value devalues
   structurally irrelevant tokens 13–18× without ever being told which they are, agrees with the exact DP, and
   allocating by it alone recovers 76% of a privileged oracle.
2. **Expansion is a property of grader *type*.** Grounded-evaluative expands (PR 1.7×, plannability 3×,
   control 15×); endogenous-dense caps below the no-loop floor and degrades depth. Reproduced at 3 seeds in
   3 worlds under 2 anchors.
3. **Drift does not sustain expansion**, and **drift alone does not induce climbing** — with both instruments
   corrected in the direction that disfavours the null. But the climb null is scoped: it holds *where surface
   re-fit suffices*, which was 91% of the damage per round.
4. **Necessity, not invariance, is what the climbing argument was missing.** Starve samples per event at fixed
   drift magnitude and the drifting arm's depth advantage migrates from the surface to the deep levels,
   monotonically across four budgets (t = −5.44, 3/3 seeds). Small, single-instrument, but the first positive
   on this axis.
5. **A floor-corrected reducibility tap fixes the `e` tap**, −0.054 ± 0.009 against the counterfactual fit with
   an 8× drop in allocation variance — and shows the ladder cannot separate `value` from `visits_only` by
   construction, because this geometry has no visited-but-irreducible cell.
6. **Satiety does what §10 predicts** (irreducible leak 5.3% → 0.8% of the learned budget) **and exposes that
   §12 needs a level-ordered allocation space**, which a channel-indexed budget is not.
7. **A metered, on-policy loop is reproducible on RHM at E3's own subsidy ratio** (1.78–1.89× vs 1.84×), with
   the ballistic/reactive split behaving exactly as the length-gen arc predicts.

## 9. Caveats

- **Repair-per-event has never reported.** Out of range in §4, sign-broken in §6. Every climb conclusion rests
  on the depth probe alone, against §6's own two-instrument requirement.
- **The `root` drift arm was measuring task difficulty, not damage** (§4) and should not be read.
- **The migration result is small and single-instrument** (§6), and its cross-run comparison is confounded by a
  simultaneous drive change.
- The ladder is now 3-seeded and the climb sweep is 3-seeded; §4's original 4-arm climb is single-seed.
- **`own`-world grading remains favourable by construction** even after the entropy fix (the arm just fitted
  that world); `base` and `novel` are the fair columns and are the ones quoted.
- **L=4, not the L=5 the port was specced for.** L=5 was run through `env_check` and agrees on every DGP-level
  property (floors ordered, value separation 0.61 vs 0.05, top-1 share 94.5%); L=4 was chosen because the whole
  sculpting arc is calibrated there and the loop is ~3× cheaper.
- **PR is not certified as "the frontier"** — the idea doc's own warning. It is reported alongside depth
  throughout, and no PR rise with flat depth is read as expansion. **Superseded 2026-08-01 and the
  guard is now known to be insufficient**: [`endo_expansion/`](endo_expansion/README.md) §3 shows a
  zero-information target raising PR *and* depth while producing no control at all, so PR must be
  read against a content control, not against depth. See the correction box in §5.

## 10. Children

| Child | What |
|---|---|
| [`endogenous_expansion/`](endogenous_expansion/README.md) | **The central gap closed, weakly — and §5's headline metric retired as a ranker.** §5's evaluative grader is external and precomputed; §3's `visits` is endogenous but was only ever tested for *allocation*. This runs the missing cell: 7 arms differing in **one variable, where `belief_update`'s target comes from**, with `frozen`/`dense`/`evaluative` reproduced in-run as anchors. An endogenous MC critic — no DP, no rule table, no channel label, paying 10.08M rollout transitions for terminal-success labels — **does expand** above the no-loop floor on all four readouts, 3/3 seeds in two replicates, but recovers only **19%** of the DP teacher's ballistic lift (66% depth, 51% transfer). Two *external* teachers degraded to matched fidelity bracket it, and it beats them: **the residual gap is target fidelity, not endogeneity**, and §8's mirror-grader hazard did not appear. The mechanism is one number — policy move-quality converges to its target's fidelity *from whichever side it starts* (DP 0.674→0.748 up, critic 0.400→0.333 down), so **a grader is a ceiling**. Two findings that cut against this node: **belief PR anti-correlates with target quality** (ρ = −0.80; the exact DP has the *lowest* ΔPR of the four target arms), and `R_res_participation` does not replace it — it is 1.1× `R_act` and inherits the inversion. Every dimension-*count* fails; only *magnitude* measures track function, with a compression sign. PR remains a valid **detector** (clean target-present/absent separation, which is the regime §5 used, so §5 stands) and an invalid **ranker** |
| [`endo_expansion/`](endo_expansion/README.md) | **E4 — can an *endogenous* evaluative grader expand?** The metered-data doc names this the program's central gap: §5's teacher expands but is external and precomputed, §3's `visits` is endogenous but was only ever allocated, and no run was both. Nine arms on one warm start. **It expands**: a Monte-Carlo teacher reading only the environment's terminal reward — no DP, no rule tables, no channel labels — recovers **34%** of the DP teacher's ballistic lift and **53%** of its transferable plannability, 3/3 seeds, and its rank-correlation with the dense inner grader (**−0.130**) sits on the *external* teacher's side (−0.220) of a homogeneous floor of **+0.946**, so the two loops did not collapse into one. Two further results: endogenous *allocation* of an external target works but is beaten by endogenous *target generation* (+0.069 ballistic, t=+5.8), and §7's paid-vs-reported design rule comes back a scoped negative. **And the correction in §5 above**: belief PR is anti-informative on this substrate |
| [`metering_sweep/`](metering_sweep/README.md) | **The metering falsifier, run — and the ladder is over-trained.** §3's ladder reports `meter_ratio` = 1.78125 in *every* run in the repo, so [`ideas/meta_learning_under_metered_data.md`](../../../../ideas/meta_learning_under_metered_data.md)'s own load-bearing test had never been attempted. It needed a design change first: the `Meter` is **inert** (built with `budget=None`, read by nothing), so a bare charge multiplier is a provable no-op, and `uniform`/`oracle` are both **tap-blind**, so the prize can respond to monitoring only through the collection it displaces. 87 runs over two pre-registered rounds. **Three results this node should carry.** (1) **`fm_epochs=8` is well past optimal**: 16× the gradient steps on the *same* data costs **+0.117** tree error. The published ten-arm ladder re-run at 1/2/4 epochs puts **every arm 0.03–0.07 better**, and the headline uniform→oracle prize falls **0.0629 → 0.0521** (~17% inflated by over-training) — but **every published contrast survives and the ordering is unchanged** on the arms that carry conclusions (the tap repair is *larger* at −0.0625; `value_red`/`visits_only` tie *exactly*; the recovery split and the 8× allocation-variance drop both reproduce). The science holds; the absolute levels and the prize should be requoted. (2) That confound is what made the prize look like it *grew* as data got cheap; at **matched compute** the prize is **humped and turns over** at the abundant end (−0.0190, t = −6.10 and −0.0221, t = −21.67 at two compute levels, 3/3 seeds), with `E(d)` flattening to −0.007/doubling — the doc's predicted limb, and its claim supported under the iso-compute scaling its own analogy implies. (3) **Priced, the reducibility tap is worth nothing.** Charging each arm only for the taps its drive reads, `visits_only` pays 512 sequences/round (2.2% of the total) and keeps **87%** of what a privileged oracle gets free, while `reducible_only` pays 14080 and recovers **4%** — falling to −0.0001 at the largest total. A ~1200× difference in cost-effectiveness, and the whole of its apparent 13–18% was the equal-charge subsidy. Also: drift magnitude does **not** move the prize (2×2, ±0.0009 over 30×) while budget does (+0.023, t = +5.2); the monitor:collect **ratio is not the operative variable** (matched-ratio pairs differ by +0.023 while 4× monitoring at fixed budget is bit-identical to ten decimals); and the prize splits into **interference** (falls to 9% of the total by B=262144 — the doc's intuition, vindicated) and **data quantity**. Anchor reproduces `ladder_fix_s*` to +0.0004 |
| [`level_moves/`](level_moves/README.md) | **⚠ Corrected 2026-08-03 — read the child's §9–§12 first.** The level *magnitude* profile below is span (a depth-matched distractor with Δ`d*` ≡ 0.000 rises **2.79×** L1→L4 against the tree's **2.56×**), and the per-node top-1 headline is one seed of three (per-seed L4 share 0.400 / 0.041 / 0.119). What replaces them is a **matched-span paired control** — each committed level-ℓ move paired with a *lazy twin* rewriting identical tokens without committing, so the confound cancels inside the pair. On that readout the value genuinely prefers the abstract commitment (**0.557 ± 0.018** at L2, **0.618 ± 0.052** at L3, against an exact oracle's 0.610 / 0.642), and it is task-referenced rather than grammaticality (**4.4%** of the premium survives where legality buys Δ`d*` = 0). The finding the node now carries comes from a DGP repair: the published damage is random symbols, **off-grammar in 29% of tree blocks** and so repairable one block at a time, meaning abstraction never had to matter. Under **hierarchical damage** (a level-k subtree swapped for a legal derivation of a feature it provably cannot produce — 100% on-grammar at matched `d*`), the value's abstraction premium is **graded by whether a commitment at that level can reach the error**: L2 slope **−0.033, t = −6.81**, 3/3 seeds, against the oracle's −0.025, t = −4.14, while L3's oracle slope is half that. And value–oracle rank agreement roughly **doubles**, +0.290 → **+0.583** (t = +8.14), while terminal success *falls* 0.475 → 0.307 — which is the internal control against "the value just got flatter". The value is systematically conservative, and the shortfall widens with depth (L2 −0.053 → −0.078; L3 −0.024 → −0.097). **So: sensitivity to the hierarchy's compositional structure, not to depth as such.** Two by-products worth carrying: §3's relevance separation now holds at **all four levels against a full depth-4 distractor** (12.1× / 13.1× / 12.7× / 11.1×), the one finding that came out stronger; and the **root cell is structurally unreadable** — a level-4 move masks every tree token, so its Δ`d*` premium is bit-identical across all three damage levels, which is why it never resolved and why more seeds cannot fix it |
| [`level_moves/`](level_moves/README.md) (as first published) | **A level-indexed action space — open item 2's missing axis, half built.** §3's smoking gun says the value separates real tokens from distractors 13–18×, and then says nothing about the RHM's *hierarchy*, because the action space is flat: `n_blocks = seq_len // s`, so a block **is** one level-1 feature and `value_sensitivity` is aggregated per channel. This adds a move that commits to one level-ℓ **feature** and renders the whole legal subtree beneath it (exactly reducing to `regenerate_block` at ℓ=1, asserted; grammatical above it, asserted). Read on it, the exact DP says **depth raises the floor, not the ceiling** — the *best* move is nearly flat across levels (+1.74 → +1.88 in Δ`d*`) while a *random* one rises **+0.283 → +1.884**, 6.7×, because one abstract commitment re-derives a legal subtree without having to find the right block. **The value tracks this**: per-node top-1 preference relative to level 1 runs 1.0× / 1.8× / 3.1× / **5.0×** against the DP's 1.0× / 0.9× / 3.6× / **8.2×** — real, monotone, correctly ordered, and calibrated ~40% low at the root. Two controls carry it: a **span null** (distractor channels, Δ`d*` exactly 0.000 at every level, still show \|ΔV\| rising 43% from L1 to L2) which shows a naive \|ΔV\|-by-level reading is substantially the branching factor, and the DP's own top-1 distribution, which is what "under-shoots" is measured against. **Relevance separation is level-invariant** (8.3× at L1, 8.9× at L2), so §3's headline holds at every scale of move. Training the value *on* the axis makes the policy clearly better (terminal success 0.400 → 0.523, t = +18.1) and the value less oracle-like (rank corr 0.284 → 0.169) — reported but not concluded from, because ΔV under a state value depends on the policy it was trained under, so the arms differ in reference policy as well as action space. **This is the action space and the readout, not the allocation space**: open item 2 proper needs an FM over spans, and no FM, planner or allocator was touched. **Child added 2026-08-04**: [`level_moves/level_ladder/`](level_moves/level_ladder/README.md) builds that allocation space — span FM (bit-identical to the block FM at level 1), a budget over (channel, level) cells, a level-indexed planner and ballistic grader, and a damage schedule that deepens the error across the run — and finds **the level index inert as an allocation axis** (+0.00016 ± 0.00226, t = +0.13, against a paired floor of ±0.0009; a level-blind oracle takes 99.2% of the prize) **and large as an action axis** (ballistic 0.121 → 0.424, **3.51×**, t = +10.40) |
| [`composed_loop/`](composed_loop/README.md) | **The composed loop (2026-08-05)** — [`endo_expansion/`](endo_expansion/README.md)'s endogenous MC judge run on [`level_moves/`](level_moves/README.md)'s level-indexed action space under hierarchical damage, the combination three sibling nodes imply and none had run. Adds `belief_update_moves` (the plan term stacked over *moves* through the span FM; reduces to the published block term at `max_level=1`) and a **block-matched** dense-term move draw so both action spaces see an identical channel mix — without it, "level moves help" would be partly "the level arm saw more tree". **Single seed throughout, and nothing here is resolvable against seed variance.** The headline 2×2 is a large negative interaction (−0.264 at the deepest damage): the endogenous judge's level gain **+0.189** is the no-loop arm's **+0.186**, against the external DP teacher's **+0.453** — so ~40% of the level action space's value is free (planner search + FM rollout), the rest needs a good judge, and the endogenous one captures none of the surplus. Two results survive their own controls. **(1) The endogenous judge has no upward direction**: its mean target level is flat at 1.72/1.72/1.73 while the exact DP climbs 2.23 → 2.48 → 2.80 on the same states, and re-running at [`endo_expansion/`](endo_expansion/README.md)'s published teacher budget separates the two axes cleanly — accuracy improves (2.65× → 3.16× chance, `informative_frac` 0.898 → 0.921) while altitude moves 0.06. Suggested mechanism, argued not measured: terminal task success is a scalar over a short horizon, so more rolls reduce its variance but nothing in it is *about* depth. This is [`level_moves/`](level_moves/README.md) §10's *"error-depth matching, not a depth ordering"* arriving from the **teacher** side. **(2) The value's exploration distribution is a large, cheap lever**: varying only which behaviour policy generates the value's training states moves no-teacher ballistic **0.316 → 0.479 (+52%)** — 45% of what the whole privileged teacher is worth on that arm — under a comparison where the controller comes out **bit-identical** across settings. Two pre-registered predictions failed there, and the instrument built to catch one of them (`planner_mean_level`) **excluded** the explanation rather than confirming it. An external ~14-bit-per-run **level-floor instruction** ("aim higher") moves the judge's target level 1.72 → 2.47 and lifts it from below to above the no-loop floor — but its **reversed-schedule control does as well or better**, refuting the curriculum reading the arms were built for and leaving only a post-hoc pattern; at the matched teacher budget the effect on control is inside noise. Also reproduces, more starkly than published, that **PR and depth are anti-informative** (a zero-information target posts the run's highest PR 13.3/13.6 and highest d4 0.687/0.699 with the *worst* control, 0.033) and that **level moves cost self-predictability while buying control** |
| [`partial_hetero/`](partial_hetero/README.md) | **Cut-3 — the partially-heterogeneous DGP**, built and running. §3's ladder measures allocation in a geometry where the distractors share *nothing* with the tree, so ground-truth relevance is exactly 0.000 off-tree and the selector faces no judgment call. `partial_hetero` adds a sharing-depth knob: `structA` is depth-matched to the tree and takes its top *k* rule tables, giving a sweep from this node's independent-grammar geometry (*k*=0) to [`specialization`](../../specialization/README.md)'s single-ruleset one (*k*=4). Structural irrelevance is untouched at every depth — what changes is what the distractor's data is worth to the *learner*. Two measurements: a static coverage-matched **transfer curve**, and this ladder re-run at each depth with a **`oracle_shared`** rung that separates *stipulated relevance* from *actual data value*. Built, run and written up in the child — read it there; the result is deliberately not summarised at this altitude, because what it mainly establishes is about the *readout* rather than about this node's findings |

## 11. Open items

1. **A repair readout that actually reports.** The cross-arm root-CE difference has now failed three times for
   three different reasons. Until one works, §6's two-instrument discipline is unavailable and every climbing
   claim here is half-instrumented.
2. **A level-ordered allocation space**, so §12's satiety-gated *"recruit ℓ+1"* has an upward direction to
   recruit in. Currently it recruits sideways into a distractor (§3). **Half done**:
   [`level_moves/`](level_moves/README.md) builds the level-indexed *action* space ~~and shows the
   value's level preference is real and monotone (5.0× per-node at the root against the oracle's
   8.2×)~~ — **corrected 2026-08-03**: that profile was span and seed spread. What the value does
   have is an **abstraction preference graded by error depth**, which is not the same thing as a
   depth ordering: §12's *"recruit ℓ+1"* wants a signal that rates higher levels higher, and the
   value supplies one that rates the level *matching the error* higher. **So §12 still has no
   endogenous upward direction — but that is now a measured constraint rather than an untested
   gap.** **The allocation half is now built too, and the axis is inert —
   [`level_moves/level_ladder/`](level_moves/level_ladder/README.md) (2026-08-04).** The span
   forward model that was the standing blocker turned out small (`BlockLatentFM`'s target is
   already the full latent delta; only the conditioning is per-block) and reduces to the published
   FM bit-identically at level 1. With a budget over (channel, level) cells, a level-indexed
   planner, and a damage schedule that moves the error *deeper across the run* so climbing is
   necessary, every gate passes and a privileged allocator climbs on schedule (mean level of tree
   spend 2.16 → 2.42 → 2.68, slope +0.287 ± 0.038, t = +13.15, 3/3 seeds) — **and it buys nothing**:
   holding the channel allocation fixed, the level index is worth **+0.00016 ± 0.00226 (t = +0.13)**
   against a paired floor of ±0.0009, with a level-*blind* oracle recovering **99.2%** of the
   +0.0217 allocation prize. Proposed mechanism: channels are *disjoint* sets of positions, levels
   are *nested* on the same blocks, so no level can be starved by spending at another. **The
   complementary contrast is large** — swapping only the *action* space from block-only to
   level-indexed, at matched task and allocation, moves ballistic control 0.121 → 0.424 (**3.51×**,
   t = +10.40) while making the FM *worse* on its dense proxy at the one shared cell (−0.106,
   t = −3.14). So §12's difficulty is not the missing axis: **on this geometry levels are an
   action-space axis, not an allocation-space one.**
   **Extended 2026-08-05 by [`composed_loop/`](composed_loop/README.md)**, which asks the same
   question of the *teacher* rather than of the value and gets the matching answer: an endogenous
   MC judge's target level is flat at ~1.72 while an exact DP climbs 2.23 → 2.80 on the same states,
   and sharpening the judge buys accuracy without buying altitude. So §12's missing upward direction
   is now measured on both the value side and the teacher side, and the suggested reason is the same
   one in both cases — a scalar outcome signal carries no depth information. An external level-floor
   instruction supplies the direction and moves the target level to 2.47, but its reversed-schedule
   control does as well or better, so *why* it helps is unresolved.
3. **Aliasing drift** — the second, unbuilt route to making surface repair insufficient (§6).
4. **The in-tree irreducible cell** is still unplaced (inherited from `../README.md` open item 1), which is why
   `value_red` and `visits_only` tie by construction.
5. **Why does the dense grader land *below* the no-loop floor?** Stage 5 saw it cap; here it actively degrades
   depth and control in both worlds. Worth understanding before "endogenous predictability pressure is merely
   insufficient" is restated as "actively harmful". **Partially answered 2026-08-01**:
   [`endo_expansion/`](endo_expansion/README.md) §3 lands two *non-dense* arms below the floor too
   (a shuffled and a random target, ballistic 0.011 and 0.016 against frozen's 0.024), so this is
   about reshaping the belief toward a target that does not track the task rather than about
   density. Suggestive, not settled — those deltas are t = −1.9 and −1.1.
6. **Push the sweep past S=64.** The migration crosses zero at the starved end but the deep gain is still only
   +0.014; whether it keeps growing or the whole system collapses first is unmeasured.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                                   # the gate
modal run rhm/directed_sculpting/full_loop/env_check.py::env_check --quick                        # smoke
modal run --detach rhm/directed_sculpting/full_loop/env_check.py::fm_bisect --tag l4_v1           # back-compat
modal run --detach rhm/directed_sculpting/full_loop/env_check.py::env_check --tag l4_v2 \
    --tree-depth 4 --struct-depths 2,2 --noise-blocks 1,1 --edit-budget 6 --chunk-sweep 1,2,3,6
modal run --detach rhm/directed_sculpting/full_loop/ladder.py::ladder --tag l4_v1 \
    --rounds 12 --n-eval 1024 --mon-n 3000 --collect-budget 8192
for s in 1 2 3; do   # the necessity sweep (samples/event at fixed KL/event)
  modal run --detach rhm/directed_sculpting/full_loop/climb.py::climb --tag nec_s$s --seed $s --rounds 10
done
modal run --detach rhm/directed_sculpting/full_loop/climb.py::climb --tag em_v1 --rounds 10 \
    --arms nodrift,surface,mid,root --policy value --samples-sweep 4096      # §4's original
for s in 1 2 3; do                                    # launch each seed as its own client
  modal run --detach rhm/directed_sculpting/full_loop/ladder.py::ladder --tag fix_s$s --seed $s \
      --rounds 12 --n-eval 1024
  modal run --detach rhm/directed_sculpting/full_loop/expansion.py::expansion --tag em_s$s \
      --seed $s --rounds 10 --grade-every 3 --fresh-fm-steps 5000 --n-eval 1024
done
```

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/`, mirrored to
[`figures/`](figures/). `--entropy-matched-base false` and the default `calibrate="stationary"` reproduce the
pre-fix runs (`expansion_xw_s*`, `climb_l4_v1`) exactly.
