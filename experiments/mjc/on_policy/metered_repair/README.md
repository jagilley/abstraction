# Metered repair: what to collect (E4) and what to represent (E5)

**Up**: [`../README.md`](../README.md) (on_policy) · **Node**: [`../../README.md`](../../README.md) (mjc) · **Files**: [FILES.md](FILES.md)
**Direct parent**: [`../directed_on_policy/`](../directed_on_policy/README.md) (E3, PR #4) — E4 is a copy-and-modify of its ladder; E3's `ladder_s{0,1,2}` are untouched.
**Ports from**: [`rhm/directed_sculpting/full_loop/`](../../../rhm/directed_sculpting/full_loop/README.md) (PRs #15/#16) — §3's tap repair and §6's necessity sweep.
**Idea docs**: [`ideas/meta_learning_under_metered_data.md`](../../../../ideas/meta_learning_under_metered_data.md) · [`ideas/breadth_as_grader_heterogeneity.md`](../../../../ideas/breadth_as_grader_heterogeneity.md)
**Status**: both cuts built, gated and run. E4 = 10 arms × 26 rounds × 3 seeds. E5 = 3 arms × 4 budgets × 16 rounds × 3 seeds. Single task family (planar arm, curl drift, reaching). **Date**: 2026-07-30.

---

## One-liner

Two RHM claims were carried onto the on-policy arm. **One ports, one does not, and one is
unanswerable here.** E4's *diagnosis* ports exactly — E3's learning-progress tap is the same
fixed-budget counterfactual fit RHM found inverted, and it is inverted here too (separation
**−0.010 ± 0.008** against the floor-corrected tap's **+0.108 ± 0.017**) — and the repaired tap
allocates far better by every process measure (leak into irreducible regions **57.8% → 19.7%**).
But **the repair does not cash out**: `value-red` − `value` = **+0.029 ± 0.027, 1/3 seeds**, against
RHM's −0.054 ± 0.009, 3/3. What *does* land is the contrast neither substrate could run before:
with a **visited-but-irreducible region placed**, relevance alone becomes the ladder's *worst* arm
and the conjunction beats it by **−0.067 ± 0.014, t = −4.99, 3/3 seeds**. E5 returns a clean
negative with a mechanical diagnosis: even when two regions have **identical** curl gains at every
step (corr = +1.000), repairing one does not repair the other beyond what a never-drifting region
gets — so a fixed-DOF plant with spatially-gated fields gives an MLP no route to shared-parameter
inference, and the necessity question cannot be posed on it.

---

## 1. Why these two cuts

[`full_loop`](../../../rhm/directed_sculpting/full_loop/README.md) landed two things on E3's doorstep.

**(a) E3's reducibility tap is the estimator RHM found broken.** `directed_on_policy.py:559-567`
fine-tunes a fork on half the in-region monitor batch and reads the held-out drop. RHM ran exactly
that and found it ordered its channels *backwards* — lowest on the target (+0.044), highest on the
irreducible channels (+0.066) — because a fixed-budget fit measures **marginal return**, and early
marginal return is dominated by **data starvation** rather than reducibility. E3's geometry has the
same hazard by a different route: its off-reach distractors are the starved regions. E3's own caveat
is the signature — `value` pours **29%** of budget into the irreducible regions against `oracle`'s 0%.

**(b) The two substrates have mirror-image degenerate geometries.** RHM has no
visited-but-**irreducible** cell, so its `value_red` can only *tie* `visits_only` (+0.0019 ± 0.0030)
by construction. E3 has no visited-but-**irrelevant** cell ("you only go where you reach"), so
`visits-only` sits near the top. Between them the conjunction `lprog × visits` has **never** been
shown to need both terms.

E5 asks a different question — whether RHM's necessity result (starve samples/event at fixed damage,
and repair migrates from surface to deep) is about the shape of the repair problem or about RHM's
grammar. mjc offers a readout RHM cannot have: RHM's repair instrument has failed **four** times
because `KL(w‖uniform) = log m − H(w)` breaks its cross-arm root-CE difference, so every RHM
climbing claim rests on the depth probe alone. A *transfer* measurement is positive-definite and
immune to that identity.

## 2. What got built

Both scripts are copy-and-modify descendants of E3 rather than edits to it, per `CLAUDE.md`'s
preference for duplication over semantic mush. `arm_env.py` and `embodied.py` are **untouched**, so
[`../verify_backcompat.py`](../verify_backcompat.py) and every prior cut are unaffected.

**E4 ([`floor_tap.py`](floor_tap.py))** — four changes to E3's ladder:

- **A measured aleatoric floor.** `matched_pairs_floor` estimates it by k-nearest-neighbour matching
  in normalised `(s,u)` space, from the agent's own metered in-region batch. RHM measured its floor
  by *re-executing* the same `(x,k)` four times; `Body` forbids that outright (no `set_state`), and
  the RHM README says so itself. **Run it on the FM's residual, not on raw Δs** — this is the one
  choice that makes it work on a continuous substrate, because the curvature bias then covers only
  what the model has not already captured. Measured: on raw Δs the bias is **1.00–1.15**, *larger
  than the noise regions' true floor*; on the residual it is 0.36–0.46.
- **The stock tap** `red = (err − floor)/err`, floor **pooled across rounds** (it is stationary; the
  error is not) with neighbour matching still strictly within a round.
- **The visited-but-irreducible cell**, via a widened eval sector. E3 reported it unplaceable
  on-policy; it was unplaceable in E3's *narrow eval cone*. Widening the reach-goal distribution
  turns the visited set from a tube into a band and region C appears at visitation 0.42–0.56.
- **Both taps computed every round for every arm**, so the metered monitor charge is identical and
  only the allocation rule differs. E3's drift rate and budget are kept.

**E5 ([`necessity.py`](necessity.py))** — a two-level drift **generator**, `b_j(t) = c_j·β(t) + ε_j(t)`:
one shared latent β (deep — explains damage everywhere at once) plus independent per-region walks ε
(surface). Per-event damage is matched across arms **analytically** (`s_shared = scale·√f`,
`s_local = scale·√(1−f)` gives equal increment variance for every mixing fraction) and measured
anyway. Fixed allocation on region A, so allocation is not a variable; two instruments — repair at A
(surface) and at an off-reach region A′ that receives **zero** collection (deep); plus a `pure_shared`
calibration arm and a **frozen placebo** region whose gain never drifts.

> **No expansion claim is made anywhere here.** [`../../expansion/`](../../expansion/README.md)
> settled that a fixed-DOF plant has no hierarchy to expand into. E5 needs only that damage be
> repairable at two levels of *generality*, which a drift generator supplies. §5 is about how that
> turned out.

## 3. The gates, which caught more than the experiments did

Each script's gate is a strict **prefix** of the real run (`--certify-only`, `--calibrate-only`), so
the certified instrument is literally the one the experiment uses. They earned their keep:

| gate | what it caught |
|---|---|
| E5 probe count | A′ placed 0.81 m from the start tip against a 0.50 m reach band → **0/500** in-region probes. Every repair number would have been NaN, reading as "no effect". |
| E5 damage readout | `sd(Δb)` 0.714/2.117 against a construction guaranteeing 1.60 — the reflecting walk was **parking at its bounds**, so events silently stopped. |
| E4 batch sweep | the floor certified on ~600-sample probe sets but the loop reads ~18. Re-estimating at m = 600/200/60/24/16 gave separations **0.70 / 0.71 / 0.57 / 0.37 / 0.48** — flat, so the tap survives where it is used. |
| E4 probe count | seed 2 of the full ladder **aborted** (probe counts `[2365, 2292, 10, 174]`) rather than producing a NaN column. |
| E5 calibration arm | reported a **dead instrument** twice (dynamic range +0.007, +0.012) before the third replay design fixed it. |

**Reachability is dynamic, not kinematic**, and this cost three iterations. Every candidate region is
kinematically reachable by construction (it *is* the FK of some `qg`), but `collect_toward` plans one
14-step open-loop CEM sequence, so what binds is how far the tip can be *driven* in `ep_len` steps.
Measured directly: regions at 0.22 / 0.30 / 0.33 / 0.43 m from the start tip returned 272–1308
in-region probe transitions; one at 0.55 m returned **zero**. The band is now calibrated to that
(`reach_pad = 0.90`), with a density floor and an adaptive top-up behind it.

**The floor estimator's own certification** (3 seeds, matched-ceiling FM, off-budget probes):

| | bias on curl regions (true noise = 0) | floor on noise regions | separation |
|---|---|---|---|
| residual | 0.361 / 0.384 / 0.456 | 0.829 / 0.933 / 0.847 | **+0.469 ± 0.045** |

## 4. E4 — the ladder (10 arms × 26 rounds × 3 seeds)

Region-A FM error, mean over rounds, lower = better. Monitor:collect **1.82×** (E3's 1.84×).

| policy | A-err ↓ | leak(irred) | A-share | A-share sd |
|---|---|---|---|---|
| `value` = lprog × visits | **0.4151 ± 0.0149** | 57.8% | 39.9% | 0.347 |
| `oracle` (privileged) | 0.4158 ± 0.0099 | 0.0% | 97.0% | 0.010 |
| `lprog-only` | 0.4171 ± 0.0109 | 60.9% | 26.5% | 0.306 |
| `error-only` | 0.4442 ± 0.0130 | 66.3% | 10.3% | 0.027 |
| `value-red` = red × visits | 0.4444 ± 0.0167 | **19.7%** | 80.0% | 0.161 |
| `value-reddelta` (rate) | 0.4544 ± 0.0115 | 47.6% | 45.3% | 0.399 |
| `reddelta-only` (rate) | 0.4548 ± 0.0300 | 51.7% | 25.3% | 0.309 |
| `reducible-only` | 0.4850 ± 0.0398 | 32.1% | 43.0% | 0.121 |
| `uniform` | 0.4921 ± 0.0444 | 50.0% | 25.0% | 0.000 |
| `visits-only` | 0.5117 ± 0.0297 | 35.8% | 63.9% | 0.005 |

### 4a. The conjunction result — the one clear positive

**With a visited-but-irreducible cell present, relevance alone is the worst arm in the ladder** —
`visits-only` at 0.5117, *below uniform* — because region C is genuinely visited and it pours budget
in. Adding reducibility rescues it:

> `value-red` − `visits-only` = **−0.0673 ± 0.0135, t = −4.99, 3/3 seeds.**

In E3, without the cell, `visits-only` sat near the top (0.390 against uniform's 0.475). Placing the
cell is what makes both terms necessary, and it closes the mirror-degeneracy that made this
unaskable on either substrate: RHM's ladder cannot separate `value_red` from `visits_only` because it
has no visited-but-irreducible cell, and E3's cannot because it has no visited-but-irrelevant one.

### 4b. The estimator diagnosis ports; the repair does not

Tap ordering, reducible minus irreducible (positive = points the right way):

| tap | separation |
|---|---|
| **stock** (measured floor) | **+0.1076 ± 0.0167** |
| **flow** (E3's counterfactual fit) | **−0.0101 ± 0.0075** |

The flow tap is **inverted** here, exactly as RHM found. And the allocation consequences follow:
leak 57.8% → 19.7%, A-share 39.9% → 80.0%, allocation sd 0.347 → 0.161. Per region (`value-red` arm):
`red` reads 0.146 on the target and 0.043 / 0.042 on the two noise regions, while `lprog` reads 0.008
on the target and **0.023** on the on-reach noise region.

**And none of that produces a better outcome.** `value-red` − `value` = **+0.0293 ± 0.0273, t = +1.07,
1/3 seeds**, against RHM's −0.054 ± 0.009, 3/3. `value`, running on the inverted tap, **matches the
privileged oracle** (0.4151 vs 0.4158).

### 4c. Rate vs level — a hypothesis of ours, tested and not supported

The dissociation above suggested the two taps measure different quantities sharing a name: a
counterfactual fit is a **rate** (how fast can error fall here *now*), `(err − floor)/err` is a
**level** (how much is reducible in principle), and under continuous drift what pays is spending
where damage *just* happened. So we built a rate from the honest estimator —
`red_delta = max(Δerr, 0) × red` — and ran it two ways.

> `value-reddelta` − `value-red` = **+0.0100 ± 0.0183, t = +0.55, 1/3 seeds.** Not distinguishable,
> point estimate favouring the level. `value-reddelta` − `value` = +0.0393 ± 0.0261 — the honest
> rate is if anything *worse* than the inverted one it was built to replace.

**The mechanism is a cleaner negative than the arm outcome.** Mean `red_delta` per region over the
full run: A (target, curl) 0.0051, Con1 (noise) 0.0049, Boff1 (curl) 0.0047, Doff1 (noise) 0.0064.
**Flat, with the largest value on a noise region.** The tap carries no reducibility information at
all. The gate that was supposed to prevent this cannot: a region whose error is aleatoric has the
largest round-to-round `|Δerr|`, and multiplying by a level that is itself small does not reorder
them. *(A first version of this tap — deviation of `red` above its own EMA — fired in ~1 round in 6
and made both arms bit-identical; that is recorded because a silent tap is not a rate.)*

### 4d. What is left unexplained

`oracle` (97% of budget on target, 0% leak, sd 0.010) and `value` (39.9%, 57.8% leak, sd 0.347) reach
the *same* A-error by completely different allocations, while `visits-only` at 63.9% A-share reaches
the worst. **A-error is not a monotone function of how much budget the target gets, and no process
column in the table predicts the outcome column.** One candidate the data is consistent with and does
not establish: among *uninformed* taps, bursty allocation may beat steady allocation under continuous
drift — `value` and `lprog-only` are the two highest-variance arms and the two best, while `oracle`
needs no timing because it is told what is stale. Our rate tap produced burstiness (sd 0.399, highest
in the ladder) *without* targeting, which would explain why burstiness alone did not help. Untested.

## 5. E5 — necessity (3 arms × 4 budgets × 16 rounds × 3 seeds)

Every gate passes except the one that decides readability. Generator exact (increment-sd spread 1.7%,
`corr(b_A, b_A′)` = **+1.000 / +0.702 / +0.003** for f = 1.0 / 0.7 / 0.0); A′ genuinely uncollected
(in-region collection share **0.0022 ± 0.0007**). Then:

> **Instrument dynamic range (`pure_shared` − `local`) = −0.889 ± 0.917.**

Negative, with error bars larger than the effect. The calibration arm — built precisely to detect
this — reports that the transfer readout cannot see shared structure.

**The diagnosis is in the raw repair columns, not the ratio.** Per-round repair in the `pure_shared`
arm, where `c_load` is equal and the start is shared so **b_A and b_A′ are identically equal at every
step**:

| | S=432 | S=144 | S=48 | S=16 |
|---|---|---|---|---|
| repair at held-out A′ | +0.0010 | −0.0010 | +0.0023 | +0.0027 |
| repair at **frozen placebo** | +0.0050 | +0.0078 | +0.0097 | +0.0086 |

If the forward model had any route to representing "the curl gain" as a shared quantity, repairing A
would repair A′ for free. It does not — A′ repairs **less** than a region that never drifted at all.

**What this says, at the right altitude.** Not "metered data does not migrate repair upward". It is
that a fixed-DOF plant with **spatially-gated** local fields gives an MLP forward model no route to
shared-parameter inference: the gates are disjoint in tip space, so the model fits each locally and
there is no level for a meter to push it toward. The hierarchy in this design lives in the drift
**generator**, and the learner never sees the generator — only its per-region consequences. The
caveat written into the cut's own docstring turned out to be the result.

This extends [`../../expansion/`](../../expansion/README.md)'s conclusion (*"expansion plausibly needs
a domain with hierarchical structure to expand into, which a motor plant does not have"*) from the
expansion question to the **necessity** question, and it sharpens what RHM's result requires: a
learner whose representation *already* has level-ordered structure that a data budget can shift
weight between. RHM's grammar supplies that by construction; a curl field over a 3-link arm does not.

## 6. Caveats

- **The off-reach reducible region is effectively unmeasured in E4** — `n_in` = 0 in the live loop and
  ~53 in-region probe transitions even after adaptive top-up. So the `lprog-only` trap is weakly
  instrumented and `red` is the cross-region-mean fallback there all run.
- **Ballistic control separates nothing** (0.087–0.098 across all eight original arms against a
  near-saturated reference). Every E4 conclusion rests on the FM-error grader alone — E3's own
  blind-grader lesson, unresolved here.
- **A-error gaps are 0.03–0.07 against sems of 0.013–0.028.** Only the conjunction contrast
  (t = −4.99) is strongly significant; the repair and rate contrasts are consistent with zero.
- **3 seeds, one task family, one region geometry**, and each seed places its own regions, so
  seed-to-seed level differences are large and the paired within-seed contrasts are the readable ones.
- **E5's ratio degrades badly at the starved end** (`local` at S=16: +3.303 ± 3.181) because the
  denominator, surface repair, falls to 0.005. Any future version wants a difference, not a ratio.
- **The floor estimator's bias is not small** — 0.36–0.46 on regions whose true noise is 0, against a
  target error of ~0.45. It is common across regions in the tame band, which is what the tap consumes,
  but it is not negligible and it grows as the batch thins.
- **E4's monitor:collect is 1.82×**, matching E3, but K = 4 here against E3's 6; the anti-subsidy
  property is that looking is charged and O(1), not that the constant is comparable.

## 7. Open next steps

1. **Explain §4d before trusting any allocation tap here.** The outcome column is not predicted by
   leak, A-share, or allocation variance. The cheapest discriminator is a **scheduled-burst control**:
   an arm with `oracle`'s target but `value`'s temporal profile (all-or-nothing every k rounds),
   against one with `value`'s mean share spread evenly. If bursty beats steady at matched mean share,
   the ladder has been measuring timing, not reducibility — which would re-scope E3's headline too.
2. **Fix the off-reach measurement hole.** `Boff1` is the arm that catches `lprog-only` and it is the
   one region the body cannot survey. Either place off-reach regions inside the dynamic reach band by
   construction (they can be off the *eval path* without being far from the start posture), or accept
   that off-reach means unsurveyable and say so as a finding about embodiment.
3. **Give the FM a route to the shared parameter, or drop E5's question on this substrate.** Disjoint
   spatial gates are the reason §5 is a null. A *globally* parameterised drift (one curl gain over the
   whole workspace, with regions differing in something else) would let a shared representation exist;
   whether a meter then drives the learner toward it is the original question, still open.
4. **A slower drift** for E4, which E3 also names — no policy fully repairs A, and it would give the
   near-saturated control instrument some dynamic range so the ladder is not single-instrument.
5. **The learned-value head** (E3 next-step 1a) remains open, and RHM's grader-type result constrains
   it: the endogenous *dense* grader landed **below the no-loop floor** there, so the head must be
   graded on outcomes, not on FM self-predictability.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/on_policy/verify_backcompat.py::verify                                  # env gate

# E4 -- gates first (a strict prefix of the real run), then the ladder
modal run mjc/on_policy/metered_repair/floor_tap.py::floor_tap --quick --certify-only --tag cert
for s in 0 1 2; do   # launch each seed as its OWN client -- detached siblings evict each other
  modal run --detach mjc/on_policy/metered_repair/floor_tap.py::floor_tap --tag tap_s$s --seed $s
done
for s in 0 1 2; do   # the rate arms, merged by seed with the above
  modal run --detach mjc/on_policy/metered_repair/floor_tap.py::floor_tap --tag tapd_s$s --seed $s \
      --policies "reddelta-only,value-reddelta"
done
python3 mjc/on_policy/metered_repair/floor_tap_agg.py --tags tap_s0 tap_s1 tap_s2 tapd_s0 tapd_s1 tapd_s2

# E5 -- calibration gate, then the sweep
modal run mjc/on_policy/metered_repair/necessity.py::necessity --quick --calibrate-only --tag ncal
for s in 0 1 2; do
  modal run --detach mjc/on_policy/metered_repair/necessity.py::necessity --tag nec_s$s --seed $s
done
python3 mjc/on_policy/metered_repair/necessity_agg.py --tags nec_s0 nec_s1 nec_s2
```

Results on the `mujoco-control-data` volume under `metered_repair/`, mirrored to
[`figures/`](figures/). The aggregator merges tags by trailing `_s<N>`, which is sound because every
RNG stream in the loop is policy-independent — geometry, drift trajectory, base FM, probes and the
per-round seeds all derive from `cfg["seed"]` and the round index, never from which policies were
requested — and it errors rather than silently overwriting if a policy appears in two tags.
