# Directed sculpting: porting mjc E3's inner+outer loop to RHM — the DGP groundwork

**Status**: substrate built and verified, **and the loop now built and run** — see [`full_loop/`](full_loop/README.md). Phase 1 (this README) is the DGP and the instruments. Four primitives, each with its own validation entrypoint and committed results. Single rule-seed throughout (repo convention).
**Date**: 2026-07-28
**Up**: [../README.md](../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../ideas/adaptive_core_and_hierarchy_climb.md) — this node builds its §6 instrument and the DGP its §11 argument needs.
**Direct parent**: [`mjc/on_policy/directed_on_policy/`](../../mjc/on_policy/directed_on_policy/README.md) (E3, PR #4) — the loop being ported.
**Recovered design**: `ideas/mjc_learnings_on_rhm.md` §3, deleted in `b727bfd`, readable at `git show b727bfd^:ideas/mjc_learnings_on_rhm.md`. It specifies the drift primitive below almost verbatim; it was dropped for its *allocation-over-NTP-cells* framing, not for this.

---

## One-liner

Porting E3's `value = lprog × visits` outer loop to RHM sculpting hits a wall the plain DGP
cannot get past — **RHM rule usage is uniform by construction, so there is no relevance
measure to allocate over**. Adding non-tree distractor channels supplies one structurally;
support-fixed mixture drift keeps it live; and the repair-cost instrument that reads the
homeostatic claim turns out to have **two traps in it that each produce a confident wrong
answer**, the first being that raw cross-entropy is *exactly* blind to this drift.

## Why the naive port fails

E3's outer loop needs a non-uniform visitation measure. RHM's rule tables are `(v, m, s)`
arrays shared across every position and every root, with the root drawn uniformly
([`rhm_data.py:17-90`](../rhm_data.py)), so the marginal over rule cells is flat by
construction. Measured — normalised entropy of the per-level feature marginal:

| conditioning | level 1 | level 2 | level 3+ |
|---|---|---|---|
| unconditional | ≥ 0.93 | ≥ 0.93 | ≥ 0.93 |
| conditioned on root `r*` | 0.41–0.72 | 0.77–0.95 | **back at the unconditional floor** |

Root-conditional relevance decays to the floor **within two levels of the root**, in every
setting we run (L=4/m=2 and L=6/m=4, v=8 and v=16). For the specialization setting the
level-3 root-conditional entropy is 0.976 against an unconditional 0.979 — indistinguishable.
This is the specialization line's *"only d6 has a thin A-specific slice"*
([`../specialization/README.md`](../specialization/README.md)) as an exact number, and it means
`lprog × visits` would collapse to `lprog` — the confounded null
[`two_timescale_value_loop.md`](../../../ideas/two_timescale_value_loop.md) (2026-07-17 substrate
correction) warns about.

There is a second, sharper problem the idea doc does not anticipate. Relevance only
discriminates for drift near the root; the homeostatic climbing payoff only exists for drift
at the *surface* (§6: if deep rules drift too, climbing buys nothing). **The two arms want
opposite ends of the same knob**, so one run cannot deliver both.

## The fix: distractor channels ([`../rhm_channels.py`](../rhm_channels.py))

Put information in the sequence that is not drawn from the RHM tree (Jasper's). A distractor
channel is *structurally* incapable of moving the parsed root, so it is irrelevant by
construction rather than by a corruption-distribution choice — and it can still be learnable
(an independent grammar) or unlearnable (iid draws). This reproduces E3's actual geometry,
which was always one target among five distractors rather than graded allocation *within* the
target. It also **dissolves the anti-correlation**: relevance now keys on tree-vs-distractor
and no longer rides the drift-level knob, so drift can sit at the surface where the climbing
payoff is largest while relevance stays live.

Layout at L=5 — 56 tokens / 28 blocks, same `s=2` block granularity as the tree so the edit
command space `k` reaches distractor blocks on equal footing (deliberate: if they were not
editable there would be no trap for `lprog-only` to fall into):

| channel | kind | blocks | E3 counterpart | catches |
|---|---|---|---|---|
| `tree` | RHM L=5, m=2 | 16 | on-reach reducible target A | — |
| `structA` | RHM L=3, m=2 | 4 | off-reach reducible | `lprog-only` |
| `structB` | RHM L=3, m=4 | 4 | off-reach reducible | `lprog-only` |
| `noiseA/B` | iid | 2 each | off-reach irreducible | `error-only` |

### Verified (20k steps, [`verify_distractors.py`](verify_distractors.py))

| | result |
|---|---|
| **P1** distractor content cannot move the root | **PASS** — scrambling all 24 non-tree tokens leaves parsed root, validity, and DP `d*` **bit-identical** (mean `d*` 4.42) |
| **P2** noise is exactly irreducible | **PASS** — lands at the information-theoretic floor 1.9595 nats: excess **+0.0002** / **−0.0010** |
| **P3** noise traps `error-only` | **PASS** from step 1000, min margin **+0.509** nats. Final ordering noise 1.960 > structB 1.360 > structA 0.848 > tree 0.687 |
| **P5** channels not separable by unigram statistics | **PASS** for noise-vs-tree (H 1.960 vs 1.960, TV-from-uniform 0.176 vs 0.176) |

**P2 is worth dwelling on.** E3 needed a hand-found `amp < gear` band and lost a smoke run to
noise bleeding into the target (region A's ceiling jumped 0.14 → 0.37). Here irreducibility is
exact by construction, with no tuning — the idea doc's *"RHM makes irreducibility exact rather
than tuned"* cashed out.

### P4 is the load-bearing result, and it has two edges

Against an LP floor **calibrated off the noise channels themselves** (0.0084/1k — the noise
channels are irreducible by construction, so their post-warmup LP is the built-in null):

| channel | peak LP /1k | saturates |
|---|---|---|
| noise A/B | 0.004 | 1500 |
| **tree** | 0.022 | **4000** |
| structA | 0.038 | 7500 |
| structB | 0.078 | 7500 |

- **A *static* geometry goes inert.** Every channel is flat by step 7500 of 20000, so
  allocation stops being a live question for two thirds of a run. The struct channels must
  drift alongside the tree — which is exactly why E3 put *all* its reducible regions on a
  continuous OU walk.
- **The tree satiates *before* the distractors** (4000 vs 7500), opening a window where
  `lprog-only` correctly reads more learning progress in the irrelevant channels and would
  chase it. That is the sharpest place for `value` to separate from `lprog-only`, and it is
  identifiable in advance rather than hoped for.

The tree's "saturation" is the known RHM shallow-frontier stall, not mastery — a 2.68M model
plateauing at CE 0.687 has learned d1–d3 and stopped. That is the baseline the climbing claim
must beat, so it is the phenomenon rather than a nuisance.

## Drift: support-fixed OU walk on synonym mixtures ([`../rhm_drift.py`](../rhm_drift.py))

A cell-wise Ornstein-Uhlenbeck walk on the mixture weights over each `(level, feature)` cell's
`m` synonymous rules. Chosen over mutating rule *entries* for three reasons:

1. **Support-fixed.** The legal tuple set never changes, only the frequencies. Verified: after
   200 heavy OU steps the on-grammar rate is **1.0000**, the rule tables are bit-identical, and
   the inverse maps are unchanged — so the DP `d*`, a function of the rule support alone,
   *cannot* move. That is immunity by construction to the catastrophic forgetting that made the
   wholesale rule-seed swap measure nothing (val on old rules 1.41 → 4.51,
   [`../residual_decomposition/README.md:148`](../residual_decomposition/README.md); hazard
   recorded at [`beliefs/dimensionality_expansion.md:144`](../../../beliefs/dimensionality_expansion.md)).
2. **Task-invariant.** The root stays recoverable under any drift, so what the agent is graded
   on never moves. We drift the **encoding**, not the **content** — the condition the
   homeostatic argument needs, obtained as a property rather than an assumption.
3. **Continuous.** [`curiosity_drift.py:183-188`](../../a2a_forward/reaching/curiosity_drift.py)
   found abrupt swap breaks a naive LP drive while smooth morph does not.

### Magnitude is closed-form, not estimated

The process is a chain of independent categorical draws differing only in weights, so the KL
per sequence decomposes exactly:

```
KL(new‖old) = Σ_level Σ_feature  s^ℓ · p_ℓ(f) · KL(w_new[ℓ,f] ‖ w_old[ℓ,f])
```

Verified against the realised log-likelihood ratio along recorded derivations: analytic
**10.786** vs empirical **10.759**, **0.26%** error.

This closes §6's *matched drift magnitude* hygiene item — and it turns out to be **load-bearing
rather than cosmetic**. A cell at depth ℓ fires `s^ℓ` times per sequence, so at a shared σ the
per-level KL spans **11.9×** at L=5. Uncalibrated, the level sweep would have measured each
level's *fan-out* rather than its *invariance*, and the resulting monotone curve would have read
as the climbing dose-response. `calibrate_sigma` inverts the closed form per level:

| level | nodes `s^ℓ` | σ for matched KL |
|---|---|---|
| 0 (root) | 1 | 0.587 |
| 1 | 2 | 0.308 |
| 2 | 4 | 0.192 |
| 3 | 8 | 0.129 |
| 4 (surface) | 16 | 0.088 |

Within a drift realisation the cross-level KL spread drops to **1.03–1.13×** (from 11.9×).
Calibration samples the stationary Gaussian directly under common random numbers; simulating a
correlated path instead gave ±15% wobble (κ=0.05 gives ~10 effective samples over 200 steps).
**D4**: the walk is stationary — TV-from-uniform 0.320 / 0.315 / 0.319 / 0.307 across 2000 steps.

## Repair cost: the homeostatic instrument ([`../rhm_repair_cost.py`](../rhm_repair_cost.py))

§6's claim is that drift event #8 costs less to repair than #1 at matched magnitude. Magnitude
matching is now exact; this is the cost half. **Two traps, each of which produces a confident
wrong answer.**

### Trap 1 — raw cross-entropy is *exactly* blind to this drift

For a mixture drifting away from uniform, the per-node KL is precisely the entropy it destroys
(`KL(w‖uniform) = log m − H(w)`). Summing over nodes, `H(p_old) − H(p_new) = KL` exactly, so a
model that perfectly learned the pre-drift DGP scores

```
CE(p_new, q=p_old) = H(p_new) + KL(p_new‖p_old) = H(p_old)
```

— *precisely its pre-drift CE*. Verified analytically at **0.000e+00** difference, and
empirically at the leaf level the model is actually graded on:

| drift KL /seq | stale model's raw CE shift | attributable gap | raw as % of gap |
|---|---|---|---|
| 0.000 | −0.0011 | −0.0015 (null) | — |
| 0.335 | −0.0003 | +0.0096 | 3.5% |
| 0.677 | −0.0000 | +0.0203 | 0.1% |
| 1.372 | +0.0004 | +0.0421 | 1.0% |

Across drifts spanning 4× in magnitude the raw CE moves by less than half a millinat and is
literally zero at KL=0.677. **The obvious instrument — "budget until CE returns to its
pre-drift value" — would have reported zero repair needed, always.** The drift makes the world
easier by exactly as much as it makes the model wrong, and raw CE sees neither half. A null by
construction that reads as a finding.

The working readout is the gap to a matched reference — E3's `gap_closed` /
`regA_recovery_auc` idiom ported.

### Trap 2 — that gap is itself confounded by ordinary continued training

A matched reference improves both because it adapts to the drift and because it is further
along. Measured on the first (fixed-pool) run: with **zero** drift the stale→matched gap was
**+0.0187** nats — *larger* than the **+0.0167** the actual drift contributed at KL=0.68.
Uncorrected, every repair number is overstated more than twofold. This is §6's confound
reappearing *inside* the instrument.

Fixed with a paired matched-**compute** arm (same base model, same budgets, undrifted data) and
a difference-in-differences readout. At zero drift the attributable gap collapses to −0.0015.
This also builds in the matched-data arm the idea doc flags as possibly needed later to
*apportion* a falling cost curve — available rather than retrofitted.

### The cross-check that validates it

The measured attributable gap recovers the **analytically known** drift magnitude:

| closed-form KL | per token | measured gap | recovery |
|---|---|---|---|
| 0.335 /seq | 0.01081 | 0.00955 | **88.4%** |
| 0.677 /seq | 0.02184 | 0.02033 | **93.1%** |
| 1.372 /seq | 0.04426 | 0.04210 | **95.1%** |

`rhm_drift`'s closed form and `rhm_repair_cost`'s empirical gap were built independently and
agree to within 5–12%, with recovery rising toward 100% as magnitude clears the noise floor.
The instrument is calibrated in absolute nats, not merely ordinally. **R1**: the zero-drift arm
is correctly reported *out of range* rather than as a number.

### Two gotchas inside the instrument

- **Cost-to-a-*fraction*-of-the-gap does not increase with magnitude** and should not be
  expected to: measured 48449 / 12225 / 12135 sequences for KL 0.335 / 0.677 / 1.372. A small
  drift sits near the noise floor, so closing 90% of its small gap takes *more* samples. The
  quantity that tracks magnitude is the **gap**. Harmless for the homeostatic claim (which
  compares at matched magnitude by construction) but a cross-magnitude cost comparison is not
  the sensitivity check it appears to be.
- **The out-of-range guard must be calibrated off the null arm**, not an absolute constant. A
  null gap of −0.0044 is statistically zero but a thousandfold above a `1e-6` guard, and
  dividing by it produced a confident *"progress 1.052"* on a drift that never happened.

### The harness bug, and what it revealed

The first full run trained from a fixed 20k-sequence pool — ~38 epochs at 6000 steps. The base
model was **worse** held-out at 6000 steps than at 800 (0.7420 vs 0.7306) and the matched
reference collapsed to CE **1.31**, making every `gap_total` negative. Fresh per-step sampling
fixed it (control gap **−0.5697 → +0.00322**, base **0.7420 → 0.6909**) and also repaired the
meter's semantics: it charges *sequences consumed*, and the reason metering makes "where should
I spend" a question at all (E3's 1.84× vs S2's fatal 22×) is that each charged unit is data the
agent had to go and get. Charging for resampled pool draws was measuring gradient steps wearing
a costume.

**Notably the difference-in-differences rescued the broken run anyway** — both arms overfit
equally, the subtraction cancelled, and R1–R3 came out clean *through* a collapsing ceiling.
Good evidence the DiD is load-bearing; also a reminder that its passing does not certify what
sits underneath it.

## Back-compat gate ([`../verify_backcompat.py`](../verify_backcompat.py))

RHM never had one; `mjc/on_policy` does, and E3's README makes it the first reproduce step.
B1 rule generators byte-identical across the 10 `(v,s,L,m)` settings appearing in committed
writeups × 3 seeds · B2 `generate_sequences_batched` deterministic · B3 `make_corpus` defaults
byte-identical with `meta` a strict superset · B4 weighted sampler at uniform weights matches
plain (worst leaf TV 0.0019 — it consumes the rng differently, so bit-identity is not the
claim) · **B5** every grammar readout invariant under 300 heavy OU steps. B5 is the formal
statement of the design property: drift moves the encoding, not the content.

## What this establishes

1. **RHM's uniform rule usage is an exact obstacle to the E3 port**, quantified rather than
   suspected — and non-tree distractor channels remove it without needing specialization's
   unbuilt cut-3 (context-sensitive rules).
2. **Support-fixed drift gives non-stationarity without task-switching.** The DP `d*` is
   provably invariant, so the catastrophic-forgetting failure mode is structurally unavailable.
3. **Drift magnitude has a closed form**, which makes §6's matched-magnitude sweep exact and
   catches an 11.9× per-level mis-scaling that would have masqueraded as the dose-response.
4. **The homeostatic readout cannot be raw loss.** Verified exactly, and the correct readout
   cross-validates against the closed-form magnitude at 88–95%.

## The loop itself — [`full_loop/`](full_loop/README.md)

**Goal**: wire E3's inner loop (reward-free block-FM re-adaptation + ballistic control) and outer loop
(`value = lprog × visits` over a metered budget) onto this substrate, and test the idea doc's three drift-
dependent claims.

**Finding**: the reward-free **relevance** signal ports and is a smoking gun — a value trained only on terminal
task success, never told which channel is which, separates real tokens from distractors **13–18×** and
allocating by it alone recovers **76%** of a privileged oracle's advantage over uniform. But **expansion turns
out to be a property of the grader's *type*, not of non-stationarity**: a grounded evaluative grader expands the
belief ~1.7× in effective dimension and ~15× in ballistic control where an endogenous dense grader caps *below
the no-loop floor*, and that holds identically in a static and a drifting world. Both drift-dependent
predictions — hierarchy-climbing (§6/§11) and drift-sustained expansion (§5's missing cell) — come back
negative, under instruments that had to be repaired twice, each repair removing a bias that pointed toward the
positive result. The port passes a back-compat gate against the single-channel apparatus (`delta_cos` 0.492 vs
the reference's 0.486).

**A follow-up sweep re-opens the climb question.** The climb null held only because surface re-fit repaired
**91%** of each drift event, so invariance was free and nothing ever priced the integrated cost — *invariance ≠
necessity*. Starving **samples per event** at fixed drift magnitude makes the drifting arm's depth advantage
**migrate from the surface to the deep levels**, monotonically across four budgets (t = −5.44, 3/3 seeds). Small
and single-instrument, but the first positive on the climbing axis. Separately, repairing the `e` tap
(floor-corrected reducibility, with the aleatoric floor *measured* by repeat-execution) buys −0.054 ± 0.009 over
the counterfactual fit and an 8× drop in allocation variance — and shows the ladder cannot separate
relevance×reducibility from relevance alone *by construction*, since this geometry has no
visited-but-irreducible cell.

**Also lands two corrections to this node's own primitives**, both now default-off in
[`../rhm_drift.py`](../rhm_drift.py): the OU walk anchors at the *maximum-entropy* point of the simplex, so a
uniform-anchored static control is a handicap rather than a control (~18% on the FM's stochasticity floor); and
`calibrate_sigma` matches accumulated displacement rather than event size, which left a matched-magnitude level
sweep running at a 4.5× spread. `stationary_theta` / `calibrate_sigma_event` fix both.

## Caveats

- **This README covers the DGP and the instruments only.** The arity-2 block FM, the DP `k*` teacher
  and the six-policy ladder live in [`full_loop/`](full_loop/README.md), which carries its own caveats.
- Single rule-seed; single L=5 setting. Directional.
- The distractor verification is **token-CE on a plain LM**, not the sculpting apparatus. It
  establishes DGP properties (irreducibility, structural irrelevance, the learnability
  ordering); the FM-level LP that the real loop reads is a separate check.
- `structA`/`structB` unigram statistics differ from the tree (1.922 / 2.048 vs 1.960) — they
  have their own grammars. Only the noise-vs-tree match was engineered, because that is the
  `error-only` trap.
- The committed `repair_cost_l5_fresh/results.json` cost column was computed off the **raw**
  progress curve; `cost_from_progress` (reading the attributable curve) landed after that run.
  The gaps and verdicts are unaffected.
- `repair_cost_l5_v1/results.json` is retained as **instrument validation only** — its absolute
  numbers come from the broken fixed-pool harness.

## Known open items carried into the loop

1. **The in-tree irreducible cell.** E3 could not place an *on-path irreducible* region
   ("you only go where you reach"), which is what separates `visits-only` from `value` — hence
   0.390 vs 0.350 there. Our noise channels are off-path, so we inherit the gap. The naive fix
   (a tree feature with all `v^s` tuples legal) **breaks the task**: such a feature makes the DP
   cost through it zero, so it becomes *free* rather than *unlearnable*. The slippery-actuator
   construction from Stage 3d ([`../rhm_sculpt_latent_stoch.py`](../rhm_sculpt_latent_stoch.py))
   localised to tree blocks looks right — grammar untouched, outcome unpredictable — but whether
   forecast visitation survives the value learning those blocks are uncontrollable needs its own
   check.
2. **`edit_budget` must grow at L=5.** DP solvability of corrupted (c=3) starts within budget 6
   is **30.6%** at L=5 against **67.6%** at L=4 (mean `d*` 5.85 → 7.92). Budget ≈ 9–10 holds
   L=4's solvable fraction; ≈ 12 gives ~95%.
3. **The token beam gets ~5.8× more expensive**: `n_regions` 8 → 28 (3.5×) and budget 6 → 10
   (1.67×). The latent beam materialises only `W` tips/step, so it scales far better.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                       # the gate
modal run rhm/directed_sculpting/verify_distractors.py::verify_distractors --quick    # smoke
modal run --detach rhm/directed_sculpting/verify_distractors.py::verify_distractors --tag l5_v1
modal run rhm/directed_sculpting/verify_drift.py::verify_drift --tag l5_v1
modal run --detach rhm/directed_sculpting/verify_repair_cost.py::verify_repair_cost --tag l5_fresh
```

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/`, mirrored to
[`figures/`](figures/).
