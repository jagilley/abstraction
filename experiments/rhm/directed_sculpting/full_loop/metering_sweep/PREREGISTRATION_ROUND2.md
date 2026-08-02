# Pre-registration — round 2: reaching the domain where the claim is testable

**Written before any round-2 run.** Round 1's result and its two scoping limits are in
[README.md](README.md); this fixes the direction and the kill criteria for the follow-up.
**Date**: 2026-08-01.

---

## What round 1 left open

Round 1 answered the doc's question as asked and produced two facts that bound the answer:

1. **`E(d)` steepens** (−0.011 → −0.025 per doubling over 317×), leaving ~16 doublings to the
   aleatoric floor. **The saturation shoulder the doc's prediction lives on was never reached**,
   so the negative is scoped: the claim survives as an asymptotic one.
2. **The prize is two effects** — interference (falls with budget) and data quantity (rises) —
   and the doc's intuition fits only the first.

Round 2 is built to (a) reach saturation, (b) rule out a compute artefact, and (c) ask the
metering question in the form the ladder has never been able to pose.

## The key move: damage is the denominator

The round is `advance_drift → collect → refit`. What sets the steady-state error is not data in
the abstract but **data per unit of damage**. A budget `B` against drift `κ` sits at the same
point of the repair balance as `2B` against `2κ`. Round 1 moved the numerator 66× at real GPU
cost; **the denominator moves 30× for free.**

| `drift_kl` | vs published | same repair balance as |
|---|---|---|
| 0.30 (published) | 1× | `B` = 8192 |
| 0.15 | 2× | `B` = 16384 |
| 0.075 | 4× | `B` = 32768 |
| 0.03 | 10× | `B` = 81920 |
| 0.01 | 30× | **`B` = 245760** |
| 0.01 **with `B` = 32768** | 120× | **`B` ≈ 983000** |

That corner is ~15× past round 1's most abundant point, at ~1/8 its cost. This is the same axis
[`../README.md`](../README.md) §6 already calls *"samples per event at fixed KL/event"*, and
`verify_distractors` P4 supplies the anchor: **a static geometry goes inert** — every channel's LP
below the noise floor by step 7500. *Inert is what saturation looks like*, and it is exactly where
the metering claim says the prize must vanish.

## Pre-registered predictions

| # | experiment | prediction |
|---|---|---|
| **P7** | **E1-C** `B` = 262144 | `E(d)` either turns or keeps steepening. If it keeps steepening, round 1's scoping hardens and "unreachable" stops being an extrapolation |
| **P8** | **E3** damage sweep | absolute error falls toward the FM's own floor, and **the prize falls toward the INTERFERENCE term (~0.03–0.04), not to zero** |
| **P9** | **E4** data vs compute | if round 1's steepening is real, the **fixed-compute** cut (16× data at 256 steps/round) still shows error falling with `B`, and the **fixed-data** cut (16× compute at `B`=8192) is flatter |
| **P10** | **E5** honest ladder | `visits_only` beats `uniform` decisively; `value_red` beats it by much less. Per sequence of monitoring bought, **relevance is ~25–30× more cost-effective than reducibility** |

**P8 is the sharp one and it is where round 1's decomposition earns its keep.** If the prize goes
to ~0 at near-static, the doc's *mechanism* is vindicated and round 1's negative is purely about
reachability. If the prize instead plateaus at the interference term, then **"train on all of it"
is never optimal here at any price**, because pouring 80% of the budget into distractors keeps
damaging the shared-parameter FM no matter how saturated the task is — and the metering claim
would need restating as a claim about the data-quantity half alone. Round 1's decomposition
(interference is 68% of the prize at the anchor and falls only slowly) makes the second outcome
the one to bet on.

## E5 — the metering question, first-class

The published ladder charges **every arm the same monitor bill on purpose**, *"so the ladder
measures allocation only"*. That isolates the allocation rule and hides the economics: **a smart
allocator never pays for its own smartness.** E5 splits one shared total by what each policy's
drive actually reads (`ladder.py::TAP_READS`):

| arm | reads | bill/round | `B` at `T` = 22784 |
|---|---|---|---|
| `uniform`, `oracle`, `oracle_dup` | nothing | **0** | 22784 |
| `visits_only` | forecast | **512** | 22272 |
| `reducible_only` | monitor set + floors | **14080** | 8704 |
| `value_red` | all three | **14592** | 8192 |

`uniform` needs no monitoring at all, so it collects the entire budget while `value_red` buys
**64% less data** in exchange for knowing where to spend it. *Is being smart worth what it costs
to be smart?* has never been askable on this ladder.

It also puts the first price tag on the idea doc's §6 claim about cheap vs weak curation — and
predicts an **inversion of it**. §6 argues LLM curation does the cheap *epistemic* thing while the
*relevance* tap is what pays but costs weeks of wall-clock. Here the endogenous relevance tap is
**27.5× cheaper** than the reducibility tap *and* recovers ~5× more of the oracle prize. If P10
holds, the cheap curation is the **strong** one, exactly as the doc's own remark that
`forecast_visits` computes it *"in one round, with zero environment interaction and zero labels"*
would suggest.

## Gates (unchanged from round 1, plus two)

- **C1–C5** as before; `oracle_dup` carries the pointwise noise floor at every new point.
- **C6 — the damage knob is real.** `drift_spec`'s reported per-channel KL/round must scale with
  `drift_kl` as commanded, and `kl_per_channel` in the round rows must track it. A damage sweep
  that did not move the damage would look exactly like saturation.
- **C7 — E5's budgets are what the bill says.** Each arm's recorded `collect_budget` must equal
  `T − bill(arm)` exactly, and `meter_collect / rounds` must match it.

**Reading rules fixed in advance.** In E3, a prize that shrinks is only saturation if the absolute
errors *fall* toward a floor (round 1's P4 discriminator, reused). In E5, `value_red` losing to
`uniform` is a real result about price and **not** evidence that the tap is bad — the tap's quality
is measured by round 1's equal-charge ladder, and E5 measures only whether it is worth buying.
