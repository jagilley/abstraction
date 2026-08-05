# Horizon convergence — is the composition horizon a property of the substrate or of the budget?

**Up**: [../README.md](../README.md) (ballistic_depth)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: preliminary — the `N=893` arm is 3 seeds × 2 budgets from existing runs; the `N=9853`
arm is 1 seed × 2 budgets and the long run was killed after its `base` arm finished. **Date**: 2026-08-03.

---

## One-liner

Every number in this cut is a *composition horizon* — the first depth at which exact match falls
below 50%. It has always been read as a property of the trained operator. It is also a function of
the training budget, and **in-distribution accuracy does not tell you when it has converged**:

| `N` | reachable states | steps | ID (T≤6) | horizon | |
|---|---|---|---|---|---|
| 893 | 207 | 8,000 | 1.000 | **15, 13, 13** | |
| 893 | 207 | 20,000 *(2.5×)* | 1.000 | **13, 13, 12** | flat, if anything slightly down |
| 9853 | 2407 | 60,000 | 1.000 | **7, 7, 7** | |
| 9853 | 2407 | 120,000 *(2×)* | 1.000 | **10** | still climbing |

At `N=893` the horizon is **converged** — 2.5× the budget does not move it, which is what makes
§1–§9's numbers safe. At `N=9853` it is **not** — doubling the budget takes exact-match at T=7
from **0.176 to 0.996** and the horizon from 7 to 10, with in-distribution accuracy already
saturated at 1.000 in *both* runs, 60k steps apart.

So the budget at which the horizon converges scales with the state set, and ID accuracy is not
the signal that it has. This invalidates a claim that had already been written into
[`../README.md`](../README.md) §10 — see §3.

## Why this got measured at all

§10 reported the base horizon falling **13 → 7** at 11.6× the state set and called it
capacity-independent, on the strength of a real capacity control: an 8× wider operator moved
in-distribution accuracy 0.958 → 1.000 and left the horizon at exactly 7 in all three seeds. The
inference from that — that the horizon was therefore *converged* — does not follow, and this cut
is the check.

The `N=9853` comparison came from a run launched to test exactly this and killed at ~step 100k of
its second arm; its `base` arm had completed, so the numbers below are read from the run log and
its saved checkpoint rather than from a results JSON.

## 1. `N = 893`: converged, and the extra budget mildly hurts

`base`, 3 seeds, from runs that already existed — `deep60`/`cut1` at 8k steps against
`iso_compute` at 20k:

| steps | T=7 | T=10 | T=13 | horizon (per seed) |
|---|---|---|---|---|
| 8,000 | 1.000 | 0.966 / 0.915 / 0.875 | 0.685 / 0.452 / 0.431 | 15, 13, 13 |
| 20,000 | 1.000 | 0.911 / 0.932 / 0.883 | 0.396 / 0.330 / 0.329 | 13, 13, 12 |

2.5× the gradient steps leaves the horizon inside the ±2 run-to-run band the parent cut
documents, and depth accuracy is consistently *lower* at the deeper rungs. This is the same
observation the parent's gotcha 4 records as an iso-compute control against the cycle result; what
is new here is reading it as a **convergence** statement rather than an overfitting one. Both
readings hold, and they are the same fact.

## 2. `N = 9853`: not converged at the budget every scale conclusion was drawn from

`base`, wide operator (`d_op_ff` 8192), seed 0:

| steps | T=6 | T=7 | T=8 | T=9 | T=10 | T=12 | horizon |
|---|---|---|---|---|---|---|---|
| 60,000 | 1.000 | 0.176 | 0.047 | — | 0.000 | 0.000 | **7** |
| 120,000 | 1.000 | **0.996** | **0.922** | **0.589** | 0.195 | 0.004 | **10** |

The 60k row is not a shallow-but-real horizon; it is a **cliff one step past the training range**,
and the cliff is an artifact of stopping early. At 120k the same architecture composes three
further steps and is still improving, so **10 is a lower bound, not a replacement number**.

The 60k run's in-distribution accuracy was already 1.000 — the arm looked finished by every
readout the cut normally uses.

## 3. What this invalidates, and what it leaves standing

**Invalidated as written**: §10's *"the base horizon halves, 13 → 7, independent of capacity."*
The capacity control was sound and its conclusion (width is not the binding constraint) stands.
The error was treating capacity-independence plus saturated ID as evidence of convergence. The
honest current statement is that the horizon at `N=9853` is **≥ 10 and unconverged**, and no
scale comparison against `N=893`'s 13 can be made until both sides are converged.

**Also premature**: everything §10 says about the cycle term at scale. Those arms were all trained
at 60k steps, so "no weight extends the horizon beyond base's 7" is a statement about an
unconverged baseline, and the closure/horizon decoupling reported there needs re-measuring at a
converged budget before it means anything.

**Left standing**: §1–§9. `N=893` is converged at 8k steps by the 2.5× control above, so the 13,
the 51, and the closure-vs-horizon relationship measured across 72 runs are not budget artifacts.
The same applies to the parent's iso-compute defence of the cycle result, which is strengthened
rather than weakened — at that scale, more steps genuinely do not help.

**Untouched**: [`../rule_structure/`](../rule_structure/README.md). Its null is measured on
representations from both budgets and both moduli, including the ID-perfect wide model, and
structure fell *toward* the null as scale grew. Nothing here bears on it.

## Honest caveats

- **The `N=9853` arm is a single seed at each budget**, and a single doubling. It establishes
  "not converged at 60k"; it does not locate where convergence happens, and 10 should not be
  quoted as the converged horizon.
- **No results JSON for the 120k run.** It was killed at ~step 100k of its `consist` arm, after
  `base` had completed and checkpointed. The `base` numbers come from the run log; the checkpoint
  survives at `/ballistic_depth/scale9853_long/ckpt/base_seed0.pt` so they are re-derivable.
- **The two budget multipliers differ** (2.5× at `N=893`, 2× at `N=9853`), and the `N=893`
  comparison crosses `eval_max_depth` 60 vs 20 — harmless, since both horizons fall well inside
  20, but it means the 20k runs could not have observed a horizon above 20 had one existed.
- **Only the `base` arm.** Whether the *grounded* arm's horizon converges at the same budget is
  untested, and it is the quantity §10's scale claim actually turned on.
- **This says nothing about why** the horizon grows with budget after ID saturates. That the
  operator keeps becoming more composable long after it has memorised the one-step map is the
  interesting object here, and it is not measured — only observed twice.

## Reproduce

```bash
cd experiments/
# N=893, the two budgets (both already exist as tags)
#   deep60 / cut1 = 8k steps, iso_compute = 20k steps, 3 seeds each

# N=9853, 2x budget. Warmup 0.3 of 120k steps puts the cycle unlock at 36k absolute steps,
# which is where it worked in the 60k/warmup-0.6 runs.
MODAL_PROFILE=chromatic modal run --detach \
  one_layer_deeper/ballistic_depth/ballistic_depth.py::ballistic_depth \
  --tag scale9853_long --arms "base,consist" --p 59 --q 167 --d-op-ff 8192 \
  --steps 120000 --eval-max-depth 120 --eval-cap 4096 \
  --consist-cycle 10.0 --consist-reentry 0.0 --consist-warmup 0.3 --seed 0 --save-ckpt
```

Horizons are recomputed from `results/<tag>/results_seed<N>.json` as the first depth where
`exact_seen_x` drops below 0.5.

## Next steps

1. **The budget ladder, properly.** `base` at `N=9853` across 60k / 120k / 240k / 480k, to find
   where the horizon flattens rather than establishing only that 60k is too early. Until that
   exists, no `N=893` vs `N=9853` horizon comparison should be quoted.
2. **The grounded arm at a converged budget** — the quantity §10's scale claim turned on, and the
   only way to answer whether the closure advantage survives a larger state set.
3. **Check convergence as a standing instrument.** If the horizon keeps moving after ID saturates,
   then every horizon in this experiment and in [`../../variable_modulus/`](../../variable_modulus/README.md)
   (60k steps, 3520 reachable states — between the two scales measured here, and never
   budget-checked) carries an unmeasured convergence assumption.
