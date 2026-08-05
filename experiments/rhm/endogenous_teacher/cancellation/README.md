# Cancellation on RHM: the mechanism replicates, the payoffs do not

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Idea**: [ideas/efference_copy_cancellation.md](../../../../ideas/efference_copy_cancellation.md) ·
**Prior run**: [a2a_forward/CANCELLATION_README.md](../../../a2a_forward/CANCELLATION_README.md) (looped ViT)
**Date**: 2026-08-04 · **Status**: single seed, one caveat outstanding (see *What would settle it*).

## Goal

The parent cut collapsed the FM residual to a scalar and used it to weight the NTP loss; that was a
null, and over-determined. This cut changes the **channel** instead: subtract the forecast and
propagate the deviation, so the residual is the tensor downstream computes on.

```
summation (the whole a2a arc):   x = x + gate·p                       at the inject block
cancellation (efference copy):   x = x − gate·p  …  x = x + gate·p    restored at the re-add block
```

Two reasons to run it here rather than repeat the ViT.
[`efference_copy_cancellation`](../../../../ideas/efference_copy_cancellation.md) records **Payoff 1**
(modular vs entangled dependency, via a fresh-FM swap) as inconclusive on the looped ViT *"because the
low-rank loop's FM is near-unique, so a 'different but equally-good' FM barely exists to swap in — too
weak a perturbation… Needs a gauge-free substrate (RHM / language)."* And RHM's ground-truth latents
give a per-level readout the ViT lacks.

**Regime is m2** (`v16 s2 L6 m2`), not m4: at m2 token-NTP reaches the root, so the injection has
something to strip and the summation pathology is largest and already measured. `λ_local = 0`
throughout, isolating the **injection channel** — [`RHM_LATENT_LOOP`](../../RHM_LATENT_LOOP_README.md)
Exp 3 showed the offloading is λ-independent while the local loss is a separate consolidating effect.

## Results

| arm | val standalone | val injected | dep | d4 | d5 | **root (d6)** | fresh SK b7 | **cos(Δ,inj)** | **swap cost** |
|---|---|---|---|---|---|---|---|---|---|
| `ntp` | 0.8232 | — | — | 1.000 | 1.000 | **0.863** | 0.448 | — | — |
| `cl_sum` | 1.6285 | 0.8232 | +0.805 | 1.000 | 0.926 | 0.420 | 0.373 | **+0.293** | +0.0024 |
| `cl_cancel` | 1.6140 | 0.8236 | +0.790 | 0.999 | 0.905 | 0.380 | 0.349 | **+0.088** | +0.0015 |

Reference check: `RHM_LATENT_LOOP` m2 gives `ntp` root 0.819 / SK 0.463 and `ntp_cl@0.0` root 0.453 /
SK 0.398 — reproduced on every line.

**Mechanism (P3) — confirmed, and a close cross-substrate replication.** `cos(Δ,inj)` falls **+0.293 →
+0.088** at *matched* injection norm (17.25 vs 17.22). The looped ViT read **+0.89 → +0.095**. The
cancel-side numbers agree to two decimals across a different architecture, a different task, and a
feedforward stack rather than a loop — consistent with the idea doc's claim that the effect is
topological rather than learned. (Our summation side is far below the ViT's +0.89: on RHM the
*additive* injection is already substantially orthogonalised. Unexplained; noted.)

**Payoff 1 (fresh-FM swap) — directionally right, negligible in size.** Cancellation costs less
(+0.0015 vs +0.0024, 3 fresh FMs per arm, all within 0.0004 of each other), but against a **dependency
of 0.80 nats** and a random-FM floor of **1.38–1.43**. So the forecast's *content* matters enormously
while *which* competent forecaster supplies it does not — **downstream depends on a forecast, not on
this forecaster, in both wirings**. On this substrate the *entangled* dependency that cancellation was
proposed to fix does not appear.

**P2 (does cancellation avoid the token-CL self-knowledge degradation?) — falsified.** Fresh SK
0.448 → 0.373 (`cl_sum`, Δ −0.075) → 0.349 (`cl_cancel`, Δ −0.099). Cancellation is marginally
*worse*. Co-trained SK is identical for both (0.465 at b7), so the cotrained−fresh gap is 0.092 for
summation and 0.116 for cancellation: cancellation makes M *slightly more* FM-specific, the opposite
of the prediction.

**Offloading — reproduced.** Both wirings halve the standalone root (0.863 → 0.420 / 0.380). Injected
root is 0.874 / 0.879, *above* `ntp`'s 0.863, while injected val ties `ntp` exactly (0.8232/0.8236 vs
0.8232): **pure computation relocation**, reproduced on RHM. Note that standalone-removal is the wrong
readout for cancellation and the idea doc says so — subtracting a forecast that is then absent is a
large distribution shift, so *"cancellation makes dependency the design"*.

## What would settle the Payoff-1 null

The three fresh FMs all land at cos **0.871–0.872** against the target. We did not measure whether they
are *functionally* distinct or merely *gauge*-distinct. If "predict `post_block6` from `post_block0`
for this frozen model" has a near-unique solution in function space, the swap perturbation does not
exist here either and this reproduces the ViT's inconclusive result for a different reason rather than
resolving it. The instrument already exists — the input-centered pairwise residual cosine (`ens_cos`)
from [`RHM_LATENT_LOOP`](../../RHM_LATENT_LOOP_README.md) Exp 4. Worth pairing with the `post_block6`
activation norm, since `|inj| = 17.2` is large and the FM's MSE *grew* monotonically over training
(1.28 → 2.95), which is consistent with activation-norm inflation.

## Code changes outside this folder, and their safety

`rhm/model.py::GPT.forward` gained `cerebellar_mode` (`"add"`/`"cancel"`) and `cerebellar_readd_block`.
Both default to the historical summation path. [`test_backcompat.py`](test_backcompat.py) asserts this
against a verbatim transcription of the pre-change loop using `torch.equal` (bit-exact, not
`allclose`): logits, loss and every intermediate match on the injected path, the plain path, and
non-default inject/input blocks; cancel with a zero injection equals the plain forward (so a zero-init
gate makes this a strict generalization); and a misconfigured re-add block now raises instead of
silently dropping the forecast. **All prior RHM callers pass neither argument**, so they are unaffected.
`a2a_forward/model.py`, `a2a_forward/vit.py` and `language_reduction/model.py` are separate
implementations and were not touched.

## Gotchas

- **Under cancellation the FM's target must be `post_block{k}_eff`** (the restored read-out), never
  `post_block{k}` (the deviation stream, recorded pre-re-add for consistency with every other block).
  Training the FM on the deviation is degenerate — it chases the deviation its own forecast creates and
  drives the injection to zero. This is the easiest way to get a spurious null here.
- The SK forward model is fitted on **standalone** activations for every arm, because `_sk_probes` runs
  standalone forward passes. Fitting it on closed-loop activations for the closed arms only would break
  comparability with the `ntp .463 / cl_sum .398` reference.

## Reproduction

```bash
cd experiments
modal run --detach -m rhm.endogenous_teacher.cancellation.cancellation::cancellation \
    --n-steps 20000 --tag cut2
modal run -m rhm.endogenous_teacher.cancellation.test_backcompat::test_backcompat
```

Results land at `/data/v16_s2_L6_m2_distinct/cancellation/results_<tag>_seed<seed>.json`.
