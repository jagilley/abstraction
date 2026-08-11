# FILES — cancellation (RHM port)

**Up**: [../README.md](../README.md) · [../DESIGN.md](../DESIGN.md)
**Idea**: [`ideas/efference_copy_cancellation.md`](../../../../ideas/efference_copy_cancellation.md) ·
**Prior run**: [`a2a_forward/CANCELLATION_README.md`](../../../a2a_forward/CANCELLATION_README.md) (looped ViT)

## Code files

| file | purpose |
|---|---|
| `cancellation.py` | The cut: `ntp` / `cl_sum` / `cl_cancel` on RHM m2 at matched everything, plus the fresh-FM swap (Payoff 1), fresh-FM SK probes, and the `cos(Δ,inj)` mechanism check. One Modal L4 function. |

## The one change outside this folder

`rhm/model.py::GPT.forward` gained `cerebellar_mode` (`"add"` / `"cancel"`) and
`cerebellar_readd_block`. **Backwards-compatible by construction**: both default to the
historical summation path, so every prior RHM result re-runs bit-identically. Under
`"cancel"` the forecast is subtracted at the inject block, the deviation propagates, and
the forecast is restored at the re-add block.

## Design points a future agent should not re-derive

- **The FM's training target under cancellation is `post_block{k}_eff`, not `post_block{k}`.**
  `post_block{k}` is recorded pre-re-add (the *deviation* stream, consistent with every
  other block); `_eff` is the restored read-out. Training the FM against the deviation is
  degenerate — the FM would chase the deviation its own forecast creates, driving the
  injection to zero. This is the single easiest way to get a spurious null here.
- **Standalone-removal is the wrong readout for cancellation**, and the idea doc says so:
  subtracting a forecast that is then absent is a large distribution shift, so "can't run
  without it" is the design, not a bug. The discriminating readout is the **fresh-FM swap** —
  is downstream entangled with its own FM's idiosyncrasies, or does it own only the residual?
- **Why RHM and not the ViT.** `efference_copy_cancellation.md` records Payoff 1 as
  inconclusive on the looped ViT "because the low-rank loop's FM is near-unique, so a
  'different but equally-good' FM barely exists to swap in — too weak a perturbation to test
  entanglement. Needs a gauge-free substrate (RHM / language)." This is that substrate.
- **Why m2 and not m4.** At m2 token-NTP reaches the root (0.819) and summation strips it to
  ~0.453, so the summation pathology is largest and already measured. At m4 the root is
  ~0.08 and there is nothing to strip.
- **`λ_local = 0` throughout** — this isolates the *injection channel*. `RHM_LATENT_LOOP`
  Exp 3 showed the offloading is λ-independent (it is the injection) while the local loss is a
  separate consolidating effect that inverts SK on its own; mixing them would confound the cut.
- **The SK forward model is fitted on standalone activations for every arm**, because
  `_sk_probes` runs standalone forward passes. Fitting it on closed-loop activations for the
  closed arms only would break comparability with the `ntp .463 / cl_sum .398` references.
- **Zero-init gate** ⇒ at step 0 `cl_sum` ≡ `cl_cancel` ≡ `ntp`. A strict generalization,
  learned into rather than imposed.

## Reference lines (from [`RHM_LATENT_LOOP`](../../RHM_LATENT_LOOP_README.md), m2)

| | root (d6) | val | fresh SK b7 |
|---|---|---|---|
| `ntp` | 0.819 | 0.826 | 0.463 |
| `ntp_cl@0.0` (injection only) | 0.453 | 1.855 | 0.398 |
| `ntp_cl@1.0` | 0.433 | 1.047 | −1.078 |

ViT mechanism reference: `cos(Δ,inj)` summation **+0.89** → cancellation **+0.095**.
