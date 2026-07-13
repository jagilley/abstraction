# Efference-copy cancellation: subtract the forecast, propagate the residual

**Idea doc**: [ideas/efference_copy_cancellation.md](../../ideas/efference_copy_cancellation.md)
**Parent experiment**: [LOOPED_README.md](LOOPED_README.md) — this closes its next-step #3 ("chase the cancellation signature; its absence (summation/amplification) is a real gap between this model and the biology").
**Status**: done, single seed, one setting (Fashion-MNIST). Primary claim (Payoff 3) confirmed three ways; Payoff-4 mechanism confirmed; Payoff-1 substrate-limited.
**Date**: 2026-07-13

## The question

The whole a2a arc **adds** the forward model's forecast to the residual stream — `s_{t+1} = G(s_t + gate·p)` (*summation*, a side-channel). The idea doc argues it should instead **subtract** the forecast and propagate the residual — `s_{t+1} = G(s_t − gate·p) + gate·p` (*cancellation*, the efference-copy / Rao–Ballard predictive-coding wiring): the forecast is removed from the forward path, only the prediction *error* propagates through the operator, and the forecast is restored at read-out. The claim is that **the sign/topology — not any learned behavior — is what drives the closed-loop pathologies** (entangled dependency, runaway gate, rollout drift). LOOPED Control 3 had already found this architecture *summates and amplifies* the forecast (`cos(effect,inj) ≈ +0.8`), never cancels it — the gap this experiment addresses by *building cancellation in* and testing whether it delivers.

## The change: one flag, a strict generalization

`LoopedViT.forward(..., inject_mode=...)` in [`looped_vit.py`](looped_vit.py); threaded through [`mnist_looped_injection.py`](mnist_looped_injection.py) `train_condition(--inject-mode add|cancel)`.

```
inj = gate(FM(s_t + p) − s_t)                      # the gated forecast (update form)
add    (summation, the arc's default):  s_{t+1} = G(s_t + p + inj)
cancel (efference copy, proposed):      s_{t+1} = G(s_t + p − inj) + inj
```

With a **zero-init gate** (init ≈ 0.018, `BoundedScalarGate`) `inj → 0`, so `cancel` reduces to the plain loop *exactly* at init — a strict generalization, learned into, not imposed. Everything else is held fixed: the confound-free bounded-gate config (`prelude=coda=0`, 1 block × T=8, 4H/128D, 0.21M-param loop; 5.4K-param FM; `predict_k=3`; scalar gate), Fashion-MNIST (the loop-load-bearing task), seed 42.

`injection_form ∈ {update = gate·(FM−s), next_state = gate·FM}` is orthogonal to `inject_mode` (it defines the *signal*; the mode defines *how the signal combines with the operator*). The **update form** is the arc's canonical stable form and the cleaner-cancelling one; results below lead with it.

## Validation — the summation re-run reproduces the LOOPED headline exactly

The `add`-mode path is algebraically identical to the pre-existing code, and the re-run confirms it bit-for-bit at the result level: `sum_update` gives accT8 **0.847**, no-inj **0.667**, dependency **0.180**, gate **0.016**, and the robustness inversion **−0.18 at ε=2.0** — matching the documented LOOPED headline. So the cancellation numbers below are a trustworthy single-flag contrast against a faithful baseline.

## The core A/B (`compare_cancel`)

| condition | inject_mode·form | val_acc | accT8 | no-inj | dependency | inj_benefit | gate | fm_cos |
|---|---|---|---|---|---|---|---|---|
| OL | — | 0.869 | 0.852 | — | — | — | — | 0.994 |
| `sum_update` | add·update | 0.852 | 0.847 | 0.667 | **0.180** | −0.86 | 0.016 | 0.994 |
| `cancel_update` | cancel·update | 0.850 | 0.844 | 0.711 | **0.133** | −0.40 | 0.017 | 0.997 |
| `sum_nextstate` | add·next_state | 0.856 | 0.836 | 0.803 | 0.033 | −0.11 | 0.017 | 0.993 |

**Task accuracy is unchanged** (cancel 0.850 ≈ sum 0.852 ≈ OL 0.869) — cancellation pays no accuracy cost, as the doc predicts ("no net loss benefit is expected"). Clean-data dependency is *lower* under cancellation (0.133 vs 0.180; standalone no-inj acc 0.711 > 0.667). The scalar gate self-regulates to ~0.017 under **both** modes — cancellation does not change the resting gate value (see Payoff 4 caveat).

## Payoff 3 — the restoring force, confirmed three ways

### (a) Perturbation `error_correction` flips sign (`compare_cancel`)

`error_correction = dloss_noinj − dloss_inj` (>0 = the injection *recovers* from a mid-loop state perturbation; <0 = it *amplifies* the damage):

| condition | ε=0.5 | ε=1.0 | ε=2.0 |
|---|---|---|---|
| `sum_update` | −0.012 | −0.052 | **−0.177** |
| `cancel_update` | +0.025 | +0.054 | **+0.048** |

Decomposing ε=2.0 into damage *with* vs *without* the injection: summation goes no-inj **+0.038** → with-inj **+0.215** (the injection amplifies the perturbation); cancellation goes no-inj **+0.257** → with-inj **+0.209** (the injection *reduces* the damage). Same perturbation, same model family — only the wiring sign changed.

### (b) The mechanism: the forecast direction is cancelled (`mnist_looped_cancel_probe.py`, `cancel_mechanism`)

Net effect of the injection on the next state under the model's *own* wiring, `Δ = s_new(inj) − G(s_in)`:

| condition | mode | gain (‖Δ‖/‖inj‖) | **cos(Δ, inj)** | |
|---|---|---|---|---|
| `sum_update` | add | 4.05 | **+0.893** | amplifies (forecast propagates) |
| `cancel_update` | cancel | 2.06 | **+0.095** | forecast direction cancelled |
| `sum_nextstate` | add | 3.70 | +0.894 | amplifies |
| `cancel_nextstate` | cancel | 2.06 | +0.444 | partial |

The directional alignment collapses **+0.893 → +0.095** on the sign flip: the next state stops moving in the direction of the forecast — the corollary-discharge signature ("you can't tickle yourself"). What survives (gain ~2, orthogonalized) is the operator's processing of the *residual/deviation*, which is what predictive coding wants to propagate. **The cancellation is topological, not learned**: the counterfactual "feed each operator `+inj`" gives `cos_add ≈ 0.87–0.89` for *every* operator including the cancellation ones — the operators respond identically; only the subtract-then-re-add topology produces the cancellation ("restoring force by construction"). The `update` form cancels cleanly (0.095); `next_state` only partially (0.444) — subtracting the full forecast overlaps with the state itself.

*Note on magnitude:* the pre-registered prediction was gain ≪ 1; the operator is not near-identity (gain_add ≈ 2–4), so cancellation *orthogonalizes* (removes the forecast direction) rather than nulling the magnitude. The directional axis is the meaningful one (the arc's self-knowledge is directional throughout).

### (c) OOD-input robustness (`mnist_looped_p4_ood_probe.py`, `p4_ood`)

On Gaussian-noise-corrupted (OOD) inputs, the **summation** injection flips from helpful (b_acc +0.18) to **actively harmful** (−0.07 to −0.09 at every σ), while the **cancellation** injection stays **helpful at every corruption level** (never negative; peaks +0.25). Summation amplifies the OOD disruption; cancellation keeps the injection safe under distribution shift — a distribution-shift version of (a), connecting to the arc's OOD-robustness theme. *(Caveat: this sweep did not create a bad-*forecast* regime — forecast self-consistency `q` actually **rises** under input corruption because the low-rank loop's degenerate OOD dynamics are more predictable. It tests OOD robustness, not the subtract-wrong-forecast mechanism; that is Payoff 4.)*

## Payoff 4 — benefit tracks forecast quality (the basis for gate self-closing)

The doc's sharpest claim: under cancellation, subtracting a *wrong* forecast actively corrupts the forward path, so opening the gate is beneficial where the forecast is accurate and *harmful* where it is not — the endogenous pressure that would make an input-conditioned gate close on poorly-predicted inputs (which OOD_GATE needed bilevel meta-learning to obtain).

### The clean graded test: corrupt the forecast directly (`mnist_looped_p4_fc_probe.py`, `p4_forecast_corruption`)

Inject `(1−α)·FM_pred + α·random_pred`, norm-matched so only the *direction/quality* degrades (magnitude held constant); clean inputs. `b_acc = acc_inj − acc_noinj`:

| α | q_mix (cancel) | `sum_update` | `cancel_update` |
|---|---|---|---|
| 0.00 | 0.965 | +0.176 | +0.136 |
| 0.25 | 0.956 | +0.175 | +0.122 |
| 0.50 | 0.919 | +0.159 | +0.078 |
| 0.75 | 0.859 | +0.118 | +0.003 |
| 1.00 | 0.797 | **+0.037** | **−0.148** |

As the forecast degrades, **cancellation's benefit falls monotonically and flips negative** (+0.136 → −0.148; b_loss +0.41 → −0.51) — subtracting a wrong forecast corrupts. **Summation stays positive** (+0.176 → +0.037) — a wrong forecast is a benign extra input, right down to garbage. `acc_noinj` is constant (0.721), so the flip is *purely* the injected forecast getting wronger — clean causal attribution. The effect is **conservative**: even at α=1 the forecast is only partially degraded (q_mix 0.797, since the low-rank loop keeps even a random prediction ~0.8-cos with the future), yet cancellation already flips negative.

This is the loss-landscape asymmetry the doc says drives gate self-closing: under cancellation, opening the gate on a bad forecast **hurts** (endogenous pressure to close on poorly-predicted inputs); under summation it doesn't (no such pressure → the documented runaway gate). Corroborated by the `freshfm_swap` **random-FM arm**: a garbage forecast *helps* summation (0.696 > noinj 0.660) but *hurts* cancellation (0.667 < noinj 0.697).

**Untested**: the gate *dynamics* — whether a trainable *input-conditioned* gate actually closes on poorly-predicted inputs. The loss-landscape pressure that would drive it is established; the scalar gate here can't express input-dependence.

## Payoff 1 — modular dependency: substrate-limited (inconclusive)

The fresh-FM swap (`mnist_looped_freshfm_probe.py`) costs an **identical** 0.016 accuracy for both summation and cancellation — no differential modularity. But the reason is diagnostic: the fresh FM converges to *nearly the original* (forecast cos 0.968 / 0.993) because the low-rank loop's near-deterministic FM target is nearly unique, so a "different but equally-good" FM barely exists to swap in. The perturbation is too weak to test entanglement — the same low-rank property that sank the LOOPED runnable-simulator probe. **Not** evidence for or against Payoff 1; needs a substrate where the FM has real gauge freedom (language / RHM) or a genuinely-degraded swap.

## Interpretation

Flipping the injection wiring from summation to cancellation — **one flag, a strict generalization** — reverses the loop's response to disruption (amplify → correct) and makes the injection's benefit contingent on the forecast being *right*, exactly as `efference_copy_cancellation.md` predicts, with the mechanism directly measured (forecast direction cancelled, cos 0.89 → 0.095) and shown to be **topological, not learned**. Task accuracy is unchanged. The doc's central thesis — that the sign/topology, not learned behavior, drives the closed-loop pathologies — holds on the primary axes (Payoffs 3 and 4's mechanism).

## Caveats

- **Single seed, single setting** (Fashion, this confound-free config). The `add`-mode re-run reproducing the LOOPED headline exactly is the main cross-check that the setup is faithful.
- **Payoff 1 is substrate-limited** (FM near-unique in the low-rank loop), not resolved.
- **Payoff 4's gate *dynamics*** (does a trainable input-conditioned gate close on bad forecasts?) is untested — only the loss-landscape pressure that would drive it.
- The mechanism cancellation is **directional** (cos → 0.095), not a magnitude null (gain stays ~2); the operator amplifies, so what remains is the orthogonalized residual.
- `next_state` cancellation is only *partial* (cos 0.444; flips negative only at extreme forecast corruption). The clean effects are all in the `update` (deviation) form.
- The confounded first Payoff-4 attempt (`mnist_looped_p4_probe.py`, per-input `corr(b,q)` on clean ID data) is **superseded** by the α-corruption test: on ID data the trained FM is uniformly excellent (no bad-forecast regime) and benefit anti-correlates with `q` only via a classification-headroom confound.

## Reproduce

```bash
cd experiments/

# --- The A/B: train the 5 conditions (each a detached container). NOTE: stagger the
#     launches by ~25s -- 5 simultaneous `modal run` calls trip Modal's app-creation rate limit.
COMMON="--dataset fashion_mnist --prelude-layers 0 --coda-layers 0 --fwd-d-head 1 --fwd-mlp-mult 0.125 --predict-k 3 --gate-type scalar"
modal run --detach a2a_forward/mnist_looped_injection.py::train_condition $COMMON --condition ol_last --injection-form update     --inject-mode add
modal run --detach a2a_forward/mnist_looped_injection.py::train_condition $COMMON --condition cl_last --injection-form update     --inject-mode add
modal run --detach a2a_forward/mnist_looped_injection.py::train_condition $COMMON --condition cl_last --injection-form update     --inject-mode cancel
modal run --detach a2a_forward/mnist_looped_injection.py::train_condition $COMMON --condition cl_last --injection-form next_state --inject-mode cancel
modal run --detach a2a_forward/mnist_looped_injection.py::train_condition $COMMON --condition cl_last --injection-form next_state --inject-mode add

# --- Reads (all eval-only, on the checkpoints above) ---
# Core A/B table: gate, dependency, restoring-force error_correction (Payoff 3a)
modal run a2a_forward/mnist_looped_injection.py::compare_cancel --dataset fashion_mnist
# Mechanism: forecast-direction cancellation, gain + cos(Δ,inj) (Payoff 3b)
modal run a2a_forward/mnist_looped_cancel_probe.py::cancel_mechanism --dataset fashion_mnist --conditions sum_update,cancel_update,sum_nextstate,cancel_nextstate
# OOD-input robustness (Payoff 3c) + note the q-rises premise failure
modal run a2a_forward/mnist_looped_p4_ood_probe.py::p4_ood --dataset fashion_mnist --conditions sum_update,cancel_update,sum_nextstate,cancel_nextstate
# Payoff 4 clean graded test: benefit vs forecast corruption alpha
modal run a2a_forward/mnist_looped_p4_fc_probe.py::p4_forecast_corruption --dataset fashion_mnist --conditions sum_update,cancel_update,sum_nextstate,cancel_nextstate
# Fresh-FM swap (Payoff 1, inconclusive) + random-FM arm
modal run a2a_forward/mnist_looped_freshfm_probe.py::freshfm_swap --dataset fashion_mnist --conditions sum_update,cancel_update
```

Results JSON on the `language-reduction-data` volume under `/data/a2a_forward/mnist_looped_injection/` (`..._scalargate_cancel/`, `..._next_state_..._cancel/` tags) and `.../probes/`.

## Next steps

1. **Gate dynamics for Payoff 4**: add an input-conditioned gate to the cancellation loop on a mixed-difficulty / mildly non-stationary stream — does it close on poorly-predicted inputs *from the wiring alone* (no bilevel), given the loss-landscape pressure now established?
2. **Payoff 1 on a gauge-free substrate**: the fresh-FM swap needs a domain where the FM is not near-unique (language / RHM), or a genuinely-degraded swap, to test modular-vs-entangled dependency.
3. **Cancellation × the weight-shared loop's other threads**: does cancellation compose with internalization (subtracting the model's *own* endogenous forecast — the purest efference copy)?
4. **Port to RHM** (the idea doc's suggested substrate): ground-truth latents let us check whether downstream literally carries the residual `h_L − p` and whether the recombined read-out matches the true `h_L` — and the sculpting FM's re-grounding / Kalman patches are the drift-correction this generalizes.
