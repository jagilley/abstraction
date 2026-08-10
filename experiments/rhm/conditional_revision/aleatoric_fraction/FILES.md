# Files — aleatoric_fraction

**Writeup**: [README.md](README.md) · **Up**: [../README.md](../README.md) · **Parent**: [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `aleatoric_fraction.py` | **Sizing step for the idea doc's §8** — how much of `h6[t+1]`'s conditional variance is irreducible, and how much of *that* an NTP-optimal model would be free to discard. Pure measurement on the frozen cached m4 base; nothing is trained. Applies the law of total variance to the state update using BP-exact weights (`oracle.py`'s `return_leaf_posteriors`): conditional on the prefix a causal model's `h[t+1]` takes exactly `v = 16` values, so `A_D` and `A_D + E_D` are 16-term weighted sums over counterfactual activations (one frozen forward pass per arriving token per position), and the readout `A_D/(A_D+E_D)` is scale-free — immune to the gauge collapse that made `local_loss/`'s raw term uninterpretable. `ntp_classes` supplies the **NTP-protected floor** by an exact local criterion (two arriving tokens are interchangeable iff `down_u ⊗ #{r : rule matches}` is proportional), giving a second law-of-total-variance split of `A_D` into `A_prot` (NTP forces it) and `A_free` (the prize). `verify_classes` proves that criterion against direct BP **inside every run**; `synonym_swap_check` is the model-free version of the floor argument (re-realise a completed constituent from the same feature, ask whether the exact Bayes next token moves). Reports all nine blocks, the per-arrival-level stratification, bootstrap CIs over sequences, a model-free isotropic-token-code ceiling for the prize, and an A/E principal-angle statistic. Runnable directly for the DGP-only checks with no Modal and no model. |
| `__init__.py` | package marker |

## Outputs

| file | contents |
|---|---|
| `/data/v16_s2_L6_m4_distinct/conditional_revision/aleatoric_fraction_af1_seed42.json` | The run. Per-block × per-`D` aleatoric fraction / floor / prize / free-over-aleatoric with model-free references, the arrival-level stratification, bootstrap CIs, subspace statistics, the DGP synonym-swap table, and every self-check (oracle identity, emitted-posterior consistency, criterion-vs-BP, local-criterion-vs-oracle, substitution-vs-plain-forward). |
