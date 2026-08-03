# Files — `variable_modulus`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `variable_modulus.py` | The cut. Builds the shared encoder / tied operator / decoder and runs the five arms (`blind`, `fold`, `cond`, `fold_cyc`, `cond_cyc`), which differ in **where the rule lives** rather than whether the model has it. Owns the digit-only `RuleEncoder` (deliberately *not* a readout from the bidirectional prompt encoder, which would leak `x_0` into the re-injected rule), the conditioned `Block` (`h + MLP(LN(h) + W_r r)`), on-the-fly trajectory generation in int64, the three fixed eval pools (seen-`N`/seen-`x`, seen-`N`/held-out-`x`, held-out-`N`) built once so every arm is scored on identical problems, and the instruments: `exact@T`, `verid@t`, `inrange@t`, `on_manifold_cos@t`, `rule_separation@t` (**confounded — see README §5**), `coldstart`, and the causal `ruleswap`. Logs `reachable_states` at startup, which is the quantity governing learnability. All knobs are CLI flags; `--test-x-fraction` is the coverage manipulation and `--max-moduli` / `--d-ff` the capacity ones. |
| `reprojection.py` | **README §2/§4.** Test-time re-projection, on Modal: loads a checkpoint, trains nothing, re-runs the rollout with a periodic snap `h <- (1-alpha)*h + alpha*Enc(N, decode(h))`. Sweeps period `k`, blend `alpha`, and `mode` — `self` (own decode, deployable) vs `oracle` (true residue, the ceiling, and therefore a direct read of the per-restart rate `p`). Rebuilds the DGP and the train/test split from the checkpoint's own `cfg`, and **asserts that the `k=0` condition reproduces the stored rollout** before any re-projection number is read — a silent pool mismatch would otherwise corrupt every comparison. |
| `analyze.py` | Cross-seed aggregation for one tag: exact-match across all three OOD axes, the composition horizon (right-censored, with the 50%-crossing interpolated), the cycle×conditioning gain table, the rule-swap contrast, the instrument curves, cold start, and `figures/<tag>/fig_variable_modulus.png` (depth extrapolation with the rule-swap control dashed · the three generalisation axes · rule separation · state closure). |

The modulus *family* itself (`ModulusFamily`) lives one level up in
[`../squaring_mod.py`](../squaring_mod.py) alongside `depth_first_repeat`, since it is a DGP
knob of the shared task rather than of this cut; [`../shared.py`](../shared.py) carries the Modal
app/image/volume.

## Result tags

| tag | what it is |
|---|---|
| `cut1` | **The main cut** — 5 arms, 3 seeds, eval to T=28, checkpointed. 26 train / 7 held-out moduli. |
| `reproj_cut1` | Test-time re-projection over `cut1`'s checkpoints, 3 seeds. The oracle ceiling is 0.894–0.903 across arms whose closure spans 0.15→0.97 — README §2. |
| `cov0.02` / `cov0.30` / `cov0.50` | **The coverage sweep** — the `p` manipulation, with `cut1` supplying the 0.90 point. `p ≈ coverage` (0.982 / 0.894 / 0.473) while the raw horizon stays flat at 9.5–11.9. Single seed. |
| `grok_d1_wd1` / `grok_d1_wd01` / `grok_d16_wd1` | **The grokking probe** — 8 moduli, 50% of bases held out, constant LR, 500k steps. No phase transition at any setting; held-out accuracy sits at the analytic no-reduction floor — README §3. |
| `cap_m8` … `cap_m47`, `arity_m8` / `arity_m16` | Capacity and arity sweeps over train-modulus count × operator width. Horizons 8.6–11.6 across 7× state count — the basis for the retraction in README §5. |
| `probe_wide` / `probe_mid` / `probe_narrow` | 4-digit modulus bands at 15k steps. All at chance; **abandoned before the 60k budget that makes 3-digit work was known**, so they do not establish that 4 digits is infeasible. |
| `probe_d3` | First working 3-digit configuration (47 moduli, 15k steps) — still climbing at cutoff. |

Results land on the Modal volume `one-layer-deeper-data` at
`/variable_modulus/<tag>/results_seed<N>.json`, checkpoints at
`/variable_modulus/<tag>/ckpt/<arm>_seed<N>.pt`, and mirror locally to `results/<tag>/`.

## Figures

| path | content |
|---|---|
| `figures/cut1/fig_variable_modulus.png` | Depth extrapolation for all five arms with the wrong-rule control dashed; the three generalisation axes side by side (held-out `N` is the arity axis); rule separation vs `t`; state closure and `inrange` vs `t`. |
