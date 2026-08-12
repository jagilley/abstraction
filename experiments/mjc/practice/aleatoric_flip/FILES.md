# FILES — aleatoric_flip

**Up**: [../README.md](../README.md) (practice) · [../../FILES.md](../../FILES.md) (mjc)

## Code files

| file | purpose |
|---|---|
| `aleatoric_flip.py` | The runner (Modal). Fork of [`../priced_plasticity/priced_plasticity.py`](../priced_plasticity/priced_plasticity.py) with one changed thing in the task geometry: a region spec gains a `noise_pre` field, so a region can carry its command rotation φ in **both** worlds while its aleatoric noise switches on at t=0. Builds three worlds (`pre` / `post` / `clean`), probes the flip region in the `clean` world (the retention instrument), measures whether the noise is mean-preserving (`cond_mean_check`), walks a shadow random-prior ensemble over the stream to precompute a causal per-sample disagreement signal in two variants (`shared` / `boot`), and runs the arm set. Two modes: `--mode calibrate` (FM probes only — the preconditions and the damage-channel ladder) and `--mode main` (adds control grading and the uniform-lr frontier). No change to `pusher_env.py` is required: `_apply_rot_regions` already sums a region's rotation and its stochastic force. |
| `analyze_flip.py` | Local aggregation + figures. Fork of [`../priced_plasticity/analyze_priced.py`](../priced_plasticity/analyze_priced.py) — the Pareto-frontier machinery is unchanged — plus three tables this node needs: `--damage` (the flip-amplitude × learning-rate ladder, with the `f0` no-flip column that separates over-writing damage from interference-from-elsewhere), `--condmean` (the mean-preservation check with its own MC null), `--disag` (the ensemble's per-class disagreement profile by quartile). A *cell* here is (capacity, flip amplitude). |
| `launch_detached.py` | Session-isolated (`start_new_session=True`) `modal run --detach` launcher — the `plasticity_gain` gotcha fix, so a harness TaskStop cannot cancel the remote input and swallow the `volume.commit()`. |

## Arm grammar

`kind[@lr][:spend][:norm][:rN][:xM][:fA][:tauon][:boot][:hH][:LL]`

| token | meaning |
|---|---|
| `fixed` / `raw_err` / `delta` | inherited from the parent: uniform w≡1, w ∝ e, w = exp(−δ/τ) capped |
| `disag` | w ∝ the shadow ensemble's predictive spread `d` |
| `conj` | w ∝ exp(−δ/τ)·`d` — δ certifies convergence, disagreement certifies reducibility |
| `:fA` | flip amplitude for this arm's world (`:f0` = the no-flip control cell) |
| `:tauon` | τ re-scaled to the online δ spread instead of the pretrain MAD (the graded gain) |
| `:boot` | use the bootstrap disagreement variant instead of the shared-batch one |
| `:free` / `:unit` / `:raw_adam` | the parent's spend factorization (free spend / allocation-only / parent pipeline) |

## Data layout

Modal volume `mujoco-control-data:/data/aleatoric_flip/<tag>/results.json` (+ `done.txt`);
local mirrors in `data/<tag>.json`, launcher logs in `results/launch_<tag>.log`, figures in
`figures/`.
