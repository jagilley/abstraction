# Files — `one_layer_deeper`

**Up**: [README.md](README.md) (this node) · [../CLAUDE.md](../CLAUDE.md) (experiments)

## Code files

| file | purpose |
|---|---|
| `squaring_mod.py` | The repeated-modular-squaring task, vendored from `tilde-research/one-layer-deeper` (`data/squaring_mod.py` + `data/counting.py`). Verbatim: `TOKEN_IDS`, `DIGIT_OFFSET`, `number_tokens`, `trapdoor_squaring_mod`. Ours: `TaskSpec` (the DGP knobs, with `describe()` reporting the depth-periodicity margin and reachable-state count), `depth_first_repeat` (the tail+period of `2^T mod λ(N)` — the depth at which the task admits a periodicity shortcut instead of a serial rollout, which any eval range must sit strictly below), `build_trajectories` (full `x_0..x_T` trajectories, cross-checked against the independent trapdoor label path), `tokenize_prompt` / `answer_digits` / `encode_split`. Two documented departures from upstream: it emits intermediate residues (research-only instrumentation, never a training target) and pads answers to fixed width. |
| `shared.py` | Modal infrastructure — image (torch 2.7.0), volume `one-layer-deeper-data`, app `one-layer-deeper`, `NumpyEncoder`. |

## Children

| child | summary |
|---|---|
| [`ballistic_depth/`](ballistic_depth/README.md) ([FILES](ballistic_depth/FILES.md)) | **What actually extends a learned operator's composition horizon.** Terminal-CE-only gives T≈13; one label-free "the state you rolled into must be one your encoder could have produced" constraint gives T≈51 (3.9×) at its swept optimum, and held-out-x 0.001 → 0.32. Mechanism is **manifold closure** (cos 0.19 → 0.99) — and acting on it *at test time*, with no retraining, takes the grounded arm to **1.000 at T=60**. Reproducing *rollability ≠ depth* on a new substrate; the predicted error-amplification mechanism is falsified five ways with the instrument calibrated on-manifold first. A non-recurrent untied stack is perfect at every trained depth and at chance one step past it. Hard quantisation fails to learn twice while achieving genuine per-step contraction — *error correction, achieved and useless* — and the label-reusing half of the consistency term is dose-dependently harmful, inverting the dense/evaluative complementarity the cut was designed around. |
