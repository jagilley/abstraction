# FILES — practice/estimability

**Up**: [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)

## Code files

| file | purpose |
|---|---|
| `estimability.py` | The runner. Forks `mjc/bridge_assembly/bridge_assembly.py` (corrected δ assembled live: context-conditional b(s), centered gate σ((g−g₀)/θ), gain consumption, budget matching) and replaces **only** the stream generator: the online stream becomes 1000 context-pure batches, 200 per context over 5 contexts, re-ordered by a burst-length schedule (massed → spaced → interleaved) at **exactly fixed per-context sample count**. Adds the measured oracle error field E[e\|pos] (recomputed with each arm's own FM, so the truth moves with competence), the passive benchmark panel (net×4 lrs, per-context tabular EWMA×3 αs, global scalar EWMA, frozen), revisit-lag / position-in-burst logging, and the credit-fidelity readouts. Modal entrypoint `estimability`; one container per schedule via `.map()`. |
| `analyze_estimability.py` | Local post-processing (no compute). Prints the learning-outcome table, the estimability surface (every estimator × every schedule), the active-arm fidelity table, and the absent-context Δb-vs-Δtruth regression; writes `figures/<tag>/fig1..fig6`. |
| `launch_detached.py` | setsid-isolated `modal run --detach` launcher (the `plasticity_gain` cancellation gotcha fix). Logs to `results/launch_<tag>.log`. |

## Children

None yet.

## Runs on disk

| tag | what it is |
|---|---|
| `est_s0/s1/s2` | the burst ladder (200/50/20/5/2/1), all five arms, 3 seeds — the main result |
| `est_replay_s0/s1/s2` | the same at bursts 200/20/1 with `--replay-mode uniform` — the interleaver control |
| `est_panel_s0/s1` | `--arms fixed` only, extended panel (`net` 1e-4…1e-2, `ewma_ctx` 0.02…0.5) — the benchmark-timescale × revisit-rate crossing |
| `est_panel_edges_s0` | the same, closing both grid edges (`net` 3e-5…3e-3, `ewma_ctx` 0.2…1.0) — confirms both optima are interior |
| `smoke` | `--quick` smoke (bursts 20/1, tiny nets) |

## Data layout

- Modal volume `mujoco-control-data`: `/data/practice_estimability/<tag>/burst<k>/results.json` (+ `done.txt`).
- Local mirror written by the entrypoint: `results/<tag>/burst<k>.json`.
- Figures: `figures/<tag>/`.
