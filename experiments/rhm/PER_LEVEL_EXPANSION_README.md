# Per-Level Loss Expansion: bracketing the composition ceiling (2026-07-13)

**Code**: `rhm_per_level_expansion.py` (orchestrator), `rhm_per_level_loss.py::per_level_trajectory` (primitive), `plot_per_level_expansion.py` (figure)
**Prior experiment**: [Per-Level Loss Decomposition](PER_LEVEL_LOSS_README.md)
**Figure**: `per_level_expansion_ceiling.png`

## Idea

The original per-level-loss experiments showed bottom-up learning and a composition
ceiling on **two** settings that confound model size, data, and synonymity `m`:

| | model | m | tokens | reached |
|---|---|---|---|---|
| Exp1 | 4L/4H/128D (0.8M) | 2 | 5M | ~2–3 levels |
| Exp2 | 6L/6H/192D (2.7M) | 4 | 20M | ~1–2 levels |

Because Exp2 changed **both** the model and `m`, the drop in ceiling could not be
cleanly attributed. The essay draft (`papers/neural_networks_learn_bottom_up.md`)
read this pair as evidence for "roughly one level of hierarchy per layer." This
experiment turns those two anecdotes into **controlled sweeps** that vary one axis at
a time, to answer: *is the ceiling set by model capacity (depth/width) or by `m`?*

## Design

All runs: L=6, s=2, 20M tokens, 40k steps, per-level cross-entropy ("wave") metric,
on the same plain `generate_rules` (with-replacement) corpus pipeline as the original
per-level-loss experiments. Chance accuracy = 1/v.

- **Depth sweep** — v8, m2, width fixed 256D, `n_layer ∈ {2,4,6,8}`. Isolates depth.
- **Synonymity sweep** — v8, capacity fixed 8L/8H/256D, `m ∈ {2,4,6,8}`. Isolates `m`.
  (Shares its m=2 point with the depth sweep's 8L run.)
- **Root existence proof** — v16, m2, 8L/256D. The one plain-NTP setting prior work
  validated (via latent-ancestor probe = Bayes-optimal) to solve to the root.
- **Capacity control** — v8, m2, `n_layer ∈ {2,8}` × `n_embd ∈ {32,64,128}`. Kills
  the "depth looked flat only because 256D width already saturated capacity" confound.

Code changes to `rhm_per_level_loss.py::per_level_trajectory` (backwards-compatible;
defaults reproduce prior behavior): added `n_steps_override` (the function hard-capped
steps at 20k; the root climb needs ~40k) and `weight_decay`; bumped GPU T4→L4 and
timeout. The orchestrator must import `app` from `rhm_per_level_loss` (that module
defines its own `rhm-per-level-loss` app, **not** the shared `rhm-scaling` app) or
`.spawn()` fails to hydrate.

## Results

### Capacity is irrelevant at m=2

Final per-level accuracy, v8/m2 (chance 0.125). Ten models spanning ~30K to ~6.3M
params (>200×):

| model | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 2L/32D  | 0.809 | 0.415 | 0.417 | 0.244 | 0.267 | 0.231 |
| 8L/32D  | 0.823 | 0.439 | 0.432 | 0.278 | 0.288 | 0.240 |
| 2L/64D  | 0.813 | 0.428 | 0.425 | 0.267 | 0.274 | 0.245 |
| 8L/64D  | 0.828 | 0.442 | 0.438 | 0.283 | 0.291 | 0.243 |
| 2L/128D | 0.822 | 0.435 | 0.431 | 0.272 | 0.280 | 0.238 |
| 8L/128D | 0.830 | 0.443 | 0.438 | 0.282 | 0.293 | 0.240 |
| 2L/256D | 0.827 | 0.440 | 0.435 | 0.281 | 0.287 | 0.241 |
| 4L/256D | 0.830 | 0.444 | 0.435 | 0.282 | 0.293 | 0.244 |
| 6L/256D | 0.830 | 0.444 | 0.439 | 0.281 | 0.288 | 0.240 |
| 8L/256D | 0.831 | 0.444 | 0.440 | 0.279 | 0.294 | 0.244 |

The total spread across the entire ~200× capacity range is ~1–2pp at every level. The
tiniest model (2L/32D) reaches the same levels as the largest (8L/256D); both stall at
~L2 with L3–L5 hovering at ~2× chance. There is a slight, monotonic capacity benefit
(8L consistently ~1pp above 2L; wider ~1pp above narrower), but it does not change
*which* levels are reached. **The old Exp1 (4L/128D) reproduces almost exactly** (L0
0.815→0.822, L3 0.263→0.272), so the original numbers stand; only the interpretation
changes.

### Synonymity sets the ceiling

Final per-level accuracy, v8, capacity fixed 8L/8H/256D (chance 0.125):

| m | occupancy m/v | L0 | L1 | L2 | L3 | L4 | L5 | reaches |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.25 | 0.831 | 0.444 | 0.440 | 0.279 | 0.294 | 0.244 | ~level 2 |
| 4 | 0.50 | 0.592 | 0.310 | 0.209 | 0.199 | 0.200 | 0.209 | ~level 1 |
| 6 | 0.75 | 0.476 | 0.193 | 0.175 | 0.160 | 0.162 | 0.167 | ~level 0 |
| 8 | 1.00 | 0.343 | 0.180 | 0.179 | 0.189 | 0.195 | 0.195 | barely level 0 |

At **fixed, generous capacity**, increasing `m` walks the ceiling monotonically down
from ~level 2 (m=2) to ~level 0 (m=8). m=8 is the clean "synonymity crushes it to a
single level" endpoint (and even L0 is weak at 34%).

### Reaches-root proof

v16/m2/8L (chance 0.0625) vs v8/m2/8L (chance 0.125), as accuracy ÷ chance:

| level | v8, m2 (× chance) | v16, m2 (× chance) |
|---|---|---|
| 0 | 6.65 | 14.3 |
| 1 | 3.55 | 6.66 |
| 2 | 3.52 | 5.94 |
| 3 | 2.23 | 5.44 |
| 4 | 2.35 | 4.03 |
| 5 (root) | 1.95 | 3.46 |

Lower occupancy (v16) lifts the entire wave and keeps it clearly above chance all the
way to the root (3.5× at L5 vs ~2× for v8). Note the per-level **cross-entropy** at the
root cannot collapse to zero even for a perfect model — predicting the actual next token
across the root boundary carries irreducible entropy — so the CE wave under-dramatizes
root-solving. The sharp "root solved" signal lives in the *latent-ancestor probe* metric
from prior work ([RHM_FRONTIER_AND_LEGIBILITY](RHM_FRONTIER_AND_LEGIBILITY_README.md):
root recovery 0.93 = Bayes-optimal at exactly v16/m2/8L).

## Key findings

1. **The composition ceiling at m=2 is not a capacity limit.** Depth (2→8L) and width
   (32→256D), across >200× params, leave it essentially unchanged. This refutes the
   "one level of hierarchy per layer" reading of the original Exp1-vs-Exp2 comparison:
   that comparison changed `m` alongside the model, and `m` was doing the work.
2. **Synonymity `m` is the ceiling dial.** At fixed generous capacity, m=2→4→6→8 walks
   the reachable depth from ~2 levels down to ~0. This is consistent with the
   signal-dilution account (Cagnetta & Wyart; KFW per-level cost ~ `v·m^{ℓ+2}`) and with
   the frontier arc's "signal not capacity" diagnosis, now shown directly on the essay's
   CE-wave metric with capacity held fixed.
3. **Occupancy (`m/v^{s-1}`) sets the height, not the reach.** Lowering occupancy via
   larger `v` (v8→v16 at m=2) lifts the whole wave and lets it reach the root above
   chance; it does not change the depth/capacity irrelevance.
4. **Bottom-up ordering is untouched.** Every setting still learns strictly leaves-first;
   only the stall depth moves.

## Relation to the essay

These results bear directly on `papers/neural_networks_learn_bottom_up.md` (the ceiling
section's "one level per layer" claim and the "stalls when depth runs out" mechanism).
**Per request, no essay edits were made** — this doc records the results for a later
decision on how (or whether) to reframe.

## Reproduction

```bash
cd experiments/
# Main grid (depth sweep + synonymity sweep + root proof)
modal run --detach rhm/rhm_per_level_expansion.py::per_level_expansion
# Capacity control (depth x width at v8/m2)
modal run --detach rhm/rhm_per_level_expansion.py::per_level_width_control
# Figure (local)
python3 rhm/plot_per_level_expansion.py
```

Per-run trajectories saved to `rhm-scaling-data` volume at
`/data/rhm_per_level_loss/trajectory_*.json`.
