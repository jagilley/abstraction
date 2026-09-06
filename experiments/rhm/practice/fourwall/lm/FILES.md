# fourwall/lm — files, machinery, gates, and caveats

**Up**: [`README.md`](README.md) (the writeup; read it first — this file is the lookup record).

## Code files

| file | purpose |
|---|---|
| `wall.py` | Wall primitives: key/rotation schedules, exact BP index oracles (`exact_predictive`, Gate 0 quantities), no GPU — gate-testable standalone. |
| `wall_lm.py` | The Modal app: `gate` (G-1–G-7), `run_arm` (one arm = one worker = one GPU), `wall_lm` driver (spawns arms in parallel). Round 2 added the merge schedules, the `none` eval mode, `vs_none_idx`, and the 3-condition probes. |
| `analyze_wall_lm.py` | Fetch + reduction + figures (`fig1–fig5` per tag; `fig6_merge` when `--ref-tag` given). Section 9a = cross-tag bit-identity gate; 9b = op admissibility. |
| `launch_detached.py` | Session-isolated `modal run --detach -m` launcher (fork+setsid so a harness TaskStop cannot cancel the remote run); logs to `results/launch_<tag>.log`. |

## Tags and arms

Shared config (both tags): v16/s2/L6/m4, rule_seed 0; key = level-2 node 0 (indexed span = leaves
0–15 of 64); 8L/8H/256D GPT, 20 000 steps × batch 64 = 81.9M tokens/arm; lr 3e-4, wd 0.01 (AdamW,
all params); data_seed 7, seed 42; rotation = cyclic +4 mod 16 (orbit 4). One worker per arm,
identical seeds → bit-identical trajectories until schedules diverge. Exact per-tag config:
`figures/<tag>/setup.json`.

**`fwlm0`** (ckpt_every 250): `true_wall` (bijection, never rotates) · `wall` (phase 1 to 8000,
rotate every 2000, identity return at 14000) · `wall_fast` (rotate every 250 from 250) ·
`dead_wall` (wall token present, uninformative) · `no_wall` (neutral filler always) ·
`dead_wall_b` (= dead_wall at seed 43; the seed/stream floor pair).

**`fwlm1`** (ckpt_every 125, + `merge_at + {0,25,50,75}` micro-grid): `no_wall` (in-tag ref) ·
`merge_1000` / `merge_8000` / `merge_13000` (the `wall` schedule, merged at the named step) ·
`merge_fast_8000` (`wall_fast` schedule, merged at 8000) · `merge_dead_8000` (`dead_wall`
schedule, token swap at 8000 — the op's admissibility control). Merge = wall token → the neutral
filler `no_wall` consumes; post-merge batches are token-identical to `no_wall`'s (same pool, same
sampler).

## Instruments (glossary)

- `nll idx / out` — indexed-span (leaves 0–15) vs outside-span NLL. **Consumed condition** = the
  token the arm actually sees (`true` pre-merge, `none` post-merge).
- `bind` — NLL(wrong wall) − NLL(right wall): comparative binding (3.7–5.6× the exact bracket by
  construction; a wrong key misleads, an absent one doesn't). `vsnone` (`vs_none_idx`, fwlm1+) —
  NLL(neutral) − NLL(right wall): matches Gate 0's definition (right key vs no key, 0.1520).
- `gap` / `jsd` / `card` — forced-transfer gap, wall-conditional behavioural spread, and
  behavioural index cardinality over the 16 wall ids (clustering thresholds τ = 0.0002/0.001/0.005).
  Post-merge these read the **abandoned** circuitry (decay curves).
- `map x/y` — implied routing table match to the current / previous rotation map (granularity
  1/15 = 0.0667 = its floor).
- `key-anchor d4 true/rand/none` — probe recovery of the keyed latent with wall present /
  randomised / neutral; exact ceilings 1.000 / 0.921 / 0.921. `rand`/`none` read the token-derived
  inference pathway. Probe-free twin: indexed-span excess over exact Bayes per arrival level.
- Oracle capture — (arm NLL − `no_wall` NLL) / exact bracket, indexed span, pre-merge only.

## Gates and calibration record

- **G-1/G-2** schedule structure: bijection; every rotation a derangement; orbit 4; identity return
  at 14000 (`wall` family).
- **G-3** MI(w; z) = 2.578 = H(z) for live walls; 0.028 nats for `dead_wall`'s token.
- **G-4** `wall.exact_predictive` vs the donor's `prefix_beliefs` leaf posterior: max|Δ| = 0.0.
- **G-5** clamping the keyed node pins its posterior at exactly 1.000.
- **G-7 / Gate 0** (exact, model-free): right index worth 0.1520 nats/token on the indexed span
  (Bayes level 1.4503; 10.5%), 0.0028 outside — concentration 54.6×; total 2.566 ≈ H(z) = 2.578.
  Probe ceilings: key anchor d4 0.921 no-wall / 1.000 given; last anchor d6 0.798 / d4 0.981
  (reproduces `reread/lm`'s 0.798 / 0.984 on a fresh eval draw). Key demand is the DGP's own:
  H = 2.578/2.773, one feature at exactly zero mass → 15 live cells; per-cell readouts run over
  live cells only.
- **Floors** (dead pair, last quarter, mean |Δ|): nll idx 0.0012 / out 0.0014 / bind 0.00005 /
  gap 0.0001 / map 0.0635. Placebo rotation response at non-rotation checkpoints (p90, n=71):
  nll idx 0.0039 / bind 0.032 / gap 0.037.
- **9a bit-identity** (fwlm1 vs fwlm0 twins, all shared pre-merge checkpoints, five instruments):
  max|Δ| = 0.0 for all six arms (4–81 checkpoints each). Structural basis: training consumes zero
  global torch RNG (Dropout(0.0) short-circuits; batch `ix` drawn before the merge branch).
- **9b op admissibility**: `merge_dead_8000` transient +0.0032 at M+0, never above +0.0071 —
  ~97% of the merge arms' transient (+0.102/+0.165/+0.086) is index deletion, not the token swap.
  Transients decay within 25–125 steps; unindexed-span movement ≤ +0.014 at M+0 (12× index-local)
  and ≤ +0.004 by M+50.

## Caveats (read before trusting any single number)

- **`dead_wall_b` is the only replicate.** Terminal ±0.004
  differences among merged arms and `no_wall` are unrankable (1–3× placebo floor). Claims ride on
  trajectories, integrals, transients, floor-multiples.
- **The fwlm0 `wall_fast` correction (README finding fwlm1-3), in numbers**: fwlm0's 250-step grid
  ≡ its rotation period, so every readout sat 0 steps into a new map. fwlm1's 125-grid, steps
  4000–8000 pre-merge: era-boundary 1.6593 / mid-era 1.4218 / `no_wall` 1.5217 — mid-era 0.100
  *better*; era-averaged net +0.019 (~5× floor) vs the reported +0.149/+0.202 (~8× overstated).
  fwlm0's `e_old`/`gap_old`/`map_match_old` for that arm were always unconfounded. Nothing else in
  fwlm0 is touched.
- **fwlm0 probe-checkpoint placement** sat on rotation boundaries; the last mature in-era probe
  point for `wall` is 14250. fwlm1's grid fixes this. fwlm0 terminal readouts for rotating arms
  use the last in-era checkpoint (19750), since 20000 lands on an unrun era boundary.
- **The collapse target is out-of-distribution for a wall arm at the instant of the op**; priced
  by `merge_dead_8000` at +0.003 (at floor, not zero).
- **The last-anchor (unindexed node) d4 contrast** (~0.50 keyed vs ~0.63 controls) is within ~1.5×
  the seed spread on that readout (`dead_wall` 0.666 vs `dead_wall_b` 0.567) — suggestive only;
  the key-anchor contrast is ~6× it.
- **Local launch logs truncate**: `launch_fwlm0.log` is complete; `launch_fwlm1.log` ends at
  Modal's detach message (client stream dropped mid-run) — all fwlm1 numbers were fetched from the
  volume. Results are committed per-checkpoint remotely; the local log is convenience only.
- d5/d6 lack dynamic range at this model size; d4 anchors the story. `map_match` floor = 1/15.
- Cross-tag comparisons are licensed **only** through gate 9a's bit-identity; anything not covered
  by a twin pair should be compared in-tag.

## Volume layout & figures

`/data/rhm_practice_fourwall_lm/{fwlm0,fwlm1}/` on `rhm-scaling-data`: per-arm records + per-tag
`setup.json`. Fetched: `figures/<tag>/…` + `results/launch_<tag>.log`. Figures per tag:
`fig1_competence` (indexed/unindexed NLL + Bayes lines), `fig2_binding` (capture + stale
attachment), `fig3_index` (transfer gap / routing match / behavioural spread), `fig4_levels`
(probe grid: anchor × wall condition), `fig5_perwall` (per-latent routing rasters); fwlm1 adds
`fig6_merge` (consumed-condition NLL + d4, round-1 references dashed).
