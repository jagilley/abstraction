# crystallize — File Index

Complete file-by-file reference for this node. Summary and findings: [README.md](README.md).

## Code files

| File | Purpose |
|---|---|
| `units.py` | The node's primitives, all ported off `directed_sculpting/full_loop/`'s channel layout onto the plain Stage-3b grammar. `build_move_set` / `node_features` / `apply_move` / `apply_sequence` — the level-indexed action space (a move commits to one level-ℓ feature via a max-sum DP over the generator's evidence and renders the legal subtree beneath it; level 1 is bit-identical to `_regenerate`, gate C1; optional NOOP so a program can spend less than the budget). `corrupt_hier` — hierarchical damage, 100% on-grammar, so no error is block-locally visible. `on_grammar_rate`, `oracle_rollout` (exact-DP greedy over the same move set — the floor reference), `grade` (possible-set success + residual `d*`) |
| `crystallize.py` | The Modal app. `build_shared` trains controller/generator/value once (value on the **generic** distribution, so it arrives stale on the hierarchical contexts); `beam_moves` is the re-grounded token beam over level moves with per-tip sequence tracking, MC-label trajectories and grounding counts; `run_unit` / `run_library` / `run_fixed_span` are the committed-unit routings; `select_units` is the compile op (uniform candidate draw, open-loop scoring on held-out consumption instances, in-sample **and** half/half held-out audition, `key ∈ {root, global}`); `perform` + `priced` are metering and pricing; `run_arm` is the practice → plasticity → metering → width-ladder → **shadow compile** → gate loop; `discriminate` is the compile-hit probe (selected library/single, modal move, verbatim span, modal span with on-grammar rate, random, DP oracle, beam ladder). Entrypoints: `crystallize` (main), `cal_unit` (exhaustive compilation headroom), `cal_plant` (the plant precheck, with `headroom_cell` + `finetune_generator`), `selfcheck{,_remote}` (gates C1/C2/G-D) |
| `analyze_crystallize.py` | Reduction. `--fetch` pulls from the volume; default report gives the priced grade, matched-priced-time margins against `never`'s own interpolated trajectory, the commit/anchor table with optimism gaps, and the discriminator ladder. `--cal` reads `cal_unit.json`; `--detector` replays the δ-silence detector offline over a recorded e-series across a (c, c_v, W, hold) grid — the calibration that costs no GPU; `--figures` writes fig1–fig3 |
| `launch_detached.py` | Session-isolated detached launcher (`start_new_session=True`), so a harness signal cannot cancel the remote input mid-run. `--fn` selects the entrypoint (`crystallize`, `cal_unit`, `cal_plant`); logs to `results/launch_<tag>.log` |

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_metering.png` | Per-context `e` and `δ` over cycles, compile events marked, stale and DP-oracle references |
| `figures/<tag>/fig2_priced.png` | Cumulative priced time × mean error — the headline axis |
| `figures/<tag>/fig3_ground.png` | Groundings per solve (log scale): 46 → 2 at commit |
| `figures/<tag>/` | Also the mirrored `setup.json`, per-arm `results.json`, and `cal_unit.json` / `cal_plant.json` |

## Results on the volume

`rhm-scaling-data` → `/data/rhm_practice_crystallize/<tag>/`, with `<arm>/results.json` per arm plus
`setup.json`, and `cal_unit.json` / `cal_plant.json` for the calibration entrypoints. Tags:
`calu_s0`, `calu_s1`, `cald_s0`, `cald_s1`, `cg_s0`, `calp_s0`.
