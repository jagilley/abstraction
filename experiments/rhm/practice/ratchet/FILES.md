# ratchet — File Index

Complete file-by-file reference for this node. Summary and findings: [README.md](README.md).

## Code files

| File | Purpose |
|---|---|
| `macros.py` | **The earned vocabulary.** `true_tables` renders the DGP's own composition rules into the macro representation (`T[l]` = entries, each an s-tuple of `T[l-1]` **entry indices** — which is what makes the ratchet structural: a level-3 chunk whose halves are not in the level-2 vocabulary is unrepresentable and dropped at build time). `Miner` counts level-1 feature tuples observed over one span and `build`s a table against a given lower table at a support threshold. `parse_features` is the agent's own read (`mask_block` exists only because an earlier diagnosis blamed distribution shift; `calp2_s0` showed that was wrong — see the README's calibration record). `macro_features` / `apply_any` / `apply_seq` are the max-sum DP over the learned table and the move dispatch; `to_device` materialises the table chain once. `grade_table` is the oracle check (precision / recall of an earned table against the grammar's own, as sets of flattened level-1 tuples) |
| `ratchet.py` | The Modal app. `parse_eras` / `era_ctx` / `context_instances` build the nested depth ladder (L1 n6 ⊂ L2 n3 ⊂ L3 n1) with `with_clean` for the holdout tagging; `beam_ground` / `fit_width` implement matched pricing (the widest beam that fits the declared per-solve grounding budget, per action-set size); `beam_moves` is the re-grounded token beam over a **mixed** action set; `train_reader` trains the bottom-level parse the generator's masked-only loss never supervises; `holdout_set` / `filter_pool` / `needs_excluded` implement the stale plant and its solvability gate; `finetune_generator` and `value_steps` are the online plant and selector; `run_arm` is the era loop (practice → mine → plant → selector → metering → shadow audition → certificate → commit → probes); `plant_probe` is the self-imitation guard; `measure_refs` takes the stale reference **on the metering set itself** (round 1's bug, fixed). Entrypoints: `ratchet` (main), `cal_ladder` (the depth ladder + coverage curve), `cal_parse` (the perception diagnostic), `cal_stale` (the plant frontier and its gate), `selfcheck{,_remote}` (gates C-M / C-R / G-D) |
| `analyze_ratchet.py` | Reduction. `--fetch` pulls from the volume; the default report gives per-era competence and priced time, the **earned-vs-given fraction**, matched-priced-time margins, commit events with table precision/recall, the depth seam (commit audition vs realised competence one era later), the unit-LP trajectories (`A_cand` / `A_true` / recall), the compounding readout and the plant guard. `--cal` reads `cal_ladder.json`. `replay_unitlp` / `unitlp_grid` replay the certificate offline over a recorded audition series across a (c, c_v, W, hold, `lp_min_drop`) grid — the calibration that costs no GPU and that set the run's detector. `--figures` writes fig1–fig3 |
| `launch_detached.py` | Session-isolated detached launcher (`start_new_session=True`), so a harness signal cannot cancel the remote input mid-run. `--fn` selects the entrypoint (`ratchet`, `cal_ladder`, `cal_parse`, `cal_stale`); logs to `results/launch_<tag>.log` |

## Figures

| Path | What |
|---|---|
| `figures/rr_s0/fig1_competence.png` | Competence per cycle and against cumulative priced time, era boundaries and commit events marked |
| `figures/rr_s0/fig2_unitlp.png` | The level-2 and level-3 shadow-audition trajectories per arm against the true-table floor, commits marked — the unit-LP curves |
| `figures/rr_s0/fig3_vocab.png` | Earned-table recall against the DGP's own vocabulary, per level |
| `figures/<tag>/` | Also the mirrored `setup.json`, per-arm `results.json`, and `cal_ladder.json` / `cal_parse.json` / `cal_stale.json` |

## Results on the volume

`rhm-scaling-data` → `/data/rhm_practice_ratchet/<tag>/`, with `<arm>/results.json` per arm plus
`setup.json`, and `cal_ladder.json` / `cal_parse.json` / `cal_stale.json` for the calibration
entrypoints. Tags: `smoke0`, `smoke1`, `call_s0`, `calr_s0`, `calp2_s0`, `cals_s0`, `calr_h3`,
`calr_h5`, `calr_h0`, `rr_s0`.
