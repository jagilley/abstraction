# ear — File Index

Complete file-by-file reference for this node. Summary and findings: `README.md` (written after
the results are discussed).

## Code files

| File | Purpose |
|---|---|
| `grader.py` | **The evaluation layer, one level up.** `manufacture_damage` builds a self-manufactured audition context: the agent overwrites the level-(k+1) node enclosing its current damage cell with chunks drawn from its own tables — either two level-k children independently (`mfg_source="child"`, available from cycle 1, drawn from a different table than the one under audition) or one whole-node level-(k+1) entry (`mfg_source="node"`, a legal chunk but only once catalogued) — reading what it is overwriting with its own reader and rendering with the same `canon` every move already uses. Nothing in it touches `rules`. `context_stats` is the oracle check (on-grammar, `d*`, `frac_broken`, `node_empty`, `node_overlap`) against the real damage's signature — an instrument, never acted on. `policy_e` is the second coordinate of the climb: grade a candidate by running the agent's OWN performance policy (widest beam fitting the declared grounding budget) with the candidate macro instantiated at every node of its level, rather than by one isolated application. `gate_manufacture` (G-N nesting + G-S on-grammar/confinement/levelling) and `gate_recert` (G-R: a live swap restores the level-3 table the ratchet's cap truncated) |
| `ear.py` | The Modal app. Forks `ratchet` by **importing** its substrate (`build_shared`, `beam_moves`, `context_instances`, `finetune_generator`, `value_steps`, `measure_refs`, `plant_probe`, `macros`) so nothing about the world changes and the ratchet-path arms stay bit-comparable; `_shared_plus` only adds the numpy renderings self-manufacturing needs. `ARMS` carries three new axes per arm — `grader` (which cell of the (ctx × ev) grid the unit-LP certificate reads), `pay_era` (whether the ratchet's per-cycle era-k audition is on this arm's meter), `recert` (whether a committed table is re-graded live in consumption and can be **swapped**). `run_arm` adds: a **bank** of the agent's own clean derivations (the raw material a self-manufactured test is made of), the once-per-era manufactured audition set, the full 3×2 grader grid computed as an instrument for every arm every cycle with only the read cell priced, provisional commitment (`prov` / `delta_prov`, which skips the pre-commit audition entirely), and the priced live recert. Entrypoints: `ear` (main), `cal_mfg` (calibration 1 — is a manufactured test a faithful test, and does the policy evaluator have the range the action evaluator lacks), `selfcheck{,_remote}` (the ratchet's C-M / C-R / G-D plus G-N / G-S / G-R) |
| `analyze_ear.py` | Reduction. `--fetch` pulls from the volume. `--fidelity` compares this fork's ratchet-path arms against `ratchet/figures/rr_s0` cycle for cycle. The default report gives per-era competence and priced time, the earned-vs-given fraction, matched-priced-time margins, commit **and recert** events, the **grader grid** (every audition series per arm with the read cell marked), the **oracle check** (manufactured-vs-real signature and, the part that matters, the same candidate's reading on both sets), **grader-cost accounting** (paid vs would-have-cost per grid cell), the depth seam against every grader, compounding with an explicit step-noise readout, mined-vs-random and the plant guard. `grader_grid` replays the certificate offline over every cell of the grid across a (c, c_v, W, hold, `lp_min_drop`) grid — the detector calibration that costs no GPU. `--cal` reads `cal_mfg.json`. `--figures` writes fig1–fig4 |
| `launch_detached.py` | Session-isolated detached launcher (`start_new_session=True`), so a harness signal cannot cancel the remote input mid-run. `--fn` selects the entrypoint (`ear`, `cal_mfg`); logs to `results/launch_<tag>.log` |

## Inherited, not copied

The substrate is `../ratchet/`'s, imported rather than duplicated so that `rr_s0` stays
reproducible and this round's ratchet-path arms are comparable to it cycle for cycle:
`ratchet/macros.py` (the earned vocabulary, the max-sum macro operator, `grade_table`) and
`ratchet/ratchet.py` (the depth ladder, matched pricing, the beam, the reader, the online plant
and selector, the references). `crystallize/units.py` supplies the level-indexed action space,
hierarchical damage and the exact-DP oracle.

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_competence.png` | Competence per cycle and against cumulative priced time; era boundaries, commits (dotted) and recert swaps (stars) marked |
| `figures/<tag>/fig2_graders.png` | The grader grid: the level-2 and level-3 candidate's audition under `era/act` (the ratchet's), `mfg/pol` and `real/pol`, per arm, commits marked |
| `figures/<tag>/fig3_oracle.png` | The oracle check as a scatter — the same candidate's audition on self-manufactured contexts against its audition on real level-k+1 contexts, both evaluators |
| `figures/<tag>/fig4_vocab_cost.png` | Earned-table recall against the DGP's vocabulary, and the cumulative priced cost of the whole grader grid |

## Results on the volume

`rhm-scaling-data` → `/data/rhm_practice_ear/<tag>/`, with `<arm>/results.json` per arm plus
`setup.json`, and `cal_mfg.json` for the calibration entrypoint.
