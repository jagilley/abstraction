# partial_hetero — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md);
parent index at [`../FILES.md`](../FILES.md).

The DGP primitive lives three levels up at [`../../../rhm_channels.py`](../../../rhm_channels.py)
(per [`STRUCTURE.md`](../../../../../STRUCTURE.md) — code lives at the lowest node that shares it),
because the sharing knob is a property of the channel DGP, not of this experiment:
`share_rules_top` splices a channel's top-*k* rule tables in from an earlier channel, and
`make_layout` accepts `share_top` / `share_from` on any grammar channel. The environment and the
loop are one level up in [`../channel_env.py`](../channel_env.py) (`make_spec`'s `struct_shares` /
`rule_seed_offset`, and the `oracle_shared` allocation rung) and [`../ladder.py`](../ladder.py)
(`--struct-shares`). All of those additions default to the published behaviour, so every prior run
on this node is reproducible from the same scripts.

## Code files

| File | Purpose |
|---|---|
| `geometry_check.py` | **G1–G3** certifying the partially-heterogeneous DGP before the loop runs on it. G1 re-asserts P1 structural irrelevance at every sharing depth (sharing rule tables is exactly the kind of change that could break it). G2 shows the surface stays distinct — legal-leaf-tuple overlap with the tree, plus `level_sibling_stats`, an exact forward-sampled per-level TV that reads the sharing depth straight out of the data and is what makes a G3 null interpretable. G3 is the **transfer curve**: fresh block FMs trained under fixed channel allocations at matched budget and matched state distribution, in a **coverage-matched** design (`tree_structA` vs `tree_half` holds tree-block transitions exactly fixed) because the FM's per-block `action_embedding` makes the naive `structA_only` arm confound grammar transfer with parameter coverage. Static — no drift anywhere |
| `aggregate.py` | Reads the mirrored `figures/` JSONs and prints the two curves: G3's added-value/exchange-rate table with the `+structA − +structB` contrast that cancels the generic extra-optimizer-steps effect, and E1-PH's per-policy tree FM error with the uniform→oracle, uniform→visits_only and oracle→oracle_shared prizes. Per-seed least-squares slopes against sharing depth with t-stats, so monotonicity is tested rather than eyeballed |

## Results

| Path | What |
|---|---|
| `figures/geometry_g_s{1,2,3}/results.json` | G1–G3 across sharing depths 0–4 on the **published block FM** — the readout the ladder actually runs on. One rule draw per seed (`rule_seed_offset = seed − 1`, so seed 1 is the ladder sweep's own rule draw) |
| `figures/geometry_gsm_s{1,2,3}/results.json` | The same sweep with `--fm-arch shared_marker` — the **positive control for the readout**, removing the only parameters private to a block. Retained because it did not rescue the measurement, which is what ruled the parameterisation out as the binding constraint |
| `figures/geometry_gcl_s{1,2,3}/results.json` | The same sweep with `--fm-arch channel_local` — the **position-invariant readout** (no absolute-block parameters, within-channel attention, per-block latent centering), plus the **G4 encoder-alignment** diagnostic. The sharing knob is flat here too, and the `share_top=4` cell carries the crisp invariance failure: at statistically identical channels the half-tree/half-structA arm lands on `tree_half` (0.683 vs 0.677), not on `tree_only` (0.621) |
| `figures/ladder_ph_k{0..4}_s{1,2,3}/results.json` | E1-PH — the allocation ladder at each sharing depth, six policies (`uniform`, `reducible_only`, `visits_only`, `value_red`, `oracle`, `oracle_shared`), 12 rounds under continuous drift |
