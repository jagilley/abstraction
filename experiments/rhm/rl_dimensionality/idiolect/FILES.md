# idiolect — file reference

**Parent**: [../FILES.md](../FILES.md) · **Writeup**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `idiolect_drift.py` | Exact `(feature, rule)` recovery from generated sequences (`build_rule_inverse`, `recover_rule_usage`), the synonym-choice statistics (`idiolect_stats`: `kl_cond`, `H_marg`/`H_cond`, `valid_frac`), `js_divergence_bits` for comparing two models' dialects, the `analyze` Modal sweep over every `rl_dim_ablation` checkpoint (inference only), and `self_test` (recovery vs generation traces, validity vs `possible_set_parse`, metric calibration at both endpoints) |
| `aggregate_idiolect.py` | Cross-m tables from a results JSON: drift (sampled and greedy), grammaticality, root validity, EI round-by-round trajectories, and the pairwise JS divergences |

## Result files

| File | Contents |
|---|---|
| `idiolect_results_seed43.json` | The complete sweep — all mechanisms × 5 m × 2 verifiers, EI per-round checkpoints, the same-seed rerun, and the seed-43 privacy arm. **Quote numbers from this one**; it is the aggregator's default. |
| `idiolect_results_full.json` | The first sweep, before the seed-43 arm existed. Kept as the record of the run reported mid-session; its generations are independently sampled, so values differ from the above in the third decimal |

## Gotchas

- **`generate_rules_invertible` is load-bearing.** Exact rule recovery requires collision-free rules; `build_rule_inverse` raises if the rules are not. It also means possible-set cardinality is always 0 or 1, so parse *ambiguity* is definitionally absent from this substrate — grammaticality is the only parse-side signal available.
- **The greedy `kl_cond` columns are near-saturated for every model, pretrained included**, because argmax decoding is deterministic and so necessarily commits to one synonym. Use the sampled columns; the greedy ones are retained in the JSON but do not measure drift.
- **`kl_cond` is upward-biased at finite `n`**; always read it against the `dgp` row at matched node counts, and note that low-grammaticality conditions contribute fewer valid nodes and so carry more bias.
