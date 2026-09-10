# idiolect — file reference

**Parent**: [../FILES.md](../FILES.md) · **Writeup**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `idiolect_drift.py` | Exact `(feature, rule)` recovery from generated sequences (`build_rule_inverse`, `recover_rule_usage`), the synonym-choice statistics (`idiolect_stats`: `kl_cond`, `H_marg`/`H_cond`, `valid_frac`), `js_divergence_bits` for comparing two models' dialects, the `analyze` Modal sweep over every `rl_dim_ablation` checkpoint (inference only), and `self_test` (recovery vs generation traces, validity vs `possible_set_parse`, metric calibration at both endpoints) |
| `aggregate_idiolect.py` | Cross-m tables from a results JSON: drift (sampled and greedy), grammaticality, root validity, EI round-by-round trajectories, and the pairwise JS divergences |
| `idiolect_colab.ipynb` | Self-contained Colab notebook for external readers, telling the same story as `papers/rl_idiolect_slides.tex`. Ports the grammar, the exact recovery, the two verifiers and the expert-iteration loop inline (no `rhm` imports, no volume reads); runs the four self-tests and the corpus floor live at the sweep's own sample size; runs a scaled-down replica (4L/128D, ~37K rollouts per arm, ~15 min on a free T4) across all four arms plus REINFORCE; and embeds the full sweep as a gzip+base64 extract of `idiolect_results_seed43.json`, from which every distance and angle is recomputed rather than quoted. Built by a script kept out of the repo; regenerate by re-running it |
| `lexicon.py` | The same drift read out as a dictionary rather than as bits: at the bottom level a `(feature, rule)` pair *is* a literal token string, so the committed `joint_hist` says how often each model uses each word. Prints one feature's full synonym table across the arms, that word's round-by-round trajectory in both seeds, and how much of the level's `v*m`-word vocabulary each run moved. Numpy only — reads the committed JSON and regenerates the rules locally |

## Result files

| File | Contents |
|---|---|
| `idiolect_results_seed43.json` | The complete sweep — all mechanisms × 5 m × 2 verifiers, EI per-round checkpoints, the same-seed rerun, and the seed-43 privacy arm. **Quote numbers from this one**; it is the aggregator's default. |
| `idiolect_results_full.json` | The first sweep, before the seed-43 arm existed. Kept as the record of the run reported mid-session; its generations are independently sampled, so values differ from the above in the third decimal |

## Children

| Folder | Summary |
|---|---|
| [`direction/`](direction/README.md) | Why the two seeds' drift directions converge as m grows: verifier selection on grammaticality leaking onto the synonym coordinate through uneven competence at synonyms; selection-free and fresh-pretraining-seed controls. File index in [`direction/FILES.md`](direction/FILES.md) |

## Gotchas

- **`generate_rules_invertible` is load-bearing.** Exact rule recovery requires collision-free rules; `build_rule_inverse` raises if the rules are not. It also means possible-set cardinality is always 0 or 1, so parse *ambiguity* is definitionally absent from this substrate — grammaticality is the only parse-side signal available.
- **The greedy `kl_cond` columns are near-saturated for every model, pretrained included**, because argmax decoding is deterministic and so necessarily commits to one synonym. Use the sampled columns; the greedy ones are retained in the JSON but do not measure drift.
- **Seed 42's round-by-round checkpoints are the `ei02` run's, not `ei01`'s.** `ROUND_CONDITIONS` builds `ei_{arm}_r{1..6}` from the `ei02` tag, so any round trajectory for seed 42 ends at the same-seed rerun's final value rather than at the `ei_{arm}` (ei01) row's. The two agree to ~0.2 points on a synonym rate, which is itself a useful reading of the noise floor — but don't present a trajectory's last point and the `ei_parse` row as the same number.
- **`kl_cond` is upward-biased at finite `n`**; always read it against the `dgp` row at matched node counts, and note that low-grammaticality conditions contribute fewer valid nodes and so carry more bias.
