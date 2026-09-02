# direction — file reference

**Parent**: [../FILES.md](../FILES.md) · **Writeup**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `drift_direction.py` | Three inference-only Modal functions over saved checkpoints: `selection_test` (replays one best-of-k parse selection step at the pretrained checkpoint and every EI-parse round of both seeds; valid-conditioned — see gotcha), `competence_test` (teacher-forced loss per (level, feature, synonym) on ground-truth sequences), `measure_controls` (synonym histograms + loss for the selection-free and fresh-pretraining-seed arms). Shared helpers `selection_stats`, `loss_by_rule` |
| `aggregate_direction.py` | Parts A–D: angle decomposition of the parent's committed sweep (geometry check, per level, shared/private, bias projection, rounds, bootstrap CI and noise null), the replayed selection step, competence alignment, and the causal controls |

## Result files

| File | Contents |
|---|---|
| `selection_differential.json` | Sample vs winner synonym histograms, per-rule reward and upward validity, at pretrained and every EI-parse round of both seeds (m = 2, 3, 4, 6; 1500 prompts × 16) |
| `competence.json` | Teacher-forced loss by (level, feature, synonym) at pretrained and both EI-parse finals (20 000 ground-truth sequences) |
| `controls.json` | Same protocol for the control arms at m = 2 and 6: random-selection EI rounds; fresh-pretraining-seed-44 pretrained + EI-parse rounds |

## Gotchas

- **The replayed selection differential is conditioned on grammaticality on both sides**, so it cannot see selection acting *through* grammaticality; it sits at its noise floor at every m. `competence.json` is the instrument for the competence question.
- **Control-arm run dirs carry the reward type in their name**: `v8_s2_L6_m{M}_random_seed42_ei_rand/ei_random_*.pt` and `v8_s2_L6_m6_parse_seed44_pt44/{pretrained,ei_parse_*}.pt`. The parent's `_run_dir` assumes `both`; `_control_checkpoints` here spells these paths out.
- **`reward_type=random` is EI-only in spirit.** It is wired through `compute_reward`, so a REINFORCE arm would also accept it, but a random reward there is just noise; only `--only-ei` uses it meaningfully.
