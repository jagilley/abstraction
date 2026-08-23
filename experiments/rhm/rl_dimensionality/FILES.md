# rl_dimensionality — file reference

## Code files

| File | Purpose |
|---|---|
| `rl_dim_ablation.py` | The full experiment: BP references (exact-match ceilings/floors, parse floors, NTP Bayes floor), pretrain-to-plateau, conditions (scratch_rl / pretrain_rl / pretrain_only; `--only-kl` anchored arm; `--only-ei` expert-iteration arm), torch parse reward, per-checkpoint evals (per-level NTP, per-layer eta², greedy+sampled generation), `self_test` |
| `aggregate_results.py` | Base-sweep cross-m tables from the five `results.json` |
| `aggregate_kl.py` | KL-anchored (kl01) vs unanchored comparison |
| `aggregate_ei.py` | Three-way EI (ei01) vs KL (kl01) vs base comparison |

## Auxiliary docs

| File | Summary |
|---|---|
| `README.md` | The writeup: bounds vs attainment across m ∈ {1,2,3,4,6}, three-mechanism ladder, the validity-up/rules-flat dissociation |
| `DESIGN.md` | Pre-run design record: knob rationale (m vs L/s/v), verifier choices, pretrain-to-plateau, reference-value findings from self-tests, follow-up-arm rationale, claim framing |
| `CONVERSATION.md` | Session provenance: Jasper's prompts verbatim + summarized assistant turns, preserving the motivations behind the work for future agents |
