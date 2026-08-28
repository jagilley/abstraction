# rl_dimensionality — file reference

## Code files

| File | Purpose |
|---|---|
| `rl_dim_ablation.py` | The full experiment: BP references (exact-match ceilings/floors, parse floors, NTP Bayes floor), pretrain-to-plateau, conditions (scratch_rl / pretrain_rl / pretrain_only; `--only-kl` anchored arm; `--only-ei` expert-iteration arm), torch parse reward, per-checkpoint evals (per-level NTP, per-layer eta², greedy+sampled generation), `self_test` |
| `aggregate_results.py` | Base-sweep cross-m tables from the five `results.json` |
| `aggregate_kl.py` | KL-anchored (kl01) vs unanchored comparison |
| `aggregate_ei.py` | Three-way EI (ei01) vs KL (kl01) vs base comparison |
| `rl_dimensionality_colab.ipynb` | Self-contained Colab walkthrough: the setup explained, a scaled-down live replica (pretrain-to-plateau → REINFORCE → +KL → EI at one m, ~15 min on a T4), and the full-scale results embedded with the README's interpretation inline. No Modal/repo dependency — upload and run |

## Children

| Folder | Summary |
|---|---|
| [`idiolect/`](idiolect/README.md) | Synonym-choice drift under the synonymy-invariant verifier: reproducible in magnitude, largely arbitrary in direction, and co-occurring with rising validity and grammaticality. Inference-only re-analysis of this folder's checkpoints, plus a seed-43 privacy arm. File index in [`idiolect/FILES.md`](idiolect/FILES.md) |

## Auxiliary docs

| File | Summary |
|---|---|
| `README.md` | The writeup: bounds vs attainment across m ∈ {1,2,3,4,6}, three-mechanism ladder, the validity-up/rules-flat dissociation |
| `DESIGN.md` | Pre-run design record: knob rationale (m vs L/s/v), verifier choices, pretrain-to-plateau, reference-value findings from self-tests, follow-up-arm rationale, claim framing |
| `CONVERSATION.md` | Session provenance: Jasper's prompts verbatim + summarized assistant turns, preserving the motivations behind the work for future agents |
