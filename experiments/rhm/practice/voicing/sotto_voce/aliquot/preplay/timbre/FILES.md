# timbre — machinery record

**Up**: [`../FILES.md`](../FILES.md) (preplay) · **Decisions, gates, defects**: [`NOTES.md`](NOTES.md).
Written up in [`../../rubato/README.md`](../../rubato/README.md) §2 (2026-09-24).

## Code files

| file | purpose |
|---|---|
| `readers.py` | the reader forms over the projection's pooled state: `feats` (the hidden block, bit-identical to `preplay.pj_features`, and the plant's block-head log-softmax from the same pass, pooled the same two ways), `fit_lin` (`pj_fit`'s solve on any feature block), `fit_mlp` (128 GELU units, BCE, an (lr x wd) validation ladder trained as one batched ensemble, early stopping), `read_heads` (one trunk pass per trunk per fire, every head read off it), `xr_keys` (the leakage hash) |
| `timbre.py` | the Modal job: `rr1_arm` (pp1's single and delta forms and pp3's step 2, every reader on the same fires, arm 0 gated in-container against the banked pp1 / pp3 JSON), `rr5_arm` (pp5's walk with every reader's level as a gate, the banked gates gated in-container against the banked pp5 JSON), `decide_g` / `gated_walk_g` (pp5's rules for any reader key), the CPU coordinator `sweep`, `gates_t`, `falsify_t`, `mlp_recipe` |
| `reduce_timbre.py` | the reducer: rewrites rr1 / rr5 in pp1's / pp5's own schema per reader and runs `../reduce_preplay.py` and `../reduce_readgate.py` UNCHANGED on each view (pointed at a scratch directory through their `HERE`), then writes the cross-reader tables |

## Figures (facts only)

| file | what |
|---|---|
| `figures/timbre_reduction.txt` | the cross-reader tables of record: [T0] gates and leakage, [T1] each reader on the bank's held-out rows, [T2] pp1 true-vs-wrong AUC by reader x trunk x level, [T3] Spearman with the world's price and within the true class, [T4] the level and its compression, [T5] per-instance transfer, [T6] pp3's pricing table, [T7] pp5 capture and composition by reader x trunk, pooled and per seed, [T8] the bill |
| `figures/rr1_<R>_reduction.txt` | `reduce_preplay.py`, unchanged, on reader R's columns (R in ridge, belief, bonly_lin, mlp, bonly, slot); the `banked` column is the in-loop projection in every view |
| `figures/rr5_<R>_reduction.txt` | `reduce_readgate.py`, unchanged, on reader R's gates (R in ridge = arm 0 = pp5's own, ridge_refit, belief, mlp, bonly) |
| `figures/<tag>/rr1/<arm>.{json.gz,log}`, `figures/<tag>/rr5/<arm>_ab.{json.gz,log}` | the per-arm outputs as written on the volume, gzipped locally by the reducer |

## Runs

| tag | what | containers | cost | app |
|---|---|---|---|---|
| `tb_smoke` | rr1 + rr5 on s0_sv at smoke sizes, the first MLP recipe (volume only) | 3 L4 | 0.09 GPU-h | attached |
| `tb1` | rr1 on s0_sv, s2_sv, s0_so, s2_so, s0_yd, s2_yd; rr5 on s0_sv, s2_sv | 8 L4 | 0.65 GPU-h | `ap-FeFKiR0oYSJ4zEQObQZ6p5` |

Volume: `rhm-scaling-data:/rhm_practice_preplay/<tag>/{rr1,rr5}/`. Inputs read: the banked soundboard arms
(`/rhm_practice_soundboard/<tag>/<arm>/{vo_heads.pt, vo_bank.npz, results.json}`), overtone's frozen plants, and
the banked `/rhm_practice_preplay/{pp1,pp3,pp5}/` JSON the gates compare against.

## Gates

| script | gates | falsified |
|---|---|---|
| `timbre.py` | T-1 features (hidden == `pj_features` bitwise, belief == `block_logits`); T-2 heads (banked == `pj_predict` bitwise, `fit_lin` == `pj_fit`); T-3 the MLP fits an XOR; T-4 the walk == `readgate.gated_walk`, generic forms == pp5's, handed -e == world; T-5 arm 0 == banked pp1 / pp3 (in-container); T-6 pp5's banked walks exactly (in-container); T-7 the leakage counter | 7/7 |
