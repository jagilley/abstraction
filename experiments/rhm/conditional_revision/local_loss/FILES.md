# Files — local_loss

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `temporal_local_loss.py` | The whole cut. A pure target swap on [`../../rhm_fm_regularizer.py`](../../rhm_fm_regularizer.py) (`post_block6[≤t] → post_block6[t+1]` instead of `post_embed[t] → post_block6[t]`), keeping its λ warmup, open-loop gradient path and FM config so the torch RNG stream matches the incumbent batch-for-batch. `train_temporal_reg` / `tll_sweep` train the arms; `rank_traj` walks FM-free activation rank over the **incumbent** depth checkpoints (read-only, no FM trained) to locate the matched-step read against the endpoints; `analyze_ll` / `analyze_ll_ckpt` compute every readout on every checkpoint with **both** fresh forward models — knowledge gate, per-level rule probe, FM-free act rank / top1 / mean dim var, fresh-FM cos and res/tgt ratio, SK and meta-object probes in both alignments, ensemble agreement, the `delta_over_tgt` and `mse_over_deltavar` scale-free ratios, the η² feature/rule allocation, and the matched counterfactual twins. `smoke` is the end-to-end check. Read-only against `_fmreg_dir` / `_sweep_dir`; writes under the `tll_` prefix so the incumbent cannot be clobbered. |
| `__init__.py` | package marker |

## Auxiliary docs

| file | contents |
|---|---|
| `SPEC.md` | The design written before the run: the degeneracy argument being tested, why this was preferred to the parent's remaining gates, the arms and readouts, the substrate tradeoff (m2 anchors vs m4 frontier), the replacement degeneracy and the twin instrument sketched for it, and what each outcome would have meant. Retained as written; the README records what actually happened, including where the spec's framing did not survive. |
| `README.md` | The writeup — what we tried, the gauge finding, compression, self-knowledge on both forward models, the synonym-discard null, and a fairly long account of what the cut does **not** establish. |
