"""Layer gap sweep: how does forward model prediction quality scale with
the number of intervening layers?

Loads the frozen 77M model (8L/8H/512D) from the model scale experiment
and trains identical forward models at three prediction gaps:
  - post_block0 → post_block1 (1-layer gap)
  - post_block0 → post_block2 (2-layer gap)
  - post_block0 → post_block3 (3-layer gap)

All forward models use the same architecture (1L/1H/64d/mlp×2, ~1.18M params,
~1.5% of main model). The only variable is the prediction gap.

Includes causal substitution for each gap: replace the skipped blocks with
the forward model's prediction and measure KL divergence vs normal operation.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

GAP_CONFIGS = [
    {"name": "1_layer", "predict_to": "post_block1", "skip_blocks": [1]},
    {"name": "2_layer", "predict_to": "post_block2", "skip_blocks": [1, 2]},
    {"name": "3_layer", "predict_to": "post_block3", "skip_blocks": [1, 2, 3]},
]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_layer_gap_sweep(
    n_tokens: int = 2_000_000,
    block_size: int = 128,
    batch_size: int = 64,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    n_analysis_batches: int = 20,
    n_causal_batches: int = 50,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Main model config (77M)
    n_layer, n_head, n_embd = 8, 8, 512

    print(f"Layer gap sweep on {device}")
    print(f"  Main model: {n_layer}L/{n_head}H/{n_embd}D (frozen)")
    print(f"  Forward model: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp×{fwd_mlp_mult}")
    print(f"  Gaps: {[c['name'] for c in GAP_CONFIGS]}")

    # --- Load data ---
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    actual_n_tokens = len(data)
    data = torch.from_numpy(data.astype(np.int64))
    print(f"  Loaded {actual_n_tokens:,} tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # --- Load frozen 77M model ---
    ckpt_dir = f"{DATA_DIR}/a2a_forward/model_scale/gpt_8L_8H_512D/P_100000000"
    model_path = os.path.join(ckpt_dir, "model.pt")
    assert os.path.exists(model_path), f"77M model not found at {model_path}"

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    main_n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded frozen model: {main_n_params:,} params from {ckpt_dir}")

    # =========================================================
    # Phase 1: Cache frozen activations at all relevant layers
    # =========================================================
    print("\n=== Phase 1: Caching activations ===")
    layers_to_cache = ["post_block0", "post_block1", "post_block2", "post_block3"]

    def cache_activations(split_data, label):
        n_sequences = len(split_data) // block_size
        cached = {layer: [] for layer in layers_to_cache}
        cached_x = []
        with torch.no_grad():
            for start in range(0, n_sequences * block_size, block_size * batch_size):
                end = min(start + block_size * batch_size, n_sequences * block_size)
                batch_seqs = []
                for s in range(start, end, block_size):
                    if s + block_size <= len(split_data):
                        batch_seqs.append(split_data[s:s + block_size])
                if not batch_seqs:
                    break
                x = torch.stack(batch_seqs).to(device)
                _, _, intermediates = model(x, return_intermediates=True)
                for layer in layers_to_cache:
                    cached[layer].append(intermediates[layer].cpu())
                cached_x.append(x.cpu())
        result = {layer: torch.cat(cached[layer], dim=0) for layer in layers_to_cache}
        xs = torch.cat(cached_x, dim=0)
        print(f"  {label}: {xs.shape[0]} sequences")
        mem_gb = sum(v.numel() * 4 for v in result.values()) / 1e9
        print(f"  Memory: {mem_gb:.1f} GB")
        return result, xs

    train_acts, train_x = cache_activations(train_data, "train")
    val_acts, val_x = cache_activations(val_data, "val")

    # Pre-compute per-position LM losses on val data
    print("  Computing per-position LM losses...")
    all_lm_losses = []
    with torch.no_grad():
        for i in range(0, val_x.shape[0], batch_size):
            batch_x = val_x[i:i + batch_size].to(device)
            per_pos = model.get_ngram_losses(batch_x)
            all_lm_losses.append(per_pos.cpu())
    val_lm_losses = torch.cat(all_lm_losses, dim=0)

    # Pre-decode tokens for behavioral analysis
    enc = tiktoken.get_encoding("gpt2")
    val_x_np = val_x.numpy()
    N_val, T = val_x_np.shape

    sentence_start_mask = np.zeros((N_val, T), dtype=bool)
    before_closer_mask = np.zeros((N_val, T), dtype=bool)
    SENTENCE_END = set(".!?")
    CLOSER_CHARS = set(")]}")

    for b in range(N_val):
        decoded = [enc.decode([int(val_x_np[b, t])]) for t in range(T)]
        for t in range(T):
            if t > 0:
                prev_s = decoded[t - 1].strip()
                if prev_s and prev_s[-1] in SENTENCE_END:
                    sentence_start_mask[b, t] = True
            if t < T - 1:
                next_s = decoded[t + 1].strip()
                if next_s and next_s[0] in CLOSER_CHARS:
                    before_closer_mask[b, t] = True

    print(f"  Sentence starts: {sentence_start_mask.sum():,}, "
          f"before closers: {before_closer_mask.sum():,}")

    # --- Causal substitution helpers ---
    def run_embedding(idx):
        tok_emb = model.transformer.wte(idx)
        pos = torch.arange(0, idx.size(1), device=idx.device)
        pos_emb = model.transformer.wpe(pos)
        return model.transformer.drop(tok_emb + pos_emb)

    def run_blocks(x, start, end):
        for i in range(start, end):
            x = model.transformer.h[i](x)
        return x

    def run_head(x):
        x = model.transformer.ln_f(x)
        return model.lm_head(x)

    def get_val_batch():
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # =========================================================
    # Phase 2: Train and analyze forward models for each gap
    # =========================================================
    save_root = f"{DATA_DIR}/a2a_forward/layer_gap_sweep"
    os.makedirs(save_root, exist_ok=True)

    all_results = []

    for gap_cfg in GAP_CONFIGS:
        gap_name = gap_cfg["name"]
        predict_to = gap_cfg["predict_to"]
        skip_blocks = gap_cfg["skip_blocks"]
        resume_block = skip_blocks[-1] + 1

        print(f"\n{'=' * 60}")
        print(f"Gap: {gap_name} (post_block0 → {predict_to})")
        print(f"  Skipped blocks: {skip_blocks}, resume from block {resume_block}")
        print(f"{'=' * 60}")

        # --- Train forward model ---
        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
        capacity_ratio = fwd_n_params / main_n_params
        print(f"  Forward model: {fwd_n_params:,} params ({capacity_ratio:.1%})")

        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01
        )

        train_src = train_acts["post_block0"]
        train_tgt = train_acts[predict_to]
        n_train = train_src.shape[0]

        history = {
            "fwd_mse": [], "val_fwd_mse": [],
            "val_cosine_sim": [], "val_residual_norm": [],
        }

        for step in range(n_steps):
            fwd_model.train()
            idx = torch.randint(n_train, (batch_size,))
            src_batch = train_src[idx].to(device)
            tgt_batch = train_tgt[idx].to(device)

            pred = fwd_model(src_batch)
            fwd_loss = F.mse_loss(pred, tgt_batch)

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == n_steps - 1:
                fwd_model.eval()
                with torch.no_grad():
                    val_mse_acc = 0.0
                    val_cos_acc = 0.0
                    val_res_acc = 0.0
                    val_src = val_acts["post_block0"]
                    val_tgt = val_acts[predict_to]
                    n_val_seqs = val_src.shape[0]
                    n_vb = min(n_eval_batches, max(1, n_val_seqs // batch_size))
                    for _ in range(n_vb):
                        vidx = torch.randint(n_val_seqs, (batch_size,))
                        vs = val_src[vidx].to(device)
                        vt = val_tgt[vidx].to(device)
                        vp = fwd_model(vs)
                        val_mse_acc += float(F.mse_loss(vp, vt))
                        val_cos_acc += float(
                            F.cosine_similarity(vp, vt, dim=-1).mean()
                        )
                        val_res_acc += float((vt - vp).norm(dim=-1).mean())

                    val_mse = val_mse_acc / n_vb
                    val_cos = val_cos_acc / n_vb
                    val_res = val_res_acc / n_vb

                    history["fwd_mse"].append((step, float(fwd_loss)))
                    history["val_fwd_mse"].append((step, val_mse))
                    history["val_cosine_sim"].append((step, val_cos))
                    history["val_residual_norm"].append((step, val_res))

                    if step % (eval_interval * 5) == 0 or step == n_steps - 1:
                        print(f"  step {step:6d}: "
                              f"mse={float(fwd_loss):.5f} val_mse={val_mse:.5f} "
                              f"cos={val_cos:.4f} res={val_res:.3f}")

        # =========================================================
        # Analysis on cached val activations
        # =========================================================
        print(f"\n--- Analysis for {gap_name} ---")
        fwd_model.eval()

        val_src = val_acts["post_block0"]
        val_tgt = val_acts[predict_to]
        n_val_seqs = val_src.shape[0]

        all_res_norms = []
        all_cos_sims = []
        all_residuals = []

        with torch.no_grad():
            for i in range(0, n_val_seqs, batch_size):
                vs = val_src[i:i + batch_size].to(device)
                vt = val_tgt[i:i + batch_size].to(device)
                vp = fwd_model(vs)
                residual = vt - vp
                all_res_norms.append(residual.norm(dim=-1).cpu())
                all_cos_sims.append(
                    F.cosine_similarity(vp, vt, dim=-1).cpu()
                )
                all_residuals.append(residual.cpu())

        res_norms = torch.cat(all_res_norms, dim=0).numpy()
        cos_sims = torch.cat(all_cos_sims, dim=0).numpy()
        residuals = torch.cat(all_residuals, dim=0)

        # Basic quality
        mean_cos = float(cos_sims.mean())
        mean_mse = float(history["val_fwd_mse"][-1][1])
        mean_res = float(res_norms.mean())
        std_res = float(res_norms.std())
        print(f"  Cosine: {mean_cos:.4f}, MSE: {mean_mse:.5f}, "
              f"Residual norm: {mean_res:.3f}")

        # Residual-LM loss correlation
        lm_np = val_lm_losses.numpy()
        n_shared = min(res_norms.shape[0], lm_np.shape[0])
        flat_res = res_norms[:n_shared, :lm_np.shape[1]].flatten()
        flat_lm = lm_np[:n_shared].flatten()
        valid = np.isfinite(flat_res) & np.isfinite(flat_lm)
        residual_lm_corr = float(
            np.corrcoef(flat_res[valid], flat_lm[valid])[0, 1]
        )
        print(f"  r(res,LM): {residual_lm_corr:+.4f}")

        # Residual PCA
        res_flat = residuals[:n_shared].reshape(-1, n_embd).numpy()
        subsample = min(50000, res_flat.shape[0])
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(res_flat.shape[0], subsample, replace=False)
        res_sub = res_flat[idx_sub] - res_flat[idx_sub].mean(axis=0)
        _, S, _ = np.linalg.svd(res_sub, full_matrices=False)

        S_sq = S ** 2
        total_var = S_sq.sum()
        cumvar = np.cumsum(S_sq) / total_var
        S_norm = S / S.sum()
        eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))

        pca = {
            "effective_rank": eff_rank,
            "max_rank": n_embd,
            "effective_rank_ratio": eff_rank / n_embd,
            "rank_for_50pct_variance": int(np.searchsorted(cumvar, 0.50) + 1),
            "rank_for_90pct_variance": int(np.searchsorted(cumvar, 0.90) + 1),
            "rank_for_95pct_variance": int(np.searchsorted(cumvar, 0.95) + 1),
            "top1_pc_variance": float(S_sq[0] / total_var),
            "top5_pc_variance": float(S_sq[:5].sum() / total_var),
            "top10_pc_variance": float(S_sq[:10].sum() / total_var),
        }
        print(f"  Effective rank: {eff_rank:.1f}/{n_embd} "
              f"({eff_rank/n_embd:.1%})")
        print(f"  Top-1 PC: {pca['top1_pc_variance']:.3f}, "
              f"Top-5: {pca['top5_pc_variance']:.3f}")

        # Behavioral effects
        res_2d = res_norms[:n_shared]
        g_mean = float(res_2d.mean())
        g_std = float(res_2d.std())

        ss_vals = res_2d[sentence_start_mask[:n_shared]]
        bc_vals = res_2d[before_closer_mask[:n_shared]]
        d_ss = (float((ss_vals.mean() - g_mean) / (g_std + 1e-10))
                if len(ss_vals) > 0 else 0.0)
        d_bc = (float((bc_vals.mean() - g_mean) / (g_std + 1e-10))
                if len(bc_vals) > 0 else 0.0)
        print(f"  d(sentence_start): {d_ss:+.3f}")
        print(f"  d(before_closer): {d_bc:+.3f}")

        behavioral = {
            "cohen_d_sentence_start": d_ss,
            "cohen_d_before_closer": d_bc,
            "n_sentence_starts": int(sentence_start_mask[:n_shared].sum()),
            "n_before_closers": int(before_closer_mask[:n_shared].sum()),
        }

        # =========================================================
        # Causal substitution
        # =========================================================
        print(f"\n--- Causal substitution for {gap_name} ---")
        print(f"  Normal: all blocks")
        print(f"  Substituted: skip blocks {skip_blocks}, "
              f"use forward model prediction")
        print(f"  Ablated: skip blocks {skip_blocks}, pass input through")

        all_ce_n, all_ce_s, all_ce_a = [], [], []
        all_acc_n, all_acc_s, all_acc_a = [], [], []
        all_kl_sub, all_kl_abl = [], []

        with torch.no_grad():
            for bi in range(n_causal_batches):
                if bi % 10 == 0:
                    print(f"  causal batch {bi}/{n_causal_batches}")

                x, y = get_val_batch()
                B, Tseq = x.shape

                # Run embedding + block0 (shared)
                h = run_embedding(x)
                post_block0 = model.transformer.h[0](h)

                # Normal: run all remaining blocks
                h_normal = run_blocks(post_block0, 1, n_layer)
                logits_n = run_head(h_normal)

                # Substituted: forward model predicts target, then continue
                h_sub = fwd_model(post_block0)
                h_sub = run_blocks(h_sub, resume_block, n_layer)
                logits_s = run_head(h_sub)

                # Ablated: skip blocks, pass post_block0 through
                h_abl = run_blocks(post_block0, resume_block, n_layer)
                logits_a = run_head(h_abl)

                # CE loss
                ce_n = F.cross_entropy(
                    logits_n.view(-1, vocab_size), y.view(-1), reduction="none"
                ).view(B, Tseq)
                ce_s = F.cross_entropy(
                    logits_s.view(-1, vocab_size), y.view(-1), reduction="none"
                ).view(B, Tseq)
                ce_a = F.cross_entropy(
                    logits_a.view(-1, vocab_size), y.view(-1), reduction="none"
                ).view(B, Tseq)

                # Accuracy
                acc_n = (logits_n.argmax(-1) == y).float()
                acc_s = (logits_s.argmax(-1) == y).float()
                acc_a = (logits_a.argmax(-1) == y).float()

                # KL divergence (substituted/ablated vs normal)
                log_p_n = F.log_softmax(logits_n, dim=-1)
                log_p_s = F.log_softmax(logits_s, dim=-1)
                log_p_a = F.log_softmax(logits_a, dim=-1)
                p_n = log_p_n.exp()

                kl_sub = (p_n * (log_p_n - log_p_s)).sum(-1)
                kl_abl = (p_n * (log_p_n - log_p_a)).sum(-1)

                all_ce_n.append(ce_n.cpu())
                all_ce_s.append(ce_s.cpu())
                all_ce_a.append(ce_a.cpu())
                all_acc_n.append(acc_n.cpu())
                all_acc_s.append(acc_s.cpu())
                all_acc_a.append(acc_a.cpu())
                all_kl_sub.append(kl_sub.cpu())
                all_kl_abl.append(kl_abl.cpu())

        ce_n = torch.cat(all_ce_n, dim=0).mean().item()
        ce_s = torch.cat(all_ce_s, dim=0).mean().item()
        ce_a = torch.cat(all_ce_a, dim=0).mean().item()
        acc_n = torch.cat(all_acc_n, dim=0).mean().item()
        acc_s = torch.cat(all_acc_s, dim=0).mean().item()
        acc_a = torch.cat(all_acc_a, dim=0).mean().item()
        kl_sub = torch.cat(all_kl_sub, dim=0).mean().item()
        kl_abl = torch.cat(all_kl_abl, dim=0).mean().item()
        recovery = 1.0 - kl_sub / kl_abl if kl_abl > 0 else float("nan")

        print(f"  Normal:      acc={acc_n:.3f}  CE={ce_n:.3f}")
        print(f"  Substituted: acc={acc_s:.3f}  CE={ce_s:.3f}  "
              f"KL={kl_sub:.3f}")
        print(f"  Ablated:     acc={acc_a:.3f}  CE={ce_a:.3f}  "
              f"KL={kl_abl:.3f}")
        print(f"  Recovery: {recovery:.1%}")

        causal = {
            "normal_accuracy": acc_n,
            "substituted_accuracy": acc_s,
            "ablated_accuracy": acc_a,
            "normal_ce": ce_n,
            "substituted_ce": ce_s,
            "ablated_ce": ce_a,
            "kl_substituted": kl_sub,
            "kl_ablated": kl_abl,
            "recovery": recovery,
        }

        # --- Save this gap's results ---
        gap_dir = os.path.join(save_root, gap_name)
        os.makedirs(gap_dir, exist_ok=True)
        torch.save(fwd_model.state_dict(), os.path.join(gap_dir, "fwd_model.pt"))

        result = {
            "gap_name": gap_name,
            "predict_from": "post_block0",
            "predict_to": predict_to,
            "skip_blocks": skip_blocks,
            "n_layers_skipped": len(skip_blocks),
            "fwd_model": {
                "n_layer": fwd_n_layer, "n_head": fwd_n_head,
                "d_head": fwd_d_head, "mlp_mult": fwd_mlp_mult,
                "n_params": fwd_n_params, "capacity_ratio": capacity_ratio,
            },
            "basic_quality": {
                "mean_cosine_sim": mean_cos,
                "final_val_mse": mean_mse,
                "mean_residual_norm": mean_res,
                "std_residual_norm": std_res,
            },
            "residual_lm_loss_correlation": residual_lm_corr,
            "residual_pca": pca,
            "behavioral_effects": behavioral,
            "causal_substitution": causal,
            "history": history,
        }

        with open(os.path.join(gap_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)

        all_results.append(result)

        del fwd_model, opt_fwd
        torch.cuda.empty_cache()

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'=' * 60}")
    print("LAYER GAP SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"Main model: {n_layer}L/{n_head}H/{n_embd}D ({main_n_params:,} params)")
    print(f"Forward model: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d "
          f"({all_results[0]['fwd_model']['n_params']:,} params, "
          f"{all_results[0]['fwd_model']['capacity_ratio']:.1%})")
    print()

    header = (f"{'Gap':>8s} {'Cosine':>7s} {'MSE':>8s} "
              f"{'EffRank':>8s} {'d_SS':>6s} {'d_BC':>6s} "
              f"{'KL_sub':>7s} {'KL_abl':>7s} {'Recov':>6s}")
    print(header)
    for r in all_results:
        print(f"{r['gap_name']:>8s} "
              f"{r['basic_quality']['mean_cosine_sim']:>7.4f} "
              f"{r['basic_quality']['final_val_mse']:>8.5f} "
              f"{r['residual_pca']['effective_rank']:>8.1f} "
              f"{r['behavioral_effects']['cohen_d_sentence_start']:>+6.3f} "
              f"{r['behavioral_effects']['cohen_d_before_closer']:>+6.3f} "
              f"{r['causal_substitution']['kl_substituted']:>7.3f} "
              f"{r['causal_substitution']['kl_ablated']:>7.3f} "
              f"{r['causal_substitution']['recovery']:>5.1%}")

    summary = {
        "experiment": "layer_gap_sweep",
        "main_model": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "n_params": main_n_params,
        },
        "training": {
            "n_tokens": actual_n_tokens, "n_steps": n_steps,
            "batch_size": batch_size, "block_size": block_size,
            "fwd_lr": fwd_lr,
        },
        "gaps": all_results,
    }

    with open(os.path.join(save_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nAll results saved to {save_root}")
    return summary


@app.local_entrypoint()
def main(
    n_tokens: int = 2_000_000,
    n_steps: int = 10_000,
    fwd_lr: float = 1e-3,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_layer_gap_sweep.remote(
        n_tokens=n_tokens, n_steps=n_steps, fwd_lr=fwd_lr,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print("\nLayer gap sweep complete:")
    print(f"Main model: {result['main_model']['n_params']:,} params")
    print()
    for r in result["gaps"]:
        q = r["basic_quality"]
        c = r["causal_substitution"]
        print(f"  {r['gap_name']:>8s}: cos={q['mean_cosine_sim']:.4f}  "
              f"KL_sub={c['kl_substituted']:.3f}  "
              f"recovery={c['recovery']:.1%}")
