"""Forward model capacity scaling sweep over a frozen main model.

Tests the bias-to-variance transition hypothesis: at low forward model
capacity, the residual captures computational novelty (what the architecture
can't represent). As capacity increases toward saturation, the residual
should transition to epistemic novelty (what the main model is uncertain
about), and the residual-LM loss correlation should become positive.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR

SCALING_CONFIGS = [
    {"name": "1pct",  "n_layer": 1, "n_head": 1, "d_head": 64,  "mlp_mult": 2},   # ~330K (1.1%)
    {"name": "3pct",  "n_layer": 2, "n_head": 2, "d_head": 64,  "mlp_mult": 2},   # ~790K (2.7%)
    {"name": "10pct", "n_layer": 3, "n_head": 4, "d_head": 128, "mlp_mult": 4},   # ~3.2M (10.9%)
    {"name": "20pct", "n_layer": 4, "n_head": 4, "d_head": 256, "mlp_mult": 2},   # ~5.3M (18.2%)
    {"name": "30pct", "n_layer": 4, "n_head": 4, "d_head": 256, "mlp_mult": 4},   # ~6.3M (21.8%)
]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_scaling_sweep(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    n_analysis_batches: int = 20,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A scaling sweep on {device}")
    print(f"  Main model: {n_layer}L {n_head}H {n_embd}D")
    print(f"  Gap: {predict_from} -> {predict_to}")
    print(f"  {len(SCALING_CONFIGS)} capacity points")

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
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(data):,} tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # =========================================================
    # Phase 1: Train the main model
    # =========================================================
    save_root = f"{DATA_DIR}/a2a_forward/scaling_sweep"
    os.makedirs(save_root, exist_ok=True)

    main_ckpt_path = os.path.join(save_root, "main_model.pt")

    print("\n=== Phase 1: Training main model ===")
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    main_n_params = sum(p.numel() for p in model.parameters())
    print(f"  Main model params: {main_n_params:,}")

    opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val_loss = float("inf")

    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data)
        _, lm_loss = model(x, y)
        opt_main.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                val_acc = 0.0
                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)
                    _, vl = model(vx, vy)
                    val_acc += float(vl)
                val_lm = val_acc / n_eval_batches
                if val_lm < best_val_loss:
                    best_val_loss = val_lm
                print(f"  step {step:6d}: lm={float(lm_loss):.4f} val_lm={val_lm:.4f}")

    torch.save(model.state_dict(), main_ckpt_path)
    print(f"  Main model saved (best val loss: {best_val_loss:.4f})")

    # =========================================================
    # Phase 2: Pre-compute and cache frozen activations
    # =========================================================
    print("\n=== Phase 2: Caching frozen activations ===")
    model.eval()

    def cache_activations(split_data, label):
        n_sequences = len(split_data) // block_size
        all_src = []
        all_tgt = []
        all_x = []
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
                all_src.append(intermediates[predict_from].cpu())
                all_tgt.append(intermediates[predict_to].cpu())
                all_x.append(x.cpu())
        src = torch.cat(all_src, dim=0)
        tgt = torch.cat(all_tgt, dim=0)
        xs = torch.cat(all_x, dim=0)
        print(f"  Cached {label}: {src.shape[0]} sequences, "
              f"src={src.shape}, tgt={tgt.shape}")
        return src, tgt, xs

    train_src, train_tgt, train_x = cache_activations(train_data, "train")
    val_src, val_tgt, val_x = cache_activations(val_data, "val")

    # Pre-compute per-position LM losses on val data for analysis
    print("  Computing per-position LM losses on val data...")
    all_lm_losses = []
    with torch.no_grad():
        for i in range(0, val_x.shape[0], batch_size):
            batch_x = val_x[i:i + batch_size].to(device)
            per_pos = model.get_ngram_losses(batch_x)
            all_lm_losses.append(per_pos.cpu())
    val_lm_losses = torch.cat(all_lm_losses, dim=0)  # (N_val, T-1)
    print(f"  LM losses shape: {val_lm_losses.shape}")

    # Pre-decode tokens for behavioral analysis
    enc = tiktoken.get_encoding("gpt2")
    print("  Decoding tokens for behavioral categories...")
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

    # =========================================================
    # Phase 3: Train forward models at each capacity point
    # =========================================================
    all_results = []

    for cfg_idx, cfg in enumerate(SCALING_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Config {cfg_idx}: {cfg['name']} "
              f"({cfg['n_layer']}L {cfg['n_head']}H {cfg['d_head']}d)")
        print(f"{'='*60}")

        fwd_model = TransformerForwardModel(
            d_model=n_embd,
            d_head=cfg["d_head"],
            n_head=cfg["n_head"],
            n_layer=cfg["n_layer"],
            mlp_mult=cfg["mlp_mult"],
            block_size=block_size,
        ).to(device)

        fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
        capacity_ratio = fwd_n_params / main_n_params
        print(f"  Forward model params: {fwd_n_params:,} "
              f"({capacity_ratio:.1%} of main model)")

        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01
        )

        n_train = train_src.shape[0]
        history = {"fwd_mse": [], "val_fwd_mse": [], "val_cosine_sim": [],
                    "val_residual_norm": []}

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
                    n_val_seqs = val_src.shape[0]
                    n_val_batches = min(n_eval_batches,
                                        max(1, n_val_seqs // batch_size))
                    for _ in range(n_val_batches):
                        vidx = torch.randint(n_val_seqs, (batch_size,))
                        vs = val_src[vidx].to(device)
                        vt = val_tgt[vidx].to(device)
                        vp = fwd_model(vs)
                        val_mse_acc += float(F.mse_loss(vp, vt))
                        val_cos_acc += float(
                            F.cosine_similarity(vp, vt, dim=-1).mean()
                        )
                        val_res_acc += float((vt - vp).norm(dim=-1).mean())

                    val_mse = val_mse_acc / n_val_batches
                    val_cos = val_cos_acc / n_val_batches
                    val_res = val_res_acc / n_val_batches

                    history["fwd_mse"].append((step, float(fwd_loss)))
                    history["val_fwd_mse"].append((step, val_mse))
                    history["val_cosine_sim"].append((step, val_cos))
                    history["val_residual_norm"].append((step, val_res))

                    print(f"  step {step:6d}: "
                          f"fwd_mse={float(fwd_loss):.4f} val_mse={val_mse:.4f} "
                          f"cos={val_cos:.4f} res_norm={val_res:.3f}")

        # =========================================================
        # Phase 4: Analysis
        # =========================================================
        print(f"\n--- Analysis for {cfg['name']} ---")
        fwd_model.eval()

        all_res_norms = []
        all_cos_sims = []
        all_residuals = []

        n_val_seqs = val_src.shape[0]
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

        res_norms = torch.cat(all_res_norms, dim=0).numpy()  # (N_val, T)
        cos_sims = torch.cat(all_cos_sims, dim=0).numpy()
        residuals = torch.cat(all_residuals, dim=0)  # (N_val, T, d_model)

        # 1. Basic quality
        mean_res = float(res_norms.mean())
        std_res = float(res_norms.std())
        mean_cos = float(cos_sims.mean())
        mean_mse = float(history["val_fwd_mse"][-1][1])
        print(f"  Mean residual norm: {mean_res:.4f}")
        print(f"  Mean cosine sim: {mean_cos:.4f}")

        # 2. Residual-LM loss correlation
        lm_np = val_lm_losses.numpy()  # (N_val, T-1)
        n_shared = min(res_norms.shape[0], lm_np.shape[0])
        flat_res = res_norms[:n_shared, :lm_np.shape[1]].flatten()
        flat_lm = lm_np[:n_shared].flatten()
        valid = np.isfinite(flat_res) & np.isfinite(flat_lm)
        residual_lm_corr = float(
            np.corrcoef(flat_res[valid], flat_lm[valid])[0, 1]
        )
        print(f"  Residual-LM loss correlation: {residual_lm_corr:+.4f}")

        # 3. Residual-LM correlation by quartile
        lm_valid = flat_lm[valid]
        res_valid = flat_res[valid]
        quartile_boundaries = np.percentile(lm_valid, [25, 50, 75])
        quartile_labels = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
        quartile_bins = np.digitize(lm_valid, quartile_boundaries)
        residual_by_quartile = {}
        for q in range(4):
            mask = quartile_bins == q
            if mask.sum() > 0:
                residual_by_quartile[quartile_labels[q]] = float(
                    res_valid[mask].mean()
                )
        print(f"  Residual by LM-loss quartile: {residual_by_quartile}")

        # 4. Residual effective rank (PCA)
        residual_flat = residuals[:n_shared].reshape(-1, n_embd).numpy()
        subsample = min(50000, residual_flat.shape[0])
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(residual_flat.shape[0], subsample, replace=False)
        residual_sub = residual_flat[idx_sub]
        residual_sub = residual_sub - residual_sub.mean(axis=0)
        _, S, _ = np.linalg.svd(residual_sub, full_matrices=False)
        S_norm = S / S.sum()
        effective_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))
        var_explained_90 = int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.9) + 1)
        print(f"  Effective rank: {effective_rank:.1f}/{n_embd}")
        print(f"  Rank for 90% variance: {var_explained_90}")

        # 5. Behavioral effect sizes
        res_2d = res_norms[:n_shared]  # (N_val, T)
        ss_mask = sentence_start_mask[:n_shared]
        bc_mask = before_closer_mask[:n_shared]

        all_res_flat = res_2d.flatten()
        global_mean = float(all_res_flat.mean())
        global_std = float(all_res_flat.std())

        ss_vals = res_2d[ss_mask]
        bc_vals = res_2d[bc_mask]
        cohen_d_sentence_start = float(
            (ss_vals.mean() - global_mean) / (global_std + 1e-10)
        ) if len(ss_vals) > 0 else 0.0
        cohen_d_before_closer = float(
            (bc_vals.mean() - global_mean) / (global_std + 1e-10)
        ) if len(bc_vals) > 0 else 0.0
        print(f"  Cohen's d sentence_start: {cohen_d_sentence_start:+.3f}")
        print(f"  Cohen's d before_closer: {cohen_d_before_closer:+.3f}")

        # --- Save this config's results ---
        cfg_dir = os.path.join(save_root, f"config_{cfg_idx}")
        os.makedirs(cfg_dir, exist_ok=True)
        torch.save(fwd_model.state_dict(), os.path.join(cfg_dir, "fwd_model.pt"))

        result = {
            "config_index": cfg_idx,
            "config_name": cfg["name"],
            "fwd_n_layer": cfg["n_layer"],
            "fwd_n_head": cfg["n_head"],
            "fwd_d_head": cfg["d_head"],
            "fwd_mlp_mult": cfg["mlp_mult"],
            "fwd_n_params": fwd_n_params,
            "main_n_params": main_n_params,
            "capacity_ratio": capacity_ratio,
            "n_tokens": n_tokens,
            "n_steps": n_steps,
            "predict_from": predict_from,
            "predict_to": predict_to,
            "basic_quality": {
                "mean_residual_norm": mean_res,
                "std_residual_norm": std_res,
                "mean_cosine_sim": mean_cos,
                "final_val_mse": mean_mse,
            },
            "residual_lm_loss_correlation": residual_lm_corr,
            "residual_by_lm_quartile": residual_by_quartile,
            "residual_pca": {
                "effective_rank": effective_rank,
                "max_rank": n_embd,
                "rank_90_variance": var_explained_90,
            },
            "behavioral_effects": {
                "cohen_d_sentence_start": cohen_d_sentence_start,
                "cohen_d_before_closer": cohen_d_before_closer,
                "n_sentence_starts": int(ss_mask.sum()),
                "n_before_closers": int(bc_mask.sum()),
            },
            "history": history,
        }

        with open(os.path.join(cfg_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, cls=_NumpyEncoder)

        all_results.append(result)

        del fwd_model, opt_fwd
        torch.cuda.empty_cache()

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'='*60}")
    print("SCALING SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':>8s} {'Params':>10s} {'Ratio':>7s} {'CosS':>6s} "
          f"{'ResNorm':>8s} {'r(res,LM)':>10s} {'EffRank':>8s} "
          f"{'d_SS':>6s} {'d_BC':>6s}")
    for r in all_results:
        print(f"{r['config_name']:>8s} "
              f"{r['fwd_n_params']:>10,} "
              f"{r['capacity_ratio']:>6.1%} "
              f"{r['basic_quality']['mean_cosine_sim']:>6.3f} "
              f"{r['basic_quality']['mean_residual_norm']:>8.3f} "
              f"{r['residual_lm_loss_correlation']:>+10.4f} "
              f"{r['residual_pca']['effective_rank']:>8.1f} "
              f"{r['behavioral_effects']['cohen_d_sentence_start']:>+6.3f} "
              f"{r['behavioral_effects']['cohen_d_before_closer']:>+6.3f}")

    summary = {
        "main_model": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "n_params": main_n_params, "best_val_loss": best_val_loss,
        },
        "predict_from": predict_from,
        "predict_to": predict_to,
        "n_tokens": n_tokens,
        "n_steps": n_steps,
        "configs": all_results,
    }
    with open(os.path.join(save_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    volume.commit()
    print(f"\nAll results saved to {save_root}")
    return summary


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_steps: int = 10_000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
):
    result = a2a_scaling_sweep.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        predict_from=predict_from, predict_to=predict_to,
    )
    print("A2A scaling sweep complete:")
    mm = result["main_model"]
    print(f"  Main model: {mm['n_params']:,} params, "
          f"best_val_loss={mm['best_val_loss']:.4f}")
    print(f"  {'Config':>8s} {'Ratio':>7s} {'r(res,LM)':>10s} "
          f"{'EffRank':>8s} {'d_SS':>6s} {'d_BC':>6s}")
    for c in result["configs"]:
        print(f"  {c['config_name']:>8s} "
              f"{c['capacity_ratio']:>6.1%} "
              f"{c['residual_lm_loss_correlation']:>+10.4f} "
              f"{c['residual_pca']['effective_rank']:>8.1f} "
              f"{c['behavioral_effects']['cohen_d_sentence_start']:>+6.3f} "
              f"{c['behavioral_effects']['cohen_d_before_closer']:>+6.3f}")
