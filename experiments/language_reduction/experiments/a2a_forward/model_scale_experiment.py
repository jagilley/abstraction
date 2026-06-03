"""Model scale experiment: how does A2A residual structure change with main model size?

Tests whether larger language models develop more structured computation
(lower-rank residuals, stronger behavioral effects) as they better understand
the language DGP, analogous to how grokking produces low-rank Fourier residuals.

Hypothesis: the 29M GPT's full-rank, diffuse residual reflects the model being
too small to develop clean computational structure. Bigger models that understand
language better should produce residuals with more interpretable structure —
concentrated in fewer dimensions and/or with stronger behavioral conditioning.

Compare with the grokking case (cnb_self_regulation), where the Fourier solution
produced low-rank residuals that cleanly tracked mechanism-specific structure.
"""

import json

from language_reduction.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=43200,
    memory=32768,
)
def a2a_model_scale(
    n_tokens: int = 100_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 30_000,
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    n_analysis_batches: int = 20,
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    ckpt_interval: int = 2000,
):
    """Train a GPT + capacity-matched forward model, with comprehensive
    residual structure analysis designed for cross-scale comparison.

    Tracks effective rank during training (does rank compress like grokking?)
    and runs full analysis at the end: residual PCA, behavioral effects,
    LM-loss correlation, SV spectrum shape.
    """
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.experiments.a2a_forward.forward_model import (
        TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A model scale experiment on {device}")
    print(f"  Main model: {n_layer}L {n_head}H {n_embd}D")
    print(f"  Forward model: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}d mlp×{fwd_mlp_mult}")
    print(f"  Gap: {predict_from} → {predict_to}")
    print(f"  P={n_tokens:,}, steps={n_steps:,}, batch={batch_size}")

    # --- Load data (τ=0.0, unchanged corpus) ---
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
    print(f"Loaded {actual_n_tokens:,} tokens (vocab_size={vocab_size})")
    if actual_n_tokens < n_tokens:
        print(f"  WARNING: requested {n_tokens:,} but only {actual_n_tokens:,} available.")
        print(f"  Run tokenization first: modal run language_reduction/modal_app.py "
              f"--stage tokenize --n-tokens {n_tokens}")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    tokens_per_step = batch_size * block_size
    total_token_steps = n_steps * tokens_per_step
    epochs = total_token_steps / actual_n_tokens
    print(f"  {tokens_per_step:,} tokens/step, "
          f"{total_token_steps:,} total → {epochs:.1f} epochs")

    # --- Create models ---
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    main_n_params = sum(p.numel() for p in model.parameters())

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    capacity_ratio = fwd_n_params / main_n_params

    print(f"  Main model: {main_n_params:,} params")
    print(f"  Forward model: {fwd_n_params:,} params ({capacity_ratio:.1%})")
    print(f"  Tokens per param: {actual_n_tokens / main_n_params:.1f}")

    opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- Checkpoint resume ---
    model_tag = f"gpt_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/model_scale/{model_tag}/P_{actual_n_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")

    history = {
        "lm_loss": [], "fwd_mse": [],
        "val_lm_loss": [], "val_fwd_mse": [], "val_cosine_sim": [],
        "val_residual_norm": [], "val_effective_rank": [],
    }
    best_val_loss = float("inf")
    start_step = 0

    if os.path.exists(ckpt_path):
        print(f"\n  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        fwd_model.load_state_dict(ckpt["fwd_model"])
        opt_main.load_state_dict(ckpt["opt_main"])
        opt_fwd.load_state_dict(ckpt["opt_fwd"])
        start_step = ckpt["step"] + 1
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"  Resuming from step {start_step} "
              f"(best_val_loss={best_val_loss:.4f})")

    # =========================================================
    # Training loop with periodic residual structure tracking
    # =========================================================
    print(f"\n=== Training (steps {start_step}–{n_steps - 1}) ===")
    for step in range(start_step, n_steps):
        model.train()
        fwd_model.train()

        x, y = get_batch(train_data)
        _, lm_loss, intermediates = model(x, y, return_intermediates=True)

        source = intermediates[predict_from].detach()
        target = intermediates[predict_to].detach()
        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)

        opt_main.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()

        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            fwd_model.eval()
            with torch.no_grad():
                val_lm_acc = 0.0
                val_fwd_acc = 0.0
                val_cos_acc = 0.0
                val_res_acc = 0.0
                eval_residuals = []

                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)
                    _, vl, vi = model(vx, vy, return_intermediates=True)
                    val_lm_acc += float(vl)

                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fwd_model(src)
                    val_fwd_acc += float(F.mse_loss(pred, tgt))
                    val_cos_acc += float(
                        F.cosine_similarity(pred, tgt, dim=-1).mean()
                    )
                    residual = tgt - pred
                    val_res_acc += float(residual.norm(dim=-1).mean())
                    eval_residuals.append(residual.cpu())

                n = n_eval_batches
                val_lm = val_lm_acc / n
                val_fwd = val_fwd_acc / n
                val_cos = val_cos_acc / n
                val_res = val_res_acc / n

                if val_lm < best_val_loss:
                    best_val_loss = val_lm

                eff_rank = _compute_effective_rank(eval_residuals, n_embd)

                history["lm_loss"].append((step, float(lm_loss)))
                history["fwd_mse"].append((step, float(fwd_loss)))
                history["val_lm_loss"].append((step, val_lm))
                history["val_fwd_mse"].append((step, val_fwd))
                history["val_cosine_sim"].append((step, val_cos))
                history["val_residual_norm"].append((step, val_res))
                history["val_effective_rank"].append((step, eff_rank))

                print(f"  step {step:6d}: "
                      f"lm={float(lm_loss):.4f} val_lm={val_lm:.4f} "
                      f"fwd={float(fwd_loss):.4f} cos={val_cos:.4f} "
                      f"res={val_res:.3f} rank={eff_rank:.1f}/{n_embd}")

            # --- Periodic checkpoint ---
            if step > 0 and step % ckpt_interval == 0:
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "fwd_model": fwd_model.state_dict(),
                    "opt_main": opt_main.state_dict(),
                    "opt_fwd": opt_fwd.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                volume.commit()
                print(f"  [checkpoint saved at step {step}]")

    # =========================================================
    # Comprehensive final analysis
    # =========================================================
    print("\n=== Final analysis ===")
    analysis = _run_scale_analysis(
        model, fwd_model, val_data, get_batch,
        predict_from, predict_to,
        n_analysis_batches, n_embd, device,
    )

    for k, v in analysis.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, float):
                    print(f"    {kk}: {vv:.4f}")
                elif isinstance(vv, int):
                    print(f"    {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # --- Save final results and clean up checkpoint ---
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "experiment": "model_scale",
        "main_model": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "n_params": main_n_params,
            "best_val_loss": best_val_loss,
            "final_val_loss": history["val_lm_loss"][-1][1],
        },
        "forward_model": {
            "n_layer": fwd_n_layer, "n_head": fwd_n_head,
            "d_head": fwd_d_head, "mlp_mult": fwd_mlp_mult,
            "n_params": fwd_n_params, "capacity_ratio": capacity_ratio,
        },
        "training": {
            "n_tokens": actual_n_tokens, "block_size": block_size,
            "n_steps": n_steps, "batch_size": batch_size,
            "lr": lr, "fwd_lr": fwd_lr,
            "predict_from": predict_from, "predict_to": predict_to,
            "tokens_per_param": actual_n_tokens / main_n_params,
            "epochs": epochs,
        },
        "analysis": analysis,
        "history": history,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return result


def _compute_effective_rank(residual_list, d_model, max_samples=20000):
    """Compute effective rank (Shannon entropy of normalized SVs) from residuals."""
    import torch
    import numpy as np

    flat = torch.cat(residual_list, dim=0).reshape(-1, d_model).numpy()
    n = min(max_samples, flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(flat.shape[0], n, replace=False)
    sub = flat[idx]
    sub = sub - sub.mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    S_norm = S / S.sum()
    return float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))


def _run_scale_analysis(model, fwd_model, val_data, get_batch,
                        predict_from, predict_to,
                        n_batches, n_embd, device):
    """Comprehensive residual analysis for cross-scale comparison.

    Returns dict with: basic_quality, residual_lm_loss_correlation,
    residual_by_lm_quartile, residual_pca, behavioral_effects.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken

    all_res_norms = []
    all_cos_sims = []
    all_lm_losses = []
    all_residuals = []
    all_tokens = []

    model.eval()
    fwd_model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)

            src = vi[predict_from]
            tgt = vi[predict_to]
            pred = fwd_model(src)

            residual = tgt - pred
            all_res_norms.append(residual.norm(dim=-1).cpu())
            all_cos_sims.append(
                F.cosine_similarity(pred, tgt, dim=-1).cpu()
            )
            all_lm_losses.append(model.get_ngram_losses(vx).cpu())
            all_residuals.append(residual.cpu())
            all_tokens.append(vx.cpu())

    res_norms = torch.cat(all_res_norms, dim=0).numpy()
    cos_sims = torch.cat(all_cos_sims, dim=0).numpy()
    lm_losses = torch.cat(all_lm_losses, dim=0).numpy()
    residuals = torch.cat(all_residuals, dim=0)
    tokens = torch.cat(all_tokens, dim=0).numpy()

    N, T = res_norms.shape
    n_shared = min(N, lm_losses.shape[0])

    # --- Basic quality ---
    basic = {
        "mean_residual_norm": float(res_norms.mean()),
        "std_residual_norm": float(res_norms.std()),
        "mean_cosine_sim": float(cos_sims.mean()),
    }

    # --- Residual-LM loss correlation ---
    flat_res = res_norms[:n_shared, :lm_losses.shape[1]].flatten()
    flat_lm = lm_losses[:n_shared].flatten()
    valid = np.isfinite(flat_res) & np.isfinite(flat_lm)
    residual_lm_corr = float(
        np.corrcoef(flat_res[valid], flat_lm[valid])[0, 1]
    )

    # --- Residual by LM-loss quartile ---
    lm_valid = flat_lm[valid]
    res_valid = flat_res[valid]
    q_bounds = np.percentile(lm_valid, [25, 50, 75])
    q_labels = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
    q_bins = np.digitize(lm_valid, q_bounds)
    by_quartile = {}
    for q in range(4):
        mask = q_bins == q
        if mask.sum() > 0:
            by_quartile[q_labels[q]] = float(res_valid[mask].mean())

    # --- Residual PCA ---
    res_flat = residuals[:n_shared].reshape(-1, n_embd).numpy()
    subsample = min(50000, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], subsample, replace=False)
    res_sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(res_sub, full_matrices=False)

    S_sq = S ** 2
    total_var = S_sq.sum()
    cumvar = np.cumsum(S_sq) / total_var
    S_norm = S / S.sum()
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))

    pca = {
        "effective_rank": eff_rank,
        "max_rank": n_embd,
        "rank_for_50pct_variance": int(np.searchsorted(cumvar, 0.50) + 1),
        "rank_for_75pct_variance": int(np.searchsorted(cumvar, 0.75) + 1),
        "rank_for_90pct_variance": int(np.searchsorted(cumvar, 0.90) + 1),
        "rank_for_95pct_variance": int(np.searchsorted(cumvar, 0.95) + 1),
        "top1_pc_variance": float(S_sq[0] / total_var),
        "top5_pc_variance": float(S_sq[:5].sum() / total_var),
        "top10_pc_variance": float(S_sq[:10].sum() / total_var),
        "sv_spectrum_normalized": (S / S[0]).tolist(),
    }

    # --- Behavioral effects ---
    enc = tiktoken.get_encoding("gpt2")
    tok_np = tokens[:n_shared]
    ss_mask = np.zeros((n_shared, T), dtype=bool)
    bc_mask = np.zeros((n_shared, T), dtype=bool)
    SENT_END = set(".!?")
    CLOSER = set(")]}")

    for b in range(n_shared):
        decoded = [enc.decode([int(tok_np[b, t])]) for t in range(T)]
        for t in range(T):
            if t > 0:
                prev = decoded[t - 1].strip()
                if prev and prev[-1] in SENT_END:
                    ss_mask[b, t] = True
            if t < T - 1:
                nxt = decoded[t + 1].strip()
                if nxt and nxt[0] in CLOSER:
                    bc_mask[b, t] = True

    r2d = res_norms[:n_shared]
    g_mean = float(r2d.mean())
    g_std = float(r2d.std())

    ss_vals = r2d[ss_mask]
    bc_vals = r2d[bc_mask]
    d_ss = float((ss_vals.mean() - g_mean) / (g_std + 1e-10)) if len(ss_vals) > 0 else 0.0
    d_bc = float((bc_vals.mean() - g_mean) / (g_std + 1e-10)) if len(bc_vals) > 0 else 0.0

    behavioral = {
        "cohen_d_sentence_start": d_ss,
        "cohen_d_before_closer": d_bc,
        "n_sentence_starts": int(ss_mask.sum()),
        "n_before_closers": int(bc_mask.sum()),
    }

    return {
        "basic_quality": basic,
        "residual_lm_loss_correlation": residual_lm_corr,
        "residual_by_lm_quartile": by_quartile,
        "residual_pca": pca,
        "behavioral_effects": behavioral,
    }
