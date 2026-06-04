"""Analysis of A2A forward model structure.

Loads trained GPT + forward model, then characterizes:
1. Residual structure (PCA, effective rank)
2. Representational alignment (CKA between forward model and block1 internals)
3. Attention pattern comparison (forward model's 1 head vs block1's 4 heads)
4. Residual conditioning (by token frequency, position, LM loss)
5. Weight-space comparison (QKV projection cosine similarity)
"""

import json
import math

from a2a_forward.shared import app, volume, DATA_DIR


def linear_cka(X, Y):
    """Compute linear CKA between X (n, p1) and Y (n, p2)."""
    import numpy as np
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Y.T @ X, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(cross / (xx * yy)) if (xx * yy) > 0 else 0.0


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_analyze(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    n_eval_batches: int = 50,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    import os
    import glob
    import torch
    import numpy as np
    import torch.nn.functional as F
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A structure analysis on {device}")

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
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    token_counts = torch.bincount(train_data, minlength=vocab_size).float()
    token_freq = token_counts / token_counts.sum()

    # --- Load models ---
    model_dir = f"{DATA_DIR}/a2a_forward/transformer/P_{n_tokens}"

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True
    ))
    model.eval()

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device, weights_only=True
    ))
    fwd_model.eval()

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # =============================================
    # Collect activations
    # =============================================
    print("Collecting activations...")

    all_residuals = []
    all_fwd_post_attn = []
    all_block1_post_attn = []
    all_fwd_output = []
    all_block1_output = []
    all_source = []
    all_fwd_attn_weights = []
    all_block1_attn_weights = []
    all_tokens_batch = []
    all_residual_norms = []
    all_cos_sims = []
    all_lm_losses = []

    n_pca_batches = min(20, n_eval_batches)

    with torch.no_grad():
        for batch_i in range(n_eval_batches):
            if batch_i % 10 == 0:
                print(f"  batch {batch_i}/{n_eval_batches}")

            x, y = get_batch(val_data)
            _, _, intermediates = model(x, y, return_intermediates=True)

            source = intermediates["post_block0"]
            target = intermediates["post_block1"]
            B, T, C = source.size()

            # --- Forward model: manual forward for intermediates ---
            fwd_h = fwd_model.ln1(source)
            fwd_q = fwd_model.q_proj(fwd_h).view(
                B, T, fwd_model.n_head, fwd_model.d_head
            ).transpose(1, 2)
            fwd_k = fwd_model.k_proj(fwd_h).view(
                B, T, fwd_model.n_head, fwd_model.d_head
            ).transpose(1, 2)
            fwd_v = fwd_model.v_proj(fwd_h).view(
                B, T, fwd_model.n_head, fwd_model.d_head
            ).transpose(1, 2)
            fwd_att = (fwd_q @ fwd_k.transpose(-2, -1)) * (
                1.0 / math.sqrt(fwd_model.d_head)
            )
            fwd_att = fwd_att.masked_fill(
                fwd_model.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            fwd_att_w = F.softmax(fwd_att, dim=-1)
            fwd_y = fwd_att_w @ fwd_v
            fwd_y = fwd_y.transpose(1, 2).contiguous().view(
                B, T, fwd_model.d_head * fwd_model.n_head
            )
            fwd_post_attn = source + fwd_model.out_proj(fwd_y)
            fwd_output = fwd_post_attn + fwd_model.mlp(fwd_model.ln2(fwd_post_attn))

            # --- Block1: manual forward for intermediates ---
            block1 = model.transformer.h[1]
            b1_h = block1.ln_1(source)
            b1_qkv = block1.attn.c_attn(b1_h)
            b1_q, b1_k, b1_v = b1_qkv.split(block1.attn.n_embd, dim=2)
            b1_q = b1_q.view(
                B, T, block1.attn.n_head, block1.attn.head_dim
            ).transpose(1, 2)
            b1_k = b1_k.view(
                B, T, block1.attn.n_head, block1.attn.head_dim
            ).transpose(1, 2)
            b1_v = b1_v.view(
                B, T, block1.attn.n_head, block1.attn.head_dim
            ).transpose(1, 2)
            b1_att = (b1_q @ b1_k.transpose(-2, -1)) * (
                1.0 / math.sqrt(block1.attn.head_dim)
            )
            b1_att = b1_att.masked_fill(
                block1.attn.bias[:, :, :T, :T] == 0, float("-inf")
            )
            b1_att_w = F.softmax(b1_att, dim=-1)
            b1_y = b1_att_w @ b1_v
            b1_y = b1_y.transpose(1, 2).contiguous().view(B, T, C)
            b1_attn_out = block1.attn.c_proj(b1_y)
            block1_post_attn = source + b1_attn_out
            block1_output = block1_post_attn + block1.mlp(block1.ln_2(block1_post_attn))

            # --- Metrics ---
            residual = target - fwd_output
            res_norm = residual.norm(dim=-1)
            cos = F.cosine_similarity(fwd_output, target, dim=-1)
            per_pos_loss = model.get_ngram_losses(x)

            # Store full vectors for PCA/CKA (first N batches only)
            if batch_i < n_pca_batches:
                all_residuals.append(residual.cpu())
                all_fwd_post_attn.append(fwd_post_attn.cpu())
                all_block1_post_attn.append(block1_post_attn.cpu())
                all_fwd_output.append(fwd_output.cpu())
                all_block1_output.append(block1_output.cpu())
                all_source.append(source.cpu())

            # Attention patterns (averaged over batch for memory)
            all_fwd_attn_weights.append(fwd_att_w.mean(dim=0).cpu())
            all_block1_attn_weights.append(b1_att_w.mean(dim=0).cpu())

            # Scalar metrics (all batches)
            all_tokens_batch.append(x.cpu())
            all_residual_norms.append(res_norm.cpu())
            all_cos_sims.append(cos.cpu())
            all_lm_losses.append(per_pos_loss.cpu())

    # =============================================
    # 1. Residual PCA
    # =============================================
    print("\n=== Residual PCA ===")
    residuals = torch.cat(all_residuals, dim=0)
    N, T, D = residuals.shape
    residuals_flat = residuals.reshape(-1, D).numpy()

    residuals_centered = residuals_flat - residuals_flat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(residuals_centered, full_matrices=False)
    explained_var = S**2 / (S**2).sum()
    cumulative_var = np.cumsum(explained_var)

    rank_50 = int(np.searchsorted(cumulative_var, 0.50) + 1)
    rank_75 = int(np.searchsorted(cumulative_var, 0.75) + 1)
    rank_90 = int(np.searchsorted(cumulative_var, 0.90) + 1)
    rank_95 = int(np.searchsorted(cumulative_var, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumulative_var, 0.99) + 1)

    p = S**2 / (S**2).sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    effective_rank_entropy = float(np.exp(entropy))

    print(f"  Rank for 50%: {rank_50}, 75%: {rank_75}, 90%: {rank_90}, "
          f"95%: {rank_95}, 99%: {rank_99}")
    print(f"  Effective rank (entropy): {effective_rank_entropy:.1f}")
    print(f"  Top-1: {explained_var[0]:.4f}, Top-5: {cumulative_var[4]:.4f}, "
          f"Top-10: {cumulative_var[9]:.4f}")

    # Per-position projection onto top PCs
    top_k = 10
    projections = residuals_centered @ Vt[:top_k].T
    projections_by_pos = projections.reshape(N, T, top_k)
    mean_proj_by_pos = projections_by_pos.mean(axis=0)

    # =============================================
    # 2. CKA
    # =============================================
    print("\n=== CKA ===")
    fwd_post_attn_arr = torch.cat(all_fwd_post_attn, dim=0).reshape(-1, D).numpy()
    block1_post_attn_arr = torch.cat(all_block1_post_attn, dim=0).reshape(-1, D).numpy()
    fwd_output_arr = torch.cat(all_fwd_output, dim=0).reshape(-1, D).numpy()
    block1_output_arr = torch.cat(all_block1_output, dim=0).reshape(-1, D).numpy()
    source_arr = torch.cat(all_source, dim=0).reshape(-1, D).numpy()

    n_cka = min(8000, fwd_post_attn_arr.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(fwd_post_attn_arr.shape[0], n_cka, replace=False)

    cka_post_attn = linear_cka(fwd_post_attn_arr[idx], block1_post_attn_arr[idx])
    cka_output = linear_cka(fwd_output_arr[idx], block1_output_arr[idx])
    cka_input_to_fwd = linear_cka(source_arr[idx], fwd_output_arr[idx])
    cka_input_to_block1 = linear_cka(source_arr[idx], block1_output_arr[idx])

    print(f"  Post-attention (fwd vs block1): {cka_post_attn:.4f}")
    print(f"  Final output (fwd vs block1):   {cka_output:.4f}")
    print(f"  Input→fwd output:               {cka_input_to_fwd:.4f}")
    print(f"  Input→block1 output:            {cka_input_to_block1:.4f}")

    # =============================================
    # 3. Attention pattern comparison
    # =============================================
    print("\n=== Attention patterns ===")
    fwd_attn_avg = torch.stack(all_fwd_attn_weights).mean(dim=0)  # (1, T, T)
    block1_attn_avg = torch.stack(all_block1_attn_weights).mean(dim=0)  # (4, T, T)

    fwd_attn_flat = fwd_attn_avg[0].reshape(-1)
    attn_cosine_sims = []
    for h in range(n_head):
        b1_flat = block1_attn_avg[h].reshape(-1)
        sim = F.cosine_similarity(
            fwd_attn_flat.unsqueeze(0), b1_flat.unsqueeze(0)
        ).item()
        attn_cosine_sims.append(sim)
        print(f"  Fwd head vs Block1 head {h}: cosine={sim:.4f}")

    # Per-position KL divergence (averaged over positions)
    attn_kl_divs = []
    for h in range(n_head):
        kl = 0.0
        count = 0
        for t in range(T):
            p_dist = fwd_attn_avg[0, t, :t+1]
            q_dist = block1_attn_avg[h, t, :t+1]
            if p_dist.sum() > 0 and q_dist.sum() > 0:
                p_dist = p_dist.clamp(min=1e-10)
                q_dist = q_dist.clamp(min=1e-10)
                kl += float(F.kl_div(
                    q_dist.log(), p_dist, reduction='sum', log_target=False
                ))
                count += 1
        attn_kl_divs.append(kl / count if count > 0 else 0.0)

    # =============================================
    # 4. Residual conditioning
    # =============================================
    print("\n=== Residual conditioning ===")
    tokens_all = torch.cat(all_tokens_batch, dim=0)
    res_norms_all = torch.cat(all_residual_norms, dim=0)
    cos_sims_all = torch.cat(all_cos_sims, dim=0)
    lm_losses_all = torch.cat(all_lm_losses, dim=0)

    # By token frequency
    token_freq_cpu = token_freq.cpu()
    freq_buckets = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
    freq_labels = ["<1e-6", "1e-6–1e-5", "1e-5–1e-4", "1e-4–1e-3",
                   "1e-3–1e-2", ">1e-2"]

    flat_tokens = tokens_all.reshape(-1)
    flat_res = res_norms_all.reshape(-1)
    flat_freqs = token_freq_cpu[flat_tokens]

    bucket_stats = []
    for i in range(len(freq_buckets) - 1):
        mask = (flat_freqs >= freq_buckets[i]) & (flat_freqs < freq_buckets[i + 1])
        n_in = int(mask.sum())
        if n_in > 0:
            bucket_stats.append({
                "label": freq_labels[i],
                "mean": float(flat_res[mask].mean()),
                "std": float(flat_res[mask].std()),
                "count": n_in,
            })
        else:
            bucket_stats.append({
                "label": freq_labels[i], "mean": 0, "std": 0, "count": 0,
            })

    for bs in bucket_stats:
        if bs["count"] > 0:
            print(f"  freq {bs['label']:>12s}: res_norm={bs['mean']:.4f} "
                  f"(n={bs['count']:,})")

    # By position
    mean_res_by_pos = res_norms_all.mean(dim=0).numpy()
    mean_cos_by_pos = cos_sims_all.mean(dim=0).numpy()

    # By LM loss quartile
    flat_lm = lm_losses_all.reshape(-1)
    flat_res_for_lm = res_norms_all[:, :lm_losses_all.shape[1]].reshape(-1)
    quartiles = torch.quantile(flat_lm, torch.tensor([0.25, 0.5, 0.75]))
    q_labels = ["Q1 (low loss)", "Q2", "Q3", "Q4 (high loss)"]
    q_bounds = [float('-inf'), quartiles[0].item(), quartiles[1].item(),
                quartiles[2].item(), float('inf')]
    lm_quartile_stats = []
    for i in range(4):
        mask = (flat_lm >= q_bounds[i]) & (flat_lm < q_bounds[i + 1])
        lm_quartile_stats.append(float(flat_res_for_lm[mask].mean()) if mask.sum() > 0 else 0.0)
        print(f"  LM {q_labels[i]:>16s}: res_norm={lm_quartile_stats[-1]:.4f}")

    # =============================================
    # 5. Weight-space comparison
    # =============================================
    print("\n=== Weight comparison ===")
    head_dim = n_embd // n_head

    block1_qkv_w = model.transformer.h[1].attn.c_attn.weight.data.cpu()
    block1_q_w = block1_qkv_w[:n_embd].view(n_head, head_dim, n_embd)
    block1_k_w = block1_qkv_w[n_embd:2*n_embd].view(n_head, head_dim, n_embd)
    block1_v_w = block1_qkv_w[2*n_embd:].view(n_head, head_dim, n_embd)

    fwd_q_w = fwd_model.q_proj.weight.data.cpu()
    fwd_k_w = fwd_model.k_proj.weight.data.cpu()
    fwd_v_w = fwd_model.v_proj.weight.data.cpu()

    weight_sims = {"q": [], "k": [], "v": []}
    for h in range(n_head):
        q_sim = F.cosine_similarity(
            fwd_q_w.reshape(1, -1), block1_q_w[h].reshape(1, -1)
        ).item()
        k_sim = F.cosine_similarity(
            fwd_k_w.reshape(1, -1), block1_k_w[h].reshape(1, -1)
        ).item()
        v_sim = F.cosine_similarity(
            fwd_v_w.reshape(1, -1), block1_v_w[h].reshape(1, -1)
        ).item()
        weight_sims["q"].append(q_sim)
        weight_sims["k"].append(k_sim)
        weight_sims["v"].append(v_sim)
        print(f"  Head {h}: Q={q_sim:.4f}, K={k_sim:.4f}, V={v_sim:.4f}")

    # Subspace overlap: project fwd weights into block1's head subspaces
    # For each block1 head, compute how much of fwd's Q lives in that head's Q subspace
    fwd_q_norm = fwd_q_w / fwd_q_w.norm()
    subspace_overlaps = []
    for h in range(n_head):
        # block1 head h's Q subspace is spanned by its rows (64 x 256)
        # Project fwd's Q onto this subspace
        basis = block1_q_w[h]  # (64, 256)
        # Orthonormalize
        U_h, _, _ = torch.linalg.svd(basis, full_matrices=False)  # (64, 64)
        # U_h columns are orthonormal basis for the head's row space
        # But we want column space of basis.T... let me think.
        # basis has shape (64, 256). Its row space is a 64-dim subspace of R^256.
        # fwd_q_w has shape (64, 256). Same structure.
        # Overlap = how much of fwd_q's row space overlaps with block1_head_h's row space
        # Compute via: ||P_h @ fwd_q_w.T||_F / ||fwd_q_w||_F
        # where P_h projects onto block1_head_h's row space
        U_basis, _, _ = torch.linalg.svd(basis.T, full_matrices=False)  # (256, 64)
        proj = U_basis @ (U_basis.T @ fwd_q_w.T)  # (256, 64)
        overlap = proj.norm() / fwd_q_w.norm()
        subspace_overlaps.append(float(overlap))
    print(f"  Q subspace overlaps: {[f'{o:.3f}' for o in subspace_overlaps]}")

    # =============================================
    # Generate figure
    # =============================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("A2A Forward Model Structure Analysis\n"
                 f"(post_block0 → post_block1, {n_eval_batches} eval batches)",
                 fontsize=13, y=0.995)

    # --- Row 1: Residual PCA ---
    ax = axes[0, 0]
    n_show = min(50, len(cumulative_var))
    ax.plot(range(1, n_show + 1), cumulative_var[:n_show], 'b-o', markersize=3)
    ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='99%')
    ax.axvline(x=rank_90, color='r', linestyle=':', alpha=0.3)
    ax.axvline(x=rank_95, color='orange', linestyle=':', alpha=0.3)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title(f'Residual PCA (eff. rank={effective_rank_entropy:.1f})')
    ax.legend(fontsize=8)
    ax.set_xlim(1, n_show)

    ax = axes[0, 1]
    ax.semilogy(range(1, n_show + 1), (S[:n_show] / S[0]), 'b-o', markersize=3)
    ax.set_xlabel('Component')
    ax.set_ylabel('Relative singular value (log)')
    ax.set_title(f'Residual SV spectrum\n'
                 f'(r90={rank_90}, r95={rank_95}, r99={rank_99})')

    ax = axes[0, 2]
    for k in range(min(5, top_k)):
        ax.plot(range(T), mean_proj_by_pos[:, k],
                label=f'PC{k+1} ({explained_var[k]:.3f})', alpha=0.7)
    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Mean projection onto PC')
    ax.set_title('Top residual PCs by position')
    ax.legend(fontsize=7)

    # --- Row 2: CKA + Attention ---
    ax = axes[1, 0]
    cka_labels = ['Post-attn\n(fwd vs b1)', 'Output\n(fwd vs b1)',
                  'Input→fwd', 'Input→b1']
    cka_values = [cka_post_attn, cka_output, cka_input_to_fwd, cka_input_to_block1]
    colors = ['steelblue', 'coral', 'lightsteelblue', 'lightsalmon']
    bars = ax.bar(cka_labels, cka_values, color=colors)
    ax.set_ylabel('Linear CKA')
    ax.set_title('Representational alignment')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, cka_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

    ax = axes[1, 1]
    x_pos = range(n_head)
    bars = ax.bar(x_pos, attn_cosine_sims, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Head {h}' for h in range(n_head)])
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Fwd attention pattern vs Block1 heads')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(attn_cosine_sims):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    ax = axes[1, 2]
    x_pos_arr = np.arange(n_head)
    width = 0.2
    ax.bar(x_pos_arr - width, weight_sims["q"], width, label='Q', color='steelblue')
    ax.bar(x_pos_arr, weight_sims["k"], width, label='K', color='coral')
    ax.bar(x_pos_arr + width, weight_sims["v"], width, label='V', color='forestgreen')
    ax.set_xticks(x_pos_arr)
    ax.set_xticklabels([f'Head {h}' for h in range(n_head)])
    ax.set_ylabel('Weight cosine similarity')
    ax.set_title('Fwd QKV weights vs Block1 heads')
    ax.legend(fontsize=8)

    # --- Row 3: Residual conditioning ---
    ax = axes[2, 0]
    valid_buckets = [bs for bs in bucket_stats if bs["count"] > 0]
    if valid_buckets:
        ax.bar([bs["label"] for bs in valid_buckets],
               [bs["mean"] for bs in valid_buckets], color='steelblue')
        ax.errorbar(range(len(valid_buckets)),
                    [bs["mean"] for bs in valid_buckets],
                    yerr=[bs["std"] / max(1, bs["count"]**0.5) for bs in valid_buckets],
                    fmt='none', color='black', capsize=3)
    ax.set_ylabel('Mean residual norm')
    ax.set_title('Residual by token frequency')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[2, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(range(T), mean_res_by_pos, 'b-', label='Residual norm', alpha=0.7)
    l2 = ax2.plot(range(T), mean_cos_by_pos, 'r-', label='Cosine sim', alpha=0.7)
    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Residual norm', color='b')
    ax2.set_ylabel('Cosine similarity', color='r')
    ax.set_title('Prediction quality by position')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')

    ax = axes[2, 2]
    ax.bar(q_labels, lm_quartile_stats, color='steelblue')
    ax.set_ylabel('Mean residual norm')
    ax.set_title('Residual by LM loss quartile')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "structure_analysis.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    # =============================================
    # Save results
    # =============================================
    results = {
        "residual_pca": {
            "rank_50": rank_50,
            "rank_75": rank_75,
            "rank_90": rank_90,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "effective_rank_entropy": effective_rank_entropy,
            "explained_variance_top20": explained_var[:20].tolist(),
            "cumulative_variance_top50": cumulative_var[:50].tolist(),
            "singular_values_top50_relative": (S[:50] / S[0]).tolist(),
        },
        "cka": {
            "post_attention_fwd_vs_block1": cka_post_attn,
            "output_fwd_vs_block1": cka_output,
            "input_to_fwd_output": cka_input_to_fwd,
            "input_to_block1_output": cka_input_to_block1,
        },
        "attention_similarity": {
            "cosine": {f"head_{h}": attn_cosine_sims[h] for h in range(n_head)},
            "kl_div": {f"head_{h}": attn_kl_divs[h] for h in range(n_head)},
        },
        "weight_similarity": {
            f"head_{h}": {
                "q_cosine": weight_sims["q"][h],
                "k_cosine": weight_sims["k"][h],
                "v_cosine": weight_sims["v"][h],
            } for h in range(n_head)
        },
        "q_subspace_overlap": {
            f"head_{h}": subspace_overlaps[h] for h in range(n_head)
        },
        "residual_by_frequency": {
            bs["label"]: {"mean": bs["mean"], "std": bs["std"], "count": bs["count"]}
            for bs in bucket_stats
        },
        "residual_by_lm_quartile": {
            q_labels[i]: lm_quartile_stats[i] for i in range(4)
        },
        "mean_residual_by_position": mean_res_by_pos.tolist(),
        "mean_cosine_by_position": mean_cos_by_pos.tolist(),
        "top_pc_by_position": mean_proj_by_pos.tolist(),
    }

    results_path = os.path.join(save_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\nResults saved to {results_path}")
    return results


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_loop_analyze(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    n_eval_batches: int = 40,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    probe_steps: int = 500,
    open_loop: bool = False,
):
    """Post-hoc analysis of closed-loop (or open-loop baseline) training.

    Loads saved checkpoints and runs self-map probes:
    1. Can post_block3 predict fwd_pred? (internalized forward model)
    2. Does this differ with vs without injection active?
    3. Can post_block3 predict the residual (actual - predicted)?

    Set open_loop=True to analyze the open-loop baseline model instead.
    """
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    mode_label = "OPEN-LOOP baseline" if open_loop else "CLOSED-LOOP"
    print(f"A2A {mode_label} analysis on {device}")

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
    split = int(0.9 * len(data))
    val_data = data[split:]

    def get_batch():
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- Load models ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    if open_loop:
        model_dir = (f"{DATA_DIR}/a2a_forward/transformer_L{fwd_n_layer}/"
                     f"{gap_tag}/P_{n_tokens}")
    else:
        model_dir = (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                     f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading from {model_dir}")

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True,
    ))
    model.eval()

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device, weights_only=True,
    ))
    fwd_model.eval()

    gate = None
    if not open_loop:
        gate = CerebellarGate(n_embd).to(device)
        gate.load_state_dict(torch.load(
            os.path.join(model_dir, "gate.pt"), map_location=device, weights_only=True,
        ))
        gate.eval()
        print(f"Gate injection norm: {gate.injection_norm():.4f}")
    else:
        print("Open-loop mode: no gate")

    # --- Collect activations ---
    print(f"Collecting activations ({n_eval_batches} batches)...")
    early_acts = []        # post_block0
    late_with_inj = []     # post_block3, injection active
    late_no_inj = []       # post_block3, no injection
    fwd_preds = []         # forward model predictions
    residuals = []         # actual - predicted

    with torch.no_grad():
        for i in range(n_eval_batches):
            vx, vy = get_batch()

            if gate is not None:
                def eval_cb_fn(act):
                    return gate(fwd_model(act))

                # With injection
                _, _, vi_with = model(
                    vx, vy, return_intermediates=True,
                    cerebellar_fn=eval_cb_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
            # Without injection (always computed)
            _, _, vi_without = model(vx, vy, return_intermediates=True)

            src = vi_without[predict_from]
            pred = fwd_model(src)
            actual_without = vi_without[predict_to]

            if gate is not None:
                actual_with = vi_with[predict_to]
            else:
                actual_with = actual_without

            early_acts.append(src.cpu())
            late_with_inj.append(actual_with.cpu())
            late_no_inj.append(actual_without.cpu())
            fwd_preds.append(pred.cpu())
            residuals.append((actual_without - pred).cpu())

            if i % 10 == 0:
                print(f"  batch {i}/{n_eval_batches}")

    # Flatten
    early = torch.cat(early_acts, dim=0).reshape(-1, n_embd)
    late_w = torch.cat(late_with_inj, dim=0).reshape(-1, n_embd)
    late_wo = torch.cat(late_no_inj, dim=0).reshape(-1, n_embd)
    fp = torch.cat(fwd_preds, dim=0).reshape(-1, n_embd)
    res = torch.cat(residuals, dim=0).reshape(-1, n_embd)

    n = early.shape[0]
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    print(f"Probe dataset: {n} samples ({n_train} train, {n - n_train} test)")

    # --- Run probes ---
    def train_probe(X_name, X_data, Y_name, Y_data):
        probe = nn.Linear(n_embd, n_embd).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        X_tr = X_data[train_idx].to(device)
        Y_tr = Y_data[train_idx].to(device)
        X_te = X_data[test_idx].to(device)
        Y_te = Y_data[test_idx].to(device)

        probe_bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (probe_bs,))
            pred = probe(X_tr[idx])
            loss = F.mse_loss(pred, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred_test = probe(X_te)
            mse = F.mse_loss(pred_test, Y_te).item()
            var = Y_te.var().item()
            r2 = 1.0 - mse / var if var > 0 else 0.0
            cos = F.cosine_similarity(pred_test, Y_te, dim=-1).mean().item()

        print(f"  {X_name} → {Y_name}: R²={r2:.4f}, cosine={cos:.4f}")
        return {"r2": r2, "cosine": cos, "mse": mse}

    results = {}

    # Probe 1: post_block3 (with inj) → fwd_pred
    print("\n=== Self-map probes: can post_block3 predict fwd_pred? ===")
    results["late_with_inj__to__fwd_pred"] = train_probe(
        "post_block3 (with inj)", late_w, "fwd_pred", fp,
    )
    # Probe 2: post_block3 (no inj) → fwd_pred
    results["late_no_inj__to__fwd_pred"] = train_probe(
        "post_block3 (no inj)", late_wo, "fwd_pred", fp,
    )

    # Probe 3: post_block3 (with inj) → residual
    print("\n=== Novelty probes: can post_block3 predict the residual? ===")
    results["late_with_inj__to__residual"] = train_probe(
        "post_block3 (with inj)", late_w, "residual", res,
    )
    # Probe 4: post_block3 (no inj) → residual
    results["late_no_inj__to__residual"] = train_probe(
        "post_block3 (no inj)", late_wo, "residual", res,
    )

    # Probe 5-6: original self-map (post_block3 → post_block0)
    print("\n=== Original self-map: can post_block3 predict post_block0? ===")
    results["late_with_inj__to__early"] = train_probe(
        "post_block3 (with inj)", late_w, "post_block0", early,
    )
    results["late_no_inj__to__early"] = train_probe(
        "post_block3 (no inj)", late_wo, "post_block0", early,
    )

    # Baseline: how well does fwd_pred predict post_block3?
    print("\n=== Forward model quality ===")
    with torch.no_grad():
        cos_with = F.cosine_similarity(
            fp.to(device), late_w.to(device), dim=-1
        ).mean().item()
        cos_without = F.cosine_similarity(
            fp.to(device), late_wo.to(device), dim=-1
        ).mean().item()
    results["fwd_pred_cosine_vs_late_with_inj"] = cos_with
    results["fwd_pred_cosine_vs_late_no_inj"] = cos_without
    print(f"  fwd_pred vs post_block3 (with inj):  cosine={cos_with:.4f}")
    print(f"  fwd_pred vs post_block3 (no inj):    cosine={cos_without:.4f}")

    # --- Save ---
    tag = "openloop_analysis" if open_loop else "loop_analysis"
    save_path = os.path.join(model_dir, f"{tag}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {save_path}")
    return results


@app.local_entrypoint()
def analyze(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_analyze.remote(
        n_tokens=n_tokens, block_size=block_size,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print(f"A2A analysis complete:")
    pca = result["residual_pca"]
    print(f"  Residual effective rank: {pca['effective_rank_entropy']:.1f}")
    print(f"  Rank for 90%: {pca['rank_90']}, 95%: {pca['rank_95']}")
    cka = result["cka"]
    print(f"  CKA post-attn: {cka['post_attention_fwd_vs_block1']:.4f}")
    print(f"  CKA output: {cka['output_fwd_vs_block1']:.4f}")


@app.local_entrypoint()
def loop_analyze(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    open_loop: bool = False,
):
    result = a2a_loop_analyze.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        open_loop=open_loop,
    )
    print(f"A2A loop analysis complete:")
    for k, v in result.items():
        if isinstance(v, dict):
            r2 = v.get('r2', v.get('cosine', ''))
            cos = v.get('cosine', '')
            print(f"  {k}: R²={r2:.4f}, cos={cos:.4f}" if isinstance(r2, float)
                  else f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
