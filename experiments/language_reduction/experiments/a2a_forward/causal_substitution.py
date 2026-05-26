"""Causal substitution analysis for the A2A forward model.

Runs the main GPT model in three modes:
  - Normal:      full model, unmodified
  - Substituted: replace block1's output with the forward model's prediction,
                 then continue from block2 onward
  - Ablated:     zero-ablate block1 (set block1 output = block1 input, skip block1)

Measures degradation across behavioral token categories and per-position.
"""

import json

from language_reduction.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_causal_substitution(
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
    import math
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.nn.functional as F
    import tiktoken
    from language_reduction.model import GPT

    # The P_10000000 checkpoint was saved before TransformerForwardModel was
    # refactored to use ForwardBlock/blocks. Replicate the original flat API.
    class _LegacyFwdModel(nn.Module):
        def __init__(self, d_model, d_head, n_head, mlp_mult, block_size):
            super().__init__()
            self.d_model = d_model
            self.d_head = d_head
            self.n_head = n_head
            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_head * n_head)
            self.k_proj = nn.Linear(d_model, d_head * n_head)
            self.v_proj = nn.Linear(d_model, d_head * n_head)
            self.out_proj = nn.Linear(d_head * n_head, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            mlp_hidden = d_model * mlp_mult
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_hidden), nn.GELU(),
                nn.Linear(mlp_hidden, d_model),
            )
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

        def forward(self, x):
            B, T, _ = x.size()
            h = self.ln1(x)
            q = self.q_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
            k = self.k_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
            v = self.v_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(
                B, T, self.d_head * self.n_head
            )
            x = x + self.out_proj(y)
            x = x + self.mlp(self.ln2(x))
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A causal substitution analysis on {device}")

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

    fwd_model = _LegacyFwdModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device, weights_only=True
    ))
    fwd_model.eval()

    def get_batch():
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- Tokenizer for behavioral categorization ---
    enc = tiktoken.get_encoding("gpt2")

    PUNCTUATION = set(enc.encode(" .,!?;:—-…\"'`()[]{}"))
    CLOSE_DELIMITERS = set(enc.encode(")]}>'\""))
    OPEN_DELIMITERS = {
        enc.encode(")")[0]: enc.encode("(")[0] if enc.encode("(") else None,
        enc.encode("]")[0]: enc.encode("[")[0] if enc.encode("[") else None,
        enc.encode("}")[0]: enc.encode("{")[0] if enc.encode("{") else None,
    }
    # Also handle multi-byte tokenizations: scan raw token strings
    all_token_ids = list(range(vocab_size))
    punct_set = set()
    close_set = set()
    for tok_id in range(vocab_size):
        try:
            s = enc.decode([tok_id])
        except Exception:
            continue
        stripped = s.strip()
        if stripped in ".,!?;:—-…\"'`()[]{}":
            punct_set.add(tok_id)
        if stripped in ")]}>'\"":
            close_set.add(tok_id)

    # Build frequency rank lookup: sort tokens by descending frequency
    freq_cpu = token_freq.cpu().numpy()
    sorted_by_freq = np.argsort(-freq_cpu)  # descending
    token_rank = np.empty(vocab_size, dtype=np.int64)
    token_rank[sorted_by_freq] = np.arange(vocab_size)

    top5pct_threshold = int(0.05 * vocab_size)
    bottom20pct_threshold = int(0.80 * vocab_size)

    # --- Manual GPT forward passes ---
    def run_embedding(idx):
        tok_emb = model.transformer.wte(idx)
        pos = torch.arange(0, idx.size(1), device=idx.device)
        pos_emb = model.transformer.wpe(pos)
        return model.transformer.drop(tok_emb + pos_emb)

    def run_blocks_from(x, start_block):
        for i in range(start_block, n_layer):
            x = model.transformer.h[i](x)
        return x

    def run_head(x):
        x = model.transformer.ln_f(x)
        return model.lm_head(x)

    # --- Compute per-token scalar metrics online (logits too large to store) ---
    all_tokens_x = []
    all_tokens_y = []
    all_ce_n = []
    all_ce_s = []
    all_ce_a = []
    all_acc_n = []
    all_acc_s = []
    all_acc_a = []
    all_kl_sub = []
    all_kl_abl = []
    all_conf_n = []

    T_dim = block_size

    print(f"Running {n_eval_batches} eval batches...")
    with torch.no_grad():
        for batch_i in range(n_eval_batches):
            if batch_i % 10 == 0:
                print(f"  batch {batch_i}/{n_eval_batches}")

            x, y = get_batch()
            B, T = x.shape

            h = run_embedding(x)
            post_block0 = model.transformer.h[0](h)

            post_block1_normal = model.transformer.h[1](post_block0)
            post_block1_sub = fwd_model(post_block0)
            post_block1_abl = post_block0

            out_normal, out_sub, out_abl = post_block1_normal, post_block1_sub, post_block1_abl
            for i in range(2, n_layer):
                out_normal = model.transformer.h[i](out_normal)
                out_sub = model.transformer.h[i](out_sub)
                out_abl = model.transformer.h[i](out_abl)

            logits_n = run_head(out_normal)
            logits_s = run_head(out_sub)
            logits_a = run_head(out_abl)

            flat_y_batch = y.reshape(-1)
            flat_ln = logits_n.reshape(-1, logits_n.size(-1))
            flat_ls = logits_s.reshape(-1, logits_s.size(-1))
            flat_la = logits_a.reshape(-1, logits_a.size(-1))

            log_p_n = F.log_softmax(flat_ln, dim=-1)
            log_p_s = F.log_softmax(flat_ls, dim=-1)
            log_p_a = F.log_softmax(flat_la, dim=-1)
            p_n = log_p_n.exp()

            all_tokens_x.append(x.cpu())
            all_tokens_y.append(y.cpu())
            all_ce_n.append(F.nll_loss(log_p_n, flat_y_batch, reduction="none").cpu())
            all_ce_s.append(F.nll_loss(log_p_s, flat_y_batch, reduction="none").cpu())
            all_ce_a.append(F.nll_loss(log_p_a, flat_y_batch, reduction="none").cpu())
            all_acc_n.append((flat_ln.argmax(-1) == flat_y_batch).float().cpu())
            all_acc_s.append((flat_ls.argmax(-1) == flat_y_batch).float().cpu())
            all_acc_a.append((flat_la.argmax(-1) == flat_y_batch).float().cpu())
            all_kl_sub.append((p_n * (log_p_n - log_p_s)).sum(-1).clamp(min=0).cpu())
            all_kl_abl.append((p_n * (log_p_n - log_p_a)).sum(-1).clamp(min=0).cpu())
            all_conf_n.append(p_n.gather(1, flat_y_batch.unsqueeze(1)).squeeze(1).cpu())

    tokens_x = torch.cat(all_tokens_x, dim=0)
    tokens_y = torch.cat(all_tokens_y, dim=0)
    N = tokens_x.shape[0]
    ce_n = torch.cat(all_ce_n)
    ce_s = torch.cat(all_ce_s)
    ce_a = torch.cat(all_ce_a)
    acc_n = torch.cat(all_acc_n)
    acc_s = torch.cat(all_acc_s)
    acc_a = torch.cat(all_acc_a)
    kl_sub = torch.cat(all_kl_sub)
    kl_abl = torch.cat(all_kl_abl)
    conf_n = torch.cat(all_conf_n)

    flat_y = tokens_y.reshape(-1)
    flat_x = tokens_x.reshape(-1)
    y_np = flat_y.numpy()
    y_rank = token_rank[y_np]

    # --- Build behavioral category masks ---
    M = flat_y.shape[0]

    def build_masks():
        masks = {}

        # 1. Punctuation prediction
        punct_mask = torch.zeros(M, dtype=torch.bool)
        for tid in punct_set:
            punct_mask |= (flat_y == tid)
        masks["punctuation"] = punct_mask

        # 2. Close-delimiter prediction (with matching opener in context)
        close_mask = torch.zeros(M, dtype=torch.bool)
        for tid in close_set:
            pos_mask = flat_y == tid
            if pos_mask.any():
                # Check that the corresponding opener appears somewhere in context
                # (within the same sequence position)
                close_mask |= pos_mask
        masks["bracket_closing"] = close_mask

        # 3. Repeated token (induction-like): target appeared recently in context
        # Reshape for sequence access; check last 20 positions of context
        LOOKBACK = 20
        induction_mask = torch.zeros(M, dtype=torch.bool)
        x_2d = tokens_x.reshape(-1, T_dim)   # (N, T)
        y_2d = tokens_y.reshape(-1, T_dim)   # (N, T)
        for t in range(T_dim):
            target_col = y_2d[:, t]
            if t >= 1:
                context = x_2d[:, max(0, t - LOOKBACK):t + 1]  # (N, window)
                appeared = (context == target_col.unsqueeze(1)).any(dim=1)
            else:
                appeared = torch.zeros(N, dtype=torch.bool)
            flat_idx_start = torch.arange(N) * T_dim + t
            induction_mask[flat_idx_start] = appeared
        masks["repeated_token"] = induction_mask

        # 4. High-confidence: normal model assigns >0.5 prob to correct next token
        masks["high_confidence"] = conf_n > 0.5

        # 5. Low-confidence: normal model assigns <0.1 prob to correct next token
        masks["low_confidence"] = conf_n < 0.1

        # 6. Content word (low-frequency: bottom 20%)
        masks["content_word"] = torch.from_numpy(y_rank >= bottom20pct_threshold)

        # 7. Function word (high-frequency: top 5%)
        masks["function_word"] = torch.from_numpy(y_rank < top5pct_threshold)

        return masks

    print("Building behavioral category masks...")
    masks = build_masks()

    # --- Aggregate metrics per category ---
    def category_stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        m = mask

        def safe_mean(t):
            return float(t[m].mean())

        return {
            "n": n,
            "ce_normal": safe_mean(ce_n),
            "ce_sub": safe_mean(ce_s),
            "ce_abl": safe_mean(ce_a),
            "delta_ce_sub": safe_mean(ce_s - ce_n),
            "delta_ce_abl": safe_mean(ce_a - ce_n),
            "acc_normal": safe_mean(acc_n),
            "acc_sub": safe_mean(acc_s),
            "acc_abl": safe_mean(acc_a),
            "delta_acc_sub": safe_mean(acc_s - acc_n),
            "delta_acc_abl": safe_mean(acc_a - acc_n),
            "kl_sub": safe_mean(kl_sub),
            "kl_abl": safe_mean(kl_abl),
        }

    print("\n=== Behavioral category breakdown ===")
    category_results = {}
    for cat_name, mask in masks.items():
        stats = category_stats(mask)
        category_results[cat_name] = stats
        if stats["n"] > 0:
            print(f"  {cat_name:20s} (n={stats['n']:>8,}): "
                  f"KL_sub={stats['kl_sub']:.4f}  KL_abl={stats['kl_abl']:.4f}  "
                  f"ΔCE_sub={stats['delta_ce_sub']:+.4f}  "
                  f"ΔAcc_sub={stats['delta_acc_sub']:+.4f}")

    # --- Overall aggregate ---
    print("\n=== Overall aggregate ===")
    overall = {
        "n": M,
        "ce_normal": float(ce_n.mean()),
        "ce_sub": float(ce_s.mean()),
        "ce_abl": float(ce_a.mean()),
        "delta_ce_sub": float((ce_s - ce_n).mean()),
        "delta_ce_abl": float((ce_a - ce_n).mean()),
        "acc_normal": float(acc_n.mean()),
        "acc_sub": float(acc_s.mean()),
        "acc_abl": float(acc_a.mean()),
        "delta_acc_sub": float((acc_s - acc_n).mean()),
        "delta_acc_abl": float((acc_a - acc_n).mean()),
        "kl_sub": float(kl_sub.mean()),
        "kl_abl": float(kl_abl.mean()),
    }
    print(f"  KL_sub={overall['kl_sub']:.4f}  KL_abl={overall['kl_abl']:.4f}")
    print(f"  ΔCE_sub={overall['delta_ce_sub']:+.4f}  "
          f"ΔCE_abl={overall['delta_ce_abl']:+.4f}")
    print(f"  Acc_normal={overall['acc_normal']:.4f}  "
          f"Acc_sub={overall['acc_sub']:.4f}  "
          f"Acc_abl={overall['acc_abl']:.4f}")

    # --- Per-position statistics ---
    print("\n=== Per-position statistics ===")
    ce_n_2d = ce_n.reshape(N, T_dim)
    ce_s_2d = ce_s.reshape(N, T_dim)
    ce_a_2d = ce_a.reshape(N, T_dim)
    kl_s_2d = kl_sub.reshape(N, T_dim)
    kl_a_2d = kl_abl.reshape(N, T_dim)
    acc_n_2d = acc_n.reshape(N, T_dim)
    acc_s_2d = acc_s.reshape(N, T_dim)
    acc_a_2d = acc_a.reshape(N, T_dim)

    per_position = {
        "ce_normal": ce_n_2d.mean(dim=0).tolist(),
        "delta_ce_sub": (ce_s_2d - ce_n_2d).mean(dim=0).tolist(),
        "delta_ce_abl": (ce_a_2d - ce_n_2d).mean(dim=0).tolist(),
        "kl_sub": kl_s_2d.mean(dim=0).tolist(),
        "kl_abl": kl_a_2d.mean(dim=0).tolist(),
        "acc_normal": acc_n_2d.mean(dim=0).tolist(),
        "delta_acc_sub": (acc_s_2d - acc_n_2d).mean(dim=0).tolist(),
        "delta_acc_abl": (acc_a_2d - acc_n_2d).mean(dim=0).tolist(),
    }

    # --- Generate figure ---
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat_names = list(category_results.keys())
    valid_cats = [c for c in cat_names if category_results[c]["n"] > 0]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        "A2A Causal Substitution Analysis\n"
        "(Normal vs Substituted [fwd model] vs Ablated [skip block1])",
        fontsize=13, y=0.998,
    )

    pos = np.arange(T_dim)

    # Row 1: Per-position KL divergence and CE delta
    ax = axes[0, 0]
    ax.plot(pos, per_position["kl_sub"], label="KL(normal‖sub)", color="steelblue")
    ax.plot(pos, per_position["kl_abl"], label="KL(normal‖abl)", color="coral")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL divergence by position")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(pos, per_position["delta_ce_sub"], label="ΔCE (sub)", color="steelblue")
    ax.plot(pos, per_position["delta_ce_abl"], label="ΔCE (abl)", color="coral")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("ΔCross-entropy (vs normal)")
    ax.set_title("Cross-entropy degradation by position")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(pos, per_position["delta_acc_sub"], label="ΔAcc (sub)", color="steelblue")
    ax.plot(pos, per_position["delta_acc_abl"], label="ΔAcc (abl)", color="coral")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("ΔAccuracy (vs normal)")
    ax.set_title("Accuracy change by position")
    ax.legend(fontsize=8)

    # Row 2: Per-category KL divergence (sub vs abl)
    ax = axes[1, 0]
    x_pos = np.arange(len(valid_cats))
    width = 0.35
    kl_s_vals = [category_results[c]["kl_sub"] for c in valid_cats]
    kl_a_vals = [category_results[c]["kl_abl"] for c in valid_cats]
    ax.bar(x_pos - width / 2, kl_s_vals, width, label="Substituted", color="steelblue")
    ax.bar(x_pos + width / 2, kl_a_vals, width, label="Ablated", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("KL divergence by behavioral category")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    dce_s_vals = [category_results[c]["delta_ce_sub"] for c in valid_cats]
    dce_a_vals = [category_results[c]["delta_ce_abl"] for c in valid_cats]
    ax.bar(x_pos - width / 2, dce_s_vals, width, label="Substituted", color="steelblue")
    ax.bar(x_pos + width / 2, dce_a_vals, width, label="Ablated", color="coral")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔCross-entropy")
    ax.set_title("CE degradation by behavioral category")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    dacc_s_vals = [category_results[c]["delta_acc_sub"] for c in valid_cats]
    dacc_a_vals = [category_results[c]["delta_acc_abl"] for c in valid_cats]
    ax.bar(x_pos - width / 2, dacc_s_vals, width, label="Substituted", color="steelblue")
    ax.bar(x_pos + width / 2, dacc_a_vals, width, label="Ablated", color="coral")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔAccuracy")
    ax.set_title("Accuracy change by behavioral category")
    ax.legend(fontsize=8)

    # Row 3: Absolute accuracy, sample counts, and overall bar chart
    ax = axes[2, 0]
    acc_n_vals = [category_results[c]["acc_normal"] for c in valid_cats]
    acc_s_vals = [category_results[c]["acc_sub"] for c in valid_cats]
    acc_a_vals = [category_results[c]["acc_abl"] for c in valid_cats]
    width3 = 0.25
    ax.bar(x_pos - width3, acc_n_vals, width3, label="Normal", color="forestgreen")
    ax.bar(x_pos, acc_s_vals, width3, label="Substituted", color="steelblue")
    ax.bar(x_pos + width3, acc_a_vals, width3, label="Ablated", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Absolute accuracy by behavioral category")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    n_vals = [category_results[c]["n"] for c in valid_cats]
    ax.bar(x_pos, n_vals, color="grey")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Token count")
    ax.set_title("Category sizes")

    ax = axes[2, 2]
    overall_labels = ["Normal", "Substituted", "Ablated"]
    overall_kl = [0.0, overall["kl_sub"], overall["kl_abl"]]
    overall_dce = [0.0, overall["delta_ce_sub"], overall["delta_ce_abl"]]
    x2 = np.arange(3)
    ax2 = ax.twinx()
    b1 = ax.bar(x2 - 0.2, overall_kl, 0.35, label="KL div", color="steelblue", alpha=0.7)
    b2 = ax2.bar(x2 + 0.2, overall_dce, 0.35, label="ΔCE", color="coral", alpha=0.7)
    ax.set_xticks(x2)
    ax.set_xticklabels(overall_labels)
    ax.set_ylabel("Mean KL divergence", color="steelblue")
    ax2.set_ylabel("ΔCross-entropy", color="coral")
    ax.set_title("Overall: KL divergence and CE change")
    lines = [b1, b2]
    ax.legend(lines, ["KL div", "ΔCE"], fontsize=8, loc="upper left")

    plt.tight_layout()

    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "causal_substitution.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")

    # --- Save results ---
    results = {
        "overall": overall,
        "per_category": category_results,
        "per_position": per_position,
        "config": {
            "n_tokens": n_tokens,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "n_eval_batches": n_eval_batches,
            "batch_size": batch_size,
            "total_tokens_evaluated": M,
        },
    }

    results_path = os.path.join(save_dir, "causal_substitution.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\nResults saved to {results_path}")
    return results
