"""Behavior-conditioned residual analysis for the A2A forward model.

Tests the "epistemic novelty" hypothesis: does the forward model's residual
(actual post_block1 - predicted post_block1) have structure that correlates
with *behavioral* context — what kind of computation block1 is doing —
rather than just surface features like token frequency?
"""

import json
import math

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_behavioral_residual(
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
    import torch.nn as nn
    import numpy as np
    import torch.nn.functional as F
    import tiktoken
    from a2a_forward.model import GPT

    # The P_10000000 checkpoint was trained with the legacy single-block flat API.
    # Replicate it here to load without key remapping.
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
                nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model),
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
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, self.d_head * self.n_head)
            x = x + self.out_proj(y)
            x = x + self.mlp(self.ln2(x))
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Behavior-conditioned residual analysis on {device}")

    enc = tiktoken.get_encoding("gpt2")

    # --- Load data ---
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_token_data = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_token_data.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_token_data)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    token_counts = torch.bincount(train_data, minlength=vocab_size).float()
    token_freq = (token_counts / token_counts.sum()).cpu()

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

    # Punctuation token IDs using tiktoken
    PUNCT_CHARS = set(".,;:!?")
    OPENER_CHARS = set("([{\"'`")
    CLOSER_CHARS = set(")]}")

    def classify_tokens_syntactic(token_ids_np):
        """Returns dict of syntactic category masks, shape (B*T,).

        Processes per sequence to avoid boundary artifacts.
        """
        B, T = token_ids_np.shape
        n = B * T
        after_punct = np.zeros(n, dtype=bool)
        after_opener = np.zeros(n, dtype=bool)
        before_closer = np.zeros(n, dtype=bool)
        sentence_start = np.zeros(n, dtype=bool)
        mid_sentence = np.zeros(n, dtype=bool)

        for b in range(B):
            seq = token_ids_np[b]
            decoded = [enc.decode([int(t)]) for t in seq]
            base = b * T
            for t in range(T):
                prev_text = decoded[t - 1] if t > 0 else ""
                next_text = decoded[t + 1] if t < T - 1 else ""
                prev_s = prev_text.strip()
                i = base + t
                if prev_s and prev_s[-1] in PUNCT_CHARS:
                    after_punct[i] = True
                    if prev_s[-1] in ".!?" or '\n' in prev_text:
                        sentence_start[i] = True
                if prev_s and prev_s[-1] in OPENER_CHARS:
                    after_opener[i] = True
                next_s = next_text.strip()
                if next_s and next_s[0] in CLOSER_CHARS:
                    before_closer[i] = True
                if not after_punct[i] and not after_opener[i]:
                    mid_sentence[i] = True

        return {
            "after_punct": after_punct,
            "after_opener": after_opener,
            "before_closer": before_closer,
            "sentence_start": sentence_start,
            "mid_sentence": mid_sentence,
        }

    def compute_attention_categories(attn_w_per_token):
        """
        attn_w_per_token: (B, T, n_head, T) — averaged over heads gives (B, T, T).
        Returns masks of shape (B*T,).
        """
        B, T, H, _ = attn_w_per_token.shape
        # Average over heads: (B, T, T)
        avg_attn = attn_w_per_token.mean(dim=2)

        # Attention entropy per position
        eps = 1e-10
        attn_entropy = -(avg_attn * (avg_attn + eps).log()).sum(dim=-1)  # (B, T)

        # Dominant attended position (argmax)
        dominant_pos = avg_attn.argmax(dim=-1)  # (B, T)
        query_pos = torch.arange(T, device=attn_w_per_token.device).unsqueeze(0).expand(B, -1)
        dist_to_dominant = query_pos - dominant_pos  # (B, T), non-negative by causality

        # Max attention weight
        max_attn = avg_attn.max(dim=-1).values  # (B, T)

        # How much attention goes to tokens >10 positions back
        # mask[t, s] = 1 if s < t - 10 (i.e., s is more than 10 before t)
        tpos = torch.arange(T, device=attn_w_per_token.device)
        distant_mask_2d = (tpos.unsqueeze(1) - tpos.unsqueeze(0) > 10).float()  # (T, T)
        distant_attn = (avg_attn * distant_mask_2d.unsqueeze(0)).sum(dim=-1)  # (B, T)

        median_entropy = attn_entropy.median()

        local = (dist_to_dominant <= 3).reshape(-1).cpu().numpy()
        distant = (distant_attn > 0.1).reshape(-1).cpu().numpy()
        distributed = (attn_entropy > median_entropy).reshape(-1).cpu().numpy()
        focused = (max_attn > 0.5).reshape(-1).cpu().numpy()

        return {
            "local_attn": local,
            "distant_attn": distant,
            "distributed_attn": distributed,
            "focused_attn": focused,
            "attn_entropy": attn_entropy.reshape(-1).cpu().numpy(),
            "dist_to_dominant": dist_to_dominant.float().reshape(-1).cpu().numpy(),
        }

    # =============================================
    # Collect per-token data across batches
    # =============================================
    print("Collecting activations...")

    # Scalar accumulators
    all_res_norms = []
    all_cos_sims = []
    all_lm_losses = []
    all_output_entropies = []
    all_block1_contrib_norms = []
    all_tokens_ids = []

    # Attention-derived
    all_attn_entropy = []
    all_dist_to_dominant = []

    # Syntactic category masks
    syn_cats = {k: [] for k in
                ["after_punct", "after_opener", "before_closer", "sentence_start", "mid_sentence"]}
    attn_cats = {k: [] for k in
                 ["local_attn", "distant_attn", "distributed_attn", "focused_attn"]}

    with torch.no_grad():
        for batch_i in range(n_eval_batches):
            if batch_i % 10 == 0:
                print(f"  batch {batch_i}/{n_eval_batches}")

            x, y = get_batch()
            logits, _, intermediates = model(x, y, return_intermediates=True)

            source = intermediates["post_block0"]   # (B, T, C)
            target = intermediates["post_block1"]   # (B, T, C)
            B, T, C = source.size()

            # --- Forward model pass (manual to extract intermediates) ---
            # The checkpoint uses the old single-block API: attributes at top level.
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

            # --- Block1 attention weights (manual) ---
            block1 = model.transformer.h[1]
            b1_h = block1.ln_1(source)
            b1_qkv = block1.attn.c_attn(b1_h)
            b1_q, b1_k, b1_v = b1_qkv.split(block1.attn.n_embd, dim=2)
            b1_q = b1_q.view(B, T, block1.attn.n_head, block1.attn.head_dim).transpose(1, 2)
            b1_k = b1_k.view(B, T, block1.attn.n_head, block1.attn.head_dim).transpose(1, 2)
            b1_v = b1_v.view(B, T, block1.attn.n_head, block1.attn.head_dim).transpose(1, 2)
            b1_att = (b1_q @ b1_k.transpose(-2, -1)) * (
                1.0 / math.sqrt(block1.attn.head_dim)
            )
            b1_att = b1_att.masked_fill(
                block1.attn.bias[:, :, :T, :T] == 0, float("-inf")
            )
            b1_att_w = F.softmax(b1_att, dim=-1)  # (B, n_head, T, T)

            # --- Scalar metrics ---
            residual = target - fwd_output
            res_norm = residual.norm(dim=-1)              # (B, T)
            cos = F.cosine_similarity(fwd_output, target, dim=-1)  # (B, T)
            block1_contrib = (target - source).norm(dim=-1)  # (B, T)

            # Output distribution entropy
            log_probs = F.log_softmax(logits, dim=-1)    # (B, T, V)
            probs = log_probs.exp()
            output_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

            # Per-position LM loss (shifted: loss[t] = -logp(x_{t+1}|x_{0:t}))
            tgt = x[:, 1:]  # (B, T-1)
            per_pos_loss = -log_probs[:, :-1].gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

            # Pad last position with NaN sentinel so shapes stay (B, T)
            nan_col = torch.full((B, 1), float('nan'), device=device)
            per_pos_loss = torch.cat([per_pos_loss, nan_col], dim=1)  # (B, T)

            # --- Attention categories ---
            # b1_att_w: (B, n_head, T, T) → rearrange to (B, T, n_head, T)
            b1_attn_per_token = b1_att_w.permute(0, 2, 1, 3)  # (B, T, n_head, T)
            attn_info = compute_attention_categories(b1_attn_per_token)

            # --- Syntactic categories ---
            x_np = x.cpu().numpy()
            syn_info = classify_tokens_syntactic(x_np)

            # --- Accumulate ---
            all_res_norms.append(res_norm.cpu())
            all_cos_sims.append(cos.cpu())
            all_lm_losses.append(per_pos_loss.cpu())
            all_output_entropies.append(output_entropy.cpu())
            all_block1_contrib_norms.append(block1_contrib.cpu())
            all_tokens_ids.append(x.cpu())
            all_attn_entropy.append(attn_info["attn_entropy"])
            all_dist_to_dominant.append(attn_info["dist_to_dominant"])

            for k in attn_cats:
                attn_cats[k].append(attn_info[k])
            for k in syn_cats:
                syn_cats[k].append(syn_info[k])

    # =============================================
    # Assemble flat arrays
    # =============================================
    flat_res = torch.cat(all_res_norms, dim=0).reshape(-1).numpy()
    flat_cos = torch.cat(all_cos_sims, dim=0).reshape(-1).numpy()
    flat_lm = torch.cat(all_lm_losses, dim=0).reshape(-1).numpy()
    flat_ent = torch.cat(all_output_entropies, dim=0).reshape(-1).numpy()
    flat_b1c = torch.cat(all_block1_contrib_norms, dim=0).reshape(-1).numpy()
    flat_attn_ent = np.concatenate(all_attn_entropy)
    flat_dist_dom = np.concatenate(all_dist_to_dominant)

    for k in attn_cats:
        attn_cats[k] = np.concatenate(attn_cats[k])
    for k in syn_cats:
        syn_cats[k] = np.concatenate(syn_cats[k])

    # Valid mask: exclude NaN LM positions (last in each sequence)
    valid = np.isfinite(flat_lm) & np.isfinite(flat_res)
    N = valid.sum()
    print(f"\nTotal valid tokens: {N:,}")

    def category_stats(mask, name=""):
        """Compute residual statistics for a boolean mask."""
        m = mask & valid
        n = int(m.sum())
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0, "cohen_d": 0.0,
                    "mean_cos": 0.0}
        vals = flat_res[m]
        mean_all = flat_res[valid].mean()
        std_all = flat_res[valid].std()
        cohen_d = float((vals.mean() - mean_all) / (std_all + 1e-10))
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "count": n,
            "cohen_d": cohen_d,
            "mean_cos": float(flat_cos[m].mean()),
        }

    # =============================================
    # A. Attention-based categories
    # =============================================
    print("\n=== A. Attention categories ===")
    attn_results = {}
    for cat_name, label in [
        ("local_attn", "Local attention (dom. within 3)"),
        ("distant_attn", "Distant attention (>10 pos back)"),
        ("distributed_attn", "Distributed attention (high entropy)"),
        ("focused_attn", "Focused attention (max>0.5)"),
    ]:
        s = category_stats(attn_cats[cat_name], cat_name)
        attn_results[cat_name] = s
        print(f"  {label:45s}: res={s['mean']:.4f}±{s['std']:.4f}  "
              f"d={s['cohen_d']:+.3f}  n={s['count']:,}")

    # =============================================
    # B. Syntactic context categories
    # =============================================
    print("\n=== B. Syntactic categories ===")
    syn_results = {}
    for cat_name, label in [
        ("after_punct", "After punctuation"),
        ("after_opener", "After opener"),
        ("before_closer", "Before closer"),
        ("sentence_start", "Sentence start"),
        ("mid_sentence", "Mid-sentence"),
    ]:
        s = category_stats(syn_cats[cat_name], cat_name)
        syn_results[cat_name] = s
        print(f"  {label:45s}: res={s['mean']:.4f}±{s['std']:.4f}  "
              f"d={s['cohen_d']:+.3f}  n={s['count']:,}")

    # =============================================
    # C. Prediction difficulty categories
    # =============================================
    print("\n=== C. Prediction difficulty ===")
    lm_valid = flat_lm[valid]
    ent_valid = flat_ent[valid]
    q25_lm = float(np.percentile(lm_valid, 25))
    q75_lm = float(np.percentile(lm_valid, 75))
    med_ent = float(np.median(ent_valid))

    diff_results = {}
    for mask_fn, cat_name, label in [
        (lambda: flat_lm < q25_lm, "easy_pred", "Easy prediction (loss<Q25)"),
        (lambda: flat_lm > q75_lm, "hard_pred", "Hard prediction (loss>Q75)"),
        (lambda: flat_ent > med_ent, "high_entropy", "High output entropy"),
        (lambda: flat_ent <= med_ent, "low_entropy", "Low output entropy"),
    ]:
        s = category_stats(mask_fn(), cat_name)
        diff_results[cat_name] = s
        print(f"  {label:45s}: res={s['mean']:.4f}±{s['std']:.4f}  "
              f"d={s['cohen_d']:+.3f}  n={s['count']:,}")

    # =============================================
    # D. Context integration categories
    # =============================================
    print("\n=== D. Context integration ===")

    # Unigram baseline: compute unigram loss for each actual next token
    # flat_tokens[i] predicts flat_tokens[i+1] nominally, but we stored x not y.
    # Use per-pos LM loss vs unigram loss of the actual next token.
    # Reconstruct: next_token[i] = flat_tokens[i+1] within each sequence.
    # We'll approximate: since flat_lm[i] = -log p_model(x_{t+1}|x_{0:t}),
    # and unigram loss = -log p_unigram(x_{t+1}), we need x_{t+1}.
    # flat_tokens is (B*T,), but sequences are contiguous in T dimension.
    # Reshape to get next tokens properly.
    tids_2d = torch.cat(all_tokens_ids, dim=0)  # (n_batches*B, T)
    next_tids = tids_2d[:, 1:].numpy()           # (n_batches*B, T-1)
    pad = np.full((next_tids.shape[0], 1), -1, dtype=np.int64)
    next_tids = np.concatenate([next_tids, pad], axis=1).reshape(-1)  # (N_total,)

    valid_next = valid & (next_tids >= 0)
    unigram_loss = np.full(len(flat_lm), float('nan'))
    unigram_loss[valid_next] = -np.log(
        token_freq.numpy()[next_tids[valid_next]].clip(1e-10)
    )

    # "Local-only sufficient": unigram model loss < 25th percentile of unigram losses
    uniqloss_valid = unigram_loss[valid_next]
    q25_uni = float(np.percentile(uniqloss_valid, 25))
    q75_uni = float(np.percentile(uniqloss_valid, 75))

    # "Context-dependent": model improves substantially over unigram
    # i.e., unigram_loss - lm_loss > median of that difference
    lm_improvement = unigram_loss - flat_lm
    med_improvement = float(np.nanmedian(lm_improvement[valid_next]))

    ctx_results = {}
    for mask_fn, cat_name, label in [
        (lambda: valid_next & (unigram_loss < q25_uni),
         "local_sufficient", "Local-only sufficient (easy unigram)"),
        (lambda: valid_next & (lm_improvement > med_improvement),
         "context_dependent", "Context-dependent (model >> unigram)"),
        (lambda: valid_next & (unigram_loss > q75_uni),
         "hard_unigram", "Hard unigram (rare next token)"),
    ]:
        s = category_stats(mask_fn(), cat_name)
        ctx_results[cat_name] = s
        print(f"  {label:45s}: res={s['mean']:.4f}±{s['std']:.4f}  "
              f"d={s['cohen_d']:+.3f}  n={s['count']:,}")

    # =============================================
    # E. Block1 contribution magnitude buckets
    # =============================================
    print("\n=== E. Block1 contribution magnitude ===")
    b1c_valid = flat_b1c[valid]
    b1c_q25 = float(np.percentile(b1c_valid, 25))
    b1c_q50 = float(np.percentile(b1c_valid, 50))
    b1c_q75 = float(np.percentile(b1c_valid, 75))

    b1c_results = {}
    for mask_fn, cat_name, label in [
        (lambda: flat_b1c < b1c_q25, "b1c_low", "Block1 contrib Q1 (small)"),
        (lambda: (flat_b1c >= b1c_q25) & (flat_b1c < b1c_q50), "b1c_q2", "Block1 contrib Q2"),
        (lambda: (flat_b1c >= b1c_q50) & (flat_b1c < b1c_q75), "b1c_q3", "Block1 contrib Q3"),
        (lambda: flat_b1c >= b1c_q75, "b1c_high", "Block1 contrib Q4 (large)"),
    ]:
        s = category_stats(mask_fn(), cat_name)
        b1c_results[cat_name] = s
        print(f"  {label:45s}: res={s['mean']:.4f}±{s['std']:.4f}  "
              f"d={s['cohen_d']:+.3f}  n={s['count']:,}")

    # =============================================
    # F. Key correlation analysis
    # =============================================
    print("\n=== F. Correlations ===")
    v = valid
    r_attn_ent = float(np.corrcoef(flat_attn_ent[v], flat_res[v])[0, 1])
    r_dist_dom = float(np.corrcoef(flat_dist_dom[v], flat_res[v])[0, 1])
    r_b1c = float(np.corrcoef(flat_b1c[v], flat_res[v])[0, 1])
    r_lm = float(np.corrcoef(flat_lm[v], flat_res[v])[0, 1])
    r_ent = float(np.corrcoef(flat_ent[v], flat_res[v])[0, 1])
    print(f"  r(attn entropy, residual norm):         {r_attn_ent:+.4f}")
    print(f"  r(dist to dominant token, residual):    {r_dist_dom:+.4f}")
    print(f"  r(block1 contrib norm, residual):       {r_b1c:+.4f}")
    print(f"  r(LM loss, residual):                   {r_lm:+.4f}")
    print(f"  r(output entropy, residual):            {r_ent:+.4f}")

    # Check: is residual ~ block1 contribution, or does it capture specific types?
    # Partial: residual norm / block1_contrib (ratio)
    ratio = flat_res[v] / (flat_b1c[v] + 1e-8)
    r_ratio_attn_ent = float(np.corrcoef(flat_attn_ent[v], ratio)[0, 1])
    r_ratio_lm = float(np.corrcoef(flat_lm[v], ratio)[0, 1])
    print(f"  r(attn entropy, residual/block1_contrib):  {r_ratio_attn_ent:+.4f}")
    print(f"  r(LM loss, residual/block1_contrib):       {r_ratio_lm:+.4f}")

    # =============================================
    # Generate figure
    # =============================================
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle(
        "Behavior-Conditioned Residual Analysis\n"
        f"(post_block0→post_block1 forward model residual, {n_eval_batches} eval batches)",
        fontsize=13, y=0.998,
    )

    def plot_category_bars(ax, results_dict, title, color='steelblue', rotation=30):
        labels = list(results_dict.keys())
        means = [results_dict[k]["mean"] for k in labels]
        stds = [results_dict[k]["std"] for k in labels]
        counts = [results_dict[k]["count"] for k in labels]
        sems = [s / max(1, c ** 0.5) for s, c in zip(stds, counts)]
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, means, color=color, alpha=0.8)
        ax.errorbar(x_pos, means, yerr=sems, fmt='none', color='black', capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=rotation, fontsize=8)
        ax.set_ylabel('Mean residual norm')
        ax.set_title(title)
        for bar, d in zip(bars, [results_dict[k]["cohen_d"] for k in labels]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(sems) * 0.3,
                    f'd={d:+.2f}', ha='center', fontsize=7, rotation=0)

    def plot_cohen_d_bars(ax, results_dict, title, color='coral'):
        labels = list(results_dict.keys())
        ds = [results_dict[k]["cohen_d"] for k in labels]
        x_pos = range(len(labels))
        colors = ['steelblue' if d >= 0 else 'coral' for d in ds]
        ax.bar(x_pos, ds, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_ylabel("Cohen's d")
        ax.set_title(title)

    # Row 0: Attention categories
    plot_category_bars(axes[0, 0], attn_results, "A. Attention-based categories",
                       color='steelblue')
    plot_cohen_d_bars(axes[0, 1], attn_results, "A. Effect sizes (attention)")

    ax = axes[0, 2]
    ax.scatter(flat_attn_ent[valid][::20], flat_res[valid][::20],
               alpha=0.05, s=2, color='steelblue', rasterized=True)
    ax.set_xlabel("Block1 attention entropy")
    ax.set_ylabel("Residual norm")
    ax.set_title(f"Residual vs attn entropy\nr={r_attn_ent:+.4f}")

    # Row 1: Syntactic + difficulty
    plot_category_bars(axes[1, 0], syn_results, "B. Syntactic categories",
                       color='forestgreen')
    plot_cohen_d_bars(axes[1, 1], syn_results, "B. Effect sizes (syntactic)")

    plot_category_bars(axes[1, 2], diff_results, "C. Prediction difficulty",
                       color='darkorange')

    # Row 2: Context integration + block1 contribution
    plot_category_bars(axes[2, 0], ctx_results, "D. Context integration",
                       color='mediumpurple')

    plot_category_bars(axes[2, 1], b1c_results, "E. Block1 contrib quartiles",
                       color='crimson')

    ax = axes[2, 2]
    ax.scatter(flat_b1c[valid][::20], flat_res[valid][::20],
               alpha=0.05, s=2, color='crimson', rasterized=True)
    ax.set_xlabel("Block1 contribution norm")
    ax.set_ylabel("Residual norm")
    ax.set_title(f"Residual vs block1 contrib\nr={r_b1c:+.4f}")

    # Row 3: Correlation summary + scatter plots
    ax = axes[3, 0]
    corr_labels = [
        "attn entropy", "dist dominant", "block1 contrib", "LM loss", "output entropy",
        "ratio×attn ent", "ratio×LM loss",
    ]
    corr_vals = [r_attn_ent, r_dist_dom, r_b1c, r_lm, r_ent,
                 r_ratio_attn_ent, r_ratio_lm]
    colors_c = ['steelblue' if v >= 0 else 'coral' for v in corr_vals]
    ax.barh(range(len(corr_labels)), corr_vals, color=colors_c, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(corr_labels)))
    ax.set_yticklabels(corr_labels, fontsize=9)
    ax.set_xlabel("Pearson r with residual norm")
    ax.set_title("Correlation summary")

    ax = axes[3, 1]
    ax.scatter(flat_dist_dom[valid][::20], flat_res[valid][::20],
               alpha=0.05, s=2, color='darkorange', rasterized=True)
    ax.set_xlabel("Distance to dominant attended token")
    ax.set_ylabel("Residual norm")
    ax.set_title(f"Residual vs attention distance\nr={r_dist_dom:+.4f}")

    ax = axes[3, 2]
    ax.scatter(flat_lm[valid][::20], flat_res[valid][::20],
               alpha=0.05, s=2, color='mediumpurple', rasterized=True)
    ax.set_xlabel("LM loss")
    ax.set_ylabel("Residual norm")
    ax.set_title(f"Residual vs LM loss\nr={r_lm:+.4f}")

    plt.tight_layout()

    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "behavioral_residual.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    # =============================================
    # Save results
    # =============================================
    results = {
        "n_valid_tokens": int(N),
        "A_attention_categories": attn_results,
        "B_syntactic_categories": syn_results,
        "C_prediction_difficulty": diff_results,
        "D_context_integration": ctx_results,
        "E_block1_contribution_quartiles": b1c_results,
        "F_correlations": {
            "attn_entropy_vs_residual": r_attn_ent,
            "dist_dominant_vs_residual": r_dist_dom,
            "block1_contrib_vs_residual": r_b1c,
            "lm_loss_vs_residual": r_lm,
            "output_entropy_vs_residual": r_ent,
            "ratio_attn_entropy": r_ratio_attn_ent,
            "ratio_lm_loss": r_ratio_lm,
        },
        "thresholds": {
            "lm_q25": q25_lm,
            "lm_q75": q75_lm,
            "median_output_entropy": med_ent,
            "unigram_loss_q25": q25_uni,
            "unigram_loss_q75": q75_uni,
            "median_lm_improvement": med_improvement,
            "block1_contrib_quartiles": [b1c_q25, b1c_q50, b1c_q75],
        },
    }

    results_path = os.path.join(save_dir, "behavioral_residual_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"Results saved to {results_path}")
    return results


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_behavioral_residual.remote(
        n_tokens=n_tokens, block_size=block_size,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A behavioral residual analysis complete:")
    corr = result["F_correlations"]
    print(f"  r(attn entropy, residual):   {corr['attn_entropy_vs_residual']:+.4f}")
    print(f"  r(block1 contrib, residual): {corr['block1_contrib_vs_residual']:+.4f}")
    print(f"  r(LM loss, residual):        {corr['lm_loss_vs_residual']:+.4f}")
    print(f"  r(dist dominant, residual):  {corr['dist_dominant_vs_residual']:+.4f}")
    for section in ["A_attention_categories", "B_syntactic_categories",
                    "C_prediction_difficulty", "D_context_integration"]:
        print(f"  {section}:")
        for cat, stats in result[section].items():
            print(f"    {cat}: mean={stats['mean']:.4f}, d={stats['cohen_d']:+.3f}")
