"""Representational divergence analysis: open-loop vs closed-loop models.

Step 1: CKA between the two models at each layer — where do representations diverge?
Step 2: PCA on activation differences — what directions in activation space changed?
Step 3: Alignment between divergence directions and self-knowledge directions —
        did the closed-loop model reorganize *along the dimensions that encode
        what the forward model misses*?
"""

import json

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
    timeout=7200,
    memory=32768,
)
def a2a_representational_divergence(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    n_eval_batches: int = 40,
    probe_steps: int = 500,
):
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
    print(f"REPRESENTATIONAL DIVERGENCE ANALYSIS on {device}")

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
    print(f"Loaded {len(data):,} tokens, using {len(val_data):,} for eval")

    # --- Load checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(
        torch.load(os.path.join(ckpt_root, "open_loop", "model.pt"),
                    map_location=device, weights_only=True))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(
        torch.load(os.path.join(ckpt_root, "closed_loop", "model.pt"),
                    map_location=device, weights_only=True))

    fwd_open = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_open.load_state_dict(
        torch.load(os.path.join(ckpt_root, "open_loop", "fwd_model.pt"),
                    map_location=device, weights_only=True))

    fwd_closed = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_closed.load_state_dict(
        torch.load(os.path.join(ckpt_root, "closed_loop", "fwd_model.pt"),
                    map_location=device, weights_only=True))

    model_open.eval()
    model_closed.eval()
    fwd_open.eval()
    fwd_closed.eval()

    # --- Generate eval batches ---
    eval_gen = torch.Generator().manual_seed(seed + 100)
    eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(n_eval_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # =========================================================
    # Collect activations from both models on the same inputs
    # =========================================================
    layer_keys = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    acts_open = {k: [] for k in layer_keys}
    acts_closed = {k: [] for k in layer_keys}
    residual_vecs_open = []
    residual_vecs_closed = []

    print(f"\nCollecting activations from {n_eval_batches} batches...")
    with torch.no_grad():
        for bi, idx in enumerate(eval_indices):
            x, y = make_batch(idx)

            _, _, vi_open = model_open(x, y, return_intermediates=True)
            _, _, vi_closed = model_closed(x, y, return_intermediates=True)

            for k in layer_keys:
                acts_open[k].append(vi_open[k].reshape(-1, n_embd).cpu())
                acts_closed[k].append(vi_closed[k].reshape(-1, n_embd).cpu())

            # Forward model residuals (each model's fwd model on its own acts)
            src_o = vi_open[predict_from]
            tgt_o = vi_open[predict_to]
            pred_o = fwd_open(src_o)
            residual_vecs_open.append((tgt_o - pred_o).reshape(-1, n_embd).cpu())

            src_c = vi_closed[predict_from]
            tgt_c = vi_closed[predict_to]
            pred_c = fwd_closed(src_c)
            residual_vecs_closed.append((tgt_c - pred_c).reshape(-1, n_embd).cpu())

            if bi % 10 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    acts_open = {k: torch.cat(v).numpy() for k, v in acts_open.items()}
    acts_closed = {k: torch.cat(v).numpy() for k, v in acts_closed.items()}
    residual_vecs_open = torch.cat(residual_vecs_open).numpy()
    residual_vecs_closed = torch.cat(residual_vecs_closed).numpy()

    n_samples = acts_open["post_embed"].shape[0]
    print(f"Total samples: {n_samples}")

    # =========================================================
    # STEP 1: CKA between open-loop and closed-loop at each layer
    # =========================================================
    print(f"\n{'='*60}")
    print("STEP 1: CKA between open-loop and closed-loop models")
    print(f"{'='*60}")

    n_cka = min(8000, n_samples)
    rng = np.random.default_rng(seed)
    cka_idx = rng.choice(n_samples, n_cka, replace=False)

    cka_results = {}
    for k in layer_keys:
        cka_val = linear_cka(acts_open[k][cka_idx], acts_closed[k][cka_idx])
        cka_results[k] = cka_val
        print(f"  {k:>15s}: CKA = {cka_val:.6f}")

    # =========================================================
    # STEP 2: PCA on activation differences
    # =========================================================
    print(f"\n{'='*60}")
    print("STEP 2: PCA on activation differences (closed - open)")
    print(f"{'='*60}")

    pca_results = {}
    diff_pcs = {}

    for k in layer_keys:
        diff = acts_closed[k] - acts_open[k]

        diff_centered = diff - diff.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)

        total_var = (S ** 2).sum()
        cumvar = np.cumsum(S ** 2) / total_var
        frac_explained = (S ** 2) / total_var

        # Effective rank (exponential of entropy of normalized singular values)
        p = (S ** 2) / total_var
        p = p[p > 1e-12]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))

        # Rank thresholds
        rank_50 = int(np.searchsorted(cumvar, 0.50) + 1)
        rank_75 = int(np.searchsorted(cumvar, 0.75) + 1)
        rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)

        # Mean difference norm
        mean_diff_norm = float(np.linalg.norm(diff, axis=1).mean())

        # Mean activation norms for scale reference
        mean_open_norm = float(np.linalg.norm(acts_open[k], axis=1).mean())
        mean_closed_norm = float(np.linalg.norm(acts_closed[k], axis=1).mean())

        pca_results[k] = {
            "mean_diff_norm": mean_diff_norm,
            "relative_diff_norm": mean_diff_norm / mean_open_norm,
            "mean_open_norm": mean_open_norm,
            "mean_closed_norm": mean_closed_norm,
            "effective_rank": eff_rank,
            "rank_50": rank_50,
            "rank_75": rank_75,
            "rank_90": rank_90,
            "rank_95": rank_95,
            "top1_frac": float(frac_explained[0]),
            "top5_frac": float(frac_explained[:5].sum()),
            "top10_frac": float(frac_explained[:10].sum()),
            "top20_frac": float(frac_explained[:20].sum()),
        }

        # Store top PCs for step 3
        diff_pcs[k] = {
            "Vt": Vt,  # (256, 256) — rows are PCs
            "S": S,
            "projections": diff_centered @ Vt.T,  # (n_samples, 256)
        }

        print(f"\n  {k}:")
        print(f"    Mean diff norm: {mean_diff_norm:.4f} "
              f"({pca_results[k]['relative_diff_norm']:.4f} relative to open norm)")
        print(f"    Effective rank: {eff_rank:.1f} / {n_embd}")
        print(f"    Rank for 50%: {rank_50}, 75%: {rank_75}, "
              f"90%: {rank_90}, 95%: {rank_95}")
        print(f"    Top-1: {frac_explained[0]:.4f}, top-5: {frac_explained[:5].sum():.4f}, "
              f"top-10: {frac_explained[:10].sum():.4f}, "
              f"top-20: {frac_explained[:20].sum():.4f}")

    # =========================================================
    # STEP 3: Alignment between divergence and self-knowledge
    # =========================================================
    print(f"\n{'='*60}")
    print("STEP 3: Do divergence directions carry self-knowledge?")
    print(f"{'='*60}")

    # 3a: Subspace overlap between divergence PCs and residual-prediction directions
    #     Train a linear probe: acts → residual_vec, extract its weight matrix,
    #     then measure overlap between probe weight's column space and diff PCs.

    alignment_results = {}
    block_layer_keys = [f"post_block{i}" for i in range(n_layer)]

    n_train = int(0.8 * n_samples)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    for k in block_layer_keys:
        print(f"\n  {k}:")

        # --- Train linear probe: closed-loop acts → forward model residual ---
        X_closed = acts_closed[k]
        T_resid = residual_vecs_closed.astype(np.float32)

        mu_x = X_closed[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_closed[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_closed - mu_x) / sd_x

        probe = nn.Linear(n_embd, n_embd).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ttr = torch.from_numpy(T_resid[train_idx]).float().to(device)
        bs = min(4096, n_train)

        for step in range(probe_steps):
            idx_b = torch.randint(n_train, (bs,), device=device)
            pred = probe(Xtr[idx_b])
            loss = F.mse_loss(pred, Ttr[idx_b])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred_test = probe(Xte).cpu().numpy()
        Tte = T_resid[test_idx]
        probe_r2 = float(1.0 - ((pred_test - Tte) ** 2).mean() / Tte.var())
        print(f"    Residual probe R²: {probe_r2:.4f}")

        # Extract probe weight matrix (the "self-knowledge directions")
        W_probe = probe.weight.detach().cpu().numpy()  # (256, 256)
        # SVD of probe weight to get its principal directions
        U_w, S_w, Vt_w = np.linalg.svd(W_probe, full_matrices=False)

        # Diff PCs for this layer
        Vt_diff = diff_pcs[k]["Vt"]  # rows = diff PCs

        # --- 3b: Subspace overlap at various ranks ---
        # For each rank r, compute overlap between top-r diff PCs and
        # top-r probe directions (right singular vectors of W_probe)
        overlaps = {}
        for r in [5, 10, 20, 50, 128]:
            if r > n_embd:
                continue
            # Top-r diff PCs (rows of Vt_diff[:r])
            V_diff_r = Vt_diff[:r].T  # (256, r)
            # Top-r probe input directions (rows of Vt_w[:r])
            V_probe_r = Vt_w[:r].T  # (256, r)
            # Subspace overlap: mean squared cosine between all pairs
            # = ||V_diff_r.T @ V_probe_r||_F^2 / r
            cross = V_diff_r.T @ V_probe_r
            overlap = float((cross ** 2).sum() / r)
            # Random baseline: r/d for two random r-dimensional subspaces in R^d
            random_baseline = r / n_embd
            overlaps[f"rank_{r}"] = {
                "overlap": overlap,
                "random_baseline": random_baseline,
                "ratio": overlap / random_baseline if random_baseline > 0 else 0,
            }
            print(f"    Subspace overlap (rank {r:3d}): {overlap:.4f} "
                  f"(random baseline: {random_baseline:.4f}, "
                  f"ratio: {overlap / random_baseline:.2f}x)")

        # --- 3c: Per-PC correlation with residual prediction ---
        # For each of the top diff PCs, how much does projecting onto it
        # correlate with the forward model's residual?
        # Project residual vectors onto diff PCs
        resid_centered = T_resid - T_resid.mean(axis=0, keepdims=True)
        resid_proj_on_diff = resid_centered @ Vt_diff.T  # (n, 256)
        resid_total_var = float((resid_centered ** 2).sum())

        # Variance of residual captured by top-k diff PCs
        resid_in_diff = {}
        for topk in [5, 10, 20, 50, 128]:
            if topk > n_embd:
                continue
            captured = float((resid_proj_on_diff[:, :topk] ** 2).sum())
            frac = captured / resid_total_var
            # Random baseline: topk/256
            random_frac = topk / n_embd
            resid_in_diff[f"top_{topk}"] = {
                "frac_residual_var": frac,
                "random_baseline": random_frac,
                "ratio": frac / random_frac if random_frac > 0 else 0,
            }
            print(f"    Residual var in top-{topk:3d} diff PCs: {frac:.4f} "
                  f"(random: {random_frac:.4f}, ratio: {frac / random_frac:.2f}x)")

        # --- 3d: Does the closed-loop model's *extra* R² live in diff directions? ---
        # Train residual probes on BOTH models, project probe weights onto diff PCs
        # Open-loop probe
        X_open = acts_open[k]
        T_resid_open = residual_vecs_open.astype(np.float32)
        mu_xo = X_open[train_idx].mean(axis=0, keepdims=True)
        sd_xo = X_open[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_so = (X_open - mu_xo) / sd_xo

        probe_open = nn.Linear(n_embd, n_embd).to(device)
        opt_o = torch.optim.Adam(probe_open.parameters(), lr=1e-3)
        Xtr_o = torch.from_numpy(X_so[train_idx]).float().to(device)
        Ttr_o = torch.from_numpy(T_resid_open[train_idx]).float().to(device)

        for step in range(probe_steps):
            idx_b = torch.randint(n_train, (bs,), device=device)
            pred = probe_open(Xtr_o[idx_b])
            loss = F.mse_loss(pred, Ttr_o[idx_b])
            opt_o.zero_grad()
            loss.backward()
            opt_o.step()

        with torch.no_grad():
            Xte_o = torch.from_numpy(X_so[test_idx]).float().to(device)
            pred_test_o = probe_open(Xte_o).cpu().numpy()
        Tte_o = T_resid_open[test_idx]
        probe_r2_open = float(1.0 - ((pred_test_o - Tte_o) ** 2).mean() / Tte_o.var())
        print(f"    Open-loop residual probe R²: {probe_r2_open:.4f}")
        print(f"    Closed-loop residual probe R²: {probe_r2:.4f}")
        print(f"    Δ R²: {probe_r2 - probe_r2_open:+.4f}")

        # Compare probe weight matrices
        W_probe_closed = probe.weight.detach().cpu().numpy()
        W_probe_open = probe_open.weight.detach().cpu().numpy()
        W_diff_probe = W_probe_closed - W_probe_open

        # SVD of the probe weight DIFFERENCE — the "extra self-knowledge directions"
        U_wd, S_wd, Vt_wd = np.linalg.svd(W_diff_probe, full_matrices=False)

        # Overlap between extra-self-knowledge directions and diff PCs
        extra_sk_overlaps = {}
        for r in [5, 10, 20, 50]:
            if r > n_embd:
                continue
            V_diff_r = Vt_diff[:r].T
            V_sk_r = Vt_wd[:r].T
            cross = V_diff_r.T @ V_sk_r
            overlap = float((cross ** 2).sum() / r)
            random_baseline = r / n_embd
            extra_sk_overlaps[f"rank_{r}"] = {
                "overlap": overlap,
                "random_baseline": random_baseline,
                "ratio": overlap / random_baseline if random_baseline > 0 else 0,
            }
            print(f"    Extra-SK ∩ diff PCs (rank {r:3d}): {overlap:.4f} "
                  f"(ratio: {overlap / random_baseline:.2f}x)")

        alignment_results[k] = {
            "probe_r2_closed": probe_r2,
            "probe_r2_open": probe_r2_open,
            "delta_r2": probe_r2 - probe_r2_open,
            "subspace_overlaps": overlaps,
            "residual_var_in_diff_pcs": resid_in_diff,
            "extra_selfknowledge_overlaps": extra_sk_overlaps,
        }

    # =========================================================
    # STEP 3e: Direct test — does projecting closed-loop acts onto
    # diff PCs predict the residual better than random directions?
    # =========================================================
    print(f"\n{'='*60}")
    print("STEP 3e: Residual prediction from diff-PC-projected acts")
    print(f"{'='*60}")

    projection_probe_results = {}

    for k in block_layer_keys:
        print(f"\n  {k}:")
        Vt_diff = diff_pcs[k]["Vt"]

        X_closed_s = (acts_closed[k] - acts_closed[k][train_idx].mean(axis=0, keepdims=True)) / \
                     (acts_closed[k][train_idx].std(axis=0, keepdims=True) + 1e-6)
        T_resid = residual_vecs_closed.astype(np.float32)

        results_by_rank = {}
        for r in [10, 20, 50, 128, 256]:
            if r > n_embd:
                continue

            # Project onto top-r diff PCs
            X_proj = X_closed_s @ Vt_diff[:r].T  # (n, r)

            probe_r_diff = nn.Linear(r, n_embd).to(device)
            opt_r = torch.optim.Adam(probe_r_diff.parameters(), lr=1e-3)
            Xtr_r = torch.from_numpy(X_proj[train_idx].astype(np.float32)).to(device)
            Ttr_r = torch.from_numpy(T_resid[train_idx]).to(device)

            for step in range(probe_steps):
                idx_b = torch.randint(n_train, (bs,), device=device)
                pred = probe_r_diff(Xtr_r[idx_b])
                loss = F.mse_loss(pred, Ttr_r[idx_b])
                opt_r.zero_grad()
                loss.backward()
                opt_r.step()

            with torch.no_grad():
                Xte_r = torch.from_numpy(X_proj[test_idx].astype(np.float32)).to(device)
                pred_r = probe_r_diff(Xte_r).cpu().numpy()
            Tte_r = T_resid[test_idx]
            r2_diff = float(1.0 - ((pred_r - Tte_r) ** 2).mean() / Tte_r.var())

            # Compare against random r directions
            rand_dirs = rng.standard_normal((n_embd, r)).astype(np.float32)
            rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
            X_rand = X_closed_s @ rand_dirs

            probe_r_rand = nn.Linear(r, n_embd).to(device)
            opt_rr = torch.optim.Adam(probe_r_rand.parameters(), lr=1e-3)
            Xtr_rr = torch.from_numpy(X_rand[train_idx].astype(np.float32)).to(device)

            for step in range(probe_steps):
                idx_b = torch.randint(n_train, (bs,), device=device)
                pred = probe_r_rand(Xtr_rr[idx_b])
                loss = F.mse_loss(pred, Ttr_r[idx_b])
                opt_rr.zero_grad()
                loss.backward()
                opt_rr.step()

            with torch.no_grad():
                Xte_rr = torch.from_numpy(X_rand[test_idx].astype(np.float32)).to(device)
                pred_rr = probe_r_rand(Xte_rr).cpu().numpy()
            r2_rand = float(1.0 - ((pred_rr - Tte_r) ** 2).mean() / Tte_r.var())

            results_by_rank[f"rank_{r}"] = {
                "r2_diff_pcs": r2_diff,
                "r2_random_dirs": r2_rand,
                "delta": r2_diff - r2_rand,
            }
            print(f"    rank {r:3d}: R²(diff PCs)={r2_diff:.4f}, "
                  f"R²(random)={r2_rand:.4f}, "
                  f"Δ={r2_diff - r2_rand:+.4f}")

        projection_probe_results[k] = results_by_rank

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nStep 1 — CKA (1.0 = identical, lower = more divergent):")
    for k in layer_keys:
        marker = ""
        if k == f"post_block{inject_after_block}":
            marker = " ← injection point"
        print(f"  {k:>15s}: {cka_results[k]:.6f}{marker}")

    print("\nStep 2 — Diff structure (lower effective rank = more concentrated change):")
    for k in layer_keys:
        p = pca_results[k]
        print(f"  {k:>15s}: eff_rank={p['effective_rank']:.1f}, "
              f"top-10={p['top10_frac']:.3f}, "
              f"rel_norm={p['relative_diff_norm']:.4f}")

    print("\nStep 3 — Alignment (ratio > 1 = divergence directions carry self-knowledge):")
    for k in block_layer_keys:
        a = alignment_results[k]
        r10 = a["residual_var_in_diff_pcs"].get("top_10", {})
        r20 = a["residual_var_in_diff_pcs"].get("top_20", {})
        sk10 = a["extra_selfknowledge_overlaps"].get("rank_10", {})
        print(f"  {k:>15s}: resid-in-diff ratio (top-10)={r10.get('ratio', 0):.2f}x, "
              f"(top-20)={r20.get('ratio', 0):.2f}x, "
              f"extra-SK overlap (rank-10)={sk10.get('ratio', 0):.2f}x")

    print("\nStep 3e — Projection probes (diff PCs vs random directions):")
    for k in block_layer_keys:
        for rk, rv in projection_probe_results[k].items():
            if rk == "rank_20":
                print(f"  {k:>15s} ({rk}): diff={rv['r2_diff_pcs']:.4f}, "
                      f"rand={rv['r2_random_dirs']:.4f}, "
                      f"Δ={rv['delta']:+.4f}")

    # =========================================================
    # Save results
    # =========================================================
    save_dir = os.path.join(ckpt_root, "representational_divergence")
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "n_tokens": n_tokens, "n_eval_batches": n_eval_batches,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "seed": seed,
        },
        "step1_cka": cka_results,
        "step2_pca": pca_results,
        "step3_alignment": alignment_results,
        "step3e_projection_probes": projection_probe_results,
    }

    from a2a_forward.shared import NumpyEncoder
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {save_dir}")
    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_representational_divergence.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A representational divergence analysis complete:")
    print("\n  CKA (open vs closed):")
    for k, v in result["step1_cka"].items():
        print(f"    {k:>15s}: {v:.6f}")
    print("\n  Diff PCA:")
    for k, v in result["step2_pca"].items():
        print(f"    {k:>15s}: eff_rank={v['effective_rank']:.1f}, "
              f"top10={v['top10_frac']:.3f}")
    print("\n  Alignment (residual var in diff PCs):")
    for k, v in result["step3_alignment"].items():
        r10 = v["residual_var_in_diff_pcs"].get("top_10", {})
        print(f"    {k:>15s}: Δ R²={v['delta_r2']:+.4f}, "
              f"ratio(top10)={r10.get('ratio', 0):.2f}x")
