"""Language model computational property geometry probes.

Adapts the MNIST geometry experiment (mnist_geometry.py) to language models.
Tests whether CL and post-distillation GPT models develop more compositionally
structured representations of computational properties.

Three conditions: OL, CL (no injection at eval), Distilled
Probes at per-position activations (128 positions per sequence)

Properties (language analogs of MNIST digit/residual/confidence):
  - Token frequency quartile (analog of digit identity — data property)
  - FM residual norm (computational property)
  - FM residual direction cluster (computational property, categorical)
  - Block1 contribution norm (computational property)
  - Per-token LM loss (difficulty property)

Same 6 tests as MNIST:
  1. Probe accuracy
  2. Probe direction orthogonality
  3. Compositionality (centroid additivity)
  4. Vector arithmetic (analogy completion)
  5. Digit-computation independence (freq vs computation independence)
  6. Subspace overlap
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_language_geometry(
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
    probe_batches: int = 40,
    probe_steps: int = 500,
    n_residual_clusters: int = 8,
    n_freq_bins: int = 4,
    seed: int = 42,
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"LANGUAGE COMPUTATIONAL GEOMETRY PROBES on {device}")

    # --- Load data ---
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(
        os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
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

    # Compute token frequencies from training data
    unique, counts = torch.unique(train_data, return_counts=True)
    token_freq = torch.zeros(vocab_size)
    token_freq[unique] = counts.float()
    token_freq = token_freq / token_freq.sum()
    print(f"Token freq range: [{token_freq[token_freq > 0].min():.2e}, "
          f"{token_freq.max():.2e}]")

    # Pre-generate probe indices (same seed+2 as controlled_retrain.py)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # --- Load checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ctrl_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    dist_root = (f"{DATA_DIR}/a2a_forward/distillation/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    def make_gpt():
        return GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=block_size,
        ).to(device)

    ol_model = make_gpt()
    ol_model.load_state_dict(
        torch.load(f"{ctrl_root}/open_loop/model.pt", map_location=device,
                    weights_only=True))
    ol_fm = make_fm()
    ol_fm.load_state_dict(
        torch.load(f"{ctrl_root}/open_loop/fwd_model.pt", map_location=device,
                    weights_only=True))

    cl_model = make_gpt()
    cl_model.load_state_dict(
        torch.load(f"{ctrl_root}/closed_loop/model.pt", map_location=device,
                    weights_only=True))
    cl_fm = make_fm()
    cl_fm.load_state_dict(
        torch.load(f"{ctrl_root}/closed_loop/fwd_model.pt",
                    map_location=device, weights_only=True))

    dist_model = make_gpt()
    dist_model.load_state_dict(
        torch.load(f"{dist_root}/distilled/model.pt", map_location=device,
                    weights_only=True))
    dist_fm = make_fm()
    dist_fm.load_state_dict(
        torch.load(f"{dist_root}/distilled/fresh_fm.pt", map_location=device,
                    weights_only=True))

    print("Loaded all checkpoints")

    models = [
        ("OL", ol_model, ol_fm),
        ("CL", cl_model, cl_fm),
        ("Distilled", dist_model, dist_fm),
    ]

    # =============================================
    # Collect per-position activations and properties
    # =============================================
    analysis_layers = ["post_block0", "post_block3"]

    all_data = {}
    for name, model, fm in models:
        model.eval()
        fm.eval()
        print(f"\nCollecting data for {name}...")

        layer_acts = {k: [] for k in analysis_layers}
        token_freqs = []
        fm_residual_norms = []
        fm_residual_vecs = []
        block_contrib_norms = []
        per_token_losses = []

        with torch.no_grad():
            for pidx in probe_indices:
                x, y = make_batch(pidx)
                logits, _, intermediates = model(
                    x, y, return_intermediates=True)

                # Per-token LM loss
                log_probs = F.log_softmax(logits, dim=-1)
                per_tok_loss = -log_probs.gather(
                    2, y.unsqueeze(-1)).squeeze(-1)
                per_token_losses.append(per_tok_loss.cpu())

                # Token frequencies for input tokens
                freqs = token_freq[x.cpu()]
                token_freqs.append(freqs)

                # FM residual
                src = intermediates[predict_from]
                tgt = intermediates[predict_to]
                pred = fm(src)
                residual = tgt - pred

                fm_residual_norms.append(
                    residual.norm(dim=-1).cpu())
                # Subsample residual vectors to save memory:
                # take every 4th position
                fm_residual_vecs.append(
                    residual[:, ::4, :].cpu())

                # Block1 contribution
                b0 = intermediates["post_block0"]
                b1 = intermediates["post_block1"]
                block_contrib_norms.append(
                    (b1 - b0).norm(dim=-1).cpu())

                # Layer activations (subsample positions for memory)
                for k in analysis_layers:
                    layer_acts[k].append(
                        intermediates[k][:, ::4, :].cpu())

        # Flatten: (probe_batches * batch_size * positions_per_sample)
        # For subsampled: positions_per_sample = block_size // 4
        sub_positions = block_size // 4

        # Full position data (for properties that weren't subsampled)
        token_freqs_full = torch.cat(token_freqs).reshape(-1)
        fm_res_norms_full = torch.cat(fm_residual_norms).reshape(-1)
        block_contrib_full = torch.cat(block_contrib_norms).reshape(-1)
        losses_full = torch.cat(per_token_losses).reshape(-1)

        # Subsampled data (matching the activation subsampling)
        token_freqs_sub = torch.cat(token_freqs)[:, ::4].reshape(-1)
        fm_res_norms_sub = torch.cat(fm_residual_norms)[:, ::4].reshape(-1)
        block_contrib_sub = torch.cat(block_contrib_norms)[:, ::4].reshape(-1)
        losses_sub = torch.cat(per_token_losses)[:, ::4].reshape(-1)
        fm_res_vecs = torch.cat(fm_residual_vecs).reshape(-1, n_embd)

        all_data[name] = {
            "token_freqs": token_freqs_sub,
            "fm_res_norms": fm_res_norms_sub,
            "fm_res_vecs": fm_res_vecs,
            "block_contrib": block_contrib_sub,
            "per_token_loss": losses_sub,
            "layer_acts": {
                k: torch.cat(v).reshape(-1, n_embd)
                for k, v in layer_acts.items()
            },
        }
        n_pos = all_data[name]["token_freqs"].shape[0]
        print(f"  {name}: {n_pos} position-samples "
              f"(from {probe_batches}×{batch_size}×{sub_positions})")

    # --- Discretize token frequency into bins ---
    for name in all_data:
        freqs = all_data[name]["token_freqs"].numpy()
        freq_percentiles = np.percentile(
            freqs[freqs > 0],
            np.linspace(0, 100, n_freq_bins + 1)[1:-1])
        all_data[name]["freq_bin"] = torch.tensor(
            np.digitize(freqs, freq_percentiles))

    # --- Cluster FM residual directions ---
    print(f"\n{'='*60}")
    print("  CLUSTERING FM RESIDUALS")
    print(f"{'='*60}")

    from scipy.cluster.vq import kmeans2

    for name in all_data:
        vecs = all_data[name]["fm_res_vecs"].numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = vecs / norms
        # Subsample for clustering speed
        n_for_cluster = min(50000, len(normed))
        rng = np.random.default_rng(seed)
        cluster_idx = rng.choice(len(normed), n_for_cluster, replace=False)
        centroids, _ = kmeans2(
            normed[cluster_idx], n_residual_clusters, minit="++", seed=seed)
        # Assign all points
        dists = np.linalg.norm(
            normed[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        all_data[name]["res_clusters"] = torch.tensor(labels)
        sizes = [int((labels == i).sum()) for i in range(n_residual_clusters)]
        print(f"  {name}: cluster sizes = {sizes}")

    # =============================================
    # Probe infrastructure
    # =============================================
    n_samples = all_data["OL"]["token_freqs"].shape[0]
    n_train = int(0.8 * n_samples)
    perm = torch.randperm(
        n_samples, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train].numpy()
    test_idx = perm[n_train:].numpy()

    def train_regression_probe(X_np, y_np):
        mu = X_np[train_idx].mean(axis=0, keepdims=True)
        sd = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu) / sd

        y_mu = y_np[train_idx].mean()
        y_sd = y_np[train_idx].std() + 1e-6
        y_s = (y_np - y_mu) / y_sd

        probe = nn.Linear(X_np.shape[1], 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ytr = torch.from_numpy(y_s[train_idx]).float().to(device).unsqueeze(1)
        bs = min(4096, len(train_idx))

        for _ in range(probe_steps):
            si = torch.randint(len(train_idx), (bs,), device=device)
            loss = F.mse_loss(probe(Xtr[si]), Ytr[si])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).cpu().numpy().squeeze()
        y_te = y_s[test_idx]
        mse = ((pred - y_te) ** 2).mean()
        var = y_te.var()
        r2 = float(1.0 - mse / var) if var > 0 else 0.0

        w = probe.weight.data.cpu().numpy().squeeze() / sd.squeeze()
        w = w / (np.linalg.norm(w) + 1e-8)
        return r2, w

    def train_classification_probe(X_np, y_np, n_classes):
        mu = X_np[train_idx].mean(axis=0, keepdims=True)
        sd = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu) / sd

        probe = nn.Linear(X_np.shape[1], n_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ytr = torch.from_numpy(y_np[train_idx]).long().to(device)
        bs = min(4096, len(train_idx))

        for _ in range(probe_steps):
            si = torch.randint(len(train_idx), (bs,), device=device)
            loss = F.cross_entropy(probe(Xtr[si]), Ytr[si])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            preds = probe(Xte).argmax(dim=-1).cpu().numpy()
        acc = float((preds == y_np[test_idx]).mean())

        W = probe.weight.data.cpu().numpy() / sd.squeeze()
        row_norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / (row_norms + 1e-8)
        return acc, W

    # =============================================
    # TEST 1: PROBE ACCURACY
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 1: PROBE ACCURACY")
    print(f"{'='*60}")

    scalar_properties = [
        "fm_res_norms", "block_contrib", "per_token_loss"]
    scalar_labels = [
        "FM Residual Norm", "Block1 Contribution", "Per-token LM Loss"]

    probe_results = {}
    probe_weights = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        probe_results[name] = {}
        probe_weights[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            probe_results[name][layer] = {}
            probe_weights[name][layer] = {}

            # Frequency bin classification
            acc, W_freq = train_classification_probe(
                acts, data["freq_bin"].numpy(), n_freq_bins)
            probe_results[name][layer]["freq_bin_acc"] = acc
            probe_weights[name][layer]["freq_bin"] = W_freq
            print(f"  {name:10s} | {layer:12s} | freq bin acc = {acc:.4f}")

            # Residual cluster classification
            acc, W_cluster = train_classification_probe(
                acts, data["res_clusters"].numpy(), n_residual_clusters)
            probe_results[name][layer]["cluster_acc"] = acc
            probe_weights[name][layer]["cluster"] = W_cluster
            print(f"  {name:10s} | {layer:12s} | cluster acc = {acc:.4f}")

            # Scalar properties
            for prop, label in zip(scalar_properties, scalar_labels):
                r2, w = train_regression_probe(
                    acts, data[prop].numpy())
                probe_results[name][layer][f"{prop}_r2"] = r2
                probe_weights[name][layer][prop] = w
                print(f"  {name:10s} | {layer:12s} | {label:20s} "
                      f"R² = {r2:.4f}")

    # =============================================
    # TEST 2: PROBE DIRECTION ORTHOGONALITY
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 2: PROBE DIRECTION ORTHOGONALITY")
    print(f"{'='*60}")

    orthogonality_results = {}

    for name in ["OL", "CL", "Distilled"]:
        orthogonality_results[name] = {}

        for layer in analysis_layers:
            weights = probe_weights[name][layer]

            dirs = []
            dir_names = []

            # Principal direction of frequency subspace
            W_freq = weights["freq_bin"]
            _, _, Vt = np.linalg.svd(W_freq, full_matrices=False)
            dirs.append(Vt[0])
            dir_names.append("freq_PC1")

            # Principal direction of cluster subspace
            W_cluster = weights["cluster"]
            _, _, Vt = np.linalg.svd(W_cluster, full_matrices=False)
            dirs.append(Vt[0])
            dir_names.append("cluster_PC1")

            for prop in scalar_properties:
                dirs.append(weights[prop])
                dir_names.append(prop)

            dirs = np.stack(dirs)
            n_dirs = len(dirs)

            cos_matrix = np.abs(dirs @ dirs.T)
            np.fill_diagonal(cos_matrix, 0)

            n_pairs = n_dirs * (n_dirs - 1) / 2
            mean_abs_cos = cos_matrix.sum() / (2 * n_pairs)

            freq_vs_comp = {}
            for i, dn in enumerate(dir_names):
                if dn == "freq_PC1":
                    for j, dn2 in enumerate(dir_names):
                        if dn2 not in ("freq_PC1", "cluster_PC1"):
                            freq_vs_comp[dn2] = float(cos_matrix[i, j])

            result = {
                "mean_abs_cos": float(mean_abs_cos),
                "cos_matrix": cos_matrix.tolist(),
                "direction_names": dir_names,
                "freq_vs_computational": freq_vs_comp,
            }
            orthogonality_results[name][layer] = result

            print(f"\n  {name:10s} | {layer}")
            print(f"    Mean |cos|: {mean_abs_cos:.4f}")
            for i in range(n_dirs):
                for j in range(i + 1, n_dirs):
                    print(f"      {dir_names[i]:15s} × {dir_names[j]:15s}: "
                          f"|cos| = {cos_matrix[i, j]:.4f}")

    # =============================================
    # TEST 3: COMPOSITIONALITY (CENTROID ADDITIVITY)
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 3: COMPOSITIONALITY (CENTROID ADDITIVITY)")
    print(f"{'='*60}")

    def compositionality_r2(acts, factor_a, factor_b, n_a, n_b, min_count=20):
        grand_mean = acts.mean(axis=0)

        a_centroids = {}
        for a in range(n_a):
            mask = factor_a == a
            if mask.sum() >= min_count:
                a_centroids[a] = acts[mask].mean(axis=0)

        b_centroids = {}
        for b in range(n_b):
            mask = factor_b == b
            if mask.sum() >= min_count:
                b_centroids[b] = acts[mask].mean(axis=0)

        actual_list, pred_list, count_list = [], [], []
        for a in range(n_a):
            for b in range(n_b):
                mask = (factor_a == a) & (factor_b == b)
                count = mask.sum()
                if (count >= min_count
                        and a in a_centroids and b in b_centroids):
                    actual_list.append(acts[mask].mean(axis=0))
                    pred_list.append(
                        a_centroids[a] + b_centroids[b] - grand_mean)
                    count_list.append(count)

        if len(actual_list) < 4:
            return {"r2": float("nan"), "n_cells": len(actual_list)}

        actual = np.stack(actual_list)
        predicted = np.stack(pred_list)
        counts = np.array(count_list, dtype=float)
        weights = counts / counts.sum()

        residuals = actual - predicted
        deviations = actual - grand_mean
        ss_res = np.sum(weights[:, None] * residuals ** 2)
        ss_tot = np.sum(weights[:, None] * deviations ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"r2": r2, "n_cells": len(actual_list),
                "n_samples": int(counts.sum())}

    compositionality_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        freq_bins = data["freq_bin"].numpy()

        res_norms = data["fm_res_norms"].numpy()
        res_quartile = np.digitize(
            res_norms, np.percentile(res_norms, [25, 50, 75]))

        losses = data["per_token_loss"].numpy()
        loss_quartile = np.digitize(
            losses, np.percentile(losses, [25, 50, 75]))

        compositionality_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()

            fr = compositionality_r2(
                acts, freq_bins, res_quartile, n_freq_bins, 4)
            fl = compositionality_r2(
                acts, freq_bins, loss_quartile, n_freq_bins, 4)

            compositionality_results[name][layer] = {
                "freq_x_residual": fr,
                "freq_x_loss": fl,
            }
            print(f"  {name:10s} | {layer}: "
                  f"freq×res R²={fr['r2']:.4f} ({fr['n_cells']} cells)  "
                  f"freq×loss R²={fl['r2']:.4f} ({fl['n_cells']} cells)")

    # =============================================
    # TEST 4: VECTOR ARITHMETIC (ANALOGY COMPLETION)
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 4: VECTOR ARITHMETIC (ANALOGY COMPLETION)")
    print(f"{'='*60}")

    arithmetic_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        freq_bins = data["freq_bin"].numpy()
        res_norms = data["fm_res_norms"].numpy()

        median_res = np.median(res_norms)
        high_res = res_norms > median_res

        arithmetic_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            grand_mean = acts.mean(axis=0)

            centroids = {}
            for fb in range(n_freq_bins):
                for is_high in [True, False]:
                    mask = (freq_bins == fb) & (
                        high_res if is_high else ~high_res)
                    if mask.sum() >= 20:
                        centroids[(fb, is_high)] = acts[mask].mean(axis=0)

            cosines = []
            l2_ratios = []

            for a in range(n_freq_bins):
                for b in range(n_freq_bins):
                    if a == b:
                        continue
                    keys = [(a, True), (a, False), (b, True), (b, False)]
                    if all(k in centroids for k in keys):
                        diff_a = centroids[(a, True)] - centroids[(a, False)]
                        predicted = centroids[(b, False)] + diff_a
                        actual = centroids[(b, True)]

                        pv = predicted - grand_mean
                        av = actual - grand_mean
                        cos = float(np.dot(pv, av) / (
                            np.linalg.norm(pv) * np.linalg.norm(av) + 1e-8))
                        cosines.append(cos)

                        l2 = np.linalg.norm(predicted - actual)
                        baseline = np.linalg.norm(actual - grand_mean)
                        l2_ratios.append(float(l2 / (baseline + 1e-8)))

            result = {
                "mean_cos": float(np.mean(cosines)) if cosines else 0.0,
                "median_cos": float(np.median(cosines)) if cosines else 0.0,
                "mean_l2_ratio": float(np.mean(l2_ratios))
                if l2_ratios else 0.0,
                "n_analogies": len(cosines),
            }
            arithmetic_results[name][layer] = result

            print(f"  {name:10s} | {layer}: "
                  f"mean_cos={result['mean_cos']:.4f}  "
                  f"L2_ratio={result['mean_l2_ratio']:.4f}  "
                  f"(n={result['n_analogies']})")

    # =============================================
    # TEST 5: FREQ-COMPUTATION INDEPENDENCE
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 5: FREQ-COMPUTATION INDEPENDENCE")
    print(f"{'='*60}")

    independence_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        independence_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            freq_bins = data["freq_bin"].numpy()

            freq_centroids = []
            for fb in range(n_freq_bins):
                mask = freq_bins == fb
                if mask.sum() > 20:
                    freq_centroids.append(acts[mask].mean(axis=0))
            freq_centroids = np.stack(freq_centroids)
            freq_centered = freq_centroids - freq_centroids.mean(axis=0)

            _, S, Vt = np.linalg.svd(freq_centered, full_matrices=False)
            n_pcs = min(3, len(S))
            freq_subspace = Vt[:n_pcs]

            weights = probe_weights[name][layer]
            projections = {}
            for prop in scalar_properties:
                w = weights[prop]
                proj = freq_subspace @ w
                projections[prop] = float(np.linalg.norm(proj))

            independence_results[name][layer] = {
                "projections_onto_freq_subspace": projections,
                "freq_var_in_top_pcs": float(S[:n_pcs].sum() / S.sum()),
            }

            print(f"  {name:10s} | {layer}")
            for prop, proj_norm in projections.items():
                print(f"    {prop:20s}: ||proj|| = {proj_norm:.4f}")

    # =============================================
    # TEST 6: SUBSPACE OVERLAP (FREQ vs COMPUTATION)
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 6: SUBSPACE OVERLAP (FREQ vs COMPUTATION)")
    print(f"{'='*60}")

    subspace_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        subspace_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            freq_bins = data["freq_bin"].numpy()
            clusters = data["res_clusters"].numpy()

            f_cents = []
            for fb in range(n_freq_bins):
                mask = freq_bins == fb
                if mask.sum() > 20:
                    f_cents.append(acts[mask].mean(axis=0))
            f_cents = np.stack(f_cents)
            f_cents -= f_cents.mean(axis=0)
            _, _, Vt_f = np.linalg.svd(f_cents, full_matrices=False)

            c_cents = []
            for c in range(n_residual_clusters):
                mask = clusters == c
                if mask.sum() > 20:
                    c_cents.append(acts[mask].mean(axis=0))
            c_cents = np.stack(c_cents)
            c_cents -= c_cents.mean(axis=0)
            _, _, Vt_c = np.linalg.svd(c_cents, full_matrices=False)

            overlaps = {}
            for k in [3]:
                k_f = min(k, Vt_f.shape[0])
                k_c = min(k, Vt_c.shape[0])
                D = Vt_f[:k_f]
                C = Vt_c[:k_c]
                M = D @ C.T
                svals = np.linalg.svd(M, compute_uv=False)
                overlaps[f"top{k}_mean_cos"] = float(svals.mean())

            subspace_results[name][layer] = overlaps

            print(f"  {name:10s} | {layer}: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in overlaps.items()))

    # =============================================
    # Save results
    # =============================================
    save_dir = (f"{DATA_DIR}/a2a_forward/language_geometry/"
                f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "n_tokens": n_tokens, "block_size": block_size,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer,
            "n_residual_clusters": n_residual_clusters,
            "n_freq_bins": n_freq_bins,
            "probe_batches": probe_batches, "probe_steps": probe_steps,
            "analysis_layers": analysis_layers,
            "seed": seed,
        },
        "probe_accuracy": probe_results,
        "orthogonality": orthogonality_results,
        "compositionality": compositionality_results,
        "vector_arithmetic": arithmetic_results,
        "freq_computation_independence": independence_results,
        "subspace_overlap": subspace_results,
    }

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    print("\n  Probe Accuracy (post_block3):")
    print(f"    {'Metric':25s} {'OL':>8s} {'CL':>8s} {'Dist':>8s}")
    metrics = ["freq_bin_acc", "cluster_acc"] + [
        f"{p}_r2" for p in scalar_properties]
    for metric in metrics:
        vals = [probe_results[n]["post_block3"][metric]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {metric:25s} {vals[0]:>8.4f} {vals[1]:>8.4f} "
              f"{vals[2]:>8.4f}")

    print("\n  Direction Orthogonality (mean |cos|, lower = more orthogonal):")
    for layer in analysis_layers:
        vals = [orthogonality_results[n][layer]["mean_abs_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Compositionality (freq × residual quartile, R²):")
    for layer in analysis_layers:
        vals = [compositionality_results[n][layer]["freq_x_residual"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Compositionality (freq × LM loss quartile, R²):")
    for layer in analysis_layers:
        vals = [compositionality_results[n][layer]["freq_x_loss"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Vector Arithmetic (analogy cosine):")
    for layer in analysis_layers:
        vals = [arithmetic_results[n][layer]["mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Subspace Overlap (freq vs cluster, top3 mean cos):")
    for layer in analysis_layers:
        vals = [subspace_results[n][layer]["top3_mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print(f"\n  Saved to {save_dir}")
    return result


@app.local_entrypoint()
def main():
    result = a2a_language_geometry.remote()
    print("\nLanguage geometry probes complete.")

    pr = result["probe_accuracy"]
    print(f"\n  Probe Accuracy (post_block3):")
    print(f"    {'Metric':25s} {'OL':>8s} {'CL':>8s} {'Dist':>8s}")
    for metric in ["freq_bin_acc", "cluster_acc", "fm_res_norms_r2",
                    "block_contrib_r2", "per_token_loss_r2"]:
        vals = [pr[n]["post_block3"][metric]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {metric:25s} {vals[0]:>8.4f} {vals[1]:>8.4f} "
              f"{vals[2]:>8.4f}")

    orth = result["orthogonality"]
    print(f"\n  Orthogonality (mean |cos|):")
    for layer in ["post_block0", "post_block3"]:
        vals = [orth[n][layer]["mean_abs_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    comp = result["compositionality"]
    print(f"\n  Compositionality (R²):")
    for layer in ["post_block0", "post_block3"]:
        vals = [comp[n][layer]["freq_x_residual"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer} freq×res: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    va = result["vector_arithmetic"]
    print(f"\n  Vector Arithmetic (cosine):")
    for layer in ["post_block0", "post_block3"]:
        vals = [va[n][layer]["mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")
