"""MNIST computational property geometry probes.

Tests whether CL and post-distillation models develop more compositionally
structured representations of computational properties compared to OL.

Inspired by classic word embedding geometry (king - man + woman = queen):
tests whether computational state (FM residual, block contribution, confidence)
and data identity (digit) are encoded as orthogonal, additively separable
directions in activation space.

Three conditions: OL, CL (no injection at eval), Distilled
Probes at: post_block0, post_block3

Measurements:
  1. Probe accuracy (R² or classification accuracy) for each property
  2. Probe direction orthogonality (pairwise |cos| between regression weights)
  3. Compositionality (centroid additivity R²: does digit × computation factor?)
  4. Vector arithmetic (analogy completion: consistency of computational
     direction across digits)
  5. Digit-computation independence (projection of computational probe
     weights onto the digit subspace)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist_geometry(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    probe_batches: int = 40,
    probe_steps: int = 500,
    n_residual_clusters: int = 8,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_positions = (28 // patch_size) ** 2 + 1  # 50

    print(f"MNIST COMPUTATIONAL GEOMETRY PROBES on {device}")

    # --- Load MNIST test set ---
    ds = load_dataset("ylecun/mnist")
    test_pil = ds["test"]["image"]
    test_imgs = np.stack([np.array(img) for img in test_pil])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    probe_gen = torch.Generator().manual_seed(seed + 2)
    probe_indices = [
        torch.randint(len(test_images), (batch_size,), generator=probe_gen)
        for _ in range(probe_batches)
    ]

    # --- Load checkpoints ---
    def make_vit():
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)

    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    base_dir = f"{DATA_DIR}/a2a_forward/mnist/{model_tag}/{gap_tag}"
    dist_dir = f"{DATA_DIR}/a2a_forward/mnist_distillation/{model_tag}/{gap_tag}"

    ol_model = make_vit()
    ol_model.load_state_dict(
        torch.load(f"{base_dir}/ol_model.pt", map_location=device,
                    weights_only=True))
    ol_fm = make_fm()
    ol_fm.load_state_dict(
        torch.load(f"{base_dir}/ol_fwd.pt", map_location=device,
                    weights_only=True))

    cl_model = make_vit()
    cl_model.load_state_dict(
        torch.load(f"{base_dir}/cl_model.pt", map_location=device,
                    weights_only=True))
    cl_fm = make_fm()
    cl_fm.load_state_dict(
        torch.load(f"{base_dir}/cl_fwd.pt", map_location=device,
                    weights_only=True))

    dist_model = make_vit()
    dist_model.load_state_dict(
        torch.load(f"{dist_dir}/distilled_model.pt", map_location=device,
                    weights_only=True))
    dist_fm = make_fm()
    dist_fm.load_state_dict(
        torch.load(f"{dist_dir}/fresh_fm.pt", map_location=device,
                    weights_only=True))

    print("Loaded all checkpoints")

    models = [
        ("OL", ol_model, ol_fm),
        ("CL", cl_model, cl_fm),
        ("Distilled", dist_model, dist_fm),
    ]

    # =============================================
    # Collect activations and properties
    # =============================================
    analysis_layers = ["post_block0", "post_block3"]

    all_data = {}
    for name, model, fm in models:
        model.eval()
        fm.eval()
        print(f"\nCollecting data for {name}...")

        layer_acts = {k: [] for k in analysis_layers}
        digits_list = []
        fm_residual_norms = []
        fm_residual_vecs = []
        block_contrib_norms = []
        confidences = []

        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                labels = test_labels[pidx]

                logits, _, intermediates = model(
                    images, return_intermediates=True)

                digits_list.append(labels)

                probs = F.softmax(logits, dim=-1)
                confidences.append(probs.max(dim=-1).values.cpu())

                src = intermediates[predict_from]
                tgt = intermediates[predict_to]
                pred = fm(src)
                residual = tgt - pred

                fm_residual_norms.append(
                    residual.norm(dim=-1).mean(dim=1).cpu())
                fm_residual_vecs.append(residual.mean(dim=1).cpu())

                b0 = intermediates["post_block0"]
                b1 = intermediates["post_block1"]
                block_contrib_norms.append(
                    (b1 - b0).norm(dim=-1).mean(dim=1).cpu())

                for k in analysis_layers:
                    layer_acts[k].append(
                        intermediates[k].mean(dim=1).cpu())

        all_data[name] = {
            "digits": torch.cat(digits_list),
            "confidences": torch.cat(confidences),
            "fm_res_norms": torch.cat(fm_residual_norms),
            "fm_res_vecs": torch.cat(fm_residual_vecs),
            "block_contrib": torch.cat(block_contrib_norms),
            "layer_acts": {k: torch.cat(v) for k, v in layer_acts.items()},
        }
        print(f"  {name}: {all_data[name]['digits'].shape[0]} images")

    # --- Cluster FM residual directions ---
    print(f"\n{'='*60}")
    print("  CLUSTERING FM RESIDUALS")
    print(f"{'='*60}")

    from scipy.cluster.vq import kmeans2

    for name in all_data:
        res_vecs = all_data[name]["fm_res_vecs"].numpy()
        norms = np.linalg.norm(res_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        res_normed = res_vecs / norms
        centroids, labels = kmeans2(
            res_normed, n_residual_clusters, minit="++", seed=seed)
        all_data[name]["res_clusters"] = torch.tensor(labels)
        sizes = [int((labels == i).sum()) for i in range(n_residual_clusters)]
        print(f"  {name}: cluster sizes = {sizes}")

    # =============================================
    # Probe infrastructure
    # =============================================
    n_samples = all_data["OL"]["digits"].shape[0]
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

    scalar_properties = ["fm_res_norms", "block_contrib", "confidences"]
    scalar_labels = ["FM Residual Norm", "Block1 Contribution", "Confidence"]

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

            acc, W_digit = train_classification_probe(
                acts, data["digits"].numpy(), 10)
            probe_results[name][layer]["digit_acc"] = acc
            probe_weights[name][layer]["digit"] = W_digit
            print(f"  {name:10s} | {layer:12s} | digit acc = {acc:.4f}")

            acc, W_cluster = train_classification_probe(
                acts, data["res_clusters"].numpy(), n_residual_clusters)
            probe_results[name][layer]["cluster_acc"] = acc
            probe_weights[name][layer]["cluster"] = W_cluster
            print(f"  {name:10s} | {layer:12s} | cluster acc = {acc:.4f}")

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

            # Principal direction of digit subspace
            W_digit = weights["digit"]
            _, _, Vt = np.linalg.svd(W_digit, full_matrices=False)
            dirs.append(Vt[0])
            dir_names.append("digit_PC1")

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

            # Digit vs each computational property
            digit_vs_comp = {}
            for i, dn in enumerate(dir_names):
                if dn == "digit_PC1":
                    for j, dn2 in enumerate(dir_names):
                        if dn2 not in ("digit_PC1", "cluster_PC1"):
                            digit_vs_comp[dn2] = float(cos_matrix[i, j])

            result = {
                "mean_abs_cos": float(mean_abs_cos),
                "cos_matrix": cos_matrix.tolist(),
                "direction_names": dir_names,
                "digit_vs_computational": digit_vs_comp,
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

    def compositionality_r2(acts, factor_a, factor_b, n_a, n_b, min_count=5):
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
                if count >= min_count and a in a_centroids and b in b_centroids:
                    actual_list.append(acts[mask].mean(axis=0))
                    pred_list.append(
                        a_centroids[a] + b_centroids[b] - grand_mean)
                    count_list.append(count)

        if len(actual_list) < 10:
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
        digits = data["digits"].numpy()

        res_norms = data["fm_res_norms"].numpy()
        res_quartile = np.digitize(
            res_norms, np.percentile(res_norms, [25, 50, 75]))

        conf = data["confidences"].numpy()
        conf_quartile = np.digitize(
            conf, np.percentile(conf, [25, 50, 75]))

        compositionality_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()

            dr = compositionality_r2(acts, digits, res_quartile, 10, 4)
            dc = compositionality_r2(acts, digits, conf_quartile, 10, 4)

            compositionality_results[name][layer] = {
                "digit_x_residual": dr,
                "digit_x_confidence": dc,
            }
            print(f"  {name:10s} | {layer}: "
                  f"digit×res R²={dr['r2']:.4f} ({dr['n_cells']} cells)  "
                  f"digit×conf R²={dc['r2']:.4f} ({dc['n_cells']} cells)")

    # =============================================
    # TEST 4: VECTOR ARITHMETIC (ANALOGY COMPLETION)
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 4: VECTOR ARITHMETIC (ANALOGY COMPLETION)")
    print(f"{'='*60}")

    arithmetic_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        digits = data["digits"].numpy()
        res_norms = data["fm_res_norms"].numpy()

        median_res = np.median(res_norms)
        high_res = res_norms > median_res

        arithmetic_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            grand_mean = acts.mean(axis=0)

            centroids = {}
            for d in range(10):
                for is_high in [True, False]:
                    mask = (digits == d) & (high_res if is_high else ~high_res)
                    if mask.sum() >= 5:
                        centroids[(d, is_high)] = acts[mask].mean(axis=0)

            cosines = []
            l2_ratios = []

            for a in range(10):
                for b in range(10):
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
    # TEST 5: DIGIT-COMPUTATION INDEPENDENCE
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 5: DIGIT-COMPUTATION INDEPENDENCE")
    print(f"{'='*60}")

    independence_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        independence_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            digits = data["digits"].numpy()

            digit_centroids = []
            for d in range(10):
                mask = digits == d
                if mask.sum() > 5:
                    digit_centroids.append(acts[mask].mean(axis=0))
            digit_centroids = np.stack(digit_centroids)
            digit_centered = digit_centroids - digit_centroids.mean(axis=0)

            _, S, Vt = np.linalg.svd(digit_centered, full_matrices=False)
            digit_subspace = Vt[:5]

            weights = probe_weights[name][layer]
            projections = {}
            for prop in scalar_properties:
                w = weights[prop]
                proj = digit_subspace @ w
                projections[prop] = float(np.linalg.norm(proj))

            independence_results[name][layer] = {
                "projections_onto_digit_subspace": projections,
                "digit_var_in_top5": float(S[:5].sum() / S.sum()),
            }

            print(f"  {name:10s} | {layer}")
            for prop, proj_norm in projections.items():
                print(f"    {prop:20s}: ||proj|| = {proj_norm:.4f}")

    # =============================================
    # TEST 6: SUBSPACE OVERLAP (DIGIT vs COMPUTATION)
    # =============================================
    print(f"\n{'='*60}")
    print("  TEST 6: SUBSPACE OVERLAP (DIGIT vs COMPUTATION)")
    print(f"{'='*60}")

    subspace_results = {}

    for name in ["OL", "CL", "Distilled"]:
        data = all_data[name]
        subspace_results[name] = {}

        for layer in analysis_layers:
            acts = data["layer_acts"][layer].numpy()
            digits = data["digits"].numpy()
            clusters = data["res_clusters"].numpy()

            # Digit subspace (from class centroids)
            d_cents = []
            for d in range(10):
                mask = digits == d
                if mask.sum() > 5:
                    d_cents.append(acts[mask].mean(axis=0))
            d_cents = np.stack(d_cents)
            d_cents -= d_cents.mean(axis=0)
            _, _, Vt_d = np.linalg.svd(d_cents, full_matrices=False)

            # Cluster subspace (from cluster centroids)
            c_cents = []
            for c in range(n_residual_clusters):
                mask = clusters == c
                if mask.sum() > 5:
                    c_cents.append(acts[mask].mean(axis=0))
            c_cents = np.stack(c_cents)
            c_cents -= c_cents.mean(axis=0)
            _, _, Vt_c = np.linalg.svd(c_cents, full_matrices=False)

            overlaps = {}
            for k in [3, 5]:
                D = Vt_d[:k]
                C = Vt_c[:k]
                M = D @ C.T
                svals = np.linalg.svd(M, compute_uv=False)
                mean_cos = float(svals.mean())
                overlaps[f"top{k}_mean_cos"] = mean_cos

            subspace_results[name][layer] = overlaps

            print(f"  {name:10s} | {layer}: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in overlaps.items()))

    # =============================================
    # Save results
    # =============================================
    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_geometry"
                f"/{model_tag}/{gap_tag}")
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "patch_size": patch_size,
            "predict_from": predict_from, "predict_to": predict_to,
            "n_residual_clusters": n_residual_clusters,
            "probe_batches": probe_batches, "probe_steps": probe_steps,
            "analysis_layers": analysis_layers,
            "seed": seed,
        },
        "probe_accuracy": probe_results,
        "orthogonality": orthogonality_results,
        "compositionality": compositionality_results,
        "vector_arithmetic": arithmetic_results,
        "digit_computation_independence": independence_results,
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
    metrics = ["digit_acc", "cluster_acc"] + [
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

    print("\n  Compositionality (digit × residual quartile, R²):")
    for layer in analysis_layers:
        vals = [compositionality_results[n][layer]["digit_x_residual"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Compositionality (digit × confidence, R²):")
    for layer in analysis_layers:
        vals = [compositionality_results[n][layer]["digit_x_confidence"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Vector Arithmetic (analogy cosine):")
    for layer in analysis_layers:
        vals = [arithmetic_results[n][layer]["mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print("\n  Subspace Overlap (digit vs cluster, top5 mean cos):")
    for layer in analysis_layers:
        vals = [subspace_results[n][layer]["top5_mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer:12s}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    print(f"\n  Saved to {save_dir}")
    return result


@app.local_entrypoint()
def main():
    result = a2a_mnist_geometry.remote()
    print("\nMNIST geometry probes complete.")

    pr = result["probe_accuracy"]
    print(f"\n  Probe Accuracy (post_block3):")
    print(f"    {'Metric':25s} {'OL':>8s} {'CL':>8s} {'Dist':>8s}")
    for metric in ["digit_acc", "cluster_acc", "fm_res_norms_r2",
                    "block_contrib_r2", "confidences_r2"]:
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
        vals = [comp[n][layer]["digit_x_residual"]["r2"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer} digit×res: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")

    va = result["vector_arithmetic"]
    print(f"\n  Vector Arithmetic (cosine):")
    for layer in ["post_block0", "post_block3"]:
        vals = [va[n][layer]["mean_cos"]
                for n in ["OL", "CL", "Distilled"]]
        print(f"    {layer}: OL={vals[0]:.4f}  CL={vals[1]:.4f}  "
              f"Dist={vals[2]:.4f}")
