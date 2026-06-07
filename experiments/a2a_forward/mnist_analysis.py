"""MNIST residual direction analysis and causal substitution.

Loads trained models from mnist_experiment.py and runs:
1. Residual direction analysis: PCA of forward model residuals,
   conditioned by digit class and patch position. With eff rank 18/128,
   the residual has far more structure than language (200/256).
2. Causal substitution: replace blocks 1-3 with forward model prediction,
   measure per-digit degradation. Tests whether the low-rank residual
   corresponds to specific missing sub-circuits.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_mnist_analysis(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 256,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    n_pcs: int = 10,
):
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_positions = (28 // patch_size) ** 2 + 1

    # --- Load MNIST ---
    print("Loading MNIST...")
    ds = load_dataset("ylecun/mnist")
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Test: {len(test_images)}")

    # --- Load saved models ---
    gap_tag = "post_block0_to_post_block3"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist/{model_tag}/{gap_tag}"

    model = ViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(save_dir, "ol_model.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
        block_size=n_positions, causal=False,
    ).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(save_dir, "ol_fwd.pt"),
        map_location=device, weights_only=True,
    ))
    fwd_model.eval()
    print("Loaded OL model and forward model")

    # =============================================
    # 1. Residual direction analysis
    # =============================================
    print(f"\n{'=' * 60}")
    print("  RESIDUAL DIRECTION ANALYSIS")
    print(f"{'=' * 60}")

    all_residuals = []
    all_digits = []

    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            end = min(start + batch_size, len(test_images))
            images = test_images[start:end].to(device)

            _, _, vi = model(images, return_intermediates=True)
            src = vi["post_block0"]
            tgt = vi["post_block3"]
            pred = fwd_model(src)

            all_residuals.append((tgt - pred).cpu())
            all_digits.append(test_labels[start:end])

    residuals = torch.cat(all_residuals, dim=0)  # (N, T, d)
    digits = torch.cat(all_digits, dim=0)
    N, T, d = residuals.shape
    print(f"Collected residuals: {residuals.shape}")

    # --- PCA on position-flattened residuals ---
    res_flat = residuals.reshape(-1, d).numpy()
    mean = res_flat.mean(axis=0)
    centered = res_flat - mean
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()

    total_var = eigenvalues.sum()
    proportions = eigenvalues / total_var
    cumulative = np.cumsum(proportions)

    print(f"\n  PCA of residuals:")
    for i in range(min(n_pcs, d)):
        print(f"    PC{i + 1}: {proportions[i] * 100:.1f}% "
              f"(cumul: {cumulative[i] * 100:.1f}%)")

    # --- Project onto top PCs, analyze by digit ---
    top_pcs = eigenvectors[:, :n_pcs]
    projections = centered @ top_pcs  # (N*T, n_pcs)
    proj_per_image = projections.reshape(N, T, n_pcs)

    digit_pc_means = np.zeros((10, n_pcs))
    for dig in range(10):
        mask = (digits == dig).numpy()
        if mask.sum() > 0:
            digit_pc_means[dig] = proj_per_image[mask].mean(axis=(0, 1))

    # eta-squared: fraction of variance explained by digit class
    eta_sq = []
    for pc in range(n_pcs):
        pc_vals = proj_per_image[:, :, pc].mean(axis=1)
        grand_mean = pc_vals.mean()
        ss_between = sum(
            ((digits == dig).sum().item()
             * (pc_vals[digits == dig].mean() - grand_mean) ** 2)
            for dig in range(10)
        )
        ss_total = ((pc_vals - grand_mean) ** 2).sum()
        eta_sq.append(float(ss_between / ss_total) if ss_total > 0 else 0.0)

    print(f"\n  Per-digit mean projection (top 5 PCs):")
    header = "       " + "  ".join(f"d={d}" for d in range(10))
    print(f"  {header}")
    for pc in range(min(n_pcs, 5)):
        vals = "  ".join(f"{digit_pc_means[d, pc]:+.2f}" for d in range(10))
        print(f"    PC{pc + 1} (η²={eta_sq[pc]:.3f}): {vals}")

    # --- CLS vs patch residual norm ---
    res_norms = residuals.norm(dim=-1)
    cls_res = float(res_norms[:, 0].mean())
    patch_res = float(res_norms[:, 1:].mean())
    print(f"\n  Position residual norms:")
    print(f"    CLS:     {cls_res:.4f}")
    print(f"    Patches: {patch_res:.4f}")
    print(f"    CLS/Patch ratio: {cls_res / patch_res:.3f}")

    # --- Spatial residual norm (7x7 grid) ---
    n_h = 28 // patch_size
    patch_norms = res_norms[:, 1:].reshape(N, n_h, n_h)
    spatial_mean = patch_norms.mean(dim=0).numpy()
    print(f"\n  Spatial residual norm ({n_h}x{n_h} grid):")
    for row in range(n_h):
        row_str = "  ".join(f"{spatial_mean[row, col]:.2f}" for col in range(n_h))
        print(f"    {row_str}")

    # --- Per-digit residual norms ---
    print(f"\n  Per-digit residual norms:")
    digit_res_norms = {}
    for dig in range(10):
        mask = (digits == dig).numpy()
        if mask.sum() > 0:
            val = float(res_norms[mask].mean())
            digit_res_norms[str(dig)] = val
            print(f"    digit {dig}: {val:.4f} (n={mask.sum()})")

    # --- Cross-digit residual cosine similarity ---
    digit_mean_res = np.zeros((10, d))
    for dig in range(10):
        mask = (digits == dig).numpy()
        if mask.sum() > 0:
            digit_mean_res[dig] = residuals[mask].reshape(-1, d).numpy().mean(axis=0)

    norms_d = np.linalg.norm(digit_mean_res, axis=1, keepdims=True)
    norms_d[norms_d == 0] = 1
    cos_matrix = (digit_mean_res / norms_d) @ (digit_mean_res / norms_d).T

    print(f"\n  Cross-digit residual cosine similarity:")
    print(f"       " + "  ".join(f"  {d}" for d in range(10)))
    for i in range(10):
        vals = "  ".join(f"{cos_matrix[i, j]:+.2f}" for j in range(10))
        print(f"    {i}: {vals}")

    # --- Per-digit, per-PC: which PCs are most digit-discriminative? ---
    # For each PC, which digit pair has the largest gap?
    print(f"\n  Most discriminative digit pairs per PC:")
    for pc in range(min(n_pcs, 5)):
        means = digit_pc_means[:, pc]
        max_d = int(np.argmax(means))
        min_d = int(np.argmin(means))
        gap = means[max_d] - means[min_d]
        print(f"    PC{pc + 1}: digit {max_d} ({means[max_d]:+.3f}) vs "
              f"digit {min_d} ({means[min_d]:+.3f}), gap={gap:.3f}")

    residual_analysis = {
        "pca_proportions": proportions[:n_pcs].tolist(),
        "pca_cumulative": cumulative[:n_pcs].tolist(),
        "digit_pc_means": digit_pc_means.tolist(),
        "eta_squared_per_pc": eta_sq,
        "cls_residual_norm": cls_res,
        "patch_residual_norm": patch_res,
        "spatial_residual_norm": spatial_mean.tolist(),
        "digit_residual_norms": digit_res_norms,
        "cross_digit_cosine": cos_matrix.tolist(),
    }

    # =============================================
    # 2. Causal substitution
    # =============================================
    print(f"\n{'=' * 60}")
    print("  CAUSAL SUBSTITUTION")
    print(f"{'=' * 60}")
    print("  Normal: all 4 blocks")
    print("  Substituted: block 0 + forward model (replaces blocks 1-3)")
    print("  Ablated: block 0 only (skip blocks 1-3)")

    def embed(images):
        B = images.shape[0]
        patches = model._patchify(images)
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        pos = torch.arange(model.n_positions, device=images.device)
        return x + model.pos_embed(pos)

    def classify(x):
        x = model.ln_f(x)
        return model.head(x[:, 0])

    ce_n_all, ce_s_all, ce_a_all = [], [], []
    acc_n_all, acc_s_all, acc_a_all = [], [], []
    kl_s_all, kl_a_all = [], []
    sub_digits = []

    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            end = min(start + batch_size, len(test_images))
            images = test_images[start:end].to(device)
            labels = test_labels[start:end].to(device)

            # Normal
            x_n = embed(images)
            for blk in model.blocks:
                x_n = blk(x_n)
            logits_n = classify(x_n)

            # Substituted: block 0 + forward model
            x_s = embed(images)
            x_s = model.blocks[0](x_s)
            x_s = fwd_model(x_s)
            logits_s = classify(x_s)

            # Ablated: block 0 only
            x_a = embed(images)
            x_a = model.blocks[0](x_a)
            logits_a = classify(x_a)

            ce_n = F.cross_entropy(logits_n, labels, reduction="none")
            ce_s = F.cross_entropy(logits_s, labels, reduction="none")
            ce_a = F.cross_entropy(logits_a, labels, reduction="none")

            ce_n_all.append(ce_n.cpu())
            ce_s_all.append(ce_s.cpu())
            ce_a_all.append(ce_a.cpu())
            acc_n_all.append((logits_n.argmax(-1) == labels).float().cpu())
            acc_s_all.append((logits_s.argmax(-1) == labels).float().cpu())
            acc_a_all.append((logits_a.argmax(-1) == labels).float().cpu())

            log_pn = F.log_softmax(logits_n, dim=-1)
            log_ps = F.log_softmax(logits_s, dim=-1)
            log_pa = F.log_softmax(logits_a, dim=-1)
            pn = log_pn.exp()
            kl_s_all.append((pn * (log_pn - log_ps)).sum(-1).clamp(min=0).cpu())
            kl_a_all.append((pn * (log_pn - log_pa)).sum(-1).clamp(min=0).cpu())
            sub_digits.append(labels.cpu())

    ce_n = torch.cat(ce_n_all)
    ce_s = torch.cat(ce_s_all)
    ce_a = torch.cat(ce_a_all)
    acc_n = torch.cat(acc_n_all)
    acc_s = torch.cat(acc_s_all)
    acc_a = torch.cat(acc_a_all)
    kl_s = torch.cat(kl_s_all)
    kl_a = torch.cat(kl_a_all)
    all_dig = torch.cat(sub_digits)

    print(f"\n  Overall:")
    print(f"    {'Mode':<12s} {'Acc':>7s} {'CE':>7s} {'ΔCE':>8s} {'KL':>7s}")
    print(f"    {'Normal':<12s} {acc_n.mean():.4f} {ce_n.mean():.4f}"
          f"      —       —")
    print(f"    {'Substituted':<12s} {acc_s.mean():.4f} {ce_s.mean():.4f}"
          f" {(ce_s - ce_n).mean():+.4f} {kl_s.mean():.4f}")
    print(f"    {'Ablated':<12s} {acc_a.mean():.4f} {ce_a.mean():.4f}"
          f" {(ce_a - ce_n).mean():+.4f} {kl_a.mean():.4f}")

    # KL recovery: how much of the ablation KL does substitution recover?
    kl_recovery = 1.0 - float(kl_s.mean()) / float(kl_a.mean()) if kl_a.mean() > 0 else 0
    print(f"\n  Forward model recovers {kl_recovery * 100:.1f}% of block1-3 KL contribution")

    print(f"\n  Per-digit:")
    print(f"    {'Digit':>5s} {'n':>5s} {'Acc_N':>6s} {'Acc_S':>6s} {'Acc_A':>6s}"
          f" {'ΔCE_S':>7s} {'ΔCE_A':>7s} {'KL_S':>6s} {'KL_A':>6s}")

    per_digit = {}
    for dig in range(10):
        mask = all_dig == dig
        nd = mask.sum().item()
        if nd == 0:
            continue
        an = float(acc_n[mask].mean())
        asub = float(acc_s[mask].mean())
        aabl = float(acc_a[mask].mean())
        dce_s = float((ce_s[mask] - ce_n[mask]).mean())
        dce_a = float((ce_a[mask] - ce_n[mask]).mean())
        kls = float(kl_s[mask].mean())
        kla = float(kl_a[mask].mean())

        print(f"    {dig:5d} {nd:5d} {an:.4f} {asub:.4f} {aabl:.4f}"
              f" {dce_s:+.4f} {dce_a:+.4f} {kls:.4f} {kla:.4f}")

        per_digit[str(dig)] = {
            "n": nd, "acc_normal": an, "acc_sub": asub, "acc_abl": aabl,
            "delta_ce_sub": dce_s, "delta_ce_abl": dce_a,
            "kl_sub": kls, "kl_abl": kla,
        }

    substitution = {
        "overall": {
            "acc_normal": float(acc_n.mean()),
            "acc_sub": float(acc_s.mean()),
            "acc_abl": float(acc_a.mean()),
            "ce_normal": float(ce_n.mean()),
            "ce_sub": float(ce_s.mean()),
            "ce_abl": float(ce_a.mean()),
            "delta_ce_sub": float((ce_s - ce_n).mean()),
            "delta_ce_abl": float((ce_a - ce_n).mean()),
            "kl_sub": float(kl_s.mean()),
            "kl_abl": float(kl_a.mean()),
            "kl_recovery_pct": kl_recovery * 100,
        },
        "per_digit": per_digit,
    }

    # --- Save ---
    result = {
        "residual_analysis": residual_analysis,
        "causal_substitution": substitution,
    }
    out_path = os.path.join(save_dir, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {out_path}")
    return result


@app.local_entrypoint()
def main(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
):
    result = a2a_mnist_analysis.remote(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    )
    print("\nMNIST analysis complete.")

    ra = result["residual_analysis"]
    cs = result["causal_substitution"]
    print(f"\nResidual PCA: top-1={ra['pca_proportions'][0] * 100:.1f}%")
    print(f"  CLS/Patch ratio: {ra['cls_residual_norm'] / ra['patch_residual_norm']:.3f}")
    print(f"  Top PC η²: {', '.join(f'{e:.3f}' for e in ra['eta_squared_per_pc'][:5])}")

    o = cs["overall"]
    print(f"\nCausal substitution:")
    print(f"  Normal acc={o['acc_normal']:.4f}")
    print(f"  Substituted acc={o['acc_sub']:.4f} (KL={o['kl_sub']:.4f})")
    print(f"  Ablated acc={o['acc_abl']:.4f} (KL={o['kl_abl']:.4f})")
    print(f"  KL recovery: {o['kl_recovery_pct']:.1f}%")
