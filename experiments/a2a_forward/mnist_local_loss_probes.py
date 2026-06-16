"""Representation probes for MNIST local loss experiment.

Loads checkpoints from the local loss experiment and runs probes to distinguish:
  - Object-level knowledge: does the model encode the FM's prediction?
  - Meta-knowledge: does the model encode the FM's residual (where FM is wrong)?

Four probe types at each layer (no injection, native activations):
  prediction:     acts -> FM prediction vector
  residual:       acts -> FM residual vector (target - prediction)
  ortho_residual: acts -> residual perp prediction (pure error signal)
  target:         acts -> actual post_block3 (ceiling)

If CL_LL > CL on prediction probe: internalized FM content (object-level)
If CL_LL > CL on ortho_residual probe: internalized where FM is wrong (meta)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist_ll_probes(
    lambda_local: float = 1.0,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    probe_batches: int = 40,
    batch_size: int = 128,
    probe_steps: int = 300,
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
    n_positions = (28 // patch_size) ** 2 + 1

    # --- Checkpoint directory ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    ckpt_dir = (f"{DATA_DIR}/a2a_forward/mnist_local_loss"
                f"/{model_tag}/{gap_tag}/lambda_{lambda_local}")
    print(f"Loading checkpoints from {ckpt_dir}")

    # --- Load MNIST test set ---
    print("Loading MNIST...")
    ds = load_dataset("ylecun/mnist")
    test_pil = ds["test"]["image"]
    test_imgs = np.stack([np.array(img) for img in test_pil])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    # --- Pre-generate probe indices (same seed as training) ---
    probe_gen = torch.Generator().manual_seed(seed + 2)
    probe_indices = [
        torch.randint(len(test_images), (batch_size,), generator=probe_gen)
        for _ in range(probe_batches)
    ]

    # --- Probe training helper ---
    def run_probe(X_tr, Y_tr, X_te, Y_te):
        d_in, d_out = X_tr.shape[1], Y_tr.shape[1]
        probe = nn.Linear(d_in, d_out).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        n_train = X_tr.shape[0]
        bs = min(4096, n_train)

        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,))
            pred = probe(X_tr[idx])
            loss = F.mse_loss(pred, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = probe(X_te)
            mse = F.mse_loss(pred, Y_te).item()
            var = Y_te.var().item()
            r2 = 1.0 - mse / var if var > 0 else 0.0
            cos = F.cosine_similarity(pred, Y_te, dim=-1).mean().item()
        return {"r2": r2, "cosine": cos}

    # --- Train/test split ---
    n_total = probe_batches * batch_size * n_positions
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:]

    # --- Run probes for each condition ---
    conditions = ["OL", "CL", "LL", "CL_LL"]
    layer_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    probe_types = ["prediction", "residual", "ortho_residual", "target"]

    all_results = {}
    target_stats = {}

    for cond in conditions:
        print(f"\n{'=' * 60}")
        print(f"  Probing {cond}")
        print(f"{'=' * 60}")

        # Load model and FM
        model = ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f"{cond}_model.pt"),
                        weights_only=True),
        )
        model.eval()

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)
        fwd_model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f"{cond}_fwd.pt"),
                        weights_only=True),
        )
        fwd_model.eval()

        # Collect activations (no injection -- native activations)
        layer_acts = {k: [] for k in layer_names}
        fm_predictions = []
        fm_residuals = []
        fm_ortho_residuals = []
        targets = []

        with torch.no_grad():
            for bi in range(probe_batches):
                pidx = probe_indices[bi]
                pim = test_images[pidx].to(device)

                _, _, vi = model(pim, return_intermediates=True)

                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                residual = tgt - pred

                # Orthogonalize: remove prediction direction from residual
                pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
                r_parallel = (
                    (residual * pred_norm).sum(dim=-1, keepdim=True)
                    * pred_norm
                )
                r_perp = residual - r_parallel

                fm_predictions.append(pred.cpu())
                fm_residuals.append(residual.cpu())
                fm_ortho_residuals.append(r_perp.cpu())
                targets.append(tgt.cpu())

                for key in layer_names:
                    layer_acts[key].append(vi[key].cpu())

        # Flatten: (batches * batch_size * n_positions, n_embd)
        fm_pred_flat = torch.cat(fm_predictions).reshape(-1, n_embd)
        fm_res_flat = torch.cat(fm_residuals).reshape(-1, n_embd)
        fm_ortho_flat = torch.cat(fm_ortho_residuals).reshape(-1, n_embd)
        tgt_flat = torch.cat(targets).reshape(-1, n_embd)
        for key in layer_names:
            layer_acts[key] = torch.cat(layer_acts[key]).reshape(-1, n_embd)

        # Target statistics (helps interpret R2)
        pred_var = fm_pred_flat.var().item()
        res_var = fm_res_flat.var().item()
        ortho_var = fm_ortho_flat.var().item()
        tgt_var = tgt_flat.var().item()
        pred_tgt_cos = F.cosine_similarity(
            fm_pred_flat, tgt_flat, dim=-1,
        ).mean().item()
        res_norm = fm_res_flat.norm(dim=-1).mean().item()

        target_stats[cond] = {
            "pred_var": pred_var,
            "res_var": res_var,
            "ortho_var": ortho_var,
            "tgt_var": tgt_var,
            "pred_tgt_cosine": pred_tgt_cos,
            "mean_residual_norm": res_norm,
        }
        print(f"  pred_var={pred_var:.5f}  res_var={res_var:.5f}  "
              f"ortho_var={ortho_var:.5f}")
        print(f"  pred-tgt cos={pred_tgt_cos:.4f}  "
              f"mean |residual|={res_norm:.4f}")

        # Prepare probe targets dict
        probe_targets = {
            "prediction": fm_pred_flat,
            "residual": fm_res_flat,
            "ortho_residual": fm_ortho_flat,
            "target": tgt_flat,
        }

        # Run all probes
        cond_results = {}
        for ptype in probe_types:
            cond_results[ptype] = {}
            Y = probe_targets[ptype]
            Y_tr = Y[tr_idx].to(device)
            Y_te = Y[te_idx].to(device)

            for layer in layer_names:
                X = layer_acts[layer]
                X_tr = X[tr_idx].to(device)
                X_te = X[te_idx].to(device)

                result = run_probe(X_tr, Y_tr, X_te, Y_te)
                cond_results[ptype][layer] = result
                print(f"  {ptype:16s} | {layer:12s}: "
                      f"R2={result['r2']:.4f}  cos={result['cosine']:.4f}")

        all_results[cond] = cond_results

        # Free GPU memory
        del model, fwd_model, layer_acts
        del fm_pred_flat, fm_res_flat, fm_ortho_flat, tgt_flat
        torch.cuda.empty_cache()

    # =============================================
    # Summary tables
    # =============================================
    print(f"\n{'=' * 60}")
    print("  SUMMARY TABLES")
    print(f"{'=' * 60}")

    print("\n  Target statistics (variance of probe targets):")
    print(f"    {'Cond':5s} {'pred_var':>10s} {'res_var':>10s} "
          f"{'ortho_var':>10s} {'pred-tgt cos':>13s} {'|residual|':>11s}")
    for cond in conditions:
        s = target_stats[cond]
        print(f"    {cond:5s} {s['pred_var']:>10.5f} {s['res_var']:>10.5f} "
              f"{s['ortho_var']:>10.5f} {s['pred_tgt_cosine']:>13.4f} "
              f"{s['mean_residual_norm']:>11.4f}")

    for ptype in probe_types:
        print(f"\n  Probe: {ptype} (R2)")
        header = f"    {'Cond':5s}"
        for layer in layer_names:
            short = layer.replace("post_", "")
            header += f" {short:>10s}"
        print(header)

        for cond in conditions:
            row = f"    {cond:5s}"
            for layer in layer_names:
                r2 = all_results[cond][ptype][layer]["r2"]
                row += f" {r2:>10.4f}"
            print(row)

    # Key comparison: CL_LL vs CL differences
    print(f"\n  KEY: CL_LL minus CL (positive = CL_LL encodes more)")
    for ptype in probe_types:
        print(f"    {ptype}:")
        for layer in layer_names:
            cl_r2 = all_results["CL"][ptype][layer]["r2"]
            clll_r2 = all_results["CL_LL"][ptype][layer]["r2"]
            diff = clll_r2 - cl_r2
            marker = " ***" if abs(diff) > 0.05 else ""
            print(f"      {layer:12s}: CL={cl_r2:.4f}  "
                  f"CL_LL={clll_r2:.4f}  delta={diff:+.4f}{marker}")

    # Object-level vs meta-knowledge verdict
    print(f"\n  INTERPRETATION:")
    for layer in ["post_block0", "post_block1"]:
        pred_diff = (all_results["CL_LL"]["prediction"][layer]["r2"]
                     - all_results["CL"]["prediction"][layer]["r2"])
        res_diff = (all_results["CL_LL"]["residual"][layer]["r2"]
                    - all_results["CL"]["residual"][layer]["r2"])
        ortho_diff = (all_results["CL_LL"]["ortho_residual"][layer]["r2"]
                      - all_results["CL"]["ortho_residual"][layer]["r2"])
        print(f"    {layer}:")
        print(f"      prediction delta (object-level):  {pred_diff:+.4f}")
        print(f"      residual delta (composite):       {res_diff:+.4f}")
        print(f"      ortho_residual delta (pure meta): {ortho_diff:+.4f}")

    # =============================================
    # Save results
    # =============================================
    save_path = os.path.join(ckpt_dir, "probe_results.json")
    save_data = {
        "config": {
            "lambda_local": lambda_local,
            "probe_batches": probe_batches,
            "probe_steps": probe_steps,
            "seed": seed,
        },
        "target_stats": target_stats,
        "probes": all_results,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_path}")

    return save_data


@app.local_entrypoint()
def main(lambda_local: float = 1.0):
    result = a2a_mnist_ll_probes.remote(lambda_local=lambda_local)
    print("\nProbe analysis complete.")
