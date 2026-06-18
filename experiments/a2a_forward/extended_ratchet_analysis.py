"""Quick analysis of extended ratchet results: activation norm hypothesis.

Tests whether the growing residual norm and robustness are explained by
growing activation magnitudes (making fixed-size perturbations negligible).
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def analyze_activation_norms():
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    n_embd = 128
    n_positions = (28 // 4) ** 2 + 1  # 50
    seed = 42

    # Load MNIST test
    ds = load_dataset("ylecun/mnist")
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    def make_vit():
        return ViT(img_size=28, patch_size=4, in_channels=1, n_classes=10,
                    n_layer=4, n_head=4, n_embd=128).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=128, d_head=32, n_head=1, n_layer=1,
            mlp_mult=2, block_size=n_positions, causal=False).to(device)

    save_root = (f"{DATA_DIR}/a2a_forward/mnist_extended_ratchet"
                 f"/vit_4L_4H_128D/post_block0_to_post_block3")

    # Load models
    wslg_model = make_vit()
    wslg_model.load_state_dict(torch.load(
        f"{save_root}/wslg_model.pt", map_location=device, weights_only=True))
    wslg_model.eval()

    ol_model = make_vit()
    ol_model.load_state_dict(torch.load(
        f"{save_root}/ol_model.pt", map_location=device, weights_only=True))
    ol_model.eval()

    wslg_fm = make_fm()
    wslg_fm.load_state_dict(torch.load(
        f"{save_root}/wslg_fm.pt", map_location=device, weights_only=True))
    wslg_fm.eval()

    # Also load the 4-cycle gated ratchet models for comparison
    save_root_4c = (f"{DATA_DIR}/a2a_forward/mnist_gated_ratchet"
                    f"/vit_4L_4H_128D/post_block0_to_post_block3")
    wslg_4c_model = make_vit()
    wslg_4c_model.load_state_dict(torch.load(
        f"{save_root_4c}/wslg_model.pt", map_location=device, weights_only=True))
    wslg_4c_model.eval()

    ol_4c_model = make_vit()
    ol_4c_model.load_state_dict(torch.load(
        f"{save_root_4c}/ol_model.pt", map_location=device, weights_only=True))
    ol_4c_model.eval()

    # Measure activation norms and related quantities
    batch_size = 256
    n_batches = 20
    torch.manual_seed(seed)

    layer_keys = [f"post_block{i}" for i in range(4)]
    models = {
        "WS_LG_16c": wslg_model,
        "OL_16c": ol_model,
        "WS_LG_4c": wslg_4c_model,
        "OL_4c": ol_4c_model,
    }

    print("=" * 70)
    print("  ACTIVATION NORM ANALYSIS")
    print("=" * 70)

    all_stats = {}
    for name, model in models.items():
        norms_by_layer = {k: [] for k in layer_keys}
        norms_by_layer["post_embed"] = []

        with torch.no_grad():
            for bi in range(n_batches):
                idx = torch.randint(len(test_images), (batch_size,))
                images = test_images[idx].to(device)
                _, _, intermediates = model(images, return_intermediates=True)

                for k in ["post_embed"] + layer_keys:
                    act = intermediates[k]
                    # Per-position, per-sample norm (shape: B × T)
                    norms = act.norm(dim=-1)
                    norms_by_layer[k].append(norms.cpu())

        stats = {}
        for k in ["post_embed"] + layer_keys:
            all_norms = torch.cat(norms_by_layer[k])
            stats[k] = {
                "mean_norm": float(all_norms.mean()),
                "std_norm": float(all_norms.std()),
                "max_norm": float(all_norms.max()),
            }

        all_stats[name] = stats
        print(f"\n  {name}:")
        for k in ["post_embed"] + layer_keys:
            s = stats[k]
            print(f"    {k:15s}: mean={s['mean_norm']:.3f} "
                  f"std={s['std_norm']:.3f} max={s['max_norm']:.3f}")

    # Ratio table: WS_LG / OL
    print(f"\n  === Activation norm ratios (WS_LG / OL) ===")
    print(f"  {'Layer':>15s} | {'4-cycle':>10s} | {'16-cycle':>10s}")
    for k in ["post_embed"] + layer_keys:
        r4 = (all_stats["WS_LG_4c"][k]["mean_norm"] /
              all_stats["OL_4c"][k]["mean_norm"])
        r16 = (all_stats["WS_LG_16c"][k]["mean_norm"] /
               all_stats["OL_16c"][k]["mean_norm"])
        print(f"  {k:>15s} | {r4:10.3f} | {r16:10.3f}")

    # Growth ratio: 16-cycle / 4-cycle
    print(f"\n  === Activation norm growth (16c / 4c) ===")
    print(f"  {'Layer':>15s} | {'WS_LG':>10s} | {'OL':>10s}")
    for k in ["post_embed"] + layer_keys:
        rw = (all_stats["WS_LG_16c"][k]["mean_norm"] /
              all_stats["WS_LG_4c"][k]["mean_norm"])
        ro = (all_stats["OL_16c"][k]["mean_norm"] /
              all_stats["OL_4c"][k]["mean_norm"])
        print(f"  {k:>15s} | {rw:10.3f} | {ro:10.3f}")

    # Residual analysis with norm context
    print(f"\n  === Residual in context of activation norms ===")
    with torch.no_grad():
        for name, model in [("WS_LG_16c", wslg_model), ("OL_16c", ol_model)]:
            res_norms = []
            tgt_norms = []
            cosines = []
            for bi in range(n_batches):
                idx = torch.randint(len(test_images), (batch_size,))
                images = test_images[idx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                tgt = vi["post_block3"]
                if name == "WS_LG_16c":
                    pred = wslg_fm(vi["post_block0"])
                    res = tgt - pred
                    res_norms.append(res.norm(dim=-1).cpu())
                    cosines.append(
                        F.cosine_similarity(pred, tgt, dim=-1).cpu())
                tgt_norms.append(tgt.norm(dim=-1).cpu())

            tgt_n = torch.cat(tgt_norms).mean().item()
            print(f"\n  {name}:")
            print(f"    post_block3 mean norm: {tgt_n:.3f}")
            if res_norms:
                res_n = torch.cat(res_norms).mean().item()
                cos = torch.cat(cosines).mean().item()
                print(f"    residual mean norm:    {res_n:.3f}")
                print(f"    FM cosine:             {cos:.5f}")
                print(f"    residual / activation: {res_n / tgt_n:.4f}")
                # Expected residual from angular error alone
                angle = np.arccos(min(cos, 1.0))
                expected_res = tgt_n * np.sin(angle)
                print(f"    angular error (deg):   {np.degrees(angle):.2f}")
                print(f"    expected res (angular):{expected_res:.3f}")

    # Norm-scaled robustness test
    print(f"\n  === Robustness with norm-scaled perturbations ===")
    print(f"  (eps scaled to be constant fraction of activation norm)")
    inject_block = 1

    with torch.no_grad():
        for name, model in [("WS_LG_16c", wslg_model), ("OL_16c", ol_model)]:
            # First measure activation norm at injection point
            act_norms = []
            for bi in range(10):
                idx = torch.randint(len(test_images), (batch_size,))
                images = test_images[idx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                act_norms.append(
                    vi[f"post_block{inject_block}"].norm(dim=-1).mean().item())
            mean_act_norm = np.mean(act_norms)

            print(f"\n  {name} (act norm at block{inject_block}: {mean_act_norm:.3f}):")

            for rel_eps in [0.05, 0.1, 0.2]:
                abs_eps = rel_eps * mean_act_norm
                deltas = []
                for bi in range(20):
                    idx = torch.randint(len(test_images), (batch_size,))
                    images = test_images[idx].to(device)
                    labels = test_labels[idx].to(device)
                    _, base_loss = model(images, labels)
                    torch.manual_seed(seed + bi + 9000)
                    noise = torch.randn(
                        batch_size, n_positions, n_embd, device=device) * abs_eps
                    _, pert_loss = model(
                        images, labels,
                        perturbation=(inject_block, noise),
                    )
                    deltas.append(pert_loss.item() - base_loss.item())
                mean_d = np.mean(deltas)
                print(f"    rel_eps={rel_eps:.2f} (abs={abs_eps:.2f}): "
                      f"delta_loss={mean_d:+.4f}")

    # Weight norm comparison
    print(f"\n  === Weight norms ===")
    for name, model in models.items():
        total_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        print(f"  {name:15s}: total_weight_norm={total_norm:.3f}")

    return all_stats
