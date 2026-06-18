"""Gate structure analysis: what is the learning gate open/closed FOR?

Compares gate selectivity at two points:
1. 4-cycle peak (10% FM, gate mean ~0.76) — from mnist_gated_ratchet
2. 10-cycle closing (1.6% FM, gate mean ~0.38) — from mnist_extended_ratchet

Analyses:
- Per-dimension gate weight profiles (which dims are open vs closed?)
- Digit-conditional gate weights (does the gate treat digits differently?)
- Input-adaptiveness (how much does the gate vary across inputs vs being static?)
- Correlation between gate weights and FM error variance per dimension
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def analyze_gate_structure():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    n_embd = 128
    n_positions = (28 // 4) ** 2 + 1
    seed = 42

    class LearningGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )

        def forward(self, source_cls, fm_error_cls):
            x = torch.cat([source_cls, fm_error_cls], dim=-1)
            return torch.sigmoid(self.net(x))

    # Load MNIST test
    ds = load_dataset("ylecun/mnist")
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    def make_vit():
        return ViT(img_size=28, patch_size=4, in_channels=1, n_classes=10,
                    n_layer=4, n_head=4, n_embd=128).to(device)

    def make_fm(d_head, mlp_mult):
        return TransformerForwardModel(
            d_model=128, d_head=d_head, n_head=1, n_layer=1,
            mlp_mult=mlp_mult, block_size=n_positions, causal=False).to(device)

    configs = {
        "4c_peak_10pct": {
            "root": f"{DATA_DIR}/a2a_forward/mnist_gated_ratchet/vit_4L_4H_128D/post_block0_to_post_block3",
            "fm_d_head": 32, "fm_mlp_mult": 2.0,
            "desc": "4-cycle, 10% FM (gate peak ~0.76)",
        },
        "10c_closing_1.6pct": {
            "root": f"{DATA_DIR}/a2a_forward/mnist_extended_ratchet/vit_4L_4H_128D/post_block0_to_post_block3",
            "fm_d_head": 8, "fm_mlp_mult": 0.25,
            "desc": "10-cycle, 1.6% FM (gate closing ~0.38)",
        },
    }

    all_results = {}

    for run_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {cfg['desc']}")
        print(f"{'='*60}")

        model = make_vit()
        model.load_state_dict(torch.load(
            f"{cfg['root']}/wslg_model.pt", map_location=device,
            weights_only=True))
        model.eval()

        fm = make_fm(cfg["fm_d_head"], cfg["fm_mlp_mult"])
        fm.load_state_dict(torch.load(
            f"{cfg['root']}/wslg_fm.pt", map_location=device,
            weights_only=True))
        fm.eval()

        gate = LearningGate(n_embd, 64).to(device)
        gate.load_state_dict(torch.load(
            f"{cfg['root']}/wslg_lgate.pt", map_location=device,
            weights_only=True))
        gate.eval()

        # Collect gate weights, FM errors, and digit labels
        all_gate_weights = []  # (N, 128)
        all_fm_errors = []     # (N, 128)
        all_digits = []
        all_source_norms = []
        batch_size = 256

        with torch.no_grad():
            for start in range(0, len(test_images), batch_size):
                end = min(start + batch_size, len(test_images))
                images = test_images[start:end].to(device)
                digits = test_labels[start:end]

                _, _, vi = model(images, return_intermediates=True)
                src = vi["post_block0"]
                tgt = vi["post_block3"]
                pred = fm(src)

                source_cls = src[:, 0]
                fm_error_cls = (tgt - pred)[:, 0]
                gw = gate(source_cls, fm_error_cls)

                all_gate_weights.append(gw.cpu())
                all_fm_errors.append(fm_error_cls.cpu())
                all_digits.append(digits)
                all_source_norms.append(src[:, 0].norm(dim=-1).cpu())

        gw = torch.cat(all_gate_weights)       # (10000, 128)
        fm_err = torch.cat(all_fm_errors)       # (10000, 128)
        digits = torch.cat(all_digits)           # (10000,)
        src_norms = torch.cat(all_source_norms)  # (10000,)

        print(f"\n  Overall gate: mean={gw.mean():.4f}, std={gw.std():.4f}")

        # === 1. Per-dimension gate weight profile ===
        dim_means = gw.mean(dim=0)  # (128,)
        dim_stds = gw.std(dim=0)    # (128,)
        sorted_dims = dim_means.argsort()

        print(f"\n  Per-dimension gate weights:")
        print(f"    Min 5 dims: {dim_means[sorted_dims[:5]].tolist()}")
        print(f"    Max 5 dims: {dim_means[sorted_dims[-5:]].tolist()}")
        print(f"    Spread (max-min): {dim_means.max() - dim_means.min():.4f}")
        n_low = (dim_means < 0.2).sum().item()
        n_high = (dim_means > 0.8).sum().item()
        n_mid = 128 - n_low - n_high
        print(f"    Dims <0.2: {n_low}, >0.8: {n_high}, middle: {n_mid}")

        # === 2. Input-adaptiveness: how much does gate vary per input? ===
        # Decompose variance: across-dim vs across-input
        total_var = gw.var().item()
        across_dim_var = dim_means.var().item()
        # Mean within-dim variance (how much does each dim vary across inputs)
        within_dim_var = dim_stds.pow(2).mean().item()
        print(f"\n  Input-adaptiveness:")
        print(f"    Total variance:     {total_var:.5f}")
        print(f"    Across-dim variance:{across_dim_var:.5f} "
              f"({across_dim_var/total_var*100:.1f}%)")
        print(f"    Within-dim variance:{within_dim_var:.5f} "
              f"({within_dim_var/total_var*100:.1f}%)")
        print(f"    -> Gate is {'mostly static' if across_dim_var > within_dim_var else 'mostly input-adaptive'}")

        # === 3. Digit-conditional gate weights ===
        print(f"\n  Digit-conditional gate weights:")
        digit_means = torch.zeros(10, 128)
        digit_counts = torch.zeros(10)
        for d in range(10):
            mask = digits == d
            digit_counts[d] = mask.sum()
            digit_means[d] = gw[mask].mean(dim=0)

        # Overall mean gate weight per digit
        digit_overall = digit_means.mean(dim=1)
        print(f"    Per-digit mean gate weight:")
        for d in range(10):
            print(f"      digit {d}: {digit_overall[d]:.4f} "
                  f"(n={int(digit_counts[d])})")

        # Digit selectivity: how much do different digits get different gate weights?
        # eta^2: fraction of gate variance explained by digit identity
        grand_mean_gw = gw.mean(dim=0, keepdim=True)  # (1, 128)
        ss_total = ((gw - grand_mean_gw) ** 2).sum().item()
        ss_between = 0.0
        for d in range(10):
            mask = digits == d
            n_d = mask.sum().item()
            ss_between += n_d * ((digit_means[d:d+1] - grand_mean_gw) ** 2).sum().item()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        print(f"    Digit eta^2 (fraction of gate variance from digit): {eta_sq:.4f}")

        # Which dimensions are most digit-selective?
        per_dim_eta = torch.zeros(128)
        for dim in range(128):
            gw_dim = gw[:, dim]
            gm = gw_dim.mean()
            ss_t = ((gw_dim - gm) ** 2).sum().item()
            ss_b = 0.0
            for d in range(10):
                mask = digits == d
                if mask.sum() > 0:
                    dm = gw_dim[mask].mean()
                    ss_b += mask.sum().item() * (dm - gm).item() ** 2
            per_dim_eta[dim] = ss_b / ss_t if ss_t > 0 else 0.0

        top_selective = per_dim_eta.argsort(descending=True)[:10]
        print(f"    Most digit-selective dims (eta^2):")
        for i, dim_idx in enumerate(top_selective[:5]):
            print(f"      dim {dim_idx.item()}: eta^2={per_dim_eta[dim_idx]:.4f}, "
                  f"mean_gw={dim_means[dim_idx]:.4f}")

        # === 4. Correlation between gate weights and FM error structure ===
        fm_err_var = fm_err.var(dim=0)  # per-dim FM error variance
        fm_err_mean_abs = fm_err.abs().mean(dim=0)
        corr_gw_errvar = torch.corrcoef(
            torch.stack([dim_means, fm_err_var]))[0, 1].item()
        corr_gw_errmag = torch.corrcoef(
            torch.stack([dim_means, fm_err_mean_abs]))[0, 1].item()
        print(f"\n  Gate-FM error correlation:")
        print(f"    corr(gate_weight, FM_error_variance): {corr_gw_errvar:+.4f}")
        print(f"    corr(gate_weight, FM_error_magnitude): {corr_gw_errmag:+.4f}")

        # === 5. Digit-pair gate similarity ===
        # Do similar digits get similar gate patterns?
        digit_cos = torch.zeros(10, 10)
        for i in range(10):
            for j in range(10):
                digit_cos[i, j] = F.cosine_similarity(
                    digit_means[i:i+1], digit_means[j:j+1]).item()

        print(f"\n  Digit-pair gate cosine similarity (most/least similar):")
        pairs = []
        for i in range(10):
            for j in range(i+1, 10):
                pairs.append((i, j, digit_cos[i, j].item()))
        pairs.sort(key=lambda x: x[2])
        print(f"    Least similar: ", end="")
        print(", ".join(f"{a}-{b}:{c:.3f}" for a, b, c in pairs[:5]))
        print(f"    Most similar:  ", end="")
        print(", ".join(f"{a}-{b}:{c:.3f}" for a, b, c in pairs[-5:]))

        all_results[run_name] = {
            "overall_mean": float(gw.mean()),
            "overall_std": float(gw.std()),
            "dim_means": dim_means.tolist(),
            "dim_stds": dim_stds.tolist(),
            "n_low": n_low, "n_high": n_high, "n_mid": n_mid,
            "across_dim_var": across_dim_var,
            "within_dim_var": within_dim_var,
            "digit_overall_means": digit_overall.tolist(),
            "digit_eta_sq": eta_sq,
            "per_dim_eta_sq": per_dim_eta.tolist(),
            "corr_gw_errvar": corr_gw_errvar,
            "corr_gw_errmag": corr_gw_errmag,
        }

    # === Cross-run comparison ===
    print(f"\n{'='*60}")
    print(f"  CROSS-RUN COMPARISON")
    print(f"{'='*60}")

    gw_peak = torch.tensor(all_results["4c_peak_10pct"]["dim_means"])
    gw_close = torch.tensor(all_results["10c_closing_1.6pct"]["dim_means"])

    print(f"\n  Per-dimension profile correlation:")
    corr = torch.corrcoef(torch.stack([gw_peak, gw_close]))[0, 1].item()
    print(f"    corr(peak_dim_means, closing_dim_means): {corr:+.4f}")
    cos = F.cosine_similarity(gw_peak.unsqueeze(0), gw_close.unsqueeze(0)).item()
    print(f"    cosine(peak_dims, closing_dims): {cos:.4f}")

    # Which dimensions changed most?
    delta = gw_close - gw_peak
    biggest_drop = delta.argsort()[:5]
    biggest_rise = delta.argsort(descending=True)[:5]
    print(f"\n  Dimensions that closed most (peak → closing):")
    for d in biggest_drop:
        print(f"    dim {d.item()}: {gw_peak[d]:.3f} → {gw_close[d]:.3f} "
              f"(Δ={delta[d]:+.3f})")
    print(f"  Dimensions that opened most:")
    for d in biggest_rise:
        print(f"    dim {d.item()}: {gw_peak[d]:.3f} → {gw_close[d]:.3f} "
              f"(Δ={delta[d]:+.3f})")

    # Digit selectivity comparison
    print(f"\n  Digit selectivity (eta^2):")
    print(f"    Peak (4c, 10% FM):    {all_results['4c_peak_10pct']['digit_eta_sq']:.4f}")
    print(f"    Closing (10c, 1.6% FM): "
          f"{all_results['10c_closing_1.6pct']['digit_eta_sq']:.4f}")

    print(f"\n  Input-adaptiveness:")
    for name in ["4c_peak_10pct", "10c_closing_1.6pct"]:
        r = all_results[name]
        total = r["across_dim_var"] + r["within_dim_var"]
        print(f"    {name}: across-dim={r['across_dim_var']/total*100:.1f}%, "
              f"within-dim={r['within_dim_var']/total*100:.1f}%")

    return all_results
