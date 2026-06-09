"""Depth-resolved robustness: desensitization vs reorganization.

The baseline battery showed the shifted condition achieves much of forward
prediction's robustness gain (0.477 vs 0.413 loss-degradation ratio) with zero
reliance and no self-knowledge. That suggests two distinct mechanisms could
produce robustness at the injection point:

1. Desensitization: training with any open-gate injection at post_block1
   teaches the model to absorb additive vectors at that specific location
   (structured-noise regularization). Predicts the robustness gain is LOCAL
   to the injection point.
2. Reorganization: the model reorganizes its representations around the
   forward model's prediction of its own computation. Like the self-knowledge
   it produces, this is depth-distributed. Predicts the robustness gain
   appears at perturbation sites other than the injection point.

This experiment discriminates them by extending two prior analyses from the
injection point to every layer, across all 5 baseline-battery conditions
(open_loop, forward, shifted, random_proj, autoencoder), all evaluated
without injection (matching the battery's robustness protocol):

  Phase A: empirical perturbation at post_embed and after each block --
           loss degradation at multiple magnitudes (extends the battery's
           post_block1-only robustness test)
  Phase B: loss gradient norm and Hessian trace (Hutchinson) at each layer --
           the curvature profile that should predict Phase A via
           E[dL] ~= eps^2/(2d) * tr(H) (extends jacobian_analysis.py to the
           baseline conditions)

post_block3 perturbations pass only through ln_f and the LM head, so they
serve as a negative control: conditions should barely differ there.

Perturbation scale at each layer is s * std(open-loop activations at that
layer), shared across conditions. Perturbation directions and eval batches
use the same seeds as the baseline battery (seed+300, seed+400) so the
post_block1 numbers are directly comparable to the battery's.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "shifted", "random_proj", "autoencoder"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def a2a_depth_robustness(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    seed: int = 42,
    perturb_layers: str = ("post_embed,post_block0,post_block1,"
                           "post_block2,post_block3"),
    s_values: str = "0.5,1.0,2.0",
    n_perturb_dirs: int = 16,
    n_emp_batches: int = 20,
    n_grad_batches: int = 20,
    n_hess_batches: int = 10,
    n_hutchinson_vectors: int = 10,
    eps_scale: str = "open_loop",
    skip_hessian: bool = False,
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT

    assert eps_scale in ("open_loop", "per_condition")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [s.strip() for s in perturb_layers.split(",")]
    s_vals = [float(s) for s in s_values.split(",")]

    print(f"DEPTH-RESOLVED ROBUSTNESS on {device}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Perturbation layers: {layers}")
    print(f"  Magnitudes: {s_vals}, {n_perturb_dirs} dirs, "
          f"{n_emp_batches} batches")
    print(f"  Hessian: {n_hess_batches} batches x "
          f"{n_hutchinson_vectors} Hutchinson vectors")

    # =========================================================
    # Load data
    # =========================================================
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = torch.from_numpy(
        np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]
    print(f"Loaded {len(data):,} tokens, {len(val_data):,} for eval")

    # =========================================================
    # Load the 5 baseline-battery models
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/baseline_battery/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading checkpoints from: {ckpt_root}")

    models = {}
    for cond in CONDITIONS:
        m = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        m.load_state_dict(torch.load(
            os.path.join(ckpt_root, cond, "model.pt"),
            map_location=device, weights_only=True))
        m.eval()
        models[cond] = m
    print(f"Loaded {len(models)} models")

    # =========================================================
    # Eval batches (same seed as battery's robustness test)
    # =========================================================
    n_batches_needed = max(n_emp_batches, n_grad_batches, n_hess_batches)
    perturb_gen = torch.Generator().manual_seed(seed + 400)
    batch_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                      generator=perturb_gen)
        for _ in range(n_batches_needed)
    ]

    def make_batch(indices):
        x = torch.stack(
            [val_data[i.item():i.item() + block_size] for i in indices])
        y = torch.stack(
            [val_data[i.item() + 1:i.item() + block_size + 1]
             for i in indices])
        return x.to(device), y.to(device)

    batches = [make_batch(idx) for idx in batch_indices]

    # Per-layer activation std per condition. eps_scale="open_loop" uses the
    # open-loop model's scale for everyone (matches the baseline battery);
    # "per_condition" scales each model's perturbations by its own activation
    # std, controlling for activation-scale inflation.
    with torch.no_grad():
        x_ref, y_ref = batches[0]
        std_by_cond = {}
        for cond in CONDITIONS:
            _, _, vi_ref = models[cond](
                x_ref, y_ref, return_intermediates=True)
            std_by_cond[cond] = {lk: float(vi_ref[lk].std()) for lk in layers}
    if eps_scale == "per_condition":
        cond_std = std_by_cond
    else:
        cond_std = {cond: std_by_cond["open_loop"] for cond in CONDITIONS}
    layer_std = std_by_cond["open_loop"]
    print(f"eps scaling mode: {eps_scale}")
    for cond in CONDITIONS:
        print(f"  {cond:>12s} std: " + ", ".join(
            f"{lk}={std_by_cond[cond][lk]:.3f}" for lk in layers))

    # =========================================================
    # Manual forward pass with perturbation at an arbitrary layer
    # =========================================================
    def loss_with_delta(model, x, y, layer=None, delta=None):
        tok_emb = model.transformer.wte(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_emb = model.transformer.wpe(pos)
        h = model.transformer.drop(tok_emb + pos_emb)
        if delta is not None and layer == "post_embed":
            h = h + delta
        for i in range(n_layer):
            h = model.transformer.h[i](h)
            if delta is not None and layer == f"post_block{i}":
                h = h + delta
        h = model.transformer.ln_f(h)
        logits = model.lm_head(h)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    # Sanity check: manual forward matches model forward
    with torch.no_grad():
        manual = loss_with_delta(models["open_loop"], x_ref, y_ref).item()
        _, builtin, _ = models["open_loop"](
            x_ref, y_ref, return_intermediates=True)
        assert abs(manual - float(builtin)) < 1e-4, \
            f"manual forward mismatch: {manual} vs {float(builtin)}"
    print(f"Manual forward pass verified (loss {manual:.4f})")

    # =========================================================
    # PHASE A: empirical perturbation at every layer
    # =========================================================
    print(f"\n{'='*70}")
    print("PHASE A: empirical loss degradation by perturbation layer")
    print(f"{'='*70}")

    rng_pert = np.random.default_rng(seed + 300)
    perturb_dirs = []
    for _ in range(n_perturb_dirs):
        d = rng_pert.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs.append(torch.from_numpy(d).to(device))

    base_losses = {}
    for cond in CONDITIONS:
        with torch.no_grad():
            base_losses[cond] = [
                loss_with_delta(models[cond], x, y).item()
                for x, y in batches[:n_emp_batches]
            ]

    emp_results = {cond: {lk: {} for lk in layers} for cond in CONDITIONS}

    for cond in CONDITIONS:
        model = models[cond]
        print(f"\n  [{cond}] base loss = "
              f"{np.mean(base_losses[cond]):.4f}")
        for lk in layers:
            for s_mag in s_vals:
                eps = s_mag * cond_std[cond][lk]
                dl_per_dir = []
                with torch.no_grad():
                    for d in perturb_dirs:
                        delta = eps * d
                        dls = [
                            loss_with_delta(model, x, y, lk, delta).item()
                            - base_losses[cond][bi]
                            for bi, (x, y) in enumerate(
                                batches[:n_emp_batches])
                        ]
                        dl_per_dir.append(np.mean(dls))
                emp_results[cond][lk][s_mag] = {
                    "mean_dL": float(np.mean(dl_per_dir)),
                    "std_dL": float(np.std(dl_per_dir)),
                }
            row = "  ".join(
                f"s={s}: {emp_results[cond][lk][s]['mean_dL']:+.4f}"
                for s in s_vals)
            print(f"    {lk:>12s}: {row}")

    # =========================================================
    # PHASE B: gradient norms and Hessian trace at every layer
    # =========================================================
    print(f"\n{'='*70}")
    print("PHASE B: loss gradient norm and Hessian trace by layer")
    print(f"{'='*70}")

    grad_results = {cond: {} for cond in CONDITIONS}
    hess_results = {cond: {} for cond in CONDITIONS}
    rng_hess = np.random.default_rng(seed + 500)

    for cond in CONDITIONS:
        model = models[cond]
        print(f"\n  [{cond}]")
        for lk in layers:
            # --- shared-perturbation gradient norm ---
            norms = []
            for x, y in batches[:n_grad_batches]:
                delta = torch.zeros(n_embd, device=device, requires_grad=True)
                loss = loss_with_delta(model, x, y, lk, delta)
                g = torch.autograd.grad(loss, delta)[0]
                norms.append(g.norm().item())
            grad_results[cond][lk] = {
                "grad_norm_mean": float(np.mean(norms)),
                "grad_norm_std": float(np.std(norms)),
            }

            # --- Hessian trace (Hutchinson) ---
            if skip_hessian:
                print(f"    {lk:>12s}: ||g||={np.mean(norms):.5f}")
                continue
            traces = []
            for x, y in batches[:n_hess_batches]:
                batch_traces = []
                for _ in range(n_hutchinson_vectors):
                    delta = torch.zeros(
                        n_embd, device=device, requires_grad=True)
                    loss = loss_with_delta(model, x, y, lk, delta)
                    g = torch.autograd.grad(loss, delta, create_graph=True)[0]
                    v = torch.from_numpy(
                        2.0 * rng_hess.integers(0, 2, size=n_embd).astype(
                            np.float32) - 1.0
                    ).to(device)
                    Hv = torch.autograd.grad((g * v).sum(), delta)[0]
                    batch_traces.append((v * Hv).sum().item())
                traces.append(np.mean(batch_traces))
            hess_results[cond][lk] = {
                "trace_mean": float(np.mean(traces)),
                "trace_std": float(np.std(traces)),
            }
            print(f"    {lk:>12s}: ||g||={np.mean(norms):.5f}  "
                  f"tr(H)={np.mean(traces):.5f} +- {np.std(traces):.5f}")

    # =========================================================
    # SUMMARY TABLES
    # =========================================================
    s_main = 2.0 if 2.0 in s_vals else s_vals[-1]

    print(f"\n{'='*70}")
    print(f"SUMMARY 1: empirical dL ratio vs open_loop, by layer (s={s_main})")
    print("(injection entered after block "
          f"{inject_after_block} during training)")
    print(f"{'='*70}")
    header = f"{'condition':>12s}" + "".join(f"{lk:>14s}" for lk in layers)
    print(header)
    for cond in CONDITIONS:
        row = f"{cond:>12s}"
        for lk in layers:
            ol = emp_results["open_loop"][lk][s_main]["mean_dL"]
            dl = emp_results[cond][lk][s_main]["mean_dL"]
            ratio = dl / ol if abs(ol) > 1e-10 else float("nan")
            row += f"{ratio:>14.3f}"
        print(row)

    if not skip_hessian:
        print(f"\n{'='*70}")
        print("SUMMARY 2: Hessian trace ratio vs open_loop, by layer")
        print(f"{'='*70}")
        print(header)
        for cond in CONDITIONS:
            row = f"{cond:>12s}"
            for lk in layers:
                ol = hess_results["open_loop"][lk]["trace_mean"]
                tr = hess_results[cond][lk]["trace_mean"]
                ratio = tr / ol if abs(ol) > 1e-10 else float("nan")
                row += f"{ratio:>14.3f}"
            print(row)

        print(f"\n{'='*70}")
        print(f"SUMMARY 3: Hessian-predicted vs empirical dL (s={s_main})")
        print(f"{'='*70}")
        print(f"{'condition':>12s} {'layer':>12s} {'pred dL':>10s} "
              f"{'emp dL':>10s} {'pred/emp':>9s}")
        for cond in CONDITIONS:
            for lk in layers:
                eps = s_main * cond_std[cond][lk]
                pred = (eps ** 2 / (2 * n_embd)) * \
                    hess_results[cond][lk]["trace_mean"]
                emp = emp_results[cond][lk][s_main]["mean_dL"]
                ratio = pred / emp if abs(emp) > 1e-10 else float("nan")
                print(f"{cond:>12s} {lk:>12s} {pred:>+10.5f} "
                      f"{emp:>+10.5f} {ratio:>9.3f}")

    # Locality index: robustness gain at the injection layer vs other
    # mid-network layers. ~1.0 means depth-uniform, <1 means localized
    # to the injection point.
    inj_layer = f"post_block{inject_after_block}"
    other_layers = [lk for lk in layers
                    if lk not in (inj_layer, "post_embed", "post_block3")]
    print(f"\n{'='*70}")
    print(f"SUMMARY 4: locality of the robustness gain (s={s_main})")
    print(f"(gain = 1 - dL/OL; locality = gain at {other_layers} "
          f"/ gain at {inj_layer})")
    print(f"{'='*70}")
    locality = {}
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        ol_inj = emp_results["open_loop"][inj_layer][s_main]["mean_dL"]
        gain_inj = 1.0 - (
            emp_results[cond][inj_layer][s_main]["mean_dL"] / ol_inj)
        gains_other = []
        for lk in other_layers:
            ol_lk = emp_results["open_loop"][lk][s_main]["mean_dL"]
            gains_other.append(
                1.0 - emp_results[cond][lk][s_main]["mean_dL"] / ol_lk)
        gain_other = float(np.mean(gains_other)) if gains_other else 0.0
        loc = gain_other / gain_inj if abs(gain_inj) > 1e-10 else float("nan")
        locality[cond] = {
            "gain_at_injection": float(gain_inj),
            "gain_at_other_layers": gain_other,
            "locality_ratio": float(loc),
        }
        print(f"  {cond:>12s}: gain@{inj_layer}={gain_inj:+.3f}, "
              f"gain@other={gain_other:+.3f}, ratio={loc:.3f}")

    # =========================================================
    # Save results
    # =========================================================
    save_dir = os.path.join(ckpt_root, "depth_robustness")
    os.makedirs(save_dir, exist_ok=True)
    output = {
        "config": {
            "n_tokens": n_tokens,
            "perturb_layers": layers,
            "s_values": s_vals,
            "n_perturb_dirs": n_perturb_dirs,
            "n_emp_batches": n_emp_batches,
            "n_grad_batches": n_grad_batches,
            "n_hess_batches": n_hess_batches,
            "n_hutchinson_vectors": n_hutchinson_vectors,
            "layer_std": layer_std,
            "std_by_cond": std_by_cond,
            "eps_scale": eps_scale,
            "inject_after_block": inject_after_block,
        },
        "base_losses": {c: float(np.mean(v)) for c, v in base_losses.items()},
        "empirical": {
            cond: {lk: {str(s): v for s, v in sv.items()}
                   for lk, sv in lkv.items()}
            for cond, lkv in emp_results.items()
        },
        "gradient": grad_results,
        "hessian": hess_results,
        "locality": locality,
    }
    results_name = ("results.json" if eps_scale == "open_loop"
                    else f"results_{eps_scale}.json")
    results_path = os.path.join(save_dir, results_name)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nResults saved to {results_path}")

    return output


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=32768)
def a2a_activation_scale_check(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    seed: int = 42,
    n_batches: int = 20,
):
    """Per-condition, per-layer activation scale on the same eval batches.

    Control for the depth_robustness result: perturbations were scaled by the
    OPEN-LOOP model's layer std. If a condition's activations are larger, the
    same absolute perturbation is relatively smaller (especially through the
    final LayerNorm), which would mimic robustness. Reports per-position std
    and mean vector norm at every layer for all 5 conditions.
    """
    import os
    import glob
    import torch
    import numpy as np
    from a2a_forward.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = torch.from_numpy(
        np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]

    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/baseline_battery/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    models = {}
    for cond in CONDITIONS:
        m = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        m.load_state_dict(torch.load(
            os.path.join(ckpt_root, cond, "model.pt"),
            map_location=device, weights_only=True))
        m.eval()
        models[cond] = m

    perturb_gen = torch.Generator().manual_seed(seed + 400)
    batch_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                      generator=perturb_gen)
        for _ in range(n_batches)
    ]

    layer_keys = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    results = {}
    for cond in CONDITIONS:
        stats = {lk: {"std": [], "norm": []} for lk in layer_keys}
        with torch.no_grad():
            for idx in batch_indices:
                x = torch.stack(
                    [val_data[i.item():i.item() + block_size] for i in idx]
                ).to(device)
                y = torch.stack(
                    [val_data[i.item() + 1:i.item() + block_size + 1]
                     for i in idx]).to(device)
                _, _, vi = models[cond](x, y, return_intermediates=True)
                for lk in layer_keys:
                    stats[lk]["std"].append(float(vi[lk].std()))
                    stats[lk]["norm"].append(
                        float(vi[lk].norm(dim=-1).mean()))
        results[cond] = {
            lk: {"std": float(np.mean(v["std"])),
                 "norm": float(np.mean(v["norm"]))}
            for lk, v in stats.items()
        }

    print(f"\n{'='*70}")
    print("Per-layer activation std (and ratio vs open_loop)")
    print(f"{'='*70}")
    print(f"{'condition':>12s}" + "".join(f"{lk:>14s}" for lk in layer_keys))
    for cond in CONDITIONS:
        row = f"{cond:>12s}"
        for lk in layer_keys:
            ol = results["open_loop"][lk]["std"]
            s = results[cond][lk]["std"]
            row += f"{s:>7.3f}({s/ol:.2f})"
        print(row)
    print(f"\n{'='*70}")
    print("Per-layer mean activation vector norm (and ratio vs open_loop)")
    print(f"{'='*70}")
    print(f"{'condition':>12s}" + "".join(f"{lk:>14s}" for lk in layer_keys))
    for cond in CONDITIONS:
        row = f"{cond:>12s}"
        for lk in layer_keys:
            ol = results["open_loop"][lk]["norm"]
            n = results[cond][lk]["norm"]
            row += f"{n:>7.2f}({n/ol:.2f})"
        print(row)

    import json as _json
    save_dir = os.path.join(ckpt_root, "depth_robustness")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "activation_scale.json"), "w") as f:
        _json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return results


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    perturb_layers: str = ("post_embed,post_block0,post_block1,"
                           "post_block2,post_block3"),
    s_values: str = "0.5,1.0,2.0",
    n_perturb_dirs: int = 16,
    n_emp_batches: int = 20,
    n_grad_batches: int = 20,
    n_hess_batches: int = 10,
    n_hutchinson_vectors: int = 10,
):
    result = a2a_depth_robustness.remote(
        n_tokens=n_tokens,
        perturb_layers=perturb_layers,
        s_values=s_values,
        n_perturb_dirs=n_perturb_dirs,
        n_emp_batches=n_emp_batches,
        n_grad_batches=n_grad_batches,
        n_hess_batches=n_hess_batches,
        n_hutchinson_vectors=n_hutchinson_vectors,
    )

    layers = result["config"]["perturb_layers"]
    s_main = result["config"]["s_values"][-1]
    print("\n=== DEPTH ROBUSTNESS COMPLETE ===")
    print(f"\nEmpirical dL ratio vs open_loop (s={s_main}):")
    print(f"{'condition':>12s}" + "".join(f"{lk:>14s}" for lk in layers))
    for cond in CONDITIONS:
        row = f"{cond:>12s}"
        for lk in layers:
            ol = result["empirical"]["open_loop"][lk][str(s_main)]["mean_dL"]
            dl = result["empirical"][cond][lk][str(s_main)]["mean_dL"]
            row += f"{dl / ol if abs(ol) > 1e-10 else float('nan'):>14.3f}"
        print(row)
    print("\nLocality ratios (1.0 = depth-uniform gain, "
          "0 = injection-point-only):")
    for cond, v in result["locality"].items():
        print(f"  {cond:>12s}: {v['locality_ratio']:.3f}")
