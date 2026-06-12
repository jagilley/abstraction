"""Post-distillation internalization probes.

Two tests on existing checkpoints (OL, CL, distilled), no retraining:

1. Inter-layer self-predictability: linear probe post_block0 → post_block3
   (same model's own activations). Does the distilled model's early computation
   predict its own later computation better than OL's?

2. Old FM prediction accessibility: run old FM on each model's post_block0,
   probe each layer to recover f(a_0). Is the old FM's function more
   linearly transparent in the distilled model?
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_distillation_probes(
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
    print(f"DISTILLATION INTERNALIZATION PROBES on {device}")

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

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    probe_gen = torch.Generator().manual_seed(seed + 200)
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]

    # --- Load checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ctrl_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    dist_root = (f"{DATA_DIR}/a2a_forward/distillation/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    save_root = dist_root

    models = {}

    # Open-loop
    ol = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ol.load_state_dict(torch.load(f"{ctrl_root}/open_loop/model.pt",
                                   map_location=device, weights_only=True))
    models["OL"] = ol

    # CL (pre-distillation, no injection at eval)
    cl = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    cl.load_state_dict(torch.load(f"{ctrl_root}/closed_loop/model.pt",
                                   map_location=device, weights_only=True))
    models["CL"] = cl

    # Distilled
    dist = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    dist.load_state_dict(torch.load(f"{dist_root}/distilled/model.pt",
                                     map_location=device, weights_only=True))
    models["Distilled"] = dist

    # Old FM (trained on CL model)
    old_fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    old_fm.load_state_dict(torch.load(f"{ctrl_root}/closed_loop/fwd_model.pt",
                                       map_location=device, weights_only=True))
    old_fm.eval()
    for p in old_fm.parameters():
        p.requires_grad = False

    print(f"Loaded OL, CL, Distilled models + old FM")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    # =============================================
    # Collect activations for all models
    # =============================================
    def collect_acts(model, label):
        model.eval()
        acts = {k: [] for k in layer_keys}
        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx)
                _, _, vi = model(vx, vy, return_intermediates=True)
                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())
        acts = {k: torch.cat(v) for k, v in acts.items()}
        print(f"  [{label}] Collected {acts[layer_keys[0]].shape[0]} samples")
        return acts

    print("\nCollecting activations...")
    all_acts = {}
    for label, model in models.items():
        all_acts[label] = collect_acts(model, label)

    # Collect old FM predictions on each model's post_block0
    print("\nComputing old FM predictions...")
    fm_preds = {}
    for label, model in models.items():
        model.eval()
        preds = []
        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx)
                _, _, vi = model(vx, vy, return_intermediates=True)
                src = vi[predict_from]
                pred = old_fm(src)
                preds.append(pred.reshape(-1, n_embd).cpu())
        fm_preds[label] = torch.cat(preds)
        print(f"  [{label}] FM pred shape: {fm_preds[label].shape}")

    # =============================================
    # Probe infrastructure
    # =============================================
    n_samples = all_acts["OL"][layer_keys[0]].shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    def vector_probe(X, T_target, label=""):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = T_target.numpy() if isinstance(T_target, torch.Tensor) else T_target
        T_np = T_np.astype(np.float32)

        mu_x = X_np[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu_x) / sd_x

        probe = nn.Linear(X_np.shape[1], T_np.shape[1]).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ttr = torch.from_numpy(T_np[train_idx]).float().to(device)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,), device=device)
            pred = probe(Xtr[idx])
            loss = F.mse_loss(pred, Ttr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).cpu().numpy()
        Tte = T_np[test_idx]
        mse = ((pred - Tte) ** 2).mean()
        var = Tte.var()
        r2 = float(1.0 - mse / var) if var > 0 else 0.0
        cos = float(F.cosine_similarity(
            torch.from_numpy(pred), torch.from_numpy(Tte), dim=-1
        ).mean())
        return {"r2": r2, "cosine": cos}

    # =============================================
    # Test 1: Inter-layer self-predictability
    # =============================================
    print(f"\n{'='*60}")
    print(f"  TEST 1: INTER-LAYER SELF-PREDICTABILITY")
    print(f"  (probe: post_block0 → post_block3, same model)")
    print(f"{'='*60}")

    test1_results = {}
    print(f"\n  {'Model':>12s} {'R²':>8s} {'Cosine':>8s}")
    for label in ["OL", "CL", "Distilled"]:
        source = all_acts[label]["post_block0"]
        target = all_acts[label]["post_block3"]
        result = vector_probe(source, target, label)
        test1_results[label] = result
        print(f"  {label:>12s} {result['r2']:>8.4f} {result['cosine']:>8.4f}")

    print(f"\n  Δ R² (Distilled - OL): "
          f"{test1_results['Distilled']['r2'] - test1_results['OL']['r2']:+.4f}")
    print(f"  Δ R² (CL - OL):        "
          f"{test1_results['CL']['r2'] - test1_results['OL']['r2']:+.4f}")

    # Also probe from other layers
    print(f"\n  Full depth profile (source → post_block3):")
    print(f"  {'Source':>15s} {'OL R²':>8s} {'CL R²':>8s} {'Dist R²':>9s} "
          f"{'Δ(D-OL)':>8s} {'Δ(CL-OL)':>9s}")
    test1_depth = {}
    for src_key in layer_keys[:-1]:  # all except post_block3 (trivial)
        test1_depth[src_key] = {}
        for label in ["OL", "CL", "Distilled"]:
            source = all_acts[label][src_key]
            target = all_acts[label]["post_block3"]
            result = vector_probe(source, target, f"{label}:{src_key}")
            test1_depth[src_key][label] = result

        r2_ol = test1_depth[src_key]["OL"]["r2"]
        r2_cl = test1_depth[src_key]["CL"]["r2"]
        r2_d = test1_depth[src_key]["Distilled"]["r2"]
        print(f"  {src_key:>15s} {r2_ol:>8.4f} {r2_cl:>8.4f} {r2_d:>9.4f} "
              f"{r2_d - r2_ol:>+8.4f} {r2_cl - r2_ol:>+9.4f}")

    # =============================================
    # Test 2: Old FM prediction accessibility
    # =============================================
    print(f"\n{'='*60}")
    print(f"  TEST 2: OLD FM PREDICTION ACCESSIBILITY")
    print(f"  (probe: each layer → old FM's prediction f(post_block0))")
    print(f"{'='*60}")

    test2_results = {}
    print(f"\n  {'Layer':>15s} {'OL R²':>8s} {'CL R²':>8s} {'Dist R²':>9s} "
          f"{'Δ(D-OL)':>8s} {'Δ(CL-OL)':>9s}")
    for lk in layer_keys:
        test2_results[lk] = {}
        for label in ["OL", "CL", "Distilled"]:
            source = all_acts[label][lk]
            target = fm_preds[label]  # old FM's prediction on THIS model's acts
            result = vector_probe(source, target, f"{label}:{lk}")
            test2_results[lk][label] = result

        r2_ol = test2_results[lk]["OL"]["r2"]
        r2_cl = test2_results[lk]["CL"]["r2"]
        r2_d = test2_results[lk]["Distilled"]["r2"]
        print(f"  {lk:>15s} {r2_ol:>8.4f} {r2_cl:>8.4f} {r2_d:>9.4f} "
              f"{r2_d - r2_ol:>+8.4f} {r2_cl - r2_ol:>+9.4f}")

    # =============================================
    # Test 2b: Cross-model FM pred (control)
    # =============================================
    # Also probe each model for the FM's prediction on the OL model's acts
    # This is the cross-model control: if the distilled model encodes the FM's
    # function generally (not just its own version), it should predict what
    # the FM would say about OTHER models' activations too.
    print(f"\n  Cross-model control: probe each model for FM pred on OL's activations")
    print(f"  {'Layer':>15s} {'OL R²':>8s} {'CL R²':>8s} {'Dist R²':>9s} "
          f"{'Δ(D-OL)':>8s}")
    test2b_results = {}
    fm_pred_on_ol = fm_preds["OL"]  # fixed target for all
    for lk in layer_keys:
        test2b_results[lk] = {}
        for label in ["OL", "CL", "Distilled"]:
            source = all_acts[label][lk]
            result = vector_probe(source, fm_pred_on_ol, f"xmodel:{label}:{lk}")
            test2b_results[lk][label] = result

        r2_ol = test2b_results[lk]["OL"]["r2"]
        r2_cl = test2b_results[lk]["CL"]["r2"]
        r2_d = test2b_results[lk]["Distilled"]["r2"]
        print(f"  {lk:>15s} {r2_ol:>8.4f} {r2_cl:>8.4f} {r2_d:>9.4f} "
              f"{r2_d - r2_ol:>+8.4f}")

    # =============================================
    # Test 3: Cosine between old FM pred and model's actual post_block3
    # =============================================
    print(f"\n{'='*60}")
    print(f"  TEST 3: HOW WELL DOES OLD FM PREDICT EACH MODEL?")
    print(f"{'='*60}")

    for label in ["OL", "CL", "Distilled"]:
        pred = fm_preds[label]
        actual = all_acts[label]["post_block3"]
        cos = float(F.cosine_similarity(pred, actual, dim=-1).mean())
        mse = float(F.mse_loss(pred, actual))
        res_norm = float((actual - pred).norm(dim=-1).mean())
        print(f"  [{label:>10s}] cos={cos:.4f}, mse={mse:.5f}, "
              f"res_norm={res_norm:.4f}")

    # =============================================
    # Save
    # =============================================
    result = {
        "test1_self_predictability": {
            "headline": {k: v for k, v in test1_results.items()},
            "depth": {sk: {k: v for k, v in sv.items()}
                      for sk, sv in test1_depth.items()},
        },
        "test2_fm_pred_accessibility": test2_results,
        "test2b_cross_model": test2b_results,
    }

    results_path = os.path.join(save_root, "internalization_probes.json")
    os.makedirs(save_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {results_path}")
    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_distillation_probes.remote(
        n_tokens=n_tokens,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("Distillation internalization probes complete.")
