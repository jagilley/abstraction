"""Forward model swap test: is self-knowledge specific to the training FM or general?

Loads existing closed-loop and open-loop checkpoints from the controlled retrain.
Trains N fresh forward models (different seeds) on the closed-loop model's frozen
activations. Then probes: does the closed-loop model predict FM-original's residual
better than the fresh FMs' residuals?

If the closed-loop model equally predicts all FMs' residuals → the self-knowledge
is about the model's own computational structure, not the specific auxiliary input.
If it specifically predicts FM-original's residual → the self-knowledge is an
adaptation to the particular compressed view it was trained with.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_forward_model_swap(
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
    fwd_lr: float = 1e-3,
    fwd_train_steps: int = 10_000,
    fwd_seeds: list = [100, 200, 300],
    seed: int = 42,
    probe_batches: int = 40,
    probe_steps: int = 500,
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
    print(f"A2A FORWARD MODEL SWAP TEST on {device}")
    print(f"  Fresh FM seeds: {fwd_seeds}")
    print(f"  FM architecture: {fwd_n_layer}L, {fwd_n_head}H, {fwd_d_head}D")

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
    print(f"Loaded {len(data):,} tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + block_size] for i in indices])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # --- Load checkpoints from controlled retrain ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    print("\nLoading controlled retrain checkpoints...")
    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(torch.load(
        os.path.join(ckpt_root, "open_loop", "model.pt"), map_location=device))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(torch.load(
        os.path.join(ckpt_root, "closed_loop", "model.pt"), map_location=device))

    fm_original = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fm_original.load_state_dict(torch.load(
        os.path.join(ckpt_root, "closed_loop", "fwd_model.pt"), map_location=device))

    model_open.eval()
    model_closed.eval()
    fm_original.eval()
    print("  Loaded open-loop model, closed-loop model, and FM-original")

    # --- Pre-generate batch indices ---
    train_gen = torch.Generator().manual_seed(seed)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    eval_gen = torch.Generator().manual_seed(seed + 3)

    fm_train_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(fwd_train_steps)
    ]
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]
    fm_eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(5)
    ]

    # =============================================
    # Train fresh forward models on frozen closed-loop activations
    # =============================================
    print(f"\n{'='*60}")
    print(f"  TRAINING {len(fwd_seeds)} FRESH FORWARD MODELS")
    print(f"{'='*60}")

    fresh_fms = {}
    for fs in fwd_seeds:
        print(f"\n  --- Training FM-{fs} (seed={fs}) ---")
        torch.manual_seed(fs)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(fwd_train_steps):
            fm.train()
            x, y = make_batch(fm_train_indices[step], train_data)

            with torch.no_grad():
                _, _, vi = model_closed(x, y, return_intermediates=True)
                source = vi[predict_from]
                target = vi[predict_to]

            pred = fm(source)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()

            if step % 2000 == 0 or step == fwd_train_steps - 1:
                fm.eval()
                with torch.no_grad():
                    cos_acc = []
                    for eb_idx in fm_eval_indices:
                        vx, vy = make_batch(eb_idx, val_data)
                        _, _, vi = model_closed(vx, vy, return_intermediates=True)
                        src = vi[predict_from]
                        tgt = vi[predict_to]
                        p = fm(src)
                        cos_acc.append(
                            F.cosine_similarity(p, tgt, dim=-1).mean().item())
                    cos = np.mean(cos_acc)
                print(f"    step {step:6d}: loss={loss.item():.5f} "
                      f"val_cos={cos:.4f}")

        fresh_fms[fs] = fm
        print(f"  FM-{fs} trained.")

    # =============================================
    # Compare residuals: FM-original vs fresh FMs
    # =============================================
    print(f"\n{'='*60}")
    print(f"  RESIDUAL COMPARISON")
    print(f"{'='*60}")

    all_fms = {"original": fm_original}
    for fs in fwd_seeds:
        all_fms[f"seed{fs}"] = fresh_fms[fs]

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def collect_residuals_and_acts(model, fms_dict, label):
        """Collect activations and residuals for all FMs on one main model."""
        model.eval()
        for fm in fms_dict.values():
            fm.eval()

        acts = {k: [] for k in layer_keys}
        residuals = {name: [] for name in fms_dict}
        residual_norms = {name: [] for name in fms_dict}

        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)

                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())

                src = vi[predict_from]
                tgt = vi[predict_to]

                for name, fm in fms_dict.items():
                    pred = fm(src)
                    res = tgt - pred
                    residuals[name].append(res.reshape(-1, n_embd).cpu())
                    residual_norms[name].append(
                        res.norm(dim=-1).reshape(-1).cpu())

        acts = {k: torch.cat(v) for k, v in acts.items()}
        residuals = {k: torch.cat(v) for k, v in residuals.items()}
        residual_norms = {k: torch.cat(v) for k, v in residual_norms.items()}
        return acts, residuals, residual_norms

    print("\nCollecting closed-loop model data...")
    cl_acts, cl_residuals, cl_res_norms = collect_residuals_and_acts(
        model_closed, all_fms, "closed")
    print("Collecting open-loop model data...")
    ol_acts, ol_residuals, ol_res_norms = collect_residuals_and_acts(
        model_open, all_fms, "open")

    # --- Residual similarity across FMs ---
    fm_names = list(all_fms.keys())
    print("\n=== Residual similarity (cosine) on closed-loop activations ===")
    print(f"{'':>12s}", end="")
    for name in fm_names:
        print(f" {name:>10s}", end="")
    print()
    for i, n1 in enumerate(fm_names):
        print(f"{n1:>12s}", end="")
        for j, n2 in enumerate(fm_names):
            cos = F.cosine_similarity(
                cl_residuals[n1], cl_residuals[n2], dim=-1).mean().item()
            print(f" {cos:>10.4f}", end="")
        print()

    # --- Residual norm comparison ---
    print("\n=== Mean residual norm on closed-loop activations ===")
    for name in fm_names:
        norm = cl_res_norms[name].mean().item()
        print(f"  {name:>12s}: {norm:.4f}")

    # --- Prediction quality comparison ---
    print("\n=== FM prediction quality (cosine with target) ===")
    for name in fm_names:
        cos = (1.0 - cl_res_norms[name].pow(2).mean() /
               (cl_residuals[name] + cl_residuals[name]).pow(2).mean())
        pred_cos_acc = []
        with torch.no_grad():
            for pidx in probe_indices[:10]:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model_closed(vx, vy, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = all_fms[name](src)
                pred_cos_acc.append(
                    F.cosine_similarity(pred, tgt, dim=-1).mean().item())
        cos = np.mean(pred_cos_acc)
        print(f"  {name:>12s}: cosine={cos:.4f}")

    # =============================================
    # Probes: does the closed-loop model encode each FM's residual?
    # =============================================
    print(f"\n{'='*60}")
    print(f"  SELF-KNOWLEDGE PROBES ({probe_batches} batches, {probe_steps} steps)")
    print(f"{'='*60}")

    n_samples = cl_acts[layer_keys[0]].shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    print(f"Probe dataset: {n_samples} samples ({n_train} train, "
          f"{n_samples - n_train} test)")

    def vector_probe(X, target_vec):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = target_vec.numpy() if isinstance(target_vec, torch.Tensor) \
            else target_vec
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

    # --- Run probes: closed-loop model → each FM's residual ---
    print("\n=== Closed-loop model probing each FM's residual (R²) ===")
    header = f"{'layer':>12s}"
    for name in fm_names:
        header += f" {name:>10s}"
    print(header)

    cl_probe_results = {}
    for lk in layer_keys:
        cl_probe_results[lk] = {}
        row = f"{lk:>12s}"
        for name in fm_names:
            res = vector_probe(cl_acts[lk], cl_residuals[name])
            cl_probe_results[lk][name] = res
            row += f" {res['r2']:>10.4f}"
        print(row)

    # --- Control: open-loop model → each FM's residual ---
    print("\n=== Open-loop model probing each FM's residual (R²) ===")
    print(header)

    ol_probe_results = {}
    for lk in layer_keys:
        ol_probe_results[lk] = {}
        row = f"{lk:>12s}"
        for name in fm_names:
            res = vector_probe(ol_acts[lk], ol_residuals[name])
            ol_probe_results[lk][name] = res
            row += f" {res['r2']:>10.4f}"
        print(row)

    # --- Delta: closed - open for each FM ---
    print("\n=== Δ R² (closed - open) for each FM ===")
    print(header)
    for lk in layer_keys:
        row = f"{lk:>12s}"
        for name in fm_names:
            delta = cl_probe_results[lk][name]["r2"] - \
                    ol_probe_results[lk][name]["r2"]
            row += f" {delta:>+10.4f}"
        print(row)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  SWAP TEST SUMMARY")
    print(f"{'='*60}")
    print("\nIf self-knowledge is about the model's own computation (general):")
    print("  - Closed-loop R² should be similar for FM-original and fresh FMs")
    print("  - Δ R² should be similar across all FMs")
    print("\nIf self-knowledge is specific to FM-original (adapted to input):")
    print("  - Closed-loop R² should be higher for FM-original")
    print("  - Δ R² for FM-original >> Δ R² for fresh FMs")

    avg_orig = np.mean([cl_probe_results[lk]["original"]["r2"]
                        for lk in layer_keys])
    fresh_avgs = []
    for fs in fwd_seeds:
        avg = np.mean([cl_probe_results[lk][f"seed{fs}"]["r2"]
                       for lk in layer_keys])
        fresh_avgs.append(avg)
    avg_fresh = np.mean(fresh_avgs)

    print(f"\n  Mean closed-loop R² (FM-original):   {avg_orig:.4f}")
    print(f"  Mean closed-loop R² (fresh FMs avg):  {avg_fresh:.4f}")
    print(f"  Ratio (original / fresh):             {avg_orig / avg_fresh:.3f}")

    # =============================================
    # Save
    # =============================================
    save_root = (f"{DATA_DIR}/a2a_forward/swap_test/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    os.makedirs(save_root, exist_ok=True)

    result = {
        "config": {
            "seed": seed, "fwd_seeds": fwd_seeds,
            "fwd_train_steps": fwd_train_steps,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
        },
        "cl_probe_results": cl_probe_results,
        "ol_probe_results": ol_probe_results,
        "summary": {
            "mean_cl_r2_original": avg_orig,
            "mean_cl_r2_fresh": avg_fresh,
            "ratio": avg_orig / avg_fresh,
        },
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)

    # Save fresh FM checkpoints
    for fs in fwd_seeds:
        fm_path = os.path.join(save_root, f"fm_seed{fs}.pt")
        torch.save(fresh_fms[fs].state_dict(), fm_path)

    volume.commit()
    print(f"\nAll results saved to {save_root}")
    return result


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
    fwd_train_steps: int = 10_000,
):
    import numpy as np
    result = a2a_forward_model_swap.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        fwd_train_steps=fwd_train_steps,
    )
    print("Forward model swap test complete:")
    s = result["summary"]
    print(f"  Mean CL R² (FM-original):  {s['mean_cl_r2_original']:.4f}")
    print(f"  Mean CL R² (fresh FMs):    {s['mean_cl_r2_fresh']:.4f}")
    print(f"  Ratio:                     {s['ratio']:.3f}")
    print("\n  Per-layer Δ R² (closed - open):")
    for lk in sorted(result["cl_probe_results"].keys()):
        cl = result["cl_probe_results"][lk]
        ol = result["ol_probe_results"][lk]
        orig_d = cl["original"]["r2"] - ol["original"]["r2"]
        fresh_ds = []
        for k in cl:
            if k != "original":
                fresh_ds.append(cl[k]["r2"] - ol[k]["r2"])
        print(f"    {lk}: original Δ={orig_d:+.4f}, "
              f"fresh avg Δ={np.mean(fresh_ds):+.4f}")
