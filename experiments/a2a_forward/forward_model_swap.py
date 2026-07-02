"""Forward model swap test: is self-knowledge about the specific FM or general?

Three main models (OL, CL, Distilled) from existing checkpoints.
For each: train N fresh FMs on its frozen activations, then probe whether
that model's activations encode each FM's residual.

Key questions:
1. CL: does it predict FM-original's residual better than fresh FMs'?
   (FM-specific adaptation vs general computational self-knowledge)
2. Distilled: does it predict fresh FMs' residuals better than OL?
   (did distillation make the model's computation more self-transparent?)
3. Cross-model: train fresh FMs on each model, probe with each model.
   Separates "my computation is predictable" from "I encode where the
   prediction fails."
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


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
    fwd_seeds: str = "100,200,300",
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

    fwd_seeds = [int(s.strip()) for s in fwd_seeds.split(",")]
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

    # --- Load all three main model checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ctrl_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    dist_root = (f"{DATA_DIR}/a2a_forward/distillation/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    def load_gpt(path):
        m = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    def load_fm(path):
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fm.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        fm.eval()
        return fm

    print("\nLoading checkpoints...")
    models = {
        "OL": load_gpt(f"{ctrl_root}/open_loop/model.pt"),
        "CL": load_gpt(f"{ctrl_root}/closed_loop/model.pt"),
        "Distilled": load_gpt(f"{dist_root}/distilled/model.pt"),
    }
    fm_original = load_fm(f"{ctrl_root}/closed_loop/fwd_model.pt")
    print(f"  Loaded OL, CL, Distilled models + FM-original")

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
    # Load or train fresh FMs on each main model's frozen activations
    # =============================================
    # fresh_fms[model_name][seed] = trained FM
    fresh_fms = {}
    ckpt_root_swap = (f"{DATA_DIR}/a2a_forward/swap_test/"
                      f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    for model_name, model in models.items():
        fresh_fms[model_name] = {}

        for fs in fwd_seeds:
            fm = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
            ).to(device)

            saved_path = os.path.join(
                ckpt_root_swap, f"fm_{model_name}_s{fs}", "fm.pt")
            if os.path.exists(saved_path):
                fm.load_state_dict(torch.load(
                    saved_path, map_location=device, weights_only=True))
                fm.eval()
                fresh_fms[model_name][fs] = fm
                print(f"  Loaded FM-{fs} on {model_name} from {saved_path}")
                continue

            # Train from scratch if no checkpoint
            print(f"\n  --- Training FM-{fs} on {model_name} ---")
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
                    _, _, vi = model(x, y, return_intermediates=True)
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
                            _, _, vi = model(vx, vy, return_intermediates=True)
                            src = vi[predict_from]
                            tgt = vi[predict_to]
                            p = fm(src)
                            cos_acc.append(
                                F.cosine_similarity(p, tgt, dim=-1).mean().item())
                        cos = np.mean(cos_acc)
                    print(f"    step {step:6d}: loss={loss.item():.5f} "
                          f"val_cos={cos:.4f}")

            fresh_fms[model_name][fs] = fm
            print(f"  FM-{fs} on {model_name} done.")

    # =============================================
    # Collect activations and residuals
    # =============================================
    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def collect_acts_and_residuals(model, fms_dict):
        """Collect activations and residuals for a set of FMs on one model."""
        model.eval()
        for fm in fms_dict.values():
            fm.eval()

        acts = {k: [] for k in layer_keys}
        residuals = {name: [] for name in fms_dict}

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

        acts = {k: torch.cat(v) for k, v in acts.items()}
        residuals = {k: torch.cat(v) for k, v in residuals.items()}
        return acts, residuals

    # For each main model, we probe with:
    # - FM-original (trained on CL activations during co-training)
    # - Fresh FMs trained on THIS model's activations ("own FMs")
    # - Fresh FMs trained on OTHER models' activations ("cross FMs")

    # Build FM dictionaries for each probing scenario
    # Key structure: results[main_model][probe_layer][fm_label] = R²

    print(f"\n{'='*60}")
    print(f"  COLLECTING ACTIVATIONS AND RESIDUALS")
    print(f"{'='*60}")

    # For each main model, build the full set of FMs to probe with
    all_fm_sets = {}
    for model_name in models:
        fms = {"fm_original": fm_original}
        # Own FMs (trained on this model's activations)
        for fs in fwd_seeds:
            fms[f"own_s{fs}"] = fresh_fms[model_name][fs]
        # Cross FMs (trained on other models' activations)
        for other_name in models:
            if other_name == model_name:
                continue
            for fs in fwd_seeds:
                fms[f"cross_{other_name}_s{fs}"] = fresh_fms[other_name][fs]
        all_fm_sets[model_name] = fms

    collected = {}
    for model_name, model in models.items():
        print(f"\n  Collecting for {model_name}...")
        acts, residuals = collect_acts_and_residuals(model, all_fm_sets[model_name])
        collected[model_name] = (acts, residuals)
        print(f"    {acts[layer_keys[0]].shape[0]} samples collected")

    # =============================================
    # Compute ensemble residuals
    # =============================================
    # For each model, the ensemble residual = target - mean(FM predictions)
    # This averages out seed-specific local minima, leaving the component
    # that any FM of this architecture consistently misses.
    print(f"\n{'='*60}")
    print(f"  COMPUTING ENSEMBLE RESIDUALS")
    print(f"{'='*60}")

    for model_name in models:
        acts, residuals = collected[model_name]
        own_keys = [f"own_s{fs}" for fs in fwd_seeds]
        # ensemble residual = mean of individual residuals
        # (equivalent to target - mean(predictions), since
        #  mean(target - pred_i) = target - mean(pred_i))
        ensemble_res = torch.stack([residuals[k] for k in own_keys]).mean(dim=0)
        residuals[f"own_ensemble"] = ensemble_res

        # Per-seed cosine with the ensemble (how much is shared vs unique)
        for k in own_keys:
            cos = F.cosine_similarity(residuals[k], ensemble_res, dim=-1).mean()
            print(f"  {model_name} {k} vs ensemble: cos={cos:.4f}")

        # Also add ensemble from FM-original + own FMs for CL
        if model_name == "CL":
            all_cl_keys = ["fm_original"] + own_keys
            ensemble_all = torch.stack(
                [residuals[k] for k in all_cl_keys]).mean(dim=0)
            residuals["ensemble_with_orig"] = ensemble_all
            cos_orig = F.cosine_similarity(
                residuals["fm_original"], ensemble_all, dim=-1).mean()
            print(f"  CL fm_original vs ensemble_with_orig: cos={cos_orig:.4f}")

        # Cross-model ensembles
        for other_name in models:
            if other_name == model_name:
                continue
            cross_keys = [f"cross_{other_name}_s{fs}" for fs in fwd_seeds]
            cross_ensemble = torch.stack(
                [residuals[k] for k in cross_keys]).mean(dim=0)
            residuals[f"cross_{other_name}_ensemble"] = cross_ensemble

    # =============================================
    # Probes
    # =============================================
    print(f"\n{'='*60}")
    print(f"  SELF-KNOWLEDGE PROBES ({probe_batches} batches, {probe_steps} steps)")
    print(f"{'='*60}")

    n_samples = collected["OL"][0][layer_keys[0]].shape[0]
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
        return r2

    # Run all probes (individual FMs + ensemble targets)
    probe_results = {}  # probe_results[model_name][layer][fm_label] = R²

    for model_name in models:
        probe_results[model_name] = {}
        acts, residuals = collected[model_name]

        # Build list of all targets to probe for
        probe_targets = list(all_fm_sets[model_name].keys())
        # Add ensemble targets
        probe_targets.append("own_ensemble")
        for other_name in models:
            if other_name != model_name:
                probe_targets.append(f"cross_{other_name}_ensemble")
        if model_name == "CL":
            probe_targets.append("ensemble_with_orig")

        print(f"\n  --- Probing {model_name} ({len(probe_targets)} targets) ---")
        for lk in layer_keys:
            probe_results[model_name][lk] = {}
            for fm_label in probe_targets:
                r2 = vector_probe(acts[lk], residuals[fm_label])
                probe_results[model_name][lk][fm_label] = r2

    # =============================================
    # Display results
    # =============================================

    # --- Question 1: CL - FM-original vs own fresh FMs ---
    print(f"\n{'='*60}")
    print(f"  Q1: IS CL SELF-KNOWLEDGE FM-SPECIFIC OR GENERAL?")
    print(f"{'='*60}")
    print("  (CL model probed with FM-original vs fresh FMs trained on CL)")
    print(f"\n{'layer':>15s} {'fm_orig':>10s} {'own_fresh':>10s} {'delta':>10s}")
    for lk in layer_keys:
        r2_orig = probe_results["CL"][lk]["fm_original"]
        own_r2s = [probe_results["CL"][lk][f"own_s{fs}"] for fs in fwd_seeds]
        r2_own = np.mean(own_r2s)
        print(f"{lk:>15s} {r2_orig:>10.4f} {r2_own:>10.4f} "
              f"{r2_orig - r2_own:>+10.4f}")

    # --- Question 2: Distilled vs OL on fresh FMs ---
    print(f"\n{'='*60}")
    print(f"  Q2: DID DISTILLATION MAKE COMPUTATION MORE SELF-TRANSPARENT?")
    print(f"{'='*60}")
    print("  (Each model probed with fresh FMs trained on its own activations)")
    print(f"\n{'layer':>15s} {'OL':>10s} {'CL':>10s} {'Dist':>10s} "
          f"{'Dist-OL':>10s}")
    for lk in layer_keys:
        ol_r2 = np.mean([probe_results["OL"][lk][f"own_s{fs}"]
                         for fs in fwd_seeds])
        cl_r2 = np.mean([probe_results["CL"][lk][f"own_s{fs}"]
                         for fs in fwd_seeds])
        dist_r2 = np.mean([probe_results["Distilled"][lk][f"own_s{fs}"]
                           for fs in fwd_seeds])
        print(f"{lk:>15s} {ol_r2:>10.4f} {cl_r2:>10.4f} {dist_r2:>10.4f} "
              f"{dist_r2 - ol_r2:>+10.4f}")

    # --- Question 2b: Ensemble probes ---
    print(f"\n{'='*60}")
    print(f"  Q2b: ENSEMBLE PROBES (averaging out seed-specific local minima)")
    print(f"{'='*60}")
    print("  Ensemble residual = target - mean(FM predictions across seeds)")
    print("  Tests whether self-knowledge transfers to the shared component")
    print(f"\n{'layer':>15s} {'OL_ens':>10s} {'CL_ens':>10s} {'Dist_ens':>10s} "
          f"{'CL-OL':>10s} {'Dist-OL':>10s}")
    for lk in layer_keys:
        ol_ens = probe_results["OL"][lk]["own_ensemble"]
        cl_ens = probe_results["CL"][lk]["own_ensemble"]
        dist_ens = probe_results["Distilled"][lk]["own_ensemble"]
        print(f"{lk:>15s} {ol_ens:>10.4f} {cl_ens:>10.4f} {dist_ens:>10.4f} "
              f"{cl_ens - ol_ens:>+10.4f} {dist_ens - ol_ens:>+10.4f}")

    # CL: fm_original vs own_ensemble vs ensemble_with_orig
    print(f"\n  CL model: fm_original vs ensembles")
    print(f"{'layer':>15s} {'fm_orig':>10s} {'own_ens':>10s} {'all_ens':>10s}")
    for lk in layer_keys:
        r2_orig = probe_results["CL"][lk]["fm_original"]
        r2_own_ens = probe_results["CL"][lk]["own_ensemble"]
        r2_all_ens = probe_results["CL"][lk]["ensemble_with_orig"]
        print(f"{lk:>15s} {r2_orig:>10.4f} {r2_own_ens:>10.4f} "
              f"{r2_all_ens:>10.4f}")

    # --- Question 3: Cross-model probes ---
    print(f"\n{'='*60}")
    print(f"  Q3: CROSS-MODEL FM TRANSFER")
    print(f"{'='*60}")
    print("  (Each model probed with FMs trained on OTHER models)")
    print("  Tests: is 'predictability of computation' vs 'encoding of errors'")

    for lk in [layer_keys[0], layer_keys[-1]]:
        print(f"\n  Layer: {lk}")
        print(f"  {'Model':>12s} {'own_FMs':>10s}", end="")
        for other in models:
            print(f" {'cross_'+other:>14s}", end="")
        print()

        for model_name in models:
            own_r2 = np.mean([probe_results[model_name][lk][f"own_s{fs}"]
                              for fs in fwd_seeds])
            row = f"  {model_name:>12s} {own_r2:>10.4f}"
            for other in models:
                if other == model_name:
                    row += f" {'(self)':>14s}"
                else:
                    cross_r2 = np.mean([
                        probe_results[model_name][lk][f"cross_{other}_s{fs}"]
                        for fs in fwd_seeds])
                    row += f" {cross_r2:>14.4f}"
            print(row)

    # --- Question 4: FM prediction quality per main model ---
    print(f"\n{'='*60}")
    print(f"  FM PREDICTION QUALITY (cosine with target)")
    print(f"{'='*60}")
    print("  How predictable is each model's computation?")
    print(f"\n{'FM trained on':>15s} {'→ OL':>10s} {'→ CL':>10s} {'→ Dist':>10s}")

    for fm_source in list(models.keys()) + ["original"]:
        row = f"{'fm_orig' if fm_source == 'original' else fm_source:>15s}"
        for target_name, target_model in models.items():
            cos_acc = []
            with torch.no_grad():
                for eb_idx in fm_eval_indices:
                    vx, vy = make_batch(eb_idx, val_data)
                    _, _, vi = target_model(vx, vy, return_intermediates=True)
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    if fm_source == "original":
                        pred = fm_original(src)
                    else:
                        pred = fresh_fms[fm_source][fwd_seeds[0]](src)
                    cos_acc.append(
                        F.cosine_similarity(pred, tgt, dim=-1).mean().item())
            row += f" {np.mean(cos_acc):>10.4f}"
        print(row)

    # --- Residual similarity across FMs on each model ---
    print(f"\n{'='*60}")
    print(f"  RESIDUAL SIMILARITY ACROSS OWN FMs (cosine)")
    print(f"{'='*60}")
    for model_name in models:
        _, residuals = collected[model_name]
        own_keys = [f"own_s{fs}" for fs in fwd_seeds]
        print(f"\n  {model_name}:")
        for i, k1 in enumerate(own_keys):
            for j, k2 in enumerate(own_keys):
                if j <= i:
                    continue
                cos = F.cosine_similarity(
                    residuals[k1], residuals[k2], dim=-1).mean().item()
                print(f"    {k1} vs {k2}: {cos:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    # Q1 summary
    q1_orig = np.mean([probe_results["CL"][lk]["fm_original"]
                       for lk in layer_keys])
    q1_own = np.mean([
        np.mean([probe_results["CL"][lk][f"own_s{fs}"] for fs in fwd_seeds])
        for lk in layer_keys
    ])
    print(f"\n  Q1 (CL FM-specific?): orig R²={q1_orig:.4f}, "
          f"own fresh R²={q1_own:.4f}, ratio={q1_orig / q1_own:.3f}")
    if q1_orig / q1_own > 1.2:
        print("     → FM-SPECIFIC: CL adapted to this particular FM")
    elif q1_orig / q1_own > 0.8:
        print("     → GENERAL: self-knowledge transfers across FM instances")
    else:
        print("     → INVERTED: fresh FMs captured better than original")

    # Q2 summary
    q2_ol = np.mean([
        np.mean([probe_results["OL"][lk][f"own_s{fs}"] for fs in fwd_seeds])
        for lk in layer_keys
    ])
    q2_dist = np.mean([
        np.mean([probe_results["Distilled"][lk][f"own_s{fs}"]
                 for fs in fwd_seeds])
        for lk in layer_keys
    ])
    print(f"\n  Q2 (Distilled more self-transparent?): OL R²={q2_ol:.4f}, "
          f"Dist R²={q2_dist:.4f}, delta={q2_dist - q2_ol:+.4f}")
    if q2_dist > q2_ol + 0.03:
        print("     → YES: distillation made computation more self-transparent")
    elif q2_dist > q2_ol - 0.03:
        print("     → NO DIFFERENCE: distilled ≈ OL self-transparency")
    else:
        print("     → LESS: distilled computation is harder to self-model")

    # Q2b summary (ensemble)
    q2b_ol_ens = np.mean([probe_results["OL"][lk]["own_ensemble"]
                          for lk in layer_keys])
    q2b_cl_ens = np.mean([probe_results["CL"][lk]["own_ensemble"]
                          for lk in layer_keys])
    q2b_dist_ens = np.mean([probe_results["Distilled"][lk]["own_ensemble"]
                            for lk in layer_keys])
    print(f"\n  Q2b (Ensemble — shared component only):")
    print(f"    OL R²={q2b_ol_ens:.4f}, CL R²={q2b_cl_ens:.4f}, "
          f"Dist R²={q2b_dist_ens:.4f}")
    print(f"    CL - OL = {q2b_cl_ens - q2b_ol_ens:+.4f}, "
          f"Dist - OL = {q2b_dist_ens - q2b_ol_ens:+.4f}")
    if q2b_cl_ens > q2b_ol_ens + 0.03:
        print("     → CL encodes the SHARED component better than OL")
        print("       (self-knowledge IS general, individual FM probes missed it)")
    else:
        print("     → No CL advantage for shared component")
        print("       (self-knowledge is truly FM-specific)")

    # Q3 summary
    print(f"\n  Q3 (Cross-model transfer):")
    for model_name in models:
        own = np.mean([
            np.mean([probe_results[model_name][lk][f"own_s{fs}"]
                     for fs in fwd_seeds])
            for lk in layer_keys
        ])
        cross_vals = []
        for other in models:
            if other == model_name:
                continue
            for lk in layer_keys:
                for fs in fwd_seeds:
                    cross_vals.append(
                        probe_results[model_name][lk][f"cross_{other}_s{fs}"])
        cross = np.mean(cross_vals)
        print(f"    {model_name}: own R²={own:.4f}, cross R²={cross:.4f}, "
              f"ratio={own / cross:.3f}")

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
        "probe_results": probe_results,
        "summary": {
            "q1_cl_fm_original_r2": q1_orig,
            "q1_cl_own_fresh_r2": q1_own,
            "q1_ratio": q1_orig / q1_own,
            "q2_ol_r2": q2_ol,
            "q2_distilled_r2": q2_dist,
            "q2_delta": q2_dist - q2_ol,
            "q2b_ol_ensemble_r2": q2b_ol_ens,
            "q2b_cl_ensemble_r2": q2b_cl_ens,
            "q2b_dist_ensemble_r2": q2b_dist_ens,
            "q2b_cl_minus_ol": q2b_cl_ens - q2b_ol_ens,
        },
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    # Save fresh FM checkpoints
    for model_name in models:
        for fs in fwd_seeds:
            fm_dir = os.path.join(save_root, f"fm_{model_name}_s{fs}")
            os.makedirs(fm_dir, exist_ok=True)
            torch.save(fresh_fms[model_name][fs].state_dict(),
                       os.path.join(fm_dir, "fm.pt"))

    volume.commit()
    print(f"\nAll results saved to {save_root}")
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
    s = result["summary"]
    print("\n" + "=" * 60)
    print("  FORWARD MODEL SWAP TEST — RESULTS")
    print("=" * 60)
    print(f"\n  Q1 (CL FM-specific?)")
    print(f"    FM-original R²: {s['q1_cl_fm_original_r2']:.4f}")
    print(f"    Own fresh R²:   {s['q1_cl_own_fresh_r2']:.4f}")
    print(f"    Ratio:          {s['q1_ratio']:.3f}")
    print(f"\n  Q2 (Distilled more self-transparent?)")
    print(f"    OL R²:       {s['q2_ol_r2']:.4f}")
    print(f"    Dist R²:     {s['q2_distilled_r2']:.4f}")
    print(f"    Delta:       {s['q2_delta']:+.4f}")
    print(f"\n  Q2b (Ensemble — shared component)")
    print(f"    OL R²:       {s['q2b_ol_ensemble_r2']:.4f}")
    print(f"    CL R²:       {s['q2b_cl_ensemble_r2']:.4f}")
    print(f"    Dist R²:     {s['q2b_dist_ensemble_r2']:.4f}")
    print(f"    CL - OL:     {s['q2b_cl_minus_ol']:+.4f}")
