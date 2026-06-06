"""Jacobian analysis: spectral structure of sensitivity at the injection point.

Formalizes the connection between self-knowledge and robustness by analyzing
the model's loss-sensitivity structure at post_block1 (the injection point).

Computes:
1. Loss gradient norms: ||∂L/∂h|| at the injection point for OL, CL+M, CL-M
2. Hessian trace: expected loss increase from random perturbation (via Hutchinson)
3. Gradient covariance spectrum: effective rank and concentration
4. Self-knowledge subspace alignment: what fraction of sensitivity falls in SK dirs
5. Empirical perturbation validation: actual ΔL vs spectral predictions

The core argument: for random perturbation δ with ||δ|| = ε,
  E[ΔL] ≈ (ε²/2d) tr(H)   where H = ∂²L/∂δ²
  E[ΔL²] ≈ (ε²/d) ||∇_δ L||²

If the closed-loop model has smaller tr(H) and/or ||∇L||, that directly
explains the 2x robustness factor. If the self-knowledge subspace captures
a disproportionate fraction of the gradient covariance, that connects
self-knowledge to robustness through a specific alignment claim.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_jacobian_analysis(
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
    seed: int = 42,
    n_eval_batches: int = 40,
    n_hutchinson_vectors: int = 10,
    n_perturb_dirs: int = 16,
    sk_probe_steps: int = 500,
    sk_top_k_values: str = "10,25,50,100,128",
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))
    sk_k_values = [int(k) for k in sk_top_k_values.split(",")]

    print(f"JACOBIAN ANALYSIS on {device}")
    print(f"  Predict: {predict_from} -> {predict_to}")
    print(f"  Inject after block {inject_after_block}")
    print(f"  SK subspace dims to test: {sk_k_values}")

    # ==========================================================
    # Load data
    # ==========================================================
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

    # ==========================================================
    # Load models (from controlled retrain)
    # ==========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    open_dir = os.path.join(ckpt_root, "open_loop")
    closed_dir = os.path.join(ckpt_root, "closed_loop")
    print(f"Loading from: {ckpt_root}")

    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(torch.load(
        os.path.join(open_dir, "model.pt"),
        map_location=device, weights_only=True))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "model.pt"),
        map_location=device, weights_only=True))

    fwd_closed = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "fwd_model.pt"),
        map_location=device, weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(closed_dir, "gate.pt"),
        map_location=device, weights_only=True))

    for m in [model_open, model_closed, fwd_closed, gate]:
        m.eval()
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    # ==========================================================
    # Eval batches
    # ==========================================================
    eval_gen = torch.Generator().manual_seed(seed + 200)
    eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(n_eval_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # Compute activation std for perturbation scaling
    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0])
        _, _, vi_ref = model_open(x_ref, y_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"post_block1 activation std: {b1_std:.4f}")

    # ==========================================================
    # Helpers
    # ==========================================================
    def compute_h_base(model, x, use_injection=False):
        """Compute activation entering block 2 (after any injection), detached."""
        with torch.no_grad():
            tok_emb = model.transformer.wte(x)
            pos = torch.arange(0, x.size(1), device=x.device)
            pos_emb = model.transformer.wpe(pos)
            h = model.transformer.drop(tok_emb + pos_emb)

            cb_src = None
            for i in range(inject_after_block + 1):
                h = model.transformer.h[i](h)
                if use_injection and i == cerebellar_input_block:
                    cb_src = h

            if use_injection and cb_src is not None:
                h = h + gate(fwd_closed(cb_src))

        return h

    def loss_from_h(model, h, y):
        """Forward from h through remaining blocks to loss."""
        x = h
        for i in range(inject_after_block + 1, n_layer):
            x = model.transformer.h[i](x)
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    conditions = {
        "OL":   (model_open, False),
        "CL+M": (model_closed, True),
        "CL-M": (model_closed, False),
    }

    # ==========================================================
    # PHASE 1: Loss gradient analysis
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Loss gradient analysis at injection point")
    print(f"{'='*60}")

    grad_results = {}
    all_per_pos_grads = {}
    n_grad_store_batches = min(20, n_eval_batches)

    for cond_name, (model, use_inj) in conditions.items():
        shared_sens_norms = []
        per_pos_grad_norms = []
        all_grads_flat = []

        for bi, idx in enumerate(eval_indices):
            x, y = make_batch(idx)
            h_base = compute_h_base(model, x, use_inj)

            # --- Shared perturbation sensitivity ---
            # δ in R^D added to all positions; ∂L/∂δ = sum of per-pos grads
            delta = torch.zeros(n_embd, device=device, requires_grad=True)
            h_pert = h_base + delta.unsqueeze(0).unsqueeze(0)
            loss = loss_from_h(model, h_pert, y)
            s = torch.autograd.grad(loss, delta)[0]
            shared_sens_norms.append(s.norm().item())

            # --- Per-position gradients ---
            h_req = h_base.clone().requires_grad_(True)
            loss2 = loss_from_h(model, h_req, y)
            g = torch.autograd.grad(loss2, h_req)[0]  # (B, T, D)
            g_flat = g.reshape(-1, n_embd)
            per_pos_grad_norms.append(g_flat.norm(dim=-1).mean().item())

            if bi < n_grad_store_batches:
                all_grads_flat.append(g_flat.detach().cpu())

        grad_results[cond_name] = {
            "shared_sens_norm_mean": float(np.mean(shared_sens_norms)),
            "shared_sens_norm_std": float(np.std(shared_sens_norms)),
            "per_pos_grad_norm_mean": float(np.mean(per_pos_grad_norms)),
            "per_pos_grad_norm_std": float(np.std(per_pos_grad_norms)),
        }
        all_per_pos_grads[cond_name] = torch.cat(all_grads_flat)

        print(f"  [{cond_name}] shared ||s||={np.mean(shared_sens_norms):.6f} "
              f"± {np.std(shared_sens_norms):.6f}, "
              f"per-pos ||g||={np.mean(per_pos_grad_norms):.6f}")

    print(f"\n  --- Ratios vs OL ---")
    ol_shared = grad_results["OL"]["shared_sens_norm_mean"]
    ol_perpos = grad_results["OL"]["per_pos_grad_norm_mean"]
    for cn in conditions:
        r = grad_results[cn]
        print(f"  {cn:>6s}: shared={r['shared_sens_norm_mean']/ol_shared:.3f}x, "
              f"per-pos={r['per_pos_grad_norm_mean']/ol_perpos:.3f}x")

    # ==========================================================
    # PHASE 2: Hessian trace estimation (Hutchinson)
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Hessian trace (shared perturbation curvature)")
    print(f"{'='*60}")
    print(f"  Using {n_hutchinson_vectors} Rademacher vectors per batch")

    hess_results = {}
    rng_hess = np.random.default_rng(seed + 500)
    n_hess_batches = min(20, n_eval_batches)

    for cond_name, (model, use_inj) in conditions.items():
        trace_estimates = []

        for bi in range(n_hess_batches):
            x, y = make_batch(eval_indices[bi])
            h_base = compute_h_base(model, x, use_inj)

            batch_traces = []
            for vi in range(n_hutchinson_vectors):
                delta = torch.zeros(n_embd, device=device, requires_grad=True)
                h_pert = h_base + delta.unsqueeze(0).unsqueeze(0)
                loss = loss_from_h(model, h_pert, y)
                s = torch.autograd.grad(loss, delta, create_graph=True)[0]

                v = torch.from_numpy(
                    2.0 * rng_hess.integers(0, 2, size=n_embd).astype(
                        np.float32) - 1.0
                ).to(device)

                sv = (s * v).sum()
                Hv = torch.autograd.grad(sv, delta)[0]
                batch_traces.append((v * Hv).sum().item())

            trace_estimates.append(np.mean(batch_traces))

        tr_mean = float(np.mean(trace_estimates))
        tr_std = float(np.std(trace_estimates))
        hess_results[cond_name] = {
            "trace_mean": tr_mean,
            "trace_std": tr_std,
        }
        print(f"  [{cond_name}] tr(H) = {tr_mean:.6f} ± {tr_std:.6f}")

    ol_tr = hess_results["OL"]["trace_mean"]
    print(f"\n  --- Hessian trace ratios vs OL ---")
    for cn in conditions:
        ratio = hess_results[cn]["trace_mean"] / ol_tr if ol_tr != 0 else 0
        print(f"  {cn:>6s}: {ratio:.3f}x OL")

    print(f"\n  Predicted E[ΔL] for random perturbation (E[ΔL] = ε²/(2d) · tr(H)):")
    for s_mag in [0.5, 1.0, 2.0, 3.0]:
        eps = s_mag * b1_std
        print(f"  s={s_mag}:")
        for cn in conditions:
            pred = (eps**2 / (2 * n_embd)) * hess_results[cn]["trace_mean"]
            print(f"    {cn:>6s}: {pred:+.6f}")

    # ==========================================================
    # PHASE 3: Gradient covariance spectral analysis
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Gradient covariance spectrum")
    print(f"{'='*60}")

    spectral_results = {}

    for cond_name in conditions:
        G = all_per_pos_grads[cond_name].numpy()
        G_c = G - G.mean(axis=0, keepdims=True)
        C = (G_c.T @ G_c) / (G.shape[0] - 1)

        eigvals = np.linalg.eigvalsh(C)[::-1]
        eigvals = np.maximum(eigvals, 0)
        total_var = eigvals.sum()

        p = eigvals / (total_var + 1e-30)
        p = p[p > 1e-30]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))

        cumvar = np.cumsum(eigvals) / total_var
        rank_50 = int(np.searchsorted(cumvar, 0.5)) + 1
        rank_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1

        spectral_results[cond_name] = {
            "effective_rank": eff_rank,
            "rank_50": rank_50,
            "rank_90": rank_90,
            "rank_95": rank_95,
            "top1_frac": float(eigvals[0] / total_var),
            "top5_frac": float(eigvals[:5].sum() / total_var),
            "top10_frac": float(eigvals[:10].sum() / total_var),
            "eigvals_top30": eigvals[:30].tolist(),
            "total_variance": float(total_var),
        }

        print(f"\n  [{cond_name}]")
        print(f"    Effective rank: {eff_rank:.1f} / {n_embd}")
        print(f"    Rank for 50%/90%/95% var: {rank_50}/{rank_90}/{rank_95}")
        print(f"    Top-1/5/10: {eigvals[0]/total_var:.4f} / "
              f"{eigvals[:5].sum()/total_var:.4f} / "
              f"{eigvals[:10].sum()/total_var:.4f}")

    # ==========================================================
    # PHASE 4: Self-knowledge probe and subspace alignment
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Self-knowledge subspace alignment")
    print(f"{'='*60}")

    # Collect CL model activations and residuals for probe training
    print("Collecting CL model activations for SK probe...")
    probe_gen = torch.Generator().manual_seed(seed + 100)
    n_probe_batches = 40
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(n_probe_batches)
    ]

    probe_acts = []
    probe_residuals = []

    with torch.no_grad():
        for pidx in probe_indices:
            x, y = make_batch(pidx)
            _, _, inter = model_closed(x, y, return_intermediates=True)
            src = inter[predict_from]
            tgt = inter[predict_to]
            pred = fwd_closed(src)
            res = tgt - pred
            probe_acts.append(
                inter[f"post_block{inject_after_block}"]
                .reshape(-1, n_embd).cpu())
            probe_residuals.append(res.reshape(-1, n_embd).cpu())

    probe_acts_np = torch.cat(probe_acts).numpy()
    probe_res_np = torch.cat(probe_residuals).numpy()
    n_probe = probe_acts_np.shape[0]
    n_probe_train = int(0.8 * n_probe)

    rng_probe = np.random.default_rng(seed + 99)
    perm = rng_probe.permutation(n_probe)
    ptr = perm[:n_probe_train]
    pte = perm[n_probe_train:]

    # Standardize activations for probe training
    mu_a = probe_acts_np[ptr].mean(axis=0, keepdims=True)
    sd_a = probe_acts_np[ptr].std(axis=0, keepdims=True) + 1e-6
    A_s = (probe_acts_np - mu_a) / sd_a

    print(f"Training SK probe ({n_probe_train} train, "
          f"{n_probe - n_probe_train} test)...")
    probe = nn.Linear(n_embd, n_embd).to(device)
    opt_probe = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Atr = torch.from_numpy(A_s[ptr].astype(np.float32)).to(device)
    Rtr = torch.from_numpy(probe_res_np[ptr].astype(np.float32)).to(device)
    bs_probe = min(4096, n_probe_train)

    for step in range(sk_probe_steps):
        idx_p = torch.randint(n_probe_train, (bs_probe,), device=device)
        pred_p = probe(Atr[idx_p])
        loss_p = F.mse_loss(pred_p, Rtr[idx_p])
        opt_probe.zero_grad()
        loss_p.backward()
        opt_probe.step()

    with torch.no_grad():
        Ate = torch.from_numpy(A_s[pte].astype(np.float32)).to(device)
        Rte_pred = probe(Ate).cpu().numpy()
    Rte = probe_res_np[pte]
    sk_r2 = float(1.0 - ((Rte_pred - Rte)**2).mean() / Rte.var())
    print(f"  SK probe R² = {sk_r2:.4f}")

    # Extract SK subspace in raw activation space
    W = probe.weight.data.cpu().numpy()  # (D, D)
    sd_inv = (1.0 / sd_a).squeeze().astype(np.float32)  # (D,)
    W_raw = W * sd_inv[np.newaxis, :]  # undo standardization
    _, S_w, Vt_w = np.linalg.svd(W_raw, full_matrices=False)
    V_w = Vt_w.T  # (D, D) — columns are SK directions in raw space

    sv_energy = S_w**2 / (S_w**2).sum()
    print(f"  Probe SV spectrum: top-5 explain "
          f"{sv_energy[:5].sum():.1%} of energy")

    # Alignment test across SK subspace dimensions
    print(f"\n  --- Alignment: gradient covariance vs SK subspace ---")

    alignment_results = {}
    n_random_baselines = 100

    for sk_k in sk_k_values:
        if sk_k > n_embd:
            continue
        sk_basis = V_w[:, :sk_k]  # (D, k)
        random_baseline = sk_k / n_embd

        print(f"\n  SK subspace dim = {sk_k} "
              f"(random baseline = {random_baseline:.4f}):")

        alignment_results[sk_k] = {}

        for cond_name in conditions:
            G = all_per_pos_grads[cond_name].numpy()
            G_c = G - G.mean(axis=0, keepdims=True)
            C = (G_c.T @ G_c) / (G.shape[0] - 1)

            C_sk = sk_basis.T @ C @ sk_basis
            energy_sk = float(np.trace(C_sk))
            energy_total = float(np.trace(C))
            alignment = energy_sk / energy_total if energy_total > 0 else 0

            random_aligns = []
            for _ in range(n_random_baselines):
                Q, _ = np.linalg.qr(
                    rng_probe.standard_normal((n_embd, sk_k)).astype(
                        np.float32))
                C_rand = Q.T @ C @ Q
                random_aligns.append(float(np.trace(C_rand) / energy_total))
            rand_mean = float(np.mean(random_aligns))
            rand_std = float(np.std(random_aligns))

            enrichment = alignment / rand_mean if rand_mean > 0 else 0

            alignment_results[sk_k][cond_name] = {
                "sk_alignment": alignment,
                "random_mean": rand_mean,
                "random_std": rand_std,
                "enrichment": enrichment,
            }

            print(f"    {cond_name:>6s}: {alignment:.4f} "
                  f"(random: {rand_mean:.4f} ± {rand_std:.4f}, "
                  f"{enrichment:.2f}x)")

    # ==========================================================
    # PHASE 5: Empirical perturbation validation
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 5: Empirical perturbation validation")
    print(f"{'='*60}")

    rng_pert = np.random.default_rng(seed + 300)
    perturb_dirs = []
    for _ in range(n_perturb_dirs):
        d = rng_pert.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs.append(d)

    s_values = [0.5, 1.0, 2.0, 3.0]
    n_emp_batches = min(10, n_eval_batches)
    emp_results = {}

    for cond_name, (model, use_inj) in conditions.items():
        emp_results[cond_name] = {}
        for s_mag in s_values:
            eps = s_mag * b1_std
            loss_deltas = []

            for di in range(n_perturb_dirs):
                dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
                perturbation = eps * dir_t

                batch_dL = []
                with torch.no_grad():
                    for bi in range(n_emp_batches):
                        x, y = make_batch(eval_indices[bi])
                        h_base = compute_h_base(model, x, use_inj)

                        loss_base = loss_from_h(model, h_base, y)
                        h_pert = h_base + perturbation.unsqueeze(0).unsqueeze(0)
                        loss_pert = loss_from_h(model, h_pert, y)

                        batch_dL.append((loss_pert - loss_base).item())

                loss_deltas.append(np.mean(batch_dL))

            emp_results[cond_name][s_mag] = {
                "mean_dL": float(np.mean(loss_deltas)),
                "mean_abs_dL": float(np.mean(np.abs(loss_deltas))),
                "std_dL": float(np.std(loss_deltas)),
            }

    print(f"\n  Empirical vs predicted loss changes:")
    print(f"  {'Cond':>6s} {'s':>4s} {'Emp E[ΔL]':>10s} {'Emp E[|ΔL|]':>12s}"
          f" {'Pred(Hess)':>11s} {'Pred(Grad)':>11s}")
    print(f"  {'-'*60}")
    for s_mag in s_values:
        eps = s_mag * b1_std
        for cn in conditions:
            emp = emp_results[cn][s_mag]
            pred_hess = ((eps**2) / (2 * n_embd)) * hess_results[cn]["trace_mean"]
            pred_grad = (eps / np.sqrt(n_embd)) * \
                grad_results[cn]["shared_sens_norm_mean"]
            print(f"  {cn:>6s} {s_mag:>4.1f} {emp['mean_dL']:>+10.6f}"
                  f" {emp['mean_abs_dL']:>12.6f}"
                  f" {pred_hess:>+11.6f} {pred_grad:>11.6f}")

    # Check linearity: does ΔL scale as ε² (Hessian-dominated) or ε (gradient)?
    print(f"\n  Linearity check (ΔL scaling with perturbation magnitude):")
    for cn in conditions:
        dLs = [emp_results[cn][s]["mean_dL"] for s in s_values]
        eps_vals = [s * b1_std for s in s_values]
        # If quadratic: ΔL/ε² should be constant
        ratios_quad = [dL / (eps**2) if eps > 0 else 0
                       for dL, eps in zip(dLs, eps_vals)]
        # If linear: |ΔL|/ε should be constant
        ratios_lin = [abs(dL) / eps if eps > 0 else 0
                      for dL, eps in zip(dLs, eps_vals)]
        quad_cv = np.std(ratios_quad) / (np.mean(ratios_quad) + 1e-30)
        lin_cv = np.std(ratios_lin) / (np.mean(ratios_lin) + 1e-30)
        print(f"    {cn:>6s}: ΔL/ε² CV={quad_cv:.3f}, |ΔL|/ε CV={lin_cv:.3f} "
              f"({'quadratic' if quad_cv < lin_cv else 'linear'} scaling)")

    # ==========================================================
    # SUMMARY
    # ==========================================================
    print(f"\n{'='*60}")
    print("JACOBIAN ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print("\n1. Loss gradient norms (shared perturbation sensitivity):")
    for cn in conditions:
        r = grad_results[cn]
        ratio = r["shared_sens_norm_mean"] / ol_shared
        print(f"   {cn:>6s}: ||s|| = {r['shared_sens_norm_mean']:.6f} "
              f"({ratio:.3f}x OL)")

    print(f"\n2. Hessian trace (loss curvature at injection point):")
    for cn in conditions:
        ratio = hess_results[cn]["trace_mean"] / ol_tr if ol_tr != 0 else 0
        print(f"   {cn:>6s}: tr(H) = {hess_results[cn]['trace_mean']:.6f} "
              f"({ratio:.3f}x OL)")

    print(f"\n3. Gradient covariance spectrum:")
    for cn in conditions:
        r = spectral_results[cn]
        print(f"   {cn:>6s}: eff_rank={r['effective_rank']:.1f}/{n_embd}, "
              f"top-5={r['top5_frac']:.1%}, top-10={r['top10_frac']:.1%}")

    sk_k_main = 50
    if sk_k_main in alignment_results:
        print(f"\n4. Self-knowledge alignment (top-{sk_k_main} SK subspace):")
        baseline = sk_k_main / n_embd
        print(f"   Random baseline: {baseline:.1%}")
        for cn in conditions:
            r = alignment_results[sk_k_main][cn]
            print(f"   {cn:>6s}: {r['sk_alignment']:.1%} "
                  f"({r['enrichment']:.2f}x random)")

    print(f"\n5. Quantitative robustness prediction (s=2.0):")
    emp_ol = emp_results["OL"][2.0]["mean_dL"]
    for cn in conditions:
        emp_dL = emp_results[cn][2.0]["mean_dL"]
        pred_hess = ((2.0 * b1_std)**2 / (2 * n_embd)) * \
            hess_results[cn]["trace_mean"]
        emp_ratio = emp_dL / emp_ol if abs(emp_ol) > 1e-10 else float("nan")
        pred_ratio = (hess_results[cn]["trace_mean"] / ol_tr
                      if abs(ol_tr) > 1e-10 else float("nan"))
        print(f"   {cn:>6s}: emp ΔL={emp_dL:+.6f} ({emp_ratio:.3f}x OL), "
              f"pred ΔL={pred_hess:+.6f} ({pred_ratio:.3f}x OL)")

    # ==========================================================
    # Save results
    # ==========================================================
    save_dir = os.path.join(ckpt_root, "jacobian_analysis")
    os.makedirs(save_dir, exist_ok=True)

    from a2a_forward.shared import NumpyEncoder
    output = {
        "config": {
            "n_tokens": n_tokens,
            "n_eval_batches": n_eval_batches,
            "n_hutchinson_vectors": n_hutchinson_vectors,
            "n_perturb_dirs": n_perturb_dirs,
            "sk_k_values": sk_k_values,
            "sk_probe_r2": sk_r2,
            "b1_activation_std": b1_std,
            "n_grad_store_batches": n_grad_store_batches,
            "n_hess_batches": n_hess_batches,
        },
        "gradient_analysis": grad_results,
        "hessian_trace": hess_results,
        "spectral_analysis": spectral_results,
        "sk_alignment": {
            str(k): v for k, v in alignment_results.items()
        },
        "sk_probe_sv_top20": S_w[:20].tolist(),
        "sk_probe_sv_energy_top20": sv_energy[:20].tolist(),
        "empirical_perturbation": {
            cn: {str(s): v for s, v in sv.items()}
            for cn, sv in emp_results.items()
        },
    }

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {save_dir}")

    return output


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
    result = a2a_jacobian_analysis.remote(
        n_tokens=n_tokens,
        predict_from=predict_from,
        predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )

    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)

    g = result["gradient_analysis"]
    h = result["hessian_trace"]
    s = result["spectral_analysis"]
    a = result["sk_alignment"]

    ol_s = g["OL"]["shared_sens_norm_mean"]
    ol_tr = h["OL"]["trace_mean"]

    for cn in ["OL", "CL+M", "CL-M"]:
        print(f"\n{cn}:")
        print(f"  ||grad||   = {g[cn]['shared_sens_norm_mean']:.6f} "
              f"({g[cn]['shared_sens_norm_mean']/ol_s:.3f}x OL)")
        print(f"  tr(H)      = {h[cn]['trace_mean']:.6f} "
              f"({h[cn]['trace_mean']/ol_tr:.3f}x OL)")
        print(f"  eff_rank   = {s[cn]['effective_rank']:.1f}")
        if "50" in a and cn in a["50"]:
            print(f"  SK align   = {a['50'][cn]['sk_alignment']:.4f} "
                  f"({a['50'][cn]['enrichment']:.2f}x random)")
