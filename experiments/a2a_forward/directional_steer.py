"""Directional causal steering of self-knowledge.

Tests whether the closed-loop model uses its *directional* self-knowledge to
modulate how it processes the injected prediction. Extends the scalar steering
test (novelty_steer.py, Test 2) which found no novelty-gated reliance along
the residual-norm direction.

The self-knowledge is 6x more directional than scalar (vector probe DR2=+0.18
vs scalar DR2=+0.03), and residual direction clusters organize injection help
5x more than norm octiles (eta2=0.0017 vs 0.0003). This test asks whether that
directional structure is *causally* used.

Design:
1. K-means on residual unit vectors -> k cluster centroids (types of fwd error)
2. Ridge-regression probe W: post_block1 -> residual vector (multivariate)
3. Per-centroid steering direction v_j = normalize(W @ c_j) -- the post_block1
   direction whose perturbation moves the *predicted* residual toward centroid j
4. Steer by +/-s*sigma_j*v_j in both inj and no-inj passes (direct effects
   cancel)
5. Measure help = loss_no_inj - loss_inj, broken down by cluster membership
6. Effect matrix M[j,c] = slope of help_c vs s for direction j
7. Diagonality test: is M[j,j] systematically different from M[j,c!=j]?

Predictions:
- Outcome A (strong positive): M is approximately diagonal -- steering along
  direction j primarily changes help for tokens in cluster j.
- Outcome B (intermediate): Effects exist but are not direction-specific.
- Outcome C (negative): No effects beyond random controls.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=65536,
)
def a2a_directional_steer(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    n_fit_batches: int = 30,
    n_eval_batches: int = 20,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    k_clusters: int = 8,
    ridge_lambda: float = 0.1,
    n_random_dirs: int = 3,
    seed: int = 0,
):
    import os
    import glob
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A directional steering on {device}")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

    # --- Data ---
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
    g = torch.Generator().manual_seed(seed)

    def make_batches(n):
        out = []
        for _ in range(n):
            ix = torch.randint(
                len(val_data) - block_size - 1, (batch_size,), generator=g)
            out.append(torch.stack([val_data[i:i + block_size] for i in ix]))
        return out

    fit_batches = make_batches(n_fit_batches)
    eval_batches = make_batches(n_eval_batches)

    # --- Models ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_dir = (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading closed-loop from {model_dir}")
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"), map_location=device,
        weights_only=True))
    model.eval()
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
        block_size=block_size).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device,
        weights_only=True))
    fwd_model.eval()
    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(model_dir, "gate.pt"), map_location=device,
        weights_only=True))
    gate.eval()
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    def per_tok_loss(logits, tgt):
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    # =============================================
    # PHASE 1: Fit probes
    # =============================================
    print("\nPhase 1: Collecting fit data...")
    pb1_all, res_all, rn_all = [], [], []
    with torch.no_grad():
        for x in fit_batches:
            x = x.to(device)
            _, _, inter = model(x, return_intermediates=True)
            pb1 = inter["post_block1"][:, :-1]
            pred = fwd_model(inter[predict_from])
            residual = (inter[predict_to] - pred)[:, :-1]
            pb1_all.append(pb1.reshape(-1, n_embd).cpu())
            res_all.append(residual.reshape(-1, n_embd).cpu())
            rn_all.append(residual.norm(dim=-1).reshape(-1).cpu())
    pb1_all = torch.cat(pb1_all)
    res_all = torch.cat(res_all)
    rn_all = torch.cat(rn_all)
    N_fit = pb1_all.shape[0]
    print(f"  Fit data: {N_fit:,} tokens")

    # K-means on unit-normalized residuals
    print(f"  K-means (k={k_clusters})...")
    unit_np = res_all.numpy()
    unit_np = unit_np / (np.linalg.norm(unit_np, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)
    centroids = unit_np[rng.choice(N_fit, k_clusters, replace=False)].copy()
    for _ in range(30):
        dists = -2.0 * unit_np @ centroids.T + (centroids ** 2).sum(1)[None, :]
        assign = dists.argmin(1)
        for ci in range(k_clusters):
            m = assign == ci
            if m.sum() > 0:
                centroids[ci] = unit_np[m].mean(0)
                centroids[ci] /= np.linalg.norm(centroids[ci]) + 1e-8
    centroids_t = torch.from_numpy(centroids.astype(np.float32)).to(device)

    # Ridge regression: post_block1 -> residual vector
    print(f"  Ridge regression (lambda={ridge_lambda})...")
    perm = rng.permutation(N_fit)
    n_tr = int(0.8 * N_fit)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    X_tr = pb1_all[tr_idx].to(device)
    Y_tr = res_all[tr_idx].to(device)
    X_te = pb1_all[te_idx].to(device)
    Y_te = res_all[te_idx].to(device)
    X_mean = X_tr.mean(0)
    Y_mean = Y_tr.mean(0)
    Xc = X_tr - X_mean
    Yc = Y_tr - Y_mean
    W = torch.linalg.solve(
        Xc.T @ Xc + ridge_lambda * torch.eye(n_embd, device=device),
        Xc.T @ Yc)

    # Full vector probe R2
    Y_pred = (X_te - X_mean) @ W + Y_mean
    ss_res = ((Y_te - Y_pred) ** 2).sum().item()
    ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum().item()
    probe_r2_full = 1.0 - ss_res / ss_tot
    print(f"  Full vector probe R2 = {probe_r2_full:.4f}")

    # Per-centroid R2 and steering directions
    steer_dirs = {}
    probe_r2_per = {}
    dir_vecs, dir_names = [], []
    for ci in range(k_clusters):
        c = centroids_t[ci]
        v = W @ c
        v_hat = v / (v.norm() + 1e-8)
        sigma = float((Xc @ v_hat).std())
        proj_true = (Y_te - Y_mean) @ c
        proj_pred = (X_te - X_mean) @ W @ c
        r2_c = 1.0 - float(((proj_true - proj_pred) ** 2).sum()
                            / ((proj_true - proj_true.mean()) ** 2).sum())
        probe_r2_per[f"cluster_{ci}"] = r2_c
        steer_dirs[f"cluster_{ci}"] = (v_hat.cpu().numpy(), sigma)
        dir_vecs.append(v_hat.cpu().numpy())
        dir_names.append(f"cluster_{ci}")
        print(f"    cluster_{ci}: R2={r2_c:.4f}, sigma={sigma:.3f}")

    # Cross-influence: cosine between W@c_j directions
    Wc_vecs = [(W @ centroids_t[ci]).cpu().numpy() for ci in range(k_clusters)]
    Wc_norms = [v / (np.linalg.norm(v) + 1e-8) for v in Wc_vecs]
    cross_cos = np.array([[np.dot(Wc_norms[i], Wc_norms[j])
                           for j in range(k_clusters)]
                          for i in range(k_clusters)])
    print(f"  Cross-cosine of W@c directions (mean off-diag): "
          f"{cross_cos[~np.eye(k_clusters, dtype=bool)].mean():.3f}")

    # Scalar novelty (residual norm) probe
    print("  Fitting scalar novelty direction...")
    pb1_np = pb1_all.numpy()
    rn_np = rn_all.numpy()
    rn_tr, rn_te = rn_np[tr_idx], rn_np[te_idx]
    rn_mu, rn_sd = rn_tr.mean(), rn_tr.std() + 1e-6
    rn_tr_z = (rn_tr - rn_mu) / rn_sd
    rn_te_z = (rn_te - rn_mu) / rn_sd
    probe_nov = nn.Linear(n_embd, 1).to(device)
    opt = torch.optim.Adam(probe_nov.parameters(), lr=1e-3)
    Xtr_d = torch.from_numpy(pb1_np[tr_idx]).float().to(device)
    ttr = torch.from_numpy(rn_tr_z.astype(np.float32)).to(device)
    bs = min(4096, n_tr)
    for _ in range(800):
        idx = torch.randint(n_tr, (bs,), device=device)
        loss = F.mse_loss(probe_nov(Xtr_d[idx]).squeeze(-1), ttr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_nov = probe_nov(
            torch.from_numpy(pb1_np[te_idx]).float().to(device)
        ).squeeze(-1).cpu().numpy()
    r2_nov = float(1.0 - ((pred_nov - rn_te_z) ** 2).mean() / rn_te_z.var())
    w_nov = probe_nov.weight.detach().squeeze(0).cpu().numpy()
    v_nov = w_nov / (np.linalg.norm(w_nov) + 1e-8)
    sig_nov = float((pb1_np @ v_nov).std())
    steer_dirs["novelty"] = (v_nov, sig_nov)
    dir_vecs.append(v_nov)
    dir_names.append("novelty")
    probe_r2_per["novelty"] = r2_nov
    print(f"    novelty: R2={r2_nov:.4f}, sigma={sig_nov:.3f}")

    # Random controls
    for ri in range(n_random_dirs):
        r = rng.standard_normal(n_embd).astype(np.float32)
        r /= np.linalg.norm(r)
        sig_r = float((pb1_np @ r).std())
        steer_dirs[f"random_{ri}"] = (r, sig_r)
        dir_vecs.append(r)
        dir_names.append(f"random_{ri}")

    # Free fit data
    del pb1_all, res_all, rn_all, unit_np, X_tr, Y_tr, X_te, Y_te, Xc, Yc
    del Xtr_d
    torch.cuda.empty_cache()

    # Pairwise cosine of all steering directions
    n_dirs = len(dir_vecs)
    cos_matrix = np.array([[float(np.dot(dir_vecs[i], dir_vecs[j]))
                            for j in range(n_dirs)]
                           for i in range(n_dirs)])
    print("\n  Pairwise cosine (cluster dirs + novelty):")
    for i in range(k_clusters + 1):
        row = [f"{cos_matrix[i, j]:+.3f}" for j in range(k_clusters + 1)]
        print(f"    {dir_names[i]:>10s}: {' '.join(row)}")

    # =============================================
    # PHASE 2: Baseline eval (s=0)
    # =============================================
    print(f"\nPhase 2: Baseline eval (s=0, {n_eval_batches} batches)...")
    base_help_all, base_res_all = [], []
    base_ent_all, base_maxa_all = [], []
    with torch.no_grad():
        for bi, x in enumerate(eval_batches):
            x = x.to(device)
            tgt = x[:, 1:]
            logits_inj, _ = model(
                x, cerebellar_fn=lambda act: gate(fwd_model(act)),
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block)
            logits_no, _, inter = model(x, return_intermediates=True)
            l_inj = per_tok_loss(logits_inj, tgt)
            l_no = per_tok_loss(logits_no, tgt)
            base_help_all.append((l_no - l_inj).reshape(-1).cpu())
            pred = fwd_model(inter[predict_from])
            residual = (inter[predict_to] - pred)[:, :-1]
            base_res_all.append(residual.reshape(-1, n_embd).cpu())
            # Block-1 attention features for cluster characterization
            block1 = model.transformer.h[1]
            B, T, C = inter["post_block0"].size()
            h = block1.ln_1(inter["post_block0"])
            qkv = block1.attn.c_attn(h)
            q, k_, v_ = qkv.split(block1.attn.n_embd, dim=2)
            nh, hd = block1.attn.n_head, block1.attn.head_dim
            q = q.view(B, T, nh, hd).transpose(1, 2)
            k_ = k_.view(B, T, nh, hd).transpose(1, 2)
            att = (q @ k_.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            att = att.masked_fill(
                block1.attn.bias[:, :, :T, :T] == 0, float("-inf"))
            aw = F.softmax(att, dim=-1).mean(1)
            ent = -(aw * (aw + 1e-10).log()).sum(-1)[:, :-1]
            maxa = aw.max(-1).values[:, :-1]
            base_ent_all.append(ent.reshape(-1).cpu())
            base_maxa_all.append(maxa.reshape(-1).cpu())
            if bi % 5 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    base_help = torch.cat(base_help_all).numpy()
    base_res = torch.cat(base_res_all).numpy()
    base_ent = torch.cat(base_ent_all).numpy()
    base_maxa = torch.cat(base_maxa_all).numpy()
    N_eval = len(base_help)
    print(f"  Eval tokens: {N_eval:,}, mean help: {base_help.mean():+.5f}")

    # Assign clusters (fixed for all steering conditions)
    base_unit = base_res / (
        np.linalg.norm(base_res, axis=1, keepdims=True) + 1e-8)
    dists = -2.0 * base_unit @ centroids.T + (centroids ** 2).sum(1)[None, :]
    cluster_assign = dists.argmin(1)
    cluster_counts = np.array([(cluster_assign == ci).sum()
                               for ci in range(k_clusters)])

    # Cluster characterization
    print("\n  Cluster characterization:")
    cluster_char = {}
    for ci in range(k_clusters):
        m = cluster_assign == ci
        if m.sum() == 0:
            continue
        cluster_char[f"cluster_{ci}"] = {
            "mean_help": float(base_help[m].mean()),
            "mean_norm": float(np.linalg.norm(base_res[m], axis=1).mean()),
            "mean_attn_entropy": float(base_ent[m].mean()),
            "frac_focused": float((base_maxa[m] > 0.5).mean()),
            "n": int(m.sum()),
        }
        r = cluster_char[f"cluster_{ci}"]
        print(f"    cluster_{ci}: help={r['mean_help']:+.5f} "
              f"norm={r['mean_norm']:.3f} ent={r['mean_attn_entropy']:.2f} "
              f"foc={r['frac_focused']:.2f} n={r['n']:,}")

    del base_res, base_unit, base_ent, base_maxa
    torch.cuda.empty_cache()

    # =============================================
    # PHASE 3: Steering sweep
    # =============================================
    s_grid = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    all_dir_names = ([f"cluster_{ci}" for ci in range(k_clusters)]
                     + ["novelty"]
                     + [f"random_{ri}" for ri in range(n_random_dirs)])

    # help_data[dir_name][s] -> per-cluster mean help (k_clusters,)
    help_data = {dn: {} for dn in all_dir_names}

    # s=0 baseline (shared across all directions)
    s0_help = np.array([
        base_help[cluster_assign == ci].mean()
        if cluster_counts[ci] > 0 else 0.0
        for ci in range(k_clusters)])
    for dn in all_dir_names:
        help_data[dn][0.0] = s0_help.copy()

    n_nonzero = sum(1 for s in s_grid if s != 0.0)
    total_sweeps = len(all_dir_names) * n_nonzero
    print(f"\nPhase 3: Steering sweep ({len(all_dir_names)} dirs x "
          f"{n_nonzero} nonzero s = {total_sweeps} conditions)...")
    sweep_done = 0
    for di, dir_name in enumerate(all_dir_names):
        v, sigma = steer_dirs[dir_name]
        for s in s_grid:
            if s == 0.0:
                continue
            steer_vec = (s * sigma) * v
            steer_t = torch.from_numpy(steer_vec).to(device)
            clu_sum = np.zeros(k_clusters)
            clu_cnt = np.zeros(k_clusters)
            tok_offset = 0
            with torch.no_grad():
                for x in eval_batches:
                    x = x.to(device)
                    tgt = x[:, 1:]
                    n_tok = tgt.numel()

                    def inj_fn(act, _st=steer_t):
                        return gate(fwd_model(act)) + _st

                    def noinj_fn(act, _st=steer_t):
                        return torch.zeros_like(
                            gate(fwd_model(act))) + _st

                    logits_inj, _ = model(
                        x, cerebellar_fn=inj_fn,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block)
                    logits_no, _ = model(
                        x, cerebellar_fn=noinj_fn,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block)
                    help_tok = (per_tok_loss(logits_no, tgt)
                                - per_tok_loss(logits_inj, tgt)
                                ).reshape(-1).cpu().numpy()
                    ca = cluster_assign[tok_offset:tok_offset + n_tok]
                    for ci in range(k_clusters):
                        m = ca == ci
                        if m.sum() > 0:
                            clu_sum[ci] += help_tok[m].sum()
                            clu_cnt[ci] += m.sum()
                    tok_offset += n_tok

            per_cluster = np.where(clu_cnt > 0, clu_sum / clu_cnt, 0.0)
            help_data[dir_name][s] = per_cluster
            sweep_done += 1
        print(f"  [{di+1}/{len(all_dir_names)}] {dir_name} done "
              f"({sweep_done}/{total_sweeps})")

    # =============================================
    # PHASE 4: Analysis
    # =============================================
    print("\nPhase 4: Analysis...")
    s_arr = np.array(s_grid)

    def compute_slope(dir_name, ci):
        y = np.array([help_data[dir_name][s][ci] for s in s_grid])
        return float(np.polyfit(s_arr, y, 1)[0])

    # Effect matrix
    effect_matrix = {}
    for dn in all_dir_names:
        effect_matrix[dn] = {
            f"cluster_{ci}": compute_slope(dn, ci)
            for ci in range(k_clusters)}

    # Print effect matrix
    print("\n=== Effect matrix M[dir,cluster] = slope d(help)/ds ===")
    hdr = f"{'direction':>12s}" + "".join(f"  clu{ci}" for ci in range(k_clusters))
    print(hdr)
    for dn in all_dir_names:
        row = f"{dn:>12s}" + "".join(
            f" {effect_matrix[dn][f'cluster_{ci}']:+.4f}"
            for ci in range(k_clusters))
        match = ""
        if dn.startswith("cluster_"):
            ci = int(dn.split("_")[1])
            match = f"  <-- diag={effect_matrix[dn][f'cluster_{ci}']:+.4f}"
        print(row + match)

    # Diagonality metrics (cluster directions only)
    M = np.array([[effect_matrix[f"cluster_{j}"][f"cluster_{c}"]
                   for c in range(k_clusters)]
                  for j in range(k_clusters)])
    diag_vals = np.diag(M)
    off_vals = M[~np.eye(k_clusters, dtype=bool)]
    mean_abs_diag = float(np.abs(diag_vals).mean())
    mean_abs_off = float(np.abs(off_vals).mean())
    diag_ratio = float(mean_abs_diag / (mean_abs_off + 1e-10))
    diag_enrichment = float(
        np.abs(diag_vals).sum() / (np.abs(M).sum() + 1e-10))
    uniform_enrichment = 1.0 / k_clusters

    per_dir_sel = {}
    for ci in range(k_clusters):
        row_abs = np.abs(M[ci, :])
        per_dir_sel[f"cluster_{ci}"] = float(
            row_abs[ci] / (row_abs.mean() + 1e-10))

    # Signed diagonal test: are diagonal elements systematically positive or
    # negative relative to off-diagonal?
    mean_diag_signed = float(diag_vals.mean())
    mean_off_signed = float(off_vals.mean())

    print(f"\n=== Diagonality metrics ===")
    print(f"  Mean |diagonal|:     {mean_abs_diag:.6f}")
    print(f"  Mean |off-diagonal|: {mean_abs_off:.6f}")
    print(f"  |diag|/|off| ratio:  {diag_ratio:.3f} "
          f"(1.0 = no enrichment)")
    print(f"  Diagonal enrichment: {diag_enrichment:.4f} "
          f"(uniform: {uniform_enrichment:.4f})")
    print(f"  Signed: diag={mean_diag_signed:+.6f}, "
          f"off={mean_off_signed:+.6f}")
    print(f"\n  Per-direction selectivity (|M[j,j]|/mean|M[j,:]|):")
    for ci in range(k_clusters):
        print(f"    cluster_{ci}: {per_dir_sel[f'cluster_{ci}']:.3f}")

    # Controls
    nov_slopes = [effect_matrix["novelty"][f"cluster_{ci}"]
                  for ci in range(k_clusters)]
    rand_slopes = [[effect_matrix[f"random_{ri}"][f"cluster_{ci}"]
                    for ci in range(k_clusters)]
                   for ri in range(n_random_dirs)]
    print(f"\n=== Controls ===")
    print(f"  Novelty slopes:  "
          f"{' '.join(f'{v:+.4f}' for v in nov_slopes)}  "
          f"(std={np.std(nov_slopes):.5f})")
    for ri in range(n_random_dirs):
        print(f"  Random_{ri} slopes: "
              f"{' '.join(f'{v:+.4f}' for v in rand_slopes[ri])}  "
              f"(std={np.std(rand_slopes[ri]):.5f})")

    # Overall help at each s for each direction (sanity check)
    overall_help = {}
    for dn in all_dir_names:
        overall_help[dn] = [
            float(sum(help_data[dn][s][ci] * cluster_counts[ci]
                      for ci in range(k_clusters))
                  / cluster_counts.sum())
            for s in s_grid]

    # Full per-cluster s-curves
    help_curves = {
        dn: {f"cluster_{ci}": [float(help_data[dn][s][ci]) for s in s_grid]
             for ci in range(k_clusters)}
        for dn in all_dir_names}

    results = {
        "probe_r2_full_vector": probe_r2_full,
        "probe_r2_per_centroid": probe_r2_per,
        "cross_cosine_Wc": cross_cos.tolist(),
        "cluster_characterization": cluster_char,
        "cluster_counts": cluster_counts.tolist(),
        "pairwise_cosine_dirs": {
            dir_names[i]: {dir_names[j]: float(cos_matrix[i, j])
                           for j in range(n_dirs)}
            for i in range(n_dirs)},
        "s_grid": s_grid,
        "effect_matrix": effect_matrix,
        "diagonality": {
            "mean_abs_diagonal": mean_abs_diag,
            "mean_abs_off_diagonal": mean_abs_off,
            "diag_to_off_ratio": diag_ratio,
            "diagonal_enrichment": diag_enrichment,
            "uniform_enrichment": uniform_enrichment,
            "mean_signed_diagonal": mean_diag_signed,
            "mean_signed_off_diagonal": mean_off_signed,
            "per_direction_selectivity": per_dir_sel,
        },
        "controls": {
            "novelty_slopes": nov_slopes,
            "novelty_slope_std": float(np.std(nov_slopes)),
            "random_slopes": rand_slopes,
        },
        "overall_help_curves": overall_help,
        "help_curves": help_curves,
    }

    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "directional_steer_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {save_path}")
    return results


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
):
    result = a2a_directional_steer.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A directional steering complete:")
    diag = result["diagonality"]
    print(f"  Full vector probe R2: {result['probe_r2_full_vector']:.4f}")
    print(f"  |diag|/|off| ratio:   {diag['diag_to_off_ratio']:.3f}")
    print(f"  Diagonal enrichment:  {diag['diagonal_enrichment']:.4f} "
          f"(uniform: {diag['uniform_enrichment']:.4f})")
    print(f"  Signed diag: {diag['mean_signed_diagonal']:+.6f}, "
          f"off: {diag['mean_signed_off_diagonal']:+.6f}")
    print("  Per-direction selectivity:")
    for k, v in diag["per_direction_selectivity"].items():
        print(f"    {k}: {v:.3f}")
