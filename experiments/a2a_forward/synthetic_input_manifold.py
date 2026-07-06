"""Idea A, manifold-matched rung: does CHEAP NONLINEAR structure close the
remaining `gaussian_perpos -> real` gap?

Companion to synthetic_input.py. That experiment found off-manifold-trained
forward models recover most of the on-manifold ceiling, with a residual gap
(update-R2: 0.055 at 1 block, 0.101 at 3 blocks) between the best Gaussian
sampler (gaussian_perpos: per-position mean+cov) and training on real
activations. That gap is exactly the manifold's *nonlinear* structure that
second moments miss. Here we add two cheap manifold models above perpos:

  gmm_global : fitted mixture of Gaussians (K-means centroids + per-cluster full
               covariance). Multimodal -> captures piecewise-nonlinear structure.
  mixup      : convex combinations of pairs of REAL activation sequences. A
               nonparametric near-manifold sampler (stays in the data convex
               hull; preserves positional + nonlinear structure for free).

We run ONLY the new conditions and reuse the IDENTICAL eval pipeline (same seed,
n_stat_batches, n_eval_batches) so real_eval reproduces byte-for-byte. This is
verified at runtime via the identity_full_cosine gate against the saved run.
The saved perpos/real numbers are loaded and printed alongside for a combined
ladder.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


NEW_CONDITIONS = ["gmm_global", "mixup"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=65536,
)
def manifold_experiment(
    ckpt_dir: str = "a2a_forward/transformer/P_10000000",
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 1000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    n_stat_batches: int = 60,     # MUST match synthetic_input.py for eval reprod.
    n_eval_batches: int = 40,     # MUST match synthetic_input.py
    cov_ridge: float = 1e-3,
    gmm_k: int = 64,
    kmeans_iters: int = 15,
    seed: int = 42,
):
    import os
    import glob
    import torch
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from_idx = -1 if predict_from == "post_embed" else int(
        predict_from.replace("post_block", ""))
    to_idx = int(predict_to.replace("post_block", ""))

    print(f"Manifold-matched rung on {device}")
    print(f"  Frozen main model: {n_layer}L {n_head}H {n_embd}D  ({ckpt_dir})")
    print(f"  Layer program f: {predict_from} -> {predict_to}")
    print(f"  New conditions: {NEW_CONDITIONS}")

    # --- Load data (identical to synthetic_input.py) ---
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
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # --- Load & freeze main model ---
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    sd = torch.load(os.path.join(DATA_DIR, ckpt_dir, "model.pt"),
                    map_location=device)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def get_tokens(split_data, bs=batch_size):
        ix = torch.randint(len(split_data) - block_size - 1, (bs,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        return x.to(device)

    @torch.no_grad()
    def real_pair(split_data, bs=batch_size):
        x = get_tokens(split_data, bs)
        _, _, inter = model(x, return_intermediates=True)
        return inter[predict_from], inter[predict_to]

    @torch.no_grad()
    def apply_f(a):
        for i in range(from_idx + 1, to_idx + 1):
            a = model.transformer.h[i](a)
        return a

    # --- Estimate stats: IDENTICAL RNG-consuming path to synthetic_input.py so
    #     real_eval below reproduces exactly. Do NOT add RNG draws before eval. ---
    print("\nEstimating activation statistics (reproducing eval pipeline)...")
    pool = []
    with torch.no_grad():
        for _ in range(n_stat_batches):
            a_i, _ = real_pair(val_data)
            pool.append(a_i.cpu())
    pool = torch.cat(pool, dim=0)
    N, T, C = pool.shape

    # gaussian_perpos anchor stats were computed in the original run here too, but
    # they draw no RNG (torch.cov / cholesky are deterministic), so recomputing or
    # skipping them does not shift the RNG state. We skip them.

    # --- Fixed held-out eval sets: IDENTICAL construction (same RNG order) ---
    print("Building held-out eval sets...")
    with torch.no_grad():
        real_eval = [real_pair(val_data) for _ in range(n_eval_batches)]
        iso_eval, perpos_eval = [], []
        # NOTE: original built iso/perpos eval via sample_synthetic which draws
        # torch.randn on device. To keep RNG identical we must reproduce those
        # draws in the same order. We replicate the exact sampler calls here.
        mu_global = pool.reshape(-1, C).mean(0).to(device)
        sigma_iso = pool.reshape(-1, C).var(0).mean().sqrt().to(device)
        # per-position cholesky for perpos eval reproduction
        def cholesky_ridge(cov):
            eye = torch.eye(cov.shape[-1])
            jitter = cov_ridge * cov.diagonal(dim1=-2, dim2=-1).mean(
                -1, keepdim=True).unsqueeze(-1)
            for _ in range(6):
                try:
                    return torch.linalg.cholesky(cov + jitter * eye)
                except Exception:
                    jitter = jitter * 10
            return torch.linalg.cholesky(cov + 1e-1 * eye)
        mu_pos = pool.mean(0).to(device)
        L_pos = torch.empty(T, C, C)
        for t in range(T):
            L_pos[t] = cholesky_ridge(torch.cov(pool[:, t, :].T))
        L_pos = L_pos.to(device)
        for _ in range(n_eval_batches // 2):
            z = torch.randn(batch_size, T, C, device=device)
            xi = mu_global + sigma_iso * z
            iso_eval.append((xi, apply_f(xi)))
            z2 = torch.randn(batch_size, T, C, device=device)
            xp = mu_pos + torch.einsum("btc,tdc->btd", z2, L_pos)
            perpos_eval.append((xp, apply_f(xp)))

    id_full_cos = []
    with torch.no_grad():
        for a_i, a_j in real_eval:
            id_full_cos.append(torch.nn.functional.cosine_similarity(
                a_i, a_j, dim=-1).mean().item())
    identity_full_cosine = float(np.mean(id_full_cos))
    print(f"  identity_full_cosine = {identity_full_cosine:.4f}  "
          f"(GATE: must match saved run to validate comparison)")

    # ===== Everything below adds NEW RNG draws; safe now that eval is built =====

    # --- Fit GMM (K-means + per-cluster full covariance) on the real pool ---
    print(f"\nFitting GMM (K={gmm_k} via K-means, {kmeans_iters} iters)...")
    X = pool.reshape(-1, C).to(device)
    perm = torch.randperm(X.shape[0], device=device)[:gmm_k]
    centroids = X[perm].clone()
    for it in range(kmeans_iters):
        d = torch.cdist(X, centroids)
        assign = d.argmin(1)
        for k in range(gmm_k):
            m = assign == k
            if m.any():
                centroids[k] = X[m].mean(0)
    d = torch.cdist(X, centroids)
    assign = d.argmin(1)
    gmm_means, gmm_L, gmm_w = [], [], []
    for k in range(gmm_k):
        m = assign == k
        cnt = int(m.sum())
        gmm_w.append(cnt)
        gmm_means.append(centroids[k])
        if cnt > C:
            L = cholesky_ridge(torch.cov(X[m].T).cpu()).to(device)
        else:  # tiny cluster: isotropic fallback
            L = (sigma_iso * torch.eye(C, device=device))
        gmm_L.append(L)
    gmm_means = torch.stack(gmm_means)          # (K, C)
    gmm_L = torch.stack(gmm_L)                   # (K, C, C)
    gmm_w = torch.tensor(gmm_w, device=device, dtype=torch.float)
    gmm_w = gmm_w / gmm_w.sum()
    print(f"  cluster sizes: min={int(gmm_w.min()*X.shape[0])} "
          f"max={int(gmm_w.max()*X.shape[0])}, "
          f"nonempty={(gmm_w>0).sum().item()}/{gmm_k}")

    def sample_new(condition, bs=batch_size):
        if condition == "gmm_global":
            comp = torch.multinomial(gmm_w, bs * T, replacement=True)
            z = torch.randn(bs * T, C, device=device)
            x = gmm_means[comp] + torch.einsum(
                "nc,ndc->nd", z, gmm_L[comp])
            return x.view(bs, T, C)
        if condition == "mixup":
            a, _ = real_pair(train_data, bs)
            b, _ = real_pair(train_data, bs)
            lam = torch.rand(bs, 1, 1, device=device)
            return lam * a + (1 - lam) * b
        raise ValueError(condition)

    @torch.no_grad()
    def evaluate(fwd, eval_pairs):
        fwd.eval()
        fc, fmse, tvar, uc, umse, uvar = [], [], [], [], [], []
        for a_i, a_j in eval_pairs:
            pred = fwd(a_i)
            fc.append(torch.nn.functional.cosine_similarity(
                pred, a_j, dim=-1).mean().item())
            fmse.append(torch.nn.functional.mse_loss(pred, a_j).item())
            tvar.append(a_j.var().item())
            upd_t, upd_p = a_j - a_i, pred - a_i
            uc.append(torch.nn.functional.cosine_similarity(
                upd_p, upd_t, dim=-1).mean().item())
            umse.append(torch.nn.functional.mse_loss(upd_p, upd_t).item())
            uvar.append(upd_t.var().item())
        fm, fv = float(np.mean(fmse)), float(np.mean(tvar))
        um, uv = float(np.mean(umse)), float(np.mean(uvar))
        return {
            "full_cosine": float(np.mean(fc)),
            "full_r2": 1.0 - fm / fv if fv > 0 else 0.0,
            "update_cosine": float(np.mean(uc)),
            "update_r2": 1.0 - um / uv if uv > 0 else 0.0,
        }

    # --- Train one forward model per NEW condition ---
    results = {}
    for condition in NEW_CONDITIONS:
        print(f"\n=== Condition: {condition} ===")
        torch.manual_seed(seed)  # identical fwd init to synthetic_input.py runs
        fwd = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        opt = torch.optim.AdamW(fwd.parameters(), lr=fwd_lr, weight_decay=0.01)
        for step in range(n_steps):
            fwd.train()
            a_i = sample_new(condition)
            with torch.no_grad():
                a_j = apply_f(a_i)
            pred = fwd(a_i)
            loss = torch.nn.functional.mse_loss(pred, a_j)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
            opt.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                m = evaluate(fwd, real_eval)
                print(f"  step {step:6d}: train_mse={loss.item():.4f} | "
                      f"REAL full_cos={m['full_cosine']:.4f} "
                      f"upd_cos={m['update_cosine']:.4f} "
                      f"upd_r2={m['update_r2']:.4f}")
        results[condition] = {
            "eval_real": evaluate(fwd, real_eval),
            "eval_gaussian_iso": evaluate(fwd, iso_eval),
            "eval_gaussian_perpos": evaluate(fwd, perpos_eval),
        }

    # --- Load saved ladder for combined printout ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    prior_path = (f"{DATA_DIR}/a2a_forward/synthetic_input/"
                  f"{gap_tag}/fwd_L{fwd_n_layer}/results.json")
    prior = None
    if os.path.exists(prior_path):
        prior = json.load(open(prior_path))

    print("\n\n=== COMBINED LADDER: metric on held-out REAL pairs ===")
    print(f"{'condition':<18} {'full_cos':>9} {'upd_cos':>9} "
          f"{'upd_r2':>9} {'full_r2':>9}   src")
    rows = []
    if prior is not None:
        gate = prior["identity_full_cosine"]
        match = abs(gate - identity_full_cosine) < 1e-3
        print(f"  [GATE] saved identity={gate:.4f} vs "
              f"here={identity_full_cosine:.4f}  "
              f"-> {'MATCH (comparison valid)' if match else 'MISMATCH!!'}")
        for c, r in prior["conditions"].items():
            rows.append((c, r["eval_real"], "prior"))
    for c in NEW_CONDITIONS:
        rows.append((c, results[c]["eval_real"], "NEW"))
    # order: prior gaussian ladder, new, then perturb/real if present
    for c, r, src in rows:
        print(f"{c:<18} {r['full_cosine']:>9.4f} {r['update_cosine']:>9.4f} "
              f"{r['update_r2']:>9.4f} {r['full_r2']:>9.4f}   {src}")

    out = {
        "config": {
            "predict_from": predict_from, "predict_to": predict_to,
            "fwd_n_layer": fwd_n_layer, "n_steps": n_steps,
            "gmm_k": gmm_k, "seed": seed,
        },
        "identity_full_cosine": identity_full_cosine,
        "identity_gate_saved": prior["identity_full_cosine"] if prior else None,
        "new_conditions": results,
    }
    save_dir = (f"{DATA_DIR}/a2a_forward/synthetic_input/"
                f"{gap_tag}/fwd_L{fwd_n_layer}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results_manifold.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/results_manifold.json")
    return out


@app.local_entrypoint()
def main(
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    n_steps: int = 10_000,
):
    out = manifold_experiment.remote(
        predict_from=predict_from, predict_to=predict_to,
        fwd_n_layer=fwd_n_layer, n_steps=n_steps,
    )
    print("\nDone.")
