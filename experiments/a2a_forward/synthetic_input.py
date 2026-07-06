"""Idea A: can a forward self-model learn a frozen layer's *program* from
off-manifold inputs?

A transformer block (or stack of blocks) is a deterministic function f. We can
evaluate it on ANY (B, T, C) tensor, not just the activations the main model
naturally visits. This experiment trains fresh forward models on synthetic
input -> f(input) pairs drawn from a ladder of input distributions (from
isotropic noise up to real activations) and evaluates every model on the SAME
held-out REAL (a_i, a_j) pairs.

Core question (from the friend's messages / diagram): if the training inputs are
off-manifold (and hence f's outputs are off-manifold too), does the self-model
still predict correctly for on-manifold inputs? I.e. can you learn the "program"
the layers compute without on-manifold inputs?

Sharpening: a block is near-identity (block(x) = x + small update), so
full-activation cosine is lenient (predicting identity already scores high). We
also report UPDATE-space metrics on Delta = a_j - a_i (the block's genuine
contribution) -- the metric that says whether the model learned the *program*,
not just the residual-stream carry-through.

Main model is FROZEN (reused 29M checkpoint). Only the forward models train.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


# Ladder of synthetic-input conditions. Mean is matched for every Gaussian
# variant; only the covariance structure changes, so the ladder isolates "how
# much manifold structure must the training inputs carry".
CONDITIONS = [
    "gaussian_iso",     # isotropic: matched mean + scalar variance (crudest)
    "gaussian_global",  # matched global mean + full covariance
    "gaussian_perpos",  # matched per-position mean + per-position covariance
    "perturb_1.0",      # real activation + noise at 1.0 * ||a|| (near-manifold)
    "perturb_0.5",      # real activation + noise at 0.5 * ||a||
    "real",             # ceiling: train on real activations (on-manifold)
]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=65536,
)
def synthetic_input_experiment(
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
    n_stat_batches: int = 60,     # batches used to estimate activation stats
    n_eval_batches: int = 40,     # held-out real pairs for the money metric
    cov_ridge: float = 1e-3,      # relative ridge for covariance Cholesky
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

    print(f"Synthetic-input experiment on {device}")
    print(f"  Frozen main model: {n_layer}L {n_head}H {n_embd}D  ({ckpt_dir})")
    print(f"  Layer program f: {predict_from} -> {predict_to} "
          f"(blocks {from_idx + 1}..{to_idx})")
    print(f"  Forward model: {fwd_n_layer}L transformer")
    print(f"  Conditions: {CONDITIONS}")

    # --- Load data ---
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
    print(f"Loaded {len(data):,} tokens (vocab_size={vocab_size})")

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
        """Real (a_i, a_j) from fresh tokens via the frozen main model."""
        x = get_tokens(split_data, bs)
        _, _, inter = model(x, return_intermediates=True)
        return inter[predict_from], inter[predict_to]

    @torch.no_grad()
    def apply_f(a):
        """The frozen layer program: a_i -> a_j."""
        for i in range(from_idx + 1, to_idx + 1):
            a = model.transformer.h[i](a)
        return a

    # --- Estimate activation statistics at predict_from (from val tokens) ---
    print("\nEstimating activation statistics...")
    pool = []
    with torch.no_grad():
        for _ in range(n_stat_batches):
            a_i, _ = real_pair(val_data)
            pool.append(a_i.cpu())
    pool = torch.cat(pool, dim=0)  # (N, T, C)
    N, T, C = pool.shape
    print(f"  stat pool: {N} sequences x {T} positions x {C} dims")

    def cholesky_ridge(cov):
        eye = torch.eye(cov.shape[-1])
        jitter = cov_ridge * cov.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True
                                                                 ).unsqueeze(-1)
        for _ in range(6):
            try:
                return torch.linalg.cholesky(cov + jitter * eye)
            except Exception:
                jitter = jitter * 10
        return torch.linalg.cholesky(cov + 1e-1 * eye)

    flat = pool.reshape(-1, C)
    mu_global = flat.mean(0)                                  # (C,)
    cov_global = torch.cov(flat.T)                            # (C, C)
    L_global = cholesky_ridge(cov_global)                    # (C, C)
    sigma_iso = flat.var(0).mean().sqrt()                    # scalar

    mu_pos = pool.mean(0)                                     # (T, C)
    L_pos = torch.empty(T, C, C)
    for t in range(T):
        L_pos[t] = cholesky_ridge(torch.cov(pool[:, t, :].T))
    print(f"  global sigma_iso={sigma_iso:.3f}, "
          f"||mu_global||={mu_global.norm():.3f}")

    mu_global_d = mu_global.to(device)
    L_global_d = L_global.to(device)
    mu_pos_d = mu_pos.to(device)
    L_pos_d = L_pos.to(device)
    sigma_iso_d = sigma_iso.to(device)

    def sample_synthetic(condition, bs=batch_size, split_data=train_data):
        """Return a synthetic (or real) input batch (bs, T, C)."""
        if condition == "real":
            a_i, _ = real_pair(split_data, bs)
            return a_i
        if condition.startswith("perturb"):
            alpha = float(condition.split("_")[1])
            a_i, _ = real_pair(split_data, bs)
            noise = torch.randn_like(a_i)
            scale = alpha * a_i.norm(dim=-1, keepdim=True) / (C ** 0.5)
            return a_i + scale * noise
        z = torch.randn(bs, T, C, device=device)
        if condition == "gaussian_iso":
            return mu_global_d + sigma_iso_d * z
        if condition == "gaussian_global":
            return mu_global_d + z @ L_global_d.T
        if condition == "gaussian_perpos":
            # z: (bs, T, C); L_pos: (T, C, C) -> einsum over C
            return mu_pos_d + torch.einsum("btc,tdc->btd", z, L_pos_d)
        raise ValueError(condition)

    # --- Fixed held-out eval sets (same for every model) ---
    print("Building held-out eval sets...")
    with torch.no_grad():
        real_eval = [real_pair(val_data) for _ in range(n_eval_batches)]
        # generalization probes: eval on off-manifold too
        iso_eval, perpos_eval = [], []
        for _ in range(n_eval_batches // 2):
            xi = sample_synthetic("gaussian_iso")
            iso_eval.append((xi, apply_f(xi)))
            xp = sample_synthetic("gaussian_perpos")
            perpos_eval.append((xp, apply_f(xp)))

    # Identity reference: how lenient is full cosine?
    id_full_cos, id_upd_var = [], []
    with torch.no_grad():
        for a_i, a_j in real_eval:
            id_full_cos.append(torch.nn.functional.cosine_similarity(
                a_i, a_j, dim=-1).mean().item())
            id_upd_var.append((a_j - a_i).var().item())
    identity_full_cosine = float(np.mean(id_full_cos))
    print(f"  identity(a_i,a_j) full cosine = {identity_full_cosine:.4f} "
          f"(this is the free/lenient part)")

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
        full_mse, full_var = float(np.mean(fmse)), float(np.mean(tvar))
        upd_mse, upd_var = float(np.mean(umse)), float(np.mean(uvar))
        return {
            "full_cosine": float(np.mean(fc)),
            "full_r2": 1.0 - full_mse / full_var if full_var > 0 else 0.0,
            "update_cosine": float(np.mean(uc)),
            "update_r2": 1.0 - upd_mse / upd_var if upd_var > 0 else 0.0,
        }

    # --- Train one forward model per condition ---
    results = {}
    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===")
        torch.manual_seed(seed)  # identical init across conditions
        fwd = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        opt = torch.optim.AdamW(fwd.parameters(), lr=fwd_lr, weight_decay=0.01)

        traj = []
        for step in range(n_steps):
            fwd.train()
            a_i = sample_synthetic(condition)
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
                traj.append((step, m["full_cosine"], m["update_cosine"]))
                print(f"  step {step:6d}: train_mse={loss.item():.4f} | "
                      f"REAL full_cos={m['full_cosine']:.4f} "
                      f"upd_cos={m['update_cosine']:.4f} "
                      f"upd_r2={m['update_r2']:.4f}")

        results[condition] = {
            "eval_real": evaluate(fwd, real_eval),
            "eval_gaussian_iso": evaluate(fwd, iso_eval),
            "eval_gaussian_perpos": evaluate(fwd, perpos_eval),
            "trajectory": traj,
        }

    # --- Summary matrix ---
    print("\n\n=== SUMMARY: metric on held-out REAL pairs ===")
    print(f"{'condition':<18} {'full_cos':>9} {'upd_cos':>9} "
          f"{'upd_r2':>9} {'full_r2':>9}")
    print(f"{'(identity)':<18} {identity_full_cosine:>9.4f} "
          f"{'0.0':>9} {'0.0':>9} {'--':>9}")
    for c in CONDITIONS:
        r = results[c]["eval_real"]
        print(f"{c:<18} {r['full_cosine']:>9.4f} {r['update_cosine']:>9.4f} "
              f"{r['update_r2']:>9.4f} {r['full_r2']:>9.4f}")

    out = {
        "config": {
            "ckpt_dir": ckpt_dir, "n_tokens": n_tokens,
            "predict_from": predict_from, "predict_to": predict_to,
            "fwd_n_layer": fwd_n_layer, "n_steps": n_steps, "seed": seed,
        },
        "identity_full_cosine": identity_full_cosine,
        "conditions": results,
    }
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_dir = f"{DATA_DIR}/a2a_forward/synthetic_input/{gap_tag}/fwd_L{fwd_n_layer}"
    import os as _os
    _os.makedirs(save_dir, exist_ok=True)
    with open(_os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return out


@app.local_entrypoint()
def main(
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    n_steps: int = 10_000,
    n_tokens: int = 10_000_000,
):
    out = synthetic_input_experiment.remote(
        predict_from=predict_from, predict_to=predict_to,
        fwd_n_layer=fwd_n_layer, n_steps=n_steps, n_tokens=n_tokens,
    )
    print("\nDone. Real-eval update_cosine by condition:")
    for c, r in out["conditions"].items():
        print(f"  {c:<18} {r['eval_real']['update_cosine']:.4f}")
