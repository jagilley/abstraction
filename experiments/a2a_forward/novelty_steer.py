"""Idea 3: causal steering of the novelty direction.

Tests whether the closed-loop model's *own* internal novelty representation
is a functional control variable for how much it relies on the injected
prediction — not merely a decodable correlate. This is a within-model causal
test, so it is free of the lr / forward-model target-difference confounds that
muddied the cross-model probe (idea 1).

Setup
-----
The injection enters the residual stream after block 1: stream <- post_block1 +
gate(fwd_model(post_block0)). Blocks 2-3 then process the augmented stream.

- Novelty direction v_hat: the post_block1 linear-probe direction that predicts
  residual_norm = ||post_block3 - fwd_pred|| (the forward model's per-token
  error = the novelty signal). Unit-normalized in raw activation space.
- Reliance R(s): per-token KL(p_with_inj || p_no_inj) measured WHILE steering
  the stream by s * sigma * v_hat (sigma = std of the natural novelty
  coordinate, so s is in std units). The steering is applied identically in
  both the with-inj and no-inj passes (folded into the injection hook), so its
  direct effect on the logits differences out; R isolates the injection's
  causal influence on the output IN the steered context.

Gating hypothesis: dR/ds < 0. Pushing the model toward "this position is novel /
the prediction is unreliable" makes it rely LESS on the injection.

Controls: random directions (matched magnitude) and the block1_contrib
direction. If only v_hat modulates reliance, the effect is novelty-specific.

Free pre-check (s=0): per-token reliance vs residual_norm should correlate
negatively (high novelty -> low reliance) if the model already gates.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_novelty_steer(
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
    probe_steps: int = 800,
    n_random_dirs: int = 3,
    seed: int = 0,
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A novelty steering on {device}")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

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
    data = torch.from_numpy(np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]

    g = torch.Generator().manual_seed(seed)

    def make_batches(n):
        out = []
        for _ in range(n):
            ix = torch.randint(len(val_data) - block_size - 1, (batch_size,),
                               generator=g)
            out.append(torch.stack([val_data[i:i + block_size] for i in ix]))
        return out

    fit_batches = make_batches(n_fit_batches)
    eval_batches = make_batches(n_eval_batches)

    # --- Load closed-loop model + forward model + gate ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_dir = (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading closed-loop from {model_dir}")

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device,
        weights_only=True))
    fwd_model.eval()
    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(model_dir, "gate.pt"), map_location=device, weights_only=True))
    gate.eval()
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    # =============================================
    # Fit the novelty direction (and contrib control) on fit_batches
    # =============================================
    print("\nCollecting post_block1 + residual_norm (no inj) to fit directions...")
    pb1, res_norm, b1c_norm = [], [], []
    with torch.no_grad():
        for x in fit_batches:
            x = x.to(device)
            _, _, inter = model(x, return_intermediates=True)
            pb1.append(inter["post_block1"].reshape(-1, n_embd).cpu())
            pred = fwd_model(inter[predict_from])
            res_norm.append((inter[predict_to] - pred).norm(dim=-1).reshape(-1).cpu())
            b1c_norm.append((inter["post_block1"] - inter["post_block0"])
                            .norm(dim=-1).reshape(-1).cpu())
    pb1 = torch.cat(pb1).numpy()
    res_norm = torch.cat(res_norm).numpy()
    b1c_norm = torch.cat(b1c_norm).numpy()

    def fit_direction(X, target, name):
        """Linear probe in RAW activation space; return unit weight direction,
        the std of the projection (natural coordinate scale), and test R²."""
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        ntr = int(0.8 * n)
        tr, te = perm[:ntr], perm[ntr:]
        Xtr = torch.from_numpy(X[tr]).float().to(device)
        Xte = torch.from_numpy(X[te]).float().to(device)
        t = target.astype(np.float32)
        tmu, tsd = t[tr].mean(), t[tr].std() + 1e-6
        ttr = torch.from_numpy((t[tr] - tmu) / tsd).to(device)
        tte = (t[te] - tmu) / tsd
        probe = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        bs = min(4096, ntr)
        for _ in range(probe_steps):
            idx = torch.randint(ntr, (bs,), device=device)
            loss = F.mse_loss(probe(Xtr[idx]).squeeze(-1), ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = probe(Xte).squeeze(-1).cpu().numpy()
        r2 = float(1.0 - ((pred - tte) ** 2).mean() / tte.var())
        w = probe.weight.detach().squeeze(0).cpu().numpy()
        v_hat = w / (np.linalg.norm(w) + 1e-8)
        sigma = float((X @ v_hat).std())
        print(f"  direction[{name}]: probe R²={r2:.4f}, coord sigma={sigma:.3f}")
        return v_hat.astype(np.float32), sigma, r2

    v_nov, sig_nov, r2_nov = fit_direction(pb1, res_norm, "novelty")
    v_b1c, sig_b1c, r2_b1c = fit_direction(pb1, b1c_norm, "block1_contrib")

    rng = np.random.default_rng(seed + 1)
    rand_dirs = []
    for j in range(n_random_dirs):
        r = rng.standard_normal(n_embd).astype(np.float32)
        r /= np.linalg.norm(r)
        rand_dirs.append((r, float((pb1 @ r).std())))

    # =============================================
    # Steering sweep
    # =============================================
    s_grid = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    def reliance_for_steer(steer_vec):
        """Mean per-token KL(p_with_inj || p_no_inj) under a fixed steer vector
        (C,) added to the post-block1 stream in BOTH passes. Also returns the
        flat per-token KL and residual_norm arrays for s=0 correlation."""
        steer_t = torch.from_numpy(steer_vec).to(device) if steer_vec is not None \
            else None
        kl_sum, ntok = 0.0, 0
        kl_flat, res_flat = [], []
        with torch.no_grad():
            for x in eval_batches:
                x = x.to(device)

                def inj_fn(act):
                    out = gate(fwd_model(act))
                    return out + steer_t if steer_t is not None else out

                def noinj_fn(act):
                    z = torch.zeros_like(gate(fwd_model(act)))
                    return z + steer_t if steer_t is not None else z

                logits_inj, _, inter = model(
                    x, return_intermediates=True, cerebellar_fn=inj_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block)
                logits_no, _ = model(
                    x, cerebellar_fn=noinj_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block)

                logp = F.log_softmax(logits_inj, dim=-1)
                logq = F.log_softmax(logits_no, dim=-1)
                kl = (logp.exp() * (logp - logq)).sum(-1)  # (B,T)
                kl_sum += float(kl.sum()); ntok += kl.numel()

                pred = fwd_model(inter[predict_from])
                rn = (inter[predict_to] - pred).norm(dim=-1)  # (B,T)
                kl_flat.append(kl.reshape(-1).cpu())
                res_flat.append(rn.reshape(-1).cpu())
        mean_kl = kl_sum / ntok
        return mean_kl, torch.cat(kl_flat).numpy(), torch.cat(res_flat).numpy()

    results = {
        "gate_injection_norm": gate.injection_norm(),
        "direction_r2": {"novelty": r2_nov, "block1_contrib": r2_b1c},
        "s_grid": s_grid,
        "reliance_curves": {},
    }

    print("\n=== Steering sweep: mean reliance R(s)=KL(inj||no_inj) ===")
    print(f"{'s (std)':>8s} {'novelty':>10s} {'contrib':>10s} {'random':>10s}")

    # Precompute curves
    nov_curve, b1c_curve, rand_curve = [], [], []
    corr_at_zero = None
    for s in s_grid:
        steer_nov = (s * sig_nov) * v_nov if s != 0 else np.zeros(n_embd, np.float32)
        mk_nov, kl0, res0 = reliance_for_steer(steer_nov)
        nov_curve.append(mk_nov)

        steer_b1c = (s * sig_b1c) * v_b1c if s != 0 else np.zeros(n_embd, np.float32)
        mk_b1c, _, _ = reliance_for_steer(steer_b1c)
        b1c_curve.append(mk_b1c)

        rand_vals = []
        for (r, sig_r) in rand_dirs:
            steer_r = (s * sig_r) * r if s != 0 else np.zeros(n_embd, np.float32)
            mk_r, _, _ = reliance_for_steer(steer_r)
            rand_vals.append(mk_r)
        mk_rand = float(np.mean(rand_vals))
        rand_curve.append(mk_rand)

        if s == 0.0:
            # free correlational pre-check at no steering
            corr_at_zero = float(np.corrcoef(kl0, res0)[0, 1])

        print(f"{s:>8.1f} {mk_nov:>10.5f} {mk_b1c:>10.5f} {mk_rand:>10.5f}")

    results["reliance_curves"] = {
        "novelty": nov_curve, "block1_contrib": b1c_curve, "random": rand_curve}
    results["reliance_vs_residual_corr_at_s0"] = corr_at_zero

    # Slope of R(s) via least squares over the grid (units: ΔKL per std)
    s_arr = np.array(s_grid)
    def slope(curve):
        return float(np.polyfit(s_arr, np.array(curve), 1)[0])
    results["slopes"] = {
        "novelty": slope(nov_curve),
        "block1_contrib": slope(b1c_curve),
        "random": slope(rand_curve),
    }

    print(f"\nBaseline reliance R(0) = {nov_curve[s_grid.index(0.0)]:.5f}")
    print(f"Per-token corr(reliance, residual_norm) at s=0: {corr_at_zero:+.4f}")
    print("Slope dR/ds (ΔKL per std of steering):")
    print(f"  novelty:        {results['slopes']['novelty']:+.5f}")
    print(f"  block1_contrib: {results['slopes']['block1_contrib']:+.5f}")
    print(f"  random:         {results['slopes']['random']:+.5f}")

    # --- Save ---
    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "novelty_steer_results.json")
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
    result = a2a_novelty_steer.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A novelty steering complete:")
    print(f"  corr(reliance, residual_norm) at s=0: "
          f"{result['reliance_vs_residual_corr_at_s0']:+.4f}")
    print("  Slope dR/ds (ΔKL per std of steering):")
    for k, v in result["slopes"].items():
        print(f"    {k:>16s}: {v:+.5f}")
