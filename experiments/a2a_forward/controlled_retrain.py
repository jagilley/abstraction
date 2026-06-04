"""Controlled retrain: open-loop vs closed-loop with identical lr/seed.

Eliminates the lr confound (open lr=1e-4, closed lr=3e-4) from the original
comparison. Both models start from the same initial weights, see the same
batches in the same order, and differ ONLY in whether the cerebellar loop
is closed (prediction injected after block 1).

After training, runs layerwise self-map probes comparing whether the closed-loop
model's downstream layers encode more information about the forward model's
prediction residual.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_controlled_retrain(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
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
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    print(f"A2A CONTROLLED RETRAIN on {device}")
    print(f"  lr={lr} for BOTH open-loop and closed-loop (seed={seed})")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} → {predict_to}")
    print(f"  Injection: after block {inject_after_block}")

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

    # --- Pre-generate ALL batch indices so both runs see identical data ---
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)

    n_evals = n_steps // eval_interval + 2  # +2 for safety
    train_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(n_steps)
    ]
    eval_indices = [
        [torch.randint(len(val_data) - block_size - 1, (batch_size,),
                        generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]
    print(f"Pre-generated {n_steps} train batches, {n_evals} eval points, "
          f"{probe_batches} probe batches")

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + block_size] for i in indices])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # --- Initialize models with fixed seed, save initial state ---
    torch.manual_seed(seed)
    init_model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    init_fwd = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)

    init_model_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
    init_fwd_state = {k: v.cpu().clone() for k, v in init_fwd.state_dict().items()}

    del init_model, init_fwd
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # --- Output directory ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    # =============================================
    # Training function (shared by both runs)
    # =============================================
    def train_run(closed_loop: bool):
        label = "closed_loop" if closed_loop else "open_loop"
        label_upper = label.upper().replace("_", "-")
        print(f"\n{'='*60}")
        print(f"  Training {label_upper} (lr={lr}, seed={seed})")
        print(f"{'='*60}")

        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_model_state)
        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)
        gate = CerebellarGate(n_embd).to(device)

        if closed_loop:
            opt_main = torch.optim.AdamW(
                list(model.parameters()) + list(gate.parameters()),
                lr=lr, weight_decay=0.01,
            )
        else:
            opt_main = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.01,
            )
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
        )

        history = {
            "lm_loss": [], "fwd_mse": [],
            "val_lm": [], "val_lm_no_inj": [],
            "val_fwd_mse": [], "val_cosine": [],
            "val_residual_norm": [],
        }
        if closed_loop:
            history["gate_norm"] = []

        eval_idx = 0
        for step in range(n_steps):
            model.train()
            fwd_model.train()
            gate.train()

            x, y = make_batch(train_indices[step], train_data)

            if closed_loop:
                fwd_pred_cache = {}

                def cerebellar_fn(act):
                    fwd_pred = fwd_model(act.detach())
                    fwd_pred_cache["pred"] = fwd_pred
                    return gate(fwd_pred.detach())

                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                target = intermediates[predict_to].detach()
                fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)
            else:
                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True,
                )
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()
                predicted = fwd_model(source)
                fwd_loss = F.mse_loss(predicted, target)

            opt_main.zero_grad()
            lm_loss.backward()
            if closed_loop:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(gate.parameters()), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                gate.eval()

                with torch.no_grad():
                    val_lm_acc = 0.0
                    val_lm_noinj_acc = 0.0
                    val_fwd_acc = 0.0
                    val_cos_acc = 0.0
                    val_res_acc = 0.0

                    for eb_idx in eval_indices[eval_idx]:
                        vx, vy = make_batch(eb_idx, val_data)

                        if closed_loop:
                            def eval_cb_fn(act):
                                return gate(fwd_model(act))
                            _, vl, _ = model(
                                vx, vy, return_intermediates=True,
                                cerebellar_fn=eval_cb_fn,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                            val_lm_acc += float(vl)

                        _, vl_noinj, vi = model(
                            vx, vy, return_intermediates=True,
                        )
                        if not closed_loop:
                            val_lm_acc += float(vl_noinj)
                        val_lm_noinj_acc += float(vl_noinj)

                        src = vi[predict_from]
                        tgt = vi[predict_to]
                        pred = fwd_model(src)
                        val_fwd_acc += float(F.mse_loss(pred, tgt))
                        val_cos_acc += float(
                            F.cosine_similarity(pred, tgt, dim=-1).mean())
                        val_res_acc += float((tgt - pred).norm(dim=-1).mean())

                    n = n_eval_batches
                    val_lm = val_lm_acc / n
                    val_lm_noinj = val_lm_noinj_acc / n
                    val_fwd = val_fwd_acc / n
                    val_cos = val_cos_acc / n
                    val_res = val_res_acc / n

                    history["lm_loss"].append((step, float(lm_loss)))
                    history["fwd_mse"].append((step, float(fwd_loss)))
                    history["val_lm"].append((step, val_lm))
                    history["val_lm_no_inj"].append((step, val_lm_noinj))
                    history["val_fwd_mse"].append((step, val_fwd))
                    history["val_cosine"].append((step, val_cos))
                    history["val_residual_norm"].append((step, val_res))

                    if closed_loop:
                        gn = gate.injection_norm()
                        history["gate_norm"].append((step, gn))
                        delta = val_lm - val_lm_noinj
                        print(f"  [{label_upper}] step {step:6d}: "
                              f"lm={val_lm:.4f} no_inj={val_lm_noinj:.4f} "
                              f"Δ={delta:+.4f} | "
                              f"cos={val_cos:.4f} gate={gn:.4f}")
                    else:
                        print(f"  [{label_upper}] step {step:6d}: "
                              f"lm={val_lm:.4f} | cos={val_cos:.4f}")

                eval_idx += 1

        # --- Save checkpoint ---
        save_dir = os.path.join(save_root, label)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(fwd_model.state_dict(),
                    os.path.join(save_dir, "fwd_model.pt"))
        if closed_loop:
            torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))
        print(f"  Saved checkpoint to {save_dir}")

        return model, fwd_model, gate, history

    # =============================================
    # Run both
    # =============================================
    model_open, fwd_open, _, hist_open = train_run(closed_loop=False)
    model_closed, fwd_closed, gate_closed, hist_closed = train_run(
        closed_loop=True)

    # =============================================
    # Self-map probes (identical probe data for both)
    # =============================================
    print(f"\n{'='*60}")
    print(f"  SELF-MAP PROBES ({probe_batches} batches, {probe_steps} steps)")
    print(f"{'='*60}")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def collect_for_probes(model, fwd_model, gate, label, closed_loop):
        model.eval()
        fwd_model.eval()
        if gate is not None:
            gate.eval()

        acts = {k: [] for k in layer_keys}
        residual_vecs = []
        residual_norms = []
        block1_contrib_norms = []

        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)

                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())

                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                res = tgt - pred
                residual_vecs.append(res.reshape(-1, n_embd).cpu())
                residual_norms.append(
                    res.norm(dim=-1).reshape(-1).cpu())
                block1_contrib_norms.append(
                    (vi["post_block1"] - vi["post_block0"])
                    .norm(dim=-1).reshape(-1).cpu())

        acts = {k: torch.cat(v) for k, v in acts.items()}
        return {
            "acts": acts,
            "residual_vec": torch.cat(residual_vecs),
            "residual_norm": torch.cat(residual_norms),
            "block1_contrib_norm": torch.cat(block1_contrib_norms),
        }

    print("Collecting open-loop activations...")
    data_open = collect_for_probes(model_open, fwd_open, None,
                                    "open", False)
    print("Collecting closed-loop activations (no injection pass)...")
    data_closed = collect_for_probes(model_closed, fwd_closed, gate_closed,
                                     "closed", True)

    n_samples = data_open["residual_norm"].shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    print(f"Probe dataset: {n_samples} samples ({n_train} train, "
          f"{n_samples - n_train} test)")

    # --- Scalar probe (R²) ---
    def scalar_probe(X, target_vals, label=""):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        t_np = target_vals.numpy() if isinstance(target_vals, torch.Tensor) \
            else target_vals
        t_np = t_np.astype(np.float32)

        mu_x = X_np[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu_x) / sd_x

        t_mu = t_np[train_idx].mean()
        t_sd = t_np[train_idx].std() + 1e-6
        t_s = (t_np - t_mu) / t_sd

        probe = nn.Linear(X_np.shape[1], 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        ttr = torch.from_numpy(t_s[train_idx]).float().to(device)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,), device=device)
            pred = probe(Xtr[idx]).squeeze(-1)
            loss = F.mse_loss(pred, ttr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).squeeze(-1).cpu().numpy()
        tte = t_s[test_idx]
        var = tte.var()
        r2 = float(1.0 - ((pred - tte) ** 2).mean() / var) if var > 0 else 0.0
        return r2

    # --- Vector probe (R², predicting full 256-d residual) ---
    def vector_probe(X, target_vec, label=""):
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

    # --- Run probes ---
    probe_results = {"scalar": {}, "vector": {}}

    print("\n=== Scalar probes (R²): residual_norm & block1_contrib (control) ===")
    print(f"{'target':>20s} {'layer':>12s} {'open R²':>8s} "
          f"{'closed R²':>10s} {'Δ':>7s}")
    for tgt in ["residual_norm", "block1_contrib_norm"]:
        probe_results["scalar"][tgt] = {}
        for lk in layer_keys:
            r2_o = scalar_probe(data_open["acts"][lk],
                                data_open[tgt])
            r2_c = scalar_probe(data_closed["acts"][lk],
                                data_closed[tgt])
            probe_results["scalar"][tgt][lk] = {
                "open": r2_o, "closed": r2_c, "delta": r2_c - r2_o,
            }
            print(f"{tgt:>20s} {lk:>12s} {r2_o:>8.4f} {r2_c:>10.4f} "
                  f"{r2_c - r2_o:>+7.4f}")

    print("\n=== Vector probes (R²): full 256-d residual ===")
    print(f"{'layer':>12s} {'open R²':>8s} {'closed R²':>10s} {'Δ':>7s} "
          f"{'open cos':>9s} {'closed cos':>11s}")
    probe_results["vector"]["residual"] = {}
    for lk in layer_keys:
        v_o = vector_probe(data_open["acts"][lk], data_open["residual_vec"])
        v_c = vector_probe(data_closed["acts"][lk], data_closed["residual_vec"])
        probe_results["vector"]["residual"][lk] = {
            "open": v_o, "closed": v_c,
            "delta_r2": v_c["r2"] - v_o["r2"],
        }
        print(f"{lk:>12s} {v_o['r2']:>8.4f} {v_c['r2']:>10.4f} "
              f"{v_c['r2'] - v_o['r2']:>+7.4f} "
              f"{v_o['cosine']:>9.4f} {v_c['cosine']:>11.4f}")

    # --- Forward model quality comparison ---
    print("\n=== Forward model quality (no injection) ===")
    fwd_quality = {}
    for label_q, model_q, fwd_q in [("open", model_open, fwd_open),
                                      ("closed", model_closed, fwd_closed)]:
        model_q.eval()
        fwd_q.eval()
        cos_acc = []
        mse_acc = []
        with torch.no_grad():
            for pidx in probe_indices[:10]:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model_q(vx, vy, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_q(src)
                cos_acc.append(
                    F.cosine_similarity(pred, tgt, dim=-1).mean().item())
                mse_acc.append(F.mse_loss(pred, tgt).item())
        fwd_quality[label_q] = {
            "cosine": float(np.mean(cos_acc)),
            "mse": float(np.mean(mse_acc)),
        }
        print(f"  {label_q}: cosine={fwd_quality[label_q]['cosine']:.4f} "
              f"mse={fwd_quality[label_q]['mse']:.5f}")

    # --- LM loss comparison ---
    print("\n=== Final LM loss comparison ===")
    final_lm_open = hist_open["val_lm"][-1][1]
    final_lm_closed_with = hist_closed["val_lm"][-1][1]
    final_lm_closed_no = hist_closed["val_lm_no_inj"][-1][1]
    print(f"  Open-loop:               {final_lm_open:.4f}")
    print(f"  Closed-loop (with inj):  {final_lm_closed_with:.4f}")
    print(f"  Closed-loop (no inj):    {final_lm_closed_no:.4f}")
    print(f"  Δ (closed_with - open):  {final_lm_closed_with - final_lm_open:+.4f}")
    print(f"  Δ (closed_no - open):    {final_lm_closed_no - final_lm_open:+.4f}")

    # --- Summary ---
    print("\n=== LOCALIZATION TEST ===")
    print("If the self-map is built by the loop (not a training artifact),")
    print("the probe gap should grow from block0→block1 (pre-injection)")
    print("to block2→block3 (post-injection).")
    vec_res = probe_results["vector"]["residual"]
    for lk in layer_keys:
        d = vec_res[lk]["delta_r2"]
        block_num = int(lk.replace("post_block", ""))
        marker = "  (pre-injection)" if block_num <= inject_after_block \
            else "  (POST-injection)"
        print(f"  {lk}: Δ R² = {d:+.4f}{marker}")

    # =============================================
    # Save everything
    # =============================================
    result = {
        "config": {
            "lr": lr, "fwd_lr": fwd_lr, "seed": seed,
            "n_tokens": n_tokens, "n_steps": n_steps,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
        },
        "lm_loss": {
            "open_final": final_lm_open,
            "closed_with_inj_final": final_lm_closed_with,
            "closed_no_inj_final": final_lm_closed_no,
        },
        "fwd_quality": fwd_quality,
        "probe_results": probe_results,
        "history_open": hist_open,
        "history_closed": hist_closed,
    }

    results_path = os.path.join(save_root, "results.json")
    os.makedirs(save_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)

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
    n_steps: int = 10_000,
    lr: float = 3e-4,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_controlled_retrain.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A controlled retrain complete:")
    lm = result["lm_loss"]
    print(f"  Open-loop LM:              {lm['open_final']:.4f}")
    print(f"  Closed-loop LM (with inj): {lm['closed_with_inj_final']:.4f}")
    print(f"  Closed-loop LM (no inj):   {lm['closed_no_inj_final']:.4f}")
    fq = result["fwd_quality"]
    print(f"  Fwd cosine — open: {fq['open']['cosine']:.4f}, "
          f"closed: {fq['closed']['cosine']:.4f}")
    print("\n  Localization test (Δ R² = closed − open):")
    vec = result["probe_results"]["vector"]["residual"]
    for lk in sorted(vec.keys()):
        d = vec[lk]["delta_r2"]
        print(f"    {lk}: Δ R² = {d:+.4f}")
