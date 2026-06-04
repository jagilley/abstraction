"""Extended co-training: does the closed-loop model resist overfitting?

Same design as controlled_retrain.py (identical lr/seed/init), but trains for
50K steps (~45 epochs on 10M tokens) to observe long-run dynamics:

  1. Does the open-loop model overfit while the closed-loop model doesn't?
  2. Does forward model cosine continue to evolve (ongoing co-adaptation)?
  3. Does self-knowledge (probe gap) continue to grow?

Saves intermediate checkpoints + probes every checkpoint_interval steps,
with volume.commit() for crash resilience.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=86400,  # 24 hours
    memory=32768,
)
def a2a_extended_training(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 50_000,
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    checkpoint_interval: int = 10_000,
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
    print(f"A2A EXTENDED TRAINING on {device}")
    print(f"  lr={lr}, seed={seed}, n_steps={n_steps}")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} → {predict_to}")
    print(f"  Injection: after block {inject_after_block}")
    print(f"  Checkpoints every {checkpoint_interval} steps")

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
    tokens_per_step = batch_size * block_size
    n_epochs = (n_steps * tokens_per_step) / len(train_data)
    print(f"  Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    print(f"  ~{n_epochs:.1f} epochs over {n_steps} steps")

    # --- Pre-generate ALL batch indices ---
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)

    n_evals = n_steps // eval_interval + 2
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
    # Separate probe indices for each checkpoint
    n_checkpoints = n_steps // checkpoint_interval + 1
    probe_indices_all = [
        [torch.randint(len(val_data) - block_size - 1, (batch_size,),
                        generator=probe_gen)
         for _ in range(probe_batches)]
        for _ in range(n_checkpoints)
    ]
    print(f"Pre-generated {n_steps} train batches, {n_evals} eval points, "
          f"{n_checkpoints} probe checkpoints")

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

    # --- Output directory ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    fwd_tag = f"fwd{fwd_n_layer}L{fwd_n_head}H{fwd_d_head}d_mlp{fwd_mlp_mult}"
    save_root = (f"{DATA_DIR}/a2a_forward/extended/"
                 f"{gap_tag}/inject{inject_after_block}/{fwd_tag}/P_{n_tokens}")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    # =============================================
    # Probe functions
    # =============================================
    def scalar_probe(X, target_vals, train_idx, test_idx):
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

        n_train = len(train_idx)
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
        return float(1.0 - ((pred - tte) ** 2).mean() / var) if var > 0 else 0.0

    def vector_probe(X, target_vec, train_idx, test_idx):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = target_vec.numpy() if isinstance(target_vec, torch.Tensor) \
            else target_vec
        T_np = T_np.astype(np.float32)

        mu_x = X_np[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu_x) / sd_x

        n_train = len(train_idx)
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

    def collect_probe_data(model, fwd_model, p_indices):
        model.eval()
        fwd_model.eval()

        acts = {k: [] for k in layer_keys}
        residual_vecs = []
        residual_norms = []

        with torch.no_grad():
            for pidx in p_indices:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)

                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())

                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                res = tgt - pred
                residual_vecs.append(res.reshape(-1, n_embd).cpu())
                residual_norms.append(res.norm(dim=-1).reshape(-1).cpu())

        acts = {k: torch.cat(v) for k, v in acts.items()}
        return {
            "acts": acts,
            "residual_vec": torch.cat(residual_vecs),
            "residual_norm": torch.cat(residual_norms),
        }

    def run_probes(data_open, data_closed, rng_seed):
        n_samples = data_open["residual_norm"].shape[0]
        n_train = int(0.8 * n_samples)
        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        results = {"scalar": {}, "vector": {}}

        results["scalar"]["residual_norm"] = {}
        for lk in layer_keys:
            r2_o = scalar_probe(data_open["acts"][lk],
                                data_open["residual_norm"],
                                train_idx, test_idx)
            r2_c = scalar_probe(data_closed["acts"][lk],
                                data_closed["residual_norm"],
                                train_idx, test_idx)
            results["scalar"]["residual_norm"][lk] = {
                "open": r2_o, "closed": r2_c, "delta": r2_c - r2_o,
            }

        results["vector"]["residual"] = {}
        for lk in layer_keys:
            v_o = vector_probe(data_open["acts"][lk],
                               data_open["residual_vec"],
                               train_idx, test_idx)
            v_c = vector_probe(data_closed["acts"][lk],
                               data_closed["residual_vec"],
                               train_idx, test_idx)
            results["vector"]["residual"][lk] = {
                "open": v_o, "closed": v_c,
                "delta_r2": v_c["r2"] - v_o["r2"],
            }

        return results

    # =============================================
    # Training function (shared by both runs)
    # =============================================
    def train_run(closed_loop: bool, other_model_history=None):
        label = "closed_loop" if closed_loop else "open_loop"
        label_upper = label.upper().replace("_", "-")
        print(f"\n{'='*60}")
        print(f"  Training {label_upper} (lr={lr}, seed={seed}, "
              f"n_steps={n_steps})")
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
            "lm_loss_train": [], "fwd_mse_train": [],
            "val_lm": [], "val_lm_no_inj": [],
            "val_fwd_mse": [], "val_cosine": [],
            "val_residual_norm": [],
        }
        if closed_loop:
            history["gate_norm"] = []

        # Running average for train loss
        train_lm_running = 0.0
        train_fwd_running = 0.0
        running_n = 0

        eval_idx = 0
        ckpt_idx = 0
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

            train_lm_running += float(lm_loss)
            train_fwd_running += float(fwd_loss)
            running_n += 1

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                gate.eval()

                # Record train loss average
                avg_train_lm = train_lm_running / max(running_n, 1)
                avg_train_fwd = train_fwd_running / max(running_n, 1)
                history["lm_loss_train"].append((step, avg_train_lm))
                history["fwd_mse_train"].append((step, avg_train_fwd))
                train_lm_running = 0.0
                train_fwd_running = 0.0
                running_n = 0

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

                    history["val_lm"].append((step, val_lm))
                    history["val_lm_no_inj"].append((step, val_lm_noinj))
                    history["val_fwd_mse"].append((step, val_fwd))
                    history["val_cosine"].append((step, val_cos))
                    history["val_residual_norm"].append((step, val_res))

                    if closed_loop:
                        gn = gate.injection_norm()
                        history["gate_norm"].append((step, gn))
                        delta = val_lm - val_lm_noinj
                        overfit = avg_train_lm - val_lm
                        print(f"  [{label_upper}] step {step:6d}: "
                              f"train={avg_train_lm:.4f} val={val_lm:.4f} "
                              f"(gap={overfit:+.3f}) "
                              f"no_inj={val_lm_noinj:.4f} "
                              f"Δ={delta:+.4f} | "
                              f"cos={val_cos:.4f} gate={gn:.4f}")
                    else:
                        overfit = avg_train_lm - val_lm
                        print(f"  [{label_upper}] step {step:6d}: "
                              f"train={avg_train_lm:.4f} val={val_lm:.4f} "
                              f"(gap={overfit:+.3f}) | "
                              f"cos={val_cos:.4f}")

                eval_idx += 1

            # --- Checkpoint + probes ---
            if (step > 0 and step % checkpoint_interval == 0) \
                    or step == n_steps - 1:
                ckpt_step_dir = os.path.join(
                    save_root, label, f"step_{step:06d}")
                os.makedirs(ckpt_step_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_step_dir, "model.pt"))
                torch.save(fwd_model.state_dict(),
                           os.path.join(ckpt_step_dir, "fwd_model.pt"))
                if closed_loop:
                    torch.save(gate.state_dict(),
                               os.path.join(ckpt_step_dir, "gate.pt"))

                # Save history so far
                hist_path = os.path.join(save_root, label, "history.json")
                with open(hist_path, "w") as f:
                    json.dump(history, f, indent=2, cls=NumpyEncoder)

                volume.commit()
                print(f"  Checkpoint saved at step {step} → {ckpt_step_dir}")
                ckpt_idx += 1

        return model, fwd_model, gate, history

    # =============================================
    # Run both
    # =============================================
    model_open, fwd_open, _, hist_open = train_run(closed_loop=False)
    model_closed, fwd_closed, gate_closed, hist_closed = train_run(
        closed_loop=True)

    # =============================================
    # Final probes (same as controlled retrain)
    # =============================================
    print(f"\n{'='*60}")
    print(f"  FINAL SELF-MAP PROBES")
    print(f"{'='*60}")

    print("Collecting open-loop activations...")
    data_open = collect_probe_data(model_open, fwd_open,
                                   probe_indices_all[-1])
    print("Collecting closed-loop activations (no injection pass)...")
    data_closed = collect_probe_data(model_closed, fwd_closed,
                                     probe_indices_all[-1])

    probe_results = run_probes(data_open, data_closed, seed)

    print("\n=== Scalar probes: residual_norm ===")
    print(f"{'layer':>12s} {'open R²':>8s} {'closed R²':>10s} {'Δ':>7s}")
    for lk in layer_keys:
        r = probe_results["scalar"]["residual_norm"][lk]
        print(f"{lk:>12s} {r['open']:>8.4f} {r['closed']:>10.4f} "
              f"{r['delta']:>+7.4f}")

    print("\n=== Vector probes: full residual ===")
    print(f"{'layer':>12s} {'open R²':>8s} {'closed R²':>10s} {'Δ':>7s} "
          f"{'open cos':>9s} {'closed cos':>11s}")
    for lk in layer_keys:
        v = probe_results["vector"]["residual"][lk]
        print(f"{lk:>12s} {v['open']['r2']:>8.4f} {v['closed']['r2']:>10.4f} "
              f"{v['delta_r2']:>+7.4f} "
              f"{v['open']['cosine']:>9.4f} {v['closed']['cosine']:>11.4f}")

    # --- Forward model quality ---
    print("\n=== Forward model quality (no injection) ===")
    fwd_quality = {}
    for label_q, model_q, fwd_q in [("open", model_open, fwd_open),
                                      ("closed", model_closed, fwd_closed)]:
        model_q.eval()
        fwd_q.eval()
        cos_acc = []
        mse_acc = []
        with torch.no_grad():
            for pidx in probe_indices_all[-1][:10]:
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

    # --- Overfitting summary ---
    print(f"\n{'='*60}")
    print(f"  OVERFITTING COMPARISON")
    print(f"{'='*60}")
    for label_s, hist in [("OPEN", hist_open), ("CLOSED", hist_closed)]:
        train_losses = hist["lm_loss_train"]
        val_losses = hist["val_lm"]
        if len(train_losses) >= 2 and len(val_losses) >= 2:
            early_train = train_losses[1][1]
            late_train = train_losses[-1][1]
            early_val = val_losses[1][1]
            late_val = val_losses[-1][1]
            early_gap = early_train - early_val
            late_gap = late_train - late_val
            print(f"  [{label_s}] Train: {early_train:.4f} → {late_train:.4f} "
                  f"({late_train - early_train:+.4f})")
            print(f"  [{label_s}] Val:   {early_val:.4f} → {late_val:.4f} "
                  f"({late_val - early_val:+.4f})")
            print(f"  [{label_s}] Gap:   {early_gap:+.3f} → {late_gap:+.3f} "
                  f"(Δgap={late_gap - early_gap:+.3f})")

    # =============================================
    # Save everything
    # =============================================
    result = {
        "config": {
            "lr": lr, "fwd_lr": fwd_lr, "seed": seed,
            "n_tokens": n_tokens, "n_steps": n_steps,
            "eval_interval": eval_interval,
            "checkpoint_interval": checkpoint_interval,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
        },
        "final_lm_loss": {
            "open": hist_open["val_lm"][-1][1],
            "closed_with_inj": hist_closed["val_lm"][-1][1],
            "closed_no_inj": hist_closed["val_lm_no_inj"][-1][1],
        },
        "fwd_quality": fwd_quality,
        "probe_results": probe_results,
        "history_open": hist_open,
        "history_closed": hist_closed,
    }

    results_path = os.path.join(save_root, "results.json")
    os.makedirs(save_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nAll results saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_steps: int = 50_000,
    lr: float = 3e-4,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_extended_training.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("A2A extended training complete:")
    lm = result["final_lm_loss"]
    print(f"  Open-loop LM:              {lm['open']:.4f}")
    print(f"  Closed-loop LM (with inj): {lm['closed_with_inj']:.4f}")
    print(f"  Closed-loop LM (no inj):   {lm['closed_no_inj']:.4f}")
    fq = result["fwd_quality"]
    print(f"  Fwd cosine — open: {fq['open']['cosine']:.4f}, "
          f"closed: {fq['closed']['cosine']:.4f}")
    print("\n  Final probe results (Δ R² = closed − open):")
    vec = result["probe_results"]["vector"]["residual"]
    for lk in sorted(vec.keys()):
        d = vec[lk]["delta_r2"]
        print(f"    {lk}: Δ R² = {d:+.4f}")
