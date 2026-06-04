"""A2A forward model experiment stages.

Co-trains a small transformer forward model (the "cerebellum") alongside a GPT.
The forward model predicts post-block1 activations from post-block0 activations
using a capacity-bottlenecked 1-layer transformer. The residual captures
computational novelty (what the forward model's capacity can't represent),
not positional blindness.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_train(
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
    predict_to: str = "post_block1",
    fwd_type: str = "transformer",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    import os
    import glob
    import torch
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        ForwardModel, TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A forward model training on {device}")
    print(f"  Main model: {n_layer}L {n_head}H {n_embd}D")
    print(f"  Forward model ({fwd_type}): {predict_from} → {predict_to}")
    print(f"  P={n_tokens:,}, T={block_size}")

    # --- Load data (τ=0.0, unchanged corpus) ---
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

    # --- Create models ---
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)

    if fwd_type == "transformer":
        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
    else:
        fwd_model = ForwardModel(n_embd, hidden_mult=fwd_mlp_mult).to(device)

    opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- Training loop ---
    history = {
        "lm_loss": [], "fwd_mse": [],
        "val_lm_loss": [], "val_fwd_mse": [], "val_cosine_sim": [],
        "val_residual_norm": [], "val_residual_by_position": [],
    }
    best_val_loss = float("inf")

    for step in range(n_steps):
        model.train()
        fwd_model.train()

        x, y = get_batch(train_data)
        _, lm_loss, intermediates = model(x, y, return_intermediates=True)

        source = intermediates[predict_from].detach()
        target = intermediates[predict_to].detach()
        predicted = fwd_model(source)
        fwd_loss = torch.nn.functional.mse_loss(predicted, target)

        opt_main.zero_grad()
        lm_loss.backward()
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
            with torch.no_grad():
                val_lm_accum = 0.0
                val_fwd_accum = 0.0
                val_cos_accum = 0.0
                val_res_norm_accum = 0.0
                val_res_by_pos = torch.zeros(block_size, device=device)

                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)
                    _, vl, vi = model(vx, vy, return_intermediates=True)
                    val_lm_accum += float(vl)

                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fwd_model(src)
                    val_fwd_accum += float(
                        torch.nn.functional.mse_loss(pred, tgt)
                    )

                    cos = torch.nn.functional.cosine_similarity(
                        pred, tgt, dim=-1
                    )
                    val_cos_accum += cos.mean().item()

                    residual = tgt - pred
                    res_norm = residual.norm(dim=-1)
                    val_res_norm_accum += res_norm.mean().item()
                    val_res_by_pos += res_norm.mean(dim=0)

                n = n_eval_batches
                val_lm = val_lm_accum / n
                val_fwd = val_fwd_accum / n
                val_cos = val_cos_accum / n
                val_res = val_res_norm_accum / n
                val_pos = (val_res_by_pos / n).cpu().tolist()

                if val_lm < best_val_loss:
                    best_val_loss = val_lm

                history["lm_loss"].append((step, float(lm_loss)))
                history["fwd_mse"].append((step, float(fwd_loss)))
                history["val_lm_loss"].append((step, val_lm))
                history["val_fwd_mse"].append((step, val_fwd))
                history["val_cosine_sim"].append((step, val_cos))
                history["val_residual_norm"].append((step, val_res))
                history["val_residual_by_position"].append((step, val_pos))

                print(f"  step {step:6d}: "
                      f"lm={float(lm_loss):.4f} val_lm={val_lm:.4f} "
                      f"fwd_mse={float(fwd_loss):.4f} val_fwd={val_fwd:.4f} "
                      f"cos={val_cos:.4f} res_norm={val_res:.3f}")

    # --- Final detailed analysis ---
    print("\n=== Final analysis ===")
    model.eval()
    fwd_model.eval()
    analysis = _run_analysis(
        model, fwd_model, val_data, get_batch,
        predict_from, predict_to, n_eval_batches * 4, device,
    )
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # --- Save ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"gpt_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/{model_tag}/{fwd_type}_L{fwd_n_layer}/{gap_tag}/P_{n_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "n_tokens": n_tokens, "block_size": block_size,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "predict_from": predict_from, "predict_to": predict_to,
        "fwd_type": fwd_type,
        "fwd_n_layer": fwd_n_layer,
        "fwd_d_head": fwd_d_head, "fwd_n_head": fwd_n_head,
        "fwd_mlp_mult": fwd_mlp_mult,
        "best_val_loss": best_val_loss,
        "final_val_lm": history["val_lm_loss"][-1][1],
        "final_val_fwd_mse": history["val_fwd_mse"][-1][1],
        "final_val_cosine": history["val_cosine_sim"][-1][1],
        "history": history,
        "analysis": analysis,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")
    return result


def _run_analysis(model, fwd_model, val_data, get_batch,
                  predict_from, predict_to, n_batches, device):
    """Detailed residual analysis at end of training."""
    import torch
    import numpy as np

    all_residual_norms = []
    all_cos_sims = []
    all_lm_losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)

            src = vi[predict_from]
            tgt = vi[predict_to]
            pred = fwd_model(src)

            residual = tgt - pred
            res_norm = residual.norm(dim=-1)
            cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)

            per_pos_loss = model.get_ngram_losses(vx)

            all_residual_norms.append(res_norm.cpu())
            all_cos_sims.append(cos.cpu())
            all_lm_losses.append(per_pos_loss.cpu())

    residual_norms = torch.cat(all_residual_norms, dim=0).numpy()
    cos_sims = torch.cat(all_cos_sims, dim=0).numpy()
    lm_losses = torch.cat(all_lm_losses, dim=0).numpy()

    mean_res_by_pos = residual_norms.mean(axis=0).tolist()
    mean_cos_by_pos = cos_sims.mean(axis=0).tolist()

    flat_res = residual_norms[:, :lm_losses.shape[1]].flatten()
    flat_loss = lm_losses.flatten()
    correlation = float(np.corrcoef(flat_res, flat_loss)[0, 1])

    per_pos_corr = []
    for t in range(lm_losses.shape[1]):
        c = float(np.corrcoef(residual_norms[:, t], lm_losses[:, t])[0, 1])
        per_pos_corr.append(c)

    return {
        "mean_residual_norm": float(residual_norms.mean()),
        "std_residual_norm": float(residual_norms.std()),
        "mean_cosine_sim": float(cos_sims.mean()),
        "residual_lm_loss_correlation": correlation,
        "mean_residual_by_position": mean_res_by_pos,
        "mean_cosine_by_position": mean_cos_by_pos,
        "per_position_residual_loss_correlation": per_pos_corr,
    }


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_loop_train(
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
):
    """Closed-loop cerebellar training: forward model predictions
    injected back into the main model's residual stream.

    The forward model predicts a later layer from an early layer.
    Its prediction is transformed by a learned gate/projection
    (CerebellarGate) and added to the residual stream at an
    intermediate block. The main model learns to use this signal;
    the forward model learns to predict the main model's computation.
    """
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", "").replace("post_embed", "-1"))
    print(f"A2A CLOSED-LOOP training on {device}")
    print(f"  Main model: {n_layer}L {n_head}H {n_embd}D")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} → {predict_to}")
    print(f"  Injection: after block {inject_after_block}")
    print(f"  P={n_tokens:,}, T={block_size}")

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

    # --- Create models ---
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    gate = CerebellarGate(n_embd).to(device)

    opt_main = torch.optim.AdamW(
        list(model.parameters()) + list(gate.parameters()),
        lr=lr, weight_decay=0.01,
    )
    opt_fwd = torch.optim.AdamW(
        fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
    )

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- Training loop ---
    history = {
        "lm_loss": [], "lm_loss_no_injection": [], "fwd_mse": [],
        "val_lm_loop": [], "val_lm_base": [],
        "val_fwd_mse": [], "val_cosine_sim": [],
        "val_residual_norm": [],
        "gate_injection_norm": [], "val_injection_norm": [],
    }
    best_val_loss = float("inf")

    for step in range(n_steps):
        model.train()
        fwd_model.train()
        gate.train()

        x, y = get_batch(train_data)

        # Cache forward model prediction for reuse
        fwd_pred_cache = {}

        def cerebellar_fn(act):
            fwd_pred = fwd_model(act.detach())
            fwd_pred_cache["pred"] = fwd_pred
            return gate(fwd_pred.detach())

        # Single forward pass with injection
        _, lm_loss, intermediates = model(
            x, y, return_intermediates=True,
            cerebellar_fn=cerebellar_fn,
            cerebellar_input_block=cerebellar_input_block,
            cerebellar_inject_block=inject_after_block,
        )

        # Forward model MSE loss on the closed-loop target
        target = intermediates[predict_to].detach()
        fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)

        opt_main.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(gate.parameters()), 1.0,
        )
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

            gate_norm = gate.injection_norm()
            history["gate_injection_norm"].append((step, gate_norm))

            with torch.no_grad():
                val_lm_loop_acc = 0.0
                val_lm_base_acc = 0.0
                val_fwd_acc = 0.0
                val_cos_acc = 0.0
                val_res_acc = 0.0
                val_inj_acc = 0.0

                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)

                    # With injection
                    def eval_cb_fn(act):
                        return gate(fwd_model(act))
                    _, vl_loop, vi_loop = model(
                        vx, vy, return_intermediates=True,
                        cerebellar_fn=eval_cb_fn,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    val_lm_loop_acc += float(vl_loop)

                    # Without injection (baseline)
                    _, vl_base, vi_base = model(
                        vx, vy, return_intermediates=True,
                    )
                    val_lm_base_acc += float(vl_base)

                    # Forward model quality (on base intermediates)
                    src = vi_base[predict_from]
                    tgt = vi_base[predict_to]
                    pred = fwd_model(src)
                    val_fwd_acc += float(F.mse_loss(pred, tgt))
                    val_cos_acc += float(
                        F.cosine_similarity(pred, tgt, dim=-1).mean()
                    )

                    residual = tgt - pred
                    val_res_acc += float(residual.norm(dim=-1).mean())

                    inj = gate(pred)
                    val_inj_acc += float(inj.norm(dim=-1).mean())

                n = n_eval_batches
                val_lm_loop = val_lm_loop_acc / n
                val_lm_base = val_lm_base_acc / n
                val_fwd = val_fwd_acc / n
                val_cos = val_cos_acc / n
                val_res = val_res_acc / n
                val_inj = val_inj_acc / n

                if val_lm_loop < best_val_loss:
                    best_val_loss = val_lm_loop

                history["lm_loss"].append((step, float(lm_loss)))
                history["fwd_mse"].append((step, float(fwd_loss)))
                history["val_lm_loop"].append((step, val_lm_loop))
                history["val_lm_base"].append((step, val_lm_base))
                history["val_fwd_mse"].append((step, val_fwd))
                history["val_cosine_sim"].append((step, val_cos))
                history["val_residual_norm"].append((step, val_res))
                history["val_injection_norm"].append((step, val_inj))

                lm_delta = val_lm_loop - val_lm_base
                print(f"  step {step:6d}: "
                      f"lm_loop={val_lm_loop:.4f} lm_base={val_lm_base:.4f} "
                      f"Δ={lm_delta:+.4f} | "
                      f"fwd_cos={val_cos:.4f} "
                      f"gate_norm={gate_norm:.4f} inj_norm={val_inj:.4f}")

    # --- Self-map probe ---
    print("\n=== Self-map probe ===")
    selfmap = _run_selfmap_probe(
        model, fwd_model, gate, val_data, get_batch,
        predict_from, predict_to,
        cerebellar_input_block, inject_after_block,
        n_batches=20, device=device,
    )
    print(f"  With injection:    R²={selfmap['r2_with_injection']:.4f}, "
          f"cosine={selfmap['cosine_with_injection']:.4f}")
    print(f"  Without injection: R²={selfmap['r2_without_injection']:.4f}, "
          f"cosine={selfmap['cosine_without_injection']:.4f}")

    # --- Save ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_dir = (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))
    torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))

    result = {
        "n_tokens": n_tokens, "block_size": block_size,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "predict_from": predict_from, "predict_to": predict_to,
        "inject_after_block": inject_after_block,
        "fwd_n_layer": fwd_n_layer,
        "fwd_d_head": fwd_d_head, "fwd_n_head": fwd_n_head,
        "fwd_mlp_mult": fwd_mlp_mult,
        "best_val_loss": best_val_loss,
        "final_val_lm_loop": history["val_lm_loop"][-1][1],
        "final_val_lm_base": history["val_lm_base"][-1][1],
        "final_val_fwd_mse": history["val_fwd_mse"][-1][1],
        "final_val_cosine": history["val_cosine_sim"][-1][1],
        "final_gate_norm": history["gate_injection_norm"][-1][1],
        "selfmap_probe": selfmap,
        "history": history,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")
    return result


def _run_selfmap_probe(
    model, fwd_model, gate, val_data, get_batch,
    predict_from, predict_to,
    cerebellar_input_block, inject_after_block,
    n_batches, device,
):
    """Train linear probes to predict early-layer activations from
    late-layer activations, comparing with vs without injection.

    Tests whether co-training with prediction injection causes the
    model to develop more linearly accessible self-referential
    representations in its later layers.
    """
    import torch
    import torch.nn as nn

    n_embd = model.transformer.ln_f.normalized_shape[0]

    # Collect data
    early_acts = []
    late_with = []
    late_without = []

    model.eval()
    fwd_model.eval()
    gate.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            vx, vy = get_batch(val_data)

            def eval_cb_fn(act):
                return gate(fwd_model(act))

            _, _, vi_with = model(
                vx, vy, return_intermediates=True,
                cerebellar_fn=eval_cb_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )
            _, _, vi_without = model(vx, vy, return_intermediates=True)

            early_acts.append(vi_with[predict_from].cpu())
            late_with.append(vi_with[predict_to].cpu())
            late_without.append(vi_without[predict_to].cpu())

    early = torch.cat(early_acts, dim=0).reshape(-1, n_embd)
    lw = torch.cat(late_with, dim=0).reshape(-1, n_embd)
    lwo = torch.cat(late_without, dim=0).reshape(-1, n_embd)

    # Split into train/test
    n = early.shape[0]
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    results = {}
    for label, late_data in [("with_injection", lw), ("without_injection", lwo)]:
        probe = nn.Linear(n_embd, n_embd).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        X_train = late_data[train_idx].to(device)
        Y_train = early[train_idx].to(device)
        X_test = late_data[test_idx].to(device)
        Y_test = early[test_idx].to(device)

        # Train probe
        probe_bs = min(4096, n_train)
        for _ in range(200):
            idx = torch.randint(n_train, (probe_bs,))
            pred = probe(X_train[idx])
            loss = torch.nn.functional.mse_loss(pred, Y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        with torch.no_grad():
            pred_test = probe(X_test)
            mse = torch.nn.functional.mse_loss(pred_test, Y_test).item()
            var = Y_test.var().item()
            r2 = 1.0 - mse / var if var > 0 else 0.0
            cos = torch.nn.functional.cosine_similarity(
                pred_test, Y_test, dim=-1
            ).mean().item()

        results[f"r2_{label}"] = r2
        results[f"cosine_{label}"] = cos

    return results


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
def train(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_steps: int = 10_000,
    lr: float = 3e-4,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    fwd_type: str = "transformer",
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_train.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps, lr=lr,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        fwd_type=fwd_type, predict_from=predict_from, predict_to=predict_to,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print(f"A2A training complete:")
    print(f"  best_val_loss={result['best_val_loss']:.4f}")
    print(f"  final_fwd_mse={result['final_val_fwd_mse']:.4f}")
    print(f"  final_cosine={result['final_val_cosine']:.4f}")
    if "analysis" in result:
        a = result["analysis"]
        print(f"  residual-LM correlation={a.get('residual_lm_loss_correlation', 0):.4f}")


@app.local_entrypoint()
def loop_train(
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
    result = a2a_loop_train.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps, lr=lr,
        fwd_lr=1e-3,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print(f"A2A closed-loop training complete:")
    print(f"  best_val_loss={result['best_val_loss']:.4f}")
    print(f"  final lm_loop={result['final_val_lm_loop']:.4f} "
          f"lm_base={result['final_val_lm_base']:.4f}")
    print(f"  final_fwd_cosine={result['final_val_cosine']:.4f}")
    print(f"  gate_norm={result['final_gate_norm']:.4f}")
    sm = result.get("selfmap_probe", {})
    if sm:
        print(f"  self-map R² with_inj={sm.get('r2_with_injection', 0):.4f} "
              f"without={sm.get('r2_without_injection', 0):.4f}")
