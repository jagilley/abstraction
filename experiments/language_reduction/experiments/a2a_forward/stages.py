"""A2A forward model experiment stages.

Co-trains a small transformer forward model (the "cerebellum") alongside a GPT.
The forward model predicts post-block1 activations from post-block0 activations
using a capacity-bottlenecked 1-layer transformer. The residual captures
computational novelty (what the forward model's capacity can't represent),
not positional blindness.
"""

import json

from language_reduction.shared import app, volume, DATA_DIR


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
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    import os
    import glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.experiments.a2a_forward.forward_model import (
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
            mlp_mult=fwd_mlp_mult, block_size=block_size,
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
    save_dir = f"{DATA_DIR}/a2a_forward/{fwd_type}/P_{n_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "n_tokens": n_tokens, "block_size": block_size,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "predict_from": predict_from, "predict_to": predict_to,
        "fwd_type": fwd_type,
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
