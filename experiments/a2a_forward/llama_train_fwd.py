"""Train a forward model on cached Llama 3.2 1B activations.

Stage 2 of the Llama-scale A2A experiment. Trains a transformer forward
model on pre-cached (source_layer, target_layer) activation pairs, without
needing the Llama model itself. Replicates the toy model's Run 2 (1-layer
gap, transformer forward model) at Llama scale (d=2048, 1.24B param model).

The forward model is ~1.4% of Llama's parameters (17.8M at default config),
creating a capacity bottleneck. The residual (actual - predicted) captures
computational novelty: what the main model computes that the forward model's
capacity can't anticipate.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=65536,
)
def a2a_train_llama_fwd(
    source_layer: int = 7,
    target_layer: int = 8,
    seq_len: int = 2048,
    batch_size: int = 16,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 200,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 128,
    fwd_mlp_mult: int = 2,
    n_val_shards: int = 2,
    steps_per_shard: int = 100,
):
    """Train a transformer forward model on cached Llama activations.

    Streams shards from the volume: loads one train shard at a time, trains
    for `steps_per_shard` steps (random sampling with replacement), then
    rotates. Val shards are preloaded once at startup for fast evaluation.
    """
    import os
    import time
    import glob
    import numpy as np
    import torch
    import torch.nn.functional as F
    from a2a_forward.forward_model import (
        TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts_dir = f"{DATA_DIR}/a2a_llama/acts_L{source_layer}_L{target_layer}"

    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)

    d_model = meta["d_model"]
    print(f"=== Llama A2A Forward Model Training ===")
    print(f"  Source: {meta['model_name']} ({meta['n_model_params']/1e9:.2f}B params)")
    print(f"  Gap: post_block_{source_layer} -> post_block_{target_layer}")
    print(f"  d_model={d_model}, seq_len={seq_len}")
    print(f"  Cached: {meta['n_tokens_cached']:,} tokens")

    # --- Discover and split shards ---
    shard_paths = sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz")))
    n_shards = len(shard_paths)
    n_val = min(n_val_shards, max(1, n_shards // 10))
    train_paths = shard_paths[:n_shards - n_val]
    val_paths = shard_paths[n_shards - n_val:]
    print(f"  Shards: {n_shards} total, {len(train_paths)} train, {len(val_paths)} val")

    # --- Shard loading ---
    def load_shard_sequences(path):
        """Load npz shard -> (N_seqs, seq_len, d_model) float16 tensors."""
        data = np.load(path)
        source = data["source"]
        target = data["target"]
        n_seqs = source.shape[0] // seq_len
        if n_seqs == 0:
            return None, None
        source = source[:n_seqs * seq_len].reshape(n_seqs, seq_len, d_model)
        target = target[:n_seqs * seq_len].reshape(n_seqs, seq_len, d_model)
        return torch.from_numpy(source), torch.from_numpy(target)

    # --- Preload val data ---
    print("  Loading val shards...")
    t0 = time.time()
    val_sources, val_targets = [], []
    for vp in val_paths:
        vs, vt = load_shard_sequences(vp)
        if vs is not None:
            val_sources.append(vs)
            val_targets.append(vt)
    val_source = torch.cat(val_sources, dim=0)
    val_target = torch.cat(val_targets, dim=0)
    n_val_seqs = val_source.shape[0]
    print(f"  Val: {n_val_seqs} sequences ({n_val_seqs * seq_len / 1e6:.1f}M tokens) "
          f"loaded in {time.time() - t0:.1f}s")
    del val_sources, val_targets

    # --- Create forward model ---
    fwd_model = TransformerForwardModel(
        d_model=d_model,
        d_head=fwd_d_head,
        n_head=fwd_n_head,
        n_layer=fwd_n_layer,
        mlp_mult=fwd_mlp_mult,
        block_size=seq_len,
    ).to(device)

    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    main_n_params = meta["n_model_params"]
    capacity_ratio = fwd_n_params / main_n_params
    print(f"  Forward model: {fwd_n_params:,} params ({capacity_ratio:.2%} of main)")

    optimizer = torch.optim.AdamW(
        fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
    )

    # --- Eval function ---
    def evaluate(max_batches=None):
        fwd_model.eval()
        mse_acc, cos_acc, res_acc = 0.0, 0.0, 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, n_val_seqs, batch_size):
                if max_batches and n_batches >= max_batches:
                    break
                src = val_source[i:i + batch_size].float().to(device)
                tgt = val_target[i:i + batch_size].float().to(device)
                pred = fwd_model(src)
                mse_acc += float(F.mse_loss(pred, tgt))
                cos_acc += float(F.cosine_similarity(pred, tgt, dim=-1).mean())
                res_acc += float((tgt - pred).norm(dim=-1).mean())
                n_batches += 1
        if n_batches == 0:
            return float("nan"), float("nan"), float("nan")
        return mse_acc / n_batches, cos_acc / n_batches, res_acc / n_batches

    # --- Training loop ---
    history = {
        "fwd_mse": [], "val_fwd_mse": [], "val_cosine_sim": [],
        "val_residual_norm": [],
    }

    train_shard_idx = 0
    cur_source = None
    cur_target = None
    cur_n_seqs = 0
    steps_on_shard = 0

    def rotate_train_shard():
        nonlocal train_shard_idx, cur_source, cur_target, cur_n_seqs, steps_on_shard
        path = train_paths[train_shard_idx % len(train_paths)]
        cur_source, cur_target = load_shard_sequences(path)
        cur_n_seqs = cur_source.shape[0] if cur_source is not None else 0
        steps_on_shard = 0
        train_shard_idx += 1

    rotate_train_shard()
    t_start = time.time()

    for step in range(n_steps):
        if steps_on_shard >= steps_per_shard:
            rotate_train_shard()

        fwd_model.train()
        idx = torch.randint(cur_n_seqs, (batch_size,))
        src = cur_source[idx].float().to(device)
        tgt = cur_target[idx].float().to(device)

        pred = fwd_model(src)
        loss = F.mse_loss(pred, tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        optimizer.step()

        steps_on_shard += 1

        if step % eval_interval == 0 or step == n_steps - 1:
            val_mse, val_cos, val_res = evaluate(max_batches=30)

            history["fwd_mse"].append((step, float(loss)))
            history["val_fwd_mse"].append((step, val_mse))
            history["val_cosine_sim"].append((step, val_cos))
            history["val_residual_norm"].append((step, val_res))

            elapsed = time.time() - t_start
            tokens_seen = (step + 1) * batch_size * seq_len
            print(f"  step {step:6d} ({elapsed:5.0f}s, {tokens_seen/1e6:.0f}M tok): "
                  f"mse={float(loss):.6f} val_mse={val_mse:.6f} "
                  f"cos={val_cos:.4f} res={val_res:.3f}")

    # --- Final full eval ---
    print("\n=== Final evaluation (all val data) ===")
    final_mse, final_cos, final_res = evaluate(max_batches=None)
    print(f"  MSE={final_mse:.6f}  cosine={final_cos:.4f}  residual_norm={final_res:.3f}")

    # --- Residual PCA on val data ---
    print("\n=== Residual structure ===")
    all_residuals = []
    fwd_model.eval()
    with torch.no_grad():
        for i in range(0, min(n_val_seqs, 200), batch_size):
            src = val_source[i:i + batch_size].float().to(device)
            tgt = val_target[i:i + batch_size].float().to(device)
            pred = fwd_model(src)
            all_residuals.append((tgt - pred).cpu())
    residuals = torch.cat(all_residuals, dim=0)
    residual_flat = residuals.reshape(-1, d_model).numpy()

    rng = np.random.RandomState(42)
    n_sub = min(50000, residual_flat.shape[0])
    idx_sub = rng.choice(residual_flat.shape[0], n_sub, replace=False)
    residual_sub = residual_flat[idx_sub]
    residual_sub = residual_sub - residual_sub.mean(axis=0)
    _, S, _ = np.linalg.svd(residual_sub, full_matrices=False)
    S_norm = S / S.sum()
    effective_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    rank_50 = int(np.searchsorted(cumvar, 0.5) + 1)
    rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    top1_var = float(cumvar[0])

    print(f"  Effective rank: {effective_rank:.1f} / {d_model}")
    print(f"  Top-1 PC explains: {top1_var:.3%}")
    print(f"  Rank for 50%: {rank_50}, 90%: {rank_90}, 95%: {rank_95}")

    residual_pca = {
        "effective_rank": effective_rank,
        "max_rank": d_model,
        "rank_50": rank_50,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "top1_variance": top1_var,
    }

    # --- Save ---
    gap_tag = f"L{source_layer}_to_L{target_layer}"
    save_dir = (f"{DATA_DIR}/a2a_llama/fwd_{fwd_n_layer}L_{fwd_n_head}H_"
                f"{fwd_d_head}d_mlp{fwd_mlp_mult}/{gap_tag}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    elapsed_total = time.time() - t_start
    result = {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "d_model": d_model,
        "seq_len": seq_len,
        "fwd_n_layer": fwd_n_layer,
        "fwd_n_head": fwd_n_head,
        "fwd_d_head": fwd_d_head,
        "fwd_mlp_mult": fwd_mlp_mult,
        "fwd_n_params": fwd_n_params,
        "main_n_params": main_n_params,
        "capacity_ratio": capacity_ratio,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "fwd_lr": fwd_lr,
        "n_train_shards": len(train_paths),
        "n_val_shards": len(val_paths),
        "n_val_sequences": n_val_seqs,
        "final_val_mse": final_mse,
        "final_val_cosine": final_cos,
        "final_val_residual_norm": final_res,
        "residual_pca": residual_pca,
        "training_seconds": elapsed_total,
        "history": history,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)

    volume.commit()
    print(f"\n=== Done ({elapsed_total:.0f}s) ===")
    print(f"  Cosine: {final_cos:.4f}")
    print(f"  MSE: {final_mse:.6f}")
    print(f"  Residual norm: {final_res:.3f}")
    print(f"  Effective rank: {effective_rank:.1f}/{d_model}")
    print(f"  Saved to {save_dir}")
    return result


@app.local_entrypoint()
def main(
    source_layer: int = 7,
    target_layer: int = 8,
    seq_len: int = 2048,
    n_steps: int = 10_000,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 64,
    fwd_mlp_mult: int = 2,
):
    result = a2a_train_llama_fwd.remote(
        source_layer=source_layer,
        target_layer=target_layer,
        seq_len=seq_len,
        batch_size=16,
        fwd_lr=1e-3,
        n_steps=n_steps,
        fwd_n_layer=fwd_n_layer,
        fwd_n_head=fwd_n_head,
        fwd_d_head=fwd_d_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print(f"Llama forward model training complete:")
    print(f"  Capacity: {result['fwd_n_params']:,} params "
          f"({result['capacity_ratio']:.2%} of main)")
    print(f"  Cosine: {result['final_val_cosine']:.4f}")
    print(f"  MSE: {result['final_val_mse']:.6f}")
    print(f"  Residual norm: {result['final_val_residual_norm']:.3f}")
    pca = result["residual_pca"]
    print(f"  Effective rank: {pca['effective_rank']:.1f}/{pca['max_rank']}")
    print(f"  Training time: {result['training_seconds']:.0f}s")


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
