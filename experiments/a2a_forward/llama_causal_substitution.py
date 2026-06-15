"""Causal substitution analysis for the A2A forward model on Llama 3.2 1B.

Llama-scale version of causal_substitution.py. Runs Llama in three modes:
  - Normal:      full model, unmodified
  - Substituted: replace target_layer's output with the forward model's
                 prediction from source_layer's activations
  - Ablated:     skip target_layer entirely (output = input)

Uses forward hooks to intercept and modify hidden states, avoiding the need
to manually replicate causal mask and RoPE computation.
"""

import json

import modal
from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

llama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
        "transformers",
        "huggingface-hub",
        "accelerate",
        "matplotlib",
    )
    .add_local_python_source("a2a_forward")
)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L40S",
    timeout=7200,
    memory=65536,
    image=llama_image,
)
def a2a_llama_causal_substitution(
    source_layer: int = 7,
    target_layer: int = 8,
    model_name: str = "unsloth/Llama-3.2-1B",
    seq_len: int = 2048,
    batch_size: int = 4,
    n_eval_batches: int = 100,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 64,
    fwd_mlp_mult: float = 2,
    fwd_use_swiglu: bool = False,
):
    import os
    import time
    import glob
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"

    # --- Load Llama ---
    t0 = time.time()
    print(f"=== Llama Causal Substitution ===")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params, {n_layers} layers, d={d_model}, vocab={vocab_size}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Load forward model ---
    acts_dir = f"{DATA_DIR}/a2a_llama/acts_L{source_layer}_L{target_layer}"
    gap_tag = f"L{source_layer}_to_L{target_layer}"
    prefix = f"swiglu" if fwd_use_swiglu else f"mlp"
    # Try normalized (int) and raw (float) path variants
    candidates = []
    for mm_fmt in [str(int(fwd_mlp_mult)) if fwd_mlp_mult == int(fwd_mlp_mult) else None,
                   str(fwd_mlp_mult)]:
        if mm_fmt is None:
            continue
        d = (f"{DATA_DIR}/a2a_llama/fwd_{fwd_n_layer}L_{fwd_n_head}H_"
             f"{fwd_d_head}d_{prefix}{mm_fmt}/{gap_tag}")
        candidates.append(d)
    fwd_dir = None
    for d in candidates:
        if os.path.exists(os.path.join(d, "fwd_model.pt")):
            fwd_dir = d
            break
    if fwd_dir is None:
        raise FileNotFoundError(
            f"No forward model checkpoint found at any of: {candidates}")

    fwd_model = TransformerForwardModel(
        d_model=d_model, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=seq_len,
        use_swiglu=fwd_use_swiglu,
    ).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(fwd_dir, "fwd_model.pt"),
        map_location=device, weights_only=True,
    ))
    fwd_model.eval()
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    print(f"  Forward model: {fwd_n_params:,} params ({fwd_n_params/n_params:.2%} of main)")
    print(f"  Loaded from {fwd_dir}")

    # --- Load eval data from cached activation shards ---
    shard_paths = sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz")))
    n_shards = len(shard_paths)
    val_paths = shard_paths[n_shards - 2:]

    print(f"  Loading eval token_ids from {len(val_paths)} val shards...")
    all_tokens = []
    for path in val_paths:
        data = np.load(path)
        all_tokens.append(data["token_ids"])
    tokens_flat = np.concatenate(all_tokens)

    n_seqs = len(tokens_flat) // seq_len
    tokens = tokens_flat[:n_seqs * seq_len].reshape(n_seqs, seq_len)
    print(f"  Eval data: {n_seqs} sequences ({n_seqs * seq_len / 1e6:.1f}M tokens)")

    # --- Token frequency from training shards (for category analysis) ---
    print("  Computing token frequencies from training shards...")
    freq_counts = np.zeros(vocab_size, dtype=np.int64)
    for path in shard_paths[:min(10, n_shards - 2)]:
        ids = np.load(path)["token_ids"].astype(np.int64)
        valid = ids[ids < vocab_size]
        freq_counts += np.bincount(valid, minlength=vocab_size)
    sorted_by_freq = np.argsort(-freq_counts)
    token_rank = np.empty(vocab_size, dtype=np.int64)
    token_rank[sorted_by_freq] = np.arange(vocab_size)
    top5pct = int(0.05 * vocab_size)
    bottom20pct = int(0.80 * vocab_size)

    # --- Punctuation token set ---
    punct_ids = set()
    for tid in range(min(vocab_size, 200_000)):
        try:
            s = tokenizer.decode([tid]).strip()
        except Exception:
            continue
        if len(s) == 1 and s in ".,!?;:—-…\"'`()[]{}":
            punct_ids.add(tid)
    print(f"  Punctuation tokens: {len(punct_ids)}")

    # --- Define hooks ---
    # In transformers 5.x, decoder layers return a plain tensor (not a tuple)
    # when use_cache=False and output_attentions=False. Handle both formats.
    def make_sub_hook(fwd_model_ref):
        def hook(module, input, output):
            src = input[0]
            pred = fwd_model_ref(src.float()).to(src.dtype)
            if isinstance(output, tuple):
                return (pred,) + output[1:]
            return pred
        return hook

    def abl_hook(module, input, output):
        if isinstance(output, tuple):
            return (input[0],) + output[1:]
        return input[0]

    target_layer_module = model.model.layers[target_layer]
    sub_hook_fn = make_sub_hook(fwd_model)

    # --- Verification on first batch ---
    print("\n--- Verification ---")
    first_ids = torch.from_numpy(tokens[:batch_size]).long().to(device)
    with torch.no_grad():
        def verify_hook(module, input, output):
            src = input[0].float()
            tgt = (output[0] if isinstance(output, tuple) else output).float()
            pred = fwd_model(src)
            cos = F.cosine_similarity(pred, tgt, dim=-1).mean()
            print(f"  Forward model cosine vs actual layer {target_layer}: {cos:.4f}")
            print(f"  Residual norm: {(tgt - pred).norm(dim=-1).mean():.3f}")
            return output

        h = target_layer_module.register_forward_hook(verify_hook)
        model(first_ids, use_cache=False)
        h.remove()

    # --- Main eval loop ---
    all_ce_n, all_ce_s, all_ce_a = [], [], []
    all_acc_n, all_acc_s, all_acc_a = [], [], []
    all_kl_sub, all_kl_abl = [], []
    all_conf_n = []
    all_target_ids = []

    n_batches = min(n_eval_batches, n_seqs // batch_size)
    print(f"\nRunning {n_batches} eval batches (batch_size={batch_size}, "
          f"total={n_batches * batch_size * (seq_len - 1):,} tokens)...")
    t_eval = time.time()

    with torch.no_grad():
        for bi in range(n_batches):
            if bi % 20 == 0:
                elapsed = time.time() - t_eval
                print(f"  batch {bi}/{n_batches} ({elapsed:.0f}s)")

            start = bi * batch_size
            input_ids = torch.from_numpy(
                tokens[start:start + batch_size]
            ).long().to(device)
            target_ids = input_ids[:, 1:]
            flat_t = target_ids.reshape(-1)

            # --- Normal ---
            logits_n = model(input_ids, use_cache=False).logits[:, :-1, :]
            log_p_n = F.log_softmax(logits_n.float(), dim=-1).reshape(-1, vocab_size)
            p_n = log_p_n.exp()
            del logits_n

            ce_n = F.nll_loss(log_p_n, flat_t, reduction="none")
            acc_n = (log_p_n.argmax(-1) == flat_t).float()
            conf_n = p_n.gather(1, flat_t.unsqueeze(1)).squeeze(1)

            # --- Substituted ---
            h = target_layer_module.register_forward_hook(sub_hook_fn)
            logits_s = model(input_ids, use_cache=False).logits[:, :-1, :]
            h.remove()
            log_p_s = F.log_softmax(logits_s.float(), dim=-1).reshape(-1, vocab_size)
            del logits_s

            ce_s = F.nll_loss(log_p_s, flat_t, reduction="none")
            acc_s = (log_p_s.argmax(-1) == flat_t).float()
            kl_sub = (p_n * (log_p_n - log_p_s)).sum(-1).clamp(min=0)
            del log_p_s

            # --- Ablated ---
            h = target_layer_module.register_forward_hook(abl_hook)
            logits_a = model(input_ids, use_cache=False).logits[:, :-1, :]
            h.remove()
            log_p_a = F.log_softmax(logits_a.float(), dim=-1).reshape(-1, vocab_size)
            del logits_a

            ce_a = F.nll_loss(log_p_a, flat_t, reduction="none")
            acc_a = (log_p_a.argmax(-1) == flat_t).float()
            kl_abl = (p_n * (log_p_n - log_p_a)).sum(-1).clamp(min=0)
            del log_p_a, log_p_n, p_n

            all_ce_n.append(ce_n.cpu())
            all_ce_s.append(ce_s.cpu())
            all_ce_a.append(ce_a.cpu())
            all_acc_n.append(acc_n.cpu())
            all_acc_s.append(acc_s.cpu())
            all_acc_a.append(acc_a.cpu())
            all_kl_sub.append(kl_sub.cpu())
            all_kl_abl.append(kl_abl.cpu())
            all_conf_n.append(conf_n.cpu())
            all_target_ids.append(flat_t.cpu())

            torch.cuda.empty_cache()

    eval_time = time.time() - t_eval
    print(f"  Eval complete in {eval_time:.0f}s")

    # --- Aggregate ---
    ce_n = torch.cat(all_ce_n)
    ce_s = torch.cat(all_ce_s)
    ce_a = torch.cat(all_ce_a)
    acc_n = torch.cat(all_acc_n)
    acc_s = torch.cat(all_acc_s)
    acc_a = torch.cat(all_acc_a)
    kl_sub = torch.cat(all_kl_sub)
    kl_abl = torch.cat(all_kl_abl)
    conf_n = torch.cat(all_conf_n)
    target_ids_flat = torch.cat(all_target_ids)
    M = ce_n.shape[0]

    # --- Overall ---
    overall = {
        "n": int(M),
        "ce_normal": float(ce_n.mean()),
        "ce_sub": float(ce_s.mean()),
        "ce_abl": float(ce_a.mean()),
        "delta_ce_sub": float((ce_s - ce_n).mean()),
        "delta_ce_abl": float((ce_a - ce_n).mean()),
        "acc_normal": float(acc_n.mean()),
        "acc_sub": float(acc_s.mean()),
        "acc_abl": float(acc_a.mean()),
        "kl_sub": float(kl_sub.mean()),
        "kl_abl": float(kl_abl.mean()),
        "kl_recovery": float(1.0 - kl_sub.mean() / kl_abl.mean()),
    }

    print(f"\n=== Overall ({M:,} tokens) ===")
    print(f"  Acc:  normal={overall['acc_normal']:.4f}  "
          f"sub={overall['acc_sub']:.4f}  abl={overall['acc_abl']:.4f}")
    print(f"  KL:   sub={overall['kl_sub']:.4f}  abl={overall['kl_abl']:.4f}  "
          f"recovery={overall['kl_recovery']:.1%}")
    print(f"  ΔCE:  sub={overall['delta_ce_sub']:+.4f}  "
          f"abl={overall['delta_ce_abl']:+.4f}")

    # --- Behavioral categories ---
    y_np = target_ids_flat.numpy()
    y_rank = token_rank[y_np]

    masks = {}
    masks["high_confidence"] = conf_n > 0.5
    masks["low_confidence"] = conf_n < 0.1
    masks["function_word"] = torch.from_numpy(y_rank < top5pct)
    masks["content_word"] = torch.from_numpy(y_rank >= bottom20pct)

    punct_mask = torch.zeros(M, dtype=torch.bool)
    for tid in punct_ids:
        punct_mask |= (target_ids_flat == tid)
    masks["punctuation"] = punct_mask

    def category_stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "ce_normal": float(ce_n[mask].mean()),
            "ce_sub": float(ce_s[mask].mean()),
            "ce_abl": float(ce_a[mask].mean()),
            "delta_ce_sub": float((ce_s - ce_n)[mask].mean()),
            "delta_ce_abl": float((ce_a - ce_n)[mask].mean()),
            "acc_normal": float(acc_n[mask].mean()),
            "acc_sub": float(acc_s[mask].mean()),
            "acc_abl": float(acc_a[mask].mean()),
            "kl_sub": float(kl_sub[mask].mean()),
            "kl_abl": float(kl_abl[mask].mean()),
        }

    print(f"\n=== Per-category ===")
    category_results = {}
    for name, mask in masks.items():
        stats = category_stats(mask)
        category_results[name] = stats
        if stats["n"] > 0:
            print(f"  {name:20s} (n={stats['n']:>8,}): "
                  f"KL_sub={stats['kl_sub']:.4f}  KL_abl={stats['kl_abl']:.4f}  "
                  f"ΔCE_sub={stats['delta_ce_sub']:+.4f}")

    # --- Per-position statistics ---
    T_eval = seq_len - 1
    n_seqs_eval = M // T_eval

    per_position = {}
    if n_seqs_eval > 0:
        kl_s_2d = kl_sub[:n_seqs_eval * T_eval].reshape(n_seqs_eval, T_eval)
        kl_a_2d = kl_abl[:n_seqs_eval * T_eval].reshape(n_seqs_eval, T_eval)
        ce_n_2d = ce_n[:n_seqs_eval * T_eval].reshape(n_seqs_eval, T_eval)
        ce_s_2d = ce_s[:n_seqs_eval * T_eval].reshape(n_seqs_eval, T_eval)
        ce_a_2d = ce_a[:n_seqs_eval * T_eval].reshape(n_seqs_eval, T_eval)

        per_position = {
            "kl_sub": kl_s_2d.mean(dim=0).tolist(),
            "kl_abl": kl_a_2d.mean(dim=0).tolist(),
            "delta_ce_sub": (ce_s_2d - ce_n_2d).mean(dim=0).tolist(),
            "delta_ce_abl": (ce_a_2d - ce_n_2d).mean(dim=0).tolist(),
            "ce_normal": ce_n_2d.mean(dim=0).tolist(),
        }

    # --- Generate figure ---
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Llama 3.2 1B — Causal Substitution (layer {source_layer}→{target_layer})\n"
        f"Forward model: {fwd_n_params/1e6:.1f}M params ({fwd_n_params/n_params:.1%} of main)",
        fontsize=13, y=0.995,
    )

    # (0,0) Overall: accuracy and KL
    ax = axes[0, 0]
    modes = ["Normal", "Substituted", "Ablated"]
    accs = [overall["acc_normal"], overall["acc_sub"], overall["acc_abl"]]
    colors = ["forestgreen", "steelblue", "coral"]
    bars = ax.bar(modes, accs, color=colors)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Accuracy by mode")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # (0,1) KL divergence
    ax = axes[0, 1]
    kl_vals = [overall["kl_sub"], overall["kl_abl"]]
    bars = ax.bar(["Substituted", "Ablated"], kl_vals,
                  color=["steelblue", "coral"])
    ax.set_ylabel("Mean KL divergence from normal")
    ax.set_title(f"KL divergence (recovery = {overall['kl_recovery']:.1%})")
    for bar, val in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # (1,0) Per-category KL
    ax = axes[1, 0]
    valid_cats = [c for c in category_results if category_results[c]["n"] > 0]
    x_pos = np.arange(len(valid_cats))
    width = 0.35
    kl_s_vals = [category_results[c]["kl_sub"] for c in valid_cats]
    kl_a_vals = [category_results[c]["kl_abl"] for c in valid_cats]
    ax.bar(x_pos - width / 2, kl_s_vals, width, label="Substituted", color="steelblue")
    ax.bar(x_pos + width / 2, kl_a_vals, width, label="Ablated", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("KL by behavioral category")
    ax.legend(fontsize=8)

    # (1,1) Per-position KL (smoothed)
    ax = axes[1, 1]
    if per_position:
        positions = np.arange(T_eval)
        window = 32
        def smooth(arr):
            kernel = np.ones(window) / window
            return np.convolve(arr, kernel, mode="valid")
        pos_smooth = positions[window // 2: window // 2 + len(smooth(per_position["kl_sub"]))]
        ax.plot(pos_smooth, smooth(per_position["kl_sub"]),
                label="KL(normal‖sub)", color="steelblue", linewidth=1)
        ax.plot(pos_smooth, smooth(per_position["kl_abl"]),
                label="KL(normal‖abl)", color="coral", linewidth=1)
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("KL divergence")
        ax.set_title(f"KL by position (smoothed, window={window})")
        ax.legend(fontsize=8)

    plt.tight_layout()

    save_dir = f"{DATA_DIR}/a2a_llama/analysis"
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "causal_substitution.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")

    # --- Save results ---
    results = {
        "overall": overall,
        "per_category": category_results,
        "per_position": per_position,
        "config": {
            "model_name": model_name,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "n_model_params": int(n_params),
            "fwd_n_params": int(fwd_n_params),
            "fwd_capacity_ratio": fwd_n_params / n_params,
            "fwd_config": {
                "n_layer": fwd_n_layer,
                "n_head": fwd_n_head,
                "d_head": fwd_d_head,
                "mlp_mult": fwd_mlp_mult,
                "use_swiglu": fwd_use_swiglu,
            },
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_eval_batches": n_batches,
            "total_tokens_evaluated": int(M),
            "eval_seconds": eval_time,
        },
    }

    results_path = os.path.join(save_dir, "causal_substitution.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {results_path}")
    return results


@app.local_entrypoint()
def main(
    source_layer: int = 7,
    target_layer: int = 8,
    model_name: str = "unsloth/Llama-3.2-1B",
    seq_len: int = 2048,
    batch_size: int = 4,
    n_eval_batches: int = 100,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 64,
    fwd_mlp_mult: float = 2,
    fwd_use_swiglu: bool = False,
):
    result = a2a_llama_causal_substitution.remote(
        source_layer=source_layer,
        target_layer=target_layer,
        model_name=model_name,
        seq_len=seq_len,
        batch_size=batch_size,
        n_eval_batches=n_eval_batches,
        fwd_n_layer=fwd_n_layer,
        fwd_n_head=fwd_n_head,
        fwd_d_head=fwd_d_head,
        fwd_mlp_mult=fwd_mlp_mult,
        fwd_use_swiglu=fwd_use_swiglu,
    )
    ov = result["overall"]
    print(f"\nLlama causal substitution results:")
    print(f"  Acc:  normal={ov['acc_normal']:.4f}  sub={ov['acc_sub']:.4f}  "
          f"abl={ov['acc_abl']:.4f}")
    print(f"  KL:   sub={ov['kl_sub']:.4f}  abl={ov['kl_abl']:.4f}  "
          f"recovery={ov['kl_recovery']:.1%}")
    print(f"  ΔCE:  sub={ov['delta_ce_sub']:+.4f}  abl={ov['delta_ce_abl']:+.4f}")
    print(f"\n  Per category (KL_sub / KL_abl):")
    for cat, stats in result["per_category"].items():
        if stats["n"] > 0:
            print(f"    {cat:20s}: {stats['kl_sub']:.4f} / {stats['kl_abl']:.4f}")
