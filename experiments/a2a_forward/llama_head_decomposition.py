"""Per-head decomposition of what the A2A forward model captures in Llama 3.2 1B.

Decomposes layer 8's computation into its 32 attention head contributions + MLP,
then measures how each component relates to the forward model's prediction residual.
The forward model's limited capacity forces it to prioritize certain components,
and the capture profile reveals which computations are most predictable from the
preceding layer state.

Llama 3.2 1B layer 8 (GQA: 32 query heads, 8 KV heads, d_head=64):
  output = input + attn_out + mlp_out
  attn_out = sum(head_i_contribution for i in range(32))
  head_i_contribution = per_head_attn_output_i @ o_proj.weight[:, i*64:(i+1)*64].T

Per-component metrics:
  - mean_norm: how large is this component's contribution
  - cos_with_residual: directional alignment with what the forward model missed
  - cos_with_fwd_delta: directional alignment with what the forward model predicted
  - corr_norm_with_residual_norm: when this component is large, is the error large?
  - residual_variance_fraction: fraction of residual variance in this component's
    output subspace
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
def a2a_llama_head_decomposition(
    source_layer: int = 7,
    target_layer: int = 8,
    model_name: str = "unsloth/Llama-3.2-1B",
    seq_len: int = 2048,
    batch_size: int = 4,
    n_eval_batches: int = 80,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 128,
    fwd_mlp_mult: float = 2,
    fwd_use_swiglu: bool = False,
):
    import os
    import time
    import glob
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"

    # --- Load Llama ---
    t0 = time.time()
    print("=== Llama Per-Head Decomposition ===")
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    d_model = model.config.hidden_size
    n_llama_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    d_head = d_model // n_llama_heads
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params, d={d_model}, "
          f"{n_llama_heads} Q heads, {n_kv_heads} KV heads, d_head={d_head}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Load forward model ---
    acts_dir = f"{DATA_DIR}/a2a_llama/acts_L{source_layer}_L{target_layer}"
    gap_tag = f"L{source_layer}_to_L{target_layer}"
    prefix = "swiglu" if fwd_use_swiglu else "mlp"

    def find_checkpoint(d_h):
        """Try normalized (int) and raw (float) path variants."""
        for mm_fmt in [str(int(fwd_mlp_mult)) if fwd_mlp_mult == int(fwd_mlp_mult) else None,
                       str(fwd_mlp_mult)]:
            if mm_fmt is None:
                continue
            d = (f"{DATA_DIR}/a2a_llama/fwd_{fwd_n_layer}L_{fwd_n_head}H_"
                 f"{d_h}d_{prefix}{mm_fmt}/{gap_tag}")
            if os.path.exists(os.path.join(d, "fwd_model.pt")):
                return d
        return None

    fwd_dir = find_checkpoint(fwd_d_head)
    if fwd_dir is None:
        print(f"  WARNING: checkpoint not found for d_head={fwd_d_head}")
        print(f"  Checking for d_head=64 fallback...")
        fwd_dir = find_checkpoint(64)
        if fwd_dir is not None:
            fwd_d_head = 64
        else:
            raise FileNotFoundError(
                f"No forward model checkpoint found for d_head={fwd_d_head} or 64")

    fwd_model = TransformerForwardModel(
        d_model=d_model, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=seq_len,
        use_swiglu=fwd_use_swiglu,
    ).to(device)
    ckpt_path = os.path.join(fwd_dir, "fwd_model.pt")

    fwd_model.load_state_dict(torch.load(
        ckpt_path, map_location=device, weights_only=True,
    ))
    fwd_model.eval()
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    print(f"  Forward model: {fwd_n_params:,} params, d_head={fwd_d_head}")
    print(f"  Loaded from {fwd_dir}")

    # --- Load eval token_ids from cached activation shards ---
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
    print(f"  Eval: {n_seqs} sequences ({n_seqs * seq_len / 1e6:.1f}M tokens)")

    # --- Get o_proj weight for per-head decomposition ---
    # o_proj: nn.Linear(n_heads * d_head, d_model), no bias in Llama
    # weight shape: (d_model, n_heads * d_head) = (2048, 2048)
    target_layer_module = model.model.layers[target_layer]
    o_proj_weight = target_layer_module.self_attn.o_proj.weight.detach().float()
    print(f"  o_proj weight: {o_proj_weight.shape}, "
          f"bias={target_layer_module.self_attn.o_proj.bias is not None}")

    # Pre-compute orthonormal bases for each head's output subspace
    # For measuring what fraction of the residual lies in each head's subspace
    head_bases = []
    for hi in range(n_llama_heads):
        cols = o_proj_weight[:, hi * d_head:(hi + 1) * d_head]  # (D, d_head)
        Q, _ = torch.linalg.qr(cols)  # (D, d_head), orthonormal
        head_bases.append(Q.to(device))

    # --- Hook setup ---
    captured = {}

    def capture_attn_pre_proj(module, input, output):
        captured["attn_pre_proj"] = input[0].detach().float()

    def capture_mlp_output(module, input, output):
        captured["mlp_out"] = output.detach().float()

    h_attn = target_layer_module.self_attn.o_proj.register_forward_hook(
        capture_attn_pre_proj)
    h_mlp = target_layer_module.mlp.register_forward_hook(capture_mlp_output)

    # --- Accumulators ---
    n_components = n_llama_heads + 1  # 32 heads + MLP
    comp_norm_sum = torch.zeros(n_components, device=device)
    comp_norm_sq_sum = torch.zeros(n_components, device=device)
    comp_res_cos_sum = torch.zeros(n_components, device=device)
    comp_fwd_cos_sum = torch.zeros(n_components, device=device)
    comp_res_norm_prod_sum = torch.zeros(n_components, device=device)
    comp_res_var_frac_sum = torch.zeros(n_components, device=device)

    residual_norm_sum = torch.tensor(0.0, device=device)
    residual_norm_sq_sum = torch.tensor(0.0, device=device)
    fwd_delta_norm_sum = torch.tensor(0.0, device=device)
    actual_delta_norm_sum = torch.tensor(0.0, device=device)

    overall_cosine_sum = 0.0
    overall_mse_sum = 0.0
    total_positions = 0

    kv_group_size = n_llama_heads // n_kv_heads

    # --- Verification on first batch ---
    print("\n--- Verification (first batch) ---")
    first_ids = torch.from_numpy(tokens[:batch_size]).long().to(device)
    with torch.no_grad():
        outputs = model(first_ids, output_hidden_states=True, use_cache=False)
        src_v = outputs.hidden_states[source_layer + 1].float()
        tgt_v = outputs.hidden_states[target_layer + 1].float()
        pred_v = fwd_model(src_v)
        cos_v = F.cosine_similarity(
            pred_v.reshape(-1, d_model), tgt_v.reshape(-1, d_model), dim=-1
        ).mean()
        res_v = (tgt_v - pred_v).norm(dim=-1).mean()
        print(f"  Forward model cosine vs live Llama: {cos_v:.4f}")
        print(f"  Residual norm: {res_v:.3f}")

        # Verify decomposition: sum(heads) + mlp should equal delta
        attn_pre = captured["attn_pre_proj"]  # (B, T, n_heads*d_head)
        mlp_o = captured["mlp_out"]
        recon_attn = attn_pre.reshape(-1, n_llama_heads * d_head) @ o_proj_weight.T
        recon_attn = recon_attn.reshape(src_v.shape)
        delta_v = tgt_v - src_v
        recon_delta = recon_attn + mlp_o
        recon_err = (delta_v - recon_delta).norm() / delta_v.norm()
        print(f"  Decomposition error (should be ~0): {recon_err:.6f}")
        del outputs, src_v, tgt_v, pred_v, attn_pre, mlp_o, recon_attn, delta_v

    # --- Main eval loop ---
    n_batches = min(n_eval_batches, n_seqs // batch_size)
    print(f"\nRunning {n_batches} eval batches "
          f"({n_batches * batch_size * seq_len / 1e6:.1f}M tokens)...")
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

            # Run Llama — hooks capture attn_pre_proj and mlp_out
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            source_h = outputs.hidden_states[source_layer + 1].float()
            target_h = outputs.hidden_states[target_layer + 1].float()
            del outputs

            B, T, D = source_h.shape
            N = B * T

            # Forward model prediction
            fwd_pred = fwd_model(source_h)

            actual_delta = (target_h - source_h).reshape(-1, D)
            fwd_delta = (fwd_pred - source_h).reshape(-1, D)
            residual = (target_h - fwd_pred).reshape(-1, D)

            r_norms = residual.norm(dim=-1)
            fd_norms = fwd_delta.norm(dim=-1)
            ad_norms = actual_delta.norm(dim=-1)

            residual_norm_sum += r_norms.sum()
            residual_norm_sq_sum += (r_norms ** 2).sum()
            fwd_delta_norm_sum += fd_norms.sum()
            actual_delta_norm_sum += ad_norms.sum()

            r_norm_sq = (r_norms ** 2).unsqueeze(1)  # for variance fraction

            # Per-head decomposition
            attn_pre_proj = captured["attn_pre_proj"].reshape(-1, n_llama_heads * d_head)
            mlp_out = captured["mlp_out"].reshape(-1, D)

            for hi in range(n_llama_heads):
                head_attn = attn_pre_proj[:, hi * d_head:(hi + 1) * d_head]
                head_w = o_proj_weight[:, hi * d_head:(hi + 1) * d_head]
                head_contrib = head_attn @ head_w.T  # (N, D)

                h_norms = head_contrib.norm(dim=-1)
                comp_norm_sum[hi] += h_norms.sum()
                comp_norm_sq_sum[hi] += (h_norms ** 2).sum()
                comp_res_cos_sum[hi] += F.cosine_similarity(
                    head_contrib, residual, dim=-1).sum()
                comp_fwd_cos_sum[hi] += F.cosine_similarity(
                    head_contrib, fwd_delta, dim=-1).sum()
                comp_res_norm_prod_sum[hi] += (h_norms * r_norms).sum()

                # Residual variance in this head's output subspace
                proj = residual @ head_bases[hi]  # (N, d_head)
                proj_norm_sq = (proj ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
                comp_res_var_frac_sum[hi] += (
                    proj_norm_sq / (r_norm_sq + 1e-12)).sum()

            # MLP component
            mlp_norms = mlp_out.norm(dim=-1)
            comp_norm_sum[n_llama_heads] += mlp_norms.sum()
            comp_norm_sq_sum[n_llama_heads] += (mlp_norms ** 2).sum()
            comp_res_cos_sum[n_llama_heads] += F.cosine_similarity(
                mlp_out, residual, dim=-1).sum()
            comp_fwd_cos_sum[n_llama_heads] += F.cosine_similarity(
                mlp_out, fwd_delta, dim=-1).sum()
            comp_res_norm_prod_sum[n_llama_heads] += (mlp_norms * r_norms).sum()

            # Overall quality
            overall_cosine_sum += F.cosine_similarity(
                fwd_pred.reshape(-1, D), target_h.reshape(-1, D), dim=-1
            ).sum().item()
            overall_mse_sum += F.mse_loss(
                fwd_pred, target_h, reduction="sum"
            ).item() / D
            total_positions += N

            del source_h, target_h, fwd_pred, actual_delta, fwd_delta, residual
            del attn_pre_proj, mlp_out
            torch.cuda.empty_cache()

    h_attn.remove()
    h_mlp.remove()

    eval_time = time.time() - t_eval
    N = total_positions
    print(f"  Done in {eval_time:.0f}s ({N:,} positions)")

    # --- Compute final statistics ---
    mean_r_norm = (residual_norm_sum / N).item()
    mean_fd_norm = (fwd_delta_norm_sum / N).item()
    mean_ad_norm = (actual_delta_norm_sum / N).item()
    E_r_sq = (residual_norm_sq_sum / N).item()

    per_component = []
    for ci in range(n_components):
        name = f"head_{ci}" if ci < n_llama_heads else "mlp"
        kv_group = ci // kv_group_size if ci < n_llama_heads else -1

        E_x = (comp_norm_sum[ci] / N).item()
        E_x2 = (comp_norm_sq_sum[ci] / N).item()
        E_xy = (comp_res_norm_prod_sum[ci] / N).item()
        var_x = max(E_x2 - E_x ** 2, 1e-12)
        var_y = max(E_r_sq - mean_r_norm ** 2, 1e-12)
        corr = (E_xy - E_x * mean_r_norm) / (var_x ** 0.5 * var_y ** 0.5)

        entry = {
            "name": name,
            "index": ci,
            "kv_group": kv_group,
            "mean_norm": E_x,
            "cos_with_residual": (comp_res_cos_sum[ci] / N).item(),
            "cos_with_fwd_delta": (comp_fwd_cos_sum[ci] / N).item(),
            "corr_norm_residual": corr,
        }
        if ci < n_llama_heads:
            entry["residual_var_fraction"] = (comp_res_var_frac_sum[ci] / N).item()
        per_component.append(entry)

    # --- Print: ranked by residual alignment ---
    by_residual = sorted(per_component, key=lambda x: x["cos_with_residual"],
                         reverse=True)

    print(f"\n=== Overall ===")
    print(f"  Fwd model cosine: {overall_cosine_sum / N:.4f}")
    print(f"  Fwd model MSE: {overall_mse_sum / N:.6f}")
    print(f"  Mean residual norm: {mean_r_norm:.3f}")
    print(f"  Mean actual delta norm: {mean_ad_norm:.3f}")
    print(f"  Mean fwd delta norm: {mean_fd_norm:.3f}")

    print(f"\n=== All Components (ranked by cos(comp, residual)) ===")
    print(f"{'Component':<10} {'Norm':>7} {'cos(res)':>9} {'cos(fwd)':>9} "
          f"{'corr':>7} {'res_var%':>8} {'KV':>4}")
    print("-" * 58)
    for c in by_residual:
        kv = f"{c['kv_group']}" if c['kv_group'] >= 0 else "—"
        rv = f"{c.get('residual_var_fraction', 0):.4f}" if c['kv_group'] >= 0 else "—"
        print(f"{c['name']:<10} {c['mean_norm']:>7.3f} {c['cos_with_residual']:>9.4f} "
              f"{c['cos_with_fwd_delta']:>9.4f} {c['corr_norm_residual']:>7.3f} "
              f"{rv:>8} {kv:>4}")

    # --- Aggregate by KV group ---
    print(f"\n=== By KV Group (avg of {kv_group_size} Q heads each) ===")
    print(f"{'Group':<12} {'Norm':>7} {'cos(res)':>9} {'cos(fwd)':>9} "
          f"{'corr':>7} {'res_var%':>8}")
    print("-" * 56)
    kv_groups = []
    for kv in range(n_kv_heads):
        heads = [c for c in per_component if c["kv_group"] == kv]
        group = {
            "kv_group": kv,
            "mean_norm": np.mean([c["mean_norm"] for c in heads]),
            "cos_with_residual": np.mean([c["cos_with_residual"] for c in heads]),
            "cos_with_fwd_delta": np.mean([c["cos_with_fwd_delta"] for c in heads]),
            "corr_norm_residual": np.mean([c["corr_norm_residual"] for c in heads]),
            "residual_var_fraction": np.mean(
                [c["residual_var_fraction"] for c in heads]),
        }
        kv_groups.append(group)
        print(f"KV group {kv:<3} {group['mean_norm']:>7.3f} "
              f"{group['cos_with_residual']:>9.4f} "
              f"{group['cos_with_fwd_delta']:>9.4f} "
              f"{group['corr_norm_residual']:>7.3f} "
              f"{group['residual_var_fraction']:>8.4f}")

    mlp_stats = per_component[n_llama_heads]
    print(f"{'MLP':<12} {mlp_stats['mean_norm']:>7.3f} "
          f"{mlp_stats['cos_with_residual']:>9.4f} "
          f"{mlp_stats['cos_with_fwd_delta']:>9.4f} "
          f"{mlp_stats['corr_norm_residual']:>7.3f} {'—':>8}")

    # --- Attention vs MLP ---
    attn_norms = [c["mean_norm"] for c in per_component[:n_llama_heads]]
    total_attn_norm = sum(attn_norms)
    mlp_norm = mlp_stats["mean_norm"]
    attn_cos_res = np.mean([c["cos_with_residual"] for c in per_component[:n_llama_heads]])
    total_res_var = sum(c.get("residual_var_fraction", 0)
                        for c in per_component[:n_llama_heads])

    print(f"\n=== Attention vs MLP ===")
    print(f"  Attn total norm: {total_attn_norm:.3f} "
          f"({total_attn_norm / (total_attn_norm + mlp_norm):.1%} of delta)")
    print(f"  MLP norm: {mlp_norm:.3f} "
          f"({mlp_norm / (total_attn_norm + mlp_norm):.1%} of delta)")
    print(f"  Attn avg cos(residual): {attn_cos_res:.4f}")
    print(f"  MLP cos(residual): {mlp_stats['cos_with_residual']:.4f}")
    print(f"  Total residual variance in attn subspaces: {total_res_var:.4f}")

    # --- Generate figure ---
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kv_colors = plt.cm.Set2(np.linspace(0, 1, n_kv_heads))
    head_colors = [kv_colors[ci // kv_group_size] for ci in range(n_llama_heads)]
    all_colors = list(head_colors) + [(0.3, 0.3, 0.3, 1.0)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Llama 3.2 1B Layer {target_layer} — Per-Head Decomposition\n"
        f"Forward model: {fwd_n_params/1e6:.1f}M params "
        f"(d_head={fwd_d_head}, cosine={overall_cosine_sum/N:.3f})",
        fontsize=13, y=0.995,
    )

    comp_names = [f"H{i}" for i in range(n_llama_heads)] + ["MLP"]
    x_pos = np.arange(n_components)

    # (0,0) Component norms
    ax = axes[0, 0]
    norms = [c["mean_norm"] for c in per_component]
    ax.bar(x_pos, norms, color=all_colors, width=0.8)
    ax.set_xticks(x_pos[::4].tolist() + [n_llama_heads])
    ax.set_xticklabels(
        [comp_names[i] for i in x_pos[::4].tolist()] + ["MLP"], fontsize=7)
    ax.set_ylabel("Mean L2 norm")
    ax.set_title("Component contribution magnitude")

    # (0,1) cos(component, residual)
    ax = axes[0, 1]
    cos_res = [c["cos_with_residual"] for c in per_component]
    ax.bar(x_pos, cos_res, color=all_colors, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos[::4].tolist() + [n_llama_heads])
    ax.set_xticklabels(
        [comp_names[i] for i in x_pos[::4].tolist()] + ["MLP"], fontsize=7)
    ax.set_ylabel("Mean cosine")
    ax.set_title("cos(component, residual) — alignment with what was missed")

    # (1,0) corr(||component||, ||residual||)
    ax = axes[1, 0]
    corrs = [c["corr_norm_residual"] for c in per_component]
    ax.bar(x_pos, corrs, color=all_colors, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos[::4].tolist() + [n_llama_heads])
    ax.set_xticklabels(
        [comp_names[i] for i in x_pos[::4].tolist()] + ["MLP"], fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("corr(||component||, ||residual||) — difficulty correlation")

    # (1,1) Scatter: cos(fwd_delta) vs cos(residual) — the money plot
    ax = axes[1, 1]
    for ci, c in enumerate(per_component):
        marker = "s" if ci == n_llama_heads else "o"
        size = 80 if ci == n_llama_heads else 40
        color = all_colors[ci]
        ax.scatter(c["cos_with_residual"], c["cos_with_fwd_delta"],
                   c=[color], s=size, marker=marker, edgecolors="black",
                   linewidths=0.5, zorder=3)
        if ci == n_llama_heads or abs(c["cos_with_residual"]) > 0.03:
            ax.annotate(comp_names[ci], (c["cos_with_residual"],
                        c["cos_with_fwd_delta"]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("cos(component, residual) — missed")
    ax.set_ylabel("cos(component, fwd_delta) — captured")
    ax.set_title("Capture vs miss profile per component")

    # KV group legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=kv_colors[kv], label=f"KV {kv}")
        for kv in range(n_kv_heads)
    ] + [Patch(facecolor=(0.3, 0.3, 0.3, 1.0), label="MLP")]
    axes[0, 0].legend(handles=legend_elements, fontsize=6, ncol=3,
                       loc="upper right")

    plt.tight_layout()

    save_dir = f"{DATA_DIR}/a2a_llama/analysis"
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_swiglu" if fwd_use_swiglu else ""
    fig_path = os.path.join(save_dir, f"head_decomposition{suffix}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")

    # --- Save results ---
    results = {
        "overall": {
            "cosine": overall_cosine_sum / N,
            "mse": overall_mse_sum / N,
            "mean_residual_norm": mean_r_norm,
            "mean_actual_delta_norm": mean_ad_norm,
            "mean_fwd_delta_norm": mean_fd_norm,
            "total_positions": N,
        },
        "per_component": per_component,
        "kv_groups": kv_groups,
        "attention_vs_mlp": {
            "total_attn_norm": total_attn_norm,
            "mlp_norm": mlp_norm,
            "attn_fraction_of_delta": total_attn_norm / (total_attn_norm + mlp_norm),
            "attn_avg_cos_residual": attn_cos_res,
            "mlp_cos_residual": mlp_stats["cos_with_residual"],
            "total_residual_var_in_attn_subspaces": total_res_var,
        },
        "config": {
            "model_name": model_name,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "d_model": d_model,
            "n_llama_heads": n_llama_heads,
            "n_kv_heads": n_kv_heads,
            "d_head": d_head,
            "n_model_params": int(n_params),
            "fwd_n_params": int(fwd_n_params),
            "fwd_d_head": fwd_d_head,
            "fwd_use_swiglu": fwd_use_swiglu,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_eval_batches": n_batches,
            "eval_seconds": eval_time,
        },
    }

    results_path = os.path.join(save_dir, f"head_decomposition{suffix}.json")
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
    n_eval_batches: int = 80,
    fwd_n_layer: int = 1,
    fwd_n_head: int = 1,
    fwd_d_head: int = 128,
    fwd_mlp_mult: float = 2,
    fwd_use_swiglu: bool = False,
):
    result = a2a_llama_head_decomposition.remote(
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
    print(f"\nLlama per-head decomposition results:")
    print(f"  Fwd model cosine: {ov['cosine']:.4f}")
    print(f"  Mean residual norm: {ov['mean_residual_norm']:.3f}")

    avm = result["attention_vs_mlp"]
    print(f"\n  Attention vs MLP:")
    print(f"    Attn norm: {avm['total_attn_norm']:.3f} "
          f"({avm['attn_fraction_of_delta']:.1%} of delta)")
    print(f"    MLP norm: {avm['mlp_norm']:.3f}")
    print(f"    Attn avg cos(res): {avm['attn_avg_cos_residual']:.4f}")
    print(f"    MLP cos(res): {avm['mlp_cos_residual']:.4f}")

    print(f"\n  Top 5 heads by cos(residual):")
    by_res = sorted(result["per_component"],
                    key=lambda x: x["cos_with_residual"], reverse=True)
    for c in by_res[:5]:
        print(f"    {c['name']}: cos(res)={c['cos_with_residual']:.4f}, "
              f"cos(fwd)={c['cos_with_fwd_delta']:.4f}, "
              f"norm={c['mean_norm']:.3f}")
