"""Geometry control for the mirror test.

Computes the SK fraction of each model's GENERAL activation variance at
post_block3 (no perturbation). If the CL model's general activations have
the same SK fraction as the perturbation response (~1.6x), the mirror test
result is explained by activation geometry. If the general SK fraction is
lower, the perturbation response is specifically self-knowledge-channeled.
"""

import json

from language_reduction.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_mirror_test_geometry_control(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    n_eval_batches: int = 40,
    sk_rank: int = 10,
    ckpt_source: str = "controlled",
    ckpt_step: int = 0,
):
    import os
    import glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.experiments.a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MIRROR TEST GEOMETRY CONTROL on {device}")
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
    data = torch.from_numpy(
        np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]

    # --- Load models ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    if ckpt_source == "controlled":
        ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                     f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
        open_dir = os.path.join(ckpt_root, "open_loop")
        closed_dir = os.path.join(ckpt_root, "closed_loop")
    else:
        fwd_tag = (f"fwd{fwd_n_layer}L{fwd_n_head}H{fwd_d_head}d"
                   f"_mlp{fwd_mlp_mult}")
        ckpt_root = (f"{DATA_DIR}/a2a_forward/extended/"
                     f"{gap_tag}/inject{inject_after_block}/"
                     f"{fwd_tag}/P_{n_tokens}")
        step_dir = f"step_{ckpt_step:06d}"
        open_dir = os.path.join(ckpt_root, "open_loop", step_dir)
        closed_dir = os.path.join(ckpt_root, "closed_loop", step_dir)

    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(torch.load(
        os.path.join(open_dir, "model.pt"),
        map_location=device, weights_only=True))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "model.pt"),
        map_location=device, weights_only=True))

    fwd_closed = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "fwd_model.pt"),
        map_location=device, weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(closed_dir, "gate.pt"),
        map_location=device, weights_only=True))

    model_open.eval()
    model_closed.eval()
    fwd_closed.eval()
    gate.eval()

    # --- Eval batches ---
    eval_gen = torch.Generator().manual_seed(seed + 200)
    eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(n_eval_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # =========================================================
    # Collect post_block3 activations for all 3 conditions
    # =========================================================
    print("Collecting post_block3 activations...")

    conditions = {
        "CL+M": ("closed", True),
        "CL-M": ("closed", False),
        "OL":   ("open", False),
    }

    all_acts = {cond: [] for cond in conditions}

    with torch.no_grad():
        for bi, idx in enumerate(eval_indices):
            x, y = make_batch(idx)

            # OL
            _, _, vi_ol = model_open(x, y, return_intermediates=True)
            all_acts["OL"].append(
                vi_ol["post_block3"][:, :-1].reshape(-1, n_embd).cpu())

            # CL-M (no injection)
            _, _, vi_cl_nm = model_closed(x, y, return_intermediates=True)
            all_acts["CL-M"].append(
                vi_cl_nm["post_block3"][:, :-1].reshape(-1, n_embd).cpu())

            # CL+M (with injection)
            _, _, vi_cl_m = model_closed(
                x, y, return_intermediates=True,
                cerebellar_fn=lambda act: gate(fwd_closed(act)),
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )
            all_acts["CL+M"].append(
                vi_cl_m["post_block3"][:, :-1].reshape(-1, n_embd).cpu())

            if bi % 10 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    all_acts = {k: torch.cat(v).numpy() for k, v in all_acts.items()}
    n_samples = all_acts["OL"].shape[0]
    print(f"Collected {n_samples:,} positions per condition")

    # =========================================================
    # Compute divergence PCs (SK subspace) — same as mirror test
    # =========================================================
    # Use CL-M vs OL (no injection in either) for clean comparison
    diff = all_acts["CL-M"] - all_acts["OL"]
    diff_centered = diff - diff.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)
    sk_subspace = Vt[:sk_rank]  # (sk_rank, 256)

    total_var = (S ** 2).sum()
    sk_var_frac = (S[:sk_rank] ** 2).sum() / total_var
    print(f"\nSK subspace (top-{sk_rank} div PCs): "
          f"{sk_var_frac:.3f} of divergence variance")

    random_baseline = sk_rank / n_embd
    print(f"Random baseline: {random_baseline:.4f}")

    # =========================================================
    # Compute SK fraction of GENERAL activation variance
    # =========================================================
    print(f"\n{'='*60}")
    print("GEOMETRY CONTROL: SK fraction of general activation variance")
    print(f"{'='*60}")

    results = {}
    for cond in conditions:
        acts = all_acts[cond]
        acts_centered = acts - acts.mean(axis=0, keepdims=True)

        # Total variance
        total_act_var = float((acts_centered ** 2).sum())

        # Variance in SK subspace
        proj = acts_centered @ sk_subspace.T  # (n, sk_rank)
        sk_act_var = float((proj ** 2).sum())

        sk_frac = sk_act_var / total_act_var
        ratio = sk_frac / random_baseline

        results[cond] = {
            "sk_frac_of_variance": sk_frac,
            "ratio_vs_random": ratio,
        }
        print(f"  {cond:>6s}: SK frac of variance = {sk_frac:.4f} "
              f"({ratio:.2f}x random)")

    # =========================================================
    # Per-position SK fraction (matching the perturbation metric)
    # =========================================================
    print(f"\n{'='*60}")
    print("Per-position SK fraction (same metric as mirror test)")
    print(f"{'='*60}")

    # The mirror test computes: for each position, what fraction of that
    # position's activation falls in the SK subspace. This is a per-position
    # metric, then averaged. Different from the variance-based metric above
    # because it weights all positions equally regardless of activation norm.

    sk_basis_t = torch.from_numpy(sk_subspace.T).float()  # (256, sk_rank)

    for cond in conditions:
        acts_t = torch.from_numpy(all_acts[cond]).float()
        acts_centered_t = acts_t - acts_t.mean(dim=0, keepdim=True)

        norm_sq = (acts_centered_t ** 2).sum(dim=-1)  # (n,)
        proj = acts_centered_t @ sk_basis_t  # (n, sk_rank)
        sk_norm_sq = (proj ** 2).sum(dim=-1)  # (n,)

        valid = norm_sq > 1e-10
        per_pos_sk_frac = float(
            (sk_norm_sq[valid] / norm_sq[valid]).mean())
        ratio = per_pos_sk_frac / random_baseline

        results[cond]["per_position_sk_frac"] = per_pos_sk_frac
        results[cond]["per_position_ratio"] = ratio
        print(f"  {cond:>6s}: per-position SK frac = {per_pos_sk_frac:.4f} "
              f"({ratio:.2f}x random)")

    # =========================================================
    # Compare with mirror test perturbation results
    # =========================================================
    print(f"\n{'='*60}")
    print("COMPARISON: general geometry vs perturbation response")
    print(f"{'='*60}")

    print(f"\n  {'Condition':>8s} {'General SK':>12s} {'Perturb SK':>12s} "
          f"{'Difference':>12s}")
    print(f"  {'-'*48}")

    # Load mirror test results if available
    if ckpt_source == "controlled":
        mt_path = os.path.join(ckpt_root, "mirror_test", "results.json")
    else:
        mt_path = os.path.join(
            ckpt_root, f"mirror_test_step{ckpt_step}", "results.json")

    mt_loaded = False
    if os.path.exists(mt_path):
        with open(mt_path) as f:
            mt = json.load(f)
        mt_loaded = True
        for cond in conditions:
            gen_sk = results[cond]["per_position_sk_frac"]
            pert_sk = mt["verdicts"][cond]["mean_sk_frac"]
            diff_val = pert_sk - gen_sk
            print(f"  {cond:>8s} {gen_sk:>12.4f} {pert_sk:>12.4f} "
                  f"{diff_val:>+12.4f}")

        print(f"\n  If 'Difference' is positive, the perturbation response "
              f"is MORE SK-aligned\n  than general activation variance — "
              f"evidence of self-referential response\n  beyond mere geometry.")
        print(f"  If 'Difference' is near zero or negative, the perturbation "
              f"result\n  is explained by activation geometry.")
    else:
        print(f"  (Mirror test results not found at {mt_path})")

    # =========================================================
    # Save
    # =========================================================
    if ckpt_source == "extended":
        save_dir = os.path.join(
            ckpt_root, f"mirror_test_step{ckpt_step}")
    else:
        save_dir = os.path.join(ckpt_root, "mirror_test")
    os.makedirs(save_dir, exist_ok=True)

    output = {
        "config": {
            "ckpt_source": ckpt_source, "ckpt_step": ckpt_step,
            "sk_rank": sk_rank, "n_eval_batches": n_eval_batches,
            "n_samples": n_samples,
        },
        "random_baseline": random_baseline,
        "results": results,
    }

    if mt_loaded:
        output["comparison"] = {
            cond: {
                "general_sk_frac": results[cond]["per_position_sk_frac"],
                "perturbation_sk_frac": mt["verdicts"][cond]["mean_sk_frac"],
                "difference": (mt["verdicts"][cond]["mean_sk_frac"]
                               - results[cond]["per_position_sk_frac"]),
            }
            for cond in conditions
        }

    from language_reduction.shared import NumpyEncoder
    save_path = os.path.join(save_dir, "geometry_control.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_path}")
    return output
