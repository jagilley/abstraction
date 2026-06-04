"""Mirror test for neural self-knowledge.

Analog of the biological mirror test (Gallup, 1970) for the A2A cerebellar
architecture. Tests whether the closed-loop model uses its self-knowledge to
produce direction-specific responses to perturbations of its own computation.

Core design:
- The "mark": perturbation of post_block1 activations (arbitrary direction)
- The "mirror": forward model prediction (injected in CL+M condition)
- "Reaching for the mark": downstream response at post_block3 channeled
  through the self-knowledge subspace (top divergence PCs between models)

Three model conditions:
1. CL+M: closed-loop model with injection (mirror + self-knowledge)
2. CL-M: closed-loop model without injection (self-knowledge weights only)
3. OL:   open-loop model (neither mirror nor self-knowledge)

Key prediction: for ARBITRARY perturbations at post_block1, the CL model's
response at post_block3 is more aligned with the self-knowledge subspace
than the OL model's response. The CL model "reaches for the mark" —
its response is structured by self-knowledge — while the OL model shows
generic, isotropic degradation.

The interaction test: the CL model should show HIGHER SK-fraction for
perturbations than the OL model. This parallels the mirror test's core
logic: the animal with self-recognition reaches for the mark on its own
face, while the animal without self-recognition displays generic behavior.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mirror_test(
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
    n_perturb_dirs: int = 16,
    sk_rank: int = 10,
    ckpt_source: str = "controlled",
    ckpt_step: int = 0,
):
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
    print(f"MIRROR TEST on {device}")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

    # =========================================================
    # Load data
    # =========================================================
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
    print(f"Loaded {len(data):,} tokens, {len(val_data):,} for eval")

    # =========================================================
    # Load models
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    if ckpt_source == "controlled":
        ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                     f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
        open_dir = os.path.join(ckpt_root, "open_loop")
        closed_dir = os.path.join(ckpt_root, "closed_loop")
    elif ckpt_source == "extended":
        fwd_tag = (f"fwd{fwd_n_layer}L{fwd_n_head}H{fwd_d_head}d"
                   f"_mlp{fwd_mlp_mult}")
        ckpt_root = (f"{DATA_DIR}/a2a_forward/extended/"
                     f"{gap_tag}/inject{inject_after_block}/"
                     f"{fwd_tag}/P_{n_tokens}")
        step_dir = f"step_{ckpt_step:06d}"
        open_dir = os.path.join(ckpt_root, "open_loop", step_dir)
        closed_dir = os.path.join(ckpt_root, "closed_loop", step_dir)
    else:
        raise ValueError(f"Unknown ckpt_source: {ckpt_source}")
    print(f"Loading checkpoints from: {ckpt_root}")
    print(f"  open:   {open_dir}")
    print(f"  closed: {closed_dir}")

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
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    # =========================================================
    # Generate eval batches
    # =========================================================
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

    def per_tok_loss(logits, tgt):
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    # =========================================================
    # PHASE 1: Collect activations to compute divergence PCs
    #          (the self-knowledge subspace)
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Computing self-knowledge subspace")
    print(f"{'='*60}")

    acts_open_b3, acts_closed_b3 = [], []
    acts_open_b1, acts_closed_b1 = [], []

    with torch.no_grad():
        for bi, idx in enumerate(eval_indices):
            x, y = make_batch(idx)
            _, _, vi_open = model_open(x, y, return_intermediates=True)
            _, _, vi_closed = model_closed(x, y, return_intermediates=True)

            acts_open_b3.append(
                vi_open["post_block3"].reshape(-1, n_embd).cpu())
            acts_closed_b3.append(
                vi_closed["post_block3"].reshape(-1, n_embd).cpu())
            acts_open_b1.append(
                vi_open["post_block1"].reshape(-1, n_embd).cpu())
            acts_closed_b1.append(
                vi_closed["post_block1"].reshape(-1, n_embd).cpu())

            if bi % 10 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    acts_open_b3 = torch.cat(acts_open_b3).numpy()
    acts_closed_b3 = torch.cat(acts_closed_b3).numpy()
    acts_open_b1 = torch.cat(acts_open_b1).numpy()
    acts_closed_b1 = torch.cat(acts_closed_b1).numpy()
    n_samples = acts_open_b3.shape[0]
    print(f"Collected {n_samples:,} positions")

    # Divergence PCs at post_block3 = self-knowledge subspace
    diff_b3 = acts_closed_b3 - acts_open_b3
    diff_b3_centered = diff_b3 - diff_b3.mean(axis=0, keepdims=True)
    _, S_b3, Vt_b3 = np.linalg.svd(diff_b3_centered, full_matrices=False)
    sk_subspace = Vt_b3[:sk_rank]  # (sk_rank, 256) — rows are SK PCs

    total_var = (S_b3 ** 2).sum()
    sk_var_frac = (S_b3[:sk_rank] ** 2).sum() / total_var
    print(f"SK subspace (top-{sk_rank} divergence PCs at post_block3): "
          f"{sk_var_frac:.3f} of divergence variance")

    # Divergence PCs at post_block0 = ORTHOGONAL to self-knowledge
    # (extra-SK overlap 0.49x, below random — from rep divergence analysis)
    # We use post_block1 divergence PCs for comparison
    diff_b1 = acts_closed_b1 - acts_open_b1
    diff_b1_centered = diff_b1 - diff_b1.mean(axis=0, keepdims=True)
    _, S_b1, Vt_b1 = np.linalg.svd(diff_b1_centered, full_matrices=False)
    b1_div_pcs = Vt_b1[:sk_rank]  # (sk_rank, 256)

    # Effective rank of SK subspace for reference
    p = (S_b3 ** 2) / total_var
    p = p[p > 1e-12]
    eff_rank_b3 = float(np.exp(-np.sum(p * np.log(p))))
    print(f"Effective rank of post_block3 divergence: {eff_rank_b3:.1f}")

    del acts_open_b3, acts_closed_b3, acts_open_b1, acts_closed_b1
    del diff_b3, diff_b3_centered, diff_b1, diff_b1_centered

    # =========================================================
    # PHASE 2: Define perturbation directions
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Perturbation directions")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed + 300)
    perturb_dirs = {}

    # Random directions (the "rouge mark" — arbitrary)
    for i in range(n_perturb_dirs):
        d = rng.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs[f"random_{i}"] = d

    # Post_block1 divergence PCs (partially SK-aligned)
    for i in range(min(3, sk_rank)):
        perturb_dirs[f"b1_div_pc{i}"] = b1_div_pcs[i].astype(np.float32)

    print(f"  {len(perturb_dirs)} perturbation directions "
          f"({n_perturb_dirs} random + {min(3, sk_rank)} b1 divergence PCs)")

    # =========================================================
    # PHASE 3: Perturbation sweep
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Perturbation sweep")
    print(f"{'='*60}")

    # Compute perturbation scale: std of post_block1 activations
    with torch.no_grad():
        x_ref, _ = make_batch(eval_indices[0])
        _, _, vi_ref = model_closed(x_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"  post_block1 activation std: {b1_std:.4f}")

    s_grid = [0.0, 0.5, 1.0, 2.0, 3.0]

    sk_basis = torch.from_numpy(sk_subspace.T).float().to(device)  # (256, sk_rank)

    conditions = {
        "CL+M": ("closed", True),
        "CL-M": ("closed", False),
        "OL":   ("open", False),
    }

    results = {cond: {} for cond in conditions}

    for cond_name, (model_key, use_injection) in conditions.items():
        model = model_closed if model_key == "closed" else model_open
        print(f"\n  [{cond_name}] Precomputing baselines...")

        # --- Precompute baseline for this condition (once) ---
        baseline_b3 = []  # per-batch post_block3, stored on CPU
        baseline_loss_per_batch = []
        with torch.no_grad():
            for bi, idx in enumerate(eval_indices):
                x, y = make_batch(idx)
                tgt = x[:, 1:]
                if use_injection:
                    logits, _, inter = model(
                        x, y, return_intermediates=True,
                        cerebellar_fn=lambda act: gate(fwd_closed(act)),
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                else:
                    logits, _, inter = model(
                        x, y, return_intermediates=True,
                    )
                baseline_b3.append(inter["post_block3"].cpu())
                baseline_loss_per_batch.append(
                    per_tok_loss(logits, tgt).sum().item())

        n_tok_per_batch = eval_indices[0].shape[0] * (block_size - 1)
        total_tok = n_tok_per_batch * n_eval_batches
        baseline_loss = sum(baseline_loss_per_batch) / total_tok

        # Store s=0 results for all directions (they're all the same)
        for dir_name in perturb_dirs:
            results[cond_name][dir_name] = {
                0.0: {
                    "mean_loss": baseline_loss,
                    "mean_sk_frac": 0.0,
                    "mean_response_norm": 0.0,
                    "n_tokens": total_tok,
                }
            }

        # --- Perturbation sweep (only perturbed passes) ---
        n_dirs_done = 0
        for dir_name, dir_vec in perturb_dirs.items():
            dir_t = torch.from_numpy(dir_vec).to(device)

            for s in s_grid:
                if s == 0.0:
                    continue
                perturbation = (s * b1_std) * dir_t

                loss_sum = 0.0
                sk_frac_sum = 0.0
                response_norm_sum = 0.0
                n_tok = 0

                with torch.no_grad():
                    for bi, idx in enumerate(eval_indices):
                        x, y = make_batch(idx)
                        tgt = x[:, 1:]

                        if use_injection:
                            def _cb_fn(act, _p=perturbation):
                                return gate(fwd_closed(act)) + _p
                            logits_pert, _, inter_pert = model(
                                x, y, return_intermediates=True,
                                cerebellar_fn=_cb_fn,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                        else:
                            def _cb_fn_noinj(act, _p=perturbation):
                                return _p.unsqueeze(0).unsqueeze(0).expand(
                                    act.shape[0], act.shape[1], -1)
                            logits_pert, _, inter_pert = model(
                                x, y, return_intermediates=True,
                                cerebellar_fn=_cb_fn_noinj,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                        pert_b3 = inter_pert["post_block3"]

                        loss_tok = per_tok_loss(logits_pert, tgt)
                        loss_sum += loss_tok.sum().item()

                        base_b3 = baseline_b3[bi].to(device)
                        response = (pert_b3 - base_b3)[:, :-1]
                        resp_flat = response.reshape(-1, n_embd)

                        resp_norm_sq = (resp_flat ** 2).sum(dim=-1)
                        resp_in_sk = resp_flat @ sk_basis
                        sk_norm_sq = (resp_in_sk ** 2).sum(dim=-1)

                        valid = resp_norm_sq > 1e-10
                        if valid.sum() > 0:
                            sk_frac_sum += float(
                                (sk_norm_sq[valid] / resp_norm_sq[valid]).sum())
                            response_norm_sum += float(
                                resp_norm_sq[valid].sqrt().sum())
                            n_tok += int(valid.sum())

                results[cond_name][dir_name][s] = {
                    "mean_loss": loss_sum / max(n_tok, 1),
                    "mean_sk_frac": sk_frac_sum / max(n_tok, 1),
                    "mean_response_norm": response_norm_sum / max(n_tok, 1),
                    "n_tokens": n_tok,
                }

            n_dirs_done += 1
            if n_dirs_done % 5 == 0:
                print(f"    {n_dirs_done}/{len(perturb_dirs)} directions done")

        print(f"  [{cond_name}] done")

    # =========================================================
    # PHASE 4: Analysis — Mirror test metrics
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Mirror test analysis")
    print(f"{'='*60}")

    random_baseline_sk_frac = sk_rank / n_embd
    print(f"\nRandom baseline SK fraction: {random_baseline_sk_frac:.4f} "
          f"({sk_rank}/{n_embd})")

    # 4a: SK fraction by condition and perturbation strength
    print(f"\n--- SK fraction of response (higher = more self-referential) ---")
    print(f"{'Condition':>8s} {'s':>5s} {'SK frac':>10s} {'ratio':>8s} "
          f"{'resp norm':>10s} {'Δloss':>10s}")
    print("-" * 60)

    baseline_losses = {}
    for cond_name in conditions:
        # Aggregate over random directions for the main signal
        random_dirs = [d for d in perturb_dirs if d.startswith("random_")]

        # Get baseline loss (s=0)
        losses_s0 = [results[cond_name][d][0.0]["mean_loss"]
                      for d in random_dirs]
        base_loss = np.mean(losses_s0)
        baseline_losses[cond_name] = base_loss

        for s in s_grid:
            if s == 0.0:
                continue
            sk_fracs = [results[cond_name][d][s]["mean_sk_frac"]
                        for d in random_dirs]
            resp_norms = [results[cond_name][d][s]["mean_response_norm"]
                          for d in random_dirs]
            losses = [results[cond_name][d][s]["mean_loss"]
                      for d in random_dirs]

            mean_sk = np.mean(sk_fracs)
            mean_rn = np.mean(resp_norms)
            mean_loss = np.mean(losses)
            delta_loss = mean_loss - base_loss
            ratio = mean_sk / random_baseline_sk_frac

            print(f"{cond_name:>8s} {s:>5.1f} {mean_sk:>10.4f} "
                  f"{ratio:>7.2f}x {mean_rn:>10.4f} {delta_loss:>+10.4f}")

    # 4b: Summary — the mirror test verdict
    print(f"\n{'='*60}")
    print("MIRROR TEST VERDICT")
    print(f"{'='*60}")

    # Average SK fraction at moderate perturbation (s=2.0) for random dirs
    random_dirs = [d for d in perturb_dirs if d.startswith("random_")]
    verdicts = {}
    for cond_name in conditions:
        s_test = 2.0
        sk_fracs = [results[cond_name][d][s_test]["mean_sk_frac"]
                    for d in random_dirs]
        verdicts[cond_name] = {
            "mean_sk_frac": float(np.mean(sk_fracs)),
            "std_sk_frac": float(np.std(sk_fracs)),
            "ratio_vs_random": float(np.mean(sk_fracs) / random_baseline_sk_frac),
        }

    for cond_name, v in verdicts.items():
        print(f"  {cond_name:>6s}: SK frac = {v['mean_sk_frac']:.4f} "
              f"± {v['std_sk_frac']:.4f} "
              f"({v['ratio_vs_random']:.2f}x random baseline)")

    # Test: is CL+M > OL?
    cl_m_sk = verdicts["CL+M"]["mean_sk_frac"]
    ol_sk = verdicts["OL"]["mean_sk_frac"]
    cl_no_m_sk = verdicts["CL-M"]["mean_sk_frac"]
    print(f"\n  CL+M vs OL:  {cl_m_sk:.4f} vs {ol_sk:.4f} "
          f"(Δ = {cl_m_sk - ol_sk:+.4f})")
    print(f"  CL-M vs OL:  {cl_no_m_sk:.4f} vs {ol_sk:.4f} "
          f"(Δ = {cl_no_m_sk - ol_sk:+.4f})")
    print(f"  CL+M vs CL-M: {cl_m_sk:.4f} vs {cl_no_m_sk:.4f} "
          f"(Δ = {cl_m_sk - cl_no_m_sk:+.4f}, mirror effect)")

    passes_mirror_test = cl_m_sk > ol_sk and cl_m_sk > random_baseline_sk_frac * 1.2
    print(f"\n  PASSES MIRROR TEST: {passes_mirror_test}")
    if passes_mirror_test:
        print("  → The closed-loop model channels perturbation responses "
              "through its self-knowledge subspace.")
        print("  → It 'reaches for the mark' — responding to arbitrary "
              "perturbations in a self-referential way.")

    # 4c: Per-direction breakdown (random vs b1 divergence PCs)
    print(f"\n--- Per-direction type breakdown (s=2.0) ---")
    dir_types = {
        "random": [d for d in perturb_dirs if d.startswith("random_")],
        "b1_div": [d for d in perturb_dirs if d.startswith("b1_div")],
    }
    for dtype, dirs in dir_types.items():
        if not dirs:
            continue
        print(f"\n  Perturbation type: {dtype} ({len(dirs)} directions)")
        for cond_name in conditions:
            sk_fracs = [results[cond_name][d][2.0]["mean_sk_frac"]
                        for d in dirs]
            delta_losses = [
                results[cond_name][d][2.0]["mean_loss"]
                - baseline_losses[cond_name]
                for d in dirs]
            print(f"    {cond_name:>6s}: SK frac = {np.mean(sk_fracs):.4f} "
                  f"± {np.std(sk_fracs):.4f}, "
                  f"Δloss = {np.mean(delta_losses):+.4f}")

    # 4d: Robustness comparison — does SK model degrade less?
    print(f"\n--- Loss degradation by perturbation strength ---")
    for cond_name in conditions:
        print(f"  {cond_name}: baseline loss = {baseline_losses[cond_name]:.4f}")
        for s in [1.0, 2.0, 3.0]:
            losses = [results[cond_name][d][s]["mean_loss"]
                      for d in random_dirs]
            delta = np.mean(losses) - baseline_losses[cond_name]
            print(f"    s={s:.1f}: Δloss = {delta:+.4f}")

    # =========================================================
    # Save results
    # =========================================================
    if ckpt_source == "extended":
        save_dir = os.path.join(ckpt_root, f"mirror_test_step{ckpt_step}")
    else:
        save_dir = os.path.join(ckpt_root, "mirror_test")
    os.makedirs(save_dir, exist_ok=True)

    # Convert results to serializable format
    serializable_results = {}
    for cond_name, cond_data in results.items():
        serializable_results[cond_name] = {}
        for dir_name, dir_data in cond_data.items():
            serializable_results[cond_name][dir_name] = {
                str(s): v for s, v in dir_data.items()
            }

    output = {
        "config": {
            "n_tokens": n_tokens, "n_eval_batches": n_eval_batches,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "seed": seed,
            "n_perturb_dirs": n_perturb_dirs, "sk_rank": sk_rank,
            "s_grid": s_grid, "b1_activation_std": b1_std,
        },
        "sk_subspace": {
            "rank": sk_rank,
            "variance_fraction": float(sk_var_frac),
            "effective_rank_b3_divergence": float(eff_rank_b3),
            "random_baseline_fraction": float(random_baseline_sk_frac),
        },
        "verdicts": verdicts,
        "baseline_losses": {k: float(v) for k, v in baseline_losses.items()},
        "passes_mirror_test": passes_mirror_test,
        "full_results": serializable_results,
    }

    from a2a_forward.shared import NumpyEncoder
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {save_dir}")
    return output


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_steps: int = 10_000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    ckpt_source = "controlled"
    ckpt_step = 0
    if fwd_n_layer > 2 or fwd_d_head > 64 or fwd_n_head > 1 or fwd_mlp_mult > 2:
        ckpt_source = "extended"
        ckpt_step = n_steps
    result = a2a_mirror_test.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
        ckpt_source=ckpt_source, ckpt_step=ckpt_step,
    )
    print("A2A mirror test complete:")
    v = result["verdicts"]
    for cond, d in v.items():
        print(f"  {cond:>6s}: SK frac = {d['mean_sk_frac']:.4f} "
              f"({d['ratio_vs_random']:.2f}x random)")
    print(f"  PASSES: {result['passes_mirror_test']}")
