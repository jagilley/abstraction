"""Mirror test v2: compensatory response and perturbation discrimination.

Two tests that avoid the subspace cherry-picking problem of the original
mirror test (mirror_test.py), inspired by Gallup (1970) and Vogel (2025).

Test 1 — Compensatory response ("reaching for the mark"):
  Apply perturbation δ to post_block1. Measure whether the model's
  downstream response at post_block3 *opposes* the passive propagation
  of δ — i.e., whether the model actively corrects the perturbation.
  If CL+M compensates more than CL-M > OL, the mirror is being used
  for real-time self-correction.

Test 2 — Perturbation discrimination at the logit level:
  Apply two different perturbations (δ_A, δ_B) and measure whether the
  model's output distribution shift carries information about *which*
  perturbation was applied. If the CL model's logit shifts are more
  perturbation-specific than the OL model's, the self-knowledge is
  reaching the model's output in a structured way.

Neither test depends on externally-defined subspaces (no divergence PCs).
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mirror_test_v2(
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
    print(f"MIRROR TEST v2 on {device}")
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
    print(f"Loading from: {ckpt_root}")

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
    # Eval batches
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

    # =========================================================
    # Perturbation directions
    # =========================================================
    rng = np.random.default_rng(seed + 300)
    perturb_dirs = []
    for i in range(n_perturb_dirs):
        d = rng.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs.append(d)

    # Compute perturbation scale
    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0])
        _, _, vi_ref = model_closed(x_ref, y_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"post_block1 activation std: {b1_std:.4f}")

    s_values = [0.5, 1.0, 2.0, 3.0]

    conditions = {
        "CL+M": ("closed", True),
        "CL-M": ("closed", False),
        "OL":   ("open", False),
    }

    # =========================================================
    # Helper: run model with optional perturbation at post_block1
    # =========================================================
    def run_model(model, x, y, use_injection, perturbation=None):
        """Run model, optionally adding perturbation at post_block1.

        The perturbation is added to the residual stream AFTER block1,
        at the same point the cerebellar injection goes. The forward
        model's prediction is computed from post_block0 (upstream), so
        it does NOT see the perturbation — the "mirror" reveals the
        "mark" via the prediction-actual discrepancy.
        """
        if use_injection and perturbation is not None:
            def cb_fn(act, _p=perturbation):
                return gate(fwd_closed(act)) + _p
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        elif use_injection:
            def cb_fn(act):
                return gate(fwd_closed(act))
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        elif perturbation is not None:
            def cb_fn(act, _p=perturbation):
                return _p.unsqueeze(0).unsqueeze(0).expand(
                    act.shape[0], act.shape[1], -1)
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        else:
            return model(x, y, return_intermediates=True)

    # =========================================================
    # TEST 1: Compensatory response
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST 1: Compensatory response")
    print(f"{'='*60}")
    print("Does the model's response OPPOSE the perturbation?")
    print("Compensation ratio < 1 means the model dampens the perturbation.")
    print("Compensation ratio > 1 means the perturbation is amplified.\n")

    # For each condition:
    #   1. Get baseline post_block3 (no perturbation)
    #   2. Get perturbed post_block3
    #   3. Response R = perturbed_b3 - baseline_b3
    #   4. Also get "passive propagation" by running the perturbation through
    #      blocks 2-3 with the OL model (no self-knowledge) — this estimates
    #      what R would look like if the model just let δ flow through.
    #   5. Compensation = ||R|| / ||R_passive|| — if < 1, the model is damping.
    #
    # Simpler version that avoids needing a Jacobian estimate:
    #   Just measure ||R|| directly and compare across conditions.
    #   If CL models have smaller ||R|| for the same perturbation, they're
    #   compensating (this is the robustness gap, but measured geometrically).
    #
    # Even better: measure the cosine between R and the perturbation direction δ.
    #   If the model propagates passively, R should be correlated with δ.
    #   If it compensates, R should be LESS correlated or even anti-correlated.

    comp_results = {cond: {s: [] for s in s_values} for cond in conditions}

    for cond_name, (model_key, use_injection) in conditions.items():
        model = model_closed if model_key == "closed" else model_open
        print(f"\n  [{cond_name}]")

        for s in s_values:
            response_norms = []
            cosine_with_delta = []
            loss_deltas = []

            for di in range(n_perturb_dirs):
                dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
                perturbation = (s * b1_std) * dir_t

                rn_batch, cos_batch, ld_batch = [], [], []

                with torch.no_grad():
                    for bi, idx in enumerate(eval_indices):
                        x, y = make_batch(idx)
                        tgt = x[:, 1:]

                        # Baseline
                        logits_base, _, inter_base = run_model(
                            model, x, y, use_injection)
                        b3_base = inter_base["post_block3"][:, :-1]
                        loss_base = -F.log_softmax(logits_base[:, :-1], dim=-1
                            ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                        # Perturbed
                        logits_pert, _, inter_pert = run_model(
                            model, x, y, use_injection, perturbation)
                        b3_pert = inter_pert["post_block3"][:, :-1]
                        loss_pert = -F.log_softmax(logits_pert[:, :-1], dim=-1
                            ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                        # Response
                        response = (b3_pert - b3_base).reshape(-1, n_embd)
                        r_norm = response.norm(dim=-1)

                        # Cosine with perturbation direction
                        cos = F.cosine_similarity(
                            response, dir_t.unsqueeze(0).expand_as(response),
                            dim=-1)

                        valid = r_norm > 1e-10
                        if valid.sum() > 0:
                            rn_batch.append(r_norm[valid].mean().item())
                            cos_batch.append(cos[valid].mean().item())
                            ld_batch.append(
                                (loss_pert - loss_base).mean().item())

                if rn_batch:
                    response_norms.append(np.mean(rn_batch))
                    cosine_with_delta.append(np.mean(cos_batch))
                    loss_deltas.append(np.mean(ld_batch))

            comp_results[cond_name][s] = {
                "response_norm": float(np.mean(response_norms)),
                "response_norm_std": float(np.std(response_norms)),
                "cosine_with_delta": float(np.mean(cosine_with_delta)),
                "cosine_with_delta_std": float(np.std(cosine_with_delta)),
                "loss_delta": float(np.mean(loss_deltas)),
                "loss_delta_std": float(np.std(loss_deltas)),
                "n_dirs": len(response_norms),
            }

            print(f"    s={s:.1f}: ||R||={np.mean(response_norms):.4f} "
                  f"cos(R,δ)={np.mean(cosine_with_delta):+.4f} "
                  f"Δloss={np.mean(loss_deltas):+.4f}")

    # Compensation summary
    print(f"\n--- Compensatory response summary ---")
    print(f"{'Condition':>8s} {'s':>5s} {'||R||':>8s} {'cos(R,δ)':>10s} "
          f"{'Δloss':>8s} {'||R||/OL':>8s}")
    print("-" * 55)
    for s in s_values:
        ol_norm = comp_results["OL"][s]["response_norm"]
        for cond_name in conditions:
            r = comp_results[cond_name][s]
            ratio = r["response_norm"] / ol_norm if ol_norm > 0 else float("nan")
            print(f"{cond_name:>8s} {s:>5.1f} {r['response_norm']:>8.4f} "
                  f"{r['cosine_with_delta']:>+10.4f} "
                  f"{r['loss_delta']:>+8.4f} {ratio:>8.3f}")

    # =========================================================
    # TEST 2: Perturbation discrimination at logit level
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST 2: Perturbation discrimination (logit level)")
    print(f"{'='*60}")
    print("Can the model's output distribution distinguish which")
    print("perturbation was applied? Higher = more discriminative.\n")

    # For each pair of perturbation directions (A, B):
    #   1. Compute Δlogits_A = logits(perturbed_A) - logits(baseline)
    #   2. Compute Δlogits_B = logits(perturbed_B) - logits(baseline)
    #   3. Measure distinguishability: cosine distance between Δlogits_A and
    #      Δlogits_B. If the model's output is perturbation-specific, different
    #      perturbations should produce DIFFERENT logit shifts (low cosine).
    #      If perturbation-agnostic, all perturbations produce similar shifts
    #      (high cosine).
    #
    # We also measure: mutual information proxy via a simple classifier.
    # Given the logit shift vector, can a linear classifier identify which
    # perturbation direction produced it?

    s_test = 2.0
    n_disc_dirs = min(8, n_perturb_dirs)  # use first 8 directions for pairs
    disc_results = {}

    for cond_name, (model_key, use_injection) in conditions.items():
        model = model_closed if model_key == "closed" else model_open
        print(f"\n  [{cond_name}]")

        # Collect logit shifts for each perturbation direction
        # Shape per direction: (n_eval_batches * batch_size * (block_size-1), vocab_size)
        # Too large to store full vocab — use top-k logit diffs instead
        # Better approach: store the logit shift projected onto top singular vectors

        # Actually, let's keep it simple: for each perturbation direction,
        # compute the mean logit shift vector (averaged over positions and batches).
        # Then measure pairwise cosine distances between mean shift vectors.
        mean_logit_shifts = []  # (n_disc_dirs, vocab_size)

        for di in range(n_disc_dirs):
            dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
            perturbation = (s_test * b1_std) * dir_t

            shift_sum = None
            n_tok = 0

            with torch.no_grad():
                for bi, idx in enumerate(eval_indices):
                    x, y = make_batch(idx)

                    logits_base, _, _ = run_model(model, x, y, use_injection)
                    logits_pert, _, _ = run_model(
                        model, x, y, use_injection, perturbation)

                    # Logit shift (before softmax — raw logit difference)
                    shift = (logits_pert[:, :-1] - logits_base[:, :-1])
                    shift_flat = shift.reshape(-1, shift.shape[-1])

                    if shift_sum is None:
                        shift_sum = shift_flat.sum(dim=0)
                    else:
                        shift_sum += shift_flat.sum(dim=0)
                    n_tok += shift_flat.shape[0]

            mean_shift = (shift_sum / n_tok).cpu()
            mean_logit_shifts.append(mean_shift)

        # Stack and compute pairwise cosine similarity
        shifts = torch.stack(mean_logit_shifts)  # (n_disc_dirs, vocab_size)

        # Normalize
        shifts_normed = F.normalize(shifts, dim=-1)
        cosine_matrix = shifts_normed @ shifts_normed.T  # (n_disc_dirs, n_disc_dirs)

        # Off-diagonal mean = how similar different perturbations' effects are
        mask = ~torch.eye(n_disc_dirs, dtype=torch.bool)
        off_diag_cosine = cosine_matrix[mask].mean().item()

        # Mean shift magnitude (how much the perturbation affects logits at all)
        shift_norms = shifts.norm(dim=-1)
        mean_shift_norm = shift_norms.mean().item()

        # Per-direction shift norm variability (do different perturbations
        # produce different magnitudes of effect?)
        shift_norm_cv = float(shift_norms.std() / shift_norms.mean())

        # KL divergence between perturbed and baseline output distributions
        # (averaged over positions — measures how much the perturbation matters
        # to the output, regardless of direction specificity)
        kl_divs = []
        for di in range(n_disc_dirs):
            dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
            perturbation = (s_test * b1_std) * dir_t
            kl_sum = 0.0
            n_tok = 0
            with torch.no_grad():
                for bi, idx in enumerate(eval_indices[:5]):  # subset for speed
                    x, y = make_batch(idx)
                    logits_base, _, _ = run_model(model, x, y, use_injection)
                    logits_pert, _, _ = run_model(
                        model, x, y, use_injection, perturbation)
                    p_base = F.log_softmax(logits_base[:, :-1], dim=-1)
                    p_pert = F.softmax(logits_pert[:, :-1], dim=-1)
                    kl = F.kl_div(p_base, p_pert, reduction='none',
                                  log_target=False).sum(dim=-1)
                    kl_sum += kl.sum().item()
                    n_tok += kl.numel()
            kl_divs.append(kl_sum / n_tok)

        mean_kl = float(np.mean(kl_divs))

        disc_results[cond_name] = {
            "off_diag_cosine": off_diag_cosine,
            "mean_shift_norm": mean_shift_norm,
            "shift_norm_cv": shift_norm_cv,
            "mean_kl_divergence": mean_kl,
            "cosine_matrix": cosine_matrix.numpy().tolist(),
        }

        print(f"    Off-diag cosine: {off_diag_cosine:.4f} "
              f"(lower = more discriminative)")
        print(f"    Mean shift norm: {mean_shift_norm:.4f}")
        print(f"    Shift norm CV:   {shift_norm_cv:.4f}")
        print(f"    Mean KL div:     {mean_kl:.6f}")

    # Discrimination summary
    print(f"\n--- Perturbation discrimination summary (s={s_test}) ---")
    print(f"{'Condition':>8s} {'cos(off-diag)':>14s} {'||shift||':>10s} "
          f"{'CV':>8s} {'KL':>10s}")
    print("-" * 55)
    for cond_name in conditions:
        r = disc_results[cond_name]
        print(f"{cond_name:>8s} {r['off_diag_cosine']:>14.4f} "
              f"{r['mean_shift_norm']:>10.4f} "
              f"{r['shift_norm_cv']:>8.4f} {r['mean_kl_divergence']:>10.6f}")

    print("\nInterpretation:")
    print("  Lower off-diagonal cosine = perturbations produce more")
    print("  distinguishable output effects = model is more 'aware' of")
    print("  perturbation identity, not just perturbation presence.")

    # =========================================================
    # TEST 3: Per-position compensation analysis
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST 3: Position-level compensation structure")
    print(f"{'='*60}")
    print("Is compensation uniform or position-dependent?\n")

    # For a single perturbation strength, measure compensation by position
    # in the sequence. If the CL model compensates more at positions where
    # it has strong self-knowledge (focused attention positions), that's
    # evidence the compensation is self-knowledge-mediated.

    s_test = 2.0
    n_dirs_struct = min(8, n_perturb_dirs)

    pos_comp = {cond: np.zeros(block_size - 1) for cond in conditions}
    pos_counts = np.zeros(block_size - 1)

    for cond_name, (model_key, use_injection) in conditions.items():
        model = model_closed if model_key == "closed" else model_open

        for di in range(n_dirs_struct):
            dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
            perturbation = (s_test * b1_std) * dir_t

            with torch.no_grad():
                for bi, idx in enumerate(eval_indices[:10]):
                    x, y = make_batch(idx)

                    _, _, inter_base = run_model(model, x, y, use_injection)
                    _, _, inter_pert = run_model(
                        model, x, y, use_injection, perturbation)

                    b3_base = inter_base["post_block3"][:, :-1]
                    b3_pert = inter_pert["post_block3"][:, :-1]

                    response = b3_pert - b3_base  # (B, T-1, D)
                    r_norm = response.norm(dim=-1)  # (B, T-1)

                    r_norm_mean = r_norm.mean(dim=0).cpu().numpy()  # (T-1,)
                    pos_comp[cond_name] += r_norm_mean

                    if cond_name == list(conditions.keys())[0] and di == 0:
                        pos_counts += batch_size

        pos_comp[cond_name] /= (n_dirs_struct * pos_counts / batch_size)

    # Position-level response norm ratio (CL/OL)
    print(f"  Position-level response norm (averaged over {n_dirs_struct} dirs):")
    print(f"  {'Position':>10s} {'CL+M':>8s} {'CL-M':>8s} {'OL':>8s} "
          f"{'CL+M/OL':>8s} {'CL-M/OL':>8s}")
    # Show first, middle, last positions
    positions_to_show = [0, 1, 2, 3, 15, 31, 63, min(126, block_size - 2)]
    for p in positions_to_show:
        if p >= block_size - 1:
            continue
        ol = pos_comp["OL"][p]
        clm = pos_comp["CL+M"][p]
        clnm = pos_comp["CL-M"][p]
        print(f"  {p:>10d} {clm:>8.4f} {clnm:>8.4f} {ol:>8.4f} "
              f"{clm/ol if ol > 0 else 0:>8.3f} {clnm/ol if ol > 0 else 0:>8.3f}")

    # =========================================================
    # Save results
    # =========================================================
    if ckpt_source == "extended":
        save_dir = os.path.join(ckpt_root, f"mirror_test_v2_step{ckpt_step}")
    else:
        save_dir = os.path.join(ckpt_root, "mirror_test_v2")
    os.makedirs(save_dir, exist_ok=True)

    output = {
        "config": {
            "n_tokens": n_tokens, "n_eval_batches": n_eval_batches,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "seed": seed,
            "n_perturb_dirs": n_perturb_dirs,
            "b1_activation_std": b1_std,
            "s_values": s_values,
        },
        "test1_compensatory": comp_results,
        "test2_discrimination": disc_results,
        "test3_position_structure": {
            cond: pos_comp[cond].tolist() for cond in conditions
        },
    }

    from a2a_forward.shared import NumpyEncoder
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {save_dir}")

    # =========================================================
    # Final summary
    # =========================================================
    print(f"\n{'='*60}")
    print("MIRROR TEST v2 SUMMARY")
    print(f"{'='*60}")

    print("\nTest 1 — Compensatory response (s=2.0):")
    s_sum = 2.0
    ol_norm = comp_results["OL"][s_sum]["response_norm"]
    ol_cos = comp_results["OL"][s_sum]["cosine_with_delta"]
    for cond_name in conditions:
        r = comp_results[cond_name][s_sum]
        ratio = r["response_norm"] / ol_norm if ol_norm > 0 else float("nan")
        print(f"  {cond_name:>6s}: ||R||={r['response_norm']:.4f} "
              f"({ratio:.3f}x OL), "
              f"cos(R,δ)={r['cosine_with_delta']:+.4f} "
              f"(OL: {ol_cos:+.4f})")

    cl_damps = comp_results["CL+M"][s_sum]["response_norm"] < ol_norm
    print(f"\n  CL+M dampens perturbation: {cl_damps}")
    if cl_damps:
        pct = (1 - comp_results["CL+M"][s_sum]["response_norm"] / ol_norm) * 100
        print(f"  → CL+M absorbs {pct:.1f}% more of the perturbation than OL")

    print(f"\nTest 2 — Perturbation discrimination (s={s_test}):")
    ol_cos_disc = disc_results["OL"]["off_diag_cosine"]
    for cond_name in conditions:
        r = disc_results[cond_name]
        print(f"  {cond_name:>6s}: off-diag cos={r['off_diag_cosine']:.4f} "
              f"(lower = more discriminative)")

    cl_more_disc = disc_results["CL+M"]["off_diag_cosine"] < ol_cos_disc
    print(f"\n  CL+M more discriminative: {cl_more_disc}")
    if cl_more_disc:
        pct = (1 - disc_results["CL+M"]["off_diag_cosine"] / ol_cos_disc) * 100
        print(f"  → CL+M's output shifts are {pct:.1f}% more perturbation-specific")

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
    result = a2a_mirror_test_v2.remote(
        n_tokens=n_tokens, block_size=block_size,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
        ckpt_source=ckpt_source, ckpt_step=ckpt_step,
    )
    print("Mirror test v2 complete:")
    t1 = result["test1_compensatory"]
    print("\n  Test 1 — Compensatory response (s=2.0):")
    def _get(d, k):
        return d.get(k, d.get(str(k), d.get(float(k) if isinstance(k, str) else k)))
    ol_norm = _get(t1["OL"], 2.0)["response_norm"]
    for cond in ["CL+M", "CL-M", "OL"]:
        r = _get(t1[cond], 2.0)
        ratio = r["response_norm"] / ol_norm if ol_norm > 0 else 0
        print(f"    {cond:>6s}: ||R||={r['response_norm']:.4f} "
              f"({ratio:.3f}x OL), "
              f"cos(R,δ)={r['cosine_with_delta']:+.4f}")
    t2 = result["test2_discrimination"]
    print("\n  Test 2 — Perturbation discrimination:")
    for cond in ["CL+M", "CL-M", "OL"]:
        r = t2[cond]
        print(f"    {cond:>6s}: off-diag cos={r['off_diag_cosine']:.4f}")
