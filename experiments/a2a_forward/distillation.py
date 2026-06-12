"""Single-cycle wake-sleep distillation: can the main model absorb the FM's contribution?

Loads the controlled retrain closed-loop checkpoint (the "wake" product).
Three phases:

Phase 1 — DISTILL (Sleep):
  Freeze a teacher copy of the CL model. Run teacher with injection active to get
  soft targets. Train the student (same starting weights, NO injection) to match
  the teacher's logit distribution via KL divergence + NTP loss.
  Measure: does the dependency gap close?

Phase 2 — RE-POINT:
  Train a fresh forward model (same architecture, different seed) on the
  consolidated (distilled) model's activations. Train for the same number of
  steps as the original FM.

Phase 3 — COMPARE (Innovation migration):
  Compare the fresh FM's residual structure against the original FM's.
  Key measurements:
    - Residual eigenspectrum overlap (do they miss the same directions?)
    - Behavioral conditioning (do they struggle at the same positions?)
    - Effective rank (did complexity change?)
    - Robustness of consolidated model vs original CL and OL models
    - Self-knowledge probes on consolidated model
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,  # 8 hours
    memory=32768,
)
def a2a_distillation(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    # Distillation params
    distill_steps: int = 5_000,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,  # weight on KL vs NTP: loss = alpha*KL + (1-alpha)*CE
    distill_temperature: float = 1.0,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    # Fresh FM retraining params
    retrain_steps: int = 10_000,
    fwd_lr: float = 1e-3,
    # Architecture (must match controlled retrain)
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    # Analysis
    probe_batches: int = 40,
    probe_steps: int = 500,
    n_perturbation_dirs: int = 16,
    perturbation_scale: float = 2.0,
    seed: int = 42,
    fresh_fm_seed: int = 137,
):
    import os
    import glob
    import copy
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
    print(f"A2A DISTILLATION EXPERIMENT on {device}")
    print(f"  Distill: {distill_steps} steps, lr={distill_lr}, "
          f"alpha={distill_alpha}, T={distill_temperature}")
    print(f"  Fresh FM retrain: {retrain_steps} steps, seed={fresh_fm_seed}")

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

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + block_size] for i in indices])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # Pre-generate batch indices
    train_gen = torch.Generator().manual_seed(seed + 100)  # different from retrain
    eval_gen = torch.Generator().manual_seed(seed + 101)
    probe_gen = torch.Generator().manual_seed(seed + 102)

    max_steps = max(distill_steps, retrain_steps)
    n_evals = max_steps // eval_interval + 2
    train_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(max_steps)
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

    # --- Load checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    save_root = (f"{DATA_DIR}/a2a_forward/distillation/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    print(f"\nLoading checkpoints from {ckpt_root}")

    # Load CL model
    cl_model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    cl_model.load_state_dict(
        torch.load(f"{ckpt_root}/closed_loop/model.pt", map_location=device,
                   weights_only=True))

    # Load FM and gate
    orig_fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    orig_fm.load_state_dict(
        torch.load(f"{ckpt_root}/closed_loop/fwd_model.pt", map_location=device,
                   weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(
        torch.load(f"{ckpt_root}/closed_loop/gate.pt", map_location=device,
                   weights_only=True))

    # Load OL model (for comparison)
    ol_model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ol_model.load_state_dict(
        torch.load(f"{ckpt_root}/open_loop/model.pt", map_location=device,
                   weights_only=True))

    print(f"  CL model loaded, gate norm: {gate.injection_norm():.4f}")

    # =============================================
    # Pre-distillation baseline measurements
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PRE-DISTILLATION BASELINES")
    print(f"{'='*60}")

    def eval_lm_loss(model, fwd_model=None, gate_m=None, use_injection=False):
        model.eval()
        if fwd_model is not None:
            fwd_model.eval()
        if gate_m is not None:
            gate_m.eval()
        total = 0.0
        with torch.no_grad():
            for eb_idx in eval_indices[0]:
                vx, vy = make_batch(eb_idx, val_data)
                if use_injection and fwd_model is not None and gate_m is not None:
                    def cb_fn(act):
                        return gate_m(fwd_model(act))
                    _, vl, _ = model(
                        vx, vy, return_intermediates=True,
                        cerebellar_fn=cb_fn,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                else:
                    _, vl = model(vx, vy)
                total += float(vl)
        return total / n_eval_batches

    ol_loss = eval_lm_loss(ol_model)
    cl_with_inj = eval_lm_loss(cl_model, orig_fm, gate, use_injection=True)
    cl_no_inj = eval_lm_loss(cl_model)
    dependency_pre = cl_no_inj - cl_with_inj

    print(f"  Open-loop LM loss:         {ol_loss:.4f}")
    print(f"  CL with injection:         {cl_with_inj:.4f}")
    print(f"  CL without injection:      {cl_no_inj:.4f}")
    print(f"  Dependency gap:            {dependency_pre:+.4f}")
    print(f"  CL(no inj) vs OL:         {cl_no_inj - ol_loss:+.4f}")

    # =============================================
    # Phase 1: DISTILLATION
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: DISTILLATION ({distill_steps} steps)")
    print(f"{'='*60}")

    # Freeze teacher: a copy of the CL model + FM + gate
    teacher_model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    teacher_model.load_state_dict(cl_model.state_dict())
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    teacher_fm.load_state_dict(orig_fm.state_dict())
    teacher_fm.eval()
    for p in teacher_fm.parameters():
        p.requires_grad = False

    teacher_gate = CerebellarGate(n_embd).to(device)
    teacher_gate.load_state_dict(gate.state_dict())
    teacher_gate.eval()
    for p in teacher_gate.parameters():
        p.requires_grad = False

    # Student: the CL model being optimized (run WITHOUT injection)
    student = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    student.load_state_dict(cl_model.state_dict())

    opt = torch.optim.AdamW(student.parameters(), lr=distill_lr, weight_decay=0.01)

    T = distill_temperature
    history = {
        "distill_loss": [], "kl_loss": [], "ce_loss": [],
        "val_student": [], "val_student_vs_teacher": [],
    }

    eval_idx = 0
    for step in range(distill_steps):
        student.train()

        x, y = make_batch(train_indices[step], train_data)

        # Teacher forward pass (with injection, frozen)
        with torch.no_grad():
            def teacher_cb_fn(act):
                return teacher_gate(teacher_fm(act))
            teacher_logits, _, _ = teacher_model(
                x, y, return_intermediates=True,
                cerebellar_fn=teacher_cb_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

        # Student forward pass (no injection)
        student_logits, ce_loss = student(x, y)

        # KL divergence on logit distributions
        if T != 1.0:
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        else:
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )

        loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        # --- Eval ---
        if step % eval_interval == 0 or step == distill_steps - 1:
            student.eval()
            with torch.no_grad():
                val_student_acc = 0.0
                val_kl_acc = 0.0
                for eb_idx in eval_indices[eval_idx]:
                    vx, vy = make_batch(eb_idx, val_data)

                    # Student (no injection)
                    s_logits, s_loss = student(vx, vy)
                    val_student_acc += float(s_loss)

                    # Teacher (with injection)
                    def eval_teacher_cb(act):
                        return teacher_gate(teacher_fm(act))
                    t_logits, _, _ = teacher_model(
                        vx, vy, return_intermediates=True,
                        cerebellar_fn=eval_teacher_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    kl = F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        F.softmax(t_logits, dim=-1),
                        reduction="batchmean",
                    )
                    val_kl_acc += float(kl)

                n = n_eval_batches
                val_s = val_student_acc / n
                val_kl = val_kl_acc / n

                history["distill_loss"].append((step, float(loss)))
                history["kl_loss"].append((step, float(kl_loss)))
                history["ce_loss"].append((step, float(ce_loss)))
                history["val_student"].append((step, val_s))
                history["val_student_vs_teacher"].append((step, val_kl))

                print(f"  [DISTILL] step {step:5d}: "
                      f"val_lm={val_s:.4f} "
                      f"KL(s||t)={val_kl:.5f} | "
                      f"train: kl={float(kl_loss):.5f} ce={float(ce_loss):.4f}")

            eval_idx += 1

    # Post-distillation measurements
    distilled_loss = eval_lm_loss(student)
    dependency_post = distilled_loss - cl_with_inj

    print(f"\n  POST-DISTILLATION:")
    print(f"    Distilled model LM loss:   {distilled_loss:.4f}")
    print(f"    Teacher (CL+inj):          {cl_with_inj:.4f}")
    print(f"    Original CL (no inj):      {cl_no_inj:.4f}")
    print(f"    Open-loop:                 {ol_loss:.4f}")
    print(f"    Dependency gap: {dependency_pre:+.4f} → {dependency_post:+.4f}")
    gap_closed = 1.0 - (dependency_post / dependency_pre) if dependency_pre != 0 else 0
    print(f"    Gap closed: {gap_closed:.1%}")

    # Save distilled checkpoint
    distill_save = os.path.join(save_root, "distilled")
    os.makedirs(distill_save, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(distill_save, "model.pt"))
    volume.commit()
    print(f"  Saved distilled checkpoint to {distill_save}")

    # Clean up teacher
    del teacher_model, teacher_fm, teacher_gate
    torch.cuda.empty_cache()

    # =============================================
    # Phase 2: RE-POINT (train fresh FM on consolidated model)
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: FRESH FM TRAINING ({retrain_steps} steps, seed={fresh_fm_seed})")
    print(f"{'='*60}")

    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    # Fresh FM with different seed
    torch.manual_seed(fresh_fm_seed)
    fresh_fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)

    opt_fm = torch.optim.AdamW(fresh_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

    fm_history = {"mse": [], "cosine": [], "val_mse": [], "val_cosine": []}

    fm_eval_idx = 0
    for step in range(retrain_steps):
        fresh_fm.train()
        x, y = make_batch(train_indices[step], train_data)

        with torch.no_grad():
            _, _, intermediates = student(x, y, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]

        predicted = fresh_fm(source)
        fwd_loss = F.mse_loss(predicted, target)

        opt_fm.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fresh_fm.parameters(), 1.0)
        opt_fm.step()

        if step % eval_interval == 0 or step == retrain_steps - 1:
            fresh_fm.eval()
            with torch.no_grad():
                val_mse_acc = 0.0
                val_cos_acc = 0.0
                for eb_idx in eval_indices[fm_eval_idx]:
                    vx, vy = make_batch(eb_idx, val_data)
                    _, _, vi = student(vx, vy, return_intermediates=True)
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fresh_fm(src)
                    val_mse_acc += float(F.mse_loss(pred, tgt))
                    val_cos_acc += float(
                        F.cosine_similarity(pred, tgt, dim=-1).mean())

                n = n_eval_batches
                vm = val_mse_acc / n
                vc = val_cos_acc / n
                fm_history["val_mse"].append((step, vm))
                fm_history["val_cosine"].append((step, vc))
                fm_history["mse"].append((step, float(fwd_loss)))
                fm_history["cosine"].append((step, float(
                    F.cosine_similarity(predicted, target, dim=-1).mean())))

                print(f"  [FRESH FM] step {step:5d}: "
                      f"val_mse={vm:.5f} val_cos={vc:.4f}")
            fm_eval_idx += 1

    # Save fresh FM
    torch.save(fresh_fm.state_dict(), os.path.join(distill_save, "fresh_fm.pt"))
    volume.commit()

    # Re-enable student gradients for probing (not needed but clean)
    for p in student.parameters():
        p.requires_grad = True

    # =============================================
    # Phase 3: COMPARE — innovation migration
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 3: INNOVATION MIGRATION ANALYSIS")
    print(f"{'='*60}")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    # --- Collect residuals from original FM and fresh FM ---
    def collect_residuals(model, fwd_model, label):
        model.eval()
        fwd_model.eval()
        residuals = []
        residual_norms = []
        acts = {k: [] for k in layer_keys}

        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                res = tgt - pred
                residuals.append(res.reshape(-1, n_embd).cpu())
                residual_norms.append(res.norm(dim=-1).reshape(-1).cpu())
                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())

        residuals = torch.cat(residuals)
        residual_norms = torch.cat(residual_norms)
        acts = {k: torch.cat(v) for k, v in acts.items()}
        print(f"  [{label}] Collected {residuals.shape[0]} samples, "
              f"mean residual norm: {residual_norms.mean():.4f}")
        return residuals, residual_norms, acts

    # Original FM on original CL model (the pre-distillation reference)
    print("\nCollecting original FM residuals (on CL model)...")
    orig_res, orig_res_norms, _ = collect_residuals(cl_model, orig_fm, "orig_FM→CL")

    # Original FM on distilled model (how much did the model change?)
    print("Collecting original FM residuals (on distilled model)...")
    orig_on_distilled_res, orig_on_distilled_norms, _ = collect_residuals(
        student, orig_fm, "orig_FM→distilled")

    # Fresh FM on distilled model (the new innovation)
    print("Collecting fresh FM residuals (on distilled model)...")
    fresh_res, fresh_res_norms, distilled_acts = collect_residuals(
        student, fresh_fm, "fresh_FM→distilled")

    # Also collect OL model acts for probe comparison
    print("Collecting OL model activations...")
    _, _, ol_acts = collect_residuals(ol_model, orig_fm, "orig_FM→OL")

    # --- 3a: Residual eigenspectrum comparison ---
    print(f"\n--- 3a: Residual eigenspectrum ---")

    def compute_residual_stats(residuals, label):
        R = residuals.numpy().astype(np.float64)
        R_centered = R - R.mean(axis=0, keepdims=True)
        cov = (R_centered.T @ R_centered) / (R_centered.shape[0] - 1)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        total_var = eigenvalues.sum()
        fracs = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Effective rank
        fracs_pos = fracs[fracs > 1e-12]
        entropy = -np.sum(fracs_pos * np.log(fracs_pos))
        eff_rank = np.exp(entropy)

        # Cumulative variance thresholds
        cumvar = np.cumsum(fracs)
        rank_50 = int(np.searchsorted(cumvar, 0.5)) + 1
        rank_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1

        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"top1={fracs[0]:.3f}, top5={fracs[:5].sum():.3f}, "
              f"rank50={rank_50}, rank90={rank_90}, rank95={rank_95}")

        return {
            "eigenvalues": eigenvalues.tolist()[:50],
            "eff_rank": float(eff_rank),
            "top1_frac": float(fracs[0]),
            "top5_frac": float(fracs[:5].sum()),
            "rank_50": rank_50, "rank_90": rank_90, "rank_95": rank_95,
            "mean_norm": float(residuals.norm(dim=-1).mean()),
        }

    orig_stats = compute_residual_stats(orig_res, "orig FM on CL")
    orig_on_dist_stats = compute_residual_stats(
        orig_on_distilled_res, "orig FM on distilled")
    fresh_stats = compute_residual_stats(fresh_res, "fresh FM on distilled")

    # --- 3b: Innovation direction overlap ---
    print(f"\n--- 3b: Innovation direction overlap ---")

    def top_k_eigenvecs(residuals, k=20):
        R = residuals.numpy().astype(np.float64)
        R_centered = R - R.mean(axis=0, keepdims=True)
        cov = (R_centered.T @ R_centered) / (R_centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        return eigenvectors[:, idx[:k]], eigenvalues[idx[:k]]

    orig_vecs, orig_vals = top_k_eigenvecs(orig_res, k=20)
    fresh_vecs, fresh_vals = top_k_eigenvecs(fresh_res, k=20)

    # Subspace overlap: fraction of orig top-k variance captured by fresh top-k
    # Using principal angles between the two subspaces
    for k in [5, 10, 20]:
        O = orig_vecs[:, :k]  # (d, k)
        F_v = fresh_vecs[:, :k]  # (d, k)
        # Compute cos of principal angles via SVD of O^T @ F
        M = O.T @ F_v
        svals = np.linalg.svd(M, compute_uv=False)
        # Grassmann distance would be arccos of singular values
        # Simpler: mean cos(angle) = mean of singular values
        mean_cos = float(svals.mean())
        # Fraction of orig variance in fresh subspace
        # Project orig residuals onto fresh top-k and measure variance captured
        orig_np = orig_res.numpy().astype(np.float64)
        orig_centered = orig_np - orig_np.mean(axis=0, keepdims=True)
        total_var = np.var(orig_centered)
        projected = orig_centered @ F_v @ F_v.T
        captured_var = np.var(projected)
        frac = captured_var / total_var if total_var > 0 else 0
        print(f"  k={k}: mean_cos(principal_angles)={mean_cos:.4f}, "
              f"orig_var_in_fresh_subspace={frac:.4f}")

    # Direct cosine between top eigenvectors
    for i in range(min(5, orig_vecs.shape[1])):
        cos = float(np.abs(orig_vecs[:, i] @ fresh_vecs[:, i]))
        print(f"  |cos(orig_PC{i}, fresh_PC{i})| = {cos:.4f}")

    # --- 3c: Behavioral conditioning ---
    print(f"\n--- 3c: Behavioral conditioning ---")

    # Collect behavioral categories for probe positions
    student.eval()
    fresh_fm.eval()
    orig_fm.eval()
    behavioral_data = {
        "orig_res_norm": [], "fresh_res_norm": [],
        "attn_entropy": [], "max_attn": [],
    }
    with torch.no_grad():
        for pidx in probe_indices[:10]:  # subset for speed
            vx, vy = make_batch(pidx, val_data)

            # Get attention patterns from block0 for categorization
            _, _, vi = student(vx, vy, return_intermediates=True)
            src = vi[predict_from]
            tgt = vi[predict_to]

            # Original FM residuals
            orig_pred = orig_fm(src)
            orig_r = (tgt - orig_pred).norm(dim=-1).reshape(-1)
            behavioral_data["orig_res_norm"].append(orig_r.cpu())

            # Fresh FM residuals
            fresh_pred = fresh_fm(src)
            fresh_r = (tgt - fresh_pred).norm(dim=-1).reshape(-1)
            behavioral_data["fresh_res_norm"].append(fresh_r.cpu())

    orig_norms = torch.cat(behavioral_data["orig_res_norm"]).numpy()
    fresh_norms = torch.cat(behavioral_data["fresh_res_norm"]).numpy()

    # Correlation between original and fresh residual norms
    corr = float(np.corrcoef(orig_norms, fresh_norms)[0, 1])
    print(f"  Correlation(orig_res_norm, fresh_res_norm) = {corr:.4f}")
    print(f"  Mean orig residual norm: {orig_norms.mean():.4f}")
    print(f"  Mean fresh residual norm: {fresh_norms.mean():.4f}")

    # --- 3d: Robustness comparison ---
    print(f"\n--- 3d: Robustness ---")

    torch.manual_seed(seed + 200)
    perturbation_dirs = torch.randn(n_perturbation_dirs, n_embd, device=device)
    perturbation_dirs = perturbation_dirs / perturbation_dirs.norm(dim=-1, keepdim=True)

    def measure_robustness(model, label):
        model.eval()
        # Get activation scale at injection point
        act_norms = []
        with torch.no_grad():
            for eb_idx in eval_indices[0][:3]:
                vx, vy = make_batch(eb_idx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)
                act_norms.append(
                    vi[f"post_block{inject_after_block}"]
                    .norm(dim=-1).mean().item())
        act_scale = np.mean(act_norms)

        # Clean loss
        clean_loss = eval_lm_loss(model)

        # Perturbed losses
        perturbed_losses = []
        for d_idx in range(n_perturbation_dirs):
            delta = perturbation_dirs[d_idx] * perturbation_scale * act_scale
            p_loss_acc = 0.0
            with torch.no_grad():
                for eb_idx in eval_indices[0]:
                    vx, vy = make_batch(eb_idx, val_data)

                    def perturb_fn(act, _delta=delta):
                        return _delta.unsqueeze(0).unsqueeze(0).expand_as(act)

                    _, vl, _ = model(
                        vx, vy, return_intermediates=True,
                        cerebellar_fn=perturb_fn,
                        cerebellar_input_block=inject_after_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    p_loss_acc += float(vl)
            perturbed_losses.append(p_loss_acc / n_eval_batches)

        mean_delta = float(np.mean(perturbed_losses) - clean_loss)
        print(f"  [{label}] clean={clean_loss:.4f}, "
              f"mean Δloss={mean_delta:+.4f}")
        return {
            "clean_loss": clean_loss,
            "mean_delta_loss": mean_delta,
            "per_dir_delta": [float(p - clean_loss) for p in perturbed_losses],
        }

    rob_ol = measure_robustness(ol_model, "OL")
    rob_cl = measure_robustness(cl_model, "CL (pre-distill)")
    rob_distilled = measure_robustness(student, "Distilled")

    if rob_ol["mean_delta_loss"] > 0:
        print(f"\n  Robustness ratios (vs OL):")
        print(f"    CL:        {rob_cl['mean_delta_loss'] / rob_ol['mean_delta_loss']:.3f}")
        print(f"    Distilled: {rob_distilled['mean_delta_loss'] / rob_ol['mean_delta_loss']:.3f}")

    # --- 3e: Self-knowledge probes ---
    print(f"\n--- 3e: Self-knowledge probes ---")

    n_samples = fresh_res.shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    def vector_probe(X, target_vec):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = target_vec.numpy() if isinstance(target_vec, torch.Tensor) \
            else target_vec
        T_np = T_np.astype(np.float32)

        mu_x = X_np[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu_x) / sd_x

        probe = nn.Linear(X_np.shape[1], T_np.shape[1]).to(device)
        opt_p = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ttr = torch.from_numpy(T_np[train_idx]).float().to(device)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,), device=device)
            pred = probe(Xtr[idx])
            loss = F.mse_loss(pred, Ttr[idx])
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).cpu().numpy()
        Tte = T_np[test_idx]
        mse = ((pred - Tte) ** 2).mean()
        var = Tte.var()
        r2 = float(1.0 - mse / var) if var > 0 else 0.0
        return r2

    # Probe the distilled model's activations for the fresh FM's residual
    # Compare with OL model's activations probed for the same target
    print(f"\n  Probing distilled model vs OL model for fresh FM residual:")
    print(f"  {'layer':>12s} {'OL R²':>8s} {'Distilled R²':>13s} {'Δ':>7s}")
    probe_results = {}
    for lk in layer_keys:
        r2_ol = vector_probe(ol_acts[lk], fresh_res)
        r2_dist = vector_probe(distilled_acts[lk], fresh_res)
        delta = r2_dist - r2_ol
        probe_results[lk] = {
            "ol": r2_ol, "distilled": r2_dist, "delta": delta,
        }
        print(f"  {lk:>12s} {r2_ol:>8.4f} {r2_dist:>13.4f} {delta:>+7.4f}")

    # =============================================
    # Save everything
    # =============================================
    result = {
        "config": {
            "distill_steps": distill_steps, "distill_lr": distill_lr,
            "distill_alpha": distill_alpha, "distill_temperature": distill_temperature,
            "retrain_steps": retrain_steps, "fwd_lr": fwd_lr,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "seed": seed, "fresh_fm_seed": fresh_fm_seed,
            "n_tokens": n_tokens,
        },
        "pre_distillation": {
            "ol_loss": ol_loss,
            "cl_with_inj": cl_with_inj,
            "cl_no_inj": cl_no_inj,
            "dependency_gap": dependency_pre,
        },
        "post_distillation": {
            "distilled_loss": distilled_loss,
            "dependency_gap": dependency_post,
            "gap_closed_frac": gap_closed,
        },
        "fresh_fm": {
            "final_cosine": fm_history["val_cosine"][-1][1] if fm_history["val_cosine"] else None,
            "final_mse": fm_history["val_mse"][-1][1] if fm_history["val_mse"] else None,
        },
        "innovation_migration": {
            "residual_norm_correlation": corr,
            "orig_stats": orig_stats,
            "orig_on_distilled_stats": orig_on_dist_stats,
            "fresh_stats": fresh_stats,
        },
        "robustness": {
            "ol": rob_ol,
            "cl": rob_cl,
            "distilled": rob_distilled,
        },
        "self_knowledge_probes": probe_results,
        "distill_history": history,
        "fm_history": fm_history,
    }

    results_path = os.path.join(save_root, "results.json")
    os.makedirs(save_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED to {save_root}")
    print(f"{'='*60}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Dependency gap: {dependency_pre:+.4f} → {dependency_post:+.4f} "
          f"({gap_closed:.1%} closed)")
    print(f"  Distilled LM loss: {distilled_loss:.4f} "
          f"(OL: {ol_loss:.4f}, CL+inj: {cl_with_inj:.4f})")
    print(f"  Fresh FM cosine: {fm_history['val_cosine'][-1][1]:.4f}" if fm_history['val_cosine'] else "")
    print(f"  Residual norm corr (orig↔fresh): {corr:.4f}")
    print(f"  Robustness (Δloss ratio vs OL):")
    if rob_ol["mean_delta_loss"] > 0:
        print(f"    CL:        {rob_cl['mean_delta_loss'] / rob_ol['mean_delta_loss']:.3f}")
        print(f"    Distilled: {rob_distilled['mean_delta_loss'] / rob_ol['mean_delta_loss']:.3f}")
    print(f"  Eigenspectrum:")
    print(f"    Orig FM:  eff_rank={orig_stats['eff_rank']:.1f}, "
          f"top1={orig_stats['top1_frac']:.3f}")
    print(f"    Fresh FM: eff_rank={fresh_stats['eff_rank']:.1f}, "
          f"top1={fresh_stats['top1_frac']:.3f}")

    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    distill_steps: int = 5_000,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    retrain_steps: int = 10_000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_distillation.remote(
        n_tokens=n_tokens,
        distill_steps=distill_steps, distill_lr=distill_lr,
        distill_alpha=distill_alpha,
        retrain_steps=retrain_steps,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("\nDistillation experiment complete.")
    print(f"  Dependency gap: "
          f"{result['pre_distillation']['dependency_gap']:+.4f} → "
          f"{result['post_distillation']['dependency_gap']:+.4f} "
          f"({result['post_distillation']['gap_closed_frac']:.1%} closed)")
