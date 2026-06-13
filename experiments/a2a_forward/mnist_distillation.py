"""MNIST single-cycle wake-sleep distillation.

Adapts the language distillation experiment to MNIST classification.
The MNIST residual is low-rank (eff rank 18/128) and digit-discriminative,
so structural internalization should be cleaner and more interpretable
than on language (eff rank 200/256).

Phase 1 — Distillation (Sleep): absorb FM contribution into main model
Phase 2 — Re-point: train fresh FM on distilled model
Phase 3 — Innovation migration analysis:
  - Eigenspectrum comparison (did residual concentration change?)
  - Digit-discriminative structure (did different digits become easy/hard?)
  - Innovation direction overlap (did PCs rotate?)
  - Internalization probes (Tests 1-3 from language distillation)
  - Robustness comparison
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist_distillation(
    # Architecture (must match mnist_experiment.py)
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    # FM architecture
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    # Distillation params
    distill_steps: int = 2000,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    distill_temperature: float = 1.0,
    eval_interval: int = 100,
    n_eval_batches: int = 5,
    # Fresh FM params
    retrain_steps: int = 5000,
    fwd_lr: float = 1e-3,
    fresh_fm_seed: int = 137,
    # Analysis params
    probe_batches: int = 40,
    probe_steps: int = 300,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    n_positions = (28 // patch_size) ** 2 + 1

    print(f"MNIST DISTILLATION EXPERIMENT on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  Distill: {distill_steps} steps, lr={distill_lr}, alpha={distill_alpha}")
    print(f"  Fresh FM: {retrain_steps} steps, seed={fresh_fm_seed}")

    # --- Load MNIST ---
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")

    train_pil = ds["train"]["image"]
    train_imgs = np.stack([np.array(img) for img in train_pil])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])

    test_pil = ds["test"]["image"]
    test_imgs = np.stack([np.array(img) for img in test_pil])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # --- Pre-generate batch indices ---
    train_gen = torch.Generator().manual_seed(seed + 200)
    eval_gen = torch.Generator().manual_seed(seed + 201)
    probe_gen = torch.Generator().manual_seed(seed + 202)

    max_steps = max(distill_steps, retrain_steps)
    n_evals = max_steps // eval_interval + 2
    train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=train_gen)
        for _ in range(max_steps)
    ]
    eval_indices = [
        [torch.randint(len(test_images), (batch_size,), generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(test_images), (batch_size,), generator=probe_gen)
        for _ in range(probe_batches)
    ]

    # --- Load checkpoints ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    ckpt_root = f"{DATA_DIR}/a2a_forward/mnist/{model_tag}/{gap_tag}"
    save_root = f"{DATA_DIR}/a2a_forward/mnist_distillation/{model_tag}/{gap_tag}"

    print(f"\nLoading checkpoints from {ckpt_root}")

    def make_vit():
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)

    ol_model = make_vit()
    ol_model.load_state_dict(
        torch.load(f"{ckpt_root}/ol_model.pt", map_location=device, weights_only=True))

    cl_model = make_vit()
    cl_model.load_state_dict(
        torch.load(f"{ckpt_root}/cl_model.pt", map_location=device, weights_only=True))

    orig_fm = make_fm()
    orig_fm.load_state_dict(
        torch.load(f"{ckpt_root}/cl_fwd.pt", map_location=device, weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(
        torch.load(f"{ckpt_root}/cl_gate.pt", map_location=device, weights_only=True))

    print(f"  Loaded OL, CL, FM, gate. Gate norm: {gate.injection_norm():.4f}")

    # =============================================
    # Pre-distillation baselines
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PRE-DISTILLATION BASELINES")
    print(f"{'='*60}")

    def eval_model(model, fwd_model=None, gate_m=None, use_injection=False):
        model.eval()
        if fwd_model:
            fwd_model.eval()
        if gate_m:
            gate_m.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for eidx in eval_indices[0]:
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)
                if use_injection and fwd_model and gate_m:
                    def cb(act):
                        return gate_m(fwd_model(act))
                    logits, loss, _ = model(
                        images, labels, return_intermediates=True,
                        cerebellar_fn=cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                else:
                    logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        n = n_eval_batches
        return total_loss / n, total_acc / n

    ol_loss, ol_acc = eval_model(ol_model)
    cl_inj_loss, cl_inj_acc = eval_model(cl_model, orig_fm, gate, use_injection=True)
    cl_noinj_loss, cl_noinj_acc = eval_model(cl_model)
    dependency_pre = cl_noinj_loss - cl_inj_loss

    print(f"  Open-loop:       loss={ol_loss:.4f}  acc={ol_acc:.4f}")
    print(f"  CL with inj:     loss={cl_inj_loss:.4f}  acc={cl_inj_acc:.4f}")
    print(f"  CL without inj:  loss={cl_noinj_loss:.4f}  acc={cl_noinj_acc:.4f}")
    print(f"  Dependency gap:  {dependency_pre:+.4f}")

    # =============================================
    # Phase 1: DISTILLATION
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: DISTILLATION ({distill_steps} steps)")
    print(f"{'='*60}")

    # Freeze teacher (CL + FM + gate)
    teacher = make_vit()
    teacher.load_state_dict(cl_model.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_fm = make_fm()
    teacher_fm.load_state_dict(orig_fm.state_dict())
    teacher_fm.eval()
    for p in teacher_fm.parameters():
        p.requires_grad = False

    teacher_gate = CerebellarGate(n_embd).to(device)
    teacher_gate.load_state_dict(gate.state_dict())
    teacher_gate.eval()
    for p in teacher_gate.parameters():
        p.requires_grad = False

    # Student: CL weights, trained without injection
    student = make_vit()
    student.load_state_dict(cl_model.state_dict())

    opt = torch.optim.AdamW(student.parameters(), lr=distill_lr, weight_decay=0.01)

    T = distill_temperature
    distill_history = {
        "total_loss": [], "kl_loss": [], "ce_loss": [],
        "val_loss": [], "val_acc": [], "val_kl": [],
    }

    eval_idx = 0
    for step in range(distill_steps):
        student.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        with torch.no_grad():
            def teacher_cb(act):
                return teacher_gate(teacher_fm(act))
            teacher_logits, _, _ = teacher(
                images, labels, return_intermediates=True,
                cerebellar_fn=teacher_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

        student_logits, ce_loss = student(images, labels)

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

        if step % eval_interval == 0 or step == distill_steps - 1:
            student.eval()
            with torch.no_grad():
                v_loss, v_acc, v_kl = 0.0, 0.0, 0.0
                for eidx in eval_indices[eval_idx]:
                    vim = test_images[eidx].to(device)
                    vlb = test_labels[eidx].to(device)

                    s_logits, s_loss = student(vim, vlb)
                    v_loss += s_loss.item()
                    v_acc += (s_logits.argmax(-1) == vlb).float().mean().item()

                    def eval_cb(act):
                        return teacher_gate(teacher_fm(act))
                    t_logits, _, _ = teacher(
                        vim, vlb, return_intermediates=True,
                        cerebellar_fn=eval_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    kl = F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        F.softmax(t_logits, dim=-1),
                        reduction="batchmean",
                    )
                    v_kl += kl.item()

                n = n_eval_batches
                v_loss /= n
                v_acc /= n
                v_kl /= n

                distill_history["total_loss"].append((step, loss.item()))
                distill_history["kl_loss"].append((step, kl_loss.item()))
                distill_history["ce_loss"].append((step, ce_loss.item()))
                distill_history["val_loss"].append((step, v_loss))
                distill_history["val_acc"].append((step, v_acc))
                distill_history["val_kl"].append((step, v_kl))

                print(f"  [DISTILL] step {step:5d}: "
                      f"val_loss={v_loss:.4f} acc={v_acc:.4f} "
                      f"KL(s||t)={v_kl:.5f}")

            eval_idx += 1

    # Post-distillation measurements
    dist_loss, dist_acc = eval_model(student)
    dependency_post = dist_loss - cl_inj_loss
    gap_closed = 1.0 - (dependency_post / dependency_pre) if dependency_pre != 0 else 0

    print(f"\n  POST-DISTILLATION:")
    print(f"    Distilled:   loss={dist_loss:.4f}  acc={dist_acc:.4f}")
    print(f"    Teacher:     loss={cl_inj_loss:.4f}  acc={cl_inj_acc:.4f}")
    print(f"    OL:          loss={ol_loss:.4f}  acc={ol_acc:.4f}")
    print(f"    Dep gap: {dependency_pre:+.4f} -> {dependency_post:+.4f} "
          f"({gap_closed:.1%} closed)")

    # Save distilled checkpoint
    os.makedirs(save_root, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(save_root, "distilled_model.pt"))
    volume.commit()

    del teacher, teacher_fm, teacher_gate
    torch.cuda.empty_cache()

    # =============================================
    # Phase 2: FRESH FM TRAINING
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: FRESH FM ({retrain_steps} steps, seed={fresh_fm_seed})")
    print(f"{'='*60}")

    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    torch.manual_seed(fresh_fm_seed)
    fresh_fm = make_fm()

    opt_fm = torch.optim.AdamW(fresh_fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    fm_history = {"mse": [], "cosine": [], "val_mse": [], "val_cosine": []}

    fm_eval_idx = 0
    for step in range(retrain_steps):
        fresh_fm.train()
        idx = train_indices[step]
        images = train_images[idx].to(device)

        with torch.no_grad():
            _, _, intermediates = student(images, return_intermediates=True)
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
                vm, vc = 0.0, 0.0
                for eidx in eval_indices[fm_eval_idx]:
                    vim = test_images[eidx].to(device)
                    _, _, vi = student(vim, return_intermediates=True)
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fresh_fm(src)
                    vm += F.mse_loss(pred, tgt).item()
                    vc += F.cosine_similarity(pred, tgt, dim=-1).mean().item()

                n = n_eval_batches
                vm /= n
                vc /= n
                fm_history["val_mse"].append((step, vm))
                fm_history["val_cosine"].append((step, vc))
                fm_history["mse"].append((step, fwd_loss.item()))
                fm_history["cosine"].append((step, float(
                    F.cosine_similarity(predicted, target, dim=-1).mean())))

                print(f"  [FRESH FM] step {step:5d}: "
                      f"val_mse={vm:.5f} val_cos={vc:.4f}")
            fm_eval_idx += 1

    torch.save(fresh_fm.state_dict(), os.path.join(save_root, "fresh_fm.pt"))
    volume.commit()

    for p in student.parameters():
        p.requires_grad = True

    # =============================================
    # Phase 3: INNOVATION MIGRATION ANALYSIS
    # =============================================
    print(f"\n{'='*60}")
    print(f"  PHASE 3: INNOVATION MIGRATION ANALYSIS")
    print(f"{'='*60}")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    # --- Collect residuals ---
    def collect_residuals(model, fwd_model, label):
        model.eval()
        fwd_model.eval()
        all_res = []
        per_image_res = []
        per_image_digits = []
        layer_acts = {k: [] for k in layer_keys}

        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                digits = test_labels[pidx]
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                res = tgt - pred

                all_res.append(res.reshape(-1, n_embd).cpu())
                per_image_res.append(res.mean(dim=1).cpu())
                per_image_digits.append(digits)
                for k in layer_keys:
                    layer_acts[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(all_res)
        per_image_res = torch.cat(per_image_res)
        per_image_digits = torch.cat(per_image_digits)
        layer_acts = {k: torch.cat(v) for k, v in layer_acts.items()}
        print(f"  [{label}] {all_res.shape[0]} position-samples, "
              f"{per_image_res.shape[0]} images, "
              f"mean res norm: {all_res.norm(dim=-1).mean():.4f}")
        return all_res, per_image_res, per_image_digits, layer_acts

    print("\nCollecting residuals...")
    orig_res, orig_img_res, orig_digits, cl_acts = collect_residuals(
        cl_model, orig_fm, "orig_FM on CL")
    orig_on_dist_res, _, _, _ = collect_residuals(
        student, orig_fm, "orig_FM on distilled")
    fresh_res, fresh_img_res, fresh_digits, dist_acts = collect_residuals(
        student, fresh_fm, "fresh_FM on distilled")
    _, _, _, ol_acts = collect_residuals(
        ol_model, orig_fm, "orig_FM on OL")

    # --- 3a: Residual eigenspectrum ---
    print(f"\n--- 3a: Residual eigenspectrum ---")

    def compute_eigen_stats(residuals, label):
        R = residuals.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals = np.maximum(evals, 0)
        total = evals.sum()
        fracs = evals / total if total > 0 else evals
        fracs_pos = fracs[fracs > 1e-12]
        eff_rank = float(np.exp(-np.sum(fracs_pos * np.log(fracs_pos))))
        cumvar = np.cumsum(fracs)
        rank_50 = int(np.searchsorted(cumvar, 0.5)) + 1
        rank_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        stats = {
            "eff_rank": eff_rank,
            "top1": float(fracs[0]),
            "top5": float(fracs[:5].sum()),
            "top10": float(fracs[:10].sum()),
            "rank_50": rank_50,
            "rank_90": rank_90,
            "mean_norm": float(residuals.norm(dim=-1).mean()),
        }
        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"top1={fracs[0]*100:.1f}%, top5={fracs[:5].sum()*100:.1f}%, "
              f"rank50={rank_50}, rank90={rank_90}, norm={stats['mean_norm']:.3f}")
        return stats

    orig_stats = compute_eigen_stats(orig_res, "orig FM on CL")
    orig_on_dist_stats = compute_eigen_stats(orig_on_dist_res, "orig FM on distilled")
    fresh_stats = compute_eigen_stats(fresh_res, "fresh FM on distilled")

    # --- 3b: Innovation direction overlap ---
    print(f"\n--- 3b: Innovation direction overlap ---")

    def top_k_eigenvecs(residuals, k=20):
        R = residuals.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        return evecs[:, idx[:k]], evals[idx[:k]]

    orig_vecs, _ = top_k_eigenvecs(orig_res, k=20)
    fresh_vecs, _ = top_k_eigenvecs(fresh_res, k=20)

    direction_overlap = {}
    for k in [5, 10, 20]:
        O = orig_vecs[:, :k]
        Fv = fresh_vecs[:, :k]
        M = O.T @ Fv
        svals = np.linalg.svd(M, compute_uv=False)
        mean_cos = float(svals.mean())
        orig_np = orig_res.numpy().astype(np.float64)
        orig_c = orig_np - orig_np.mean(axis=0, keepdims=True)
        total_var = np.var(orig_c)
        proj = orig_c @ Fv @ Fv.T
        captured = np.var(proj)
        frac = captured / total_var if total_var > 0 else 0
        direction_overlap[str(k)] = {
            "mean_cos_principal_angles": mean_cos,
            "orig_var_in_fresh_subspace": frac,
        }
        print(f"  k={k}: mean_cos={mean_cos:.4f}, "
              f"orig_var_in_fresh={frac:.4f}")

    pc_cosines = []
    for i in range(min(10, orig_vecs.shape[1])):
        cos = float(np.abs(orig_vecs[:, i] @ fresh_vecs[:, i]))
        pc_cosines.append(cos)
        if i < 5:
            print(f"  |cos(orig_PC{i}, fresh_PC{i})| = {cos:.4f}")

    # --- 3c: Digit-discriminative analysis ---
    print(f"\n--- 3c: Digit-discriminative analysis ---")

    def digit_analysis(per_image_res, digits, label, n_pcs=5):
        res_np = per_image_res.numpy().astype(np.float64)
        res_c = res_np - res_np.mean(axis=0, keepdims=True)
        digits_np = digits.numpy()

        # PCA on per-image residuals
        cov = (res_c.T @ res_c) / (res_c.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        vecs = evecs[:, idx[:n_pcs]]
        projections = res_c @ vecs

        result = {"pcs": []}
        print(f"\n  [{label}] Digit-discriminative PCs:")
        for pc in range(n_pcs):
            proj = projections[:, pc]
            grand_mean = proj.mean()
            ss_total = ((proj - grand_mean) ** 2).sum()
            ss_between = 0.0
            digit_means = {}
            for d in range(10):
                mask = digits_np == d
                if mask.sum() > 0:
                    dm = proj[mask].mean()
                    digit_means[d] = float(dm)
                    ss_between += mask.sum() * (dm - grand_mean) ** 2
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

            sorted_d = sorted(digit_means.items(), key=lambda x: x[1])
            low_d, low_v = sorted_d[0]
            high_d, high_v = sorted_d[-1]

            pc_info = {
                "eta_squared": eta_sq,
                "digit_means": {str(k): v for k, v in digit_means.items()},
                "low_digit": int(low_d), "low_val": low_v,
                "high_digit": int(high_d), "high_val": high_v,
            }
            result["pcs"].append(pc_info)
            print(f"    PC{pc}: eta²={eta_sq:.3f}  "
                  f"digit {low_d} ({low_v:+.3f}) vs {high_d} ({high_v:+.3f})")

        # Cross-digit cosine similarity matrix
        cos_matrix = np.zeros((10, 10))
        for d1 in range(10):
            for d2 in range(10):
                m1 = digits_np == d1
                m2 = digits_np == d2
                if m1.sum() > 0 and m2.sum() > 0:
                    v1 = torch.from_numpy(res_np[m1].mean(axis=0)).float()
                    v2 = torch.from_numpy(res_np[m2].mean(axis=0)).float()
                    cos_matrix[d1, d2] = float(
                        F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
        result["cross_digit_cosine"] = cos_matrix.tolist()

        # Per-digit residual norms
        digit_norms = {}
        for d in range(10):
            mask = digits_np == d
            if mask.sum() > 0:
                digit_norms[str(d)] = float(per_image_res[mask].norm(dim=-1).mean())
        result["digit_norms"] = digit_norms

        return result

    orig_digit = digit_analysis(orig_img_res, orig_digits, "orig FM")
    fresh_digit = digit_analysis(fresh_img_res, fresh_digits, "fresh FM")

    # Compare cross-digit similarity matrices
    orig_cos = np.array(orig_digit["cross_digit_cosine"])
    fresh_cos = np.array(fresh_digit["cross_digit_cosine"])
    triu_idx = np.triu_indices(10, k=1)
    cos_matrix_corr = float(np.corrcoef(
        orig_cos[triu_idx], fresh_cos[triu_idx])[0, 1])
    print(f"\n  Cross-digit similarity matrix correlation: {cos_matrix_corr:.4f}")

    # Most similar/dissimilar pairs comparison
    print(f"\n  Top cross-digit pairs:")
    for label_str, cos_mat in [("orig", orig_cos), ("fresh", fresh_cos)]:
        pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                pairs.append((i, j, cos_mat[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top3 = pairs[:3]
        bot3 = pairs[-3:]
        print(f"    [{label_str}] Most similar: "
              + ", ".join(f"{a}↔{b} ({c:+.2f})" for a, b, c in top3))
        print(f"    [{label_str}] Most different: "
              + ", ".join(f"{a}↔{b} ({c:+.2f})" for a, b, c in bot3))

    # PC rotation summary
    print(f"\n  PC rotation (which digits discriminated):")
    print(f"  {'PC':>4s} {'Orig':>20s} {'eta²':>6s}  "
          f"{'Fresh':>20s} {'eta²':>6s}")
    for pc in range(5):
        o = orig_digit["pcs"][pc]
        f = fresh_digit["pcs"][pc]
        o_str = f"{o['low_digit']}↔{o['high_digit']}"
        f_str = f"{f['low_digit']}↔{f['high_digit']}"
        print(f"  {pc:>4d} {o_str:>20s} {o['eta_squared']:>6.3f}  "
              f"{f_str:>20s} {f['eta_squared']:>6.3f}")

    # --- 3d: Behavioral conditioning ---
    print(f"\n--- 3d: Behavioral conditioning ---")

    orig_norms = orig_img_res.norm(dim=-1).numpy()
    fresh_norms = fresh_img_res.norm(dim=-1).numpy()
    norm_corr = float(np.corrcoef(orig_norms, fresh_norms)[0, 1])
    print(f"  Per-image residual norm correlation: {norm_corr:.4f}")
    print(f"  Mean orig: {orig_norms.mean():.4f}, fresh: {fresh_norms.mean():.4f}")

    print(f"\n  Per-digit residual norms:")
    print(f"  {'Digit':>6s} {'Orig':>8s} {'Fresh':>8s} {'Δ':>8s}")
    for d in range(10):
        on = float(orig_digit["digit_norms"].get(str(d), 0))
        fn = float(fresh_digit["digit_norms"].get(str(d), 0))
        print(f"  {d:>6d} {on:>8.4f} {fn:>8.4f} {fn - on:>+8.4f}")

    # --- 3e: Robustness ---
    print(f"\n--- 3e: Robustness ---")
    perturbation_strengths = [0.5, 1.0, 2.0]

    def measure_robustness(model, label):
        model.eval()
        results = {}
        with torch.no_grad():
            for eps in perturbation_strengths:
                deltas = []
                for bi in range(min(probe_batches, 20)):
                    pidx = probe_indices[bi]
                    vim = test_images[pidx].to(device)
                    vlb = test_labels[pidx].to(device)

                    _, base_loss = model(vim, vlb)

                    torch.manual_seed(seed + bi + 3000)
                    noise = torch.randn(
                        batch_size, n_positions, n_embd, device=device) * eps
                    _, pert_loss = model(
                        vim, vlb,
                        perturbation=(inject_after_block, noise),
                    )
                    deltas.append(pert_loss.item() - base_loss.item())

                mean_delta = float(np.mean(deltas))
                results[str(eps)] = mean_delta
                print(f"  [{label:>12s}] eps={eps:.1f}: Δloss={mean_delta:+.4f}")

        return results

    rob_ol = measure_robustness(ol_model, "OL")
    rob_cl = measure_robustness(cl_model, "CL (no inj)")
    rob_dist = measure_robustness(student, "Distilled")

    ol_ref = rob_ol.get("1.0", 0)
    if ol_ref > 0:
        print(f"\n  Robustness ratios (vs OL) at eps=1.0:")
        print(f"    CL (no inj): {rob_cl.get('1.0', 0) / ol_ref:.3f}")
        print(f"    Distilled:   {rob_dist.get('1.0', 0) / ol_ref:.3f}")

    # --- 3f: Self-knowledge probes ---
    print(f"\n--- 3f: Self-knowledge probes ---")

    n_samples = dist_acts[layer_keys[0]].shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    def vector_probe(X, T_target):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = T_target.numpy() if isinstance(T_target, torch.Tensor) else T_target
        T_np = T_np.astype(np.float32)

        mu = X_np[train_idx].mean(axis=0, keepdims=True)
        sd = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu) / sd

        probe = nn.Linear(X_np.shape[1], T_np.shape[1]).to(device)
        opt_p = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ttr = torch.from_numpy(T_np[train_idx]).float().to(device)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            si = torch.randint(n_train, (bs,), device=device)
            p = probe(Xtr[si])
            ploss = F.mse_loss(p, Ttr[si])
            opt_p.zero_grad()
            ploss.backward()
            opt_p.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).cpu().numpy()
        Tte = T_np[test_idx]
        mse = ((pred - Tte) ** 2).mean()
        var = Tte.var()
        return float(1.0 - mse / var) if var > 0 else 0.0

    # Probe distilled vs OL for fresh FM residual
    print(f"  Probing for fresh FM residual:")
    print(f"  {'Layer':>12s} {'OL R²':>8s} {'Dist R²':>9s} {'Δ':>7s}")
    sk_probes = {}
    for lk in layer_keys:
        r2_ol = vector_probe(ol_acts[lk], fresh_res)
        r2_dist = vector_probe(dist_acts[lk], fresh_res)
        delta = r2_dist - r2_ol
        sk_probes[lk] = {"ol": r2_ol, "distilled": r2_dist, "delta": delta}
        print(f"  {lk:>12s} {r2_ol:>8.4f} {r2_dist:>9.4f} {delta:>+7.4f}")

    # --- 3g: Internalization probes ---
    print(f"\n--- 3g: Internalization probes ---")

    # Test 1: Inter-layer self-predictability (post_block_i -> post_block3)
    print(f"\n  Test 1: Inter-layer self-predictability:")
    print(f"  {'Source':>15s} {'OL R²':>8s} {'Dist R²':>9s} {'Δ':>7s}")
    test1 = {}
    for src_key in layer_keys[:-1]:
        r2_ol = vector_probe(ol_acts[src_key], ol_acts["post_block3"])
        r2_dist = vector_probe(dist_acts[src_key], dist_acts["post_block3"])
        delta = r2_dist - r2_ol
        test1[src_key] = {"ol": r2_ol, "distilled": r2_dist, "delta": delta}
        print(f"  {src_key:>15s} {r2_ol:>8.4f} {r2_dist:>9.4f} {delta:>+7.4f}")

    # Test 2: Old FM prediction accessibility
    print(f"\n  Test 2: Old FM prediction accessibility:")
    print(f"  {'Layer':>15s} {'OL R²':>8s} {'Dist R²':>9s} {'Δ':>7s}")

    def get_fm_preds(model, fm):
        model.eval()
        fm.eval()
        preds = []
        with torch.no_grad():
            for pidx in probe_indices:
                vim = test_images[pidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                preds.append(fm(src).reshape(-1, n_embd).cpu())
        return torch.cat(preds)

    fm_pred_ol = get_fm_preds(ol_model, orig_fm)
    fm_pred_dist = get_fm_preds(student, orig_fm)

    test2 = {}
    for lk in layer_keys:
        r2_ol = vector_probe(ol_acts[lk], fm_pred_ol)
        r2_dist = vector_probe(dist_acts[lk], fm_pred_dist)
        delta = r2_dist - r2_ol
        test2[lk] = {"ol": r2_ol, "distilled": r2_dist, "delta": delta}
        print(f"  {lk:>15s} {r2_ol:>8.4f} {r2_dist:>9.4f} {delta:>+7.4f}")

    # Test 2b: Cross-model control
    print(f"\n  Test 2b: Cross-model control (probe each for FM pred on OL):")
    print(f"  {'Layer':>15s} {'OL R²':>8s} {'Dist R²':>9s} {'Δ':>7s}")
    test2b = {}
    for lk in layer_keys:
        r2_ol = vector_probe(ol_acts[lk], fm_pred_ol)
        r2_dist = vector_probe(dist_acts[lk], fm_pred_ol)
        delta = r2_dist - r2_ol
        test2b[lk] = {"ol": r2_ol, "distilled": r2_dist, "delta": delta}
        print(f"  {lk:>15s} {r2_ol:>8.4f} {r2_dist:>9.4f} {delta:>+7.4f}")

    # Test 3: How well does old FM predict each model?
    print(f"\n  Test 3: Old FM prediction quality:")
    test3 = {}
    for label, model_ref, acts in [
        ("OL", ol_model, ol_acts),
        ("CL", cl_model, cl_acts),
        ("Distilled", student, dist_acts),
    ]:
        preds = get_fm_preds(model_ref, orig_fm)
        actual = acts["post_block3"]
        cos = float(F.cosine_similarity(preds, actual, dim=-1).mean())
        mse = float(F.mse_loss(preds, actual))
        res_norm = float((actual - preds).norm(dim=-1).mean())
        test3[label] = {"cosine": cos, "mse": mse, "residual_norm": res_norm}
        print(f"  [{label:>10s}] cos={cos:.4f}  mse={mse:.5f}  "
              f"res_norm={res_norm:.4f}")

    # =============================================
    # Save results
    # =============================================
    result = {
        "config": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "patch_size": patch_size,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "distill_steps": distill_steps, "distill_lr": distill_lr,
            "distill_alpha": distill_alpha, "retrain_steps": retrain_steps,
            "seed": seed, "fresh_fm_seed": fresh_fm_seed,
        },
        "pre_distillation": {
            "ol_loss": ol_loss, "ol_acc": ol_acc,
            "cl_inj_loss": cl_inj_loss, "cl_inj_acc": cl_inj_acc,
            "cl_noinj_loss": cl_noinj_loss, "cl_noinj_acc": cl_noinj_acc,
            "dependency_gap": dependency_pre,
        },
        "post_distillation": {
            "distilled_loss": dist_loss, "distilled_acc": dist_acc,
            "dependency_gap": dependency_post,
            "gap_closed_frac": gap_closed,
        },
        "fresh_fm": {
            "final_cosine": fm_history["val_cosine"][-1][1]
            if fm_history["val_cosine"] else None,
            "final_mse": fm_history["val_mse"][-1][1]
            if fm_history["val_mse"] else None,
        },
        "eigenspectrum": {
            "orig_on_cl": orig_stats,
            "orig_on_distilled": orig_on_dist_stats,
            "fresh_on_distilled": fresh_stats,
        },
        "direction_overlap": direction_overlap,
        "pc_cosines": pc_cosines,
        "digit_analysis": {
            "orig": orig_digit,
            "fresh": fresh_digit,
            "cross_digit_matrix_corr": cos_matrix_corr,
        },
        "behavioral_conditioning": {
            "norm_correlation": norm_corr,
            "orig_mean_norm": float(orig_norms.mean()),
            "fresh_mean_norm": float(fresh_norms.mean()),
        },
        "robustness": {
            "ol": rob_ol, "cl_noinj": rob_cl, "distilled": rob_dist,
        },
        "self_knowledge_probes": sk_probes,
        "internalization": {
            "test1_self_predictability": test1,
            "test2_fm_accessibility": test2,
            "test2b_cross_model": test2b,
            "test3_fm_prediction_quality": test3,
        },
        "distill_history": distill_history,
        "fm_history": fm_history,
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Dependency gap: {dependency_pre:+.4f} -> {dependency_post:+.4f} "
          f"({gap_closed:.1%} closed)")
    print(f"  Distilled: loss={dist_loss:.4f} acc={dist_acc:.4f} "
          f"(OL: {ol_loss:.4f}/{ol_acc:.4f})")
    if fm_history["val_cosine"]:
        print(f"  Fresh FM cosine: {fm_history['val_cosine'][-1][1]:.4f}")
    print(f"\n  Eigenspectrum:")
    print(f"    Orig FM on CL:       eff_rank={orig_stats['eff_rank']:.1f}, "
          f"top1={orig_stats['top1']*100:.1f}%")
    print(f"    Orig FM on distilled: eff_rank={orig_on_dist_stats['eff_rank']:.1f}, "
          f"top1={orig_on_dist_stats['top1']*100:.1f}%")
    print(f"    Fresh FM on distilled: eff_rank={fresh_stats['eff_rank']:.1f}, "
          f"top1={fresh_stats['top1']*100:.1f}%")
    print(f"\n  Innovation direction overlap:")
    print(f"    PC cosines: {', '.join(f'{c:.3f}' for c in pc_cosines[:5])}")
    print(f"    Cross-digit matrix corr: {cos_matrix_corr:.4f}")
    print(f"    Norm correlation: {norm_corr:.4f}")
    if ol_ref > 0:
        print(f"\n  Robustness ratios (eps=1.0):")
        print(f"    CL (no inj): {rob_cl.get('1.0', 0) / ol_ref:.3f}")
        print(f"    Distilled:   {rob_dist.get('1.0', 0) / ol_ref:.3f}")
    print(f"\n  Internalization (Test 3 - old FM prediction quality):")
    for label in ["OL", "CL", "Distilled"]:
        t3 = test3[label]
        print(f"    [{label}] cos={t3['cosine']:.4f} res_norm={t3['residual_norm']:.4f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    distill_steps: int = 2000,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    retrain_steps: int = 5000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
):
    result = a2a_mnist_distillation.remote(
        distill_steps=distill_steps,
        distill_lr=distill_lr,
        distill_alpha=distill_alpha,
        retrain_steps=retrain_steps,
        predict_from=predict_from,
        predict_to=predict_to,
        inject_after_block=inject_after_block,
    )
    print("\nMNIST distillation complete.")
    post = result["post_distillation"]
    print(f"  Gap closed: {post['gap_closed_frac']:.1%}")
    print(f"  Distilled: loss={post['distilled_loss']:.4f} "
          f"acc={post['distilled_acc']:.4f}")

    eigen = result["eigenspectrum"]
    print(f"  Eigenspectrum shift:")
    print(f"    Orig: eff_rank={eigen['orig_on_cl']['eff_rank']:.1f}, "
          f"top1={eigen['orig_on_cl']['top1']*100:.1f}%")
    print(f"    Fresh: eff_rank={eigen['fresh_on_distilled']['eff_rank']:.1f}, "
          f"top1={eigen['fresh_on_distilled']['top1']*100:.1f}%")

    t3 = result["internalization"]["test3_fm_prediction_quality"]
    print(f"  Old FM prediction (cos): "
          f"OL={t3['OL']['cosine']:.4f}, "
          f"CL={t3['CL']['cosine']:.4f}, "
          f"Dist={t3['Distilled']['cosine']:.4f}")
