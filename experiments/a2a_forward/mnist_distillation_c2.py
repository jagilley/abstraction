"""MNIST two-cycle wake-sleep diagnostic.

Loads cycle-1 outputs (distilled model + fresh FM) and runs cycle 2:
  Wake:  CL co-train the distilled model with the cycle-1 fresh FM (5K steps)
  Sleep: Distill to absorb new dependency (2K steps)
  Re-point: Train cycle-2 fresh FM on cycle-2 distilled model (5K steps)

Then compares the trajectory across cycles:
  - Does new dependency form during wake?
  - Does eta² stay class-blind or develop new structure?
  - Does the eigenspectrum keep evolving?
  - Does Test 3 cosine keep improving?
  - Does robustness accumulate?
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def a2a_mnist_distillation_c2(
    # Architecture (must match original)
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
    # Wake params
    wake_steps: int = 5000,
    wake_lr: float = 3e-4,
    wake_fwd_lr: float = 1e-3,
    # Sleep params
    distill_steps: int = 2000,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    # Re-point params
    retrain_steps: int = 5000,
    retrain_fwd_lr: float = 1e-3,
    c2_fresh_fm_seed: int = 237,
    # Shared
    eval_interval: int = 100,
    n_eval_batches: int = 5,
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

    print(f"MNIST CYCLE-2 WAKE-SLEEP DIAGNOSTIC on {device}")

    # --- Helpers ---
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

    # --- Load MNIST ---
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # --- Batch indices (different offsets from cycle 1) ---
    max_steps = max(wake_steps, retrain_steps)
    n_evals = max_steps // eval_interval + 2

    train_gen = torch.Generator().manual_seed(seed + 400)
    eval_gen = torch.Generator().manual_seed(seed + 401)
    probe_gen = torch.Generator().manual_seed(seed + 402)

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
    orig_root = f"{DATA_DIR}/a2a_forward/mnist/{model_tag}/{gap_tag}"
    c1_root = f"{DATA_DIR}/a2a_forward/mnist_distillation/{model_tag}/{gap_tag}"
    save_root = f"{DATA_DIR}/a2a_forward/mnist_distillation/{model_tag}/{gap_tag}/cycle2"

    print(f"\nLoading checkpoints...")

    # OL model (fixed baseline)
    ol_model = make_vit()
    ol_model.load_state_dict(
        torch.load(f"{orig_root}/ol_model.pt", map_location=device, weights_only=True))

    # Original FM (from first CL training — for Test 3 trajectory)
    orig_fm = make_fm()
    orig_fm.load_state_dict(
        torch.load(f"{orig_root}/cl_fwd.pt", map_location=device, weights_only=True))

    # Cycle-1 distilled model (starting point for cycle-2 wake)
    c1_distilled = make_vit()
    c1_distilled.load_state_dict(
        torch.load(f"{c1_root}/distilled_model.pt", map_location=device, weights_only=True))

    # Cycle-1 fresh FM (starting point for cycle-2 wake FM)
    c1_fresh_fm = make_fm()
    c1_fresh_fm.load_state_dict(
        torch.load(f"{c1_root}/fresh_fm.pt", map_location=device, weights_only=True))

    print(f"  Loaded OL, orig FM, cycle-1 distilled, cycle-1 fresh FM")

    # --- Eval helper ---
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

    # --- Pre-wake baselines ---
    print(f"\n{'='*60}")
    print(f"  PRE-WAKE BASELINES")
    print(f"{'='*60}")

    ol_loss, ol_acc = eval_model(ol_model)
    c1d_loss, c1d_acc = eval_model(c1_distilled)
    print(f"  OL:              loss={ol_loss:.4f}  acc={ol_acc:.4f}")
    print(f"  Cycle-1 distilled: loss={c1d_loss:.4f}  acc={c1d_acc:.4f}")

    # =============================================
    # CYCLE-2 WAKE: CL co-training
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CYCLE-2 WAKE: CL CO-TRAINING ({wake_steps} steps)")
    print(f"{'='*60}")

    # Start from cycle-1 distilled weights
    c2_model = make_vit()
    c2_model.load_state_dict(c1_distilled.state_dict())

    # Start FM from cycle-1 fresh FM weights (co-trains)
    c2_fm = make_fm()
    c2_fm.load_state_dict(c1_fresh_fm.state_dict())

    # New zero-init gate
    c2_gate = CerebellarGate(n_embd).to(device)

    main_params = list(c2_model.parameters()) + list(c2_gate.parameters())
    opt_main = torch.optim.AdamW(main_params, lr=wake_lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(c2_fm.parameters(), lr=wake_fwd_lr, weight_decay=0.01)

    wake_history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_loss_no_inj": [], "val_acc_no_inj": [],
        "fwd_mse": [], "val_cosine": [], "gate_norm": [],
    }

    eval_idx = 0
    for step in range(wake_steps):
        c2_model.train()
        c2_fm.train()
        c2_gate.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        fwd_pred_cache = {}

        def cerebellar_fn(act):
            fwd_pred = c2_fm(act.detach())
            fwd_pred_cache["pred"] = fwd_pred
            return c2_gate(fwd_pred.detach())

        logits, cls_loss, intermediates = c2_model(
            images, labels, return_intermediates=True,
            cerebellar_fn=cerebellar_fn,
            cerebellar_input_block=cerebellar_input_block,
            cerebellar_inject_block=inject_after_block,
        )
        target = intermediates[predict_to].detach()
        fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)

        opt_main.zero_grad()
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, 1.0)
        opt_main.step()

        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(c2_fm.parameters(), 1.0)
        opt_fwd.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % eval_interval == 0 or step == wake_steps - 1:
            c2_model.eval()
            c2_fm.eval()
            c2_gate.eval()

            v_loss, v_acc = 0.0, 0.0
            v_loss_noinj, v_acc_noinj = 0.0, 0.0
            v_cos = 0.0

            with torch.no_grad():
                for bi in range(n_eval_batches):
                    eidx = eval_indices[eval_idx][bi]
                    vim = test_images[eidx].to(device)
                    vlb = test_labels[eidx].to(device)

                    # With injection
                    def eval_cb(act):
                        return c2_gate(c2_fm(act))
                    vlog, vl, vi = c2_model(
                        vim, vlb, return_intermediates=True,
                        cerebellar_fn=eval_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    v_loss += vl.item()
                    v_acc += (vlog.argmax(-1) == vlb).float().mean().item()

                    # Without injection
                    vlog2, vl2 = c2_model(vim, vlb)
                    v_loss_noinj += vl2.item()
                    v_acc_noinj += (vlog2.argmax(-1) == vlb).float().mean().item()

                    # FM quality
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = c2_fm(src)
                    v_cos += F.cosine_similarity(pred, tgt, dim=-1).mean().item()

                n = n_eval_batches
                v_loss /= n
                v_acc /= n
                v_loss_noinj /= n
                v_acc_noinj /= n
                v_cos /= n

                dep = v_loss_noinj - v_loss
                gn = c2_gate.injection_norm()

                wake_history["val_loss"].append((step, v_loss))
                wake_history["val_acc"].append((step, v_acc))
                wake_history["val_loss_no_inj"].append((step, v_loss_noinj))
                wake_history["val_acc_no_inj"].append((step, v_acc_noinj))
                wake_history["val_cosine"].append((step, v_cos))
                wake_history["gate_norm"].append((step, gn))
                wake_history["train_loss"].append((step, cls_loss.item()))
                wake_history["train_acc"].append((step, acc))
                wake_history["fwd_mse"].append((step, fwd_loss.item()))

                print(f"  [WAKE] step {step:5d}: "
                      f"loss={v_loss:.4f} acc={v_acc:.4f} "
                      f"dep={dep:+.4f} gate={gn:.3f} "
                      f"cos={v_cos:.4f}")

            eval_idx += 1

    # Post-wake measurements
    c2_wake_loss, c2_wake_acc = eval_model(c2_model, c2_fm, c2_gate, use_injection=True)
    c2_wake_noinj_loss, c2_wake_noinj_acc = eval_model(c2_model)
    c2_dep_gap = c2_wake_noinj_loss - c2_wake_loss

    print(f"\n  POST-WAKE:")
    print(f"    With injection:    loss={c2_wake_loss:.4f}  acc={c2_wake_acc:.4f}")
    print(f"    Without injection: loss={c2_wake_noinj_loss:.4f}  acc={c2_wake_noinj_acc:.4f}")
    print(f"    Dependency gap:    {c2_dep_gap:+.4f}")
    print(f"    Gate norm:         {c2_gate.injection_norm():.4f}")

    # Save wake checkpoint
    os.makedirs(save_root, exist_ok=True)
    torch.save(c2_model.state_dict(), os.path.join(save_root, "wake_model.pt"))
    torch.save(c2_fm.state_dict(), os.path.join(save_root, "wake_fm.pt"))
    torch.save(c2_gate.state_dict(), os.path.join(save_root, "wake_gate.pt"))
    volume.commit()

    # =============================================
    # CYCLE-2 SLEEP: DISTILLATION
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CYCLE-2 SLEEP: DISTILLATION ({distill_steps} steps)")
    print(f"{'='*60}")

    # Freeze teacher (cycle-2 wake model + FM + gate)
    teacher = make_vit()
    teacher.load_state_dict(c2_model.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_fm = make_fm()
    teacher_fm.load_state_dict(c2_fm.state_dict())
    teacher_fm.eval()
    for p in teacher_fm.parameters():
        p.requires_grad = False

    teacher_gate = CerebellarGate(n_embd).to(device)
    teacher_gate.load_state_dict(c2_gate.state_dict())
    teacher_gate.eval()
    for p in teacher_gate.parameters():
        p.requires_grad = False

    # Student: cycle-2 wake weights, no injection
    student = make_vit()
    student.load_state_dict(c2_model.state_dict())

    opt = torch.optim.AdamW(student.parameters(), lr=distill_lr, weight_decay=0.01)

    distill_history = {
        "val_loss": [], "val_acc": [], "val_kl": [],
    }

    distill_eval_idx = 0
    for step in range(distill_steps):
        student.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        with torch.no_grad():
            def t_cb(act):
                return teacher_gate(teacher_fm(act))
            teacher_logits, _, _ = teacher(
                images, labels, return_intermediates=True,
                cerebellar_fn=t_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

        student_logits, ce_loss = student(images, labels)
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
                for eidx in eval_indices[distill_eval_idx]:
                    vim = test_images[eidx].to(device)
                    vlb = test_labels[eidx].to(device)
                    s_logits, s_loss = student(vim, vlb)
                    v_loss += s_loss.item()
                    v_acc += (s_logits.argmax(-1) == vlb).float().mean().item()

                    def eval_t_cb(act):
                        return teacher_gate(teacher_fm(act))
                    t_logits, _, _ = teacher(
                        vim, vlb, return_intermediates=True,
                        cerebellar_fn=eval_t_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    v_kl += F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        F.softmax(t_logits, dim=-1),
                        reduction="batchmean",
                    ).item()

                n = n_eval_batches
                v_loss /= n
                v_acc /= n
                v_kl /= n
                distill_history["val_loss"].append((step, v_loss))
                distill_history["val_acc"].append((step, v_acc))
                distill_history["val_kl"].append((step, v_kl))

                print(f"  [SLEEP] step {step:5d}: "
                      f"val_loss={v_loss:.4f} acc={v_acc:.4f} "
                      f"KL={v_kl:.5f}")

            distill_eval_idx += 1

    c2d_loss, c2d_acc = eval_model(student)
    c2_dep_post = c2d_loss - c2_wake_loss
    c2_gap_closed = (1.0 - c2_dep_post / c2_dep_gap) if c2_dep_gap != 0 else 0

    print(f"\n  POST-SLEEP:")
    print(f"    Cycle-2 distilled: loss={c2d_loss:.4f}  acc={c2d_acc:.4f}")
    print(f"    Dep gap: {c2_dep_gap:+.4f} -> {c2_dep_post:+.4f} "
          f"({c2_gap_closed:.1%} closed)")

    torch.save(student.state_dict(), os.path.join(save_root, "distilled_model.pt"))
    volume.commit()

    del teacher, teacher_fm, teacher_gate
    torch.cuda.empty_cache()

    # =============================================
    # CYCLE-2 RE-POINT: FRESH FM
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CYCLE-2 RE-POINT: FRESH FM ({retrain_steps} steps, seed={c2_fresh_fm_seed})")
    print(f"{'='*60}")

    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    torch.manual_seed(c2_fresh_fm_seed)
    c2_fresh_fm = make_fm()

    opt_fm = torch.optim.AdamW(c2_fresh_fm.parameters(), lr=retrain_fwd_lr,
                                weight_decay=0.01)
    fm_history = {"val_mse": [], "val_cosine": []}

    fm_eval_idx = 0
    for step in range(retrain_steps):
        c2_fresh_fm.train()
        idx = train_indices[step]
        images = train_images[idx].to(device)

        with torch.no_grad():
            _, _, intermediates = student(images, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]

        predicted = c2_fresh_fm(source)
        fwd_loss = F.mse_loss(predicted, target)

        opt_fm.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(c2_fresh_fm.parameters(), 1.0)
        opt_fm.step()

        if step % eval_interval == 0 or step == retrain_steps - 1:
            c2_fresh_fm.eval()
            with torch.no_grad():
                vm, vc = 0.0, 0.0
                for eidx in eval_indices[fm_eval_idx]:
                    vim = test_images[eidx].to(device)
                    _, _, vi = student(vim, return_intermediates=True)
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = c2_fresh_fm(src)
                    vm += F.mse_loss(pred, tgt).item()
                    vc += F.cosine_similarity(pred, tgt, dim=-1).mean().item()

                n = n_eval_batches
                vm /= n
                vc /= n
                fm_history["val_mse"].append((step, vm))
                fm_history["val_cosine"].append((step, vc))

                if step % 500 == 0 or step == retrain_steps - 1:
                    print(f"  [RE-POINT] step {step:5d}: "
                          f"val_mse={vm:.5f} val_cos={vc:.4f}")
            fm_eval_idx += 1

    torch.save(c2_fresh_fm.state_dict(), os.path.join(save_root, "fresh_fm.pt"))
    volume.commit()

    for p in student.parameters():
        p.requires_grad = True

    # =============================================
    # CROSS-CYCLE ANALYSIS
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CROSS-CYCLE ANALYSIS")
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
        print(f"  [{label}] mean res norm: {all_res.norm(dim=-1).mean():.4f}")
        return all_res, per_image_res, per_image_digits, layer_acts

    print("\nCollecting residuals...")
    # Cycle-1 fresh FM on cycle-1 distilled (reference)
    c1_res, c1_img_res, c1_digits, _ = collect_residuals(
        c1_distilled, c1_fresh_fm, "c1 fresh on c1 distilled")
    # Cycle-2 fresh FM on cycle-2 distilled (new)
    c2_res, c2_img_res, c2_digits, c2d_acts = collect_residuals(
        student, c2_fresh_fm, "c2 fresh on c2 distilled")
    # Cycle-1 fresh FM on cycle-2 distilled (cross-cycle)
    c1_on_c2_res, c1_on_c2_img_res, _, _ = collect_residuals(
        student, c1_fresh_fm, "c1 fresh on c2 distilled")
    # OL acts (for probes)
    _, _, _, ol_acts = collect_residuals(
        ol_model, orig_fm, "orig FM on OL")

    # --- 4a: Eigenspectrum trajectory ---
    print(f"\n--- 4a: Eigenspectrum trajectory ---")

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
            "rank_50": rank_50,
            "rank_90": rank_90,
            "mean_norm": float(residuals.norm(dim=-1).mean()),
        }
        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"top1={fracs[0]*100:.1f}%, top5={fracs[:5].sum()*100:.1f}%, "
              f"rank50={rank_50}, rank90={rank_90}, norm={stats['mean_norm']:.3f}")
        return stats

    c1_stats = compute_eigen_stats(c1_res, "c1 fresh FM")
    c2_stats = compute_eigen_stats(c2_res, "c2 fresh FM")
    c1_on_c2_stats = compute_eigen_stats(c1_on_c2_res, "c1 FM on c2 distilled")

    # Reference values from cycle-1 results (orig FM on CL model)
    print(f"\n  Trajectory (from cycle-1 results.json + new):")
    print(f"    Cycle 0 (orig FM on CL):     eff_rank=12.4, top1=24.5%")
    print(f"    Cycle 1 (c1 fresh on c1d):   eff_rank={c1_stats['eff_rank']:.1f}, "
          f"top1={c1_stats['top1']*100:.1f}%")
    print(f"    Cycle 2 (c2 fresh on c2d):   eff_rank={c2_stats['eff_rank']:.1f}, "
          f"top1={c2_stats['top1']*100:.1f}%")

    # --- 4b: Digit-discriminative trajectory ---
    print(f"\n--- 4b: Digit-discriminative trajectory ---")

    def digit_eta_squared(per_image_res, digits, label, n_pcs=5):
        res_np = per_image_res.numpy().astype(np.float64)
        res_c = res_np - res_np.mean(axis=0, keepdims=True)
        digits_np = digits.numpy()

        cov = (res_c.T @ res_c) / (res_c.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        vecs = evecs[:, idx[:n_pcs]]
        projections = res_c @ vecs

        pcs = []
        print(f"  [{label}]")
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

            pcs.append({
                "eta_squared": eta_sq,
                "low_digit": int(low_d), "high_digit": int(high_d),
                "digit_means": {str(k): v for k, v in digit_means.items()},
            })
            print(f"    PC{pc}: eta²={eta_sq:.3f}  "
                  f"{low_d}↔{high_d}")

        # Per-digit norms
        digit_norms = {}
        for d in range(10):
            mask = digits_np == d
            if mask.sum() > 0:
                digit_norms[str(d)] = float(per_image_res[mask].norm(dim=-1).mean())

        # Cross-digit cosine matrix
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

        return {
            "pcs": pcs,
            "digit_norms": digit_norms,
            "cross_digit_cosine": cos_matrix.tolist(),
        }

    c1_digit = digit_eta_squared(c1_img_res, c1_digits, "c1 fresh FM")
    c2_digit = digit_eta_squared(c2_img_res, c2_digits, "c2 fresh FM")

    print(f"\n  Eta² trajectory (mean of top-5 PCs):")
    print(f"    Cycle 0 (orig FM): ~0.39 (from cycle-1 results)")
    c1_mean_eta = np.mean([pc["eta_squared"] for pc in c1_digit["pcs"]])
    c2_mean_eta = np.mean([pc["eta_squared"] for pc in c2_digit["pcs"]])
    print(f"    Cycle 1 (c1 fresh): {c1_mean_eta:.3f}")
    print(f"    Cycle 2 (c2 fresh): {c2_mean_eta:.3f}")

    # --- 4c: Cross-cycle innovation ---
    print(f"\n--- 4c: Cross-cycle innovation ---")

    # Per-image norm correlation between c1 and c2 fresh FM residuals
    c1_norms = c1_img_res.norm(dim=-1).numpy()
    c2_norms = c2_img_res.norm(dim=-1).numpy()
    cross_cycle_norm_corr = float(np.corrcoef(c1_norms, c2_norms)[0, 1])
    print(f"  Norm correlation (c1 fresh ↔ c2 fresh): {cross_cycle_norm_corr:.4f}")
    print(f"  Mean c1 norm: {c1_norms.mean():.4f}, c2: {c2_norms.mean():.4f}")

    # PC cosines between c1 and c2 fresh FM residuals
    def top_k_eigenvecs(residuals, k=10):
        R = residuals.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = evals.argsort()[::-1]
        return evecs[:, idx[:k]]

    c1_vecs = top_k_eigenvecs(c1_res, k=10)
    c2_vecs = top_k_eigenvecs(c2_res, k=10)

    pc_cosines_c1_c2 = []
    for i in range(10):
        cos = float(np.abs(c1_vecs[:, i] @ c2_vecs[:, i]))
        pc_cosines_c1_c2.append(cos)
    print(f"  PC cosines (c1↔c2): "
          f"{', '.join(f'{c:.3f}' for c in pc_cosines_c1_c2[:5])}")

    # Cross-digit matrix correlation (c1 vs c2)
    c1_cos_mat = np.array(c1_digit["cross_digit_cosine"])
    c2_cos_mat = np.array(c2_digit["cross_digit_cosine"])
    triu_idx = np.triu_indices(10, k=1)
    digit_matrix_corr = float(np.corrcoef(
        c1_cos_mat[triu_idx], c2_cos_mat[triu_idx])[0, 1])
    print(f"  Cross-digit matrix correlation (c1↔c2): {digit_matrix_corr:.4f}")

    # --- 4d: Test 3 trajectory (original FM on each model) ---
    print(f"\n--- 4d: Test 3 trajectory ---")

    def fm_prediction_quality(model, fm, label):
        model.eval()
        fm.eval()
        preds = []
        actuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                vim = test_images[pidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                preds.append(pred.reshape(-1, n_embd).cpu())
                actuals.append(tgt.reshape(-1, n_embd).cpu())
        preds = torch.cat(preds)
        actuals = torch.cat(actuals)
        cos = float(F.cosine_similarity(preds, actuals, dim=-1).mean())
        mse = float(F.mse_loss(preds, actuals))
        res_norm = float((actuals - preds).norm(dim=-1).mean())
        print(f"  [{label:>25s}] cos={cos:.4f}  mse={mse:.5f}  "
              f"res_norm={res_norm:.4f}")
        return {"cosine": cos, "mse": mse, "residual_norm": res_norm}

    print(f"  Original FM predicting each model:")
    t3_ol = fm_prediction_quality(ol_model, orig_fm, "OL")
    t3_c1d = fm_prediction_quality(c1_distilled, orig_fm, "Cycle-1 distilled")
    t3_c2d = fm_prediction_quality(student, orig_fm, "Cycle-2 distilled")

    print(f"\n  Cycle-1 fresh FM predicting cycle-2 distilled:")
    t3_c1fm_c2d = fm_prediction_quality(student, c1_fresh_fm, "c1 FM on c2 distilled")

    print(f"\n  Cosine trajectory (orig FM): "
          f"{t3_ol['cosine']:.4f} → {t3_c1d['cosine']:.4f} → {t3_c2d['cosine']:.4f}")

    # --- 4e: Robustness trajectory ---
    print(f"\n--- 4e: Robustness trajectory ---")
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
                    torch.manual_seed(seed + bi + 5000)
                    noise = torch.randn(
                        batch_size, n_positions, n_embd, device=device) * eps
                    _, pert_loss = model(
                        vim, vlb,
                        perturbation=(inject_after_block, noise),
                    )
                    deltas.append(pert_loss.item() - base_loss.item())
                mean_delta = float(np.mean(deltas))
                results[str(eps)] = mean_delta
                print(f"  [{label:>20s}] eps={eps:.1f}: Δloss={mean_delta:+.4f}")
        return results

    rob_ol = measure_robustness(ol_model, "OL")
    rob_c1d = measure_robustness(c1_distilled, "Cycle-1 distilled")
    rob_c2d = measure_robustness(student, "Cycle-2 distilled")

    ol_ref = rob_ol.get("1.0", 0)
    if ol_ref > 0:
        print(f"\n  Robustness ratios (eps=1.0):")
        print(f"    Cycle-1 distilled: {rob_c1d.get('1.0', 0) / ol_ref:.3f}")
        print(f"    Cycle-2 distilled: {rob_c2d.get('1.0', 0) / ol_ref:.3f}")

    # --- 4f: Self-knowledge probes ---
    print(f"\n--- 4f: Self-knowledge probes ---")

    n_samples = c2d_acts[layer_keys[0]].shape[0]
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

    print(f"  Probing c2 distilled vs OL for c2 fresh FM residual:")
    print(f"  {'Layer':>12s} {'OL R²':>8s} {'C2D R²':>9s} {'Δ':>7s}")
    sk_probes = {}
    for lk in layer_keys:
        r2_ol = vector_probe(ol_acts[lk], c2_res)
        r2_c2d = vector_probe(c2d_acts[lk], c2_res)
        delta = r2_c2d - r2_ol
        sk_probes[lk] = {"ol": r2_ol, "c2_distilled": r2_c2d, "delta": delta}
        print(f"  {lk:>12s} {r2_ol:>8.4f} {r2_c2d:>9.4f} {delta:>+7.4f}")

    # =============================================
    # SAVE
    # =============================================
    result = {
        "config": {
            "wake_steps": wake_steps, "wake_lr": wake_lr,
            "distill_steps": distill_steps, "distill_lr": distill_lr,
            "retrain_steps": retrain_steps, "c2_fresh_fm_seed": c2_fresh_fm_seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "pre_wake": {
            "ol_loss": ol_loss, "ol_acc": ol_acc,
            "c1d_loss": c1d_loss, "c1d_acc": c1d_acc,
        },
        "wake": {
            "final_loss_inj": c2_wake_loss, "final_acc_inj": c2_wake_acc,
            "final_loss_noinj": c2_wake_noinj_loss,
            "final_acc_noinj": c2_wake_noinj_acc,
            "dependency_gap": c2_dep_gap,
            "gate_norm": c2_gate.injection_norm(),
        },
        "sleep": {
            "c2d_loss": c2d_loss, "c2d_acc": c2d_acc,
            "dependency_post": c2_dep_post,
            "gap_closed": c2_gap_closed,
        },
        "fresh_fm": {
            "final_cosine": fm_history["val_cosine"][-1][1]
            if fm_history["val_cosine"] else None,
        },
        "eigenspectrum": {
            "c1_fresh": c1_stats,
            "c2_fresh": c2_stats,
            "c1_on_c2": c1_on_c2_stats,
        },
        "digit_analysis": {
            "c1": c1_digit,
            "c2": c2_digit,
            "c1_mean_eta": c1_mean_eta,
            "c2_mean_eta": c2_mean_eta,
        },
        "cross_cycle": {
            "norm_correlation": cross_cycle_norm_corr,
            "pc_cosines": pc_cosines_c1_c2,
            "digit_matrix_corr": digit_matrix_corr,
        },
        "test3_trajectory": {
            "ol": t3_ol, "c1_distilled": t3_c1d, "c2_distilled": t3_c2d,
            "c1_fm_on_c2d": t3_c1fm_c2d,
        },
        "robustness": {
            "ol": rob_ol, "c1_distilled": rob_c1d, "c2_distilled": rob_c2d,
        },
        "self_knowledge_probes": sk_probes,
        "wake_history": wake_history,
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

    print(f"\n  === Cycle-2 dynamics ===")
    print(f"  Wake dependency gap:  {c2_dep_gap:+.4f}")
    print(f"  Gate norm:            {c2_gate.injection_norm():.4f}")
    print(f"  Sleep gap closed:     {c2_gap_closed:.1%}")

    print(f"\n  === Accuracy trajectory ===")
    print(f"    OL:              {ol_acc:.4f}")
    print(f"    C1 distilled:    {c1d_acc:.4f}")
    print(f"    C2 distilled:    {c2d_acc:.4f}")

    print(f"\n  === Loss trajectory ===")
    print(f"    OL:              {ol_loss:.4f}")
    print(f"    C1 distilled:    {c1d_loss:.4f}")
    print(f"    C2 distilled:    {c2d_loss:.4f}")

    print(f"\n  === Eigenspectrum trajectory ===")
    print(f"    Cycle 0 (orig FM on CL):   eff_rank=12.4")
    print(f"    Cycle 1 (c1 fresh on c1d): eff_rank={c1_stats['eff_rank']:.1f}")
    print(f"    Cycle 2 (c2 fresh on c2d): eff_rank={c2_stats['eff_rank']:.1f}")

    print(f"\n  === Eta² trajectory (mean top-5 PCs) ===")
    print(f"    Cycle 0 (orig FM):  ~0.39")
    print(f"    Cycle 1 (c1 fresh): {c1_mean_eta:.3f}")
    print(f"    Cycle 2 (c2 fresh): {c2_mean_eta:.3f}")

    print(f"\n  === Test 3 (orig FM cosine) ===")
    print(f"    OL:              {t3_ol['cosine']:.4f}")
    print(f"    C1 distilled:    {t3_c1d['cosine']:.4f}")
    print(f"    C2 distilled:    {t3_c2d['cosine']:.4f}")

    print(f"\n  === Cross-cycle ===")
    print(f"    Norm corr (c1↔c2):  {cross_cycle_norm_corr:.4f}")
    print(f"    PC cosines (c1↔c2): "
          f"{', '.join(f'{c:.3f}' for c in pc_cosines_c1_c2[:5])}")
    print(f"    Digit matrix corr:  {digit_matrix_corr:.4f}")

    if ol_ref > 0:
        print(f"\n  === Robustness (eps=1.0 ratio vs OL) ===")
        print(f"    C1 distilled: {rob_c1d.get('1.0', 0) / ol_ref:.3f}")
        print(f"    C2 distilled: {rob_c2d.get('1.0', 0) / ol_ref:.3f}")

    print(f"\n  === Fresh FM cosine ===")
    if fm_history["val_cosine"]:
        print(f"    C2 fresh FM: {fm_history['val_cosine'][-1][1]:.4f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    wake_steps: int = 5000,
    wake_lr: float = 3e-4,
    distill_steps: int = 2000,
    retrain_steps: int = 5000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
):
    result = a2a_mnist_distillation_c2.remote(
        wake_steps=wake_steps,
        wake_lr=wake_lr,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        predict_from=predict_from,
        predict_to=predict_to,
        inject_after_block=inject_after_block,
    )
    print("\nCycle-2 diagnostic complete.")

    print(f"\n  Wake dependency: {result['wake']['dependency_gap']:+.4f}")
    print(f"  Sleep gap closed: {result['sleep']['gap_closed']:.1%}")

    print(f"\n  Loss trajectory: "
          f"OL={result['pre_wake']['ol_loss']:.4f} → "
          f"C1={result['pre_wake']['c1d_loss']:.4f} → "
          f"C2={result['sleep']['c2d_loss']:.4f}")

    print(f"\n  Eigenspectrum: "
          f"12.4 → {result['eigenspectrum']['c1_fresh']['eff_rank']:.1f} → "
          f"{result['eigenspectrum']['c2_fresh']['eff_rank']:.1f}")

    print(f"\n  Eta² (mean): "
          f"~0.39 → {result['digit_analysis']['c1_mean_eta']:.3f} → "
          f"{result['digit_analysis']['c2_mean_eta']:.3f}")

    t3 = result["test3_trajectory"]
    print(f"\n  Test 3 (orig FM cos): "
          f"{t3['ol']['cosine']:.4f} → "
          f"{t3['c1_distilled']['cosine']:.4f} → "
          f"{t3['c2_distilled']['cosine']:.4f}")
