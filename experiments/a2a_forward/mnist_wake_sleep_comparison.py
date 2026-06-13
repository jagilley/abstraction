"""MNIST multi-cycle wake-sleep comparison.

Three conditions from fresh random init (seed=42), identical data order:

1. Wake-Sleep (N cycles):
   Per cycle: wake_steps CL co-training + distill_steps distillation sleep
   Plus retrain_steps FM re-point per cycle (frozen model, doesn't count)
   Main-model gradient steps: N × (wake_steps + distill_steps)

2. CL Continuous:
   Same total main-model steps of closed-loop co-training

3. OL Continuous:
   Same total main-model steps of open-loop training

Evaluation at cycle boundaries: val loss/acc, robustness, self-knowledge.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_mnist_wake_sleep_comparison(
    # Architecture
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
    # Wake-sleep protocol
    n_cycles: int = 4,
    wake_steps: int = 5000,
    distill_steps: int = 2000,
    retrain_steps: int = 5000,
    # Training params
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    # Eval params
    eval_interval: int = 500,
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

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    print(f"MNIST WAKE-SLEEP COMPARISON on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  Checkpoints at: {checkpoint_steps}")

    # =============================================
    # SETUP
    # =============================================
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # Pre-generate batch indices (shared across conditions)
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)

    max_train_steps = total_main_steps
    max_repoint_steps = n_cycles * retrain_steps
    n_evals = max_train_steps // eval_interval + 10

    train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=train_gen)
        for _ in range(max_train_steps)
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
    repoint_indices = [
        torch.randint(len(train_images), (batch_size,), generator=repoint_gen)
        for _ in range(max_repoint_steps)
    ]

    # Factories
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

    # Initialize shared weights
    torch.manual_seed(seed)
    init_model = make_vit()
    init_fm = make_fm()
    init_model_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
    init_fm_state = {k: v.cpu().clone() for k, v in init_fm.state_dict().items()}
    del init_model, init_fm
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =============================================
    # SHARED HELPERS
    # =============================================
    def eval_model(model, n_batches=None):
        model.eval()
        if n_batches is None:
            n_batches = n_eval_batches
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_batches):
                eidx = eval_indices[0][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)
                logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_batches, total_acc / n_batches

    def eval_model_with_inj(model, fm, gate):
        model.eval()
        fm.eval()
        gate.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)

                def cb(act):
                    return gate(fm(act))

                logits, loss, _ = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def measure_robustness(model, label, n_batches=20):
        model.eval()
        eps_list = [0.5, 1.0, 2.0]
        results = {}
        with torch.no_grad():
            for eps in eps_list:
                deltas = []
                for bi in range(min(probe_batches, n_batches)):
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
        print(f"  [{label:>20s}] "
              + " | ".join(f"ε={e}: Δ={results[str(e)]:+.4f}" for e in eps_list))
        return results

    def train_fresh_fm(model, steps, fm_seed, label):
        """Train a fresh FM on frozen model activations."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(steps):
            fm.train()
            idx = repoint_indices[step % len(repoint_indices)]
            images = train_images[idx].to(device)

            with torch.no_grad():
                _, _, vi = model(images, return_intermediates=True)
                source = vi[predict_from]
                target = vi[predict_to]

            pred = fm(source)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()

        fm.eval()
        cos_total = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                cos_total += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cosine: {cos:.4f}")

        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def self_knowledge_probes(model, fm, label):
        """Train linear probes at each layer to predict FM residual."""
        model.eval()
        fm.eval()

        acts_by_layer = {k: [] for k in layer_keys}
        residuals = []

        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                res = tgt - pred
                residuals.append(res.reshape(-1, n_embd).cpu())
                for k in layer_keys:
                    acts_by_layer[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(residuals)
        for k in layer_keys:
            acts_by_layer[k] = torch.cat(acts_by_layer[k])

        n_total = all_res.shape[0]
        n_train = int(0.8 * n_total)
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
        tr = perm[:n_train]
        te = perm[n_train:]

        r2_by_layer = {}
        for lk in layer_keys:
            X = acts_by_layer[lk]
            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

            X_tr = X[tr].to(device)
            Y_tr = all_res[tr].to(device)
            X_te = X[te].to(device)
            Y_te = all_res[te].to(device)

            bs = min(4096, n_train)
            for _ in range(probe_steps):
                si = torch.randint(n_train, (bs,))
                p = probe(X_tr[si])
                ploss = F.mse_loss(p, Y_tr[si])
                opt.zero_grad()
                ploss.backward()
                opt.step()

            with torch.no_grad():
                pred = probe(X_te)
                mse = F.mse_loss(pred, Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            r2_by_layer[lk] = r2
            del probe, X_tr, Y_tr, X_te, Y_te
            torch.cuda.empty_cache()

        print(f"  [{label}] SK probes: "
              + " | ".join(f"{k}={v:.3f}" for k, v in r2_by_layer.items()))
        return r2_by_layer

    def compute_eigen_stats(model, fm, label):
        """Compute eigenspectrum of FM residual."""
        model.eval()
        fm.eval()
        all_res = []
        per_image_res = []
        per_image_digits = []

        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                digits = test_labels[pidx]
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                res = tgt - pred
                all_res.append(res.reshape(-1, n_embd).cpu())
                per_image_res.append(res.mean(dim=1).cpu())
                per_image_digits.append(digits)

        all_res = torch.cat(all_res)
        per_image_res = torch.cat(per_image_res)
        per_image_digits = torch.cat(per_image_digits)

        R = all_res.numpy().astype(np.float64)
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

        # Digit-discriminative eta² for top-5 PCs
        res_np = per_image_res.numpy().astype(np.float64)
        res_c = res_np - res_np.mean(axis=0, keepdims=True)
        img_cov = (res_c.T @ res_c) / (res_c.shape[0] - 1)
        img_evals, img_evecs = np.linalg.eigh(img_cov)
        idx = img_evals.argsort()[::-1]
        vecs = img_evecs[:, idx[:5]]
        projections = res_c @ vecs
        digits_np = per_image_digits.numpy()

        eta_sqs = []
        for pc in range(5):
            proj = projections[:, pc]
            grand_mean = proj.mean()
            ss_total = ((proj - grand_mean) ** 2).sum()
            ss_between = 0.0
            for d in range(10):
                mask = digits_np == d
                if mask.sum() > 0:
                    dm = proj[mask].mean()
                    ss_between += mask.sum() * (dm - grand_mean) ** 2
            eta_sqs.append(float(ss_between / ss_total) if ss_total > 0 else 0.0)

        mean_eta = float(np.mean(eta_sqs))

        stats = {
            "eff_rank": eff_rank,
            "top1_pc": float(fracs[0]),
            "top5_pcs": float(fracs[:5].sum()),
            "rank_50": rank_50,
            "rank_90": rank_90,
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "eta_squared_top5": eta_sqs,
            "mean_eta_squared": mean_eta,
        }

        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"top1={fracs[0]*100:.1f}%, mean_eta²={mean_eta:.3f}, "
              f"res_norm={stats['mean_res_norm']:.3f}")
        return stats

    # =============================================
    # CONDITION 1: WAKE-SLEEP
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: WAKE-SLEEP ({n_cycles} cycles)")
    print(f"{'='*60}")

    ws_model = make_vit()
    ws_model.load_state_dict(init_model_state)
    ws_fm = make_fm()
    ws_fm.load_state_dict(init_fm_state)

    ws_checkpoints = {}
    ws_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ws_global_step = 0  # main-model gradient step counter

    for cycle in range(n_cycles):
        print(f"\n  --- Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: CL co-training ===
        print(f"  [WAKE] CL co-training ({wake_steps} steps)")
        ws_gate = CerebellarGate(n_embd).to(device)
        main_params = list(ws_model.parameters()) + list(ws_gate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(ws_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            ws_model.train()
            ws_fm.train()
            ws_gate.train()

            idx = train_indices[ws_global_step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            fwd_pred_cache = {}

            def cerebellar_fn(act, _cache=fwd_pred_cache):
                fwd_pred = ws_fm(act.detach())
                _cache["pred"] = fwd_pred
                return ws_gate(fwd_pred.detach())

            logits, cls_loss, intermediates = ws_model(
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
            torch.nn.utils.clip_grad_norm_(ws_fm.parameters(), 1.0)
            opt_fwd.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss, v_acc = eval_model(ws_model)
                ws_history["train_loss"].append((ws_global_step, cls_loss.item()))
                ws_history["train_acc"].append((ws_global_step, acc))
                ws_history["val_loss"].append((ws_global_step, v_loss))
                ws_history["val_acc"].append((ws_global_step, v_acc))
                dep_loss, _ = eval_model_with_inj(ws_model, ws_fm, ws_gate)
                dep = v_loss - dep_loss
                print(f"    step {ws_global_step:5d} (wake {step}): "
                      f"val={v_loss:.4f} acc={v_acc:.4f} "
                      f"dep={dep:+.4f} gate={ws_gate.injection_norm():.3f}")

            ws_global_step += 1

        wake_dep_loss, _ = eval_model_with_inj(ws_model, ws_fm, ws_gate)
        wake_noinj_loss, _ = eval_model(ws_model)
        wake_dep_gap = wake_noinj_loss - wake_dep_loss
        print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}, "
              f"gate norm: {ws_gate.injection_norm():.3f}")

        # === SLEEP: Distillation ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")

        teacher = make_vit()
        teacher.load_state_dict(ws_model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        teacher_fm = make_fm()
        teacher_fm.load_state_dict(ws_fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False

        teacher_gate = CerebellarGate(n_embd).to(device)
        teacher_gate.load_state_dict(ws_gate.state_dict())
        teacher_gate.eval()
        for p in teacher_gate.parameters():
            p.requires_grad = False

        student = make_vit()
        student.load_state_dict(ws_model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()

            idx = train_indices[ws_global_step]
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

            opt_student.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_student.step()

            if step % eval_interval == 0 or step == distill_steps - 1:
                v_loss, v_acc = eval_model(student)
                ws_history["val_loss"].append((ws_global_step, v_loss))
                ws_history["val_acc"].append((ws_global_step, v_acc))
                print(f"    step {ws_global_step:5d} (sleep {step}): "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            ws_global_step += 1

        # Update model to distilled weights
        ws_model.load_state_dict(student.state_dict())

        del teacher, teacher_fm, teacher_gate, student, opt_student
        del ws_gate, opt_main, opt_fwd
        torch.cuda.empty_cache()

        # === RE-POINT: Fresh FM ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, seed={fm_seed})")

        ws_model.eval()
        for p in ws_model.parameters():
            p.requires_grad = False

        torch.manual_seed(fm_seed)
        ws_fm = make_fm()
        opt_fm = torch.optim.AdamW(ws_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(retrain_steps):
            ws_fm.train()
            ridx = repoint_indices[cycle * retrain_steps + step]
            images = train_images[ridx].to(device)

            with torch.no_grad():
                _, _, vi = ws_model(images, return_intermediates=True)
                source = vi[predict_from]
                target_act = vi[predict_to]

            pred = ws_fm(source)
            fwd_loss = F.mse_loss(pred, target_act)
            opt_fm.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(ws_fm.parameters(), 1.0)
            opt_fm.step()

            if step == retrain_steps - 1:
                ws_fm.eval()
                with torch.no_grad():
                    cos_t = 0.0
                    for bi in range(n_eval_batches):
                        eidx = eval_indices[0][bi]
                        vim = test_images[eidx].to(device)
                        _, _, vii = ws_model(vim, return_intermediates=True)
                        p = ws_fm(vii[predict_from])
                        cos_t += F.cosine_similarity(
                            p, vii[predict_to], dim=-1).mean().item()
                    print(f"    FM cosine: {cos_t / n_eval_batches:.4f}")

        del opt_fm
        for p in ws_model.parameters():
            p.requires_grad = True

        # === CHECKPOINT EVALUATION ===
        cycle_step = (cycle + 1) * steps_per_cycle
        print(f"\n  CHECKPOINT at step {cycle_step} (end cycle {cycle + 1})")

        ws_loss, ws_acc = eval_model(ws_model)
        ws_rob = measure_robustness(ws_model, f"WS cycle {cycle+1}")

        ws_checkpoints[cycle_step] = {
            "val_loss": ws_loss,
            "val_acc": ws_acc,
            "robustness": ws_rob,
            "wake_dep_gap": wake_dep_gap,
        }
        print(f"    loss={ws_loss:.4f} acc={ws_acc:.4f}")

    # =============================================
    # CONDITION 2: CL CONTINUOUS
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: CL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")

    cl_model = make_vit()
    cl_model.load_state_dict(init_model_state)
    cl_fm = make_fm()
    cl_fm.load_state_dict(init_fm_state)
    cl_gate = CerebellarGate(n_embd).to(device)

    cl_main_params = list(cl_model.parameters()) + list(cl_gate.parameters())
    cl_opt_main = torch.optim.AdamW(cl_main_params, lr=lr, weight_decay=0.01)
    cl_opt_fwd = torch.optim.AdamW(cl_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

    cl_checkpoints = {}
    cl_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for step in range(total_main_steps):
        cl_model.train()
        cl_fm.train()
        cl_gate.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        fwd_pred_cache = {}

        def cl_cerebellar_fn(act, _cache=fwd_pred_cache):
            fwd_pred = cl_fm(act.detach())
            _cache["pred"] = fwd_pred
            return cl_gate(fwd_pred.detach())

        logits, cls_loss, intermediates = cl_model(
            images, labels, return_intermediates=True,
            cerebellar_fn=cl_cerebellar_fn,
            cerebellar_input_block=cerebellar_input_block,
            cerebellar_inject_block=inject_after_block,
        )
        target = intermediates[predict_to].detach()
        fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)

        cl_opt_main.zero_grad()
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(cl_main_params, 1.0)
        cl_opt_main.step()

        cl_opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(cl_fm.parameters(), 1.0)
        cl_opt_fwd.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(cl_model)
            cl_history["train_loss"].append((step, cls_loss.item()))
            cl_history["train_acc"].append((step, acc))
            cl_history["val_loss"].append((step, v_loss))
            cl_history["val_acc"].append((step, v_acc))
            print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f} "
                  f"gate={cl_gate.injection_norm():.3f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  CL CHECKPOINT at step {ckpt_step}")
            cl_loss, cl_acc = eval_model(cl_model)
            cl_rob = measure_robustness(cl_model, f"CL step {ckpt_step}")

            cl_inj_loss, _ = eval_model_with_inj(cl_model, cl_fm, cl_gate)
            dep = cl_loss - cl_inj_loss

            cl_checkpoints[ckpt_step] = {
                "val_loss": cl_loss,
                "val_acc": cl_acc,
                "robustness": cl_rob,
                "dependency_gap": dep,
                "gate_norm": cl_gate.injection_norm(),
            }
            print(f"    loss={cl_loss:.4f} acc={cl_acc:.4f} dep={dep:+.4f}")

    del cl_opt_main, cl_opt_fwd
    torch.cuda.empty_cache()

    # =============================================
    # CONDITION 3: OL CONTINUOUS
    # =============================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 3: OL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")

    ol_model = make_vit()
    ol_model.load_state_dict(init_model_state)
    ol_opt = torch.optim.AdamW(ol_model.parameters(), lr=lr, weight_decay=0.01)

    ol_checkpoints = {}
    ol_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for step in range(total_main_steps):
        ol_model.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        logits, cls_loss = ol_model(images, labels)

        ol_opt.zero_grad()
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(ol_model.parameters(), 1.0)
        ol_opt.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(ol_model)
            ol_history["train_loss"].append((step, cls_loss.item()))
            ol_history["train_acc"].append((step, acc))
            ol_history["val_loss"].append((step, v_loss))
            ol_history["val_acc"].append((step, v_acc))
            print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  OL CHECKPOINT at step {ckpt_step}")
            ol_loss, ol_acc = eval_model(ol_model)
            ol_rob = measure_robustness(ol_model, f"OL step {ckpt_step}")

            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_loss,
                "val_acc": ol_acc,
                "robustness": ol_rob,
            }
            print(f"    loss={ol_loss:.4f} acc={ol_acc:.4f}")

    del ol_opt
    torch.cuda.empty_cache()

    # =============================================
    # FINAL ANALYSIS (at total_main_steps)
    # =============================================
    print(f"\n{'='*60}")
    print(f"  FINAL ANALYSIS")
    print(f"{'='*60}")

    # Train fresh FM on each model for self-knowledge probes
    print("\n--- Training fresh FMs ---")
    ws_fresh_fm, ws_fm_cos = train_fresh_fm(
        ws_model, retrain_steps, seed + 900, "WS")
    cl_fresh_fm, cl_fm_cos = train_fresh_fm(
        cl_model, retrain_steps, seed + 901, "CL")
    ol_fresh_fm, ol_fm_cos = train_fresh_fm(
        ol_model, retrain_steps, seed + 902, "OL")

    # Self-knowledge probes
    print("\n--- Self-knowledge probes ---")
    ws_sk = self_knowledge_probes(ws_model, ws_fresh_fm, "WS")
    cl_sk = self_knowledge_probes(cl_model, cl_fresh_fm, "CL")
    ol_sk = self_knowledge_probes(ol_model, ol_fresh_fm, "OL")

    # Eigenspectrum & digit-discriminative analysis
    print("\n--- Eigenspectrum & digit analysis ---")
    ws_eigen = compute_eigen_stats(ws_model, ws_fresh_fm, "WS")
    cl_eigen = compute_eigen_stats(cl_model, cl_fresh_fm, "CL")
    ol_eigen = compute_eigen_stats(ol_model, ol_fresh_fm, "OL")

    # =============================================
    # SAVE
    # =============================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_wake_sleep_comparison"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(ws_model.state_dict(), os.path.join(save_root, "ws_model.pt"))
    torch.save(cl_model.state_dict(), os.path.join(save_root, "cl_model.pt"))
    torch.save(ol_model.state_dict(), os.path.join(save_root, "ol_model.pt"))
    torch.save(cl_fm.state_dict(), os.path.join(save_root, "cl_fm.pt"))
    torch.save(cl_gate.state_dict(), os.path.join(save_root, "cl_gate.pt"))
    torch.save(ws_fresh_fm.state_dict(), os.path.join(save_root, "ws_fresh_fm.pt"))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "lr": lr, "fwd_lr": fwd_lr, "distill_lr": distill_lr,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "wake_sleep": {
            "checkpoints": ws_checkpoints,
            "history": ws_history,
            "final_fm_cosine": ws_fm_cos,
            "self_knowledge": ws_sk,
            "eigenspectrum": ws_eigen,
        },
        "cl_continuous": {
            "checkpoints": cl_checkpoints,
            "history": cl_history,
            "final_fm_cosine": cl_fm_cos,
            "self_knowledge": cl_sk,
            "eigenspectrum": cl_eigen,
        },
        "ol_continuous": {
            "checkpoints": ol_checkpoints,
            "history": ol_history,
            "final_fm_cosine": ol_fm_cos,
            "self_knowledge": ol_sk,
            "eigenspectrum": ol_eigen,
        },
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

    print(f"\n  === Val loss / accuracy trajectory ===")
    print(f"  {'Step':>6s} | {'WS loss':>8s} {'WS acc':>7s} | "
          f"{'CL loss':>8s} {'CL acc':>7s} | "
          f"{'OL loss':>8s} {'OL acc':>7s}")
    print(f"  {'-'*6}-+-{'-'*8}-{'-'*7}-+-{'-'*8}-{'-'*7}-+-{'-'*8}-{'-'*7}")
    for cs in checkpoint_steps:
        ws = ws_checkpoints[cs]
        cl = cl_checkpoints[cs]
        ol = ol_checkpoints[cs]
        print(f"  {cs:6d} | {ws['val_loss']:8.4f} {ws['val_acc']:7.4f} | "
              f"{cl['val_loss']:8.4f} {cl['val_acc']:7.4f} | "
              f"{ol['val_loss']:8.4f} {ol['val_acc']:7.4f}")

    print(f"\n  === Robustness trajectory (Δloss at ε=1.0) ===")
    print(f"  {'Step':>6s} | {'WS':>8s} {'CL':>8s} {'OL':>8s} | "
          f"{'WS/OL':>6s} {'CL/OL':>6s}")
    print(f"  {'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*6}-{'-'*6}")
    for cs in checkpoint_steps:
        ws_r = ws_checkpoints[cs]["robustness"]["1.0"]
        cl_r = cl_checkpoints[cs]["robustness"]["1.0"]
        ol_r = ol_checkpoints[cs]["robustness"]["1.0"]
        ws_ratio = ws_r / ol_r if ol_r > 0 else float("inf")
        cl_ratio = cl_r / ol_r if ol_r > 0 else float("inf")
        print(f"  {cs:6d} | {ws_r:+8.4f} {cl_r:+8.4f} {ol_r:+8.4f} | "
              f"{ws_ratio:6.3f} {cl_ratio:6.3f}")

    print(f"\n  === Self-knowledge (R² at final checkpoint) ===")
    print(f"  {'Layer':>12s} | {'WS':>6s} {'CL':>6s} {'OL':>6s}")
    print(f"  {'-'*12}-+-{'-'*6}-{'-'*6}-{'-'*6}")
    for lk in layer_keys:
        print(f"  {lk:>12s} | {ws_sk[lk]:6.3f} {cl_sk[lk]:6.3f} {ol_sk[lk]:6.3f}")

    print(f"\n  === Eigenspectrum (final checkpoint) ===")
    print(f"  {'':>12s} | {'WS':>8s} {'CL':>8s} {'OL':>8s}")
    print(f"  {'eff_rank':>12s} | {ws_eigen['eff_rank']:8.1f} "
          f"{cl_eigen['eff_rank']:8.1f} {ol_eigen['eff_rank']:8.1f}")
    print(f"  {'mean_eta²':>12s} | {ws_eigen['mean_eta_squared']:8.3f} "
          f"{cl_eigen['mean_eta_squared']:8.3f} {ol_eigen['mean_eta_squared']:8.3f}")
    print(f"  {'res_norm':>12s} | {ws_eigen['mean_res_norm']:8.3f} "
          f"{cl_eigen['mean_res_norm']:8.3f} {ol_eigen['mean_res_norm']:8.3f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    wake_steps: int = 5000,
    distill_steps: int = 2000,
    retrain_steps: int = 5000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
):
    result = a2a_mnist_wake_sleep_comparison.remote(
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        predict_from=predict_from,
        predict_to=predict_to,
        inject_after_block=inject_after_block,
    )
    print("\nWake-sleep comparison complete.")

    config = result["config"]
    print(f"\n  {config['n_cycles']} cycles × "
          f"({config['wake_steps']} wake + {config['distill_steps']} sleep) = "
          f"{config['total_main_steps']} main-model steps")

    ws = result["wake_sleep"]
    cl = result["cl_continuous"]
    ol = result["ol_continuous"]

    final_step = config["checkpoint_steps"][-1]

    # Handle both int and string keys (JSON serialization converts int keys)
    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    ws_final = get_ckpt(ws["checkpoints"], final_step)
    cl_final = get_ckpt(cl["checkpoints"], final_step)
    ol_final = get_ckpt(ol["checkpoints"], final_step)

    print(f"\n  Final ({final_step} steps):")
    print(f"    WS: loss={ws_final['val_loss']:.4f} acc={ws_final['val_acc']:.4f}")
    print(f"    CL: loss={cl_final['val_loss']:.4f} acc={cl_final['val_acc']:.4f}")
    print(f"    OL: loss={ol_final['val_loss']:.4f} acc={ol_final['val_acc']:.4f}")

    ws_r = ws_final["robustness"]["1.0"]
    cl_r = cl_final["robustness"]["1.0"]
    ol_r = ol_final["robustness"]["1.0"]
    print(f"\n  Robustness (ε=1.0 Δloss):")
    print(f"    WS: {ws_r:+.4f} ({ws_r/ol_r:.3f}x OL)")
    print(f"    CL: {cl_r:+.4f} ({cl_r/ol_r:.3f}x OL)")
    print(f"    OL: {ol_r:+.4f}")
