"""MNIST extended gated ratchet: long-run convergence test.

Runs WS_LG (injection + bilevel-gated local loss + distillation + FM reinit)
for many cycles to test the absorbing state prediction: does the model's
computation converge to a fully FM-predictable state (residual → 0)?

Only runs WS_LG + lightweight OL reference. Same architecture, hyperparams,
and per-cycle measurements as mnist_gated_ratchet.py (4-cycle experiment).

Prior: mnist_gated_ratchet.py (GATED_RATCHET_README.md)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def a2a_mnist_extended_ratchet(
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
    fwd_mlp_mult: float = 2,
    # Wake-sleep protocol
    n_cycles: int = 16,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    # Training params
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    gate_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    # Eval params
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    probe_batches: int = 40,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.func import functional_call
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    n_positions = (28 // patch_size) ** 2 + 1

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    print(f"MNIST EXTENDED RATCHET on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  lambda_local={lambda_local}, lg_hidden={lg_hidden}")

    # -----------------------------------------------------------------
    # Learning Gate (same as mnist_gated_ratchet.py)
    # -----------------------------------------------------------------
    class LearningGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            n_params = sum(p.numel() for p in self.parameters())
            print(f"LearningGate: {n_params/1e3:.1f}K params "
                  f"(input={2*d_model}, hidden={d_hidden}, output={d_model})")

        def forward(self, source_cls, fm_error_cls):
            x = torch.cat([source_cls, fm_error_cls], dim=-1)
            return torch.sigmoid(self.net(x))

    # -----------------------------------------------------------------
    # Load MNIST
    # -----------------------------------------------------------------
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # -----------------------------------------------------------------
    # Pre-generate batch indices (same seed as mnist_gated_ratchet.py)
    # -----------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)

    max_repoint_steps = n_cycles * retrain_steps

    train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=train_gen)
        for _ in range(total_main_steps)
    ]
    n_evals = total_main_steps // eval_interval + 10
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

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Initialize shared weights
    # -----------------------------------------------------------------
    torch.manual_seed(seed)
    init_model = make_vit()
    init_fm = make_fm()
    init_model_state = {k: v.cpu().clone()
                        for k, v in init_model.state_dict().items()}
    init_fm_state = {k: v.cpu().clone()
                     for k, v in init_fm.state_dict().items()}
    del init_model, init_fm
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =================================================================
    # SHARED HELPERS (identical to mnist_gated_ratchet.py)
    # =================================================================
    def eval_model(model, eval_idx=0):
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)
                logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def eval_model_with_inj(model, fm, gate, eval_idx=0):
        model.eval(); fm.eval(); gate.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
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
                results[str(eps)] = float(np.mean(deltas))
        print(f"  [{label:>20s}] "
              + " | ".join(f"eps={e}: d={results[str(e)]:+.4f}"
                           for e in eps_list))
        return results

    def compute_residual_stats(model, fm, label):
        model.eval(); fm.eval()
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
            eta_sqs.append(float(ss_between / ss_total) if ss_total > 0 else 0.)

        top5_vecs = vecs.copy()

        stats = {
            "eff_rank": eff_rank,
            "top1_pc_frac": float(fracs[0]),
            "top5_pc_frac": float(fracs[:5].sum()),
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "eta_squared_top5": eta_sqs,
            "mean_eta_squared": float(np.mean(eta_sqs)),
        }

        with torch.no_grad():
            cos_total = 0.0
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                cos_total += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
        stats["fwd_cosine"] = cos_total / n_eval_batches

        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"mean_eta2={float(np.mean(eta_sqs)):.3f}, "
              f"res_norm={stats['mean_res_norm']:.3f}, "
              f"fwd_cos={stats['fwd_cosine']:.4f}")
        return stats, top5_vecs

    def train_fresh_fm(model, steps, fm_seed, label):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(steps):
            fm.train()
            ridx = repoint_indices[step % len(repoint_indices)]
            images = train_images[ridx].to(device)
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
                p = fm(vi[predict_from])
                cos_total += F.cosine_similarity(
                    p, vi[predict_to], dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cosine: {cos:.4f}")

        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    def compute_innovation_migration(vecs_old, vecs_new):
        if vecs_old is None or vecs_new is None:
            return None
        overlap = np.abs(vecs_old.T @ vecs_new)
        diag_cos = [float(overlap[i, i]) for i in range(min(5, overlap.shape[0]))]
        max_cos = [float(overlap[i].max()) for i in range(min(5, overlap.shape[0]))]
        proj = vecs_old.T @ vecs_new
        subspace_overlap = float(np.trace(proj.T @ proj) / 5.0)
        return {
            "diag_cosines": diag_cos,
            "max_cosines": max_cos,
            "subspace_overlap": subspace_overlap,
        }

    def get_gate_stats(lgate, model, fwd_model):
        lgate.eval(); model.eval(); fwd_model.eval()
        all_gw = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                vim = test_images[pidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                gw = lgate(src[:, 0], (tgt - pred)[:, 0])
                all_gw.append(gw.cpu())
        gw_cat = torch.cat(all_gw, dim=0)
        dim_mean = gw_cat.mean(dim=0)
        return {
            "overall_mean": float(gw_cat.mean()),
            "overall_std": float(gw_cat.std()),
            "dim_mean_min": float(dim_mean.min()),
            "dim_mean_max": float(dim_mean.max()),
            "dim_mean_std": float(dim_mean.std()),
            "sparsity_below_0.1": float((dim_mean < 0.1).float().mean()),
            "sparsity_below_0.2": float((dim_mean < 0.2).float().mean()),
            "dim_mean": dim_mean.tolist(),
        }

    # =================================================================
    # CONDITION 1: WS_LG (Wake-Sleep with Learning Gate)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: WS_LG ({n_cycles} cycles)")
    print(f"{'='*60}")

    wslg_model = make_vit()
    wslg_model.load_state_dict(init_model_state)
    wslg_fm = make_fm()
    wslg_fm.load_state_dict(init_fm_state)
    wslg_lgate = LearningGate(n_embd, lg_hidden).to(device)

    wslg_checkpoints = {}
    wslg_history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "gate_stats": [],
    }
    wslg_global_step = 0
    prev_top5_vecs = None

    for cycle in range(n_cycles):
        print(f"\n  --- WS_LG Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: CL_LG co-training (bilevel) ===
        print(f"  [WAKE] CL_LG co-training ({wake_steps} steps)")
        wslg_cgate = CerebellarGate(n_embd).to(device)
        main_params = list(wslg_model.parameters()) + list(wslg_cgate.parameters())
        n_model_params = len(list(wslg_model.parameters()))
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            wslg_fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        opt_lg = torch.optim.AdamW(
            wslg_lgate.parameters(), lr=gate_lr, weight_decay=0.01)

        for step in range(wake_steps):
            wslg_model.train(); wslg_fm.train()
            wslg_cgate.train(); wslg_lgate.train()

            idx = train_indices[wslg_global_step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            fwd_pred_cache = {}

            def cerebellar_fn(act, _cache=fwd_pred_cache):
                fp = wslg_fm(act.detach())
                _cache["pred"] = fp
                return wslg_cgate(fp.detach())

            logits, cls_loss, intermediates = wslg_model(
                images, labels, return_intermediates=True,
                cerebellar_fn=cerebellar_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )
            fwd_pred = fwd_pred_cache["pred"]
            target = intermediates[predict_to]
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            source_cls = intermediates[predict_from][:, 0].detach()
            fm_error_cls = (target - fwd_pred).detach()[:, 0]
            gate_w = wslg_lgate(source_cls, fm_error_cls)

            r = target - fwd_pred.detach()
            gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()
            L_total = cls_loss + lambda_local * gated_ll

            inner_grads = torch.autograd.grad(
                L_total, main_params, create_graph=True)
            grads_for_update = [g.detach().clone() for g in inner_grads]

            virtual_model_params = {
                pname: p - lr * g
                for (pname, p), g in zip(
                    wslg_model.named_parameters(),
                    inner_grads[:n_model_params],
                )
            }
            virtual_cgate_params = {
                pname: p - lr * g
                for (pname, p), g in zip(
                    wslg_cgate.named_parameters(),
                    inner_grads[n_model_params:],
                )
            }

            def meta_cerebellar_fn(act):
                fp = wslg_fm(act.detach())
                return functional_call(
                    wslg_cgate, virtual_cgate_params, (fp.detach(),))

            meta_out = functional_call(
                wslg_model, virtual_model_params,
                args=(images,),
                kwargs={
                    "targets": labels,
                    "cerebellar_fn": meta_cerebellar_fn,
                    "cerebellar_input_block": cerebellar_input_block,
                    "cerebellar_inject_block": inject_after_block,
                },
            )
            cls_loss_prime = meta_out[1]

            opt_lg.zero_grad()
            cls_loss_prime.backward()
            torch.nn.utils.clip_grad_norm_(wslg_lgate.parameters(), 1.0)
            opt_lg.step()

            opt_main.zero_grad()
            for p, g in zip(main_params, grads_for_update):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(wslg_fm.parameters(), 1.0)
            opt_fwd.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss, v_acc = eval_model(wslg_model)
                wslg_history["train_loss"].append(
                    (wslg_global_step, cls_loss.item()))
                wslg_history["train_acc"].append((wslg_global_step, acc))
                wslg_history["val_loss"].append((wslg_global_step, v_loss))
                wslg_history["val_acc"].append((wslg_global_step, v_acc))

                dep_loss, _ = eval_model_with_inj(
                    wslg_model, wslg_fm, wslg_cgate)
                dep = v_loss - dep_loss
                print(f"    step {wslg_global_step:5d} (wake {step}): "
                      f"val={v_loss:.4f} acc={v_acc:.4f} "
                      f"dep={dep:+.4f} gate={wslg_cgate.injection_norm():.3f}"
                      f" gw={gate_w.mean().item():.3f}")

            wslg_global_step += 1

        gs = get_gate_stats(wslg_lgate, wslg_model, wslg_fm)
        wslg_history["gate_stats"].append(
            (wslg_global_step, f"post_wake_c{cycle+1}", gs))
        print(f"    Gate: mean={gs['overall_mean']:.3f} "
              f"std={gs['overall_std']:.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f}")

        wake_dep_loss, _ = eval_model_with_inj(
            wslg_model, wslg_fm, wslg_cgate)
        wake_noinj_loss, _ = eval_model(wslg_model)
        wake_dep_gap = wake_noinj_loss - wake_dep_loss
        print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}")

        # === SLEEP: Distillation ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")

        teacher = make_vit()
        teacher.load_state_dict(wslg_model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(wslg_fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False
        teacher_gate = CerebellarGate(n_embd).to(device)
        teacher_gate.load_state_dict(wslg_cgate.state_dict())
        teacher_gate.eval()
        for p in teacher_gate.parameters():
            p.requires_grad = False

        student = make_vit()
        student.load_state_dict(wslg_model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            idx = train_indices[wslg_global_step]
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
                wslg_history["val_loss"].append((wslg_global_step, v_loss))
                wslg_history["val_acc"].append((wslg_global_step, v_acc))
                print(f"    step {wslg_global_step:5d} (sleep {step}): "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            wslg_global_step += 1

        wslg_model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_gate, student, opt_student
        del wslg_cgate, opt_main, opt_fwd, opt_lg
        torch.cuda.empty_cache()

        # === RE-POINT: Fresh FM ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, seed={fm_seed})")
        wslg_fm, fm_cos = train_fresh_fm(
            wslg_model, retrain_steps, fm_seed,
            f"WS_LG c{cycle+1}")

        # === CHECKPOINT EVALUATION ===
        cycle_step = (cycle + 1) * steps_per_cycle
        print(f"\n  WS_LG CHECKPOINT at step {cycle_step} "
              f"(end cycle {cycle + 1})")

        ws_loss, ws_acc = eval_model(wslg_model)
        ws_rob = measure_robustness(wslg_model, f"WS_LG c{cycle+1}")
        ws_eigen, top5_vecs = compute_residual_stats(
            wslg_model, wslg_fm, f"WS_LG c{cycle+1}")

        innov_mig = compute_innovation_migration(prev_top5_vecs, top5_vecs)
        prev_top5_vecs = top5_vecs

        gs_post = get_gate_stats(wslg_lgate, wslg_model, wslg_fm)
        wslg_history["gate_stats"].append(
            (cycle_step, f"post_repoint_c{cycle+1}", gs_post))
        print(f"    Gate (post-repoint): mean={gs_post['overall_mean']:.3f} "
              f"std={gs_post['overall_std']:.3f} "
              f"sparse(<0.1)={gs_post['sparsity_below_0.1']:.3f}")

        wslg_checkpoints[cycle_step] = {
            "val_loss": ws_loss, "val_acc": ws_acc,
            "robustness": ws_rob,
            "residual_stats": ws_eigen,
            "innovation_migration": innov_mig,
            "gate_stats_post_wake": wslg_history["gate_stats"][-2][2],
            "gate_stats_post_repoint": gs_post,
            "wake_dep_gap": wake_dep_gap,
            "fm_cosine": fm_cos,
        }
        print(f"    loss={ws_loss:.4f} acc={ws_acc:.4f} "
              f"res_norm={ws_eigen['mean_res_norm']:.4f} "
              f"fwd_cos={fm_cos:.4f}")

    # =================================================================
    # CONDITION 2: OL Continuous (lightweight reference)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: OL CONTINUOUS ({total_main_steps} steps)")
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
            if step % (eval_interval * 10) == 0:
                print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            ol_l, ol_a = eval_model(ol_model)
            ol_r = measure_robustness(ol_model, f"OL step {ckpt_step}")
            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_l, "val_acc": ol_a,
                "robustness": ol_r,
            }
            print(f"  OL CHECKPOINT at step {ckpt_step}: "
                  f"loss={ol_l:.4f} acc={ol_a:.4f}")

    del ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_extended_ratchet"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(wslg_model.state_dict(),
               os.path.join(save_root, "wslg_model.pt"))
    torch.save(wslg_fm.state_dict(),
               os.path.join(save_root, "wslg_fm.pt"))
    torch.save(wslg_lgate.state_dict(),
               os.path.join(save_root, "wslg_lgate.pt"))
    torch.save(ol_model.state_dict(),
               os.path.join(save_root, "ol_model.pt"))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "lr": lr, "fwd_lr": fwd_lr, "gate_lr": gate_lr,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "lambda_local": lambda_local, "lg_hidden": lg_hidden,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "ws_lg": {
            "checkpoints": wslg_checkpoints,
            "history": wslg_history,
        },
        "ol": {
            "checkpoints": ol_checkpoints,
            "history": ol_history,
        },
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {n_cycles} cycles × ({wake_steps} wake + {distill_steps} sleep)"
          f" = {total_main_steps} main-model steps")

    print(f"\n  === Val loss / accuracy trajectory ===")
    print(f"  {'Cycle':>5s} {'Step':>6s} | {'WS_LG':>12s} | {'OL':>12s} "
          f"| {'gap':>6s}")
    for i, cs in enumerate(checkpoint_steps):
        wc = wslg_checkpoints[cs]
        oc = ol_checkpoints[cs]
        gap = (wc["val_loss"] - oc["val_loss"]) / oc["val_loss"] * 100
        print(f"  {i+1:5d} {cs:6d} | {wc['val_loss']:.4f} {wc['val_acc']:.3f}"
              f" | {oc['val_loss']:.4f} {oc['val_acc']:.3f}"
              f" | {gap:+5.1f}%")

    print(f"\n  === Convergence indicators ===")
    print(f"  {'Cycle':>5s} | {'res_norm':>8s} {'fwd_cos':>8s} "
          f"{'eff_rank':>8s} {'eta2':>6s} "
          f"{'gate_m':>6s} {'rob1.0':>8s}")
    for i, cs in enumerate(checkpoint_steps):
        wc = wslg_checkpoints[cs]
        rs = wc.get("residual_stats", {})
        gs = wc.get("gate_stats_post_repoint", {})
        rob = wc.get("robustness", {}).get("1.0", 0)
        print(f"  {i+1:5d} | {rs.get('mean_res_norm', 0):8.4f} "
              f"{wc.get('fm_cosine', 0):8.4f} "
              f"{rs.get('eff_rank', 0):8.1f} "
              f"{rs.get('mean_eta_squared', 0):6.3f} "
              f"{gs.get('overall_mean', 0):6.3f} "
              f"{rob:+8.4f}")

    if any(wslg_checkpoints[cs].get("innovation_migration")
           for cs in checkpoint_steps):
        print(f"\n  === Innovation migration (subspace overlap) ===")
        for i, cs in enumerate(checkpoint_steps):
            im = wslg_checkpoints[cs].get("innovation_migration")
            if im:
                print(f"    Cycle {i+1}: overlap={im['subspace_overlap']:.3f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 16,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    lambda_local: float = 1.0,
    gate_lr: float = 1e-3,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_extended_ratchet.remote(
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        gate_lr=gate_lr,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nExtended ratchet experiment complete.")
