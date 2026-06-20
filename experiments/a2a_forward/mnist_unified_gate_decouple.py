"""MNIST unified gate decoupling test.

Tests whether the coupling between inference selectivity and learning selectivity
is the mechanism behind the unified gate's compounding. The null hypothesis:
the gate's per-dimension local loss weighting doesn't matter — any local loss
with matched magnitude would produce the same compounding.

Two gated conditions (+ OL reference), all compute-matched:

1. WS_UG: Unified gate, selective local loss (gate_w.detach() * FM_error^2)
   Reference — should reproduce the unified gate experiment result (0.0564).

2. WS_UG_uniform: Unified gate for injection, but UNIFORM local loss.
   Uses gate_w.detach().mean() * FM_error^2 — same magnitude, no selectivity.
   If WS_UG >> WS_UG_uniform: the per-dimension coupling is doing real work.
   If WS_UG ≈ WS_UG_uniform: the coupling doesn't matter on stationary MNIST.

3. OL: Open-loop baseline (sanity check, should reproduce 0.1005).
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def a2a_mnist_ug_decouple(
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
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    # Training params
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    # Eval params
    eval_interval: int = 200,
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
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    n_positions = (28 // patch_size) ** 2 + 1

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    print(f"MNIST UNIFIED GATE DECOUPLING TEST on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  lambda_local={lambda_local}, lg_hidden={lg_hidden}")

    # -----------------------------------------------------------------
    # Unified Gate
    # -----------------------------------------------------------------
    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            self.projection = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            n_params = sum(p.numel() for p in self.parameters())
            print(f"UnifiedGate: {n_params/1e3:.1f}K params "
                  f"(input={2*d_model}, hidden={d_hidden}, output={d_model})")

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            injection = gate_w * self.projection(fwd_pred)
            return injection, gate_w

        def injection_norm(self):
            return self.projection.weight.norm().item()

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
    # Pre-generate batch indices (same seed as full experiment)
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
    # SHARED HELPERS
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

    def eval_model_with_ug(model, fm, ugate, eval_idx=0):
        model.eval(); fm.eval(); ugate.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)

                def cb(act):
                    fp = fm(act)
                    inj, _ = ugate(act, fp)
                    return inj

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

    def get_ug_gate_stats(ugate, model, fwd_model):
        ugate.eval(); model.eval(); fwd_model.eval()
        all_gw_cls = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                vim = test_images[pidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                pred = fwd_model(src)
                _, gw = ugate(src, pred)
                all_gw_cls.append(gw[:, 0].cpu())
        gw_cls = torch.cat(all_gw_cls, dim=0)
        dim_mean = gw_cls.mean(dim=0)
        return {
            "overall_mean": float(gw_cls.mean()),
            "overall_std": float(gw_cls.std()),
            "dim_mean_min": float(dim_mean.min()),
            "dim_mean_max": float(dim_mean.max()),
            "dim_mean_std": float(dim_mean.std()),
            "sparsity_below_0.1": float((dim_mean < 0.1).float().mean()),
            "sparsity_below_0.2": float((dim_mean < 0.2).float().mean()),
            "dim_mean": dim_mean.tolist(),
        }

    # =================================================================
    # Generic WS_UG training loop (parameterized by local loss mode)
    # =================================================================
    def run_ws_ug(condition_name, uniform_local_loss=False):
        print(f"\n{'='*60}")
        print(f"  {condition_name} ({n_cycles} cycles)")
        print(f"{'='*60}")

        model = make_vit()
        model.load_state_dict(init_model_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)
        ugate = UnifiedGate(n_embd, lg_hidden).to(device)

        checkpoints = {}
        history = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "gate_stats": [],
        }
        global_step = 0

        for cycle in range(n_cycles):
            print(f"\n  --- {condition_name} Cycle {cycle + 1}/{n_cycles} ---")

            # === WAKE ===
            ll_mode = "uniform" if uniform_local_loss else "selective"
            print(f"  [WAKE] UG co-training, {ll_mode} local loss "
                  f"({wake_steps} steps)")
            main_params = (list(model.parameters())
                           + list(ugate.parameters()))
            opt_main = torch.optim.AdamW(
                main_params, lr=lr, weight_decay=0.01)
            opt_fwd = torch.optim.AdamW(
                fm.parameters(), lr=fwd_lr, weight_decay=0.01)

            for step in range(wake_steps):
                model.train(); fm.train(); ugate.train()

                idx = train_indices[global_step]
                images = train_images[idx].to(device)
                labels = train_labels[idx].to(device)

                fwd_pred_cache = {}

                def cerebellar_fn(act, _cache=fwd_pred_cache):
                    fp = fm(act.detach())
                    _cache["pred"] = fp
                    inj, gw = ugate(act.detach(), fp.detach())
                    _cache["gate_w"] = gw
                    return inj

                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )

                fwd_pred = fwd_pred_cache["pred"]
                target = intermediates[predict_to]
                fwd_loss = F.mse_loss(fwd_pred, target.detach())

                gate_w = fwd_pred_cache["gate_w"]
                r = target - fwd_pred.detach()

                if uniform_local_loss:
                    gw_scalar = gate_w.detach().mean()
                    gated_ll = (gw_scalar * r ** 2).mean()
                else:
                    gated_ll = (gate_w.detach() * r ** 2).mean()

                L_total = cls_loss + lambda_local * gated_ll

                opt_main.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                opt_main.step()

                opt_fwd.zero_grad()
                fwd_loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt_fwd.step()

                acc = (logits.argmax(-1) == labels).float().mean().item()

                if step % eval_interval == 0 or step == wake_steps - 1:
                    v_loss, v_acc = eval_model(model)
                    history["train_loss"].append(
                        (global_step, cls_loss.item()))
                    history["train_acc"].append((global_step, acc))
                    history["val_loss"].append((global_step, v_loss))
                    history["val_acc"].append((global_step, v_acc))

                    dep_loss, _ = eval_model_with_ug(model, fm, ugate)
                    dep = v_loss - dep_loss
                    print(f"    step {global_step:5d} (wake {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f} "
                          f"dep={dep:+.4f} "
                          f"proj={ugate.injection_norm():.3f}"
                          f" gw={gate_w.mean().item():.3f}"
                          f" ll={gated_ll.item():.4f}")

                global_step += 1

            gs = get_ug_gate_stats(ugate, model, fm)
            history["gate_stats"].append(
                (global_step, f"post_wake_c{cycle+1}", gs))
            print(f"    Gate: mean={gs['overall_mean']:.3f} "
                  f"std={gs['overall_std']:.3f} "
                  f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f}")

            wake_dep_loss, _ = eval_model_with_ug(model, fm, ugate)
            wake_noinj_loss, _ = eval_model(model)
            wake_dep_gap = wake_noinj_loss - wake_dep_loss

            # === SLEEP ===
            print(f"  [SLEEP] Distillation ({distill_steps} steps)")

            teacher = make_vit()
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            teacher_fm = make_fm()
            teacher_fm.load_state_dict(fm.state_dict())
            teacher_fm.eval()
            for p in teacher_fm.parameters():
                p.requires_grad = False
            teacher_ugate = UnifiedGate(n_embd, lg_hidden).to(device)
            teacher_ugate.load_state_dict(ugate.state_dict())
            teacher_ugate.eval()
            for p in teacher_ugate.parameters():
                p.requires_grad = False

            student = make_vit()
            student.load_state_dict(model.state_dict())
            opt_student = torch.optim.AdamW(
                student.parameters(), lr=distill_lr, weight_decay=0.01)

            for step in range(distill_steps):
                student.train()
                idx = train_indices[global_step]
                images = train_images[idx].to(device)
                labels = train_labels[idx].to(device)

                with torch.no_grad():
                    def t_cb(act):
                        fp = teacher_fm(act)
                        inj, _ = teacher_ugate(act, fp)
                        return inj
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
                    history["val_loss"].append((global_step, v_loss))
                    history["val_acc"].append((global_step, v_acc))
                    print(f"    step {global_step:5d} (sleep {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f}")

                global_step += 1

            model.load_state_dict(student.state_dict())
            del teacher, teacher_fm, teacher_ugate, student, opt_student
            del opt_main, opt_fwd
            torch.cuda.empty_cache()

            # === RE-POINT ===
            fm_seed = seed + 100 * (cycle + 1)
            print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, "
                  f"seed={fm_seed})")
            fm, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed,
                f"{condition_name} c{cycle+1}")

            # === CHECKPOINT ===
            cycle_step = (cycle + 1) * steps_per_cycle
            print(f"\n  {condition_name} CHECKPOINT at step {cycle_step} "
                  f"(end cycle {cycle + 1})")

            c_loss, c_acc = eval_model(model)
            c_rob = measure_robustness(
                model, f"{condition_name} c{cycle+1}")

            gs_post = get_ug_gate_stats(ugate, model, fm)
            history["gate_stats"].append(
                (cycle_step, f"post_repoint_c{cycle+1}", gs_post))
            print(f"    Gate (post-repoint): "
                  f"mean={gs_post['overall_mean']:.3f} "
                  f"std={gs_post['overall_std']:.3f} "
                  f"sparse(<0.1)={gs_post['sparsity_below_0.1']:.3f}")

            checkpoints[cycle_step] = {
                "val_loss": c_loss, "val_acc": c_acc,
                "robustness": c_rob,
                "gate_stats_post_wake": history["gate_stats"][-2][2],
                "gate_stats_post_repoint": gs_post,
                "wake_dep_gap": wake_dep_gap,
                "fm_cosine": fm_cos,
            }
            print(f"    loss={c_loss:.4f} acc={c_acc:.4f}")

        return model, fm, ugate, checkpoints, history

    # =================================================================
    # Run conditions
    # =================================================================
    ug_model, ug_fm, ug_gate, ug_ckpts, ug_hist = run_ws_ug(
        "WS_UG", uniform_local_loss=False)

    del ug_model, ug_fm, ug_gate
    torch.cuda.empty_cache()

    uu_model, uu_fm, uu_gate, uu_ckpts, uu_hist = run_ws_ug(
        "WS_UG_uniform", uniform_local_loss=True)

    del uu_model, uu_fm, uu_gate
    torch.cuda.empty_cache()

    # =================================================================
    # OL Baseline
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  OL CONTINUOUS ({total_main_steps} steps)")
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
            if step % 1000 == 0:
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

    del ol_model, ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_ug_decouple"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    result = {
        "config": {
            "n_cycles": n_cycles,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "lr": lr, "fwd_lr": fwd_lr,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "lambda_local": lambda_local, "lg_hidden": lg_hidden,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "ws_ug": {
            "checkpoints": ug_ckpts,
            "history": ug_hist,
        },
        "ws_ug_uniform": {
            "checkpoints": uu_ckpts,
            "history": uu_hist,
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
    print(f"  SUMMARY: DECOUPLING TEST")
    print(f"{'='*60}")

    print(f"\n  === Val loss / accuracy trajectory ===")
    print(f"  {'Step':>6s} | {'WS_UG':>12s} | {'WS_UG_uni':>12s} | "
          f"{'OL':>12s}")
    for cs in checkpoint_steps:
        ug_c = ug_ckpts[cs]
        uu_c = uu_ckpts[cs]
        ol_c = ol_checkpoints[cs]
        print(f"  {cs:6d} | {ug_c['val_loss']:.4f} {ug_c['val_acc']:.3f}"
              f" | {uu_c['val_loss']:.4f} {uu_c['val_acc']:.3f}"
              f" | {ol_c['val_loss']:.4f} {ol_c['val_acc']:.3f}")

    print(f"\n  === Robustness (delta-loss at eps=1.0) ===")
    print(f"  {'Step':>6s} | {'WS_UG':>8s} {'WS_UG_uni':>10s} "
          f"{'OL':>8s}")
    for cs in checkpoint_steps:
        vals = [ug_ckpts[cs]["robustness"]["1.0"],
                uu_ckpts[cs]["robustness"]["1.0"],
                ol_checkpoints[cs]["robustness"]["1.0"]]
        print(f"  {cs:6d} | " +
              " ".join(f"{v:+10.4f}" for v in vals))

    print(f"\n  === Gate dynamics ===")
    print(f"  {'Phase':>25s} | {'UG mean':>8s} {'UG spr':>8s} "
          f"| {'UU mean':>8s} {'UU spr':>8s}")
    ug_gs = ug_hist["gate_stats"]
    uu_gs = uu_hist["gate_stats"]
    for i in range(min(len(ug_gs), len(uu_gs))):
        _, ug_label, ug_s = ug_gs[i]
        _, _, uu_s = uu_gs[i]
        print(f"  {ug_label:>25s} | "
              f"{ug_s['overall_mean']:8.3f} "
              f"{ug_s['sparsity_below_0.1']:8.3f} | "
              f"{uu_s['overall_mean']:8.3f} "
              f"{uu_s['sparsity_below_0.1']:8.3f}")

    # Gate dimension correlation between selective and uniform
    final_cs = checkpoint_steps[-1]
    ug_dims = np.array(
        ug_ckpts[final_cs]["gate_stats_post_repoint"]["dim_mean"])
    uu_dims = np.array(
        uu_ckpts[final_cs]["gate_stats_post_repoint"]["dim_mean"])
    dim_corr = float(np.corrcoef(ug_dims, uu_dims)[0, 1])
    print(f"\n  Gate dimension correlation (selective vs uniform): "
          f"r={dim_corr:.3f}")

    # Key comparison
    ug_final = ug_ckpts[final_cs]["val_loss"]
    uu_final = uu_ckpts[final_cs]["val_loss"]
    ol_final = ol_checkpoints[final_cs]["val_loss"]
    print(f"\n  Final val loss: WS_UG={ug_final:.4f} "
          f"WS_UG_uniform={uu_final:.4f} OL={ol_final:.4f}")
    if ug_final < uu_final - 0.005:
        print(f"  >> WS_UG beats WS_UG_uniform by "
              f"{uu_final - ug_final:.4f} — coupling matters!")
    elif abs(ug_final - uu_final) < 0.005:
        print(f"  >> WS_UG ≈ WS_UG_uniform (diff={ug_final-uu_final:+.4f})"
              f" — selectivity not needed on stationary MNIST")
    else:
        print(f"  >> WS_UG_uniform beats WS_UG by "
              f"{ug_final - uu_final:.4f} — uniform is better!")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_ug_decouple.remote(
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nDecoupling test complete.")
