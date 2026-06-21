"""MNIST ratchet OOD adaptation experiment.

Tests whether the compounding val loss improvement from the gated ratchet
translates to better OOD adaptation — the most direct test of whether
the meta-learning is genuine.

Four conditions, all compute-matched (8400 main-model gradient steps):

1. WS_LG: FOMAML bilevel gate + injection + local loss + distillation ratchet
   (4 cycles × 1500 wake + 600 sleep)

2. WS_UG_uniform: Unified gate (NTP-trained) + uniform local loss + distillation
   (4 cycles × 1500 wake + 600 sleep)

3. CL: Closed-loop injection only, continuous (8400 steps, no distillation/gate)

4. OL: Open-loop baseline (8400 steps)

After training, all models are evaluated standalone (no injection) on:
- Zero-shot rotated MNIST (15, 30, 45, 60, 90 degrees)
- Fine-tuning adaptation curves (500 steps, lr=1e-4)
- Catastrophic forgetting (ID accuracy after OOD fine-tuning)

Predictions:
- Zero-shot OOD: WS_LG ≈ WS_UG_uniform >> CL ≈ OL (distillation effect)
- Adaptation speed: uninformative (prior result)
- Forgetting: WS_LG ≈ WS_UG_uniform ≈ CL >> OL (injection experience)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def a2a_mnist_ratchet_adaptation(
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
    gate_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    # Adaptation params
    adapt_steps: int = 500,
    adapt_lr: float = 1e-4,
    adapt_eval_interval: int = 10,
    # Eval params
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    probe_batches: int = 40,
    seed: int = 42,
):
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.func import functional_call
    from scipy.ndimage import rotate as scipy_rotate
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    n_positions = (28 // patch_size) ** 2 + 1

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]
    rotation_angles = [15, 30, 45, 60, 90]
    conditions = ["WS_LG", "WS_UG_uniform", "CL", "OL"]

    print(f"MNIST RATCHET OOD ADAPTATION on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  Adaptation: {adapt_steps} steps, lr={adapt_lr}")
    print(f"  Rotation angles: {rotation_angles}")

    # -----------------------------------------------------------------
    # Gate classes
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
            print(f"LearningGate: {n_params/1e3:.1f}K params")

        def forward(self, source_cls, fm_error_cls):
            x = torch.cat([source_cls, fm_error_cls], dim=-1)
            return torch.sigmoid(self.net(x))

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
            print(f"UnifiedGate: {n_params/1e3:.1f}K params")

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            injection = gate_w * self.projection(fwd_pred)
            return injection, gate_w

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
    # Pre-generate batch indices
    # -----------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
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
    eval_counter = [0]

    def eval_model(model):
        model.eval()
        idx = eval_counter[0] % len(eval_indices)
        eval_counter[0] += 1
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[idx][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)
                logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def eval_full(model, images, labels):
        model.eval()
        total_loss, total_correct = 0.0, 0
        n = len(images)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                b_imgs = images[start:end].to(device)
                b_labs = labels[start:end].to(device)
                logits, loss = model(b_imgs, b_labs)
                total_loss += loss.item() * (end - start)
                total_correct += (logits.argmax(-1) == b_labs).sum().item()
        return total_loss / n, total_correct / n

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

    # =================================================================
    # CONDITION 1: WS_LG (FOMAML bilevel gate + distillation ratchet)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: WS_LG ({n_cycles} cycles)")
    print(f"{'='*60}")
    t0 = time.time()
    eval_counter[0] = 0

    wslg_model = make_vit()
    wslg_model.load_state_dict(init_model_state)
    wslg_fm = make_fm()
    wslg_fm.load_state_dict(init_fm_state)
    wslg_lgate = LearningGate(n_embd, lg_hidden).to(device)

    wslg_training = {"val_loss": [], "val_acc": []}
    wslg_global_step = 0

    for cycle in range(n_cycles):
        print(f"\n  --- WS_LG Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: CL_LG co-training (FOMAML bilevel) ===
        print(f"  [WAKE] FOMAML bilevel ({wake_steps} steps)")
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

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss, v_acc = eval_model(wslg_model)
                wslg_training["val_loss"].append((wslg_global_step, v_loss))
                wslg_training["val_acc"].append((wslg_global_step, v_acc))
                print(f"    step {wslg_global_step:5d}: "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            wslg_global_step += 1

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
                wslg_training["val_loss"].append((wslg_global_step, v_loss))
                wslg_training["val_acc"].append((wslg_global_step, v_acc))
                print(f"    step {wslg_global_step:5d} (sleep): "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            wslg_global_step += 1

        wslg_model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_gate, student, opt_student
        del wslg_cgate, opt_main, opt_fwd, opt_lg
        torch.cuda.empty_cache()

        # === RE-POINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM (seed={fm_seed})")
        wslg_fm, fm_cos = train_fresh_fm(
            wslg_model, retrain_steps, fm_seed, f"WS_LG c{cycle+1}")

        cycle_step = (cycle + 1) * steps_per_cycle
        v_loss, v_acc = eval_model(wslg_model)
        print(f"  WS_LG CHECKPOINT step {cycle_step}: "
              f"loss={v_loss:.4f} acc={v_acc:.4f}")

    wslg_final_state = {k: v.cpu().clone()
                        for k, v in wslg_model.state_dict().items()}
    del wslg_model, wslg_fm, wslg_lgate
    torch.cuda.empty_cache()
    print(f"  WS_LG done in {time.time() - t0:.0f}s")

    # =================================================================
    # CONDITION 2: WS_UG_uniform (unified gate + uniform local loss)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: WS_UG_uniform ({n_cycles} cycles)")
    print(f"{'='*60}")
    t0 = time.time()
    eval_counter[0] = 0

    wsug_model = make_vit()
    wsug_model.load_state_dict(init_model_state)
    wsug_fm = make_fm()
    wsug_fm.load_state_dict(init_fm_state)
    wsug_gate = UnifiedGate(n_embd, lg_hidden).to(device)

    wsug_training = {"val_loss": [], "val_acc": []}
    wsug_global_step = 0

    for cycle in range(n_cycles):
        print(f"\n  --- WS_UG_uniform Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: Unified gate + uniform local loss ===
        print(f"  [WAKE] Unified gate, uniform LL ({wake_steps} steps)")
        main_params = list(wsug_model.parameters()) + list(wsug_gate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            wsug_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            wsug_model.train(); wsug_fm.train(); wsug_gate.train()

            idx = train_indices[wsug_global_step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            fwd_pred_cache = {}

            def ug_cb(act, _cache=fwd_pred_cache):
                fp = wsug_fm(act.detach())
                _cache["pred"] = fp
                inj, gw = wsug_gate(act.detach(), fp.detach())
                _cache["gate_w"] = gw
                return inj

            logits, cls_loss, intermediates = wsug_model(
                images, labels, return_intermediates=True,
                cerebellar_fn=ug_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            fwd_pred = fwd_pred_cache["pred"]
            target = intermediates[predict_to]
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            gate_w = fwd_pred_cache["gate_w"]
            r = target - fwd_pred.detach()
            gw_scalar = gate_w.detach().mean()
            gated_ll = (gw_scalar * r ** 2).mean()

            L_total = cls_loss + lambda_local * gated_ll

            opt_main.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(wsug_fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss, v_acc = eval_model(wsug_model)
                wsug_training["val_loss"].append((wsug_global_step, v_loss))
                wsug_training["val_acc"].append((wsug_global_step, v_acc))
                print(f"    step {wsug_global_step:5d}: "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            wsug_global_step += 1

        # === SLEEP: Distillation ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")

        teacher = make_vit()
        teacher.load_state_dict(wsug_model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(wsug_fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False
        teacher_ugate = UnifiedGate(n_embd, lg_hidden).to(device)
        teacher_ugate.load_state_dict(wsug_gate.state_dict())
        teacher_ugate.eval()
        for p in teacher_ugate.parameters():
            p.requires_grad = False

        student = make_vit()
        student.load_state_dict(wsug_model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            idx = train_indices[wsug_global_step]
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
                wsug_training["val_loss"].append((wsug_global_step, v_loss))
                wsug_training["val_acc"].append((wsug_global_step, v_acc))
                print(f"    step {wsug_global_step:5d} (sleep): "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            wsug_global_step += 1

        wsug_model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_ugate, student, opt_student
        del opt_main, opt_fwd
        torch.cuda.empty_cache()

        # === RE-POINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM (seed={fm_seed})")
        wsug_fm, fm_cos = train_fresh_fm(
            wsug_model, retrain_steps, fm_seed, f"WS_UG c{cycle+1}")

        cycle_step = (cycle + 1) * steps_per_cycle
        v_loss, v_acc = eval_model(wsug_model)
        print(f"  WS_UG_uniform CHECKPOINT step {cycle_step}: "
              f"loss={v_loss:.4f} acc={v_acc:.4f}")

    wsug_final_state = {k: v.cpu().clone()
                        for k, v in wsug_model.state_dict().items()}
    del wsug_model, wsug_fm, wsug_gate
    torch.cuda.empty_cache()
    print(f"  WS_UG_uniform done in {time.time() - t0:.0f}s")

    # =================================================================
    # CONDITION 3: CL (injection only, continuous, no distillation)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 3: CL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")
    t0 = time.time()
    eval_counter[0] = 0

    cl_model = make_vit()
    cl_model.load_state_dict(init_model_state)
    cl_fm = make_fm()
    cl_fm.load_state_dict(init_fm_state)
    cl_cgate = CerebellarGate(n_embd).to(device)

    cl_main_params = list(cl_model.parameters()) + list(cl_cgate.parameters())
    cl_opt_main = torch.optim.AdamW(cl_main_params, lr=lr, weight_decay=0.01)
    cl_opt_fwd = torch.optim.AdamW(
        cl_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

    cl_training = {"val_loss": [], "val_acc": []}

    for step in range(total_main_steps):
        cl_model.train(); cl_fm.train(); cl_cgate.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        fwd_pred_cache = {}

        def cl_cb(act, _cache=fwd_pred_cache):
            fp = cl_fm(act.detach())
            _cache["pred"] = fp
            return cl_cgate(fp.detach())

        logits, cls_loss, intermediates = cl_model(
            images, labels, return_intermediates=True,
            cerebellar_fn=cl_cb,
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

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(cl_model)
            cl_training["val_loss"].append((step, v_loss))
            cl_training["val_acc"].append((step, v_acc))
            print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f} "
                  f"gate={cl_cgate.injection_norm():.3f}")

    cl_final_state = {k: v.cpu().clone()
                      for k, v in cl_model.state_dict().items()}
    del cl_model, cl_fm, cl_cgate, cl_opt_main, cl_opt_fwd
    torch.cuda.empty_cache()
    print(f"  CL done in {time.time() - t0:.0f}s")

    # =================================================================
    # CONDITION 4: OL (open-loop baseline)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 4: OL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")
    t0 = time.time()
    eval_counter[0] = 0

    ol_model = make_vit()
    ol_model.load_state_dict(init_model_state)
    ol_opt = torch.optim.AdamW(ol_model.parameters(), lr=lr, weight_decay=0.01)

    ol_training = {"val_loss": [], "val_acc": []}

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

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(ol_model)
            ol_training["val_loss"].append((step, v_loss))
            ol_training["val_acc"].append((step, v_acc))
            print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f}")

    ol_final_state = {k: v.cpu().clone()
                      for k, v in ol_model.state_dict().items()}
    del ol_model, ol_opt
    torch.cuda.empty_cache()
    print(f"  OL done in {time.time() - t0:.0f}s")

    # =================================================================
    # ADAPTATION TEST
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  ADAPTATION TEST")
    print(f"{'='*60}")

    # Pre-generate rotated datasets
    print("\nGenerating rotated datasets...")
    rotated_train = {}
    rotated_test = {}
    for angle in rotation_angles:
        t0 = time.time()
        tr_np = train_images[:, 0].numpy()
        te_np = test_images[:, 0].numpy()
        rot_tr = np.stack([
            np.clip(scipy_rotate(tr_np[i], angle, reshape=False,
                                 mode='constant', cval=0.0), 0.0, 1.0)
            for i in range(len(tr_np))
        ])
        rot_te = np.stack([
            np.clip(scipy_rotate(te_np[i], angle, reshape=False,
                                 mode='constant', cval=0.0), 0.0, 1.0)
            for i in range(len(te_np))
        ])
        rotated_train[angle] = torch.from_numpy(rot_tr).float().unsqueeze(1)
        rotated_test[angle] = torch.from_numpy(rot_te).float().unsqueeze(1)
        print(f"  {angle:3d} deg: {time.time() - t0:.1f}s")

    # Shared adaptation batch indices
    adapt_gen = torch.Generator().manual_seed(seed + 500)
    adapt_train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=adapt_gen)
        for _ in range(adapt_steps)
    ]

    final_states = {
        "WS_LG": wslg_final_state,
        "WS_UG_uniform": wsug_final_state,
        "CL": cl_final_state,
        "OL": ol_final_state,
    }

    # ID performance
    print("\n--- ID performance (standalone, no injection) ---")
    id_perf = {}
    for cond in conditions:
        model = make_vit()
        model.load_state_dict(final_states[cond])
        loss, acc = eval_full(model, test_images, test_labels)
        id_perf[cond] = {"loss": loss, "acc": acc}
        print(f"  {cond:>15s}: loss={loss:.4f} acc={acc:.4f}")
        del model
        torch.cuda.empty_cache()

    # Adaptation
    all_results = {}

    for angle in rotation_angles:
        print(f"\n{'='*50}")
        print(f"  ROTATION: {angle} deg")
        print(f"{'='*50}")

        rot_tr = rotated_train[angle]
        rot_te = rotated_test[angle]
        angle_results = {}

        for cond in conditions:
            print(f"\n  [{cond}]")
            model = make_vit()
            model.load_state_dict(final_states[cond])

            # Zero-shot OOD eval
            zs_loss, zs_acc = eval_full(model, rot_te, test_labels)
            print(f"    Zero-shot: loss={zs_loss:.4f} acc={zs_acc:.3f}")

            # Fine-tune on rotated training set
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=adapt_lr, weight_decay=0.01)

            curve = [(0, zs_loss, zs_acc)]

            for step in range(1, adapt_steps + 1):
                model.train()
                idx = adapt_train_indices[step - 1]
                imgs = rot_tr[idx].to(device)
                labs = train_labels[idx].to(device)

                logits, loss = model(imgs, labs)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % adapt_eval_interval == 0 or step == adapt_steps:
                    v_loss, v_acc = eval_full(model, rot_te, test_labels)
                    curve.append((step, v_loss, v_acc))
                    if step % 100 == 0 or step == adapt_steps:
                        print(f"    step {step:4d}: "
                              f"loss={v_loss:.4f} acc={v_acc:.3f}")

            # Catastrophic forgetting
            id_loss_post, id_acc_post = eval_full(
                model, test_images, test_labels)
            id_drop = id_acc_post - id_perf[cond]["acc"]
            print(f"    Forgetting: ID acc={id_acc_post:.3f} "
                  f"(delta={id_drop:+.3f})")

            # Summary metrics
            steps_arr = [c[0] for c in curve]
            accs_arr = [c[2] for c in curve]
            auc = float(np.trapz(accs_arr, x=steps_arr) / adapt_steps)

            target_acc = id_perf[cond]["acc"] * 0.90
            steps_to_90 = None
            for s, _, a in curve:
                if a >= target_acc:
                    steps_to_90 = s
                    break

            angle_results[cond] = {
                "zero_shot": {"loss": zs_loss, "acc": zs_acc},
                "final": {"loss": curve[-1][1], "acc": curve[-1][2]},
                "learning_curve": curve,
                "forgetting": {
                    "id_loss": id_loss_post,
                    "id_acc": id_acc_post,
                    "id_drop": id_drop,
                },
                "auc": auc,
                "steps_to_90pct_id": steps_to_90,
            }

            del model, optimizer
            torch.cuda.empty_cache()

        all_results[angle] = angle_results

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_ratchet_adaptation"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    for cond, state in final_states.items():
        fname = cond.lower().replace("_uniform", "u") + "_model.pt"
        torch.save(state, os.path.join(save_root, fname))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "lr": lr, "fwd_lr": fwd_lr, "gate_lr": gate_lr,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "lambda_local": lambda_local, "lg_hidden": lg_hidden,
            "adapt_steps": adapt_steps, "adapt_lr": adapt_lr,
            "rotation_angles": rotation_angles,
            "conditions": conditions,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "training": {
            "WS_LG": wslg_training,
            "WS_UG_uniform": wsug_training,
            "CL": cl_training,
            "OL": ol_training,
        },
        "id_performance": id_perf,
        "adaptation": {str(k): v for k, v in all_results.items()},
    }

    with open(os.path.join(save_root, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  ID performance (standalone):")
    for c in conditions:
        print(f"    {c:>15s}: loss={id_perf[c]['loss']:.4f} "
              f"acc={id_perf[c]['acc']:.4f}")

    header = (f"  {'Angle':>6s} | "
              + " | ".join(f"{c:>15s}" for c in conditions))
    sep = (f"  {'-'*6}-+-"
           + "-+-".join(["-" * 15] * len(conditions)))

    print(f"\n  Zero-shot OOD accuracy:")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["zero_shot"]["acc"] for c in conditions]
        print(f"  {angle:5d}  | "
              + " | ".join(f"{v:15.4f}" for v in vals))

    print(f"\n  Adaptation AUC:")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["auc"] for c in conditions]
        print(f"  {angle:5d}  | "
              + " | ".join(f"{v:15.4f}" for v in vals))

    print(f"\n  Post-adaptation accuracy ({adapt_steps} steps):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["final"]["acc"] for c in conditions]
        print(f"  {angle:5d}  | "
              + " | ".join(f"{v:15.4f}" for v in vals))

    print(f"\n  Steps to 90% of ID accuracy:")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = []
        for c in conditions:
            s = all_results[angle][c]["steps_to_90pct_id"]
            vals.append(f"{s:15d}" if s is not None
                        else f"   >{adapt_steps:>11d}")
        print(f"  {angle:5d}  | " + " | ".join(vals))

    print(f"\n  Forgetting (ID accuracy drop after adaptation):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["forgetting"]["id_drop"]
                for c in conditions]
        print(f"  {angle:5d}  | "
              + " | ".join(f"{v:+15.4f}" for v in vals))

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    adapt_steps: int = 500,
    adapt_lr: float = 1e-4,
    seed: int = 42,
):
    result = a2a_mnist_ratchet_adaptation.remote(
        n_cycles=n_cycles,
        adapt_steps=adapt_steps,
        adapt_lr=adapt_lr,
        seed=seed,
    )
    print("\nRatchet adaptation experiment complete.")
