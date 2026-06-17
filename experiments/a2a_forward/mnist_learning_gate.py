"""MNIST learning gate experiment.

Bilevel optimization: a learned gate controls which dimensions of the FM
prediction error get used as local learning signals. The gate is trained
to minimize classification loss after a virtual model update (MAML-style),
so it discovers which compression directions help classification.

Motivation: the precision-weighted local loss (mnist_precision_weighted.py)
failed because it used FM error statistics to decide what to compress —
those directions are unrelated to task relevance. The learning gate uses
the task loss itself: if compressing direction d hurts classification after
one update step, the gate learns to exempt it.

Six conditions with identical seed/lr/init:
  OL:    open-loop baseline
  CL:    closed-loop injection only
  LL:    raw MSE local loss (no injection)
  CL_LL: injection + raw MSE local loss
  LG:    learning-gated local loss (no injection)
  CL_LG: injection + learning-gated local loss

The bilevel training procedure (LG/CL_LG conditions):
  1. Forward pass → cls_loss, gated_local_loss
  2. Inner gradient: g = ∇_θ(cls_loss + λ·gated_local_loss), create_graph=True
  3. Virtual update: θ' = θ - lr·g
  4. Meta forward with θ' via functional_call → cls_loss'
  5. Gate update: ∂cls_loss'/∂gate_params (flows through θ' → g → gate_w → gate)
  6. Model update: apply g.detach()
  7. FM update: independent

Because gate_w enters L_total linearly, the "second-order" terms are just
mixed partials ∂²L/(∂θ ∂gate), not the model Hessian — cheaper than MAML.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_mnist_learning_gate(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    gate_lr: float = 1e-3,
    n_steps: int = 5000,
    eval_interval: int = 100,
    n_eval_batches: int = 5,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    seed: int = 42,
    probe_batches: int = 40,
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

    print(f"MNIST LEARNING GATE EXPERIMENT on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch_size={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  lambda_local={lambda_local}, lg_hidden={lg_hidden}, gate_lr={gate_lr}")

    # -----------------------------------------------------------------
    # Learning Gate
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
    print("Loading MNIST...")
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
    n_positions = (28 // patch_size) ** 2 + 1

    # -----------------------------------------------------------------
    # Pre-generate batch indices (shared across all conditions)
    # -----------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)

    n_evals = n_steps // eval_interval + 2
    train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=train_gen)
        for _ in range(n_steps)
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

    # -----------------------------------------------------------------
    # Initialize shared weights
    # -----------------------------------------------------------------
    torch.manual_seed(seed)
    init_model = ViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    ).to(device)
    init_fwd = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
        block_size=n_positions, causal=False,
    ).to(device)

    init_model_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
    init_fwd_state = {k: v.cpu().clone() for k, v in init_fwd.state_dict().items()}
    del init_model, init_fwd
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =================================================================
    # Shared eval helper
    # =================================================================
    def eval_model(model, fwd_model, cgate, eval_idx):
        """Evaluate val loss, accuracy, FM metrics. Returns dict of scalars."""
        model.eval()
        fwd_model.eval()
        if cgate:
            cgate.eval()

        v_loss, v_acc = 0.0, 0.0
        v_loss_noinj, v_acc_noinj = 0.0, 0.0
        v_fwd, v_cos, v_res = 0.0, 0.0, 0.0

        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                vim = test_images[eidx].to(device)
                vlb = test_labels[eidx].to(device)

                if cgate is not None:
                    def eval_cb(act):
                        return cgate(fwd_model(act))
                    vlog, vl, vi = model(
                        vim, vlb, return_intermediates=True,
                        cerebellar_fn=eval_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    v_loss += vl.item()
                    v_acc += (vlog.argmax(-1) == vlb).float().mean().item()

                    vlog2, vl2, vi2 = model(vim, vlb, return_intermediates=True)
                    v_loss_noinj += vl2.item()
                    v_acc_noinj += (vlog2.argmax(-1) == vlb).float().mean().item()
                    src = vi2[predict_from]
                    tgt = vi2[predict_to]
                else:
                    vlog, vl, vi = model(vim, vlb, return_intermediates=True)
                    v_loss += vl.item()
                    v_acc += (vlog.argmax(-1) == vlb).float().mean().item()
                    src = vi[predict_from]
                    tgt = vi[predict_to]

                pred = fwd_model(src)
                v_fwd += F.mse_loss(pred, tgt).item()
                v_cos += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                v_res += (tgt - pred).norm(dim=-1).mean().item()

        n = n_eval_batches
        out = {
            "val_loss": v_loss / n, "val_acc": v_acc / n,
            "val_fwd_mse": v_fwd / n, "val_cosine": v_cos / n,
            "val_residual_norm": v_res / n,
        }
        if cgate is not None:
            out["val_loss_no_inj"] = v_loss_noinj / n
            out["val_acc_no_inj"] = v_acc_noinj / n
            out["cgate_norm"] = cgate.injection_norm()
        return out

    # =================================================================
    # Standard training (OL, CL, LL, CL_LL)
    # =================================================================
    def train_standard(name, use_injection, use_local_loss):
        print(f"\n{'='*60}")
        print(f"  Training {name} "
              f"(injection={use_injection}, local_loss={use_local_loss})")
        print(f"{'='*60}")

        model = ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)
        model.load_state_dict(init_model_state)

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)

        cgate = None
        if use_injection:
            cgate = CerebellarGate(n_embd).to(device)
            main_params = list(model.parameters()) + list(cgate.parameters())
        else:
            main_params = list(model.parameters())

        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
        )

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "fwd_mse": [], "val_fwd_mse": [], "val_cosine": [],
            "val_residual_norm": [],
        }
        if use_injection:
            history.update({"val_loss_no_inj": [], "val_acc_no_inj": [],
                            "cgate_norm": []})
        if use_local_loss:
            history["local_loss"] = []

        eval_idx = 0

        for step in range(n_steps):
            model.train(); fwd_model.train()
            if cgate:
                cgate.train()

            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            if use_injection:
                fwd_pred_cache = {}
                def cerebellar_fn(act):
                    fp = fwd_model(act.detach())
                    fwd_pred_cache["pred"] = fp
                    return cgate(fp.detach())
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                fwd_pred = fwd_pred_cache["pred"]
            else:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                )
                fwd_pred = fwd_model(intermediates[predict_from].detach())

            target = intermediates[predict_to]
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            total_loss = cls_loss
            if use_local_loss:
                local_loss_val = F.mse_loss(fwd_pred.detach(), target)
                total_loss = total_loss + lambda_local * local_loss_val

            opt_main.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()

            if step % eval_interval == 0 or step == n_steps - 1:
                ev = eval_model(model, fwd_model, cgate, eval_idx)
                history["train_loss"].append((step, cls_loss.item()))
                history["train_acc"].append((step, acc))
                for k in ["val_loss", "val_acc", "val_fwd_mse",
                           "val_cosine", "val_residual_norm"]:
                    history[k].append((step, ev[k]))
                extra = ""
                if use_injection:
                    history["val_loss_no_inj"].append(
                        (step, ev["val_loss_no_inj"]))
                    history["val_acc_no_inj"].append(
                        (step, ev["val_acc_no_inj"]))
                    history["cgate_norm"].append((step, ev["cgate_norm"]))
                    extra += (f" inj_delta="
                              f"{ev['val_loss'] - ev['val_loss_no_inj']:+.4f}"
                              f" gate={ev['cgate_norm']:.3f}")
                if use_local_loss:
                    history["local_loss"].append(
                        (step, local_loss_val.item()))
                    extra += f" ll={local_loss_val.item():.5f}"
                print(f"  [{name:5s}] step {step:5d}: "
                      f"loss={ev['val_loss']:.4f} acc={ev['val_acc']:.4f}"
                      f"{extra} | fwd_cos={ev['val_cosine']:.4f}")
                eval_idx += 1

        return model, fwd_model, cgate, history

    # =================================================================
    # Gated training (LG, CL_LG) — bilevel optimization
    # =================================================================
    def train_gated(name, use_injection):
        print(f"\n{'='*60}")
        print(f"  Training {name} "
              f"(injection={use_injection}, learning_gate=True)")
        print(f"{'='*60}")

        model = ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)
        model.load_state_dict(init_model_state)

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)

        cgate = None
        if use_injection:
            cgate = CerebellarGate(n_embd).to(device)
            main_params = list(model.parameters()) + list(cgate.parameters())
        else:
            main_params = list(model.parameters())

        lgate = LearningGate(n_embd, lg_hidden).to(device)

        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
        )
        opt_lg = torch.optim.AdamW(
            lgate.parameters(), lr=gate_lr, weight_decay=0.01,
        )

        n_model_params = len(list(model.parameters()))

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "fwd_mse": [], "val_fwd_mse": [], "val_cosine": [],
            "val_residual_norm": [],
            "local_loss": [], "meta_loss": [],
            "gate_weight_mean": [], "gate_weight_std": [],
            "gate_weight_per_dim": [],
        }
        if use_injection:
            history.update({"val_loss_no_inj": [], "val_acc_no_inj": [],
                            "cgate_norm": []})

        eval_idx = 0

        for step in range(n_steps):
            model.train(); fwd_model.train(); lgate.train()
            if cgate:
                cgate.train()

            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            # --- Phase 1: Forward pass ---
            if use_injection:
                fwd_pred_cache = {}
                def cerebellar_fn(act):
                    fp = fwd_model(act.detach())
                    fwd_pred_cache["pred"] = fp
                    return cgate(fp.detach())
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                fwd_pred = fwd_pred_cache["pred"]
            else:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                )
                fwd_pred = fwd_model(intermediates[predict_from].detach())

            target = intermediates[predict_to]
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            # --- Phase 2: Learning gate weights ---
            source_cls = intermediates[predict_from][:, 0].detach()
            fm_error_cls = (target - fwd_pred).detach()[:, 0]
            gate_w = lgate(source_cls, fm_error_cls)  # (B, D)

            # --- Phase 3: Gated local loss ---
            r = target - fwd_pred.detach()
            gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()
            L_total = cls_loss + lambda_local * gated_ll

            # --- Phase 4: Inner gradient with create_graph ---
            inner_grads = torch.autograd.grad(
                L_total, main_params, create_graph=True,
            )
            grads_for_update = [g.detach().clone() for g in inner_grads]

            # --- Phase 5: Virtual parameters ---
            virtual_model_params = {
                pname: p - lr * g
                for (pname, p), g in zip(
                    model.named_parameters(),
                    inner_grads[:n_model_params],
                )
            }
            if use_injection:
                virtual_cgate_params = {
                    pname: p - lr * g
                    for (pname, p), g in zip(
                        cgate.named_parameters(),
                        inner_grads[n_model_params:],
                    )
                }

            # --- Phase 6: Meta forward pass ---
            if use_injection:
                def meta_cerebellar_fn(act):
                    fp = fwd_model(act.detach())
                    return functional_call(
                        cgate, virtual_cgate_params, (fp.detach(),),
                    )
                meta_out = functional_call(
                    model, virtual_model_params,
                    args=(images,),
                    kwargs={
                        "targets": labels,
                        "cerebellar_fn": meta_cerebellar_fn,
                        "cerebellar_input_block": cerebellar_input_block,
                        "cerebellar_inject_block": inject_after_block,
                    },
                )
            else:
                meta_out = functional_call(
                    model, virtual_model_params,
                    args=(images,),
                    kwargs={"targets": labels},
                )
            cls_loss_prime = meta_out[1]

            # --- Phase 7: Gate update (meta-gradient) ---
            opt_lg.zero_grad()
            cls_loss_prime.backward()
            torch.nn.utils.clip_grad_norm_(lgate.parameters(), 1.0)
            opt_lg.step()

            # --- Phase 8: Model update (inner gradient) ---
            opt_main.zero_grad()
            for p, g in zip(main_params, grads_for_update):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # --- Phase 9: FM update ---
            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                ev = eval_model(model, fwd_model, cgate, eval_idx)

                # Gate weight statistics on eval data
                gw_all = []
                with torch.no_grad():
                    for bi in range(n_eval_batches):
                        eidx = eval_indices[eval_idx][bi]
                        vim = test_images[eidx].to(device)
                        _, _, vi = model(vim, return_intermediates=True)
                        src = vi[predict_from]
                        tgt = vi[predict_to]
                        pred = fwd_model(src)
                        gw = lgate(src[:, 0], (tgt - pred)[:, 0])
                        gw_all.append(gw.cpu())
                gw_cat = torch.cat(gw_all, dim=0)
                gw_mean = gw_cat.mean().item()
                gw_std = gw_cat.std().item()
                gw_per_dim = gw_cat.mean(dim=0).tolist()

                history["train_loss"].append((step, cls_loss.item()))
                history["train_acc"].append((step, acc))
                for k in ["val_loss", "val_acc", "val_fwd_mse",
                           "val_cosine", "val_residual_norm"]:
                    history[k].append((step, ev[k]))
                history["local_loss"].append((step, gated_ll.item()))
                history["meta_loss"].append((step, cls_loss_prime.item()))
                history["gate_weight_mean"].append((step, gw_mean))
                history["gate_weight_std"].append((step, gw_std))
                history["gate_weight_per_dim"].append((step, gw_per_dim))
                if use_injection:
                    history["val_loss_no_inj"].append(
                        (step, ev["val_loss_no_inj"]))
                    history["val_acc_no_inj"].append(
                        (step, ev["val_acc_no_inj"]))
                    history["cgate_norm"].append((step, ev["cgate_norm"]))

                extra = (f" gw={gw_mean:.3f}±{gw_std:.3f}"
                         f" meta={cls_loss_prime.item():.4f}")
                if use_injection:
                    extra += (
                        f" inj_delta="
                        f"{ev['val_loss'] - ev['val_loss_no_inj']:+.4f}"
                        f" gate={ev['cgate_norm']:.3f}")
                print(f"  [{name:5s}] step {step:5d}: "
                      f"loss={ev['val_loss']:.4f} acc={ev['val_acc']:.4f}"
                      f"{extra} | fwd_cos={ev['val_cosine']:.4f}")
                eval_idx += 1

        return model, fwd_model, cgate, lgate, history

    # =================================================================
    # Train all 6 conditions
    # =================================================================
    standard_conditions = [
        ("OL", False, False),
        ("CL", True, False),
        ("LL", False, True),
        ("CL_LL", True, True),
    ]
    gated_conditions = [
        ("LG", False),
        ("CL_LG", True),
    ]

    trained = {}
    histories = {}
    lgates = {}

    for cname, inj, ll in standard_conditions:
        model, fm, cgate, hist = train_standard(cname, inj, ll)
        trained[cname] = (model, fm, cgate)
        histories[cname] = hist

    for cname, inj in gated_conditions:
        model, fm, cgate, lgate, hist = train_gated(cname, inj)
        trained[cname] = (model, fm, cgate)
        lgates[cname] = lgate
        histories[cname] = hist

    cond_names = ["OL", "CL", "LL", "CL_LL", "LG", "CL_LG"]

    # =================================================================
    # Self-knowledge probes
    # =================================================================
    print(f"\n{'='*60}")
    print("  SELF-KNOWLEDGE PROBES")
    print(f"{'='*60}")

    layer_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    probe_results = {}

    for cname, (model, fwd_model, cgate) in trained.items():
        model.eval(); fwd_model.eval()

        layer_acts = {k: [] for k in layer_names}
        residual_vecs = []

        with torch.no_grad():
            for bi in range(probe_batches):
                pidx = probe_indices[bi]
                pim = test_images[pidx].to(device)
                _, _, vi = model(pim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                residual_vecs.append((tgt - pred).cpu())
                for key in layer_names:
                    layer_acts[key].append(vi[key].cpu())

        all_residual = torch.cat(residual_vecs, dim=0)
        for key in layer_names:
            layer_acts[key] = torch.cat(layer_acts[key], dim=0)

        residual_flat = all_residual.reshape(-1, n_embd)
        n_total = residual_flat.shape[0]
        n_train_probe = int(0.8 * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed),
        )
        tr = perm[:n_train_probe]
        te = perm[n_train_probe:]

        layer_r2 = {}
        for lname in layer_names:
            acts_flat = layer_acts[lname].reshape(-1, n_embd)
            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            X_tr = acts_flat[tr].to(device)
            Y_tr = residual_flat[tr].to(device)
            X_te = acts_flat[te].to(device)
            Y_te = residual_flat[te].to(device)

            probe_bs = min(4096, n_train_probe)
            for _ in range(300):
                sidx = torch.randint(n_train_probe, (probe_bs,))
                ploss = F.mse_loss(probe(X_tr[sidx]), Y_tr[sidx])
                opt.zero_grad(); ploss.backward(); opt.step()

            with torch.no_grad():
                mse = F.mse_loss(probe(X_te), Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            layer_r2[lname] = r2
            print(f"  {cname:5s} | {lname:12s}: R2 = {r2:.4f}")

        probe_results[cname] = {"layer_r2": layer_r2}

    # =================================================================
    # Robustness test
    # =================================================================
    print(f"\n{'='*60}")
    print("  ROBUSTNESS TEST")
    print(f"{'='*60}")

    perturbation_strengths = [0.1, 0.5, 1.0, 2.0]
    rob_results = {}

    for cname, (model, fwd_model, cgate) in trained.items():
        model.eval(); fwd_model.eval()
        if cgate:
            cgate.eval()

        eps_results = {}
        for eps in perturbation_strengths:
            loss_deltas = []
            n_rob_batches = min(probe_batches, 20)

            with torch.no_grad():
                for bi in range(n_rob_batches):
                    pidx = probe_indices[bi]
                    pim = test_images[pidx].to(device)
                    plb = test_labels[pidx].to(device)

                    if cgate:
                        def cb(act):
                            return cgate(fwd_model(act))
                        _, base_loss, _ = model(
                            pim, plb, return_intermediates=True,
                            cerebellar_fn=cb,
                            cerebellar_input_block=cerebellar_input_block,
                            cerebellar_inject_block=inject_after_block,
                        )
                    else:
                        _, base_loss, _ = model(
                            pim, plb, return_intermediates=True,
                        )

                    torch.manual_seed(seed + bi + 1000)
                    noise = torch.randn(
                        batch_size, n_positions, n_embd, device=device,
                    ) * eps

                    if cgate:
                        def cb_p(act):
                            return cgate(fwd_model(act))
                        _, pert_loss, _ = model(
                            pim, plb, return_intermediates=True,
                            cerebellar_fn=cb_p,
                            cerebellar_input_block=cerebellar_input_block,
                            cerebellar_inject_block=inject_after_block,
                            perturbation=(inject_after_block, noise),
                        )
                    else:
                        _, pert_loss, _ = model(
                            pim, plb, return_intermediates=True,
                            perturbation=(inject_after_block, noise),
                        )

                    loss_deltas.append(pert_loss.item() - base_loss.item())

            mean_delta = float(np.mean(loss_deltas))
            eps_results[str(eps)] = mean_delta
            print(f"  {cname:5s} | eps={eps:.1f}: "
                  f"delta_loss = {mean_delta:+.4f}")

        rob_results[cname] = eps_results

    # =================================================================
    # Gate weight analysis
    # =================================================================
    print(f"\n{'='*60}")
    print("  GATE WEIGHT ANALYSIS")
    print(f"{'='*60}")

    gate_analysis = {}

    for cname in ["LG", "CL_LG"]:
        model, fwd_model, cgate = trained[cname]
        lgate = lgates[cname]
        model.eval(); fwd_model.eval(); lgate.eval()

        all_gate_w = []
        all_labels_g = []
        all_fm_errors = []

        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                pim = test_images[pidx].to(device)
                plb = test_labels[pidx]
                _, _, vi = model(pim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                err = tgt - pred

                gw = lgate(src[:, 0], err[:, 0])
                all_gate_w.append(gw.cpu())
                all_labels_g.append(plb)
                all_fm_errors.append(err[:, 0].cpu())

        gw_cat = torch.cat(all_gate_w, dim=0).numpy()  # (N, D)
        lab_cat = torch.cat(all_labels_g, dim=0).numpy()
        err_cat = torch.cat(all_fm_errors, dim=0).numpy()

        dim_mean = gw_cat.mean(axis=0)
        dim_std = gw_cat.std(axis=0)
        input_variance = dim_std.mean()

        # Per-dimension FM error variance (what precision weighting uses)
        err_var_per_dim = err_cat.var(axis=0)  # (D,)

        # Per-dimension digit discriminability (eta^2)
        eta2_per_dim = np.zeros(n_embd)
        for d in range(n_embd):
            vals = err_cat[:, d]
            grand_mean = vals.mean()
            ss_total = ((vals - grand_mean) ** 2).sum()
            ss_between = 0.0
            for digit in range(10):
                mask = lab_cat == digit
                if mask.sum() > 0:
                    gm = vals[mask].mean()
                    ss_between += mask.sum() * (gm - grand_mean) ** 2
            eta2_per_dim[d] = ss_between / (ss_total + 1e-8)

        # Correlations
        corr_gate_errvar = float(np.corrcoef(dim_mean, err_var_per_dim)[0, 1])
        corr_gate_eta2 = float(np.corrcoef(dim_mean, eta2_per_dim)[0, 1])

        analysis = {
            "dim_mean_gate_weight": dim_mean.tolist(),
            "dim_std_gate_weight": dim_std.tolist(),
            "mean_input_variance": float(input_variance),
            "corr_gate_vs_fm_error_var": corr_gate_errvar,
            "corr_gate_vs_digit_eta2": corr_gate_eta2,
            "overall_mean": float(gw_cat.mean()),
            "overall_std": float(gw_cat.std()),
            "min_dim_mean": float(dim_mean.min()),
            "max_dim_mean": float(dim_mean.max()),
        }
        gate_analysis[cname] = analysis

        print(f"  {cname:5s}:")
        print(f"    Gate weights: mean={gw_cat.mean():.3f} "
              f"std={gw_cat.std():.3f} "
              f"range=[{dim_mean.min():.3f}, {dim_mean.max():.3f}]")
        print(f"    Input variance (mean per-dim std): "
              f"{input_variance:.4f}")
        print(f"    Corr(gate, FM error var): {corr_gate_errvar:+.3f}")
        print(f"    Corr(gate, digit eta2):   {corr_gate_eta2:+.3f}")

    # =================================================================
    # Save results
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_learning_gate"
                f"/{model_tag}/{gap_tag}/lambda_{lambda_local}")
    os.makedirs(save_dir, exist_ok=True)

    for cname, (model, fm, cgate) in trained.items():
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"{cname}_model.pt"))
        torch.save(fm.state_dict(),
                   os.path.join(save_dir, f"{cname}_fwd.pt"))
        if cgate:
            torch.save(cgate.state_dict(),
                       os.path.join(save_dir, f"{cname}_cgate.pt"))
    for cname, lgate in lgates.items():
        torch.save(lgate.state_dict(),
                   os.path.join(save_dir, f"{cname}_lgate.pt"))

    result = {
        "config": {
            "dataset": "mnist",
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "patch_size": patch_size, "n_positions": n_positions,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "lr": lr, "fwd_lr": fwd_lr, "gate_lr": gate_lr,
            "n_steps": n_steps, "batch_size": batch_size, "seed": seed,
            "lambda_local": lambda_local, "lg_hidden": lg_hidden,
        },
    }

    for cname in cond_names:
        hist = histories[cname]
        cond_result = {
            "history": hist,
            "final_val_loss": hist["val_loss"][-1][1],
            "final_val_acc": hist["val_acc"][-1][1],
            "final_fwd_cosine": hist["val_cosine"][-1][1],
            "final_fwd_mse": hist["val_fwd_mse"][-1][1],
            "final_residual_norm": hist["val_residual_norm"][-1][1],
        }
        if "val_loss_no_inj" in hist:
            cond_result["final_val_loss_no_inj"] = (
                hist["val_loss_no_inj"][-1][1])
            cond_result["final_val_acc_no_inj"] = (
                hist["val_acc_no_inj"][-1][1])
            cond_result["final_cgate_norm"] = hist["cgate_norm"][-1][1]
        result[cname] = cond_result

    result["probes"] = probe_results
    result["robustness"] = rob_results
    result["gate_analysis"] = gate_analysis

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Val accuracy at key steps:")
    for cname in cond_names:
        accs = dict(histories[cname]["val_acc"])
        steps_show = [s for s in [0, 500, 1000, 2500, n_steps - 1]
                      if s in accs]
        acc_str = "  ".join(f"s{s}={accs[s]:.4f}" for s in steps_show)
        print(f"    {cname:5s}: {acc_str}")

    print(f"\n  Final metrics:")
    print(f"    {'Cond':5s} {'val_loss':>9s} {'val_acc':>8s} "
          f"{'fwd_cos':>8s}")
    for cname in cond_names:
        r = result[cname]
        print(f"    {cname:5s} {r['final_val_loss']:>9.4f} "
              f"{r['final_val_acc']:>8.4f} "
              f"{r['final_fwd_cosine']:>8.4f}")

    print(f"\n  Dependency (val_loss_no_inj - OL_val_loss):")
    ol_loss = result["OL"]["final_val_loss"]
    for cname in ["CL", "CL_LL", "CL_LG"]:
        if "final_val_loss_no_inj" in result[cname]:
            noinj = result[cname]["final_val_loss_no_inj"]
            dep = noinj - ol_loss
            inj_benefit = result[cname]["final_val_loss"] - noinj
            print(f"    {cname:5s}: dep={dep:+.4f}  "
                  f"inj_benefit={inj_benefit:+.4f}")

    print(f"\n  Self-knowledge probes (R2 at key layers):")
    for layer in ["post_block0", f"post_block{n_layer - 1}"]:
        r2s = {c: probe_results[c]["layer_r2"].get(layer, 0)
               for c in cond_names}
        print(f"    {layer:12s}: " +
              "  ".join(f"{c}={r2s[c]:.3f}" for c in cond_names))

    print(f"\n  Robustness (delta_loss at eps=1.0):")
    ol_rob = rob_results["OL"].get("1.0", 0)
    for cname in cond_names:
        rob = rob_results[cname].get("1.0", 0)
        ratio = rob / ol_rob if ol_rob > 0 else float("nan")
        print(f"    {cname:5s}: delta={rob:+.4f}  ratio={ratio:.3f}")

    print(f"\n  Gate analysis:")
    for cname in ["LG", "CL_LG"]:
        ga = gate_analysis[cname]
        print(f"    {cname:5s}: mean_w={ga['overall_mean']:.3f}  "
              f"input_var={ga['mean_input_variance']:.4f}  "
              f"corr(gate,errvar)={ga['corr_gate_vs_fm_error_var']:+.3f}  "
              f"corr(gate,eta2)={ga['corr_gate_vs_digit_eta2']:+.3f}")

    print(f"\n  COMPARISON WITH RAW MSE (from mnist_local_loss.py):")
    print(f"    LL:    acc=0.972  fwd_cos=0.997  "
          f"rob_ratio=13.4  SK_R2(blk0)=-0.03")
    print(f"    CL_LL: acc=0.969  fwd_cos=0.985  "
          f"rob_ratio=3.84  SK_R2(blk0)=0.76")
    for cname in ["LG", "CL_LG"]:
        r = result[cname]
        rob = rob_results[cname].get("1.0", 0)
        ratio = rob / ol_rob if ol_rob > 0 else float("nan")
        sk = probe_results[cname]["layer_r2"].get("post_block0", 0)
        print(f"    {cname:5s}: acc={r['final_val_acc']:.3f}  "
              f"fwd_cos={r['final_fwd_cosine']:.3f}  "
              f"rob_ratio={ratio:.2f}  SK_R2(blk0)={sk:.2f}")

    return result


@app.local_entrypoint()
def main(
    n_steps: int = 5000,
    lambda_local: float = 1.0,
    gate_lr: float = 1e-3,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_learning_gate.remote(
        n_steps=n_steps,
        lambda_local=lambda_local,
        gate_lr=gate_lr,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nMNIST learning gate experiment complete.")
    print(f"\n  Final metrics:")
    for cname in ["OL", "CL", "LL", "CL_LL", "LG", "CL_LG"]:
        r = result[cname]
        print(f"    {cname:5s}: acc={r['final_val_acc']:.4f} "
              f"fwd_cos={r['final_fwd_cosine']:.4f}")
