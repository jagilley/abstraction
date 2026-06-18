"""MNIST multi-cycle gated ratchet experiment.

Tests whether multi-cycle CL_LG (injection + bilevel-gated local loss) with
periodic distillation + FM reinitialization produces compounding improvement
beyond standard wake-sleep (injection only).

The learning gate persists across cycles — it accumulates a meta-learning policy
for "what makes an error direction worth compressing" that must generalize across
FM reinitializations. This is the core test of the developmental trajectory
hypothesis: Phase 1 builds priors (self-knowledge, innovation map), and the gate
progressively closes as the model matures.

Four conditions, all compute-matched on main-model gradient steps:

1. WS_LG: Wake-sleep with learning gate
   Per cycle: wake_steps CL_LG co-training + distill_steps distillation sleep
   Learning gate persists across cycles. FM + cerebellar gate reinitialized.

2. WS: Standard wake-sleep (injection only, no local loss/gate)
   Per cycle: wake_steps CL co-training + distill_steps distillation sleep

3. CL_LG: CL_LG continuous training (gate + injection, no distillation)
   Same total main-model steps.

4. OL: Open-loop continuous baseline.

Key measurements per cycle:
- Val loss/acc (model standing alone, no injection)
- Robustness (perturbation delta-loss)
- Self-knowledge probes (R^2 for fresh FM residual)
- Gate weight statistics (sparsity evolution across cycles)
- Residual eigenspectrum (effective rank, digit discriminability)
- Innovation migration (PC overlap between successive FMs)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def a2a_mnist_gated_ratchet(
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

    print(f"MNIST GATED RATCHET on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  Checkpoints at: {checkpoint_steps}")
    print(f"  lambda_local={lambda_local}, lg_hidden={lg_hidden}")

    # -----------------------------------------------------------------
    # Learning Gate (same as mnist_learning_gate.py)
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
    # Pre-generate batch indices (shared across all conditions)
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

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def self_knowledge_probes(model, fm, label):
        model.eval(); fm.eval()
        acts_by_layer = {k: [] for k in layer_keys}
        residuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                residuals.append((tgt - pred).reshape(-1, n_embd).cpu())
                for k in layer_keys:
                    acts_by_layer[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(residuals)
        for k in layer_keys:
            acts_by_layer[k] = torch.cat(acts_by_layer[k])

        n_total = all_res.shape[0]
        n_train = int(0.8 * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed))
        tr, te = perm[:n_train], perm[n_train:]

        r2_by_layer = {}
        for lk in layer_keys:
            X = acts_by_layer[lk]
            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            X_tr, Y_tr = X[tr].to(device), all_res[tr].to(device)
            X_te, Y_te = X[te].to(device), all_res[te].to(device)

            bs = min(4096, n_train)
            for _ in range(probe_steps):
                si = torch.randint(n_train, (bs,))
                ploss = F.mse_loss(probe(X_tr[si]), Y_tr[si])
                opt.zero_grad(); ploss.backward(); opt.step()

            with torch.no_grad():
                mse = F.mse_loss(probe(X_te), Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            r2_by_layer[lk] = r2
            del probe, X_tr, Y_tr, X_te, Y_te
            torch.cuda.empty_cache()

        print(f"  [{label}] SK: "
              + " | ".join(f"{k}={v:.3f}" for k, v in r2_by_layer.items()))
        return r2_by_layer

    def compute_residual_stats(model, fm, label):
        """Eigenspectrum and digit discriminability of FM residual."""
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

        # Digit-discriminative eta^2 for top-5 PCs
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

        # Top-5 PC directions (for innovation migration)
        top5_vecs = vecs.copy()

        stats = {
            "eff_rank": eff_rank,
            "top1_pc_frac": float(fracs[0]),
            "top5_pc_frac": float(fracs[:5].sum()),
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "eta_squared_top5": eta_sqs,
            "mean_eta_squared": float(np.mean(eta_sqs)),
            "fwd_cosine": float(
                F.cosine_similarity(
                    torch.cat([per_image_res]),
                    torch.cat([per_image_res]),  # placeholder
                    dim=-1,
                ).mean()
            ),
        }

        # Compute FM cosine properly
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
        """Train a fresh FM on frozen model activations."""
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
        """Cosine overlap between two sets of top-5 PC directions."""
        if vecs_old is None or vecs_new is None:
            return None
        # Canonical alignment: |cos| between each pair
        overlap = np.abs(vecs_old.T @ vecs_new)  # (5, 5)
        diag_cos = [float(overlap[i, i]) for i in range(min(5, overlap.shape[0]))]
        max_cos = [float(overlap[i].max()) for i in range(min(5, overlap.shape[0]))]
        # Subspace overlap: fraction of old top-5 variance captured by new top-5
        proj = vecs_old.T @ vecs_new  # (5, 5)
        subspace_overlap = float(np.trace(proj.T @ proj) / 5.0)
        return {
            "diag_cosines": diag_cos,
            "max_cosines": max_cos,
            "subspace_overlap": subspace_overlap,
        }

    def get_gate_stats(lgate, model, fwd_model):
        """Compute gate weight statistics on eval data."""
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

    # Learning gate persists across cycles
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

            # Forward pass with injection
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

            # Learning gate weights
            source_cls = intermediates[predict_from][:, 0].detach()
            fm_error_cls = (target - fwd_pred).detach()[:, 0]
            gate_w = wslg_lgate(source_cls, fm_error_cls)

            # Gated local loss
            r = target - fwd_pred.detach()
            gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()
            L_total = cls_loss + lambda_local * gated_ll

            # Inner gradient with create_graph
            inner_grads = torch.autograd.grad(
                L_total, main_params, create_graph=True)
            grads_for_update = [g.detach().clone() for g in inner_grads]

            # Virtual parameters
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

            # Meta forward pass
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

            # Gate update (meta-gradient)
            opt_lg.zero_grad()
            cls_loss_prime.backward()
            torch.nn.utils.clip_grad_norm_(wslg_lgate.parameters(), 1.0)
            opt_lg.step()

            # Model update (inner gradient)
            opt_main.zero_grad()
            for p, g in zip(main_params, grads_for_update):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # FM update
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

        # Gate stats at end of wake
        gs = get_gate_stats(wslg_lgate, wslg_model, wslg_fm)
        wslg_history["gate_stats"].append(
            (wslg_global_step, f"post_wake_c{cycle+1}", gs))
        print(f"    Gate: mean={gs['overall_mean']:.3f} "
              f"std={gs['overall_std']:.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f}")

        # Dependency gap at end of wake
        wake_dep_loss, _ = eval_model_with_inj(
            wslg_model, wslg_fm, wslg_cgate)
        wake_noinj_loss, _ = eval_model(wslg_model)
        wake_dep_gap = wake_noinj_loss - wake_dep_loss
        print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}")

        # === SLEEP: Distillation (no gate, no injection in student) ===
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
        ws_sk = self_knowledge_probes(wslg_model, wslg_fm, f"WS_LG c{cycle+1}")
        ws_eigen, top5_vecs = compute_residual_stats(
            wslg_model, wslg_fm, f"WS_LG c{cycle+1}")

        innov_mig = compute_innovation_migration(prev_top5_vecs, top5_vecs)
        prev_top5_vecs = top5_vecs

        # Gate stats on distilled model with fresh FM
        gs_post = get_gate_stats(wslg_lgate, wslg_model, wslg_fm)
        wslg_history["gate_stats"].append(
            (cycle_step, f"post_repoint_c{cycle+1}", gs_post))
        print(f"    Gate (post-repoint): mean={gs_post['overall_mean']:.3f} "
              f"std={gs_post['overall_std']:.3f} "
              f"sparse(<0.1)={gs_post['sparsity_below_0.1']:.3f}")

        wslg_checkpoints[cycle_step] = {
            "val_loss": ws_loss, "val_acc": ws_acc,
            "robustness": ws_rob,
            "self_knowledge": ws_sk,
            "residual_stats": ws_eigen,
            "innovation_migration": innov_mig,
            "gate_stats_post_wake": wslg_history["gate_stats"][-2][2],
            "gate_stats_post_repoint": gs_post,
            "wake_dep_gap": wake_dep_gap,
            "fm_cosine": fm_cos,
        }
        print(f"    loss={ws_loss:.4f} acc={ws_acc:.4f}")

    # =================================================================
    # CONDITION 2: WS (Standard Wake-Sleep, injection only)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: WS ({n_cycles} cycles)")
    print(f"{'='*60}")

    ws_model = make_vit()
    ws_model.load_state_dict(init_model_state)
    ws_fm = make_fm()
    ws_fm.load_state_dict(init_fm_state)

    ws_checkpoints = {}
    ws_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ws_global_step = 0

    for cycle in range(n_cycles):
        print(f"\n  --- WS Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: CL co-training ===
        print(f"  [WAKE] CL co-training ({wake_steps} steps)")
        ws_cgate = CerebellarGate(n_embd).to(device)
        ws_main_params = list(ws_model.parameters()) + list(ws_cgate.parameters())
        ws_opt_main = torch.optim.AdamW(
            ws_main_params, lr=lr, weight_decay=0.01)
        ws_opt_fwd = torch.optim.AdamW(
            ws_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            ws_model.train(); ws_fm.train(); ws_cgate.train()

            idx_step = train_indices[ws_global_step]
            images = train_images[idx_step].to(device)
            labels = train_labels[idx_step].to(device)

            fwd_pred_cache = {}

            def ws_cb(act, _cache=fwd_pred_cache):
                fp = ws_fm(act.detach())
                _cache["pred"] = fp
                return ws_cgate(fp.detach())

            logits, cls_loss, intermediates = ws_model(
                images, labels, return_intermediates=True,
                cerebellar_fn=ws_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )
            target = intermediates[predict_to].detach()
            fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)

            ws_opt_main.zero_grad()
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(ws_main_params, 1.0)
            ws_opt_main.step()

            ws_opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(ws_fm.parameters(), 1.0)
            ws_opt_fwd.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss, v_acc = eval_model(ws_model)
                ws_history["train_loss"].append(
                    (ws_global_step, cls_loss.item()))
                ws_history["train_acc"].append((ws_global_step, acc))
                ws_history["val_loss"].append((ws_global_step, v_loss))
                ws_history["val_acc"].append((ws_global_step, v_acc))
                print(f"    step {ws_global_step:5d} (wake {step}): "
                      f"val={v_loss:.4f} acc={v_acc:.4f}")

            ws_global_step += 1

        wake_dep_loss, _ = eval_model_with_inj(ws_model, ws_fm, ws_cgate)
        wake_noinj_loss, _ = eval_model(ws_model)
        ws_wake_dep = wake_noinj_loss - wake_dep_loss

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
        teacher_gate.load_state_dict(ws_cgate.state_dict())
        teacher_gate.eval()
        for p in teacher_gate.parameters():
            p.requires_grad = False

        student = make_vit()
        student.load_state_dict(ws_model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            idx_step = train_indices[ws_global_step]
            images = train_images[idx_step].to(device)
            labels = train_labels[idx_step].to(device)

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

        ws_model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_gate, student, opt_student
        del ws_cgate, ws_opt_main, ws_opt_fwd
        torch.cuda.empty_cache()

        # === RE-POINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, seed={fm_seed})")
        ws_fm, fm_cos = train_fresh_fm(
            ws_model, retrain_steps, fm_seed, f"WS c{cycle+1}")

        # === CHECKPOINT ===
        cycle_step = (cycle + 1) * steps_per_cycle
        print(f"\n  WS CHECKPOINT at step {cycle_step}")

        ws_l, ws_a = eval_model(ws_model)
        ws_r = measure_robustness(ws_model, f"WS c{cycle+1}")
        ws_s = self_knowledge_probes(ws_model, ws_fm, f"WS c{cycle+1}")
        ws_e, _ = compute_residual_stats(ws_model, ws_fm, f"WS c{cycle+1}")

        ws_checkpoints[cycle_step] = {
            "val_loss": ws_l, "val_acc": ws_a,
            "robustness": ws_r,
            "self_knowledge": ws_s,
            "residual_stats": ws_e,
            "wake_dep_gap": ws_wake_dep,
            "fm_cosine": fm_cos,
        }
        print(f"    loss={ws_l:.4f} acc={ws_a:.4f}")

    # =================================================================
    # CONDITION 3: CL_LG Continuous
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 3: CL_LG CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")

    cllg_model = make_vit()
    cllg_model.load_state_dict(init_model_state)
    cllg_fm = make_fm()
    cllg_fm.load_state_dict(init_fm_state)
    cllg_cgate = CerebellarGate(n_embd).to(device)
    cllg_lgate = LearningGate(n_embd, lg_hidden).to(device)

    cllg_main_params = list(cllg_model.parameters()) + list(cllg_cgate.parameters())
    cllg_n_model_params = len(list(cllg_model.parameters()))
    cllg_opt_main = torch.optim.AdamW(cllg_main_params, lr=lr, weight_decay=0.01)
    cllg_opt_fwd = torch.optim.AdamW(
        cllg_fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    cllg_opt_lg = torch.optim.AdamW(
        cllg_lgate.parameters(), lr=gate_lr, weight_decay=0.01)

    cllg_checkpoints = {}
    cllg_history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "gate_stats": [],
    }

    for step in range(total_main_steps):
        cllg_model.train(); cllg_fm.train()
        cllg_cgate.train(); cllg_lgate.train()

        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        fwd_pred_cache = {}

        def cllg_cb(act, _cache=fwd_pred_cache):
            fp = cllg_fm(act.detach())
            _cache["pred"] = fp
            return cllg_cgate(fp.detach())

        logits, cls_loss, intermediates = cllg_model(
            images, labels, return_intermediates=True,
            cerebellar_fn=cllg_cb,
            cerebellar_input_block=cerebellar_input_block,
            cerebellar_inject_block=inject_after_block,
        )
        fwd_pred = fwd_pred_cache["pred"]
        target = intermediates[predict_to]
        fwd_loss = F.mse_loss(fwd_pred, target.detach())

        source_cls = intermediates[predict_from][:, 0].detach()
        fm_error_cls = (target - fwd_pred).detach()[:, 0]
        gate_w = cllg_lgate(source_cls, fm_error_cls)

        r = target - fwd_pred.detach()
        gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()
        L_total = cls_loss + lambda_local * gated_ll

        inner_grads = torch.autograd.grad(
            L_total, cllg_main_params, create_graph=True)
        grads_for_update = [g.detach().clone() for g in inner_grads]

        virtual_model_params = {
            pname: p - lr * g
            for (pname, p), g in zip(
                cllg_model.named_parameters(),
                inner_grads[:cllg_n_model_params],
            )
        }
        virtual_cgate_params = {
            pname: p - lr * g
            for (pname, p), g in zip(
                cllg_cgate.named_parameters(),
                inner_grads[cllg_n_model_params:],
            )
        }

        def meta_cllg_cb(act):
            fp = cllg_fm(act.detach())
            return functional_call(
                cllg_cgate, virtual_cgate_params, (fp.detach(),))

        meta_out = functional_call(
            cllg_model, virtual_model_params,
            args=(images,),
            kwargs={
                "targets": labels,
                "cerebellar_fn": meta_cllg_cb,
                "cerebellar_input_block": cerebellar_input_block,
                "cerebellar_inject_block": inject_after_block,
            },
        )
        cls_loss_prime = meta_out[1]

        cllg_opt_lg.zero_grad()
        cls_loss_prime.backward()
        torch.nn.utils.clip_grad_norm_(cllg_lgate.parameters(), 1.0)
        cllg_opt_lg.step()

        cllg_opt_main.zero_grad()
        for p, g in zip(cllg_main_params, grads_for_update):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(cllg_main_params, 1.0)
        cllg_opt_main.step()

        cllg_opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(cllg_fm.parameters(), 1.0)
        cllg_opt_fwd.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(cllg_model)
            cllg_history["train_loss"].append((step, cls_loss.item()))
            cllg_history["train_acc"].append((step, acc))
            cllg_history["val_loss"].append((step, v_loss))
            cllg_history["val_acc"].append((step, v_acc))
            print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f} "
                  f"gate_cg={cllg_cgate.injection_norm():.3f} "
                  f"gw={gate_w.mean().item():.3f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  CL_LG CHECKPOINT at step {ckpt_step}")

            cllg_l, cllg_a = eval_model(cllg_model)
            cllg_r = measure_robustness(cllg_model, f"CL_LG step {ckpt_step}")

            # Train a fresh FM for fair SK comparison
            cllg_fresh_fm, cllg_fm_cos = train_fresh_fm(
                cllg_model, retrain_steps, seed + 800 + ckpt_step,
                f"CL_LG s{ckpt_step}")
            cllg_s = self_knowledge_probes(
                cllg_model, cllg_fresh_fm, f"CL_LG s{ckpt_step}")
            cllg_e, _ = compute_residual_stats(
                cllg_model, cllg_fresh_fm, f"CL_LG s{ckpt_step}")

            gs = get_gate_stats(cllg_lgate, cllg_model, cllg_fm)
            cllg_history["gate_stats"].append((ckpt_step, gs))

            inj_loss, _ = eval_model_with_inj(
                cllg_model, cllg_fm, cllg_cgate)
            dep = cllg_l - inj_loss

            cllg_checkpoints[ckpt_step] = {
                "val_loss": cllg_l, "val_acc": cllg_a,
                "robustness": cllg_r,
                "self_knowledge": cllg_s,
                "residual_stats": cllg_e,
                "gate_stats": gs,
                "dependency_gap": dep,
                "fm_cosine": cllg_fm_cos,
            }
            print(f"    loss={cllg_l:.4f} acc={cllg_a:.4f} dep={dep:+.4f} "
                  f"gate_mean={gs['overall_mean']:.3f}")

            del cllg_fresh_fm
            torch.cuda.empty_cache()

    del cllg_opt_main, cllg_opt_fwd, cllg_opt_lg
    torch.cuda.empty_cache()

    # =================================================================
    # CONDITION 4: OL Continuous
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 4: OL CONTINUOUS ({total_main_steps} steps)")
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
            ol_l, ol_a = eval_model(ol_model)
            ol_r = measure_robustness(ol_model, f"OL step {ckpt_step}")

            ol_fresh_fm, ol_fm_cos = train_fresh_fm(
                ol_model, retrain_steps, seed + 700 + ckpt_step,
                f"OL s{ckpt_step}")
            ol_s = self_knowledge_probes(
                ol_model, ol_fresh_fm, f"OL s{ckpt_step}")
            ol_e, _ = compute_residual_stats(
                ol_model, ol_fresh_fm, f"OL s{ckpt_step}")

            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_l, "val_acc": ol_a,
                "robustness": ol_r,
                "self_knowledge": ol_s,
                "residual_stats": ol_e,
                "fm_cosine": ol_fm_cos,
            }
            print(f"    loss={ol_l:.4f} acc={ol_a:.4f}")

            del ol_fresh_fm
            torch.cuda.empty_cache()

    del ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_gated_ratchet"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(wslg_model.state_dict(),
               os.path.join(save_root, "wslg_model.pt"))
    torch.save(wslg_fm.state_dict(),
               os.path.join(save_root, "wslg_fm.pt"))
    torch.save(wslg_lgate.state_dict(),
               os.path.join(save_root, "wslg_lgate.pt"))
    torch.save(ws_model.state_dict(),
               os.path.join(save_root, "ws_model.pt"))
    torch.save(ws_fm.state_dict(),
               os.path.join(save_root, "ws_fm.pt"))
    torch.save(cllg_model.state_dict(),
               os.path.join(save_root, "cllg_model.pt"))
    torch.save(cllg_fm.state_dict(),
               os.path.join(save_root, "cllg_fm.pt"))
    torch.save(cllg_cgate.state_dict(),
               os.path.join(save_root, "cllg_cgate.pt"))
    torch.save(cllg_lgate.state_dict(),
               os.path.join(save_root, "cllg_lgate.pt"))
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
        "ws": {
            "checkpoints": ws_checkpoints,
            "history": ws_history,
        },
        "cl_lg": {
            "checkpoints": cllg_checkpoints,
            "history": cllg_history,
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
    print(f"  {'Step':>6s} | {'WS_LG':>12s} | {'WS':>12s} | "
          f"{'CL_LG':>12s} | {'OL':>12s}")
    for cs in checkpoint_steps:
        wslg_c = wslg_checkpoints[cs]
        ws_c = ws_checkpoints[cs]
        cllg_c = cllg_checkpoints[cs]
        ol_c = ol_checkpoints[cs]
        print(f"  {cs:6d} | {wslg_c['val_loss']:.4f} {wslg_c['val_acc']:.3f}"
              f" | {ws_c['val_loss']:.4f} {ws_c['val_acc']:.3f}"
              f" | {cllg_c['val_loss']:.4f} {cllg_c['val_acc']:.3f}"
              f" | {ol_c['val_loss']:.4f} {ol_c['val_acc']:.3f}")

    print(f"\n  === Robustness (delta-loss at eps=1.0) ===")
    print(f"  {'Step':>6s} | {'WS_LG':>8s} {'WS':>8s} "
          f"{'CL_LG':>8s} {'OL':>8s}")
    for cs in checkpoint_steps:
        vals = []
        for ckpts in [wslg_checkpoints, ws_checkpoints,
                      cllg_checkpoints, ol_checkpoints]:
            vals.append(ckpts[cs]["robustness"]["1.0"])
        print(f"  {cs:6d} | " +
              " ".join(f"{v:+8.4f}" for v in vals))

    print(f"\n  === Self-knowledge (R^2 at post_block0) ===")
    print(f"  {'Step':>6s} | {'WS_LG':>6s} {'WS':>6s} "
          f"{'CL_LG':>6s} {'OL':>6s}")
    for cs in checkpoint_steps:
        vals = []
        for ckpts in [wslg_checkpoints, ws_checkpoints,
                      cllg_checkpoints, ol_checkpoints]:
            sk = ckpts[cs].get("self_knowledge", {})
            vals.append(sk.get("post_block0", 0))
        print(f"  {cs:6d} | " +
              " ".join(f"{v:6.3f}" for v in vals))

    print(f"\n  === Gate sparsity (WS_LG, fraction dims < 0.1) ===")
    for _, label, gs in wslg_history["gate_stats"]:
        print(f"    {label:25s}: mean={gs['overall_mean']:.3f} "
              f"std={gs['overall_std']:.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f} "
              f"sparse(<0.2)={gs['sparsity_below_0.2']:.3f}")

    print(f"\n  === Residual structure (eff rank, mean eta^2) ===")
    print(f"  {'Step':>6s} | {'WS_LG':>16s} | {'WS':>16s} | "
          f"{'CL_LG':>16s} | {'OL':>16s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [wslg_checkpoints, ws_checkpoints,
                      cllg_checkpoints, ol_checkpoints]:
            rs = ckpts[cs].get("residual_stats", {})
            er = rs.get("eff_rank", 0)
            eta = rs.get("mean_eta_squared", 0)
            row.append(f"{er:5.1f} eta={eta:.3f}")
        print(f"  {cs:6d} | " + " | ".join(row))

    if any(wslg_checkpoints[cs].get("innovation_migration")
           for cs in checkpoint_steps):
        print(f"\n  === Innovation migration (WS_LG, subspace overlap) ===")
        for cs in checkpoint_steps:
            im = wslg_checkpoints[cs].get("innovation_migration")
            if im:
                print(f"    Cycle ending at {cs}: "
                      f"subspace_overlap={im['subspace_overlap']:.3f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    lambda_local: float = 1.0,
    gate_lr: float = 1e-3,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_gated_ratchet.remote(
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        gate_lr=gate_lr,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nGated ratchet experiment complete.")

    config = result["config"]
    print(f"\n  {config['n_cycles']} cycles × "
          f"({config['wake_steps']} wake + {config['distill_steps']} sleep) = "
          f"{config['total_main_steps']} main-model steps")

    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")
    for name, key in [("WS_LG", "ws_lg"), ("WS", "ws"),
                      ("CL_LG", "cl_lg"), ("OL", "ol")]:
        c = get_ckpt(result[key]["checkpoints"], final_step)
        if c:
            rob = c.get("robustness", {}).get("1.0", 0)
            print(f"    {name:5s}: loss={c['val_loss']:.4f} "
                  f"acc={c['val_acc']:.4f} "
                  f"rob(eps=1)={rob:+.4f}")
