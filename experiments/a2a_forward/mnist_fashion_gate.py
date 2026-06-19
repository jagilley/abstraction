"""MNIST → Fashion-MNIST gate experiment: hard distribution shift.

Tests whether the bilevel learning gate closes on task-specific dimensions
when the training distribution shifts to a fundamentally different visual
domain. Unlike the digit-shift experiment (0-6 → 0-9, where all inputs
share the "handwritten digit" manifold), MNIST → Fashion-MNIST puts digit
features and clothing features in genuine representational tension.

Design:
  Phase 1 (cycles 1-2): Train on MNIST (classes 0-9)
  Phase 2 (cycles 3-4): Train on MNIST + Fashion-MNIST (classes 0-19)
  20-class output head. Both datasets are 28x28 grayscale, 60K train / 10K test.

Three conditions:
  1. WS_LG_shift: Gated ratchet, MNIST → MNIST + Fashion-MNIST
  2. WS_LG_full: Gated ratchet on MNIST + Fashion-MNIST throughout (stationary)
  3. OL_shift: Open-loop with same data schedule (mechanism control)

Key predictions:
  - WS_LG_full: gate opens uniformly (replicates stationary result)
  - WS_LG_shift after shift: gate closes on MNIST-specific dimensions (protect
    digit features from over-compression by clothing-feature pressure) and opens
    on Fashion-MNIST dimensions (learn new visual features)
  - The digit-shift experiment showed selective opening (novel > known) but no
    closing. The MNIST → Fashion-MNIST shift should produce genuine closing
    because the visual features are in direct tension.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def a2a_mnist_fashion_gate(
    # Architecture
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    n_classes: int = 20,
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
    phase1_cycles: int = 2,
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

    print(f"MNIST-FASHION GATE EXPERIMENT on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}, "
          f"{n_classes} classes")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block "
          f"{inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep")
    print(f"  Phase 1 (cycles 1-{phase1_cycles}): MNIST only (classes 0-9)")
    print(f"  Phase 2 (cycles {phase1_cycles+1}-{n_cycles}): "
          f"MNIST + Fashion-MNIST (classes 0-19)")
    print(f"  Total main-model steps: {total_main_steps}")

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
    # Load MNIST + Fashion-MNIST
    # -----------------------------------------------------------------
    print("\nLoading MNIST...")
    ds_mnist = load_dataset("ylecun/mnist")
    mnist_train_imgs = np.stack(
        [np.array(img) for img in ds_mnist["train"]["image"]])
    mnist_train_images = torch.from_numpy(
        mnist_train_imgs).float().unsqueeze(1) / 255.0
    mnist_train_labels = torch.tensor(ds_mnist["train"]["label"])
    mnist_test_imgs = np.stack(
        [np.array(img) for img in ds_mnist["test"]["image"]])
    mnist_test_images = torch.from_numpy(
        mnist_test_imgs).float().unsqueeze(1) / 255.0
    mnist_test_labels = torch.tensor(ds_mnist["test"]["label"])
    print(f"  MNIST train: {len(mnist_train_images)}, "
          f"test: {len(mnist_test_images)}")

    print("Loading Fashion-MNIST...")
    ds_fashion = load_dataset("zalando-datasets/fashion_mnist")
    fashion_train_imgs = np.stack(
        [np.array(img) for img in ds_fashion["train"]["image"]])
    fashion_train_images = torch.from_numpy(
        fashion_train_imgs).float().unsqueeze(1) / 255.0
    fashion_train_labels = torch.tensor(
        ds_fashion["train"]["label"]) + 10  # offset to 10-19
    fashion_test_imgs = np.stack(
        [np.array(img) for img in ds_fashion["test"]["image"]])
    fashion_test_images = torch.from_numpy(
        fashion_test_imgs).float().unsqueeze(1) / 255.0
    fashion_test_labels = torch.tensor(
        ds_fashion["test"]["label"]) + 10  # offset to 10-19
    print(f"  Fashion train: {len(fashion_train_images)}, "
          f"test: {len(fashion_test_images)}")

    # Combined training set (for phase 2 and full condition)
    combined_train_images = torch.cat(
        [mnist_train_images, fashion_train_images], dim=0)
    combined_train_labels = torch.cat(
        [mnist_train_labels, fashion_train_labels], dim=0)
    print(f"  Combined train: {len(combined_train_images)}")

    # Combined test set for probe/eval
    combined_test_images = torch.cat(
        [mnist_test_images, fashion_test_images], dim=0)
    combined_test_labels = torch.cat(
        [mnist_test_labels, fashion_test_labels], dim=0)
    # Track which test images are MNIST vs Fashion
    test_is_mnist = torch.cat([
        torch.ones(len(mnist_test_images), dtype=torch.bool),
        torch.zeros(len(fashion_test_images), dtype=torch.bool),
    ])
    print(f"  Combined test: {len(combined_test_images)} "
          f"({test_is_mnist.sum()} MNIST, "
          f"{(~test_is_mnist).sum()} Fashion)")

    # -----------------------------------------------------------------
    # Pre-generate batch indices
    # -----------------------------------------------------------------
    # Shift: phase 1 from MNIST only, phase 2 from combined
    shift_gen = torch.Generator().manual_seed(seed)
    phase1_steps = phase1_cycles * steps_per_cycle
    phase2_steps = (n_cycles - phase1_cycles) * steps_per_cycle

    shift_batch_indices = []
    shift_batch_sources = []  # track MNIST vs combined
    for _ in range(phase1_steps):
        shift_batch_indices.append(
            torch.randint(len(mnist_train_images), (batch_size,),
                          generator=shift_gen))
        shift_batch_sources.append("mnist")
    for _ in range(phase2_steps):
        shift_batch_indices.append(
            torch.randint(len(combined_train_images), (batch_size,),
                          generator=shift_gen))
        shift_batch_sources.append("combined")

    # Full: combined throughout
    full_gen = torch.Generator().manual_seed(seed)
    full_batch_indices = [
        torch.randint(len(combined_train_images), (batch_size,),
                       generator=full_gen)
        for _ in range(total_main_steps)
    ]

    # Eval indices from combined test set
    eval_gen = torch.Generator().manual_seed(seed + 1)
    n_evals = total_main_steps // eval_interval + 10
    eval_indices = [
        [torch.randint(len(combined_test_images), (batch_size,),
                        generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]

    # Probe indices from combined test set
    probe_gen = torch.Generator().manual_seed(seed + 2)
    probe_indices = [
        torch.randint(len(combined_test_images), (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]

    # Repoint indices: shift uses phase-appropriate data
    repoint_gen_shift = torch.Generator().manual_seed(seed + 3)
    repoint_gen_full = torch.Generator().manual_seed(seed + 3)

    shift_repoint_indices = []
    shift_repoint_sources = []
    for cycle in range(n_cycles):
        if cycle < phase1_cycles:
            for _ in range(retrain_steps):
                shift_repoint_indices.append(
                    torch.randint(len(mnist_train_images), (batch_size,),
                                  generator=repoint_gen_shift))
                shift_repoint_sources.append("mnist")
        else:
            for _ in range(retrain_steps):
                shift_repoint_indices.append(
                    torch.randint(len(combined_train_images), (batch_size,),
                                  generator=repoint_gen_shift))
                shift_repoint_sources.append("combined")

    full_repoint_indices = [
        torch.randint(len(combined_train_images), (batch_size,),
                       generator=repoint_gen_full)
        for _ in range(n_cycles * retrain_steps)
    ]

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------
    def make_vit():
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1,
            n_classes=n_classes,
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
    def get_batch(batch_idx, source):
        """Get images and labels from the right dataset."""
        if source == "mnist":
            return (mnist_train_images[batch_idx].to(device),
                    mnist_train_labels[batch_idx].to(device))
        else:
            return (combined_train_images[batch_idx].to(device),
                    combined_train_labels[batch_idx].to(device))

    def eval_model(model, eval_idx=0):
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = combined_test_images[eidx].to(device)
                labels = combined_test_labels[eidx].to(device)
                logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def eval_by_dataset(model):
        """Full test set evaluation with MNIST vs Fashion-MNIST breakdown."""
        model.eval()
        mnist_correct, mnist_total = 0, 0
        fashion_correct, fashion_total = 0, 0
        mnist_loss_sum, fashion_loss_sum = 0.0, 0.0

        with torch.no_grad():
            for start in range(0, len(combined_test_images), batch_size):
                end = min(start + batch_size, len(combined_test_images))
                imgs = combined_test_images[start:end].to(device)
                lbls = combined_test_labels[start:end].to(device)
                is_m = test_is_mnist[start:end]

                logits, _ = model(imgs, lbls)
                preds = logits.argmax(-1).cpu()
                per_loss = F.cross_entropy(
                    logits, lbls, reduction='none').cpu()
                lbls_cpu = lbls.cpu()

                m_mask = is_m
                f_mask = ~is_m
                if m_mask.sum() > 0:
                    mnist_correct += (
                        preds[m_mask] == lbls_cpu[m_mask]).sum().item()
                    mnist_total += m_mask.sum().item()
                    mnist_loss_sum += per_loss[m_mask].sum().item()
                if f_mask.sum() > 0:
                    fashion_correct += (
                        preds[f_mask] == lbls_cpu[f_mask]).sum().item()
                    fashion_total += f_mask.sum().item()
                    fashion_loss_sum += per_loss[f_mask].sum().item()

        overall_correct = mnist_correct + fashion_correct
        overall_total = mnist_total + fashion_total

        return {
            "overall_acc": float(overall_correct / overall_total),
            "overall_loss": float(
                (mnist_loss_sum + fashion_loss_sum) / overall_total),
            "mnist_acc": float(mnist_correct / mnist_total)
                if mnist_total > 0 else 0,
            "fashion_acc": float(fashion_correct / fashion_total)
                if fashion_total > 0 else 0,
            "mnist_loss": float(mnist_loss_sum / mnist_total)
                if mnist_total > 0 else 0,
            "fashion_loss": float(fashion_loss_sum / fashion_total)
                if fashion_total > 0 else 0,
        }

    def eval_model_with_inj(model, fm, gate, eval_idx=0):
        model.eval(); fm.eval(); gate.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = combined_test_images[eidx].to(device)
                labels = combined_test_labels[eidx].to(device)

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
                    vim = combined_test_images[pidx].to(device)
                    vlb = combined_test_labels[pidx].to(device)
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
                images = combined_test_images[pidx].to(device)
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
        model.eval(); fm.eval()
        all_res = []

        with torch.no_grad():
            for pidx in probe_indices:
                images = combined_test_images[pidx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                res = tgt - pred
                all_res.append(res.reshape(-1, n_embd).cpu())

        all_res = torch.cat(all_res)

        R = all_res.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals = np.maximum(evals, 0)
        total = evals.sum()
        fracs = evals / total if total > 0 else evals
        fracs_pos = fracs[fracs > 1e-12]
        eff_rank = float(np.exp(-np.sum(fracs_pos * np.log(fracs_pos))))

        # FM cosine
        with torch.no_grad():
            cos_total = 0.0
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = combined_test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                cos_total += F.cosine_similarity(
                    pred, tgt, dim=-1).mean().item()

        stats = {
            "eff_rank": eff_rank,
            "top1_pc_frac": float(fracs[0]),
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "fwd_cosine": cos_total / n_eval_batches,
        }
        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"res_norm={stats['mean_res_norm']:.3f}, "
              f"fwd_cos={stats['fwd_cosine']:.4f}")
        return stats

    def train_fresh_fm(model, steps, fm_seed, label, repoint_idxs,
                       repoint_srcs):
        """Train fresh FM on frozen model activations."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(steps):
            fm.train()
            ridx = repoint_idxs[step % len(repoint_idxs)]
            src = repoint_srcs[step % len(repoint_srcs)]
            if src == "mnist":
                images = mnist_train_images[ridx].to(device)
            else:
                images = combined_train_images[ridx].to(device)
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
                vim = combined_test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                p = fm(vi[predict_from])
                cos_total += F.cosine_similarity(
                    p, vi[predict_to], dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cosine: {cos:.4f}")

        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    def get_gate_stats_by_dataset(lgate, model, fwd_model, label):
        """Gate weight statistics conditioned on dataset (MNIST vs Fashion)."""
        lgate.eval(); model.eval(); fwd_model.eval()
        all_gw = []
        all_is_mnist = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                vim = combined_test_images[pidx].to(device)
                is_m = test_is_mnist[pidx]
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                gw = lgate(src[:, 0], (tgt - pred)[:, 0])
                all_gw.append(gw.cpu())
                all_is_mnist.append(is_m)

        gw_cat = torch.cat(all_gw, dim=0)
        is_mnist_cat = torch.cat(all_is_mnist, dim=0)
        dim_mean = gw_cat.mean(dim=0)

        mnist_mask = is_mnist_cat
        fashion_mask = ~is_mnist_cat
        mnist_mean = float(gw_cat[mnist_mask].mean()) \
            if mnist_mask.sum() > 0 else 0
        fashion_mean = float(gw_cat[fashion_mask].mean()) \
            if fashion_mask.sum() > 0 else 0

        # Per-dimension means for each dataset
        mnist_dim_mean = gw_cat[mnist_mask].mean(dim=0) \
            if mnist_mask.sum() > 0 else torch.zeros(n_embd)
        fashion_dim_mean = gw_cat[fashion_mask].mean(dim=0) \
            if fashion_mask.sum() > 0 else torch.zeros(n_embd)
        dim_diff = fashion_dim_mean - mnist_dim_mean

        stats = {
            "overall_mean": float(gw_cat.mean()),
            "overall_std": float(gw_cat.std()),
            "mnist_mean": mnist_mean,
            "fashion_mean": fashion_mean,
            "fashion_mnist_diff": fashion_mean - mnist_mean,
            "dim_mean": dim_mean.tolist(),
            "mnist_dim_mean": mnist_dim_mean.tolist(),
            "fashion_dim_mean": fashion_dim_mean.tolist(),
            "dim_diff_mean": float(dim_diff.mean()),
            "dim_diff_std": float(dim_diff.std()),
            "n_dims_fashion_higher": int((dim_diff > 0).sum()),
            "sparsity_below_0.1": float((dim_mean < 0.1).float().mean()),
            "sparsity_below_0.2": float((dim_mean < 0.2).float().mean()),
        }

        print(f"  [{label}] Gate: overall={stats['overall_mean']:.3f}, "
              f"mnist={mnist_mean:.3f}, fashion={fashion_mean:.3f}, "
              f"diff={stats['fashion_mnist_diff']:+.3f}, "
              f"dims_fash_higher={stats['n_dims_fashion_higher']}/{n_embd}")
        return stats

    # =================================================================
    # WS_LG TRAINING LOOP
    # =================================================================
    def run_wslg(condition_label, batch_indices, batch_sources,
                 repoint_indices_list, repoint_sources_list):
        print(f"\n{'='*60}")
        print(f"  {condition_label} ({n_cycles} cycles)")
        print(f"{'='*60}")

        model = make_vit()
        model.load_state_dict(init_model_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)
        lgate = LearningGate(n_embd, lg_hidden).to(device)

        checkpoints = {}
        history = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "gate_stats": [],
        }
        global_step = 0
        repoint_offset = 0

        for cycle in range(n_cycles):
            phase = "phase1" if cycle < phase1_cycles else "phase2"
            print(f"\n  --- {condition_label} Cycle {cycle+1}/{n_cycles} "
                  f"({phase}) ---")

            # === WAKE: CL_LG co-training ===
            print(f"  [WAKE] CL_LG co-training ({wake_steps} steps)")
            cgate = CerebellarGate(n_embd).to(device)
            main_params = list(model.parameters()) + list(cgate.parameters())
            n_model_params = len(list(model.parameters()))
            opt_main = torch.optim.AdamW(
                main_params, lr=lr, weight_decay=0.01)
            opt_fwd = torch.optim.AdamW(
                fm.parameters(), lr=fwd_lr, weight_decay=0.01)
            opt_lg = torch.optim.AdamW(
                lgate.parameters(), lr=gate_lr, weight_decay=0.01)

            for step in range(wake_steps):
                model.train(); fm.train()
                cgate.train(); lgate.train()

                bidx = batch_indices[global_step]
                bsrc = batch_sources[global_step]
                if bsrc == "mnist":
                    images = mnist_train_images[bidx].to(device)
                    labels = mnist_train_labels[bidx].to(device)
                else:
                    images = combined_train_images[bidx].to(device)
                    labels = combined_train_labels[bidx].to(device)

                fwd_pred_cache = {}

                def cerebellar_fn(act, _cache=fwd_pred_cache):
                    fp = fm(act.detach())
                    _cache["pred"] = fp
                    return cgate(fp.detach())

                logits, cls_loss, intermediates = model(
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
                gate_w = lgate(source_cls, fm_error_cls)

                r = target - fwd_pred.detach()
                gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()
                L_total = cls_loss + lambda_local * gated_ll

                inner_grads = torch.autograd.grad(
                    L_total, main_params, create_graph=True)
                grads_for_update = [g.detach().clone() for g in inner_grads]

                virtual_model_params = {
                    pname: p - lr * g
                    for (pname, p), g in zip(
                        model.named_parameters(),
                        inner_grads[:n_model_params],
                    )
                }
                virtual_cgate_params = {
                    pname: p - lr * g
                    for (pname, p), g in zip(
                        cgate.named_parameters(),
                        inner_grads[n_model_params:],
                    )
                }

                def meta_cerebellar_fn(act):
                    fp = fm(act.detach())
                    return functional_call(
                        cgate, virtual_cgate_params, (fp.detach(),))

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
                cls_loss_prime = meta_out[1]

                opt_lg.zero_grad()
                cls_loss_prime.backward()
                torch.nn.utils.clip_grad_norm_(lgate.parameters(), 1.0)
                opt_lg.step()

                opt_main.zero_grad()
                for p, g in zip(main_params, grads_for_update):
                    p.grad = g
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

                    dep_loss, _ = eval_model_with_inj(model, fm, cgate)
                    dep = v_loss - dep_loss
                    print(f"    step {global_step:5d} (wake {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f} "
                          f"dep={dep:+.4f} gw={gate_w.mean().item():.3f}")

                global_step += 1

            gs = get_gate_stats_by_dataset(
                lgate, model, fm,
                f"{condition_label} c{cycle+1} wake")
            history["gate_stats"].append(
                (global_step, f"post_wake_c{cycle+1}", gs))

            wake_dep_loss, _ = eval_model_with_inj(model, fm, cgate)
            wake_noinj_loss, _ = eval_model(model)
            wake_dep_gap = wake_noinj_loss - wake_dep_loss
            print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}")

            # === SLEEP: Distillation ===
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
            teacher_gate = CerebellarGate(n_embd).to(device)
            teacher_gate.load_state_dict(cgate.state_dict())
            teacher_gate.eval()
            for p in teacher_gate.parameters():
                p.requires_grad = False

            student = make_vit()
            student.load_state_dict(model.state_dict())
            opt_student = torch.optim.AdamW(
                student.parameters(), lr=distill_lr, weight_decay=0.01)

            for step in range(distill_steps):
                student.train()
                bidx = batch_indices[global_step]
                bsrc = batch_sources[global_step]
                if bsrc == "mnist":
                    images = mnist_train_images[bidx].to(device)
                    labels = mnist_train_labels[bidx].to(device)
                else:
                    images = combined_train_images[bidx].to(device)
                    labels = combined_train_labels[bidx].to(device)

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
                    history["val_loss"].append((global_step, v_loss))
                    history["val_acc"].append((global_step, v_acc))
                    print(f"    step {global_step:5d} (sleep {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f}")

                global_step += 1

            model.load_state_dict(student.state_dict())
            del teacher, teacher_fm, teacher_gate, student, opt_student
            del cgate, opt_main, opt_fwd, opt_lg
            torch.cuda.empty_cache()

            # === RE-POINT: Fresh FM ===
            fm_seed = seed + 100 * (cycle + 1)
            rp_start = repoint_offset
            rp_end = rp_start + retrain_steps
            rp_idxs = repoint_indices_list[rp_start:rp_end]
            rp_srcs = repoint_sources_list[rp_start:rp_end]
            repoint_offset = rp_end

            print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, "
                  f"seed={fm_seed})")
            fm, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed,
                f"{condition_label} c{cycle+1}", rp_idxs, rp_srcs)

            # === CHECKPOINT ===
            cycle_step = (cycle + 1) * steps_per_cycle
            print(f"\n  {condition_label} CHECKPOINT at step {cycle_step}")

            c_loss, c_acc = eval_model(model)
            c_ds = eval_by_dataset(model)
            c_rob = measure_robustness(
                model, f"{condition_label} c{cycle+1}")
            c_sk = self_knowledge_probes(
                model, fm, f"{condition_label} c{cycle+1}")
            c_res = compute_residual_stats(
                model, fm, f"{condition_label} c{cycle+1}")

            gs_post = get_gate_stats_by_dataset(
                lgate, model, fm,
                f"{condition_label} c{cycle+1} repoint")
            history["gate_stats"].append(
                (cycle_step, f"post_repoint_c{cycle+1}", gs_post))

            checkpoints[cycle_step] = {
                "val_loss": c_loss, "val_acc": c_acc,
                "dataset_breakdown": c_ds,
                "robustness": c_rob,
                "self_knowledge": c_sk,
                "residual_stats": c_res,
                "gate_stats_post_wake": history["gate_stats"][-2][2],
                "gate_stats_post_repoint": gs_post,
                "wake_dep_gap": wake_dep_gap,
                "fm_cosine": fm_cos,
                "phase": phase,
            }
            print(f"    loss={c_loss:.4f} acc={c_acc:.4f}")
            print(f"    mnist_acc={c_ds['mnist_acc']:.4f} "
                  f"fashion_acc={c_ds['fashion_acc']:.4f}")

        return model, fm, lgate, checkpoints, history

    # =================================================================
    # CONDITION 1: WS_LG_shift (MNIST → MNIST + Fashion)
    # =================================================================
    shift_sources = []
    for _ in range(phase1_steps):
        shift_sources.append("mnist")
    for _ in range(phase2_steps):
        shift_sources.append("combined")

    wslg_shift_model, wslg_shift_fm, wslg_shift_lgate, \
        wslg_shift_ckpts, wslg_shift_hist = run_wslg(
            "WS_LG_shift", shift_batch_indices, shift_sources,
            shift_repoint_indices, shift_repoint_sources)

    # =================================================================
    # CONDITION 2: WS_LG_full (MNIST + Fashion throughout)
    # =================================================================
    full_sources = ["combined"] * total_main_steps
    full_rp_sources = ["combined"] * (n_cycles * retrain_steps)

    wslg_full_model, wslg_full_fm, wslg_full_lgate, \
        wslg_full_ckpts, wslg_full_hist = run_wslg(
            "WS_LG_full", full_batch_indices, full_sources,
            full_repoint_indices, full_rp_sources)

    # =================================================================
    # CONDITION 3: OL_shift (open-loop, same data schedule)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  OL_shift ({total_main_steps} steps)")
    print(f"{'='*60}")

    ol_model = make_vit()
    ol_model.load_state_dict(init_model_state)
    ol_opt = torch.optim.AdamW(ol_model.parameters(), lr=lr, weight_decay=0.01)

    ol_checkpoints = {}
    ol_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for step in range(total_main_steps):
        ol_model.train()
        bidx = shift_batch_indices[step]
        bsrc = shift_sources[step]
        if bsrc == "mnist":
            images = mnist_train_images[bidx].to(device)
            labels = mnist_train_labels[bidx].to(device)
        else:
            images = combined_train_images[bidx].to(device)
            labels = combined_train_labels[bidx].to(device)

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
            cycle = checkpoint_steps.index(ckpt_step)
            phase = "phase1" if cycle < phase1_cycles else "phase2"
            print(f"\n  OL_shift CHECKPOINT at step {ckpt_step} ({phase})")

            ol_l, ol_a = eval_model(ol_model)
            ol_ds = eval_by_dataset(ol_model)
            ol_r = measure_robustness(ol_model, f"OL_shift c{cycle+1}")

            rp_start = cycle * retrain_steps
            rp_idxs = shift_repoint_indices[rp_start:rp_start + retrain_steps]
            rp_srcs = shift_repoint_sources[rp_start:rp_start + retrain_steps]
            ol_fresh_fm, ol_fm_cos = train_fresh_fm(
                ol_model, retrain_steps, seed + 700 + ckpt_step,
                f"OL_shift c{cycle+1}", rp_idxs, rp_srcs)
            ol_s = self_knowledge_probes(
                ol_model, ol_fresh_fm, f"OL_shift c{cycle+1}")
            ol_e = compute_residual_stats(
                ol_model, ol_fresh_fm, f"OL_shift c{cycle+1}")

            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_l, "val_acc": ol_a,
                "dataset_breakdown": ol_ds,
                "robustness": ol_r,
                "self_knowledge": ol_s,
                "residual_stats": ol_e,
                "fm_cosine": ol_fm_cos,
                "phase": phase,
            }
            print(f"    loss={ol_l:.4f} acc={ol_a:.4f}")
            print(f"    mnist_acc={ol_ds['mnist_acc']:.4f} "
                  f"fashion_acc={ol_ds['fashion_acc']:.4f}")

            del ol_fresh_fm
            torch.cuda.empty_cache()

    del ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_fashion_gate"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(wslg_shift_model.state_dict(),
               os.path.join(save_root, "wslg_shift_model.pt"))
    torch.save(wslg_shift_fm.state_dict(),
               os.path.join(save_root, "wslg_shift_fm.pt"))
    torch.save(wslg_shift_lgate.state_dict(),
               os.path.join(save_root, "wslg_shift_lgate.pt"))
    torch.save(wslg_full_model.state_dict(),
               os.path.join(save_root, "wslg_full_model.pt"))
    torch.save(wslg_full_fm.state_dict(),
               os.path.join(save_root, "wslg_full_fm.pt"))
    torch.save(wslg_full_lgate.state_dict(),
               os.path.join(save_root, "wslg_full_lgate.pt"))
    torch.save(ol_model.state_dict(),
               os.path.join(save_root, "ol_shift_model.pt"))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "phase1_cycles": phase1_cycles,
            "n_classes": n_classes,
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
        "wslg_shift": {
            "checkpoints": wslg_shift_ckpts,
            "history": wslg_shift_hist,
        },
        "wslg_full": {
            "checkpoints": wslg_full_ckpts,
            "history": wslg_full_hist,
        },
        "ol_shift": {
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

    print(f"\n  Phase 1: MNIST (classes 0-9), cycles 1-{phase1_cycles}")
    print(f"  Phase 2: MNIST + Fashion-MNIST (classes 0-19), "
          f"cycles {phase1_cycles+1}-{n_cycles}")

    print(f"\n  === MNIST accuracy (forgetting measure) ===")
    print(f"  {'Cycle':>6s} | {'WS_LG_shift':>12s} | {'WS_LG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [wslg_shift_ckpts, wslg_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            ds = c.get("dataset_breakdown", {})
            row.append(f"{ds.get('mnist_acc', 0):.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    print(f"\n  === Fashion-MNIST accuracy (adaptation measure) ===")
    print(f"  {'Cycle':>6s} | {'WS_LG_shift':>12s} | {'WS_LG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [wslg_shift_ckpts, wslg_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            ds = c.get("dataset_breakdown", {})
            row.append(f"{ds.get('fashion_acc', 0):.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    print(f"\n  === Gate dynamics (WS_LG_shift): "
          f"MNIST vs Fashion gate weights ===")
    for _, label, gs in wslg_shift_hist["gate_stats"]:
        print(f"    {label:30s}: overall={gs['overall_mean']:.3f} "
              f"mnist={gs['mnist_mean']:.3f} "
              f"fashion={gs['fashion_mean']:.3f} "
              f"diff={gs['fashion_mnist_diff']:+.3f}")

    print(f"\n  === Gate dynamics (WS_LG_full): "
          f"MNIST vs Fashion gate weights ===")
    for _, label, gs in wslg_full_hist["gate_stats"]:
        print(f"    {label:30s}: overall={gs['overall_mean']:.3f} "
              f"mnist={gs['mnist_mean']:.3f} "
              f"fashion={gs['fashion_mean']:.3f} "
              f"diff={gs['fashion_mnist_diff']:+.3f}")

    print(f"\n  === Robustness (delta-loss at eps=1.0) ===")
    print(f"  {'Cycle':>6s} | {'WS_LG_shift':>12s} | {'WS_LG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [wslg_shift_ckpts, wslg_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            r = c.get("robustness", {}).get("1.0", 0)
            row.append(f"{r:+.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    phase1_cycles: int = 2,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    lambda_local: float = 1.0,
    gate_lr: float = 1e-3,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_fashion_gate.remote(
        n_cycles=n_cycles,
        phase1_cycles=phase1_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        gate_lr=gate_lr,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nMNIST-Fashion gate experiment complete.")

    config = result["config"]
    print(f"\n  Phase 1: MNIST (cycles 1-{config['phase1_cycles']})")
    print(f"  Phase 2: MNIST + Fashion "
          f"(cycles {config['phase1_cycles']+1}-{config['n_cycles']})")

    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")
    for name, key in [("WS_LG_shift", "wslg_shift"),
                      ("WS_LG_full", "wslg_full"),
                      ("OL_shift", "ol_shift")]:
        c = get_ckpt(result[key]["checkpoints"], final_step)
        if c:
            ds = c.get("dataset_breakdown", {})
            rob = c.get("robustness", {}).get("1.0", 0)
            print(f"    {name:12s}: mnist={ds.get('mnist_acc', 0):.4f} "
                  f"fashion={ds.get('fashion_acc', 0):.4f} "
                  f"rob(eps=1)={rob:+.4f}")
