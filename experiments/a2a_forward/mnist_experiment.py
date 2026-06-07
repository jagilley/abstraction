"""MNIST A2A forward model experiment.

Adapts the cerebellar forward model to MNIST classification via ViT.
Tests whether self-knowledge, robustness, and division-of-labor phenomena
generalize from autoregressive language modeling to image classification.

Architecture:
  Main model: 4-layer, 4-head, 128-dim ViT with 4x4 patches (50 positions)
  Forward model: 1-layer transformer, 1 head, 32-dim (bidirectional)
  Prediction: post_block0 -> post_block3 (3-layer gap)
  Injection: after block 1

Both open-loop and closed-loop models trained with identical seed/lr/init.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
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
    seed: int = 42,
    probe_batches: int = 40,
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

    print(f"MNIST A2A experiment on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch_size={patch_size}")
    print(f"  Forward model: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")

    # --- Load MNIST ---
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

    # --- Pre-generate batch indices for reproducibility ---
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

    # --- Initialize models with fixed seed, save initial state ---
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

    # =============================================
    # Training function (shared by both runs)
    # =============================================
    def train_run(closed_loop):
        tag = "CLOSED-LOOP" if closed_loop else "OPEN-LOOP"
        print(f"\n{'=' * 60}")
        print(f"  Training {tag}")
        print(f"{'=' * 60}")

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

        gate = None
        if closed_loop:
            gate = CerebellarGate(n_embd).to(device)
            main_params = list(model.parameters()) + list(gate.parameters())
        else:
            main_params = list(model.parameters())

        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "fwd_mse": [], "val_fwd_mse": [], "val_cosine": [],
            "val_residual_norm": [],
        }
        if closed_loop:
            history["val_loss_no_inj"] = []
            history["val_acc_no_inj"] = []
            history["gate_norm"] = []

        eval_idx = 0

        for step in range(n_steps):
            model.train()
            fwd_model.train()
            if gate:
                gate.train()

            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            if closed_loop:
                fwd_pred_cache = {}

                def cerebellar_fn(act):
                    fwd_pred = fwd_model(act.detach())
                    fwd_pred_cache["pred"] = fwd_pred
                    return gate(fwd_pred.detach())

                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                target = intermediates[predict_to].detach()
                fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)
            else:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                )
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()
                fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)

            opt_main.zero_grad()
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean().item()

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                if gate:
                    gate.eval()

                v_loss, v_acc = 0.0, 0.0
                v_loss_noinj, v_acc_noinj = 0.0, 0.0
                v_fwd, v_cos, v_res = 0.0, 0.0, 0.0

                with torch.no_grad():
                    for bi in range(n_eval_batches):
                        eidx = eval_indices[eval_idx][bi]
                        vim = test_images[eidx].to(device)
                        vlb = test_labels[eidx].to(device)

                        if closed_loop:
                            def eval_cb(act):
                                return gate(fwd_model(act))

                            vlog, vl, vi = model(
                                vim, vlb, return_intermediates=True,
                                cerebellar_fn=eval_cb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                            v_loss += vl.item()
                            v_acc += (vlog.argmax(-1) == vlb).float().mean().item()

                            vlog2, vl2, vi2 = model(
                                vim, vlb, return_intermediates=True,
                            )
                            v_loss_noinj += vl2.item()
                            v_acc_noinj += (vlog2.argmax(-1) == vlb).float().mean().item()

                            src = vi2[predict_from]
                            tgt = vi2[predict_to]
                        else:
                            vlog, vl, vi = model(
                                vim, vlb, return_intermediates=True,
                            )
                            v_loss += vl.item()
                            v_acc += (vlog.argmax(-1) == vlb).float().mean().item()

                            src = vi[predict_from]
                            tgt = vi[predict_to]

                        pred = fwd_model(src)
                        v_fwd += F.mse_loss(pred, tgt).item()
                        v_cos += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                        v_res += (tgt - pred).norm(dim=-1).mean().item()

                n = n_eval_batches
                v_loss /= n
                v_acc /= n
                v_fwd /= n
                v_cos /= n
                v_res /= n

                history["train_loss"].append((step, cls_loss.item()))
                history["train_acc"].append((step, acc))
                history["val_loss"].append((step, v_loss))
                history["val_acc"].append((step, v_acc))
                history["fwd_mse"].append((step, fwd_loss.item()))
                history["val_fwd_mse"].append((step, v_fwd))
                history["val_cosine"].append((step, v_cos))
                history["val_residual_norm"].append((step, v_res))

                extra = ""
                if closed_loop:
                    v_loss_noinj /= n
                    v_acc_noinj /= n
                    history["val_loss_no_inj"].append((step, v_loss_noinj))
                    history["val_acc_no_inj"].append((step, v_acc_noinj))
                    history["gate_norm"].append((step, gate.injection_norm()))
                    delta = v_loss - v_loss_noinj
                    extra = (f" Δ={delta:+.4f}"
                             f" gate={gate.injection_norm():.3f}")

                print(f"  step {step:5d}: "
                      f"loss={v_loss:.4f} acc={v_acc:.4f}{extra} | "
                      f"fwd_cos={v_cos:.4f} res={v_res:.3f}")

                eval_idx += 1

        return model, fwd_model, gate, history

    # --- Train both ---
    ol_model, ol_fwd, _, ol_history = train_run(closed_loop=False)
    cl_model, cl_fwd, cl_gate, cl_history = train_run(closed_loop=True)

    # =============================================
    # Self-knowledge probes
    # =============================================
    print(f"\n{'=' * 60}")
    print("  SELF-KNOWLEDGE PROBES")
    print(f"{'=' * 60}")

    probe_results = {}
    for label, model, fwd_model in [
        ("open_loop", ol_model, ol_fwd),
        ("closed_loop", cl_model, cl_fwd),
    ]:
        model.eval()
        fwd_model.eval()

        layer_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
        layer_acts = {k: [] for k in layer_names}
        residual_vecs = []
        residual_norms = []
        digit_labels_list = []

        with torch.no_grad():
            for bi in range(probe_batches):
                pidx = probe_indices[bi]
                pim = test_images[pidx].to(device)
                plb = test_labels[pidx]

                _, _, vi = model(pim, return_intermediates=True)

                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                residual = tgt - pred

                residual_vecs.append(residual.cpu())
                residual_norms.append(residual.norm(dim=-1).cpu())
                digit_labels_list.append(plb)

                for key in layer_names:
                    layer_acts[key].append(vi[key].cpu())

        all_residual = torch.cat(residual_vecs, dim=0)
        all_res_norms = torch.cat(residual_norms, dim=0)
        all_digits = torch.cat(digit_labels_list, dim=0)

        for key in layer_names:
            layer_acts[key] = torch.cat(layer_acts[key], dim=0)

        residual_flat = all_residual.reshape(-1, n_embd)

        layer_r2 = {}
        for layer_name in layer_names:
            acts_flat = layer_acts[layer_name].reshape(-1, n_embd)

            n_total = acts_flat.shape[0]
            n_train = int(0.8 * n_total)
            perm = torch.randperm(
                n_total, generator=torch.Generator().manual_seed(seed)
            )
            tr = perm[:n_train]
            te = perm[n_train:]

            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

            X_tr = acts_flat[tr].to(device)
            Y_tr = residual_flat[tr].to(device)
            X_te = acts_flat[te].to(device)
            Y_te = residual_flat[te].to(device)

            probe_bs = min(4096, n_train)
            for _ in range(300):
                sidx = torch.randint(n_train, (probe_bs,))
                p = probe(X_tr[sidx])
                ploss = F.mse_loss(p, Y_tr[sidx])
                opt.zero_grad()
                ploss.backward()
                opt.step()

            with torch.no_grad():
                pred_te = probe(X_te)
                mse = F.mse_loss(pred_te, Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0

            layer_r2[layer_name] = r2
            print(f"  {label:12s} | {layer_name:12s}: R² = {r2:.4f}")

        # Per-digit residual norms (averaged over patches per image)
        mean_res_per_img = all_res_norms.mean(dim=1)
        digit_res = {}
        for d in range(10):
            mask = all_digits == d
            if mask.sum() > 0:
                digit_res[str(d)] = float(mean_res_per_img[mask].mean())

        probe_results[label] = {
            "layer_r2": layer_r2,
            "digit_residual_norms": digit_res,
        }

    # =============================================
    # Residual PCA (on OL model)
    # =============================================
    print("\n--- Residual PCA ---")
    ol_model.eval()
    ol_fwd.eval()
    residuals_for_pca = []
    with torch.no_grad():
        for bi in range(min(probe_batches, 20)):
            pidx = probe_indices[bi]
            pim = test_images[pidx].to(device)
            _, _, vi = ol_model(pim, return_intermediates=True)
            src = vi[predict_from]
            tgt = vi[predict_to]
            pred = ol_fwd(src)
            residuals_for_pca.append((tgt - pred).cpu())

    all_res_pca = torch.cat(residuals_for_pca, dim=0).reshape(-1, n_embd).numpy()
    cov = np.cov(all_res_pca, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    if total_var > 0:
        proportions = eigenvalues / total_var
        cumulative = np.cumsum(proportions)
        top1_pct = float(proportions[0] * 100)
        top5_pct = float(cumulative[4] * 100) if len(cumulative) > 4 else 100.0
        top10_pct = float(cumulative[9] * 100) if len(cumulative) > 9 else 100.0

        p = proportions[proportions > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))

        rank_50 = int(np.searchsorted(cumulative, 0.5)) + 1
        rank_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    else:
        top1_pct = top5_pct = top10_pct = 0.0
        eff_rank = 0.0
        rank_50 = rank_90 = 0

    pca_results = {
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
        "top10_pct": top10_pct,
        "effective_rank": eff_rank,
        "rank_for_50pct_var": rank_50,
        "rank_for_90pct_var": rank_90,
        "total_dims": n_embd,
    }
    print(f"  Top-1 PC: {top1_pct:.1f}%, Top-5: {top5_pct:.1f}%, "
          f"Top-10: {top10_pct:.1f}%")
    print(f"  Effective rank: {eff_rank:.1f}/{n_embd}")
    print(f"  50% var at rank {rank_50}, 90% at rank {rank_90}")

    # =============================================
    # Robustness test
    # =============================================
    print(f"\n{'=' * 60}")
    print("  ROBUSTNESS TEST")
    print(f"{'=' * 60}")

    perturbation_strengths = [0.1, 0.5, 1.0, 2.0]
    rob_results = {}

    for label, model, fwd_model, gate in [
        ("open_loop", ol_model, ol_fwd, None),
        ("closed_loop", cl_model, cl_fwd, cl_gate),
    ]:
        model.eval()
        fwd_model.eval()
        if gate:
            gate.eval()

        eps_results = {}
        for eps in perturbation_strengths:
            loss_deltas = []
            n_rob_batches = min(probe_batches, 20)

            with torch.no_grad():
                for bi in range(n_rob_batches):
                    pidx = probe_indices[bi]
                    pim = test_images[pidx].to(device)
                    plb = test_labels[pidx].to(device)

                    if gate:
                        def cb(act):
                            return gate(fwd_model(act))
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

                    if gate:
                        def cb_p(act):
                            return gate(fwd_model(act))
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
            print(f"  {label:12s} | eps={eps:.1f}: "
                  f"Δloss = {mean_delta:+.4f}")

        rob_results[label] = eps_results

    # =============================================
    # Save results
    # =============================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist/{model_tag}/{gap_tag}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(ol_model.state_dict(), os.path.join(save_dir, "ol_model.pt"))
    torch.save(ol_fwd.state_dict(), os.path.join(save_dir, "ol_fwd.pt"))
    torch.save(cl_model.state_dict(), os.path.join(save_dir, "cl_model.pt"))
    torch.save(cl_fwd.state_dict(), os.path.join(save_dir, "cl_fwd.pt"))
    torch.save(cl_gate.state_dict(), os.path.join(save_dir, "cl_gate.pt"))

    result = {
        "config": {
            "dataset": "mnist",
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "patch_size": patch_size, "n_positions": n_positions,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "lr": lr, "fwd_lr": fwd_lr, "n_steps": n_steps,
            "batch_size": batch_size, "seed": seed,
        },
        "open_loop": {
            "history": ol_history,
            "final_val_loss": ol_history["val_loss"][-1][1],
            "final_val_acc": ol_history["val_acc"][-1][1],
            "final_fwd_cosine": ol_history["val_cosine"][-1][1],
            "final_fwd_mse": ol_history["val_fwd_mse"][-1][1],
            "final_residual_norm": ol_history["val_residual_norm"][-1][1],
        },
        "closed_loop": {
            "history": cl_history,
            "final_val_loss": cl_history["val_loss"][-1][1],
            "final_val_acc": cl_history["val_acc"][-1][1],
            "final_val_loss_no_inj": cl_history["val_loss_no_inj"][-1][1],
            "final_val_acc_no_inj": cl_history["val_acc_no_inj"][-1][1],
            "final_fwd_cosine": cl_history["val_cosine"][-1][1],
            "final_fwd_mse": cl_history["val_fwd_mse"][-1][1],
            "final_residual_norm": cl_history["val_residual_norm"][-1][1],
            "final_gate_norm": cl_history["gate_norm"][-1][1],
        },
        "probes": probe_results,
        "residual_pca": pca_results,
        "robustness": rob_results,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    ol = result["open_loop"]
    cl = result["closed_loop"]
    print(f"\n  Open-loop:  val_loss={ol['final_val_loss']:.4f}  "
          f"val_acc={ol['final_val_acc']:.4f}  "
          f"fwd_cos={ol['final_fwd_cosine']:.4f}")

    delta = cl["final_val_loss"] - cl["final_val_loss_no_inj"]
    dependency = cl["final_val_loss_no_inj"] - ol["final_val_loss"]
    print(f"  Closed-loop: val_loss={cl['final_val_loss']:.4f}  "
          f"val_acc={cl['final_val_acc']:.4f}  "
          f"fwd_cos={cl['final_fwd_cosine']:.4f}")
    print(f"  Injection benefit: Δ={delta:+.4f}")
    print(f"  Dependency: {dependency:+.4f}")
    print(f"  Gate norm: {cl['final_gate_norm']:.4f}")

    print(f"\n  Self-knowledge probes (R²):")
    for layer in ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]:
        ol_r2 = probe_results["open_loop"]["layer_r2"].get(layer, 0)
        cl_r2 = probe_results["closed_loop"]["layer_r2"].get(layer, 0)
        print(f"    {layer:12s}: OL={ol_r2:.4f}  CL={cl_r2:.4f}  "
              f"Δ={cl_r2 - ol_r2:+.4f}")

    print(f"\n  Per-digit residual norms (OL forward model):")
    for d in range(10):
        ol_dr = probe_results["open_loop"]["digit_residual_norms"].get(str(d), 0)
        print(f"    digit {d}: {ol_dr:.4f}")

    print(f"\n  Robustness (Δloss at eps=1.0):")
    ol_rob = rob_results["open_loop"].get("1.0", 0)
    cl_rob = rob_results["closed_loop"].get("1.0", 0)
    print(f"    OL: {ol_rob:+.4f}")
    print(f"    CL: {cl_rob:+.4f}")
    if ol_rob > 0:
        print(f"    CL/OL ratio: {cl_rob / ol_rob:.3f}")

    return result


@app.local_entrypoint()
def main(
    n_steps: int = 5000,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    lr: float = 3e-4,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
):
    result = a2a_mnist.remote(
        n_steps=n_steps,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        seed=seed,
    )
    print("\nMNIST A2A experiment complete.")
    ol = result["open_loop"]
    cl = result["closed_loop"]
    print(f"OL: acc={ol['final_val_acc']:.4f} "
          f"fwd_cos={ol['final_fwd_cosine']:.4f}")
    print(f"CL: acc={cl['final_val_acc']:.4f} "
          f"fwd_cos={cl['final_fwd_cosine']:.4f}")
    delta = cl["final_val_loss"] - cl["final_val_loss_no_inj"]
    dependency = cl["final_val_loss_no_inj"] - ol["final_val_loss"]
    print(f"Injection benefit: {delta:+.4f}, Dependency: {dependency:+.4f}")
