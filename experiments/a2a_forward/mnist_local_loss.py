"""MNIST local prediction-error learning experiment.

Tests whether using the forward model's prediction errors as local learning
signals improves learning, reduces injection dependency, and produces
robustness -- the simplest test of the local prediction-error learning idea.

Four conditions with identical seed/lr/init:
  OL:    open-loop baseline (NTP only)
  CL:    closed-loop injection only (current system)
  LL:    local loss only (no injection)
  CL_LL: injection + local loss

The local loss: lambda * MSE(sg(FM(post_block0)), post_block3)
  - sg = stop-gradient through FM (prediction is a frozen target)
  - gradient flows through post_block3 into main model blocks 0-3
  - "be predictable" pressure: pushes computation toward FM-compressible form

Key predictions:
  1. LL learns faster than OL (more supervision per example --
     NTP gives gradient to 1 position (CLS), local loss gives gradient
     to all 50 positions x 128 dims)
  2. CL_LL has less dependency than CL (continuous internalization)
  3. LL produces robustness (regularity pressure flattens loss landscape)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_mnist_local_loss(
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
    lambda_local: float = 1.0,
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

    print(f"MNIST LOCAL PREDICTION-ERROR LEARNING on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch_size={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  lambda_local={lambda_local}")

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

    # --- Pre-generate batch indices (shared across all conditions) ---
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

    # --- Initialize shared weights ---
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
    # Training function (parameterized by condition)
    # =============================================
    def train_run(name, use_injection, use_local_loss):
        print(f"\n{'=' * 60}")
        print(f"  Training {name} "
              f"(injection={use_injection}, local_loss={use_local_loss})")
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
        if use_injection:
            gate = CerebellarGate(n_embd).to(device)
            main_params = list(model.parameters()) + list(gate.parameters())
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
            history["val_loss_no_inj"] = []
            history["val_acc_no_inj"] = []
            history["gate_norm"] = []
        if use_local_loss:
            history["local_loss"] = []

        eval_idx = 0

        for step in range(n_steps):
            model.train()
            fwd_model.train()
            if gate:
                gate.train()

            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            # --- Forward pass ---
            if use_injection:
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
                fwd_pred = fwd_pred_cache["pred"]
            else:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                )
                source = intermediates[predict_from]
                fwd_pred = fwd_model(source.detach())

            target = intermediates[predict_to]

            # --- FM loss (gradient into FM only) ---
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            # --- Main model loss ---
            total_main_loss = cls_loss
            if use_local_loss:
                local_loss_val = F.mse_loss(fwd_pred.detach(), target)
                total_main_loss = total_main_loss + lambda_local * local_loss_val

            # --- Backward: main model ---
            opt_main.zero_grad()
            total_main_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # --- Backward: FM (independent graph) ---
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

                        if use_injection:
                            def eval_cb(act):
                                return gate(fwd_model(act))

                            vlog, vl, vi = model(
                                vim, vlb, return_intermediates=True,
                                cerebellar_fn=eval_cb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                            v_loss += vl.item()
                            v_acc += (
                                (vlog.argmax(-1) == vlb).float().mean().item()
                            )

                            vlog2, vl2, vi2 = model(
                                vim, vlb, return_intermediates=True,
                            )
                            v_loss_noinj += vl2.item()
                            v_acc_noinj += (
                                (vlog2.argmax(-1) == vlb).float().mean().item()
                            )

                            src = vi2[predict_from]
                            tgt = vi2[predict_to]
                        else:
                            vlog, vl, vi = model(
                                vim, vlb, return_intermediates=True,
                            )
                            v_loss += vl.item()
                            v_acc += (
                                (vlog.argmax(-1) == vlb).float().mean().item()
                            )

                            src = vi[predict_from]
                            tgt = vi[predict_to]

                        pred = fwd_model(src)
                        v_fwd += F.mse_loss(pred, tgt).item()
                        v_cos += (
                            F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                        )
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
                if use_injection:
                    v_loss_noinj /= n
                    v_acc_noinj /= n
                    history["val_loss_no_inj"].append((step, v_loss_noinj))
                    history["val_acc_no_inj"].append((step, v_acc_noinj))
                    history["gate_norm"].append((step, gate.injection_norm()))
                    delta = v_loss - v_loss_noinj
                    extra += (f" inj_delta={delta:+.4f}"
                              f" gate={gate.injection_norm():.3f}")
                if use_local_loss:
                    history["local_loss"].append(
                        (step, local_loss_val.item()),
                    )
                    extra += f" ll={local_loss_val.item():.5f}"

                print(f"  [{name:5s}] step {step:5d}: "
                      f"loss={v_loss:.4f} acc={v_acc:.4f}{extra} | "
                      f"fwd_cos={v_cos:.4f}")

                eval_idx += 1

        return model, fwd_model, gate, history

    # --- Train all 4 conditions ---
    conditions = [
        ("OL", False, False),
        ("CL", True, False),
        ("LL", False, True),
        ("CL_LL", True, True),
    ]

    trained = {}
    histories = {}
    for name, inj, ll in conditions:
        model, fm, gate, hist = train_run(name, inj, ll)
        trained[name] = (model, fm, gate)
        histories[name] = hist

    # =============================================
    # Self-knowledge probes
    # =============================================
    print(f"\n{'=' * 60}")
    print("  SELF-KNOWLEDGE PROBES")
    print(f"{'=' * 60}")

    layer_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    probe_results = {}

    for name, (model, fwd_model, gate) in trained.items():
        model.eval()
        fwd_model.eval()

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
                residual = tgt - pred

                residual_vecs.append(residual.cpu())
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
        for layer_name in layer_names:
            acts_flat = layer_acts[layer_name].reshape(-1, n_embd)

            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

            X_tr = acts_flat[tr].to(device)
            Y_tr = residual_flat[tr].to(device)
            X_te = acts_flat[te].to(device)
            Y_te = residual_flat[te].to(device)

            probe_bs = min(4096, n_train_probe)
            for _ in range(300):
                sidx = torch.randint(n_train_probe, (probe_bs,))
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
            print(f"  {name:5s} | {layer_name:12s}: R2 = {r2:.4f}")

        probe_results[name] = {"layer_r2": layer_r2}

    # =============================================
    # Robustness test
    # =============================================
    print(f"\n{'=' * 60}")
    print("  ROBUSTNESS TEST")
    print(f"{'=' * 60}")

    perturbation_strengths = [0.1, 0.5, 1.0, 2.0]
    rob_results = {}

    for name, (model, fwd_model, gate) in trained.items():
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
            print(f"  {name:5s} | eps={eps:.1f}: "
                  f"delta_loss = {mean_delta:+.4f}")

        rob_results[name] = eps_results

    # =============================================
    # Save results
    # =============================================
    import numpy as np

    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_local_loss"
                f"/{model_tag}/{gap_tag}/lambda_{lambda_local}")
    os.makedirs(save_dir, exist_ok=True)

    for name, (model, fm, gate) in trained.items():
        torch.save(model.state_dict(),
                    os.path.join(save_dir, f"{name}_model.pt"))
        torch.save(fm.state_dict(),
                    os.path.join(save_dir, f"{name}_fwd.pt"))
        if gate:
            torch.save(gate.state_dict(),
                        os.path.join(save_dir, f"{name}_gate.pt"))

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
            "lambda_local": lambda_local,
        },
    }

    for name in ["OL", "CL", "LL", "CL_LL"]:
        hist = histories[name]
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
                hist["val_loss_no_inj"][-1][1]
            )
            cond_result["final_val_acc_no_inj"] = (
                hist["val_acc_no_inj"][-1][1]
            )
            cond_result["final_gate_norm"] = hist["gate_norm"][-1][1]
        result[name] = cond_result

    result["probes"] = probe_results
    result["robustness"] = rob_results

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")

    # =============================================
    # Summary
    # =============================================
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    # Learning curves at key steps
    print(f"\n  Val accuracy at key steps:")
    for name in ["OL", "CL", "LL", "CL_LL"]:
        hist = histories[name]
        accs = dict(hist["val_acc"])
        steps_to_show = [
            s for s in [0, 500, 1000, 2500, n_steps - 1] if s in accs
        ]
        acc_str = "  ".join(f"s{s}={accs[s]:.4f}" for s in steps_to_show)
        print(f"    {name:5s}: {acc_str}")

    # Final metrics
    print(f"\n  Final metrics:")
    print(f"    {'Cond':5s} {'val_loss':>9s} {'val_acc':>8s} {'fwd_cos':>8s}")
    for name in ["OL", "CL", "LL", "CL_LL"]:
        r = result[name]
        print(f"    {name:5s} {r['final_val_loss']:>9.4f} "
              f"{r['final_val_acc']:>8.4f} "
              f"{r['final_fwd_cosine']:>8.4f}")

    # Dependency gap
    print(f"\n  Dependency (val_loss_no_inj - OL_val_loss):")
    ol_loss = result["OL"]["final_val_loss"]
    for name in ["CL", "CL_LL"]:
        noinj = result[name]["final_val_loss_no_inj"]
        dep = noinj - ol_loss
        inj_benefit = result[name]["final_val_loss"] - noinj
        print(f"    {name:5s}: dep={dep:+.4f}  "
              f"inj_benefit={inj_benefit:+.4f}  "
              f"gate={result[name]['final_gate_norm']:.3f}")

    # Self-knowledge probes
    print(f"\n  Self-knowledge probes (R2 at key layers):")
    for layer in ["post_block0", f"post_block{n_layer - 1}"]:
        r2s = {
            name: probe_results[name]["layer_r2"].get(layer, 0)
            for name in ["OL", "CL", "LL", "CL_LL"]
        }
        print(f"    {layer:12s}: " +
              "  ".join(f"{n}={r2s[n]:.4f}"
                        for n in ["OL", "CL", "LL", "CL_LL"]))

    # Robustness
    print(f"\n  Robustness (delta_loss at eps=1.0):")
    ol_rob = rob_results["OL"].get("1.0", 0)
    for name in ["OL", "CL", "LL", "CL_LL"]:
        rob = rob_results[name].get("1.0", 0)
        ratio = rob / ol_rob if ol_rob > 0 else float("nan")
        print(f"    {name:5s}: delta={rob:+.4f}  ratio={ratio:.3f}")

    return result


@app.local_entrypoint()
def main(
    n_steps: int = 5000,
    lambda_local: float = 1.0,
    seed: int = 42,
):
    result = a2a_mnist_local_loss.remote(
        n_steps=n_steps,
        lambda_local=lambda_local,
        seed=seed,
    )
    print("\nMNIST local prediction-error learning complete.")
    print(f"\n  Final metrics:")
    for name in ["OL", "CL", "LL", "CL_LL"]:
        r = result[name]
        print(f"    {name:5s}: acc={r['final_val_acc']:.4f} "
              f"fwd_cos={r['final_fwd_cosine']:.4f}")
    ol_loss = result["OL"]["final_val_loss"]
    for name in ["CL", "CL_LL"]:
        noinj = result[name]["final_val_loss_no_inj"]
        print(f"    {name:5s} dependency: {noinj - ol_loss:+.4f}")
