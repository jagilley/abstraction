"""MNIST baseline battery: is forward self-prediction uniquely useful?

5 conditions with identical lr/seed/init/data order:
  1. open_loop:   no injection
  2. forward:     inject gate(fwd_model(post_block0))
  3. shifted:     inject gate(shift(fwd_model(post_block0), k=10))
  4. random_proj: inject gate(frozen_random_proj(post_block0))
  5. autoencoder: inject gate(autoenc(post_block0)) -- reconstructs post_block0

All conditions co-train a forward model (post_block0 -> post_block3) for
consistent self-knowledge probing. After training: classification metrics,
self-knowledge probes at each layer, robustness test.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "shifted", "random_proj", "autoencoder"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_mnist_baseline_battery(
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 5000,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    shift_k: int = 10,
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
    n_positions = (28 // patch_size) ** 2 + 1

    print(f"MNIST BASELINE BATTERY on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  shift_k={shift_k}, seed={seed}")

    # --- Load MNIST ---
    print("Loading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # --- Pre-generate batch indices ---
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

    # Separate init for autoencoder
    torch.manual_seed(seed + 100)
    init_autoenc = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
        block_size=n_positions, causal=False,
    ).to(device)
    init_autoenc_state = {k: v.cpu().clone() for k, v in init_autoenc.state_dict().items()}

    # Frozen random projection
    rng = np.random.default_rng(seed + 200)
    random_proj_weight = torch.from_numpy(
        (rng.standard_normal((n_embd, n_embd)) / np.sqrt(n_embd)).astype(np.float32)
    ).to(device)

    del init_model, init_fwd, init_autoenc
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =============================================
    # Training function
    # =============================================
    def train_run(condition):
        has_inj = condition != "open_loop"
        print(f"\n{'=' * 60}")
        print(f"  Training {condition.upper().replace('_', '-')}")
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

        gate = CerebellarGate(n_embd).to(device) if has_inj else None

        autoenc = None
        opt_autoenc = None
        if condition == "autoencoder":
            autoenc = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                block_size=n_positions, causal=False,
            ).to(device)
            autoenc.load_state_dict(init_autoenc_state)
            opt_autoenc = torch.optim.AdamW(
                autoenc.parameters(), lr=fwd_lr, weight_decay=0.01,
            )

        main_params = list(model.parameters())
        if gate:
            main_params += list(gate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01,
        )

        history = {
            "val_loss": [], "val_acc": [],
            "val_loss_no_inj": [], "val_cosine": [],
        }
        if gate:
            history["gate_norm"] = []

        eval_idx = 0
        for step in range(n_steps):
            model.train()
            fwd_model.train()
            if gate:
                gate.train()
            if autoenc:
                autoenc.train()

            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            # Build cerebellar_fn
            fwd_pred_cache = {}

            def make_cb():
                if condition == "forward":
                    def fn(act):
                        p = fwd_model(act.detach())
                        fwd_pred_cache["pred"] = p
                        return gate(p.detach())
                    return fn
                elif condition == "shifted":
                    def fn(act):
                        p = fwd_model(act.detach())
                        fwd_pred_cache["pred"] = p
                        shifted = torch.zeros_like(p)
                        if shift_k < p.shape[1]:
                            shifted[:, shift_k:] = p[:, :-shift_k]
                        return gate(shifted.detach())
                    return fn
                elif condition == "random_proj":
                    def fn(act):
                        proj = act.detach() @ random_proj_weight.T
                        return gate(proj)
                    return fn
                elif condition == "autoencoder":
                    def fn(act):
                        recon = autoenc(act.detach())
                        fwd_pred_cache["autoenc_pred"] = recon
                        return gate(recon.detach())
                    return fn
                return None

            cb = make_cb()

            if cb is not None:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
            else:
                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                )

            # Forward model MSE (all conditions)
            if condition in ("forward", "shifted") and "pred" in fwd_pred_cache:
                fwd_pred = fwd_pred_cache["pred"]
            else:
                fwd_pred = fwd_model(intermediates[predict_from].detach())
            fwd_target = intermediates[predict_to].detach()
            fwd_loss = F.mse_loss(fwd_pred, fwd_target)

            # Main model backward
            opt_main.zero_grad()
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # Forward model backward
            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            # Autoencoder backward
            if autoenc and "autoenc_pred" in fwd_pred_cache:
                ae_target = intermediates[predict_from].detach()
                ae_loss = F.mse_loss(fwd_pred_cache["autoenc_pred"], ae_target)
                opt_autoenc.zero_grad()
                ae_loss.backward()
                torch.nn.utils.clip_grad_norm_(autoenc.parameters(), 1.0)
                opt_autoenc.step()

            # Eval
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                if gate:
                    gate.eval()
                if autoenc:
                    autoenc.eval()

                v_loss, v_acc, v_loss_noinj, v_cos = 0.0, 0.0, 0.0, 0.0
                with torch.no_grad():
                    for bi in range(n_eval_batches):
                        eidx = eval_indices[eval_idx][bi]
                        vim = test_images[eidx].to(device)
                        vlb = test_labels[eidx].to(device)

                        if cb is not None:
                            ecb = make_cb()
                            vlog, vl, _ = model(
                                vim, vlb, return_intermediates=True,
                                cerebellar_fn=ecb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                        else:
                            vlog, vl, _ = model(
                                vim, vlb, return_intermediates=True,
                            )
                        v_loss += vl.item()
                        v_acc += (vlog.argmax(-1) == vlb).float().mean().item()

                        vlog2, vl2, vi2 = model(vim, vlb, return_intermediates=True)
                        v_loss_noinj += vl2.item()

                        src = vi2[predict_from]
                        tgt = vi2[predict_to]
                        pred = fwd_model(src)
                        v_cos += F.cosine_similarity(pred, tgt, dim=-1).mean().item()

                n = n_eval_batches
                v_loss /= n
                v_acc /= n
                v_loss_noinj /= n
                v_cos /= n

                history["val_loss"].append((step, v_loss))
                history["val_acc"].append((step, v_acc))
                history["val_loss_no_inj"].append((step, v_loss_noinj))
                history["val_cosine"].append((step, v_cos))
                if gate:
                    history["gate_norm"].append((step, gate.injection_norm()))

                extra = ""
                if has_inj:
                    delta = v_loss - v_loss_noinj
                    extra = f" Δ={delta:+.4f} gate={gate.injection_norm():.3f}"

                print(f"  step {step:5d}: loss={v_loss:.4f} acc={v_acc:.4f}"
                      f"{extra} | fwd_cos={v_cos:.4f}")

                eval_idx += 1

        return model, fwd_model, gate, history

    # --- Train all conditions ---
    all_models = {}
    all_histories = {}
    for cond in CONDITIONS:
        m, fm, g, h = train_run(cond)
        all_models[cond] = (m, fm, g)
        all_histories[cond] = h

    # =============================================
    # Self-knowledge probes
    # =============================================
    print(f"\n{'=' * 60}")
    print("  SELF-KNOWLEDGE PROBES")
    print(f"{'=' * 60}")

    layer_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    probe_r2 = {}

    for cond in CONDITIONS:
        model, fwd_model, _ = all_models[cond]
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
                residual_vecs.append((tgt - pred).cpu())
                for k in layer_names:
                    layer_acts[k].append(vi[k].cpu())

        res_flat = torch.cat(residual_vecs, dim=0).reshape(-1, n_embd)
        for k in layer_names:
            layer_acts[k] = torch.cat(layer_acts[k], dim=0).reshape(-1, n_embd)

        cond_r2 = {}
        for layer_name in layer_names:
            acts = layer_acts[layer_name]
            n_total = acts.shape[0]
            n_train = int(0.8 * n_total)
            perm = torch.randperm(
                n_total, generator=torch.Generator().manual_seed(seed)
            )
            tr, te = perm[:n_train], perm[n_train:]

            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            X_tr = acts[tr].to(device)
            Y_tr = res_flat[tr].to(device)
            X_te = acts[te].to(device)
            Y_te = res_flat[te].to(device)

            for _ in range(300):
                sidx = torch.randint(n_train, (min(4096, n_train),))
                ploss = F.mse_loss(probe(X_tr[sidx]), Y_tr[sidx])
                opt.zero_grad()
                ploss.backward()
                opt.step()

            with torch.no_grad():
                mse = F.mse_loss(probe(X_te), Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            cond_r2[layer_name] = r2

        probe_r2[cond] = cond_r2

    # Print probe comparison
    print(f"\n  {'Layer':<14s}", end="")
    for cond in CONDITIONS:
        print(f" {cond:>11s}", end="")
    print()
    for layer in layer_names:
        print(f"  {layer:<14s}", end="")
        for cond in CONDITIONS:
            print(f" {probe_r2[cond][layer]:>11.4f}", end="")
        print()

    # =============================================
    # Robustness test
    # =============================================
    print(f"\n{'=' * 60}")
    print("  ROBUSTNESS TEST")
    print(f"{'=' * 60}")

    eps_values = [0.5, 1.0, 2.0]
    rob_results = {}

    for cond in CONDITIONS:
        model, fwd_model, gate = all_models[cond]
        model.eval()
        fwd_model.eval()
        if gate:
            gate.eval()

        has_inj = cond != "open_loop"
        cond_rob = {}

        for eps in eps_values:
            deltas = []
            with torch.no_grad():
                for bi in range(min(probe_batches, 20)):
                    pidx = probe_indices[bi]
                    pim = test_images[pidx].to(device)
                    plb = test_labels[pidx].to(device)

                    if has_inj and gate:
                        def cb_base(act):
                            if cond == "forward":
                                return gate(fwd_model(act))
                            elif cond == "random_proj":
                                return gate(act @ random_proj_weight.T)
                            else:
                                return gate(fwd_model(act))
                        _, bl, _ = model(
                            pim, plb, return_intermediates=True,
                            cerebellar_fn=cb_base,
                            cerebellar_input_block=cerebellar_input_block,
                            cerebellar_inject_block=inject_after_block,
                        )
                    else:
                        _, bl, _ = model(pim, plb, return_intermediates=True)

                    torch.manual_seed(seed + bi + 1000)
                    noise = torch.randn(
                        len(pidx), n_positions, n_embd, device=device
                    ) * eps

                    if has_inj and gate:
                        def cb_pert(act):
                            if cond == "forward":
                                return gate(fwd_model(act))
                            elif cond == "random_proj":
                                return gate(act @ random_proj_weight.T)
                            else:
                                return gate(fwd_model(act))
                        _, pl, _ = model(
                            pim, plb, return_intermediates=True,
                            cerebellar_fn=cb_pert,
                            cerebellar_input_block=cerebellar_input_block,
                            cerebellar_inject_block=inject_after_block,
                            perturbation=(inject_after_block, noise),
                        )
                    else:
                        _, pl, _ = model(
                            pim, plb, return_intermediates=True,
                            perturbation=(inject_after_block, noise),
                        )

                    deltas.append(pl.item() - bl.item())

            mean_d = float(np.mean(deltas))
            cond_rob[str(eps)] = mean_d

        rob_results[cond] = cond_rob

    print(f"\n  {'Condition':<14s}", end="")
    for eps in eps_values:
        print(f" {'ε=' + str(eps):>8s}", end="")
    print()
    for cond in CONDITIONS:
        print(f"  {cond:<14s}", end="")
        for eps in eps_values:
            print(f" {rob_results[cond][str(eps)]:+8.4f}", end="")
        print()

    # Robustness ratio vs open_loop at eps=1.0
    ol_rob = rob_results["open_loop"]["1.0"]
    if ol_rob > 0:
        print(f"\n  Robustness ratio vs open_loop (eps=1.0):")
        for cond in CONDITIONS:
            ratio = rob_results[cond]["1.0"] / ol_rob
            print(f"    {cond:<14s}: {ratio:.3f}")

    # =============================================
    # Save
    # =============================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_baseline_battery/"
                f"{model_tag}/{gap_tag}")
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "conditions": CONDITIONS,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "n_steps": n_steps, "lr": lr, "seed": seed,
            "shift_k": shift_k,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
        },
        "final_metrics": {},
        "probes": probe_r2,
        "robustness": rob_results,
    }

    for cond in CONDITIONS:
        h = all_histories[cond]
        fm = {
            "final_val_loss": h["val_loss"][-1][1],
            "final_val_acc": h["val_acc"][-1][1],
            "final_val_loss_no_inj": h["val_loss_no_inj"][-1][1],
            "final_fwd_cosine": h["val_cosine"][-1][1],
        }
        if "gate_norm" in h and h["gate_norm"]:
            fm["final_gate_norm"] = h["gate_norm"][-1][1]
        fm["injection_benefit"] = fm["final_val_loss"] - fm["final_val_loss_no_inj"]
        result["final_metrics"][cond] = fm

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    # Save histories separately (large)
    with open(os.path.join(save_dir, "histories.json"), "w") as f:
        json.dump(all_histories, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Condition':<14s} {'Acc':>6s} {'Loss':>6s} {'Δ_inj':>7s}"
          f" {'Gate':>6s} {'Cos':>6s}")
    for cond in CONDITIONS:
        fm = result["final_metrics"][cond]
        gn = fm.get("final_gate_norm", 0)
        print(f"  {cond:<14s} {fm['final_val_acc']:.4f} "
              f"{fm['final_val_loss']:.4f} "
              f"{fm['injection_benefit']:+.4f} "
              f"{gn:.3f} {fm['final_fwd_cosine']:.4f}")

    print(f"\n  Self-knowledge depth profile (R² at post_block3):")
    for cond in CONDITIONS:
        r2 = probe_r2[cond]["post_block3"]
        print(f"    {cond:<14s}: {r2:.4f}")

    return result


@app.local_entrypoint()
def main(
    n_steps: int = 5000,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    seed: int = 42,
):
    result = a2a_mnist_baseline_battery.remote(
        n_steps=n_steps, n_layer=n_layer, n_head=n_head,
        n_embd=n_embd, seed=seed,
    )
    print("\nMNIST baseline battery complete.")
    for cond in CONDITIONS:
        fm = result["final_metrics"][cond]
        print(f"  {cond}: acc={fm['final_val_acc']:.4f} "
              f"Δ={fm['injection_benefit']:+.4f}")
