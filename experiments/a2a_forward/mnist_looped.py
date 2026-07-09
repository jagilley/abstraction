"""Phase 1: plain looped (weight-shared) ViT on MNIST -- no FM injection yet.

Goal: stand up the looped-transformer substrate for the self-model-needs-a-loop
hypothesis (ideas/self_model_needs_a_loop.md) and confirm it behaves as expected
BEFORE adding forward-model injection. Three things we verify here:

  1. It trains to good accuracy (compare to the 4-layer feedforward ViT baseline).
  2. Convergence dynamics: does ||s_{t+1} - s_t|| shrink across steps -- i.e. does
     the loop settle toward a fixed point? (Prerequisite for the fixed-point
     self-consistency argument the theory rests on.)
  3. Compute-vs-T generalization: train at T steps, eval at 1..T_max. A genuine
     fixed point -> accuracy climbs then stays stable past training-T. A
     non-converging loop degrades when run longer than trained.

Phase 2 (separate file) will add the gated FM forecast via step_inject_fn.

Run:
  modal run --detach a2a_forward/mnist_looped.py::a2a_mnist_looped
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist_looped(
    n_loop_layers: int = 1,
    prelude_layers: int = 0,        # non-shared feedforward bookend (front)
    coda_layers: int = 0,           # non-shared feedforward bookend (back)
    deep_sup: bool = False,         # apply readout loss at every supervised step
    deep_sup_frac: float = 0.5,     # supervise the last frac of steps
    n_head: int = 4,
    n_embd: int = 128,
    train_steps_loop: int = 8,      # recurrence depth T at train time
    eval_t_max: int = 20,           # sweep eval T from 1..eval_t_max
    inject_input_each_step: bool = True,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    n_steps: int = 5000,            # optimizer steps
    eval_interval: int = 100,
    n_eval_batches: int = 5,
    seed: int = 42,
    train_ff_baseline: bool = True,
    ff_n_layer: int = 4,            # feedforward baseline depth
):
    import os
    import torch
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.looped_vit import LoopedViT
    from a2a_forward.vit import ViT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MNIST looped-ViT phase 1 on {device}")
    print(f"  Loop: {n_loop_layers} block(s) x T={train_steps_loop}, "
          f"{n_head}H {n_embd}D, input_inject={inject_input_each_step}")

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

    # --- Pre-generate indices for reproducibility (shared across models) ---
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
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

    def make_looped():
        torch.manual_seed(seed)
        return LoopedViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd,
            n_steps=train_steps_loop,
            prelude_layers=prelude_layers, coda_layers=coda_layers,
            inject_input_each_step=inject_input_each_step,
        ).to(device)

    def make_ff():
        torch.manual_seed(seed)
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=ff_n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)

    use_deep_sup = deep_sup
    sup_start = int(train_steps_loop * (1.0 - deep_sup_frac))

    def train_model(model, is_looped, tag):
        print(f"\n{'=' * 60}\n  Training {tag}\n{'=' * 60}")
        if is_looped and use_deep_sup:
            print(f"  Deep supervision: readout loss on steps "
                  f"{sup_start}..{train_steps_loop - 1}")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        eval_idx = 0
        for step in range(n_steps):
            model.train()
            idx = train_indices[step]
            images = train_images[idx].to(device)
            labels = train_labels[idx].to(device)

            if is_looped and use_deep_sup:
                logits, _, step_logits = model(images, readout_all_steps=True)
                # average CE over the supervised tail of steps
                sup = step_logits[sup_start:]  # (K, B, C)
                loss = torch.stack([
                    torch.nn.functional.cross_entropy(sl, labels) for sl in sup
                ]).mean()
            else:
                logits, loss = model(images, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc = (logits.argmax(-1) == labels).float().mean().item()

            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                v_loss, v_acc = 0.0, 0.0
                with torch.no_grad():
                    for bi in range(n_eval_batches):
                        eidx = eval_indices[eval_idx][bi]
                        vim = test_images[eidx].to(device)
                        vlb = test_labels[eidx].to(device)
                        vlog, vl = model(vim, vlb)
                        v_loss += vl.item()
                        v_acc += (vlog.argmax(-1) == vlb).float().mean().item()
                v_loss /= n_eval_batches
                v_acc /= n_eval_batches
                history["train_loss"].append((step, loss.item()))
                history["train_acc"].append((step, acc))
                history["val_loss"].append((step, v_loss))
                history["val_acc"].append((step, v_acc))
                print(f"  step {step:5d}: train_loss={loss.item():.4f} "
                      f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}")
                eval_idx += 1
        return history

    # =============================================
    # Train the looped model
    # =============================================
    loop_model = make_looped()
    loop_history = train_model(loop_model, True, f"LOOPED (T={train_steps_loop})")

    # =============================================
    # Convergence dynamics on the eval set (at train T)
    # =============================================
    print(f"\n{'=' * 60}\n  CONVERGENCE DYNAMICS (T={train_steps_loop})\n{'=' * 60}")
    loop_model.eval()
    conv_delta = np.zeros(train_steps_loop)
    conv_rel = np.zeros(train_steps_loop)
    n_conv_batches = 20
    with torch.no_grad():
        for bi in range(n_conv_batches):
            eidx = eval_indices[bi % n_evals][0]
            vim = test_images[eidx].to(device)
            _, _, traj = loop_model(vim, return_trajectory=True)
            conv_delta += np.array(traj["delta"])
            conv_rel += np.nan_to_num(np.array(traj["rel_delta"]))
    conv_delta /= n_conv_batches
    conv_rel /= n_conv_batches
    print("  step:  ||s_{t+1}-s_t||   rel(||.||/||s_t||)")
    for t in range(train_steps_loop):
        print(f"    {t:2d}:   {conv_delta[t]:10.4f}      {conv_rel[t]:8.4f}")

    # =============================================
    # Accuracy vs eval-T sweep (fixed-point generalization)
    # =============================================
    print(f"\n{'=' * 60}\n  ACCURACY vs EVAL-T  (trained at T={train_steps_loop})\n{'=' * 60}")
    acc_vs_T = {}
    with torch.no_grad():
        for T in range(1, eval_t_max + 1):
            v_acc, v_loss = 0.0, 0.0
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = test_images[eidx].to(device)
                vlb = test_labels[eidx].to(device)
                vlog, vl = loop_model(vim, vlb, n_steps=T)
                v_acc += (vlog.argmax(-1) == vlb).float().mean().item()
                v_loss += vl.item()
            v_acc /= n_eval_batches
            v_loss /= n_eval_batches
            acc_vs_T[str(T)] = {"acc": v_acc, "loss": v_loss}
            marker = "  <-- train T" if T == train_steps_loop else ""
            print(f"    T={T:2d}: acc={v_acc:.4f} loss={v_loss:.4f}{marker}")

    # =============================================
    # Feedforward baseline (reference)
    # =============================================
    ff_history = None
    if train_ff_baseline:
        ff_model = make_ff()
        ff_history = train_model(ff_model, False, f"FEEDFORWARD ({ff_n_layer}L)")

    # =============================================
    # Save
    # =============================================
    model_tag = f"looped_{n_loop_layers}x{train_steps_loop}_{n_head}H_{n_embd}D"
    if prelude_layers or coda_layers:
        model_tag += f"_pre{prelude_layers}coda{coda_layers}"
    if deep_sup:
        model_tag += f"_deepsup{deep_sup_frac}"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_looped/{model_tag}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(loop_model.state_dict(), os.path.join(save_dir, "loop_model.pt"))

    result = {
        "config": {
            "n_loop_layers": n_loop_layers, "n_head": n_head, "n_embd": n_embd,
            "prelude_layers": prelude_layers, "coda_layers": coda_layers,
            "deep_sup": deep_sup, "deep_sup_frac": deep_sup_frac,
            "train_steps_loop": train_steps_loop, "eval_t_max": eval_t_max,
            "inject_input_each_step": inject_input_each_step,
            "patch_size": patch_size, "lr": lr, "n_steps": n_steps,
            "batch_size": batch_size, "seed": seed, "ff_n_layer": ff_n_layer,
        },
        "looped": {
            "history": loop_history,
            "final_val_loss": loop_history["val_loss"][-1][1],
            "final_val_acc": loop_history["val_acc"][-1][1],
        },
        "convergence": {
            "delta": conv_delta.tolist(),
            "rel_delta": conv_rel.tolist(),
        },
        "acc_vs_T": acc_vs_T,
    }
    if ff_history is not None:
        result["feedforward"] = {
            "history": ff_history,
            "final_val_loss": ff_history["val_loss"][-1][1],
            "final_val_acc": ff_history["val_acc"][-1][1],
        }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")

    # --- Summary ---
    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    print(f"  Looped (T={train_steps_loop}): "
          f"val_acc={result['looped']['final_val_acc']:.4f} "
          f"val_loss={result['looped']['final_val_loss']:.4f}")
    if ff_history is not None:
        print(f"  Feedforward ({ff_n_layer}L): "
              f"val_acc={result['feedforward']['final_val_acc']:.4f} "
              f"val_loss={result['feedforward']['final_val_loss']:.4f}")
    best_T = max(acc_vs_T, key=lambda k: acc_vs_T[k]["acc"])
    print(f"  Best eval-T: {best_T} (acc={acc_vs_T[best_T]['acc']:.4f})")
    print(f"  Convergence: rel_delta {conv_rel[0]:.3f} (step 0) -> "
          f"{conv_rel[-1]:.3f} (step {train_steps_loop - 1})")
    return result


@app.local_entrypoint()
def main(
    n_loop_layers: int = 1,
    n_embd: int = 128,
    train_steps_loop: int = 8,
    n_steps: int = 5000,
    seed: int = 42,
):
    result = a2a_mnist_looped.remote(
        n_loop_layers=n_loop_layers, n_embd=n_embd,
        train_steps_loop=train_steps_loop, n_steps=n_steps, seed=seed,
    )
    print("\nLooped-ViT phase 1 complete.")
    print(f"Looped val_acc={result['looped']['final_val_acc']:.4f}")
    if "feedforward" in result:
        print(f"FF val_acc={result['feedforward']['final_val_acc']:.4f}")
