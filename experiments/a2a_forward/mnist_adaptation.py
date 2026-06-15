"""MNIST OOD adaptation speed experiment.

Tests whether internalized self-knowledge (from wake-sleep distillation)
produces faster adaptation to out-of-distribution data.

Loads the 4 compute-matched checkpoints from the wake-sleep comparison
(WS, CL, OL, KD -- all at 8400 main-model gradient steps) and measures:

1. Zero-shot OOD accuracy (rotated MNIST at 15, 30, 45, 60, 90 degrees)
2. Fine-tuning learning curves (accuracy vs adaptation step)
3. Area under the learning curve (adaptation efficiency)
4. Catastrophic forgetting (ID accuracy after OOD fine-tuning)

OOD domain: Rotated MNIST. Rotation preserves the underlying task (digit
classification) while shifting the input distribution. The model needs
the same computational structure applied to transformed inputs. This tests
whether representational organization (from self-referential training)
helps adaptation, distinct from data statistics.

Key comparisons:
- WS vs KD: isolates self-referential component (both have distillation)
- CL vs OL: isolates closed-loop effect (no distillation)
- WS vs OL: combined effect
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mnist_adaptation(
    adapt_steps: int = 500,
    adapt_lr: float = 1e-4,
    eval_interval: int = 10,
    batch_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    seed: int = 42,
):
    import os
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.ndimage import rotate as scipy_rotate
    from datasets import load_dataset
    from a2a_forward.vit import ViT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rotation_angles = [15, 30, 45, 60, 90]
    conditions = ["WS", "CL", "OL", "KD"]

    print(f"MNIST OOD ADAPTATION EXPERIMENT on {device}")
    print(f"  Rotation angles: {rotation_angles}")
    print(f"  Adapt: {adapt_steps} steps, lr={adapt_lr}")
    print(f"  Eval every {eval_interval} steps, full test set")

    # =============================================
    # SETUP
    # =============================================
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

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

    # Load checkpoints from wake-sleep comparison
    print("\nLoading checkpoints...")
    ckpt_dir = (f"{DATA_DIR}/a2a_forward/mnist_wake_sleep_comparison"
                f"/vit_{n_layer}L_{n_head}H_{n_embd}D"
                f"/post_block0_to_post_block3")

    def make_vit():
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)

    checkpoint_files = {
        "WS": "ws_model.pt", "CL": "cl_model.pt",
        "OL": "ol_model.pt", "KD": "kd_model.pt",
    }
    checkpoints = {}
    for cond, fname in checkpoint_files.items():
        path = os.path.join(ckpt_dir, fname)
        checkpoints[cond] = torch.load(path, map_location=device, weights_only=True)
        print(f"  Loaded {cond}")

    # Pre-generate training batch indices (shared across all conditions)
    adapt_gen = torch.Generator().manual_seed(seed + 500)
    train_indices = [
        torch.randint(len(train_images), (batch_size,), generator=adapt_gen)
        for _ in range(adapt_steps)
    ]

    # =============================================
    # HELPERS
    # =============================================
    def eval_full(model, images, labels):
        """Evaluate on the full dataset."""
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

    # =============================================
    # ID PERFORMANCE
    # =============================================
    print("\n--- ID performance ---")
    id_perf = {}
    for cond in conditions:
        model = make_vit()
        model.load_state_dict(checkpoints[cond])
        loss, acc = eval_full(model, test_images, test_labels)
        id_perf[cond] = {"loss": loss, "acc": acc}
        print(f"  {cond}: loss={loss:.4f} acc={acc:.4f}")
        del model
        torch.cuda.empty_cache()

    # =============================================
    # ADAPTATION
    # =============================================
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
            model.load_state_dict(checkpoints[cond])

            # Zero-shot OOD eval
            zs_loss, zs_acc = eval_full(model, rot_te, test_labels)
            print(f"    Zero-shot: loss={zs_loss:.4f} acc={zs_acc:.3f}")

            # Fine-tune on rotated training set
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=adapt_lr, weight_decay=0.01)

            curve = [(0, zs_loss, zs_acc)]

            for step in range(1, adapt_steps + 1):
                model.train()
                idx = train_indices[step - 1]
                imgs = rot_tr[idx].to(device)
                labs = train_labels[idx].to(device)

                logits, loss = model(imgs, labs)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % eval_interval == 0 or step == adapt_steps:
                    v_loss, v_acc = eval_full(model, rot_te, test_labels)
                    curve.append((step, v_loss, v_acc))
                    if step % 100 == 0 or step == adapt_steps:
                        print(f"    step {step:4d}: loss={v_loss:.4f} acc={v_acc:.3f}")

            # Catastrophic forgetting: ID accuracy after OOD fine-tuning
            id_loss_post, id_acc_post = eval_full(model, test_images, test_labels)
            id_drop = id_acc_post - id_perf[cond]["acc"]
            print(f"    Forgetting: ID acc={id_acc_post:.3f} "
                  f"(delta={id_drop:+.3f} from {id_perf[cond]['acc']:.3f})")

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

    # =============================================
    # SAVE
    # =============================================
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_adaptation"
                 f"/vit_{n_layer}L_{n_head}H_{n_embd}D")
    os.makedirs(save_root, exist_ok=True)

    result = {
        "config": {
            "adapt_steps": adapt_steps,
            "adapt_lr": adapt_lr,
            "eval_interval": eval_interval,
            "batch_size": batch_size,
            "rotation_angles": rotation_angles,
            "conditions": conditions,
            "seed": seed,
        },
        "id_performance": id_perf,
        "adaptation": {str(k): v for k, v in all_results.items()},
    }

    with open(os.path.join(save_root, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  ID performance:")
    for c in conditions:
        print(f"    {c}: acc={id_perf[c]['acc']:.4f}")

    header = f"  {'Angle':>6s} | " + " | ".join(f"{c:>7s}" for c in conditions)
    sep = f"  {'-'*6}-+-" + "-+-".join(["-" * 7] * len(conditions))

    print(f"\n  Zero-shot OOD accuracy:")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["zero_shot"]["acc"] for c in conditions]
        print(f"  {angle:5d}  | " + " | ".join(f"{v:7.4f}" for v in vals))

    print(f"\n  Post-adaptation accuracy ({adapt_steps} steps):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["final"]["acc"] for c in conditions]
        print(f"  {angle:5d}  | " + " | ".join(f"{v:7.4f}" for v in vals))

    print(f"\n  Adaptation AUC (time-weighted mean accuracy, higher=better):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["auc"] for c in conditions]
        print(f"  {angle:5d}  | " + " | ".join(f"{v:7.4f}" for v in vals))

    print(f"\n  Steps to 90% of ID accuracy (lower=faster):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = []
        for c in conditions:
            s = all_results[angle][c]["steps_to_90pct_id"]
            vals.append(f"{s:7d}" if s is not None else f"   >{adapt_steps}")
        print(f"  {angle:5d}  | " + " | ".join(vals))

    print(f"\n  Forgetting (ID accuracy drop after OOD adaptation):")
    print(header)
    print(sep)
    for angle in rotation_angles:
        vals = [all_results[angle][c]["forgetting"]["id_drop"] for c in conditions]
        print(f"  {angle:5d}  | " + " | ".join(f"{v:+7.4f}" for v in vals))

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    adapt_steps: int = 500,
    adapt_lr: float = 1e-4,
):
    result = a2a_mnist_adaptation.remote(
        adapt_steps=adapt_steps,
        adapt_lr=adapt_lr,
    )
    print("\nAdaptation experiment complete.")
