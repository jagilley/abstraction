"""Classic MAML on rotated MNIST.

Pedagogical implementation of FOMAML (Finn et al., 2017) on a simple
task distribution: classify MNIST digits at different rotation angles.

Three conditions, all starting from the same initialization:
1. MAML: meta-trained across random rotations via FOMAML, then adapted
2. Pretrained: trained on 0-degree MNIST, then adapted
3. Scratch: random init (no training), then adapted

The key question: given a fixed compute budget, which strategy for
USING that compute produces the fastest few-shot adaptation?

MAML uses bilevel optimization:
  - Inner loop: K SGD steps on a task's support set (adapt to one rotation)
  - Outer loop: evaluate adapted model on query set, update initialization
  - The meta-learned initialization is positioned in weight space such that
    a few gradient steps can reach any rotation quickly

FOMAML simplifies full MAML by dropping second-order terms: instead of
backpropagating through the inner loop, it just uses the gradient of
the query loss w.r.t. the adapted parameters as the meta-gradient.
This is the same approximation used in the learning gate experiments.
"""

import json
import math
import copy
import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
        "datasets",
        "matplotlib",
    )
    .add_local_python_source("meta_learning")
    .add_local_python_source("a2a_forward")
)

volume = modal.Volume.from_name("language-reduction-data", create_if_missing=True)
app = modal.App("meta-learning", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def maml_rotation(
    # Meta-learning
    meta_steps: int = 2000,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    meta_lr: float = 3e-4,
    k_shot: int = 10,
    q_query: int = 15,
    # Model
    n_layer: int = 2,
    n_head: int = 2,
    n_embd: int = 64,
    patch_size: int = 4,
    # Evaluation
    test_angles: str = "30,60,90,120,150,180",
    adapt_steps: int = 50,
    # Pretraining
    pretrain_steps: int = 2000,
    pretrain_lr: float = 3e-4,
    pretrain_batch_size: int = 128,
    seed: int = 42,
):
    import os
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from a2a_forward.vit import ViT

    device = "cuda"
    test_angle_list = [int(a) for a in test_angles.split(",")]
    adapt_eval_at = [0, 1, 2, 3, 5, 10, 20, 50]
    adapt_eval_at = [s for s in adapt_eval_at if s <= adapt_steps]

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("FOMAML ON ROTATED MNIST")
    print("=" * 60)
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D ViT")
    print(f"  MAML: {meta_steps} meta-steps, {inner_steps} inner steps, "
          f"inner_lr={inner_lr}, meta_lr={meta_lr}")
    print(f"  Tasks: 10-way {k_shot}-shot, {q_query} query/class")
    print(f"  Pretrain: {pretrain_steps} steps, lr={pretrain_lr}")
    print(f"  Test angles: {test_angle_list}")
    total_maml_grads = meta_steps * (inner_steps + 1)
    total_pretrain_grads = pretrain_steps
    print(f"  Compute: MAML={total_maml_grads} grad steps, "
          f"Pretrain={total_pretrain_grads} grad steps")

    # =========================================
    # Load MNIST
    # =========================================
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    class_indices = {}
    for c in range(10):
        class_indices[c] = (train_labels == c).nonzero(as_tuple=True)[0].numpy()

    # =========================================
    # GPU rotation via affine grid sampling
    # =========================================
    def rotate_batch(images, angle_deg):
        angle_rad = angle_deg * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=torch.float32, device=images.device,
        ).unsqueeze(0).expand(images.size(0), -1, -1)
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        return F.grid_sample(images, grid, align_corners=False,
                             padding_mode="zeros", mode="bilinear")

    # =========================================
    # Task sampling
    # =========================================
    def sample_task(angle, rng):
        support_idx, query_idx = [], []
        for c in range(10):
            idx = rng.choice(class_indices[c], size=k_shot + q_query, replace=False)
            support_idx.extend(idx[:k_shot])
            query_idx.extend(idx[k_shot:])
        s_imgs = rotate_batch(train_images[support_idx].to(device), angle)
        s_lbls = train_labels[support_idx].to(device)
        q_imgs = rotate_batch(train_images[query_idx].to(device), angle)
        q_lbls = train_labels[query_idx].to(device)
        return s_imgs, s_lbls, q_imgs, q_lbls

    # =========================================
    # FOMAML inner loop
    # =========================================
    def fomaml_inner(model, support_imgs, support_labels, steps, lr):
        fast = copy.deepcopy(model)
        opt = torch.optim.SGD(fast.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            _, loss = fast(support_imgs, support_labels)
            loss.backward()
            opt.step()
        return fast

    # =========================================
    # Evaluation helpers
    # =========================================
    def eval_acc(model, images, labels, batch_size=512):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                b_img = images[i:i + batch_size].to(device)
                b_lbl = labels[i:i + batch_size].to(device)
                logits, _ = model(b_img)
                correct += (logits.argmax(1) == b_lbl).sum().item()
                total += len(b_lbl)
        model.train()
        return correct / total

    def adaptation_curve(init_model, angle, steps, lr, eval_at):
        rot_test = rotate_batch(test_images.to(device), angle)

        rng_adapt = np.random.RandomState(123)
        s_idx = []
        for c in range(10):
            s_idx.extend(rng_adapt.choice(class_indices[c], size=k_shot,
                                          replace=False))
        support = rotate_batch(train_images[s_idx].to(device), angle)
        support_lbl = train_labels[s_idx].to(device)

        model = copy.deepcopy(init_model)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        curve = {"steps": [], "accuracy": []}

        for step in range(steps + 1):
            if step in eval_at:
                acc = eval_acc(model, rot_test, test_labels)
                curve["steps"].append(step)
                curve["accuracy"].append(acc)
            if step < steps:
                opt.zero_grad()
                _, loss = model(support, support_lbl)
                loss.backward()
                opt.step()

        return curve

    # =========================================
    # Shared initialization
    # =========================================
    torch.manual_seed(seed)
    init_model = ViT(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                     patch_size=patch_size).to(device)

    # =========================================
    # CONDITION 1: MAML
    # =========================================
    print("\n" + "=" * 60)
    print("MAML META-TRAINING")
    print("=" * 60)

    maml_model = copy.deepcopy(init_model)
    meta_opt = torch.optim.Adam(maml_model.parameters(), lr=meta_lr)
    rng = np.random.RandomState(seed)
    t0 = time.time()
    maml_log = {"steps": [], "query_loss": [], "query_acc": []}

    for step in range(1, meta_steps + 1):
        angle = rng.uniform(0, 360)
        s_imgs, s_lbls, q_imgs, q_lbls = sample_task(angle, rng)

        fast = fomaml_inner(maml_model, s_imgs, s_lbls, inner_steps, inner_lr)

        meta_opt.zero_grad()
        logits, q_loss = fast(q_imgs, q_lbls)
        q_acc = (logits.argmax(1) == q_lbls).float().mean().item()
        q_loss.backward()

        for p_meta, p_fast in zip(maml_model.parameters(), fast.parameters()):
            if p_fast.grad is not None:
                p_meta.grad = p_fast.grad.clone()
        meta_opt.step()
        del fast

        if step % 200 == 0 or step == 1:
            maml_log["steps"].append(step)
            maml_log["query_loss"].append(q_loss.item())
            maml_log["query_acc"].append(q_acc)
            print(f"  [{step:5d}/{meta_steps}] q_loss={q_loss.item():.4f}  "
                  f"q_acc={q_acc:.3f}  angle={angle:.0f}deg  "
                  f"({time.time() - t0:.0f}s)")

    maml_time = time.time() - t0
    print(f"MAML done: {maml_time:.0f}s")

    # =========================================
    # CONDITION 2: Pretrained on 0deg
    # =========================================
    print("\n" + "=" * 60)
    print(f"PRETRAINING on 0deg MNIST ({pretrain_steps} steps)")
    print("=" * 60)

    pretrained_model = copy.deepcopy(init_model)
    pre_opt = torch.optim.Adam(pretrained_model.parameters(), lr=pretrain_lr)
    t0 = time.time()

    for step in range(1, pretrain_steps + 1):
        idx = torch.randint(0, len(train_images), (pretrain_batch_size,))
        imgs = train_images[idx].to(device)
        lbls = train_labels[idx].to(device)

        pre_opt.zero_grad()
        _, loss = pretrained_model(imgs, lbls)
        loss.backward()
        pre_opt.step()

        if step % 500 == 0:
            acc = eval_acc(pretrained_model, test_images, test_labels)
            print(f"  [{step:5d}/{pretrain_steps}] loss={loss.item():.4f}  "
                  f"val_acc={acc:.3f}  ({time.time() - t0:.0f}s)")

    pretrain_time = time.time() - t0
    pre_0deg_acc = eval_acc(pretrained_model, test_images, test_labels)
    print(f"Pretraining done: {pretrain_time:.0f}s, 0deg acc: {pre_0deg_acc:.3f}")

    # =========================================
    # CONDITION 3: Scratch (same init, no training)
    # =========================================
    scratch_model = copy.deepcopy(init_model)

    # =========================================
    # ADAPTATION CURVES
    # =========================================
    print("\n" + "=" * 60)
    print("ADAPTATION CURVES")
    print("=" * 60)

    conditions = {
        "maml": maml_model,
        "pretrained": pretrained_model,
        "scratch": scratch_model,
    }
    adaptation_results = {}

    for angle in test_angle_list:
        print(f"\n--- {angle}deg ---")
        adaptation_results[angle] = {}
        for name, model in conditions.items():
            curve = adaptation_curve(model, angle, adapt_steps, inner_lr,
                                     adapt_eval_at)
            adaptation_results[angle][name] = curve
            print(f"  {name:12s}: 0-shot={curve['accuracy'][0]:.3f}  "
                  f"final={curve['accuracy'][-1]:.3f}")

    # =========================================
    # SUMMARY TABLE
    # =========================================
    step5_idx = adapt_eval_at.index(5) if 5 in adapt_eval_at else -1

    print("\n" + "=" * 60)
    print("SUMMARY: accuracy at 0 / 5 / 50 adaptation steps")
    print("=" * 60)
    print(f"{'Angle':>6} | {'MAML':>21s} | {'Pretrained':>21s} | "
          f"{'Scratch':>21s}")
    print(f"{'':>6} | {'0':>6} {'5':>6} {'50':>6} | "
          f"{'0':>6} {'5':>6} {'50':>6} | "
          f"{'0':>6} {'5':>6} {'50':>6}")
    print("-" * 76)
    for angle in test_angle_list:
        row = f"{angle:6d} |"
        for name in ["maml", "pretrained", "scratch"]:
            a = adaptation_results[angle][name]["accuracy"]
            row += f" {a[0]:.3f} {a[step5_idx]:.3f} {a[-1]:.3f} |"
        print(row)

    # =========================================
    # PLOTS
    # =========================================
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = os.path.join(DATA_DIR, "meta_learning", "maml_rotation")
    os.makedirs(save_dir, exist_ok=True)

    colors = {"maml": "#2196F3", "pretrained": "#FF9800", "scratch": "#9E9E9E"}
    labels = {"maml": "MAML", "pretrained": "Pretrained (0deg)",
              "scratch": "From scratch"}

    # Plot 1: adaptation curves per angle
    n_cols = min(3, len(test_angle_list))
    n_rows = math.ceil(len(test_angle_list) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if len(test_angle_list) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, angle in enumerate(test_angle_list):
        ax = axes[i]
        for name in ["maml", "pretrained", "scratch"]:
            curve = adaptation_results[angle][name]
            ax.plot(curve["steps"], curve["accuracy"],
                    color=colors[name], label=labels[name], linewidth=2,
                    marker="o", markersize=3)
        ax.set_title(f"{angle}deg rotation", fontsize=12)
        ax.set_xlabel("Adaptation steps (SGD)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("FOMAML vs Pretrained vs Scratch: adaptation to rotated MNIST",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "adaptation_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Plot 2: accuracy at step 5 vs angle
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["maml", "pretrained", "scratch"]:
        accs = [adaptation_results[a][name]["accuracy"][step5_idx]
                for a in test_angle_list]
        ax.plot(test_angle_list, accs, color=colors[name], label=labels[name],
                linewidth=2, marker="o", markersize=6)
    ax.set_xlabel("Rotation angle (degrees)", fontsize=12)
    ax.set_ylabel("Accuracy after 5 adaptation steps", fontsize=12)
    ax.set_title(f"Few-shot adaptation: {inner_steps} SGD steps on "
                 f"{k_shot * 10} examples", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary.png"), dpi=150)
    plt.close()

    # Plot 3: MAML training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(maml_log["steps"], maml_log["query_loss"], color="#2196F3")
    ax1.set_xlabel("Meta-step")
    ax1.set_ylabel("Query loss")
    ax1.set_title("MAML meta-training: query loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(maml_log["steps"], maml_log["query_acc"], color="#2196F3")
    ax2.set_xlabel("Meta-step")
    ax2.set_ylabel("Query accuracy")
    ax2.set_title("MAML meta-training: query accuracy")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "maml_training.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {save_dir}/")

    # =========================================
    # SAVE
    # =========================================
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results = {
        "config": {
            "meta_steps": meta_steps, "inner_steps": inner_steps,
            "inner_lr": inner_lr, "meta_lr": meta_lr,
            "k_shot": k_shot, "q_query": q_query,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "pretrain_steps": pretrain_steps, "pretrain_lr": pretrain_lr,
            "test_angles": test_angle_list, "adapt_steps": adapt_steps,
            "adapt_eval_at": adapt_eval_at, "seed": seed,
            "total_maml_grad_steps": total_maml_grads,
            "total_pretrain_grad_steps": total_pretrain_grads,
        },
        "pretrain_0deg_acc": pre_0deg_acc,
        "maml_training_time_s": maml_time,
        "pretrain_training_time_s": pretrain_time,
        "maml_log": maml_log,
        "adaptation_curves": {
            str(a): adaptation_results[a] for a in test_angle_list
        },
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    torch.save(maml_model.state_dict(), os.path.join(save_dir, "maml_model.pt"))
    torch.save(pretrained_model.state_dict(),
               os.path.join(save_dir, "pretrained_model.pt"))

    volume.commit()
    print(f"\nAll results saved to {save_dir}/")
