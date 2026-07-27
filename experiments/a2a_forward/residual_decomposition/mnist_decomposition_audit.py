"""Third domain: does the shadow law hold on vision (MNIST ViT)?

Two domains so far, scored by the same decomposition.py:

    RHM      (4L/4H/128D GPT, synthetic hierarchy)  beta = 0.56-0.61 at a
             3-block gap, +/-0.009 over a 4x FM capacity range
    language (4L/4H/256D GPT, FineWeb-Edu)          beta = 0.358 +/- 0.003

The shape/level split replicated in both -- FM capacity moves the amount left
over and leaves the exponent alone -- but the exponent itself did not transfer,
so beta reads as a per-domain measurement rather than a constant. Two points
cannot say whether it tracks anything interpretable. This adds a third.

MNIST is the useful third point specifically because the ViT here is
4L/4H/128D -- the SAME dimensions as the RHM main model, and the same block
structure as both. So RHM vs MNIST is a near-clean domain contrast at matched
architecture (synthetic hierarchy vs natural images), while language adds the
d_model=256 axis. The FM capacity configs are consequently identical to RHM's.

One thing to watch that the other two domains do not have: the ViT's sequence
is 49 patches plus a CLS token, and only CLS receives classification gradient.
If the residual's structure is dominated by that asymmetry rather than by the
computation, it should show up as a difference between the CLS position and the
patch positions -- so the decomposition is run three ways (all positions, patches
only, CLS only) rather than assuming the flattening is harmless.

Reproduction:
    cd experiments/
    modal run --detach -m a2a_forward.residual_decomposition.mnist_decomposition_audit::mnist_audit
"""

import json
import os

import modal

from a2a_forward.shared import volume, DATA_DIR, NumpyEncoder

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pillow is required for `datasets` to decode the MNIST image column.
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0", "datasets",
                 "Pillow")
    .add_local_python_source("a2a_forward")
    .add_local_python_source("rhm")
)

app = modal.App("mnist-residual-decomposition", image=image)

# Identical to the RHM sweep: d_head 8->32, mlp_mult 1->4, head count matched to
# the block. At 100% the FM is an exact architectural copy of one ViT block.
FM_CONFIGS = [
    ("25%", dict(fwd_d_head=8, fwd_mlp_mult=1.0)),
    ("50%", dict(fwd_d_head=16, fwd_mlp_mult=2.0)),
    ("75%", dict(fwd_d_head=24, fwd_mlp_mult=3.0)),
    ("100%", dict(fwd_d_head=32, fwd_mlp_mult=4.0)),
]

GAPS = [("post_block0", "post_block1"), ("post_block0", "post_block3")]


def _load_mnist():
    import numpy as np
    import torch
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist")

    def pack(split):
        imgs = np.stack([np.array(i) for i in ds[split]["image"]])
        x = torch.from_numpy(imgs).float().unsqueeze(1) / 255.0
        return x, torch.tensor(ds[split]["label"])

    return pack("train"), pack("test")


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=10800, memory=32768)
def train_mnist_main(
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128, patch_size: int = 4,
    batch_size: int = 128, lr: float = 3e-4, n_steps: int = 6000, seed: int = 42,
    eval_interval: int = 500,
):
    """Train the 4L/4H/128D ViT, then freeze it."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from a2a_forward.vit import ViT

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    (train_x, train_y), (test_x, test_y) = _load_mnist()
    model = ViT(patch_size=patch_size, n_layer=n_layer, n_head=n_head,
                n_embd=n_embd).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for p in model.blocks[0].parameters())
    print(f"=== MNIST ViT {n_layer}L/{n_head}H/{n_embd}D ===")
    print(f"  {n_params:,} params (block {block_params:,}), {n_steps} steps")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def batch(x, y, bs):
        ix = torch.randint(len(x), (bs,))
        return x[ix].to(device), y[ix].to(device)

    final_acc = None
    for step in range(n_steps):
        model.train()
        xb, yb = batch(train_x, train_y, batch_size)
        logits, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                accs = []
                for _ in range(20):
                    xv, yv = batch(test_x, test_y, 256)
                    accs.append(float((model(xv)[0].argmax(-1) == yv).float().mean()))
            final_acc = float(np.mean(accs))
            print(f"  step {step:5d}: loss={float(loss):.4f} test_acc={final_acc:.4f}")

    save_dir = f"{DATA_DIR}/a2a_forward/mnist_residual_decomposition/main_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    info = {"n_params": n_params, "block_params": block_params,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "patch_size": patch_size, "n_steps": n_steps, "seed": seed,
            "final_test_acc": final_acc, "save_dir": save_dir}
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved -> {save_dir} (acc={final_acc:.4f})")
    return info


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=10800, memory=32768)
def train_fm_and_decompose(
    main_model_dir: str = "", n_layer: int = 4, n_head: int = 4,
    n_embd: int = 128, patch_size: int = 4, batch_size: int = 128,
    fwd_d_head: int = 32, fwd_mlp_mult: float = 4.0,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    fwd_lr: float = 1e-3, n_steps: int = 8000, seed: int = 137,
    eval_interval: int = 1000, n_analysis_batches: int = 60,
    max_samples: int = 50000, capacity_label: str = "",
    fwd_causal: bool = False,
):
    """Train an arch-matched FM on frozen ViT activations, then decompose."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    (train_x, train_y), (test_x, test_y) = _load_mnist()

    model = ViT(patch_size=patch_size, n_layer=n_layer, n_head=n_head,
                n_embd=n_embd).to(device)
    model.load_state_dict(torch.load(os.path.join(main_model_dir, "model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    block_params = sum(p.numel() for p in model.blocks[0].parameters())
    n_positions = model.n_positions

    # The ViT block attends bidirectionally (vit.py SelfAttention applies no
    # mask), so a causal FM would be structurally unable to reproduce it no
    # matter its capacity -- the same class of confound as the head-count
    # mismatch that RESIDUAL_RANK_README Exp. 2 had to eliminate. Default the FM
    # to bidirectional so capacity is the only thing limiting it.
    fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=n_head, n_layer=1,
        mlp_mult=fwd_mlp_mult, block_size=n_positions,
        causal=fwd_causal).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())

    gap_tag = f"{predict_from}_to_{predict_to}"
    print(f"=== MNIST | {capacity_label} FM ({fm_params:,}p, "
          f"{fm_params / block_params:.1%} of block) | {gap_tag} ===")

    opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=fwd_lr * 0.01)

    def batch(x, y, bs):
        ix = torch.randint(len(x), (bs,))
        return x[ix].to(device), y[ix].to(device)

    for step in range(n_steps):
        fm.train()
        xb, yb = batch(train_x, train_y, batch_size)
        with torch.no_grad():
            _, _, inter = model(xb, yb, return_intermediates=True)
        loss = F.mse_loss(fm(inter[predict_from]), inter[predict_to])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: mse={float(loss):.6f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    # Collect held-out activations, keeping the position axis so CLS and patch
    # positions can be scored separately.
    fm.eval()
    A_l, P_l = [], []
    with torch.no_grad():
        for _ in range(n_analysis_batches):
            xv, yv = batch(test_x, test_y, batch_size)
            _, _, vi = model(xv, yv, return_intermediates=True)
            A_l.append(vi[predict_to].float().cpu().numpy())
            P_l.append(fm(vi[predict_from]).float().cpu().numpy())
    A = np.concatenate(A_l, axis=0)       # (N, n_positions, d)
    P = np.concatenate(P_l, axis=0)

    # CLS is position 0 (vit.py concatenates cls_token before the patches).
    views = {
        "all_positions": (A.reshape(-1, n_embd), P.reshape(-1, n_embd)),
        "patches_only": (A[:, 1:].reshape(-1, n_embd), P[:, 1:].reshape(-1, n_embd)),
        "cls_only": (A[:, 0], P[:, 0]),
    }
    decomps = {k: full_decomposition(a, p, seed=seed, max_samples=max_samples)
               for k, (a, p) in views.items()}

    for k, d in decomps.items():
        b, n, g, r = d["basic"], d["naive"], d["geometry"], d["repaired"]
        print(f"  [{k}] rel_res={b['relative_residual']:.4f} "
              f"cos={b['mean_cosine']:.4f} | R_act={n['R_act']:.1f} "
              f"R_comp={n['R_comp']:.1f} R_res={n['R_res']:.1f} "
              f"ratio={n['subadditivity_ratio']:.2f} | "
              f"align={g['alignment_index']:+.3f} | "
              f"R_res_part={r['R_res_participation']:.1f} frontier={r['frontier_mass']:.4f}")

    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_residual_decomposition/"
                f"{gap_tag}/cap{capacity_label.replace('%', 'pct')}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(fm.state_dict(), os.path.join(save_dir, "fwd_model.pt"))
    result = {
        "setting": "mnist_vit", "capacity_label": capacity_label,
        "fm": {"n_params": fm_params, "d_head": fwd_d_head,
               "mlp_mult": fwd_mlp_mult, "n_head": n_head,
               "block_params": block_params,
               "block_fraction": fm_params / block_params},
        "gap": {"predict_from": predict_from, "predict_to": predict_to,
                "tag": gap_tag},
        "training": {"n_steps": n_steps, "fwd_lr": fwd_lr, "seed": seed},
        # `decomposition` mirrors the other two domains' schema so the shared
        # analyzer reads all three; the per-view breakdown sits alongside it.
        "decomposition": decomps["all_positions"],
        "decomposition_by_view": decomps,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=8192)
def mnist_audit(seed: int = 42):
    """4 capacities x 2 gaps on the MNIST ViT."""
    import numpy as np

    main = train_mnist_main.remote(seed=seed)
    print(f"  main: {main['n_params']:,} params, acc={main['final_test_acc']:.4f}")

    handles = []
    for label, cfg in FM_CONFIGS:
        for pf, pt in GAPS:
            handles.append(train_fm_and_decompose.spawn(
                main_model_dir=main["save_dir"], predict_from=pf, predict_to=pt,
                capacity_label=label, **cfg))
    results = [h.get() for h in handles]

    def shadow_beta(geom):
        a = np.asarray(geom["act_variance_spectrum"], dtype=float)
        v = np.asarray(geom["res_variance_in_act_basis"], dtype=float)
        msk = (a > 0) & (v > 0)
        beta, c = np.polyfit(np.log(a[msk]), np.log(v[msk]), 1)
        pred = beta * np.log(a[msk]) + c
        r2 = 1 - ((np.log(v[msk]) - pred) ** 2).sum() / (
            (np.log(v[msk]) - np.log(v[msk]).mean()) ** 2).sum()
        return float(beta), float(r2)

    for view in ("all_positions", "patches_only", "cls_only"):
        print("\n" + "=" * 108)
        print(f"MNIST ViT 4L/4H/128D  [{view}]")
        print("=" * 108)
        print(f"{'gap':>10} {'cap':>5} {'rel_res':>8} {'cos':>7} | {'R_act':>6} "
              f"{'R_comp':>7} {'R_res':>6} {'ratio':>6} | {'beta':>6} {'R2':>6} "
              f"{'align':>7} | {'R_res_part':>10} {'frontier':>9}")
        print("-" * 108)
        for r in sorted(results, key=lambda r: (r["gap"]["tag"],
                                                r["fm"]["n_params"])):
            d = r["decomposition_by_view"][view]
            b, n, g, rp = d["basic"], d["naive"], d["geometry"], d["repaired"]
            beta, r2 = shadow_beta(g)
            gs = r["gap"]["tag"].replace("post_block", "b").replace("_to_", ">")
            print(f"{gs:>10} {r['capacity_label']:>5} "
                  f"{b['relative_residual']:>8.4f} {b['mean_cosine']:>7.4f} | "
                  f"{n['R_act']:>6.1f} {n['R_comp']:>7.1f} {n['R_res']:>6.1f} "
                  f"{n['subadditivity_ratio']:>6.2f} | {beta:>6.2f} {r2:>6.3f} "
                  f"{g['alignment_index']:>+7.3f} | {rp['R_res_participation']:>9.1f} "
                  f"{rp['frontier_mass']:>9.4f}")

        groups = {}
        for r in results:
            groups.setdefault(r["gap"]["tag"], []).append(
                shadow_beta(r["decomposition_by_view"][view]["geometry"])[0])
        print(f"\n  beta by gap (mean +/- sd over the 4x FM capacity range):")
        for gap, betas in sorted(groups.items()):
            gs = gap.replace("post_block", "b").replace("_to_", ">")
            print(f"    {gs:>10}: beta = {np.mean(betas):.3f} "
                  f"+/- {np.std(betas):.3f}")

    print("\n  Cross-domain reference (3-block gap, all at matched arch except d):")
    print("    RHM      4L/4H/128D : beta ~ 0.56-0.61")
    print("    language 4L/4H/256D : beta = 0.358 +/- 0.003")

    save_dir = f"{DATA_DIR}/a2a_forward/mnist_residual_decomposition"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump({"main_model": main, "results": results}, f, indent=2,
                  cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/summary.json")
    return {"main_model": main, "results": results}


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=3600, memory=32768)
def smoke():
    """modal run -m a2a_forward.residual_decomposition.mnist_decomposition_audit::smoke"""
    main = train_mnist_main.local(n_steps=200, eval_interval=100)
    for pf, pt in GAPS:
        r = train_fm_and_decompose.local(
            main_model_dir=main["save_dir"], predict_from=pf, predict_to=pt,
            n_steps=200, eval_interval=100, n_analysis_batches=10,
            capacity_label="100%")
        d = r["decomposition"]
        print(f"  {pt}: ratio={d['naive']['subadditivity_ratio']:.2f} "
              f"align={d['geometry']['alignment_index']:+.3f}")
    print("SMOKE OK")
