"""Is beta a property of the MODEL's width, or of the DOMAIN?

Three domains gave, at a 3-block gap and 100% FM capacity:

    MNIST ViT   d=128   beta = 0.610 +/- 0.007
    RHM L6_m4   d=128   beta = 0.607 +/- 0.009
    language    d=256   beta = 0.358 +/- 0.003

The two d=128 domains agree to three decimals despite having nothing in common
(natural images vs a synthetic hierarchy), and the only domain at d=256 sits far
below them. That is either a coincidence or beta tracks model width rather than
data. Right now the two are fully confounded: three domains, two widths.

This sweeps width directly, holding everything else fixed, in BOTH domains at
once so the comparison is apples-to-apples:

    n_embd  in {64, 128, 256, 512}        <- the only thing that varies
    depth   4 blocks, n_head 4            <- fixed (d_head = n_embd/4 scales)
    FM      100% of one block, arch-matched, attention mode matched to the
            main model (causal for the RHM GPT, bidirectional for the ViT --
            the mismatch that had to be fixed in mnist_decomposition_audit)
    gap     post_block0 -> post_block3    <- the wide gap, where a law exists

Predictions:
  * beta tracks WIDTH  -> beta falls monotonically with n_embd in both domains,
    and the two domains agree at matched width (so d=256 should land near the
    language value of ~0.36, and d=512 lower still).
  * beta tracks DOMAIN -> beta is flat across widths within a domain, and MNIST
    and RHM need not agree; the d=128 coincidence is just that.

Holding the FM at exactly 100% of a block also tests the other open prediction:
the frontier's LEVEL is set by the FM's capacity RATIO to a block, so holding
that ratio fixed while scaling everything should leave frontier mass roughly
flat. If instead it falls with width, the level depends on absolute capacity.

Caveat this cannot remove: wider models also fit the data better, so width and
fit quality move together. Val loss / accuracy are reported per width so the
confound is visible.

The two domains live on different Modal volumes, both of which normally mount at
/data, so the RHM volume is mounted at /rhm_data here and its paths are built
explicitly rather than through rhm.shared.DATA_DIR.

Reproduction:
    cd experiments/
    modal run --detach -m a2a_forward.residual_decomposition.width_sweep::width_sweep
"""

import json
import os

import modal

from a2a_forward.shared import volume as a2a_volume, DATA_DIR, NumpyEncoder
from rhm.shared import volume as rhm_volume, setting_key

RHM_DIR = "/rhm_data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0", "datasets",
                 "Pillow")
    .add_local_python_source("a2a_forward")
    .add_local_python_source("rhm")
)

app = modal.App("residual-decomposition-width-sweep", image=image)

WIDTHS = [64, 128, 256, 512]
VOLUMES = {DATA_DIR: a2a_volume, RHM_DIR: rhm_volume}

# RHM domain: the L6/m4 setting that produced beta = 0.607 at d=128.
RHM_V, RHM_S, RHM_L, RHM_M = 8, 2, 6, 4


def _beta_fit(geom):
    import numpy as np

    a = np.asarray(geom["act_variance_spectrum"], dtype=float)
    v = np.asarray(geom["res_variance_in_act_basis"], dtype=float)
    msk = (a > 0) & (v > 0)
    beta, c = np.polyfit(np.log(a[msk]), np.log(v[msk]), 1)
    pred = beta * np.log(a[msk]) + c
    r2 = 1 - ((np.log(v[msk]) - pred) ** 2).sum() / (
        (np.log(v[msk]) - np.log(v[msk]).mean()) ** 2).sum()
    return float(beta), float(r2)


@app.function(volumes=VOLUMES, gpu="A10G", timeout=14400, memory=32768)
def rhm_width_point(
    n_embd: int = 128, n_layer: int = 4, n_head: int = 4,
    n_tokens: int = 5_000_000, batch_size: int = 64,
    lr: float = 3e-4, main_steps: int = 6000,
    fwd_lr: float = 1e-3, fwd_steps: int = 8000,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    seed: int = 42, n_analysis_batches: int = 40, max_samples: int = 50000,
):
    """One (domain=RHM, width) point: train main model, then a 100% FM."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"
    key = setting_key(RHM_V, RHM_S, RHM_L, RHM_M)
    block_size = RHM_S ** RHM_L

    rhm_volume.reload()
    arr = np.load(f"{RHM_DIR}/{key}/corpus.npy", mmap_mode="r")
    data = torch.from_numpy(np.array(arr[:n_tokens]).astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    def get_batch(sd):
        ix = torch.randint(len(sd) - block_size - 1, (batch_size,))
        x = torch.stack([sd[i:i + block_size] for i in ix])
        y = torch.stack([sd[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = GPT(RHM_V, block_size, n_layer, n_head, n_embd).to(device)
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())
    print(f"=== RHM width {n_embd} | {n_layer}L/{n_head}H/{n_embd}D "
          f"(block {block_params:,}p) ===")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    final_val = None
    for step in range(main_steps):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == main_steps - 1:
            model.eval()
            with torch.no_grad():
                final_val = float(np.mean([float(model(*get_batch(val_data))[1])
                                           for _ in range(10)]))
            print(f"  [main {step:5d}] train={float(loss):.4f} val={final_val:.4f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # FM = exact architectural copy of one block: n_head matched, d_head =
    # n_embd/n_head, mlp_mult 4. Causal, matching the GPT's causal blocks.
    fm = TransformerForwardModel(
        d_model=n_embd, d_head=n_embd // n_head, n_head=n_head, n_layer=1,
        mlp_mult=4.0, block_size=block_size, causal=True).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())
    print(f"  FM {fm_params:,}p = {fm_params / block_params:.1%} of block")

    o = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=fwd_steps, eta_min=fwd_lr * 0.01)
    for step in range(fwd_steps):
        fm.train()
        x, y = get_batch(train_data)
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
        loss = F.mse_loss(fm(inter[predict_from]), inter[predict_to])
        o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        o.step()
        sch.step()
        if step % 2000 == 0 or step == fwd_steps - 1:
            print(f"  [fm {step:5d}] mse={float(loss):.6f}")

    fm.eval()
    A_l, P_l = [], []
    with torch.no_grad():
        for _ in range(n_analysis_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)
            A_l.append(vi[predict_to].reshape(-1, n_embd).float().cpu().numpy())
            P_l.append(fm(vi[predict_from]).reshape(-1, n_embd).float().cpu().numpy())
    d = full_decomposition(np.concatenate(A_l), np.concatenate(P_l),
                           seed=seed, max_samples=max_samples)
    beta, r2 = _beta_fit(d["geometry"])
    print(f"  -> beta={beta:.3f} (R2={r2:.3f}) "
          f"frontier={d['repaired']['frontier_mass']:.4f} "
          f"R_act={d['naive']['R_act']:.1f} rel_res={d['basic']['relative_residual']:.4f}")

    return {"domain": "rhm", "n_embd": n_embd, "beta": beta, "r2": r2,
            "fit_quality": final_val, "fit_metric": "val_loss",
            "block_params": block_params, "fm_params": fm_params,
            "decomposition": d}


@app.function(volumes=VOLUMES, gpu="A10G", timeout=14400, memory=32768)
def mnist_width_point(
    n_embd: int = 128, n_layer: int = 4, n_head: int = 4, patch_size: int = 4,
    batch_size: int = 128, lr: float = 3e-4, main_steps: int = 6000,
    fwd_lr: float = 1e-3, fwd_steps: int = 8000,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    seed: int = 42, n_analysis_batches: int = 60, max_samples: int = 50000,
):
    """One (domain=MNIST, width) point."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    ds = load_dataset("ylecun/mnist")

    def pack(sp):
        imgs = np.stack([np.array(i) for i in ds[sp]["image"]])
        return (torch.from_numpy(imgs).float().unsqueeze(1) / 255.0,
                torch.tensor(ds[sp]["label"]))

    train_x, train_y = pack("train")
    test_x, test_y = pack("test")

    def batch(x, y, bs):
        ix = torch.randint(len(x), (bs,))
        return x[ix].to(device), y[ix].to(device)

    model = ViT(patch_size=patch_size, n_layer=n_layer, n_head=n_head,
                n_embd=n_embd).to(device)
    block_params = sum(p.numel() for p in model.blocks[0].parameters())
    print(f"=== MNIST width {n_embd} | {n_layer}L/{n_head}H/{n_embd}D "
          f"(block {block_params:,}p) ===")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    final_acc = None
    for step in range(main_steps):
        model.train()
        xb, yb = batch(train_x, train_y, batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == main_steps - 1:
            model.eval()
            with torch.no_grad():
                accs = []
                for _ in range(20):
                    xv, yv = batch(test_x, test_y, 256)
                    accs.append(float((model(xv)[0].argmax(-1) == yv).float().mean()))
                final_acc = float(np.mean(accs))
            print(f"  [main {step:5d}] loss={float(loss):.4f} acc={final_acc:.4f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Bidirectional FM: the ViT block attends without a mask.
    fm = TransformerForwardModel(
        d_model=n_embd, d_head=n_embd // n_head, n_head=n_head, n_layer=1,
        mlp_mult=4.0, block_size=model.n_positions, causal=False).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())
    print(f"  FM {fm_params:,}p = {fm_params / block_params:.1%} of block")

    o = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=fwd_steps, eta_min=fwd_lr * 0.01)
    for step in range(fwd_steps):
        fm.train()
        xb, yb = batch(train_x, train_y, batch_size)
        with torch.no_grad():
            _, _, inter = model(xb, yb, return_intermediates=True)
        loss = F.mse_loss(fm(inter[predict_from]), inter[predict_to])
        o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        o.step()
        sch.step()
        if step % 2000 == 0 or step == fwd_steps - 1:
            print(f"  [fm {step:5d}] mse={float(loss):.6f}")

    fm.eval()
    A_l, P_l = [], []
    with torch.no_grad():
        for _ in range(n_analysis_batches):
            xv, yv = batch(test_x, test_y, batch_size)
            _, _, vi = model(xv, yv, return_intermediates=True)
            A_l.append(vi[predict_to].reshape(-1, n_embd).float().cpu().numpy())
            P_l.append(fm(vi[predict_from]).reshape(-1, n_embd).float().cpu().numpy())
    d = full_decomposition(np.concatenate(A_l), np.concatenate(P_l),
                           seed=seed, max_samples=max_samples)
    beta, r2 = _beta_fit(d["geometry"])
    print(f"  -> beta={beta:.3f} (R2={r2:.3f}) "
          f"frontier={d['repaired']['frontier_mass']:.4f} "
          f"R_act={d['naive']['R_act']:.1f} rel_res={d['basic']['relative_residual']:.4f}")

    return {"domain": "mnist", "n_embd": n_embd, "beta": beta, "r2": r2,
            "fit_quality": final_acc, "fit_metric": "test_acc",
            "block_params": block_params, "fm_params": fm_params,
            "decomposition": d}


@app.function(volumes=VOLUMES, timeout=21600, memory=8192)
def width_sweep(widths: str = "64,128,256,512", seed: int = 42):
    ws = [int(x) for x in widths.split(",")]
    handles = ([("rhm", w, rhm_width_point.spawn(n_embd=w, seed=seed)) for w in ws]
               + [("mnist", w, mnist_width_point.spawn(n_embd=w, seed=seed))
                  for w in ws])
    results = []
    for _, _, h in handles:
        results.append(h.get())

    print("\n" + "=" * 112)
    print("WIDTH SWEEP -- does beta track model width or domain?")
    print("  (4 blocks, n_head=4, FM = 100% of one block, gap post_block0->post_block3)")
    print("=" * 112)
    print(f"{'domain':>8} {'d_model':>8} {'beta':>7} {'R2':>6} | {'R_act':>7} "
          f"{'R_act/d':>8} {'naive_R_res':>12} | {'frontier':>9} {'R_res_part':>10} "
          f"{'rel_res':>8} | {'align':>7} | {'fit':>8}")
    print("-" * 112)
    for r in sorted(results, key=lambda r: (r["domain"], r["n_embd"])):
        d = r["decomposition"]
        n, g, rp, b = d["naive"], d["geometry"], d["repaired"], d["basic"]
        print(f"{r['domain']:>8} {r['n_embd']:>8} {r['beta']:>7.3f} {r['r2']:>6.3f} | "
              f"{n['R_act']:>7.1f} {n['R_act'] / r['n_embd']:>7.1%} "
              f"{n['R_res']:>12.1f} | {rp['frontier_mass']:>9.4f} "
              f"{rp['R_res_participation']:>9.1f} {b['relative_residual']:>8.4f} | "
              f"{g['alignment_index']:>+7.3f} | {r['fit_quality']:>8.4f}")

    print("\n  beta vs width, per domain:")
    for dom in ("rhm", "mnist"):
        rows = sorted([r for r in results if r["domain"] == dom],
                      key=lambda r: r["n_embd"])
        if rows:
            print(f"    {dom:>6}: " + "  ".join(
                f"d{r['n_embd']}={r['beta']:.3f}" for r in rows))
    print("\n  Reference: language (d=256, 4L/4H) gave beta = 0.358 +/- 0.003")
    print("  WIDTH hypothesis  -> beta falls with d in both, and the two agree at")
    print("                       matched d (d=256 near ~0.36).")
    print("  DOMAIN hypothesis -> beta flat across d within each domain.")

    save_dir = f"{DATA_DIR}/a2a_forward/residual_decomposition_width_sweep"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    a2a_volume.commit()
    print(f"\nSaved to {save_dir}/summary.json")
    return results


@app.function(volumes=VOLUMES, gpu="A10G", timeout=3600, memory=32768)
def smoke():
    """modal run -m a2a_forward.residual_decomposition.width_sweep::smoke"""
    kw = dict(main_steps=200, fwd_steps=200, n_analysis_batches=10)
    r1 = rhm_width_point.local(n_embd=64, **kw)
    r2 = mnist_width_point.local(n_embd=64, **kw)
    print(f"  rhm d64 beta={r1['beta']:.3f} | mnist d64 beta={r2['beta']:.3f}")
    print("SMOKE OK")
