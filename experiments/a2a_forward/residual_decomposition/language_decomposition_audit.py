"""Confirmatory test on language: is the shadow law an RHM artifact or general?

The RHM audit (rhm/residual_decomposition/) found that the residual is not a
partition of activation space into "compressed" and "frontier" directions.
Instead, residual variance follows a power law in activation variance across the
main model's principal directions:

    res_var(i)  ~  act_var(i) ** beta

with beta ~ 0.56-0.61 at a 3-block prediction gap (R^2 ~ 0.98), held to +/-0.009
across a 4x FM capacity range, while capacity moved the overall level a lot.
At a 1-block gap beta collapsed to ~0.13-0.17 and the residual sat at chance
alignment -- there the FM is so accurate that the leftover is just float noise.

This ports that measurement to real language, holding the architecture as close
to fixed as possible so the DOMAIN is what changes:

    main model  4L/4H/256D GPT (~28.9M params), the canonical language model
                from LANGUAGE_RATCHET_README.md, on 10M FineWeb-Edu tokens
    FM          1L/4H arch-matched, capacity 25/50/75/100% of one block via
                d_head 16->64 and mlp_mult 1->4 (at 100%, n_head*d_head = 256
                = d_model, an exact architectural copy of one block)
    gaps        post_block0 -> post_block1 and post_block0 -> post_block3

Both the RHM main model (4L/4H/128D) and this one are 4-layer GPTs with the same
block structure, and the gaps are literally the same strings, so a difference in
beta is attributable to the data-generating process rather than to depth,
head count, or where the gap sits in the network.

Predictions, if the shadow law is general:
  * 3-block gap  -> beta ~ 0.6, alignment_index well above chance
  * 1-block gap  -> beta ~ 0.1-0.2, alignment_index ~ chance, naive R_res near
                    full rank (this is the regime that produced the "language
                    residual is ~200/256, i.e. full rank" reading the belief doc
                    cites -- under this account that number is a noise floor)
  * beta invariant across the capacity sweep; only the level moves

The metric library is imported from the RHM sub-experiment rather than copied,
so both domains are scored by literally the same code. That is the point of a
confirmatory run, and it is why `rhm` is added to this image's local sources.

Reproduction:
    cd experiments/
    # one-time, if {DATA_DIR}/tokens is not already populated:
    modal run --detach -m language_reduction.experiments.pipeline.stages::tokenize \
        --n-tokens 12000000 --shard-size 4000000
    modal run --detach -m a2a_forward.residual_decomposition.language_decomposition_audit::language_audit
"""

import json
import os

import modal

from a2a_forward.shared import volume, DATA_DIR, NumpyEncoder

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("a2a_forward")
    .add_local_python_source("rhm")
)

app = modal.App("language-residual-decomposition", image=image)

FM_CONFIGS = [
    ("25%", dict(fwd_d_head=16, fwd_mlp_mult=1.0)),
    ("50%", dict(fwd_d_head=32, fwd_mlp_mult=2.0)),
    ("75%", dict(fwd_d_head=48, fwd_mlp_mult=3.0)),
    ("100%", dict(fwd_d_head=64, fwd_mlp_mult=4.0)),
]

GAPS = [("post_block0", "post_block1"), ("post_block0", "post_block3")]


def _load_tokens(n_tokens):
    import glob

    import numpy as np

    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    chunks, total = [], 0
    for p in paths:
        arr = np.load(p)
        chunks.append(arr)
        total += len(arr)
        if total >= n_tokens:
            break
    return np.concatenate(chunks)[:n_tokens], meta["vocab_size"]


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=10800, memory=32768)
def train_language_main(
    n_tokens: int = 10_000_000, block_size: int = 128, batch_size: int = 64,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    lr: float = 3e-4, n_steps: int = 6000, seed: int = 42,
    eval_interval: int = 500, n_eval_batches: int = 10,
):
    """Train the canonical 4L/4H/256D language model, then freeze it."""
    import numpy as np
    import torch

    from a2a_forward.model import GPT

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    volume.reload()
    data_np, vocab_size = _load_tokens(n_tokens)
    data = torch.from_numpy(data_np.astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())
    print(f"=== Language main model: {n_layer}L/{n_head}H/{n_embd}D ===")
    print(f"  {n_params:,} params (block {block_params:,}), vocab {vocab_size}, "
          f"{len(data):,} tokens, {n_steps} steps")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(sd):
        ix = torch.randint(len(sd) - block_size - 1, (batch_size,))
        x = torch.stack([sd[i:i + block_size] for i in ix])
        y = torch.stack([sd[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    best_val, final_val = float("inf"), None
    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(*get_batch(val_data))[1])
                                    for _ in range(n_eval_batches)]))
            best_val = min(best_val, vl)
            final_val = vl
            print(f"  step {step:5d}: train={float(loss):.4f} val={vl:.4f}")

    save_dir = f"{DATA_DIR}/a2a_forward/language_residual_decomposition/main_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    info = {"n_params": n_params, "block_params": block_params,
            "vocab_size": vocab_size, "n_layer": n_layer, "n_head": n_head,
            "n_embd": n_embd, "block_size": block_size, "n_steps": n_steps,
            "seed": seed, "best_val_loss": best_val, "final_val_loss": final_val,
            "save_dir": save_dir}
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved -> {save_dir} (val={final_val:.4f})")
    return info


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=10800, memory=32768)
def train_fm_and_decompose(
    main_model_dir: str = "", n_tokens: int = 10_000_000,
    block_size: int = 128, batch_size: int = 64,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    fwd_d_head: int = 64, fwd_mlp_mult: float = 4.0,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    fwd_lr: float = 1e-3, n_steps: int = 8000, seed: int = 137,
    eval_interval: int = 1000, n_analysis_batches: int = 40,
    max_samples: int = 50000, capacity_label: str = "",
):
    """Train an arch-matched FM on frozen language activations, then decompose."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    volume.reload()
    data_np, vocab_size = _load_tokens(n_tokens)
    data = torch.from_numpy(data_np.astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(os.path.join(main_model_dir, "model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())

    fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=n_head, n_layer=1,
        mlp_mult=fwd_mlp_mult, block_size=block_size).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())

    gap_tag = f"{predict_from}_to_{predict_to}"
    print(f"=== language | {capacity_label} FM ({fm_params:,}p, "
          f"{fm_params / block_params:.1%} of block) | {gap_tag} ===")

    opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=fwd_lr * 0.01)

    def get_batch(sd):
        ix = torch.randint(len(sd) - block_size - 1, (batch_size,))
        x = torch.stack([sd[i:i + block_size] for i in ix])
        y = torch.stack([sd[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    for step in range(n_steps):
        fm.train()
        x, y = get_batch(train_data)
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
        loss = F.mse_loss(fm(inter[predict_from]), inter[predict_to])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: mse={float(loss):.6f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    fm.eval()
    A_l, P_l = [], []
    with torch.no_grad():
        for _ in range(n_analysis_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)
            A_l.append(vi[predict_to].reshape(-1, n_embd).float().cpu().numpy())
            P_l.append(fm(vi[predict_from]).reshape(-1, n_embd).float().cpu().numpy())

    decomp = full_decomposition(np.concatenate(A_l), np.concatenate(P_l),
                                seed=seed, max_samples=max_samples)
    b, n, g, r = (decomp["basic"], decomp["naive"], decomp["geometry"],
                  decomp["repaired"])
    print(f"\n  rel_residual={b['relative_residual']:.4f}  "
          f"cosine={b['mean_cosine']:.4f}")
    print(f"  NAIVE:    R_act={n['R_act']:.1f} R_comp={n['R_comp']:.1f} "
          f"R_res={n['R_res']:.1f}  ratio={n['subadditivity_ratio']:.2f}")
    print(f"  GEOMETRY: alignment_index={g['alignment_index']:+.3f}")
    print(f"  REPAIRED: R_act_H={r['R_act_H']:.1f} R_res_H={r['R_res_H']:.1f} "
          f"frontier_mass={r['frontier_mass']:.4f} R_res_part={r['R_res_participation']:.1f}")

    save_dir = (f"{DATA_DIR}/a2a_forward/language_residual_decomposition/"
                f"{gap_tag}/cap{capacity_label.replace('%', 'pct')}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(fm.state_dict(), os.path.join(save_dir, "fwd_model.pt"))
    result = {
        "setting": "language_fineweb_edu", "capacity_label": capacity_label,
        "fm": {"n_params": fm_params, "d_head": fwd_d_head,
               "mlp_mult": fwd_mlp_mult, "n_head": n_head,
               "block_params": block_params,
               "block_fraction": fm_params / block_params},
        "gap": {"predict_from": predict_from, "predict_to": predict_to,
                "tag": gap_tag},
        "training": {"n_steps": n_steps, "fwd_lr": fwd_lr, "seed": seed,
                     "n_tokens": n_tokens, "block_size": block_size},
        "decomposition": decomp,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=8192)
def language_audit(n_tokens: int = 10_000_000, seed: int = 42):
    """4 capacities x 2 gaps on the language model."""
    import numpy as np

    main = train_language_main.remote(n_tokens=n_tokens, seed=seed)
    print(f"  main: {main['n_params']:,} params, val={main['final_val_loss']:.4f}")

    handles = []
    for label, cfg in FM_CONFIGS:
        for pf, pt in GAPS:
            handles.append(train_fm_and_decompose.spawn(
                main_model_dir=main["save_dir"], n_tokens=n_tokens,
                predict_from=pf, predict_to=pt, capacity_label=label, **cfg))
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

    print("\n" + "=" * 112)
    print("LANGUAGE CONFIRMATORY: 4L/4H/256D GPT on FineWeb-Edu")
    print("=" * 112)
    print(f"{'gap':>10} {'cap':>5} {'rel_res':>8} {'cos':>7} | {'R_act':>6} "
          f"{'R_comp':>7} {'R_res':>6} {'ratio':>6} | {'beta':>6} {'R2':>6} "
          f"{'align':>7} | {'R_res_part':>10} {'frontier':>9}")
    print("-" * 112)
    for r in sorted(results, key=lambda r: (r["gap"]["tag"], r["fm"]["n_params"])):
        d = r["decomposition"]
        b, n, g, rp = d["basic"], d["naive"], d["geometry"], d["repaired"]
        beta, r2 = shadow_beta(g)
        gs = r["gap"]["tag"].replace("post_block", "b").replace("_to_", ">")
        print(f"{gs:>10} {r['capacity_label']:>5} {b['relative_residual']:>8.4f} "
              f"{b['mean_cosine']:>7.4f} | {n['R_act']:>6.1f} {n['R_comp']:>7.1f} "
              f"{n['R_res']:>6.1f} {n['subadditivity_ratio']:>6.2f} | "
              f"{beta:>6.2f} {r2:>6.3f} {g['alignment_index']:>+7.3f} | "
              f"{rp['R_res_participation']:>9.1f} {rp['frontier_mass']:>9.4f}")

    print("\n  beta by gap (mean +/- sd over the 4x FM capacity range):")
    groups = {}
    for r in results:
        groups.setdefault(r["gap"]["tag"], []).append(
            shadow_beta(r["decomposition"]["geometry"])[0])
    for gap, betas in sorted(groups.items()):
        gs = gap.replace("post_block", "b").replace("_to_", ">")
        print(f"    {gs:>10}: beta = {np.mean(betas):.3f} +/- {np.std(betas):.3f}")
    print("\n  RHM reference: b0>b1 beta ~ 0.13-0.17 | b0>b3 beta ~ 0.56-0.61")

    save_dir = f"{DATA_DIR}/a2a_forward/language_residual_decomposition"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump({"main_model": main, "results": results}, f, indent=2,
                  cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/summary.json")
    return {"main_model": main, "results": results}


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=3600, memory=32768)
def smoke():
    """modal run -m a2a_forward.residual_decomposition.language_decomposition_audit::smoke"""
    main = train_language_main.local(n_tokens=1_000_000, n_steps=200,
                                     eval_interval=100)
    for pf, pt in GAPS:
        r = train_fm_and_decompose.local(
            main_model_dir=main["save_dir"], n_tokens=1_000_000,
            predict_from=pf, predict_to=pt, n_steps=200, eval_interval=100,
            n_analysis_batches=10, capacity_label="100%")
        d = r["decomposition"]
        print(f"  {pt}: ratio={d['naive']['subadditivity_ratio']:.2f} "
              f"align={d['geometry']['alignment_index']:+.3f}")
    print("SMOKE OK")
