"""Parts A+B: does the residual-rank decomposition in beliefs/dimensionality_expansion.md hold?

The belief claims a layer's activations partition into what the forward self-model
has compressed and what it has not:

    A = FM(earlier) + Residual,   "Subadditively, R_act ~= R_comp + R_res"
    "R_res is the slice of R_act the self-model hasn't absorbed yet"

`R_res` has been measured all over this repo. `R_act` and `R_comp` never have --
grep for `R_act` returns only the belief doc and docs citing it. This script
measures all three on real RHM activations, plus the geometry that decides
whether "slice of" means anything.

Design (architecture-matched, following RESIDUAL_RANK_README Exp. 3 so the
numbers are directly comparable to its published table):

  main model  4L/4H/128D GPT-2, trained then FROZEN (seed 42)
  FM          1L/4H, head count matched to the block; capacity 25/50/75/100%
              of one block via d_head 8->32 and mlp_mult 1->4
  settings    L4/m2 (FM can nearly perfectly predict: Exp. 3 hit 0.3% relative
              residual) and L6/m4 (the RHM default, a genuinely rich regime)
  gaps        post_block0 -> post_block1 (1 block, near-zero residual) and
              post_block0 -> post_block3 (3 blocks, irreducible residual)

The gap factor is the important addition. Exp. 3 only ran the 1-block gap and
concluded rank was "invariant to FM capacity". The 3-block gap is where the FM
genuinely cannot replicate the computation, so it is where the belief's
partition has its best chance of being real.

Reproduction:
    cd experiments/
    modal run --detach -m rhm.residual_decomposition.rhm_decomposition_audit::decomposition_audit
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

# Reuse rhm_residual_rank's app + image so `train_main_model.spawn()` is hydrated
# when this orchestrator runs (one Modal app, functions registered from both
# modules). The main-model checkpoints are then bit-identical to Exp. 3's.
from rhm.rhm_residual_rank import app, train_main_model

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

FM_CONFIGS = [
    ("25%", dict(fwd_d_head=8, fwd_mlp_mult=1.0)),
    ("50%", dict(fwd_d_head=16, fwd_mlp_mult=2.0)),
    ("75%", dict(fwd_d_head=24, fwd_mlp_mult=3.0)),
    ("100%", dict(fwd_d_head=32, fwd_mlp_mult=4.0)),
]

# (depth, m): L4/m2 = the near-perfect-FM regime, L6/m4 = the RHM default.
SETTINGS = [(4, 2), (6, 4)]

GAPS = [("post_block0", "post_block1"), ("post_block0", "post_block3")]


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=7200,
    memory=16384,
)
def train_fm_and_decompose(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    main_model_dir: str = "",
    fwd_d_head: int = 32, fwd_mlp_mult: float = 4.0,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    batch_size: int = 64, fwd_lr: float = 1e-3,
    n_steps: int = 0, seed: int = 137,
    eval_interval: int = 500, n_analysis_batches: int = 40,
    capacity_label: str = "",
):
    """Train an architecture-matched FM on frozen activations, then decompose.

    Deliberately NOT reusing rhm_residual_rank.train_matched_fm: its save path
    (`archmatched_1L_{H}H_{d}d_mlp{mult}`) has no gap field, so running two
    prediction gaps through it would silently overwrite Exp. 3's results. This
    trains identically but writes into this sub-experiment's own namespace.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    block_size = s ** L
    device = "cuda"

    volume.reload()
    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    steps_per_epoch = max(1, len(train_data) // (batch_size * block_size))
    if n_steps == 0:
        n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(os.path.join(main_model_dir, "model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())

    fm = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=n_head,
        n_layer=1, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())

    gap_tag = f"{predict_from}_to_{predict_to}"
    print(f"=== {key} | {capacity_label} FM ({fm_params:,}p, "
          f"{fm_params / block_params:.1%} of block) | {gap_tag} ===")

    opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=fwd_lr * 0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
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
            print(f"  step {step:6d}: mse={float(loss):.6f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    # --- collect held-out (A, P) pairs and decompose ---
    fm.eval()
    A_list, P_list = [], []
    with torch.no_grad():
        for _ in range(n_analysis_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)
            A_list.append(vi[predict_to].reshape(-1, n_embd).float().cpu().numpy())
            P_list.append(fm(vi[predict_from]).reshape(-1, n_embd).float().cpu().numpy())

    A = np.concatenate(A_list, axis=0)
    P = np.concatenate(P_list, axis=0)
    decomp = full_decomposition(A, P, seed=seed)

    b, n, g, r = decomp["basic"], decomp["naive"], decomp["geometry"], decomp["repaired"]
    print(f"\n  --- decomposition ({key}, {capacity_label}, {gap_tag}) ---")
    print(f"  rel_residual={b['relative_residual']:.4f}  cosine={b['mean_cosine']:.4f}")
    print(f"  NAIVE:    R_act={n['R_act']:.1f}  R_comp={n['R_comp']:.1f}  "
          f"R_res={n['R_res']:.1f}  ->  (R_comp+R_res)/R_act = "
          f"{n['subadditivity_ratio']:.2f}   [belief predicts ~1.00]")
    print(f"  GEOMETRY: alignment_index={g['alignment_index']:+.3f} "
          f"[0 = residual oriented at chance w.r.t. A]  "
          f"absorption_step_excess={g['absorption_step_excess']:+.3f}")
    print(f"  REPAIRED: R_act_pr={r['R_act_pr']:.1f}  R_comp_v={r['R_comp_v']:.1f}  "
          f"R_res_v={r['R_res_v']:.2f}  frontier_mass={r['frontier_mass']:.4f}")

    save_dir = f"{DATA_DIR}/{key}/residual_decomposition/{gap_tag}/cap{capacity_label.replace('%', 'pct')}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(fm.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "capacity_label": capacity_label,
        "fm": {"n_params": fm_params, "d_head": fwd_d_head,
               "mlp_mult": fwd_mlp_mult, "n_head": n_head,
               "block_params": block_params,
               "block_fraction": fm_params / block_params},
        "gap": {"predict_from": predict_from, "predict_to": predict_to,
                "tag": gap_tag},
        "training": {"n_steps": n_steps, "fwd_lr": fwd_lr, "seed": seed},
        "decomposition": decomp,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=8192,
)
def smoke(v: int = 8, s: int = 2, depth: int = 4, m: int = 2):
    """Fast end-to-end path check: tiny corpus, tiny step counts, both gaps.

    modal run -m rhm.residual_decomposition.rhm_decomposition_audit::smoke
    """
    n_tokens = 200_000
    main = train_main_model.remote(
        v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
        n_layer=4, n_head=4, n_embd=128, n_steps=300, eval_interval=150)
    print(f"  smoke main: val={main['final_val_loss']:.4f}")

    hs = [train_fm_and_decompose.spawn(
        v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
        main_model_dir=main["save_dir"], predict_from=pf, predict_to=pt,
        n_steps=300, eval_interval=150, n_analysis_batches=10,
        capacity_label="100%", fwd_d_head=32, fwd_mlp_mult=4.0)
        for pf, pt in GAPS]
    out = [h.get() for h in hs]
    for r in out:
        d = r["decomposition"]
        print(f"  {r['gap']['tag']}: ratio={d['naive']['subadditivity_ratio']:.2f} "
              f"align={d['geometry']['alignment_index']:+.3f} "
              f"R_res_v={d['repaired']['R_res_v']:.3f}")
    print("SMOKE OK")
    return out


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=21600,
    memory=8192,
)
def decomposition_audit(
    v: int = 8, s: int = 2, n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
):
    """Full Parts A+B sweep: 2 settings x 4 capacities x 2 gaps."""
    print(f"=== Residual decomposition audit ===")
    print(f"  settings={SETTINGS}  capacities={[c for c, _ in FM_CONFIGS]}  "
          f"gaps={[f'{a}->{b}' for a, b in GAPS]}")

    print("\n--- Phase 1: main models (frozen after training) ---")
    main_handles = {
        (d, m): train_main_model.spawn(
            v=v, s=s, depth=d, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        for d, m in SETTINGS
    }
    mains = {}
    for kd, h in main_handles.items():
        mains[kd] = h.get()
        print(f"  L={kd[0]},m={kd[1]}: val={mains[kd]['final_val_loss']:.4f}, "
              f"block={mains[kd]['block_params']:,}p")

    print("\n--- Phase 2: FMs x capacities x gaps ---")
    handles = []
    for (d, m) in SETTINGS:
        for label, cfg in FM_CONFIGS:
            for pf, pt in GAPS:
                handles.append(train_fm_and_decompose.spawn(
                    v=v, s=s, depth=d, m=m, n_tokens=n_tokens,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                    main_model_dir=mains[(d, m)]["save_dir"],
                    predict_from=pf, predict_to=pt,
                    capacity_label=label, **cfg))

    results = [h.get() for h in handles]

    print("\n" + "=" * 118)
    print("PART A - the triple exactly as beliefs/dimensionality_expansion.md defines it")
    print("=" * 118)
    print(f"{'setting':>12} {'gap':>10} {'cap':>5} {'rel_res':>8} {'cos':>7} | "
          f"{'R_act':>7} {'R_comp':>7} {'R_res':>7} | {'ratio':>6} {'ratio_E':>8}")
    print("-" * 118)
    for r in sorted(results, key=lambda r: (r["setting"], r["gap"]["tag"],
                                            r["fm"]["n_params"])):
        d = r["decomposition"]
        b, n = d["basic"], d["naive"]
        gap_short = r["gap"]["tag"].replace("post_block", "b").replace("_to_", "->")
        print(f"{r['setting']:>12} {gap_short:>10} {r['capacity_label']:>5} "
              f"{b['relative_residual']:>8.4f} {b['mean_cosine']:>7.4f} | "
              f"{n['R_act']:>7.1f} {n['R_comp']:>7.1f} {n['R_res']:>7.1f} | "
              f"{n['subadditivity_ratio']:>6.2f} {n['subadditivity_ratio_energy']:>8.2f}")
    print("\n  Belief predicts ratio ~ 1.00 (R_act ~= R_comp + R_res).")

    print("\n" + "=" * 118)
    print("PART B - geometry: is the residual actually a 'slice of R_act'?")
    print("=" * 118)
    print(f"{'setting':>12} {'gap':>10} {'cap':>5} | {'align_idx':>10} "
          f"{'res_in_top16':>13} {'chance':>7} | {'step':>6} {'step_null':>10} "
          f"{'step_xs':>8} | {'rho_neg':>8}")
    print("-" * 118)
    for r in sorted(results, key=lambda r: (r["setting"], r["gap"]["tag"],
                                            r["fm"]["n_params"])):
        g = r["decomposition"]["geometry"]
        c16 = next((c for c in g["containment_curve"] if c["k"] == 16), None)
        gap_short = r["gap"]["tag"].replace("post_block", "b").replace("_to_", "->")
        print(f"{r['setting']:>12} {gap_short:>10} {r['capacity_label']:>5} | "
              f"{g['alignment_index']:>+10.3f} "
              f"{c16['res_var_inside']:>13.3f} {c16['chance']:>7.3f} | "
              f"{g['absorption_step']:>6.2f} {g['absorption_step_null']:>10.2f} "
              f"{g['absorption_step_excess']:>+8.3f} | "
              f"{g['absorption_negative_frac']:>8.3f}")
    print("\n  alignment_index: 0 = residual oriented at chance w.r.t. A's directions,")
    print("  1 = residual perfectly nested inside them. The belief needs this well above 0.")

    print("\n" + "=" * 118)
    print("PART B' - the repaired triple (partitions A's variance along A's own axes)")
    print("=" * 118)
    print(f"{'setting':>12} {'gap':>10} {'cap':>5} | {'R_act_pr':>9} {'R_comp_v':>9} "
          f"{'R_res_v':>9} | {'absorbed':>9} {'frontier':>9} | "
          f"{'R_comp_part':>11} {'R_res_part':>10}")
    print("-" * 118)
    for r in sorted(results, key=lambda r: (r["setting"], r["gap"]["tag"],
                                            r["fm"]["n_params"])):
        rp = r["decomposition"]["repaired"]
        gap_short = r["gap"]["tag"].replace("post_block", "b").replace("_to_", "->")
        print(f"{r['setting']:>12} {gap_short:>10} {r['capacity_label']:>5} | "
              f"{rp['R_act_pr']:>9.2f} {rp['R_comp_v']:>9.2f} {rp['R_res_v']:>9.3f} | "
              f"{rp['absorbed_fraction']:>9.4f} {rp['frontier_mass']:>9.4f} | "
              f"{rp['R_comp_participation']:>10.2f} {rp['R_res_participation']:>9.2f}")

    save_dir = f"{DATA_DIR}/residual_decomposition_audit"
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "settings": SETTINGS, "gaps": GAPS,
        "fm_configs": [(lbl, cfg) for lbl, cfg in FM_CONFIGS],
        "main_models": {f"L{d}_m{m}": mains[(d, m)] for d, m in SETTINGS},
        "results": results,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/summary.json")
    return summary
