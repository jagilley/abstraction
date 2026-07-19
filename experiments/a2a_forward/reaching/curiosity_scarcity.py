"""Phase 2b of the two-timescale value loop: does a curiosity DRIVE earn its keep under
SCARCITY -- a small reducible frontier in a large field of irreducible distractors --
and does the fresh-FM ENSEMBLE (disagreement) succeed where naive learning-progress
fails? (ideas/two_timescale_value_loop.md.)

Phase 2 (curiosity_drift.py): under an equal-thirds arena, LP does NOT beat uniform even
under drift -- with the reducible region = 1/3 of the space, random's 33% rehearsal
already tracks it. No SCARCITY. The doc's "select harder targets among many" assumes a
moving frontier you must FIND among distractors. This script builds that, and adds the
ensemble drive the doc prescribes for exactly this regime.

Arena (scarce): a small NEEDLE of reducible structure (a low-energy, deterministic --
hence learnable -- pattern that MORPHS over time) occupying only ~5-15% of the canvas;
the rest is a large field of fresh U(0,1) NOISE (irreducible) plus a small BLANK strip
(trivial, the dark-room control). `needle_cols` is the scarcity knob.

Drives (all PER-PATCH, no privileged region label -- the drive must DISCOVER where the
reducible structure is):
  - lp          : relu(slow-fast EMA of ||e|| per patch) -- naive learning progress.
  - surprise    : +||e||  (chases the huge noise field).
  - minsurprise : -||e||  (dark room).
  - random      : uniform.
  - disagree    : cross-model prediction VARIANCE over an ENSEMBLE of K world models
                  (bootstrapped) -- the fresh-FM prescription. Models disagree where
                  structure is learnable-but-unlearned (needle, and a freshly-morphed
                  needle); they AGREE on the mean where content is aleatoric (noise), so
                  the noise field is rejected by construction. No derivative -> no
                  detection lag; magnitude-of-disagreement, not magnitude-of-error.

Why naive LP fails here (established first): with ~126 noise patches, sampling jitter
means some noise patch always shows fake-LP, so LP trains almost entirely on noise,
predicts the mean everywhere, and scores WORSE than random on the needle. The ensemble
is the fix.

Headline metric: UNBIASED needle-error (each drive's world model -- ensemble mean for
disagree -- evaluated on the current needle). Prediction under scarcity:
  disagree << random < lp ~ surprise/minsurprise on needle-error, and disagree's win GROWS
  as the needle shrinks (scarcity increases).

Run:
  modal run a2a_forward/reaching/curiosity_scarcity.py::curiosity_scarcity --quick
  modal run --detach a2a_forward/reaching/curiosity_scarcity.py::curiosity_scarcity --dataset mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def curiosity_scarcity(
    dataset: str = "mnist",
    patch_size: int = 4,
    n_embd: int = 128,
    n_head: int = 4,
    glimpse_grid: int = 3,
    episode_len: int = 6,
    n_candidates: int = 25,
    eps0: float = 0.40,
    eps_final: float = 0.08,
    eps_decay_frac: float = 0.2,
    fm_d_head: int = 32,
    fm_mlp_mult: float = 2.0,
    fwd_lr: float = 1e-4,
    fm_updates: int = 1,
    n_iters: int = 1200,
    batch_size: int = 128,
    ema_fast: float = 0.15,
    ema_slow: float = 0.03,
    lp_threshold: float = 0.005,
    ensemble_k: int = 4,               # world models in the disagree ensemble
    bootstrap_frac: float = 0.6,       # per-model bootstrap subset of executed glimpses
    needle_amp: float = 1.0,           # full amplitude: needle distinct from blank (removes predict-0 freebie)
    blank_cols: int = 1,
    drift_mode: str = "morph",
    drift_period: int = 150,
    needle_cols_sweep: str = "4,2,1",
    arms: str = "lp,surprise,minsurprise,random,disagree",
    seed: int = 42,
    quick: bool = False,
):
    import os
    import math
    import time
    import torch
    import torch.nn as nn
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.forward_model import TransformerForwardModel

    if quick:
        n_iters, batch_size, needle_cols_sweep = 300, 64, "2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    needle_sweep = [int(x) for x in needle_cols_sweep.split(",")]

    H, W = 28, 84
    p = patch_size
    grid_rows, grid_cols = H // p, W // p
    n_patches = grid_rows * grid_cols
    n_positions = n_patches + 1
    pix = p * p
    print(f"CURIOSITY SCARCITY on {dataset}, {device}. grid={grid_rows}x{grid_cols}, "
          f"arms={arm_list}, needle_cols_sweep={needle_sweep}, drift={drift_mode}@{drift_period}, "
          f"K={ensemble_k}, iters={n_iters}")

    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    ds = load_dataset(ds_name)
    tr = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_digits = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
    print(f"  loaded {len(train_digits)} digits")

    g_noise = torch.Generator(device=device).manual_seed(seed + 8)

    def to_patches(x):
        B = x.shape[0]
        x = x.reshape(B, 1, grid_rows, p, grid_cols, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_patches, pix)

    h = glimpse_grid // 2
    Mmask = torch.zeros(n_patches, n_patches)
    for c in range(n_patches):
        cr, cc = c // grid_cols, c % grid_cols
        for dr in range(-h, h + 1):
            for dc in range(-h, h + 1):
                r, k = cr + dr, cc + dc
                if 0 <= r < grid_rows and 0 <= k < grid_cols:
                    Mmask[c, r * grid_cols + k] = 1.0
    reveal_mask = Mmask.to(device)

    def reveal_positions(u_idx):
        m = reveal_mask[u_idx]
        cls0 = torch.zeros(m.shape[0], 1, device=device)
        return torch.cat([cls0, m], dim=1)

    def masked_err(pred_p, pixels, mask_rows):
        se = ((pred_p - pixels) ** 2).mean(-1)
        return (se * mask_rows).sum(-1) / (mask_rows.sum(-1) + 1e-8)

    # ---- one world model ----
    def build_wm(seed_off):
        torch.manual_seed(seed + seed_off)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        wm_pos = nn.Embedding(n_positions, n_embd).to(device)
        decode = nn.Linear(n_embd, pix).to(device)
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            wm_pos.weight.normal_(std=0.02); reveal_vec.normal_(std=0.02)
        params = list(fm.parameters()) + list(wm_pos.parameters()) \
            + list(decode.parameters()) + [reveal_vec]
        opt = torch.optim.AdamW(params, lr=fwd_lr, weight_decay=0.01)
        all_pos = torch.arange(n_positions, device=device)

        def pred(u_idx):
            base = wm_pos(all_pos).unsqueeze(0).expand(u_idx.shape[0], -1, -1)
            inp = base + reveal_positions(u_idx).unsqueeze(-1) * reveal_vec
            return decode(fm(inp))[:, 1:, :]
        return {"pred": pred, "params": params, "opt": opt}

    # ---------------------------------------------------------------
    def run(arm, needle_cols):
        needle_w = needle_cols * p
        blank_w = blank_cols * p
        noise_w = W - needle_w - blank_w
        cols = torch.arange(n_patches, device=device) % grid_cols
        region = torch.ones(n_patches, dtype=torch.long, device=device)
        region[cols < needle_cols] = 0
        region[cols >= grid_cols - blank_cols] = 2
        needle_patch_idx = torch.nonzero(region == 0, as_tuple=True)[0]

        K = ensemble_k if arm == "disagree" else 1
        models = [build_wm(300 + 100 * j) for j in range(K)]

        struct_rng = torch.Generator().manual_seed(seed + 3)
        n_anchor = (n_iters // max(drift_period, 1)) + 3 if drift_period > 0 else 1
        aidx = torch.randint(len(train_digits), (n_anchor,), generator=struct_rng)
        c0 = (28 - needle_w) // 2
        strips = (train_digits[aidx][:, :, :, c0:c0 + needle_w] * needle_amp).to(device)

        def needle_at(it):
            if drift_period <= 0:
                return strips[0:1]
            seg = it // drift_period
            if drift_mode == "swap":
                return strips[seg:seg + 1]
            frac = (it % drift_period) / drift_period
            w = 0.5 - 0.5 * math.cos(math.pi * frac)
            nxt = min(seg + 1, n_anchor - 1)
            return (1 - w) * strips[seg:seg + 1] + w * strips[nxt:nxt + 1]

        def make_canvas(B, needle_img, noisy=True):
            needle = needle_img.expand(B, 1, 28, needle_w)
            noise = (torch.rand(B, 1, 28, noise_w, generator=g_noise, device=device)
                     if noisy else torch.zeros(B, 1, 28, noise_w, device=device))
            blank = torch.zeros(B, 1, 28, blank_w, device=device)
            return torch.cat([needle, noise, blank], dim=3)

        @torch.no_grad()
        def eval_needle_err(needle_img):
            npix = to_patches(make_canvas(1, needle_img, noisy=False))[0]
            pr = torch.stack([m["pred"](needle_patch_idx) for m in models], 0).mean(0)  # ensemble mean
            return masked_err(pr, npix.unsqueeze(0), reveal_mask[needle_patch_idx]).mean().item()

        err_fast = torch.zeros(n_patches, device=device)
        err_slow = torch.zeros(n_patches, device=device)
        seen = torch.zeros(n_patches, dtype=torch.bool, device=device)
        occ_series, needleerr_series = [], []

        for it in range(n_iters):
            needle_img = needle_at(it)
            eps = eps_final + (eps0 - eps_final) * max(0.0, 1 - it / (eps_decay_frac * n_iters))
            canvas = make_canvas(batch_size, needle_img)
            B = canvas.shape[0]
            pixels = to_patches(canvas)

            all_u, all_e = [], []
            occ_counts = torch.zeros(3, device=device)
            visited = torch.zeros(B, n_patches, dtype=torch.bool, device=device)
            lp = torch.relu(err_slow - err_fast - lp_threshold)   # per-patch LP (lp arm)

            for t in range(episode_len):
                cand = torch.randint(n_patches, (n_candidates,), device=device)
                Mc = n_candidates
                mask_c = reveal_mask[cand]
                with torch.no_grad():
                    pred0 = models[0]["pred"](cand)              # (Mc, n_patches, pix)
                    se0 = ((pred0.unsqueeze(1) - pixels.unsqueeze(0)) ** 2).mean(-1)
                    err = ((se0 * mask_c.unsqueeze(1)).sum(-1)
                           / (mask_c.sum(-1, keepdim=True) + 1e-8))  # (Mc,B) from model 0

                    if arm == "disagree":
                        P = torch.stack([m["pred"](cand) for m in models], 0)   # (K,Mc,n_patches,pix)
                        var = P.var(dim=0).mean(-1)                # (Mc, n_patches) cross-model variance
                        dis = (var * mask_c).sum(-1) / (mask_c.sum(-1) + 1e-8)   # (Mc,)
                        R = dis.unsqueeze(1).expand(Mc, B)
                    elif arm == "surprise":
                        R = err
                    elif arm == "minsurprise":
                        R = -err
                    elif arm == "lp":
                        R = lp[cand].unsqueeze(1).expand(Mc, B)
                    else:
                        R = torch.zeros(Mc, B, device=device)

                vis_cand = visited[:, cand].t()
                R = R + 1e-6 * torch.randn_like(R)
                greedy_sel = R.masked_fill(vis_cand, -1e9).argmax(dim=0)
                rand_sel = torch.randn(Mc, B, device=device).masked_fill(vis_cand, -1e9).argmax(dim=0)
                use_eps = 1.0 if arm == "random" else eps
                sel = torch.where(torch.rand(B, device=device) < use_eps, rand_sel, greedy_sel)
                u_exec = cand[sel]
                err_exec = err[sel, torch.arange(B, device=device)]
                visited[torch.arange(B, device=device), u_exec] = True
                all_u.append(u_exec); all_e.append(err_exec.detach())
                r = region[u_exec]
                for rr in range(3):
                    occ_counts[rr] += (r == rr).sum()

            # train each world model (bootstrap subset for the ensemble)
            U = torch.cat(all_u, 0)
            PIX = pixels.unsqueeze(0).expand(episode_len, B, n_patches, pix
                                             ).reshape(episode_len * B, n_patches, pix)
            NT = U.shape[0]
            for j, m in enumerate(models):
                if K > 1:
                    g = torch.Generator(device=device).manual_seed(seed + 7000 + 13 * j + it)
                    sub = torch.randperm(NT, generator=g, device=device)[:int(bootstrap_frac * NT)]
                else:
                    sub = torch.arange(NT, device=device)
                Us, PIXs, mUs = U[sub], PIX[sub], reveal_mask[U[sub]]
                for _ in range(fm_updates):
                    pr = m["pred"](Us)
                    se = ((pr - PIXs) ** 2).mean(-1)
                    loss = (se * mUs).sum() / (mUs.sum() + 1e-8)
                    m["opt"].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m["params"], 1.0)
                    m["opt"].step()

            # per-patch EMA (model 0; used by lp)
            idx = torch.cat(all_u, 0); val = torch.cat(all_e, 0)
            onesv = torch.ones_like(val)
            sum_e = torch.zeros(n_patches, device=device).scatter_add_(0, idx, val)
            cnt = torch.zeros(n_patches, device=device).scatter_add_(0, idx, onesv)
            hit = cnt > 0
            mean_e = torch.where(hit, sum_e / cnt.clamp(min=1), err_fast)
            fresh = hit & (~seen)
            err_fast = torch.where(fresh, mean_e, err_fast)
            err_slow = torch.where(fresh, mean_e, err_slow)
            upd = hit & seen
            err_fast = torch.where(upd, (1 - ema_fast) * err_fast + ema_fast * mean_e, err_fast)
            err_slow = torch.where(upd, (1 - ema_slow) * err_slow + ema_slow * mean_e, err_slow)
            seen = seen | hit

            occ_series.append((occ_counts / occ_counts.sum()).detach().cpu().numpy())
            needleerr_series.append(eval_needle_err(needle_img))

        occ_arr = np.stack(occ_series); ne = np.array(needleerr_series)
        warm = n_iters // 4
        return {
            "occ_series": occ_arr.tolist(), "needleerr_series": ne.tolist(),
            "needleerr_mean": float(ne[warm:].mean()),
            "occ_needle_mean": float(occ_arr[warm:, 0].mean()),
            "needle_frac": float(len(needle_patch_idx) / n_patches),
        }

    # ---------------------------------------------------------------
    results = {}
    t0 = time.time()
    for nc in needle_sweep:
        results[str(nc)] = {}
        for arm in arm_list:
            print(f"\n--- needle_cols={nc}  arm={arm} ---")
            results[str(nc)][arm] = run(arm, nc)
            r = results[str(nc)][arm]
            print(f"    needle-err(mean)={r['needleerr_mean']:.4f}  needle-occ={r['occ_needle_mean']:.3f} "
                  f"(needle_frac={r['needle_frac']:.3f})")
    print(f"\nall runs done in {time.time() - t0:.0f}s")

    print(f"\n{'=' * 76}\n  UNBIASED needle-error (post-warmup mean) -- lower = better tracking")
    print(f"  needle_frac = reducible fraction of the canvas (scarcity)\n{'=' * 76}")
    print("  needle_cols (frac) |  " + "  ".join(f"{a:>11s}" for a in arm_list))
    for nc in needle_sweep:
        frac = results[str(nc)][arm_list[0]]["needle_frac"]
        row = "  ".join(f"{results[str(nc)][a]['needleerr_mean']:11.4f}" for a in arm_list)
        print(f"  {nc:>2d} ({frac:.2f})           |  {row}")
    print("\n  prediction: disagree << random < lp ~ surprise/minsurprise; disagree's win grows as needle shrinks.")
    print(f"{'=' * 76}")

    tag = f"{dataset}_{drift_mode}{drift_period}_a{needle_amp}_bl{blank_cols}_K{ensemble_k}_g{glimpse_grid}_{n_head}H{n_embd}D" + ("_quick" if quick else "")
    save_dir = f"{DATA_DIR}/a2a_forward/curiosity_scarcity/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "dataset": dataset,
        "config": {
            "patch_size": patch_size, "n_embd": n_embd, "n_head": n_head,
            "glimpse_grid": glimpse_grid, "episode_len": episode_len,
            "n_candidates": n_candidates, "eps0": eps0, "eps_final": eps_final,
            "eps_decay_frac": eps_decay_frac, "fm_d_head": fm_d_head,
            "fm_updates": fm_updates, "n_iters": n_iters, "batch_size": batch_size,
            "ema_fast": ema_fast, "ema_slow": ema_slow, "lp_threshold": lp_threshold,
            "ensemble_k": ensemble_k, "bootstrap_frac": bootstrap_frac,
            "needle_amp": needle_amp, "blank_cols": blank_cols, "drift_mode": drift_mode,
            "drift_period": drift_period, "needle_cols_sweep": needle_sweep,
            "seed": seed, "quick": quick,
        },
        "results": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return {nc: {a: results[str(nc)][a]["needleerr_mean"] for a in arm_list} for nc in needle_sweep}
