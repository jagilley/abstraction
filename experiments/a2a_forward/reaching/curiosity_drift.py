"""Phase 2 of the two-timescale value loop: does a learning-progress DRIVE keep the
frontier open under NON-STATIONARY (drifting) dynamics, where a stationary task
plateaus? (ideas/two_timescale_value_loop.md -- the load-bearing claim: "a two-
timescale value loop over a fixed task provably manufactures nothing; the reward must
track a MOVING frontier"; falsification "the value loop compounds on a stationary task".)

Phase 1 (curiosity_reaching.py) established, on a STATIONARY active-vision arena, that a
learning-progress drive (LP = -d||e||/dt) rides the reducible frontier and then EXHAUSTS
it -- surprise pins to irreducible noise (noisy TV), min-surprise pins to blank (dark
room). The exhaustion is the stationary plateau: once the fixed struct image is mastered,
LP has no signal.

Phase 2 changes exactly ONE variable: the reducible structure DRIFTS. Every
`drift_period` iterations the STRUCT region's fixed image is RESAMPLED -- the reducible
frontier "re-opens" (the active-vision analog of the doc's "introduce fresh rules -> does
the ratchet re-open?"). `drift_period = 0` recovers stationary Phase 1. Everything else --
the slow world model, the arena, the four drives, the no-re-glimpse rule -- is identical.

Fair metric (the headline): an UNBIASED current-struct-error -- each drive's world model is
evaluated on the CURRENT struct every iteration, independent of how much that drive chose
to sample it. This measures "how well-adapted is your model to the moving frontier", which
is what a drive that TRACKS the frontier should win on. (The per-region EMA the drive
itself sees is biased -- a drive that never samples struct has a stale struct estimate.)

Predictions:
  - LP RE-ENGAGES struct at each swap (a sawtooth in occupancy + struct-error), keeping its
    model adapted; surprise stays ~pinned to noise, min-surprise to blank -- neither tracks
    the moving reducible frontier. (LP re-discovery of a re-opened frontier is exploration-
    gated: right after a swap the error INCREASES, so LP momentarily reads 0 until eps-
    exploration re-samples struct and the world model starts reducing the new error again.)
  - Time-averaged current-struct-error: STATIONARY -> LP ~= random (everyone eventually
    learns the one struct = the plateau); DRIFT -> LP < random < surprise/min-surprise, and
    the LP-vs-random gap GROWS as drift speeds up. That dose-response IS "compounds on a
    moving frontier, plateaus on a stationary task".

Run:
  modal run a2a_forward/reaching/curiosity_drift.py::curiosity_drift --quick
  modal run --detach a2a_forward/reaching/curiosity_drift.py::curiosity_drift --dataset mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def curiosity_drift(
    dataset: str = "mnist",
    patch_size: int = 4,
    n_embd: int = 128,
    n_head: int = 4,
    glimpse_grid: int = 3,
    episode_len: int = 6,
    n_candidates: int = 15,
    eps0: float = 0.40,
    eps_final: float = 0.05,
    eps_decay_frac: float = 0.15,      # decay fast; the eps_final floor re-seeds struct after swaps
    fm_d_head: int = 32,
    fm_mlp_mult: float = 2.0,
    fwd_lr: float = 1e-4,              # SAME slow WM as Phase 1 (change only the drift variable)
    fm_updates: int = 1,
    n_iters: int = 1500,
    batch_size: int = 128,
    ema_fast: float = 0.10,
    ema_slow: float = 0.02,
    lp_threshold: float = 0.005,       # reject noise's sampling-jitter fake-LP (only count real progress)
    drift_mode: str = "swap",          # "swap" = abrupt struct resample; "morph" = smooth interpolation
    drift_periods: str = "0,400,150",  # 0 = stationary (Phase 1); smaller = faster-moving frontier
    arms: str = "lp,surprise,minsurprise,random",
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
        n_iters, batch_size, drift_periods = 360, 64, "0,120"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    dps = [int(x) for x in drift_periods.split(",")]

    # ---- geometry ----
    H, W = 28, 84
    p = patch_size
    grid_rows, grid_cols = H // p, W // p
    n_patches = grid_rows * grid_cols
    n_positions = n_patches + 1
    pix = p * p
    panel_cols = grid_cols // 3
    _cols = torch.arange(n_patches) % grid_cols
    region_of_patch = torch.clamp(_cols // panel_cols, max=2).to(device)
    struct_patch_idx = torch.nonzero(region_of_patch == 0, as_tuple=True)[0]  # struct panel patches
    REGION_NAMES = ["struct", "noise", "blank"]
    print(f"CURIOSITY DRIFT on {dataset}, {device}. grid={grid_rows}x{grid_cols}, "
          f"arms={arm_list}, drift_periods={dps}, iters={n_iters}")

    # ---- data ----
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    ds = load_dataset(ds_name)
    tr = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_digits = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
    print(f"  loaded {len(train_digits)} digits")

    g_noise = torch.Generator(device=device).manual_seed(seed + 8)

    def sample_canvas(B, struct_img):
        """(B,1,28,84): [ current struct | fresh noise | zeros ]."""
        struct = struct_img.expand(B, 1, 28, 28)
        noise = torch.rand(B, 1, 28, 28, generator=g_noise, device=device)
        blank = torch.zeros(B, 1, 28, 28, device=device)
        return torch.cat([struct, noise, blank], dim=3)

    def to_patches(x):
        B = x.shape[0]
        x = x.reshape(B, 1, grid_rows, p, grid_cols, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_patches, pix)

    # ---- reveal mask ----
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
        """pred_p (N,n_patches,pix), pixels (...,n_patches,pix), mask_rows (N,n_patches)."""
        se = ((pred_p - pixels) ** 2).mean(-1)                  # (N, n_patches)
        return (se * mask_rows).sum(-1) / (mask_rows.sum(-1) + 1e-8)

    # ---------------------------------------------------------------
    def run(arm, drift_period):
        torch.manual_seed(seed)                                  # matched init across arms/drifts
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        wm_pos = nn.Embedding(n_positions, n_embd).to(device)
        decode = nn.Linear(n_embd, pix).to(device)
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            wm_pos.weight.normal_(std=0.02); reveal_vec.normal_(std=0.02)
        wm_params = list(fm.parameters()) + list(wm_pos.parameters()) \
            + list(decode.parameters()) + [reveal_vec]
        opt = torch.optim.AdamW(wm_params, lr=fwd_lr, weight_decay=0.01)
        all_pos = torch.arange(n_positions, device=device)

        def wm_pred_patches(u_idx):
            base = wm_pos(all_pos).unsqueeze(0).expand(u_idx.shape[0], -1, -1)
            inp = base + reveal_positions(u_idx).unsqueeze(-1) * reveal_vec
            return decode(fm(inp))[:, 1:, :]

        # IDENTICAL struct schedule across arms (reset per run): pre-sample anchor images,
        # then either SWAP between them abruptly or MORPH (smooth cosine interpolation).
        struct_rng = torch.Generator().manual_seed(seed + 3)
        n_anchor = (n_iters // max(drift_period, 1)) + 3 if drift_period > 0 else 1
        aidx = torch.randint(len(train_digits), (n_anchor,), generator=struct_rng)
        anchors = train_digits[aidx].to(device)                  # (n_anchor,1,28,28)

        def struct_at(it):
            if drift_period <= 0:
                return anchors[0:1]
            seg = it // drift_period
            if drift_mode == "swap":
                return anchors[seg:seg + 1]
            frac = (it % drift_period) / drift_period             # morph: smooth 0->1 over the segment
            w = 0.5 - 0.5 * math.cos(math.pi * frac)
            nxt = min(seg + 1, n_anchor - 1)
            return (1 - w) * anchors[seg:seg + 1] + w * anchors[nxt:nxt + 1]

        current = {"struct": struct_at(0)}

        @torch.no_grad()
        def eval_struct_err():
            """Unbiased: WM error on the CURRENT struct over all struct patches."""
            cv = torch.cat([current["struct"], torch.zeros(1, 1, 28, 56, device=device)], dim=3)
            spix = to_patches(cv)[0]                             # (n_patches, pix)
            pred_p = wm_pred_patches(struct_patch_idx)           # (nS, n_patches, pix)
            e = masked_err(pred_p, spix.unsqueeze(0), reveal_mask[struct_patch_idx])
            return e.mean().item()

        err_fast = torch.zeros(3, device=device)
        err_slow = torch.zeros(3, device=device)
        seen = torch.zeros(3, dtype=torch.bool, device=device)
        occ_series, structerr_series = [], []

        for it in range(n_iters):
            current["struct"] = struct_at(it)                    # frontier drifts (swap or morph)
            eps = eps_final + (eps0 - eps_final) * max(0.0, 1 - it / (eps_decay_frac * n_iters))
            canvas = sample_canvas(batch_size, current["struct"])
            B = canvas.shape[0]
            pixels = to_patches(canvas)

            trans_u = []
            occ_counts = torch.zeros(3, device=device)
            err_exec_by_region = [[] for _ in range(3)]
            visited = torch.zeros(B, n_patches, dtype=torch.bool, device=device)

            for t in range(episode_len):
                cand = torch.randint(n_patches, (n_candidates,), device=device)
                Mc = n_candidates
                with torch.no_grad():
                    pred_p = wm_pred_patches(cand)               # (Mc, n_patches, pix)
                    se = ((pred_p.unsqueeze(1) - pixels.unsqueeze(0)) ** 2).mean(-1)
                    mask_c = reveal_mask[cand]
                    err = ((se * mask_c.unsqueeze(1)).sum(-1)
                           / (mask_c.sum(-1, keepdim=True) + 1e-8))  # (Mc,B)

                if arm == "surprise":
                    R = err
                elif arm == "minsurprise":
                    R = -err
                elif arm == "lp":
                    lp = torch.relu(err_slow - err_fast - lp_threshold)
                    R = lp[region_of_patch[cand]].unsqueeze(1).expand(Mc, B)
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
                trans_u.append(u_exec)

                rg = region_of_patch[u_exec]
                for r in range(3):
                    m = rg == r
                    occ_counts[r] += m.sum()
                    if m.any():
                        err_exec_by_region[r].append(err_exec[m].detach())

            # train WM on executed glimpses
            U = torch.cat(trans_u, 0)
            PIX = pixels.unsqueeze(0).expand(episode_len, B, n_patches, pix
                                             ).reshape(episode_len * B, n_patches, pix)
            mask_U = reveal_mask[U]
            for _ in range(fm_updates):
                pred_p = wm_pred_patches(U)
                se = ((pred_p - PIX) ** 2).mean(-1)
                loss = (se * mask_U).sum() / (mask_U.sum() + 1e-8)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(wm_params, 1.0)
                opt.step()

            for r in range(3):
                if err_exec_by_region[r]:
                    mean_r = torch.cat(err_exec_by_region[r]).mean()
                    if not seen[r]:
                        err_fast[r] = mean_r; err_slow[r] = mean_r; seen[r] = True
                    else:
                        err_fast[r] = (1 - ema_fast) * err_fast[r] + ema_fast * mean_r
                        err_slow[r] = (1 - ema_slow) * err_slow[r] + ema_slow * mean_r

            occ_series.append((occ_counts / occ_counts.sum()).detach().cpu().numpy())
            structerr_series.append(eval_struct_err())

        occ_arr = np.stack(occ_series)
        se_arr = np.array(structerr_series)
        warm = n_iters // 5                                      # ignore initial warmup
        return {
            "occ_series": occ_arr.tolist(),
            "structerr_series": se_arr.tolist(),
            "structerr_mean": float(se_arr[warm:].mean()),       # headline: adaptation to frontier
            "occ_struct_mean": float(occ_arr[warm:, 0].mean()),
        }

    # ---------------------------------------------------------------
    results = {}
    t0 = time.time()
    for dp in dps:
        results[str(dp)] = {}
        for arm in arm_list:
            print(f"\n--- drift_period={dp}  arm={arm} ---")
            results[str(dp)][arm] = run(arm, dp)
            r = results[str(dp)][arm]
            print(f"    struct-err(mean, post-warmup)={r['structerr_mean']:.4f}  "
                  f"struct-occ={r['occ_struct_mean']:.3f}")
    print(f"\nall runs done in {time.time() - t0:.0f}s")

    # ---- headline: current-struct-error per (drift_period, arm) ----
    print(f"\n{'=' * 66}\n  UNBIASED current-struct-error (post-warmup mean)  -- lower = better")
    print(f"  (how well-adapted is the world model to the moving reducible frontier)\n{'=' * 66}")
    print("  drift_period |  " + "  ".join(f"{a:>11s}" for a in arm_list))
    for dp in dps:
        row = "  ".join(f"{results[str(dp)][a]['structerr_mean']:11.4f}" for a in arm_list)
        lab = "stationary" if dp == 0 else f"drift@{dp}"
        print(f"  {lab:>12s} |  {row}")
    print("\n  prediction: stationary -> LP ~= random; drift -> LP < random < surprise/minsurprise,")
    print("              and the LP-vs-random gap grows as drift_period shrinks.")
    print(f"{'=' * 66}")

    # ---- save ----
    tag = f"{dataset}_{drift_mode}_g{glimpse_grid}_K{episode_len}_{n_head}H{n_embd}D" + ("_quick" if quick else "")
    save_dir = f"{DATA_DIR}/a2a_forward/curiosity_drift/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "dataset": dataset,
        "config": {
            "patch_size": patch_size, "n_embd": n_embd, "n_head": n_head,
            "glimpse_grid": glimpse_grid, "episode_len": episode_len,
            "n_candidates": n_candidates, "eps0": eps0, "eps_final": eps_final,
            "eps_decay_frac": eps_decay_frac, "fm_d_head": fm_d_head,
            "fm_updates": fm_updates, "n_iters": n_iters, "batch_size": batch_size,
            "ema_fast": ema_fast, "ema_slow": ema_slow, "drift_periods": dps,
            "seed": seed, "quick": quick, "region_names": REGION_NAMES,
        },
        "results": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return {dp: {a: results[str(dp)][a]["structerr_mean"] for a in arm_list} for dp in dps}
