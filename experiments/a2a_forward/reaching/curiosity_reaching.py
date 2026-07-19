"""Phase 1 of the two-timescale value loop: a learning-progress DRIVE on an active-
vision control task, and whether it is distinguishable from raw surprise and from a
task-free predictability seeker. (ideas/two_timescale_value_loop.md, discriminator #1
"dark room broken" + the non-optional "noisy-TV" control + the LP != surprise falsifier.)

The whole a2a/rhm value side we have built so far is EXTRINSIC (goal-distance, reach-r*)
and STATIONARY. The genuinely-unbuilt atom is an INTRINSIC learning-progress reward
r = -d||e||/dt, driving the POLICY (where to attend), not injected as a residual-stream
scalar. This is the minimal instantiation of that atom.

Why reaching / active vision and not RHM for this atom: the RHM specialization line
(experiments/rhm/specialization/README.md) shows the RHM depth frontier is NOT moved by
allocation (breadth-restriction inert, level-reweighting harmful; only a DIRECT deep
target moves it) and the root is never NTP-supervised -- so an allocation-driven
curiosity signal has almost no lever there, and an LP~=uniform result would be a
confounded null. Active vision instead has an ACTIONABLE value-relevant variable (what
you glimpse is directly observable and the world model's error at a region genuinely
drops when you attend there). The price is that it is STATIONARY: the drive fires during
learning then fades. That is fine for THIS phase -- dark-room and noisy-TV are one-shot
exploration contrasts visible DURING learning; they do NOT need a moving frontier.
(Compounding / non-stationarity is Phase 2: add drifting dynamics as the single new
variable; the recurrent belief loop + value-shaping of the inner FM also enter there.)

The arena (one canvas, three regions with engineered reducibility signatures):

    |  STRUCT (a FIXED digit)  |  NOISE (fresh U(0,1))  |  BLANK (zeros)  |
       reducible: deterministic    irreducible: fresh        trivial: ~0
       position->pixel map; ||e||   random every canvas;      instantly
       DROPS with learning          ||e|| plateaus HIGH        (dark room)
       (the learnable frontier)     (the noisy TV)             floor

Two load-bearing design lessons from earlier iterations (kept here so we don't repeat them):
  1. Predict CONTENT (pixels), not the belief UPDATE. The belief-update MSE is dominated
     by the deterministic marker/pos term identical across regions, which buries the
     reducible/irreducible split -- LP then fixates noise.
  2. STRUCT must be a FIXED image and there must be NO content-carrying belief. A FRESH
     digit's exact pixels are ~as irreducible as noise; and a belief that stores glimpsed
     content lets the WM "predict" re-glimpsed noise from memory (we saw noise ||e|| fall
     BELOW its iid floor 0.083), manufacturing spurious LP on noise. A content-query world
     model with no belief is leak-proof by construction: noise stays at its true floor.

Apparatus:
  - WM: a world model that predicts the glimpse PIXELS at a queried fovea location u
    (query = a positional embedding + a spatial reveal marker at u -- the active-vision
    "where I am about to look"). Trained ONLINE on the policy's own executed glimpses, so
    it improves on whatever the drive practices -- the meta-RL coupling in miniature.
  - policy: TELEPORT allocation. Each step, score sampled candidate next patches by the
    arm's CURRENT true intrinsic reward and pick greedily (eps-explore), never
    re-glimpsing this episode. No learned value head: this isolates "what does each drive
    PREFER" from "can a head predict the drive", which is the Phase-1 question.

Four arms, identical except the intrinsic reward:
  - lp           : r = relu(slow_EMA(||e||_region) - fast_EMA(||e||_region))  (-d||e||/dt)
  - surprise     : r = +||e||                                                  (noisy-TV trap)
  - minsurprise  : r = -||e||                                                  (dark-room seeker)
  - random       : uniform allocation (baseline)

Predicted headline (region occupancy, mid/late training):
  lp -> STRUCT   surprise -> NOISE   minsurprise -> BLANK   random -> uniform
Both controls (noisy-TV, dark-room) live in the one arena.

Run:
  modal run a2a_forward/reaching/curiosity_reaching.py::curiosity_reaching --quick     # smoke
  modal run --detach a2a_forward/reaching/curiosity_reaching.py::curiosity_reaching --dataset mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def curiosity_reaching(
    dataset: str = "mnist",            # "mnist" or "fashion_mnist"
    patch_size: int = 4,
    n_embd: int = 128,
    n_head: int = 4,
    glimpse_grid: int = 3,             # GxG patch window revealed at the fovea
    episode_len: int = 6,              # K glimpses per episode
    n_candidates: int = 15,            # candidate patches scored per step
    eps0: float = 0.40,                # initial exploration rate
    eps_final: float = 0.05,           # final exploration rate
    eps_decay_frac: float = 0.3,       # fraction of the run over which eps decays
    fm_d_head: int = 32,
    fm_mlp_mult: float = 2.0,
    fwd_lr: float = 1e-4,              # deliberately slow: keeps the reducible STRUCT
    fm_updates: int = 1,              # frontier alive long enough for LP to ride it
    n_iters: int = 1000,               # training iterations (episode-batches)
    batch_size: int = 128,
    ema_fast: float = 0.10,            # fast EMA rate for the LP tracker
    ema_slow: float = 0.02,            # slow EMA rate for the LP tracker
    arms: str = "lp,surprise,minsurprise,random",
    seed: int = 42,
    quick: bool = False,
):
    import os
    import time
    import torch
    import torch.nn as nn
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.forward_model import TransformerForwardModel

    if quick:
        n_iters, batch_size = 200, 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]

    # ---- geometry (28 x 84 = three 28-wide panels) ----
    H, W = 28, 84
    p = patch_size
    grid_rows, grid_cols = H // p, W // p            # 7 x 21
    n_patches = grid_rows * grid_cols                # 147
    n_positions = n_patches + 1
    pix = p * p                                       # pixels per patch
    panel_cols = grid_cols // 3                       # 7 patch-cols per region
    _cols = torch.arange(n_patches) % grid_cols
    region_of_patch = torch.clamp(_cols // panel_cols, max=2).to(device)  # (n_patches,)
    REGION_NAMES = ["struct", "noise", "blank"]
    print(f"CURIOSITY REACHING on {dataset}, {device}. grid={grid_rows}x{grid_cols} "
          f"({n_patches} patches), arms={arm_list}, iters={n_iters}")

    # ---- data ----
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    ds = load_dataset(ds_name)
    tr = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_digits = torch.from_numpy(tr).float().unsqueeze(1) / 255.0  # (N,1,28,28)
    print(f"  loaded {len(train_digits)} digits")

    g_noise = torch.Generator(device=device).manual_seed(seed + 8)
    # STRUCT = ONE fixed digit, held constant across every canvas => deterministic =>
    # genuinely REDUCIBLE. (A FRESH digit's exact pixels are ~as irreducible as noise.)
    _gf = torch.Generator().manual_seed(seed + 3)
    fixed_struct = train_digits[torch.randint(len(train_digits), (1,), generator=_gf)
                                ].to(device)                     # (1,1,28,28)

    def sample_canvas(B):
        """(B,1,28,84): [ FIXED digit | fresh uniform noise | zeros ]."""
        struct = fixed_struct.expand(B, 1, 28, 28)               # reducible (deterministic)
        noise = torch.rand(B, 1, 28, 28, generator=g_noise, device=device)  # irreducible
        blank = torch.zeros(B, 1, 28, 28, device=device)         # trivial
        return torch.cat([struct, noise, blank], dim=3)          # (B,1,28,84)

    def to_patches(x):
        """(B,1,28,84) -> (B, n_patches, pix) raw pixel patches (the WM's target)."""
        B = x.shape[0]
        x = x.reshape(B, 1, grid_rows, p, grid_cols, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_patches, pix)

    # ---- reveal mask (GxG patch window), general (non-square) grid ----
    h = glimpse_grid // 2
    Mmask = torch.zeros(n_patches, n_patches)
    for c in range(n_patches):
        cr, cc = c // grid_cols, c % grid_cols
        for dr in range(-h, h + 1):
            for dc in range(-h, h + 1):
                r, k = cr + dr, cc + dc
                if 0 <= r < grid_rows and 0 <= k < grid_cols:
                    Mmask[c, r * grid_cols + k] = 1.0
    reveal_mask = Mmask.to(device)                               # (n_patches, n_patches)

    def reveal_positions(u_idx):
        """(N, n_positions) 0/1 marker of glimpse-revealed positions for command u_idx
        (cls position 0 is never a glimpse target). The spatial efference copy."""
        m = reveal_mask[u_idx]                                   # (N, n_patches)
        cls0 = torch.zeros(m.shape[0], 1, device=device)
        return torch.cat([cls0, m], dim=1)                        # (N, n_positions)

    # ---------------------------------------------------------------
    # one arm
    # ---------------------------------------------------------------
    def run_arm(arm):
        torch.manual_seed(seed)                                   # matched init across arms
        # world model: predict glimpse pixels at a queried location from (pos query + marker)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        wm_pos = nn.Embedding(n_positions, n_embd).to(device)
        decode = nn.Linear(n_embd, pix).to(device)
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            wm_pos.weight.normal_(std=0.02)
            reveal_vec.normal_(std=0.02)
        wm_params = list(fm.parameters()) + list(wm_pos.parameters()) \
            + list(decode.parameters()) + [reveal_vec]
        opt = torch.optim.AdamW(wm_params, lr=fwd_lr, weight_decay=0.01)
        all_pos = torch.arange(n_positions, device=device)

        def wm_pred_patches(u_idx):
            """Predicted glimpse pixels at each patch for query u (N, n_patches, pix)."""
            base = wm_pos(all_pos).unsqueeze(0).expand(u_idx.shape[0], -1, -1)
            inp = base + reveal_positions(u_idx).unsqueeze(-1) * reveal_vec
            return decode(fm(inp))[:, 1:, :]                     # drop cls position

        # LP tracker: per-region fast/slow EMA of ||e||
        err_fast = torch.zeros(3, device=device)
        err_slow = torch.zeros(3, device=device)
        seen = torch.zeros(3, dtype=torch.bool, device=device)

        occ_series, err_series, lp_series = [], [], []

        for it in range(n_iters):
            eps = eps_final + (eps0 - eps_final) * max(0.0, 1 - it / (eps_decay_frac * n_iters))
            canvas = sample_canvas(batch_size)
            B = canvas.shape[0]
            pixels = to_patches(canvas)                          # (B, n_patches, pix)

            trans_u = []
            err_exec_by_region = [[] for _ in range(3)]
            occ_counts = torch.zeros(3, device=device)
            visited = torch.zeros(B, n_patches, dtype=torch.bool, device=device)

            for t in range(episode_len):
                cand = torch.randint(n_patches, (n_candidates,), device=device)  # (Mc,)
                Mc = n_candidates
                with torch.no_grad():
                    pred_p = wm_pred_patches(cand)               # (Mc, n_patches, pix)
                    mask_c = reveal_mask[cand]                   # (Mc, n_patches)
                    # content-prediction error per (candidate, example) over revealed patches
                    se = ((pred_p.unsqueeze(1) - pixels.unsqueeze(0)) ** 2).mean(-1)  # (Mc,B,nP)
                    num = (se * mask_c.unsqueeze(1)).sum(-1)     # (Mc,B)
                    err = num / (mask_c.sum(-1, keepdim=True) + 1e-8)  # (Mc,B) content ||e||

                if arm == "surprise":
                    R = err
                elif arm == "minsurprise":
                    R = -err
                elif arm == "lp":
                    lp = torch.relu(err_slow - err_fast)          # (3,)
                    R = lp[region_of_patch[cand]].unsqueeze(1).expand(Mc, B)
                else:  # random
                    R = torch.zeros(Mc, B, device=device)

                vis_cand = visited[:, cand].t()                  # (Mc, B) already-seen?
                R = R + 1e-6 * torch.randn_like(R)               # random tie-break
                greedy_sel = R.masked_fill(vis_cand, -1e9).argmax(dim=0)
                rand_sel = torch.randn(Mc, B, device=device).masked_fill(
                    vis_cand, -1e9).argmax(dim=0)
                use_eps = 1.0 if arm == "random" else eps
                rand_mask = torch.rand(B, device=device) < use_eps
                sel = torch.where(rand_mask, rand_sel, greedy_sel)
                u_exec = cand[sel]                               # (B,)
                err_exec = err[sel, torch.arange(B, device=device)]  # (B,)
                visited[torch.arange(B, device=device), u_exec] = True
                trans_u.append(u_exec)

                rg = region_of_patch[u_exec]                     # (B,)
                for r in range(3):
                    m = rg == r
                    occ_counts[r] += m.sum()
                    if m.any():
                        err_exec_by_region[r].append(err_exec[m].detach())

            # ---- train the world model on executed glimpses (predict their pixels) ----
            U = torch.cat(trans_u, 0)                            # (K*B,)
            PIX = pixels.unsqueeze(0).expand(episode_len, B, n_patches, pix
                                             ).reshape(episode_len * B, n_patches, pix)
            mask_U = reveal_mask[U]                              # (K*B, n_patches)
            for _ in range(fm_updates):
                pred_p = wm_pred_patches(U)                      # (K*B, n_patches, pix)
                se = ((pred_p - PIX) ** 2).mean(-1)              # (K*B, n_patches)
                loss = (se * mask_U).sum() / (mask_U.sum() + 1e-8)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(wm_params, 1.0)
                opt.step()

            # ---- update LP tracker (per region, from executed scoring errors) ----
            for r in range(3):
                if err_exec_by_region[r]:
                    mean_r = torch.cat(err_exec_by_region[r]).mean()
                    if not seen[r]:
                        err_fast[r] = mean_r; err_slow[r] = mean_r; seen[r] = True
                    else:
                        err_fast[r] = (1 - ema_fast) * err_fast[r] + ema_fast * mean_r
                        err_slow[r] = (1 - ema_slow) * err_slow[r] + ema_slow * mean_r

            occ = (occ_counts / occ_counts.sum()).detach().cpu().numpy()
            occ_series.append(occ)
            err_series.append(err_fast.detach().cpu().numpy().copy())
            lp_series.append(torch.relu(err_slow - err_fast).detach().cpu().numpy().copy())

            if it % max(1, n_iters // 12) == 0 or it == n_iters - 1:
                print(f"  [{arm:11s}] it {it:4d} eps={eps:.2f} occ(s/n/b)="
                      f"{occ[0]:.2f}/{occ[1]:.2f}/{occ[2]:.2f}  "
                      f"||e||={err_fast[0]:.4f}/{err_fast[1]:.4f}/{err_fast[2]:.4f}")

        occ_arr = np.stack(occ_series); err_arr = np.stack(err_series); lp_arr = np.stack(lp_series)
        thirds = {}
        for name, (a, b) in {"early": (0, n_iters // 3),
                             "mid": (n_iters // 3, 2 * n_iters // 3),
                             "late": (2 * n_iters // 3, n_iters)}.items():
            thirds[name] = occ_arr[a:b].mean(0).tolist()
        return {
            "occ_series": occ_arr.tolist(), "err_series": err_arr.tolist(),
            "lp_series": lp_arr.tolist(), "occ_thirds": thirds,
            "final_err": err_arr[-1].tolist(),
        }

    # ---------------------------------------------------------------
    results = {}
    t0 = time.time()
    for arm in arm_list:
        print(f"\n--- arm: {arm} ---")
        results[arm] = run_arm(arm)
    print(f"\nall arms done in {time.time() - t0:.0f}s")

    # ---- summary tables (the headline) ----
    def occ_table(window):
        print(f"\n  REGION OCCUPANCY ({window}-training third)   struct / noise / blank")
        for arm in arm_list:
            s, n, b = results[arm]["occ_thirds"][window]
            print(f"    {arm:12s}  {s:.3f} / {n:.3f} / {b:.3f}")
    print(f"\n{'=' * 62}")
    occ_table("mid"); occ_table("late")
    print("\n  per-region final ||e|| (struct / noise / blank):")
    for arm in arm_list:
        s, n, b = results[arm]["final_err"]
        print(f"    {arm:12s}  {s:.4f} / {n:.4f} / {b:.4f}")
    print("\n  prediction: lp->struct  surprise->noise  minsurprise->blank  random->~uniform")
    print(f"{'=' * 62}")

    # ---- save ----
    tag = f"{dataset}_g{glimpse_grid}_K{episode_len}_{n_head}H{n_embd}D" + ("_quick" if quick else "")
    save_dir = f"{DATA_DIR}/a2a_forward/curiosity_reaching/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "dataset": dataset,
        "config": {
            "patch_size": patch_size, "n_embd": n_embd, "n_head": n_head,
            "glimpse_grid": glimpse_grid, "episode_len": episode_len,
            "n_candidates": n_candidates, "eps0": eps0, "eps_final": eps_final,
            "fm_d_head": fm_d_head, "fm_updates": fm_updates, "n_iters": n_iters,
            "batch_size": batch_size, "ema_fast": ema_fast, "ema_slow": ema_slow,
            "seed": seed, "quick": quick, "region_names": REGION_NAMES,
        },
        "results": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return {arm: results[arm]["occ_thirds"]["late"] for arm in arm_list}
