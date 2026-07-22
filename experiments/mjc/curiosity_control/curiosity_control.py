"""Curiosity-directed collection sustains control competence under CONTINUOUS drift.

Program: `ideas/two_timescale_value_loop.md` — put the intrinsic **learning-progress**
reward in the slow loop of a *control* task and show it **compounds on a moving frontier**
where a stationary loop exhausts. This is the still-open increment #2 the curiosity→control
thread flagged (`a2a_forward/reaching/CURIOSITY_CONTROL_README.md`): disc-4 (value-shaping
caused by an outer loop) is DONE on MuJoCo (Cuts #4d/#4e); the *drive directing
re-adaptation* was a single-shift NEGATIVE twice (`directed_readapt.py` — disagreement
under-visits a confident-prior patch; `meta_active.py` — VoI ≈ max-magnitude for low-dim
ID); **compounding under continuous (Type-2) drift is un-done.** This is that experiment.

    THE CLAIM. On MuJoCo puck-free reaching — the substrate where an MLP FM provably learns
    AND re-learns dynamics (Cut #3), unlike the reaching-ViT cue-copy dead-end — an intrinsic
    curiosity drive that directs WHERE to collect keeps a control agent competent as a
    *scarce, reducible* dynamics frontier DRIFTS. A task-only / random / surprise collector
    lags. On a STATIONARY frontier all drives tie (the plateau — the RHM/active-vision
    one-pass-saturation anchor, now on control). The gap is the order parameter.

THE DRIVE (settled by Step 0, `CURIOSITY_CONTROL_README.md`). The idea doc's `−d‖e‖/dt`
LP *derivative* is drift-fragile (blind to a re-opened frontier). Step 0 down-selected the
**reducible-disagreement MAGNITUDE** (variance across a bootstrapped FM ensemble; Pathak et
al. 2019) — drift-robust, no derivative lag, and it needs NO ground truth to score (a pure
ensemble forward pass), which raw surprise / the LP derivative do not. So `disagree` is the
primary drive; `lp` (derivative) and `surprise` (raw ‖e‖) are foils that reconfirm Step 0
*on control*; `taskonly` (collect on-policy toward goals = "the extrinsic task loss already
IS the value function" falsifier) and `random` are the baselines.

SUBSTRATE / NON-STATIONARITY. Puck-free momentum reaching (Cut #3). A localized, DRIFTING
`field_patch` (a smooth multi-mode force = reducible but capacity-hungry) is the moving
frontier; its Gaussian `sigma` makes it SCARCE (the Phase-2b precondition: a curiosity drive
only beats uniform when the frontier is a needle among distractors). `drift_mode=morph`
sweeps its center along a line each round (gradual — the Phase-2 lesson: derivative drives
degrade gracefully under gradual drift); `drift_mode=none` pins it = the stationary null.

COLLECTION = teleport-region allocation (isolates the drive from navigation — the confound
that muddied `meta_active` #4c, where the info-MPC "spent early transitions travelling").
Each round the drive scores a G×G grid of arena cells, softmax-samples a cell, and collects
a short in-cell rollout (set_state = a valid memoryless query). `taskonly` instead rolls the
current CEM-MPC policy toward random goals (reactive, on-policy). All arms share the FM
architecture / init seeds / update procedure and an ensemble of K FMs; ONLY the collection
differs (the `directed_readapt` control discipline). The ensemble MEAN plans.

READOUT = control competence over the drift sequence (CEM-MPC median goal-dist), plus the
unbiased frontier-tracking FM error (is the FM current WHERE the frontier now is) and the
occupancy heatmap (did the drive find/track the moving patch). Order parameters:
  * COMPOUNDING: (disagree − baseline) control gap, DRIFT vs STATIONARY (opens under drift).
  * NOISY-TV (`--noise`): an aleatoric `noise_patch` (irreducible). surprise fixates it,
    disagree rejects it (ensemble agrees on the mean) — LP ≠ surprise, on control.
  * VALUE-RELEVANCE (`--value-rel off`): patch drifts OUTSIDE the goal corridor → the drive
    still TRACKS it (frontier-err) but control does not benefit (the #4b system-ID ⊥ value
    dissociation): concentrating on a value-IRRELEVANT frontier is true-but-useless.

Run:
    cd experiments/
    modal run mjc/curiosity_control/curiosity_control.py::curiosity_control --quick                       # smoke
    modal run --detach mjc/curiosity_control/curiosity_control.py::curiosity_control --tag drift_v1        # drift (headline)
    modal run --detach mjc/curiosity_control/curiosity_control.py::curiosity_control --tag stat_v1  --drift-mode none   # stationary null
    modal run --detach mjc/curiosity_control/curiosity_control.py::curiosity_control --tag noise_v1 --noise             # noisy-TV
    modal run --detach mjc/curiosity_control/curiosity_control.py::curiosity_control --tag offpath_v1 --value-rel off    # value-relevance null
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

ALL_ARMS = ["reducible", "disagree", "lp", "surprise", "taskonly", "random"]


@app.function(gpu="L4", memory=32768, timeout=10800, volumes={DATA_DIR: volume})
def run_curiosity_control(cfg: dict) -> dict:
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    arms = cfg["arms"]
    print(f"[setup] device={device} drift={cfg['drift_mode']} value_rel={cfg['value_rel']} "
          f"noise={cfg['noise']} arms={arms}", flush=True)

    # ===================================================================== #
    # geometry: reachable square [-inner, inner]^2, a G×G collection grid
    # ===================================================================== #
    ah = cfg["dgp_base"]["arena_half"]
    # grid spans the TASK-RELEVANT workspace (goals + patch sweep + off-path/noise regions),
    # NOT the whole arena — else a pure-disagreement drive chases perpetually-data-sparse far
    # edges (control-irrelevant) instead of the reducible frontier.
    inner = min(ah - cfg["dgp_base"]["pusher_r"] - 0.02, cfg["collect_range"])
    G = cfg["grid_n"]
    edges = np.linspace(-inner, inner, G + 1)
    ccent = 0.5 * (edges[:-1] + edges[1:])                       # per-axis cell centers
    cell_half = float(inner / G)
    cell_centers = np.array([[cx, cy] for cy in ccent for cx in ccent], np.float32)  # (G*G, 2)
    n_cells = G * G

    def pos_to_cell(pos):
        ix = np.clip(np.searchsorted(edges, pos[..., 0]) - 1, 0, G - 1)
        iy = np.clip(np.searchsorted(edges, pos[..., 1]) - 1, 0, G - 1)
        return iy * G + ix

    # ===================================================================== #
    # the drifting reducible frontier + optional aleatoric noisy-TV patch
    # ===================================================================== #
    sweep = cfg["sweep_r"]
    # value-relevant: frontier sweeps the central corridor (y=0), where goal-paths live.
    # value-irrelevant (null): frontier sweeps a peripheral line (y=off_y), goals avoid it.
    fr_y = 0.0 if cfg["value_rel"] == "on" else cfg["off_y"]

    def patch_center(rnd):
        if cfg["drift_mode"] == "none":
            return np.array([0.0, fr_y], np.float32)             # pinned (stationary null)
        t = (rnd / max(cfg["rounds"] - 1, 1)) * cfg["drift_cycles"]
        tri = 2.0 * abs(t - np.floor(t + 0.5))                   # triangle wave in [0,1] (gradual, revisiting)
        return np.array([-sweep + 2 * sweep * tri, fr_y], np.float32)

    base_dgp = dict(cfg["dgp_base"])
    base_dgp["field_patch"] = dict(center=[0.0, fr_y], sigma=cfg["patch_sigma"],
                                   amp=cfg["patch_amp"], phase=cfg["patch_phase"])
    if cfg["noise"]:
        # a FIXED aleatoric patch offset from the frontier's path (its own scarce region)
        base_dgp["noise_patch"] = dict(center=[cfg["noise_x"], cfg["noise_y"]],
                                       sigma=cfg["patch_sigma"], amp=cfg["noise_amp"])
        base_dgp["noise_seed"] = cfg["seed"] + 777
    noise_cell = (pos_to_cell(np.array(base_dgp["noise_patch"]["center"]))
                  if cfg["noise"] else -1)

    env = PusherEnv(base_dgp, with_puck=False)                   # ONE env; we mutate the patch center
    eval_env = PusherEnv(base_dgp, with_puck=False)              # separate env for control eval

    def set_frontier(e, rnd):
        e.dgp["field_patch"]["center"] = patch_center(rnd).tolist()

    # ===================================================================== #
    # forward model f(s,u)->Δs (arity-2 SiLU MLP) + fixed normalization
    # ===================================================================== #
    beta = cfg["rpf_beta"]

    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, 4)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                nn.init.zeros_(m.bias)
        return net.to(device)

    def build_member(seed):
        """RANDOM-PRIOR ensemble member (Osband et al. 2018): a trainable net + a FROZEN
        random prior net. Prediction = net + β·prior. Where data covers a region the net
        compensates the prior so members MATCH the target (agree). Where data is SPARSE the
        net is unconstrained and the distinct frozen priors make members DISAGREE — the fix
        for the confident-prior blindness (`directed_readapt`) a plain warm-started ensemble
        suffers (every member confidently agrees on a STALE prediction from aged-out data,
        so disagreement→0 exactly where the drifted frontier now needs re-learning)."""
        net = _mlp(seed)
        prior = _mlp(seed + 99991)
        for p in prior.parameters():
            p.requires_grad_(False)
        return {"net": net, "prior": prior}

    def mem_forward(m, Xn):                                       # normalized-Δ prediction
        return m["net"](Xn) + beta * m["prior"](Xn) if beta > 0 else m["net"](Xn)

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    def ens_predict(members, S, U, norm):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Xn = (X - norm["mx"]) / norm["sx"]
            preds = torch.stack([mem_forward(m, Xn) * norm["sy"] + norm["my"] for m in members], 0)
        return preds                                             # (K, N, 4) torch on device

    huber = nn.HuberLoss(delta=1.0)

    def update_member(m, opt, S, U, S2, norm, steps, brng):
        """A few warm minibatch steps on a bootstrap resample of the (recency) buffer.
        Trains only the net; the RPF term ties predictions to the target where data lives."""
        n = len(S)
        if n < 8:
            return
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], n); m["net"].train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, n, size=bs), device=device)   # bootstrap
            opt.zero_grad(); huber(mem_forward(m, Xn[idx]), Yn[idx]).backward(); opt.step()
        m["net"].eval()

    # ===================================================================== #
    # collection primitives
    # ===================================================================== #
    def collect_in_cell(cell, k, rng):
        """k tightly-localized in-cell transitions: teleport to random (pos-in-cell, vel),
        random command, ONE step. Covers the cell's (s,u) space at the CURRENT dynamics."""
        c = cell_centers[cell]
        S = np.empty((k, 4), np.float32); U = np.empty((k, 2), np.float32)
        S2 = np.empty((k, 4), np.float32); AC = np.empty(k, bool)
        for i in range(k):
            pos = c + rng.uniform(-cell_half, cell_half, 2)
            vel = rng.normal(0, cfg["v_explore"], 2)
            env.set_state(pos.astype(np.float64), vel.astype(np.float64))
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            s = env.get_state(); s2, info = env.step(u, fs)
            S[i] = s; U[i] = u; S2[i] = s2; AC[i] = info["any_contact"]
        return S, U, S2, AC

    def collect_probe(cells, states, cmds):
        """Ground-truth Δs for a fixed probe set (for surprise/lp only). Measurement-only,
        NOT added to the training buffer. The 'cost of directing' for the ground-truth drives
        (disagree needs none of this)."""
        P = states.shape[1]; true = np.empty((len(cells), P, 4), np.float32)
        for ci, cell in enumerate(cells):
            for p in range(P):
                env.set_state(states[ci, p, :2].astype(np.float64), states[ci, p, 2:].astype(np.float64))
                s2, _ = env.step(cmds[ci, p], fs)
                true[ci, p] = s2 - states[ci, p]
        return true

    # fixed probe set: P states per cell (+ random commands), reused every round
    prng = np.random.default_rng(cfg["seed"] + 5)
    P = cfg["probe_p"]
    probe_states = np.empty((n_cells, P, 4), np.float32)
    probe_cmds = np.empty((n_cells, P, 2), np.float32)
    for ci in range(n_cells):
        c = cell_centers[ci]
        probe_states[ci, :, :2] = c + prng.uniform(-cell_half, cell_half, (P, 2))
        # LOW probe velocity: the field force acts on Δv independent of velocity, so a
        # near-rest probe isolates the (reducible) field signal from the momentum-dominated
        # Δpos = v·dt that all FMs trivially agree on (else disagreement/surprise are drowned).
        probe_states[ci, :, 2:] = prng.normal(0, cfg["probe_v"], (P, 2))
        probe_cmds[ci] = prng.uniform(-1, 1, (P, 2))

    # ===================================================================== #
    # CEM-MPC control eval (ensemble-mean model; value = goal-dist + terminal vel)
    # ===================================================================== #
    Hp, Kc, ne = cfg["plan_Hp"], cfg["k_shoot"], cfg["cem_elite"]
    vel_pen, re_ = cfg["vel_pen"], cfg["replan_every"]

    # fixed control eval set (identical for every arm & round -> controlled). Goals/starts
    # in the CENTRAL corridor; value-rel on => the frontier sweeps THROUGH it.
    erng = np.random.default_rng(cfg["seed"] + 9)
    B = cfg["n_eval"]; gr = cfg["goal_range"]; sr = cfg["start_range"]
    ev_starts = np.concatenate([
        erng.uniform(-sr, sr, (B, 2)),
        erng.normal(0, cfg["v0_std"], (B, 2))], 1).astype(np.float32)
    ev_goals = erng.uniform(-gr, gr, (B, 2)).astype(np.float32)
    rand_dist = (2 * cfg["reach_r"] if cfg["frontier_task"]
                 else float(np.median(np.linalg.norm(ev_starts[:, :2] - ev_goals, axis=1))))

    def mpc_action(fms, norm, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, Hp, 2), np.float32)
        sig = np.full((Bn, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, Hp, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    dpred = torch.stack([mem_forward(m, (x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                                         for m in fms], 0).mean(0)   # ensemble-mean dynamics
                    s = s + dpred
                    cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_control(fms, norm, rnd, seed):
        """Median goal-dist at the CURRENT (drifted) dynamics.

        frontier_task: start & goal STRADDLE the current patch center along a random axis, so
        the shortest path crosses the frontier — control success then depends on modeling the
        patch force (a stale FM plans a straight path, gets deflected, misses). This makes the
        scarce frontier the CRUX of control (undiluting the ~9% payoff), while the drive's
        collection problem stays scarce (the patch is still one small drifting cell)."""
        set_frontier(eval_env, rnd)
        rng = np.random.default_rng(seed)
        if cfg["frontier_task"]:
            pc = patch_center(rnd).astype(np.float32)
            ang = rng.uniform(0, 2 * np.pi, B)
            dvec = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
            spos = pc[None] + cfg["reach_r"] * dvec
            goals = (pc[None] - cfg["reach_r"] * dvec
                     + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2))).astype(np.float32)
            starts = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)
        else:
            starts = ev_starts.copy(); goals = ev_goals
        states = starts.copy(); plan = None
        for step in range(cfg["plan_H"]):
            if step % re_ == 0:
                plan = mpc_action(fms, norm, states, goals, rng)
            acts = plan[:, step % re_, :]
            for b in range(B):
                eval_env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = eval_env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - goals, axis=1)
        return float(np.median(fd))

    def taskonly_collect(fms, norm, rnd, k, rng, seed):
        """Reactive on-policy collection: roll the current CEM-MPC toward random goals from
        random starts at the current dynamics; log the trajectory. The 'extrinsic task loss
        already is the value' baseline."""
        set_frontier(env, rnd)
        nb = max(4, k // cfg["plan_H"] + 1)
        st = np.concatenate([rng.uniform(-sr, sr, (nb, 2)),
                             rng.normal(0, cfg["v0_std"], (nb, 2))], 1).astype(np.float32)
        gl = rng.uniform(-gr, gr, (nb, 2)).astype(np.float32)
        prng2 = np.random.default_rng(seed)
        S, U, S2, AC = [], [], [], []
        states = st.copy(); plan = None
        for step in range(cfg["plan_H"]):
            if step % re_ == 0:
                plan = mpc_action(fms, norm, states, gl, prng2)
            acts = plan[:, step % re_, :]
            for b in range(nb):
                env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s = env.get_state(); s2, info = env.step(acts[b], fs)
                S.append(s); U.append(acts[b].astype(np.float32)); S2.append(s2)
                AC.append(info["any_contact"]); states[b] = s2
        idx = prng2.permutation(len(S))[:k]
        return (np.asarray(S, np.float32)[idx], np.asarray(U, np.float32)[idx],
                np.asarray(S2, np.float32)[idx], np.asarray(AC, bool)[idx])

    # ===================================================================== #
    # normalization from a fixed uniform warmup pool (identical across arms)
    # ===================================================================== #
    wrng = np.random.default_rng(cfg["seed"] + 3)
    Sw = np.empty((cfg["warmup_n"], 4), np.float32); Uw = np.empty((cfg["warmup_n"], 2), np.float32)
    S2w = np.empty((cfg["warmup_n"], 4), np.float32)
    set_frontier(env, 0)
    for i in range(cfg["warmup_n"]):
        pos = wrng.uniform(-inner, inner, 2); vel = wrng.normal(0, cfg["v_explore"], 2)
        env.set_state(pos.astype(np.float64), vel.astype(np.float64))
        u = wrng.uniform(-1, 1, 2).astype(np.float32)
        s = env.get_state(); s2, _ = env.step(u, fs)
        Sw[i] = s; Uw[i] = u; S2w[i] = s2
    norm = make_norm(Sw, Uw, S2w)

    # ===================================================================== #
    # per-cell drive scores
    # ===================================================================== #
    def score_disagree(fms):
        pr = ens_predict(fms, probe_states.reshape(-1, 4), probe_cmds.reshape(-1, 2), norm)
        var = pr.var(0).mean(1)                                  # (n_cells*P,) predictive variance
        return var.reshape(n_cells, P).mean(1).cpu().numpy()

    def score_error(fms, rnd):
        """mean ‖ens_mean_pred − true Δs‖ per cell (surprise) at the current dynamics."""
        set_frontier(env, rnd)
        true = collect_probe(range(n_cells), probe_states, probe_cmds)
        pr = ens_predict(fms, probe_states.reshape(-1, 4), probe_cmds.reshape(-1, 2), norm)
        pm = pr.mean(0).cpu().numpy().reshape(n_cells, P, 4)
        return np.linalg.norm(pm - true, axis=2).mean(1)         # (n_cells,)

    def score_exploit(rnd):
        """VALUE-relevance per cell = density of the control task's start→goal paths through
        it (where the task needs FM accuracy). Peaks around the current frontier (paths
        straddle it) and is ~0 off-corridor (e.g. the noise patch). This is the EXPLOIT / value
        (`p`-tap) signal that GROUNDS the explore drive: added to the intrinsic term it breaks
        the tie between the (value-relevant) reducible frontier and a (value-irrelevant) noise
        distractor that look equally 'learnable' to a pure novelty drive. Computed from goal
        geometry (no FM dependence) so it's a clean, stationary grounding field."""
        rng = np.random.default_rng(cfg["seed"] + 4242 + rnd)
        M, T = 96, 24
        if cfg["frontier_task"]:
            pc = patch_center(rnd).astype(np.float32)
            ang = rng.uniform(0, 2 * np.pi, M)
            dvec = np.stack([np.cos(ang), np.sin(ang)], 1)
            A = pc[None] + cfg["reach_r"] * dvec
            B = pc[None] - cfg["reach_r"] * dvec
        else:
            A = rng.uniform(-cfg["start_range"], cfg["start_range"], (M, 2))
            B = rng.uniform(-cfg["goal_range"], cfg["goal_range"], (M, 2))
        ts = np.linspace(0, 1, T)[None, :, None]
        pts = (A[:, None, :] * (1 - ts) + B[:, None, :] * ts).reshape(-1, 2).astype(np.float32)
        return np.bincount(pos_to_cell(pts), minlength=n_cells).astype(np.float64)

    # ===================================================================== #
    # run one arm
    # ===================================================================== #
    def run_arm(arm):
        arng = np.random.default_rng(cfg["seed"] + 100)          # SAME across arms -> only drive differs
        fms = [build_member(cfg["seed"] + 40 + j) for j in range(cfg["K"])]
        opts = [torch.optim.Adam(m["net"].parameters(), lr=cfg["fm_lr"]) for m in fms]
        # recency FIFO buffer
        bufS = np.zeros((0, 4), np.float32); bufU = np.zeros((0, 2), np.float32)
        bufS2 = np.zeros((0, 4), np.float32); bufAC = np.zeros((0,), bool)

        def add(S, U, S2, AC):
            nonlocal bufS, bufU, bufS2, bufAC
            bufS = np.concatenate([bufS, S])[-cfg["buffer_n"]:]
            bufU = np.concatenate([bufU, U])[-cfg["buffer_n"]:]
            bufS2 = np.concatenate([bufS2, S2])[-cfg["buffer_n"]:]
            bufAC = np.concatenate([bufAC, AC])[-cfg["buffer_n"]:]

        # seed buffer: a small uniform warmup at round-0 dynamics (so the ensemble has a base
        # + per-member bootstrap disagreement is meaningful off-data)
        set_frontier(env, 0)
        add(*collect_in_cell(pos_to_cell(np.array([0.0, 0.0])), cfg["seed_n"], arng))
        for j, (fm, opt) in enumerate(zip(fms, opts)):
            update_member(fm, opt, bufS, bufU, bufS2, norm, cfg["seed_steps"],
                          np.random.default_rng(cfg["seed"] + 200 + j))

        lp_slow = np.full(n_cells, np.nan); lp_fast = np.full(n_cells, np.nan)
        occ = np.zeros(n_cells); occ_hist = []; recs = []; patch_frac_hist = []; noise_frac_hist = []
        lam = cfg["lp_ema"]

        for rnd in range(cfg["rounds"]):
            # --- pick cell(s) by the drive, collect a fixed budget ---
            if arm == "taskonly":
                S, U, S2, AC = taskonly_collect(fms, norm, rnd, cfg["collect_n"], arng,
                                                cfg["seed"] + 5000 + rnd)
                visited = pos_to_cell(S[:, :2]);
                for cidx in visited:
                    occ[cidx] += 1.0 / len(visited)
                chosen = np.bincount(visited, minlength=n_cells)
            else:
                if arm == "random":
                    scores = np.ones(n_cells)
                elif arm == "disagree":
                    scores = score_disagree(fms)
                elif arm == "surprise":
                    scores = score_error(fms, rnd)
                elif arm == "reducible":
                    # "reducible surprise" (the doc's prescription): error localizes WHERE the
                    # FM is wrong (the frontier); disagreement FILTERS OUT aleatoric noise
                    # (high error but ensemble agrees on the mean). Product is high only at
                    # reducible-unlearned structure — rejects noise AND already-mastered/edges.
                    err = score_error(fms, rnd); dis = score_disagree(fms)
                    en = (err - err.min()) / (err.max() - err.min() + 1e-9)
                    dn = (dis - dis.min()) / (dis.max() - dis.min() + 1e-9)
                    scores = en * dn + 1e-6
                elif arm == "lp":
                    err = score_error(fms, rnd)
                    lp_slow = err if np.all(np.isnan(lp_slow)) else lam * lp_slow + (1 - lam) * err
                    lp_fast = err if np.all(np.isnan(lp_fast)) else 0.5 * lp_fast + 0.5 * err
                    scores = np.maximum(lp_slow - lp_fast, 0.0) + 1e-6   # relu(slow-fast) = -d‖e‖/dt
                elif arm.startswith("grounded"):
                    # TWO ADDITIVE DRIVES (not a gate): explore (reducible-surprise, the e-tap)
                    # + exploit (value-relevance, the p-tap), mixed by `balance` b in [0,1]
                    # (b=1 → pure explore = reducible; b=0 → pure exploit). Grounding breaks the
                    # explore drive's tie between the value-relevant frontier and value-irrelevant
                    # noise. "grounded@0.5" sets b inline (so one run sweeps the mix).
                    b = float(arm.split("@")[1]) if "@" in arm else cfg["balance"]
                    err = score_error(fms, rnd); dis = score_disagree(fms)
                    en = (err - err.min()) / (err.max() - err.min() + 1e-9)
                    dn = (dis - dis.min()) / (dis.max() - dis.min() + 1e-9)
                    exn = en * dn                                        # explore = reducible-surprise
                    ap = score_exploit(rnd)
                    apn = (ap - ap.min()) / (ap.max() - ap.min() + 1e-9)  # exploit = value-relevance
                    scores = b * exn + (1 - b) * apn + 1e-6
                # softmax-sample a cell (+ eps-random to seed the frontier / disagreement)
                sc = scores / (scores.sum() + 1e-12)
                logit = np.log(sc + 1e-12) / cfg["temp"]
                p = np.exp(logit - logit.max()); p /= p.sum()
                if arng.random() < cfg["expl_eps"]:
                    cell = int(arng.integers(n_cells))
                else:
                    cell = int(arng.choice(n_cells, p=p))
                occ[cell] += 1.0
                chosen = np.zeros(n_cells); chosen[cell] = 1.0
                set_frontier(env, rnd)
                add(*collect_in_cell(cell, cfg["collect_n"], arng))

            # --- update ensemble on the recency buffer ---
            for j, (fm, opt) in enumerate(zip(fms, opts)):
                update_member(fm, opt, bufS, bufU, bufS2, norm, cfg["update_steps"],
                              np.random.default_rng(cfg["seed"] + 300 + rnd * 17 + j))

            # what fraction of THIS round's collection was in the frontier / noise patch cell
            fc = pos_to_cell(patch_center(rnd)[None])[0]
            patch_frac_hist.append(float(chosen[fc] / (chosen.sum() + 1e-12)))
            noise_frac_hist.append(float(chosen[noise_cell] / (chosen.sum() + 1e-12)) if noise_cell >= 0 else 0.0)

            # --- periodic control eval + frontier-tracking error ---
            if rnd % cfg["eval_every"] == 0 or rnd == cfg["rounds"] - 1:
                cdist = eval_control(fms, norm, rnd, cfg["seed"] + 7000 + rnd)
                fcell = pos_to_cell(patch_center(rnd)[None])[0]
                fr_err = float(score_error(fms, rnd)[fcell])     # unbiased current-frontier error
                recs.append({"round": rnd, "control_dist": cdist, "frontier_err": fr_err})
                occ_hist.append(occ.copy())
                print(f"  [{arm:9s} r{rnd:3d}] ctrl={cdist:.4f} frontier_err={fr_err:.4f} "
                      f"patch_frac(round)={patch_frac_hist[-1]:.2f}", flush=True)

        return {"records": recs, "occupancy": occ.tolist(),
                "occ_hist": [o.tolist() for o in occ_hist],
                "patch_frac_hist": patch_frac_hist, "noise_frac_hist": noise_frac_hist}

    # ===================================================================== #
    # run all arms
    # ===================================================================== #
    results = {}
    for arm in arms:
        print(f"\n[arm] {arm}", flush=True)
        results[arm] = run_arm(arm)

    # summary order parameters (final-third mean control dist; lower=better)
    def final_ctrl(arm):
        r = [x["control_dist"] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    def final_front(arm):
        r = [x["frontier_err"] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    summary = {a: {"final_control": final_ctrl(a), "final_frontier_err": final_front(a),
                   "patch_frac_last": float(np.mean(results[a]["patch_frac_hist"][-10:])),
                   "noise_frac_last": float(np.mean(results[a]["noise_frac_hist"][-10:]))}
               for a in arms}
    print("\n[summary] final control dist (lower=better) | frontier-err | patch-frac | noise-frac", flush=True)
    for a in arms:
        s = summary[a]
        print(f"  {a:9s} ctrl={s['final_control']:.4f}  front={s['final_frontier_err']:.4f}  "
              f"patch={s['patch_frac_last']:.2f}  noise={s['noise_frac_last']:.2f}", flush=True)
    if "disagree" in arms:
        for base in ["taskonly", "random", "surprise"]:
            if base in arms:
                print(f"  GAP disagree−{base}: control {summary[base]['final_control']-summary['disagree']['final_control']:+.4f} "
                      f"(>0 = disagree better)", flush=True)

    out = {"config": cfg, "regime": ("stationary" if cfg["drift_mode"] == "none" else "drift"),
           "value_rel": cfg["value_rel"], "noise": cfg["noise"], "grid_n": G,
           "cell_centers": cell_centers.tolist(), "inner": inner,
           "patch_path": [patch_center(r).tolist() for r in range(cfg["rounds"])],
           "noise_cell": int(noise_cell), "rand_dist": rand_dist,
           "results": results, "summary": summary}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "curiosity_control", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": out, "figures": figures}


def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"reducible": "#2f9e44", "disagree": "#0ca678", "lp": "#3d6fd1", "surprise": "#d1603d",
           "taskonly": "#9c36b5", "random": "#868e96"}
    arms = list(R["results"].keys())
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    reg = R["regime"]; vr = R["value_rel"]

    # ---- fig1: control competence + frontier-tracking error over rounds ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for a in arms:
        rec = R["results"][a]["records"]
        x = [r["round"] for r in rec]
        axes[0].plot(x, [r["control_dist"] for r in rec], "o-", ms=3, color=(COL.get(a) or "#e8590c"), label=a)
        axes[1].plot(x, [r["frontier_err"] for r in rec], "o-", ms=3, color=(COL.get(a) or "#e8590c"), label=a)
    axes[0].axhline(R["rand_dist"], color="#bbb", ls=":", lw=1, label="random floor")
    axes[0].set_xlabel("round"); axes[0].set_ylabel("median goal-dist (lower=better)")
    axes[0].set_title(f"control competence [{reg}, value_rel={vr}]"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("round"); axes[1].set_ylabel("frontier-region FM error (lower=better)")
    axes[1].set_title("tracking the moving frontier"); axes[1].legend(fontsize=7)
    figs["fig1_curves.png"] = _save(fig)

    # ---- fig2: occupancy heatmaps + patch drift path ----
    G = R["grid_n"]; path = np.array(R["patch_path"])
    cc = np.array(R["cell_centers"]); inner = R["inner"]
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, arms):
        occ = np.array(R["results"][a]["occupancy"]).reshape(G, G)
        ax.imshow(occ, origin="lower", extent=[-inner, inner, -inner, inner],
                  cmap="magma", aspect="equal")
        ax.plot(path[:, 0], path[:, 1], "-", color="cyan", lw=1.3, alpha=0.8)
        ax.scatter([path[0, 0]], [path[0, 1]], c="cyan", s=18, marker="o")
        ax.scatter([path[-1, 0]], [path[-1, 1]], c="white", s=18, marker="s")
        if R["noise_cell"] >= 0:
            ncc = cc[R["noise_cell"]]
            ax.scatter([ncc[0]], [ncc[1]], c="red", s=60, marker="x", lw=2)
        ax.set_title(a, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"collection occupancy (cyan=frontier drift path, red×=noise patch) [{reg}]", fontsize=9)
    figs["fig2_occupancy.png"] = _save(fig)

    # ---- fig3: order-parameter bars ----
    S = R["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].bar(range(len(arms)), [S[a]["final_control"] for a in arms],
                color=[(COL.get(a) or "#e8590c") for a in arms])
    axes[0].set_xticks(range(len(arms))); axes[0].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("final control dist (lower=better)"); axes[0].set_title("control")
    axes[1].bar(range(len(arms)), [S[a]["final_frontier_err"] for a in arms],
                color=[(COL.get(a) or "#e8590c") for a in arms])
    axes[1].set_xticks(range(len(arms))); axes[1].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("final frontier err"); axes[1].set_title("frontier tracking")
    axes[2].bar(range(len(arms)), [S[a]["patch_frac_last"] for a in arms],
                color=[(COL.get(a) or "#e8590c") for a in arms], label="patch frac")
    if R["noise_cell"] >= 0:
        axes[2].bar(range(len(arms)), [S[a]["noise_frac_last"] for a in arms],
                    color="red", alpha=0.5, width=0.4, label="noise frac")
        axes[2].legend(fontsize=7)
    axes[2].set_xticks(range(len(arms))); axes[2].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[2].set_ylabel("collection fraction"); axes[2].set_title("where they collect (last 10 rounds)")
    figs["fig3_orderparams.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def curiosity_control(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "",                       # comma list; default = all
    # regime dials
    drift_mode: str = "morph",            # morph = gradual sweep | none = stationary null
    value_rel: str = "on",                # on = frontier in goal corridor | off = peripheral (null)
    noise: bool = False,                  # add an aleatoric noisy-TV patch
    # env / family (Cut #3's momentum-dominated free-flight reaching — a KNOWN-WORKING
    # controller: frame_skip=12 so momentum bites, large arena so reaching stays off walls)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    joint_damping: float = 2.0,
    # frontier patch (reducible multi-mode force; ABSOLUTE coords). amp strong enough that a
    # STALE FM mispredicts Δv meaningfully once the patch drifts onto a region.
    patch_amp: float = 3.5,
    patch_sigma: float = 0.30,
    patch_phase: float = 0.0,
    sweep_r: float = 0.5,                  # frontier sweeps x∈[-0.5,0.5] (the reaching corridor)
    drift_cycles: float = 2.0,             # back-and-forth sweeps over the run (keeps re-opening)
    off_y: float = 1.1,                    # value_rel=off: frontier sweeps the peripheral line y=1.1
    # noisy-TV patch (aleatoric, OFF the goal corridor so it distracts but doesn't wreck all arms)
    noise_amp: float = 6.0,
    noise_x: float = -1.1,
    noise_y: float = 0.0,
    # grid / collection
    grid_n: int = 6,
    collect_range: float = 1.25,           # grid half-extent (task workspace, not whole arena)
    probe_p: int = 10,
    collect_n: int = 240,
    warmup_n: int = 2000,
    seed_n: int = 500,
    v_explore: float = 1.2,
    probe_v: float = 0.15,                 # near-rest scoring probes (isolate the field signal)
    temp: float = 0.4,
    expl_eps: float = 0.1,                 # eps-random cell choice (seed frontier + disagreement)
    balance: float = 0.5,                  # grounded drive: explore weight (1=pure explore, 0=pure exploit)
    lp_ema: float = 0.85,
    # inner-loop FM ensemble (match Cut #3's capacity so control quality isn't FM-limited)
    k_ens: int = 4,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    rpf_beta: float = 0.6,                 # random-prior scale (0 = plain warm ensemble)
    buffer_n: int = 2800,                  # recency FIFO: stale patch data must age out to re-open disagreement
    seed_steps: int = 500,
    update_steps: int = 150,
    # schedule
    rounds: int = 60,
    eval_every: int = 4,
    # CEM-MPC control eval (Cut #3 planner)
    n_eval: int = 24,
    frontier_task: bool = True,            # goals/starts STRADDLE the frontier (undiluted payoff)
    reach_r: float = 0.45,                 # straddle radius (start & goal on opposite sides of the patch)
    goal_jit: float = 0.08,
    goal_range: float = 0.5,               # (frontier_task=False) ABSOLUTE goal half-range
    start_range: float = 0.5,              # (frontier_task=False) ABSOLUTE start half-range
    v0_std: float = 0.3,
    plan_h: int = 24,
    plan_hp: int = 15,
    replan_every: int = 8,
    k_shoot: int = 256,
    cem_iters: int = 3,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    arm_list = [a for a in (arms.split(",") if arms else ALL_ARMS) if a]
    if quick:
        arm_list = arms.split(",") if arms else ["disagree", "taskonly", "random"]
        grid_n = 5; rounds = 22; eval_every = 3; warmup_n = 900; seed_n = 300
        collect_n = 150; seed_steps = 200; update_steps = 90; k_ens = 3
        fm_hidden = 128; fm_layers = 2; buffer_n = 1200        # ages out within the smoke
        n_eval = 10; plan_h = 24; k_shoot = 128; probe_p = 6
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, drift_mode=drift_mode, value_rel=value_rel, noise=noise,
        frame_skip=frame_skip,
        dgp_base=dict(arena_half=arena_half, gear=gear, joint_damping=joint_damping, pusher_r=0.12),
        patch_amp=patch_amp, patch_sigma=patch_sigma, patch_phase=patch_phase,
        sweep_r=sweep_r, drift_cycles=drift_cycles, off_y=off_y,
        noise_amp=noise_amp, noise_x=noise_x, noise_y=noise_y,
        grid_n=grid_n, collect_range=collect_range, probe_p=probe_p, collect_n=collect_n,
        warmup_n=warmup_n, seed_n=seed_n,
        v_explore=v_explore, probe_v=probe_v, temp=temp, expl_eps=expl_eps, balance=balance,
        lp_ema=lp_ema,
        K=k_ens, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        rpf_beta=rpf_beta, buffer_n=buffer_n, seed_steps=seed_steps, update_steps=update_steps,
        rounds=rounds, eval_every=eval_every,
        n_eval=n_eval, frontier_task=frontier_task, reach_r=reach_r, goal_jit=goal_jit,
        goal_range=goal_range, start_range=start_range, v0_std=v0_std,
        plan_H=plan_h, plan_Hp=plan_hp,
        replan_every=replan_every, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_curiosity_control.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "curiosity_control_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
