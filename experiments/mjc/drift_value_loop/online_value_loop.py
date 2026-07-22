"""The full online value loop, with the CORRECTED teacher: self-tune the explore/exploit
balance by the value-relevant FM prediction-error re-adaptation signal, not by downstream
control.

Program: `ideas/two_timescale_value_loop.md`. Parent: `meta_curiosity_loop.py` (the same
two-timescale online loop graded by CONTROL reward — which found the control landscape over b
is FLAT, so the loop wandered by seed) + `ONLINE_VALUE_LOOP_README.md` (control is ROBUST to
the value: CEM-MPC replanning tolerates a stale model, so a better model barely registers as
better control). Sibling: `curiosity_control.py` (the fixed-b `grounded@b` sweep; the two
value terms — explore + exploit — as collection directors).

    THE DIAGNOSIS. Every prior attempt to let the system self-tune its own explore/exploit
    balance (this file's parent) graded the meta-loop by CONTROL reward — and control is a
    NEAR-BLIND GRADER of the value: a replanning controller reaches goals about as well with a
    stale model as a fresh one, so the control-reward-over-b landscape is flat and there is no
    gradient to climb. You cannot self-tune a value by a metric that cannot see what the value
    does.

    THE FIX (this cut). Grade the meta-loop by the *value-relevant FM re-adaptation error*
    instead — the mean forward-model prediction error over the task's VALUE-RELEVANT region
    (the goal corridor), a fair, non-privileged signal (it never knows where the drifting
    needle is; it just measures "how well do I model the dynamics where the controller
    operates"). This IS the idea doc's cerebellum→VTA *prediction-error messenger*: the value
    system reads the forward model's error, not the downstream reward. It is sensitive to
    exactly what the explore/exploit balance controls (how fast the value-relevant dynamics get
    re-modeled after a drift), so it gives the outer loop a real gradient.

    THE CLAIM. On a scarce, drifting frontier crossing the goal corridor + an off-corridor
    aleatoric noise distractor, the value-relevant re-adaptation error over b has an INTERIOR
    optimum (explore pinpoints the needle; exploit keeps the budget on the value-relevant
    corridor and off the noise), while downstream control is FLAT. So:
      * `online_front` (self-tune b by −corridor FM error) CONVERGES to the interior optimum;
      * `online_ctrl`  (self-tune b by −control dist, = the parent's teacher) WANDERS.
    The two value terms (explore = reducible surprise, the e-tap; exploit = value-relevance,
    the p-tap) are exposed to the meta-system, which weighs them of its own volition — the
    basal-ganglia meta-controller, with a teacher that can actually see the values.

INNER LOOP = `curiosity_control.py`, reused near-verbatim (self-contained duplication, the
established pattern — cf. directed_readapt/meta_adapt/meta_curiosity_loop): RPF ensemble
(Osband 2018) on a recency FIFO; teleport-region collection over a GxG grid scored by
`grounded@b` = b*reducible + (1-b)*exploit; CEM-MPC control readout; unbiased frontier / corridor
FM error. ONLY the b-SOURCE and the TEACHER differ across arms (the control-variable discipline):
`online_front`/`online_ctrl` learn b (front vs control teacher); `b<val>` fixes it (the landscape
reference lines); `random` is the floor.

WIREHEADING. b in (0,1) is a bounded, on-manifold knob with no loss-scale DOF; the corridor-error
teacher is a REAL measured prediction error over a fixed task-defined region (not a learned value
the loop can game). Stated, per #4e's discipline.

Run:
    cd experiments/
    modal run mujoco_control/online_value_loop.py::online_value_loop --quick                    # smoke
    # the teacher contrast (front vs control) + the fixed-b landscape, corridor geometry + noise:
    for s in 0 1 2; do
      modal run --detach mujoco_control/online_value_loop.py::online_value_loop --tag teacher_s$s --seed $s \
          --task-geom corridor --noise \
          --arms "online_front,online_ctrl,b0.0,b0.3,b0.5,b0.7,b1.0,random"
    done
    python3 mujoco_control/online_value_loop_figure.py     # the two-teacher contrast + b-landscape
"""

import json
import math
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder

# arms: `online_front`/`online_ctrl` learn b (front vs control teacher); `b<val>` fixes it
# (landscape reference lines); `random` = floor.
DEFAULT_ARMS = ["online_front", "online_ctrl", "b0.0", "b0.5", "b1.0", "random"]


def _online_teacher(arm):
    """For an online arm, which teacher grades the outer loop: 'front' (value-relevant FM
    re-adaptation error) or 'control' (downstream CEM goal-dist = the parent's blind grader)."""
    if not arm.startswith("online"):
        return None
    return "front" if arm.endswith("front") else "control"


def _parse_fixed_b(arm):
    """Return the fixed balance for a `b<val>` arm, else None (online / random)."""
    if arm.startswith("b") and arm[1:].replace(".", "", 1).isdigit():
        return float(arm[1:])
    return None


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_online_value_loop(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    arms = cfg["arms"]
    M = cfg["outer_m"]
    onset_round = int(cfg["noise_onset"] * cfg["rounds"]) if cfg["noise"] else cfg["rounds"] + 1
    print(f"[setup] device={device} noise={cfg['noise']} onset_round={onset_round} "
          f"rounds={cfg['rounds']} outer_m={M} arms={arms}", flush=True)

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def noise_on(rnd):
        return cfg["noise"] and rnd >= onset_round

    # ===================================================================== #
    # geometry: GxG collection grid over the task-relevant workspace
    # ===================================================================== #
    ah = cfg["dgp_base"]["arena_half"]
    inner = min(ah - cfg["dgp_base"]["pusher_r"] - 0.02, cfg["collect_range"])
    G = cfg["grid_n"]
    edges = np.linspace(-inner, inner, G + 1)
    ccent = 0.5 * (edges[:-1] + edges[1:])
    cell_half = float(inner / G)
    cell_centers = np.array([[cx, cy] for cy in ccent for cx in ccent], np.float32)
    n_cells = G * G

    def pos_to_cell(pos):
        ix = np.clip(np.searchsorted(edges, pos[..., 0]) - 1, 0, G - 1)
        iy = np.clip(np.searchsorted(edges, pos[..., 1]) - 1, 0, G - 1)
        return iy * G + ix

    # value-relevant region = the goal corridor (task-defined, NOT patch-location-privileged):
    # cells within a y-band of the corridor line (y=0) and |x|<=corridor_r. The corridor-error
    # TEACHER averages FM prediction error over these cells = "how well do I model the dynamics
    # where the controller operates" — the cerebellar prediction-error signal the value reads
    # (it never sees where the scarce needle is; it only measures value-relevant fidelity).
    _cx, _cy = cell_centers[:, 0], cell_centers[:, 1]
    corridor_cells = np.where((np.abs(_cy) <= cell_half * 1.2) &
                              (np.abs(_cx) <= cfg["corridor_r"] + cell_half))[0]
    if len(corridor_cells) == 0:
        corridor_cells = np.arange(n_cells)

    # ===================================================================== #
    # the drifting reducible frontier + optional aleatoric noisy-TV patch
    # ===================================================================== #
    sweep = cfg["sweep_r"]
    fr_y = 0.0 if cfg["value_rel"] == "on" else cfg["off_y"]

    def patch_center(rnd):
        if cfg["drift_mode"] == "none":
            return np.array([0.0, fr_y], np.float32)
        t = (rnd / max(cfg["rounds"] - 1, 1)) * cfg["drift_cycles"]
        tri = 2.0 * abs(t - np.floor(t + 0.5))
        return np.array([-sweep + 2 * sweep * tri, fr_y], np.float32)

    base_dgp = dict(cfg["dgp_base"])
    base_dgp["field_patch"] = dict(center=[0.0, fr_y], sigma=cfg["patch_sigma"],
                                   amp=cfg["patch_amp"], phase=cfg["patch_phase"])
    base_dgp["noise_seed"] = cfg["seed"] + 777              # seeds the env noise RNG regardless
    noise_cell = pos_to_cell(np.array([cfg["noise_x"], cfg["noise_y"]], np.float32)) if cfg["noise"] else -1

    env = PusherEnv(base_dgp, with_puck=False)
    eval_env = PusherEnv(base_dgp, with_puck=False)

    def set_phase(e, rnd):
        """Set the frontier center AND toggle the noisy-TV patch for this round's phase."""
        e.dgp["field_patch"]["center"] = patch_center(rnd).tolist()
        if noise_on(rnd):
            e.dgp["noise_patch"] = dict(center=[cfg["noise_x"], cfg["noise_y"]],
                                        sigma=cfg["patch_sigma"], amp=cfg["noise_amp"])
        else:
            e.dgp.pop("noise_patch", None)

    # ===================================================================== #
    # forward model f(s,u)->Δs (RPF ensemble member) + fixed normalization
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
        net = _mlp(seed)
        prior = _mlp(seed + 99991)
        for p in prior.parameters():
            p.requires_grad_(False)
        return {"net": net, "prior": prior}

    def mem_forward(m, Xn):
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
        return preds

    huber = nn.HuberLoss(delta=1.0)

    def update_member(m, opt, S, U, S2, norm, steps, brng):
        n = len(S)
        if n < 8:
            return
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], n); m["net"].train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, n, size=bs), device=device)
            opt.zero_grad(); huber(mem_forward(m, Xn[idx]), Yn[idx]).backward(); opt.step()
        m["net"].eval()

    # ===================================================================== #
    # collection primitives
    # ===================================================================== #
    def collect_in_cell(cell, k, rng):
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
        P = states.shape[1]; true = np.empty((len(cells), P, 4), np.float32)
        for ci, cell in enumerate(cells):
            for p in range(P):
                env.set_state(states[ci, p, :2].astype(np.float64), states[ci, p, 2:].astype(np.float64))
                s2, _ = env.step(cmds[ci, p], fs)
                true[ci, p] = s2 - states[ci, p]
        return true

    prng = np.random.default_rng(cfg["seed"] + 5)
    P = cfg["probe_p"]
    probe_states = np.empty((n_cells, P, 4), np.float32)
    probe_cmds = np.empty((n_cells, P, 2), np.float32)
    for ci in range(n_cells):
        c = cell_centers[ci]
        probe_states[ci, :, :2] = c + prng.uniform(-cell_half, cell_half, (P, 2))
        probe_states[ci, :, 2:] = prng.normal(0, cfg["probe_v"], (P, 2))
        probe_cmds[ci] = prng.uniform(-1, 1, (P, 2))

    # ===================================================================== #
    # CEM-MPC control eval (ensemble-mean model) — the REWARD signal
    # ===================================================================== #
    Hp, Kc, ne = cfg["plan_Hp"], cfg["k_shoot"], cfg["cem_elite"]
    vel_pen, re_ = cfg["vel_pen"], cfg["replan_every"]
    B = cfg["n_eval"]
    rand_dist = 2 * (cfg["corridor_r"] if cfg["task_geom"] == "corridor" else cfg["reach_r"])

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
                                         for m in fms], 0).mean(0)
                    s = s + dpred
                    cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_control(fms, norm, rnd, seed):
        """Median goal-dist at the CURRENT phase (frontier drift + noise if active).

        task_geom:
          * `straddle` — start & goal straddle the CURRENT patch center (undiluted, but the
            exploit=path-density signal then concentrates AT the frontier, so pure-exploit is a
            strong tracker and the reward landscape over b is ~flat — the frontier_task boundary).
          * `corridor` — FIXED long straight reaches (-cr,0)->(+cr,0) that ALWAYS cross the
            drifting needle (control NEEDS it, undiluted) but whose path-density is UNIFORM over
            the whole corridor, so exploit CANNOT localize the scarce needle — only explore can.
            This is the anticipation/scarcity geometry that gives the online loop a real gradient."""
        set_phase(eval_env, rnd)
        rng = np.random.default_rng(seed)
        if cfg["task_geom"] == "corridor":
            cr = cfg["corridor_r"]
            sgn = rng.choice([-1.0, 1.0], B).astype(np.float32)
            spos = (np.stack([sgn * cr, np.zeros(B, np.float32)], 1)
                    + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32))
            goals = (np.stack([-sgn * cr, np.zeros(B, np.float32)], 1)
                     + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32))
        else:
            pc = patch_center(rnd).astype(np.float32)
            ang = rng.uniform(0, 2 * np.pi, B)
            dvec = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
            spos = pc[None] + cfg["reach_r"] * dvec
            goals = (pc[None] - cfg["reach_r"] * dvec
                     + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2))).astype(np.float32)
        starts = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)
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

    # ===================================================================== #
    # normalization from a fixed uniform warmup pool (identical across arms)
    # ===================================================================== #
    wrng = np.random.default_rng(cfg["seed"] + 3)
    Sw = np.empty((cfg["warmup_n"], 4), np.float32); Uw = np.empty((cfg["warmup_n"], 2), np.float32)
    S2w = np.empty((cfg["warmup_n"], 4), np.float32)
    set_phase(env, 0)
    for i in range(cfg["warmup_n"]):
        pos = wrng.uniform(-inner, inner, 2); vel = wrng.normal(0, cfg["v_explore"], 2)
        env.set_state(pos.astype(np.float64), vel.astype(np.float64))
        u = wrng.uniform(-1, 1, 2).astype(np.float32)
        s = env.get_state(); s2, _ = env.step(u, fs)
        Sw[i] = s; Uw[i] = u; S2w[i] = s2
    norm = make_norm(Sw, Uw, S2w)

    # ===================================================================== #
    # per-cell drive scores (identical to curiosity_control)
    # ===================================================================== #
    def score_disagree(fms):
        pr = ens_predict(fms, probe_states.reshape(-1, 4), probe_cmds.reshape(-1, 2), norm)
        var = pr.var(0).mean(1)
        return var.reshape(n_cells, P).mean(1).cpu().numpy()

    def score_error(fms, rnd):
        set_phase(env, rnd)
        true = collect_probe(range(n_cells), probe_states, probe_cmds)
        pr = ens_predict(fms, probe_states.reshape(-1, 4), probe_cmds.reshape(-1, 2), norm)
        pm = pr.mean(0).cpu().numpy().reshape(n_cells, P, 4)
        return np.linalg.norm(pm - true, axis=2).mean(1)

    def score_exploit(rnd):
        rng = np.random.default_rng(cfg["seed"] + 4242 + rnd)
        Mn, T = 96, 24
        if cfg["task_geom"] == "corridor":
            # density of the FIXED corridor reaches -> UNIFORM over [-cr,cr]x{0}; does NOT
            # localize the scarce drifting needle (only explore can). This is what breaks the
            # exploit-preempts-the-frontier flatness of the straddle geometry.
            cr = cfg["corridor_r"]
            sgn = rng.choice([-1.0, 1.0], Mn)
            A = np.stack([sgn * cr, np.zeros(Mn)], 1).astype(np.float32)
            Bx = np.stack([-sgn * cr, np.zeros(Mn)], 1).astype(np.float32)
        else:
            pc = patch_center(rnd).astype(np.float32)
            ang = rng.uniform(0, 2 * np.pi, Mn)
            dvec = np.stack([np.cos(ang), np.sin(ang)], 1)
            A = pc[None] + cfg["reach_r"] * dvec
            Bx = pc[None] - cfg["reach_r"] * dvec
        ts = np.linspace(0, 1, T)[None, :, None]
        pts = (A[:, None, :] * (1 - ts) + Bx[:, None, :] * ts).reshape(-1, 2).astype(np.float32)
        return np.bincount(pos_to_cell(pts), minlength=n_cells).astype(np.float64)

    def grounded_scores(fms, rnd, b):
        """b*reducible + (1-b)*exploit — the e-tap + p-tap as two additive drives."""
        err = score_error(fms, rnd); dis = score_disagree(fms)
        en = (err - err.min()) / (err.max() - err.min() + 1e-9)
        dn = (dis - dis.min()) / (dis.max() - dis.min() + 1e-9)
        exn = en * dn
        ap = score_exploit(rnd)
        apn = (ap - ap.min()) / (ap.max() - ap.min() + 1e-9)
        return b * exn + (1 - b) * apn + 1e-6

    # ===================================================================== #
    # run one arm
    # ===================================================================== #
    def run_arm(arm):
        fixed_b = _parse_fixed_b(arm)
        is_online = arm.startswith("online")
        teacher = _online_teacher(arm)                       # 'front' | 'control' | None
        arng = np.random.default_rng(cfg["seed"] + 100)      # SAME across arms -> only b-source differs
        fms = [build_member(cfg["seed"] + 40 + j) for j in range(cfg["K"])]
        opts = [torch.optim.Adam(m["net"].parameters(), lr=cfg["fm_lr"]) for m in fms]
        bufS = np.zeros((0, 4), np.float32); bufU = np.zeros((0, 2), np.float32)
        bufS2 = np.zeros((0, 4), np.float32); bufAC = np.zeros((0,), bool)

        def add(S, U, S2, AC):
            nonlocal bufS, bufU, bufS2, bufAC
            bufS = np.concatenate([bufS, S])[-cfg["buffer_n"]:]
            bufU = np.concatenate([bufU, U])[-cfg["buffer_n"]:]
            bufS2 = np.concatenate([bufS2, S2])[-cfg["buffer_n"]:]
            bufAC = np.concatenate([bufAC, AC])[-cfg["buffer_n"]:]

        set_phase(env, 0)
        add(*collect_in_cell(pos_to_cell(np.array([0.0, 0.0])), cfg["seed_n"], arng))
        for j, (fm, opt) in enumerate(zip(fms, opts)):
            update_member(fm, opt, bufS, bufU, bufS2, norm, cfg["seed_steps"],
                          np.random.default_rng(cfg["seed"] + 200 + j))

        # ---- outer-loop (slow) state for the `online_*` arms ----
        theta = math.log(cfg["b_init"] / (1 - cfg["b_init"]))    # b_init -> theta
        cur_eps = 0.0
        cur_b = cfg["b_init"] if is_online else (fixed_b if fixed_b is not None else 0.5)
        base = None; run_sq = None                               # REINFORCE baseline / advantage-normalizer
        warmup = cfg["outer_warmup"]

        occ = np.zeros(n_cells); occ_hist = []; recs = []
        patch_frac_hist = []; noise_frac_hist = []; b_hist = []; outer_hist = []

        for rnd in range(cfg["rounds"]):
            epoch = rnd // M
            is_epoch_start = (rnd % M == 0)
            is_epoch_end = (rnd % M == M - 1) or (rnd == cfg["rounds"] - 1)

            # --- outer loop: choose b for this epoch (online) ---
            if is_online and is_epoch_start:
                cur_eps = 0.0 if epoch < warmup else cfg["outer_sigma"] * float(arng.standard_normal())
                cur_b = sigmoid(max(-4.0, min(4.0, theta + cur_eps)))
            b_r = cur_b if is_online else fixed_b              # None for the `random` arm
            b_val = float(b_r) if b_r is not None else float("nan")

            # --- pick a cell by the drive, collect a fixed budget ---
            if arm == "random":
                scores = np.ones(n_cells)
            else:
                scores = grounded_scores(fms, rnd, b_r)
            sc = scores / (scores.sum() + 1e-12)
            logit = np.log(sc + 1e-12) / cfg["temp"]
            p = np.exp(logit - logit.max()); p /= p.sum()
            if arng.random() < cfg["expl_eps"]:
                cell = int(arng.integers(n_cells))
            else:
                cell = int(arng.choice(n_cells, p=p))
            occ[cell] += 1.0
            chosen = np.zeros(n_cells); chosen[cell] = 1.0
            set_phase(env, rnd)
            add(*collect_in_cell(cell, cfg["collect_n"], arng))

            # --- update ensemble on the recency buffer ---
            for j, (fm, opt) in enumerate(zip(fms, opts)):
                update_member(fm, opt, bufS, bufU, bufS2, norm, cfg["update_steps"],
                              np.random.default_rng(cfg["seed"] + 300 + rnd * 17 + j))

            fc = pos_to_cell(patch_center(rnd)[None])[0]
            patch_frac_hist.append(float(chosen[fc]))
            noise_frac_hist.append(float(chosen[noise_cell]) if noise_cell >= 0 else 0.0)
            b_hist.append(b_val)

            # --- epoch boundary: eval BOTH teachers; update the outer loop (online) ---
            if is_epoch_end:
                cdist = eval_control(fms, norm, rnd, cfg["seed"] + 7000 + rnd)
                all_err = score_error(fms, rnd)                  # per-cell FM prediction error
                fr_err = float(all_err[fc])                      # error AT the (privileged) needle cell
                corr_err = float(np.mean(all_err[corridor_cells]))  # value-relevant TEACHER (fair)
                recs.append({"round": rnd, "epoch": epoch, "control_dist": cdist,
                             "frontier_err": fr_err, "corridor_err": corr_err,
                             "b": b_val, "noise": noise_on(rnd)})
                occ_hist.append(occ.copy())

                if is_online:
                    # THE TEACHER SWAP: 'front' grades by the value-relevant FM re-adaptation
                    # error (higher R = lower corridor error); 'control' by downstream goal-dist
                    # (the parent's near-blind grader). Reward normalized to be scale-agnostic.
                    R = -corr_err if teacher == "front" else -cdist
                    adv = 0.0 if base is None else (R - base)
                    base = R if base is None else cfg["outer_rho"] * base + (1 - cfg["outer_rho"]) * R
                    # lazy-init the advantage-normalizer on the RIGHT scale (so a fixed run_sq=1
                    # doesn't kill adv_n for many epochs) — prime it during warmup.
                    if adv != 0.0:
                        run_sq = adv ** 2 if run_sq is None else \
                            cfg["outer_rho"] * run_sq + (1 - cfg["outer_rho"]) * (adv ** 2)
                    if epoch >= warmup and run_sq is not None:
                        adv_n = max(-3.0, min(3.0, adv / (math.sqrt(run_sq) + 1e-6)))
                        # score-function grad of the Gaussian policy b=sigmoid(theta+eps): d/dtheta = eps/sigma^2
                        grad = adv_n * (cur_eps / (cfg["outer_sigma"] ** 2))
                        theta = max(-4.0, min(4.0, theta + cfg["outer_alpha"] * grad))
                    outer_hist.append({"epoch": epoch, "theta": float(theta), "b_used": float(b_r),
                                       "eps": float(cur_eps), "reward": float(R), "teacher": teacher,
                                       "adv": float(adv), "b_theta": float(sigmoid(theta))})
                    print(f"  [{arm:12s} e{epoch:2d} r{rnd:3d}] b={b_r:.3f} theta={theta:+.2f} "
                          f"ctrl={cdist:.4f} corr_err={corr_err:.4f} R={R:+.4f} adv={adv:+.4f}", flush=True)
                else:
                    print(f"  [{arm:12s} r{rnd:3d}] b={float(b_r) if fixed_b is not None else float('nan'):.3f} "
                          f"ctrl={cdist:.4f} corr_err={corr_err:.4f} front={fr_err:.4f}", flush=True)

        return {"records": recs, "occupancy": occ.tolist(),
                "occ_hist": [o.tolist() for o in occ_hist],
                "patch_frac_hist": patch_frac_hist, "noise_frac_hist": noise_frac_hist,
                "b_hist": b_hist, "outer_hist": outer_hist,
                "final_b": float(np.mean(b_hist[-min(len(b_hist), 4 * M):]))}

    # ===================================================================== #
    # run all arms
    # ===================================================================== #
    results = {}
    for arm in arms:
        print(f"\n[arm] {arm}", flush=True)
        results[arm] = run_arm(arm)

    def final_ctrl(arm):
        r = [x["control_dist"] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    def final_metric(arm, key):
        r = [x[key] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    summary = {a: {"final_control": final_metric(a, "control_dist"),
                   "final_frontier_err": final_metric(a, "frontier_err"),
                   "final_corridor_err": final_metric(a, "corridor_err"),
                   "final_b": results[a]["final_b"],
                   "patch_frac_last": float(np.mean(results[a]["patch_frac_hist"][-10:])),
                   "noise_frac_last": float(np.mean(results[a]["noise_frac_hist"][-10:]))}
               for a in arms}
    print("\n[summary] final control (lower=better) | corridor-err (TEACHER) | learned-b | patch | noise", flush=True)
    for a in arms:
        s = summary[a]
        print(f"  {a:13s} ctrl={s['final_control']:.4f}  corr_err={s['final_corridor_err']:.4f}  "
              f"front={s['final_frontier_err']:.4f}  b={s['final_b']:.3f}  "
              f"patch={s['patch_frac_last']:.2f}  noise={s['noise_frac_last']:.2f}", flush=True)
    # the teacher contrast: which fixed-b minimizes each landscape (the target each online arm chases)
    fixed = [a for a in arms if _parse_fixed_b(a) is not None]
    if fixed:
        best_corr = min(fixed, key=lambda a: summary[a]["final_corridor_err"])
        best_ctrl = min(fixed, key=lambda a: summary[a]["final_control"])
        corr_vals = [summary[a]["final_corridor_err"] for a in fixed]
        ctrl_vals = [summary[a]["final_control"] for a in fixed]
        print(f"  [landscape] corridor-err spread over fixed-b = {max(corr_vals)-min(corr_vals):.4f} "
              f"(argmin {best_corr}) | control spread = {max(ctrl_vals)-min(ctrl_vals):.4f} (argmin {best_ctrl})", flush=True)
        print(f"              -> the TEACHER with the larger spread has the real gradient; "
              f"a flat control spread = the blind grader", flush=True)
        for a in [x for x in arms if x.startswith("online")]:
            print(f"  [online] {a}: learned b={summary[a]['final_b']:.3f}", flush=True)

    out = {"config": cfg, "regime": ("stationary" if cfg["drift_mode"] == "none" else "drift"),
           "noise": cfg["noise"], "noise_onset_round": onset_round, "grid_n": G,
           "cell_centers": cell_centers.tolist(), "inner": inner,
           "patch_path": [patch_center(r).tolist() for r in range(cfg["rounds"])],
           "noise_cell": int(noise_cell), "rand_dist": rand_dist,
           "results": results, "summary": summary}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "online_value_loop", cfg["tag"])
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
    COL = {"online_front": "#e8590c", "online_ctrl": "#7048e8",
           "b1.0": "#0ca678", "b0.7": "#40c057", "b0.5": "#2f9e44", "b0.3": "#66a80f",
           "b0.0": "#3d6fd1", "random": "#868e96"}
    arms = list(R["results"].keys())
    onset = R["noise_onset_round"]
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    def col(a):
        return COL.get(a) or "#9c36b5"

    # ---- fig1: the learned-b trajectories, front vs control teacher (the headline) ----
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for a in [x for x in arms if x.startswith("online")]:
        oh = R["results"][a]["outer_hist"]
        if not oh:
            continue
        ep = [h["epoch"] for h in oh]
        lbl = "front teacher (corridor FM err)" if a.endswith("front") else "control teacher (goal-dist)"
        ax.plot(ep, [h["b_theta"] for h in oh], "o-", color=col(a), lw=2.4, label=f"{a}: {lbl}")
        ax.plot(ep, [h["b_used"] for h in oh], "x", color=col(a), alpha=0.3, ms=5)
    for a in arms:
        fb = _parse_fixed_b(a)
        if fb is not None:
            ax.axhline(fb, color=col(a), ls="--", lw=1.2, alpha=0.8, label=f"fixed {a}")
    if R["noise"] and onset <= R["config"]["rounds"]:
        oe = onset / R["config"]["outer_m"]
        ax.axvline(oe, color="red", ls=":", lw=1.5, label="noise onset")
    ax.set_xlabel("outer-loop epoch"); ax.set_ylabel("explore/exploit balance b  (1=pure explore, 0=pure exploit)")
    ax.set_ylim(-0.02, 1.02)
    ttl = "self-tuning b: the FRONT teacher (FM error) converges; the CONTROL teacher wanders"
    ax.set_title(ttl, fontsize=9.5); ax.legend(fontsize=7.5, loc="best")
    figs["fig1_learned_b.png"] = _save(fig)

    # ---- fig2: control competence + frontier tracking over epochs ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for a in arms:
        rec = R["results"][a]["records"]
        x = [r["epoch"] for r in rec]
        axes[0].plot(x, [r["control_dist"] for r in rec], "o-", ms=3, color=col(a), label=a)
        axes[1].plot(x, [r["frontier_err"] for r in rec], "o-", ms=3, color=col(a), label=a)
    axes[0].axhline(R["rand_dist"], color="#bbb", ls=":", lw=1, label="random floor")
    for ax in axes:
        if R["noise"] and onset <= R["config"]["rounds"]:
            ax.axvline(onset / R["config"]["outer_m"], color="red", ls=":", lw=1.3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("median goal-dist (lower=better)")
    axes[0].set_title("control competence"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("frontier-region FM error (lower=better)")
    axes[1].set_title("tracking the moving frontier"); axes[1].legend(fontsize=7)
    figs["fig2_control.png"] = _save(fig)

    # ---- fig3: occupancy heatmaps ----
    G = R["grid_n"]; path = np.array(R["patch_path"])
    cc = np.array(R["cell_centers"]); inner = R["inner"]
    show = [a for a in ["online_front", "online_ctrl", "b1.0", "b0.0", "random"] if a in arms]
    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, show):
        occ = np.array(R["results"][a]["occupancy"]).reshape(G, G)
        ax.imshow(occ, origin="lower", extent=[-inner, inner, -inner, inner], cmap="magma", aspect="equal")
        ax.plot(path[:, 0], path[:, 1], "-", color="cyan", lw=1.3, alpha=0.8)
        if R["noise_cell"] >= 0:
            ncc = cc[R["noise_cell"]]
            ax.scatter([ncc[0]], [ncc[1]], c="red", s=60, marker="x", lw=2)
        lbl = a + (f" (b→{R['results'][a]['final_b']:.2f})" if a.startswith("online") else "")
        ax.set_title(lbl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"collection occupancy (cyan=frontier path, red×=noise) [{R['regime']}, noise={R['noise']}]", fontsize=9)
    figs["fig3_occupancy.png"] = _save(fig)

    # ---- fig4: order-parameter bars ----
    S = R["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].bar(range(len(arms)), [S[a]["final_control"] for a in arms], color=[col(a) for a in arms])
    axes[0].axhline(R["rand_dist"], color="#bbb", ls=":", lw=1)
    axes[0].set_xticks(range(len(arms))); axes[0].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("final control dist (lower=better)"); axes[0].set_title("control")
    axes[1].bar(range(len(arms)), [S[a]["final_frontier_err"] for a in arms], color=[col(a) for a in arms])
    axes[1].set_xticks(range(len(arms))); axes[1].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("final frontier err"); axes[1].set_title("frontier tracking")
    axes[2].bar(range(len(arms)), [S[a]["patch_frac_last"] for a in arms], color=[col(a) for a in arms], label="patch")
    if R["noise_cell"] >= 0:
        axes[2].bar(range(len(arms)), [S[a]["noise_frac_last"] for a in arms], color="red", alpha=0.5,
                    width=0.4, label="noise")
        axes[2].legend(fontsize=7)
    axes[2].set_xticks(range(len(arms))); axes[2].set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes[2].set_ylabel("collection fraction"); axes[2].set_title("where they collect (last 10 rounds)")
    figs["fig4_orderparams.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def online_value_loop(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "",
    # regime dials
    drift_mode: str = "morph",
    value_rel: str = "on",
    noise: bool = False,
    noise_onset: float = 0.0,             # fraction of the run before noise turns on (0=from start)
    # task geometry (see eval_control): `straddle` = frontier_task (exploit preempts frontier ->
    # flat over b); `corridor` = long fixed reaches crossing a scarce needle (explore load-bearing
    # -> interior optimum, the geometry the online loop needs)
    task_geom: str = "corridor",           # corridor = fixed reaches crossing a scarce needle
                                           # (exploit can't localize it -> explore load-bearing ->
                                           # interior b optimum; the geometry this loop needs)
    corridor_r: float = 1.0,              # corridor half-length; reaches span [-cr,cr]x{0}
    # outer loop (the slow reward-driven set-point tuner)
    outer_m: int = 6,                     # rounds per outer epoch (the two-timescale ratio)
    outer_alpha: float = 0.3,             # outer learning rate on theta
    outer_sigma: float = 0.7,             # exploration std in theta-space
    outer_rho: float = 0.8,               # EMA for baseline + advantage-normalizer
    outer_warmup: int = 2,                # epochs at b_init before tuning (warms the baseline)
    b_init: float = 0.5,                  # neutral prior set-point (partially-frozen start)
    # env / family (Cut #3's momentum reaching — a KNOWN-WORKING controller)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    joint_damping: float = 2.0,
    patch_amp: float = 3.5,
    patch_sigma: float = 0.30,
    patch_phase: float = 0.0,
    sweep_r: float = 0.5,
    drift_cycles: float = 2.0,
    off_y: float = 1.1,
    noise_amp: float = 6.0,
    noise_x: float = 0.0,                   # OFF the goal corridor (y=0): exploit steers away from
    noise_y: float = 0.9,                   # it, only over-eager explore chases it -> interior b optimum
    # grid / collection
    grid_n: int = 6,
    collect_range: float = 1.25,
    probe_p: int = 10,
    collect_n: int = 240,
    warmup_n: int = 2000,
    seed_n: int = 500,
    v_explore: float = 1.2,
    probe_v: float = 0.15,
    temp: float = 0.4,
    expl_eps: float = 0.1,
    # inner-loop FM ensemble
    k_ens: int = 4,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    rpf_beta: float = 0.6,
    buffer_n: int = 2800,
    seed_steps: int = 500,
    update_steps: int = 150,
    # schedule
    rounds: int = 96,
    # CEM-MPC control eval / reward
    n_eval: int = 36,
    reach_r: float = 0.45,
    goal_jit: float = 0.08,
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

    arm_list = [a for a in (arms.split(",") if arms else DEFAULT_ARMS) if a]
    if quick:
        arm_list = arms.split(",") if arms else ["online_front", "online_ctrl", "b0.0", "b1.0"]
        grid_n = 5; rounds = 30; outer_m = 5; outer_warmup = 1; warmup_n = 900; seed_n = 300
        collect_n = 150; seed_steps = 200; update_steps = 90; k_ens = 3
        fm_hidden = 128; fm_layers = 2; buffer_n = 1200
        n_eval = 12; plan_h = 24; k_shoot = 128; probe_p = 6
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, drift_mode=drift_mode, value_rel=value_rel, noise=noise,
        noise_onset=noise_onset, task_geom=task_geom, corridor_r=corridor_r,
        outer_m=outer_m, outer_alpha=outer_alpha, outer_sigma=outer_sigma,
        outer_rho=outer_rho, outer_warmup=outer_warmup, b_init=b_init,
        frame_skip=frame_skip,
        dgp_base=dict(arena_half=arena_half, gear=gear, joint_damping=joint_damping, pusher_r=0.12),
        patch_amp=patch_amp, patch_sigma=patch_sigma, patch_phase=patch_phase,
        sweep_r=sweep_r, drift_cycles=drift_cycles, off_y=off_y,
        noise_amp=noise_amp, noise_x=noise_x, noise_y=noise_y,
        grid_n=grid_n, collect_range=collect_range, probe_p=probe_p, collect_n=collect_n,
        warmup_n=warmup_n, seed_n=seed_n,
        v_explore=v_explore, probe_v=probe_v, temp=temp, expl_eps=expl_eps,
        K=k_ens, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        rpf_beta=rpf_beta, buffer_n=buffer_n, seed_steps=seed_steps, update_steps=update_steps,
        rounds=rounds,
        n_eval=n_eval, reach_r=reach_r, goal_jit=goal_jit, v0_std=v0_std,
        plan_H=plan_h, plan_Hp=plan_hp, replan_every=replan_every, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_online_value_loop.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "online_value_loop_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
