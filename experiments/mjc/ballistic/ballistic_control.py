"""Cut 4 (drift_value_loop): the BALLISTIC controller — making the FM bridge transmit to
behavior.

Program: `ideas/two_timescale_value_loop.md`. Parent: `online_value_loop.py` (Cut 3 — the
corrected teacher: grade the meta-loop by the value-relevant FM prediction-error, not
downstream control, because a REPLANNING controller is a NEAR-BLIND grader of the FM).
Writeup: `drift_value_loop/README.md`.

    THE DIAGNOSIS CARRIED FORWARD. Cut 3 found the control-reward-over-b landscape is FLAT
    (a replanning CEM-MPC controller reaches goals about as well with a stale FM as a fresh
    one, so a better FM barely registers as better control). That is not a fact about value —
    it is a fact about the CONTROLLER. CEM-MPC re-grounds to the true state every `replan_every`
    steps, so FM error never compounds and the FM is behaviorally optional.

    THE BIOLOGICAL POINT. The cerebellar forward model exists PRECISELY because you cannot
    replan fast enough under sensorimotor delay — biological motor control is ballistic /
    feedforward over a horizon. A ballistic controller commits to an open-loop plan and CANNOT
    re-ground mid-flight, so FM error compounds along the trajectory and a better FM becomes a
    better controller. The value of the FM — and hence of the meta-loop that shapes it — should
    be PROPORTIONAL TO the commitment horizon. Reactive replanning is the un-biological
    component that made the meta layer look inert.

    THE CONTROLLED TRICK. For a fixed-b arm the FM trajectory is INDEPENDENT of the controller
    (collection is drive-directed; the controller is eval-only). So we collect each b ONCE and
    grade the IDENTICAL FMs with controllers of varying commitment horizon
    `replan_every in {reactive .. ballistic}`, PLANNING HORIZON HELD FIXED at the full episode
    (only the grader's ballistic-ness changes — clean single-variable isolation).

    EXPERIMENT 1 (headline). The control-over-b landscape at each commitment horizon, with the
    corridor-FM-error landscape (controller-INVARIANT) as the reference.
      PREDICTION: control-over-b is FLAT at reactive replanning (reproduces Cut 3's blind
      grader) and develops an INTERIOR OPTIMUM tracking the FM-error optimum as it goes
      ballistic. Money readout: spread(control-over-b) grows with the commitment horizon while
      spread(corridor-FM-error) is invariant.  => the efferent value->FM bridge TRANSMITS to
      behavior once the controller is ballistic.

    EXPERIMENT 2 (payoff). Self-tune b from a displaced init. `online_ctrl` graded by a
    BALLISTIC controller should now CONVERGE toward the interior optimum (like Cut 3's
    `online_front` did), where Cut 3's reactive `online_ctrl` WANDERED. The blind grader becomes
    sighted once control is ballistic.

INNER LOOP = `online_value_loop.py`, reused near-verbatim (self-contained duplication, the
established pattern — cf. directed_readapt/meta_adapt/meta_curiosity_loop). RPF ensemble
(Osband 2018) on a recency FIFO; teleport-region collection over a GxG grid scored by
`grounded@b` = b*reducible + (1-b)*exploit; unbiased corridor FM error. The ONLY changes vs the
parent: (i) `eval_control` takes an explicit `replan_every` and plans the FULL horizon so it can
run fully open-loop (ballistic); (ii) at each epoch boundary control is evaluated at a SWEEP of
commitment horizons on the identical FM; (iii) the `online_ctrl` teacher is graded at a chosen
`teacher_replan` (ballistic by default).

WIREHEADING. b in (0,1) is a bounded, on-manifold knob with no loss-scale DOF; the corridor-error
teacher is a REAL measured prediction error over a fixed task-defined region; the ballistic
control teacher is a REAL measured goal-distance. No learned value the loop can game (per #4e's
discipline).

Run:
    cd experiments/
    modal run mjc/ballistic/ballistic_control.py::ballistic_control --quick                    # smoke
    # --- Exp 1: the commitment-horizon landscape (3 seeds) ---
    for s in 0 1 2; do
      modal run --detach mjc/ballistic/ballistic_control.py::ballistic_control --tag land_s$s --seed $s \
          --task-geom corridor --noise --replan-sweep "2,8,24" \
          --arms "b0.0,b0.3,b0.5,b0.7,b1.0,random"
    done
    # --- Exp 2: self-tuning under the ballistic teacher (displaced init, 3 seeds) ---
    for s in 0 1 2; do
      modal run --detach mjc/ballistic/ballistic_control.py::ballistic_control --tag selftune_s$s --seed $s \
          --task-geom corridor --noise --b-init 0.15 --teacher-replan 24 --replan-sweep "2,24" \
          --arms "online_ctrl,online_front"
    done
    python3 mjc/ballistic/ballistic_control_figure.py --land-tags land_s0 land_s1 land_s2 \
        --selftune-tags selftune_s0 selftune_s1 selftune_s2
"""

import json
import math
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# fixed-b arms give the landscape; `online_front`/`online_ctrl` self-tune (Exp 2); `random`=floor.
DEFAULT_ARMS = ["b0.0", "b0.3", "b0.5", "b0.7", "b1.0", "random"]


def _online_teacher(arm):
    """For an online arm, which teacher grades the outer loop: 'front' (value-relevant FM
    re-adaptation error) or 'control' (downstream ballistic-CEM goal-dist)."""
    if not arm.startswith("online"):
        return None
    return "front" if arm.endswith("front") else "control"


def _parse_fixed_b(arm):
    """Return the fixed balance for a `b<val>` arm, else None (online / random)."""
    if arm.startswith("b") and arm[1:].replace(".", "", 1).isdigit():
        return float(arm[1:])
    return None


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_ballistic_control(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    arms = cfg["arms"]
    M = cfg["outer_m"]
    replan_sweep = cfg["replan_sweep"]                        # the commitment-horizon knobs
    onset_round = int(cfg["noise_onset"] * cfg["rounds"]) if cfg["noise"] else cfg["rounds"] + 1
    print(f"[setup] device={device} noise={cfg['noise']} onset_round={onset_round} "
          f"rounds={cfg['rounds']} outer_m={M} replan_sweep={replan_sweep} "
          f"teacher_replan={cfg['teacher_replan']} plan_H={cfg['plan_H']} plan_Hp={cfg['plan_Hp']} "
          f"arms={arms}", flush=True)

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

    # value-relevant region = the goal corridor (task-defined, NOT patch-location-privileged).
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
    base_dgp["noise_seed"] = cfg["seed"] + 777
    noise_cell = pos_to_cell(np.array([cfg["noise_x"], cfg["noise_y"]], np.float32)) if cfg["noise"] else -1

    env = PusherEnv(base_dgp, with_puck=False)
    eval_env = PusherEnv(base_dgp, with_puck=False)

    def set_phase(e, rnd):
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
    # CEM-MPC control eval — now with an EXPLICIT commitment horizon (replan_every).
    # The planning horizon is FIXED at the full episode (plan_Hp = plan_H) so the SAME
    # controller can run reactive (replan often) or ballistic (plan once, open-loop) with
    # ONLY the commitment horizon changing.
    # ===================================================================== #
    Hp, Kc, ne = cfg["plan_Hp"], cfg["k_shoot"], cfg["cem_elite"]
    vel_pen = cfg["vel_pen"]
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

    def _episode_geometry(rnd, seed):
        """Deterministic start/goal states for a given (rnd, seed): SHARED across replan
        settings so the commitment horizons are graded on IDENTICAL episodes (controlled)."""
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
        return starts, goals, rng

    def eval_control(fms, norm, rnd, seed, replan_every):
        """Median goal-dist at the CURRENT phase, committing `replan_every` steps open-loop.

        replan_every=1        -> fully reactive (re-ground every step = the near-blind grader);
        replan_every>=plan_H  -> fully BALLISTIC (plan once, execute the whole episode open-loop,
                                 no re-grounding — FM error compounds along the trajectory).
        The planning horizon (plan_Hp) is fixed at the full episode, so ONLY the commitment
        horizon varies across the sweep (single-variable isolation)."""
        set_phase(eval_env, rnd)
        starts, goals, rng = _episode_geometry(rnd, seed)
        states = starts.copy(); plan = None
        for step in range(cfg["plan_H"]):
            if step % replan_every == 0:
                plan = mpc_action(fms, norm, states, goals, rng)
            acts = plan[:, min(step % replan_every, Hp - 1), :]
            for b in range(B):
                eval_env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = eval_env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - goals, axis=1)
        return float(np.median(fd))

    def eval_control_sweep(fms, norm, rnd, seed):
        """Grade the IDENTICAL FMs at every commitment horizon in the sweep (shared episodes)."""
        return {re: eval_control(fms, norm, rnd, seed, re) for re in replan_sweep}

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
    # per-cell drive scores (identical to online_value_loop / curiosity_control)
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
        teacher = _online_teacher(arm)
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
        theta = math.log(cfg["b_init"] / (1 - cfg["b_init"]))
        cur_eps = 0.0
        cur_b = cfg["b_init"] if is_online else (fixed_b if fixed_b is not None else 0.5)
        base = None; run_sq = None
        warmup = cfg["outer_warmup"]

        occ = np.zeros(n_cells); occ_hist = []; recs = []
        patch_frac_hist = []; noise_frac_hist = []; b_hist = []; outer_hist = []

        for rnd in range(cfg["rounds"]):
            epoch = rnd // M
            is_epoch_start = (rnd % M == 0)
            is_epoch_end = (rnd % M == M - 1) or (rnd == cfg["rounds"] - 1)

            if is_online and is_epoch_start:
                cur_eps = 0.0 if epoch < warmup else cfg["outer_sigma"] * float(arng.standard_normal())
                cur_b = sigmoid(max(-4.0, min(4.0, theta + cur_eps)))
            b_r = cur_b if is_online else fixed_b
            b_val = float(b_r) if b_r is not None else float("nan")

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

            for j, (fm, opt) in enumerate(zip(fms, opts)):
                update_member(fm, opt, bufS, bufU, bufS2, norm, cfg["update_steps"],
                              np.random.default_rng(cfg["seed"] + 300 + rnd * 17 + j))

            fc = pos_to_cell(patch_center(rnd)[None])[0]
            patch_frac_hist.append(float(chosen[fc]))
            noise_frac_hist.append(float(chosen[noise_cell]) if noise_cell >= 0 else 0.0)
            b_hist.append(b_val)

            if is_epoch_end:
                # grade the IDENTICAL FMs at every commitment horizon (shared episodes)
                cbr = eval_control_sweep(fms, norm, rnd, cfg["seed"] + 7000 + rnd)
                all_err = score_error(fms, rnd)
                fr_err = float(all_err[fc])
                corr_err = float(np.mean(all_err[corridor_cells]))
                recs.append({"round": rnd, "epoch": epoch,
                             "control_by_replan": {str(k): float(v) for k, v in cbr.items()},
                             "frontier_err": fr_err, "corridor_err": corr_err,
                             "b": b_val, "noise": noise_on(rnd)})
                occ_hist.append(occ.copy())

                if is_online:
                    # THE TEACHER: 'front' grades by the value-relevant FM re-adaptation error;
                    # 'control' by the BALLISTIC downstream goal-dist (at teacher_replan) — now a
                    # SIGHTED grader (Exp 2's whole point).
                    ctrl_teacher = cbr[cfg["teacher_replan"]]
                    R = -corr_err if teacher == "front" else -ctrl_teacher
                    adv = 0.0 if base is None else (R - base)
                    base = R if base is None else cfg["outer_rho"] * base + (1 - cfg["outer_rho"]) * R
                    if adv != 0.0:
                        run_sq = adv ** 2 if run_sq is None else \
                            cfg["outer_rho"] * run_sq + (1 - cfg["outer_rho"]) * (adv ** 2)
                    if epoch >= warmup and run_sq is not None:
                        adv_n = max(-3.0, min(3.0, adv / (math.sqrt(run_sq) + 1e-6)))
                        grad = adv_n * (cur_eps / (cfg["outer_sigma"] ** 2))
                        theta = max(-4.0, min(4.0, theta + cfg["outer_alpha"] * grad))
                    outer_hist.append({"epoch": epoch, "theta": float(theta), "b_used": float(b_r),
                                       "eps": float(cur_eps), "reward": float(R), "teacher": teacher,
                                       "adv": float(adv), "b_theta": float(sigmoid(theta))})
                    print(f"  [{arm:12s} e{epoch:2d} r{rnd:3d}] b={b_r:.3f} theta={theta:+.2f} "
                          f"ctrl@{cfg['teacher_replan']}={ctrl_teacher:.4f} corr_err={corr_err:.4f} "
                          f"R={R:+.4f} adv={adv:+.4f}", flush=True)
                else:
                    cstr = " ".join(f"c{k}={cbr[k]:.3f}" for k in replan_sweep)
                    print(f"  [{arm:12s} r{rnd:3d}] b={float(b_r) if fixed_b is not None else float('nan'):.3f} "
                          f"{cstr} corr_err={corr_err:.4f}", flush=True)

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

    def final_metric(arm, key):
        r = [x[key] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    def final_control(arm, re):
        r = [x["control_by_replan"][str(re)] for x in results[arm]["records"]]
        return float(np.mean(r[-max(1, len(r) // 3):]))

    summary = {a: {"final_control_by_replan": {str(re): final_control(a, re) for re in replan_sweep},
                   "final_frontier_err": final_metric(a, "frontier_err"),
                   "final_corridor_err": final_metric(a, "corridor_err"),
                   "final_b": results[a]["final_b"],
                   "patch_frac_last": float(np.mean(results[a]["patch_frac_hist"][-10:])),
                   "noise_frac_last": float(np.mean(results[a]["noise_frac_hist"][-10:]))}
               for a in arms}

    # ---- the headline: does the control-over-b landscape gain a gradient as it goes ballistic? ----
    fixed = [a for a in arms if _parse_fixed_b(a) is not None]
    landscape = {}
    if fixed:
        fixed_sorted = sorted(fixed, key=lambda a: _parse_fixed_b(a))
        corr_vals = [summary[a]["final_corridor_err"] for a in fixed_sorted]
        corr_spread = (max(corr_vals) - min(corr_vals)) / (np.mean(corr_vals) + 1e-9)
        print(f"\n[landscape] fixed-b arms (sorted): {[_parse_fixed_b(a) for a in fixed_sorted]}", flush=True)
        print(f"[landscape] corridor-FM-err (controller-INVARIANT reference): "
              f"{[round(v,4) for v in corr_vals]}  rel_spread={corr_spread:.3f}  "
              f"argmin b={_parse_fixed_b(min(fixed_sorted, key=lambda a: summary[a]['final_corridor_err']))}", flush=True)
        for re in replan_sweep:
            cvals = [summary[a]["final_control_by_replan"][str(re)] for a in fixed_sorted]
            spread = (max(cvals) - min(cvals)) / (np.mean(cvals) + 1e-9)
            argmin_b = _parse_fixed_b(min(fixed_sorted, key=lambda a: summary[a]["final_control_by_replan"][str(re)]))
            tag = "ballistic" if re >= cfg["plan_H"] else ("reactive" if re <= 2 else "mid")
            landscape[str(re)] = {"b": [_parse_fixed_b(a) for a in fixed_sorted],
                                  "control": cvals, "rel_spread": spread, "argmin_b": argmin_b}
            print(f"[landscape] replan={re:2d} ({tag:9s}) control-over-b: {[round(v,4) for v in cvals]}  "
                  f"rel_spread={spread:.3f}  argmin b={argmin_b}", flush=True)
        print("[landscape] => spread should GROW with the commitment horizon (control becomes a "
              "sighted grader), tracking the FM-error argmin.", flush=True)

    for a in [x for x in arms if x.startswith("online")]:
        print(f"[online] {a}: learned b={summary[a]['final_b']:.3f} (teacher_replan={cfg['teacher_replan']})", flush=True)

    out = {"config": cfg, "regime": ("stationary" if cfg["drift_mode"] == "none" else "drift"),
           "noise": cfg["noise"], "noise_onset_round": onset_round, "grid_n": G,
           "cell_centers": cell_centers.tolist(), "inner": inner,
           "patch_path": [patch_center(r).tolist() for r in range(cfg["rounds"])],
           "noise_cell": int(noise_cell), "rand_dist": rand_dist, "replan_sweep": replan_sweep,
           "results": results, "summary": summary, "landscape": landscape}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "ballistic_control", cfg["tag"])
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
    arms = list(R["results"].keys())
    replan_sweep = R["replan_sweep"]
    plan_H = R["config"]["plan_H"]
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # color per commitment horizon: reactive=blue -> ballistic=red
    def re_color(re):
        if len(replan_sweep) == 1:
            return "#e8590c"
        frac = (re - min(replan_sweep)) / (max(replan_sweep) - min(replan_sweep) + 1e-9)
        return plt.cm.coolwarm(frac)

    L = R.get("landscape", {})
    # ---- fig1: the control-over-b landscape at each commitment horizon + the FM-err reference ----
    if L:
        S = R["summary"]
        fixed_sorted = sorted([a for a in arms if _parse_fixed_b(a) is not None],
                              key=lambda a: _parse_fixed_b(a))
        bvals = [_parse_fixed_b(a) for a in fixed_sorted]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for re in replan_sweep:
            cvals = L[str(re)]["control"]
            tag = "ballistic" if re >= plan_H else ("reactive" if re <= 2 else "mid")
            axes[0].plot(bvals, cvals, "o-", color=re_color(re), lw=2.2,
                         label=f"replan={re} ({tag}), spread={L[str(re)]['rel_spread']:.1%}")
        axes[0].set_xlabel("explore/exploit balance b  (0=exploit, 1=explore)")
        axes[0].set_ylabel("final control goal-dist (lower=better)")
        axes[0].set_title("control-over-b: FLAT when reactive → INTERIOR OPTIMUM when ballistic")
        axes[0].legend(fontsize=7.5)
        corr = [S[a]["final_corridor_err"] for a in fixed_sorted]
        axes[1].plot(bvals, corr, "s-", color="#2f9e44", lw=2.4, label="corridor FM err (invariant)")
        axes[1].set_xlabel("explore/exploit balance b"); axes[1].set_ylabel("corridor FM prediction err")
        axes[1].set_title("the FM-error teacher (controller-INVARIANT reference)")
        axes[1].legend(fontsize=8)
        figs["fig1_landscape.png"] = _save(fig)

        # ---- fig2 (money): spread(control-over-b) vs commitment horizon, FM-err spread reference ----
        corr_spread = (max(corr) - min(corr)) / (np.mean(corr) + 1e-9)
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        xs = list(replan_sweep); ys = [L[str(re)]["rel_spread"] for re in replan_sweep]
        ax.plot(xs, ys, "o-", color="#e8590c", lw=2.6, ms=8, label="control-over-b spread")
        ax.axhline(corr_spread, color="#2f9e44", ls="--", lw=1.8, label=f"corridor FM-err spread ({corr_spread:.1%})")
        ax.axvline(plan_H, color="#868e96", ls=":", lw=1.4, label=f"fully ballistic (plan_H={plan_H})")
        ax.set_xlabel("commitment horizon  replan_every  (steps open-loop)")
        ax.set_ylabel("rel. spread of control over b")
        ax.set_title("the FM bridge TRANSMITS to control as the controller goes ballistic")
        ax.legend(fontsize=8)
        figs["fig2_transmission.png"] = _save(fig)

    # ---- fig3: self-tuning trajectories (Exp 2), if online arms present ----
    online = [a for a in arms if a.startswith("online")]
    if online:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        ocol = {"online_front": "#e8590c", "online_ctrl": "#7048e8"}
        for a in online:
            oh = R["results"][a]["outer_hist"]
            if not oh:
                continue
            ep = [h["epoch"] for h in oh]
            lbl = ("front teacher (corridor FM err)" if a.endswith("front")
                   else f"control teacher @replan={R['config']['teacher_replan']} (ballistic)")
            ax.plot(ep, [h["b_theta"] for h in oh], "o-", color=ocol.get(a, "#9c36b5"),
                    lw=2.4, label=f"{a}: {lbl}")
        if L:
            # mark the FM-err argmin (the target both sighted teachers should chase)
            fm_argmin = None
            fixed_sorted = sorted([a for a in arms if _parse_fixed_b(a) is not None],
                                  key=lambda a: _parse_fixed_b(a))
            if fixed_sorted:
                S = R["summary"]
                fm_argmin = _parse_fixed_b(min(fixed_sorted, key=lambda a: S[a]["final_corridor_err"]))
                ax.axhline(fm_argmin, color="#2f9e44", ls="--", lw=1.4, label=f"FM-err optimum (b={fm_argmin})")
        ax.set_xlabel("outer-loop epoch"); ax.set_ylabel("explore/exploit balance b")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("self-tuning under a BALLISTIC control teacher (Exp 2)")
        ax.legend(fontsize=7.5)
        figs["fig3_selftune.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def ballistic_control(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "",
    # the commitment-horizon sweep (the new knob): reactive .. ballistic
    replan_sweep: str = "2,8,24",
    teacher_replan: int = 24,              # which commitment horizon grades `online_ctrl` (ballistic)
    # regime dials
    drift_mode: str = "morph",
    value_rel: str = "on",
    noise: bool = False,
    noise_onset: float = 0.0,
    task_geom: str = "corridor",
    corridor_r: float = 1.0,
    # outer loop (the slow reward-driven set-point tuner)
    outer_m: int = 6,
    outer_alpha: float = 0.3,
    outer_sigma: float = 0.7,
    outer_rho: float = 0.8,
    outer_warmup: int = 2,
    b_init: float = 0.5,
    # env / family (Cut #3's momentum reaching)
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
    noise_x: float = 0.0,
    noise_y: float = 0.9,
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
    # CEM-MPC control eval / reward — plan_hp defaults to plan_h (full-horizon => ballistic-capable)
    n_eval: int = 28,
    reach_r: float = 0.45,
    goal_jit: float = 0.08,
    v0_std: float = 0.3,
    plan_h: int = 24,
    plan_hp: int = 24,                     # planning horizon = full episode (fixed across the sweep)
    k_shoot: int = 256,
    cem_iters: int = 3,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    arm_list = [a for a in (arms.split(",") if arms else DEFAULT_ARMS) if a]
    replan_list = sorted({int(x) for x in replan_sweep.split(",") if x})
    if quick:
        arm_list = arms.split(",") if arms else ["b0.0", "b0.5", "b1.0"]
        replan_list = replan_list or [2, 24]
        grid_n = 5; rounds = 24; outer_m = 4; outer_warmup = 1; warmup_n = 900; seed_n = 300
        collect_n = 150; seed_steps = 200; update_steps = 90; k_ens = 3
        fm_hidden = 128; fm_layers = 2; buffer_n = 1200
        n_eval = 10; plan_h = 24; plan_hp = 24; k_shoot = 128; probe_p = 6
        tag = tag or "smoke"
    tag = tag or "default"
    if teacher_replan not in replan_list:
        replan_list = sorted(set(replan_list) | {teacher_replan})
    # guard: the plan must cover the largest commitment horizon (plan_hp >= max replan)
    plan_hp = max(plan_hp, plan_h)

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, replan_sweep=replan_list, teacher_replan=teacher_replan,
        drift_mode=drift_mode, value_rel=value_rel, noise=noise,
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
        plan_H=plan_h, plan_Hp=plan_hp, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_ballistic_control.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "ballistic_control_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
