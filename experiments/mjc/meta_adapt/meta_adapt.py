"""Cut #4 (FLOOR) — Meta-learning a fast-adapting FM initialization over a
distribution of dynamics.

Program: `ideas/two_timescale_value_loop.md` (§"Meta-RL: the two-timescale
structure"; §non-stationarity is the load-bearing part) and the pivot recorded in
`README.md`/`DIRECTED_READAPT_README.md`: stop verifying a-priori mechanistic atoms;
GET THE META PHENOMENON WORKING, then back-translate the mechanism.

    THE PHENOMENON (canonical, non-tautological form). A persistent learner on ANY
    shifting environment improves with exposure -- so "adaptation gets cheaper over a
    sequence" is near-tautological and carries no information. The falsifiable claim is
    the STANDARD meta-learning one: meta-train a fast adapter over a DISTRIBUTION of
    dynamics, then measure FEW-SHOT adaptation to HELD-OUT dynamics against a
    MULTITASK/POOLED init trained on the SAME data. That comparison is exactly the one
    RHM_META_LEARNING found COLLAPSES (meta -> plain multitask) when there is no
    structure the inner loop can't already get. Nothing guarantees it separates.

    So the object we track is the META-vs-MULTITASK GAP, treated as an ORDER PARAMETER,
    not a checkbox. At the FLOOR (this file: a smooth 1-D damping family) we EXPECT it
    near zero -- meta ~= multitask = the RHM collapse, now anchored on control. The
    science (later cuts) is finding the complexity axis (task conflict -> scarcity ->
    value-in-loop) along which the gap OPENS. This file pins the protocol + the anchor.

DESIGN (a knob-sweep-with-a-baseline; only the INITIALIZATION differs across arms).
  Task family: joint_damping in [0.05, 2.0], log-spaced. Train dampings and (disjoint,
  interior = interpolation) TEST dampings. Substrate = Cut #3's puck-free momentum
  reaching; native DGP knob, so NO env changes -> Cuts #1-3 reproducible.

  FOUR initializations, each adapted to the SAME held-out tasks by the SAME inner-loop
  SGD (identical epochs/lr/data -- only the starting weights differ):
    * META (Reptile): a learned init OPTIMIZED FOR ADAPTABILITY (first-order, textbook
      "learn an initialization" -- no context net, nothing a-priori smuggled in).
    * MULTITASK (pooled): a learned init optimized for AVERAGE performance across the
      family (one FM on the union of all train buffers). <- the load-bearing control.
    * SINGLE-TASK prior: an FM trained on ONE reference damping (Cut #3's d0 setup).
    * SCRATCH: random init. Absolute lower bound.

  Headline readout (cheap, least planner-confounded, as agreed): HELD-OUT Δs R^2 vs #
  adaptation transitions N, averaged over test tasks, per init. Secondary: CEM-MPC
  planning distance at a budget subset (behavioral grounding, Cut #3's readout).

  Deliverable = the DISSOCIATION: the adaptation curves + the META-vs-MULTITASK gap
  (fig3). NOT a leaderboard number.

Run:
    cd experiments/
    modal run mjc/meta_adapt/meta_adapt.py::meta_adapt --quick          # smoke
    modal run --detach mjc/meta_adapt/meta_adapt.py::run_meta_adapt     # (full; via entrypoint)
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_meta_adapt(cfg: dict) -> dict:
    """Self-contained (duplicates dynamics_shift.py's small helpers to keep that
    validated path untouched -- the house pattern of run_replan_ablation /
    run_directed_readapt)."""
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    H, Hp, K = cfg["H"], cfg["H_plan"], cfg["K"]
    goal_r, start_r, v0, eps = cfg["goal_range"], cfg["start_range"], cfg["v0_std"], cfg["success_eps"]
    vel_pen, n_elite, re = cfg["vel_pen"], cfg["cem_elite"], cfg["replan_every"]
    budgets = cfg["budgets"]

    # ---- task family: a 1-D task-parameter grid; disjoint INTERIOR test params ---- #
    #   family="damping"  : joint_damping in [damp_min,damp_max], log-spaced (the FLOOR;
    #                       smooth, cross-task-shared -> collapse expected).
    #   family="actuator" : command-rotation phi in [-conflict,+conflict], damping fixed
    #                       (INPUT-COUPLED conflict -> the gap should OPEN with `conflict`).
    family = cfg["family"]
    n_tr, n_te = cfg["n_train_tasks"], cfg["n_test_tasks"]
    if family == "damping":
        all_p = np.logspace(np.log10(cfg["damp_min"]), np.log10(cfg["damp_max"]), n_tr + n_te)
        ref_value = cfg["damp_max"]      # single-task prior = highest-drag (Cut #3 d0)
    elif family == "actuator":
        Phi = cfg["conflict"]
        all_p = np.linspace(-Phi, Phi, n_tr + n_te)
        ref_value = 0.0                  # single-task prior = identity actuator (no rotation)
    else:
        raise ValueError(f"unknown family {family}")
    all_p = np.sort(all_p)
    te_idx = np.unique(np.linspace(1, n_tr + n_te - 2, n_te).round().astype(int))
    tr_idx = np.array([i for i in range(n_tr + n_te) if i not in te_idx])
    train_p = all_p[tr_idx].astype(np.float64)
    test_p = all_p[te_idx].astype(np.float64)
    ref_i = int(np.argmin(np.abs(train_p - ref_value)))
    ref_param = float(train_p[ref_i])    # single-task prior task
    print(f"[family] {family} | {len(train_p)} train params: {np.round(train_p, 3)}", flush=True)
    print(f"[family] {len(test_p)} test params: {np.round(test_p, 3)}  "
          f"(single-task ref={ref_param:.3f})", flush=True)

    def task_dgp(p):
        d = dict(cfg["dgp_base"])
        if family == "damping":
            d["joint_damping"] = float(p)
        else:  # actuator: fixed baseline drag + rotate the command->motion map by p
            d["joint_damping"] = cfg["base_damping"]
            d["push_rot"] = float(p)
        return d

    # --------------------------------------------------------------------- #
    # reward-free central OU collection (Cut #3's `collect` verbatim). Central
    # exploration in a large arena -> ~no wall contact -> clean free-flight.
    # --------------------------------------------------------------------- #
    def collect(env, n_transitions, seed, ou_theta=0.2, ou_sigma=0.5):
        rng = np.random.default_rng(seed)
        ep_len, pr = cfg["collect_ep_len"], cfg["collect_pos_range"]
        n_ep = int(np.ceil(n_transitions / ep_len))
        S, U, S2, C = [], [], [], []
        for _ in range(n_ep):
            p0 = rng.uniform(-pr, pr, size=2)
            env.set_state(p0, rng.normal(0, cfg["v_explore"], size=2))
            a = np.zeros(2)
            for _ in range(ep_len):
                s = env.get_state()
                a = a - ou_theta * a + ou_sigma * rng.normal(size=2)
                u = np.clip(a, -1.0, 1.0)
                s2, info = env.step(u, fs)
                S.append(s); U.append(u.astype(np.float32)); S2.append(s2)
                C.append(info["any_contact"])
        S = np.asarray(S, np.float32)[:n_transitions]
        U = np.asarray(U, np.float32)[:n_transitions]
        S2 = np.asarray(S2, np.float32)[:n_transitions]
        C = np.asarray(C, bool)[:n_transitions]
        return S, U, S2, C

    # --------------------------------------------------------------------- #
    # forward model f(s,u) -> Δs (arity-2 SiLU MLP, Huber). ONE family-level
    # normalization (from POOLED train data) fixed everywhere -> no per-task
    # norm leak, identical I/O convention across arms/tasks/adaptation.
    # --------------------------------------------------------------------- #
    def build_fm():
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(h, 4)])).to(device)

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    def train_fm(fm, S, U, S2, norm, epochs, lr, batch=None, init_state=None):
        """Train fm from its CURRENT weights (or `init_state` if given) on (S,U,S2).
        Fresh Adam each call (standard for the Reptile inner loop + eval adaptation)."""
        if init_state is not None:
            fm.load_state_dict(init_state)
        X = torch.tensor(np.concatenate([S, U], 1), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn, Yn = (X - norm["mx"]) / norm["sx"], (Y - norm["my"]) / norm["sy"]
        opt = torch.optim.Adam(fm.parameters(), lr=lr); lossf = nn.HuberLoss(delta=1.0)
        n = Xn.shape[0]; bs = min(batch or cfg["fm_batch"], n); fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(); lossf(fm(Xn[idx]), Yn[idx]).backward()
                # grad clip: few-shot fine-tuning on a slice that happens to contain a
                # rare stiff wall-contact transition (huge Δv) otherwise blows up Adam.
                torch.nn.utils.clip_grad_norm_(fm.parameters(), cfg["grad_clip"])
                opt.step()
        fm.eval(); return fm

    # Puck-free state = [px, py, vx, vy]; the damping (the task variable) acts ONLY on
    # the velocity dims (Δpos = vel·dt is damping-independent and dominates all-dim R²),
    # so VEL_DIMS R^2 is the sensitive FM readout; all-dim is kept for reference.
    VEL_DIMS = [2, 3]

    def fm_free_r2(fm, S, U, S2, norm, C=None, dims=None):
        if C is not None:
            ff = ~C; S, U, S2 = S[ff], U[ff], S2[ff]
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1), device=device)
            Xn = (X - norm["mx"]) / norm["sx"]
            pred = (fm(Xn) * norm["sy"] + norm["my"]).cpu().numpy()
        yt = (S2 - S).astype(np.float32)
        if dims is not None:
            yt, pred = yt[:, dims], pred[:, dims]
        ss_res = ((yt - pred) ** 2).sum()
        ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    # --------------------------------------------------------------------- #
    # CEM MPC (value = running distance-to-goal + terminal-vel penalty; dynamics-
    # INDEPENDENT + FIXED -- the reused planner factor). Behavioral grounding only.
    # --------------------------------------------------------------------- #
    def mpc_action(fm, norm, states, goals, rng):
        B = states.shape[0]
        mu = np.zeros((B, Hp, 2), np.float32); sig = np.full((B, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(K, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(K, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((B, K, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(B * K, Hp, 2), device=device)
                cost = torch.zeros(B * K, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (fm((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                    cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(B, K), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_mpc(env, fm, norm, seed, starts, goals):
        rng = np.random.default_rng(seed)
        states = starts.copy(); plan = None
        for step in range(H):
            if step % re == 0:
                plan = mpc_action(fm, norm, states, goals, rng)
            acts = plan[:, step % re, :]
            for b in range(starts.shape[0]):
                env.set_state(states[b, :2], states[b, 2:]); s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - goals, axis=1)
        return {"mean_dist": float(fd.mean()), "success": float((fd < eps).mean())}

    # fixed eval set (identical goals/starts everywhere -> controlled)
    ev_rng = np.random.default_rng(cfg["seed"] + 7)
    npl = cfg["n_eval_plan"]
    ev_starts = np.concatenate([ev_rng.uniform(-start_r, start_r, (npl, 2)),
                                (ev_rng.normal(0, v0, (npl, 2)) if v0 > 0
                                 else np.zeros((npl, 2)))], 1).astype(np.float32)
    ev_goals = ev_rng.uniform(-goal_r, goal_r, (npl, 2)).astype(np.float32)

    # ===================================================================== #
    # 1. collect per-task buffers (train + test) + family normalization
    # ===================================================================== #
    print(f"\n[1] collecting per-task buffers ...", flush=True)
    train_bufs = []
    for i, p in enumerate(train_p):
        env = PusherEnv(task_dgp(p), with_puck=False)
        train_bufs.append(collect(env, cfg["train_task_N"], cfg["seed"] + 1000 + i))
    # test: THREE disjoint slices per task ->
    #   adapt  = first n_adapt_pool transitions  (the few-shot adaptation pool, first N)
    #   oracle = large middle chunk              (a TRUE per-task ceiling, not data-starved)
    #   hold   = last n_holdout                  (held-out eval, disjoint from both)
    nH, nA = cfg["n_holdout"], cfg["n_adapt_pool"]
    test_bufs, test_envs = [], []
    for i, p in enumerate(test_p):
        env = PusherEnv(task_dgp(p), with_puck=False)
        test_envs.append(env)
        S, U, S2, C = collect(env, cfg["test_task_N"], cfg["seed"] + 2000 + i)
        adapt = (S[:nA], U[:nA], S2[:nA], C[:nA])
        oracle = (S[nA:-nH], U[nA:-nH], S2[nA:-nH], C[nA:-nH])
        hold = (S[-nH:], U[-nH:], S2[-nH:], C[-nH:])
        test_bufs.append({"adapt": adapt, "oracle": oracle, "hold": hold})

    # pooled family normalization (from ALL train buffers)
    allS = np.concatenate([b[0] for b in train_bufs])
    allU = np.concatenate([b[1] for b in train_bufs])
    allS2 = np.concatenate([b[2] for b in train_bufs])
    norm = make_norm(allS, allU, allS2)
    print(f"[1] pooled train transitions={len(allS)}  "
          f"contact_frac={np.concatenate([b[3] for b in train_bufs]).mean():.3f}", flush=True)

    # ===================================================================== #
    # 2. per-test-task ORACLE ceiling (fresh FM fully fit on the task's pool)
    # ===================================================================== #
    print(f"\n[2] per-test-task oracle ceilings ...", flush=True)
    oracle_r2, oracle_r2_all = [], []
    for i, tb in enumerate(test_bufs):
        S, U, S2, C = tb["oracle"]
        fm_o = train_fm(build_fm(), S, U, S2, norm, cfg["fm_epochs"], cfg["fm_lr"])
        r2 = fm_free_r2(fm_o, *tb["hold"][:3], norm, tb["hold"][3], dims=VEL_DIMS)
        r2a = fm_free_r2(fm_o, *tb["hold"][:3], norm, tb["hold"][3])
        oracle_r2.append(r2); oracle_r2_all.append(r2a)
        print(f"    task {i} param={test_p[i]:.3f}: oracle held-out VEL R^2={r2:.4f} "
              f"(all-dim {r2a:.4f}, pool={len(S)})", flush=True)
    oracle_mean = float(np.mean(oracle_r2))

    # ===================================================================== #
    # 3. build the FOUR initializations
    # ===================================================================== #
    # SCRATCH: random init (seeded via build_fm's fresh params)
    init_states = {}
    init_states["scratch"] = {k: v.detach().clone() for k, v in build_fm().state_dict().items()}

    # SINGLE-TASK prior: FM on the reference train task only (ref_i chosen above)
    print(f"\n[3] training single-task / multitask / meta initializations ...", flush=True)
    Sr, Ur, S2r, _ = train_bufs[ref_i]
    fm_single = train_fm(build_fm(), Sr, Ur, S2r, norm, cfg["fm_epochs"], cfg["fm_lr"])
    init_states["single"] = {k: v.detach().clone() for k, v in fm_single.state_dict().items()}

    # MULTITASK / pooled: one FM on the UNION of all train buffers (the load-bearing
    # control -- same data as meta, no inner/outer structure)
    fm_multi = train_fm(build_fm(), allS, allU, allS2, norm, cfg["fm_epochs"], cfg["fm_lr"])
    init_states["multitask"] = {k: v.detach().clone() for k, v in fm_multi.state_dict().items()}

    # META (Reptile): first-order "learn an initialization". Inner loop = the SAME
    # (adapt_epochs, adapt_lr) used at eval, on a randomly-sized buffer sampled from the
    # eval budgets (so the init is robust across the whole few-shot sweep, not tuned to
    # one N). Outer step: theta <- theta + meta_lr * mean_over_tasks(theta' - theta).
    rng = np.random.default_rng(cfg["seed"] + 5)
    inner_Ns = [b for b in budgets if b > 0]
    meta = build_fm()
    if cfg["meta_warm_start"]:
        # warm-start from the POOLED init: meta = multitask + adaptability refinement, so
        # the meta-multitask gap isolates purely the adaptability objective (and it removes
        # the from-scratch Reptile convergence confound).
        meta.load_state_dict(init_states["multitask"])
    inner = build_fm()
    for it in range(cfg["meta_iters"]):
        theta0 = {k: v.detach().clone() for k, v in meta.state_dict().items()}
        accum = {k: torch.zeros_like(v) for k, v in theta0.items()}
        tids = rng.choice(len(train_bufs), size=cfg["meta_task_batch"], replace=False)
        for tid in tids:
            N = int(rng.choice(inner_Ns))
            S, U, S2, _ = train_bufs[tid]
            inner.load_state_dict(theta0)
            train_fm(inner, S[:N], U[:N], S2[:N], norm, cfg["adapt_epochs"], cfg["adapt_lr"])
            for k, v in inner.state_dict().items():
                accum[k] += (v.detach() - theta0[k])
        eps_meta = cfg["meta_lr"] * (1.0 - it / cfg["meta_iters"])   # linear decay -> stable
        with torch.no_grad():
            new_state = {k: theta0[k] + eps_meta * accum[k] / len(tids) for k in theta0}
        meta.load_state_dict(new_state)
        if (it + 1) % max(1, cfg["meta_iters"] // 8) == 0:
            # cheap progress probe: mean adapted held-out R^2 at a mid budget
            Nprobe = inner_Ns[len(inner_Ns) // 2]
            r2s = []
            for tb in test_bufs:
                fm_p = build_fm(); fm_p.load_state_dict(meta.state_dict())
                S, U, S2, _ = tb["adapt"]
                train_fm(fm_p, S[:Nprobe], U[:Nprobe], S2[:Nprobe], norm,
                         cfg["adapt_epochs"], cfg["adapt_lr"])
                r2s.append(fm_free_r2(fm_p, *tb["hold"][:3], norm, tb["hold"][3], dims=VEL_DIMS))
            print(f"    [reptile] iter {it+1}/{cfg['meta_iters']}: "
                  f"mean adapted VEL R^2 @N={Nprobe} = {np.mean(r2s):.4f}", flush=True)
    init_states["meta"] = {k: v.detach().clone() for k, v in meta.state_dict().items()}

    # ===================================================================== #
    # 4. ADAPTATION SWEEP: each init adapted to each test task at each budget N
    #    by the IDENTICAL inner loop. Only the initialization differs.
    # ===================================================================== #
    print(f"\n[4] adaptation sweep (budgets={budgets}) ...", flush=True)
    arm_names = ["meta", "multitask", "single", "scratch"]
    plan_budgets = set(cfg["plan_budgets"]) if cfg["eval_planning"] else set()

    def _safe_nanmean(A):   # per-column mean; all-NaN column -> NaN, no warning
        return [float(np.nanmean(A[:, j])) if np.any(~np.isnan(A[:, j])) else float("nan")
                for j in range(A.shape[1])]

    def _safe_nanstd(A):
        return [float(np.nanstd(A[:, j])) if np.any(~np.isnan(A[:, j])) else float("nan")
                for j in range(A.shape[1])]

    def _safe_nanmedian(A):
        return [float(np.nanmedian(A[:, j])) if np.any(~np.isnan(A[:, j])) else float("nan")
                for j in range(A.shape[1])]

    arms = {}
    for name in arm_names:
        r2_per_task = np.full((len(test_bufs), len(budgets)), np.nan)    # VEL R^2 (primary)
        r2all_per_task = np.full((len(test_bufs), len(budgets)), np.nan)  # all-dim (reference)
        plan_per_task = np.full((len(test_bufs), len(budgets)), np.nan)
        for ti, tb in enumerate(test_bufs):
            Sp, Up, S2p, _ = tb["adapt"]
            for bi, N in enumerate(budgets):
                fm = build_fm()
                if N == 0:
                    fm.load_state_dict(init_states[name])      # raw init, no adaptation
                else:
                    train_fm(fm, Sp[:N], Up[:N], S2p[:N], norm, cfg["adapt_epochs"],
                             cfg["adapt_lr"], init_state=init_states[name])
                r2_per_task[ti, bi] = fm_free_r2(fm, *tb["hold"][:3], norm, tb["hold"][3], dims=VEL_DIMS)
                r2all_per_task[ti, bi] = fm_free_r2(fm, *tb["hold"][:3], norm, tb["hold"][3])
                if N in plan_budgets:
                    plan_per_task[ti, bi] = eval_mpc(
                        test_envs[ti], fm, norm, cfg["seed"] + 300 + N, ev_starts, ev_goals)["mean_dist"]
        with np.errstate(invalid="ignore"):
            arms[name] = {
                "budgets": list(budgets),
                "r2_median": _safe_nanmedian(r2_per_task),   # robust headline
                "r2_q25": [float(np.nanpercentile(r2_per_task[:, j], 25)) for j in range(len(budgets))],
                "r2_q75": [float(np.nanpercentile(r2_per_task[:, j], 75)) for j in range(len(budgets))],
                "r2_mean": np.nanmean(r2_per_task, 0).tolist(),
                "r2_per_task": r2_per_task.tolist(),
                "r2all_mean": np.nanmean(r2all_per_task, 0).tolist(),
                "plan_median": _safe_nanmedian(plan_per_task),
                "plan_mean": _safe_nanmean(plan_per_task),
            }
        # transitions-to-adapt: smallest N>0 with MEDIAN VEL R^2 >= adapt_frac * oracle
        thr = oracle_mean * cfg["adapt_frac"]
        adapt_N = next((N for N, v in zip(budgets, arms[name]["r2_median"])
                        if N > 0 and v >= thr), None)
        arms[name]["adapt_N"] = adapt_N
        print(f"    [{name:9s}] VEL R^2 (median): " + "  ".join(
            f"N{N}={v:.3f}" for N, v in zip(budgets, arms[name]["r2_median"]))
            + f"   -> adapt@{adapt_N} (thr={thr:.3f})", flush=True)

    # the ORDER PARAMETER: per-task (meta - multitask) VEL R^2 gap, median over tasks
    meta_pt = np.asarray(arms["meta"]["r2_per_task"])
    multi_pt = np.asarray(arms["multitask"]["r2_per_task"])
    with np.errstate(invalid="ignore"):
        gap_mean = np.nanmedian(meta_pt - multi_pt, 0).tolist()
        gap_std = np.nanstd(meta_pt - multi_pt, 0).tolist()
    # behavioral order parameter: multitask - meta PLANNING distance (>0 = meta plans better)
    plan_gap = [float(m - e) if not (np.isnan(m) or np.isnan(e)) else float("nan")
                for e, m in zip(arms["meta"]["plan_median"], arms["multitask"]["plan_median"])]

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    results = {
        "config": cfg,
        "family": family, "conflict": cfg["conflict"],
        "train_params": train_p.tolist(), "test_params": test_p.tolist(),
        "ref_param": ref_param,
        "oracle_r2_per_task": oracle_r2, "oracle_mean": oracle_mean,
        "oracle_r2all_per_task": oracle_r2_all,
        "adapt_threshold": oracle_mean * cfg["adapt_frac"],
        "budgets": list(budgets), "arms": arms,
        "meta_minus_multitask_gap": {"mean": gap_mean, "std": gap_std},
        "plan_gap_multi_minus_meta": plan_gap,
        "plan_budgets": sorted(plan_budgets),
    }

    print(f"\n===== META-ADAPT SUMMARY (family={family}, conflict={cfg['conflict']:.3f}) =====",
          flush=True)
    print(f"oracle mean held-out R^2 = {oracle_mean:.4f}  (adapt thr {oracle_mean*cfg['adapt_frac']:.4f})",
          flush=True)
    for name in arm_names:
        print(f"  {name:9s}: adapt@{arms[name]['adapt_N']}  "
              f"(median R^2 @maxN={arms[name]['r2_median'][-1]:.4f}, "
              f"zero-shot={arms[name]['r2_median'][0]:.4f})", flush=True)
    print(f"META - MULTITASK VEL-R^2 gap (order param, median): " + "  ".join(
        f"N{N}={g:+.3f}" for N, g in zip(budgets, gap_mean)), flush=True)
    if cfg["eval_planning"]:
        print(f"MULTITASK - META planning gap (>0=meta better): " + "  ".join(
            f"N{N}={g:+.3f}" for N, g in zip(budgets, plan_gap) if not np.isnan(g)), flush=True)

    figures = _make_meta_figures(results)

    outdir = os.path.join(DATA_DIR, "meta_adapt", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_meta_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"meta": "#3d6fd1", "multitask": "#d1603d", "single": "#8452c9", "scratch": "#888888"}
    MRK = {"meta": "o-", "multitask": "s-", "single": "^-", "scratch": "d--"}
    C_ORC = "#2f9e44"
    figs = {}
    budgets = R["budgets"]
    xN = [max(N, 1) for N in budgets]          # N=0 plotted at x=1 (log axis)
    arms = R["arms"]
    oracle = R["oracle_mean"]

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: adaptation curves (held-out R^2 vs N), per init, +/- std over tasks ----
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for name in ["scratch", "single", "multitask", "meta"]:
        m = np.asarray(arms[name]["r2_median"])
        q25, q75 = np.asarray(arms[name]["r2_q25"]), np.asarray(arms[name]["r2_q75"])
        ax.plot(xN, m, MRK[name], color=COL[name], lw=2.1,
                label=f"{name} (adapt@{arms[name]['adapt_N']})")
        ax.fill_between(xN, q25, q75, color=COL[name], alpha=0.12)
    ax.axhline(oracle, color=C_ORC, ls="-", lw=1.3, label=f"oracle ceiling {oracle:.3f}")
    ax.axhline(R["adapt_threshold"], color=C_ORC, ls="--", lw=1.0,
               label=f"adapt threshold {R['adapt_threshold']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("post-shift adaptation transitions N  (N=0 = raw init, no fine-tune)")
    ax.set_ylabel("held-out velocity-dim Δs R^2  (higher = better)")
    ax.set_title("Few-shot adaptation to HELD-OUT dynamics, by initialization\n"
                 "(velocity dims carry the task variation; only the init differs)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    figs["fig1_adaptation_curves.png"] = _save(fig)

    # ---- fig2: transitions-to-adapt bars ----
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    names = ["meta", "multitask", "single", "scratch"]
    maxb = max(budgets)
    heights = [arms[n]["adapt_N"] if arms[n]["adapt_N"] is not None else maxb for n in names]
    reached = [arms[n]["adapt_N"] is not None for n in names]
    xs = np.arange(len(names))
    ax.bar(xs, heights, color=[COL[n] for n in names], alpha=0.85,
           hatch=["" if r else "//" for r in reached])
    for xi, h, r, n in zip(xs, heights, reached, names):
        ax.text(xi, h, (f"{arms[n]['adapt_N']}" if r else f"≥{maxb}"),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel(f"transitions to reach {R['config']['adapt_frac']:g}×oracle R^2")
    ax.set_ylim(0, maxb * 1.2)
    ax.set_title("Transitions-to-adapt per initialization (lower = faster learner)", fontsize=10)
    figs["fig2_transitions_to_adapt.png"] = _save(fig)

    # ---- fig3: the ORDER PARAMETER -- META minus MULTITASK R^2 gap vs N ----
    gap = R["meta_minus_multitask_gap"]
    gm, gs = np.asarray(gap["mean"]), np.asarray(gap["std"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0, color="#444", lw=1.2)
    ax.plot(xN, gm, "o-", color="#3d6fd1", lw=2.3)
    ax.fill_between(xN, gm - gs, gm + gs, color="#3d6fd1", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("post-shift adaptation transitions N")
    ax.set_ylabel("velocity-dim R^2(meta) − R^2(multitask)")
    fam = R.get("family", "damping")
    sub = ("FLOOR (damping): expectation ≈ 0 = RHM collapse anchor"
           if fam == "damping" else
           f"CONFLICT (actuator, Φ={R.get('conflict', 0):.2f} rad): does the gap open?")
    ax.set_title("The order parameter: does learn-to-adapt beat pooling?\n" + sub, fontsize=10)
    figs["fig3_meta_vs_multitask_gap.png"] = _save(fig)

    # ---- fig4 (optional): planning distance vs N, behavioral grounding ----
    if R["plan_budgets"]:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        pbs = R["plan_budgets"]
        for name in ["scratch", "single", "multitask", "meta"]:
            pm = np.asarray(arms[name]["plan_median"])
            xs = [max(N, 1) for N, v in zip(budgets, pm) if not np.isnan(v)]
            ys = [v for v in pm if not np.isnan(v)]
            if ys:
                ax.plot(xs, ys, MRK[name], color=COL[name], lw=2.1, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("post-shift adaptation transitions N")
        ax.set_ylabel("CEM-MPC final ||pusher_pos - goal||  (lower = better)")
        ax.set_title("Behavioral grounding: planning distance vs adaptation budget", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper right")
        figs["fig4_planning.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_adapt(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # env / task (Cut #3's momentum-dominated free-flight reaching)
    frame_skip: int = 12,
    horizon: int = 24,
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    success_eps: float = 0.1,
    gear: float = 10.0,
    arena_half: float = 2.0,
    # task family selector
    family: str = "damping",           # "damping" (the FLOOR) | "actuator" (conflict)
    conflict: float = 3.14159,         # actuator: phi half-range Phi (rad); pi = max conflict
    base_damping: float = 1.0,         # actuator: fixed drag baseline while phi varies
    # damping-family grid: 1-D joint_damping in [damp_min, damp_max] (log-spaced)
    damp_min: float = 0.05,
    damp_max: float = 2.0,
    n_train_tasks: int = 16,
    n_test_tasks: int = 6,
    train_task_n: int = 1500,
    test_task_n: int = 6000,
    n_adapt_pool: int = 512,
    n_holdout: int = 1500,
    v_explore: float = 1.2,
    collect_ep_len: int = 20,
    collect_pos_range: float = 1.0,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_epochs: int = 40,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    # inner-loop adaptation (SHARED by Reptile inner loop AND eval adaptation)
    adapt_epochs: int = 40,
    adapt_lr: float = 5e-4,
    adapt_frac: float = 0.9,
    grad_clip: float = 1.0,
    # Reptile (first-order meta-init), warm-started from the pooled/multitask init
    meta_iters: int = 200,
    meta_task_batch: int = 4,
    meta_lr: float = 0.1,
    meta_warm_start: bool = True,
    # MPC (behavioral grounding only; value fixed across tasks)
    k_shoot: int = 256,
    h_plan: int = 15,
    cem_iters: int = 3,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    replan_every: int = 8,
    eval_planning: bool = True,
    n_eval_plan: int = 48,
):
    import os

    budgets = [0, 5, 10, 20, 40, 80, 160, 320]
    plan_budgets = list(budgets)   # planning is the sensitive readout for this easy family

    if quick:
        n_train_tasks, n_test_tasks = 4, 2
        train_task_n, test_task_n, n_adapt_pool, n_holdout = 600, 1600, 400, 500
        fm_hidden, fm_layers, fm_epochs = 128, 2, 15
        adapt_epochs = 25
        meta_iters, meta_task_batch = 40, 3
        budgets = [0, 20, 80, 320]
        plan_budgets = [0, 320]
        n_eval_plan = 16
        k_shoot, h_plan, horizon = 96, 12, 26
        cem_elite = 12
        replan_every = min(replan_every, h_plan)
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed,
        frame_skip=frame_skip, H=horizon, goal_range=goal_range, start_range=start_range,
        v0_std=v0_std, success_eps=success_eps, v_explore=v_explore,
        collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0, gear=gear),
        family=family, conflict=conflict, base_damping=base_damping,
        damp_min=damp_min, damp_max=damp_max,
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks,
        train_task_N=train_task_n, test_task_N=test_task_n,
        n_adapt_pool=n_adapt_pool, n_holdout=n_holdout,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_epochs=fm_epochs, fm_lr=fm_lr,
        fm_batch=fm_batch,
        adapt_epochs=adapt_epochs, adapt_lr=adapt_lr, adapt_frac=adapt_frac,
        grad_clip=grad_clip,
        meta_iters=meta_iters, meta_task_batch=meta_task_batch, meta_lr=meta_lr,
        meta_warm_start=meta_warm_start,
        K=k_shoot, H_plan=h_plan, vel_pen=vel_pen, replan_every=replan_every,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma,
        eval_planning=eval_planning, n_eval_plan=n_eval_plan,
        budgets=budgets, plan_budgets=plan_budgets,
    )
    out = run_meta_adapt.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_adapt_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
