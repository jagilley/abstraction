"""Two clocks: circuit-matched eligibility windows for a fast performance-error
channel and a slow reward channel -- idea doc S7 / S8 Experiment 4, the operational
form of S6's separate-channel bet.

Program: `ideas/performance_error_is_the_bridge.md`. The two predictions, under
delayed reward with fast FM feedback:
  (i)  performance peaks when the fast channel's eligibility window matches the
       circuit's own feedback delay, tau_fast ~= d_fm (Suvrathan et al. 2016:
       flocculus PF-Purkinje plasticity tuned to ~120 ms = that circuit's error
       return delay; the peak should TRACK the delay);
  (ii) a TWO-CHANNEL architecture -- a fast performance-error channel with its own
       fast trace plus a slow reward channel with a slow trace -- beats a single
       SHARED-WINDOW channel at matched capacity (Gadagkar et al. 2016: performance
       error lives in a separate projection-defined dopaminergic population with
       51-58 ms latencies, not on the reward channel's 0.3-2 s trace [Yagishita]).
       This is the novel one: if performance error only helps by displacing or
       monopolizing the reward channel, shared should do just as well (Parvin).

    THE MECHANIC (what "clock" means here). Teaching signals are ANONYMOUS SCALARS
    at arrival time -- nothing tags them to the sample that earned them. Credit
    reaches samples only through an eligibility kernel over the recent stream:

        signal v arrives at time a  ->  credit[t] += v * K_tau(a - t),  a-t <= A_max
        K_tau(age) = unit-area Gaussian peaked at age tau (width max(0.35*tau, 1))

    Unit area = temporal capacity conservation: a window cannot integrate more
    total credit, only place it. A broad window is diluted, a mismatched window is
    misassigned, a matched window concentrates credit on the sample that earned it.

    THE TWO SIGNALS (identical streams for every arm; consumption differs):
      * fast, dense, self-supervised: delta_t = (b(s_t) - e_t) * gate(g_t), the S13
        canonical performance error (context-conditional benchmark net b(s),
        centered agency gate), computed from the ARM'S OWN forward model as of
        prediction time (the Smith-held prediction) and broadcast d_fm steps later
        (the sensory return delay). One event per transition.
      * slow, sparse, extrinsic: trials are scripted ballistic reaches planned with
        the PRE-DRIFT FM and executed open-loop (frozen dysmetria through the
        on-path drift region -- arm-independent, so the stream stays identical);
        reward advantage = habituating per-family benchmark minus outcome, arrives
        d_r steps after trial end. One event per L+E transitions. Reward knows
        VALUE-RELEVANCE (only on-path failures fire it); delta knows PRECISION
        (which transition was mispredicted) -- non-redundant by construction.

    THE ARMS -- the minimal contrast. Every arm consumes its accumulated credit as
    a per-sample plasticity gain at MATCHED AVERAGE WEIGHT (running-mean
    normalized, clipped; META_ADAPT #4e budget discipline):

        w = exp( - sum_ledgers  clip(credit_l / kappa_l, +-8) ),  capped, normalized

    so the ONLY difference between two_channel and shared_* is WHERE THE SUM
    HAPPENS RELATIVE TO THE WINDOWING: shared sums both signals into ONE kernel;
    two_channel gives each signal its own matched kernel and sums in log-weight
    space. Same signals, same budget, same FM, same data, same combination rule.
      fixed            w = 1 (ungated floor)
      fast:tX:dY       delta only, kernel at tau=X, feedback delay d_fm=Y
      slow:TX          reward only, kernel at tau=X
      two:tX:dY:TZ     both, separate kernels (the Gadagkar architecture)
      shared:tX:dY     both summed into one kernel at tau=X (sweep X; optional
                       sM = width override for the broad variant)

    READOUTS (dense; none require an outer loop to converge -- the teacher_snr
    lesson): (a) credit fidelity: corr(applied credit, oracle credit) per channel
    -- a pure signal readout, and the offset curve for (i) comes free from the
    logged series; (b) FM probe error by region class (R1-on / R2-off / noise /
    base) along re-adaptation; (c) allocation: where the plasticity budget went;
    (d) retention (base-probe drift); (e) ballistic control recovery on the
    value task (family-a reaches through R1) at milestone snapshots, with clean
    family-b reaches + reactive control as it-should-not-matter contrasts.

Run:
    cd experiments/
    modal run mjc/two_clocks/two_clocks.py::two_clocks --quick            # smoke
    modal run --detach mjc/two_clocks/two_clocks.py::two_clocks --tag main_s0 --seed 0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


def parse_arm(spec: str) -> dict:
    """'two:t4:d4:T24' -> {name, kind, d_fm, ledgers:[{ch, tau, sig}]}.
    Keys: t = fast/shared tau, s = its width override, d = d_fm,
          T = slow tau, S = slow width override."""
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    kind = parts[0]
    p = {}
    for tok in parts[1:]:
        p[tok[0]] = float(tok[1:])

    def width(tau, override):
        return float(override) if override is not None else max(0.35 * tau, 1.0)

    ledgers = []
    if kind == "fast":
        ledgers = [dict(ch="f", tau=p["t"], sig=width(p["t"], p.get("s")))]
    elif kind == "slow":
        ledgers = [dict(ch="r", tau=p["T"], sig=width(p["T"], p.get("S")))]
    elif kind == "two":
        ledgers = [dict(ch="f", tau=p["t"], sig=width(p["t"], p.get("s"))),
                   dict(ch="r", tau=p["T"], sig=width(p["T"], p.get("S")))]
    elif kind == "shared":
        ledgers = [dict(ch="fr", tau=p["t"], sig=width(p["t"], p.get("s")))]
    elif kind != "fixed":
        raise ValueError(f"unknown arm kind {kind!r} in {spec!r}")
    return dict(name=spec, kind=kind, d_fm=int(p.get("d", 4)), ledgers=ledgers)


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_two_clocks(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; NK = len(REG)
    T = cfg["T"]; B = cfg["batch"]; A_max = cfg["a_max"]
    L = cfg["trial_len"]; E = cfg["explore_len"]
    H_ev = cfg["plan_H"]
    lag_cycles = A_max // B
    n_cycles = T // B
    n_trials = T // (L + E)
    lam_r = cfg["lam_r"] if cfg["lam_r"] > 0 else float(L + E)
    milestone_set = set(cfg["milestones"])
    arms = [parse_arm(a) for a in cfg["arms"]]
    print(f"[setup] device={device} T={T} B={B} A_max={A_max} trial L={L} explore E={E} "
          f"n_trials={n_trials} d_r={cfg['d_r']} lam_r={lam_r}", flush=True)
    print("[setup] arms: " + " | ".join(a["name"] for a in arms), flush=True)
    print("[setup] regions: " + "; ".join(
        f"{r['name']}@({r['center'][0]:+.2f},{r['center'][1]:+.2f}) phi={r['phi']:+.2f} "
        f"noise={r['noise']:.0f}" for r in REG), flush=True)

    def make_env(drifted: bool):
        d = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                 joint_damping=cfg["damping"], pusher_r=0.12,
                 noise_seed=cfg["seed"] + 999)
        if drifted:
            d["rot_regions"] = [dict(center=tuple(r["center"]), sigma=r["sigma"],
                                     phi=float(r["phi"]), noise=float(r["noise"])) for r in REG]
        return PusherEnv(d, with_puck=False)

    env0 = make_env(False)          # clean world: pretraining + trial-outcome baselines
    env1 = make_env(True)           # drifted world: the stream, probes, control eval

    # ---------------- nets (plasticity_gain idioms) ----------------
    def _mlp(seed, din, dout, h, Ln):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(Ln - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, dout)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    # ---------------- collection ----------------
    def collect_at(cenv, pos, rng):
        n = len(pos)
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        for i in range(n):
            vel = rng.normal(0, cfg["v_explore"], 2).astype(np.float32)
            cenv.set_state(pos[i].astype(np.float64), vel.astype(np.float64))
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            s = cenv.get_state(); s2, _ = cenv.step(u, fs)
            S[i] = s; U[i] = u; S2[i] = s2
        return S, U, S2

    def collect_box(cenv, n, rng):
        pos = np.stack([rng.uniform(-cfg["box_x"], cfg["box_x"], n),
                        rng.uniform(-cfg["box_y"], cfg["box_y"], n)], 1).astype(np.float32)
        return collect_at(cenv, pos, rng)

    def collect_region(cenv, j, n, rng):
        c = REG[j]["center"]; s = REG[j]["sigma"] * cfg["collect_sigma_frac"]
        pos = np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1).astype(np.float32)
        return collect_at(cenv, pos, rng)

    def gates(pos):
        w = np.empty((len(pos), NK), np.float32)
        for j, r in enumerate(REG):
            d2 = (pos[:, 0] - r["center"][0]) ** 2 + (pos[:, 1] - r["center"][1]) ** 2
            w[:, j] = np.exp(-d2 / (2.0 * r["sigma"] ** 2))
        return w

    def classify(S):
        w = gates(S)
        return np.where(w.max(1) > 0.3, w.argmax(1), NK)

    CLS_NAMES = [r["name"] for r in REG] + ["base"]

    # ---------------- normalization (clean pool) ----------------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)
    huber_ps = nn.HuberLoss(delta=1.0, reduction="none")

    def train_steps(net, opt, S, U, S2, steps, brng, bs_cap=512):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (Xt - norm["mx"]) / norm["sx"]; Yn = (Yt - norm["my"]) / norm["sy"]
        bs = min(bs_cap, len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fm2_delta(net, S, U):
        with torch.no_grad():
            Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            return (net((Xt - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def fm1_delta(net, S):
        with torch.no_grad():
            Xt = torch.tensor(S, device=device, dtype=torch.float32)
            return (net((Xt - norm["mx"][:4]) / norm["sx"][:4]) * norm["sy"] + norm["my"]).cpu().numpy()

    def per_sample_err(net, S, U, S2):
        return np.linalg.norm(fm2_delta(net, S, U) - (S2 - S), axis=1).astype(np.float32)

    # ---------------- benchmark net b(s) ----------------
    def make_bench(seed):
        return _mlp(seed, 4, 1, cfg["bench_hidden"], cfg["bench_layers"])

    def bench_pred(bnet, S):
        with torch.no_grad():
            Xt = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
            return bnet(Xt).squeeze(-1).cpu().numpy().astype(np.float32)

    def bench_step(bnet, bopt, S, e):
        Xt = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
        t = torch.tensor(e, device=device, dtype=torch.float32)
        bnet.train(); bopt.zero_grad()
        huber(bnet(Xt).squeeze(-1), t).backward(); bopt.step(); bnet.eval()

    # ===================================================================== #
    # 1) PRETRAIN on the clean world: FM2, FM1, benchmark; calibrate scales
    # ===================================================================== #
    fm2_0 = _mlp(cfg["seed"] + 40, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    opt2 = torch.optim.Adam(fm2_0.parameters(), lr=cfg["fm_lr"])
    train_steps(fm2_0, opt2, nS, nU, nS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))

    fm1_0 = _mlp(cfg["seed"] + 41, 4, 4, cfg["fm_hidden"], cfg["fm_layers"])
    X1 = torch.tensor(nS, device=device, dtype=torch.float32)
    Y1 = torch.tensor(Y, device=device)
    X1n = (X1 - norm["mx"][:4]) / norm["sx"][:4]; Y1n = (Y1 - norm["my"]) / norm["sy"]
    opt1 = torch.optim.Adam(fm1_0.parameters(), lr=cfg["fm_lr"])
    brng1 = np.random.default_rng(cfg["seed"] + 301); fm1_0.train()
    for _ in range(cfg["fm_steps"]):
        idx = torch.tensor(brng1.integers(0, len(nS), size=512), device=device)
        opt1.zero_grad(); huber(fm1_0(X1n[idx]), Y1n[idx]).backward(); opt1.step()
    fm1_0.eval()

    e_pool = per_sample_err(fm2_0, nS, nU, nS2)
    bench_0 = make_bench(cfg["seed"] + 42)
    bopt_0 = torch.optim.Adam(bench_0.parameters(), lr=cfg["bench_lr"])
    brngb = np.random.default_rng(cfg["seed"] + 302)
    for _ in range(cfg["bench_pretrain_steps"]):
        idx = brngb.integers(0, len(nS), size=256)
        bench_step(bench_0, bopt_0, nS[idx], e_pool[idx])

    # calibration: kappa_f from clean residuals (plasticity_gain's tau); centered
    # agency gate g0/theta from active vs passive medians (agency_gate's recipe)
    cS, cU, cS2 = collect_box(env0, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    e_cal = per_sample_err(fm2_0, cS, cU, cS2)
    b_cal = bench_pred(bench_0, cS)
    resid = b_cal - e_cal
    kappa_f = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)
    g_act = np.linalg.norm(fm2_delta(fm2_0, cS, cU) - fm1_delta(fm1_0, cS), axis=1)
    g_pas = np.linalg.norm(fm2_delta(fm2_0, cS, np.zeros_like(cU)) - fm1_delta(fm1_0, cS), axis=1)
    g0 = 0.5 * (float(np.median(g_act)) + float(np.median(g_pas)))
    theta = max((float(np.median(g_act)) - float(np.median(g_pas))) / 8.0, 1e-6)
    print(f"[calib] fm_err(clean)={e_cal.mean():.4f} kappa_f={kappa_f:.4f} "
          f"g_act_med={np.median(g_act):.4f} g_pas_med={np.median(g_pas):.4f} "
          f"g0={g0:.4f} theta={theta:.5f}", flush=True)

    # ===================================================================== #
    # 2) CEM planner (used for scripted trials AND milestone control grading)
    # ===================================================================== #
    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]

    def mpc_plan(net, states, goals, rng, H):
        Bn = states.shape[0]
        mu = np.zeros((Bn, H, 2), np.float32); sig = np.full((Bn, H, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            eps = rng.standard_normal((Bn, Kc, H, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * eps, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, H, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(H):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                    s = s + d; cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    # ===================================================================== #
    # 3) THE STREAM: scripted trials (pre-drift-FM ballistic reaches, frozen)
    #    interleaved with teleport exploration; reward events precomputed.
    # ===================================================================== #
    trng = np.random.default_rng(cfg["seed"] + 800)
    prng_cem = np.random.default_rng(cfg["seed"] + 801)
    fam_y = [0.0, cfg["fam_b_y"]]
    tr_fam = np.array([k % 2 for k in range(n_trials)], np.int32)   # a,b,a,b,...
    tr_sgn = trng.choice([-1.0, 1.0], n_trials).astype(np.float32)
    jit = cfg["goal_jit"]
    tr_start = np.stack([-tr_sgn * cr, np.array([fam_y[f] for f in tr_fam], np.float32)], 1) \
        + trng.uniform(-jit, jit, (n_trials, 2)).astype(np.float32)
    tr_goal = np.stack([tr_sgn * cr, np.array([fam_y[f] for f in tr_fam], np.float32)], 1) \
        + trng.uniform(-jit, jit, (n_trials, 2)).astype(np.float32)
    tr_s0 = np.concatenate([tr_start, np.zeros((n_trials, 2), np.float32)], 1)

    # plan every trial with the PRE-DRIFT FM (frozen scripted behavior, arm-independent)
    plans = np.empty((n_trials, L, 2), np.float32)
    for lo in range(0, n_trials, 128):
        hi = min(lo + 128, n_trials)
        plans[lo:hi] = mpc_plan(fm2_0, tr_s0[lo:hi], tr_goal[lo:hi], prng_cem, L)

    # execute open-loop: env1 -> the stream; env0 -> clean-outcome baselines
    sS = np.empty((T, 4), np.float32); sU = np.empty((T, 2), np.float32); sS2 = np.empty((T, 4), np.float32)
    trial_id = np.full(T, -1, np.int32)
    tr_out = np.empty(n_trials, np.float32); tr_out0 = np.empty(n_trials, np.float32)
    erng = np.random.default_rng(cfg["seed"] + 500)
    t_ptr = 0
    for k in range(n_trials):
        for cenv, rec in ((env1, True), (env0, False)):
            cenv.set_state(tr_s0[k, :2].astype(np.float64), tr_s0[k, 2:].astype(np.float64))
            for j in range(L):
                s = cenv.get_state(); u = plans[k, j]
                s2, _ = cenv.step(u, fs)
                if rec:
                    sS[t_ptr + j] = s; sU[t_ptr + j] = u; sS2[t_ptr + j] = s2
            if rec:
                tr_out[k] = float(np.linalg.norm(sS2[t_ptr + L - 1, :2] - tr_goal[k]))
                trial_id[t_ptr:t_ptr + L] = k
            else:
                tr_out0[k] = float(np.linalg.norm(cenv.get_state()[:2] - tr_goal[k]))
        t_ptr += L
        eS, eU, eS2 = collect_box(env1, E, erng)
        sS[t_ptr:t_ptr + E] = eS; sU[t_ptr:t_ptr + E] = eU; sS2[t_ptr:t_ptr + E] = eS2
        t_ptr += E
    assert t_ptr == T
    tr_end_t = np.array([k * (L + E) + L - 1 for k in range(n_trials)], np.int64)

    # reward events: habituating per-family benchmark (Gadagkar's flexible benchmark
    # at the trial level), arm-independent because behavior is frozen
    base_fam = [float(tr_out0[tr_fam == f].mean()) for f in (0, 1)]
    tr_adv = np.empty(n_trials, np.float32)
    bf = list(base_fam)
    for k in range(n_trials):
        f = int(tr_fam[k])
        tr_adv[k] = bf[f] - tr_out[k]
        bf[f] = (1 - cfg["trial_base_alpha"]) * bf[f] + cfg["trial_base_alpha"] * float(tr_out[k])
    kappa_r = max(float(np.sqrt(np.mean(tr_adv[:min(32, n_trials)] ** 2))), 0.02)
    z_r = np.clip(tr_adv / kappa_r, -cfg["z_clip_r"], cfg["z_clip_r"]).astype(np.float32)
    rew_arrival = tr_end_t + cfg["d_r"]
    s_cls = classify(sS)
    frac = {CLS_NAMES[c]: float((s_cls == c).mean()) for c in range(NK + 1)}
    print(f"[stream] class fractions: " + " ".join(f"{k}={v:.3f}" for k, v in frac.items()), flush=True)
    print(f"[trials] fam-a out(clean)={tr_out0[tr_fam == 0].mean():.3f} out(drift)={tr_out[tr_fam == 0].mean():.3f}  "
          f"fam-b out(clean)={tr_out0[tr_fam == 1].mean():.3f} out(drift)={tr_out[tr_fam == 1].mean():.3f}  "
          f"kappa_r={kappa_r:.3f}", flush=True)

    # agency gate precomputed from the pretrained pair (all-self-generated stream)
    s_g = np.linalg.norm(fm2_delta(fm2_0, sS, sU) - fm1_delta(fm1_0, sS), axis=1).astype(np.float32)
    s_gate = (1.0 / (1.0 + np.exp(-(s_g - g0) / theta))).astype(np.float32)
    print(f"[gate] stream gate p10/p50/p90={np.percentile(s_gate, [10, 50, 90]).round(3)}", flush=True)

    # probes in the drifted world (identical for every arm)
    prng = np.random.default_rng(cfg["seed"] + 600)
    probes = {"global": collect_box(env1, cfg["probe_n"], prng)}
    for j in range(NK):
        probes[REG[j]["name"]] = collect_region(env1, j, cfg["probe_n"], prng)
    bs_pos = []
    while sum(len(p) for p in bs_pos) < cfg["probe_n"]:
        cand = np.stack([prng.uniform(-cfg["box_x"], cfg["box_x"], 512),
                         prng.uniform(-cfg["box_y"], cfg["box_y"], 512)], 1).astype(np.float32)
        keep = cand[gates(cand).max(1) < 0.05]
        bs_pos.append(keep)
    bs_pos = np.concatenate(bs_pos)[:cfg["probe_n"]]
    probes["base"] = collect_at(env1, bs_pos, prng)

    def probe_errs(net):
        return {k: float(per_sample_err(net, *p).mean()) for k, p in probes.items()}

    stale_probe = probe_errs(fm2_0)
    print("[stale] probe errs: " + " ".join(f"{k}={v:.4f}" for k, v in stale_probe.items()), flush=True)

    # oracle slow credit (for fidelity readouts): trial membership * z_r
    oracle_slow = np.zeros(T, np.float32)
    for k in range(n_trials):
        oracle_slow[trial_id == k] = z_r[k]

    # ===================================================================== #
    # 4) THE ARMS -- eligibility ledgers over the same stream
    # ===================================================================== #
    def kernel(tau, sig):
        ages = np.arange(A_max + 1, dtype=np.float64)
        Kk = np.exp(-(ages - tau) ** 2 / (2.0 * sig ** 2))
        return (Kk / max(Kk.sum(), 1e-12)).astype(np.float32)

    w_clip = cfg["w_clip"]; ew_a = cfg["ewma_alpha"]

    def run_arm(arm):
        fm = copy.deepcopy(fm2_0)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
        bnet = copy.deepcopy(bench_0)
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        rng = np.random.default_rng(cfg["seed"] + 700)
        needs_fast = any("f" in l["ch"] for l in arm["ledgers"])
        leds = [dict(ch=l["ch"], K=kernel(l["tau"], l["sig"]),
                     credit=np.zeros(T, np.float32), kappa=None) for l in arm["ledgers"]]
        d_fm = arm["d_fm"]
        e_gen = np.zeros(T, np.float32); z_f = np.zeros(T, np.float32); b_gen = np.zeros(T, np.float32)
        w_first = np.ones(T, np.float32)
        w_norm = None
        snaps = {}
        trace = {"t": [], **{f"probe_{k}": [] for k in probes}}
        blog = {"t": []}
        for nm in CLS_NAMES:
            for q in ("w", "e", "c0", "c1"):
                blog[f"{q}_{nm}"] = []
        cum_w = np.zeros(NK + 1, np.float64); cum_n = np.zeros(NK + 1, np.float64)
        w_sum = 0.0; w_cnt = 0
        rew_ptr = 0

        def deposit(led, val, arrival):
            lo = max(arrival - A_max, 0); hi = min(arrival, T - 1)
            if lo > hi:
                return
            ages = arrival - np.arange(lo, hi + 1)
            led["credit"][lo:hi + 1] += val * led["K"][ages]

        def kappa_of(led, consumed_end):
            if led["kappa"] is not None:
                return led["kappa"]
            n = max(consumed_end, B)
            k = max(float(np.sqrt(np.mean(led["credit"][:n] ** 2))), 1e-3)
            if consumed_end >= cfg["kappa_burnin"]:
                led["kappa"] = k
            return k

        def weights_for(idx, consumed_end):
            nonlocal w_norm
            if arm["kind"] == "fixed" or not leds:
                return np.ones(len(idx), np.float32)
            c_tot = np.zeros(len(idx), np.float32)
            for led in leds:
                kap = kappa_of(led, consumed_end)
                c_tot += np.clip(led["credit"][idx] / kap, -8.0, 8.0)
            w_raw = np.exp(np.clip(-c_tot, -30.0, np.log(cfg["w_raw_cap"]))).astype(np.float32)
            if w_norm is None:
                w_norm = float(w_raw.mean())
            w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
            return np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)

        def grad_step(idx, w_hat):
            Xb = torch.tensor(np.concatenate([sS[idx], sU[idx]], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((sS2[idx] - sS[idx]).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w_hat, device=device)
            fm.train(); opt.zero_grad()
            (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward()
            opt.step(); fm.eval()

        if 0 in milestone_set:
            snaps[0] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
        gen_end = 0
        for c in range(n_cycles + lag_cycles):
            head_prev = min(c * B, T + (max(c - n_cycles, 0)) * B) if c <= n_cycles \
                else T + (c - n_cycles) * B
            head = head_prev + B
            # --- experience/generation (Smith-held prediction: e from fm-as-of-now,
            #     broadcast d_fm steps later) ---
            if c < n_cycles:
                idx = np.arange(c * B, (c + 1) * B)
                e = per_sample_err(fm, sS[idx], sU[idx], sS2[idx])
                bb = bench_pred(bnet, sS[idx])
                dlt = (bb - e) * s_gate[idx]
                e_gen[idx] = e; b_gen[idx] = bb
                z_f[idx] = np.clip(dlt / kappa_f, -cfg["z_clip_f"], cfg["z_clip_f"])
                if needs_fast:
                    bench_step(bnet, bopt, sS[idx], e)
                gen_end = (c + 1) * B
            # --- broadcast: anonymous arrivals deposited through the kernels ---
            for led in leds:
                if "f" in led["ch"]:
                    t_lo = max(head_prev - d_fm, 0); t_hi = min(head - d_fm, gen_end)
                    for t_evt in range(t_lo, t_hi):
                        if z_f[t_evt] != 0.0:
                            deposit(led, z_f[t_evt], t_evt + d_fm)
            while rew_ptr < n_trials and rew_arrival[rew_ptr] < head:
                for led in leds:
                    if "r" in led["ch"]:
                        deposit(led, lam_r * z_r[rew_ptr], int(rew_arrival[rew_ptr]))
                rew_ptr += 1
            # --- consume the matured batch (lag A_max) + replay ---
            m = c - lag_cycles
            if m < 0 or m >= n_cycles:
                continue
            cidx = np.arange(m * B, (m + 1) * B)
            consumed_end = (m + 1) * B
            w_hat = weights_for(cidx, consumed_end)
            w_first[cidx] = w_hat
            grad_step(cidx, w_hat)
            w_sum += float(w_hat.sum()); w_cnt += len(cidx)
            cls = s_cls[cidx]
            blog["t"].append(consumed_end)
            c0 = leds[0]["credit"][cidx] if leds else np.zeros(B, np.float32)
            c1 = leds[1]["credit"][cidx] if len(leds) > 1 else np.zeros(B, np.float32)
            for ci, nm in enumerate(CLS_NAMES):
                msk = cls == ci
                for q, arr in (("w", w_hat), ("e", e_gen[cidx]), ("c0", c0), ("c1", c1)):
                    blog[f"{q}_{nm}"].append(float(arr[msk].mean()) if msk.any() else np.nan)
                cum_w[ci] += float(w_hat[msk].sum()); cum_n[ci] += int(msk.sum())
            for _ in range(cfg["n_replay"]):
                ridx = rng.integers(0, consumed_end, size=cfg["replay_batch"])
                w_r = weights_for(ridx, consumed_end)
                grad_step(ridx, w_r)
                w_sum += float(w_r.sum()); w_cnt += len(ridx)
            if m % cfg["probe_every"] == 0 or consumed_end in milestone_set:
                pe = probe_errs(fm)
                trace["t"].append(consumed_end)
                for kk, v in pe.items():
                    trace[f"probe_{kk}"].append(v)
            if consumed_end in milestone_set:
                snaps[consumed_end] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
                print(f"[{arm['name']} m={consumed_end:5d}] " +
                      " ".join(f"{kk}={v:.4f}" for kk, v in pe.items()), flush=True)

        # fidelity: applied credit vs oracle credit, per ledger x channel
        def corr(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        fid = {}
        for li, led in enumerate(leds):
            fid[f"led{li}_vs_fast"] = corr(led["credit"], z_f)
            fid[f"led{li}_vs_slow"] = corr(led["credit"], oracle_slow)
        budget = {"mean_w": w_sum / max(w_cnt, 1),
                  "kappa": [led["kappa"] for led in leds],
                  "cum_w_share": {CLS_NAMES[ci]: float(cum_w[ci] / max(cum_w.sum(), 1e-9))
                                  for ci in range(NK + 1)},
                  "sample_share": {CLS_NAMES[ci]: float(cum_n[ci] / max(cum_n.sum(), 1e-9))
                                   for ci in range(NK + 1)}}
        print(f"[{arm['name']}] mean_w={budget['mean_w']:.3f} fid={ {k: round(v, 3) for k, v in fid.items()} } "
              f"shares={ {k: round(v, 3) for k, v in budget['cum_w_share'].items()} }", flush=True)
        arrs = dict(e_gen=e_gen, z_f=z_f, b_gen=b_gen, w_first=w_first,
                    **{f"cred{li}": led["credit"] for li, led in enumerate(leds)})
        return snaps, trace, blog, budget, fid, arrs

    outdir = os.path.join(DATA_DIR, "two_clocks", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)

    def checkpoint(obj):
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(obj, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    def json_view(arm_out, extra):
        return {"config": cfg, "complete": False,
                "arms": {a: {k: v for k, v in d.items() if k not in ("snaps", "arrs")}
                         for a, d in arm_out.items()}, **extra}

    trials_info = {"fam_a_out_clean": float(tr_out0[tr_fam == 0].mean()),
                   "fam_a_out_drift": float(tr_out[tr_fam == 0].mean()),
                   "fam_b_out_clean": float(tr_out0[tr_fam == 1].mean()),
                   "fam_b_out_drift": float(tr_out[tr_fam == 1].mean()),
                   "kappa_r": kappa_r, "lam_r": lam_r,
                   "adv_mean_a_first16": float(tr_adv[tr_fam == 0][:16].mean()),
                   "adv_mean_a_last16": float(tr_adv[tr_fam == 0][-16:].mean()),
                   "adv_mean_b": float(tr_adv[tr_fam == 1].mean())}
    calib = {"kappa_f": kappa_f, "g0": g0, "theta": theta,
             "pretrain_fm_err_clean": float(e_cal.mean()),
             "stream_gate_p10_p50_p90": np.percentile(s_gate, [10, 50, 90]).tolist(),
             "stream_class_fractions": frac, "stale_probe": stale_probe}

    arm_out = {}
    for arm in arms:
        print(f"\n===== arm: {arm['name']} =====", flush=True)
        snaps, trace, blog, budget, fid, arrs = run_arm(arm)
        arm_out[arm["name"]] = {"snaps": snaps, "trace": trace, "blog": blog,
                                "budget": budget, "fidelity": fid, "arrs": arrs}
        checkpoint(json_view(arm_out, {"calibration": calib, "trials": trials_info}))

    # persist per-sample series (fidelity offset curves are computed offline from these)
    npz = dict(cls=s_cls, trial_id=trial_id, tr_fam=tr_fam, tr_out=tr_out, tr_out0=tr_out0,
               tr_adv=tr_adv, z_r=z_r, rew_arrival=rew_arrival, oracle_slow=oracle_slow,
               s_gate=s_gate)
    for a, d in arm_out.items():
        for k, v in d["arrs"].items():
            npz[f"{a}__{k}"] = v.astype(np.float32)
    np.savez_compressed(os.path.join(outdir, "traces.npz"), **npz)
    volume.commit()

    # ===================================================================== #
    # 5) CEILING FM + CONTROL GRADING (ballistic on the value task)
    # ===================================================================== #
    dS, dU, dS2 = collect_box(env1, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 20))
    fm_ceil = _mlp(cfg["seed"] + 43, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
    train_steps(fm_ceil, optc, dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 303))

    Bev = cfg["n_eval"]
    ev_rng = np.random.default_rng(cfg["seed"] + 7)

    def make_eval_set(y0):
        sgn = ev_rng.choice([-1.0, 1.0], Bev).astype(np.float32)
        starts = np.concatenate([
            np.stack([-sgn * cr, np.full(Bev, y0, np.float32)], 1)
            + ev_rng.uniform(-jit, jit, (Bev, 2)).astype(np.float32),
            ev_rng.normal(0, cfg["v0_std"], (Bev, 2)).astype(np.float32)], 1)
        goals = np.stack([sgn * cr, np.full(Bev, y0, np.float32)], 1) \
            + ev_rng.uniform(-jit, jit, (Bev, 2)).astype(np.float32)
        return starts.astype(np.float32), goals.astype(np.float32)

    ev_a = make_eval_set(0.0)
    ev_b = make_eval_set(cfg["fam_b_y"])

    def rollout(net, replan_every, ev):
        starts, goals = ev
        states = starts.copy(); plan = None
        rng = np.random.default_rng(cfg["seed"] + 7000 + replan_every)
        for step in range(H_ev):
            if step % replan_every == 0:
                plan = mpc_plan(net, states, goals, rng, H_ev)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for bbi in range(Bev):
                env1.set_state(states[bbi, :2].astype(np.float64), states[bbi, 2:].astype(np.float64))
                s2, _ = env1.step(acts[bbi], fs); states[bbi] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - goals, axis=1)))

    ms_sorted = sorted(milestone_set)
    light_ms = {ms_sorted[0], ms_sorted[(2 * len(ms_sorted)) // 3], ms_sorted[-1]}

    def grade(net, full):
        rec = {"fm": probe_errs(net), "ballistic_a": rollout(net, H_ev, ev_a)}
        if full and not cfg["light_grading"]:
            rec["ballistic_b"] = rollout(net, H_ev, ev_b)
            rec["reactive_a"] = rollout(net, 1, ev_a)
        return rec

    print("\n===== grading =====", flush=True)
    g_stale = grade(fm2_0, True); g_stale["transitions"] = 0
    print("[grade stale   ] " + " ".join(f"{k}={v:.4f}" for k, v in g_stale.items() if k != "fm"), flush=True)
    g_ceil = grade(fm_ceil, True); g_ceil["transitions"] = -1
    print("[grade ceiling ] " + " ".join(f"{k}={v:.4f}" for k, v in g_ceil.items() if k != "fm"), flush=True)
    checkpoint(json_view(arm_out, {"calibration": calib, "trials": trials_info,
                                   "stale": g_stale, "ceiling": g_ceil}))

    lad_net = _mlp(0, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    for arm in arms:
        a = arm["name"]; lad = []
        for m in sorted(arm_out[a]["snaps"]):
            if m == 0:
                rec = dict(g_stale); lad.append(rec); continue
            lad_net.load_state_dict(arm_out[a]["snaps"][m]); lad_net.to(device).eval()
            rec = grade(lad_net, full=(m in light_ms)); rec["transitions"] = int(m)
            lad.append(rec)
            print(f"[grade {a:20s} m={m:5d}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k not in ("fm", "transitions")),
                  flush=True)
        arm_out[a]["ladder"] = lad
        checkpoint(json_view(arm_out, {"calibration": calib, "trials": trials_info,
                                       "stale": g_stale, "ceiling": g_ceil}))

    for a in arm_out:
        del arm_out[a]["snaps"]; del arm_out[a]["arrs"]

    out = {"config": cfg, "complete": True, "calibration": calib, "trials": trials_info,
           "stale": g_stale, "ceiling": g_ceil,
           "arms": {a: dict(arm_out[a]) for a in arm_out}}
    figures = _make_figures(out)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("complete\n")
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
    arms = list(R["arms"].keys())
    REG = R["config"]["regions"]
    PALETTE = ["#495057", "#e8590c", "#1971c2", "#7048e8", "#2f9e44",
               "#c2255c", "#f08c00", "#0c8599", "#9c36b5", "#5c940d"]
    ARMC = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(arms)}
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    def _logx(t):
        return np.where(np.asarray(t, float) == 0, R["config"]["batch"] / 2.0, np.asarray(t, float))

    # fig1: ballistic recovery on the value task (+ clean-family / reactive contrasts)
    panels = [("ballistic_a", "ballistic on value task (through R1)"),
              ("ballistic_b", "ballistic on clean family-b"),
              ("reactive_a", "reactive on value task")]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))
    for pi, (key, title) in enumerate(panels):
        ax = axes[pi]
        for a in arms:
            lad = R["arms"][a].get("ladder", [])
            pts = [(r["transitions"], r[key]) for r in lad if key in r]
            if not pts:
                continue
            t, y = zip(*pts)
            ax.plot(_logx(t), y, "o-", color=ARMC[a], lw=1.8, ms=4, label=a)
        if key in R["ceiling"]:
            ax.axhline(R["ceiling"][key], color="#999", ls=":", lw=1.5, label="ceiling")
        ax.set_xscale("log"); ax.set_xlabel("consumed transitions"); ax.set_title(title)
        if pi == 0:
            ax.set_ylabel("median goal dist (lower=better)"); ax.legend(fontsize=7)
    fig.suptitle("Control recovery under two-clock credit assignment (matched budget)", fontsize=10)
    figs["fig1_recovery.png"] = _save(fig)

    # fig2: FM probe error by region class
    keys = [r["name"] for r in REG] + ["base"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0))
    for pi, kk in enumerate(keys):
        ax = axes[pi]
        for a in arms:
            tr = R["arms"][a]["trace"]
            ax.plot(_logx(tr["t"]), tr[f"probe_{kk}"], "-", color=ARMC[a], lw=1.5, label=a)
        ax.set_xscale("log"); ax.set_title(kk); ax.set_xlabel("transitions")
        if pi == 0:
            ax.set_ylabel("FM probe error"); ax.legend(fontsize=6)
    fig.suptitle("Where the model got better", fontsize=10)
    figs["fig2_fm_traces.png"] = _save(fig)

    # fig3: allocation -- cumulative weight share / sample share per class
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(arms), 4.2))
    xw = np.arange(len(arms))
    width = 0.8 / len(keys)
    CLSC = {"base": "#868e96"}
    reds = ["#c92a2a", "#f08c00"]; ri = 0
    for r in REG:
        if r["noise"] > 0:
            CLSC[r["name"]] = "#1971c2"
        else:
            CLSC[r["name"]] = reds[ri % 2]; ri += 1
    for ci, kk in enumerate(keys):
        vals = []
        for a in arms:
            bud = R["arms"][a]["budget"]
            ss = bud["sample_share"].get(kk, 1e-9)
            vals.append(bud["cum_w_share"].get(kk, 0.0) / max(ss, 1e-9))
        ax.bar(xw + ci * width, vals, width, color=CLSC[kk], label=kk)
    ax.axhline(1.0, color="#bbb", ls=":", lw=1)
    ax.set_xticks(xw + 0.4 - width / 2); ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("weight share / sample share"); ax.legend(fontsize=7)
    ax.set_title("Where the plasticity budget went (1 = neutral)", fontsize=10)
    figs["fig3_allocation.png"] = _save(fig)

    # fig4: signal -- credit fidelity per arm/ledger + the reward event stream
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    labels, ffast, fslow = [], [], []
    for a in arms:
        fid = R["arms"][a].get("fidelity", {})
        for li in range(2):
            if f"led{li}_vs_fast" in fid:
                labels.append(f"{a}[{li}]")
                ffast.append(fid[f"led{li}_vs_fast"]); fslow.append(fid[f"led{li}_vs_slow"])
    xp = np.arange(len(labels))
    ax.bar(xp - 0.18, ffast, 0.36, color="#e8590c", label="vs fast oracle (own delta)")
    ax.bar(xp + 0.18, fslow, 0.36, color="#1971c2", label="vs slow oracle (trial adv)")
    ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("corr(applied credit, oracle)"); ax.legend(fontsize=7)
    ax.set_title("Credit fidelity per ledger", fontsize=10)
    ax = axes[1]
    tri = R["trials"]
    ax.bar(["a clean", "a drift", "b clean", "b drift"],
           [tri["fam_a_out_clean"], tri["fam_a_out_drift"],
            tri["fam_b_out_clean"], tri["fam_b_out_drift"]],
           color=["#adb5bd", "#c92a2a", "#adb5bd", "#1971c2"])
    ax.set_ylabel("scripted trial outcome (goal dist)")
    ax.set_title(f"Trial outcome contrast (kappa_r={tri['kappa_r']:.3f})", fontsize=10)
    figs["fig4_signal.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def two_clocks(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # arms: comma-separated specs (see parse_arm). Default = the (ii) architecture test.
    arms: str = ("fixed,fast:t4:d4,slow:T24,two:t4:d4:T24,"
                 "shared:t4:d4,shared:t10:d4,shared:t24:d4,shared:t14:s12:d4"),
    light_grading: bool = False,          # sweep mode: ballistic_a only
    # the drift: R1-on (on the reach path, value-relevant), R2-off (off-path decoy,
    # reducible but value-irrelevant), R3-noise (aleatoric noisy-TV)
    regions: str = ("0.30,0.0,1.2,0.0,R1-on; "
                    "0.0,-0.95,-1.2,0.0,R2-off; "
                    "0.0,0.85,0.0,30.0,R3-noise"),
    region_sigma: float = 0.18,
    # stream structure / clocks
    total_t: int = 16384,
    trial_len: int = 32,                  # scripted ballistic reach horizon
    explore_len: int = 32,                # teleport block between trials
    batch: int = 16,
    a_max: int = 48,                      # eligibility horizon (= consumption lag)
    d_r: int = 8,                         # reward delay after trial end
    lam_r: float = 0.0,                   # reward deposit gain; 0 -> rate comp (L+E)
    trial_base_alpha: float = 0.012,      # habituating trial benchmark EWMA (per trial)
    z_clip_f: float = 8.0,
    z_clip_r: float = 3.0,
    kappa_burnin: int = 2048,             # matured samples before ledger scale freezes
    # learning
    n_replay: int = 4,
    replay_batch: int = 64,
    adapt_lr: float = 3e-4,
    milestones: str = "0,512,1024,2048,4096,8192,16384",
    probe_every: int = 16,
    # gain law (plasticity_gain discipline)
    w_clip: float = 4.0,
    w_raw_cap: float = 20.0,
    ewma_alpha: float = 0.05,
    # benchmark net b(s)
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 3e-3,
    bench_pretrain_steps: int = 1500,
    # env (the 4c ballistic-competent regime)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
    fam_b_y: float = -0.45,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    pool_n: int = 9000,
    probe_n: int = 500,
    v_explore: float = 1.2,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_steps: int = 4000,
    # control eval
    n_eval: int = 40,
    goal_jit: float = 0.06,
    v0_std: float = 0.3,
    plan_h: int = 34,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    REG = []
    for chunk in [c for c in regions.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        REG.append(dict(center=(float(q[0]), float(q[1])), sigma=region_sigma,
                        phi=float(q[2]), noise=float(q[3]), name=q[4]))
    ms = [int(x) for x in milestones.split(",") if x]
    if quick:
        total_t = 2048; ms = [0, 512, 2048]; pool_n = 2500; probe_n = 250
        fm_steps = 1200; bench_pretrain_steps = 500; fm_hidden = 128; fm_layers = 2
        n_eval = 12; k_shoot = 96; probe_every = 8; kappa_burnin = 512
        arms = "fixed,two:t4:d4:T24,shared:t10:d4"
        tag = tag or "smoke"
    tag = tag or "default"
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arm_list:
        parse_arm(a)                       # validate early, locally
    assert a_max % batch == 0, "a_max must be a multiple of batch"
    assert total_t % (trial_len + explore_len) == 0, "T must be a multiple of L+E"
    assert all(m % batch == 0 for m in ms), "milestones must be multiples of batch"
    assert total_t == max(ms), "total_t should equal the last milestone"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, light_grading=light_grading, regions=REG,
        T=total_t, trial_len=trial_len, explore_len=explore_len, batch=batch,
        a_max=a_max, d_r=d_r, lam_r=lam_r, trial_base_alpha=trial_base_alpha,
        z_clip_f=z_clip_f, z_clip_r=z_clip_r, kappa_burnin=kappa_burnin,
        n_replay=n_replay, replay_batch=replay_batch, adapt_lr=adapt_lr, milestones=ms,
        probe_every=probe_every, w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha,
        bench_hidden=bench_hidden, bench_layers=bench_layers, bench_lr=bench_lr,
        bench_pretrain_steps=bench_pretrain_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, fam_b_y=fam_b_y, box_x=box_x, box_y=box_y,
        collect_sigma_frac=collect_sigma_frac, pool_n=pool_n, probe_n=probe_n,
        v_explore=v_explore, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr,
        fm_steps=fm_steps, n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std,
        plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_two_clocks.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
