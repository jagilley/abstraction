"""practice/estimability: does making a context RECUR make b(s) estimable?

Program: `ideas/practice_manufactures_its_own_credit.md` S1 -- allocation's job is to
"make a context recur at a rate fast relative to competence drift, against a stable
target -- the estimability condition for b(s)". Every node in the bridge line held that
condition CONSTANT by construction: `bridge_assembly.py` draws the whole online stream as
one spatially-uniform i.i.d. box sample consumed in order; `plasticity_gain` uses one
shared reward-free stream identical across arms; `two_clocks` scripts its trials. The
variable practice is about has never varied. This node varies it and nothing else.

    THE MANIPULATION (the massed-vs-spaced control from the human practice literature).
    Hold per-context sample count EXACTLY fixed and vary only temporal concentration.
    The stream is 1000 CONTEXT-PURE batches (B=16), 200 per context, over 5 contexts. A
    schedule is a PERMUTATION of that batch sequence with burst length `burst` (in
    batches):

        burst=200  massed        A A A ... A B B B ... B ... (each context visited once)
        burst=20                 [20xA][20xB][20xC][20xN][20xM] repeated 10 times
        burst=1    interleaved   A B C N M A B C N M ...

    Revisit interval = K * burst batches. The TRANSITIONS ARE LITERALLY IDENTICAL across
    schedules -- same collected data, same batch composition, same count per context --
    so any difference is order, and only order. Replay is OFF by default (`replay_mode`
    none): uniform replay over the past is itself an interleaver and would blunt the IV;
    it is available as an explicit contrast cell (`--replay-mode uniform`).

    THE QUESTION THE DOC ASSUMES AND THE IMPLEMENTATION DOES NOT ANSWER. b(s) in the
    corrected delta is a NET over state (plasticity_gain #3: the scalar EWMA is
    mechanistically pathological), not a scalar EWMA. A per-context EWMA updated on
    visits is EXACTLY spacing-invariant at fixed count -- the sequence of e values it
    sees at a context does not depend on when the visits happened -- so under the doc's
    own EWMA intuition this manipulation should do NOTHING except through cross-context
    coupling. A net has that coupling by construction (generalization / interference).
    So the entire effect must route through the thing the doc did not model. Hence the
    estimator panel.

    THE BENCHMARK PANEL (rides passively on the `fixed` arm, so every estimator reads
    the SAME (s,e) stream off the SAME FM trajectory -- the tightest possible control on
    the estimability comparison):
      * net@lr in {1e-3, 3e-3, 1e-2, 3e-2}   the committed form, timescale swept
      * ewma_ctx@alpha in {0.01, 0.05, 0.2}  per-context tabular scalar (GIVEN the
                                             context label -- oracle information, a
                                             handicap in the EWMA's favor)
      * ewma_scalar@0.05                     the idea doc's literal S1 global scalar
      * frozen                               pretrained b, never updated (the floor)
    Crossing the panel with the schedule sweep gives the 2D picture benchmark_vs_cost
    fig4 asks for: benchmark timescale x revisit rate are the two halves of one
    estimability condition.

    THE ORACLE. b(s)'s target is E[e | pos] under the CURRENT FM (b sees state, not
    action, so the target is marginal over u and over the world's aleatoric draw). It is
    measured, not fitted: a grid of `grid_pos` positions per class x `grid_rep` repeats
    (fresh v, u, noise) gives E[e|pos] directly; it is recomputed with each arm's own FM
    every `oracle_every` cycles, so the ground truth MOVES with competence -- which is
    the whole point (a static-competence version measures the wrong thing).

    READOUTS
      * b(s) fidelity vs the moving truth: corr / RMSE / per-class signed bias, where
        b is marginalized over the same repeats as the truth.
      * delta credit fidelity: corr(applied delta, oracle delta), corr of applied vs
        oracle plasticity weight, material sign-error rate, and the allocation TV
        distance between where plasticity actually went and where an oracle benchmark
        would have sent it.
      * REVISIT LAG (the within-run dose-response, and the sharp instrument): every
        consumed batch carries `lag` = cycles since that context was last practiced.
        Binning credit error by lag measures the estimability condition inside a single
        run, pooled across schedules.
      * The Delta-decomposition that answers net-vs-EWMA: over each away-interval,
        d_truth = change in E[e|pos] at the absent context (competence drift), d_b =
        change in b there (pure generalization/interference -- a tabular EWMA has
        d_b == 0 by construction). Regressing d_b on d_truth measures whether the net's
        generalization TRACKS the drift (slope ~1: generalization substitutes for
        recurrence) or corrupts the estimate (slope ~0 or negative).
      * Learning outcome: per-class FM error over time from the same grid (acquisition
        AUC + the end-of-run retention profile). No control grading: this node measures
        COLLECTION and ESTIMATION; bridge_assembly measured consumption -> behavior.

Run:
    cd experiments/
    modal run mjc/practice/estimability/estimability.py::estimability --quick   # smoke
    python3 mjc/practice/estimability/launch_detached.py --tag est_s0 --seed 0
"""

import copy
import json

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_schedule(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    CTX = cfg["contexts"]; K = len(CTX)
    CLS = [c["name"] for c in CTX] + ["base"]
    B = cfg["batch"]; NB = cfg["n_bpc"]; burst = cfg["burst"]
    n_ctx = NB * B; T = K * n_ctx
    G, R = cfg["grid_pos"], cfg["grid_rep"]
    print(f"[setup] device={device} burst={burst} K={K} B={B} n_bpc={NB} T={T} "
          f"arms={[a for a in cfg['arms']]}", flush=True)
    print("[setup] contexts: " + "; ".join(
        f"{c['name']}@({c['center'][0]:+.2f},{c['center'][1]:+.2f}) phi={c['phi']:+.2f} "
        f"noise={c['noise']:.0f} pre={c['pre']}" for c in CTX), flush=True)

    def make_env(drifted):
        d = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                 joint_damping=cfg["damping"], pusher_r=0.12,
                 noise_seed=cfg["seed"] + 999)
        regs = [c for c in CTX if (drifted or c["pre"])]
        if regs:
            d["rot_regions"] = [dict(center=tuple(c["center"]), sigma=c["sigma"],
                                     phi=float(c["phi"]), noise=float(c["noise"]))
                                for c in regs]
        return PusherEnv(d, with_puck=False)

    env0 = make_env(False)     # pre-drift: only the pre=1 (mastered) contexts exist
    env1 = make_env(True)      # post-drift: everything

    # ---------------- nets ----------------
    def _mlp(seed, din, dout, h, L):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, dout)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    # ---------------- collection (bridge_assembly's, unchanged) ----------------
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

    def ctx_pos(j, n, rng):
        c = CTX[j]["center"]; s = CTX[j]["sigma"] * cfg["collect_sigma_frac"]
        return np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1).astype(np.float32)

    def gates(pos):
        w = np.empty((len(pos), K), np.float32)
        for j, c in enumerate(CTX):
            d2 = (pos[:, 0] - c["center"][0]) ** 2 + (pos[:, 1] - c["center"][1]) ** 2
            w[:, j] = np.exp(-d2 / (2.0 * c["sigma"] ** 2))
        return w

    def base_pos(n, rng):
        acc = []
        while sum(len(a) for a in acc) < n:
            cand = np.stack([rng.uniform(-cfg["box_x"], cfg["box_x"], 512),
                             rng.uniform(-cfg["box_y"], cfg["box_y"], 512)], 1).astype(np.float32)
            acc.append(cand[gates(cand).max(1) < 0.05])
        return np.concatenate(acc)[:n]

    # ---------------- normalization + pretraining pool ----------------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
    if cfg["master_extra_n"] > 0:
        mrng = np.random.default_rng(cfg["seed"] + 3)
        for j, c in enumerate(CTX):
            if c["pre"]:
                eS, eU, eS2 = collect_at(env0, ctx_pos(j, cfg["master_extra_n"], mrng), mrng)
                nS = np.concatenate([nS, eS]); nU = np.concatenate([nU, eU])
                nS2 = np.concatenate([nS2, eS2])
                print(f"[pretrain] +{cfg['master_extra_n']} practice samples in {c['name']}", flush=True)
        Y = (nS2 - nS).astype(np.float32)

    huber = nn.HuberLoss(delta=1.0)
    huber_ps = nn.HuberLoss(delta=1.0, reduction="none")

    def train_steps(net, opt, S, U, S2, steps, brng, bs=512):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (Xt - norm["mx"]) / norm["sx"]; Yn = (Yt - norm["my"]) / norm["sy"]
        bs = min(bs, len(S)); net.train()
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
    # 1) PRETRAIN on the pre-drift world: FM2, FM1, benchmark b0
    # ===================================================================== #
    fm2_0 = _mlp(cfg["seed"] + 40, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    opt2 = torch.optim.Adam(fm2_0.parameters(), lr=cfg["fm_lr"])
    train_steps(fm2_0, opt2, nS, nU, nS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))

    fm1_0 = _mlp(cfg["seed"] + 41, 4, 4, cfg["fm_hidden"], cfg["fm_layers"])
    X1n = (torch.tensor(nS, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
    Y1n = (torch.tensor(Y, device=device) - norm["my"]) / norm["sy"]
    opt1 = torch.optim.Adam(fm1_0.parameters(), lr=cfg["fm_lr"])
    brng1 = np.random.default_rng(cfg["seed"] + 301); fm1_0.train()
    for _ in range(cfg["fm_steps"]):
        idx = torch.tensor(brng1.integers(0, len(nS), size=512), device=device)
        opt1.zero_grad(); huber(fm1_0(X1n[idx]), Y1n[idx]).backward(); opt1.step()
    fm1_0.eval()

    e_pool = per_sample_err(fm2_0, nS, nU, nS2)
    bench_0 = _mlp(cfg["seed"] + 42, 4, 1, cfg["bench_hidden"], cfg["bench_layers"])
    bopt_0 = torch.optim.Adam(bench_0.parameters(), lr=cfg["bench_lr"])
    brngb = np.random.default_rng(cfg["seed"] + 302)
    for _ in range(cfg["bench_pretrain_steps"]):
        idx = brngb.integers(0, len(nS), size=256)
        bench_step(bench_0, bopt_0, nS[idx], e_pool[idx])

    # ------------- calibration (bridge_assembly's protocol, unchanged) -------------
    cS, cU, cS2 = collect_box(env0, cfg["cal_n"], np.random.default_rng(cfg["seed"] + 2))
    e_cal = per_sample_err(fm2_0, cS, cU, cS2)
    resid = bench_pred(bench_0, cS) - e_cal
    tau = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)

    p1_pool = fm1_delta(fm1_0, nS)
    g_act = np.linalg.norm(fm2_delta(fm2_0, nS, nU) - p1_pool, axis=1)
    g_pas = np.linalg.norm(fm2_delta(fm2_0, nS, np.zeros_like(nU)) - p1_pool, axis=1)
    m_a, m_p = float(np.median(g_act)), float(np.median(g_pas))
    g0 = 0.5 * (m_a + m_p); theta_g = max((m_a - m_p) / 8.0, 1e-9)
    allv = np.concatenate([g_act, g_pas])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    auroc = float((ranks[:len(g_act)].sum() - len(g_act) * (len(g_act) + 1) / 2.0)
                  / (len(g_act) * len(g_pas) + 1e-12))

    def gate_fn(g):
        return (1.0 / (1.0 + np.exp(-(g - g0) / theta_g))).astype(np.float32)

    print(f"[calib] fm_err(clean)={e_cal.mean():.4f} tau={tau:.4f} | gate g0={g0:.5f} "
          f"theta={theta_g:.6f} AUROC={auroc:.4f}", flush=True)

    # ===================================================================== #
    # 2) THE ORACLE GRID: E[e | pos] under the CURRENT FM, measured not fitted
    # ===================================================================== #
    grng = np.random.default_rng(cfg["seed"] + 600)
    gp = [ctx_pos(j, G, grng) for j in range(K)] + [base_pos(G, grng)]
    gpos = np.concatenate(gp).astype(np.float32)
    gcls = np.concatenate([np.full(G, j) for j in range(K + 1)])
    P = len(gpos)
    grS, grU, grS2 = collect_at(env1, np.repeat(gpos, R, axis=0), grng)
    starts = np.arange(0, P * R, R)

    def field_e(net):
        return (np.add.reduceat(per_sample_err(net, grS, grU, grS2), starts) / R).astype(np.float32)

    def field_b(pred_fn):
        return (np.add.reduceat(pred_fn(grS), starts) / R).astype(np.float32)

    def per_class(v):
        return {CLS[c]: float(v[gcls == c].mean()) for c in range(K + 1)}

    stale_field = field_e(fm2_0)
    print("[stale] E[e|pos] per class: " +
          " ".join(f"{k}={v:.4f}" for k, v in per_class(stale_field).items()), flush=True)

    # ===================================================================== #
    # 3) THE STREAM: context-pure batches, identical across every schedule
    # ===================================================================== #
    srng = np.random.default_rng(cfg["seed"] + 500)
    sS = np.empty((T, 4), np.float32); sU = np.empty((T, 2), np.float32); sS2 = np.empty((T, 4), np.float32)
    for j in range(K):
        a, b_ = j * n_ctx, (j + 1) * n_ctx
        sS[a:b_], sU[a:b_], sS2[a:b_] = collect_at(env1, ctx_pos(j, n_ctx, srng), srng)
    s_cls = np.repeat(np.arange(K), n_ctx)
    p1_stream = fm1_delta(fm1_0, sS)

    # k-NN from every stream sample to the oracle grid (position space; contexts are
    # separated by >= 3 sigma so neighbours are in-context by construction)
    d2 = ((sS[:, None, :2] - gpos[None, :, :2]) ** 2).sum(-1)
    nbr = np.argpartition(d2, cfg["knn"], axis=1)[:, :cfg["knn"]].astype(np.int32)
    del d2

    # the schedule: a permutation of the (ctx, batch) sequence with burst length `burst`
    order = []
    for s0 in range(0, NB, burst):
        for j in range(K):
            for bi in range(s0, min(s0 + burst, NB)):
                order.append((j, bi))
    n_cycles = len(order)
    assert n_cycles == K * NB
    # lag = cycles since this context was last practiced (n_cycles = first visit ever).
    # pib = position within the current contiguous run of this context (0 = re-entry).
    # Within a burst lag==1 ("continuation"); at a burst boundary lag == (K-1)*burst
    # ("re-entry after an absence") -- those are the diagnostic batches.
    lag_of_cycle = np.zeros(n_cycles, np.int64)
    pib_of_cycle = np.zeros(n_cycles, np.int64)
    last_seen = {j: -1 for j in range(K)}
    for c, (j, _) in enumerate(order):
        lag_of_cycle[c] = n_cycles if last_seen[j] < 0 else c - last_seen[j]
        pib_of_cycle[c] = 0 if lag_of_cycle[c] != 1 else pib_of_cycle[c - 1] + 1
        last_seen[j] = c
    reentry = (lag_of_cycle > 1) & (lag_of_cycle < n_cycles)
    print(f"[sched] burst={burst} cycles={n_cycles} revisit interval="
          f"{K * burst} batches ({K * burst * B} transitions); absence at re-entry="
          f"{(K - 1) * burst} cycles; n_reentry={int(reentry.sum())}", flush=True)

    # ===================================================================== #
    # 4) BENCHMARK ESTIMATORS
    # ===================================================================== #
    rep_cls = np.repeat(gcls, R)
    b0_all = bench_pred(bench_0, grS)
    b0_cls = np.array([float(b0_all[rep_cls == c].mean()) for c in range(K + 1)], np.float32)
    b0_scalar = float(bench_pred(bench_0, nS).mean())
    ORACLE_HOLD = {"bstar": np.zeros(T, np.float32)}

    def make_est(kind, param):
        """Uniform interface: predict(idx)->b, update(idx,e), field()->[P]."""
        if kind == "net":
            net = copy.deepcopy(bench_0)
            opt = torch.optim.Adam(net.parameters(), lr=param)
            return dict(kind=kind, param=param,
                        predict=lambda idx: bench_pred(net, sS[idx]),
                        update=lambda idx, e: [bench_step(net, opt, sS[idx], e)
                                               for _ in range(cfg["inner_steps"])],
                        field=lambda: field_b(lambda S: bench_pred(net, S)))
        if kind == "ewma_ctx":
            tab = b0_cls.copy()

            def up(idx, e):
                for c in np.unique(s_cls[idx]):
                    m = s_cls[idx] == c
                    tab[c] = (1 - param) * tab[c] + param * float(e[m].mean())
            return dict(kind=kind, param=param,
                        predict=lambda idx: tab[s_cls[idx]].copy(),
                        update=up, field=lambda: tab[gcls].copy())
        if kind == "ewma_scalar":
            st = {"b": b0_scalar}

            def up(idx, e):
                st["b"] = (1 - param) * st["b"] + param * float(e.mean())
            return dict(kind=kind, param=param,
                        predict=lambda idx: np.full(len(idx), st["b"], np.float32),
                        update=up, field=lambda: np.full(P, st["b"], np.float32))
        if kind == "frozen":
            return dict(kind=kind, param=0.0,
                        predict=lambda idx: bench_pred(bench_0, sS[idx]),
                        update=lambda idx, e: None,
                        field=lambda: field_b(lambda S: bench_pred(bench_0, S)))
        if kind == "oracle":
            return dict(kind=kind, param=0.0,
                        predict=lambda idx: ORACLE_HOLD["bstar"][idx].copy(),
                        update=lambda idx, e: None,
                        field=lambda: ORACLE_HOLD["field"].copy())
        raise ValueError(kind)

    def est_name(kind, param):
        return kind if kind in ("frozen", "oracle") else f"{kind}@{param:g}"

    panel_spec = ([("net", lr) for lr in cfg["panel_net_lrs"]]
                  + [("ewma_ctx", a) for a in cfg["panel_ewma_alphas"]]
                  + [("ewma_scalar", cfg["panel_scalar_alpha"]), ("frozen", 0.0)])

    # ===================================================================== #
    # 5) ARMS
    # ===================================================================== #
    outdir = os.path.join(DATA_DIR, "practice_estimability", cfg["tag"], f"burst{burst}")
    os.makedirs(outdir, exist_ok=True)
    calibration = dict(tau=tau, g0=g0, theta_g=theta_g, gate_auroc_train=auroc,
                       pretrain_fm_err_clean=float(e_cal.mean()),
                       stale_field_per_class=per_class(stale_field),
                       b0_cls=b0_cls.tolist(), b0_scalar=b0_scalar)
    result = {"config": cfg, "complete": False, "calibration": calibration,
              "classes": CLS, "arms": {},
              "schedule": {"burst": burst, "n_cycles": n_cycles,
                           "revisit_batches": K * burst,
                           "absence_at_reentry": (K - 1) * burst,
                           "n_reentry": int(reentry.sum()),
                           "ctx_of_cycle": [int(j) for j, _ in order]}}

    def checkpoint():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    w_clip = cfg["w_clip"]; cap = cfg["w_raw_cap"]; ew_a = cfg["ewma_alpha"]

    def wlaw(dlt):
        return np.exp(np.clip(-dlt / tau, -30.0, np.log(cap))).astype(np.float32)

    def run_arm(name):
        rule = ("fixed" if name == "fixed" else "raw" if name == "raw_err" else "delta")
        est = None
        if rule == "delta":
            kind, param = cfg["arm_est"][name]
            est = make_est(kind, param)
        panel = ([dict(name=est_name(k, p), **make_est(k, p)) for k, p in panel_spec]
                 if name == cfg["panel_arm"] else [])

        fm = copy.deepcopy(fm2_0)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
        rrng = np.random.default_rng(cfg["seed"] + 700)
        w_norm = 1.0 if rule == "delta" else float(e_pool.mean())
        w_norm_or = 1.0
        pan_norm = {p["name"]: 1.0 for p in panel}

        # per-sample records over the consumed stream
        rec = {k: np.full(T, np.nan, np.float32) for k in
               ("w", "w_or", "e", "b", "bstar", "dlt", "dlt_or", "gate")}
        rec_lag = np.zeros(T, np.int64); rec_pib = np.zeros(T, np.int64)
        pan_rec = {p["name"]: {k: np.full(T, np.nan, np.float32) for k in ("b", "dlt", "w")}
                   for p in panel}
        ck = {"t": [], "cycle": [], "truth": [], "arm_b": [],
              **{f"pan_{p['name']}": [] for p in panel}}
        blog = {k: [] for k in ("t", "ctx", "lag", "pib", "e", "b", "bstar", "dlt", "w", "gate")}
        cur_field = field_e(fm)
        ORACLE_HOLD["field"] = cur_field
        ORACLE_HOLD["bstar"] = cur_field[nbr].mean(1).astype(np.float32)
        seen_arr = np.empty(T, np.int64); n_seen = 0

        def nmean(a):
            return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

        def snap_ck(c):
            ck["t"].append((c + 1) * B); ck["cycle"].append(c)
            ck["truth"].append(per_class(cur_field))
            ck["arm_b"].append(per_class(est["field"]()) if est is not None else {})
            for p in panel:
                ck[f"pan_{p['name']}"].append(per_class(p["field"]()))

        def weights_for(idx):
            """The arm's per-sample plasticity weights, assembled at use time from the
            arm's CURRENT fm and benchmark (bridge_assembly's discipline)."""
            nonlocal w_norm, w_norm_or
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            pred = fm2_delta(fm, S, U)
            e = np.linalg.norm(pred - (S2 - S), axis=1).astype(np.float32)
            gate = gate_fn(np.linalg.norm(pred - p1_stream[idx], axis=1).astype(np.float32))
            bstar = ORACLE_HOLD["bstar"][idx]
            dlt_or = ((bstar - e) * gate).astype(np.float32)
            w_norm_or = (1 - ew_a) * w_norm_or + ew_a * float(wlaw(dlt_or).mean())
            w_or = np.clip(wlaw(dlt_or) / max(w_norm_or, 1e-8), 0.0, w_clip).astype(np.float32)
            nan = np.full(len(idx), np.nan, np.float32)
            if rule == "fixed":
                b = nan; dlt = np.zeros(len(idx), np.float32); w = np.ones(len(idx), np.float32)
            else:
                if rule == "raw":
                    b = nan; dlt = np.zeros(len(idx), np.float32); w_raw = e
                else:
                    b = est["predict"](idx).astype(np.float32)
                    dlt = ((b - e) * gate).astype(np.float32)
                    w_raw = wlaw(dlt)
                w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
                w = np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)
            return dict(w=w, w_or=w_or, e=e, b=b, bstar=bstar, dlt=dlt, dlt_or=dlt_or, gate=gate)

        def grad_step(idx, w):
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            Xb = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w, device=device)
            fm.train()
            for _ in range(cfg["inner_steps"]):
                opt.zero_grad()
                (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward(); opt.step()
            fm.eval()

        def bench_updates(idx, e, gate, record):
            """Arm benchmark + the passive panel consume the same (s, e) stream."""
            if est is not None:
                est["update"](idx, e)
            for p in panel:
                pb = p["predict"](idx).astype(np.float32)
                pdl = ((pb - e) * gate).astype(np.float32)
                pw_raw = wlaw(pdl)
                pan_norm[p["name"]] = ((1 - ew_a) * pan_norm[p["name"]]
                                       + ew_a * float(pw_raw.mean()))
                if record:
                    pan_rec[p["name"]]["b"][idx] = pb
                    pan_rec[p["name"]]["dlt"][idx] = pdl
                    pan_rec[p["name"]]["w"][idx] = np.clip(
                        pw_raw / max(pan_norm[p["name"]], 1e-8), 0.0, w_clip)
                p["update"](idx, e)

        for c, (j, bi) in enumerate(order):
            idx = np.arange(j * n_ctx + bi * B, j * n_ctx + (bi + 1) * B)
            q = weights_for(idx)
            grad_step(idx, q["w"])
            bench_updates(idx, q["e"], q["gate"], record=True)
            seen_arr[n_seen:n_seen + len(idx)] = idx; n_seen += len(idx)

            # optional interleaver contrast: uniform replay over everything seen so far,
            # weighted by the SAME rule (so the arm comparison stays clean) and feeding
            # the benchmarks, exactly as bridge_assembly's replay does.
            if cfg["replay_mode"] == "uniform" and cfg["n_replay"] > 0:
                for _ in range(cfg["n_replay"]):
                    ridx = seen_arr[rrng.integers(0, n_seen, size=cfg["replay_batch"])]
                    qr = weights_for(ridx)
                    grad_step(ridx, qr["w"])
                    bench_updates(ridx, qr["e"], qr["gate"], record=False)

            for k in ("w", "w_or", "e", "b", "bstar", "dlt", "dlt_or", "gate"):
                rec[k][idx] = q[k]
            rec_lag[idx] = lag_of_cycle[c]; rec_pib[idx] = pib_of_cycle[c]
            blog["t"].append((c + 1) * B); blog["ctx"].append(int(j))
            blog["lag"].append(int(lag_of_cycle[c])); blog["pib"].append(int(pib_of_cycle[c]))
            for k in ("e", "b", "bstar", "dlt", "w", "gate"):
                blog[k].append(nmean(q[k]))

            if (c + 1) % cfg["oracle_every"] == 0 or c == n_cycles - 1:
                cur_field = field_e(fm)
                ORACLE_HOLD["field"] = cur_field
                ORACLE_HOLD["bstar"] = cur_field[nbr].mean(1).astype(np.float32)
            if c % cfg["ckpt_every"] == 0 or c == n_cycles - 1:
                snap_ck(c)
                if c % (cfg["ckpt_every"] * 10) == 0 or c == n_cycles - 1:
                    print(f"[{name} c={c:4d}] " + " ".join(
                        f"{k}={v:.4f}" for k, v in per_class(cur_field).items()), flush=True)

        # ---------------- fidelity summaries ----------------
        def corr(a, b_):
            m = np.isfinite(a) & np.isfinite(b_)
            if m.sum() < 8 or np.std(a[m]) < 1e-12 or np.std(b_[m]) < 1e-12:
                return 0.0
            return float(np.corrcoef(a[m], b_[m])[0, 1])

        def fid_of(b_arr, d_arr, w_arr):
            out = {}
            ok = np.isfinite(b_arr)
            out["b_rmse"] = float(np.sqrt(np.nanmean((b_arr - rec["bstar"]) ** 2))) if ok.any() else None
            out["b_bias"] = float(np.nanmean(b_arr - rec["bstar"])) if ok.any() else None
            out["b_corr"] = corr(b_arr, rec["bstar"])
            out["delta_corr"] = corr(d_arr, rec["dlt_or"])
            out["w_corr"] = corr(w_arr, rec["w_or"])
            mat = np.isfinite(d_arr) & (np.abs(rec["dlt_or"]) > cfg["sign_eps"] * tau)
            out["sign_err_material"] = (float((np.sign(d_arr[mat]) != np.sign(rec["dlt_or"][mat])).mean())
                                        if mat.sum() > 8 else None)
            out["sign_err_all"] = float((np.sign(d_arr) != np.sign(rec["dlt_or"])).mean())
            sw = np.array([np.nansum(w_arr[s_cls == cc]) for cc in range(K)])
            so = np.array([np.nansum(rec["w_or"][s_cls == cc]) for cc in range(K)])
            out["alloc_share"] = {CLS[cc]: float(sw[cc] / max(sw.sum(), 1e-9)) for cc in range(K)}
            out["alloc_share_oracle"] = {CLS[cc]: float(so[cc] / max(so.sum(), 1e-9)) for cc in range(K)}
            out["alloc_tv"] = float(0.5 * np.abs(sw / max(sw.sum(), 1e-9)
                                                 - so / max(so.sum(), 1e-9)).sum())
            # per-class signed b bias and rmse
            out["b_bias_cls"] = {CLS[cc]: (float(np.nanmean(b_arr[s_cls == cc] - rec["bstar"][s_cls == cc]))
                                           if ok.any() else None) for cc in range(K)}
            # binned by revisit lag
            bins = cfg["lag_bins"]; lb = {}
            # the first visit ever to a context has no predecessor -- kept separate so
            # the long-lag bins mean "returned after an absence", not "never seen".
            for lo, hi in list(zip([0] + bins, bins + [n_cycles])) + [("first", "first")]:
                m = ((rec_lag == n_cycles) if lo == "first"
                     else (rec_lag >= lo) & (rec_lag < hi) & (rec_lag < n_cycles))
                m = m & np.isfinite(d_arr)
                if m.sum() < 16:
                    continue
                mm = m & (np.abs(rec["dlt_or"]) > cfg["sign_eps"] * tau)
                lb["first" if lo == "first" else f"{lo}-{hi}"] = dict(
                    n=int(m.sum()),
                    b_abs_err=float(np.nanmean(np.abs(b_arr[m] - rec["bstar"][m]))) if ok.any() else None,
                    b_bias=float(np.nanmean(b_arr[m] - rec["bstar"][m])) if ok.any() else None,
                    delta_corr=corr(d_arr[m], rec["dlt_or"][m]),
                    sign_err=float((np.sign(d_arr[mm]) != np.sign(rec["dlt_or"][mm])).mean())
                    if mm.sum() > 8 else None)
            out["by_lag"] = lb
            # binned by position within the burst: the RE-ESTIMATION TRANSIENT (how many
            # batches of practice it takes to re-fit b after an absence). pib=0 is the
            # re-entry batch itself.
            pb_ = {}
            for lo, hi in zip([0, 1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 10 ** 9]):
                m = (rec_pib >= lo) & (rec_pib < hi) & np.isfinite(d_arr) & (rec_lag < n_cycles)
                if m.sum() < 16:
                    continue
                pb_[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = dict(
                    n=int(m.sum()),
                    b_abs_err=float(np.nanmean(np.abs(b_arr[m] - rec["bstar"][m]))) if ok.any() else None,
                    b_bias=float(np.nanmean(b_arr[m] - rec["bstar"][m])) if ok.any() else None,
                    delta_corr=corr(d_arr[m], rec["dlt_or"][m]))
            out["by_pib"] = pb_
            return out

        arm_out = {
            "rule": rule,
            "est": (cfg["arm_est"][name] if rule == "delta" else None),
            "final_field": per_class(cur_field),
            "stale_field": per_class(stale_field),
            "ckpt": ck,
            "blog": blog,
            "mean_w": float(np.nanmean(rec["w"])),
            "fidelity": fid_of(rec["b"], rec["dlt"], rec["w"]) if rule == "delta" else None,
            "alloc_share": {CLS[cc]: float(np.nansum(rec["w"][s_cls == cc])
                                           / max(np.nansum(rec["w"]), 1e-9)) for cc in range(K)},
            "sample_share": {CLS[cc]: float((s_cls == cc).mean()) for cc in range(K)},
        }
        if panel:
            arm_out["panel"] = {p["name"]: fid_of(pan_rec[p["name"]]["b"],
                                                  pan_rec[p["name"]]["dlt"],
                                                  pan_rec[p["name"]]["w"]) for p in panel}
        # the oracle-weight allocation (what an unconstrained benchmark would have done)
        arm_out["alloc_share_oracle"] = {
            CLS[cc]: float(np.nansum(rec["w_or"][s_cls == cc])
                           / max(np.nansum(rec["w_or"]), 1e-9)) for cc in range(K)}
        return arm_out

    for name in cfg["arms"]:
        print(f"\n===== burst={burst} arm: {name} =====", flush=True)
        result["arms"][name] = run_arm(name)
        a = result["arms"][name]
        print(f"[{name}] final E[e|pos]: " +
              " ".join(f"{k}={v:.4f}" for k, v in a["final_field"].items()) +
              f"  mean_w={a['mean_w']:.3f}", flush=True)
        if a["fidelity"]:
            f = a["fidelity"]
            print(f"[{name}] b_corr={f['b_corr']:.3f} b_rmse={f['b_rmse']:.4f} "
                  f"delta_corr={f['delta_corr']:.3f} sign_err={f['sign_err_material']} "
                  f"alloc_tv={f['alloc_tv']:.3f}", flush=True)
        if "panel" in a:
            for pn, pf in a["panel"].items():
                print(f"  [panel {pn:16s}] b_corr={pf['b_corr']:.3f} b_rmse={pf['b_rmse']:.4f} "
                      f"delta_corr={pf['delta_corr']:.3f} sign_err={pf['sign_err_material']} "
                      f"alloc_tv={pf['alloc_tv']:.3f}", flush=True)
        checkpoint()

    result["complete"] = True
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("complete\n")
    volume.commit()
    print(f"[save] wrote {outdir}", flush=True)
    return {"burst": burst, "result": result}


@app.local_entrypoint()
def estimability(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # contexts: "cx,cy,phi,noise,pre,name;..."   pre=1 -> present pre-drift (mastered)
    contexts: str = ("-0.55,0.55,1.2,0.0,0,A-rot+; "
                     "0.55,0.55,-1.2,0.0,0,B-rot-; "
                     "-0.55,-0.55,0.6,0.0,0,C-rot~; "
                     "0.55,-0.55,0.0,30.0,0,N-noise; "
                     "0.0,0.0,-0.9,0.0,1,M-mast"),
    ctx_sigma: float = 0.18,
    bursts: str = "200,50,20,5,2,1",     # burst length in batches; K*burst = revisit interval
    arms: str = "fixed,delta_net,delta_ewma,oracle_delta,raw_err",
    panel_arm: str = "fixed",
    n_bpc: int = 200,                     # batches per context (per-context count is FIXED)
    batch: int = 16,
    inner_steps: int = 4,
    adapt_lr: float = 3e-4,
    replay_mode: str = "none",            # none | uniform (the interleaver contrast)
    n_replay: int = 4,
    replay_batch: int = 64,
    # gain law (bridge_assembly's)
    w_clip: float = 4.0,
    w_raw_cap: float = 20.0,
    ewma_alpha: float = 0.05,
    sign_eps: float = 0.25,
    # benchmark
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 3e-3,               # the delta_net arm's b (inside the estimability window)
    bench_ewma_alpha: float = 0.05,       # the delta_ewma arm's b
    bench_pretrain_steps: int = 1500,
    panel_net_lrs: str = "1e-3,3e-3,1e-2,3e-2",
    panel_ewma_alphas: str = "0.01,0.05,0.2",
    panel_scalar_alpha: float = 0.05,
    # oracle grid
    grid_pos: int = 100,
    grid_rep: int = 16,
    knn: int = 8,
    oracle_every: int = 4,
    ckpt_every: int = 20,
    lag_bins: str = "2,5,10,25,60,150,400",
    # env (bridge_assembly / 4c ballistic-competent regime)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    pool_n: int = 9000,
    master_extra_n: int = 3000,
    cal_n: int = 500,
    v_explore: float = 1.2,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_steps: int = 4000,
):
    import os

    CTX = []
    for chunk in [c for c in contexts.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        CTX.append(dict(center=(float(q[0]), float(q[1])), sigma=ctx_sigma,
                        phi=float(q[2]), noise=float(q[3]), pre=bool(int(q[4])), name=q[5]))
    arm_list = [a for a in arms.split(",") if a]
    burst_list = [int(b) for b in bursts.split(",") if b]
    if quick:
        n_bpc = 20; burst_list = [20, 1]; pool_n = 2500; master_extra_n = 800
        fm_steps = 1000; bench_pretrain_steps = 400; fm_hidden = 128; fm_layers = 2
        grid_pos = 40; grid_rep = 8; ckpt_every = 5; oracle_every = 2; cal_n = 250
        tag = tag or "smoke"
    tag = tag or "default"
    assert all(n_bpc % b == 0 for b in burst_list), "burst must divide n_bpc"
    assert panel_arm in arm_list, "panel_arm must be one of arms"

    base = dict(
        tag=tag, seed=seed, contexts=CTX, arms=arm_list, panel_arm=panel_arm,
        arm_est={"delta_net": ("net", bench_lr), "delta_ewma": ("ewma_ctx", bench_ewma_alpha),
                 "oracle_delta": ("oracle", 0.0), "delta_scalar": ("ewma_scalar", panel_scalar_alpha),
                 "delta_frozen": ("frozen", 0.0)},
        n_bpc=n_bpc, batch=batch, inner_steps=inner_steps, adapt_lr=adapt_lr,
        replay_mode=replay_mode, n_replay=n_replay, replay_batch=replay_batch,
        w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha, sign_eps=sign_eps,
        bench_hidden=bench_hidden, bench_layers=bench_layers, bench_lr=bench_lr,
        bench_pretrain_steps=bench_pretrain_steps,
        panel_net_lrs=[float(x) for x in panel_net_lrs.split(",") if x],
        panel_ewma_alphas=[float(x) for x in panel_ewma_alphas.split(",") if x],
        panel_scalar_alpha=panel_scalar_alpha,
        grid_pos=grid_pos, grid_rep=grid_rep, knn=knn, oracle_every=oracle_every,
        ckpt_every=ckpt_every, lag_bins=[int(x) for x in lag_bins.split(",") if x],
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, master_extra_n=master_extra_n, cal_n=cal_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_steps=fm_steps,
    )
    cfgs = []
    for b in burst_list:
        c = dict(base); c["burst"] = b; cfgs.append(c)

    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for out in run_schedule.map(cfgs, order_outputs=True):
        p = os.path.join(localdir, f"burst{out['burst']}.json")
        with open(p, "w") as fh:
            json.dump(out["result"], fh, indent=2, cls=NumpyEncoder)
        print(f"[local] wrote {p}")
