"""bridge_assembly: does delta's hygiene advantage TRANSMIT TO CONTROL when retention is
load-bearing -- with the fully corrected delta assembled live as one mechanism?

Program: `ideas/performance_error_is_the_bridge.md`. Direct parent:
`mjc/plasticity_gain/` (delta as a per-sample plasticity gain). That node found delta-gated
plasticity beats ungated on ballistic re-adaptation, but a raw-error gain MATCHES delta on
control: delta's specific content (withdrawing plasticity from unimprovable noise; best
retention among modulated arms) never reached the control score because the noise decoy was
off-path and nothing in the task ever needed the retained base competence. Both
plasticity_gain and curiosity_control/benchmark_vs_cost independently nominate the same
follow-up: a substrate where MASTERED COMPETENCE STAYS BEHAVIORALLY LOAD-BEARING while
adaptation is demanded elsewhere.

    THE DESIGN (one changed thing vs plasticity_gain: the task geometry).
    Two reach corridors, one rotation region each, on the 4c ballistic-competent substrate:

      * B-mastered (0, -0.30)  phi=-1.2, present PRE-drift: the pretrained FM has mastered
        it (stale probe ~clean there; stale ballistic reach through it ~ceiling). Eval keeps
        visiting it -- retained competence is behaviorally load-bearing by construction.
      * A-drift    (0, +0.30)  phi=+1.2, appears at t=0: the localized drift that demands
        adaptation elsewhere.
      * R-noise    (0, +0.90)  aleatoric amp 30, appears at t=0: the noisy-TV decoy,
        off both reach paths (its behavioral cost must route through ALLOCATION -> churn ->
        retention, not through eval exposure to the noise itself).

    The eval suite reaches through A (adaptation demanded) AND through B (retention
    demanded), reactive + ballistic_cem, at milestone snapshots. Aggregate = mean of the
    two families = a task distribution that keeps visiting mastered regions.

    The mechanism hypothesis space this separates (interpretation strictly a posteriori):
    raw_err gives mastered-region samples near-zero weight (w ~ e, and e is small where you
    are good) while spending a permanent share of its budget on the noise decoy; delta
    holds mastered regions at NOMINAL weight (b(s)~e -> f(0)=1) and boosts them exactly
    when churn lifts e above b(s) -- retention as an active property of the performance
    error. Whether that reaches the control score is what this run measures.

    THE ASSEMBLY (idea doc S11: "the assembly does not have support"). The delta arm runs
    S1's signal fully corrected, live, as one mechanism:
      e     = || (s'-s) - FM2_t(s,u) ||          current FM, at use time
      b(s)  = context-conditional error-predictor net, trained online on (s, e)
              (S13(b)'s correction; bench_lr set INSIDE the estimability window --
              above sampling jitter, below competence drift, per benchmark_vs_cost fig4
              and plasticity_gain #4: 1e-3 sat on the too-slow side, 1e-2 traded AUC;
              3e-3 is the reasoned middle)
      g     = || FM2_t(s,u) - FM1_0(s) ||        live arity gap, current FM2 vs frozen FM1
      gate  = sigmoid((g - g0) / theta_g)        S13(a)'s CENTERED correction, calibrated
              per agency_gate's protocol: g0 = midpoint of active/passive g medians on the
              TRAINING split, theta_g = gap/8. On all-self-generated data the gate is
              expected mostly-open (its job here is to be present, calibrated, and not
              hurt); a NO-GATE counterfactual weight is logged per batch so its
              contribution stays dissectible.
      delta = (b(s) - e) * gate
      w     = f(delta) = exp(-delta/tau), capped; budget-matched by running-mean
              normalization (META_ADAPT #4e discipline; realized mean w reported).

    ARMS (identical pretrained FM, identical stream, identical optimizer; ONLY the
    per-sample plasticity weights differ, at matched average weight):
      * fixed    -- w = 1
      * raw_err  -- w ~ e            ("any error-modulated lr")
      * delta    -- the corrected assembly above

    READOUTS: control ladders per family (drift / mastered / aggregate) x controller
    (reactive / ballistic_cem); FM probes per region class (A-drift, B-mastered, R-noise,
    base, global); allocation (per-class w traces, cumulative shares); the assembled
    signal's own traces (e, b, g, gate, delta, w, w_nogate per class); budget checks.

Run:
    cd experiments/
    modal run mjc/bridge_assembly/bridge_assembly.py::bridge_assembly --quick    # smoke
    modal run --detach mjc/bridge_assembly/bridge_assembly.py::bridge_assembly \
        --tag asm_s0 --seed 0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_bridge_assembly(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    print(f"[setup] device={device} arms={cfg['arms']} T={cfg['T']} batch={cfg['batch']} "
          f"milestones={cfg['milestones']}", flush=True)
    print("[setup] regions: " + "; ".join(
        f"{r['name']}@({r['center'][0]:+.2f},{r['center'][1]:+.2f}) phi={r['phi']:+.2f} "
        f"noise={r['noise']:.0f} pre={r['pre']}" for r in REG), flush=True)
    print(f"[setup] eval families: {cfg['eval_families']}", flush=True)

    def make_env(drifted: bool):
        d = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                 joint_damping=cfg["damping"], pusher_r=0.12,
                 noise_seed=cfg["seed"] + 999)
        regs = [r for r in REG if (drifted or r["pre"])]
        if regs:
            d["rot_regions"] = [dict(center=tuple(r["center"]), sigma=r["sigma"],
                                     phi=float(r["phi"]), noise=float(r["noise"]))
                                for r in regs]
        return PusherEnv(d, with_puck=False)

    env0 = make_env(False)   # pre-drift world: B-mastered only -> pretraining masters it
    env1 = make_env(True)    # post-drift world: B unchanged + A-drift + R-noise

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
        w = np.empty((len(pos), K), np.float32)
        for j, r in enumerate(REG):
            d2 = (pos[:, 0] - r["center"][0]) ** 2 + (pos[:, 1] - r["center"][1]) ** 2
            w[:, j] = np.exp(-d2 / (2.0 * r["sigma"] ** 2))
        return w

    def classify(S):
        w = gates(S)
        return np.where(w.max(1) > 0.3, w.argmax(1), K)

    CLS_NAMES = [r["name"] for r in REG] + ["base"]

    # ---------------- normalization (clean pool; note env0 CONTAINS B-mastered) --------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    # norm stats from the UNIFORM pool only (spatially unbiased, plasticity_gain-comparable)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    # mastery oversampling: pre-drift regions are PRACTICED, not merely visited (the
    # massed-repetition analog). Without this a uniform box gives the B rotation too few
    # informative samples for the pretrained FM to actually master it (smoke: stale
    # B-probe 0.095 ~ A-drift 0.14, i.e. "mastered" competence that was never mastered).
    if cfg["master_extra_n"] > 0:
        mrng = np.random.default_rng(cfg["seed"] + 3)
        for j, r in enumerate(REG):
            if r["pre"]:
                eS, eU, eS2 = collect_region(env0, j, cfg["master_extra_n"], mrng)
                nS = np.concatenate([nS, eS]); nU = np.concatenate([nU, eU])
                nS2 = np.concatenate([nS2, eS2])
                print(f"[pretrain] +{cfg['master_extra_n']} practice samples in {r['name']}",
                      flush=True)
        Y = (nS2 - nS).astype(np.float32)

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
    # 1) PRETRAIN on the pre-drift world (masters B): FM2, FM1, benchmark
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

    # ------------- calibration -------------
    # tau: MAD of (b - e) on a held-out clean probe (as plasticity_gain).
    cS, cU, cS2 = collect_box(env0, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    e_cal = per_sample_err(fm2_0, cS, cU, cS2)
    b_cal = bench_pred(bench_0, cS)
    resid = b_cal - e_cal
    tau = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)

    # CENTERED gate, agency_gate protocol: active vs passive (u=0) arity gap on the
    # TRAINING split; g0 = midpoint of medians, theta_g = gap/8. Self-supervised.
    p1_pool = fm1_delta(fm1_0, nS)
    g_act = np.linalg.norm(fm2_delta(fm2_0, nS, nU) - p1_pool, axis=1)
    g_pas = np.linalg.norm(fm2_delta(fm2_0, nS, np.zeros_like(nU)) - p1_pool, axis=1)
    m_a, m_p = float(np.median(g_act)), float(np.median(g_pas))
    g0 = 0.5 * (m_a + m_p)
    theta_g = max((m_a - m_p) / 8.0, 1e-9)
    allv = np.concatenate([g_act, g_pas])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    auroc = float((ranks[:len(g_act)].sum() - len(g_act) * (len(g_act) + 1) / 2.0)
                  / (len(g_act) * len(g_pas) + 1e-12))

    def gate_fn(g):
        return (1.0 / (1.0 + np.exp(-(g - g0) / theta_g))).astype(np.float32)

    gate_act = gate_fn(g_act); gate_pas = gate_fn(g_pas)
    print(f"[calib] pretrain fm_err(clean)={e_cal.mean():.4f}  tau={tau:.4f}", flush=True)
    print(f"[calib/gate] g medians: active={m_a:.5f} passive={m_p:.5f} -> g0={g0:.5f} "
          f"theta_g={theta_g:.6f}  AUROC={auroc:.4f}  degenerate={m_a <= m_p}", flush=True)
    print(f"[calib/gate] gate(active) p05/p50/p95="
          f"{np.percentile(gate_act, [5, 50, 95]).round(3)}  "
          f"gate(passive) p05/p50/p95={np.percentile(gate_pas, [5, 50, 95]).round(3)}", flush=True)

    # ===================================================================== #
    # 2) DRIFT + the shared stream (identical data for every arm)
    # ===================================================================== #
    sS, sU, sS2 = collect_box(env1, cfg["T"], np.random.default_rng(cfg["seed"] + 500))
    s_cls = classify(sS)
    frac = {CLS_NAMES[c]: float((s_cls == c).mean()) for c in range(K + 1)}
    print(f"[stream] T={cfg['T']} class fractions: " +
          " ".join(f"{k}={v:.3f}" for k, v in frac.items()), flush=True)

    # frozen-FM1 predictions over the stream (the live g's reference leg), plus the
    # PRETRAINED-pair g for the dissection trace (what plasticity_gain used).
    p1_stream = fm1_delta(fm1_0, sS)
    g_pre = np.linalg.norm(fm2_delta(fm2_0, sS, sU) - p1_stream, axis=1).astype(np.float32)
    g_pas_stream = np.linalg.norm(
        fm2_delta(fm2_0, sS, np.zeros_like(sU)) - p1_stream, axis=1).astype(np.float32)
    print(f"[stream/gate] pretrained-pair gate p10/p50/p90="
          f"{np.percentile(gate_fn(g_pre), [10, 50, 90]).round(3)}  "
          f"passive-counterfactual gate p50={np.median(gate_fn(g_pas_stream)):.4f}", flush=True)

    # probes in the drifted world (fixed; identical for every arm)
    prng = np.random.default_rng(cfg["seed"] + 600)
    probes = {"global": collect_box(env1, cfg["probe_n"], prng)}
    for j in range(K):
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

    # ===================================================================== #
    # 3) CEILING FM + CONTROL GRADING (two reach families)
    # ===================================================================== #
    dS, dU, dS2 = collect_box(env1, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 20))
    fm_ceil = _mlp(cfg["seed"] + 43, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
    train_steps(fm_ceil, optc, dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 303))

    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]; Bev = cfg["n_eval"]

    def mpc_plan(net, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, H, 2), np.float32); sig = np.full((Bn, H, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, H, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
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

    # eval starts/goals per family: a corridor reach along y = y_c, straight through
    # that family's region center (region A for 'drift', region B for 'mastered').
    fam_eval = {}
    for fi, (fam, y_c) in enumerate(cfg["eval_families"].items()):
        ev_rng = np.random.default_rng(cfg["seed"] + 7 + 13 * fi)
        sgn = ev_rng.choice([-1.0, 1.0], Bev).astype(np.float32)
        starts = np.concatenate([
            np.stack([sgn * cr, np.full(Bev, y_c, np.float32)], 1)
            + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32),
            ev_rng.normal(0, cfg["v0_std"], (Bev, 2)).astype(np.float32)], 1)
        goals = np.stack([-sgn * cr, np.full(Bev, y_c, np.float32)], 1) \
            + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32)
        fam_eval[fam] = (starts, goals)

    def rollout(net, replan_every, starts, goals):
        states = starts.copy(); plan = None
        rng = np.random.default_rng(cfg["seed"] + 7000 + replan_every)
        for step in range(H):
            if step % replan_every == 0:
                plan = mpc_plan(net, states, goals, rng)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for bb in range(Bev):
                env1.set_state(states[bb, :2].astype(np.float64), states[bb, 2:].astype(np.float64))
                s2, _ = env1.step(acts[bb], fs); states[bb] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - goals, axis=1)))

    def grade(net):
        rec = {"fm": probe_errs(net)}
        for fam, (starts, goals) in fam_eval.items():
            if "reactive" in cfg["controllers"]:
                rec[f"{fam}/reactive"] = rollout(net, 1, starts, goals)
            if "ballistic_cem" in cfg["controllers"]:
                rec[f"{fam}/ballistic_cem"] = rollout(net, H, starts, goals)
        for c in cfg["controllers"]:
            rec[f"agg/{c}"] = float(np.mean([rec[f"{fam}/{c}"] for fam in fam_eval]))
        return rec

    def fmt_grade(rec):
        return "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "fm")

    outdir = os.path.join(DATA_DIR, "bridge_assembly", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)

    calibration = {
        "tau": tau, "g0": g0, "theta_g": theta_g,
        "g_median_active": m_a, "g_median_passive": m_p,
        "gate_auroc_train": auroc,
        "gate_active_p05_p50_p95": np.percentile(gate_act, [5, 50, 95]).tolist(),
        "gate_passive_p05_p50_p95": np.percentile(gate_pas, [5, 50, 95]).tolist(),
        "stream_gate_pretrained_p10_p50_p90": np.percentile(gate_fn(g_pre), [10, 50, 90]).tolist(),
        "stream_gate_passive_p50": float(np.median(gate_fn(g_pas_stream))),
        "pretrain_fm_err_clean": float(e_cal.mean()),
        "stream_class_fractions": frac,
        "stale_probe": stale_probe,
    }

    result = {"config": cfg, "complete": False, "calibration": calibration, "arms": {}}

    def checkpoint():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    print("\n===== grading stale + ceiling =====", flush=True)
    g_stale = grade(fm2_0); g_stale["transitions"] = 0
    print("[grade stale   m=0] " + fmt_grade(g_stale), flush=True)
    g_ceil = grade(fm_ceil); g_ceil["transitions"] = -1
    print("[grade ceiling    ] " + fmt_grade(g_ceil), flush=True)
    result["stale"] = g_stale; result["ceiling"] = g_ceil
    checkpoint()

    # ===================================================================== #
    # 4) ARMS -- per-sample plasticity gains over the same stream, graded per arm
    # ===================================================================== #
    B = cfg["batch"]; n_cycles = cfg["T"] // B
    milestone_set = set(cfg["milestones"])
    w_clip = cfg["w_clip"]; ew_a = cfg["ewma_alpha"]
    LOGQ = ("w", "e", "b", "delta", "g", "gate", "wng")

    def run_arm(arm):
        fm = copy.deepcopy(fm2_0)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
        bnet = copy.deepcopy(bench_0)
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        rng = np.random.default_rng(cfg["seed"] + 700)      # same replay draws per arm
        w_norm = 1.0 if arm == "delta" else float(e_pool.mean())
        snaps = {}
        trace = {"t": [], **{f"probe_{k}": [] for k in probes}}
        blog = {"t": []}
        for nm in CLS_NAMES:
            for q in LOGQ:
                blog[f"{q}_{nm}"] = []
        cum_w = np.zeros(K + 1, np.float64); cum_n = np.zeros(K + 1, np.float64)
        w_sum = 0.0; w_cnt = 0

        def weights_for(idx):
            """Per-sample plasticity weights for stream rows idx under this arm's rule,
            with the arm's CURRENT fm/benchmark -- the signal is assembled at use time."""
            nonlocal w_norm
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            pred = fm2_delta(fm, S, U)
            e = np.linalg.norm(pred - (S2 - S), axis=1).astype(np.float32)
            g = np.linalg.norm(pred - p1_stream[idx], axis=1).astype(np.float32)  # live g
            gt = gate_fn(g)
            if arm == "fixed":
                z = np.zeros_like(e)
                return np.ones(len(idx), np.float32), e, z, z, g, gt, np.full_like(e, np.nan)
            if arm == "raw_err":
                w_raw = e
                b = np.zeros_like(e); dlt = np.zeros_like(e); wng = np.full_like(e, np.nan)
            else:
                b = bench_pred(bnet, S)
                dlt = ((b - e) * gt).astype(np.float32)
                w_raw = np.exp(np.clip(-dlt / tau, -30.0,
                                       np.log(cfg["w_raw_cap"]))).astype(np.float32)
                # no-gate counterfactual (logged only; dissects the gate's contribution)
                wng = np.exp(np.clip(-(b - e) / tau, -30.0,
                                     np.log(cfg["w_raw_cap"]))).astype(np.float32)
            w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
            w_hat = np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)
            return w_hat, e, b, dlt, g, gt, wng

        def gated_step(idx):
            nonlocal w_sum, w_cnt
            w_hat, e, b, dlt, g, gt, wng = weights_for(idx)
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            Xb = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w_hat, device=device)
            fm.train(); opt.zero_grad()
            (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward()
            opt.step(); fm.eval()
            if arm != "fixed":
                bench_step(bnet, bopt, S, e)                # b(s) tracks the own-error field
            w_sum += float(w_hat.sum()); w_cnt += len(idx)
            return w_hat, e, b, dlt, g, gt, wng

        if 0 in milestone_set:
            snaps[0] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
        for c in range(n_cycles):
            inc = np.arange(c * B, (c + 1) * B)
            w_hat, e, b, dlt, g, gt, wng = gated_step(inc)   # the just-produced batch
            cls = s_cls[inc]
            blog["t"].append((c + 1) * B)
            for ci, nm in enumerate(CLS_NAMES):
                m = cls == ci
                for q, arr in zip(LOGQ, (w_hat, e, b, dlt, g, gt, wng)):
                    blog[f"{q}_{nm}"].append(float(np.nanmean(arr[m])) if m.any() else np.nan)
                cum_w[ci] += float(w_hat[m].sum()); cum_n[ci] += int(m.sum())
            for _ in range(cfg["n_replay"]):                 # replay from the stream so far
                ridx = rng.integers(0, (c + 1) * B, size=cfg["replay_batch"])
                gated_step(ridx)
            t_now = (c + 1) * B
            if c % cfg["probe_every"] == 0 or t_now in milestone_set:
                pe = probe_errs(fm)
                trace["t"].append(t_now)
                for k, v in pe.items():
                    trace[f"probe_{k}"].append(v)
            if t_now in milestone_set:
                snaps[t_now] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
                print(f"[{arm} m={t_now:5d}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in pe.items()), flush=True)
        budget = {"mean_w": w_sum / max(w_cnt, 1),
                  "cum_w_share": {CLS_NAMES[ci]: float(cum_w[ci] / max(cum_w.sum(), 1e-9))
                                  for ci in range(K + 1)},
                  "sample_share": {CLS_NAMES[ci]: float(cum_n[ci] / max(cum_n.sum(), 1e-9))
                                   for ci in range(K + 1)}}
        print(f"[{arm}] realized mean w={budget['mean_w']:.3f}  cum-weight shares: " +
              " ".join(f"{k}={v:.3f}" for k, v in budget["cum_w_share"].items()), flush=True)
        return snaps, trace, blog, budget

    lad_net = _mlp(0, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])   # shell for snapshots
    for arm in cfg["arms"]:
        print(f"\n===== arm: {arm} =====", flush=True)
        snaps, trace, blog, budget = run_arm(arm)
        result["arms"][arm] = {"trace": trace, "blog": blog, "budget": budget}
        checkpoint()
        lad = []
        for m in sorted(snaps):
            if m == 0:
                lad.append(dict(g_stale)); continue
            lad_net.load_state_dict(snaps[m]); lad_net.to(device).eval()
            rec = grade(lad_net); rec["transitions"] = int(m)
            lad.append(rec)
            print(f"[grade {arm:8s} m={m:5d}] " + fmt_grade(rec), flush=True)
            result["arms"][arm]["ladder"] = lad
            checkpoint()                                     # per-milestone durability
        result["arms"][arm]["ladder"] = lad
        checkpoint()
        del snaps

    result["complete"] = True
    figures = _make_figures(result)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("complete\n")
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures + done.txt to {outdir}", flush=True)
    return {"results": result, "figures": figures}


def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    ARMC = {"fixed": "#495057", "raw_err": "#e8590c", "delta": "#7048e8"}
    arms = R["config"]["arms"]; REG = R["config"]["regions"]
    ctrls = R["config"]["controllers"]
    fams = list(R["config"]["eval_families"].keys()) + ["agg"]
    cls_names = [r["name"] for r in REG] + ["base"]
    CLSC = {"base": "#868e96"}
    reds = ["#c92a2a", "#f08c00"]; blues = ["#1971c2", "#0c8599"]
    ri = bi = 0
    for r in REG:
        if r["noise"] > 0:
            CLSC[r["name"]] = blues[bi % 2]; bi += 1
        else:
            CLSC[r["name"]] = reds[ri % 2]; ri += 1
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    def _logx(t):
        return np.where(np.asarray(t, float) == 0, R["config"]["batch"] / 2.0, np.asarray(t, float))

    def _sm(y, n_div=60, fill=0.0):
        y = np.asarray(y, float)
        n_sm = max(len(y) // n_div, 1); ker = np.ones(n_sm) / n_sm
        return np.convolve(np.nan_to_num(y, nan=fill), ker, mode="same")

    # fig1: control recovery -- rows = controllers, cols = families (+ aggregate)
    fig, axes = plt.subplots(len(ctrls), len(fams),
                             figsize=(4.6 * len(fams), 3.9 * len(ctrls)), squeeze=False)
    for rci, c in enumerate(ctrls):
        for fci, fam in enumerate(fams):
            ax = axes[rci][fci]; key = f"{fam}/{c}"
            for a in arms:
                lad = R["arms"][a].get("ladder", [])
                t = [r["transitions"] for r in lad]; y = [r.get(key, np.nan) for r in lad]
                ax.plot(_logx(t), y, "o-", color=ARMC.get(a, "#333"), lw=2.0, ms=4, label=a)
            ax.axhline(R["ceiling"].get(key, np.nan), color="#999", ls=":", lw=1.5,
                       label="ceiling (fresh FM)")
            ax.set_xscale("log")
            ax.set_title(f"{fam} / {c}")
            if fci == 0:
                ax.set_ylabel(f"{c} goal-dist (lower=better)")
            if rci == len(ctrls) - 1:
                ax.set_xlabel("re-adaptation transitions (reward-free)")
            if rci == 0 and fci == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Control recovery per reach family (drift = adaptation demanded; "
                 "mastered = retention demanded; matched avg. budget)", fontsize=10)
    figs["fig1_recovery.png"] = _save(fig)

    # fig2: FM probe traces per region class (mechanism)
    panels = [(nm, [nm]) for nm in cls_names] + [("global", ["global"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.9))
    for pi, (title, keys) in enumerate(panels):
        ax = axes[pi]
        for a in arms:
            tr = R["arms"][a]["trace"]
            y = np.nanmean([tr[f"probe_{k}"] for k in keys], axis=0)
            ax.plot(_logx(tr["t"]), y, "-", color=ARMC.get(a, "#333"), lw=1.8, label=a)
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("transitions")
        if pi == 0:
            ax.set_ylabel("FM probe error"); ax.legend(fontsize=8)
    fig.suptitle("Probe error by region class (B-mastered = the retention trace)", fontsize=10)
    figs["fig2_fm_traces.png"] = _save(fig)

    # fig3: allocation (per-class weight traces, gated arms)
    gated = [a for a in arms if a != "fixed"]
    if gated:
        fig, axes = plt.subplots(1, len(gated), figsize=(5.2 * len(gated), 4.0), squeeze=False)
        for gi, a in enumerate(gated):
            ax = axes[0][gi]; bl = R["arms"][a]["blog"]
            t = np.asarray(bl["t"], float)
            for nm in cls_names:
                ax.plot(t, _sm(bl[f"w_{nm}"], fill=1.0), "-", color=CLSC[nm], lw=1.6, label=nm)
            ax.axhline(1.0, color="#bbb", ls=":", lw=1)
            ax.set_title(a); ax.set_xlabel("transitions"); ax.set_xscale("log")
            if gi == 0:
                ax.set_ylabel("mean plasticity weight $\\hat{w}$"); ax.legend(fontsize=7)
        fig.suptitle("Where plasticity was spent (per-class mean weight, smoothed)", fontsize=10)
        figs["fig3_allocation.png"] = _save(fig)

    # fig4: the assembled signal (delta arm): delta per class; e vs b(s); gate + w vs w_nogate
    if "delta" in arms:
        bl = R["arms"]["delta"]["blog"]; t = np.asarray(bl["t"], float)
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
        for nm in cls_names:
            axes[0].plot(t, _sm(bl[f"delta_{nm}"]), color=CLSC[nm], lw=1.6, label=nm)
        axes[0].axhline(0, color="#bbb", ls=":", lw=1)
        axes[0].set_xscale("log"); axes[0].set_title("delta = (b(s) - e) * gate, per class")
        axes[0].set_xlabel("transitions"); axes[0].set_ylabel("delta"); axes[0].legend(fontsize=7)
        for nm in cls_names:
            for q, ls in (("e", "-"), ("b", "--")):
                axes[1].plot(t, _sm(bl[f"{q}_{nm}"]), ls, color=CLSC[nm], lw=1.4,
                             label=f"{nm} {'error' if q == 'e' else 'benchmark'}")
        axes[1].set_xscale("log"); axes[1].set_title("error (solid) vs benchmark b(s) (dashed)")
        axes[1].set_xlabel("transitions"); axes[1].legend(fontsize=6)
        for nm in cls_names:
            axes[2].plot(t, _sm(bl[f"gate_{nm}"], fill=np.nan), color=CLSC[nm], lw=1.4, label=nm)
        axes[2].set_xscale("log"); axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_title("gate sigma((g - g0)/theta_g), per class (live g)")
        axes[2].set_xlabel("transitions"); axes[2].legend(fontsize=7)
        fig.suptitle("The assembled performance-error signal over re-adaptation (arm: delta)",
                     fontsize=10)
        figs["fig4_delta_mech.png"] = _save(fig)

        # fig5: gate dissection -- w vs the no-gate counterfactual, per class
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for nm in cls_names:
            axes[0].plot(t, _sm(bl[f"w_{nm}"], fill=1.0), "-", color=CLSC[nm], lw=1.5,
                         label=f"{nm} gated")
            axes[0].plot(t, _sm(bl[f"wng_{nm}"], fill=1.0), "--", color=CLSC[nm], lw=1.2,
                         label=f"{nm} no-gate")
        axes[0].set_xscale("log")
        axes[0].set_title("raw weight f(delta): gated (solid) vs no-gate counterfactual (dashed)")
        axes[0].set_xlabel("transitions"); axes[0].set_ylabel("f(delta) (pre-normalization)")
        axes[0].legend(fontsize=6)
        cal = R["calibration"]
        for nm in cls_names:
            axes[1].plot(t, _sm(bl[f"g_{nm}"], fill=np.nan), color=CLSC[nm], lw=1.4, label=nm)
        axes[1].axhline(cal["g0"], color="#444", ls="--", lw=1.2)
        axes[1].text(t[2], cal["g0"], " g0 (train calib)", fontsize=7, color="#444")
        axes[1].axhline(cal["g_median_passive"], color="#999", ls=":", lw=1.0)
        axes[1].set_xscale("log"); axes[1].set_title("live arity gap g per class")
        axes[1].set_xlabel("transitions"); axes[1].legend(fontsize=7)
        fig.suptitle("Gate dissection: present, calibrated (centered), and its effect on w",
                     fontsize=10)
        figs["fig5_gate.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def bridge_assembly(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "fixed,raw_err,delta",
    controllers: str = "reactive,ballistic_cem",
    # regions: "cx,cy,phi,noise,pre,name;..."  pre=1 -> present pre-drift (mastered).
    # B-mastered is pre-drift and on the 'mastered' eval corridor; A-drift + R-noise
    # appear at t=0 (the Type-2 event). R-noise is off both corridors.
    regions: str = ("0.0,0.30,1.2,0.0,0,A-drift; "
                    "0.0,-0.30,-1.2,0.0,1,B-mastered; "
                    "0.0,0.90,0.0,30.0,0,R-noise"),
    region_sigma: float = 0.18,
    # eval families: "name:y_center;..." -- corridor reach along y = y_center
    eval_families: str = "drift:0.30; mastered:-0.30",
    # online adaptation (plasticity_gain gain2 settings)
    total_t: int = 16384,
    batch: int = 16,
    n_replay: int = 4,
    replay_batch: int = 64,
    adapt_lr: float = 3e-4,
    milestones: str = "0,256,512,1024,2048,4096,8192,16384",
    probe_every: int = 16,
    # gain law
    w_clip: float = 4.0,
    w_raw_cap: float = 20.0,
    ewma_alpha: float = 0.05,
    # benchmark net b(s): bench_lr inside the estimability window (see docstring)
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 3e-3,
    bench_pretrain_steps: int = 1500,
    # env (the 4c/plasticity_gain regime: ballistic-competent at damping 2)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    pool_n: int = 9000,
    master_extra_n: int = 3000,   # extra pretrain practice samples per pre-drift region
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

    arm_list = [a for a in arms.split(",") if a]
    ctrl_list = [c for c in controllers.split(",") if c]
    REG = []
    for chunk in [c for c in regions.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        REG.append(dict(center=(float(q[0]), float(q[1])), sigma=region_sigma,
                        phi=float(q[2]), noise=float(q[3]), pre=bool(int(q[4])), name=q[5]))
    fams = {}
    for chunk in [c for c in eval_families.split(";") if c.strip()]:
        nm, y = chunk.split(":")
        fams[nm.strip()] = float(y)
    ms = [int(x) for x in milestones.split(",") if x]
    if quick:
        total_t = 2048; ms = [0, 512, 2048]; pool_n = 2500; probe_n = 250
        fm_steps = 1200; bench_pretrain_steps = 500; fm_hidden = 128; fm_layers = 2
        n_eval = 12; k_shoot = 96; probe_every = 8; replay_batch = 32
        master_extra_n = 800
        tag = tag or "smoke"
    tag = tag or "default"
    assert all(m % batch == 0 for m in ms), "milestones must be multiples of batch"
    assert total_t == max(ms), "total_t should equal the last milestone"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, controllers=ctrl_list, regions=REG,
        eval_families=fams,
        T=total_t, batch=batch, n_replay=n_replay, replay_batch=replay_batch,
        adapt_lr=adapt_lr, milestones=ms,
        probe_every=probe_every, w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha,
        bench_hidden=bench_hidden, bench_layers=bench_layers,
        bench_lr=bench_lr, bench_pretrain_steps=bench_pretrain_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, master_extra_n=master_extra_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_steps=fm_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_bridge_assembly.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "bridge_assembly_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
