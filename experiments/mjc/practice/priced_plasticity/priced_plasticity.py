"""priced_plasticity: stop matching delta's budget -- compare it against the WHOLE
one-parameter family of ungated uniform learning rates, on a substrate where plasticity
is COSTLY.

Program: `ideas/practice_manufactures_its_own_credit.md` (component 2, metering) and
`ideas/performance_error_is_the_bridge.md` S14(b). Direct parent (forked wholesale):
`mjc/bridge_assembly/bridge_assembly.py` -- same two-corridor geometry, same corrected
delta, same gain law, same calibration protocol. THREE things change, plus the grading
instrument.

    WHY. delta = (b(s) - e) * sigmoid((g - g0)/theta) has now lost to ungated uniform
    plasticity in bridge_assembly (3 seeds), difficulty_sweep (8 cells x 3 seeds) and
    practice/estimability (every burst). The consumption law it is modelled on (Kim,
    Parvin & Ivry 2019) is ATTENUATION ON SUCCESS -- a protection mechanism. Protection
    only has value when plasticity is costly. In every node so far it is free:
      * fm_hidden=256/fm_layers=3, far above the capacity-competition boundary
        (meta_adapt #4d/#4e) -> learning A never costs you B;
      * n_replay=4 -> a uniform replay buffer repairs whatever gets eroded
        (estimability Q4: uniform replay removes ~all of the schedule's forgetting);
      * the running-mean weight normalizer -> when delta withdraws from mastered
        content the normalizer scales everything else up, so the budget is
        REDISTRIBUTED, never SAVED.

    THE GRADING INSTRUMENT (the methodological point). Matched-average-weight blocks a
    real confound (META_ADAPT #4e: delta winning by simply learning more) but it blocks
    it by forbidding delta to choose its own total spend -- which is the mechanism under
    test. The better control is post hoc: sweep the ungated uniform learning rate across
    a range, trace the RETENTION x ADAPTATION Pareto frontier that single scalar knob
    produces, and ask where delta lands relative to it. If delta is secretly just
    "learn less/more", some uniform lr reproduces its tradeoff and delta sits ON the
    frontier. If delta is genuinely allocation, no scalar reproduces it.

    THE SPEND/ALLOCATION FACTORIZATION (and the Adam caveat that forces it). Under Adam
    a GLOBAL rescaling of the loss is (near-)exactly cancelled by the second-moment
    normalization, so the parent's per-sample weights only ever expressed RELATIVE
    allocation; the running-mean normalizer's "budget matching" acts only through the
    weight clip. To let delta actually spend more or less, this fork factorizes the
    weight field into

        w_rel = w_hat / mean(w_hat)        (relative allocation; what Adam can see)
        spend = mean(w_hat)                (total plasticity; applied to the LR)

    and runs the step as loss = mean(huber_ps * w_rel) with Adam lr = base_lr * spend.
    `spend_mode` selects: `raw_adam` (the parent's exact pipeline: loss weights = w_hat,
    lr = base_lr -- kept so the parent is reproducible inside this fork), `free` (delta
    chooses its own spend), `unit` (allocation only, spend clamped to 1 -- the
    decomposition arm).

    `norm_mode` selects whether the normalizer w_norm adapts (`ewma`, the parent) or is
    frozen at its pretrain value (`static`, so attenuation is SAVED rather than
    redistributed). Under `static` the delta arm is the literal Kim et al. law:
    f(0) = 1 at benchmark, boost when worse, attenuate toward 0 when better.

    ARMS. `fixed` at a grid of learning rates traces the frontier; `delta` (free spend),
    `delta` (unit spend) and `raw_err` are single points on the same plane. delta itself
    is UNCHANGED from the parent -- this node isolates the pricing, not a new signal.

    MODES.
      * `--mode calibrate` -- the precondition pass, per the repo's calibrate->measure
        idiom. FM probes only (no control grading, so it is cheap): a grid over
        fm_hidden x n_replay x adapt_lr on the `fixed` arm, to check (i) that the cost
        knobs produce meaningful forgetting ON THIS GEOMETRY and (ii) that the uniform-lr
        sweep spans a frontier rather than a point. Also runs the Adam scale-invariance
        probe (`fixed` with a constant weight multiplier under `raw_adam`).
      * `--mode main` -- the full comparison with control grading.

Run (from experiments/):
    modal run mjc/practice/priced_plasticity/priced_plasticity.py::priced --quick
    python3 mjc/practice/priced_plasticity/launch_detached.py --mode calibrate --tag cal_s0
    python3 mjc/practice/priced_plasticity/launch_detached.py --mode main --tag pp_s0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


# ------------------------------------------------------------------ arm spec parsing
def parse_arm(token, d_lr, d_spend, d_norm, d_replay):
    """`kind[@lr][:spend][:norm][:replay]` -> spec dict.  Name = the token verbatim."""
    parts = token.split(":")
    head = parts[0]
    kind, _, lr = head.partition("@")
    spec = dict(name=token, kind=kind.strip(),
                lr=float(lr) if lr else d_lr,
                spend_mode=d_spend, norm_mode=d_norm, n_replay=d_replay, w_mult=1.0)
    for p in parts[1:]:
        p = p.strip()
        if p in ("raw_adam", "free", "unit"):
            spec["spend_mode"] = p
        elif p in ("ewma", "static"):
            spec["norm_mode"] = p
        elif p.startswith("r"):
            spec["n_replay"] = int(p[1:])
        elif p.startswith("x"):
            spec["w_mult"] = float(p[1:])
        else:
            raise ValueError(f"unparsed arm qualifier {p!r} in {token!r}")
    assert spec["kind"] in ("fixed", "raw_err", "delta"), spec["kind"]
    return spec


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_priced(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    MODE = cfg["mode"]
    print(f"[setup] device={device} mode={MODE} T={cfg['T']} batch={cfg['batch']}", flush=True)
    print("[setup] regions: " + "; ".join(
        f"{r['name']}@({r['center'][0]:+.2f},{r['center'][1]:+.2f}) phi={r['phi']:+.2f} "
        f"noise={r['noise']:.0f} pre={r['pre']}" for r in REG), flush=True)
    for s in cfg["arm_specs"]:
        print(f"[setup/arm] {s['name']:22s} kind={s['kind']:8s} lr={s['lr']:.2e} "
              f"spend={s['spend_mode']:9s} norm={s['norm_mode']:6s} replay={s['n_replay']} "
              f"h={s['fm_hidden']}x{s['fm_layers']} wx={s['w_mult']:.2f}", flush=True)

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

    # ---------------- pools + normalization (identical to the parent) ----------------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

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
    # 1) THE SHARED STREAM + PROBES (capacity-independent: identical for every cell)
    # ===================================================================== #
    sS, sU, sS2 = collect_box(env1, cfg["T"], np.random.default_rng(cfg["seed"] + 500))
    s_cls = classify(sS)
    frac = {CLS_NAMES[c]: float((s_cls == c).mean()) for c in range(K + 1)}
    print(f"[stream] T={cfg['T']} class fractions: " +
          " ".join(f"{k}={v:.3f}" for k, v in frac.items()), flush=True)

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

    # calibration probe pool (clean, pre-drift) for tau
    cS, cU, cS2 = collect_box(env0, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    # ceiling-FM pool (post-drift)
    dS, dU, dS2 = collect_box(env1, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 20))

    # ===================================================================== #
    # 2) THE PER-CAPACITY STACK (pretrained FM2/FM1/b(s), calibration, ceiling)
    # ===================================================================== #
    _stacks = {}

    def build_stack(fm_hidden, fm_layers):
        key = (fm_hidden, fm_layers)
        if key in _stacks:
            return _stacks[key]
        print(f"\n===== building stack h={fm_hidden} L={fm_layers} =====", flush=True)
        fm2_0 = _mlp(cfg["seed"] + 40, 6, 4, fm_hidden, fm_layers)
        opt2 = torch.optim.Adam(fm2_0.parameters(), lr=cfg["fm_lr"])
        train_steps(fm2_0, opt2, nS, nU, nS2, cfg["fm_steps"],
                    np.random.default_rng(cfg["seed"] + 300))

        fm1_0 = _mlp(cfg["seed"] + 41, 4, 4, fm_hidden, fm_layers)
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
        bench_0 = _mlp(cfg["seed"] + 42, 4, 1, cfg["bench_hidden"], cfg["bench_layers"])
        bopt_0 = torch.optim.Adam(bench_0.parameters(), lr=cfg["bench_lr"])
        brngb = np.random.default_rng(cfg["seed"] + 302)
        for _ in range(cfg["bench_pretrain_steps"]):
            idx = brngb.integers(0, len(nS), size=256)
            bench_step(bench_0, bopt_0, nS[idx], e_pool[idx])

        # tau: MAD of (b - e) on a held-out clean probe (parent protocol)
        e_cal = per_sample_err(fm2_0, cS, cU, cS2)
        b_cal = bench_pred(bench_0, cS)
        resid = b_cal - e_cal
        tau = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)

        # CENTERED gate, agency_gate protocol, on the TRAINING split
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

        p1_stream = fm1_delta(fm1_0, sS)
        stale_probe = probe_errs(fm2_0)
        print(f"[calib h={fm_hidden}] fm_err(clean)={e_cal.mean():.4f} tau={tau:.5f} "
              f"gate AUROC={auroc:.4f} g0={g0:.5f}", flush=True)
        print(f"[stale h={fm_hidden}] " +
              " ".join(f"{k}={v:.4f}" for k, v in stale_probe.items()), flush=True)

        fm_ceil = None; ceil_grade = None
        if cfg["grade_control"]:
            fm_ceil = _mlp(cfg["seed"] + 43, 6, 4, fm_hidden, fm_layers)
            optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
            train_steps(fm_ceil, optc, dS, dU, dS2, cfg["fm_steps"],
                        np.random.default_rng(cfg["seed"] + 303))

        st = dict(fm_hidden=fm_hidden, fm_layers=fm_layers, fm2_0=fm2_0, fm1_0=fm1_0,
                  bench_0=bench_0, tau=tau, gate_fn=gate_fn, p1_stream=p1_stream,
                  e_pool_mean=float(e_pool.mean()), stale_probe=stale_probe,
                  fm_ceil=fm_ceil, ceil_grade=ceil_grade,
                  calib=dict(tau=tau, g0=g0, theta_g=theta_g, gate_auroc_train=auroc,
                             pretrain_fm_err_clean=float(e_cal.mean()),
                             e_pool_mean=float(e_pool.mean()),
                             stale_probe=stale_probe))
        _stacks[key] = st
        return st

    # ===================================================================== #
    # 3) CONTROL GRADING (two reach families, reactive + ballistic CEM)
    # ===================================================================== #
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

    outdir = os.path.join(DATA_DIR, "priced_plasticity", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    result = {"config": cfg, "complete": False, "stream_class_fractions": frac,
              "stacks": {}, "arms": {}}

    def checkpoint():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ===================================================================== #
    # 4) THE ARM RUNNER -- weights factorized into (relative allocation) x (spend)
    # ===================================================================== #
    B = cfg["batch"]; n_cycles = cfg["T"] // B
    milestone_set = set(cfg["milestones"])
    w_clip = cfg["w_clip"]; ew_a = cfg["ewma_alpha"]
    LOGQ = ("w", "e", "b", "delta", "g", "gate")

    def run_arm(spec, stack):
        kind = spec["kind"]; base_lr = spec["lr"]
        tau = stack["tau"]; gate_fn = stack["gate_fn"]; p1_stream = stack["p1_stream"]
        fm = copy.deepcopy(stack["fm2_0"])
        opt = torch.optim.Adam(fm.parameters(), lr=base_lr)
        bnet = copy.deepcopy(stack["bench_0"])
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        rng = np.random.default_rng(cfg["seed"] + 700)      # same replay draws per arm
        # w_norm: parent's initialization; `static` freezes it, `ewma` lets it track.
        w_norm = 1.0 if kind == "delta" else float(stack["e_pool_mean"])
        if kind == "fixed":
            w_norm = 1.0
        snaps = {}
        trace = {"t": [], **{f"probe_{k}": [] for k in probes}}
        blog = {"t": [], "spend": [], "clipfrac": []}
        for nm in CLS_NAMES:
            for q in LOGQ:
                blog[f"{q}_{nm}"] = []
        cum_w = np.zeros(K + 1, np.float64); cum_n = np.zeros(K + 1, np.float64)
        w_sum = 0.0; w_cnt = 0; spend_sum = 0.0; spend_cnt = 0; clip_sum = 0.0

        def weights_for(idx):
            nonlocal w_norm
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            pred = fm2_delta(fm, S, U)
            e = np.linalg.norm(pred - (S2 - S), axis=1).astype(np.float32)
            g = np.linalg.norm(pred - p1_stream[idx], axis=1).astype(np.float32)  # live g
            gt = gate_fn(g)
            z = np.zeros_like(e)
            if kind == "fixed":
                w_raw = np.full(len(idx), spec["w_mult"], np.float32); b = z; dlt = z
            elif kind == "raw_err":
                w_raw = (e * spec["w_mult"]).astype(np.float32); b = z; dlt = z
            else:
                b = bench_pred(bnet, S)
                dlt = ((b - e) * gt).astype(np.float32)
                w_raw = (spec["w_mult"] * np.exp(np.clip(
                    -dlt / tau, -30.0, np.log(cfg["w_raw_cap"])))).astype(np.float32)
            if spec["norm_mode"] == "ewma" and kind != "fixed":
                w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
            w_hat = np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)
            return w_hat, e, b, dlt, g, gt

        def gated_step(idx):
            nonlocal w_sum, w_cnt, spend_sum, spend_cnt, clip_sum
            w_hat, e, b, dlt, g, gt = weights_for(idx)
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            spend = float(w_hat.mean())
            if spec["spend_mode"] == "raw_adam":
                w_use = w_hat; lr_now = base_lr           # the parent's exact pipeline
            elif spec["spend_mode"] == "unit":
                w_use = w_hat / max(spend, 1e-8); lr_now = base_lr
            else:                                          # "free"
                w_use = w_hat / max(spend, 1e-8); lr_now = base_lr * spend
            for gp in opt.param_groups:
                gp["lr"] = lr_now
            Xb = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w_use, device=device)
            fm.train(); opt.zero_grad()
            (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward()
            opt.step(); fm.eval()
            if kind != "fixed":
                bench_step(bnet, bopt, S, e)                # b(s) tracks the own-error field
            w_sum += float(w_hat.sum()); w_cnt += len(idx)
            spend_sum += spend; spend_cnt += 1
            clip_sum += float((w_hat >= w_clip - 1e-6).mean())
            return w_hat, e, b, dlt, g, gt, spend

        if cfg["grade_control"] and 0 in milestone_set:
            snaps[0] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
        for c in range(n_cycles):
            inc = np.arange(c * B, (c + 1) * B)
            w_hat, e, b, dlt, g, gt, spend = gated_step(inc)
            cls = s_cls[inc]
            blog["t"].append((c + 1) * B)
            blog["spend"].append(spend)
            blog["clipfrac"].append(float((w_hat >= w_clip - 1e-6).mean()))
            for ci, nm in enumerate(CLS_NAMES):
                m = cls == ci
                for q, arr in zip(LOGQ, (w_hat, e, b, dlt, g, gt)):
                    blog[f"{q}_{nm}"].append(float(np.nanmean(arr[m])) if m.any() else np.nan)
                cum_w[ci] += float(w_hat[m].sum()); cum_n[ci] += int(m.sum())
            for _ in range(spec["n_replay"]):
                ridx = rng.integers(0, (c + 1) * B, size=cfg["replay_batch"])
                gated_step(ridx)
            t_now = (c + 1) * B
            if c % cfg["probe_every"] == 0 or t_now in milestone_set:
                pe = probe_errs(fm)
                trace["t"].append(t_now)
                for k, v in pe.items():
                    trace[f"probe_{k}"].append(v)
            if t_now in milestone_set:
                if cfg["grade_control"]:
                    snaps[t_now] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
                print(f"[{spec['name']} m={t_now:5d}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in pe.items()), flush=True)
        budget = {"mean_w": w_sum / max(w_cnt, 1),
                  "mean_spend": spend_sum / max(spend_cnt, 1),
                  "clip_frac": clip_sum / max(spend_cnt, 1),
                  "eff_lr": base_lr * (spend_sum / max(spend_cnt, 1))
                  if spec["spend_mode"] == "free" else base_lr,
                  "cum_w_share": {CLS_NAMES[ci]: float(cum_w[ci] / max(cum_w.sum(), 1e-9))
                                  for ci in range(K + 1)},
                  "sample_share": {CLS_NAMES[ci]: float(cum_n[ci] / max(cum_n.sum(), 1e-9))
                                   for ci in range(K + 1)}}
        print(f"[{spec['name']}] mean_w={budget['mean_w']:.3f} spend={budget['mean_spend']:.3f} "
              f"eff_lr={budget['eff_lr']:.2e} clip={budget['clip_frac']:.3f}  shares: " +
              " ".join(f"{k}={v:.3f}" for k, v in budget["cum_w_share"].items()), flush=True)
        return snaps, trace, blog, budget

    # ===================================================================== #
    # 5) DRIVER
    # ===================================================================== #
    for spec in cfg["arm_specs"]:
        stack = build_stack(spec["fm_hidden"], spec["fm_layers"])
        skey = f"h{spec['fm_hidden']}x{spec['fm_layers']}"
        if skey not in result["stacks"]:
            result["stacks"][skey] = dict(stack["calib"])
            if cfg["grade_control"]:
                g_stale = grade(stack["fm2_0"]); g_stale["transitions"] = 0
                g_ceil = grade(stack["fm_ceil"]); g_ceil["transitions"] = -1
                print(f"[grade stale   {skey}] " + fmt_grade(g_stale), flush=True)
                print(f"[grade ceiling {skey}] " + fmt_grade(g_ceil), flush=True)
                result["stacks"][skey]["stale"] = g_stale
                result["stacks"][skey]["ceiling"] = g_ceil
            checkpoint()
        print(f"\n===== arm: {spec['name']} (stack {skey}) =====", flush=True)
        snaps, trace, blog, budget = run_arm(spec, stack)
        result["arms"][spec["name"]] = {"spec": spec, "stack": skey, "trace": trace,
                                        "blog": blog, "budget": budget}
        checkpoint()
        if cfg["grade_control"]:
            lad_net = _mlp(0, 6, 4, spec["fm_hidden"], spec["fm_layers"])
            lad = []
            for m in sorted(snaps):
                if m == 0:
                    lad.append(dict(result["stacks"][skey]["stale"])); continue
                lad_net.load_state_dict(snaps[m]); lad_net.to(device).eval()
                rec = grade(lad_net); rec["transitions"] = int(m)
                lad.append(rec)
                print(f"[grade {spec['name']:22s} m={m:5d}] " + fmt_grade(rec), flush=True)
                result["arms"][spec["name"]]["ladder"] = lad
                checkpoint()
            result["arms"][spec["name"]]["ladder"] = lad
            checkpoint()
        del snaps

    result["complete"] = True
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("complete\n")
    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": result}


@app.local_entrypoint()
def priced(
    quick: bool = False,
    mode: str = "main",
    tag: str = "",
    seed: int = 0,
    # ---- arms.  token: kind[@lr][:spend][:norm][:rN][:xM]
    arms: str = "",
    lr_grid: str = "",           # main mode: fixed-arm lr grid (traces the frontier)
    controllers: str = "reactive,ballistic_cem",
    # ---- calibrate mode grid
    cal_hidden: str = "32,64,256",
    cal_replay: str = "0,4",
    cal_lr: str = "1e-4,3e-4,1e-3,3e-3",
    cal_extra: str = "fixed@3e-4:raw_adam:r4:x3,fixed@3e-4:raw_adam:r4:x1",
    # regions / eval geometry (bridge_assembly's, unchanged)
    regions: str = ("0.0,0.30,1.2,0.0,0,A-drift; "
                    "0.0,-0.30,-1.2,0.0,1,B-mastered; "
                    "0.0,0.90,0.0,30.0,0,R-noise"),
    region_sigma: float = 0.18,
    eval_families: str = "drift:0.30; mastered:-0.30",
    # ---- the three cost knobs (defaults = PRICED; the parent's values in brackets)
    fm_hidden: int = 32,          # [256] capacity: below meta_adapt's competition boundary
    fm_layers: int = 2,           # [3]
    n_replay: int = 0,            # [4]  no repair buffer
    norm_mode: str = "static",    # [ewma] attenuation is SAVED, not redistributed
    spend_mode: str = "free",     # [raw_adam] delta chooses its own total spend
    # online adaptation
    total_t: int = 16384,
    batch: int = 16,
    replay_batch: int = 64,
    adapt_lr: float = 3e-4,
    milestones: str = "0,256,512,1024,2048,4096,8192,16384",
    probe_every: int = 16,
    # gain law
    w_clip: float = 4.0,
    w_raw_cap: float = 20.0,
    ewma_alpha: float = 0.05,
    # benchmark net b(s)
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 3e-3,
    bench_pretrain_steps: int = 1500,
    # env (the 4c/plasticity_gain regime)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    pool_n: int = 9000,
    master_extra_n: int = 3000,
    probe_n: int = 500,
    v_explore: float = 1.2,
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
        fm_steps = 1200; bench_pretrain_steps = 500
        n_eval = 12; k_shoot = 96; probe_every = 8; replay_batch = 32
        master_extra_n = 800
        cal_hidden = "32,256"; cal_replay = "0"; cal_lr = "3e-4,3e-3"
        cal_extra = "fixed@3e-4:raw_adam:r0:x3,fixed@3e-4:raw_adam:r0:x1"
        lr_grid = lr_grid or "1e-4,1e-3"
        tag = tag or ("smoke_" + mode)
    tag = tag or ("cal" if mode == "calibrate" else "pp")

    grade_control = (mode == "main")
    tokens = []
    if arms:
        tokens = [a for a in arms.split(",") if a]
    elif mode == "calibrate":
        for h in [int(x) for x in cal_hidden.split(",") if x]:
            for r in [int(x) for x in cal_replay.split(",") if x]:
                for lr in [float(x) for x in cal_lr.split(",") if x]:
                    tokens.append(f"fixed@{lr:g}:r{r}:h{h}")
        tokens += [t for t in cal_extra.split(",") if t]
    else:
        grid = [float(x) for x in (lr_grid or "1e-4,3e-4,1e-3,3e-3,1e-2").split(",") if x]
        tokens = [f"fixed@{lr:g}" for lr in grid]
        tokens += [f"delta@{adapt_lr:g}", f"delta@{adapt_lr:g}:unit", f"raw_err@{adapt_lr:g}"]

    specs = []
    for tk in tokens:
        # `:hN` / `:LN` capacity qualifiers are handled here; the rest in parse_arm
        h, L = fm_hidden, fm_layers
        keep = []
        for p in tk.split(":"):
            if p.startswith("h") and p[1:].isdigit():
                h = int(p[1:])
            elif p.startswith("L") and p[1:].isdigit():
                L = int(p[1:])
            else:
                keep.append(p)
        sp = parse_arm(":".join(keep), adapt_lr, spend_mode, norm_mode, n_replay)
        sp["name"] = tk
        sp["fm_hidden"] = h; sp["fm_layers"] = L
        specs.append(sp)
    # group by stack so each capacity is pretrained once
    specs.sort(key=lambda s: (s["fm_hidden"], s["fm_layers"]))

    assert all(m % batch == 0 for m in ms), "milestones must be multiples of batch"
    assert total_t == max(ms), "total_t should equal the last milestone"

    cfg = dict(
        tag=tag, seed=seed, mode=mode, arm_specs=specs, controllers=ctrl_list,
        regions=REG, eval_families=fams, grade_control=grade_control,
        T=total_t, batch=batch, replay_batch=replay_batch,
        adapt_lr=adapt_lr, milestones=ms, probe_every=probe_every,
        w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha,
        norm_mode=norm_mode, spend_mode=spend_mode,
        fm_hidden=fm_hidden, fm_layers=fm_layers, n_replay=n_replay,
        bench_hidden=bench_hidden, bench_layers=bench_layers,
        bench_lr=bench_lr, bench_pretrain_steps=bench_pretrain_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y,
        collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, master_extra_n=master_extra_n, probe_n=probe_n, v_explore=v_explore,
        fm_lr=fm_lr, fm_steps=fm_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_priced.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, f"{tag}.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}/{tag}.json")
