"""delta-gated plasticity: is the performance error, consumed as a PER-TRIAL GAIN on the
forward model's learning rate, behaviorally load-bearing during reward-free re-adaptation?

Program: `ideas/performance_error_is_the_bridge.md`. This replaces that doc's Experiment 1
(the outer-REINFORCE retrofit), which `drift_value_loop/teacher_snr` found to be a no-op as
specified (the benchmark was already the REINFORCE baseline) on a substrate whose obstruction
is the SEARCH PROCEDURE, not the teacher's form. Re-reading Gadagkar et al. 2016: the bird
consumes its performance-error signal as a per-rendition gain on plasticity (dopamine-gated
corticostriatal plasticity on the just-produced vocal variant), never as a reward for a
low-sample meta-parameter search. Kim, Parvin & Ivry 2019 is the human analog: task outcome
acts as a GAIN on implicit adaptation. The idea doc's own S1 already specifies this
consumption ("the efferent return is a gain, not an addend: delta multiplies the FM's
learning rate"). That use has never been run in this repo. This experiment runs it.

    THE SIGNAL (idea doc S1), per transition (s, u, s'):
        e     = || (s'-s) - FM2(s,u) ||              # arity-2 prediction error
        b(s)  = benchmark of MY OWN recent error     # see "benchmark form" below
        g     = || FM2(s,u) - FM1(s) ||              # arity gap = what my action explains
        delta = (b(s) - e) * sigmoid(g / theta)      # the performance error
    consumed as a PER-SAMPLE PLASTICITY WEIGHT
        w     = f(delta) = exp(-delta / tau)         # direction per Kim et al. 2019:
                                                     #   worse than benchmark (delta<0) -> boost,
                                                     #   at benchmark  f(0)=1 -> nominal,
                                                     #   better than benchmark -> attenuate.
        w_hat = clip(w / EWMA(w), 0, w_clip)         # matched-average-budget normalization
                                                     # (META_ADAPT #4e: fix the budget or the
                                                     # loop games scale, not allocation) plus
                                                     # saturation (Kim's effect is categorical).

    BENCHMARK FORM (a design choice, made principledly): Gadagkar's "flexible performance
    benchmark" is PER-SYLLABLE -- context-conditional, not a global scalar. We implement
    b(s) as a small error-predictor net trained online on (s, e): "how badly do I usually
    do HERE". A global scalar EWMA cannot habituate to a localized noise region (its delta
    stays negative there forever -> fixation, the opposite of the anti-noisy-TV property);
    the doc's literal scalar form runs as its own arm (`delta_scalar`) to dissect this.

    THE SUBSTRATE: the one where FM quality actually transmits to behavior --
    ballistic/README.md Cut 4b/4c (ballistic ~3x transmission slope; re-adaptation restores
    ballistic competence ~4.3x more than reactive). 4c's caveat: a pure damping drift
    recovers step-like (~200 transitions), no room for a gain to matter. Fix per its own
    suggestion + the directed cut's machinery: a RICHER drift of spatially-localized
    command->motion rotations (`rot_regions`; local -> only in-region samples inform each
    phi -> uniform collection stretches recovery into a graded trajectory; open-loop
    COMPENSABLE -> ballistic control can cash the fix in), plus one off-reach ALEATORIC
    noise region (the noisy-TV distractor; off-reach so the control readout stays clean,
    but ~6% of uniformly-collected samples land in it).

    ARMS (identical pretrained FM, identical transition stream, identical optimizer/batch
    structure, matched average weight ~= 1; ONLY the per-sample plasticity weights differ):
      * fixed        -- w_hat = 1 (the ungated baseline)
      * raw_err      -- w ~ e            ("any error-modulated lr" control; no benchmark)
      * delta        -- w = f(delta), b = context-conditional net   (the proposal)
      * delta_scalar -- w = f(delta), b = global scalar EWMA        (the doc's literal S1)

    READOUTS: (i) control (reactive + ballistic_cem) at milestone snapshots along the
    re-adaptation -- recovery speed; (ii) FM probe error decomposed by region (reducible
    R1/R2, noise R3, base) -- the mechanism trace, logged densely; (iii) where plasticity
    was spent (per-class weight traces, cumulative weight shares) and the delta/benchmark/
    gate traces themselves; (iv) retention: base-region error drift = churn damage from
    fitting unimprovable noise. Interpretation strictly a posteriori.

    g NOTE: every sample here is self-generated (the agent acts), so the agency gate
    sigmoid(g/theta) is expected near-inert BY DESIGN -- its live test is the playback
    control (idea doc Experiment 2), not this. g is precomputed per sample from the
    PRETRAINED FM2/FM1 pair (the drift barely moves the action-attribution structure);
    we log its distribution so the gate's (in)activity is checkable.

Run:
    cd experiments/
    modal run mjc/plasticity_gain/plasticity_gain.py::plasticity_gain --quick     # smoke
    modal run --detach mjc/plasticity_gain/plasticity_gain.py::plasticity_gain \
        --tag gain_s0 --seed 0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_plasticity_gain(cfg: dict) -> dict:
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
        f"noise={r['noise']:.0f}" for r in REG), flush=True)

    def make_env(drifted: bool):
        d = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                 joint_damping=cfg["damping"], pusher_r=0.12,
                 noise_seed=cfg["seed"] + 999)
        if drifted:
            d["rot_regions"] = [dict(center=tuple(r["center"]), sigma=r["sigma"],
                                     phi=float(r["phi"]), noise=float(r["noise"])) for r in REG]
        return PusherEnv(d, with_puck=False)

    env0 = make_env(False)          # pre-drift (clean) world: FM + benchmark pretraining
    env1 = make_env(True)           # post-drift world: stream, probes, control eval

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
        """(n, K) Gaussian gate weights at positions pos[:, :2]."""
        w = np.empty((len(pos), K), np.float32)
        for j, r in enumerate(REG):
            d2 = (pos[:, 0] - r["center"][0]) ** 2 + (pos[:, 1] - r["center"][1]) ** 2
            w[:, j] = np.exp(-d2 / (2.0 * r["sigma"] ** 2))
        return w

    def classify(S):
        """Per-sample class id: 0..K-1 = argmax region if its gate > 0.3, else K = 'base'."""
        w = gates(S)
        cls = np.where(w.max(1) > 0.3, w.argmax(1), K)
        return cls

    CLS_NAMES = [r["name"] for r in REG] + ["base"]

    # ---------------- normalization (clean pool, as every cut in this node) ----------------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)
    huber_ps = nn.HuberLoss(delta=1.0, reduction="none")

    def train_steps(net, opt, S, U, S2, steps, brng, bs_cap=512):
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(bs_cap, len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fm2_delta(net, S, U):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def fm1_delta(net, S):
        with torch.no_grad():
            X = torch.tensor(S, device=device, dtype=torch.float32)
            return (net((X - norm["mx"][:4]) / norm["sx"][:4]) * norm["sy"] + norm["my"]).cpu().numpy()

    def per_sample_err(net, S, U, S2):
        return np.linalg.norm(fm2_delta(net, S, U) - (S2 - S), axis=1).astype(np.float32)

    # ---------------- benchmark net b(s): predicted own error at s ----------------
    def make_bench(seed):
        return _mlp(seed, 4, 1, cfg["bench_hidden"], cfg["bench_layers"])

    def bench_pred(bnet, S):
        with torch.no_grad():
            X = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
            return bnet(X).squeeze(-1).cpu().numpy().astype(np.float32)

    def bench_step(bnet, bopt, S, e):
        X = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
        t = torch.tensor(e, device=device, dtype=torch.float32)
        bnet.train(); bopt.zero_grad()
        huber(bnet(X).squeeze(-1), t).backward(); bopt.step(); bnet.eval()

    # ===================================================================== #
    # 1) PRETRAIN on the clean world: FM2, FM1, benchmark; calibrate tau/theta
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

    # benchmark pretrained on the clean pool's own-error field (the bird enters calibrated)
    e_pool = per_sample_err(fm2_0, nS, nU, nS2)
    bench_0 = make_bench(cfg["seed"] + 42)
    bopt_0 = torch.optim.Adam(bench_0.parameters(), lr=cfg["bench_lr"])
    brngb = np.random.default_rng(cfg["seed"] + 302)
    for _ in range(cfg["bench_pretrain_steps"]):
        idx = brngb.integers(0, len(nS), size=256)
        bench_step(bench_0, bopt_0, nS[idx], e_pool[idx])

    # calibration on a held-out clean probe
    cS, cU, cS2 = collect_box(env0, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    e_cal = per_sample_err(fm2_0, cS, cU, cS2)
    b_cal = bench_pred(bench_0, cS)
    g_cal = np.linalg.norm(fm2_delta(fm2_0, cS, cU) - fm1_delta(fm1_0, cS), axis=1)
    resid = b_cal - e_cal
    tau = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)
    theta = max(float(np.median(g_cal)), 1e-6)
    print(f"[calib] pretrain fm_err(clean)={e_cal.mean():.4f}  tau={tau:.4f}  "
          f"theta={theta:.4f}  gate(sigma(g/theta)) p10/p50/p90="
          f"{np.percentile(1/(1+np.exp(-g_cal/theta)), [10,50,90]).round(3)}", flush=True)

    # ===================================================================== #
    # 2) THE DRIFT + the shared stream (identical data for every arm)
    # ===================================================================== #
    sS, sU, sS2 = collect_box(env1, cfg["T"], np.random.default_rng(cfg["seed"] + 500))
    s_cls = classify(sS)
    frac = {CLS_NAMES[c]: float((s_cls == c).mean()) for c in range(K + 1)}
    print(f"[stream] T={cfg['T']} class fractions: " +
          " ".join(f"{k}={v:.3f}" for k, v in frac.items()), flush=True)

    # g precomputed per sample from the PRETRAINED pair (see docstring g NOTE)
    s_g = np.linalg.norm(fm2_delta(fm2_0, sS, sU) - fm1_delta(fm1_0, sS), axis=1).astype(np.float32)
    s_gate = (1.0 / (1.0 + np.exp(-s_g / theta))).astype(np.float32)

    # probes in the drifted world (fixed; identical for every arm)
    prng = np.random.default_rng(cfg["seed"] + 600)
    probes = {"global": collect_box(env1, cfg["probe_n"], prng)}
    for j in range(K):
        probes[REG[j]["name"]] = collect_region(env1, j, cfg["probe_n"], prng)
    # base probe: uniform box, rejecting anything with any gate > 0.05
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
    # 3) THE FOUR ARMS -- per-sample plasticity gains over the same stream
    # ===================================================================== #
    B = cfg["batch"]; n_cycles = cfg["T"] // B
    milestone_set = set(cfg["milestones"])
    w_clip = cfg["w_clip"]; ew_a = cfg["ewma_alpha"]; sb_a = cfg["scalar_b_alpha"]

    def run_arm(arm):
        fm = copy.deepcopy(fm2_0)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
        bnet = copy.deepcopy(bench_0)
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        rng = np.random.default_rng(cfg["seed"] + 700)      # same replay draws per arm
        w_norm = 1.0 if arm.startswith("delta") else float(e_pool.mean())  # EWMA of raw w
        b_scalar = float(e_pool.mean())
        snaps = {}                                          # transitions -> state_dict
        trace = {"t": [], **{f"probe_{k}": [] for k in probes}}
        blog = {"t": []}
        for nm in CLS_NAMES:
            for q in ("w", "e", "b", "delta"):
                blog[f"{q}_{nm}"] = []
        cum_w = np.zeros(K + 1, np.float64); cum_n = np.zeros(K + 1, np.float64)
        w_sum = 0.0; w_cnt = 0

        def weights_for(idx):
            """Per-sample plasticity weights for stream rows idx, under this arm's rule,
            evaluated with the arm's CURRENT fm/benchmark (gain applied at use time)."""
            nonlocal w_norm
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            e = per_sample_err(fm, S, U, S2)
            if arm == "fixed":
                return np.ones(len(idx), np.float32), e, np.zeros_like(e), np.zeros_like(e)
            if arm == "raw_err":
                w_raw = e
                b = np.zeros_like(e); dlt = np.zeros_like(e)
            else:
                b = bench_pred(bnet, S) if arm == "delta" else np.full_like(e, b_scalar)
                dlt = (b - e) * s_gate[idx]
                # f(delta)=exp(-delta/tau), capped BEFORE normalization so the budget
                # EWMA stays finite (the cap is the categorical saturation, cf. Kim 2019)
                w_raw = np.exp(np.clip(-dlt / tau, -30.0,
                                       np.log(cfg["w_raw_cap"]))).astype(np.float32)
            w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
            w_hat = np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)
            return w_hat, e, b, dlt

        def gated_step(idx):
            nonlocal b_scalar, w_sum, w_cnt
            w_hat, e, b, dlt = weights_for(idx)
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            Xb = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w_hat, device=device)
            fm.train(); opt.zero_grad()
            (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward()
            opt.step(); fm.eval()
            if arm != "fixed":
                bench_step(bnet, bopt, S, e)                # benchmark tracks own-error field
            w_sum += float(w_hat.sum()); w_cnt += len(idx)
            return w_hat, e, b, dlt

        if 0 in milestone_set:
            snaps[0] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
        for c in range(n_cycles):
            inc = np.arange(c * B, (c + 1) * B)
            w_hat, e, b, dlt = gated_step(inc)              # the just-produced batch
            if arm == "delta_scalar":
                b_scalar = (1 - sb_a) * b_scalar + sb_a * float(e.mean())
            # per-class mechanism log (incoming batch only = arrival-ordered trace)
            cls = s_cls[inc]
            blog["t"].append((c + 1) * B)
            for ci, nm in enumerate(CLS_NAMES):
                m = cls == ci
                for q, arr in (("w", w_hat), ("e", e), ("b", b), ("delta", dlt)):
                    blog[f"{q}_{nm}"].append(float(arr[m].mean()) if m.any() else np.nan)
                cum_w[ci] += float(w_hat[m].sum()); cum_n[ci] += int(m.sum())
            for _ in range(cfg["n_replay"]):                # replay from the stream so far
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

    # incremental checkpointing: a kill (or client cancellation) loses minutes, not the run
    outdir = os.path.join(DATA_DIR, "plasticity_gain", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)

    def checkpoint(obj):
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(obj, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    def json_view(arm_out, extra):
        return {"config": cfg, "complete": False,
                "arms": {a: {k: v for k, v in d.items() if k != "snaps"}
                         for a, d in arm_out.items()}, **extra}

    arm_out = {}
    for arm in cfg["arms"]:
        print(f"\n===== arm: {arm} =====", flush=True)
        snaps, trace, blog, budget = run_arm(arm)
        arm_out[arm] = {"snaps": snaps, "trace": trace, "blog": blog, "budget": budget}
        checkpoint(json_view(arm_out, {"stale_probe": stale_probe}))

    # ===================================================================== #
    # 4) CEILING FM (fresh, on the drifted world) + CONTROL GRADING
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

    ev_rng = np.random.default_rng(cfg["seed"] + 7)
    sgn = ev_rng.choice([-1.0, 1.0], Bev).astype(np.float32)
    ev_starts = np.concatenate([
        np.stack([sgn * cr, np.zeros(Bev, np.float32)], 1)
        + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32),
        ev_rng.normal(0, cfg["v0_std"], (Bev, 2)).astype(np.float32)], 1)
    ev_goals = np.stack([-sgn * cr, np.zeros(Bev, np.float32)], 1) \
        + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32)

    def rollout(net, replan_every):
        states = ev_starts.copy(); plan = None
        rng = np.random.default_rng(cfg["seed"] + 7000 + replan_every)
        for step in range(H):
            if step % replan_every == 0:
                plan = mpc_plan(net, states, ev_goals, rng)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for bb in range(Bev):
                env1.set_state(states[bb, :2].astype(np.float64), states[bb, 2:].astype(np.float64))
                s2, _ = env1.step(acts[bb], fs); states[bb] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - ev_goals, axis=1)))

    def grade(net):
        rec = {"fm": probe_errs(net)}
        if "reactive" in cfg["controllers"]:
            rec["reactive"] = rollout(net, 1)
        if "ballistic_cem" in cfg["controllers"]:
            rec["ballistic_cem"] = rollout(net, H)
        return rec

    print("\n===== grading =====", flush=True)
    g_stale = grade(fm2_0); g_stale["transitions"] = 0
    print("[grade stale   m=0] " + "  ".join(f"{c}={g_stale[c]:.4f}" for c in cfg["controllers"]), flush=True)
    g_ceil = grade(fm_ceil); g_ceil["transitions"] = -1
    print("[grade ceiling    ] " + "  ".join(f"{c}={g_ceil[c]:.4f}" for c in cfg["controllers"]), flush=True)
    checkpoint(json_view(arm_out, {"stale_probe": stale_probe, "stale": g_stale, "ceiling": g_ceil}))

    lad_net = _mlp(0, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])   # shell for loading snapshots
    for arm in cfg["arms"]:
        lad = []
        for m in sorted(arm_out[arm]["snaps"]):
            if m == 0:
                rec = dict(g_stale); lad.append(rec); continue
            lad_net.load_state_dict(arm_out[arm]["snaps"][m]); lad_net.to(device).eval()
            rec = grade(lad_net); rec["transitions"] = int(m)
            lad.append(rec)
            print(f"[grade {arm:12s} m={m:5d}] " +
                  "  ".join(f"{c}={rec[c]:.4f}" for c in cfg["controllers"]) +
                  f"  fm_R1={rec['fm'].get(REG[0]['name'], float('nan')):.4f}", flush=True)
        arm_out[arm]["ladder"] = lad
        checkpoint(json_view(arm_out, {"stale_probe": stale_probe, "stale": g_stale, "ceiling": g_ceil}))

    for arm in cfg["arms"]:
        del arm_out[arm]["snaps"]

    out = {"config": cfg, "complete": True,
           "calibration": {"tau": tau, "theta": theta,
                           "pretrain_fm_err_clean": float(e_cal.mean()),
                           "gate_p10_p50_p90": np.percentile(
                               1 / (1 + np.exp(-g_cal / theta)), [10, 50, 90]).tolist(),
                           "stream_gate_p10_p50_p90": np.percentile(s_gate, [10, 50, 90]).tolist(),
                           "stream_class_fractions": frac,
                           "stale_probe": stale_probe},
           "stale": g_stale, "ceiling": g_ceil,
           "arms": {a: {k: v for k, v in arm_out[a].items()} for a in cfg["arms"]}}
    figures = _make_figures(out)
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
    ARMC = {"fixed": "#495057", "raw_err": "#e8590c", "delta": "#7048e8", "delta_scalar": "#2f9e44"}
    arms = R["config"]["arms"]; REG = R["config"]["regions"]
    red_names = [r["name"] for r in REG if r["noise"] == 0.0]
    noise_names = [r["name"] for r in REG if r["noise"] > 0.0]
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    def _logx(t):
        return np.where(np.asarray(t, float) == 0, R["config"]["batch"] / 2.0, np.asarray(t, float))

    # fig1: control recovery per controller
    ctrls = R["config"]["controllers"]
    fig, axes = plt.subplots(1, len(ctrls), figsize=(6.0 * len(ctrls), 4.6), squeeze=False)
    for ci, c in enumerate(ctrls):
        ax = axes[0][ci]
        for a in arms:
            lad = R["arms"][a]["ladder"]
            t = [r["transitions"] for r in lad]; y = [r.get(c, np.nan) for r in lad]
            ax.plot(_logx(t), y, "o-", color=ARMC.get(a, "#333"), lw=2.0, ms=5, label=a)
        ax.axhline(R["ceiling"].get(c, np.nan), color="#999", ls=":", lw=1.5, label="ceiling (fresh FM)")
        ax.set_xscale("log"); ax.set_xlabel("re-adaptation transitions (reward-free)")
        ax.set_ylabel(f"{c} goal-dist (lower=better)"); ax.set_title(c)
        if ci == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Control recovery under per-trial plasticity gains (matched avg. budget)", fontsize=10)
    figs["fig1_recovery.png"] = _save(fig)

    # fig2: FM probe-error traces (mechanism)
    panels = [("reducible regions " + "+".join(red_names), red_names),
              ("noise region " + "+".join(noise_names), noise_names),
              ("base (retention)", ["base"])]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for pi, (title, keys) in enumerate(panels):
        ax = axes[pi]
        for a in arms:
            tr = R["arms"][a]["trace"]
            y = np.nanmean([tr[f"probe_{k}"] for k in keys], axis=0)
            ax.plot(_logx(tr["t"]), y, "-", color=ARMC.get(a, "#333"), lw=1.8, label=a)
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("transitions")
        if pi == 0:
            ax.set_ylabel("FM probe error"); ax.legend(fontsize=8)
    fig.suptitle("Where the model got better (probe error by region class)", fontsize=10)
    figs["fig2_fm_traces.png"] = _save(fig)

    # fig3: where plasticity was spent (per-class weight traces, gated arms)
    gated = [a for a in arms if a != "fixed"]
    cls_names = [r["name"] for r in REG] + ["base"]
    CLSC = {"base": "#868e96"}
    reds = ["#c92a2a", "#e8590c", "#f08c00"]; blues = ["#1971c2", "#0c8599"]
    ri = bi = 0
    for r in REG:
        if r["noise"] > 0:
            CLSC[r["name"]] = blues[bi % 2]; bi += 1
        else:
            CLSC[r["name"]] = reds[ri % 3]; ri += 1
    if not gated:
        return figs
    fig, axes = plt.subplots(1, len(gated), figsize=(5.2 * len(gated), 4.2), squeeze=False)
    for gi, a in enumerate(gated):
        ax = axes[0][gi]; bl = R["arms"][a]["blog"]
        t = np.asarray(bl["t"], float)
        for nm in cls_names:
            y = np.asarray(bl[f"w_{nm}"], float)
            n_sm = max(len(y) // 60, 1)
            ker = np.ones(n_sm) / n_sm
            ysm = np.convolve(np.nan_to_num(y, nan=1.0), ker, mode="same")
            ax.plot(t, ysm, "-", color=CLSC[nm], lw=1.6, label=nm)
        ax.axhline(1.0, color="#bbb", ls=":", lw=1)
        ax.set_title(a); ax.set_xlabel("transitions"); ax.set_xscale("log")
        if gi == 0:
            ax.set_ylabel("mean plasticity weight $\\hat{w}$"); ax.legend(fontsize=7)
    fig.suptitle("Where plasticity was spent (per-class mean weight, smoothed)", fontsize=10)
    figs["fig3_allocation.png"] = _save(fig)

    # fig4: the delta mechanism (delta arm): per-class delta + benchmark habituation
    if "delta" in arms:
        bl = R["arms"]["delta"]["blog"]; t = np.asarray(bl["t"], float)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for nm in cls_names:
            y = np.asarray(bl[f"delta_{nm}"], float)
            n_sm = max(len(y) // 60, 1); ker = np.ones(n_sm) / n_sm
            axes[0].plot(t, np.convolve(np.nan_to_num(y), ker, mode="same"),
                         color=CLSC[nm], lw=1.6, label=nm)
        axes[0].axhline(0, color="#bbb", ls=":", lw=1)
        axes[0].set_xscale("log"); axes[0].set_title("delta = (b(s) - e) * gate, per class")
        axes[0].set_xlabel("transitions"); axes[0].set_ylabel("delta"); axes[0].legend(fontsize=7)
        for nm in cls_names:
            for q, ls in (("e", "-"), ("b", "--")):
                y = np.asarray(bl[f"{q}_{nm}"], float)
                n_sm = max(len(y) // 60, 1); ker = np.ones(n_sm) / n_sm
                axes[1].plot(t, np.convolve(np.nan_to_num(y), ker, mode="same"),
                             ls, color=CLSC[nm], lw=1.4,
                             label=f"{nm} {'error' if q == 'e' else 'benchmark'}")
        axes[1].set_xscale("log"); axes[1].set_title("error (solid) vs benchmark b(s) (dashed)")
        axes[1].set_xlabel("transitions"); axes[1].legend(fontsize=6)
        fig.suptitle("The performance-error signal over re-adaptation (arm: delta)", fontsize=10)
        figs["fig4_delta_mech.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def plasticity_gain(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "fixed,raw_err,delta,delta_scalar",
    controllers: str = "reactive,ballistic_cem",
    # the drift: two ON-REACH command rotations (reducible; distinct pairwise-far angles per
    # directed S0) + one OFF-REACH aleatoric noise region (the noisy-TV distractor).
    # "cx,cy,phi,noise,name;..." -- all appear at t=0 (the Type-2 event).
    regions: str = ("0.30,0.0,1.2,0.0,R1-on; "
                    "-0.30,0.0,-1.2,0.0,R2-on; "
                    "0.0,0.85,0.0,30.0,R3-noise"),
    region_sigma: float = 0.18,
    # online adaptation
    total_t: int = 8192,                  # total reward-free transitions in the stream
    batch: int = 16,                      # arrival batch (the 'rendition')
    n_replay: int = 4,                    # replay steps per arrival batch (buffer = stream so far)
    replay_batch: int = 64,               # replay batch size (consolidation; gains recomputed at use)
    adapt_lr: float = 3e-4,               # base fine-tune lr (identical across arms)
    milestones: str = "0,256,512,1024,2048,4096,8192",
    probe_every: int = 16,                # probe-error trace cadence (in arrival batches)
    # gain law
    w_clip: float = 4.0,                  # saturation of the normalized weight (categorical cap)
    w_raw_cap: float = 20.0,              # cap on raw f(delta) before normalization
    ewma_alpha: float = 0.05,             # normalizer EWMA (budget matching)
    scalar_b_alpha: float = 0.03,         # the delta_scalar arm's benchmark EWMA (per batch)
    # benchmark net b(s)
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 1e-3,
    bench_pretrain_steps: int = 1500,
    # env (the 4c/directed regime: ballistic-competent at damping 2)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
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

    arm_list = [a for a in arms.split(",") if a]
    ctrl_list = [c for c in controllers.split(",") if c]
    REG = []
    for chunk in [c for c in regions.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        REG.append(dict(center=(float(q[0]), float(q[1])), sigma=region_sigma,
                        phi=float(q[2]), noise=float(q[3]), name=q[4]))
    ms = [int(x) for x in milestones.split(",") if x]
    if quick:
        total_t = 2048; ms = [0, 512, 2048]; pool_n = 2500; probe_n = 250
        fm_steps = 1200; bench_pretrain_steps = 500; fm_hidden = 128; fm_layers = 2
        n_eval = 12; k_shoot = 96; probe_every = 8; replay_batch = 32
        tag = tag or "smoke"
    tag = tag or "default"
    assert all(m % batch == 0 for m in ms), "milestones must be multiples of batch"
    assert total_t == max(ms), "total_t should equal the last milestone"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, controllers=ctrl_list, regions=REG,
        T=total_t, batch=batch, n_replay=n_replay, replay_batch=replay_batch,
        adapt_lr=adapt_lr, milestones=ms,
        probe_every=probe_every, w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha,
        scalar_b_alpha=scalar_b_alpha, bench_hidden=bench_hidden, bench_layers=bench_layers,
        bench_lr=bench_lr, bench_pretrain_steps=bench_pretrain_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_steps=fm_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_plasticity_gain.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "plasticity_gain_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
