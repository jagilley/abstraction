"""Cut 5 stage S1 (ballistic/directed): the fixed-arm allocation ladder — does WHERE you spend
a fixed reward-free collection budget change ballistic competence, and does the value signal
pick the right place? No learning yet; the arms are hand-specified allocations.

Program: `ideas/two_timescale_value_loop.md`. Parents: `directed_separability.py` (S0 — the
partition is compensable, control-relevant and LOCAL) and `ballistic/README.md` (Cuts 4a/4b/4c).
S0 established that the substrate CAN support directed collection. This stage asks whether a
*reward-free value signal* actually selects the right region, by pitting it against the decoys
that cut 4a's confounded `b`-drive fell into.

    THE 2x2 PARTITION (this is the whole design). Four regions, crossing the two properties
    the value signal must jointly respect:

                        |  REDUCIBLE (a stale rotation)   |  IRREDUCIBLE (aleatoric noise)
        ----------------+---------------------------------+------------------------------
        ON  the reach   |  A  <- the right answer          |  C  <- raw-surprise decoy
        OFF the reach   |  B  <- novelty/frontier decoy    |  D  <- neither

    No single naive heuristic wins: a drive that chases prediction ERROR fixates on C/D (the
    noisy TV); one that chases REDUCIBLE surprise but ignores relevance goes to B; one that
    just follows on-policy visitation splits A/C and wastes half its budget on noise. Only
    the conjunction — reducible AND where I am going to be — lands on A.

    THE VALUE SIGNAL (and why it stays REWARD-FREE, which is the arc's whole point).

        value_j  =  learning_progress_j  x  visitation_j

      * `learning_progress_j` — from a cheap MONITORING survey (`monitor_n` real transitions
        per region, far too few to repair anything): fit half, and measure how much held-out
        error the fit actually REMOVES. Large where error is REDUCIBLE (a stale rotation),
        ~0 on aleatoric noise, which no amount of fitting removes. Reward-free — ordinary
        experience, no goal, no return. Two-tier: you passively experience a little
        everywhere, and you CHOOSE where to practise.
      * `visitation_j` — region occupancy of the BALLISTIC plan rolled through the FM ITSELF.
        Purely internal: the forward model tells you where you are going to be, hence where it
        is worth looking. No reward, no rollout in the world.

    So the whole afferent link — value -> where-to-gather -> FM -> ballistic behavior — runs
    on reward-free signals, exactly the asset cut 4's synergy says you can maintain safely.

    WHY NOT ENSEMBLE DISAGREEMENT (a negative worth stating). The obvious reducible-uncertainty
    signal is ensemble spread, and it does NOT work here: every member is trained on the
    PRE-drift pool, where the drifted regions were unambiguous, so the members AGREE — and are
    all confidently wrong. An early build measured a flat 0.0003-0.0004 across all four regions.
    Epistemic disagreement detects where you LACK data, not where the world CHANGED under you.
    Kept as an arm precisely to show it fails.

    PREDICTIONS. (1) Under BALLISTIC control the arms spread widely and `lprog x visits`
    approaches the `oracle` arm; (2) under REACTIVE control the arms are ~flat (it re-grounds
    past a stale FM either way) -> directed collection has a BALLISTIC-SPECIFIC payoff, the
    same shape as 4b's transmission slopes and 4c's recovery gains; (3) each ablated signal
    fails in its PREDICTED direction (error->C/D, lprog-only->B, visitation-only->A/C,
    disagreement-only->flat/uninformative).

Run:
    cd experiments/
    modal run mujoco_control/directed_ladder.py::directed_ladder --quick        # smoke
    for s in 0 1 2; do
      modal run --detach mujoco_control/directed_ladder.py::directed_ladder --tag lad_s$s --seed $s
    done
"""

import copy
import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_directed_ladder(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    names = [r["name"] for r in REG]
    print(f"[setup] device={device} K={K} budget={cfg['budget']}", flush=True)
    for r in REG:
        print(f"    {r['name']:>14s} @({r['center'][0]:+.2f},{r['center'][1]:+.2f}) "
              f"phi_pre={r['phi_pre']:+.2f} -> phi_post={r['phi_post']:+.2f}  noise={r['noise']:.2f}",
              flush=True)

    # ===================================================================== #
    # envs: pre-drift (what the FM was trained on) and post-drift (the true world). They
    # differ ONLY in the `phi_post != phi_pre` regions -> exactly those regions are stale.
    # Noise regions are identical in both (irreducible in both, so nothing to re-learn).
    # ===================================================================== #
    def make_env(key):
        regions = [dict(center=tuple(r["center"]), sigma=r["sigma"],
                        phi=float(r[key]), noise=float(r["noise"])) for r in REG]
        return PusherEnv(dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                              joint_damping=cfg["damping"], pusher_r=0.12,
                              rot_regions=regions, noise_seed=cfg["seed"] + 999),
                         with_puck=False)

    env_pre = make_env("phi_pre")
    env_post = make_env("phi_post")
    stale_j = [j for j in range(K) if abs(REG[j]["phi_post"] - REG[j]["phi_pre"]) > 1e-9]
    print(f"[setup] stale (reducible) regions after the drift: {[names[j] for j in stale_j]}", flush=True)

    def gate(pos, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((pos[..., 0] - c[0]) ** 2 + (pos[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    # ===================================================================== #
    # FM stack (transmission-cut)
    # ===================================================================== #
    def _mlp(seed, hidden=None, layers=None):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h = hidden or cfg["fm_hidden"]; L = layers or cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, 4)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1), device=device, dtype=torch.float32)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    # ---------------- collection ----------------
    def _roll(cenv, pos, rng):
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
        return _roll(cenv, pos, rng)

    def collect_region(cenv, j, n, rng):
        c = REG[j]["center"]; s = REG[j]["sigma"] * cfg["collect_sigma_frac"]
        pos = np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1).astype(np.float32)
        return _roll(cenv, pos, rng)

    def query_region(j, n, rng):
        """(s,u) query points inside region j — NO environment interaction. This is what the
        internal signals (ensemble disagreement) are evaluated on."""
        c = REG[j]["center"]; s = REG[j]["sigma"] * cfg["collect_sigma_frac"]
        pos = np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1)
        vel = rng.normal(0, cfg["v_explore"], (n, 2))
        S = np.concatenate([pos, vel], 1).astype(np.float32)
        U = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        return S, U

    nS, nU, nS2 = collect_box(env_post, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    probes = []
    for j in range(K):
        pS, pU, pS2 = collect_region(env_post, j, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 100 + j))
        probes.append((pS, pU, (pS2 - pS).astype(np.float32)))
    gS, gU, gS2 = collect_box(env_post, cfg["probe_n"] * 2, np.random.default_rng(cfg["seed"] + 2))
    gT = (gS2 - gS).astype(np.float32)

    def err_region(net, j):
        pS, pU, pT = probes[j]
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    def err_vec(net):
        return {"global": float(np.linalg.norm(fm_delta(net, gS, gU) - gT, axis=1).mean()),
                "per_region": [err_region(net, j) for j in range(K)]}

    # ===================================================================== #
    # CEM + rollout
    # ===================================================================== #
    Kc, ne_el = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]; B = cfg["n_eval"]

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
                idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    rng_ev = np.random.default_rng(cfg["seed"] + 7)
    sgn = rng_ev.choice([-1.0, 1.0], B).astype(np.float32)
    spos = np.stack([sgn * cr, np.zeros(B, np.float32)], 1) + rng_ev.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
    ev_goals = np.stack([-sgn * cr, np.zeros(B, np.float32)], 1) + rng_ev.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
    ev_starts = np.concatenate([spos, rng_ev.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)

    def rollout(plan_fn, replan_every):
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                env_post.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = env_post.step(acts[b], fs); states[b] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - ev_goals, axis=1)))

    def grade(net):
        rec = err_vec(net)
        for cname, re_every, off in (("reactive", 1, 7000), ("ballistic_cem", H, 7001)):
            if cname not in cfg["controllers"]:
                continue
            rng = np.random.default_rng(cfg["seed"] + off)
            rec[cname] = rollout(lambda s, g: mpc_plan(net, s, g, rng), re_every)
        return rec

    # ===================================================================== #
    # the pre-drift FM (control) + the pre-drift ENSEMBLE (signal only)
    # ===================================================================== #
    print("\n=== pretrain on the PRE-drift world ===", flush=True)
    S0_, U0_, S20_ = collect_box(env_pre, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 10))
    fm0 = _mlp(cfg["seed"] + 40)
    train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                S0_, U0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    stale_rec = grade(fm0); stale_rec["arm"] = "stale (no collection)"
    print(f"[stale] fm_err(glob)={stale_rec['global']:.4f} per-region=["
          + " ".join(f"{e:.3f}" for e in stale_rec["per_region"]) + "]  "
          + "  ".join(f"{c}={stale_rec[c]:.4f}" for c in cfg["controllers"]), flush=True)

    ens = []
    for e in range(cfg["ens_n"]):
        m = _mlp(cfg["seed"] + 900 + 7 * e, hidden=cfg["ens_hidden"], layers=2)
        bs_rng = np.random.default_rng(cfg["seed"] + 950 + e)
        bidx = bs_rng.integers(0, len(S0_), size=len(S0_))          # bootstrap resample
        train_steps(m, torch.optim.Adam(m.parameters(), lr=cfg["fm_lr"]),
                    S0_[bidx], U0_[bidx], S20_[bidx], cfg["ens_steps"],
                    np.random.default_rng(cfg["seed"] + 970 + e))
        ens.append(m)
    print(f"[ens] trained {cfg['ens_n']} members (hidden={cfg['ens_hidden']})", flush=True)

    # ===================================================================== #
    # the three per-region signals
    # ===================================================================== #
    q_rng = np.random.default_rng(cfg["seed"] + 1500)
    disag = np.zeros(K)
    for j in range(K):
        qS, qU = query_region(j, cfg["sig_n"], q_rng)                 # NO env interaction
        P = np.stack([fm_delta(m, qS, qU) for m in ens])              # (E, n, 4)
        disag[j] = float(P.std(0).mean())

    # visitation: roll the BALLISTIC plan through the FM ITSELF (internal, no env interaction)
    plan0 = mpc_plan(fm0, ev_starts, ev_goals, np.random.default_rng(cfg["seed"] + 7001))
    visits = np.zeros(K)
    with torch.no_grad():
        s = torch.tensor(ev_starts, device=device)
        for h in range(H):
            pos = s[:, :2].cpu().numpy()
            for j in range(K):
                visits[j] += float(gate(pos, j).sum())
            x = torch.cat([s, torch.tensor(plan0[:, h, :], device=device)], 1)
            s = s + (fm0((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
    visits = visits / max(visits.sum(), 1e-9)

    # replay: pre-drift data OUTSIDE every region is still valid post-drift (gates ~0 there).
    # Defined here because the learning-progress signal fine-tunes with the same recipe the
    # arms use, so that LP actually predicts what spending the budget there would buy.
    gmax = np.max(np.stack([gate(S0_[:, :2], j) for j in range(K)]), 0)
    oidx = np.flatnonzero(gmax < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = S0_[rp], U0_[rp], S20_[rp]
    base_state = copy.deepcopy(fm0.state_dict())

    # A cheap MONITORING survey: `monitor_n` real transitions per region, far too few to
    # repair anything (the budget is what repairs), but enough to *measure* two things. This
    # is the two-tier structure: you passively experience a little everywhere, and you CHOOSE
    # where to practise. Both readouts below are reward-free — ordinary experience, no goal.
    mon = np.zeros(K)          # raw prediction error   -> fires on aleatoric noise too
    lprog = np.zeros(K)        # LEARNING PROGRESS      -> fires only where error is REDUCIBLE
    for j in range(K):
        mS, mU, mS2 = collect_region(env_post, j, cfg["monitor_n"],
                                     np.random.default_rng(cfg["seed"] + 1600 + j))
        mon[j] = float(np.linalg.norm(fm_delta(fm0, mS, mU) - (mS2 - mS), axis=1).mean())
        h = len(mS) // 2                                   # fit half / held-out half
        e_before = float(np.linalg.norm(fm_delta(fm0, mS[h:], mU[h:]) - (mS2[h:] - mS[h:]), axis=1).mean())
        probe = _mlp(cfg["seed"] + 40); probe.load_state_dict(base_state)
        train_steps(probe, torch.optim.Adam(probe.parameters(), lr=cfg["finetune_lr"]),
                    np.concatenate([rS, mS[:h]]), np.concatenate([rU, mU[:h]]),
                    np.concatenate([rS2, mS2[:h]]), cfg["lp_steps"],
                    np.random.default_rng(cfg["seed"] + 1700 + j))
        e_after = float(np.linalg.norm(fm_delta(probe, mS[h:], mU[h:]) - (mS2[h:] - mS[h:]), axis=1).mean())
        lprog[j] = max(e_before - e_after, 0.0)

    print("\n=== per-region signals (pre-collection) ===", flush=True)
    print("                " + " ".join(f"{n:>14s}" for n in names), flush=True)
    for lbl, v in (("disagreement", disag), ("visitation", visits),
                   ("raw error", mon), ("learn-progress", lprog)):
        print(f"    {lbl:>14s}  " + " ".join(f"{x:>14.4f}" for x in v), flush=True)
    print(f"    (truth: reducible-and-stale = {[names[j] for j in stale_j]}; "
          f"on-reach = {[names[j] for j in range(K) if abs(REG[j]['center'][1]) <= 0.3]})", flush=True)
    print("    NOTE: ensemble DISAGREEMENT is expected to be flat here and that is the point —\n"
          "    every member trained on the PRE-drift pool agrees (confidently wrong). Epistemic\n"
          "    disagreement detects where you LACK data, not where the world CHANGED under you.\n"
          "    Learning progress is the reward-free signal that can see a drift.", flush=True)

    # ===================================================================== #
    # arms: each is an allocation over the K regions (or the uniform box)
    # ===================================================================== #
    def _norm(v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return v / v.sum() if v.sum() > 1e-12 else np.full(K, 1.0 / K)

    oracle_w = np.zeros(K)
    for j in stale_j:
        oracle_w[j] = visits[j] + 1e-6          # among stale regions, weight by relevance
    arms = [
        ("uniform-box", None),
        ("oracle", _norm(oracle_w)),
        ("random", _norm(np.random.default_rng(cfg["seed"] + 3).random(K))),
        ("visitation-only", _norm(visits)),
        ("error-only", _norm(mon)),
        ("disagreement-only", _norm(disag)),
        ("lprog-only", _norm(lprog)),
        ("error x visits", _norm(mon * visits)),
        ("lprog x visits  [VALUE]", _norm(lprog * visits)),
    ]

    print(f"\n=== arms (budget={cfg['budget']}, replay={len(rp)}) ===", flush=True)
    arm_recs = []
    for ai, (aname, w) in enumerate(arms):
        rng_a = np.random.default_rng(cfg["seed"] + 2000 + 13 * ai)
        if w is None:
            cS, cU, cS2 = collect_box(env_post, cfg["budget"], rng_a)
            alloc = None
        else:
            counts = np.floor(w * cfg["budget"]).astype(int)
            counts[int(np.argmax(w))] += cfg["budget"] - counts.sum()
            parts = [collect_region(env_post, j, int(counts[j]), rng_a) for j in range(K) if counts[j] > 0]
            cS = np.concatenate([p[0] for p in parts]); cU = np.concatenate([p[1] for p in parts])
            cS2 = np.concatenate([p[2] for p in parts]); alloc = counts
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        train_steps(net, torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"]),
                    np.concatenate([rS, cS]), np.concatenate([rU, cU]), np.concatenate([rS2, cS2]),
                    cfg["finetune_steps"], np.random.default_rng(cfg["seed"] + 2500 + ai))
        rec = grade(net); rec["arm"] = aname
        rec["alloc"] = (alloc.tolist() if alloc is not None else None)
        arm_recs.append(rec)
        astr = ("box" if alloc is None else "/".join(str(int(x)) for x in alloc))
        print(f"[{aname:>24s}] alloc={astr:>18s}  fm_err=["
              + " ".join(f"{e:.3f}" for e in rec["per_region"]) + "]  "
              + "  ".join(f"{c}={rec[c]:.4f}" for c in cfg["controllers"]), flush=True)

    # ---- headline: gap closed toward the oracle, per controller ----
    by = {r["arm"]: r for r in arm_recs}
    print("\n=== gap closed toward ORACLE (1.0 = matches oracle, 0.0 = no better than uniform) ===", flush=True)
    gaps = {}
    for c in cfg["controllers"]:
        u = by["uniform-box"][c]; o = by["oracle"][c]; den = u - o
        gaps[c] = {r["arm"]: (u - r[c]) / den if abs(den) > 1e-9 else float("nan") for r in arm_recs}
        print(f"  {c:14s} stale={stale_rec[c]:.4f} uniform={u:.4f} oracle={o:.4f} "
              f"(spread={den:+.4f})", flush=True)
        for r in arm_recs:
            print(f"      {r['arm']:>24s}  {r[c]:.4f}   gap-closed={gaps[c][r['arm']]:+.2f}", flush=True)
    if len(cfg["controllers"]) == 2:
        db = by["uniform-box"]["ballistic_cem"] - by["oracle"]["ballistic_cem"]
        dr = by["uniform-box"]["reactive"] - by["oracle"]["reactive"]
        print(f"\n[headline] allocation matters {db / max(dr, 1e-9):.1f}x more for BALLISTIC "
              f"than REACTIVE (uniform-oracle spread {db:+.4f} vs {dr:+.4f})", flush=True)
    print("[headline] PREDICTION: 'lprog x visits' ~ oracle; 'error-*' drawn to the noise "
          "regions; 'lprog-only' drawn off-reach; 'disagreement-only' uninformative.", flush=True)

    out = {"config": cfg, "region_names": names, "stale_regions": stale_j,
           "signals": {"disagreement": disag, "visitation": visits, "raw_error": mon,
                       "learning_progress": lprog},
           "stale": stale_rec, "arms": arm_recs, "gaps": gaps}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "directed_ladder", cfg["tag"])
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

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"reactive": "#7048e8", "ballistic_cem": "#e8590c"}
    ctrls = R["config"]["controllers"]; names = R["region_names"]; K = len(names)
    arms = R["arms"]; figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # 1) the arm ladder
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = np.arange(len(arms)); w = 0.36
    for i, c in enumerate(ctrls):
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, [a[c] for a in arms], w,
               color=COL.get(c, "#555"), label=c)
        ax.axhline(R["stale"][c], color=COL.get(c, "#555"), ls=":", lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels([a["arm"] for a in arms], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("control goal-dist (lower=better)")
    ax.set_title("S1) fixed-arm allocation ladder — identical budget, different WHERE\n"
                 "dotted = stale (no collection); want the VALUE arm ~ oracle for ballistic only")
    ax.legend(fontsize=8.5)
    figs["fig1_ladder.png"] = _save(fig)

    # 2) signals + allocations
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    sg = R["signals"]; xx = np.arange(K); w = 0.20
    keys = [("disagreement", "disagreement"), ("visitation", "visitation"),
            ("raw error", "raw_error"), ("learn-progress", "learning_progress")]
    for i, (lbl, key) in enumerate(keys):
        v = np.array(sg[key], float); v = v / max(v.sum(), 1e-9)
        axes[0].bar(xx + (i - 1.5) * w, v, w, label=lbl)
    axes[0].set_xticks(xx); axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("normalized signal"); axes[0].legend(fontsize=8.5)
    axes[0].set_title("per-region signals (pre-collection)")
    alloc_arms = [a for a in arms if a["alloc"] is not None]
    M = np.array([a["alloc"] for a in alloc_arms], float)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    im = axes[1].imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_yticks(range(len(alloc_arms)))
    axes[1].set_yticklabels([a["arm"] for a in alloc_arms], fontsize=8)
    for i in range(len(alloc_arms)):
        for j in range(K):
            axes[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    axes[1].set_title("where each arm spent its budget")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    figs["fig2_signals_alloc.png"] = _save(fig)

    # 3) gap closed
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(arms)); w = 0.36
    for i, c in enumerate(ctrls):
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, [R["gaps"][c][a["arm"]] for a in arms], w,
               color=COL.get(c, "#555"), label=c)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="#2f9e44", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([a["arm"] for a in arms], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("gap closed toward oracle")
    ax.set_title("S1) how much of the uniform->oracle gap each signal closes")
    ax.legend(fontsize=8.5)
    figs["fig3_gap_closed.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def directed_ladder(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    controllers: str = "reactive,ballistic_cem",
    # the 2x2 partition: "cx,cy,phi_pre,phi_post,noise,name;..."
    #   A on-reducible  : on the reach, rotation appears at the drift   -> the right answer
    #   B off-reducible : off the reach, rotation appears at the drift  -> frontier/novelty decoy
    #   C on-noise      : on the reach, aleatoric (irreducible)         -> raw-surprise decoy
    #   D off-noise     : off the reach, aleatoric                      -> neither
    regions: str = ("0.30,0.0,0.0,1.2,0.0,A-on-reducible; "
                    "0.0,0.85,0.0,-1.2,0.0,B-off-reducible; "
                    "-0.30,0.0,0.0,0.0,30.0,C-on-noise; "
                    "0.0,-0.85,0.0,0.0,30.0,D-off-noise"),
    region_sigma: float = 0.18,
    # env
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    # collection / probes
    pool_n: int = 9000,
    probe_n: int = 500,
    v_explore: float = 1.2,
    # FM + ensemble
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    ens_n: int = 5,
    ens_hidden: int = 128,
    ens_steps: int = 1500,
    sig_n: int = 600,                     # query points per region for the internal signals
    monitor_n: int = 160,                 # free real transitions per region: the cheap survey
    lp_steps: int = 400,                  # fine-tune steps for the learning-progress probe
    # budgeted collection
    budget: int = 400,
    replay_n: int = 2500,
    finetune_lr: float = 3e-4,
    finetune_steps: int = 1200,
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
        parts = [p.strip() for p in chunk.split(",")]
        REG.append(dict(center=(float(parts[0]), float(parts[1])), sigma=region_sigma,
                        phi_pre=float(parts[2]), phi_post=float(parts[3]),
                        noise=float(parts[4]), name=parts[5]))
    if quick:
        pool_n = 2500; probe_n = 200; fm_steps = 1200; finetune_steps = 400; replay_n = 1200
        fm_hidden = 128; fm_layers = 2; n_eval = 12; k_shoot = 96; budget = 250
        ens_n = 3; ens_steps = 600; sig_n = 300; monitor_n = 120; lp_steps = 200
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, controllers=ctrl_list, regions=REG,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        ens_n=ens_n, ens_hidden=ens_hidden, ens_steps=ens_steps, sig_n=sig_n, monitor_n=monitor_n,
        lp_steps=lp_steps,
        budget=budget, replay_n=replay_n, finetune_lr=finetune_lr, finetune_steps=finetune_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_directed_ladder.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "directed_ladder_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
