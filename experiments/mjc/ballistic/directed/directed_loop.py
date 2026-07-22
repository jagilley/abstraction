"""Cut 5 stage S2 (ballistic/directed): the ONLINE closed loop — a reward-free value signal
chooses where to collect, round after round, while the drift target MOVES. The full afferent
link running continuously: value -> where-to-gather -> FM -> ballistic behavior.

Program: `ideas/two_timescale_value_loop.md`. Parents: `directed_separability.py` (S0 — the
partition is compensable, control-relevant, LOCAL) and `directed_ladder.py` (S1 — at a fixed
budget, `learning_progress x visitation` recovers the oracle allocation while every ablated
signal fails in its predicted direction). S1 is a ONE-SHOT choice against a STATIC drift. This
stage makes it a loop against a MOVING one, which is the regime cut 4a identified as the only
one where a value loop can earn anything ("the value benefit is an adaptation-SPEED effect that
needs sufficient non-stationarity").

    THE SCHEDULE. Every `drift_every` rounds one of the two REDUCIBLE regions is re-drifted
    (its command rotation phi_j is redrawn), cycling A -> B -> A -> B. A is ON the reach, B is
    OFF it. So the loop must not merely find the stale region, it must TRACK a target that
    moves, and — the sharper test — it must DECLINE to chase the drift when it lands off-reach.

    THE ASYMMETRY THAT IS THE WHOLE POINT. A value-directed learner should end up with a model
    that is permanently, knowingly WRONG in region B: it never goes there, so it never spends
    budget there, so B's error stays high forever. That is not a failure — it is the claim. You
    do not need a globally accurate forward model, only one that is accurate where you act. The
    `lprog-only` policy, which chases reducible surprise without asking whether it matters,
    keeps paying for B and is measurably worse at the thing behavior actually needs.

    POLICIES (identical drift schedule, budget, eval geometry, and starting FM — controlled):
      * `value`  = learning_progress x visitation   <- the reward-free afferent signal
      * `oracle` = ground-truth stale-region mask x visitation
      * `uniform`, `lprog-only`, `visits-only`, `error-only`  <- baseline + the S1 decoys
    Each runs the whole T-round loop from the same pretrained FM, carrying its own FM and its
    own per-region buffers forward. Reward is never used by any policy except as a metric.

    METRICS. (1) ballistic goal-distance vs round -> area under the curve per policy, and the
    fraction of the uniform->oracle gap the value policy closes; (2) TRACKING: the share of
    each round's budget that lands in the currently-stale ON-reach region, and how many rounds
    it takes to re-allocate after a drift; (3) per-region FM error over time — showing the
    value policy deliberately abandons region B; (4) reactive control on a coarser grid, to
    re-confirm the payoff is BALLISTIC-SPECIFIC (4b/4c's ~3-4x, S1's spread ratio).

Run:
    cd experiments/
    modal run mjc/ballistic/directed/directed_loop.py::directed_loop --quick        # smoke
    for s in 0 1 2; do
      modal run --detach mjc/ballistic/directed/directed_loop.py::directed_loop --tag loop_s$s --seed $s
    done
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_directed_loop(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    names = [r["name"] for r in REG]
    T = cfg["rounds"]
    reducible = [j for j in range(K) if REG[j]["reducible"]]
    on_reach = [j for j in range(K) if abs(REG[j]["center"][1]) <= 0.3]
    print(f"[setup] device={device} K={K} rounds={T} budget={cfg['budget']} "
          f"reducible={[names[j] for j in reducible]} on-reach={[names[j] for j in on_reach]}", flush=True)

    # ---- the drift schedule: at each event, one reducible region's phi is REDRAWN ----
    sch_rng = np.random.default_rng(cfg["seed"] + 77)
    schedule = {}                                   # round -> (region_idx, new_phi)
    cyc = 0
    # `drift_cycle` is the ORDER in which regions get re-drifted. Default cycles A,A,B so that
    # 2/3 of drift events land ON the reach: an off-reach drift is behaviorally inert by
    # construction, so a 1:1 cycle spends half the run generating events that cannot
    # discriminate any policy, and leaves the recovery metric badly underpowered.
    cycle = cfg["drift_cycle"] or reducible
    for t in range(0, T, cfg["drift_every"]):
        j = cycle[cyc % len(cycle)]; cyc += 1
        mag = cfg["phi_mag"]
        new_phi = float(sch_rng.choice([-1.0, 1.0]) * mag)
        schedule[t] = (j, new_phi)
    print("[setup] drift schedule: " + "; ".join(
        f"r{t}: {names[j]} phi->{p:+.2f}" for t, (j, p) in sorted(schedule.items())), flush=True)

    phi_state = [float(r["phi0"]) for r in REG]

    def make_env(phis):
        regions = [dict(center=tuple(r["center"]), sigma=r["sigma"], phi=float(p),
                        noise=float(r["noise"])) for r, p in zip(REG, phis)]
        return PusherEnv(dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                              joint_damping=cfg["damping"], pusher_r=0.12,
                              rot_regions=regions, noise_seed=cfg["seed"] + 999),
                         with_puck=False)

    def gate(pos, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((pos[..., 0] - c[0]) ** 2 + (pos[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    # ===================================================================== #
    # FM stack
    # ===================================================================== #
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
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

    env0 = make_env(phi_state)
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

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

    def rollout(net, cenv, replan_every, off):
        rng = np.random.default_rng(cfg["seed"] + off)
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = mpc_plan(net, states, ev_goals, rng)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                cenv.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = cenv.step(acts[b], fs); states[b] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - ev_goals, axis=1)))

    def fm_visits(net):
        """Region occupancy of the ballistic plan rolled through the FM ITSELF — no env."""
        plan = mpc_plan(net, ev_starts, ev_goals, np.random.default_rng(cfg["seed"] + 7001))
        v = np.zeros(K)
        with torch.no_grad():
            s = torch.tensor(ev_starts, device=device)
            for h in range(H):
                pos = s[:, :2].cpu().numpy()
                for j in range(K):
                    v[j] += float(gate(pos, j).sum())
                x = torch.cat([s, torch.tensor(plan[:, h, :], device=device)], 1)
                s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
        return v / max(v.sum(), 1e-9)

    # ===================================================================== #
    # pretrain the shared starting FM on the round-0 world (before any drift)
    # ===================================================================== #
    S0_, U0_, S20_ = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 10))
    fm_base = _mlp(cfg["seed"] + 40)
    train_steps(fm_base, torch.optim.Adam(fm_base.parameters(), lr=cfg["fm_lr"]),
                S0_, U0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    base_state = copy.deepcopy(fm_base.state_dict())
    gmax = np.max(np.stack([gate(S0_[:, :2], j) for j in range(K)]), 0)
    oidx = np.flatnonzero(gmax < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = S0_[rp], U0_[rp], S20_[rp]
    print(f"[pretrain] base FM trained; replay={len(rp)} out-of-region transitions", flush=True)

    def _norm_w(v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return v / v.sum() if v.sum() > 1e-12 else np.full(K, 1.0 / K)

    # ===================================================================== #
    # one policy's full T-round loop
    # ===================================================================== #
    def run_policy(pname):
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"])
        phis = [float(r["phi0"]) for r in REG]
        bufs = {j: None for j in range(K)}          # per-region ring buffers
        box_buf = None
        hist = []
        for t in range(T):
            if t in schedule:                        # ---- drift event ----
                j, newp = schedule[t]
                phis[j] = newp
                bufs[j] = None                       # data from a region that just changed is void
            env = make_env(phis)
            # ---------- probes + signals (all reward-free) ----------
            probe = []
            for j in range(K):
                pS, pU, pS2 = collect_region(env, j, cfg["probe_n"],
                                             np.random.default_rng(cfg["seed"] + 8000 + 31 * t + j))
                probe.append((pS, pU, (pS2 - pS).astype(np.float32)))
            per_err = [float(np.linalg.norm(fm_delta(net, p[0], p[1]) - p[2], axis=1).mean())
                       for p in probe]
            visits = fm_visits(net)

            # Ground-truth "stale AND worth fixing", for the ORACLE only (privileged) and as
            # the tracking target: a region counts iff it is REDUCIBLE by construction and its
            # measured error is above the matched-FM floor. Defined from the measured error
            # rather than from the drift bookkeeping so it stays correct when a policy has only
            # PARTIALLY repaired a region.
            stale_mask = np.array([1.0 if (REG[j]["reducible"] and per_err[j] > cfg["err_floor"])
                                   else 0.0 for j in range(K)])

            mon = np.zeros(K); lprog = np.zeros(K)
            if pname in ("value", "value-floor", "lprog-only", "error-only"):
                cur = copy.deepcopy(net.state_dict())
                for j in range(K):
                    mS, mU, mS2 = collect_region(env, j, cfg["monitor_n"],
                                                 np.random.default_rng(cfg["seed"] + 9000 + 41 * t + j))
                    mon[j] = float(np.linalg.norm(fm_delta(net, mS, mU) - (mS2 - mS), axis=1).mean())
                    h = len(mS) // 2
                    eb = float(np.linalg.norm(fm_delta(net, mS[h:], mU[h:]) - (mS2[h:] - mS[h:]), axis=1).mean())
                    pr = _mlp(cfg["seed"] + 40); pr.load_state_dict(cur)
                    train_steps(pr, torch.optim.Adam(pr.parameters(), lr=cfg["finetune_lr"]),
                                np.concatenate([rS, mS[:h]]), np.concatenate([rU, mU[:h]]),
                                np.concatenate([rS2, mS2[:h]]), cfg["lp_steps"],
                                np.random.default_rng(cfg["seed"] + 9500 + 41 * t + j))
                    ea = float(np.linalg.norm(fm_delta(pr, mS[h:], mU[h:]) - (mS2[h:] - mS[h:]), axis=1).mean())
                    lprog[j] = max(eb - ea, 0.0)

            # ---------- the policy: choose WHERE to spend this round's budget ----------
            if pname == "uniform":
                w = None
            elif pname == "oracle":
                # privileged: knows exactly which error is REDUCIBLE and how big it is
                tgt = stale_mask * np.array(per_err) * visits
                w = _norm_w(tgt) if tgt.sum() > 1e-9 else None
            elif pname == "value":
                w = _norm_w(lprog * visits) if (lprog * visits).sum() > 1e-12 else None
            elif pname == "value-floor":
                # `value` with a NO-OP: the product lprog x visits has no lower bound, so once
                # the on-reach region is repaired (lprog_A -> 0) the argmax is whatever still
                # has ANY learning progress -- including an off-reach region whose Gaussian-tail
                # visitation is ~1e-4. An early build spent 4 consecutive rounds' budget in the
                # OFF-reach region for exactly this reason. A real on-reach target scores
                # ~9e-3; the degenerate off-reach case ~2e-6, so the floor separates them by
                # ~4 orders of magnitude rather than being tuned. Below it: default behaviour
                # (uniform box), which keeps the per-round budget IDENTICAL across policies.
                tv = lprog * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None
            elif pname == "value-maint":
                # `value-floor`'s sibling, differing ONLY in what the no-op does. When nothing
                # is notably reducible, `value-floor` reverts to uniform-box (spend it
                # anywhere); `value-maint` reverts to collecting proportional to VISITATION —
                # keep rehearsing where you actually act. loopB/loopC showed continuous
                # top-up beats episodic repair once buffers age out (the `oracle`, which is
                # repair-when-broken, lost to policies that keep maintaining), so which no-op
                # you pick is a real design question, not a detail.
                tv = lprog * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else _norm_w(visits)
            elif pname == "lprog-only":
                w = _norm_w(lprog) if lprog.sum() > 1e-12 else None
            elif pname == "visits-only":
                w = _norm_w(visits)
            elif pname == "error-only":
                w = _norm_w(mon)
            else:
                raise ValueError(pname)

            rng_c = np.random.default_rng(cfg["seed"] + 11000 + 97 * t)
            if w is None:
                cS, cU, cS2 = collect_box(env, cfg["budget"], rng_c)
                alloc = np.full(K, np.nan)
                box_buf = (cS, cU, cS2) if box_buf is None else tuple(
                    np.concatenate([box_buf[i], (cS, cU, cS2)[i]])[-cfg["buf_cap"]:] for i in range(3))
            else:
                counts = np.floor(w * cfg["budget"]).astype(int)
                counts[int(np.argmax(w))] += cfg["budget"] - counts.sum()
                alloc = counts.astype(float)
                for j in range(K):
                    if counts[j] <= 0:
                        continue
                    d = collect_region(env, j, int(counts[j]), rng_c)
                    bufs[j] = d if bufs[j] is None else tuple(
                        np.concatenate([bufs[j][i], d[i]])[-cfg["buf_cap"]:] for i in range(3))

            # ---------- fine-tune on everything currently believed valid ----------
            parts = [(rS, rU, rS2)] + [b for b in bufs.values() if b is not None]
            if box_buf is not None:
                parts.append(box_buf)
            train_steps(net, opt, np.concatenate([p[0] for p in parts]),
                        np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
                        cfg["finetune_steps"], np.random.default_rng(cfg["seed"] + 12000 + t))

            # ---------- grade ----------
            bal = rollout(net, env, H, 7001)
            rec = {"round": t, "ballistic_cem": bal, "alloc": alloc.tolist(),
                   "per_err": per_err, "visits": visits.tolist(),
                   "lprog": lprog.tolist(), "mon": mon.tolist(),
                   "stale_mask": stale_mask.tolist(), "phis": list(phis)}
            if (t % cfg["reactive_every"] == 0) or (t == T - 1):
                rec["reactive"] = rollout(net, env, 1, 7000)
            hist.append(rec)
            astr = "box" if np.isnan(alloc[0]) else "/".join(str(int(x)) for x in alloc)
            print(f"[{pname:>12s} r{t:02d}] alloc={astr:>16s} ball={bal:.4f}"
                  + (f" reac={rec['reactive']:.4f}" if "reactive" in rec else "")
                  + "  err=[" + " ".join(f"{e:.3f}" for e in per_err) + "]", flush=True)
        return hist

    results = {}
    for pname in cfg["policies"]:
        print(f"\n=== policy: {pname} ===", flush=True)
        results[pname] = run_policy(pname)

    # ===================================================================== #
    # summary
    # ===================================================================== #
    def auc(pname, key="ballistic_cem"):
        v = [r[key] for r in results[pname] if key in r]
        return float(np.mean(v))

    print("\n=== summary: mean ballistic goal-dist over all rounds (lower=better) ===", flush=True)
    summary = {}
    for pname in cfg["policies"]:
        a_b = auc(pname, "ballistic_cem"); a_r = auc(pname, "reactive")
        summary[pname] = {"ballistic_auc": a_b, "reactive_auc": a_r}
        print(f"    {pname:>12s}  ballistic={a_b:.4f}   reactive={a_r:.4f}", flush=True)
    if "uniform" in results and "oracle" in results:
        for key in ("ballistic_cem", "reactive"):
            u = auc("uniform", key); o = auc("oracle", key); den = u - o
            print(f"\n  [{key}] uniform={u:.4f} oracle={o:.4f} spread={den:+.4f}", flush=True)
            for pname in cfg["policies"]:
                g = (u - auc(pname, key)) / den if abs(den) > 1e-9 else float("nan")
                summary[pname][key + "_gap_closed"] = g
                print(f"      {pname:>12s}  gap-closed={g:+.2f}", flush=True)
        db = auc("uniform") - auc("oracle")
        dr = auc("uniform", "reactive") - auc("oracle", "reactive")
        print(f"\n[headline] allocation matters {db / max(dr, 1e-9):.1f}x more for BALLISTIC "
              f"than REACTIVE (spread {db:+.4f} vs {dr:+.4f})", flush=True)

    # tracking: share of budget landing in the currently-stale ON-reach region
    print("\n=== tracking: budget share in the stale ON-reach region (when one exists) ===", flush=True)
    track = {}
    for pname in cfg["policies"]:
        sh = []
        for r in results[pname]:
            sm = np.array(r["stale_mask"]); al = np.array(r["alloc"], float)
            tgt = [j for j in on_reach if sm[j] > 0]
            if not tgt or np.isnan(al[0]):
                continue
            sh.append(float(al[tgt].sum() / max(al.sum(), 1e-9)))
        track[pname] = float(np.mean(sh)) if sh else float("nan")
        print(f"    {pname:>12s}  {track[pname]:.2f}", flush=True)

    # the abandonment asymmetry: final per-region FM error
    print("\n=== final per-region FM error (does the value policy ABANDON the off-reach region?) ===", flush=True)
    print("                " + " ".join(f"{n:>14s}" for n in names), flush=True)
    for pname in cfg["policies"]:
        pe = results[pname][-1]["per_err"]
        print(f"    {pname:>12s}  " + " ".join(f"{e:>14.4f}" for e in pe), flush=True)

    out = {"config": cfg, "region_names": names, "schedule": {str(k): v for k, v in schedule.items()},
           "results": results, "summary": summary, "tracking": track}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "directed_loop", cfg["tag"])
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
    pol = R["config"]["policies"]; names = R["region_names"]; K = len(names)
    res = R["results"]; sched = {int(k): v for k, v in R["schedule"].items()}
    COL = {"value": "#e8590c", "value-floor": "#f76707", "oracle": "#2f9e44", "uniform": "#868e96",
           "lprog-only": "#7048e8", "visits-only": "#1c7ed6", "error-only": "#c92a2a"}
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # 1) ballistic control over rounds
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for p in pol:
        v = [r["ballistic_cem"] for r in res[p]]
        ax.plot(range(len(v)), v, "o-", ms=4, lw=2.0, color=COL.get(p, "#333"),
                label=f"{p}  (mean {np.mean(v):.3f})")
    for t, (j, _) in sched.items():
        ax.axvline(t - 0.5, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.text(t - 0.4, ax.get_ylim()[1], names[j], rotation=90, va="top", fontsize=7, alpha=0.6)
    ax.set_xlabel("round (dotted = a drift event re-randomizes one region)")
    ax.set_ylabel("ballistic goal-dist (lower=better)")
    ax.set_title("S2) online directed collection against a MOVING drift target")
    ax.legend(fontsize=8.5)
    figs["fig1_rounds.png"] = _save(fig)

    # 2) allocation traces
    fig, axes = plt.subplots(len(pol), 1, figsize=(9.0, 1.5 * len(pol)), sharex=True)
    if len(pol) == 1:
        axes = [axes]
    for ax, p in zip(axes, pol):
        M = np.array([r["alloc"] for r in res[p]], float)
        if np.isnan(M).all():
            M = np.full_like(M, 1.0 / K)
        M = np.nan_to_num(M, nan=1.0 / K)
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
        ax.imshow(M.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=7)
        ax.set_ylabel(p, fontsize=8, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("round")
    axes[0].set_title("where each policy spent its budget (bright = more)")
    figs["fig2_alloc_trace.png"] = _save(fig)

    # 3) per-region FM error over rounds, value vs lprog-only
    fig, axes = plt.subplots(1, min(3, len(pol)), figsize=(4.3 * min(3, len(pol)), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, [q for q in ("value", "lprog-only", "uniform") if q in res]):
        E = np.array([r["per_err"] for r in res[p]])
        for j in range(K):
            ax.plot(E[:, j], lw=1.8, label=names[j])
        ax.set_title(p, fontsize=10); ax.set_xlabel("round")
    axes[0].set_ylabel("per-region FM error"); axes[0].legend(fontsize=7)
    fig.suptitle("does the value policy knowingly ABANDON the off-reach region?", fontsize=10)
    figs["fig3_region_err.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def directed_loop(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    policies: str = "uniform,oracle,value,value-floor,lprog-only,visits-only,error-only",
    rounds: int = 72,
    drift_every: int = 12,                # long enough that recovery COMPLETES inside an
                                          # epoch at budget=100/round (at 6 it was censored)
    drift_cycle: str = "0,0,1",           # region indices, in order; 2/3 land on the reach
    phi_mag: float = 1.2,
    # regions: "cx,cy,phi0,noise,reducible,name;..."
    regions: str = ("0.30,0.0,0.0,0.0,1,A-on-reducible; "
                    "0.0,0.85,0.0,0.0,1,B-off-reducible; "
                    "-0.30,0.0,0.0,30.0,0,C-on-noise; "
                    "0.0,-0.85,0.0,30.0,0,D-off-noise"),
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
    pool_n: int = 9000,
    probe_n: int = 400,
    v_explore: float = 1.2,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    # per-round loop
    budget: int = 100,                    # per ROUND. Scarce on purpose: at 400 the loop is
    buf_cap: int = 400,                   # budget-RICH in aggregate and every concentrating
                                          # policy saturates the control metric (see README).
    monitor_n: int = 160,
    lp_steps: int = 400,
    replay_n: int = 2500,
    finetune_lr: float = 3e-4,
    finetune_steps: int = 900,
    reactive_every: int = 6,
    err_floor: float = 0.03,              # matched-FM error is ~0.01; stale is ~0.12
    value_floor: float = 1e-3,            # no-op threshold on max(lprog x visits)
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

    pol = [p for p in policies.split(",") if p]
    REG = []
    for chunk in [c for c in regions.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        REG.append(dict(center=(float(q[0]), float(q[1])), sigma=region_sigma,
                        phi0=float(q[2]), noise=float(q[3]), reducible=bool(int(q[4])), name=q[5]))
    if quick:
        rounds = 12; drift_every = 6; pool_n = 2500; probe_n = 150; fm_steps = 1200
        finetune_steps = 300; lp_steps = 150; replay_n = 1200; fm_hidden = 128; fm_layers = 2
        n_eval = 12; k_shoot = 96; budget = 250; monitor_n = 120; reactive_every = 4
        pol = ["uniform", "oracle", "value", "value-floor", "lprog-only"]
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, policies=pol, regions=REG, rounds=rounds, drift_every=drift_every,
        drift_cycle=[int(x) for x in drift_cycle.split(",") if x.strip()],
        phi_mag=phi_mag, frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        budget=budget, buf_cap=buf_cap, monitor_n=monitor_n, lp_steps=lp_steps, replay_n=replay_n,
        finetune_lr=finetune_lr, finetune_steps=finetune_steps, reactive_every=reactive_every,
        err_floor=err_floor, value_floor=value_floor,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_directed_loop.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "directed_loop_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
