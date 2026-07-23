"""E2 -- the LOCAL-drift re-adaptation, the test E1 predicted. Does making the drift spatial
stretch the on-policy recovery that a GLOBAL drift did not?

Program: `ideas/two_timescale_value_loop.md`. Memo: [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md).
Parents:
  * [`readapt_both_ways.py`](readapt_both_ways.py) -- E1, the same experiment under a GLOBAL curl.
    Its result: the recovery "step" is real and even SHARPER on the task manifold (~50 transitions),
    on-policy did NOT stretch it, and the 5x ballistic-over-reactive dissociation survived all three
    collection modes. The diagnostic reason on-policy didn't stretch: a global curl is learnable
    from ANY motion, so coverage growth doesn't gate repair.
  * [`coverage_probe.py`](coverage_probe.py) -- E0 (the flag, the ladder, the mistuned-knob finding).

THE PREDICTION E1 SET UP. "On-policy should stretch the control step only under a LOCAL drift,
where *where you are* gates *what you can learn*." This makes the drift local -- a Gaussian-gated
curl (`arm_env` `curl_field["center"]`, added for this) that acts only when the HAND is in one
region of the workspace -- and asks whether that flips E1's null into the memo's predicted stretch.

WHY LOCAL CHANGES THE MECHANISM. Under a global curl every moving transition teaches the field, so
even a stale-FM reach re-adapts fast (E1). Under a local curl, only transitions taken IN THE REGION
inform it, and a body accumulates in-region visits only by reaching through the region -- coverage
GROWS over time. A teleporter (broad OR task-matched) has in-region samples available immediately.
So the memo's "coverage grows as you explore" gets a place to bite.

THREE ARMS (identical except how the re-adaptation buffer is filled -- see E1):
  * `teleport`          -- broad teleport (q_range=0.9); samples the region from step 1.
  * `teleport_matched`  -- E0's fix: resample from a competent-reach reference pool; in-region
                           samples present immediately, in task-relevant proportion.
  * `on_policy`         -- reaches under the current (adapting) FM; in-region visits ACCUMULATE.

THE KEY READOUT is NOT overall control. E1 established control SATURATES (drift_value_loop Cut 3:
control is a near-blind grader), so it cannot see gradual FM improvement. The clean readout is the
FM task-probe error SPLIT BY REGION:
  * IN-REGION FM error   -- recovery here should be GRADUAL under on_policy (must visit to learn)
                            and FAST under the teleporters (samples present from the start).
  * OUT-REGION FM error   -- should be ~floor for everyone (nothing changed there).
plus, as the mechanism, each arm's IN-REGION COLLECTION FRACTION per milestone: does on_policy's
in-region share start low and grow, while the teleporters' is flat?

Region membership: a transition is IN-REGION iff its tip (FK of qpos) is within `region_k * sigma`
of the curl center. The task probe is partitioned once, up front, and reused for every snapshot so
the x-axis is controlled.

Geometry note (the 4a warning, `COLLECTION_REALISM.md` §"the one time we tried it"): eval reaches
are random-direction (no fixed corridor), and the region sits on PART of the reach fan so some
reaches pass through it and some do not -- the `region_frac` diagnostic reports what fraction of a
matched reach's transitions land in it, so the geometry can be checked rather than assumed.

Run:
    cd experiments/
    modal run mjc/on_policy/readapt_local.py::readapt_local --quick        # smoke
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/readapt_local.py::readapt_local --tag loc_s$s --seed $s
    done
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_readapt_local(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from mjc.embodied import ReachBehaviour, make_arm_goal_sampler, pool_diagnostics

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    b0, b1 = cfg["b0"], cfg["b1"]
    B = cfg["n_eval"]
    center = np.asarray(cfg["curl_center"], dtype=np.float64)
    csig = float(cfg["curl_sigma"]); rthr = cfg["region_k"] * csig
    print(f"[setup] device={device} n={n} H={H} LOCAL curl b={b1} center={center.tolist()} "
          f"sigma={csig} region<{rthr:.2f}m arms={cfg['arms']} milestones={cfg['milestones']}",
          flush=True)

    def make_env(b, local):
        cf = {"b": float(b)}
        if local and b != 0.0:
            cf["center"] = tuple(center.tolist()); cf["sigma"] = csig
        return ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                           link_masses=cfg["link_masses"][:n],
                           joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                           curl_field=cf))

    env0 = make_env(b0, local=False)      # pre-drift field-free world
    env1 = make_env(b1, local=True)       # post-drift world: curl acts only IN the region
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    def in_region(states):
        """Boolean mask: is the tip (FK of the qpos block) inside the drift region?"""
        tips = fk(np.asarray(states)[:, :n].astype(np.float64), Ls)
        return np.linalg.norm(tips - center, axis=1) < rthr

    # ------------------------------------------------------------------ FM f(s,u)->Δs
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(SD + AD, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, SD)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32), device=device)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def teleport_pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs,
                            qc, cfg["q_range"], cfg["v_explore"])

    nS, nU, nS2 = teleport_pool(env1, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ------------------------------------------------------------------ CEM
    def make_plan_fn(net, k_shoot, cem_iters, rng):
        def plan(states, goals):
            Bn = states.shape[0]
            mu = np.zeros((Bn, H, AD), np.float32)
            sig = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(np.asarray(goals, np.float32), device=device).repeat_interleave(k_shoot, 0)
            s0 = torch.tensor(np.asarray(states, np.float32), device=device).repeat_interleave(k_shoot, 0)
            for _ in range(cem_iters):
                e = rng.standard_normal((Bn, k_shoot, H, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * k_shoot, H, AD), device=device)
                    cost = torch.zeros(Bn * k_shoot, device=device)
                    for h in range(H):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                        cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, k_shoot), cfg["cem_elite"], dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)
        return plan

    goal_sampler = make_arm_goal_sampler(Ls, cfg["reach_amp"], cfg["reach_lo"],
                                         cfg["reach_hi"], cfg["reach_tries"])

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (B, n))
        t0 = fk(q0, Ls)
        qg = np.empty_like(q0)
        lo, hi = cfg["reach_lo"], cfg["reach_hi"]
        for b in range(B):
            best, best_pen = None, np.inf
            for _ in range(cfg["reach_tries"]):
                d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                cand = q0[b] + cfg["reach_amp"] * d
                dist = float(np.linalg.norm(fk(cand, Ls) - t0[b]))
                pen = max(0.0, lo - dist) + max(0.0, dist - hi)
                if pen < best_pen:
                    best, best_pen = cand, pen
                if pen == 0.0:
                    break
            qg[b] = best
        goals = fk(qg, Ls).astype(np.float32)
        starts = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (B, n))], 1).astype(np.float32)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every, cenv, record=None):
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                cenv.set_state(states[b, :n].astype(np.float64), states[b, n:].astype(np.float64))
                s0b = cenv.get_state()
                s2, _ = cenv.step(acts[b], fs)
                if record is not None:
                    record.append((s0b, acts[b].astype(np.float32), s2))
                states[b] = s2
        tips = fk(states[:, :n].astype(np.float64), Ls).astype(np.float32)
        return float(np.median(np.linalg.norm(tips - ev_goals, axis=1)))

    # ------------------------------------------------------------------ probes + ceiling
    ref_net = _mlp(cfg["seed"] + 40)
    rS, rU, rS2 = teleport_pool(env1, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    rec = []
    rollout(make_plan_fn(ref_net, cfg["k_shoot"], cfg["cem_iters"],
                         np.random.default_rng(cfg["seed"] + 6000)), 1, env1, record=rec)
    rollout(make_plan_fn(ref_net, cfg["k_shoot"], cfg["cem_iters"],
                         np.random.default_rng(cfg["seed"] + 6001)), H, env1, record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    pT = (np.array([r[2] for r in rec], np.float32) - pS).astype(np.float32)
    # THE PARTITION: which matched-reach transitions live in the drifted region.
    pmask = in_region(pS)
    region_frac = float(pmask.mean())
    print(f"[probe] {len(pS)} matched-reach transitions; {region_frac:.0%} land IN the drift "
          f"region (want ~0.2-0.5: some reaches pass through it, some do not)", flush=True)

    def fm_err_in(net):
        if pmask.sum() == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(net, pS[pmask], pU[pmask]) - pT[pmask], axis=1).mean())

    def fm_err_out(net):
        m = ~pmask
        if m.sum() == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(net, pS[m], pU[m]) - pT[m], axis=1).mean())

    def fm_err_task(net):
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    # reference on-policy pool for teleport_matched (B0r recipe), collected in the LOCAL-field world
    op_beh = ReachBehaviour(make_plan_fn(ref_net, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                                         np.random.default_rng(cfg["seed"] + 6100)),
                            np.zeros((cfg["n_par"], 2), np.float32),
                            np.random.default_rng(cfg["seed"] + 6101),
                            sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
    oprefS, oprefU, oprefS2 = collect_pool(
        env1, cfg["ref_op_n"], np.random.default_rng(cfg["seed"] + 6102), fs, qc, cfg["op_q_range"],
        None, collection_mode="on_policy", behaviour=op_beh, goal_sampler=goal_sampler,
        ep_len=cfg["ep_len"], n_par=cfg["n_par"], v0_std=cfg["v0_std"], wrap_limit=cfg["wrap_limit"])
    print(f"[ref-op] competent-reach pool {len(oprefS)}; in-region share "
          f"{in_region(oprefS).mean():.0%}", flush=True)

    # THE CEILING IS TASK-MATCHED, NOT BROAD-TELEPORT (E0's lesson, load-bearing here). Under a
    # LOCAL field, a broad-teleport FM is doubly mistuned in-region: it samples the region rarely
    # AND at velocities the task never uses -- so a broad-teleport "ceiling" scores WORSE in-region
    # than a task-matched FM, and the recovery-fraction normalisation (stale - ceiling) sign-flips.
    # (E0 measured exactly this: refTP task error 0.30 vs refOP 0.08.) So the ceiling is trained on
    # the competent-reach reference pool -- the same task manifold `teleport_matched` draws from,
    # which is the honest floor for a task-relevant, in-region metric. (E1 used broad teleport for
    # its ceiling and reproduced P7's 0.0993 exactly -- correct there because the field was global,
    # where broad teleport is a fine reference. That is the E1-vs-E2 difference, not an
    # inconsistency.)
    fm_ceil = _mlp(cfg["seed"] + 42)
    train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                oprefS, oprefU, oprefS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
    ceil = {"fm_err_task": fm_err_task(fm_ceil), "fm_err_in": fm_err_in(fm_ceil),
            "fm_err_out": fm_err_out(fm_ceil),
            "reactive": rollout(make_plan_fn(fm_ceil, cfg["k_shoot"], cfg["cem_iters"],
                                             np.random.default_rng(cfg["seed"] + 7000)), 1, env1),
            "ballistic_cem": rollout(make_plan_fn(fm_ceil, cfg["k_shoot"], cfg["cem_iters"],
                                                  np.random.default_rng(cfg["seed"] + 7001)), H, env1)}
    print(f"[ceiling] task-matched FM: task={ceil['fm_err_task']:.4f} in={ceil['fm_err_in']:.4f} "
          f"out={ceil['fm_err_out']:.4f} ballistic={ceil['ballistic_cem']:.4f}", flush=True)

    S0, U0, S20 = teleport_pool(env0, cfg["pool_n"], cfg["seed"] + 10)

    def grade(net):
        return {"fm_err_task": fm_err_task(net), "fm_err_in": fm_err_in(net),
                "fm_err_out": fm_err_out(net),
                "reactive": rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                 np.random.default_rng(cfg["seed"] + 7000)), 1, env1),
                "ballistic_cem": rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                      np.random.default_rng(cfg["seed"] + 7001)), H, env1)}

    # ================================================================= #
    def fill(mode, fm, need, rng_buf, k):
        if mode == "teleport":
            return collect_pool(env1, need, rng_buf, fs, qc, cfg["q_range"], cfg["v_explore"])
        if mode == "teleport_matched":
            idx = rng_buf.integers(0, len(oprefS), size=need)
            S = np.empty((need, SD), np.float32); U = np.empty((need, AD), np.float32)
            S2 = np.empty((need, SD), np.float32)
            for i in range(need):
                env1.set_state(oprefS[idx[i], :n].astype(np.float64), oprefS[idx[i], n:].astype(np.float64))
                u = rng_buf.uniform(-1, 1, AD).astype(np.float32)
                S[i] = env1.get_state(); s2, _ = env1.step(u, fs); U[i] = u; S2[i] = s2
            return S, U, S2
        if mode == "on_policy":
            plan_fn = make_plan_fn(fm, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                                   np.random.default_rng(cfg["seed"] + 8000 + k))
            beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng_buf,
                                 sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
            return collect_pool(env1, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                collection_mode="on_policy", behaviour=beh,
                                goal_sampler=goal_sampler, ep_len=cfg["ep_len"],
                                n_par=cfg["n_par"], v0_std=cfg["v0_std"], wrap_limit=cfg["wrap_limit"])
        raise ValueError(mode)

    def run_arm(mode):
        fm = _mlp(cfg["seed"] + 41); opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
        train_steps(fm, opt, S0, U0, S20, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
        rng_buf = np.random.default_rng(cfg["seed"] + 500)
        bufS = np.zeros((0, SD), np.float32); bufU = np.zeros((0, AD), np.float32)
        bufS2 = np.zeros((0, SD), np.float32)
        ladder = []
        for k, m in enumerate(cfg["milestones"]):
            need = m - len(bufS)
            in_share = float("nan")
            if need > 0:
                aS, aU, aS2 = fill(mode, fm, need, rng_buf, k)
                in_share = float(in_region(aS).mean())      # the mechanism: in-region coverage
                bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU])
                bufS2 = np.concatenate([bufS2, aS2])
                train_steps(fm, opt, bufS, bufU, bufS2, cfg["finetune_steps"],
                            np.random.default_rng(cfg["seed"] + 600 + k))
            r = grade(copy.deepcopy(fm)); r["transitions"] = int(m); r["in_share"] = in_share
            ladder.append(r)
            print(f"[{mode:>16s} m={m:6d}] task={r['fm_err_task']:.4f} IN={r['fm_err_in']:.4f} "
                  f"OUT={r['fm_err_out']:.4f} ballistic={r['ballistic_cem']:.4f}"
                  + (f"  in-region collected={in_share:.0%}" if need > 0 else ""), flush=True)
        return ladder

    results = {}
    for mode in cfg["arms"]:
        print(f"\n=== arm: {mode} ===", flush=True)
        results[mode] = run_arm(mode)

    # ================================================================= #
    # step-vs-curve, focused on the IN-REGION FM error (the readout control cannot see)
    # ================================================================= #
    def recovery(mode, key):
        st = results[mode][0][key]; cl = ceil[key]; den = st - cl
        return [((st - r[key]) / den if abs(den) > 1e-9 else float("nan")) for r in results[mode]]

    ms = cfg["milestones"]
    first_nz = next(i for i, m in enumerate(ms) if m > 0)
    m400 = min(range(len(ms)), key=lambda i: abs(ms[i] - 400))
    print("\n" + "=" * 92, flush=True)
    print(f"=== recovery(m): fraction of gap closed. region={region_frac:.0%} of matched reaches ===",
          flush=True)
    summary = {}
    for key in ("fm_err_in", "fm_err_out", "ballistic_cem", "fm_err_task"):
        print(f"\n[{key}]  ceiling={ceil[key]:.4f}  milestones={ms}", flush=True)
        for mode in cfg["arms"]:
            rc = recovery(mode, key)
            share = rc[first_nz] / rc[-1] if abs(rc[-1]) > 1e-9 else float("nan")
            summary.setdefault(mode, {})[key] = {
                "stale": results[mode][0][key], "final": results[mode][-1][key],
                "recovery_curve": rc, "recovery_at_400": rc[m400],
                "share_in_first_milestone": share}
            print(f"    {mode:>16s} " + " ".join(f"{v:+.2f}" for v in rc)
                  + f"   | m{ms[first_nz]}-share={100 * share:.0f}% "
                  f"[{'STEP' if share > 0.8 else 'curve'}]", flush=True)

    # the mechanism: does on_policy's in-region collection share start low and GROW?
    print("\n=== in-region COLLECTION share per milestone (the coverage-growth mechanism) ===",
          flush=True)
    for mode in cfg["arms"]:
        shares = [r.get("in_share", float("nan")) for r in results[mode][1:]]
        summary[mode]["in_share_trace"] = shares
        print(f"    {mode:>16s} " + " ".join(
            f"{s:.0%}" if s == s else "  -" for s in shares), flush=True)

    print("\n=== ballistic-over-reactive recovery gain per arm ===", flush=True)
    for mode in cfg["arms"]:
        gb = results[mode][0]["ballistic_cem"] - results[mode][-1]["ballistic_cem"]
        gr = results[mode][0]["reactive"] - results[mode][-1]["reactive"]
        summary[mode]["ballistic_over_reactive_gain"] = gb / gr if abs(gr) > 1e-9 else float("nan")
        print(f"    {mode:>16s} ratio={gb / gr if abs(gr) > 1e-9 else float('nan'):.2f}x", flush=True)

    # the headline contrast for E2: in-region FM-error first-milestone share, on_policy vs matched
    print("\n=== E2 verdict: IN-REGION FM-error first-milestone share (LOWER on_policy = the "
          "predicted stretch) ===", flush=True)
    for mode in cfg["arms"]:
        s = summary[mode]["fm_err_in"]["share_in_first_milestone"]
        print(f"    {mode:>16s}: {100 * s:.0f}%", flush=True)

    out = {"config": cfg, "ceiling": ceil, "results": results, "summary": summary,
           "milestones": ms, "region_frac": region_frac}
    outdir = os.path.join(DATA_DIR, "readapt_local", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def readapt_local(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "teleport,teleport_matched,on_policy",
    b0: float = 0.0,
    b1: float = 6.0,
    milestones: str = "0,50,100,200,400,800,1600,3200,6400,14000",
    # --- the LOCAL drift region (tip-space). Defaults sit on part of the reach fan; the
    #     `region_frac` print verifies ~20-50% of matched-reach transitions land inside. ---
    curl_center: str = "0.78,0.55",
    curl_sigma: float = 0.25,
    region_k: float = 1.4,               # in-region iff ‖tip - center‖ < region_k * sigma
    # --- arm geometry: 4c-arm's n=3 curl design point, byte-identical ---
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    ep_len: int = 14,
    op_q_range: float = 0.25,
    ref_op_n: int = 14000,
    n_par: int = 16,
    sigma_u: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14,
    wrap_limit: float = 3.0,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    finetune_steps: int = 1500,
    pool_n: int = 14000,
    probe_n: int = 1500,
    n_eval: int = 48,
    plan_h: int = 14,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    q_jit: float = 0.25,
    v0_std: float = 0.0,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    reach_tries: int = 40,
):
    import os

    ms = [int(x) for x in milestones.split(",") if x.strip()]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    if quick:
        ms = [0, 100, 400, 2000]
        pool_n = 3000; fm_steps = 1500; finetune_steps = 400; probe_n = 600; ref_op_n = 2000
        n_eval = 12; k_shoot = 256; cem_iters = 4
        collect_k_shoot = 128; collect_cem_iters = 3
        fm_hidden = 128; fm_layers = 2; n_par = 8
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, b0=b0, b1=b1, milestones=ms,
        curl_center=[float(x) for x in curl_center.split(",")], curl_sigma=curl_sigma,
        region_k=region_k,
        n_links=n_links,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        ep_len=ep_len, op_q_range=op_q_range, ref_op_n=ref_op_n, n_par=n_par, sigma_u=sigma_u,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        collect_replan_every=collect_replan_every, wrap_limit=wrap_limit,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, finetune_steps=finetune_steps, pool_n=pool_n, probe_n=probe_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters,
        cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, reach_tries=reach_tries,
    )
    out = run_readapt_local.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "readapt_local_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
