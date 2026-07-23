"""E1 -- Cut 4c-arm re-adaptation, run under THREE collection modes on a milestone grid refined
below 400. Does the "step-like recovery" survive when the body must earn its data?

Program: `ideas/two_timescale_value_loop.md`. Standing memo: [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md).
Parents:
  * [`../ballistic/arm/arm_readapt.py`](../ballistic/arm/arm_readapt.py) -- Cut 4c-arm, the endogenous
    reward-free re-adaptation loop (curl drift b0=0 -> b1=6). Its 3-seed result is 5.24x ± 0.18
    ballistic-over-reactive recovery gain, ballistic reaching the matched-FM ceiling. **This script
    re-runs that experiment; `arm_readapt.py` itself is untouched and stays reproducible.**
  * [`coverage_probe.py`](coverage_probe.py) -- E0, which priced the collection flag and, via the
    B0m/B0r oracle-teleport controls, found that MOST of on-policy's task-probe advantage was a
    MISTUNED TELEPORT KNOB (the default samples a 12.7-rad/s, wide-posture region the task never
    enters), NOT embodiment. What DID survive as structural to embodiment was command-state
    entanglement (|corr(u,s)| ~0.7 vs teleport's ~0.03).

THE QUESTION. `COLLECTION_REALISM.md` §2 (Tier B) records a live hypothesis: 4c-arm's recovery is a
STEP (96% recovered by the first non-zero milestone, m=400) rather than the graded curve a harder
plant "should" produce, and *"omniscient i.i.d. sampling repairs the model everywhere at once, so a
step is exactly what it should produce. On-policy coverage grows as the agent explores, which would
naturally stretch the step into a trajectory."* From `arm_readapt`'s own logs the entire step is
INSIDE the first milestone (ballistic 0.373 -> 0.086 at m=400 -> 0.086 at m=14000), so at the old
grid the question is unanswerable in either mode. This refines the grid and swaps the collection
mode.

THREE ARMS -- identical in EVERYTHING except how the re-adaptation buffer is filled (same plant,
FM, per-milestone training, probes, eval geometry, controllers, milestone counts, and env-step
budget, since on-policy costs one step per transition exactly like teleport):

  * `teleport`          -- the ORIGINAL 4c-arm collection (q_range=0.9, v_explore=8, i.i.d.
                           commands). The baseline the step was first measured under.
  * `teleport_matched`  -- E0's fix (STEP 1), applied: still teleport (free, discontinuous, i.i.d.
                           commands, |corr|~0) but resampling (q, qd) from a reference pool of
                           COMPETENT reaches in the b1 world -- the B0r recipe E0 validated. The
                           honest teleport baseline: an oracle teleporter told exactly where the
                           task lives. Privileged by construction (like `directed_loop`'s oracle),
                           and that is the point.
  * `on_policy`         -- the body EARNS its data: each chunk is gathered by ballistic reaches
                           under the CURRENT (re-adapting) FM, so early data is collected while the
                           model is still wrong about the field -- the genuine bootstrap. This is
                           the mode `COLLECTION_REALISM.md` §3 makes the default for new cuts.

THE THREE-WAY READ (this is why `teleport_matched` is not optional):
  * step in all three            -> recovery is just FAST on this plant; not a teleport artifact.
  * step in both teleport arms,
    stretches only on-policy      -> the memo's hypothesis, CONFIRMED: earned/bootstrapped coverage
                                     is what stretches it, not sampling breadth.
  * step in default teleport only,
    gone in matched + on-policy    -> it was a sampling-REGION artifact (a different mechanism than
                                     the memo guessed, and E0's knob finding carried into a live cut).

METRIC. Per milestone, per controller: recovery(m) = (stale - ctrl(m)) / (stale - ceiling), so
1.0 = fully recovered. The step-vs-curve summary is `recovery(400)` and the share of total recovery
banked in the FIRST non-zero milestone. Graded also by the value-relevant FM task-probe error
(`drift_value_loop` Cut 3's messenger), which this node trusts over control as an FM-quality read.

Held to `arm_readapt`/P5 byte-identical: geometry, damping, gear, frame_skip, FM, probes, CEM sizing
(k_shoot=1024/cem_iters=8 to the 42-dim action sequence -- P3's warning). BC and the aftereffect are
`arm_readapt`'s readouts and are NOT re-run here; this experiment is only about the recovery-curve
SHAPE under collection mode.

Run:
    cd experiments/
    modal run mjc/on_policy/readapt_both_ways.py::readapt_both_ways --quick        # smoke
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/readapt_both_ways.py::readapt_both_ways --tag rbw_s$s --seed $s
    done
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_readapt_both_ways(cfg: dict) -> dict:
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
    print(f"[setup] device={device} n={n} H={H} curl {b0}->{b1} arms={cfg['arms']} "
          f"milestones={cfg['milestones']}", flush=True)

    def make_env(b):
        return ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                           link_masses=cfg["link_masses"][:n],
                           joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                           curl_field={"b": float(b)}))

    env0, env1 = make_env(b0), make_env(b1)
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

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

    # normalisation fixed once from the b1 operating dynamics (shared by every snapshot & arm)
    nS, nU, nS2 = teleport_pool(env1, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ------------------------------------------------------------------ CEM (tip-space cost)
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

    # ------------------------------------------------------------------ eval geometry (shared)
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
        print(f"[geometry] mean reach distance = "
              f"{float(np.mean(np.linalg.norm(goals - t0, axis=1))):.3f} m", flush=True)
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

    # ------------------------------------------------------------------ probes + ceiling (shared)
    # THE probe correction (arm_substrate finding #1): FM error graded on a matched-FM reach's own
    # transitions, not the broad pool. Built from a reference FM trained in the true (b1) world.
    ref_net = _mlp(cfg["seed"] + 40)
    rS, rU, rS2 = teleport_pool(env1, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    rec = []
    ref_plan = make_plan_fn(ref_net, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 6000))
    rollout(ref_plan, 1, env1, record=rec)
    rollout(make_plan_fn(ref_net, cfg["k_shoot"], cfg["cem_iters"],
                         np.random.default_rng(cfg["seed"] + 6001)), H, env1, record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    pT = (np.array([r[2] for r in rec], np.float32) - pS).astype(np.float32)
    bS, bU, bS2 = teleport_pool(env1, cfg["probe_n"], cfg["seed"] + 12)
    bT = (bS2 - bS).astype(np.float32)

    def fm_err_task(net):
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    def fm_err_broad(net):
        return float(np.linalg.norm(fm_delta(net, bS, bU) - bT, axis=1).mean())

    # ---- the REFERENCE ON-POLICY POOL for `teleport_matched` (STEP 1's fix), the B0r recipe: -----
    # competent reaches in b1 under the matched (reference) FM. This is the task manifold teleport
    # should have been sampling all along; teleport_matched resamples (q, qd) from it. An
    # experimenter instrument (like the probes) -- collected once, off every arm's re-adaptation
    # budget.
    op_plan = make_plan_fn(ref_net, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                           np.random.default_rng(cfg["seed"] + 6100))
    op_beh = ReachBehaviour(op_plan, np.zeros((cfg["n_par"], 2), np.float32),
                            np.random.default_rng(cfg["seed"] + 6101),
                            sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
    oprefS, oprefU, oprefS2 = collect_pool(
        env1, cfg["ref_op_n"], np.random.default_rng(cfg["seed"] + 6102), fs, qc, cfg["op_q_range"],
        None, collection_mode="on_policy", behaviour=op_beh, goal_sampler=goal_sampler,
        ep_len=cfg["ep_len"], n_par=cfg["n_par"], v0_std=cfg["v0_std"], wrap_limit=cfg["wrap_limit"])
    print(f"[ref-op] competent-reach reference pool {len(oprefS)} transitions "
          f"(mean speed {np.linalg.norm(oprefS[:, n:], axis=1).mean():.2f} rad/s)", flush=True)

    # ceiling: fresh matched FM in b1 (its own control numbers are the recovery target)
    fm_ceil = _mlp(cfg["seed"] + 42)
    Sc, Uc, S2c = teleport_pool(env1, cfg["pool_n"], cfg["seed"] + 20)
    train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
    ceil = {"fm_err_task": fm_err_task(fm_ceil), "fm_err_broad": fm_err_broad(fm_ceil),
            "reactive": rollout(make_plan_fn(fm_ceil, cfg["k_shoot"], cfg["cem_iters"],
                                             np.random.default_rng(cfg["seed"] + 7000)), 1, env1),
            "ballistic_cem": rollout(make_plan_fn(fm_ceil, cfg["k_shoot"], cfg["cem_iters"],
                                                  np.random.default_rng(cfg["seed"] + 7001)), H, env1)}
    print(f"[ceiling] matched-b1 FM: task={ceil['fm_err_task']:.4f} reactive={ceil['reactive']:.4f} "
          f"ballistic={ceil['ballistic_cem']:.4f}", flush=True)

    # ------------------------------------------------------------------ the stale pre-drift FM
    S0, U0, S20 = teleport_pool(env0, cfg["pool_n"], cfg["seed"] + 10)

    def grade(net):
        r = {"fm_err_task": fm_err_task(net), "fm_err_broad": fm_err_broad(net)}
        r["reactive"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                             np.random.default_rng(cfg["seed"] + 7000)), 1, env1)
        r["ballistic_cem"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                  np.random.default_rng(cfg["seed"] + 7001)), H, env1)
        return r

    # ================================================================= #
    # one arm's full milestone ladder
    # ================================================================= #
    def fill(mode, fm, need, rng_buf, k):
        """Return `need` fresh transitions in the b1 world, gathered per `mode`."""
        if mode == "teleport":
            return collect_pool(env1, need, rng_buf, fs, qc, cfg["q_range"], cfg["v_explore"])
        if mode == "teleport_matched":
            idx = rng_buf.integers(0, len(oprefS), size=need)
            S = np.empty((need, SD), np.float32); U = np.empty((need, AD), np.float32)
            S2 = np.empty((need, SD), np.float32)
            for i in range(need):
                env1.set_state(oprefS[idx[i], :n].astype(np.float64),
                               oprefS[idx[i], n:].astype(np.float64))
                u = rng_buf.uniform(-1, 1, AD).astype(np.float32)
                S[i] = env1.get_state(); s2, _ = env1.step(u, fs); U[i] = u; S2[i] = s2
            return S, U, S2
        if mode == "on_policy":
            # THE BOOTSTRAP: plan with the CURRENT (still-adapting) FM. Early chunks are gathered
            # while the model is wrong about the field, so the reaches are systematically deviated
            # and the data is off the competent-reach manifold -- which is exactly the gradual,
            # earned coverage the memo's hypothesis is about.
            plan_fn = make_plan_fn(fm, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                                   np.random.default_rng(cfg["seed"] + 8000 + k))
            beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng_buf,
                                 sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
            return collect_pool(env1, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                collection_mode="on_policy", behaviour=beh,
                                goal_sampler=goal_sampler, ep_len=cfg["ep_len"],
                                n_par=cfg["n_par"], v0_std=cfg["v0_std"],
                                wrap_limit=cfg["wrap_limit"])
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
            if need > 0:
                env1.reset_wrap()
                aS, aU, aS2 = fill(mode, fm, need, rng_buf, k)
                bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU])
                bufS2 = np.concatenate([bufS2, aS2])
                train_steps(fm, opt, bufS, bufU, bufS2, cfg["finetune_steps"],
                            np.random.default_rng(cfg["seed"] + 600 + k))
            r = grade(copy.deepcopy(fm)); r["transitions"] = int(m)
            if need > 0:
                d = pool_diagnostics(aS, aU, nq=n, refS=rS, refU=rU, seed=cfg["seed"])
                r["corr"] = d.get("max_abs_corr"); r["speed"] = d.get("speed_mean")
                r["cover"] = d.get("ref_to_pool_mean")
            ladder.append(r)
            print(f"[{mode:>16s} m={m:6d}] task={r['fm_err_task']:.4f} "
                  f"reactive={r['reactive']:.4f} ballistic={r['ballistic_cem']:.4f}"
                  + (f"  |corr|={r.get('corr', float('nan')):.3f} "
                     f"speed={r.get('speed', float('nan')):.2f}" if need > 0 else ""), flush=True)
        return ladder

    results = {}
    for mode in cfg["arms"]:
        print(f"\n=== arm: {mode} ===", flush=True)
        results[mode] = run_arm(mode)

    # ================================================================= #
    # step-vs-curve summary
    # ================================================================= #
    def recovery(mode, key):
        st = results[mode][0][key]; cl = ceil[key]
        den = st - cl
        return [((st - r[key]) / den if abs(den) > 1e-9 else float("nan")) for r in results[mode]]

    ms = cfg["milestones"]
    first_nz = next((i for i, m in enumerate(ms) if m > 0), 1)
    m400_idx = min(range(len(ms)), key=lambda i: abs(ms[i] - 400))
    print("\n" + "=" * 92, flush=True)
    print("=== recovery(m): fraction of the stale->ceiling gap closed (1.0 = fully recovered) ===",
          flush=True)
    summary = {}
    for key in ("ballistic_cem", "reactive", "fm_err_task"):
        print(f"\n[{key}]  milestones = {ms}", flush=True)
        print(f"    ceiling={ceil[key]:.4f}", flush=True)
        for mode in cfg["arms"]:
            rc = recovery(mode, key)
            stale = results[mode][0][key]
            frac_first = rc[first_nz] / rc[-1] if abs(rc[-1]) > 1e-9 else float("nan")
            summary.setdefault(mode, {})[key] = {
                "stale": stale, "final": results[mode][-1][key],
                "recovery_curve": rc, "recovery_at_400": rc[m400_idx],
                "share_in_first_milestone": frac_first}
            print(f"    {mode:>16s} stale={stale:.4f}  " + " ".join(f"{v:+.2f}" for v in rc),
                  flush=True)
            print(f"    {'':>16s}   -> recovery@m{ms[m400_idx]}={rc[m400_idx]:.2f}, "
                  f"{100 * frac_first:.0f}% of total recovery banked by m{ms[first_nz]} "
                  f"({'STEP' if frac_first > 0.8 else 'curve'})", flush=True)

    # the dissociation, reproduced per arm: does allocation stay ballistic-specific?
    print("\n=== ballistic-over-reactive recovery gain per arm (pusher 4c: 4.3x; arm 4c: 5.24x) ===",
          flush=True)
    for mode in cfg["arms"]:
        gb = results[mode][0]["ballistic_cem"] - results[mode][-1]["ballistic_cem"]
        gr = results[mode][0]["reactive"] - results[mode][-1]["reactive"]
        summary[mode]["ballistic_over_reactive_gain"] = gb / gr if abs(gr) > 1e-9 else float("nan")
        print(f"    {mode:>16s} ballistic gain={gb:+.4f}  reactive gain={gr:+.4f}  "
              f"ratio={gb / gr if abs(gr) > 1e-9 else float('nan'):.2f}x", flush=True)

    out = {"config": cfg, "ceiling": ceil, "results": results, "summary": summary,
           "milestones": ms}
    outdir = os.path.join(DATA_DIR, "readapt_both_ways", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def readapt_both_ways(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "teleport,teleport_matched,on_policy",
    # --- the drift + the refined grid (4 points below the old first milestone of 400) ---
    b0: float = 0.0,
    b1: float = 6.0,
    milestones: str = "0,50,100,200,400,800,1600,3200,6400,14000",
    # --- arm geometry: arm_readapt's n=3 curl design point, byte-identical ---
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- on-policy collection knobs (E0's, byte-identical) ---
    ep_len: int = 14,
    op_q_range: float = 0.25,
    ref_op_n: int = 14000,
    n_par: int = 16,
    sigma_u: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14,
    wrap_limit: float = 3.0,
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    finetune_steps: int = 1500,
    pool_n: int = 14000,
    probe_n: int = 1500,
    # --- control eval (P3 sizing) ---
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
    out = run_readapt_both_ways.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "readapt_both_ways_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
