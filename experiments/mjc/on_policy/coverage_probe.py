"""E0 -- pricing the collection flag: what does on-policy acquisition actually cost?

Program: `ideas/two_timescale_value_loop.md`. Standing memo: [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md).
Machinery: [`../embodied.py`](../embodied.py). Plant: [`../arm_env.py`](../arm_env.py) at
`ballistic/arm/`'s Cut-4c-arm design point, byte-identical, so the only thing varying is HOW the
transitions were obtained.

This is characterization, not a cut -- the analogue of `arm_substrate` P0-P6. It prices the flag
BEFORE a cut is spent on it, because the memo's recommendation ("make on_policy the default for
new cuts") is only sound if we know what the default costs.

THE LADDER (`embodied.py`): four rungs, matched EXACTLY on environment steps.

    B0  teleport                          -- unchanged; the honest baseline
    B0m teleport, on-policy MARGINALS     -- "a teleporter told roughly where to look"
    B0r teleport, on-policy STATES resampled -- "a teleporter told exactly where to look"
    B1  OU random-torque episodes         -- continuity WITHOUT the model in the loop
    B2  reaches under a FROZEN reference planner  -- on-task, decoupled from model quality
    B3  reaches under the LIVE FM         -- the real phenomenon, incl. bootstrapping

B0 vs B3 alone would confound four separable effects: WHERE the samples sit (posture and velocity
band), whether they CHAIN, whether the commands are i.i.d., and the model<->data coupling. The
ladder decomposes them. B0m/B0r are the ones that matter for reading anything into B1-B3's
advantage -- they give teleport the on-policy state distribution for free, so whatever survives is
attributable to trajectories rather than to sampling the right region. Read B0->B0m->B0r->B2 as a
chain, not as a contest: each step hands teleport more of what embodiment supplies.

WHAT IS HELD FIXED ACROSS EVERY RUNG -- the control that makes the comparison mean anything.
The agent's ACQUISITION changes; the experimenter's instruments do not.
  * normalisation statistics, from one fixed reference TELEPORT pool;
  * the TASK probe (transitions a matched-FM reach actually visits) and the BROAD probe (teleport,
    workspace-wide) -- both off-budget, both shared;
  * evaluation geometry, the planner config used for grading, the FM init seed, the number of
    training steps.
Grade both probes: `arm_substrate` finding #1 is that a broad probe turned this arc's effect into a
null on the arm (0.81x -> 24x), and here the two probes are the whole point -- they are predicted
to move in OPPOSITE directions across the ladder.

THREE PREDICTIONS THIS TESTS DIRECTLY
  1. Broad-probe error much worse under on-policy; task-probe error competitive or better. If so,
     the Tier-B correction sharpens from "the sample counts are inflated" to "teleport's advantage
     is on the broad probe and largely evaporates on the task probe".
  2. Part of the loss is COMMAND-CHANNEL IDENTIFIABILITY, not state coverage -- `max |corr(u,s)|`
     rises B0 -> B3 (Cut #2 measured 0.003 under i.i.d. commands), and `sigma_u` is the knob that
     trades it back.
  3. The ballistic-over-reactive dissociation is Tier A (both controllers share one FM) and should
     SURVIVE the mode change. If it grows, that is a finding, not noise: on-policy data is on-task,
     and on-task is where ballistic control lives.

Run:
    cd experiments/
    modal run mjc/on_policy/coverage_probe.py::coverage_probe --quick          # smoke
    modal run --detach mjc/on_policy/coverage_probe.py::coverage_probe --tag e0_s0 --seed 0
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_coverage_probe(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from mjc.embodied import (OUBehaviour, ReachBehaviour, make_arm_goal_sampler,
                              pool_diagnostics)

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    B = cfg["n_eval"]
    print(f"[setup] device={device} n_links={n} H={H} curl b={cfg['b']} rungs={cfg['rungs']} "
          f"n_list={cfg['n_list']}", flush=True)

    env = ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                      link_masses=cfg["link_masses"][:n],
                      joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                      curl_field={"b": float(cfg["b"])}))

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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)     # cut #4 gotcha (c)
            opt.step()
        net.eval()

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32), device=device)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def teleport_pool(nn_, seed):
        return collect_pool(env, nn_, np.random.default_rng(seed), fs,
                            qc, cfg["q_range"], cfg["v_explore"])

    # ================================================================= #
    # The experimenter's instruments -- teleport, off-budget, shared by every rung.
    # Built FIRST so nothing downstream can accidentally depend on a rung's own data.
    # ================================================================= #
    refS, refU, refS2 = teleport_pool(cfg["pool_n"], cfg["seed"] + 11)
    X = np.concatenate([refS, refU], 1).astype(np.float32)
    Y = (refS2 - refS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
    print(f"[ref] reference teleport pool {len(refS)} transitions -> norm stats", flush=True)

    # ------------------------------------------------------------------ CEM (tip-space cost)
    def make_plan_fn(net, k_shoot, cem_iters, rng, horizon=None):
        Hh = horizon or H

        def plan(states, goals):
            Bn = states.shape[0]
            mu = np.zeros((Bn, Hh, AD), np.float32)
            sig = np.full((Bn, Hh, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(np.asarray(goals, np.float32), device=device
                               ).repeat_interleave(k_shoot, 0)
            s0 = torch.tensor(np.asarray(states, np.float32), device=device
                              ).repeat_interleave(k_shoot, 0)
            for _ in range(cem_iters):
                e = rng.standard_normal((Bn, k_shoot, Hh, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * k_shoot, Hh, AD), device=device)
                    cost = torch.zeros(Bn * k_shoot, device=device)
                    for h in range(Hh):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                        cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, k_shoot), cfg["cem_elite"], dim=1
                                     ).indices.cpu().numpy()
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
        print(f"[geometry] mean task-space reach distance = "
              f"{float(np.mean(np.linalg.norm(goals - t0, axis=1))):.3f} m "
              f"(band [{lo}, {hi}])", flush=True)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every, record=None):
        """Executes in the TRUE world. This is the experimenter's grader, so it keeps the
        established `set_state`-multiplexed idiom (`ballistic/arm/arm_readapt.py::rollout`) and is
        deliberately NOT metered -- evaluation is an instrument, not an agent capability."""
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                cenv_state = states[b]
                env.set_state(cenv_state[:n].astype(np.float64), cenv_state[n:].astype(np.float64))
                s0b = env.get_state()
                s2, _ = env.step(acts[b], fs)
                if record is not None:
                    record.append((s0b, acts[b].astype(np.float32), s2))
                states[b] = s2
        tips = fk(states[:, :n].astype(np.float64), Ls).astype(np.float32)
        return float(np.median(np.linalg.norm(tips - ev_goals, axis=1)))

    # ------------------------------------------------------------------ the reference FM
    # Serves two jobs, both experimenter-side: it generates the TASK probe (the transitions a
    # matched-FM reach actually visits -- `arm_substrate` finding #1), and it is the FROZEN planner
    # that drives rung B2. Trained on the reference teleport pool, so B2's behaviour quality is
    # exogenous and identical no matter which rung's FM is being graded.
    ref_net = _mlp(cfg["seed"] + 40)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                refS, refU, refS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    eval_plan_ref = make_plan_fn(ref_net, cfg["k_shoot"], cfg["cem_iters"],
                                 np.random.default_rng(cfg["seed"] + 6000))
    rec = []
    rollout(eval_plan_ref, 1, record=rec)
    rollout(eval_plan_ref, H, record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    pT = (np.array([r[2] for r in rec], np.float32) - pS).astype(np.float32)
    bS, bU, bS2 = teleport_pool(cfg["probe_n"], cfg["seed"] + 12)
    bT = (bS2 - bS).astype(np.float32)
    print(f"[probes] task probe {len(pS)} transitions (matched-FM reaches); "
          f"broad probe {len(bS)} (teleport)", flush=True)

    def fm_err_task(net):
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    def fm_err_broad(net):
        return float(np.linalg.norm(fm_delta(net, bS, bU) - bT, axis=1).mean())

    # ------------------------------------------------------------------ the two REFERENCES
    # NOT one "ceiling". The smoke made the reason concrete: a large TELEPORT pool produces an FM
    # that is WORSE on the task probe (0.64) than an on-policy FM trained on 300 transitions
    # (0.21-0.34), because the teleport pool spends its capacity over a velocity band the task
    # never enters (`v_explore=8` -> collected speed ~13 rad/s; reaches live at ~3). Calling that a
    # ceiling would report every on-policy rung as beating the ceiling by a wide margin, which is
    # not a finding, it is a mislabelled axis. So there are two references, one per mode, each a
    # LARGE pool of its own kind, and each probe is read against the reference that is meant to
    # bound it: teleport bounds the broad probe, on-policy bounds the task probe.
    def grade_ref(net, name):
        r = {"fm_err_task": fm_err_task(net), "fm_err_broad": fm_err_broad(net)}
        if cfg["do_control"]:
            r["reactive"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                 np.random.default_rng(cfg["seed"] + 7000)), 1)
            r["ballistic_cem"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                      np.random.default_rng(cfg["seed"] + 7001)), H)
        print(f"[reference:{name}] task={r['fm_err_task']:.4f} broad={r['fm_err_broad']:.4f}"
              + (f" reactive={r['reactive']:.4f} ballistic={r['ballistic_cem']:.4f}"
                 if cfg["do_control"] else ""), flush=True)
        return r

    ref_tel = grade_ref(ref_net, f"teleport n={cfg['pool_n']}")

    op_plan = make_plan_fn(ref_net, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                           np.random.default_rng(cfg["seed"] + 6100))
    op_beh = ReachBehaviour(op_plan, np.zeros((cfg["n_par"], 2), np.float32),
                            np.random.default_rng(cfg["seed"] + 6101),
                            sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
    opS, opU, opS2 = collect_pool(env, cfg["ref_op_n"], np.random.default_rng(cfg["seed"] + 6102),
                                  fs, qc, cfg["op_q_range"], None, collection_mode="on_policy",
                                  behaviour=op_beh, goal_sampler=goal_sampler,
                                  ep_len=cfg["ep_len"], n_par=cfg["n_par"], v0_std=cfg["v0_std"],
                                  wrap_limit=cfg["wrap_limit"])
    ref_op_net = _mlp(cfg["seed"] + 43)
    train_steps(ref_op_net, torch.optim.Adam(ref_op_net.parameters(), lr=cfg["fm_lr"]),
                opS, opU, opS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 303))
    ref_op = grade_ref(ref_op_net, f"on-policy(B2) n={cfg['ref_op_n']}")
    refs = {"teleport": ref_tel, "on_policy": ref_op}
    ceil = ref_tel                      # kept for the excess columns, explicitly the TELEPORT one

    # ================================================================= #
    # Collection: one function per rung. Every rung returns exactly `n_take` transitions,
    # each costing exactly one env.step, so the arms are matched on the only honest currency.
    # ================================================================= #
    def collect(rung, n_take, seed):
        rng = np.random.default_rng(seed)
        if rung == "B0":
            S, U, S2 = collect_pool(env, n_take, rng, fs, qc, cfg["q_range"], cfg["v_explore"])
            return S, U, S2, {"steps_used": n_take, "n_resets": n_take, "n_episodes": n_take,
                              "n_forced_resets": 0}

        # ---- B0m / B0r: THE ORACLE TELEPORTERS -------------------------------------- #
        # The on-policy rungs differ from B0 in THREE ways at once, not one: continuity, the
        # velocity band (~4.8 vs 12.7 rad/s), and the posture spread (episodes reset over `q_jit`,
        # teleport samples `q_range=0.9`). So "embodiment wins" is confounded with "on-policy
        # happens to sample the region the task occupies". These two rungs hand teleport that
        # region for free -- which is exactly the omniscience teleport is being credited with --
        # and ask what is left.
        #
        #   B0m -- teleport with the on-policy pool's MARGINALS (per-dim mean/std of q and qd).
        #          "A teleporter told roughly where to look."
        #   B0r -- teleport to states RESAMPLED FROM THE ON-POLICY VISITED SET itself, so the
        #          state distribution matches exactly rather than in marginals. "A teleporter told
        #          exactly where to look." What remains is only the discontinuity and the i.i.d.
        #          command channel -- i.e. this rung isolates whether trajectories buy anything
        #          BEYOND where they take you.
        #
        # Both are privileged (they read a pool the agent had to behave to obtain), like
        # `directed_loop`'s `oracle`. That is the point: they are the strongest possible teleport.
        if rung in ("B0m", "B0r"):
            n_dof = n
            S = np.empty((n_take, SD), np.float32)
            U = np.empty((n_take, AD), np.float32)
            S2 = np.empty((n_take, SD), np.float32)
            if rung == "B0m":
                qm, qs = opS[:, :n_dof].mean(0), opS[:, :n_dof].std(0)
                vm, vs = opS[:, n_dof:].mean(0), opS[:, n_dof:].std(0)
            else:
                idx = rng.integers(0, len(opS), size=n_take)
            for i in range(n_take):
                if rung == "B0m":
                    q = qm + qs * rng.normal(0, 1, n_dof)
                    qd = vm + vs * rng.normal(0, 1, n_dof)
                else:
                    q, qd = opS[idx[i], :n_dof].astype(np.float64), opS[idx[i], n_dof:].astype(np.float64)
                env.set_state(q, qd)
                u = rng.uniform(-1, 1, AD).astype(np.float32)
                S[i] = env.get_state()
                s2, _ = env.step(u, fs)
                U[i] = u; S2[i] = s2
            return S, U, S2, {"steps_used": n_take, "n_resets": n_take, "n_episodes": n_take,
                              "n_forced_resets": 0}

        # `op_q_range` (not `q_range`) is the EPISODE-START spread: it matches the eval geometry's
        # `q_jit`, so on-policy episodes start where the graded reaches start. `q_range=0.9` is the
        # teleport SAMPLING spread and has no business setting a reset distribution -- using it
        # here would make on-policy reaches start from postures the task never visits.
        common = dict(collection_mode="on_policy", ep_len=cfg["ep_len"], n_par=cfg["n_par"],
                      v0_std=cfg["v0_std"], wrap_limit=cfg["wrap_limit"], return_info=True)

        if rung == "B1":
            beh = OUBehaviour(AD, cfg["n_par"], rng, sigma=cfg["ou_sigma"], theta=cfg["ou_theta"])
            return collect_pool(env, n_take, rng, fs, qc, cfg["op_q_range"], None,
                                behaviour=beh, **common)

        if rung == "B2":
            plan_fn = make_plan_fn(ref_net, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                                   np.random.default_rng(seed + 1))
            beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng,
                                 sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
            return collect_pool(env, n_take, rng, fs, qc, cfg["op_q_range"], None,
                                behaviour=beh, goal_sampler=goal_sampler, **common)

        if rung == "B3":
            # The honest bootstrap. Chunk 0 has no model to plan with, so it is driven by the same
            # OU behaviour as B1; after each chunk the behaviour FM is refit on everything gathered
            # so far and drives the next chunk. This is the model<->data coupling the memo names as
            # the genuine cost of on-policy collection, made explicit rather than assumed away.
            #
            # The behaviour FM is NOT the graded FM: every rung's graded FM is retrained from the
            # same init on that rung's pool with identical steps (below), so training is controlled
            # and only the DATA differs.
            nc = max(1, int(cfg["b3_chunks"]))
            per = int(np.ceil(n_take / nc))
            beh_net = _mlp(cfg["seed"] + 44)
            beh_opt = torch.optim.Adam(beh_net.parameters(), lr=cfg["fm_lr"])
            aS = np.zeros((0, SD), np.float32); aU = np.zeros((0, AD), np.float32)
            aS2 = np.zeros((0, SD), np.float32)
            agg = {"steps_used": 0, "n_resets": 0, "n_episodes": 0, "n_forced_resets": 0}
            for c in range(nc):
                take = min(per, n_take - len(aS))
                if take <= 0:
                    break
                if c == 0:
                    beh = OUBehaviour(AD, cfg["n_par"], rng, sigma=cfg["ou_sigma"],
                                      theta=cfg["ou_theta"])
                    kw = dict(behaviour=beh)
                else:
                    plan_fn = make_plan_fn(beh_net, cfg["collect_k_shoot"],
                                           cfg["collect_cem_iters"],
                                           np.random.default_rng(seed + 100 + c))
                    beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng,
                                         sigma_u=cfg["sigma_u"],
                                         replan_every=cfg["collect_replan_every"])
                    kw = dict(behaviour=beh, goal_sampler=goal_sampler)
                cS, cU, cS2, ci = collect_pool(env, take, rng, fs, qc, cfg["op_q_range"], None,
                                               **kw, **common)
                aS = np.concatenate([aS, cS]); aU = np.concatenate([aU, cU])
                aS2 = np.concatenate([aS2, cS2])
                for k in agg:
                    agg[k] += int(ci.get(k, 0))
                train_steps(beh_net, beh_opt, aS, aU, aS2, cfg["b3_bootstrap_steps"],
                            np.random.default_rng(seed + 500 + c))
                print(f"    [B3 chunk {c+1}/{nc}] pool={len(aS)} "
                      f"behaviour-FM task-err={fm_err_task(beh_net):.4f}", flush=True)
            return aS, aU, aS2, agg

        raise ValueError(rung)

    # ================================================================= #
    # The ladder
    # ================================================================= #
    results = {}
    for rung in cfg["rungs"]:
        results[rung] = []
        for n_take in cfg["n_list"]:
            print(f"\n=== {rung} @ n={n_take} ===", flush=True)
            env.reset_wrap()
            S, U, S2, info = collect(rung, n_take, cfg["seed"] + 1000 + 7 * n_take)
            diag = pool_diagnostics(S, U, nq=n, refS=refS, refU=refU, info=info,
                                    seed=cfg["seed"])
            diag["max_absq_env"] = env.max_absq()
            diag["nonfinite"] = env.nonfinite()

            # identical training protocol for every rung: same init, same steps, same norm
            net = _mlp(cfg["seed"] + 41)
            train_steps(net, torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"]),
                        S, U, S2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
            r = {"rung": rung, "n": int(n_take), "n_actual": int(len(S)),
                 "fm_err_task": fm_err_task(net), "fm_err_broad": fm_err_broad(net)}
            r["fm_err_task_excess"] = r["fm_err_task"] - ceil["fm_err_task"]
            r["fm_err_broad_excess"] = r["fm_err_broad"] - ceil["fm_err_broad"]
            if cfg["do_control"]:
                r["reactive"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                     np.random.default_rng(cfg["seed"] + 7000)), 1)
                r["ballistic_cem"] = rollout(
                    make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                 np.random.default_rng(cfg["seed"] + 7001)), H)
            r["diag"] = diag
            results[rung].append(r)
            print(f"[{rung} n={n_take}] task={r['fm_err_task']:.4f} "
                  f"(excess {r['fm_err_task_excess']:+.4f})  broad={r['fm_err_broad']:.4f} "
                  f"(excess {r['fm_err_broad_excess']:+.4f})"
                  + (f"  reactive={r['reactive']:.4f} ballistic={r['ballistic_cem']:.4f}"
                     if cfg["do_control"] else ""), flush=True)
            print(f"    [diag] max|corr(u,s)|={diag.get('max_abs_corr', float('nan')):.4f} "
                  f"speed_mean={diag.get('speed_mean', float('nan')):.2f} "
                  f"p95={diag.get('speed_p95', float('nan')):.2f} "
                  f"cover(ref->pool)={diag.get('ref_to_pool_mean', float('nan')):.3f} "
                  f"reach(pool->ref)={diag.get('pool_to_ref_mean', float('nan')):.3f} "
                  f"eps={diag.get('n_episodes', 0)} forced_resets={diag.get('n_forced_resets', 0)}",
                  flush=True)

    # ================================================================= #
    # summary
    # ================================================================= #
    print("\n" + "=" * 96, flush=True)
    print("=== the price of the flag: identical steps, identical training, only the data differs ===",
          flush=True)
    hdr = (f"{'rung':>5s} {'n':>6s} {'task_err':>9s} {'broad_err':>10s} {'|corr|':>7s} "
           f"{'speed':>7s} {'cover':>7s} {'reactive':>9s} {'ballistic':>10s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for nm, rr in (("refTP", ref_tel), ("refOP", ref_op)):
        print(f"{nm:>5s} {'-':>6s} {rr['fm_err_task']:>9.4f} {rr['fm_err_broad']:>10.4f} "
              f"{'-':>7s} {'-':>7s} {'-':>7s} "
              + (f"{rr['reactive']:>9.4f} {rr['ballistic_cem']:>10.4f}"
                 if cfg["do_control"] else f"{'-':>9s} {'-':>10s}"), flush=True)
    for rung in cfg["rungs"]:
        for r in results[rung]:
            d = r["diag"]
            print(f"{rung:>5s} {r['n']:>6d} {r['fm_err_task']:>9.4f} {r['fm_err_broad']:>10.4f} "
                  f"{d.get('max_abs_corr', float('nan')):>7.4f} "
                  f"{d.get('speed_mean', float('nan')):>7.2f} "
                  f"{d.get('ref_to_pool_mean', float('nan')):>7.3f} "
                  + (f"{r['reactive']:>9.4f} {r['ballistic_cem']:>10.4f}"
                     if cfg["do_control"] else f"{'-':>9s} {'-':>10s}"), flush=True)

    # The three predictions, checked explicitly at the largest budget.
    nmax = cfg["n_list"][-1]
    last = {rg: [r for r in results[rg] if r["n"] == nmax][0] for rg in cfg["rungs"]
            if any(r["n"] == nmax for r in results[rg])}
    verdict = {}
    if "B0" in last:
        b0 = last["B0"]
        for rg, r in last.items():
            if rg == "B0":
                continue
            verdict[rg] = {
                "task_ratio_vs_B0": r["fm_err_task"] / max(b0["fm_err_task"], 1e-9),
                "broad_ratio_vs_B0": r["fm_err_broad"] / max(b0["fm_err_broad"], 1e-9),
                "corr_vs_B0": (r["diag"].get("max_abs_corr", float("nan")),
                               b0["diag"].get("max_abs_corr", float("nan"))),
            }
        print("\n[P1] on-policy / teleport error ratio at n=%d (>1 = worse than teleport):" % nmax,
              flush=True)
        for rg, v in verdict.items():
            print(f"    {rg}: task {v['task_ratio_vs_B0']:.2f}x   broad {v['broad_ratio_vs_B0']:.2f}x",
                  flush=True)
        # B0's own value is the FINITE-SAMPLE FLOOR at this n, not zero: max over 18 (state,command)
        # pairs of a sample correlation has an O(1/sqrt(n)) floor, so read each rung against B0 at
        # the SAME n. Cut #2's 0.003 was measured on a far larger pool.
        print("\n[P2] command-state correlation, read against B0 at the same n "
              "(Cut #2: 0.003 on a large i.i.d. pool):", flush=True)
        for rg in cfg["rungs"]:
            if rg in last:
                v = last[rg]["diag"].get("max_abs_corr", float("nan"))
                b = last["B0"]["diag"].get("max_abs_corr", float("nan"))
                print(f"    {rg}: max|corr(u,s)| = {v:.4f}"
                      + ("" if rg == "B0" else f"   ({v / b:.1f}x the teleport floor)"), flush=True)

    # P3 is a SLOPE, not an excess ratio. `ballistic_readapt`/`ballistic_transmission` measure
    # transmission as d(control)/d(FM-error) along an FM-quality axis; here the axis is each rung's
    # own sample-size sweep, which is exactly such an axis (more data -> lower task error). Reading
    # it as "excess over a ceiling" was what broke in the smoke: with the teleport reference worse
    # on the task probe than every on-policy rung, the reactive excess went negative and the ratio
    # became uninterpretable. A within-rung slope needs no ceiling at all.
    if cfg["do_control"] and len(cfg["n_list"]) >= 2:
        print("\n[P3] transmission slope d(control)/d(task FM-error) within each rung "
              "(Cut 4b: reactive 1.0x, ballistic 3.0x; 4c-arm 5.24x):", flush=True)
        for rg in cfg["rungs"]:
            rs = sorted(results[rg], key=lambda r: r["fm_err_task"])
            if len(rs) < 2:
                continue
            x = np.array([r["fm_err_task"] for r in rs])
            sl = {}
            for c in ("reactive", "ballistic_cem"):
                y = np.array([r[c] for r in rs])
                sl[c] = float(np.polyfit(x, y, 1)[0]) if x.std() > 1e-9 else float("nan")
            ratio = (sl["ballistic_cem"] / sl["reactive"]
                     if abs(sl["reactive"]) > 1e-9 else float("nan"))
            verdict.setdefault(rg, {})["slopes"] = sl
            verdict[rg]["ballistic_over_reactive_slope"] = ratio
            print(f"    {rg}: reactive {sl['reactive']:+.3f}  ballistic {sl['ballistic_cem']:+.3f}"
                  f"   -> {ratio:.2f}x", flush=True)

    out = {"config": cfg, "references": refs, "ceiling": ceil, "results": results,
           "verdict": verdict}
    outdir = os.path.join(DATA_DIR, "on_policy_coverage", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def coverage_probe(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    rungs: str = "B0,B0m,B0r,B1,B2,B3",
    n_list: str = "150,400,1000,2500,6000",   # the sample-efficiency curve per rung; also
                                             # the FM-quality axis P3's slope is read on

    do_control: bool = True,
    # --- plant: ballistic/arm's Cut-4c-arm design point, byte-identical ---
    b: float = 6.0,                 # the operating world (4c-arm's post-drift b1)
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,         # teleport-only; asserted unused on the on-policy path
    # --- on-policy collection knobs (the new axis) ---
    ep_len: int = 14,               # one episode = one reach, matched to plan_h
    op_q_range: float = 0.25,       # EPISODE-START spread; = q_jit, so on-policy episodes start
                                    # where the graded reaches start. NOT q_range (=0.9), which is
                                    # the teleport SAMPLING spread and has no business setting a
                                    # reset distribution.
    ref_op_n: int = 14000,          # the on-policy reference pool (bounds the task probe)
    n_par: int = 16,
    sigma_u: float = 0.15,          # motor noise: trades on-task-ness vs command identifiability
    ou_sigma: float = 0.7,
    ou_theta: float = 0.15,
    wrap_limit: float = 3.0,        # `ArmEnv.wrapped()`'s default legible range
    collect_k_shoot: int = 512,     # behaviour planner: cheaper than the grader's, deliberately
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14, # = ep_len -> ballistic behaviour (this arc's regime)
    b3_chunks: int = 6,
    b3_bootstrap_steps: int = 1500,
    # --- FM (identical for every rung) ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    probe_n: int = 1500,
    # --- control eval: k_shoot/cem_iters sized to the 42-dim action sequence (arm_substrate P3) ---
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

    rg = [r.strip() for r in rungs.split(",") if r.strip()]
    nl = [int(x) for x in n_list.split(",") if x.strip()]
    if quick:
        nl = [300, 900]
        pool_n = 3000; fm_steps = 1200; probe_n = 600
        n_eval = 12; k_shoot = 256; cem_iters = 4
        collect_k_shoot = 128; collect_cem_iters = 3
        fm_hidden = 128; fm_layers = 2
        b3_chunks = 3; b3_bootstrap_steps = 300
        n_par = 8; ref_op_n = 3000
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, rungs=rg, n_list=nl, do_control=do_control, b=b,
        n_links=n_links,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        ep_len=ep_len, op_q_range=op_q_range, ref_op_n=ref_op_n,
        n_par=n_par, sigma_u=sigma_u, ou_sigma=ou_sigma, ou_theta=ou_theta,
        wrap_limit=wrap_limit, collect_k_shoot=collect_k_shoot,
        collect_cem_iters=collect_cem_iters, collect_replan_every=collect_replan_every,
        b3_chunks=b3_chunks, b3_bootstrap_steps=b3_bootstrap_steps,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, pool_n=pool_n, probe_n=probe_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters,
        cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, reach_tries=reach_tries,
    )
    out = run_coverage_probe.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "coverage_probe_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
