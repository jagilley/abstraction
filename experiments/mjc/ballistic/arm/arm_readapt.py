"""Cut 4c-arm: the END-TO-END re-adaptation loop on the SECOND task family — reward-free online
FM re-adaptation after a force-field drift restores BALLISTIC competence; reactive control never
needed it. Plus the AFTEREFFECT, the readout the pusher structurally could not produce.

Program: `ideas/two_timescale_value_loop.md`. Parents:
  * [../README.md](../README.md) Cut 4c (`../ballistic_readapt.py`) — the same experiment on the
    pusher (damping drift d0=6 -> d1=2): ballistic recovers 0.295 -> 0.108 (the matched-FM
    ceiling) while reactive is flat and already competent, a 4.3x recovery-gain dissociation.
    Its standing caveat is **single-family**. This is the second family.
  * [../../arm_substrate/README.md](../../arm_substrate/README.md) — the substrate, characterized.
    P5 already ran the *exogenous-axis* half here (4b-arm): ballistic damage +0.226 vs reactive
    +0.038 across a controlled curl-staleness axis, a 6.0x dissociation, single seed. What is
    missing is the ENDOGENOUS version — FM quality produced by the learning layer re-adapting
    online — which is what this file adds, multi-seed.

    THE DRIFT IS THE SHADMEHR PROTOCOL, RUN FORWARD. Pretrain the FM in a field-free world
    (`b0=0`, the naive arm), then switch the world on to `b1` (the curl force field) and let the
    learning layer re-adapt from REWARD-FREE `b1` transitions. Snapshot the FM along that
    trajectory and grade each snapshot under `reactive` / `ballistic_cem` / `ballistic_bc`.
    This is exactly the human force-field adaptation experiment, with the internal model made
    observable.

    WHY THE CURL AXIS AND NOT PAYLOAD (arm_substrate P2/P5). The curl field perturbs the hand
    PERPENDICULAR to its motion, so a wrong internal model produces a lateral deviation that
    ACCUMULATES over an open-loop window and is corrected step-by-step by feedback — precisely
    the ballistic/reactive asymmetry. It is smooth, global, one-signed and open-loop
    COMPENSABLE (Cut 4b's precondition; a localized force jet is incompensable and saturates
    ballistic control at "fail" for every FM quality). Payload staleness washes out on the task
    distribution (P2) and is not usable here.

    FOUR PREDICTIONS:
      1. BALLISTIC control RECOVERS as the FM re-adapts, reaching the matched-FM ceiling.
      2. REACTIVE is ~FLAT and already competent — it never needed the re-adaptation, so the
         behavioral value of re-adaptation is BALLISTIC-SPECIFIC (pusher: 4.3x).
      3. Task-distribution FM error TRACKS the ballistic recovery (Cut 3's afferent teacher is a
         valid proxy for the behavioral payoff, and only because control is ballistic).
      4. THE AFTEREFFECT, new here: grade the re-adapted FM back in the FIELD-FREE world. A
         model that has learned to pre-compensate a field that is no longer there must
         MIS-reach in the mirror direction — and that error should also be ballistic-specific,
         because feedback corrects it away within a step. This is the canonical behavioral
         signature that adaptation is a MODEL update rather than impedance/co-contraction, and
         the pusher could never produce it (no perpendicular, motion-dependent axis). Reported
         as a directed lateral deviation, signed against the field's curl direction, not just a
         distance — an unsigned miss is consistent with "the model got worse", a mirror-signed
         one is not.

    METHOD RULES INHERITED (all three cost days elsewhere; see arm_substrate §Three
    transferable methodological findings):
      * grade FM error on the TASK distribution (matched-FM reach transitions), not the broad
        collection pool — a broad probe turned this effect into a null on the arm (0.81x -> 24x);
      * size the open-loop CEM to the ACTION-SEQUENCE dimension (H x n_act = 42 here): a
        pusher-tuned `k_shoot=256, cem_iters=4` silently flattens the transmission slope;
      * keep H below the composition horizon (n=3: 20-23 steps; H=14).

Run:
    cd experiments/
    modal run mjc/ballistic/arm/arm_readapt.py::arm_readapt --quick          # smoke
    for s in 0 1 2; do
      modal run --detach mjc/ballistic/arm/arm_readapt.py::arm_readapt --tag armre_s$s --seed $s
    done
    python3 mjc/ballistic/arm/arm_readapt_figure.py --tags armre_s0 armre_s1 armre_s2
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_arm_readapt(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    b0, b1 = cfg["b0"], cfg["b1"]
    B = cfg["n_eval"]
    print(f"[setup] device={device} n_links={n} H={H} curl drift b0={b0} -> b1={b1} "
          f"milestones={cfg['milestones']} controllers={cfg['controllers']}", flush=True)

    def make_env(b):
        return ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                           link_masses=cfg["link_masses"][:n],
                           joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                           curl_field={"b": float(b)}))

    env0 = make_env(b0)      # pre-drift world (FM pretraining) — the naive, field-free arm
    env1 = make_env(b1)      # post-drift world — re-adaptation AND control eval = the TRUE env

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

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

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

    def pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs,
                            qc, cfg["q_range"], cfg["v_explore"])

    # Normalization fixed once from the OPERATING dynamics (b1), shared by every FM on the
    # ladder — otherwise each snapshot would be scored in its own units.
    nS, nU, nS2 = pool(env1, cfg["pool_n"], cfg["seed"] + 11)
    norm = make_norm(nS, nU, nS2)

    # ------------------------------------------------------------------ CEM (tip-space cost)
    Kc, ne_el = cfg["k_shoot"], cfg["cem_elite"]

    def mpc_plan(net, states, goals, rng, horizon=None):
        Hh = horizon or H
        Bn = states.shape[0]
        mu = np.zeros((Bn, Hh, AD), np.float32)
        sig = np.full((Bn, Hh, AD), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device, dtype=torch.float32).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device, dtype=torch.float32).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, Hh, AD)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone()
                seqs_t = torch.tensor(seqs.reshape(Bn * Kc, Hh, AD), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(Hh):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                    cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    # ------------------------------------------------------------------ reaches
    def eval_geometry(seed):
        """Goal sampled in JOINT space (reachable without IK), rejected until the induced TIP
        displacement lands in [reach_lo, reach_hi] — for a redundant arm a random joint delta
        often lands in the null space and barely moves the tip, silently filling the eval set
        with trivial reaches (arm_substrate `eval_geometry`)."""
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
        return starts, goals, t0.astype(np.float32)

    ev_starts, ev_goals, ev_t0 = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every, cenv, record=None):
        """Executes in `cenv` — the TRUE world. Returns (median tip-goal distance, signed
        lateral deviation). The lateral sign is taken against the curl direction: the field is
        F = b·[[0,-1],[1,0]]·v, i.e. a +90° rotation of the velocity, so `perp` below is the
        direction the field pushes the hand. A model that over-compensates a field that is no
        longer present deviates the OTHER way — that sign flip is the aftereffect."""
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
        dist = float(np.median(np.linalg.norm(tips - ev_goals, axis=1)))
        reach_dir = ev_goals - ev_t0
        reach_dir /= np.maximum(np.linalg.norm(reach_dir, axis=1, keepdims=True), 1e-9)
        perp = np.stack([-reach_dir[:, 1], reach_dir[:, 0]], 1)          # +90° of the reach
        lateral = float(np.median(((tips - ev_goals) * perp).sum(1)))
        nf = cenv.nonfinite()
        if nf:
            print(f"[warn] {nf} non-finite states during rollout — see arm_substrate "
                  f"methodological finding #3 (silent NaN past the integration stability limit)",
                  flush=True)
        return dist, lateral

    # ------------------------------------------------------------------ BC motor program
    def build_policy():
        h, L = cfg["pol_hidden"], cfg["pol_layers"]
        lyr = [nn.Linear(SD + 2, h), nn.SiLU()]                # state + Cartesian goal
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, AD * H), nn.Tanh()]
        return nn.Sequential(*lyr).to(device)

    def ballistic_bc(net_fm, cenv):
        """The genuinely feedforward controller: a behavior-cloned motor program mapping
        (state, goal) -> the WHOLE action sequence, with no online optimization. On the pusher
        this transmitted FM quality the MOST (slope 3.3x vs ballistic-CEM's 3.0x) because it
        cannot re-optimize per episode and so bakes the FM's staleness in."""
        rng = np.random.default_rng(cfg["seed"] + 55); nt = cfg["bc_tuples"]
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (nt, n))
        t0 = fk(q0, Ls)
        gp = np.empty((nt, 2), np.float64)
        for i in range(nt):
            d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
            gp[i] = fk(q0[i] + cfg["reach_amp"] * d, Ls)
        keep = np.flatnonzero((np.linalg.norm(gp - t0, axis=1) >= cfg["reach_lo"])
                              & (np.linalg.norm(gp - t0, axis=1) <= cfg["reach_hi"]))
        # The reach-band rejection keeps only ~25% of joint-space candidates (most land in the
        # null space and barely move the tip — `eval_geometry`'s problem again), so `bc_tuples`
        # must be sized to the KEPT count, not the drawn count. The smoke kept 78/300 and the
        # BC arm under-fit visibly.
        q0, gp = q0[keep], gp[keep]; nt = len(keep)
        print(f"[bc] reach-band filter kept {nt}/{cfg['bc_tuples']} candidates "
              f"({100.0 * nt / max(cfg['bc_tuples'], 1):.0f}%)", flush=True)
        s0 = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (nt, n))], 1).astype(np.float32)
        gp = gp.astype(np.float32)
        plans = np.empty((nt, H, AD), np.float32)
        for i in range(0, nt, 128):
            j = min(i + 128, nt); plans[i:j] = mpc_plan(net_fm, s0[i:j], gp[i:j], rng)
        X = np.concatenate([s0, gp], 1).astype(np.float32)
        pn = {"mu": torch.tensor(X.mean(0), device=device),
              "sd": torch.tensor(X.std(0) + 1e-6, device=device)}
        Xt = (torch.tensor(X, device=device) - pn["mu"]) / pn["sd"]
        Yt = torch.tensor(plans.reshape(nt, AD * H), device=device)
        pol = build_policy(); optp = torch.optim.Adam(pol.parameters(), lr=cfg["pol_lr"])
        lossf = nn.MSELoss(); brng = np.random.default_rng(cfg["seed"] + 56)
        pol.train(); bs = min(512, nt)
        for _ in range(cfg["pol_steps"]):
            idx = torch.tensor(brng.integers(0, nt, size=bs), device=device)
            optp.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); optp.step()
        pol.eval()
        print(f"[bc] cloned {nt} (state, goal) -> {AD*H}-dim motor programs", flush=True)

        def plan_fn(states, goals):
            with torch.no_grad():
                x = (torch.tensor(np.concatenate([states, goals], 1).astype(np.float32),
                                  device=device) - pn["mu"]) / pn["sd"]
                return pol(x).cpu().numpy().astype(np.float32).reshape(len(states), H, AD)
        return rollout(plan_fn, H, cenv)

    # ------------------------------------------------------------------ the task probe set
    # THE probe correction (arm_substrate finding #1): FM error is graded on the transitions a
    # MATCHED-FM reach actually visits, not on the broad collection pool. Built once from a
    # reference FM trained in the true (b1) world, then reused for every snapshot so the x-axis
    # is controlled.
    ref_net = _mlp(cfg["seed"] + 40)
    rS, rU, rS2 = pool(env1, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    rec = []
    rollout(lambda s, g: mpc_plan(ref_net, s, g, np.random.default_rng(cfg["seed"] + 6000)),
            1, env1, record=rec)
    rollout(lambda s, g: mpc_plan(ref_net, s, g, np.random.default_rng(cfg["seed"] + 6001)),
            H, env1, record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    pT = (np.array([r[2] for r in rec], np.float32) - pS).astype(np.float32)
    bS, bU, bS2 = pool(env1, cfg["probe_n"], cfg["seed"] + 12)      # broad probe: the contrast
    bT = (bS2 - bS).astype(np.float32)
    print(f"[probes] task probe {len(pS)} transitions (matched-FM reaches); broad probe {len(bS)}",
          flush=True)

    def fm_err(net):
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    def fm_err_broad(net):
        return float(np.linalg.norm(fm_delta(net, bS, bU) - bT, axis=1).mean())

    def grade(net, want_bc, cenv=None, prefix=""):
        cenv = cenv if cenv is not None else env1
        r = {}
        if not prefix:
            r["fm_err"] = fm_err(net); r["fm_err_broad"] = fm_err_broad(net)
        if "reactive" in cfg["controllers"]:
            rng = np.random.default_rng(cfg["seed"] + 7000)
            d, lat = rollout(lambda s, g: mpc_plan(net, s, g, rng), 1, cenv)
            r[prefix + "reactive"] = d; r[prefix + "reactive_lat"] = lat
        if "ballistic_cem" in cfg["controllers"]:
            rng = np.random.default_rng(cfg["seed"] + 7001)
            d, lat = rollout(lambda s, g: mpc_plan(net, s, g, rng), H, cenv)
            r[prefix + "ballistic_cem"] = d; r[prefix + "ballistic_cem_lat"] = lat
        if want_bc and "ballistic_bc" in cfg["controllers"]:
            d, lat = ballistic_bc(net, cenv)
            r[prefix + "ballistic_bc"] = d; r[prefix + "ballistic_bc_lat"] = lat
        return r

    # ================================================================= #
    # 0) the CEILING FIRST — a fresh FM trained on b1, i.e. full re-adaptation.
    #
    # Trained BEFORE the ladder rather than after, so `fm_err_excess = fm_err − ceiling` is
    # available at every milestone. This is arm_substrate P5's own prescription for the one
    # thing that went wrong there: `fm_err(task)` came out NON-MONOTONIC (0.516, 0.494, 0.479,
    # 0.577, 0.860) because reaches run to ~15 rad/s while collection samples at `v_explore=8`,
    # so the task probe is dominated by a high-velocity tail where EVERY FM extrapolates badly
    # and the axis-specific signal is second-order. P5: *"prefer the damage reading; fix by
    # widening v_explore or scoring EXCESS error (stale − matched) rather than raw."* Widening
    # `v_explore` would change the substrate away from P5's; scoring excess does not, so that
    # is the fix taken here. The smoke reproduced the problem exactly (raw fm_err 1.04 -> 0.58
    # -> 0.61 with the ceiling at 0.66), which is why it is worth fixing before the real runs.
    # ================================================================= #
    fm_ceil = _mlp(cfg["seed"] + 42)
    Sc, Uc, S2c = pool(env1, cfg["pool_n"], cfg["seed"] + 20)
    train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
    fm_err_floor = fm_err(fm_ceil)
    print(f"[ceiling-fm] fresh b1 FM task-probe error = {fm_err_floor:.4f} "
          f"(the irreducible floor; excess = raw − this)", flush=True)

    # ================================================================= #
    # 1) pretrain the pre-drift FM in the FIELD-FREE world (the naive arm)
    # ================================================================= #
    S0, U0, S20 = pool(env0, cfg["pool_n"], cfg["seed"] + 10)
    fm = _mlp(cfg["seed"] + 41); opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
    train_steps(fm, opt, S0, U0, S20, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    print(f"[pretrain] b0={b0} FM: fm_err@b1(stale, task)={fm_err(fm):.4f} "
          f"broad={fm_err_broad(fm):.4f}", flush=True)

    # ================================================================= #
    # 2) online reward-free re-adaptation at b1; snapshot & grade per milestone
    # ================================================================= #
    rng_buf = np.random.default_rng(cfg["seed"] + 500)
    bufS = np.zeros((0, SD), np.float32); bufU = np.zeros((0, AD), np.float32)
    bufS2 = np.zeros((0, SD), np.float32)
    ladder = []
    for k, m in enumerate(cfg["milestones"]):
        need = m - len(bufS)
        if need > 0:
            aS, aU, aS2 = collect_pool(env1, need, rng_buf, fs, qc, cfg["q_range"],
                                       cfg["v_explore"])
            bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU])
            bufS2 = np.concatenate([bufS2, aS2])
        if m > 0:
            train_steps(fm, opt, bufS, bufU, bufS2, cfg["finetune_steps"],
                        np.random.default_rng(cfg["seed"] + 600 + k))
        want_bc = (m == cfg["milestones"][0]) or (m == cfg["milestones"][-1])
        r = grade(copy.deepcopy(fm), want_bc)
        r["transitions"] = int(m)
        # the AFTEREFFECT: same FM, graded back in the FIELD-FREE world it was pretrained in
        if cfg["do_aftereffect"] and want_bc:
            r.update(grade(copy.deepcopy(fm), False, cenv=env0, prefix="ae_"))
        r["fm_err_excess"] = r["fm_err"] - fm_err_floor
        ladder.append(r)
        msg = (f"[readapt m={m:6d}] fm_err={r['fm_err']:.4f} "
               f"excess={r['fm_err_excess']:+.4f}")
        for c in cfg["controllers"]:
            if c in r:
                msg += f"  {c}={r[c]:.4f}"
        print(msg, flush=True)

    # ---- the ceiling's own control numbers (the FM itself was trained in step 0) ----
    ceil = grade(fm_ceil, want_bc=True)
    ceil["transitions"] = -1; ceil["fm_err_excess"] = 0.0
    print(f"[ceiling] fresh b1 FM: fm_err={ceil['fm_err']:.4f}  "
          + "  ".join(f"{c}={ceil.get(c, float('nan')):.4f}" for c in cfg["controllers"]),
          flush=True)

    # ---- the headline: recovery gain per controller (stale − recovered) ----
    recovery = {}
    for c in cfg["controllers"]:
        st, rc = ladder[0].get(c), ladder[-1].get(c)
        if st is not None and rc is not None:
            recovery[c] = {"stale": st, "recovered": rc, "gain": st - rc}
    print("\n[recovery] control gain from reward-free re-adaptation (stale − recovered; larger "
          "= more behaviorally load-bearing):", flush=True)
    for c, v in recovery.items():
        print(f"  {c:14s} stale={v['stale']:.4f} -> recovered={v['recovered']:.4f}  "
              f"gain={v['gain']:+.4f}", flush=True)
    if "reactive" in recovery and recovery["reactive"]["gain"] > 1e-9:
        for c in ("ballistic_cem", "ballistic_bc"):
            if c in recovery:
                print(f"  -> {c} recovery gain is "
                      f"{recovery[c]['gain'] / recovery['reactive']['gain']:.2f}x reactive's "
                      f"(pusher 4c: 4.3x / 4.4x)", flush=True)

    if cfg["do_aftereffect"]:
        print("\n[aftereffect] the re-adapted FM graded back in the FIELD-FREE world "
              "(signed lateral deviation; a sign FLIP vs the stale model is the model-update "
              "signature):", flush=True)
        for c in cfg["controllers"]:
            k = "ae_" + c
            s_, r_ = ladder[0].get(k), ladder[-1].get(k)
            sl, rl = ladder[0].get(k + "_lat"), ladder[-1].get(k + "_lat")
            if s_ is not None and r_ is not None:
                print(f"  {c:14s} naive-FM dist={s_:.4f} lat={sl:+.4f}  ->  "
                      f"adapted-FM dist={r_:.4f} lat={rl:+.4f}", flush=True)

    out = {"config": cfg, "ladder": ladder, "ceiling": ceil, "recovery": recovery}
    outdir = os.path.join(DATA_DIR, "arm_readapt", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def arm_readapt(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    controllers: str = "reactive,ballistic_cem,ballistic_bc",
    # --- the drift: Shadmehr force-field, naive -> field ---
    b0: float = 0.0,
    b1: float = 6.0,
    milestones: str = "0,400,1000,2500,6000,14000",
    do_aftereffect: bool = True,
    # --- arm geometry: arm_substrate's n=3 curl design point (P5) ---
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    # joint_damping / gear / vel_pen / v0_std / cem_elite / reach_tries / probe_n below are
    # arm_probe.py's defaults, i.e. exactly what P5 ran. Held identical on purpose: P5 is the
    # exogenous-axis half of this same claim and the only thing that should differ between them
    # is where the FM-quality axis comes from.
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    finetune_steps: int = 1500,
    pool_n: int = 14000,
    probe_n: int = 1500,
    # --- control eval. k_shoot/cem_iters sized to the 42-dim action sequence (P3): a
    #     pusher-tuned 256/4 flattens the transmission effect into a null.
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
    # --- BC motor program ---
    bc_tuples: int = 6000,   # ~25% survive the reach-band filter -> ~1500 kept, matching
                             # the pusher 4c BC arm's 1500 tuples
    pol_hidden: int = 256,
    pol_layers: int = 3,
    pol_lr: float = 1e-3,
    pol_steps: int = 4000,
):
    import os

    ms = [int(x) for x in milestones.split(",") if x.strip()]
    if quick:
        milestones_q = [0, 400, 2000]
        ms = milestones_q
        pool_n = 3000; fm_steps = 1500; finetune_steps = 400; probe_n = 800
        n_eval = 12; k_shoot = 256; cem_iters = 4; bc_tuples = 300; pol_steps = 600
        fm_hidden = 128; fm_layers = 2
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, controllers=[c for c in controllers.split(",") if c],
        b0=b0, b1=b1, milestones=ms, do_aftereffect=do_aftereffect,
        n_links=n_links,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, finetune_steps=finetune_steps, pool_n=pool_n, probe_n=probe_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters,
        cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, reach_tries=reach_tries,
        bc_tuples=bc_tuples, pol_hidden=pol_hidden, pol_layers=pol_layers,
        pol_lr=pol_lr, pol_steps=pol_steps,
    )
    out = run_arm_readapt.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "arm_readapt_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
