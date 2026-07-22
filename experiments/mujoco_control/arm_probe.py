"""Arm substrate characterization: does the planar arm have the properties we need?

Before porting any cut onto `arm_env.py` we verify the substrate itself, because the
ballistic arc's history is that BOTH of its smoke tests looked like nulls until a regime
constraint was fixed (cut 4b: the FM-quality axis must be smooth not localized; the reach
must be feasible). This script front-loads exactly those checks. Six probes, each mapped
to a property the arc depends on:

  P0  SANITY        -- analytic FK == MuJoCo's own site_xpos, and the payload measurably
                       changes the dynamics. If FK disagrees, the CEM cost is scoring a
                       different arm than the simulator executes and everything downstream
                       is silently wrong.
  P1  CAPACITY      -- an arity-2 FM capacity sweep. THE headline property. On the pusher
                       this curve is FLAT at R²=0.999 from 232 parameters (cut #2), which
                       is why capacity competition had to be manufactured with a force
                       field. If the arm shows a real frontier, nonlinearity is intrinsic.
                       Arity-1 f(s) is carried as the command-blind anchor.
  P2  LOCALITY      -- a payload-mass staleness axis. (a) does FM error on the TRUE
                       dynamics grow monotonically with |payload_train - payload_test|
                       (a controlled quality axis, cut 4b's requirement)? (b) is that
                       error CONFIGURATION-DEPENDENT -- binned by tip radius r(q)=‖FK(q)‖,
                       does staleness hurt more when the arm is EXTENDED? The payload's
                       contribution to the inertia scales with r², so this is a
                       quantitative physical prediction, and it is the intrinsic version
                       of the local-but-compensable drift cut #5 had to synthesize with
                       `rot_regions`.
  P3  FEASIBILITY   -- with a MATCHED FM, does the reach actually succeed, both reactively
                       and ballistically, against the do-nothing and random-action floors?
                       Cut 4b's precondition #2. If ballistic saturates at "fail" nothing
                       downstream can transmit.
  P4  COMPOSITION   -- how many steps does a one-step FM compose before its rolled-out tip
                       trajectory diverges? The pusher's answer was ~6-8 steps
                       (`dynamics_shift.replan_ablation`, and the a2a REACHING_LOOKAHEAD
                       limit reappearing on physics). Under nonlinear coupling this may be
                       SHORTER, which would bound the usable ballistic window -- a number
                       we need before designing the port, not after.
  P5  TRANSMISSION  -- the mini cut-4b: reactive vs ballistic_cem control distance over the
                       P2 quality axis, and the transmission slopes. The pusher gave
                       reactive +0.35 / ballistic_cem +1.07 (3.0x). Does a second, NONLINEAR
                       task family reproduce the dissociation? This is the check that kills
                       the "single-family" caveat.

Reuses ballistic_transmission.py's FM/CEM/rollout stack (self-contained duplication, the
established pattern in this directory). The one structural difference: the goal is a
CARTESIAN tip position, so the CEM cost applies analytic forward kinematics to the FM's
predicted joint state -- the KINEMATIC map is known and fixed, the DYNAMICS is what is
learned and what drifts.

Run:
    cd experiments/
    modal run mujoco_control/arm_probe.py::arm_probe --quick          # smoke (~2 min)
    modal run --detach mujoco_control/arm_probe.py::arm_probe --tag char_v1
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_arm_probe(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.arm_env import ArmEnv, collect_pool, fk

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    H = cfg["plan_H"]
    n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    AXIS = cfg["axis"]
    a_test, a_trains = cfg["a_test"], cfg["a_trains"]
    out = {"config": cfg}
    print(f"[setup] device={device} n_links={n} state_dim={SD} frame_skip={fs} H={H} "
          f"axis={AXIS} a_test={a_test} a_trains={a_trains}", flush=True)

    def make_env(av):
        """`av` is the value of the FM-QUALITY AXIS. Two axes are available and they test
        different physics:
          * `payload` -- a tip point mass. Changes M(q) NON-UNIFORMLY over configuration
            space (the payload's inertia contribution scales with tip radius r(q)^2), so it
            is the INERTIAL-LOCALITY axis: staleness should hurt more when extended.
          * `curl`    -- a velocity-dependent curl force field at the end effector, i.e. the
            Shadmehr & Mussa-Ivaldi force-field adaptation protocol. Smooth, global,
            one-signed and open-loop COMPENSABLE, and (unlike a payload) it perturbs the
            hand PERPENDICULAR to its motion, so a wrong internal model produces a
            systematic lateral deviation that ACCUMULATES over an open-loop window and is
            corrected step-by-step by feedback. That is precisely the asymmetry the
            ballistic/reactive transmission claim is about, which makes curl the a-priori
            better transmission axis -- and it is the axis the AFTEREFFECT readout is
            defined on.
        """
        d = dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                 link_masses=cfg["link_masses"][:n],
                 joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                 payload_mass=cfg["payload_base"])
        if AXIS == "payload":
            d["payload_mass"] = av
        else:
            d["curl_field"] = {"b": av}
        return ArmEnv(d)

    eval_env = make_env(a_test)                # the TRUE dynamics the controller runs in
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    # ================================================================= #
    # P0 -- sanity: analytic FK vs MuJoCo, and does the axis knob bite?
    # ================================================================= #
    rng0 = np.random.default_rng(cfg["seed"] + 1)
    fk_err = 0.0
    for _ in range(200):
        q = qc + rng0.uniform(-1.2, 1.2, n)
        eval_env.set_state(q, np.zeros(n))
        fk_err = max(fk_err, float(np.abs(eval_env.tip_pos() - fk(q, Ls)).max()))
    stale_env = make_env(max(a_trains, key=lambda v: abs(v - a_test)))
    d_ref, d_alt = [], []
    rng0b = np.random.default_rng(cfg["seed"] + 2)
    for _ in range(400):
        q = qc + rng0b.uniform(-1.0, 1.0, n)
        qd = rng0b.normal(0, cfg["v_explore"], n)
        u = rng0b.uniform(-1, 1, n)
        eval_env.set_state(q, qd); s = eval_env.get_state(); s2, _ = eval_env.step(u, fs)
        d_ref.append(s2 - s)
        stale_env.set_state(q, qd); s = stale_env.get_state(); s2, _ = stale_env.step(u, fs)
        d_alt.append(s2 - s)
    d_ref, d_alt = np.array(d_ref), np.array(d_alt)
    axis_bite = float(np.linalg.norm(d_alt - d_ref, axis=1).mean()
                      / (np.linalg.norm(d_ref, axis=1).mean() + 1e-9))
    out["P0"] = {"fk_max_abs_err": fk_err, "axis_bite_rel": axis_bite}
    print(f"\n[P0 sanity] analytic-FK vs MuJoCo max|err| = {fk_err:.3e}   "
          f"(must be ~1e-12; else the CEM cost scores a different arm)", flush=True)
    print(f"[P0 sanity] the stalest {AXIS} setting changes Δs by {100*axis_bite:.1f}% of "
          f"‖Δs‖ -- the drift knob has teeth", flush=True)

    # ================================================================= #
    # shared FM machinery (ballistic_transmission.py idiom)
    # ================================================================= #
    def _mlp(seed, hidden, layers, in_dim, out_dim):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(hidden, out_dim)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def make_norm(S, U, S2, arity=2):
        X = (np.concatenate([S, U], 1) if arity == 2 else S).astype(np.float32)
        Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)

    def train_fm(net, norm, S, U, S2, steps, brng, arity=2):
        opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
        X = torch.tensor((np.concatenate([S, U], 1) if arity == 2 else S).astype(np.float32),
                         device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad()
            huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # cut #4 gotcha (c)
            opt.step()
        net.eval()

    def fm_delta(net, norm, states, cmds, arity=2):
        with torch.no_grad():
            X = torch.tensor((np.concatenate([states, cmds], 1) if arity == 2
                              else states).astype(np.float32), device=device)
            Xn = (X - norm["mx"]) / norm["sx"]
            return (net(Xn) * norm["sy"] + norm["my"]).cpu().numpy()

    def pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs,
                            qc, cfg["q_range"], cfg["v_explore"])

    # Training/normalization pool is BROAD (that is what reward-free collection gives you,
    # and the FM should be good everywhere). GRADING is a separate question -- see below.
    nS, nU, nS2 = pool(eval_env, cfg["pool_n"], cfg["seed"] + 11)
    norm = make_norm(nS, nU, nS2)

    def r2_dims(pred, true, dims):
        sse = ((pred[:, dims] - true[:, dims]) ** 2).sum(0)
        sst = ((true[:, dims] - true[:, dims].mean(0)) ** 2).sum(0)
        return float(np.mean(1.0 - sse / (sst + 1e-12)))

    VEL = list(range(n, 2 * n))

    # ================================================================= #
    # CEM planner over an FM -- cost in CARTESIAN tip space via analytic FK
    # ================================================================= #
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
                    d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                    s = s + d
                    cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    # ================================================================= #
    # reaches: start posture + a goal a controlled TASK-SPACE distance away
    # ================================================================= #
    B = cfg["n_eval"]

    def eval_geometry(seed):
        """Sampling the goal in JOINT space guarantees reachability without IK, but for a
        REDUNDANT arm (n > 2) a random joint-space delta often lands in the null space and
        barely moves the tip -- which would silently fill the eval set with trivial
        "reaches" and flatten every controller toward the do-nothing floor. So we reject
        until the induced tip displacement sits in [reach_lo, reach_hi] metres."""
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
        achieved = float(np.mean(np.linalg.norm(goals - t0, axis=1)))
        out.setdefault("geometry", {})["mean_reach_dist_m"] = achieved
        print(f"[geometry] mean task-space reach distance = {achieved:.3f} m "
              f"(target band [{lo}, {hi}])", flush=True)
        starts = np.concatenate(
            [q0, rng.normal(0, cfg["v0_std"], (B, n))], 1).astype(np.float32)
        return starts, goals

    vel_seen = {}

    def rollout(plan_fn, starts, goals, replan_every, label=None, record=None):
        states = starts.copy(); plan = None; vmax = 0.0; vsum = 0.0; nv = 0
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                eval_env.set_state(states[b, :n].astype(np.float64),
                                   states[b, n:].astype(np.float64))
                s0b = eval_env.get_state()
                s2, _ = eval_env.step(acts[b], fs)
                if record is not None:
                    record.append((s0b, acts[b].astype(np.float32), s2))
                states[b] = s2
            v = np.abs(states[:, n:]).max(1)
            vmax = max(vmax, float(v.max())); vsum += float(v.mean()); nv += 1
        if label:
            vel_seen[label] = {"max_absqd": vmax, "mean_absqd": vsum / max(nv, 1)}
        tips = fk(states[:, :n].astype(np.float64), Ls)
        return float(np.median(np.linalg.norm(tips - goals, axis=1)))

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    # ================================================================= #
    # THE TASK-DISTRIBUTION PROBE SET (the fix for the first null)
    #
    # A first pass graded FM quality on the broad COLLECTION distribution (a wide
    # configuration box, isotropic random velocities) while grading control on REACHES --
    # two different distributions. The result was a null: fm_err moved 6x while control
    # wandered non-monotonically, because most of that error lived in states the reach
    # never visits. FM quality and control were literally not measured on the same states.
    #
    # This is the same correction drift_value_loop Cut 3 landed for the meta-loop's TEACHER:
    # the signal that predicts behaviour is the VALUE-RELEVANT FM prediction error (the
    # error over the goal corridor), not the global one. So the probe set is the set of
    # transitions actually visited by matched-FM reaches -- fixed once, reused for every FM
    # on the ladder, so the axis is controlled.
    # ================================================================= #
    ref_net = _mlp(cfg["seed"] + 40, cfg["fm_hidden"], cfg["fm_layers"], SD + AD, SD)
    rS, rU, rS2 = pool(make_env(a_test), cfg["pool_n"], cfg["seed"] + 400)
    train_fm(ref_net, norm, rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    rec = []
    rng_ref = np.random.default_rng(cfg["seed"] + 6000)
    rollout(lambda s, g: mpc_plan(ref_net, s, g, rng_ref), ev_starts, ev_goals, 1,
            "reference", record=rec)
    rng_ref2 = np.random.default_rng(cfg["seed"] + 6001)
    rollout(lambda s, g: mpc_plan(ref_net, s, g, rng_ref2), ev_starts, ev_goals, H,
            "reference_ballistic", record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    pS2 = np.array([r[2] for r in rec], np.float32)
    pT = (pS2 - pS).astype(np.float32)
    # a broad probe set is kept alongside, purely as the CONTRAST that documents the fix
    bS, bU, bS2 = pool(eval_env, cfg["probe_n"], cfg["seed"] + 12)
    bT = (bS2 - bS).astype(np.float32)
    print(f"[probes] task-distribution probe set: {len(pS)} transitions from matched-FM "
          f"reaches (reactive + ballistic); broad probe set: {len(bS)} transitions", flush=True)

    def probe_err(net):
        return float(np.linalg.norm(fm_delta(net, norm, pS, pU) - pT, axis=1).mean())

    def probe_err_broad(net):
        return float(np.linalg.norm(fm_delta(net, norm, bS, bU) - bT, axis=1).mean())

    # ================================================================= #
    # P1 -- capacity frontier, measured on the TASK distribution
    # ================================================================= #
    if cfg["do_capacity"]:
        cS, cU, cS2 = pool(eval_env, cfg["cap_pool_n"], cfg["seed"] + 21)
        norm1 = make_norm(cS, cU, cS2, arity=1)
        rows = []
        for h in cfg["cap_hidden"]:
            net2 = _mlp(cfg["seed"] + 40, h, 2, SD + AD, SD)
            train_fm(net2, norm, cS, cU, cS2, cfg["fm_steps"],
                     np.random.default_rng(cfg["seed"] + 31), arity=2)
            p2 = fm_delta(net2, norm, pS, pU, arity=2)
            net1 = _mlp(cfg["seed"] + 41, h, 2, SD, SD)
            train_fm(net1, norm1, cS, cU, cS2, cfg["fm_steps"],
                     np.random.default_rng(cfg["seed"] + 32), arity=1)
            p1 = fm_delta(net1, norm1, pS, pU, arity=1)
            rows.append({"hidden": h, "params": sum(p.numel() for p in net2.parameters()),
                         "r2_vel_arity2": r2_dims(p2, pT, VEL),
                         "r2_all_arity2": r2_dims(p2, pT, list(range(SD))),
                         "r2_vel_arity1": r2_dims(p1, pT, VEL),
                         "err_arity2": float(np.linalg.norm(p2 - pT, axis=1).mean())})
            print(f"[P1 capacity] hidden={h:4d} ({rows[-1]['params']:6d} par)  arity2 vel-R²="
                  f"{rows[-1]['r2_vel_arity2']:+.4f}  arity1 vel-R²="
                  f"{rows[-1]['r2_vel_arity1']:+.4f}", flush=True)
        out["P1"] = {"rows": rows}
        lo, hi = rows[0]["r2_vel_arity2"], rows[-1]["r2_vel_arity2"]
        print(f"[P1 capacity] frontier span (smallest→largest): {lo:+.4f} → {hi:+.4f}   "
              f"Δ={hi-lo:+.4f}\n[P1 capacity] pusher reference: FLAT at ~0.999 from 232 "
              f"params (cut #2) -- a large Δ means capacity binds INTRINSICALLY", flush=True)

    # ================================================================= #
    # the FM-quality ladder (shared by P2 / P5): one FM per training axis value
    # ================================================================= #
    fms = {}
    ladder = []
    for a_tr in a_trains:
        tenv = make_env(a_tr)
        S, U, S2 = pool(tenv, cfg["pool_n"], cfg["seed"] + 400 + int(100 * a_tr))
        net = _mlp(cfg["seed"] + 40, cfg["fm_hidden"], cfg["fm_layers"], SD + AD, SD)
        train_fm(net, norm, S, U, S2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
        fms[a_tr] = net
        ladder.append({"a_train": a_tr, "staleness": abs(a_tr - a_test),
                       "fm_err": probe_err(net), "fm_err_broad": probe_err_broad(net)})
        print(f"[ladder] a_train={a_tr:.3f} (stale {ladder[-1]['staleness']:.3f})  "
              f"fm_err(task)={ladder[-1]['fm_err']:.5f}  "
              f"fm_err(broad)={ladder[-1]['fm_err_broad']:.5f}", flush=True)

    matched = fms[min(fms, key=lambda a: abs(a - a_test))]
    stalest = fms[max(fms, key=lambda a: abs(a - a_test))]

    # ---- P2: is the staleness error state-dependent in the way the physics predicts? ---- #
    if cfg["do_locality"]:
        # payload -> inertia scales with tip RADIUS r(q); curl -> force scales with tip SPEED
        if AXIS == "payload":
            bvar = np.linalg.norm(fk(pS[:, :n].astype(np.float64), Ls), axis=1)
            bname, pred = "tip radius r(q)=‖FK(q)‖", "a tip payload's inertial effect scales with r²"
        else:
            ang = np.cumsum(pS[:, :n].astype(np.float64), axis=1)
            jx = -(Ls * np.sin(ang))[:, ::-1].cumsum(1)[:, ::-1]
            jy = (Ls * np.cos(ang))[:, ::-1].cumsum(1)[:, ::-1]
            qd = pS[:, n:].astype(np.float64)
            bvar = np.linalg.norm(np.stack([(jx * qd).sum(1), (jy * qd).sum(1)], 1), axis=1)
            bname, pred = "tip speed ‖v_tip‖", "a curl field's force is proportional to ‖v_tip‖"
        edges = np.quantile(bvar, np.linspace(0, 1, cfg["r_bins"] + 1)); edges[-1] += 1e-9
        e_m = np.linalg.norm(fm_delta(matched, norm, pS, pU) - pT, axis=1)
        e_s = np.linalg.norm(fm_delta(stalest, norm, pS, pU) - pT, axis=1)
        bins = []
        for i in range(cfg["r_bins"]):
            m = (bvar >= edges[i]) & (bvar < edges[i + 1])
            if m.sum() < 5:
                continue
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                         "mid": float(bvar[m].mean()), "n": int(m.sum()),
                         "err_matched": float(e_m[m].mean()), "err_stale": float(e_s[m].mean()),
                         "excess": float((e_s[m] - e_m[m]).mean())})
        exc = np.array([b["excess"] for b in bins]); rm = np.array([b["mid"] for b in bins])
        slope = float(np.polyfit(rm, exc, 1)[0]) if len(bins) >= 2 else float("nan")
        ratio = float(exc[-1] / (exc[0] + 1e-12)) if len(bins) >= 2 else float("nan")
        out["P2"] = {"ladder": ladder, "bin_var": bname, "r_bins": bins,
                     "excess_vs_r_slope": slope, "excess_extended_over_folded": ratio}
        for b in bins:
            print(f"[P2 locality] {bname}≈{b['mid']:.3f}  excess stale-err={b['excess']:.5f}  "
                  f"(matched {b['err_matched']:.5f} → stale {b['err_stale']:.5f})", flush=True)
        print(f"[P2 locality] d(excess)/d({bname}) = {slope:+.4f}, high/low ratio = {ratio:.2f}x\n"
              f"[P2 locality] PREDICTION: >1 -- {pred}. That is intrinsic, physically-derived "
              f"collection scarcity (no rot_regions scaffolding).", flush=True)
    else:
        out["P2"] = {"ladder": ladder}

    # ================================================================= #
    # P3 -- is the reach feasible with a MATCHED FM? (cut 4b precondition #2)
    # ================================================================= #
    if cfg["do_feasibility"]:
        rngA = np.random.default_rng(cfg["seed"] + 7000)
        rngB = np.random.default_rng(cfg["seed"] + 7001)
        rngR = np.random.default_rng(cfg["seed"] + 7002)
        zero_floor = rollout(lambda s, g: np.zeros((len(s), H, AD), np.float32), ev_starts, ev_goals, H)
        rand_floor = rollout(
            lambda s, g: rngR.uniform(-1, 1, (len(s), H, AD)).astype(np.float32), ev_starts, ev_goals, H)
        eval_env.reset_wrap()
        reac = rollout(lambda s, g: mpc_plan(matched, s, g, rngA), ev_starts, ev_goals, 1, "reactive")
        ball = rollout(lambda s, g: mpc_plan(matched, s, g, rngB), ev_starts, ev_goals, H, "ballistic")
        out["P3"] = {"do_nothing": zero_floor, "random_action": rand_floor,
                     "reactive_matched": reac, "ballistic_matched": ball,
                     "ballistic_frac_of_floor": float(ball / (zero_floor + 1e-9)),
                     "controlled_max_absq": float(eval_env.max_absq()),
                     "operating_velocity": vel_seen, "collection_v_explore": cfg["v_explore"]}
        print(f"\n[P3 feasibility] do-nothing floor={zero_floor:.4f}  random={rand_floor:.4f}"
              f"  |  reactive={reac:.4f}  ballistic={ball:.4f}", flush=True)
        print(f"[P3 feasibility] ballistic reaches {100*(1-ball/zero_floor):.0f}% of the way to "
              f"the goal vs doing nothing.", flush=True)
        print(f"[P3 feasibility] OPERATING |q̇|: ballistic mean="
              f"{vel_seen.get('ballistic', {}).get('mean_absqd', float('nan')):.2f} max="
              f"{vel_seen.get('ballistic', {}).get('max_absqd', float('nan')):.2f} rad/s  vs "
              f"collection v_explore={cfg['v_explore']}", flush=True)

    # ================================================================= #
    # P4 -- composition horizon of the one-step FM (pusher reference: ~6-8 steps)
    # ================================================================= #
    if cfg["do_composition"]:
        rngC = np.random.default_rng(cfg["seed"] + 900)
        Bc = cfg["comp_batch"]
        q0 = qc[None, :] + rngC.uniform(-cfg["q_jit"], cfg["q_jit"], (Bc, n))
        v0 = rngC.normal(0, cfg["v0_std"] if cfg["v0_std"] > 0 else 0.3, (Bc, n))
        seq = rngC.uniform(-1, 1, (Bc, cfg["comp_H"], AD)).astype(np.float32)
        s_true = np.concatenate([q0, v0], 1).astype(np.float32)
        s_fm = s_true.copy()
        curve = []
        for h in range(cfg["comp_H"]):
            s_fm = s_fm + fm_delta(matched, norm, s_fm, seq[:, h, :])
            nxt = np.empty_like(s_true)
            for b in range(Bc):
                eval_env.set_state(s_true[b, :n].astype(np.float64), s_true[b, n:].astype(np.float64))
                nxt[b], _ = eval_env.step(seq[b, h], fs)
            s_true = nxt
            tf = fk(s_fm[:, :n].astype(np.float64), Ls)
            tt = fk(s_true[:, :n].astype(np.float64), Ls)
            curve.append({"h": h + 1, "tip_err": float(np.linalg.norm(tf - tt, axis=1).mean())})
        thr = cfg["comp_thresh"]
        horizon = next((c["h"] for c in curve if c["tip_err"] > thr), cfg["comp_H"])
        out["P4"] = {"curve": curve, "threshold": thr, "composition_horizon": horizon}
        print(f"\n[P4 composition] tip-error of an FM rollout vs the true sim:", flush=True)
        for c in curve:
            if c["h"] <= 4 or c["h"] % 4 == 0:
                print(f"    h={c['h']:3d}  {c['tip_err']:.4f} m", flush=True)
        print(f"[P4 composition] horizon @ {thr} m = {horizon} steps (plan H={H}; an open-loop "
              f"plan longer than this runs past where the FM is trustworthy -- cut #3 saw even "
              f"the ORACLE degrade there)", flush=True)

    # ================================================================= #
    # P5 -- mini cut-4b: does the ballistic/reactive transmission dissociation reproduce?
    # ================================================================= #
    if cfg["do_transmission"]:
        for rec_i in ladder:
            net = fms[rec_i["a_train"]]
            rngA = np.random.default_rng(cfg["seed"] + 7100)
            rngB = np.random.default_rng(cfg["seed"] + 7101)
            rec_i["reactive"] = rollout(lambda s, g: mpc_plan(net, s, g, rngA), ev_starts, ev_goals, 1)
            rec_i["ballistic_cem"] = rollout(lambda s, g: mpc_plan(net, s, g, rngB), ev_starts, ev_goals, H)
            print(f"[P5 transmission] a_train={rec_i['a_train']:.3f} fm_err={rec_i['fm_err']:.5f}  "
                  f"reactive={rec_i['reactive']:.4f}  ballistic={rec_i['ballistic_cem']:.4f}", flush=True)
        fe = np.array([r["fm_err"] for r in ladder])
        slopes = {}
        for c in ("reactive", "ballistic_cem"):
            cv = np.array([r[c] for r in ladder])
            slopes[c] = float(np.polyfit(fe, cv, 1)[0]) if fe.std() > 1e-12 else float("nan")
        slopes["ratio_ballistic_over_reactive"] = float(
            slopes["ballistic_cem"] / (slopes["reactive"] + 1e-12))
        out["P5"] = {"ladder": ladder, "slopes": slopes}
        print(f"\n[P5 transmission] slope d(control)/d(fm_err): reactive="
              f"{slopes['reactive']:+.3f}  ballistic_cem={slopes['ballistic_cem']:+.3f}  "
              f"ratio={slopes['ratio_ballistic_over_reactive']:.2f}x", flush=True)
        print(f"[P5 transmission] pusher reference: +0.35 / +1.07 = 3.0x.", flush=True)

    mq = out.get("P3", {}).get("controlled_max_absq")
    if mq is not None:
        out["wrapped_controlled"] = bool(mq > 3.0)
        print(f"\n[diag] max |q| under CONTROLLED rollouts = {mq:.2f} rad "
              f"({'OUT OF' if mq > 3.0 else 'inside'} the legible range)", flush=True)

    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "arm_probe", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"\n[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": out, "figures": figures}

def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figs = {}

    def _save(fig, nm):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); figs[nm] = buf.getvalue()

    if "P1" in R:
        rows = R["P1"]["rows"]
        h = [r["hidden"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.semilogx(h, [r["r2_vel_arity2"] for r in rows], "o-", color="#1c7ed6", lw=2.4,
                    ms=7, label="arity-2  f(s,u)")
        ax.semilogx(h, [r["r2_vel_arity1"] for r in rows], "s--", color="#adb5bd", lw=2.0,
                    ms=6, label="arity-1  f(s)  [command-blind anchor]")
        ax.axhline(0.999, color="#e8590c", ls=":", lw=1.6,
                   label="pusher: flat at 0.999 from 232 params")
        ax.set_xlabel("hidden width"); ax.set_ylabel("velocity-dim Δs  R²")
        ax.set_title("P1 — does capacity bind INTRINSICALLY on the arm?")
        ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
        _save(fig, "fig1_capacity.png")

    if "r_bins" in R.get("P2", {}):
        b = R["P2"]["r_bins"]
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        rm = [x["mid"] for x in b]
        ax.plot(rm, [x["err_matched"] for x in b], "o-", color="#2f9e44", lw=2.2, ms=6,
                label="matched FM")
        ax.plot(rm, [x["err_stale"] for x in b], "o-", color="#c92a2a", lw=2.2, ms=6,
                label="stale FM (wrong payload)")
        ax.fill_between(rm, [x["err_matched"] for x in b], [x["err_stale"] for x in b],
                        color="#c92a2a", alpha=0.13, label="excess = staleness damage")
        ax.set_xlabel(R["P2"].get("bin_var", "state variable") + "   (low → high)")
        ax.set_ylabel("FM prediction error on the true dynamics")
        ax.set_title(f"P2 — is staleness STATE-LOCAL in the predicted way?\n"
                     f"high/low excess ratio = {R['P2']['excess_extended_over_folded']:.2f}×")
        ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
        _save(fig, "fig2_locality.png")

    if "P4" in R:
        c = R["P4"]["curve"]
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot([x["h"] for x in c], [x["tip_err"] for x in c], "o-", color="#7048e8", lw=2.2, ms=5)
        ax.axhline(R["P4"]["threshold"], color="#868e96", ls=":", lw=1.5)
        ax.axvline(R["P4"]["composition_horizon"], color="#e8590c", ls="--", lw=1.8,
                   label=f"horizon = {R['P4']['composition_horizon']} steps")
        ax.set_xlabel("open-loop rollout steps"); ax.set_ylabel("tip error of FM rollout (m)")
        ax.set_title("P4 — how far does the one-step FM compose?")
        ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
        _save(fig, "fig3_composition.png")

    if "P5" in R:
        lad = R["P5"]["ladder"]; sl = R["P5"]["slopes"]
        fe = np.array([r["fm_err"] for r in lad]); o = np.argsort(fe)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for c, col, lbl in (("reactive", "#7048e8", "reactive (re-ground every step)"),
                            ("ballistic_cem", "#e8590c", "ballistic CEM (open-loop)")):
            ax.plot(fe[o], np.array([r[c] for r in lad])[o], "o-", color=col, lw=2.4, ms=7,
                    label=f"{lbl}   (slope {sl[c]:+.2f})")
        ax.set_xlabel("FM prediction error on the true dynamics (payload-staleness axis)")
        ax.set_ylabel("control tip-goal distance (m)")
        ax.set_title(f"P5 — transmission on a NONLINEAR second family\n"
                     f"ballistic/reactive slope ratio = {sl['ratio_ballistic_over_reactive']:.2f}× "
                     f"(pusher: 3.0×)")
        ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
        _save(fig, "fig4_transmission.png")
    return figs


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_arm_capacity_sweep(cfg: dict) -> dict:
    """Where, if anywhere, does FM capacity bind INTRINSICALLY on the arm?

    The first characterization run returned a NEGATIVE on this property: at the reference
    2-link / v_explore=1.5 / frame_skip=10 regime, an 8-wide (164-parameter) arity-2 FM
    already reaches velocity-dim R²=0.982, span to 256-wide only +0.018. That is nearly as
    flat as the pusher, i.e. the arm's nonlinearity is REAL but CHEAP -- Cut #4d's lesson
    ("energy != prediction cost; a reducible-but-simple factor is free to model") restated
    on new physics.

    The mechanism is structural, and it says exactly where to look. For a 2-link arm the
    inertia matrix M(q) depends on the ELBOW ANGLE ALONE, so the whole configuration-
    dependence of the dynamics is a smooth function of ONE variable -- trivial for a small
    MLP. Three knobs should each raise the intrinsic complexity, and they are separable:

      * `n_links`      -- M(q) for an n-link chain depends on n-1 angles, and the Coriolis
                          tensor gains cross terms. Complexity grows with the DIMENSION of
                          the configuration-dependence, not just its presence.
      * `v_explore`    -- the Coriolis/centrifugal term is QUADRATIC in q̇. At low speeds it
                          is a small correction to a near-linear map; at high speeds it
                          dominates. This is the amplitude knob on the nonlinearity.
      * `frame_skip`   -- a longer control interval integrates further along a curving
                          trajectory, so more of the flow's nonlinearity is folded into the
                          one-step map s,u -> Δs (the same integration effect that made
                          sub-timestep collision aliasing matter in cut #1).

    Readout is the CAPACITY REQUIREMENT: the smallest hidden width reaching R² >= `r2_target`
    on the velocity dims. The pusher's answer is 8 (cut #2: R²=0.998 at the smallest width
    tried). A regime that needs 64 or 128 is one where capacity genuinely competes, which is
    the precondition for every value-shaping / re-allocation cut in this directory.
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.arm_env import ArmEnv, collect_pool

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huber = nn.HuberLoss(delta=1.0)
    rows = []

    for reg in cfg["regimes"]:
        n = int(reg["n_links"])
        SD, AD = 2 * n, n
        fs = int(reg["frame_skip"])
        env = ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                          link_masses=cfg["link_masses"][:n],
                          joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                          payload_mass=cfg["p_test"]))
        qc = np.array(cfg["q_center"][:n], dtype=np.float64)
        S, U, S2 = collect_pool(env, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 21),
                                fs, qc, cfg["q_range"], reg["v_explore"])
        tS, tU, tS2 = collect_pool(env, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 22),
                                   fs, qc, cfg["q_range"], reg["v_explore"])
        tT = (tS2 - tS).astype(np.float32)
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        mx, sx = X.mean(0), X.std(0) + 1e-6
        my, sy = Y.mean(0), Y.std(0) + 1e-6
        Xt = torch.tensor((X - mx) / sx, device=device)
        Yt = torch.tensor((Y - my) / sy, device=device)
        Xv = torch.tensor((np.concatenate([tS, tU], 1).astype(np.float32) - mx) / sx, device=device)
        VEL = list(range(n, 2 * n))

        widths = []
        for h in cfg["cap_hidden"]:
            g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 40)
            lyr = [nn.Linear(SD + AD, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, SD)]
            net = nn.Sequential(*lyr)
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
            net = net.to(device)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
            brng = np.random.default_rng(cfg["seed"] + 31)
            bs = min(cfg["fm_batch"], len(S)); net.train()
            for _ in range(cfg["fm_steps"]):
                idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
                opt.zero_grad(); loss = huber(net(Xt[idx]), Yt[idx]); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                pred = (net(Xv) * torch.tensor(sy, device=device)
                        + torch.tensor(my, device=device)).cpu().numpy()
            sse = ((pred[:, VEL] - tT[:, VEL]) ** 2).sum(0)
            sst = ((tT[:, VEL] - tT[:, VEL].mean(0)) ** 2).sum(0)
            r2 = float(np.mean(1.0 - sse / (sst + 1e-12)))
            widths.append({"hidden": h, "params": sum(p.numel() for p in net.parameters()),
                           "r2_vel": r2})
        req = next((w["hidden"] for w in widths if w["r2_vel"] >= cfg["r2_target"]), None)
        # a scale-free companion readout: how much of the velocity variance is NOT explained
        # by the best LINEAR model of (s,u) -- i.e. how nonlinear the map is at all
        A = np.concatenate([np.concatenate([tS, tU], 1), np.ones((len(tS), 1), np.float32)], 1)
        coef, *_ = np.linalg.lstsq(A, tT, rcond=None)
        lin = A @ coef
        sse_l = ((lin[:, VEL] - tT[:, VEL]) ** 2).sum(0)
        sst_l = ((tT[:, VEL] - tT[:, VEL].mean(0)) ** 2).sum(0)
        r2_lin = float(np.mean(1.0 - sse_l / (sst_l + 1e-12)))
        rows.append({**reg, "widths": widths, "capacity_req": req, "r2_linear": r2_lin,
                     "r2_smallest": widths[0]["r2_vel"], "r2_largest": widths[-1]["r2_vel"],
                     "span": widths[-1]["r2_vel"] - widths[0]["r2_vel"]})
        print(f"[regime n={n} v={reg['v_explore']} fs={fs}] linear-R²={r2_lin:+.4f}  "
              f"width→R²≥{cfg['r2_target']}: {req}   "
              + "  ".join(f"h{w['hidden']}={w['r2_vel']:+.4f}" for w in widths), flush=True)

    print(f"\n[capacity] pusher reference: capacity_req = 8 (the smallest width tried already "
          f"gave R²=0.998, cut #2). A regime needing 64+ is one where capacity COMPETES.",
          flush=True)
    out = {"config": cfg, "rows": rows}
    figs = _sweep_figures(out)
    outdir = os.path.join(DATA_DIR, "arm_capacity_sweep", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figs.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figs)} figures to {outdir}", flush=True)
    return {"results": out, "figures": figs}


def _sweep_figures(R):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    rows = R["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        w = [x["hidden"] for x in r["widths"]]
        ax.semilogx(w, [x["r2_vel"] for x in r["widths"]], "o-", lw=2.2, ms=6,
                    color=cmap(i / max(1, len(rows) - 1)),
                    label=f"n={r['n_links']} v={r['v_explore']} fs={r['frame_skip']}  "
                          f"(lin-R²={r['r2_linear']:+.2f})")
    ax.axhline(R["config"]["r2_target"], color="#c92a2a", ls=":", lw=1.6,
               label=f"R² target {R['config']['r2_target']}")
    ax.set_xlabel("hidden width"); ax.set_ylabel("velocity-dim Δs  R²")
    ax.set_title("Where does FM capacity bind intrinsically on the arm?\n"
                 "flat & high = the pusher's problem (nonlinearity present but cheap)")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return {"fig_capacity_regimes.png": buf.getvalue()}


@app.local_entrypoint()
def arm_capacity_sweep(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_links_list: str = "2,3",
    v_explore_list: str = "1.5,4.0,8.0",
    frame_skip_list: str = "10",
    link_lengths: str = "0.5,0.5,0.3",
    link_masses: str = "1.0,1.0,0.5",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    p_test: float = 0.0,
    q_center: str = "0.4,1.2,0.0",
    q_range: float = 0.9,
    pool_n: int = 9000,
    probe_n: int = 1500,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    cap_hidden: str = "8,16,32,64,128,256",
    r2_target: float = 0.99,
):
    import os

    regimes = [dict(n_links=n, v_explore=v, frame_skip=f)
               for f in [int(x) for x in frame_skip_list.split(",") if x]
               for n in [int(x) for x in n_links_list.split(",") if x]
               for v in [float(x) for x in v_explore_list.split(",") if x]]
    CH = [int(x) for x in cap_hidden.split(",") if x]
    if quick:
        regimes = regimes[:2]; CH = [8, 64, 256]; pool_n = 3000; probe_n = 500; fm_steps = 1200
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(tag=tag, seed=seed, regimes=regimes,
               link_lengths=[float(x) for x in link_lengths.split(",") if x],
               link_masses=[float(x) for x in link_masses.split(",") if x],
               joint_damping=joint_damping, gear=gear, p_test=p_test,
               q_center=[float(x) for x in q_center.split(",") if x], q_range=q_range,
               pool_n=pool_n, probe_n=probe_n, fm_lr=fm_lr, fm_batch=fm_batch,
               fm_steps=fm_steps, cap_hidden=CH, r2_target=r2_target)
    out = run_arm_capacity_sweep.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "arm_capacity_sweep_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")


@app.local_entrypoint()
def arm_probe(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # arm geometry / DGP
    n_links: int = 2,
    link_lengths: str = "0.5,0.5,0.3",
    link_masses: str = "1.0,1.0,0.5",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    # the FM-quality axis. `payload` = tip point mass (the INERTIAL-LOCALITY axis);
    # `curl` = a velocity-dependent end-effector force field (the Shadmehr force-field
    # protocol -- perpendicular to motion, so its error ACCUMULATES open-loop, which makes
    # it the a-priori better TRANSMISSION axis, and it is what aftereffects are defined on).
    # Staleness must be ONE-SIGNED (cut 4b) -- keep a_test at one end of a_trains.
    axis: str = "payload",
    a_test: float = 0.0,
    a_trains: str = "0.0,0.25,0.5,1.0,1.5",
    payload_base: float = 0.0,   # payload held fixed when axis="curl"
    # collection
    frame_skip: int = 10,
    q_center: str = "0.4,1.2,0.0",
    q_range: float = 0.9,
    v_explore: float = 1.5,
    pool_n: int = 9000,
    probe_n: int = 1500,
    cap_pool_n: int = 9000,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    cap_hidden: str = "8,16,32,64,128,256",
    # reaches / control
    n_eval: int = 32,
    q_jit: float = 0.25,
    reach_amp: float = 0.6,
    reach_lo: float = 0.20,     # accepted task-space reach distance band (metres) -- for a
    reach_hi: float = 0.45,     # redundant arm a random joint delta can be a null-space no-op
    reach_tries: int = 40,
    v0_std: float = 0.0,
    plan_h: int = 25,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    # probes
    r_bins: int = 6,
    comp_batch: int = 128,
    comp_h: int = 30,
    comp_thresh: float = 0.05,
    do_capacity: bool = True,
    do_locality: bool = True,
    do_feasibility: bool = True,
    do_composition: bool = True,
    do_transmission: bool = True,
):
    import os

    LL = [float(x) for x in link_lengths.split(",") if x]
    LM = [float(x) for x in link_masses.split(",") if x]
    QC = [float(x) for x in q_center.split(",") if x]
    AT = [float(x) for x in a_trains.split(",") if x]
    CH = [int(x) for x in cap_hidden.split(",") if x]
    if quick:
        pool_n = 3000; probe_n = 500; cap_pool_n = 3000; fm_steps = 1200
        fm_hidden = 128; fm_layers = 2; CH = [8, 32, 256]
        AT = [AT[0], AT[-1]]; n_eval = 12; k_shoot = 128; comp_batch = 48
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, n_links=n_links, link_lengths=LL, link_masses=LM,
        joint_damping=joint_damping, gear=gear, axis=axis, a_test=a_test,
        a_trains=AT, payload_base=payload_base,
        frame_skip=frame_skip, q_center=QC, q_range=q_range, v_explore=v_explore,
        pool_n=pool_n, probe_n=probe_n, cap_pool_n=cap_pool_n,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, cap_hidden=CH,
        n_eval=n_eval, q_jit=q_jit, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, reach_tries=reach_tries, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        r_bins=r_bins, comp_batch=comp_batch, comp_H=comp_h, comp_thresh=comp_thresh,
        do_capacity=do_capacity, do_locality=do_locality, do_feasibility=do_feasibility,
        do_composition=do_composition, do_transmission=do_transmission,
    )
    out = run_arm_probe.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "arm_probe_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_arm_tool_probe(cfg: dict) -> dict:
    """P6 -- capacity competition from the BODY: a passive tool, not a designed force field.

    On the pusher, capacity competition had to be MANUFACTURED (`puck_field`: a multi-mode
    Taylor-Green force field, with hand-picked modes/amplitudes/phases, plus a second
    `pusher_amp` copy bolted on so the value-RELEVANT side was capacity-hungry too). That is
    a lot of designer freedom carrying the scientific load. The arm can get the same
    competition from MORPHOLOGY: make the last `n_passive` joints unactuated -- a floppy
    tool or a hanging load. Real physics (a driven damped oscillator coupled to the arm),
    one legible knob, no invented field.

    Two properties make this the principled version:

      * `tool_mass` continuously spans Cut #4d's BOUNDARY CONDITION. A light tool is
        value-irrelevant AND dynamically decoupled -- free to drop, and 4d predicts
        value-shaping is control-NEUTRAL there. A heavy tool is value-irrelevant but
        strongly coupled through reaction torques -- dropping it should cost control. One
        physical parameter, in kg, sweeping the axis 4d had to build with `field_pusher_amp`.

      * `goal_site` flips VALUE-RELEVANCE with the physics held byte-identical. `hand`
        (upstream of the tool) makes the tool's state value-irrelevant; `tip` makes it
        value-critical. Same XML, same dynamics, same trajectories -- the intervention is on
        the VALUE ALONE. The pusher could not do this: making the puck value-relevant meant
        changing the task, which also changed its difficulty.

    Readouts, per tool mass:
      (a) COUPLING -- a tool-BLIND FM f(s_arm, u) -> Δs_arm vs the full FM, both scored on
          the ARM's velocity dims. The gap is exactly "what does ignoring the tool cost me
          on the part I care about", i.e. an operational definition of
          value-irrelevant-AND-decoupled that needs no hand-set mask.
      (b) CAPACITY COST -- width needed to reach the R² target, full vs blind. How much of
          the FM's capacity the tool consumes.
      (c) CONTROL -- CEM over the blind FM vs over the full FM, both executed in the TRUE
          env with a `hand` goal. This is the arm's version of Cut #4d's headline: dropping
          the value-irrelevant subsystem should be free when decoupled and costly when
          competing. The blind planner is legitimate because the hand's kinematics depend
          only on the actuated joints -- verified against MuJoCo's own `hand` site below.
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.arm_env import ArmEnv, collect_pool, fk

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs, H, n = cfg["frame_skip"], cfg["plan_H"], cfg["n_links"]
    n_pas = cfg["n_passive"]; n_act = n - n_pas
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    ARM_S = list(range(n_act)) + list(range(n, n + n_act))     # arm slice of the full state
    ARM_V = list(range(n_act, 2 * n_act))                      # velocity half of the arm slice
    FULL_V = list(range(n, n + n_act))                         # same dims inside the full state
    huber = nn.HuberLoss(delta=1.0)
    out = {"config": cfg, "rows": []}
    print(f"[setup] n={n} n_passive={n_pas} n_act={n_act} goal_site={cfg['goal_site']} "
          f"tool_masses={cfg['tool_masses']}", flush=True)

    def make_env(tm):
        return ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                           link_masses=cfg["link_masses"][:n],
                           joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                           n_passive=n_pas, tool_mass=tm, goal_site=cfg["goal_site"]))

    def _mlp(seed, hidden, in_dim, out_dim, layers=3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(hidden, out_dim)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def fit(S_in, Y, hidden, steps, seed, layers=3):
        mx, sx = S_in.mean(0), S_in.std(0) + 1e-6
        my, sy = Y.mean(0), Y.std(0) + 1e-6
        Xt = torch.tensor((S_in - mx) / sx, device=device, dtype=torch.float32)
        Yt = torch.tensor((Y - my) / sy, device=device, dtype=torch.float32)
        net = _mlp(seed, hidden, S_in.shape[1], Y.shape[1], layers)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
        brng = np.random.default_rng(seed + 7); bs = min(cfg["fm_batch"], len(S_in)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S_in), size=bs), device=device)
            opt.zero_grad(); huber(net(Xt[idx]), Yt[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        nrm = {"mx": torch.tensor(mx, device=device), "sx": torch.tensor(sx, device=device),
               "my": torch.tensor(my, device=device), "sy": torch.tensor(sy, device=device)}
        return net, nrm

    def pred(net, nrm, S_in):
        with torch.no_grad():
            X = torch.tensor(S_in, device=device, dtype=torch.float32)
            return ((net((X - nrm["mx"]) / nrm["sx"]) * nrm["sy"] + nrm["my"])
                    .cpu().numpy())

    def r2(p, t, dims):
        sse = ((p[:, dims] - t[:, dims]) ** 2).sum(0)
        sst = ((t[:, dims] - t[:, dims].mean(0)) ** 2).sum(0)
        return float(np.mean(1.0 - sse / (sst + 1e-12)))

    # --- sanity: is the HAND kinematically independent of the tool joints? ------------- #
    e0 = make_env(cfg["tool_masses"][0])
    rngk = np.random.default_rng(cfg["seed"] + 3)
    hand_err = 0.0
    for _ in range(200):
        q = qc + rngk.uniform(-1.0, 1.0, n)
        e0.set_state(q, np.zeros(n))
        hand_err = max(hand_err, float(np.abs(e0.goal_pos() - fk(q, Ls, upto=e0.goal_link)).max()))
    out["hand_fk_max_abs_err"] = hand_err
    print(f"[P6 sanity] analytic FK(upto={e0.goal_link}) vs MuJoCo '{cfg['goal_site']}' site: "
          f"max|err|={hand_err:.3e}  -- the blind planner's cost is well-defined iff this is ~0",
          flush=True)

    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q, k):
        ang = torch.cumsum(q[:, :k], dim=1)
        return torch.stack([(Lt[:k] * torch.cos(ang)).sum(1),
                            (Lt[:k] * torch.sin(ang)).sum(1)], 1)

    B = cfg["n_eval"]
    goal_k = e0.goal_link

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (B, n))
        t0 = fk(q0, Ls, upto=goal_k)
        qg = np.empty_like(q0)
        for b in range(B):
            best, bp = None, np.inf
            for _ in range(cfg["reach_tries"]):
                d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                cand = q0[b] + cfg["reach_amp"] * d
                dist = float(np.linalg.norm(fk(cand, Ls, upto=goal_k) - t0[b]))
                pen = max(0.0, cfg["reach_lo"] - dist) + max(0.0, dist - cfg["reach_hi"])
                if pen < bp:
                    best, bp = cand, pen
                if pen == 0.0:
                    break
            qg[b] = best
        goals = fk(qg, Ls, upto=goal_k).astype(np.float32)
        starts = np.concatenate([q0, np.zeros((B, n))], 1).astype(np.float32)
        print(f"[geometry] mean task-space reach = "
              f"{float(np.mean(np.linalg.norm(goals - t0, axis=1))):.3f} m", flush=True)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    for tm in cfg["tool_masses"]:
        env = make_env(tm)
        S, U, S2 = collect_pool(env, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 21),
                                fs, qc, cfg["q_range"], cfg["v_explore"])
        tS, tU, tS2 = collect_pool(env, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 22),
                                   fs, qc, cfg["q_range"], cfg["v_explore"])
        Yf, tYf = (S2 - S).astype(np.float32), (tS2 - tS).astype(np.float32)
        Xf = np.concatenate([S, U], 1).astype(np.float32)
        tXf = np.concatenate([tS, tU], 1).astype(np.float32)
        Xb = np.concatenate([S[:, ARM_S], U], 1).astype(np.float32)
        tXb = np.concatenate([tS[:, ARM_S], tU], 1).astype(np.float32)
        Yb, tYb = Yf[:, ARM_S], tYf[:, ARM_S]

        widths = []
        for h in cfg["cap_hidden"]:
            nf, rf = fit(Xf, Yf, h, cfg["fm_steps"], cfg["seed"] + 40)
            nb, rb = fit(Xb, Yb, h, cfg["fm_steps"], cfg["seed"] + 41)
            widths.append({"hidden": h,
                           "r2_full_armvel": r2(pred(nf, rf, tXf), tYf, FULL_V),
                           "r2_blind_armvel": r2(pred(nb, rb, tXb), tYb, ARM_V)})
        req_f = next((w["hidden"] for w in widths if w["r2_full_armvel"] >= cfg["r2_target"]), None)
        req_b = next((w["hidden"] for w in widths if w["r2_blind_armvel"] >= cfg["r2_target"]), None)
        big = widths[-1]
        coupling = big["r2_full_armvel"] - big["r2_blind_armvel"]

        # ---- control: CEM over the blind FM vs the full FM, both in the TRUE env ---- #
        net_f, nrm_f = fit(Xf, Yf, cfg["fm_hidden"], cfg["fm_steps"], cfg["seed"] + 50)
        net_b, nrm_b = fit(Xb, Yb, cfg["fm_hidden"], cfg["fm_steps"], cfg["seed"] + 51)

        def mpc(net, nrm, states_full, goals, rng, blind):
            s_in = states_full[:, ARM_S] if blind else states_full
            Bn, AD = s_in.shape[0], n_act
            mu = np.zeros((Bn, H, AD), np.float32)
            sig = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(goals, device=device, dtype=torch.float32).repeat_interleave(cfg["k_shoot"], 0)
            s0 = torch.tensor(s_in, device=device, dtype=torch.float32).repeat_interleave(cfg["k_shoot"], 0)
            nq = n_act if blind else n
            for _ in range(cfg["cem_iters"]):
                e = rng.standard_normal((Bn, cfg["k_shoot"], H, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    sq = torch.tensor(seqs.reshape(Bn * cfg["k_shoot"], H, AD), device=device)
                    cost = torch.zeros(Bn * cfg["k_shoot"], device=device)
                    for hh in range(H):
                        x = torch.cat([s, sq[:, hh, :]], 1)
                        s = s + (net((x - nrm["mx"]) / nrm["sx"]) * nrm["sy"] + nrm["my"])
                        cost = cost + (fk_torch(s[:, :nq], goal_k) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, nq:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, cfg["k_shoot"]), cfg["cem_elite"], 1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)

        def rollout(plan_fn, replan_every):
            states = ev_starts.copy(); plan = None
            for step in range(H):
                if step % replan_every == 0:
                    plan = plan_fn(states)
                acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
                for b in range(B):
                    env.set_state(states[b, :n].astype(np.float64), states[b, n:].astype(np.float64))
                    states[b], _ = env.step(acts[b], fs)
            pos = fk(states[:, :n].astype(np.float64), Ls, upto=goal_k)
            return float(np.median(np.linalg.norm(pos - ev_goals, axis=1)))

        ctl = {}
        for nm, (net, nrm, bl) in {"full": (net_f, nrm_f, False), "blind": (net_b, nrm_b, True)}.items():
            for mode, re_ in (("reactive", 1), ("ballistic", H)):
                r = np.random.default_rng(cfg["seed"] + 800)
                ctl[f"{nm}_{mode}"] = rollout(lambda s, _n=net, _r=nrm, _b=bl, _rg=r:
                                              mpc(_n, _r, s, ev_goals, _rg, _b), re_)
        zero = rollout(lambda s: np.zeros((B, H, n_act), np.float32), H)

        row = {"tool_mass": tm, "widths": widths, "capacity_req_full": req_f,
               "capacity_req_blind": req_b, "coupling_r2_gap": coupling,
               "control": ctl, "do_nothing": zero,
               "blind_cost_reactive": ctl["blind_reactive"] - ctl["full_reactive"],
               "blind_cost_ballistic": ctl["blind_ballistic"] - ctl["full_ballistic"]}
        out["rows"].append(row)
        print(f"\n[tool_mass={tm}] coupling (full−blind arm-vel R² @h{big['hidden']}) = "
              f"{coupling:+.4f}   capacity_req full={req_f} blind={req_b}", flush=True)
        print(f"[tool_mass={tm}] control (do-nothing {zero:.4f}): "
              f"reactive full={ctl['full_reactive']:.4f} blind={ctl['blind_reactive']:.4f} "
              f"(cost {row['blind_cost_reactive']:+.4f})  |  ballistic "
              f"full={ctl['full_ballistic']:.4f} blind={ctl['blind_ballistic']:.4f} "
              f"(cost {row['blind_cost_ballistic']:+.4f})", flush=True)

    print(f"\n[P6] PREDICTION (Cut #4d's boundary condition, now in kg): dropping the tool is "
          f"FREE at low tool_mass (coupling≈0, blind cost≈0) and COSTLY at high tool_mass.",
          flush=True)
    figs = _tool_figures(out)
    outdir = os.path.join(DATA_DIR, "arm_tool_probe", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figs.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figs)} figures to {outdir}", flush=True)
    return {"results": out, "figures": figs}


def _tool_figures(R):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    rows = R["rows"]; tm = [r["tool_mass"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    axes[0].plot(tm, [r["coupling_r2_gap"] for r in rows], "o-", color="#1c7ed6", lw=2.4, ms=7)
    axes[0].axhline(0, color="#adb5bd", lw=1)
    axes[0].set_xlabel("tool mass (kg)"); axes[0].set_ylabel("arm-vel R²:  full − tool-blind")
    axes[0].set_title("COUPLING — what ignoring the tool costs\nthe part the task cares about")
    axes[0].grid(alpha=0.25)
    axes[1].plot(tm, [r["blind_cost_reactive"] for r in rows], "o-", color="#7048e8",
                 lw=2.4, ms=7, label="reactive")
    axes[1].plot(tm, [r["blind_cost_ballistic"] for r in rows], "o-", color="#e8590c",
                 lw=2.4, ms=7, label="ballistic")
    axes[1].axhline(0, color="#adb5bd", lw=1)
    axes[1].set_xlabel("tool mass (kg)"); axes[1].set_ylabel("control cost of dropping the tool (m)")
    axes[1].set_title("CONTROL — Cut #4d's boundary condition,\nnow spanned by one physical knob")
    axes[1].legend(fontsize=8.5); axes[1].grid(alpha=0.25)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return {"fig_tool_coupling.png": buf.getvalue()}


@app.local_entrypoint()
def arm_tool_probe(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_links: int = 5,
    n_passive: int = 2,
    goal_site: str = "hand",
    tool_masses: str = "0.02,0.1,0.3,0.8,1.6",
    link_lengths: str = "0.4,0.4,0.3,0.25,0.2",
    link_masses: str = "1.0,1.0,0.6,0.4,0.3",
    q_center: str = "0.4,0.8,0.6,0.0,0.0",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 6.0,
    pool_n: int = 14000,
    probe_n: int = 1500,
    fm_hidden: int = 256,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 5000,
    cap_hidden: str = "8,32,128,256",
    r2_target: float = 0.99,
    n_eval: int = 32,
    q_jit: float = 0.25,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    reach_tries: int = 40,
    plan_h: int = 14,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    TM = [float(x) for x in tool_masses.split(",") if x]
    CH = [int(x) for x in cap_hidden.split(",") if x]
    if quick:
        TM = [TM[0], TM[-1]]; CH = [8, 256]; pool_n = 3000; probe_n = 500
        fm_steps = 1200; n_eval = 12; k_shoot = 128
        tag = tag or "smoke"
    tag = tag or "default"
    cfg = dict(tag=tag, seed=seed, n_links=n_links, n_passive=n_passive, goal_site=goal_site,
               tool_masses=TM,
               link_lengths=[float(x) for x in link_lengths.split(",") if x],
               link_masses=[float(x) for x in link_masses.split(",") if x],
               q_center=[float(x) for x in q_center.split(",") if x],
               joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
               q_range=q_range, v_explore=v_explore, pool_n=pool_n, probe_n=probe_n,
               fm_hidden=fm_hidden, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
               cap_hidden=CH, r2_target=r2_target, n_eval=n_eval, q_jit=q_jit,
               reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
               reach_tries=reach_tries, plan_H=plan_h, k_shoot=k_shoot,
               cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma,
               vel_pen=vel_pen)
    out = run_arm_tool_probe.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "arm_tool_probe_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
