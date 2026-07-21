"""The FULLY-ONLINE efferent two-timescale loop: reward continuously shapes ONE LIVE FM.

Program: `ideas/two_timescale_value_loop.md` — the keystone gap the doc names verbatim
(interface next-piece #2): Cut #4e closed "an outer loop CAUSES the shaping" but OFFLINE
(a CEM outer loop over FMs RETRAINED-from-scratch per candidate weight). The doc / the
mujoco README both flag the remaining rung:

    "Still a rung short of the FULLY-ONLINE single two-timescale loop (reward continuously
     shaping a *live* FM, vs an outer optimization over retrained FMs)."

This cut closes that rung. ONE live context-FM trains continuously; a SLOW reward-driven
outer loop nudges its per-dim capacity allocation ONLINE (every `outer_every` inner SGD
steps), driven ONLY by real-env CEM-MPC control performance. This is the Wang et al. (2018)
two-timescale meta-RL structure in its purest form: a slow scalar (dopaminergic) reward loop
modulating a fast learner that never stops learning — no retraining, no backprop through the
simulator.

    Why efferent, not afferent. The sibling afferent build (`meta_curiosity_loop.py`,
    self-tuning the explore/exploit balance of the collection drive) is a clean CONTROL-
    NEGATIVE on this substrate: the drive's tracking gradient is real (pure-explore tracks a
    drifting needle ~3x worse, noisy-TV-seduced) but CEM-MPC control is ROBUST to it (feedback
    tolerates a stale model; the exploit/task signal already localizes the frontier), so the
    reward-over-balance landscape is flat and the loop has nothing to discover. The EFFERENT
    lever is different: Cut #4e PROVED its reward landscape has a real control gradient under
    capacity competition (reward prefers the puck-drop, +0.036 at h=64; converged allocation
    corr->support +0.39). So the efferent online loop has a genuine set-point to find.

THE CLAIM (the 2x2, from #4e made online).
  * CAPACITY COMPETITION (`--field-pusher-amp 2.0`, a pusher force field makes the value-
    relevant dynamics capacity-hungry too): the online loop DRIVES w_puck DOWN — it discovers,
    from control reward alone, that it should stop spending a capacity-limited live FM on the
    value-irrelevant puck, freeing capacity for the pusher. Control improves vs a veridical
    (w_puck=1) online baseline; per-dim R^2 shows the re-allocation (pusher-vel up, puck-vel
    down) happening IN ONE LIVE FM.
  * EASY PUSHER (`--field-pusher-amp 0`): no competition -> reward is INDIFFERENT to w_puck
    (the #4e control-neutrality), so the online w_puck stays flat / wanders. The dissociation.

ARMS (all share FM init + data order -> only the w-source differs; the control-variable
discipline). `online` tunes w_puck = sigmoid(theta) by advantage-normalized REINFORCE on
control reward; `w1.0` (veridical), `w0.5`, `w0.0` (puck-dropped) hold it fixed = the
reference lines the online loop should match/beat.

WIREHEADING (the non-optional control). w_puck in (0,1) with the pusher weight FIXED at 1,
and the reward is REAL-env CEM-MPC control (NOT the training loss) -> lowering w_puck cannot
game the reward (it only helps if dropping the puck genuinely frees control-relevant
capacity). No loss-scale DOF to hack (that was #4e-A2's free-w concern; a scalar puck-weight
with a fixed pusher weight has none). Stated, not separately tested.

Substrate/machinery reused verbatim from meta_value_learn.py (Cut #4e). Only the training is
restructured from retrain-per-candidate (offline) to one continuously-trained live FM (online).

Run:
    cd experiments/
    modal run mujoco_control/meta_value_online.py::meta_value_online --quick                       # smoke
    # the 2x2 (3 seeds each):
    for s in 0 1 2; do
      modal run --detach mujoco_control/meta_value_online.py::meta_value_online --tag comp_s$s --seed $s --field-pusher-amp 2.0
      modal run --detach mujoco_control/meta_value_online.py::meta_value_online --tag easy_s$s --seed $s --field-pusher-amp 0.0
    done
"""

import json
import math
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder

PUSHER_POS = [0, 1]
PUCK_POS = [2, 3]
PUSHER_VEL = [4, 5]
PUCK_VEL = [6, 7]
PUSHER = [0, 1, 4, 5]     # value-relevant (support of V = -||pos-g|| - beta*||vel||)
PUCK = [2, 3, 6, 7]       # value-irrelevant distractor

DEFAULT_ARMS = ["online", "w1.0", "w0.5", "w0.0"]


def _parse_fixed_w(arm):
    if arm.startswith("w") and arm[1:].replace(".", "", 1).isdigit():
        return float(arm[1:])
    return None


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_meta_value_online(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import collect_transitions, PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = cfg["arms"]
    print(f"[setup] device={device} field_pusher_amp={cfg['field_pusher_amp']} "
          f"({'CAPACITY COMPETITION' if cfg['field_pusher_amp'] > 0 else 'easy pusher'}) arms={arms}", flush=True)

    fs = cfg["frame_skip"]
    d = cfg["latent_dim"]
    hidden = cfg["hidden"]
    N_ctx = cfg["n_context"]

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    # ---- task family: input-coupled command rotation phi in [-Phi, Phi] ------ #
    Phi = cfg["conflict"]
    n_tr, n_te = cfg["n_train_tasks"], cfg["n_test_tasks"]
    all_p = np.sort(np.linspace(-Phi, Phi, n_tr + n_te))
    te_idx = np.unique(np.linspace(1, n_tr + n_te - 2, n_te).round().astype(int))
    tr_idx = np.array([i for i in range(n_tr + n_te) if i not in te_idx])
    train_p = all_p[tr_idx].astype(np.float64)
    test_p = all_p[te_idx].astype(np.float64)

    def task_dgp(p):
        dd = dict(cfg["dgp_base"]); dd["push_rot"] = float(p); return dd

    def collect(dgp, seed):
        return collect_transitions(dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
                                   frame_skip=fs, seed=seed, sigma=cfg["sigma"], theta=cfg["theta"],
                                   seek_gain=cfg["seek_gain"])

    # ===================================================================== #
    # 1. collect per-task buffers + family normalization (verbatim from 4e)
    # ===================================================================== #
    print("[1] collecting per-task buffers ...", flush=True)
    train_data = [collect(task_dgp(p), cfg["seed"] + 1000 + i) for i, p in enumerate(train_p)]
    test_data = [collect(task_dgp(p), cfg["seed"] + 2000 + i) for i, p in enumerate(test_p)]

    allS = np.concatenate([dt["S"] for dt in train_data])
    allU = np.concatenate([dt["U"] for dt in train_data])
    allS2 = np.concatenate([dt["S2"] for dt in train_data])
    X = np.concatenate([allS, allU], 1).astype(np.float32)
    Y = (allS2 - allS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
    mx, sx, my, sy = norm["mx"], norm["sx"], norm["my"], norm["sy"]

    def feats(S, U, S2):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (Xt - mx) / sx; dy_n = (Yt - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    tr_feat = [feats(dt["S"], dt["U"], dt["S2"]) for dt in train_data]

    nH = cfg["n_holdout"]
    test_bufs = []
    for dt in test_data:
        S, U, S2 = dt["S"], dt["U"], dt["S2"]; ac = dt["any_contact"]
        Yd = (S2 - S).astype(np.float32)
        puck_moving = np.abs(Yd[:, PUCK_VEL]).max(1) > cfg["puck_move_thresh"]
        n = len(S); hi = np.arange(n - nH, n); ai = np.arange(0, min(cfg["n_adapt_pool"], n - nH))
        test_bufs.append({
            "adapt_ctx": feats(S[ai], U[ai], S2[ai])[2],
            "hold_raw": (S[hi], U[hi], S2[hi]),
            "hold_su_n": feats(S[hi], U[hi], S2[hi])[0],
            "hold_ff": ~ac[hi],
            "hold_slide": (~ac[hi]) & puck_moving[hi],
        })

    pe = np.random.default_rng(cfg["seed"] + 11)
    B = cfg["n_eval_plan"]; ah = cfg["dgp_base"]["arena_half"]; sr = cfg["start_range"]
    ppos = pe.uniform(-sr, sr, (B, 2)); qpos_puck = pe.uniform(-0.6 * ah, 0.6 * ah, (B, 2))
    pvel = pe.normal(0, cfg["v0_std"], (B, 2)); qvel_puck = np.zeros((B, 2))
    ev_starts = np.concatenate([ppos, qpos_puck, pvel, qvel_puck], 1).astype(np.float32)
    ev_goals = pe.uniform(-cfg["goal_range"], cfg["goal_range"], (B, 2)).astype(np.float32)
    rand_dist = float(np.median(np.linalg.norm(ev_starts[:, PUSHER_POS] - ev_goals, axis=1)))
    test_envs = [PusherEnv(task_dgp(p), with_puck=True) for p in test_p]
    print(f"[1] pooled train transitions={len(allS)}  plan random-dist median={rand_dist:.3f}", flush=True)

    # ===================================================================== #
    # 2. encoder E (DeepSets) + conditional FM f(s,u,z) (verbatim from 4e)
    # ===================================================================== #
    eh = cfg["enc_hidden"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(18, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, d))

        def forward(self, ctx):
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm():
        lyr = [nn.Linear(10 + d, hidden), nn.SiLU()]
        for _ in range(cfg["fm_layers"] - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(hidden, 8)])).to(device)

    huber = nn.HuberLoss(delta=1.0, reduction="none")
    Bt, Pp = min(cfg["task_batch"], len(tr_feat)), cfg["pred_P"]

    def train_chunk(enc, cfm, opt, w_vec, steps, rng):
        """Advance the LIVE FM `steps` minibatch steps under per-dim weight `w_vec`. This is the
        FAST inner loop; called repeatedly on the SAME enc/cfm/opt (the FM never restarts)."""
        w = torch.tensor(np.asarray(w_vec, np.float32), device=device)
        enc.train(); cfm.train()
        for _ in range(steps):
            tids = rng.choice(len(tr_feat), size=Bt, replace=False)
            ctx = torch.empty(Bt, N_ctx, 18, device=device)
            psu = torch.empty(Bt, Pp, 10, device=device); pdy = torch.empty(Bt, Pp, 8, device=device)
            for j, tid in enumerate(tids):
                su_n, dy_n, cn = tr_feat[tid]; n = su_n.shape[0]
                ci = torch.tensor(rng.integers(0, n, size=N_ctx), device=device)
                pi = torch.tensor(rng.integers(0, n, size=Pp), device=device)
                ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
            z = enc(ctx); zc = z.unsqueeze(1).expand(-1, Pp, -1)
            pred = cfm(torch.cat([psu, zc], 2))
            loss = (huber(pred, pdy) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
            opt.step()
        enc.eval(); cfm.eval()

    # ---- per-dim R^2 + CEM-MPC control (verbatim from 4e) ----
    def r2(pred, yt, dims):
        p, t = pred[:, dims], yt[:, dims]
        ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    def eval_perdim(enc, cfm):
        rows = {k: [] for k in ["pusher_vel_ff", "puck_vel_slide", "all_ff"]}
        for tb in test_bufs:
            S, U, S2 = tb["hold_raw"]
            with torch.no_grad():
                z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
                su_n = tb["hold_su_n"]; zc = z.unsqueeze(0).expand(su_n.shape[0], -1)
                pred = (cfm(torch.cat([su_n, zc], 1)) * sy + my).cpu().numpy()
            yt = (S2 - S).astype(np.float32); ff, sl = tb["hold_ff"], tb["hold_slide"]
            if ff.sum() >= 20:
                rows["pusher_vel_ff"].append(r2(pred[ff], yt[ff], PUSHER_VEL))
                rows["all_ff"].append(r2(pred[ff], yt[ff], list(range(8))))
            if sl.sum() >= 20:
                rows["puck_vel_slide"].append(r2(pred[sl], yt[sl], PUCK_VEL))
        return {k: (float(np.median(v)) if v else float("nan")) for k, v in rows.items()}

    Hep, Hp, re_ = cfg["plan_H"], cfg["plan_Hp"], cfg["replan_every"]
    Kc, n_elite, vel_pen = cfg["k_shoot"], cfg["cem_elite"], cfg["vel_pen"]

    def mpc_action(cfm, z, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, Hp, 2), np.float32); sig = np.full((Bn, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        zc = z.unsqueeze(0).expand(Bn * Kc, -1)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, Hp, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (cfm(torch.cat([(x - mx) / sx, zc], 1)) * sy + my)
                    cost = cost + (s[:, PUSHER_POS] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, PUSHER_VEL].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_plan(cfm, z, env, seed):
        rng = np.random.default_rng(seed); states = ev_starts.copy(); plan = None
        for step in range(Hep):
            if step % re_ == 0:
                plan = mpc_action(cfm, z, states, ev_goals, rng)
            acts = plan[:, step % re_, :]
            for b in range(states.shape[0]):
                env.set_state(states[b, 0:4], states[b, 4:8]); s2, _ = env.step(acts[b], fs); states[b] = s2
        return float(np.median(np.linalg.norm(states[:, PUSHER_POS] - ev_goals, axis=1)))

    def control_cost(enc, cfm):
        dists = []
        for ti, (tb, env) in enumerate(zip(test_bufs, test_envs)):
            with torch.no_grad():
                z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
            dists.append(eval_plan(cfm, z, env, cfg["seed"] + 300 + ti))
        return float(np.median(dists))

    def w_from_puck(w_puck):
        w = np.ones(8, np.float32)
        for i in PUCK:
            w[i] = w_puck
        return w

    # ===================================================================== #
    # 3. run one arm: ONE live FM, continuous training, slow reward-driven w-loop
    # ===================================================================== #
    import copy

    def run_arm(arm):
        fixed_w = _parse_fixed_w(arm)
        torch.manual_seed(cfg["seed"] + 7)               # SAME init across arms
        rng = np.random.default_rng(cfg["seed"] + 7)     # SAME data order across arms
        enc, cfm = Encoder().to(device), build_cfm()
        opt = torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()), lr=cfg["meta_lr"])

        def snapshot():
            return (copy.deepcopy(enc.state_dict()), copy.deepcopy(cfm.state_dict()),
                    copy.deepcopy(opt.state_dict()))

        def restore(snap):
            enc.load_state_dict(snap[0]); cfm.load_state_dict(snap[1]); opt.load_state_dict(snap[2])

        theta = math.log(cfg["w_init"] / (1 - cfg["w_init"]))    # online: w_puck=sigmoid(theta)
        base = None; run_sq = None; warmup = cfg["outer_warmup"]
        cur_eps = 0.0
        hist = []

        # WARMUP: `single` mode warms to near-CONVERGENCE (kills the trend confound but the
        # value-shaping signal is a pre-convergence transient, so it also washes the signal out);
        # `fork` mode warms only MODESTLY and stays PRE-convergence, where the signal lives — the
        # antithetic forks (below) cancel the trend without needing convergence. Fixed arms warm
        # at their own w (a fair, fully-at-w reference for the online live FM).
        fork = (cfg["outer_mode"] == "fork")
        w_start = cfg["w_init"] if arm == "online" else fixed_w
        if cfg["warmup_steps"] > 0:
            train_chunk(enc, cfm, opt, w_from_puck(w_start), cfg["warmup_steps"], rng)
            print(f"    [{arm}] warmed {cfg['warmup_steps']} steps at w_puck={w_start:.2f} "
                  f"-> cost={control_cost(enc, cfm):.4f}", flush=True)

        for epoch in range(cfg["n_epochs"]):
            # ----- fixed arms: just progress the live FM at their fixed w -----
            if arm != "online":
                w_puck = fixed_w
                train_chunk(enc, cfm, opt, w_from_puck(w_puck), cfg["outer_every"], rng)
                cost = control_cost(enc, cfm); pd = eval_perdim(enc, cfm)
                hist.append({"epoch": epoch, "w_puck": float(w_puck), "cost": cost,
                             "pusher_vel_ff": pd["pusher_vel_ff"], "puck_vel_slide": pd["puck_vel_slide"],
                             "theta": 0.0})
                print(f"  [{arm:6s} e{epoch:2d}] w_puck={w_puck:.3f} cost={cost:.4f} "
                      f"pV={pd['pusher_vel_ff']:.3f} kV={pd['puck_vel_slide']:.3f}", flush=True)
                continue

            # ----- online, ANTITHETIC-FORK mode (past warmup epochs) -----
            if fork and epoch >= warmup:
                # Probe w+eps and w-eps from the SAME live snapshot with COMMON RANDOM NUMBERS
                # (identical minibatches) -> the training-progress trend + minibatch noise cancel,
                # so cost(wB)-cost(wA) is PURELY the value-shaping effect. This captures the
                # pre-convergence signal the single-mode REINFORCE needs convergence to de-trend.
                cur_eps = cfg["outer_sigma"] * float(rng.standard_normal())
                wA = sigmoid(max(-6.0, min(6.0, theta + cur_eps)))
                wB = sigmoid(max(-6.0, min(6.0, theta - cur_eps)))
                snap = snapshot(); fseed = cfg["seed"] + 9000 + epoch
                train_chunk(enc, cfm, opt, w_from_puck(wA), cfg["probe_steps"], np.random.default_rng(fseed))
                costA = control_cost(enc, cfm); restore(snap)
                train_chunk(enc, cfm, opt, w_from_puck(wB), cfg["probe_steps"], np.random.default_rng(fseed))
                costB = control_cost(enc, cfm); restore(snap)
                diff = costB - costA                     # >0 (eps>0): the lower-w side is better -> raise? no:
                # eps>0 => wA=w(theta+eps) is the HIGHER-w probe. diff=costB-costA>0 => costA<costB =>
                # higher w better => raise theta. Antithetic ES: theta += alpha * eps * (R_A-R_B)/(2 sigma^2),
                # R=-cost => (R_A-R_B)=(costB-costA)=diff. Normalize diff for a stable step size.
                if run_sq is None:
                    run_sq = diff ** 2 if diff != 0 else 1e-9
                else:
                    run_sq = cfg["outer_rho"] * run_sq + (1 - cfg["outer_rho"]) * (diff ** 2)
                adv_n = max(-3.0, min(3.0, diff / (math.sqrt(run_sq) + 1e-9)))
                grad = cur_eps * adv_n / (2 * cfg["outer_sigma"])
                theta = max(-6.0, min(6.0, theta + cfg["outer_alpha"] * grad))
                # live progression at the updated (unperturbed) w
                w_live = sigmoid(theta)
                train_chunk(enc, cfm, opt, w_from_puck(w_live), cfg["outer_every"], rng)
                cost = control_cost(enc, cfm); pd = eval_perdim(enc, cfm)
                hist.append({"epoch": epoch, "w_puck": float(w_live), "cost": cost,
                             "pusher_vel_ff": pd["pusher_vel_ff"], "puck_vel_slide": pd["puck_vel_slide"],
                             "theta": float(theta), "w_theta": float(w_live),
                             "costA": costA, "costB": costB, "diff": float(diff)})
                print(f"  [online e{epoch:2d}] w_live={w_live:.3f} theta={theta:+.2f} cost={cost:.4f} "
                      f"cA(w{wA:.2f})={costA:.4f} cB(w{wB:.2f})={costB:.4f} diff={diff:+.4f} "
                      f"pV={pd['pusher_vel_ff']:.3f} kV={pd['puck_vel_slide']:.3f}", flush=True)
                continue

            # ----- online, SINGLE-mode REINFORCE (or fork warmup epochs: just progress at w_init) -----
            if fork:
                w_puck = sigmoid(theta)                  # warmup epoch: progress at neutral w_init
            else:
                cur_eps = 0.0 if epoch < warmup else cfg["outer_sigma"] * float(rng.standard_normal())
                w_puck = sigmoid(max(-6.0, min(6.0, theta + cur_eps)))
            train_chunk(enc, cfm, opt, w_from_puck(w_puck), cfg["outer_every"], rng)
            cost = control_cost(enc, cfm); pd = eval_perdim(enc, cfm)
            rec = {"epoch": epoch, "w_puck": float(w_puck), "cost": cost,
                   "pusher_vel_ff": pd["pusher_vel_ff"], "puck_vel_slide": pd["puck_vel_slide"],
                   "theta": float(theta), "w_theta": float(sigmoid(theta))}
            if not fork:
                R = -cost
                adv = 0.0 if base is None else (R - base)
                base = R if base is None else cfg["outer_rho"] * base + (1 - cfg["outer_rho"]) * R
                if adv != 0.0:
                    run_sq = adv ** 2 if run_sq is None else \
                        cfg["outer_rho"] * run_sq + (1 - cfg["outer_rho"]) * (adv ** 2)
                if epoch >= warmup and run_sq is not None:
                    adv_n = max(-3.0, min(3.0, adv / (math.sqrt(run_sq) + 1e-6)))
                    grad = adv_n * (cur_eps / (cfg["outer_sigma"] ** 2))
                    theta = max(-6.0, min(6.0, theta + cfg["outer_alpha"] * grad))
                rec["adv"] = float(adv); rec["w_theta"] = float(sigmoid(theta))
            print(f"  [online e{epoch:2d}] w_puck={w_puck:.3f} theta={theta:+.2f} cost={cost:.4f} "
                  f"pV={pd['pusher_vel_ff']:.3f} kV={pd['puck_vel_slide']:.3f}", flush=True)
            hist.append(rec)

        tail = max(1, cfg["n_epochs"] // 3)
        final_cost = float(np.mean([h["cost"] for h in hist[-tail:]]))
        final_w = float(np.mean([h["w_puck"] for h in hist[-tail:]]))
        return {"history": hist, "final_cost": final_cost, "final_w_puck": final_w}

    results = {}
    for arm in arms:
        print(f"\n[arm] {arm}", flush=True)
        results[arm] = run_arm(arm)

    print("\n[summary] final control cost (lower=better) | final w_puck", flush=True)
    for a in arms:
        print(f"  {a:6s} cost={results[a]['final_cost']:.4f}  w_puck={results[a]['final_w_puck']:.3f}", flush=True)
    if "online" in arms:
        for base_a in ["w1.0", "w0.0", "w0.5"]:
            if base_a in arms:
                print(f"  GAP online−{base_a}: control "
                      f"{results[base_a]['final_cost'] - results['online']['final_cost']:+.4f} "
                      f"(>0 = online better)", flush=True)

    out = {"config": cfg, "field_pusher_amp": cfg["field_pusher_amp"],
           "regime": "capacity_competition" if cfg["field_pusher_amp"] > 0 else "easy_pusher",
           "train_params": train_p.tolist(), "test_params": test_p.tolist(),
           "plan_random_dist": rand_dist, "results": results}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "meta_value_online", cfg["tag"])
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

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"online": "#e8590c", "w1.0": "#d1603d", "w0.5": "#9c36b5", "w0.0": "#2f9e44"}
    arms = list(R["results"].keys())
    reg = R["regime"]; reg_lbl = "capacity competition" if reg == "capacity_competition" else "easy pusher"
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    def col(a):
        return COL.get(a) or "#868e96"

    # ---- fig1: the learned w_puck trajectory (headline) ----
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    if "online" in R["results"]:
        h = R["results"]["online"]["history"]
        ep = [x["epoch"] for x in h]
        ax.plot(ep, [x.get("w_theta", x["w_puck"]) for x in h], "o-", color=COL["online"], lw=2.4,
                label="online: w_puck=σ(θ) (learned)")
        ax.plot(ep, [x["w_puck"] for x in h], "x", color=COL["online"], alpha=0.4, ms=6,
                label="online: w_puck sampled")
    for a in arms:
        fw = _parse_fixed_w(a)
        if fw is not None:
            ax.axhline(fw, color=col(a), ls="--", lw=1.2, alpha=0.8, label=f"fixed {a}")
    ax.set_xlabel("outer-loop epoch"); ax.set_ylabel("puck-weight w_puck  (1=veridical, 0=drop the puck)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"reward self-tunes FM capacity allocation [{reg_lbl}]\n"
                 f"{'competition → w_puck should DROP' if reg=='capacity_competition' else 'easy → w_puck should stay flat/wander'}",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    figs["fig1_learned_w.png"] = _save(fig)

    # ---- fig2: control cost + the re-allocation (per-dim R^2) over epochs ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for a in arms:
        h = R["results"][a]["history"]; ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["cost"] for x in h], "o-", ms=3, color=col(a), label=a)
    axes[0].axhline(R["plan_random_dist"], color="#bbb", ls=":", lw=1, label="random floor")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("median goal-dist (lower=better)")
    axes[0].set_title(f"control competence [{reg_lbl}]"); axes[0].legend(fontsize=7)
    if "online" in R["results"]:
        h = R["results"]["online"]["history"]; ep = [x["epoch"] for x in h]
        axes[1].plot(ep, [x["pusher_vel_ff"] for x in h], "o-", color="#2f9e44", lw=2, label="pusher-vel R² (value-relevant)")
        axes[1].plot(ep, [max(x["puck_vel_slide"], -1.0) for x in h], "d-", color="#d1603d", lw=2, label="puck-vel R² (value-irrelevant)")
        ax2 = axes[1].twinx()
        ax2.plot(ep, [x.get("w_theta", x["w_puck"]) for x in h], "-", color="#e8590c", lw=1.4, alpha=0.6)
        ax2.set_ylabel("w_puck (orange)", color="#e8590c"); ax2.set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("per-dim R²")
    axes[1].set_title("the re-allocation, in one LIVE FM"); axes[1].legend(fontsize=7, loc="best")
    figs["fig2_control_realloc.png"] = _save(fig)

    # ---- fig3: order-parameter bars ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(range(len(arms)), [R["results"][a]["final_cost"] for a in arms], color=[col(a) for a in arms])
    axes[0].axhline(R["plan_random_dist"], color="#bbb", ls=":", lw=1)
    axes[0].set_xticks(range(len(arms))); axes[0].set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("final control dist (lower=better)"); axes[0].set_title("control")
    axes[1].bar(range(len(arms)), [R["results"][a]["final_w_puck"] for a in arms], color=[col(a) for a in arms])
    axes[1].axhline(1.0, color="#888", ls="--", lw=1, label="veridical")
    axes[1].set_xticks(range(len(arms))); axes[1].set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("final w_puck"); axes[1].set_title("learned allocation"); axes[1].legend(fontsize=8)
    figs["fig3_orderparams.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_value_online(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "",
    # env / family
    frame_skip: int = 3,
    conflict: float = 1.5708,
    arena_half: float = 0.9,
    gear: float = 10.0,
    puck_mass: float = 0.5,
    pusher_r: float = 0.12,
    puck_r: float = 0.12,
    field_amp: float = 1.3,
    field_pusher_amp: float = 2.0,       # >0 = CAPACITY COMPETITION; 0 = easy null
    field_central: float = 0.5,
    # tasks / collection
    n_train_tasks: int = 12,
    n_test_tasks: int = 6,
    n_episodes: int = 80,
    ep_len: int = 200,
    sigma: float = 0.7,
    theta: float = 0.15,
    seek_gain: float = 0.9,
    n_adapt_pool: int = 512,
    n_holdout: int = 3000,
    puck_move_thresh: float = 0.02,
    n_context: int = 128,
    # encoder + conditional FM (the 4d capacity sweet spot)
    hidden: int = 64,
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_layers: int = 2,
    meta_lr: float = 1e-3,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    # ONLINE two-timescale loop
    outer_mode: str = "fork",            # fork = antithetic pre-convergence (captures the signal);
                                         # single = REINFORCE (needs convergence, washes signal out)
    warmup_steps: int = 300,             # fork: MODEST (stay pre-convergence). single: raise to ~1500.
    probe_steps: int = 150,              # fork: SGD steps per antithetic branch
    n_epochs: int = 16,
    outer_every: int = 180,              # live-FM SGD steps per outer epoch (the timescale ratio)
    outer_warmup: int = 1,
    outer_alpha: float = 0.3,
    outer_sigma: float = 0.7,
    outer_rho: float = 0.8,
    w_init: float = 0.5,                 # neutral start (can go up toward veridical or down toward drop)
    # CEM-MPC control reward
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    plan_h: int = 100,
    plan_hp: int = 16,
    replan_every: int = 8,
    k_shoot: int = 192,
    cem_iters: int = 3,
    cem_elite: int = 24,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.3,
    n_eval_plan: int = 24,
):
    import os

    arm_list = [a for a in (arms.split(",") if arms else DEFAULT_ARMS) if a]
    if quick:
        arm_list = arms.split(",") if arms else ["online", "w1.0", "w0.0"]
        n_train_tasks, n_test_tasks = 6, 3
        n_episodes, ep_len = 40, 140
        enc_hidden = 64; n_holdout = 2200
        warmup_steps = 200; probe_steps = 80; n_epochs = 6; outer_every = 100; outer_warmup = 1
        plan_h, k_shoot, n_eval_plan = 60, 128, 8
        tag = tag or "smoke"
    tag = tag or "default"

    puck_field = dict(amp=field_amp, pusher_amp=field_pusher_amp, central=field_central)
    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, frame_skip=frame_skip, conflict=conflict,
        field_pusher_amp=field_pusher_amp,
        dgp_base=dict(arena_half=arena_half, gear=gear, puck_mass=puck_mass,
                      pusher_r=pusher_r, puck_r=puck_r, puck_field=puck_field),
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks, n_episodes=n_episodes, ep_len=ep_len,
        sigma=sigma, theta=theta, seek_gain=seek_gain, n_adapt_pool=n_adapt_pool, n_holdout=n_holdout,
        puck_move_thresh=puck_move_thresh, n_context=n_context, hidden=hidden, latent_dim=latent_dim,
        enc_hidden=enc_hidden, fm_layers=fm_layers, meta_lr=meta_lr, task_batch=task_batch,
        pred_P=pred_p, grad_clip=grad_clip, outer_mode=outer_mode, probe_steps=probe_steps,
        warmup_steps=warmup_steps, n_epochs=n_epochs, outer_every=outer_every, outer_warmup=outer_warmup,
        outer_alpha=outer_alpha, outer_sigma=outer_sigma, outer_rho=outer_rho, w_init=w_init,
        goal_range=goal_range, start_range=start_range, v0_std=v0_std, plan_H=plan_h, plan_Hp=plan_hp,
        replan_every=replan_every, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, n_eval_plan=n_eval_plan,
    )
    out = run_meta_value_online.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_value_online_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
