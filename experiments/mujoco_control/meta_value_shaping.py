"""Cut #4d — Value-shaping x meta-conditioning: two separable capacity levers (disc-4 + fusion).

Program: `ideas/two_timescale_value_loop.md` discriminator 4 ("does a value/OUTER loop
CAUSE the FM to become value-shaped") and the FM<->value interface ("one efferent gain:
value-shaping = capacity RE-ALLOCATION, a priority field over prediction targets").
Parents: `value_shaping.py` (STATIONARY value-shaping via a per-dim weight; single task,
no meta) and `meta_context.py` / `META_ADAPT_README.md` (the collapse->gap under
actuator-rotation conflict; the dissectible context latent `z` that decodes phi).

    THE FUSION. value_shaping showed a value-relevance weight re-allocates a small FM off
    the value-irrelevant puck -- but STATIONARY (one task, no distribution). meta_context
    showed a context latent `z` holds a per-task conflicted map -- but on the VERIDICAL
    objective (predict everything). This cut crosses them: on ONE substrate (puck +
    push_rot conflict, capacity-limited FM) run the 2x2

                        VERIDICAL (match all 8 dims)   vs   VALUE-SHAPED (match pusher dims)
        CONTEXT z=E(N)          .                                  .
        POOLED   z=0            .                                  .

    and ask how the two levers interact. The value-relevance weight is DERIVED from the
    goal-reaching value V(s) = -||pusher_pos - g|| - beta*||pusher_vel|| (arrive AND stop,
    = the CEM cost in meta_adapt): its state-support is {pusher pos, pusher vel}, zero on
    the puck. So "value-shaped" = match the PUSHER dims [0,1,4,5], drop the PUCK dims
    [2,3,6,7] -- not a tuned lambda, the support of the reaching value.

    Predictions (Stage 1 = FM-side; CEM planning = Stage 2, staged as value_shaping
    staged its planner):
      1. disc-4 / capacity: value-shaped DROPS the puck (puck-vel R^2 down) that veridical
         KEEPS, and at a small binding FM the freed capacity lifts the value-relevant
         pusher-vel R^2 above veridical's.
      2. fusion / meta-necessity: the conflict lever is ORTHOGONAL to value-shaping --
         POOLED collapses on the phi-conflicted pusher map for BOTH objectives (value-
         shaping does not rescue pooling); only CONTEXT holds it. At a small FM you need
         BOTH: z to hold the per-phi map, value-shaping to free the capacity to do so.
      3. system-ID: z->phi probe holds under the value-shaped objective too.

Substrate/machinery = meta_context (encoder E over 18-dim (s,u,ds) context, conditional
FM f(s,u,z)->ds(8), push_rot actuator family) + value_shaping (puck force field = value-
irrelevant capacity sink; per-dim value weight). Only the per-dim loss weight and the
context mode differ across arms -- the arity_torque/value_shaping discipline.

Run:
    cd experiments/
    modal run mujoco_control/meta_value_shaping.py::meta_value_shaping --quick                 # smoke
    modal run --detach mujoco_control/meta_value_shaping.py::run_meta_value_shaping            # (full; via entrypoint)
    modal run mujoco_control/meta_value_shaping.py::meta_value_shaping --tag vs_p2 --conflict 1.5708
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (pusher_env.STATE_LABELS)
PUSHER_POS = [0, 1]
PUCK_POS = [2, 3]
PUSHER_VEL = [4, 5]
PUCK_VEL = [6, 7]
PUSHER = [0, 1, 4, 5]     # value-relevant: support of V = -||pos-g|| - beta*||vel||
PUCK = [2, 3, 6, 7]       # value-irrelevant: the distractor's own motion


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_meta_value_shaping(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import collect_transitions, PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    d = cfg["latent_dim"]
    caps = cfg["caps"]
    N_ctx = cfg["n_context"]
    puck_w = cfg["shaped_puck_weight"]          # value-shaped puck weight (0 = drop the puck)

    # ---- task family: input-coupled command rotation phi in [-Phi, Phi] ------ #
    Phi = cfg["conflict"]
    n_tr, n_te = cfg["n_train_tasks"], cfg["n_test_tasks"]
    all_p = np.sort(np.linspace(-Phi, Phi, n_tr + n_te))
    te_idx = np.unique(np.linspace(1, n_tr + n_te - 2, n_te).round().astype(int))
    tr_idx = np.array([i for i in range(n_tr + n_te) if i not in te_idx])
    train_p = all_p[tr_idx].astype(np.float64)
    test_p = all_p[te_idx].astype(np.float64)
    print(f"[family] actuator phi | train {np.round(train_p,3)}\n[family] test {np.round(test_p,3)}",
          flush=True)

    def task_dgp(p):
        dd = dict(cfg["dgp_base"])
        dd["push_rot"] = float(p)               # rotate the pusher's command->motion map
        return dd

    def collect(dgp, seed):
        return collect_transitions(
            dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
            frame_skip=fs, seed=seed, sigma=cfg["sigma"], theta=cfg["theta"],
            seek_gain=cfg["seek_gain"])

    # ===================================================================== #
    # 1. collect per-task buffers + family normalization
    # ===================================================================== #
    print("\n[1] collecting per-task buffers ...", flush=True)
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
    print(f"[1] pooled train transitions={len(allS)}  "
          f"any_contact={np.concatenate([dt['any_contact'] for dt in train_data]).mean():.3f}  "
          f"puck_contact={np.concatenate([dt['puck_contact'] for dt in train_data]).mean():.3f}",
          flush=True)

    def feats(S, U, S2):
        """normalized (su_n [.,10], dy_n [.,8], ctx_n [.,18]) tensors on device."""
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (Xt - mx) / sx; dy_n = (Yt - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    tr_feat = [feats(dt["S"], dt["U"], dt["S2"]) for dt in train_data]

    # per-test-task: adapt (context source) / hold (eval) + regime masks on hold
    nH = cfg["n_holdout"]
    test_bufs = []
    for dt in test_data:
        S, U, S2 = dt["S"], dt["U"], dt["S2"]
        ac = dt["any_contact"]
        Yd = (S2 - S).astype(np.float32)
        puck_moving = np.abs(Yd[:, PUCK_VEL]).max(1) > cfg["puck_move_thresh"]
        n = len(S); hi = np.arange(n - nH, n); ai = np.arange(0, min(cfg["n_adapt_pool"], n - nH))
        test_bufs.append({
            "adapt_ctx": feats(S[ai], U[ai], S2[ai])[2],
            "hold_raw": (S[hi], U[hi], S2[hi]),
            "hold_su_n": feats(S[hi], U[hi], S2[hi])[0],
            "hold_ff": ~ac[hi],                          # free-flight (clean fidelity regime)
            "hold_slide": (~ac[hi]) & puck_moving[hi],   # puck sliding, no contact (reducible)
        })

    # planning eval set (shared starts/goals/puck across ALL arms+tasks -> controlled).
    # Start state s = [pusher_pos, puck_pos, pusher_vel, puck_vel] (8-dim).
    do_plan = cfg["plan"]
    if do_plan:
        pe = np.random.default_rng(cfg["seed"] + 11)
        B = cfg["n_eval_plan"]; ah = cfg["dgp_base"]["arena_half"]; sr = cfg["start_range"]
        ppos = pe.uniform(-sr, sr, (B, 2)); qpos_puck = pe.uniform(-0.6 * ah, 0.6 * ah, (B, 2))
        pvel = pe.normal(0, cfg["v0_std"], (B, 2)); qvel_puck = np.zeros((B, 2))
        ev_starts = np.concatenate([ppos, qpos_puck, pvel, qvel_puck], 1).astype(np.float32)
        ev_goals = pe.uniform(-cfg["goal_range"], cfg["goal_range"], (B, 2)).astype(np.float32)
        rand_dist = float(np.median(np.linalg.norm(ev_starts[:, PUSHER_POS] - ev_goals, axis=1)))
        test_envs = [PusherEnv(task_dgp(p), with_puck=True) for p in test_p]
        print(f"[plan] eval set B={B}  random start-dist median={rand_dist:.3f}", flush=True)

    # ===================================================================== #
    # 2. encoder E (DeepSets over 18-dim ctx) + conditional FM f(s,u,z)
    # ===================================================================== #
    eh = cfg["enc_hidden"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(18, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, d))

        def forward(self, ctx):                          # (B,K,18) -> (B,d)
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm(hidden, layers):
        lyr = [nn.Linear(10 + d, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(hidden, 8)])).to(device)

    huber = nn.HuberLoss(delta=1.0, reduction="none")

    def value_weight(shaped):
        w = torch.ones(8, device=device)
        if shaped:
            for i in PUCK:
                w[i] = puck_w                            # drop the value-irrelevant puck
        return w

    # ===================================================================== #
    # 3. META-TRAIN E + f for each (objective, capacity). Only the per-dim
    #    loss weight differs (value_shaping discipline); one-step (s,u,z)->ds.
    # ===================================================================== #
    def meta_train(hidden, shaped):
        rng = np.random.default_rng(cfg["seed"] + 7)
        enc = Encoder().to(device)
        cfm = build_cfm(hidden, cfg["fm_layers"])
        opt = torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()), lr=cfg["meta_lr"])
        w = value_weight(shaped)
        Bt, P = min(cfg["task_batch"], len(tr_feat)), cfg["pred_P"]
        enc.train(); cfm.train()
        for step in range(cfg["meta_steps"]):
            tids = rng.choice(len(tr_feat), size=Bt, replace=False)
            ctx = torch.empty(Bt, N_ctx, 18, device=device)
            psu = torch.empty(Bt, P, 10, device=device)
            pdy = torch.empty(Bt, P, 8, device=device)
            for j, tid in enumerate(tids):
                su_n, dy_n, cn = tr_feat[tid]; n = su_n.shape[0]
                ci = torch.tensor(rng.integers(0, n, size=N_ctx), device=device)
                pi = torch.tensor(rng.integers(0, n, size=P), device=device)
                ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
            z = enc(ctx)                                            # (Bt,d)
            zc = z.unsqueeze(1).expand(-1, P, -1)
            pred = cfm(torch.cat([psu, zc], 2))                     # (Bt,P,8)
            loss = (huber(pred, pdy) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
            opt.step()
            if (step + 1) % max(1, cfg["meta_steps"] // 5) == 0:
                print(f"    [{'shaped' if shaped else 'verid':6s} h={hidden:4d}] "
                      f"step {step+1}/{cfg['meta_steps']} loss={loss.item():.4f}", flush=True)
        enc.eval(); cfm.eval()
        return enc, cfm

    # ===================================================================== #
    # 4. per-dim R^2 eval (held-out test tasks, median), context + z0 modes
    # ===================================================================== #
    def r2(pred, yt, dims):
        p, t = pred[:, dims], yt[:, dims]
        ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    def eval_perdim(enc, cfm, use_context):
        rows = {k: [] for k in ["pusher_pos_ff", "pusher_vel_ff", "puck_vel_slide", "all_ff"]}
        zero_z = torch.zeros(d, device=device)
        for tb in test_bufs:
            S, U, S2 = tb["hold_raw"]
            if use_context:
                with torch.no_grad():
                    z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
            else:
                z = zero_z
            with torch.no_grad():
                su_n = tb["hold_su_n"]; zc = z.unsqueeze(0).expand(su_n.shape[0], -1)
                pred = (cfm(torch.cat([su_n, zc], 1)) * sy + my).cpu().numpy()
            yt = (S2 - S).astype(np.float32)
            ff, sl = tb["hold_ff"], tb["hold_slide"]
            if ff.sum() >= 20:
                rows["pusher_pos_ff"].append(r2(pred[ff], yt[ff], PUSHER_POS))
                rows["pusher_vel_ff"].append(r2(pred[ff], yt[ff], PUSHER_VEL))
                rows["all_ff"].append(r2(pred[ff], yt[ff], list(range(8))))
            if sl.sum() >= 20:
                rows["puck_vel_slide"].append(r2(pred[sl], yt[sl], PUCK_VEL))
        return {k: (float(np.median(v)) if v else float("nan")) for k, v in rows.items()}

    # ===================================================================== #
    # 4b. CEM-MPC goal-reaching (fixed value = pusher-goal dist + terminal
    #     pusher-vel penalty; cost IGNORES the puck. Only the FM/z differ.
    #     The planner COMMITS to `replan_every` open-loop steps so the world
    #     model is load-bearing (Cut #3's lesson).
    # ===================================================================== #
    Hep, Hp, re_ = cfg["plan_H"], cfg["plan_Hp"], cfg["replan_every"]
    K, n_elite, vel_pen = cfg["k_shoot"], cfg["cem_elite"], cfg["vel_pen"]

    def mpc_action_puck(cfm, z, states, goals, rng):
        B = states.shape[0]
        mu = np.zeros((B, Hp, 2), np.float32)
        sig = np.full((B, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(K, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(K, 0)
        zc = z.unsqueeze(0).expand(B * K, -1)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((B, K, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(B * K, Hp, 2), device=device)
                cost = torch.zeros(B * K, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (cfm(torch.cat([(x - mx) / sx, zc], 1)) * sy + my)
                    cost = cost + (s[:, PUSHER_POS] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, PUSHER_VEL].norm(dim=1)
                idx = torch.topk(-cost.reshape(B, K), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_plan(cfm, z, env, seed):
        rng = np.random.default_rng(seed)
        states = ev_starts.copy(); plan = None
        for step in range(Hep):
            if step % re_ == 0:
                plan = mpc_action_puck(cfm, z, states, ev_goals, rng)
            acts = plan[:, step % re_, :]
            for b in range(states.shape[0]):
                env.set_state(states[b, 0:4], states[b, 4:8])
                s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, PUSHER_POS] - ev_goals, axis=1)
        return float(np.median(fd))

    def eval_plan_arm(enc, cfm, use_context):
        zero_z = torch.zeros(d, device=device)
        dists = []
        for ti, (tb, env) in enumerate(zip(test_bufs, test_envs)):
            if use_context:
                with torch.no_grad():
                    z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
            else:
                z = zero_z
            dists.append(eval_plan(cfm, z, env, cfg["seed"] + 300 + ti))
        return float(np.median(dists))

    # ===================================================================== #
    # 5. z->phi linear probe (system-ID)
    # ===================================================================== #
    def z_probe(enc):
        prng = np.random.default_rng(cfg["seed"] + 9)
        ndraw = cfg["probe_draws"]

        def gather(ctx_list, params):
            Z, Yp = [], []
            for tid, cn in enumerate(ctx_list):
                nrow = cn.shape[0]
                for _ in range(ndraw):
                    ci = torch.tensor(prng.integers(0, nrow, size=N_ctx), device=device)
                    with torch.no_grad():
                        Z.append(enc(cn[ci].unsqueeze(0))[0].cpu().numpy())
                    Yp.append(params[tid])
            return np.asarray(Z, np.float64), np.asarray(Yp, np.float64)

        Ztr, Ytr = gather([f[2] for f in tr_feat], train_p)
        Zte, Yte = gather([tb["adapt_ctx"] for tb in test_bufs], test_p)
        lam = 1e-3
        Xtr = np.concatenate([Ztr, np.ones((len(Ztr), 1))], 1)
        Wp = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        Xte = np.concatenate([Zte, np.ones((len(Zte), 1))], 1)
        Ypred = Xte @ Wp
        ssr = float(((Yte - Ypred) ** 2).sum()); sst = float(((Yte - Yte.mean()) ** 2).sum())
        per_task = [float(np.median(Ypred[ti * ndraw:(ti + 1) * ndraw])) for ti in range(len(test_p))]
        return {"r2": 1.0 - ssr / (sst + 1e-12), "true": test_p.tolist(), "pred": per_task,
                "draws_true": Yte.tolist(), "draws_pred": Ypred.tolist()}

    # ===================================================================== #
    # 6. per-task oracle ceiling (plain f(s,u), all dims) at ref capacity
    # ===================================================================== #
    def oracle_pusher_vel():
        vals = []
        for tb, dt in zip(test_bufs, test_data):
            S, U, S2 = dt["S"], dt["U"], dt["S2"]
            n = len(S); mid = np.arange(cfg["n_adapt_pool"], n - nH)
            h0 = cfg["ref_hidden"]
            lyr = [nn.Linear(10, h0), nn.SiLU()]
            for _ in range(cfg["fm_layers"] - 1):
                lyr += [nn.Linear(h0, h0), nn.SiLU()]
            fm = nn.Sequential(*(lyr + [nn.Linear(h0, 8)])).to(device)
            xn = (torch.tensor(np.concatenate([S[mid], U[mid]], 1), device=device) - mx) / sx
            yn = (torch.tensor((S2[mid] - S[mid]).astype(np.float32), device=device) - my) / sy
            o = torch.optim.Adam(fm.parameters(), lr=1e-3)
            nb = xn.shape[0]; bs = min(512, nb); fm.train()
            for _ in range(cfg["fm_epochs"]):
                perm = torch.randperm(nb, device=device)
                for i in range(0, nb, bs):
                    idx = perm[i:i + bs]
                    o.zero_grad(); huber(fm(xn[idx]), yn[idx]).mean().backward(); o.step()
            fm.eval()
            S_h, U_h, S2_h = tb["hold_raw"]; ff = tb["hold_ff"]
            with torch.no_grad():
                xh = (torch.tensor(np.concatenate([S_h, U_h], 1), device=device) - mx) / sx
                pred = (fm(xh) * sy + my).cpu().numpy()
            yt = (S2_h - S_h).astype(np.float32)
            if ff.sum() >= 20:
                vals.append(r2(pred[ff], yt[ff], PUSHER_VEL))
        return float(np.median(vals)) if vals else float("nan")

    print("\n[2] oracle pusher-vel ceiling ...", flush=True)
    oracle_pv = oracle_pusher_vel()
    print(f"    oracle pusher-vel free-flight R^2 (median) = {oracle_pv:.4f}", flush=True)

    # ===================================================================== #
    # 7. RUN: meta-train each (objective, cap); eval both modes
    # ===================================================================== #
    print("\n[3] meta-training arms (objective x capacity) ...", flush=True)
    sweep = {"caps": caps, "verid": {}, "shaped": {}}
    probes = {}
    plan_caps = set(cfg["plan_caps"]) if do_plan else set()
    for shaped, key in [(False, "verid"), (True, "shaped")]:
        for hidden in caps:
            enc, cfm = meta_train(hidden, shaped)
            ctx_r = eval_perdim(enc, cfm, use_context=True)
            z0_r = eval_perdim(enc, cfm, use_context=False)
            row = {"context": ctx_r, "z0": z0_r}
            if hidden in plan_caps:
                row["plan_context"] = eval_plan_arm(enc, cfm, True)
                row["plan_z0"] = eval_plan_arm(enc, cfm, False)
            sweep[key][str(hidden)] = row
            pmsg = (f" | plan ctx={row['plan_context']:.3f} z0={row['plan_z0']:.3f}"
                    if hidden in plan_caps else "")
            print(f"  [{key:6s} h={hidden:4d}] ctx: pusherV_ff={ctx_r['pusher_vel_ff']:.3f} "
                  f"puckV_slide={ctx_r['puck_vel_slide']:.3f} | z0: pusherV_ff={z0_r['pusher_vel_ff']:.3f}"
                  f"{pmsg}", flush=True)
            if hidden == cfg["ref_hidden"]:
                probes[key] = z_probe(enc)

    results = {
        "config": cfg, "family": "actuator", "conflict": Phi,
        "train_params": train_p.tolist(), "test_params": test_p.tolist(),
        "caps": caps, "ref_hidden": cfg["ref_hidden"], "oracle_pusher_vel": oracle_pv,
        "sweep": sweep, "probes": probes,
        "plan": do_plan, "plan_caps": sorted(plan_caps) if do_plan else [],
        "plan_random_dist": rand_dist if do_plan else None,
    }

    rh = str(cfg["ref_hidden"])
    vc, sc = sweep["verid"][rh]["context"], sweep["shaped"][rh]["context"]
    sz = sweep["shaped"][rh]["z0"]
    print(f"\n===== META-VALUE-SHAPING (Stage 1) SUMMARY (Phi={Phi:.3f}, ref h={rh}) =====", flush=True)
    print(f"disc-4  @h={rh} (context): puck-vel slide  verid={vc['puck_vel_slide']:.3f} -> "
          f"shaped={sc['puck_vel_slide']:.3f}  (shaped should DROP the puck)", flush=True)
    print(f"        pusher-vel ff  verid={vc['pusher_vel_ff']:.3f}  shaped={sc['pusher_vel_ff']:.3f}  "
          f"(shaped >= verid at small FM; oracle {oracle_pv:.3f})", flush=True)
    print(f"fusion  shaped pusher-vel ff: context={sc['pusher_vel_ff']:.3f}  z0(pooled)={sz['pusher_vel_ff']:.3f}  "
          f"(pooled collapses under conflict; value-shaping does not rescue it)", flush=True)
    print(f"sysID   z->phi probe R^2: verid={probes['verid']['r2']:.3f}  shaped={probes['shaped']['r2']:.3f}",
          flush=True)
    if do_plan:
        print(f"\nplanning (median goal-dist, lower=better; random floor {rand_dist:.3f}):", flush=True)
        for h in sorted(plan_caps):
            v = sweep["verid"][str(h)]; s = sweep["shaped"][str(h)]
            print(f"  h={h:4d} context: verid={v['plan_context']:.3f}  shaped={s['plan_context']:.3f}  "
                  f"(shaped<=verid = value-shaping helps control) | "
                  f"pooled: verid={v['plan_z0']:.3f} shaped={s['plan_z0']:.3f}", flush=True)

    figures = _make_figures(results)
    outdir = os.path.join(DATA_DIR, "meta_value_shaping", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_VER, C_SHP = "#3d6fd1", "#d1603d"
    figs, caps = {}, R["caps"]
    sweep, rh = R["sweep"], str(R["ref_hidden"])

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    FLOOR = -1.5   # clip display of collapsed/dropped R^2 so the axis stays readable

    def _series(obj, mode, metric, clip=False):
        v = [sweep[obj][str(h)][mode][metric] for h in caps]
        return [max(x, FLOOR) if (clip and x == x) else x for x in v]

    # ---- fig1: capacity frontier (context): pusher-vel + puck-vel, verid vs shaped ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, metric, title in [
        (axes[0], "pusher_vel_ff", "R² pusher velocity, free-flight (VALUE-RELEVANT)"),
        (axes[1], "puck_vel_slide", "R² puck velocity, slide (VALUE-IRRELEVANT)"),
    ]:
        ax.plot(caps, _series("verid", "context", metric, clip=True), "o-", color=C_VER, lw=2.1,
                label="veridical (match all dims)")
        ax.plot(caps, _series("shaped", "context", metric, clip=True), "o-", color=C_SHP, lw=2.1,
                label="value-shaped (match pusher dims)")
        if metric == "pusher_vel_ff" and not np.isnan(R["oracle_pusher_vel"]):
            ax.axhline(R["oracle_pusher_vel"], color="#2f9e44", lw=1.2, ls="--",
                       label=f"oracle {R['oracle_pusher_vel']:.3f}")
        if metric == "puck_vel_slide":
            ax.text(caps[0], FLOOR + 0.08, "↓ dropped (R²≪0)", fontsize=8, color=C_SHP)
        ax.set_xscale("log", base=2); ax.set_xticks(caps); ax.set_xticklabels(caps)
        ax.set_ylim(FLOOR - 0.1, 1.03)
        ax.set_xlabel("FM hidden width (capacity)"); ax.set_ylabel("R²")
        ax.set_title(title, fontsize=9.5); ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("disc-4: value-shaping drops the value-irrelevant puck at every capacity "
                 "(pusher is easy → freed capacity barely lifts it)", fontsize=10.5)
    figs["fig1_capacity_frontier.png"] = _save(fig)

    # ---- fig2: per-dim bars at ref capacity, verid vs shaped (context) ----
    groups = ["pusher_pos_ff", "pusher_vel_ff", "puck_vel_slide", "all_ff"]
    glabels = ["pusherPos (ff)", "pusherVel (ff)", "puckVel (slide)", "all (ff)"]
    vc, sc = sweep["verid"][rh]["context"], sweep["shaped"][rh]["context"]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(groups)); w = 0.4
    ax.bar(x - w / 2, [vc[g] for g in groups], w, color=C_VER, label="veridical")
    ax.bar(x + w / 2, [sc[g] for g in groups], w, color=C_SHP, label="value-shaped")
    ax.set_xticks(x); ax.set_xticklabels(glabels, rotation=15, ha="right")
    ax.axhline(0, color="#888", lw=0.8); ax.set_ylabel("R²")
    ax.set_title(f"Per-group fidelity at h={rh} (context): shaped drops the puck, keeps the pusher")
    ax.legend(fontsize=9)
    figs["fig2_perdim_bars.png"] = _save(fig)

    # ---- fig3: fusion — 2x2 pusher-vel vs capacity (context vs pooled, both objectives) ----
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(caps, _series("shaped", "context", "pusher_vel_ff", clip=True), "o-", color=C_SHP, lw=2.3,
            label="shaped + context")
    ax.plot(caps, _series("verid", "context", "pusher_vel_ff", clip=True), "o-", color=C_VER, lw=2.1,
            label="veridical + context")
    ax.plot(caps, _series("shaped", "z0", "pusher_vel_ff", clip=True), "s--", color=C_SHP, lw=1.9, alpha=0.7,
            label="shaped + pooled (z=0)")
    ax.plot(caps, _series("verid", "z0", "pusher_vel_ff", clip=True), "s--", color=C_VER, lw=1.7, alpha=0.6,
            label="veridical + pooled (z=0)")
    ax.set_xscale("log", base=2); ax.set_xticks(caps); ax.set_xticklabels(caps)
    ax.set_ylim(FLOOR - 0.1, 1.03)
    ax.set_xlabel("FM hidden width"); ax.set_ylabel("R² pusher velocity (free-flight)")
    ax.set_title("Fusion (2×2): pooling collapses under conflict for BOTH objectives\n"
                 "you need z (per-φ map) AND value-shaping (freed capacity)", fontsize=9.5)
    ax.legend(fontsize=8.2, loc="lower right")
    figs["fig3_fusion_2x2.png"] = _save(fig)

    # ---- fig4: z->phi probe scatter, verid vs shaped ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, obj, col in [(axs[0], "verid", C_VER), (axs[1], "shaped", C_SHP)]:
        pr = R["probes"][obj]
        dt = np.asarray(pr["draws_true"]); dp = np.asarray(pr["draws_pred"])
        ax.scatter(dt, dp, s=9, alpha=0.2, color=col)
        ax.scatter(pr["true"], pr["pred"], s=60, color=col, edgecolor="k", zorder=3)
        lo, hi = min(dt.min(), dp.min()), max(dt.max(), dp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true φ"); ax.set_ylabel("z-decoded φ")
        ax.set_title(f"{obj}: z→φ probe R²={pr['r2']:.3f}", fontsize=10)
    fig.suptitle("System-ID is preserved under the value-shaped objective", fontsize=11)
    figs["fig4_z_probe.png"] = _save(fig)

    # ---- fig5: planning (goal-dist vs capacity) + veridicality-vs-plannability ----
    if R.get("plan") and R.get("plan_caps"):
        pcaps = R["plan_caps"]
        rand = R["plan_random_dist"]

        def _pseries(obj, mode):
            return [sweep[obj][str(h)].get(mode, float("nan")) for h in pcaps]

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
        ax = axes[0]
        ax.plot(pcaps, _pseries("shaped", "plan_context"), "o-", color=C_SHP, lw=2.3, label="shaped + context")
        ax.plot(pcaps, _pseries("verid", "plan_context"), "o-", color=C_VER, lw=2.1, label="veridical + context")
        ax.plot(pcaps, _pseries("shaped", "plan_z0"), "s--", color=C_SHP, lw=1.8, alpha=0.6, label="shaped + pooled")
        ax.plot(pcaps, _pseries("verid", "plan_z0"), "s--", color=C_VER, lw=1.6, alpha=0.6, label="veridical + pooled")
        ax.axhline(rand, color="#888", ls=":", lw=1.2, label=f"random floor {rand:.2f}")
        ax.set_xscale("log", base=2); ax.set_xticks(pcaps); ax.set_xticklabels(pcaps)
        ax.set_xlabel("FM hidden width"); ax.set_ylabel("median goal-dist (lower = better)")
        ax.set_title("Does value-shaping buy CONTROL?", fontsize=10); ax.legend(fontsize=8, loc="upper right")

        # veridicality (all-dim ff R²) vs plannability (goal-dist), context mode
        ax = axes[1]
        for obj, col, mk in [("verid", C_VER, "o"), ("shaped", C_SHP, "D")]:
            xs = [sweep[obj][str(h)]["context"]["all_ff"] for h in pcaps]
            ys = _pseries(obj, "plan_context")
            ax.scatter(xs, ys, s=70, color=col, marker=mk, edgecolor="k", zorder=3, label=obj)
            for h, xx, yy in zip(pcaps, xs, ys):
                ax.annotate(str(h), (xx, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("full-state veridicality (all-dim ff R²)")
        ax.set_ylabel("median goal-dist (lower = better)")
        ax.set_title("Veridicality ⊥ usefulness: shaped is a worse\nsimulator (left) at equal/better control",
                     fontsize=9.5)
        ax.legend(fontsize=9)
        figs["fig5_planning.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_value_shaping(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # env / family
    frame_skip: int = 3,             # fine step -> the puck force field stays reducible (value_shaping)
    conflict: float = 1.5708,        # push_rot half-range Phi (rad); pi/2 = the clean regime
    arena_half: float = 0.9,
    gear: float = 10.0,
    puck_mass: float = 0.5,
    pusher_r: float = 0.12,
    puck_r: float = 0.12,
    field_amp: float = 1.3,          # puck force field (value-irrelevant capacity sink)
    field_pusher_amp: float = 0.0,   # pusher force field (value-RELEVANT capacity draw; 0 = easy pusher)
    field_central: float = 0.5,
    # tasks / collection
    n_train_tasks: int = 12,
    n_test_tasks: int = 6,
    n_episodes: int = 80,
    ep_len: int = 200,
    sigma: float = 0.7,
    theta: float = 0.15,
    seek_gain: float = 0.9,          # STRONG seek -> puck engaged (Guardrail 1)
    n_adapt_pool: int = 512,
    n_holdout: int = 3000,
    puck_move_thresh: float = 0.02,
    n_context: int = 128,
    shaped_puck_weight: float = 0.0,
    # encoder + conditional FM
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_layers: int = 2,
    ref_hidden: int = 32,            # binding capacity for the per-dim / probe readouts
    # meta-training
    meta_steps: int = 4000,
    meta_lr: float = 1e-3,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    fm_epochs: int = 40,             # oracle
    probe_draws: int = 20,
    # ---- Stage 2: CEM-MPC goal-reaching (does value-shaping buy CONTROL?) ---- #
    plan: bool = False,
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    plan_h: int = 80,                # env steps per episode (fs=3 -> reach needs many steps)
    plan_hp: int = 16,               # CEM plan horizon
    replan_every: int = 8,           # open-loop commit -> the world model is load-bearing
    k_shoot: int = 192,
    cem_iters: int = 3,
    cem_elite: int = 24,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.3,
    success_eps: float = 0.12,
    n_eval_plan: int = 32,
):
    import os

    caps = [16, 32, 64, 128, 256]
    plan_caps = [16, 32, 64, 256]
    if quick:
        n_train_tasks, n_test_tasks = 6, 3
        n_episodes, ep_len = 50, 160
        enc_hidden = 64
        meta_steps = 1200
        caps = [16, 64, 256]
        plan_caps = [16, 64]
        ref_hidden = 64
        probe_draws = 10
        n_holdout = 2500
        plan_h, k_shoot, n_eval_plan = 60, 128, 16
        tag = tag or "smoke"
    tag = tag or "default"

    puck_field = dict(amp=field_amp, pusher_amp=field_pusher_amp, central=field_central)
    cfg = dict(
        tag=tag, seed=seed, frame_skip=frame_skip, conflict=conflict,
        dgp_base=dict(arena_half=arena_half, gear=gear, puck_mass=puck_mass,
                      pusher_r=pusher_r, puck_r=puck_r, puck_field=puck_field),
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks,
        n_episodes=n_episodes, ep_len=ep_len, sigma=sigma, theta=theta, seek_gain=seek_gain,
        n_adapt_pool=n_adapt_pool, n_holdout=n_holdout, puck_move_thresh=puck_move_thresh,
        n_context=n_context, shaped_puck_weight=shaped_puck_weight,
        latent_dim=latent_dim, enc_hidden=enc_hidden, fm_layers=fm_layers,
        ref_hidden=ref_hidden, caps=caps,
        meta_steps=meta_steps, meta_lr=meta_lr, task_batch=task_batch, pred_P=pred_p,
        grad_clip=grad_clip, fm_epochs=fm_epochs, probe_draws=probe_draws,
        plan=plan, plan_caps=plan_caps, goal_range=goal_range, start_range=start_range,
        v0_std=v0_std, plan_H=plan_h, plan_Hp=plan_hp, replan_every=replan_every,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, success_eps=success_eps,
        n_eval_plan=n_eval_plan,
    )
    out = run_meta_value_shaping.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_value_shaping_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
