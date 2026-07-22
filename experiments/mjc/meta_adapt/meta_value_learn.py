"""Cut #4e — Closing the reward loop: value-shaping CAUSED by a reward signal.

Program: `ideas/two_timescale_value_loop.md` discriminator 4, the still-open piece
that Cut #4d (`meta_value_shaping.py`) named explicitly:

    "'Reward' in 4d is a decision-aware, value-SUPPORT-weighted PREDICTION objective
     (the shaping is DERIVED from V), NOT a closed-loop reward-DRIVEN outer loop.
     Closing that loop (the true two-timescale meta-objective) is the next step."

4d hand-set the per-dim loss weight from the goal-reaching value's support (match the
pusher dims [0,1,4,5], drop the puck dims [2,3,6,7]) and showed the resulting FM
re-allocation buys control UNDER CAPACITY COMPETITION (pusher force-field on, h=64:
planning gap +0.036+-0.012, 4 seeds) and is control-NEUTRAL on an easy pusher. That is
"value-shaping helps IF you impose it." This cut asks the causal question: does a
REWARD signal, flowing back, DISCOVER that shaping on its own?

    The change vs 4d: the per-dim weight `w` is no longer hand-set. It is chosen by an
    OUTER loop whose only signal is CONTROL PERFORMANCE (- median CEM-MPC goal distance
    in the real env). Nothing tells the loop the puck is irrelevant.

Two readouts, staged robust-core -> dissectible-upgrade (the value_shaping discipline):

  A1  REWARD LANDSCAPE (robust core). Sweep a scalar puck-weight w_puck (pusher dims=1,
      puck dims=w_puck), train the context-FM at h=64 for each, eval control. Run on
      BOTH regimes. Prediction:
        - capacity competition (field_pusher_amp>0): goal-dist MINIMIZED at low w_puck
          -> reward PREFERS the puck-drop (the shaping is reward-optimal, not just
          imposable).
        - easy pusher (field_pusher_amp=0): goal-dist FLAT in w_puck -> reward
          INDIFFERENT (the 4d control-neutrality, now as a null gradient).
      The pair is disc-4's causal-preference claim + the capacity-competition
      dissociation, at low variance.

  A2  REWARD-DRIVEN OUTER LOOP (dissectible upgrade). A free per-dim weight w in R^8,
      optimized by a derivative-free (CEM) outer loop driven ONLY by control reward.
      Does it REDISCOVER V's support -- drive the 4 puck dims down, keep the 4 pusher
      dims -- with no knowledge of the groups? Readout: converged per-dim weight
      (grouped pusher vs puck), control gain vs the uniform (veridical) baseline, and
      the correlation of the learned weight to the hand-derived support mask.

Why derivative-free (CEM) and not backprop-through-planner: a scalar reward nudging an
allocation is both the faster engineering path and the better analogy to a dopaminergic
signal (which does not backprop through a differentiable simulator).

Controls:
  * FM init + minibatch order are FIXED across all candidates (re-seed torch each
    meta_train) so control-cost is a clean function of `w` (variable control).
  * The reward is grounded in REAL env rollouts (not a learned value the loop can game)
    and w is bounded -> nothing to wirehead (the on-manifold-veto analog).

Substrate/machinery reused verbatim from meta_value_shaping.py (encoder E over 18-dim
(s,u,ds) context -> z; conditional FM f(s,u,z)->ds(8); push_rot actuator family; puck +
optional pusher force field; CEM-MPC goal-reaching whose cost IGNORES the puck). Only
the source of `w` changes. 4d is untouched (this is a new file).

Run:
    cd experiments/
    modal run mjc/meta_adapt/meta_value_learn.py::meta_value_learn --quick                 # smoke
    # A1 landscape, the dissociation (two tags):
    modal run --detach mjc/meta_adapt/meta_value_learn.py::meta_value_learn --tag land_pfield --field-pusher-amp 2.0
    modal run --detach mjc/meta_adapt/meta_value_learn.py::meta_value_learn --tag land_easy
    # A2 reward-driven outer loop (capacity-competition regime):
    modal run --detach mjc/meta_adapt/meta_value_learn.py::meta_value_learn --tag outer_pfield --field-pusher-amp 2.0 --outer
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (pusher_env.STATE_LABELS)
PUSHER_POS = [0, 1]
PUCK_POS = [2, 3]
PUSHER_VEL = [4, 5]
PUCK_VEL = [6, 7]
PUSHER = [0, 1, 4, 5]     # value-relevant: support of V = -||pos-g|| - beta*||vel||
PUCK = [2, 3, 6, 7]       # value-irrelevant: the distractor's own motion


@app.function(gpu="L4", memory=32768, timeout=10800, volumes={DATA_DIR: volume})
def run_meta_value_learn(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import collect_transitions, PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}  field_pusher_amp={cfg['field_pusher_amp']}  "
          f"(>0 = capacity competition)", flush=True)

    fs = cfg["frame_skip"]
    d = cfg["latent_dim"]
    hidden = cfg["hidden"]
    N_ctx = cfg["n_context"]

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
        dd["push_rot"] = float(p)
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
          f"any_contact={np.concatenate([dt['any_contact'] for dt in train_data]).mean():.3f}",
          flush=True)

    def feats(S, U, S2):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (Xt - mx) / sx; dy_n = (Yt - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    tr_feat = [feats(dt["S"], dt["U"], dt["S2"]) for dt in train_data]

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
            "hold_ff": ~ac[hi],
            "hold_slide": (~ac[hi]) & puck_moving[hi],
        })

    # shared planning eval set (fixed across ALL candidates + regimes -> controlled)
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
    # 2. encoder E (DeepSets) + conditional FM f(s,u,z) -- verbatim from 4d
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

    # ===================================================================== #
    # 3. INNER LOOP: meta-train E + f under an ARBITRARY per-dim weight `w`.
    #    Re-seeds torch so init + minibatch order are IDENTICAL across
    #    candidates -> control-cost is a clean function of `w` (variable control).
    # ===================================================================== #
    def meta_train(w_vec):
        torch.manual_seed(cfg["seed"] + 7)               # <- fixed init across candidates
        rng = np.random.default_rng(cfg["seed"] + 7)     # <- fixed data order across candidates
        w = torch.tensor(np.asarray(w_vec, np.float32), device=device)
        enc = Encoder().to(device)
        cfm = build_cfm()
        opt = torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()), lr=cfg["meta_lr"])
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
            z = enc(ctx)
            zc = z.unsqueeze(1).expand(-1, P, -1)
            pred = cfm(torch.cat([psu, zc], 2))
            loss = (huber(pred, pdy) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
            opt.step()
        enc.eval(); cfm.eval()
        return enc, cfm

    # ---- per-dim R^2 eval (context mode, held-out test tasks, median) ----
    def r2(pred, yt, dims):
        p, t = pred[:, dims], yt[:, dims]
        ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    def eval_perdim(enc, cfm):
        rows = {k: [] for k in ["pusher_pos_ff", "pusher_vel_ff", "puck_vel_slide", "all_ff"]}
        for tb in test_bufs:
            S, U, S2 = tb["hold_raw"]
            with torch.no_grad():
                z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
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
    # 4. THE REWARD: CEM-MPC goal-reaching control cost (cost IGNORES the puck).
    #    Committed `replan_every` open-loop steps -> world model load-bearing.
    # ===================================================================== #
    Hep, Hp, re_ = cfg["plan_H"], cfg["plan_Hp"], cfg["replan_every"]
    K, n_elite, vel_pen = cfg["k_shoot"], cfg["cem_elite"], cfg["vel_pen"]

    def mpc_action(cfm, z, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, Hp, 2), np.float32)
        sig = np.full((Bn, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(K, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(K, 0)
        zc = z.unsqueeze(0).expand(Bn * K, -1)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, K, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * K, Hp, 2), device=device)
                cost = torch.zeros(Bn * K, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (cfm(torch.cat([(x - mx) / sx, zc], 1)) * sy + my)
                    cost = cost + (s[:, PUSHER_POS] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, PUSHER_VEL].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, K), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_plan(cfm, z, env, seed):
        rng = np.random.default_rng(seed)
        states = ev_starts.copy(); plan = None
        for step in range(Hep):
            if step % re_ == 0:
                plan = mpc_action(cfm, z, states, ev_goals, rng)
            acts = plan[:, step % re_, :]
            for b in range(states.shape[0]):
                env.set_state(states[b, 0:4], states[b, 4:8])
                s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, PUSHER_POS] - ev_goals, axis=1)
        return float(np.median(fd))

    def control_cost(enc, cfm):
        """The reward signal (lower = better): median goal-dist over test tasks, context mode."""
        dists = []
        for ti, (tb, env) in enumerate(zip(test_bufs, test_envs)):
            with torch.no_grad():
                z = enc(tb["adapt_ctx"][:N_ctx].unsqueeze(0))[0]
            dists.append(eval_plan(cfm, z, env, cfg["seed"] + 300 + ti))
        return float(np.median(dists))

    def evaluate(w_vec):
        """One inner-loop step: train FM under weight w, return control cost + per-dim R^2."""
        enc, cfm = meta_train(w_vec)
        return {"cost": control_cost(enc, cfm), "perdim": eval_perdim(enc, cfm)}

    # ===================================================================== #
    # A1. REWARD LANDSCAPE over a scalar puck-weight (pusher dims fixed at 1)
    # ===================================================================== #
    def w_from_puck(w_puck):
        w = np.ones(8, np.float32)
        for i in PUCK:
            w[i] = w_puck
        return w

    print("\n[A1] reward landscape over puck-weight ...", flush=True)
    grid = cfg["puck_grid"]
    landscape = []
    for wp in grid:
        r = evaluate(w_from_puck(wp))
        landscape.append({"w_puck": float(wp), "cost": r["cost"], **r["perdim"]})
        print(f"  w_puck={wp:5.3f}  ctrl_cost={r['cost']:.4f}  "
              f"pusherV={r['perdim']['pusher_vel_ff']:.3f}  puckV={r['perdim']['puck_vel_slide']:.3f}",
              flush=True)
    costs = [x["cost"] for x in landscape]
    best_i = int(np.argmin(costs))
    veridical_i = int(np.argmin([abs(x["w_puck"] - 1.0) for x in landscape]))  # w_puck=1 = veridical
    print(f"[A1] reward-optimal w_puck={landscape[best_i]['w_puck']:.3f} (cost {costs[best_i]:.4f}) | "
          f"veridical w_puck=1 cost {costs[veridical_i]:.4f} | "
          f"spread {max(costs)-min(costs):.4f}", flush=True)

    # ===================================================================== #
    # A2. REWARD-DRIVEN OUTER LOOP: free w in R^8, CEM on control cost.
    #     w_i = 2*sigmoid(theta_i) in [0,2] (theta=0 -> w=1 = uniform/veridical).
    # ===================================================================== #
    outer = None
    if cfg["outer"]:
        print("\n[A2] reward-driven outer loop (CEM over free per-dim weight) ...", flush=True)

        def theta_to_w(theta):
            if cfg["outer_normalize"]:
                # 8*softmax(theta) -> sum(w)=8 (mean 1): a PURE ALLOCATION over the 8
                # prediction targets, no global-scale DOF. Removes the grad-clip/scale
                # confound (an unbounded w lets Adam+clip improve control by lowering the
                # overall weight, NOT by re-allocating) so any gain MUST be re-allocation.
                e = np.exp(theta - theta.max())
                return (8.0 * e / e.sum()).astype(np.float32)
            return (2.0 / (1.0 + np.exp(-theta))).astype(np.float32)   # legacy: w_i in [0,2] (scale confound)

        orng = np.random.default_rng(cfg["seed"] + 555)
        POP, EL, IT = cfg["outer_pop"], cfg["outer_elite"], cfg["outer_iters"]
        mu = np.zeros(8, np.float32)                     # -> w=1 uniform (veridical) start
        sig = np.full(8, cfg["outer_sigma0"], np.float32)
        # uniform (veridical) baseline cost, for the gain readout
        base = evaluate(w_from_puck(1.0))
        base_cost = base["cost"]
        history = []
        best = {"cost": base_cost, "w": np.ones(8, np.float32).tolist(), "theta": mu.tolist()}
        for it in range(IT):
            thetas = mu[None] + sig[None] * orng.standard_normal((POP, 8)).astype(np.float32)
            cands = []
            for p in range(POP):
                w = theta_to_w(thetas[p])
                r = evaluate(w)
                cands.append({"theta": thetas[p], "w": w, "cost": r["cost"], "perdim": r["perdim"]})
            cands.sort(key=lambda c: c["cost"])
            elite = cands[:EL]
            mu = np.mean([c["theta"] for c in elite], 0).astype(np.float32)
            sig = (np.std([c["theta"] for c in elite], 0) + 1e-2).astype(np.float32)
            if elite[0]["cost"] < best["cost"]:
                best = {"cost": elite[0]["cost"], "w": elite[0]["w"].tolist(),
                        "theta": elite[0]["theta"].tolist(), "perdim": elite[0]["perdim"]}
            wbar = theta_to_w(mu)
            history.append({"iter": it, "best_cost": float(elite[0]["cost"]),
                            "mean_w": wbar.tolist(),
                            "puck_w": float(np.mean(wbar[PUCK])), "pusher_w": float(np.mean(wbar[PUSHER]))})
            print(f"  [outer it {it}] best_cost={elite[0]['cost']:.4f}  "
                  f"mean_w pusher={np.mean(wbar[PUSHER]):.3f} puck={np.mean(wbar[PUCK]):.3f}",
                  flush=True)

        w_star = np.asarray(best["w"], np.float32)
        support_mask = np.array([1.0 if i in PUSHER else 0.0 for i in range(8)], np.float32)
        # correlation of learned weight with the hand-derived support (pusher=1,puck=0)
        wc = w_star - w_star.mean(); sc = support_mask - support_mask.mean()
        corr = float((wc @ sc) / (np.linalg.norm(wc) * np.linalg.norm(sc) + 1e-9))
        outer = {
            "base_cost_uniform": base_cost, "best_cost": best["cost"],
            "w_star": best["w"], "w_star_pusher": float(np.mean(w_star[PUSHER])),
            "w_star_puck": float(np.mean(w_star[PUCK])),
            "corr_to_support": corr, "history": history,
            "gain_vs_uniform": float(base_cost - best["cost"]),
        }
        print(f"[A2] learned w: pusher={outer['w_star_pusher']:.3f}  puck={outer['w_star_puck']:.3f}  "
              f"(ratio {outer['w_star_puck']/(outer['w_star_pusher']+1e-9):.2f})  "
              f"corr->support={corr:.3f}  control gain vs uniform={outer['gain_vs_uniform']:+.4f} "
              f"(uniform {base_cost:.4f} -> learned {best['cost']:.4f})", flush=True)

    # ===================================================================== #
    # save
    # ===================================================================== #
    results = {
        "config": cfg, "family": "actuator", "conflict": Phi, "hidden": hidden,
        "field_pusher_amp": cfg["field_pusher_amp"],
        "regime": "capacity_competition" if cfg["field_pusher_amp"] > 0 else "easy_pusher",
        "train_params": train_p.tolist(), "test_params": test_p.tolist(),
        "plan_random_dist": rand_dist, "landscape": landscape,
        "landscape_best_w_puck": landscape[best_i]["w_puck"],
        "landscape_veridical_cost": costs[veridical_i],
        "landscape_best_cost": costs[best_i], "outer": outer,
    }
    figures = _make_figures(results)
    outdir = os.path.join(DATA_DIR, "meta_value_learn", cfg["tag"])
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
    C_COST, C_PUSH, C_PUCK = "#3d6fd1", "#2f9e44", "#d1603d"
    figs = {}
    regime = R["regime"]
    reg_lbl = "capacity competition (pusher field on)" if regime == "capacity_competition" else "easy pusher"

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: reward landscape over puck-weight ----
    L = R["landscape"]
    wp = [x["w_puck"] for x in L]
    cost = [x["cost"] for x in L]
    pv = [x["pusher_vel_ff"] for x in L]
    kv = [max(x["puck_vel_slide"], -1.5) for x in L]   # clip dropped puck for display
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(wp, cost, "o-", color=C_COST, lw=2.4, label="control cost (goal-dist)")
    ax.axhline(R["plan_random_dist"], color="#888", ls=":", lw=1.1, label=f"random floor {R['plan_random_dist']:.2f}")
    bi = int(np.argmin(cost))
    ax.scatter([wp[bi]], [cost[bi]], s=140, facecolor="none", edgecolor=C_COST, lw=2.2, zorder=5,
               label=f"reward-optimal w_puck={wp[bi]:.3f}")
    ax.set_xlabel("puck-weight  w_puck   (0 = drop the puck = value-shaped   |   1 = veridical)")
    ax.set_ylabel("median goal-dist (lower = better control)", color=C_COST)
    ax.tick_params(axis="y", labelcolor=C_COST)
    ax2 = ax.twinx()
    ax2.plot(wp, pv, "s--", color=C_PUSH, lw=1.6, alpha=0.85, label="pusher-vel R² (value-relevant)")
    ax2.plot(wp, kv, "d--", color=C_PUCK, lw=1.6, alpha=0.85, label="puck-vel R² (value-irrelevant)")
    ax2.set_ylabel("per-dim R²"); ax2.spines["top"].set_visible(False)
    lines1, lab1 = ax.get_legend_handles_labels(); lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    verdict = ("reward PREFERS the puck-drop" if cost[bi] < cost[-1] - 1e-4 and wp[bi] < 0.5
               else "reward ~INDIFFERENT to the puck")
    ax.set_title(f"A1 — reward landscape [{reg_lbl}]\n{verdict}  "
                 f"(veridical {cost[-1]:.3f} → optimal {cost[bi]:.3f})", fontsize=10)
    figs["fig1_reward_landscape.png"] = _save(fig)

    # ---- fig2: the reward-driven outer loop (A2) ----
    if R.get("outer"):
        O = R["outer"]
        w_star = np.asarray(O["w_star"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        ax = axes[0]
        cols = [C_PUSH if i in PUSHER else C_PUCK for i in range(8)]
        labels = ["puX", "puY", "pkX", "pkY", "puVx", "puVy", "pkVx", "pkVy"]
        ax.bar(range(8), w_star, color=cols)
        ax.axhline(1.0, color="#888", ls="--", lw=1, label="uniform (veridical)")
        ax.set_xticks(range(8)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("learned weight w*")
        ax.set_title(f"A2 — reward-DISCOVERED weight\npusher(green)={O['w_star_pusher']:.2f}  "
                     f"puck(orange)={O['w_star_puck']:.2f}  corr→support={O['corr_to_support']:.2f}",
                     fontsize=9.5)
        ax.legend(fontsize=8)
        ax = axes[1]
        hist = O["history"]
        its = [h["iter"] for h in hist]
        ax.plot(its, [h["pusher_w"] for h in hist], "o-", color=C_PUSH, lw=2, label="mean pusher weight")
        ax.plot(its, [h["puck_w"] for h in hist], "o-", color=C_PUCK, lw=2, label="mean puck weight")
        ax.axhline(1.0, color="#888", ls="--", lw=1)
        ax.set_xlabel("outer-loop iteration"); ax.set_ylabel("mean group weight")
        gain = O["gain_vs_uniform"]
        ax.set_title(f"outer loop drives puck ↓, pusher ↑\ncontrol gain vs uniform "
                     f"{gain:+.3f} ({O['base_cost_uniform']:.3f}→{O['best_cost']:.3f})", fontsize=9.5)
        ax.legend(fontsize=8)
        figs["fig2_outer_loop.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_value_learn(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # env / family
    frame_skip: int = 3,
    conflict: float = 1.5708,        # push_rot half-range Phi (rad); pi/2 = the clean regime
    arena_half: float = 0.9,
    gear: float = 10.0,
    puck_mass: float = 0.5,
    pusher_r: float = 0.12,
    puck_r: float = 0.12,
    field_amp: float = 1.3,          # puck force field (value-irrelevant capacity sink)
    field_pusher_amp: float = 0.0,   # pusher force field (>0 = CAPACITY COMPETITION; 0 = easy null)
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
    # encoder + conditional FM (single capacity = the 4d sweet spot)
    hidden: int = 64,
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_layers: int = 2,
    # inner meta-training
    meta_steps: int = 3000,
    meta_lr: float = 1e-3,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    # A1 landscape grid
    puck_grid: str = "0.0,0.125,0.25,0.5,1.0",
    # A2 outer loop
    outer: bool = False,
    outer_normalize: bool = True,    # w = 8*softmax(theta) (pure allocation); False = legacy w in [0,2]
    outer_pop: int = 10,
    outer_elite: int = 3,
    outer_iters: int = 5,
    outer_sigma0: float = 1.5,
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

    grid = [float(x) for x in puck_grid.split(",")]
    if quick:
        n_train_tasks, n_test_tasks = 6, 3
        n_episodes, ep_len = 40, 140
        enc_hidden = 64
        meta_steps = 500
        n_holdout = 2200
        grid = [0.0, 0.5, 1.0]
        outer_pop, outer_iters = 4, 2
        plan_h, k_shoot, n_eval_plan = 60, 128, 8
        tag = tag or "smoke"
    tag = tag or "default"

    puck_field = dict(amp=field_amp, pusher_amp=field_pusher_amp, central=field_central)
    cfg = dict(
        tag=tag, seed=seed, frame_skip=frame_skip, conflict=conflict,
        field_pusher_amp=field_pusher_amp,
        dgp_base=dict(arena_half=arena_half, gear=gear, puck_mass=puck_mass,
                      pusher_r=pusher_r, puck_r=puck_r, puck_field=puck_field),
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks,
        n_episodes=n_episodes, ep_len=ep_len, sigma=sigma, theta=theta, seek_gain=seek_gain,
        n_adapt_pool=n_adapt_pool, n_holdout=n_holdout, puck_move_thresh=puck_move_thresh,
        n_context=n_context, hidden=hidden, latent_dim=latent_dim, enc_hidden=enc_hidden,
        fm_layers=fm_layers, meta_steps=meta_steps, meta_lr=meta_lr, task_batch=task_batch,
        pred_P=pred_p, grad_clip=grad_clip, puck_grid=grid,
        outer=outer, outer_normalize=outer_normalize, outer_pop=outer_pop,
        outer_elite=outer_elite, outer_iters=outer_iters, outer_sigma0=outer_sigma0,
        goal_range=goal_range, start_range=start_range, v0_std=v0_std, plan_H=plan_h,
        plan_Hp=plan_hp, replan_every=replan_every, k_shoot=k_shoot, cem_iters=cem_iters,
        cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        n_eval_plan=n_eval_plan,
    )
    out = run_meta_value_learn.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_value_learn_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
