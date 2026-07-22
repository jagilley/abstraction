"""Cut 4b (drift_value_loop): does a BALLISTIC controller transmit FM quality to behavior?
The clean, variable-controlled test.

Program: `ideas/two_timescale_value_loop.md`. Parent: `ballistic_control.py` (Cut 4 — the
b-drive version), which returned a partial-negative AND a diagnostic: in the corridor geometry
the explore/exploit balance `b` is CONFOUNDED (the needle sits ON the corridor so "exploit" =
path-density actually COVERS it, while "explore" = reducible-surprise CHASES the off-corridor
noise — the intended roles invert; needle-error is LOWEST at b=0). So `b` was never cleanly
varying the control-relevant FM quality, and ballistic-vs-reactive transmission was confounded
with a broken FM-quality source.

    THE FIX (control variables). Drop the confounded b-drive. Manufacture a CONTROLLED
    FM-quality axis directly — and a SMOOTH/GLOBAL one, not a localized needle. (A localized
    perturbation is open-loop-INCOMPENSABLE: even a perfect FM can't counteract a strong local
    force kick in open-loop, because small timing errors change the kick — which is *why*
    biological control uses feedback there. So a needle saturates ballistic control at "fail"
    for every FM quality — an early build confirmed this.) The reach ITSELF — accelerate then
    decelerate to stop at the goal — requires good MOMENTUM/DAMPING modeling, which IS smooth
    and open-loop-compensable. So the quality axis is damping-STALENESS: train each FM on data
    from `d_train` dynamics and test on fixed `d_test`; FM prediction-error on the true dynamics
    grows monotonically with |d_train − d_test| (d_train=d_test → a matched FM; far → a stale
    one). No perturbation, no drive machinery.

    THE CLAIM (the biological one, isolated). The cerebellar FM matters to the degree you must
    commit open-loop (feedforward), because you cannot replan fast enough. So on a plain reach:
      * a REACTIVE controller (re-ground every step) is ROBUST to a stale FM — it mis-steps,
        re-plans from the true state, and recovers;
      * a BALLISTIC controller (commit the whole trajectory open-loop) TRANSMITS the FM's
        staleness — a stale-damping FM mis-times the deceleration, overshoots, and cannot correct.
    So control-dist vs FM-error should be STEEP for the ballistic controllers and FLAT for
    the reactive one. Transmission = slope(ballistic) >> slope(reactive).

    THREE CONTROLLERS (same FM, true-env execution, identical threading reaches — controlled):
      * `reactive`      — CEM-MPC, re-plan every step (re=1): the FM-robust grader (Cut 3's).
      * `ballistic_cem` — CEM-MPC, plan once, execute the whole horizon open-loop (re=H):
                          committed but still MODEL-OPTIMAL given the FM.
      * `ballistic_bc`  — a genuinely-ballistic feedforward MOTOR PROGRAM π(s0,g)→(H×2 open-loop
                          sequence), behavior-cloned from ballistic-CEM-on-FM plans (the
                          dynamics_shift.py pattern): amortized, NOT re-optimized per episode —
                          the most faithful feedforward controller (bakes in the FM's blind
                          spots). Tests whether amortization transmits MORE than open-loop CEM.

Reuses online_value_loop.py's env/FM/CEM machinery (self-contained duplication, the established
pattern). Static dynamics, no drift, no outer loop — the clean mechanism isolation.

Run:
    cd experiments/
    modal run mujoco_control/ballistic_transmission.py::ballistic_transmission --quick     # smoke
    for s in 0 1 2; do
      modal run --detach mujoco_control/ballistic_transmission.py::ballistic_transmission \
          --tag trans_s$s --seed $s
    done
    python3 mujoco_control/ballistic_transmission_figure.py --tags trans_s0 trans_s1 trans_s2
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_ballistic_transmission(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]
    H = cfg["plan_H"]
    cr = cfg["corridor_r"]
    d_test = cfg["d_test"]
    print(f"[setup] device={device} d_test={d_test} damp_trains={cfg['damp_trains']} "
          f"controllers={cfg['controllers']} H={H} corridor_r={cr}", flush=True)

    # ===================================================================== #
    # env: a clean puck-free reach world. The TRUE (test) dynamics use joint_damping
    # = d_test. FM quality is controlled by training on data from d_train dynamics
    # (a separate env per d_train); staleness = |d_train - d_test|. No perturbation.
    # ===================================================================== #
    def make_env(damp):
        return PusherEnv(dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                              joint_damping=damp, pusher_r=0.12), with_puck=False)

    eval_env = make_env(d_test)                          # control eval = the true dynamics

    # ===================================================================== #
    # FM f(s,u)->Δs (single deterministic MLP is enough here — controlled data, no ensemble)
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

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)

    def train_fm(net, norm, S, U, S2, steps, brng):
        opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fm_delta(net, norm, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1), device=device, dtype=torch.float32)
            Xn = (X - norm["mx"]) / norm["sx"]
            return (net(Xn) * norm["sy"] + norm["my"]).cpu().numpy()

    # ===================================================================== #
    # collect a pool over the corridor band from a GIVEN env (its own damping)
    # ===================================================================== #
    def collect_pool(cenv, n, rng):
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        for i in range(n):
            pos = np.array([rng.uniform(-cr - 0.2, cr + 0.2), rng.uniform(-cfg["band_h"], cfg["band_h"])], np.float32)
            vel = rng.normal(0, cfg["v_explore"], 2).astype(np.float32)
            cenv.set_state(pos.astype(np.float64), vel.astype(np.float64))
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            s = cenv.get_state(); s2, _ = cenv.step(u, fs)
            S[i] = s; U[i] = u; S2[i] = s2
        return S, U, S2

    # shared normalization from a d_test pool (identical across FMs -> controlled)
    normS, normU, normS2 = collect_pool(eval_env, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    norm = make_norm(normS, normU, normS2)

    # held-out probes on the TRUE (d_test) dynamics: the FM-quality axis (fm-err = how well the
    # FM predicts the dynamics the controller actually operates in).
    pS, pU, pS2 = collect_pool(eval_env, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    pT = (pS2 - pS).astype(np.float32)

    def probe_err(net):
        return float(np.linalg.norm(fm_delta(net, norm, pS, pU) - pT, axis=1).mean())

    # ===================================================================== #
    # CEM planner over an FM (returns a full H-step plan from each state)
    # ===================================================================== #
    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]
    vel_pen = cfg["vel_pen"]

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
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)                     # (Bn, H, 2)

    # ===================================================================== #
    # threading reaches: start/goal straddle the needle so the path MUST cross it
    # ===================================================================== #
    B = cfg["n_eval"]

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        sgn = rng.choice([-1.0, 1.0], B).astype(np.float32)
        spos = np.stack([sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        goals = np.stack([-sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        starts = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)
        return starts, goals

    def rollout(plan_fn, starts, goals, replan_every):
        states = starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                eval_env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = eval_env.step(acts[b], fs); states[b] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - goals, axis=1)))

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    # ===================================================================== #
    # BC motor-program policy pi(s,g)->(H*2) open-loop sequence (dynamics_shift pattern)
    # ===================================================================== #
    def build_policy():
        h, L = cfg["pol_hidden"], cfg["pol_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, 2 * H), nn.Tanh()]
        return nn.Sequential(*lyr).to(device)

    def bc_train_and_eval(net_fm):
        """Behavior-clone ballistic-CEM-on-this-FM plans into a feedforward motor program,
        then execute it OPEN-LOOP in the true env."""
        rng = np.random.default_rng(cfg["seed"] + 55)
        nt = cfg["bc_tuples"]
        # sample (s0,g) threading reaches (same distribution as eval, more of them)
        sgn = rng.choice([-1.0, 1.0], nt).astype(np.float32)
        spos = np.stack([sgn * cr, np.zeros(nt, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (nt, 2)).astype(np.float32)
        gpos = np.stack([-sgn * cr, np.zeros(nt, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (nt, 2)).astype(np.float32)
        s0 = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (nt, 2)).astype(np.float32)], 1)
        # targets = ballistic-CEM-on-FM full-H plans (in minibatches to bound memory)
        plans = np.empty((nt, H, 2), np.float32)
        for i in range(0, nt, 256):
            j = min(i + 256, nt)
            plans[i:j] = mpc_plan(net_fm, s0[i:j], gpos[i:j], rng)
        X = np.concatenate([s0, gpos], 1).astype(np.float32)
        pnorm = {"mu": torch.tensor(X.mean(0), device=device), "sd": torch.tensor(X.std(0) + 1e-6, device=device)}
        Xt = (torch.tensor(X, device=device) - pnorm["mu"]) / pnorm["sd"]
        Yt = torch.tensor(plans.reshape(nt, 2 * H), device=device)
        pol = build_policy(); optp = torch.optim.Adam(pol.parameters(), lr=cfg["pol_lr"]); lossf = nn.MSELoss()
        brng = np.random.default_rng(cfg["seed"] + 56); pol.train()
        bs = min(512, nt)
        for _ in range(cfg["pol_steps"]):
            idx = torch.tensor(brng.integers(0, nt, size=bs), device=device)
            optp.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); optp.step()
        pol.eval()

        def plan_fn(states, goals):
            with torch.no_grad():
                x = (torch.tensor(np.concatenate([states, goals], 1), device=device) - pnorm["mu"]) / pnorm["sd"]
                return pol(x).cpu().numpy().astype(np.float32).reshape(len(states), H, 2)
        return rollout(plan_fn, ev_starts, ev_goals, replan_every=H)   # fully open-loop

    # ===================================================================== #
    # the FM-quality ladder: train one FM per d_train (collected on its own dynamics),
    # measure its error on the TRUE dynamics, grade with each controller on the true env
    # ===================================================================== #
    results = []
    for d_train in cfg["damp_trains"]:
        tenv = make_env(d_train)
        S, U, S2 = collect_pool(tenv, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 400 + int(100 * d_train)))
        net = _mlp(cfg["seed"] + 40)
        train_fm(net, norm, S, U, S2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
        fm_err = probe_err(net)                          # error on the TRUE (d_test) dynamics

        rec = {"d_train": d_train, "staleness": abs(d_train - d_test), "fm_err": fm_err}
        rng = np.random.default_rng(cfg["seed"] + 7000)
        if "reactive" in cfg["controllers"]:
            rec["reactive"] = rollout(lambda s, g: mpc_plan(net, s, g, rng), ev_starts, ev_goals, replan_every=1)
        if "ballistic_cem" in cfg["controllers"]:
            rng2 = np.random.default_rng(cfg["seed"] + 7001)
            rec["ballistic_cem"] = rollout(lambda s, g: mpc_plan(net, s, g, rng2), ev_starts, ev_goals, replan_every=H)
        if "ballistic_bc" in cfg["controllers"]:
            rec["ballistic_bc"] = bc_train_and_eval(net)
        results.append(rec)
        msg = f"[d_train={d_train:.1f} (stale {rec['staleness']:.1f})] fm_err={fm_err:.4f}"
        for c in cfg["controllers"]:
            msg += f"  {c}={rec.get(c, float('nan')):.4f}"
        print(msg, flush=True)

    # ---- transmission slopes: d(control-dist)/d(fm-err), per controller ----
    ne_arr = np.array([r["fm_err"] for r in results])
    slopes = {}
    for c in cfg["controllers"]:
        cvals = np.array([r[c] for r in results])
        if len(ne_arr) >= 2 and ne_arr.std() > 1e-9:
            slopes[c] = float(np.polyfit(ne_arr, cvals, 1)[0])
        else:
            slopes[c] = float("nan")
    print(f"\n[transmission] slope d(control)/d(fm_err): "
          + "  ".join(f"{c}={slopes[c]:+.2f}" for c in cfg["controllers"]), flush=True)
    print("[transmission] PREDICTION: ballistic slopes >> reactive slope (ballistic transmits FM "
          "staleness; reactive re-grounds past it).", flush=True)

    out = {"config": cfg, "results": results, "slopes": slopes}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "ballistic_transmission", cfg["tag"])
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
    COL = {"reactive": "#7048e8", "ballistic_cem": "#e8590c", "ballistic_bc": "#c92a2a"}
    res = R["results"]; ctrls = R["config"]["controllers"]
    ne = np.array([r["fm_err"] for r in res])
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # control-dist vs needle-error, per controller (the headline)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    order = np.argsort(ne)
    for c in ctrls:
        cv = np.array([r[c] for r in res])[order]
        lbl = {"reactive": "reactive (re-ground every step)",
               "ballistic_cem": "ballistic CEM (open-loop, model-optimal)",
               "ballistic_bc": "ballistic BC motor-program (feedforward)"}.get(c, c)
        ax.plot(ne[order], cv, "o-", color=COL.get(c, "#333"), lw=2.4, ms=7,
                label=f"{lbl}   (slope {R['slopes'][c]:+.1f})")
    ax.set_xlabel("FM prediction error on the TRUE dynamics  (damping-staleness quality axis)")
    ax.set_ylabel("control goal-dist (lower=better)")
    ax.set_title("Does the ballistic controller transmit FM quality?\n"
                 "steep = transmits (FM-sensitive); flat = robust (re-grounds past the error)")
    ax.legend(fontsize=8.5, loc="best")
    figs["fig1_transmission.png"] = _save(fig)

    # bars: control-dist at the stalest vs matched FM per controller — the transmission gap
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    q0 = res[int(np.argmax([r["fm_err"] for r in res]))]   # stalest FM
    q1 = res[int(np.argmin([r["fm_err"] for r in res]))]   # matched FM
    x = np.arange(len(ctrls)); w = 0.38
    ax.bar(x - w/2, [q1[c] for c in ctrls], w, color="#2f9e44", label=f"matched FM (fm_err={q1['fm_err']:.3f})")
    ax.bar(x + w/2, [q0[c] for c in ctrls], w, color="#c92a2a", label=f"stale FM (fm_err={q0['fm_err']:.3f})")
    for i, c in enumerate(ctrls):
        gap = q0[c] - q1[c]
        ax.text(i, max(q0[c], q1[c]) + 0.005, f"Δ={gap:+.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(ctrls, rotation=15, ha="right")
    ax.set_ylabel("control goal-dist"); ax.set_title("transmission gap: blind−good FM per controller")
    ax.legend(fontsize=8.5)
    figs["fig2_gap.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def ballistic_transmission(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    controllers: str = "reactive,ballistic_cem,ballistic_bc",
    d_test: float = 2.0,                   # the TRUE dynamics the controller operates in
    damp_trains: str = "2.0,3.0,4.5,7.0",  # FM training dynamics: staleness = |d_train - d_test|
    # env
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    corridor_r: float = 0.5,
    band_h: float = 0.5,                   # collection band half-height around the corridor
    # collection / probes
    pool_n: int = 9000,
    probe_n: int = 800,
    v_explore: float = 1.2,
    probe_v: float = 0.15,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    # BC motor-program policy
    pol_hidden: int = 256,
    pol_layers: int = 3,
    pol_lr: float = 1e-3,
    pol_steps: int = 4000,
    bc_tuples: int = 2000,
    # control eval
    n_eval: int = 40,
    goal_jit: float = 0.06,
    v0_std: float = 0.3,
    plan_h: int = 30,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    ctrl_list = [c for c in controllers.split(",") if c]
    damp_list = [float(x) for x in damp_trains.split(",") if x]
    if quick:
        damp_list = [2.0, 7.0]; ctrl_list = ["reactive", "ballistic_cem", "ballistic_bc"]
        pool_n = 3000; probe_n = 300; fm_steps = 1200; pol_steps = 1200; bc_tuples = 600
        fm_hidden = 128; fm_layers = 2; pol_hidden = 128; pol_layers = 2
        n_eval = 16; k_shoot = 128
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, controllers=ctrl_list, d_test=d_test, damp_trains=damp_list,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear,
        corridor_r=corridor_r, band_h=band_h,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore, probe_v=probe_v,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps, bc_tuples=bc_tuples,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_ballistic_transmission.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "ballistic_transmission_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
