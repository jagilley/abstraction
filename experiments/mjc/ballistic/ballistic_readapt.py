"""Cut 4c (drift_value_loop): the END-TO-END loop — online FM re-adaptation after a Type-2
drift restores BALLISTIC motor competence; reactive control never needed it.

Program: `ideas/two_timescale_value_loop.md`. Parents: `ballistic_transmission.py` (Cut 4b —
a BALLISTIC controller transmits FM quality ~3× more than reactive, on a CONTROLLED
damping-staleness axis) and `dynamics_shift.py` (reward-free FM re-adaptation recovers MB
control after a drag→ice shift). This cut CLOSES THE LOOP those two leave open: it makes the
FM-quality axis ENDOGENOUS — produced by the learning loop re-adapting online after a drift —
and grades it under both controllers.

    THE CHAIN (value-loop → FM-quality → ballistic-control). A Type-2 damping drift d0→d1
    leaves the cerebellar FM STALE. The learning layer re-adapts it from reward-free d1
    transitions (self-supervised — no reward, the cerebellum re-learning "what will happen"
    under the new body/table). We snapshot the FM along the re-adaptation trajectory and grade
    each snapshot under:
      * `reactive`      — CEM-MPC re-plan every step (FM-robust: re-grounds past staleness);
      * `ballistic_cem` — CEM-MPC open-loop (committed; FM-sensitive);
      * `ballistic_bc`  — a behavior-cloned feedforward motor program (at the endpoints).

    THREE PREDICTIONS, one figure:
      1. BALLISTIC control RECOVERS as the FM re-adapts (stale→matched): re-adaptation is
         behaviorally LOAD-BEARING under feedforward commitment.
      2. REACTIVE control is ~FLAT and already competent throughout: it never needed the
         re-adaptation. So the *behavioral value of re-adaptation is ballistic-specific*.
      3. The FM prediction-error (Cut 3's cerebellum→VTA teacher) TRACKS the ballistic recovery
         — so the value signal the meta-layer reads is a valid proxy for the behavioral payoff,
         but only because control is ballistic (for reactive it over-predicts the need).

    This is the whole thesis in one run: the learning layer delivers FM quality (fast
    re-adaptation), the value/meta signal (FM error) tracks it, and it only cashes out as
    behavior because biological motor control is feedforward.

Reuses ballistic_transmission.py's env/FM/CEM/BC/rollout (self-contained duplication). Uniform
reward-free re-adaptation collection (NO drive — the corridor-geometry drive is confounded,
per Cut 4; the learning-layer re-adaptation is the clean, drive-free piece).

Run:
    cd experiments/
    modal run mujoco_control/ballistic_readapt.py::ballistic_readapt --quick        # smoke
    for s in 0 1 2; do
      modal run --detach mujoco_control/ballistic_readapt.py::ballistic_readapt --tag readapt_s$s --seed $s
    done
    python3 mujoco_control/ballistic_readapt_figure.py --tags readapt_s0 readapt_s1 readapt_s2
"""

import copy
import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_ballistic_readapt(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    d0, d1 = cfg["d0"], cfg["d1"]
    print(f"[setup] device={device} drift d0={d0}->d1={d1} milestones={cfg['milestones']} "
          f"controllers={cfg['controllers']} H={H}", flush=True)

    def make_env(damp):
        return PusherEnv(dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                              joint_damping=damp, pusher_r=0.12), with_puck=False)

    env0 = make_env(d0)              # pre-drift dynamics (FM pretraining)
    env1 = make_env(d1)             # post-drift dynamics (re-adaptation + control eval = TRUE)

    # ---------------- FM f(s,u)->Δs ----------------
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

    def train_steps(net, opt, norm, S, U, S2, steps, brng):
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
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    # ---------------- collection ----------------
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

    # normalization fixed from a d1 pool (the operating dynamics); probes on TRUE (d1)
    nS, nU, nS2 = collect_pool(env1, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    norm = make_norm(nS, nU, nS2)
    pS, pU, pS2 = collect_pool(env1, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))
    pT = (pS2 - pS).astype(np.float32)

    def fm_err(net):
        return float(np.linalg.norm(fm_delta(net, norm, pS, pU) - pT, axis=1).mean())

    # ---------------- CEM planner ----------------
    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]; B = cfg["n_eval"]

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
        return mu.astype(np.float32)

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        sgn = rng.choice([-1.0, 1.0], B).astype(np.float32)
        spos = np.stack([sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        goals = np.stack([-sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        starts = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every):
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                env1.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = env1.step(acts[b], fs); states[b] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - ev_goals, axis=1)))

    # ---------------- BC feedforward motor program (endpoints only) ----------------
    def build_policy():
        h, L = cfg["pol_hidden"], cfg["pol_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, 2 * H), nn.Tanh()]
        return nn.Sequential(*lyr).to(device)

    def ballistic_bc(net_fm):
        rng = np.random.default_rng(cfg["seed"] + 55); nt = cfg["bc_tuples"]
        sgn = rng.choice([-1.0, 1.0], nt).astype(np.float32)
        spos = np.stack([sgn * cr, np.zeros(nt, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (nt, 2)).astype(np.float32)
        gpos = np.stack([-sgn * cr, np.zeros(nt, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (nt, 2)).astype(np.float32)
        s0 = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (nt, 2)).astype(np.float32)], 1)
        plans = np.empty((nt, H, 2), np.float32)
        for i in range(0, nt, 256):
            j = min(i + 256, nt); plans[i:j] = mpc_plan(net_fm, s0[i:j], gpos[i:j], rng)
        X = np.concatenate([s0, gpos], 1).astype(np.float32)
        pn = {"mu": torch.tensor(X.mean(0), device=device), "sd": torch.tensor(X.std(0) + 1e-6, device=device)}
        Xt = (torch.tensor(X, device=device) - pn["mu"]) / pn["sd"]; Yt = torch.tensor(plans.reshape(nt, 2 * H), device=device)
        pol = build_policy(); optp = torch.optim.Adam(pol.parameters(), lr=cfg["pol_lr"]); lossf = nn.MSELoss()
        brng = np.random.default_rng(cfg["seed"] + 56); pol.train(); bs = min(512, nt)
        for _ in range(cfg["pol_steps"]):
            idx = torch.tensor(brng.integers(0, nt, size=bs), device=device)
            optp.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); optp.step()
        pol.eval()

        def plan_fn(states, goals):
            with torch.no_grad():
                x = (torch.tensor(np.concatenate([states, goals], 1), device=device) - pn["mu"]) / pn["sd"]
                return pol(x).cpu().numpy().astype(np.float32).reshape(len(states), H, 2)
        return rollout(plan_fn, replan_every=H)

    def grade(net, want_bc):
        rec = {"fm_err": fm_err(net)}
        if "reactive" in cfg["controllers"]:
            rng = np.random.default_rng(cfg["seed"] + 7000)
            rec["reactive"] = rollout(lambda s, g: mpc_plan(net, s, g, rng), replan_every=1)
        if "ballistic_cem" in cfg["controllers"]:
            rng = np.random.default_rng(cfg["seed"] + 7001)
            rec["ballistic_cem"] = rollout(lambda s, g: mpc_plan(net, s, g, rng), replan_every=H)
        if want_bc and "ballistic_bc" in cfg["controllers"]:
            rec["ballistic_bc"] = ballistic_bc(net)
        return rec

    # ===================================================================== #
    # 1) pretrain FM at d0 (the pre-drift cerebellar model)
    # ===================================================================== #
    S0, U0, S20 = collect_pool(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 10))
    fm = _mlp(cfg["seed"] + 40); opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
    train_steps(fm, opt, norm, S0, U0, S20, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    print(f"[pretrain] d0 FM: fm_err@d1(stale)={fm_err(fm):.4f}", flush=True)

    # ===================================================================== #
    # 2) online re-adaptation at d1: grow a reward-free buffer, keep fine-tuning,
    #    snapshot & grade at each milestone (transitions of d1 experience)
    # ===================================================================== #
    rng_buf = np.random.default_rng(cfg["seed"] + 500)
    bufS = np.zeros((0, 4), np.float32); bufU = np.zeros((0, 2), np.float32); bufS2 = np.zeros((0, 4), np.float32)
    ladder = []
    for k, m in enumerate(cfg["milestones"]):
        need = m - len(bufS)
        if need > 0:
            aS, aU, aS2 = collect_pool(env1, need, rng_buf)
            bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU]); bufS2 = np.concatenate([bufS2, aS2])
        if m > 0:
            train_steps(fm, opt, norm, bufS, bufU, bufS2, cfg["finetune_steps"],
                        np.random.default_rng(cfg["seed"] + 600 + k))
        want_bc = (m == cfg["milestones"][0]) or (m == cfg["milestones"][-1])   # endpoints only
        rec = grade(copy.deepcopy(fm), want_bc)
        rec["transitions"] = int(m)
        ladder.append(rec)
        msg = f"[readapt m={m:5d}] fm_err={rec['fm_err']:.4f}"
        for c in cfg["controllers"]:
            if c in rec:
                msg += f"  {c}={rec[c]:.4f}"
        print(msg, flush=True)

    # ---- reference ceilings: an FM trained FRESH & large on d1 (fully re-adapted) ----
    Sc, Uc, S2c = collect_pool(env1, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 20))
    fm_ceil = _mlp(cfg["seed"] + 41); optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
    train_steps(fm_ceil, optc, norm, Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    ceil = grade(fm_ceil, want_bc=True); ceil["transitions"] = -1
    print(f"[ceiling] fresh d1 FM: fm_err={ceil['fm_err']:.4f} "
          + "  ".join(f"{c}={ceil.get(c, float('nan')):.4f}" for c in cfg["controllers"]), flush=True)

    # ---- recovery value = stale − recovered, per controller (the headline number) ----
    recovery = {}
    for c in cfg["controllers"]:
        stale = ladder[0].get(c)
        recov = ladder[-1].get(c)
        if stale is not None and recov is not None:
            recovery[c] = {"stale": stale, "recovered": recov, "gain": stale - recov}
    print("\n[recovery] control gain from re-adaptation (stale − recovered; larger = more "
          "behaviorally load-bearing):", flush=True)
    for c in cfg["controllers"]:
        if c in recovery:
            print(f"  {c:14s} stale={recovery[c]['stale']:.4f} -> recovered={recovery[c]['recovered']:.4f}  "
                  f"gain={recovery[c]['gain']:+.4f}", flush=True)

    out = {"config": cfg, "ladder": ladder, "ceiling": ceil, "recovery": recovery}
    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "ballistic_readapt", cfg["tag"])
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
    lad = R["ladder"]; ctrls = R["config"]["controllers"]
    tr = np.array([r["transitions"] for r in lad], float)
    xr = np.where(tr == 0, 1.0, tr)            # log-friendly (m=0 stale plotted at x=1)
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # recovery curves: control (left y) + FM-err (right y) vs re-adaptation transitions
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for c in ctrls:
        yv = [r.get(c, np.nan) for r in lad]
        ax.plot(xr, yv, "o-", color=COL.get(c, "#333"), lw=2.4, ms=7,
                label=f"{c} control (gain {R['recovery'].get(c, {}).get('gain', float('nan')):+.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("reward-free re-adaptation transitions at d1  (learning-loop experience →)")
    ax.set_ylabel("control goal-dist (lower=better)")
    ax2 = ax.twinx()
    ax2.plot(xr, [r["fm_err"] for r in lad], "s--", color="#2f9e44", lw=2.0, alpha=0.8, label="FM error (Cut-3 teacher)")
    ax2.set_ylabel("FM prediction error on d1", color="#2f9e44"); ax2.tick_params(axis="y", labelcolor="#2f9e44")
    ax2.spines["top"].set_visible(False)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8.5, loc="best")
    ax.set_title("Online FM re-adaptation restores BALLISTIC control; reactive never needed it\n"
                 "(FM error — the value signal — tracks the ballistic recovery)", fontsize=10)
    figs["fig1_recovery.png"] = _save(fig)

    # bars: recovery gain (stale−recovered) per controller = behavioral value of re-adaptation
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.arange(len(ctrls))
    gains = [R["recovery"].get(c, {}).get("gain", 0.0) for c in ctrls]
    ax.bar(xs, gains, color=[COL.get(c, "#333") for c in ctrls], width=0.6)
    for i, c in enumerate(ctrls):
        ax.text(i, gains[i] + 0.003, f"{gains[i]:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("_", "\n") for c in ctrls])
    ax.set_ylabel("control gain from re-adaptation (stale − recovered)")
    ax.set_title("Behavioral value of FM re-adaptation is ballistic-specific")
    figs["fig2_gain.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def ballistic_readapt(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    controllers: str = "reactive,ballistic_cem,ballistic_bc",
    d0: float = 6.0,                       # pre-drift dynamics (FM pretrained here → stale at d1)
    d1: float = 2.0,                       # post-drift dynamics (re-adapt + operate here; ballistic
                                           # is COMPETENT here with a matched FM, so re-adaptation
                                           # can restore competence — cf. Cut 4b's d_test=2.0)
    milestones: str = "0,200,500,1200,3000,6000",   # reward-free d1 transitions
    finetune_steps: int = 800,             # fine-tune steps per milestone
    # env
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    corridor_r: float = 0.4,
    band_h: float = 0.5,
    # collection / probes
    pool_n: int = 6000,
    probe_n: int = 800,
    v_explore: float = 1.2,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    # BC motor program
    pol_hidden: int = 256,
    pol_layers: int = 3,
    pol_lr: float = 1e-3,
    pol_steps: int = 4000,
    bc_tuples: int = 1500,
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

    ctrl_list = [c for c in controllers.split(",") if c]
    ms = [int(x) for x in milestones.split(",") if x]
    if quick:
        ms = [0, 500, 3000]; ctrl_list = ["reactive", "ballistic_cem", "ballistic_bc"]
        pool_n = 3000; probe_n = 300; fm_steps = 1200; finetune_steps = 400
        pol_steps = 1200; bc_tuples = 600; fm_hidden = 128; fm_layers = 2; pol_hidden = 128; pol_layers = 2
        n_eval = 16; k_shoot = 128
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, controllers=ctrl_list, d0=d0, d1=d1, milestones=ms, finetune_steps=finetune_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, corridor_r=corridor_r, band_h=band_h,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps, bc_tuples=bc_tuples,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_ballistic_readapt.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "ballistic_readapt_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
