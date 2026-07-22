"""Cut #2 — Arity on torque: a command-blind forward model floors on actuated dynamics.

Program: `ideas/physical_control_substrate.md`, cut #2. The claim (the a2a/RHM
arity thread — ACTIVE_VISION, REACHING_INTERNAL, RHM length-gen — now on a REAL
actuator):

    A forward model of a *controlled* system must take the command as a second
    input. An arity-2 f(s,u) can predict command-driven dynamics; an arity-1,
    command-blind f(s) can only predict the command-AVERAGED next state
    E_u[Δs | s], and NO CAPACITY buys the missing slot. Adding the command turns
    an observational map (Pearl rung 1) into an interventional one (rung 2).

The one confound to kill (the RHM "received-wisdom"/generator confound): if the
scripted command u is a deterministic function of the state s, arity-1 recovers
u from s and there is no gap. We use **i.i.d. commands** (theta=1, no seek), so
u_t ⊥ s_t exactly — arity-1 cannot access u_t at all and its best prediction of
the u-driven part of Δs is E[u]=0. Airtight, by construction.

Three readouts:
  1. capacity sweep: R² vs hidden size for arity-1 vs arity-2 — arity BEATS
     resolution (the smallest arity-2 beats the largest arity-1).
  2. per-dim: the arity gap is LOCALIZED on the directly-actuated pusher-velocity
     dims (the puck is unactuated; its dynamics depend on u only through contact,
     which is mostly encoded in s) — command influence is physically legible.
  3. command sensitivity (interventional, via the perfect simulator): reset to a
     held-out state, vary u, measure the true Δs spread. arity-2 reproduces it;
     arity-1 captures ZERO by construction (its prediction is constant in u).

Run:
    cd experiments/
    modal run mjc/arity_torque/arity_torque.py::arity_torque --quick
    modal run --detach mjc/arity_torque/arity_torque.py::run_arity  # (full; see entrypoint)
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (see pusher_env.STATE_LABELS)
PUSHER_VEL = [4, 5]   # directly actuated -> where the command lives
PUCK_VEL = [6, 7]
POS = [0, 1, 2, 3]


def _r2(y_true, y_pred, dims=None):
    import numpy as np

    yt = y_true if dims is None else y_true[:, dims]
    yp = y_pred if dims is None else y_pred[:, dims]
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


@app.function(cpu=8.0, memory=16384, timeout=5400, volumes={DATA_DIR: volume})
def run_arity(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import collect_transitions, PusherEnv, STATE_LABELS

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dgp = cfg["dgp"]

    # ---- 1. collect with i.i.d. commands (u ⊥ s) -------------------------- #
    print(f"[collect] {cfg['n_episodes']} eps x {cfg['ep_len']} (i.i.d. commands) ...",
          flush=True)
    data = collect_transitions(
        dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
        frame_skip=cfg["frame_skip"], seed=cfg["seed"],
        sigma=cfg["sigma"], theta=1.0, seek_gain=0.0,   # theta=1 => i.i.d.
    )
    S, U, S2, ep = data["S"], data["U"], data["S2"], data["ep"]
    contact = data["any_contact"]
    N = len(S)
    print(f"[collect] N={N}, contact_frac={contact.mean():.3f}", flush=True)

    # sanity: command should be ~uncorrelated with state (received-wisdom check)
    cmd_state_corr = float(np.max(np.abs([
        np.corrcoef(U[:, a], S[:, b])[0, 1]
        for a in range(2) for b in range(S.shape[1])])))
    print(f"[check] max |corr(u, s)| = {cmd_state_corr:.3f} (want ~0)", flush=True)

    # ---- 2. split + normalize -------------------------------------------- #
    n_ep = int(ep.max()) + 1
    test_eps = set(range(n_ep - max(1, int(round(0.2 * n_ep))), n_ep))
    te = np.array([e in test_eps for e in ep])
    tr = ~te

    Xa2 = np.concatenate([S, U], axis=1).astype(np.float32)  # arity-2 input
    Xa1 = S.astype(np.float32)                                # arity-1 input
    Y = (S2 - S).astype(np.float32)

    def _norm(A):
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        return (A - mu) / sd, mu, sd

    Xa2n, mx2, sx2 = _norm(Xa2)
    Xa1n, _, _ = _norm(Xa1)
    Yn, my, sy = _norm(Y)
    Yte_raw = Y[te]
    # Arity is evaluated on FREE-FLIGHT test transitions (the actuated dynamics
    # the FM can fit). Contact is cut #1's regime: wall-bounce velocity reversals
    # are unpredictable from anything and would cap BOTH arities' R² ceiling,
    # hiding that arity-2 fully captures the smooth command-driven dynamics.
    ff = ~contact[te]
    print(f"[eval] free-flight test transitions: {int(ff.sum())}/{len(ff)}", flush=True)

    def _train(use_cmd, hidden, layers, epochs):
        Xn = Xa2n if use_cmd else Xa1n
        din = Xn.shape[1]
        Xtr = torch.tensor(Xn[tr], device=device)
        Ytr = torch.tensor(Yn[tr], device=device)
        Xte = torch.tensor(Xn[te], device=device)
        lyr = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        lyr += [nn.Linear(hidden, Y.shape[1])]
        net = nn.Sequential(*lyr).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        lossf = nn.HuberLoss(delta=1.0)
        ntr = Xtr.shape[0]
        bs = min(cfg["batch"], ntr)
        for _e in range(epochs):
            perm = torch.randperm(ntr, device=device)
            for i in range(0, ntr, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(net(Xtr[idx]), Ytr[idx]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred_n = net(Xte).cpu().numpy()
        pred_raw = pred_n * sy + my
        nparam = sum(p.numel() for p in net.parameters())
        return net, pred_raw, nparam

    # ---- 3. capacity sweep: arity-1 vs arity-2 --------------------------- #
    caps = cfg["caps"]
    sweep = {"caps": caps, "a1": [], "a2": []}
    best = {}
    print(f"[sweep] caps={caps}, layers={cfg['layers']}, epochs={cfg['epochs']}",
          flush=True)
    for use_cmd, key in [(False, "a1"), (True, "a2")]:
        for h in caps:
            net, pred, nparam = _train(use_cmd, h, cfg["layers"], cfg["epochs"])
            yt, yp = Yte_raw[ff], pred[ff]     # free-flight test subset
            rec = {
                "hidden": h, "nparam": nparam,
                "r2_all": _r2(yt, yp),
                "r2_pusher_vel": _r2(yt, yp, PUSHER_VEL),
                "r2_puck_vel": _r2(yt, yp, PUCK_VEL),
                "r2_pos": _r2(yt, yp, POS),
            }
            sweep[key].append(rec)
            print(f"  {key} h={h:4d} ({nparam:6d}p)  R²_all={rec['r2_all']:.3f}  "
                  f"R²_pusherV={rec['r2_pusher_vel']:.3f}  "
                  f"R²_puckV={rec['r2_puck_vel']:.3f}", flush=True)
            if h == caps[-1]:
                best[key] = (net, pred)

    # arity-beats-resolution headline
    a1_max = max(r["r2_pusher_vel"] for r in sweep["a1"])
    a2_min = min(r["r2_pusher_vel"] for r in sweep["a2"])
    beats = a2_min > a1_max

    # ---- 4. per-dim arity gap (largest-capacity models) ------------------ #
    _, pred_a1 = best["a1"]
    _, pred_a2 = best["a2"]
    perdim_r2_a1 = [_r2(Yte_raw[ff], pred_a1[ff], [d]) for d in range(Y.shape[1])]
    perdim_r2_a2 = [_r2(Yte_raw[ff], pred_a2[ff], [d]) for d in range(Y.shape[1])]

    # ---- 5. command sensitivity (interventional, perfect simulator) ------ #
    print("[cmd-sens] querying the simulator with counterfactual u ...", flush=True)
    net_a2 = best["a2"][0]
    rng = np.random.default_rng(cfg["seed"] + 1)
    te_idx = np.where(te)[0]
    sel = rng.choice(te_idx, size=min(cfg["n_cmd_states"], len(te_idx)), replace=False)
    env = PusherEnv(dgp)
    n_u = cfg["n_cmd_samples"]
    true_spread, a2_spread = [], []
    for j in sel:
        s0 = S[j]
        d_true, d_a2 = [], []
        us = np.clip(cfg["sigma"] * rng.normal(size=(n_u, 2)), -1, 1).astype(np.float32)
        for u in us:
            env.set_state(s0[:4], s0[4:])
            s2, _ = env.step(u, cfg["frame_skip"])
            d_true.append(s2 - s0)
        # arity-2 predictions for the SAME counterfactual commands at fixed s0
        xin = np.concatenate([np.tile(s0, (n_u, 1)), us], axis=1).astype(np.float32)
        xn = (xin - mx2) / sx2
        with torch.no_grad():
            p = net_a2(torch.tensor(xn, device=device)).cpu().numpy() * sy + my
        true_spread.append(np.std(np.asarray(d_true), axis=0))
        a2_spread.append(np.std(p, axis=0))
    true_spread = np.mean(true_spread, axis=0)   # per-dim command-driven std (truth)
    a2_spread = np.mean(a2_spread, axis=0)        # captured by arity-2
    # arity-1 spread is exactly 0 (prediction is constant in u)

    results = {
        "config": cfg, "n_transitions": N,
        "contact_frac": float(contact.mean()),
        "cmd_state_max_corr": cmd_state_corr,
        "state_labels": STATE_LABELS,
        "sweep": sweep,
        "arity_beats_resolution": bool(beats),
        "a1_max_r2_pusher_vel": a1_max,
        "a2_min_r2_pusher_vel": a2_min,
        "perdim_r2_a1": perdim_r2_a1,
        "perdim_r2_a2": perdim_r2_a2,
        "cmd_sens_true": true_spread.tolist(),
        "cmd_sens_a2": a2_spread.tolist(),
        "cmd_sens_pusher_vel_true": float(np.mean(true_spread[PUSHER_VEL])),
        "cmd_sens_pusher_vel_a2": float(np.mean(a2_spread[PUSHER_VEL])),
    }

    print("\n===== ARITY-ON-TORQUE SUMMARY =====", flush=True)
    print(f"max |corr(u,s)| = {cmd_state_corr:.3f} (i.i.d. command check)", flush=True)
    print(f"R²(pusher-vel): arity-1 max={a1_max:.3f}  arity-2 min={a2_min:.3f}  "
          f"-> arity beats resolution: {beats}", flush=True)
    print(f"cmd sensitivity (pusher-vel std): true={results['cmd_sens_pusher_vel_true']:.4f}  "
          f"arity-2={results['cmd_sens_pusher_vel_a2']:.4f}  arity-1=0.0 (by construction)",
          flush=True)

    figures = _make_figures(sweep, perdim_r2_a1, perdim_r2_a2,
                            true_spread, a2_spread, STATE_LABELS, results)

    outdir = os.path.join(DATA_DIR, "arity_torque", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_figures(sweep, perdim_a1, perdim_a2, cmd_true, cmd_a2, labels, results):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_A1, C_A2, C_TRUE = "#3d6fd1", "#d1603d", "#444"
    figs = {}

    def _save(fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    caps = sweep["caps"]

    # (1) capacity sweep — arity beats resolution
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in [
        (axes[0], "r2_all", "R² all state dims (free flight)"),
        (axes[1], "r2_pusher_vel", "R² pusher velocity (free flight — where u lives)"),
    ]:
        a1 = [r[metric] for r in sweep["a1"]]
        a2 = [r[metric] for r in sweep["a2"]]
        ax.plot(caps, a1, "o-", color=C_A1, label="arity-1  f(s)")
        ax.plot(caps, a2, "o-", color=C_A2, label="arity-2  f(s,u)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels(caps)
        ax.set_xlabel("hidden width (capacity)")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.legend(loc="center right")
    a1_max = results["a1_max_r2_pusher_vel"]
    axes[1].axhline(a1_max, color=C_A1, ls=":", lw=1)
    axes[1].text(caps[0], a1_max, " arity-1 ceiling", color=C_A1, va="bottom", fontsize=8)
    fig.suptitle(
        "Arity beats resolution: the smallest command-aware model beats the "
        f"largest command-blind one  (arity_beats={results['arity_beats_resolution']})",
        fontsize=11)
    figs["fig1_capacity_sweep.png"] = _save(fig)

    # (2) per-dim arity gap
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, perdim_a1, w, color=C_A1, label="arity-1  f(s)")
    ax.bar(x + w / 2, perdim_a2, w, color=C_A2, label="arity-2  f(s,u)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("per-dim R²")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("The arity gap is localized on the directly-actuated pusher velocities")
    ax.legend()
    figs["fig2_perdim_arity.png"] = _save(fig)

    # (3) command sensitivity (interventional): true vs captured
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - w / 2, cmd_true, w, color=C_TRUE, label="TRUE (perfect simulator)")
    ax.bar(x + w / 2, cmd_a2, w, color=C_A2, label="captured by arity-2")
    ax.plot(x, np.zeros_like(x), "_", color=C_A1, ms=14, mew=2,
            label="captured by arity-1 (=0, by construction)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("command-driven std of Δs")
    ax.set_title("Interventional command sensitivity: vary u at a fixed state")
    ax.legend(fontsize=8)
    figs["fig3_command_sensitivity.png"] = _save(fig)

    return figs


@app.local_entrypoint()
def arity_torque(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_episodes: int = 400,
    ep_len: int = 250,
    frame_skip: int = 5,
    sigma: float = 0.7,
    layers: int = 2,
    epochs: int = 40,
    batch: int = 512,
    lr: float = 1e-3,
    n_cmd_states: int = 400,
    n_cmd_samples: int = 24,
):
    import os

    caps = [8, 16, 32, 64, 128, 256]
    if quick:
        n_episodes, ep_len, epochs = 60, 150, 20
        caps = [8, 32, 128]
        n_cmd_states = 120
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, n_episodes=n_episodes, ep_len=ep_len,
        frame_skip=frame_skip, sigma=sigma, layers=layers, epochs=epochs,
        batch=batch, lr=lr, caps=caps,
        n_cmd_states=n_cmd_states, n_cmd_samples=n_cmd_samples,
        # smaller/contact-rich world; i.i.d. commands keep u ⊥ s
        dgp=dict(arena_half=0.7, pusher_r=0.13, puck_r=0.13),
    )
    out = run_arity.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", "arity_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
