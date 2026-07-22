"""Stage 1 — Value-shaping as capacity re-allocation (disc-4, made per-dim and physical).

Program: `ideas/two_timescale_value_loop.md` (discriminator 4 — "does a value/outer
loop CAUSE the forward model to become value-shaped"), staged plan in
`experiments/a2a_forward/reaching/CURIOSITY_CONTROL_README.md` (the MuJoCo
redirection: bare-state disc-4 first, on the pusher, with the puck as the
value-irrelevant distractor).

THE CLAIM (Stage 1, stationary, goal-value):

    A forward model has LIMITED capacity. When a value signal shapes its training,
    it RE-ALLOCATES that capacity toward the value-relevant state directions and
    away from the value-irrelevant ones — a WORSE literal simulator, but a better
    substrate for the thing you actually care about. On the pusher, the value is
    "reach a goal with the PUSHER"; the puck is a causally-coupled but VALUE-
    IRRELEVANT distractor (its own post-contact motion never enters the goal).

    So a value-shaped FM should DROP the puck's own-motion prediction while keeping
    the pusher's — versus a raw-Δs FM that spends capacity on both. Under CAPACITY
    PRESSURE (a small FM), dropping the puck frees capacity, so the shaped FM should
    hit target PUSHER fidelity at a SMALLER model than the unshaped one.

WHY THE PUCK IS THE RIGHT DISTRACTOR (Guardrail 1, CURIOSITY_CONTROL):
    It must be HIGH-ENERGY (frequently, forcefully engaged) — else a capacity-
    limited FM ignores it for free and "the value loop dropped the puck" is
    indistinguishable from "the puck was always negligible" (exactly the cut #2
    inert-puck trap). We collect with a strong seek-the-puck policy so the puck is
    hit constantly and its own motion is a genuinely capacity-hungry, REDUCIBLE
    prediction target (checked: big-FM puck-vel R² must be high).

THE CLEAN SPLIT (sharpening): value-relevant = pusher dims [0,1,4,5] (INCLUDING how
    a contact deflects the pusher — that IS value-relevant); value-irrelevant = the
    PUCK's own post-contact state [2,3,6,7]. The FM still READS puck state (input —
    to know whether it will hit) but a shaped FM need not PREDICT the puck's next
    velocity (output). Capacity re-allocation over prediction TARGETS.

Arms (identical data + FM class — only the per-dim loss weight differs, the
arity_torque discipline):
  * unshaped     : Huber on all 8 Δs dims (λ_puck = 1). The raw-Δs FM.
  * value-shaped : Huber weighted by value-relevance (λ_puck = 0 -> pusher-only).
                   The causal ablation is BUILT IN: λ=1 IS the unshaped arm.
  * λ-frontier   : λ_puck in {1, .5, .25, .1, 0} at a fixed binding capacity.

Readouts:
  1. capacity-efficiency frontier: pusher-vel R² (value-relevant) vs FM width,
     unshaped vs shaped — shaped should hold up at SMALLER widths (capacity freed).
  2. per-dim re-allocation: shaped drops puck-vel R² (esp. on puck-ENGAGED
     transitions) while keeping pusher-vel R².
  3. preconditions: puck engagement frac; big-FM puck-vel reducibility ceiling.

Stage 1b (CEM planning frontier + planner-co-trained "goal-value discovers it" arm)
is added once these preconditions check out. Stage 2 (the OUTER-LOOP drive causes
the shaping, under non-stationarity) is a separate script.

Run:
    cd experiments/
    modal run mujoco_control/value_shaping.py::value_shaping --quick   # smoke
    modal run --detach mujoco_control/value_shaping.py::run_value_shaping  # (full; see entrypoint)
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (see pusher_env.STATE_LABELS)
PUSHER_POS = [0, 1]
PUCK_POS = [2, 3]
PUSHER_VEL = [4, 5]
PUCK_VEL = [6, 7]
PUSHER = [0, 1, 4, 5]   # value-relevant (goal is over the pusher)
PUCK = [2, 3, 6, 7]     # value-irrelevant (the distractor's own motion)


def _r2(y_true, y_pred, dims=None):
    import numpy as np

    yt = y_true if dims is None else y_true[:, dims]
    yp = y_pred if dims is None else y_pred[:, dims]
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


@app.function(cpu=8.0, memory=16384, timeout=5400, volumes={DATA_DIR: volume})
def run_value_shaping(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import collect_transitions, STATE_LABELS

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dgp = cfg["dgp"]
    print(f"[setup] device={device}  dgp={dgp}", flush=True)

    # ---- 1. collect with a STRONG seek-the-puck policy (high-energy puck) ---- #
    print(f"[collect] {cfg['n_episodes']} eps x {cfg['ep_len']}  seek_gain={cfg['seek_gain']}",
          flush=True)
    data = collect_transitions(
        dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
        frame_skip=cfg["frame_skip"], seed=cfg["seed"],
        sigma=cfg["sigma"], theta=cfg["theta"], seek_gain=cfg["seek_gain"],
    )
    S, U, S2, ep = data["S"], data["U"], data["S2"], data["ep"]
    any_contact = data["any_contact"]
    puck_contact = data["puck_contact"]
    N = len(S)
    Y = (S2 - S).astype(np.float32)

    # puck "engaged" = the puck actually moved this step (its own dynamics are live).
    # This is the subset where predicting the puck COSTS capacity — the honest
    # denominator for the value-irrelevant readout (predicting a still puck is free).
    puck_moving = np.abs(Y[:, PUCK_VEL]).max(1) > cfg["puck_move_thresh"]
    print(f"[collect] N={N}  any_contact={any_contact.mean():.3f}  "
          f"puck_contact={puck_contact.mean():.3f}  puck_moving={puck_moving.mean():.3f}",
          flush=True)

    # ---- 2. split (by episode) + normalize --------------------------------- #
    n_ep = int(ep.max()) + 1
    test_eps = set(range(n_ep - max(1, int(round(0.2 * n_ep))), n_ep))
    te = np.array([e in test_eps for e in ep])
    tr = ~te

    X = np.concatenate([S, U], axis=1).astype(np.float32)   # arity-2 input f(s,u)

    def _norm(A):
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        return (A - mu) / sd, mu, sd

    Xn, mx, sx = _norm(X)
    Yn, my, sy = _norm(Y)
    Yte_raw = Y[te]
    te_moving = puck_moving[te]
    te_contact = any_contact[te]
    te_ff = ~te_contact                      # free-flight test transitions (cut #2 discipline)
    te_slide = te_ff & te_moving             # puck sliding, NO contact -> reducible puck dyn
    te_impact = te_contact & te_moving       # puck moving WITH contact -> the stiff impulse

    def _r2_sub(yt, yp, mask, dims):
        if int(mask.sum()) < 20:
            return float("nan")
        return _r2(yt[mask], yp[mask], dims)

    # per-dim loss weight for a given puck weight λ (1 on pusher dims, λ on puck dims)
    def _weight_vec(puck_weight):
        w = np.ones(Y.shape[1], np.float32)
        w[PUCK] = puck_weight
        return torch.tensor(w, device=device)

    Xtr = torch.tensor(Xn[tr], device=device)
    Ytr = torch.tensor(Yn[tr], device=device)
    Xte = torch.tensor(Xn[te], device=device)
    my_t = torch.tensor(my, device=device)
    sy_t = torch.tensor(sy, device=device)

    def _weighted_huber(pred, target, w, delta=1.0):
        """Per-dim Huber, weighted by w (value-relevance), then mean over N and dims.
        Weighting is on the NORMALIZED Δs so the pusher/puck scales are comparable."""
        err = pred - target
        absr = err.abs()
        quad = torch.clamp(absr, max=delta)
        lin = absr - quad
        per_elem = 0.5 * quad ** 2 + delta * lin        # (N, D)
        return (per_elem * w).mean()

    def _train(hidden, layers, epochs, puck_weight):
        w = _weight_vec(puck_weight)
        din = Xtr.shape[1]
        lyr = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
        lyr += [nn.Linear(hidden, Y.shape[1])]
        net = nn.Sequential(*lyr).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        ntr = Xtr.shape[0]
        bs = min(cfg["batch"], ntr)
        net.train()
        for _e in range(epochs):
            perm = torch.randperm(ntr, device=device)
            for i in range(0, ntr, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                _weighted_huber(net(Xtr[idx]), Ytr[idx], w).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred_raw = (net(Xte) * sy_t + my_t).cpu().numpy()
        nparam = sum(p.numel() for p in net.parameters())
        return net, pred_raw, nparam

    def _perdim_report(pred_raw):
        """Per-dim R², broken out by regime. Fidelity is evaluated on FREE-FLIGHT
        (cut #2 discipline — contact's stiff Δv caps R² for everything); the puck's
        reducibility is split into its post-contact SLIDE (free-flight, reducible)
        vs the contact IMPULSE (stiff, cut #1's partly-irreducible component)."""
        rec = {
            # all-transitions (contaminated by contact — kept for reference only)
            "r2_all": _r2(Yte_raw, pred_raw),
            "r2_pusher_vel": _r2(Yte_raw, pred_raw, PUSHER_VEL),
            "r2_puck_vel": _r2(Yte_raw, pred_raw, PUCK_VEL),
            # free-flight (the clean fidelity regime)
            "r2_pusher_vel_ff": _r2_sub(Yte_raw, pred_raw, te_ff, PUSHER_VEL),
            "r2_puck_vel_ff": _r2_sub(Yte_raw, pred_raw, te_ff, PUCK_VEL),
            # puck reducibility split: slide (reducible?) vs impact (irreducible?)
            "r2_puck_vel_slide": _r2_sub(Yte_raw, pred_raw, te_slide, PUCK_VEL),
            "r2_puck_vel_impact": _r2_sub(Yte_raw, pred_raw, te_impact, PUCK_VEL),
        }
        return rec

    # ---- 3. capacity sweep: unshaped (λ=1) vs value-shaped (λ=0) ------------ #
    caps = cfg["caps"]
    sweep = {"caps": caps, "unshaped": [], "shaped": []}
    print(f"[sweep] caps={caps} layers={cfg['layers']} epochs={cfg['epochs']}", flush=True)
    for lam, key in [(1.0, "unshaped"), (0.0, "shaped")]:
        for h in caps:
            _, pred, nparam = _train(h, cfg["layers"], cfg["epochs"], lam)
            rec = {"hidden": h, "nparam": nparam, **_perdim_report(pred)}
            sweep[key].append(rec)
            print(f"  {key:8s} h={h:4d} ({nparam:6d}p)  "
                  f"pusherV_ff={rec['r2_pusher_vel_ff']:.3f}  "
                  f"puckV_slide={rec['r2_puck_vel_slide']:.3f}  "
                  f"puckV_impact={rec['r2_puck_vel_impact']:.3f}", flush=True)

    # precondition (2): is the puck's own motion REDUCIBLE (worth dropping)? Split into
    # the post-contact slide (should be reducible) vs the contact impulse (cut #1 says
    # partly irreducible). Take the biggest UNSHAPED FM as the reducibility ceiling.
    import math

    def _ceil(key):
        vals = [r[key] for r in sweep["unshaped"] if not math.isnan(r[key])]
        return max(vals) if vals else float("nan")

    puck_slide_ceiling = _ceil("r2_puck_vel_slide")
    puck_impact_ceiling = _ceil("r2_puck_vel_impact")
    pusher_ceiling = _ceil("r2_pusher_vel_ff")
    print(f"[precond] pusher-vel free-flight ceiling = {pusher_ceiling:.3f}  "
          f"(sanity: want ~0.99 like cut #2)", flush=True)
    print(f"[precond] puck-vel reducibility: slide ceiling={puck_slide_ceiling:.3f}  "
          f"impact ceiling={puck_impact_ceiling:.3f}  "
          f"(want the value-irrelevant target REDUCIBLE -> high)", flush=True)

    # ---- 4. λ-frontier at a fixed binding capacity ------------------------- #
    lam_cap = cfg["frontier_hidden"]
    frontier = {"puck_weight": cfg["lambdas"], "hidden": lam_cap, "rows": []}
    print(f"[frontier] λ sweep at hidden={lam_cap}", flush=True)
    for lam in cfg["lambdas"]:
        _, pred, nparam = _train(lam_cap, cfg["layers"], cfg["epochs"], lam)
        rec = {"puck_weight": lam, "nparam": nparam, **_perdim_report(pred)}
        frontier["rows"].append(rec)
        print(f"  λ_puck={lam:.2f}  pusherV_ff={rec['r2_pusher_vel_ff']:.3f}  "
              f"puckV_slide={rec['r2_puck_vel_slide']:.3f}  "
              f"puckV_impact={rec['r2_puck_vel_impact']:.3f}", flush=True)

    results = {
        "config": cfg, "n_transitions": N, "state_labels": STATE_LABELS,
        "engagement": {"any_contact": float(any_contact.mean()),
                       "puck_contact": float(puck_contact.mean()),
                       "puck_moving": float(puck_moving.mean()),
                       "n_slide": int(te_slide.sum()), "n_impact": int(te_impact.sum())},
        "sweep": sweep,
        "puck_slide_ceiling": puck_slide_ceiling,
        "puck_impact_ceiling": puck_impact_ceiling,
        "pusher_ceiling": pusher_ceiling,
        "frontier": frontier,
    }

    print("\n===== VALUE-SHAPING (Stage 1) SUMMARY =====", flush=True)
    print(f"engagement: puck_contact={results['engagement']['puck_contact']:.3f}  "
          f"puck_moving={results['engagement']['puck_moving']:.3f}  "
          f"(slide n={int(te_slide.sum())}, impact n={int(te_impact.sum())})", flush=True)
    print(f"pusher free-flight ceiling = {pusher_ceiling:.3f}  |  "
          f"puck slide ceiling = {puck_slide_ceiling:.3f}  "
          f"impact ceiling = {puck_impact_ceiling:.3f}", flush=True)
    # capacity-efficiency headline at the smallest capacity
    u0 = sweep["unshaped"][0]
    s0 = sweep["shaped"][0]
    print(f"smallest FM (h={caps[0]}): pusherV_ff  unshaped={u0['r2_pusher_vel_ff']:.3f}  "
          f"shaped={s0['r2_pusher_vel_ff']:.3f}   |   puckV_slide  "
          f"unshaped={u0['r2_puck_vel_slide']:.3f}  shaped={s0['r2_puck_vel_slide']:.3f}",
          flush=True)

    figures = _make_figures(results, STATE_LABELS)

    outdir = os.path.join(DATA_DIR, "value_shaping", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_figures(R, labels):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_UNS, C_SHP = "#3d6fd1", "#d1603d"
    figs = {}

    def _save(fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    sweep = R["sweep"]
    caps = sweep["caps"]

    # (1) capacity-efficiency frontier: pusher-vel (value-relevant) + puck-vel slide (irrelevant)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, metric, title in [
        (axes[0], "r2_pusher_vel_ff", "R² pusher velocity, free-flight (VALUE-RELEVANT)"),
        (axes[1], "r2_puck_vel_slide", "R² puck velocity, slide (VALUE-IRRELEVANT)"),
    ]:
        ax.plot(caps, [r[metric] for r in sweep["unshaped"]], "o-", color=C_UNS,
                label="unshaped  (λ_puck=1)")
        ax.plot(caps, [r[metric] for r in sweep["shaped"]], "o-", color=C_SHP,
                label="value-shaped  (λ_puck=0)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps); ax.set_xticklabels(caps)
        ax.set_xlabel("FM hidden width (capacity)")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Value-shaping re-allocates capacity: keep the pusher, drop the puck",
                 fontsize=11)
    figs["fig1_capacity_frontier.png"] = _save(fig)

    # (2) λ-frontier: pusher-vel vs puck-vel as the puck loss-weight is dialed to 0
    fr = R["frontier"]
    lams = [row["puck_weight"] for row in fr["rows"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(lams, [row["r2_pusher_vel_ff"] for row in fr["rows"]], "o-", color=C_UNS,
            label="pusher-vel R² free-flight (value-relevant)")
    ax.plot(lams, [row["r2_puck_vel_slide"] for row in fr["rows"]], "o-", color=C_SHP,
            label="puck-vel R² slide (value-irrelevant)")
    ax.set_xlabel("λ_puck  (puck loss weight: 1 = unshaped → 0 = pusher-only)")
    ax.set_ylabel("R²")
    ax.invert_xaxis()
    ax.set_title(f"Capacity re-allocation frontier (hidden={fr['hidden']})")
    ax.legend(fontsize=9)
    figs["fig2_lambda_frontier.png"] = _save(fig)

    # (3) per-dim R² bars at the frontier capacity: unshaped vs shaped
    uns = next(r for r in sweep["unshaped"] if r["hidden"] == fr["hidden"]) \
        if any(r["hidden"] == fr["hidden"] for r in sweep["unshaped"]) else sweep["unshaped"][0]
    unshaped_row = fr["rows"][0]     # λ=1
    shaped_row = fr["rows"][-1]      # λ=0
    groups = ["r2_pusher_vel_ff", "r2_puck_vel_slide", "r2_puck_vel_impact", "r2_all"]
    glabels = ["pusherV (ff)", "puckV (slide)", "puckV (impact)", "all"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(groups)); w = 0.4
    ax.bar(x - w / 2, [unshaped_row[g] for g in groups], w, color=C_UNS, label="unshaped")
    ax.bar(x + w / 2, [shaped_row[g] for g in groups], w, color=C_SHP, label="value-shaped")
    ax.set_xticks(x); ax.set_xticklabels(glabels, rotation=20, ha="right")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("R²")
    ax.set_title(f"Per-group fidelity at hidden={fr['hidden']}: the puck is dropped, pusher kept")
    ax.legend(fontsize=9)
    figs["fig3_perdim.png"] = _save(fig)

    return figs


@app.local_entrypoint()
def value_shaping(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_episodes: int = 350,
    ep_len: int = 200,
    frame_skip: int = 3,          # fine control step -> the smooth field stays reducible
    sigma: float = 0.7,
    theta: float = 0.15,
    seek_gain: float = 0.9,       # STRONG seek -> the puck is hit constantly (Guardrail 1)
    layers: int = 2,
    epochs: int = 50,
    batch: int = 512,
    lr: float = 1e-3,
    frontier_hidden: int = 16,     # a small/binding capacity for the λ-frontier
    puck_move_thresh: float = 0.02,
    arena_half: float = 0.9,       # roomy -> the field-advected puck stays in free-flight
    puck_mass: float = 0.5,
    pusher_r: float = 0.12,
    puck_r: float = 0.12,
    # nonlinear multi-mode force field (capacity-hungry, reducible). On the PUCK it is
    # the value-irrelevant distractor; on the PUSHER (distinct phase) it makes the
    # value-relevant dynamics capacity-hungry too, so freed capacity buys pusher fidelity.
    field_amp: float = 1.3,
    field_pusher_amp: float = 1.3,
    field_central: float = 0.5,
):
    import os

    caps = [8, 16, 32, 64, 128, 256]
    lambdas = [1.0, 0.5, 0.25, 0.1, 0.0]
    if quick:
        n_episodes, ep_len, epochs = 140, 150, 45
        caps = [8, 16, 32, 64, 128, 256]
        lambdas = [1.0, 0.25, 0.0]
        tag = tag or "smoke"
    tag = tag or "default"

    puck_field = (dict(amp=field_amp, pusher_amp=field_pusher_amp, central=field_central)
                  if field_amp > 0 else None)
    cfg = dict(
        tag=tag, seed=seed, n_episodes=n_episodes, ep_len=ep_len,
        frame_skip=frame_skip, sigma=sigma, theta=theta, seek_gain=seek_gain,
        layers=layers, epochs=epochs, batch=batch, lr=lr,
        caps=caps, lambdas=lambdas, frontier_hidden=frontier_hidden,
        puck_move_thresh=puck_move_thresh,
        dgp=dict(arena_half=arena_half, puck_mass=puck_mass,
                 pusher_r=pusher_r, puck_r=puck_r, puck_field=puck_field),
    )
    out = run_value_shaping.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", "valshape_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
